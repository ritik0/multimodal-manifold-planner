from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


def _clip_to_bounds(x: np.ndarray, ambient_bounds):
    y = np.asarray(x, dtype=float).copy()
    for i, (lo, hi) in enumerate(ambient_bounds):
        y[i] = np.clip(y[i], lo, hi)
    return y


def _numerical_jacobian(h_func: Callable[[np.ndarray], np.ndarray], x: np.ndarray, eps: float = 1e-6):
    x = np.asarray(x, dtype=float)
    r0 = np.asarray(h_func(x), dtype=float).reshape(-1)
    m = r0.size
    n = x.size
    J = np.zeros((m, n), dtype=float)

    for j in range(n):
        xp = x.copy()
        xm = x.copy()
        xp[j] += eps
        xm[j] -= eps
        rp = np.asarray(h_func(xp), dtype=float).reshape(-1)
        rm = np.asarray(h_func(xm), dtype=float).reshape(-1)
        J[:, j] = (rp - rm) / (2.0 * eps)

    return J


@dataclass
class TaskGeometry:
    L: float = 0.45
    W: float = 0.40
    G: float = 0.30
    table_height: float = 0.70
    z_max: float = 1.60

    object_half_length: float = 0.10
    object_radius: float = 0.04

    carry_clearance: float = 0.03
    support_side_clearance: float = 0.01

    projection_tol: float = 1e-6
    projection_damping: float = 1e-3
    projection_max_iters: int = 50


class Mode:
    def __init__(
        self,
        name: str,
        family: str,
        projector,
        ambient_bounds,
        *,
        is_valid=None,
        cost_weight=None,
        params=None,
        meta=None,
    ):
        self.name = name
        self.family = family
        self.project = projector
        self.ambient_bounds = ambient_bounds
        self.is_valid = is_valid if is_valid is not None else (lambda x: True)
        self.cost_weight = cost_weight if cost_weight is not None else {}
        self.params = params if params is not None else {}
        self.meta = meta if meta is not None else {}

    def sample_ambient(self) -> np.ndarray:
        return np.array([np.random.uniform(lo, hi) for lo, hi in self.ambient_bounds], dtype=float)


class ImplicitMode(Mode):
    def __init__(
        self,
        name: str,
        family: str,
        h_func,
        J_func,
        ambient_bounds,
        *,
        is_valid=None,
        cost_weight=None,
        params=None,
        meta=None,
        max_iters: int = 50,
        tol: float = 1e-6,
        damping: float = 1e-3,
        step_scale: float = 1.0,
        clip_each_iter: bool = True,
        backtracking: bool = True,
        min_alpha: float = 1e-3,
    ):
        super().__init__(
            name=name,
            family=family,
            projector=lambda x: x,
            ambient_bounds=ambient_bounds,
            is_valid=is_valid,
            cost_weight=cost_weight,
            params=params,
            meta=meta,
        )
        self.h_func = h_func
        self.J_func = J_func if J_func is not None else (lambda x: _numerical_jacobian(h_func, x))
        self.max_iters = int(max_iters)
        self.tol = float(tol)
        self.damping = float(damping)
        self.step_scale = float(step_scale)
        self.clip_each_iter = bool(clip_each_iter)
        self.backtracking = bool(backtracking)
        self.min_alpha = float(min_alpha)

        self.project = self._project_impl

    def _project_impl(self, x0: np.ndarray) -> np.ndarray:
        x = np.asarray(x0, dtype=float).copy()
        if self.clip_each_iter:
            x = _clip_to_bounds(x, self.ambient_bounds)

        def residual_norm(xx):
            r = np.asarray(self.h_func(xx), dtype=float).reshape(-1)
            return float(np.linalg.norm(r))

        for _ in range(self.max_iters):
            r = np.asarray(self.h_func(x), dtype=float).reshape(-1)
            if r.size == 0:
                return _clip_to_bounds(x, self.ambient_bounds)

            nr = float(np.linalg.norm(r))
            if nr < self.tol:
                return _clip_to_bounds(x, self.ambient_bounds)

            J = np.asarray(self.J_func(x), dtype=float)
            m = J.shape[0]

            A = J @ J.T + (self.damping ** 2) * np.eye(m)
            try:
                y = np.linalg.solve(A, r)
            except np.linalg.LinAlgError:
                break

            dx = -(J.T @ y)

            alpha = self.step_scale
            x_new = x + alpha * dx
            if self.clip_each_iter:
                x_new = _clip_to_bounds(x_new, self.ambient_bounds)

            if self.backtracking:
                base = nr
                nn = residual_norm(x_new)
                while nn > base and alpha > self.min_alpha:
                    alpha *= 0.5
                    x_new = x + alpha * dx
                    if self.clip_each_iter:
                        x_new = _clip_to_bounds(x_new, self.ambient_bounds)
                    nn = residual_norm(x_new)

            x = x_new

            if not np.all(np.isfinite(x)):
                break

        return _clip_to_bounds(x, self.ambient_bounds)


def _orientation_semantics_for_family(family: str) -> dict:
    if family == "SupportContact":
        return {
            "keep_payload_horizontal": True,
            "max_tilt_deg": 10.0,
            "yaw_window_deg": 45.0,
            "rot_tol_deg": 18.0,
            "orientation_weight": 0.18,
        }
    if family == "SupportTransition":
        return {
            "keep_payload_horizontal": True,
            "max_tilt_deg": 12.0,
            "yaw_window_deg": 55.0,
            "rot_tol_deg": 22.0,
            "orientation_weight": 0.15,
        }
    if family == "FreeTransfer":
        return {
            "keep_payload_horizontal": True,
            "max_tilt_deg": 15.0,
            "yaw_window_deg": 70.0,
            "rot_tol_deg": 28.0,
            "orientation_weight": 0.12,
        }
    return {
        "keep_payload_horizontal": True,
        "max_tilt_deg": 15.0,
        "yaw_window_deg": 70.0,
        "rot_tol_deg": 28.0,
        "orientation_weight": 0.12,
    }


def make_two_tables_problem_3d(
    *,
    L=0.45,
    W=0.40,
    G=0.30,
    table_height=0.70,
    z_max=1.60,
    object_half_length=0.10,
    object_radius=0.04,
    carry_clearance=0.03,
    support_side_clearance=0.01,
    projection_tol=1e-6,
    projection_damping=1e-3,
    projection_max_iters=50,
    transition_width=None,
    lift_epsilon=None,
    feasible_edge_margin=None,
    **kwargs,
):
    geom = TaskGeometry(
        L=float(L),
        W=float(W),
        G=float(G),
        table_height=float(table_height),
        z_max=float(z_max),
        object_half_length=float(object_half_length),
        object_radius=float(object_radius),
        carry_clearance=float(carry_clearance),
        support_side_clearance=float(support_side_clearance),
        projection_tol=float(projection_tol),
        projection_damping=float(projection_damping),
        projection_max_iters=int(projection_max_iters),
    )

    L = geom.L
    W = geom.W
    G = geom.G
    ht = geom.table_height
    z_max = geom.z_max

    x_min, x_max = 0.0, 2.0 * L + G
    y_half = 0.5 * W
    y_min, y_max = -y_half, y_half
    ambient_bounds = [(x_min, x_max), (y_min, y_max), (0.0, z_max)]

    support_y_limit = max(0.0, y_half - geom.object_radius - geom.support_side_clearance)
    free_y_limit = max(0.0, y_half - geom.object_radius)

    support_z = ht
    carry_z_min = ht + geom.object_radius + geom.carry_clearance

    overlap_margin = 0.015
    min_transition_band = 0.05
    transition_band = max(min_transition_band, (carry_z_min - support_z) + overlap_margin)
    lift_band = transition_band
    place_band = transition_band

    edge_transition_x_width = transition_width
    if edge_transition_x_width is None:
        edge_transition_x_width = max(0.08, geom.object_half_length + 0.01)
    edge_transition_x_width = float(np.clip(edge_transition_x_width, 0.04, L))

    left_transition_x_min = max(0.0, L - edge_transition_x_width)
    left_transition_x_max = L
    right_transition_x_min = L + G
    right_transition_x_max = min(2.0 * L + G, L + G + edge_transition_x_width)

    def h_support_contact(p):
        return np.array([p[2] - support_z], dtype=float)

    def J_support_contact(p):
        return np.array([[0.0, 0.0, 1.0]], dtype=float)

    def proj_lift_from_left(p):
        pp = np.asarray(p, dtype=float).copy()
        pp[0] = np.clip(pp[0], left_transition_x_min, left_transition_x_max)
        pp[1] = np.clip(pp[1], -support_y_limit, support_y_limit)
        pp[2] = np.clip(pp[2], support_z, min(z_max, support_z + lift_band))
        return pp

    def proj_place_on_right(p):
        pp = np.asarray(p, dtype=float).copy()
        pp[0] = np.clip(pp[0], right_transition_x_min, right_transition_x_max)
        pp[1] = np.clip(pp[1], -support_y_limit, support_y_limit)
        pp[2] = np.clip(pp[2], support_z, min(z_max, support_z + place_band))
        return pp

    def proj_free_transfer_workspace(p):
        pp = np.asarray(p, dtype=float).copy()
        pp[0] = np.clip(pp[0], x_min, x_max)
        pp[1] = np.clip(pp[1], -free_y_limit, free_y_limit)
        pp[2] = np.clip(pp[2], carry_z_min, z_max)
        return pp

    def valid_slide_left(p):
        x, y, z = p
        return (
            0.0 <= x <= L
            and -support_y_limit <= y <= support_y_limit
            and abs(z - support_z) <= 5e-3
        )

    def valid_slide_right(p):
        x, y, z = p
        return (
            L + G <= x <= 2.0 * L + G
            and -support_y_limit <= y <= support_y_limit
            and abs(z - support_z) <= 5e-3
        )

    def valid_lift_from_left(p):
        x, y, z = p
        return (
            left_transition_x_min <= x <= left_transition_x_max
            and -support_y_limit <= y <= support_y_limit
            and support_z <= z <= min(z_max, support_z + lift_band)
        )

    def valid_place_on_right(p):
        x, y, z = p
        return (
            right_transition_x_min <= x <= right_transition_x_max
            and -support_y_limit <= y <= support_y_limit
            and support_z <= z <= min(z_max, support_z + place_band)
        )

    def valid_free_transfer_workspace(p):
        x, y, z = p
        return (
            x_min <= x <= x_max
            and -free_y_limit <= y <= free_y_limit
            and carry_z_min <= z <= z_max
        )

    slide_cost = {"z_penalty": 0.0, "mode_penalty": 0.0}
    transition_cost = {"z_penalty": 0.5, "mode_penalty": 0.2}
    free_cost = {"z_penalty": 1.2, "mode_penalty": 0.8}

    common_meta = {
        "representation": "hybrid-explicit-implicit",
        "state_abstraction": "reduced task position x=(x,y,z)",
        "future_state_model": "object pose + robot configuration",
        "tsr_ready": True,
        "projection_method": "damped_jacobian_least_squares",
        "geometry_parameters": {
            "L": L,
            "W": W,
            "G": G,
            "table_height": ht,
            "object_half_length": geom.object_half_length,
            "object_radius": geom.object_radius,
            "carry_clearance": geom.carry_clearance,
            "support_side_clearance": geom.support_side_clearance,
        },
        "derived_parameters": {
            "support_z": support_z,
            "support_y_limit": support_y_limit,
            "free_y_limit": free_y_limit,
            "carry_z_min": carry_z_min,
            "place_band": place_band,
            "lift_band": lift_band,
            "edge_transition_x_width": edge_transition_x_width,
            "left_transition_x_min": left_transition_x_min,
            "left_transition_x_max": left_transition_x_max,
            "right_transition_x_min": right_transition_x_min,
            "right_transition_x_max": right_transition_x_max,
        },
        "planning_policy_note": (
            "Support-contact motion is intentionally preferred. "
            "Free transfer is allowed but penalized, so it is chosen only when needed."
        ),
    }

    modes = [
        ImplicitMode(
            name="SlideLeft",
            family="SupportContact",
            h_func=h_support_contact,
            J_func=J_support_contact,
            ambient_bounds=ambient_bounds,
            is_valid=valid_slide_left,
            cost_weight=slide_cost,
            params={"side": "left", "support_z": support_z, "support_y_limit": support_y_limit},
            meta={
                **common_meta,
                "semantic_role": "preferred support-contact transport on left table",
                "orientation_semantics": _orientation_semantics_for_family("SupportContact"),
            },
            max_iters=geom.projection_max_iters,
            tol=geom.projection_tol,
            damping=geom.projection_damping,
            backtracking=True,
        ),
        Mode(
            name="LiftFromLeftSupport",
            family="SupportTransition",
            projector=proj_lift_from_left,
            ambient_bounds=ambient_bounds,
            is_valid=valid_lift_from_left,
            cost_weight=transition_cost,
            params={
                "side": "left",
                "x_range": [left_transition_x_min, left_transition_x_max],
                "support_z": support_z,
                "lift_band": lift_band,
            },
            meta={
                **common_meta,
                "semantic_role": "lift initiation from left support",
                "orientation_semantics": _orientation_semantics_for_family("SupportTransition"),
            },
        ),
        Mode(
            name="FreeTransferWorkspace",
            family="FreeTransfer",
            projector=proj_free_transfer_workspace,
            ambient_bounds=ambient_bounds,
            is_valid=valid_free_transfer_workspace,
            cost_weight=free_cost,
            params={"x_range": [x_min, x_max], "y_range": [-free_y_limit, free_y_limit], "carry_z_min": carry_z_min},
            meta={
                **common_meta,
                "semantic_role": "workspace-wide unsupported transfer, penalized",
                "orientation_semantics": _orientation_semantics_for_family("FreeTransfer"),
            },
        ),
        Mode(
            name="PlaceOnRightSupport",
            family="SupportTransition",
            projector=proj_place_on_right,
            ambient_bounds=ambient_bounds,
            is_valid=valid_place_on_right,
            cost_weight=transition_cost,
            params={
                "side": "right",
                "x_range": [right_transition_x_min, right_transition_x_max],
                "support_z": support_z,
                "place_band": place_band,
            },
            meta={
                **common_meta,
                "semantic_role": "placement corridor onto right support",
                "orientation_semantics": _orientation_semantics_for_family("SupportTransition"),
            },
        ),
        ImplicitMode(
            name="SlideRight",
            family="SupportContact",
            h_func=h_support_contact,
            J_func=J_support_contact,
            ambient_bounds=ambient_bounds,
            is_valid=valid_slide_right,
            cost_weight=slide_cost,
            params={"side": "right", "support_z": support_z, "support_y_limit": support_y_limit},
            meta={
                **common_meta,
                "semantic_role": "preferred support-contact transport on right table",
                "orientation_semantics": _orientation_semantics_for_family("SupportContact"),
            },
            max_iters=geom.projection_max_iters,
            tol=geom.projection_tol,
            damping=geom.projection_damping,
            backtracking=True,
        ),
    ]

    meta = {
        "L": L,
        "W": W,
        "G": G,
        "table_height": ht,
        "z_max": z_max,
        "ambient_bounds": ambient_bounds,
        "representation": "hybrid-explicit-implicit",
        "mode_families": ["SupportContact", "SupportTransition", "FreeTransfer"],
        "frozen_mode_semantics": True,
        "geometry_parameters": {
            "object_half_length": geom.object_half_length,
            "object_radius": geom.object_radius,
            "carry_clearance": geom.carry_clearance,
            "support_side_clearance": geom.support_side_clearance,
        },
        "derived_parameters": {
            "support_z": support_z,
            "support_y_limit": support_y_limit,
            "free_y_limit": free_y_limit,
            "carry_z_min": carry_z_min,
            "place_band": place_band,
            "lift_band": lift_band,
            "edge_transition_x_width": edge_transition_x_width,
            "left_transition_x_min": left_transition_x_min,
            "left_transition_x_max": left_transition_x_max,
            "right_transition_x_min": right_transition_x_min,
            "right_transition_x_max": right_transition_x_max,
        },
        "planning_preference": "prefer_support_contact_over_free_transfer",
        "projection_method": "damped_jacobian_least_squares",
        "tsr_ready": True,
    }

    return modes, ambient_bounds, meta