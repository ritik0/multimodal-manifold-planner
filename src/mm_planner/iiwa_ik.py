from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from urdfpy import URDF


def rotz_deg(yaw_deg: float) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


class IiwaIK:
    def __init__(self, urdf_path: str, ee_link_name: str, base_xyz, base_yaw_deg: float):
        self.robot = URDF.load(str(urdf_path))
        self.ee_link_name = ee_link_name
        self.base_xyz = np.asarray(base_xyz, dtype=float)
        self.base_R = rotz_deg(base_yaw_deg)

        self.ee_link = None
        for l in self.robot.links:
            if l.name == ee_link_name:
                self.ee_link = l
                break
        if self.ee_link is None:
            raise ValueError(f"End-effector link '{ee_link_name}' not found")

        self.actuated_joints = [j for j in self.robot.joints if j.joint_type != "fixed"]
        self.joint_names = [j.name for j in self.actuated_joints]

        lower, upper = [], []
        for j in self.actuated_joints:
            if j.limit is not None:
                lower.append(j.limit.lower if j.limit.lower is not None else -np.pi)
                upper.append(j.limit.upper if j.limit.upper is not None else np.pi)
            else:
                lower.append(-np.pi)
                upper.append(np.pi)

        self.lower = np.array(lower, dtype=float)
        self.upper = np.array(upper, dtype=float)

        self.main_link_names = [
            "iiwa_link_0",
            "iiwa_link_1",
            "iiwa_link_2",
            "iiwa_link_3",
            "iiwa_link_4",
            "iiwa_link_5",
            "iiwa_link_6",
            "iiwa_link_7",
            "iiwa_link_ee",
        ]

    def _cfg_dict(self, q):
        return {name: float(val) for name, val in zip(self.joint_names, q)}

    def fk_local_transform(self, q: np.ndarray) -> np.ndarray:
        fk = self.robot.link_fk(cfg=self._cfg_dict(q))
        return np.asarray(fk[self.ee_link], dtype=float)

    def fk_world_transform(self, q: np.ndarray) -> np.ndarray:
        T_local = self.fk_local_transform(q)
        T_world = np.eye(4, dtype=float)
        T_world[:3, :3] = self.base_R @ T_local[:3, :3]
        T_world[:3, 3] = self.base_R @ T_local[:3, 3] + self.base_xyz
        return T_world

    def fk_world(self, q: np.ndarray) -> np.ndarray:
        return self.fk_world_transform(q)[:3, 3]

    def fk_all_link_world_points(self, q: np.ndarray) -> np.ndarray:
        fk = self.robot.link_fk(cfg=self._cfg_dict(q))
        pts = []

        for link_name in self.main_link_names:
            link_obj = None
            for l in self.robot.links:
                if l.name == link_name:
                    link_obj = l
                    break
            if link_obj is None or link_obj not in fk:
                continue

            T_local = np.asarray(fk[link_obj], dtype=float)
            p_local = T_local[:3, 3]
            p_world = self.base_R @ p_local + self.base_xyz
            pts.append(p_world)

        return np.asarray(pts, dtype=float)

    @staticmethod
    def _sample_segment_points(points: np.ndarray, samples_per_segment: int = 10) -> np.ndarray:
        if len(points) <= 1:
            return points.copy()

        dense = []
        for i in range(len(points) - 1):
            a = points[i]
            b = points[i + 1]
            for t in np.linspace(0.0, 1.0, samples_per_segment, endpoint=False):
                dense.append((1.0 - t) * a + t * b)
        dense.append(points[-1])
        return np.asarray(dense, dtype=float)

    @staticmethod
    def _points_in_box(points: np.ndarray, bounds, margin: float = 0.0) -> np.ndarray:
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        return (
            (points[:, 0] >= xmin - margin)
            & (points[:, 0] <= xmax + margin)
            & (points[:, 1] >= ymin - margin)
            & (points[:, 1] <= ymax + margin)
            & (points[:, 2] >= zmin - margin)
            & (points[:, 2] <= zmax + margin)
        )

    def collides_with_boxes(
        self,
        q: np.ndarray,
        boxes,
        margin: float = 0.05,
        samples_per_segment: int = 10,
        skip_first_n_points: int = 1,
    ) -> bool:
        link_pts = self.fk_all_link_world_points(q)
        if len(link_pts) == 0:
            return False

        dense_pts = self._sample_segment_points(link_pts, samples_per_segment=samples_per_segment)

        if skip_first_n_points > 0 and len(dense_pts) > skip_first_n_points:
            dense_pts = dense_pts[skip_first_n_points:]

        for box in boxes:
            if np.any(self._points_in_box(dense_pts, box, margin=margin)):
                return True
        return False

    def solve_position_ik(
        self,
        target_xyz,
        q0,
        pos_tol: float = 0.04,
        collision_boxes=None,
        collision_margin: float = 0.05,
        min_ee_z: float | None = None,
        n_restarts: int = 3,
        smooth_weight: float = 0.08,
    ):
        target_xyz = np.asarray(target_xyz, dtype=float)
        q0 = np.asarray(q0, dtype=float)

        def residual(q):
            pos_res = self.fk_world(q) - target_xyz
            smooth_res = smooth_weight * (q - q0)
            return np.concatenate([pos_res, smooth_res])

        seeds = [q0.copy()]
        for _ in range(n_restarts - 1):
            noise = np.random.normal(scale=0.08, size=q0.shape)
            q_seed = np.clip(q0 + noise, self.lower, self.upper)
            seeds.append(q_seed)

        best_q = None
        best_err = np.inf
        best_ok = False

        for seed in seeds:
            res = least_squares(
                residual,
                seed,
                bounds=(self.lower, self.upper),
                xtol=1e-5,
                ftol=1e-5,
                gtol=1e-5,
                max_nfev=250,
            )

            q_sol = res.x
            p_sol = self.fk_world(q_sol)
            err = np.linalg.norm(p_sol - target_xyz)

            ee_height_ok = True if min_ee_z is None else (p_sol[2] >= min_ee_z)

            collision_ok = True
            if collision_boxes is not None:
                collision_ok = not self.collides_with_boxes(
                    q_sol,
                    collision_boxes,
                    margin=collision_margin,
                    samples_per_segment=10,
                    skip_first_n_points=1,
                )

            ok = (err < pos_tol) and ee_height_ok and collision_ok

            if ok and err < best_err:
                best_q = q_sol
                best_err = err
                best_ok = True

            if (not best_ok) and (err < best_err):
                best_q = q_sol
                best_err = err

        return best_q, best_ok, float(best_err)