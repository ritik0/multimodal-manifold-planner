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


def rotz_rad(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def wrap_to_pi(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def rotation_matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    cos_theta = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if theta < 1e-9:
        return np.zeros(3, dtype=float)

    if np.pi - theta < 1e-6:
        axis = np.sqrt(np.maximum((np.diag(R) + 1.0) * 0.5, 0.0))
        axis = axis.astype(float)

        if abs(axis[0]) > 1e-6:
            axis[1] = np.copysign(axis[1], R[0, 1] + R[1, 0])
            axis[2] = np.copysign(axis[2], R[0, 2] + R[2, 0])
        elif abs(axis[1]) > 1e-6:
            axis[0] = np.copysign(axis[0], R[0, 1] + R[1, 0])
            axis[2] = np.copysign(axis[2], R[1, 2] + R[2, 1])
        else:
            axis[0] = np.copysign(axis[0], R[0, 2] + R[2, 0])
            axis[1] = np.copysign(axis[1], R[1, 2] + R[2, 1])

        n = np.linalg.norm(axis)
        if n < 1e-9:
            return np.zeros(3, dtype=float)
        axis = axis / n
        return axis * theta

    skew = np.array(
        [
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ],
        dtype=float,
    )
    axis = skew / (2.0 * np.sin(theta))
    return axis * theta


def orientation_error_rotvec(R_current: np.ndarray, R_desired: np.ndarray) -> np.ndarray:
    R_err = np.asarray(R_desired, dtype=float) @ np.asarray(R_current, dtype=float).T
    return rotation_matrix_to_rotvec(R_err)


def _xy_yaw_from_R(R: np.ndarray) -> float:
    x_axis = np.asarray(R, dtype=float)[:3, 0]
    v = x_axis[:2]
    n = np.linalg.norm(v)
    if n < 1e-9:
        return 0.0
    return float(np.arctan2(v[1], v[0]))


def make_payload_orientation(
    mode_family: str,
    travel_dir_world: np.ndarray | None = None,
    R_nominal_world: np.ndarray | None = None,
) -> np.ndarray:
    """
    Reachable orientation target for the iiwa flange.

    Strategy:
    - use the robot's actual home EE orientation as the nominal carrying pose
    - only apply a *small yaw correction around world z* following travel direction
    - keep the flange tilt/roll close to the nominal reachable posture

    This is much more stable than directly synthesizing a new flange frame.
    """
    if R_nominal_world is None:
        # Safe fallback if caller forgets to pass the nominal home EE orientation
        R_nominal_world = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    R_nominal_world = np.asarray(R_nominal_world, dtype=float)

    if travel_dir_world is None:
        return R_nominal_world.copy()

    d = np.asarray(travel_dir_world, dtype=float).copy()
    d[2] = 0.0
    n = np.linalg.norm(d)
    if n < 1e-9:
        return R_nominal_world.copy()

    desired_yaw = float(np.arctan2(d[1], d[0]))
    nominal_yaw = _xy_yaw_from_R(R_nominal_world)
    yaw_error = wrap_to_pi(desired_yaw - nominal_yaw)

    # Small, mode-dependent yaw following only.
    if mode_family == "SupportContact":
        yaw_gain = 0.15
        yaw_limit_deg = 12.0
    elif mode_family == "SupportTransition":
        yaw_gain = 0.20
        yaw_limit_deg = 18.0
    else:
        yaw_gain = 0.25
        yaw_limit_deg = 25.0

    yaw_cmd = yaw_gain * yaw_error
    yaw_lim = np.deg2rad(yaw_limit_deg)
    yaw_cmd = float(np.clip(yaw_cmd, -yaw_lim, yaw_lim))

    return rotz_rad(yaw_cmd) @ R_nominal_world


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

    def fk_world_rotation(self, q: np.ndarray) -> np.ndarray:
        return self.fk_world_transform(q)[:3, :3]

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

    def solve_pose_ik(
        self,
        target_xyz,
        target_R,
        q0,
        *,
        pos_tol: float = 0.07,
        rot_tol_deg: float = 20.0,
        collision_boxes=None,
        collision_margin: float = 0.02,
        min_ee_z: float | None = None,
        n_restarts: int = 4,
        smooth_weight: float = 0.08,
        orientation_weight: float = 0.10,
    ):
        target_xyz = np.asarray(target_xyz, dtype=float)
        target_R = np.asarray(target_R, dtype=float)
        q0 = np.asarray(q0, dtype=float)

        def residual(q):
            T = self.fk_world_transform(q)
            pos_res = T[:3, 3] - target_xyz
            rot_res = orientation_error_rotvec(T[:3, :3], target_R)
            smooth_res = smooth_weight * (q - q0)
            return np.concatenate([pos_res, orientation_weight * rot_res, smooth_res])

        seeds = [q0.copy()]
        for _ in range(n_restarts - 1):
            noise = np.random.normal(scale=0.10, size=q0.shape)
            q_seed = np.clip(q0 + noise, self.lower, self.upper)
            seeds.append(q_seed)

        best_q = None
        best_pos_err = np.inf
        best_rot_err_deg = np.inf
        best_ok = False

        for seed in seeds:
            res = least_squares(
                residual,
                seed,
                bounds=(self.lower, self.upper),
                xtol=1e-5,
                ftol=1e-5,
                gtol=1e-5,
                max_nfev=350,
            )

            q_sol = res.x
            T_sol = self.fk_world_transform(q_sol)
            p_sol = T_sol[:3, 3]
            R_sol = T_sol[:3, :3]

            pos_err = float(np.linalg.norm(p_sol - target_xyz))
            rot_err = orientation_error_rotvec(R_sol, target_R)
            rot_err_deg = float(np.rad2deg(np.linalg.norm(rot_err)))

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

            ok = (pos_err < pos_tol) and (rot_err_deg < rot_tol_deg) and ee_height_ok and collision_ok

            score = pos_err + 0.01 * rot_err_deg

            if ok:
                if (not best_ok) or (score < best_pos_err + 0.01 * best_rot_err_deg):
                    best_q = q_sol
                    best_pos_err = pos_err
                    best_rot_err_deg = rot_err_deg
                    best_ok = True
            else:
                if (not best_ok) and (score < best_pos_err + 0.01 * best_rot_err_deg):
                    best_q = q_sol
                    best_pos_err = pos_err
                    best_rot_err_deg = rot_err_deg

        return best_q, best_ok, float(best_pos_err), float(best_rot_err_deg)