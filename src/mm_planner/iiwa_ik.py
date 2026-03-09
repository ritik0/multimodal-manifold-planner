from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from urdfpy import URDF


def rotz_deg(yaw_deg: float) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)


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

    def _cfg_dict(self, q):
        return {name: float(val) for name, val in zip(self.joint_names, q)}

    def fk_local(self, q: np.ndarray) -> np.ndarray:
        fk = self.robot.link_fk(cfg=self._cfg_dict(q))
        T = fk[self.ee_link]
        return np.asarray(T[:3, 3], dtype=float)

    def fk_world(self, q: np.ndarray) -> np.ndarray:
        p_local = self.fk_local(q)
        return self.base_R @ p_local + self.base_xyz

    def solve_position_ik(self, target_xyz, q0):
        target_xyz = np.asarray(target_xyz, dtype=float)
        q0 = np.asarray(q0, dtype=float)

        def residual(q):
            return self.fk_world(q) - target_xyz

        res = least_squares(
            residual,
            q0,
            bounds=(self.lower, self.upper),
            xtol=1e-5,
            ftol=1e-5,
            gtol=1e-5,
            max_nfev=300,
        )

        q_sol = res.x
        p_sol = self.fk_world(q_sol)
        err = np.linalg.norm(p_sol - target_xyz)

        # simple practical success test
        ok = (err < 0.08) and (p_sol[2] >= 0.83)
        return q_sol, ok, err