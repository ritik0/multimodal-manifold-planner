from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pyvista as pv
from robot_visualization import Robot


def rotz_deg(yaw_deg: float) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _build_two_table_scene(plotter, meta):
    L = meta["L"]
    W = meta["W"]
    G = meta["G"]
    ht = meta["table_height"]

    t = 0.05
    y_min, y_max = -W / 2.0, W / 2.0

    left_table = pv.Box(bounds=(0.0, L, y_min, y_max, ht - t, ht))
    right_table = pv.Box(bounds=(L + G, 2 * L + G, y_min, y_max, ht - t, ht))

    plotter.add_mesh(left_table, color="lightsteelblue", opacity=0.45)
    plotter.add_mesh(right_table, color="lightsteelblue", opacity=0.45)


def visualize_two_tables_with_robot(
    path,
    meta,
    urdf_path,
    q_home,
    *,
    q_traj=None,
    robot_base_xyz=(0.72, -0.55, 0.0),
    robot_base_yaw_deg=90.0,
    show_points=False,
    animate_payload=False,
    full_path=None,
    executed_path=None,
    title="iiwa execution",
):
    pts = np.asarray(path, dtype=float)
    q_home = np.asarray(q_home, dtype=float)
    q_traj = None if q_traj is None else np.asarray(q_traj, dtype=float)

    p = pv.Plotter()
    _build_two_table_scene(p, meta)

    # Full planner path: faint gray
    if full_path is not None:
        full_path = np.asarray(full_path, dtype=float)
        if len(full_path) >= 2:
            full_line = pv.Spline(full_path, len(full_path))
            p.add_mesh(full_line, color="lightgray", line_width=2, opacity=0.35)

    # Executed valid chunk: green
    if executed_path is not None:
        executed_path = np.asarray(executed_path, dtype=float)
        if len(executed_path) >= 2:
            exec_line = pv.Spline(executed_path, len(executed_path))
            p.add_mesh(exec_line, color="limegreen", line_width=6, opacity=0.95)

    # Animated path: red
    if len(pts) >= 2:
        anim_line = pv.Spline(pts, len(pts))
        p.add_mesh(anim_line, color="red", line_width=3, opacity=0.85)

    if show_points and len(pts) > 0:
        poly = pv.PolyData(pts)
        p.add_mesh(poly, color="gold", point_size=6, render_points_as_spheres=True)

    robot = Robot(
        str(Path(urdf_path).resolve()),
        plotter=p,
        p0=np.asarray(robot_base_xyz, dtype=float),
        R0=rotz_deg(robot_base_yaw_deg),
    )
    robot.set_robot_mesh()
    robot.update(q_home)

    instruction = "R = replay   Q = close"
    p.add_axes()
    p.add_title(title)
    p.add_text(instruction, position="upper_left", font_size=10, color="white")

    payload_actor = None
    if animate_payload and len(pts) > 0:
        payload = pv.Sphere(radius=0.03, center=pts[0])
        payload_actor = p.add_mesh(payload, color="orange")

    state = {
        "replay": True,   # auto-play once on launch
        "running": True,
    }

    def request_replay():
        state["replay"] = True

    def request_close():
        state["running"] = False

    p.add_key_event("r", request_replay)
    p.add_key_event("q", request_close)

    p.show(auto_close=False, interactive_update=True)

    def animate_once():
        robot.update(q_home)
        p.update()
        time.sleep(0.05)

        if q_traj is not None:
            for q in q_traj:
                if not state["running"]:
                    return
                robot.update(q)
                p.update()
                time.sleep(0.04)

        if animate_payload and payload_actor is not None and len(pts) > 0:
            for pt in pts:
                if not state["running"]:
                    return
                payload_i = pv.Sphere(radius=0.03, center=pt)
                payload_actor.mapper.SetInputData(payload_i)
                p.update()
                time.sleep(0.03)

    while state["running"]:
        if state["replay"]:
            state["replay"] = False
            animate_once()

        try:
            p.update()
        except Exception:
            break

        time.sleep(0.02)

    try:
        p.close()
    except Exception:
        pass