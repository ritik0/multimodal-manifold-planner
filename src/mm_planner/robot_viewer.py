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
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _build_two_table_scene(plotter: pv.Plotter, meta: dict) -> None:
    L = meta["L"]
    W = meta["W"]
    G = meta["G"]
    ht = meta["table_height"]

    t = 0.05
    y_min, y_max = -W / 2.0, W / 2.0

    left_table = pv.Box(bounds=(0.0, L, y_min, y_max, ht - t, ht))
    right_table = pv.Box(bounds=(L + G, 2 * L + G, y_min, y_max, ht - t, ht))

    plotter.add_mesh(left_table, color="lightsteelblue", opacity=0.55)
    plotter.add_mesh(right_table, color="lightsteelblue", opacity=0.55)


def _set_default_front_camera(plotter: pv.Plotter, meta: dict) -> None:
    L = meta["L"]
    G = meta["G"]
    ht = meta["table_height"]

    scene_center_x = (2.0 * L + G) * 0.5
    focal = (scene_center_x, 0.0, ht - 0.03)

    position = (scene_center_x, 1.95, ht + 0.38)
    viewup = (0.0, 0.0, 1.0)

    plotter.camera_position = [position, focal, viewup]
    plotter.camera.zoom(0.92)


def _add_marker(plotter: pv.Plotter, point, color="yellow", radius=0.018):
    sphere = pv.Sphere(radius=radius, center=np.asarray(point, dtype=float))
    plotter.add_mesh(sphere, color=color)


def _add_path_lines(
    plotter: pv.Plotter,
    *,
    full_path: np.ndarray | None,
    executed_path: np.ndarray | None,
    animated_path: np.ndarray | None,
) -> None:
    if full_path is not None and len(full_path) >= 2:
        full_line = pv.Spline(full_path, len(full_path))
        plotter.add_mesh(full_line, color="lightgray", line_width=2, opacity=0.25)
        _add_marker(plotter, full_path[0], color="white", radius=0.013)
        _add_marker(plotter, full_path[-1], color="black", radius=0.013)

    if executed_path is not None and len(executed_path) >= 2:
        exec_line = pv.Spline(executed_path, len(executed_path))
        plotter.add_mesh(exec_line, color="limegreen", line_width=6, opacity=0.95)
        _add_marker(plotter, executed_path[0], color="lime", radius=0.018)
        _add_marker(plotter, executed_path[-1], color="lime", radius=0.018)

    if animated_path is not None and len(animated_path) >= 2:
        anim_line = pv.Spline(animated_path, len(animated_path))
        plotter.add_mesh(anim_line, color="red", line_width=3, opacity=0.85)


def _make_payload_actor(plotter: pv.Plotter, start_pt: np.ndarray):
    payload_mesh = pv.Sphere(radius=0.03, center=start_pt)
    return plotter.add_mesh(payload_mesh, color="orange")


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
    frame_dt=0.05,
    start_delay=0.15,
):
    pts = np.asarray(path, dtype=float)
    q_home = np.asarray(q_home, dtype=float)
    q_traj = None if q_traj is None else np.asarray(q_traj, dtype=float)

    full_path = None if full_path is None else np.asarray(full_path, dtype=float)
    executed_path = None if executed_path is None else np.asarray(executed_path, dtype=float)

    p = pv.Plotter(window_size=(1400, 900))
    _build_two_table_scene(p, meta)
    _add_path_lines(
        p,
        full_path=full_path,
        executed_path=executed_path,
        animated_path=pts,
    )
    _set_default_front_camera(p, meta)

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

    q_initial = q_home.copy()
    if q_traj is not None and len(q_traj) > 0:
        q_initial = q_traj[0].copy()

    robot.update(q_initial)

    payload_actor = None
    if animate_payload and len(pts) > 0:
        payload_actor = _make_payload_actor(p, pts[0])

    p.add_axes()
    p.add_title(title)
    p.add_text(
        "gray: full path | green: robot-can-follow part | red: current animated motion",
        position="upper_left",
        font_size=10,
        color="white",
    )

    p.show(auto_close=False, interactive_update=True)
    time.sleep(start_delay)

    if q_traj is not None and len(q_traj) > 0:
        if animate_payload and payload_actor is not None and len(pts) == len(q_traj):
            for q, pt in zip(q_traj, pts):
                robot.update(q)
                payload_mesh = pv.Sphere(radius=0.03, center=pt)
                payload_actor.mapper.SetInputData(payload_mesh)
                p.update()
                time.sleep(frame_dt)
        else:
            for q in q_traj:
                robot.update(q)
                p.update()
                time.sleep(frame_dt)

    p.show(auto_close=True)