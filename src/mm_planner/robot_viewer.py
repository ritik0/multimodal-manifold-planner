from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pyvista as pv
from robot_visualization import Robot


def _build_two_table_scene(plotter, meta):
    L = meta["L"]
    W = meta["W"]
    G = meta["G"]
    ht = meta["table_height"]
    w = meta["transition_width"]
    z_max = meta["z_max"]
    eps = meta["lift_epsilon"]

    t = 0.05
    y_min, y_max = -W / 2.0, W / 2.0

    left_table = pv.Box(bounds=(0.0, L, y_min, y_max, ht - t, ht))
    right_table = pv.Box(bounds=(L + G, 2 * L + G, y_min, y_max, ht - t, ht))

    lift_left_zone = pv.Box(bounds=(L - w, L, y_min, y_max, ht, z_max))
    carry_zone = pv.Box(bounds=(L, L + G, y_min, y_max, ht + eps, z_max))
    lift_right_zone = pv.Box(bounds=(L + G, L + G + w, y_min, y_max, ht, z_max))

    plotter.add_mesh(left_table, opacity=0.5)
    plotter.add_mesh(right_table, opacity=0.5)
    plotter.add_mesh(lift_left_zone, opacity=0.10)
    plotter.add_mesh(carry_zone, opacity=0.06)
    plotter.add_mesh(lift_right_zone, opacity=0.10)


def visualize_two_tables_with_robot(
    path,
    meta,
    urdf_path,
    q_home,
    *,
    show_points=True,
    animate_payload=True,
):
    pts = np.asarray(path, dtype=float)
    q_home = np.asarray(q_home, dtype=float)

    p = pv.Plotter()
    _build_two_table_scene(p, meta)

    poly = pv.PolyData(pts)
    line = pv.Spline(pts, len(pts))
    start_s = pv.Sphere(radius=0.03, center=pts[0])
    goal_s = pv.Sphere(radius=0.03, center=pts[-1])

    p.add_mesh(line, line_width=4)
    if show_points:
        p.add_mesh(poly, point_size=6, render_points_as_spheres=True)

    p.add_mesh(start_s)
    p.add_mesh(goal_s)

    robot = Robot(str(Path(urdf_path).resolve()), plotter=p)
    robot.set_robot_mesh()
    robot.update(q_home)

    payload = pv.Sphere(radius=0.03, center=pts[0])
    payload_actor = p.add_mesh(payload)

    p.add_axes()
    p.add_title("Two-table multimodal with robot")

    if not animate_payload:
        p.show()
        return

    p.show(auto_close=False)

    for i in range(len(pts)):
        payload_i = pv.Sphere(radius=0.03, center=pts[i])
        payload_actor.mapper.SetInputData(payload_i)
        p.render()
        time.sleep(0.03)

    p.show()