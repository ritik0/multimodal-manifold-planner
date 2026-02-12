import numpy as np
import pyvista as pv
from .modes import make_two_tables_problem_3d
from .planner_v2 import plan_multimodal_v2


def visualize_two_tables_3d_pyvista(path, meta, show_points=True):

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

    # Lift zones include z == ht so that they intersect the slide manifolds.
    lift_left_zone = pv.Box(bounds=(L - w, L, y_min, y_max, ht, z_max))
    carry_zone = pv.Box(bounds=(L, L + G, y_min, y_max, ht + eps, z_max))
    lift_right_zone = pv.Box(bounds=(L + G, L + G + w, y_min, y_max, ht, z_max))

    pts = np.asarray(path, dtype=float)
    poly = pv.PolyData(pts)
    line = pv.Spline(pts, len(pts))

    start_s = pv.Sphere(radius=0.03, center=pts[0])
    goal_s = pv.Sphere(radius=0.03, center=pts[-1])

    p = pv.Plotter()
    p.add_mesh(left_table, opacity=0.5)
    p.add_mesh(right_table, opacity=0.5)

    p.add_mesh(lift_left_zone, opacity=0.10)
    p.add_mesh(carry_zone, opacity=0.06)
    p.add_mesh(lift_right_zone, opacity=0.10)

    p.add_mesh(line, line_width=4)
    if show_points:
        p.add_mesh(poly, point_size=6, render_points_as_spheres=True)

    p.add_mesh(start_s)
    p.add_mesh(goal_s)

    p.add_axes()
    p.add_title("Two-table multimodal")
    p.show()


def demo_run_and_visualize_v2():
    # Scene params
    L = 1.0
    W = 0.6
    G = 1.0
    table_height = 0.75

    # Fixed start/goal (change if you want random)
    x_start = np.array([0.2, 0.0, table_height])
    x_goal = np.array([2.0 * L + G - 0.2, 0.0, table_height])
    # Put it around middle of the gap at safe carry height
    # x_goal = np.array([L + 0.5 * G, 0.0, table_height + 0.25])

    modes, ambient_bounds, meta = make_two_tables_problem_3d(
        L=L,
        W=W,
        G=G,
        table_height=table_height,
        transition_width=0.2,
        z_max=2.0,
        lift_epsilon=0.02,
        feasible_edge_margin=0.25,
    )

    path, dbg = plan_multimodal_v2(
        x_start,
        x_goal,
        modes=modes,
        ambient_bounds=ambient_bounds,
        meta=meta,
        # graph build
        attempts_per_pair=4000,
        max_transitions_per_edge=5,
        base_switch_cost=1.0,
        # continuous planner
        rrt_step=0.12,
        rrt_iters=9000,
        rrt_time_budget_sec=4.0,
        goal_bias=0.30,
        # keep carry planned (no direct fallback in CarryFree)
        forbid_direct_in_modes=["CarryFree"],
        transition_pick_policy="closest_on_src",
    )

    seq_names = [modes[i].name for i in dbg.mode_sequence]
    print("v2 mode sequence:", " -> ".join(seq_names))
    print("path points:", len(path))

    visualize_two_tables_3d_pyvista(path, meta, show_points=True)


# Keep old name if scripts/run_demo.py expects demo_run_and_visualize()
def demo_run_and_visualize():
    demo_run_and_visualize_v2()


if __name__ == "__main__":
    demo_run_and_visualize_v2()