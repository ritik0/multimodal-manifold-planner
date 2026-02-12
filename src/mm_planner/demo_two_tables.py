import numpy as np

from .modes import make_two_tables_problem_3d
from .rrt import (
    rrt_connect_on_mode,
    direct_connect_on_mode,
)
from .transitions import sample_transition


def plan_multimodal_two_tables_3d(
    x_start,
    x_goal,
    *,
    L=1.0, W=0.6, G=1.0,
    table_height=0.75,
    transition_width=0.2,
    z_max=2.0,
    lift_epsilon=0.02,
    feasible_edge_margin=0.25,
    seed=7,
    step=0.12,
):
    np.random.seed(seed)

    modes, _, meta = make_two_tables_problem_3d(
        L=L, W=W, G=G,
        table_height=table_height,
        transition_width=transition_width,
        z_max=z_max,
        lift_epsilon=lift_epsilon,
        feasible_edge_margin=feasible_edge_margin,
    )

    # Project start/goal to their respective slide manifolds.
    x_cur = np.asarray(modes[0].project(x_start), dtype=float)
    x_goal_proj = np.asarray(modes[-1].project(x_goal), dtype=float)

    # ------------------------------------------------------------
    # Explicit intersection-based transitions
    # ------------------------------------------------------------
    # We sample a transition configuration in the intersection for each
    # adjacent pair (A,B), then plan within A to reach it, then continue.
    # This forces the Carry segment to be planned (not directly jumped).
    ambient_bounds = meta["ambient_bounds"]

    transition_points = []
    for i in range(len(modes) - 1):
        A = modes[i]
        B = modes[i + 1]
        xT = sample_transition(A, B, ambient_bounds, attempts=4000)
        if xT is None:
            raise RuntimeError(
                f"Could not find an intersection sample for transition {A.name} -> {B.name}. "
                "Try increasing transition_width, reducing lift_epsilon, or increasing attempts."
            )
        transition_points.append(np.asarray(xT, dtype=float))

    full_path = [x_cur.copy()]

    # Plan sequentially in each mode to the intersection point
    for i, A in enumerate(modes[:-1]):
        xT = transition_points[i]
        xT_A = np.asarray(A.project(xT), dtype=float)

        seg = rrt_connect_on_mode(
            x_cur,
            xT_A,
            A,
            iters=8000,
            step=step,
            goal_bias=0.25,
            table_height_for_cost=table_height,
            max_nodes=20000,
            time_budget_sec=4.0,
            stop_after_first=True,
        )
        if seg is None:
            seg = direct_connect_on_mode(x_cur, xT_A, A, step=min(step, 0.06))
        if seg is None:
            raise RuntimeError(
                f"Could not plan inside mode {A.name} to reach its transition point. "
                "Try increasing iters/time_budget or lowering step."
            )

        full_path.extend([np.asarray(p, dtype=float) for p in seg[1:]])

        # Switch: since the endpoint lies in the intersection, we can enter B.
        B = modes[i + 1]
        x_cur = np.asarray(B.project(full_path[-1]), dtype=float)

    # Final segment on SlideRight to goal.
    seg_last = rrt_connect_on_mode(
        x_cur,
        x_goal_proj,
        modes[-1],
        iters=8000,
        step=step,
        goal_bias=0.35,
        table_height_for_cost=table_height,
        max_nodes=20000,
        time_budget_sec=4.0,
        stop_after_first=True,
    )
    if seg_last is None:
        seg_last = direct_connect_on_mode(x_cur, x_goal_proj, modes[-1], step=min(step, 0.06))
    if seg_last is None:
        raise RuntimeError(f"Could not plan on {modes[-1].name} to goal.")

    full_path.extend([np.asarray(p, dtype=float) for p in seg_last[1:]])
    return np.asarray(full_path, dtype=float), modes, meta


def visualize_two_tables_3d_pyvista(path, meta, show_points=True):
    import pyvista as pv

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
    p.add_title("Two-table multimodal (intersection-based switching + planned carry)")
    p.show()


def demo_run_and_visualize():
    L = 1.0
    W = 0.6
    G = 1.0
    table_height = 0.75
    #### START AND GOAL POSITIONS
    x_start = np.array([0.2, 0.0, table_height])
    x_goal = np.array([2.0 * L + G - 0.2, 0.0, table_height])

    path, modes, meta = plan_multimodal_two_tables_3d(
        x_start=x_start,
        x_goal=x_goal,
        L=L, W=W, G=G,
        table_height=table_height,
        transition_width=0.2,
        z_max=2.0,
        lift_epsilon=0.02,
        feasible_edge_margin=0.25,
        seed=7,
        step=0.12,
    )

    print("Modes:", " -> ".join([m.name for m in modes]))
    print("Path points:", len(path))
    visualize_two_tables_3d_pyvista(path, meta, show_points=True)


if __name__ == "__main__":
    demo_run_and_visualize()