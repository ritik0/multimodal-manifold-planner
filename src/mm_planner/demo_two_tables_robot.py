from __future__ import annotations

import numpy as np

from .modes import make_two_tables_problem_3d
from .planner_v2 import plan_multimodal_v2
from .robot_viewer import visualize_two_tables_with_robot
from .robot_config import get_default_q_home, get_default_urdf_path


def demo_run_and_visualize_robot():
    L = 1.0
    W = 0.6
    G = 1.0
    table_height = 0.75

    x_start = np.array([0.2, 0.0, table_height])
    x_goal = np.array([2.0 * L + G - 0.2, 0.0, table_height])

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
        attempts_per_pair=4000,
        max_transitions_per_edge=5,
        base_switch_cost=1.0,
        rrt_step=0.12,
        rrt_iters=9000,
        rrt_time_budget_sec=4.0,
        goal_bias=0.30,
        forbid_direct_in_modes=["CarryFree"],
        transition_pick_policy="closest_on_src",
    )

    seq_names = [modes[i].name for i in dbg.mode_sequence]
    print("v2 mode sequence:", " -> ".join(seq_names))
    print("path points:", len(path))

    visualize_two_tables_with_robot(
        path=path,
        meta=meta,
        urdf_path=get_default_urdf_path(),
        q_home=get_default_q_home(),
        show_points=True,
        animate_payload=True,
    )