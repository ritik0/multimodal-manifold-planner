from __future__ import annotations

import numpy as np

from .modes import make_two_tables_problem_3d
from .planner_v2 import plan_multimodal_v2
from .robot_viewer import visualize_two_tables_with_robot
from .robot_config import (
    get_default_q_home,
    get_default_robot_base_pose,
    get_default_urdf_path,
    get_default_ee_link_name,
)
from .iiwa_ik import IiwaIK


def _largest_true_run(mask: np.ndarray):
    best_s = 0
    best_e = 0
    cur_s = None

    for i, v in enumerate(mask):
        if v and cur_s is None:
            cur_s = i
        elif not v and cur_s is not None:
            if i - cur_s > best_e - best_s:
                best_s, best_e = cur_s, i
            cur_s = None

    if cur_s is not None and len(mask) - cur_s > best_e - best_s:
        best_s, best_e = cur_s, len(mask)

    return best_s, best_e


def demo_run_and_visualize_robot():
    L = 0.45
    W = 0.40
    G = 0.30
    table_height = 0.75

    x_start = np.array([0.10, 0.0, table_height])
    x_goal = np.array([2.0 * L + G - 0.10, 0.0, table_height])

    modes, ambient_bounds, meta = make_two_tables_problem_3d(
        L=L,
        W=W,
        G=G,
        table_height=table_height,
        transition_width=0.12,
        z_max=1.60,
        lift_epsilon=0.02,
        feasible_edge_margin=0.12,
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
        rrt_step=0.06,
        rrt_iters=9000,
        rrt_time_budget_sec=4.0,
        goal_bias=0.30,
        forbid_direct_in_modes=["CarryFree"],
        transition_pick_policy="closest_on_src",
    )

    seq_names = [modes[i].name for i in dbg.mode_sequence]
    print("v2 mode sequence:", " -> ".join(seq_names))
    print("path points:", len(path))

    base_pose = get_default_robot_base_pose()
    q_home = np.asarray(get_default_q_home(), dtype=float)

    ik = IiwaIK(
        urdf_path=str(get_default_urdf_path()),
        ee_link_name=get_default_ee_link_name(),
        base_xyz=base_pose["xyz"],
        base_yaw_deg=base_pose["yaw_deg"],
    )

    clearance_z = table_height + 0.08
    base_tool_offset = 0.20
    extra_table_offset = 0.08

    exec_path = path.copy()

    ik_targets = exec_path.copy()
    ik_targets[:, 2] += base_tool_offset

    on_table_mask = exec_path[:, 2] <= table_height + 1e-3
    ik_targets[on_table_mask, 2] += extra_table_offset
    ik_targets[:, 2] = np.maximum(ik_targets[:, 2], clearance_z + base_tool_offset)

    stride = 2
    full_vis_path = exec_path[::stride]
    ik_targets = ik_targets[::stride]

    print("Visualized waypoints:", len(full_vis_path))
    print("First vis point:", full_vis_path[0])
    print("Last vis point :", full_vis_path[-1])

    q_list = []
    oks = []
    errs = []

    q_prev = q_home.copy()

    for i, target in enumerate(ik_targets):
        q_sol, ok, err = ik.solve_position_ik(target, q_prev)
        q_list.append(q_sol.copy())
        oks.append(ok)
        errs.append(err)
        print(f"[{i:02d}] target={target} ok={ok} err={err:.6f}")
        q_prev = q_sol

    q_waypoints = np.asarray(q_list, dtype=float)
    oks = np.asarray(oks, dtype=bool)
    errs = np.asarray(errs, dtype=float)

    print("IK solved:", int(np.sum(oks)), "/", len(oks))
    print("Max IK err:", float(np.max(errs)))
    print("Mean IK err:", float(np.mean(errs)))

    s, e = _largest_true_run(oks)
    if e - s < 2:
        raise RuntimeError("No sufficiently long successful IK chunk found.")

    print("Using successful chunk:", s, "to", e - 1, f"(length {e - s})")

    q_waypoints_ok = q_waypoints[s:e]
    executed_path = full_vis_path[s:e]

    frames_per_segment = 12
    q_traj = []
    executed_anim_path = []

    if len(q_waypoints_ok) == 1:
        q_traj = q_waypoints_ok.copy()
        executed_anim_path = np.tile(executed_path[0][None, :], (len(q_traj), 1))
    else:
        for i in range(len(q_waypoints_ok) - 1):
            qa = q_waypoints_ok[i]
            qb = q_waypoints_ok[i + 1]
            pa = executed_path[i]
            pb = executed_path[i + 1]

            for a in np.linspace(0.0, 1.0, frames_per_segment, endpoint=False):
                q_traj.append((1.0 - a) * qa + a * qb)
                executed_anim_path.append((1.0 - a) * pa + a * pb)

        q_traj.append(q_waypoints_ok[-1])
        executed_anim_path.append(executed_path[-1])

        q_traj = np.asarray(q_traj, dtype=float)
        executed_anim_path = np.asarray(executed_anim_path, dtype=float)

    print("Animation frames:", len(q_traj))

    visualize_two_tables_with_robot(
        path=executed_anim_path,
        meta=meta,
        urdf_path=get_default_urdf_path(),
        q_home=q_home,
        q_traj=q_traj,
        robot_base_xyz=base_pose["xyz"],
        robot_base_yaw_deg=base_pose["yaw_deg"],
        show_points=False,
        animate_payload=False,
        full_path=full_vis_path,
        executed_path=executed_path,
        title=f"iiwa execution | executed chunk {s}-{e-1}",
    )