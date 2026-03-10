from __future__ import annotations

import time
from typing import Any

import numpy as np

from .modes import make_two_tables_problem_3d
from .planner_v2 import plan_multimodal_v2, pick_mode_for_state
from .robot_config import (
    get_default_q_home,
    get_default_robot_base_pose,
    get_default_urdf_path,
    get_default_ee_link_name,
    get_planning_height_limits,
)
from .iiwa_ik import IiwaIK
from .robot_viewer import visualize_two_tables_with_robot


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


def _start_anchored_true_run(mask: np.ndarray):
    e = 0
    while e < len(mask) and mask[e]:
        e += 1
    return 0, e


def _pick_execution_chunk(mask: np.ndarray, min_len: int = 2):
    s0, e0 = _start_anchored_true_run(mask)
    if e0 - s0 >= min_len:
        return s0, e0, "start-anchored"
    s, e = _largest_true_run(mask)
    return s, e, "largest-feasible"


def _make_table_collision_boxes(meta: dict):
    L = meta["L"]
    W = meta["W"]
    G = meta["G"]
    ht = meta["table_height"]

    y_min, y_max = -W / 2.0, W / 2.0
    left_box = (0.0, L, y_min, y_max, 0.0, ht)
    right_box = (L + G, 2.0 * L + G, y_min, y_max, 0.0, ht)
    return [left_box, right_box]


def _build_ik_targets(full_planned_path: np.ndarray, table_height: float, planning_max_z: float):
    clearance_z = table_height + 0.04
    base_tool_offset = 0.11
    extra_table_offset = 0.04

    ik_targets_full = full_planned_path.copy()
    ik_targets_full[:, 2] += base_tool_offset

    on_table_mask = full_planned_path[:, 2] <= table_height + 1e-3
    ik_targets_full[on_table_mask, 2] += extra_table_offset

    ik_targets_full[:, 2] = np.maximum(
        ik_targets_full[:, 2],
        clearance_z + base_tool_offset,
    )
    ik_targets_full[:, 2] = np.minimum(ik_targets_full[:, 2], planning_max_z + base_tool_offset)
    return ik_targets_full


def _make_mode_preference_costs(modes):
    extra_node_cost = {}
    for i, m in enumerate(modes):
        if m.family == "SupportContact":
            extra_node_cost[i] = 0.0
        elif m.family == "SupportTransition":
            extra_node_cost[i] = 0.25
        elif m.family == "FreeTransfer":
            extra_node_cost[i] = 1.00
        else:
            extra_node_cost[i] = 0.5
    return extra_node_cost


def _sample_path_keep_ends(path: np.ndarray, ik_targets: np.ndarray, stride: int = 2):
    n = len(path)
    idx = list(range(0, n, stride))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    idx = np.array(sorted(set(idx)), dtype=int)
    return path[idx], ik_targets[idx], idx


def evaluate_single_case(
    x_start: np.ndarray,
    x_goal: np.ndarray,
    *,
    planner_seed: int = 7,
    L: float = 0.45,
    W: float = 0.40,
    G: float = 0.30,
    table_height: float = 0.70,
    stride: int = 2,
    attempts_per_pair: int = 1000,
    max_transitions_per_edge: int = 3,
    rrt_step: float = 0.06,
    rrt_iters: int = 6000,
    rrt_time_budget_sec: float = 2.0,
    goal_bias: float = 0.35,
    verbose: bool = False,
) -> dict[str, Any]:
    total_t0 = time.perf_counter()

    z_limits = get_planning_height_limits(table_height)
    planning_max_z = z_limits["max_task_z"]

    modes, ambient_bounds, meta = make_two_tables_problem_3d(
        L=L,
        W=W,
        G=G,
        table_height=table_height,
        z_max=planning_max_z,
        object_half_length=0.10,
        object_radius=0.04,
        carry_clearance=0.03,
        support_side_clearance=0.01,
    )

    extra_node_cost = _make_mode_preference_costs(modes)

    s_mode_idx, x_start_proj = pick_mode_for_state(x_start, modes)
    g_mode_idx, x_goal_proj = pick_mode_for_state(x_goal, modes)

    t_plan_0 = time.perf_counter()
    path, dbg = plan_multimodal_v2(
        x_start,
        x_goal,
        modes=modes,
        ambient_bounds=ambient_bounds,
        meta=meta,
        attempts_per_pair=attempts_per_pair,
        max_transitions_per_edge=max_transitions_per_edge,
        base_switch_cost=1.0,
        extra_node_cost=extra_node_cost,
        rrt_step=rrt_step,
        rrt_iters=rrt_iters,
        rrt_time_budget_sec=rrt_time_budget_sec,
        goal_bias=goal_bias,
        forbid_direct_in_modes=["FreeTransferWorkspace"],
        transition_pick_policy="closest_on_src",
        seed=planner_seed,
    )
    planner_time_sec = time.perf_counter() - t_plan_0

    full_planned_path = path.copy()
    ik_targets_full = _build_ik_targets(full_planned_path, table_height, planning_max_z)
    sampled_exec_path, sampled_ik_targets, sampled_idx = _sample_path_keep_ends(
        full_planned_path, ik_targets_full, stride=stride
    )

    q_home = np.asarray(get_default_q_home(), dtype=float)
    base_pose = get_default_robot_base_pose()
    urdf_path = get_default_urdf_path()

    ik = IiwaIK(
        urdf_path=str(urdf_path),
        ee_link_name=get_default_ee_link_name(),
        base_xyz=base_pose["xyz"],
        base_yaw_deg=base_pose["yaw_deg"],
    )

    collision_boxes = _make_table_collision_boxes(meta)

    t_ik_0 = time.perf_counter()
    q_list = []
    oks = []
    errs = []

    pos_tol = 0.04
    collision_margin = 0.05
    min_ee_z = table_height + 0.10

    q_prev = q_home.copy()
    for target in sampled_ik_targets:
        q_sol, ok, err = ik.solve_position_ik(
            target,
            q_prev,
            pos_tol=pos_tol,
            collision_boxes=collision_boxes,
            collision_margin=collision_margin,
            min_ee_z=min_ee_z,
            n_restarts=3,
            smooth_weight=0.08,
        )
        q_list.append(q_sol.copy())
        oks.append(ok)
        errs.append(err)
        q_prev = q_sol

    ik_time_sec = time.perf_counter() - t_ik_0

    q_waypoints = np.asarray(q_list, dtype=float)
    oks = np.asarray(oks, dtype=bool)
    errs = np.asarray(errs, dtype=float)

    s, e, chunk_policy = _pick_execution_chunk(oks, min_len=2)

    planner_success = True
    ik_success_count = int(np.sum(oks))
    ik_total_count = len(oks)

    executed_start = None
    executed_goal = None
    start_error = None
    goal_error = None

    if e - s >= 2:
        executed_path = sampled_exec_path[s:e]
        executed_start = executed_path[0]
        executed_goal = executed_path[-1]
        start_error = float(np.linalg.norm(executed_start - full_planned_path[0]))
        goal_error = float(np.linalg.norm(executed_goal - full_planned_path[-1]))
    else:
        planner_success = False

    total_prep_time_sec = time.perf_counter() - total_t0

    result = {
        "planner_success": planner_success,
        "planner_seed": planner_seed,
        "start_x": float(x_start[0]),
        "start_y": float(x_start[1]),
        "start_z": float(x_start[2]),
        "goal_x": float(x_goal[0]),
        "goal_y": float(x_goal[1]),
        "goal_z": float(x_goal[2]),
        "projected_start_x": float(x_start_proj[0]),
        "projected_start_y": float(x_start_proj[1]),
        "projected_start_z": float(x_start_proj[2]),
        "projected_goal_x": float(x_goal_proj[0]),
        "projected_goal_y": float(x_goal_proj[1]),
        "projected_goal_z": float(x_goal_proj[2]),
        "start_mode": modes[s_mode_idx].name,
        "goal_mode": modes[g_mode_idx].name,
        "mode_sequence": " -> ".join([modes[i].name for i in dbg.mode_sequence]),
        "planner_points": int(len(full_planned_path)),
        "planner_time_sec": float(planner_time_sec),
        "ik_time_sec": float(ik_time_sec),
        "total_prep_time_sec": float(total_prep_time_sec),
        "ik_success_count": int(ik_success_count),
        "ik_total_count": int(ik_total_count),
        "ik_success_ratio": float(ik_success_count / max(1, ik_total_count)),
        "chunk_policy": chunk_policy,
        "chunk_start_idx": int(s),
        "chunk_end_idx": int(e - 1),
        "chunk_len": int(max(0, e - s)),
        "mean_ik_err": float(np.mean(errs)) if len(errs) else None,
        "max_ik_err": float(np.max(errs)) if len(errs) else None,
        "executed_start_x": None if executed_start is None else float(executed_start[0]),
        "executed_start_y": None if executed_start is None else float(executed_start[1]),
        "executed_start_z": None if executed_start is None else float(executed_start[2]),
        "executed_goal_x": None if executed_goal is None else float(executed_goal[0]),
        "executed_goal_y": None if executed_goal is None else float(executed_goal[1]),
        "executed_goal_z": None if executed_goal is None else float(executed_goal[2]),
        "start_error": start_error,
        "goal_error": goal_error,
        "planning_max_z": float(planning_max_z),
        "raw_start_x": float(full_planned_path[0][0]),
        "raw_start_y": float(full_planned_path[0][1]),
        "raw_start_z": float(full_planned_path[0][2]),
        "raw_goal_x": float(full_planned_path[-1][0]),
        "raw_goal_y": float(full_planned_path[-1][1]),
        "raw_goal_z": float(full_planned_path[-1][2]),
        "sampled_last_idx": int(sampled_idx[-1]),
    }

    if verbose:
        for k, v in result.items():
            print(f"{k}: {v}")

    return result


def demo_run_and_visualize_robot():
    planner_seed = 7
    L = 0.45
    W = 0.40
    G = 0.30
    table_height = 0.70

    x_start = np.array([0.35, 0.00, 0.70], dtype=float)
    x_goal  = np.array([1.05, 0.00, 0.70], dtype=float)

    result = evaluate_single_case(
        x_start,
        x_goal,
        planner_seed=planner_seed,
        L=L,
        W=W,
        G=G,
        table_height=table_height,
        verbose=True,
    )

    # Re-run once for visualization data path
    z_limits = get_planning_height_limits(table_height)
    planning_max_z = z_limits["max_task_z"]

    modes, ambient_bounds, meta = make_two_tables_problem_3d(
        L=L,
        W=W,
        G=G,
        table_height=table_height,
        z_max=planning_max_z,
        object_half_length=0.10,
        object_radius=0.04,
        carry_clearance=0.03,
        support_side_clearance=0.01,
    )
    extra_node_cost = _make_mode_preference_costs(modes)

    path, dbg = plan_multimodal_v2(
        x_start,
        x_goal,
        modes=modes,
        ambient_bounds=ambient_bounds,
        meta=meta,
        attempts_per_pair=1000,
        max_transitions_per_edge=3,
        base_switch_cost=1.0,
        extra_node_cost=extra_node_cost,
        rrt_step=0.06,
        rrt_iters=6000,
        rrt_time_budget_sec=2.0,
        goal_bias=0.35,
        forbid_direct_in_modes=["FreeTransferWorkspace"],
        transition_pick_policy="closest_on_src",
        seed=planner_seed,
    )

    full_planned_path = path.copy()
    ik_targets_full = _build_ik_targets(full_planned_path, table_height, planning_max_z)
    sampled_exec_path, sampled_ik_targets, _ = _sample_path_keep_ends(full_planned_path, ik_targets_full, stride=2)

    q_home = np.asarray(get_default_q_home(), dtype=float)
    base_pose = get_default_robot_base_pose()
    urdf_path = get_default_urdf_path()

    ik = IiwaIK(
        urdf_path=str(urdf_path),
        ee_link_name=get_default_ee_link_name(),
        base_xyz=base_pose["xyz"],
        base_yaw_deg=base_pose["yaw_deg"],
    )
    collision_boxes = _make_table_collision_boxes(meta)

    q_list = []
    oks = []
    q_prev = q_home.copy()
    for target in sampled_ik_targets:
        q_sol, ok, _ = ik.solve_position_ik(
            target,
            q_prev,
            pos_tol=0.04,
            collision_boxes=collision_boxes,
            collision_margin=0.05,
            min_ee_z=table_height + 0.10,
            n_restarts=3,
            smooth_weight=0.08,
        )
        q_list.append(q_sol.copy())
        oks.append(ok)
        q_prev = q_sol

    q_waypoints = np.asarray(q_list, dtype=float)
    oks = np.asarray(oks, dtype=bool)
    s, e, _ = _pick_execution_chunk(oks, min_len=2)

    if e - s < 2:
        raise RuntimeError("No sufficiently long collision-free IK chunk found for visualization.")

    q_waypoints_ok = q_waypoints[s:e]
    executed_path = sampled_exec_path[s:e]

    frames_per_segment = 6
    q_traj = []
    executed_anim_path = []

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

    visualize_two_tables_with_robot(
        path=executed_anim_path,
        meta=meta,
        urdf_path=urdf_path,
        q_home=q_home,
        q_traj=q_traj,
        robot_base_xyz=base_pose["xyz"],
        robot_base_yaw_deg=base_pose["yaw_deg"],
        show_points=False,
        animate_payload=False,
        full_path=full_planned_path,
        executed_path=executed_path,
        title=f"iiwa execution | support preferred | seed {planner_seed}",
        frame_dt=0.06,
    )