"""
  1) Assign start and goal to valid modes (projection + validity).
  2) Build a ModeGraph by trying to sample intersection "doors" xT ∈ M_A ∩ M_B
     for all mode pairs (A -> B). Only successful pairs become graph edges.
  3) Use Dijkstra on the mode graph to choose the cheapest mode sequence from
     start_mode to goal_mode (costs encode preferences like "carry is expensive").
  4) For each chosen mode transition A -> B:
       - pick a cached door point (by default: closest-to-current in mode A),
       - plan inside mode A using constrained RRT to reach that door,
       - switch to mode B at the door (projection into B),
     and append the segment to the full path.
  5) Plan the final segment inside the goal mode to reach the projected goal.
  6) Return the concatenated path plus debug info (chosen modes, doors used, segment stats).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import random

from .mode_graph import ModeGraph
from .discrete_search import dijkstra_mode_sequence
from .rrt import rrt_connect_on_mode, direct_connect_on_mode


@dataclass
class SegmentDebug:
    mode_name: str
    start: np.ndarray
    goal: np.ndarray
    success: bool
    n_points: int


@dataclass
class PlanDebug:
    start_mode: int
    goal_mode: int
    mode_sequence: List[int]
    transitions_used: List[np.ndarray]
    segments: List[SegmentDebug]


def _mode_family_priority(mode) -> int:
    """
    Lower is preferred when projection scores are equal.

    This enforces the intended semantic bias:
      SupportContact > FreeTransfer > SupportTransition
    """
    family = getattr(mode, "family", "")
    if family == "SupportContact":
        return 0
    if family == "FreeTransfer":
        return 1
    if family == "SupportTransition":
        return 2
    return 10


def pick_mode_for_state(x: np.ndarray, modes) -> Tuple[int, np.ndarray]:
    x = np.asarray(x, dtype=float)
    best = None
    best_score = float("inf")
    best_prio = float("inf")

    for i, m in enumerate(modes):
        xp = np.asarray(m.project(x), dtype=float)
        if not np.all(np.isfinite(xp)):
            continue
        if not m.is_valid(xp):
            continue

        score = float(np.linalg.norm(xp - x))
        prio = _mode_family_priority(m)

        better = False
        if score < best_score - 1e-12:
            better = True
        elif abs(score - best_score) <= 1e-12 and prio < best_prio:
            better = True

        if better:
            best_score = score
            best_prio = prio
            best = (i, xp)

    if best is None:
        raise RuntimeError("Could not assign state to any mode (no valid projection).")
    return best


def plan_multimodal_v2(
    x_start: np.ndarray,
    x_goal: np.ndarray,
    *,
    modes,
    ambient_bounds,
    meta: Dict,
    attempts_per_pair: int = 4000,
    max_transitions_per_edge: int = 1,
    base_switch_cost: float = 1.0,
    extra_node_cost: Optional[Dict[int, float]] = None,
    rrt_step: float = 0.12,
    rrt_iters: int = 9000,
    rrt_time_budget_sec: float = 4.0,
    goal_bias: float = 0.30,
    allow_direct_fallback: bool = True,
    forbid_direct_in_modes: Optional[List[str]] = None,
    transition_pick_policy: str = "closest_on_src",
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, PlanDebug]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    forbid_direct_in_modes = forbid_direct_in_modes or ["CarryFree"]

    # 1) start/goal mode
    s_mode, x_s = pick_mode_for_state(x_start, modes)
    g_mode, x_g = pick_mode_for_state(x_goal, modes)

    # 2) build graph
    graph = ModeGraph(modes, ambient_bounds)
    graph.build(
        attempts_per_pair=attempts_per_pair,
        max_transitions_per_edge=max_transitions_per_edge,
        base_switch_cost=base_switch_cost,
        use_all_pairs=True,
        verbose=False,
    )

    # 3) discrete search (mode sequence)
    seq = dijkstra_mode_sequence(graph, s_mode, g_mode, extra_node_cost=extra_node_cost)

    # 4) continuous planning along seq
    table_height = float(meta.get("table_height", 0.75))

    x_cur = np.asarray(x_s, dtype=float)
    full: List[np.ndarray] = [x_cur.copy()]

    transitions_used: List[np.ndarray] = []
    segments_dbg: List[SegmentDebug] = []

    for k in range(len(seq) - 1):
        i = seq[k]
        j = seq[k + 1]
        A = modes[i]
        B = modes[j]

        xT = graph.get_best_transition_for_state(i, j, x_cur, policy=transition_pick_policy)
        if xT is None:
            raise RuntimeError(f"No cached transition for edge {A.name}->{B.name} (graph edge missing).")

        transitions_used.append(np.asarray(xT, dtype=float))
        x_target = np.asarray(A.project(xT), dtype=float)

        seg = rrt_connect_on_mode(
            x_cur,
            x_target,
            A,
            iters=rrt_iters,
            step=rrt_step,
            goal_bias=goal_bias,
            table_height_for_cost=table_height,
            max_nodes=25000,
            time_budget_sec=rrt_time_budget_sec,
            stop_after_first=True,
        )

        if seg is None and allow_direct_fallback and (A.name not in forbid_direct_in_modes):
            seg = direct_connect_on_mode(x_cur, x_target, A, step=min(rrt_step, 0.06))

        if seg is None:
            segments_dbg.append(
                SegmentDebug(A.name, x_cur.copy(), x_target.copy(), False, 0)
            )
            raise RuntimeError(f"Failed to plan within mode {A.name} to reach transition to {B.name}.")

        seg_np = [np.asarray(p, dtype=float) for p in seg]
        segments_dbg.append(
            SegmentDebug(A.name, x_cur.copy(), x_target.copy(), True, len(seg_np))
        )

        full.extend(seg_np[1:])
        x_cur = np.asarray(B.project(full[-1]), dtype=float)

    # 5) final segment in goal mode to x_g
    GoalMode = modes[g_mode]
    seg_last = rrt_connect_on_mode(
        x_cur,
        x_g,
        GoalMode,
        iters=rrt_iters,
        step=rrt_step,
        goal_bias=max(goal_bias, 0.35),
        table_height_for_cost=table_height,
        max_nodes=30000,
        time_budget_sec=rrt_time_budget_sec,
        stop_after_first=True,
    )

    if seg_last is None and allow_direct_fallback and (GoalMode.name not in forbid_direct_in_modes):
        seg_last = direct_connect_on_mode(x_cur, x_g, GoalMode, step=min(rrt_step, 0.06))

    if seg_last is None:
        segments_dbg.append(SegmentDebug(GoalMode.name, x_cur.copy(), x_g.copy(), False, 0))
        raise RuntimeError(f"Failed to plan final segment in goal mode {GoalMode.name}.")

    seg_last_np = [np.asarray(p, dtype=float) for p in seg_last]
    segments_dbg.append(SegmentDebug(GoalMode.name, x_cur.copy(), x_g.copy(), True, len(seg_last_np)))
    full.extend(seg_last_np[1:])

    dbg = PlanDebug(
        start_mode=s_mode,
        goal_mode=g_mode,
        mode_sequence=seq,
        transitions_used=transitions_used,
        segments=segments_dbg,
    )
    return np.asarray(full, dtype=float), dbg