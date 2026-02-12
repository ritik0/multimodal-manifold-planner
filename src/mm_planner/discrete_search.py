from __future__ import annotations

import heapq
from typing import Dict, List, Tuple, Optional

from .mode_graph import ModeGraph


def dijkstra_mode_sequence(
    graph: ModeGraph,
    start_mode: int,
    goal_mode: int,
    *,
    extra_node_cost: Optional[Dict[int, float]] = None,
) -> List[int]:
    """
    Dijkstra over modes (nodes), using edge.base_cost.

    extra_node_cost: optional dict of per-node costs added when entering a node.
    Returns: list of mode indices [start, ..., goal].
    """
    if start_mode == goal_mode:
        return [start_mode]

    extra_node_cost = extra_node_cost or {}

    pq: List[Tuple[float, int]] = []
    heapq.heappush(pq, (0.0, start_mode))

    dist: Dict[int, float] = {start_mode: 0.0}
    parent: Dict[int, Optional[int]] = {start_mode: None}

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float("inf")):
            continue
        if u == goal_mode:
            break

        for e in graph.adj.get(u, []):
            v = e.dst
            w = float(e.base_cost) + float(extra_node_cost.get(v, 0.0))
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))

    if goal_mode not in parent:
        raise RuntimeError("No mode-sequence path found in graph from start to goal.")

    # reconstruct
    seq = []
    cur = goal_mode
    while cur is not None:
        seq.append(cur)
        cur = parent[cur]
    seq.reverse()
    return seq