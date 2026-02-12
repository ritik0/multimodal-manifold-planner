from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

from .transitions import sample_transition


@dataclass
class Edge:
    src: int
    dst: int
    base_cost: float
    # cached intersection samples xT ∈ M_src ∩ M_dst
    transitions: List[np.ndarray]


class ModeGraph:
    """
    Directed graph over modes.
    Edge i->j exists if we can sample at least one intersection transition xT ∈ M_i ∩ M_j.
    """

    def __init__(self, modes, ambient_bounds):
        self.modes = modes
        self.ambient_bounds = ambient_bounds

        # adjacency list: src -> list of Edge
        self.adj: Dict[int, List[Edge]] = {}

        # transition cache: (i,j) -> [xT1, xT2, ...]
        self._T: Dict[Tuple[int, int], List[np.ndarray]] = {}

    def add_transition(self, i: int, j: int, xT: np.ndarray, base_cost: float):
        key = (i, j)
        xT = np.asarray(xT, dtype=float)

        if key not in self._T:
            self._T[key] = []
        self._T[key].append(xT)

        # rebuild / append edge in adjacency
        if i not in self.adj:
            self.adj[i] = []

        # see if edge exists already
        for e in self.adj[i]:
            if e.dst == j:
                e.transitions.append(xT)
                e.base_cost = min(e.base_cost, float(base_cost))
                return

        self.adj[i].append(Edge(src=i, dst=j, base_cost=float(base_cost), transitions=[xT]))

    def get_transitions(self, i: int, j: int) -> List[np.ndarray]:
        return list(self._T.get((i, j), []))

    def get_best_transition_for_state(
        self,
        i: int,
        j: int,
        x_cur: np.ndarray,
        policy: str = "closest_on_src",
    ) -> Optional[np.ndarray]:
        """
        Choose one cached transition sample for edge i->j.
        policy:
          - "first": first cached
          - "closest_on_src": choose xT that is closest to x_cur after projecting to src mode
        """
        Ts = self.get_transitions(i, j)
        if not Ts:
            return None

        if policy == "first":
            return Ts[0]

        if policy == "closest_on_src":
            A = self.modes[i]
            x_cur = np.asarray(x_cur, dtype=float)
            best = None
            best_d = float("inf")
            for xT in Ts:
                xA = np.asarray(A.project(xT), dtype=float)
                d = float(np.linalg.norm(xA - x_cur))
                if d < best_d:
                    best_d = d
                    best = xT
            return best

        raise ValueError(f"Unknown transition selection policy: {policy}")

    def build(
        self,
        *,
        attempts_per_pair: int = 4000,
        max_transitions_per_edge: int = 1,
        use_all_pairs: bool = True,
        candidate_pairs: Optional[List[Tuple[int, int]]] = None,
        base_switch_cost: float = 1.0,
        verbose: bool = True,
    ):
        """
        Build edges by sampling intersections.

        If use_all_pairs=True: tries all ordered pairs (i,j).
        If candidate_pairs is provided: only tries those pairs.

        Edge cost: base_switch_cost + dst_mode_penalty (from modes[j].cost_weight["mode_penalty"] if present)
        """
        n = len(self.modes)

        if not use_all_pairs and candidate_pairs is None:
            raise ValueError("Either use_all_pairs=True or provide candidate_pairs.")

        pairs = []
        if use_all_pairs:
            for i in range(n):
                for j in range(n):
                    if i != j:
                        pairs.append((i, j))
        else:
            pairs = list(candidate_pairs)

        for (i, j) in pairs:
            A = self.modes[i]
            B = self.modes[j]

            # compute base cost using mode penalties (keeps “carry expensive” etc.)
            dst_pen = 0.0
            if hasattr(B, "cost_weight") and isinstance(B.cost_weight, dict):
                dst_pen = float(B.cost_weight.get("mode_penalty", 0.0))
            base_cost = float(base_switch_cost + dst_pen)

            # sample up to k transitions for this edge
            found = 0
            for _ in range(max_transitions_per_edge):
                xT = sample_transition(A, B, self.ambient_bounds, attempts=attempts_per_pair)
                if xT is None:
                    break
                # require validity in both modes (important)
                xT = np.asarray(xT, dtype=float)
                if A.is_valid(A.project(xT)) and B.is_valid(B.project(xT)):
                    self.add_transition(i, j, xT, base_cost=base_cost)
                    found += 1

            if verbose and found > 0:
                print(f"[graph] edge {A.name} ({i}) -> {B.name} ({j}) : {found} transition(s)")

        # ensure every node appears in adj
        for i in range(n):
            self.adj.setdefault(i, [])