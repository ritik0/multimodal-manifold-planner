import time
import numpy as np


def norm(x) -> float:
    return float(np.linalg.norm(x))


def steer(x_from: np.ndarray, x_to: np.ndarray, step: float) -> np.ndarray:
    v = x_to - x_from
    d = norm(v)
    if d <= step:
        return x_to.copy()
    return x_from + (step / d) * v


def build_path_from_arrays(X, parent_idx, goal_i: int):
    path = []
    i = goal_i
    while i != -1:
        path.append(X[i])
        i = parent_idx[i]
    return list(reversed(path))


def edge_cost(x_from: np.ndarray, x_to: np.ndarray, mode, *, table_height: float = 0.0) -> float:
    length = float(np.linalg.norm(x_to - x_from))
    z_penalty_w = float(mode.cost_weight.get("z_penalty", 0.0))
    mode_penalty = float(mode.cost_weight.get("mode_penalty", 0.0))

    z_above = max(0.0, float(x_to[2] - table_height))
    z_pen = z_penalty_w * z_above
    return length + z_pen + mode_penalty


def nearest_index(X: list[np.ndarray], x: np.ndarray) -> int:
    """Vectorized nearest neighbor search."""
    A = np.asarray(X, dtype=float)              # (N, d)
    x = np.asarray(x, dtype=float).reshape(1, -1)
    d2 = np.sum((A - x) ** 2, axis=1)           # (N,)
    return int(np.argmin(d2))


def constrained_extend(
    X, parent_idx, cost_arr,
    x_target: np.ndarray,
    mode,
    *,
    step: float,
    max_steps: int,
    stuck_tol: float,
    table_height_for_cost: float,
    max_nodes: int,
):
    """
    Extend toward x_target with projection, validity, and cost accumulation.
    Returns index of last added node or None.
    """
    if len(X) >= max_nodes:
        return None

    i_near = nearest_index(X, x_target)
    x = X[i_near].copy()
    i_parent = i_near

    for _ in range(max_steps):
        x_next = steer(x, x_target, step)
        x_proj = mode.project(x_next)
        if x_proj is None:
            return None
        x_proj = np.asarray(x_proj, dtype=float)

        if not np.all(np.isfinite(x_proj)):
            return None

        if float(np.linalg.norm(x_proj - x)) < stuck_tol:
            return None

        if not mode.is_valid(x_proj):
            return None

        new_cost = cost_arr[i_parent] + edge_cost(X[i_parent], x_proj, mode, table_height=table_height_for_cost)

        X.append(x_proj)
        parent_idx.append(i_parent)
        cost_arr.append(float(new_cost))

        i_parent = len(X) - 1
        x = x_proj

        if len(X) >= max_nodes:
            return i_parent

        if float(np.linalg.norm(x - x_target)) <= step:
            return i_parent

    return i_parent


def direct_connect_on_mode(x_start, x_goal, mode, *, step=0.05, max_steps=2000, tol=1e-10):
    xs = np.asarray(mode.project(x_start), dtype=float)
    xg = np.asarray(mode.project(x_goal), dtype=float)
    if not mode.is_valid(xs) or not mode.is_valid(xg):
        return None

    path = [xs.copy()]
    x = xs.copy()
    local_step = float(step)

    for _ in range(max_steps):
        if float(np.linalg.norm(x - xg)) <= local_step:
            path.append(xg.copy())
            return path

        x_next = steer(x, xg, local_step)
        x_proj = np.asarray(mode.project(x_next), dtype=float)

        if not np.all(np.isfinite(x_proj)) or not mode.is_valid(x_proj):
            return None

        if float(np.linalg.norm(x_proj - x)) < tol:
            local_step *= 0.5
            if local_step < 1e-4:
                return None
            continue

        path.append(x_proj)
        x = x_proj

    return None


def rrt_connect_on_mode(
    x_start, x_goal, mode,
    *,
    iters=6000,
    step=0.12,
    goal_radius=None,
    goal_bias=0.35,
    extend_max_steps=80,
    stuck_tol=1e-9,
    table_height_for_cost=0.75,
    max_nodes=15000,
    time_budget_sec=3.0,          # <--- makes it “finish” fast
    stop_after_first=True,        # <--- set False if you want best-cost
):
    """
    Constrained RRT:
    - vectorized nearest search
    - optional time budget to prevent long runs
    - optional stop_after_first for interactive demos
    """
    t0 = time.perf_counter()

    xs = np.asarray(mode.project(x_start), dtype=float)
    xg = np.asarray(mode.project(x_goal), dtype=float)
    if not mode.is_valid(xs) or not mode.is_valid(xg):
        return None

    if goal_radius is None:
        goal_radius = max(2.0 * step, 0.15)

    X = [xs]
    parent_idx = [-1]
    cost_arr = [0.0]

    best_goal_i = None
    best_cost = float("inf")

    for k in range(iters):
        if (time.perf_counter() - t0) > time_budget_sec:
            break

        x_rand = xg if np.random.rand() < goal_bias else mode.sample_ambient()

        i_new = constrained_extend(
            X, parent_idx, cost_arr,
            x_rand, mode,
            step=step,
            max_steps=extend_max_steps,
            stuck_tol=stuck_tol,
            table_height_for_cost=table_height_for_cost,
            max_nodes=max_nodes,
        )
        if i_new is None:
            continue

        # if close to goal, try “connect”
        if float(np.linalg.norm(X[i_new] - xg)) <= goal_radius:
            # attempt a few steps directly toward goal
            i_final = constrained_extend(
                X, parent_idx, cost_arr,
                xg, mode,
                step=step,
                max_steps=extend_max_steps * 2,
                stuck_tol=stuck_tol,
                table_height_for_cost=table_height_for_cost,
                max_nodes=max_nodes,
            )
            cand_i = i_final if i_final is not None else i_new
            if float(np.linalg.norm(X[cand_i] - xg)) <= goal_radius:
                if stop_after_first:
                    return build_path_from_arrays(X, parent_idx, cand_i)
                if cost_arr[cand_i] < best_cost:
                    best_cost = cost_arr[cand_i]
                    best_goal_i = cand_i

    if best_goal_i is None:
        return None
    return build_path_from_arrays(X, parent_idx, best_goal_i)


def can_switch(x_on_A, modeA, modeB, *, eps_back=1e-6):
    xA = np.asarray(modeA.project(x_on_A), dtype=float)
    xB = np.asarray(modeB.project(xA), dtype=float)

    if not modeB.is_valid(xB):
        return False, xB

    xA_back = np.asarray(modeA.project(xB), dtype=float)
    if float(np.linalg.norm(xA_back - xA)) > eps_back:
        return False, xB

    return True, xB


def rrt_reach_switch(
    x_start, x_goal_hint, modeA, modeB,
    *,
    iters=4000,
    step=0.12,
    goal_bias=0.35,
    extend_max_steps=80,
    stuck_tol=1e-9,
    table_height_for_cost=0.75,
    eps_back=1e-6,
    max_nodes=12000,
    time_budget_sec=2.0,         # <--- prevents “stuck feeling”
):
    """
    Grow tree on modeA; return path to a node from which we can switch into modeB.
    Uses time budget to keep interactive runs fast.
    """
    t0 = time.perf_counter()

    xs = np.asarray(modeA.project(x_start), dtype=float)
    xgA = np.asarray(modeA.project(x_goal_hint), dtype=float)
    if not modeA.is_valid(xs):
        return None, None

    X = [xs]
    parent_idx = [-1]
    cost_arr = [0.0]

    best_i = None
    best_cost = float("inf")
    best_xB = None

    for k in range(iters):
        if (time.perf_counter() - t0) > time_budget_sec:
            break

        x_rand = xgA if np.random.rand() < goal_bias else modeA.sample_ambient()

        i_new = constrained_extend(
            X, parent_idx, cost_arr,
            x_rand, modeA,
            step=step,
            max_steps=extend_max_steps,
            stuck_tol=stuck_tol,
            table_height_for_cost=table_height_for_cost,
            max_nodes=max_nodes,
        )
        if i_new is None:
            continue

        ok, xB = can_switch(X[i_new], modeA, modeB, eps_back=eps_back)
        if ok and cost_arr[i_new] < best_cost:
            best_cost = cost_arr[i_new]
            best_i = i_new
            best_xB = xB

    if best_i is None:
        return None, None

    return build_path_from_arrays(X, parent_idx, best_i), np.asarray(best_xB, dtype=float)