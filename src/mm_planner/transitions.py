from __future__ import annotations

import numpy as np


def _norm(x):
    return float(np.linalg.norm(np.asarray(x, dtype=float)))


def _is_implicit(mode) -> bool:
    return hasattr(mode, "h_func") and hasattr(mode, "J_func")


def _clip(x, bounds):
    y = np.asarray(x, dtype=float).copy()
    for i, (lo, hi) in enumerate(bounds):
        y[i] = np.clip(y[i], lo, hi)
    return y


def _stacked_project_intersection_implicit(
    modeA,
    modeB,
    x0,
    *,
    max_iters=80,
    tol=1e-6,
    damping=1e-3,
    backtracking=True,
    min_alpha=1e-3,
):
    """
    Solve intersection by stacked equalities:
      h_AB(x) = [hA(x); hB(x)] = 0

    using damped least-squares Gauss-Newton.
    """
    x = np.asarray(x0, dtype=float).copy()
    bounds = getattr(modeA, "ambient_bounds", None)
    if bounds is not None:
        x = _clip(x, bounds)

    def resnorm(xx):
        hA = np.asarray(modeA.h_func(xx), dtype=float).reshape(-1)
        hB = np.asarray(modeB.h_func(xx), dtype=float).reshape(-1)
        return float(np.linalg.norm(np.concatenate([hA, hB], axis=0)))

    for _ in range(max_iters):
        hA = np.asarray(modeA.h_func(x), dtype=float).reshape(-1)
        hB = np.asarray(modeB.h_func(x), dtype=float).reshape(-1)
        r = np.concatenate([hA, hB], axis=0)
        nr = float(np.linalg.norm(r))
        if nr < tol:
            return x, True

        JA = np.asarray(modeA.J_func(x), dtype=float)
        JB = np.asarray(modeB.J_func(x), dtype=float)
        J = np.vstack([JA, JB])

        m = J.shape[0]
        A = J @ J.T + (damping ** 2) * np.eye(m)

        try:
            y = np.linalg.solve(A, r)
        except np.linalg.LinAlgError:
            return x, False

        dx = -(J.T @ y)

        alpha = 1.0
        x_new = x + alpha * dx
        if bounds is not None:
            x_new = _clip(x_new, bounds)

        if backtracking:
            base = nr
            nn = resnorm(x_new)
            while nn > base and alpha > min_alpha:
                alpha *= 0.5
                x_new = x + alpha * dx
                if bounds is not None:
                    x_new = _clip(x_new, bounds)
                nn = resnorm(x_new)

        x = x_new

        if not np.all(np.isfinite(x)):
            return x, False

    return x, resnorm(x) < 10 * tol


def project_intersection(modeA, modeB, z0, max_iter=50, tol=1e-6):
    """
    Alternating projections:
      x <- P_A(x), then x <- P_B(x)

    Works for explicit/implicit mixtures because both expose mode.project(x).
    """
    x = np.asarray(z0, dtype=float).copy()

    for _ in range(max_iter):
        x_prev = x.copy()
        x = np.asarray(modeA.project(x), dtype=float)
        x = np.asarray(modeB.project(x), dtype=float)

        if _norm(x - x_prev) < tol:
            xa = np.asarray(modeA.project(x), dtype=float)
            xb = np.asarray(modeB.project(x), dtype=float)
            ok = (_norm(x - xa) < 1e-6) and (_norm(x - xb) < 1e-6)
            return x, ok

    xa = np.asarray(modeA.project(x), dtype=float)
    xb = np.asarray(modeB.project(x), dtype=float)
    ok = (_norm(x - xa) < 1e-6) and (_norm(x - xb) < 1e-6)
    return x, ok


def sample_transition(modeA, modeB, ambient_bounds, attempts=5000):
    """
    Sample a transition ("door") xT between two modes.

    Strategy:
      - sample ambient
      - build a few candidate initializations
      - if both modes are implicit, solve the stacked intersection
      - otherwise use alternating projections
      - require final validity in both modes
    """
    implicit_pair = _is_implicit(modeA) and _is_implicit(modeB)

    for _ in range(attempts):
        z = np.array([np.random.uniform(lo, hi) for lo, hi in ambient_bounds], dtype=float)

        seeds = []
        try:
            seeds.append(np.asarray(modeA.project(z), dtype=float))
        except Exception:
            pass
        try:
            seeds.append(np.asarray(modeB.project(z), dtype=float))
        except Exception:
            pass
        if len(seeds) >= 2:
            seeds.append(0.5 * (seeds[0] + seeds[1]))
        seeds.append(z)

        for x0 in seeds:
            if not np.all(np.isfinite(x0)):
                continue

            if implicit_pair:
                xT, ok = _stacked_project_intersection_implicit(
                    modeA,
                    modeB,
                    x0,
                    max_iters=80,
                    tol=1e-6,
                    damping=1e-3,
                    backtracking=True,
                )
            else:
                xT, ok = project_intersection(modeA, modeB, x0, max_iter=40, tol=1e-6)

            if not ok:
                continue

            xT = np.asarray(xT, dtype=float)
            xA = np.asarray(modeA.project(xT), dtype=float)
            xB = np.asarray(modeB.project(xT), dtype=float)

            if modeA.is_valid(xA) and modeB.is_valid(xB):
                return xT

    return None


def sample_k_transitions(modeA, modeB, ambient_bounds, k=3, attempts=12000):
    Ts = []
    seen = set()

    for _ in range(attempts):
        xT = sample_transition(modeA, modeB, ambient_bounds, attempts=250)
        if xT is None:
            continue

        xr = tuple(np.round(np.asarray(xT, dtype=float), 3))
        if xr in seen:
            continue

        seen.add(xr)
        Ts.append(np.asarray(xT, dtype=float))

        if len(Ts) >= k:
            break

    return Ts