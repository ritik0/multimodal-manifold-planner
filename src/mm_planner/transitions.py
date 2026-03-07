import numpy as np
from .rrt import norm


def _is_implicit(mode) -> bool:
    """Detect ImplicitMode-like interface (has h_func and J_func)."""
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
    Solve intersection by stacked constraints:
        h_AB(x) = [hA(x); hB(x)] = 0
    using damped least squares Gauss-Newton.
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

    ok = resnorm(x) < 10 * tol
    return x, ok


def project_intersection(modeA, modeB, z0, max_iter=50, tol=1e-6):
    """
    Alternating projections (ping-pong): x <- P_A(x), x <- P_B(x) until stable.
    Works for explicit and implicit modes because both provide mode.project(x).
    """
    x = np.array(z0, dtype=float).copy()
    for _ in range(max_iter):
        x_prev = x.copy()
        x = modeA.project(x)
        x = modeB.project(x)
        if norm(x - x_prev) < tol:
            xa = modeA.project(x)
            xb = modeB.project(x)
            ok = (norm(x - xa) < 1e-6) and (norm(x - xb) < 1e-6)
            return x, ok

    xa = modeA.project(x)
    xb = modeB.project(x)
    ok = (norm(x - xa) < 1e-6) and (norm(x - xb) < 1e-6)
    return x, ok


def sample_transition(modeA, modeB, ambient_bounds, attempts=5000):
    """
    Sample a transition ("door") xT between two modes A and B.

    - If both modes are implicit, solve intersection with stacked Jacobian GN:
        [hA(x); hB(x)] = 0
    - Otherwise fall back to alternating projections A->B->A (ping-pong).

    Returns a point xT if successful, else None.
    """
    implicit_pair = _is_implicit(modeA) and _is_implicit(modeB)

    for _ in range(attempts):
        z = np.array([np.random.uniform(lo, hi) for lo, hi in ambient_bounds], dtype=float)

        # Start from something valid on A
        xA0 = modeA.project(z)
        if xA0 is None:
            continue
        xA0 = np.asarray(xA0, dtype=float)
        if not modeA.is_valid(xA0):
            continue

        # Compute candidate intersection
        if implicit_pair:
            xT, ok = _stacked_project_intersection_implicit(
                modeA,
                modeB,
                xA0,
                max_iters=80,
                tol=1e-6,
                damping=1e-3,
                backtracking=True,
            )
            if not ok:
                continue
        else:
            xT, ok = project_intersection(modeA, modeB, xA0, max_iter=40, tol=1e-6)
            if not ok:
                continue

        xT = np.asarray(xT, dtype=float)

        # Final validity in both modes (use each mode's own projection)
        xA = np.asarray(modeA.project(xT), dtype=float)
        xB = np.asarray(modeB.project(xT), dtype=float)
        if modeA.is_valid(xA) and modeB.is_valid(xB):
            return xT

    return None


def sample_k_transitions(modeA, modeB, ambient_bounds, k=3, attempts=12000):
    """Collect up to k distinct transition samples between two modes."""
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