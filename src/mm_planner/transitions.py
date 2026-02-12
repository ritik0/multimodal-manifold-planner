import numpy as np
from .rrt import norm, can_switch


def sample_switch_hint(modeA, modeB, ambient_bounds, *, attempts=8000, eps_back=1e-6):
    """
    Sample a point on modeA from which switching into modeB is feasible under can_switch().

    Returns:
      xA (on modeA) if a feasible switch exists, otherwise None.

    NOTE: This does NOT require intersection of constraint sets (important because
    Slide has z=ht while Lift/Carry has z>=ht+eps, i.e., disjoint).
    """
    for _ in range(attempts):
        z = np.array([np.random.uniform(lo, hi) for lo, hi in ambient_bounds], dtype=float)

        xA = np.asarray(modeA.project(z), dtype=float)
        if not np.all(np.isfinite(xA)):
            continue
        if not modeA.is_valid(xA):
            continue

        ok, _xB = can_switch(xA, modeA, modeB, eps_back=eps_back)
        if ok:
            return xA

    return None


# (Optional) keep your old functions if you want, but they won't work for disjoint sets
def project_intersection(modeA, modeB, z0, max_iter=50, tol=1e-6):
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
    # Kept for reference: requires intersection, will fail when sets are disjoint (Slide vs Lift w/ eps).
    for _ in range(attempts):
        z = np.array([np.random.uniform(lo, hi) for lo, hi in ambient_bounds], dtype=float)

        xa = modeA.project(z)
        xb = modeB.project(xa)
        xT = modeA.project(xb)

        xb2 = modeB.project(xT)
        ok = (norm(xT - modeA.project(xT)) < 1e-9) and (norm(xb2 - modeB.project(xb2)) < 1e-9)

        if ok:
            xTT, ok2 = project_intersection(modeA, modeB, xT, max_iter=30)
            if ok2:
                return xTT

    return None