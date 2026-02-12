import numpy as np
from .rrt import norm


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
    """
    Sample xT intended to lie in intersection M_A ∩ M_B using alternating projections.
    Requires passing both modes' validity checks.
    """
    for _ in range(attempts):
        z = np.array([np.random.uniform(lo, hi) for lo, hi in ambient_bounds], dtype=float)

        xa = modeA.project(z)
        if xa is None:
            continue
        xa = np.asarray(xa, dtype=float)
        if not modeA.is_valid(xa):
            continue

        xb = modeB.project(xa)
        if xb is None:
            continue
        xb = np.asarray(xb, dtype=float)
        if not modeB.is_valid(xb):
            continue

        xT = modeA.project(xb)
        if xT is None:
            continue
        xT = np.asarray(xT, dtype=float)
        if not modeA.is_valid(xT):
            continue

        # check also valid in B after projection
        xb2 = modeB.project(xT)
        if xb2 is None:
            continue
        xb2 = np.asarray(xb2, dtype=float)
        if not modeB.is_valid(xb2):
            continue

        # refine
        xTT, ok2 = project_intersection(modeA, modeB, xT, max_iter=30)
        if ok2 and modeA.is_valid(xTT) and modeB.is_valid(xTT):
            return np.asarray(xTT, dtype=float)

    return None


def sample_k_transitions(modeA, modeB, ambient_bounds, k=3, attempts=12000):
    """
    Collect up to k distinct transition samples (useful later for robustness).
    """
    Ts = []
    seen = set()
    for _ in range(attempts):
        xT = sample_transition(modeA, modeB, ambient_bounds, attempts=200)
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