import numpy as np


def _clip_to_bounds(x: np.ndarray, ambient_bounds):
    """Clip each coordinate of x to ambient bounds."""
    y = np.asarray(x, dtype=float).copy()
    for i, (lo, hi) in enumerate(ambient_bounds):
        y[i] = np.clip(y[i], lo, hi)
    return y


class Mode:
    """
    Explicit mode:
    - project(x): explicit projector (clamp/snap)
    - is_valid(x): feasibility check (region + proxy; later torque/learned feasibility)
    - cost_weight: used for cost shaping / mode penalties
    """

    def __init__(self, name: str, projector, ambient_bounds, *, is_valid=None, cost_weight=None):
        self.name = name
        self.project = projector
        self.ambient_bounds = ambient_bounds
        self.is_valid = is_valid if is_valid is not None else (lambda x: True)
        self.cost_weight = cost_weight if cost_weight is not None else {}

    def sample_ambient(self) -> np.ndarray:
        return np.array([np.random.uniform(lo, hi) for lo, hi in self.ambient_bounds], dtype=float)


class ImplicitMode(Mode):
    """
    Implicit manifold mode defined by equality constraints h(x)=0 and Jacobian J(x).

    Projection uses damped least squares (Levenberg–Marquardt / Gauss-Newton):
        dx = - J^T (J J^T + λ^2 I)^-1 h
    with simple backtracking to avoid overshoot.

    IMPORTANT: We make sure `self.project` is callable by setting:
        self.project = self._project_impl
    so the rest of your pipeline can keep calling mode.project(x) without changes.
    """

    def __init__(
        self,
        name: str,
        h_func,
        J_func,
        ambient_bounds,
        *,
        is_valid=None,
        cost_weight=None,
        max_iters: int = 50,
        tol: float = 1e-6,
        damping: float = 1e-3,
        step_scale: float = 1.0,
        clip_each_iter: bool = True,
        backtracking: bool = True,
        min_alpha: float = 1e-3,
    ):
        # Pass a dummy callable projector to base class; we overwrite below.
        super().__init__(
            name=name,
            projector=lambda x: x,
            ambient_bounds=ambient_bounds,
            is_valid=is_valid,
            cost_weight=cost_weight,
        )

        self.h_func = h_func
        self.J_func = J_func

        self.max_iters = int(max_iters)
        self.tol = float(tol)
        self.damping = float(damping)
        self.step_scale = float(step_scale)
        self.clip_each_iter = bool(clip_each_iter)

        self.backtracking = bool(backtracking)
        self.min_alpha = float(min_alpha)

        # CRITICAL: guarantee Mode-like interface (callable .project)
        self.project = self._project_impl

    def _project_impl(self, x0: np.ndarray) -> np.ndarray:
        x = np.asarray(x0, dtype=float).copy()
        if self.clip_each_iter:
            x = _clip_to_bounds(x, self.ambient_bounds)

        def residual_norm(xx):
            r = np.asarray(self.h_func(xx), dtype=float).reshape(-1)
            return float(np.linalg.norm(r))

        for _ in range(self.max_iters):
            r = np.asarray(self.h_func(x), dtype=float).reshape(-1)

            # No equality constraints -> identity projection
            if r.size == 0:
                return _clip_to_bounds(x, self.ambient_bounds)

            nr = float(np.linalg.norm(r))
            if nr < self.tol:
                return _clip_to_bounds(x, self.ambient_bounds)

            J = np.asarray(self.J_func(x), dtype=float)  # shape (m, n)
            m = J.shape[0]

            # Damped normal solve in constraint space
            A = J @ J.T + (self.damping ** 2) * np.eye(m)
            try:
                y = np.linalg.solve(A, r)
            except np.linalg.LinAlgError:
                break

            dx = -(J.T @ y)

            # Backtracking line search (optional)
            alpha = self.step_scale
            x_new = x + alpha * dx
            if self.clip_each_iter:
                x_new = _clip_to_bounds(x_new, self.ambient_bounds)

            if self.backtracking:
                base = nr
                nn = residual_norm(x_new)
                while nn > base and alpha > self.min_alpha:
                    alpha *= 0.5
                    x_new = x + alpha * dx
                    if self.clip_each_iter:
                        x_new = _clip_to_bounds(x_new, self.ambient_bounds)
                    nn = residual_norm(x_new)

            x = x_new
            if not np.all(np.isfinite(x)):
                break

        # Best-effort return
        return _clip_to_bounds(x, self.ambient_bounds)


def make_two_tables_problem_3d(
    *,
    L=1.0,
    W=0.6,
    G=1.0,
    table_height=0.75,
    transition_width=0.2,
    z_max=2.0,
    lift_epsilon=0.02,
    # Feasibility proxy parameters (placeholder for torque/power feasibility)
    feasible_edge_margin=0.25,   # must be within last 25cm of table to lift/carry
    feasible_y_margin=None,      # optional: restrict y (e.g., near centerline)
):
    L = float(L)
    W = float(W)
    G = float(G)
    ht = float(table_height)
    w = float(transition_width)
    z_max = float(z_max)
    eps = float(lift_epsilon)

    x_min, x_max = 0.0, 2 * L + G
    y_min, y_max = -W / 2.0, W / 2.0
    ambient_bounds = [(x_min, x_max), (y_min, y_max), (0.0, z_max)]

    # ------------------------
    # Feasibility proxy
    # ------------------------
    def feasible_left(p):
        x, y, z = p
        if x < L - feasible_edge_margin:
            return False
        if feasible_y_margin is not None and abs(y) > feasible_y_margin:
            return False
        return True

    def feasible_right(p):
        x, y, z = p
        if x > (L + G + feasible_edge_margin):
            return False
        if feasible_y_margin is not None and abs(y) > feasible_y_margin:
            return False
        return True

    def feasible_lift(p):
        x, y, z = p
        return z >= ht

    def feasible_carry(p):
        x, y, z = p
        return z >= ht + eps

    # ------------------------
    # Implicit constraints (equalities) and Jacobians
    # ------------------------
    # Slide plane: z - ht = 0
    def h_slide(p):
        return np.array([p[2] - ht], dtype=float)

    def J_slide(p):
        return np.array([[0.0, 0.0, 1.0]], dtype=float)

    # Edge planes: x - L = 0 and x - (L+G) = 0
    def h_edge_left(p):
        return np.array([p[0] - L], dtype=float)

    def J_edge_left(p):
        return np.array([[1.0, 0.0, 0.0]], dtype=float)

    def h_edge_right(p):
        return np.array([p[0] - (L + G)], dtype=float)

    def J_edge_right(p):
        return np.array([[1.0, 0.0, 0.0]], dtype=float)

    # ------------------------
    # Carry mode: explicit region (volume) above the gap
    # ------------------------
    def proj_carry_free(p):
        pp = np.array(p, dtype=float).copy()
        # keep exact boundaries to allow intersection at x=L and x=L+G
        pp[0] = np.clip(pp[0], L, L + G)
        pp[1] = np.clip(pp[1], y_min, y_max)
        pp[2] = np.clip(pp[2], ht + eps, z_max)
        return pp

    # ------------------------
    # Validity checks (inequalities / region constraints)
    # Key fix: tolerate tiny numerical noise for implicit edge planes
    # ------------------------
    tol_x = 1e-3

    def valid_slide_left(p):
        x, y, z = p
        return (0.0 <= x <= L) and (y_min <= y <= y_max)

    def valid_slide_right(p):
        x, y, z = p
        return (L + G <= x <= 2 * L + G) and (y_min <= y <= y_max)

    def valid_lift_left(p):
        x, y, z = p
        if abs(x - L) > tol_x:
            return False
        if not (L - w <= x <= L + tol_x):
            return False
        if not (y_min <= y <= y_max):
            return False
        if not (ht <= z <= z_max):
            return False
        return feasible_left(p) and feasible_lift(p)

    def valid_lift_right(p):
        x, y, z = p
        if abs(x - (L + G)) > tol_x:
            return False
        if not (L + G - tol_x <= x <= L + G + w):
            return False
        if not (y_min <= y <= y_max):
            return False
        if not (ht <= z <= z_max):
            return False
        return feasible_right(p) and feasible_lift(p)

    # ------------------------
    # Costs (mode penalties used by mode_graph + RRT shaping)
    # ------------------------
    slide_cost = {"z_penalty": 0.0, "mode_penalty": 0.0}
    lift_cost  = {"z_penalty": 2.0, "mode_penalty": 0.3}
    carry_cost = {"z_penalty": 2.0, "mode_penalty": 0.6}

    modes = [
        ImplicitMode(
            "SlideLeft",
            h_slide,
            J_slide,
            ambient_bounds,
            is_valid=valid_slide_left,
            cost_weight=slide_cost,
            max_iters=25,
            damping=1e-3,
            backtracking=True,
        ),
        ImplicitMode(
            "LiftLeftZone",
            h_edge_left,
            J_edge_left,
            ambient_bounds,
            is_valid=valid_lift_left,
            cost_weight=lift_cost,
            max_iters=30,
            damping=1e-3,
            backtracking=True,
        ),
        Mode(
            "CarryFree",
            proj_carry_free,
            ambient_bounds,
            is_valid=lambda p: feasible_carry(p),
            cost_weight=carry_cost,
        ),
        ImplicitMode(
            "LiftRightZone",
            h_edge_right,
            J_edge_right,
            ambient_bounds,
            is_valid=valid_lift_right,
            cost_weight=lift_cost,
            max_iters=30,
            damping=1e-3,
            backtracking=True,
        ),
        ImplicitMode(
            "SlideRight",
            h_slide,
            J_slide,
            ambient_bounds,
            is_valid=valid_slide_right,
            cost_weight=slide_cost,
            max_iters=25,
            damping=1e-3,
            backtracking=True,
        ),
    ]

    meta = {
        "L": L,
        "W": W,
        "G": G,
        "table_height": ht,
        "transition_width": w,
        "z_max": z_max,
        "lift_epsilon": eps,
        "feasible_edge_margin": feasible_edge_margin,
        "ambient_bounds": ambient_bounds,
        "implicit_projection": True,
        "tol_x": tol_x,
    }
    return modes, ambient_bounds, meta