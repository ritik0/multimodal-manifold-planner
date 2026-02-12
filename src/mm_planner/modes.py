import numpy as np


class Mode:
    """
    A mode is defined by:
      - project(x): snaps any x to the mode's constraint set
      - is_valid(x): checks if x is allowed in this mode (e.g., feasibility gates)
      - cost_weight: per-mode weights for cost shaping (soft preferences)
      - ambient_bounds: global bounds used for sampling
    """

    def __init__(self, name: str, projector, ambient_bounds, *, is_valid=None, cost_weight=None):
        self.name = name
        self.project = projector
        self.ambient_bounds = ambient_bounds
        self.is_valid = is_valid if is_valid is not None else (lambda x: True)
        self.cost_weight = cost_weight if cost_weight is not None else {}

    def sample_ambient(self) -> np.ndarray:
        return np.array(
            [np.random.uniform(lo, hi) for lo, hi in self.ambient_bounds],
            dtype=float,
        )


def make_two_tables_problem_3d(
    *,
    L=1.0,
    W=0.6,
    G=1.0,
    table_height=0.75,
    transition_width=0.2,
    z_max=2.0,
    # eps is used to define when we consider the object "truly off the table".
    # IMPORTANT: we intentionally allow Lift manifolds to include z == table_height
    # so that Slide <-> Lift can transition through an actual intersection (z=table_height).
    # Carry still requires z >= table_height + eps.
    lift_epsilon=0.02,
    # Feasibility proxy parameters (placeholder for torque/power feasibility)
    feasible_edge_margin=0.25,   # must be within last 25cm of table to lift/carry
    feasible_y_margin=None,      # optional: restrict y (e.g., near centerline)
):
    """
    3D two-table world with feasibility-gated lift/carry.

    Tables:
      Left tabletop:  x in [0, L], y in [-W/2, W/2], z = table_height
      Right tabletop: x in [L+G, 2L+G], y in [-W/2, W/2], z = table_height

    Modes:
      SlideLeft      : tabletop patch (2D manifold)
      LiftLeftZone   : vertical box above edge strip (region) + feasibility gate
      CarryFree      : box over gap (region) + feasibility gate
      LiftRightZone  : vertical box above right strip (region) + feasibility gate
      SlideRight     : tabletop patch (2D manifold)
    """

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
    # Interpretation: lifting/carry is only feasible when you're close to the table edge strip.
    # Later replace this with torque checks or a learned model.
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
        # Lift is permitted (geometrically) at and above the tabletop.
        # The *carry* mode will enforce the true off-table condition z >= ht + eps.
        x, y, z = p
        return z >= ht

    def feasible_carry(p):
        # Stand-in for "payload/torque feasibility" while carrying.
        # For now we only require being sufficiently above the tabletop.
        x, y, z = p
        return z >= ht + eps

    # ------------------------
    # Projectors (constraints)
    # ------------------------
    def proj_slide_left(p):
        pp = np.array(p, dtype=float).copy()
        pp[0] = np.clip(pp[0], 0.0, L)
        pp[1] = np.clip(pp[1], y_min, y_max)
        pp[2] = ht
        return pp

    def proj_slide_right(p):
        pp = np.array(p, dtype=float).copy()
        pp[0] = np.clip(pp[0], L + G, 2 * L + G)
        pp[1] = np.clip(pp[1], y_min, y_max)
        pp[2] = ht
        return pp

    def proj_lift_left_zone(p):
        pp = np.array(p, dtype=float).copy()
        pp[0] = np.clip(pp[0], L - w, L)
        pp[1] = np.clip(pp[1], y_min, y_max)
        # Allow z == ht to enable SlideLeft <-> LiftLeft intersection.
        pp[2] = np.clip(pp[2], ht, z_max)
        return pp

    def proj_carry_free(p):
        pp = np.array(p, dtype=float).copy()
        pp[0] = np.clip(pp[0], L, L + G)
        pp[1] = np.clip(pp[1], y_min, y_max)
        # Carry requires being "truly off the table".
        pp[2] = np.clip(pp[2], ht + eps, z_max)
        return pp

    def proj_lift_right_zone(p):
        pp = np.array(p, dtype=float).copy()
        pp[0] = np.clip(pp[0], L + G, L + G + w)
        pp[1] = np.clip(pp[1], y_min, y_max)
        # Allow z == ht to enable SlideRight <-> LiftRight intersection.
        pp[2] = np.clip(pp[2], ht, z_max)
        return pp

    # ------------------------
    # Mode costs
    # ------------------------
    slide_cost = {"z_penalty": 0.0, "mode_penalty": 0.0}
    lift_cost  = {"z_penalty": 2.0, "mode_penalty": 0.3}   # discourage high lift + discourage being in lift
    carry_cost = {"z_penalty": 2.0, "mode_penalty": 0.6}   # carry is "more expensive" than lift

    modes = [
        Mode("SlideLeft", proj_slide_left, ambient_bounds, is_valid=lambda p: True, cost_weight=slide_cost),

        Mode("LiftLeftZone", proj_lift_left_zone, ambient_bounds,
             is_valid=lambda p: feasible_left(p) and feasible_lift(p),
             cost_weight=lift_cost),

        Mode("CarryFree", proj_carry_free, ambient_bounds,
             is_valid=lambda p: feasible_carry(p),
             cost_weight=carry_cost),

        Mode("LiftRightZone", proj_lift_right_zone, ambient_bounds,
             is_valid=lambda p: feasible_right(p) and feasible_lift(p),
             cost_weight=lift_cost),

        Mode("SlideRight", proj_slide_right, ambient_bounds, is_valid=lambda p: True, cost_weight=slide_cost),
    ]

    meta = {
        "L": L, "W": W, "G": G,
        "table_height": ht,
        "transition_width": w,
        "z_max": z_max,
        "lift_epsilon": eps,
        "feasible_edge_margin": feasible_edge_margin,
        "ambient_bounds": ambient_bounds,
    }
    return modes, ambient_bounds, meta