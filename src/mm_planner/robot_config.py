from __future__ import annotations

from pathlib import Path


def _looks_like_valid_urdf(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return False
    if not txt:
        return False
    head = txt[:1000]
    return txt.startswith("<?xml") or ("<robot" in head)


def get_default_urdf_path():
    """
    Find a valid non-empty iiwa7.urdf.
    """
    here = Path(__file__).resolve().parent
    repo_root = here.parent.parent

    candidates = [
        repo_root / "assets" / "robots" / "iiwa7" / "iiwa7.urdf",
        repo_root / "third_party" / "iiwa_description" / "urdf" / "iiwa7.urdf",
        here / "assets" / "iiwa7.urdf",
        here / "iiwa7.urdf",
        repo_root / "assets" / "iiwa7.urdf",
        repo_root / "robot_assets" / "iiwa7.urdf",
        repo_root / "urdf" / "iiwa7.urdf",
        repo_root / "iiwa7.urdf",
    ]

    for p in candidates:
        if _looks_like_valid_urdf(p):
            return p

    for p in repo_root.rglob("iiwa7.urdf"):
        if _looks_like_valid_urdf(p):
            return p

    raise FileNotFoundError(
        "Could not find a valid non-empty iiwa7.urdf in the repo. "
        "Please place the generated iiwa7.urdf in a known folder or update get_default_urdf_path()."
    )


def get_default_ee_link_name():
    return "iiwa_link_ee"


def get_default_q_home():
    return [0.0, 0.35, 0.0, -1.25, 0.0, 1.10, 0.0]


def get_default_robot_base_pose():
    """
    Fixed best base from automatic search.
    """
    return {
        "xyz": [0.56, -0.30, 0.0],
        "yaw_deg": 90.0,
    }


def get_planning_height_limits(table_height: float):
    """
    Conservative robot-aware height limits for the abstract planner.

    This is not a full reachability model. It is a practical cap to stop the
    planner from generating obviously too-high trajectories that later fail in IK.

    You should tune max_task_z empirically from successful executions.
    """
    return {
        "min_task_z": table_height,
        "max_task_z": 0.92,   # conservative reachable planning cap for current iiwa setup
        "preferred_carry_z": table_height + 0.08,
    }