from pathlib import Path


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_default_urdf_path() -> Path:
    return get_repo_root() / "assets" / "robots" / "simple_arm" / "simple_arm.urdf"


def get_default_q_home():
    return [0.0, 0.4, -0.6]