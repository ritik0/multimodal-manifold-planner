from pathlib import Path


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_default_urdf_path() -> Path:
    return get_repo_root() / "third_party" / "iiwa_description" / "urdf" / "iiwa7.urdf"


def get_default_q_home():
    return [0.0, 0.5, 0.0, -1.0, 0.0, 1.0, 0.0]


def get_default_robot_base_pose():
    return {
        "xyz": [0.72, -0.35, 0.0],
        "yaw_deg": 90.0,
    }


def get_default_ee_link_name():
    return "iiwa_link_ee"