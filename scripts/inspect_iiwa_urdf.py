# scripts/inspect_iiwa_urdf.py
from urdfpy import URDF
from pathlib import Path

urdf_path = Path("third_party/iiwa_description/urdf/iiwa7.urdf").resolve()
robot = URDF.load(str(urdf_path))

print("JOINTS:")
for j in robot.joints:
    print(j.name, j.joint_type)

print("\nLINKS:")
for l in robot.links:
    print(l.name)