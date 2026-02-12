Two-Table Multimodal Planner (Intersection-Based, Fixed Chain)







What this version demonstrates



This prototype plans a path for a point-object (â€œgimbalâ€) from a start pose on a left table to a goal pose on a right table across a gap using multiple motion modes:







SlideLeft (on left tabletop)



LiftLeftZone (near left edge)



CarryFree (over gap, minimum height constraint)



LiftRightZone (near right edge)



SlideRight (on right tabletop)







Each mode is represented by:



a projector `project(x)` that maps a sample to the mode constraint set



a validity check `is\\\_valid(x)` used as a feasibility proxy (stand-in for torque/payload feasibility)







Mode switching is performed via intersection-based transitions:



For each adjacent mode pair (A â†’ B), a transition configuration `xT âˆˆ A âˆ© B` is sampled using alternating projections.



Planning proceeds segment-by-segment:



1) plan in mode A from current state to transition point `xT`


2) switch to mode B at `xT`


3) continue







Carry motion across the gap is explicitly planned (RRT within CarryFree), not â€œdirect connected.â€







How to run



From repo root:







```powershell



python scripts\run_demo.py

