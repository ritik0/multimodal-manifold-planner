from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from mm_planner.demo_two_tables_robot import evaluate_single_case


def main():
    L = 0.45
    W = 0.40
    G = 0.30
    table_height = 0.70

    # Conservative support region on table
    start_x_vals = np.linspace(0.10, 0.35, 4)
    start_y_vals = np.linspace(-0.10, 0.10, 5)

    goal_x_vals = np.linspace(0.85, 1.10, 4)
    goal_y_vals = np.linspace(-0.10, 0.10, 5)

    # Surface goals first; you can later add airborne goals too
    goal_z_vals = [table_height, 0.90]

    results = []

    case_idx = 0
    total_cases = len(start_x_vals) * len(start_y_vals) * len(goal_x_vals) * len(goal_y_vals) * len(goal_z_vals)

    print(f"Running {total_cases} cases...")

    for sx in start_x_vals:
        for sy in start_y_vals:
            for gx in goal_x_vals:
                for gy in goal_y_vals:
                    for gz in goal_z_vals:
                        case_idx += 1
                        x_start = np.array([sx, sy, table_height], dtype=float)
                        x_goal = np.array([gx, gy, gz], dtype=float)

                        print(f"[{case_idx}/{total_cases}] start={x_start} goal={x_goal}")

                        try:
                            result = evaluate_single_case(
                                x_start,
                                x_goal,
                                planner_seed=7,
                                L=L,
                                W=W,
                                G=G,
                                table_height=table_height,
                                verbose=False,
                            )
                            result["case_status"] = "ok"
                        except Exception as e:
                            result = {
                                "planner_success": False,
                                "start_x": float(x_start[0]),
                                "start_y": float(x_start[1]),
                                "start_z": float(x_start[2]),
                                "goal_x": float(x_goal[0]),
                                "goal_y": float(x_goal[1]),
                                "goal_z": float(x_goal[2]),
                                "case_status": "exception",
                                "exception": str(e),
                            }

                        results.append(result)

    out_path = Path("start_goal_grid_results.csv")

    # union of all keys
    fieldnames = sorted({k for r in results for k in r.keys()})

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved results to: {out_path.resolve()}")


if __name__ == "__main__":
    main()