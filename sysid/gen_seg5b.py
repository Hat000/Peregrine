#!/usr/bin/env python3
"""Battery-5b (2026-07-16): HOVER-COLLECTIVE inflow lapse -- nails the commander's cited
0.4g point. Hold hover-collective (col 0.266, 1g nominal) while the drone is CLIMBING
fast: launch up with a 3g burst to build +vz, then drop to hover-col and hold -> thrust
lapses below 1g, read accel_z as vz coasts down. Repeat. Reconstruct vz post as before.
Format: t,a_thrust,a_roll,a_pitch,a_yaw,seg   dt=0.025.
"""
import csv
DT = 0.025
G2A = lambda g: g / 2.5 - 1.0


def write(path, rows):
    out, t = [], 0.0
    for a_thrust, seg, n in rows:
        for _ in range(n):
            out.append([f"{t:.5f}", f"{a_thrust:.6f}", "0.000000", "0.000000", "0.000000", seg]); t += DT
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["t", "a_thrust", "a_roll", "a_pitch", "a_yaw", "seg"]); w.writerows(out)
    print(f"  {path}: {len(out)} rows ({len(out)*DT:.2f}s)")


def hover_lapse():
    rows = []
    for _ in range(2):
        rows.append((G2A(3.0), "launch_3g", 24))       # 0.6s 3g -> vz ~+12 m/s
        rows.append((G2A(1.0), "hold_hover", 32))       # 0.8s hover-col at high vz -> lapse sweep
    return rows


if __name__ == "__main__":
    write("sysid/seg5b_hover_vzsweep.csv", hover_lapse())
