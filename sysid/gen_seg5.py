#!/usr/bin/env python3
"""Battery-5 sysID (2026-07-16, commander ask): CLIMB-VELOCITY THRUST SWEEP for the
inflow lapse (thrust falls as climb-rate rises; 1g@v=0 -> ~0.4g climbing).
vz is NOT on the wire (ODOMETRY blocked) -> reconstruct vz = -cumsum(accel_z+9.81)*dt
(clean for pure-vertical motion, no roll/pitch). Design: free-fall to build -vz, then
HOLD a fixed collective so vz sweeps up through 0 into a climb while accel_z is logged
-> accel_z vs vz at fixed collective = the lapse curve. A few collectives per fly.
Format: t,a_thrust,a_roll,a_pitch,a_yaw,seg   dt=0.025. hover a_thrust=-0.6 (1g).
"""
import csv
DT = 0.025
G2A = lambda g: g / 2.5 - 1.0
FF, HOLD = 16, 28          # free-fall reset 0.40s (vz-> ~-4 m/s), hold 0.70s (vz sweeps -4 -> +2..+9)


def write(path, rows):
    out, t = [], 0.0
    for a_thrust, seg, n in rows:
        for _ in range(n):
            out.append([f"{t:.5f}", f"{a_thrust:.6f}", "0.000000", "0.000000", "0.000000", seg]); t += DT
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["t", "a_thrust", "a_roll", "a_pitch", "a_yaw", "seg"]); w.writerows(out)
    print(f"  {path}: {len(out)} rows ({len(out)*DT:.2f}s)")


def vzsweep(collectives_g):
    """per collective: free-fall(build -vz) then HOLD it (vz sweeps up through 0)."""
    rows = []
    for g in collectives_g:
        rows.append((G2A(0.0), "ff", FF))                 # 0g reset -> negative vz
        rows.append((G2A(g), f"hold_{g:g}g", HOLD))       # fixed collective, vz sweeps
    return rows


if __name__ == "__main__":
    # high-inflow collectives first (biggest lapse, most valuable) while envelope has room
    write("sysid/seg5_thrust_vzsweep.csv", vzsweep([2.0, 2.5, 3.0]))
    # a gentler companion (lower collectives, slower climb -> wider vz range before ceiling)
    write("sysid/seg5_thrust_vzsweep_lo.csv", vzsweep([1.5, 2.0]))
