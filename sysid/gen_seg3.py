#!/usr/bin/env python3
"""Generate battery-3 sysID programs (2026-07-16, commander re-fly asks):
  seg3_roll_ampsweep  -- roll  rate-gain curve, priority-ordered (0.1 & 0.6 first)
  seg3_pitch_ampsweep -- pitch rate-gain curve, same
  seg3_thrust_curve_ff-- thrust accel_z vs collective, FREE-FALL FIRST then 3g (both endpoints early)
Format: t,a_thrust,a_roll,a_pitch,a_yaw,seg   (6th col = segment label, logged by fly_rl passthrough)
dt=0.025 (40Hz). hover thrust a=-0.6 (=1g). Rate axis stepped; other rate axes 0.
"""
import csv

DT = 0.025
HOVER = -0.6                     # a_thrust for 1g hover
G_TO_ATHRUST = lambda g: g / 2.5 - 1.0   # normed_thrust=2.5(a+1)=g  ->  a = g/2.5 - 1


def write(path, rows):
    """rows: list of (a_thrust,a_roll,a_pitch,a_yaw,seg,n_ticks) -> expand to per-tick CSV."""
    out = []
    t = 0.0
    for a_thrust, a_roll, a_pitch, a_yaw, seg, n in rows:
        for _ in range(n):
            out.append([f"{t:.5f}", f"{a_thrust:.6f}", f"{a_roll:.6f}",
                        f"{a_pitch:.6f}", f"{a_yaw:.6f}", seg])
            t += DT
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "a_thrust", "a_roll", "a_pitch", "a_yaw", "seg"])
        w.writerows(out)
    print(f"  {path}: {len(out)} rows ({len(out)*DT:.2f}s)")
    return len(out)


HOLD = 20     # 0.5s settled hold
REST = 10     # 0.25s hover rest between steps
# priority-ordered amplitudes: low (0.1) & a high (0.6) FIRST, sign-paired to bound drift,
# then 0.2/0.4/0.8 as bonus before the wall.
AMP_ORDER = [0.1, -0.1, 0.6, -0.6, 0.2, -0.2, 0.4, -0.4, 0.8, -0.8]


def ampsweep(axis):
    """axis in {'roll','pitch'} -> rows stepping that rate axis, hover thrust, hover rests between."""
    rows = []
    for amp in AMP_ORDER:
        ar = amp if axis == "roll" else 0.0
        ap = amp if axis == "pitch" else 0.0
        rows.append((HOVER, ar, ap, 0.0, f"{axis}{amp:+.2f}", HOLD))
        rows.append((HOVER, 0.0, 0.0, 0.0, "rest", REST))
    return rows


def thrust_ff():
    """free-fall(0g) FIRST -> 3g SECOND (both endpoints early while low) -> mids descending."""
    seq = [  # (g, n_ticks)
        (0.0, 24),   # free-fall: accel_z~0, drops to build ceiling margin (endpoint #1)
        (3.0, 24),   # full-stick: accel_z~-3g, arrests the fall then climbs (endpoint #2)
        (2.5, 16),
        (2.0, 16),
        (1.5, 16),
        (1.0, 16),   # back to hover
    ]
    return [(G_TO_ATHRUST(g), 0.0, 0.0, 0.0, f"thr_{g:g}g", n) for g, n in seq]


if __name__ == "__main__":
    write("sysid/seg3_roll_ampsweep.csv", ampsweep("roll"))
    write("sysid/seg3_pitch_ampsweep.csv", ampsweep("pitch"))
    write("sysid/seg3_thrust_curve_ff.csv", thrust_ff())
