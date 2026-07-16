#!/usr/bin/env python3
"""Battery-4 sysID fill programs (2026-07-16): cover the FULL amplitude range reliably.
Battery-3 sweeps only landed 0.1 & 0.6 before the wall (~2.3s). Rate settles by ~tick4
(0.1s) so use SHORTER 0.3s holds -> less attitude windup -> survive longer -> more amps.
  seg4_roll_fill  -- fills roll 0.4/0.2/0.8 (both signs) + repeat 0.6
  seg4_pitch_fill -- same on pitch
  seg4_thrust_hover3g -- clean static 3g FROM HOVER first (v~0), then mids at climbing v
Format: t,a_thrust,a_roll,a_pitch,a_yaw,seg   dt=0.025 (40Hz). hover a_thrust=-0.6 (1g).
"""
import csv
DT = 0.025
HOVER = -0.6
G2A = lambda g: g / 2.5 - 1.0


def write(path, rows):
    out, t = [], 0.0
    for a_thrust, a_roll, a_pitch, a_yaw, seg, n in rows:
        for _ in range(n):
            out.append([f"{t:.5f}", f"{a_thrust:.6f}", f"{a_roll:.6f}",
                        f"{a_pitch:.6f}", f"{a_yaw:.6f}", seg]); t += DT
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["t", "a_thrust", "a_roll", "a_pitch", "a_yaw", "seg"]); w.writerows(out)
    print(f"  {path}: {len(out)} rows ({len(out)*DT:.2f}s)")


HOLD, REST = 12, 6      # 0.30s hold (settles by ~tick4) + 0.15s hover rest
# front-load the MISSING amplitudes in survivable order: mids (0.4,0.2) both signs first,
# then extreme 0.8, then repeat 0.6 for consistency.
FILL_ORDER = [0.4, -0.4, 0.2, -0.2, 0.8, -0.8, 0.6, -0.6]


def fill(axis):
    rows = []
    for amp in FILL_ORDER:
        ar = amp if axis == "roll" else 0.0
        ap = amp if axis == "pitch" else 0.0
        rows.append((HOVER, ar, ap, 0.0, f"{axis}{amp:+.2f}", HOLD))
        rows.append((HOVER, 0.0, 0.0, 0.0, "rest", REST))
    return rows


def thrust_hover3g():
    """3g FIRST from the zero-velocity bootstrap hover (cleanest static 3g), then mids
    interleaved with 1g -- complements free-fall-first (falling v) with climbing v."""
    seq = [(3.0, 12), (1.0, 10), (2.5, 12), (1.0, 10), (2.0, 12), (1.0, 10), (1.5, 12), (1.0, 8)]
    return [(G2A(g), 0.0, 0.0, 0.0, f"thr_{g:g}g", n) for g, n in seq]


if __name__ == "__main__":
    write("sysid/seg4_roll_fill.csv", fill("roll"))
    write("sysid/seg4_pitch_fill.csv", fill("pitch"))
    write("sysid/seg4_thrust_hover3g.csv", thrust_hover3g())
