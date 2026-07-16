#!/usr/bin/env python3
"""Battery-6 (2026-07-16): THRUST AT CONTROLLED PITCH -- the start block is 17deg
nose-down-forward, and zero-rate sysID HELD that pitch, so ALL prior thrust tests
(batteries 3/4/5) were at ~17deg with forward drift -- NOT clean vertical.

SIGN (CONFIRMED by the pilot's visual, NOT trustable from the pose-blind gyro integral):
  a_pitch = +0.10  -> nose pitches DOWN / FORWARD (increases pitch angle)
  a_pitch = -0.10  -> nose pitches UP / BACK (toward level)
  rate ~1.04 deg/tick at |a_pitch|=0.10 (measured). theta_actual = 17 - rot_computed,
  where rot_computed = cumsum(-gyro_y_raw)*dt.  <-- my first analysis had this BACKWARD;
  the pilot's eyes corrected it (a_pitch=-0.10 was seen to level the drone).

CAVEAT: even after leveling, the drone CARRIES FORWARD MOMENTUM from the 17deg-start
drift -> the thrust hold is NOT pure-vertical (mixed axial+edgewise inflow). A clean
pure-vertical lapse is not achievable from the 17deg start in this confined scene.

FLOWN (actual pitch from gyro-mag + pilot visual):
  20260716_035325 (a_pitch=+0.10,14t) ~= 32deg forward   (was mislabeled 'level')
  20260716_035550 (a_pitch=-0.10,13t) ~=  4deg near-level (was mislabeled '30')
  20260716_040219 (a_pitch=-0.10,16t) ~=  0deg level (pilot-confirmed roughly level)
  20260716_035727 (a_pitch=-0.10,26t)  crashed mid-rotate (nose-up past level)
Format: t,a_thrust,a_roll,a_pitch,a_yaw,seg   dt=0.025. hover a_thrust=-0.6.
"""
import csv
DT = 0.025
G2A = lambda g: g / 2.5 - 1.0
HOVER = -0.6


def write(path, rows):
    out, t = [], 0.0
    for a_thrust, a_pitch, seg, n in rows:
        for _ in range(n):
            out.append([f"{t:.5f}", f"{a_thrust:.6f}", "0.000000", f"{a_pitch:.6f}", "0.000000", seg]); t += DT
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["t", "a_thrust", "a_roll", "a_pitch", "a_yaw", "seg"]); w.writerows(out)
    print(f"  {path}: {len(out)} rows ({len(out)*DT:.2f}s)")


def pitch_then_thrust(a_pitch, pitch_ticks, tag):
    return [
        (HOVER, a_pitch, f"pitch_{tag}", pitch_ticks),
        (HOVER, 0.0, "settle", 6),
        (G2A(2.0), 0.0, f"thrust2g_{tag}", 44),
    ]


if __name__ == "__main__":
    # corrected: -a_pitch levels the 17deg start; +a_pitch pitches further forward.
    write("sysid/seg6_pitch_trulevel.csv", pitch_then_thrust(-0.10, 16, "trulevel"))  # 17 -> ~0deg
    write("sysid/seg6_pitch_fwd30.csv",    pitch_then_thrust(+0.10, 13, "fwd30"))      # 17 +13 -> ~30deg fwd
