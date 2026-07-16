#!/usr/bin/env python3
"""Battery-6 (2026-07-16): THRUST AT CONTROLLED PITCH -- the start block is 17deg
nose-down-forward, and zero-rate sysID HELD that pitch, so all prior thrust tests were
at 17deg (thrust 0.29 horizontal -> forward drift; lapse vz axis = axial-along-tilt, not
vertical). Here: command a pitch RATE open-loop to rotate to a target attitude, settle,
then hold thrust while vz builds -> lapse at a KNOWN pitch. Achieved pitch verified post
by integrating gyro_y from the 17deg start. First fly targets LEVEL (calibrate sign).
Format: t,a_thrust,a_roll,a_pitch,a_yaw,seg   dt=0.025. hover a_thrust=-0.6.
Pitch math: cmd_wy=3.14*a_pitch, actual rate ~2.7*cmd_wy; 17deg=0.297rad.
a_pitch=0.10 -> ~0.85 rad/s -> ~0.35s (14t) per 17deg.
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
    """rotate open-loop (hover thrust during the rotate to stay aloft), settle, then hold 2g."""
    return [
        (HOVER, a_pitch, f"pitch_{tag}", pitch_ticks),   # rotate toward target attitude
        (HOVER, 0.0, "settle", 6),                        # zero-rate: hold new attitude, null rate
        (G2A(2.0), 0.0, f"thrust2g_{tag}", 44),           # 2g hold: vz builds -> lapse at this pitch
    ]


if __name__ == "__main__":
    # CALIBRATED (fly 20260716_035325): a_pitch=+0.10 -> nose UP ~1.0deg/tick; 14t leveled
    # the 17deg start to ~2.4deg. So -a_pitch pitches nose DOWN (more forward tilt).
    write("sysid/seg6_pitch_level.csv", pitch_then_thrust(+0.10, 14, "level"))   # 17 -> ~2deg
    write("sysid/seg6_pitch_30.csv",    pitch_then_thrust(-0.10, 13, "30"))      # 17 +13 -> ~30deg
    write("sysid/seg6_pitch_45.csv",    pitch_then_thrust(-0.10, 26, "45"))      # 17 +26 -> ~43deg
