"""DOES TRAINING GO BLIND AT THE SAME RANGE AS THE WIRE?

The wire, measured over 899 confirmed passes: the last vision fix lands at median 1.78 m and NEVER
inside 1.19 m, because the gate overflows the 58.7 deg vertical frame.

Training is NOT naive about this -- rl/gate_visibility.py projects 8 gate keypoints through the
SAME intrinsics imported from racer.frames and requires >= MIN_VISIBLE_CORNERS (4) in-frame and
unoccluded. So the question is NOT "does training model a blackout" (it does). It is:

    DOES TRAINING'S BLACKOUT BEGIN AT THE SAME RANGE AS THE WIRE'S?

If training keeps seeing the gate materially closer than the wire does, the policy trained with
guidance through an interval it must fly blind in deployment -- a `rate`-class contract mismatch,
and the arm would be to match training's visibility horizon to the measured wire horizon.
If they agree, the "train it to commit blind" arm is NOT justified and must not be run.

Run a head-on, level approach and a set of realistic off-axis/pitched ones, sweeping range.
"""
import math
import os
import sys

import numpy as np
import torch

SCR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCR, "TRAINSRC", "rl"))
sys.path.insert(0, os.path.join(SCR, "TRAINSRC", "src"))

import gate_visibility as gv  # noqa: E402

print("training visibility model: MIN_VISIBLE_CORNERS=%d  FAR_CAP=%.1f m"
      % (gv.MIN_VISIBLE_CORNERS, gv.FAR_CAP_M_DEFAULT))
print("intrinsics in use by TRAINING:")
K = np.asarray(gv.CAMERA_INTRINSICS_K_NP)
print("  fx=%.1f fy=%.1f  W=%d H=%d  -> HFOV %.1f  VFOV %.1f deg"
      % (K[0, 0], K[1, 1], gv.IMAGE_WIDTH, gv.IMAGE_HEIGHT,
         math.degrees(2 * math.atan(gv.IMAGE_WIDTH / (2 * K[0, 0]))),
         math.degrees(2 * math.atan(gv.IMAGE_HEIGHT / (2 * K[1, 1])))))
print()


def quat_from_euler_zup(roll, pitch, yaw):
    """XYZW quaternion for a Z-up FLU body, from intrinsic roll/pitch/yaw (rad)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.tensor([x, y, z, w], dtype=torch.float64)


def horizon(pitch_deg=0.0, dz=0.0, dy=0.0):
    """Closest range at which the gate is still DETECTABLE on a straight-in approach.
    dz = drone height minus gate height (m). dy = lateral offset (m)."""
    rs = np.arange(0.20, 8.001, 0.01)
    last_ok = None
    for r in rs[::-1]:
        drone = torch.tensor([[-float(r), float(dy), float(dz)]], dtype=torch.float64)
        quat = quat_from_euler_zup(0.0, math.radians(pitch_deg), 0.0)[None, :]
        gate = torch.zeros((1, 1, 3), dtype=torch.float64)
        yaw = torch.zeros((1, 1), dtype=torch.float64)
        det, ncorn = gv.gate_detectable(drone, quat, gate, yaw, is_quat=True)
        if bool(det[0, 0]):
            last_ok = float(r)
        elif last_ok is not None:
            break
    return last_ok


print("TRAINING visibility horizon (closest range still detectable), by approach geometry")
print("%-38s %s" % ("geometry", "closest detectable range"))
for lbl, kw in [
    ("head-on, level, centred", dict()),
    ("level, 0.3 m high", dict(dz=0.3)),
    ("level, 0.3 m low", dict(dz=-0.3)),
    ("level, 0.3 m lateral", dict(dy=0.3)),
    ("pitched +10 deg nose-up", dict(pitch_deg=10.0)),
    ("pitched +24 deg (the flown median)", dict(pitch_deg=24.0)),
    ("pitched +24, 0.3 m high", dict(pitch_deg=24.0, dz=0.3)),
    ("pitched -10 deg nose-down", dict(pitch_deg=-10.0)),
]:
    h = horizon(**kw)
    print("%-38s %s" % (lbl, ("%.2f m" % h) if h else "never detectable"))

print()
print("WIRE, measured over 899 confirmed passes: last fix p10 1.48 / MEDIAN 1.78 / p90 2.80 m,")
print("                                          and 0 of 899 ever inside 1.19 m.")
print()
print("READ THE COMPARISON THIS WAY:")
print("  training horizon MUCH CLOSER than 1.78 m -> training guides the policy through an")
print("    interval it must fly blind on the wire = a rate-class contract mismatch, arm justified.")
print("  training horizon ~= 1.78 m -> the policy already trains on this blackout; the")
print("    'teach it to commit blind' arm is NOT justified and must not be run.")
