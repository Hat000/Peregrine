"""Compare fit-A vs fit-B predicted body-frame error across the roll range (extrapolation)."""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from racer import frames as F  # noqa: E402

SWAP = np.array([[0., 1, 0], [0, 0, 1], [1, 0, 0]])
TAU = float(F.CAMERA_PITCH_RAD)
Rz = lambda a: Rotation.from_euler("Z", a).as_matrix()  # noqa: E731
Ry = lambda a: Rotation.from_euler("Y", a).as_matrix()  # noqa: E731
Rx = lambda a: Rotation.from_euler("X", a).as_matrix()  # noqa: E731


def model(ps, th, ph):
    return Rz(ps) @ Ry(th) @ Rx(ph) @ Ry(TAU) @ SWAP.T


def cand(ps, th, ph, tp, tq, k):
    return Rz(ps) @ Ry(th + tp) @ Rx(k * ph) @ Ry(tq) @ SWAP.T


A = (np.deg2rad(29.11), TAU - np.deg2rad(29.11), 1.0)
B = (np.deg2rad(162.37), TAU - np.deg2rad(162.37), 1.038)
th, ps = np.deg2rad(-8.0), np.pi
print("predicted body-frame error rotvec e_b (deg) vs roll [model.T @ cand -> body]")
print(f"{'roll':>6} | {'fit A: x / y / z':>26} | {'fit B: x / y / z':>26}")
for rdeg in (-45, -20, -10, -5, -2, 0, 2, 5, 10, 20, 45, 65):
    ph = np.deg2rad(rdeg)
    out = []
    for tp, tq, k in (A, B):
        E = model(ps, th, ph).T @ cand(ps, th, ph, tp, tq, k)
        e_b = (Ry(TAU) @ SWAP.T) @ Rotation.from_matrix(E).as_rotvec()
        out.append(np.degrees(e_b))
    a, b = out
    print(f"{rdeg:6d} | {a[0]:+7.2f} {a[1]:+7.2f} {a[2]:+7.2f} | {b[0]:+7.2f} {b[1]:+7.2f} {b[2]:+7.2f}")
