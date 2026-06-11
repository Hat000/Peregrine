"""Fit the camera-chain composition on the lever channel (the clean 2-DOF/frame observable).

Hypothesis family for the TRUE render chain (vs model = tilt fully after roll):

    R_wc_true = Rz(yaw) @ Ry(pitch + tau_pre) @ Rx(k * roll) @ Ry(tau_post) @ SWAP.T

  model = (tau_pre=0, tau_post=20deg, k=1). tau_pre+tau_post ~ 20deg keeps the level-attitude
  geometry (verified at first contact); the SPLIT moves the roll-coupling:
  pre-roll tilt makes the body-frame error ~ roll * (1-cos tau_pre, 0, -sin tau_pre).

For params p, the predicted angular error rotvec is e_cam = rotvec(R_wc_model.T @ R_wc_true),
mapped to a predicted world-fix error off_pred = (R_wc_model e_cam... applied via the lever:
off_pred = cross(delta_w, L) with delta_w = R_wb e_b. We fit p by least squares of the
PERPENDICULAR residual (off - off_pred) projected off the lever direction (depth excluded:
that channel is PnP noise, not attitude), over good 4-corner fixes; plus a constant world
vertical offset c_D as nuisance (the measured +0.3 m map/centre bias) so it cannot leak into
the rotation fit.

Cross-validates the winning params on the task2_frames gate-0 bundle (different run/day) if
a dump for it exists (task2_*.json).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from racer import frames as F                      # noqa: E402

HERE = Path(__file__).resolve().parent
SWAP = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
TAU = float(F.CAMERA_PITCH_RAD)


def Rz(a): return Rotation.from_euler("Z", a).as_matrix()
def Ry(a): return Rotation.from_euler("Y", a).as_matrix()
def Rx(a): return Rotation.from_euler("X", a).as_matrix()


def load(pattern, gis=range(6), good_max=1.5):
    rows = []
    for gi in gis:
        p = HERE / (pattern % gi) if "%" in pattern else HERE / pattern
        d = json.loads(p.read_text())
        for r in d["rows"]:
            if not (r.get("associated") and "world_fix_err_m" in r):
                continue
            if r["n_corners"] != 4 or r["world_fix_err_m"] >= good_max:
                continue
            roll, pitch, yaw = (float(np.radians(a)) for a in r["rpy_deg"])
            R_wb = F.R_world_from_body(roll, pitch, yaw)
            R_wc_model = R_wb @ F.R_camera_from_body().T
            lever = R_wc_model @ np.asarray(r["t_cam_solved"], float)
            rows.append(dict(roll=roll, pitch=pitch, yaw=yaw, R_wc_model=R_wc_model,
                             lever=lever, off=np.asarray(r["off_ned"], float),
                             gate_id=r["gate_id"]))
        if "%" not in pattern:
            break
    return rows


def Rwc_cand(r, tau_pre, tau_post, k):
    return (Rz(r["yaw"]) @ Ry(r["pitch"] + tau_pre) @ Rx(k * r["roll"]) @ Ry(tau_post) @ SWAP.T)


def perp_resid(rows, tau_pre, tau_post, k, c_D):
    """Perpendicular-to-lever residual (2 DOF/frame) after removing the candidate-composition
    prediction and a constant world-D offset."""
    out = []
    for r in rows:
        Rm, L = r["R_wc_model"], r["lever"]
        e_cam = Rotation.from_matrix(Rm.T @ Rwc_cand(r, tau_pre, tau_post, k)).as_rotvec()
        delta_w = Rm @ e_cam                        # world-frame error rotvec
        # model error rotates the lever: predicted off = -delta x L? sign fixed by fit of k,
        # taus (free signs); use off_pred = cross(delta_w, L)
        off_pred = np.cross(delta_w, L) + np.array([0.0, 0.0, c_D])
        res = r["off"] - off_pred
        Lh = L / np.linalg.norm(L)
        res_perp = res - (res @ Lh) * Lh
        out.append(res_perp)
    return np.concatenate(out)


def roll_slope_of_residual(rows, rv):
    res = rv.reshape(-1, 3)
    out = []
    for r, e in zip(rows, res):
        L = r["lever"]; nL = np.linalg.norm(L)
        d_w = np.cross(L / nL, e) / nL
        R_wb = F.R_world_from_body(r["roll"], r["pitch"], r["yaw"])
        out.append(np.degrees(R_wb.T @ d_w))
    db = np.array(out); roll = np.degrees([r["roll"] for r in rows])
    s_yaw = np.polyfit(roll, db[:, 2], 1)[0]
    r2 = np.corrcoef(roll, db[:, 2])[0, 1] ** 2
    return s_yaw, r2, db


def report(rows, label, tau_pre, tau_post, k, c_D):
    rv = perp_resid(rows, tau_pre, tau_post, k, c_D)
    rms = np.sqrt(np.mean(rv ** 2))
    s, r2, db = roll_slope_of_residual(rows, rv)
    print(f"  {label:42s} perp-rms {rms:.3f} m  resid yaw-slope {s:+.3f} deg/deg (r2 {r2:.2f})  "
          f"resid delta_b std [{db[:,0].std():.2f} {db[:,1].std():.2f} {db[:,2].std():.2f}] deg")
    return rms, db


def main():
    rows = load("fixedframe_g%d.json")
    print(f"fit population: {len(rows)} good 4-corner fixes (fixedframe dumps)\n")

    print("== fixed-composition baselines ==")
    report(rows, "model (0, 20deg, 1) no c_D", 0.0, TAU, 1.0, 0.0)
    report(rows, "model + c_D=-0.30", 0.0, TAU, 1.0, -0.30)

    print("\n== fits (least squares on perp residual) ==")
    # 1: tau_pre free, net tilt fixed 20deg, k=1, c_D free
    f1 = least_squares(lambda p: perp_resid(rows, p[0], TAU - p[0], 1.0, p[1]),
                       x0=[0.3, -0.3], method="lm")
    print(f"  [A] tau_pre free, net=20, k=1:   tau_pre={np.degrees(f1.x[0]):+.2f} deg  "
          f"c_D={f1.x[1]:+.3f} m")
    report(rows, "      residuals", f1.x[0], TAU - f1.x[0], 1.0, f1.x[1])

    # 2: + k free
    f2 = least_squares(lambda p: perp_resid(rows, p[0], TAU - p[0], p[1], p[2]),
                       x0=[0.3, 1.0, -0.3], method="lm")
    print(f"  [B] + roll-scale k:              tau_pre={np.degrees(f2.x[0]):+.2f} deg  "
          f"k={f2.x[1]:.3f}  c_D={f2.x[2]:+.3f} m")
    report(rows, "      residuals", f2.x[0], TAU - f2.x[0], f2.x[1], f2.x[2])

    # 3: tau_pre AND tau_post fully free (net tilt can deviate from 20)
    f3 = least_squares(lambda p: perp_resid(rows, p[0], p[1], p[2], p[3]),
                       x0=[0.3, TAU - 0.3, 1.0, -0.3], method="lm")
    print(f"  [C] both taus + k free:          tau_pre={np.degrees(f3.x[0]):+.2f}  "
          f"tau_post={np.degrees(f3.x[1]):+.2f}  net={np.degrees(f3.x[0]+f3.x[1]):+.2f} deg  "
          f"k={f3.x[2]:.3f}  c_D={f3.x[3]:+.3f} m")
    report(rows, "      residuals", f3.x[0], f3.x[1], f3.x[2], f3.x[3])

    # cross-validation on task2 gate-0 bundle (different day) if dumped
    t2 = HERE / "task2_fixedframe.json"
    if t2.exists():
        rows2 = load("task2_fixedframe.json", gis=[0])
        print(f"\n== cross-validation: task2_frames (N={len(rows2)}) ==")
        report(rows2, "model (0,20,1) + c_D fit-A", 0.0, TAU, 1.0, f1.x[1])
        report(rows2, "fit-A params", f1.x[0], TAU - f1.x[0], 1.0, f1.x[1])
        report(rows2, "fit-B params", f2.x[0], TAU - f2.x[0], f2.x[1], f2.x[2])


if __name__ == "__main__":
    main()
