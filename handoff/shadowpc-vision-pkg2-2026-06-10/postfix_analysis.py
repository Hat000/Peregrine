"""Post-map-frame-fix re-analysis on the instrumented dumps (self-contained: extras carry pose).

1. population shift BEFORE (broken prior) vs FIXEDFRAME: solved / good / catastrophic / leak;
2. rotation channel: solved-vs-predicted gate rotation now that the prior is aligned --
   in-plane bias (optical-roll channel) + out-of-plane scatter;
3. lever channel: does the roll-coupling of delta_perp persist? per-gate means? residual stds
   (the inputs to the honest sigma_theta).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from racer import frames as F                      # noqa: E402

HERE = Path(__file__).resolve().parent


def rodr(v):
    return cv2.Rodrigues(np.ascontiguousarray(v, dtype=np.float64))[0]


def load(pattern):
    rows = []
    for gi in range(6):
        d = json.loads((HERE / (pattern % gi)).read_text())
        for r in d["rows"]:
            if r.get("associated") and "world_fix_err_m" in r:
                rows.append(r)
    return rows


def popstats(rows, label):
    offered = [r for r in rows if r.get("range_ok", True)]
    good1 = [r for r in offered if r["world_fix_err_m"] < 1.0]
    cat = [r for r in offered if r["world_fix_err_m"] >= 3.0]
    rej1 = sum(r["maha"] > 16.27 for r in good1 if np.isfinite(r["maha"]))
    leak = sum(r["maha"] <= 16.27 for r in cat if np.isfinite(r["maha"]))
    fx = np.array([r["world_fix_err_m"] for r in offered])
    print(f"  {label:12s} solved {len(rows):3d}  offered {len(offered):3d}  good<1m {len(good1):3d} "
          f"(rej {rej1:2d} = {100*rej1/max(len(good1),1):3.0f}%)  cat {len(cat)} (leak {leak})  "
          f"|fix| p50 {np.percentile(fx,50):.2f} p90 {np.percentile(fx,90):.2f}")


def main():
    before = load("before_g%d.json")
    fixed = load("fixedframe_g%d.json")
    print("== population (production gate settings in both) ==")
    popstats(before, "before")
    popstats(fixed, "fixedframe")

    # rotation channel on the FIXED dumps
    print("\n== rotation channel (fixedframe; good 4-corner) ==")
    rows = [r for r in fixed if r["n_corners"] == 4 and r["world_fix_err_m"] < 1.5
            and "rvec_cam_gate" in r]
    ip, oop, mag = [], [], []
    for r in rows:
        Rs = rodr(np.asarray(r["rvec_cam_gate"], float))
        Rp = rodr(np.asarray(r["rvec_cam_gate_pred"], float))
        ev_gate = Rotation.from_matrix(Rs.T @ Rp).as_rotvec()
        ip.append(np.degrees(ev_gate[2]))
        oop.append(np.degrees(np.hypot(ev_gate[0], ev_gate[1])))
        mag.append(np.degrees(np.linalg.norm(Rotation.from_matrix(Rp @ Rs.T).as_rotvec())))
    ip, oop, mag = np.array(ip), np.array(oop), np.array(mag)
    print(f"  |E| p50 {np.percentile(mag,50):.2f}  p90 {np.percentile(mag,90):.2f} deg "
          f"(was 174 deg before the frame fix)")
    print(f"  in-plane: mean {ip.mean():+.2f}  std {ip.std():.2f} deg   "
          f"out-of-plane p50 {np.percentile(oop,50):.2f}  p90 {np.percentile(oop,90):.2f} deg")

    # lever channel: delta_perp + roll coupling (self-contained from extras)
    print("\n== lever channel (fixedframe; good 4-corner) ==")
    db, roll, gate, rng = [], [], [], []
    for r in rows:
        R_wb = F.R_world_from_body(*F.euler_from_quat_wxyz(np.asarray(r["odo_q_wxyz"], float)))
        lever = (R_wb @ F.R_camera_from_body().T) @ np.asarray(r["t_cam_solved"], float)
        nL = float(np.linalg.norm(lever))
        d_w = np.cross(lever / nL, np.asarray(r["off_ned"], float)) / nL
        db.append(np.degrees(R_wb.T @ d_w))
        roll.append(r["rpy_deg"][0]); gate.append(r["gate_id"]); rng.append(nL)
    db = np.array(db); roll = np.array(roll); gate = np.array(gate)
    print(f"  delta_b mean [{db[:,0].mean():+.2f} {db[:,1].mean():+.2f} {db[:,2].mean():+.2f}]  "
          f"std [{db[:,0].std():.2f} {db[:,1].std():.2f} {db[:,2].std():.2f}] deg")
    for ax, nm in ((2, "yaw"), (0, "roll-ax")):
        s = np.polyfit(roll, db[:, ax], 1)[0]
        r2 = np.corrcoef(roll, db[:, ax])[0, 1] ** 2
        print(f"  delta_{nm} vs roll: slope {s:+.3f} deg/deg (r2 {r2:.2f})   [was -0.55 (0.87) / -0.12 (0.72)]")
    print("  per-gate mean delta_b (deg):")
    for g in sorted(set(gate)):
        sub = db[gate == g]
        print(f"    gate {g}: N={len(sub):3d}  [{sub[:,0].mean():+5.2f} {sub[:,1].mean():+5.2f} "
              f"{sub[:,2].mean():+5.2f}]  std [{sub[:,0].std():4.2f} {sub[:,1].std():4.2f} {sub[:,2].std():4.2f}]")


if __name__ == "__main__":
    main()
