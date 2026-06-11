"""Test the LATENCY hypothesis: the image content corresponds to a sim instant offset by a
constant Delta from the recv-paired telemetry pose, so the measured 'fix error' contains a
velocity-proportional term:

    off_i  =  -Delta * v_i  +  c  (+ true sensor error)

Fit on the canonical course dumps (joined to frames.json lpn_vel by frame_id), check:
  * structure: full 3x3 M in off = M v + c should be ~ -Delta * I (isotropic, diagonal)
  * does removing it kill the roll-coupling (roll was a banking proxy for lateral velocity)?
  * does Delta TRANSFER to the task2 bundle (different day/flight profile)? The roll-coupling
    did not -- a real chain constant must.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from racer import frames as F                      # noqa: E402

HERE = Path(__file__).resolve().parent
PG = ROOT / "handoff/perception-char-2026-06-08/pg"
T2 = ROOT / "handoff/shadowpc-followups-2026-06-05/task2_frames"


def load(dump_pattern, frames_dirs, good_max=1.5):
    rows = []
    for gi, fdir in frames_dirs:
        p = HERE / (dump_pattern % gi) if "%" in dump_pattern else HERE / dump_pattern
        d = json.loads(p.read_text())
        meta = json.loads((fdir / "frames.json").read_text())
        by_fid = {fr["frame_id"]: fr for fr in meta["frames"]}
        for r in d["rows"]:
            if not (r.get("associated") and "world_fix_err_m" in r):
                continue
            if r["n_corners"] != 4 or r["world_fix_err_m"] >= good_max:
                continue
            fr = by_fid[r["frame_id"]]
            roll = float(np.radians(r["rpy_deg"][0]))
            R_wb = F.R_world_from_body(*(np.radians(a) for a in r["rpy_deg"]))
            lever = (R_wb @ F.R_camera_from_body().T) @ np.asarray(r["t_cam_solved"], float)
            rows.append(dict(off=np.asarray(r["off_ned"], float),
                             v=np.asarray(fr["lpn_vel"], float), roll=roll, R_wb=R_wb,
                             lever=lever, gate_id=r["gate_id"], range_m=r["true_range_m"]))
    return rows


def delta_b_of(rows, residuals):
    out = []
    for r, e in zip(rows, residuals):
        L = r["lever"]; nL = np.linalg.norm(L)
        out.append(np.degrees(r["R_wb"].T @ (np.cross(L / nL, e) / nL)))
    return np.array(out)


def roll_stats(rows, residuals, label):
    db = delta_b_of(rows, residuals)
    roll = np.degrees([r["roll"] for r in rows])
    s = np.polyfit(roll, db[:, 2], 1)[0]
    r2 = np.corrcoef(roll, db[:, 2])[0, 1] ** 2
    rms = np.sqrt(np.mean(np.concatenate(residuals) ** 2)) if isinstance(residuals, list) else \
        np.sqrt(np.mean(np.asarray(residuals) ** 2))
    print(f"  {label:34s} rms {rms:.3f} m  yaw-roll slope {s:+.3f} (r2 {r2:.2f})  "
          f"delta_b std [{db[:,0].std():.2f} {db[:,1].std():.2f} {db[:,2].std():.2f}] deg")


def main():
    rows = load("fixedframe_g%d.json", [(g, PG / f"course_g{g}") for g in range(6)])
    print(f"canonical: {len(rows)} good fixes; speed p50 "
          f"{np.percentile([np.linalg.norm(r['v']) for r in rows],50):.1f} m/s")

    off = np.array([r["off"] for r in rows])
    v = np.array([r["v"] for r in rows])

    # full 3x3 + const:  off = M v + c   (structure check: M ~ -Delta I?)
    X = np.hstack([v, np.ones((len(v), 1))])
    M = np.zeros((3, 3)); c = np.zeros(3)
    for ax in range(3):
        coef, *_ = np.linalg.lstsq(X, off[:, ax], rcond=None)
        M[ax] = coef[:3]; c[ax] = coef[3]
    print("\nfull M (off = M v + c), ms:")
    for ax, nm in enumerate("N E D".split()):
        print(f"  {nm}: [{M[ax,0]*1000:+7.1f} {M[ax,1]*1000:+7.1f} {M[ax,2]*1000:+7.1f}]   c={c[ax]:+.3f} m")

    # scalar-Delta + c fit:  off = -Delta v + c
    Arows, brows = [], []
    for i in range(len(v)):
        for ax in range(3):
            row = np.zeros(4)
            row[0] = -v[i, ax]
            row[1 + ax] = 1.0
            Arows.append(row); brows.append(off[i, ax])
    theta, *_ = np.linalg.lstsq(np.array(Arows), np.array(brows), rcond=None)
    delta, cc = theta[0], theta[1:4]
    res = off - (-delta * v + cc)
    print(f"\nscalar fit: Delta = {delta*1000:+.1f} ms   c = [{cc[0]:+.3f} {cc[1]:+.3f} {cc[2]:+.3f}] m")
    print(f"  rms before {np.sqrt(np.mean(off**2)):.3f} -> after {np.sqrt(np.mean(res**2)):.3f} m")

    print("\nroll-coupling before/after latency removal (canonical):")
    roll_stats(rows, off, "raw off")
    roll_stats(rows, off - (-delta * v), "off + Delta*v (no const)")
    roll_stats(rows, res, "off + Delta*v - c")

    # per-gate residual means (do the per-gate 'map offsets' survive?)
    print("\nper-gate residual means after latency+c removal (m):")
    for g in sorted({r["gate_id"] for r in rows}):
        sel = [i for i, r in enumerate(rows) if r["gate_id"] == g]
        m = res[sel].mean(axis=0)
        print(f"  gate {g}: N={len(sel):3d}  [{m[0]:+.3f} {m[1]:+.3f} {m[2]:+.3f}]")

    # ---- transfer test: task2 (different day) ----
    rows2 = load("task2_fixedframe.json", [(0, T2)])
    off2 = np.array([r["off"] for r in rows2]); v2 = np.array([r["v"] for r in rows2])
    print(f"\ntask2: {len(rows2)} good fixes; speed p50 "
          f"{np.percentile([np.linalg.norm(x) for x in v2],50):.1f} m/s")
    res2_canonDelta = off2 - (-delta * v2 + cc)
    print("apply CANONICAL Delta+c to task2:")
    roll_stats(rows2, off2, "raw off")
    roll_stats(rows2, res2_canonDelta, "canonical Delta+c removed")
    # task2's own Delta
    Arows, brows = [], []
    for i in range(len(v2)):
        for ax in range(3):
            row = np.zeros(4); row[0] = -v2[i, ax]; row[1 + ax] = 1.0
            Arows.append(row); brows.append(off2[i, ax])
    th2, *_ = np.linalg.lstsq(np.array(Arows), np.array(brows), rcond=None)
    print(f"task2's own fit: Delta = {th2[0]*1000:+.1f} ms   "
          f"c = [{th2[1]:+.3f} {th2[2]:+.3f} {th2[3]:+.3f}] m")


if __name__ == "__main__":
    main()
