"""Rule attitude-decode errors in/out: does the per-frame angular error delta_perp correlate
with the attitude itself (roll/pitch), with bearing off optical axis, or with apparent size?

A roll-sign (or any axis-convention) bug in the nav R_wb would make delta ~ k*roll (k~2 for a
full sign flip). A detector/optics effect would track bearing or apparent size instead.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from racer import frames as F                      # noqa: E402
from racer.navigator import load_track_map         # noqa: E402

CHAR_DIR = ROOT / "handoff/shadowpc-assoc-flipfix-2026-06-09"
PG_DIR = ROOT / "handoff/perception-char-2026-06-08/pg"
MAP = ROOT / "handoff/shadowpc-firstcontact-2026-06-02/track_map.json"


def main():
    gates = {g.gate_id: g for g in load_track_map(MAP, corner_to_center=True)}
    rows = []
    for gi in range(6):
        char = json.loads((CHAR_DIR / f"char_g{gi}_robust_k1.0.json").read_text())
        meta = json.loads((PG_DIR / f"course_g{gi}/frames.json").read_text())
        by_fid = {fr["frame_id"]: fr for fr in meta["frames"]}
        for r in char["rows"]:
            if not (r.get("associated") and "world_fix_err_m" in r):
                continue
            if r["world_fix_err_m"] >= 1.5 or r["n_corners"] != 4:
                continue
            fr = by_fid[r["frame_id"]]
            roll, pitch, yaw = F.euler_from_quat_wxyz(np.asarray(fr["odo_q_wxyz"], float))
            R_wb = F.R_world_from_body(roll, pitch, yaw)
            off = np.asarray(r["off_ned"], float)
            L = gates[r["gate_id"]].position_ned - (np.asarray(fr["drone_position_ned"], float) + off)
            nL = float(np.linalg.norm(L))
            d_w = np.cross(L / nL, off) / nL
            rows.append(dict(gate=r["gate_id"], d_b=np.degrees(R_wb.T @ d_w),
                             roll=np.degrees(roll), pitch=np.degrees(pitch), yaw=np.degrees(yaw),
                             bearing=r["bearing_deg"], range_m=nL, score=r["score"]))
    print(f"N={len(rows)}")
    roll = np.array([r["roll"] for r in rows]); pitch = np.array([r["pitch"] for r in rows])
    bear = np.array([r["bearing"] for r in rows])
    db = np.array([r["d_b"] for r in rows])
    print(f"attitude in these frames: roll p5/p50/p95 = {np.percentile(roll,5):+.1f}/"
          f"{np.percentile(roll,50):+.1f}/{np.percentile(roll,95):+.1f} deg ; "
          f"pitch {np.percentile(pitch,5):+.1f}/{np.percentile(pitch,50):+.1f}/"
          f"{np.percentile(pitch,95):+.1f} ; bearing p50/p95 {np.percentile(bear,50):.1f}/"
          f"{np.percentile(bear,95):.1f}")
    for xn, x in (("roll", roll), ("pitch", pitch), ("bearing", bear)):
        line = f"  delta_b vs {xn:>7}: "
        for ax, an in enumerate(("roll", "pitch", "yaw")):
            if x.std() < 1e-9:
                continue
            slope = np.polyfit(x, db[:, ax], 1)[0]
            r2 = np.corrcoef(x, db[:, ax])[0, 1] ** 2
            line += f"d_{an}: {slope:+.3f}deg/deg (r2 {r2:.2f})   "
        print(line)
    # within-gate (remove per-gate means first -- gate identity confounds attitude on a course)
    db_c = db.copy()
    roll_c = roll.copy(); pitch_c = pitch.copy(); bear_c = bear.copy()
    for g in sorted({r["gate"] for r in rows}):
        sel = np.array([r["gate"] == g for r in rows])
        db_c[sel] -= db_c[sel].mean(axis=0)
        roll_c[sel] -= roll_c[sel].mean(); pitch_c[sel] -= pitch_c[sel].mean()
        bear_c[sel] -= bear_c[sel].mean()
    print("within-gate (per-gate means removed):")
    for xn, x in (("roll", roll_c), ("pitch", pitch_c), ("bearing", bear_c)):
        line = f"  delta_b vs {xn:>7}: "
        for ax, an in enumerate(("roll", "pitch", "yaw")):
            slope = np.polyfit(x, db_c[:, ax], 1)[0]
            r2 = np.corrcoef(x, db_c[:, ax])[0, 1] ** 2
            line += f"d_{an}: {slope:+.3f}deg/deg (r2 {r2:.2f})   "
        print(line)


if __name__ == "__main__":
    main()
