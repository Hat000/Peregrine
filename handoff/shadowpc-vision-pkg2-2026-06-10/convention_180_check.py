"""Identify the fixed ~180-deg discrepancy between the SOLVED gate frame (detector corner
convention) and the MAP gate frame (gates_from_track_records). Try post-rotating the solved
frame by 180 deg about each gate axis and see which collapses E to a few degrees."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
HERE = Path(__file__).resolve().parent


def rodr(v):
    return cv2.Rodrigues(np.ascontiguousarray(v, dtype=np.float64))[0]


def main():
    rows = []
    for gi in range(6):
        d = json.loads((HERE / f"before_g{gi}.json").read_text())
        for r in d["rows"]:
            if r.get("associated") and "rvec_cam_gate" in r and r["n_corners"] == 4 \
                    and r["world_fix_err_m"] < 1.5:
                rows.append(r)
    print(f"good rows: {len(rows)}")
    Rz180 = Rotation.from_euler("Z", np.pi).as_matrix()
    Ry180 = Rotation.from_euler("Y", np.pi).as_matrix()
    Rx180 = Rotation.from_euler("X", np.pi).as_matrix()
    for nm, P in (("none", np.eye(3)), ("gateZ180 (in-plane)", Rz180),
                  ("gateY180 (face flip)", Ry180), ("gateX180", Rx180)):
        angs = []
        for r in rows:
            Rs = rodr(np.asarray(r["rvec_cam_gate"], float)) @ P
            Rp = rodr(np.asarray(r["rvec_cam_gate_pred"], float))
            ang = np.degrees(np.linalg.norm(Rotation.from_matrix(Rp @ Rs.T).as_rotvec()))
            angs.append(ang)
        a = np.array(angs)
        print(f"  post-rot {nm:22s}: |E| p10 {np.percentile(a,10):6.2f}  p50 {np.percentile(a,50):6.2f}  "
              f"p90 {np.percentile(a,90):6.2f}  max {a.max():6.2f} deg")

    # ALSO: how ambiguous was the IPPE tie-break under the broken prior? Reproject both:
    # compare geodesic(cand_i, prior) for the two IPPE branches on a few frontal frames later.
    # First: per-gate breakdown of the best convention (is it the same fix for all gates?)
    best = ("gateZ180", Rz180)
    print(f"\nper-gate |E| medians with {best[0]}:")
    for gi in range(6):
        a = []
        for r in rows:
            if r["gate_id"] != gi:
                continue
            Rs = rodr(np.asarray(r["rvec_cam_gate"], float)) @ best[1]
            Rp = rodr(np.asarray(r["rvec_cam_gate_pred"], float))
            a.append(np.degrees(np.linalg.norm(Rotation.from_matrix(Rp @ Rs.T).as_rotvec())))
        if a:
            print(f"  gate {gi}: N={len(a):3d}  median {np.median(a):6.2f}  p90 {np.percentile(a,90):6.2f}")


if __name__ == "__main__":
    main()
