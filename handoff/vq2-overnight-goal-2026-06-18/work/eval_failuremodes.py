"""Failure-mode analysis: which gate TYPES are least precise (or undetected), to target in training.

Per GT gate in the photoreal val set (full keypoint GT), match to the detector's nearest prediction and
measure inner-4 corner pixel error (a miss = fail). Bin by apparent size (range proxy), hue set, and
clipping (off-frame corners) -> the systematic weaknesses. Also counts false positives on hard negatives.

Finding (2026-06-16): the dominant failure is NEAR/large/overflowing gates (corner error blows up) -- a
direct symptom of the OKS area-normalization in the keypoint loss (large gates under-penalized). Clipped
gates are the second. Hue is NOT a failure axis (appearance randomization worked). See COMMANDER_REPORT §10.

Usage:  python eval_failuremodes.py <weights.pt> [more ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

SETS = Path("C:/Users/Shadow/Peregrine-vq2data/handoff/vq2-blender-render-2026-06-15/sets")
W, H = 640, 360


def gts_of(ip: str):
    lp = Path(ip.replace("\\images\\", "\\labels\\").replace("/images/", "/labels/")).with_suffix(".txt")
    out = []
    for line in lp.read_text().splitlines():
        v = line.split()
        if len(v) != 29:
            continue
        cx, cy, w, h = (float(x) for x in v[1:5])
        k = np.array(v[5:], float).reshape(8, 3); k[:, 0] *= W; k[:, 1] *= H
        out.append((cx * W, cy * H, w * W, h * H, k))
    return out


def analyze(weights, vals):
    m = YOLO(weights); recs = []; fp = nn = 0
    for ip in vals:
        img = cv2.imread(ip); setn = Path(ip).parents[2].name; gts = gts_of(ip)
        r = m.predict(img, verbose=False, device=0)[0]
        pb = r.boxes.xywh.cpu().numpy() if (r.boxes is not None and len(r.boxes)) else np.zeros((0, 4))
        pk = r.keypoints.xy.cpu().numpy() if (r.keypoints is not None and r.keypoints.xy.numel()) else np.zeros((0, 8, 2))
        if not gts:
            nn += 1; fp += len(pb); continue
        used = set()
        for (cx, cy, w, h, k) in gts:
            inn = k[:4]; rec = dict(set=setn, w=w, off=int((inn[:, 2] == 0).sum()), det=False, err=np.nan)
            if len(pb):
                d = np.linalg.norm(pb[:, :2] - np.array([cx, cy]), axis=1); j = int(d.argmin())
                if d[j] < max(20, 0.6 * w) and j not in used:
                    used.add(j); rec["det"] = True; vis = inn[:, 2] > 0
                    if vis.any() and len(pk) > j:
                        rec["err"] = float(np.mean(np.linalg.norm(pk[j][:4][vis] - inn[vis, :2], axis=1)))
            recs.append(rec)
    return recs, fp, nn


def binrep(recs, label, key, bins):
    print(f"  by {label}:")
    for bl, pred in bins:
        sub = [r for r in recs if pred(key(r))]
        if not sub:
            continue
        dd = sum(r["det"] for r in sub)
        ee = np.array([r["err"] for r in sub if r["det"] and not np.isnan(r["err"])])
        bad = sum(1 for r in sub if (not r["det"]) or (not np.isnan(r["err"]) and r["err"] > 4))
        me = f"{np.median(ee):.2f}px" if len(ee) else "--"
        print(f"    {bl:18} n={len(sub):3d}  det {100*dd/len(sub):3.0f}%  med_err {me:>7}  bad {100*bad/len(sub):3.0f}%")


def main() -> int:
    weights = sys.argv[1:] or ["C:/Users/Shadow/Peregrine/runs/pose/runs/vq2_pose_8kp/weights/best.pt"]
    vals = [l.strip() for l in (SETS / "val.txt").read_text().splitlines() if l.strip()]
    for w in weights:
        recs, fp, nn = analyze(w, vals); N = len(recs)
        errs = np.array([r["err"] for r in recs if r["det"] and not np.isnan(r["err"])])
        print(f"\n### {Path(w).name}: {N} GT gates | det {100*sum(r['det'] for r in recs)/N:.0f}% | "
              f"median inner-corner err {np.median(errs):.2f}px | FP on {nn} neg-imgs: {fp}")
        binrep(recs, "APPARENT SIZE (bbox w px = range proxy)", lambda r: r["w"],
               [("far  (<55)", lambda x: x < 55), ("mid  (55-120)", lambda x: 55 <= x < 120), ("near (>=120)", lambda x: x >= 120)])
        binrep(recs, "HUE SET", lambda r: r["set"], [("allhue(broad)", lambda x: x == "allhue"), ("vq1red", lambda x: x == "vq1red")])
        binrep(recs, "CLIPPING", lambda r: r["off"], [("fully-in", lambda x: x == 0), ("clipped(>=1 off)", lambda x: x >= 1)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
