"""Overlay YOLO-pose labels on rendered frames + recover gate range via PnP (validation tool).

Pure-Python (cv2 + the deployed ``gate_pose`` estimator), NO Blender -- runs on the laptop against
ANY dataset dir produced by :mod:`racer.vision.blender_gen.dataset` (procedural or Blender backend).
For each labelled image it: denormalizes the on-disk YOLO-pose row, draws the bbox + the 4 inner
corners in canonical colour order (0=LL red, 1=LR green, 2=UR blue, 3=UL yellow) + the gate skeleton,
runs ``estimate_gate_pose`` on any 4-visible gate and annotates the recovered range. This is THE
visual arbiter that the rendered pixels, the on-disk labels and PnP all agree (the keypoints must
land on the rendered gate's inner-square corners; the recovered range must be sane, 2..30 m).

Run:
  python -m racer.vision.blender_gen.viz_overlay --data <dataset_dir> --split train --out <viz_dir> [--limit N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from racer.contracts import GateObservation          # noqa: E402
from racer.vision.gate_pose import estimate_gate_pose  # noqa: E402

from .contract import IMAGE_HEIGHT, IMAGE_WIDTH        # noqa: E402

# canonical corner order (contract): 0=lower-left, 1=lower-right, 2=upper-right, 3=upper-left.
_CORNER_BGR = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]  # LL red, LR grn, UR blu, UL yel
_CORNER_NAME = ["LL", "LR", "UR", "UL"]


def parse_pose_rows(text: str, w: int = IMAGE_WIDTH, h: int = IMAGE_HEIGHT) -> list[dict]:
    """Parse a YOLO-pose label file into per-gate dicts. Handles the VQ2 8-keypoint row (29 fields:
    inner 0..3 then outer 4..7) and the legacy 4-keypoint row (17 fields)."""
    gates = []
    for line in text.splitlines():
        f = line.split()
        n = (len(f) - 5) // 3
        if n not in (4, 8) or len(f) != 5 + 3 * n:
            continue
        cx, cy, bw, bh = (float(f[i]) for i in range(1, 5))
        kp = np.array([[float(f[5 + 3 * i]) * w, float(f[6 + 3 * i]) * h] for i in range(n)])
        vis = np.array([int(f[7 + 3 * i]) for i in range(n)])
        bbox = np.array([(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h])
        gates.append({"bbox": bbox, "kp": kp[:4], "vis": vis[:4],          # inner (PnP) keypoints
                      "outer": kp[4:8] if n == 8 else None})
    return gates


def draw_overlay(img: np.ndarray, gates: list[dict]) -> tuple[np.ndarray, list[dict]]:
    """Draw bbox + corners + skeleton on a copy of ``img``; run PnP per 4-visible gate.
    Returns (annotated_image, per_gate_pnp_results)."""
    out = img.copy()
    results = []
    for gi, g in enumerate(gates):
        kp, vis, bbox = g["kp"], g["vis"], g["bbox"]
        outer = g.get("outer")
        x, y, bw, bh = bbox
        cv2.rectangle(out, (int(x), int(y)), (int(x + bw), int(y + bh)), (200, 200, 200), 1)
        # outer keypoints (4..7): hollow rings + their quad, drawn first (under the inner markers)
        if outer is not None:
            for a in range(4):
                cv2.line(out, tuple(outer[a].astype(int)), tuple(outer[(a + 1) % 4].astype(int)), (90, 90, 90), 1)
            for ci in range(4):
                cv2.circle(out, tuple(outer[ci].astype(int)), 4, _CORNER_BGR[ci], 1)   # ring = outer
        # skeleton: connect the 4 INNER corners in order (closed quad)
        for a in range(4):
            b = (a + 1) % 4
            cv2.line(out, tuple(kp[a].astype(int)), tuple(kp[b].astype(int)), (160, 160, 160), 1)
        for ci in range(4):
            p = tuple(kp[ci].astype(int))
            cv2.circle(out, p, 4, _CORNER_BGR[ci], -1)                                  # filled = inner
            cv2.circle(out, p, 4, (0, 0, 0), 1)
        # PnP recovery (only meaningful with all 4 corners visible)
        res = {"gate_index": gi, "n_vis": int((vis == 2).sum()), "range_m": None, "reproj_px": None}
        if int((vis == 2).sum()) == 4:
            gp = estimate_gate_pose(GateObservation(frame_id=0, sim_time_ns=0, corners_px=kp))
            if gp is not None:
                res["range_m"] = float(gp.range_m)
                res["reproj_px"] = float(getattr(gp, "reprojection_error_px", np.nan))
                cen = kp.mean(axis=0).astype(int)
                cv2.putText(out, f"{gp.range_m:.1f}m", (cen[0] - 14, cen[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        results.append(res)
    return out, results


def run(data_dir: str, split: str, out_dir: str, limit: int | None = None) -> dict:
    data, out = Path(data_dir), Path(out_dir)
    img_dir, lbl_dir = data / "images" / split, data / "labels" / split
    out.mkdir(parents=True, exist_ok=True)
    imgs = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
    if limit:
        imgs = imgs[:limit]
    ranges, n_pnp, n_frames_labelled = [], 0, 0
    for ip in imgs:
        img = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        if img is None:
            continue
        lp = lbl_dir / f"{ip.stem}.txt"
        gates = parse_pose_rows(lp.read_text()) if lp.exists() else []
        ann, results = draw_overlay(img, gates)
        if gates:
            n_frames_labelled += 1
        for r in results:
            if r["range_m"] is not None:
                ranges.append(r["range_m"])
                n_pnp += 1
        cv2.imwrite(str(out / f"{ip.stem}_overlay.png"), ann)
        tag = "NEG(empty)" if not gates else " ".join(
            (f"g{r['gate_index']}:{r['range_m']:.1f}m" if r["range_m"] is not None
             else f"g{r['gate_index']}:{r['n_vis']}vis") for r in results)
        print(f"  {ip.name}: {tag}")
    summary = {
        "n_images": len(imgs), "n_frames_labelled": n_frames_labelled, "n_pnp_recovered": n_pnp,
        "range_min": (min(ranges) if ranges else None), "range_max": (max(ranges) if ranges else None),
        "range_mean": (float(np.mean(ranges)) if ranges else None),
    }
    print(f"[viz] {summary}")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Overlay VQ2 YOLO-pose labels + PnP range recovery")
    ap.add_argument("--data", required=True, help="dataset dir (with images/ + labels/)")
    ap.add_argument("--split", default="train", choices=("train", "val"))
    ap.add_argument("--out", required=True, help="dir for *_overlay.png")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    run(a.data, a.split, a.out, a.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
