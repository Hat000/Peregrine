"""Overlay YOLO-pose labels back onto the rendered images -- the primary label-correctness guard.

Draws each label's bbox + 4 keypoints (colour-coded by visibility: green=visible, yellow=occluded,
red=off-frame) with the corner index (0=LL 1=LR 2=UR 3=UL) next to each, and -- when the 4 corners
are visible -- runs PnP on the parsed keypoints and prints the recovered range so you can eyeball
that the on-disk label, the rendered gate and the estimator all agree. Works on ANY dataset in the
frozen YOLO-pose layout (procedural OR Blender output -- the labels are identical).

  python scripts/visualize_blender_labels.py --dataset data/vq2/appearance_broad --n 24
  python scripts/visualize_blender_labels.py --image foo.png --label foo.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

_VIS_BGR = {2: (0, 220, 0), 1: (0, 210, 210), 0: (0, 0, 230)}   # visible / occluded / off-frame
_CORNER = {0: "LL", 1: "LR", 2: "UR", 3: "UL"}


def _draw(img: np.ndarray, rows: list[str], run_pnp: bool) -> np.ndarray:
    H, W = img.shape[:2]
    vis = img.copy()
    for row in rows:
        f = row.split()
        if len(f) != 17:
            continue
        cx, cy, w, h = (float(f[i]) for i in (1, 2, 3, 4))
        x0, y0 = int((cx - w / 2) * W), int((cy - h / 2) * H)
        x1, y1 = int((cx + w / 2) * W), int((cy + h / 2) * H)
        cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 160, 0), 1)
        kp = []
        for i in range(4):
            px, py = float(f[5 + 3 * i]) * W, float(f[6 + 3 * i]) * H
            v = int(f[7 + 3 * i])
            kp.append((px, py, v))
            cv2.circle(vis, (int(px), int(py)), 4, _VIS_BGR.get(v, (255, 255, 255)), -1)
            cv2.putText(vis, f"{i}:{_CORNER[i]}", (int(px) + 5, int(py) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, _VIS_BGR.get(v, (255, 255, 255)), 1)
        if run_pnp and all(v == 2 for *_, v in kp):
            rng = _pnp_range(np.array([[p[0], p[1]] for p in kp]))
            if rng is not None:
                cv2.putText(vis, f"PnP {rng:.1f}m", (x0, max(12, y0 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return vis


def _pnp_range(corners_px: np.ndarray) -> float | None:
    from racer.contracts import GateObservation
    from racer.vision.gate_pose import estimate_gate_pose

    gp = estimate_gate_pose(GateObservation(frame_id=0, sim_time_ns=0, corners_px=corners_px))
    return None if gp is None else gp.range_m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=None, help="dataset dir (uses images/train + labels/train)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--image", default=None, help="single image (with --label)")
    ap.add_argument("--label", default=None, help="single label .txt")
    ap.add_argument("--n", type=int, default=16, help="how many to overlay from a dataset")
    ap.add_argument("--out", default=None, help="output dir (default: <dataset>/_viz)")
    ap.add_argument("--no-pnp", action="store_true", help="skip the PnP range readout")
    args = ap.parse_args()

    pairs: list[tuple[Path, Path]] = []
    if args.image and args.label:
        pairs = [(Path(args.image), Path(args.label))]
        out_dir = Path(args.out or ".")
    elif args.dataset:
        ds = Path(args.dataset)
        img_dir, lbl_dir = ds / "images" / args.split, ds / "labels" / args.split
        imgs = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
        for img in imgs[: args.n]:
            lbl = lbl_dir / f"{img.stem}.txt"
            if lbl.exists():
                pairs.append((img, lbl))
        out_dir = Path(args.out or (ds / "_viz"))
    else:
        ap.error("pass --dataset OR (--image and --label)")

    out_dir.mkdir(parents=True, exist_ok=True)
    for img_path, lbl_path in pairs:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  skip (unreadable): {img_path}")
            continue
        rows = lbl_path.read_text().splitlines()
        vis = _draw(img, rows, run_pnp=not args.no_pnp)
        dst = out_dir / f"{img_path.stem}_overlay.png"
        cv2.imwrite(str(dst), vis)
        print(f"  {img_path.name}: {len(rows)} gate(s) -> {dst}")
    print(f"[viz] wrote {len(pairs)} overlay(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
