"""Train the two-class gate SEGMENTATION model that replaces the line solver's HSV mask.

Curriculum, and why it is shaped this way:

  STAGE 1 -- synthetic + real pose-derived labels (this script's default).
      Every existing 8-keypoint label converts to exact seg polygons for free
      (scripts/build_seg_dataset.py; 99.3% of 1728 real gate rows convert), so stage 1 costs no
      hand-labelling at all. Start from the COCO-pretrained yolo11n-seg: the gate is a large,
      high-contrast, low-variety object, so the small model is the right capacity and leaves GPU
      headroom -- the detector already shares one RTX 2000 Ada with the simulator, and the flight
      loop's budget is what killed the async-detect experiment.

  STAGE 2 -- failure-mined frames, hand-verified.
      The 998-frame inbox is mined from real flights where the KEYPOINT path failed, so it is
      exactly the distribution the second path exists for. Convert whatever is labelled, eyeball the
      overlays, fix in the labeler, retrain from the stage-1 weights at a lower LR.

  SELECT ON THE SOLVER, NOT ON mAP. mask-mAP measures overlap; what this model is for is feeding
  homography_from_lines, and a mask can score well while its BOUNDARY is too ragged to fit clean
  edges. Rank checkpoints with scripts/eval_gate_seg.py (centre error through the real solver on
  the task2 GT bundle). This is the same lesson as the pose model, where val-mAP mispicked best.pt
  and the ep49 last.pt was the actual champion.

  KEEP THE KEYPOINT PATH. This is an independent second emitter, not a replacement. Nothing here
  is wired into flight.

Usage:
  python scripts/build_seg_dataset.py --src <pose-dataset> --out C:/Users/Shadow/vq2_seg_2026-07-22
  python scripts/train_gate_seg.py --data C:/Users/Shadow/vq2_seg_2026-07-22/data.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

# 640x360 letterboxes to 640; the pose model trains at 640 and the deployed TRT engine is 384x640,
# so matching it here keeps the export path identical.
DEFAULT_IMGSZ = 640


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="seg data.yaml from build_seg_dataset.py")
    ap.add_argument("--model", default="yolo11n-seg.pt",
                    help="base weights; pass a stage-1 run's best.pt to do stage 2")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr0", type=float, default=None,
                    help="override initial LR; use ~0.2x the default when fine-tuning for stage 2")
    ap.add_argument("--name", default="gate_seg")
    ap.add_argument("--device", default=0)
    args = ap.parse_args()

    from ultralytics import YOLO

    kw = dict(
        data=str(Path(args.data)),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        # The gate's colour IS its signal and the warehouse is dark, so keep the photometric
        # augmentation that made the pose model's sensor-aug load-bearing (good-fix 80% -> 22%
        # without it), but do NOT flip vertically: gates are near-upright and a vertical flip
        # invents a pose the course never shows.
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        fliplr=0.5, flipud=0.0,
        degrees=10.0, translate=0.1, scale=0.5, shear=2.0,
        # Mosaic manufactures the CROPPED gates this path exists to serve, so leave it on -- but
        # close it for the last epochs so the model finishes on undistorted geometry.
        mosaic=1.0, close_mosaic=10,
        # Overlapping instances matter here: gate_opening sits INSIDE gate_frame by construction.
        overlap_mask=False,
    )
    if args.lr0 is not None:
        kw["lr0"] = args.lr0

    model = YOLO(args.model)
    model.train(**kw)
    print("\nNEXT: rank checkpoints through the SOLVER, not mask-mAP:")
    print("  python scripts/eval_gate_seg.py --weights runs/segment/"
          f"{args.name}/weights/best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
