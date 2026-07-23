"""Train "M+1" -- the 5-keypoint gate model: inner LL,LR,UR,UL + the gate CENTRE.

WHY 5 KEYPOINTS, NOT 8. The old model M regresses 8 corners and the centre is DERIVED from them
(diagonal intersection / PnP). That derivation dies close-up: when the gate fills the view its
corners leave the frame and ultralytics clamps them to the border (v=0), so the derived centre is
poisoned in exactly the regime that matters -- measured 55.7% of real gate views have the centre
in-frame while >=1 corner is cropped. M+1 regresses the CENTRE DIRECTLY as its own keypoint, so it
emits even when every corner is gone, as long as the centre itself is in-frame. The 4 inner corners
stay because, WHEN visible, they give the range-adjusted apparent area the RL obs needs
(rel_pos = range x bearing; range from the un-foreshortened inner square). The outer 4 are dropped:
M no longer solves the centre from them, and they were the least-confident, most-clamped keypoints.

The centre is the best-conditioned keypoint of the five -- it is the centroid, so it averages out
per-corner noise, and it is in-frame far more often than any single corner -- so it should be the
EASIEST to learn, not the hardest.

FLIP SAFETY. fliplr=0.5 is load-bearing photometric-adjacent augmentation, but a horizontal flip
MIRRORS the gate, so the keypoints must be relabelled: LL<->LR, UL<->UR, centre->centre. That is
flip_idx [1,0,3,2,4], which the dataset's data.yaml MUST carry (build_m1_dataset writes it). If it
is missing, a flip silently trains the corners to the wrong identity -- the same class of bug that
cost days on the pose sign. This script asserts the data.yaml has the right kpt_shape + flip_idx
before spending a GPU-hour on it.

SELECT ON THE REAL METRIC, NOT pose-mAP. What this model is for is emitting a gate centre the RL can
steer to. Rank checkpoints by centre error on the REAL failure-mined inbox, measured specifically
against the population the already-shipped partial-corner RESCUE still drops (fully-cropped /
one-edge / <4-keypoint views) -- NOT against raw M, and NOT on synthetic fixtures. mAP mispicked the
pose champion once already (val-mAP chose best.pt; ep49 last.pt was the real winner), and this repo
has been burned by a change that passed unit tests while a real metric regressed 12x.

Usage:
  python scripts/build_m1_dataset.py --out C:/Users/Shadow/vq2_m1_2026-07-23   (writes data.yaml)
  python scripts/train_m1.py --data C:/Users/Shadow/vq2_m1_2026-07-23/data.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

# 640x360 letterboxes to 640; M trains at 640 and the deployed TRT engine is 384x640, so matching
# it here keeps the fp16 export path identical to M's (parity was 0.01 px on the corner keypoints).
DEFAULT_IMGSZ = 640

EXPECTED_KPT_SHAPE = [5, 3]
EXPECTED_FLIP_IDX = [1, 0, 3, 2, 4]     # LL<->LR, UR<->UL, centre->centre


def _assert_contract(data_yaml: Path) -> None:
    """Fail LOUDLY before training if the dataset's keypoint contract is not the M+1 one.

    A wrong flip_idx does not error -- it trains silently on mirror-mislabelled corners, and you only
    find out when the deployed model's left/right is backwards. Cheap to check, expensive to miss."""
    import yaml

    cfg = yaml.safe_load(data_yaml.read_text())
    ks = list(cfg.get("kpt_shape", []))
    fi = list(cfg.get("flip_idx", []))
    if ks != EXPECTED_KPT_SHAPE:
        raise SystemExit(f"kpt_shape in {data_yaml} is {ks}, expected {EXPECTED_KPT_SHAPE} "
                         "(inner LL,LR,UR,UL + centre). Rebuild with build_m1_dataset.py.")
    if fi != EXPECTED_FLIP_IDX:
        raise SystemExit(f"flip_idx in {data_yaml} is {fi}, expected {EXPECTED_FLIP_IDX}. "
                         "fliplr augmentation would mislabel the mirrored corners -- refusing to train.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="M+1 pose data.yaml from build_m1_dataset.py")
    ap.add_argument("--model", default="yolo11n-pose.pt",
                    help="base weights (COCO pose); the 5-kpt head reinits, the backbone transfers. "
                         "Pass a stage-1 M+1 best.pt to fine-tune on hand-verified frames.")
    # MEASURED 2026-07-23: a 100-epoch run PEAKED AT EPOCH 10 (pose mAP50 0.503 on the REAL
    # hand-labelled val) and decayed to 0.282 by epoch 100 -- centre error 20.5 -> 35.6 px and
    # coverage 98% -> 84%. The training set is synthetic-only, so the extra 90 epochs learned
    # synthetic APPEARANCE, not gates. Until the render's domain matches the real dark warehouse,
    # long schedules are actively harmful here: keep the budget short and let patience stop it.
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=8,
                    help="early-stop after this many epochs with no val improvement (0 = off)")
    ap.add_argument("--close-mosaic", type=int, default=6,
                    help="disable mosaic for the last N epochs so training ends on clean geometry")
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr0", type=float, default=None,
                    help="override initial LR; ~0.2x the default when fine-tuning for stage 2")
    ap.add_argument("--name", default="gate_m1")
    ap.add_argument("--device", default=0)
    args = ap.parse_args()

    data_yaml = Path(args.data)
    _assert_contract(data_yaml)

    from ultralytics import YOLO

    kw = dict(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        # The gate's colour IS its signal and the warehouse is dark, so keep the photometric
        # augmentation that made M's sensor-aug load-bearing (good-fix 80% -> 22% without it).
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        # Horizontal flip only, with the flip_idx asserted above. NO vertical flip: gates are
        # near-upright and flipud invents a pose the course never shows.
        fliplr=0.5, flipud=0.0,
        degrees=10.0, translate=0.1, scale=0.5, shear=2.0,
        # Mosaic manufactures the CROPPED gates that are M+1's entire reason to exist -- a gate
        # sliced by a mosaic tile boundary is a free centre-in-frame-corners-out example. Close it
        # for the last epochs so training finishes on undistorted geometry.
        mosaic=1.0, close_mosaic=args.close_mosaic,
    )
    if args.patience:
        kw["patience"] = args.patience
    if args.lr0 is not None:
        kw["lr0"] = args.lr0

    model = YOLO(args.model)
    model.train(**kw)
    run = f"runs/pose/{args.name}"
    print("\nNEXT -- rank on the REAL metric, not pose-mAP:")
    print(f"  python scripts/eval_centre_ab.py --arms m1 --models {run}/weights/best.pt \\")
    print("      # centre error on the real inbox, vs the partial-corner RESCUE residual")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
