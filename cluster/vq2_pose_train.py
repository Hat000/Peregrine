"""Train the VQ2 8-keypoint YOLO-pose gate detector with on-the-fly augmentation (Ultralytics).

Augmentation strategy (see the 2026-06-15 research):
  * GEOMETRIC augmentation -> Ultralytics' BUILT-IN pipeline. It is pose-aware (transforms the 8
    keypoints with the image) and uses the data.yaml ``flip_idx`` to swap the inner/outer L<->R corner
    pairs on horizontal flip. Tuned below (rotation/translate/scale/shear/perspective/mosaic/flip).
  * PHOTOMETRIC / SENSOR realism -> a custom AlbumentationsX list passed via ``augmentations=``. These
    are the sim-to-real VQ-stream degradations (motion blur, JPEG, ISO/Gauss noise, defocus, downscale,
    brightness). They DO NOT move keypoints, so they are safe through the hook regardless of whether the
    Ultralytics Albumentations wrapper transforms pose keypoints on a given version.

The clean photoreal BASE renders carry the appearance diversity (HDRI venues, hue, props/people); this
augmentation adds the camera/sensor layer on top, fresh every epoch (no pre-baked multiply needed).

Requires: ultralytics >= 8.3 (the ``augmentations=`` train kwarg) and albumentationsx. CLI cannot pass
custom transforms -- that is why this is the Python API, not ``yolo pose train``.
"""
from __future__ import annotations

import argparse


def sensor_transforms():
    """Photometric/sensor AlbumentationsX transforms (keypoint-invariant) -> the sim-to-real layer."""
    import albumentations as A  # albumentationsx installs as the same 'albumentations' import
    return [
        A.MotionBlur(blur_limit=(3, 15), p=0.30),          # racing motion at speed
        A.Defocus(radius=(2, 6), p=0.08),
        A.MedianBlur(blur_limit=5, p=0.05),
        A.ImageCompression(quality_range=(25, 70), p=0.30),  # the VQ stream IS jpeg
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.20),
        A.GaussNoise(std_range=(0.04, 0.16), p=0.10),
        A.Downscale(scale_range=(0.4, 0.85), p=0.10),       # low-res / re-up of the stream
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.25),
        A.RandomGamma(gamma_limit=(85, 115), p=0.15),
    ]


# Built-in (pose-aware) geometric augmentation hyperparameters. The base already has roll/lighting
# diversity, so these are moderate -- enough motion/viewpoint jitter without destroying the gate.
BUILTIN_AUG = dict(
    hsv_h=0.015, hsv_s=0.6, hsv_v=0.4,                 # mild colour jitter (hue stays a nuisance axis)
    degrees=10.0, translate=0.1, scale=0.5, shear=2.0, perspective=0.0005,
    fliplr=0.5, flipud=0.0,                            # horizontal flip uses flip_idx; never vertical
    mosaic=1.0, close_mosaic=10, mixup=0.0, copy_paste=0.0,
    erasing=0.0,                                       # don't erase patches: could hide a keypoint
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="VQ2 8-keypoint YOLO-pose training (on-the-fly aug)")
    ap.add_argument("--data", required=True, help="merged data.yaml (cluster/vq2_pose_dataset.py)")
    ap.add_argument("--model", default="yolo11s-pose.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)   # champion = 16 (F-REPRO-1); 32 silently changes the run
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="runs")
    ap.add_argument("--name", default="vq2_pose_8kp")
    ap.add_argument("--seed", type=int, default=0)     # deterministic; ultralytics default is also 0
    ap.add_argument("--save-period", type=int, default=10, help="snapshot every N epochs (good-fix peak is narrow)")
    ap.add_argument("--resume", action="store_true",
                    help="resume from <project>/<name>/weights/last.pt (ShadowPC closeout survival)")
    ap.add_argument("--no-sensor-aug", action="store_true", help="disable the AlbumentationsX hook")
    ap.add_argument("--precision-loss", action="store_true",
                    help="LEVER 2: tighter inner-4 OKS sigma + area-un-normalized kpt term")
    ap.add_argument("--inner-sigma-scale", type=float, default=0.5, help="inner-4 sigma multiplier (<1 tighter)")
    ap.add_argument("--l1-weight", type=float, default=0.05, help="area-un-normalized distance term weight")
    args = ap.parse_args(argv)

    import pathlib
    from ultralytics import YOLO

    if args.precision_loss:   # LEVER 2: monkeypatch v8PoseLoss BEFORE train() (no effect on other runs)
        import sys as _sys
        _sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        from vq2_precision_loss import apply_precision_loss_patch
        apply_precision_loss_patch(args.inner_sigma_scale, args.l1_weight)

    run_dir = pathlib.Path(args.project) / args.name
    last_pt = run_dir / "weights" / "last.pt"
    done_marker = run_dir / "VQ2_DONE"
    # re-pass the AlbumentationsX hook on BOTH fresh and resume (it is a runtime object, never
    # serialized into args.yaml -- a resume without it would silently drop the load-bearing sensor aug).
    aug = None if args.no_sensor_aug else sensor_transforms()

    if args.resume and last_pt.exists():
        model = YOLO(str(last_pt))
        resume_kwargs = dict(resume=True)
        if aug is not None:
            resume_kwargs["augmentations"] = aug
        results = model.train(**resume_kwargs)
    else:
        model = YOLO(args.model)
        train_kwargs = dict(
            data=args.data, epochs=args.epochs, patience=args.patience, imgsz=args.imgsz,
            batch=args.batch, device=args.device, workers=8, plots=True, seed=args.seed,
            save_period=args.save_period,   # periodic snapshots -> select deploy ckpt by GOOD-FIX, not mAP
            project=args.project, name=args.name, exist_ok=True, **BUILTIN_AUG,
        )
        if aug is not None:
            train_kwargs["augmentations"] = aug   # photometric/sensor layer (ultralytics>=8.3)
        results = model.train(**train_kwargs)

    # NOTE: pass an ABSOLUTE --project so save_dir == <project>/<name> (ultralytics prefixes a
    # RELATIVE project with runs/<task>/, which would desync this run_dir from the real save_dir).
    done_marker.parent.mkdir(parents=True, exist_ok=True)
    done_marker.write_text("done\n")   # sentinel: lets the supervisor / agent-watchdog know to stop resuming
    print("VQ2_TRAIN_DONE", getattr(results, "save_dir", "?"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
