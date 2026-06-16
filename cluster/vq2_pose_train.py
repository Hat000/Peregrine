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
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="runs")
    ap.add_argument("--name", default="vq2_pose_8kp")
    ap.add_argument("--no-sensor-aug", action="store_true", help="disable the AlbumentationsX hook")
    args = ap.parse_args(argv)

    from ultralytics import YOLO
    model = YOLO(args.model)

    train_kwargs = dict(
        data=args.data, epochs=args.epochs, patience=args.patience, imgsz=args.imgsz,
        batch=args.batch, device=args.device, workers=8, plots=True,
        project=args.project, name=args.name, exist_ok=True, **BUILTIN_AUG,
    )
    if not args.no_sensor_aug:
        train_kwargs["augmentations"] = sensor_transforms()   # photometric/sensor layer (ultralytics>=8.3)

    results = model.train(**train_kwargs)
    print("VQ2_TRAIN_DONE", getattr(results, "save_dir", "?"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
