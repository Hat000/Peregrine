"""ShadowPC entrypoint: render a photoreal YOLO-pose dataset inside Blender.

Run:
  blender --background --python src/racer/vision/blender_gen/render_entry.py -- \
      --preset appearance_broad --out data/vq2/appearance_broad --n-train 4000 --n-val 400 --seed 0

Everything after the standalone ``--`` is parsed here (Blender swallows the rest). This wires the
Blender Cycles backend into the SHARED ``dataset.generate_dataset`` -- identical geometry, augment,
labels and layout as the laptop procedural run, only the pixels differ. The first thing it does is
the intrinsics self-check (project known 3D points through the real Blender camera vs K); a result
> 1 px means the camera is misconfigured and the labels would be wrong, so it aborts.

NOTE (ShadowPC setup): Blender ships its own Python. ``racer`` (this repo's ``src``) is added to
sys.path here, but numpy / opencv-python (cv2) / albumentations / scipy must be importable by
BLENDER's Python -- install them into it once (see RUN_GUIDE.md) or point Blender at a venv.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

_SRC = Path(__file__).resolve().parents[3]          # .../src
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Blender ships its OWN Python and launches it in ISOLATED mode (sys.flags.isolated == 1), which
# STRIPS PYTHONPATH from sys.path and disables user-site -- so pip-installed deps (cv2 / scipy /
# albumentations) are invisible even though numpy ships with Blender. os.environ['PYTHONPATH'] is
# still readable, so we re-honour it here: point PYTHONPATH at the dir holding those packages
# before launching (see RUN_GUIDE.md). No machine-specific path lives in the code.
for _extra in os.environ.get("PYTHONPATH", "").split(os.pathsep):
    if _extra and _extra not in sys.path:
        sys.path.append(_extra)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="VQ2 photoreal dataset render (Blender)")
    ap.add_argument("--preset", required=True, help="preset name (presets/<name>.json) or path")
    ap.add_argument("--out", required=True, help="output dataset dir")
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--track", default=None, help="track_map.json (default: baked VQ1 course)")
    ap.add_argument("--image-ext", default="png", choices=("png", "jpg"))
    ap.add_argument("--engine", default=None, choices=("CYCLES", "BLENDER_EEVEE_NEXT"),
                    help="override the preset render engine (EEVEE Next = fast preview)")
    ap.add_argument("--samples", type=int, default=None, help="override Cycles samples")
    ap.add_argument("--eevee", action="store_true", help="shortcut for --engine BLENDER_EEVEE_NEXT")
    ap.add_argument("--max-intrinsics-err-px", type=float, default=1.0,
                    help="abort if the Blender camera disagrees with K by more than this")
    ap.add_argument("--seg", action="store_true",
                    help="also emit 2-class YOLO-seg labels to seg/<split>/*.txt, built from the "
                         "EXACT projected corners (0 gate_frame = outer square, 1 gate_opening = "
                         "inner). Clipped to frame, never clamped.")
    ap.add_argument("--masks", action="store_true",
                    help="also emit per-frame gate-ring instance masks to masks/<split>/*.png")
    ap.add_argument("--no-augment", action="store_true",
                    help="render CLEAN base frames (disable the in-line albumentations pass) -- use when "
                         "the augmentation will be applied later as an offline multiplier")
    return ap.parse_args(argv)


def main() -> int:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    args = _parse_args(argv)

    from racer.vision.blender_gen.backends.blender import BlenderBackend
    from racer.vision.blender_gen.config import load_preset
    from racer.vision.blender_gen.dataset import generate_dataset

    preset = load_preset(args.preset)
    engine = "BLENDER_EEVEE_NEXT" if args.eevee else args.engine
    render_over = {}
    if engine is not None:
        render_over["engine"] = engine
    if args.samples is not None:
        render_over["samples"] = args.samples
    if render_over:
        preset = replace(preset, render=replace(preset.render, **render_over))
    if args.no_augment:
        preset = replace(preset, augment=replace(preset.augment, enable=False))

    print(f"[vq2] preset={preset.name} engine={preset.render.engine} samples={preset.render.samples} "
          f"-> {args.out}  (train={args.n_train} val={args.n_val})")

    backend = BlenderBackend(preset)
    err = backend.intrinsics_error_px()
    print(f"[vq2] intrinsics self-check: max reprojection error = {err:.4f} px")
    if err > args.max_intrinsics_err_px:
        print(f"[vq2] ABORT: camera intrinsics disagree with K by {err:.3f} px > "
              f"{args.max_intrinsics_err_px} px -- labels would be wrong. Fix bpy_camera before rendering.")
        return 2

    yaml_path = generate_dataset(
        args.out, preset, backend,
        n_train=args.n_train, n_val=args.n_val, seed=args.seed,
        image_ext=args.image_ext, track_path=args.track, emit_masks=args.masks, emit_seg=args.seg,
    )
    print(f"[vq2] done. data.yaml -> {yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
