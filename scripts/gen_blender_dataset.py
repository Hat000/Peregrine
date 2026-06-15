"""Generate a VQ2 gate dataset on the LAPTOP with the procedural backend (no Blender).

This runs the full generator pipeline -- the real geometry/viewpoints, the keypoint-aware
albumentations aug, the YOLO-pose labels and layout -- with flat-shaded gate rings instead of a
Cycles render. Use it to (a) smoke-test a preset end-to-end, (b) produce the viz overlays that
prove the labels are correct, and (c) make a procedural baseline dataset. For the PHOTOREAL data
that actually drives VQ2 generalization, run ``render_entry.py`` under Blender on ShadowPC (same
labels, real pixels).

  python scripts/gen_blender_dataset.py --preset vq1_faithful --out data/vq2_proc/vq1_faithful \
      --n-train 200 --n-val 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from racer.vision.blender_gen.config import available_presets, load_preset
from racer.vision.blender_gen.dataset import ProceduralBackend, generate_dataset


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="vq1_faithful", help=f"one of {available_presets()}")
    ap.add_argument("--out", default="data/vq2_proc", help="output dataset dir")
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--n-val", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--track", default=None, help="track_map.json (default: baked VQ1 course)")
    ap.add_argument("--image-ext", default="png", choices=("png", "jpg"))
    args = ap.parse_args()

    preset = load_preset(args.preset)
    print(f"[proc] preset={preset.name} -> {args.out} (train={args.n_train} val={args.n_val})")
    yaml_path = generate_dataset(
        args.out, preset, ProceduralBackend(),
        n_train=args.n_train, n_val=args.n_val, seed=args.seed,
        image_ext=args.image_ext, track_path=args.track,
    )
    print(f"[proc] done. data.yaml -> {yaml_path}")
    print(f"[proc] verify labels: python scripts/visualize_blender_labels.py --dataset {args.out} --n 16")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
