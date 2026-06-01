"""Generate a procedural synthetic YOLO-pose gate dataset (curriculum levels 1-3).

Examples:
  python scripts/gen_synthetic_dataset.py data/synthetic --level 2 --n-train 2000 --n-val 200
  python scripts/gen_synthetic_dataset.py data/synthetic --all-levels      # writes l1/ l2/ l3/

Levels: 1 = clean geometry (high contrast, low distraction); 2 = + directional shading,
colour, noise, clutter, occlusion; 3 = + heavy chaos (motion blur, JPEG, glare, dropout).
Geometry (full perspective + transit/clipping/occlusion 3-corner cases) is present at every
level. Produces images/{train,val}, labels/{train,val}, and data.yaml ready for
`yolo pose train data=<out>/data.yaml ...`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from racer.vision.synthetic import write_dataset


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", nargs="?", default="data/synthetic", help="output dir (default: data/synthetic)")
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--level", type=int, choices=(1, 2, 3), default=2,
                    help="curriculum level (ignored with --all-levels)")
    ap.add_argument("--all-levels", action="store_true",
                    help="write l1/ l2/ l3/ subdatasets under <out> for curriculum training")
    ap.add_argument("--mix", action="store_true",
                    help="ONE mixed-difficulty set: L1/L2/L3 sampled per image (weighted to harder)")
    args = ap.parse_args()

    if args.mix:
        weights = [1, 2, 2, 3, 3]   # L1 x1, L2 x2, L3 x2 -> mostly realistic + chaos
        print(f"[mix] generating {args.n_train} train + {args.n_val} val (L1/L2/L3) -> {args.out}")
        yaml_path = write_dataset(args.out, args.n_train, args.n_val, level=weights, seed=args.seed)
        print(f"   data.yaml -> {yaml_path}")
        return 0

    levels = (1, 2, 3) if args.all_levels else (args.level,)
    for lvl in levels:
        out = str(Path(args.out) / f"l{lvl}") if args.all_levels else args.out
        print(f"[L{lvl}] generating {args.n_train} train + {args.n_val} val -> {out}")
        yaml_path = write_dataset(out, args.n_train, args.n_val, level=lvl, seed=args.seed + lvl)
        print(f"   data.yaml -> {yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
