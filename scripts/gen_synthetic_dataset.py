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
    ap.add_argument("--max-gates", type=int, default=3,
                    help="max gates per image (>1 = multi-gate scenes; 1 = legacy single-gate)")
    ap.add_argument("--high-roll-prob", type=float, default=0.0,
                    help="fraction of gates forced to large |roll| (26-49 deg) -- the swap-prone tail")
    ap.add_argument("--edge-prob", type=float, default=0.0,
                    help="fraction of gates biased to a frame edge (a corner clips) -- the near-edge tail")
    ap.add_argument("--hard", action="store_true",
                    help="shortcut: --high-roll-prob 0.3 --edge-prob 0.3 (oversample the tail configs)")
    args = ap.parse_args()

    hr = 0.3 if args.hard else args.high_roll_prob
    ep = 0.3 if args.hard else args.edge_prob
    kw = dict(max_gates=args.max_gates, high_roll_prob=hr, edge_prob=ep)
    tag = f"<= {args.max_gates} gates, roll+{hr:.0%}, edge+{ep:.0%}"

    if args.mix:
        weights = [1, 2, 2, 3, 3]   # L1 x1, L2 x2, L3 x2 -> mostly realistic + chaos
        print(f"[mix] generating {args.n_train} train + {args.n_val} val (L1/L2/L3, {tag}) -> {args.out}")
        yaml_path = write_dataset(args.out, args.n_train, args.n_val, level=weights, seed=args.seed, **kw)
        print(f"   data.yaml -> {yaml_path}")
        return 0

    levels = (1, 2, 3) if args.all_levels else (args.level,)
    for lvl in levels:
        out = str(Path(args.out) / f"l{lvl}") if args.all_levels else args.out
        print(f"[L{lvl}] generating {args.n_train} train + {args.n_val} val ({tag}) -> {out}")
        yaml_path = write_dataset(out, args.n_train, args.n_val, level=lvl, seed=args.seed + lvl, **kw)
        print(f"   data.yaml -> {yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
