"""Generate a procedural synthetic YOLO-pose gate dataset.

Usage:  python scripts/gen_synthetic_dataset.py [out_dir] [n_train] [n_val] [seed]
        defaults: data/synthetic 2000 200 0

Produces images/{train,val}, labels/{train,val}, and data.yaml ready for
`yolo pose train data=<out_dir>/data.yaml ...`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from racer.vision.synthetic import write_dataset


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "data/synthetic"
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    n_val = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    print(f"generating {n_train} train + {n_val} val -> {out} (seed {seed})")
    yaml_path = write_dataset(out, n_train, n_val, seed)
    print(f"done. data.yaml at {yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
