"""Build a merged ultralytics YOLO-pose data.yaml + train/val split for the VQ2 photoreal sets.

The base render produced two sets with NO val split (all in images/train): all-hue (appearance_broad)
and VQ1-red (vq1_faithful). This merges them and carves a deterministic val hold-out, writing:

  <root>/data.yaml      (kpt_shape [8,3], flip_idx, train=train.txt, val=val.txt, path=<root>)
  <root>/train.txt      image paths relative to <root>
  <root>/val.txt

Run on the box that holds the data (ShadowPC now; re-run on the cluster after transfer if paths move):
  python cluster/vq2_pose_dataset.py --root handoff/vq2-blender-render-2026-06-15/sets --val-frac 0.1
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from racer.vision.blender_gen.contract import KEYPOINT_FLIP_IDX, N_KEYPOINTS  # noqa: E402

DATA_YAML = """\
# VQ2 photoreal gate dataset (8-keypoint YOLO-pose). Built by cluster/vq2_pose_dataset.py.
# Geometric augmentation is Ultralytics' built-in (pose-aware, uses flip_idx); photometric/sensor
# augmentation is the AlbumentationsX hook in vq2_pose_train.py. See that script.
path: {root}
train: train.txt
val: val.txt
names:
  0: gate
kpt_shape: [{nk}, 3]
flip_idx: {flip}
"""


def _images(set_dir: Path) -> list[Path]:
    d = set_dir / "images" / "train"
    return sorted(d.glob("*.png")) + sorted(d.glob("*.jpg")) if d.is_dir() else []


def build(root: Path, val_frac: float, seed: int) -> dict:
    sets = [p for p in sorted(root.iterdir()) if (p / "images" / "train").is_dir()] if root.is_dir() else []
    if not sets:
        raise SystemExit(f"no sets with images/train under {root}")
    rng = random.Random(seed)
    train, val = [], []
    per_set = {}
    for s in sets:
        imgs = _images(s)
        rng.shuffle(imgs)
        n_val = int(round(len(imgs) * val_frac))
        val += imgs[:n_val]
        train += imgs[n_val:]
        per_set[s.name] = {"total": len(imgs), "val": n_val, "train": len(imgs) - n_val}
    rng.shuffle(train)
    rng.shuffle(val)
    # ABSOLUTE, OS-native image paths. ultralytics resolves .txt entries against the CWD (not the
    # data.yaml `path`), and img2label_paths swaps os.sep+'images'+os.sep -> labels; absolute native
    # paths resolve unambiguously on any CWD/OS. The script runs WHERE the data lives (cluster sbatch
    # regenerates it post-transfer), so absolute paths are correct there.
    (root / "train.txt").write_text("\n".join(str(p.resolve()) for p in train) + "\n")
    (root / "val.txt").write_text("\n".join(str(p.resolve()) for p in val) + "\n")
    (root / "data.yaml").write_text(DATA_YAML.format(
        root=str(root.resolve()).replace("\\", "/"), nk=N_KEYPOINTS, flip=KEYPOINT_FLIP_IDX))
    return {"sets": per_set, "train": len(train), "val": len(val), "data_yaml": str(root / "data.yaml")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Merge VQ2 sets into a YOLO-pose data.yaml + split")
    ap.add_argument("--root", required=True, help="dir containing the per-set subdirs (e.g. sets/)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    print(build(Path(a.root), a.val_frac, a.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
