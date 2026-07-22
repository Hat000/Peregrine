"""Build a YOLO-SEG dataset from the existing 8-keypoint POSE datasets.

No new hand-labelling: every pose label already pins the gate plane, and the two canonical squares
re-project through it (see racer.vision.seg_labels for why segmentation is the right target for
cropped gates, and why off-frame corners must be RECONSTRUCTED rather than read).

Input is the same file-list layout the pose datasets use -- a data.yaml with ``train:``/``val:``
pointing at .txt files of absolute image paths, with labels at the image path with /images/ ->
/labels/ and the extension swapped.

OUTPUT MIRRORS AN IMAGE TREE, and it must -- two earlier shortcuts were both silently wrong:

  * Pointing the new file lists at the ORIGINAL images does not work. Ultralytics locates a label by
    substituting /images/ -> /labels/ in the IMAGE path, so it would have read each frame's original
    POSE label and never opened the seg labels at all -- training on the wrong target with no error.
  * Naming outputs by image STEM collides. The pose lists mix several source datasets that each
    number frames from 000000, so 437 of 1572 stems collided and later labels overwrote earlier
    ones, silently pairing images with another dataset's geometry.

So each frame is re-keyed as ``<source-dataset>__<stem>`` and hard-linked into out/images/<split>/
(hardlinks cost no disk on NTFS; falls back to copy across volumes). Uniqueness is ASSERTED, not
assumed.

Usage:
  python scripts/build_seg_dataset.py --src C:/Users/Shadow/vq2_darkred_partial_2026-07-06 \
                                      --out C:/Users/Shadow/vq2_seg_2026-07-22
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from racer.vision.seg_labels import CLASS_NAMES, seg_label_from_pose_label  # noqa: E402

IMG_W, IMG_H = 640, 360


def label_path_for(img_path: Path) -> Path:
    """Pose-dataset convention: .../images/<split>/x.png -> .../labels/<split>/x.txt."""
    parts = list(img_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def read_list(p: Path) -> list[Path]:
    return [Path(ln.strip()) for ln in p.read_text().splitlines() if ln.strip()]


def dataset_tag(img_path: Path) -> str:
    """The source dataset's own directory name (the parent of ``images/``), used to disambiguate
    stems that several datasets reuse."""
    parts = list(img_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "images" and i > 0:
            return parts[i - 1]
    return img_path.parent.name


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        import os
        os.link(src, dst)                  # NTFS hardlink: no extra disk, no copy time
    except OSError:
        import shutil
        shutil.copy2(src, dst)             # different volume / filesystem without link support


def convert_split(img_paths, out_root: Path, split: str, census: Counter):
    """Write one seg label + one linked image per frame. Returns the new image paths."""
    out_img = out_root / "images" / split
    out_lbl = out_root / "labels" / split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    kept, used = [], set()
    for img in img_paths:
        lp = label_path_for(img)
        if not lp.exists():
            census["missing-pose-label"] += 1
            continue
        text, c = seg_label_from_pose_label(lp.read_text(), IMG_W, IMG_H)
        for k, v in c.items():
            census[k] += v
        # An EMPTY label is a true negative and must still be written -- ultralytics reads a missing
        # file as "unlabelled image", which silently turns hard negatives into nothing at all.
        # But a frame whose gates ALL failed conversion is not a negative; dropping it is correct.
        if not text and c and not c.get("ok"):
            census["frame-dropped"] += 1
            continue
        name = f"{dataset_tag(img)}__{img.stem}"
        assert name not in used, f"name collision after re-keying: {name} ({img})"
        used.add(name)
        _link_or_copy(img, out_img / (name + img.suffix))
        (out_lbl / (name + ".txt")).write_text(text, encoding="utf-8")
        kept.append(out_img / (name + img.suffix))
        census["frames-kept"] += 1
    return kept


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="pose dataset root (holds data.yaml + train/val .txt)")
    ap.add_argument("--out", required=True, help="output seg dataset root")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    census = Counter()

    lists = {}
    for split in ("train", "val"):
        lp = src / f"{split}.txt"
        if not lp.exists():
            print(f"  [skip] {lp} not found")
            continue
        imgs = read_list(lp)
        kept = convert_split(imgs, out, split, census)
        (out / f"{split}.txt").write_text("\n".join(str(p) for p in kept) + "\n", encoding="utf-8")
        lists[split] = (len(imgs), len(kept))
        print(f"  {split}: {len(kept)}/{len(imgs)} frames kept")

    names = "\n".join(f"  {k}: {v}" for k, v in sorted(CLASS_NAMES.items()))
    (out / "data.yaml").write_text(
        f"path: {out}\ntrain: train.txt\nval: val.txt\nnames:\n{names}\n", encoding="utf-8")

    print("\nper-gate conversion census:")
    for k, v in census.most_common():
        print(f"  {k:24s} {v}")
    ok, bad = census["ok"], sum(v for k, v in census.items()
                                if k not in ("ok", "frames-kept", "frame-dropped"))
    if ok + bad:
        print(f"\ngates converted: {ok}/{ok + bad} = {100 * ok / (ok + bad):.1f}%")
    print(f"wrote {out / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
