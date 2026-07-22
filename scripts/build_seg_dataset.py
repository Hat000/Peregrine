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
        # 'frames' as well as 'images': the hand-labeler inbox uses frames/, and tagging those by
        # the bare directory name gave every such corpus the SAME tag "frames" -- two of them would
        # then collide on identical stems and be silently dropped as duplicates.
        if parts[i].lower() in ("images", "frames") and i > 0:
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


def build_scanned(args) -> int:
    """Recycle EVERY labelled frame under the given roots into one seg dataset."""
    out = Path(args.out)
    census, per_ds = Counter(), {}
    pairs = []
    for root in args.scan:
        r = Path(root)
        p = scan_pairs(r)
        per_ds[r.name] = len(p)
        pairs.extend(p)
    # de-dup: the same frame can be reachable from two roots
    seen, uniq = set(), []
    for img, lp in pairs:
        key = (dataset_tag(img), img.stem)
        if key in seen:
            census["duplicate-frame"] += 1
            continue
        seen.add(key)
        uniq.append((img, lp))

    held = set()
    if args.val_from and Path(args.val_from).exists():
        for ln in Path(args.val_from).read_text().splitlines():
            if ln.strip():
                held.add(Path(ln.strip()).stem)

    # deterministic split, and anything previously held out STAYS held out
    train, val = [], []
    for i, (img, lp) in enumerate(uniq):
        name = f"{dataset_tag(img)}__{img.stem}"
        (val if (name in held or img.stem in held or (i % int(1 / max(args.val_frac, 1e-6))) == 0)
         else train).append((img, lp))

    for split, items in (("train", train), ("val", val)):
        kept = convert_pairs(items, out, split, census)
        (out / f"{split}.txt").write_text("\n".join(str(p) for p in kept) + "\n", encoding="utf-8")
        print(f"  {split}: {len(kept)}/{len(items)} frames kept")

    names = "\n".join(f"  {k}: {v}" for k, v in sorted(CLASS_NAMES.items()))
    (out / "data.yaml").write_text(
        f"path: {out}\ntrain: train.txt\nval: val.txt\nnames:\n{names}\n", encoding="utf-8")
    print("\nper-source frames found:")
    for k, v in sorted(per_ds.items(), key=lambda t: -t[1]):
        print(f"  {k:44s} {v}")
    print("\nconversion census:")
    for k, v in census.most_common():
        print(f"  {k:24s} {v}")
    ok, bad = census["ok"], sum(v for k, v in census.items()
                                if k not in ("ok", "frames-kept", "frame-dropped", "duplicate-frame",
                                             "negative-kept"))
    if ok + bad:
        print(f"\ngates converted: {ok}/{ok + bad} = {100 * ok / (ok + bad):.1f}%")
    print(f"wrote {out / 'data.yaml'}")
    return 0


def convert_pairs(items, out_root: Path, split: str, census: Counter):
    """Like convert_split but taking explicit (image, label) pairs."""
    out_img, out_lbl = out_root / "images" / split, out_root / "labels" / split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    kept, used = [], set()
    for img, lp in items:
        raw = lp.read_text()
        text, c = seg_label_from_pose_label(raw, IMG_W, IMG_H)
        for k, v in c.items():
            census[k] += v
        if not raw.strip():
            census["negative-kept"] += 1          # a TRUE negative: keep it, with an empty label
        elif not text and c and not c.get("ok"):
            census["frame-dropped"] += 1
            continue
        name = f"{dataset_tag(img)}__{img.stem}"
        if name in used:
            census["duplicate-frame"] += 1
            continue
        used.add(name)
        _link_or_copy(img, out_img / (name + img.suffix))
        (out_lbl / (name + ".txt")).write_text(text, encoding="utf-8")
        kept.append(out_img / (name + img.suffix))
        census["frames-kept"] += 1
    return kept


def scan_pairs(root: Path):
    """Every (image, pose-label) pair under ``root``, found by walking labels/ and matching images.

    The file-list mode only sees what a given pose dataset's train.txt happened to include; several
    labelled corpora were never referenced by one. This walks the tree instead so nothing labelled
    goes unused. Negative labels (empty files) are KEPT -- they are what stops a seg model
    hallucinating gates, and they are 779 of the 3675 labels on this box."""
    out = []
    for lp in sorted(set(list(root.rglob("labels/**/*.txt")) + list(root.glob("labels/*.txt")))):
        # The hand-labeler's inbox layout is labels/ + frames/, the render pipeline's is
        # labels/ + images/. Try both, then the label's own directory, or the failure is SILENT:
        # a whole corpus scans to zero and you never learn it was skipped.
        found = None
        for img_dir in ("images", "frames"):
            parts = list(lp.parts)
            for i in range(len(parts) - 1, -1, -1):
                if parts[i].lower() == "labels":
                    parts[i] = img_dir
                    break
            else:
                continue
            # Build the name by CONCATENATION, never Path.with_suffix: the failure-mined frames are
            # named like "..._07.25_crash_..._f330.txt", and with_suffix operates on the LAST dot
            # segment, so it rewrote that to "..._07.png" and the whole corpus scanned to zero.
            d = Path(*parts).parent
            for ext in (".png", ".jpg", ".jpeg"):
                cand = d / (lp.stem + ext)
                if cand.exists():
                    found = cand
                    break
            if found:
                break
        if found is None:
            # the hand-labeled sets keep images in the dataset ROOT beside labels/, and a few keep
            # them in the label directory itself
            for base in (lp.parent, lp.parent.parent):
                for ext in (".png", ".jpg", ".jpeg"):
                    cand = base / (lp.stem + ext)
                    if cand.exists():
                        found = cand
                        break
                if found:
                    break
        if found is not None:
            out.append((found, lp))
    return sorted(set(out))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="pose dataset root (holds data.yaml + train/val .txt)")
    ap.add_argument("--scan", nargs="*", default=None,
                    help="dataset roots to WALK for images/+labels/ pairs (recycles every label, "
                         "including corpora no train.txt ever referenced)")
    ap.add_argument("--val-from", default=None,
                    help="an existing val.txt; those frames stay in VAL so a rebuild cannot leak "
                         "previously-held-out frames into train")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--out", required=True, help="output seg dataset root")
    args = ap.parse_args()
    if args.scan:
        return build_scanned(args)

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
