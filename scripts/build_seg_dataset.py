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

# Census keys that count FRAMES or bookkeeping rather than per-gate conversion outcomes. Kept in
# one place because the success-rate print used to re-list them and the two lists drifted.
_NON_GATE_CENSUS = {"ok", "frames-kept", "frame-dropped", "duplicate-frame", "negative-kept",
                    "direct-area-frames", "direct-area-rows"}


def direct_seg_path(pose_label: Path) -> Path | None:
    """The AREA label written beside a pose label, or None. TWO layouts, both real:

      hand-labeler   <labels>/<stem>.txt          -> <labels>/seg/<stem>.txt
      blender render <root>/labels/<split>/x.txt  -> <root>/seg/<split>/x.txt

    Only the first was handled, which would have silently discarded every exact-corner seg label
    the re-render produces and re-derived them from the clamped pose rows instead -- dropping
    exactly the cropped gates the re-render exists to supply, with no error anywhere.

    These are authored, not derived: the writer clips the quads from the TRUE corners, so they
    carry the gates whose corners left the frame. Re-deriving those from the pose row is not merely
    lossy -- it drops them (the refit needs >= 4 in-frame keypoints).
    """
    beside = pose_label.parent / "seg" / pose_label.name
    if beside.is_file():
        return beside
    parts = list(pose_label.parts)                      # swap the LAST 'labels' component for 'seg'
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "labels":
            parts[i] = "seg"
            cand = Path(*parts)
            return cand if cand.is_file() else None
    return None


def seg_text_for(pose_label: Path, census: Counter):
    """(seg_text, used_direct). Prefers the authored area label; falls back to pose-row derivation."""
    direct = direct_seg_path(pose_label)
    if direct is not None:
        text = direct.read_text()
        census["direct-area-frames"] += 1
        census["direct-area-rows"] += len(text.splitlines())
        return text, True
    text, c = seg_label_from_pose_label(pose_label.read_text(), IMG_W, IMG_H)
    for k, v in c.items():
        census[k] += v
    return text, False


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
    """File-list mode: resolve each image's pose label, then run the SAME writer as --scan.

    (These were two near-identical loops that had already drifted apart once; one body now.)"""
    pairs = []
    for img in img_paths:
        lp = label_path_for(img)
        if lp.exists():
            pairs.append((img, lp))
        else:
            census["missing-pose-label"] += 1
    return convert_pairs(pairs, out_root, split, census)


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
                                if k not in _NON_GATE_CENSUS)
    if ok + bad:
        print(f"\ngates converted: {ok}/{ok + bad} = {100 * ok / (ok + bad):.1f}% (derived from "
              f"pose rows); {census['direct-area-rows']} area rows taken DIRECT from hand labels")
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
        text, _direct = seg_text_for(lp, census)
        # An EMPTY label is a true negative and must still be WRITTEN -- ultralytics reads a missing
        # file as "unlabelled image", which silently turns hard negatives into nothing at all. But a
        # frame whose gates all failed conversion is not a negative; dropping it is correct.
        if not raw.strip():
            census["negative-kept"] += 1
        elif not text.strip():
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
        # labels/seg/ holds AREA labels, not pose rows -- they are picked up via direct_seg_path()
        # from their pose sibling. Scanning them as pose labels would double-count every frame.
        if lp.parent.name in ("seg", "geom"):
            continue
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
                                if k not in _NON_GATE_CENSUS)
    if ok + bad:
        print(f"\ngates converted: {ok}/{ok + bad} = {100 * ok / (ok + bad):.1f}% (derived from "
              f"pose rows); {census['direct-area-rows']} area rows taken DIRECT from hand labels")
    print(f"wrote {out / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
