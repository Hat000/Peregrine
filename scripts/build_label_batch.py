"""Pool every labelling inbox into ONE batch, and re-open the labels that do not survive an audit.

Two inboxes were mined a day apart and are disjoint (crash approaches vs close-range failures), so
labelling them meant switching servers. This hardlinks both into one frames/ dir and decides, per
already-labelled frame, whether it can be trusted:

  * TRUSTED   -> its label (+ seg/geom sidecars) is copied into labels/, so it shows as done.
  * SUSPECT   -> its geometry is written into seeds/ INSTEAD, which makes the labeler re-open it
                 pre-filled and count it UNLABELED until a human saves again. Nothing is deleted;
                 the source inbox keeps its original file either way.

That reuses the seed mechanism as a review queue, so a doubtful label cannot silently stay in the
training set -- the invariant "a file in labels/ means a human approved it" survives, and here the
human is being asked to approve it a second time with better information.

WHAT MAKES A LABEL SUSPECT
  1. no geom sidecar AND a corner off-frame -- it predates the area path, so that corner was stored
     CLAMPED to the border and the polygon is wrong exactly where cropping matters most.
  2. the ring between its quads is not red / its opening is not see-through -- the same self-check
     the seeder applies to its own output, turned on the humans' (see seed_area_labels.ring_score).
  3. a confident machine detection sits where the label has no gate at all -- the missed-gate case
     Fengyou caught in the render gallery ("distant gate forgot to label the near gates").
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tools" / "gate_labeler"))

import labelio  # noqa: E402
import seed_area_labels as S  # noqa: E402
from racer.vision.gate_lines import gate_mask  # noqa: E402

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
IMG_W, IMG_H = 640, 360


def link(src: Path, dst: Path):
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def audit(frame: Path, gates, has_geom: bool, segdet, args):
    """[] if the label is trustworthy, else the reasons it is not."""
    bgr = cv2.imread(str(frame))
    if bgr is None:
        return ["unreadable"]
    red = gate_mask(bgr) > 0
    why = []
    if not gates:
        return why                       # an intentional NEGATIVE: nothing to second-guess
    for g in gates:
        inn = np.asarray(g["inner"], float)
        out = np.asarray(g["outer"], float)
        off = sum(1 for p in np.vstack([inn, out])
                  if not (0 <= p[0] <= IMG_W - 1 and 0 <= p[1] <= IMG_H - 1))
        if not has_geom and off:
            why.append(f"clamped-legacy({off}-corners-off-frame)")
        ring, see, _ = S.ring_score(inn, out, red)
        if ring < args.min_ring:
            why.append(f"ring={ring:.2f}")
        # ring_score reports see_through = 0 when the clipped opening is under 200 px, because for
        # SEEDING an unmeasurable opening should be rejected. Auditing a human's label is the
        # opposite situation: "too small to measure" is not evidence of a mistake, and flagging it
        # would accuse every correctly-labelled DISTANT gate. Only complain when it was measurable.
        from racer.vision.seg_labels import clip_polygon, polygon_area
        if polygon_area(clip_polygon(inn, IMG_W, IMG_H)) >= 200.0 and see < args.min_see:
            why.append(f"opening-not-see-through={see:.2f}")
    if segdet is not None:               # a confident detection where the human labelled nothing
        centres = [np.asarray(g["inner"], float).mean(axis=0) for g in gates]
        sizes = [max(np.ptp(np.asarray(g["inner"], float), axis=0).max(), 1.0) for g in gates]
        for sq in S.from_seg(segdet, bgr):
            if not S.sane(*sq):
                continue
            r, se, _ = S.ring_score(sq[0], sq[1], red)
            if r < 0.9 or se < 0.5:
                continue
            c = np.asarray(sq[0], float).mean(axis=0)
            if not any(np.linalg.norm(c - cc) < 0.6 * sz for cc, sz in zip(centres, sizes)):
                why.append("possible-missed-gate")
                break
    return why


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", nargs="+", required=True, help="inbox roots (frames/ + labels/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seg-weights", default=None)
    ap.add_argument("--min-ring", type=float, default=0.7)
    ap.add_argument("--min-see", type=float, default=0.4)
    args = ap.parse_args()

    out = Path(args.out)
    for sub in ("frames", "labels/seg", "labels/geom", "seeds/geom"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    segdet = None
    if args.seg_weights:
        from racer.vision.gate_lines import SegGateLineDetector
        segdet = SegGateLineDetector.load(args.seg_weights, conf=0.25)

    census, review = Counter(), {}
    for root in map(Path, args.src):
        fdir = root / "frames" if (root / "frames").is_dir() else root
        for f in sorted(p for p in fdir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
            stem = f.stem
            # Only the LINK is skippable on a re-run -- an early `continue` here also skipped the
            # label audit, so re-running after a fix silently produced an empty review queue.
            if not (out / "frames" / f.name).exists():
                link(f, out / "frames" / f.name)
            census["frames"] += 1

            lp = root / "labels" / f"{stem}.txt"
            if not lp.is_file():
                census["unlabelled"] += 1
                continue
            gp = root / "labels" / "geom" / f"{stem}.json"
            has_geom = gp.is_file()
            gates = (labelio.decode_geometry(json.loads(gp.read_text())) if has_geom
                     else labelio.decode_label(lp.read_text()))
            why = audit(f, gates, has_geom, segdet, args)
            if not why:
                shutil.copy2(lp, out / "labels" / f"{stem}.txt")
                for sub, ext in (("seg", ".txt"), ("geom", ".json")):
                    s = root / "labels" / sub / f"{stem}{ext}"
                    if s.is_file():
                        shutil.copy2(s, out / "labels" / sub / f"{stem}{ext}")
                census["label-trusted"] += 1
            else:
                # SUSPECT -> becomes a seed, so the labeler walks the human back through it.
                payload = [{"inner": g["inner"], "outer": g["outer"],
                            "occluded": [v == labelio.V_OCC for v in g["vis"]],
                            "outer_detached": g.get("outer_detached") or [True] * 4}
                           for g in gates]
                (out / "seeds" / f"{stem}.txt").write_text(labelio.encode_label(payload, IMG_W, IMG_H))
                (out / "seeds" / "geom" / f"{stem}.json").write_text(
                    json.dumps(labelio.encode_geometry(payload, IMG_W, IMG_H)))
                review[f.name] = {"source": "REVIEW your own label", "ring": 0.0,
                                  "see_through": 0.0, "kpts_in_frame": 0, "reasons": why}
                census["label-sent-for-review"] += 1
                for w in why:
                    census[f"  reason:{w.split('=')[0]}"] += 1

    (out / "_review.json").write_text(json.dumps(review, indent=1))
    print(f"\npooled -> {out}")
    for k, v in census.most_common():
        print(f"  {k:34s} {v}")
    print(f"\n{len(review)} existing labels queued for re-review; "
          f"run seed_area_labels.py next (it will not touch them or the trusted labels)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
