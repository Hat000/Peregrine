"""Rank gate-seg checkpoints by what they are FOR: the gate centre the line solver gets out of them.

mask-mAP is the wrong selector. It scores region overlap, but this model's output is consumed by
homography_from_lines, which fits the mask's BOUNDARY -- a blobby mask can score well on mAP and
still give edges too ragged to fit. The pose model taught the same lesson from the other side
(val-mAP mispicked best.pt; the ep49 last.pt was the real champion), so select on the downstream
measurement.

Two numbers, and neither needs an external ground truth:

  COVERAGE  -- fraction of frames that yield a centre at all. Reported against the HSV mask
               baseline on the SAME frames, since replacing that mask is the entire point.
  AGREEMENT -- distance to the KEYPOINT path's centre, restricted to frames where the keypoint path
               is at its most trustworthy (a full, measured 4-corner fix, no rescue). There the
               keypoint centre is worth ~0.008-0.022 m, so it is a fair reference. This is a
               CROSS-CHECK, not truth: the two paths share no front-end, so agreement is real
               evidence, but a disagreement says only that they differ, not which is wrong.

Coverage is measured on frames where the keypoint path FAILS -- recall there is the reason this
path exists, and it is where no reference is available by construction.

Usage:
  python scripts/eval_gate_seg.py --weights runs/segment/gate_seg/weights/best.pt \
                                  --frames C:/Users/Shadow/vq2_label_inbox_2026-07-21/frames
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from racer.contracts import Frame  # noqa: E402
from racer.vision.gate_lines import SegGateLineDetector, centre_from_lines  # noqa: E402
from racer.vision.gate_pose import estimate_gate_pose  # noqa: E402

DEFAULT_FRAMES = "C:/Users/Shadow/vq2_label_inbox_2026-07-21/frames"


def keypoint_centres(detector, frame):
    """(full_fix_centres_px, any_fix) from the keypoint path. A 'full fix' is a MEASURED 4-corner
    observation -- no partial rescue -- which is the only case trustworthy enough to compare against."""
    full, any_fix = [], False
    for obs in detector.detect(frame):
        pose = estimate_gate_pose(obs, compute_covariance=False)
        if pose is None or not np.isfinite(pose.t_cam_gate).all():
            continue
        any_fix = True
        if getattr(obs, "derived_corners", False) or int(pose.n_corners) < 4:
            continue
        c = np.asarray(obs.corners_px, float)
        if c.shape == (4, 2):
            full.append(c.mean(axis=0))
    return full, any_fix


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="gate-seg .pt / .engine to score")
    ap.add_argument("--frames", default=DEFAULT_FRAMES)
    ap.add_argument("--kpt-weights", default=None,
                    help="pose weights for the cross-check arm; omit to skip AGREEMENT")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    paths = sorted(glob.glob(str(Path(args.frames) / "*.png")))
    if args.limit:
        paths = paths[:args.limit]
    imgs = [(p, cv2.imread(p)) for p in paths]
    imgs = [(p, im) for p, im in imgs if im is not None]
    print(f"{len(imgs)} frames from {args.frames}\n")

    seg = SegGateLineDetector.load(args.weights)
    kpt = None
    if args.kpt_weights:
        from racer.vision.detector import GateDetector
        kpt = GateDetector.load(args.kpt_weights)

    n_hsv = n_seg = 0
    n_kpt_fail = n_seg_on_kpt_fail = 0
    agree = []
    for i, (p, im) in enumerate(imgs):
        hsv = centre_from_lines(im)
        n_hsv += hsv is not None
        seg_hits = seg.centres(im)
        n_seg += bool(seg_hits)

        if kpt is None:
            continue
        full, any_fix = keypoint_centres(kpt, Frame(frame_id=i, sim_time_ns=i, image_bgr=im))
        if not any_fix:
            n_kpt_fail += 1
            n_seg_on_kpt_fail += bool(seg_hits)
        for c in full:                       # nearest seg centre to each trusted keypoint centre
            if seg_hits:
                agree.append(min(float(np.linalg.norm(c - s[0])) for s in seg_hits))

    n = len(imgs)
    print(f"COVERAGE   HSV mask  : {n_hsv:4d}/{n} ({100 * n_hsv / n:5.1f}%)   <- the baseline being replaced")
    print(f"           seg model : {n_seg:4d}/{n} ({100 * n_seg / n:5.1f}%)")
    if kpt is not None:
        if n_kpt_fail:
            print(f"\n  on the {n_kpt_fail} frames where the KEYPOINT path finds nothing, "
                  f"seg recovers {n_seg_on_kpt_fail} ({100 * n_seg_on_kpt_fail / n_kpt_fail:.1f}%)")
        if agree:
            a = np.array(agree)
            print(f"\nAGREEMENT vs trusted 4-corner keypoint centres (n={len(a)}): "
                  f"med {np.median(a):.1f} px, p90 {np.percentile(a, 90):.1f} px")
        else:
            print("\nAGREEMENT: no frame had both a trusted keypoint fix and a seg centre")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
