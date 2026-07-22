"""Pre-seed the hand-labeler so labelling is CORRECTION, not clicking from scratch.

Two sources, tried per frame, because they fail in opposite places:

  * **M** -- the 8-keypoint detector. Excellent when it fires, but this inbox was MINED from M's
    close-range failures, so on much of it there is nothing to seed from. Its off-frame keypoints
    arrive CLAMPED to the border by ultralytics, so those are unpinned and the squares are
    reconstructed through the gate-plane homography rather than read.
  * **lines** -- the HSV mask -> edge lines -> gate-plane homography path (racer.vision.gate_lines).
    A line is determined by the in-frame pixels it passes through, so it SURVIVES cropping: this is
    the source that works exactly where M does not. Its weakness is the mask, not the geometry.

Both end at a homography, so both give all 8 corners INCLUDING the off-frame ones -- which is the
whole point now that the label is the clipped area.

SELF-CHECK, NOT FAITH. Every candidate is scored by how much of the RING between its two quads is
actually red structure (:func:`ring_score`), and anything under ``--min-ring`` is not written. The
score rides along in the manifest and is shown in the labeler, so a weak seed announces itself
instead of looking authoritative -- a seed that is wrong in a *plausible* way costs more time than
no seed at all.

EVERY gate in a frame is emitted, not the best one. These frames routinely show the gate being flown
plus two more behind it, and all of them are labels.

SAFETY INVARIANT (unchanged): seeds go to a SEPARATE directory. A file in the labels dir means a
human approved it. The server serves a seed only when no human label exists, flags it
``seed: true``, and the frame stays UNLABELED until the human saves.

Usage:
  python scripts/seed_area_labels.py --frames <FRAMES> --seeds <SEEDS> [--weights <M.pt|.engine>]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "gate_labeler"))

import labelio  # noqa: E402
from racer.vision.gate_lines import extract_gate_lines, gate_mask, homography_from_lines  # noqa: E402
from racer.vision.gate_pose import gate_plane_homography, project_gate_squares  # noqa: E402
from racer.vision.seg_labels import MIN_VISIBLE_AREA_PX, clip_polygon, polygon_area  # noqa: E402

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
IMG_W, IMG_H = 640, 360
SANE_SPAN = 4.0          # a corner further out than this many image widths is not a real pixel
_CLAMP_EPS_PX = 0.51     # ultralytics clips keypoints to exactly [0,W]x[0,H]


# ---------------------------------------------------------------------------------------------
# corner identity
# ---------------------------------------------------------------------------------------------
# The line path's homography is only determined up to a SYMMETRY OF THE SQUARE: relabelling the
# edges replaces H with H@R, which leaves the centre (and the polygon, and therefore the AREA label)
# completely unchanged -- but permutes which corner is called LL. That matters only for the pose
# row, so resolve it with the gate's soft upright prior instead of pretending it is unknowable.
_EXPECT = np.array([[-1.0, 1.0], [1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]])   # LL,LR,UR,UL, y-DOWN
_SYMS = [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2],
         [3, 2, 1, 0], [0, 3, 2, 1], [1, 0, 3, 2], [2, 1, 0, 3]]


def canonicalise(inner: np.ndarray, outer: np.ndarray):
    """Re-index both squares so LL,LR,UR,UL point roughly down-left ... up-left in the IMAGE."""
    c = inner.mean(axis=0)
    d = inner - c
    n = np.linalg.norm(d, axis=1, keepdims=True)
    d = d / np.maximum(n, 1e-9)
    best = max(_SYMS, key=lambda p: float((d[p] * _EXPECT).sum()))
    return inner[best], outer[best]


# ---------------------------------------------------------------------------------------------
# scoring: does this seed actually sit on the gate in the picture?
# ---------------------------------------------------------------------------------------------
def frame_truth_mask(bgr):
    """(red structure | its opening) -- what an AMODAL gate_frame polygon should cover."""
    m = gate_mask(bgr) > 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n < 2:
        return None, None
    gid = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    gate = lab == gid
    filled = cv2.morphologyEx(gate.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, hier = cv2.findContours(filled, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    solid = np.zeros_like(filled)
    if len(cnts):
        cv2.drawContours(solid, [max(cnts, key=cv2.contourArea)], -1, 1, -1)
    interior = None
    if hier is not None:                       # the opening = the largest hole, and an INTERIOR point
        holes = [c for i, c in enumerate(cnts) if hier[0][i][3] != -1]
        if holes:
            big = max(holes, key=cv2.contourArea)
            mo = cv2.moments(big)
            if mo["m00"] > 1.0:
                interior = (mo["m10"] / mo["m00"], mo["m01"] / mo["m00"])
    return solid.astype(bool), interior


def _fill(poly):
    m = np.zeros((IMG_H, IMG_W), np.uint8)
    p = clip_polygon(np.asarray(poly, float), IMG_W, IMG_H)
    if len(p) >= 3:
        cv2.fillPoly(m, [p.round().astype(np.int32)], 1)
    return m.astype(bool)


def ring_score(inner_px, outer_px, red):
    """Does this quad pair look like a GATE? -> (score, ring_px). score = min(red ring, see-through opening).

    This replaced a global IoU against "all the red in the picture", which was measuring the wrong
    thing: a frame usually shows SEVERAL gates and every one of them should be labelled, so a seed
    that correctly found the far gate scored ~0 against a truth mask dominated by the near one.
    Rendered nine rejects and looked -- the scorer was the bug, not (only) the seeds.

    The ring is local, and it is the one region whose appearance the geometry actually predicts: a
    real gate's ring IS the red structure, while a quad lying across the floor or the racing line
    has a ring of whatever it happened to cover. Scored on the ring rather than the filled outer
    square because the outer square is amodal -- most of its area is the OPENING, which is
    background by construction and would dilute the very signal being tested.
    """
    inner_m = _fill(inner_px)
    ring = _fill(outer_px) & ~inner_m
    n = int(ring.sum())
    if n < 200:
        return 0.0, 0.0, n
    red_ring = float((red & ring).sum() / n)
    # ...AND THE OPENING MUST BE SEE-THROUGH. A red ring alone is not enough: the near gate carries
    # solid red SIGNAGE panels (the checkerboard strip, the "VQ-01" board), and a quad laid over one
    # of those has a perfectly red ring. It scored 0.86+ and was accepted; rendering the boundary
    # cases is what showed it. What no panel can fake is an opening you can see THROUGH, so require
    # the inner quad to be mostly non-red and take the weaker of the two as the seed's score.
    # They are two INDEPENDENT criteria, not one: min() of the pair was far too harsh, because a
    # real gate's opening usually shows the warehouse behind it -- including distant red gates --
    # so see-through sits well below the ring score even when the seed is perfect. A solid signage
    # panel, by contrast, scores near zero. So: a strict bar on the ring, a loose one on see-through.
    n_in = int(inner_m.sum())
    see_through = 1.0 - float((red & inner_m).sum() / n_in) if n_in >= 200 else 0.0
    return red_ring, see_through, n


def sane(inner, outer):
    pts = np.vstack([inner, outer])
    if not np.isfinite(pts).all():
        return False
    if (np.abs(pts[:, 0]) > SANE_SPAN * IMG_W).any() or (np.abs(pts[:, 1]) > SANE_SPAN * IMG_H).any():
        return False
    return (labelio.quad_is_simple(inner) and labelio.quad_is_simple(outer)
            and polygon_area(clip_polygon(outer, IMG_W, IMG_H)) >= MIN_VISIBLE_AREA_PX)


# ---------------------------------------------------------------------------------------------
# the two sources
# ---------------------------------------------------------------------------------------------
def _unpin(pts):
    """Push border-PINNED keypoints outside the frame. They are not measurements (12.3 px median
    error vs 0.73 px in-frame) and must not be fitted as if they were."""
    out = np.asarray(pts, float).copy()
    for j, (x, y) in enumerate(out):
        if x <= _CLAMP_EPS_PX:
            out[j, 0] = -2.0
        elif x >= IMG_W - _CLAMP_EPS_PX:
            out[j, 0] = IMG_W + 1.0
        if y <= _CLAMP_EPS_PX:
            out[j, 1] = -2.0
        elif y >= IMG_H - _CLAMP_EPS_PX:
            out[j, 1] = IMG_H + 1.0
    return out


def from_keypoints(kpts_px, conf, kpt_thresh=0.2):
    """M's raw 8 keypoints -> both squares, reconstructed rather than read."""
    k = _unpin(kpts_px)
    usable = np.array([(conf[i] >= kpt_thresh)
                       and (0.0 <= k[i, 0] <= IMG_W - 1) and (0.0 <= k[i, 1] <= IMG_H - 1)
                       for i in range(8)])
    if int(usable.sum()) < 4:
        return None
    H = gate_plane_homography(k, usable)
    if H is None:
        return None
    sq = project_gate_squares(H)
    return None if sq is None else sq


def from_lines(bgr, interior):
    m = gate_mask(bgr)
    inner_segs, outer_segs = extract_gate_lines(m)
    if not outer_segs:
        return None
    H = homography_from_lines(inner_segs, outer_segs, (IMG_W, IMG_H), interior_px=interior)
    if H is None:
        return None
    sq = project_gate_squares(H)
    return None if sq is None else sq


def from_seg(segdet, bgr):
    """Same line solver, but with a LEARNED mask instead of the HSV rule.

    The HSV mask is the weak link at close range: ``extract_gate_lines`` wants a contour with a
    HOLE, and once the gate fills the view its opening runs off the border so there is no hole to
    find. A seg model predicts the opening as its own instance, which both restores the inner edges
    and supplies the interior point the solver needs to sign a lone edge. Yields every gate found."""
    from racer.vision.gate_lines import pair_gate_instances

    fm, om = segdet.masks_for(bgr)
    for f, o in pair_gate_instances(fm, om):
        got = None
        from racer.vision.gate_lines import centre_from_seg_masks
        try:
            got = centre_from_seg_masks(f, o, image_wh=(IMG_W, IMG_H))
        except Exception:
            got = None
        if got is None:
            continue
        sq = project_gate_squares(got[1])
        if sq is not None:
            yield sq


# ---------------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frames", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--labels", default=None,
                    help="if given, frames that already have a HUMAN label are skipped")
    ap.add_argument("--weights", default=None, help="M weights (.pt/.engine); omit to use lines only")
    ap.add_argument("--seg-weights", default=None,
                    help="yolo-seg weights: a LEARNED mask for the line solver, which is what "
                         "actually works at close range (the HSV rule loses the opening contour "
                         "once the gate fills the view)")
    ap.add_argument("--seg-conf", type=float, default=0.25)
    ap.add_argument("--min-see", type=float, default=0.5,
                    help="reject a seed whose OPENING is not see-through: the near gate's solid red "
                         "signage panels otherwise pass the ring test perfectly")
    ap.add_argument("--min-ring", type=float, default=0.6,
                    help="reject a seed whose RING between the two quads is not mostly red structure")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    frames_dir, seeds_dir = Path(args.frames), Path(args.seeds)
    if seeds_dir.resolve() == frames_dir.resolve():
        sys.exit("--seeds must not be the frames dir")
    if args.labels and Path(args.labels).resolve() == seeds_dir.resolve():
        sys.exit("--seeds must NOT be the labels dir (seeds are unverified)")
    (seeds_dir / "geom").mkdir(parents=True, exist_ok=True)
    labels_dir = Path(args.labels) if args.labels else None

    det = None
    if args.weights:
        from racer.vision.detector import GateDetector          # heavy: only when asked for
        det = GateDetector(weights=args.weights, use_outer=True, partial_rescue=True)
        print(f"M loaded: {args.weights}")
    segdet = None
    if args.seg_weights:
        from racer.vision.gate_lines import SegGateLineDetector
        segdet = SegGateLineDetector.load(args.seg_weights, conf=args.seg_conf)
        print(f"seg loaded: {args.seg_weights}")

    names = sorted(p.name for p in frames_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if args.limit:
        names = names[:args.limit]

    census, manifest, ious = Counter(), {}, []
    for i, name in enumerate(names):
        stem = Path(name).stem
        if labels_dir and (labels_dir / f"{stem}.txt").exists():
            census["already-labelled"] += 1
            continue
        bgr = cv2.imread(str(frames_dir / name))
        if bgr is None or bgr.shape[1] != IMG_W or bgr.shape[0] != IMG_H:
            census["unreadable-or-wrong-size"] += 1
            continue
        red = gate_mask(bgr) > 0
        _, interior = frame_truth_mask(bgr)

        # EVERY gate in the frame is a label, so collect every candidate from every source rather
        # than keeping one "best". A frame commonly shows the gate being flown plus two behind it.
        cands = []
        if det is not None:
            try:
                for kp, cf in raw_keypoints(det, bgr):
                    sq = from_keypoints(kp, cf)
                    if sq and sane(*sq):
                        cands.append(("M", sq))
            except Exception as e:                              # a detector hiccup must not stop the run
                census[f"M-error:{type(e).__name__}"] += 1
        if segdet is not None:
            try:
                cands += [("seg", sq) for sq in from_seg(segdet, bgr) if sane(*sq)]
            except Exception as e:
                census[f"seg-error:{type(e).__name__}"] += 1
        sq = from_lines(bgr, interior)
        if sq and sane(*sq):
            cands.append(("lines", sq))

        scored = []
        for src, (inn, out) in cands:
            sc, see, n_ring = ring_score(inn, out, red)
            if sc < args.min_ring:
                census[f"rejected-ring({src})"] += 1
                continue
            if see < args.min_see:
                census[f"rejected-opaque-opening({src})"] += 1
                continue
            scored.append({"src": src, "score": sc, "see": see, "ring": n_ring,
                           "inner": np.asarray(inn, float), "outer": np.asarray(out, float)})
        scored.sort(key=lambda c: -c["score"])

        kept = []                                               # de-dup: sources often agree
        for c in scored:
            cc, sz = c["inner"].mean(axis=0), np.ptp(c["inner"], axis=0).max()
            if any(np.linalg.norm(cc - k["inner"].mean(axis=0)) < 0.4 * max(sz, 1.0) for k in kept):
                census["duplicate-gate"] += 1
                continue
            kept.append(c)
        if not kept:
            census["no-seed" if not cands else "all-candidates-rejected"] += 1
            continue

        gates, info = [], []
        for c in kept:
            inn, out = canonicalise(c["inner"], c["outer"])
            gates.append({"inner": [tuple(p) for p in inn], "outer": [tuple(p) for p in out],
                          "occluded": [False] * 8, "outer_detached": [True] * 4})
            n_in = sum(1 for p in list(inn) + list(out)
                       if 0 <= p[0] <= IMG_W - 1 and 0 <= p[1] <= IMG_H - 1)
            info.append({"source": c["src"], "ring": round(c["score"], 3),
                         "see_through": round(c["see"], 3), "kpts_in_frame": n_in})
            ious.append(c["score"])
            census[f"seeded-{c['src']}"] += 1
        (seeds_dir / f"{stem}.txt").write_text(labelio.encode_label(gates, IMG_W, IMG_H))
        (seeds_dir / "geom" / f"{stem}.json").write_text(
            json.dumps(labelio.encode_geometry(gates, IMG_W, IMG_H)))
        manifest[name] = {"gates": info, "source": "+".join(sorted({g["source"] for g in info})),
                          "iou": round(min(g["ring"] for g in info), 3),
                          "kpts_in_frame": min(g["kpts_in_frame"] for g in info)}
        census["frames-seeded"] += 1
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(names)}  frames seeded={census['frames-seeded']}")

    (seeds_dir / "_manifest.json").write_text(json.dumps(
        {"min_ring": args.min_ring, "min_see": args.min_see, "weights": args.weights, "frames": manifest}, indent=1))
    print(f"\nseeded {census['frames-seeded']}/{len(names)} frames "
          f"({len(ious)} gates) -> {seeds_dir}")
    for k, v in census.most_common():
        print(f"  {k:28s} {v}")
    if ious:
        a = np.array(ious)
        print(f"\nring redness (the seed's own self-check): median {np.median(a):.3f}  "
              f"p10 {np.percentile(a, 10):.3f}  >=0.9: {(a >= 0.9).sum()}  >=0.8: {(a >= 0.8).sum()}")
        kin = np.array([m["kpts_in_frame"] for m in manifest.values()])
        print(f"corners in frame (worst gate per frame): median {int(np.median(kin))}, "
              f"{(kin < 4).sum()} frames have a gate under 4 — the pose path could not "
              f"have seeded those at all")
    return 0


def raw_keypoints(det, bgr):
    """M's RAW per-instance keypoints + confidences. Deliberately not GateDetector.detect(): that
    applies pose solving and filtering, and seeding wants the model's own points."""
    res = det.model.predict(bgr, verbose=False, conf=0.20)[0]
    if res.keypoints is None or res.keypoints.data is None:
        return
    data = res.keypoints.data.cpu().numpy()          # (n, 8, 2 or 3)
    for inst in data:
        pts = inst[:, :2]
        cf = inst[:, 2] if inst.shape[1] > 2 else np.ones(len(inst))
        if pts.shape[0] >= 8:
            yield pts[:8], cf[:8]


if __name__ == "__main__":
    raise SystemExit(main())
