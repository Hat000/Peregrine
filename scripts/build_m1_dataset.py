"""Build the "M+1" 5-keypoint YOLO-pose dataset: the 4 inner corners PLUS the gate-opening CENTRE.

M+1 is the 30 Hz deploy emitter. Its 5th keypoint, the directly-regressed centre, is the whole
reason it exists: 55.7% of real gate views have the centre IN frame while an inner corner is cropped
off-frame (the regime the corner-only model M is blind to). The 4 inner corners still give
range-adjusted apparent area whenever they are in frame.

TWO SOURCES, and the centre is derived DIFFERENTLY from each -- this is the crux:

  * SYNTHETIC RENDER (-> TRAIN). The Blender/procedural pose is known exactly, so the true projected
    centre is the diagonal intersection of the UNCLAMPED projected inner corners. That was computed
    at render time from GateRender.keypoints_px (before the pose row's [0,1] clamp) and written to
    ``center/<split>/*.txt`` by dataset.py. Here we just read it. The inner corners come straight
    from the pose label slots 0-3 with their v flags as-is (a v=0 cropped corner is correctly masked
    by the pose loss).

  * HAND LABELS (-> TRAIN as of 2026-07-23; see the SPLIT POLICY note below). Drawn for AREA, not
    for keypoint position:
    the labeller placed each inner handle to make the visible CLIPPED opening correct, and any
    off-frame handle was nudged only so the visible edges cross the frame boundary correctly -- its
    position ALONG the edge is arbitrary. So for the ~67% of hand gates with an off-frame inner
    corner, the diagonal intersection of the handles is WRONG. We therefore DERIVE THE CENTRE FROM
    THE AREA: fit the KNOWN gate model to the exact human-drawn opening (+frame) polygon and read the
    centre off the fitted pose (racer.vision.gate_model_fit / gate_lines -- the same solvers the A/B
    harness uses). This is correct even off-frame because it uses the true EDGES + the known square,
    not the arbitrary handles. When all 4 inner corners ARE in frame the handles are trustworthy, so
    we use their exact diagonal intersection there (0 px, no fit error). See :func:`hand_gate_centre`.

REUSES build_seg_dataset's hard-won machinery (imported, not copied):
  * re-key every frame ``<dataset-tag>__<stem>`` -- several source datasets number frames from
    000000, and 437/1572 stems collided once, silently pairing images with another dataset's labels;
  * MIRROR a hard-linked image tree -- ultralytics resolves a label by swapping /images/ -> /labels/
    on the IMAGE path, so the images must live next to the new labels, not at their origin;
  * recognise BOTH the render layout (labels/<split>/) and the hand-labeler layout (labels/).

Output: a standard ultralytics pose dataset (images/ + labels/ + data.yaml, kpt_shape [5,3]). The
UNCLAMPED centre is ALSO preserved in ``center/<split>/*.txt`` so an evaluator can score the
off-frame-centre gates the [0,1] pose-row clamp would otherwise destroy (there is no human truth for
an off-frame point; the model-fit estimate is the best GT available and these are labelled + counted).

Usage:
  python scripts/build_m1_dataset.py \
      --render C:/Users/Shadow/vq2_m1_render_2026-07-22 \
      --hand   C:/Users/Shadow/vq2_label_batch_2026-07-22 \
      --out    C:/Users/Shadow/vq2_m1_2026-07-22
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "tools" / "gate_labeler"))
sys.path.insert(0, str(Path(__file__).resolve().parent))          # sibling scripts (build_seg_dataset)

# Reuse, do NOT re-implement, the collision-safe layout helpers. These carry the scars of the
# stem-collision and /images/->/labels/ bugs; a second copy would re-earn them.
import build_seg_dataset as bsd                                   # noqa: E402
import labelio                                                    # noqa: E402  (tools/gate_labeler)

IMG_W, IMG_H = 640, 360
N_INNER = 4
KPT_SHAPE = [5, 3]                        # 4 inner corners + centre, each (x, y, v)
FLIP_IDX = [1, 0, 3, 2, 4]                # L<->R swaps inner corners; the centre maps to itself
_POSE8_FIELDS = 5 + 3 * 8                 # class cx cy w h (x y v)*8 -- the source render/ pose row


# ======================================================================================
# SYNTHETIC RENDER side: read the pre-computed centre sidecar, splice into a 5-kpt row
# ======================================================================================
def center_path_for(pose_label: Path) -> Path | None:
    """The centre sidecar beside a pose label, or None. Mirrors build_seg_dataset.direct_seg_path:

        render      <root>/labels/<split>/x.txt -> <root>/center/<split>/x.txt
        (labeler)   <labels>/x.txt              -> <labels>/center/x.txt
    """
    beside = pose_label.parent / "center" / pose_label.name
    if beside.is_file():
        return beside
    parts = list(pose_label.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "labels":
            parts[i] = "center"
            cand = Path(*parts)
            return cand if cand.is_file() else None
    return None


def m1_row_from_render(pose_row: str, center_row: str) -> str:
    """Splice one 8-kpt render pose row + its centre sidecar row into one 5-kpt (inner+centre) row.

    Keeps the class, bbox and inner-corner fields VERBATIM from the pose row (byte-identical inner
    block) and appends the centre keypoint, re-applying the pose-row [0,1] clamp the sidecar omits
    (the sidecar is unclamped on purpose; v already encodes off-frame)."""
    p = pose_row.split()
    if len(p) != _POSE8_FIELDS:
        raise ValueError(f"expected {_POSE8_FIELDS}-field 8-kpt pose row, got {len(p)}")
    head = p[0:5]                                   # class cx cy w h
    inner = p[5:5 + 3 * N_INNER]                     # inner keypoints 0..3 (x y v)*4 -- as-is
    cxn, cyn, cv = center_row.split()
    nx = min(max(float(cxn), 0.0), 1.0)
    ny = min(max(float(cyn), 0.0), 1.0)
    return " ".join(head + inner + [f"{nx:.6g}", f"{ny:.6g}", str(int(float(cv)))])


def convert_render_frame(pose_text: str, center_text: str, census: Counter):
    """(m1_label_text, unclamped_center_text) for one render frame, or (None, None) to drop it.

    A negative (empty pose label) stays a negative -- WRITE the empty file. A frame whose centre
    sidecar count disagrees with its pose-row count is DROPPED loudly: pairing gate A's corners with
    gate B's centre is exactly the silent misalignment this whole design guards against."""
    pose_rows = [ln for ln in pose_text.splitlines() if ln.strip()]
    if not pose_rows:
        census["render-negative"] += 1
        return "", ""                                # negative: empty label, empty center
    center_rows = [ln for ln in (center_text or "").splitlines() if ln.strip()]
    if len(center_rows) != len(pose_rows):
        census["render-center-count-mismatch"] += 1
        return None, None
    out = []
    for pr, cr in zip(pose_rows, center_rows):
        try:
            out.append(m1_row_from_render(pr, cr))
            census["render-gate-ok"] += 1
        except ValueError:
            census["render-bad-row"] += 1
            return None, None
    return "\n".join(out) + "\n", (center_text if center_text.endswith("\n") else center_text + "\n")


# ======================================================================================
# HAND-LABEL side (the ORACLE): centre from the AREA, not from the arbitrary handles
# ======================================================================================
_HAND_CENTRE = None            # lazily-built HandCentreSolver (imports cv2 + the fit stack)


def _hand_centre_solver():
    global _HAND_CENTRE
    if _HAND_CENTRE is None:
        from m1_hand_centre import HandCentreSolver          # local module (scripts/)
        _HAND_CENTRE = HandCentreSolver(IMG_W, IMG_H)
    return _HAND_CENTRE


def hand_gate_rows(geom_gates, census: Counter, centre_census: Counter):
    """(m1_label_text, unclamped_center_text) for one hand-labelled frame's gates.

    Empty (a 0-gate geom, a negative) -> ("", ""). Each gate becomes one 5-kpt row: inner corners
    from the geom handles (normalised + [0,1]-clamped, v from the hand vis), centre from the AREA
    (see :class:`m1_hand_centre.HandCentreSolver`). A gate whose centre cannot be recovered is
    DROPPED and counted -- the oracle must not carry a guessed centre."""
    if not geom_gates:
        census["hand-negative"] += 1
        return "", ""
    solver = _hand_centre_solver()
    rows, crows = [], []
    for g in geom_gates:
        inner = g["inner"]                            # 4 UNCLAMPED (x,y), LL,LR,UR,UL
        outer = g["outer"]
        vis_inner = list(g["vis"][:N_INNER])
        # One malformed gate must not abort the whole oracle build -- count it and move on. The area
        # solvers touch cv2/SVD paths that can raise on a degenerate real polygon; an offline oracle
        # over messy failure-mined frames should be robust to a single bad gate, loudly.
        try:
            got = solver.centre(g)
        except Exception as exc:                       # noqa: BLE001 -- report, don't crash the build
            census["hand-centre-error"] += 1
            print(f"[m1] centre solver error on a hand gate ({type(exc).__name__}: {exc}); skipped")
            got = None
        if got is None:
            census["hand-centre-failed"] += 1
            continue
        cx, cy, cv, how = got
        centre_census[how] += 1
        census["hand-gate-ok"] += 1
        kpts = [(float(px), float(py)) for px, py in inner] + [(cx, cy)]
        vis = vis_inner + [int(cv)]
        bbox = labelio.bbox_from_outer(outer, IMG_W, IMG_H)
        rows.append(labelio.to_yolo_pose_row(kpts, vis, bbox, 0, IMG_W, IMG_H))
        crows.append(f"{cx / IMG_W:.6g} {cy / IMG_H:.6g} {int(cv)}")
    if not rows:
        # every gate failed -> not a clean negative; drop the frame rather than mislabel it empty
        census["hand-frame-dropped"] += 1
        return None, None
    return "\n".join(rows) + "\n", "\n".join(crows) + "\n"


def collect_hand(batch: Path, census: Counter, centre_census: Counter):
    """[(image_path, m1_label_text, center_text)] for every hand-labelled frame in the batch.

    Keyed off the geom sidecars (the exact unclamped handles) + the approved empty pose labels as
    extra negatives. Frames are matched to their image in frames/ (the labeler layout)."""
    geom_dir = batch / "labels" / "geom"
    frames_dir = batch / "frames"
    items = []
    geom_stems = set()
    import json
    for gf in sorted(geom_dir.glob("*.json")):
        stem = gf.stem
        geom_stems.add(stem)
        img = _find_frame(frames_dir, stem)
        if img is None:
            census["hand-missing-frame"] += 1
            continue
        gates = labelio.decode_geometry(json.loads(gf.read_text()))
        text, ctext = hand_gate_rows(gates, census, centre_census)
        if text is None:
            continue
        items.append((img, text, ctext))
    # approved negatives that never got a geom sidecar (empty pose label, no gate to encode)
    for lf in sorted((batch / "labels").glob("*.txt")):
        if lf.stem in geom_stems or lf.read_text().strip():
            continue
        img = _find_frame(frames_dir, lf.stem)
        if img is None:
            census["hand-missing-frame"] += 1
            continue
        census["hand-negative"] += 1
        items.append((img, "", ""))
    return items


def collect_pose_hand(root: Path, census: Counter, centre_census: Counter):
    """Hand corpora that carry only 8-kpt POSE rows and no geom sidecars (e.g. the 2026-07-02 real
    set, images/ + labels/).

    THE CENTRE RULE IS THE WHOLE POINT HERE. Fengyou: the hand labels were drawn for AREA, so an
    off-frame handle's position ALONG its edge is arbitrary -- the diagonal of the handles is NOT
    the centre for a cropped gate. Without a geom sidecar there is no drawn opening to fit, so the
    area-fit derivation is unavailable and the only sound centre is the diagonal intersection of
    four handles that are ALL in frame (v=2), where the handles are exact. Gates with any clamped
    corner are DROPPED rather than given a fabricated centre -- that is precisely the poisoning
    this whole rebuild exists to undo."""
    items = []
    for img, lab in bsd.scan_pairs(root):
        text = lab.read_text()
        if not text.strip():
            census["posehand-negative"] += 1
            items.append((img, "", ""))
            continue
        rows, crows = [], []
        for line in text.splitlines():
            if not line.strip():
                continue
            r, c = pose_row_all_inframe(line, census, centre_census)
            if r is None:
                continue
            rows.append(r)
            crows.append(c)
        if not rows:
            census["posehand-frame-dropped"] += 1
            continue
        items.append((img, "\n".join(rows) + "\n", "\n".join(crows) + "\n"))
    return items


def pose_row_all_inframe(pose_row: str, census: Counter, centre_census: Counter):
    """One 8-kpt pose row -> (5-kpt row, centre row), or (None, None) if any inner corner is
    clamped. Centre = intersection of the inner diagonals, exact when all four are in frame."""
    p = pose_row.split()
    if len(p) < 5 + 8 * 3:
        census["posehand-short-row"] += 1
        return None, None
    kp = [(float(p[5 + 3 * i]), float(p[6 + 3 * i]), float(p[7 + 3 * i])) for i in range(8)]
    inner = kp[:4]
    if any(v < 2 for (_, _, v) in inner):
        census["posehand-dropped-cropped"] += 1     # no geom -> no sound centre; see the docstring
        return None, None
    c = _diag_intersection([(x, y) for (x, y, _) in inner])
    if c is None:
        census["posehand-degenerate-quad"] += 1
        return None, None
    centre_census["diag-handles (all 4 in frame)"] += 1
    kv = " ".join(f"{x:.6f} {y:.6f} {int(v)}" for (x, y, v) in inner)
    return (f"{' '.join(p[:5])} {kv} {c[0]:.6f} {c[1]:.6f} 2",
            f"{c[0]:.6f} {c[1]:.6f} 2")


def _diag_intersection(q):
    """Intersection of the diagonals of quad q=[LL,LR,UR,UL] -- the projective centre of a square."""
    import numpy as _np
    (x1, y1), (x2, y2) = q[0], q[2]
    (x3, y3), (x4, y4) = q[1], q[3]
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(d) < 1e-12:
        return None
    a = x1 * y2 - y1 * x2
    b = x3 * y4 - y3 * x4
    cx = (a * (x3 - x4) - (x1 - x2) * b) / d
    cy = (a * (y3 - y4) - (y1 - y2) * b) / d
    if not (_np.isfinite(cx) and _np.isfinite(cy)):
        return None
    return float(cx), float(cy)


def _find_frame(frames_dir: Path, stem: str) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg"):
        cand = frames_dir / (stem + ext)
        if cand.exists():
            return cand
    return None


# ======================================================================================
# assembly (collision-safe, image tree mirrored)
#
# SPLIT POLICY CHANGED 2026-07-23 (Fengyou: "do not gatekeep hand label to just val, we NEED ALL
# the data we can during train, ALL hand drawn gates get sent to training").
#
# The old policy was render -> train, hand -> val. It had a consequence nobody intended: renders
# ALWAYS contain a gate, so every negative that existed lived in the hand set, and the hand set was
# val-only. The shipped M+1 therefore trained on 4000 frames with ZERO negatives and 38 negatives
# (17.3%) sat unused in val. It hallucinates confidently as a direct result -- p90 false positive
# 0.841 against a MEDIAN true positive of 0.832, i.e. unseparable by any threshold.
#
# Now: hand + renders + negative corpora ALL go to train. VAL is built from explicitly held-out
# sources (--val-render / --val-negatives) so it never overlaps train. Note the cost, deliberately
# accepted: with every hand frame in train there is no longer an uncontaminated REAL-frame accuracy
# oracle. Real accuracy is measured in flight; --val-negatives (fresh renders, never trained on)
# preserves the one metric that caused this change, the false-positive rate.
# ======================================================================================
def _write_frame(img: Path, text: str, ctext: str, out: Path, split: str,
                 used: set, census: Counter) -> Path | None:
    """Re-key <tag>__<stem>, hard-link the image, write the 5-kpt label + centre sidecar. Uniqueness
    is ASSERTED via ``used`` -- a duplicate key is the stem-collision bug and must never silently
    overwrite."""
    name = f"{bsd.dataset_tag(img)}__{img.stem}"
    if name in used:
        census["duplicate-frame"] += 1
        return None
    used.add(name)
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    (out / "center" / split).mkdir(parents=True, exist_ok=True)
    bsd._link_or_copy(img, out / "images" / split / (name + img.suffix))
    (out / "labels" / split / (name + ".txt")).write_text(text, encoding="utf-8")
    (out / "center" / split / (name + ".txt")).write_text(ctext, encoding="utf-8")
    census[f"{split}-frames-kept"] += 1
    return out / "images" / split / (name + img.suffix)


def build(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    census, centre_census = Counter(), Counter()
    used: set = set()
    kept = {"train": [], "val": []}

    def _add_render(roots, split):
        for root in (roots or []):
            for img, pose_label in bsd.scan_pairs(Path(root)):
                cpath = center_path_for(pose_label)
                if cpath is None and pose_label.read_text().strip():
                    census["render-missing-center"] += 1
                    continue
                text, ctext = convert_render_frame(
                    pose_label.read_text(), cpath.read_text() if cpath else "", census)
                if text is None:
                    continue
                p = _write_frame(img, text, ctext, out, split, used, census)
                if p is not None:
                    kept[split].append(p)

    def _add_negatives(roots, split, tag):
        """Frames whose label file exists and is EMPTY. Anything unlabelled is skipped: a missing
        label means 'nobody looked', not 'no gate', and feeding those as background would teach the
        detector to suppress real gates."""
        for root in (roots or []):
            for img, lab in bsd.scan_pairs(Path(root)):
                if lab.read_text().strip():
                    census[f"{tag}-skipped-has-gate"] += 1
                    continue
                p = _write_frame(img, "", "", out, split, used, census)
                if p is not None:
                    census[f"{tag}-negative"] += 1
                    kept[split].append(p)

    # ---- TRAIN ----------------------------------------------------------------------------------
    _add_render(args.render, "train")

    # Hand corpora share FRAMES but not LABELS, so dedup on the frame STEM, not on _write_frame's
    # <corpus-tag>__<stem> key -- the same picture under two corpus tags passes that check and would
    # be emitted twice, once per labelling. Measured across the three inboxes: 43 stems are labelled
    # in more than one, and exactly ONE disagrees (positive in the 07-22 batch, negative in the 07-21
    # inbox). FIRST LISTED WINS, so pass the newest corpus first. Nothing is lost by preferring it:
    # all 25 inbox-only positives already carry a batch geom sidecar.
    hand_seen: set = set()

    def _add_hand(roots, collector):
        for batch in (roots or []):
            for img, text, ctext in collector(Path(batch), census, centre_census):
                if img.stem in hand_seen:
                    census["hand-superseded-by-newer-corpus"] += 1
                    continue
                hand_seen.add(img.stem)
                p = _write_frame(img, text, ctext, out, "train", used, census)
                if p is not None:
                    kept["train"].append(p)

    _add_hand(args.hand, collect_hand)                    # hand gates: centre from the AREA
    _add_hand(args.hand_pose, collect_pose_hand)          # pose-row corpora with no geom sidecars
    _add_negatives(args.negatives, "train", "neg")

    # ---- VAL: explicitly held-out sources only, never a slice of the above ----------------------
    _add_render(args.val_render, "val")
    _add_negatives(args.val_negatives, "val", "valneg")

    for split in ("train", "val"):
        (out / f"{split}.txt").write_text(
            "\n".join(str(p) for p in kept[split]) + ("\n" if kept[split] else ""), encoding="utf-8")

    _write_yaml(out)
    _report(census, centre_census, kept, out)
    return 0


def _write_yaml(out: Path) -> None:
    flip = ", ".join(str(i) for i in FLIP_IDX)
    (out / "data.yaml").write_text(
        "# M+1 gate dataset: 4 inner corners + gate-opening CENTRE (5 keypoints). Built by "
        "scripts/build_m1_dataset.py.\n"
        f"path: {out.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n  0: gate\n"
        f"kpt_shape: [{KPT_SHAPE[0]}, {KPT_SHAPE[1]}]   # inner LL,LR,UR,UL then CENTRE, (x,y,v)\n"
        f"flip_idx: [{flip}]   # horizontal flip swaps inner L<->R; the centre is its own mirror\n",
        encoding="utf-8")


def _report(census: Counter, centre_census: Counter, kept, out: Path) -> None:
    print("\nbuild census:")
    for k, v in census.most_common():
        print(f"  {k:28s} {v}")
    if centre_census:
        print("\nhand-label CENTRE derivation:")
        tot = sum(centre_census.values())
        for k, v in centre_census.most_common():
            print(f"  {k:28s} {v}  ({100 * v / tot:.1f}%)")
    print(f"\ntrain frames: {len(kept['train'])}   val frames: {len(kept['val'])}")
    print(f"wrote {out / 'data.yaml'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--render", nargs="*", default=None,
                    help="render dataset root(s) with labels/ + center/ + images/ -> TRAIN")
    ap.add_argument("--hand", nargs="*", default=None,
                    help="hand-label batch root(s) with frames/ + labels/geom/ -> TRAIN. Pass every "
                         "inbox: they are NOT nested. vq2_label_batch_2026-07-22 holds the same "
                         "FRAMES as the two inboxes but not the same LABELS -- 4 positives exist "
                         "only in close_inbox and 11 negatives only in inbox_2026-07-21.")
    ap.add_argument("--hand-pose", nargs="*", default=None,
                    help="hand corpora with 8-kpt pose rows but NO geom sidecars -> TRAIN. Only "
                         "gates with all 4 inner corners in frame are kept (see collect_pose_hand).")
    ap.add_argument("--negatives", nargs="*", default=None,
                    help="gate-free corpora (empty label files) -> TRAIN. Frames with no label file "
                         "at all are skipped: unlabelled is not the same as gate-free.")
    ap.add_argument("--val-render", nargs="*", default=None,
                    help="held-out render root(s) -> VAL")
    ap.add_argument("--val-negatives", nargs="*", default=None,
                    help="held-out gate-free root(s) -> VAL. This is the false-positive benchmark; "
                         "keep it disjoint from --negatives or the number is meaningless.")
    ap.add_argument("--out", required=True, help="output 5-keypoint pose dataset root")
    args = ap.parse_args()
    if not any((args.render, args.hand, args.hand_pose, args.negatives,
                args.val_render, args.val_negatives)):
        ap.error("need at least one input source")
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())
