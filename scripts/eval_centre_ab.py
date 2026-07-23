"""A/B the mask -> gate-centre solvers on HAND-LABELLED REAL FRAMES.

Three arms, fed the IDENTICAL predicted masks:
  * LINE SOLVER   -- fit lines to contour segments, solve a homography (gate_lines).
  * HULL QUAD     -- collapse the mask's convex hull to 4 vertices, intersect the diagonals.
  * MODEL FIT     -- take the KNOWN 3-D gate and search its pose for maximum silhouette/mask
                     overlap (gate_model_fit). The first two INFER the gate from the boundary;
                     this one never asks the mask "which of your edges is a gate edge?".

GROUND TRUTH is the diagonal intersection of the hand-labelled INNER quad, read from the labeler's
exact unclamped geometry sidecar (labels/geom/*.json), not from the pose row -- the pose row clamps
off-frame corners onto the border, which would destroy exactly the cropped gates this is about.
Using the diagonal formula for the truth as well is a DEFINITION, not circularity: the GT quad is
four human-placed handles, the prediction is a machine-fitted quad from a mask, and the formula in
between is exact projective geometry that adds no free parameters to either side.

⚠ THE MATCHER IS THE EASIEST THING TO GET WRONG HERE, and getting it wrong invents error out of
nothing. An earlier harness matched every GT centre to its NEAREST prediction: on a frame with 3
labelled gates and 1 prediction, all three matched that one prediction and two of them scored as
huge misses that no solver could have avoided. Matching here is GREEDY ONE-TO-ONE -- sort every
(gt, pred) pair by distance and consume both sides -- and COVERAGE (how many labelled gates got a
prediction at all) is reported SEPARATELY from the error over matched pairs, because they are
different failure modes and averaging them together hides both.

Both arms are fed the IDENTICAL masks: inference runs once per frame per model and both solvers
consume the same ``pair_gate_instances`` output. Anything that differs is the solver.

Usage:
  python scripts/eval_centre_ab.py                       # both models, table + gallery
  python scripts/eval_centre_ab.py --no-gallery --device cpu
  python scripts/eval_centre_ab.py --models v2 --arms line,quad,model,model-flat
"""
from __future__ import annotations

import argparse
import html
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "gate_labeler"))

from racer.contracts import Frame  # noqa: E402
from racer.vision.detector import (  # noqa: E402
    KPT_CONF_THRESH_DEFAULT,   # 0.2 -- the deploy threshold; a keypoint below it is not "visible"
    GateDetector,
    _to_numpy,                 # torch->numpy without importing torch, the SAME one detect() uses
    _unclamped_mask,           # border-clamp test, shared so the "clamped" rule can't drift from
                               # the one the rescue path applies (ultralytics pins off-frame kpts)
    observations_from_results,
)
from racer.vision.gate_lines import (  # noqa: E402
    SegGateLineDetector,
    _on_border,            # private on purpose, but the gallery must colour clipped fits with the
                           # SAME rule the solver used -- a second copy would drift
    centre_from_masks_quad,
    centre_from_quad,      # diagonal intersection of a 4-pt quad -- M's centre, derived the SAME
                           # way the ground truth is (load_truth applies it to the human handles)
    centre_from_seg_masks,
    pair_gate_instances,
    quad_from_mask_ex,
)
from racer.vision.gate_model_fit import FLAT_MODEL, fit_gate_model  # noqa: E402
# DEFAULT_MODEL and the score internals are imported LOCALLY inside run_ambiguity / run_task2 --
# only those two experiments need them.

DEFAULT_BATCH = "C:/Users/Shadow/vq2_label_batch_2026-07-22"
DEFAULT_VAL = ("C:/Users/Shadow/AppData/Local/Temp/claude/C--Users-Shadow-Peregrine/"
               "1b6563db-3b60-4944-905f-3646a289f780/scratchpad/real_val.txt")
DEFAULT_OUT = "C:/Users/Shadow/centre_ab_2026-07-22"
WEIGHTS = {
    "v2": "C:/Users/Shadow/Peregrine/runs/segment/gate_seg_v2/weights/best.pt",
    "s1": "C:/Users/Shadow/Peregrine/runs/segment/gate_seg_s1/weights/best.pt",
}
PAD_X, PAD_Y = 160, 90          # gallery canvas margin, so OFF-SCREEN centres are still drawable


def quad_centre(q):
    """Diagonal intersection of a 4-point quad -- the definition of a projected square's centre."""
    q = np.asarray(q, float)
    d1 = np.cross([q[0][0], q[0][1], 1.0], [q[2][0], q[2][1], 1.0])
    d2 = np.cross([q[1][0], q[1][1], 1.0], [q[3][0], q[3][1], 1.0])
    p = np.cross(d1, d2)
    return None if abs(p[2]) < 1e-12 else p[:2] / p[2]


def load_truth(batch: Path, stems):
    """[(stem, image_path, [centre_px], [inner_quad], [above_area_floor])] per val stem with geometry.

    Read through the labeler's own ``decode_geometry`` so the truth is the EXACT unclamped handles.
    The pose .txt rows would clamp off-frame corners onto the border, which is precisely the
    information the cropped-gate case turns on."""
    from labelio import decode_geometry            # the labeler's own decoder, not a re-derivation
    from racer.vision.seg_labels import clip_polygon, polygon_area

    # THE TRUTH COMES FROM THE AREA, NOT THE HANDLES. The hand labels were drawn for AREA: the
    # labeller placed each inner handle to make the visible CLIPPED opening correct, and an
    # off-frame handle was nudged only so the visible edges cross the frame boundary correctly --
    # its position ALONG the edge is arbitrary. So for the ~67% of gates with an off-frame corner,
    # the diagonal intersection of the handles is NOT the true centre, and using it as ground truth
    # silently mis-scores every arm on exactly the cropped population this harness exists to judge.
    # HandCentreSolver fits the KNOWN gate model to the exact drawn opening and reads the centre off
    # the fitted pose -- correct off-frame because it uses the true EDGES plus the known square. It
    # still uses the plain diagonal when all four corners ARE in frame, where the handles are exact.
    # --gt-from-handles restores the old behaviour so the two can be compared rather than argued.

    out, missing = [], []
    for s in stems:
        gp = batch / "labels" / "geom" / f"{s}.json"
        img = next((p for p in (batch / "frames" / f"{s}{e}" for e in (".png", ".jpg", ".jpeg"))
                    if p.exists()), None)
        if not gp.exists() or img is None:
            missing.append(s)
            continue
        gates = decode_geometry(json.loads(gp.read_text()))
        cs, qs, big = [], [], []
        for g in gates:
            c = _truth_centre(g)
            if c is not None and np.isfinite(c).all():
                cs.append(c)
                qs.append(np.asarray(g["inner"], float))
                big.append(polygon_area(clip_polygon(g["inner"], 640, 360)) >= 200.0)
        if cs:
            out.append((s, img, cs, qs, big))
        else:
            missing.append(s)                      # a negative frame: labelled, but no gate in it
    return out, missing


_GT_FROM_HANDLES = False        # --gt-from-handles: the OLD (area-blind) ground truth
_HAND_SOLVER = None


def _truth_centre(g):
    """Ground-truth centre for one decoded geom gate. See the note in load_truth."""
    global _HAND_SOLVER
    if _GT_FROM_HANDLES:
        return quad_centre(g["inner"])
    if _HAND_SOLVER is None:
        from m1_hand_centre import HandCentreSolver
        _HAND_SOLVER = HandCentreSolver(640, 360)
    got = _HAND_SOLVER.centre(g)
    if got is None:
        return None
    return np.array([got[0], got[1]], float)


def greedy_match(gt_pts, pred_pts):
    """[(gt_i, pred_j, dist)] -- ONE-TO-ONE, cheapest pair first. See the module note: a nearest-
    neighbour matcher lets one prediction absorb several GT gates and manufactures huge errors."""
    pairs = sorted((float(np.linalg.norm(np.asarray(g) - np.asarray(p))), i, j)
                   for i, g in enumerate(gt_pts) for j, p in enumerate(pred_pts))
    used_g, used_p, out = set(), set(), []
    for d, i, j in pairs:
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        out.append((i, j, d))
    return out


class Arm:
    """One solver's tally. Errors and coverage are kept apart on purpose.

    Coverage is also tracked over the SOLVABLE subset -- labelled gates whose visible inner square
    clears the 200 px^2 area floor both solvers apply to a mask. 10 of the 54 labelled gates are
    below it (distant gates a few pixels across), so a headline coverage number alone reads as a
    solver failure when it is really a floor that neither arm was ever going to clear."""

    def __init__(self, name):
        self.name = name
        self.n_gt = self.n_pred = self.n_offscreen = self.n_big = self.n_big_hit = 0
        self.err = []                    # matched pairs only
        self.err_gt_offscreen = []       # subset where the LABELLED centre is outside the image

    def add_frame(self, gt_pts, preds, wh, big):
        self.n_gt += len(gt_pts)
        self.n_pred += len(preds)
        self.n_big += sum(big)
        w, h = wh
        for c in preds:
            if not (0 <= c[0] < w and 0 <= c[1] < h):
                self.n_offscreen += 1
        per_gt = {}
        for i, j, d in greedy_match(gt_pts, preds):
            self.err.append(d)
            per_gt[i] = d
            self.n_big_hit += bool(big[i])
            g = gt_pts[i]
            if not (0 <= g[0] < w and 0 <= g[1] < h):
                self.err_gt_offscreen.append(d)
        return per_gt

    def row(self):
        e = np.array(self.err) if self.err else np.array([np.nan])
        cov = 100.0 * len(self.err) / max(self.n_gt, 1)
        big = 100.0 * self.n_big_hit / max(self.n_big, 1)
        return (self.name, f"{len(self.err)}/{self.n_gt}", f"{cov:5.1f}%", f"{big:5.1f}%",
                f"{np.median(e):7.1f}", f"{np.percentile(e, 90):7.1f}",
                f"{self.n_pred:5d}", f"{self.n_offscreen:4d}")


# --- arm registry -------------------------------------------------------------------------------
# Each entry solves ONE gate's (frame_mask, opening_mask) and returns (centre_px, extra) or None.
# ``extra`` is whatever the gallery needs to draw that arm; the scoring only ever uses the centre.
# Adding an arm here is the only place a new solver has to be wired in, so the greedy one-to-one
# matcher and the coverage/error bookkeeping below can never accidentally fork per arm.

def _arm_line(f, o, wh):
    got = centre_from_seg_masks(f, o, image_wh=wh)
    return None if got is None else (got[0], None)


def _arm_quad(f, o, wh):
    got = centre_from_masks_quad(f, o, image_wh=wh)
    return None if got is None else (got[0], got[1])


def _arm_model(f, o, wh, **kw):
    fit = fit_gate_model(f, o, wh, **kw)
    return None if fit is None else (fit.centre_px, fit)


ARMS = {
    "line": ("line-solver (baseline)", _arm_line),
    "quad": ("quad+diagonal", _arm_quad),
    "model": ("MODEL FIT (3-D, depth 0.26)", _arm_model),
    # --- ablations. Each isolates ONE design choice, so a win or a loss can be attributed. ---
    "model-flat": ("model fit, flat (depth 0)",
                   lambda f, o, wh: _arm_model(f, o, wh, model=FLAT_MODEL)),
    "model-open": ("model fit, opening IoU only",
                   lambda f, o, wh: _arm_model(f, o, wh, w_frame=0.0)),
    "model-frame": ("model fit, frame IoU only",
                    lambda f, o, wh: _arm_model(f, o, wh, w_open=0.0)),
    "model-o2": ("model fit, opening weighted 2x",
                 lambda f, o, wh: _arm_model(f, o, wh, w_open=2.0)),
    "model-b60": ("model fit, budget 60",
                  lambda f, o, wh: _arm_model(f, o, wh, budget=60)),
    "model-b240": ("model fit, budget 240",
                   lambda f, o, wh: _arm_model(f, o, wh, budget=240)),
    "model-s4": ("model fit, coarse raster (scale 4)",
                 lambda f, o, wh: _arm_model(f, o, wh, scales=(4.0,))),
    "model-noline": ("model fit, no line-H init",
                     lambda f, o, wh: _arm_model(f, o, wh, line_init=False)),
    "model-pre": ("model fit, translation pre-search",
                  lambda f, o, wh: _arm_model(f, o, wh, presearch=True)),
    "quad-strict": ("quad, no clipped fits",
                    lambda f, o, wh: (lambda g: None if g is None else (g[0], g[1]))(
                        centre_from_masks_quad(f, o, image_wh=wh, max_border_edges=0))),
}


def run_model(tag, weights, truth, device, arm_ids):
    """Score every arm for one seg model. Inference runs ONCE per frame; the arms share its masks."""
    det = SegGateLineDetector.load(weights, device=device)
    arms = {a: Arm(f"{tag} {ARMS[a][0]}") for a in arm_ids}
    timing = {a: [] for a in arm_ids}        # per-GATE wall time, ms
    init_ms = []                             # quad_from_mask_ex alone: the model arm's shared init
    diag = {"evals": [], "ambig": [], "iou_f": [], "iou_o": [], "init_kind": {}}
    per_frame = []
    for stem, img_path, gt_pts, gt_quads, big in truth:
        im = cv2.imread(str(img_path))
        if im is None:
            continue
        h, w = im.shape[:2]
        pairs = pair_gate_instances(*det.masks_for(im))
        preds = {a: [] for a in arm_ids}
        for f, o in pairs:
            if "model" in arm_ids:
                t0 = time.perf_counter()
                quad_from_mask_ex(o if o is not None else f)
                init_ms.append((time.perf_counter() - t0) * 1000.0)
            for a in arm_ids:
                t0 = time.perf_counter()
                got = ARMS[a][1](f, o, (w, h))
                timing[a].append((time.perf_counter() - t0) * 1000.0)
                if got is not None:
                    preds[a].append(got)
                if a == "model" and got is not None:
                    fit = got[1]
                    diag["evals"].append(fit.n_evals)
                    diag["ambig"].append(fit.ambiguity_margin)
                    diag["iou_f"].append(fit.iou_frame)
                    diag["iou_o"].append(fit.iou_opening)
                    diag["init_kind"][fit.init_kind] = diag["init_kind"].get(fit.init_kind, 0) + 1
        per_gt = {a: arms[a].add_frame(gt_pts, [c for c, _ in preds[a]], (w, h), big)
                  for a in arm_ids}
        per_frame.append({"stem": stem, "img": img_path, "im": im, "pairs": pairs,
                          "gt": gt_pts, "gt_quads": gt_quads,
                          "preds": preds, "per_gt": per_gt})
    return arms, per_frame, timing, init_ms, diag


# ==================================================================================================
# M ARM -- the KEYPOINT detector's own gate-centre, on the SAME 54 gates as the three mask arms.
# ==================================================================================================
# M is a different animal from the mask arms: an 8-keypoint YOLO-pose net, not a mask consumer, so
# it does not fit the (frame_mask, opening_mask) signature the ARMS registry uses. It is wired in
# HERE instead -- but it reuses the EXACT same ``Arm`` tally and ``greedy_match`` as the mask arms
# (the load-bearing one-to-one matcher), so its coverage / median / p90 land in the same table on
# the same footing, matched against the same ground truth the same way.
#
# Centre derivation MIRRORS the ground truth: the diagonal intersection of the INNER quad
# (``centre_from_quad``, the very formula ``load_truth`` applies to the hand-labelled inner handles).
# We read the RAW model keypoints (``model.predict``), NOT ``GateDetector.detect()``: detect() runs
# the partial-corner rescue, which DERIVES off-frame inner corners through a homography -- exactly
# the failure mode this measurement exists to expose. ultralytics itself clamps any off-frame
# keypoint to the image border (scale_coords -> clip_coords), so a cropped gate returns with inner
# corners pinned to the edge; that pinning is RECORDED (``inner_clamped``), never papered over. A
# separate ``m-deploy`` arm DOES use detect() (rescue ON) so the current mitigation is in the table
# too. The OUTER quad is offered as its own arm so we can report whichever square M does better on.

M_WEIGHTS = {
    "engine": "C:/Users/Shadow/Peregrine/models/vq2_partial_m_2026-07-06_fp16_384x640.engine",
    "pt": "C:/Users/Shadow/Peregrine/models/vq2_partial_m_2026-07-06.pt",
}
M_SCORE_THRESH = 0.25                          # deploy default (GateDetector.score_thresh)
M_KPT_CONF_THRESH = KPT_CONF_THRESH_DEFAULT    # 0.2 deploy default; a corner below it is unusable

M_ARMS = {
    #  id             (label,                                     kind)
    "m-inner":       ("M inner-quad (raw kpts)",                 "inner"),
    "m-inner-clean": ("M inner-quad (usable kpts only)",         "inner-clean"),
    "m-outer":       ("M outer-quad (raw kpts)",                 "outer"),
    "m-outer-clean": ("M outer-quad (usable kpts only)",         "outer-clean"),
    "m-deploy":      ("M deploy path (kpts+partial rescue)",     "deploy"),
}


def load_m(device):
    """Load M, PREFERRING the fp16 TRT engine and falling back to the .pt if this venv has no
    tensorrt. Returns (GateDetector, which, attempts). The engine only fails at PREDICT time (its
    tensorrt import is lazy inside ultralytics), so a dummy predict is forced here to surface that at
    load, not mid-loop. The two weights are the SAME net -- parity was proven at export (corner
    median 0.01 px), so they differ in speed, not geometry."""
    dummy = np.zeros((360, 640, 3), np.uint8)
    attempts = []
    for which in ("engine", "pt"):
        try:
            det = GateDetector.load(M_WEIGHTS[which], device=device,
                                    score_thresh=M_SCORE_THRESH, kpt_conf_thresh=M_KPT_CONF_THRESH)
            det.model.predict(dummy, verbose=False, device=det.device)   # force the lazy TRT import
            return det, which, attempts
        except Exception as ex:                                          # noqa: BLE001
            attempts.append((which, f"{type(ex).__name__}: {ex}"))
    raise RuntimeError(f"could not load M from any weights: {attempts}")


def _dets_from_res(res, wh, score_thresh=M_SCORE_THRESH, conf_thresh=M_KPT_CONF_THRESH):
    """RAW per-detection keypoint records from one ultralytics pose Result. Each record carries the
    naive inner/outer diagonal-intersection centre AND the honest usability flags: a corner counts
    as 'usable' only if it clears the conf threshold AND is not clamped to the image border (an
    off-frame keypoint comes back pinned to the edge, carrying ~12 px error -- detector.py)."""
    if not res:
        return []
    r = res[0]
    kp, boxes = getattr(r, "keypoints", None), getattr(r, "boxes", None)
    if kp is None or boxes is None:
        return []
    xy = _to_numpy(getattr(kp, "xy", None))
    if xy is None or xy.size == 0:
        return []
    conf = _to_numpy(getattr(kp, "conf", None))
    if conf is None:
        conf = np.ones(xy.shape[:2])
    scores = _to_numpy(getattr(boxes, "conf", None))
    if scores is None:
        scores = np.ones(xy.shape[0])
    out = []
    for i in range(xy.shape[0]):
        if scores[i] < score_thresh:
            continue
        if xy.shape[1] == 8:
            inner, outer = xy[i, :4], xy[i, 4:]
            ic, oc = conf[i, :4], conf[i, 4:]
        else:                                    # a 4-kpt model: inner only, no outer square
            inner, outer, ic, oc = xy[i, :4], None, conf[i, :4], None
        inner_unclamped = _unclamped_mask(inner, wh)
        rec = {
            "c_inner": centre_from_quad(inner),
            "inner_usable": bool(inner_unclamped.all() and (ic >= conf_thresh).all()),
            "inner_clamped": bool(not inner_unclamped.all()),
            "inner_lowconf": bool((ic < conf_thresh).any()),
            "score": float(scores[i]),
            "c_outer": None, "outer_usable": False,
        }
        if outer is not None:
            outer_unclamped = _unclamped_mask(outer, wh)
            rec["c_outer"] = centre_from_quad(outer)
            rec["outer_usable"] = bool(outer_unclamped.all() and (oc >= conf_thresh).all())
        out.append(rec)
    return out


def _m_centre(rec, kind):
    """The centre this M arm emits for one detection, or None to withhold (an unusable-kpt gate on a
    -clean arm -- exactly the population a direct centre keypoint would have to cover)."""
    if kind == "inner":
        return rec["c_inner"]
    if kind == "outer":
        return rec["c_outer"]
    if kind == "inner-clean":
        return rec["c_inner"] if rec["inner_usable"] else None
    if kind == "outer-clean":
        return rec["c_outer"] if rec["outer_usable"] else None
    return None


def _deploy_centres(det, im, fid, res):
    """The CURRENT deploy path's centres: detect()'s observations (partial rescue ON), each turned
    into a centre by the same diagonal formula. Reuses the already-computed ``res`` so no second
    inference is paid. Only a 4-corner observation yields a quad centre; a 3-corner P3P subset needs
    a pose, not a quad, and is counted as no-centre here (honest -- this arm is the quad path)."""
    obs = observations_from_results(
        Frame(fid, 0, im), res[0], score_thresh=det.score_thresh,
        kpt_conf_thresh=det.kpt_conf_thresh, use_outer=det.use_outer, partial_rescue=det.partial_rescue)
    out = []
    for o in obs:
        c = np.asarray(o.corners_px, float)
        if c.shape[0] == 4 and o.corner_ids is None:
            cc = centre_from_quad(c)
            if cc is not None and np.isfinite(cc).all():
                out.append(cc)
    return out


def run_m(truth, device, arm_ids):
    """Score every requested M arm on the SAME 54 gates. ONE inference per frame; all raw-keypoint
    arms share it. Returns (arms, per_frame, which). ``per_frame`` keeps the per-detection records so
    the structural-miss count and the cross-check against the mask arms can be computed after."""
    det, which, attempts = load_m(device)
    for w, e in attempts:
        print(f"  [M] {w} weights unavailable -> {e}")
    print(f"  [M] using {which.upper()} weights on device={det.device!r}: {M_WEIGHTS[which]}")
    arms = {a: Arm(f"M[{which}] {M_ARMS[a][0]}") for a in arm_ids}
    raw_ids = [a for a in arm_ids if M_ARMS[a][1] != "deploy"]
    per_frame = []
    for fid, (stem, img_path, gt_pts, gt_quads, big) in enumerate(truth):
        im = cv2.imread(str(img_path))
        if im is None:
            continue
        h, w = im.shape[:2]
        res = det.model.predict(im, verbose=False, device=det.device)
        dets = _dets_from_res(res, (w, h))
        preds = {a: [c for c in (_m_centre(rec, M_ARMS[a][1]) for rec in dets)
                     if c is not None and np.isfinite(c).all()] for a in raw_ids}
        if "m-deploy" in arm_ids:
            preds["m-deploy"] = _deploy_centres(det, im, fid, res)
        per_gt = {a: arms[a].add_frame(gt_pts, preds[a], (w, h), big) for a in arm_ids}
        per_frame.append({"stem": stem, "gt": gt_pts, "wh": (w, h), "dets": dets, "per_gt": per_gt})
    return arms, per_frame, which


def report_m_structural(m_per_frame, seg_per_frame, seg_tag, n_gates):
    """THE crucial column: for how many of the labelled gates does M produce NO usable inner-quad
    centre because its inner corners are off-frame/clamped (or washed out)? That set is the
    population a direct centre keypoint would exist to cover. Then two honest cross-checks:
      (1) on the gates where M IS clean, how do the mask arms do there? (is M's edge real or just
          an easy subset?)
      (2) on the gates M drops, do the mask arms recover a centre? (is the mask path already
          covering M's blind spot?)
    Uses ONE greedy GT<->detection match per frame (naive inner centre, every detection), so the
    linkage is the same one-to-one rule as the scoring."""
    # classify every labelled gate; buckets partition the 54 exactly
    usable, clamped_off, washed, notdet = [], [], [], []   # each: (stem, gt_i, m_inner_err_or_None)
    for r in m_per_frame:
        gt, dets = r["gt"], r["dets"]
        valid = [(j, d["c_inner"]) for j, d in enumerate(dets) if d["c_inner"] is not None]
        link = {i: valid[k][0] for i, k, _d in greedy_match(gt, [c for _, c in valid])}
        for i in range(len(gt)):
            if i not in link:
                notdet.append((r["stem"], i, None))
                continue
            rec = dets[link[i]]
            err = float(np.linalg.norm(np.asarray(gt[i]) - np.asarray(rec["c_inner"])))
            if rec["inner_usable"]:
                usable.append((r["stem"], i, err))
            elif rec["inner_clamped"]:            # >=1 inner corner off-frame (pinned to the border)
                clamped_off.append((r["stem"], i, err))
            else:                                 # all inner corners in-frame, but one washed out
                washed.append((r["stem"], i, err))
    dropped = notdet + clamped_off + washed
    print(f"\nM STRUCTURAL MISS -- inner-quad centre on the {n_gates} labelled gates (RAW keypoints, "
          f"NO rescue). This is a FAILURE-MINED val set (crash / pose-recover / tail frames), i.e. "
          f"deliberately\nheavy on cropped and close gates -- the regime the keypoint path is weakest "
          f"in.")
    print(f"  usable inner centre (all 4 inner kpts in-frame AND conf>= {M_KPT_CONF_THRESH}) : "
          f"{len(usable):2d}/{n_gates}   (median inner error here {np.median([e for *_, e in usable]):.1f} px)")
    print(f"  M produced NO usable inner centre                              : "
          f"{len(dropped):2d}/{n_gates}   <== the population a direct centre keypoint must cover")
    print(f"    not detected by M at all                        {len(notdet):2d}")
    print(f"    detected, >=1 inner corner OFF-FRAME (clamped)  {len(clamped_off):2d}   "
          f"<- the direct-keypoint case; naive inner centre here misses by "
          f"{np.median([e for *_, e in clamped_off]):.0f} px median")
    print(f"    detected, inner in-frame but a corner washed out {len(washed):2d}   "
          f"(glare/occlusion, conf< {M_KPT_CONF_THRESH}; not a cropping failure)")
    # cross-check against the mask arms of one seg model
    seg_map = {r["stem"]: r["per_gt"] for r in seg_per_frame} if seg_per_frame else {}
    have = set().union(*[set(pg) for pg in seg_map.values()]) if seg_map else set()
    xarms = [a for a in ("quad", "model") if a in have]
    if not (seg_map and xarms):
        print("  (cross-checks skipped: run the mask arms 'quad'/'model' alongside M to fill them)")
        return

    def _seg_errs(gate_set, arm):
        return [seg_map[s][arm][i] for s, i, _ in gate_set if s in seg_map and i in seg_map[s].get(arm, {})]

    print(f"  CROSS-CHECK (1) -- on the {len(usable)} WELL-FRAMED gates M nails, how do the masks do "
          f"THERE? (mask model {seg_tag!r})")
    for a in xarms:
        errs = _seg_errs(usable, a)
        print(f"    {ARMS[a][0]:26s}: {len(errs):2d}/{len(usable)} matched, median {np.median(errs):.1f} px"
              if errs else f"    {ARMS[a][0]:26s}: none matched")
    print(f"  CROSS-CHECK (2) -- on the {len(dropped)} gates M drops, do the masks recover a centre?")
    for a in xarms:
        errs = _seg_errs(dropped, a)
        print(f"    {ARMS[a][0]:26s}: recovered {len(errs):2d}/{len(dropped)}, median error on the "
              f"recovered {np.median(errs):.1f} px" if errs else f"    {ARMS[a][0]:26s}: recovered 0")


# ==================================================================================================
# TIMING (PART 2): honest per-arm latency against the 33 ms / frame (30 Hz) flight budget.
# ==================================================================================================
# The budget question is "inference + post-processing per frame at ~2-3 gates", so the SHARED cost
# (one seg inference, or one M inference, per frame) is separated from the PER-GATE post-processing
# and both are reported. Warm up (discard the first N frames), single-thread, device named. The GPU
# is shared here (a big Blender render was running), so p90 carries the contention and the headline
# is the MEDIAN over many samples.

def _cuda_sync():
    """Make a GPU inference's wall time honest: predict() can return before the kernels retire."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:                            # noqa: BLE001
        pass


def _timing_row(label, infer_ms, post_ms):
    """One row: shared infer (per frame) + post (per gate) -> total at 2 gates + a 33 ms verdict."""
    im = np.array(infer_ms) if infer_ms else np.array([np.nan])
    pm = np.array(post_ms) if post_ms else np.array([0.0])
    infer_med, infer_p90 = float(np.median(im)), float(np.percentile(im, 90))
    post_med, post_p90 = float(np.median(pm)), float(np.percentile(pm, 90))
    total2 = infer_med + 2.0 * post_med
    return (label, f"{infer_med:5.1f} / {infer_p90:5.1f}", f"{post_med:5.2f} / {post_p90:5.2f}",
            f"{total2:5.1f}", "yes" if total2 <= 33.0 else "NO")


def run_timing(model_tags, seg_ids, m_ids, truth, device, warmup, repeats):
    imgs = [im for im in (cv2.imread(str(p)) for _, p, *_ in truth) if im is not None]
    seq = imgs * repeats
    print(f"\nTIMING -- device requested {device or 'auto (cuda:0 if avail)'!r}, single-thread, "
          f"first {warmup} frames discarded as warmup, {repeats} pass(es) over {len(imgs)} frames "
          f"({len(seq)} timed calls).\nThe GPU is SHARED (a large Blender render + other sessions "
          f"were running) -- read the MEDIAN as the number, p90 as the contention tax.")
    rows = []
    if seg_ids:
        for tag in model_tags:
            det = SegGateLineDetector.load(WEIGHTS[tag], device=device)
            infer, post = [], {a: [] for a in seg_ids}
            for k, im in enumerate(seq):
                h, w = im.shape[:2]
                t0 = time.perf_counter()
                fm, om = det.masks_for(im)
                _cuda_sync()
                dt = (time.perf_counter() - t0) * 1000.0
                warm = k < warmup
                for f, o in pair_gate_instances(fm, om):
                    for a in seg_ids:
                        t1 = time.perf_counter()
                        ARMS[a][1](f, o, (w, h))
                        if not warm:
                            post[a].append((time.perf_counter() - t1) * 1000.0)
                if not warm:
                    infer.append(dt)
            print(f"  [seg {tag}] device={det.device!r}")
            for a in seg_ids:
                rows.append(_timing_row(f"{tag} {ARMS[a][0]}", infer, post[a]))
    if m_ids:
        det, which, attempts = load_m(device)
        for w, e in attempts:
            print(f"  [M] {w} weights unavailable -> {e}")
        print(f"  [M] using {which.upper()} weights on device={det.device!r}")
        raw_ids = [a for a in m_ids if M_ARMS[a][1] != "deploy"]
        infer, post = [], {a: [] for a in m_ids}
        for k, im in enumerate(seq):
            h, w = im.shape[:2]
            t0 = time.perf_counter()
            res = det.model.predict(im, verbose=False, device=det.device)
            _cuda_sync()
            dt = (time.perf_counter() - t0) * 1000.0
            warm = k < warmup
            dets = _dets_from_res(res, (w, h))
            for rec in dets:
                for a in raw_ids:
                    t1 = time.perf_counter()
                    _m_centre(rec, M_ARMS[a][1])
                    if not warm:
                        post[a].append((time.perf_counter() - t1) * 1000.0)
            if "m-deploy" in m_ids:
                t1 = time.perf_counter()
                obs = observations_from_results(
                    Frame(k, 0, im), res[0], score_thresh=det.score_thresh,
                    kpt_conf_thresh=det.kpt_conf_thresh, use_outer=det.use_outer,
                    partial_rescue=det.partial_rescue)
                _cuda_sync()
                dtp = (time.perf_counter() - t1) * 1000.0
                if not warm:
                    n = max(len(obs), 1)
                    post["m-deploy"].extend([dtp / n] * n)
            if not warm:
                infer.append(dt)
        for a in m_ids:
            rows.append(_timing_row(f"M[{which}] {M_ARMS[a][0]}", infer, post[a]))

    hdr = ("arm", "infer ms/frame med/p90", "post ms/gate med/p90", "total ms @2 gates", "fits 33ms?")
    wid = [max(len(str(r[i])) for r in rows + [hdr]) for i in range(len(hdr))]
    print("\n" + "  ".join(h.ljust(w) for h, w in zip(hdr, wid)))
    print("  ".join("-" * w for w in wid))
    for r in rows:
        print("  ".join(str(v).ljust(w) for v, w in zip(r, wid)))
    print("  ".join("-" * w for w in wid))
    print("total @2 gates = infer(med) + 2*post(med). 33 ms = one 30 Hz frame. Raw-keypoint M post "
          "is just\ncentre_from_quad (microseconds), so M's total IS its inference; m-deploy post is "
          "the rescue geometry.")


# --- gallery ----------------------------------------------------------------------------------
# Drawn on a PADDED canvas: a centre that legitimately sits outside the image is the whole point of
# this path, and a gallery that clips it to the border shows the one thing that is not happening.

def _canvas(im):
    h, w = im.shape[:2]
    c = np.full((h + 2 * PAD_Y, w + 2 * PAD_X, 3), 28, np.uint8)
    c[PAD_Y:PAD_Y + h, PAD_X:PAD_X + w] = im
    cv2.rectangle(c, (PAD_X - 1, PAD_Y - 1), (PAD_X + w, PAD_Y + h), (70, 70, 70), 1)
    return c


def _pt(p):
    return (int(round(p[0])) + PAD_X, int(round(p[1])) + PAD_Y)


def _cross(canvas, p, colour, r=9, t=2):
    x, y = _pt(p)
    hh, ww = canvas.shape[:2]
    if not (-4 * r <= x < ww + 4 * r and -4 * r <= y < hh + 4 * r):
        return False
    cv2.line(canvas, (x - r, y), (x + r, y), colour, t, cv2.LINE_AA)
    cv2.line(canvas, (x, y - r), (x, y + r), colour, t, cv2.LINE_AA)
    return True


def _poly(canvas, q, colour, t=1):
    p = [_pt(v) for v in q]
    for i in range(len(p)):
        cv2.line(canvas, p[i], p[(i + 1) % len(p)], colour, t, cv2.LINE_AA)


def _mask_outline(canvas, mask, colour):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        cv2.drawContours(canvas, [c + np.array([[PAD_X, PAD_Y]])], -1, colour, 1, cv2.LINE_AA)


_PANEL_LABEL = {
    "line": "lines -> homography",
    "quad": "hull quad -> diagonals",
    "model": "MODEL FIT: 3-D gate silhouette",
    "model-flat": "model fit, flat (depth 0)",
    "quad-strict": "hull quad, no clipped fits",
}


def _panel(rec, which):
    c = _canvas(rec["im"])
    for f, o in rec["pairs"]:
        _mask_outline(c, f, (90, 90, 60))
        if o is not None:
            _mask_outline(c, o, (120, 110, 40))
    for q in rec["gt_quads"]:
        _poly(c, q, (170, 170, 170))
    for g in rec["gt"]:
        _cross(c, g, (255, 255, 255), r=11, t=2)
    for cen, extra in rec["preds"].get(which, []):
        if which.startswith("model"):
            # Draw the FITTED SILHOUETTE, not just the centre: the whole claim of this arm is that
            # it settled the KNOWN SHAPE onto the mask, and only the shape shows whether it did.
            # Cyan = the amodal outer silhouette, green = the see-through opening.
            if extra.frame_poly is not None:
                _poly(c, extra.frame_poly, (255, 200, 0), 2)
            if extra.opening_poly is not None and len(extra.opening_poly) >= 3:
                _poly(c, extra.opening_poly, (120, 255, 120), 2)
            _cross(c, cen, (255, 0, 255))
        elif which.startswith("quad") and extra is not None:
            # ORANGE = the quad still has an image-border edge, i.e. only 3 gate edges were visible
            # and this is a fit to the crop, biased inward. Magenta = all four edges are gate edges.
            clipped = any(_on_border(extra[i], extra[(i + 1) % 4], *rec["im"].shape[1::-1])
                          for i in range(4))
            col = (0, 165, 255) if clipped else (255, 0, 255)
            _poly(c, extra, col, 2)
            _cross(c, cen, col)
        else:
            _cross(c, cen, (255, 0, 255))
    cv2.putText(c, _PANEL_LABEL.get(which, which), (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (220, 220, 220), 1, cv2.LINE_AA)
    return c


def _score(rec, key):
    """Worst-first key for one arm: unmatched labelled gates rank above any finite error."""
    d = rec["per_gt"].get(key, {})
    n_unmatched = len(rec["gt"]) - len(d)
    return (-n_unmatched, -(max(d.values()) if d else 0.0))


def write_gallery(out_dir: Path, per_frame, tag, arm_ids, sort_by):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "img").mkdir(exist_ok=True)
    recs = sorted(per_frame, key=lambda r: _score(r, sort_by))
    cards = []
    for k, rec in enumerate(recs):
        panels = [_panel(rec, a) for a in arm_ids]
        sep = np.full((panels[0].shape[0], 4, 3), 60, np.uint8)
        side = panels[0]
        for p in panels[1:]:
            side = np.concatenate([side, sep, p], axis=1)
        name = f"{k:03d}_{rec['stem'][:70]}.jpg"
        cv2.imwrite(str(out_dir / "img" / name), side, [cv2.IMWRITE_JPEG_QUALITY, 86])
        lines = []
        for a in arm_ids:
            d = rec["per_gt"].get(a, {})
            miss = len(rec["gt"]) - len(d)
            lines.append(f'<span class=a>{html.escape(ARMS[a][0])}</span> err '
                         f'{", ".join(f"{v:.0f}" for v in d.values()) or "-"} px'
                         f'{f" &middot; <b>{miss} unmatched</b>" if miss else ""}')
        cards.append(
            f'<div class=c><img src="img/{html.escape(name)}" loading=lazy>'
            f'<div class=m><b>{html.escape(rec["stem"])}</b> &middot; '
            f'{len(rec["gt"])} labelled gate(s)<br>' + "<br>".join(lines) + "</div></div>")
    (out_dir / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>centre A/B " + tag + "</title>"
        "<style>body{background:#141414;color:#ddd;font:14px/1.45 system-ui,sans-serif;margin:24px}"
        "h1{font-size:18px;font-weight:600}p{color:#999;max-width:78em}"
        ".c{margin:0 0 26px}img{width:100%;max-width:2400px;display:block;border:1px solid #333}"
        ".m{color:#aaa;padding:6px 2px}.a{color:#f4f}"
        "code{color:#ccc}</style>"
        f"<h1>gate-centre A/B &mdash; {html.escape(tag)}, worst-first for "
        f"<code>{html.escape(sort_by)}</code></h1>"
        "<p>Panels left to right: " + " &middot; ".join(
            f"<b>{html.escape(ARMS[a][0])}</b>" for a in arm_ids) + ". "
        "<b>White cross</b> = hand-labelled ground truth (diagonal intersection of the labelled "
        "inner quad), <b>white outline</b> = that labelled quad, <b>magenta cross</b> = the "
        "solver's centre. Olive outlines are the predicted masks EVERY arm was fed. The frame is "
        "inset in a padded canvas so a centre that legitimately falls outside the image is still "
        "drawn where it actually is.</p>"
        "<p>On the <b>model-fit</b> panel the <b style='color:#0c8'>amber outline</b> is the fitted "
        "3-D gate's outer silhouette and the <b style='color:#7f7'>green outline</b> is its "
        "see-through opening &mdash; i.e. where the solver believes the known gate actually is. "
        "Judge this arm by whether that shape settled on the gate, not only by the cross.</p>"
        "<p><b style='color:#fa5'>Orange</b> on a quad panel marks a quad that still has an image "
        "border edge: only three gate edges were visible, so nothing could be extended and the fit "
        "is the visible trapezoid, biased toward the middle of the frame.</p>"
        + "".join(cards), encoding="utf-8")
    return out_dir / "index.html"


# ==================================================================================================
# EXPERIMENT: does the gate's 0.26 m DEPTH break the IPPE 2-fold ambiguity?
# ==================================================================================================
# Scope, deliberately narrow. The 2-fold ambiguity does NOT affect the centre pixel (both poses
# project it identically -- that is what makes them ambiguous), it does NOT affect the bearing (the
# two translations are parallel, since the gate origin must land on the same pixel), and it does not
# affect range-from-size (it flips the SIGN of the tilt, not its magnitude). The only output it
# corrupts is the gate's ORIENTATION -- its normal / relative yaw. So that is the only thing this
# experiment asks about.
#
# The hypothesis: planar IPPE is ambiguous because a FLAT target's two poses project identically.
# This gate is 0.26 m DEEP, so the two solutions expose different inner side walls and produce
# genuinely different silhouettes -- an area fit may therefore break an ambiguity no planar method
# can. The FLAT model is the control: on it the two branches must tie, or the experiment is broken.

def _rot_geodesic(a, b):
    return float(np.arccos(np.clip((np.trace(a.T @ b) - 1.0) / 2.0, -1.0, 1.0)))


def run_ambiguity(n: int = 240, seed: int = 3, noise_px: float = 0.0):
    from racer.frames import CAMERA_INTRINSICS_K as K
    from racer.vision.gate_model_fit import (DEFAULT_MODEL, FLAT_MODEL, _ippe_candidates,
                                             _Target, project_silhouette, score_pose,
                                             range_from_apparent_size)
    rng = np.random.default_rng(seed)
    W, H = 640, 360
    recs = []
    while len(recs) < n:
        d = rng.uniform(2.0, 14.0)
        tilt = rng.uniform(0.05, 0.9)                    # the ambiguity is trivial at zero tilt
        axis = rng.uniform(0, 2 * np.pi)
        rv = np.array([np.cos(axis) * tilt, np.sin(axis) * tilt, rng.uniform(-0.2, 0.2)])
        R = cv2.Rodrigues(rv)[0]
        t = np.array([rng.uniform(-.2, .2) * d, rng.uniform(-.15, .15) * d, d])
        got = project_silhouette(R, t, K, DEFAULT_MODEL)
        if got is None:
            continue
        fp, op = got
        fm = np.zeros((H, W), np.uint8); cv2.fillPoly(fm, [fp.round().astype(np.int32)], 1)
        om = np.zeros((H, W), np.uint8)
        if len(op) >= 3:
            cv2.fillPoly(om, [np.asarray(op).round().astype(np.int32)], 1)
        if fm.sum() < 800 or om.sum() < 200:
            continue
        # the PLANAR observation the ambiguity comes from: the 4 inner-square corners
        h = DEFAULT_MODEL.inner_m / 2.0
        sq = np.array([[-h, h, 0.], [h, h, 0.], [h, -h, 0.], [-h, -h, 0.]])
        cam = sq @ R.T + t
        uv = (cam @ K.T)[:, :2] / (cam @ K.T)[:, 2:3]
        if noise_px > 0:
            uv = uv + rng.normal(0, noise_px, uv.shape)
        cands = _ippe_candidates(uv, DEFAULT_MODEL.inner_m, K)
        if len(cands) != 2:
            continue
        # THE BASELINE DISCRIMINATOR, and it must be in this table or the experiment is meaningless:
        # IPPE's own reprojection-error ranking, which gate_pose._estimate_ippe already uses. The
        # planar 2-fold ambiguity is only EXACT under weak perspective; under full perspective the
        # wrong branch reprojects slightly worse, and that residual is what the existing code reads.
        # The question is therefore not "can anything break the tie" but "does the 0.26 m depth
        # break it BETTER, and does it survive corner noise where the reprojection residual does not".
        rep = [float(np.sqrt(np.mean(np.sum((
            ((sq @ Rc.T + tc) @ K.T)[:, :2] / ((sq @ Rc.T + tc) @ K.T)[:, 2:3] - uv) ** 2, axis=1))))
            for Rc, tc in cands]
        tgt = _Target(fm, om, (W, H), 2.0)
        s_depth = [score_pose(Rc, tc, tgt, K, DEFAULT_MODEL) for Rc, tc in cands]
        s_flat = [score_pose(Rc, tc, tgt, K, FLAT_MODEL) for Rc, tc in cands]
        # "true" branch = the one whose ORIENTATION matches, up to the gate's own 90-degree symmetry
        def orient_err(Rc):
            best = np.pi
            for k in range(4):
                Rz = cv2.Rodrigues(np.array([0., 0., k * np.pi / 2]))[0]
                best = min(best, _rot_geodesic(Rc @ Rz, R))
            return best
        oe = [orient_err(Rc) for Rc, _ in cands]
        true_i = int(np.argmin(oe))
        recs.append(dict(
            tilt=float(tilt), d=float(d),
            ippe_ok=bool(np.argmin(rep) == true_i),
            depth_ok=bool(np.argmax(s_depth) == true_i),
            flat_ok=bool(np.argmax(s_flat) == true_i),
            depth_margin=float(s_depth[true_i] - s_depth[1 - true_i]),
            flat_margin=float(s_flat[true_i] - s_flat[1 - true_i]),
            wrong_orient_deg=float(np.degrees(oe[1 - true_i])),
            # branch invariance of the outputs the coordinator asked us to emit
            d_rng_size=float(abs(range_from_apparent_size(*cands[0], K, DEFAULT_MODEL.inner_m)
                                 - range_from_apparent_size(*cands[1], K, DEFAULT_MODEL.inner_m))),
            d_t_norm=float(abs(np.linalg.norm(cands[0][1]) - np.linalg.norm(cands[1][1]))),
            d_bearing_deg=float(np.degrees(np.arccos(np.clip(np.dot(
                cands[0][1] / np.linalg.norm(cands[0][1]),
                cands[1][1] / np.linalg.norm(cands[1][1])), -1, 1)))),
            rng_true=float(np.linalg.norm(t)),
        ))
    r = {k: np.array([x[k] for x in recs]) for k in recs[0]}
    print(f"\nAMBIGUITY EXPERIMENT -- {len(recs)} synthetic poses, 2-14 m, tilt 3-52 deg, "
          f"corner noise {noise_px:.1f} px.\nMasks are the TRUE rendered silhouette of the 3-D "
          f"model; the two IPPE branches are scored against them.\n")
    print(f"  IPPE reprojection ranking (what ships today)     : {100 * r['ippe_ok'].mean():5.1f}%")
    print(f"  FLAT silhouette IoU  -- the shape-only control   : {100 * r['flat_ok'].mean():5.1f}%"
          f"   median IoU margin {np.median(r['flat_margin']):+.4f}")
    print(f"  3-D silhouette IoU (depth 0.26 m)                : {100 * r['depth_ok'].mean():5.1f}%"
          f"   median IoU margin {np.median(r['depth_margin']):+.4f}")
    print(f"  (picking the WRONG branch costs a median "
          f"{np.median(r['wrong_orient_deg']):.1f} deg of gate orientation)")
    print("\n  by tilt:")
    for lo, hi in ((0.05, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.9)):
        m = (r["tilt"] >= lo) & (r["tilt"] < hi)
        if m.sum():
            print(f"    {np.degrees(lo):4.0f}-{np.degrees(hi):4.0f} deg  n={int(m.sum()):3d}  "
                  f"IPPE {100 * r['ippe_ok'][m].mean():5.1f}%  flat {100 * r['flat_ok'][m].mean():5.1f}%"
                  f"  3-D {100 * r['depth_ok'][m].mean():5.1f}%"
                  f"   3-D margin {np.median(r['depth_margin'][m]):+.4f}")
    print("\n  BRANCH INVARIANCE of the emitted quantities (difference between the two branches).")
    print("  NB the two branches are NOT an exact tie for this target -- see the note above -- so")
    print("  these differences are small but not zero, and they BOUND what the ambiguity can cost:")
    print(f"    bearing direction        median {np.median(r['d_bearing_deg']):.3f} deg")
    print(f"    range from apparent SIZE median {np.median(r['d_rng_size']):.4f} m  "
          f"p90 {np.percentile(r['d_rng_size'], 90):.4f} m")
    print(f"    range from ||t||         median {np.median(r['d_t_norm']):.4f} m  "
          f"p90 {np.percentile(r['d_t_norm'], 90):.4f} m")
    return recs


# ==================================================================================================
# EXPERIMENT: range from apparent SIZE vs the fitted ||t||, against task2 GROUND-TRUTH range
# ==================================================================================================
def run_task2(weights, device, bundle=None):
    from racer.vision.gate_model_fit import fit_gate_model
    bundle = Path(bundle or ROOT / "handoff/shadowpc-followups-2026-06-05/task2_frames")
    meta = json.loads((bundle / "frames.json").read_text())
    det = SegGateLineDetector.load(weights, device=device)
    rows = []
    for fr in meta["frames"]:
        im = cv2.imread(str(bundle / fr["png"]))
        if im is None:
            continue
        h, w = im.shape[:2]
        # Take the LARGEST instance, not the best-scoring one. The bundle's ground truth is the
        # range to GATE 0, the gate being approached; the frames also contain gates further down
        # the course, and a tiny distant mask is TRIVIALLY easy to cover (measured: score 1.00 at
        # 38 m), so ranking by score picks the wrong gate and manufactures a +15 m bias.
        best, best_area = None, 0
        for f, o in pair_gate_instances(*det.masks_for(im)):
            a = int(np.count_nonzero(np.asarray(f)))
            if a <= best_area:
                continue
            fit = fit_gate_model(f, o, (w, h))
            if fit is not None:
                best, best_area = fit, a
        if best is None:
            continue
        rows.append((fr["range_m"], best.range_m, best.t_norm_m, best.range_from_size_ok))
    if not rows:
        print("task2: no fits")
        return
    gt = np.array([r[0] for r in rows]); rs = np.array([r[1] for r in rows])
    rt = np.array([r[2] for r in rows]); ok = np.array([r[3] for r in rows])
    print(f"\nTASK2 RANGE CHECK -- {len(rows)}/{len(meta['frames'])} frames fitted, "
          f"GT range {gt.min():.1f}-{gt.max():.1f} m (bundle's own per-frame range_m).")
    for name, v in (("range from apparent SIZE (ambiguity-safe)", rs),
                    ("||t|| of the fitted pose  (NOT safe)     ", rt)):
        e = v - gt
        print(f"  {name}: bias {e.mean():+6.2f} m  median |err| {np.median(np.abs(e)):5.2f} m  "
              f"p90 {np.percentile(np.abs(e), 90):5.2f} m  rel {np.median(np.abs(e) / gt) * 100:4.1f}%")
    print("  by range bin (median |err| m, then as % of range):")
    for lo, hi in ((0, 4), (4, 10), (10, 25)):
        m = (gt >= lo) & (gt < hi)
        if not m.any():
            continue
        es, et = np.abs(rs[m] - gt[m]), np.abs(rt[m] - gt[m])
        print(f"    {lo:2d}-{hi:2d} m  n={int(m.sum()):2d}   size {np.median(es):5.2f} m "
              f"({np.median(es / gt[m]) * 100:4.1f}%)   ||t|| {np.median(et):5.2f} m "
              f"({np.median(et / gt[m]) * 100:4.1f}%)")
    print(f"  ({int((~ok).sum())} frames fell back to ||t|| because the inner square could not be "
          f"fully projected)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default=DEFAULT_BATCH)
    ap.add_argument("--val", default=DEFAULT_VAL)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--device", default=None, help="'cpu' to stay off a busy GPU")
    ap.add_argument("--models", default="v2,s1")
    ap.add_argument("--gallery-model", default="v2")
    ap.add_argument("--no-gallery", action="store_true")
    ap.add_argument("--arms", default="line,quad,model",
                    help=f"comma-separated subset of mask arms {sorted(ARMS)} and/or keypoint "
                         f"arms {sorted(M_ARMS)} (M is model-independent -- run once)")
    ap.add_argument("--gallery-sort", default="model", help="arm to order the gallery worst-first")
    ap.add_argument("--timing", action="store_true",
                    help="run ONLY the PART-2 latency table (infer + post per arm vs the 33 ms budget)")
    ap.add_argument("--timing-warmup", type=int, default=5, help="frames discarded before timing")
    ap.add_argument("--timing-repeats", type=int, default=3,
                    help="passes over the frame set (medians over more samples ride out GPU contention)")
    ap.add_argument("--ambiguity", action="store_true",
                    help="run ONLY the 3-D-depth vs IPPE 2-fold ORIENTATION ambiguity experiment")
    ap.add_argument("--ambiguity-noise", type=float, default=0.0,
                    help="corner noise (px) fed to IPPE in the ambiguity experiment")
    ap.add_argument("--task2", action="store_true",
                    help="run ONLY the task2 ground-truth RANGE check (size-based vs ||t||)")
    ap.add_argument("--task2-bundle", default=None,
                    help="task2 frame bundle; the PNGs are gitignored, so they may live in another "
                         "checkout (C:/Users/Shadow/Peregrine/handoff/.../task2_frames)")
    ap.add_argument("--gt-from-handles", action="store_true",
                    help="use the OLD ground truth (diagonal of the hand handles). The handles were "
                         "placed for AREA, so this is wrong wherever a corner is off-frame; kept "
                         "only so the corrected and uncorrected tables can be compared.")
    args = ap.parse_args()
    globals()["_GT_FROM_HANDLES"] = bool(args.gt_from_handles)
    if args.ambiguity:
        run_ambiguity(noise_px=args.ambiguity_noise)
        return 0
    if args.task2:
        run_task2(WEIGHTS[args.models.split(",")[0].strip()], args.device, args.task2_bundle)
        return 0
    arm_ids = [a.strip() for a in args.arms.split(",") if a.strip()]
    bad = [a for a in arm_ids if a not in ARMS and a not in M_ARMS]
    if bad:
        ap.error(f"unknown arm(s) {bad}; mask arms {sorted(ARMS)}, keypoint arms {sorted(M_ARMS)}")
    seg_ids = [a for a in arm_ids if a in ARMS]       # mask arms: run per seg model
    m_ids = [a for a in arm_ids if a in M_ARMS]       # keypoint arms: model-independent, run once

    batch = Path(args.batch)
    stems = [s.strip() for s in Path(args.val).read_text().splitlines() if s.strip()]
    truth, missing = load_truth(batch, stems)
    n_gates = sum(len(c) for _, _, c, _, _ in truth)
    n_off = sum(1 for _, _, cs, _, _ in truth for c in cs
                if not (0 <= c[0] < 640 and 0 <= c[1] < 360))
    n_small = sum(1 for _, _, _, _, b in truth for v in b if not v)
    print(f"{len(truth)}/{len(stems)} val frames carry at least one labelled gate "
          f"({len(missing)} skipped: no geometry sidecar, or a negative frame), "
          f"{n_gates} labelled gates.\n"
          f"  {n_off} have an OFF-SCREEN centre; {n_small} are below the 200 px^2 mask-area floor "
          f"both solvers apply (a few pixels across).\n")

    if args.timing:
        run_timing([t.strip() for t in args.models.split(",") if t.strip()],
                   seg_ids, m_ids, truth, args.device, args.timing_warmup, args.timing_repeats)
        return 0

    rows, gallery, off_rows, time_rows = [], None, [], []
    seg_per_frame_first, seg_tag_first = None, None
    if seg_ids:
        for tag in [t.strip() for t in args.models.split(",") if t.strip()]:
            arms, per_frame, timing, init_ms, diag = run_model(
                tag, WEIGHTS[tag], truth, args.device, seg_ids)
            if seg_per_frame_first is None:
                seg_per_frame_first, seg_tag_first = per_frame, tag
            rows += [a.row() for a in arms.values()]
            for a in seg_ids:
                e = arms[a].err_gt_offscreen
                off_rows.append((f"{tag} {ARMS[a][0]}", f"{len(e)}/{n_off}",
                                 f"{np.median(e) if e else float('nan'):8.1f}",
                                 f"{np.percentile(e, 90) if e else float('nan'):8.1f}"))
                t = np.array(timing[a]) if timing[a] else np.array([np.nan])
                time_rows.append((f"{tag} {ARMS[a][0]}", f"{len(t):4d}",
                                  f"{np.median(t):7.1f}", f"{np.percentile(t, 90):7.1f}"))
            if diag["evals"]:
                ev, am = np.array(diag["evals"]), np.array(diag["ambig"])
                print(f"[{tag}] model fit: {len(ev)} gates, evals med {np.median(ev):.0f} "
                      f"(max {ev.max()}), final IoU frame {np.nanmedian(diag['iou_f']):.3f} / opening "
                      f"{np.nanmedian(diag['iou_o']):.3f}; init {diag['init_kind']}; "
                      f"IPPE-branch score margin med {np.median(am):.4f} "
                      f"(>0 on {100.0 * float((am > 1e-6).mean()):.0f}% -- see the ambiguity note)")
                print(f"[{tag}] shared init (quad_from_mask_ex, ALSO paid by the quad arm): "
                      f"med {np.median(init_ms):.1f} ms of the model arm's per-gate total")
            if not args.no_gallery and tag == args.gallery_model:
                sort_by = args.gallery_sort if args.gallery_sort in seg_ids else seg_ids[-1]
                gallery = write_gallery(Path(args.out), per_frame, tag, seg_ids, sort_by)

    m_per_frame, m_which = None, None
    if m_ids:
        m_arms, m_per_frame, m_which = run_m(truth, args.device, m_ids)
        rows += [a.row() for a in m_arms.values()]
        for a in m_ids:
            e = m_arms[a].err_gt_offscreen
            off_rows.append((f"M[{m_which}] {M_ARMS[a][0]}", f"{len(e)}/{n_off}",
                             f"{np.median(e) if e else float('nan'):8.1f}",
                             f"{np.percentile(e, 90) if e else float('nan'):8.1f}"))

    def _table(hdr, rows_):
        wid = [max(len(str(r[i])) for r in rows_ + [hdr]) for i in range(len(hdr))]
        rule = "  ".join("-" * w for w in wid)
        print("\n" + "  ".join(h.ljust(w) for h, w in zip(hdr, wid)))
        print(rule)
        for r in rows_:
            print("  ".join(str(v).ljust(w) for v, w in zip(r, wid)))
        print(rule)

    _table(("arm", "matched", "cov", "cov>=floor", "med px", "p90 px", "preds", "off-sc"), rows)
    print("cov = labelled gates that got a ONE-TO-ONE matched prediction; cov>=floor is the same "
          "over gates\nabove the 200 px^2 area floor; med/p90 are over MATCHED pairs only.\n"
          "preds = centres emitted over all frames; off-sc = those that land outside the image.\n"
          "M arms read RAW keypoints: 'raw kpts' emit a centre for every detection (a clamped inner\n"
          "corner is scored, not hidden); '-clean' emit only when all 4 kpts are in-frame AND conf>= "
          f"{M_KPT_CONF_THRESH}.")

    _table(("arm", f"matched/{n_off} OFF-SCREEN gt", "med px", "p90 px"), off_rows)
    print(f"THE OFF-SCREEN SUBSET: the {n_off} labelled gates whose TRUE centre falls outside the "
          "640x360 image.\nThis is the configuration the quad path cannot represent (its clipped "
          "hull is already a\nquadrilateral, so the fit is the visible trapezoid, biased inward).")

    if time_rows:
        _table(("arm", "gates", "med ms", "p90 ms"), time_rows)
        print("per-GATE wall time, single-threaded CPU, EXCLUDING seg inference (shared by all arms)."
              "\nThis is the per-solver POST cost only; the full latency budget is `--timing`.")

    if m_ids and m_per_frame is not None:
        report_m_structural(m_per_frame, seg_per_frame_first, seg_tag_first or "", n_gates)

    if gallery:
        print(f"\ngallery: {gallery}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
