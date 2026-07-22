"""LINE-BASED gate detection -- an INDEPENDENT second path, NOT wired into flight.

STATUS (2026-07-21): the geometry is VALIDATED, the segmentation front-end is NOT.
  * against task2 ground truth: 34/40 frames, centre error 13.7 px median / 32.3 px p90, and
    FULL coverage close in (7/7 at 0-4 m, 13/13 at 4-8 m) where the keypoint path is weakest --
    at 1.8 m the keypoint path returns a 23.9 m pose error while this returns ~0.13 m.
  * on the failure-mined inbox it produced a centre on 41.8% of the frames where keypoints fail,
    but disagrees with the keypoint centre by 100 px median on single-gate frames. That is NOT
    gate-mismatch (multi-gate frames disagree LESS, 56 px) -- it is the colour mask breaking on
    close/oblique/blurred imagery.
CONCLUSION: the chain mask -> lines -> homography -> centre is correct; the weak link is the MASK.
Replacing the HSV mask with a learned yolo-seg mask should carry the whole path over, because
everything downstream of the mask is already measured against real ground truth. Until then this
stays out of the flight loop and the keypoint path remains the only emitter.


Why lines: a corner that leaves the frame must be EXTRAPOLATED, and ultralytics clamps it to the
border anyway, so point methods die on cropped gates (317/428 dropped detections have <=3 usable
keypoints). A LINE is fixed by the in-frame pixels it passes through, so it survives cropping.

ASSIGNMENT is the hard part -- deciding which line is which gate edge. Solved structurally here:
the gate is a square ANNULUS, so in the mask the OUTER contour is the 2.72 m square and the HOLE
inside it is the 1.5 m opening. No cross-ratio bookkeeping, no matching against surface markings:
the checkerboard is interior texture, closed over by morphology, and never reaches either contour.

Border segments are DISCARDED: where the mask is cut by the image edge, the contour runs along the
frame boundary. That is an artefact of the crop, not a gate edge, and fitting it would be exactly
the "collapsed to the border" error the keypoint path already suffers.

Pose comes from the LINES, not from intersecting them: a homography maps lines by H^-T, so 4
identified lines determine the gate plane directly, and corners (in frame or not) fall out of H.
"""
from __future__ import annotations

import cv2
import numpy as np

GATE_INNER_HALF = 0.75
GATE_OUTER_HALF = 1.36
BORDER_EPS = 3.0        # a contour segment this close to the image edge is a crop artefact
MIN_SEG_PX = 25.0       # shorter than this and the direction is noise
MIN_AREA_FRAC = 0.010   # ignore specks


def gate_mask(bgr, close_px: int = 15):
    """Saturated orange/red gate structure. ``close_px`` must exceed the checkerboard square size
    so the surface markings are filled and never punch a false hole in the annulus."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    m = (((h <= 25) | (h >= 160)) & (s >= 90) & (v >= 70)).astype(np.uint8) * 255
    k = np.ones((close_px, close_px), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return m


def _on_border(p, q, w, h):
    """True if the whole segment hugs one image edge (i.e. it is a crop artefact, not a gate edge)."""
    for lo, idx in ((0.0, 0), (0.0, 1)):
        if abs(p[idx] - lo) < BORDER_EPS and abs(q[idx] - lo) < BORDER_EPS:
            return True
    if abs(p[0] - (w - 1)) < BORDER_EPS and abs(q[0] - (w - 1)) < BORDER_EPS:
        return True
    if abs(p[1] - (h - 1)) < BORDER_EPS and abs(q[1] - (h - 1)) < BORDER_EPS:
        return True
    return False


def _segments(contour, w, h):
    """Polygon-approximate a contour and return its non-border, long-enough edge segments."""
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.012 * peri, True).reshape(-1, 2).astype(float)
    out = []
    n = len(approx)
    for i in range(n):
        p, q = approx[i], approx[(i + 1) % n]
        if np.linalg.norm(q - p) < MIN_SEG_PX or _on_border(p, q, w, h):
            continue
        out.append((p, q))
    return out


def _line(p, q):
    """Homogeneous image line through two points, unit (a,b) and a CANONICAL sign.

    l and -l are the same line, so the signed point-line distance l @ [x,y,1] is only meaningful
    once the sign is pinned. Fixing (a,b) to the upper half-plane makes that distance comparable
    across lines -- without it, opposite edges (equidistant from the gate centroid, opposite sides)
    look identical under |d| and get merged into one."""
    l = np.cross([p[0], p[1], 1.0], [q[0], q[1], 1.0])
    n = np.hypot(l[0], l[1])
    if n < 1e-9:
        return None
    l = l / n
    if l[1] < -1e-12 or (abs(l[1]) <= 1e-12 and l[0] < 0):
        l = -l
    return l


def extract_gate_lines(mask):
    """Return (inner_segs, outer_segs) -- the opening's edges and the frame's outer edges."""
    h, w = mask.shape[:2]
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None or not len(cnts):
        return [], []
    hier = hier[0]
    best_i, best_a = -1, MIN_AREA_FRAC * w * h
    for i, c in enumerate(cnts):
        if hier[i][3] == -1 and cv2.contourArea(c) > best_a:
            best_i, best_a = i, cv2.contourArea(c)
    if best_i < 0:
        return [], []
    outer = _segments(cnts[best_i], w, h)
    # largest HOLE of that component = the gate opening
    hole, hole_a = None, 0.0
    j = hier[best_i][2]
    while j != -1:
        a = cv2.contourArea(cnts[j])
        if a > hole_a:
            hole, hole_a = cnts[j], a
        j = hier[j][0]
    inner = _segments(hole, w, h) if hole is not None and hole_a > 0.2 * best_a else []
    return inner, outer


def _families_index(segs):
    """Family label (0/1) per segment -- the SHARED clustering used for inner and outer together."""
    if len(segs) < 2:
        return [0] * len(segs)
    ang = np.array([np.arctan2(q[1] - p[1], q[0] - p[0]) % np.pi for p, q in segs])
    z = np.stack([np.cos(2 * ang), np.sin(2 * ang)], 1)      # doubled angle: lines are pi-periodic
    c = np.stack([z[0], z[int(np.argmin(z @ z[0]))]])
    lab = np.zeros(len(segs), int)
    for _ in range(12):
        lab = np.argmax(z @ c.T, 1)
        for k in (0, 1):
            if (lab == k).any():
                v = z[lab == k].mean(0)
                nv = np.linalg.norm(v)
                if nv > 1e-9:
                    c[k] = v / nv
    return list(lab)


def _families(segs):
    """Split segments into the gate's two edge families (plane-parallel pencils) by direction."""
    if len(segs) < 2:
        return [segs, []]
    ang = np.array([np.arctan2(q[1] - p[1], q[0] - p[0]) % np.pi for p, q in segs])
    # circular 2-means on the doubled angle (lines are pi-periodic)
    z = np.stack([np.cos(2 * ang), np.sin(2 * ang)], 1)
    c = np.stack([z[0], z[int(np.argmin(z @ z[0]))]])
    for _ in range(12):
        lab = np.argmax(z @ c.T, 1)
        for k in (0, 1):
            if (lab == k).any():
                v = z[lab == k].mean(0)
                nv = np.linalg.norm(v)
                if nv > 1e-9:
                    c[k] = v / nv
    return [[segs[i] for i in np.flatnonzero(lab == k)] for k in (0, 1)]


def _merge_collinear(lines, ref, tol_px=8.0):
    """Collapse duplicate lines (approxPolyDP often splits one edge into collinear pieces).
    Two lines are the same edge if they pass within ``tol_px`` of each other at the gate centroid."""
    keep = []
    for l in lines:
        d = float(l @ [ref[0], ref[1], 1.0])          # SIGNED (see _line's canonical sign)
        for i, (l2, d2) in enumerate(keep):
            if abs(d - d2) < tol_px and float(np.dot(l[:2], l2[:2])) > 0.985:
                if abs(d) > abs(d2):          # keep the one further out: the true silhouette edge
                    keep[i] = (l, d)
                break
        else:
            keep.append((l, d))
    return [l for l, _ in keep]


def _at(l, coord, value, axis):
    """Where line ``l`` crosses the image row/column ``value``. axis=0 -> given y, return x."""
    a, b, c = l
    if axis == 0:
        return None if abs(a) < 1e-9 else -(b * value + c) / a
    return None if abs(b) < 1e-9 else -(a * value + c) / b


def homography_from_lines(inner_segs, outer_segs, image_wh=(640, 360)):
    """Gate-plane -> image homography from the gate's edge lines. Returns H or None.

    WHY THERE IS NO ORIENTATION PRIOR HERE. A square's CENTRE is invariant under every labelling
    confusion that worries you at first glance: relabelling the edges by any symmetry of the square
    replaces H with H@R where R fixes the plane origin, so H@R@(0,0,1) == H@(0,0,1) -- same centre.
    Confusing the INNER square for the OUTER one merely rescales an axis about the origin, which
    also fixes it. So the centre survives rotation, reflection and scale errors alike.

    What it does NOT survive is INCONSISTENCY between the two squares. Clustering inner and outer
    separately (the earlier version) let inner's first pencil be the vertical edges while outer's
    was the horizontal ones, so both were declared "plane x" -- not a symmetry but a contradiction,
    and the joint fit collapses (measured: 98 px median centre error, outliers past 3000 px). That,
    not any real ambiguity, was the bug.

    So: cluster ALL lines from BOTH squares in ONE pass. The pencils are then shared, inner and
    outer are automatically consistent, and every remaining degree of freedom is a symmetry the
    centre ignores -- which also means no upright/roll assumption is needed at all.

    A homography maps lines by H^-T, so each identified line gives 2 linear constraints on H^-T and
    >=4 lines determine it -- no corner intersection, so foreshortened edges degrade gracefully.

    KNOWN GAP, and a FAILED fix -- do not re-attempt it the same way (2026-07-22). A square
    contributes only when BOTH its extremes are present in a pencil, so a gate cropped on two edges
    (one line per square per pencil, e.g. outer-left + inner-left and nothing else) gets no fit at
    all, on exactly the close-range frames this path exists to serve. Two generalisations were
    written, unit-tested green, and MEASURED against the 998-frame failure-mined inbox:
      * label by monotone order (concentric squares nest, so the plane coordinate is monotone in
        signed distance): 82.4% -> 6.8% coverage. Strict monotonicity is unattainable once mask
        noise leaves near-duplicate lines, so the whole pencil is discarded. It is also UNSOUND:
        with both lines on the same side, "-1.36 then +0.75" and "-1.36 then -0.75" are both
        monotone but are different geometries, not a symmetry -- the rule picks arbitrarily.
      * sign from the gate centroid, magnitude from the source square: 82.4% -> 72.8%, a strict
        SUBSET of the old fits (they agree to 0.00 px median where both fire, so the geometry is
        right) plus one 104000 px outlier. Both extremes can land on the same side of a biased
        centroid, giving two lines the SAME plane coordinate and a degenerate system.
    The sign genuinely is not recoverable from the LINES alone here; it needs the interior direction,
    which only the mask knows. Pass that in before trying again.
    """
    tagged = [(s, GATE_INNER_HALF) for s in inner_segs] + [(s, GATE_OUTER_HALF) for s in outer_segs]
    if len(tagged) < 4:
        return None
    segs = [s for s, _ in tagged]
    cen = np.mean([np.mean([p, q], 0) for p, q in segs], 0)
    fam_of = _families_index(segs)                 # ONE clustering over inner+outer together
    rows, used = [], 0
    for fi in (0, 1):
        # group this pencil's lines by which square they belong to, keeping the shared axis
        fam_lines = [(_line(seg[0], seg[1]), half)
                     for (seg, half), f in zip(tagged, fam_of) if f == fi]
        fam_lines = [(l, half) for l, half in fam_lines if l is not None]
        if not fam_lines:
            continue
        # Align every normal in the pencil to a SHARED reference before using signed distance.
        # _line's fixed half-plane convention is ill-conditioned for a pencil that happens to run
        # near it (for "b >= 0" that is vertical lines, where b ~ 0 and its sign is numerical
        # noise): two opposite edges then get opposing normals and BOTH signed distances come out
        # positive, so sorting cannot separate -half from +half. Referencing the pencil itself is
        # orientation-free and keeps inner and outer mutually consistent, which is what the centre
        # actually depends on.
        ref = fam_lines[0][0][:2]
        fam_lines = [((l if float(np.dot(l[:2], ref)) >= 0.0 else -l), half) for l, half in fam_lines]
        per_half: dict[float, list] = {}
        for l, half in fam_lines:
            per_half.setdefault(half, []).append(l)
        for half, lines in per_half.items():
            lines = _merge_collinear(lines, cen)
            if len(lines) < 2:
                continue                 # see KNOWN GAP in the docstring
            order = sorted(lines, key=lambda l: float(l @ [cen[0], cen[1], 1.0]))
            for l, val in ((order[0], -half), (order[-1], +half)):
                # fi picks WHICH plane axis; the choice is arbitrary but now SHARED by both
                # squares, and any residual swap/flip is a square symmetry the centre ignores.
                pl = (np.array([1.0, 0.0, -val]) if fi == 0 else np.array([0.0, 1.0, -val]))
                rows.append((pl, l))
                used += 1
    if used < 4:
        return None
    # l_img ~ H^-T l_plane  =>  cross(l_img, H^-T l_plane) = 0, linear in G = H^-T
    A = []
    for pl, li in rows:
        a, b, c = li
        A.append(np.concatenate([np.zeros(3), -c * pl, b * pl]))
        A.append(np.concatenate([c * pl, np.zeros(3), -a * pl]))
    A = np.asarray(A, float)
    try:
        _, _, Vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None
    G = Vt[-1].reshape(3, 3)
    if abs(np.linalg.det(G)) < 1e-12:
        return None
    H = np.linalg.inv(G).T
    return H / H[2, 2] if abs(H[2, 2]) > 1e-12 else None


def centre_from_lines(bgr):
    """Gate centre in pixels from line evidence alone. Returns (centre_px, H, n_lines) or None."""
    m = gate_mask(bgr)
    inner, outer = extract_gate_lines(m)
    H = homography_from_lines(inner, outer)
    if H is None:
        return None
    w = float(H[2, 2])
    if abs(w) < 1e-12:
        return None
    c = np.array([H[0, 2] / w, H[1, 2] / w])
    return (c, H, len(inner) + len(outer)) if np.isfinite(c).all() else None


# ---------------------------------------------------------------------------------------------
# SEGMENTATION FRONT-END (2026-07-22) -- the replacement for gate_mask's HSV colour rule.
#
# Everything above this line is already measured against real ground truth (13.7 px median centre
# on task2); the module docstring records that the ONLY weak link is the mask. A learned two-class
# seg model (gate_frame = the 2.72 m square, gate_opening = the 1.5 m square -- see
# racer.vision.seg_labels) removes that link, and it also removes the most fragile STEP: with the
# opening predicted as its own instance there is no contour-hierarchy hole-finding, so
# ``extract_gate_lines``' RETR_CCOMP pass and its 0.2*area hole heuristic are not needed at all.
#
# Still NOT wired into flight. The keypoint path remains the only emitter.
# ---------------------------------------------------------------------------------------------

SEG_CLASS_FRAME = 0
SEG_CLASS_OPENING = 1
_MIN_MASK_AREA_PX = 200.0


def segments_from_mask(mask) -> list:
    """Non-border edge segments of a single binary instance mask's OUTER contour.

    Unlike ``extract_gate_lines`` this takes ONE convex-ish blob and never looks for holes -- the
    seg model predicts the opening separately, so the hole logic has nothing to do."""
    m = np.ascontiguousarray(mask.astype(np.uint8))
    h, w = m.shape[:2]
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < _MIN_MASK_AREA_PX:
        return []
    return _segments(c, w, h)


def _centroid(mask):
    ys, xs = np.nonzero(mask)
    return None if not len(xs) else (float(xs.mean()), float(ys.mean()))


def pair_gate_instances(frames_masks, opening_masks):
    """Group predicted instances into gates: each opening belongs to the frame that CONTAINS it.

    Containment (is the opening's centroid inside the frame mask?) rather than IoU, because a frame
    and its own opening barely overlap in area -- the frame is amodal over the opening, so IoU-style
    matching would rank a neighbouring gate's frame just as highly. Returns
    [(frame_mask, opening_mask_or_None)], including frames with no opening (a gate seen so obliquely
    that the opening closed up still gives 4 usable outer edges)."""
    used = set()
    out = []
    for fi, fm in enumerate(frames_masks):
        best, best_d = None, None
        fc = _centroid(fm)
        for oi, om in enumerate(opening_masks):
            if oi in used:
                continue
            oc = _centroid(om)
            if oc is None or fc is None:
                continue
            x, y = int(round(oc[0])), int(round(oc[1]))
            if not (0 <= y < fm.shape[0] and 0 <= x < fm.shape[1] and fm[y, x]):
                continue
            d = (oc[0] - fc[0]) ** 2 + (oc[1] - fc[1]) ** 2
            if best_d is None or d < best_d:
                best, best_d = oi, d
        if best is not None:
            used.add(best)
        out.append((fm, None if best is None else opening_masks[best]))
    return out


def centre_from_seg_masks(frame_mask, opening_mask=None, image_wh=(640, 360)):
    """Gate centre + homography from ONE gate's predicted masks. Returns (centre_px, H) or None."""
    outer = segments_from_mask(frame_mask)
    inner = [] if opening_mask is None else segments_from_mask(opening_mask)
    H = homography_from_lines(inner, outer, image_wh=image_wh)
    if H is None:
        return None
    w = float(H[2, 2])
    if abs(w) < 1e-12:
        return None
    c = np.array([H[0, 2] / w, H[1, 2] / w])
    return (c, H) if np.isfinite(c).all() else None


class SegGateLineDetector:
    """yolo-seg front-end -> the validated line/homography solver. Model is injectable for tests."""

    def __init__(self, model, *, conf: float = 0.25, device: str | None = None):
        self.model = model
        self.conf = conf
        self.device = device

    @classmethod
    def load(cls, weights, **kw):
        from ultralytics import YOLO   # lazy: the heavy, GPU-only [detector] dependency

        task = "segment" if str(weights).lower().endswith((".engine", ".onnx")) else None
        model = YOLO(str(weights), task=task) if task else YOLO(str(weights))
        return cls(model, **kw)

    def masks_for(self, image_bgr):
        """(frame_masks, opening_masks) as full-resolution boolean arrays."""
        res = self.model.predict(image_bgr, verbose=False, conf=self.conf, device=self.device)
        if not res:
            return [], []
        r = res[0]
        masks, boxes = getattr(r, "masks", None), getattr(r, "boxes", None)
        if masks is None or boxes is None or getattr(masks, "data", None) is None:
            return [], []
        h, w = image_bgr.shape[:2]
        data = masks.data
        data = data.cpu().numpy() if hasattr(data, "cpu") else np.asarray(data)
        cls = boxes.cls
        cls = cls.cpu().numpy() if hasattr(cls, "cpu") else np.asarray(cls)
        frames, openings = [], []
        for m, c in zip(data, cls):
            mm = (m > 0.5).astype(np.uint8)
            if mm.shape != (h, w):
                mm = cv2.resize(mm, (w, h), interpolation=cv2.INTER_NEAREST)
            (frames if int(c) == SEG_CLASS_FRAME else openings).append(mm.astype(bool))
        return frames, openings

    def centres(self, image_bgr):
        """[(centre_px, H)] -- one entry per gate the model found and the solver could fit."""
        h, w = image_bgr.shape[:2]
        fm, om = self.masks_for(image_bgr)
        out = []
        for f, o in pair_gate_instances(fm, om):
            got = centre_from_seg_masks(f, o, image_wh=(w, h))
            if got is not None:
                out.append(got)
        return out
