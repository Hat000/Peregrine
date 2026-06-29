"""Red-glow gate detector — a classical (no-model) detector tuned to the VQ2 appearance.

VQ2 (build 1.0.3379, track "Now You See Me, Now You Don't") renders gates as **glowing-red
square frames** in a very low-light warehouse (scene mean gray ~36/255). This module segments
the gate's **saturated red core**, finds the square ring, and extracts the 4 inner-square
corners (the gate opening) — producing the SAME :class:`GateObservation` contract as the
YOLO-pose :mod:`racer.vision.detector`, so it is a drop-in for the
``frame -> detect() -> [GateObservation] -> gate_pose.estimate_gate_pose -> localization`` seam.

It is ADDITIVE and OPT-IN: the YOLO ``GateDetector`` path is untouched. Select this detector
explicitly (``RedGlowGateDetector()``) where you want the appearance-specific classical path.

Why the SATURATED CORE, not a low red threshold (the recon's binding finding)
-----------------------------------------------------------------------------
The glowing gate has an asymmetric bloom: a crisp (~3 px) inner core edge, a soft (~12 px)
outer halo, plus a pervasive low-amplitude red *ambient wash* across the upper scene (ceiling
+ multiple distant gates). Keying the corners off a LOW red/brightness threshold grabs the
halo + wash and **inflates the apparent square -> biases the PnP range NEAR**. Keying off the
saturated red core (``R >= R_CORE`` AND red-dominant ``R - G`` / ``R - B >= DOM``) tracks the
crisp inner edge, so the apparent square is only mildly inflated. The validation script
quantifies the core-vs-naive span inflation on the recon frames.

Corner contract (MUST match :mod:`racer.vision.gate_pose`)
----------------------------------------------------------
We emit the 4 INNER-square corners (the gate opening) in canonical IPPE_SQUARE order, with the
gate Y axis pointing DOWN in the image::

    idx   image position (head-on gate)
     0    lower-left
     1    lower-right
     2    upper-right
     3    upper-left

A gate appears as a red RING with a dark inner hole. We take the hole's quadrilateral as the
opening. When the ring has no resolvable hole (extreme-near blow-out, or far/small where the
core is a thin sliver) we do NOT fabricate a pose — the detection is dropped (the final
approach dead-reckons on IMU; see RECON.md §2). Multiple gates -> all returned, ``primary``
is the largest / most-centred (final target selection is a later step via RACE_STATUS).

Clutter rejected by construction
--------------------------------
Blue direction-chevrons inside the gate, blue floor lane-lines, green pole markers, white
ceiling-truss/floor-grid: all fail the **red-dominance** core test. Orange floor light-beams
are bright but not red-dominant enough at the core threshold (orange has high G); they also
fail the **square-ring shape** test (a beam is an elongated streak, not a ~square frame with a
central hole). The distant red gates that DO pass color are kept as low-priority candidates.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # OpenCV is already a hard dep of gate_pose; guard only to keep import errors legible.
    import cv2
except Exception as exc:  # pragma: no cover - cv2 is installed in the test venv
    cv2 = None  # type: ignore
    _CV2_IMPORT_ERROR = exc

from racer.contracts import Frame, GateObservation

N_CORNERS = 4


@dataclass(frozen=True)
class RedGlowParams:
    """Tunables for the red-glow segmentation + shape gates. Defaults are calibrated on the
    2026-06-29 VQ2 recon frames (640x360). All thresholds are on 0..255 BGR channels."""

    # --- Saturated red CORE segmentation (the bloom-robust threshold) ---
    r_core: int = 250            # R must be at/near saturation for the crisp inner edge
    dom_gr: int = 80             # red-dominance: R - G >= this (rejects orange/white/blue)
    dom_rb: int = 80             # red-dominance: R - B >= this
    # A slightly relaxed "core-or-just-below" band used ONLY to fill 1-px bloom notches in the
    # ring via morphology (NOT to grow the square): R high-ish AND still strongly red-dominant.
    r_fill: int = 235

    # --- Morphology ---
    close_ksize: int = 5         # close gaps in the saturated ring (bloom flicker)

    # --- Candidate shape gates (on the OUTER red-ring contour) ---
    min_ring_area: float = 250.0     # px^2; below this the gate is too far/small to localize
    min_fill_frac: float = 0.04      # ring area / bbox area lower bound (reject tiny specks)
    max_aspect: float = 3.0          # bbox long/short side; a beam streak is very elongated
    min_hole_area_frac: float = 0.04 # inner hole area / outer bbox area: a real frame has a hole

    # --- Outer-ring inset fallback (when the inner hole is fragmented by chevron/bloom but the
    #     OUTER red ring is a clean, non-clipped square) ---
    use_outer_inset: bool = True
    bar_thickness_frac: float = 0.175   # gate-bar thickness as a fraction of the outer side
                                        # (recon frame 01: outer ~80 px -> inner ~52 px)
    border_margin_px: int = 3           # a ring touching the image border this close is clipped
                                        # -> outer quad is unreliable, skip the inset fallback

    # --- Confidence model (maps geometry quality -> corner_confidence, score) ---
    conf_ref_area: float = 4000.0    # ring area at which size-confidence saturates to ~1


# Canonical corner order the gate_pose IPPE_SQUARE PnP expects, in IMAGE space for a head-on
# gate: 0=lower-left, 1=lower-right, 2=upper-right, 3=upper-left (gate Y is DOWN).
def _order_corners_canonical(pts: np.ndarray) -> np.ndarray:
    """Order 4 quad corners as [LL, LR, UR, UL] in image pixels (x right, y DOWN).

    Robust to rotation up to ~45 deg (oblique gates): split by the centroid into top/bottom
    halves (smaller y = top), then left/right within each half. This matches the
    gate_object_points order in gate_pose (idx0 lower-left ... idx3 upper-left)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    c = pts.mean(axis=0)
    top = pts[pts[:, 1] < c[1]]
    bot = pts[pts[:, 1] >= c[1]]
    # Guard degenerate splits (collinear-ish): fall back to angle sort.
    if len(top) != 2 or len(bot) != 2:
        ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        order = np.argsort(ang)
        pts = pts[order]
        # angle sort gives CCW from +x; rotate so we start near lower-left
        return pts
    top = top[np.argsort(top[:, 0])]   # [top-left, top-right]
    bot = bot[np.argsort(bot[:, 0])]   # [bot-left, bot-right]
    ll, lr = bot[0], bot[1]
    ul, ur = top[0], top[1]
    return np.array([ll, lr, ur, ul], dtype=np.float64)


def _quad_from_contour(contour: np.ndarray) -> np.ndarray | None:
    """Best 4-corner quadrilateral for a contour: polygon-approx to 4, else min-area rect."""
    peri = cv2.arcLength(contour, True)
    for eps_frac in (0.02, 0.04, 0.06, 0.08, 0.10):
        approx = cv2.approxPolyDP(contour, eps_frac * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float64)
    # Fallback: rotated min-area rectangle (handles bloom-rounded corners).
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return np.asarray(box, dtype=np.float64)


def _red_core_mask(image_bgr: np.ndarray, p: RedGlowParams) -> np.ndarray:
    b = image_bgr[:, :, 0].astype(np.int16)
    g = image_bgr[:, :, 1].astype(np.int16)
    r = image_bgr[:, :, 2].astype(np.int16)
    core = (r >= p.r_core) & ((r - g) >= p.dom_gr) & ((r - b) >= p.dom_rb)
    fill = (r >= p.r_fill) & ((r - g) >= p.dom_gr) & ((r - b) >= p.dom_rb)
    mask = (core | fill).astype(np.uint8) * 255
    if p.close_ksize > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.close_ksize, p.close_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


@dataclass(frozen=True)
class RedGlowCandidate:
    """One gate-frame candidate from the red-glow detector (before downstream selection)."""

    corners_px: np.ndarray       # (4,2) inner-opening corners, canonical order
    ring_area: float             # px^2 of the saturated red ring (size / range proxy)
    hole_area: float             # px^2 of the inner opening
    centroid: np.ndarray         # (2,) opening centroid in pixels
    score: float                 # detection confidence in [0,1]


def detect_red_glow_candidates(
    image_bgr: np.ndarray, params: RedGlowParams | None = None
) -> list[RedGlowCandidate]:
    """Core, model-free detection: BGR image -> list of gate candidates (corner quads).

    Returns candidates sorted by ``score`` descending (so ``[0]`` is the primary pick). Empty
    list when no gate frame clears the color + shape gates (correct on no-gate / blow-out /
    far-no-core frames). This is the pure function the runtime wrapper and the validation
    harness both call.
    """
    if cv2 is None:  # pragma: no cover
        raise RuntimeError(f"red_glow_detector requires OpenCV (cv2): {_CV2_IMPORT_ERROR}")
    p = params or RedGlowParams()
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must be (H,W,3) BGR uint8")

    mask = _red_core_mask(image_bgr, p)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return []
    hierarchy = hierarchy[0]

    cands: list[RedGlowCandidate] = []
    for i, cnt in enumerate(contours):
        parent = hierarchy[i][3]
        if parent != -1:
            continue  # only EXTERNAL contours seed a candidate; holes handled via children
        ring_area = float(cv2.contourArea(cnt))
        if ring_area < p.min_ring_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        bbox_area = float(w * h)
        if bbox_area <= 0:
            continue
        long_side, short_side = max(w, h), max(1, min(w, h))
        if long_side / short_side > p.max_aspect:
            continue  # elongated streak (orange floor beam, lane line) — not a square frame
        fill_frac = ring_area / bbox_area
        if fill_frac < p.min_fill_frac:
            continue  # tiny speck — not a gate frame

        # Find the LARGEST inner hole (child contour) = the gate opening.
        best_hole = None
        best_hole_area = 0.0
        child = hierarchy[i][2]
        while child != -1:
            ha = float(cv2.contourArea(contours[child]))
            if ha > best_hole_area:
                best_hole_area, best_hole = ha, contours[child]
            child = hierarchy[child][0]

        H, W = image_bgr.shape[:2]
        clipped = (
            x <= p.border_margin_px or y <= p.border_margin_px
            or x + w >= W - p.border_margin_px or y + h >= H - p.border_margin_px
        )
        inset_used = False
        if best_hole is not None and best_hole_area >= p.min_hole_area_frac * bbox_area:
            # PRIMARY path: the inner hole resolves the gate opening directly (crisp inner edge).
            quad = _quad_from_contour(best_hole)
            opening_area = best_hole_area
        elif p.use_outer_inset and not clipped and best_hole is not None:
            # FALLBACK: an inner hole EXISTS but is too fragmented (chevron panel / bloom notches)
            # to trust its quad directly — yet the OUTER red ring is a clean non-clipped square. The
            # presence of *any* child hole proves this is a hollow frame, not a solid blow-out blob,
            # so we may infer the opening by insetting the outer quad by the bar thickness. Lower
            # confidence (the inner edge is inferred, not measured).
            outer = _quad_from_contour(cnt)
            if outer is None or len(outer) != 4:
                continue
            outer_o = _order_corners_canonical(outer)
            ctr = outer_o.mean(axis=0)
            side = float(np.linalg.norm(outer_o[1] - outer_o[0]))  # LL->LR
            t = p.bar_thickness_frac
            quad = ctr + (outer_o - ctr) * (1.0 - 2.0 * t)  # shrink toward centre by bar thickness
            opening_area = ring_area * (1.0 - 2.0 * t) ** 2
            inset_used = True
        else:
            continue  # no resolvable opening (blow-out / clipped / thin far sliver) — drop

        if quad is None or len(quad) != 4:
            continue
        corners = _order_corners_canonical(quad)
        centroid = corners.mean(axis=0)
        # Confidence: geometric quality = size (range) x squareness of the opening.
        size_conf = float(np.clip(ring_area / p.conf_ref_area, 0.0, 1.0))
        # squareness: opening bbox aspect close to 1 (perspective lowers it gracefully)
        ox = corners[:, 0]; oy = corners[:, 1]
        ow, oh = ox.max() - ox.min(), oy.max() - oy.min()
        sq = float(min(ow, oh) / max(1.0, max(ow, oh)))
        score = float(np.clip(0.5 * size_conf + 0.5 * sq, 0.0, 1.0))
        if inset_used:
            score *= 0.7  # inferred inner edge -> trust less than a measured hole
        best_hole_area = opening_area
        cands.append(
            RedGlowCandidate(
                corners_px=corners,
                ring_area=ring_area,
                hole_area=best_hole_area,
                centroid=centroid,
                score=score,
            )
        )

    cands.sort(key=lambda c: c.score, reverse=True)
    return cands


class RedGlowGateDetector:
    """Runtime classical red-glow gate detector. Interface-compatible with the YOLO
    :class:`racer.vision.detector.GateDetector` (a ``detect(frame) -> [GateObservation]``),
    so it is a drop-in for the navigator's SENSE step. No model / no GPU / no ultralytics.

    OPT-IN: construct this explicitly where the VQ2 appearance-specific path is wanted; the
    YOLO path stays the default everywhere else. Emits the 4 inner-square corners in canonical
    order, ready for :func:`racer.vision.gate_pose.estimate_gate_pose`.
    """

    def __init__(self, params: RedGlowParams | None = None):
        self.params = params or RedGlowParams()

    def detect(self, frame: Frame) -> list[GateObservation]:
        cands = detect_red_glow_candidates(frame.image_bgr, self.params)
        out: list[GateObservation] = []
        for c in cands:
            conf = np.full(N_CORNERS, c.score, dtype=np.float64)
            xs, ys = c.corners_px[:, 0], c.corners_px[:, 1]
            bbox = np.array(
                [xs.min(), ys.min(), xs.max() - xs.min(), ys.max() - ys.min()],
                dtype=np.float64,
            )
            out.append(
                GateObservation(
                    frame_id=frame.frame_id,
                    sim_time_ns=frame.sim_time_ns,
                    corners_px=c.corners_px.copy(),
                    corner_ids=None,            # full 4 corners -> canonical [0,1,2,3]
                    corner_confidence=conf,
                    score=c.score,
                    bbox_xywh=bbox,
                    gate_id=None,               # association is the mapper's job
                )
            )
        return out
