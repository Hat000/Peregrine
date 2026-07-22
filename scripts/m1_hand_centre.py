"""Ground-truth gate-CENTRE for the M+1 hand-label ORACLE, derived from the drawn AREA.

WHY THIS IS NOT the diagonal intersection of the handles. The hand labels were drawn for AREA: the
labeller placed each inner handle to make the visible CLIPPED opening correct, and any OFF-frame
handle was nudged only so the visible edges cross the frame boundary correctly -- its position ALONG
the edge is arbitrary. For the ~67% of hand gates with an off-frame inner corner, the diagonal
intersection of those handles is therefore WRONG exactly where the M+1 centre matters most. The
opening POLYGON is right (correct visible edges); only the handle POSITIONS are not.

So the centre is read from the AREA, in two regimes:

  * ALL FOUR inner corners VISIBLE in frame (v==2): the handles are trustworthy (a correct visible
    quad requires the corners to sit on the real corners), so the centre is their exact diagonal
    intersection -- 0 px, no fit error. ``how = "diag-handles"``.

  * ANY inner corner off-frame or occluded: fit the KNOWN gate model to the exact human-drawn
    opening (+ frame) polygon and read the centre off the fitted pose. The centre is ambiguity-safe
    (every square-relabelling confusion fixes the projected origin; see gate_model_fit) and stays
    defined when it projects OFF the image -- the "through the gate" case, for which there is no
    human truth and the model-fit estimate is simply the best GT available. ``how = "area-fit"``.

The area fit combines the two solvers the M+1 brief named: the LINE solver's plane homography
(accurate on cropped gates, because it uses a SECOND concentric square with a known metric ratio)
SEEDS the model-fit, whose rasterised-IoU objective then refines it AND rejects the line solver's
one real failure mode -- a sign-flipped plane that reprojects to low overlap. The emitted centre is
the one whose reprojected silhouette best matches the exact masks, so a confidently-wrong line plane
can never win. All masks are the EXACT human polygons (labelio.seg_rows_for_gate), not ragged model
predictions, so this is measured well below the real-frame A/B numbers.
"""
from __future__ import annotations

import numpy as np
import cv2

from racer.vision.gate_lines import centre_from_quad, centre_from_seg_masks
from racer.vision.gate_model_fit import (
    FLAT_MODEL, _Target, _pose_from_homography, fit_gate_model, score_pose,
)
from racer.frames import CAMERA_INTRINSICS_K

_V_VIS = 2                 # a corner/centre VISIBLE in frame (contract.V_VIS); off-frame is 0
_FIT_BUDGET = 500          # offline oracle: a generous evaluation budget, wall time is irrelevant
_MIN_FRAME_IOU = 0.20      # a fit that reprojects below this never found the gate -> refuse it


class HandCentreSolver:
    """Per-gate centre for the hand-label oracle. Stateless apart from image size + intrinsics."""

    def __init__(self, img_w: int = 640, img_h: int = 360):
        self.w, self.h = int(img_w), int(img_h)
        self.K = CAMERA_INTRINSICS_K

    # -- exact human-drawn clipped masks (the correct input for an area fit) -------------------
    def _masks(self, geom_gate) -> tuple[np.ndarray | None, np.ndarray | None]:
        """(frame_mask, opening_mask) rasterised from the EXACT drawn polygons, or None each.

        Uses labelio.seg_rows_for_gate -- the same clip + trust rules that write labels/seg/*.txt --
        so the masks are byte-for-byte the human-drawn area, including its past-horizon / self-
        intersection suppression, rather than a re-derivation that could disagree."""
        import labelio
        rows, _rep = labelio.seg_rows_for_gate(
            geom_gate["inner"], geom_gate["outer"], geom_gate.get("outer_detached"),
            self.w, self.h)
        frame_m = opening_m = None
        for r in rows:
            p = r.split()
            cid = int(p[0])
            poly = np.array([float(v) for v in p[1:]], float).reshape(-1, 2)
            poly[:, 0] *= self.w
            poly[:, 1] *= self.h
            m = np.zeros((self.h, self.w), np.uint8)
            if len(poly) >= 3:
                cv2.fillPoly(m, [poly.round().astype(np.int32)], 1)
            if cid == 0:
                frame_m = m
            else:
                opening_m = m
        return frame_m, opening_m

    def _visibility(self, cx: float, cy: float) -> int:
        return _V_VIS if (0.0 <= cx <= self.w - 1 and 0.0 <= cy <= self.h - 1) else 0

    def _iou(self, R, t, fm, om) -> float:
        tgt = _Target(fm, om, (self.w, self.h), 1.0)
        return score_pose(R, t, tgt, self.K, FLAT_MODEL, 1.0, 1.0)

    def _area_centre(self, fm, om):
        """(centre_px, R, t, iou) with the highest reprojection IoU over the seeded candidates, or
        None. The model-fit (line-seeded) is the primary candidate; the RAW line plane is scored
        alongside it so an exact-but-unrefined line answer can still win, and a sign-flipped one
        cannot (its IoU is low)."""
        cands = []
        f = fit_gate_model(fm, om, (self.w, self.h), model=FLAT_MODEL,
                           budget=_FIT_BUDGET, scales=(2.0,), line_init=True, min_score=_MIN_FRAME_IOU)
        if f is not None:
            cands.append((self._iou(f.R_cam_gate, f.t_cam_gate, fm, om), f.centre_px))
        gl = centre_from_seg_masks(fm, om, image_wh=(self.w, self.h))
        if gl is not None:
            p = _pose_from_homography(gl[1], self.K)
            if p is not None:
                cands.append((self._iou(p[0], p[1], fm, om), np.asarray(gl[0], float)))
        if not cands:
            return None
        cands.sort(key=lambda c: -c[0])
        iou, c = cands[0]
        if iou < _MIN_FRAME_IOU:
            return None
        return c

    def centre(self, geom_gate):
        """(cx_px, cy_px, v, how) for one decoded geom gate, or None if no centre is recoverable.

        ``geom_gate`` is a dict from labelio.decode_geometry: inner/outer (unclamped px, LL,LR,UR,UL)
        + vis (8 ints, inner 0-3 then outer 4-7)."""
        inner = np.asarray(geom_gate["inner"], float)
        vis_inner = list(geom_gate["vis"][:4])

        # ALL inner corners VISIBLE -> the handles are exact; diagonal intersection is the truth.
        if all(v == _V_VIS for v in vis_inner):
            c = centre_from_quad(inner)
            if c is not None:
                cx, cy = float(c[0]), float(c[1])
                return cx, cy, self._visibility(cx, cy), "diag-handles"
            # a degenerate (edge-on) all-visible quad is pathological; fall through to the area fit.

        fm, om = self._masks(geom_gate)
        if fm is None and om is None:
            return None
        got = self._area_centre(fm, om)
        if got is None:
            return None
        cx, cy = float(got[0]), float(got[1])
        return cx, cy, self._visibility(cx, cy), "area-fit"
