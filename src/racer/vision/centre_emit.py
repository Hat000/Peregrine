"""Centre + apparent-size gate emit: ``rel_pos = range x bearing``, with NO PnP translation.

WHY THIS EXISTS (Fengyou, 2026-07-23). The 3-D emit the RL obs needs has two terms and BOTH are
free of the IPPE 2-fold ambiguity, so the ambiguous PnP translation is never required:

  * BEARING is ambiguity-free. IPPE's two solutions are ambiguous *precisely because* they project
    the same points to the same pixels -- so the gate centre lands on the SAME pixel in both.
    Back-project that pixel through the intrinsics and the ray is clean.
  * RANGE is ambiguity-free. It comes from the gate's known metric size vs its apparent size, and
    the 2-fold ambiguity flips the SIGN of the tilt, not its MAGNITUDE -- both solutions foreshorten
    identically, so range survives.

Consuming ``t_cam_gate`` instead is what produced the known ~1.2 m systematic error on the task2
bundle; this path deletes that error at the root rather than correcting it.

RANGE FROM THE CORNERS -- the min-over-pairs rule
------------------------------------------------
The 4 inner keypoints carry the scale. For any two corners whose canonical ids are known, the metric
separation is known exactly (side 1.5 m, diagonal 1.5*sqrt(2) m), so the pinhole model gives a depth
estimate per pair::

    Z_ij = f * L_metric(i,j) / d_px(i,j)

Foreshortening can only ever SHRINK an apparent length, and a shrunken ``d_px`` INFLATES Z. So the
**minimum over all usable pairs is the least-foreshortened estimate**, i.e. the closest to truth --
no tilt has to be measured, and no un-foreshortening step is needed. This is what makes the rule work
with ANY 2+ corners instead of requiring the full quad: 2 corners give one estimate, 4 give six and a
better minimum. It degrades gracefully in exactly the regime M+1 was built for.

Bias note, deliberately accepted: because it takes a minimum over noisy estimates it is slightly
biased NEAR (pixel noise that lengthens a pair is discarded, noise that shortens it is kept). Aiming
slightly near on a gate you are flying at is the safe direction, and the bias shrinks as the gate
grows in frame.

CAVEAT recorded rather than hidden: with fewer than 2 identified corners the quad carries no scale at
all, and a centre pixel plus a detection box conflates range with tilt (a small apparent box is
far-and-square-on OR near-and-tilted). Those emits are marked ``range_src="bbox"`` and are a coarse
fallback; the caller should prefer a coasted track range when it has one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from racer.frames import CAMERA_INTRINSICS_K

# Inner opening is a 1.5 m square: canonical corner ids 0=LL, 1=LR, 2=UR, 3=UL.
GATE_INNER_SIZE_M = 1.5
_SIDE = GATE_INNER_SIZE_M
_DIAG = GATE_INNER_SIZE_M * np.sqrt(2.0)

# Metric separation between each pair of canonical inner corners.
_PAIR_METRIC = {
    (0, 1): _SIDE, (1, 2): _SIDE, (2, 3): _SIDE, (0, 3): _SIDE,   # sides
    (0, 2): _DIAG, (1, 3): _DIAG,                                  # diagonals
}

# A pair separated by fewer than this many pixels carries no usable scale. Kept LOW deliberately:
# measured on 300 real frames, a 12 px floor pushed 120 gates that HAD all 4 corners onto the bbox
# fallback (a 1.5 m side spans only ~5 px at 87 m), and the fallback is 35% wrong vs the corner path's
# 2.4%. Corners are better than the box at every span, so only reject where the estimate is pure
# noise: at 4 px a 1 px keypoint error is already ~25% of range. Gates that far out are dropped by
# --ego-max-valid-range (30 m) long before the policy sees them.
MIN_PAIR_SPAN_PX = 4.0


@dataclass(frozen=True)
class CentreEmit:
    """A gate emit built from the regressed centre + apparent size. Camera optical frame."""

    p_cam: np.ndarray          # (3,) metres, camera optical frame (X-right, Y-down, Z-fwd)
    range_m: float             # distance along the bearing ray to the gate centre
    depth_m: float             # Z component (along the optical axis)
    bearing: np.ndarray        # (2,) (az, el) radians, from the centre pixel
    range_src: str             # "corners" (min-over-pairs) or "bbox" (coarse fallback)
    n_pairs: int               # how many corner pairs voted; 0 when range_src == "bbox"


def bearing_from_px(centre_px, K: np.ndarray = CAMERA_INTRINSICS_K) -> tuple[np.ndarray, np.ndarray]:
    """(unit_ray, (az, el)) for a pixel. The ray is the ambiguity-free half of the emit."""
    u, v = float(centre_px[0]), float(centre_px[1])
    x_n = (u - K[0, 2]) / K[0, 0]
    y_n = (v - K[1, 2]) / K[1, 1]
    d = np.array([x_n, y_n, 1.0], dtype=np.float64)
    ray = d / float(np.linalg.norm(d))
    # az about the optical vertical (+right), el above the optical axis (+up => -y)
    return ray, np.array([np.arctan2(x_n, 1.0), np.arctan2(-y_n, 1.0)], dtype=np.float64)


def depth_from_corner_pairs(corners_px, corner_ids=None, corner_conf=None,
                            conf_thresh: float = 0.0,
                            K: np.ndarray = CAMERA_INTRINSICS_K) -> tuple[float | None, int]:
    """Depth (m) by the min-over-pairs rule, and the number of pairs that voted.

    Returns (None, 0) when fewer than 2 corners are usable -- the quad then carries no scale.
    """
    if corners_px is None:
        return None, 0
    pts = np.asarray(corners_px, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return None, 0
    ids = (np.arange(pts.shape[0]) if corner_ids is None
           else np.asarray(corner_ids, dtype=int))
    keep = np.ones(pts.shape[0], dtype=bool) if corner_conf is None else \
        (np.asarray(corner_conf, dtype=np.float64) >= conf_thresh)
    f = 0.5 * (float(K[0, 0]) + float(K[1, 1]))

    ests: list[float] = []
    for a in range(pts.shape[0]):
        for b in range(a + 1, pts.shape[0]):
            if not (keep[a] and keep[b]):
                continue
            L = _PAIR_METRIC.get((int(ids[a]), int(ids[b]))) or \
                _PAIR_METRIC.get((int(ids[b]), int(ids[a])))
            if L is None:                       # duplicate/unknown ids -> no known separation
                continue
            d_px = float(np.linalg.norm(pts[a] - pts[b]))
            if d_px < MIN_PAIR_SPAN_PX:
                continue
            ests.append(f * L / d_px)
    if not ests:
        return None, 0
    # MINIMUM: foreshortening only shrinks d_px, which only inflates Z.
    return float(min(ests)), len(ests)


def depth_from_bbox(bbox_xywh, K: np.ndarray = CAMERA_INTRINSICS_K) -> float | None:
    """Coarse depth from the detection box, for the corner-free case. The box bounds the OUTER frame
    (2.72 m), and a cropped box under-reports size, so this reads FAR -- a floor, not a measurement."""
    if bbox_xywh is None:
        return None
    w, h = float(bbox_xywh[2]), float(bbox_xywh[3])
    s = max(w, h)                       # the less-foreshortened box dimension
    if s < MIN_PAIR_SPAN_PX:
        return None
    f = 0.5 * (float(K[0, 0]) + float(K[1, 1]))
    return f * 2.72 / s


def emit_from_observation(obs, *, conf_thresh: float = 0.0,
                          K: np.ndarray = CAMERA_INTRINSICS_K) -> CentreEmit | None:
    """Build the centre+size emit for one :class:`GateObservation`, or None if it carries no centre.

    ``rel_pos`` is ``depth * [x_n, y_n, 1]`` -- the standard back-projection, which is exactly
    ``range * unit_ray`` written in its numerically cleanest form (no normalise-then-rescale).
    """
    centre = getattr(obs, "centre_px", None)
    if centre is None:
        return None
    ray, bearing = bearing_from_px(centre, K)

    z, n_pairs = depth_from_corner_pairs(
        getattr(obs, "corners_px", None), getattr(obs, "corner_ids", None),
        getattr(obs, "corner_confidence", None), conf_thresh=conf_thresh, K=K)
    src = "corners"
    if z is None:
        z, src = depth_from_bbox(getattr(obs, "bbox_xywh", None), K), "bbox"
    if z is None or not np.isfinite(z) or z <= 0.0:
        return None

    u, v = float(centre[0]), float(centre[1])
    d = np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0], dtype=np.float64)
    p_cam = z * d
    return CentreEmit(p_cam=p_cam, range_m=float(np.linalg.norm(p_cam)), depth_m=float(z),
                      bearing=bearing, range_src=src, n_pairs=n_pairs)
