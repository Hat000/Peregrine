"""Camera height above the floor from the Manhattan floor grid — a MAP-FREE z source for VQ2.

THE PROBLEM. VQ2 blocks LOCAL_POSITION_NED/ODOMETRY and the IMU has no barometer, so z (altitude)
is a double-integrated accel state that drifts quadratically. The gate-relative vertical fix pins z
*relative to the active gate's known world height* whenever a gate is in view (see the scope doc
§2.1) — but during no-gate stretches, and if the gate map is wrong, there is no map-free vertical
anchor. The warehouse floor grid is one: the camera's metric height above the floor, read from the
floor plane geometry, bounds z drift independent of any gate.

THE GEOMETRY (and its honest limitation). The floor is the world ground plane (normal = world-down,
gravity-known). The +20deg camera mount points the optical axis UP, so the floor is only visible in a
THIN near-horizon strip in the lower image: with the drone level, the floor below ~9 m forward falls
off the bottom of the frame and the visible floor (~9-100 m) is crammed into ``v in [~300, 360]``
(horizon at v~=296). Any nose-up pitch pushes the floor out of frame entirely. So this channel is
**well-conditioned only when the floor grid is visible (level / nose-down) and ill-conditioned near
the horizon** — it is a *backstop*, gated hard on quality, NOT a primary z source.

METRIC ANCHOR. A monocular floor view is scale-ambiguous: height and the floor-grid CELL SIZE are
perfectly coupled (you cannot get absolute height from grid geometry alone). The metric anchor is the
**known floor-grid cell size in metres** (``grid_cell_m`` — a warehouse property, measurable once from
a calibration frame, exactly like the gate's known 1.5 m inner square). Given the cell size and the
gravity-known roll/pitch, consecutive cross-course grid lines pin the height linearly.

KEY PROPERTY: height is YAW-INVARIANT. The ground RANGE to a floor point depends only on roll/pitch
(gravity-known) and the pixel — not on yaw. So this channel needs no heading and composes cleanly
with the (separately-estimated) vanishing-point yaw.

OUTPUT CONTRACT (for the later ESKF / KF z pseudo-measurement; this module does NOT wire anything):
    FloorHeightEstimate(
        height_m   : float   # camera height above the floor (metres)
        quality    : float   # in [0,1]; gate the z update on this
        n_support  : int     # number of floor grid lines used
        std_m      : float   # spread-derived 1-sigma on the height (large near the horizon)
    )
``estimate_floor_height`` returns ``None`` when the floor is not visible / too few grid lines.

SIGN/FRAME CONVENTION (FD-checked vs ``frames.py``). World NED (z down). ``ground_range_at_pixel``
returns the horizontal distance from the camera nadir to the floor point under a pixel, at a
reference height — yaw-free, mount-aware (+20deg), exact at all roll/pitch.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from racer.frames import CAMERA_INTRINSICS_K, R_camera_from_body
from racer.vision.manhattan_lines import extract_line_segments, segment_angles

_K_INV = np.linalg.inv(CAMERA_INTRINSICS_K)

# Default warehouse floor-grid cell size (metres). PLACEHOLDER — calibrate once from a recon frame
# of known scale (analogue of the gate's 1.5 m). The height estimate scales LINEARLY with this, so
# the consumer should override it with the measured value before trusting absolute metres.
DEFAULT_GRID_CELL_M = 2.0


@dataclass(frozen=True)
class FloorHeightEstimate:
    """Camera height above the floor from the floor grid. See module docstring for the contract and
    the near-horizon ill-conditioning caveat (``quality``/``std_m`` carry it)."""

    height_m: float
    quality: float
    n_support: int
    std_m: float


def ground_range_at_pixel(
    u: float, v: float, roll: float, pitch: float, ref_height_m: float = 1.0
) -> float | None:
    """Horizontal ground distance (metres) from the camera nadir to the floor point seen at pixel
    ``(u, v)``, at reference height ``ref_height_m``. YAW-INVARIANT — depends only on the
    gravity-known ``roll``/``pitch`` and the +20deg mount. Returns ``None`` when the ray does not
    descend to the floor (the pixel is at or above the horizon).

    The true ground range scales LINEARLY with the true height, which is what makes the known-cell
    height recovery a single division (see ``height_from_floor_lines``)."""
    d_cam = _K_INV @ np.array([float(u), float(v), 1.0])
    n = np.linalg.norm(d_cam)
    if n < 1e-12:
        return None
    d_cam /= n
    d_body = R_camera_from_body().T @ d_cam
    # Rotate body->world by roll/pitch only (yaw irrelevant to a horizontal magnitude).
    d_level = Rotation.from_euler("YX", [float(pitch), float(roll)]).as_matrix() @ d_body
    if d_level[2] <= 1e-9:  # world-down component must be positive (ray points at the floor)
        return None
    t = float(ref_height_m) / d_level[2]
    if t <= 0:
        return None
    return float(np.hypot(t * d_level[0], t * d_level[1]))


def height_from_floor_lines(
    line_v_coords: np.ndarray,
    roll: float,
    pitch: float,
    grid_cell_m: float,
    u_eval: float = 320.0,
    ref_height_m: float = 1.0,
) -> tuple[float, int, float] | None:
    """Recover camera height from the image v-coordinates of consecutive cross-course floor-grid
    lines that are a known ``grid_cell_m`` apart on the floor.

    Back-projects each line's v (at column ``u_eval``) to a ground range at ``ref_height_m``; the
    measured inter-line ground spacing scales with the true height, so
    ``height = grid_cell_m / median_spacing * ref_height_m``. Returns ``(height_m, n_lines,
    std_m)`` or ``None`` (fewer than 2 floor lines below the horizon). ``std_m`` propagates the
    spacing spread into a height 1-sigma (large near the horizon where ranges compress)."""
    vs = np.asarray(line_v_coords, dtype=np.float64).ravel()
    ranges = []
    for v in vs:
        gr = ground_range_at_pixel(u_eval, v, roll, pitch, ref_height_m)
        if gr is not None:
            ranges.append(gr)
    ranges = np.sort(np.array(ranges))
    if len(ranges) < 2:
        return None
    # The NEAREST grid lines (smallest ground range) are the best-conditioned: away from the
    # horizon their image rows are well separated, so consecutive detected lines are reliably
    # consecutive GRID lines (spacing == one cell). Near the horizon ranges compress and lines
    # drop out, so a far gap can silently span several cells. We therefore measure the cell from
    # the spacings among the CLOSEST few lines only, and reject gaps that are obvious multi-cell
    # jumps (> ~1.6x the smallest gap) so a missed line cannot inflate the cell estimate.
    n_keep = min(len(ranges), 5)
    near_ranges = ranges[:n_keep]
    gaps = np.diff(near_ranges)
    gaps = gaps[gaps > 1e-6]
    if len(gaps) == 0:
        return None
    base = float(np.min(gaps))
    single_cell_gaps = gaps[gaps <= 1.6 * base]  # drop multi-cell jumps from missed lines
    cell_px_range = float(np.median(single_cell_gaps))
    if cell_px_range <= 0:
        return None
    height = float(grid_cell_m) / cell_px_range * float(ref_height_m)
    if len(single_cell_gaps) >= 2:
        mad = float(np.median(np.abs(single_cell_gaps - cell_px_range))) * 1.4826
        frac = mad / cell_px_range
    else:
        frac = 0.3  # single gap -> wide uncertainty (the consumer gates on it)
    std_m = float(height * max(frac, 0.05))
    return height, int(len(ranges)), std_m


def _detect_floor_grid_line_vs(
    frame_bgr: np.ndarray,
    roll: float,
    pitch: float,
    *,
    min_length_px: float,
    max_slope_deg: float,
    horizon_pad_px: float,
) -> np.ndarray:
    """Find the image v-coordinates of cross-course floor-grid lines: near-horizontal segments in
    the lower image, below the floor horizon. Returns one representative v per detected line."""
    segs = extract_line_segments(frame_bgr, min_length_px=min_length_px)
    if len(segs) == 0:
        return np.zeros(0)
    angs = np.abs(np.rad2deg(segment_angles(segs)))  # 0 = horizontal
    mid_x = (segs[:, 0] + segs[:, 2]) * 0.5
    mid_y = (segs[:, 1] + segs[:, 3]) * 0.5
    h_span = np.abs(segs[:, 2] - segs[:, 0])  # horizontal extent
    # Floor horizon row: the v at which a ray grazes the floor (ground range -> infinity). Pixels
    # ABOVE it never hit the floor; keep lines a little below it.
    horizon_v = _floor_horizon_v(roll, pitch)
    # CROSS-COURSE floor lines run perpendicular to forward: near-horizontal AND wide (they span
    # the frame). ALONG-COURSE lines converge to the heading VP and, where near-horizontal, are
    # short — the wide-span gate rejects them so the spacing ladder stays clean.
    keep = (
        (angs <= float(max_slope_deg))
        & (mid_y >= horizon_v + float(horizon_pad_px))
        & (h_span >= float(min_length_px))
    )
    if not keep.any():
        return np.zeros(0)
    vs = mid_y[keep]
    _ = mid_x  # (kept for readability; centroid column not needed beyond gating)
    # Merge near-duplicate rows (one grid line yields several collinear segments).
    vs = np.sort(vs)
    merged = []
    for v in vs:
        if not merged or (v - merged[-1]) > 3.0:
            merged.append(float(v))
        else:
            merged[-1] = 0.5 * (merged[-1] + float(v))
    return np.array(merged)


def _floor_horizon_v(roll: float, pitch: float, u_eval: float = 320.0) -> float:
    """Image v of the floor's horizon (the grazing row). Below it the floor is visible; above it
    rays escape to the sky. Found by bisection on ``ground_range_at_pixel`` (it returns None above
    the horizon, a finite value below)."""
    lo, hi = 0.0, float(CAMERA_INTRINSICS_K[1, 2] * 2.0)  # full image height
    # Scan downward for the first v that hits the floor.
    for v in np.linspace(0.0, hi, 73):
        if ground_range_at_pixel(u_eval, v, roll, pitch) is not None:
            hi = v
            lo = max(0.0, v - hi / 72.0)
            break
    else:
        return hi  # floor never visible
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        if ground_range_at_pixel(u_eval, mid, roll, pitch) is None:
            lo = mid
        else:
            hi = mid
    return hi


def estimate_floor_height(
    frame_bgr: np.ndarray,
    roll: float = 0.0,
    pitch: float = 0.0,
    *,
    grid_cell_m: float = DEFAULT_GRID_CELL_M,
    min_length_px: float = 20.0,
    max_slope_deg: float = 12.0,
    horizon_pad_px: float = 2.0,
    min_lines: int = 3,
) -> FloorHeightEstimate | None:
    """Estimate camera height above the floor from one BGR frame.

    Detects cross-course floor-grid lines (near-horizontal segments below the floor horizon),
    converts their image rows to ground ranges (yaw-free, using gravity-known roll/pitch + the
    +20deg mount), and divides the known grid cell by the measured inter-line spacing.

    Returns ``None`` when the floor is not visible (nose-up) or too few grid lines are found — the
    common case for this map-free backstop, by design. ``grid_cell_m`` is the metric anchor; the
    absolute height scales linearly with it, so override the default with the measured warehouse
    cell before trusting absolute metres.
    """
    vs = _detect_floor_grid_line_vs(
        frame_bgr, roll, pitch,
        min_length_px=min_length_px, max_slope_deg=max_slope_deg, horizon_pad_px=horizon_pad_px,
    )
    if len(vs) < int(min_lines):
        return None
    res = height_from_floor_lines(vs, roll, pitch, grid_cell_m)
    if res is None:
        return None
    height, n, std_m = res
    if not np.isfinite(height) or height <= 0.0:
        return None
    # Quality: more grid lines + tighter relative spread -> higher. Near-horizon frames yield few
    # lines and a large std -> low quality, which is the intended hard gate for the consumer.
    support_q = float(np.clip(n / 8.0, 0.0, 1.0))
    spread_q = float(np.clip(1.0 - (std_m / max(height, 1e-3)) / 0.5, 0.0, 1.0))
    quality = float(np.clip(support_q * spread_q, 0.0, 1.0))
    return FloorHeightEstimate(height_m=float(height), quality=quality, n_support=int(n), std_m=float(std_m))
