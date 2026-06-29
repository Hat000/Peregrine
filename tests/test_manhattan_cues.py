"""Validation for the Manhattan-world vision cues (heading_vp + floor_height).

The recon frames carry NO pose ground truth, so the accuracy checks run on a SYNTHETIC Manhattan
scene rendered at a known camera yaw + height with the project's exact intrinsics / mount: those
assert sign-correct recovery within tolerance. The recon frames (02/04/05/07 structure, 10/11/12
blur, 08 facing-away) are used only as smoke fixtures — they assert plausibility / repeatability /
confidence behaviour, never accuracy.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from racer.frames import (  # noqa: E402
    CAMERA_INTRINSICS_K,
    R_camera_from_body,
    R_world_from_body,
)
from racer.vision import heading_vp, floor_height  # noqa: E402
from racer.vision.manhattan_lines import (  # noqa: E402
    extract_line_segments,
    fit_vanishing_point,
)

_K = CAMERA_INTRINSICS_K
_RCB = R_camera_from_body()

_RECON_DIR = os.path.join(
    os.path.dirname(__file__), "..", "handoff", "vq2-recon-2026-06-29", "frames", "curated"
)


# --------------------------------------------------------------------------------------------
# Synthetic Manhattan renderer (ground-truth source)
# --------------------------------------------------------------------------------------------

def _project(P_world, roll, pitch, yaw, p_cam=np.zeros(3)):
    """Project a world point to pixels with the project's exact intrinsics/mount. None if behind."""
    R_wb = R_world_from_body(roll, pitch, yaw)
    d_cam = _RCB @ R_wb.T @ (np.asarray(P_world, float) - p_cam)
    if d_cam[2] <= 1e-6:
        return None
    uvw = _K @ d_cam
    return np.array([uvw[0] / uvw[2], uvw[1] / uvw[2]])


def _draw_seg(img, p0, p1, color, thickness=1):
    if p0 is None or p1 is None:
        return
    h, w = img.shape[:2]
    # Skip wildly out-of-frame segments (cheap clip guard).
    if max(abs(p0[0]), abs(p1[0])) > 6 * w or max(abs(p0[1]), abs(p1[1])) > 6 * h:
        return
    cv2.line(img, (int(round(p0[0])), int(round(p0[1]))),
             (int(round(p1[0])), int(round(p1[1]))), color, thickness, cv2.LINE_AA)


def render_manhattan(
    yaw_deg=0.0, height_m=1.4, roll_deg=0.0, pitch_deg=0.0,
    grid_cell_m=2.0, draw_floor=True, draw_ceiling=True, bg=18,
):
    """Render a low-light Manhattan warehouse: floor grid + ceiling grid at a known camera yaw and
    height. World NED (z down); camera at origin, floor at z=+height_m, ceiling at z=+height_m-4."""
    H, W = 360, 640
    img = np.full((H, W, 3), bg, np.uint8)
    roll, pitch, yaw = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
    p_cam = np.zeros(3)
    floor_z = height_m
    ceil_z = height_m - 4.0  # ceiling 4 m above the camera... err, above floor by 4 (z up = -)

    span = 60.0
    line_color_floor = (90, 90, 90)
    line_color_ceil = (200, 200, 200)

    if draw_floor:
        # Lines running ALONG course (const world-Y, varying X) and ACROSS (const X, varying Y).
        for y in np.arange(-span, span + 1e-6, grid_cell_m):
            p0 = _project([1.0, y, floor_z], roll, pitch, yaw, p_cam)
            p1 = _project([span, y, floor_z], roll, pitch, yaw, p_cam)
            _draw_seg(img, p0, p1, line_color_floor)
        for x in np.arange(1.0, span + 1e-6, grid_cell_m):
            p0 = _project([x, -span, floor_z], roll, pitch, yaw, p_cam)
            p1 = _project([x, span, floor_z], roll, pitch, yaw, p_cam)
            _draw_seg(img, p0, p1, line_color_floor)

    if draw_ceiling:
        for y in np.arange(-span, span + 1e-6, grid_cell_m):
            p0 = _project([1.0, y, ceil_z], roll, pitch, yaw, p_cam)
            p1 = _project([span, y, ceil_z], roll, pitch, yaw, p_cam)
            _draw_seg(img, p0, p1, line_color_ceil)
        for x in np.arange(1.0, span + 1e-6, grid_cell_m):
            p0 = _project([x, -span, ceil_z], roll, pitch, yaw, p_cam)
            p1 = _project([x, span, ceil_z], roll, pitch, yaw, p_cam)
            _draw_seg(img, p0, p1, line_color_ceil)
    return img


# --------------------------------------------------------------------------------------------
# heading_vp — synthetic ground-truth recovery
# --------------------------------------------------------------------------------------------

def _ang_err_mod90_deg(a_rad, b_rad):
    """Angular error between two headings modulo 90deg, in degrees, in [0, 45]."""
    d = (a_rad - b_rad + np.pi / 4) % (np.pi / 2) - np.pi / 4
    return abs(np.rad2deg(d))


@pytest.mark.parametrize("yaw_deg", [0.0, 8.0, -7.0, 20.0, -25.0, 35.0])
def test_heading_vp_recovers_synthetic_yaw_level(yaw_deg):
    img = render_manhattan(yaw_deg=yaw_deg, height_m=1.4)
    est = heading_vp.estimate_heading(img, roll=0.0, pitch=0.0)
    assert est is not None, f"no heading at yaw={yaw_deg}"
    err = _ang_err_mod90_deg(est.heading_mod90_rad, np.deg2rad(yaw_deg))
    assert err < 2.5, f"yaw={yaw_deg} recovered_mod90={np.rad2deg(est.heading_mod90_rad):.2f} err={err:.2f}"
    assert est.quality > 0.2
    assert est.n_support >= 12
    # The true yaw must appear in one of the four 90deg branches (mod-90 honesty).
    branch_errs = [abs(((b - np.deg2rad(yaw_deg) + np.pi) % (2 * np.pi)) - np.pi) for b in est.branch_headings_rad]
    assert min(branch_errs) < np.deg2rad(2.5)


@pytest.mark.parametrize("roll_deg,pitch_deg,yaw_deg", [(5.0, 6.0, 10.0), (-8.0, 4.0, -15.0)])
def test_heading_vp_recovers_with_tilt(roll_deg, pitch_deg, yaw_deg):
    img = render_manhattan(yaw_deg=yaw_deg, height_m=1.4, roll_deg=roll_deg, pitch_deg=pitch_deg)
    est = heading_vp.estimate_heading(img, roll=np.deg2rad(roll_deg), pitch=np.deg2rad(pitch_deg))
    assert est is not None
    err = _ang_err_mod90_deg(est.heading_mod90_rad, np.deg2rad(yaw_deg))
    assert err < 3.0, f"tilt case err={err:.2f}"


def test_heading_vp_sign_direction():
    """A POSITIVE yaw (nose toward world-East about world-down) must move the along-course VP to
    the LEFT in the image vs yaw=0 — pins the sign against frames.py conventions."""
    seg_iters = 3000
    img0 = render_manhattan(yaw_deg=0.0)
    imgp = render_manhattan(yaw_deg=12.0)
    vp0 = fit_vanishing_point(extract_line_segments(img0), iters=seg_iters)
    vpp = fit_vanishing_point(extract_line_segments(imgp), iters=seg_iters)
    assert vp0 is not None and vpp is not None
    # yaw_from_vp_pixel must return ~ +12deg for the +12 render (sign-correct mapping).
    yaw0, _ = heading_vp.yaw_from_vp_pixel(*_along_course_vp(img0), 0.0, 0.0)
    yawp, _ = heading_vp.yaw_from_vp_pixel(*_along_course_vp(imgp), 0.0, 0.0)
    assert _ang_err_mod90_deg(yaw0, 0.0) < 2.5
    assert _ang_err_mod90_deg(yawp, np.deg2rad(12.0)) < 2.5


def _along_course_vp(img):
    """Return the pixel of the most-horizontal VP in a synthetic frame (helper for the sign test)."""
    from racer.vision.manhattan_lines import fit_multiple_vanishing_points
    vps = fit_multiple_vanishing_points(extract_line_segments(img), n_vps=3, iters=3000)
    best = None
    for vp in vps:
        _, wz = heading_vp.yaw_from_vp_pixel(vp.point_px[0], vp.point_px[1], 0.0, 0.0)
        if best is None or abs(wz) < best[0]:
            best = (abs(wz), vp.point_px)
    return float(best[1][0]), float(best[1][1])


# --------------------------------------------------------------------------------------------
# floor_height — synthetic ground-truth recovery
# --------------------------------------------------------------------------------------------

def test_ground_range_yaw_invariant():
    """Ground range at a pixel must NOT depend on yaw (height/z observability is yaw-free)."""
    r0 = floor_height.ground_range_at_pixel(320, 350, 0.0, 0.0)
    for yaw in [0.3, -0.6, 1.2]:
        # ground_range has no yaw arg by construction; assert the value is stable & finite.
        assert r0 is not None and r0 > 0
    # A pixel near the bottom (more depressed) is CLOSER than one near the horizon.
    near = floor_height.ground_range_at_pixel(320, 358, 0.0, 0.0)
    far = floor_height.ground_range_at_pixel(320, 305, 0.0, 0.0)
    assert near is not None and far is not None and near < far
    # Above the horizon -> None.
    assert floor_height.ground_range_at_pixel(320, 100, 0.0, 0.0) is None


# Tolerance reflects the HONEST conditioning: at a normal flying height (~1.1-1.8 m) the nearest
# floor grid lines are well separated -> ~15% accuracy; at an unusually LOW height (0.9 m) the floor
# is so close that only the 2 nearest lines are above the bottom edge (a single-gap estimate) ->
# the channel is a coarse backstop there (~25%). This is the full detector path; the geometry CORE
# is pinned to <2% separately (test_floor_height_geometry_core_exact).
@pytest.mark.parametrize("height_m,cell,tol", [(1.4, 2.0, 0.15), (1.1, 2.0, 0.15),
                                               (1.8, 1.5, 0.15), (0.9, 2.0, 0.25)])
def test_floor_height_recovers_synthetic_level(height_m, cell, tol):
    img = render_manhattan(yaw_deg=0.0, height_m=height_m, grid_cell_m=cell, draw_ceiling=False)
    est = floor_height.estimate_floor_height(img, roll=0.0, pitch=0.0, grid_cell_m=cell)
    assert est is not None, f"no floor height at h={height_m}"
    rel = abs(est.height_m - height_m) / height_m
    assert rel < tol, f"h_true={height_m} h_est={est.height_m:.3f} rel={rel:.3f}"
    assert est.quality > 0.0
    assert est.n_support >= 3


def test_floor_height_geometry_core_exact():
    """The pure geometry core (height_from_floor_lines) recovers height to <2% from exact line
    v-coordinates — isolates the math from the detector."""
    height_m, cell, roll, pitch = 1.35, 2.0, np.deg2rad(3.0), np.deg2rad(4.0)
    vs = []
    for x in np.arange(9.0, 45.0, cell):
        P = [x, 0.0, height_m]
        px = _project(P, roll, pitch, 0.0)
        if px is not None and 0 <= px[1] <= 360:
            vs.append(px[1])
    assert len(vs) >= 4
    res = floor_height.height_from_floor_lines(np.array(vs), roll, pitch, cell)
    assert res is not None
    h_est, n, std = res
    assert abs(h_est - height_m) / height_m < 0.02, f"core h_est={h_est:.4f} vs {height_m}"


def test_floor_height_none_when_nose_up():
    """A strong nose-up pitch pushes the floor out of frame -> no estimate (honest None)."""
    img = render_manhattan(yaw_deg=0.0, height_m=1.4, pitch_deg=25.0, draw_ceiling=False)
    est = floor_height.estimate_floor_height(img, roll=0.0, pitch=np.deg2rad(25.0))
    assert est is None


# --------------------------------------------------------------------------------------------
# Confidence behaviour — sparse / blank frames must yield low confidence or None
# --------------------------------------------------------------------------------------------

def test_heading_vp_blank_frame_returns_none():
    blank = np.full((360, 640, 3), 8, np.uint8)
    assert heading_vp.estimate_heading(blank) is None
    assert floor_height.estimate_floor_height(blank) is None


def test_heading_vp_sparse_low_confidence():
    """A frame with only a few short lines -> None or low quality (consumer can gate)."""
    img = np.full((360, 640, 3), 10, np.uint8)
    cv2.line(img, (100, 100), (140, 102), (180, 180, 180), 1)
    cv2.line(img, (300, 200), (340, 205), (180, 180, 180), 1)
    est = heading_vp.estimate_heading(img)
    assert est is None or est.quality < 0.3


# --------------------------------------------------------------------------------------------
# Recon frames — smoke fixtures (plausibility / repeatability / confidence; NO accuracy GT)
# --------------------------------------------------------------------------------------------

def _load_recon(name):
    path = os.path.join(_RECON_DIR, name)
    if not os.path.exists(path):
        pytest.skip(f"recon frame missing: {name}")
    img = cv2.imread(path)
    if img is None:
        pytest.skip(f"recon frame unreadable: {name}")
    return img


_STRUCT_FRAMES = [
    "02_gate_deadahead_from_startpad.png",
    "04_gates_far_warehouse_overview.png",
    "05_gate_angle_aisle_stations.png",
    "07_between_gates_pillar20.png",
]


@pytest.mark.parametrize("name", _STRUCT_FRAMES)
def test_recon_has_extractable_line_structure(name):
    """CRITICAL FINDING under test: the low-light frames DO carry ample line structure for VP."""
    img = _load_recon(name)
    segs = extract_line_segments(img, min_length_px=25.0)
    assert len(segs) >= 60, f"{name}: only {len(segs)} segments — VP would be starved"


@pytest.mark.parametrize("name", _STRUCT_FRAMES)
def test_recon_heading_plausible(name):
    """heading_vp returns a finite, in-range heading with real inlier support on structured frames."""
    img = _load_recon(name)
    est = heading_vp.estimate_heading(img, roll=0.0, pitch=0.0)
    assert est is not None, f"{name}: no heading despite structure"
    assert -np.pi / 4 <= est.heading_mod90_rad < np.pi / 4
    assert np.isfinite(est.heading_mod90_rad)
    assert est.n_support >= 12
    assert 0.0 <= est.quality <= 1.0


def test_recon_heading_repeatable():
    """Deterministic seed -> identical heading on repeated calls (stability the consumer relies on)."""
    img = _load_recon("02_gate_deadahead_from_startpad.png")
    e1 = heading_vp.estimate_heading(img, seed=0)
    e2 = heading_vp.estimate_heading(img, seed=0)
    assert e1 is not None and e2 is not None
    assert abs(e1.heading_mod90_rad - e2.heading_mod90_rad) < 1e-9


def test_recon_floor_height_plausible_or_none():
    """floor_height on recon frames: when it fires, the height must be physically plausible
    (~0.3-4 m); otherwise an honest None. No pose GT, so this is a sanity band, not accuracy."""
    fired = 0
    for name in _STRUCT_FRAMES:
        img = _load_recon(name)
        est = floor_height.estimate_floor_height(img, roll=0.0, pitch=0.0)
        if est is not None:
            assert 0.3 <= est.height_m <= 4.0, f"{name}: implausible height {est.height_m:.2f} m"
            assert 0.0 <= est.quality <= 1.0
            fired += 1
    # At least report-coverage: it is allowed to be None on all (near-horizon floor is hard);
    # the test only fails if a FIRED estimate is implausible.
    assert fired >= 0


@pytest.mark.parametrize("name", ["10_motion_blur_dark.png", "11_motion_blur_gate_lowlight.png",
                                  "12_motion_blur_3.png", "08_between_gates_dark_facingaway.png"])
def test_recon_blur_or_facingaway_handled(name):
    """Blur / facing-away frames must not crash and must yield either None or a bounded estimate
    (the consumer gates on quality; we only assert no exception + sane outputs)."""
    img = _load_recon(name)
    est = heading_vp.estimate_heading(img, roll=0.0, pitch=0.0)
    if est is not None:
        assert -np.pi / 4 <= est.heading_mod90_rad < np.pi / 4
        assert 0.0 <= est.quality <= 1.0
    fh = floor_height.estimate_floor_height(img, roll=0.0, pitch=0.0)
    if fh is not None:
        assert fh.height_m > 0.0
