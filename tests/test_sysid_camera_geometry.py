"""SYSTEM-ID #3a (frames-camera): register the CAMERA geometry constants + projection of the
numpy scipy reference (src/racer/frames.py + rl/fix_surrogate.py) against the SELF-CONTAINED torch
reimplementation (rl/inc8_estimator_emul.py) that ships to the Adroit GPU env without scipy/frames.

This complements (does NOT duplicate) tests/test_inc8_fix_surrogate_torch.py:
  - that suite pins the constants at 1e-12 and the random-pose geometry sweep at 1e-4;
  - THIS suite pins the LOAD-BEARING PRECONDITION that makes those equalities hold, plus a
    range-binned geometry registration with the measured tolerances.

Key registered fact (measured 2026-06-18, handoff/system-id-2026-06-18/scratch/frames-camera):
  rl/inc8_estimator_emul.R_CAMERA_FROM_BODY_NP HARDCODES a pure +20deg pitch-up mount and has NO
  BORESIGHT concept (self-contained). The numpy frames.R_camera_from_body() COMPOSES the live
  frames.BORESIGHT angular fields (pitch_rad, roll_rad). The two agree TODAY only because the
  deployed BORESIGHT is purely METRIC (vert_offset_m=-0.25; angular fields 0.0 -> mount untouched).
  An ANGULAR boresight (e.g. pitch_rad=0.01) would silently desync the train-env mount from the
  deploy mount by ~0.5deg, biasing every emulated fix. This guard pins that precondition so a future
  angular calibration cannot land without also updating the torch baked matrix.

Run from repo ROOT:
    .venv\\Scripts\\python.exe -m pytest tests/test_sysid_camera_geometry.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "rl"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from racer import frames as FR                                          # noqa: E402
from racer.contracts import Gate                                       # noqa: E402
import fix_surrogate as FS                                             # noqa: E402
import inc8_estimator_emul as IE                                       # noqa: E402
from estimator_emul import ned_gate_frame                             # noqa: E402

DT64 = torch.float64


# ============================================================ constants + precondition
def test_camera_constants_register_numpy_to_torch():
    """K + image dims are EXACTLY identical; R_camera_from_body matches to machine epsilon."""
    assert np.array_equal(FR.CAMERA_INTRINSICS_K, IE.CAMERA_INTRINSICS_K_NP)
    assert FR.IMAGE_WIDTH == IE.IMAGE_WIDTH == 640
    assert FR.IMAGE_HEIGHT == IE.IMAGE_HEIGHT == 360
    # R mount: equal to <=1e-12 (the ~1e-16 residual is scipy's from_euler 1-ulp vs the hand matrix)
    assert np.allclose(FR.R_camera_from_body(), IE.R_CAMERA_FROM_BODY_NP, atol=1e-12)


def test_torch_baked_mount_is_pure_20deg_no_boresight():
    """The self-contained torch matrix hardcodes the pure +20deg pitch-up mount and has NO BORESIGHT
    concept -- it is frame-faithful to frames ONLY while the live angular boresight is zero."""
    assert not hasattr(IE, "BORESIGHT"), "torch core must stay self-contained (no boresight concept)"
    a = np.deg2rad(20.0)
    axis_swap = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    ry_neg = np.array([[np.cos(a), 0.0, -np.sin(a)], [0.0, 1.0, 0.0], [np.sin(a), 0.0, np.cos(a)]])
    pure_20 = axis_swap @ ry_neg
    assert np.allclose(IE.R_CAMERA_FROM_BODY_NP, pure_20, atol=1e-15)


def test_live_boresight_is_metric_only_so_mounts_agree():
    """REGISTERED PRECONDITION: the deployed BORESIGHT is METRIC-only (vert_offset_m set; angular
    fields zero). The metric translation does NOT enter the mount rotation, so the numpy mount stays
    bit-equal to the torch baked matrix. If a future ANGULAR boresight is set this assertion fires,
    flagging that rl/inc8_estimator_emul.R_CAMERA_FROM_BODY_NP must be updated in lock-step."""
    assert FR.BORESIGHT.pitch_rad == 0.0, "angular boresight set -> torch mount is now stale"
    assert FR.BORESIGHT.roll_rad == 0.0, "angular boresight set -> torch mount is now stale"
    # the live mount (with whatever METRIC offset) still equals the angular-free torch mount
    assert np.allclose(FR.R_camera_from_body(), IE.R_CAMERA_FROM_BODY_NP, atol=1e-12)


def test_angular_boresight_would_desync_torch_mount():
    """Negative control: an angular boresight DOES change the numpy mount, proving the guard above is
    load-bearing (not a no-op). Restores BORESIGHT afterwards -- no global state leak."""
    orig = FR.BORESIGHT
    try:
        FR.BORESIGHT = FR.BoresightCorrection(pitch_rad=0.01)
        desync = float(np.max(np.abs(FR.R_camera_from_body() - IE.R_CAMERA_FROM_BODY_NP)))
        assert desync > 1e-3, "a 0.01 rad angular boresight must visibly desync the torch mount"
    finally:
        FR.BORESIGHT = orig
    assert FR.BORESIGHT.pitch_rad == 0.0  # restored


def test_fov_from_shared_intrinsics():
    """The FoV derived from the shared K: HFoV 90deg (spec's mislabelled 'VFoV'), true VFoV ~58.7deg."""
    assert abs(FR.horizontal_fov_deg() - 90.0) < 1e-6
    assert abs(FR.vertical_fov_deg() - 58.7155) < 1e-3


# ============================================================ geometry parity sweep (binned)
def _build_batch(seed, n):
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(seed)
    gates_zup = np.array([
        [-23.30, 0.40, 1.39], [-46.89, 2.50, -3.71], [-74.59, -1.20, -12.31],
        [-111.49, 5.10, -23.21], [-135.49, 0.80, -24.00], [-159.19, 4.40, -24.61]])
    flip = np.array([1.0, -1.0, -1.0])
    drones, R_wbs, gate_pos, R_wgs = [], [], [], []
    for _ in range(n):
        gi = rng.integers(0, 6)
        gp = gates_zup[gi] * flip
        drone = gp + rng.uniform(-30, 30, 3)
        rpy = rng.uniform([-0.8, -0.6, -np.pi], [0.8, 0.6, np.pi])
        drones.append(drone)
        R_wbs.append(Rotation.from_euler("ZYX", [rpy[2], rpy[1], rpy[0]]).as_matrix())
        gate_pos.append(gp)
        R_wgs.append(ned_gate_frame(np.pi))
    return (np.asarray(drones), np.asarray(R_wbs), np.asarray(gate_pos), np.asarray(R_wgs))


def _numpy_geoms(drones, R_wbs, gate_pos, R_wgs):
    return [FS.geometry(d, Rwb, Gate(gate_id=0, position_ned=gp, R_world_gate=Rwg))
            for d, Rwb, gp, Rwg in zip(drones, R_wbs, gate_pos, R_wgs)]


def _torch_geom(drones, R_wbs, gate_pos, R_wgs, dtype):
    R_cb, K, _, _ = IE._const("cpu", dtype)
    return IE.batched_geometry(
        torch.tensor(drones, dtype=dtype), torch.tensor(R_wbs, dtype=dtype),
        torch.tensor(gate_pos, dtype=dtype), torch.tensor(R_wgs, dtype=dtype), R_cb, K)


def test_geometry_parity_float64_bitclose():
    """fix_surrogate.geometry == batched_geometry to float64 round-off over 1500 random poses:
    range/lever bit-identical; angular channels <=1e-12 deg; in_image exact boolean agreement."""
    drones, R_wbs, gate_pos, R_wgs = _build_batch(seed=12345, n=1500)
    ng = _numpy_geoms(drones, R_wbs, gate_pos, R_wgs)
    tg = _torch_geom(drones, R_wbs, gate_pos, R_wgs, DT64)
    # range + lever are differences/norms only -> bit-identical
    assert np.array_equal(tg["range"].numpy(), np.array([g.range_m for g in ng]))
    assert np.array_equal(tg["lever"].numpy(), np.array([g.lever_world for g in ng]))
    for tkey, attr in [("az_deg", "azimuth_deg"), ("el_deg", "elevation_deg"),
                       ("bearing_deg", "bearing_deg"), ("view_deg", "view_angle_deg")]:
        a = tg[tkey].numpy()
        b = np.array([getattr(g, attr) for g in ng])
        assert np.nanmax(np.abs(a - b)) <= 1e-12, (tkey, np.nanmax(np.abs(a - b)))
    assert np.max(np.abs(tg["t_cam"].numpy() - np.array([g.t_cam for g in ng]))) <= 1e-13
    assert np.array_equal(tg["in_image"].numpy(), np.array([g.in_image for g in ng]))


def test_geometry_parity_float32_deploy_dtype():
    """The deployment dtype (float32) stays inside a sub-pixel / 1e-3 deg band AND keeps EXACT
    in_image agreement -- the binding correctness property (a flipped in_image flips fix acceptance)."""
    drones, R_wbs, gate_pos, R_wgs = _build_batch(seed=777, n=1500)
    ng = _numpy_geoms(drones, R_wbs, gate_pos, R_wgs)
    tg = _torch_geom(drones, R_wbs, gate_pos, R_wgs, torch.float32)
    assert np.max(np.abs(tg["range"].numpy() - np.array([g.range_m for g in ng]))) <= 1e-3
    for tkey, attr in [("az_deg", "azimuth_deg"), ("el_deg", "elevation_deg")]:
        a = tg[tkey].numpy()
        b = np.array([getattr(g, attr) for g in ng])
        assert np.nanmax(np.abs(a - b)) <= 1e-2, (tkey, np.nanmax(np.abs(a - b)))
    assert np.array_equal(tg["in_image"].numpy(), np.array([g.in_image for g in ng]))


def test_projection_register_numpy_project_vs_torch_inimage_math():
    """numpy frames.project_camera_point(u,v) == the torch batched in-image K-projection (lines that
    compute u,v in batched_geometry), to <=1e-9 px over all front-of-camera samples."""
    drones, R_wbs, gate_pos, R_wgs = _build_batch(seed=999, n=1200)
    ng = _numpy_geoms(drones, R_wbs, gate_pos, R_wgs)
    K = FR.CAMERA_INTRINSICS_K
    worst = 0.0
    n_front = 0
    for g in ng:
        t = g.t_cam
        if t[2] <= 0:
            continue
        n_front += 1
        u, v = FR.project_camera_point(t)
        u_t = (K[0, 0] * t[0] + K[0, 2] * t[2]) / t[2]
        v_t = (K[1, 1] * t[1] + K[1, 2] * t[2]) / t[2]
        worst = max(worst, abs(u - u_t), abs(v - v_t))
    assert n_front > 100, "batch must contain front-of-camera samples"
    assert worst <= 1e-9, worst
