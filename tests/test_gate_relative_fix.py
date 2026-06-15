"""Gate-relative in-plane +L fix + relative-innovation outlier gate — G2 (C2, BLUEPRINT §1.2/§1.3/§1.5).

Pins:
  - the in-plane cov is the TIGHT PnP lateral sigma (0.265 m, flat in-band) with NO 0.40 m floor; the
    along-track (gate-normal) axis is LOOSE and KEEPS the floor (anisotropic, gate-plane shaped);
  - sigma is a single swappable constant, flat in the gate-4 window, a1*r law only beyond ~10 m;
  - the relative-innovation gate (chi2(2,0.999)=13.82) keeps ~99.9% of clean fixes and rejects ~99.8%
    of depth-flips -- the in-plane backstop reproj + the absolute 3-DOF Maha gate cannot provide.

Torch-free (numpy/scipy only). [C2-ESTIMATOR-CHAIN 2026-06-13; MC ported from d2_relinnov_gate_check]
"""
import numpy as np
import pytest

from racer.contracts import Gate, GatePose
from racer.frames import ATTITUDE_NOISE_STD_RAD, R_camera_from_body
from racer.localization import (
    FIX_COV_FLOOR_STD,
    GATE_REL_ALONG_SIGMA,
    GATE_REL_INPLANE_SIGMA,
    GATE_REL_RANGE_GROWTH_A1,
    gate_relative_inplane_fix,
)

CHI2_2_999 = 13.815510557964274

# Gate-4-like frame: gate faces -N, so in-plane = (E, D), along-track = N. R_world_gate columns are
# X=right(E)=[0,1,0], Y=down(D)=[0,0,1], Z=through(N)=[1,0,0] (same convention as test_casec_foundation).
_R_GATE_FACING_N = np.column_stack([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
_GATE_POS = np.array([-135.5, -0.8, 25.36])


def _fix_for_lever(L_world, *, R_wb=np.eye(3), range_growth_a1=GATE_REL_RANGE_GROWTH_A1):
    """Build (z_ned, cov_ned) for a sighting whose world +L lever is exactly ``L_world``."""
    R_wc = R_wb @ R_camera_from_body().T
    t_cam_gate = R_wc.T @ np.asarray(L_world, float)          # invert L = R_wc @ t_cam_gate
    gp = GatePose(frame_id=0, sim_time_ns=0, R_cam_gate=np.eye(3), t_cam_gate=t_cam_gate,
                  reproj_error_px=0.0, gate_id=4, covariance=None, n_corners=4)
    gate = Gate(gate_id=4, position_ned=_GATE_POS, R_world_gate=_R_GATE_FACING_N)
    return gate_relative_inplane_fix(gp, gate, R_wb, range_growth_a1=range_growth_a1)


# ---------------------------------------------------------------------------
# (1) anisotropic cov: tight in-plane (no floor) / loose along-track (floor kept)
# ---------------------------------------------------------------------------
def test_inplane_cov_is_tight_pnp_lateral_with_no_floor():
    # L along -N at 8 m (in the gate-4 window): in-plane = E,D, along = N.
    z, cov = _fix_for_lever([-8.0, 0.0, 0.0])
    # gate axes align with NED here -> cov is diagonal: [N(along), E(in-plane), D(in-plane)].
    var_E, var_D, var_N = cov[1, 1], cov[2, 2], cov[0, 0]
    sig_ip = GATE_REL_INPLANE_SIGMA                                  # 8 m < 10.2 m -> flat 0.265
    np.testing.assert_allclose([var_E, var_D], [sig_ip**2, sig_ip**2], rtol=0, atol=1e-12)
    # the in-plane variance carries NO 0.40 m bias-absorption floor (that is the whole point):
    assert var_E < (sig_ip**2 + FIX_COV_FLOOR_STD**2) - 1e-9
    # along-track is LOOSE and KEEPS the floor + the attitude lever term:
    expected_along = GATE_REL_ALONG_SIGMA**2 + (ATTITUDE_NOISE_STD_RAD * 8.0)**2 + FIX_COV_FLOOR_STD**2
    np.testing.assert_allclose(var_N, expected_along, rtol=0, atol=1e-12)
    assert var_N > 5.0 * var_E                                       # along >> in-plane


def test_cov_is_symmetric_psd_off_axis_gate():
    # A tilted gate (non-axis-aligned) must still produce a symmetric PSD cov.
    rng = np.random.default_rng(0)
    from scipy.spatial.transform import Rotation
    R_gate = Rotation.from_euler("ZYX", rng.normal(0, 0.4, 3)).as_matrix()
    gate = Gate(gate_id=4, position_ned=_GATE_POS, R_world_gate=R_gate)
    gp = GatePose(frame_id=0, sim_time_ns=0, R_cam_gate=np.eye(3),
                  t_cam_gate=R_camera_from_body() @ np.array([0.0, 0.0, 9.0]),
                  reproj_error_px=0.0, gate_id=4, covariance=None, n_corners=4)
    _, cov = gate_relative_inplane_fix(gp, gate, np.eye(3))
    np.testing.assert_allclose(cov, cov.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(cov) > 0)


def test_inplane_sigma_flat_in_band_grows_beyond_10m():
    _, cov_near = _fix_for_lever([-8.0, 0.0, 0.0])     # in-band
    _, cov_far = _fix_for_lever([-20.0, 0.0, 0.0])     # beyond ~10 m
    np.testing.assert_allclose(cov_near[1, 1], GATE_REL_INPLANE_SIGMA**2, atol=1e-12)   # flat 0.265
    assert cov_far[1, 1] > GATE_REL_INPLANE_SIGMA**2 + 1e-6                              # a1*r grows
    np.testing.assert_allclose(np.sqrt(cov_far[1, 1]), GATE_REL_RANGE_GROWTH_A1 * 20.0, rtol=1e-9)


def test_z_ned_matches_absolute_fix_position(monkeypatch):
    # The pseudo-fix world position equals gate.position_ned - L (== the absolute fix); the WIN is the
    # cov shaping + obs sourcing, not a different z. Pins the UNBIASED +L lever output, so zero the
    # shipped metric bake (vert_offset_m=-0.25, boresight-closure-2026-06-14) here -- this tests lever
    # geometry (z == gate - L), a zero-calibration property, not the deployed calibration offset.
    import racer.frames as _F
    monkeypatch.setattr(_F, "BORESIGHT", _F.BoresightCorrection())
    L = np.array([-9.0, 0.3, -0.2])
    z, _ = _fix_for_lever(L)
    np.testing.assert_allclose(z, _GATE_POS - L, atol=1e-9)


# ---------------------------------------------------------------------------
# (2) relative-innovation gate separates depth-flips (reproj cannot)
# ---------------------------------------------------------------------------
def test_relinnov_gate_keeps_clean_rejects_flips():
    # MC (gate-4 plane, in-plane axes = E,D): clean fixes = true offset + N(0, 0.265^2); depth-flips
    # throw a LARGE (1.5-3 m) in-plane displacement (the flipped R_cam_gate rotates the lever). The
    # relative-innovation statistic d2 = nu^T S^-1 nu (S = warm prior P_ip + R_ip, R_ip = 0.265^2 I,
    # the helper's in-plane cov) cleanly separates them at chi2(2,0.999). [d2_relinnov_gate_check]
    rng = np.random.default_rng(20260613)
    sig = GATE_REL_INPLANE_SIGMA
    P_ip = (0.08**2) * np.eye(2)
    S_inv = np.linalg.inv(P_ip + (sig**2) * np.eye(2))
    n, flip_frac = 6000, 0.15
    n_flip = int(n * flip_frac)
    n_clean = n - n_flip

    e_true_c = rng.normal(0.0, 0.03, size=(n_clean, 2))
    e_pred_c = e_true_c + rng.multivariate_normal(np.zeros(2), P_ip, size=n_clean)
    e_obs_c = e_true_c + rng.normal(0.0, sig, size=(n_clean, 2))

    e_true_f = rng.normal(0.0, 0.03, size=(n_flip, 2))
    e_pred_f = e_true_f + rng.multivariate_normal(np.zeros(2), P_ip, size=n_flip)
    mag = rng.uniform(1.5, 3.0, size=n_flip)
    ang = rng.uniform(0, 2 * np.pi, size=n_flip)
    e_obs_f = (e_true_f + rng.normal(0.0, sig, size=(n_flip, 2))
               + np.column_stack([mag * np.cos(ang), mag * np.sin(ang)]))

    def d2(e_obs, e_pred):
        nu = e_obs - e_pred
        return np.einsum("ij,jk,ik->i", nu, S_inv, nu)

    clean_kept = float(np.mean(d2(e_obs_c, e_pred_c) <= CHI2_2_999))
    flip_survived = float(np.mean(d2(e_obs_f, e_pred_f) <= CHI2_2_999))
    assert clean_kept > 0.99, f"clean kept only {clean_kept:.4f} (< 99.9% target)"
    assert flip_survived < 0.02, f"flips survived {flip_survived:.4f} (gate not rejecting depth-flips)"
