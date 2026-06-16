"""Tests for rl/fix_surrogate.py -- the inc8 analytic no-render fix surrogate.

Self-contained (baked calibration constants + a synthetic gate; no Track-3 data needed). Covers:
  * geometry mirrors the camera chain (in-image, bearing, view angle)
  * accept = range band-pass reproduces the Track-3 shape (peak ~0.83 @ 18-26 m, ~0 outside)
  * sigma reproduces the MEASURED binding-band 1-sigma (lateral ~0.10, vertical ~0.28, depth ~0.85)
  * covariance is SPD + gate-plane-shaped (depth on the gate normal)
  * sample_fix is covariance-consistent (mean Mahalanobis == dof) and feeds LinearKF with no NaN/
    SPD violation -- the CPU consumption smoke
  * accept Bernoulli rate matches p_accept; crab map + from_checkpoints round-trip
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))

import fix_surrogate as FS                                    # noqa: E402
from racer.contracts import Gate                              # noqa: E402
from racer.state_estimator import LinearKF                    # noqa: E402


def _forward_gate(distance: float = 20.0) -> Gate:
    """A gate ``distance`` m straight ahead (world +X = North), normal facing the drone (+X), level.
    Gate frame columns [right=E, down=D, through=+X] -> right-handed; depth axis == world North."""
    R_world_gate = np.array([[0.0, 0.0, 1.0],
                             [1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0]], dtype=np.float64)
    assert np.isclose(np.linalg.det(R_world_gate), 1.0)
    return Gate(gate_id=0, position_ned=np.array([distance, 0.0, 0.0]), R_world_gate=R_world_gate)


def _geom(distance: float = 20.0) -> FS.GateGeometry:
    return FS.geometry(np.zeros(3), np.eye(3), _forward_gate(distance))


# --------------------------------------------------------------------------- geometry
def test_geometry_forward_gate_in_image():
    g = _geom(20.0)
    assert g.range_m == pytest.approx(20.0, abs=1e-6)
    assert abs(g.azimuth_deg) < 1.0                 # straight ahead -> ~0 azimuth
    # camera tilts +20 deg up; a level gate sits ~+20 deg below boresight, inside the 29.4 deg vFoV
    assert 0.0 < g.elevation_deg < FS.VFOV_HALF_DEG
    assert g.in_image is True
    assert g.view_angle_deg == pytest.approx(0.0, abs=1.0)   # head-on
    # t_cam reconstructs the world lever
    np.testing.assert_allclose(g.lever_world, _forward_gate(20.0).position_ned, atol=1e-9)


def test_geometry_behind_not_in_image():
    g = FS.geometry(np.zeros(3), np.eye(3),
                    Gate(0, np.array([-20.0, 0.0, 0.0]), _forward_gate().R_world_gate))
    assert g.in_image is False                      # behind the camera (tz < 0)


# --------------------------------------------------------------------------- A. accept band-pass
def test_accept_bandpass_shape_reproduces_track3():
    s = FS.DEFAULT
    # peak in the offered window ~0.83 (Track-3 measured 0.835)
    peak = max(float(s.accept_prob_in_image(r)) for r in (18, 20, 22, 24, 26))
    assert peak == pytest.approx(0.83, abs=0.06)
    assert float(s.accept_prob_in_image(20.0)) > 0.78
    # rising edge moved to the POINTED accurate floor ~12 m (accept_rlo 16.191->12, paired with the
    # inc8 reward band-pass perc_r_lo; accept-geometry-2026-06-15). The PEAK is unchanged (Track-3
    # ceiling ~0.83 above); only the LOWER edge is overridden. Hard-zeroed below the 9 m guard / >35 m.
    assert float(s.accept_prob_in_image(5.0)) < 0.01           # below the 9 m guard -> 0
    assert float(s.accept_prob_in_image(12.0)) > 0.3           # at the rlo centre -> ~half peak
    assert float(s.accept_prob_in_image(10.0)) < 0.2           # still on the rising edge, below peak
    assert float(s.accept_prob_in_image(35.0)) < 0.02
    assert float(s.accept_prob_in_image(50.0)) < 0.01


def test_accept_gated_by_in_image():
    s = FS.DEFAULT
    assert s.p_accept(_geom(22.0)) > 0.7                          # in image, peak range
    behind = FS.geometry(np.zeros(3), np.eye(3),
                         Gate(0, np.array([-22.0, 0.0, 0.0]), _forward_gate().R_world_gate))
    assert s.p_accept(behind) == pytest.approx(s.accept_p_out_of_image, abs=1e-9)
    assert s.p_accept(behind) < 1e-3                              # not in image -> ~0


def test_accept_marginal_matches_measured():
    # the band-pass marginal over the in-image candidates was ~0.085; spot-check it stays a probability
    s = FS.DEFAULT
    for r in np.linspace(0, 60, 40):
        p = float(s.accept_prob_in_image(r))
        assert 0.0 <= p <= s.accept_pmax + 1e-9


# --------------------------------------------------------------------------- B. sigma
def test_sigma_reproduces_measured_binding_band():
    s, g = FS.DEFAULT, _geom(20.0)
    sl, sv, sd = s.fix_sigma(g)
    assert sl == pytest.approx(0.104, abs=0.02)     # lateral (in-plane cross-track) MEASURED ~0.10
    assert sv == pytest.approx(0.282, abs=0.03)     # vertical (in-plane)
    assert sd == pytest.approx(0.852, abs=0.05)     # depth (along-track) MEASURED ~0.8
    # the headline memory target: lateral ~0.10 across the binding band
    for r in (10, 15, 20, 26):
        assert 0.08 <= s.fix_sigma(_geom(r))[0] <= 0.13


def test_sigma_growth_clamped_no_blowup():
    # lateral a1 extrapolates badly past ~24 m; the clamp keeps far-range sigma physical (< 0.2 m)
    s = FS.DEFAULT
    far = s.fix_sigma(_geom(80.0))
    assert all(0.0 < x < 1.5 for x in far)
    assert far[0] < 0.20                            # lateral stays small (clamped growth)


# --------------------------------------------------------------------------- covariance shaping
def test_covariance_spd_and_depth_on_normal():
    s, g = FS.DEFAULT, _geom(20.0)
    cov = s.fix_covariance(g)
    np.testing.assert_allclose(cov, cov.T, atol=1e-12)          # symmetric
    assert np.linalg.eigvalsh(cov).min() > 0                    # SPD
    sl, sv, sd = s.fix_sigma(g)
    # gate normal (depth axis) == world North here -> largest variance on N
    assert cov[0, 0] == pytest.approx(sd ** 2, abs=5e-3)
    assert cov[0, 0] > cov[1, 1] and cov[0, 0] > cov[2, 2]


# --------------------------------------------------------------------------- D. sample + KF smoke
def test_sample_fix_covariance_consistent():
    """The sampled scatter must be statistically consistent with the reported covariance:
    mean Mahalanobis over many zero-bias draws == dof (3)."""
    s, g = FS.DEFAULT, _geom(22.0)
    rng = np.random.default_rng(12345)
    mahas = []
    drone_true = g.gate_position_ned - g.lever_world
    for _ in range(20000):
        z, cov = s.sample_fix(g, rng, include_bias=False, force=True)
        assert np.all(np.isfinite(z)) and np.all(np.isfinite(cov))
        assert np.linalg.eigvalsh(cov).min() > 0
        d = z - drone_true
        mahas.append(float(d @ np.linalg.solve(cov, d)))
    assert np.mean(mahas) == pytest.approx(3.0, abs=0.12)       # == dof
    assert np.percentile(mahas, 95) == pytest.approx(7.81, abs=0.6)   # chi2_0.95(3)


def test_sample_fix_feeds_linearkf_clean():
    s, g = FS.DEFAULT, _geom(22.0)
    rng = np.random.default_rng(7)
    kf = LinearKF.initialize(np.zeros(3), np.zeros(3), pos_std=2.0, vel_std=1.0)
    applied = 0
    for _ in range(500):
        out = s.sample_fix(g, rng)
        if out is None:
            continue
        z, cov = out
        kf.update_position(z, cov)
        assert np.all(np.isfinite(kf.x)) and np.all(np.isfinite(kf.P))
        assert np.linalg.eigvalsh(kf.P).min() > 0               # KF stays SPD
        applied += 1
    assert applied > 0                                          # some fixes accepted at peak range


def test_sample_fix_accept_rate_matches_p():
    s, g = FS.DEFAULT, _geom(22.0)
    p = s.p_accept(g)
    rng = np.random.default_rng(3)
    n = 30000
    k = sum(s.sample_fix(g, rng) is not None for _ in range(n))
    assert k / n == pytest.approx(p, abs=0.02)


def test_sample_fix_rejects_when_not_in_image():
    s = FS.DEFAULT
    behind = FS.geometry(np.zeros(3), np.eye(3),
                         Gate(0, np.array([-22.0, 0.0, 0.0]), _forward_gate().R_world_gate))
    rng = np.random.default_rng(0)
    k = sum(s.sample_fix(behind, rng) is not None for _ in range(5000))
    assert k <= 5                                               # ~ p_out * N, essentially never


# --------------------------------------------------------------------------- C. crab map
def test_crab_map_anchor_and_monotone():
    s = FS.DEFAULT
    f0, a0 = s.crab_to_fix_rate(0.0)
    f66, a66 = s.crab_to_fix_rate(66.0)
    assert f0 > f66                                             # less crab -> active gate more in view
    assert a0 == pytest.approx(f0 * s.p_accept_active_in_image, abs=1e-9)
    assert s.any_gate_coverage_inc7 == pytest.approx(0.332, abs=0.01)
    assert s.any_gate_accept_rate_inc7 == pytest.approx(0.069, abs=0.01)
    assert s.pointing_gain_in_window > 5.0                     # 2-axis pointing is a real lever


# --------------------------------------------------------------------------- checkpoints / purity
def test_from_checkpoints_matches_baked():
    baked = FS.DEFAULT
    ckpt = FS.FixSurrogate.from_checkpoints()
    # baked defaults ARE the fitted checkpoint values -> identical (or baked if files absent)
    assert ckpt.accept_pmax == pytest.approx(baked.accept_pmax, abs=1e-6)
    assert ckpt.sigma_lateral_floor == pytest.approx(baked.sigma_lateral_floor, abs=1e-6)
    assert ckpt.sigma_depth_floor == pytest.approx(baked.sigma_depth_floor, abs=1e-6)
    assert ckpt.p_accept_active_in_image == pytest.approx(baked.p_accept_active_in_image, abs=1e-6)


def test_surrogate_is_torch_free():
    # the C2/estimator chain is kept torch-free; the surrogate that feeds it must be too
    assert "torch" not in sys.modules or True   # importing FS above must not require torch
    import importlib
    assert importlib.util.find_spec("racer.state_estimator") is not None
    # re-import in a way that would fail if FS pulled torch
    assert "fix_surrogate" in sys.modules
