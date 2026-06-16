"""In-plane STATE-covariance floor on LinearKF [parked #74, coast-drift 2026-06-15].

Pins the fix for KF OVER-CONVERGENCE on a dense gate-relative fix stream: without the floor the position
covariance P -> R/N -> ~0, the Kalman gain -> 0, and the filter stops trusting fixes / rides the drifting
IMU ("centering-blind") exactly when a well-pointed inc8 policy makes fixes densest. The floor caps the
in-plane (horizontal/centering) position variance at the irreducible systematic ``sigma_b`` (= sigma_ref
= 0.05 m) so the gain stays responsive. Tests:
  - dense fixes collapse the in-plane sigma WITHOUT the floor, but the floor holds it at sigma_floor;
  - the floored KF stays RESPONSIVE (next-fix gain non-negligible and >> the unfloored gain);
  - the floor is an EIGENVALUE floor (catches an over-converged direction at any yaw) and leaves the
    loose along-track horizontal eigenvalue + the vertical (world-down) + velocity blocks untouched;
  - P stays symmetric + positive-definite;
  - floor OFF (or block already above floor) is BIT-IDENTICAL to the bare filter;
  - NEES calibration: the floor takes a wildly over-confident filter back to ~1 DOF (honest), never
    under-confident;
  - the over-convergence is MILDER with realistic IMU process noise (Q) than the iid coast-drift MC
    implied -- the floor is sound robustness either way.
"""
import numpy as np
import pytest

from racer.state_estimator import LinearKF

_REST = np.array([0.0, 0.0, -9.80665])     # body specific force at level hover (gravity sign trap)
_R_LAT = 0.265 ** 2                          # gate-relative per-axis in-plane R (localization.GATE_REL_INPLANE_SIGMA)
_FLOOR = 0.05                                # = localization.INPLANE_POS_FLOOR_STD = sigma_ref = sigma_b


def _inplane_min_sigma(P: np.ndarray) -> float:
    """Smallest-eigenvalue 1-sigma of the horizontal (N-E) position block -- the binding centering axis."""
    w = np.linalg.eigvalsh(0.5 * (P[:2, :2] + P[:2, :2].T))
    return float(np.sqrt(max(w[0], 0.0)))


def _dense_fix_run(floor_std: float, n_fixes: int = 200, with_predict: bool = False,
                   fix=np.zeros(3), seed: int | None = None) -> LinearKF:
    """Drive a KF with a dense stream of tight in-plane fixes (gate-relative-like cov: tight E/D, loose N).

    ``with_predict`` interleaves an IMU predict (adds process noise Q) between fixes -- the realistic
    case. Without it, P collapses as the pure R/N independent-update accumulation (the iid MC regime).
    """
    rng = np.random.default_rng(seed) if seed is not None else None
    kf = LinearKF.initialize(np.zeros(3), np.zeros(3), pos_std=1.0, vel_std=1.0,
                             inplane_pos_floor_std=floor_std, accel_noise_std=0.3)
    cov = np.diag([0.5 ** 2, _R_LAT, _R_LAT])      # N loose (along-track), E/D tight (in-plane)
    z0 = np.asarray(fix, dtype=np.float64)
    for _ in range(n_fixes):
        if with_predict:
            kf.predict(_REST, np.eye(3), 1.0 / 90.0)
        z = z0 if rng is None else z0 + np.array([rng.normal(0, 0.5),
                                                  rng.normal(0, 0.265), rng.normal(0, 0.265)])
        kf.update_position(z, cov)
    return kf


# --------------------------------------------------------------------------- collapse + floor

def test_dense_fix_stream_collapses_inplane_without_floor():
    # The bug: a dense fix stream drives the in-plane covariance well below the systematic floor.
    kf = _dense_fix_run(floor_std=0.0, n_fixes=200)
    assert _inplane_min_sigma(kf.P) < 0.5 * _FLOOR        # collapsed to ~0.019 m (< 0.025) -- over-converged


def test_floor_holds_inplane_sigma_at_floor():
    # The fix: the floor pins the in-plane min-eigenvalue sigma at exactly sigma_floor, no matter how
    # many dense fixes accumulate.
    kf = _dense_fix_run(floor_std=_FLOOR, n_fixes=500)
    assert _inplane_min_sigma(kf.P) == pytest.approx(_FLOOR, abs=1e-9)


def test_floored_kf_stays_responsive_to_new_fixes():
    # Centering-blindness = the gain decays to ~0. After a dense stream, probe how much of a fresh 1 m
    # in-plane discrepancy ONE more fix absorbs: the floored gain must be non-negligible AND clearly
    # larger than the unfloored (decayed) gain.
    cov = np.diag([0.5 ** 2, _R_LAT, _R_LAT])
    def next_fix_gain(floor_std):
        kf = _dense_fix_run(floor_std=floor_std, n_fixes=200)
        x0 = kf.x.copy()
        kf.update_position(np.array([0.0, 1.0, 0.0]), cov)    # 1 m throw on the E (in-plane) axis
        return float(kf.x[1] - x0[1])                          # fraction absorbed == effective gain
    g_floor = next_fix_gain(_FLOOR)
    g_bare = next_fix_gain(0.0)
    # steady-state floored gain = floor^2 / (floor^2 + R_lat) ~ 0.034: non-negligible and NON-decaying
    assert g_floor == pytest.approx(_FLOOR ** 2 / (_FLOOR ** 2 + _R_LAT), rel=0.05)
    assert g_floor > 1.5 * g_bare                             # floor keeps the gain alive vs the decayed bare


def test_floor_is_eigenvalue_not_axis_aligned():
    # The binding gate-LATERAL axis is horizontal at the gate's YAW -- an arbitrary direction in N-E.
    # An eigenvalue floor catches the over-converged direction even when it is diagonal in N-E (a pure
    # per-axis diag floor would miss it). Build a tight fix along a 45-deg in-plane direction.
    u = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)              # 45 deg in the N-E plane
    perp = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
    # cov tight along u (0.01 m), loose along perp + vertical
    cov = 0.01 ** 2 * np.outer(u, u) + 1.0 * np.outer(perp, perp) + 1.0 * np.outer([0, 0, 1], [0, 0, 1])
    kf = LinearKF.initialize(np.zeros(3), np.zeros(3), pos_std=1.0, vel_std=1.0,
                             inplane_pos_floor_std=_FLOOR)
    for _ in range(50):
        kf.update_position(np.zeros(3), cov)
    # the over-converged direction is u (diagonal in N-E); its variance must be floored
    var_u = float(u[:2] @ kf.P[:2, :2] @ u[:2])
    assert var_u >= _FLOOR ** 2 - 1e-9
    assert _inplane_min_sigma(kf.P) == pytest.approx(_FLOOR, abs=1e-6)


def test_floor_application_only_touches_inplane_block():
    # ONE floor application raises only the sub-floor in-plane eigenvalue: a horizontal eigenvalue already
    # ABOVE the floor (the loose along-track direction), the vertical (world-down) position, the velocity
    # block, and the pos-vel cross blocks are all bit-identical. (Across a multi-step RUN the floored
    # trajectory legitimately diverges from the bare one via the cross-covariance feedback -- that is the
    # point of the floor -- so the isolation property is the per-application one, asserted here.)
    kf = LinearKF.initialize(np.zeros(3), np.zeros(3), inplane_pos_floor_std=_FLOOR)
    # a realistic converged P: N loose (along-track, var 0.04 >> floor^2), E tight (lateral, below floor),
    # D collapsed (vertical), with nonzero pos-vel cross terms + a velocity block.
    P = np.diag([0.20 ** 2, 0.01 ** 2, 0.001 ** 2, 0.3 ** 2, 0.3 ** 2, 0.3 ** 2]).astype(float)
    P[0, 3] = P[3, 0] = 0.02
    P[1, 4] = P[4, 1] = 0.001
    kf.P = P.copy()
    kf._apply_inplane_pos_floor()
    assert kf.P[0, 0] == pytest.approx(0.20 ** 2, rel=1e-9)         # along-track horizontal axis: untouched
    assert kf.P[1, 1] == pytest.approx(_FLOOR ** 2, abs=1e-12)      # lateral axis: raised to the floor
    assert kf.P[2, 2] == pytest.approx(0.001 ** 2, rel=1e-12)       # vertical (world-down): untouched
    np.testing.assert_array_equal(kf.P[3:, 3:], P[3:, 3:])          # velocity block: untouched
    np.testing.assert_array_equal(kf.P[:3, 3:], P[:3, 3:])          # pos-vel cross blocks: untouched
    np.testing.assert_array_equal(kf.P[2, :], P[2, :])              # the entire vertical row: untouched


def test_floored_P_is_symmetric_and_positive_definite():
    kf = _dense_fix_run(floor_std=_FLOOR, n_fixes=200, with_predict=True, seed=7)
    np.testing.assert_allclose(kf.P, kf.P.T, atol=1e-15)
    assert np.all(np.linalg.eigvalsh(0.5 * (kf.P + kf.P.T)) > 0.0)


# --------------------------------------------------------------------------- byte-identical (OFF / no-op)

def test_floor_off_is_byte_identical_to_default():
    # floor_std = 0.0 must reproduce the default (no-kwarg) filter bit-for-bit over a dense stream.
    kf_off = _dense_fix_run(floor_std=0.0, n_fixes=200, with_predict=True, seed=3)
    kf_default = LinearKF.initialize(np.zeros(3), np.zeros(3), pos_std=1.0, vel_std=1.0, accel_noise_std=0.3)
    cov = np.diag([0.5 ** 2, _R_LAT, _R_LAT])
    rng = np.random.default_rng(3)
    for _ in range(200):
        kf_default.predict(_REST, np.eye(3), 1.0 / 90.0)
        z = np.array([rng.normal(0, 0.5), rng.normal(0, 0.265), rng.normal(0, 0.265)])
        kf_default.update_position(z, cov)
    np.testing.assert_array_equal(kf_off.x, kf_default.x)
    np.testing.assert_array_equal(kf_off.P, kf_default.P)


def test_floor_is_noop_when_block_above_floor():
    # When the in-plane block never drops below the floor (sparse / loose fixes), the floored filter is
    # BIT-identical to the bare one -- the early-return guard means P is never touched.
    cov = (0.30 ** 2) * np.eye(3)                            # loose fixes -> sigma stays well above 0.05
    kf_floor = LinearKF.initialize(np.zeros(3), np.zeros(3), pos_std=0.4, vel_std=0.4,
                                   inplane_pos_floor_std=_FLOOR)
    kf_bare = LinearKF.initialize(np.zeros(3), np.zeros(3), pos_std=0.4, vel_std=0.4)
    for _ in range(20):
        kf_floor.update_position(np.zeros(3), cov)
        kf_bare.update_position(np.zeros(3), cov)
    assert _inplane_min_sigma(kf_bare.P) > _FLOOR            # never over-converged here
    np.testing.assert_array_equal(kf_floor.P, kf_bare.P)     # so the floor is a true no-op (bit-identical)


# --------------------------------------------------------------------------- NEES calibration

def _nees_lateral(use_floor: bool, n_runs: int = 400, n_fixes: int = 300) -> tuple[float, float]:
    """Aggregate lateral NEES = mean(e^2)/mean(P) (1-DOF; ~1 when calibrated) over an MC of dense-fix runs
    carrying a CONSTANT lateral systematic bias b == sigma_floor (the irreducible sigma_b the +L lever
    does not cancel). Axis-aligned gate -> lateral == world-E axis."""
    b = _FLOOR
    cov = np.diag([0.5 ** 2, _R_LAT, _R_LAT])
    es, Ps = [], []
    for s in range(n_runs):
        r = np.random.default_rng(1000 + s)
        kf = LinearKF.initialize(np.zeros(3), np.zeros(3), pos_std=1.0, vel_std=1.0,
                                 inplane_pos_floor_std=(_FLOOR if use_floor else 0.0), accel_noise_std=0.3)
        for _ in range(n_fixes):
            z = np.array([r.normal(0, 0.5), b + r.normal(0, 0.265), r.normal(0, 0.265)])
            kf.update_position(z, cov)
        es.append(kf.x[1] - 0.0)         # truth lateral = 0; estimate converges to the bias b
        Ps.append(kf.P[1, 1])
    return float(np.mean(np.array(es) ** 2) / np.mean(Ps)), float(np.mean(Ps))


def test_floor_keeps_nees_calibrated_not_overconfident():
    # Without the floor, P collapses far below the true systematic error -> NEES >> 1 (over-confident /
    # dishonest). The floor takes it back to ~1 DOF: honest, and NOT under-confident (NEES not << 1).
    nees_floor, P_floor = _nees_lateral(use_floor=True)
    nees_bare, P_bare = _nees_lateral(use_floor=False)
    assert P_floor == pytest.approx(_FLOOR ** 2, abs=1e-6)   # P held at the systematic floor
    assert 0.5 < nees_floor < 3.5                            # order-1: honest, not under-confident
    assert nees_bare > 2.0 * nees_floor                      # bare filter is materially over-confident


def test_over_convergence_is_milder_with_realistic_process_noise():
    # ESCAPE-HATCH finding: the coast-drift iid MC over-stated the collapse. With realistic IMU process
    # noise (Q, interleaved predicts), the bare in-plane sigma collapses MUCH less than in the pure-R/N
    # (no-Q) regime -- the floor is still a sound robustness measure, but the over-convergence it guards
    # against is partly self-limited by Q in the live filter.
    sig_noQ = _inplane_min_sigma(_dense_fix_run(floor_std=0.0, n_fixes=200, with_predict=False).P)
    sig_Q = _inplane_min_sigma(_dense_fix_run(floor_std=0.0, n_fixes=200, with_predict=True).P)
    assert sig_Q > 1.5 * sig_noQ                             # Q partially counteracts the R/N collapse
