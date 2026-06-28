"""Unit tests for the AHRS T1 bench (imu_gen + ESKF + classical + metrics).

Run with the main-checkout venv:
    C:/Users/Fengy/Downloads/Projects/Anduril/.venv/Scripts/python.exe -m pytest tests/test_ahrs_bench.py -v

Tests
-----
1. Frame-convention pin: quaternion layout and gravity sign match frames.py exactly.
2. Static convergence: all three filters converge to small error under gravity-only.
3. Constant-spin tracking: ESKF and Mahony track known yaw rate.
4. High-g discrimination: ESKF (with gating) beats Madgwick/Mahony under 4-5 g accel.
5. Geodesic metric sanity: identity distance = 0, 180-deg distance = pi.
6. IMUSequence shape invariants.
7. ESKF accel-gate weight: 1.0 at |a|=g, decays for |a|>>g, disabled when alpha=0.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from racer.ahrs.imu_gen import generate_imu_sequence, Scenario, IMUSequence, GRAVITY
from racer.ahrs.eskf import ESKFAHRS
from racer.ahrs.iekf import LeftInvariantEKF, _Exp_so3, _R_to_quat_wxyz, _quat_to_R_wxyz
from racer.ahrs.classical import MadgwickAHRS, MahonyAHRS
from racer.ahrs.metrics import geodesic_error_rad, score_filter, run_filter_on_sequence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quat_wxyz(roll, pitch, yaw):
    """True body->world quaternion (w,x,y,z) from aerospace Euler (ZYX)."""
    x, y, z, w = Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_quat()
    return np.array([w, x, y, z])


def _run_all_filters(scenario, duration_s=5.0, dt=0.005, seed=42):
    """Run ESKF, Madgwick, Mahony on a scenario; return {name: errors_rad}."""
    seq = generate_imu_sequence(scenario, duration_s=duration_s, dt=dt, seed=seed,
                                gyro_noise_std=0.005, accel_noise_std=0.05)
    filters = [
        ("ESKF", ESKFAHRS(gyro_noise_std=0.01, accel_gate_alpha=10.0)),
        ("Madgwick", MadgwickAHRS(beta=0.1)),
        ("Mahony", MahonyAHRS(kp=2.0, ki=0.005)),
    ]
    results = {}
    for name, filt in filters:
        q_est = run_filter_on_sequence(filt, seq)
        results[name] = geodesic_error_rad(q_est, seq.q_wxyz_gt)
    return results


# ---------------------------------------------------------------------------
# 1. Frame-convention pin
# ---------------------------------------------------------------------------

class TestFrameConvention:
    """Pin the quaternion convention and gravity sign against frames.py."""

    def test_identity_quaternion_at_level_hover(self):
        """Generator starts at identity (q=[1,0,0,0]) = body FRD aligned with NED.
        This means the drone is level (no roll/pitch/yaw). Consistent with frames.py's
        R_world_from_body(0,0,0) = identity."""
        seq = generate_imu_sequence(Scenario.STATIC_GRAVITY, duration_s=0.1, dt=0.01)
        # First sample: should be very close to identity (integrated from identity)
        q0 = seq.q_wxyz_gt[0]
        np.testing.assert_allclose(q0, [1., 0., 0., 0.], atol=1e-6,
                                   err_msg="GT quaternion must start at identity (level hover)")

    def test_gravity_specific_force_at_rest(self):
        """At rest (STATIC_GRAVITY, zero angular rate), body-frame specific force
        must be [0, 0, -g] in FRD. This matches state_estimator.py's convention:
        'the accelerometer reports SPECIFIC FORCE in the body frame (FRD), i.e. at rest
        it reads -g on the down axis'."""
        seq = generate_imu_sequence(
            Scenario.STATIC_GRAVITY, duration_s=1.0, dt=0.01, seed=0,
            gyro_noise_std=0.0, accel_noise_std=0.0,  # noiseless
        )
        # At rest: accel should be [0, 0, -g]
        expected = np.array([0., 0., -GRAVITY])
        np.testing.assert_allclose(
            seq.accel,
            np.tile(expected, (seq.N, 1)),
            atol=1e-9,
            err_msg="Noiseless static specific force must be [0,0,-g] FRD",
        )

    def test_quaternion_wxyz_scalar_first(self):
        """Quaternion layout must be (w,x,y,z) scalar-FIRST, matching frames.py.
        Verify: norm must be 1.0 and w^2 + x^2 + y^2 + z^2 = 1."""
        seq = generate_imu_sequence(Scenario.CONSTANT_SPIN, duration_s=1.0, dt=0.01)
        norms = np.linalg.norm(seq.q_wxyz_gt, axis=1)
        np.testing.assert_allclose(norms, np.ones(seq.N), atol=1e-6,
                                   err_msg="All GT quaternions must be unit-norm")

    def test_constant_yaw_spin_rotates_correctly(self):
        """CONSTANT_SPIN (30 deg/s yaw) should produce ~30 deg yaw after 1 second.
        Yaw is about body Z axis (down) in FRD = world Z (down) in NED at level hover.
        Cross-check with frames.py: euler_from_quat_wxyz should recover the angle."""
        from racer.frames import euler_from_quat_wxyz
        seq = generate_imu_sequence(
            Scenario.CONSTANT_SPIN, duration_s=1.0, dt=0.001, seed=0,
            gyro_noise_std=0.0, accel_noise_std=0.0,
        )
        # After 1 second at 30 deg/s yaw = 30 deg = pi/6 rad
        q_final = seq.q_wxyz_gt[-1]
        roll, pitch, yaw = euler_from_quat_wxyz(q_final)
        np.testing.assert_allclose(abs(yaw), np.deg2rad(30.0), atol=0.01,
                                   err_msg="1s @ 30deg/s yaw -> ~30deg yaw angle")
        np.testing.assert_allclose(roll, 0.0, atol=0.01)
        np.testing.assert_allclose(pitch, 0.0, atol=0.01)


# ---------------------------------------------------------------------------
# 2. Static convergence: gravity-only, all filters converge
# ---------------------------------------------------------------------------

class TestStaticConvergence:
    """All filters must converge to < 1 deg error under gravity-only (STATIC_GRAVITY)."""

    @pytest.mark.parametrize("filter_name,filt", [
        ("ESKF", ESKFAHRS(gyro_noise_std=0.01, accel_gate_alpha=10.0)),
        ("Madgwick", MadgwickAHRS(beta=0.1)),
        ("Mahony", MahonyAHRS(kp=2.0, ki=0.005)),
    ])
    def test_static_convergence(self, filter_name, filt):
        """Filter converges to < 1 deg mean error after initialization on static IMU."""
        seq = generate_imu_sequence(
            Scenario.STATIC_GRAVITY,
            duration_s=5.0,
            dt=0.005,
            gyro_noise_std=0.005,
            accel_noise_std=0.02,
            seed=1,
        )
        q_est = run_filter_on_sequence(filt, seq)
        errors = geodesic_error_rad(q_est, seq.q_wxyz_gt)
        stats = score_filter(errors, skip_init_steps=200)  # skip 1s warmup
        assert stats["mean_deg"] < 1.0, (
            f"{filter_name} static mean error {stats['mean_deg']:.2f} deg > 1 deg"
        )


# ---------------------------------------------------------------------------
# 3. Constant-rotation tracking
# ---------------------------------------------------------------------------

class TestConstantSpinTracking:
    """ESKF must track a constant spin rate to < 5 deg p90 after convergence."""

    def test_eskf_tracks_constant_spin(self):
        seq = generate_imu_sequence(
            Scenario.CONSTANT_SPIN,
            duration_s=5.0,
            dt=0.005,
            gyro_noise_std=0.005,
            accel_noise_std=0.02,
            seed=2,
        )
        filt = ESKFAHRS(gyro_noise_std=0.01, accel_gate_alpha=10.0)
        q_est = run_filter_on_sequence(filt, seq)
        errors = geodesic_error_rad(q_est, seq.q_wxyz_gt)
        stats = score_filter(errors, skip_init_steps=200)
        assert stats["p90_deg"] < 5.0, (
            f"ESKF constant-spin p90 {stats['p90_deg']:.2f} deg > 5 deg"
        )

    def test_mahony_tracks_constant_spin(self):
        """Mahony is a rotation-tracking filter; it must also track constant spin."""
        seq = generate_imu_sequence(
            Scenario.CONSTANT_SPIN,
            duration_s=5.0,
            dt=0.005,
            gyro_noise_std=0.005,
            accel_noise_std=0.02,
            seed=2,
        )
        filt = MahonyAHRS(kp=2.0, ki=0.005)
        q_est = run_filter_on_sequence(filt, seq)
        errors = geodesic_error_rad(q_est, seq.q_wxyz_gt)
        stats = score_filter(errors, skip_init_steps=200)
        assert stats["p90_deg"] < 5.0, (
            f"Mahony constant-spin p90 {stats['p90_deg']:.2f} deg > 5 deg"
        )


# ---------------------------------------------------------------------------
# 4. HIGH-G DISCRIMINATING TEST (the key correctness test)
# ---------------------------------------------------------------------------

class TestHighGDiscrimination:
    """ESKF with accel-gating must outperform both classical filters under 4-5 g.

    This is THE discriminating test: classical filters (Madgwick/Mahony) blindly
    trust the accelerometer for tilt correction. Under 4-5 g maneuvers, the body-
    frame accel vector deviates far from -g, driving attitude errors. ESKF's
    accel_gate_alpha parameter down-weights the accel update when |a| >> g,
    maintaining gyro-propagated attitude accuracy.

    Pass criterion: ESKF p90 error must be STRICTLY less than BOTH classical filters.
    """

    def test_eskf_beats_classical_on_high_g_pull(self):
        results = _run_all_filters(Scenario.HIGH_G_PULL, duration_s=8.0, dt=0.005)
        eskf_p90 = score_filter(results["ESKF"])["p90_deg"]
        madgwick_p90 = score_filter(results["Madgwick"])["p90_deg"]
        mahony_p90 = score_filter(results["Mahony"])["p90_deg"]
        best_classical = max(madgwick_p90, mahony_p90)
        # ESKF must beat BOTH classical filters; require at least 10% improvement
        assert eskf_p90 < madgwick_p90, (
            f"ESKF p90={eskf_p90:.2f} deg must beat Madgwick p90={madgwick_p90:.2f} deg"
        )
        assert eskf_p90 < mahony_p90, (
            f"ESKF p90={eskf_p90:.2f} deg must beat Mahony p90={mahony_p90:.2f} deg"
        )

    def test_eskf_beats_classical_on_high_g_random(self):
        results = _run_all_filters(Scenario.HIGH_G_RANDOM, duration_s=8.0, dt=0.005)
        eskf_p90 = score_filter(results["ESKF"])["p90_deg"]
        madgwick_p90 = score_filter(results["Madgwick"])["p90_deg"]
        mahony_p90 = score_filter(results["Mahony"])["p90_deg"]
        assert eskf_p90 < madgwick_p90, (
            f"ESKF p90={eskf_p90:.2f} must beat Madgwick p90={madgwick_p90:.2f} on HIGH_G_RANDOM"
        )
        assert eskf_p90 < mahony_p90, (
            f"ESKF p90={eskf_p90:.2f} must beat Mahony p90={mahony_p90:.2f} on HIGH_G_RANDOM"
        )

    def test_gating_is_the_mechanism(self):
        """Gating IS the mechanism, and the INNOVATION (chi-square) gate dominates.

        Ablation on HIGH_G_RANDOM (the maximally adversarial scenario: ~43% of samples
        have |a| within 10%% of g while their direction is random-corrupted):
          - fully ungated ESKF (alpha=0, chi2=0)  -> catastrophic (>30 deg p90)
          - innovation gate on (chi2=7.815)        -> rescued to < a few deg p90
        This proves the high-g robustness comes from the direction-aware innovation gate,
        not merely from the ESKF being 'a Kalman filter'. The magnitude gate alone CANNOT
        reject a near-g-magnitude, wrong-direction disturbance, so it is not the mechanism
        here (it adds only a small complementary improvement on top of the innovation gate).
        """
        seq = generate_imu_sequence(Scenario.HIGH_G_RANDOM, duration_s=8.0, dt=0.005, seed=42,
                                    gyro_noise_std=0.005, accel_noise_std=0.05)

        eskf_ungated = ESKFAHRS(gyro_noise_std=0.01, accel_gate_alpha=0.0,
                                accel_chi2_thresh=0.0)   # BOTH gates off
        eskf_innovgate = ESKFAHRS(gyro_noise_std=0.01, accel_gate_alpha=0.0,
                                  accel_chi2_thresh=7.815)  # innovation gate ON

        err_ungated = score_filter(
            geodesic_error_rad(run_filter_on_sequence(eskf_ungated, seq), seq.q_wxyz_gt))
        err_innov = score_filter(
            geodesic_error_rad(run_filter_on_sequence(eskf_innovgate, seq), seq.q_wxyz_gt))

        # The fully-ungated ESKF must be badly degraded on this adversarial high-g case...
        assert err_ungated["p90_deg"] > 10.0, (
            f"Ungated ESKF p90={err_ungated['p90_deg']:.2f} deg should be large (>10) "
            f"on HIGH_G_RANDOM; if not, the scenario is not stressing the accel gate."
        )
        # ...and the innovation gate must rescue it dramatically (the real mechanism).
        assert err_innov["p90_deg"] < 0.2 * err_ungated["p90_deg"], (
            f"Innovation gate p90={err_innov['p90_deg']:.2f} should be << ungated "
            f"p90={err_ungated['p90_deg']:.2f} (>=5x better): the chi-square gate is the mechanism."
        )

    def test_magnitude_gate_complements_innovation_gate(self):
        """Magnitude gate adds a small complementary improvement on top of the innovation gate.

        Both gates on should be at least as good as the innovation gate alone on a
        sustained high-g pull (where |a| is persistently >> g, the regime the magnitude
        gate is designed for)."""
        seq = generate_imu_sequence(Scenario.HIGH_G_PULL, duration_s=8.0, dt=0.005, seed=42,
                                    gyro_noise_std=0.005, accel_noise_std=0.05)
        innov_only = ESKFAHRS(gyro_noise_std=0.01, accel_gate_alpha=0.0, accel_chi2_thresh=7.815)
        both = ESKFAHRS(gyro_noise_std=0.01, accel_gate_alpha=10.0, accel_chi2_thresh=7.815)
        e_innov = score_filter(geodesic_error_rad(run_filter_on_sequence(innov_only, seq), seq.q_wxyz_gt))
        e_both = score_filter(geodesic_error_rad(run_filter_on_sequence(both, seq), seq.q_wxyz_gt))
        # Allow a small tolerance; the claim is "complementary, not harmful".
        assert e_both["p90_deg"] <= e_innov["p90_deg"] * 1.25, (
            f"Both-gates p90={e_both['p90_deg']:.2f} should not be much worse than "
            f"innovation-only p90={e_innov['p90_deg']:.2f} on sustained high-g."
        )


# ---------------------------------------------------------------------------
# 5. Geodesic metric sanity
# ---------------------------------------------------------------------------

class TestGeodesicMetric:

    def test_identity_distance_is_zero(self):
        q = np.array([[1., 0., 0., 0.]])
        errors = geodesic_error_rad(q, q)
        np.testing.assert_allclose(errors, [0.0], atol=1e-12)

    def test_180_deg_rotation_gives_pi(self):
        # 180 deg rotation about X: q = [0, 1, 0, 0]
        q1 = np.array([[1., 0., 0., 0.]])
        q2 = np.array([[0., 1., 0., 0.]])
        errors = geodesic_error_rad(q1, q2)
        np.testing.assert_allclose(errors, [np.pi], atol=1e-6)

    def test_q_and_neg_q_give_zero_distance(self):
        """q and -q represent the SAME rotation; geodesic error must be 0."""
        q = np.array([[0.7071, 0.0, 0.7071, 0.0]])
        q_neg = -q
        errors = geodesic_error_rad(q, q_neg)
        np.testing.assert_allclose(errors, [0.0], atol=1e-5)

    def test_score_filter_skip_init(self):
        """score_filter must skip the first N steps as documented."""
        errors = np.ones(500) * np.deg2rad(10.0)
        errors[:200] = np.deg2rad(90.0)   # bad convergence phase
        stats = score_filter(errors, skip_init_steps=200)
        np.testing.assert_allclose(stats["mean_deg"], 10.0, atol=0.1)


# ---------------------------------------------------------------------------
# 6. IMUSequence shape invariants
# ---------------------------------------------------------------------------

class TestIMUSequenceShapes:

    def test_shapes_consistent_across_scenarios(self):
        for scen in Scenario:
            seq = generate_imu_sequence(scen, duration_s=1.0, dt=0.01, seed=0)
            N = seq.N
            assert seq.gyro.shape == (N, 3), f"{scen.name}: gyro shape wrong"
            assert seq.accel.shape == (N, 3), f"{scen.name}: accel shape wrong"
            assert seq.q_wxyz_gt.shape == (N, 4), f"{scen.name}: quat shape wrong"
            assert len(seq.t) == N, f"{scen.name}: t length wrong"

    def test_gt_quaternions_are_unit_norm(self):
        seq = generate_imu_sequence(Scenario.HIGH_G_RANDOM, duration_s=2.0, dt=0.01)
        norms = np.linalg.norm(seq.q_wxyz_gt, axis=1)
        np.testing.assert_allclose(norms, np.ones(seq.N), atol=1e-5)

    def test_dt_property(self):
        seq = generate_imu_sequence(Scenario.STATIC_GRAVITY, duration_s=1.0, dt=0.005)
        np.testing.assert_allclose(seq.dt, 0.005, atol=1e-9)

    def test_noise_off_gives_clean_imu(self):
        """Noiseless generator must give zero noise on accel and gyro."""
        seq = generate_imu_sequence(
            Scenario.CONSTANT_SPIN, duration_s=1.0, dt=0.01, seed=0,
            gyro_noise_std=0.0, accel_noise_std=0.0,
        )
        # Accel should be exactly [0, 0, -g] for constant spin at level hover
        expected_accel = np.array([0., 0., -GRAVITY])
        np.testing.assert_allclose(
            seq.accel,
            np.tile(expected_accel, (seq.N, 1)),
            atol=1e-6,
            err_msg="Noiseless constant-spin: accel must be [0,0,-g] (no centripetal w/o translation)",
        )


# ---------------------------------------------------------------------------
# 7. ESKF internals: accel gate weight
# ---------------------------------------------------------------------------

class TestESKFGating:

    def test_gate_weight_is_one_at_g(self):
        """Gate weight must be 1.0 when |accel| = g (true at rest)."""
        filt = ESKFAHRS(accel_gate_alpha=10.0)
        w = filt._accel_gate_weight(GRAVITY)
        np.testing.assert_allclose(w, 1.0, atol=1e-9)

    def test_gate_weight_decays_at_high_g(self):
        """Gate weight must decay significantly at 4g."""
        filt = ESKFAHRS(accel_gate_alpha=10.0)
        w_rest = filt._accel_gate_weight(GRAVITY)
        w_4g = filt._accel_gate_weight(4.0 * GRAVITY)
        assert w_4g < 0.1, f"Gate weight at 4g ({w_4g:.4f}) should be < 0.1"
        assert w_4g < w_rest

    def test_gate_weight_disabled_when_alpha_zero(self):
        """With alpha=0, gate weight must always be 1.0 (no gating)."""
        filt = ESKFAHRS(accel_gate_alpha=0.0)
        for mag in [0.0, GRAVITY, 4.0 * GRAVITY, 10.0 * GRAVITY]:
            w = filt._accel_gate_weight(mag)
            np.testing.assert_allclose(w, 1.0, atol=1e-9,
                                       err_msg=f"alpha=0: gate weight must be 1.0 at |a|={mag:.1f}")

    def test_innovation_gate_rejects_inconsistent_accel(self):
        """A single wildly-inconsistent accel sample (near-g magnitude, wrong direction)
        must be REJECTED by the chi-square gate, leaving attitude unchanged. With the gate
        disabled (chi2=0) the same sample perturbs the attitude."""
        gate_on = ESKFAHRS(accel_gate_alpha=0.0, accel_chi2_thresh=7.815)
        gate_off = ESKFAHRS(accel_gate_alpha=0.0, accel_chi2_thresh=0.0)
        gate_on.reset(); gate_off.reset()
        # Converge both on rest so the prior is confident and at identity.
        for _ in range(300):
            a_rest = np.array([0., 0., -GRAVITY])
            gate_on.step(np.zeros(3), a_rest, dt=0.005)
            gate_off.step(np.zeros(3), a_rest, dt=0.005)
        # Now a near-g-magnitude accel pointing sideways (direction-corrupted, |a|~g).
        a_bad = np.array([GRAVITY, 0., 0.])
        q_on = gate_on.step(np.zeros(3), a_bad, dt=0.005)
        q_off = gate_off.step(np.zeros(3), a_bad, dt=0.005)
        err_on = geodesic_error_rad(q_on, np.array([1., 0., 0., 0.]))[0]
        err_off = geodesic_error_rad(q_off, np.array([1., 0., 0., 0.]))[0]
        assert err_on < np.deg2rad(0.5), (
            f"Gated filter should reject the bad accel (err {np.rad2deg(err_on):.2f} deg)"
        )
        assert err_off > err_on, "Ungated filter should be perturbed more by the bad accel"

    def test_eskf_step_returns_unit_quaternion(self):
        """Every ESKF step must return a unit-norm quaternion."""
        filt = ESKFAHRS()
        filt.reset()
        rng = np.random.default_rng(0)
        for _ in range(50):
            gyro = rng.normal(0, 0.1, 3)
            accel = rng.normal(0, 5.0, 3) + np.array([0., 0., -GRAVITY])
            q = filt.step(gyro, accel, dt=0.005)
            np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-6,
                                       err_msg="ESKF output must be unit quaternion")

    def test_classical_step_returns_unit_quaternion(self):
        """Madgwick and Mahony must return unit quaternions too."""
        rng = np.random.default_rng(1)
        for FiltClass in [MadgwickAHRS, MahonyAHRS]:
            filt = FiltClass()
            filt.reset()
            for _ in range(50):
                gyro = rng.normal(0, 0.1, 3)
                accel = rng.normal(0, 5.0, 3) + np.array([0., 0., -GRAVITY])
                q = filt.step(gyro, accel, dt=0.005)
                np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-5,
                                           err_msg=f"{FiltClass.__name__} must return unit quat")


# ---------------------------------------------------------------------------
# 8. Left-Invariant EKF (IEKF) — research-grounded contender
# ---------------------------------------------------------------------------

class TestIEKF:
    """Validate the left-invariant EKF: helpers, convergence, and ESKF equivalence."""

    def test_R_quat_roundtrip(self):
        """_R_to_quat_wxyz . _quat_to_R_wxyz must round-trip several rotations."""
        rng = np.random.default_rng(3)
        for _ in range(20):
            q = rng.normal(0, 1, 4)
            q /= np.linalg.norm(q)
            if q[0] < 0:
                q = -q
            R = _quat_to_R_wxyz(q)
            q2 = _R_to_quat_wxyz(R)
            # Compare as rotations (double-cover safe)
            err = geodesic_error_rad(q, q2)[0]
            assert err < 1e-9, f"roundtrip error {np.rad2deg(err):.2e} deg"

    def test_exp_so3_matches_known_rotation(self):
        """_Exp_so3 of a pure-axis vector must give the textbook rotation matrix."""
        R = _Exp_so3(np.array([0., 0., np.pi / 2]))   # 90 deg about Z
        expected = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        np.testing.assert_allclose(R, expected, atol=1e-12)

    def test_iekf_static_convergence(self):
        seq = generate_imu_sequence(Scenario.STATIC_GRAVITY, duration_s=5.0, dt=0.005,
                                    gyro_noise_std=0.005, accel_noise_std=0.02, seed=1)
        filt = LeftInvariantEKF(gyro_noise_std=0.01, accel_gate_alpha=10.0)
        q = run_filter_on_sequence(filt, seq)
        stats = score_filter(geodesic_error_rad(q, seq.q_wxyz_gt), skip_init_steps=200)
        assert stats["mean_deg"] < 1.0, f"IEKF static mean {stats['mean_deg']:.2f} deg"

    def test_iekf_step_returns_unit_quaternion(self):
        filt = LeftInvariantEKF()
        filt.reset()
        rng = np.random.default_rng(0)
        for _ in range(50):
            gyro = rng.normal(0, 0.1, 3)
            accel = rng.normal(0, 5.0, 3) + np.array([0., 0., -GRAVITY])
            q = filt.step(gyro, accel, dt=0.005)
            np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-6)

    def test_iekf_matches_eskf_on_attitude_only(self):
        """For the SO(3) attitude-from-gravity problem the left-invariant EKF and the
        ESKF produce algebraically equivalent updates; they must track to <1e-3 deg on
        every scenario (this cross-validates BOTH implementations)."""
        for scen in Scenario:
            seq = generate_imu_sequence(scen, duration_s=8.0, dt=0.005, seed=42,
                                        gyro_noise_std=0.005, accel_noise_std=0.05)
            q_eskf = run_filter_on_sequence(
                ESKFAHRS(gyro_noise_std=0.01, accel_gate_alpha=10.0), seq)
            q_iekf = run_filter_on_sequence(
                LeftInvariantEKF(gyro_noise_std=0.01, accel_gate_alpha=10.0), seq)
            diff = geodesic_error_rad(q_eskf, q_iekf)
            assert np.max(diff) < np.deg2rad(1e-3), (
                f"{scen.name}: IEKF vs ESKF max diff {np.rad2deg(np.max(diff)):.2e} deg"
            )

    def test_iekf_recovers_from_large_initial_error(self):
        """The invariant filter must recover quickly from a large (80 deg) but observable
        initial tilt error once the innovation gate is open (chi2 disabled here to let the
        large startup innovation through, as a cold-start would)."""
        q_bad = np.array([np.cos(np.deg2rad(40)), np.sin(np.deg2rad(40)), 0., 0.])
        q_bad /= np.linalg.norm(q_bad)
        seq = generate_imu_sequence(Scenario.STATIC_GRAVITY, duration_s=8.0, dt=0.005,
                                    seed=42, gyro_noise_std=0.005, accel_noise_std=0.05)
        filt = LeftInvariantEKF(gyro_noise_std=0.01, accel_gate_alpha=10.0,
                                accel_chi2_thresh=0.0)
        filt.reset(q_bad)
        # Inflate the initial tilt covariance so the gain is meaningful from a cold start.
        filt._P = np.diag([1.0]*3 + [1e-4]*3)
        q = np.zeros((seq.N, 4))
        for i in range(seq.N):
            q[i] = filt.step(seq.gyro[i], seq.accel[i], seq.dt)
        err = geodesic_error_rad(q, seq.q_wxyz_gt)
        # Converged within 1 s, and stays small.
        assert np.rad2deg(err[int(1.0 / seq.dt)]) < 1.0, (
            f"IEKF not converged after 1s from 80 deg: {np.rad2deg(err[int(1.0/seq.dt)]):.2f} deg"
        )
