"""Unit tests for the AHRS adapter shim (src/racer/ahrs/ahrs_adapter.py).

Run with the main-checkout venv:
    C:/Users/Fengy/Downloads/Projects/Anduril/.venv/Scripts/python.exe -m pytest tests/test_ahrs_adapter.py -v

The adapter wraps ESKFAHRS into the EXACT attitude-source contract the navigator's future
``use_ahrs`` seam consumes (replaces navigator.py:416 R_wb and state_estimator.py:213-216
euler/rates). These tests pin:

1. Output-convention parity with the navigator's downstream:
   - ``R_wb`` matches the OUTPUT convention of frames.R_world_from_odo_quat_wxyz
     (true FRD->NED, NOT conjugation-doubled).
   - ``euler_rpy`` is byte-identical to frames.euler_from_quat_wxyz of the same quat
     (what make_nav_state stores).
2. Attitude tracking: reproduces a known attitude from a synthetic IMU sequence (imu_gen).
3. The body-rate FOOTGUN: ``body_rate`` is the bias-corrected gyro (omega = R_i2b @ w),
   NOT the raw gyro and NOT a quaternion finite-difference.
4. reset()/seed() re-seed cleanly; master-clock dt<=0 is a no-op.
5. The full existing AHRS suite stays green (verified separately by test_ahrs_bench.py).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from racer.ahrs.ahrs_adapter import AHRSAttitudeSource
from racer.ahrs.eskf import ESKFAHRS, _quat_to_R_wxyz
from racer.ahrs.imu_gen import generate_imu_sequence, Scenario
from racer.ahrs.metrics import geodesic_error_rad, score_filter
from racer.frames import (
    R_world_from_odo_quat_wxyz,
    euler_from_quat_wxyz,
    ODO_QUAT_TRUE_CONJ_WXYZ,
)


def _quat_wxyz(roll, pitch, yaw):
    """True body->world quaternion (w,x,y,z) from aerospace Euler (ZYX)."""
    x, y, z, w = Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_quat()
    return np.array([w, x, y, z])


# ---------------------------------------------------------------------------
# 1. Output-convention parity with the navigator's consumption sites
# ---------------------------------------------------------------------------

class TestConventionParity:
    """Pin the adapter's outputs to the exact conventions navigator.py / state_estimator.py expect."""

    def test_R_wb_is_true_frd_to_ned(self):
        """``R_wb`` must equal the OUTPUT of R_world_from_odo_quat_wxyz for the SAME true attitude.

        The wire helper conjugates a RAW ODOMETRY quat; the AHRS emits the TRUE quat directly.
        So: build a true attitude, form the corresponding RAW (conjugated) wire quat, and check
        that adapter.R_wb (from the true quat) == R_world_from_odo_quat_wxyz(raw wire quat).
        This guards against accidentally double-conjugating (the biggest wiring footgun).
        """
        q_true = _quat_wxyz(roll=0.30, pitch=-0.20, yaw=0.50)
        # The wire would carry the R_y(pi)-conjugated quat; the helper un-conjugates it.
        q_raw_wire = q_true * ODO_QUAT_TRUE_CONJ_WXYZ   # involutory: conj(conj)=identity
        R_via_wire = R_world_from_odo_quat_wxyz(q_raw_wire)

        src = AHRSAttitudeSource()
        src.seed(q_true)
        np.testing.assert_allclose(src.R_wb, R_via_wire, atol=1e-12,
                                   err_msg="adapter R_wb must match the TRUE FRD->NED output of "
                                           "R_world_from_odo_quat_wxyz (no double conjugation)")

    def test_R_wb_matches_quat(self):
        """R_wb is exactly the rotation matrix of q_wxyz (same _quat_to_R_wxyz as the ESKF)."""
        src = AHRSAttitudeSource()
        src.seed(_quat_wxyz(0.1, 0.2, -0.3))
        np.testing.assert_allclose(src.R_wb, _quat_to_R_wxyz(src.q_wxyz), atol=1e-15)

    def test_euler_matches_frames_decoder(self):
        """euler_rpy must be byte-identical to frames.euler_from_quat_wxyz of the same quat
        (the function make_nav_state's downstream / build_obs share)."""
        q = _quat_wxyz(0.25, -0.15, 0.7)
        src = AHRSAttitudeSource()
        src.seed(q)
        r, p, y = src.euler_rpy
        r2, p2, y2 = euler_from_quat_wxyz(src.q_wxyz)
        assert (r, p, y) == (r2, p2, y2)

    def test_euler_roundtrips_known_attitude(self):
        """A seeded known (roll,pitch,yaw) decodes back to itself."""
        roll, pitch, yaw = 0.2, -0.35, 0.9
        src = AHRSAttitudeSource()
        src.seed(_quat_wxyz(roll, pitch, yaw))
        r, p, y = src.euler_rpy
        np.testing.assert_allclose([r, p, y], [roll, pitch, yaw], atol=1e-9)

    def test_R_wb_is_proper_rotation(self):
        """R_wb is orthonormal with det +1 (a valid SO(3) element)."""
        src = AHRSAttitudeSource()
        src.seed(_quat_wxyz(0.4, 0.3, -0.6))
        R = src.R_wb
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 2. Attitude tracking from a synthetic IMU sequence
# ---------------------------------------------------------------------------

class TestTracking:
    """The adapter reproduces a known attitude from a synthetic IMU stream (imu_gen)."""

    def _run(self, scenario, duration_s=4.0, dt=0.005, seed=7, **eskf_kw):
        seq = generate_imu_sequence(scenario, duration_s=duration_s, dt=dt, seed=seed,
                                    gyro_noise_std=0.005, accel_noise_std=0.05)
        src = AHRSAttitudeSource(eskf=ESKFAHRS(gyro_noise_std=0.01, accel_gate_alpha=10.0,
                                               **eskf_kw))
        src.seed(seq.q_wxyz_gt[0])
        q_est = np.zeros((seq.N, 4))
        for i in range(seq.N):
            q_est[i] = src.ingest(seq.accel[i], seq.gyro[i], dt)
        err = geodesic_error_rad(q_est, seq.q_wxyz_gt)
        return seq, q_est, err

    def test_static_gravity_converges(self):
        """Hover: the adapter holds level attitude tightly (sub-degree steady state)."""
        _, _, err = self._run(Scenario.STATIC_GRAVITY)
        stats = score_filter(err)
        assert stats["p90_deg"] < 1.0, f"static p90 {stats['p90_deg']:.3f} deg too high"

    def test_constant_spin_tracks(self):
        """Constant yaw rate: gyro integration tracks the spinning attitude (few-degree band)."""
        _, _, err = self._run(Scenario.CONSTANT_SPIN)
        stats = score_filter(err)
        assert stats["p90_deg"] < 5.0, f"spin p90 {stats['p90_deg']:.3f} deg too high"

    def test_rolling_maneuver_tracks(self):
        """Mild S-curve with ~0.5 g lateral: still well-tracked."""
        _, _, err = self._run(Scenario.ROLLING_MANEUVER)
        stats = score_filter(err)
        # ~0.5 g lateral perturbs the accel tilt update; a few-degree p90 is expected ESKF
        # performance (the transparent-wrapper test pins exactness vs the bare filter).
        assert stats["p90_deg"] < 8.0, f"rolling p90 {stats['p90_deg']:.3f} deg too high"

    def test_adapter_matches_bare_eskf(self):
        """The adapter must be a transparent wrapper: identical quaternion stream to a bare
        ESKFAHRS driven with the same seed + samples (no behaviour change)."""
        seq = generate_imu_sequence(Scenario.ROLLING_MANEUVER, duration_s=3.0, dt=0.005,
                                    seed=11, gyro_noise_std=0.005, accel_noise_std=0.05)
        dt = seq.dt

        bare = ESKFAHRS(gyro_noise_std=0.01, accel_gate_alpha=10.0)
        bare.reset(seq.q_wxyz_gt[0])
        q_bare = np.array([bare.step(seq.gyro[i], seq.accel[i], dt) for i in range(seq.N)])

        src = AHRSAttitudeSource(eskf=ESKFAHRS(gyro_noise_std=0.01, accel_gate_alpha=10.0))
        src.seed(seq.q_wxyz_gt[0])
        q_adp = np.array([src.ingest(seq.accel[i], seq.gyro[i], dt) for i in range(seq.N)])

        np.testing.assert_allclose(q_adp, q_bare, atol=1e-15,
                                   err_msg="adapter must not alter ESKF behaviour")


# ---------------------------------------------------------------------------
# 3. Body-rate footgun: omega = R_i2b @ w = gyro - bias (NOT raw gyro / NOT quat-FD)
# ---------------------------------------------------------------------------

class TestBodyRateConvention:
    """Guard the documented footgun: body_rate is the bias-corrected gyro, not the raw gyro."""

    def test_body_rate_equals_gyro_minus_bias(self):
        """With a deliberately seeded/learned bias, body_rate == gyro - eskf.gyro_bias,
        snapshotted at the step that produced the current attitude."""
        src = AHRSAttitudeSource()
        src.seed(None)
        gyro = np.array([0.10, -0.05, 0.20])
        accel = np.array([0.0, 0.0, -9.80665])
        src.ingest(accel, gyro, dt=0.005)
        expected = gyro - src.gyro_bias
        np.testing.assert_allclose(src.body_rate, expected, atol=1e-12)

    def test_body_rate_is_not_raw_gyro_under_bias(self):
        """Inject a non-zero learned bias; body_rate must differ from the raw gyro by exactly it."""
        src = AHRSAttitudeSource()
        src.seed(None)
        # Force a known bias estimate into the wrapped filter, then ingest one sample.
        src.eskf._b_g = np.array([0.02, -0.01, 0.03])
        gyro = np.array([0.30, 0.10, -0.20])
        src.ingest(np.array([0.0, 0.0, -9.80665]), gyro, dt=0.005)
        # body_rate snapshot used the bias present BEFORE the step.
        np.testing.assert_allclose(src.body_rate, gyro - np.array([0.02, -0.01, 0.03]),
                                   atol=1e-12)
        assert not np.allclose(src.body_rate, gyro), "body_rate must NOT be the raw gyro"

    def test_body_rate_reconstructs_attitude_derivative(self):
        """omega = R_i2b @ w means R_dot = R_wb @ skew(omega). On a bias-free, noise-free
        constant-spin step the body rate should reconstruct the attitude increment:
        R_next ~ R_cur @ exp(skew(omega) * dt)."""
        src = AHRSAttitudeSource(eskf=ESKFAHRS(gyro_noise_std=0.01, accel_gate_alpha=0.0))
        src.seed(None)
        dt = 0.005
        omega_true = np.array([0.0, 0.0, np.deg2rad(30.0)])   # pure yaw rate, FRD
        accel = np.array([0.0, 0.0, -9.80665])                 # level: no tilt correction pull
        R_before = src.R_wb.copy()
        src.ingest(accel, omega_true, dt)
        w = src.body_rate
        # Predicted next rotation from the reported body rate.
        from scipy.spatial.transform import Rotation as _R
        R_pred = R_before @ _R.from_rotvec(w * dt).as_matrix()
        np.testing.assert_allclose(src.R_wb, R_pred, atol=1e-6,
                                   err_msg="body_rate must be the omega that propagated R_wb")

    def test_body_rate_zero_before_first_step(self):
        """Before any dt>0 ingest the reported body rate is zeros (no stale garbage)."""
        src = AHRSAttitudeSource()
        src.seed(None)
        np.testing.assert_array_equal(src.body_rate, np.zeros(3))


# ---------------------------------------------------------------------------
# 4. Lifecycle: seed / reset / master-clock dt<=0
# ---------------------------------------------------------------------------

class TestLifecycle:

    def test_seed_sets_attitude_and_flag(self):
        src = AHRSAttitudeSource()
        assert not src.seeded
        q = _quat_wxyz(0.1, 0.2, 0.3)
        src.seed(q)
        assert src.seeded
        np.testing.assert_allclose(src.q_wxyz, q / np.linalg.norm(q), atol=1e-12)

    def test_auto_seed_on_first_ingest(self):
        """Ingesting before an explicit seed auto-seeds GRAVITY-ALIGNED from the first accel sample
        (NOT identity) -- the 2026-06-29 cold-start fix. A level first accel -> level seed; a tilted
        first accel (the VQ2 in-gate spawn) -> a roll/pitch-levelled seed at ~0deg error, not 18deg."""
        src = AHRSAttitudeSource()
        assert not src.seeded
        src.ingest(np.array([0.0, 0.0, -9.80665]), np.zeros(3), dt=0.005)
        assert src.seeded
        np.testing.assert_allclose(src.q_wxyz, [1.0, 0.0, 0.0, 0.0], atol=1e-9)  # level -> level

        # A TILTED first accel (drone spawned pitched ~18deg in the gate): the auto-seed must
        # gravity-align so the ESKF starts at ~0deg error, not ~18deg (the old identity seed).
        pitch0 = np.deg2rad(18.0)
        R = _quat_to_R_wxyz(_quat_wxyz(roll=0.0, pitch=pitch0, yaw=0.0))
        sf_body = R.T @ np.array([0.0, 0.0, -9.80665])      # specific force at rest, tilted
        src2 = AHRSAttitudeSource()
        src2.ingest(sf_body, np.zeros(3), dt=0.005)
        r, p, _ = euler_from_quat_wxyz(src2.q_wxyz)
        np.testing.assert_allclose([r, p], [0.0, pitch0], atol=1e-3)   # seeded at the true tilt

    def test_reset_reseeds_cleanly(self):
        """After running, reset() returns to a fresh seeded state (attitude + body rate + bias)."""
        seq = generate_imu_sequence(Scenario.CONSTANT_SPIN, duration_s=2.0, dt=0.005, seed=3)
        src = AHRSAttitudeSource()
        src.seed(seq.q_wxyz_gt[0])
        for i in range(seq.N):
            src.ingest(seq.accel[i], seq.gyro[i], seq.dt)
        # Now re-seed to identity; everything resets.
        src.reset(None)
        np.testing.assert_allclose(src.q_wxyz, [1.0, 0.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_array_equal(src.body_rate, np.zeros(3))
        np.testing.assert_allclose(src.gyro_bias, np.zeros(3), atol=1e-12)
        assert src.seeded

    def test_dt_nonpositive_is_noop(self):
        """dt<=0 (control loop faster than IMU) re-packages the current estimate, no integration."""
        src = AHRSAttitudeSource()
        src.seed(_quat_wxyz(0.2, -0.1, 0.4))
        q_before = src.q_wxyz.copy()
        out = src.ingest(np.array([0.0, 0.0, -9.80665]), np.array([1.0, 2.0, 3.0]), dt=0.0)
        np.testing.assert_allclose(out, q_before, atol=1e-15)
        np.testing.assert_allclose(src.q_wxyz, q_before, atol=1e-15)
        np.testing.assert_array_equal(src.body_rate, np.zeros(3))   # no dt>0 step yet

    def test_level_seed_from_accel(self):
        """level_seed_from_accel produces a roll/pitch-levelled seed; a tilted accel gives tilt."""
        # Level rest accel -> identity (level), yaw 0.
        q_level = AHRSAttitudeSource.level_seed_from_accel(np.array([0.0, 0.0, -9.80665]))
        np.testing.assert_allclose(q_level, [1.0, 0.0, 0.0, 0.0], atol=1e-9)
        # Degenerate -> identity.
        q_deg = AHRSAttitudeSource.level_seed_from_accel(np.zeros(3))
        np.testing.assert_allclose(q_deg, [1.0, 0.0, 0.0, 0.0], atol=1e-12)
        # A pitched-back rest attitude: specific force tilts. Seed should recover roll/pitch.
        q_true = _quat_wxyz(roll=0.0, pitch=0.25, yaw=0.0)
        R = _quat_to_R_wxyz(q_true)
        sf_body = R.T @ np.array([0.0, 0.0, -9.80665])   # specific force at rest in body
        q_seed = AHRSAttitudeSource.level_seed_from_accel(sf_body)
        r, p, y = euler_from_quat_wxyz(q_seed)
        np.testing.assert_allclose([r, p], [0.0, 0.25], atol=1e-6)


# ---------------------------------------------------------------------------
# 5. Uncertainty channel sanity (cold-start gating feeder)
# ---------------------------------------------------------------------------

def test_attitude_uncertainty_decreases_with_fixes():
    """Under a benign (level) stream the accel updates should shrink attitude uncertainty
    from its seeded value -- the channel the wiring step gates the alignment transient on."""
    src = AHRSAttitudeSource()
    src.seed(None)
    u0 = src.attitude_uncertainty_rad
    accel = np.array([0.0, 0.0, -9.80665])
    for _ in range(200):
        src.ingest(accel, np.zeros(3), dt=0.005)
    assert src.attitude_uncertainty_rad <= u0 + 1e-9
    assert np.isfinite(src.attitude_uncertainty_rad)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
