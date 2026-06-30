"""Acceleration-aware accel rejection: the VQ2 A8 climb-and-retreat regression pin.

WHY THIS EXISTS (VQ2 A8 nose-up-and-retreat, ground-truth-confirmed from live telem)
------------------------------------------------------------------------------------
A8 diverged under SUSTAINED LINEAR ACCELERATION. The decisive live signature over a 4 s
gyro-quiet / throttle-on segment: the accel-APPARENT pitch drifted -23.6 deg while the GYRO
integrated only -0.8 deg. A gyro measures rotation directly, so ~23 deg of apparent tilt with
the gyro flat means the accelerometer is contaminated by LINEAR acceleration, not rotation:
sustained thrust grows body-x linear accel to ~+4 m/s^2 while |a| HOLDS ~= g (the specific-force
vector keeps g-MAGNITUDE but tilts ~23 deg in DIRECTION). It therefore sails right past the
MAGNITUDE-only free-fall/high-g band (|a| ~ g throughout). The estimator levels to the tilted
vector -> false pitch -> the controller "corrects" the wrong way -> more accel -> more tilt -> a
divergent loop.

THE FIX UNDER TEST: acceleration-aware accel rejection. The discriminating signal is the
linear-acceleration residual w.r.t. the current attitude estimate,
    a_lin = R_hat @ f_body + g_ned   (== true world kinematic accel, ~0 at equilibrium),
which the magnitude gate is blind to (|a| ~ g says nothing about whether |a_lin| ~ 0). The accel
update's measurement covariance is inflated by 1 + (|a_lin|/scale)^2, gracefully starving the bad
leveling under powered tilted flight while leaving hover/coast (|a_lin| ~ 0) byte-identical.

THIS FILE PINS:
  1. A8 reproduction: OLD/guard-OFF diverges to >=20 deg false pitch on the synthetic
     sustained-accel scenario; NEW/guard-ON holds pitch within a few degrees (gyro-truth).
  2. Inert at genuine equilibrium: at |a_lin| ~ 0 the inflation factor is exactly 1.0.
  3. No false rejection of a REAL rotation: when the body genuinely rotates (gyro != 0) and the
     accel tracks it (|a_lin| ~ 0 throughout), guard-ON tracks the tilt as well as guard-OFF.
  4. Normal-flight byte-identity: with the gate disabled the quaternion stream is bit-identical to
     today (maxdiff 0.0) across the benign scenarios.
  5. ESKF<->IEKF mirror: the IEKF carries the same gate and inflates identically.

Pure numpy/scipy; torch-free. Run with the main-checkout venv:
    .../.venv/Scripts/python.exe -m pytest tests/test_accel_motion_reject.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from racer.ahrs.eskf import ESKFAHRS, GRAVITY, _quat_to_R_wxyz
from racer.ahrs.iekf import LeftInvariantEKF
from racer.ahrs.imu_gen import generate_imu_sequence, Scenario
from racer.ahrs.metrics import geodesic_error_rad
from racer.frames import euler_from_quat_wxyz


G = GRAVITY
DT = 0.005


# ---------------------------------------------------------------------------
# Synthetic A8 scenario: sustained forward accel, gyro ~ 0, |a| held ~ g, the
# specific-force DIRECTION ramped ~23 deg over ~4 s (mimics body-x accel ~+4 m/s^2).
# ---------------------------------------------------------------------------

def _a8_specific_force(n_steps: int, tilt_deg: float = 23.0, mag: float = G) -> np.ndarray:
    """Body-frame specific force for the A8 segment: TRUE attitude is level (gyro ~ 0), but the
    accel vector tilts smoothly from rest [0,0,-g] toward a +tilt about body-Y (a nose-up apparent
    lean) while its MAGNITUDE is held at ``mag`` ~ g. This is exactly the live signature: |a| ~ g
    (so the magnitude band is a no-op) yet the DIRECTION moves ~23 deg with the gyro flat."""
    sf = np.zeros((n_steps, 3))
    for i in range(n_steps):
        frac = (i + 1) / n_steps
        theta = np.deg2rad(tilt_deg) * frac          # ramp 0 -> tilt_deg
        # Rest specific force is [0,0,-g] (up). Tilt it in the x-z plane (about body-Y): a positive
        # apparent nose-up lean tips -g toward +x. |sf| held at `mag`.
        sf[i] = mag * np.array([np.sin(theta), 0.0, -np.cos(theta)])
    return sf


def _run_a8(use_reject: bool, n_settle: int = 300, n_a8: int = 800, p_grown: float = 0.5,
            klass=ESKFAHRS) -> float:
    """Settle level, GROW the prior (sparse vision fixes during the run), then feed the A8 segment.
    Returns the absolute estimated pitch (deg) at the end. True pitch is ~0 throughout (gyro ~ 0).
    Uses the SHIPPED defaults (accel_motion_scale / accel_motion_anchor_thr) so this pins the
    real configuration, not a tuned-for-the-test one."""
    filt = klass(gyro_noise_std=0.01, accel_gate_alpha=10.0, accel_chi2_thresh=7.815,
                 accel_freefall_tol_lo=0.75, accel_freefall_tol_hi=9.0,
                 use_accel_motion_reject=use_reject)
    filt.reset(np.array([1.0, 0.0, 0.0, 0.0]))
    for _ in range(n_settle):
        filt.step(np.zeros(3), np.array([0.0, 0.0, -G]), DT)
    # Grow the attitude prior: between sparse vision yaw/tilt fixes the covariance inflates, so a
    # contaminated accel gets a meaningful Kalman gain (the faithful in-flight condition).
    filt._P = np.diag([p_grown] * 3 + [1e-5] * 3)
    sf = _a8_specific_force(n_a8)
    for i in range(n_a8):
        filt.step(np.zeros(3), sf[i], DT)            # gyro ~ 0 (no real rotation)
    _, pitch, _ = euler_from_quat_wxyz(filt.q_wxyz)
    return float(abs(np.rad2deg(pitch)))


class TestA8Reproduction:
    """The headline pin: the synthetic A8 scenario diverges WITHOUT the gate and is held WITH it."""

    def test_old_path_diverges(self):
        """OLD behaviour (gate OFF): the sustained-accel segment drags the estimate to a large
        false pitch. The magnitude band + chi2 gate (both ON) do NOT catch it -- |a| ~ g and the
        innovation creeps in slowly, staying chi2-consistent."""
        pitch_off = _run_a8(use_reject=False)
        assert pitch_off >= 20.0, (
            f"A8 scenario should drive a large false pitch WITHOUT the gate "
            f"(got {pitch_off:.2f} deg) -- the magnitude/chi2 gates alone do not catch sustained accel"
        )

    def test_new_path_holds_pitch(self):
        """NEW behaviour (gate ON): the accel leveling is starved as |a_lin| grows, so the gyro
        (truthfully flat) holds the estimate near gyro-truth. A brief onset transient (~5-6 deg) leaks
        before the graceful gate fully engages -- a 'few degrees', not the 22 deg open-loop divergence
        (and far short of the inversion the in-the-loop controller would otherwise amplify)."""
        pitch_on = _run_a8(use_reject=True)
        assert pitch_on < 7.0, (
            f"with acceleration-aware rejection the pitch must stay near gyro-truth "
            f"(got {pitch_on:.2f} deg)"
        )

    def test_gate_strictly_improves(self):
        """Direct OLD-vs-NEW comparison on the identical scenario: the gate cuts the false pitch
        by a large margin (>=3x)."""
        pitch_off = _run_a8(use_reject=False)
        pitch_on = _run_a8(use_reject=True)
        assert pitch_on < 0.34 * pitch_off, (
            f"gate must sharply reduce the A8 false pitch (off {pitch_off:.2f} deg, on {pitch_on:.2f} deg)"
        )


class TestInertAtEquilibrium:
    """At genuine equilibrium the gate is a NO-OP (factor exactly 1.0): it must not starve the accel
    correction during normal slow flight (that would let gyro bias drift unbounded)."""

    def test_inflation_factor_unity_at_rest(self):
        """Level rest, |a_lin| ~ 0: the inflation factor is exactly 1.0 (and 1.0 when disabled)."""
        filt = ESKFAHRS(use_accel_motion_reject=True, accel_motion_scale=2.0)
        filt.reset(np.array([1.0, 0.0, 0.0, 0.0]))
        rest = np.array([0.0, 0.0, -G])              # level rest specific force -> a_lin ~ 0
        assert abs(filt._accel_motion_inflation(rest) - 1.0) < 1e-12
        off = ESKFAHRS(use_accel_motion_reject=False)
        off.reset(np.array([1.0, 0.0, 0.0, 0.0]))
        assert off._accel_motion_inflation(rest) == 1.0   # disabled -> always 1.0

    def test_inflation_grows_with_linear_accel(self):
        """A body-x linear accel of ~4 m/s^2 (the A8 magnitude) inflates R by ~ 1+(4/2)^2 = 5x."""
        filt = ESKFAHRS(use_accel_motion_reject=True, accel_motion_scale=2.0)
        filt.reset(np.array([1.0, 0.0, 0.0, 0.0]))
        # True level attitude, body-x world accel +4 m/s^2: f_body = [a_x,0,-g] -> a_lin=[4,0,0].
        sf = np.array([4.0, 0.0, -G])
        factor = filt._accel_motion_inflation(sf)
        assert abs(factor - 5.0) < 1e-9, f"expected ~5x inflation at |a_lin|=4 (got {factor:.4f})"

    def test_equilibrium_stream_byte_identical(self):
        """A long level-hover stream: gate-ON must be BYTE-IDENTICAL to gate-OFF (|a_lin| ~ 0 the
        whole time, so the factor is exactly 1.0 every step)."""
        rng = np.random.default_rng(0)
        on = ESKFAHRS(gyro_noise_std=0.01, accel_gate_alpha=10.0, use_accel_motion_reject=True)
        off = ESKFAHRS(gyro_noise_std=0.01, accel_gate_alpha=10.0, use_accel_motion_reject=False)
        on.reset(np.array([1.0, 0.0, 0.0, 0.0]))
        off.reset(np.array([1.0, 0.0, 0.0, 0.0]))
        for _ in range(500):
            a = np.array([0.0, 0.0, -G])             # exact rest -> a_lin == 0 -> factor == 1.0
            q_on = on.step(np.zeros(3), a, DT)
            q_off = off.step(np.zeros(3), a, DT)
            assert np.array_equal(q_on, q_off)


class TestNoFalseRejectionOnRealRotation:
    """A REAL rotation (gyro != 0, accel tracks it -> |a_lin| ~ 0) must NOT be rejected: the gate
    keys on |a_lin|, not on |a| or on motion per se, so genuine leveling is preserved."""

    def test_real_tilt_tracked_with_gate_on(self):
        """The drone physically pitches up at a constant rate with NO linear accel (pure rotation,
        accel stays the gravity reference). Gate-ON must track the true tilt as well as gate-OFF
        (within a tiny tolerance) -- it must not mistake honest rotation for contamination."""
        target_deg = 20.0
        wy = np.deg2rad(target_deg) / (400 * DT)     # reach 20 deg over 400 steps (2 s)
        n = 400

        def run(use_reject):
            filt = ESKFAHRS(gyro_noise_std=0.01, gyro_bias_std=0.0, accel_gate_alpha=10.0,
                            accel_chi2_thresh=7.815, use_accel_motion_reject=use_reject,
                            accel_motion_scale=2.0)
            filt.reset(np.array([1.0, 0.0, 0.0, 0.0]))
            theta = 0.0
            for _ in range(n):
                theta += wy * DT
                # Pure-rotation specific force: drone truly at pitch theta, no kinematic accel ->
                # f_body = R(theta)^T @ (-g_ned). a_lin = R_hat @ f_body + g_ned ~ 0 when R_hat tracks.
                sf = G * np.array([np.sin(theta), 0.0, -np.cos(theta)])
                filt.step(np.array([0.0, wy, 0.0]), sf, DT)
            _, pitch, _ = euler_from_quat_wxyz(filt.q_wxyz)
            return float(np.rad2deg(pitch))

        p_off = run(False)
        p_on = run(True)
        assert abs(p_on - target_deg) < 2.0, f"gate-ON lost a REAL tilt (got {p_on:.2f}, want ~{target_deg})"
        assert abs(p_on - p_off) < 1.0, (
            f"gate must be inert on honest rotation (off {p_off:.2f}, on {p_on:.2f} deg)"
        )


class TestNormalFlightByteIdentity:
    """With the gate DISABLED (the default) the full quaternion stream must be bit-identical to today
    across the benign synthetic scenarios -- the 'normal flight unchanged' invariant."""

    @pytest.mark.parametrize("scen", [Scenario.STATIC_GRAVITY, Scenario.CONSTANT_SPIN,
                                      Scenario.ROLLING_MANEUVER])
    def test_default_off_byte_identical_to_pre_gate(self, scen):
        """Construct the filter WITHOUT touching the new fields (default off) vs explicitly OFF:
        identical. (The cross-validation/bench tests pin that default-constructed filters are
        unchanged from before the gate existed; this pins that the new field defaults to inert.)"""
        seq = generate_imu_sequence(scen, duration_s=5.0, dt=0.005, seed=7,
                                    gyro_noise_std=0.005, accel_noise_std=0.05)
        default = ESKFAHRS(gyro_noise_std=0.01, accel_gate_alpha=10.0)
        explicit_off = ESKFAHRS(gyro_noise_std=0.01, accel_gate_alpha=10.0,
                                use_accel_motion_reject=False)
        default.reset(seq.q_wxyz_gt[0])
        explicit_off.reset(seq.q_wxyz_gt[0])
        q_d = np.array([default.step(seq.gyro[i], seq.accel[i], seq.dt) for i in range(seq.N)])
        q_e = np.array([explicit_off.step(seq.gyro[i], seq.accel[i], seq.dt) for i in range(seq.N)])
        assert np.array_equal(q_d, q_e), (
            f"{scen.name}: default must equal explicit-off (max diff {np.max(np.abs(q_d - q_e)):.2e})"
        )


class TestIEKFMirror:
    """The IEKF carries the identical gate so the ESKF<->IEKF cross-validation stays exact."""

    def test_iekf_inflation_matches_eskf(self):
        """At the same attitude and accel the IEKF inflation factor equals the ESKF's."""
        sf = np.array([4.0, 0.0, -G])
        e = ESKFAHRS(use_accel_motion_reject=True, accel_motion_scale=2.0)
        i = LeftInvariantEKF(use_accel_motion_reject=True, accel_motion_scale=2.0)
        e.reset(np.array([1.0, 0.0, 0.0, 0.0]))
        i.reset(np.array([1.0, 0.0, 0.0, 0.0]))
        assert abs(e._accel_motion_inflation(sf) - i._accel_motion_inflation(sf)) < 1e-12

    def test_iekf_a8_holds_pitch(self):
        """The IEKF reproduces the same A8 hold-with-gate behaviour as the ESKF."""
        pitch_off = _run_a8(use_reject=False, klass=LeftInvariantEKF)
        pitch_on = _run_a8(use_reject=True, klass=LeftInvariantEKF)
        assert pitch_off >= 20.0 and pitch_on < 7.0, (
            f"IEKF A8: off {pitch_off:.2f} deg should diverge, on {pitch_on:.2f} deg should hold"
        )

    def test_iekf_inflation_unity_at_rest(self):
        i = LeftInvariantEKF(use_accel_motion_reject=True, accel_motion_scale=2.0)
        i.reset(np.array([1.0, 0.0, 0.0, 0.0]))
        assert abs(i._accel_motion_inflation(np.array([0.0, 0.0, -G])) - 1.0) < 1e-12


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
