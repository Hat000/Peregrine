"""VQ2 A25 — gate-relative altitude term (ẑ_off), 2026-07-02.

THE BUG THIS FIXES (run 20260702_203428_rl_s1_f1): the deployed vertical law had NO position
term (kp_alt=0), so it could null a velocity but never SEEK a gate-relative height -- the drone
climbed monotonically (z +3.7 m, vz +3.5 m/s at gate-1-top impact) and never descended. THE FIX:
``VerticalEstimator`` gains a gate-relative vertical-offset state ``z_off`` -- latched from the
seeker's fresh pose, propagated at IMU rate by the SAME washout vz, latency-compensated, clamped
+/-3 m, and frozen during contact events -- driving a NEW controller term ``-kp_gate*z_off`` that
goes BELOW hover when the drone is ABOVE the gate (z_off>0), so it sinks onto gate height instead
of climbing into it.

Every sign in this file is PINNED by a test per the build spec
(handoff/vq2_gate_relative_altitude_spec_2026-07-02.md §8.1); the spec's own §3.2 draft had the
WRONG sign (a plus), corrected by §8.1(h)/§9 to the MINUS used here and in ``controller.py``.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from racer.gate_seeker import make_seeker_controller  # noqa: E402
from racer.contracts import NavState, Setpoint  # noqa: E402
from racer.vertical_estimator import VerticalEstimator  # noqa: E402

IMU_DT = 1.0 / 140.0    # the live HIGHRES_IMU rate (~143 Hz)


def _burn_bias_capture(est: VerticalEstimator, dt: float = IMU_DT) -> None:
    """Advance past the pre-arm bias-capture window at true-zero bias (a_up=0)."""
    n_capture = int(np.ceil(est.bias_capture_s / dt)) + 1
    for _ in range(n_capture):
        est.predict(0.0, dt)


# ---------------------------------------------------------------------------
# (a) latch
# ---------------------------------------------------------------------------
def test_latch_sets_z_off_and_seen():
    est = VerticalEstimator()
    est.seed()
    assert np.isnan(est.z_off)
    est.latch_offset(offset_z_world=1.5, obs_age_s=0.0)
    assert est.z_off == pytest.approx(1.5)
    assert est._z_off_seen is True


def test_latch_noop_when_unseeded():
    est = VerticalEstimator()
    est.latch_offset(offset_z_world=1.5, obs_age_s=0.0)
    assert np.isnan(est.z_off)


# ---------------------------------------------------------------------------
# (b) propagate sign (SIGN-PINNING, spec §1.3.1)
# ---------------------------------------------------------------------------
def test_propagate_sign_climb_grows_z_off_positive():
    """Seed at rest (z_off=0 via a latch of 0), feed a sustained UPWARD accel (a_up>0) through
    predict for 1s: vz must go NEGATIVE (down-positive convention -> climb reads negative) AND
    z_off must grow POSITIVE (drone rose above the gate, gate is further below). This pins the
    §1.3.1 propagate sign: z_off <- z_off - vz*dt."""
    est = VerticalEstimator()
    est.seed()
    _burn_bias_capture(est)
    est.latch_offset(offset_z_world=0.0, obs_age_s=0.0)
    assert est.z_off == pytest.approx(0.0)

    dt = IMU_DT
    n = int(1.0 / dt)
    for _ in range(n):
        est.predict(5.0, dt)          # a_up = +5 (climbing) sustained for ~1s
    assert est.vz < 0.0, "climb must read vz<0 in the down-positive convention"
    assert est.z_off > 0.0, "z_off must grow POSITIVE on a sustained climb (gate falls further below)"


def test_propagate_sign_descent_shrinks_z_off():
    """The mirror check: a sustained DOWNWARD accel (a_up<0) makes vz>0 and z_off DECREASE (the
    drone sinks toward / past the gate height)."""
    est = VerticalEstimator()
    est.seed()
    _burn_bias_capture(est)
    est.latch_offset(offset_z_world=2.0, obs_age_s=0.0)
    dt = IMU_DT
    n = int(1.0 / dt)
    for _ in range(n):
        est.predict(-5.0, dt)          # a_up = -5 (descending) sustained for ~1s
    assert est.vz > 0.0, "descent must read vz>0 in the down-positive convention"
    assert est.z_off < 2.0, "z_off must SHRINK on a sustained descent (closing on gate height)"


# ---------------------------------------------------------------------------
# (c) latency-comp sign (spec §2.3)
# ---------------------------------------------------------------------------
def test_latency_comp_climb_makes_latched_offset_more_positive():
    """Drive vz to a steady climb value (vz<0), then latch_offset(offset_z=0.0, obs_age=0.5): the
    gate is now further below than the raw capture says (the drone climbed during the 0.5s lag),
    so the LATCHED z_off must be POSITIVE. Pins z_off_latched = offset_z_captured - vz*obs_age."""
    est = VerticalEstimator()
    est.seed()
    _burn_bias_capture(est)
    dt = IMU_DT
    for _ in range(int(0.5 / dt)):
        est.predict(3.0, dt)           # climb -> vz goes negative
    assert est.vz < 0.0, "setup: expected a climb (vz<0) before latching"
    est.latch_offset(offset_z_world=0.0, obs_age_s=0.5)
    assert est.z_off > 0.0, "latency-compensated latch must read positive after a climb during the lag"


def test_latency_comp_zero_age_is_passthrough():
    est = VerticalEstimator()
    est.seed()
    _burn_bias_capture(est)
    est.predict(3.0, IMU_DT)
    est.latch_offset(offset_z_world=1.234, obs_age_s=0.0)
    assert est.z_off == pytest.approx(1.234)


# ---------------------------------------------------------------------------
# (d) clamp +/-3 m
# ---------------------------------------------------------------------------
def test_z_off_clamped_to_plus_minus_3():
    est = VerticalEstimator()
    est.seed()
    _burn_bias_capture(est)
    est.latch_offset(offset_z_world=0.0, obs_age_s=0.0)
    dt = IMU_DT
    # sustained max-clamp climb for a long stretch -- z_off must never exceed the clamp
    for _ in range(6000):
        est.predict(30.0, dt)          # a_up clamp is 30 m/s^2; sustained hard climb
        assert -3.0 - 1e-9 <= est.z_off <= 3.0 + 1e-9


def test_z_off_clamp_value_is_3_metres():
    assert VerticalEstimator().z_off_clip_m == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# (e) contact-freeze skips BOTH vz and z_off updates, then resumes after the refractory window
# ---------------------------------------------------------------------------
def test_contact_freeze_holds_vz_and_z_off_then_resumes():
    est = VerticalEstimator()
    est.seed()
    _burn_bias_capture(est)
    est.latch_offset(offset_z_world=1.0, obs_age_s=0.0)
    dt = IMU_DT
    # a few normal ticks first (mild climb) so vz/z_off are moving
    for _ in range(20):
        est.predict(2.0, dt)
    vz_before = est.vz
    z_off_before = est.z_off

    # a contact spike: |a_dn - b_hat| > a_contact_mps2 (20 m/s^2)
    est.predict(-50.0, dt)             # a_up=-50 -> a_dn=+50, way past the 20 m/s^2 contact threshold
    assert est.contact_frozen()
    assert est.vz == pytest.approx(vz_before), "vz must be UNCHANGED on the contact tick"
    assert est.z_off == pytest.approx(z_off_before), "z_off must be UNCHANGED on the contact tick"

    # stay frozen through the refractory hold (contact_hold_s = 0.25s default)
    n_hold = int(est.contact_hold_s / dt)
    for _ in range(max(n_hold - 2, 0)):
        est.predict(2.0, dt)           # even a normal input must NOT move vz/z_off during the hold
    assert est.vz == pytest.approx(vz_before), "vz moved during the refractory hold"
    assert est.z_off == pytest.approx(z_off_before), "z_off moved during the refractory hold"

    # after the hold window elapses, normal updates resume
    for _ in range(int(0.5 / dt)):
        est.predict(2.0, dt)
    assert not est.contact_frozen()
    assert est.vz != pytest.approx(vz_before), "vz must resume updating after the refractory hold"
    assert est.z_off != pytest.approx(z_off_before), "z_off must resume updating after the refractory hold"


def test_contact_gate_defaults():
    est = VerticalEstimator()
    assert est.a_contact_mps2 == pytest.approx(20.0)
    assert est.contact_hold_s == pytest.approx(0.25)


def test_contact_freeze_does_not_corrupt_bias_capture_window():
    """A spike DURING the pre-arm bias-capture window must not crash / must not engage the contact
    gate early (the capture window is grounded pre-flight; the gate only arms once capture is done,
    per the spec §4.3)."""
    est = VerticalEstimator()
    est.seed()
    dt = IMU_DT
    # feed a huge spike partway through capture -- must not raise, must not freeze prematurely
    n_capture = int(np.ceil(est.bias_capture_s / dt)) + 1
    for i in range(n_capture):
        est.predict(-50.0 if i == n_capture // 2 else 0.0, dt)
    assert est._bias_capture_done


# ---------------------------------------------------------------------------
# (f) NaN fallback / no-gate
# ---------------------------------------------------------------------------
def test_z_off_nan_when_never_latched():
    est = VerticalEstimator()
    est.seed()
    _burn_bias_capture(est)
    for _ in range(50):
        est.predict(1.0, IMU_DT)
    assert np.isnan(est.z_off), "z_off must stay NaN until the first latch_offset call"


def test_controller_z_off_nan_contributes_zero():
    """Via the controller: a NaN z_off_est (no gate ever latched) must contribute exactly 0 to the
    thrust term regardless of kp_gate."""
    ctrl = make_seeker_controller(ff_owns_vertical=True, use_vertical_estimator=True, kp_gate=0.06)
    nav_nan = NavState(sim_time_ns=0, position_ned=np.zeros(3), velocity_ned=np.zeros(3),
                       roll=0.0, pitch=0.0, yaw=0.0, angular_rate_body=np.zeros(3),
                       vert_vz_est=0.0, z_off_est=float("nan"))
    nav_zero = NavState(sim_time_ns=0, position_ned=np.zeros(3), velocity_ned=np.zeros(3),
                        roll=0.0, pitch=0.0, yaw=0.0, angular_rate_body=np.zeros(3),
                        vert_vz_est=0.0, z_off_est=0.0)
    sp = Setpoint(sim_time_ns=0, accel_ned=np.array([1.0, 0.0, 0.0]), yaw=0.0)
    thr_nan = ctrl.command(nav_nan, sp).thrust
    ctrl2 = make_seeker_controller(ff_owns_vertical=True, use_vertical_estimator=True, kp_gate=0.06)
    thr_zero = ctrl2.command(nav_zero, sp).thrust
    assert thr_nan == pytest.approx(thr_zero), "NaN z_off_est must fall back to 0 (identical to z_off=0)"


# ---------------------------------------------------------------------------
# (g) min-range hold (seeker-level guard -- covered at the seeker layer)
# ---------------------------------------------------------------------------
def test_min_range_hold_seeker_level():
    """§5.2/§8.1(g): latch at range 3 (above min-trust), then attempt to latch again at a range
    below min_trust_elevation_range_m -- the seeker must SUPPRESS the second latch (ẑ_off holds the
    first value, only propagating by vz), not overwrite it with the close-range (degenerate) pose."""
    import racer.frames as frames_mod
    from racer.gate_seeker import GateSeeker, GateSeekerConfig
    from racer.contracts import GatePose

    class _FakeNavOwner:
        def __init__(self, vert_est):
            self._vert_est = vert_est

    est = VerticalEstimator()
    est.seed()
    _burn_bias_capture(est)

    seeker = GateSeeker(config=GateSeekerConfig(), nav_owner=_FakeNavOwner(est))
    nav = NavState(sim_time_ns=0, position_ned=np.zeros(3), velocity_ned=np.zeros(3),
                   roll=0.0, pitch=0.0, yaw=0.0, angular_rate_body=np.zeros(3))

    # a pose straight ahead (in the camera's optical Z) at range 3m -- above min-trust (2.0m)
    def _pose(sim_time_ns, range_m, gate_id=None):
        return GatePose(frame_id=0, sim_time_ns=sim_time_ns,
                       R_cam_gate=np.eye(3), t_cam_gate=np.array([0.0, 0.0, range_m]),
                       reproj_error_px=1.0, gate_id=gate_id)

    pose_far = _pose(1_000_000_000, 3.0)
    seeker._maybe_latch_z_off(nav, pose_far)
    z_off_after_far = est.z_off
    assert np.isfinite(z_off_after_far), "the above-min-range pose should have latched"

    # a DIFFERENT fresh pose (new sim_time_ns) at close range (< 2.0m) -- must NOT overwrite
    pose_close = _pose(2_000_000_000, 1.0)
    seeker._maybe_latch_z_off(nav, pose_close)
    assert est.z_off == pytest.approx(z_off_after_far), \
        "a close-range (<min_trust_elevation_range_m) pose must not overwrite the latched z_off"


def test_fresh_pose_dedupe_same_capture_not_relatched():
    """§1.4: a re-fed pose with the SAME sim_time_ns (async ZOH re-feed) must not re-latch."""
    from racer.gate_seeker import GateSeeker, GateSeekerConfig
    from racer.contracts import GatePose

    class _FakeNavOwner:
        def __init__(self, vert_est):
            self._vert_est = vert_est

    est = VerticalEstimator()
    est.seed()
    _burn_bias_capture(est)
    for _ in range(10):
        est.predict(2.0, IMU_DT)       # move vz off zero so a re-latch would be detectable

    seeker = GateSeeker(config=GateSeekerConfig(), nav_owner=_FakeNavOwner(est))
    nav = NavState(sim_time_ns=1_000_000_000, position_ned=np.zeros(3), velocity_ned=np.zeros(3),
                   roll=0.0, pitch=0.0, yaw=0.0, angular_rate_body=np.zeros(3))
    pose = GatePose(frame_id=0, sim_time_ns=1_000_000_000,
                   R_cam_gate=np.eye(3), t_cam_gate=np.array([0.0, 0.0, 5.0]),
                   reproj_error_px=1.0)
    seeker._maybe_latch_z_off(nav, pose)
    z_off_1 = est.z_off
    # advance vz further, then re-feed the SAME pose (same sim_time_ns) -- must not re-latch
    for _ in range(10):
        est.predict(2.0, IMU_DT)
    z_off_propagated = est.z_off
    assert z_off_propagated != pytest.approx(z_off_1), "setup: propagate should have moved z_off"
    seeker._maybe_latch_z_off(nav, pose)
    assert est.z_off == pytest.approx(z_off_propagated), \
        "re-feeding the SAME captured pose must not re-latch (would overwrite the propagated value)"


# ---------------------------------------------------------------------------
# (h) sign + authority -- THE load-bearing sign test (spec §8.1(h) / §9)
# ---------------------------------------------------------------------------
def test_z_off_positive_two_yields_thrust_below_hover():
    """THE GUARD AGAINST SHIPPING THE WRONG SIGN. With kp_gate=0.06, use_vertical_estimator=ON,
    z_off_est=+2.0 (drone 2m ABOVE the gate) and vz=vz_t (zero damping-term contribution), the
    thrust MUST be below hover -- the drone must be commanded to SINK. If this test fails, the
    control-law sign is wrong (see spec §8.1(h) / §9: the correct term is `-kp_gate*z_off`, NOT
    `+kp_gate*z_off` as originally mis-signed in §3.2)."""
    ctrl = make_seeker_controller(ff_owns_vertical=True, use_vertical_estimator=True, kp_gate=0.06)
    hover = ctrl.hover_thrust
    vz_t = 0.0
    nav = NavState(sim_time_ns=0, position_ned=np.zeros(3), velocity_ned=np.zeros(3),
                   roll=0.0, pitch=0.0, yaw=0.0, angular_rate_body=np.zeros(3),
                   vert_vz_est=vz_t, z_off_est=2.0)     # vz == vz_t -> the damping term is exactly 0
    sp = Setpoint(sim_time_ns=0, accel_ned=np.array([1.0, 0.0, 0.0]), yaw=0.0,
                  velocity_ned=np.array([0.0, 0.0, vz_t]))
    thrust = ctrl.command(nav, sp).thrust
    assert thrust < hover, (
        "z_off=+2 (drone ABOVE the gate) must produce thrust BELOW hover (a commanded sink); "
        f"got thrust={thrust} vs hover={hover} -- the control-law sign is WRONG")
    # authority sanity (spec §3.3): the reduction should be materially close to kp_gate*z_off
    # (0.06*2 = 0.12) modulo tilt-comp (level here, cos_tilt=1) and the [0.05, 0.6] clamp.
    assert thrust == pytest.approx(hover - 0.06 * 2.0, abs=1e-6)


def test_z_off_negative_two_yields_thrust_above_hover():
    """The mirror of the sign test: below the gate (z_off<0) must INCREASE thrust (climb up onto
    gate height)."""
    ctrl = make_seeker_controller(ff_owns_vertical=True, use_vertical_estimator=True, kp_gate=0.06)
    hover = ctrl.hover_thrust
    nav = NavState(sim_time_ns=0, position_ned=np.zeros(3), velocity_ned=np.zeros(3),
                   roll=0.0, pitch=0.0, yaw=0.0, angular_rate_body=np.zeros(3),
                   vert_vz_est=0.0, z_off_est=-2.0)
    sp = Setpoint(sim_time_ns=0, accel_ned=np.array([1.0, 0.0, 0.0]), yaw=0.0,
                  velocity_ned=np.array([0.0, 0.0, 0.0]))
    thrust = ctrl.command(nav, sp).thrust
    assert thrust > hover, "z_off=-2 (drone BELOW the gate) must produce thrust ABOVE hover (climb)"


def test_kp_gate_default_is_zero_byte_identical():
    """Controller.kp_gate defaults to 0.0 -> the term is identically zero regardless of z_off ->
    VQ1/case-A byte-identical (the only fork is the explicit vq2_case_c override)."""
    from racer.controller import Controller
    assert Controller().kp_gate == pytest.approx(0.0)
    ctrl = make_seeker_controller(ff_owns_vertical=True, use_vertical_estimator=True)  # kp_gate default
    assert ctrl.kp_gate == pytest.approx(0.0)
    nav_hi = NavState(sim_time_ns=0, position_ned=np.zeros(3), velocity_ned=np.zeros(3),
                      roll=0.0, pitch=0.0, yaw=0.0, angular_rate_body=np.zeros(3),
                      vert_vz_est=0.0, z_off_est=3.0)
    nav_lo = NavState(sim_time_ns=0, position_ned=np.zeros(3), velocity_ned=np.zeros(3),
                      roll=0.0, pitch=0.0, yaw=0.0, angular_rate_body=np.zeros(3),
                      vert_vz_est=0.0, z_off_est=-3.0)
    sp = Setpoint(sim_time_ns=0, accel_ned=np.array([1.0, 0.0, 0.0]), yaw=0.0,
                  velocity_ned=np.array([0.0, 0.0, 0.0]))
    ctrl_a = make_seeker_controller(ff_owns_vertical=True, use_vertical_estimator=True)
    ctrl_b = make_seeker_controller(ff_owns_vertical=True, use_vertical_estimator=True)
    thr_hi = ctrl_a.command(nav_hi, sp).thrust
    thr_lo = ctrl_b.command(nav_lo, sp).thrust
    assert thr_hi == pytest.approx(thr_lo), "kp_gate=0 must ignore z_off entirely"


def test_vq2_profile_sets_kp_gate_0_06():
    from racer.deploy_profile import vq1_case_a, vq2_case_c
    ov = vq2_case_c().controller_overrides
    assert ov is not None
    assert ov["kp_gate"] == pytest.approx(0.06)
    assert vq1_case_a().controller_overrides is None or "kp_gate" not in (vq1_case_a().controller_overrides or {})
