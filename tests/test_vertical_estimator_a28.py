"""VQ2 A28 -- 2-state complementary-filter vertical estimator + single-PD law, 2026-07-03.

THE BUG THIS FIXES (run 20260703_013748, TIMEOUT, floor->ceiling divergence -- see
handoff/vq2_a28_vertical_stability_spec_2026-07-03.md): the vertical loop's only velocity
feedback (``vert_vz_est``) was fiction three ways -- (a) sustained SUB-threshold contact accel
(~-2.4 m/s^2 scraping, under the 20 m/s^2 contact gate) poisoned the washout to its +1.5 rail;
(b) the A26 finite-difference pose fusion injected a standing -0.67 m/s DC fiction; (c) the
washout's DC is tau*(any accel residual) by construction -- AND the controller's velocity TARGET
``vz_t`` flickered 0<->+/-1 at 2.3 Hz through the hold-last-demand bridge, stepping the damping
term by 0.25 collective (94% of hover) as a square wave. The loop was a stiff (omega_n ~3.1
rad/s, via the HIDDEN -kd*vz_t position path) position-only servo with anti-damping feedback.

THE FIX, pinned here:
  ESTIMATOR (``use_zoff_filter``, default OFF = byte-identical): IMU predicts BOTH states
  (z_off, vz_rel) with NO leak while innovations flow; each fresh pose corrects both via
  alpha-beta POSITION innovations (alpha=0.4, beta=0.15 -- the beta line bleeds off the contact
  poison within a few pose periods); innovation gate 2 m + reseed-after-4 consecutive rejects
  (the observed +/-8 m gate-track jumps vs a real gate handoff); internal state UNCLAMPED (the
  A25 +/-3 state clamp destroyed all lodged-phase information); washout leak resumes after
  ``blind_coast_after_s`` without an accepted innovation (pose-gap coast degrades to exactly the
  A24 bounded behaviour).
  CONTROLLER (``gate_pd_vertical``, default OFF = byte-identical): the single PD
  ``thrust = hover - kp_gate*clip(z_off, +/-3) + ff_vertical_kd_alt*vz_lp`` -- ``vz_t`` is OUT of
  the law (bridge flicker harmless, hidden position path gone). Sizing on the MEASURED plant
  (c ~= 37 m/s^2/collective): kp_gate=0.04, kd=0.06 -> omega_n=1.21 rad/s, zeta=0.92, steady
  2.0 m/s descent emerging from Kp/Kd.
  A27 (floor 0.15 + thrust slew 2.0): KEPT as dormant safety (spec §2.4 verdict + pre-registered
  revert criterion).

[VQ2 A28, 2026-07-03]
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from racer.contracts import NavState, Setpoint  # noqa: E402
from racer.gate_seeker import make_seeker_controller  # noqa: E402
from racer.vertical_estimator import VerticalEstimator  # noqa: E402

IMU_DT = 1.0 / 140.0    # the live HIGHRES_IMU rate (~143 Hz)


def _burn_bias_capture(est: VerticalEstimator, dt: float = IMU_DT) -> None:
    """Advance past the pre-arm bias-capture window at true-zero bias (a_up=0)."""
    n_capture = int(np.ceil(est.bias_capture_s / dt)) + 1
    for _ in range(n_capture):
        est.predict(0.0, dt)


def _advance_clock(est: VerticalEstimator, seconds: float, dt: float = IMU_DT) -> None:
    """Advance the estimator's internal elapsed clock with zero-accel predict ticks."""
    n = max(int(round(seconds / dt)), 1)
    for _ in range(n):
        est.predict(0.0, dt)


def _fresh_filter(latch0: float | None = None) -> VerticalEstimator:
    """A seeded, bias-captured A28 filter, optionally with an initial lock at ``latch0``."""
    est = VerticalEstimator(use_zoff_filter=True)
    est.seed()
    _burn_bias_capture(est)
    if latch0 is not None:
        est.latch_offset(offset_z_world=latch0, obs_age_s=0.0)
    return est


# ===========================================================================
# 1. Defaults: everything OFF / neutral (VQ1 + pre-A28 vq2 byte-identical)
# ===========================================================================
def test_estimator_defaults_flag_off_and_spec_params():
    est = VerticalEstimator()
    assert est.use_zoff_filter is False
    assert est.zoff_alpha == pytest.approx(0.4)
    assert est.zoff_beta == pytest.approx(0.15)
    assert est.zoff_beta_dt_floor_s == pytest.approx(0.15)
    assert est.innov_gate_m == pytest.approx(2.0)
    assert est.reseed_after == 4
    assert est.blind_coast_after_s == pytest.approx(1.0)
    assert est.export_clip_mps == pytest.approx(1.5)   # field default UNCHANGED (profile sets 2.5)


def test_controller_defaults_flag_off():
    from racer.controller import Controller
    assert Controller().gate_pd_vertical is False
    assert Controller().gate_pd_z_off_clip_m == pytest.approx(3.0)
    assert make_seeker_controller().gate_pd_vertical is False


def test_navigator_config_default_no_estimator_overrides():
    from racer.navigator import NavigatorConfig
    assert NavigatorConfig().vertical_estimator_overrides is None


# ===========================================================================
# 2. alpha-beta CORRECT math (exact)
# ===========================================================================
def test_first_latch_is_initial_lock():
    """The first-ever latch has no prior state to innovate against: it LOCKS z_off = z_meas
    (latency comp included) and starts the accept clock -- no alpha attenuation."""
    est = _fresh_filter()
    est.latch_offset(offset_z_world=5.0, obs_age_s=0.0)
    assert est.z_off == pytest.approx(5.0)
    assert est._zoff_last_accept_t_s is not None


def test_alpha_beta_update_exact():
    """One accepted innovation moves the states by EXACTLY the spec's alpha-beta step:
    z_off += alpha*innov ; vz += -beta*innov/max(dt_since_last_accept, 0.15)."""
    est = _fresh_filter(latch0=5.0)
    est._zoff_last_accept_t_s = 0.0          # pin clean timestamps (no float-sum noise)
    est._contact_elapsed_s = 0.5
    vz_before = est._vz
    est.latch_offset(offset_z_world=4.8, obs_age_s=0.0)   # innov = 4.8 - 5.0 = -0.2
    assert est.z_off == pytest.approx(5.0 + 0.4 * (-0.2), abs=1e-12)
    assert est._vz == pytest.approx(vz_before - 0.15 * (-0.2) / 0.5, abs=1e-12)
    assert est.zoff_last_accepted is True
    assert est.zoff_last_innov == pytest.approx(-0.2)
    assert est.zoff_miss == 0


def test_beta_dt_floor_bounds_the_velocity_kick():
    """Two accepts landing (near-)simultaneously must divide by the 0.15 s floor, not by ~0 --
    else a modest innovation becomes an unbounded velocity kick."""
    est = _fresh_filter(latch0=5.0)
    est._zoff_last_accept_t_s = est._contact_elapsed_s    # dt_since_last_accept == 0
    vz_before = est._vz
    est.latch_offset(offset_z_world=4.5, obs_age_s=0.0)   # innov = -0.5
    assert est._vz == pytest.approx(vz_before - 0.15 * (-0.5) / 0.15, abs=1e-12)


# ===========================================================================
# 3. SIGN pin: descending toward the gate => vz_rel goes POSITIVE
#    (mirrors A26's test_vz_gate_sign_descending_toward_gate_is_positive)
# ===========================================================================
def test_zoff_filter_sign_descending_toward_gate_drives_vz_positive():
    """Drone descending toward a static gate: the measured down-positive offsets SHRINK faster
    than the (stationary) state predicts -> negative innovations -> the beta line must push
    vz_rel MORE POSITIVE (down-positive 'descending = +' convention)."""
    est = _fresh_filter(latch0=5.0)
    _advance_clock(est, 0.2)
    vz_before = est.vz
    est.latch_offset(offset_z_world=4.8, obs_age_s=0.0)   # shrinking = descending toward it
    assert est.vz > vz_before, \
        "descending toward the gate (shrinking offsets) must drive vz_rel POSITIVE"


def test_zoff_filter_sign_climbing_away_drives_vz_negative():
    est = _fresh_filter(latch0=5.0)
    _advance_clock(est, 0.2)
    vz_before = est.vz
    est.latch_offset(offset_z_world=5.2, obs_age_s=0.0)   # growing = climbing away
    assert est.vz < vz_before, \
        "climbing away (growing offsets) must drive vz_rel NEGATIVE"


# ===========================================================================
# 4. Innovation gate: reject leaves the state untouched; accept resets miss
# ===========================================================================
def test_innovation_gate_rejects_gate_track_jump():
    """A +/-8 m class offset jump (the observed gate-track instability) must be REJECTED: both
    states untouched, miss counter bumped, accepted flag False."""
    est = _fresh_filter(latch0=5.0)
    _advance_clock(est, 0.2)
    z_off_before, vz_before = est.z_off, est._vz
    est.latch_offset(offset_z_world=13.0, obs_age_s=0.0)   # innov = +8 >> 2.0
    assert est.z_off == pytest.approx(z_off_before)
    assert est._vz == pytest.approx(vz_before)
    assert est.zoff_miss == 1
    assert est.zoff_last_accepted is False


def test_accepted_innovation_resets_miss_counter():
    est = _fresh_filter(latch0=5.0)
    _advance_clock(est, 0.2)
    est.latch_offset(offset_z_world=13.0, obs_age_s=0.0)   # reject -> miss=1
    assert est.zoff_miss == 1
    _advance_clock(est, 0.2)
    est.latch_offset(offset_z_world=4.9, obs_age_s=0.0)    # accept -> miss resets
    assert est.zoff_miss == 0
    assert est.zoff_last_accepted is True


def test_reseed_after_four_consecutive_rejects_relocks_vz_untouched():
    """FOUR consecutive out-of-gate innovations = a REAL retarget (gate handoff), not noise:
    re-lock z_off = z_meas with vz_rel UNTOUCHED (the drone's motion did not jump with the
    track)."""
    est = _fresh_filter(latch0=5.0)
    vz_before = est._vz
    for i in range(3):
        _advance_clock(est, 0.1)
        est.latch_offset(offset_z_world=13.0, obs_age_s=0.0)
        assert est.zoff_miss == i + 1
        assert est.z_off == pytest.approx(5.0), "state must hold through the reject streak"
    _advance_clock(est, 0.1)
    est.latch_offset(offset_z_world=13.0, obs_age_s=0.0)   # the 4th consecutive reject: RESEED
    assert est.z_off == pytest.approx(13.0), "reseed_after=4 must re-lock z_off onto z_meas"
    assert est._vz == pytest.approx(vz_before), "reseed must leave vz_rel untouched"
    assert est.zoff_miss == 0


# ===========================================================================
# 5. Internal state UNCLAMPED (the A25 +/-3 clamp is gone under the flag)
# ===========================================================================
def test_state_unclamped_beyond_3m_under_flag():
    """After a reseed to +13 m the exported z_off must READ +13 (the A25 clip pinned at +3.00 the
    entire lodged back half of 20260703_013748, destroying the information); the clamp now lives
    at the controller consumption point only."""
    est = _fresh_filter(latch0=5.0)
    for _ in range(4):
        _advance_clock(est, 0.1)
        est.latch_offset(offset_z_world=13.0, obs_age_s=0.0)
    assert est.z_off == pytest.approx(13.0), "z_off export must be UNCLAMPED under use_zoff_filter"


def test_legacy_path_keeps_the_clamp():
    """Flag OFF: the A25 +/-3 read clip is preserved -- byte-identical legacy behaviour."""
    est = VerticalEstimator()                      # use_zoff_filter default False
    est.seed()
    _burn_bias_capture(est)
    est.latch_offset(offset_z_world=13.0, obs_age_s=0.0)
    assert est.z_off == pytest.approx(3.0), "legacy latch must still clamp to +/-3"


# ===========================================================================
# 6. Contact/scraping poison (the -2.4 m/s^2 SUB-threshold case, spec §1.3a):
#    the beta innovations must bleed it off -- the old washout railed on it
# ===========================================================================
def test_sub_threshold_scraping_poison_does_not_rail_the_filter():
    """5 s of sustained a_up = -2.4 m/s^2 (floor-scraping contact residual, ZERO samples over the
    20 m/s^2 contact gate -- the exact signature that railed the washout to +1.5 export / +4.8
    internal in 20260703_013748) while the drone is actually STATIONARY at z_off=1.0 with 10 Hz
    fresh poses: the A28 filter's beta corrections must keep vz_rel BOUNDED (far off the tau*b =
    4.8 rail) and z_off near the truth. Reference: the same stream through the OLD washout rails
    to ~tau*b internally."""
    est = _fresh_filter(latch0=1.0)
    ref = VerticalEstimator()                     # the OLD washout (no fusion), same stream
    ref.seed()
    _burn_bias_capture(ref)
    ref.latch_offset(offset_z_world=1.0, obs_age_s=0.0)

    n_per_pose = max(int(round(0.1 / IMU_DT)), 1)     # ~10 Hz pose cadence
    t = 0.0
    for k in range(int(5.0 / (n_per_pose * IMU_DT))):
        for _ in range(n_per_pose):
            est.predict(-2.4, IMU_DT)             # sustained sub-threshold poison
            ref.predict(-2.4, IMU_DT)
        t += n_per_pose * IMU_DT
        # the truth: stationary at z_off = 1.0; poses keep saying so
        est.latch_offset(offset_z_world=1.0, obs_age_s=0.0)
        # (the legacy path latches too -- its z_off overwrite hides the drift, but its vz is
        #  never corrected by position innovations)

    assert abs(ref._vz) > 3.0, "repro lost: the OLD washout should rail toward tau*b = 4.8"
    assert abs(est._vz) < 1.5, \
        f"A28 filter railed on sub-threshold contact poison: vz_rel={est._vz:+.2f}"
    assert abs(est._vz) < abs(ref._vz) / 2.5, "A28 must beat the washout decisively here"
    assert abs(est.z_off - 1.0) < 0.5, f"z_off drifted off truth: {est.z_off:+.2f}"


# ===========================================================================
# 7. Pose-gap coast: NO leak while innovations flow; washout leak after 1 s blind
# ===========================================================================
def test_no_leak_while_measurements_flow():
    """Inside the blind_coast window after an accept, the predict is PURE integration: with
    a_dn - b_hat == 0 the velocity must hold EXACTLY (the washout would decay it by
    exp(-0.5/2) ~= 0.78 over 0.5 s)."""
    est = _fresh_filter(latch0=2.0)
    for _ in range(int(0.2 / IMU_DT)):
        est.predict(-3.0, IMU_DT)                 # kick vz positive (descent)
    # re-accept so the freshness clock is recent (the kick consumed ~0.2 s of it)
    est.latch_offset(offset_z_world=est._z_off, obs_age_s=0.0)   # zero-innovation accept
    vz_kicked = est._vz
    assert vz_kicked > 0.3, "setup: kick did not raise vz"
    for _ in range(int(0.5 / IMU_DT)):            # 0.5 s < blind_coast_after_s = 1.0
        est.predict(0.0, IMU_DT)
    assert est._vz == pytest.approx(vz_kicked, abs=1e-9), \
        "vz must NOT leak while the innovation stream is fresh"


def test_blind_coast_resumes_the_washout_leak():
    """After > blind_coast_after_s without an accepted innovation, the washout leak resumes and
    the velocity decays toward 0 -- exactly today's bounded blind behaviour (vision gone)."""
    est = _fresh_filter(latch0=2.0)
    for _ in range(int(0.2 / IMU_DT)):
        est.predict(-3.0, IMU_DT)
    vz_kicked = est._vz
    assert vz_kicked > 0.3, "setup: kick did not raise vz"
    # 5 s with NO latches: the first ~1.0 s is still inside the fresh window (leak off, vz
    # holds), the remaining ~4 s decay at tau=2 -> vz ~ e^-2 ~= 0.135 of the kicked value.
    _advance_clock(est, 5.0)
    assert abs(est._vz) < 0.25 * vz_kicked, \
        "vz must wash out (leak resumed) once blind for > blind_coast_after_s"


def test_never_latched_behaves_as_pure_washout():
    """Before any gate is ever seen, the flag-ON filter is in blind coast from tick 1: identical
    recurrence to the legacy washout on the same input stream."""
    est_new = VerticalEstimator(use_zoff_filter=True)
    est_old = VerticalEstimator()
    for est in (est_new, est_old):
        est.seed()
        _burn_bias_capture(est)
    rng = np.random.default_rng(11)
    for a in rng.normal(0.0, 3.0, size=400):
        est_new.predict(float(a), IMU_DT)
        est_old.predict(float(a), IMU_DT)
    assert est_new._vz == pytest.approx(est_old._vz, abs=0.0), \
        "never-latched flag-ON filter must be BIT-IDENTICAL to the legacy washout"


# ===========================================================================
# 8. Contact gate still held (KEPT as-is under the flag)
# ===========================================================================
def test_contact_gate_still_freezes_the_filter():
    est = _fresh_filter(latch0=1.0)
    _advance_clock(est, 0.1)
    vz_before, z_off_before = est._vz, est.z_off
    est.predict(-50.0, IMU_DT)                    # a real impact spike, over the 20 m/s^2 gate
    assert est.contact_frozen()
    assert est._vz == pytest.approx(vz_before), "vz must be held on a contact tick"
    assert est.z_off == pytest.approx(z_off_before), "z_off must be held on a contact tick"


# ===========================================================================
# 9. A26 fusion superseded: never runs under the flag
# ===========================================================================
def test_fusion_is_skipped_under_zoff_filter_even_if_both_flags_on():
    """Belt-and-braces: a (mis)configured estimator with BOTH use_zoff_filter and
    use_gate_vz_fusion on must run ONLY the A28 correct step -- the A26 finite-difference
    bookkeeping stays untouched (the beta innovations supersede it)."""
    est = VerticalEstimator(use_zoff_filter=True, use_gate_vz_fusion=True)
    est.seed()
    _burn_bias_capture(est)
    est.latch_offset(offset_z_world=5.0, obs_age_s=0.0)
    _advance_clock(est, 0.2)
    est.latch_offset(offset_z_world=4.8, obs_age_s=0.0)
    assert est._prev_fresh_offset_z is None, "A26 fusion bookkeeping must never run under A28"
    assert est._prev_fresh_offset_t_s is None


def test_seed_resets_a28_state():
    est = _fresh_filter(latch0=5.0)
    _advance_clock(est, 0.2)
    est.latch_offset(offset_z_world=13.0, obs_age_s=0.0)   # one reject -> miss=1, innov stashed
    assert est.zoff_miss == 1
    est.seed()
    assert est.zoff_miss == 0
    assert est._zoff_last_accept_t_s is None
    assert np.isnan(est.zoff_last_innov)
    assert est.zoff_last_accepted is None
    assert np.isnan(est.z_off)


# ===========================================================================
# 10. CONTROLLER: the single PD law (gate_pd_vertical)
# ===========================================================================
def _nav(vz=0.0, z_off=0.0, t_ns=0):
    return NavState(sim_time_ns=t_ns, position_ned=np.zeros(3), velocity_ned=np.zeros(3),
                    roll=0.0, pitch=0.0, yaw=0.0, angular_rate_body=np.zeros(3),
                    vert_vz_est=vz, z_off_est=z_off)


def _sp(vz_t=0.0, t_ns=0):
    return Setpoint(sim_time_ns=t_ns, accel_ned=np.array([1.0, 0.0, 0.0]), yaw=0.0,
                    velocity_ned=np.array([0.0, 0.0, vz_t]))


def _a28_ctrl(**over):
    params = dict(ff_owns_vertical=True, use_vertical_estimator=True, gate_pd_vertical=True,
                  kp_alt=0.0, kp_gate=0.04, ff_vertical_kd_alt=0.06, ff_vertical_vz_lp_alpha=1.0)
    params.update(over)
    return make_seeker_controller(**params)


def test_gate_pd_law_exact_formula():
    """thrust = clip(hover - kp_gate*clip(z_off,+/-3) + kd*vz_lp) -- exact, level attitude
    (tilt-comp = 1), lp_alpha=1 so vz_lp == vz_meas on the first tick."""
    ctrl = _a28_ctrl()
    thrust = ctrl.command(_nav(vz=0.5, z_off=2.0), _sp(vz_t=0.0)).thrust
    expected = ctrl.hover_thrust - 0.04 * 2.0 + 0.06 * 0.5
    expected = float(np.clip(expected, ctrl.alt_thrust_lo, ctrl.alt_thrust_hi))
    assert thrust == pytest.approx(expected, abs=1e-9)


def test_gate_pd_above_gate_sinks_below_gate_climbs():
    """Sign pins (spec tests i/ii): above the gate (z_off>0) => thrust BELOW hover (sink);
    below (z_off<0) => ABOVE hover (climb); descending (vz>0) => MORE thrust (brake)."""
    hover = _a28_ctrl().hover_thrust
    assert _a28_ctrl().command(_nav(vz=0.0, z_off=2.0), _sp()).thrust < hover
    assert _a28_ctrl().command(_nav(vz=0.0, z_off=-2.0), _sp()).thrust > hover
    assert _a28_ctrl().command(_nav(vz=1.0, z_off=0.0), _sp()).thrust > hover


def test_gate_pd_ignores_vz_t_bridge_flicker_zero_thrust_delta():
    """THE 2.3 Hz bridge-flicker kill (spec test v): two identical states differing ONLY in vz_t
    (the seeker's target flicking 0 <-> +1.0, exactly the hold-last-demand square wave) must
    produce BIT-IDENTICAL thrust under gate_pd_vertical -- vz_t is no longer in the law."""
    thr_a = _a28_ctrl().command(_nav(vz=0.4, z_off=1.5), _sp(vz_t=1.0)).thrust
    thr_b = _a28_ctrl().command(_nav(vz=0.4, z_off=1.5), _sp(vz_t=0.0)).thrust
    assert thr_a == thr_b, "vz_t must produce ZERO thrust delta under the A28 single PD"
    # and multi-tick: a full flicker train changes nothing vs a constant-vz_t train
    c_flick, c_const = _a28_ctrl(), _a28_ctrl()
    dt_ns = int(1e9 / 20)
    for i in range(20):
        t = i * dt_ns
        thr_f = c_flick.command(_nav(vz=0.4, z_off=1.5, t_ns=t),
                                _sp(vz_t=(1.0 if i % 2 else 0.0), t_ns=t)).thrust
        thr_c = c_const.command(_nav(vz=0.4, z_off=1.5, t_ns=t), _sp(vz_t=1.0, t_ns=t)).thrust
        assert thr_f == thr_c, f"flicker leaked into thrust at tick {i}"


def test_gate_pd_clamps_z_off_at_consumption_point():
    """The +/-3 clamp moved from the estimator state to the controller: z_off_est = +10 (an
    honest, unclamped lodged-phase estimate) must command exactly what +3 commands -- bounded
    authority, unbounded knowledge."""
    thr_10 = _a28_ctrl().command(_nav(vz=0.0, z_off=10.0), _sp()).thrust
    thr_3 = _a28_ctrl().command(_nav(vz=0.0, z_off=3.0), _sp()).thrust
    assert thr_10 == pytest.approx(thr_3, abs=0.0)


def test_gate_pd_off_is_byte_identical_to_a27_law():
    """Flag OFF (the default): the thrust equals EXACTLY the pre-A28 law
    ``hover + kp_alt*(z - z_t) - kp_gate*z_off + kd*(vz_lp - vz_t)`` -- inserting the A28 fork
    changed nothing on the OFF path (first tick: z_target latches to z, lp_alpha=1 -> vz_lp =
    vz_meas)."""
    ctrl = make_seeker_controller(ff_owns_vertical=True, use_vertical_estimator=True,
                                  kp_alt=0.0, kp_gate=0.06, ff_vertical_kd_alt=0.25,
                                  ff_vertical_vz_lp_alpha=1.0)
    vz, z_off, vz_t = 0.7, 1.2, -1.0
    thrust = ctrl.command(_nav(vz=vz, z_off=z_off), _sp(vz_t=vz_t)).thrust
    expected = ctrl.hover_thrust - 0.06 * z_off + 0.25 * (vz - vz_t)
    expected = float(np.clip(expected, ctrl.alt_thrust_lo, ctrl.alt_thrust_hi))
    assert thrust == pytest.approx(expected, abs=0.0), \
        "gate_pd_vertical OFF must reproduce the pre-A28 formula bit-for-bit"


def test_a27_floor_and_slew_kept_as_dormant_safety():
    """A28 spec §2.4 kept the A27 floor (0.15) + slew (2.0) as dormant safety. A33 V-2a (2026-07-03)
    then OPENED the floor 0.15 -> 0.05 (run 20260703_210632 reconciliation: the FLOOR, not the plant,
    capped the gate-PD's ~0.0 worst-case descent demand while ~+7 m/s^2 translational lift kept the
    drone rising). The slew (2.0/s) STAYS -- now the ramp guard into the sub-0.15 net-down region."""
    from racer.deploy_profile import vq2_case_c
    ov = vq2_case_c().controller_overrides
    assert ov["alt_thrust_lo"] == 0.05          # A33 V-2a: floor OPENED
    assert ov["alt_thrust_slew_per_s"] == 2.0   # slew RETAINED (ramp guard)


# ===========================================================================
# 11. Deploy-profile wiring (vq2_case_c ON, vq1_case_a untouched)
# ===========================================================================
def test_vq2_profile_ships_the_a28_bundle():
    from racer.deploy_profile import vq2_case_c
    prof = vq2_case_c()
    ov = prof.controller_overrides
    assert ov["gate_pd_vertical"] is True
    assert ov["kp_gate"] == pytest.approx(0.04)
    assert ov["ff_vertical_kd_alt"] == pytest.approx(0.06)
    veo = prof.nav_config.vertical_estimator_overrides
    assert veo == {"use_zoff_filter": True, "export_clip_mps": 2.5,
                   "use_soft_innov_weight": True,      # A32: Huber-soft innovation weighting
                   "zoff_reseed_min_w": 0.3}           # A33 V-1: weight-qualified reseed
    assert prof.nav_config.use_gate_vz_fusion is False   # A26 fusion superseded


def test_vq1_profile_carries_none_of_it():
    from racer.deploy_profile import vq1_case_a
    from racer.navigator import NavigatorConfig
    prof = vq1_case_a()
    assert prof.controller_overrides is None
    assert prof.nav_config.vertical_estimator_overrides is None
    assert prof.nav_config == NavigatorConfig(), \
        "vq1_case_a nav_config must remain EXACTLY the dataclass defaults (byte-identical)"


def test_a28_gain_sizing_sanity():
    """The omega_n/zeta arithmetic the gains were chosen by (spec §2.3), pinned so a future gain
    edit that silently breaks the damping ratio fails a test: c=37 m/s^2/collective (measured),
    omega_n = sqrt(c*Kp) ~= 1.21 rad/s, zeta = c*Kd/(2*omega_n) ~= 0.92, steady descent at the
    3 m clamp = (Kp/Kd)*3 = 2.0 m/s (inside the raised 2.5 export clip)."""
    from racer.deploy_profile import vq2_case_c
    ov = vq2_case_c().controller_overrides
    c = 37.0
    kp, kd = ov["kp_gate"], ov["ff_vertical_kd_alt"]
    omega_n = float(np.sqrt(c * kp))
    zeta = c * kd / (2.0 * omega_n)
    v_steady = (kp / kd) * 3.0
    assert 1.0 < omega_n < 1.5
    assert 0.7 < zeta < 1.1, f"vertical PD must stay near-critically damped, got zeta={zeta:.2f}"
    assert v_steady == pytest.approx(2.0)
    assert v_steady < vq2_case_c().nav_config.vertical_estimator_overrides["export_clip_mps"], \
        "the legitimate steady descent must fit inside the export clip (else the damper lies)"


# ===========================================================================
# 12. Navigator threads the construction overrides
# ===========================================================================
def test_navigator_threads_vertical_estimator_overrides():
    from racer.contracts import DroneState
    from racer.navigator import Navigator, NavigatorConfig

    G = 9.80665

    def _ds(t_s):
        return DroneState(sim_time_ns=int(t_s * 1e9),
                          accel_body=np.array([0.0, 0.0, -G]),
                          gyro_body=np.zeros(3),
                          orientation_ned_wxyz=np.array([1.0, 0.0, 0.0, 0.0]))

    cfg = NavigatorConfig(use_given_position=False, use_given_velocity=False,
                          use_ahrs=True, use_vertical_estimator=True,
                          vertical_estimator_overrides={"use_zoff_filter": True,
                                                        "export_clip_mps": 2.5})
    nav = Navigator(gates=[], detector=None, config=cfg)
    nav.update(_ds(0.0))
    assert nav._vert_est is not None
    assert nav._vert_est.use_zoff_filter is True
    assert nav._vert_est.export_clip_mps == pytest.approx(2.5)

    # None (default) => the pre-A28 construction, byte-identical
    cfg_off = NavigatorConfig(use_given_position=False, use_given_velocity=False,
                              use_ahrs=True, use_vertical_estimator=True)
    nav_off = Navigator(gates=[], detector=None, config=cfg_off)
    nav_off.update(_ds(0.0))
    assert nav_off._vert_est.use_zoff_filter is False
    assert nav_off._vert_est.export_clip_mps == pytest.approx(1.5)


def test_export_clip_2_5_actually_clips_at_2_5():
    est = VerticalEstimator(use_zoff_filter=True, export_clip_mps=2.5)
    est.seed()
    _burn_bias_capture(est)
    for _ in range(3000):
        est.predict(-30.0, IMU_DT)               # sustained hard fictional descent, blind coast
        assert -2.5 - 1e-9 <= est.vz <= 2.5 + 1e-9
