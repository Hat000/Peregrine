"""VQ2 R2-2 — the gate-2 turn-dive package (2026-07-04, run 20260704_183049).

Four flag-gated pieces, each pinned here (see the Controller / VerticalEstimator field comments
for the full diagnosis chain):

  A-1  ``Controller.gate_pd_brake_imu_vz``  — inside the R2-1 terminal zone (s < 1) the rate brake
       consumes the estimator's PARALLEL IMU-only washout vz (``NavState.vert_vz_imu``) instead of
       the vision-corrected vz_lp. The close-range bias sweep sign-INVERTED vz_lp via high-weight
       accepted innovations (read -1.1 "climbing" during the wire-corroborated sink that floored
       the drone at 395.24s) — no weight gate can catch that at consumption time; only a channel
       vision never touches.
  A-3  ``Controller.gate_pd_brake_neg_max``  — terminal-zone insurance: no rate signal, honest or
       garbage, pulls more than this below hover (the arrest direction is uncapped).
  B1   ``VerticalEstimator.zoff_prop_bound_m`` — dead-reckoning may never GROW |z_off| beyond
       max(bound, |current|); shrink always allowed; MEASUREMENTS untouched (the honest gate-2
       high-gate case: a large measured offset at good weight still sets the state).
  R2-2b ``Controller.gate_pd_zoff_trust_taper`` — fade the position term to q_floor when the z_off
       state is railed (|z_off| >= zoff_trust_m) AND only collapsed-weight vision is feeding it
       (zoff_w < zoff_trust_w_lo). B1 alone does NOT unpin the balloon position term (its 3.5 cap
       still exceeds the 3.0 consumption clip) — THIS release is what unpins it.

All four default OFF => byte-identical (pinned below). vq2_case_c flips them ON (pinned below).

Run: .venv\\Scripts\\python.exe -m pytest tests/test_vq2_r2_2_package.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from racer.contracts import NavState, Setpoint  # noqa: E402
from racer.controller import Controller  # noqa: E402
from racer.gate_seeker import make_seeker_controller  # noqa: E402
from racer.vertical_estimator import VerticalEstimator  # noqa: E402

IMU_DT = 1.0 / 140.0    # the live HIGHRES_IMU rate (~143 Hz)


def _burn_bias_capture(est: VerticalEstimator, dt: float = IMU_DT) -> None:
    """Advance past the pre-arm bias-capture window at true-zero bias (a_up=0)."""
    n_capture = int(np.ceil(est.bias_capture_s / dt)) + 1
    for _ in range(n_capture):
        est.predict(0.0, dt)


def _ctrl(**overrides) -> "Controller":
    """A vq2-shaped vertical-law controller: gate-PD + R2-1 terminal taper live (the R2-2 pieces
    are added per-test via overrides)."""
    kw = dict(ff_owns_vertical=True, use_vertical_estimator=True,
              gate_pd_vertical=True, gate_pd_terminal=True, gate_pd_rate_boost=1.0,
              kp_gate=0.04, ff_vertical_kd_alt=0.06, kp_alt=0.0)
    kw.update(overrides)
    return make_seeker_controller(**kw)


def _nav(vz_est=0.0, z_off=0.0, vz_imu=float("nan"), zoff_w=float("nan")) -> NavState:
    return NavState(sim_time_ns=0, position_ned=np.zeros(3), velocity_ned=np.zeros(3),
                    roll=0.0, pitch=0.0, yaw=0.0, angular_rate_body=np.zeros(3),
                    vert_vz_est=vz_est, z_off_est=z_off,
                    vert_vz_imu=vz_imu, zoff_w=zoff_w)


def _sp(s=None) -> Setpoint:
    return Setpoint(sim_time_ns=0, accel_ned=np.array([1.0, 0.0, 0.0]), yaw=0.0,
                    velocity_ned=np.array([0.0, 0.0, 0.0]), gate_pd_scale=s)


# ---------------------------------------------------------------------------
# A-1: terminal brake on the IMU-only washout vz
# ---------------------------------------------------------------------------
def test_a1_terminal_brake_consumes_imu_vz_not_vz_lp():
    """THE dip fix (ticks 46-57): fiction vz_lp says CLIMB (-1.0) while the IMU channel says SINK
    (+1.0). Flag ON at s=0.5: the brake must ARREST the sink (thrust ABOVE hover by
    kd*(1+kb*(1-s))*vz_imu = 0.06*1.5*1.0 = +0.09), not amplify the fiction below hover."""
    hover = _ctrl().hover_thrust
    on = _ctrl(gate_pd_brake_imu_vz=True)
    thrust_on = on.command(_nav(vz_est=-1.0, vz_imu=+1.0), _sp(s=0.5)).thrust
    assert thrust_on == pytest.approx(hover + 0.06 * 1.5 * 1.0, abs=1e-9)

    off = _ctrl()   # flag OFF: today's law -- the fiction cuts thrust below hover
    thrust_off = off.command(_nav(vz_est=-1.0, vz_imu=+1.0), _sp(s=0.5)).thrust
    assert thrust_off == pytest.approx(hover + 0.06 * 1.5 * (-1.0), abs=1e-9)


def test_a1_far_field_still_uses_vz_lp():
    """s = 1.0 (far) => the vz_lp path EXACTLY even with the flag ON (terminal-zone-only)."""
    hover = _ctrl().hover_thrust
    on = _ctrl(gate_pd_brake_imu_vz=True)
    thrust = on.command(_nav(vz_est=-1.0, vz_imu=+1.0), _sp(s=1.0)).thrust
    assert thrust == pytest.approx(hover + 0.06 * (-1.0), abs=1e-9)


def test_a1_no_terminal_taper_never_uses_imu():
    """gate_pd_terminal OFF forces s=1.0 regardless of the setpoint scale => vz_lp path."""
    on = _ctrl(gate_pd_terminal=False, gate_pd_brake_imu_vz=True)
    hover = on.hover_thrust
    thrust = on.command(_nav(vz_est=-1.0, vz_imu=+1.0), _sp(s=0.2)).thrust
    assert thrust == pytest.approx(hover + 0.06 * (-1.0), abs=1e-9)


def test_a1_nan_imu_vz_falls_back_to_vz_lp():
    """vert_vz_imu NaN (estimator off/unseeded) => the vz_lp path, never a NaN thrust."""
    hover = _ctrl().hover_thrust
    on = _ctrl(gate_pd_brake_imu_vz=True)
    thrust = on.command(_nav(vz_est=-1.0, vz_imu=float("nan")), _sp(s=0.5)).thrust
    assert np.isfinite(thrust)
    assert thrust == pytest.approx(hover + 0.06 * 1.5 * (-1.0), abs=1e-9)


# ---------------------------------------------------------------------------
# A-3: terminal-zone negative-pull bound
# ---------------------------------------------------------------------------
def test_a3_negative_pull_bounded_at_flag_value():
    """s=0: a railed garbage vz (-2.5, x2 boost => -0.30 unbounded) may pull at most 0.08 below
    hover with the bound on."""
    hover = _ctrl().hover_thrust
    on = _ctrl(gate_pd_brake_neg_max=0.08)
    thrust_on = on.command(_nav(vz_est=-2.5), _sp(s=0.0)).thrust
    assert thrust_on == pytest.approx(hover - 0.08, abs=1e-9)
    off = _ctrl()
    thrust_off = off.command(_nav(vz_est=-2.5), _sp(s=0.0)).thrust
    assert thrust_off < thrust_on   # unbounded pulls far deeper (into the collective floor clip)


def test_a3_arrest_direction_uncapped():
    """The sink-ARREST direction (term_damp > 0) is never capped -- the bound only guards the
    below-hover pull."""
    hover = _ctrl().hover_thrust
    on = _ctrl(gate_pd_brake_neg_max=0.08)
    thrust = on.command(_nav(vz_est=+2.5), _sp(s=0.0)).thrust
    assert thrust > hover + 0.2     # ~hover + 0.30 (modulo the hi clip)


def test_a3_far_field_unbounded():
    """s=1.0 (outside the terminal zone): the bound does not apply -- the far-field brake is
    exactly today's kd*vz_lp, unbounded (it was flying fine pre-R2-1)."""
    hover = _ctrl().hover_thrust
    on = _ctrl(gate_pd_brake_neg_max=0.08)
    thrust = on.command(_nav(vz_est=-2.5), _sp(s=1.0)).thrust
    assert thrust == pytest.approx(hover + 0.06 * (-2.5), abs=1e-9)   # -0.15 < -0.08: not capped


# ---------------------------------------------------------------------------
# R2-2b: the q-release on the gate-PD position term
# ---------------------------------------------------------------------------
def test_q_release_fires_on_railed_state_with_collapsed_weight():
    """THE balloon unpin (ticks 88-105 signature): |z_off| = 6 (railed 2x past the clip) on
    zoff_w = 0.05 (collapsed) => term_gate fades to q_floor: -0.04*0.3*clip(-6,+/-3) = +0.036."""
    hover = _ctrl().hover_thrust
    on = _ctrl(gate_pd_zoff_trust_taper=True)
    thrust = on.command(_nav(z_off=-6.0, zoff_w=0.05), _sp(s=1.0)).thrust
    assert thrust == pytest.approx(hover + 0.04 * 0.3 * 3.0, abs=1e-9)


def test_q_stays_one_on_large_measured_offset_at_good_weight():
    """THE gate-2 protection pin (commander-required): a large offset the VISION is asserting at
    decent weight keeps FULL position authority -- q = 1.0, term_gate = +0.12."""
    hover = _ctrl().hover_thrust
    on = _ctrl(gate_pd_zoff_trust_taper=True)
    thrust = on.command(_nav(z_off=-6.0, zoff_w=0.9), _sp(s=1.0)).thrust
    assert thrust == pytest.approx(hover + 0.04 * 1.0 * 3.0, abs=1e-9)


def test_q_stays_one_below_rail_threshold():
    """|z_off| < zoff_trust_m (3.4): the in-band state keeps full authority even at low weight
    (low weight alone is the seeker/estimator's problem, not a consumption-time distrust)."""
    hover = _ctrl().hover_thrust
    on = _ctrl(gate_pd_zoff_trust_taper=True)
    thrust = on.command(_nav(z_off=-3.0, zoff_w=0.05), _sp(s=1.0)).thrust
    assert thrust == pytest.approx(hover + 0.04 * 1.0 * 3.0, abs=1e-9)


def test_q_nan_weight_stays_one():
    """zoff_w NaN (absent-marker: soft path off / no latch yet) => q = 1.0, byte-identical."""
    hover = _ctrl().hover_thrust
    on = _ctrl(gate_pd_zoff_trust_taper=True)
    thrust = on.command(_nav(z_off=-6.0, zoff_w=float("nan")), _sp(s=1.0)).thrust
    assert thrust == pytest.approx(hover + 0.04 * 1.0 * 3.0, abs=1e-9)


# ---------------------------------------------------------------------------
# B1: estimator-side propagation bound
# ---------------------------------------------------------------------------
def _est_b1(bound=3.5) -> VerticalEstimator:
    est = VerticalEstimator(use_zoff_filter=True, zoff_prop_bound_m=bound)
    est.seed()
    _burn_bias_capture(est)
    return est


def test_b1_propagation_cannot_grow_past_bound():
    """Dead-reckoning on a railed vz may not push |z_off| past the bound (the -3.2 -> -6.4
    runaway). Drive vz positive (sinking) from a -3.4 lock: unbounded runs away past -4;
    bounded holds at >= -3.5."""
    bounded, unbounded = _est_b1(3.5), _est_b1(None)
    for est in (bounded, unbounded):
        est.latch_offset(offset_z_world=-3.4, obs_age_s=0.0)   # initial lock (measurement)
        for _ in range(200):                                   # ~1.4 s of a_dn = +4 m/s^2
            est.predict(-4.0, IMU_DT)                          # a_up = -4 => sinking, vz > 0
    assert unbounded.z_off < -4.0, "control twin must run away (the observed failure)"
    assert bounded.z_off >= -3.5 - 1e-9, f"bound violated: {bounded.z_off}"
    assert bounded.z_off <= -3.4 + 1e-9   # it did dead-reckon down to the bound, not freeze at lock


def test_b1_shrink_always_allowed():
    """From a beyond-bound state the propagate may still SHRINK |z_off| toward zero (a real climb
    toward a high gate must keep updating the state)."""
    est = _est_b1(3.5)
    est.latch_offset(offset_z_world=-5.0, obs_age_s=0.0)       # measurement beyond the bound: kept
    assert est.z_off == pytest.approx(-5.0)
    for _ in range(200):
        est.predict(+4.0, IMU_DT)                              # a_up = +4 => climbing, vz < 0
    assert est.z_off > -5.0 + 0.05, "shrink toward zero must not be blocked"


def test_b1_growth_blocked_even_beyond_bound():
    """Already past the bound (measured -5.0): dead-reckoning may not grow it FURTHER (max(bound,
    |current|) formulation) -- the state holds until vision moves it."""
    est = _est_b1(3.5)
    est.latch_offset(offset_z_world=-5.0, obs_age_s=0.0)
    for _ in range(200):
        est.predict(-4.0, IMU_DT)                              # sinking: growth direction
    assert est.z_off == pytest.approx(-5.0, abs=1e-9), "propagation-only growth must hold"


def test_b1_measurements_untouched():
    """The CORRECT step is not bounded: an initial lock beyond the bound sets the state exactly,
    and a subsequent alpha correction moves it FURTHER beyond (vision may, dead-reckoning may not)."""
    est = _est_b1(3.5)
    est.latch_offset(offset_z_world=-5.0, obs_age_s=0.0)
    assert est.z_off == pytest.approx(-5.0)
    est.latch_offset(offset_z_world=-6.0, obs_age_s=0.0)       # innov -1.0, in-gate => alpha corr
    assert est.z_off == pytest.approx(-5.0 + 0.4 * (-1.0), abs=1e-6)   # -5.4: beyond bound, allowed


def test_b1_none_is_byte_identical_in_band():
    """bound=None vs bound=3.5 on an in-band trajectory (never approaching the bound): identical
    z_off and vz at every step -- the default path is untouched."""
    a, b = _est_b1(None), _est_b1(3.5)
    for est in (a, b):
        est.latch_offset(offset_z_world=-1.0, obs_age_s=0.0)
    for k in range(300):
        acc = 2.0 * np.sin(k / 17.0)                           # bounded wander, |z_off| stays << 3.4
        a.predict(acc, IMU_DT)
        b.predict(acc, IMU_DT)
        assert a.z_off == b.z_off and a.vz == b.vz
    for est in (a, b):
        est.latch_offset(offset_z_world=-0.5, obs_age_s=0.1)
    assert a.z_off == b.z_off and a.vz == b.vz


# ---------------------------------------------------------------------------
# A-1 estimator side: the parallel IMU-only washout state
# ---------------------------------------------------------------------------
def test_vz_imu_ignores_vision_corrections():
    """vz_imu is the pure A24 recurrence: a stream of descending-toward-gate innovations drives
    the FILTER vz (beta corrections) but must leave vz_imu at ~0 (no accel => no rate)."""
    est = VerticalEstimator(use_zoff_filter=True)
    est.seed()
    _burn_bias_capture(est)
    est.latch_offset(offset_z_world=3.0, obs_age_s=0.0)
    off = 3.0
    for _ in range(10):                                        # shrinking offsets = "descending"
        for _ in range(14):
            est.predict(0.0, IMU_DT)                           # zero accel between poses
        off -= 0.15
        est.latch_offset(offset_z_world=off, obs_age_s=0.0)
    assert abs(est.vz) > 0.05, "beta corrections must move the filter vz (sanity)"
    assert abs(est.vz_imu) < 1e-6, "vz_imu must be untouched by vision"


def test_vz_imu_tracks_accel_with_leak():
    """vz_imu integrates a_dn with the washout leak (A24 semantics: leak ALWAYS on, even while
    innovations are fresh -- unlike the filter vz, which suspends its leak when fresh)."""
    est = VerticalEstimator(use_zoff_filter=True)
    est.seed()
    _burn_bias_capture(est)
    n = 140                                                    # 1.0 s of a_dn = +1 m/s^2
    for _ in range(n):
        est.predict(-1.0, IMU_DT)
    # closed form: vz = sum a*dt*leak^k -> tau*(1-exp(-t/tau))*a for constant a
    expect = 2.0 * (1.0 - np.exp(-1.0 / 2.0)) * 1.0
    assert est.vz_imu == pytest.approx(expect, rel=0.02)
    assert est.vz_imu > 0.0                                    # sinking sign convention (down +)


def test_vz_imu_nan_until_seeded():
    est = VerticalEstimator()
    assert np.isnan(est.vz_imu)
    est.seed()
    assert est.vz_imu == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# byte-identity + profile pins
# ---------------------------------------------------------------------------
def test_defaults_are_off():
    c = Controller()
    assert c.gate_pd_brake_imu_vz is False
    assert c.gate_pd_brake_neg_max is None
    assert c.gate_pd_zoff_trust_taper is False
    assert VerticalEstimator().zoff_prop_bound_m is None


def test_flags_off_byte_identical_thrust():
    """All R2-2 flags OFF: supplying vert_vz_imu / zoff_w on the NavState changes NOTHING --
    the pre-R2-2 law byte-for-byte across a sweep of states."""
    for s in (None, 1.0, 0.5, 0.0):
        for vz_est, z_off in ((-1.0, 0.0), (2.5, -6.0), (-2.5, 2.0), (0.3, 3.0)):
            base = _ctrl().command(_nav(vz_est=vz_est, z_off=z_off), _sp(s=s)).thrust
            loaded = _ctrl().command(
                _nav(vz_est=vz_est, z_off=z_off, vz_imu=9.9, zoff_w=0.01), _sp(s=s)).thrust
            assert loaded == base, (s, vz_est, z_off)


def test_navstate_defaults_are_absent_markers():
    ns = _nav()
    assert np.isnan(ns.vert_vz_imu)
    assert np.isnan(ns.zoff_w)


def test_vq2_profile_sets_r2_2_package():
    from racer.deploy_profile import vq1_case_a, vq2_case_c
    ov = vq2_case_c().controller_overrides
    assert ov["gate_pd_brake_imu_vz"] is True
    assert ov["gate_pd_brake_neg_max"] == pytest.approx(0.08)
    assert ov["gate_pd_zoff_trust_taper"] is True
    vov = vq2_case_c().nav_config.vertical_estimator_overrides
    assert vov["zoff_prop_bound_m"] == pytest.approx(3.5)
    vq1_ov = vq1_case_a().controller_overrides or {}
    for k in ("gate_pd_brake_imu_vz", "gate_pd_brake_neg_max", "gate_pd_zoff_trust_taper"):
        assert k not in vq1_ov, "VQ1 must never see the R2-2 package"
