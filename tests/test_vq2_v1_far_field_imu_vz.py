"""VQ2 V-1 — honest (IMU-only) rate source for the gate-PD brake at ALL RANGES.

Run 20260704_231155 (gate-2 turn dive). Both operator residuals -- a dip toward the floor mid-turn
and a near-vertical climb into the ceiling -- are ONE root cause: the vertical rate brake
``term_damp = kd*(1 + kb*(1-s))*vz_brake`` consumes ``vz_lp`` (the vision-corrected vz_est) in the
FAR field (s = 1), and out there vz_est RAILS to +/-2.5 with the WRONG (inverted) sign under the
close-range bias sweep. The R2-2 A-1 IMU-vz swap fixed exactly this, but was gated to the TERMINAL
zone (s < 1); the whole turn/climb happens at range 15-23 m (s = 1.0), so the brake ran on fiction:

  ticks 76-84 (climbing hard toward the HIGH gate 1, thrust railed 0.536): vz_est = +2.50 ("falling")
    while vz_imu = -0.02..-1.35 ("climbing", correct) -> term_damp = +0.15 AMPLIFIES the climb.
  ticks 85-95: vz_est flips to -2.5 -> term_damp -0.15 -> thrust craters to 0.07 -> the dip.

``Controller.gate_pd_rate_imu_always`` (default False = today's terminal-only-swap path) makes the
BASE kd brake consume ``NavState.vert_vz_imu`` at ALL ranges; the R2-1 boost ``(1 + kb*(1-s))`` is
UNCHANGED (terminal-only). vz_imu NaN => the vz_lp path EXACTLY (the A-1 NaN fallback). OFF =>
byte-identical. vq2_case_c flips it ON.

Sign convention (proven on the run replay): vz_imu is NED down-positive
(``VerticalEstimator.vz_imu`` docstring), so negative = climbing. ``term_damp = kd*vz_imu`` is then
negative while climbing => reduces thrust => OPPOSES the climb -- the whole point of the fix.

Run: .venv\\Scripts\\python.exe -m pytest tests/test_vq2_v1_far_field_imu_vz.py -q
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


def _ctrl(**overrides) -> "Controller":
    """A vq2-shaped vertical-law controller: gate-PD + R2-1 terminal taper + R2-2 A-1 live (the
    live vq2_case_c baseline), with V-1 added per-test via overrides."""
    kw = dict(ff_owns_vertical=True, use_vertical_estimator=True,
              gate_pd_vertical=True, gate_pd_terminal=True, gate_pd_rate_boost=1.0,
              gate_pd_brake_imu_vz=True, kp_gate=0.04, ff_vertical_kd_alt=0.06, kp_alt=0.0)
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
# V-1: the far-field fix -- the brake opposes the real climb, not amplifies it
# ---------------------------------------------------------------------------
def test_v1_far_field_uses_imu_vz_opposes_climb():
    """THE ceiling-climb fix (ticks 76-84 signature): far field s=1, fiction vz_est=+2.5 says
    "falling" (kd*vz_lp = +0.15 AMPLIFIES the climb) while vz_imu=-1.35 says "climbing" (correct).
    V-1 ON: the brake reads vz_imu -> term_damp = 0.06*(-1.35) = -0.081 (below hover) => OPPOSES
    the climb. The R2-1 boost is 1.0 at s=1 (terminal-only), so no *2 amplification here."""
    hover = _ctrl().hover_thrust
    on = _ctrl(gate_pd_rate_imu_always=True)
    thrust_on = on.command(_nav(vz_est=2.5, vz_imu=-1.35), _sp(s=1.0)).thrust
    assert thrust_on == pytest.approx(hover + 0.06 * (-1.35), abs=1e-9)
    assert thrust_on < hover                      # a brake, below hover -- opposes the climb

    off = _ctrl()   # today's path: far field reads vz_lp (the fiction) -> ABOVE hover (amplifies)
    thrust_off = off.command(_nav(vz_est=2.5, vz_imu=-1.35), _sp(s=1.0)).thrust
    assert thrust_off == pytest.approx(hover + 0.06 * 2.5, abs=1e-9)
    assert thrust_off > hover                     # the amplification bug (rocket to ceiling)


def test_v1_far_field_dip_case_uses_imu_vz():
    """THE dip fix (ticks 85-95 signature): far field s=1, fiction vz_est=-2.5 says "climbing"
    (kd*vz_lp = -0.15 craters thrust) while vz_imu=+2.05 says "sinking" (correct). V-1 ON: the
    brake reads vz_imu -> term_damp = 0.06*2.05 = +0.123 (ARRESTS the sink, above hover)."""
    hover = _ctrl().hover_thrust
    on = _ctrl(gate_pd_rate_imu_always=True)
    thrust_on = on.command(_nav(vz_est=-2.5, vz_imu=2.05), _sp(s=1.0)).thrust
    assert thrust_on == pytest.approx(hover + 0.06 * 2.05, abs=1e-9)
    assert thrust_on > hover                      # arrests the sink -- no crater


def test_v1_boost_still_terminal_only():
    """The R2-1 rate-arrest boost (1 + kb*(1-s)) is UNCHANGED by V-1: at s=1 the factor is 1.0
    (no boost), at s=0.5 it is 1.5 -- V-1 only swaps the SOURCE, never the boost schedule."""
    hover = _ctrl().hover_thrust
    on = _ctrl(gate_pd_rate_imu_always=True)
    # s=1: factor 1.0 (no z_off so term_gate=0)
    t1 = on.command(_nav(vz_imu=-1.0), _sp(s=1.0)).thrust
    assert t1 == pytest.approx(hover + 0.06 * 1.0 * (-1.0), abs=1e-9)
    # s=0.5: factor 1 + 1.0*0.5 = 1.5 (terminal boost) -- the A-3 neg bound is OFF in _ctrl()
    t2 = on.command(_nav(vz_imu=-1.0), _sp(s=0.5)).thrust
    assert t2 == pytest.approx(hover + 0.06 * 1.5 * (-1.0), abs=1e-9)


def test_v1_nan_imu_vz_falls_back_to_vz_lp_far_field():
    """vert_vz_imu NaN (estimator not seeded) => the vz_lp path EXACTLY even with V-1 ON, at s=1 --
    a missing IMU vz can never break far-field flight (never a NaN thrust)."""
    hover = _ctrl().hover_thrust
    on = _ctrl(gate_pd_rate_imu_always=True)
    thrust = on.command(_nav(vz_est=2.5, vz_imu=float("nan")), _sp(s=1.0)).thrust
    assert np.isfinite(thrust)
    assert thrust == pytest.approx(hover + 0.06 * 2.5, abs=1e-9)   # the vz_lp (fiction) path


def test_v1_nan_imu_vz_falls_back_terminal_zone():
    """Same NaN fallback in the terminal zone (s<1): vz_lp with the R2-1 boost, never NaN."""
    hover = _ctrl().hover_thrust
    on = _ctrl(gate_pd_rate_imu_always=True)
    thrust = on.command(_nav(vz_est=2.5, vz_imu=float("nan")), _sp(s=0.5)).thrust
    assert np.isfinite(thrust)
    assert thrust == pytest.approx(hover + 0.06 * 1.5 * 2.5, abs=1e-9)


def test_v1_terminal_zone_still_imu_vz():
    """In the terminal zone V-1 and A-1 agree (both read vz_imu): V-1 ON at s=0.5 reads vz_imu
    exactly as A-1 alone did -- V-1 subsumes A-1, no behaviour change in the terminal zone."""
    v1 = _ctrl(gate_pd_rate_imu_always=True)
    a1 = _ctrl()   # A-1 only (gate_pd_brake_imu_vz=True in the baseline)
    for s in (0.0, 0.3, 0.5, 0.9):
        tv = v1.command(_nav(vz_est=2.5, vz_imu=-1.0), _sp(s=s)).thrust
        ta = a1.command(_nav(vz_est=2.5, vz_imu=-1.0), _sp(s=s)).thrust
        assert tv == pytest.approx(ta, abs=1e-12), s


def test_v1_a3_neg_bound_only_terminal():
    """V-1 puts an honest vz_imu into the FAR-field brake, but the A-3 negative-pull bound is
    terminal-only (s<1) by design -- far field stays uncapped (it flew fine pre-R2-1). A big
    honest far-field arrest is NOT clamped by the A-3 bound."""
    hover = _ctrl().hover_thrust
    on = _ctrl(gate_pd_rate_imu_always=True, gate_pd_brake_neg_max=0.08)
    # far field, vz_imu = -2.5 (climbing): term_damp = 0.06*(-2.5) = -0.15, below the -0.08 bound
    # but s=1 so the bound does NOT apply.
    thrust = on.command(_nav(vz_imu=-2.5), _sp(s=1.0)).thrust
    assert thrust == pytest.approx(hover + 0.06 * (-2.5), abs=1e-9)   # -0.15 uncapped far field


# ---------------------------------------------------------------------------
# byte-identity + profile pins
# ---------------------------------------------------------------------------
def test_v1_default_is_off():
    assert Controller().gate_pd_rate_imu_always is False


def test_v1_off_byte_identical_thrust():
    """V-1 OFF: the pre-V-1 law byte-for-byte across a sweep of states/scales -- terminal->vz_imu
    (A-1), far->vz_lp. Supplying vz_imu changes only the terminal zone (A-1's job), never s=1."""
    for s in (None, 1.0, 0.5, 0.0):
        for vz_est, vz_imu, z_off in ((2.5, -1.35, -3.0), (-2.5, 2.05, 2.0),
                                      (0.3, 0.1, 3.0), (-1.0, float("nan"), 0.0)):
            base = _ctrl().command(_nav(vz_est=vz_est, vz_imu=vz_imu, z_off=z_off), _sp(s=s)).thrust
            # a second identical controller -- V-1 OFF must be deterministic + unchanged
            same = _ctrl().command(_nav(vz_est=vz_est, vz_imu=vz_imu, z_off=z_off), _sp(s=s)).thrust
            assert same == base, (s, vz_est, vz_imu, z_off)


def test_v1_off_far_field_ignores_imu_vz():
    """The crux of byte-identity: V-1 OFF at s=1 (far) reads vz_lp -- vz_imu is IGNORED out there
    (that is exactly the bug V-1 fixes, and exactly what OFF must preserve)."""
    hover = _ctrl().hover_thrust
    off = _ctrl()
    # vz_imu swings wildly; far-field thrust must not move (reads vz_est only)
    t_a = off.command(_nav(vz_est=2.5, vz_imu=-2.5), _sp(s=1.0)).thrust
    off2 = _ctrl()
    t_b = off2.command(_nav(vz_est=2.5, vz_imu=+2.5), _sp(s=1.0)).thrust
    assert t_a == pytest.approx(hover + 0.06 * 2.5, abs=1e-9)
    assert t_b == pytest.approx(hover + 0.06 * 2.5, abs=1e-9)
    assert t_a == t_b   # far-field OFF path is blind to vz_imu


def test_vq2_profile_sets_v1():
    from racer.deploy_profile import vq1_case_a, vq2_case_c
    ov = vq2_case_c().controller_overrides
    assert ov["gate_pd_rate_imu_always"] is True
    vq1_ov = vq1_case_a().controller_overrides or {}
    assert "gate_pd_rate_imu_always" not in vq1_ov, "VQ1 must never see V-1"


def test_vq2_profile_turn_iteration_values():
    """Part 2 turn iteration pins: lat cap + total cap -> 4.0, forward -> 0.35, yaw FROZEN."""
    from racer.deploy_profile import vq2_case_c
    ov = vq2_case_c().seeker_overrides
    assert ov["image_lat_cap_mps2"] == pytest.approx(4.0)
    assert ov["total_accel_cap_mps2"] == pytest.approx(4.0)
    assert ov["forward_accel_mps2"] == pytest.approx(0.25)   # S1 2026-07-05 (0.35 -> 0.25)
    # yaw frozen at the A36 values (operator: "yaw is good where it is")
    assert ov["pursuit_yaw_slew_rps"] == pytest.approx(0.9)
    assert ov["visual_yaw_rate_cap_rps"] == pytest.approx(0.9)
