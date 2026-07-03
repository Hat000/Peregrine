"""VQ2 A27 — decouple forward propulsion from the vertical bob (2026-07-03).

THE TRAP (live VQ2, run 20260703_002244): the gate-seeker's FORWARD pursuit is correct -- it
commands/achieves the -7deg forward tilt, +1.2 m/s^2 demand -- but a ~2 Hz VERTICAL bob rail-slams
the collective down to its ``alt_thrust_lo`` floor (0.05) repeatedly. Forward aerodynamic force
scales with the collective magnitude, so the forward push collapses from +1.02 to +0.14 m/s^2 on
every low-thrust half-cycle: the drone cannot translate forward, stuck between spawn and gate 1,
even though the pursuit/forward path itself is untouched and correct.

THE FIX -- two levers, both vq2_case_c-ONLY, both default-off (VQ1 / case-A byte-identical):

  Lever 1: ``alt_thrust_lo`` 0.05 -> 0.15 in ``vq2_case_c().controller_overrides`` (hover is 0.2656):
           a bob can no longer zero out the collective, so forward thrust never collapses near zero.
  Lever 2: ``alt_thrust_slew_max_rps2``-style limiter -- new Controller field
           ``alt_thrust_slew_per_s`` (None => off), a stateful per-tick slew limit on the FINAL
           commanded collective (|thrust - prev| <= slew*dt), mirroring ``body_rate_slew_max_rps2``.
           vq2_case_c sets it to 2.0 (a 0.15->0.6 swing takes ~0.22 s instead of one tick).

Both thread through the existing ``DeployProfile.controller_overrides`` seam, splatted into
``make_seeker_controller(**controller_overrides)`` -- no new seam needed.

[VQ2 A27, 2026-07-03]
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from racer.contracts import ControlMode, NavState, Setpoint  # noqa: E402
from racer.controller import Controller  # noqa: E402
from racer.deploy_profile import get_profile, vq1_case_a, vq2_case_c  # noqa: E402
from racer.gate_seeker import GateSeeker, GateSeekerConfig, make_seeker_controller  # noqa: E402


def _nav(sim_time_ns=0, z=-2.5, vz=0.0):
    return NavState(
        sim_time_ns=sim_time_ns,
        position_ned=np.array([0.0, 0.0, z]),
        velocity_ned=np.array([0.0, 0.0, vz]),
        roll=0.0,
        pitch=0.0,
        yaw=0.0,
        time_since_vision_update_s=0.05,
    )


def _sp(sim_time_ns=0, vz_t=0.0):
    """A decoupled-controller setpoint that drives the alt-hold via a commanded vz (no accel_ned,
    no ff_owns_vertical needed) -- exercises the legacy alt-hold branch directly."""
    return Setpoint(sim_time_ns=sim_time_ns, velocity_ned=np.array([0.0, 0.0, vz_t]), yaw=0.0,
                    launch_ramp=1.0)


def _make(**overrides):
    params = dict(mode=ControlMode.BODY_RATE, decoupled=True, hover_thrust=0.2656,
                  kp_alt=2.0, kd_alt=3.0, alt_thrust_lo=0.05, alt_thrust_hi=0.6, tilt_comp=True)
    params.update(overrides)
    return Controller(**params)


# ===========================================================================
# 1. Controller field defaults: alt_thrust_slew_per_s ships OFF
# ===========================================================================
def test_controller_default_alt_thrust_slew_is_none():
    """The Controller dataclass default keeps ``alt_thrust_slew_per_s`` at None -- OFF, byte-identical
    unless a caller opts in (VQ1 / case-A / any bare Controller())."""
    assert Controller().alt_thrust_slew_per_s is None


def test_make_seeker_controller_default_alt_thrust_lo_is_original():
    """The factory default must stay the ORIGINAL floor (0.05) and the slew OFF -- the A27 fix is
    OPT-IN via vq2_case_c's overrides only, never a default change."""
    c = make_seeker_controller()
    assert c.alt_thrust_lo == 0.05
    assert c.alt_thrust_slew_per_s is None


# ===========================================================================
# 2. Slew OFF => byte-identical thrust output (no-op path)
# ===========================================================================
def test_thrust_slew_off_is_byte_identical():
    """With ``alt_thrust_slew_per_s=None`` the commanded thrust equals the un-slewed value on EVERY
    tick -- no state kept, no clamping -- across a rail-to-rail step, so VQ1 / case-A is
    byte-identical."""
    c_off = _make(alt_thrust_slew_per_s=None)
    c_ref = _make(alt_thrust_slew_per_s=None)
    dt_ns = int(1e9 / 30.0)
    # tick 0: at rest (no vz demand) -> some mid thrust
    nav0, sp0 = _nav(0, z=-2.5), _sp(0, vz_t=0.0)
    cmd0 = c_off.command(nav0, sp0)
    # tick 1: a huge vz_t demand -> the alt-hold would want a very different thrust (a big step)
    nav1, sp1 = _nav(dt_ns, z=-2.5), _sp(dt_ns, vz_t=50.0)
    cmd1 = c_off.command(nav1, sp1)
    # a fresh controller (never ticked) must produce the SAME thrust for tick 1's inputs alone,
    # proving the OFF limiter carried no state forward from tick 0.
    ref1 = c_ref.command(nav1, sp1)
    assert cmd1.thrust == pytest.approx(ref1.thrust)
    # sanity: the step really was large (otherwise this test would not exercise anything)
    assert abs(cmd1.thrust - cmd0.thrust) > 0.05


def test_thrust_slew_off_matches_pre_change_formula():
    """With the limiter OFF, the thrust equals exactly ``clip(hover + kp_alt*(z-z_t) + kd_alt*(vz-vz_t),
    lo, hi)`` -- i.e. inserting ``_apply_thrust_slew`` changed nothing on the OFF path (byte-identical
    to the pre-A27 formula)."""
    c = _make(alt_thrust_slew_per_s=None)
    nav, sp = _nav(0, z=-2.5, vz=0.3), _sp(0, vz_t=-1.0)
    cmd = c.command(nav, sp)
    expected = c.hover_thrust + c.kp_alt * 0.0 + c.kd_alt * (0.3 - (-1.0))
    expected = float(np.clip(expected, c.alt_thrust_lo, c.alt_thrust_hi))
    # tilt_comp divides by cos(0)*cos(0) = 1.0, so no change here (roll=pitch=0)
    assert cmd.thrust == pytest.approx(expected)


# ===========================================================================
# 3. Slew ON => bounds the per-tick change; rail-to-rail step takes multiple ticks
# ===========================================================================
def test_thrust_slew_on_bounds_the_per_tick_change():
    """With the limiter ON, a rail-to-rail thrust step is bounded to ``slew*dt`` this tick."""
    slew = 2.0
    dt = 1.0 / 30.0
    dt_ns = int(dt * 1e9)
    c = _make(alt_thrust_slew_per_s=slew, alt_thrust_lo=0.15, alt_thrust_hi=0.6)
    # tick 0: drive the collective to the LOW rail (large positive vz -> sink -> low thrust... use a
    # large NEGATIVE vz_t demand with vz=0 so kd_alt*(vz - vz_t) is large positive; instead directly
    # force a low-thrust tick by asking for a big climb-rate error the other way). Simplify: use kp_alt
    # position error to hit a known low value first.
    nav0, sp0 = _nav(0, z=-2.5, vz=0.0), _sp(0, vz_t=10.0)   # kd_alt*(0-10) very negative -> floors
    cmd0 = c.command(nav0, sp0)
    assert cmd0.thrust == pytest.approx(0.15)   # first tick unclamped -> hits the floor directly
    # tick 1: now demand the HIGH rail (vz_t very negative -> kd_alt*(vz - vz_t) very positive)
    nav1, sp1 = _nav(dt_ns, z=-2.5, vz=0.0), _sp(dt_ns, vz_t=-10.0)
    cmd1 = c.command(nav1, sp1)
    max_step = slew * dt + 1e-9
    step = abs(cmd1.thrust - cmd0.thrust)
    assert step <= max_step, f"slew must bound the step to {max_step:.4f}, got {step:.4f}"
    # sanity: the RAW (un-slewed) rail-to-rail step would be much larger than the bound
    raw = _make(alt_thrust_slew_per_s=None, alt_thrust_lo=0.15, alt_thrust_hi=0.6)
    raw0 = raw.command(nav0, sp0)
    raw1 = raw.command(nav1, sp1)
    raw_step = abs(raw1.thrust - raw0.thrust)
    assert raw_step > max_step, "sanity: the un-slewed rail-to-rail step should exceed the slew bound"
    assert raw_step == pytest.approx(0.45)   # 0.6 - 0.15, the full rail-to-rail swing


def test_thrust_slew_multi_tick_ramp_reaches_target():
    """A full floor->ceiling swing under the slew ramps up over successive ticks and eventually
    reaches (clamps at) the target rather than jumping there in one tick."""
    slew = 2.0
    dt = 1.0 / 30.0
    dt_ns = int(dt * 1e9)
    c = _make(alt_thrust_slew_per_s=slew, alt_thrust_lo=0.15, alt_thrust_hi=0.6)
    t_ns = 0
    nav, sp = _nav(t_ns, z=-2.5, vz=0.0), _sp(t_ns, vz_t=10.0)
    prev = c.command(nav, sp).thrust
    assert prev == pytest.approx(0.15)
    thrusts = [prev]
    for i in range(1, 20):
        t_ns = int(i * dt_ns)
        nav_i, sp_i = _nav(t_ns, z=-2.5, vz=0.0), _sp(t_ns, vz_t=-10.0)
        cur = c.command(nav_i, sp_i).thrust
        step = abs(cur - thrusts[-1])
        assert step <= slew * dt + 1e-9
        thrusts.append(cur)
    # after enough ticks the ramp reaches the high rail (0.6)
    assert thrusts[-1] == pytest.approx(0.6)
    # and it was monotonically non-decreasing while ramping (no overshoot/oscillation introduced)
    assert all(b >= a - 1e-9 for a, b in zip(thrusts, thrusts[1:]))


def test_thrust_slew_first_tick_passes_through_unclamped():
    """The FIRST command (no prior) passes through unclamped -- the limiter only bounds CHANGES, so a
    fresh controller's first thrust equals the un-slewed one, exactly like the body-rate slew."""
    c_on = _make(alt_thrust_slew_per_s=2.0)
    c_off = _make(alt_thrust_slew_per_s=None)
    nav, sp = _nav(0, z=-2.5, vz=0.3), _sp(0, vz_t=-1.0)
    cmd_on = c_on.command(nav, sp)
    cmd_off = c_off.command(nav, sp)
    assert cmd_on.thrust == pytest.approx(cmd_off.thrust)


def test_thrust_slew_state_independent_of_body_rate_slew_state():
    """The thrust-slew state (``_prev_thrust``/``_prev_thrust_t_ns``) is tracked separately from the
    body-rate slew state -- enabling one does not perturb the other's byte-identical OFF path."""
    c = _make(alt_thrust_slew_per_s=2.0, body_rate_slew_max_rps2=None)
    nav, sp = _nav(0), _sp(0, vz_t=1.0)
    cmd = c.command(nav, sp)
    assert c._prev_thrust is not None
    assert c._prev_body_rate is None   # body-rate slew stayed OFF/untouched


# ===========================================================================
# 4. DEPLOY PROFILE seam: alt_thrust_lo + alt_thrust_slew_per_s carried for vq2_case_c only
# ===========================================================================
def test_deploy_profile_vq2_case_c_carries_raised_floor_and_slew():
    ov = get_profile("vq2_case_c").controller_overrides
    assert ov["alt_thrust_lo"] == 0.15
    assert ov["alt_thrust_slew_per_s"] == 2.0
    ov2 = vq2_case_c().controller_overrides
    assert ov2["alt_thrust_lo"] == 0.15
    assert ov2["alt_thrust_slew_per_s"] == 2.0


def test_deploy_profile_vq1_case_a_has_no_overrides():
    """vq1_case_a carries NO controller_overrides at all (None) -- so it can never pick up the raised
    floor or the slew; the Controller/factory defaults (0.05 / None) apply untouched."""
    assert get_profile("vq1_case_a").controller_overrides is None
    assert vq1_case_a().controller_overrides is None


def test_make_seeker_controller_threads_a27_overrides_for_vq2_case_c_only():
    """A controller built from the vq2_case_c profile has the raised floor (0.15) AND the thrust slew
    (2.0); one from vq1_case_a (no overrides) keeps the original floor (0.05) + slew OFF -- VQ1 /
    case-A byte-identical."""
    c_vq2 = make_seeker_controller(**(get_profile("vq2_case_c").controller_overrides or {}))
    assert c_vq2.alt_thrust_lo == 0.15
    assert c_vq2.alt_thrust_slew_per_s == 2.0
    c_vq1 = make_seeker_controller(**(get_profile("vq1_case_a").controller_overrides or {}))
    assert c_vq1.alt_thrust_lo == 0.05
    assert c_vq1.alt_thrust_slew_per_s is None


def test_gate_seeker_built_with_profile_controller_has_a27_fix():
    """End-to-end of the wiring: a GateSeeker handed a controller built from the vq2_case_c overrides
    carries the raised floor + slew; the default seeker (no overrides) does not."""
    fixed = GateSeeker(config=GateSeekerConfig(),
                       controller=make_seeker_controller(
                           **(get_profile("vq2_case_c").controller_overrides or {})))
    default = GateSeeker(config=GateSeekerConfig())   # default factory controller
    assert fixed.controller.alt_thrust_lo == 0.15
    assert fixed.controller.alt_thrust_slew_per_s == 2.0
    assert default.controller.alt_thrust_lo == 0.05
    assert default.controller.alt_thrust_slew_per_s is None
