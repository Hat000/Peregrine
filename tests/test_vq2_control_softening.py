"""VQ2 seeker control-softening — de-saturate the A11 egress->pursuit handoff overshoot + bang-bang.

THE TRAP (live VQ2 A11, 2026-06-30): the seeker's stiff attitude loop (``make_seeker_controller``
kp_att=10) versus the ``pursuit_pitch_rate_cap_rps`` = +/-1.5 rad/s cap saturates on ANY attitude
error > ~8.6 deg (1.5/10). The egress->pursuit HANDOFF holds the drone ~-18 deg nose-down through
egress, then pursuit wants ~-5 deg cruise — a ~13 deg error that, times kp_att=10, demands ~2.3 rad/s,
clips to 1.5, and OVERSHOOTS past level into nose-UP -> the +20 deg camera whips up -> gate lost ->
spiral. Plus 15/18 command bursts slam the +/-1.5 clip (bang-bang). THE FIX — two softening levers,
both vq2_case_c-ONLY, both default-off (VQ1 / case-A byte-identical):

  Lever 1: kp_att 10 -> 4 (max non-saturating error = cap/kp = 1.5/4 ~= 21.5 deg, vs 8.6 deg).
  Lever 2: body_rate_slew_max_rps2 = 8.0 — a stateful per-tick slew limit on the commanded body
           rate (|omega - prev| <= slew*dt per axis), so a step ramps over a few ticks (None => off).

Both thread through the new ``DeployProfile.controller_overrides`` seam (parallel to seeker_overrides),
splatted into ``make_seeker_controller(**controller_overrides)`` in rl.fly_rl.make_seeker.

[VQ2 slow-is-smooth, A11, 2026-06-30]
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from racer.contracts import ControlMode, NavState, Setpoint  # noqa: E402
from racer.deploy_profile import get_profile, vq1_case_a, vq2_case_c  # noqa: E402
from racer.gate_seeker import GateSeeker, GateSeekerConfig, make_seeker_controller  # noqa: E402


# ===========================================================================
# 1. DEFAULT controller byte-identity: make_seeker_controller() gains unchanged
# ===========================================================================
def test_make_seeker_controller_default_kp_att_is_original():
    """The factory default must stay the ORIGINAL stiff value (10.0) so the VQ1 / case-A seeker is
    byte-identical — the softening is OPT-IN via overrides, never a default change."""
    c = make_seeker_controller()
    assert c.kp_att == 10.0
    # the slew limiter ships OFF by default (None => no limit => byte-identical body-rate path)
    assert c.body_rate_slew_max_rps2 is None
    # and the rest of the flight-proven gain set is untouched
    assert c.kd_att == 0.30 and c.ff_gain == 1.0 and c.max_body_rate_rps == 4.0


# ===========================================================================
# 2. LEVER 1: softer kp_att de-saturates a ~13deg handoff error
# ===========================================================================
def _tilted_nav(sim_time_ns=0, pitch_deg=-18.0):
    """A NavState carrying a held nose-down spawn/egress tilt (the handoff start attitude)."""
    return NavState(
        sim_time_ns=sim_time_ns,
        position_ned=np.array([0.0, 0.0, -2.5]),
        velocity_ned=np.zeros(3),
        roll=0.0,
        pitch=np.deg2rad(pitch_deg),   # held nose-down -> pursuit commands a nose-up re-level
        yaw=0.0,
        time_since_vision_update_s=0.05,
    )


def _level_cruise_setpoint(sim_time_ns=0):
    """A setpoint asking for a gentle ~-5deg cruise lean (small forward demand) — the pursuit target
    after egress. A small forward velocity along +x produces a slight nose-down cruise attitude."""
    return Setpoint(
        sim_time_ns=sim_time_ns,
        velocity_ned=np.array([3.0, 0.0, 0.0]),   # forward cruise -> gentle nose-down target
        yaw=0.0,
        launch_ramp=1.0,
    )


def _pitch_rate_cmd(controller, nav, sp):
    """The FRD pitch (body-rate Y) component of the controller's decoupled CTBR command."""
    cmd = controller.command(nav, sp)
    assert cmd.mode is ControlMode.BODY_RATE and cmd.body_rate is not None
    return abs(float(cmd.body_rate[1]))


def test_kp_att_4_desaturates_vs_kp_att_10_saturating():
    """For a fixed ~13 deg handoff attitude error, the soft (kp_att=4) controller's commanded pitch
    rate is BELOW the +/-1.5 rad/s pursuit cap (de-saturated), while the stiff (kp_att=10) one is at
    or above it (saturating) — and the soft command is strictly SMALLER. This is the core lever."""
    cap = 1.5
    nav, sp = _tilted_nav(), _level_cruise_setpoint()
    stiff = make_seeker_controller(kp_att=10.0)
    soft = make_seeker_controller(kp_att=4.0)
    r_stiff = _pitch_rate_cmd(stiff, nav, sp)
    r_soft = _pitch_rate_cmd(soft, nav, sp)
    # the stiff loop saturates the cap on this handoff-sized error...
    assert r_stiff >= cap - 1e-9, f"kp_att=10 should saturate the {cap} cap, got {r_stiff:.3f}"
    # ...the soft loop does NOT (it stays under the cap, the de-saturation the fix wants)...
    assert r_soft < cap, f"kp_att=4 should stay below the {cap} cap, got {r_soft:.3f}"
    # ...and the soft command is strictly gentler than the stiff one.
    assert r_soft < r_stiff


def test_make_seeker_controller_override_sets_kp_att():
    """The factory override seam actually lands the gain: make_seeker_controller(kp_att=4.0) builds a
    controller whose attitude-error gain is 4.0 (the documented fix entry point)."""
    assert make_seeker_controller(kp_att=4.0).kp_att == 4.0


# ===========================================================================
# 3. LEVER 2: the body-rate slew limiter (OFF == byte-identical; ON == bounded step)
# ===========================================================================
def test_slew_off_is_byte_identical_to_raw_command():
    """With the slew limiter OFF (None) the commanded body rate equals the raw controller output on
    EVERY tick — no state kept, no clamping — so VQ1 / case-A is byte-identical."""
    c_off = make_seeker_controller(body_rate_slew_max_rps2=None)
    c_ref = make_seeker_controller(body_rate_slew_max_rps2=None)
    # two successive, very different commands; the OFF limiter must never alter either
    dt_ns = int(1e9 / 12.0)   # ~12 Hz
    nav0, sp0 = _tilted_nav(0), _level_cruise_setpoint(0)
    nav1, sp1 = _tilted_nav(dt_ns, pitch_deg=20.0), _level_cruise_setpoint(dt_ns)
    a0 = c_off.command(nav0, sp0).body_rate
    a1 = c_off.command(nav1, sp1).body_rate
    # a fresh controller (never ticked) must produce the same vectors -> OFF kept no state
    b0 = c_ref.command(nav0, sp0).body_rate
    np.testing.assert_array_equal(a0, b0)
    # the second (very different) command is also untouched by the OFF limiter
    assert a1 is not None


def test_slew_on_bounds_the_per_tick_change():
    """With the limiter ON a STEP in the commanded rate is bounded to ``slew*dt`` per axis per tick,
    so 0 -> a large target ramps over successive ticks instead of one whip. Drive a big attitude
    error after a (latched) near-zero command and assert the realized step <= slew*dt."""
    slew = 8.0
    dt = 1.0 / 12.0
    dt_ns = int(dt * 1e9)
    c = make_seeker_controller(kp_att=4.0, body_rate_slew_max_rps2=slew)
    # tick 0: near-level nav + level target -> ~zero command, latched as prev
    nav0 = _tilted_nav(0, pitch_deg=0.0)
    sp0 = Setpoint(sim_time_ns=0, velocity_ned=np.zeros(3), yaw=0.0, launch_ramp=0.0)
    cmd0 = c.command(nav0, sp0)
    # tick 1: a big nose-down tilt -> the raw command would be large; the slew bounds the STEP
    nav1 = _tilted_nav(dt_ns, pitch_deg=-25.0)
    sp1 = _level_cruise_setpoint(dt_ns)
    cmd1 = c.command(nav1, sp1)
    max_step = slew * dt + 1e-9
    step = np.abs(np.asarray(cmd1.body_rate) - np.asarray(cmd0.body_rate))
    assert np.all(step <= max_step), f"slew must bound the per-axis step to {max_step:.4f}, got {step}"
    # the bound actually BIT on the trap axis (pitch): a stiff/un-slewed controller would step more
    raw = make_seeker_controller(kp_att=4.0).command(nav1, sp1)
    raw_step = abs(float(raw.body_rate[1]) - float(cmd0.body_rate[1]))
    assert raw_step > max_step, "sanity: the un-slewed pitch step should exceed the slew bound"


def test_slew_first_tick_passes_through():
    """The FIRST command (no prior) passes through unclamped — the limiter only bounds CHANGES, so a
    fresh controller's first body rate equals the un-slewed one."""
    c_on = make_seeker_controller(kp_att=4.0, body_rate_slew_max_rps2=8.0)
    c_off = make_seeker_controller(kp_att=4.0, body_rate_slew_max_rps2=None)
    nav, sp = _tilted_nav(0), _level_cruise_setpoint(0)
    np.testing.assert_allclose(c_on.command(nav, sp).body_rate, c_off.command(nav, sp).body_rate)


# ===========================================================================
# 4. DEPLOY PROFILE seam: controller_overrides carries the softening for vq2_case_c only
# ===========================================================================
def test_deploy_profile_controller_overrides_wires_the_softening():
    """vq2_case_c carries the softening via the new DeployProfile.controller_overrides seam; vq1_case_a
    leaves it None (explicit, to pin VQ1 byte-identity)."""
    # Subset key-checks (not exact-dict) so the softening pin survives the profile legitimately
    # growing new vq2-only overrides (e.g. the A15b ff_owns_horizontal fix), per the A13 precedent.
    ov = get_profile("vq2_case_c").controller_overrides
    assert ov["kp_att"] == 4.0 and ov["body_rate_slew_max_rps2"] == 8.0
    assert get_profile("vq1_case_a").controller_overrides is None
    # the constructors agree with the named lookup
    ov2 = vq2_case_c().controller_overrides
    assert ov2["kp_att"] == 4.0 and ov2["body_rate_slew_max_rps2"] == 8.0
    assert vq1_case_a().controller_overrides is None


def test_default_deploy_profile_controller_overrides_is_none():
    """The dataclass default keeps existing callers / pickles forward-compatible: controller_overrides
    defaults to None (no overrides == today's controller gains)."""
    from racer.deploy_profile import DeployProfile
    from racer.navigator import NavigatorConfig
    p = DeployProfile(name="x", nav_config=NavigatorConfig(), cmd_rate_scale=1.0, self_localizing=False)
    assert p.controller_overrides is None


# ===========================================================================
# 5. make_seeker_controller-level threading: the softening reaches the controller on case-C only
# ===========================================================================
def _controller_from_profile(profile_name):
    """Mirror rl.fly_rl.make_seeker's controller construction: splat the profile's controller
    overrides into make_seeker_controller. Avoids importing torch via rl.fly_rl at module load."""
    profile = get_profile(profile_name)
    return make_seeker_controller(**(profile.controller_overrides or {}))


def test_make_seeker_controller_threads_softening_for_vq2_case_c_only():
    """A controller built from the vq2_case_c profile has the softened kp_att (4.0) AND the slew limit
    (8.0); one from vq1_case_a (no overrides) keeps the original stiff gains + slew OFF — VQ1 / case-A
    byte-identical."""
    c_vq2 = _controller_from_profile("vq2_case_c")
    assert c_vq2.kp_att == 4.0
    assert c_vq2.body_rate_slew_max_rps2 == 8.0
    c_vq1 = _controller_from_profile("vq1_case_a")
    assert c_vq1.kp_att == 10.0
    assert c_vq1.body_rate_slew_max_rps2 is None


def test_gate_seeker_built_with_profile_controller_is_soft():
    """End-to-end of the wiring: a GateSeeker handed a controller built from the vq2_case_c overrides
    drives the SOFT loop; the default seeker (no overrides) drives the stiff one."""
    soft = GateSeeker(config=GateSeekerConfig(),
                      controller=_controller_from_profile("vq2_case_c"))
    stiff = GateSeeker(config=GateSeekerConfig())   # default factory controller
    assert soft.controller.kp_att == 4.0 and soft.controller.body_rate_slew_max_rps2 == 8.0
    assert stiff.controller.kp_att == 10.0 and stiff.controller.body_rate_slew_max_rps2 is None
