"""Egress ATTITUDE FREEZE — the A10 acquisition-trap fix.

THE TRAP (live VQ2, 2026-06-30): the spawn-gate egress ramps its forward demand from ~0 while the
spawn attitude tilt was FROZEN through settle/anchor (the hold's ``hold_freeze_attitude``). So the
decoupled body-rate controller (kp_att) sees a LEVEL target vs a TILTED current attitude and commands
a re-level that SATURATES the ``pursuit_pitch_rate_cap_rps`` cap → +1.50 NOSE-UP at the exact ticks
forward demand is ~0. That swings the +20° camera OFF the spawn gate → detector dark → ``pose=None``
→ pursuit never acquires. THE FIX: ``egress_freeze_attitude`` zeroes the egress' OWN roll/pitch rate
command (like the hold's freeze) so the camera stays ON the gate; yaw + thrust-floor + forward
feedforward remain. Default OFF (VQ1/case-A byte-identical); vq2_case_c flips it ON via the new
``DeployProfile.seeker_overrides`` seam.

[VQ2 slow-is-smooth, A10, 2026-06-30]
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from racer.contracts import ControlMode, NavState  # noqa: E402
from racer.deploy_profile import get_profile, vq1_case_a, vq2_case_c  # noqa: E402
from racer.gate_seeker import GateSeeker, GateSeekerConfig  # noqa: E402


def _tilted_nav(sim_time_ns=0):
    """A NavState carrying a frozen spawn TILT (the controller will try to re-level off it)."""
    return NavState(
        sim_time_ns=sim_time_ns,
        position_ned=np.array([0.0, 0.0, -2.5]),
        velocity_ned=np.zeros(3),
        roll=np.deg2rad(8.0),
        pitch=np.deg2rad(-10.0),   # nose-down spawn tilt -> the controller commands a nose-up re-level
        yaw=0.0,
        time_since_vision_update_s=0.05,
    )


# ===========================================================================
# 1. DEFAULT: the freeze ships OFF (VQ1 / case-A byte-identical)
# ===========================================================================
def test_egress_freeze_attitude_defaults_off():
    """The freeze is opt-in: a bare GateSeekerConfig has it OFF so the legacy egress (and VQ1) is
    byte-identical."""
    assert GateSeekerConfig().egress_freeze_attitude is False


# ===========================================================================
# 2. OFF: the egress still commands the (bug-preserving) re-level pitch
# ===========================================================================
def test_egress_freeze_off_preserves_relevel_pitch():
    """With the freeze OFF the egress command is the legacy behaviour: from a frozen spawn tilt the
    decoupled controller commands a NON-zero (re-levelling) roll/pitch rate — the exact A10 trap the
    fix targets. Pins that OFF == today's egress (no behaviour change for VQ1/case-A)."""
    seeker = GateSeeker(config=GateSeekerConfig(
        cruise_speed=3.0, launch_ramp_s=0.0, egress_freeze_attitude=False))
    cmd = seeker._egress_command(_tilted_nav())
    assert cmd.mode is ControlMode.BODY_RATE
    # the tilted spawn attitude drives a non-zero re-level pitch (the trap: nose-up off a nose-down tilt)
    assert abs(float(cmd.body_rate[1])) > 0.05, \
        "freeze-off egress must still command the (bug-preserving) re-levelling pitch"


# ===========================================================================
# 3. ON: the freeze zeroes roll/pitch, leaves yaw + (floored) thrust UNCHANGED
# ===========================================================================
def test_egress_freeze_on_zeroes_roll_pitch_keeps_yaw_and_thrust():
    """With the freeze ON the egress roll/pitch rate command is EXACTLY zero (hold the spawn attitude
    → camera stays on the gate), while yaw and the floored thrust are UNCHANGED vs the non-frozen
    command (the freeze only frees the two tilt axes; it does not touch yaw or the egress thrust floor)."""
    cfg_kw = dict(cruise_speed=3.0, launch_ramp_s=0.0)
    off = GateSeeker(config=GateSeekerConfig(egress_freeze_attitude=False, **cfg_kw))
    on = GateSeeker(config=GateSeekerConfig(egress_freeze_attitude=True, **cfg_kw))
    cmd_off = off._egress_command(_tilted_nav())
    cmd_on = on._egress_command(_tilted_nav())
    # roll + pitch FROZEN (exactly zero)
    assert float(cmd_on.body_rate[0]) == 0.0, "freeze must zero the egress roll-rate command"
    assert float(cmd_on.body_rate[1]) == 0.0, "freeze must zero the egress pitch-rate command (the trap axis)"
    # yaw passes through unchanged (freeze never touches the yaw axis)
    assert float(cmd_on.body_rate[2]) == float(cmd_off.body_rate[2]), "yaw must pass through the freeze"
    # thrust UNCHANGED vs the floored non-frozen command (the egress thrust floor still applies)
    assert cmd_on.thrust == pytest.approx(cmd_off.thrust), "the egress thrust floor must still apply under the freeze"
    # and the floor is real: the egress collective sits at/above hover (no free-fall)
    assert cmd_on.thrust >= on.controller.hover_thrust - 1e-6


# ===========================================================================
# 4. DEPLOY PROFILE seam: seeker_overrides carries the freeze for vq2_case_c only
# ===========================================================================
def test_deploy_profile_seeker_overrides_wires_the_freeze():
    """vq2_case_c carries the freeze via the new DeployProfile.seeker_overrides seam; vq1_case_a leaves
    it None (explicit, to pin VQ1 byte-identity)."""
    # the freeze is carried (exact-equality would be brittle: vq2_case_c now also carries the A13
    # hold_last_demand_s bridge override -- assert the freeze KEY is present + ON instead).
    assert get_profile("vq2_case_c").seeker_overrides.get("egress_freeze_attitude") is True
    assert get_profile("vq1_case_a").seeker_overrides is None
    # the constructors agree with the named lookup
    assert vq2_case_c().seeker_overrides.get("egress_freeze_attitude") is True
    assert vq1_case_a().seeker_overrides is None


def test_default_deploy_profile_seeker_overrides_is_none():
    """The dataclass default keeps existing callers / pickles forward-compatible: seeker_overrides
    defaults to None (no overrides == today's seeker config)."""
    from racer.deploy_profile import DeployProfile
    from racer.navigator import NavigatorConfig
    p = DeployProfile(name="x", nav_config=NavigatorConfig(), cmd_rate_scale=1.0, self_localizing=False)
    assert p.seeker_overrides is None


# ===========================================================================
# 5. make_seeker seam: the override reaches GateSeekerConfig on the case-C path only
# ===========================================================================
def _seeker_from_profile(profile_name):
    """Mirror rl.fly_rl.make_seeker's GateSeekerConfig construction: splat the profile's overrides
    alongside the hardcoded CLI-driven fields. Avoids importing torch via rl.fly_rl at module load."""
    profile = get_profile(profile_name)
    return GateSeeker(
        config=GateSeekerConfig(
            cruise_speed=3.0,
            settle_s=0.5,
            anchor_release_detections=3,
            **(profile.seeker_overrides or {}),
        ),
        detector=None,
    )


def test_make_seeker_threads_freeze_for_vq2_case_c_only():
    """A seeker built from the vq2_case_c profile has egress_freeze_attitude ON (the A10 fix wired);
    one from vq1_case_a (no overrides) has it OFF — VQ1/case-A byte-identical."""
    assert _seeker_from_profile("vq2_case_c").config.egress_freeze_attitude is True
    assert _seeker_from_profile("vq1_case_a").config.egress_freeze_attitude is False
    # the hardcoded CLI-driven fields still land (no collision with the override)
    s = _seeker_from_profile("vq2_case_c")
    assert s.config.cruise_speed == 3.0 and s.config.settle_s == 0.5
    assert s.config.anchor_release_detections == 3
