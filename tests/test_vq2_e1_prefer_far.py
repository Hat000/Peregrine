"""E1 PREFER-FAR — the gate-1 turn stale-target fix (2026-07-05 turn-readiness audit,
handoff/vq2_turn_readiness_audit_2026-07-05.md, run 20260705_012253).

THE BUG (ticks 50-88 of run 012253): the A36 refine's "downrange" bound (> pass_degenerate_range_m
= 3.0) still admits the 3.0..pass_arm_range_m (4.5) CLOSE-RESIDUAL band — where the just-passed
gate lives while it exits the FOV. The refine latched _pass_turn_yaw = +0.35 rad off a ~4.3 m
residual, the hold-until-pointed released "pointed" at that stub within ~0.2 s, and
_pass_acquired_next (same > degenerate bound) handed pursuit the SAME residual (yaw_des -0.33).
The REAL gate 1 appeared ~0.9 s later at ~16 m — acquired in PURSUIT, where the 0.9 rad/s setpoint
slew let yaw_des run ~3.0 rad ahead of the heading (the rate-limited tail-chase: wz pinned at the
cap for 0.9 s stretches while roll was starved by the omega norm-clip).

THE FIX (one flag, ``pass_refine_prefer_far``, default OFF => byte-identical): a pose must be a
PLAUSIBLE NEXT GATE (range > pass_arm_range_m, not merely past the degenerate band) BOTH to re-aim
the turn target (_maybe_refine_turn_target) AND to end the pass as "next gate acquired"
(_pass_acquired_next). The residual band can then neither aim the turn nor hand off to pursuit;
the turn stays PENDING (the hold defers acquire-next, bounded by pass_turn_coast_s) until a
genuine downrange gate is seen.

Also here: the OMEGA NORM-CLIP ROLL-SURVIVAL pin (audit Bug #2 / tuning lever R1). This one is
about Controller behavior, not the new flag: with a large yaw error the kp_att*rotvec norm
saturates max_body_rate_rps and the proportional _clip_norm scales ROLL down with it (the
"barely any roll while yawing hard" contest). Parameterized 4.0 vs 6.0 to document the R1 effect.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from racer.contracts import GatePose, NavState, Setpoint  # noqa: E402
from racer.deploy_profile import vq2_case_c  # noqa: E402
from racer.gate_seeker import GateSeeker, GateSeekerConfig, make_seeker_controller  # noqa: E402

_NS = 1_000_000_000


def _nav(t_ns, *, yaw=0.0):
    return NavState(sim_time_ns=int(t_ns), position_ned=np.zeros(3), velocity_ned=np.zeros(3),
                    yaw=float(yaw), time_since_vision_update_s=float("inf"))


def _residual_pose(rng=4.3, xoff=1.5):
    """A pose in the CLOSE-RESIDUAL band (pass_degenerate_range_m < rng <= pass_arm_range_m): the
    just-passed gate while it exits the FOV — the 012253 tick-51 latch source (~4.3 m, ~+0.35 rad)."""
    t = np.array([float(xoff), 0.0, float(np.sqrt(max(rng * rng - xoff * xoff, 1.0)))])
    return GatePose(frame_id=0, sim_time_ns=0, R_cam_gate=np.eye(3), t_cam_gate=t,
                    reproj_error_px=0.5)


def _far_pose(rng=16.5, xoff=12.0):
    """A genuine DOWNRANGE gate-2 pose to the RIGHT (the real gate 1 of run 012253, ~16 m)."""
    t = np.array([float(xoff), 0.0, float(np.sqrt(max(rng * rng - xoff * xoff, 1.0)))])
    return GatePose(frame_id=0, sim_time_ns=0, R_cam_gate=np.eye(3), t_cam_gate=t,
                    reproj_error_px=0.5)


def _e1_seeker(**cfg_kw) -> GateSeeker:
    """The vq2_case_c-shaped pass-turn seeker (arm 4.5 / degenerate 3.0, refine + hold ON, fast
    wire window) with prefer_far ON unless overridden. The H-1a exclusion snapshot is NULLED after
    _begin_pass by the tests (its cone interplay is pinned elsewhere; these tests isolate the E1
    range seam)."""
    base = dict(use_pass_dead_reckon=True, pass_arm_range_m=4.5, pass_degenerate_range_m=3.0,
                pass_turn_through=True, pass_turn_hold_until_pointed=True, pass_turn_refine=True,
                pass_exclude_prev_gate=True, pass_blind_turn_cap_rad=1.6, pass_refine_min_bw=0.3,
                use_soft_bearing_weight=True, track_abs_range_cap_m=35.0,
                pass_turn_coast_s=1.5, pass_turn_point_tol_rad=0.17,
                pass_coast_s=0.0, pass_wire_coast_s=0.0,
                pass_refine_prefer_far=True)
    base.update(cfg_kw)
    s = GateSeeker(config=GateSeekerConfig(**base))
    s._anchored = True
    s._last_yaw = 0.0
    return s


def _begin(s: GateSeeker, *, wire=True) -> None:
    s._begin_pass(0, wire=wire)
    s._pass_prev_dir_world = None    # isolate the E1 range seam from the H-1a cone


# ===========================================================================
# (a) FLAG ON — the hold does NOT release on a <8 m residual target
# ===========================================================================
def test_a_hold_does_not_release_on_close_residual():
    """The 012253 replay, flag ON: a 4.3 m residual pose must NOT re-aim the turn (target stays
    None, turn stays PENDING), the hold stays ACTIVE (deferring acquire-next), and the pass does
    NOT end on that residual — the exact chain that released at ~0.2 s in flight is closed."""
    s = _e1_seeker()
    _begin(s)
    assert s._pass_turn_pending, "refine mode marks the turn pending at begin"
    s._last_pose = _residual_pose(rng=4.3)
    s._last_bearing_w = 0.9                        # consistent frame — range alone must reject it
    s._maybe_refine_turn_target(_nav(int(0.1 * _NS)))
    assert s._pass_turn_yaw is None, "a residual-band pose must NOT re-aim the turn target"
    assert s._pass_turn_pending, "the turn must stay PENDING through the residual window"
    assert s._turn_hold_active(int(0.3 * _NS)), "pending turn must HOLD (defer acquire-next)"
    assert not s._pass_acquired_next(_nav(int(0.3 * _NS)), _residual_pose(rng=4.3)), \
        "the pass must NOT end on the residual (hold active + range bound)"


def test_a_hold_completes_against_the_real_far_gate():
    """Continuation of the replay: once the REAL downrange gate appears (~16 m), the refine re-aims,
    the hold keeps holding until POINTED at THAT target, and only then does the far pose end the
    pass — the turn completes against the real gate, not a stub."""
    s = _e1_seeker()
    _begin(s)
    s._last_pose = _residual_pose(rng=4.3)
    s._last_bearing_w = 0.9
    s._maybe_refine_turn_target(_nav(int(0.1 * _NS)))
    assert s._pass_turn_yaw is None
    # the real gate 1 clears the frame
    s._last_pose = _far_pose(rng=16.5)
    s._last_bearing_w = 0.9
    s._maybe_refine_turn_target(_nav(int(0.4 * _NS)))
    assert s._pass_turn_yaw is not None, "a genuine downrange pose must re-aim the turn"
    assert s._pass_turn_yaw > 0.2, f"target aims RIGHT at the far gate; got {s._pass_turn_yaw:+.2f}"
    assert not s._pass_turn_pending
    # not yet pointed (heading 0.0 vs target > 0.2, tol 0.17) -> hold, no acquire
    assert s._turn_hold_active(int(0.5 * _NS)), "must hold while not yet pointed at the REAL gate"
    assert not s._pass_acquired_next(_nav(int(0.5 * _NS)), _far_pose(rng=16.5))
    # the coast slews the heading onto the target -> pointed -> release -> far pose acquires
    s._last_yaw = float(s._pass_turn_yaw)
    assert not s._turn_hold_active(int(0.6 * _NS)), "pointed at the real gate -> hold releases"
    assert s._pass_acquired_next(_nav(int(0.6 * _NS)), _far_pose(rng=16.5)), \
        "the REAL far gate ends the pass once pointed"


def test_a_acquire_next_range_bound_without_the_hold():
    """The acquire-next half of the flag in isolation (hold OFF): a residual-band pose must not end
    the pass under prefer_far, a plausible far pose must. This is the tick-54 hand-off seam."""
    s = _e1_seeker(pass_turn_hold_until_pointed=False)
    _begin(s)
    assert not s._pass_acquired_next(_nav(int(0.3 * _NS)), _residual_pose(rng=4.3)), \
        "residual band (<= pass_arm_range_m) must not count as the next gate"
    assert s._pass_acquired_next(_nav(int(0.3 * _NS)), _far_pose(rng=16.5)), \
        "a plausible downrange pose (> pass_arm_range_m) still ends the pass"


# ===========================================================================
# (b) OMEGA NORM-CLIP ROLL-SURVIVAL — audit Bug #2 / tuning lever R1 (controller
#     behavior; parameterized max_body_rate_rps 4.0 vs 6.0 to document the R1 effect)
# ===========================================================================
def _turn_cmd(max_body_rate_rps: float, *, sp_yaw: float = 2.5):
    """One vq2_case_c-controller command with a big yaw error (sp_yaw vs nav yaw 0) AND a lateral
    (roll) accel demand — the pass-turn slew contest. Returns the body-rate vector."""
    co = dict(vq2_case_c().controller_overrides)
    co["max_body_rate_rps"] = float(max_body_rate_rps)
    c = make_seeker_controller(**co)
    nav = NavState(sim_time_ns=0, position_ned=np.zeros(3), velocity_ned=np.zeros(3),
                   roll=0.0, pitch=0.0, yaw=0.0, angular_rate_body=np.zeros(3),
                   time_since_vision_update_s=0.1)
    sp = Setpoint(sim_time_ns=0, accel_ned=np.array([0.0, 2.5, 0.0]), yaw=float(sp_yaw))
    return np.asarray(c.command(nav, sp).body_rate, float)


def test_b_norm_clip_binds_during_the_turn_and_starves_roll():
    """With a 2.5 rad yaw error, kp_att*|rotvec| >> max_body_rate: the norm-clip BINDS (|omega| ==
    the cap) — measured in-flight on 55-69% of pursuit ticks (runs 183049/231155/012253). The roll
    axis is scaled down WITH the yaw axis (proportional clip), far below its unstarved value: the
    'barely any roll while yawing hard' contest."""
    br4 = _turn_cmd(4.0)
    assert np.linalg.norm(br4) == pytest.approx(4.0, rel=1e-9), \
        "the norm-clip must bind under the big-yaw-error turn"
    # the same lateral demand WITHOUT the yaw contest: roll gets its full (unclipped) authority
    br_noyaw = _turn_cmd(4.0, sp_yaw=0.0)
    assert np.linalg.norm(br_noyaw) < 4.0 - 1e-6, "no yaw contest -> the clip must NOT bind"
    assert abs(br_noyaw[0]) > 1.5 * abs(br4[0]), \
        (f"roll must be STARVED by the yaw contest: unstarved {br_noyaw[0]:+.3f} vs "
         f"contested {br4[0]:+.3f}")


def test_b_raising_the_ceiling_frees_roll_proportionally():
    """R1 (max_body_rate_rps 4.0 -> 6.0): while the clip binds, _clip_norm preserves direction, so
    raising the ceiling scales EVERY axis by exactly 6/4 — roll is un-starved by 1.5x with no
    change to the yaw/roll balance. (Downstream the seeker's per-axis caps still bound the realized
    rates: visual_yaw_rate_cap_rps 0.9, pursuit_roll/pitch_rate_cap_rps 1.5 — gate_seeker.py's
    _feedforward_command applies them AFTER this clip, so raising the norm ceiling cannot raise
    the realized yaw above 0.9.)"""
    br4 = _turn_cmd(4.0)
    br6 = _turn_cmd(6.0)
    assert np.linalg.norm(br6) == pytest.approx(6.0, rel=1e-9), "the clip still binds at 6.0"
    np.testing.assert_allclose(br6, 1.5 * br4, rtol=1e-9)
    assert abs(br6[0]) == pytest.approx(1.5 * abs(br4[0]), rel=1e-9), \
        "roll is freed exactly in proportion to the raised ceiling"


# ===========================================================================
# (c) FLAG ON — the refine latches the true far gate over a close residual
# ===========================================================================
def test_c_residual_first_far_gate_still_wins():
    """Residual seen FIRST (the 012253 order): no latch; the far gate then latches. The target is
    identical to what a far-gate-only pass would have latched (the residual left no residue)."""
    s = _e1_seeker()
    _begin(s)
    s._last_pose = _residual_pose(rng=4.3)
    s._last_bearing_w = 0.9
    s._maybe_refine_turn_target(_nav(int(0.1 * _NS)))
    assert s._pass_turn_yaw is None
    s._last_pose = _far_pose(rng=16.5)
    s._maybe_refine_turn_target(_nav(int(0.2 * _NS)))
    got = s._pass_turn_yaw
    assert got is not None and got > 0.2
    # reference: far gate only
    s2 = _e1_seeker()
    _begin(s2)
    s2._last_pose = _far_pose(rng=16.5)
    s2._last_bearing_w = 0.9
    s2._maybe_refine_turn_target(_nav(int(0.2 * _NS)))
    assert got == pytest.approx(s2._pass_turn_yaw, abs=1e-12), \
        "the residual must leave NO residue on the latched target"


def test_c_far_first_residual_cannot_drag_the_target_back():
    """Far gate latched FIRST: a later residual-band pose must NOT re-aim (latch-follow only
    follows plausible next-gate poses under prefer_far)."""
    s = _e1_seeker()
    _begin(s)
    s._last_pose = _far_pose(rng=16.5)
    s._last_bearing_w = 0.9
    s._maybe_refine_turn_target(_nav(int(0.1 * _NS)))
    t1 = s._pass_turn_yaw
    assert t1 is not None
    s._last_pose = _residual_pose(rng=4.3)
    s._maybe_refine_turn_target(_nav(int(0.2 * _NS)))
    assert s._pass_turn_yaw == pytest.approx(t1, abs=1e-12), \
        "a residual-band pose must not drag the latched target off the real gate"
    assert not s._pass_turn_pending


# ===========================================================================
# (d) FLAG OFF — byte-identity on the touched paths (the 012253 legacy behavior)
# ===========================================================================
def test_d_default_off_and_profile_staged_off():
    """REGRESSION PIN: the field default is False (VQ1 + every offline path byte-identical);
    vq2_case_c ships it ON (GO-time flip applied 2026-07-05, operator-approved flight package)."""
    assert GateSeekerConfig().pass_refine_prefer_far is False
    so = vq2_case_c().seeker_overrides or {}
    assert so.get("pass_refine_prefer_far") is True, "flight package flips E1 ON in vq2_case_c"
    GateSeekerConfig(**so)     # the override dict must construct cleanly


def test_d_flag_off_residual_latches_exactly_as_before():
    """Flag OFF: the 4.3 m residual DOES re-aim the turn target (the legacy > degenerate bound —
    the exact 012253 tick-51 behavior this flag exists to fix), pending clears."""
    s = _e1_seeker(pass_refine_prefer_far=False)
    _begin(s)
    s._last_pose = _residual_pose(rng=4.3)
    s._last_bearing_w = 0.9
    s._maybe_refine_turn_target(_nav(int(0.1 * _NS)))
    assert s._pass_turn_yaw is not None, "legacy path must latch the residual (byte-identical)"
    assert not s._pass_turn_pending


def test_d_flag_off_acquire_next_accepts_the_residual():
    """Flag OFF: _pass_acquired_next keeps the legacy > pass_degenerate_range_m bound — a 4.3 m
    residual ends the pass (the tick-54 hand-off), byte-identical pre-E1."""
    s = _e1_seeker(pass_refine_prefer_far=False, pass_turn_hold_until_pointed=False)
    _begin(s)
    assert s._pass_acquired_next(_nav(int(0.3 * _NS)), _residual_pose(rng=4.3)), \
        "legacy path must accept the residual as the next gate (byte-identical)"
    # and the degenerate band is still rejected on BOTH paths (unchanged behavior)
    s2 = _e1_seeker(pass_turn_hold_until_pointed=False)
    _begin(s2)
    assert not s2._pass_acquired_next(_nav(int(0.3 * _NS)), _residual_pose(rng=2.5, xoff=1.0))
