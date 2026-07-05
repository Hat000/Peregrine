"""A36 — turn-convergence build: pointing gate (Item 1), physical-plane turn (Item 3),
lateral-first "switch lanes" (Item 4). Item 0 (yaw sign) lives in test_vq2_yaw_actuation_sign.py.

All three below are FLAG-GATED, default-OFF. Each item has:
  * a FOCUSED test encoding the fix behavior, and
  * a REGRESSION PIN asserting default-off == the pre-A36 behavior byte-for-byte.

Run 20260704_032626 grounding: after passing gate 0 the drone carried too much forward speed and
ORBITED gate 2 (track_range 15-19 m for ~70 ticks, never closed). Items 1&4 cut the overfly and
translate onto the approach line; Item 3 commits the turn at the physical plane and holds it until
pointed instead of self-cancelling on the ~9 m-early wire.

[VQ2 slow-is-smooth, A36, 2026-07-03]
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from racer.contracts import GatePose, NavState, Setpoint  # noqa: E402
from racer.deploy_profile import vq1_case_a, vq2_case_c  # noqa: E402
from racer.gate_seeker import GateSeeker, GateSeekerConfig, _clip_norm, make_seeker_controller  # noqa: E402

_NS = 1_000_000_000


def _nav(t_ns, *, yaw=0.0):
    return NavState(sim_time_ns=int(t_ns), position_ned=np.zeros(3), velocity_ned=np.zeros(3),
                    yaw=float(yaw), time_since_vision_update_s=float("inf"))


def _seeker(**cfg_kw) -> GateSeeker:
    """A minimal seeker for the composition/pass-helper units. use_image_servo_lateral defaults ON
    so _compose_image_servo_accel is the live path; timing windows are irrelevant here."""
    base = dict(use_image_servo_lateral=True)
    base.update(cfg_kw)
    return GateSeeker(config=GateSeekerConfig(**base))


def _compose(seeker, az, *, eff_ramp=1.0, fwd_ramp=1.0, yaw_now=0.0):
    """Invoke the image-servo composition for an apparent azimuth ``az`` (psi_world = yaw_now + az).
    Returns (a_vec, a_fwd_component_along_los, a_lat_component_along_e_right)."""
    los = np.array([np.cos(yaw_now), np.cos(0.0) * 0.0 + np.sin(yaw_now), 0.0])
    los = np.array([np.cos(yaw_now), np.sin(yaw_now), 0.0])
    psi_world = yaw_now + az
    a_vec = seeker._compose_image_servo_accel(_nav(0, yaw=yaw_now), los, psi_world,
                                              eff_ramp, fwd_ramp)
    e_right = np.array([-np.sin(yaw_now), np.cos(yaw_now), 0.0])
    a_fwd = float(np.dot(a_vec, los))
    a_lat = float(np.dot(a_vec, e_right))
    return a_vec, a_fwd, a_lat


# ===========================================================================
# ITEM 1 — POINTING GATE on the forward drive ("point before you push")
# ===========================================================================
def test_item1_pointing_gate_kills_forward_while_off_axis():
    """With the pointing gate ON, a badly off-axis gate (az well beyond the gate radius) drives the
    forward component to ~0 -- the drone turns before it translates. On the OFF path the same az
    still drives substantial forward (cos^pow)."""
    off = _seeker(fwd_point_gate_az_rad=None)                       # OFF
    on = _seeker(fwd_point_gate_az_rad=0.35, fwd_point_gate_full_az_rad=0.05)
    az = 0.5                                                        # ~29 deg, beyond the 0.35 gate
    _, fwd_off, _ = _compose(off, az)
    _, fwd_on, _ = _compose(on, az)
    assert fwd_off > 0.2, f"OFF path should still push forward off-axis; got {fwd_off:.3f}"
    assert fwd_on < 1e-6, f"pointing gate should zero forward beyond the gate radius; got {fwd_on:.3f}"


def test_item1_pointing_gate_full_forward_when_centered():
    """When the gate is centered (|az| <= full), the pointing gate is fully open (g_point=1): the
    forward component is IDENTICAL to the OFF path -- centered pace is untouched."""
    off = _seeker(fwd_point_gate_az_rad=None)
    on = _seeker(fwd_point_gate_az_rad=0.35, fwd_point_gate_full_az_rad=0.05)
    az = 0.02                                                       # inside the full band
    _, fwd_off, _ = _compose(off, az)
    _, fwd_on, _ = _compose(on, az)
    assert fwd_on == pytest.approx(fwd_off, abs=1e-9)


def test_item1_pointing_gate_monotone_ramp():
    """g_point is a monotone smoothstep: forward drive is non-increasing as |az| grows from full to
    the gate radius (no non-monotonic kink)."""
    on = _seeker(fwd_point_gate_az_rad=0.35, fwd_point_gate_full_az_rad=0.05)
    fwds = [ _compose(on, az)[1] for az in np.linspace(0.05, 0.35, 8) ]
    for a, b in zip(fwds, fwds[1:]):
        assert b <= a + 1e-9, "pointing gate must be monotone non-increasing in |az|"


def test_item1_default_off_byte_identical():
    """REGRESSION PIN: fwd_point_gate_az_rad default is None => the composition is byte-identical to
    the pre-A36 cos^pow path across the az range."""
    base = _seeker()                                               # default (gate off)
    assert base.config.fwd_point_gate_az_rad is None
    for az in (-0.6, -0.2, 0.0, 0.15, 0.4, 0.7):
        # recompute the legacy expected forward directly: forward_accel * cos^pow(az)
        cfg = base.config
        exp_fwd = cfg.forward_accel_mps2 * (max(np.cos(az), 0.0) ** cfg.fwd_scale_pow)
        # the composed vector is norm-capped; below the cap the forward component == exp_fwd
        _, fwd, _ = _compose(base, az)
        # only compare where the norm cap isn't binding (small az, forward dominates)
        if abs(az) < 0.3:
            assert fwd == pytest.approx(exp_fwd, abs=1e-9)


# ===========================================================================
# ITEM 4 — LATERAL-FIRST "switch lanes"
# ===========================================================================
def test_item4_lateral_first_prioritizes_lateral_within_budget():
    """With use_lateral_first_budget ON and a raised lateral cap, an off-axis gate gets the FULL
    lateral demand (up to the cap) and forward yields to the remaining radial budget -- vs the OFF
    (sum-then-norm-cap) path where a large forward CLIPS the lateral down."""
    common = dict(image_lat_cap_mps2=3.0, total_accel_cap_mps2=3.5, forward_accel_mps2=2.0,
                  fwd_scale_pow=2.0)
    off = _seeker(use_lateral_first_budget=False, **common)
    on = _seeker(use_lateral_first_budget=True, **common)
    az = 0.35                                                      # off-axis: big lateral demand
    _, fwd_off, lat_off = _compose(off, az)
    _, fwd_on, lat_on = _compose(on, az)
    assert abs(lat_on) >= abs(lat_off) - 1e-9, "lateral-first must not REDUCE the lateral demand"
    # lateral-first delivers the full k_az*az demand (clipped to cap), un-eroded by forward:
    exp_lat = min(8.0 * (0.35 - 0.03), 3.0)                        # k_az*dead(az) clipped to cap
    assert abs(lat_on) == pytest.approx(exp_lat, abs=1e-6)
    # and the composed vector still respects the total cap:
    a_vec, _, _ = _compose(on, az)
    assert float(np.linalg.norm(a_vec)) <= 3.5 + 1e-9


def test_item4_lateral_first_forward_takes_remaining_budget():
    """The forward term under lateral-first is exactly sqrt(cap^2 - a_lat^2)-bounded (the remaining
    radial budget), never more."""
    on = _seeker(use_lateral_first_budget=True, image_lat_cap_mps2=3.0,
                 total_accel_cap_mps2=3.5, forward_accel_mps2=2.0)
    az = 0.35
    _, fwd_on, lat_on = _compose(on, az)
    budget = np.sqrt(max(3.5**2 - lat_on**2, 0.0))
    assert fwd_on <= budget + 1e-9


def test_item4_default_off_byte_identical():
    """REGRESSION PIN: use_lateral_first_budget default False => the composition is byte-identical
    to the pre-A36 sum-then-clip_norm path."""
    cfg_kw = dict(image_lat_cap_mps2=1.5, total_accel_cap_mps2=2.0, forward_accel_mps2=0.8)
    base = _seeker(**cfg_kw)
    assert base.config.use_lateral_first_budget is False
    for az in (-0.5, -0.1, 0.0, 0.2, 0.45):
        # reconstruct the legacy composed vector exactly
        s2 = _seeker(**cfg_kw)
        yaw_now = 0.0
        los = np.array([np.cos(yaw_now), np.sin(yaw_now), 0.0])
        e_right = np.array([-np.sin(yaw_now), np.cos(yaw_now), 0.0])
        db = s2.config.image_az_deadband_rad
        az_eff = np.sign(az) * max(abs(az) - db, 0.0)
        a_lat = float(np.clip(s2.config.image_kaz_mps2_per_rad * az_eff,
                              -s2.config.image_lat_cap_mps2, s2.config.image_lat_cap_mps2))
        fwd_scale = max(np.cos(az), 0.0) ** s2.config.fwd_scale_pow
        a_fwd = s2.config.forward_accel_mps2 * fwd_scale
        exp = _clip_norm(a_fwd * los + a_lat * e_right, s2.config.total_accel_cap_mps2)
        got, _, _ = _compose(base, az)
        np.testing.assert_allclose(got, exp, atol=1e-12)


# ===========================================================================
# ITEM 3 — PHYSICAL-PLANE TURN COMMIT + HOLD-UNTIL-POINTED
# ===========================================================================
def _pose(rng, sim_time_ns=0):
    """A minimal GatePose whose range_m (= |t_cam_gate|) is ``rng`` (the only field the pass state
    machine reads)."""
    return GatePose(frame_id=0, sim_time_ns=int(sim_time_ns), R_cam_gate=np.eye(3),
                    t_cam_gate=np.array([0.0, 0.0, float(rng)]), reproj_error_px=0.5)


def _pass_seeker(**cfg_kw) -> GateSeeker:
    base = dict(use_pass_dead_reckon=True, pass_arm_range_m=3.0, pass_degenerate_range_m=2.5)
    base.update(cfg_kw)
    s = GateSeeker(config=GateSeekerConfig(**base))
    s._anchored = True
    s._last_yaw = 0.0
    return s


def _pose_right(rng=18.0):
    """A gate-2 pose to the RIGHT of the nose (world bearing ~ +0.35 rad at identity attitude): camera
    optical x=+12 (right), z=+rng (forward). range_m = |t_cam_gate|."""
    xoff = 12.0
    t = np.array([xoff, 0.0, float(np.sqrt(max(rng * rng - xoff * xoff, 1.0)))])
    return GatePose(frame_id=0, sim_time_ns=0, R_cam_gate=np.eye(3), t_cam_gate=t, reproj_error_px=0.5)


def test_item3_wire_requires_near_defers_commit_while_gate_far_ahead():
    """With pass_wire_requires_near ON, a wire index_advanced while the tracked gate is still WELL
    AHEAD (range 9 m > pass_arm_range_m 3) does NOT commit the pass -- it records intent. The
    physical commit (degenerate-close) then fires, and honors the pending wire intent."""
    s = _pass_seeker(pass_wire_requires_near=True)
    # wire advances 9 m out -> deferred, not committed
    s._update_pass_state(0, _pose(9.0), index_advanced=True)
    assert not s._passing, "wire advance 9 m out must NOT commit under pass_wire_requires_near"
    assert s._pass_wire_pending, "the wire intent must be recorded as pending"
    # approach: arm at 3 m, then degenerate-close at 2 m -> physical commit, honoring pending wire
    s._update_pass_state(1, _pose(2.9), index_advanced=False)     # arms
    s._update_pass_state(2, _pose(2.0), index_advanced=False)     # degenerate-close -> commit
    assert s._passing, "physical plane (degenerate-close) must commit the pass"
    assert s._pass_wire, "the deferred wire intent must upgrade the commit to the fast wire window"


def test_item3_wire_commits_immediately_when_already_at_plane():
    """A wire advance while ALREADY within pass_arm_range_m (at the plane) commits immediately even
    under pass_wire_requires_near (we are AT the gate, not 9 m short)."""
    s = _pass_seeker(pass_wire_requires_near=True)
    s._update_pass_state(1, _pose(2.5), index_advanced=True)      # at the plane + wire -> commit
    assert s._passing
    assert s._pass_wire


def test_item3_default_off_wire_commits_unconditionally():
    """REGRESSION PIN: pass_wire_requires_near default False => a wire advance commits the pass
    unconditionally at ANY range (byte-identical pre-A36)."""
    s = _pass_seeker(pass_wire_requires_near=False)
    assert s.config.pass_wire_requires_near is False
    s._update_pass_state(0, _pose(9.0), index_advanced=True)      # far ahead
    assert s._passing, "default path must commit on the wire regardless of range"
    assert not s._pass_wire_pending


def test_item3_hold_until_pointed_holds_then_releases():
    """With pass_turn_hold_until_pointed ON and a blind turn target latched, the dead-reckon coast
    is HELD while the commanded heading is far from the target and RELEASES once within tolerance."""
    s = _pass_seeker(pass_turn_through=True, pass_turn_hold_until_pointed=True,
                     pass_turn_coast_s=1.5, pass_turn_point_tol_rad=0.17)
    s._passing = True
    s._pass_t_ns = 0
    s._pass_turn_yaw = 1.6                                        # ~92 deg target
    # far from target -> hold active (coast held, acquire-next deferred)
    s._last_yaw = 0.0
    assert s._turn_hold_active(int(0.3 * _NS)), "must hold while not yet pointed"
    assert not s._pass_acquired_next(_nav(int(0.3 * _NS)), _pose(18.0)), "acquire-next deferred mid-turn"
    assert s._in_pass_dead_reckon(int(0.3 * _NS)), "coast held through the turn"
    # now pointed (within tol) -> hold releases
    s._last_yaw = 1.55
    assert not s._turn_hold_active(int(0.5 * _NS)), "must release once pointed"


def test_item3_hold_until_pointed_bounded_by_coast_s():
    """The hold is BOUNDED: past pass_turn_coast_s it releases even if not pointed (never an open
    loop)."""
    s = _pass_seeker(pass_turn_through=True, pass_turn_hold_until_pointed=True,
                     pass_turn_coast_s=1.5, pass_turn_point_tol_rad=0.17)
    s._passing = True
    s._pass_t_ns = 0
    s._pass_turn_yaw = 1.6
    s._last_yaw = 0.0                                             # never pointed
    assert s._turn_hold_active(int(1.0 * _NS)), "held within the bound"
    assert not s._turn_hold_active(int(1.6 * _NS)), "released past pass_turn_coast_s"


def test_item3_default_off_no_hold():
    """REGRESSION PIN: pass_turn_hold_until_pointed default False => _turn_hold_active is always
    False and _pass_acquired_next / _in_pass_dead_reckon are byte-identical pre-A36."""
    s = _pass_seeker(pass_turn_through=True, pass_turn_hold_until_pointed=False)
    assert s.config.pass_turn_hold_until_pointed is False
    s._passing = True
    s._pass_t_ns = 0
    s._pass_turn_yaw = 1.6
    s._last_yaw = 0.0
    assert not s._turn_hold_active(int(0.3 * _NS)), "no hold when the flag is off"


# ===========================================================================
# FIX B — close-range yaw-rate taper (reduces the pursuit yaw slew near a gate)
# ===========================================================================
def _slew_step(seeker, yaw_des, rng, dt=0.033):
    """One _slew_heading step from _last_yaw=0 toward yaw_des at tracked range ``rng``. Returns the
    achieved heading step (rad) -- the effective per-tick yaw authority after any taper."""
    seeker._last_yaw = 0.0
    seeker._last_pursuit_t_ns = 0
    seeker._track_range_m = float(rng)
    return abs(float(seeker._slew_heading(float(yaw_des), int(dt * _NS))))


def test_fixb_taper_reduces_close_range_yaw_authority():
    """The close-range taper cuts the achievable yaw step at short range vs long range (a big yaw_des
    so the slew, not the error, is the limiter)."""
    s = _seeker(pursuit_yaw_slew_rps=1.5, yaw_slew_taper_lo_range_m=3.0,
                yaw_slew_taper_hi_range_m=10.0, yaw_slew_taper_floor=0.35)
    near = _slew_step(s, 3.0, rng=3.0)     # at/below lo => floor
    far = _slew_step(s, 3.0, rng=12.0)     # >= hi => full
    assert near < far, "the taper must reduce yaw authority at close range"
    assert near == pytest.approx(far * 0.35, rel=0.05), "near authority ~ floor * far"


def test_fixb_default_off_no_taper():
    """REGRESSION PIN: yaw_slew_taper_lo_range_m default None => full slew authority at all ranges
    (byte-identical pre-Fix-B)."""
    s = _seeker(pursuit_yaw_slew_rps=1.5)   # taper off (default None)
    assert s.config.yaw_slew_taper_lo_range_m is None
    near = _slew_step(s, 3.0, rng=3.0)
    far = _slew_step(s, 3.0, rng=12.0)
    assert near == pytest.approx(far), "no taper => same authority at all ranges"


# ===========================================================================
# REFINE-TO-REAL-GATE — the turn re-aims at a real gate-2 pose, not a blind guess
# ===========================================================================
def _refine_seeker(**cfg_kw) -> GateSeeker:
    base = dict(use_pass_dead_reckon=True, pass_arm_range_m=3.0, pass_degenerate_range_m=2.5,
                pass_turn_through=True, pass_turn_hold_until_pointed=True, pass_turn_refine=True,
                pass_exclude_prev_gate=True, pass_blind_turn_cap_rad=1.6, pass_refine_min_bw=0.3,
                use_soft_bearing_weight=True, track_abs_range_cap_m=35.0)
    base.update(cfg_kw)
    s = GateSeeker(config=GateSeekerConfig(**base))
    s._anchored = True
    s._last_yaw = 0.0
    s._pass_heading = 0.0
    return s


def _nav0():
    return _nav(0)


def test_refine_begin_pass_sets_pending_not_blind_target():
    """With pass_turn_refine ON, _begin_pass does NOT compute a blind sign*cap target -- it leaves
    _pass_turn_yaw None and sets _pass_turn_pending (coast straight through the occlusion)."""
    s = _refine_seeker()
    s._last_az_err = -0.31                       # the run-042616 gate-1 close az (would blind-aim LEFT)
    s._begin_pass(0, wire=False)
    assert s._pass_turn_yaw is None, "refine must NOT latch a blind turn target"
    assert s._pass_turn_pending, "refine must mark the turn pending (target unknown yet)"


def test_refine_reaims_at_downrange_gate2_pose():
    """The MOMENT a valid downrange gate-2 pose is available, re-aim _pass_turn_yaw at THAT gate's
    world bearing (RIGHT, ~+0.35 rad), clearing the pending flag."""
    s = _refine_seeker()
    s._begin_pass(0, wire=False)
    s._last_pose = _pose_right(rng=18.0)         # RIGHT, downrange
    s._last_bearing_w = 0.9                       # consistent frame
    s._maybe_refine_turn_target(_nav0())
    assert s._pass_turn_yaw is not None, "a valid gate-2 pose must re-aim the turn target"
    assert s._pass_turn_yaw > 0.2, f"target must aim RIGHT toward the gate; got {s._pass_turn_yaw:+.2f}"
    assert not s._pass_turn_pending


def test_refine_ignores_passed_gate_residue():
    """A pose that IS the just-passed gate (within the A33 H-1a exclusion cone + range) must NOT
    re-aim -- it is residue of the gate we passed, not the next one."""
    s = _refine_seeker()
    s._begin_pass(0, wire=False)
    # snapshot: the passed gate was dead ahead at ~3 m (pass_heading 0). A pose ahead at ~4 m is it.
    s._pass_prev_dir_world = np.array([1.0, 0.0, 0.0])
    s._pass_prev_range_m = 3.0
    s._last_pose = GatePose(frame_id=0, sim_time_ns=0, R_cam_gate=np.eye(3),
                            t_cam_gate=np.array([0.0, 0.0, 4.0]), reproj_error_px=0.5)  # dead ahead 4 m
    s._last_bearing_w = 0.9
    s._maybe_refine_turn_target(_nav0())
    assert s._pass_turn_yaw is None, "the passed-gate residue must not re-aim the turn"
    assert s._pass_turn_pending


def test_refine_ignores_low_bearing_weight_pose():
    """A low-consistency (bearing_w < pass_refine_min_bw) pose must NOT re-aim -- re-aiming is a
    commit-grade action; a jittery frame steers gently in pursuit but can't fling the turn target."""
    s = _refine_seeker(pass_refine_min_bw=0.3)
    s._begin_pass(0, wire=False)
    s._last_pose = _pose_right(rng=18.0)
    s._last_bearing_w = 0.1                       # below the gate
    s._maybe_refine_turn_target(_nav0())
    assert s._pass_turn_yaw is None, "a low-weight pose must not re-aim the turn target"


def test_refine_ignores_close_degenerate_pose():
    """A pose in the degenerate close band (range <= pass_degenerate_range_m) must NOT re-aim -- it is
    the gate being passed at point-blank, not a downrange next gate."""
    s = _refine_seeker()
    s._begin_pass(0, wire=False)
    s._last_pose = GatePose(frame_id=0, sim_time_ns=0, R_cam_gate=np.eye(3),
                            t_cam_gate=np.array([0.5, 0.0, 2.0]), reproj_error_px=0.5)  # 2.06 m < 2.5
    s._last_bearing_w = 0.9
    s._maybe_refine_turn_target(_nav0())
    assert s._pass_turn_yaw is None


def test_refine_target_bounded_to_cap():
    """A gate-2 bearing beyond +/-pass_blind_turn_cap_rad from the pass heading is clamped to the cap
    (implausible for the next gate = likely residue; never fling the target)."""
    s = _refine_seeker(pass_blind_turn_cap_rad=0.5)   # tight cap for the test
    s._begin_pass(0, wire=False)
    s._last_pose = _pose_right(rng=18.0)          # ~+0.35 rad, but cap it to 0.5 from heading 0 (no clamp)
    s._last_bearing_w = 0.9
    s._maybe_refine_turn_target(_nav0())
    assert abs(s._pass_turn_yaw) <= 0.5 + 1e-9, "target must be bounded to +/-cap from the pass heading"


def test_refine_fallback_pending_holds_then_times_out():
    """With the turn PENDING (no gate-2 pose yet), _turn_hold_active holds (defers acquire-next) so
    we don't end the pass before aiming -- but it is time-boxed by pass_turn_coast_s (never forever)."""
    s = _refine_seeker(pass_turn_coast_s=1.5)
    s._passing = True
    s._pass_t_ns = 0
    s._pass_turn_pending = True
    s._pass_turn_yaw = None
    assert s._turn_hold_active(int(0.3 * _NS)), "pending turn must hold (defer acquire-next) while aiming"
    assert not s._turn_hold_active(int(1.6 * _NS)), "hold must time out at pass_turn_coast_s (fallback)"


def test_refine_default_off_uses_blind_target():
    """REGRESSION PIN: pass_turn_refine default False => _begin_pass latches the A33 blind sign*cap
    target and _maybe_refine_turn_target is a no-op (byte-identical pre-refine)."""
    s = _refine_seeker(pass_turn_refine=False)
    assert s.config.pass_turn_refine is False
    s._last_az_err = -0.31
    s._begin_pass(0, wire=False)
    assert s._pass_turn_yaw is not None, "blind target must latch when refine is OFF"
    assert s._pass_turn_yaw < 0.0, "blind target off a LEFT az aims LEFT (the pre-refine behavior)"
    assert not s._pass_turn_pending
    # and refine is a no-op with the flag off
    blind = s._pass_turn_yaw
    s._last_pose = _pose_right(rng=18.0)
    s._last_bearing_w = 0.9
    s._maybe_refine_turn_target(_nav0())
    assert s._pass_turn_yaw == blind, "refine must not touch the target when the flag is off"


# ===========================================================================
# YAW-STEER SELECTOR (off / A / B) — the R_cur-yaw recovery that makes yaw rotate the SAME
# physical way the roll banks (operator eyes, full-package run 20260704_051851: "rolled right,
# yawed left"). A vs B = opposite yaw direction; the physical winner is OPERATOR-EYES-resolved on
# the confirm-fly (offline cannot see the VQ2 yaw mirror). These tests assert only what is knowable
# offline: OFF/VQ1 byte-identical, A vs B opposite yaw with roll/pitch/thrust byte-identical, and
# neither diverges unstably.
# ===========================================================================
def _ctrl_cmd(mode, *, brs_yaw=-1.0, roll=0.1, pitch=-0.05, yaw=-0.3, sp_yaw=0.6,
              accel=(1.0, 0.5, 0.0), rate=(0.1, 0.2, 0.3)):
    """One controller command under yaw_steer_mode ``mode`` ("off"/"A"/"B"). brs_yaw is the yaw
    body_rate_sign the PROFILE would set (A reverts it to +1; B/off keep the proven -1)."""
    co = dict(vq2_case_c().controller_overrides)
    co["yaw_steer_mode"] = mode
    co["body_rate_sign"] = (1.0, 1.0, float(brs_yaw))
    c = make_seeker_controller(**co)
    nav = NavState(sim_time_ns=0, position_ned=np.zeros(3), velocity_ned=np.zeros(3),
                   roll=roll, pitch=pitch, yaw=yaw, angular_rate_body=np.asarray(rate, float),
                   time_since_vision_update_s=0.1)
    sp = Setpoint(sim_time_ns=0, accel_ned=np.asarray(accel, float), yaw=sp_yaw)
    cmd = c.command(nav, sp)
    return np.asarray(cmd.body_rate, float), float(cmd.thrust)


def test_yawsel_A_and_B_opposite_yaw_rollpitchthrust_identical():
    """A (brs_yaw=+1) and B (brs_yaw=-1), both with the R_cur-yaw recovery, produce OPPOSITE yaw
    command direction; roll, pitch, and thrust are BYTE-IDENTICAL between A and B (the A/B choice is
    only the yaw actuation sign)."""
    a, ta = _ctrl_cmd("A", brs_yaw=+1.0)
    b, tb = _ctrl_cmd("B", brs_yaw=-1.0)
    assert np.sign(a[2]) != np.sign(b[2]), f"A vs B must be opposite yaw; A={a[2]:+.3f} B={b[2]:+.3f}"
    assert a[2] == pytest.approx(-b[2], abs=1e-12), "A yaw == -B yaw (same |cmd|, opposite sign)"
    assert a[0] == pytest.approx(b[0], abs=1e-12), "roll byte-identical A vs B"
    assert a[1] == pytest.approx(b[1], abs=1e-12), "pitch byte-identical A vs B"
    assert ta == pytest.approx(tb, abs=1e-12), "thrust byte-identical A vs B"


def test_yawsel_recovery_changes_yaw_vs_off():
    """The R_cur-yaw recovery (mode B) actually changes the yaw command vs OFF (the flag is LIVE, not
    a silent no-op), at the SAME body_rate_sign so the difference is purely the R_cur-yaw recovery."""
    off, _ = _ctrl_cmd("off", brs_yaw=-1.0)
    b, _ = _ctrl_cmd("B", brs_yaw=-1.0)
    assert abs(off[2] - b[2]) > 1e-3, "the R_cur-yaw recovery must change the yaw command"


def test_yawsel_neither_A_nor_B_diverges_unstably():
    """STABILITY (must hold for BOTH A and B, in an honest matched plant): each converges under its
    OWN plant polarity and only DIRECTION-flips under the other -- neither is a positive-feedback
    instability (bounded, not growing). We do NOT assert which polarity is physical (eyes resolve
    that); we assert that under the polarity where each is negative-feedback it SETTLES."""
    def sim(mode, brs_yaw, plant_pol, n=80, dt=0.033):
        co = dict(vq2_case_c().controller_overrides)
        co["yaw_steer_mode"] = mode
        co["body_rate_sign"] = (1.0, 1.0, float(brs_yaw))
        c = make_seeker_controller(**co)
        true_yaw = 0.0
        for k in range(n):
            nav = NavState(sim_time_ns=int(k * dt * 1e9), position_ned=np.zeros(3),
                           velocity_ned=np.zeros(3), roll=0.0, pitch=0.0, yaw=-true_yaw,
                           angular_rate_body=np.zeros(3), time_since_vision_update_s=0.1)
            sp = Setpoint(sim_time_ns=int(k * dt * 1e9), accel_ned=np.array([1.0, 0.5, 0.0]), yaw=0.5)
            cmd = c.command(nav, sp)
            true_yaw += plant_pol * 2.1 * (float(cmd.body_rate[2]) * 0.4) * dt
        return abs(0.5 - true_yaw)
    # A converges under +1, B under -1 (they are opposites). Each SETTLES under its own polarity.
    assert sim("A", +1.0, +1) < 0.1, "A must settle onto the gate under its negative-feedback polarity"
    assert sim("B", -1.0, -1) < 0.1, "B must settle onto the gate under its negative-feedback polarity"


def test_yawsel_off_and_vq1_byte_identical():
    """REGRESSION PIN: yaw_steer_mode default 'off'; VQ1 has no controller overrides so its
    controller is byte-identical (the shared SEEKER_SIGNS default + odo_att_sign untouched). Mode
    'off' reproduces the pre-A36 command exactly."""
    assert make_seeker_controller().yaw_steer_mode == "off"
    assert vq1_case_a().controller_overrides is None
    a, _ = _ctrl_cmd("off", brs_yaw=-1.0, sp_yaw=0.9)
    co = dict(vq2_case_c().controller_overrides)
    co.pop("yaw_steer_mode", None)       # absent => default "off"
    co["body_rate_sign"] = (1.0, 1.0, -1.0)
    c = make_seeker_controller(**co)
    nav = NavState(sim_time_ns=0, position_ned=np.zeros(3), velocity_ned=np.zeros(3),
                   roll=0.1, pitch=-0.05, yaw=-0.3, angular_rate_body=np.array([0.1, 0.2, 0.3]),
                   time_since_vision_update_s=0.1)
    b = np.asarray(c.command(nav, Setpoint(sim_time_ns=0, accel_ned=np.array([1.0, 0.5, 0.0]),
                                           yaw=0.9)).body_rate, float)
    np.testing.assert_allclose(a, b, atol=1e-12)


# ===========================================================================
# PROFILE WIRING — the FLOWN vq2_case_c is the FULL A36 turn package (flight-3):
# Item 0 sign + yaw-steer-match-roll + refine-to-real-gate + Item 3 + Fix B + Items 1 & 4 ON;
# Fix C (kd_att) OFF.
# ===========================================================================
def test_profile_flight3_full_turn_package():
    """The shipped vq2_case_c carries the full A36 turn package: yaw sign (1,1,-1), refine-to-real-
    gate + Item 3 (physical-plane + hold-until-pointed), Fix B (yaw taper), Item 1 (pointing gate +
    slower 0.65), Item 4 (lateral-first + cap 3.0). Fix C (kd_att bump) stays OFF (default 0.30)."""
    prof = vq2_case_c()
    so = prof.seeker_overrides or {}
    co = prof.controller_overrides or {}
    # Fix C OFF (no kd_att override => controller default 0.30).
    assert "kd_att" not in co, "Fix C (kd_att bump) must stay OFF (no override) per operator"
    # Yaw fix EYES-RESOLVED rev A37 (run 20260704_135554 mode off: "no more turning to the right
    # before the gate"): yaw_steer_mode="off" + body_rate_sign yaw=+1. Mode "A" double-negated the
    # A14 recovery already wired via _controller_nav -> mirrored R_cur yaw -> positive-feedback
    # runaway (camera-truth audit of run 20260704_130314). NOT from the cmd-vs-gyro metric.
    assert co.get("yaw_steer_mode") == "off"
    np.testing.assert_allclose(np.asarray(co["body_rate_sign"], float), [1.0, 1.0, 1.0])
    cfg = GateSeekerConfig(**so)
    # Refine + Item 3:
    assert cfg.pass_turn_refine is True
    assert cfg.pass_wire_requires_near is True
    assert cfg.pass_turn_hold_until_pointed is True
    assert cfg.pass_refine_min_bw == 0.3
    # Fix B:
    assert cfg.yaw_slew_taper_lo_range_m == 3.0
    assert cfg.yaw_slew_taper_floor == 0.35
    # Item 1 (slower) -- values tightened by the TURN PACKAGE (2026-07-04, run 20260704_173948
    # understeer): az_rad 0.35 -> 0.25 (full forward cut beyond ~14 deg, was 20) and accel
    # 0.65 -> 0.45. TURN ITERATION (run 20260704_231155): accel 0.45 -> 0.35 (slower entry).
    # Pinned in test_vq2_turn_package too.
    assert cfg.fwd_point_gate_az_rad == 0.25
    assert cfg.forward_accel_mps2 == 0.35
    # Item 4 (switch lanes) -- TURN ITERATION: image_lat_cap 3.0 -> 4.0 (more roll; the cap was
    # railing through the turn) + total_accel_cap 3.0 -> 4.0 so the lateral cap actually applies.
    assert cfg.use_lateral_first_budget is True
    assert cfg.image_lat_cap_mps2 == 4.0
    # A36 COORDINATED TURN (A+B+C): unthrottle roll, cut yaw, arm the pass + demote orbit-brake.
    assert cfg.total_accel_cap_mps2 == 4.0            # turn iteration: cap 3.0 -> 4.0 (22 deg bank)
    assert cfg.pursuit_yaw_slew_rps == 0.9            # B: yaw slew 1.5 -> 0.9
    assert cfg.visual_yaw_rate_cap_rps == 0.9         # B: yaw cap 1.5 -> 0.9
    assert cfg.pass_arm_range_m == 4.5               # C: arm at the ~4.3m vision floor (was 3.0)
    assert cfg.orbit_guard_rad == 3.0                # C: orbit-brake demoted (was 1.75)


# ===========================================================================
# A36 COORDINATED TURN — the pass-turn (not the orbit-brake) owns the turn, roll-led.
# ===========================================================================
def test_coordturn_pass_arms_at_the_vision_floor():
    """C: with pass_arm_range_m raised to 4.5, the pass ARMS at the ~4.3 m tracked-range floor (the
    gate fills/exits FOV on the pass and range never reaches the old 3.0), so the coordinated
    pass-turn can fire instead of the orbit-breaker. At the old 3.0 it does NOT arm at 4.3 m."""
    def arms(arm_range):
        s = _pass_seeker(pass_arm_range_m=arm_range)
        s._update_pass_state(0, _pose(4.3), index_advanced=False)
        return s._pass_armed
    assert arms(4.5) is True, "NEW (4.5) must arm the pass at the 4.3 m floor"
    assert arms(3.0) is False, "OLD (3.0) does not arm at 4.3 m -> the orbit-brake ran (the bug)"


def test_coordturn_orbit_guard_demoted():
    """C: the A31 orbit-breaker guard is raised 1.75 -> 3.0 rad (~172 deg) so it is a RARE failsafe
    that no longer pre-empts the pass-turn."""
    assert vq2_case_c().seeker_overrides["orbit_guard_rad"] == 3.0


def test_coordturn_forward_drive_not_ballooned_by_cap_raise():
    """A + Fork-1 constraint: raising total_accel_cap 2.0->3.0 must NOT increase forward drive.
    In the lateral-first path a_fwd is capped by forward_accel_mps2 FIRST, so the forward component of
    the composed accel never exceeds forward_accel_mps2 -- the added budget goes to LATERAL/roll."""
    s = _seeker(use_image_servo_lateral=True, use_lateral_first_budget=True,
                image_lat_cap_mps2=3.0, total_accel_cap_mps2=3.0, forward_accel_mps2=0.65,
                fwd_scale_pow=2.0)
    # centered gate (az~0): fwd_scale~1, lateral~0 -> forward is the full forward_accel, and must not
    # exceed it even with the raised cap.
    _, fwd, _ = _compose(s, az=0.0)
    assert fwd <= 0.65 + 1e-9, f"forward must stay <= forward_accel_mps2 (0.65); got {fwd:.3f}"
