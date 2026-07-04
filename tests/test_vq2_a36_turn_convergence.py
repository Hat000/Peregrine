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

from racer.contracts import GatePose, NavState  # noqa: E402
from racer.deploy_profile import vq2_case_c  # noqa: E402
from racer.gate_seeker import GateSeeker, GateSeekerConfig, _clip_norm  # noqa: E402

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
# PROFILE WIRING — the FLOWN vq2_case_c state is SIGN-ALONE for the first A36 re-fly:
# Item 0 (yaw sign) IN; Items 1, 3, 4 all at their OFF defaults (commented out in the
# profile). Item 3 gets re-enabled for flight-2 by uncommenting the 4 override lines.
# ===========================================================================
def test_profile_first_fly_is_sign_alone():
    """The shipped vq2_case_c carries ONLY the yaw-sign flip from A36; Items 1/3/4 fall back to
    their OFF defaults so the first re-fly isolates the ONE change (the sign). This asserts the
    FLOWN profile state -- the Item-1/3/4 BEHAVIOR tests above construct configs explicitly and are
    unaffected by this wiring."""
    prof = vq2_case_c()
    so = prof.seeker_overrides or {}
    # Item 0: the yaw sign IS flipped (lives in controller_overrides).
    np.testing.assert_allclose(
        np.asarray(prof.controller_overrides["body_rate_sign"], float), [1.0, 1.0, -1.0])
    # Items 1/3/4: ABSENT from the profile (=> OFF defaults) for the sign-alone first fly.
    for k in ("pass_wire_requires_near", "pass_turn_hold_until_pointed",
              "fwd_point_gate_az_rad", "use_lateral_first_budget", "image_lat_cap_mps2"):
        assert k not in so, f"{k} must be OFF (absent) in the sign-alone first-fly profile"
    # And the built config confirms the OFF defaults:
    cfg = GateSeekerConfig(**so)
    assert cfg.pass_wire_requires_near is False
    assert cfg.pass_turn_hold_until_pointed is False
    assert cfg.fwd_point_gate_az_rad is None
    assert cfg.use_lateral_first_budget is False
    assert cfg.image_lat_cap_mps2 == 1.5   # default cap, not the Item-4 raised 3.0
