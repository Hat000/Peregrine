"""A31 — immediate turn on the wire gate-pass + slow approach + IMU-consistency bearing
gate + startup-swell fix.

Run 20260703_160715 (A30): the drone threaded gate 0 (first ever) but the turn to gate 1 was
STRUCTURALLY LATE — ``pass_coast_s`` is a 1.2 s blind ACCELERATING glide on a frozen heading, so
the drone carried ~3.9 m/s into a gate ~30 deg off-axis at short range; the world LOS then swept
203 deg in 4.9 s and the yaw lag-followed it all the way to BACKWARDS (a tail-chase orbit, not a
yaw bug). Secondary: a one-frame close-range track hop (~0.33 rad total bearing + ~0.83 m of
offset_z) slipped UNDER the fixed 0.35 rad continuity gate at t=3.25, snapped the A30 lateral to
the -1.5 rail (the left roll) and latched a 2x-exaggerated z offset into the vertical estimator
(the floor tap); and the SETTLE hold's wide thrust band [0.6, 1.4]x hover let the cold estimator
rail the collective (~+1.2 m/s injected upward before pursuit began).

THE A31 FIXES (spec handoff/vq2_a31_immediate_turn_spec_2026-07-03.md, all default-off /
legacy-default, activated only via vq2_case_c seeker_overrides):

1. ``pass_wire_coast_s`` — on the AUTHORITATIVE wire pass (RACE_STATUS active_gate_index
   increment) the drone is PAST the gate plane: use a 0.25 s acquire-eligibility window instead
   of the blind 1.2 s; a vision-committed pass confirmed by the wire MID-GLIDE upgrades to the
   fast window. ``pass_coast_accel_mps2`` overridden to 0.0 (spend momentum, don't build it) and
   ``reramp_forward_after_pass`` re-ramps the forward drive from zero after EVERY pass (point
   before pushing, enforced). ``forward_accel_mps2`` 1.2 -> 0.8 in the profile slows the whole
   approach (commander directive): less carried speed => lower LOS sweep rate at acquisition =>
   the A30 +/-1.5 lateral brake (saturated 63% of the A30 chase) becomes sufficient.
2. ORBIT-BREAKER: the guard observable is CUMULATIVE unwrapped LOS rotation since acquisition
   (``orbit_guard_rad``) — the instantaneous bearing stayed SMALL through the whole 203 deg whip
   (az pinned +0.3..0.5, the yaw obediently lag-following), so an instantaneous-bearing guard is
   structurally blind to it. Trip => a bounded ``orbit_break`` brake regime (forward 0, capped
   lateral toward the apparent gate, yaw HELD); second trip on one acquisition => drop the track
   and hold (refuse the spin). ``orbit_yaw_clamp_rad`` hard-pins the slewed pursuit yaw to
   +/-2.4 rad of the yaw at acquisition — never turn to backwards chasing a gate.
3. IMU-CONSISTENCY BEARING GATE (``use_imu_bearing_gate`` — the operator-directed REPLACEMENT
   for the tightened-fixed-threshold band-aid): gates are STATIC, so over one inter-frame dt the
   gate's bearing can only change as fast as the drone's own MEASURED motion — rotation
   (AHRS/gyro attitude delta, measured cleanly) + translation parallax (bounded, ∝ dt/range).
   Predict the bearing from the last tracked WORLD direction (capture-time attitudes rotate the
   camera levers, so the measured rotation is compensated EXACTLY — the horizontal analog of the
   A28 vertical complementary filter) and reject a vision bearing whose deviation exceeds
   sensor-noise + the parallax allowance — REGARDLESS of the deviation's absolute size (no fixed
   threshold a right-sized hop can defeat). Replaces ``track_max_bearing_jump_rad`` for
   vq2_case_c; the legacy fixed check stays for VQ1 byte-identity and as the fallback when no
   capture-time attitude is available.
4. STARTUP SWELL: settle-hold thrust band tightened to [0.90, 1.12] x hover (profile override
   only) so the cold, unseeded estimator can't rail the collective. ``image_lat_slew_mps3``
   rate-limits the A30 lateral demand (defense-in-depth: even an ACCEPTED noisy bearing can't
   snap the roll to the rail in one tick).

[VQ2 slow-is-smooth, A31, 2026-07-03]
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from racer.contracts import ControlMode, Frame, GatePose, NavState  # noqa: E402
from racer.deploy_profile import get_profile, vq1_case_a, vq2_case_c  # noqa: E402
from racer.frames import R_camera_from_body, R_world_from_body  # noqa: E402
from racer.gate_seeker import (  # noqa: E402
    GateSeeker, GateSeekerConfig, make_seeker_controller,
)

_NS = 1_000_000_000
_FIXTURE = Path(__file__).resolve().parent / "data" / "vq2_a31_byteid_reference.npz"

# The FROZEN pre-A31 vq2_case_c override dicts (what 3dcbcc3 shipped). The byte-identity replay
# runs the post-A31 code with THESE overrides (every A31 field at its legacy default) and must
# reproduce the pre-A31 command stream bit-for-bit.
_PRE_A31_SEEKER_OVERRIDES = {
    "egress_freeze_attitude": True, "hold_last_demand_s": 0.6,
    "true_attitude_from_ahrs": True, "use_image_servo_lateral": True,
    "total_accel_cap_mps2": 2.0, "pursuit_yaw_slew_rps": 1.5,
}
_PRE_A31_CONTROLLER_OVERRIDES = {
    "kp_att": 4.0, "body_rate_slew_max_rps2": 8.0,
    "ff_owns_horizontal": True, "ff_owns_vertical": True,
    "body_rate_sign": (1.0, 1.0, 1.0),
    "kp_alt": 0.0, "gate_pd_vertical": True,
    "ff_vertical_kd_alt": 0.06, "ff_vertical_vz_lp_alpha": 0.8, "kp_gate": 0.04,
    "alt_thrust_lo": 0.15, "alt_thrust_slew_per_s": 2.0,
}


def _wrap(a):
    return float(np.arctan2(np.sin(a), np.cos(a)))


def _nav(t_ns, *, yaw=0.0, vz=0.0):
    """A level NavState past the cold-start (finite tsv releases the anchor when needed)."""
    return NavState(
        sim_time_ns=int(t_ns),
        position_ned=np.array([0.0, 0.0, -2.5]),
        velocity_ned=np.array([0.0, 0.0, float(vz)]),
        roll=0.0,
        pitch=0.0,
        yaw=float(yaw),
        time_since_vision_update_s=0.05,
    )


def _pose_cam(az, el, r, t_ns, fid=0):
    """A GatePose whose camera bearing is exactly (az, el) at range r (camera optical frame:
    +x right, +y down, +z forward; az = atan2(x, z), el = atan2(y, z))."""
    d = np.array([np.tan(float(az)), np.tan(float(el)), 1.0])
    t_cam = float(r) * d / np.linalg.norm(d)
    return GatePose(frame_id=int(fid), sim_time_ns=int(t_ns),
                    R_cam_gate=np.eye(3), t_cam_gate=t_cam, reproj_error_px=0.0)


def _pose_from_world_dir(d_world, r, rpy_capture, t_ns, fid=0):
    """A GatePose AS CAPTURED: the gate sits along unit world direction ``d_world`` at range
    ``r`` and the drone's attitude at the capture instant was ``rpy_capture`` — the camera lever
    encodes the gate's bearing IN THAT frame. Rotating it back with the capture attitude
    recovers d_world exactly (the IMU-consistency prediction seam)."""
    d_world = np.asarray(d_world, dtype=np.float64)
    d_world = d_world / np.linalg.norm(d_world)
    tr, tp, ty = rpy_capture
    d_body = R_world_from_body(tr, tp, ty).T @ d_world
    t_cam = R_camera_from_body() @ (float(r) * d_body)
    return GatePose(frame_id=int(fid), sim_time_ns=int(t_ns),
                    R_cam_gate=np.eye(3), t_cam_gate=t_cam, reproj_error_px=0.0)


def _pursuit_cfg(**overrides):
    """A GateSeekerConfig with every ramp/slew/settle inert so a direct
    ``_visual_pursuit_command`` call is a pure function of (pose, nav)."""
    cfg = GateSeekerConfig(launch_ramp_s=0.0, pursuit_ramp_s=0.0, forward_ramp_s=0.0,
                           pursuit_yaw_slew_rps=0.0, settle_s=0.0, use_spawn_egress=False)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _frame(fid, t_ns):
    return Frame(frame_id=int(fid), sim_time_ns=int(t_ns),
                 image_bgr=np.ones((360, 640, 3), dtype=np.uint8))


def _e_right(psi):
    return np.array([-np.sin(psi), np.cos(psi), 0.0])


# ===========================================================================
# 0. BYTE-IDENTITY REPLAY (the flag-off guarantee, bit for bit)
# ===========================================================================
def _replay_scenario(seeker_overrides, controller_overrides):
    """A deterministic multi-regime ``command_visual`` replay traversing settle -> anchor ->
    egress -> pursuit -> pass ARM -> degenerate COMMIT -> blind glide (with a MID-GLIDE wire
    index advance, the A31 upgrade seam's flag-off path) -> acquire-next -> pursuit -> hold.
    The pose stream is INJECTED via the ``_last_pose`` ZOH seam (frame=None every tick) so no
    detector runs — pure guidance + controller. Returns an (N, 4) float64 array of
    [body_rate x, y, z, thrust] per tick, recorded exactly."""
    seeker = GateSeeker(config=GateSeekerConfig(**seeker_overrides),
                        controller=make_seeker_controller(**controller_overrides))
    dt_ns = _NS // 30
    out = []
    gi = 0
    for k in range(240):
        t_ns = k * dt_ns
        if 45 <= k < 75:
            pose = _pose_cam(0.05, 0.03, 10.0 - 0.2 * (k - 45), t_ns, fid=k)   # approach
        elif 75 <= k < 80:
            pose = _pose_cam(0.02, 0.02, 2.8, t_ns, fid=k)                     # arms (<=3.0)
        elif 80 <= k < 82:
            pose = _pose_cam(0.02, 0.02, 2.3, t_ns, fid=k)                     # degenerate commit
        elif 100 <= k < 200 or k >= 215:
            pose = _pose_cam(0.4, 0.05, 9.0, t_ns, fid=k)                      # next gate
        else:
            pose = None                                                        # gaps / glide
        seeker._last_frame_id = k          # scripted injection: bypass the detector entirely
        seeker._last_pose = pose
        if k == 90:
            gi = 1                          # wire advance MID-GLIDE (upgrade seam, flag-off path)
        cmd = seeker.command_visual(_nav(t_ns), None, gi)
        assert cmd.mode is ControlMode.BODY_RATE
        out.append([float(cmd.body_rate[0]), float(cmd.body_rate[1]),
                    float(cmd.body_rate[2]), float(cmd.thrust)])
    return np.asarray(out, dtype=np.float64)


def test_a31_defaults_are_legacy():
    """Every A31 field ships at its legacy/off default — the VQ1 byte-identity precondition."""
    cfg = GateSeekerConfig()
    assert cfg.pass_wire_coast_s is None
    assert cfg.pass_coast_accel_mps2 == 1.2          # field default UNCHANGED (override only)
    assert cfg.forward_accel_mps2 == 1.2             # field default UNCHANGED (override only)
    assert cfg.reramp_forward_after_pass is False
    assert cfg.orbit_guard_rad == 0.0
    assert cfg.orbit_break_s == 1.0
    assert cfg.orbit_break_exit_az_rad == 0.15
    assert cfg.orbit_break_exit_rate_rps == 0.3
    assert cfg.orbit_yaw_clamp_rad == 0.0
    assert cfg.use_imu_bearing_gate is False
    assert cfg.image_lat_slew_mps3 == 0.0
    assert cfg.hold_thrust_lo_frac == 0.6            # field default UNCHANGED (override only)
    assert cfg.hold_thrust_hi_frac == 1.4
    assert cfg.track_max_bearing_jump_rad == 0.35    # legacy fixed gate UNTOUCHED


def test_flag_off_byte_identical_to_pre_a31_reference():
    """THE OFF-PATH PIN: the post-A31 code, with every A31 field at its default, reproduces the
    PRE-A31 (3dcbcc3) command stream BIT FOR BIT over the full multi-regime replay — for both
    (a) the bare default config (the VQ1/case-A seeker) and (b) the frozen pre-A31 vq2_case_c
    override bundle (image servo / bridge / true-attitude ON, A31 fields at defaults). The
    reference fixture was recorded by running THIS scenario at 3dcbcc3 before the A31 edits."""
    ref = np.load(_FIXTURE)
    got_default = _replay_scenario({}, {})
    np.testing.assert_array_equal(got_default, ref["default"],
                                  err_msg="default-config command stream changed (VQ1 path!)")
    got_vq2 = _replay_scenario(_PRE_A31_SEEKER_OVERRIDES, _PRE_A31_CONTROLLER_OVERRIDES)
    np.testing.assert_array_equal(got_vq2, ref["vq2_pre_a31"],
                                  err_msg="pre-A31 vq2 override bundle command stream changed")


# ===========================================================================
# 1. FIX 1 — immediate turn on the wire pass (fast window + mid-glide upgrade)
# ===========================================================================
def test_effective_coast_wire_vs_vision_vs_legacy():
    """The window selector: wire-committed pass + configured fast window => 0.25 s; a
    vision-committed pass keeps the legacy 1.2 s; a wire pass with the field at None (VQ1)
    keeps the legacy 1.2 s (state written, never read)."""
    s = GateSeeker(config=_pursuit_cfg(pass_wire_coast_s=0.25))
    s._begin_pass(0, wire=True)
    assert s._pass_wire is True
    assert s._effective_pass_coast_s() == pytest.approx(0.25)
    s._end_pass()
    assert s._pass_wire is False                       # cleared for the next gate
    s._begin_pass(0, wire=False)
    assert s._effective_pass_coast_s() == pytest.approx(s.config.pass_coast_s)
    # legacy (None): even a wire pass keeps the 1.2 s window — VQ1 byte-identity.
    s2 = GateSeeker(config=_pursuit_cfg())
    s2._begin_pass(0, wire=True)
    assert s2._pass_wire is True
    assert s2._effective_pass_coast_s() == pytest.approx(s2.config.pass_coast_s)
    # positional-arg compat (the A29 test calls _begin_pass(123)).
    s3 = GateSeeker(config=_pursuit_cfg())
    s3._begin_pass(123)
    assert s3._pass_wire is False


def test_wire_pass_turns_within_the_fast_window():
    """END-TO-END through command_visual: the wire flips active_gate_index (no arm, the real
    20260703_160715 signature) -> regime 'pass_wire'; a fresh downrange pose 0.30 s later is
    ACTED ON (regime 'pursuit', yaw_des toward the new gate) — turn latency 0.30 s, was 1.22 s.
    The same script with pass_wire_coast_s=None stays in the blind glide (regime 'pass')."""
    for fast, want_regime_at_030 in ((True, "pursuit"), (False, "pass_wire")):
        cfg = _pursuit_cfg(pass_wire_coast_s=0.25 if fast else None)
        s = GateSeeker(config=cfg)
        # tick 0: anchored pursuit on gate 0 (finite tsv anchors; ramps inert).
        s._last_frame_id, s._last_pose = 0, _pose_cam(0.02, 0.0, 6.0, 0, fid=0)
        s.command_visual(_nav(0), None, 0)
        assert s._last_regime == "pursuit"
        # tick 1 (t=0.1 s): the WIRE advances the index -> pass committed by the wire.
        s._last_frame_id, s._last_pose = 1, None
        s.command_visual(_nav(int(0.1 * _NS)), None, 1)
        assert s._last_regime == "pass_wire"
        assert s._pass_wire is True
        # tick 2 (t=0.4 s = 0.3 s after the flip): a fresh gate-1 pose at a real range.
        s._last_frame_id = 2
        s._last_pose = _pose_cam(0.5, 0.0, 9.0, int(0.4 * _NS), fid=2)
        s.command_visual(_nav(int(0.4 * _NS)), None, 1)
        assert s._last_regime == want_regime_at_030, \
            f"fast={fast}: expected {want_regime_at_030} 0.3 s after the wire flip"
        if fast:
            # turning toward gate 1 (the +20 deg camera mount shifts the world azimuth slightly).
            assert s._last_yaw_des == pytest.approx(0.5, abs=0.05)


def test_midglide_wire_upgrade():
    """A VISION-committed pass (degenerate close range) later CONFIRMED by the wire mid-glide
    upgrades to the fast window: poses become eligible pass_wire_coast_s after the commit."""
    cfg = _pursuit_cfg(pass_wire_coast_s=0.25)
    s = GateSeeker(config=cfg)
    # tick 0: pursuit, gate close enough to ARM + degenerate-commit (range 2.0 <= 2.5).
    s._last_frame_id, s._last_pose = 0, _pose_cam(0.0, 0.0, 2.0, 0, fid=0)
    s.command_visual(_nav(0), None, 0)
    assert s._passing and s._pass_wire is False
    assert s._last_regime == "pass_vis"
    # tick 1 (t=0.1): the wire confirms mid-glide -> upgrade to the fast window.
    s._last_frame_id, s._last_pose = 1, None
    s.command_visual(_nav(int(0.1 * _NS)), None, 1)
    assert s._pass_wire is True
    assert s._last_regime == "pass_wire"
    # tick 2 (t=0.3 s after commit): fresh downrange pose -> acquired (fast window elapsed).
    s._last_frame_id = 2
    s._last_pose = _sweep_pose(0.4, int(0.3 * _NS), 2)
    s.command_visual(_nav(int(0.3 * _NS)), None, 1)
    assert s._last_regime == "pursuit"


def test_wire_window_cannot_relock_the_passed_gate():
    """_pass_acquired_next keeps the range_m > pass_degenerate_range_m filter: inside the fast
    window a close-range (behind/degenerate) sighting does NOT end the pass."""
    cfg = _pursuit_cfg(pass_wire_coast_s=0.25)
    s = GateSeeker(config=cfg)
    s._begin_pass(0, wire=True)
    nav = _nav(int(0.3 * _NS))
    assert s._pass_acquired_next(nav, _pose_cam(0.0, 0.0, 2.0, int(0.3 * _NS))) is False
    assert s._pass_acquired_next(nav, _pose_cam(0.0, 0.0, 9.0, int(0.3 * _NS))) is True


# ===========================================================================
# 2. FIX 2 — stop accelerating through the pass + post-pass forward re-ramp
# ===========================================================================
def test_reramp_forward_after_pass():
    """With the flag ON, _end_pass restarts the forward-accel ramp clock: the first second of
    the next acquisition ramps the forward drive from ZERO again (point before pushing,
    enforced). Flag OFF: the ramp stays measured from the spawn release (legacy, 1.0 here)."""
    for flag, want_half in ((True, 0.5), (False, 1.0)):
        cfg = GateSeekerConfig(forward_ramp_s=1.0, reramp_forward_after_pass=flag)
        s = GateSeeker(config=cfg)
        s._release_t_ns = 0
        assert s._forward_accel_ramp(int(10.0 * _NS)) == 1.0     # long past the release ramp
        s._begin_pass(int(10.0 * _NS))
        s._end_pass(int(11.0 * _NS))                             # next gate acquired at t=11
        assert s._forward_accel_ramp(int(11.5 * _NS)) == pytest.approx(want_half)
        assert s._forward_accel_ramp(int(12.5 * _NS)) == 1.0


# ===========================================================================
# 3. FIX 3 — orbit-breaker (cumulative-LOS guard) + hard yaw-excursion clamp
# ===========================================================================
def _sweep_seeker(**cfg_over):
    """A pursuit seeker fed fresh poses whose WORLD LOS sweeps in 0.3 rad steps (a forming
    whip: each pose is fresh, dt = 0.1 s, drone yaw held at 0 so az stays instantaneous-small
    per step while the CUMULATIVE rotation grows — exactly the §1.2 observable)."""
    cfg = _pursuit_cfg(use_image_servo_lateral=True, **cfg_over)
    s = GateSeeker(config=cfg)
    s._last_yaw = 0.0
    return s


def _sweep_pose(theta_world, t_ns, fid, r=8.0):
    """A fresh pose whose WORLD LOS azimuth is exactly ``theta_world`` (level capture attitude —
    exact at ANY angle, unlike the tan-based camera builder)."""
    return _pose_from_world_dir([np.cos(theta_world), np.sin(theta_world), 0.0], r,
                                (0.0, 0.0, 0.0), t_ns, fid=fid)


def test_orbit_guard_trips_on_cumulative_los_and_brakes():
    """Sweep the LOS +0.3 rad per fresh pose: the cumulative integrator crosses 1.75 rad on the
    7th pose (2.1 > 1.75) -> regime 'orbit_break' with ZERO forward drive, the lateral CAPPED
    toward the CURRENT apparent gate, and the yaw setpoint HELD (not following the sweep)."""
    s = _sweep_seeker(orbit_guard_rad=1.75)
    yaw_before = None
    for i in range(7):
        t_ns = int(i * 0.1 * _NS)
        pose = _sweep_pose(0.3 * (i + 1), t_ns, i)   # az grows: drone yaw fixed at 0
        yaw_before = s._last_yaw
        s._visual_pursuit_command(_nav(t_ns), pose)
        if i < 6:
            assert s._last_regime != "orbit_break"
    assert s._last_regime == "orbit_break"
    assert abs(s._last_chase_dpsi) > 1.75
    # break command: forward scale 0, lateral = +cap toward the (right-of-nose) apparent gate.
    assert s._last_fwd_scale == 0.0
    assert s._last_alat == pytest.approx(s.config.image_lat_cap_mps2)
    # yaw setpoint HELD: the slewed heading did not advance on the break tick.
    assert s._last_yaw == pytest.approx(yaw_before)


def test_orbit_break_bounded_and_early_exit_rebases():
    """The break regime is bounded by orbit_break_s, exits EARLY when the gate re-centers with a
    low fresh-pose LOS drift, and REBASES the integrator so pursuit resumes cleanly."""
    s = _sweep_seeker(orbit_guard_rad=1.75, orbit_break_s=1.0)
    for i in range(7):
        t_ns = int(i * 0.1 * _NS)
        s._visual_pursuit_command(_nav(t_ns), _sweep_pose(0.3 * (i + 1), t_ns, i))
    assert s._last_regime == "orbit_break"
    # a centered, slow-drifting fresh pose -> early exit (az 0.05 < 0.15; drift 0.5 rad/s -> no,
    # use a small step: 0.02 rad over 0.1 s = 0.2 rad/s < 0.3) -> resume pursuit, rebased.
    t_ns = int(0.8 * _NS)
    s._visual_pursuit_command(_nav(t_ns), _sweep_pose(0.05, t_ns, 90))
    t_ns = int(0.9 * _NS)
    s._visual_pursuit_command(_nav(t_ns), _sweep_pose(0.07, t_ns, 91))
    assert s._last_regime == "pursuit"
    assert abs(s._last_chase_dpsi) < 1.75              # integrator rebased
    assert s._orbit_break_t_ns is None


def test_orbit_break_hard_timeout():
    """No re-centering: the break regime ends at orbit_break_s regardless (bounded — never a
    standing brake), then pursuit resumes on the rebased integrator."""
    s = _sweep_seeker(orbit_guard_rad=1.75, orbit_break_s=1.0)
    for i in range(7):
        t_ns = int(i * 0.1 * _NS)
        s._visual_pursuit_command(_nav(t_ns), _sweep_pose(0.3 * (i + 1), t_ns, i))
    trip_t = s._orbit_break_t_ns
    assert trip_t is not None
    # still off-center + drifting at the timeout tick -> exit on the clock.
    t_ns = trip_t + int(1.05 * _NS)
    s._visual_pursuit_command(_nav(t_ns), _sweep_pose(0.5, t_ns, 95))
    assert s._orbit_break_t_ns is None
    assert s._last_regime == "pursuit"


def test_second_trip_drops_track_and_holds():
    """Second guard trip on the SAME acquisition: refuse the spin — drop the track and fall to
    the no-detection hold (level, wait, reacquire clean)."""
    s = _sweep_seeker(orbit_guard_rad=1.75, orbit_break_s=0.2)
    fid = 0
    # trip 1 (7 sweeping poses), then run past the 0.2 s break window to resume pursuit.
    for i in range(7):
        t_ns = int(i * 0.1 * _NS)
        s._visual_pursuit_command(_nav(t_ns), _sweep_pose(0.3 * (i + 1), t_ns, fid))
        fid += 1
    assert s._orbit_trips == 1
    t_ns = int(0.95 * _NS)                             # past the 0.2 s break -> timeout exit
    s._visual_pursuit_command(_nav(t_ns), _sweep_pose(0.1, t_ns, fid)); fid += 1
    assert s._last_regime == "pursuit"
    # trip 2: sweep again from the rebased baseline.
    s._track_range_m, s._track_bearing = 8.0, np.array([0.1, 0.0])   # a live track to drop
    for i in range(7):
        t_ns = int((1.0 + i * 0.1) * _NS)
        s._visual_pursuit_command(_nav(t_ns), _sweep_pose(0.1 + 0.3 * (i + 1), t_ns, fid))
        fid += 1
        if s._last_regime == "hold":
            break
    assert s._last_regime == "hold"
    assert s._orbit_trips == 0                         # chase state reset for the CLEAN reacquisition
    assert s._track_range_m is None and s._track_bearing is None
    assert s._last_pose is None                        # ZOH cleared: genuinely wait for a NEW frame


def test_yaw_clamp_pins_excursion_from_acquisition():
    """orbit_yaw_clamp_rad: the slewed pursuit yaw physically cannot rotate past the clamp from
    the yaw at acquisition, whatever the bearing demands — the 'never turn to backwards' pin."""
    cfg = _pursuit_cfg(use_image_servo_lateral=True, orbit_yaw_clamp_rad=2.4,
                       pursuit_yaw_slew_rps=100.0)        # slew wide open: the clamp must bind
    s = GateSeeker(config=cfg)
    s._last_yaw = 0.0
    # first pursuit tick seeds the baseline at _last_yaw = 0; yaw_des = 3.0 > 2.4 -> clamped.
    t0 = 0
    s._visual_pursuit_command(_nav(t0), _sweep_pose(3.0, t0, 0))
    # second tick (dt so the wide-open slew would reach 3.0 without the clamp).
    t1 = int(0.5 * _NS)
    s._visual_pursuit_command(_nav(t1), _sweep_pose(3.0, t1, 1))
    assert abs(_wrap(s._last_yaw - 0.0)) <= 2.4 + 1e-9
    assert s._last_yaw == pytest.approx(2.4, abs=1e-6)


def test_chase_state_resets_with_the_gate():
    """The chase integrator describes ONE acquisition: _begin_pass/_end_pass/reset drop it."""
    s = _sweep_seeker(orbit_guard_rad=1.75)
    for i in range(4):
        t_ns = int(i * 0.1 * _NS)
        s._visual_pursuit_command(_nav(t_ns), _sweep_pose(0.3 * (i + 1), t_ns, i))
    assert abs(s._chase_dpsi) > 0.5
    s._begin_pass(0)
    assert s._chase_psi0 is None and s._chase_dpsi == 0.0 and s._orbit_trips == 0
    for i in range(4):
        t_ns = int((1 + i * 0.1) * _NS)
        s._visual_pursuit_command(_nav(t_ns), _sweep_pose(0.3 * (i + 1), t_ns, 10 + i))
    s.reset()
    assert s._chase_psi0 is None and s._chase_dpsi == 0.0
    assert s._orbit_break_t_ns is None and s._last_chase_dpsi is None


# ===========================================================================
# 4. FIX 4 (defense-in-depth) — lateral-demand slew
# ===========================================================================
def test_image_lat_slew_limits_rail_snap():
    """image_lat_slew_mps3=6: a one-frame full-FOV hop (az 0 -> 0.4, unslewed demand 0 -> 1.5)
    is rate-limited to 6 * dt per tick (0.2 at 30 Hz) — the t=3.25 rail-snap becomes a ramp.
    An honest bearing ramp (below the slew) is never limited. Flag OFF (0.0): legacy snap."""
    cfg = _pursuit_cfg(use_image_servo_lateral=True, image_lat_slew_mps3=6.0)
    s = GateSeeker(config=cfg)
    dt_ns = _NS // 30
    s._visual_pursuit_command(_nav(0), _sweep_pose(0.0, 0, 0))
    assert s._last_alat == 0.0
    s._visual_pursuit_command(_nav(dt_ns), _sweep_pose(0.4, dt_ns, 1))
    assert s._last_alat == pytest.approx(6.0 * (dt_ns / 1e9), rel=1e-6)   # 0.2, not 1.5
    s._visual_pursuit_command(_nav(2 * dt_ns), _sweep_pose(0.4, 2 * dt_ns, 2))
    assert s._last_alat == pytest.approx(0.4, rel=1e-6)
    # honest small step stays untouched: 8*(0.05-0.03) = 0.16 change from 0.4 -> allowed... the
    # DEMAND drops to 0.16; |0.16 - 0.4| = 0.24 > 0.2 -> limited to 0.4-0.2 = 0.2 this tick.
    s._visual_pursuit_command(_nav(3 * dt_ns), _sweep_pose(0.05, 3 * dt_ns, 3))
    assert s._last_alat == pytest.approx(0.2, rel=1e-6)
    # flag OFF: the legacy one-tick snap (the pre-A31 behaviour).
    off = GateSeeker(config=_pursuit_cfg(use_image_servo_lateral=True))
    off._visual_pursuit_command(_nav(0), _sweep_pose(0.0, 0, 0))
    off._visual_pursuit_command(_nav(dt_ns), _sweep_pose(0.4, dt_ns, 1))
    assert off._last_alat == pytest.approx(off.config.image_lat_cap_mps2)


# ===========================================================================
# 5. THE UPGRADE — IMU-consistency bearing gate (replaces the fixed-threshold band-aid)
# ===========================================================================
def _gate_seeker(**cfg_over):
    """A tracking seeker whose _valid_poses is scripted per-frame (no detector model). The
    attitude ring buffer is fed manually (raw same-epoch stamps; nav_owner None => identity)."""
    cfg = _pursuit_cfg(use_gate_track=True, **cfg_over)
    s = GateSeeker(config=cfg)
    s.detector = object()                      # pass the no-detector guard
    return s


def _detect(s, poses, fid, t_ns):
    """Run detect_gate_lever on a scripted candidate list (all stamped t_ns)."""
    s._valid_poses = lambda frame: poses       # instance-attr shadow: script the candidates
    return s.detect_gate_lever(_frame(fid, t_ns))


def test_bad_hop_rejected_where_legacy_accepted():
    """THE VALIDATION PIN — the actual t=3.25 bad frame of run 20260703_150755/160715 replayed:
    with the gate 0 tracked nearly centered at close range, ONE frame hops the solution 0.33 rad
    (left + down: at r=4 the down-component alone re-reads the vertical offset by ~+0.83 m) while
    the attitude delta between the two captures says the drone BARELY ROTATED (< 0.3 deg). The
    legacy fixed 0.35 rad gate ACCEPTS the hop (the flight's left-roll + floor-tap injector);
    the IMU-consistency gate REJECTS it: predicted bearing ~= previous (no measured rotation),
    deviation 0.33 rad >> noise(0.06) + parallax(4 m/s * 0.045 s / 4 m = 0.045) = 0.105 rad."""
    r, dt_s = 4.0, 0.045
    el_hop = 0.83 / r                                   # the +0.83 m down-read at range 4
    az_hop = -float(np.sqrt(0.33 ** 2 - el_hop ** 2))   # total hop magnitude 0.33 rad, leftward
    t0, t1 = 0, int(dt_s * _NS)
    prev = _pose_cam(0.015, 0.17, r, t0, fid=0)         # tracked: centered, opening below path
    hop = _pose_cam(0.015 + az_hop, 0.17 + el_hop, r, t1, fid=1)
    # sanity: the hop is exactly the under-the-old-gate size (<= 0.35, > 0.25).
    step = float(np.linalg.norm(GateSeeker._pose_bearing(hop) - GateSeeker._pose_bearing(prev)))
    assert 0.30 < step <= 0.35, f"harness sanity: hop bearing step {step:.3f} not the t=3.25 size"

    # LEGACY (the pre-A31 flight config): the hop passes the fixed 0.35 gate -> ACCEPTED.
    legacy = _gate_seeker()
    assert _detect(legacy, [prev], 0, t0) is prev
    assert _detect(legacy, [hop], 1, t1) is hop, "harness sanity: the old gate must ACCEPT the hop"

    # IMU-CONSISTENCY: attitude delta ~0 between captures -> prediction = previous bearing ->
    # the hop deviates 0.33 rad >> the noise+parallax allowance -> REJECTED (track coasts).
    imu = _gate_seeker(use_imu_bearing_gate=True)
    imu._append_att_hist(t0, (0.0, 0.0, 0.0))
    assert _detect(imu, [prev], 0, t0) is prev
    imu._append_att_hist(t1, (0.0, 0.0, 0.005))         # < 0.3 deg of measured rotation
    assert _detect(imu, [hop], 1, t1) is None, \
        "the IMU-consistency gate must REJECT the motion-inconsistent hop"
    assert imu._last_none_reason == "continuity_reject"
    assert imu._last_bearing_dev_rad == pytest.approx(0.33, abs=0.02)
    assert imu._last_bearing_allow_rad < 0.15


def test_legit_close_range_parallax_sweep_accepted():
    """NO FALSE REJECT: a legitimate close-range sweep CONSISTENT with bounded translation —
    world bearing rotating v*dt/r = 2.4*0.045/3 = 0.036 rad between frames at r=3 (the honest
    gate-0 crossing geometry), no rotation — is ACCEPTED (0.036 < 0.06 + 4*0.045/3 = 0.12).
    A longer 0.15 s pose gap with the same speed also passes: the parallax allowance grows with
    dt, so the gate predicts THROUGH short dropouts instead of rejecting the re-acquisition."""
    for dt_s in (0.045, 0.15):
        r, v = 3.0, 2.4
        sweep = v * dt_s / r
        t0, t1 = 0, int(dt_s * _NS)
        s = _gate_seeker(use_imu_bearing_gate=True)
        s._append_att_hist(t0, (0.0, 0.0, 0.0))
        prev = _pose_from_world_dir([np.cos(0.1), np.sin(0.1), 0.05], r, (0.0, 0.0, 0.0), t0, fid=0)
        assert _detect(s, [prev], 0, t0) is prev
        s._append_att_hist(t1, (0.0, 0.0, 0.0))
        cand = _pose_from_world_dir([np.cos(0.1 + sweep), np.sin(0.1 + sweep), 0.05], r,
                                    (0.0, 0.0, 0.0), t1, fid=1)
        assert _detect(s, [cand], 1, t1) is cand, \
            f"dt={dt_s}: legitimate parallax sweep falsely rejected"


def test_rotation_prediction_sign_pinned():
    """THE PREDICTION SIGN PIN: the drone yaws +0.15 rad between captures (measured by the
    AHRS). A STATIC gate's camera bearing must move -0.15 (opposite the rotation): that
    candidate is ACCEPTED even though its raw camera-frame step (0.15) is far above the noise
    floor. A candidate whose camera bearing moved WITH the rotation (+0.15 in world = the
    IMU-inconsistent direction) is REJECTED. A flipped prediction sign would swap these two
    outcomes — this test pins it."""
    r, dyaw, dt_s = 10.0, 0.15, 0.05
    t0, t1 = 0, int(dt_s * _NS)
    d_world = np.array([np.cos(0.10), np.sin(0.10), 0.0])       # gate fixed at world az 0.10

    s = _gate_seeker(use_imu_bearing_gate=True)
    s._append_att_hist(t0, (0.0, 0.0, 0.0))
    prev = _pose_from_world_dir(d_world, r, (0.0, 0.0, 0.0), t0, fid=0)
    assert _detect(s, [prev], 0, t0) is prev
    s._append_att_hist(t1, (0.0, 0.0, dyaw))                    # the measured rotation
    # consistent: SAME world direction, captured at the rotated attitude (camera az moved -0.15).
    good = _pose_from_world_dir(d_world, r, (0.0, 0.0, dyaw), t1, fid=1)
    # inconsistent: camera bearing UNCHANGED despite the rotation (world az moved +0.15).
    bad = GatePose(frame_id=1, sim_time_ns=t1, R_cam_gate=np.eye(3),
                   t_cam_gate=np.asarray(prev.t_cam_gate, float).copy(), reproj_error_px=0.0)
    assert _detect(s, [good], 1, t1) is good, "rotation-consistent bearing falsely rejected"
    # (fresh seeker pair so the accept above doesn't move the reference for the reject check)
    s2 = _gate_seeker(use_imu_bearing_gate=True)
    s2._append_att_hist(t0, (0.0, 0.0, 0.0))
    assert _detect(s2, [prev], 0, t0) is prev
    s2._append_att_hist(t1, (0.0, 0.0, dyaw))
    assert _detect(s2, [bad], 2, t1) is None, \
        "IMU-inconsistent bearing (moved WITH the rotation) must be rejected"
    assert s2._last_bearing_dev_rad == pytest.approx(dyaw, abs=0.02)


def test_bearing_gate_fallback_and_state_hygiene():
    """No capture-time attitude available (empty ring buffer / gap-guard miss) => fall back to
    the LEGACY fixed-threshold check (never fail closed on a dead buffer). Flag OFF => no
    bearing-gate state is ever written. Track drop / pass / reset clear the reference."""
    # fallback: flag ON but no attitude history -> the 0.30 rad hop passes the legacy 0.35 gate.
    s = _gate_seeker(use_imu_bearing_gate=True)
    t0, t1 = 0, int(0.045 * _NS)
    prev = _sweep_pose(0.0, t0, 0)
    assert _detect(s, [prev], 0, t0) is prev
    assert s._bg_prev_dir_world is None                # no attitude -> no reference stored
    hop = _sweep_pose(-0.30, t1, 1)
    assert _detect(s, [hop], 1, t1) is hop             # legacy accepts (0.30 <= 0.35)
    # flag OFF: zero side effects (VQ1 byte-identity).
    off = _gate_seeker()
    assert _detect(off, [prev], 0, t0) is prev
    assert off._bg_prev_dir_world is None and off._bg_prev_pose_ns is None
    # hygiene: an ON seeker with a live reference drops it with the track.
    s2 = _gate_seeker(use_imu_bearing_gate=True)
    s2._append_att_hist(t0, (0.0, 0.0, 0.0))
    assert _detect(s2, [prev], 0, t0) is prev
    assert s2._bg_prev_dir_world is not None
    s2._begin_pass(0)
    assert s2._bg_prev_dir_world is None
    s2._append_att_hist(t1, (0.0, 0.0, 0.0))
    assert _detect(s2, [prev], 1, t1) is not None      # re-seeds on the next acquisition
    assert s2._bg_prev_dir_world is not None
    s2.reset()
    assert s2._bg_prev_dir_world is None and s2._bg_prev_pose_ns is None
    assert s2._last_bearing_dev_rad is None and s2._last_bearing_allow_rad is None


# ===========================================================================
# 6. FIX 4 — startup swell: the tightened settle hold band
# ===========================================================================
def test_settle_hold_band_bounds_the_cold_collective():
    """With the vq2 override band [0.90, 1.12] x hover, the settle-hold collective cannot rail:
    a cold garbage z/vz estimate that would demand a large climb (the +1.2 m/s injector) is
    clamped to at most 1.12 x hover; the floor is 0.90 x hover (near-hover — A7's free-fall
    concern stays covered). Default band [0.6, 1.4] is unchanged (legacy)."""
    hover = 0.2656
    for lo, hi in ((0.90, 1.12), (0.6, 1.4)):
        cfg = GateSeekerConfig(settle_s=0.75, hold_thrust_lo_frac=lo, hold_thrust_hi_frac=hi)
        s = GateSeeker(config=cfg)
        for vz in (-5.0, 0.0, +5.0):                    # garbage cold vertical estimates
            cmd = s.command_visual(_nav(0, vz=vz), None, 0)
            assert s._last_regime == "settle"
            assert lo * hover - 1e-9 <= cmd.thrust <= hi * hover + 1e-9


# ===========================================================================
# 7. PROFILE WIRING — vq2_case_c carries A31; VQ1 untouched
# ===========================================================================
def test_profile_wiring_a31():
    ov = get_profile("vq2_case_c").seeker_overrides
    # FIX 1/2: immediate turn + spend momentum + re-ramp + slow approach.
    # A33 H-1(b) SUPERSEDES the A31 wire window: the just-passed gate is now excluded by GEOMETRY
    # (pass_exclude_prev_gate), so the eligibility clock collapses (wire 0.25 -> 0.0, vis 1.2 -> 0.3).
    assert ov["pass_wire_coast_s"] == 0.0
    assert ov["pass_coast_s"] == 0.3
    assert ov["pass_coast_accel_mps2"] == 0.0
    assert ov["reramp_forward_after_pass"] is True
    assert ov["forward_accel_mps2"] == 0.65   # A36 Item 1: slowed 0.8 -> 0.65 ("slow is smooth")
    # FIX 3: orbit-breaker + yaw clamp. (A36 coordinated-turn DEMOTED the guard 1.75 -> 3.0 so it no
    # longer pre-empts the pass-turn -- a rare failsafe.)
    assert ov["orbit_guard_rad"] == 3.0
    assert ov["orbit_break_s"] == 1.0
    assert ov["orbit_yaw_clamp_rad"] == 2.4
    # THE UPGRADE: IMU-consistency bearing gate REPLACES the tightened fixed threshold —
    # track_max_bearing_jump_rad is deliberately NOT overridden (it stays the 0.35 legacy
    # fallback for attitude-unavailable frames).
    assert ov["use_imu_bearing_gate"] is True
    assert "track_max_bearing_jump_rad" not in ov
    # FIX 4: lateral slew + settle band.
    assert ov["image_lat_slew_mps3"] == 6.0
    # A35 Fix-3 (2026-07-03): settle hold floor 0.90 -> 1.00 (never sink off the line -- the startup
    # dip that tapped ground on run 20260704_024434). The 1.12 anti-swell cap stays.
    assert ov["hold_thrust_lo_frac"] == 1.00
    assert ov["hold_thrust_hi_frac"] == 1.12
    # A30/A29/A28 kept. (A36 coordinated-turn: yaw slew 1.5->0.9, total_accel_cap 2.0->3.0.)
    assert ov["use_image_servo_lateral"] is True
    assert ov["pursuit_yaw_slew_rps"] == 0.9
    assert ov["total_accel_cap_mps2"] == 3.0
    prof = vq2_case_c()
    assert prof.nav_config.reconcile_vision_clock_continuous is True
    assert prof.vertical_estimator is True
    # the effective config constructs cleanly.
    eff = GateSeekerConfig(**ov)
    assert eff.pass_wire_coast_s == 0.0 and eff.use_imu_bearing_gate is True   # A33 H-1(b)
    # A33 flags wired: geometric old-gate exclusion, turn-through-occlusion, cos^4.
    assert eff.pass_exclude_prev_gate is True
    assert eff.pass_turn_through is True
    assert eff.fwd_scale_pow == 4.0
    # A34 SUPERSEDES H-3: soft_range_hard_reject DROPPED (back to its False default); the ABSOLUTE
    # range cap replaces it (a >35 m reading is impossible-on-course garbage, hard-discarded).
    assert eff.soft_range_hard_reject is False
    assert eff.track_abs_range_cap_m == 35.0
    # VQ1 / case-A: no overrides at all (byte-identical).
    assert vq1_case_a().seeker_overrides is None
    assert vq1_case_a().controller_overrides is None
