"""Scheduler tests for the hardcoded VQ2 mapping mission (scripts/mapping_mission.py).

Network-free: exercises ONLY the pure scheduler + per-tick open-loop synthesiser (build_schedule,
segment_setpoint, describe_schedule). The flight loop (AHRS / vz-washout / MAVLink) needs a live
sim and is out of scope here -- these tests pin the command TIMELINE the flight loop replays, so a
scheduler regression is caught offline before any flight.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

# Import scripts/mapping_mission.py by path (scripts/ is not a package on sys.path). Register it
# in sys.modules BEFORE exec so @dataclass (under `from __future__ import annotations`) can resolve
# the module's namespace for its string-annotation type checks.
_SPEC = importlib.util.spec_from_file_location(
    "mapping_mission",
    Path(__file__).resolve().parent.parent / "scripts" / "mapping_mission.py",
)
mm = importlib.util.module_from_spec(_SPEC)
sys.modules["mapping_mission"] = mm
_SPEC.loader.exec_module(mm)


# ---------------------------------------------------------------------------
# schedule shape / durations (M3, operator-eyewitness revision)
# ---------------------------------------------------------------------------
def test_schedule_is_the_m3_sequence():
    """M3: settle after each leg ('stop, level, then spin'), clean 360 panos only, and the
    re-heading as an explicit named TURN segment."""
    sched = mm.build_schedule()
    names = [s.name for s in sched]
    kinds = [s.kind for s in sched]
    assert names == [
        "takeoff", "pano_1", "leg_1", "settle_1", "pano_2", "turn_120",
        "settle_2", "leg_2", "settle_3", "pano_3", "settle",
    ]
    assert kinds == [
        mm.SEG_TAKEOFF, mm.SEG_PANO, mm.SEG_LEG, mm.SEG_SETTLE, mm.SEG_PANO,
        mm.SEG_PANO, mm.SEG_SETTLE, mm.SEG_LEG, mm.SEG_SETTLE, mm.SEG_PANO, mm.SEG_SETTLE,
    ]


def test_total_duration_matches_design():
    """Pins the M3 total: 5 + 18 + 14.5 + 4 + 18 + 6 + 2 + 14.5 + 4 + 18 + 10 = 114.0 s.
    (The M3 brief predicted ~125-130 s; the specified knobs sum to 114.0 -- flagged to the
    coordinator, pinned here so any knob change shows up as an explicit diff.)"""
    total = mm.schedule_total_s(mm.build_schedule())
    assert total == pytest_approx(114.0)


def test_segment_durations_are_sum_of_subphase_params():
    cfg = mm.MissionConfig()
    sched = mm.build_schedule(cfg)
    by_name = {s.name: s for s in sched}

    takeoff = by_name["takeoff"]
    assert takeoff.duration_s == cfg.takeoff_climb_s + cfg.takeoff_trim_s
    assert takeoff.climb_s == cfg.takeoff_climb_s

    for leg_name in ("leg_1", "leg_2"):
        leg = by_name[leg_name]
        assert leg.duration_s == (
            cfg.leg_accel_s + cfg.leg_coast_s + cfg.leg_brake_s + cfg.leg_retrim_s
        )
        assert leg.coast_s == pytest_approx(10.0)     # M3: more forward flight per leg

    # M3 settles: 4 s after each leg ("stop, level, then spin"), 2 s post-turn, 10 s final
    assert by_name["settle_1"].duration_s == cfg.post_leg_settle_s == 4.0
    assert by_name["settle_3"].duration_s == cfg.post_leg_settle_s == 4.0
    assert by_name["settle_2"].duration_s == cfg.post_turn_settle_s == 2.0
    assert by_name["settle"].duration_s == cfg.settle_s == 10.0


def test_pano_duration_matches_revs_over_rate():
    cfg = mm.MissionConfig()
    sched = mm.build_schedule(cfg)
    by_name = {s.name: s for s in sched}
    rate = math.radians(cfg.pano_yaw_rate_dps)

    # every named panorama = exactly pano_revs (1.0) turns
    for name in ("pano_1", "pano_2", "pano_3"):
        assert by_name[name].duration_s == pytest_approx(cfg.pano_revs * 2 * math.pi / rate)

    # the explicit turn = turn_deg at the same rate (120 deg / 20 dps = 6.0 s)
    turn = by_name["turn_120"]
    assert turn.duration_s == pytest_approx(math.radians(cfg.turn_deg) / rate)
    assert turn.duration_s == pytest_approx(6.0)


def test_describe_schedule_lists_every_segment():
    sched = mm.build_schedule()
    text = mm.describe_schedule(sched)
    for s in sched:
        assert s.name in text
    # the dry-run header carries the flown control constants (auditable)
    assert "body_rate_sign=(1,1,1)" in text
    assert "cmd_rate_scale=0.4" in text
    # the turn prints as an explicit re-heading, not a >1-rev pano (the M2 defect-read)
    assert "re-heading turn +120 deg" in text


# ---------------------------------------------------------------------------
# M3 yaw plan: clean 360s + an explicit +120 deg re-heading turn
# ---------------------------------------------------------------------------
def test_panos_sweep_exactly_360_and_turn_sweeps_120():
    """M3 operator fix: every pano sweeps EXACTLY one revolution (the M2 1.33-rev middle pano
    read as a defect and muddied the footage); the +120 deg between-leg offset lives in the
    separate turn_120 segment, so leg_2 still departs 120 deg off leg_1."""
    sched = mm.build_schedule()
    by_name = {s.name: s for s in sched}
    for name in ("pano_1", "pano_2", "pano_3"):
        p = by_name[name]
        assert p.yaw_rate_rps * p.duration_s == pytest_approx(2 * math.pi)
    turn = by_name["turn_120"]
    assert turn.yaw_rate_rps * turn.duration_s == pytest_approx(math.radians(120.0))


# ---------------------------------------------------------------------------
# per-tick open-loop synthesiser (the command timeline)
# ---------------------------------------------------------------------------
def test_takeoff_climb_then_trim_thrust_modes():
    seg = _seg("takeoff")
    # during the climb window: open-loop climb ramp
    sp = mm.segment_setpoint(seg, t_in_seg=0.5, dt=0.01)
    assert sp.thrust_mode == "climb"
    assert sp.climb_thrust_delta > 0.0
    assert sp.roll_des_rad == 0.0 and sp.pitch_des_rad == 0.0 and sp.dyaw_rad == 0.0
    # after the climb window: vz-damped hover trim, level
    sp2 = mm.segment_setpoint(seg, t_in_seg=seg.climb_s + 0.5, dt=0.01)
    assert sp2.thrust_mode == "damp"
    assert sp2.pitch_des_rad == 0.0


def test_pano_commands_constant_yaw_rate_and_stays_level():
    seg = _seg("pano_1")
    dt = 0.01
    sp = mm.segment_setpoint(seg, t_in_seg=3.0, dt=dt)
    assert sp.thrust_mode == "damp"
    assert sp.roll_des_rad == 0.0 and sp.pitch_des_rad == 0.0
    # yaw advance per tick == rate * dt, constant across the segment
    assert sp.dyaw_rad == pytest_approx(seg.yaw_rate_rps * dt)
    sp_late = mm.segment_setpoint(seg, t_in_seg=seg.duration_s - 0.1, dt=dt)
    assert sp_late.dyaw_rad == pytest_approx(sp.dyaw_rad)


def test_leg_pitch_schedule_accel_coast_brake_retrim():
    seg = _seg("leg_1")
    mag = seg.pitch_mag_rad
    brake = seg.brake_pitch_rad
    assert mag > 0.0 and brake > 0.0

    # accel window: nose-DOWN (negative NED pitch) -> forward, at the ACCEL tilt (5 deg)
    sp_a = mm.segment_setpoint(seg, t_in_seg=seg.accel_s * 0.5, dt=0.01)
    assert sp_a.pitch_des_rad == pytest_approx(-mag)
    assert mag == pytest_approx(math.radians(5.0))

    # coast window: level
    sp_c = mm.segment_setpoint(seg, t_in_seg=seg.accel_s + seg.coast_s * 0.5, dt=0.01)
    assert sp_c.pitch_des_rad == 0.0

    # brake window: nose-UP (positive) at the SMALLER brake tilt (4 deg, M3 under-brake)
    t_brake = seg.accel_s + seg.coast_s + seg.brake_s * 0.5
    sp_b = mm.segment_setpoint(seg, t_in_seg=t_brake, dt=0.01)
    assert sp_b.pitch_des_rad == pytest_approx(+brake)
    assert brake == pytest_approx(math.radians(4.0))

    # re-trim window: level again
    t_trim = seg.accel_s + seg.coast_s + seg.brake_s + 0.1
    sp_t = mm.segment_setpoint(seg, t_in_seg=t_trim, dt=0.01)
    assert sp_t.pitch_des_rad == 0.0

    # a LEG never commands yaw or roll
    for t in (0.1, seg.accel_s + 1.0, t_brake, t_trim):
        sp = mm.segment_setpoint(seg, t_in_seg=t, dt=0.01)
        assert sp.roll_des_rad == 0.0 and sp.dyaw_rad == 0.0


def test_leg_brake_impulse_is_a_deliberate_under_brake():
    """M3 operator eyewitness: the M2 full-impulse brake reversed the drag-bled residual and
    the drone drifted BACKWARD through the following pano. Velocity is unobservable, so the
    brake must UNDER-shoot: brake impulse (tilt x time) strictly LESS than the accel impulse
    -- drag covers the rest; a forward residual is harmless parallax, a backward one poisons
    the pano."""
    for leg_name in ("leg_1", "leg_2"):
        seg = _seg(leg_name)
        accel_impulse = seg.pitch_mag_rad * seg.accel_s        # 5 deg x 2.0 s = 10 deg-s
        brake_impulse = seg.brake_pitch_rad * seg.brake_s      # 4 deg x 1.0 s =  4 deg-s
        assert brake_impulse > 0.0
        assert brake_impulse < accel_impulse                   # NEVER fully cancel open-loop
        # pin the designed margin (~40% of the accel impulse) so a knob change is explicit
        assert brake_impulse / accel_impulse == pytest_approx(0.4)


def test_settle_is_level_vz_damped_hover():
    seg = _seg("settle")
    sp = mm.segment_setpoint(seg, t_in_seg=1.0, dt=0.01)
    assert sp.thrust_mode == "damp"
    assert sp.roll_des_rad == 0.0 and sp.pitch_des_rad == 0.0 and sp.dyaw_rad == 0.0


def test_integrated_yaw_advance_equals_full_sweep():
    """Summing the per-tick dyaw across a whole yaw segment reproduces its total sweep -- the
    timeline the flight loop integrates onto the yaw target is consistent with the schedule.
    Checked on both a clean 360 pano and the explicit +120 deg turn."""
    for name in ("pano_2", "turn_120"):
        seg = _seg(name)
        dt = 1.0 / 100.0
        n = int(round(seg.duration_s / dt))
        total = sum(mm.segment_setpoint(seg, t_in_seg=k * dt, dt=dt).dyaw_rad for k in range(n))
        assert total == pytest_approx(seg.yaw_rate_rps * seg.duration_s, rel=1e-3)


def test_dry_run_matches_segment_durations():
    """The dry-run text's stated total equals the summed segment durations (no drift between the
    printed schedule and the flown one)."""
    sched = mm.build_schedule()
    text = mm.describe_schedule(sched)
    total = mm.schedule_total_s(sched)
    assert f"total {total:.1f} s" in text


# ---------------------------------------------------------------------------
# M2 vertical channel (post-M1-ceiling-crash): PseudoVertical + gate + thrust signs
# ---------------------------------------------------------------------------
def test_leaky_vz_keeps_steady_climb_visible_for_30s():
    """M1 root cause regression: the racing washout (tau=2 s) forgets a steady climb within
    seconds. The mission channel (tau=45 s) must keep a 1 m/s climb >0.5 m/s visible after
    30 s of constant-velocity coast (a_up = 0), so the damper keeps opposing it."""
    pv = mm.PseudoVertical(tau_s=45.0)
    dt = 0.01
    for _ in range(10):                 # 10 m/s^2 x 0.1 s accel pulse -> ~1.0 m/s climb
        pv.step(10.0, dt)
    assert 0.95 <= pv.vz_leak <= 1.01
    for _ in range(3000):               # 30 s steady climb: zero kinematic accel
        pv.step(0.0, dt)
    assert pv.vz_leak > 0.5             # exp(-30/45) = 0.513 of the 1 m/s still visible
    assert pv.vz_leak < 0.6             # ...and the leak IS working (not a pure integrator)


def test_accel_bias_error_is_bounded():
    """A sustained accel bias (the measured ground-stationary bound ~0.004 m/s^2) must NOT run
    away: vz_leak converges to bias*tau = 0.18 m/s (< 0.4) and z_pseudo drifts bounded-linearly
    (< 20 m over 100 s) -- the leak is what buys this over a pure integrator."""
    pv = mm.PseudoVertical(tau_s=45.0)
    dt = 0.01
    for _ in range(10_000):             # 100 s of constant +0.004 m/s^2 bias
        pv.step(0.004, dt)
    assert abs(pv.vz_leak) < 0.4        # spec bound (analytic ss: 0.004*45 = 0.18)
    assert abs(pv.vz_leak) > 0.1        # sanity: it converged near the analytic value
    assert abs(pv.z_pseudo) < 20.0      # spec bound (analytic: ~10.8 m at t=100 s)


def test_takeoff_gate_fires_on_vz_threshold():
    seg = _seg("takeoff")
    assert seg.climb_gate_vz_mps == pytest_approx(0.8)
    # well inside the time cap, climb rate reaches the gate -> climb OVER
    assert mm.climb_phase_over(seg, t_in_seg=0.6, vz_up_mps=0.85) is True
    # same time, climb rate below the gate -> still climbing
    assert mm.climb_phase_over(seg, t_in_seg=0.6, vz_up_mps=0.5) is False


def test_takeoff_gate_fires_on_time_cap():
    seg = _seg("takeoff")
    assert seg.climb_s == pytest_approx(2.0)
    # cap reached with NO climb registered -> still over (whichever comes FIRST)
    assert mm.climb_phase_over(seg, t_in_seg=2.0, vz_up_mps=0.0) is True
    assert mm.climb_phase_over(seg, t_in_seg=1.9, vz_up_mps=0.0) is False


def test_takeoff_setpoint_honors_the_latched_gate():
    """Once the caller's latch says the climb is over, the setpoint is trim (damp) even before
    the time cap -- and the segment DURATION is unchanged (schedule-stable early gate)."""
    seg = _seg("takeoff")
    sp_climbing = mm.segment_setpoint(seg, t_in_seg=0.5, dt=0.01, climb_done=False)
    assert sp_climbing.thrust_mode == "climb"
    sp_gated = mm.segment_setpoint(seg, t_in_seg=0.5, dt=0.01, climb_done=True)
    assert sp_gated.thrust_mode == "damp"
    cfg = mm.MissionConfig()
    assert seg.duration_s == cfg.takeoff_climb_s + cfg.takeoff_trim_s


def test_thrust_sign_climbing_reduces_thrust():
    """THE M1 sign bug regression (anti-damping flew into the ceiling): with the UP-positive
    convention, CLIMBING (vz_leak > 0) must REDUCE thrust below hover; DESCENDING must raise it."""
    at_target = mm.Z_TARGET_M
    hover = mm.thrust_command(0.0, at_target, kd_vz=0.06, kp_zp=0.004)
    climbing = mm.thrust_command(+1.0, at_target, kd_vz=0.06, kp_zp=0.004)
    descending = mm.thrust_command(-1.0, at_target, kd_vz=0.06, kp_zp=0.004)
    assert hover == pytest_approx(mm.HOVER_THRUST)
    assert climbing < hover < descending


def test_thrust_sign_above_target_reduces_thrust_and_trim_is_clipped():
    """Pseudo-altitude trim: ABOVE target must REDUCE thrust, BELOW must raise it, and the trim
    authority is clipped to +/-0.010 regardless of the error magnitude."""
    kd, kp = 0.06, 0.004
    at = mm.thrust_command(0.0, mm.Z_TARGET_M, kd, kp)
    above = mm.thrust_command(0.0, mm.Z_TARGET_M + 1.0, kd, kp)
    below = mm.thrust_command(0.0, mm.Z_TARGET_M - 1.0, kd, kp)
    assert above < at < below
    assert at - above == pytest_approx(kp * 1.0)
    # clip: a huge error contributes at most +/-0.010
    way_above = mm.thrust_command(0.0, mm.Z_TARGET_M + 100.0, kd, kp)
    way_below = mm.thrust_command(0.0, mm.Z_TARGET_M - 100.0, kd, kp)
    assert way_above == pytest_approx(mm.HOVER_THRUST - mm.ZP_TRIM_CLIP)
    assert way_below == pytest_approx(mm.HOVER_THRUST + mm.ZP_TRIM_CLIP)
    # kp_zp = 0 disables the trim entirely
    assert mm.thrust_command(0.0, 50.0, kd, 0.0) == pytest_approx(mm.HOVER_THRUST)


def test_thrust_outer_clamp_never_rails():
    """The [0.18, 0.42] outer clamp bounds the composed command (the sim mixer couples
    thrust<->rates near the rails)."""
    assert mm.thrust_command(+10.0, 0.0, kd_vz=0.06, kp_zp=0.004) == pytest_approx(mm.THRUST_LO)
    assert mm.thrust_command(-10.0, 0.0, kd_vz=0.06, kp_zp=0.004) == pytest_approx(mm.THRUST_HI)


def test_pseudo_vertical_reset_and_glitch_guards():
    pv = mm.PseudoVertical(tau_s=45.0)
    pv.step(1.0, 0.01)
    assert pv.vz_leak > 0.0
    # glitch guards: non-positive dt, oversized dt (sim reset/stutter), non-finite a_up
    v0, z0 = pv.vz_leak, pv.z_pseudo
    pv.step(1.0, 0.0)
    pv.step(1.0, 1.0)
    pv.step(float("nan"), 0.01)
    assert pv.vz_leak == v0 and pv.z_pseudo == z0
    # reset (race restart / backward sim-clock jump) zeroes both states
    pv.reset()
    assert pv.vz_leak == 0.0 and pv.z_pseudo == 0.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _seg(name: str):
    return {s.name: s for s in mm.build_schedule()}[name]


def pytest_approx(value, rel=None, abs=None):
    import pytest
    return pytest.approx(value, rel=rel, abs=abs)
