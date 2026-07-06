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
# schedule shape / durations (M3 freeze-window SPLIT: m3a / m3b)
# ---------------------------------------------------------------------------
def test_m3a_schedule_sequence():
    """m3a: on-pad bias cal (M4) opens, then panos + one leg, inside the clean-IMU window."""
    sched = mm.build_schedule(mission="m3a")
    assert [s.name for s in sched] == [
        "ground_cal", "takeoff", "pano_1", "leg_1", "settle_1", "pano_2", "settle",
    ]
    assert [s.kind for s in sched] == [
        mm.SEG_GROUND_CAL, mm.SEG_TAKEOFF, mm.SEG_PANO, mm.SEG_LEG, mm.SEG_SETTLE,
        mm.SEG_PANO, mm.SEG_SETTLE,
    ]


def test_m3b_schedule_sequence():
    """m3b: cal, opening +120 deg turn -> its leg departs ~120 deg off m3a's; then ONE
    extended leg + ONE far-out pano (leg_2 AND the co-located pano_2 dropped per the
    freeze-split fallback -- flagged)."""
    sched = mm.build_schedule(mission="m3b")
    assert [s.name for s in sched] == [
        "ground_cal", "takeoff", "turn_120", "leg_1", "settle_1", "pano_1", "settle",
    ]
    assert [s.kind for s in sched] == [
        mm.SEG_GROUND_CAL, mm.SEG_TAKEOFF, mm.SEG_PANO, mm.SEG_LEG, mm.SEG_SETTLE,
        mm.SEG_PANO, mm.SEG_SETTLE,
    ]


def test_ground_cal_disabled_removes_the_segment():
    """--ground-cal-s 0 disables the M4 cal phase entirely (bias stays 0 == pre-M4 behaviour);
    the schedule simply starts at takeoff."""
    cfg = mm.MissionConfig(ground_cal_s=0.0)
    for mission in mm.MISSIONS:
        sched = mm.build_schedule(cfg, mission=mission)
        assert sched[0].kind == mm.SEG_TAKEOFF
        assert all(s.kind != mm.SEG_GROUND_CAL for s in sched)


def test_unknown_mission_rejected():
    import pytest
    with pytest.raises(ValueError):
        mm.build_schedule(mission="m3c")


def test_totals_fit_the_imu_freeze_window():
    """HARD freeze-window bounds: the sim IMU freezes at t~=60-90 s of race time (~62 s clean
    segments corroborated), so each mission MUST complete inside it. M4 totals INCLUDE the
    1.5 s ground-cal: m3a pins 56.3 s, m3b 52.4 s; both hard-capped at 58 s."""
    t_a = mm.schedule_total_s(mm.build_schedule(mission="m3a"))
    t_b = mm.schedule_total_s(mm.build_schedule(mission="m3b"))
    assert t_a == pytest_approx(56.3)
    assert t_b == pytest_approx(52.4)
    assert t_a <= 58.0 and t_b <= 58.0   # the freeze-window hard cap, both flights


def test_segment_durations_are_sum_of_subphase_params():
    cfg = mm.MissionConfig()
    for mission, coast in (("m3a", cfg.leg_coast_s), ("m3b", cfg.m3b_leg_coast_s)):
        by_name = {s.name: s for s in mm.build_schedule(cfg, mission=mission)}

        takeoff = by_name["takeoff"]
        assert takeoff.duration_s == cfg.takeoff_climb_s + cfg.takeoff_trim_s
        assert takeoff.climb_s == cfg.takeoff_climb_s

        leg = by_name["leg_1"]
        assert leg.duration_s == cfg.leg_accel_s + coast + cfg.leg_brake_s + cfg.leg_retrim_s
        assert leg.coast_s == pytest_approx(coast)

        assert by_name["settle_1"].duration_s == cfg.post_leg_settle_s == 4.0
        assert by_name["settle"].duration_s == cfg.settle_s == 5.0
    # m3b reinvests the dropped-leg_2 budget into a LONGER baseline out of the shared spawn
    assert cfg.m3b_leg_coast_s > cfg.leg_coast_s


def test_pano_duration_matches_revs_over_rate():
    cfg = mm.MissionConfig()
    rate = math.radians(cfg.pano_yaw_rate_dps)          # 25 dps -> a 360 takes 14.4 s
    a = {s.name: s for s in mm.build_schedule(cfg, mission="m3a")}
    b = {s.name: s for s in mm.build_schedule(cfg, mission="m3b")}

    for pano in (a["pano_1"], a["pano_2"], b["pano_1"]):
        assert pano.duration_s == pytest_approx(cfg.pano_revs * 2 * math.pi / rate)
        assert pano.duration_s == pytest_approx(14.4)

    # the m3b opening turn = turn_deg at the SPEC-PINNED 20 dps turn rate (6.0 s)
    turn = b["turn_120"]
    assert turn.duration_s == pytest_approx(math.radians(cfg.turn_deg)
                                            / math.radians(cfg.turn_rate_dps))
    assert turn.duration_s == pytest_approx(6.0)


def test_describe_schedule_lists_every_segment():
    for mission in mm.MISSIONS:
        sched = mm.build_schedule(mission=mission)
        text = mm.describe_schedule(sched, mission=mission)
        for s in sched:
            assert s.name in text
        assert mission.upper() in text                  # title carries the mission
        # the dry-run header carries the flown control constants (auditable)
        assert "body_rate_sign=(1,1,1)" in text
        assert "cmd_rate_scale=0.4" in text
    # the turn prints as an explicit re-heading (m3b only), not a >1-rev pano
    text_b = mm.describe_schedule(mm.build_schedule(mission="m3b"), mission="m3b")
    assert "re-heading turn +120 deg" in text_b


# ---------------------------------------------------------------------------
# M3 yaw plan: clean 360s + the explicit +120 deg re-heading opening m3b
# ---------------------------------------------------------------------------
def test_panos_sweep_exactly_360_and_turn_sweeps_120():
    """M3 operator fix: every pano sweeps EXACTLY one revolution (the M2 1.33-rev middle pano
    read as a defect and muddied the footage); the +120 deg cross-flight offset lives in
    m3b's opening turn_120 segment, so m3b's leg departs ~120 deg off m3a's."""
    a = {s.name: s for s in mm.build_schedule(mission="m3a")}
    b = {s.name: s for s in mm.build_schedule(mission="m3b")}
    for pano in (a["pano_1"], a["pano_2"], b["pano_1"]):
        assert pano.yaw_rate_rps * pano.duration_s == pytest_approx(2 * math.pi)
    turn = b["turn_120"]
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


def test_leg_pitch_schedule_is_brakeless_accel_coast_retrim():
    """M4: the default leg has NO brake window (sim drag alone over-stops -- the M3 flights
    still drifted backward with the 1.0 s @ 4 deg under-brake): accel, coast, then straight
    to level re-trim."""
    seg = _seg("leg_1")
    mag = seg.pitch_mag_rad
    assert mag == pytest_approx(math.radians(5.0))
    assert seg.brake_s == 0.0                       # M4: brake removed by default

    # accel window: nose-DOWN (negative NED pitch) -> forward, at the ACCEL tilt (5 deg)
    sp_a = mm.segment_setpoint(seg, t_in_seg=seg.accel_s * 0.5, dt=0.01)
    assert sp_a.pitch_des_rad == pytest_approx(-mag)

    # coast window: level
    sp_c = mm.segment_setpoint(seg, t_in_seg=seg.accel_s + seg.coast_s * 0.5, dt=0.01)
    assert sp_c.pitch_des_rad == 0.0

    # immediately after the coast: LEVEL re-trim -- never a nose-up pulse anywhere post-coast
    for frac in (0.01, 0.5, 0.99):
        t = seg.accel_s + seg.coast_s + frac * (seg.duration_s - seg.accel_s - seg.coast_s)
        sp = mm.segment_setpoint(seg, t_in_seg=t, dt=0.01)
        assert sp.pitch_des_rad == 0.0

    # a LEG never commands yaw or roll
    for t in (0.1, seg.accel_s + 1.0, seg.duration_s - 0.1):
        sp = mm.segment_setpoint(seg, t_in_seg=t, dt=0.01)
        assert sp.roll_des_rad == 0.0 and sp.dyaw_rad == 0.0


def test_leg_brake_impulse_under_brake_when_reenabled():
    """M4 default = NO brake at all (impulse 0). If a pilot re-enables it via --brake-s, the
    M3 under-brake principle still holds: brake impulse strictly LESS than the accel impulse
    (velocity is unobservable; a backward residual poisons the pano)."""
    # default: brake fully removed on both flights' legs
    for mission in mm.MISSIONS:
        seg = _seg("leg_1", mission)
        assert seg.brake_pitch_rad * seg.brake_s == 0.0
    # re-enabled via the knobs (the old M3 values): still a deliberate under-brake
    cfg = mm.MissionConfig(leg_brake_s=1.0, leg_brake_pitch_deg=4.0)
    seg = {s.name: s for s in mm.build_schedule(cfg, mission="m3a")}["leg_1"]
    accel_impulse = seg.pitch_mag_rad * seg.accel_s        # 5 deg x 2.0 s = 10 deg-s
    brake_impulse = seg.brake_pitch_rad * seg.brake_s      # 4 deg x 1.0 s =  4 deg-s
    assert 0.0 < brake_impulse < accel_impulse
    assert brake_impulse / accel_impulse == pytest_approx(0.4)


def test_settle_is_level_vz_damped_hover():
    seg = _seg("settle")
    sp = mm.segment_setpoint(seg, t_in_seg=1.0, dt=0.01)
    assert sp.thrust_mode == "damp"
    assert sp.roll_des_rad == 0.0 and sp.pitch_des_rad == 0.0 and sp.dyaw_rad == 0.0


def test_integrated_yaw_advance_equals_full_sweep():
    """Summing the per-tick dyaw across a whole yaw segment reproduces its total sweep -- the
    timeline the flight loop integrates onto the yaw target is consistent with the schedule.
    Checked on a clean 360 pano (m3a) and the explicit +120 deg turn (m3b)."""
    for name, mission in (("pano_2", "m3a"), ("turn_120", "m3b")):
        seg = _seg(name, mission)
        dt = 1.0 / 100.0
        n = int(round(seg.duration_s / dt))
        total = sum(mm.segment_setpoint(seg, t_in_seg=k * dt, dt=dt).dyaw_rad for k in range(n))
        assert total == pytest_approx(seg.yaw_rate_rps * seg.duration_s, rel=1e-3)


def test_dry_run_matches_segment_durations():
    """The dry-run text's stated total equals the summed segment durations (no drift between the
    printed schedule and the flown one) -- both missions."""
    for mission in mm.MISSIONS:
        sched = mm.build_schedule(mission=mission)
        text = mm.describe_schedule(sched, mission=mission)
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
    assert seg.climb_gate_vz_mps == pytest_approx(1.0)   # M4 taller takeoff: 0.8 -> 1.0
    # well inside the time cap, climb rate reaches the gate -> climb OVER
    assert mm.climb_phase_over(seg, t_in_seg=0.6, vz_up_mps=1.05) is True
    # same time, climb rate below the gate -> still climbing
    assert mm.climb_phase_over(seg, t_in_seg=0.6, vz_up_mps=0.9) is False


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


def test_vertical_correction_clamp_pinned_both_directions():
    """M4: the TOTAL vertical correction (damper + z-trim) is clamped to +/-VERT_CORR_CLIP
    (0.020) around hover -- an estimator lie is worth at most ~+/-0.75 m/s^2, so the M3
    phantom-climb continuous 10%-below-hover sink is physically impossible. Pinned in BOTH
    directions with wildly lying inputs."""
    kd, kp = 0.06, 0.004
    # huge phantom CLIMB + huge phantom over-altitude -> at most -0.020 below hover
    lo = mm.thrust_command(+10.0, mm.Z_TARGET_M + 100.0, kd, kp)
    assert lo == pytest_approx(mm.HOVER_THRUST - mm.VERT_CORR_CLIP)
    # huge phantom DESCENT + huge phantom under-altitude -> at most +0.020 above hover
    hi = mm.thrust_command(-10.0, mm.Z_TARGET_M - 100.0, kd, kp)
    assert hi == pytest_approx(mm.HOVER_THRUST + mm.VERT_CORR_CLIP)
    # the M3 failure case itself: 0.28 m/s phantom climb + railed trim used to command
    # 0.239 (10% below hover, continuous); now bounded to hover - 0.020 = 0.2456
    m3_case = mm.thrust_command(+0.28, +15.0, kd, kp)
    assert m3_case >= mm.HOVER_THRUST - mm.VERT_CORR_CLIP - 1e-12
    # small, honest corrections pass through UNclamped
    small = mm.thrust_command(+0.1, mm.Z_TARGET_M, kd, kp)
    assert small == pytest_approx(mm.HOVER_THRUST - 0.06 * 0.1)


# ---------------------------------------------------------------------------
# M4 ground-cal: pad-rest bias measurement + subtraction
# ---------------------------------------------------------------------------
def test_ground_cal_subtraction_kills_the_m3_phantom_climb():
    """THE M4 root-cause regression: a constant a_up bias of ~+0.006 m/s^2 locked vz_leak at
    bias*tau ~= +0.28 m/s (the measured M3 phantom climb). With the pad-measured bias
    subtracted, the steady-state vz_leak under the SAME bias stays < 0.05 m/s."""
    bias = 0.0062
    # pad cal: average the biased rest samples (as the GROUND_CAL segment does)
    cal = mm.GroundCal()
    for _ in range(150):                    # ~1.5 s of ~100 Hz ticks on the pad
        cal.add(bias)
    pv = mm.PseudoVertical(tau_s=45.0, a_up_bias=cal.bias)
    dt = 0.01
    for _ in range(10_000):                 # 100 s of flight under the SAME constant bias
        pv.step(bias, dt)
    assert abs(pv.vz_leak) < 0.05           # phantom climb killed (was ~0.28 uncal)
    assert abs(pv.z_pseudo) < 2.0           # z_pseudo no longer inflates to +15 m
    # contrast: WITHOUT the cal the same bias heads for bias*tau ~= 0.28 (the M3 bug);
    # at t=100 s the analytic value is bias*tau*(1 - e^(-100/45)) ~= 0.249
    pv_uncal = mm.PseudoVertical(tau_s=45.0)
    for _ in range(10_000):
        pv_uncal.step(bias, dt)
    expected = bias * 45.0 * (1.0 - math.exp(-100.0 / 45.0))
    assert pv_uncal.vz_leak == pytest_approx(expected, rel=0.02)
    assert pv_uncal.vz_leak > 0.2       # the phantom climb is unmistakably present uncal


def test_ground_cal_averaging_ignores_glitch_samples():
    """The cal window applies the SAME glitch guards as the integrator: non-finite samples
    are ignored entirely, extreme samples are clamped to +/-30 m/s^2."""
    cal = mm.GroundCal()
    for _ in range(100):
        cal.add(0.0062)
    n_before = cal.n
    cal.add(float("nan"))                   # ignored: not counted at all
    cal.add(float("inf"))
    assert cal.n == n_before
    assert cal.bias == pytest_approx(0.0062)
    # a wild spike is clamped, not averaged at face value
    cal2 = mm.GroundCal()
    cal2.add(1000.0)
    assert cal2.bias == pytest_approx(30.0)
    # empty window (cal disabled/starved) -> bias 0.0, never a ZeroDivisionError
    assert mm.GroundCal().bias == 0.0


def test_ground_cal_setpoint_is_idle_zero_command():
    """The GROUND_CAL segment commands idle thrust with zero attitude/yaw intent -- the
    flight loop sends zero body rates and IDLE_THRUST for the whole window."""
    seg = _seg("ground_cal")
    for t in (0.0, 0.7, 1.4):
        sp = mm.segment_setpoint(seg, t_in_seg=t, dt=0.01)
        assert sp.thrust_mode == "idle"
        assert sp.roll_des_rad == 0.0 and sp.pitch_des_rad == 0.0 and sp.dyaw_rad == 0.0


def test_pseudo_vertical_reset_keeps_the_calibrated_bias():
    """A race-restart reset zeroes the states but KEEPS the pad-calibrated bias (the sensor
    bias survives a restart; a fresh flight re-runs the cal and overwrites it anyway)."""
    pv = mm.PseudoVertical(tau_s=45.0, a_up_bias=0.0062)
    pv.step(1.0, 0.01)
    pv.reset()
    assert pv.vz_leak == 0.0 and pv.z_pseudo == 0.0
    assert pv.a_up_bias == pytest_approx(0.0062)


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
def _seg(name: str, mission: str = "m3a"):
    return {s.name: s for s in mm.build_schedule(mission=mission)}[name]


def pytest_approx(value, rel=None, abs=None):
    import pytest
    return pytest.approx(value, rel=rel, abs=abs)
