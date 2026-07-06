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
# schedule shape / durations
# ---------------------------------------------------------------------------
def test_schedule_is_the_m1_sequence():
    kinds = [s.kind for s in mm.build_schedule()]
    assert kinds == [
        mm.SEG_TAKEOFF, mm.SEG_PANO, mm.SEG_LEG,
        mm.SEG_PANO, mm.SEG_LEG, mm.SEG_PANO, mm.SEG_SETTLE,
    ]


def test_total_duration_in_target_window():
    total = mm.schedule_total_s(mm.build_schedule())
    assert 90.0 <= total <= 120.0   # spec: ~100-120 s mission


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

    assert by_name["settle"].duration_s == cfg.settle_s


def test_pano_duration_matches_revs_over_rate():
    cfg = mm.MissionConfig()
    sched = mm.build_schedule(cfg)
    by_name = {s.name: s for s in sched}
    rate = math.radians(cfg.pano_yaw_rate_dps)

    # a plain panorama = exactly pano_revs turns
    pano1 = by_name["pano_1"]
    assert pano1.duration_s == pytest_approx(cfg.pano_revs * 2 * math.pi / rate)

    # the middle panorama carries the extra +120 deg fractional turn
    pano_mid = by_name["pano_2_turn"]
    expected_revs = cfg.pano_revs + cfg.mid_pano_extra_turn
    assert pano_mid.duration_s == pytest_approx(expected_revs * 2 * math.pi / rate)


def test_describe_schedule_lists_every_segment():
    sched = mm.build_schedule()
    text = mm.describe_schedule(sched)
    for s in sched:
        assert s.name in text
    # the dry-run header carries the flown control constants (auditable)
    assert "body_rate_sign=(1,1,1)" in text
    assert "cmd_rate_scale=0.4" in text


# ---------------------------------------------------------------------------
# the +120 deg heading offset between the two legs
# ---------------------------------------------------------------------------
def test_middle_pano_yaw_sweep_offsets_second_leg_by_120deg():
    """The PANO before leg_2 sweeps an INTEGER number of turns PLUS 120 deg, so the net heading
    change across it (mod 360) is +120 deg -> leg_2 departs 120 deg off leg_1."""
    cfg = mm.MissionConfig()
    sched = mm.build_schedule(cfg)
    by_name = {s.name: s for s in sched}
    mid = by_name["pano_2_turn"]
    net_sweep_rad = mid.yaw_rate_rps * mid.duration_s          # total yaw slewed across the pano
    net_mod = net_sweep_rad % (2 * math.pi)
    assert net_mod == pytest_approx(math.radians(120.0), abs=1e-6)

    # the plain panoramas net to ~0 heading change (a whole number of turns)
    for name in ("pano_1", "pano_3"):
        p = by_name[name]
        assert (p.yaw_rate_rps * p.duration_s) % (2 * math.pi) == pytest_approx(0.0, abs=1e-6)


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
    assert mag > 0.0

    # accel window: nose-DOWN (negative NED pitch) -> forward
    sp_a = mm.segment_setpoint(seg, t_in_seg=seg.accel_s * 0.5, dt=0.01)
    assert sp_a.pitch_des_rad == pytest_approx(-mag)

    # coast window: level
    sp_c = mm.segment_setpoint(seg, t_in_seg=seg.accel_s + seg.coast_s * 0.5, dt=0.01)
    assert sp_c.pitch_des_rad == 0.0

    # brake window: nose-UP (positive) -> decelerate
    t_brake = seg.accel_s + seg.coast_s + seg.brake_s * 0.5
    sp_b = mm.segment_setpoint(seg, t_in_seg=t_brake, dt=0.01)
    assert sp_b.pitch_des_rad == pytest_approx(+mag)

    # re-trim window: level again
    t_trim = seg.accel_s + seg.coast_s + seg.brake_s + 0.1
    sp_t = mm.segment_setpoint(seg, t_in_seg=t_trim, dt=0.01)
    assert sp_t.pitch_des_rad == 0.0

    # a LEG never commands yaw or roll
    for t in (0.1, seg.accel_s + 1.0, t_brake, t_trim):
        sp = mm.segment_setpoint(seg, t_in_seg=t, dt=0.01)
        assert sp.roll_des_rad == 0.0 and sp.dyaw_rad == 0.0


def test_settle_is_level_vz_damped_hover():
    seg = _seg("settle")
    sp = mm.segment_setpoint(seg, t_in_seg=1.0, dt=0.01)
    assert sp.thrust_mode == "damp"
    assert sp.roll_des_rad == 0.0 and sp.pitch_des_rad == 0.0 and sp.dyaw_rad == 0.0


def test_integrated_pano_yaw_advance_equals_full_sweep():
    """Summing the per-tick dyaw across a whole PANO reproduces the segment's total sweep --
    the timeline the flight loop integrates onto the yaw target is consistent with the schedule."""
    seg = _seg("pano_2_turn")
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
# helpers
# ---------------------------------------------------------------------------
def _seg(name: str):
    return {s.name: s for s in mm.build_schedule()}[name]


def pytest_approx(value, rel=None, abs=None):
    import pytest
    return pytest.approx(value, rel=rel, abs=abs)
