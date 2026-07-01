"""Case-C yaw-frame fix (A14) — the seeker must read the TRUE AHRS attitude, not the ODO-conjugated
one the case-C Navigator hands it, so it steers toward the gate the camera actually sees instead of a
yaw-mirrored phantom bearing.

ROOT CAUSE (A14, root-caused offline): under ``use_ahrs`` (vq2_case_c) the Navigator re-encodes the
TRUE AHRS attitude into the legacy ODOMETRY R_y(pi) conjugation before writing ``NavState``, so every
tick ``NavState = (roll=-true_roll, pitch=+true_pitch, yaw=-true_yaw)``. The seeker's gate-direction
geometry (``_gate_dir_world``) rotated the seen-gate lever into WORLD NED with this conjugated euler,
so the world heading it steered at (``yaw_des``) was MIRRORED in yaw (and skewed by the un-un-mirrored
roll). The controller's ``odo_att_sign=[-1,1,1]`` un-conjugates roll but NOT yaw, so ``R_cur`` carried
the SAME yaw mirror -- the controller half of the bug. For a gate the camera plainly sees at world
bearing +B, the buggy chain aimed at a DIFFERENT (mirrored) world bearing.

THE FIX: ``GateSeekerConfig.true_attitude_from_ahrs`` (ON in vq2_case_c) makes the seeker recover the
TRUE euler ``(-nav.roll, nav.pitch, -nav.yaw)`` for all geometry + heading bookkeeping and pass the
controller a yaw-negated nav so ``R_cur`` is R_true -- NO sign knob touched. Default OFF is byte-
identical to the raw-nav path (VQ1 / case-A).

WHAT THESE TESTS PIN. The DECISIVE, attitude-independent invariant is the world HEADING the seeker
steers at (``_last_yaw_des`` = ``atan2`` of ``_gate_dir_world``): with the fix ON, for a gate at a
FIXED world position the recovered world bearing equals the TRUE world bearing to that gate REGARDLESS
of the drone's true attitude (the camera sees the same gate; the correct world heading is fixed). The
buggy flag-off path yaws that heading with the mirrored attitude -- a wrong, attitude-dependent bearing.
We also confirm the full seeker->controller wire command is finite + bounded, and that the default
(flag-off) path is byte-identical to the pre-fix raw-nav command.

[VQ2 A14 yaw-mirror fix, 2026-06-30]
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from racer.contracts import ControlMode, Frame, Gate, GateObservation, NavState  # noqa: E402
from racer.frames import CAMERA_INTRINSICS_K, R_camera_from_body, R_world_from_body  # noqa: E402
from racer.gate_seeker import GateSeeker, GateSeekerConfig, make_seeker_controller  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures — mirror tests/test_gate_seeker.py's _ProjDetector / _frame conventions
# ---------------------------------------------------------------------------
def _u(v):
    return np.asarray(v, float) / np.linalg.norm(v)


def _gate(position, normal, gate_id=0) -> Gate:
    n = _u(normal)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = _u(np.cross(a, n))
    y = np.cross(n, x)
    R = np.column_stack([x, y, n])
    return Gate(gate_id=gate_id, position_ned=np.asarray(position, float), R_world_gate=R)


class _ProjDetector:
    """Project a gate's inner corners into the camera from a TRUE pose -> a clean detection the
    seeker's own PnP solves into a relative lever. ``R_wb`` is the drone's TRUE body->world attitude
    (what the physical camera actually sees); the seeker separately consumes the CONJUGATED NavState."""

    def __init__(self, gate: Gate, drone_pos, R_wb):
        self.gate = gate
        self.drone_pos = np.asarray(drone_pos, float)
        self.R_wb = np.asarray(R_wb, float)
        self._R_cb = R_camera_from_body()

    def detect(self, frame):
        half = self.gate.inner_size_m / 2.0
        corners_gate = np.array([[-half, half, 0.0], [half, half, 0.0],
                                 [half, -half, 0.0], [-half, -half, 0.0]])
        R_wg = np.asarray(self.gate.R_world_gate, float)
        K = CAMERA_INTRINSICS_K
        px = []
        for cg in corners_gate:
            p_world = self.gate.position_ned + R_wg @ cg
            p_cam = self._R_cb @ (self.R_wb.T @ (p_world - self.drone_pos))
            if p_cam[2] <= 0.05:
                return []
            px.append([K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2],
                       K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]])
        return [GateObservation(frame_id=frame.frame_id, sim_time_ns=frame.sim_time_ns,
                                corners_px=np.asarray(px, float), corner_ids=np.array([0, 1, 2, 3]),
                                corner_confidence=np.ones(4))]


def _frame(fid=0, sim_time_ns=0):
    return Frame(frame_id=fid, sim_time_ns=sim_time_ns,
                 image_bgr=np.ones((360, 640, 3), dtype=np.uint8))


DRONE_POS = np.array([0.0, 0.0, -2.5])


def _conj_nav(true_roll, true_pitch, true_yaw, *, sim_time_ns):
    """The CONJUGATED NavState the case-C Navigator emits under use_ahrs from a TRUE attitude:
    roll + yaw NEGATED, pitch kept (the R_y(pi) ODOMETRY conjugation). Rates zero (static test)."""
    return NavState(
        sim_time_ns=sim_time_ns,
        position_ned=DRONE_POS.copy(),
        velocity_ned=np.zeros(3),
        roll=-float(true_roll),
        pitch=float(true_pitch),
        yaw=-float(true_yaw),
        angular_rate_body=np.zeros(3),
        time_since_vision_update_s=0.05,   # finite -> anchored (past the launch hold)
    )


def _clean_seeker_config(**overrides) -> GateSeekerConfig:
    """A config that routes straight into regime-3 pursuit (egress / pass / vertical-align OFF,
    settle 0, anchor on the first detection) so the STEERING sign is isolated from the launch guards."""
    cfg = dict(
        cruise_speed=3.0, launch_ramp_s=0.0, settle_s=0.0,
        anchor_release_detections=1,
        use_spawn_egress=False, use_pass_dead_reckon=False, use_vertical_align=False,
        pursuit_ramp_s=0.0, forward_ramp_s=0.0, pursuit_yaw_slew_rps=100.0,  # no slew/ramp masking
        visual_yaw_rate_cap_rps=10.0,       # don't clamp the sign under test
    )
    cfg.update(overrides)
    return GateSeekerConfig(**cfg)


def _drive_pursuit(seeker: GateSeeker, true_rpy, ticks_ns):
    """Drive command_visual for a sequence of sim-time ticks at the SAME true attitude (fed as the
    CONJUGATED nav); return the LAST command. First tick anchors the latch; the second is pursuit."""
    tr, tp, ty = true_rpy
    cmd = None
    for k, t_ns in enumerate(ticks_ns):
        cmd = seeker.command_visual(_conj_nav(tr, tp, ty, sim_time_ns=t_ns), _frame(k, t_ns), 0)
    return cmd


def _right_gate(true_rpy, forward_m=12.0, right_m=3.0):
    """A gate ``forward_m`` ahead + ``right_m`` to the RIGHT of the drone in the TRUE body frame, so it
    is always inside the +20deg-up camera FOV for the parametrized attitude. Returns ``(gate,
    expected_world_bearing)`` -- the world-NED heading from the drone to the gate (the correct steering
    heading a fully-consistent TRUE-attitude pipeline must recover)."""
    tr, tp, ty = true_rpy
    R = R_world_from_body(tr, tp, ty)
    fwd = R @ np.array([1.0, 0.0, 0.0])       # body +x (nose) in world
    right = R @ np.array([0.0, 1.0, 0.0])     # body +y (starboard) in world
    pos = DRONE_POS + forward_m * fwd + right_m * right
    gate = _gate(pos, normal=-fwd)            # faces back toward the drone (we fly along +fwd)
    to = pos - DRONE_POS
    return gate, float(np.arctan2(to[1], to[0]))


def _seeker(flag, true_rpy, gate) -> GateSeeker:
    tr, tp, ty = true_rpy
    det = _ProjDetector(gate, DRONE_POS.copy(), R_world_from_body(tr, tp, ty))
    return GateSeeker(config=_clean_seeker_config(true_attitude_from_ahrs=flag),
                      controller=make_seeker_controller(), detector=det)


# ===========================================================================
# THE FIX: the recovered world steering heading == the TRUE world bearing to the seen gate.
# The camera sees a gate to the drone's RIGHT; the fix must steer at that gate's TRUE world bearing
# (un-mirrored), for EVERY true attitude. The buggy flag-off path mirrors the heading with the
# conjugated yaw -> aims at a phantom bearing on the WRONG side.
# ===========================================================================
@pytest.mark.parametrize("true_yaw", [0.0, 0.5, -0.5])
@pytest.mark.parametrize("true_roll", [0.0, 0.2])
def test_true_attitude_recovers_the_correct_world_heading(true_yaw, true_roll):
    """With the flag ON, fed the CONJUGATED NavState the case-C Navigator emits, the seeker steers at
    the TRUE world bearing to the RIGHT gate the camera sees (``_last_yaw_des`` == the true world
    bearing), for every true yaw + a nonzero-roll case (the roll case proves roll handling -- a
    mishandled roll leaks into the recovered world heading). The full seeker->controller wire command
    is finite + bounded. Verified through the real ``make_seeker_controller`` (odo_att_sign=[-1,1,1])."""
    rpy = (true_roll, 0.05, true_yaw)
    gate, expected = _right_gate(rpy)
    seeker = _seeker(True, rpy, gate)
    cmd = _drive_pursuit(seeker, rpy, [0, 10_000_000])
    assert cmd.mode is ControlMode.BODY_RATE
    assert np.all(np.isfinite(cmd.body_rate)) and np.isfinite(cmd.thrust)
    assert np.linalg.norm(cmd.body_rate) <= seeker.controller.max_body_rate_rps + 1e-9
    assert seeker._last_yaw_des is not None, "fix must reach pursuit (gate is in view)"
    assert seeker._last_yaw_des == pytest.approx(expected, abs=1e-3), (
        f"fix must steer at the TRUE world bearing {expected:+.4f} to the RIGHT gate; "
        f"got {seeker._last_yaw_des:+.4f} at true_yaw={true_yaw}, true_roll={true_roll}")


def test_flag_off_mirrors_the_heading_the_A14_bug():
    """COMPANION: with the flag OFF (default) and a NONZERO true yaw, the buggy MIRRORED steering is
    reproduced -- the recovered world heading is yaw-mirrored (aims at a phantom bearing on the WRONG
    side of the true one), while the fix recovers the true bearing. Pins WHY the flag is needed and
    that it flips the mirror."""
    rpy = (0.0, 0.05, 0.3)      # true yaw +0.3 (nose east), gate to the right
    gate, expected = _right_gate(rpy)      # expected ~ +0.545 (further east, toward the gate)
    on = _seeker(True, rpy, gate)
    _drive_pursuit(on, rpy, [0, 10_000_000])
    off = _seeker(False, rpy, gate)
    _drive_pursuit(off, rpy, [0, 10_000_000])

    assert on._last_yaw_des == pytest.approx(expected, abs=1e-3)   # fix: aims at the gate
    # bug: the heading is yaw-mirrored -> it lands well away from the true bearing, on the WRONG side
    # (here it aims back near north / west of the true east-of-nose gate, off by > 2*true_yaw).
    assert abs(off._last_yaw_des - expected) > 0.3
    assert off._last_yaw_des < on._last_yaw_des - 0.3


def test_nonzero_roll_changes_the_wire_command_between_flag_states():
    """A nonzero true roll makes the fix and the bug produce DIFFERENT wire commands (the fix un-mirrors
    the roll+yaw geometry AND the controller R_cur). Guards that the flag is load-bearing on the WIRE,
    not just on the instrumentation heading (with zero roll the geometry + R_cur mirrors cancel on the
    wire, so a nonzero roll is required to exercise the wire-level difference)."""
    rpy = (0.2, 0.05, 0.3)
    gate, _ = _right_gate(rpy)
    s_fix = _seeker(True, rpy, gate)
    s_bug = _seeker(False, rpy, gate)
    fix = _drive_pursuit(s_fix, rpy, [0, 10_000_000])
    bug = _drive_pursuit(s_bug, rpy, [0, 10_000_000])
    assert s_fix._last_yaw_des is not None and s_bug._last_yaw_des is not None  # both reached pursuit
    assert not np.allclose(fix.body_rate, bug.body_rate, atol=1e-6)


# ===========================================================================
# BYTE-IDENTITY: flag OFF (default) == today's raw-nav path (VQ1 / case-A unchanged)
# ===========================================================================
def test_flag_off_byte_identical_to_raw_nav_path():
    """The default (flag OFF) command must be BYTE-IDENTICAL to the pre-fix raw-nav path: for a plain
    (non-conjugated) NavState the flag-DEFAULT seeker command equals a seeker built with the flag
    explicitly False. Pins that the guard never leaks into the default path (VQ1 / case-A byte-id)."""
    gate = _gate([12.0, 6.0, -2.5], normal=[1.0, 0.0, 0.0])
    R_wb = R_world_from_body(0.0, 0.05, 0.3)

    def _run(flag):
        det = _ProjDetector(gate, DRONE_POS.copy(), R_wb)
        cfg = (_clean_seeker_config() if flag is None
               else _clean_seeker_config(true_attitude_from_ahrs=flag))
        s = GateSeeker(config=cfg, controller=make_seeker_controller(), detector=det)
        # a plain NavState (NOT conjugated): raw euler == the reported euler.
        for k, t in enumerate([0, 10_000_000]):
            nav = NavState(sim_time_ns=t, position_ned=DRONE_POS.copy(), velocity_ned=np.zeros(3),
                           roll=0.0, pitch=0.05, yaw=0.3, angular_rate_body=np.zeros(3),
                           time_since_vision_update_s=0.05)
            cmd = s.command_visual(nav, _frame(k, t), 0)
        return cmd

    cmd_default = _run(None)     # flag not set in config (default False)
    cmd_off = _run(False)        # flag explicitly False
    np.testing.assert_array_equal(cmd_default.body_rate, cmd_off.body_rate)
    assert cmd_default.thrust == cmd_off.thrust
