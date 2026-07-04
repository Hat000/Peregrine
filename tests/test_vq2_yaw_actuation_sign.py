"""VQ2 yaw-actuation sign + nearest-gate targeting — sign OPERATOR-EYES-RESOLVED (2026-07-04).

FINAL RESOLUTION (run 20260704_120357, mode A): the operator's eyes on a clean flight confirmed the
drone "yawed in the correct direction, RIGHT." The VQ2 operative yaw path (the A36 yaw_steer_mode="A"
R_cur-yaw recovery + body_rate_sign yaw = +1) turns the drone TOWARD the gate. So on the VQ2 wire the
yaw path does NOT invert: body_rate_sign yaw = +1 is correct, and a gate on the RIGHT commands a
wire-POSITIVE yaw that physically turns the drone RIGHT.

  HISTORY OF THE SIGN (why this file was rewritten twice): A22 set +1, then A36 flipped to -1 based
  on the cmd-vs-gyro same-sign metric (114/0 "opposed"). That metric is the FOOTGUN — it is invariant
  to the flip (post-sign command vs the estimator's mirrored gyro) and gave the wrong sign conclusion
  repeatedly. Two clean flights with operator eyes finally resolved it: mode B (body_rate_sign yaw=-1)
  yawed the WRONG way; mode A (body_rate_sign yaw=+1) yaws RIGHT. The magnitude ~2.1x realization gain
  is fine; only the INVERTING SIGN was the metric artifact. The real seam was the R_cur yaw frame not
  recovering true-yaw (A36 yaw_steer_mode) -- NOT the actuation sign.

  THE MEASURED PLANT (corrected): realized TRUE yaw rate = +2.1 x the emitted wire value (NON-
  inverting) at the ~2.1x realization gain. So with body_rate_sign yaw=+1 the loop is negative
  feedback and CONVERGES onto the gate; body_rate_sign yaw=-1 (mode B) would diverge.

  THE FIX (as flown): vq2_case_c controller_overrides body_rate_sign = (1,1,+1) with
  yaw_steer_mode="A" (the R_cur-yaw recovery). VQ1/case-A keep the seeker default [1,1,-1] +
  yaw_steer_mode "off" (their wire/AHRS is NOT yaw-mirrored) -- byte-identical, untouched.

These tests feed SYNTHETIC gate observations (projected corners -> the seeker's own real PnP)
through the FULL vq2_case_c-configured seeker + controller stack and assert on the WIRE-frame command
(``cmd.body_rate`` is post-body_rate_sign — the exact vector MavlinkClient scales and emits). On the
NON-inverting VQ2 path, wire-POSITIVE yaw = turn RIGHT. No sim required.

NOTE ON PROVENANCE: the physical turn direction asserted here is OPERATOR-EYES-RESOLVED (2026-07-04
run 120357, mode A). It is NOT re-derived from the cmd-vs-gyro metric (which is unreliable/invariant
to the sign). Do not "correct" these asserts back using that metric.

[VQ2 A22 gate-2 turn dive; A36 yaw fix eyes-resolved 2026-07-04]
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from racer.contracts import Frame, Gate, GateObservation, NavState  # noqa: E402
from racer.deploy_profile import VQ2_CMD_RATE_SCALE, vq1_case_a, vq2_case_c  # noqa: E402
from racer.gate_seeker import (  # noqa: E402
    SEEKER_SIGNS,
    GateSeeker,
    GateSeekerConfig,
    make_seeker_controller,
)

# The live-measured VQ2 command->realized body-rate gain MAGNITUDE (~2.1x the wire value). Only the
# MAGNITUDE is measured reliably; the SIGN was OPERATOR-EYES-RESOLVED (2026-07-04 run 120357, mode A)
# as NON-inverting. Used to bound the authority check conservatively.
_VQ2_REALIZED_GAIN_MAG = 2.1
# SIGNED plant: realized TRUE yaw rate = +2.1 x wire (NON-inverting -- eyes-resolved run 120357 mode
# A; NOT the cmd-vs-gyro metric, which is unreliable/invariant to the sign).
_VQ2_PLANT_SIGN = +1.0


# ---------------------------------------------------------------------------
# fixtures: gates, a multi-gate projecting detector, conjugated NavStates
# ---------------------------------------------------------------------------
def _u(v):
    return np.asarray(v, float) / np.linalg.norm(v)


def _gate(position, normal, gate_id=0) -> Gate:
    """A Gate whose through-direction (R_world_gate[:,2]) is ``normal``."""
    n = _u(normal)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = _u(np.cross(a, n))
    y = np.cross(n, x)
    return Gate(gate_id=gate_id, position_ned=np.asarray(position, float),
                R_world_gate=np.column_stack([x, y, n]))


class _MultiGateProjDetector:
    """Project SEVERAL gates' inner corners into the camera from the TRUE pose — clean detections
    the seeker's own PnP solves back into relative levers. Deterministic, model-free: the apparent
    corner span in pixels IS the range signal (bigger box = closer = smaller PnP range_m), which is
    exactly the H1 'nearest = largest apparent size' channel under test."""

    def __init__(self, gates, drone_pos, R_wb):
        from racer.frames import R_camera_from_body
        self.gates = list(gates)
        self.drone_pos = np.asarray(drone_pos, float)
        self.R_wb = np.asarray(R_wb, float)
        self._R_cb = R_camera_from_body()

    def _project(self, gate):
        from racer.frames import CAMERA_INTRINSICS_K
        half = gate.inner_size_m / 2.0
        cg = np.array([[-half, half, 0.0], [half, half, 0.0],
                       [half, -half, 0.0], [-half, -half, 0.0]])
        R_wg = np.asarray(gate.R_world_gate, float)
        K = CAMERA_INTRINSICS_K
        px = []
        for c in cg:
            p_cam = self._R_cb @ (self.R_wb.T @ (gate.position_ned + R_wg @ c - self.drone_pos))
            if p_cam[2] <= 0.05:
                return None
            px.append([K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2],
                       K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]])
        return np.asarray(px, float)

    def detect(self, frame):
        out = []
        for gate in self.gates:
            px = self._project(gate)
            if px is None:
                continue
            out.append(GateObservation(frame_id=frame.frame_id, sim_time_ns=frame.sim_time_ns,
                                       corners_px=px, corner_ids=np.array([0, 1, 2, 3]),
                                       corner_confidence=np.ones(4)))
        return out


def _frame(fid, sim_time_ns):
    return Frame(frame_id=fid, sim_time_ns=sim_time_ns,
                 image_bgr=np.ones((360, 640, 3), dtype=np.uint8))


def _nav_conj(sim_time_ns) -> NavState:
    """A hovering NavState at TRUE attitude identity, encoded the way the case-C Navigator hands it
    to the seeker under ``use_ahrs`` (the ODO conjugation: roll+yaw NEGATED, pitch kept). TRUE
    (0,0,0) conjugates to (0,0,0), so the fixture is trivially exact — and the profile's
    ``true_attitude_from_ahrs`` decode is exercised on the identity (any decode-sign regression
    that scrambles the identity would break the bearing geometry and fail the sign asserts)."""
    return NavState(sim_time_ns=int(sim_time_ns), position_ned=np.zeros(3),
                    velocity_ned=np.zeros(3), yaw=0.0,
                    time_since_vision_update_s=float("inf"))


def _vq2_seeker(gates) -> GateSeeker:
    """Build the seeker EXACTLY as rl.fly_rl.make_seeker does for --deploy-profile vq2_case_c
    (profile seeker_overrides + controller_overrides through the same construction seams), with
    the launch/settle/egress/ramp TIMING zeroed so tick 0 is already PURSUIT. The signs + target
    selection under test are timing-independent; zeroing the windows only removes ticks."""
    profile = vq2_case_c()
    det = _MultiGateProjDetector(gates, np.zeros(3), np.eye(3))
    cfg = GateSeekerConfig(
        settle_s=0.0, launch_ramp_s=0.0, use_spawn_egress=False,
        anchor_release_detections=1, pursuit_ramp_s=0.0, forward_ramp_s=0.0,
        vertical_align_ramp_s=0.0,
        **(profile.seeker_overrides or {}),
    )
    return GateSeeker(config=cfg,
                      controller=make_seeker_controller(**(profile.controller_overrides or {})),
                      detector=det)


def _pursue(seeker, n_ticks=12, dt_s=0.033):
    """Drive ``n_ticks`` of command_visual from a fixed hover (drone held static so the geometry is
    constant — this isolates selection + steering sign from flight dynamics). Returns the LAST
    command (the controller's body-rate slew limiter has converged by then) after asserting every
    tick was a real pursuit tick (a regime routing change would silently invalidate the asserts)."""
    cmd = None
    for k in range(n_ticks):
        t_ns = int(k * dt_s * 1e9)
        cmd = seeker.command_visual(_nav_conj(t_ns), _frame(k, t_ns), 0)
    assert seeker.diag_counts["pursuit"] == n_ticks, (
        f"expected all {n_ticks} ticks in the PURSUIT regime, got {seeker.diag_counts}")
    return cmd


# ===========================================================================
# H2 — the yaw ACTUATION sign, end-to-end. VQ2 yaw path is NON-inverting (mode A,
# body_rate_sign yaw=+1): gate-RIGHT -> wire-POSITIVE cmd -> turns RIGHT (toward gate).
# Physical direction OPERATOR-EYES-RESOLVED (2026-07-04 run 120357 mode A).
# ===========================================================================
def test_gate_on_the_right_commands_wire_positive_yaw():
    """A gate seen RIGHT of the nose (az ~ +0.46 rad): the flown VQ2 config (mode A,
    body_rate_sign yaw=+1) commands a wire-POSITIVE yaw, and the NON-inverting VQ2 wire turns the
    drone RIGHT — toward the gate. (Physical sign OPERATOR-EYES-RESOLVED 2026-07-04 run 120357 mode
    A; NOT re-derived from the cmd-vs-gyro metric, which is unreliable/invariant to the sign.)"""
    cmd = _pursue(_vq2_seeker([_gate([10.0, 5.0, -2.5], normal=[1, 0, 0])]))
    wire_yaw = float(cmd.body_rate[2])
    assert wire_yaw > 0.0, (
        f"gate on the RIGHT must command wire-POSITIVE yaw (non-inverting wire turns right); "
        f"got {wire_yaw:+.3f}")
    # ADEQUATE MAGNITUDE: az=0.46 rad saturates the visual yaw cap; after the slew limiter converges
    # the wire command sits AT the +cap (A36 coordinated-turn cut it 1.5 -> 0.9 to be roll-led).
    assert wire_yaw >= 0.85, f"steady saturated yaw expected at the +0.9 cap, got {wire_yaw:+.3f}"


def test_gate_on_the_left_commands_wire_negative_yaw():
    """Mirror case: a gate seen LEFT of the nose (az ~ -0.46 rad) commands wire-NEGATIVE yaw, which
    the NON-inverting VQ2 wire turns into a LEFT body rotation — toward the gate. (Physical sign
    OPERATOR-EYES-RESOLVED 2026-07-04 run 120357 mode A.)"""
    cmd = _pursue(_vq2_seeker([_gate([10.0, -5.0, -2.5], normal=[1, 0, 0])]))
    wire_yaw = float(cmd.body_rate[2])
    assert wire_yaw < 0.0, (
        f"gate on the LEFT must command wire-NEGATIVE yaw (non-inverting wire turns left); "
        f"got {wire_yaw:+.3f}")
    assert wire_yaw <= -0.85, f"steady saturated yaw expected at the -0.9 cap, got {wire_yaw:+.3f}"


def test_centered_gate_commands_near_zero_yaw():
    """A gate ~dead ahead (az ~ +0.02 — EXACTLY head-on makes the square-PnP two-fold ambiguity
    degenerate, so a hair of offset keeps the pose finite) commands near-ZERO yaw: the loop must be
    QUIET at the fixed point (a sign inversion hides at az~0; the off-axis tests bracket this)."""
    cmd = _pursue(_vq2_seeker([_gate([10.0, 0.2, -2.5], normal=[1, 0, 0])]))
    assert abs(float(cmd.body_rate[2])) < 0.3


def test_saturated_turn_authority_completes_a_90deg_turn_in_time():
    """AUTHORITY: the saturated wire yaw (cap x cmd_rate_scale) at the ~2.1x realization gain must
    still complete a 90-deg gate-to-gate turn in a reasonable time. A36 coordinated-turn cut the
    visual yaw cap 1.5 -> 0.9 DELIBERATELY (the turn is now ROLL-led / banked, not yaw-spun, per the
    operator: "yawed a lot more than needed") -- so the yaw-alone 90 deg time is ~2.8 s and the ROLL
    does the cross-track. We assert the yaw authority is still meaningful (< ~3.5 s for 90 deg
    yaw-alone), not that yaw alone completes it fast."""
    cmd = _pursue(_vq2_seeker([_gate([10.0, 5.0, -2.5], normal=[1, 0, 0])]))
    realized_rps = abs(float(cmd.body_rate[2])) * VQ2_CMD_RATE_SCALE * _VQ2_REALIZED_GAIN_MAG
    t_90deg = (np.pi / 2) / realized_rps
    assert t_90deg < 3.5, (
        f"saturated yaw realizes only {realized_rps:.2f} rad/s -> {t_90deg:.2f} s for 90 deg; "
        "yaw authority too low even for a roll-led turn")


# ===========================================================================
# H1 — nearest-gate targeting: two gates in frame -> lock + steer to the NEAR one
# (steering sign follows H2: gate-RIGHT -> wire-POSITIVE on the non-inverting VQ2 wire).
# ===========================================================================
def test_two_gates_near_right_and_far_centered_targets_the_near_one():
    """NEAR gate off to the RIGHT (8 m, az ~ +0.30) + FAR gate dead-centered (18 m, within the
    22 m acquire range): first-acquisition must lock the NEAR one (its larger apparent corner span
    -> smaller PnP range wins the range+bearing score: 8 + 20*|b| ~ 17 beats 18 + 20*|b| ~ 25)
    and the pursuit must steer TOWARD it (wire-POSITIVE yaw = non-inverting 'turn right'). A
    far-but-centered gate must never outrank the next gate to fly. (Sign eyes-resolved run 120357.)"""
    seeker = _vq2_seeker([_gate([18.0, 0.0, -2.5], normal=[1, 0, 0], gate_id=1),   # far, centered
                          _gate([8.0, 2.5, -2.5], normal=[1, 0, 0], gate_id=0)])   # NEAR, right
    cmd = _pursue(seeker)
    assert seeker._track_range_m is not None
    assert abs(seeker._track_range_m - np.hypot(8.0, 2.5)) < 1.5, (
        f"locked range {seeker._track_range_m:.1f} m — expected the NEAR gate (~8.4 m), "
        "not the far centered one (~18 m)")
    assert float(cmd.body_rate[2]) > 0.0, "must steer RIGHT toward the locked NEAR gate (wire-pos)"


def test_two_gates_near_left_and_far_beyond_acquire_range_targets_the_near_one():
    """NEAR gate off to the LEFT + a FAR gate at 30 m (beyond max_acquire_range_m=22 -> REJECTED
    outright as 'never the next gate to fly'): the near gate is locked and steered toward (wire-
    NEGATIVE yaw = non-inverting 'turn left'). Covers the reject branch of the A5 far-gate trap
    alongside the score branch. (Sign eyes-resolved run 120357 mode A.)"""
    seeker = _vq2_seeker([_gate([30.0, 0.0, -2.5], normal=[1, 0, 0], gate_id=1),   # far, rejected
                          _gate([8.0, -2.5, -2.5], normal=[1, 0, 0], gate_id=0)])  # NEAR, left
    cmd = _pursue(seeker)
    assert seeker._track_range_m is not None
    assert abs(seeker._track_range_m - np.hypot(8.0, 2.5)) < 1.5, (
        f"locked range {seeker._track_range_m:.1f} m — expected the NEAR gate (~8.4 m)")
    assert float(cmd.body_rate[2]) < 0.0, "must steer LEFT toward the locked NEAR gate (wire-neg)"


# ===========================================================================
# CLOSED LOOP — NON-inverting VQ2 plant (realized = +2.1 x wire), sign eyes-resolved.
# ===========================================================================
def _closed_loop_final_bearing(controller_override: dict, plant_sign=_VQ2_PLANT_SIGN,
                               n_ticks=90, dt_s=0.033):
    """Fly the yaw channel closed-loop against the MEASURED wire plant. The SIGN is NON-inverting
    (realized TRUE yaw rate = +2.1 x the emitted wire value) -- OPERATOR-EYES-RESOLVED (2026-07-04
    run 120357 mode A), NOT the cmd-vs-gyro metric (unreliable/invariant to the sign). Each tick: the
    detector projects the gate from the CURRENT true yaw -> seeker+controller command -> the wire
    value integrates the true yaw. Returns the final gate bearing error |yaw_to_gate - true_yaw|.

    Tests the ACTUATION-sign loop in the harness's own SELF-CONSISTENT (no-yaw-mirror) frame: it uses
    nav.yaw=-psi and _gate_dir_world in that same frame, so the R_cur-yaw SEAM (yaw_steer_mode) is
    forced "off" here -- the harness CANNOT represent the real VQ2 yaw mirror that mode A corrects
    (that physical direction is OPERATOR-EYES-RESOLVED, run 120357 mode A, + asserted by the profile
    test). In this no-mirror proxy the ACTUATION sign body_rate_sign yaw=+1 against the non-inverting
    plant is NEGATIVE feedback and CONVERGES; the counterfactual yaw=-1 DIVERGES under the same
    plant."""
    from racer.frames import R_world_from_body
    profile = vq2_case_c()
    gate = _gate([10.0, 5.0, -2.5], normal=[1, 0, 0])          # bearing atan2(5,10) ~ +0.46 rad
    det = _MultiGateProjDetector([gate], np.zeros(3), np.eye(3))
    cfg = GateSeekerConfig(
        settle_s=0.0, launch_ramp_s=0.0, use_spawn_egress=False,
        anchor_release_detections=1, pursuit_ramp_s=0.0, forward_ramp_s=0.0,
        vertical_align_ramp_s=0.0,
        **(profile.seeker_overrides or {}),
    )
    # yaw_steer_mode "off": the harness is a NO-MIRROR proxy (see docstring). It isolates the
    # actuation sign; the real-wire R_cur-yaw mirror (mode A) is eyes-resolved + profile-tested.
    overrides = {**(profile.controller_overrides or {}), "yaw_steer_mode": "off",
                 **controller_override}
    seeker = GateSeeker(config=cfg, controller=make_seeker_controller(**overrides), detector=det)
    psi = 0.0                                                   # TRUE yaw (world), starts north
    for k in range(n_ticks):
        t_ns = int(k * dt_s * 1e9)
        det.R_wb = R_world_from_body(0.0, 0.0, psi)             # camera follows the true yaw
        nav = NavState(sim_time_ns=t_ns, position_ned=np.zeros(3), velocity_ned=np.zeros(3),
                       yaw=-psi,                                # the ODO conjugation (yaw negated)
                       time_since_vision_update_s=float("inf"))
        cmd = seeker.command_visual(nav, _frame(k, t_ns), 0)
        wire = float(cmd.body_rate[2]) * VQ2_CMD_RATE_SCALE     # what MavlinkClient emits
        # MEASURED PLANT: realized TRUE yaw rate = plant_sign(+1) * 2.1 * wire (NON-inverting).
        psi += plant_sign * _VQ2_REALIZED_GAIN_MAG * wire * dt_s
    los = np.arctan2(5.0, 10.0)
    return abs(float(np.arctan2(np.sin(los - psi), np.cos(los - psi))))


def test_closed_loop_converges_onto_the_gate_with_the_flown_sign():
    """In the no-mirror proxy harness, the flown ACTUATION sign body_rate_sign yaw=+1 against the
    NON-inverting plant is NEGATIVE feedback: ~3 s of ticks turn the nose onto the gate bearing
    (error -> ~0). (Physical real-wire direction OPERATOR-EYES-RESOLVED 2026-07-04 run 120357 mode A;
    the harness cannot see the mirror, so it forces yaw_steer_mode off -- see _closed_loop_final_bearing.)"""
    err = _closed_loop_final_bearing({})                        # the flown actuation sign (brs yaw=+1)
    assert err < 0.08, f"loop must converge onto the gate; final bearing error {err:.3f} rad"


def test_closed_loop_diverges_with_the_mode_b_minus_one_yaw_sign():
    """COUNTERFACTUAL pin: the SAME non-inverting harness with body_rate_sign yaw=-1 (mode B)
    positive-feedbacks — the bearing error GROWS (the mode-B mis-fly: yawed the WRONG way, run
    20260704_051851-family). If the plant convention here ever drifts, this and the test above fail
    together, flagging the harness. (Physical polarity eyes-resolved; not from the cmd-vs-gyro metric.)"""
    err = _closed_loop_final_bearing({"body_rate_sign": (1.0, 1.0, -1.0)})
    assert err > 0.5, f"the mode-B -1 sign must diverge away from the gate; final bearing error {err:.3f} rad"


# ===========================================================================
# The profile seam: VQ2 yaw path is NON-inverting (body_rate_sign yaw=+1, eyes-resolved);
# VQ1 / the seeker default stay byte-identical (their wire is not yaw-mirrored).
# ===========================================================================
def test_vq2_profile_carries_noninverting_yaw_body_rate_sign():
    """The flown vq2_case_c carries body_rate_sign yaw=+1 (the VQ2 yaw path does NOT invert) paired
    with the A36 yaw_steer_mode="A" R_cur-yaw recovery. OPERATOR-EYES-RESOLVED (2026-07-04 run 120357
    mode A: "yawed in the correct direction, RIGHT"); NOT re-derived from the cmd-vs-gyro metric."""
    co = vq2_case_c().controller_overrides
    np.testing.assert_allclose(np.asarray(co["body_rate_sign"], float), [1.0, 1.0, 1.0])
    assert co.get("yaw_steer_mode") == "A"


def test_vq1_and_seeker_default_signs_unchanged():
    """BYTE-IDENTITY guard: the VQ2 yaw fix must NOT touch the VQ1-measured seeker default ([1,1,-1],
    flight-proven on the VQ1 wire, which is NOT yaw-mirrored) nor give vq1_case_a any controller
    overrides. The +1 + yaw_steer_mode recovery are VQ2-ONLY."""
    assert vq1_case_a().controller_overrides is None
    np.testing.assert_allclose(SEEKER_SIGNS["body_rate_sign"], [1.0, 1.0, -1.0])
    np.testing.assert_allclose(make_seeker_controller().body_rate_sign, [1.0, 1.0, -1.0])
    assert make_seeker_controller().yaw_steer_mode == "off"
