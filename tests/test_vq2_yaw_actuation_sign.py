"""VQ2 yaw-actuation sign + nearest-gate targeting — A22 (2026-07-02), CORRECTED by A36 (2026-07-03).

A36 OVERTURNS A22's sign. A22 concluded the VQ2 wire honors PLAIN FRD yaw sign and set
``vq2_case_c().controller_overrides["body_rate_sign"] = (1,1,1)`` off ONE run (20260702_152528).
That run PREDATED A22's own fix commit by ~36 min and flew on the seeker default z=-1; A22 read the
pre-sign/post-sign relationship backwards and flipped a WORKING sign. Every one of the 16 flights
since (all on +1) shows the SENT yaw command correctly aimed at the gate yet the drone yawing AWAY:
run 20260704_032626 whole-flight cmd_yaw vs raw_gyro = 114/0 OPPOSED, cmd-toward-gate 132/5 while
realized-toward-gate only 23/141 — a positive-feedback yaw loop, the ROOT of the never-turns-to-
gate-2 orbit (and the 203deg whip the A31 orbit-breaker chased). Cross-run: ALL 16 post-A22 flights
inverted. Verdict: the VQ2 wire INVERTS the yaw-rate command; the correct actuation sign is the
VQ1-proven seeker default z=-1.

  THE MEASURED PLANT: realized TRUE yaw rate = -2.1 x the emitted wire value (the wire NEGATES the
  FRD yaw-rate command at the known ~2.1x realization gain). A22's harness baked in +2.1 (the plain
  FRD premise) — that sign was the error; A36 corrects it to -2.1.

  THE FIX: ``vq2_case_c().controller_overrides["body_rate_sign"] = (1,1,-1)`` — the per-wire
  actuation convention lives in the deploy profile, exactly like ``gyro_sign``. This equals the
  VQ1-measured seeker default, so with the correct sign the SENT (post-sign) wire command for a gate
  on the RIGHT is NEGATIVE, and the inverting plant turns the drone RIGHT (toward the gate).

  T2 (estimator) and T3 (vision) from A22 STILL HOLD and are unaffected — they only ever exonerated
  the estimator + vision (d(true_yaw)/dt == -raw_gyro corr +0.96; camera bearing agrees with gyro).
  They never proved the actuation sign; A36 fixes only the actuation sign.

These tests feed SYNTHETIC gate observations (projected corners -> the seeker's own real PnP)
through the FULL vq2_case_c-configured seeker + controller stack and assert on the WIRE-frame
command (``cmd.body_rate`` is post-body_rate_sign — the exact vector MavlinkClient scales and
emits). With the inverting plant, wire-NEGATIVE yaw = turn RIGHT. No sim required.

[VQ2 A22 gate-2 turn dive, 2026-07-02; A36 sign correction, 2026-07-03]
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

# The live-measured VQ2 command->realized body-rate gain MAGNITUDE (run 20260702_152528: |realized
# yaw rate| = ~2.1x the wire value). The wire INVERTS the sign, so the SIGNED plant gain is -2.1
# (A36 correction; A22 wrongly used +2.1). Used to bound the authority check conservatively.
_VQ2_REALIZED_GAIN_MAG = 2.1
# SIGNED plant: realized TRUE yaw rate = -2.1 x wire (the VQ2 wire negates the FRD yaw command).
_VQ2_PLANT_SIGN = -1.0


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
# H2 — the yaw ACTUATION sign, end-to-end. The VQ2 wire INVERTS (A36): with the
# corrected z=-1 sign, gate-RIGHT -> wire-NEGATIVE cmd -> inverting plant turns RIGHT.
# ===========================================================================
def test_gate_on_the_right_commands_wire_negative_yaw():
    """A gate seen RIGHT of the nose (az ~ +0.46 rad): with the A36-corrected sign z=-1 the SENT
    wire yaw-rate command is NEGATIVE, and because the VQ2 wire INVERTS (realized = -2.1 x wire)
    that turns the drone RIGHT — toward the gate. On the A22 z=+1 code the command came out POSITIVE
    and the inverting plant turned the drone LEFT: az grows, the loop positive-feedbacks, the gate
    spins out of frame ('never turned toward gate 2'). This test FAILS on the A22 sign."""
    cmd = _pursue(_vq2_seeker([_gate([10.0, 5.0, -2.5], normal=[1, 0, 0])]))
    wire_yaw = float(cmd.body_rate[2])
    assert wire_yaw < 0.0, (
        f"gate on the RIGHT must command wire-NEGATIVE yaw (the inverting VQ2 wire then turns "
        f"right); got {wire_yaw:+.3f} — the yaw actuation sign is wrong (A22 regression)")
    # ADEQUATE MAGNITUDE: az=0.46 rad at kp_att=4 demands ~1.84 -> the visual yaw cap (1.5)
    # saturates; after the slew limiter converges the wire command must sit AT the -cap.
    assert wire_yaw <= -1.4, f"steady saturated yaw expected at the -1.5 cap, got {wire_yaw:+.3f}"


def test_gate_on_the_left_commands_wire_positive_yaw():
    """Mirror case: a gate seen LEFT of the nose (az ~ -0.46 rad) commands wire-POSITIVE yaw, which
    the inverting VQ2 wire turns into a LEFT body rotation — toward the gate."""
    cmd = _pursue(_vq2_seeker([_gate([10.0, -5.0, -2.5], normal=[1, 0, 0])]))
    wire_yaw = float(cmd.body_rate[2])
    assert wire_yaw > 0.0, (
        f"gate on the LEFT must command wire-POSITIVE yaw (the inverting wire then turns left); "
        f"got {wire_yaw:+.3f}")
    assert wire_yaw >= 1.4, f"steady saturated yaw expected at the +1.5 cap, got {wire_yaw:+.3f}"


def test_centered_gate_commands_near_zero_yaw():
    """A gate ~dead ahead (az ~ +0.02 — EXACTLY head-on makes the square-PnP two-fold ambiguity
    degenerate, so a hair of offset keeps the pose finite) commands near-ZERO yaw: the loop must be
    QUIET at the fixed point (a sign inversion hides at az~0; the off-axis tests bracket this)."""
    cmd = _pursue(_vq2_seeker([_gate([10.0, 0.2, -2.5], normal=[1, 0, 0])]))
    assert abs(float(cmd.body_rate[2])) < 0.3


def test_saturated_turn_authority_completes_a_90deg_turn_in_time():
    """AUTHORITY: the saturated wire yaw (cap x cmd_rate_scale) at the live-measured ~2.1x
    realization gain MAGNITUDE must complete a 90-deg gate-to-gate turn in well under 2 s — i.e. the
    cmd_rate_scale=0.4 + the 1.5 rad/s visual cap leave ENOUGH yaw rate once the sign is right.
    (run 20260702_152528: |realized/wire| 2.13; the profile assumes 2.5 — use the smaller.)"""
    cmd = _pursue(_vq2_seeker([_gate([10.0, 5.0, -2.5], normal=[1, 0, 0])]))
    realized_rps = abs(float(cmd.body_rate[2])) * VQ2_CMD_RATE_SCALE * _VQ2_REALIZED_GAIN_MAG
    t_90deg = (np.pi / 2) / realized_rps
    assert t_90deg < 2.0, (
        f"saturated yaw realizes only {realized_rps:.2f} rad/s -> {t_90deg:.2f} s for 90 deg; "
        "not enough authority for a gate-to-gate turn")


# ===========================================================================
# H1 — nearest-gate targeting: two gates in frame -> lock + steer to the NEAR one
# (steering sign follows H2: gate-RIGHT -> wire-NEGATIVE with the inverting wire).
# ===========================================================================
def test_two_gates_near_right_and_far_centered_targets_the_near_one():
    """NEAR gate off to the RIGHT (8 m, az ~ +0.30) + FAR gate dead-centered (18 m, within the
    22 m acquire range): first-acquisition must lock the NEAR one (its larger apparent corner span
    -> smaller PnP range wins the range+bearing score: 8 + 20*|b| ~ 17 beats 18 + 20*|b| ~ 25)
    and the pursuit must steer TOWARD it (wire-NEGATIVE yaw, the inverting-wire 'turn right'). A
    far-but-centered gate must never outrank the next gate to fly."""
    seeker = _vq2_seeker([_gate([18.0, 0.0, -2.5], normal=[1, 0, 0], gate_id=1),   # far, centered
                          _gate([8.0, 2.5, -2.5], normal=[1, 0, 0], gate_id=0)])   # NEAR, right
    cmd = _pursue(seeker)
    assert seeker._track_range_m is not None
    assert abs(seeker._track_range_m - np.hypot(8.0, 2.5)) < 1.5, (
        f"locked range {seeker._track_range_m:.1f} m — expected the NEAR gate (~8.4 m), "
        "not the far centered one (~18 m)")
    assert float(cmd.body_rate[2]) < 0.0, "must steer RIGHT toward the locked NEAR gate (wire-neg)"


def test_two_gates_near_left_and_far_beyond_acquire_range_targets_the_near_one():
    """NEAR gate off to the LEFT + a FAR gate at 30 m (beyond max_acquire_range_m=22 -> REJECTED
    outright as 'never the next gate to fly'): the near gate is locked and steered toward (wire-
    POSITIVE yaw = the inverting-wire 'turn left'). Covers the reject branch of the A5 far-gate trap
    alongside the score branch."""
    seeker = _vq2_seeker([_gate([30.0, 0.0, -2.5], normal=[1, 0, 0], gate_id=1),   # far, rejected
                          _gate([8.0, -2.5, -2.5], normal=[1, 0, 0], gate_id=0)])  # NEAR, left
    cmd = _pursue(seeker)
    assert seeker._track_range_m is not None
    assert abs(seeker._track_range_m - np.hypot(8.0, 2.5)) < 1.5, (
        f"locked range {seeker._track_range_m:.1f} m — expected the NEAR gate (~8.4 m)")
    assert float(cmd.body_rate[2]) > 0.0, "must steer LEFT toward the locked NEAR gate (wire-pos)"


# ===========================================================================
# CLOSED LOOP — the run-20260702_152528 physics, A36-CORRECTED: realized = -2.1 x wire
# ===========================================================================
def _closed_loop_final_bearing(controller_sign_override: dict, plant_sign=_VQ2_PLANT_SIGN,
                               n_ticks=90, dt_s=0.033):
    """Fly the yaw channel closed-loop against the MEASURED wire plant (A36: realized TRUE yaw rate
    = -2.1 x the emitted wire value — the wire NEGATES the FRD yaw command at the known realization
    gain). Each tick: detector projects the gate from the CURRENT true yaw -> seeker+controller
    command -> the (negated) wire value integrates the true yaw. Returns the final gate bearing
    error |yaw_to_gate - true_yaw|.

    This is the loop-STABILITY regression the per-tick sign asserts can't express: with the correct
    z=-1 sign against the inverting plant the bearing error CONVERGES to ~0 (the drone turns onto
    the gate); with the A22 z=+1 sign the SAME harness positive-feedbacks and the bearing DIVERGES
    — the live gate-2 spin-away, reproduced offline."""
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
    overrides = {**(profile.controller_overrides or {}), **controller_sign_override}
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
        # A36 MEASURED PLANT: realized TRUE yaw rate = plant_sign(-1) * 2.1 * wire (the wire inverts)
        psi += plant_sign * _VQ2_REALIZED_GAIN_MAG * wire * dt_s
    los = np.arctan2(5.0, 10.0)
    return abs(float(np.arctan2(np.sin(los - psi), np.cos(los - psi))))


def test_closed_loop_converges_onto_the_gate_with_the_corrected_sign():
    """With the A36-corrected yaw sign z=-1 against the INVERTING wire plant, the closed yaw loop is
    NEGATIVE feedback: 3 s of ticks turn the nose onto the gate bearing (error -> ~0). The live
    physics, made a unit test. (The vq2_case_c profile as shipped now carries z=-1.)"""
    err = _closed_loop_final_bearing({})                        # the vq2_case_c profile as shipped
    assert err < 0.08, f"loop must converge onto the gate; final bearing error {err:.3f} rad"


def test_closed_loop_diverges_with_the_a22_plus_one_yaw_sign():
    """COUNTERFACTUAL pin: the SAME inverting-wire harness with the A22 z=+1 sign positive-feedbacks
    — the bearing error GROWS (the drone turns away from gate 2, the two-day orbit). If the plant
    convention here ever drifts, this and the test above fail together, flagging the harness."""
    err = _closed_loop_final_bearing({"body_rate_sign": (1.0, 1.0, 1.0)})
    assert err > 0.5, f"the A22 +1 sign must diverge away from the gate; final bearing error {err:.3f} rad"


# ===========================================================================
# The profile seam: VQ2 opts in to z=-1; VQ1 / the seeker default stay byte-identical
# ===========================================================================
def test_vq2_profile_carries_inverting_wire_body_rate_sign():
    """A36: vq2_case_c carries the CORRECTED yaw actuation sign z=-1 (the VQ2 wire INVERTS the
    FRD yaw command; this reverts A22's mistaken +1). The per-wire actuation convention lives in the
    profile beside gyro_sign; it equals the VQ1-measured seeker default."""
    sign = np.asarray(vq2_case_c().controller_overrides["body_rate_sign"], float)
    np.testing.assert_allclose(sign, [1.0, 1.0, -1.0])


def test_vq1_and_seeker_default_signs_unchanged():
    """BYTE-IDENTITY guard: the fix must NOT touch the VQ1-measured seeker default ([1,1,-1],
    flight-proven on the VQ1 wire) nor give vq1_case_a any controller overrides."""
    assert vq1_case_a().controller_overrides is None
    np.testing.assert_allclose(SEEKER_SIGNS["body_rate_sign"], [1.0, 1.0, -1.0])
    np.testing.assert_allclose(make_seeker_controller().body_rate_sign, [1.0, 1.0, -1.0])
