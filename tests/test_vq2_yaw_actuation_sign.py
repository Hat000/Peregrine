"""VQ2 yaw-actuation sign + nearest-gate targeting — the A22 gate-2 no-turn fix (2026-07-02).

RUN 20260702_152528_rl_s1_f1 forensics (live 30 s VQ2 flight, vq2_case_c): the drone threaded
gate 1 cleanly, then NEVER turned toward the well-detected gate 2 — it turned the OTHER way,
saturated, spun the gate out of frame and flew into the wall. Three independent channels agree
on the mechanism (pre-collision window, t < 9.5 s):

  T1 realized-vs-commanded: corr(pre-sign FRD yaw cmd, estimator yaw rate) = -0.79 with only 5%
     sign agreement (mean cmd +1.30 rad/s -> mean realized -1.09 rad/s), while the realized rate
     follows the POST-sign WIRE value at corr +0.79 and the known ~2.1x realization gain.
     => the VQ2 wire honors the PLAIN FRD yaw-rate sign; the seeker default ``SEEKER_SIGNS``
        body_rate_sign z=-1 (the VQ1-measured inversion) flips every VQ2 yaw command.
  T2 estimator self-consistency: d(true_yaw)/dt == -raw_gyro (corr +0.96) — the estimate was
     faithfully integrating the A9 sign-corrected gyro. The ESTIMATOR was right.
  T3 optical cross-check: the camera-measured gate bearing swept +0.43 rad/s (gate sliding RIGHT
     across the frame) while the estimated yaw went -0.96 rad/s (body turning LEFT) — vision and
     gyro independently confirm the body turned OPPOSITE the command.

The yaw loop was therefore POSITIVE feedback: any bearing error grew, the command saturated the
wrong way, and the gate was spun OUT of frame — the "never turned toward gate 2" signature.

THE FIX: ``vq2_case_c().controller_overrides["body_rate_sign"] = (1,1,1)`` — the per-wire
actuation convention lives in the deploy profile, exactly like ``gyro_sign``. VQ1 keeps the
measured seeker default ``[1,1,-1]`` (flight-proven on THAT wire): the off path is byte-identical.

These tests feed SYNTHETIC gate observations (projected corners -> the seeker's own real PnP)
through the FULL vq2_case_c-configured seeker + controller stack and assert on the WIRE-frame
command (``cmd.body_rate`` is post-body_rate_sign — the exact vector MavlinkClient scales and
emits). Wire-positive yaw = turn RIGHT (T1). No sim required.

[VQ2 A22 gate-2 turn dive, 2026-07-02]
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

# The live-measured VQ2 command->realized body-rate gain (run 20260702_152528: realized yaw rate
# = ~2.1x the wire value; the profile's cmd_rate_scale=0.4 assumes ~2.5). Used only to bound the
# authority check conservatively (we take the SMALLER of the two so the check cannot flatter).
_VQ2_REALIZED_GAIN = 2.1


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
# H2 — the yaw ACTUATION sign, end-to-end (the A22 bug): wire-positive = turn RIGHT
# ===========================================================================
def test_gate_on_the_right_commands_wire_positive_yaw():
    """A gate seen RIGHT of the nose (az ~ +0.46 rad) must yield a wire-POSITIVE yaw-rate command
    (turn right — the plain FRD sign the VQ2 wire realizes at ~2.1x, run 20260702_152528 T1).
    On the pre-fix code (body_rate_sign z=-1 inherited from the VQ1 set) this command comes out
    NEGATED — the drone turns LEFT, az grows, and the loop positive-feedback saturates: the exact
    'never turned toward gate 2' failure. This test FAILS on that code by construction."""
    cmd = _pursue(_vq2_seeker([_gate([10.0, 5.0, -2.5], normal=[1, 0, 0])]))
    wire_yaw = float(cmd.body_rate[2])
    assert wire_yaw > 0.0, (
        f"gate on the RIGHT must command wire-POSITIVE yaw (turn right); got {wire_yaw:+.3f} "
        "— the yaw actuation sign is inverted (the A22 gate-2 no-turn bug)")
    # ADEQUATE MAGNITUDE: az=0.46 rad at kp_att=4 demands ~1.84 -> the visual yaw cap (1.5)
    # saturates; after the slew limiter converges the wire command must sit AT the cap.
    assert wire_yaw >= 1.4, f"steady saturated yaw expected at the 1.5 cap, got {wire_yaw:+.3f}"


def test_gate_on_the_left_commands_wire_negative_yaw():
    """Mirror case: a gate seen LEFT of the nose (az ~ -0.46 rad) must yield a wire-NEGATIVE
    yaw-rate command (turn left)."""
    cmd = _pursue(_vq2_seeker([_gate([10.0, -5.0, -2.5], normal=[1, 0, 0])]))
    wire_yaw = float(cmd.body_rate[2])
    assert wire_yaw < 0.0, (
        f"gate on the LEFT must command wire-NEGATIVE yaw (turn left); got {wire_yaw:+.3f}")
    assert wire_yaw <= -1.4, f"steady saturated yaw expected at the -1.5 cap, got {wire_yaw:+.3f}"


def test_centered_gate_commands_near_zero_yaw():
    """A gate ~dead ahead (az ~ +0.02 — EXACTLY head-on makes the square-PnP two-fold ambiguity
    degenerate, so a hair of offset keeps the pose finite) commands near-ZERO yaw: the loop must be
    QUIET at the fixed point (a sign inversion hides at az~0; the off-axis tests bracket this)."""
    cmd = _pursue(_vq2_seeker([_gate([10.0, 0.2, -2.5], normal=[1, 0, 0])]))
    assert abs(float(cmd.body_rate[2])) < 0.3


def test_saturated_turn_authority_completes_a_90deg_turn_in_time():
    """AUTHORITY: the saturated wire yaw (cap x cmd_rate_scale) at the live-measured ~2.1x
    realization gain must complete a 90-deg gate-to-gate turn in well under 2 s — i.e. the
    cmd_rate_scale=0.4 + the 1.5 rad/s visual cap leave ENOUGH yaw rate once the sign is right.
    (run 20260702_152528: realized/wire gain 2.13; the profile assumes 2.5 — use the smaller.)"""
    cmd = _pursue(_vq2_seeker([_gate([10.0, 5.0, -2.5], normal=[1, 0, 0])]))
    realized_rps = abs(float(cmd.body_rate[2])) * VQ2_CMD_RATE_SCALE * _VQ2_REALIZED_GAIN
    t_90deg = (np.pi / 2) / realized_rps
    assert t_90deg < 2.0, (
        f"saturated yaw realizes only {realized_rps:.2f} rad/s -> {t_90deg:.2f} s for 90 deg; "
        "not enough authority for a gate-to-gate turn")


# ===========================================================================
# H1 — nearest-gate targeting: two gates in frame -> lock + steer to the NEAR one
# ===========================================================================
def test_two_gates_near_right_and_far_centered_targets_the_near_one():
    """NEAR gate off to the RIGHT (8 m, az ~ +0.30) + FAR gate dead-centered (18 m, within the
    22 m acquire range): first-acquisition must lock the NEAR one (its larger apparent corner span
    -> smaller PnP range wins the range+bearing score: 8 + 20*|b| ~ 17 beats 18 + 20*|b| ~ 25)
    and the pursuit must steer TOWARD it (wire-positive yaw). A far-but-centered gate must never
    outrank the next gate to fly."""
    seeker = _vq2_seeker([_gate([18.0, 0.0, -2.5], normal=[1, 0, 0], gate_id=1),   # far, centered
                          _gate([8.0, 2.5, -2.5], normal=[1, 0, 0], gate_id=0)])   # NEAR, right
    cmd = _pursue(seeker)
    assert seeker._track_range_m is not None
    assert abs(seeker._track_range_m - np.hypot(8.0, 2.5)) < 1.5, (
        f"locked range {seeker._track_range_m:.1f} m — expected the NEAR gate (~8.4 m), "
        "not the far centered one (~18 m)")
    assert float(cmd.body_rate[2]) > 0.0, "must steer RIGHT toward the locked NEAR gate"


def test_two_gates_near_left_and_far_beyond_acquire_range_targets_the_near_one():
    """NEAR gate off to the LEFT + a FAR gate at 30 m (beyond max_acquire_range_m=22 -> REJECTED
    outright as 'never the next gate to fly'): the near gate is locked and steered toward (wire-
    NEGATIVE yaw). Covers the reject branch of the A5 far-gate trap alongside the score branch."""
    seeker = _vq2_seeker([_gate([30.0, 0.0, -2.5], normal=[1, 0, 0], gate_id=1),   # far, rejected
                          _gate([8.0, -2.5, -2.5], normal=[1, 0, 0], gate_id=0)])  # NEAR, left
    cmd = _pursue(seeker)
    assert seeker._track_range_m is not None
    assert abs(seeker._track_range_m - np.hypot(8.0, 2.5)) < 1.5, (
        f"locked range {seeker._track_range_m:.1f} m — expected the NEAR gate (~8.4 m)")
    assert float(cmd.body_rate[2]) < 0.0, "must steer LEFT toward the locked NEAR gate"


# ===========================================================================
# The profile seam: VQ2 opts in; VQ1 / the seeker default stay byte-identical
# ===========================================================================
def test_vq2_profile_carries_identity_body_rate_sign():
    """vq2_case_c carries the A22 fix: identity body_rate_sign (the VQ2 wire honors plain FRD;
    only the PROFILE opts in — the per-wire actuation convention lives beside gyro_sign)."""
    sign = np.asarray(vq2_case_c().controller_overrides["body_rate_sign"], float)
    np.testing.assert_allclose(sign, [1.0, 1.0, 1.0])


def test_vq1_and_seeker_default_signs_unchanged():
    """BYTE-IDENTITY guard: the fix must NOT touch the VQ1-measured seeker default ([1,1,-1],
    flight-proven on the VQ1 wire) nor give vq1_case_a any controller overrides."""
    assert vq1_case_a().controller_overrides is None
    np.testing.assert_allclose(SEEKER_SIGNS["body_rate_sign"], [1.0, 1.0, -1.0])
    np.testing.assert_allclose(make_seeker_controller().body_rate_sign, [1.0, 1.0, -1.0])
