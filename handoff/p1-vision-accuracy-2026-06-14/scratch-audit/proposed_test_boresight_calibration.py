"""PROPOSED regression test for the boresight calibration. NOT in tests/ this session (no src
edits); lands in tests/ WITH the frames.py patch (proposed_frames_boresight.patch).

Pins, in order of importance:
  1. DEFAULT byte-identity: BORESIGHT_*_RAD == 0.0 and R_camera_from_body() is bit-identical to the
     20deg-only mount  => the stack stays VQ1 byte-identical until calibrated.   (needs the src patch)
  2. SENSITIVITY SIGN: a physical camera pitched up by +eps vs the decode model yields
     gate_vert ~ -range*tan(eps)  (the sign the calibration estimator inverts).   (runs today)
  3. CORRECTION: render and decode at the SAME (20+eps) mount round-trips < 1 mm.  (runs today)
  4. ESTIMATOR: solve_boresight_pitch_rad recovers an injected eps and drives the post-correction
     vertical residual <= 0.05 m.                                                 (runs today)

Run standalone:  py -3.13 -m pytest handoff/.../scratch-audit/proposed_test_boresight_calibration.py
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from racer import frames as F
from racer.contracts import Gate, GateObservation, GatePose
from racer.localization import gate_pose_to_world_position
from racer.vision.gate_pose import GATE_INNER_SIZE_M, estimate_gate_pose, project_gate_corners

DEG = np.pi / 180.0


def _mount(pitch_deg: float) -> np.ndarray:
    return F._R_CAMERA_FROM_TILTED_BODY @ Rotation.from_euler("Y", -np.deg2rad(pitch_deg)).as_matrix()


def _head_on_gate(range_m: float) -> Gate:
    return Gate(gate_id=0, position_ned=np.array([range_m, 0.0, 0.0]),
                R_world_gate=np.column_stack([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).astype(float),
                inner_size_m=GATE_INNER_SIZE_M)


def _fix(gate, drone_pos, R_wb, render_pitch, decode_pitch):
    """Render at render_pitch, decode lever at decode_pitch; return gate-vertical (down) error (m)."""
    R_cw_r = (R_wb @ _mount(render_pitch).T).T
    t_cam = R_cw_r @ (gate.position_ned - drone_pos)
    corners = project_gate_corners(R_cw_r @ gate.R_world_gate, t_cam, gate.inner_size_m)
    R_cw_d = (R_wb @ _mount(decode_pitch).T).T
    prior = GatePose(0, 0, R_cw_d @ gate.R_world_gate, R_cw_d @ (gate.position_ned - drone_pos), 0.0)
    gp = estimate_gate_pose(GateObservation(0, 0, corners_px=corners, corner_confidence=np.ones(4)),
                            prior=prior, weighted_refine=False)
    lever = (R_wb @ _mount(decode_pitch).T) @ gp.t_cam_gate
    pos_rec = gate.position_ned - lever
    return float((pos_rec - drone_pos) @ gate.R_world_gate[:, 1])


def solve_boresight_pitch_rad(residual_m: float, range_m: float) -> float:
    return float(-np.arctan(residual_m / range_m))


# 1 -------------------------------------------------------------------------- byte-identity (needs patch)
@pytest.mark.skipif(not hasattr(F, "BORESIGHT_PITCH_RAD"),
                    reason="frames.BORESIGHT_PITCH_RAD not present until the boresight patch lands")
def test_boresight_default_zero_is_byte_identical():
    assert F.BORESIGHT_PITCH_RAD == 0.0
    assert getattr(F, "BORESIGHT_YAW_RAD", 0.0) == 0.0
    expected = F._R_CAMERA_FROM_TILTED_BODY @ Rotation.from_euler("Y", -F.CAMERA_PITCH_RAD).as_matrix()
    assert np.array_equal(F.R_camera_from_body(), expected)   # bit-identical, not just allclose


# 2 -------------------------------------------------------------------------- sensitivity sign
@pytest.mark.parametrize("eps_deg", [-0.56, -0.25, 0.25, 0.56])
def test_boresight_sensitivity_sign_and_magnitude(eps_deg):
    rng_m = 22.0
    vd = _fix(_head_on_gate(rng_m), np.zeros(3), np.eye(3), 20.0 + eps_deg, 20.0)
    assert vd == pytest.approx(-rng_m * np.tan(eps_deg * DEG), abs=2e-3)   # gate_vert = -range*tan(eps)


# 3 -------------------------------------------------------------------------- correction round-trips
@pytest.mark.parametrize("eps_deg", [0.0, 0.56, 1.2])
@pytest.mark.parametrize("rng_m", [12.0, 22.0, 33.0])
def test_matched_mount_round_trips(eps_deg, rng_m):
    vd = _fix(_head_on_gate(rng_m), np.zeros(3), np.eye(3), 20.0 + eps_deg, 20.0 + eps_deg)
    assert abs(vd) < 1e-3


# 4 -------------------------------------------------------------------------- estimator recovers eps
def test_calibration_estimator_recovers_injected_boresight():
    eps_true = 0.56
    # measure at decode=20 (uncalibrated), solve, then re-measure at decode=20+eps_est
    resid = _fix(_head_on_gate(22.0), np.zeros(3), np.eye(3), 20.0 + eps_true, 20.0)
    eps_est = np.rad2deg(solve_boresight_pitch_rad(resid, 22.0))
    assert eps_est == pytest.approx(eps_true, abs=0.05)
    post = _fix(_head_on_gate(22.0), np.zeros(3), np.eye(3), 20.0 + eps_true, 20.0 + eps_est)
    assert abs(post) <= 0.05      # acceptance: post-calibration mean |vert residual| <= 5 cm
