"""Obs SIGN faithfulness — the gate-relative pos_g slot is +L = (gate_pos - pos), NOT -L.

This pins the project's hardest-won bug class (4 prior frame/convention bugs): the policy obs
slot ``pos_g = R_w2g @ (gate_pos - pos) = R_w2g @ (+L_seen)``, where the lever
``L = R_world_camera @ t_cam_gate`` (world NED) is the drone->gate offset to the SEEN opening.
The d1/d2 design spec TEXT ("the estimator delivers -L") is WRONG for the obs slot -- delivering
-L is a FULL sign flip ~24 m at gate range (BLUEPRINT §0.2). When the C2 estimator is built it
MUST hand the obs builder +L_seen; this test is the contract that catches a -L regression.

Ported from handoff/ultracode-gate-relative-pipeline-design-2026-06-13/v_obs_adversarial_check.py
(check_end_to_end_caseC), driving the REAL localization.gate_pose_to_world_position + the REAL
fly_rl.obs_from_zup through the NED<->Z-up conversion -- not the d1 "0.0 unification" tautology
(which computed R_w2g@(gate-pos) twice in one frame and never exercised localization).

Imports fly_rl (the shipped obs builder), which pulls torch -- skipped if torch is absent.
[P0-CASEC-FOUNDATION 2026-06-13]
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("torch")   # fly_rl imports torch at module top

_RL = Path(__file__).resolve().parents[1] / "rl"
if str(_RL) not in sys.path:
    sys.path.insert(0, str(_RL))

from fly_rl import (  # noqa: E402  the shipped obs builder under contract
    _FLIP,
    _GATE_POS_ZUP,
    _GATE_YAW_ZUP,
    _gate_rotmat_w2g,
    make_gate_map,
    obs_from_zup,
)
from racer.contracts import Gate, GatePose  # noqa: E402
from racer.frames import R_camera_from_body, R_world_from_body  # noqa: E402
from racer.localization import gate_pose_to_world_position  # noqa: E402

# Mission gate (BLUEPRINT §0.2 / G1): +L matches the obs builder to <= this; -L is a ~24 m flip.
_PLUS_L_TOL = 1e-5
_FLIP_BREAK_M = 10.0     # the -L negative control must miss by at least this (observed ~24 m)


def _rand_attitude_zup(rng):
    rpy = rng.normal(0, 0.3, 3)
    return Rotation.from_euler("ZYX", [rpy[2], rpy[1], rpy[0]]).as_matrix()


def _end_to_end_pos_g_residuals(n=1500, seed=4242):
    """For n noiseless gate-4 sightings, return (max|obs - R_w2g@(+L_zup)|,
    max|obs - R_w2g@(-L_zup)|) driving the REAL NED localization + Z-up obs builder."""
    rng = np.random.default_rng(seed)
    tg = 4                                          # gate 4 = the binding gate
    gate_true_zup = _GATE_POS_ZUP[tg].copy()
    gate_true_ned = gate_true_zup * _FLIP           # Z-up -> NED (the flip is its own inverse)
    yaw_gate_zup = float(_GATE_YAW_ZUP[tg])
    gm = make_gate_map(_GATE_POS_ZUP, _GATE_YAW_ZUP)
    R_w2g_zup = _gate_rotmat_w2g(yaw_gate_zup)
    max_plus, max_minus = 0.0, 0.0
    for _ in range(n):
        drone_true_ned = gate_true_ned - np.array(
            [rng.uniform(4, 12), rng.uniform(-1, 1), rng.uniform(-1, 1)])
        roll, pitch, yaw = rng.normal(0, 0.15), rng.normal(-0.3, 0.1), rng.normal(np.pi, 0.1)
        R_wb = R_world_from_body(roll, pitch, yaw)
        R_wc = R_wb @ R_camera_from_body().T
        L_true_ned = gate_true_ned - drone_true_ned          # gate - drone, NED  (== +L)
        t_cam_gate = R_wc.T @ L_true_ned                     # noiseless PnP to the SEEN opening
        gp = GatePose(frame_id=0, sim_time_ns=0, R_cam_gate=np.eye(3), t_cam_gate=t_cam_gate,
                      reproj_error_px=0.0, gate_id=tg, covariance=None, n_corners=4)
        gate_obj = Gate(gate_id=tg, position_ned=gate_true_ned, R_world_gate=np.eye(3))
        # REAL localization: p_abs (NED) = gate.position_ned - L
        p_abs_ned, _ = gate_pose_to_world_position(gp, gate_obj, R_wb,
                                                   attitude_noise_std=0.0, fix_cov_floor_std=0.0)
        # (i) the obs as the shipped builder forms it (pose -> Z-up, gate_map yaw path)
        p_zup = p_abs_ned * _FLIP
        obs = obs_from_zup(p_zup, np.zeros(3), _rand_attitude_zup(rng), np.zeros(3), tg, 0.0,
                           virtual_flip=False, gate_map=gm)
        pos_g_obs = obs[:3].astype(np.float64)
        L_seen_zup = (R_wc @ t_cam_gate) * _FLIP             # +L in NED -> Z-up
        max_plus = max(max_plus, float(np.max(np.abs(pos_g_obs - R_w2g_zup @ (+L_seen_zup)))))
        max_minus = max(max_minus, float(np.max(np.abs(pos_g_obs - R_w2g_zup @ (-L_seen_zup)))))
    return max_plus, max_minus


def test_obs_pos_g_is_plus_L_seen_and_minus_L_breaks():
    max_plus, max_minus = _end_to_end_pos_g_residuals()
    # +L identity: the shipped obs builder's pos_g slot IS R_w2g @ (gate_seen - pos).
    assert max_plus <= _PLUS_L_TOL, (
        f"obs pos_g diverges from R_w2g @ (+L_seen) by {max_plus:.3e} m (> {_PLUS_L_TOL:.0e}); "
        f"the +L gate-relative convention is broken.")
    # -L negative control: the spec-text sign MUST be a gross miss (~24 m), not a near-match.
    assert max_minus >= _FLIP_BREAK_M, (
        f"the -L sign-flip negative control did not break (only {max_minus:.3e} m); the sign test "
        f"is not actually discriminating the +L vs -L convention.")
