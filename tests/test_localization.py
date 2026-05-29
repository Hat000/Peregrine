import numpy as np
import pytest

from racer.contracts import Gate, GateObservation, GatePose
from racer.frames import R_camera_from_body, R_world_from_body
from racer.localization import apply_gate_pose_update, gate_pose_to_world_position
from racer.state_estimator import LinearKF
from racer.vision.gate_pose import estimate_gate_pose, project_gate_corners


def _world_setup():
    """Drone level facing north at [0,0,-2]; a north-facing gate 6 m ahead, 1 m above.

    Returns the world truth plus the *true* camera-relative gate pose (R_cam_gate, t).
    """
    p_drone = np.array([0.0, 0.0, -2.0])
    R_wb = R_world_from_body(0.0, 0.0, 0.0)  # level, yaw 0 = facing north
    gate_pos = np.array([6.0, 0.0, -3.0])
    # Gate frame columns in world: X=right->east, Y=down->down, Z=downrange->north.
    R_world_gate = np.column_stack([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).astype(float)
    gate = Gate(gate_id=2, position_ned=gate_pos, R_world_gate=R_world_gate)

    R_wc = R_wb @ R_camera_from_body().T          # camera -> world
    t_cam_gate = R_wc.T @ (gate_pos - p_drone)    # gate origin in camera frame
    R_cam_gate = R_wc.T @ R_world_gate            # gate orientation in camera frame
    return p_drone, R_wb, gate, R_cam_gate, t_cam_gate


def test_gate_normal_is_through_direction():
    _, _, gate, _, _ = _world_setup()
    np.testing.assert_allclose(gate.normal_ned, [1.0, 0.0, 0.0])  # north = downrange


def test_glue_recovers_drone_position_exactly():
    p_drone, R_wb, gate, R_cg, t_cg = _world_setup()
    gp = GatePose(frame_id=0, sim_time_ns=0, R_cam_gate=R_cg, t_cam_gate=t_cg, reproj_error_px=0.0)
    pos, _ = gate_pose_to_world_position(gp, gate, R_wb)
    np.testing.assert_allclose(pos, p_drone, atol=1e-9)


def test_glue_covariance_is_rotated_translation_block():
    p_drone, R_wb, gate, R_cg, t_cg = _world_setup()
    sigma_tt = np.diag([0.01, 0.02, 0.05])
    cov6 = np.zeros((6, 6))
    cov6[:3, :3] = sigma_tt
    gp = GatePose(frame_id=0, sim_time_ns=0, R_cam_gate=R_cg, t_cam_gate=t_cg,
                  reproj_error_px=0.0, covariance=cov6)
    _, cov = gate_pose_to_world_position(gp, gate, R_wb)
    R_wc = R_wb @ R_camera_from_body().T
    np.testing.assert_allclose(cov, R_wc @ sigma_tt @ R_wc.T, atol=1e-12)


def test_glue_default_covariance_when_none():
    p_drone, R_wb, gate, R_cg, t_cg = _world_setup()
    gp = GatePose(frame_id=0, sim_time_ns=0, R_cam_gate=R_cg, t_cam_gate=t_cg, reproj_error_px=0.0)
    _, cov = gate_pose_to_world_position(gp, gate, R_wb, default_position_std=0.3)
    np.testing.assert_allclose(cov, 0.09 * np.eye(3))


def test_end_to_end_project_pnp_glue_recovers_position():
    # World -> projected corners -> PnP -> glue should round-trip the drone position.
    p_drone, R_wb, gate, R_cg, t_cg = _world_setup()
    corners = project_gate_corners(R_cg, t_cg)
    obs = GateObservation(frame_id=1, sim_time_ns=0, corners_px=corners, gate_id=gate.gate_id)
    prior = GatePose(frame_id=0, sim_time_ns=0, R_cam_gate=R_cg, t_cam_gate=t_cg, reproj_error_px=0.0)
    gp = estimate_gate_pose(obs, prior=prior)
    assert gp is not None
    pos, _ = gate_pose_to_world_position(gp, gate, R_wb)
    np.testing.assert_allclose(pos, p_drone, atol=1e-4)


def test_end_to_end_with_pixel_noise():
    p_drone, R_wb, gate, R_cg, t_cg = _world_setup()
    rng = np.random.default_rng(0)
    corners = project_gate_corners(R_cg, t_cg) + rng.normal(0, 1.0, (4, 2))
    obs = GateObservation(frame_id=1, sim_time_ns=0, corners_px=corners, gate_id=gate.gate_id)
    prior = GatePose(frame_id=0, sim_time_ns=0, R_cam_gate=R_cg, t_cam_gate=t_cg, reproj_error_px=0.0)
    gp = estimate_gate_pose(obs, prior=prior, compute_covariance=True, corner_sigma_px=1.0,
                            n_samples=40, rng=rng)
    pos, cov = gate_pose_to_world_position(gp, gate, R_wb)
    assert np.linalg.norm(pos - p_drone) < 0.15          # ~1px noise at 6 m -> cm-level
    assert cov.shape == (3, 3)
    assert np.all(np.linalg.eigvalsh(cov) >= -1e-9)      # valid covariance


def test_apply_update_moves_kf_toward_truth():
    p_drone, R_wb, gate, R_cg, t_cg = _world_setup()
    kf = LinearKF.initialize(np.zeros(3), np.zeros(3), pos_std=3.0, vel_std=1.0)
    before = np.linalg.norm(kf.position - p_drone)
    sigma = np.zeros((6, 6))
    sigma[:3, :3] = np.diag([0.01, 0.01, 0.01])
    gp = GatePose(frame_id=0, sim_time_ns=0, R_cam_gate=R_cg, t_cam_gate=t_cg,
                  reproj_error_px=0.0, covariance=sigma)
    apply_gate_pose_update(kf, gp, gate, R_wb)
    assert np.linalg.norm(kf.position - p_drone) < before
    np.testing.assert_allclose(kf.position, p_drone, atol=0.1)
