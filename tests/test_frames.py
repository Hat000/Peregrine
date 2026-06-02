import numpy as np
from scipy.spatial.transform import Rotation

from racer.frames import (
    CAMERA_PITCH_RAD,
    R_camera_from_body,
    R_world_from_body,
    euler_from_quat_wxyz,
    project_camera_point,
    world_point_in_camera,
)


def test_euler_from_quat_wxyz_inverts_R_world_from_body():
    # euler_from_quat_wxyz must be the exact inverse of R_world_from_body's 'ZYX' convention,
    # so the controller's commanded attitude (_euler_to_wxyz) and the ODOMETRY feedback share
    # ONE convention. Round-trips for several attitudes incl. a nose-down (negative pitch).
    for roll, pitch, yaw in [(0.1, -0.3, 1.2), (-0.2, 0.4, -2.0), (0.0, np.deg2rad(-17.8), 0.0)]:
        x, y, z, w = Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_quat()
        r, p, yw = euler_from_quat_wxyz([w, x, y, z])
        assert abs(r - roll) < 1e-9
        assert abs(p - pitch) < 1e-9
        assert abs(yw - yaw) < 1e-9
        np.testing.assert_allclose(
            R_world_from_body(r, p, yw), Rotation.from_quat([x, y, z, w]).as_matrix(), atol=1e-9
        )


def test_euler_from_quat_wxyz_degenerate_returns_zeros():
    # A zero-norm quaternion (an unpopulated field) must return zeros, not raise.
    assert euler_from_quat_wxyz([0.0, 0.0, 0.0, 0.0]) == (0.0, 0.0, 0.0)


def test_body_forward_projects_below_principal_point():
    # The camera tilts +20deg up, so a point straight ahead in body frame
    # appears below the principal point (v > cy).
    R = R_camera_from_body()
    p_cam = R @ np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(
        p_cam,
        [0.0, np.sin(CAMERA_PITCH_RAD), np.cos(CAMERA_PITCH_RAD)],
        atol=1e-9,
    )
    uv = project_camera_point(p_cam)
    assert uv is not None
    u, v = uv
    np.testing.assert_allclose(u, 320.0)
    assert v > 180.0


def test_body_right_maps_to_camera_x():
    # 1m to body right -> camera X axis (image plane right) with zero depth.
    R = R_camera_from_body()
    p_cam = R @ np.array([0.0, 1.0, 0.0])
    np.testing.assert_allclose(p_cam, [1.0, 0.0, 0.0], atol=1e-9)
    assert project_camera_point(p_cam) is None  # behind/on the image plane


def test_optical_axis_when_level():
    # Drone level. Optical axis (camera +Z) expressed in world should be
    # [cos(20), 0, -sin(20)]: north and slightly up.
    R_wb = R_world_from_body(0.0, 0.0, 0.0)
    R_cb = R_camera_from_body()
    cam_z_in_world = R_wb @ (R_cb.T @ np.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(
        cam_z_in_world,
        [np.cos(CAMERA_PITCH_RAD), 0.0, -np.sin(CAMERA_PITCH_RAD)],
        atol=1e-9,
    )


def test_world_point_in_camera_basic():
    # Drone at origin level. A point 5m north is in front of the camera,
    # centered horizontally, biased low in the image (camera tilts up).
    R_wb = R_world_from_body(0.0, 0.0, 0.0)
    p_cam = world_point_in_camera(np.array([5.0, 0.0, 0.0]), R_wb, np.zeros(3))
    assert p_cam[2] > 0
    np.testing.assert_allclose(p_cam[0], 0.0, atol=1e-9)


def test_yaw_rotates_world_to_body():
    # Drone yawed 90deg (facing east). A point 5m north of origin should be
    # to the LEFT (-Y) of the drone in body frame.
    R_wb = R_world_from_body(0.0, 0.0, np.pi / 2)
    p_world = np.array([5.0, 0.0, 0.0])
    p_body = R_wb.T @ p_world
    np.testing.assert_allclose(p_body, [0.0, -5.0, 0.0], atol=1e-9)
