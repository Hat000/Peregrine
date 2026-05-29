import dataclasses

import numpy as np
import pytest

from racer.contracts import (
    ControlCommand,
    ControlMode,
    DroneState,
    Frame,
    GateObservation,
    GatePose,
    NavState,
)
from racer.mavlink_client import (
    _POS_IGNORE_PX,
    _POS_IGNORE_VX,
    _POS_IGNORE_YAW,
    _POS_IGNORE_YAW_RATE,
    _pos_type_mask,
)


def _obs() -> GateObservation:
    return GateObservation(
        frame_id=1,
        sim_time_ns=1000,
        corners_px=np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64),
    )


def test_frame_is_immutable():
    f = Frame(frame_id=1, sim_time_ns=2, image_bgr=np.zeros((360, 640, 3), np.uint8))
    with pytest.raises(dataclasses.FrozenInstanceError):
        f.frame_id = 5  # type: ignore[misc]


def test_dronestate_defaults_are_safe():
    s = DroneState()
    # The hedge fields must default to None (we derive them) and vectors to zeros.
    assert s.position_ned is None
    assert s.velocity_ned is None
    assert s.mag_body is None
    assert s.baro_pressure_hpa is None
    assert s.armed is False
    np.testing.assert_array_equal(s.accel_body, np.zeros(3))
    # Distinct instances must not share the same default array object.
    assert DroneState().accel_body is not s.accel_body


def test_gateobservation_corner_contract():
    # 4 corners with no ids -> OK (assumed canonical [0,1,2,3]).
    GateObservation(frame_id=1, sim_time_ns=1, corners_px=np.zeros((4, 2)))
    # 3 corners are allowed (clipped gate) BUT need corner_ids to name which canonical
    # corners they are -- order alone is ambiguous for a partial set.
    with pytest.raises(AssertionError):
        GateObservation(frame_id=1, sim_time_ns=1, corners_px=np.zeros((3, 2)))
    GateObservation(frame_id=1, sim_time_ns=1, corners_px=np.zeros((3, 2)),
                    corner_ids=np.array([0, 1, 3]))
    # Fewer than 3 or more than 4 corners are rejected.
    with pytest.raises(AssertionError):
        GateObservation(frame_id=1, sim_time_ns=1, corners_px=np.zeros((2, 2)),
                        corner_ids=np.array([0, 1]))
    with pytest.raises(AssertionError):
        GateObservation(frame_id=1, sim_time_ns=1, corners_px=np.zeros((5, 2)))
    # corner_ids must be distinct indices within 0..3, and match the corner count.
    with pytest.raises(AssertionError):
        GateObservation(frame_id=1, sim_time_ns=1, corners_px=np.zeros((3, 2)),
                        corner_ids=np.array([0, 1, 7]))
    with pytest.raises(AssertionError):
        GateObservation(frame_id=1, sim_time_ns=1, corners_px=np.zeros((3, 2)),
                        corner_ids=np.array([0, 1, 1]))
    with pytest.raises(AssertionError):
        GateObservation(frame_id=1, sim_time_ns=1, corners_px=np.zeros((4, 2)),
                        corner_ids=np.array([0, 1, 2]))


def test_gatepose_geometry():
    # Gate centre 5 m straight ahead of the camera, no rotation.
    pose = GatePose(
        frame_id=1,
        sim_time_ns=1,
        R_cam_gate=np.eye(3),
        t_cam_gate=np.array([0.0, 0.0, 5.0]),
        reproj_error_px=0.3,
    )
    assert pose.range_m == pytest.approx(5.0)
    # camera position in the gate frame = -R^T t
    np.testing.assert_allclose(pose.cam_in_gate, [0.0, 0.0, -5.0], atol=1e-12)


def test_gatepose_rejects_bad_shapes():
    with pytest.raises(AssertionError):
        GatePose(
            frame_id=1, sim_time_ns=1,
            R_cam_gate=np.eye(2), t_cam_gate=np.zeros(3), reproj_error_px=0.0,
        )


def test_navstate_minimal():
    n = NavState(sim_time_ns=10)
    assert n.time_since_vision_update_s == float("inf")
    np.testing.assert_array_equal(n.position_ned, np.zeros(3))


def test_controlcommand_modes():
    c = ControlCommand(mode=ControlMode.BODY_RATE, body_rate=np.zeros(3), thrust=0.5)
    assert c.mode is ControlMode.BODY_RATE
    assert c.position_ned is None


def test_pos_type_mask_position_mode():
    # Position + yaw provided; velocity/accel/yaw_rate omitted.
    m = _pos_type_mask(np.zeros(3), None, None, 0.0, None)
    assert not (m & _POS_IGNORE_PX)          # position is used
    assert m & _POS_IGNORE_VX                # velocity ignored
    assert not (m & _POS_IGNORE_YAW)         # yaw is used
    assert m & _POS_IGNORE_YAW_RATE          # yaw-rate ignored


def test_pos_type_mask_velocity_mode():
    m = _pos_type_mask(None, np.zeros(3), None, None, None)
    assert m & _POS_IGNORE_PX                # position ignored
    assert not (m & _POS_IGNORE_VX)          # velocity is used
