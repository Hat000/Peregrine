"""Tests for the receive-side clock handling in mavlink_client (the send-side type_mask
helpers are covered in test_contracts). Exercises ``_handle`` directly with fake messages,
so no socket/connection is needed."""
from types import SimpleNamespace

import numpy as np

from racer.mavlink_client import MavlinkClient


def _attitude(time_boot_ms, roll=0.0, pitch=0.0, yaw=0.0):
    m = SimpleNamespace(time_boot_ms=time_boot_ms, roll=roll, pitch=pitch, yaw=yaw,
                        rollspeed=0.0, pitchspeed=0.0, yawspeed=0.0)
    m.get_type = lambda: "ATTITUDE"
    return m


def _imu(time_usec, az=-9.80665):
    m = SimpleNamespace(time_usec=time_usec, xacc=0.0, yacc=0.0, zacc=az,
                        xmag=1.0, ymag=2.0, zmag=3.0, abs_pressure=1013.25)
    m.get_type = lambda: "HIGHRES_IMU"
    return m


def _local_pos(x=1.0, y=2.0, z=3.0, vx=0.1, vy=0.2, vz=0.3, time_boot_ms=999):
    m = SimpleNamespace(time_boot_ms=time_boot_ms, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz)
    m.get_type = lambda: "LOCAL_POSITION_NED"
    return m


def test_sim_time_driven_only_by_highres_imu():
    # [review 2A] ATTITUDE.time_boot_ms and HIGHRES_IMU.time_usec are different epochs. Only
    # HIGHRES_IMU may drive sim_time_ns; ATTITUDE must not (else the timeline oscillates).
    c = MavlinkClient()
    c._handle(_imu(time_usec=1_000_000))                  # 1.0 s in usec -> 1e9 ns
    assert c.state.sim_time_ns == 1_000_000_000
    # An interleaved ATTITUDE on a wildly different boot-ms clock must not change the stamp,
    # but must still update orientation.
    c._handle(_attitude(time_boot_ms=50, roll=0.1, pitch=0.2, yaw=0.3))
    assert c.state.sim_time_ns == 1_000_000_000
    assert (c.state.roll, c.state.pitch, c.state.yaw) == (0.1, 0.2, 0.3)
    # The next IMU advances the clock.
    c._handle(_imu(time_usec=1_033_000))
    assert c.state.sim_time_ns == 1_033_000_000


def test_sim_time_monotonic_under_interleaving():
    # Interleave the two streams as the network would; sim_time_ns must be non-decreasing
    # with no negative/huge jumps (the failure mode that diverges a dt-based estimator).
    c = MavlinkClient()
    boot_ms = usec = 0
    seq = []
    for _ in range(50):
        usec += 33_000          # ~30 Hz IMU, monotonic usec
        c._handle(_imu(time_usec=usec))
        seq.append(c.state.sim_time_ns)
        boot_ms += 5            # ATTITUDE on its own unrelated, smaller clock
        c._handle(_attitude(time_boot_ms=boot_ms))
        seq.append(c.state.sim_time_ns)
    assert seq == sorted(seq)                    # non-decreasing despite interleaving
    assert (np.diff(seq) >= 0).all()             # no negative dt


def test_local_position_does_not_drive_clock():
    # LOCAL_POSITION_NED.time_boot_ms is yet another epoch; capture pos/vel but not the clock.
    c = MavlinkClient()
    c._handle(_imu(time_usec=2_000_000))
    c._handle(_local_pos())
    assert c.state.sim_time_ns == 2_000_000_000
    np.testing.assert_array_equal(c.state.position_ned, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(c.state.velocity_ned, [0.1, 0.2, 0.3])


def test_highres_imu_populates_sensors():
    c = MavlinkClient()
    c._handle(_imu(time_usec=500_000, az=-9.0))
    np.testing.assert_array_equal(c.state.accel_body, [0.0, 0.0, -9.0])
    np.testing.assert_array_equal(c.state.mag_body, [1.0, 2.0, 3.0])
    assert c.state.baro_pressure_hpa == 1013.25
