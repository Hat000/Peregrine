"""Tests for the LIVE-WIRE gyro-sign correction (VQ2 HIGHRES_IMU pitch-axis convention flip).

The live VQ2 sim (build 1.0.3379, 2026-06-29) reports the HIGHRES_IMU gyro PITCH axis (y)
INVERTED vs the code's FRD assumption (3/3 A9 runs: commanded pitch-rate +1.5 reads -1.4 on the
raw gyro while the nose physically pitches UP). ``MavlinkClient.gyro_sign`` corrects it elementwise
at the wire, BEFORE the AHRS. DEFAULT (1,1,1) == identity, so VQ1 + offline are byte-identical.

This correction is a LIVE-WIRE fact the offline suite CANNOT validate -- synthetic IMU is
self-consistent in the code's FRD convention, so these tests only pin the SEAM (default identity,
configured flip routes to the right axis, profile wiring). The real proof is the next live flight.
"""
from types import SimpleNamespace

import numpy as np

from racer.mavlink_client import MavlinkClient
from racer.deploy_profile import get_profile


def _imu(time_usec=500_000, az=-9.80665, gx=0.0, gy=0.0, gz=0.0):
    # Real HIGHRES_IMU always carries xgyro/ygyro/zgyro; mirror test_mavlink_client's fake.
    m = SimpleNamespace(time_usec=time_usec, xacc=0.0, yacc=0.0, zacc=az,
                        xgyro=gx, ygyro=gy, zgyro=gz,
                        xmag=1.0, ymag=2.0, zmag=3.0, abs_pressure=1013.25)
    m.get_type = lambda: "HIGHRES_IMU"
    return m


def test_default_gyro_sign_is_byte_identical():
    """gyro_sign defaults to (1,1,1): gyro_body equals the raw HIGHRES_IMU gyro, no change.
    Pins byte-identity of the VQ1 / offline path against a message fake."""
    c = MavlinkClient()
    assert tuple(c.gyro_sign) == (1.0, 1.0, 1.0)
    c._handle(_imu(gx=0.1, gy=-0.2, gz=0.3))
    np.testing.assert_array_equal(c.state.gyro_body, [0.1, -0.2, 0.3])


def test_pitch_flip_negates_only_y():
    """gyro_sign=(1,-1,1) negates gyro_body[1] (pitch) and leaves [0]/[2] untouched."""
    c = MavlinkClient(gyro_sign=(1.0, -1.0, 1.0))
    c._handle(_imu(gx=0.1, gy=-0.2, gz=0.3))
    np.testing.assert_array_equal(c.state.gyro_body, [0.1, 0.2, 0.3])


def test_gyro_sign_accepts_array():
    """gyro_sign may be passed as an array/tuple; stored as float64 and applied elementwise."""
    c = MavlinkClient(gyro_sign=np.array([-1.0, -1.0, -1.0]))
    c._handle(_imu(gx=1.0, gy=2.0, gz=3.0))
    np.testing.assert_array_equal(c.state.gyro_body, [-1.0, -2.0, -3.0])


def test_gyro_sign_none_when_message_omits_gyro():
    """Defensive: a HIGHRES_IMU variant/fake without xgyro -> gyro_body None, regardless of sign."""
    m = SimpleNamespace(time_usec=1000, xacc=0.0, yacc=0.0, zacc=-9.8,
                        xmag=1.0, ymag=2.0, zmag=3.0, abs_pressure=1013.25)
    m.get_type = lambda: "HIGHRES_IMU"
    c = MavlinkClient(gyro_sign=(1.0, -1.0, 1.0))
    c._handle(m)
    assert c.state.gyro_body is None


def test_profile_gyro_sign_values():
    """vq2_case_c carries the probe-confirmed FULL sign negation (roll+pitch+yaw all
    inverted on the live VQ2 wire, 2026-06-30); vq1_case_a stays identity."""
    assert tuple(get_profile("vq2_case_c").gyro_sign) == (-1.0, -1.0, -1.0)
    assert tuple(get_profile("vq1_case_a").gyro_sign) == (1.0, 1.0, 1.0)
