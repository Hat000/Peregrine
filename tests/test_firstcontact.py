"""Tests for the first-contact probe helpers (socket-free; feed _handle fakes)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from racer.atmosphere import pressure_to_altitude_m
from racer.firstcontact import (
    attitude_gravity_residual,
    backend_summary,
    baro_altitude_m,
    drain_statustexts,
    gravity_tilt,
    telemetry_summary,
)
from racer.mavlink_client import MavlinkClient

_G = 9.80665


def _imu(abs_pressure=1013.25):
    m = SimpleNamespace(time_usec=1_000_000, xacc=0.0, yacc=0.0, zacc=-9.8,
                        xmag=1.0, ymag=2.0, zmag=3.0, abs_pressure=abs_pressure)
    m.get_type = lambda: "HIGHRES_IMU"
    return m


def _att(roll=0.0, pitch=0.0, yaw=0.0):
    m = SimpleNamespace(roll=roll, pitch=pitch, yaw=yaw,
                        rollspeed=0.0, pitchspeed=0.0, yawspeed=0.0, time_boot_ms=1)
    m.get_type = lambda: "ATTITUDE"
    return m


def _statustext(text):
    m = SimpleNamespace(severity=6, text=text)
    m.get_type = lambda: "STATUSTEXT"
    return m


def test_baro_altitude_uses_isa_inverse():
    c = MavlinkClient()
    assert baro_altitude_m(c) is None                  # no baro yet
    c._handle(_imu(abs_pressure=950.0))
    assert baro_altitude_m(c) == pressure_to_altitude_m(950.0)


def test_telemetry_summary_reports_hedge_presence():
    c = MavlinkClient()
    c._handle(_imu())                                  # baro + mag present; no pos/vel
    s = telemetry_summary(c)
    assert "baro=" in s and "pos=NO" in s and "vel=NO" in s and "mag=yes" in s


def test_drain_statustexts_prints_and_counts(capsys):
    c = MavlinkClient()
    c._handle(_statustext("hello"))
    assert drain_statustexts(c, 0) == 1
    assert "hello" in capsys.readouterr().out
    assert drain_statustexts(c, 1) == 1                # nothing new past index 1


def test_backend_summary_safe_without_heartbeat():
    c = MavlinkClient()
    s = backend_summary(c)
    assert "autopilot=" in s and "type=" in s          # '?' placeholders, no crash


# -- attitude-bias observability (red-team CRIT-2) -------------------------
def test_gravity_tilt_recovers_pitch_and_rejects_acceleration():
    assert gravity_tilt(np.array([0.0, 0.0, -_G])) == (0.0, 0.0)         # level
    th = 0.2
    roll, pitch = gravity_tilt(np.array([_G * np.sin(th), 0.0, -_G * np.cos(th)]))
    assert abs(pitch - th) < 1e-9 and abs(roll) < 1e-9                   # pitched up
    assert gravity_tilt(np.array([0.0, 0.0, -20.0])) is None             # |a| != 1g -> accelerating
    assert gravity_tilt(np.zeros(3)) is None


def test_attitude_gravity_residual_zero_when_consistent():
    c = MavlinkClient()
    c._handle(_att(roll=0.0, pitch=0.0))
    c._handle(_imu())                                   # zacc=-9.8, level -> gravity tilt 0
    res = attitude_gravity_residual(c)
    assert res is not None and abs(res[0]) < 1e-3 and abs(res[1]) < 1e-3


def test_attitude_gravity_residual_flags_static_bias():
    # Sim REPORTS level but the accelerometer says pitched 2 deg -> a 2 deg given-attitude bias.
    c = MavlinkClient()
    c._handle(_att(roll=0.0, pitch=0.0))
    th = np.deg2rad(2.0)
    imu = _imu()
    imu.xacc, imu.zacc = _G * np.sin(th), -_G * np.cos(th)
    c._handle(imu)
    res = attitude_gravity_residual(c)
    assert res is not None
    assert abs(np.rad2deg(res[1]) + 2.0) < 0.1         # residual pitch = given(0) - gravity(+2) = -2 deg
    assert abs(np.rad2deg(res[0])) < 0.1               # roll unaffected
