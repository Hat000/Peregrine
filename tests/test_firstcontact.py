"""Tests for the first-contact probe helpers (socket-free; feed _handle fakes)."""
from __future__ import annotations

from types import SimpleNamespace

from racer.atmosphere import pressure_to_altitude_m
from racer.firstcontact import (
    backend_summary,
    baro_altitude_m,
    drain_statustexts,
    telemetry_summary,
)
from racer.mavlink_client import MavlinkClient


def _imu(abs_pressure=1013.25):
    m = SimpleNamespace(time_usec=1_000_000, xacc=0.0, yacc=0.0, zacc=-9.8,
                        xmag=1.0, ymag=2.0, zmag=3.0, abs_pressure=abs_pressure)
    m.get_type = lambda: "HIGHRES_IMU"
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
