"""Shared helpers for the first-contact probe scripts.

These read ONLY the spec MAVLink interface via :class:`MavlinkClient` (VADR-TS-002 sec 4).
Nothing here knows about Elodin -- Elodin is a separate Apache-2.0 *surrogate* with an
RC-callback API and is NOT authoritative for the scoring simulator. Keep it that way: the
probes characterise the REAL sim's MAVLink behaviour, never Elodin's.
"""
from __future__ import annotations

import time

from racer.atmosphere import pressure_to_altitude_m
from racer.mavlink_client import MavlinkClient


def _enum_name(enum: str, value: int | None) -> str:
    if value is None:
        return "?"
    try:
        from pymavlink import mavutil

        return mavutil.mavlink.enums[enum][value].name
    except Exception:
        return str(value)


def backend_summary(client: MavlinkClient) -> str:
    """One line describing the backend we're actually talking to (from HEARTBEAT)."""
    return (
        f"autopilot={_enum_name('MAV_AUTOPILOT', client.autopilot)} "
        f"type={_enum_name('MAV_TYPE', client.vehicle_type)} "
        f"custom_mode={client.custom_mode}"
    )


def telemetry_summary(client: MavlinkClient) -> str:
    """One line of the current telemetry snapshot + hedge-field presence (R1)."""
    s = client.state
    pos = "yes" if s.position_ned is not None else "NO"
    vel = "yes" if s.velocity_ned is not None else "NO"
    mag = "yes" if s.mag_body is not None else "NO"
    baro = f"{s.baro_pressure_hpa:.2f}hPa" if s.baro_pressure_hpa is not None else "NO"
    return (
        f"armed={s.armed} sim_t={s.sim_time_ns / 1e9:8.3f}s "
        f"rpy=({s.roll:+.3f},{s.pitch:+.3f},{s.yaw:+.3f}) "
        f"pos={pos} vel={vel} mag={mag} baro={baro}"
    )


def baro_altitude_m(client: MavlinkClient) -> float | None:
    """ISA altitude from the latest baro pressure, or None if baro is absent.
    Only *relative* changes are meaningful (no arming-reference calibration here)."""
    p = client.state.baro_pressure_hpa
    return None if p is None else pressure_to_altitude_m(p)


def drain_statustexts(client: MavlinkClient, seen: int) -> int:
    """Print any STATUSTEXT past index ``seen``; return the new total count."""
    texts = client.statustexts
    for st in texts[seen:]:
        print(f"    STATUSTEXT[sev{st['severity']}]: {st['text']}")
    return len(texts)


def pump_for(client: MavlinkClient, seconds: float, *, show_statustext: bool = True) -> None:
    """Pump the client for a wall-clock duration (keeps the >=2 Hz heartbeat alive and
    drains RX), printing any new STATUSTEXT as it arrives."""
    seen = len(client.statustexts)
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        client.pump()
        if show_statustext:
            seen = drain_statustexts(client, seen)
        time.sleep(0.01)


def stream_setpoint(client, command, seconds: float, metric=None, rate_hz: float = 20.0):
    """Stream one ControlCommand for `seconds`, pumping each tick (sends the setpoint AND
    keeps the heartbeat alive + drains telemetry). If `metric` is given it is called each
    tick and its non-None returns are collected and returned (a simple response sampler for
    the actuating probes). Shared by control_mode_probe and innerloop_step."""
    dt = 1.0 / rate_hz
    end = time.monotonic() + seconds
    samples: list[float] = []
    while time.monotonic() < end:
        client.send_command(command)
        client.pump()
        if metric is not None:
            v = metric()
            if v is not None:
                samples.append(float(v))
        time.sleep(dt)
    return samples
