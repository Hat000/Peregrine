"""Shared helpers for the first-contact probe scripts.

These read ONLY the spec MAVLink interface via :class:`MavlinkClient` (VADR-TS-002 sec 4).
Nothing here knows about Elodin -- Elodin is a separate Apache-2.0 *surrogate* with an
RC-callback API and is NOT authoritative for the scoring simulator. Keep it that way: the
probes characterise the REAL sim's MAVLink behaviour, never Elodin's.
"""
from __future__ import annotations

import time

import numpy as np

from racer.atmosphere import pressure_to_altitude_m
from racer.mavlink_client import MavlinkClient

_G = 9.80665


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


def mission_summary(client: MavlinkClient) -> str:
    """One line on the sim-provided race/track state (repurposed ENCAPSULATED_DATA) + any
    collisions. Resolves R4 (how gate order/positions arrive) and exercises the mission
    plumbing (active gate, race start/finish, collision feedback)."""
    gates = client.track_gates
    gate_map = f"{len(gates)} gates" if gates else "NONE"
    rs = client.race_status
    if rs is None:
        race = "race_status=NONE"
    else:
        race = (
            f"active_gate={rs['active_gate_index']} "
            f"started={rs['started']} finished={rs['finished']}"
        )
    return f"gate_map={gate_map}  {race}  collisions={len(client.collisions)}"


def baro_altitude_m(client: MavlinkClient) -> float | None:
    """ISA altitude from the latest baro pressure, or None if baro is absent.
    Only *relative* changes are meaningful (no arming-reference calibration here)."""
    p = client.state.baro_pressure_hpa
    return None if p is None else pressure_to_altitude_m(p)


# -- attitude-bias observability (red-team CRIT-2: is the GIVEN attitude biased?) ---------
def gravity_tilt(accel_body) -> tuple[float, float] | None:
    """Roll/pitch (rad) implied by the accelerometer, assuming the ONLY specific force is the
    gravity reaction (quasi-static / non-accelerating). FRD specific force reads ~[0,0,-g] at
    level. Returns None when |a| is not ~1 g (the vehicle is accelerating, so gravity can't be
    isolated and the inferred tilt would be meaningless)."""
    a = np.asarray(accel_body, dtype=np.float64)
    n = float(np.linalg.norm(a))
    if n < 1e-6 or abs(n - _G) > 0.15 * _G:
        return None
    roll = float(np.arctan2(-a[1], -a[2]))
    pitch = float(np.arctan2(a[0], float(np.hypot(a[1], a[2]))))
    return roll, pitch


def attitude_gravity_residual(client: MavlinkClient) -> tuple[float, float] | None:
    """(roll, pitch) residual = GIVEN attitude minus gravity-implied tilt, in rad; None if not
    quasi-static. A persistent non-zero residual AT REST = a static bias in the sim's given
    attitude -> the ESKF bias-state may be warranted (red-team CRIT-2). Yaw is unobservable
    from gravity. NB this only catches a STATIC bias; a dynamic (high-G) bias needs in-flight
    cross-checks against vision residuals."""
    s = client.state
    tilt = gravity_tilt(s.accel_body)
    if tilt is None:
        return None
    return (s.roll - tilt[0], s.pitch - tilt[1])


def sample_attitude_bias(client: MavlinkClient, seconds: float = 2.0, dt: float = 0.02) -> dict | None:
    """Pump for ``seconds``, averaging the given-vs-gravity attitude residual while quasi-static.
    Returns a dict of roll/pitch bias mean+std (deg) and sample counts, or None if the vehicle
    was never quasi-static. Best run PRE-ARM on the ground. Feeds the ESKF-bias decision."""
    rolls: list[float] = []
    pitches: list[float] = []
    n_total = 0
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        client.pump()
        n_total += 1
        res = attitude_gravity_residual(client)
        if res is not None:
            rolls.append(res[0])
            pitches.append(res[1])
        time.sleep(dt)
    if not rolls:
        return None
    return {
        "roll_bias_deg": float(np.degrees(np.mean(rolls))),
        "pitch_bias_deg": float(np.degrees(np.mean(pitches))),
        "roll_bias_std_deg": float(np.degrees(np.std(rolls))),
        "pitch_bias_std_deg": float(np.degrees(np.std(pitches))),
        "n": len(rolls),
        "n_total": n_total,
    }


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


# -- message-rate characterisation -----------------------------------------------------------
# First contact: are we capturing every stream, and at the rates the spec implies? VADR-TS-002
# sec 4.4 pins HEARTBEAT >= 2 Hz; sec 4.6 camera = 30 Hz; physics = 120 Hz. The ATTITUDE /
# HIGHRES_IMU / LOCAL_POSITION_NED / ODOMETRY rates are UNSPECIFIED -> measured here, reported,
# not asserted.
class MessageRateTracker:
    """Per-message-type arrival rate + count over the observation window.

    Feed each inbound message's ``(type, recv_monotonic_ns)``; query per-type Hz / count.
    Rate = (count - 1) / span, i.e. the mean inter-arrival cadence (a lone message has no
    defined rate -> 0.0). Pure (no I/O), so it unit-tests cleanly and can also be driven from a
    recording replay. Stamp with ``time.monotonic_ns()`` at RECEIVE time (not sim time, so a sim
    clock reset cannot skew it)."""

    def __init__(self) -> None:
        self._first: dict[str, int] = {}
        self._last: dict[str, int] = {}
        self._count: dict[str, int] = {}

    def record(self, msg_type: str, recv_monotonic_ns: int) -> None:
        if msg_type not in self._first:
            self._first[msg_type] = recv_monotonic_ns
        self._last[msg_type] = recv_monotonic_ns
        self._count[msg_type] = self._count.get(msg_type, 0) + 1

    def count(self, msg_type: str) -> int:
        return self._count.get(msg_type, 0)

    def rate_hz(self, msg_type: str) -> float:
        n = self._count.get(msg_type, 0)
        span_ns = self._last.get(msg_type, 0) - self._first.get(msg_type, 0)
        return (n - 1) / (span_ns / 1e9) if (n > 1 and span_ns > 0) else 0.0

    def rates(self) -> dict[str, dict]:
        """``{type: {count, hz}}`` ordered by descending count."""
        return {
            t: {"count": self._count[t], "hz": self.rate_hz(t)}
            for t in sorted(self._count, key=lambda k: (-self._count[k], k))
        }

    def report(self) -> str:
        rows = self.rates()
        if not rows:
            return "  (no messages)"
        return "\n".join(f"  {t:28s} {r['count']:7d}  {r['hz']:7.1f} Hz" for t, r in rows.items())


# Only HEARTBEAT has a spec-pinned MAVLink minimum rate (sec 4.4). Camera (30 Hz, sec 4.6) is
# checked on the video stream (frames/duration), not here.
SPEC_MIN_RATES_HZ = {"HEARTBEAT": 2.0}


def rate_warnings(tracker: MessageRateTracker) -> list[str]:
    """Flag any message type measured below its spec-pinned minimum rate."""
    out: list[str] = []
    for t, lo in SPEC_MIN_RATES_HZ.items():
        if tracker.count(t) > 1 and tracker.rate_hz(t) < lo:
            out.append(f"{t} {tracker.rate_hz(t):.1f} Hz < spec minimum {lo:g} Hz")
    return out
