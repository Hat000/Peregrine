"""Minimal MAVLink2 UDP client for the AI Grand Prix simulator.

Maintains a heartbeat, parses inbound HEARTBEAT / ATTITUDE / HIGHRES_IMU /
TIMESYNC, and exposes helpers for the two supported control messages:
SET_POSITION_TARGET_LOCAL_NED and SET_ATTITUDE_TARGET.

Spec ref: VADR-TS-002 sec 4.
"""
from __future__ import annotations

import os

os.environ.setdefault("MAVLINK20", "1")

import time
from dataclasses import dataclass

from pymavlink import mavutil

# SET_POSITION_TARGET_LOCAL_NED type_mask bits (1 = ignore the corresponding input).
_POS_IGNORE_VX = 1 << 3
_POS_IGNORE_VY = 1 << 4
_POS_IGNORE_VZ = 1 << 5
_POS_IGNORE_AX = 1 << 6
_POS_IGNORE_AY = 1 << 7
_POS_IGNORE_AZ = 1 << 8
_POS_FORCE_SET = 1 << 9
_POS_IGNORE_YAW = 1 << 10
_POS_IGNORE_YAW_RATE = 1 << 11

_POS_MASK_USE_POS_AND_YAW = (
    _POS_IGNORE_VX | _POS_IGNORE_VY | _POS_IGNORE_VZ
    | _POS_IGNORE_AX | _POS_IGNORE_AY | _POS_IGNORE_AZ
    | _POS_IGNORE_YAW_RATE
)

# SET_ATTITUDE_TARGET type_mask: ignore body roll/pitch/yaw rates (bits 0-2); use attitude + thrust.
_ATT_MASK_USE_ATTITUDE_AND_THRUST = 0b00000111


@dataclass
class DroneState:
    timestamp_s: float = 0.0
    roll: float = 0.0       # rad
    pitch: float = 0.0      # rad
    yaw: float = 0.0        # rad
    rollspeed: float = 0.0  # rad/s
    pitchspeed: float = 0.0
    yawspeed: float = 0.0
    xacc: float = 0.0       # m/s^2 (body)
    yacc: float = 0.0
    zacc: float = 0.0
    abs_pressure: float = 0.0
    vx: float = 0.0         # if a velocity-bearing message turns up; see README
    vy: float = 0.0
    vz: float = 0.0


class MavlinkClient:
    HEARTBEAT_HZ = 2  # spec minimum

    def __init__(self, endpoint: str = "udp:127.0.0.1:14550"):
        self.endpoint = endpoint
        self.conn: mavutil.mavlink_connection | None = None
        self.state = DroneState()
        self.unknown_msg_types: set[str] = set()
        self._last_heartbeat_tx_s = 0.0

    def connect(self, wait_heartbeat: bool = True, timeout_s: float = 15.0) -> None:
        self.conn = mavutil.mavlink_connection(
            self.endpoint,
            source_system=255,
            autoreconnect=True,
        )
        if wait_heartbeat:
            msg = self.conn.wait_heartbeat(timeout=timeout_s)
            if msg is None:
                raise TimeoutError(
                    f"No HEARTBEAT received from {self.endpoint} within {timeout_s}s"
                )

    def pump(self) -> None:
        """Drain inbound MAVLink and emit a heartbeat if due. Call frequently."""
        assert self.conn is not None
        now = time.monotonic()
        while True:
            msg = self.conn.recv_match(blocking=False)
            if msg is None:
                break
            self._handle(msg)
        if now - self._last_heartbeat_tx_s >= 1.0 / self.HEARTBEAT_HZ:
            self._send_heartbeat()
            self._last_heartbeat_tx_s = now

    def _handle(self, msg) -> None:
        t = msg.get_type()
        if t == "ATTITUDE":
            self.state.roll = msg.roll
            self.state.pitch = msg.pitch
            self.state.yaw = msg.yaw
            self.state.rollspeed = msg.rollspeed
            self.state.pitchspeed = msg.pitchspeed
            self.state.yawspeed = msg.yawspeed
            self.state.timestamp_s = msg.time_boot_ms / 1000.0
        elif t == "HIGHRES_IMU":
            self.state.xacc = msg.xacc
            self.state.yacc = msg.yacc
            self.state.zacc = msg.zacc
            self.state.abs_pressure = msg.abs_pressure
        elif t == "LOCAL_POSITION_NED":
            # Not in spec table 4.3 but commonly emitted; capture if present.
            self.state.vx = msg.vx
            self.state.vy = msg.vy
            self.state.vz = msg.vz
        elif t in ("HEARTBEAT", "TIMESYNC", "BAD_DATA"):
            pass
        else:
            self.unknown_msg_types.add(t)

    def _send_heartbeat(self) -> None:
        assert self.conn is not None
        self.conn.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0, 0, 0,
        )

    def send_position_target(self, x: float, y: float, z: float, yaw: float = 0.0) -> None:
        """Position waypoint in MAV_FRAME_LOCAL_NED (world). Yaw in radians."""
        assert self.conn is not None
        self.conn.mav.set_position_target_local_ned_send(
            int(time.monotonic() * 1000),
            self.conn.target_system,
            self.conn.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            _POS_MASK_USE_POS_AND_YAW,
            x, y, z,
            0, 0, 0,
            0, 0, 0,
            yaw, 0,
        )

    def send_attitude_target(
        self,
        q_wxyz: tuple[float, float, float, float],
        thrust: float,
    ) -> None:
        """Attitude + thrust. q = (w, x, y, z) in body NED. thrust in [0, 1]."""
        assert self.conn is not None
        self.conn.mav.set_attitude_target_send(
            int(time.monotonic() * 1000),
            self.conn.target_system,
            self.conn.target_component,
            _ATT_MASK_USE_ATTITUDE_AND_THRUST,
            list(q_wxyz),
            0, 0, 0,
            thrust,
        )
