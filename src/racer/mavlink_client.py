"""Minimal MAVLink2 UDP client for the AI Grand Prix simulator.

Maintains a heartbeat, parses inbound HEARTBEAT / ATTITUDE / HIGHRES_IMU /
LOCAL_POSITION_NED / TIMESYNC into an immutable :class:`DroneState` snapshot, and
translates a :class:`ControlCommand` into the right control message + type_mask.

Spec ref: VADR-TS-002 sec 4. Data contracts: :mod:`racer.contracts`.
"""
from __future__ import annotations

import os

os.environ.setdefault("MAVLINK20", "1")

import time
from dataclasses import replace

import numpy as np
from pymavlink import mavutil

from racer.contracts import ControlCommand, ControlMode, DroneState

# SET_POSITION_TARGET_LOCAL_NED type_mask bits (1 = ignore the corresponding input).
_POS_IGNORE_PX = 1 << 0
_POS_IGNORE_PY = 1 << 1
_POS_IGNORE_PZ = 1 << 2
_POS_IGNORE_VX = 1 << 3
_POS_IGNORE_VY = 1 << 4
_POS_IGNORE_VZ = 1 << 5
_POS_IGNORE_AX = 1 << 6
_POS_IGNORE_AY = 1 << 7
_POS_IGNORE_AZ = 1 << 8
_POS_FORCE_SET = 1 << 9
_POS_IGNORE_YAW = 1 << 10
_POS_IGNORE_YAW_RATE = 1 << 11

# SET_ATTITUDE_TARGET type_mask bits.
_ATT_MASK_ATTITUDE = 0b00000111   # ignore the 3 body rates -> use attitude quat + thrust (angle mode)
_ATT_MASK_BODY_RATE = 0b10000000  # ignore attitude -> use body rates + thrust (acro / CTBR)

_ZERO3 = (0.0, 0.0, 0.0)


def _pos_type_mask(
    position: np.ndarray | None,
    velocity: np.ndarray | None,
    accel: np.ndarray | None,
    yaw: float | None,
    yaw_rate: float | None,
) -> int:
    """Build a SET_POSITION_TARGET type_mask that *uses* exactly the non-None terms."""
    mask = 0
    if position is None:
        mask |= _POS_IGNORE_PX | _POS_IGNORE_PY | _POS_IGNORE_PZ
    if velocity is None:
        mask |= _POS_IGNORE_VX | _POS_IGNORE_VY | _POS_IGNORE_VZ
    if accel is None:
        mask |= _POS_IGNORE_AX | _POS_IGNORE_AY | _POS_IGNORE_AZ
    if yaw is None:
        mask |= _POS_IGNORE_YAW
    if yaw_rate is None:
        mask |= _POS_IGNORE_YAW_RATE
    return mask


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
        # Update the immutable snapshot wholesale (replace) so concurrent readers never
        # see a half-written state. Each branch builds fresh arrays for the fields it owns.
        t = msg.get_type()
        recv = time.monotonic_ns()
        if t == "ATTITUDE":
            self.state = replace(
                self.state,
                sim_time_ns=int(msg.time_boot_ms) * 1_000_000,
                recv_monotonic_ns=recv,
                roll=msg.roll,
                pitch=msg.pitch,
                yaw=msg.yaw,
                angular_rate_body=np.array(
                    [msg.rollspeed, msg.pitchspeed, msg.yawspeed], dtype=np.float64
                ),
            )
        elif t == "HIGHRES_IMU":
            # NB: time_usec epoch may differ from ATTITUDE.time_boot_ms — see clock TODO.
            self.state = replace(
                self.state,
                sim_time_ns=int(msg.time_usec) * 1_000,
                recv_monotonic_ns=recv,
                accel_body=np.array([msg.xacc, msg.yacc, msg.zacc], dtype=np.float64),
                mag_body=np.array([msg.xmag, msg.ymag, msg.zmag], dtype=np.float64),
                baro_pressure_hpa=float(msg.abs_pressure),
            )
        elif t == "LOCAL_POSITION_NED":
            # Not in spec table 4.3, but if the sim emits it we get position + velocity
            # for free (the localisation problem collapses). Capture eagerly.
            self.state = replace(
                self.state,
                recv_monotonic_ns=recv,
                position_ned=np.array([msg.x, msg.y, msg.z], dtype=np.float64),
                velocity_ned=np.array([msg.vx, msg.vy, msg.vz], dtype=np.float64),
            )
        elif t == "HEARTBEAT":
            armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            self.state = replace(self.state, armed=armed, status_flags=int(msg.base_mode))
        elif t in ("TIMESYNC", "BAD_DATA"):
            pass  # TODO(clock): use TIMESYNC to reconcile sim_time_ns across streams.
        else:
            self.unknown_msg_types.add(t)

    def _send_heartbeat(self) -> None:
        assert self.conn is not None
        self.conn.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0, 0, 0,
        )

    def _now_ms(self) -> int:
        return int(time.monotonic() * 1000) & 0xFFFFFFFF

    # -- control output -----------------------------------------------------
    def send_command(self, cmd: ControlCommand) -> None:
        """Translate a :class:`ControlCommand` into the appropriate MAVLink message."""
        assert self.conn is not None
        if cmd.mode in (ControlMode.POSITION, ControlMode.VELOCITY):
            mask = _pos_type_mask(
                cmd.position_ned, cmd.velocity_ned, cmd.accel_ned, cmd.yaw, cmd.yaw_rate
            )
            px, py, pz = cmd.position_ned if cmd.position_ned is not None else _ZERO3
            vx, vy, vz = cmd.velocity_ned if cmd.velocity_ned is not None else _ZERO3
            ax, ay, az = cmd.accel_ned if cmd.accel_ned is not None else _ZERO3
            self.conn.mav.set_position_target_local_ned_send(
                self._now_ms(),
                self.conn.target_system,
                self.conn.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                mask,
                px, py, pz,
                vx, vy, vz,
                ax, ay, az,
                cmd.yaw if cmd.yaw is not None else 0.0,
                cmd.yaw_rate if cmd.yaw_rate is not None else 0.0,
            )
        elif cmd.mode == ControlMode.ATTITUDE:
            assert cmd.attitude_quat_wxyz is not None and cmd.thrust is not None
            self.conn.mav.set_attitude_target_send(
                self._now_ms(),
                self.conn.target_system,
                self.conn.target_component,
                _ATT_MASK_ATTITUDE,
                [float(x) for x in cmd.attitude_quat_wxyz],
                0.0, 0.0, 0.0,
                float(cmd.thrust),
            )
        elif cmd.mode == ControlMode.BODY_RATE:
            assert cmd.body_rate is not None and cmd.thrust is not None
            r = cmd.body_rate
            self.conn.mav.set_attitude_target_send(
                self._now_ms(),
                self.conn.target_system,
                self.conn.target_component,
                _ATT_MASK_BODY_RATE,
                [1.0, 0.0, 0.0, 0.0],  # quaternion ignored by the mask
                float(r[0]), float(r[1]), float(r[2]),
                float(cmd.thrust),
            )
        else:
            raise ValueError(f"unknown control mode: {cmd.mode!r}")

    # -- low-level helpers (kept for convenience / smoke tests) -------------
    def send_position_target(self, x: float, y: float, z: float, yaw: float = 0.0) -> None:
        """Position waypoint in MAV_FRAME_LOCAL_NED (world). Yaw in radians."""
        self.send_command(
            ControlCommand(
                mode=ControlMode.POSITION,
                position_ned=np.array([x, y, z], dtype=np.float64),
                yaw=yaw,
            )
        )

    def send_attitude_target(
        self,
        q_wxyz: tuple[float, float, float, float],
        thrust: float,
    ) -> None:
        """Attitude + thrust (angle mode). q = (w, x, y, z) in body NED. thrust in [0, 1]."""
        self.send_command(
            ControlCommand(
                mode=ControlMode.ATTITUDE,
                attitude_quat_wxyz=np.array(q_wxyz, dtype=np.float64),
                thrust=thrust,
            )
        )
