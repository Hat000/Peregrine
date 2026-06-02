"""Minimal MAVLink2 UDP client for the AI Grand Prix simulator.

Maintains a heartbeat, parses inbound telemetry into an immutable :class:`DroneState`
snapshot, and translates a :class:`ControlCommand` into the right control message +
type_mask. Message coverage was confirmed against the official ``PyAIPilotExample`` client
shipped inside the sim zip: HEARTBEAT, ATTITUDE (liveness only — its Euler pitch sign is
inverted), HIGHRES_IMU, LOCAL_POSITION_NED, ODOMETRY (canonical pose + orientation quaternion
+ reset_counter), TIMESYNC, COMMAND_ACK, STATUSTEXT, COLLISION, ACTUATOR_OUTPUT_STATUS,
and the sim's *repurposed* ENCAPSULATED_DATA carrying RACE_STATUS (active gate + race timing)
and a chunked TRACK_INFO gate map (announced via DATA_TRANSMISSION_HANDSHAKE).

Spec ref: VADR-TS-002 sec 4. Data contracts: :mod:`racer.contracts`.
"""
from __future__ import annotations

import os

os.environ.setdefault("MAVLINK20", "1")

import struct
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
from pymavlink import mavutil

from racer.contracts import ControlCommand, ControlMode, DroneState
from racer.frames import euler_from_quat_wxyz

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


def _result_name(result: int) -> str:
    """Human-readable MAV_RESULT enum name for a COMMAND_ACK result code."""
    try:
        return mavutil.mavlink.enums["MAV_RESULT"][result].name
    except Exception:
        return str(result)


# -- the sim's custom / repurposed MAVLink payloads ------------------------------------
# All confirmed from the official PyAIPilotExample shipped inside the sim zip.
MAV_CMD_SIM_RESET = 31000  # COMMAND_LONG: reset the sim to the start (clean attempt loop)

_ENCAP_RACE_STATUS = 1     # ENCAPSULATED_DATA.data[0] discriminator
_ENCAP_TRACK_INFO = 2
# RACE_STATUS payload: data_type(B), sim_boot_ms(Q), race_start_boot_ms(q), race_finish_ns(q),
#                      active_gate_index(I), last_gate_race_time(q). (<0 = not started / ongoing.)
_RACE_STATUS_FMT = "<BQqqIq"
# One gate in a TRACK_INFO payload: id(H), pos NED x/y/z(f), quat w/x/y/z(f), width(f), height(f).
_TRACK_GATE_FMT = "<Hfffffffff"


def parse_race_status(payload: bytes) -> dict | None:
    """Decode an ENCAPSULATED_DATA RACE_STATUS payload (data_type==1). None if too short."""
    if len(payload) < struct.calcsize(_RACE_STATUS_FMT):
        return None
    (_dt, sim_boot_ms, race_start_boot_ms, race_finish_ns,
     active_gate_index, last_gate_race_time) = struct.unpack_from(_RACE_STATUS_FMT, payload, 0)
    return {
        "sim_boot_time_ms": int(sim_boot_ms),
        "race_start_boot_time_ms": int(race_start_boot_ms),
        "race_finish_time_ns": int(race_finish_ns),
        "active_gate_index": int(active_gate_index),
        "last_gate_race_time": int(last_gate_race_time),
        "started": int(race_start_boot_ms) >= 0,
        "finished": int(race_finish_ns) >= 0,
    }


def parse_track_info(payload: bytes) -> list[dict]:
    """Decode a reassembled TRACK_INFO gate map: u16 num_gates then that many gate records.
    Each gate -> {gate_id, position_ned (3,), orientation_ned_wxyz (4,), width_m, height_m}."""
    gates: list[dict] = []
    if len(payload) < 2:
        return gates
    (num_gates,) = struct.unpack_from("<H", payload, 0)
    off = 2
    size = struct.calcsize(_TRACK_GATE_FMT)
    for _ in range(int(num_gates)):
        if off + size > len(payload):
            break
        gid, px, py, pz, qw, qx, qy, qz, w, h = struct.unpack_from(_TRACK_GATE_FMT, payload, off)
        gates.append({
            "gate_id": int(gid),
            "position_ned": np.array([px, py, pz], dtype=np.float64),
            "orientation_ned_wxyz": np.array([qw, qx, qy, qz], dtype=np.float64),
            "width_m": float(w),
            "height_m": float(h),
        })
        off += size
    return gates


class MavlinkClient:
    HEARTBEAT_HZ = 2  # spec minimum

    def __init__(self, endpoint: str = "udp:127.0.0.1:14550"):
        self.endpoint = endpoint
        self.conn: mavutil.mavlink_connection | None = None
        self.state = DroneState()
        self.unknown_msg_types: set[str] = set()
        self._last_heartbeat_tx_s = 0.0
        # Optional raw-message tap, called with each inbound pymavlink message BEFORE
        # parsing. The recorder sets this to capture msg.get_msgbuf() (raw wire bytes).
        # Kept orthogonal so recording never perturbs the parse/state path.
        self.on_message: Callable[[Any], None] | None = None
        # First-contact diagnostics (events / heartbeat metadata, not steady telemetry, so
        # kept OFF the immutable DroneState snapshot). Populated by _handle; read by the probes.
        self.last_command_ack: dict | None = None   # {command, result, result_name, recv_monotonic_ns}
        self.statustexts: list[dict] = []            # recent STATUSTEXT: {severity, text, recv_monotonic_ns}
        self._max_statustexts = 200
        self.autopilot: int | None = None            # HEARTBEAT.autopilot (MAV_AUTOPILOT)
        self.vehicle_type: int | None = None         # HEARTBEAT.type (MAV_TYPE)
        self.custom_mode: int | None = None          # HEARTBEAT.custom_mode (autopilot-specific)
        # Sim-provided race/track state via the repurposed ENCAPSULATED_DATA (see _handle).
        # Kept off DroneState (map data + events, not steady per-tick telemetry).
        self.race_status: dict | None = None          # latest RACE_STATUS (active gate + timing)
        self.track_gates: list[dict] | None = None    # full TRACK_INFO gate map once reassembled
        self.collisions: list[dict] = []              # COLLISION events (gate 1001 / env 1002)
        self._max_collisions = 200
        self.actuator_outputs: dict | None = None     # latest ACTUATOR_OUTPUT_STATUS (motor cmds)
        self._track_chunks: dict[int, dict[int, bytes]] = {}   # transfer_id -> {seqnr: bytes}
        self._track_expected: dict[int, int] = {}              # transfer_id -> expected chunk count

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
            if self.on_message is not None:
                self.on_message(msg)
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
            # ORIENTATION RETIRED [first-contact 2026-06-02]: the sim's ATTITUDE Euler is
            # sign-inconsistent — its pitch is inverted vs the ODOMETRY quaternion, the
            # accel-gravity vector, AND the FPV view. So orientation + body rate now come
            # SOLELY from ODOMETRY (one self-consistent frame; see that branch). ATTITUDE is
            # kept only for liveness/rate-stats (recv_monotonic_ns).
            # CLOCK ISOLATION [review 2A]: ATTITUDE.time_boot_ms is a DIFFERENT epoch than
            # HIGHRES_IMU.time_usec, so it must never drive sim_time_ns (that made the timeline
            # oscillate and diverged the estimator). HIGHRES_IMU is the sole clock driver.
            self.state = replace(self.state, recv_monotonic_ns=recv)
        elif t == "HIGHRES_IMU":
            # Sole driver of sim_time_ns — the master sim timeline (see ATTITUDE above).
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
        elif t == "ODOMETRY":
            # The CANONICAL pose+orientation source: full pose (pos/vel + quaternion) plus a
            # reset_counter that ticks when the sim epoch restarts. Orientation comes from the
            # quaternion (q = w,x,y,z, body FRD -> world NED) — NOT the sign-inverted ATTITUDE
            # Euler — and roll/pitch/yaw are DERIVED from it in the one convention the
            # controller commands in (frames). Body rate is taken here too, so orientation and
            # its derivative share one self-consistent frame. Clock stays with HIGHRES_IMU.
            q = np.array([float(v) for v in msg.q], dtype=np.float64)
            roll, pitch, yaw = euler_from_quat_wxyz(q)
            self.state = replace(
                self.state,
                recv_monotonic_ns=recv,
                position_ned=np.array([msg.x, msg.y, msg.z], dtype=np.float64),
                velocity_ned=np.array([msg.vx, msg.vy, msg.vz], dtype=np.float64),
                orientation_ned_wxyz=q,
                roll=roll,
                pitch=pitch,
                yaw=yaw,
                angular_rate_body=np.array(
                    [msg.rollspeed, msg.pitchspeed, msg.yawspeed], dtype=np.float64
                ),
                reset_counter=int(getattr(msg, "reset_counter", 0)),
            )
        elif t == "HEARTBEAT":
            armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            self.state = replace(self.state, armed=armed, status_flags=int(msg.base_mode))
            # Backend metadata: which autopilot/mode model are we actually talking to? A
            # first-contact unknown (PX4? ArduPilot? Betaflight-ish?) the probes report.
            self.autopilot = int(msg.autopilot)
            self.vehicle_type = int(msg.type)
            self.custom_mode = int(msg.custom_mode)
        elif t == "COMMAND_ACK":
            # Tells us if the sim accepted/rejected a command (e.g. arm) + why. Keystone
            # feedback for the lifecycle/control probes.
            self.last_command_ack = {
                "command": int(msg.command),
                "result": int(msg.result),
                "result_name": _result_name(int(msg.result)),
                "recv_monotonic_ns": recv,
            }
        elif t == "STATUSTEXT":
            # Sims often narrate the session here ("armed", "race started", "gate N",
            # "finished") -- can reveal the whole lifecycle for free. Capture verbatim.
            text = msg.text
            if isinstance(text, (bytes, bytearray)):
                text = bytes(text).decode("ascii", "replace")
            self.statustexts.append({
                "severity": int(getattr(msg, "severity", 6)),
                "text": str(text).rstrip("\x00").strip(),
                "recv_monotonic_ns": recv,
            })
            if len(self.statustexts) > self._max_statustexts:
                self.statustexts.pop(0)
        elif t == "ENCAPSULATED_DATA":
            # The sim repurposes this to carry RACE_STATUS + a chunked TRACK_INFO gate map.
            self._handle_encapsulated(msg)
        elif t == "DATA_TRANSMISSION_HANDSHAKE":
            # Announces an incoming chunked TRACK_INFO transfer (width=transfer_id, packets=count).
            self._track_chunks[int(msg.width)] = {}
            self._track_expected[int(msg.width)] = int(msg.packets)
        elif t == "COLLISION":
            self.collisions.append({
                "id": int(msg.id),                                       # 1001=gate, 1002=environment
                "threat_level": int(getattr(msg, "threat_level", 0)),   # 1..2 (2 = harder hit)
                "impulse": float(getattr(msg, "horizontal_minimum_delta", 0.0)),  # impulse kg*m/s
                "recv_monotonic_ns": recv,
                "sim_time_ns": self.state.sim_time_ns,
            })
            if len(self.collisions) > self._max_collisions:
                self.collisions.pop(0)
        elif t == "ACTUATOR_OUTPUT_STATUS":
            self.actuator_outputs = {
                "time_usec": int(getattr(msg, "time_usec", 0)),
                "motors": [float(x) for x in list(msg.actuator)[:4]],
            }
        elif t in ("TIMESYNC", "BAD_DATA"):
            pass  # TODO(clock): use TIMESYNC to reconcile sim_time_ns across streams.
        else:
            self.unknown_msg_types.add(t)

    def _handle_encapsulated(self, msg) -> None:
        """Route a repurposed ENCAPSULATED_DATA payload by its leading discriminator byte."""
        raw = bytes(msg.data)
        if not raw:
            return
        data_type = raw[0]
        if data_type == _ENCAP_RACE_STATUS:
            rs = parse_race_status(raw)
            if rs is not None:
                self.race_status = rs
        elif data_type == _ENCAP_TRACK_INFO:
            self._ingest_track_chunk(msg, raw)

    def _ingest_track_chunk(self, msg, raw: bytes) -> None:
        """Reassemble one TRACK_INFO chunk; parse the gate map once all chunks are in.

        Chunk layout: data_type(B), transfer_id(H), then a payload slice. Slices are
        concatenated by seqnr; only the final chunk is zero-padded, and parse_track_info
        reads exactly the gates the u16 count promises, so the trailing padding is ignored.
        """
        if len(raw) < 3:
            return
        _dt, transfer_id = struct.unpack_from("<BH", raw, 0)
        if transfer_id not in self._track_expected:
            return  # no DATA_TRANSMISSION_HANDSHAKE announced this transfer yet
        self._track_chunks.setdefault(transfer_id, {})[int(msg.seqnr)] = raw[3:]
        if len(self._track_chunks[transfer_id]) < self._track_expected[transfer_id]:
            return
        n = self._track_expected[transfer_id]
        try:
            full = b"".join(self._track_chunks[transfer_id][i] for i in range(n))
        except KeyError:
            return  # a seqnr is still missing (a duplicate filled the count) -> wait for it
        gates = parse_track_info(full)
        if gates:
            self.track_gates = gates
        self._track_chunks.pop(transfer_id, None)
        self._track_expected.pop(transfer_id, None)

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

    # -- arming + lifecycle (spec-standard MAVLink; autopilot-AGNOSTIC, NOT Elodin) --------
    def send_command_long(
        self, command: int,
        p1: float = 0.0, p2: float = 0.0, p3: float = 0.0, p4: float = 0.0,
        p5: float = 0.0, p6: float = 0.0, p7: float = 0.0, *, confirmation: int = 0,
    ) -> None:
        """Send a COMMAND_LONG. Generic so the probes can issue any MAV_CMD_*."""
        assert self.conn is not None
        self.conn.mav.command_long_send(
            self.conn.target_system, self.conn.target_component,
            command, confirmation, p1, p2, p3, p4, p5, p6, p7,
        )

    def arm(self, force: bool = False) -> None:
        """Request ARM via MAV_CMD_COMPONENT_ARM_DISARM (param1=1). ``force`` sends the 21196
        magic that bypasses prearm checks. Confirm with wait_command_ack / wait_armed. Standard
        across PX4/ArduPilot; the spec sim's actual arming policy is a first-contact unknown."""
        self.send_command_long(
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1.0, 21196.0 if force else 0.0
        )

    def disarm(self, force: bool = False) -> None:
        """Request DISARM (param1=0)."""
        self.send_command_long(
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0.0, 21196.0 if force else 0.0
        )

    def send_sim_reset(self) -> None:
        """Reset the sim to the start via the custom MAV_CMD 31000 (from PyAIPilotExample).
        The course is deterministic, so this is the clean attempt-iteration primitive."""
        self.send_command_long(MAV_CMD_SIM_RESET)

    def wait_command_ack(self, command: int, timeout_s: float = 3.0) -> dict | None:
        """Pump until a COMMAND_ACK for ``command`` arrives; return the ack dict or None on
        timeout. Set ``self.last_command_ack = None`` before sending to avoid a stale match."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self.pump()
            ack = self.last_command_ack
            if ack is not None and ack["command"] == command:
                return ack
            time.sleep(0.005)
        return None

    def wait_armed(self, armed: bool = True, timeout_s: float = 5.0) -> bool:
        """Pump until the HEARTBEAT armed flag (ground truth) reaches ``armed``. Returns bool."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self.pump()
            if self.state.armed == armed:
                return True
            time.sleep(0.01)
        return False
