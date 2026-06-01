"""Tests for the receive-side clock handling in mavlink_client (the send-side type_mask
helpers are covered in test_contracts). Exercises ``_handle`` directly with fake messages,
so no socket/connection is needed."""
import struct
from types import SimpleNamespace

import numpy as np

from racer.mavlink_client import MavlinkClient  # sets MAVLINK20 before importing mavutil
from pymavlink import mavutil


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


# -- first-contact additions: arming / acks / statustext / heartbeat metadata --------------
class _FakeMav:
    def __init__(self):
        self.sent: list[dict] = []

    def command_long_send(self, target_system, target_component, command, confirmation,
                          p1, p2, p3, p4, p5, p6, p7):
        self.sent.append(dict(command=command, confirmation=confirmation,
                              p1=p1, p2=p2, p3=p3, p4=p4, p5=p5, p6=p6, p7=p7))


class _FakeConn:
    def __init__(self):
        self.mav = _FakeMav()
        self.target_system = 1
        self.target_component = 1


def _command_ack(command, result):
    m = SimpleNamespace(command=command, result=result)
    m.get_type = lambda: "COMMAND_ACK"
    return m


def _statustext(text, severity=6):
    m = SimpleNamespace(severity=severity, text=text)
    m.get_type = lambda: "STATUSTEXT"
    return m


def _heartbeat(autopilot, vtype, base_mode=0, custom_mode=0):
    m = SimpleNamespace(autopilot=autopilot, type=vtype, base_mode=base_mode, custom_mode=custom_mode)
    m.get_type = lambda: "HEARTBEAT"
    return m


def test_arm_disarm_send_component_arm_disarm():
    c = MavlinkClient()
    c.conn = _FakeConn()
    c.arm()
    s = c.conn.mav.sent[-1]
    assert s["command"] == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM
    assert s["p1"] == 1.0 and s["p2"] == 0.0          # arm, no force
    c.arm(force=True)
    assert c.conn.mav.sent[-1]["p2"] == 21196.0       # force magic
    c.disarm()
    assert c.conn.mav.sent[-1]["p1"] == 0.0           # disarm


def test_command_ack_captured_with_name():
    c = MavlinkClient()
    c._handle(_command_ack(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                           mavutil.mavlink.MAV_RESULT_ACCEPTED))
    ack = c.last_command_ack
    assert ack["command"] == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM
    assert ack["result"] == mavutil.mavlink.MAV_RESULT_ACCEPTED
    assert ack["result_name"] == "MAV_RESULT_ACCEPTED"


def test_statustext_captured_decoded_and_capped():
    c = MavlinkClient()
    c._max_statustexts = 3
    c._handle(_statustext(b"Armed\x00\x00"))          # bytes, NUL-padded
    c._handle(_statustext("Race started"))
    assert c.statustexts[0]["text"] == "Armed"
    assert c.statustexts[1]["text"] == "Race started"
    for i in range(5):
        c._handle(_statustext(f"m{i}"))
    assert len(c.statustexts) == 3                     # cap holds
    assert c.statustexts[-1]["text"] == "m4"


def test_heartbeat_captures_backend_metadata():
    c = MavlinkClient()
    c._handle(_heartbeat(mavutil.mavlink.MAV_AUTOPILOT_PX4,
                         mavutil.mavlink.MAV_TYPE_QUADROTOR,
                         base_mode=mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED,
                         custom_mode=7))
    assert c.autopilot == mavutil.mavlink.MAV_AUTOPILOT_PX4
    assert c.vehicle_type == mavutil.mavlink.MAV_TYPE_QUADROTOR
    assert c.custom_mode == 7
    assert c.state.armed is True


# -- sim-dialect additions: ODOMETRY / ENCAPSULATED_DATA (race + track) / COLLISION ---------
def _odometry(x=5.0, y=6.0, z=-7.0, vx=1.0, vy=0.0, vz=-0.5, reset_counter=0):
    m = SimpleNamespace(x=x, y=y, z=z, vx=vx, vy=vy, vz=vz,
                        q=[1.0, 0.0, 0.0, 0.0], rollspeed=0.0, pitchspeed=0.0, yawspeed=0.0,
                        time_usec=123, reset_counter=reset_counter)
    m.get_type = lambda: "ODOMETRY"
    return m


def _collision(cid=1001, threat=2, impulse=3.5):
    m = SimpleNamespace(id=cid, threat_level=threat, horizontal_minimum_delta=impulse)
    m.get_type = lambda: "COLLISION"
    return m


def _actuator(motors=(0.1, 0.2, 0.3, 0.4)):
    m = SimpleNamespace(time_usec=42, actuator=list(motors) + [0.0, 0.0, 0.0, 0.0])
    m.get_type = lambda: "ACTUATOR_OUTPUT_STATUS"
    return m


def _encap_race_status(active_gate=3, started=True, finished=False):
    payload = struct.pack("<BQqqIq", 1, 1000,
                          500 if started else -1,
                          2000 if finished else -1,
                          active_gate, 7)
    m = SimpleNamespace(data=list(payload), seqnr=0)
    m.get_type = lambda: "ENCAPSULATED_DATA"
    return m


def _handshake(transfer_id, packets):
    m = SimpleNamespace(width=transfer_id, packets=packets)
    m.get_type = lambda: "DATA_TRANSMISSION_HANDSHAKE"
    return m


def _encap_track_chunk(transfer_id, seqnr, payload):
    raw = struct.pack("<BH", 2, transfer_id) + payload
    m = SimpleNamespace(data=list(raw), seqnr=seqnr)
    m.get_type = lambda: "ENCAPSULATED_DATA"
    return m


def _gate_map_bytes(n=2):
    buf = struct.pack("<H", n)
    for i in range(n):
        buf += struct.pack("<Hfffffffff", i, float(i), float(i + 1), float(i + 2),
                           1.0, 0.0, 0.0, 0.0, 1.5, 1.5)
    return buf


def test_odometry_captures_pose_and_reset_counter():
    c = MavlinkClient()
    c._handle(_imu(time_usec=1_000_000))
    c._handle(_odometry(x=5.0, y=6.0, z=-7.0, vx=1.0, vy=0.0, vz=-0.5, reset_counter=4))
    np.testing.assert_array_equal(c.state.position_ned, [5.0, 6.0, -7.0])
    np.testing.assert_array_equal(c.state.velocity_ned, [1.0, 0.0, -0.5])
    assert c.state.reset_counter == 4
    assert c.state.sim_time_ns == 1_000_000_000   # ODOMETRY must NOT drive the clock


def test_encapsulated_race_status_parsed():
    c = MavlinkClient()
    c._handle(_encap_race_status(active_gate=3, started=True, finished=False))
    rs = c.race_status
    assert rs is not None
    assert rs["active_gate_index"] == 3
    assert rs["started"] is True and rs["finished"] is False


def test_track_info_reassembled_into_gate_map():
    c = MavlinkClient()
    full = _gate_map_bytes(2)
    c._handle(_handshake(9, packets=2))
    c._handle(_encap_track_chunk(9, seqnr=0, payload=full[:40]))
    assert c.track_gates is None                     # incomplete: hold off
    c._handle(_encap_track_chunk(9, seqnr=1, payload=full[40:]))
    gates = c.track_gates
    assert gates is not None and len(gates) == 2
    assert gates[0]["gate_id"] == 0
    np.testing.assert_array_equal(gates[1]["position_ned"], [1.0, 2.0, 3.0])


def test_track_chunk_without_handshake_is_ignored():
    c = MavlinkClient()
    c._handle(_encap_track_chunk(5, seqnr=0, payload=_gate_map_bytes(1)))  # no handshake first
    assert c.track_gates is None


def test_collision_and_actuator_captured():
    c = MavlinkClient()
    c._handle(_collision(cid=1001, threat=2, impulse=3.5))
    c._handle(_collision(cid=1002, threat=1, impulse=0.2))
    assert len(c.collisions) == 2
    assert c.collisions[0]["id"] == 1001 and c.collisions[0]["threat_level"] == 2
    assert abs(c.collisions[0]["impulse"] - 3.5) < 1e-9
    c._handle(_actuator(motors=(0.1, 0.2, 0.3, 0.4)))
    assert c.actuator_outputs["motors"] == [0.1, 0.2, 0.3, 0.4]


def test_collisions_capped():
    c = MavlinkClient()
    c._max_collisions = 3
    for _ in range(5):
        c._handle(_collision())
    assert len(c.collisions) == 3


def test_sim_dialect_messages_not_unknown():
    c = MavlinkClient()
    for m in (_odometry(), _collision(), _actuator(), _encap_race_status(), _handshake(1, 1)):
        c._handle(m)
    assert c.unknown_msg_types == set()


def test_send_sim_reset_sends_cmd_31000():
    from racer.mavlink_client import MAV_CMD_SIM_RESET
    c = MavlinkClient()
    c.conn = _FakeConn()
    c.send_sim_reset()
    assert c.conn.mav.sent[-1]["command"] == MAV_CMD_SIM_RESET


def test_parse_helpers_pure():
    from racer.mavlink_client import parse_race_status, parse_track_info
    assert parse_race_status(b"\x00") is None              # too short
    gates = parse_track_info(_gate_map_bytes(1))
    assert len(gates) == 1 and gates[0]["gate_id"] == 0
    assert abs(gates[0]["height_m"] - 1.5) < 1e-6
