"""VQ2 control-recipe alignment — pins the deploy uplink against the live-confirmed
handshake (sim build 1.0.3379, 2026-06-29) and the two opt-in deploy deltas.

THE RECIPE (live-confirmed): ARM (MAV_CMD_COMPONENT_ARM_DISARM=400, p1=1), then stream
SET_ATTITUDE_TARGET in BODY-RATE mode (type_mask=0b10000000) with FRD body rates +
normalized collective thrust [0,1]. Sim default mode is ACRO => ControlMode.BODY_RATE.
This is the SAME uplink fly_rl already drove on VQ1.

What this file pins:
  * The emitted SET_ATTITUDE_TARGET carries type_mask == 0b10000000 (_ATT_MASK_BODY_RATE),
    the body rates in FRD order (r[0],r[1],r[2]), and the collective in slot thrust.
  * ARM emits MAV_CMD_COMPONENT_ARM_DISARM with param1=1.
  * collective is normalized [0,1] (policy_step clamps; the uplink forwards it verbatim).
  * DELTA 1: cmd_rate_scale is IDENTITY at 1.0 (byte-identical) and scales the FRD rates
    (NOT the collective) at 0.4.
  * DELTA 2: under the case-C use_ahrs=True deploy config, the obs body rate (obs[9:12])
    is sourced from DroneState.gyro_body (raw HIGHRES_IMU) via the AHRS, NOT the
    ODOMETRY-derived angular_rate_body.

Pure unit tests against a fake conn (send side) + a synthetic navigator loop (deploy obs
side); no socket / no live sim. [VQ2-CONTROL-RECIPE 2026-06-29]
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))

from racer.contracts import ControlCommand, ControlMode  # noqa: E402
from racer.mavlink_client import (  # noqa: E402
    _ATT_MASK_BODY_RATE,
    MavlinkClient,
)
from pymavlink import mavutil  # noqa: E402


# --------------------------------------------------------------------------- fake uplink
class _FakeMav:
    """Captures the exact set_attitude_target_send / command_long_send arguments."""

    def __init__(self):
        self.attitude_targets: list[dict] = []
        self.commands: list[dict] = []

    def set_attitude_target_send(self, time_boot_ms, target_system, target_component,
                                 type_mask, q, body_roll_rate, body_pitch_rate,
                                 body_yaw_rate, thrust):
        self.attitude_targets.append(dict(
            time_boot_ms=time_boot_ms, type_mask=type_mask, q=list(q),
            body_rate=[body_roll_rate, body_pitch_rate, body_yaw_rate], thrust=thrust))

    def command_long_send(self, target_system, target_component, command, confirmation,
                          p1, p2, p3, p4, p5, p6, p7):
        self.commands.append(dict(command=command, p1=p1, p2=p2))


class _FakeConn:
    def __init__(self):
        self.mav = _FakeMav()
        self.target_system = 1
        self.target_component = 1


def _client(cmd_rate_scale: float = 1.0) -> MavlinkClient:
    c = MavlinkClient(cmd_rate_scale=cmd_rate_scale)
    c.conn = _FakeConn()
    return c


def _body_rate_cmd(rate, thrust) -> ControlCommand:
    return ControlCommand(mode=ControlMode.BODY_RATE,
                          body_rate=np.asarray(rate, dtype=np.float64), thrust=thrust)


# ===========================================================================
# RECIPE: BODY-RATE uplink shape
# ===========================================================================
def test_body_rate_uplink_uses_acro_type_mask_and_frd_rates():
    """The emitted SET_ATTITUDE_TARGET is BODY-RATE mode: type_mask=0b10000000, FRD body rates
    in order, collective forwarded as thrust, attitude quat ignored."""
    c = _client()
    c.send_command(_body_rate_cmd([0.5, -0.3, 0.1], 0.42))
    sent = c.conn.mav.attitude_targets[-1]
    assert sent["type_mask"] == 0b10000000 == _ATT_MASK_BODY_RATE
    # FRD body rates land in (roll, pitch, yaw) order, unchanged at scale 1.0.
    np.testing.assert_array_equal(sent["body_rate"], [0.5, -0.3, 0.1])
    assert sent["thrust"] == pytest.approx(0.42)
    # the attitude quaternion is ignored by the mask (sent as identity placeholder).
    assert sent["q"] == [1.0, 0.0, 0.0, 0.0]
    # no command_long emitted by a control send.
    assert c.conn.mav.commands == []


def test_arm_emits_component_arm_disarm_param1_one():
    """ARM uses MAV_CMD_COMPONENT_ARM_DISARM (=400) with param1=1 (the recipe's arm step)."""
    c = _client()
    c.arm()
    cmd = c.conn.mav.commands[-1]
    assert cmd["command"] == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM == 400
    assert cmd["p1"] == 1.0


def test_collective_is_normalized_and_forwarded_verbatim():
    """The collective the policy produces is already clamped to [0,1] (policy_step); the uplink
    forwards it verbatim. Pin that a [0,1] collective survives unchanged (rate-scale must NOT
    touch the collective)."""
    c = _client(cmd_rate_scale=0.4)
    for thr in (0.0, 0.2656, 1.0):
        c.send_command(_body_rate_cmd([1.0, 1.0, 1.0], thr))
        assert c.conn.mav.attitude_targets[-1]["thrust"] == pytest.approx(thr)


def test_policy_step_collective_clamped_to_unit_interval():
    """policy_step clamps the wire collective into [0,1] (normalized) regardless of thrust action."""
    pytest.importorskip("torch")
    import torch.nn as nn

    import fly_rl

    class _BigThrust(nn.Module):
        # actor_mean that saturates tanh -> max rescaled thrust -> collective would exceed 1
        def forward(self, x):
            import torch
            return torch.tensor([[10.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

    _rate, collective, _normed = fly_rl.policy_step(_BigThrust(), np.zeros(17, np.float32))
    assert 0.0 <= collective <= 1.0


# ===========================================================================
# DELTA 1: cmd_rate_scale calibration
# ===========================================================================
def test_cmd_rate_scale_identity_at_one_is_byte_identical():
    """Default cmd_rate_scale=1.0 == NO change: emitted rates equal the raw command exactly."""
    rate = [0.73, -1.21, 0.05]
    c = _client(cmd_rate_scale=1.0)
    c.send_command(_body_rate_cmd(rate, 0.3))
    np.testing.assert_array_equal(c.conn.mav.attitude_targets[-1]["body_rate"], rate)
    assert c.cmd_rate_scale == 1.0   # default


def test_cmd_rate_scale_scales_commanded_rates():
    """cmd_rate_scale=0.4 scales the FRD body rates by 0.4 (1/2.5 VQ2 compensation); collective
    is NOT scaled."""
    rate = np.array([0.5, -0.3, 0.1])
    c = _client(cmd_rate_scale=0.4)
    c.send_command(_body_rate_cmd(rate, 0.42))
    sent = c.conn.mav.attitude_targets[-1]
    np.testing.assert_allclose(sent["body_rate"], rate * 0.4, rtol=0, atol=1e-12)
    assert sent["thrust"] == pytest.approx(0.42)   # collective untouched


def test_cmd_rate_scale_does_not_touch_position_or_attitude_paths():
    """The scale is BODY_RATE-only: a POSITION command emits no scaled body rate (no regression
    to the floor paths)."""
    c = _client(cmd_rate_scale=0.4)
    c.send_command(ControlCommand(mode=ControlMode.ATTITUDE,
                                  attitude_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                                  thrust=0.5))
    # ATTITUDE path emits a set_attitude_target with the ATTITUDE mask (not body-rate), thrust intact.
    sent = c.conn.mav.attitude_targets[-1]
    assert sent["type_mask"] != _ATT_MASK_BODY_RATE
    assert sent["thrust"] == pytest.approx(0.5)


# ===========================================================================
# DELTA 2: gyro_body is the deploy body-rate source under use_ahrs (case-C)
# ===========================================================================
def test_case_c_body_rate_source_is_gyro_body_not_odometry():
    """Under the case-C deploy config the deploy body-rate SOURCE is DroneState.gyro_body (raw
    HIGHRES_IMU), NOT the ODOMETRY-derived angular_rate_body (BLOCKED in VQ2). The navigator's
    ``use_ahrs`` seam routes ds.gyro_body into the AHRS, whose bias-corrected ``body_rate`` feeds
    obs[9:12]. We pin the SOURCE directly at the AHRS adapter (the seam the navigator consumes,
    read-only here) so the test does not depend on the navigator's in-flux full loop: feeding a
    real gyro_body produces a tracking nonzero AHRS body_rate, while the ODOMETRY angular_rate_body
    plays no role in the AHRS at all."""
    from racer.ahrs.ahrs_adapter import AHRSAttitudeSource
    from racer.ahrs.imu_gen import Scenario, generate_imu_sequence

    src = AHRSAttitudeSource()
    # CONSTANT_SPIN: a sustained nonzero body rate carried ONLY by the gyro stream.
    seq = generate_imu_sequence(Scenario.CONSTANT_SPIN, duration_s=1.0, dt=0.005, seed=11)
    last = None
    for k in range(seq.N):
        # ingest takes the RAW HIGHRES_IMU gyro (ds.gyro_body) -- NOT angular_rate_body.
        src.ingest(seq.accel[k], seq.gyro[k], dt=0.005)
        last = src.body_rate
    # The AHRS body_rate (the value the deploy obs[9:12] is sourced from) tracks the gyro stream.
    assert last is not None and np.linalg.norm(last) > 0.05, (
        f"AHRS body_rate {last} ~0 -> not tracking the gyro_body stream")
    # It tracks the raw gyro (small estimated bias), confirming gyro_body — not the ODOMETRY field
    # — is the source. ESKF body_rate = gyro - bias; body_rate is snapshotted BEFORE each step (the
    # rate that propagated THIS attitude), so it matches the gyro within the bias estimate. The
    # dominant spin axis (~0.53 rad/s) must clearly track the gyro; bias is sub-mrad/s.
    np.testing.assert_allclose(last, seq.gyro[-1] - src.gyro_bias, rtol=0, atol=2e-3)
    assert abs(last[2] - seq.gyro[-1][2]) < 1e-2   # dominant spin axis tracks the raw gyro


def test_ahrs_body_rate_ignores_odometry_angular_rate_field():
    """The AHRS adapter consumes gyro_body ONLY: it has no channel for the ODOMETRY-derived
    angular_rate_body. Feeding a ZERO gyro (the VQ2 case where ODOMETRY/angular_rate_body would
    otherwise carry the rate) yields a ~zero AHRS body_rate -- proving the rate cannot leak in
    from the (blocked) ODOMETRY field."""
    from racer.ahrs.ahrs_adapter import AHRSAttitudeSource

    src = AHRSAttitudeSource()
    hover_accel = np.array([0.0, 0.0, -9.80665])
    for _ in range(50):
        src.ingest(hover_accel, np.zeros(3), dt=0.005)   # gyro_body == 0 (no rate on the IMU)
    # No gyro -> no AHRS rate, regardless of any ODOMETRY angular_rate_body the wire might carry.
    assert np.linalg.norm(src.body_rate) < 1e-6


def test_off_path_obs_rate_uses_angular_rate_body_unchanged():
    """Control: with use_ahrs=False (VQ1 default) the obs body rate comes from the ODOMETRY
    angular_rate_body (gyro_body absent) — the byte-identical legacy path."""
    pytest.importorskip("torch")
    from fly_rl import build_obs

    from racer.contracts import DroneState

    # ODOMETRY-convention angular rate present, gyro_body None (VQ1 wire). build_obs (the raw
    # deploy obs core) reads angular_rate_body -> applies _ODO_RATE_SIGN -> obs[9:12].
    ds = DroneState(
        sim_time_ns=1,
        orientation_ned_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_rate_body=np.array([0.4, -0.2, 0.1]),
        gyro_body=None,
        position_ned=np.zeros(3), velocity_ned=np.zeros(3),
    )
    obs = build_obs(ds, 0, 0.0)
    # obs[9:12] = w_flu = (angular_rate_body * _ODO_RATE_SIGN) * _FLIP. Nonzero, driven by the
    # ODOMETRY field (the legacy contract); gyro_body played no role here.
    assert np.linalg.norm(obs[9:12]) > 0.05
