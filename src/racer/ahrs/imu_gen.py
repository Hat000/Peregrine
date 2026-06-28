"""Synthetic high-g IMU trajectory generator.

Produces body-frame gyroscope and accelerometer readings (specific force) for a
parameterised attitude+angular-rate trajectory with configurable noise, bias, and
saturation. The key design goal is scenarios where LINEAR ACCELERATION IS LARGE
(4-5 g maneuvers), so the accelerometer vector no longer points at -g — exactly where
naive tilt-from-accel breaks and where ESKF accel-gating must outperform classical filters.

Quaternion convention: (w, x, y, z) scalar-FIRST, matches frames.py throughout.

Interface for real twin data (plug-in later on Adroit):
    Build an ``IMUSequence`` directly from recorded HIGHRES_IMU messages:
        seq = IMUSequence(
            t=time_array,           # (N,) seconds
            q_wxyz_gt=None,         # None when GT not available (scoring disabled)
            gyro=gyro_array,        # (N, 3) rad/s body FRD
            accel=accel_array,      # (N, 3) m/s^2 body FRD specific force
            name="real_twin_seq",
        )
    Then pass seq to scripts/benches/ahrs_bench.py instead of a Scenario.
    The twin generates HIGHRES_IMU at ~200 Hz; resample to a constant dt before
    passing in (frame_residual_report.py's IMU stream is the right starting point).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation


GRAVITY_NED = np.array([0.0, 0.0, 9.80665])   # NED m/s^2, down = +Z
GRAVITY = 9.80665   # m/s^2 scalar gravity magnitude (alias for tests/metrics)
G = GRAVITY


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------

@dataclass
class IMUSequence:
    """A sequence of synthetic (or recorded) IMU measurements + optional GT attitude.

    Attributes
    ----------
    t : (N,) float64 seconds, monotonically increasing.
    q_wxyz_gt : (N, 4) float64 body->world quaternion (w,x,y,z); None if GT unavailable.
    gyro : (N, 3) float64 body-frame angular velocity (rad/s, FRD).
    accel : (N, 3) float64 body-frame specific force (m/s^2, FRD); rest reading = [0,0,-g].
    name : label for reporting.
    """
    t: np.ndarray          # (N,)
    q_wxyz_gt: Optional[np.ndarray]  # (N, 4) or None
    gyro: np.ndarray       # (N, 3)
    accel: np.ndarray      # (N, 3)
    name: str = "unnamed"

    def __post_init__(self) -> None:
        N = len(self.t)
        assert self.gyro.shape == (N, 3), f"gyro shape {self.gyro.shape} != ({N},3)"
        assert self.accel.shape == (N, 3), f"accel shape {self.accel.shape} != ({N},3)"
        if self.q_wxyz_gt is not None:
            assert self.q_wxyz_gt.shape == (N, 4), f"q_wxyz_gt shape != ({N},4)"

    @property
    def dt(self) -> float:
        """Nominal timestep (seconds); assumes uniform sampling."""
        if len(self.t) < 2:
            return 0.005
        return float(np.mean(np.diff(self.t)))

    @property
    def N(self) -> int:
        return len(self.t)


# ---------------------------------------------------------------------------
# Scenario enum
# ---------------------------------------------------------------------------

class Scenario(Enum):
    """Pre-defined trajectory scenarios with increasing attitude-estimation difficulty."""
    STATIC_GRAVITY   = auto()   # Hover: accel = [0,0,-g]. All filters should converge quickly.
    CONSTANT_SPIN    = auto()   # Constant yaw rate; no linear acceleration. Tests gyro integration.
    ROLLING_MANEUVER = auto()   # Gentle S-curve; ~0.5 g lateral. Easy but non-trivial.
    HIGH_G_PULL      = auto()   # 4-5 g sustained pull; accel strongly deviates from -g.
                                 # DISCRIMINATING: ESKF gating must beat naive Madgwick/Mahony.
    HIGH_G_RANDOM    = auto()   # Random-axis 4 g bursts + rate noise. Hardest scenario.


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate_imu_sequence(
    scenario: Scenario,
    duration_s: float = 5.0,
    dt: float = 0.005,
    gyro_noise_std: float = 0.01,      # rad/s 1-sigma
    accel_noise_std: float = 0.05,     # m/s^2 1-sigma
    gyro_bias: Optional[np.ndarray] = None,   # (3,) rad/s constant
    accel_bias: Optional[np.ndarray] = None,  # (3,) m/s^2 constant
    gyro_saturation: Optional[float] = None,  # rad/s clip (e.g. 34.9 rad/s = 2000 deg/s)
    accel_saturation: Optional[float] = None, # m/s^2 clip (e.g. 196 = 20 g)
    seed: int = 42,
) -> IMUSequence:
    """Generate a synthetic IMU sequence for the given Scenario.

    GT attitude is tracked as a quaternion (w,x,y,z) body->world (FRD->NED).
    At rest the body FRD -Z axis is aligned with NED +Z (down), so gravity in the
    body frame is [0, 0, -g]: specific_force_body = R^T @ (a_world - g_ned)
    where a_world is the TRUE world-frame kinematic acceleration and g_ned = [0,0,+g].

    Parameters
    ----------
    scenario : which trajectory to generate.
    duration_s : total sequence length.
    dt : timestep; 0.005 s = 200 Hz (HIGHRES_IMU nominal rate from VQ2 spec).
    gyro_noise_std : gyroscope white noise (rad/s per sample).
    accel_noise_std : accelerometer white noise (m/s^2 per sample).
    gyro_bias : constant bias added before noise (rad/s). None = zeros.
    accel_bias : constant bias added before noise (m/s^2). None = zeros.
    gyro_saturation : clip |gyro| channel-wise ±sat after noise. None = no clip.
    accel_saturation : clip |accel| channel-wise ±sat after noise. None = no clip.
    seed : RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    N = int(round(duration_s / dt))
    t = np.arange(N) * dt

    if gyro_bias is None:
        gyro_bias = np.zeros(3)
    if accel_bias is None:
        accel_bias = np.zeros(3)

    # ---- Build true angular-rate profile (body frame) -------------------------
    omega_true = _build_omega_profile(scenario, t)   # (N, 3) rad/s

    # ---- Integrate true attitude quaternion from omega -------------------------
    q_wxyz_gt = _integrate_quaternion(omega_true, dt)  # (N, 4)

    # ---- Build true world linear acceleration profile --------------------------
    a_world_true = _build_accel_profile(scenario, t, q_wxyz_gt)  # (N, 3) NED m/s^2

    # ---- Compute true specific force in body frame ----------------------------
    # specific_force_body = R_wb^T @ (a_world - g_ned)
    # At rest: a_world = 0, so sf_body = -R_wb^T @ g_ned = [0,0,-g] for level hover
    g_ned = GRAVITY_NED
    sf_true = np.zeros((N, 3))
    for i in range(N):
        R_wb = _quat_to_matrix_wxyz(q_wxyz_gt[i])
        sf_true[i] = R_wb.T @ (a_world_true[i] - g_ned)

    # ---- Add noise + bias + saturation ----------------------------------------
    gyro_noisy = omega_true + gyro_bias + rng.normal(0.0, gyro_noise_std, (N, 3))
    accel_noisy = sf_true + accel_bias + rng.normal(0.0, accel_noise_std, (N, 3))

    if gyro_saturation is not None:
        gyro_noisy = np.clip(gyro_noisy, -gyro_saturation, gyro_saturation)
    if accel_saturation is not None:
        accel_noisy = np.clip(accel_noisy, -accel_saturation, accel_saturation)

    return IMUSequence(
        t=t,
        q_wxyz_gt=q_wxyz_gt,
        gyro=gyro_noisy.astype(np.float64),
        accel=accel_noisy.astype(np.float64),
        name=scenario.name,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_omega_profile(scenario: Scenario, t: np.ndarray) -> np.ndarray:
    """Body-frame angular rate (rad/s) profile for each scenario."""
    N = len(t)
    omega = np.zeros((N, 3))

    if scenario == Scenario.STATIC_GRAVITY:
        pass  # Zero rotation

    elif scenario == Scenario.CONSTANT_SPIN:
        # Constant 30 deg/s yaw (body Z axis = yaw in FRD)
        omega[:, 2] = np.deg2rad(30.0)

    elif scenario == Scenario.ROLLING_MANEUVER:
        # S-curve: sinusoidal roll rate + mild yaw
        omega[:, 0] = np.deg2rad(20.0) * np.sin(2 * np.pi * t / 3.0)
        omega[:, 2] = np.deg2rad(15.0) * np.sin(2 * np.pi * t / 5.0)

    elif scenario == Scenario.HIGH_G_PULL:
        # Sustained 3 deg/s pitch-down (nose down pull), ~4.5 g centripetal at speed
        # + gradual yaw to keep it interesting
        omega[:, 1] = np.deg2rad(3.0)
        omega[:, 2] = np.deg2rad(10.0) * np.sin(2 * np.pi * t / 4.0)

    elif scenario == Scenario.HIGH_G_RANDOM:
        # Random-axis rates; amplitude ramps up mid-sequence
        rng_inner = np.random.default_rng(99)
        base = rng_inner.normal(0, np.deg2rad(15.0), (N, 3))
        # High-g bursts in the second half
        burst = np.zeros((N, 3))
        mid = N // 2
        burst[mid:] = rng_inner.normal(0, np.deg2rad(30.0), (N - mid, 3))
        omega = base + burst

    return omega.astype(np.float64)


def _build_accel_profile(
    scenario: Scenario,
    t: np.ndarray,
    q_wxyz: np.ndarray,
) -> np.ndarray:
    """True world-frame kinematic acceleration (NED, m/s^2) for each scenario.

    HIGH_G scenarios inject large centripetal/tangential world acceleration
    that makes the body-frame accel vector deviate significantly from -g.
    """
    N = len(t)
    a_world = np.zeros((N, 3))

    if scenario in (Scenario.STATIC_GRAVITY, Scenario.CONSTANT_SPIN):
        pass  # Zero kinematic acceleration (just gravity in specific force)

    elif scenario == Scenario.ROLLING_MANEUVER:
        # Mild horizontal accel: 0.5 g lateral sinusoid
        a_world[:, 1] = 0.5 * G * np.sin(2 * np.pi * t / 3.0)

    elif scenario == Scenario.HIGH_G_PULL:
        # 4 g centripetal in body X (forward) projected to world via GT attitude
        # This is a pull-up: large body-frame forward/down acceleration
        for i in range(N):
            R_wb = _quat_to_matrix_wxyz(q_wxyz[i])
            # 4g centripetal force in body-frame forward direction
            body_accel = np.array([4.0 * G * np.sin(np.pi * t[i] / 3.0), 0.0, 0.0])
            a_world[i] = R_wb @ body_accel

    elif scenario == Scenario.HIGH_G_RANDOM:
        # Random-axis 4 g bursts: body-frame random directions → world frame
        rng_inner = np.random.default_rng(77)
        for i in range(N):
            R_wb = _quat_to_matrix_wxyz(q_wxyz[i])
            # Burst during second half
            mag = 0.0
            if t[i] > t[-1] * 0.4:
                # Sinusoidal envelope, peaks at 4-5 g
                mag = 4.5 * G * abs(np.sin(2 * np.pi * t[i] / 0.8))
            direction = rng_inner.normal(0, 1, 3)
            norm = np.linalg.norm(direction)
            if norm > 1e-9:
                direction /= norm
            a_world[i] = R_wb @ (mag * direction)

    return a_world.astype(np.float64)


def _integrate_quaternion(omega: np.ndarray, dt: float) -> np.ndarray:
    """Integrate body angular rates to quaternion (w,x,y,z) attitude.

    Starts from level hover: q = [1, 0, 0, 0] (identity = body FRD aligned with NED).
    Uses the closed-form first-order integration:
        q_new = q * exp(omega * dt / 2)
    which is exact for constant omega over each step.
    """
    N = len(omega)
    q = np.zeros((N, 4))
    q_cur = np.array([1.0, 0.0, 0.0, 0.0])   # w, x, y, z

    for i in range(N):
        w_vec = omega[i]
        angle = np.linalg.norm(w_vec) * dt
        if angle > 1e-12:
            axis = w_vec / (np.linalg.norm(w_vec))
            dq = np.array([
                np.cos(angle / 2),
                axis[0] * np.sin(angle / 2),
                axis[1] * np.sin(angle / 2),
                axis[2] * np.sin(angle / 2),
            ])
        else:
            dq = np.array([1.0, 0.0, 0.0, 0.0])
        q_cur = _quat_multiply_wxyz(q_cur, dq)
        norm = np.linalg.norm(q_cur)
        if norm > 1e-12:
            q_cur /= norm
        q[i] = q_cur

    return q


def _quat_multiply_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion product q1 * q2 (Hamilton product, w,x,y,z layout)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _quat_to_matrix_wxyz(q: np.ndarray) -> np.ndarray:
    """Quaternion (w,x,y,z) -> 3x3 rotation matrix (body -> world)."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z),  2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),      1 - 2*(x*x + y*y)],
    ])
