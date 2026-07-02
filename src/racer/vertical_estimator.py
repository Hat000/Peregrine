"""VerticalEstimator — the 1-D vertical-channel (altitude + vertical-velocity) filter for the
state-denied VQ2 wire (the A21 egress->ceiling-climb fix, 2026-07-02).

THE BUG THIS KILLS (run 20260702_040036, quantified via scripts/analyze_flight.py): the wire is
POSITION/VELOCITY DENIED, so the 6-state KF's z is dead-reckoning between the sparse (~a few Hz,
latent) vision ``floor_height`` pins — and each accepted pin is a TIGHT z fix, so ``est_z``
TELEPORTS (−1.435 m in ONE tick at t=1569.41) while the true motion is smooth. The controller's
ff-owns-vertical damper reads a low-passed finite difference of that z: the teleport becomes a
~−4.4 m/s vz spike -> a thrust slam -> and between pins the damper is BLIND to the real climb
(integrated a_up shows +4-5 m/s upward over 2.7 s that est_z never tracked) -> the drone climbs
over gate 0 into the ceiling. Smooth true motion vs a step-function estimate = the bug.

THE FIX is the classic 1-D baro-inertial vertical channel, on WIRE-AVAILABLE signals ONLY:
integrate the IMU-derived kinematic upward acceleration ``a_up`` (fast ~140 Hz, smooth, but
drifting) for the high-frequency vz/altitude, and correct the slow drift with the absolute
``floor_height`` pins when they arrive. A tiny 3-state linear KF

    x = [z, vz, b]      z  : NED world-down position (m)
                        vz : NED world-down velocity zdot (m/s)
                        b  : residual NED-down acceleration bias (m/s^2)

    zddot = -a_up + b   (a_up is UP-positive; NED z is DOWN-positive)

rather than a fixed-gain complementary filter, because the pins arrive at IRREGULAR, sometimes
long intervals (decimation + detect starvation): the covariance-tracked gain is automatically
small at the design pin rate (a bounded, incremental correction — never a step teleport) and
grows honestly through a pin drought. The bias state soaks the residual of the attitude
projection (the AHRS tilt error rotates gravity into a_up; the airframe rests at ~−17.8 deg
pitch, so a small pitch error is a sustained a_up offset) so vz stays honest between pins
instead of carrying a constant drift. It is deliberately NOT a 3-D filter and deliberately NOT
fused into the 6-state KF: the vertical damper needs ONE smooth (altitude, vz) pair, and this
owns exactly that.

ACCEL DECODE (VERIFIED empirically 2026-07-01, docs/accel_investigation.md + analyze_flight.py):
HIGHRES_IMU accel is SPECIFIC FORCE in body FRD, gravity-INCLUDED (on-pad |f| = 9.810 exactly),
so the kinematic upward acceleration is  a_up = -( (R_body->ned @ f)_z + g ).

Consumed by the Navigator behind ``NavigatorConfig.use_vertical_estimator`` (default OFF =
byte-identical); exported on ``NavState.vert_z_est`` / ``vert_vz_est``; the controller's
ff-owns-vertical alt-hold damps on the exported vz behind ``Controller.use_vertical_estimator``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_G = 9.80665


def a_up_from_specific_force(f_body: np.ndarray, R_wb: np.ndarray) -> float:
    """Kinematic UPWARD acceleration (m/s^2) from a specific-force sample + the body->NED rotation.

    ``a_ned = R_wb @ f + g_ned`` with ``g_ned = (0, 0, +G)`` (NED z is DOWN), so ``a_up = -a_ned_z``.
    The matrix form of the VERIFIED analyze_flight.py decode — the Navigator has the TRUE R_wb
    (AHRS) in hand, so no euler round-trip. At rest (attitude correct) this reads ~0.
    """
    a_ned_z = float(np.asarray(R_wb, dtype=np.float64)[2] @ np.asarray(f_body, dtype=np.float64)) + _G
    return -a_ned_z


@dataclass
class VerticalEstimator:
    """Tiny 3-state [z, vz, bias] vertical-channel KF: a_up predicts, floor_height pins correct.

    Call ``seed(z0)`` once (the pad z the 6-state KF seeded with), then ``predict(a_up, dt)`` per
    IMU tick and ``update_z(z_meas, var_z)`` per ACCEPTED floor pin. Read ``z`` / ``vz`` (NED,
    down +). Un-seeded, every call no-ops and the outputs are NaN — the consumer's absent-marker.
    """

    # Process noise. accel_noise_std covers the white part (IMU noise + tick-scale attitude
    # jitter through the R projection). bias_rw_std lets the bias state FOLLOW the slow part
    # (AHRS tilt error rotating gravity into a_up — the sustained offset a white-noise-only
    # model would smear into vz). Both feed Q each predict; sized so the z gain at the design
    # ~5 Hz pin rate stays well under 0.5 (incremental corrections, see test).
    accel_noise_std: float = 0.4       # m/s^2, white accel-channel noise 1-sigma
    bias_rw_std: float = 0.3           # m/s^2 / sqrt(s), bias random-walk density
    # Seed 1-sigmas: on the pad z IS the seed datum (the same origin the 6-state KF seeds — known
    # tightly, so the first pin is an incremental nudge, not a half-step), vz is 0 (static), the
    # bias is the unknown initial attitude-projection residual (the AHRS is cold at arm).
    init_z_std: float = 0.15           # m
    init_vz_std: float = 0.1           # m/s
    init_bias_std: float = 0.5         # m/s^2
    # Reject implausibly large predict steps (sim reset / stutter) — mirrors LinearKF.max_dt_s.
    max_dt_s: float = 0.2

    _x: np.ndarray | None = field(default=None, repr=False)   # (3,) [z, vz, b]
    _P: np.ndarray | None = field(default=None, repr=False)   # (3,3)

    # -- lifecycle ----------------------------------------------------------
    def seed(self, z0: float) -> None:
        """(Re)initialise at altitude ``z0`` (NED down +), at rest, with the seed covariance."""
        self._x = np.array([float(z0), 0.0, 0.0])
        self._P = np.diag([self.init_z_std**2, self.init_vz_std**2, self.init_bias_std**2])

    @property
    def seeded(self) -> bool:
        return self._x is not None

    @property
    def z(self) -> float:
        """Estimated NED world-down position (m); NaN until seeded."""
        return float(self._x[0]) if self._x is not None else float("nan")

    @property
    def vz(self) -> float:
        """Estimated NED world-down velocity zdot (m/s); NaN until seeded."""
        return float(self._x[1]) if self._x is not None else float("nan")

    # -- per-IMU-tick propagation --------------------------------------------
    def predict(self, a_up: float, dt: float) -> None:
        """Integrate one accel sample: ``zddot = -a_up + b`` (a_up UP-positive, z DOWN-positive).

        Exact constant-accel discretisation; Q = white-accel + bias-random-walk. ``dt <= 0`` (a
        between-IMU control tick) and ``dt > max_dt_s`` (sim reset / stutter — integrating a
        garbage step corrupts vz worse than skipping one) are no-ops, mirroring LinearKF."""
        if self._x is None or not (0.0 < dt <= self.max_dt_s):
            return
        z, vz, b = self._x
        a = -float(a_up) + b
        self._x = np.array([z + vz * dt + 0.5 * a * dt * dt, vz + a * dt, b])
        F = np.array([[1.0, dt, 0.5 * dt * dt],
                      [0.0, 1.0, dt],
                      [0.0, 0.0, 1.0]])
        qa = self.accel_noise_std**2
        Q = np.array([[qa * dt**4 / 4.0, qa * dt**3 / 2.0, 0.0],
                      [qa * dt**3 / 2.0, qa * dt**2,       0.0],
                      [0.0,              0.0,              self.bias_rw_std**2 * dt]])
        self._P = F @ self._P @ F.T + Q

    # -- per-accepted-floor-pin correction ------------------------------------
    def update_z(self, z_meas: float, var_z: float) -> None:
        """Absolute-altitude correction from an ACCEPTED floor_height pin (H = [1, 0, 0]).

        The Kalman gain BOUNDS the correction (the anti-teleport property): at the design pin
        rate P_z stays small vs the pin variance, so each pin nudges z/vz/bias incrementally;
        after a drought the gain honestly grows with the propagated uncertainty. Joseph-form
        covariance update for symmetry/PSD hygiene."""
        if self._x is None:
            return
        S = float(self._P[0, 0]) + float(var_z)
        if S <= 0.0:
            return
        K = self._P[:, 0] / S                                  # (3,) gain column
        self._x = self._x + K * (float(z_meas) - self._x[0])
        IKH = np.eye(3)
        IKH[:, 0] -= K
        self._P = IKH @ self._P @ IKH.T + np.outer(K, K) * float(var_z)
