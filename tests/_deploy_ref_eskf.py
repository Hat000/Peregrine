"""tests/_deploy_ref_eskf.py -- VERBATIM numpy reference excerpts of the DEPLOY estimators, vendored
for the estimator-faithful T0 translation-parity tests (no cross-repo import on Adroit/CI).

PROVENANCE: copied verbatim (docstrings trimmed, mag/vision-yaw methods dropped -- DEAD under
vq2_ego_lean and not translated) from the Anduril-ego-deploy repo @ e1aa4d1:
  * src/racer/ahrs/eskf.py      -- ESKFAHRS (helpers :100-145, params :173-275, step :321-346,
                                   _predict :350-377, gates :381-438, _update_accel :440-518,
                                   _apply_eskf_update :606-638)
  * src/racer/state_estimator.py -- LinearKF (:63-175)
DO NOT EDIT except to re-vendor after a deploy-side change (then re-run the parity tests)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

GRAVITY = 9.80665
G_NED = np.array([0.0, 0.0, GRAVITY])


def _quat_multiply_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _quat_to_R_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),  2*(x*y-z*w),    2*(x*z+y*w)],
        [2*(x*y+z*w),    1-2*(x*x+z*z),  2*(y*z-x*w)],
        [2*(x*z-y*w),    2*(y*z+x*w),    1-2*(x*x+y*y)],
    ])


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0., -z,  y],
                     [z,  0., -x],
                     [-y,  x,  0.]])


def _omega_exp_wxyz(omega: np.ndarray, dt: float) -> np.ndarray:
    angle = np.linalg.norm(omega) * dt
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = omega / np.linalg.norm(omega)
    half = angle / 2.0
    return np.array([np.cos(half),
                     axis[0]*np.sin(half),
                     axis[1]*np.sin(half),
                     axis[2]*np.sin(half)])


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([1., 0., 0., 0.])


@dataclass
class ESKFAHRS:
    gyro_noise_std: float = 0.01
    gyro_bias_std: float = 1e-4
    accel_noise_std: float = 0.3
    accel_gate_alpha: float = 10.0
    accel_chi2_thresh: float = 7.815
    accel_freefall_tol_lo: float = 0.75
    accel_freefall_tol_hi: float = 9.0
    use_accel_motion_reject: bool = False
    accel_motion_scale: float = 0.1
    accel_motion_anchor_thr: float = 0.3
    accel_motion_max_inflate: float = 1e4
    mag_ned: Optional[np.ndarray] = None
    mag_noise_std: float = 0.1

    _q: np.ndarray = field(default_factory=lambda: np.array([1., 0., 0., 0.]))
    _b_g: np.ndarray = field(default_factory=lambda: np.zeros(3))
    _P: np.ndarray = field(default_factory=lambda: np.eye(6) * 1e-2)
    _R_ref: np.ndarray = field(default_factory=lambda: np.eye(3))

    def __post_init__(self) -> None:
        self._q = np.array([1., 0., 0., 0.], dtype=np.float64)
        self._b_g = np.zeros(3, dtype=np.float64)
        self._P = np.diag([1e-2]*3 + [1e-6]*3).astype(np.float64)
        self._R_ref = _quat_to_R_wxyz(self._q)

    def reset(self, q_init: Optional[np.ndarray] = None) -> None:
        self._q = _normalize_quat(
            np.asarray(q_init, dtype=np.float64) if q_init is not None
            else np.array([1., 0., 0., 0.])
        )
        self._b_g = np.zeros(3)
        self._P = np.diag([1e-2]*3 + [1e-6]*3).astype(np.float64)
        self._R_ref = _quat_to_R_wxyz(self._q)

    def step(self, gyro, accel, dt, mag_body=None) -> np.ndarray:
        if dt <= 0:
            return self._q.copy()
        self._predict(np.asarray(gyro, dtype=np.float64), dt)
        self._update_accel(np.asarray(accel, dtype=np.float64))
        return self._q.copy()

    def _predict(self, gyro: np.ndarray, dt: float) -> None:
        omega_corrected = gyro - self._b_g
        dq = _omega_exp_wxyz(omega_corrected, dt)
        self._q = _normalize_quat(_quat_multiply_wxyz(self._q, dq))
        if self.use_accel_motion_reject:
            self._R_ref = self._R_ref @ _quat_to_R_wxyz(dq)
        F = np.eye(6)
        F[:3, :3] = np.eye(3) - _skew(omega_corrected) * dt
        F[:3, 3:] = -np.eye(3) * dt
        sigma_phi = self.gyro_noise_std * np.sqrt(dt)
        sigma_bg = self.gyro_bias_std * np.sqrt(dt)
        Q = np.diag([sigma_phi**2]*3 + [sigma_bg**2]*3)
        self._P = F @ self._P @ F.T + Q

    def _accel_gate_weight(self, accel_mag: float) -> float:
        if self.accel_gate_alpha <= 0.0:
            return 1.0
        deviation = (accel_mag - GRAVITY) / GRAVITY
        return float(np.exp(-self.accel_gate_alpha * deviation**2))

    def _accel_magnitude_in_band(self, accel_mag: float) -> bool:
        if self.accel_freefall_tol_lo >= 0.0:
            lo = GRAVITY * (1.0 - self.accel_freefall_tol_lo)
            if accel_mag < lo:
                return False
        if self.accel_freefall_tol_hi >= 0.0:
            hi = GRAVITY * (1.0 + self.accel_freefall_tol_hi)
            if accel_mag > hi:
                return False
        return True

    def _accel_motion_inflation(self, accel: np.ndarray) -> float:
        if not self.use_accel_motion_reject:
            return 1.0
        a_lin = self._R_ref @ accel + G_NED
        a_lin_mag = float(np.linalg.norm(a_lin))
        if a_lin_mag < self.accel_motion_anchor_thr:
            self._R_ref = _quat_to_R_wxyz(self._q)
        scale = max(self.accel_motion_scale, 1e-9)
        factor = 1.0 + (a_lin_mag / scale) ** 2
        return float(min(factor, self.accel_motion_max_inflate))

    def _update_accel(self, accel: np.ndarray) -> None:
        accel_mag = float(np.linalg.norm(accel))
        if accel_mag < 1e-6:
            return
        if not self._accel_magnitude_in_band(accel_mag):
            return
        gate = self._accel_gate_weight(accel_mag)
        if gate < 1e-4:
            return
        R_wb = _quat_to_R_wxyz(self._q)
        g_body = R_wb.T @ G_NED
        g_hat = g_body / max(np.linalg.norm(g_body), 1e-9)
        h_hat = -g_hat
        a_hat = accel / accel_mag
        innovation = a_hat - h_hat
        H = np.zeros((3, 6))
        H[:, :3] = -_skew(g_hat)
        sigma_a = self.accel_noise_std / GRAVITY
        R_meas = (sigma_a**2 / gate) * self._accel_motion_inflation(accel) * np.eye(3)
        if self.accel_chi2_thresh > 0.0:
            S = H @ self._P @ H.T + R_meas
            try:
                md = float(innovation @ np.linalg.solve(S, innovation))
            except np.linalg.LinAlgError:
                return
            if md > self.accel_chi2_thresh:
                return
        self._apply_eskf_update(innovation, H, R_meas)

    def _apply_eskf_update(self, innovation, H, R_meas) -> None:
        PHt = self._P @ H.T
        S = H @ PHt + R_meas
        try:
            K = np.linalg.solve(S, PHt.T).T
        except np.linalg.LinAlgError:
            return
        delta_x = K @ innovation
        delta_phi = delta_x[:3]
        delta_b_g = delta_x[3:]
        angle = np.linalg.norm(delta_phi)
        if angle > 1e-12:
            dq = _omega_exp_wxyz(delta_phi, 1.0)
        else:
            dq = np.array([1., 0., 0., 0.])
        self._q = _normalize_quat(_quat_multiply_wxyz(self._q, dq))
        self._b_g = self._b_g + delta_b_g
        I_KH = np.eye(6) - K @ H
        self._P = I_KH @ self._P @ I_KH.T + K @ R_meas @ K.T


# ---------------------------------------------------------------------------
# state_estimator.py LinearKF (verbatim; baro/velocity update methods dropped -- unused here)
# ---------------------------------------------------------------------------
_I3 = np.eye(3)
_Z3 = np.zeros((3, 3))
_H_POS = np.hstack([_I3, _Z3])
ATTITUDE_NOISE_STD_RAD = float(np.deg2rad(1.4))
GRAVITY_NED = np.array([0.0, 0.0, GRAVITY])


@dataclass
class LinearKF:
    x: np.ndarray
    P: np.ndarray
    accel_noise_std: float = 0.3
    attitude_noise_std: float = ATTITUDE_NOISE_STD_RAD
    gravity_ned: np.ndarray = field(default_factory=lambda: GRAVITY_NED.copy())
    process_floor: float = 1e-6
    max_dt_s: float = 0.2
    inplane_pos_floor_std: float = 0.0

    @classmethod
    def initialize(cls, position_ned, velocity_ned=None, pos_std: float = 1.0,
                   vel_std: float = 1.0, **kw) -> "LinearKF":
        x = np.zeros(6)
        x[:3] = np.asarray(position_ned, dtype=np.float64)
        if velocity_ned is not None:
            x[3:] = np.asarray(velocity_ned, dtype=np.float64)
        P = np.diag([pos_std**2] * 3 + [vel_std**2] * 3).astype(np.float64)
        return cls(x=x, P=P, **kw)

    def predict(self, accel_body, R_world_body, dt) -> None:
        if dt <= 0 or dt > self.max_dt_s:
            return
        specific_force_world = R_world_body @ np.asarray(accel_body, dtype=np.float64)
        a_world = specific_force_world + self.gravity_ned
        F = np.block([[_I3, dt * _I3], [_Z3, _I3]])
        B = np.vstack([0.5 * dt * dt * _I3, dt * _I3])
        self.x = F @ self.x + B @ a_world
        S = _skew(specific_force_world)
        accel_cov = self.accel_noise_std**2 * _I3 + self.attitude_noise_std**2 * (S @ S.T)
        Q = B @ accel_cov @ B.T + self.process_floor * np.eye(6)
        self.P = F @ self.P @ F.T + Q

    def update(self, z, H, R) -> None:
        z = np.asarray(z, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        R = np.asarray(R, dtype=np.float64)
        y = z - H @ self.x
        PHt = self.P @ H.T
        S = H @ PHt + R
        K = np.linalg.solve(S, PHt.T).T
        self.x = self.x + K @ y
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self._apply_inplane_pos_floor()

    def _apply_inplane_pos_floor(self) -> None:
        floor_var = self.inplane_pos_floor_std ** 2
        if floor_var <= 0.0:
            return
        block = 0.5 * (self.P[:2, :2] + self.P[:2, :2].T)
        w, V = np.linalg.eigh(block)
        if w[0] >= floor_var:
            return
        w = np.maximum(w, floor_var)
        self.P[:2, :2] = (V * w) @ V.T

    def update_position(self, position_ned, covariance) -> None:
        self.update(np.asarray(position_ned, dtype=np.float64), _H_POS, covariance)

    @property
    def position(self) -> np.ndarray:
        return self.x[:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:].copy()
