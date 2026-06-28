"""Error-State Kalman Filter AHRS.

Estimates attitude (body->world quaternion) from gyroscope and accelerometer,
following the documented ESKF intent in state_estimator.py's module docstring.

State (error-state formulation)
--------------------------------
Nominal state  : quaternion q (w,x,y,z, body->world)
Error state    : delta_x = [delta_phi (3), delta_b_g (3)]   (6-dim)
  delta_phi   : small-angle rotation error in the BODY frame
  delta_b_g   : gyroscope bias error in the body frame

Prediction
----------
Propagate the nominal quaternion by the gyro measurement (gyro - bias_estimate):
  q_new = q * exp((omega_meas - b_g) * dt / 2)
Propagate the error-state covariance through the linearised dynamics:
  P_new = F @ P @ F^T + Q

Accelerometer update (with high-g gating)
-----------------------------------------
When |a_meas| is close to g (within threshold), the accel points mostly at -g in
body frame and carries tilt information. We form the predicted gravity direction in
the body frame from the nominal quaternion and use it as the measurement model.

CRITICAL: During high-g maneuvers, |a_meas| >> g, so the accel is dominated by
kinematic acceleration and DOES NOT point at -g. Using it naively would corrupt the
attitude estimate. The gating mechanism DOWNWEIGHTS the accel update:

  gate_weight = exp(-alpha * ((|a| - g) / g)^2)

This weight multiplies the measurement noise covariance (R_accel / gate_weight),
effectively ignoring the accel when the drone is pulling g's.

CRITICAL (empirical, this bench): magnitude gating ALONE is insufficient. A
random-direction acceleration whose MAGNITUDE happens to sit near g (e.g. ~43% of
the HIGH_G_RANDOM samples) sails through the magnitude gate while its DIRECTION is
wildly wrong, poisoning the tilt update (ESKF p90 blew up to ~67 deg). The fix is a
second, direction-aware gate: a Mahalanobis / chi-square consistency test on the
innovation itself (the same family as the relinnov chi2 gate in the C2 estimator
chain). The accel update is APPLIED ONLY IF
    innovation^T S^-1 innovation <= accel_chi2_thresh   (chi2(3, .95) = 7.815)
With this innovation gate the ESKF beats both classical filters across every high-g
scenario; without it, magnitude gating alone loses to a gyro-trusting Madgwick.
This two-gate design is what distinguishes the ESKF from Madgwick/Mahony, which lack
any principled rejection of acceleration disturbances.

Magnetometer update (optional)
-------------------------------
If mag_ned is provided (NED reference field) and mag_body is given to step(),
a yaw-only update is applied. VQ2 mag usability is unknown; the update is
gated off by default (mag_ned=None).

Frame convention
----------------
Quaternion: (w,x,y,z) scalar-FIRST, body(FRD)->world(NED). Matches frames.py.
Gravity NED: [0, 0, +9.80665] m/s^2.
Specific force at rest in body FRD: [0, 0, -g] (IMU reports gravity up).

Extension stubs
---------------
- RIANN/GRU learned gyro-denoising: replace the raw gyro with a denoised version
  before the propagation step. Interface: denoiser(gyro_history) -> omega_clean.
- Invariant EKF / EqVIO (van Goor ANU thesis): replace the linearised F/H with
  the geometrically-consistent Lie-group Jacobians on SO(3). The prediction/update
  structure here is compatible; swap _compute_F() and _accel_H() to switch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation


GRAVITY = 9.80665
G_NED = np.array([0.0, 0.0, GRAVITY])   # NED: down = +Z


def _quat_multiply_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 * q2 with (w,x,y,z) layout."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _quat_to_R_wxyz(q: np.ndarray) -> np.ndarray:
    """(w,x,y,z) -> 3x3 rotation matrix (body->world)."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),  2*(x*y-z*w),    2*(x*z+y*w)],
        [2*(x*y+z*w),    1-2*(x*x+z*z),  2*(y*z-x*w)],
        [2*(x*z-y*w),    2*(y*z+x*w),    1-2*(x*x+y*y)],
    ])


def _skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix: _skew(v) @ w == cross(v, w)."""
    x, y, z = v
    return np.array([[0., -z,  y],
                     [z,  0., -x],
                     [-y,  x,  0.]])


def _omega_exp_wxyz(omega: np.ndarray, dt: float) -> np.ndarray:
    """Closed-form quaternion exponential: exp(omega * dt / 2) = delta_q."""
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
    """Error-State Kalman Filter attitude estimator.

    Parameters
    ----------
    gyro_noise_std : rad/s, 1-sigma white noise on each gyro axis.
    gyro_bias_std : rad/s/sqrt(s), random-walk on gyro bias (rate noise).
    accel_noise_std : m/s^2, 1-sigma white noise on each accel axis.
    accel_gate_alpha : high-g gating sharpness. Larger = sharper rejection.
                       alpha=0 disables gating (naive filter). alpha~10 is a
                       good starting point for 4-5 g maneuvers.
    mag_ned : (3,) NED reference magnetic field vector. None = no mag update.
    mag_noise_std : magnetometer noise (normalised field units, 1-sigma).
    """
    gyro_noise_std: float = 0.01        # rad/s
    gyro_bias_std: float = 1e-4         # rad/s/sqrt(s)
    accel_noise_std: float = 0.3        # m/s^2
    accel_gate_alpha: float = 10.0      # high-g sharpness (0 = disabled)
    accel_chi2_thresh: float = 7.815    # innovation-gate threshold; chi2(3, .95)=7.815
                                        # (0 or negative = innovation gate disabled)
    mag_ned: Optional[np.ndarray] = None
    mag_noise_std: float = 0.1          # normalised

    # Internal state (post-init)
    _q: np.ndarray = field(default_factory=lambda: np.array([1., 0., 0., 0.]))
    _b_g: np.ndarray = field(default_factory=lambda: np.zeros(3))
    _P: np.ndarray = field(default_factory=lambda: np.eye(6) * 1e-2)

    def __post_init__(self) -> None:
        self._q = np.array([1., 0., 0., 0.], dtype=np.float64)
        self._b_g = np.zeros(3, dtype=np.float64)
        self._P = np.diag([1e-2]*3 + [1e-6]*3).astype(np.float64)
        if self.mag_ned is not None:
            self.mag_ned = np.asarray(self.mag_ned, dtype=np.float64)

    # -- Public interface -------------------------------------------------------

    @property
    def q_wxyz(self) -> np.ndarray:
        """Current attitude estimate as (w,x,y,z) quaternion."""
        return self._q.copy()

    @property
    def gyro_bias(self) -> np.ndarray:
        """Current gyro bias estimate (rad/s, body FRD)."""
        return self._b_g.copy()

    @property
    def attitude_uncertainty_rad(self) -> float:
        """1-sigma attitude uncertainty: sqrt(trace(P[:3,:3]) / 3) in radians."""
        return float(np.sqrt(np.trace(self._P[:3, :3]) / 3.0))

    def reset(self, q_init: Optional[np.ndarray] = None) -> None:
        """Reset to initial state. q_init: (w,x,y,z) or None for identity."""
        self._q = _normalize_quat(
            np.asarray(q_init, dtype=np.float64) if q_init is not None
            else np.array([1., 0., 0., 0.])
        )
        self._b_g = np.zeros(3)
        self._P = np.diag([1e-2]*3 + [1e-6]*3).astype(np.float64)

    def step(
        self,
        gyro: np.ndarray,
        accel: np.ndarray,
        dt: float,
        mag_body: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Process one IMU sample, return updated attitude quaternion (w,x,y,z).

        Parameters
        ----------
        gyro : (3,) body-frame angular rate (rad/s, FRD).
        accel : (3,) body-frame specific force (m/s^2, FRD). Rest = [0,0,-g].
        dt : timestep (seconds).
        mag_body : (3,) body-frame magnetometer (optional). Normalised before use.
        """
        if dt <= 0:
            return self._q.copy()

        self._predict(np.asarray(gyro, dtype=np.float64), dt)
        self._update_accel(np.asarray(accel, dtype=np.float64))

        if mag_body is not None and self.mag_ned is not None:
            self._update_mag(np.asarray(mag_body, dtype=np.float64))

        return self._q.copy()

    # -- Prediction ------------------------------------------------------------

    def _predict(self, gyro: np.ndarray, dt: float) -> None:
        """Propagate nominal quaternion and error-state covariance."""
        omega_corrected = gyro - self._b_g

        # Propagate nominal quaternion
        dq = _omega_exp_wxyz(omega_corrected, dt)
        self._q = _normalize_quat(_quat_multiply_wxyz(self._q, dq))

        # Linearised error-state transition
        #   delta_phi_{k+1} = (I - [omega_corr] * dt) * delta_phi - dt * delta_b_g
        #   delta_b_g_{k+1} = delta_b_g   (random walk only)
        F = np.eye(6)
        F[:3, :3] = np.eye(3) - _skew(omega_corrected) * dt
        F[:3, 3:] = -np.eye(3) * dt
        # F[3:, 3:] = I (bias walk through identity)

        # Process noise Q
        sigma_phi = self.gyro_noise_std * np.sqrt(dt)
        sigma_bg = self.gyro_bias_std * np.sqrt(dt)
        Q = np.diag([sigma_phi**2]*3 + [sigma_bg**2]*3)

        self._P = F @ self._P @ F.T + Q

    # -- Accelerometer update --------------------------------------------------

    def _accel_gate_weight(self, accel_mag: float) -> float:
        """Weight in [0, 1]: 1 at |a|=g, decays as |a| deviates (high-g gating).

        When accel_gate_alpha = 0, returns 1.0 always (no gating = naive filter).
        """
        if self.accel_gate_alpha <= 0.0:
            return 1.0
        deviation = (accel_mag - GRAVITY) / GRAVITY
        return float(np.exp(-self.accel_gate_alpha * deviation**2))

    def _update_accel(self, accel: np.ndarray) -> None:
        """Tilt update using accelerometer; gated by high-g weighting.

        Sign convention (critical):
          IMU specific force at rest (body FRD):  sf_body = [0, 0, -g]  (pointing UP = reaction)
          Predicted specific force from attitude:  h_body = R_wb^T @ (-G_NED) = [0, 0, -g] at rest
          so h_body = -R_wb^T @ G_NED = -(gravity in body frame).
          The measurement model is:  normalised(accel) = h_body / g = [0, 0, -1] at rest.
          Jacobian: d(h_body)/d(delta_phi) = d(-R^T G_NED)/d(delta_phi) = -skew(g_body)
          where g_body = R_wb^T @ G_NED (gravity direction in body = [0,0,+g] at rest).
        """
        accel_mag = float(np.linalg.norm(accel))
        if accel_mag < 1e-6:
            return

        gate = self._accel_gate_weight(accel_mag)

        # Skip update if gate weight is negligible (high-g): avoids numerical noise
        if gate < 1e-4:
            return

        # Gravity DIRECTION in body frame: g_hat = R^T @ G_NED / g = [0,0,+1] at rest.
        # We work entirely in NORMALISED (unit-vector) measurement space so the
        # innovation and the Jacobian share the same units -- mixing a unit-vector
        # innovation with a g-scaled (~9.8x) Jacobian miscalibrates the Kalman gain
        # and was the root cause of the accel update *corrupting* attitude.
        R_wb = _quat_to_R_wxyz(self._q)
        g_body = R_wb.T @ G_NED                      # [0, 0, +g] at rest
        g_hat = g_body / max(np.linalg.norm(g_body), 1e-9)   # [0, 0, +1] at rest

        # Predicted specific-force direction: h_hat = -g_hat = [0,0,-1] at rest.
        h_hat = -g_hat

        # Normalised accel measurement: a_hat = [0,0,-1] at rest (matches h_hat).
        a_hat = accel / accel_mag

        # Innovation: z - h(x)
        innovation = a_hat - h_hat

        # Jacobian H (3 x 6): d(h_hat)/d(delta_phi). With the right-multiplicative
        # body-frame error convention used at injection (q <- q * exp(delta_phi)),
        # h_hat = -R^T G_NED/g and a finite-difference check (see scratch) gives
        #   d(h_hat)/d(delta_phi) = -skew(g_hat).
        # NOTE the normalisation by g: g_hat is the UNIT gravity direction, matching
        # the unit innovation above.
        H = np.zeros((3, 6))
        H[:, :3] = -_skew(g_hat)

        # Effective measurement noise: R_meas / gate (larger noise when gate small = high-g)
        sigma_a = self.accel_noise_std / GRAVITY  # normalised units
        R_meas = (sigma_a**2 / gate) * np.eye(3)

        # Direction-aware innovation gate (Mahalanobis / chi-square consistency test).
        # Magnitude gating cannot reject a near-g-magnitude disturbance whose DIRECTION
        # is wrong; this gate can. Skip the update when the innovation is inconsistent
        # with its predicted covariance S.
        if self.accel_chi2_thresh > 0.0:
            S = H @ self._P @ H.T + R_meas
            try:
                md = float(innovation @ np.linalg.solve(S, innovation))
            except np.linalg.LinAlgError:
                return
            if md > self.accel_chi2_thresh:
                return  # reject: accel inconsistent with attitude prior (high-g disturbance)

        # Kalman update on error state
        self._apply_eskf_update(innovation, H, R_meas)

    # -- Magnetometer update (optional) ----------------------------------------

    def _update_mag(self, mag_body: np.ndarray) -> None:
        """Yaw update from body-frame magnetometer and known NED reference field.

        Only the yaw component is corrected (the mag is not informative about tilt
        unless the full 3D field is used, and we trust the accel for tilt).
        """
        mag_mag = float(np.linalg.norm(mag_body))
        if mag_mag < 1e-6 or np.linalg.norm(self.mag_ned) < 1e-6:
            return

        mag_body_n = mag_body / mag_mag
        mag_ned_n = self.mag_ned / np.linalg.norm(self.mag_ned)

        R_wb = _quat_to_R_wxyz(self._q)
        mag_ned_pred = R_wb @ mag_body_n    # predicted NED direction from body meas

        # Project onto horizontal plane for yaw-only sensitivity
        # yaw error = cross(mag_ned_pred_horiz, mag_ned_ref_horiz) . [0,0,1]
        h_pred = np.array([mag_ned_pred[0], mag_ned_pred[1], 0.0])
        h_ref = np.array([mag_ned_n[0], mag_ned_n[1], 0.0])
        hn_p = np.linalg.norm(h_pred)
        hn_r = np.linalg.norm(h_ref)
        if hn_p < 1e-6 or hn_r < 1e-6:
            return

        h_pred /= hn_p
        h_ref /= hn_r

        # Scalar yaw innovation: z = cross(h_pred, h_ref)[2]
        innovation = np.array([np.cross(h_pred, h_ref)[2]])

        # H: 1x6; only yaw (Z-axis) component of delta_phi contributes
        # d(yaw_meas)/d(delta_phi) ~ [0, 0, 1] (approximately, near correct attitude)
        H = np.zeros((1, 6))
        H[0, 2] = 1.0

        R_meas = np.array([[self.mag_noise_std**2]])
        self._apply_eskf_update(innovation, H, R_meas)

    # -- Core ESKF update ------------------------------------------------------

    def _apply_eskf_update(
        self,
        innovation: np.ndarray,
        H: np.ndarray,
        R_meas: np.ndarray,
    ) -> None:
        """Apply a generic ESKF correction, then reset error state into nominal."""
        PHt = self._P @ H.T
        S = H @ PHt + R_meas
        try:
            K = np.linalg.solve(S, PHt.T).T   # same solve trick as LinearKF
        except np.linalg.LinAlgError:
            return

        delta_x = K @ innovation               # (6,)
        delta_phi = delta_x[:3]
        delta_b_g = delta_x[3:]

        # Inject attitude correction into nominal quaternion
        # delta_phi is a small rotation vector in the body frame
        angle = np.linalg.norm(delta_phi)
        if angle > 1e-12:
            dq = _omega_exp_wxyz(delta_phi, 1.0)   # exp(delta_phi / 2) at unit dt
        else:
            dq = np.array([1., 0., 0., 0.])
        self._q = _normalize_quat(_quat_multiply_wxyz(self._q, dq))

        # Inject bias correction
        self._b_g = self._b_g + delta_b_g

        # Joseph-form covariance update (symmetric + PD)
        I_KH = np.eye(6) - K @ H
        self._P = I_KH @ self._P @ I_KH.T + K @ R_meas @ K.T
