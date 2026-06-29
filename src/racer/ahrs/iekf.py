"""Left-Invariant Extended Kalman Filter (IEKF) AHRS on SO(3).

A research-grounded contender alongside the ESKF and the classical complementary
filters. Implements a minimal *left-invariant* attitude+gyro-bias filter following the
invariant-filtering framework of Barrau & Bonnabel ("The Invariant Extended Kalman
Filter as a Stable Observer", IEEE TAC 2017) and the equivariant-filter line of
van Goor ("Equivariant Filters for Visual Spatial Awareness", ANU PhD thesis 2023 —
the EqVIO / EqF line that the C2 vision-estimator design already cites).

Why an invariant filter
-----------------------
The headline property of the IEKF is *consistency*: the linearised error dynamics for
an invariant error are (for group-affine systems) state-independent, so the filter does
not suffer the covariance collapse / over-confidence that a naive EKF exhibits under
LARGE attitude excursions. This is exactly the high-g / aggressive-attitude regime VQ2
racing lives in, which is why it is worth carrying as a contender rather than assuming
the ESKF is the ceiling.

State and error convention
--------------------------
Nominal state : rotation R (body->world, FRD->NED) + gyro bias b (body frame).
Left-invariant attitude error xi (3-vec, body frame):
    R = R_hat * Exp(xi)
Bias error: db = b - b_hat. Error state e = [xi; db] (6-dim).

(The attitude-only-with-bias system is not perfectly group-affine because the bias
breaks the invariance, so this is technically an "imperfect IEKF". The attitude block
is handled invariantly; the bias enters as in a standard MEKF. This is the standard,
correct minimal formulation and is what converges in practice.)

Propagation (R_hat^+ = R_hat * Exp((omega - b_hat) dt)):
    xi_dot = -[omega - b_hat]_x xi - db
    db_dot = 0
    => F = [[I - [omega-b]_x dt, -I dt], [0, I]]

Gravity (left-invariant) measurement
------------------------------------
The accelerometer specific force gives the gravity-DOWN direction in the body frame.
The measurement is the left-invariant output y = R^T g_world_down:
    y_meas = normalise(-accel)            # [0,0,+1] at rest (down in body FRD)
    y_hat  = R_hat^T @ [0,0,1]_NED
    R^T g = Exp(-xi) R_hat^T g ~ y_hat + [y_hat]_x xi
    => innovation z = y_meas - y_hat,  H_xi = +skew(y_hat),  H_db = 0.

The same TWO-GATE robustness as the ESKF is applied:
  1. magnitude gate  : down-weight R when |a| deviates from g (high-g rejection);
  2. innovation gate : chi-square consistency test on the innovation (direction-aware
                       rejection of near-g-magnitude but wrong-direction disturbances,
                       the same relinnov chi2 family used in the C2 estimator chain).

Empirical relationship to the ESKF (this bench)
-----------------------------------------------
For the pure SO(3) attitude-from-gravity problem the left-invariant EKF and a correctly
formulated MEKF/ESKF produce algebraically equivalent updates (H_iekf = +skew(R^T g)
vs H_eskf = -skew(g_hat), with y_hat = -g_hat for the gravity-down vs specific-force-down
sign), so the two filters track to ~1e-11 deg on every scenario here, INCLUDING fast
recovery from a large (80 deg) initial attitude error. The IEKF's distinctive consistency
advantage in the literature is realised on the COUPLED SE_2(3) / INS problem
(attitude+velocity+position), not on attitude-only. Carrying the IEKF therefore (a)
cross-validates the ESKF result and (b) leaves the geometrically-consistent propagation
in place for the eventual extension to a full visual-inertial estimator.

Frame convention: quaternion (w,x,y,z) scalar-FIRST, matches frames.py / the rest of the
package. Gravity NED [0,0,+g], specific force at rest [0,0,-g] in body FRD.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


GRAVITY = 9.80665
G_DOWN_W = np.array([0.0, 0.0, 1.0])   # unit gravity-DOWN direction in NED world


def _skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix: _skew(v) @ w == cross(v, w)."""
    x, y, z = v
    return np.array([[0.0, -z, y],
                     [z, 0.0, -x],
                     [-y, x, 0.0]])


def _Exp_so3(phi: np.ndarray) -> np.ndarray:
    """SO(3) exponential map (Rodrigues): rotation vector phi -> rotation matrix."""
    theta = float(np.linalg.norm(phi))
    if theta < 1e-12:
        return np.eye(3) + _skew(phi)
    K = _skew(phi / theta)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _quat_to_R_wxyz(q: np.ndarray) -> np.ndarray:
    """(w,x,y,z) -> 3x3 rotation matrix (body->world)."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),      1 - 2*(x*x + z*z),  2*(y*z - x*w)],
        [2*(x*z - y*w),      2*(y*z + x*w),      1 - 2*(x*x + y*y)],
    ])


def _R_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix (body->world) -> unit quaternion (w,x,y,z), w>=0 chosen."""
    tr = np.trace(R)
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    q = np.array([w, x, y, z])
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


@dataclass
class LeftInvariantEKF:
    """Left-invariant EKF AHRS (attitude + gyro bias) on SO(3).

    Same constructor surface and step() signature as ESKFAHRS so it drops straight
    into run_filter_on_sequence and the bench.

    Parameters
    ----------
    gyro_noise_std : rad/s, 1-sigma white gyro noise.
    gyro_bias_std : rad/s/sqrt(s), gyro-bias random walk.
    accel_noise_std : m/s^2, 1-sigma white accel noise.
    accel_gate_alpha : magnitude-gate sharpness (0 disables). exp(-alpha*((|a|-g)/g)^2).
    accel_chi2_thresh : innovation (chi-square) gate threshold; chi2(3,.95)=7.815.
                        0 or negative disables the innovation gate.
    accel_freefall_tol_lo / accel_freefall_tol_hi : free-fall / high-|a| magnitude guard
                        (catastrophic-failure safety net, identical to ESKFAHRS so the two
                        gravity-referenced filters stay algebraically equivalent). Skip the
                        accel update when |a| is outside [g*(1-tol_lo), g*(1+tol_hi)]; the
                        accel is only the gravity reference when |a| ~ g. Negative disables
                        that side. See eskf.py for the full rationale.
    """
    gyro_noise_std: float = 0.01
    gyro_bias_std: float = 1e-4
    accel_noise_std: float = 0.3
    accel_gate_alpha: float = 10.0
    accel_chi2_thresh: float = 7.815
    accel_freefall_tol_lo: float = 0.75   # reject when |a| < g*(1-0.75) = 0.25 g  (free-fall)
    accel_freefall_tol_hi: float = 9.0    # reject when |a| > g*(1+9.0)  = 10 g    (extreme high-g)

    _R: np.ndarray = field(default_factory=lambda: np.eye(3))
    _b_g: np.ndarray = field(default_factory=lambda: np.zeros(3))
    _P: np.ndarray = field(default_factory=lambda: np.diag([1e-2]*3 + [1e-6]*3))

    def __post_init__(self) -> None:
        self._R = np.eye(3)
        self._b_g = np.zeros(3, dtype=np.float64)
        self._P = np.diag([1e-2]*3 + [1e-6]*3).astype(np.float64)

    # -- Public interface -------------------------------------------------------

    @property
    def q_wxyz(self) -> np.ndarray:
        return _R_to_quat_wxyz(self._R)

    @property
    def gyro_bias(self) -> np.ndarray:
        return self._b_g.copy()

    @property
    def attitude_uncertainty_rad(self) -> float:
        return float(np.sqrt(np.trace(self._P[:3, :3]) / 3.0))

    def reset(self, q_init: Optional[np.ndarray] = None) -> None:
        if q_init is not None:
            q = np.asarray(q_init, dtype=np.float64)
            self._R = _quat_to_R_wxyz(q / np.linalg.norm(q))
        else:
            self._R = np.eye(3)
        self._b_g = np.zeros(3)
        self._P = np.diag([1e-2]*3 + [1e-6]*3).astype(np.float64)

    def step(
        self,
        gyro: np.ndarray,
        accel: np.ndarray,
        dt: float,
        mag_body: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if dt <= 0:
            return self.q_wxyz
        self._predict(np.asarray(gyro, dtype=np.float64), dt)
        self._update_accel(np.asarray(accel, dtype=np.float64))
        return self.q_wxyz

    # -- Prediction ------------------------------------------------------------

    def _predict(self, gyro: np.ndarray, dt: float) -> None:
        omega = gyro - self._b_g
        self._R = self._R @ _Exp_so3(omega * dt)

        F = np.eye(6)
        F[:3, :3] = np.eye(3) - _skew(omega) * dt
        F[:3, 3:] = -np.eye(3) * dt

        sigma_phi = self.gyro_noise_std * np.sqrt(dt)
        sigma_bg = self.gyro_bias_std * np.sqrt(dt)
        Q = np.diag([sigma_phi**2]*3 + [sigma_bg**2]*3)
        self._P = F @ self._P @ F.T + Q

    # -- Accelerometer (left-invariant gravity) update -------------------------

    def _accel_gate_weight(self, accel_mag: float) -> float:
        if self.accel_gate_alpha <= 0.0:
            return 1.0
        deviation = (accel_mag - GRAVITY) / GRAVITY
        return float(np.exp(-self.accel_gate_alpha * deviation**2))

    def _accel_magnitude_in_band(self, accel_mag: float) -> bool:
        """Free-fall / high-|a| magnitude guard (identical to ESKFAHRS). True iff |a| ~ g."""
        if self.accel_freefall_tol_lo >= 0.0 and accel_mag < GRAVITY * (1.0 - self.accel_freefall_tol_lo):
            return False
        if self.accel_freefall_tol_hi >= 0.0 and accel_mag > GRAVITY * (1.0 + self.accel_freefall_tol_hi):
            return False
        return True

    def _update_accel(self, accel: np.ndarray) -> None:
        accel_mag = float(np.linalg.norm(accel))
        if accel_mag < 1e-6:
            return
        # Free-fall / high-|a| magnitude guard: skip the accel update when |a| is not ~ g
        # (the accel is only the gravity reference near |a|=g). No-op when |a| ~ g.
        if not self._accel_magnitude_in_band(accel_mag):
            return
        gate = self._accel_gate_weight(accel_mag)
        if gate < 1e-4:
            return

        # Left-invariant output: gravity-DOWN direction in body frame.
        y_meas = -accel / accel_mag            # [0,0,+1] at rest
        y_hat = self._R.T @ G_DOWN_W           # predicted gravity-down in body

        innovation = y_meas - y_hat

        # Left-invariant output Jacobian: H_xi = +skew(y_hat).
        H = np.zeros((3, 6))
        H[:, :3] = _skew(y_hat)

        sigma_a = self.accel_noise_std / GRAVITY
        R_meas = (sigma_a**2 / gate) * np.eye(3)

        S = H @ self._P @ H.T + R_meas
        if self.accel_chi2_thresh > 0.0:
            try:
                md = float(innovation @ np.linalg.solve(S, innovation))
            except np.linalg.LinAlgError:
                return
            if md > self.accel_chi2_thresh:
                return

        PHt = self._P @ H.T
        try:
            K = np.linalg.solve(S, PHt.T).T
        except np.linalg.LinAlgError:
            return

        delta_x = K @ innovation
        xi = delta_x[:3]
        db = delta_x[3:]

        # Inject the left-invariant correction: R <- R * Exp(xi).
        self._R = self._R @ _Exp_so3(xi)
        self._b_g = self._b_g + db

        I_KH = np.eye(6) - K @ H
        self._P = I_KH @ self._P @ I_KH.T + K @ R_meas @ K.T
