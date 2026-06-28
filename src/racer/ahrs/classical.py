"""Classical complementary attitude filters: Madgwick and Mahony.

These are the benchmark comparators for the ESKF. Both are well-known, widely used,
and fast — but neither has principled gating for high-g maneuvers.

Frame convention
----------------
Quaternion: (w,x,y,z) scalar-FIRST, body(FRD)->world(NED). Matches frames.py.
Gravity NED: [0, 0, +9.80665] m/s^2 (down = +Z).
Specific force at rest in body FRD: [0, 0, -g].

Both filters use the same interface as ESKFAHRS: step(gyro, accel, dt, mag_body=None).

References
----------
Madgwick S, Harrison A, Vaidyanathan R (2011). "Estimation of IMU and MARG
orientation using a gradient descent algorithm." IEEE RAS/EMBS Int. Conf.
Biomedical Robotics and Biomechatronics. pp. 1–7.

Mahony R, Hamel T, Pflimlin J-M (2008). "Nonlinear complementary filters on the
special orthogonal group." IEEE Trans. Autom. Control. 53(5):1203–1218.

Note on high-g behaviour:
  Both filters feed the raw accelerometer (possibly normalised) into the gradient
  descent / proportional correction term WITHOUT gating its magnitude. During 4-5 g
  maneuvers, the body-frame accel vector points far from -g, so both filters will
  try to 'tilt' the attitude estimate toward the erroneous direction, degrading
  accuracy. The ESKF's accel_gate_alpha parameter addresses this gap.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from typing import Optional


GRAVITY = 9.80665


def _quat_multiply_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v * 0.0


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([1., 0., 0., 0.])


@dataclass
class MadgwickAHRS:
    """Madgwick gradient-descent complementary filter.

    Parameters
    ----------
    beta : gradient descent step size. Higher = more weight on accel/mag,
           less lag but more susceptibility to accel disturbances.
           Typical range: 0.01–0.5. Madgwick's paper uses 0.1 as a starting point.
    """
    beta: float = 0.1
    _q: np.ndarray = None   # type: ignore

    def __post_init__(self) -> None:
        self._q = np.array([1., 0., 0., 0.], dtype=np.float64)

    @property
    def q_wxyz(self) -> np.ndarray:
        return self._q.copy()

    def reset(self, q_init: Optional[np.ndarray] = None) -> None:
        self._q = _normalize_quat(
            np.asarray(q_init, dtype=np.float64) if q_init is not None
            else np.array([1., 0., 0., 0.])
        )

    def step(
        self,
        gyro: np.ndarray,
        accel: np.ndarray,
        dt: float,
        mag_body: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """One IMU step. Returns updated quaternion (w,x,y,z).

        Sign convention:
          accel is SPECIFIC FORCE: at rest = [0,0,-g] (FRD, up-reaction from floor).
          Madgwick's formulas use GRAVITY DIRECTION: at rest = [0,0,+1] (down in body FRD).
          Negate: grav_dir = -accel / |accel|. This is equivalent to what Madgwick's
          original paper does when the raw accelerometer reads [0,0,+g] (NWU convention).
          Here in FRD+specific-force convention we negate explicitly.
        """
        if dt <= 0:
            return self._q.copy()

        gyro = np.asarray(gyro, dtype=np.float64)
        accel = np.asarray(accel, dtype=np.float64)
        q = self._q
        w, x, y, z = q

        # Convert specific force to normalised gravity direction for tilt:
        # grav_dir = normalise(-sf_body) = [0,0,+1] at rest (down in FRD)
        grav_dir = _normalize(-accel)
        if np.linalg.norm(grav_dir) < 1e-6:
            # No accel info; pure gyro integration
            q_new = self._integrate_gyro(q, gyro, dt)
            self._q = _normalize_quat(q_new)
            return self._q.copy()

        ax, ay, az = grav_dir

        # Objective function f_g: predicted gravity direction vs measured gravity direction.
        # Predicted from quaternion (Madgwick 2011 Eq. 25):
        #   h(q) = [2*(x*z - w*y), 2*(w*x + y*z), 2*(0.5 - x^2 - y^2)]
        # This is the body-frame third column of R_wb^T = R_bw (the down direction in body).
        # = [0, 0, 1] at identity (level hover with FRD/NED aligned).
        f_g = np.array([
            2.0*(x*z - w*y) - ax,
            2.0*(w*x + y*z) - ay,
            2.0*(0.5 - x*x - y*y) - az,
        ])

        # Jacobian J_g^T @ f_g (from Madgwick 2011 Eq. 26):
        grad = np.array([
            -2.0*y*f_g[0] + 2.0*x*f_g[1],
            2.0*z*f_g[0]  + 2.0*w*f_g[1] - 4.0*x*f_g[2],
            -2.0*w*f_g[0] + 2.0*z*f_g[1] - 4.0*y*f_g[2],
            2.0*x*f_g[0]  + 2.0*y*f_g[1],
        ])

        grad_n = _normalize(grad)

        # Gyro integration rate
        q_dot_gyro = 0.5 * _quat_multiply_wxyz(q, np.array([0., gyro[0], gyro[1], gyro[2]]))

        # Standard Madgwick update: q += (q_dot_gyro - beta * grad_n) * dt
        q_new = q + (q_dot_gyro - self.beta * grad_n) * dt

        self._q = _normalize_quat(q_new)
        return self._q.copy()

    def _integrate_gyro(self, q: np.ndarray, gyro: np.ndarray, dt: float) -> np.ndarray:
        q_dot = 0.5 * _quat_multiply_wxyz(q, np.array([0., gyro[0], gyro[1], gyro[2]]))
        return q + q_dot * dt


@dataclass
class MahonyAHRS:
    """Mahony nonlinear complementary filter on SO(3).

    Parameters
    ----------
    kp : proportional gain (error correction strength). Higher = faster correction,
         more accel noise sensitivity. Typical: 2.0–10.0.
    ki : integral gain (bias estimation). Higher = faster bias compensation.
         Typical: 0.005–0.05. Set to 0 to disable integral (pure proportional).
    """
    kp: float = 2.0
    ki: float = 0.005
    _q: np.ndarray = None   # type: ignore
    _integral: np.ndarray = None  # type: ignore

    def __post_init__(self) -> None:
        self._q = np.array([1., 0., 0., 0.], dtype=np.float64)
        self._integral = np.zeros(3, dtype=np.float64)

    @property
    def q_wxyz(self) -> np.ndarray:
        return self._q.copy()

    def reset(self, q_init: Optional[np.ndarray] = None) -> None:
        self._q = _normalize_quat(
            np.asarray(q_init, dtype=np.float64) if q_init is not None
            else np.array([1., 0., 0., 0.])
        )
        self._integral = np.zeros(3)

    def step(
        self,
        gyro: np.ndarray,
        accel: np.ndarray,
        dt: float,
        mag_body: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """One IMU step. Returns updated quaternion (w,x,y,z)."""
        if dt <= 0:
            return self._q.copy()

        gyro = np.asarray(gyro, dtype=np.float64)
        accel = np.asarray(accel, dtype=np.float64)
        q = self._q
        w, x, y, z = q

        # Predicted gravity-DOWN direction in the body frame from current quaternion.
        # g_pred = R_wb^T @ [0,0,1]_NED = [2(xz-wy), 2(yz+wx), 1-2(x^2+y^2)];
        # at rest this is [0,0,+1] (gravity points down the body FRD +Z axis).
        g_pred = np.array([
            2.0*(x*z - w*y),
            2.0*(y*z + w*x),
            1.0 - 2.0*(x*x + y*y),
        ])

        # SIGN (critical): accel is SPECIFIC FORCE, which reads [0,0,-g] at rest
        # (reaction is UP the body -Z axis). g_pred above is the gravity-DOWN
        # direction [0,0,+1]. To compare like-with-like we measure the gravity-down
        # direction from the accelerometer by NEGATING the specific force:
        #   v_meas = normalise(-accel) = [0,0,+1] at rest.
        # Feeding the un-negated specific force would put the filter at the UNSTABLE
        # 180-deg equilibrium (cross of antiparallel vectors is ~0 at rest but the
        # correction has the wrong sign off-rest), which collapses to ~30 deg error.
        a_n = _normalize(-accel)
        if np.linalg.norm(a_n) < 1e-6:
            a_n = np.zeros(3)
            e = np.zeros(3)
        else:
            # Error = cross(v_meas, g_pred): zero when the estimate is correct,
            # otherwise the rotation axis that drives g_pred toward v_meas.
            e = np.cross(a_n, g_pred)

        # Integral feedback (bias estimation)
        self._integral += e * dt
        gyro_corrected = gyro + self.kp * e + self.ki * self._integral

        # Integrate corrected gyro into quaternion
        q_dot = 0.5 * _quat_multiply_wxyz(q, np.array([0., gyro_corrected[0],
                                                         gyro_corrected[1],
                                                         gyro_corrected[2]]))
        q_new = q + q_dot * dt
        self._q = _normalize_quat(q_new)
        return self._q.copy()
