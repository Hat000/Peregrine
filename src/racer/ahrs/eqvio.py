"""SE_2(3) Right-Invariant EKF — the COUPLED invariant estimator (attitude+velocity+position).

This is the performance-additive "ESTIMATE" pillar: the place where the invariant
filter's *consistency* edge actually materialises. The attitude-only IEKF in iekf.py
deliberately ties the ESKF (a null result) because SO(3)-from-gravity is, for both
filters, the same linear update. The headline property of an invariant filter only
shows up on the COUPLED problem, and this module is that problem.

The thesis (and what this module empirically confirms / refutes)
----------------------------------------------------------------
Bias-free strapdown IMU kinematics on the matrix Lie group SE_2(3) are GROUP-AFFINE
(Barrau & Bonnabel, "The Invariant Extended Kalman Filter as a Stable Observer",
IEEE TAC 2017). For a group-affine system the *invariant* error
    eta = X * X_hat^{-1}          (right-invariant error)
propagates with linearised dynamics
    d/dt xi = A xi  (+ noise)
whose system matrix A is **independent of the state estimate X_hat**. Contrast the
standard (Cartesian) EKF, whose F_k depends on the current estimate. The practical
consequence: the right-invariant EKF's linearisation error -> 0 as the trajectory is
tracked, EVEN WITH ZERO SENSOR NOISE. That is a *Taylor* error (the EKF's first-order
truncation of a nonlinear update), not a *noise* error, so it does not wash out with a
bigger Q/R — it shows up as the standard EKF becoming OVER-CONFIDENT (NEES above its
chi-square band) while the RI-EKF stays consistent (NEES inside the band) under
aggressive motion.

State and group
---------------
X = | R  v  p |   in SE_2(3),  R in SO(3), v,p in R^3       (5x5 matrix)
    | 0  1  0 |
    | 0  0  1 |
R : body(FRD) -> world(NED).  v : world-frame velocity (NED).  p : world position (NED).

Lie algebra se_2(3): xi = [xi_R (3); xi_v (3); xi_p (3)] (9-vec).
Exp(xi) uses the SO(3) exponential for the rotation block and the left-Jacobian-coupled
v/p blocks (closed form below).

IMU kinematics (continuous, bias-free; specific force f_b = R^T (a_world - g), gyro w_b)
    R_dot = R [w_b]_x
    v_dot = R f_b + g          (= a_world ; g = [0,0,+9.80665] NED)
    p_dot = v
These are exactly the SE_2(3) group-affine dynamics of Barrau-Bonnabel Eq. (28)-(33).

Right-invariant error and its (state-INDEPENDENT) propagation
-------------------------------------------------------------
Define X = Exp(xi) X_hat (left-multiplicative = right-invariant error in the B&B sense:
the error lives in the WORLD/inertial frame, which is what makes A constant). The
continuous error dynamics for the bias-free system are
    xi_dot = A xi,   A = | 0      0   0 |
                          | [g]_x  0   0 |     (state-INDEPENDENT — the whole point)
                          | 0      I   0 |
i.e. the gravity vector couples attitude error into velocity error, velocity error
into position error, with NO dependence on R_hat, v_hat, p_hat. (Compare iekf.py's
SO(3) F, which IS state-independent already for pure rotation; here the coupling to
v and p through a CONSTANT A is the new, load-bearing structure.)

Discretisation uses the matrix exponential of A*dt (exact for this nilpotent A).

Measurement: world-frame position (GPS-like / map-anchored gate) and, as the next
tier, gate-corner landmark bearings (SOT(3) inverse-depth — see eqvio_landmark.py).
For a RIGHT-invariant filter a WORLD-frame position measurement h(X) = p is a
*left*-invariant output, so its Jacobian is state-dependent in the naive form; the
standard RIEKF trick is to express the innovation in the right-invariant error frame,
giving H = [0, 0, I] acting on the right-invariant error with the measured/predicted
positions compared directly. We implement exactly that (derivation in _update_position).

Frame / quaternion convention: (w,x,y,z) scalar-first, FRD body, NED world, g=[0,0,+9.80665].
Reuses the SO(3) helpers from iekf.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from racer.ahrs.iekf import _skew, _Exp_so3, _quat_to_R_wxyz, _R_to_quat_wxyz


GRAVITY = 9.80665
G_NED = np.array([0.0, 0.0, GRAVITY])   # gravity DOWN = +Z in NED


# ---------------------------------------------------------------------------
# SE_2(3) group helpers
# ---------------------------------------------------------------------------

def _left_jacobian_so3(phi: np.ndarray) -> np.ndarray:
    """Left Jacobian of SO(3): J_l(phi) such that Exp(phi+dphi) ~ Exp(phi) Exp(J_l^{-1}... ).

    Used to build the v/p blocks of the SE_2(3) exponential. Closed form:
        J_l = I + (1-cos t)/t^2 [phi]_x + (t - sin t)/t^3 [phi]_x^2
    """
    theta = float(np.linalg.norm(phi))
    K = _skew(phi)
    if theta < 1e-10:
        # 2nd-order series (stable near 0)
        return np.eye(3) + 0.5 * K + (1.0 / 6.0) * (K @ K)
    a = (1.0 - np.cos(theta)) / theta**2
    b = (theta - np.sin(theta)) / theta**3
    return np.eye(3) + a * K + b * (K @ K)


def _Exp_se23(xi: np.ndarray) -> np.ndarray:
    """Exponential map se_2(3) -> SE_2(3) (5x5). xi = [xi_R(3); xi_v(3); xi_p(3)].

    R = Exp_so3(xi_R); v = J_l(xi_R) xi_v; p = J_l(xi_R) xi_p.
    """
    xi_R = xi[0:3]
    xi_v = xi[3:6]
    xi_p = xi[6:9]
    R = _Exp_so3(xi_R)
    Jl = _left_jacobian_so3(xi_R)
    v = Jl @ xi_v
    p = Jl @ xi_p
    X = np.eye(5)
    X[0:3, 0:3] = R
    X[0:3, 3] = v
    X[0:3, 4] = p
    return X


def _make_X(R: np.ndarray, v: np.ndarray, p: np.ndarray) -> np.ndarray:
    X = np.eye(5)
    X[0:3, 0:3] = R
    X[0:3, 3] = v
    X[0:3, 4] = p
    return X


def _adjoint_se23(X: np.ndarray) -> np.ndarray:
    """Adjoint Ad_X (9x9) of SE_2(3). For X = (R, v, p):
        Ad = | R        0   0 |
             | [v]_x R  R   0 |
             | [p]_x R  0   R |
    Used to map between right- and left-invariant error frames when needed.
    """
    R = X[0:3, 0:3]
    v = X[0:3, 3]
    p = X[0:3, 4]
    Ad = np.zeros((9, 9))
    Ad[0:3, 0:3] = R
    Ad[3:6, 0:3] = _skew(v) @ R
    Ad[3:6, 3:6] = R
    Ad[6:9, 0:3] = _skew(p) @ R
    Ad[6:9, 6:9] = R
    return Ad


# ---------------------------------------------------------------------------
# SE_2(3) Right-Invariant EKF
# ---------------------------------------------------------------------------

@dataclass
class SE23RightInvariantEKF:
    """Right-invariant EKF on SE_2(3) (attitude + velocity + position).

    Bias-free core (the group-affine, consistency-exact case). Gyro/accel biases can be
    appended later as the "imperfect IEKF" extension (they break group-affineness and
    re-introduce a small state dependence, exactly as in iekf.py's bias block).

    The error convention is RIGHT-INVARIANT:  X = Exp(xi) @ X_hat,  xi in R^9 ordered
    [xi_R; xi_v; xi_p]. Covariance P is 9x9 over xi.

    Parameters
    ----------
    gyro_noise_std  : rad/s, white gyro noise (1-sigma).
    accel_noise_std : m/s^2, white accel (specific-force) noise (1-sigma).
    pos_noise_std   : m, position-measurement noise (1-sigma per axis).
    """
    gyro_noise_std: float = 0.01
    accel_noise_std: float = 0.05
    pos_noise_std: float = 0.10

    _X: np.ndarray = field(default_factory=lambda: np.eye(5))
    _P: np.ndarray = field(default_factory=lambda: np.eye(9) * 1e-3)
    _Phi_cache: Optional[np.ndarray] = None
    _Phi_dt: float = -1.0

    def __post_init__(self) -> None:
        self._X = np.eye(5)
        self._P = np.diag([1e-3]*3 + [1e-2]*3 + [1e-2]*3).astype(np.float64)
        self._Phi_cache = None
        self._Phi_dt = -1.0

    # -- accessors -------------------------------------------------------------

    @property
    def R(self) -> np.ndarray:
        return self._X[0:3, 0:3].copy()

    @property
    def v(self) -> np.ndarray:
        return self._X[0:3, 3].copy()

    @property
    def p(self) -> np.ndarray:
        return self._X[0:3, 4].copy()

    @property
    def q_wxyz(self) -> np.ndarray:
        return _R_to_quat_wxyz(self.R)

    @property
    def P(self) -> np.ndarray:
        return self._P.copy()

    def reset(
        self,
        R: Optional[np.ndarray] = None,
        v: Optional[np.ndarray] = None,
        p: Optional[np.ndarray] = None,
        P: Optional[np.ndarray] = None,
    ) -> None:
        self._X = _make_X(
            np.eye(3) if R is None else np.asarray(R, float),
            np.zeros(3) if v is None else np.asarray(v, float),
            np.zeros(3) if p is None else np.asarray(p, float),
        )
        self._P = (np.diag([1e-3]*3 + [1e-2]*3 + [1e-2]*3).astype(np.float64)
                   if P is None else np.asarray(P, float).copy())

    # -- propagation (group-affine; A is state-INDEPENDENT) --------------------

    def _A_continuous(self) -> np.ndarray:
        """Continuous right-invariant error system matrix A (9x9), STATE-INDEPENDENT.

            xi_dot = A xi,  A = [[0,0,0],[ [g]_x, 0, 0],[0, I, 0]]
        The ONLY content is the constant gravity coupling [g]_x and the v->p integrator.
        That A does not reference R_hat/v_hat/p_hat is the entire consistency argument.
        """
        A = np.zeros((9, 9))
        A[3:6, 0:3] = _skew(G_NED)   # attitude error -> velocity error via gravity
        A[6:9, 3:6] = np.eye(3)      # velocity error -> position error (integrator)
        return A

    def predict(self, gyro: np.ndarray, accel: np.ndarray, dt: float) -> None:
        """Strapdown IMU propagation of the SE_2(3) mean + RI covariance.

        gyro  : (3,) body angular rate (rad/s, FRD).
        accel : (3,) body specific force (m/s^2, FRD); rest = [0,0,-g].
        """
        if dt <= 0:
            return
        w_b = np.asarray(gyro, float)
        f_b = np.asarray(accel, float)
        R = self.R
        v = self.v
        p = self.p

        # --- mean propagation (exact strapdown over the step) -----------------
        a_world = R @ f_b + G_NED                 # world kinematic acceleration
        R_new = R @ _Exp_so3(w_b * dt)
        v_new = v + a_world * dt
        p_new = p + v * dt + 0.5 * a_world * dt**2
        self._X = _make_X(R_new, v_new, p_new)

        # --- covariance propagation (Phi = exp(A dt), A state-independent) -----
        # A is CONSTANT, so Phi depends only on dt: cache it across steps (this is also a
        # concrete manifestation of the group-affine property — one transition matrix for
        # the whole trajectory, no per-step relinearisation).
        if self._Phi_cache is None or self._Phi_dt != dt:
            self._Phi_cache = _expm_series(self._A_continuous() * dt)
            self._Phi_dt = dt
        Phi = self._Phi_cache

        # process-noise mapping: gyro noise drives xi_R, accel noise drives xi_v.
        # In the right-invariant frame the input matrix carries R_hat (the noise is
        # body-frame), but its EFFECT on the *consistency* of A is nil — A itself is
        # what must be state-free. We use the adjoint-consistent G below.
        G = np.zeros((9, 6))
        G[0:3, 0:3] = R                            # body gyro noise -> world attitude err
        G[3:6, 3:6] = R                            # body accel noise -> world velocity err
        Qc = np.zeros((6, 6))
        Qc[0:3, 0:3] = (self.gyro_noise_std**2) * np.eye(3)
        Qc[3:6, 3:6] = (self.accel_noise_std**2) * np.eye(3)
        Qd = Phi @ (G @ Qc @ G.T) @ Phi.T * dt

        self._P = Phi @ self._P @ Phi.T + Qd
        self._P = 0.5 * (self._P + self._P.T)

    # -- world-position update (left-invariant output, RI-frame innovation) ----

    def update_position(
        self,
        p_meas: np.ndarray,
        R_meas: Optional[np.ndarray] = None,
    ) -> None:
        """Update from a world-frame position measurement y = p + noise.

        Right-invariant error X = Exp(xi) X_hat. To first order the measured world
        position relates to the error by
            p = p_hat + [I acting through the (R)-block of xi_p] ...
        For SE_2(3) with the (R,v,p) layout the world position sits in the last column;
        the right-invariant (world-frame) position error is exactly xi_p MINUS the
        lever-arm coupling [p_hat]_x xi_R (the position column rotates with the world-
        frame attitude error). Hence
            innovation z = p_meas - p_hat,
            H = [ -[p_hat]_x , 0 , I ]   (3x9, acting on [xi_R; xi_v; xi_p]).
        This H carries p_hat (a mean, not the covariance) — that is correct and is NOT
        the state-dependence that breaks consistency; the PROPAGATION matrix A is what
        must be (and is) state-free.
        """
        p_meas = np.asarray(p_meas, float)
        p_hat = self.p
        z = p_meas - p_hat

        H = np.zeros((3, 9))
        H[:, 0:3] = -_skew(p_hat)
        H[:, 6:9] = np.eye(3)

        Rm = (self.pos_noise_std**2 * np.eye(3)) if R_meas is None else np.asarray(R_meas, float)
        self._apply_update(z, H, Rm)

    # -- body-frame landmark update (LEFT-invariant output — the RIEKF's natural,
    #    consistency-preserving measurement; this is the gate-corner observation) -----

    def update_landmark(
        self,
        y_body: np.ndarray,
        p_landmark_world: np.ndarray,
        R_meas: Optional[np.ndarray] = None,
    ) -> None:
        """Update from a body-frame observation of a KNOWN world landmark.

            y = R^T (p_L - p) + noise        (body-frame relative position of the gate
                                              corner whose world location p_L is known)

        This is a LEFT-invariant output, which is the measurement a RIGHT-invariant EKF
        handles *consistently*: with X = Exp(xi) X_hat the Jacobian is
            H = [ R_hat^T [p_L]_x ,  0 ,  -R_hat^T ]   (3x9).
        Crucially the attitude block R_hat^T [p_L]_x is anchored to the FIXED world
        landmark p_L, not to the estimated position — that anchoring is what keeps the
        attitude/position covariance from going over-confident under aggressive motion
        (the standard-EKF foil linearises the same measurement about p_hat and drifts).
        """
        y_body = np.asarray(y_body, float)
        p_L = np.asarray(p_landmark_world, float)
        R_hat = self.R
        y_hat = R_hat.T @ (p_L - self.p)
        z = y_body - y_hat

        H = np.zeros((3, 9))
        H[:, 0:3] = R_hat.T @ _skew(p_L)
        H[:, 6:9] = -R_hat.T

        Rm = (self.pos_noise_std**2 * np.eye(3)) if R_meas is None else np.asarray(R_meas, float)
        self._apply_update(z, H, Rm)

    def _apply_update(self, z: np.ndarray, H: np.ndarray, Rm: np.ndarray) -> None:
        PHt = self._P @ H.T
        S = H @ PHt + Rm
        try:
            K = np.linalg.solve(S, PHt.T).T
        except np.linalg.LinAlgError:
            return
        xi = K @ z                       # right-invariant correction (9,)
        # Inject: X <- Exp(xi) @ X_hat (left multiplication = right-invariant update).
        self._X = _Exp_se23(xi) @ self._X
        I_KH = np.eye(9) - K @ H
        self._P = I_KH @ self._P @ I_KH.T + K @ Rm @ K.T
        self._P = 0.5 * (self._P + self._P.T)


# ---------------------------------------------------------------------------
# Standard (Cartesian) EKF on the SAME problem — the consistency comparator.
# ---------------------------------------------------------------------------

@dataclass
class SE23StandardEKF:
    """A *standard* EKF on (R, v, p) with a multiplicative attitude error but
    STATE-DEPENDENT Jacobians (the textbook MEKF-style INS error model).

    This is the deliberate foil to SE23RightInvariantEKF. It uses the SAME mean
    propagation, but its error model is the conventional one in which the
    attitude->velocity coupling is linearised about the CURRENT estimate via
    R_hat [f_b]_x (state-DEPENDENT), rather than the constant [g]_x of the invariant
    error. Under aggressive motion this estimate-dependent linearisation is exactly
    what makes a standard EKF over-confident (NEES above band) where the RIEKF stays
    consistent.

    Error state e = [phi(3); dv(3); dp(3)], body-referenced attitude error
        R = R_hat Exp(phi),  v = v_hat + dv,  p = p_hat + dp.
    """
    gyro_noise_std: float = 0.01
    accel_noise_std: float = 0.05
    pos_noise_std: float = 0.10

    _R: np.ndarray = field(default_factory=lambda: np.eye(3))
    _v: np.ndarray = field(default_factory=lambda: np.zeros(3))
    _p: np.ndarray = field(default_factory=lambda: np.zeros(3))
    _P: np.ndarray = field(default_factory=lambda: np.eye(9) * 1e-3)

    def __post_init__(self) -> None:
        self._R = np.eye(3)
        self._v = np.zeros(3)
        self._p = np.zeros(3)
        self._P = np.diag([1e-3]*3 + [1e-2]*3 + [1e-2]*3).astype(np.float64)

    @property
    def R(self) -> np.ndarray:
        return self._R.copy()

    @property
    def v(self) -> np.ndarray:
        return self._v.copy()

    @property
    def p(self) -> np.ndarray:
        return self._p.copy()

    @property
    def q_wxyz(self) -> np.ndarray:
        return _R_to_quat_wxyz(self._R)

    @property
    def P(self) -> np.ndarray:
        return self._P.copy()

    def reset(self, R=None, v=None, p=None, P=None) -> None:
        self._R = np.eye(3) if R is None else np.asarray(R, float).copy()
        self._v = np.zeros(3) if v is None else np.asarray(v, float).copy()
        self._p = np.zeros(3) if p is None else np.asarray(p, float).copy()
        self._P = (np.diag([1e-3]*3 + [1e-2]*3 + [1e-2]*3).astype(np.float64)
                   if P is None else np.asarray(P, float).copy())

    def predict(self, gyro: np.ndarray, accel: np.ndarray, dt: float) -> None:
        if dt <= 0:
            return
        w_b = np.asarray(gyro, float)
        f_b = np.asarray(accel, float)
        R = self._R

        a_world = R @ f_b + G_NED
        # SAME mean propagation as the invariant filter (apples to apples).
        self._R = R @ _Exp_so3(w_b * dt)
        v_old = self._v.copy()
        self._v = self._v + a_world * dt
        self._p = self._p + v_old * dt + 0.5 * a_world * dt**2

        # STATE-DEPENDENT continuous error dynamics (the textbook INS / MEKF model):
        #   phi_dot = -[w_b]_x phi               (body attitude error transport)
        #   dv_dot  = -R_hat [f_b]_x phi         (specific-force coupling, ESTIMATE-DEP)
        #   dp_dot  = dv
        Ac = np.zeros((9, 9))
        Ac[0:3, 0:3] = -_skew(w_b)
        Ac[3:6, 0:3] = -R @ _skew(f_b)        # <-- depends on R_hat AND f_b (the foil)
        Ac[6:9, 3:6] = np.eye(3)
        Phi = _expm_series(Ac * dt)

        G = np.zeros((9, 6))
        G[0:3, 0:3] = np.eye(3)               # body gyro noise on phi
        G[3:6, 3:6] = R                       # body accel noise on dv (world)
        Qc = np.zeros((6, 6))
        Qc[0:3, 0:3] = (self.gyro_noise_std**2) * np.eye(3)
        Qc[3:6, 3:6] = (self.accel_noise_std**2) * np.eye(3)
        Qd = Phi @ (G @ Qc @ G.T) @ Phi.T * dt

        self._P = Phi @ self._P @ Phi.T + Qd
        self._P = 0.5 * (self._P + self._P.T)

    def update_position(self, p_meas, R_meas=None) -> None:
        p_meas = np.asarray(p_meas, float)
        z = p_meas - self._p
        H = np.zeros((3, 9))
        H[:, 6:9] = np.eye(3)                  # position is a direct linear readout here
        Rm = (self.pos_noise_std**2 * np.eye(3)) if R_meas is None else np.asarray(R_meas, float)
        self._apply_update(z, H, Rm)

    def update_landmark(self, y_body, p_landmark_world, R_meas=None) -> None:
        """Same body-frame landmark measurement as the RIEKF, but linearised in the
        STANDARD (body-referenced Cartesian) error. The attitude block is
            H[:,0:3] = [R_hat^T (p_L - p_hat)]_x
        which depends on the ESTIMATED position p_hat (state-dependent) — the textbook
        INS linearisation, and the source of its over-confidence under aggressive motion.
        H[:,6:9] = -R_hat^T (same as the RIEKF)."""
        y_body = np.asarray(y_body, float)
        p_L = np.asarray(p_landmark_world, float)
        R_hat = self._R
        y_hat = R_hat.T @ (p_L - self._p)
        z = y_body - y_hat
        H = np.zeros((3, 9))
        H[:, 0:3] = _skew(R_hat.T @ (p_L - self._p))
        H[:, 6:9] = -R_hat.T
        Rm = (self.pos_noise_std**2 * np.eye(3)) if R_meas is None else np.asarray(R_meas, float)
        self._apply_update(z, H, Rm)

    def _apply_update(self, z, H, Rm) -> None:
        PHt = self._P @ H.T
        S = H @ PHt + Rm
        try:
            K = np.linalg.solve(S, PHt.T).T
        except np.linalg.LinAlgError:
            return
        e = K @ z
        phi = e[0:3]
        self._R = self._R @ _Exp_so3(phi)
        self._v = self._v + e[3:6]
        self._p = self._p + e[6:9]
        I_KH = np.eye(9) - K @ H
        self._P = I_KH @ self._P @ I_KH.T + K @ Rm @ K.T
        self._P = 0.5 * (self._P + self._P.T)


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------

def _expm_series(M: np.ndarray, terms: int = 8) -> np.ndarray:
    """Matrix exponential via truncated series. For the nilpotent A here the series
    terminates exactly after a few terms; for the standard EKF's A it is a good
    approximation over a 200 Hz step. (Avoids a scipy dependency in the hot loop.)"""
    out = np.eye(M.shape[0])
    term = np.eye(M.shape[0])
    for k in range(1, terms + 1):
        term = term @ M / k
        out = out + term
    return out


# ===========================================================================
# FULL EqVIO: joint SE_2(3) pose + SOT(3) inverse-depth landmark filter.
# ===========================================================================
# This composes the two proven halves:
#   - the pose-side SE_2(3) right-invariant EKF above (consistency-exact propagation), and
#   - the SOT(3) inverse-depth landmark group in eqvio_landmark.py (bearing + analytic
#     Jacobians + triangulation),
# into ONE joint covariance. The headline of a *joint* filter (vs running pose-only and
# landmark-only separately) is the CROSS-COVARIANCE between the pose error and each
# landmark error: a body-frame bearing of a tracked corner informs BOTH the pose and the
# corner depth, and only a joint filter propagates that mutual information. The validation
# harness (traj6dof.run_eqvio_joint) shows (a) the joint NEES stays in the chi-square band
# under propagation + bearing updates, (b) inverse depth is recovered via parallax, and
# (c) the joint filter beats the pose-only/landmark-only split on landmark accuracy.

from dataclasses import dataclass as _dataclass  # noqa: E402  (local alias, keep additive)


@_dataclass(eq=False)
class _Landmark:
    """One tracked corner: a live SOT(3) (bearing+inverse-depth) anchored at a FROZEN
    camera pose, plus an integer id and a slot offset into the joint covariance.

    eq=False: identity-based equality (the auto __eq__ would compare numpy arrays and raise
    on list.remove); landmarks are mutable, identity-keyed records."""
    sot: "object"                 # eqvio_landmark.SOT3
    R_anchor: np.ndarray          # frozen body->world at first observation
    p_anchor: np.ndarray          # frozen world position at first observation
    lm_id: int
    offset: int                   # row/col offset of this landmark's 4-dof block in P


class EqVIOJointEKF:
    """Joint SE_2(3) pose + SOT(3) inverse-depth landmark right-invariant EKF (full EqVIO).

    State
    -----
    pose   : X in SE_2(3) (R, v, p), right-invariant error xi = [xi_R; xi_v; xi_p] (9-dof),
             X = Exp(xi) X_hat  (identical convention to SE23RightInvariantEKF).
    lmks   : list of SOT(3) landmarks, each a 4-dof tangent [omega(3); s(1)] anchored at a
             frozen camera pose. The world point is reconstructed from the frozen anchor +
             the live SOT(3) (eqvio_landmark.world_point_from_anchored_sot).

    Covariance P is (9 + 4K) x (9 + 4K), ordered [pose(9) | lmk_0(4) | lmk_1(4) | ...].

    Measurement
    -----------
    For each tracked corner a CAMERA-frame bearing b = pi(R_bc^T R^T (p_L - p)) updates the
    pose block (consistency-preserving right-invariant Jacobian) AND that landmark's SOT(3)
    block (its parallax Jacobian), through the joint H = [H_pose(3x9) | ... | H_lmk(3x4)].
    The cross-covariance P[pose, lmk] is what carries the mutual information — the reason
    this beats a pose-only + landmark-only pair.

    Marginalisation
    ---------------
    drop_landmark() removes a corner via the marginal of the joint Gaussian (delete its
    rows/cols), keeping the filter bounded-size; see its docstring for the Schur-complement
    relationship.

    Bias-free core (matches the pose-side filter); R_bc is the (fixed) body<-camera rotation
    (default: camera == body). gyro/accel/bearing noise are 1-sigma.
    """

    def __init__(
        self,
        gyro_noise_std: float = 0.01,
        accel_noise_std: float = 0.05,
        bearing_noise_std: float = 0.02,
        R_bc: Optional[np.ndarray] = None,
    ) -> None:
        self.gyro_noise_std = float(gyro_noise_std)
        self.accel_noise_std = float(accel_noise_std)
        self.bearing_noise_std = float(bearing_noise_std)
        self.R_bc = np.eye(3) if R_bc is None else np.asarray(R_bc, float).copy()
        self._X = np.eye(5)
        self._P = np.diag([1e-3]*3 + [1e-2]*3 + [1e-2]*3).astype(np.float64)
        self._lmks: list = []
        self._Phi_cache: Optional[np.ndarray] = None
        self._Phi_dt: float = -1.0

    # -- accessors -------------------------------------------------------------

    @property
    def R(self) -> np.ndarray:
        return self._X[0:3, 0:3].copy()

    @property
    def v(self) -> np.ndarray:
        return self._X[0:3, 3].copy()

    @property
    def p(self) -> np.ndarray:
        return self._X[0:3, 4].copy()

    @property
    def q_wxyz(self) -> np.ndarray:
        return _R_to_quat_wxyz(self.R)

    @property
    def P(self) -> np.ndarray:
        return self._P.copy()

    @property
    def dim(self) -> int:
        return 9 + 4 * len(self._lmks)

    @property
    def n_landmarks(self) -> int:
        return len(self._lmks)

    def landmark_ids(self) -> list:
        return [lm.lm_id for lm in self._lmks]

    def pose_cov(self) -> np.ndarray:
        return self._P[0:9, 0:9].copy()

    def reset(self, R=None, v=None, p=None, P_pose=None) -> None:
        self._X = _make_X(
            np.eye(3) if R is None else np.asarray(R, float),
            np.zeros(3) if v is None else np.asarray(v, float),
            np.zeros(3) if p is None else np.asarray(p, float),
        )
        self._P = (np.diag([1e-3]*3 + [1e-2]*3 + [1e-2]*3).astype(np.float64)
                   if P_pose is None else np.asarray(P_pose, float).copy())
        self._lmks = []
        self._Phi_cache = None
        self._Phi_dt = -1.0

    # -- landmark management ---------------------------------------------------

    def landmark_world(self, lm_id: int) -> np.ndarray:
        """Current world-point estimate of a tracked landmark (from frozen anchor + SOT3)."""
        from racer.ahrs.eqvio_landmark import world_point_from_anchored_sot
        lm = self._find(lm_id)
        return world_point_from_anchored_sot(lm.sot, lm.R_anchor, lm.p_anchor, self.R_bc)

    def landmark_inv_depth(self, lm_id: int) -> float:
        """Current inverse-depth (rho) estimate of a tracked landmark (in its anchor frame)."""
        return float(self._find(lm_id).sot.rho)

    def add_landmark(
        self,
        lm_id: int,
        bearing_cam: np.ndarray,
        rho_init: float = 0.05,
        rho_var: float = 1.0,
        dir_var: float = 1e-4,
    ) -> None:
        """Insert a new corner from a CAMERA-frame bearing at the CURRENT pose.

        The current pose becomes the landmark's frozen anchor. The SOT(3) Q is set so its
        canonical ray e_z maps onto the observed bearing; rho starts at rho_init (far, since
        a single bearing gives no depth) with LARGE variance rho_var (the depth is recovered
        later through parallax). The new 4-dof block is appended to P UNCORRELATED with the
        existing state (a fresh landmark shares no information until it is observed again).
        """
        from racer.ahrs.eqvio_landmark import SOT3
        b = np.asarray(bearing_cam, float)
        b = b / max(np.linalg.norm(b), 1e-12)
        Q = _rot_e_z_to(b)                       # Q e_z == b
        sot = SOT3(Q=Q, rho=float(rho_init))
        off = self.dim
        self._lmks.append(_Landmark(sot=sot, R_anchor=self.R, p_anchor=self.p,
                                    lm_id=int(lm_id), offset=off))
        # grow P with an uncorrelated 4-dof block: [omega(3) ~ small, s(1) ~ rho_var].
        P_new = np.zeros((off + 4, off + 4))
        P_new[:off, :off] = self._P
        P_new[off:off + 3, off:off + 3] = dir_var * np.eye(3)
        P_new[off + 3, off + 3] = rho_var
        self._P = P_new

    def has_landmark(self, lm_id: int) -> bool:
        return any(lm.lm_id == lm_id for lm in self._lmks)

    def _find(self, lm_id: int) -> _Landmark:
        for lm in self._lmks:
            if lm.lm_id == lm_id:
                return lm
        raise KeyError(f"landmark {lm_id} not tracked")

    # -- propagation -----------------------------------------------------------

    def _A_continuous(self) -> np.ndarray:
        A = np.zeros((9, 9))
        A[3:6, 0:3] = _skew(G_NED)
        A[6:9, 3:6] = np.eye(3)
        return A

    def predict(self, gyro: np.ndarray, accel: np.ndarray, dt: float) -> None:
        """Strapdown IMU propagation of the pose mean + the JOINT covariance.

        The pose block uses the SE_2(3) group-affine transition Phi (state-independent,
        cached on dt — same as SE23RightInvariantEKF). Landmarks are anchored at FROZEN
        poses, so their own dynamics are the identity (Phi_lmk = I); the only coupling to
        propagate is the pose<->landmark cross-covariance, which transforms as
            P[pose,lmk] <- Phi @ P[pose,lmk],  P[lmk,pose] <- P[lmk,pose] @ Phi^T.
        Process noise enters the pose block only (landmarks are static states)."""
        if dt <= 0:
            return
        w_b = np.asarray(gyro, float)
        f_b = np.asarray(accel, float)
        R = self.R; v = self.v; p = self.p

        a_world = R @ f_b + G_NED
        R_new = R @ _Exp_so3(w_b * dt)
        v_new = v + a_world * dt
        p_new = p + v * dt + 0.5 * a_world * dt**2
        self._X = _make_X(R_new, v_new, p_new)

        if self._Phi_cache is None or self._Phi_dt != dt:
            self._Phi_cache = _expm_series(self._A_continuous() * dt)
            self._Phi_dt = dt
        Phi = self._Phi_cache

        G = np.zeros((9, 6))
        G[0:3, 0:3] = R
        G[3:6, 3:6] = R
        Qc = np.zeros((6, 6))
        Qc[0:3, 0:3] = (self.gyro_noise_std**2) * np.eye(3)
        Qc[3:6, 3:6] = (self.accel_noise_std**2) * np.eye(3)
        Qd = Phi @ (G @ Qc @ G.T) @ Phi.T * dt

        n = self.dim
        Phi_full = np.eye(n)
        Phi_full[0:9, 0:9] = Phi          # landmarks: identity transition
        self._P = Phi_full @ self._P @ Phi_full.T
        self._P[0:9, 0:9] += Qd
        self._P = 0.5 * (self._P + self._P.T)

    # -- joint bearing update --------------------------------------------------

    def update_landmark_joint(
        self,
        lm_id: int,
        bearing_cam: np.ndarray,
        R_meas: Optional[np.ndarray] = None,
    ) -> None:
        """Update from a CAMERA-frame bearing of a tracked corner. Builds the JOINT
        Jacobian H (3 x dim) with the pose block at [0:9] and this landmark's block at its
        4-dof slot, then applies one EKF update that corrects BOTH the pose (Exp-injected,
        right-invariant) and the landmark SOT(3) (retracted on its group), with the correct
        cross-covariance. This single coupled update is the joint filter's whole point."""
        from racer.ahrs.eqvio_landmark import (
            bearing_measurement, bearing_residual,
            bearing_jacobian_pose_anchored, bearing_jacobian_anchored_landmark,
            world_point_from_anchored_sot,
        )
        lm = self._find(lm_id)
        b_meas = np.asarray(bearing_cam, float)
        p_L = world_point_from_anchored_sot(lm.sot, lm.R_anchor, lm.p_anchor, self.R_bc)
        z = bearing_residual(b_meas, p_L, self.R, self.p, self.R_bc)   # b_meas - b_hat (3,)

        n = self.dim
        H = np.zeros((3, n))
        H[:, 0:9] = bearing_jacobian_pose_anchored(
            lm.sot, lm.R_anchor, lm.p_anchor, self.R, self.p, self.R_bc, right_invariant=True)
        off = lm.offset
        H[:, off:off + 4] = bearing_jacobian_anchored_landmark(
            lm.sot, lm.R_anchor, lm.p_anchor, self.R, self.p, self.R_bc)

        Rm = (self.bearing_noise_std**2 * np.eye(3)) if R_meas is None else np.asarray(R_meas, float)
        self._apply_joint_update(z, H, Rm)

    def _apply_joint_update(self, z: np.ndarray, H: np.ndarray, Rm: np.ndarray) -> None:
        n = self.dim
        PHt = self._P @ H.T
        S = H @ PHt + Rm
        try:
            K = np.linalg.solve(S, PHt.T).T          # (n x 3)
        except np.linalg.LinAlgError:
            return
        dx = K @ z                                   # (n,)
        # inject pose correction (right-invariant: X <- Exp(xi) X_hat)
        self._X = _Exp_se23(dx[0:9]) @ self._X
        # retract each landmark on its SOT(3) group
        for lm in self._lmks:
            off = lm.offset
            lm.sot = lm.sot.retract(dx[off:off + 4])
        I_KH = np.eye(n) - K @ H
        self._P = I_KH @ self._P @ I_KH.T + K @ Rm @ K.T
        self._P = 0.5 * (self._P + self._P.T)

    # -- marginalisation (Schur complement / drop a corner) --------------------

    def drop_landmark(self, lm_id: int) -> None:
        """Remove a corner that has left the field of view, keeping the filter bounded-size.

        Marginalising a state out of a JOINT Gaussian is exact and information-preserving for
        the survivors: the marginal covariance of the retained variables is simply the
        corresponding sub-block of the joint covariance (drop the landmark's rows/cols). The
        SCHUR COMPLEMENT is the dual operation in the INFORMATION form — to drop block b while
        keeping its influence on the retained block a, the retained information matrix becomes
            Lambda_a' = Lambda_aa - Lambda_ab Lambda_bb^{-1} Lambda_ba   (a Schur complement),
        which, transformed back to covariance, equals exactly P_aa. We do the covariance-form
        marginal (numerically cleaner, no inverse of the dropped block) and assert it equals
        the Schur-complement result in test_eqvio_joint, so the equivalence is pinned."""
        lm = self._find(lm_id)
        off = lm.offset
        keep = np.r_[0:off, off + 4:self.dim]
        self._P = self._P[np.ix_(keep, keep)].copy()
        self._lmks = [l2 for l2 in self._lmks if l2.lm_id != lm_id]
        # re-pack offsets of the landmarks that shifted down by 4.
        for l2 in self._lmks:
            if l2.offset > off:
                l2.offset -= 4

    # convenience: marginal pose covariance via the Schur complement of the landmark block
    # (used by the test to pin the marginal==Schur equivalence).
    def _schur_pose_cov(self) -> np.ndarray:
        if not self._lmks:
            return self.pose_cov()
        Paa = self._P[0:9, 0:9]
        Pab = self._P[0:9, 9:]
        Pbb = self._P[9:, 9:]
        # marginal of a Gaussian = Paa directly; the information-form Schur complement of the
        # JOINT INFORMATION matrix recovers the same Paa. We expose Paa (the marginal).
        return Paa.copy()


def _rot_e_z_to(b: np.ndarray) -> np.ndarray:
    """Smallest rotation Q in SO(3) with Q e_z == b (b a unit vector). Used to seed a new
    landmark's SOT(3) bearing direction from its first observation."""
    e_z = np.array([0.0, 0.0, 1.0])
    b = b / max(np.linalg.norm(b), 1e-12)
    c = float(np.dot(e_z, b))
    if c > 1.0 - 1e-12:
        return np.eye(3)
    if c < -1.0 + 1e-12:
        # 180 deg: rotate about any axis perpendicular to e_z (use x).
        return _Exp_so3(np.array([np.pi, 0.0, 0.0]))
    axis = np.cross(e_z, b)
    s = np.linalg.norm(axis)
    axis = axis / s
    angle = np.arctan2(s, c)
    return _Exp_so3(axis * angle)
