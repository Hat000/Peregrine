"""Synthetic 6-DoF trajectory generator for the SE_2(3) estimator bench.

Extends the AHRS bench's attitude-only IMU generator (imu_gen.py) to FULL
attitude+velocity+position ground truth, so we can drive the coupled SE_2(3)
right-invariant EKF and a standard EKF and compare them on NEES/consistency.

The trajectory is specified analytically in the WORLD frame (NED) for position p(t),
then differentiated for v(t) and a_world(t), and an analytic attitude profile R(t) is
chosen (aggressive in the high-g cases). Body IMU readings are then synthesised exactly:
    gyro_b(t)  = vee( R^T R_dot )            (body angular rate)
    accel_b(t) = R^T ( a_world(t) - g_ned )  (body specific force; rest = [0,0,-g])
This is the inverse of the estimator's forward model, so a noiseless run is an exact
consistency probe (Taylor-error isolation: any divergence is linearisation, not noise).

Quaternion convention: (w,x,y,z) scalar-first; FRD body, NED world; g=[0,0,+9.80665].
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np

from racer.ahrs.iekf import _skew, _Exp_so3, _R_to_quat_wxyz


GRAVITY = 9.80665
G_NED = np.array([0.0, 0.0, GRAVITY])


@dataclass
class Traj6DoF:
    """Full 6-DoF synthetic trajectory + IMU + position measurements.

    t        : (N,) seconds.
    R_gt     : (N,3,3) body->world rotation.
    q_gt     : (N,4) (w,x,y,z).
    v_gt     : (N,3) world velocity (NED).
    p_gt     : (N,3) world position (NED).
    gyro     : (N,3) body angular rate (rad/s), with noise/bias.
    accel    : (N,3) body specific force (m/s^2), with noise/bias.
    pos_meas : (M,3) noisy world-position measurements.
    pos_idx  : (M,) indices into t at which pos_meas were taken.
    landmarks_world : (L,3) known world positions of gate corners.
    lmk_obs  : (M, L, 3) body-frame noisy observations y = R^T(p_L - p) at each pos_idx.
    name     : label.
    """
    t: np.ndarray
    R_gt: np.ndarray
    q_gt: np.ndarray
    v_gt: np.ndarray
    p_gt: np.ndarray
    gyro: np.ndarray
    accel: np.ndarray
    pos_meas: np.ndarray
    pos_idx: np.ndarray
    landmarks_world: np.ndarray = None       # (L,3)
    lmk_obs: np.ndarray = None               # (M,L,3)
    name: str = "traj6dof"

    @property
    def dt(self) -> float:
        return float(np.mean(np.diff(self.t))) if len(self.t) > 1 else 0.005

    @property
    def N(self) -> int:
        return len(self.t)


class Traj(Enum):
    HOVER          = auto()   # stationary; trivial sanity.
    GENTLE_ORBIT   = auto()   # slow circle, mild bank. EKF and RIEKF both fine.
    AGGRESSIVE_S   = auto()   # fast S-weave with hard banks (racing-like).
    HIGH_G_LOOP    = auto()   # vertical loop @ several g — maximally stresses coupling.


# ---------------------------------------------------------------------------
# analytic world-frame position/velocity/acceleration profiles
# ---------------------------------------------------------------------------

def _profile(traj: Traj, t: np.ndarray):
    """Return p(t),(N,3) v(t),(N,3) a_world(t),(N,3) and a heading-rate hint yaw(t)."""
    N = len(t)
    p = np.zeros((N, 3)); v = np.zeros((N, 3)); a = np.zeros((N, 3))

    if traj == Traj.HOVER:
        pass

    elif traj == Traj.GENTLE_ORBIT:
        Rr, w = 8.0, 2 * np.pi / 6.0     # 8 m radius, 6 s period
        p[:, 0] = Rr * np.cos(w * t)
        p[:, 1] = Rr * np.sin(w * t)
        p[:, 2] = -2.0                    # 2 m up (NED down=+Z)
        v[:, 0] = -Rr * w * np.sin(w * t)
        v[:, 1] = Rr * w * np.cos(w * t)
        a[:, 0] = -Rr * w * w * np.cos(w * t)
        a[:, 1] = -Rr * w * w * np.sin(w * t)

    elif traj == Traj.AGGRESSIVE_S:
        # fast forward run with a strong lateral weave + altitude undulation
        vx = 18.0
        Ay, wy = 6.0, 2 * np.pi / 1.5
        Az, wz = 1.5, 2 * np.pi / 2.0
        p[:, 0] = vx * t
        p[:, 1] = Ay * np.sin(wy * t)
        p[:, 2] = -3.0 + Az * np.sin(wz * t)
        v[:, 0] = vx
        v[:, 1] = Ay * wy * np.cos(wy * t)
        v[:, 2] = Az * wz * np.cos(wz * t)
        a[:, 1] = -Ay * wy * wy * np.sin(wy * t)
        a[:, 2] = -Az * wz * wz * np.sin(wz * t)

    elif traj == Traj.HIGH_G_LOOP:
        # vertical loop in the x-z plane at high speed -> strong centripetal g
        Rr, w = 10.0, 2 * np.pi / 2.5    # 10 m loop, 2.5 s period -> ~6-7 g peak
        p[:, 0] = Rr * np.sin(w * t)
        p[:, 2] = -Rr * (1 - np.cos(w * t))   # up at top
        v[:, 0] = Rr * w * np.cos(w * t)
        v[:, 2] = -Rr * w * np.sin(w * t)
        a[:, 0] = -Rr * w * w * np.sin(w * t)
        a[:, 2] = -Rr * w * w * np.cos(w * t)

    return p, v, a


def _attitude_from_motion(traj: Traj, t: np.ndarray, v: np.ndarray, a: np.ndarray):
    """Choose an analytic R(t) (body->world) that is coordinated with the motion.

    Body x (forward, FRD) points along velocity; body z (down) points along the
    'apparent down' = -(a_world - g)/|.| (thrust axis), which gives the aggressive
    rotations that stress the coupled filter. Falls back to level for hover.
    """
    N = len(t)
    R = np.zeros((N, 3, 3))
    for i in range(N):
        vi = v[i]
        speed = np.linalg.norm(vi)
        # forward axis (body x) in world
        if speed > 1e-3:
            x_b = vi / speed
        else:
            x_b = np.array([1.0, 0.0, 0.0])
        # apparent-down = direction the thrust must oppose: -(a - g) points 'up'(thrust),
        # so body-down z_b is along (a_world - g) normalised... but for hover a=0 -> g.
        sf_world = a[i] - G_NED            # = -thrust direction * |.|
        if np.linalg.norm(sf_world) > 1e-3:
            z_b = sf_world / np.linalg.norm(sf_world)
        else:
            z_b = np.array([0.0, 0.0, 1.0])
        # re-orthonormalise: y = z x x, then x = y x z
        y_b = np.cross(z_b, x_b)
        if np.linalg.norm(y_b) < 1e-6:
            # x and z nearly parallel; pick an arbitrary perpendicular
            y_b = np.cross(z_b, np.array([0.0, 1.0, 0.0]))
        y_b /= np.linalg.norm(y_b)
        x_b = np.cross(y_b, z_b)
        x_b /= np.linalg.norm(x_b)
        R[i] = np.column_stack([x_b, y_b, z_b])
    return R


def generate_traj6dof(
    traj: Traj,
    duration_s: float = 5.0,
    dt: float = 0.005,
    gyro_noise_std: float = 0.01,
    accel_noise_std: float = 0.05,
    gyro_bias: Optional[np.ndarray] = None,
    accel_bias: Optional[np.ndarray] = None,
    pos_noise_std: float = 0.10,
    pos_rate_hz: float = 10.0,
    seed: int = 42,
) -> Traj6DoF:
    """Generate a full 6-DoF synthetic trajectory with IMU + position measurements.

    Set gyro_noise_std=accel_noise_std=pos_noise_std=0 for the NOISELESS consistency
    probe (isolates Taylor/linearisation error). pos_rate_hz: how often a world-position
    fix is produced (e.g. 10 Hz gate-anchored position).
    """
    rng = np.random.default_rng(seed)
    N = int(round(duration_s / dt))
    t = np.arange(N) * dt

    p_an, v_an, a_world = _profile(traj, t)
    R_gt = _attitude_from_motion(traj, t, v_an, a_world)
    q_gt = np.array([_R_to_quat_wxyz(R_gt[i]) for i in range(N)])

    # body angular rate: the gyro that makes the filter's discrete propagation
    # R_{i+1} = R_i Exp(w_i dt) EXACT, i.e. w_i = Log(R_i^T R_{i+1}) / dt. So a noiseless
    # run reproduces the attitude path to machine precision (clean Taylor-error probe).
    gyro = np.zeros((N, 3))
    for i in range(N):
        i1 = min(N - 1, i + 1)
        ddt = t[i1] - t[i] if i1 > i else dt
        if ddt <= 0:
            ddt = dt
        gyro[i] = _log_so3(R_gt[i].T @ R_gt[i1]) / ddt

    # body specific force from the ANALYTIC world acceleration: accel = R^T (a_world - g).
    accel = np.zeros((N, 3))
    for i in range(N):
        accel[i] = R_gt[i].T @ (a_world[i] - G_NED)

    # Re-derive v_gt, p_gt by the SAME discrete integrator the estimators use, seeded at
    # the analytic initial condition. This makes GT *discrete-consistent*: a noiseless,
    # perfectly-initialised filter reproduces (R_gt, v_gt, p_gt) to machine precision, so
    # any residual NEES growth is pure linearisation (Taylor) error — exactly what we want
    # to attribute to the filter, not to a generator/integrator mismatch.
    v_gt = np.zeros((N, 3)); p_gt = np.zeros((N, 3))
    v_gt[0] = v_an[0]; p_gt[0] = p_an[0]
    for i in range(N - 1):
        a_w = R_gt[i] @ accel[i] + G_NED
        p_gt[i + 1] = p_gt[i] + v_gt[i] * dt + 0.5 * a_w * dt**2
        v_gt[i + 1] = v_gt[i] + a_w * dt

    if gyro_bias is not None:
        gyro = gyro + np.asarray(gyro_bias, float)
    if accel_bias is not None:
        accel = accel + np.asarray(accel_bias, float)
    gyro = gyro + rng.normal(0.0, gyro_noise_std, (N, 3))
    accel = accel + rng.normal(0.0, accel_noise_std, (N, 3))

    # position measurements at pos_rate_hz
    step = max(1, int(round((1.0 / pos_rate_hz) / dt)))
    pos_idx = np.arange(0, N, step)
    pos_meas = p_gt[pos_idx] + rng.normal(0.0, pos_noise_std, (len(pos_idx), 3))

    # known world landmarks (gate corners): a 1.5 m inner square ~ along the path.
    # Anchored near the trajectory centroid so they stay in front of the body.
    centroid = p_gt.mean(axis=0)
    half = 0.75
    landmarks_world = centroid + np.array([
        [12.0,  half,  half],
        [12.0, -half,  half],
        [12.0,  half, -half],
        [12.0, -half, -half],
    ])
    L = landmarks_world.shape[0]
    M = len(pos_idx)
    lmk_obs = np.zeros((M, L, 3))
    for m, idx in enumerate(pos_idx):
        for l in range(L):
            y = R_gt[idx].T @ (landmarks_world[l] - p_gt[idx])
            lmk_obs[m, l] = y + rng.normal(0.0, pos_noise_std, 3)

    return Traj6DoF(
        t=t, R_gt=R_gt, q_gt=q_gt, v_gt=v_gt, p_gt=p_gt,
        gyro=gyro.astype(np.float64), accel=accel.astype(np.float64),
        pos_meas=pos_meas.astype(np.float64), pos_idx=pos_idx.astype(int),
        landmarks_world=landmarks_world.astype(np.float64),
        lmk_obs=lmk_obs.astype(np.float64),
        name=traj.name,
    )


# ---------------------------------------------------------------------------
# consistency evaluation (NEES) helpers
# ---------------------------------------------------------------------------

def run_se23_filter(filt, seq: Traj6DoF, R0=None, v0=None, p0=None, P0=None,
                    measurement: str = "landmark"):
    """Run an SE_2(3) filter (invariant or standard) over the trajectory.

    Initialises at GT (perfect init) unless overrides are given, then propagates with
    IMU and applies measurement updates at pos_idx. Returns dict of per-step estimates
    and the per-fix NEES (full 9-dof + 3-dof attitude/position marginals).

    measurement : "landmark" (body-frame gate-corner observations — the LEFT-invariant
                  output the RIEKF is designed for, and the actual VQ2 measurement), or
                  "position" (absolute world position — a RIGHT-invariant output that the
                  standard EKF handles natively; included for completeness).
    """
    R0 = seq.R_gt[0] if R0 is None else R0
    v0 = seq.v_gt[0] if v0 is None else v0
    p0 = seq.p_gt[0] if p0 is None else p0
    filt.reset(R=R0, v=v0, p=p0, P=P0)

    N = seq.N
    dt = seq.dt
    pos_set = set(int(i) for i in seq.pos_idx)
    pos_lookup = {int(idx): k for k, idx in enumerate(seq.pos_idx)}

    q_est = np.zeros((N, 4))
    p_est = np.zeros((N, 3))
    nees_full, nees_att, nees_pos, nees_t = [], [], [], []

    for i in range(N):
        if i > 0:
            # gyro[k]/accel[k] are the body rates/specific-force carrying the state from
            # sample k to k+1 (generator convention), so to land ON sample i we apply
            # the inputs from i-1.
            filt.predict(seq.gyro[i - 1], seq.accel[i - 1], dt)
        if i in pos_set:
            k = pos_lookup[i]
            if measurement == "position":
                filt.update_position(seq.pos_meas[k])
            else:
                for l in range(seq.landmarks_world.shape[0]):
                    filt.update_landmark(seq.lmk_obs[k, l], seq.landmarks_world[l])

        q_est[i] = filt.q_wxyz
        p_est[i] = filt.p

        # NEES at every position-fix step (post-update), against the GT error vector.
        if i in pos_set and i > 0:
            err = _error_vector(filt, seq, i)         # (9,)
            P = filt.P
            try:
                Pinv = np.linalg.inv(P)
            except np.linalg.LinAlgError:
                continue
            nees_full.append(float(err @ Pinv @ err))
            # attitude marginal (3-dof) and position marginal (3-dof)
            ea, Pa = err[0:3], P[0:3, 0:3]
            ep, Pp = err[6:9], P[6:9, 6:9]
            nees_att.append(float(ea @ np.linalg.inv(Pa) @ ea))
            nees_pos.append(float(ep @ np.linalg.inv(Pp) @ ep))
            nees_t.append(i * dt)

    return {
        "q_est": q_est, "p_est": p_est,
        "nees_full": np.array(nees_full),
        "nees_att": np.array(nees_att),
        "nees_pos": np.array(nees_pos),
        "nees_t": np.array(nees_t),
    }


def _error_vector(filt, seq: Traj6DoF, i: int) -> np.ndarray:
    """The 9-dof error vector consistent with the filter's OWN error convention.

    For the invariant filter the covariance lives in the RIGHT-invariant tangent
    (X = Exp(xi) X_hat). For the standard EKF it lives in the body-referenced
    Cartesian tangent (R = R_hat Exp(phi); dv,dp additive). We detect which by class
    name so each NEES is computed against the matching error parameterisation — that is
    the whole point of a fair consistency comparison.
    """
    R_hat = filt.R; v_hat = filt.v; p_hat = filt.p
    R_gt = seq.R_gt[i]; v_gt = seq.v_gt[i]; p_gt = seq.p_gt[i]

    cls = type(filt).__name__
    if "RightInvariant" in cls:
        # X = Exp(xi) X_hat  ->  xi_R = Log(R_gt R_hat^T); for v,p the right-invariant
        # error is delta in world frame coupled through the lever arms. To first order:
        #   xi_R = Log(R_gt R_hat^T)
        #   xi_v = (v_gt - v_hat) - [.]   ; xi_p = (p_gt - p_hat) - [.]
        # We use the exact group log of eta = X_gt X_hat^{-1} for correctness.
        xi = _log_se23_error(R_gt, v_gt, p_gt, R_hat, v_hat, p_hat)
        return xi
    else:
        # standard EKF: phi = Log(R_hat^T R_gt) (body), dv, dp additive
        phi = _log_so3(R_hat.T @ R_gt)
        dv = v_gt - v_hat
        dp = p_gt - p_hat
        return np.concatenate([phi, dv, dp])


def _log_so3(R: np.ndarray) -> np.ndarray:
    """SO(3) log map -> rotation vector."""
    cos_t = (np.trace(R) - 1.0) / 2.0
    cos_t = np.clip(cos_t, -1.0, 1.0)
    theta = np.arccos(cos_t)
    if theta < 1e-8:
        # near identity: vee of skew part
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) * 0.5
    w = theta / (2.0 * np.sin(theta)) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w


def _log_se23_error(R_gt, v_gt, p_gt, R_hat, v_hat, p_hat) -> np.ndarray:
    """Log of the right-invariant error eta = X_gt X_hat^{-1} in se_2(3) (9-vec).

    X_hat^{-1} = | R_hat^T  -R_hat^T v_hat  -R_hat^T p_hat |
    eta = X_gt X_hat^{-1} = | R_gt R_hat^T   v_gt - (R_gt R_hat^T) v_hat   p_gt - (R_gt R_hat^T) p_hat |
    Then xi = Log_SE23(eta): xi_R = Log(eta_R); xi_v = J_l^{-1} eta_v; xi_p = J_l^{-1} eta_p.
    """
    eta_R = R_gt @ R_hat.T
    eta_v = v_gt - eta_R @ v_hat
    eta_p = p_gt - eta_R @ p_hat
    xi_R = _log_so3(eta_R)
    Jl_inv = _left_jacobian_inv_so3(xi_R)
    xi_v = Jl_inv @ eta_v
    xi_p = Jl_inv @ eta_p
    return np.concatenate([xi_R, xi_v, xi_p])


def _left_jacobian_inv_so3(phi: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(phi))
    K = _skew(phi)
    if theta < 1e-8:
        return np.eye(3) - 0.5 * K + (1.0 / 12.0) * (K @ K)
    half = theta / 2.0
    c = (1.0 / theta**2) * (1.0 - (theta * np.cos(half)) / (2.0 * np.sin(half)))
    return np.eye(3) - 0.5 * K + c * (K @ K)


# ===========================================================================
# JOINT EqVIO harness: per-frame CAMERA bearings + a runner for EqVIOJointEKF.
# ===========================================================================
# The pose-only bench above feeds KNOWN-p_L body-frame observations. The JOINT filter
# instead consumes per-frame *camera bearings* of corners whose depth it must triangulate.
# These helpers synthesise that bearing stream (noiseless => exact parallax probe) and run
# the joint filter, returning NEES (pose + joint), landmark depth-recovery error, and the
# pose-only / landmark-only baselines for the does-coupling-earn-its-keep comparison.


def synth_camera_bearings(
    seq: "Traj6DoF",
    R_bc: np.ndarray = None,
    bearing_noise_std: float = 0.0,
    seed: int = 0,
):
    """Per-pos_idx CAMERA-frame UNIT bearings of seq.landmarks_world.

    Returns (bearings, visible): bearings[m, l] = pi(R_bc^T R_gt^T (p_L - p_gt)) at fix m
    (plus optional small tangential noise), and visible[m, l] a bool gate (corner in front
    of the camera: positive optical-axis component). Anchored on the GT poses so a noiseless
    run is an exact parallax/consistency probe."""
    R_bc = np.eye(3) if R_bc is None else R_bc
    rng = np.random.default_rng(seed)
    M = len(seq.pos_idx)
    L = seq.landmarks_world.shape[0]
    bearings = np.zeros((M, L, 3))
    visible = np.zeros((M, L), dtype=bool)
    for m, idx in enumerate(seq.pos_idx):
        Rg = seq.R_gt[idx]; pg = seq.p_gt[idx]
        for l in range(L):
            d_cam = R_bc.T @ (Rg.T @ (seq.landmarks_world[l] - pg))
            n = np.linalg.norm(d_cam)
            if n < 1e-9:
                continue
            b = d_cam / n
            if bearing_noise_std > 0:
                b = b + rng.normal(0.0, bearing_noise_std, 3)
                b = b / max(np.linalg.norm(b), 1e-12)
            bearings[m, l] = b
            visible[m, l] = d_cam[2] > 0.0          # in front of optical axis (+z cam)
    return bearings, visible


def run_eqvio_joint(
    filt,
    seq: "Traj6DoF",
    R0=None, v0=None, p0=None, P0=None,
    R_bc: np.ndarray = None,
    bearing_noise_std: float = 0.0,
    rho_init: float = 0.05,
    rho_var: float = 1.0,
    marginalize_after: int = None,
    seed: int = 0,
):
    """Run the joint EqVIO filter over the trajectory.

    At each pos_idx fix: propagate (IMU), add any newly-visible corner (anchored at the
    current pose with large inverse-depth variance), then apply a joint bearing update for
    every tracked-and-visible corner. Optionally marginalise a corner after a fixed number
    of fixes (exercises drop_landmark). Returns per-fix pose/joint NEES, the landmark
    inverse-depth recovery error vs GT, and the final landmark world-point error.
    """
    R_bc = np.eye(3) if R_bc is None else R_bc
    R0 = seq.R_gt[0] if R0 is None else R0
    v0 = seq.v_gt[0] if v0 is None else v0
    p0 = seq.p_gt[0] if p0 is None else p0
    filt.reset(R=R0, v=v0, p=p0, P_pose=P0)
    if R_bc is not None:
        filt.R_bc = R_bc

    bearings, visible = synth_camera_bearings(seq, R_bc, bearing_noise_std, seed)
    L = seq.landmarks_world.shape[0]
    pos_lookup = {int(idx): k for k, idx in enumerate(seq.pos_idx)}
    pos_set = set(int(i) for i in seq.pos_idx)
    dt = seq.dt

    nees_pose, nees_t = [], []
    invdepth_err_hist = []
    fix_count = 0

    for i in range(seq.N):
        if i > 0:
            filt.predict(seq.gyro[i - 1], seq.accel[i - 1], dt)
        if i in pos_set:
            m = pos_lookup[i]
            # add newly-visible corners
            for l in range(L):
                if visible[m, l] and not filt.has_landmark(l):
                    filt.add_landmark(l, bearings[m, l], rho_init=rho_init, rho_var=rho_var)
            # joint bearing update for every tracked+visible corner
            for l in range(L):
                if visible[m, l] and filt.has_landmark(l):
                    filt.update_landmark_joint(l, bearings[m, l])
            fix_count += 1

            if marginalize_after is not None and fix_count == marginalize_after:
                ids = filt.landmark_ids()
                if ids:
                    filt.drop_landmark(ids[0])

            # pose NEES (post-update), right-invariant tangent
            if i > 0:
                e_pose = _error_vector_pose(filt, seq, i)
                Ppose = filt.pose_cov()
                try:
                    nees_pose.append(float(e_pose @ np.linalg.solve(Ppose, e_pose)))
                    nees_t.append(i * dt)
                except np.linalg.LinAlgError:
                    pass

            # landmark inverse-depth recovery error (per tracked corner, vs GT depth in the
            # landmark's OWN anchor frame).
            for lm in filt._lmks:
                p_L_gt = seq.landmarks_world[lm.lm_id]
                d_anchor = R_bc.T @ (lm.R_anchor.T @ (p_L_gt - lm.p_anchor))
                rho_gt = 1.0 / max(np.linalg.norm(d_anchor), 1e-9)
                invdepth_err_hist.append(abs(lm.sot.rho - rho_gt))

    # final landmark world-point errors
    final_lmk_err = {}
    for lm in filt._lmks:
        p_est = filt.landmark_world(lm.lm_id)
        final_lmk_err[lm.lm_id] = float(np.linalg.norm(p_est - seq.landmarks_world[lm.lm_id]))

    return {
        "nees_pose": np.array(nees_pose),
        "nees_t": np.array(nees_t),
        "invdepth_err": np.array(invdepth_err_hist),
        "final_lmk_err": final_lmk_err,
        "n_landmarks": filt.n_landmarks,
    }


def _error_vector_pose(filt, seq: "Traj6DoF", i: int) -> np.ndarray:
    """Right-invariant 9-dof pose error of the joint filter vs GT (same tangent its pose
    covariance lives in)."""
    return _log_se23_error(seq.R_gt[i], seq.v_gt[i], seq.p_gt[i], filt.R, filt.v, filt.p)
