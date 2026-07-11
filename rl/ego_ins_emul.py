"""rl/ego_ins_emul.py -- batched-torch translation of the DEPLOY vq2_ego_lean estimation chain, for
the ESTIMATOR-FAITHFUL ACTOR OBS package (2026-07-11).

THE OWNER DIRECTIVE: the actor NEVER sees ground truth -- "however we estimate which way down is in
the real simulator, we train with that." Training obs[3:5] (roll/pitch) currently comes from a
PERFECT extraction of the truth quaternion; the wire's ONLY attitude source is an accel+gyro ESKF
leveler that was MEASURED (flight a5, 2026-07-11) to diverge from gyro-consistent truth by median
~5-7 deg / p90 ~20-27 deg PER 50-60 ms TICK under the flight's 2.98 g median specific force -- and
the policy trained on perfect attitude crashed on those lies. This module supplies the training-side
emulation of the deploy algorithms, vectorized over N envs:

  * BatchedESKFLeveler -- a line-for-line translation of
    Anduril-ego-deploy/src/racer/ahrs/eskf.py (@ ego-deploy e1aa4d1): gyro quaternion propagation,
    the FULL accel-update gate stack IN ORDER (|a|<1e-6 skip -> free-fall/high-|a| magnitude band ->
    smooth high-g weight gate (skip < 1e-4) -> A8 accel-motion-reject R-inflation with its STATEFUL
    R_ref re-anchor -> chi2(3) innovation gate -> Joseph update, right-multiplicative injection).
  * BatchedNavKF -- a translation of src/racer/state_estimator.py::LinearKF (6-state [pos, vel]):
    strapdown predict through the (emulated, lying) leveler attitude, attitude-error-projected Q,
    dt>max_dt guard, Joseph position updates + chi2(3) fix gate + the in-plane eigenvalue floor.
    Under vq2_ego_lean there is NO velocity measurement anywhere -- velocity is corrected ONLY via
    the pos/vel cross-covariance of position fixes, then projected world->body through the SAME
    corrupted attitude (deploy ego_obs.py:305-321). The training emulation reproduces both error
    injections.

FRAME-AGNOSTIC CORES: each filter takes a gravity VECTOR ``g_world`` -- the math is frame-covariant,
so the deploy NED/FRD equations carry over unchanged:
  * deploy / wire-replay mode: NED, g_world = [0, 0, +9.80665]; at-rest specific force [0,0,-g].
  * training mode: Z-up/FLU,   g_world = [0, 0, -9.80665]; at-rest specific force [0,0,+g].

RATE CONTRACT (the headline design decision, empirically pinned): the deploy ESKF/KF advance ONCE
PER CONTROL TICK on the LATEST IMU sample (navigator.py:585-607 "Estimation advances only on a NEW
IMU sample"; single-latest-rate propagation reproduces the recorded wire leveler to median
0.0003 deg -- rl/tools/imu_foundation.py tickrate_step_check). So the training emulation steps ONCE
per 30 Hz env tick with dt = env.dt, consuming the LAST plant-substep sample. A 143 Hz substep loop
would be LESS faithful AND ~5x the cost. The wire's fatal error is SAMPLING ALIASING of tick-rate
gyro dead-reckoning against continuous dynamics -- restored in-training by dynamics.n_substeps=5
(150 Hz ~ the measured 143.3 Hz wire IMU) in the NEW _pef stages only; at n_substeps=1 the plant
truth is itself piecewise-constant-rate and a tick-rate leveler would track truth exactly (the
fatal channel could not exist in-sim; pinned by tests/test_estimator_faithful.py's ZOH negative
control).

MEASURED SENSOR CONSTANTS (invariant 3 -- measured, not invented; extraction script
rl/tools/imu_foundation.py, dataset a5 pad-idle 2026-07-10, run 2026-07-11):
  * the sim IMU is NOISELESS: 11,856 samples / 101.16 s / 143.33 Hz contain EXACTLY 1 unique gyro
    row and 1 unique accel row -> gyro bias [0,0,0] rad/s EXACT, gyro/accel white-noise density
    [0,0,0]. THEREFORE the emulation adds NO sensor noise -- the measured constant is zero. (a7
    corroborates after timestamp-dedup: <=6.6e-4 rad/s bounded by arm micro-motion.)
  * accel magnitude offset +0.003351 m/s^2 (direction-inseparable from the modeled tilted pad,
    roll -0.014 / pitch -17.802 deg == obs[4] -0.31071 rad) -- NOT modeled (3.4e-4 g, far below the
    filter's accel_noise_std).
  * If a future sim build adds IMU noise, re-run rl/tools/imu_foundation.py and populate the
    (currently zero) noise fields in EgoEstimatorConfig.
FILTER GAINS are INHERITED verbatim from the deploy code (filter TUNING, not physics -- explicitly
NOT re-fit to the measured zero sensor noise); every constant below cites its deploy source line.

NO_GRAD: the training loss never backprops through the actor obs (peregrine_racing_ego module doc
"PPO-ONLY / NO_GRAD"; loss = (-reward).detach(); algo=appo is PPO-family). All step methods run
under @torch.no_grad().

PERF: per 33 ms tick at N=4096 float32 CUDA, the ESKF is ~30 small batched ops incl. two (N,6,6)
matmuls + one batched 3x3 solve + Joseph; the KF similar -- well under 1 ms/step, <5% of env step
time. The real cost of the package is dynamics.n_substeps 1->5 (the plant substep loop x5), which
rides ONLY the new _pef stages.
"""
from __future__ import annotations

import math

import numpy as np

try:
    import torch
    from torch import Tensor
except Exception:                       # pragma: no cover - torch absent in some tooling contexts
    torch = None
    Tensor = "Tensor"                   # type: ignore


GRAVITY = 9.80665                       # deploy eskf.py:96

# ================================================================================================
# DEPLOY-INHERITED FILTER CONSTANTS (verbatim; file:line provenance @ ego-deploy e1aa4d1).
# These are filter TUNING constants, never re-fit -- see the module docstring.
# ================================================================================================
ESKF_GYRO_NOISE_STD = 0.01              # rad/s          eskf.py:188 (passed by navigator.py:543-546)
ESKF_GYRO_BIAS_STD = 1e-4               # rad/s/sqrt(s)  eskf.py:189 (default, never overridden)
ESKF_ACCEL_NOISE_STD = 0.3              # m/s^2          eskf.py:190 DEFAULT -- the Navigator never
#   overrides accel_noise_std; the eskf.py:255-257 comment arithmetic assuming 0.05 is STALE.
ESKF_ACCEL_GATE_ALPHA = 10.0            # eskf.py:191 (navigator passes 10.0)
ESKF_ACCEL_CHI2_THRESH = 7.815          # chi2(3,.95)    eskf.py:192
ESKF_FREEFALL_TOL_LO = 0.75             # reject |a| < 0.25 g   eskf.py:205
ESKF_FREEFALL_TOL_HI = 9.0              # reject |a| > 10 g     eskf.py:206
ESKF_MOTION_SCALE = 0.1                 # m/s^2          eskf.py:242 (A8 motion-reject; ON in
ESKF_MOTION_ANCHOR_THR = 0.3            # m/s^2          eskf.py:252  vq2_ego_lean via
ESKF_MOTION_MAX_INFLATE = 1e4           #                eskf.py:258  deploy_profile.py:133-134)
ESKF_P0_DIAG = (1e-2, 1e-2, 1e-2, 1e-6, 1e-6, 1e-6)      # eskf.py:274 (cold-boot P0)

# PAD-CONVERGED LAUNCH COVARIANCE (measured, rl/tools/eskf_pad_state.py on the a5 pad-idle window,
# run 2026-07-11): deploy boots the ESKF ~100 s before takeoff and converges on the pad-idle accel
# stream, so at HANDOVER (where a training episode effectively starts) the covariance is NOT the
# cold P0. The converged P is EXACTLY axisymmetric about the gravity direction in body frame
# (eigenvectors align with g_hat to 1e-16): the two tilt-observable axes sit at the accel-update
# fixed point while the accel-UNOBSERVABLE axis (about gravity == the yaw datum) carries the full
# gyro random-walk growth over the 101.16 s window. Values below are the 57 ms-tick replay (the a5
# flight-median control tick); the fixed point is mildly tick-rate dependent (33 ms: 5.45e-5 /
# 100 ms: 9.28e-5 on phi-perp -- same order, band documented). phi<->bias cross-covariance
# (max |c| 1.4e-4, only on the unobservable axis) is DROPPED at seeding -- a documented
# approximation; the flight regrows it within seconds. Build the per-env P via eskf_p_launch().
ESKF_P_LAUNCH_PHI_PERP = 7.0775286913e-05    # rad^2   (~0.48 deg 1-sigma tilt at takeoff)
ESKF_P_LAUNCH_PHI_ALONG = 3.3780124305e-02   # rad^2   (yaw-datum axis, random-walked)
ESKF_P_LAUNCH_BG_PERP = 1.0082759782e-06     # (rad/s)^2
ESKF_P_LAUNCH_BG_ALONG = 2.0111800000e-06    # (rad/s)^2  (= P0 1e-6 + gyro_bias_std^2 * 101 s)

KF_ACCEL_NOISE_STD = 0.3                # m/s^2          state_estimator.py:69
KF_ATTITUDE_NOISE_STD = math.radians(1.4)  # rad         frames.py ATTITUDE_NOISE_STD_RAD
#   (measured vision-pkg2 2026-06-10; consumed at state_estimator.py:73)
KF_PROCESS_FLOOR = 1e-6                 #                state_estimator.py:75
KF_MAX_DT_S = 0.2                       # s              state_estimator.py:76 (predict DROPPED past
#   this; inert at the training 33 ms tick but kept for parity -- stated, not silently dropped)
KF_INPLANE_FLOOR_STD = 0.05             # m              localization.py INPLANE_POS_FLOOR_STD
#   (active in case-C: navigator._initialize sets inplane_pos_floor_std=0.05)
KF_INIT_POS_STD = 5.0                   # m              navigator.py:498 (case-C map-free seed)
KF_INIT_VEL_STD = 1.0                   # m/s            navigator.py:498
KF_FIX_COV_FLOOR_STD = 0.40             # m              localization.py FIX_COV_FLOOR_STD
KF_FIX_CHI2_THRESH = 16.27              # chi2(3,.999)   the deploy 3-DOF Mahalanobis fix gate
#   (navigator _process_observation chain) -- rejects teleport-class outlier fixes.

# MEASURED-ZERO sensor constants (imu_foundation.py, a5 pad-idle 2026-07-10 -- see module doc).
# Kept as named fields so a future noisy sim build has an obvious slot to populate.
MEASURED_GYRO_BIAS_RAD_S = (0.0, 0.0, 0.0)
MEASURED_GYRO_NOISE_DENSITY = (0.0, 0.0, 0.0)            # rad/s/sqrt(Hz)
MEASURED_ACCEL_NOISE_DENSITY = (0.0, 0.0, 0.0)           # m/s^2/sqrt(Hz)


# ================================================================================================
# Batched quaternion / rotation helpers (wxyz, body->world). Local to keep this module dependency-
# free (usable from the deploy-parity replay harness without the rl env stack).
# ================================================================================================
def quat_mul_wxyz(q1: Tensor, q2: Tensor) -> Tensor:
    """Hamilton product (...,4) x (...,4) -> (...,4), (w,x,y,z) layout (eskf.py:100-109)."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


def quat_to_R_wxyz(q: Tensor) -> Tensor:
    """(...,4) wxyz -> (...,3,3) body->world rotation (eskf.py:112-119)."""
    w, x, y, z = q.unbind(-1)
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - z * w)
    r02 = 2 * (x * z + y * w)
    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - x * w)
    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    r22 = 1 - 2 * (x * x + y * y)
    return torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1),
    ], dim=-2)


def rotvec_to_quat_wxyz(rv: Tensor) -> Tensor:
    """Closed-form quaternion exponential exp(rv/2) (...,3) -> (...,4) (eskf.py:130-140),
    small-angle safe (angle < 1e-12 -> identity, matching the deploy branch)."""
    angle = torch.linalg.norm(rv, dim=-1, keepdim=True)                  # (...,1)
    small = angle < 1e-12
    safe = torch.where(small, torch.ones_like(angle), angle)
    axis = rv / safe
    half = angle * 0.5
    q = torch.cat([torch.cos(half), axis * torch.sin(half)], dim=-1)
    ident = torch.zeros_like(q)
    ident[..., 0] = 1.0
    return torch.where(small, ident, q)


def normalize_quat(q: Tensor) -> Tensor:
    """(...,4) -> unit quaternion; degenerate (|q|<1e-12) -> identity (eskf.py:143-145)."""
    n = torch.linalg.norm(q, dim=-1, keepdim=True)
    small = n < 1e-12
    safe = torch.where(small, torch.ones_like(n), n)
    out = q / safe
    ident = torch.zeros_like(q)
    ident[..., 0] = 1.0
    return torch.where(small, ident, out)


def skew(v: Tensor) -> Tensor:
    """(...,3) -> (...,3,3) skew-symmetric (eskf.py:122-127)."""
    x, y, z = v.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack([
        torch.stack([zero, -z, y], dim=-1),
        torch.stack([z, zero, -x], dim=-1),
        torch.stack([-y, x, zero], dim=-1),
    ], dim=-2)


def quat_from_roll_pitch_yaw_zyx(roll: Tensor, pitch: Tensor, yaw: Tensor) -> Tensor:
    """ZYX (yaw-pitch-roll) Euler -> wxyz quaternion, batched. Used to seed the leveler at the
    truth roll/pitch with a ZERO yaw datum (the deploy-faithful takeoff state: the wire leveler is
    ~truth at takeoff after the ~100 s pad-idle convergence, with an arbitrary-0 yaw datum)."""
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    return torch.stack([
        cy * cp * cr + sy * sp * sr,
        cy * cp * sr - sy * sp * cr,
        cy * sp * cr + sy * cp * sr,
        sy * cp * cr - cy * sp * sr,
    ], dim=-1)


def tilt_vector_from_roll_pitch(roll: Tensor, pitch: Tensor) -> Tensor:
    """Unit 'world-vertical in body' vector u = R^T e_z = [-sin p, cos p sin r, cos p cos r]
    (yaw-free; the acceptance benchmark's comparison space -- imu_foundation.g_hat_from_roll_pitch)."""
    return torch.stack([
        -torch.sin(pitch),
        torch.sin(roll) * torch.cos(pitch),
        torch.cos(roll) * torch.cos(pitch),
    ], dim=-1)


def tilt_angle_rad(roll_pitch_a: Tensor, roll_pitch_b: Tensor) -> Tensor:
    """Yaw-invariant tilt-vector angle (rad) between two (N,2) [roll, pitch] pairs -- the
    acceptance benchmark's divergence metric (diagnostics: eskf_tilt_err_deg_*)."""
    ua = tilt_vector_from_roll_pitch(roll_pitch_a[..., 0], roll_pitch_a[..., 1])
    ub = tilt_vector_from_roll_pitch(roll_pitch_b[..., 0], roll_pitch_b[..., 1])
    dot = (ua * ub).sum(-1).clamp(-1.0, 1.0)
    return torch.arccos(dot)


def eskf_p_launch(g_hat_body: Tensor) -> Tensor:
    """Build the per-env PAD-CONVERGED launch covariance (m,6,6) from the gravity direction in the
    seeded body frame (m,3), using the measured gravity-principal structure (see the
    ESKF_P_LAUNCH_* provenance block): P_block = perp*(I - g g^T) + along*(g g^T) on both the
    delta_phi and delta_b_g blocks; phi<->bias cross terms dropped (documented approximation)."""
    m = g_hat_body.shape[0]
    dev, dt = g_hat_body.device, g_hat_body.dtype
    eye3 = torch.eye(3, device=dev, dtype=dt).expand(m, 3, 3)
    gg = g_hat_body.unsqueeze(-1) @ g_hat_body.unsqueeze(-2)             # (m,3,3)
    P = torch.zeros(m, 6, 6, device=dev, dtype=dt)
    P[:, :3, :3] = ESKF_P_LAUNCH_PHI_PERP * (eye3 - gg) + ESKF_P_LAUNCH_PHI_ALONG * gg
    P[:, 3:, 3:] = ESKF_P_LAUNCH_BG_PERP * (eye3 - gg) + ESKF_P_LAUNCH_BG_ALONG * gg
    return P


# ================================================================================================
# BatchedESKFLeveler -- the deploy eskf.py, vectorized over N envs.
# ================================================================================================
class BatchedESKFLeveler:
    """Batched translation of deploy ESKFAHRS (attitude-only; mag/vision-yaw paths are DEAD under
    vq2_ego_lean and not translated). State per env: nominal quaternion q (wxyz, body->world),
    gyro bias b_g, 6x6 error covariance P [delta_phi, delta_b_g], and the A8 gyro-anchored
    reference rotation R_ref. All methods no_grad; float32 default (float64 for parity tests)."""

    def __init__(self, n: int, g_world, device=None, dtype=None, *,
                 gyro_noise_std: float = ESKF_GYRO_NOISE_STD,
                 gyro_bias_std: float = ESKF_GYRO_BIAS_STD,
                 accel_noise_std: float = ESKF_ACCEL_NOISE_STD,
                 accel_gate_alpha: float = ESKF_ACCEL_GATE_ALPHA,
                 accel_chi2_thresh: float = ESKF_ACCEL_CHI2_THRESH,
                 freefall_tol_lo: float = ESKF_FREEFALL_TOL_LO,
                 freefall_tol_hi: float = ESKF_FREEFALL_TOL_HI,
                 use_accel_motion_reject: bool = True,       # vq2_ego_lean: ahrs_accel_motion_reject=True
                 motion_scale: float = ESKF_MOTION_SCALE,
                 motion_anchor_thr: float = ESKF_MOTION_ANCHOR_THR,
                 motion_max_inflate: float = ESKF_MOTION_MAX_INFLATE):
        assert torch is not None
        self.n = int(n)
        dtype = dtype or torch.float32
        self.device, self.dtype = device, dtype
        self.g_world = torch.as_tensor(g_world, device=device, dtype=dtype)      # gravity VECTOR
        self.g_mag = float(torch.linalg.norm(self.g_world))
        self.gyro_noise_std = float(gyro_noise_std)
        self.gyro_bias_std = float(gyro_bias_std)
        self.accel_noise_std = float(accel_noise_std)
        self.accel_gate_alpha = float(accel_gate_alpha)
        self.accel_chi2_thresh = float(accel_chi2_thresh)
        self.freefall_tol_lo = float(freefall_tol_lo)
        self.freefall_tol_hi = float(freefall_tol_hi)
        self.use_accel_motion_reject = bool(use_accel_motion_reject)
        self.motion_scale = float(motion_scale)
        self.motion_anchor_thr = float(motion_anchor_thr)
        self.motion_max_inflate = float(motion_max_inflate)

        N = self.n
        self.q = torch.zeros(N, 4, device=device, dtype=dtype)
        self.q[:, 0] = 1.0
        self.b_g = torch.zeros(N, 3, device=device, dtype=dtype)
        self.P = torch.diag(torch.tensor(ESKF_P0_DIAG, device=device, dtype=dtype)) \
            .unsqueeze(0).repeat(N, 1, 1)
        self.R_ref = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(N, 1, 1)
        # diagnostics: which envs applied an accel update on the last step (eskf_accel_update_duty)
        self.last_update_mask = torch.zeros(N, dtype=torch.bool, device=device)

    # -------------------------------------------------------------------- lifecycle
    @torch.no_grad()
    def reset_idx(self, idx: Tensor, q0: Tensor, P0=None, b0: Tensor | None = None) -> None:
        """Cold-init the given envs. q0 (m,4) wxyz seed (truth roll/pitch + yaw datum 0 in training --
        the deploy-faithful takeoff state: the wire level-seeds on the pad and converges over ~100 s
        idle, so at handover the leveler is ~truth). P0:
          * None (default) -> the PAD-CONVERGED launch covariance built per env from the seeded
            attitude's gravity direction (eskf_p_launch; measured by rl/tools/eskf_pad_state.py) --
            NOT the cold-boot P0 -- so the early-flight chi2/S behaviour matches a real takeoff;
          * a 6-tuple -> diag (e.g. the deploy cold-boot ESKF_P0_DIAG, for pad replays);
          * a (m,6,6) tensor -> used verbatim.
        b0 defaults to 0 (the pad replay converges bias to ~0 exactly under the measured-noiseless
        gyro)."""
        idx = idx.reshape(-1)
        if int(idx.numel()) == 0:
            return
        q0 = normalize_quat(q0.to(dtype=self.dtype))
        self.q[idx] = q0
        self.b_g[idx] = 0.0 if b0 is None else b0.to(dtype=self.dtype)
        if P0 is None:
            R0 = quat_to_R_wxyz(q0)                                          # (m,3,3)
            g_hat = torch.einsum("mji,j->mi", R0, self.g_world) / self.g_mag
            self.P[idx] = eskf_p_launch(g_hat)
        elif torch.is_tensor(P0):
            self.P[idx] = P0.to(dtype=self.dtype)
        else:
            self.P[idx] = torch.diag(torch.tensor(P0, device=self.device, dtype=self.dtype))
        self.R_ref[idx] = quat_to_R_wxyz(self.q[idx])
        self.last_update_mask[idx] = False

    # -------------------------------------------------------------------- step
    @torch.no_grad()
    def step(self, gyro: Tensor, accel: Tensor, dt) -> Tensor:
        """One estimator advance for ALL envs (== one deploy nav tick): _predict then _update_accel.
        gyro (N,3) rad/s, accel (N,3) specific force m/s^2 (the LATEST sample, zero-order-held over
        dt -- the deploy rate contract), dt scalar or (N,) seconds. Returns q (N,4). A dt<=0 env is
        a no-op (deploy eskf.step:337-338)."""
        dt_t = torch.as_tensor(dt, device=self.q.device, dtype=self.dtype).expand(self.n).clone() \
            if not torch.is_tensor(dt) else dt.to(dtype=self.dtype).expand(self.n).clone()
        live = dt_t > 0
        dt_t = torch.where(live, dt_t, torch.zeros_like(dt_t))              # dt<=0 -> zero-motion no-op
        self._predict(gyro, dt_t, live)
        self._update_accel(accel, live)
        return self.q

    def _predict(self, gyro: Tensor, dt: Tensor, live: Tensor) -> None:
        """eskf.py:350-377 verbatim, batched. NO dt cap (the ESKF has none -- unlike the KF)."""
        omega_corr = gyro - self.b_g                                        # (N,3)
        dq = rotvec_to_quat_wxyz(omega_corr * dt.unsqueeze(-1))             # exp(w dt / 2)
        q_new = normalize_quat(quat_mul_wxyz(self.q, dq))
        self.q = torch.where(live.unsqueeze(-1), q_new, self.q)
        if self.use_accel_motion_reject:
            R_dq = quat_to_R_wxyz(dq)
            R_ref_new = self.R_ref @ R_dq
            self.R_ref = torch.where(live[:, None, None], R_ref_new, self.R_ref)
        # F = [[I - skew(w)dt, -I dt], [0, I]]
        eye3 = torch.eye(3, device=self.q.device, dtype=self.dtype).expand(self.n, 3, 3)
        F = torch.zeros(self.n, 6, 6, device=self.q.device, dtype=self.dtype)
        F[:, :3, :3] = eye3 - skew(omega_corr) * dt[:, None, None]
        F[:, :3, 3:] = -eye3 * dt[:, None, None]
        F[:, 3:, 3:] = eye3
        sig_phi2 = (self.gyro_noise_std ** 2) * dt                          # (gyro_std*sqrt(dt))^2
        sig_bg2 = (self.gyro_bias_std ** 2) * dt
        Q = torch.zeros_like(F)
        Q[:, 0, 0] = Q[:, 1, 1] = Q[:, 2, 2] = sig_phi2
        Q[:, 3, 3] = Q[:, 4, 4] = Q[:, 5, 5] = sig_bg2
        P_new = F @ self.P @ F.transpose(-1, -2) + Q
        self.P = torch.where(live[:, None, None], P_new, self.P)

    def _update_accel(self, accel: Tensor, live: Tensor) -> None:
        """eskf.py:440-518 verbatim, batched with per-env gating masks (a gated-out env keeps its
        predicted state). GATE ORDER is load-bearing and preserved:
          (a) |a| < 1e-6 skip; (b) magnitude band skip; (c) smooth gate < 1e-4 skip;
          (d) A8 motion inflation (STATEFUL R_ref re-anchor -- computed exactly for the envs that
              reached the R_meas construction, i.e. passed (a)-(c), BEFORE the chi2 gate);
          (e) chi2(3) innovation gate; (f) Joseph update + right-multiplicative injection."""
        g = self.g_mag
        amag = torch.linalg.norm(accel, dim=-1)                             # (N,)
        m_valid = live & (amag >= 1e-6)                                     # (a)
        lo = g * (1.0 - self.freefall_tol_lo)
        hi = g * (1.0 + self.freefall_tol_hi)
        m_band = (amag >= lo) & (amag <= hi)                                # (b) eskf.py:391-410
        dev = (amag - g) / g
        gate = torch.exp(-self.accel_gate_alpha * dev * dev)                # eskf.py:381-389
        m_gate = gate >= 1e-4                                               # (c) eskf.py:466
        pre = m_valid & m_band & m_gate                                     # envs reaching R_meas

        # normalized measurement space (eskf.py:469-485)
        R_wb = quat_to_R_wxyz(self.q)                                       # (N,3,3)
        g_body = torch.einsum("nji,j->ni", R_wb, self.g_world)              # R^T @ G
        g_hat = g_body / torch.linalg.norm(g_body, dim=-1, keepdim=True).clamp(min=1e-9)
        h_hat = -g_hat
        a_hat = accel / amag.unsqueeze(-1).clamp(min=1e-9)
        innov = a_hat - h_hat                                               # (N,3)

        # (d) A8 motion inflation + STATEFUL re-anchor (eskf.py:412-438), only for `pre` envs
        if self.use_accel_motion_reject:
            a_lin = torch.einsum("nij,nj->ni", self.R_ref, accel) + self.g_world
            a_lin_mag = torch.linalg.norm(a_lin, dim=-1)
            reanchor = pre & (a_lin_mag < self.motion_anchor_thr)
            self.R_ref = torch.where(reanchor[:, None, None], R_wb, self.R_ref)
            factor = (1.0 + (a_lin_mag / max(self.motion_scale, 1e-9)) ** 2) \
                .clamp(max=self.motion_max_inflate)
        else:
            factor = torch.ones_like(amag)

        # R_meas = (sigma_a^2 / gate) * factor * I3 (eskf.py:501-502); safe division on masked envs
        sigma_a = self.accel_noise_std / GRAVITY
        gate_safe = torch.where(pre, gate, torch.ones_like(gate))
        r_scalar = (sigma_a ** 2) / gate_safe * factor                      # (N,)
        eye3 = torch.eye(3, device=self.q.device, dtype=self.dtype)
        R_meas = r_scalar[:, None, None] * eye3

        # H (N,3,6) = [-skew(g_hat), 0] (eskf.py:493-494)
        H = torch.zeros(self.n, 3, 6, device=self.q.device, dtype=self.dtype)
        H[:, :, :3] = -skew(g_hat)

        # (e) chi2(3) innovation gate (eskf.py:508-515)
        PHt = self.P @ H.transpose(-1, -2)                                  # (N,6,3)
        S = H @ PHt + R_meas                                                # (N,3,3), PD (R>=sigma_a^2)
        sol = torch.linalg.solve(S, innov.unsqueeze(-1)).squeeze(-1)        # (N,3)
        md = (innov * sol).sum(-1)
        if self.accel_chi2_thresh > 0.0:
            m_chi2 = md <= self.accel_chi2_thresh
        else:
            m_chi2 = torch.ones_like(pre)
        upd = pre & m_chi2                                                  # the envs that update

        # (f) _apply_eskf_update (eskf.py:606-638), batched; applied via where(upd)
        K = torch.linalg.solve(S, PHt.transpose(-1, -2)).transpose(-1, -2)  # (N,6,3) = P H^T S^-1
        delta_x = torch.einsum("nij,nj->ni", K, innov)                      # (N,6)
        dq = rotvec_to_quat_wxyz(delta_x[:, :3])                            # exp(delta_phi/2)
        q_new = normalize_quat(quat_mul_wxyz(self.q, dq))                   # RIGHT-multiplicative
        b_new = self.b_g + delta_x[:, 3:]
        I_KH = torch.eye(6, device=self.q.device, dtype=self.dtype).expand(self.n, 6, 6) - K @ H
        P_new = I_KH @ self.P @ I_KH.transpose(-1, -2) \
            + r_scalar[:, None, None] * (K @ K.transpose(-1, -2))           # K R K^T, R = r*I3
        self.q = torch.where(upd[:, None], q_new, self.q)
        self.b_g = torch.where(upd[:, None], b_new, self.b_g)
        self.P = torch.where(upd[:, None, None], P_new, self.P)
        self.last_update_mask = upd

    # -------------------------------------------------------------------- readout
    def R_wb(self) -> Tensor:
        """(N,3,3) body->world rotation of the current (emulated, possibly lying) attitude."""
        return quat_to_R_wxyz(self.q)


# ================================================================================================
# BatchedNavKF -- deploy LinearKF (state_estimator.py), vectorized. 6-state [pos(3), vel(3)] in the
# per-env yaw-DATUM world frame (the ESKF's own frame -- exactly deploy, where the whole nav world
# is the AHRS-datum frame).
# ================================================================================================
class BatchedNavKF:
    """Velocity provenance under vq2_ego_lean (verified): there is NO velocity measurement anywhere
    -- velocity is pure strapdown integration of (R_emul @ sf_body + g_world), corrected ONLY
    indirectly through the pos/vel cross-covariance of position fixes. Attitude error therefore
    corrupts velocity TWICE: wrong gravity cancellation at integration + wrong world->body
    projection at the obs -- both reproduced by construction (the caller passes the EMULATED
    leveler attitude into predict() and projects the readout through the same attitude).

    RewindKF is NOT translated: replaying a fix at capture time is equivalent to applying it on
    time (kf_rewind.py:96 degenerate case) and training fixes are same-tick; the only extra deploy
    behaviour (fix older than the 0.5 s horizon dropped) cannot occur. The deploy 3-channel fix
    chain (absolute + gate-relative in-plane + range) is SURROGATED by one 3D position update with
    the training-measured per-fix sigma model (which IS the measured end-to-end fix-error
    contract), floored at KF_FIX_COV_FLOOR_STD -- documented parity cut; the full-chain translation
    is the priced upgrade if the velocity benchmark misses."""

    def __init__(self, n: int, g_world, device=None, dtype=None, *,
                 accel_noise_std: float = KF_ACCEL_NOISE_STD,
                 attitude_noise_std: float = KF_ATTITUDE_NOISE_STD,
                 process_floor: float = KF_PROCESS_FLOOR,
                 max_dt_s: float = KF_MAX_DT_S,
                 inplane_floor_std: float = KF_INPLANE_FLOOR_STD,
                 fix_chi2_thresh: float = KF_FIX_CHI2_THRESH):
        assert torch is not None
        self.n = int(n)
        dtype = dtype or torch.float32
        self.device, self.dtype = device, dtype
        self.g_world = torch.as_tensor(g_world, device=device, dtype=dtype)
        self.accel_noise_std = float(accel_noise_std)
        self.attitude_noise_std = float(attitude_noise_std)
        self.process_floor = float(process_floor)
        self.max_dt_s = float(max_dt_s)
        self.inplane_floor_std = float(inplane_floor_std)
        self.fix_chi2_thresh = float(fix_chi2_thresh)
        self.x = torch.zeros(self.n, 6, device=device, dtype=dtype)
        self.P = torch.diag(torch.tensor(
            [KF_INIT_POS_STD ** 2] * 3 + [KF_INIT_VEL_STD ** 2] * 3,
            device=device, dtype=dtype)).unsqueeze(0).repeat(self.n, 1, 1)

    @torch.no_grad()
    def reset_idx(self, idx: Tensor, pos0: Tensor, vel0: Tensor,
                  pos_std: float = KF_INIT_POS_STD, vel_std: float = KF_INIT_VEL_STD) -> None:
        """navigator._initialize seed (pos_std 5.0 / vel_std 1.0). Deploy seeds pos=origin, vel=0 --
        exact truth on the parked pad; training passes the datum-frame truth (0 / spawn velocity),
        the same 'exact at takeoff' state."""
        idx = idx.reshape(-1)
        if int(idx.numel()) == 0:
            return
        self.x[idx, :3] = pos0.to(dtype=self.dtype)
        self.x[idx, 3:] = vel0.to(dtype=self.dtype)
        self.P[idx] = torch.diag(torch.tensor([pos_std ** 2] * 3 + [vel_std ** 2] * 3,
                                              device=self.device, dtype=self.dtype))

    @torch.no_grad()
    def predict(self, sf_body: Tensor, R_wb: Tensor, dt) -> None:
        """state_estimator.py:110-135 verbatim, batched. R_wb MUST be the EMULATED (lying) leveler
        attitude -- the load-bearing error coupling. dt<=0 or dt>max_dt_s -> predict DROPPED
        silently (deploy guard; inert at the fixed 33 ms training tick -- stated for parity)."""
        dt_t = torch.as_tensor(dt, device=self.x.device, dtype=self.dtype).expand(self.n) \
            if not torch.is_tensor(dt) else dt.to(dtype=self.dtype).expand(self.n)
        live = (dt_t > 0) & (dt_t <= self.max_dt_s)
        sf_world = torch.einsum("nij,nj->ni", R_wb, sf_body)                # (N,3)
        a_world = sf_world + self.g_world
        dt1 = dt_t.unsqueeze(-1)
        pos_new = self.x[:, :3] + dt1 * self.x[:, 3:] + 0.5 * dt1 * dt1 * a_world
        vel_new = self.x[:, 3:] + dt1 * a_world
        x_new = torch.cat([pos_new, vel_new], dim=-1)
        self.x = torch.where(live.unsqueeze(-1), x_new, self.x)
        # Q = B accel_cov B^T + floor*I, accel_cov = an^2 I + att^2 [s]x [s]x^T
        S = skew(sf_world)                                                  # (N,3,3)
        accel_cov = (self.accel_noise_std ** 2) \
            * torch.eye(3, device=self.x.device, dtype=self.dtype).expand(self.n, 3, 3) \
            + (self.attitude_noise_std ** 2) * (S @ S.transpose(-1, -2))
        dt2 = dt_t[:, None, None]
        # B = [[0.5 dt^2 I], [dt I]]; Q blocks written explicitly (cheaper than forming B)
        Q = torch.zeros(self.n, 6, 6, device=self.x.device, dtype=self.dtype)
        Q[:, :3, :3] = 0.25 * dt2 ** 4 * accel_cov
        Q[:, :3, 3:] = 0.5 * dt2 ** 3 * accel_cov
        Q[:, 3:, :3] = 0.5 * dt2 ** 3 * accel_cov
        Q[:, 3:, 3:] = dt2 ** 2 * accel_cov
        Q = Q + self.process_floor * torch.eye(6, device=self.x.device, dtype=self.dtype)
        F = torch.zeros(self.n, 6, 6, device=self.x.device, dtype=self.dtype)
        F[:, 0, 0] = F[:, 1, 1] = F[:, 2, 2] = F[:, 3, 3] = F[:, 4, 4] = F[:, 5, 5] = 1.0
        F[:, 0, 3] = F[:, 1, 4] = F[:, 2, 5] = dt_t
        P_new = F @ self.P @ F.transpose(-1, -2) + Q
        self.P = torch.where(live[:, None, None], P_new, self.P)

    @torch.no_grad()
    def update_position(self, z: Tensor, cov: Tensor, mask: Tensor) -> None:
        """Joseph-form position update (state_estimator.py:138-153) for the envs in ``mask``, with
        the deploy 3-DOF chi2 fix gate (md <= 16.27 -- rejects teleport-class outliers exactly like
        the wire chain) and the in-plane eigenvalue floor after every update.
        z (N,3) datum-world position measurement; cov (N,3,3)."""
        H_P = self.P[:, :3, :]                                              # H @ P  (H = [I3, 0])
        PHt = self.P[:, :, :3]                                              # P @ H^T
        S = H_P[:, :, :3] + cov                                             # (N,3,3)
        y = z - self.x[:, :3]
        sol = torch.linalg.solve(S, y.unsqueeze(-1)).squeeze(-1)
        md = (y * sol).sum(-1)
        acc = mask & (md <= self.fix_chi2_thresh) if self.fix_chi2_thresh > 0 else mask
        K = torch.linalg.solve(S, PHt.transpose(-1, -2)).transpose(-1, -2)  # (N,6,3)
        x_new = self.x + torch.einsum("nij,nj->ni", K, y)
        I_KH = torch.eye(6, device=self.x.device, dtype=self.dtype).expand(self.n, 6, 6).clone()
        I_KH = I_KH - torch.cat([K, torch.zeros_like(K)], dim=-1)           # K @ H, H = [I3 | 0]
        P_new = I_KH @ self.P @ I_KH.transpose(-1, -2) + K @ cov @ K.transpose(-1, -2)
        self.x = torch.where(acc.unsqueeze(-1), x_new, self.x)
        self.P = torch.where(acc[:, None, None], P_new, self.P)
        self._apply_inplane_floor(acc)

    def _apply_inplane_floor(self, mask: Tensor) -> None:
        """state_estimator.py:155-175: floor the smallest eigenvalue of the horizontal position
        block P[:2,:2] to inplane_floor_std^2 (batched 2x2 eigh)."""
        floor_var = self.inplane_floor_std ** 2
        if floor_var <= 0.0:
            return
        block = 0.5 * (self.P[:, :2, :2] + self.P[:, :2, :2].transpose(-1, -2))
        w, V = torch.linalg.eigh(block)                                     # ascending eigenvalues
        need = mask & (w[:, 0] < floor_var)
        w = w.clamp(min=floor_var)
        rebuilt = (V * w.unsqueeze(-2)) @ V.transpose(-1, -2)
        self.P[:, :2, :2] = torch.where(need[:, None, None], rebuilt, self.P[:, :2, :2])

    # -------------------------------------------------------------------- readout
    @property
    def velocity(self) -> Tensor:
        return self.x[:, 3:]

    @property
    def position(self) -> Tensor:
        return self.x[:, :3]
