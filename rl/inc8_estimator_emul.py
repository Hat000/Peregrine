"""rl/inc8_estimator_emul.py -- TORCH batched estimator-emulation for the inc8 training env.

The torch-vectorised PRODUCTION TWIN of the numpy escape-hatch reference ``rl/estimator_emul.py``
(GREEN-verified on main). The numpy module is the SPEC + the PARITY TARGET; this module reproduces
its obs over N GPU envs so a PPO policy learns to point the camera (pointing -> fix density -> KF
accuracy -> the obs the policy is scored on). Built in PARITY-GATED STAGES, each pinned by a torch==
numpy test (tests/test_inc8_*_torch.py):

  S1  batched fix-surrogate camera geometry (R_camera_from_body, project, in_image, band-pass accept,
      anisotropic gate-plane cov)          -> ``batched_geometry`` / ``BatchedFixSurrogate``
  S2  batched 6-state LinearKF (predict w/ synth-IMU + attitude-noise Q; SPD gate-frame update)
                                           -> ``BatchedLinearKF``
  S3  per-env estimator-emulation step + the 20-dim obs builder (FROZEN d5 obs[17:20])
                                           -> ``BatchedEstimatorEmulator``

DESIGN INVARIANTS (each mirrors ``estimator_emul.py`` -- divergence voids the verified selection
metric, so parity is non-negotiable):
  * NED INTERNALLY. The camera mount (+20 deg about body Y), the gate normals, gravity and
    ``R_camera_from_body`` are NED/FRD conventions; running the KF + fix model in NED makes this a
    BYTE-faithful port of the numpy reference (the DiffAero env hands us Z-up state, which the env
    subclass flips to NED at the boundary -- ``_FLIP`` is involutory). Only the final obs assembly
    flips back to Z-up (mirroring ``obs_from_zup``).
  * NO_GRAD. PPO is model-free and the Bernoulli fix is non-differentiable -- the whole emulation is
    detached (the env wraps ``step``/``obs`` in ``torch.no_grad()`` and the GT-error reward anchor is
    detached too).
  * MAP bias is NOT injected. ``gate_pos`` is TRUTH (delta_map ~ 0); perception bias enters ONLY via
    the one-signed in-plane fix bias (mirror ``estimator_emul`` exactly -- never the forbidden
    ``R_w2g @ (gate_map - p_KF)`` anti-pattern).
  * SELF-CONTAINED (numpy + torch only -- NO scipy / frames / fix_surrogate import) so it loads on the
    Adroit GPU env. Every baked constant (R_camera_from_body, K, the surrogate calibration) is pinned
    against its numpy source by a laptop parity test (``test_inc8_constants_match_numpy``).

obs[17:20] FROZEN contract (d5 1.2): c_inplane = clip(sigma_ref/sigma_inplane_hat,0,1),
c_along = clip(sigma_ref/sigma_along_hat,0,1), age_norm = clip(t_since_fix/TAU_STALE,0,1).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

try:
    import torch
    from torch import Tensor
except Exception:                       # pragma: no cover - torch absent in some tooling contexts
    torch = None
    Tensor = "Tensor"                   # type: ignore


# ============================================================================ baked constants
# NED gravity specific-force convention (racer.state_estimator.GRAVITY_NED).
GRAVITY_NED_NP = np.array([0.0, 0.0, 9.80665], dtype=np.float64)
# NED<->Z-up / FRD<->FLU axis flip (fly_rl._FLIP); involutory.
FLIP_NP = np.array([1.0, -1.0, -1.0], dtype=np.float64)

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 360
CAMERA_INTRINSICS_K_NP = np.array(
    [[320.0,   0.0, 320.0],
     [  0.0, 320.0, 180.0],
     [  0.0,   0.0,   1.0]], dtype=np.float64)


def _r_camera_from_body_np() -> np.ndarray:
    """``frames.R_camera_from_body()`` recomputed WITHOUT scipy (pinned == frames by a parity test):
    axis-swap (tilted body FRD -> camera optical) composed with a passive +20 deg pitch-up about
    body Y. Rotation.from_euler('Y', -a) == Ry(-a) = [[cos,0,sin],[0,1,0],[-sin,0,cos]] with a=20deg."""
    a = np.deg2rad(20.0)
    # Rotation.from_euler('Y', -a).as_matrix() == Ry(-a) = [[c,0,-s],[0,1,0],[s,0,c]] (c=cos a, s=sin a)
    ry_neg = np.array([[np.cos(a), 0.0, -np.sin(a)],
                       [0.0, 1.0, 0.0],
                       [np.sin(a), 0.0, np.cos(a)]], dtype=np.float64)
    r_cam_from_tilted = np.array([[0.0, 1.0, 0.0],
                                  [0.0, 0.0, 1.0],
                                  [1.0, 0.0, 0.0]], dtype=np.float64)
    return r_cam_from_tilted @ ry_neg


R_CAMERA_FROM_BODY_NP = _r_camera_from_body_np()

# Obs[17:20] FROZEN contract (d5 1.2; MEMORY OBS CONTRACT 2026-06-13).
SIGMA_REF_M = 0.05          # confidence normalizer: c=1 <=> at-or-better-than the 1-sigma bar
TAU_STALE_S = 0.10          # staleness horizon (~3 ticks @ 30 Hz)


# NED<->Z-up virtual body flip (pi about body z; fly_rl._RZ_PI_BODY) -- deploy nose-first convention.
_RZ_PI_BODY_NP = np.diag([-1.0, -1.0, 1.0]).astype(np.float64)


# ============================================================================ obs builder (S3 seam)
def _gate_rotmat_w2g_torch(yaw: Tensor) -> Tensor:
    """world->gate rotation per env (N,) -> (N,3,3), rows [[c,s,0],[-s,c,0],[0,0,1]] -- EXACTLY
    fly_rl._gate_rotmat_w2g / diffaero get_gate_rotmat_w2g / world_to_gateframe."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    z, o = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack([torch.stack([c, s, z], dim=-1),
                        torch.stack([-s, c, z], dim=-1),
                        torch.stack([z, z, o], dim=-1)], dim=-2)


def ned_gate_frame_torch(yaw: Tensor) -> Tensor:
    """The contracts.Gate NED gate frame R_world_gate (gate->world NED) per gate (G,) -> (G,3,3) --
    the torch mirror of estimator_emul.ned_gate_frame: columns [right, down, downrange] =
    [[s,0,c],[c,0,-s],[0,1,0]] (c=cos yaw, s=sin yaw). NOT the Z-up obs frame (footgun)."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    z, o = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack([torch.stack([s, z, c], dim=-1),
                        torch.stack([c, z, -s], dim=-1),
                        torch.stack([z, o, z], dim=-1)], dim=-2)


def _euler_zyx_torch(R: Tensor) -> Tensor:
    """ZYX Euler [roll, pitch, yaw] from R (N,3,3) -- the torch mirror of fly_rl._euler_zyx
    (== pytorch3d matrix_to_euler_angles(R,'ZYX')[...,[2,1,0]]). The asin clip is the float32 NaN
    guard (asin(1+eps)=NaN) AND keeps numpy parity (numpy _euler_zyx clips identically)."""
    pitch = torch.asin(torch.clamp(-R[..., 2, 0], -1.0, 1.0))
    yaw = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    roll = torch.atan2(R[..., 2, 1], R[..., 2, 2])
    return torch.stack([roll, pitch, yaw], dim=-1)


def quat_xyzw_to_matrix_torch(q_xyzw: Tensor) -> Tensor:
    """R_world_body from an XYZW (real-last, DiffAero) quaternion (N,4) -> (N,3,3). Matches
    pytorch3d.quaternion_to_matrix(q.roll(1)) used by peregrine_racing.get_observations."""
    x, y, z, w = q_xyzw[..., 0], q_xyzw[..., 1], q_xyzw[..., 2], q_xyzw[..., 3]
    tx, ty, tz = 2.0 * x, 2.0 * y, 2.0 * z
    twx, twy, twz = tx * w, ty * w, tz * w
    txx, txy, txz = tx * x, ty * x, tz * x
    tyy, tyz, tzz = ty * y, tz * y, tz * z
    r00 = 1.0 - (tyy + tzz); r01 = txy - twz; r02 = txz + twy
    r10 = txy + twz; r11 = 1.0 - (txx + tzz); r12 = tyz - twx
    r20 = txz - twy; r21 = tyz + twx; r22 = 1.0 - (txx + tyy)
    return torch.stack([torch.stack([r00, r01, r02], dim=-1),
                        torch.stack([r10, r11, r12], dim=-1),
                        torch.stack([r20, r21, r22], dim=-1)], dim=-2)


def obs_zup_torch(pos_zup: Tensor, vel_zup: Tensor, R_b2w_zup: Tensor, w_flu: Tensor,
                  gate_pos_zup: Tensor, gate_yaw: Tensor, gate_rel_pos_nxt: Tensor,
                  gate_yaw_rel_nxt: Tensor, last_normed: Tensor, triple: Tensor | None = None,
                  virtual_flip: bool = False) -> Tensor:
    """The batched 17-/20-dim obs from DiffAero Z-up/FLU state -- the torch mirror of
    ``fly_rl.obs_from_zup`` (bit-exact == peregrine_racing.get_observations, P4-C05). pos/vel come
    from the KF estimate (the case-C channel) in the inc8 env; attitude/rates/collective/lookahead
    from truth (the deploy seam). ``triple`` (N,3) appended -> 20-dim; None -> 17-dim. Per-env gate
    tensors carry the runtime per-gate yaw (works on VQ1 all-pi AND procedural courses)."""
    if virtual_flip:
        Rz = torch.as_tensor(_RZ_PI_BODY_NP, device=pos_zup.device, dtype=pos_zup.dtype)
        R_b2w_zup = R_b2w_zup @ Rz
        w_flu = torch.einsum("ij,nj->ni", Rz, w_flu)
    R_w2g = _gate_rotmat_w2g_torch(gate_yaw)
    pos_g = torch.einsum("nij,nj->ni", R_w2g, gate_pos_zup - pos_zup)
    vel_g = torch.einsum("nij,nj->ni", R_w2g, vel_zup)
    rpy_g = _euler_zyx_torch(torch.matmul(R_w2g, R_b2w_zup))
    obs = torch.cat([pos_g, vel_g, rpy_g, w_flu, last_normed.unsqueeze(-1),
                     gate_rel_pos_nxt, gate_yaw_rel_nxt.unsqueeze(-1)], dim=-1)
    if triple is not None:
        obs = torch.cat([obs, triple], dim=-1)
    return obs


# ============================================================================ surrogate params
@dataclass(frozen=True)
class TorchSurrogateParams:
    """The calibrated analytic fix model's SCALAR constants (a torch-side copy of
    ``fix_surrogate.FixSurrogate`` defaults -- pinned == FS.FixSurrogate() by a parity test). Carried
    as plain floats so the train-env core never imports scipy/frames. Per-EPISODE the env overrides
    ``sigma_lateral_floor`` (the DR sigma) and the in-plane biases per env (see ``EmulConfig``)."""
    # A. accept band-pass  p = pmax * sig((r-rlo)/wlo) * sig((rhi-r)/whi), hard-zeroed outside guard
    accept_pmax: float = 0.8419703950411151
    # accept_rlo lowered 16.191 (raw Track-3 fix-density edge) -> 12.0 = the POINTED accurate floor
    # (accept-geometry-2026-06-15): a camera-POINTING policy obtains accurate fixes down to ~12 m, which
    # the un-pointing inc7 data that fit 16.191 never exercised. Paired with the inc8 reward band-pass
    # perc_r_lo=12 so reward and fix-obtainability share a lower edge. (Kept == fix_surrogate.FixSurrogate
    # by the parity test.)
    accept_rlo: float = 12.0
    accept_wlo: float = 1.0000000000000007
    accept_rhi: float = 28.137694558909192
    accept_whi: float = 1.0000000000000002
    accept_p_out_of_image: float = 0.0002098635886673662
    accept_range_guard_lo: float = 9.0
    accept_range_guard_hi: float = 35.0
    # B. sigma(range): per-axis 1-sigma = max(floor, a1*range), gate-frame (m)
    sigma_lateral_floor: float = 0.10447100671864633
    sigma_lateral_a1: float = 0.0027626940998010047
    sigma_vertical_floor: float = 0.2815829316354898
    sigma_vertical_a1: float = 0.0
    sigma_depth_floor: float = 0.8524006087680642
    sigma_depth_a1: float = 0.0
    sigma_growth_max_range_m: float = 30.0
    cov_spd_floor: float = 1e-6

    @classmethod
    def from_fix_surrogate(cls, fs) -> "TorchSurrogateParams":
        """Copy the calibration from a numpy ``fix_surrogate.FixSurrogate`` (used by parity tests /
        ``from_checkpoints`` recalibration). Only the fields this torch core consumes are read --
        the biases + the per-episode lateral floor are owned by ``EmulConfig`` (one-signed DR)."""
        return cls(
            accept_pmax=fs.accept_pmax, accept_rlo=fs.accept_rlo, accept_wlo=fs.accept_wlo,
            accept_rhi=fs.accept_rhi, accept_whi=fs.accept_whi,
            accept_p_out_of_image=fs.accept_p_out_of_image,
            accept_range_guard_lo=fs.accept_range_guard_lo,
            accept_range_guard_hi=fs.accept_range_guard_hi,
            sigma_lateral_floor=fs.sigma_lateral_floor, sigma_lateral_a1=fs.sigma_lateral_a1,
            sigma_vertical_floor=fs.sigma_vertical_floor, sigma_vertical_a1=fs.sigma_vertical_a1,
            sigma_depth_floor=fs.sigma_depth_floor, sigma_depth_a1=fs.sigma_depth_a1,
            sigma_growth_max_range_m=fs.sigma_growth_max_range_m, cov_spd_floor=fs.cov_spd_floor)


@dataclass(frozen=True)
class EmulConfig:
    """The case-C estimator-emulation DR + encoding contract -- the torch mirror of
    ``estimator_emul.EmulConfig`` (the values that SUPERSEDE d5 per the P2 INC8-RL prompt / MEMORY).
    Distributions are sampled PER EPISODE in ``reset_idx``."""
    sigma_lat_lo: float = 0.05
    sigma_lat_hi: float = 0.15
    bias_mag_lo: float = 0.0
    bias_mag_hi: float = 0.19
    inject_bias: bool = True
    sigma_ref: float = SIGMA_REF_M
    tau_stale: float = TAU_STALE_S
    pos_std_init: float = 1.0
    vel_std_init: float = 5.0
    imu_accel_noise: float = 0.3
    attitude_noise: float = 0.0          # case-C: IMU+attitude trusted-with-noise (no attitude-err Q)


# ============================================================================ S1 geometry helpers
def _const(device, dtype):
    """Build the (device, dtype)-resident geometry constants once per call site (cheap; the env
    caches them). Returns (R_cb (3,3), K (3,3), flip (3,), gravity (3,))."""
    R_cb = torch.as_tensor(R_CAMERA_FROM_BODY_NP, device=device, dtype=dtype)
    K = torch.as_tensor(CAMERA_INTRINSICS_K_NP, device=device, dtype=dtype)
    flip = torch.as_tensor(FLIP_NP, device=device, dtype=dtype)
    g = torch.as_tensor(GRAVITY_NED_NP, device=device, dtype=dtype)
    return R_cb, K, flip, g


def batched_geometry(drone_pos_ned: Tensor, R_wb_ned: Tensor, gate_pos_ned: Tensor,
                     R_world_gate: Tensor, R_cb: Tensor, K: Tensor) -> dict:
    """GT relative geometry of one (per-env) gate from the drone pose -- the batched torch mirror of
    ``fix_surrogate.geometry`` (same R_camera_from_body, same projection).

    Shapes: drone_pos_ned/gate_pos_ned (N,3); R_wb_ned/R_world_gate (N,3,3); R_cb/K (3,3).
    Returns a dict of tensors over N envs:
      range (N,), az_deg (N,), el_deg (N,), bearing_deg (N,), in_image (N,bool),
      view_deg (N,), t_cam (N,3), lever (N,3).
    """
    lever = gate_pos_ned - drone_pos_ned                                  # +L, world NED  (N,3)
    rng = torch.linalg.norm(lever, dim=-1)                               # (N,)
    # t_cam = R_camera_from_body @ (R_wb^T @ lever)  == world_point_in_camera
    body_vec = torch.einsum("nji,nj->ni", R_wb_ned, lever)              # R_wb^T @ lever
    t_cam = torch.einsum("ij,nj->ni", R_cb, body_vec)                   # (N,3)
    tx, ty, tz = t_cam[..., 0], t_cam[..., 1], t_cam[..., 2]
    az = torch.rad2deg(torch.atan2(tx, tz))
    el = torch.rad2deg(torch.atan2(ty, tz))
    bearing = torch.rad2deg(torch.atan2(torch.hypot(tx, ty), tz))
    # project_camera_point: valid only in front of the camera (tz > 0); in-image iff inside the rect.
    front = tz > 0.0
    safe_tz = torch.where(front, tz, torch.ones_like(tz))
    u = (K[0, 0] * tx + K[0, 2] * tz) / safe_tz
    v = (K[1, 1] * ty + K[1, 2] * tz) / safe_tz
    in_image = front & (u >= 0.0) & (u < IMAGE_WIDTH) & (v >= 0.0) & (v < IMAGE_HEIGHT)
    # view angle vs the gate normal (gate-frame +Z = R_world_gate[:, 2]); instrumentation only.
    normal = R_world_gate[..., :, 2]                                     # (N,3)
    safe_rng = torch.clamp(rng, min=1e-9)
    cos_view = torch.clamp((lever / safe_rng.unsqueeze(-1) * normal).sum(-1).abs(), 0.0, 1.0)
    view = torch.rad2deg(torch.arccos(cos_view))
    return {"range": rng, "az_deg": az, "el_deg": el, "bearing_deg": bearing,
            "in_image": in_image, "view_deg": view, "t_cam": t_cam, "lever": lever}


class BatchedFixSurrogate:
    """Batched torch mirror of ``fix_surrogate.FixSurrogate``: accept band-pass, sigma(range),
    anisotropic gate-plane covariance, and an injected-randomness ``sample_fix`` (the env supplies
    the Bernoulli uniform + the gate-frame standard-normal draw, so the call is deterministic given
    them -- exact parity with the numpy reference under a shared draw)."""

    def __init__(self, params: TorchSurrogateParams, device, dtype):
        self.p = params
        self.device = device
        self.dtype = dtype

    # ---- A. accept ----------------------------------------------------------
    def accept_prob_in_image(self, rng: Tensor) -> Tensor:
        p = self.p
        z_lo = torch.clamp((rng - p.accept_rlo) / p.accept_wlo, -30.0, 30.0)
        z_hi = torch.clamp((p.accept_rhi - rng) / p.accept_whi, -30.0, 30.0)
        prob = p.accept_pmax / ((1.0 + torch.exp(-z_lo)) * (1.0 + torch.exp(-z_hi)))
        guard = (rng >= p.accept_range_guard_lo) & (rng <= p.accept_range_guard_hi)
        return prob * guard.to(prob.dtype)

    def p_accept(self, rng: Tensor, in_image: Tensor) -> Tensor:
        """P(accept) per env: the band-pass GATED by in-image (the camera-pointing lever)."""
        return torch.where(in_image, self.accept_prob_in_image(rng),
                           torch.full_like(rng, self.p.accept_p_out_of_image))

    # ---- B. sigma -----------------------------------------------------------
    def fix_sigma(self, rng: Tensor, lateral_floor: Tensor) -> Tensor:
        """(sigma_lateral, sigma_vertical, sigma_depth) gate-frame 1-sigma (N,3). ``lateral_floor``
        is per-env (the per-episode DR sigma_lat); vertical/depth floors + a1s are the baked scalars."""
        p = self.p
        r = torch.clamp(rng, max=p.sigma_growth_max_range_m)
        sl = torch.maximum(lateral_floor, p.sigma_lateral_a1 * r)
        sv = torch.maximum(torch.full_like(r, p.sigma_vertical_floor), torch.full_like(r, p.sigma_vertical_a1) * r)
        sd = torch.maximum(torch.full_like(r, p.sigma_depth_floor), torch.full_like(r, p.sigma_depth_a1) * r)
        return torch.stack([sl, sv, sd], dim=-1)

    def fix_covariance(self, sigma: Tensor, R_world_gate: Tensor) -> Tensor:
        """3x3 world-NED fix covariance per env (N,3,3): gate-plane anisotropic diagonal rotated to
        world (mirrors gate_relative_inplane_fix). Strictly SPD."""
        var = sigma * sigma                                              # (N,3)
        cov_gate = torch.diag_embed(var)                                # (N,3,3)
        eye = torch.eye(3, device=sigma.device, dtype=sigma.dtype)
        return (R_world_gate @ cov_gate @ R_world_gate.transpose(-1, -2)
                + self.p.cov_spd_floor * eye)

    # ---- D. sample (injected randomness) ------------------------------------
    def sample_fix(self, geom: dict, drone_pos_ned: Tensor, R_world_gate: Tensor,
                   lateral_floor: Tensor, bias_inplane: Tensor, accept_u: Tensor, noise: Tensor,
                   force: bool = False):
        """Bernoulli(p_accept) via the injected uniform ``accept_u`` (accept iff u < p, matching the
        numpy ``rng.random() >= p -> reject``); on accept return (z_ned, cov_ned, accepted_mask).
        ``noise`` (N,3) is the gate-frame standard-normal draw; ``bias_inplane`` (N,) the per-episode
        one-signed in-plane bias (applied to lateral AND vertical axes; depth bias = 0).

        z = drone_true + R_world_gate @ (bias + sigma * noise); cov = the matching SPD covariance.
        """
        sigma = self.fix_sigma(geom["range"], lateral_floor)            # (N,3)
        # one-signed bias on lateral+vertical, 0 on depth (IMU-owned along-track) -- mirror EmulConfig.
        zeros = torch.zeros_like(bias_inplane)
        bias = torch.stack([bias_inplane, bias_inplane, zeros], dim=-1)  # (N,3)
        noise_gate = bias + sigma * noise                               # (N,3)
        z = drone_pos_ned + torch.einsum("nij,nj->ni", R_world_gate, noise_gate)
        cov = self.fix_covariance(sigma, R_world_gate)
        if force:
            accepted = torch.ones_like(geom["range"], dtype=torch.bool)
        else:
            accepted = accept_u < self.p_accept(geom["range"], geom["in_image"])
        return z, cov, accepted


# ============================================================================ S2 batched LinearKF
def _skew(v: Tensor) -> Tensor:
    """Batched skew-symmetric cross-product matrix (N,3) -> (N,3,3): _skew(v) @ w == cross(v, w)."""
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    o = torch.zeros_like(x)
    return torch.stack([torch.stack([o, -z, y], dim=-1),
                        torch.stack([z, o, -x], dim=-1),
                        torch.stack([-y, x, o], dim=-1)], dim=-2)


class BatchedLinearKF:
    """6-state [pos, vel] world-NED Kalman filter over N envs -- the batched torch mirror of
    ``racer.state_estimator.LinearKF`` (predict/update_position), op-for-op."""

    def __init__(self, n: int, device, dtype, accel_noise_std: float = 0.3,
                 attitude_noise_std: float = 0.0, process_floor: float = 1e-6,
                 max_dt_s: float = 0.2):
        self.n = n
        self.device = device
        self.dtype = dtype
        self.accel_noise_std = float(accel_noise_std)
        self.attitude_noise_std = float(attitude_noise_std)
        self.process_floor = float(process_floor)
        self.max_dt_s = float(max_dt_s)
        self.x = torch.zeros(n, 6, device=device, dtype=dtype)
        self.P = torch.zeros(n, 6, 6, device=device, dtype=dtype)
        self._I3 = torch.eye(3, device=device, dtype=dtype)
        self._I6 = torch.eye(6, device=device, dtype=dtype)
        self._g = torch.as_tensor(GRAVITY_NED_NP, device=device, dtype=dtype)

    def initialize_idx(self, idx: Tensor, pos_ned: Tensor, vel_ned: Tensor,
                       pos_std: float, vel_std: float) -> None:
        """COLD init at truth for the given env indices: pos seeded, velocity seeded (==0 at a real
        rest start), uncertainty in the COVARIANCE (pos_std / vel_std)."""
        self.x[idx, :3] = pos_ned.to(self.dtype)
        self.x[idx, 3:] = vel_ned.to(self.dtype)
        diag = torch.tensor([pos_std ** 2] * 3 + [vel_std ** 2] * 3, device=self.device, dtype=self.dtype)
        self.P[idx] = torch.diag(diag)

    def seed_truth_idx(self, idx: Tensor, pos_ned: Tensor, vel_ned: Tensor) -> None:
        """Force x to truth (pos+vel) for the given envs -- the perfect zero-noise estimator (the
        frame-seam identity test's operationalization)."""
        self.x[idx, :3] = pos_ned.to(self.dtype)
        self.x[idx, 3:] = vel_ned.to(self.dtype)

    def predict(self, accel_body: Tensor, R_wb: Tensor, dt: float) -> None:
        """Propagate ALL envs by dt using body specific force + the given body->world rotation
        (mirrors LinearKF.predict; a non-positive/implausibly-large dt is dropped wholesale)."""
        if dt <= 0.0 or dt > self.max_dt_s:
            return
        I3, dt_ = self._I3, float(dt)
        specific_force_world = torch.einsum("nij,nj->ni", R_wb, accel_body)   # (N,3)
        a_world = specific_force_world + self._g
        F = torch.zeros(6, 6, device=self.device, dtype=self.dtype)
        F[:3, :3] = I3; F[3:, 3:] = I3; F[:3, 3:] = dt_ * I3
        B = torch.zeros(6, 3, device=self.device, dtype=self.dtype)
        B[:3, :] = 0.5 * dt_ * dt_ * I3; B[3:, :] = dt_ * I3
        self.x = torch.einsum("ij,nj->ni", F, self.x) + torch.einsum("ij,nj->ni", B, a_world)
        S = _skew(specific_force_world)                                      # (N,3,3)
        accel_cov = (self.accel_noise_std ** 2) * I3 + (self.attitude_noise_std ** 2) * (S @ S.transpose(-1, -2))
        Q = torch.einsum("ij,njk,lk->nil", B, accel_cov, B) + self.process_floor * self._I6
        self.P = torch.einsum("ij,njk,lk->nil", F, self.P, F) + Q

    def update_position_idx(self, idx: Tensor, z: Tensor, cov: Tensor) -> None:
        """Vision position fix on the SUBSET ``idx`` (the envs that accepted a fix this step). H
        observes position; Joseph-form update keeps P SPD (mirrors LinearKF.update with _H_POS)."""
        if idx.numel() == 0:
            return
        x = self.x[idx]                                                     # (m,6)
        P = self.P[idx]                                                     # (m,6,6)
        R = cov.to(self.dtype)                                              # (m,3,3)
        y = z.to(self.dtype) - x[:, :3]                                     # (m,3)
        PHt = P[:, :, :3]                                                   # (m,6,3)  (H^T selects cols 0:3)
        S = P[:, :3, :3] + R                                               # (m,3,3)  H P H^T + R
        # K = solve(S, PHt^T)^T  -- batched, no explicit inverse (mirrors np.linalg.solve(S, PHt.T).T)
        K = torch.linalg.solve(S, PHt.transpose(-1, -2)).transpose(-1, -2)  # (m,6,3)
        x = x + torch.einsum("nij,nj->ni", K, y)
        H = torch.zeros(3, 6, device=self.device, dtype=self.dtype)
        H[:, :3] = self._I3
        I_KH = self._I6 - torch.einsum("nij,jk->nik", K, H)                 # (m,6,6)
        P = I_KH @ P @ I_KH.transpose(-1, -2) + K @ R @ K.transpose(-1, -2)
        self.x[idx] = x
        self.P[idx] = P

    @property
    def position(self) -> Tensor:
        return self.x[:, :3]

    @property
    def velocity(self) -> Tensor:
        return self.x[:, 3:]


# ============================================================================ S3 the emulator
class BatchedEstimatorEmulator:
    """Holds the batched LinearKF + BatchedFixSurrogate + per-episode DR draws + a per-env fix-
    staleness clock. The torch twin of ``estimator_emul.EstimatorEmulator``. ALL methods are intended
    to run under ``torch.no_grad()`` (PPO; non-differentiable Bernoulli).

    The env wires it: at ``reset_idx`` draw the per-episode sigma_lat + one-signed bias and COLD-init
    the KF at truth; per ``step`` IMU-predict (prev->cur truth) then a fix to the target gate;
    ``confidence_channel`` feeds obs[17:20]; ``kf_pos_zup``/``kf_vel_zup`` feed obs[0:6]."""

    def __init__(self, n: int, gate_pos_ned: Tensor, R_world_gate_per_gate: Tensor,
                 config: EmulConfig | None = None, params: TorchSurrogateParams | None = None,
                 device=None, dtype=None):
        assert torch is not None, "BatchedEstimatorEmulator requires torch"
        self.n = n
        self.cfg = config or EmulConfig()
        device = device if device is not None else gate_pos_ned.device
        dtype = dtype if dtype is not None else gate_pos_ned.dtype
        self.device, self.dtype = device, dtype
        self.G = gate_pos_ned.shape[0]
        # course geometry in NED: gate centres (G,3) + per-gate NED gate frame (G,3,3).
        self.gate_pos_ned = gate_pos_ned.to(device=device, dtype=dtype)
        self.R_world_gate = R_world_gate_per_gate.to(device=device, dtype=dtype)
        self.surrogate = BatchedFixSurrogate(params or TorchSurrogateParams(), device, dtype)
        self.kf = BatchedLinearKF(n, device, dtype,
                                  accel_noise_std=self.cfg.imu_accel_noise,
                                  attitude_noise_std=self.cfg.attitude_noise)
        self.R_cb, self.K, self.flip, self.g = _const(device, dtype)
        # per-env episode DR state
        self._sigma_lat = torch.full((n,), float("nan"), device=device, dtype=dtype)
        self._bias = torch.full((n,), float("nan"), device=device, dtype=dtype)
        self._t_since_fix = torch.full((n,), 1e3, device=device, dtype=dtype)
        # last computed per-env geometry to the target gate (for the reward: t_cam / range / in_image)
        self._last_geom: dict | None = None

    # ---- episode lifecycle --------------------------------------------------
    def reset_idx(self, idx: Tensor, pos_ned: Tensor, vel_ned: Tensor,
                  sigma_lat: Tensor, bias: Tensor) -> None:
        """Init the KF COLD at truth + store the per-episode DR draws (the env supplies the draws so
        the env owns the RNG stream). ``sigma_lat``/``bias`` are (m,)."""
        self._sigma_lat[idx] = sigma_lat.to(self.dtype)
        self._bias[idx] = bias.to(self.dtype)
        self.kf.initialize_idx(idx, pos_ned, vel_ned,
                               pos_std=self.cfg.pos_std_init, vel_std=self.cfg.vel_std_init)
        self._t_since_fix[idx] = 1e3       # no fix yet -> age_norm == 1 (cold/stale)

    def seed_truth_idx(self, idx: Tensor, pos_ned: Tensor, vel_ned: Tensor) -> None:
        self.kf.seed_truth_idx(idx, pos_ned, vel_ned)

    @staticmethod
    def sample_episode_dr(cfg: EmulConfig, m: int, device, dtype, generator=None):
        """Draw (sigma_lat (m,), bias (m,)) for ``m`` envs -- the torch mirror of
        ``EmulConfig.sample_episode_sigma_lat`` / ``sample_episode_bias`` (one-signed bias, per-episode
        constant random sign). The env calls this and passes the draws to ``reset_idx``."""
        u = torch.rand(m, device=device, dtype=dtype, generator=generator)
        sigma_lat = cfg.sigma_lat_lo + (cfg.sigma_lat_hi - cfg.sigma_lat_lo) * u
        if not cfg.inject_bias:
            bias = torch.zeros(m, device=device, dtype=dtype)
        else:
            mag_u = torch.rand(m, device=device, dtype=dtype, generator=generator)
            mag = cfg.bias_mag_lo + (cfg.bias_mag_hi - cfg.bias_mag_lo) * mag_u
            sign = torch.where(torch.rand(m, device=device, dtype=dtype, generator=generator) < 0.5,
                               torch.ones(m, device=device, dtype=dtype),
                               -torch.ones(m, device=device, dtype=dtype))
            bias = sign * mag
        return sigma_lat, bias

    # ---- the per-step update ------------------------------------------------
    def step(self, prev_pos_ned: Tensor, prev_vel_ned: Tensor, R_prev_wb: Tensor,
             cur_pos_ned: Tensor, cur_vel_ned: Tensor, R_cur_wb: Tensor, target_gate: Tensor,
             dt: float, accept_u: Tensor, accel_noise: Tensor, fix_noise: Tensor,
             force_accept: bool = False, noiseless: bool = False) -> Tensor:
        """Advance the KF one control step for ALL envs: IMU predict (prev->cur truth) then a fix to
        each env's ``target_gate``. Returns the per-env accepted mask (bool, N).

        Injected randomness (env-drawn, so the env owns the stream + parity is exact under a shared
        draw): ``accept_u`` (N,) Bernoulli uniform, ``accel_noise`` (N,3) IMU std-normal,
        ``fix_noise`` (N,3) gate-frame std-normal. ``noiseless`` zeroes both noise streams (frame-seam
        identity); ``force_accept`` bypasses the Bernoulli draw.
        """
        dt = float(dt)
        # -- IMU predict: synthesize body specific force from truth, add isotropic accel noise.
        if dt > 0.0:
            a_world = (cur_vel_ned - prev_vel_ned) / dt
        else:
            a_world = torch.zeros_like(cur_vel_ned)
        accel_body = torch.einsum("nji,nj->ni", R_prev_wb, a_world - self.g)   # R_prev^T @ (a_world - g)
        if not noiseless:
            accel_body = accel_body + self.cfg.imu_accel_noise * accel_noise
        self.kf.predict(accel_body, R_prev_wb, dt)
        self._t_since_fix = self._t_since_fix + max(dt, 0.0)

        # -- vision fix to the current target gate (camera-pointing gates acceptance).
        gp = self.gate_pos_ned[target_gate]                                  # (N,3)
        Rwg = self.R_world_gate[target_gate]                                 # (N,3,3)
        geom = batched_geometry(cur_pos_ned, R_cur_wb, gp, Rwg, self.R_cb, self.K)
        self._last_geom = {**geom, "R_world_gate": Rwg}
        noise = torch.zeros_like(fix_noise) if noiseless else fix_noise
        z, cov, accepted = self.surrogate.sample_fix(
            geom, cur_pos_ned, Rwg, self._sigma_lat, self._bias, accept_u, noise, force=force_accept)
        idx = accepted.nonzero(as_tuple=False).view(-1)
        if idx.numel() > 0:
            self.kf.update_position_idx(idx, z[idx], cov[idx])
            self._t_since_fix[idx] = 0.0
        return accepted

    # ---- confidence channel + obs sources -----------------------------------
    def _gate_frame_sigmas(self, target_gate: Tensor):
        """(sigma_inplane_hat (N,), sigma_along_hat (N,)): KF position cov projected into the NED gate
        frame. in-plane = RMS of the two opening-plane axis stds; along-track = the through-axis std."""
        Rwg = self.R_world_gate[target_gate]                                # (N,3,3)
        P_pos = self.kf.P[:, :3, :3]
        P_gate = Rwg.transpose(-1, -2) @ P_pos @ Rwg
        var_ip = 0.5 * (P_gate[:, 0, 0].clamp(min=0.0) + P_gate[:, 1, 1].clamp(min=0.0))
        var_al = P_gate[:, 2, 2].clamp(min=0.0)
        return torch.sqrt(var_ip), torch.sqrt(var_al)

    def confidence_channel(self, target_gate: Tensor) -> Tensor:
        """[c_inplane, c_along, age_norm] (N,3) from the CALIBRATED KF gate-frame covariance + the
        staleness clock (FROZEN d5 1.2): c = clip(sigma_ref/sigma_hat, 0, 1); age = clip(t/TAU, 0, 1)."""
        cfg = self.cfg
        sig_ip, sig_al = self._gate_frame_sigmas(target_gate)
        c_ip = torch.where(sig_ip > 0, torch.clamp(cfg.sigma_ref / sig_ip.clamp(min=1e-12), 0.0, 1.0),
                           torch.ones_like(sig_ip))
        c_al = torch.where(sig_al > 0, torch.clamp(cfg.sigma_ref / sig_al.clamp(min=1e-12), 0.0, 1.0),
                           torch.ones_like(sig_al))
        age = torch.clamp(self._t_since_fix / cfg.tau_stale, 0.0, 1.0)
        return torch.stack([c_ip, c_al, age], dim=-1)

    def kf_pos_zup(self) -> Tensor:
        """KF position estimate flipped to the DiffAero Z-up frame (obs[0:3] source)."""
        return self.kf.position * self.flip

    def kf_vel_zup(self) -> Tensor:
        """KF velocity estimate flipped to the DiffAero Z-up frame (obs[3:6] source)."""
        return self.kf.velocity * self.flip

    def gate_frame_error_inplane(self, target_gate: Tensor, cur_pos_ned: Tensor) -> Tensor:
        """|KF_pos - truth| in-plane (gate-frame E,D) per env (N,) -- the GT-estimator-error anchor
        the reward reads (reward sees TRUTH; actor sees the noisy obs). depth/along is excluded."""
        Rwg = self.R_world_gate[target_gate]
        e_world = self.kf.position - cur_pos_ned
        e_g = torch.einsum("nij,nj->ni", Rwg.transpose(-1, -2), e_world)     # gate frame [right,down,along]
        return torch.hypot(e_g[..., 0], e_g[..., 1])
