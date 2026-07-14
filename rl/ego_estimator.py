"""rl/ego_estimator.py -- batched (torch) RELATIVE-STATE estimator emulator for the VQ2 egocentric
generation (DESIGN.md `docs/vq2-egocentric-gen/DESIGN.md` component A).

Replaces the world-NED 6-state KF of ``rl/inc8_estimator_emul.py`` (which is KEPT untouched -- curr_b2
and the whole inc8 generation still run on it). This module NEVER edits that file; it IMPORTS the
measured constants/helpers from it.

THE CORE PRINCIPLE (load-bearing, DESIGN.md §2, §5.A): the estimator holds **NO world/absolute
position and NO world heading**. Its entire state is body-frame / relative:

  (1) PER-GATE RELATIVE POSITION -- for each gate g, the drone->gate vector expressed in the drone
      BODY frame (Z-up / FLU, as ``rl.peregrine_racing`` uses). Straight from vision (true relative
      vector + anisotropic gate-frame noise), KF-smoothed over frames, ego-propagated through short
      no-fix gaps (rotate by the inter-frame body rotation, translate by -v*dt), MASKED (confidence
      -> 0) once stale past the horizon.
  (2) PER-GATE NORMAL/YAW + CONFIDENCE -- the gate facing relative to the drone (the weakest monocular
      DOF: its own larger, range-collapsing angular noise + an occasional two-fold flip + its own
      confidence), plus a per-gate confidence/staleness scalar.
  (3) VELOCITY -- drone BODY-frame velocity, IMU-primary (bias-dominated drift + tiny uniform accel
      noise), vision-corrected toward truth at each accepted fix.
  (4) roll/pitch (gravity-leveled from the accelerometer) + body rates (colored gyro). NO absolute yaw.

FRAME CONVENTIONS (matched EXACTLY to rl.peregrine_racing / rl.gate_visibility -- NOT invented):
  * Inputs are DiffAero **Z-up / FLU**: drone pos/vel world Z-up, drone quat XYZW body->world Z-up,
    body rates FLU, gate_pos (N,G,3) Z-up centres, gate_yaw (N,G) about +Z.
  * The BODY frame is FLU. A per-gate relative position is ``R_wb^T @ (gate_pos - drone_pos)`` (Z-up
    world lever rotated into the body frame). This is manifestly invariant to a global world
    translation (the lever is a difference) and holds no world coordinate.
  * The vision anisotropic noise (lat/vert/depth, DESIGN.md §5.A) is MEASURED in the GATE frame. We
    build it in the Z-up gate frame ``R_world_gate_zup`` (columns [right, downrange, up] -- the Z-up
    analog of estimator_emul's NED gate frame) and rotate it into the body frame together with the
    lever, so the anisotropy sits on the correct axes.
  * Visibility is decided by ``rl.gate_visibility.gate_detectable`` (pure truth geometry). A
    non-detectable gate gets NO fresh fix: propagate-then-mask.

MEASURED NOISE MODELS (baked here; sigmas imported from ``inc8_estimator_emul.TorchSurrogateParams``):
  * VISION per-gate relative fix (gate frame, m): lateral = max(0.1045, 0.0028*range);
    vertical = 0.2816; depth = 0.8524. Per-episode small in-plane bias (bias_mag_hi ~0.19). RANGE-
    COLLAPSE (lateral shrinks with range). KF smoothing realises in-plane sigma ~ per-fix/sqrt(N_eff),
    N_eff ~ U[4,9]. FIX-B latency covariance inflation, stochastic per-frame MISS even when visible,
    and a small-probability RANDOM-IN-FRAME TELEPORT OUTLIER (replace the fix with a wildly wrong
    relative position).
  * GATE NORMAL/YAW: the weakest monocular DOF -- own larger range-collapsing angular sigma (VQ1
    char ~ +-3 deg tightening at close range) + occasional two-fold flip + own confidence.
  * IMU VELOCITY (run 024203, 82 s stationary @ 143.5 Hz): drift is BIAS-DOMINATED, not noise. Accel
    white noise tiny, UNIFORM per-axis ~0.008 m/s^2 (the measured x-axis 10x-quiet is a sim artifact,
    NOT replicated). The drift lever = a per-FLIGHT residual accel bias (post-pad-cal) integrated over
    Delta-t-since-fix; DR the residual bias per episode over a band (default per-axis ~ U[-0.05,0.05]
    m/s^2, a config knob). Velocity corrected toward truth at each accepted fix.
  * GYRO noise is COLORED: AR(1), lag-1 autocorr ~0.75, sigma ~5e-5 rad/s. Modeled AR(1), NOT white.

NO_GRAD: like the inc8 emulator, the Bernoulli fix / teleport outlier are non-differentiable; the env
wraps ``step`` in ``torch.no_grad()``.

PUBLIC API:
    EgoEstimatorConfig                                  -- the DR + noise contract (dataclass)
    BatchedEgoEstimator(n, gate_pos_zup, gate_yaw, ...) -- the stateful per-env estimator
      .reset_idx(idx, drone_pos, drone_vel, drone_quat) -- COLD init at truth spawn + per-episode DR
      .step(drone_pos, drone_vel, drone_quat, body_rates, dt, detectable=None) -> EgoEstimate
      .estimate() -> EgoEstimate                        -- read the current outputs without stepping
    EgoEstimate  -- the outputs (1)-(4): per-gate rel pos/normal/confidence, velocity, roll/pitch/rates
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

# Reuse MEASURED constants + helpers from the inc8 emulator (single source of truth; do NOT copy).
from inc8_estimator_emul import (
    TorchSurrogateParams,
    GRAVITY_NED_NP,
    quat_xyzw_to_matrix_torch,
)
# ESTIMATOR-FAITHFUL package (2026-07-11): the batched deploy-filter translations. Default-OFF --
# constructed only when the att/vel model knobs arm them (see EgoEstimatorConfig below).
from ego_ins_emul import (
    BatchedESKFLeveler,
    BatchedNavKF,
    KF_FIX_COV_FLOOR_STD,
    MEASURED_TICK_GAP_PMF,
    quat_from_roll_pitch_yaw_zyx,
)
# The visibility gate (pure truth geometry) -- optional dependency: the env passes a precomputed
# ``detectable`` mask into step(); we import gate_detectable so the estimator can compute it itself
# when the caller does not (and so a test can drive it end-to-end).
try:
    from gate_visibility import gate_detectable as _gate_detectable
    from gate_visibility import gate_apparent_area as _gate_apparent_area
except Exception:                       # pragma: no cover - allow standalone import ordering
    _gate_detectable = None
    _gate_apparent_area = None


GRAVITY_MAG = float(np.linalg.norm(GRAVITY_NED_NP))     # 9.80665


# ================================================================================================
# Config -- the DR + noise contract (mirrors the inc8 EmulConfig style; all knobs Fengyou-pinnable).
# ================================================================================================
@dataclass(frozen=True)
class EgoEstimatorConfig:
    # ---- vision per-gate relative-fix smoothing / robustness ----
    n_eff_lo: int = 4                    # N_eff ~ U[n_eff_lo, n_eff_hi]; realised sigma = per-fix/sqrt(N_eff)
    n_eff_hi: int = 9
    bias_mag_hi: float = 0.19            # per-episode small in-plane (lat+vert) bias magnitude upper bound
    inject_bias: bool = True
    # ---- per-episode bias MODEL (audit handoff/audit-ego-inc9-2026-07-09/read_estimator-kf-audit.md [major]) ----
    # 'legacy' (DEFAULT, byte-identical): ONE scalar magnitude ~U[0, bias_mag_hi] with a RANDOM sign,
    #   applied to BOTH lat and vert (perfectly correlated), depth bias 0 -- inherited from inc8
    #   sample_fix [b,b,0]. Contradicts the measurement three ways (audit red flag [major]): over-injects
    #   lateral bias ~5x, fabricates a lat/vert correlation, and randomizes the sign of a measured
    #   ONE-SIGNED (calibratable) vertical systematic.
    # 'measured': INDEPENDENT per-axis magnitude ~U[0, |band_axis|] with the MEASURED (fixed) sign per
    #   axis, from handoff/fix-surrogate-2026-06-14/models/sigma.json {lat -0.0338, vert +0.1949,
    #   depth -0.3344}. Fixes all three defects and makes depth bias NON-zero (was 0 in legacy). Still
    #   scaled by noise_scale (0 -> no bias). New path -> its own draw count; the legacy path is untouched.
    bias_model: str = "legacy"
    bias_band_lat: float = -0.033817701667839546    # sigma.json lateral.bias_band  (one-signed, negative)
    bias_band_vert: float = 0.19488467958140898     # sigma.json vertical.bias_band (one-signed, positive)
    bias_band_depth: float = -0.33441613167150736   # sigma.json depth.bias_band    (one-signed, negative)
    miss_prob: float = 0.10              # stochastic per-frame MISS even when the gate is visible
    teleport_prob: float = 0.002         # RANDOM-IN-FRAME teleport outlier probability (per visible gate)
    teleport_scale_m: float = 8.0        # magnitude of the wild wrong relative position on a teleport
    # INERT (2026-07-10 cleanup): this knob fed ONLY the dead ``_rel_var`` buffer (written, never read --
    # confidence comes solely from t_since_fix); the dead writes are now deleted, so setting this changes
    # NOTHING. The field is KEPT so a cfg that sets it does not crash. Wiring it into confidence would be
    # a BEHAVIOR change (a default-off future item, not the perception-honesty package).
    latency_cov_inflate: float = 1.5     # INERT -- see above (was: FIX-B covariance inflation on a fix)
    stale_horizon_s: float = 0.5         # propagate-through-gap horizon; past this -> MASK (confidence 0)

    # ---- gate normal/yaw (weakest monocular DOF) ----
    normal_sigma_floor_rad: float = 0.0524     # ~3 deg floor (VQ1 char) at close range
    normal_sigma_a1_rad_per_m: float = 0.00349 # +~0.2 deg/m range-growth (tightens as range shrinks)
    normal_flip_prob: float = 0.03             # occasional two-fold (pi) flip of the facing
    normal_conf_ref_rad: float = 0.0873        # ~5 deg: normal confidence = clip(ref/sigma_hat, 0, 1)

    # ---- IMU velocity (bias-dominated drift; MEASURED run 024203) ----
    accel_white_sigma: float = 0.008     # UNIFORM per-axis accel white noise (m/s^2); x-quirk NOT baked
    accel_bias_band: float = 0.05        # per-FLIGHT residual accel bias per axis ~ U[-band, +band] (m/s^2)
    dr_accel_bias: bool = True           # draw a per-episode residual bias (False -> zero, for tests)
    vel_correct_gain: float = 0.15       # INDIRECT velocity-correction gain at an accepted fix. The VQ2
                                         # wire carries NO direct vision-velocity observation; deploy
                                         # corrects velocity ONLY through the position-fix pos-vel cross-
                                         # covariance -> a small PARTIAL gain toward truth, never a snap.

    # ---- gyro colored noise (AR(1)) ----
    gyro_ar1_rho: float = 0.75           # lag-1 autocorrelation of the injected gyro noise
    gyro_sigma: float = 5e-5             # marginal std (rad/s) of the injected gyro noise

    # ---- visible-area (apparent projected opening area) channel (DESIGN.md §5.C) ----
    # A ROBUST replacement for the dropped PnP normal: the NORMALIZED apparent opening area in [0,1], 1 ==
    # square-on. Fengyou 2026-07-07 RECALIBRATION: this is the actual projected inner-opening area a
    # corner detector reports (gate_visibility.gate_apparent_area), normalized by the square-on area at
    # the range -> range-invariant, and PERSPECTIVE-accurate (the old |cos(view_ray, gate_normal)| proxy
    # diverged from the true projected area close-in and would not match the deployed detector). Head-on
    # -> ~1 (a big square, easy to thread); edge-on -> ~0 (a foreshortened sliver -> the policy learns to
    # square up first). GT geometry + a MODEST placeholder noise (config knob, flagged for the real vision
    # measurement). Masked (0) when the gate is not detectable/stale, exactly like ``confidence``.
    visible_area_sigma: float = 0.05     # placeholder additive-noise std on the ratio (config knob)

    # ---- init uncertainty (feeds the KF gain shape only; no world state) ----
    rel_pos_std_init: float = 5.0        # INERT (2026-07-10): only seeded the deleted _rel_var buffer;
    #                                      kept so a cfg that sets it does not crash (see latency_cov_inflate)
    far_cap_m: float = 30.0              # visibility far cap (passed to gate_detectable)

    # ---- ESTIMATOR-FAITHFUL obs models (2026-07-11 package; owner directive: the actor NEVER sees
    # GT). Defaults == current behaviour EXACTLY (byte-identical, zero new RNG draws on the legacy
    # branches). Armed by +env.ego_faithful=true (all three) or per-channel for ablation. The
    # emulated filters are ALGORITHM-STRUCTURAL and DELIBERATELY NOT scaled by noise_scale (blur
    # precedent: the noise-0 calibration boot must still fly the lying leveler, or the fullstack
    # inherits a boot trained on truth attitude -- the exact fatal gap this package closes); only
    # the vision-fix noise feeding the KF scales with noise_scale (it already does). ----
    # 'gt'  : roll/pitch = perfect extraction of the truth quat (LEGACY -- the measured-fatal lie).
    # 'eskf': roll/pitch = the translated deploy ESKF leveler (ego_ins_emul.BatchedESKFLeveler,
    #         full gate stack incl. A8 motion-reject), stepped ONCE per control tick on the LATEST
    #         IMU sample (the verified deploy rate contract), seeded at truth roll/pitch + yaw
    #         datum 0 with the PAD-CONVERGED launch covariance.
    att_model: str = "gt"
    # 'legacy': truth-seeded bias-drift + 0.15-gain pull toward truth at a fix (a hand-modeled
    #           surrogate -- reproduces NEITHER deploy error injection).
    # 'kf'    : the translated deploy LinearKF in the per-env yaw-DATUM world frame: strapdown
    #           predict through the EMULATED attitude + position-fix-only corrections
    #           (PnP-through-estimated-attitude measurement model), velocity projected world->body
    #           through the SAME emulated attitude -- both deploy error injections reproduced.
    vel_model: str = "legacy"
    # 'ar1'    : truth + AR(1) noise (rho .75 / sigma 5e-5 -- provenance-free, retired on the
    #            faithful path).
    # 'sampled': the raw latest gyro sample with MEASURED-ZERO added noise (a5 pad-idle: the sim
    #            IMU is noiseless -- rl/tools/imu_foundation.py; deploy obs[5:8] = s.gyro_body RAW,
    #            fly_rl.py:1810). No RNG draw on this path.
    rate_model: str = "ar1"
    # ---- CHOKED-LOOP dt EMULATION (reviewer-caught gap 2026-07-11): the wire nav loop has NEVER
    # run at the 30 Hz training tick (measured ego_obs tick gaps -- a5 median 41.6 ms / p90 107 /
    # max 139 ms; a7 median 34.8 / max 83; see ego_ins_emul.MEASURED_TICK_GAP_PMF provenance), and
    # per-tick leveler divergence SCALES WITH dt (leveler_bench: median 1.5 / p90 4.6 deg at 33 ms
    # vs 5.2 / 26.8 at the recorded ~57 ms grid on the same real a5 IMU) -- so a fixed-33 ms
    # emulation trains against a ~3.5x cleaner lie than the wire it will meet. When > 1, the
    # emulated ESKF/KF (and the sampled-rate channel) advance on a RENEWAL schedule: after each
    # advance the env draws its next hold length k from the MEASURED pooled tick-gap pmf truncated
    # to {1..est_dt_ticks_hi} and re-normalized, then holds (obs channels frozen -- the wire
    # freezes them too between loop ticks) for k-1 ticks and advances with dt = k * env.dt on the
    # k-th. The pmf and the K=4 band are MEASURED, not invented (invariant 3); the policy still
    # acts every training tick, which the wire's action-hold does not -- a conservative (harder)
    # direction, documented. 1 (default) == byte-identical (advance every tick, dt = env.dt, no
    # RNG draw). Only meaningful with the faithful models armed. Set via +env.ego_est_dt_ticks_hi.
    est_dt_ticks_hi: int = 1

    # ---- GLOBAL noise scale (DIAGNOSTIC lever, Fengyou 2026-07-09; the perception-vs-control ablation) ----
    # A single multiplier on ALL injected estimator noise/corruption: the anisotropic vision fix sigma,
    # the per-episode in-plane bias, the teleport/miss/normal-flip probabilities, the IMU accel white +
    # residual-bias drift, the colored gyro, and the visible-area noise. 1.0 == the measured contract
    # (default -> numerically identical to every prior run). 0.0 == a PERFECT (truth) estimator. Set via
    # +env.ego_noise_scale to answer "is the ~0.88 m centring floor perception- or control-limited?":
    # zeroing it and re-training warm-from-champion shows how much of the offset is estimator noise.
    noise_scale: float = 1.0

    def n_eff_mean(self) -> float:
        return 0.5 * (self.n_eff_lo + self.n_eff_hi)


# ================================================================================================
# Output container.
# ================================================================================================
@dataclass
class EgoEstimate:
    """The estimator outputs -- ALL body-frame / relative, NO world position, NO world heading.

    rel_pos      (N,G,3)  per-gate drone->gate vector in the drone BODY frame (FLU), smoothed.
    rel_normal   (N,G,3)  per-gate unit facing vector (gate through-axis) in the drone BODY frame.
    rel_yaw      (N,G)    per-gate facing angle = atan2(normal_y, normal_x) in the body frame (rad).
    confidence   (N,G)    per-gate confidence in [0,1] (staleness + fix quality); 0 == MASKED (stale).
    normal_conf  (N,G)    per-gate normal/yaw confidence in [0,1] (the weak DOF's own scalar).
    velocity     (N,3)    drone velocity in the BODY frame (IMU-primary, vision-corrected).
    roll_pitch   (N,2)    gravity-leveled roll, pitch (rad). NO yaw.
    body_rates   (N,3)    body angular rates (rad/s, FLU) with colored gyro noise added.
    visible_area (N,G)    per-gate NORMALIZED apparent opening area in [0,1] (1 == square-on) -- the
                          projected inner-opening area a corner detector reports, range-normalized (GT
                          via gate_apparent_area) + modest noise; 0 == MASKED (not detectable/stale).
    """
    rel_pos: Tensor
    rel_normal: Tensor
    rel_yaw: Tensor
    confidence: Tensor
    normal_conf: Tensor
    velocity: Tensor
    roll_pitch: Tensor
    body_rates: Tensor
    visible_area: Tensor


# ================================================================================================
# Gate frame (Z-up) -- the Z-up analog of estimator_emul.ned_gate_frame_torch.
# ================================================================================================
def zup_gate_frame_torch(yaw: Tensor) -> Tensor:
    """Per-gate Z-up gate frame R_world_gate (gate->world, Z-up FLU) from yaw (...,) -> (...,3,3),
    columns [right, downrange, up]:
        right    = [ sin(yaw), -cos(yaw), 0]   (in-plane lateral)
        downrange= [ cos(yaw),  sin(yaw), 0]   (through-axis / gate normal, Z-up)
        up       = [ 0,         0,        1]    (in-plane vertical)
    This is the Z-up mirror of ``estimator_emul.ned_gate_frame`` (which is NED [right,down,downrange]);
    here the anisotropic vision noise columns are [lateral, downrange/depth, vertical] so the
    fix-noise draw stacks as [lat, depth, vert]. The gate normal (facing) is the DOWNRANGE column."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    z, o = torch.zeros_like(c), torch.ones_like(c)
    right = torch.stack([s, -c, z], dim=-1)
    downrange = torch.stack([c, s, z], dim=-1)
    up = torch.stack([z, z, o], dim=-1)
    return torch.stack([right, downrange, up], dim=-1)      # columns -> (...,3,3)


def _euler_roll_pitch_from_R(R_wb: Tensor) -> Tensor:
    """Gravity-levelled roll, pitch (rad) from a body->world Z-up rotation (N,3,3). ZYX convention on
    the Z-up frame: pitch = asin(R[2,0])-analog. We take the body-z axis in world = R[:, :, 2] (third
    column) and read tilt from it -- the accelerometer observes exactly this (gravity direction in
    body). roll = atan2(R21, R22-ish); we use the standard Z-up extraction:
        pitch = atan2(-R[2,0], hypot(R[2,1], R[2,2]))     (nose up/down)
        roll  = atan2(R[2,1], R[2,2])                     (bank)
    NO yaw is extracted (it is world heading -- forbidden)."""
    r20 = R_wb[..., 2, 0]
    r21 = R_wb[..., 2, 1]
    r22 = R_wb[..., 2, 2]
    pitch = torch.atan2(-r20, torch.hypot(r21, r22))
    roll = torch.atan2(r21, r22)
    return torch.stack([roll, pitch], dim=-1)


# ================================================================================================
# The batched egocentric estimator.
# ================================================================================================
class BatchedEgoEstimator:
    """Stateful per-env relative-state estimator. Holds ONLY body-frame / relative quantities:
      * ``_rel_pos``   (N,G,3) smoothed per-gate drone->gate vector in the drone body frame
      * ``_rel_normal``(N,G,3) smoothed per-gate facing unit vector in the drone body frame
      * ``_t_since_fix``(N,G)  per-gate staleness clock (s) -> confidence + MASK
      * ``_vel_body``  (N,3)   body-frame velocity (IMU-primary)
      * ``_accel_bias``(N,3)   per-episode residual accel bias (the drift lever), body frame
      * ``_gyro_ar``   (N,3)   AR(1) gyro-noise state (colored)
    Course geometry (gate_pos_zup (N,G,3) OR (G,3), gate_yaw (N,G) OR (G,)) is stored per env.

    NO world position, NO world heading is ever stored. The per-gate relative vector is recovered
    from truth each step as ``R_wb^T @ (gate_pos - drone_pos)`` (a difference -> translation-invariant),
    corrupted by the measured gate-frame noise, and smoothed. Between fixes it is ego-propagated by
    the inter-frame body rotation + the body velocity (no world anchor)."""

    def __init__(self, n: int, gate_pos_zup: Tensor, gate_yaw: Tensor,
                 config: EgoEstimatorConfig | None = None,
                 params: TorchSurrogateParams | None = None,
                 device=None, dtype=None, generator=None):
        assert torch is not None, "BatchedEgoEstimator requires torch"
        self.n = int(n)
        self.cfg = config or EgoEstimatorConfig()
        self.p = params or TorchSurrogateParams()
        device = device if device is not None else gate_pos_zup.device
        dtype = dtype if dtype is not None else gate_pos_zup.dtype
        self.device, self.dtype = device, dtype
        self.gen = generator

        # course geometry -> per-env (N,G,3) / (N,G)
        gp = gate_pos_zup.to(device=device, dtype=dtype)
        gy = gate_yaw.to(device=device, dtype=dtype)
        if gp.dim() == 2:
            gp = gp.unsqueeze(0).expand(self.n, -1, -1).contiguous()
        if gy.dim() == 1:
            gy = gy.unsqueeze(0).expand(self.n, -1).contiguous()
        self.gate_pos = gp                                          # (N,G,3) Z-up
        self.gate_yaw = gy                                          # (N,G)
        self.G = gp.shape[1]
        # per-env Z-up gate frames (gate->world), columns [right, downrange(normal), up]
        self.R_world_gate = zup_gate_frame_torch(gy)               # (N,G,3,3)

        N, G = self.n, self.G
        # ---- state ----
        self._rel_pos = torch.zeros(N, G, 3, device=device, dtype=dtype)
        # (2026-07-10 cleanup) the old ``_rel_var`` variance-proxy buffer was written here + in
        # reset_idx/step but NEVER READ (confidence is purely t_since_fix-driven) -- deleted.
        self._rel_normal = torch.zeros(N, G, 3, device=device, dtype=dtype)
        self._rel_normal[..., 0] = 1.0                             # placeholder facing (body +x)
        self._normal_sig = torch.full((N, G), float(self.cfg.normal_sigma_floor_rad),
                                      device=device, dtype=dtype)
        self._t_since_fix = torch.full((N, G), 1e3, device=device, dtype=dtype)
        self._visible_area = torch.zeros(N, G, device=device, dtype=dtype)  # foreshortening ratio in [0,1]
        self._vel_body = torch.zeros(N, 3, device=device, dtype=dtype)
        self._accel_bias = torch.zeros(N, 3, device=device, dtype=dtype)
        self._gyro_ar = torch.zeros(N, 3, device=device, dtype=dtype)
        self._roll_pitch = torch.zeros(N, 2, device=device, dtype=dtype)
        self._body_rates = torch.zeros(N, 3, device=device, dtype=dtype)
        # per-episode N_eff (smoothing depth) per gate
        self._n_eff = torch.full((N, G), self.cfg.n_eff_mean(), device=device, dtype=dtype)
        self._bias = torch.zeros(N, G, 3, device=device, dtype=dtype)   # per-episode gate-frame in-plane bias

        # ---- ESTIMATOR-FAITHFUL sub-filters (2026-07-11; constructed ONLY when armed -- the
        # default 'gt'/'legacy'/'ar1' path allocates nothing and draws nothing). Both cores run in
        # the env Z-up/FLU frame with a per-env yaw-0 DATUM (the ESKF's own frame, exactly deploy):
        # gravity VECTOR [0,0,-g]; at-rest specific force [0,0,+g]. The datum-frame course tensors
        # below are estimator-INTERNAL only (deploy also holds a world-frame KF); the obs stays
        # body-frame (contract intact). ----
        self._att_emul = (self.cfg.att_model == "eskf")
        self._vel_emul = (self.cfg.vel_model == "kf")
        self._rate_sampled = (self.cfg.rate_model == "sampled")
        _g_world_zup = [0.0, 0.0, -GRAVITY_MAG]
        self._eskf = (BatchedESKFLeveler(N, _g_world_zup, device=device, dtype=dtype)
                      if self._att_emul else None)
        self._navkf = (BatchedNavKF(N, _g_world_zup, device=device, dtype=dtype)
                       if self._vel_emul else None)
        if self._vel_emul:
            # per-env datum-frame course geometry + the Rz(-yaw0) map, built at reset_idx
            self._gate_pos_datum = torch.zeros(N, G, 3, device=device, dtype=dtype)
            self._R_datum_gate = torch.zeros(N, G, 3, 3, device=device, dtype=dtype)
            self._Rz_neg_yaw0 = torch.eye(3, device=device, dtype=dtype) \
                .unsqueeze(0).repeat(N, 1, 1)
        # ---- CHOKED-LOOP dt emulation state (est_dt_ticks_hi > 1 only; see EgoEstimatorConfig) --
        # allocated ONLY when armed so the default path stays zero-alloc / zero-draw / byte-exact.
        _hi = int(self.cfg.est_dt_ticks_hi)
        self._est_hold = ((self._att_emul or self._vel_emul or self._rate_sampled) and _hi > 1)
        if self._est_hold:
            if _hi > len(MEASURED_TICK_GAP_PMF):
                raise ValueError(
                    f"est_dt_ticks_hi={_hi} exceeds the MEASURED tick-gap band "
                    f"(K={len(MEASURED_TICK_GAP_PMF)}; ego_ins_emul.MEASURED_TICK_GAP_PMF) -- a "
                    "longer band needs fresh wire measurements (leveler_bench tick_gap_report), "
                    "never an invented tail")
            pmf = torch.tensor(MEASURED_TICK_GAP_PMF[:_hi], device=device, dtype=dtype)
            self._gap_cdf = torch.cumsum(pmf / pmf.sum(), dim=0)            # (hi,) renormalized
            self._est_period = torch.ones(N, device=device, dtype=torch.long)
            self._est_phase = torch.zeros(N, device=device, dtype=torch.long)
            self._est_dt_accum = torch.zeros(N, device=device, dtype=dtype)

    # -------------------------------------------------------------------- helpers
    def _randn(self, *shape):
        return torch.randn(*shape, device=self.device, dtype=self.dtype, generator=self.gen)

    def _rand(self, *shape):
        return torch.rand(*shape, device=self.device, dtype=self.dtype, generator=self.gen)

    def _draw_gap_ticks(self, u: Tensor) -> Tensor:
        """Inverse-CDF draw of the effective loop period k (in training ticks, long (...,)) from the
        MEASURED tick-gap pmf (truncated + renormalized to est_dt_ticks_hi at __init__). u ~ U[0,1)."""
        k = torch.searchsorted(self._gap_cdf, u.to(self._gap_cdf.dtype).contiguous()) + 1
        return k.clamp(max=int(self.cfg.est_dt_ticks_hi))

    def _R_wb(self, drone_quat: Tensor) -> Tensor:
        """body->world Z-up rotation from an XYZW quat (N,4)."""
        return quat_xyzw_to_matrix_torch(drone_quat)

    def _true_rel_body(self, drone_pos: Tensor, R_wb: Tensor) -> Tensor:
        """GT drone->gate vector in the body frame per gate (N,G,3): R_wb^T @ (gate_pos - drone_pos).
        A pure difference rotated into the body frame -> invariant to a global world translation, and
        holds NO world coordinate."""
        lever = self.gate_pos - drone_pos.unsqueeze(1)                       # (N,G,3) world Z-up
        return torch.einsum("nji,ngj->ngi", R_wb, lever)                     # R_wb^T @ lever -> body

    def _R_body_gate(self, R_wb: Tensor) -> Tensor:
        """Per-gate gate->body rotation (N,G,3,3) = R_wb^T @ R_world_gate. Rotates a gate-frame noise
        vector [lat, depth, vert] into the body frame so the anisotropy lands on the right axes."""
        return torch.einsum("nji,ngjk->ngik", R_wb, self.R_world_gate)

    def _fix_sigma_gate(self, rng: Tensor) -> Tensor:
        """Per-gate anisotropic per-fix sigma in the GATE frame, columns [lat, depth, vert] (N,G,3).
        lateral = max(floor, a1*range); vertical/depth = their floors (a1=0). MEASURED, imported."""
        p = self.p
        r = torch.clamp(rng, max=p.sigma_growth_max_range_m)
        lat = torch.maximum(torch.full_like(r, p.sigma_lateral_floor), p.sigma_lateral_a1 * r)
        depth = torch.full_like(r, p.sigma_depth_floor)
        vert = torch.full_like(r, p.sigma_vertical_floor)
        return self.cfg.noise_scale * torch.stack([lat, depth, vert], dim=-1)   # (N,G,3) gate frame

    def set_noise_scale(self, value: float) -> None:
        """Live-mutate the global noise multiplier (for an in-run NOISE CURRICULUM: anneal ego_noise_scale
        0->1 within one training run). All noise-draw sites read ``self.cfg.noise_scale`` live, so replacing
        the frozen config here takes effect the next step (frozen -> dataclasses.replace, not in-place)."""
        self.cfg = replace(self.cfg, noise_scale=float(value))

    # -------------------------------------------------------------------- lifecycle
    def reset_idx(self, idx: Tensor, drone_pos: Tensor, drone_vel: Tensor, drone_quat: Tensor) -> None:
        """COLD init the given envs at the truth spawn + draw the per-episode DR.

        drone_pos/drone_vel (m,3) Z-up world; drone_quat (m,4) XYZW body->world Z-up.
        Seeds each gate's relative vector at TRUTH (a real detection would, at spawn, see the visible
        gates), the body velocity at the truth body velocity, and draws the per-episode residual accel
        bias, N_eff smoothing depth, and gate-frame in-plane bias."""
        idx = idx.reshape(-1)
        m = int(idx.numel())
        if m == 0:
            return
        # STALE-GEOMETRY FIX (same root-cause class as the coarse-map rebuild): R_world_gate is derived
        # from gate_yaw at __init__ and is otherwise NEVER rebuilt, but the env resamples gate_yaw
        # in-place every episode (peregrine_racing_ego reset -> gate_yaw[env_idx] = ...). gate_yaw is a
        # LIVE reference here (already fresh: env resamples BEFORE _reset_estimator), so refresh the reset
        # envs' gate frames from the CURRENT yaw. Feeds the downrange normal (line below) + the vel-KF
        # datum rotation (_R_datum_gate). RNG-neutral: a pure deterministic rotation of fresh yaw, no draw.
        self.R_world_gate[idx] = zup_gate_frame_torch(self.gate_yaw[idx])   # (m,G,3,3)
        R_wb = self._R_wb(drone_quat)                                       # (m,3,3)
        # per-gate truth relative vector (body frame) for these envs
        lever = self.gate_pos[idx] - drone_pos.unsqueeze(1)                 # (m,G,3)
        rel_body = torch.einsum("mji,mgj->mgi", R_wb, lever)               # (m,G,3)
        self._rel_pos[idx] = rel_body
        # facing (gate normal) in body frame = R_wb^T @ (gate downrange column, world)
        normal_world = self.R_world_gate[idx][..., :, 1]                    # (m,G,3) downrange col
        self._rel_normal[idx] = torch.einsum("mji,mgj->mgi", R_wb, normal_world)
        self._normal_sig[idx] = float(self.cfg.normal_sigma_floor_rad)
        self._t_since_fix[idx] = 1e3
        # velocity: body-frame truth v
        self._vel_body[idx] = torch.einsum("mji,mj->mi", R_wb, drone_vel)  # R_wb^T @ v_world
        self._roll_pitch[idx] = _euler_roll_pitch_from_R(R_wb)
        self._gyro_ar[idx] = 0.0

        # per-episode DR draws
        cfg = self.cfg
        # N_eff per gate ~ U[lo, hi]
        u = self._rand(m, self.G)
        self._n_eff[idx] = cfg.n_eff_lo + (cfg.n_eff_hi - cfg.n_eff_lo) * u
        # residual accel bias (the drift lever) per axis ~ U[-band, +band]
        if cfg.dr_accel_bias:
            self._accel_bias[idx] = cfg.noise_scale * cfg.accel_bias_band * (2.0 * self._rand(m, 3) - 1.0)
        else:
            self._accel_bias[idx] = 0.0
        # per-episode gate-frame bias [lat, depth, vert] (see EgoEstimatorConfig.bias_model). Applied in
        # step() as noise_gate = self._bias + sigma*randn (gate frame), so the column order is [lat, depth,
        # vert]. The 'measured' branch is a NEW code path (its own draw count); the 'legacy' else-branch
        # executes the ORIGINAL draw sequence VERBATIM (mag rand THEN sign rand) so every existing config
        # stays byte-identical -- do NOT add a draw before the else-branch or the whole stream shifts.
        if cfg.inject_bias:
            if cfg.bias_model == "measured":
                # MEASURED (sigma.json): INDEPENDENT per-axis magnitude ~U[0, |band|] with the measured
                # sign. band_axis * U[0,1) already carries the correct sign AND magnitude band (band<0 ->
                # values in (band,0]; band>0 -> [0,band)). 3 independent draws; scaled by noise_scale.
                lat = cfg.bias_band_lat * self._rand(m, self.G)
                vert = cfg.bias_band_vert * self._rand(m, self.G)
                depth = cfg.bias_band_depth * self._rand(m, self.G)
                self._bias[idx] = cfg.noise_scale * torch.stack([lat, depth, vert], dim=-1)  # [lat,depth,vert]
            else:
                # LEGACY (default, BYTE-IDENTICAL): ONE scalar mag ~U[0, bias_mag_hi], random sign, applied
                # to BOTH lat and vert (perfectly correlated), depth 0. EXACT original draw order preserved.
                mag = cfg.noise_scale * cfg.bias_mag_hi * self._rand(m, self.G)
                sign = torch.where(self._rand(m, self.G) < 0.5, torch.ones(m, self.G, device=self.device, dtype=self.dtype),
                                   -torch.ones(m, self.G, device=self.device, dtype=self.dtype))
                b = sign * mag                                                  # (m,G)
                self._bias[idx] = torch.stack([b, torch.zeros_like(b), b], dim=-1)   # [lat, depth=0, vert]
        else:
            self._bias[idx] = 0.0

        # ---- ESTIMATOR-FAITHFUL cold-init (branch-gated). Placed AFTER every legacy draw so the
        # legacy sequence stays byte-verbatim. Draw order on the armed path (its own documented
        # contract, pinned by the draw-count tests): [est_hold] ONE rand(m) for the initial loop
        # period; the filter seeds below draw NOTHING. ----
        if self._est_hold:
            # initial per-env effective loop period k ~ the MEASURED tick-gap pmf (renewal seed;
            # subsequent periods redraw at each advance inside step()).
            self._est_period[idx] = self._draw_gap_ticks(self._rand(m))
            self._est_phase[idx] = 0
            self._est_dt_accum[idx] = 0.0
        if self._att_emul:
            # deploy-faithful takeoff state: the wire level-seeds on the pad and converges over
            # ~100 s idle, so at handover the leveler is ~truth roll/pitch with an arbitrary-0 yaw
            # datum and the PAD-CONVERGED covariance (ego_ins_emul.eskf_p_launch, measured).
            rp = _euler_roll_pitch_from_R(R_wb)                              # (m,2) truth roll/pitch
            q0 = quat_from_roll_pitch_yaw_zyx(rp[:, 0], rp[:, 1], torch.zeros_like(rp[:, 0]))
            self._eskf.reset_idx(idx, q0)
        if self._vel_emul:
            # yaw-datum world frame: rotate by Rz(-yaw0) (yaw0 = spawn true yaw -- exact under the
            # measured-zero gyro noise/bias; if a future sim adds gyro noise the datum drifts and
            # this construction needs the drifting q_emul yaw instead -- flagged risk), origin at
            # the spawn (deploy: nav world origin = boot position; KF pos seeds 0 / std 5).
            yaw0 = torch.atan2(R_wb[:, 1, 0], R_wb[:, 0, 0])                 # (m,)
            c, s = torch.cos(yaw0), torch.sin(yaw0)
            z, o = torch.zeros_like(c), torch.ones_like(c)
            Rz = torch.stack([                                               # Rz(-yaw0)
                torch.stack([c, s, z], dim=-1),
                torch.stack([-s, c, z], dim=-1),
                torch.stack([z, z, o], dim=-1),
            ], dim=-2)                                                       # (m,3,3)
            self._Rz_neg_yaw0[idx] = Rz
            lever0 = self.gate_pos[idx] - drone_pos.unsqueeze(1)             # (m,G,3)
            self._gate_pos_datum[idx] = torch.einsum("mij,mgj->mgi", Rz, lever0)
            self._R_datum_gate[idx] = torch.einsum("mij,mgjk->mgik", Rz, self.R_world_gate[idx])
            vel0 = torch.einsum("mij,mj->mi", Rz, drone_vel)                 # datum-frame spawn vel
            # deploy seeds pos=origin / vel=0 with the drone PARKED (== exact truth); training
            # spawns mid-takeoff, so the datum-frame truth IS the deploy-faithful takeoff seed.
            self._navkf.reset_idx(idx, torch.zeros_like(vel0), vel0)

    # -------------------------------------------------------------------- the step
    def step(self, drone_pos: Tensor, drone_vel: Tensor, drone_quat: Tensor,
             body_rates: Tensor, dt: float, detectable: Tensor | None = None,
             prev_quat: Tensor | None = None, apparent_area: Tensor | None = None,
             *, blur_extra_miss: Tensor | None = None,
             sf_body: Tensor | None = None, gyro_sample: Tensor | None = None) -> EgoEstimate:
        """Advance ALL envs one control step and return the current estimate.

        Inputs (Z-up / FLU truth, available in training):
          drone_pos  (N,3)   world Z-up position           (used ONLY inside a difference -> no world state)
          drone_vel  (N,3)   world Z-up velocity           (truth the velocity state is corrected toward)
          drone_quat (N,4)   XYZW body->world Z-up          (attitude; roll/pitch leveled, NO yaw kept)
          body_rates (N,3)   FLU body rates (gyro truth)
          dt         float   control step (s)
          detectable (N,G)   optional precomputed visibility mask; if None, computed via gate_detectable
          prev_quat  (N,4)   optional previous attitude for the ego-propagation rotation (defaults to
                             the current attitude -> a small-rotation ego-propagation)
          apparent_area (N,G) optional precomputed NORMALIZED projected inner-opening area (in [0,1],
                             square-on==1) for the visible_area channel -- the env passes it computed with
                             the EMULATED (flipped) camera so the obs matches what the detector sees; if
                             None it is computed here from the raw drone attitude (gate_apparent_area).
          blur_extra_miss (N,G) optional MOTION-BLUR soft-band extra miss probability (PERCEPTION-HONESTY
                             package, 2026-07-10; from gate_visibility.blur_extra_miss_prob). Folded into
                             the EXISTING stochastic miss draw so stochasticity stays where it already
                             lives (fix acceptance); None (default) == byte-identical legacy behaviour.
                             DELIBERATELY NOT multiplied by noise_scale: blur is CAMERA PHYSICS, not
                             estimator corruption -- the noise-0 calibration boot stage MUST still see
                             blur, or the boot re-learns the spin-scan gait the package exists to kill.
          sf_body    (N,3)   ESTIMATOR-FAITHFUL package: the LATEST-sample body specific force (FLU,
                             m/s^2, from the plant's capture_specific_force hook -- the last substep's
                             sample, exactly the wire's newest-IMU-sample semantics). REQUIRED when
                             att_model=='eskf' or vel_model=='kf'; None (default) == legacy paths only.
          gyro_sample (N,3)  the LATEST-sample body rates (FLU); defaults to ``body_rates`` (with
                             n_substeps>1 the env passes truth self._w == the final substep's rate ==
                             the wire's newest-sample gyro, so the default is already correct).

        Returns an ``EgoEstimate``. NO world position / heading anywhere in the returned tensors or state.
        """
        dt = float(dt)
        N, G = self.n, self.G
        R_wb = self._R_wb(drone_quat)                                       # (N,3,3)

        # ---- CHOKED-LOOP dt emulation (est_dt_ticks_hi > 1 only; see EgoEstimatorConfig): the
        # faithful channels advance every k-th tick with the ACCUMULATED dt and hold (frozen)
        # in between; k redraws from the MEASURED tick-gap pmf at each advance (renewal). Draw
        # discipline: exactly ONE rand(N) per step when armed (drawn for all, consumed by the
        # advancing envs), zero otherwise. ``est_dt`` <= 0 is the filters' documented no-op. ----
        if self._est_hold:
            u_gap = self._rand(N)                                           # fixed draw count
            self._est_phase = self._est_phase + 1
            self._est_dt_accum = self._est_dt_accum + dt
            adv = self._est_phase >= self._est_period                       # (N,) bool
            est_dt = torch.where(adv, self._est_dt_accum,
                                 torch.zeros_like(self._est_dt_accum))
            self._est_phase = torch.where(adv, torch.zeros_like(self._est_phase), self._est_phase)
            self._est_dt_accum = torch.where(adv, torch.zeros_like(self._est_dt_accum),
                                             self._est_dt_accum)
            self._est_period = torch.where(adv, self._draw_gap_ticks(u_gap), self._est_period)
        else:
            adv = None                                                      # every tick advances
            est_dt = dt

        # ---- rates: sampled raw gyro (faithful) OR legacy AR(1) colored noise ----
        gyro_used = gyro_sample if gyro_sample is not None else body_rates
        if self._rate_sampled:
            # FAITHFUL path: deploy obs[5:8] = the RAW sign-corrected latest gyro sample
            # (fly_rl.py:1810 s.gyro_body, NOT the AHRS bias-corrected rate). MEASURED-ZERO sensor
            # noise (a5 pad-idle, imu_foundation.py) -> nothing added, NO RNG draw on this branch
            # (the legacy _randn below is NOT executed -- new path owns its own draw count).
            # Under the choked-loop hold the sample refreshes only when the emulated loop runs
            # (the wire delivers a new gyro obs only at loop ticks).
            self._body_rates = gyro_used if adv is None else \
                torch.where(adv.unsqueeze(-1), gyro_used, self._body_rates)
        else:
            # LEGACY (default; BYTE-IDENTICAL draw sequence): AR(1) colored gyro noise
            # e_t = rho*e_{t-1} + sqrt(1-rho^2)*sigma*w_t (stationary marginal sigma)
            rho = self.cfg.gyro_ar1_rho
            innov = ((1.0 - rho * rho) ** 0.5) * self.cfg.gyro_sigma * self.cfg.noise_scale * self._randn(N, 3)
            self._gyro_ar = rho * self._gyro_ar + innov
            self._body_rates = body_rates + self._gyro_ar

        # ---- attitude: emulated deploy ESKF leveler (faithful) OR perfect truth extraction ----
        R_datum = None                       # body->datum rotation the KF/velocity path shares
        if self._att_emul:
            if sf_body is None:
                raise RuntimeError("att_model='eskf' requires sf_body= (arm "
                                   "++dynamics.capture_specific_force=true and wire the env)")
            # ONE step per control tick on the LATEST sample -- the verified deploy rate contract
            # (navigator.py:585-607; a 143 Hz substep loop would be LESS faithful AND ~5x the cost).
            # est_dt carries the choked-loop hold (0 on held ticks = the leveler's no-op branch;
            # the accumulated gap on an advance -- exactly the wire's step-once-per-loop-tick).
            self._eskf.step(gyro_used, sf_body, est_dt)
            R_datum = self._eskf.R_wb()                                     # body->datum-world
            # THE obs-parity point: the SAME extraction the legacy path / deploy roll_pitch_zup
            # uses, applied to the EMULATED (possibly lying) attitude. Yaw datum never reaches obs.
            self._roll_pitch = _euler_roll_pitch_from_R(R_datum)
        else:
            self._roll_pitch = _euler_roll_pitch_from_R(R_wb)

        # ---- velocity: emulated deploy KF strapdown (faithful) OR legacy bias-drift ----
        if self._vel_emul:
            if sf_body is None:
                raise RuntimeError("vel_model='kf' requires sf_body= (arm "
                                   "++dynamics.capture_specific_force=true and wire the env)")
            if R_datum is None:
                # ablation combo (vel='kf', att='gt'): project through the TRUTH attitude mapped
                # into the datum frame (no attitude corruption -- that is the ablation's point).
                R_datum = torch.einsum("nij,njk->nik", self._Rz_neg_yaw0, R_wb)
            # strapdown predict through the EMULATED attitude -- the load-bearing error coupling
            # (wrong gravity cancellation; deploy state_estimator.predict). Fix updates + the
            # world->body projection happen below, after the fix stream is computed. est_dt =
            # the choked-loop hold (0 on held ticks == the KF's dt<=0 predict-drop branch).
            self._navkf.predict(sf_body, R_datum, est_dt)
        else:
            # LEGACY (default; BYTE-IDENTICAL): body-frame velocity integrates a residual accel
            # bias (the drift lever) + tiny white noise (per-episode residual; over Delta-t since
            # the last fix this integrates to bias*Delta_t -- the dominant error).
            white = self.cfg.accel_white_sigma * self.cfg.noise_scale * self._randn(N, 3)
            self._vel_body = self._vel_body + dt * (self._accel_bias + white)

        # ---- ego-propagate each gate's relative vector through the (possible) no-fix gap ----
        # rel_pos_body_new = R_prev_from_cur @ rel_pos_body_old  - v_body*dt.
        # The rotation between frames: if the body rotated by R_cur_from_prev = R_wb_cur^T @ R_wb_prev
        # in the WORLD, a fixed world vector expressed in body rotates by its inverse; we approximate the
        # small inter-frame rotation from the (noisy) body rates over dt (a body-only quantity -- no
        # world anchor). Translation: the drone moved by v_body*dt in its own frame, so the gate moves
        # -v_body*dt in body coords. This keeps the relative vector coherent through short gaps.
        dR = _small_rotation_body(self._body_rates, dt, self.device, self.dtype)   # (N,3,3) cur<-prev
        # apply to each gate: rotate then translate
        self._rel_pos = torch.einsum("nij,ngj->ngi", dR, self._rel_pos) \
            - (self._vel_body * dt).unsqueeze(1)
        self._rel_normal = torch.einsum("nij,ngj->ngi", dR, self._rel_normal)
        self._t_since_fix = self._t_since_fix + max(dt, 0.0)

        # ---- vision: per-gate detectability -> fresh noisy fix (propagate-then-mask) ----
        if detectable is None:
            if _gate_detectable is None:                                    # pragma: no cover
                raise RuntimeError("gate_visibility unavailable; pass detectable= to step()")
            # STANDALONE-ONLY fallback (no camera flip): this internal path uses the RAW drone attitude,
            # while the env path passes ``detectable`` computed with the EMULATED (pi-flipped) camera
            # (peregrine_racing_ego._cam_R_wb). Deliberate but subtle -- tests/standalone drivers only;
            # the env NEVER exercises this branch. (It also applies NO motion-blur gate.)
            detectable, _ = _gate_detectable(drone_pos, drone_quat, self.gate_pos, self.gate_yaw,
                                             far_cap_m=self.cfg.far_cap_m, is_quat=True)
        detectable = detectable.to(torch.bool)

        rel_true = self._true_rel_body(drone_pos, R_wb)                     # (N,G,3) body-frame truth
        rng = torch.linalg.norm(rel_true, dim=-1)                          # (N,G) range
        R_bg = self._R_body_gate(R_wb)                                     # (N,G,3,3) gate->body

        # anisotropic gate-frame noise [lat, depth, vert]
        sigma_gate = self._fix_sigma_gate(rng)                             # (N,G,3) (already noise_scale'd)
        noise_gate = self._bias + sigma_gate * self._randn(N, G, 3)        # (N,G,3) gate frame
        fix_body = rel_true + torch.einsum("ngij,ngj->ngi", R_bg, noise_gate)   # (N,G,3) body-frame fix

        # RANDOM-IN-FRAME TELEPORT OUTLIER: replace the fix with a wildly wrong relative position.
        teleport = (self._rand(N, G) < self.cfg.teleport_prob * self.cfg.noise_scale) & detectable
        wild = rel_true + self.cfg.teleport_scale_m * self._randn(N, G, 3)
        fix_body = torch.where(teleport.unsqueeze(-1), wild, fix_body)

        # stochastic per-frame MISS even when visible, and the visibility gate: accept iff visible,
        # not missed. A non-detectable gate gets NO fresh fix (propagate-then-mask).
        # MOTION-BLUR soft band (perception-honesty package): ``blur_extra_miss`` raises the miss
        # probability inside the [lo, hi) rate band. NOT scaled by noise_scale (camera physics, must
        # survive the noise-0 calibration boot -- see the step() docstring). None -> byte-identical.
        if blur_extra_miss is not None:
            miss_thresh = (self.cfg.miss_prob * self.cfg.noise_scale
                           + blur_extra_miss).clamp(0.0, 1.0)              # (N,G)
        else:
            miss_thresh = self.cfg.miss_prob * self.cfg.noise_scale        # scalar (legacy expression)
        missed = self._rand(N, G) < miss_thresh
        accepted = detectable & (~missed)                                  # (N,G) bool

        # ---- KF-style smoothing update per accepted gate (scalar-gain, matched to N_eff) ----
        # Realised in-plane sigma ~ per-fix_sigma / sqrt(N_eff): a steady-state low-pass with gain
        # K = 1/N_eff reproduces variance (per-fix_var)/(2*N_eff - 1) ~ per-fix_var/N_eff for large N_eff.
        # We use K = 1/N_eff so the smoothed estimate averages ~N_eff fixes (the measured smoothing).
        # RE-ACQUISITION SNAP: a gate that was MASKED (stale past the horizon) has a drifted prior, so
        # its first fresh fix uses K=1 (discard the stale prior) instead of the slow blend -- otherwise
        # confidence would read fresh (1.0) while rel_pos still carried stale-propagation error. A KF
        # would likewise apply near-unity gain when the covariance is inflated after a long gap.
        reacq = self._t_since_fix > self.cfg.stale_horizon_s               # (N,G) was masked before this fix
        K = (1.0 / self._n_eff).clamp(0.0, 1.0)                            # (N,G)
        K = torch.where(reacq, torch.ones_like(K), K)
        Kf = torch.where(accepted, K, torch.zeros_like(K)).unsqueeze(-1)   # (N,G,1)
        self._rel_pos = self._rel_pos + Kf * (fix_body - self._rel_pos)
        # (2026-07-10 cleanup) the old variance-proxy update lived here: it wrote ``_rel_var`` from
        # latency_cov_inflate * per_fix_var / n_eff -- but the buffer was NEVER READ (confidence is purely
        # t_since_fix-driven, see estimate()), so the whole block (and the latency_cov_inflate knob it
        # consumed) was INERT. Deleted; behavior-neutral by construction (no RNG draws were involved).

        # ---- gate normal/yaw: the weakest DOF -- own range-collapsing angular noise + flip ----
        cfg = self.cfg
        normal_sig = cfg.noise_scale * (cfg.normal_sigma_floor_rad
                      + cfg.normal_sigma_a1_rad_per_m * torch.clamp(rng, max=self.p.sigma_growth_max_range_m))
        true_normal_body = torch.einsum("nji,ngj->ngi", R_wb, self.R_world_gate[..., :, 1])   # (N,G,3)
        # perturb the facing by a small random rotation of angular std normal_sig about a random axis,
        # plus an occasional pi flip (two-fold ambiguity).
        ang = normal_sig * self._randn(N, G)                              # (N,G) perturbation angle
        axis = self._randn(N, G, 3)
        axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True).clamp(min=1e-9)
        dR_n = _axis_angle_rotation(axis, ang)                            # (N,G,3,3)
        noisy_normal = torch.einsum("ngij,ngj->ngi", dR_n, true_normal_body)
        flip = (self._rand(N, G) < cfg.normal_flip_prob * cfg.noise_scale) & accepted
        noisy_normal = torch.where(flip.unsqueeze(-1), -noisy_normal, noisy_normal)
        acc3 = accepted.unsqueeze(-1)
        self._rel_normal = torch.where(acc3, noisy_normal, self._rel_normal)
        self._normal_sig = torch.where(accepted, normal_sig, self._normal_sig)

        # ---- visible-area (APPARENT PROJECTED OPENING area) channel: the vision-faithful "how square-on"
        # cue (Fengyou 2026-07-07 recalibration). ratio in [0,1], 1 == perfectly square-on -- the
        # NORMALIZED image-space area of the projected inner opening (what a corner detector actually
        # reports), NOT the old |cos(view_ray, gate_normal)| proxy (which diverges from the true projected
        # area close-in, where perspective matters, and would not match the deployed detector's output).
        # The env passes ``apparent_area`` computed with the EMULATED (flipped) camera so the obs matches
        # the camera the detector sees through (and equals the reward's privileged area); a standalone
        # caller falls back to the raw attitude. Range-normalized -> invariant to distance. + modest
        # placeholder noise, refreshed only on an ACCEPTED fix, masked to 0 in estimate() when stale.
        if apparent_area is None:
            if _gate_apparent_area is None:                               # pragma: no cover
                raise RuntimeError("gate_visibility.gate_apparent_area unavailable; pass apparent_area=")
            apparent_area = _gate_apparent_area(drone_pos, drone_quat, self.gate_pos, self.gate_yaw,
                                                is_quat=True)
        area_true = apparent_area                                          # (N,G) in [0,1], square-on==1
        area_noisy = (area_true + self.cfg.visible_area_sigma * self.cfg.noise_scale * self._randn(N, G)).clamp(0.0, 1.0)
        self._visible_area = torch.where(accepted, area_noisy, self._visible_area)

        # ---- velocity correction: INDIRECT only (NO direct vision-velocity observation) ----
        if self._vel_emul:
            # FAITHFUL path: position fixes into the translated KF -- velocity corrected ONLY via
            # the pos/vel cross-covariance (the deploy mechanism, not a hand gain). Measurement
            # model = PnP-through-estimated-attitude: z = gate_pos_datum - R_datum @ fix_body,
            # exactly how deploy's gate_pose_to_world_position inherits the attitude error. The
            # fix covariance is the training-measured per-fix gate-frame sigma (which IS the
            # measured end-to-end fix-error contract) rotated to the datum frame, each axis floored
            # at the deploy FIX_COV_FLOOR_STD=0.40 m -- an explicit SURROGATE for the deploy
            # 3-channel chain (absolute + gate-rel in-plane + range); the full-chain translation is
            # the priced upgrade if the velocity benchmark misses. Teleport-class outliers are
            # rejected by the KF's chi2(3) fix gate (deploy md<=16.27), like the wire chain.
            # choked-loop hold: the wire processes fixes only when the nav loop runs -- held
            # ticks contribute NO KF update (their vision draw stream above is unchanged).
            acc_kf = accepted if adv is None else (accepted & adv.unsqueeze(-1))
            z_datum = self._gate_pos_datum - torch.einsum("nij,ngj->ngi", R_datum, fix_body)
            # deploy floor FORM (reviewer-caught 2026-07-11): localization.py:56 ADDS the floor IN
            # QUADRATURE ("added in quadrature to every vision world-fix covariance"), not a clamp
            # -- matched exactly: sig^2 = sigma^2 + FIX_COV_FLOOR_STD^2. (The other direction of
            # the deviation -- deploy's gate-relative in-plane channel REMOVES the floor in-plane
            # -- stays a documented conservative cut of the single-fix surrogate; see above.)
            sig = torch.sqrt(sigma_gate * sigma_gate
                             + KF_FIX_COV_FLOOR_STD ** 2)                   # (N,G,3) gate frame
            cov = torch.einsum("ngij,ngj,ngkj->ngik", self._R_datum_gate, sig * sig,
                               self._R_datum_gate)                          # R diag(sig^2) R^T
            for gi in range(G):                                             # sequential channel
                self._navkf.update_position(z_datum[:, gi], cov[:, gi], acc_kf[:, gi])
            # obs velocity = KF world velocity projected world->body through the SAME (emulated)
            # attitude (deploy ego_obs.py:305-321) -- the second attitude-error injection. The yaw
            # datum cancels to first order (KF velocity and projection share it; ADJUDICATED).
            self._vel_body = torch.einsum("nji,nj->ni", R_datum, self._navkf.velocity)
        else:
            # LEGACY (default; BYTE-IDENTICAL): the VQ2 wire carries no velocity measurement; the
            # deploy KF corrects velocity SOLELY through gate-relative POSITION fixes via the
            # pos-vel cross-covariance (vision commander 2026-07-06). Modeled as a small PARTIAL
            # gain toward truth body velocity at an accepted fix -- NOT a snap. Velocity stays
            # IMU-primary; the fix stream only weakly/gradually bounds the drift (and given the
            # very quiet IMU the residual is small). Body-frame v only; NO world velocity.
            any_fix = accepted.any(dim=1)                                      # (N,)
            vel_body_true = torch.einsum("nji,nj->ni", R_wb, drone_vel)        # R_wb^T @ v (body)
            gv = self.cfg.vel_correct_gain
            self._vel_body = torch.where(any_fix.unsqueeze(-1),
                                         self._vel_body + gv * (vel_body_true - self._vel_body),
                                         self._vel_body)

        # ---- staleness clock reset on a fix ----
        self._t_since_fix = torch.where(accepted, torch.zeros_like(self._t_since_fix),
                                        self._t_since_fix)

        return self.estimate()

    # -------------------------------------------------------------------- readout
    def estimate(self) -> EgoEstimate:
        """Assemble the current outputs. Confidence decays with staleness and MASKS (0) past the
        horizon; a gate never propagated past ``stale_horizon_s`` stays frozen-then-masked, never
        frozen-and-drifted forever."""
        cfg = self.cfg
        # staleness confidence: 1 at a fresh fix, linearly -> 0 at the horizon, then hard 0 (MASK).
        age = self._t_since_fix / max(cfg.stale_horizon_s, 1e-9)
        conf = torch.clamp(1.0 - age, 0.0, 1.0)                            # (N,G)
        masked = self._t_since_fix > cfg.stale_horizon_s
        conf = torch.where(masked, torch.zeros_like(conf), conf)
        # normal confidence = clip(ref / sigma_hat, 0, 1), also masked when stale
        ncf = torch.clamp(cfg.normal_conf_ref_rad / self._normal_sig.clamp(min=1e-9), 0.0, 1.0)
        ncf = torch.where(masked, torch.zeros_like(ncf), ncf)
        # visible-area: masked (0) when stale, exactly like confidence (an unseen gate has no area cue)
        vis_area = torch.where(masked, torch.zeros_like(self._visible_area), self._visible_area)

        # renormalise the facing to a unit vector; body-frame yaw of the facing
        nrm = self._rel_normal / torch.linalg.norm(self._rel_normal, dim=-1, keepdim=True).clamp(min=1e-9)
        rel_yaw = torch.atan2(nrm[..., 1], nrm[..., 0])                    # (N,G) body-frame facing angle

        return EgoEstimate(
            rel_pos=self._rel_pos.clone(),
            rel_normal=nrm,
            rel_yaw=rel_yaw,
            confidence=conf,
            normal_conf=ncf,
            velocity=self._vel_body.clone(),
            roll_pitch=self._roll_pitch.clone(),
            body_rates=self._body_rates.clone(),
            visible_area=vis_area,
        )


# ================================================================================================
# small rotation helpers (body-frame only -- no world anchor).
# ================================================================================================
def _small_rotation_body(body_rates: Tensor, dt: float, device, dtype) -> Tensor:
    """Rotation applied to body-frame vectors as the frame advances by the body rates over dt.

    A fixed WORLD vector expressed in the body frame transforms by R_curbody_from_prevbody^{-1}. For a
    body rotating at omega (body frame), the frame advances by exp([omega]x * dt); a world-fixed vector
    in body coords therefore rotates by exp(-[omega]x * dt). Returns (N,3,3). Pure body quantity -- omega
    is the (noisy) gyro; NO world state enters."""
    return _axis_angle_rotation_vec(-body_rates * dt)


def _axis_angle_rotation_vec(rotvec: Tensor) -> Tensor:
    """Rotation matrix (…,3,3) from a rotation vector (…,3) (Rodrigues, small-angle safe)."""
    theta = torch.linalg.norm(rotvec, dim=-1, keepdim=True)               # (...,1)
    small = theta < 1e-8
    theta_safe = torch.where(small, torch.ones_like(theta), theta)
    axis = rotvec / theta_safe
    return _axis_angle_rotation(axis, theta.squeeze(-1))


def _axis_angle_rotation(axis: Tensor, angle: Tensor) -> Tensor:
    """Rodrigues rotation matrix (...,3,3) from a unit ``axis`` (...,3) and ``angle`` (...,)."""
    ax, ay, az = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = torch.zeros_like(ax)
    Kx = torch.stack([
        torch.stack([zero, -az, ay], dim=-1),
        torch.stack([az, zero, -ax], dim=-1),
        torch.stack([-ay, ax, zero], dim=-1),
    ], dim=-2)                                                            # (...,3,3) skew
    c = torch.cos(angle)[..., None, None]
    s = torch.sin(angle)[..., None, None]
    eye = torch.eye(3, device=axis.device, dtype=axis.dtype).expand(Kx.shape)
    return eye + s * Kx + (1.0 - c) * torch.einsum("...ij,...jk->...ik", Kx, Kx)
