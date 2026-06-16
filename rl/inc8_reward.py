"""rl/inc8_reward.py -- the inc8 reward DELTAS over the inc7 spine (torch-pure, laptop-testable).

The inc8 env (rl/peregrine_racing_inc8.py) keeps the ENTIRE inc7 reward/termination contract frozen
(R2 passage, T1/T2/T3 terminals, R3 time, R4 60-deg tilt cone, R5 dact, R6 rate, R7 corner, T4
finish-time) and adds only these case-C deployability terms. All are pure functions of state/event
tensors so they unit-test on the laptop without diffaero; the env calls them under no_grad and the
GT-anchor reads TRUTH (the actor sees the noisy KF obs -- the asymmetric-critic anti-damping setup).

SHARED SPINE (all arms):
  R1'  arc-length progress over Gamma   rw_progress * (s_curr - s_prev)        [replaces R1-to-centre]
  GT   estimator-error anchor          -rw_estimerr * |KF_pos - truth|_inplane  [FLAT, truth-seen]
  CS   confidence/staleness shaping    +rw_conf_shape * anneal * (c_in - age)  [dense, ANNEALED late]
ARM-SPECIFIC R5' (the COWORK-3 2-axis terminal-lock perception reward; default A):
  A  r_perc = rw_perc * w_term(d) * v(alpha,beta) * max(ds, 0)   (terminal-locked, 2-axis quartic)
  B  same with w_term == 1                                       (FLAT weight)
  C  r_perc == 0                                                 (implicit-only)

R5' SHAPE (perception-reward.md / SWIFT-Geles exp(-delta^4) generalised to Qin alpha/beta):
  gate centre in camera frame [X,Y,Z] = t_cam (the surrogate geometry already computes it)
  alpha = atan2(X, Z) (deg), beta = atan2(Y, sqrt(X^2+Z^2)) (deg)
  v(alpha,beta) = exp[ -((alpha/sigma_a)^4 + (beta/sigma_b)^4) ]   sigma_a~45, sigma_b~29.5 (FoV half-angles)
  w_term(d) = w0 + (1-w0) * clip((d_acq - d)/(d_acq - d_lock), 0, 1)   d_lock~5 m (full lock), d_acq~inter-gate
  progress-gating * max(ds, 0): payable only while advancing along Gamma (kills the slow-to-look loiter).
"""
from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import Tensor
except Exception:                       # pragma: no cover
    torch = None
    Tensor = "Tensor"                   # type: ignore


@dataclass
class Inc8RewardWeights:
    """inc8 reward-delta weights (cfg-overridable as ``+env.rw_<field>=...``; the inc7 RewardWeights
    are untouched). rw_perc ~ 5% of rw_progress (SWIFT/Geles/Song); GT-anchor is the primary
    estimator-error lever (flat); confidence-shaping is small + annealed."""
    progress: float = 10.0          # R1' arc-length progress along Gamma (== inc7 rw_progress)
    estimerr: float = 2.0           # GT in-plane estimator-error anchor (flat, truth-seen)
    conf_shape: float = 0.05        # confidence/staleness dense-shaping (annealed late)
    perc: float = 0.5               # R5' perception weight (~5% of progress)
    fix_bonus: float = 0.0          # direct per-ACCEPTED-fix bonus, progress-gated (un-gameable);
                                    # 0.0 == no term (byte-identical). Earnable ONLY by getting a real
                                    # fix => pointing in the surrogate accept band; not gameable by
                                    # terminal pointing or a coasting KF. (iteration-4 fix-driven lever)
    # R5' shape
    perc_sigma_a_deg: float = 45.0  # azimuth visibility scale (~H half-FoV)
    perc_sigma_b_deg: float = 29.5  # elevation visibility scale (~V half-FoV)
    # R5' terminal-weight is a BAND-PASS in range (mirrors the surrogate accept band-pass): w_term peaks
    # (->1) in the fixable [r_lo, r_hi] band and is SUPPRESSED to ~w0 BOTH in the easy terminal zone
    # (<r_lo -- where the prior low-pass plateau let the converged policy point-blank for full reward yet
    # get 0 fixes; convergence run 3273701) AND far (>r_hi). r_lo == the pointed accurate floor ~12 m
    # (accept-geometry-2026-06-15) == the surrogate accept_rlo, so reward and fix-obtainability align.
    perc_r_lo: float = 12.0         # band lower edge (pointed accurate floor; == surrogate accept_rlo)
    perc_r_hi: float = 28.0         # band upper edge (== surrogate accept_rhi ~28)
    perc_w_lo: float = 1.5          # lower-edge sigmoid width (m)
    perc_w_hi: float = 1.5          # upper-edge sigmoid width (m)
    perc_w0: float = 0.05           # band-pass floor (terminal + far strongly suppressed)
    # confidence-shaping anneal (in CONTROL STEPS; the env tracks a global step counter)
    conf_anneal_warmup_steps: float = 0.0
    conf_anneal_ramp_steps: float = 1.0
    # dense terminal-sigma_p0 CENTERING reward (architecture pivot S1): a RANGE-WEIGHTED GT in-plane
    # estimator-error penalty that ramps UP near the gate crossing (range->0), where err_ip == the
    # terminal centering miss sigma_p0. The flat gt_estimerr_anchor is coast-satisfied; this one is not
    # (low near-gate err_ip needs an accurate band fix => pointing). centering=0 => zero term (OFF).
    centering: float = 0.0          # rw for the dense terminal-centering penalty (0 == off)
    centering_r_near: float = 8.0   # range (m) where the near-weight crosses 0.5 (peaks at crossing)
    centering_w: float = 3.0        # near-weight sigmoid width (m)


def arc_progress_reward(s_curr: Tensor, s_prev: Tensor, rw_progress: float) -> Tensor:
    """R1': rw_progress * (s_curr - s_prev) -- arc-length advanced along Gamma this step (m). Closes
    the corner-cut failure mode M-2 by construction (you only earn reward for advancing along a line
    that already clears the gate). Can be negative (the drone backed up)."""
    return rw_progress * (s_curr - s_prev)


def gt_estimerr_anchor(err_inplane_m: Tensor, rw_estimerr: float) -> Tensor:
    """GT anchor: -rw_estimerr * |KF_pos - truth|_inplane (gate-frame). Reward sees TRUTH; the actor
    sees the noisy KF obs. FLAT weight, no proximity schedule -- the truth-seeing critic (get_state
    36-dim) is what makes calibrated caution emerge WITHOUT a hand-coded damping term."""
    return -rw_estimerr * err_inplane_m


def confidence_anneal(global_step: float, w: Inc8RewardWeights) -> float:
    """Linear ramp in [0,1] over control steps: 0 until warmup, then ramps to 1 over ramp_steps. Lets
    the base racing task settle before the dense confidence shaping turns on (avoids the early
    slow-to-look local optimum; perception-reward.md 2(b))."""
    x = (global_step - w.conf_anneal_warmup_steps) / max(w.conf_anneal_ramp_steps, 1.0)
    return float(min(max(x, 0.0), 1.0))


def confidence_shaping_reward(triple: Tensor, rw_conf_shape: float, anneal: float) -> Tensor:
    """CS: rw_conf_shape * anneal * (c_inplane - age_norm) from obs[17:20]. Rewards keeping the
    in-plane confidence high + the fix fresh -- achievable ONLY by pointing the camera (-> fixes ->
    low KF sigma), so it is aligned with the GT anchor, not a damping term. Small + annealed late."""
    c_inplane, age_norm = triple[..., 0], triple[..., 2]
    return rw_conf_shape * anneal * (c_inplane - age_norm)


def visibility_2axis(t_cam: Tensor, sigma_a_deg: float, sigma_b_deg: float) -> Tensor:
    """Separable quartic visibility v(alpha,beta) = exp[-((alpha/sa)^4 + (beta/sb)^4)] in [0,1] -- the
    SWIFT/Geles exp(-delta^4) shape generalised to Qin's independent azimuth/elevation pair (so the
    narrower vertical FoV + the 20-deg pitch coupling are penalised more tightly than azimuth). Flat
    plateau while the gate is comfortably in frame, sharp cliff near either edge (zero gradient when
    centred -> nothing to chatter against)."""
    tx, ty, tz = t_cam[..., 0], t_cam[..., 1], t_cam[..., 2]
    deg = 180.0 / torch.pi
    alpha = torch.atan2(tx, tz) * deg
    beta = torch.atan2(ty, torch.sqrt(tx * tx + tz * tz)) * deg
    return torch.exp(-((alpha / sigma_a_deg) ** 4 + (beta / sigma_b_deg) ** 4))


def terminal_weight(range_m: Tensor, r_lo: float, r_hi: float, w_lo: float, w_hi: float,
                    w0: float) -> Tensor:
    """BAND-PASS w_term(r) = w0 + (1-w0)*sigmoid((r-r_lo)/w_lo)*sigmoid((r_hi-r)/w_hi): peaks (->1) in
    the fixable [r_lo, r_hi] band, decays to the floor w0 BOTH near (<r_lo) and far (>r_hi). Mirrors the
    surrogate accept band-pass so the reward gradient points the camera where fixes are ACCEPTED. The
    two-sided suppression is the whole point: the prior one-sided low-pass plateau (w_term=1.0 for ALL
    r <= d_lock) let the CONVERGED policy earn full reward by easy terminal <5 m pointing and get 0 fixes
    (convergence run 3273701: lockband_pointing=0 all 4000 updates). w_term(3 m) << w_term(20 m)."""
    z_lo = torch.clamp((range_m - r_lo) / w_lo, -30.0, 30.0)
    z_hi = torch.clamp((r_hi - range_m) / w_hi, -30.0, 30.0)
    band = torch.sigmoid(z_lo) * torch.sigmoid(z_hi)
    return w0 + (1.0 - w0) * band


def perception_reward(t_cam: Tensor, range_m: Tensor, delta_s: Tensor, in_image: Tensor,
                      arm: str, w: Inc8RewardWeights) -> Tensor:
    """R5' = rw_perc * w_term(d) * v(alpha,beta) * max(delta_s, 0). arm 'A' terminal-locked, 'B' flat
    weight (w_term==1), 'C' off (0). Progress-gated by max(delta_s,0) (anti-loiter). Gated to in-image
    so an out-of-frame gate (where v is meaningless / behind the camera) pays nothing."""
    arm = arm.upper()
    if arm == "C":
        return torch.zeros_like(range_m)
    v = visibility_2axis(t_cam, w.perc_sigma_a_deg, w.perc_sigma_b_deg)
    wt = (terminal_weight(range_m, w.perc_r_lo, w.perc_r_hi, w.perc_w_lo, w.perc_w_hi, w.perc_w0)
          if arm == "A" else torch.ones_like(range_m))
    gate = in_image.to(v.dtype)
    return w.perc * wt * v * torch.clamp(delta_s, min=0.0) * gate


def fix_bonus_reward(accepted: Tensor, delta_s: Tensor, rw_fix_bonus: float) -> Tensor:
    """Direct un-gameable bonus per ACCEPTED fix, PROGRESS-GATED by max(delta_s,0)>0 (anti-loiter, like
    R5'). A fix is obtainable ONLY by pointing the camera in the surrogate accept band (12-28 m), so this
    rewards the band-pointing OUTCOME (a real fix) rather than the weak/gameable raw-in-image proxy R5'.
    rw_fix_bonus == 0 -> the zero term (byte-identical inc8). (iteration-4 fix-driven fallback lever.)"""
    if rw_fix_bonus == 0.0:
        return torch.zeros_like(delta_s)
    advancing = (delta_s > 0).to(delta_s.dtype)
    return rw_fix_bonus * accepted.to(delta_s.dtype) * advancing


# ===== active-perception LOOK-AT primitive (architecture pivot S0) ============================
# After 4 weight-tuning NO-GOs (pointing->fix is sparse + all-or-nothing; PPO can't discover it from a
# non-pointing start), we ENGINEER the geometric pointing direction instead of learning it. The primitive
# adds a body-rate correction that rotates the camera optical axis toward the gate; the policy still
# learns the racing line (and, later, a gain). g_yaw == g_pitch == 0 -> the zero correction (OFF).
#
# Camera optical (OpenCV: X-right, Y-down, Z-fwd), body FRD (X-fwd, Y-right, Z-down), +20deg mount about
# body-Y (src/racer/frames.py). The DiffAero ACTION rates are FLU (rl/diffaero_dynamics.py:184,
# rate_frd = U[1:4]*diag(1,-1,-1)) -- so a FRD correction maps to the action convention via the SAME flip.
_FLIP_FRD_FLU = (1.0, -1.0, -1.0)       # body-rate FRD<->FLU (involutory; == diffaero _FLIP)
_SIN20, _COS20 = 0.34202014332566871, 0.93969262078590843   # sin/cos(CAMERA_PITCH_RAD == 20 deg)


def r_body_from_camera(device=None, dtype=None) -> Tensor:
    """Constant R_body_from_camera (v_body = R @ v_cam) == frames.R_camera_from_body().T for the default
    (zero pitch/roll) boresight = the pure 20deg mount. BAKED (not a torch reimpl of the rotation -- the
    documented sin-sign-flip trap) and PINNED to canonical numpy frames by tests/test_inc8_lookat.py."""
    return torch.tensor([[0.0,    _SIN20,  _COS20],
                         [1.0,    0.0,     0.0],
                         [0.0,    _COS20, -_SIN20]], device=device, dtype=dtype)


def lookat_correction(t_cam: Tensor, g_yaw: float, g_pitch: float, r_bc: Tensor,
                      flip: Tensor) -> Tensor:
    """Body-rate correction (rad/s, in the FLU ACTION convention) that rotates the camera optical axis
    toward the gate (active perception). Convention-robust: omega_cam = K*(zhat_cam x uhat) componentwise
    = [-g_pitch*uy, g_yaw*ux, 0] (uhat = t_cam/|t_cam|), so it ALWAYS reduces |alpha|/|beta| (verified
    numerically). Mapped FRD via r_bc then FLU via flip. g_pitch=0 => S0 yaw-only (the cheap heading DoF:
    null the gate azimuth, leave elevation to the racing line). g_yaw=g_pitch=0 => zero (byte-identical)."""
    if g_yaw == 0.0 and g_pitch == 0.0:
        return torch.zeros_like(t_cam)
    norm = torch.linalg.norm(t_cam, dim=-1, keepdim=True).clamp(min=1e-6)
    u = t_cam / norm
    ux, uy = u[..., 0], u[..., 1]
    w_cam = torch.stack([-g_pitch * uy, g_yaw * ux, torch.zeros_like(ux)], dim=-1)   # camera frame
    w_frd = w_cam @ r_bc.transpose(-1, -2)                                            # FRD body rate
    return w_frd * flip                                                              # -> FLU action conv


def lookat_warmup_factor(update_idx: int, warmup_updates: int) -> float:
    """Linear gain-warmup MULTIPLIER in [0, 1] for the look-at primitive (the 2/3-seed early-collapse
    fix). The full-strength correction slamming a fresh policy is what spikes value_loss / collapses
    entropy at steps ~100/800; ramping the gain in over the first ``warmup_updates`` PPO updates lets
    the policy settle first. factor = clip(update_idx / warmup_updates, 0, 1), then HOLDS at 1.0.

    🚩 MAGNITUDE-ONLY: this is a non-negative scalar multiplied onto the CONFIGURED gain (the validated
    g_yaw = -3.0); it can never flip the sign (factor >= 0) and never exceeds the target (factor <= 1).
    ``warmup_updates <= 0`` -> 1.0 at EVERY update == the no-warmup path exactly (byte-identical, since
    g * 1.0 == g for any finite gain). Pure float math (no torch) so it is import-safe everywhere."""
    if warmup_updates <= 0:
        return 1.0
    return min(max(update_idx / float(warmup_updates), 0.0), 1.0)


def centering_reward(err_inplane_m: Tensor, range_m: Tensor, rw_centering: float,
                     r_near: float, w_near: float) -> Tensor:
    """Dense terminal-sigma_p0 reward: -rw * sigmoid((r_near - range)/w) * err_inplane. Ramps the GT
    in-plane estimator-error penalty UP near the crossing (range->0), where err_inplane == the terminal
    centering miss sigma_p0. Dense + GT-anchored (reads truth -> not a gameable proxy); not coast-
    satisfiable without a band fix. rw_centering == 0 -> the zero term (byte-identical inc8)."""
    if rw_centering == 0.0:
        return torch.zeros_like(err_inplane_m)
    near = torch.sigmoid((r_near - range_m) / max(w_near, 1e-6))
    return -rw_centering * near * err_inplane_m


def bsr3_update(spin_clock: Tensor, omega_realized: Tensor, dt: float,
                rate_abort: float, time_abort: float):
    """BSR3 spin-margin gate (MANDATORY before the inc8 ladder). Gates the REALIZED body rate
    (self._w), NOT the command: a legitimate ~11 rad/s super-rate TRANSIENT does not abort because it
    is not SUSTAINED past ``time_abort`` (3.0 s). Per env: accumulate dt while |omega| > rate_abort,
    reset to 0 otherwise; abort when the sustained clock exceeds time_abort. ``rate_abort <= 0``
    disables it (byte-identical inc7). Returns (new_clock, abort_mask)."""
    if rate_abort <= 0.0:
        return spin_clock, torch.zeros_like(spin_clock, dtype=torch.bool)
    mag = torch.linalg.norm(omega_realized, dim=-1)
    over = mag > rate_abort
    new_clock = torch.where(over, spin_clock + dt, torch.zeros_like(spin_clock))
    abort = new_clock > time_abort
    return new_clock, abort
