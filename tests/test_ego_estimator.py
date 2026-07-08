"""Tests for rl/ego_estimator.py -- verify the INJECTED statistics match the MEASURED targets and
the load-bearing NO-WORLD-POSITION invariant.

Verified statistics (DESIGN.md §5.A, §6.2):
  (a) per-gate relative-fix noise reproduces the anisotropic sigmas (lat max(0.1045,0.0028*range),
      vert 0.2816, depth 0.8524) AND the sqrt(N_eff) smoothing;
  (b) velocity drift is BIAS-dominated (zero bias -> ~mm/s over a gap; nonzero bias -> error ~ bias*dt,
      RESETS at a fix), NOT noise-dominated;
  (c) gyro/rate noise is COLORED: lag-1 autocorrelation ~0.75 (NOT ~0 white);
  (d) a gate that stops being detectable is propagated for the horizon then MASKED (confidence -> 0),
      never frozen-and-drifted indefinitely;
  (e) THE INVARIANT: the estimator holds/produces NO world/absolute position or heading -- results are
      invariant to a global world translation/yaw of the whole scene.

Run from repo ROOT:
    .venv\\Scripts\\python.exe -m pytest tests/test_ego_estimator.py -q
"""
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))

import ego_estimator as EE                                                # noqa: E402
from inc8_estimator_emul import TorchSurrogateParams, quat_xyzw_to_matrix_torch  # noqa: E402

DT = torch.float64


# ---- shared builders ----------------------------------------------------------------------------
def _yaw_pitch_roll_quat(yaw=0.0, pitch=0.0, roll=0.0, n=1):
    """XYZW body->world quat for Rz(yaw)Ry(pitch)Rx(roll), batched (n,4)."""
    cy, sy = torch.cos(torch.tensor(yaw / 2, dtype=DT)), torch.sin(torch.tensor(yaw / 2, dtype=DT))
    cp, sp = torch.cos(torch.tensor(pitch / 2, dtype=DT)), torch.sin(torch.tensor(pitch / 2, dtype=DT))
    cr, sr = torch.cos(torch.tensor(roll / 2, dtype=DT)), torch.sin(torch.tensor(roll / 2, dtype=DT))
    # ZYX Hamilton product qz*qy*qx -> (w,x,y,z)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    q = torch.tensor([x, y, z, w], dtype=DT)
    return q.unsqueeze(0).expand(n, 4).contiguous()


def _identity_quat(n=1):
    q = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=DT)
    return q.unsqueeze(0).expand(n, 4).contiguous()


def _make_est(n, gate_pos, gate_yaw, cfg=None, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return EE.BatchedEgoEstimator(n, gate_pos, gate_yaw, config=cfg, device=torch.device("cpu"),
                                  dtype=DT, generator=gen)


# ================================================================================================
# (a) VISION per-gate relative-fix anisotropic sigmas + sqrt(N_eff) smoothing.
# ================================================================================================
def test_a_anisotropic_fix_sigma_matches_measured():
    """With NO smoothing (N_eff=1) and always-accept, the single-fix residual in the GATE frame
    reproduces the measured per-axis sigmas: lat=max(0.1045,0.0028*r), vert=0.2816, depth=0.8524."""
    p = TorchSurrogateParams()
    r = 20.0                                              # range where lateral floor still dominates a bit
    # place one gate straight ahead at range r (drone level at origin, identity attitude).
    gate_pos = torch.tensor([[[r, 0.0, 0.0]]], dtype=DT)
    gate_yaw = torch.zeros(1, 1, dtype=DT)
    N = 40000
    gate_pos = gate_pos.expand(N, 1, 3).contiguous()
    gate_yaw = gate_yaw.expand(N, 1).contiguous()
    cfg = EE.EgoEstimatorConfig(n_eff_lo=1, n_eff_hi=1, inject_bias=False, miss_prob=0.0,
                                teleport_prob=0.0, dr_accel_bias=False)
    est = _make_est(N, gate_pos, gate_yaw, cfg, seed=1)
    q = _identity_quat(N)
    pos = torch.zeros(N, 3, dtype=DT)
    vel = torch.zeros(N, 3, dtype=DT)
    est.reset_idx(torch.arange(N), pos, vel, q)
    # detectable all -> single fix into an EMPTY smoother (K=1 -> rel_pos == fix exactly).
    detect = torch.ones(N, 1, dtype=torch.bool)
    # seed rel_pos to truth first (reset does), then one step: with N_eff=1, K=1 => rel_pos = fix.
    out = est.step(pos, vel, q, torch.zeros(N, 3, dtype=DT), dt=1.0 / 30.0, detectable=detect,
                   prev_quat=q)
    # residual in the GATE frame: rotate (rel_pos - true_rel_body) by R_body_gate^T
    R_wb = quat_xyzw_to_matrix_torch(q)
    true_rel = torch.einsum("nji,ngj->ngi", R_wb, gate_pos - pos.unsqueeze(1))    # (N,1,3) body
    resid_body = out.rel_pos - true_rel                                           # (N,1,3) body
    R_bg = torch.einsum("nji,ngjk->ngik", R_wb, est.R_world_gate)                # gate->body
    resid_gate = torch.einsum("ngij,ngj->ngi", R_bg.transpose(-1, -2), resid_body)  # body->gate
    sig = resid_gate[:, 0, :].std(dim=0)                                          # [lat, depth, vert]
    lat_target = max(p.sigma_lateral_floor, p.sigma_lateral_a1 * r)
    print(f"\n[a] range={r} m single-fix sigma gate-frame [lat,depth,vert] = "
          f"{sig.tolist()}  targets=[{lat_target:.4f},{p.sigma_depth_floor:.4f},{p.sigma_vertical_floor:.4f}]")
    assert abs(sig[0].item() - lat_target) < 0.02, sig[0].item()
    assert abs(sig[1].item() - p.sigma_depth_floor) < 0.03, sig[1].item()
    assert abs(sig[2].item() - p.sigma_vertical_floor) < 0.02, sig[2].item()


def test_a_lateral_range_collapse():
    """lateral sigma is max(floor, a1*range) -- the RANGE-COLLAPSE lever (DESIGN.md §5.A). Within the
    30 m cap the a1*range term (a1*30 ~ 0.083) sits below the 0.1045 floor, so the floor dominates
    and the realised lateral sigma is the floor at every operating range (the collapse is fully in
    the floor regime here); above the un-capped crossover (~37.8 m) the a1 term would exceed the
    floor. We pin BOTH: (i) the formula grows with range when un-capped, (ii) the cap holds it at the
    floor across the operating band."""
    p = TorchSurrogateParams()
    crossover = p.sigma_lateral_floor / p.sigma_lateral_a1           # ~37.8 m
    assert crossover > p.sigma_growth_max_range_m                    # cap (30) is BELOW the crossover
    est = _make_est(1, torch.tensor([[[10.0, 0.0, 0.0]]], dtype=DT), torch.zeros(1, 1, dtype=DT))
    # (i) un-capped monotonic growth of the raw a1*range term
    assert p.sigma_lateral_a1 * 40.0 > p.sigma_lateral_a1 * 20.0
    # (ii) capped realised lateral sigma == floor at both near and cap range
    s_cap = est._fix_sigma_gate(torch.tensor([[40.0]], dtype=DT))[0, 0]     # clamps to 30 m
    s_near = est._fix_sigma_gate(torch.tensor([[5.0]], dtype=DT))[0, 0]
    assert abs(s_cap[0].item() - p.sigma_lateral_floor) < 1e-9
    assert abs(s_near[0].item() - p.sigma_lateral_floor) < 1e-9
    print(f"\n[a] lateral sigma: floor={p.sigma_lateral_floor:.4f}, crossover~{crossover:.1f} m > "
          f"cap {p.sigma_growth_max_range_m} m -> floor across the operating band")


def test_a_sqrt_neff_smoothing():
    """The KF smoother reduces the realised in-plane sigma by ~sqrt(N_eff) vs a single fix. With
    N_eff fixed and repeated accepted fixes at a STATIC pose, the steady-state std of rel_pos about
    truth ~ per_fix_sigma / sqrt(N_eff) (order-of-magnitude / trend check)."""
    r = 15.0
    N = 20000
    gate_pos = torch.tensor([[[r, 0.0, 0.0]]], dtype=DT).expand(N, 1, 3).contiguous()
    gate_yaw = torch.zeros(N, 1, dtype=DT)
    q = _identity_quat(N)
    pos = torch.zeros(N, 3, dtype=DT)
    vel = torch.zeros(N, 3, dtype=DT)
    detect = torch.ones(N, 1, dtype=torch.bool)
    p = TorchSurrogateParams()

    def realised_lateral_std(n_eff):
        cfg = EE.EgoEstimatorConfig(n_eff_lo=n_eff, n_eff_hi=n_eff, inject_bias=False,
                                    miss_prob=0.0, teleport_prob=0.0, dr_accel_bias=False)
        est = _make_est(N, gate_pos, gate_yaw, cfg, seed=7)
        est.reset_idx(torch.arange(N), pos, vel, q)
        # run to steady state (static pose, dt=0 so no propagation drift): many fixes.
        for _ in range(200):
            out = est.step(pos, vel, q, torch.zeros(N, 3, dtype=DT), dt=0.0, detectable=detect, prev_quat=q)
        R_wb = quat_xyzw_to_matrix_torch(q)
        true_rel = torch.einsum("nji,ngj->ngi", R_wb, gate_pos - pos.unsqueeze(1))
        resid_body = out.rel_pos - true_rel
        R_bg = torch.einsum("nji,ngjk->ngik", R_wb, est.R_world_gate)
        resid_gate = torch.einsum("ngij,ngj->ngi", R_bg.transpose(-1, -2), resid_body)
        return resid_gate[:, 0, 0].std().item()                      # lateral axis

    lat1 = realised_lateral_std(1)
    lat9 = realised_lateral_std(9)
    per_fix = max(p.sigma_lateral_floor, p.sigma_lateral_a1 * r)
    # steady-state K=1/N_eff low-pass: var = per_fix_var * K/(2-K) => std ratio (N_eff=9 vs 1)
    # ~ sqrt((1/9)/(2-1/9) / (1/(2-1))) = sqrt( (1/9)/(17/9) ) = sqrt(1/17) ~ 0.243.
    ratio = lat9 / lat1
    print(f"\n[a] smoothing: single-fix lat std={lat1:.4f} (per_fix={per_fix:.4f}); "
          f"N_eff=9 lat std={lat9:.4f}; ratio={ratio:.3f} (target ~0.24)")
    assert lat1 > lat9                                               # smoothing reduces the sigma
    assert 0.15 < ratio < 0.40                                       # ~sqrt(1/17) low-pass steady state


# ================================================================================================
# (b) VELOCITY drift is BIAS-dominated, resets at a fix.
# ================================================================================================
def test_b_zero_bias_velocity_stays_tiny_over_gap():
    """With residual accel bias == 0 (and no fix), body velocity drifts only by the tiny white noise
    over a ~0.5 s gap -> ~mm/s, NOT growing linearly."""
    N = 20000
    gate_pos = torch.tensor([[[10.0, 0.0, 0.0]]], dtype=DT).expand(N, 1, 3).contiguous()
    gate_yaw = torch.zeros(N, 1, dtype=DT)
    cfg = EE.EgoEstimatorConfig(dr_accel_bias=False, accel_bias_band=0.0)   # ZERO residual bias
    est = _make_est(N, gate_pos, gate_yaw, cfg, seed=3)
    q = _identity_quat(N)
    pos = torch.zeros(N, 3, dtype=DT)
    vel = torch.zeros(N, 3, dtype=DT)
    est.reset_idx(torch.arange(N), pos, vel, q)
    v0 = est._vel_body.clone()
    no_detect = torch.zeros(N, 1, dtype=torch.bool)                  # no fixes -> pure IMU dead-reckon
    dt = 1.0 / 30.0
    steps = 15                                                       # ~0.5 s
    for _ in range(steps):
        est.step(pos, vel, q, torch.zeros(N, 3, dtype=DT), dt=dt, detectable=no_detect, prev_quat=q)
    drift = (est._vel_body - v0).abs().mean().item()
    T = steps * dt
    # white-noise random walk over T: std ~ accel_white_sigma * sqrt(T) * ... -> a few mm/s.
    print(f"\n[b] zero-bias velocity drift over {T:.2f} s = {drift*1000:.3f} mm/s (noise-only)")
    assert drift < 0.02, drift                                      # < 2 cm/s: NOT bias-dominated


def test_b_nonzero_bias_drift_is_linear_and_corrected_by_fixes():
    """With a fixed residual accel bias, the body-velocity error grows ~ bias*Delta_t (LINEAR in the
    gap length, dominating white noise). An accepted fix corrects it only PARTIALLY (indirect gain, no
    direct vision-velocity observation); a sustained fix stream keeps velocity BOUNDED near truth."""
    N = 40000
    gate_pos = torch.tensor([[[10.0, 0.0, 0.0]]], dtype=DT).expand(N, 1, 3).contiguous()
    gate_yaw = torch.zeros(N, 1, dtype=DT)
    # fix the bias band so |bias| ~ 0.05 per axis; drone STATIONARY so truth body v stays 0.
    cfg = EE.EgoEstimatorConfig(dr_accel_bias=True, accel_bias_band=0.05, accel_white_sigma=0.008)
    est = _make_est(N, gate_pos, gate_yaw, cfg, seed=5)
    q = _identity_quat(N)
    pos = torch.zeros(N, 3, dtype=DT)
    vel = torch.zeros(N, 3, dtype=DT)                               # truth body velocity == 0
    est.reset_idx(torch.arange(N), pos, vel, q)
    bias = est._accel_bias.clone()                                  # (N,3)
    dt = 1.0 / 30.0
    no_detect = torch.zeros(N, 1, dtype=torch.bool)

    # dead-reckon for a gap; error should track bias * t (truth v is 0 so error == vel_body).
    for k in range(1, 11):
        est.step(pos, vel, q, torch.zeros(N, 3, dtype=DT), dt=dt, detectable=no_detect, prev_quat=q)
    T = 10 * dt
    predicted = bias * T                                            # (N,3) the bias*t drift lever
    err = est._vel_body - predicted
    # the residual (actual - bias*t) is only the white-noise walk: tiny vs the bias*t signal.
    signal = predicted.abs().mean().item()
    residual = err.abs().mean().item()
    print(f"\n[b] bias*t signal={signal*1000:.2f} mm/s, non-bias residual={residual*1000:.3f} mm/s "
          f"over {T:.2f} s")
    assert residual < 0.2 * signal, (residual, signal)             # bias term DOMINATES (>5x)
    assert residual < 0.01                                         # residual is only mm/s white walk

    # INDIRECT correction: a single fix does NOT snap to truth (deploy has no direct vision-velocity
    # observation) -- it applies a small partial gain. A sustained fix stream keeps velocity BOUNDED
    # near truth (does not diverge) but never one-step-resets.
    detect = torch.ones(N, 1, dtype=torch.bool)
    est.cfg = EE.EgoEstimatorConfig(**{**est.cfg.__dict__, "miss_prob": 0.0})   # force accept
    before = est._vel_body.abs().mean().item()
    est.step(pos, vel, q, torch.zeros(N, 3, dtype=DT), dt=dt, detectable=detect, prev_quat=q)
    after_one = est._vel_body.abs().mean().item()
    print(f"[b] vel err before fix={before*1000:.3f} mm/s, after ONE fix={after_one*1000:.3f} mm/s "
          f"(partial gain {est.cfg.vel_correct_gain})")
    assert after_one > 0.5 * before, (after_one, before)           # NOT a snap -- most error remains
    assert after_one < before                                      # but a fix DOES reduce it
    for _ in range(80):
        est.step(pos, vel, q, torch.zeros(N, 3, dtype=DT), dt=dt, detectable=detect, prev_quat=q)
    after_many = est._vel_body.abs().mean().item()
    print(f"[b] vel err after MANY fixes={after_many*1000:.3f} mm/s (bounded near truth, not snapped)")
    assert after_many < before                                     # the fix stream bounds/reduces it
    assert after_many < 0.02                                       # stays small (mm/s-scale, bounded)


# ================================================================================================
# (c) GYRO noise is COLORED (AR(1), lag-1 autocorr ~0.75), NOT white.
# ================================================================================================
def test_c_gyro_noise_is_colored_ar1():
    """The injected gyro noise (body_rates - truth) is AR(1): lag-1 autocorrelation ~0.75, and its
    marginal std ~ gyro_sigma. A white model would give autocorr ~0."""
    N = 4000
    T = 4000
    gate_pos = torch.tensor([[[10.0, 0.0, 0.0]]], dtype=DT).expand(N, 1, 3).contiguous()
    gate_yaw = torch.zeros(N, 1, dtype=DT)
    cfg = EE.EgoEstimatorConfig()
    est = _make_est(N, gate_pos, gate_yaw, cfg, seed=11)
    q = _identity_quat(N)
    pos = torch.zeros(N, 3, dtype=DT)
    vel = torch.zeros(N, 3, dtype=DT)
    est.reset_idx(torch.arange(N), pos, vel, q)
    truth_rates = torch.zeros(N, 3, dtype=DT)
    detect = torch.zeros(N, 1, dtype=torch.bool)
    series = []
    for _ in range(T):
        out = est.step(pos, vel, q, truth_rates, dt=1.0 / 30.0, detectable=detect, prev_quat=q)
        series.append((out.body_rates - truth_rates)[:, 0].clone())   # injected gyro noise, x-axis
    e = torch.stack(series, dim=0)                                    # (T, N)
    e = e - e.mean(dim=0, keepdim=True)
    v0 = (e[:-1] * e[:-1]).mean()
    v1 = (e[:-1] * e[1:]).mean()
    rho_hat = (v1 / v0).item()
    sig_hat = e.std().item()
    print(f"\n[c] gyro noise: lag-1 autocorr rho_hat={rho_hat:.3f} (target 0.75), "
          f"std={sig_hat:.2e} rad/s (target {cfg.gyro_sigma:.1e})")
    assert abs(rho_hat - cfg.gyro_ar1_rho) < 0.05, rho_hat           # COLORED, not white (~0)
    assert rho_hat > 0.6                                             # definitively not white
    assert abs(sig_hat - cfg.gyro_sigma) / cfg.gyro_sigma < 0.15     # marginal std matches


# ================================================================================================
# (d) A gate that stops being detectable is propagated for the horizon then MASKED.
# ================================================================================================
def test_d_propagate_then_mask():
    """A gate detectable at first, then never again: confidence stays > 0 through the horizon
    (~0.5 s), then drops to EXACTLY 0 (MASK) and stays there -- never frozen-and-drifted forever."""
    N = 8
    gate_pos = torch.tensor([[[10.0, 0.0, 0.0]]], dtype=DT).expand(N, 1, 3).contiguous()
    gate_yaw = torch.zeros(N, 1, dtype=DT)
    cfg = EE.EgoEstimatorConfig(miss_prob=0.0, teleport_prob=0.0, stale_horizon_s=0.5)
    est = _make_est(N, gate_pos, gate_yaw, cfg, seed=2)
    q = _identity_quat(N)
    pos = torch.zeros(N, 3, dtype=DT)
    vel = torch.zeros(N, 3, dtype=DT)
    est.reset_idx(torch.arange(N), pos, vel, q)
    dt = 1.0 / 30.0
    # one detectable step -> fresh fix, confidence == 1
    out = est.step(pos, vel, q, torch.zeros(N, 3, dtype=DT), dt=dt,
                   detectable=torch.ones(N, 1, dtype=torch.bool), prev_quat=q)
    assert out.confidence[0, 0].item() > 0.99, out.confidence[0, 0].item()
    no_detect = torch.zeros(N, 1, dtype=torch.bool)
    confs = []
    for step_i in range(30):                                         # 1.0 s of no detections
        out = est.step(pos, vel, q, torch.zeros(N, 3, dtype=DT), dt=dt, detectable=no_detect, prev_quat=q)
        confs.append(out.confidence[0, 0].item())
    # within the horizon (first ~15 steps -> 0.5 s) confidence decays but is still > 0 for a while;
    # past the horizon it is EXACTLY 0.
    within = confs[5]                                               # ~0.2 s in
    beyond = confs[-1]                                             # ~1.0 s in
    print(f"\n[d] confidence within horizon (~0.2 s) = {within:.3f}; beyond horizon (~1.0 s) = {beyond:.3f}")
    assert 0.0 < within < 1.0, within                              # decaying, propagated (not masked yet)
    assert beyond == 0.0, beyond                                   # MASKED past the horizon
    # and it STAYS masked (never un-masks without a fix)
    assert all(c == 0.0 for c in confs[16:]), confs[16:]


def test_d_reacquire_snaps_stale_prior():
    """After a gate is MASKED (gap past the horizon) its propagated rel_pos has drifted; the FIRST
    fresh fix must SNAP rel_pos to the measurement (K=1), discarding the stale prior -- otherwise
    confidence reads fresh (1.0) while rel_pos still carries the stale error (confidently wrong)."""
    N = 20000
    gate_pos = torch.tensor([[[12.0, 0.0, 0.0]]], dtype=DT).expand(N, 1, 3).contiguous()
    gate_yaw = torch.zeros(N, 1, dtype=DT)
    cfg = EE.EgoEstimatorConfig(miss_prob=0.0, dr_accel_bias=False)     # clean forced fixes, stationary
    est = _make_est(N, gate_pos, gate_yaw, cfg, seed=7)
    q = _identity_quat(N)
    pos = torch.zeros(N, 3, dtype=DT)
    vel = torch.zeros(N, 3, dtype=DT)
    est.reset_idx(torch.arange(N), pos, vel, q)
    # simulate a stale, drifted prior aged past the mask horizon (0.5 s)
    est._rel_pos = est._rel_pos + torch.tensor([0.0, 3.0, 0.0], dtype=DT)   # 3 m of stale drift
    est._t_since_fix = torch.full((N, 1), 1.0, dtype=DT)                    # > stale_horizon_s -> was masked
    detect = torch.ones(N, 1, dtype=torch.bool)
    out = est.step(pos, vel, q, torch.zeros(N, 3, dtype=DT), dt=1.0 / 30.0, detectable=detect, prev_quat=q)
    err = (out.rel_pos[:, 0, :] - torch.tensor([12.0, 0.0, 0.0], dtype=DT)).norm(dim=-1).mean().item()
    conf = out.confidence[0, 0].item()
    print(f"\n[d2] re-acquire rel_pos error = {err:.3f} m (stale prior was 3 m), confidence = {conf:.3f}")
    assert err < 1.5, err          # snapped to the fresh fix (~per-fix noise), NOT the 3 m stale prior
    assert conf > 0.99, conf       # confidence is honest/fresh after the snap


# ================================================================================================
# (e) THE INVARIANT: no world position / heading -- global translation + yaw invariance.
# ================================================================================================
def test_e_global_translation_invariance():
    """Translate the whole scene (drone + ALL gates) by a constant world offset: every relative
    output (rel_pos, rel_normal, rel_yaw, confidence) and the body velocity are UNCHANGED. If any world
    position leaked into the state, this would break."""
    N = 64
    G = 4
    torch.manual_seed(0)
    gate_pos = torch.randn(N, G, 3, dtype=DT) * 5.0 + torch.tensor([15.0, 0.0, 0.0], dtype=DT)
    gate_yaw = torch.rand(N, G, dtype=DT) * 0.5
    q = _yaw_pitch_roll_quat(yaw=0.3, pitch=-0.1, roll=0.05, n=N)
    pos = torch.randn(N, 3, dtype=DT)
    vel = torch.randn(N, 3, dtype=DT)
    rates = torch.randn(N, 3, dtype=DT) * 0.2
    dt = 1.0 / 30.0
    detect = torch.ones(N, G, dtype=torch.bool)

    def run(offset):
        est = _make_est(N, gate_pos + offset, gate_yaw, EE.EgoEstimatorConfig(), seed=42)
        est.reset_idx(torch.arange(N), pos + offset, vel, q)
        return est.step(pos + offset, vel, q, rates, dt=dt, detectable=detect, prev_quat=q)

    off = torch.tensor([123.4, -56.7, 8.9], dtype=DT)
    a = run(torch.zeros(3, dtype=DT))
    b = run(off)
    for name in ["rel_pos", "rel_normal", "rel_yaw", "confidence", "normal_conf", "velocity",
                 "roll_pitch", "body_rates"]:
        ta, tb = getattr(a, name), getattr(b, name)
        d = (ta - tb).abs().max().item()
        print(f"[e] translation: {name} max|delta| = {d:.2e}")
        assert d < 1e-9, (name, d)


def test_e_global_yaw_invariance():
    """Rotate the whole scene (drone attitude yaw + all gate yaws + drone/gate positions) by a global
    world yaw about +z: the BODY-frame relative outputs and the BODY-frame velocity are UNCHANGED
    (a global heading is unobservable and absent from the state). roll/pitch and rates are also
    unchanged (yaw does not affect the gravity-leveled tilt)."""
    N = 64
    G = 3
    torch.manual_seed(1)
    base_gate = torch.randn(N, G, 3, dtype=DT) * 4.0 + torch.tensor([12.0, 0.0, 1.0], dtype=DT)
    base_yaw = torch.rand(N, G, dtype=DT) * 0.4
    base_pos = torch.randn(N, 3, dtype=DT) * 2.0
    base_vel = torch.randn(N, 3, dtype=DT)
    rates = torch.randn(N, 3, dtype=DT) * 0.1
    dt = 1.0 / 30.0
    detect = torch.ones(N, G, dtype=torch.bool)
    psi = 0.7                                                       # global world yaw

    def Rz(a):
        c, s = torch.cos(torch.tensor(a, dtype=DT)), torch.sin(torch.tensor(a, dtype=DT))
        return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=DT)

    def run(yaw_off, rotate):
        gp = base_gate.clone()
        gy = base_yaw + yaw_off
        pos = base_pos.clone()
        vel = base_vel.clone()
        q = _yaw_pitch_roll_quat(yaw=0.2 + yaw_off, pitch=-0.15, roll=0.1, n=N)
        if rotate:
            R = Rz(yaw_off)
            gp = torch.einsum("ij,ngj->ngi", R, gp)
            pos = torch.einsum("ij,nj->ni", R, pos)
            vel = torch.einsum("ij,nj->ni", R, vel)
        est = _make_est(N, gp, gy, EE.EgoEstimatorConfig(), seed=99)
        est.reset_idx(torch.arange(N), pos, vel, q)
        return est.step(pos, vel, q, rates, dt=dt, detectable=detect, prev_quat=q)

    a = run(0.0, rotate=False)
    b = run(psi, rotate=True)
    # body-frame relative + velocity + tilt + rates are all invariant to the global yaw.
    for name in ["rel_pos", "rel_normal", "rel_yaw", "confidence", "normal_conf", "velocity",
                 "roll_pitch", "body_rates"]:
        ta, tb = getattr(a, name), getattr(b, name)
        d = (ta - tb).abs().max().item()
        print(f"[e] global-yaw: {name} max|delta| = {d:.2e}")
        assert d < 1e-8, (name, d)


def test_e_no_world_state_attributes():
    """Structural assertion: the estimator exposes NO attribute that is a world/absolute position or
    heading. Every stored spatial quantity is body-frame/relative. Guards against a future regression
    that re-introduces a world anchor."""
    est = _make_est(2, torch.tensor([[[10.0, 0.0, 0.0]]], dtype=DT).expand(2, 1, 3).contiguous(),
                    torch.zeros(2, 1, dtype=DT))
    # the only stored positions are RELATIVE (per-gate, body frame) and gate geometry is course config,
    # not an estimate. Assert there is no world-pos / world-yaw / heading state field.
    banned = ["_world_pos", "_pos_world", "_kf_pos", "_position", "_world_yaw", "_heading",
              "_abs_pos", "_ned_pos", "_p_world"]
    for name in banned:
        assert not hasattr(est, name), f"world-state leak: {name}"
    # the EgoEstimate dataclass fields carry no world position/heading
    est_out_fields = EE.EgoEstimate.__dataclass_fields__.keys()
    for f in est_out_fields:
        assert "world" not in f and f not in ("yaw",), f          # rel_yaw ok; bare 'yaw' would be world
    print(f"\n[e] no world-state attributes; output fields = {list(est_out_fields)}")


# ================================================================================================
# (f) VISIBLE_AREA (apparent projected opening area) channel: normalized projected inner-opening area
#     from GT + modest noise, ~1 head-on, small at a sharp angle, MASKED (0) when not detectable/stale.
# ================================================================================================
def test_f_visible_area_headon_vs_sharp_angle_and_mask():
    """visible_area = the NORMALIZED apparent projected opening area in [0,1] (Fengyou 2026-07-07
    recalibration, gate_apparent_area -- NOT the old |cos| proxy). Approaching a gate ALONG its
    through-axis (head-on) -> ~1 (a full square). A sharp angle to the normal foreshortens the projected
    opening -> smaller. A gate that is not detectable is MASKED to visible_area 0 (like confidence).
    The projected area is FLATTER than |cos| near square-on (tolerant of small misalignments) and drops
    off for genuinely shallow approaches -- the standalone path here uses the raw-attitude camera."""
    N = 20000
    # HEAD-ON: gate straight ahead (yaw=0 -> normal along +x world), drone at origin looking +x.
    # view_ray = normalize(gate - drone) = +x; gate normal (downrange) = +x -> |cos| = 1.
    gate_pos = torch.tensor([[[15.0, 0.0, 0.0]]], dtype=DT).expand(N, 1, 3).contiguous()
    gate_yaw = torch.zeros(N, 1, dtype=DT)
    cfg = EE.EgoEstimatorConfig(miss_prob=0.0, teleport_prob=0.0, visible_area_sigma=0.05)
    est = _make_est(N, gate_pos, gate_yaw, cfg, seed=21)
    q = _identity_quat(N)
    pos = torch.zeros(N, 3, dtype=DT)
    vel = torch.zeros(N, 3, dtype=DT)
    est.reset_idx(torch.arange(N), pos, vel, q)
    detect = torch.ones(N, 1, dtype=torch.bool)
    out = est.step(pos, vel, q, torch.zeros(N, 3, dtype=DT), dt=1.0 / 30.0, detectable=detect, prev_quat=q)
    va_headon = out.visible_area[:, 0]
    assert (va_headon >= 0.0).all() and (va_headon <= 1.0).all(), "visible_area must be in [0,1]"
    mean_headon = va_headon.mean().item()
    print(f"\n[f] head-on visible_area mean = {mean_headon:.3f} (target ~1.0), "
          f"std = {va_headon.std().item():.3f} (~sigma 0.05)")
    assert mean_headon > 0.9, mean_headon                        # nearly head-on -> near 1

    # SHARP ANGLE: view ray ~60 deg off the gate normal. Keep the gate straight ahead of the drone
    # (view_ray = +x) but rotate the gate's yaw so its normal points ~60 deg away from +x -> |cos60|=0.5.
    ang = math.radians(60.0)
    gate_yaw2 = torch.full((N, 1), ang, dtype=DT)               # normal = (cos60, sin60, 0)
    est2 = _make_est(N, gate_pos, gate_yaw2, cfg, seed=22)
    est2.reset_idx(torch.arange(N), pos, vel, q)
    out2 = est2.step(pos, vel, q, torch.zeros(N, 3, dtype=DT), dt=1.0 / 30.0, detectable=detect, prev_quat=q)
    va_sharp = out2.visible_area[:, 0].mean().item()
    print(f"[f] sharp-angle (60 deg) visible_area mean = {va_sharp:.3f} (projected-area ~0.60, < head-on)")
    assert va_sharp < mean_headon, (va_sharp, mean_headon)      # foreshortened -> smaller
    # projected opening area at 60 deg tilt (~0.60; higher than |cos60|=0.5 -- perspective, not a cosine):
    # clearly foreshortened (well below head-on) but not as steep as a pure cosine.
    assert 0.45 < va_sharp < 0.75, va_sharp

    # MASK: a NON-detectable gate -> visible_area 0 (stale mask), same as confidence.
    est3 = _make_est(N, gate_pos, gate_yaw, cfg, seed=23)
    est3.reset_idx(torch.arange(N), pos, vel, q)
    no_detect = torch.zeros(N, 1, dtype=torch.bool)
    # propagate past the stale horizon with no detections -> masked
    for _ in range(30):
        out3 = est3.step(pos, vel, q, torch.zeros(N, 3, dtype=DT), dt=1.0 / 30.0,
                         detectable=no_detect, prev_quat=q)
    assert out3.confidence[:, 0].max().item() == 0.0, "gate should be masked (confidence 0)"
    assert out3.visible_area[:, 0].abs().max().item() == 0.0, "masked gate -> visible_area 0"
    print(f"[f] undetectable gate: visible_area max = {out3.visible_area[:, 0].abs().max().item():.3f} (0)")
