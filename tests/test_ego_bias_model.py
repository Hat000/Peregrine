"""Tests for the per-episode BIAS MODEL knob (EgoEstimatorConfig.bias_model) in rl/ego_estimator.py.

Motivation: handoff/audit-ego-inc9-2026-07-09/read_estimator-kf-audit.md red flag [major] -- the legacy
per-episode bias draws ONE scalar magnitude with a RANDOM sign and applies it to BOTH lat and vert
(perfectly correlated), depth 0. That contradicts the measurement (handoff/fix-surrogate-2026-06-14/
models/sigma.json): per-axis biases are INDEPENDENT and ONE-signed -- lateral -0.0338, vertical +0.1949,
depth -0.3344. ``bias_model='measured'`` draws independent per-axis magnitudes ~U[0, |band|] with the
measured sign; ``bias_model='legacy'`` (default) is preserved BYTE-IDENTICAL.

Coverage (prompt task):
  (c) legacy   -> the reset_idx DR draw sequence (N_eff, accel_bias, bias mag, bias sign) is BIT-IDENTICAL
      to the original code (an independent seeded REPLAY of the exact draw order/count matches every DR
      tensor), and the bias keeps the legacy [b, 0, b] structure (lat==vert correlated, depth 0). A
      shifted stream (an extra draw in the legacy path) would desync the replay -> this is the byte-
      identity guard the prompt requires.
  (d) measured -> per-axis SIGN is exact and one-signed (lat<=0, vert>=0, depth<=0), magnitudes lie in
      [0,|band|], per-axis MEANS ~ band/2 over many resets, the axes are INDEPENDENT (near-zero cross-
      correlation, unlike legacy's lat==vert), and everything scales with noise_scale (0 -> no bias).

Driven directly on BatchedEgoEstimator (no diffaero). Run from repo ROOT:
    .venv\\Scripts\\python.exe -m pytest tests/test_ego_bias_model.py -q
"""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))

import ego_estimator as EE                                                 # noqa: E402

DT = torch.float64
CPU = torch.device("cpu")

# sigma.json measured bias bands (handoff/fix-surrogate-2026-06-14/models/sigma.json) -- the defaults.
SIGMA_JSON_LAT = -0.033817701667839546
SIGMA_JSON_VERT = 0.19488467958140898
SIGMA_JSON_DEPTH = -0.33441613167150736


def _identity_quat(n=1):
    q = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=DT)
    return q.unsqueeze(0).expand(n, 4).contiguous()


def _straight_course(n, G, spacing=12.0):
    xs = spacing * (1.0 + torch.arange(G, dtype=DT))
    gate_pos = torch.stack([xs, torch.zeros(G, dtype=DT), torch.zeros(G, dtype=DT)], dim=-1)
    gate_pos = gate_pos.unsqueeze(0).expand(n, G, 3).contiguous()
    gate_yaw = torch.zeros(n, G, dtype=DT)
    return gate_pos, gate_yaw


def _build(n, gate_pos, gate_yaw, cfg, seed):
    gen = torch.Generator().manual_seed(seed)
    return EE.BatchedEgoEstimator(n, gate_pos, gate_yaw, config=cfg, device=CPU, dtype=DT, generator=gen)


# ================================================================================================
# defaults: config fields equal the sigma.json values, default model is 'legacy'.
# ================================================================================================
def test_defaults_match_sigma_json_and_legacy():
    cfg = EE.EgoEstimatorConfig()
    assert cfg.bias_model == "legacy", "DEFAULT must stay legacy (byte-identical)"
    assert cfg.bias_band_lat == SIGMA_JSON_LAT
    assert cfg.bias_band_vert == SIGMA_JSON_VERT
    assert cfg.bias_band_depth == SIGMA_JSON_DEPTH
    print(f"\n[defaults] bias_model={cfg.bias_model!r}, bands lat/vert/depth = "
          f"{cfg.bias_band_lat:.4f}/{cfg.bias_band_vert:.4f}/{cfg.bias_band_depth:.4f}")


# ================================================================================================
# (c) legacy: reset_idx DR draw sequence bit-identical (independent replay) + [b,0,b] structure.
# ================================================================================================
def test_c_legacy_reset_draw_stream_bit_identical():
    """Reset a legacy-default estimator with a fixed seed, then INDEPENDENTLY replay the reset_idx draw
    sequence (N_eff -> accel_bias -> bias mag -> bias sign) on a fresh identically-seeded generator and
    assert _n_eff, _accel_bias, AND _bias all match bit-for-bit. Any extra/re-ordered draw in the legacy
    path would desync this replay -> proves byte-identity of the whole reset RNG stream."""
    N, G = 128, 3
    gate_pos, gate_yaw = _straight_course(N, G)
    cfg = EE.EgoEstimatorConfig()                                         # legacy default
    assert cfg.bias_model == "legacy" and cfg.dr_accel_bias and cfg.inject_bias
    seed = 20260709
    est = _build(N, gate_pos, gate_yaw, cfg, seed)
    q = _identity_quat(N)
    pos = torch.zeros(N, 3, dtype=DT)
    vel = torch.randn(N, 3, generator=torch.Generator().manual_seed(99), dtype=DT)
    est.reset_idx(torch.arange(N), pos, vel, q)

    # --- independent replay of the EXACT reset_idx draw order (the estimator's ctor consumes no RNG) ---
    g2 = torch.Generator().manual_seed(seed)
    m = N
    u = torch.rand(m, G, generator=g2, dtype=DT)                          # 1) N_eff
    ab = torch.rand(m, 3, generator=g2, dtype=DT)                         # 2) accel_bias (dr_accel_bias)
    mag = cfg.noise_scale * cfg.bias_mag_hi * torch.rand(m, G, generator=g2, dtype=DT)   # 3) bias mag
    sign_r = torch.rand(m, G, generator=g2, dtype=DT)                     # 4) bias sign
    exp_n_eff = cfg.n_eff_lo + (cfg.n_eff_hi - cfg.n_eff_lo) * u
    exp_accel_bias = cfg.noise_scale * cfg.accel_bias_band * (2.0 * ab - 1.0)
    sign = torch.where(sign_r < 0.5, torch.ones(m, G, dtype=DT), -torch.ones(m, G, dtype=DT))
    b = sign * mag
    exp_bias = torch.stack([b, torch.zeros_like(b), b], dim=-1)           # [lat, depth=0, vert]

    assert torch.equal(est._n_eff, exp_n_eff), "N_eff draw desynced (stream shifted)"
    assert torch.equal(est._accel_bias, exp_accel_bias), "accel_bias draw desynced (stream shifted)"
    assert torch.equal(est._bias, exp_bias), "legacy bias not bit-identical to the original formula"
    # legacy STRUCTURE: lat == vert (perfectly correlated), depth == 0.
    assert torch.equal(est._bias[..., 0], est._bias[..., 2]), "legacy lat/vert must be identical (correlated)"
    assert est._bias[..., 1].abs().max().item() == 0.0, "legacy depth bias must be 0"
    print(f"\n[c] legacy reset DR stream bit-identical (N_eff+accel_bias+bias replay match); "
          f"bias structure [b,0,b] confirmed (lat==vert, depth==0)")


def test_c_legacy_full_step_evolution_reproducible():
    """A multi-step legacy rollout is deterministic under a fixed seed (no hidden state-order change):
    two identically-seeded estimators produce bit-identical rel_pos / velocity / confidence over 20 steps."""
    N, G = 16, 2
    gate_pos, gate_yaw = _straight_course(N, G, spacing=13.0)
    cfg = EE.EgoEstimatorConfig()                                         # legacy default
    q = _identity_quat(N)
    pos = torch.zeros(N, 3, dtype=DT)
    vel = torch.zeros(N, 3, dtype=DT); vel[:, 0] = 4.0
    rates = torch.zeros(N, 3, dtype=DT)
    detect = torch.ones(N, G, dtype=torch.bool)

    def rollout(seed):
        est = _build(N, gate_pos, gate_yaw, cfg, seed)
        est.reset_idx(torch.arange(N), pos, vel, q)
        outs = []
        for _ in range(20):
            e = est.step(pos, vel, q, rates, dt=1 / 30, detectable=detect, prev_quat=q)
            outs.append((e.rel_pos.clone(), e.velocity.clone(), e.confidence.clone()))
        return outs

    a, b = rollout(1234), rollout(1234)
    for i, ((ra, va, ca), (rb, vb, cb)) in enumerate(zip(a, b)):
        assert torch.equal(ra, rb) and torch.equal(va, vb) and torch.equal(ca, cb), i
    print(f"\n[c] legacy 20-step rollout reproducible bit-for-bit under a fixed seed")


# ================================================================================================
# (d) measured: independent per-axis one-signed bias; band/mean stats; noise_scale scaling.
# ================================================================================================
def test_d_measured_per_axis_sign_band_and_independence():
    N, G = 40000, 1                                                       # many samples for stable stats
    gate_pos, gate_yaw = _straight_course(N, G)
    cfg = EE.EgoEstimatorConfig(bias_model="measured")
    est = _build(N, gate_pos, gate_yaw, cfg, seed=7)
    q = _identity_quat(N)
    pos = torch.zeros(N, 3, dtype=DT)
    vel = torch.zeros(N, 3, dtype=DT)
    est.reset_idx(torch.arange(N), pos, vel, q)

    lat = est._bias[:, 0, 0]                                              # gate-frame [lat, depth, vert]
    depth = est._bias[:, 0, 1]
    vert = est._bias[:, 0, 2]

    # ---- exact SIGN (one-signed per axis, per the measurement) ----
    assert (lat <= 0.0).all(), "lateral bias must be one-signed NEGATIVE (sigma.json -0.0338)"
    assert (vert >= 0.0).all(), "vertical bias must be one-signed POSITIVE (sigma.json +0.1949)"
    assert (depth <= 0.0).all(), "depth bias must be one-signed NEGATIVE (sigma.json -0.3344)"
    # ---- magnitude BANDS: |axis| in [0, |band|] ----
    assert (lat >= cfg.bias_band_lat).all() and (lat <= 0.0).all()
    assert (vert <= cfg.bias_band_vert).all() and (vert >= 0.0).all()
    assert (depth >= cfg.bias_band_depth).all() and (depth <= 0.0).all()
    # ---- per-axis MEANS ~ band/2 (magnitude ~ U[0,|band|]) ----
    for name, col, band in (("lat", lat, cfg.bias_band_lat), ("vert", vert, cfg.bias_band_vert),
                            ("depth", depth, cfg.bias_band_depth)):
        got, expect = col.mean().item(), band / 2.0
        assert abs(got - expect) < 0.01, f"{name} mean {got:.4f} != ~band/2 {expect:.4f}"
    # ---- INDEPENDENCE: axes are decorrelated (unlike legacy's lat==vert) ----
    def corr(a, bb):
        a0, b0 = a - a.mean(), bb - bb.mean()
        return (a0 * b0).mean() / (a0.std() * b0.std() + 1e-12)
    c_lv = corr(lat, vert).item()
    assert abs(c_lv) < 0.05, f"lat/vert must be INDEPENDENT (corr {c_lv:.3f}); legacy correlated them"
    assert not torch.equal(lat.abs(), vert.abs()), "measured lat/vert must be independent draws"
    print(f"\n[d] measured bias means lat/vert/depth = {lat.mean():.4f}/{vert.mean():.4f}/"
          f"{depth.mean():.4f} (~band/2 {cfg.bias_band_lat/2:.4f}/{cfg.bias_band_vert/2:.4f}/"
          f"{cfg.bias_band_depth/2:.4f}); corr(lat,vert)={c_lv:.4f}")


def test_d_measured_scales_with_noise_scale():
    """noise_scale multiplies the measured bias linearly: 0 -> exactly zero; 0.5 -> half the bands."""
    N, G = 20000, 1
    gate_pos, gate_yaw = _straight_course(N, G)
    q = _identity_quat(N)
    pos = torch.zeros(N, 3, dtype=DT)
    vel = torch.zeros(N, 3, dtype=DT)

    # noise_scale = 0 -> NO bias at all.
    cfg0 = EE.EgoEstimatorConfig(bias_model="measured", noise_scale=0.0)
    est0 = _build(N, gate_pos, gate_yaw, cfg0, seed=11)
    est0.reset_idx(torch.arange(N), pos, vel, q)
    assert est0._bias.abs().max().item() == 0.0, "noise_scale 0 -> zero measured bias"

    # noise_scale = 0.5 -> half-band means.
    cfg_half = EE.EgoEstimatorConfig(bias_model="measured", noise_scale=0.5)
    est_h = _build(N, gate_pos, gate_yaw, cfg_half, seed=12)
    est_h.reset_idx(torch.arange(N), pos, vel, q)
    for name, col_i, band in (("lat", 0, cfg_half.bias_band_lat), ("vert", 2, cfg_half.bias_band_vert),
                              ("depth", 1, cfg_half.bias_band_depth)):
        got = est_h._bias[:, 0, col_i].mean().item()
        expect = 0.5 * band / 2.0
        assert abs(got - expect) < 0.01, f"{name} half-scale mean {got:.4f} != {expect:.4f}"
    print(f"\n[d] noise_scale 0 -> zero bias; noise_scale 0.5 -> half-band means (linear)")


def test_d_measured_differs_from_legacy():
    """Same seed, same everything except bias_model: measured has NONZERO depth bias and DECORRELATED
    lat/vert, whereas legacy has depth==0 and lat==vert -- a structural (not just numeric) difference."""
    N, G = 5000, 1
    gate_pos, gate_yaw = _straight_course(N, G)
    q = _identity_quat(N)
    pos = torch.zeros(N, 3, dtype=DT)
    vel = torch.zeros(N, 3, dtype=DT)

    est_leg = _build(N, gate_pos, gate_yaw, EE.EgoEstimatorConfig(bias_model="legacy"), seed=5)
    est_leg.reset_idx(torch.arange(N), pos, vel, q)
    est_meas = _build(N, gate_pos, gate_yaw, EE.EgoEstimatorConfig(bias_model="measured"), seed=5)
    est_meas.reset_idx(torch.arange(N), pos, vel, q)

    assert est_leg._bias[:, 0, 1].abs().max().item() == 0.0, "legacy depth==0"
    assert est_meas._bias[:, 0, 1].abs().max().item() > 0.0, "measured depth NONZERO"
    assert torch.equal(est_leg._bias[:, 0, 0], est_leg._bias[:, 0, 2]), "legacy lat==vert"
    assert not torch.equal(est_meas._bias[:, 0, 0].abs(), est_meas._bias[:, 0, 2].abs()), \
        "measured lat/vert independent"
    print(f"\n[d] measured vs legacy: legacy depth=0 & lat==vert; measured depth!=0 & lat!=vert")
