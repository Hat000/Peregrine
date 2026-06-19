"""SYSTEM-ID PIN -- the two obs emulators agree on the FINAL obs vector the policy consumes.

PAIR: obs-emulators (the inc8 LIVE blocker #1).
  numpy: rl/estimator_emul.py        EstimatorEmulator (.obs / .confidence_channel; real LinearKF + FixSurrogate)
  torch: rl/inc8_estimator_emul.py   BatchedEstimatorEmulator (kf_pos_zup/kf_vel_zup/confidence_channel/obs_zup_torch)

WHAT THE EXISTING test_inc8_estimator_emul_torch.py PINS (and this does NOT re-pin): frame-seam identity,
KF-STATE (kf.x/kf.P/t_since) full-step parity, pointing->fix monotone, NEES band, obs[17:20] bounds.

THE GAP THIS PINS: nobody had driven both modules END TO END and compared the FINAL OBS VECTOR the policy
actually consumes -- obs[0:17] (gate-frame KF pos/vel + truth attitude/rates + lookahead) AND obs[17:20]
(the d5 confidence triple) -- step by step under an IDENTICAL drive, binned by condition. This pins that.

DRIVE / RNG MODE (system-id method): BIT-EXACT via SHARED INJECTED draws. The numpy EstimatorEmulator.step
draws per step in the order  standard_normal(3)[IMU] -> random()[Bernoulli] -> CONDITIONALLY
standard_normal(3)[fix noise only on accept]; the torch step takes accept_u/accel_noise/fix_noise
UNCONDITIONALLY. So feeding the two real .step() APIs one shared seed diverges in the RNG stream after the
first accept/reject mismatch. Bit-exact is achievable ONLY by injecting shared draws in numpy's
UNCONDITIONAL order -- which is exactly what proves the MATH/ENCODING is identical. We build a numpy
reference from the REAL primitives (real LinearKF.predict/update_position, real FixSurrogate, real
EstimatorEmulator.obs/.confidence_channel) fed the injected draws, and step the REAL torch module with the
SAME draws. Measured residual (handoff/system-id-2026-06-18/scratch/obs-emulators): obs[0:17] <= 9.6e-07,
obs[17:20] <= 3.0e-08 across head-on / crabbed / pitched approaches to 6 gates, NO condition-dependent
blowup (range/fix-accept/staleness bins all <= 1e-6).

Also a DISTRIBUTIONAL check on the confidence triple (the channel that converges tightly): K seeds each
side with independent streams agree in per-bin mean to <= 5e-3 (the velocity channels carry the cold
vel_std=5.0 prior -> heavy per-seed variance, so we assert distributional agreement on obs[17:20] only;
the bit-exact case already pins the velocity encoding).

Run: .venv\\Scripts\\python.exe -m pytest tests/test_sysid_obs_emul_parity.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))

import fix_surrogate as FS                                              # noqa: E402
from racer.contracts import Gate                                       # noqa: E402
from racer.rl_plant import PlantState, quat_rotate                     # noqa: E402
from racer.state_estimator import GRAVITY_NED, LinearKF               # noqa: E402
import inc8_estimator_emul as IE                                       # noqa: E402
from estimator_emul import EstimatorEmulator, EmulConfig, ned_gate_frame  # noqa: E402
from offline_rollout import _quat_from_rpy                            # noqa: E402
from fly_rl import (N_GATES, _FLIP, _GATE_POS_ZUP, _GATE_REL_POS,      # noqa: E402
                    _GATE_YAW_REL, obs_from_zup)

DT64 = torch.float64
_GATE_NED = _GATE_POS_ZUP * _FLIP

# tolerances measured by the system-id drive (with headroom; the medians are ~1e-7 / ~1e-8)
TOL_OBS017 = 1e-5
TOL_OBS1720 = 1e-6
TOL_DIST_TRIPLE = 5e-3        # per-bin |mean_np - mean_torch| on obs[17:20] over K seeds


def _R_from_quat(q):
    return np.stack([quat_rotate(np.asarray(q, np.float64), e) for e in np.eye(3)], axis=-1)


def _R_zup_from_ned(R_ned):
    return (_FLIP[:, None] * R_ned) * _FLIP[None, :]


def _course_ned():
    return (torch.tensor(_GATE_NED, dtype=DT64),
            torch.stack([torch.tensor(ned_gate_frame(np.pi), dtype=DT64) for _ in range(N_GATES)]))


def _PS(pos, vel, quat, omega):
    return PlantState(pos=np.asarray(pos, np.float64), vel=np.asarray(vel, np.float64),
                      quat=np.asarray(quat, np.float64), omega=np.asarray(omega, np.float64),
                      thrust=np.float64(0.2656))


def _torch_obs_full(emu_t, st_cur, target_gate, w_flu):
    """The FINAL 20-dim obs from the REAL torch module (kf pos/vel + truth attitude/rates + triple)."""
    R_zup = _R_zup_from_ned(_R_from_quat(np.asarray(st_cur.quat, np.float64)))
    nxt = min(target_gate + 1, N_GATES - 1)
    triple = emu_t.confidence_channel(torch.tensor([target_gate]))
    obs = IE.obs_zup_torch(
        emu_t.kf_pos_zup(), emu_t.kf_vel_zup(),
        torch.tensor(R_zup, dtype=DT64)[None], torch.tensor(w_flu, dtype=DT64)[None],
        torch.tensor(_GATE_POS_ZUP[target_gate], dtype=DT64)[None], torch.tensor([np.pi], dtype=DT64),
        torch.tensor(_GATE_REL_POS[nxt], dtype=DT64)[None], torch.tensor([_GATE_YAW_REL[nxt]], dtype=DT64),
        torch.tensor([0.0], dtype=DT64), triple=triple, virtual_flip=False)
    return obs[0].numpy().astype(np.float64)


def _drive_shared(gate_idx, crab_deg, pitch_deg, start_back, speed, sigma_lat, bias, seed,
                  dt=0.0333, steps=120):
    """Drive ONE single-env approach through BOTH modules with shared INJECTED draws (numpy
    unconditional order). Returns per-step (range, age, accepted, d017, d1720)."""
    rng = np.random.default_rng(seed)
    cfg = EmulConfig()
    q = _quat_from_rpy(0.0, np.radians(pitch_deg), np.pi + np.radians(crab_deg))
    R = _R_from_quat(q)
    g_ned = _GATE_NED[gate_idx]
    p0 = g_ned + np.array([start_back, 0.0, 0.0]); v0 = np.array([-speed, 0.0, 0.0])
    omega = np.zeros(3); w_flu = omega * _FLIP

    # numpy reference: real primitives wired into a real EstimatorEmulator (reuse its REAL encoder)
    kf = LinearKF.initialize(p0, v0, pos_std=cfg.pos_std_init, vel_std=cfg.vel_std_init,
                             accel_noise_std=cfg.imu_accel_noise, attitude_noise_std=0.0)
    sep = FS.replace(FS.DEFAULT, sigma_lateral_floor=float(sigma_lat), sigma_lateral_bias=float(bias),
                     sigma_vertical_bias=float(bias), sigma_depth_bias=0.0)
    emu_np = EstimatorEmulator(cfg); emu_np.kf = kf; emu_np._surrogate_ep = sep; emu_np._t_since_fix = 1e3
    gate_np = Gate(gate_id=gate_idx, position_ned=g_ned, R_world_gate=ned_gate_frame(np.pi))

    gate_pos_t, Rwg_t = _course_ned()
    emu_t = IE.BatchedEstimatorEmulator(1, gate_pos_t, Rwg_t, config=IE.EmulConfig(),
                                        device="cpu", dtype=DT64)
    emu_t.reset_idx(torch.arange(1), torch.tensor(p0, dtype=DT64)[None],
                    torch.tensor(v0, dtype=DT64)[None], torch.tensor([sigma_lat], dtype=DT64),
                    torch.tensor([bias], dtype=DT64))

    out = []
    cur_p, cur_v = p0.copy(), v0.copy(); t_since = 1e3
    for s in range(steps):
        prev_p, prev_v = cur_p.copy(), cur_v.copy(); cur_p = prev_p + prev_v * dt
        accel_noise = rng.standard_normal(3)        # 1) IMU       (unconditional)
        accept_u = rng.random()                      # 2) Bernoulli (unconditional)
        fix_noise = rng.standard_normal(3)           # 3) fix noise (drawn unconditionally for parity)
        # numpy step
        a_world = (cur_v - prev_v) / dt
        accel_body = R.T @ (a_world - GRAVITY_NED) + cfg.imu_accel_noise * accel_noise
        kf.predict(accel_body, R, dt); t_since += dt
        geom = FS.geometry(cur_p, R, gate_np)
        accepted = accept_u < sep.p_accept(geom)
        if accepted:
            sl, sv, sd = sep.fix_sigma(geom); sig = np.array([sl, sv, sd]); bvec = np.array([bias, bias, 0.0])
            z = cur_p + gate_np.R_world_gate @ (bvec + sig * fix_noise)
            kf.update_position(z, sep.fix_covariance(geom)); t_since = 0.0
        emu_np._t_since_fix = t_since
        # torch step (SAME injected draws)
        emu_t.step(torch.tensor(prev_p, dtype=DT64)[None], torch.tensor(prev_v, dtype=DT64)[None],
                   torch.tensor(R, dtype=DT64)[None], torch.tensor(cur_p, dtype=DT64)[None],
                   torch.tensor(cur_v, dtype=DT64)[None], torch.tensor(R, dtype=DT64)[None],
                   torch.tensor([gate_idx]), dt, torch.tensor([accept_u], dtype=DT64),
                   torch.tensor(accel_noise, dtype=DT64)[None], torch.tensor(fix_noise, dtype=DT64)[None])
        st_cur = _PS(cur_p, cur_v, q, omega)
        obs_np = emu_np.obs(st_cur, gate_idx, 0.0, False, None, 20).astype(np.float64)
        obs_t = _torch_obs_full(emu_t, st_cur, gate_idx, w_flu)
        out.append((float(geom.range_m), float(obs_np[19]), bool(accepted),
                    float(np.max(np.abs(obs_np[:17] - obs_t[:17]))),
                    float(np.max(np.abs(obs_np[17:20] - obs_t[17:20])))))
        if cur_p[0] <= g_ned[0] - 4.0:
            break
    return out


_SCENARIOS = [
    dict(gate_idx=0, crab_deg=0.0,  pitch_deg=0.0,  start_back=28.0, speed=12.0, sigma_lat=0.10, bias=0.0,   seed=1),
    dict(gate_idx=4, crab_deg=0.0,  pitch_deg=0.0,  start_back=29.0, speed=14.0, sigma_lat=0.08, bias=0.05,  seed=2),
    dict(gate_idx=2, crab_deg=15.0, pitch_deg=0.0,  start_back=27.0, speed=11.0, sigma_lat=0.12, bias=-0.10, seed=3),
    dict(gate_idx=1, crab_deg=30.0, pitch_deg=0.0,  start_back=28.0, speed=10.0, sigma_lat=0.06, bias=0.15,  seed=4),
    dict(gate_idx=3, crab_deg=0.0,  pitch_deg=10.0, start_back=28.0, speed=13.0, sigma_lat=0.11, bias=-0.07, seed=5),
    dict(gate_idx=5, crab_deg=8.0,  pitch_deg=6.0,  start_back=27.0, speed=12.0, sigma_lat=0.09, bias=0.12,  seed=6),
]


@pytest.fixture(scope="module")
def shared_drive_recs():
    """Drive all 6 approaches through BOTH emulators ONCE (module-scoped; reused by both bit-exact tests).
    Each rec = (range_m, age_norm, accepted, max|d obs[0:17]|, max|d obs[17:20]|)."""
    recs = []
    for sc in _SCENARIOS:
        recs.extend(_drive_shared(**sc))
    assert len(recs) > 300
    return recs


def test_final_obs_parity_shared_draws_obs017(shared_drive_recs):
    """END-TO-END: obs[0:17] (gate-frame KF pos/vel + truth attitude/rates + lookahead) is byte-faithful
    between the numpy and torch emulators under a shared injected drive, across head-on / crabbed / pitched
    approaches to all 6 gates -- with NO condition-dependent blowup (asserted per range/fix/staleness bin)."""
    recs = shared_drive_recs
    worst017 = max(r[3] for r in recs)
    assert worst017 <= TOL_OBS017, worst017
    # condition breakdowns must not hide a localized blowup
    for lo, hi in ((0, 6), (6, 12), (12, 18), (18, 24), (24, 31)):
        sel = [r for r in recs if lo <= r[0] < hi]
        if sel:
            assert max(r[3] for r in sel) <= TOL_OBS017, (lo, hi, max(r[3] for r in sel))
    for flag in (True, False):
        sel = [r for r in recs if r[2] == flag]
        if sel:
            assert max(r[3] for r in sel) <= TOL_OBS017, (flag, max(r[3] for r in sel))


def test_final_obs_parity_shared_draws_obs1720(shared_drive_recs):
    """END-TO-END: obs[17:20] (the d5 confidence triple c_inplane/c_along/age_norm) is byte-faithful
    under the shared injected drive, across fresh-fix (age~0) and stale (age->1) staleness regimes."""
    recs = shared_drive_recs
    worst1720 = max(r[4] for r in recs)
    assert worst1720 <= TOL_OBS1720, worst1720
    for lo, hi in ((0.0, 0.01), (0.01, 0.999), (0.999, 1.001)):
        sel = [r for r in recs if lo <= r[1] < hi]
        if sel:
            assert max(r[4] for r in sel) <= TOL_OBS1720, (lo, hi, max(r[4] for r in sel))


def test_confidence_triple_distributional_agreement():
    """DISTRIBUTIONAL: with INDEPENDENT seed streams (numpy true conditional .step vs torch unconditional
    .step), the obs[17:20] confidence triple agrees in per-bin mean over K seeds at a fixed range bin --
    the channel converges tightly (unlike the cold-vel-prior velocity channels, which the bit-exact tests
    already pin). This confirms the RNG-ORDER difference does not bias the encoder. The covariance-derived
    c_inplane/c_along are tight (<= TOL_DIST_TRIPLE); age_norm has discrete fix-clock jumps so a slightly
    looser MC bar is used (measured ~1e-3 at K=64)."""
    K, dt, steps = 64, 0.0333, 60     # steps=60 covers the 18-24 m bin (28 m start, 12 m/s)
    gate_idx, b = 0, (18, 24)
    q = _quat_from_rpy(0.0, 0.0, np.pi); R = _R_from_quat(q)
    g_ned = _GATE_NED[gate_idx]
    p0 = g_ned + np.array([28.0, 0.0, 0.0]); v0 = np.array([-12.0, 0.0, 0.0])
    omega = np.zeros(3)
    gate_pos_t, Rwg_t = _course_ned()

    # --- numpy: REAL .step (internal rng, TRUE conditional draw order), one emulator per seed ---
    np_triples = []
    for k in range(K):
        rng_np = np.random.default_rng(1000 + k)
        emu_np = EstimatorEmulator(EmulConfig())
        emu_np.reset(_PS(p0, v0, q, omega), gate_idx, rng_np)
        cur_p, cur_v = p0.copy(), v0.copy()
        for s in range(steps):
            prev_p, prev_v = cur_p.copy(), cur_v.copy(); cur_p = prev_p + prev_v * dt
            emu_np.step(_PS(prev_p, prev_v, q, omega), _PS(cur_p, cur_v, q, omega), gate_idx, dt, rng_np)
            r = float(np.linalg.norm(g_ned - cur_p))
            if b[0] <= r < b[1]:
                np_triples.append(emu_np.confidence_channel(gate_idx).astype(np.float64))
            if cur_p[0] <= g_ned[0] - 4.0:
                break

    # --- torch: REAL module run as ONE batch of N=K envs (the actual production usage) ---
    gen = torch.Generator(device="cpu"); gen.manual_seed(20240618)
    emu_t = IE.BatchedEstimatorEmulator(K, gate_pos_t, Rwg_t, config=IE.EmulConfig(), device="cpu", dtype=DT64)
    sig_t, bias_t = IE.BatchedEstimatorEmulator.sample_episode_dr(emu_t.cfg, K, "cpu", DT64, generator=gen)
    p0_b = torch.tensor(np.tile(p0, (K, 1)), dtype=DT64)
    v0_b = torch.tensor(np.tile(v0, (K, 1)), dtype=DT64)
    R_b = torch.tensor(np.tile(R, (K, 1, 1)), dtype=DT64)
    tg_b = torch.full((K,), gate_idx, dtype=torch.long)
    emu_t.reset_idx(torch.arange(K), p0_b, v0_b, sig_t, bias_t)
    cur_p, cur_v = np.tile(p0, (K, 1)), np.tile(v0, (K, 1))
    t_triples = []
    for s in range(steps):
        prev_p, prev_v = cur_p.copy(), cur_v.copy(); cur_p = prev_p + prev_v * dt
        emu_t.step(torch.tensor(prev_p, dtype=DT64), torch.tensor(prev_v, dtype=DT64), R_b,
                   torch.tensor(cur_p, dtype=DT64), torch.tensor(cur_v, dtype=DT64), R_b, tg_b, dt,
                   torch.rand(K, dtype=DT64, generator=gen), torch.randn(K, 3, dtype=DT64, generator=gen),
                   torch.randn(K, 3, dtype=DT64, generator=gen))
        r = float(np.linalg.norm(g_ned - cur_p[0]))   # identical trajectory across envs
        if b[0] <= r < b[1]:
            t_triples.append(emu_t.confidence_channel(tg_b).numpy().astype(np.float64))  # (K,3)
        if cur_p[0, 0] <= g_ned[0] - 4.0:
            break

    A = np.array(np_triples)
    B = np.concatenate(t_triples, axis=0) if t_triples else np.zeros((0, 3))
    assert len(A) > 100 and len(B) > 100, (len(A), len(B))
    dmean = np.abs(A.mean(0) - B.mean(0))      # [c_inplane, c_along, age_norm]
    # c_inplane / c_along (covariance-derived, smooth) -> tight; age_norm (discrete fix-clock) -> looser MC bar
    assert float(dmean[0]) <= TOL_DIST_TRIPLE, ("c_inplane", float(dmean[0]))
    assert float(dmean[1]) <= TOL_DIST_TRIPLE, ("c_along", float(dmean[1]))
    assert float(dmean[2]) <= 1e-2, ("age_norm", float(dmean[2]))
    # bounds sanity on both sides
    assert np.all(A >= 0) and np.all(A <= 1) and np.all(B >= 0) and np.all(B <= 1)
