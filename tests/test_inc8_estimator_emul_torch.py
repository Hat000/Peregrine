"""S3 PARITY GATE -- the batched torch estimator-emulation == numpy rl/estimator_emul.py.

Reproduces, in torch over N envs, the five GREEN conditions the numpy escape-hatch enforces:
  (a) FRAME-SEAM IDENTITY  -- KF seeded at truth -> the full obs == obs_from_truth (<=1e-4), pinning
      the NED<->Z-up<->gate-frame wiring AND the +L sign (a wrong flip is ~24 m).
  (b) FULL-STEP PARITY     -- under a SHARED injected draw (accel/accept/fix noise), the torch
      emulator's KF state == a numpy reconstruction of estimator_emul.step over an approach.
  (c) POINTING->FIX        -- accept density rises monotonically as the gate is centred (crab->0).
  (d) KF CALIBRATION       -- pooled NEES in [0.8,1.3] bias-off (the confidence channel is honest).
  (e) obs[17:20]           -- bounds [0,1] + responsiveness (age_norm->1 on dropout; c_inplane rises
      after a fix).
Run from repo ROOT: .venv\\Scripts\\python.exe -m pytest tests/test_inc8_estimator_emul_torch.py -q
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
from racer.rl_plant import quat_rotate                                 # noqa: E402
from racer.state_estimator import GRAVITY_NED, LinearKF               # noqa: E402
import inc8_estimator_emul as IE                                       # noqa: E402
from estimator_emul import ned_gate_frame                             # noqa: E402
from offline_rollout import obs_from_truth, _quat_from_rpy            # noqa: E402
from fly_rl import (N_GATES, _FLIP, _GATE_POS_ZUP, _GATE_REL_POS,      # noqa: E402
                    _GATE_YAW_REL, obs_from_zup)

DT64 = torch.float64
_GATE_NED = _GATE_POS_ZUP * _FLIP


def _R_from_quat(q):
    return np.stack([quat_rotate(np.asarray(q, np.float64), e) for e in np.eye(3)], axis=-1)


def _R_zup_from_ned(R_ned):
    return (_FLIP[:, None] * R_ned) * _FLIP[None, :]


def _course_ned():
    gate_pos = torch.tensor(_GATE_NED, dtype=DT64)
    Rwg = torch.stack([torch.tensor(ned_gate_frame(np.pi), dtype=DT64) for _ in range(N_GATES)])
    return gate_pos, Rwg


def _emu(n, cfg=None):
    gate_pos, Rwg = _course_ned()
    return IE.BatchedEstimatorEmulator(n, gate_pos, Rwg, config=cfg or IE.EmulConfig(),
                                       device="cpu", dtype=DT64)


# ============================================================ obs math == obs_from_zup
def test_obs_zup_torch_matches_numpy_obs_from_zup():
    """The torch obs builder reproduces fly_rl.obs_from_zup (the FULL 17-dim layout incl rpy_g +
    lookahead) over random Z-up states, for virtual_flip both ways."""
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(200):
        g = int(rng.integers(0, N_GATES))
        nxt = min(g + 1, N_GATES - 1)
        pos = rng.uniform(-160, 10, 3); vel = rng.uniform(-20, 20, 3)
        q = _quat_from_rpy(*rng.uniform([-1.0, -0.6, -np.pi], [1.0, 0.6, np.pi]))
        R_zup = _R_zup_from_ned(_R_from_quat(q))
        w = rng.uniform(-5, 5, 3); ln = float(rng.uniform(0, 5)); vflip = bool(rng.integers(0, 2))
        b = obs_from_zup(pos, vel, R_zup, w, g, ln, virtual_flip=vflip, gate_map=None)
        a = IE.obs_zup_torch(
            torch.tensor(pos, dtype=DT64)[None], torch.tensor(vel, dtype=DT64)[None],
            torch.tensor(R_zup, dtype=DT64)[None], torch.tensor(w, dtype=DT64)[None],
            torch.tensor(_GATE_POS_ZUP[g], dtype=DT64)[None], torch.tensor([np.pi], dtype=DT64),
            torch.tensor(_GATE_REL_POS[nxt], dtype=DT64)[None], torch.tensor([_GATE_YAW_REL[nxt]], dtype=DT64),
            torch.tensor([ln], dtype=DT64), triple=None, virtual_flip=vflip)[0].numpy()
        worst = max(worst, float(np.max(np.abs(a - b))))
    assert worst <= 1e-4, worst


def test_quat_xyzw_to_matrix_matches_rl_plant():
    """quat_xyzw_to_matrix_torch (the env's truth-attitude path) == rl_plant R_world_body."""
    rng = np.random.default_rng(2)
    worst = 0.0
    for _ in range(100):
        q = _quat_from_rpy(*rng.uniform([-1.0, -0.6, -np.pi], [1.0, 0.6, np.pi]))  # wxyz
        q_xyzw = np.array([q[1], q[2], q[3], q[0]])
        R = IE.quat_xyzw_to_matrix_torch(torch.tensor(q_xyzw, dtype=DT64)[None])[0].numpy()
        worst = max(worst, float(np.max(np.abs(R - _R_from_quat(q)))))
    assert worst <= 1e-9, worst


# ============================================================ (a) frame-seam identity
def _PS(pos, vel, quat, omega):
    from racer.rl_plant import PlantState
    return PlantState(pos=np.asarray(pos, np.float64), vel=np.asarray(vel, np.float64),
                      quat=np.asarray(quat, np.float64), omega=np.asarray(omega, np.float64),
                      thrust=np.float64(0.2656))


def test_frame_seam_identity_obs_equals_obs_from_truth():
    """KF seeded at truth -> full 17-dim obs == obs_from_truth over random states (<=1e-4). Pins the
    NED<->Z-up<->gate-frame wiring + the +L sign (a wrong flip is ~24 m). Also checks the 20-dim
    triple appends cleanly. Per-env virtual_flip mirrors the numpy GATE#1 sweep.

    Frame contract: obs_from_zup wants the Z-up FLU rate w_flu; obs_from_truth derives it as
    st.omega * _FLIP. We draw omega_flu (the env's self._w) and set st.omega = omega_flu * _FLIP so
    BOTH paths see the identical w_flu (the block [9:12])."""
    rng = np.random.default_rng(7)
    n = 250
    emu = _emu(n)
    pos = np.array([_GATE_NED[rng.integers(0, N_GATES)] + rng.uniform(-10, 10, 3) for _ in range(n)])
    vel = rng.uniform(-18, 18, (n, 3))
    omega_flu = rng.uniform(-5, 5, (n, 3))             # the env's Z-up FLU body rate (self._w)
    quats = np.array([_quat_from_rpy(*rng.uniform([-1.2, -0.7, -np.pi], [1.2, 0.7, np.pi])) for _ in range(n)])
    tg = rng.integers(0, N_GATES, n)
    ln = rng.uniform(0, 5, n)
    vflip = rng.integers(0, 2, n).astype(bool)
    idx = torch.arange(n)
    sig, bias = IE.BatchedEstimatorEmulator.sample_episode_dr(emu.cfg, n, "cpu", DT64)
    emu.reset_idx(idx, torch.tensor(pos, dtype=DT64), torch.tensor(vel, dtype=DT64), sig, bias)
    emu.seed_truth_idx(idx, torch.tensor(pos, dtype=DT64), torch.tensor(vel, dtype=DT64))

    R_zup = np.array([_R_zup_from_ned(_R_from_quat(q)) for q in quats])
    nxt = np.minimum(tg + 1, N_GATES - 1)
    triple = emu.confidence_channel(torch.tensor(tg))
    worst = 0.0
    for vf in (False, True):
        sel = vflip == vf
        if not sel.any():
            continue
        si = torch.tensor(np.nonzero(sel)[0])
        a = IE.obs_zup_torch(
            emu.kf_pos_zup()[si], emu.kf_vel_zup()[si], torch.tensor(R_zup[sel], dtype=DT64),
            torch.tensor(omega_flu[sel], dtype=DT64),
            torch.tensor(_GATE_POS_ZUP[tg[sel]], dtype=DT64), torch.tensor([np.pi] * int(sel.sum()), dtype=DT64),
            torch.tensor(_GATE_REL_POS[nxt[sel]], dtype=DT64), torch.tensor(_GATE_YAW_REL[nxt[sel]], dtype=DT64),
            torch.tensor(ln[sel], dtype=DT64), triple=None, virtual_flip=vf).numpy()
        for k, i in enumerate(np.nonzero(sel)[0]):
            b = obs_from_truth(_PS(pos[i], vel[i], quats[i], omega_flu[i] * _FLIP), int(tg[i]),
                               float(ln[i]), virtual_flip=vf, gate_map=None)
            worst = max(worst, float(np.max(np.abs(a[k] - b))))
    assert worst <= 1e-4, worst
    assert triple.shape == (n, 3)


# ============================================================ (b) full-step injected-randomness parity
def test_full_step_parity_injected_randomness():
    """Under a SHARED injected draw, the torch emulator's KF state == a numpy reconstruction of
    estimator_emul.step over a head-on approach (predict -> geometry -> fix -> update -> staleness)."""
    rng = np.random.default_rng(21)
    n, steps, dt = 24, 120, 0.0333
    cfg = IE.EmulConfig()
    emu = _emu(n, cfg)
    # per-env: a head-on approach to gate 0..5 with a random crab + start-back + sigma/bias draw
    gate_idx = rng.integers(0, N_GATES, n)
    crab = rng.uniform(-0.4, 0.4, n)
    v = rng.uniform(8.0, 16.0, n)
    start_back = rng.uniform(24.0, 30.0, n)
    sig = rng.uniform(0.05, 0.15, n)
    bias = rng.choice([-1.0, 1.0], n) * rng.uniform(0.0, 0.19, n)

    quats = np.array([_quat_from_rpy(0.0, 0.0, np.pi + crab[i]) for i in range(n)])
    R = np.array([_R_from_quat(q) for q in quats])
    p0 = np.array([_GATE_NED[gate_idx[i]] + np.array([start_back[i], 0.0, 0.0]) for i in range(n)])
    v0 = np.array([[-v[i], 0.0, 0.0] for i in range(n)])

    # numpy reconstruction of estimator_emul.step (the real LinearKF + FS primitives)
    nkfs, t_since, eps = [], [], []
    for i in range(n):
        nkfs.append(LinearKF.initialize(p0[i], v0[i], pos_std=cfg.pos_std_init,
                                        vel_std=cfg.vel_std_init, accel_noise_std=cfg.imu_accel_noise,
                                        attitude_noise_std=cfg.attitude_noise))
        t_since.append(1e3)
        eps.append(FS.replace(FS.DEFAULT, sigma_lateral_floor=float(sig[i]),
                              sigma_lateral_bias=float(bias[i]), sigma_vertical_bias=float(bias[i]),
                              sigma_depth_bias=0.0))

    emu.reset_idx(torch.arange(n), torch.tensor(p0, dtype=DT64), torch.tensor(v0, dtype=DT64),
                  torch.tensor(sig, dtype=DT64), torch.tensor(bias, dtype=DT64))

    cur_p = p0.copy()
    cur_v = v0.copy()
    for s in range(steps):
        prev_p, prev_v = cur_p.copy(), cur_v.copy()
        cur_p = prev_p + prev_v * dt        # constant-velocity head-on (R fixed)
        accel_noise = rng.standard_normal((n, 3))
        accept_u = rng.random(n)
        fix_noise = rng.standard_normal((n, 3))
        # numpy reconstruction
        for i in range(n):
            a_world = (cur_v[i] - prev_v[i]) / dt
            accel_body = R[i].T @ (a_world - GRAVITY_NED) + cfg.imu_accel_noise * accel_noise[i]
            nkfs[i].predict(accel_body, R[i], dt)
            t_since[i] += dt
            g = Gate(gate_id=int(gate_idx[i]), position_ned=_GATE_NED[gate_idx[i]],
                     R_world_gate=ned_gate_frame(np.pi))
            geom = FS.geometry(cur_p[i], R[i], g)
            p_acc = eps[i].p_accept(geom)
            if accept_u[i] < p_acc:
                sl, sv, sd = eps[i].fix_sigma(geom)
                sigv = np.array([sl, sv, sd]); bvec = np.array([bias[i], bias[i], 0.0])
                noise_gate = bvec + sigv * fix_noise[i]
                z = cur_p[i] + g.R_world_gate @ noise_gate
                nkfs[i].update_position(z, eps[i].fix_covariance(geom))
                t_since[i] = 0.0
        # torch
        emu.step(torch.tensor(prev_p, dtype=DT64), torch.tensor(prev_v, dtype=DT64),
                 torch.tensor(R, dtype=DT64), torch.tensor(cur_p, dtype=DT64),
                 torch.tensor(cur_v, dtype=DT64), torch.tensor(R, dtype=DT64),
                 torch.tensor(gate_idx), dt, torch.tensor(accept_u, dtype=DT64),
                 torch.tensor(accel_noise, dtype=DT64), torch.tensor(fix_noise, dtype=DT64))

    dx = float(np.max(np.abs(emu.kf.x.numpy() - np.array([k.x for k in nkfs]))))
    dP = float(np.max(np.abs(emu.kf.P.numpy() - np.array([k.P for k in nkfs]))))
    dt_since = float(np.max(np.abs(emu._t_since_fix.numpy() - np.array(t_since))))
    assert dx <= 1e-4, dx
    assert dP <= 1e-4, dP
    assert dt_since <= 1e-9, dt_since


# ============================================================ (c) pointing -> fix-rate monotone
def test_pointing_fix_rate_monotone_in_crab():
    """As the gate is centred (crab -> 0) the per-env accept probability rises monotonically; a
    far-off-pointed gate leaves the frame and almost never fixes (the camera-pointing lever)."""
    crabs = np.array([0, 10, 20, 30, 40, 50, 60, 70], dtype=float)
    n = len(crabs)
    emu = _emu(n)
    gate0 = _GATE_NED[0]
    drone = np.array([gate0 + np.array([20.0, 0.0, 0.0]) for _ in range(n)])
    R = np.array([_R_from_quat(_quat_from_rpy(0.0, 0.0, np.pi + np.radians(c))) for c in crabs])
    gp = emu.gate_pos_ned[torch.zeros(n, dtype=torch.long)]
    Rwg = emu.R_world_gate[torch.zeros(n, dtype=torch.long)]
    geom = IE.batched_geometry(torch.tensor(drone, dtype=DT64), torch.tensor(R, dtype=DT64),
                               gp, Rwg, emu.R_cb, emu.K)
    p = emu.surrogate.p_accept(geom["range"], geom["in_image"]).numpy()
    assert all(p[i] + 1e-9 >= p[i + 1] for i in range(n - 1)), list(zip(crabs, p))
    assert p[0] > 0.5 and p[-1] < 0.05, p
    assert bool(geom["in_image"][0]) and not bool(geom["in_image"][-1])


# ============================================================ (d) NEES calibration (bias off)
def test_kf_calibration_nees_in_band_bias_off():
    """Pooled mean(e_g^2 / diag(P_gate)) in [0.8,1.3] over a batch of head-on approaches (bias-off):
    the confidence channel is HONEST, not overconfident."""
    rng = np.random.default_rng(3)
    n, dt = 200, 0.0333
    cfg = IE.EmulConfig(inject_bias=False)
    emu = _emu(n, cfg)
    v = 12.0
    quats = np.array([_quat_from_rpy(0.0, 0.0, np.pi) for _ in range(n)])
    R = np.array([_R_from_quat(q) for q in quats])
    gate_idx = rng.integers(0, N_GATES, n)
    p0 = np.array([_GATE_NED[gate_idx[i]] + np.array([26.0, 0.0, 0.0]) for i in range(n)])
    v0 = np.tile([-v, 0.0, 0.0], (n, 1))
    sig, bias = IE.BatchedEstimatorEmulator.sample_episode_dr(cfg, n, "cpu", DT64)
    emu.reset_idx(torch.arange(n), torch.tensor(p0, dtype=DT64), torch.tensor(v0, dtype=DT64), sig, bias)
    cur_p, cur_v = p0.copy(), v0.copy()
    nees = []
    tg = torch.tensor(gate_idx)
    for s in range(140):
        prev_p, prev_v = cur_p.copy(), cur_v.copy()
        cur_p = prev_p + prev_v * dt
        emu.step(torch.tensor(prev_p, dtype=DT64), torch.tensor(prev_v, dtype=DT64),
                 torch.tensor(R, dtype=DT64), torch.tensor(cur_p, dtype=DT64),
                 torch.tensor(cur_v, dtype=DT64), torch.tensor(R, dtype=DT64), tg, dt,
                 torch.rand(n, dtype=DT64), torch.randn(n, 3, dtype=DT64), torch.randn(n, 3, dtype=DT64))
        if s > 25:
            # per-axis NEES = e_g^2 / diag(P_gate), in the NED gate frame
            Rwg = emu.R_world_gate[tg]
            e_world = emu.kf.position - torch.tensor(cur_p, dtype=DT64)
            e_g = torch.einsum("nij,nj->ni", Rwg.transpose(-1, -2), e_world)
            P_gate = Rwg.transpose(-1, -2) @ emu.kf.P[:, :3, :3] @ Rwg
            d = torch.stack([P_gate[:, 0, 0], P_gate[:, 1, 1], P_gate[:, 2, 2]], dim=-1).clamp(min=1e-12)
            # only count envs that have received a fix (t_since_fix finite-ish)
            got = emu._t_since_fix < 1.0
            if got.any():
                nees.append(((e_g ** 2 / d)[got]).reshape(-1).numpy())
    nees = np.concatenate(nees)
    assert nees.size > 500, nees.size
    pooled = float(nees.mean())
    assert 0.8 <= pooled <= 1.3, pooled


# ============================================================ (e) obs[17:20] bounds + responsiveness
def test_obs1720_bounds_and_responsiveness():
    """obs[17:20] all in [0,1]; c_inplane rises across an accepted fix; age_norm -> 1 under a
    sustained fix-dropout (coast far past the gate)."""
    dt = 0.0333
    cfg = IE.EmulConfig(inject_bias=False)
    emu = _emu(1, cfg)
    gate0 = _GATE_NED[0]
    R = _R_from_quat(_quat_from_rpy(0.0, 0.0, np.pi))[None]
    p = (gate0 + np.array([26.0, 0.0, 0.0]))[None]
    vel = np.array([[-10.0, 0.0, 0.0]])
    emu.reset_idx(torch.arange(1), torch.tensor(p, dtype=DT64), torch.tensor(vel, dtype=DT64),
                  torch.tensor([0.10], dtype=DT64), torch.tensor([0.0], dtype=DT64))
    rng = np.random.default_rng(5)
    tg = torch.zeros(1, dtype=torch.long)
    saw_rise = False
    c_pre = None
    cur_p, cur_v = p.copy(), vel.copy()
    for s in range(160):
        prev_p, prev_v = cur_p.copy(), cur_v.copy()
        cur_p = prev_p + prev_v * dt
        c_before = float(emu.confidence_channel(tg)[0, 0])
        acc = emu.step(torch.tensor(prev_p, dtype=DT64), torch.tensor(prev_v, dtype=DT64),
                       torch.tensor(R, dtype=DT64), torch.tensor(cur_p, dtype=DT64),
                       torch.tensor(cur_v, dtype=DT64), torch.tensor(R, dtype=DT64), tg, dt,
                       torch.tensor(rng.random(1), dtype=DT64), torch.tensor(rng.standard_normal((1, 3)), dtype=DT64),
                       torch.tensor(rng.standard_normal((1, 3)), dtype=DT64))
        triple = emu.confidence_channel(tg)[0].numpy()
        assert np.all(triple >= 0.0) and np.all(triple <= 1.0), triple
        if bool(acc[0]) and c_pre is not None and triple[0] >= c_pre - 1e-9:
            saw_rise = True
        c_pre = c_before
        if cur_p[0, 0] <= gate0[0]:
            break
    assert saw_rise, "c_inplane never rose across an accepted fix"
    # induced dropout: coast far past with no possible fix -> age_norm saturates to 1. accept_u=1.0
    # FORCES rejection (numpy semantics: reject iff u >= p, p<=0.84) -- NOT zeros, which would force
    # an accept since 0 < p_out_of_image.
    far = (gate0 + np.array([-60.0, 0.0, 0.0]))[None]
    cur_p, cur_v = far.copy(), vel.copy()
    for _ in range(20):
        prev_p, prev_v = cur_p.copy(), cur_v.copy()
        cur_p = prev_p + prev_v * dt
        emu.step(torch.tensor(prev_p, dtype=DT64), torch.tensor(prev_v, dtype=DT64),
                 torch.tensor(R, dtype=DT64), torch.tensor(cur_p, dtype=DT64),
                 torch.tensor(cur_v, dtype=DT64), torch.tensor(R, dtype=DT64), tg, dt,
                 torch.ones(1, dtype=DT64), torch.zeros(1, 3, dtype=DT64), torch.zeros(1, 3, dtype=DT64))
    assert float(emu.confidence_channel(tg)[0, 2]) == pytest.approx(1.0)
