"""Measure the ACTUAL torch<->numpy parity margins (S1-S4) for the REPORT (float64 + float32)."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))

import fix_surrogate as FS
from racer.contracts import Gate
from racer.rl_plant import quat_rotate
from racer.state_estimator import GRAVITY_NED, LinearKF
from racer.reference_line import ReferenceLine
import inc8_estimator_emul as IE
from reference_line_torch import BatchedReferenceLine
from estimator_emul import ned_gate_frame
from offline_rollout import obs_from_truth, _quat_from_rpy
from fly_rl import N_GATES, _FLIP, _GATE_POS_ZUP

_GATE_NED = _GATE_POS_ZUP * _FLIP


def _R(q):
    return np.stack([quat_rotate(np.asarray(q, np.float64), e) for e in np.eye(3)], axis=-1)


def s1(dtype):
    rng = np.random.default_rng(1)
    n = 256
    flip = np.array([1.0, -1.0, -1.0])
    drones, R_wbs, gp, Rwg = [], [], [], []
    for _ in range(n):
        gi = rng.integers(0, 6)
        g = _GATE_NED[gi]
        drones.append(g + rng.uniform(-30, 30, 3))
        R_wbs.append(_R(_quat_from_rpy(*rng.uniform([-.8, -.6, -np.pi], [.8, .6, np.pi]))))
        gp.append(g); Rwg.append(ned_gate_frame(np.pi))
    drones, R_wbs, gp, Rwg = map(np.asarray, (drones, R_wbs, gp, Rwg))
    R_cb, K, _, _ = IE._const("cpu", dtype)
    tg = IE.batched_geometry(torch.tensor(drones, dtype=dtype), torch.tensor(R_wbs, dtype=dtype),
                             torch.tensor(gp, dtype=dtype), torch.tensor(Rwg, dtype=dtype), R_cb, K)
    geoms = [FS.geometry(d, Rwb, Gate(0, g, R)) for d, Rwb, g, R in zip(drones, R_wbs, gp, Rwg)]
    rng_err = np.max(np.abs(tg["range"].numpy() - np.array([gg.range_m for gg in geoms])))
    p = FS.BatchedFixSurrogate if False else IE.BatchedFixSurrogate(IE.TorchSurrogateParams(), "cpu", dtype)
    pa = p.p_accept(tg["range"], tg["in_image"]).numpy()
    pa_np = np.array([FS.DEFAULT.p_accept(gg) for gg in geoms])
    img_match = np.array_equal(tg["in_image"].numpy(), np.array([gg.in_image for gg in geoms]))
    return rng_err, np.max(np.abs(pa - pa_np)), img_match


def s2(dtype):
    rng = np.random.default_rng(7)
    n, steps = 96, 60
    dt = 0.0333
    p0 = rng.uniform(-50, 10, (n, 3)); v0 = rng.uniform(-18, 18, (n, 3))
    nkfs = [LinearKF.initialize(p0[i], v0[i], pos_std=1, vel_std=5, accel_noise_std=0.3,
                                attitude_noise_std=0.0) for i in range(n)]
    bkf = IE.BatchedLinearKF(n, "cpu", dtype, accel_noise_std=0.3, attitude_noise_std=0.0)
    bkf.initialize_idx(torch.arange(n), torch.tensor(p0, dtype=dtype), torch.tensor(v0, dtype=dtype), 1, 5)
    worst_x = worst_P = 0.0
    for s in range(steps):
        accel = rng.standard_normal((n, 3)) * 4.0
        Rs = np.array([np.linalg.qr(rng.standard_normal((3, 3)))[0] for _ in range(n)])
        Rs = np.array([Q if np.linalg.det(Q) > 0 else Q @ np.diag([-1, 1, 1.]) for Q in Rs])
        for i in range(n):
            nkfs[i].predict(accel[i], Rs[i], dt)
        bkf.predict(torch.tensor(accel, dtype=dtype), torch.tensor(Rs, dtype=dtype), dt)
        if s % 3 == 1:
            idx = np.nonzero(rng.random(n) < 0.6)[0]
            if idx.size:
                z = rng.uniform(-50, 10, (idx.size, 3))
                covs = np.array([(lambda L: L @ L.T + 0.05 * np.eye(3))(rng.standard_normal((3, 3)) * 0.2)
                                 for _ in range(idx.size)])
                for k, i in enumerate(idx):
                    nkfs[i].update_position(z[k], covs[k])
                bkf.update_position_idx(torch.tensor(idx), torch.tensor(z, dtype=dtype),
                                        torch.tensor(covs, dtype=dtype))
        worst_x = max(worst_x, float(np.max(np.abs(bkf.x.numpy() - np.array([k.x for k in nkfs])))))
        worst_P = max(worst_P, float(np.max(np.abs(bkf.P.numpy() - np.array([k.P for k in nkfs])))))
    return worst_x, worst_P


def s3(dtype):
    # frame-seam identity worst case
    rng = np.random.default_rng(7)
    n = 250
    gate_pos = torch.tensor(_GATE_NED, dtype=dtype)
    Rwg = torch.stack([torch.tensor(ned_gate_frame(np.pi), dtype=dtype) for _ in range(N_GATES)])
    emu = IE.BatchedEstimatorEmulator(n, gate_pos, Rwg, config=IE.EmulConfig(), device="cpu", dtype=dtype)
    pos = np.array([_GATE_NED[rng.integers(0, N_GATES)] + rng.uniform(-10, 10, 3) for _ in range(n)])
    vel = rng.uniform(-18, 18, (n, 3))
    om = rng.uniform(-5, 5, (n, 3))
    quats = np.array([_quat_from_rpy(*rng.uniform([-1.2, -.7, -np.pi], [1.2, .7, np.pi])) for _ in range(n)])
    tg = rng.integers(0, N_GATES, n); ln = rng.uniform(0, 5, n)
    sig, bias = IE.BatchedEstimatorEmulator.sample_episode_dr(emu.cfg, n, "cpu", dtype)
    emu.reset_idx(torch.arange(n), torch.tensor(pos, dtype=dtype), torch.tensor(vel, dtype=dtype), sig, bias)
    emu.seed_truth_idx(torch.arange(n), torch.tensor(pos, dtype=dtype), torch.tensor(vel, dtype=dtype))
    R_zup = np.array([(_FLIP[:, None] * _R(q)) * _FLIP[None, :] for q in quats])
    nxt = np.minimum(tg + 1, N_GATES - 1)
    from fly_rl import _GATE_REL_POS, _GATE_YAW_REL
    worst = 0.0
    a = IE.obs_zup_torch(
        emu.kf_pos_zup(), emu.kf_vel_zup(), torch.tensor(R_zup, dtype=dtype), torch.tensor(om, dtype=dtype),
        torch.tensor(_GATE_POS_ZUP[tg], dtype=dtype), torch.full((n,), np.pi, dtype=dtype),
        torch.tensor(_GATE_REL_POS[nxt], dtype=dtype), torch.tensor(_GATE_YAW_REL[nxt], dtype=dtype),
        torch.tensor(ln, dtype=dtype)).numpy()
    for i in range(n):
        from racer.rl_plant import PlantState
        st = PlantState(pos=pos[i], vel=vel[i], quat=quats[i], omega=om[i] * _FLIP, thrust=np.float64(0.2656))
        b = obs_from_truth(st, int(tg[i]), float(ln[i]), virtual_flip=False, gate_map=None)
        worst = max(worst, float(np.max(np.abs(a[i] - b))))
    return worst


def s4(dtype):
    nrl = ReferenceLine.load(str(ROOT / "rl" / "reference_line_inc8.json"))
    brl = BatchedReferenceLine.load(str(ROOT / "rl" / "reference_line_inc8.json"), "cpu", dtype)
    rng = np.random.default_rng(4)
    M = nrl.pos.shape[0]
    q = nrl.pos[rng.integers(0, M, 4000)] + rng.uniform(-3, 3, (4000, 3))
    st = brl.progress(torch.tensor(q, dtype=dtype)).numpy().astype(np.float64)
    sn = np.array([nrl.progress(p) for p in q])
    return np.max(np.abs(st - sn)), np.percentile(np.abs(st - sn), 99)


for name, dt in [("float64", torch.float64), ("float32", torch.float32)]:
    print(f"==== {name} ====")
    r, pa, img = s1(dt)
    print(f"  S1 geometry range worst={r:.2e}  p_accept worst={pa:.2e}  in_image_exact={img}")
    wx, wP = s2(dt)
    print(f"  S2 KF  x worst={wx:.2e}  P worst={wP:.2e}")
    print(f"  S3 frame-seam identity worst={s3(dt):.2e}")
    mx, p99 = s4(dt)
    print(f"  S4 refline progress worst={mx:.2e}  p99={p99:.2e}")
