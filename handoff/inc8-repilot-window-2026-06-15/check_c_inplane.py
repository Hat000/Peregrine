"""Diagnostic (iteration 4): does the emulator KF c_inplane stay HIGH without fixes (#74-style
over-convergence gaming), or does it honestly report LOW confidence on a no-fix coast?

Traces confidence_channel.c_inplane for the BatchedEstimatorEmulator over:
  (A) a COLD no-fix coast (accept_u=1 -> u<p never true -> no accepted fix), and
  (B) one FORCED accepted fix at fixable range, then a no-fix coast (how fast does c_inplane decay?).

Run from repo ROOT: .venv\\Scripts\\python.exe handoff/inc8-repilot-window-2026-06-15/check_c_inplane.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))

import inc8_estimator_emul as IE                      # noqa: E402
from estimator_emul import ned_gate_frame            # noqa: E402
from offline_rollout import _quat_from_rpy           # noqa: E402
from racer.rl_plant import quat_rotate               # noqa: E402
from fly_rl import N_GATES, _FLIP, _GATE_POS_ZUP     # noqa: E402

DT = torch.float64
_GATE_NED = _GATE_POS_ZUP * _FLIP


def _R_from_quat(q):
    return np.stack([quat_rotate(np.asarray(q, np.float64), e) for e in np.eye(3)], axis=-1)


def _build(n):
    gate_pos = torch.tensor(_GATE_NED, dtype=DT)
    Rwg = torch.stack([torch.tensor(ned_gate_frame(np.pi), dtype=DT) for _ in range(N_GATES)])
    return IE.BatchedEstimatorEmulator(n, gate_pos, Rwg, config=IE.EmulConfig(), device="cpu", dtype=DT)


def _trace(force_one_fix_at_step=None, steps=70, gate=3, v=12.0):
    n = 256
    emu = _build(n)
    gate_ned = _GATE_NED[gate]
    q = _quat_from_rpy(0.0, 0.0, np.pi)           # head-on, centred (in_image true)
    R = _R_from_quat(q)
    R_t = torch.tensor(np.broadcast_to(R, (n, 3, 3)).copy(), dtype=DT)
    p = np.broadcast_to(gate_ned + np.array([30.0, 0.0, 0.0]), (n, 3)).copy()
    vel = np.broadcast_to([-v, 0.0, 0.0], (n, 3)).copy()
    tg = torch.full((n,), gate, dtype=torch.long)
    sig, bias = IE.BatchedEstimatorEmulator.sample_episode_dr(emu.cfg, n, "cpu", DT)
    emu.reset_idx(torch.arange(n), torch.tensor(p, dtype=DT), torch.tensor(vel, dtype=DT), sig, bias)
    dt = 0.0333
    rows = []
    for s in range(steps):
        prev_p = p.copy()
        p = prev_p + vel * dt
        rng = float(np.linalg.norm(p[0] - gate_ned))
        # accept_u: 1.0 => u<p never true => NO fix (coast). At the forced-fix step, 0.0 => accept.
        au = torch.ones(n, dtype=DT)
        if force_one_fix_at_step is not None and s == force_one_fix_at_step:
            au = torch.zeros(n, dtype=DT)
        accepted = emu.step(
            torch.tensor(prev_p, dtype=DT), torch.tensor(vel, dtype=DT), R_t,
            torch.tensor(p, dtype=DT), torch.tensor(vel, dtype=DT), R_t, tg, dt,
            au, torch.zeros(n, 3, dtype=DT), torch.zeros(n, 3, dtype=DT))
        triple = emu.confidence_channel(tg)
        rows.append((s, rng, float(accepted.float().mean()),
                     float(triple[:, 0].mean()), float(triple[:, 2].mean())))   # c_inplane, age_norm
        if p[0, 0] <= gate_ned[0]:
            break
    return rows


def _print(title, rows):
    print(f"\n=== {title} ===")
    print(f"{'step':>4} {'range_m':>8} {'fix_rate':>9} {'c_inplane':>10} {'age_norm':>9}")
    for s, rng, fr, cip, age in rows:
        if s % 8 == 0 or fr > 0:
            print(f"{s:>4} {rng:>8.2f} {fr:>9.4f} {cip:>10.4f} {age:>9.4f}")


if __name__ == "__main__":
    print("EmulConfig: pos_std_init=%.2f sigma_ref=%.3f tau_stale=%.3f accel_noise=%.2f" % (
        IE.EmulConfig().pos_std_init, IE.EmulConfig().sigma_ref,
        IE.EmulConfig().tau_stale, IE.EmulConfig().imu_accel_noise))
    a = _trace(force_one_fix_at_step=None)
    _print("(A) COLD no-fix coast -- c_inplane should stay LOW (~0.05) if HONEST", a)
    cmax_nofix = max(r[3] for r in a)
    print(f"  -> max c_inplane over the no-fix coast = {cmax_nofix:.4f}")

    b = _trace(force_one_fix_at_step=10)
    _print("(B) ONE forced fix @ step 10, then coast -- does c_inplane decay after the fix?", b)
    post = [r for r in b if r[0] >= 10]
    if post:
        cmax_post = max(r[3] for r in post)
        ctail = post[-1][3]
        print(f"  -> peak c_inplane after the fix = {cmax_post:.4f}; tail (coast end) = {ctail:.4f}")

    print("\nVERDICT: c_inplane is GAMED (over-converges, high without fixes) IFF max-no-fix c_inplane is"
          " HIGH (>~0.5). If it stays ~0.05 cold and only rises transiently after a real fix then decays,"
          " it is HONEST -> no covariance floor needed; the lever is conf_shape magnitude + age_norm.")
