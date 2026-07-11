"""rl/tools/eskf_pad_state.py -- measure the PAD-CONVERGED ESKF launch state (P_LAUNCH).

WHY: deploy boots the ESKF leveler ~100 s before takeoff and converges on the pad-idle accel
stream (a5: 101.16 s of bit-identical |a|=g samples), so at HANDOVER -- where a training episode
effectively starts -- the error covariance is NOT the cold-boot P0 = diag([1e-2]*3+[1e-6]*3):
the accel-observable attitude axes have collapsed onto the accel-update fixed point while the
accel-UNOBSERVABLE axes (yaw delta_phi_z, bias_z) have grown by the gyro random walk. Seeding
training episodes with the converged P (not cold P0) is the deploy-faithful takeoff state -- it
sets the early-flight chi2/S gate behaviour. (Graft from the design review: parity's
eskf_pad_state.py.)

METHOD: replay the a5 pad-idle window through the TRANSLATED leveler (rl/ego_ins_emul.py
BatchedESKFLeveler, N=1, NED/FRD mode, float64), level-seeded from the first accel sample
(deploy ahrs_adapter.level_seed_from_accel), stepped at a control-tick grid over the window
(deploy steps once per nav tick with the latest sample -- on the pad every sample is identical,
so only the dt sequence matters). The wire's pad-idle loop rate is not logged; we use the a5
flight median tick (57 ms) and report 33 ms / 100 ms alongside to show the result is
rate-insensitive at the attitude block (the accel fixed point) and mildly rate-sensitive only on
the unobservable-axis random walk (which the flight itself regrows anyway).

RESULT (run 2026-07-11, a5 dataset): the converged P is EXACTLY axisymmetric about the gravity
direction in body frame (eigvecs align with g_hat to 1e-16) -> baked into rl/ego_ins_emul.py as
the gravity-principal constants ESKF_P_LAUNCH_PHI_PERP/ALONG + ESKF_P_LAUNCH_BG_PERP/ALONG
(57 ms-tick replay; 33/100 ms reported alongside for the rate-sensitivity band), reconstructed
per env at reset by eskf_p_launch(g_hat_body). phi<->bias cross terms (max 1.4e-4, unobservable
axis only) dropped -- documented approximation.

Usage:
  python eskf_pad_state.py <handoff_dir> [tick_dt_s]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))            # rl/ on path
sys.path.insert(0, str(_HERE.parent))                # rl/tools on path

from ego_ins_emul import BatchedESKFLeveler, GRAVITY, quat_from_roll_pitch_yaw_zyx  # noqa: E402
from imu_foundation import load_highres_imu, pad_idle_stats                          # noqa: E402


def level_seed_from_accel(accel_frd: np.ndarray) -> tuple[float, float]:
    """deploy ahrs_adapter.level_seed_from_accel:134: g_body = -a/|a|;
    roll = atan2(g_y, g_z); pitch = atan2(-g_x, hypot(g_y, g_z)); yaw = 0."""
    gb = -accel_frd / max(np.linalg.norm(accel_frd), 1e-9)
    roll = math.atan2(gb[1], gb[2])
    pitch = math.atan2(-gb[0], math.hypot(gb[1], gb[2]))
    return roll, pitch


def replay_pad(accel_frd: np.ndarray, gyro_frd: np.ndarray, window_s: float, tick_dt: float):
    """Replay the (constant) pad sample through the translated leveler at tick_dt for window_s."""
    lev = BatchedESKFLeveler(1, g_world=[0.0, 0.0, +GRAVITY], dtype=torch.float64)
    roll, pitch = level_seed_from_accel(accel_frd)
    q0 = quat_from_roll_pitch_yaw_zyx(torch.tensor([roll], dtype=torch.float64),
                                      torch.tensor([pitch], dtype=torch.float64),
                                      torch.tensor([0.0], dtype=torch.float64))
    # deploy cold boot: P0 = diag([1e-2]*3 + [1e-6]*3), b0 = 0
    lev.reset_idx(torch.tensor([0]), q0, P0=(1e-2, 1e-2, 1e-2, 1e-6, 1e-6, 1e-6))
    gy = torch.tensor(gyro_frd, dtype=torch.float64).unsqueeze(0)
    ac = torch.tensor(accel_frd, dtype=torch.float64).unsqueeze(0)
    n_steps = int(window_s / tick_dt)
    duty = 0
    for _ in range(n_steps):
        lev.step(gy, ac, tick_dt)
        duty += int(lev.last_update_mask.item())
    return lev, n_steps, duty


def main():
    d = Path(sys.argv[1])
    tick_dt = float(sys.argv[2]) if len(sys.argv) > 2 else 0.057   # a5 flight median tick
    t_us, acc, gyr = load_highres_imu(str(d / "mavlink.tlog"))
    with open(d / "ego_obs.jsonl") as f:
        t_flight0 = json.loads(f.readline())["sim_time_ns"] / 1000.0
    pad, sl = pad_idle_stats(t_us, acc, gyr, t_flight0)
    a_pad = np.asarray(pad["accel_mean_m_s2"])                    # the (single) pad accel sample
    g_pad = np.asarray(pad["gyro_bias_rad_s"])                    # exactly zero on a5
    print(f"pad window: {pad['window_s']:.2f} s, sample accel {a_pad}, gyro {g_pad}")

    for dt in (tick_dt, 1.0 / 30.0, 0.1):
        lev, n, duty = replay_pad(a_pad, g_pad, pad["window_s"], dt)
        Pd = torch.diagonal(lev.P[0]).numpy()
        off = (lev.P[0] - torch.diag(torch.diagonal(lev.P[0]))).abs().max().item()
        print(f"\n--- tick_dt={dt*1e3:.1f} ms ({n} steps, accel-update duty {duty}/{n}) ---")
        print("P_LAUNCH diag:", " ".join(f"{v:.10e}" for v in Pd))
        print(f"max |off-diag| {off:.3e}   b_g {lev.b_g[0].numpy()}   "
              f"q {lev.q[0].numpy()}")


if __name__ == "__main__":
    main()
