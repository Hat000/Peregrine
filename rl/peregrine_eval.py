"""Stage-1 OFFLINE twin sanity: roll the trained actor in PeregrineRacing from the STANDING START
and report (a) 6/6 success rate and (b) PEAK ROLL -- the two checks the S1.3 retrain must pass before
live validation (the inc-1 "backflip-diver" rolled 104-126 deg before every gate; S1.3 must keep peak
roll < ~80 deg so it never enters the sim's untwinned +-180 deg yaw-spin regime).

Runs the EXACT training substrate (DiffAero env + our system-ID'd torch plant), but in a CLEAN twin:
``standing_start_frac=1.0`` (every reset spawns at the real 23.3 m start) and ``dynamics.dr=False``
(nominal plant -- we are checking the policy, not its DR robustness). Mirrors diffaero's TestRunner
inner loop (agent.act test=True -> env.rescale_action -> env.step) and reads attitude straight from
env.dynamics._q each step.

Usage (on Adroit, diffaero env active, PYTHONPATH = repo/rl:repo/src:diffaero):
  python peregrine_eval.py --ckpt <run>/checkpoints --n-envs 256 --max-time 40
``--ckpt`` is the directory containing actor.pth; the training cfg is read from its run's
.hydra/config.yaml (walked up from --ckpt), exactly as diffaero's script/test.py does.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
import pytorch3d.transforms as T

# register OUR injections (zero clone edits), identical to peregrine_train_racing.py
import diffaero.dynamics as _dyn
import diffaero.env as _env
from diffaero_dynamics import PeregrinePlantDynamics
from peregrine_racing import PeregrineRacing

_dyn.DYNAMICS_ALIAS["peregrine_plant"] = lambda cfg, device: PeregrinePlantDynamics(
    cfg, device, backend="torch")
_env.ENV_ALIAS["peregrine_racing"] = PeregrineRacing

from diffaero.env import build_env          # noqa: E402
from diffaero.algo import build_agent        # noqa: E402

N_GATES = 6


def find_run_root(ckpt: Path) -> Path:
    p = ckpt
    while p != p.parent:
        if (p / ".hydra" / "config.yaml").exists():
            return p
        p = p.parent
    raise SystemExit(f"no .hydra/config.yaml found walking up from {ckpt}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="dir containing actor.pth")
    ap.add_argument("--n-envs", type=int, default=256)
    ap.add_argument("--max-time", type=float, default=40.0)
    ap.add_argument("--horizons", type=float, default=2.5,
                    help="run this many max_time horizons to gather completed episodes")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    ckpt = Path(args.ckpt).resolve()
    run_root = find_run_root(ckpt)
    cfg = OmegaConf.load(run_root / ".hydra" / "config.yaml")
    print(f"[eval] run_root={run_root}  ckpt={ckpt}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # clean standing-start twin: force the start, kill DR, set env size + horizon
    OmegaConf.update(cfg, "n_envs", args.n_envs, force_add=True)
    OmegaConf.update(cfg, "env.standing_start_frac", 1.0, force_add=True)
    OmegaConf.update(cfg, "dynamics.dr", False, force_add=True)
    OmegaConf.update(cfg, "env.max_time", float(args.max_time), force_add=True)

    env = build_env(cfg.env, device=device)
    agent = build_agent(cfg.algo, env, device)
    agent.load(ckpt)
    dt = float(cfg.env.dt)
    print(f"[eval] n_envs={args.n_envs} dt={dt} max_time={args.max_time} "
          f"backend=torch dr=False standing_start_frac=1.0")

    obs = env.reset()
    n = args.n_envs
    cur_peak_roll = torch.zeros(n, device=device)
    cur_peak_tilt = torch.zeros(n, device=device)
    ep_success, ep_roll, ep_tilt = [], [], []

    n_steps = int(args.horizons * args.max_time / dt)
    body_up = torch.tensor([0.0, 0.0, 1.0], device=device)
    with torch.no_grad():
        for _ in range(n_steps):
            action, _ = agent.act(obs, test=True)
            action = env.rescale_action(action)
            obs, _loss, _term, info = env.step(action)

            q = env.dynamics._q                                  # XYZW (Z-up)
            R = T.quaternion_to_matrix(q.roll(1, dims=-1)).clamp(-1 + 1e-6, 1 - 1e-6)
            _, _, roll = T.matrix_to_euler_angles(R, "ZYX").unbind(-1)
            # total tilt = angle of body-up from world-up (sign-free; >90 = inverted regime)
            up_w = torch.matmul(R, body_up)
            tilt = torch.arccos(up_w[..., 2].clamp(-1 + 1e-6, 1 - 1e-6))
            cur_peak_roll = torch.maximum(cur_peak_roll, roll.abs())
            cur_peak_tilt = torch.maximum(cur_peak_tilt, tilt)

            reset = info["reset"]
            if reset.any():
                ridx = reset.nonzero().flatten()
                succ = info["success"]
                for i in ridx.tolist():
                    ep_success.append(bool(succ[i]))
                    ep_roll.append(float(cur_peak_roll[i]))
                    ep_tilt.append(float(cur_peak_tilt[i]))
                cur_peak_roll[ridx] = 0.0
                cur_peak_tilt[ridx] = 0.0

    succ = np.array(ep_success)
    roll = np.degrees(np.array(ep_roll))
    tilt = np.degrees(np.array(ep_tilt))
    n_ep = len(succ)
    sr = float(succ.mean()) if n_ep else 0.0
    print(f"\n[RESULT] episodes={n_ep}  SUCCESS_RATE_6of6={sr:.3f}")
    if n_ep:
        print(f"[RESULT] PEAK_ROLL_DEG all:     max {roll.max():6.1f}  p90 {np.percentile(roll,90):6.1f}  median {np.median(roll):6.1f}")
        print(f"[RESULT] PEAK_TILT_DEG all:     max {tilt.max():6.1f}  p90 {np.percentile(tilt,90):6.1f}  median {np.median(tilt):6.1f}")
    if succ.any():
        rs, ts = roll[succ], tilt[succ]
        print(f"[RESULT] PEAK_ROLL_DEG success: max {rs.max():6.1f}  p90 {np.percentile(rs,90):6.1f}  median {np.median(rs):6.1f}")
        print(f"[RESULT] PEAK_TILT_DEG success: max {ts.max():6.1f}  p90 {np.percentile(ts,90):6.1f}  median {np.median(ts):6.1f}")
    # machine-readable summary line
    print(f"EVAL_SUMMARY sr={sr:.3f} n_ep={n_ep} "
          f"roll_succ_max={ (roll[succ].max() if succ.any() else -1):.1f} "
          f"roll_all_p90={ (np.percentile(roll,90) if n_ep else -1):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
