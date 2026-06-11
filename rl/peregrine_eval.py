"""S1.4 OFFLINE twin eval: roll a trained actor in PeregrineRacing and report the acceptance
metrics: success rate, lap time, peak roll/tilt, outcome breakdown (collision/miss/oob/timeout),
command saturation, pass offsets.

Two eval axes (cfg-independent of how the checkpoint was trained):
  --course vq1     the HELD-OUT VQ1 course (the acceptance gate: success >= S1.3's 1.00)
  --course random  freshly sampled procedural courses (the generalization number)
  --plant map      measured super-rate plant (DEFAULT -- the S14 integration; ALL S1.4+ evals)
  --plant flat     legacy flat-2.5 plant (ONLY for flat-trained checkpoints: inc-1 / S1.3)

Runs the EXACT training substrate (DiffAero env + our torch plant) with ``dynamics.dr=False``
(nominal plant -- we check the policy, not its DR robustness); ``--standing-frac 1.0`` (default)
spawns every reset at the standing start. Mirrors diffaero's TestRunner inner loop.

Usage (on Adroit, diffaero env active, PYTHONPATH = repo/rl:repo/src:diffaero):
  python peregrine_eval.py --ckpt <run>/checkpoints --course vq1
  python peregrine_eval.py --ckpt <run>/checkpoints --course random --n-envs 512
``--ckpt`` is the directory containing actor.pth; the training cfg is read from its run's
.hydra/config.yaml (walked up from --ckpt), exactly as diffaero's script/test.py does.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

# register OUR injections (zero clone edits), identical to peregrine_train_racing.py
import diffaero.dynamics as _dyn
import diffaero.env as _env
from diffaero_dynamics import PeregrinePlantDynamics
from peregrine_racing import PeregrineRacing

from racer.rl_plant import (ALPHA_MAX_RPS2_MEASURED, PlantParams,  # noqa: E402
                            SUPER_RATE_S_MEASURED, QUAD_DRAG_C2_MEASURED,
                            COLL_MAP_THR_MEASURED, COLL_MAP_ACCEL_MEASURED,
                            MIXER_IDLE_MEASURED, MIXER_KAPPA_ERR_MEASURED,
                            MIXER_KAPPA_HOLD_MEASURED, MIXER_ZETA_YAW_MEASURED)

_env.ENV_ALIAS["peregrine_racing"] = PeregrineRacing

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
    ap.add_argument("--course", default="vq1", choices=["vq1", "random"],
                    help="vq1 = the held-out acceptance course; random = generalization")
    ap.add_argument("--plant", default="map", choices=["map", "flat", "aero", "mixer"],
                    help="map = measured super-rate plant (S1.4+ default); flat = legacy, "
                         "ONLY for flat-trained checkpoints; aero = map + measured aero "
                         "(quad body drag + convex collective, linear_drag=0; twin-falsify "
                         "2026-06-11) -- S1.5 (inc5) evals; mixer = aero + the measured "
                         "motor-mixer coupling (live-deploy diag 2026-06-11) -- ALL S17+ "
                         "(inc6, mixer-trained) evals")
    ap.add_argument("--standing-frac", type=float, default=1.0)
    ap.add_argument("--n-envs", type=int, default=256)
    ap.add_argument("--max-time", type=float, default=40.0)
    ap.add_argument("--horizons", type=float, default=2.5,
                    help="run this many max_time horizons to gather completed episodes")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    _aero = dict(super_rate_s=SUPER_RATE_S_MEASURED,
                 alpha_max_rps2=ALPHA_MAX_RPS2_MEASURED.copy(),
                 linear_drag=0.0,
                 quad_drag_c2=QUAD_DRAG_C2_MEASURED.copy(),
                 coll_map_thr=COLL_MAP_THR_MEASURED.copy(),
                 coll_map_accel=COLL_MAP_ACCEL_MEASURED.copy())
    if args.plant == "mixer":    # the fully measured plant as of S17 (aero + motor mixer)
        params = PlantParams(**_aero,
                             mixer_idle=MIXER_IDLE_MEASURED,
                             mixer_kappa_err=MIXER_KAPPA_ERR_MEASURED,
                             mixer_kappa_hold=MIXER_KAPPA_HOLD_MEASURED,
                             mixer_zeta_yaw=MIXER_ZETA_YAW_MEASURED)
    elif args.plant == "aero":   # the S16 fixed-params form (pre-mixer)
        params = PlantParams(**_aero)
    elif args.plant == "map":
        params = PlantParams(super_rate_s=SUPER_RATE_S_MEASURED,
                             alpha_max_rps2=ALPHA_MAX_RPS2_MEASURED)
    else:
        params = None
    _dyn.DYNAMICS_ALIAS["peregrine_plant"] = lambda cfg, device: PeregrinePlantDynamics(
        cfg, device, backend="torch", params=params)

    from diffaero.env import build_env          # noqa: E402  (after alias registration)
    from diffaero.algo import build_agent       # noqa: E402

    ckpt = Path(args.ckpt).resolve()
    run_root = find_run_root(ckpt)
    cfg = OmegaConf.load(run_root / ".hydra" / "config.yaml")
    print(f"[eval] run_root={run_root}  ckpt={ckpt}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # clean twin: chosen course, forced standing start, DR OFF (nominal plant), sized horizon
    OmegaConf.update(cfg, "n_envs", args.n_envs, force_add=True)
    OmegaConf.update(cfg, "env.course_mode", args.course, force_add=True)
    OmegaConf.update(cfg, "env.standing_start_frac", float(args.standing_frac), force_add=True)
    OmegaConf.update(cfg, "dynamics.dr", False, force_add=True)
    OmegaConf.update(cfg, "env.max_time", float(args.max_time), force_add=True)

    env = build_env(cfg.env, device=device)
    agent = build_agent(cfg.algo, env, device)
    agent.load(ckpt)
    dt = float(cfg.env.dt)
    print(f"[eval] n_envs={args.n_envs} dt={dt} max_time={args.max_time} course={args.course} "
          f"plant={args.plant} dr=False standing_frac={args.standing_frac}")

    obs = env.reset()
    ep_success, ep_roll, ep_tilt = [], [], []
    counts = {"collision": 0.0, "miss": 0.0, "oob": 0.0, "timeout": 0.0, "finish": 0.0}
    finish_times, pass_offsets, mean_speeds = [], [], []
    sat_steps = torch.zeros(4, device=device)
    n_steps_total = 0
    # S17 ACTION-RATE stats (the inc6 style gate): per-tick action delta in SPAN-NORMALISED
    # units (|a_t - a_{t-1}| / 2 of the [-1,1] pre-rescale action = fraction of full span per
    # tick; 1.0 = a full-rail flip, the inc5 bang-bang/dither signature). Episode boundaries
    # (reset ticks) are masked out. Yaw flip = consecutive |a_yaw| > 0.5 with opposite signs.
    dact_hist = [[] for _ in range(4)]
    yaw_flips = 0
    yaw_pairs = 0
    prev_action = None

    n_steps = int(args.horizons * args.max_time / dt)
    with torch.no_grad():
        for _ in range(n_steps):
            action, _ = agent.act(obs, test=True)
            sat_steps += (action.abs() > 0.95).float().sum(dim=0)
            n_steps_total += action.shape[0]
            if prev_action is not None:
                da = (action - prev_action).abs() * 0.5            # span-normalised per tick
                da = da[valid_prev]                                # mask envs reset last tick
                for ax in range(4):
                    dact_hist[ax].append(da[:, ax].cpu())
                yy, py = action[valid_prev, 3], prev_action[valid_prev, 3]
                flips = ((yy.abs() > 0.5) & (py.abs() > 0.5) & (yy * py < 0))
                yaw_flips += int(flips.sum())
                yaw_pairs += int(flips.numel())
            prev_action = action.clone()
            action = env.rescale_action(action)
            obs, _loss, _term, info = env.step(action)
            valid_prev = ~info["reset"]                            # next delta invalid for resets

            # Peak roll/tilt come from the ENV's per-episode trackers (stats_raw) -- recorded
            # PRE-reset inside step(), so the terminal/crash pose is included and the next
            # episode's spawn pose cannot contaminate the ended episode (review finding F12;
            # an eval-side tracker reads env.dynamics._q AFTER the internal auto-reset).
            sr = info["stats_raw"]
            counts["collision"] += float(sr["collision_rate"].sum())
            counts["miss"] += float(sr["miss_rate"].sum())
            counts["oob"] += float(sr["oob_rate"].sum())
            counts["timeout"] += float(sr["survive_rate"].sum())
            counts["finish"] += float(sr["success_rate"].sum())
            finish_times += sr["finish_time_s"].tolist()
            pass_offsets += sr["pass_offset_m"].tolist()
            mean_speeds += sr["mean_speed"].tolist()

            reset = info["reset"]
            if reset.any():
                ridx = reset.nonzero().flatten()
                succ = info["success"]
                for k, i in enumerate(ridx.tolist()):
                    ep_success.append(bool(succ[i]))
                    ep_roll.append(float(sr["peak_roll_deg"][k]) * np.pi / 180.0)
                    ep_tilt.append(float(sr["peak_tilt_deg"][k]) * np.pi / 180.0)

    succ = np.array(ep_success)
    roll = np.degrees(np.array(ep_roll))
    tilt = np.degrees(np.array(ep_tilt))
    n_ep = len(succ)
    sr_rate = float(succ.mean()) if n_ep else 0.0
    print(f"\n[RESULT] course={args.course} plant={args.plant} episodes={n_ep}  "
          f"SUCCESS_RATE_6of6={sr_rate:.3f}")
    tot = max(sum(counts.values()), 1.0)
    print("[RESULT] outcomes: " + "  ".join(f"{k}={v:.0f} ({v / tot:.1%})"
                                            for k, v in counts.items()))
    if finish_times:
        ft = np.array(finish_times)
        print(f"[RESULT] FINISH_TIME_S med {np.median(ft):6.2f}  p10 {np.percentile(ft,10):6.2f} "
              f" p90 {np.percentile(ft,90):6.2f}  min {ft.min():6.2f}")
    if pass_offsets:
        po = np.array(pass_offsets)
        print(f"[RESULT] PASS_OFFSET_M med {np.median(po):.3f}  p90 {np.percentile(po,90):.3f} "
              f" max {po.max():.3f}  (frame at 0.75)")
    if mean_speeds:
        ms = np.array(mean_speeds)
        print(f"[RESULT] MEAN_SPEED med {np.median(ms):5.2f} m/s  p90 {np.percentile(ms,90):5.2f}")
    sat = (sat_steps / max(n_steps_total, 1)).cpu().numpy()
    print(f"[RESULT] CMD_SATURATION |a|>0.95: thrust {sat[0]:.1%}  "
          f"roll {sat[1]:.1%}  pitch {sat[2]:.1%}  yaw {sat[3]:.1%}")
    # S17 action-rate gate: inc5 datum ~1.0 span/tick rails on thrust+yaw; the documented
    # inc6 acceptance bound is p95 <= 0.5 span/tick on thrust AND yaw, yaw flip rate <= 5%.
    if dact_hist[0]:
        names = ("thrust", "roll", "pitch", "yaw")
        parts = []
        for ax in range(4):
            v = torch.cat(dact_hist[ax]).numpy()
            parts.append(f"{names[ax]} p95 {np.percentile(v, 95):.3f} max {v.max():.3f}")
        flip_rate = yaw_flips / max(yaw_pairs, 1)
        print("[RESULT] ACTION_RATE span/tick: " + "  ".join(parts))
        print(f"[RESULT] YAW_FLIP_RATE (|a|>0.5 sign flips): {flip_rate:.1%}")
        v_thr = torch.cat(dact_hist[0]).numpy(); v_yaw = torch.cat(dact_hist[3]).numpy()
        print(f"ACTRATE_SUMMARY thr_p95={np.percentile(v_thr, 95):.3f} "
              f"yaw_p95={np.percentile(v_yaw, 95):.3f} yaw_flip={flip_rate:.3f}")
    if n_ep:
        print(f"[RESULT] PEAK_ROLL_DEG all:     max {roll.max():6.1f}  p90 {np.percentile(roll,90):6.1f}  median {np.median(roll):6.1f}")
        print(f"[RESULT] PEAK_TILT_DEG all:     max {tilt.max():6.1f}  p90 {np.percentile(tilt,90):6.1f}  median {np.median(tilt):6.1f}")
    if succ.any():
        rs, ts = roll[succ], tilt[succ]
        print(f"[RESULT] PEAK_ROLL_DEG success: max {rs.max():6.1f}  p90 {np.percentile(rs,90):6.1f}  median {np.median(rs):6.1f}")
        print(f"[RESULT] PEAK_TILT_DEG success: max {ts.max():6.1f}  p90 {np.percentile(ts,90):6.1f}  median {np.median(ts):6.1f}")
    # machine-readable summary line
    print(f"EVAL_SUMMARY course={args.course} plant={args.plant} sr={sr_rate:.3f} n_ep={n_ep} "
          f"t_med={np.median(finish_times) if finish_times else -1:.2f} "
          f"tilt_succ_max={(tilt[succ].max() if succ.any() else -1):.1f} "
          f"roll_succ_max={(roll[succ].max() if succ.any() else -1):.1f} "
          f"coll={counts['collision']:.0f} miss={counts['miss']:.0f} oob={counts['oob']:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
