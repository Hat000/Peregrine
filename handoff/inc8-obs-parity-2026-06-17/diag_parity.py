"""inc8 OBS-PARITY DIAGNOSIS -- train/eval divergence localizer.

Hypothesis: the eval (contact_true_eval.run_episode) flies the policy WITHOUT the
look-at primitive that the TRAINING env (peregrine_racing_inc8.step) composes onto the
action before the dynamics. The look-at primitive is the ENGINEERED pointing (architecture
pivot) -- the policy never LEARNED to point; the primitive does it. Remove it in eval and
the camera never points -> no fixes -> KF coasts -> reach=0.

This script drives the SAME eval rollout TWO ways from the SAME states/seeds:
  (A) NO look-at  -- exactly what inc8_sigmap0_eval.py does today  -> expect reach=0, no pointing
  (B) WITH look-at -- faithfully replicate the training control law  -> expect pointing+fixes+reach

If (B) recovers pointing/fixes/reach, the divergence is purely the MISSING PRIMITIVE in the
eval harness == INSTRUMENT-GAP. If (B) still fails, the trained pointing was hollow == POLICY-GAP.

NOT a fix -- a diagnostic. Writes nothing to the eval/train code paths.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_RL = Path(__file__).resolve().parents[2] / "rl"
_SRC = Path(__file__).resolve().parents[2] / "src"
for _p in (str(_SRC), str(_RL)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fly_rl import _FLIP, _GATE_POS_ZUP, _TRAIN_DT, _HOVER_THRUST, N_GATES, load_actor, policy_step
from contact_true_eval import (_build_plant_params, _build_start, _score_gate,
                               BODY_RADIUS_NOM, FRAME_DEPTH_NOM)
from estimator_emul import EmulConfig, EstimatorEmulator, actor_obs_dim
from offline_rollout import _R_from_quat, obs_from_truth
from racer.rl_plant import step as plant_step
import fix_surrogate as FS

# ---- look-at primitive, numpy reimpl of rl/inc8_reward.lookat_correction (physical FRD output) ----
# r_bc = R_body_from_camera == frames.R_camera_from_body().T for the pure 20deg mount (inc8_reward.py:171)
_SIN20, _COS20 = 0.34202014332566871, 0.93969262078590843
_R_BC = np.array([[0.0, _SIN20, _COS20],
                  [1.0, 0.0, 0.0],
                  [0.0, _COS20, -_SIN20]], dtype=np.float64)


def lookat_frd(t_cam: np.ndarray, g_yaw: float, g_pitch: float) -> np.ndarray:
    """PHYSICAL FRD body-rate correction (rad/s) that rotates the camera optical axis toward the gate.
    Mirrors inc8_reward.lookat_correction's w_frd = w_cam @ r_bc.T (BEFORE the final FRD->FLU action
    flip): in training diffaero re-flips action[1:4] FLU->FRD, so the PHYSICAL FRD rate added is exactly
    this w_frd. In this eval the plant consumes rate_frd directly, so we add w_frd to rate_frd."""
    n = float(np.linalg.norm(t_cam))
    if n < 1e-6:
        return np.zeros(3)
    u = t_cam / n
    w_cam = np.array([-g_pitch * u[1], g_yaw * u[0], 0.0], dtype=np.float64)
    return w_cam @ _R_BC.T


def run_one(actor, obs_dim, emul_seed, lookat, g_yaw=-3.0, g_pitch=3.0,
            r_lo=8.0, r_hi=30.0, rate_lim=3.14159, max_time=40.0, plant="mixer",
            obs_mode="emul", dump=False):
    params = _build_plant_params(plant)
    st, gate, vflip = _build_start("simstart", 0)
    dt = _TRAIN_DT
    n_steps = int(round(max_time / dt))
    emul_rng = np.random.default_rng(emul_seed)
    emu = EstimatorEmulator(EmulConfig(), gate_pos_zup=_GATE_POS_ZUP, gate_yaw=None)
    emu.reset(st, gate, emul_rng)

    gp = _GATE_POS_ZUP
    pts = np.vstack([gp, [0.0, 0.0, -0.02]])
    oob_lo = pts.min(0) - np.array([15, 15, 12])
    oob_hi = pts.max(0) + np.array([15, 15, 12])

    st_prev_obs = st
    last_normed = 0.0
    outcome = "TIMEOUT"
    max_gate_reached = -1          # highest gate index PASSED
    n_applied = 0                  # steps the look-at band fired
    for k in range(n_steps):
        if k > 0:
            emu.step(st_prev_obs, st, gate, dt, emul_rng)
        emul_obs = emu.obs(st, gate, last_normed, vflip, None, obs_dim)
        if obs_mode == "truth":
            obs = obs_from_truth(st, gate, last_normed, vflip, gate_map=None)
            if obs_dim > 17 and obs.shape[0] == 17:
                # append the SAME confidence triple the emul would (keep obs width = policy width)
                obs = np.concatenate([obs, emul_obs[17:20]]).astype(np.float32)
        else:
            obs = emul_obs
        rate_frd, _coll, last_normed = policy_step(actor, obs, 0.0, vflip)
        if dump and gate == 0 and (k % 10 == 0 or k < 5):
            kf_e = float(np.linalg.norm(emu.kf.position - np.asarray(st.pos)))
            print(f"      k={k:3d} g{gate} obs[0:3]={np.round(obs[0:3],3).tolist()} "
                  f"truthpos={np.round(st.pos,2).tolist()} kf_err={kf_e:.3f}")

        # gate-in-camera geom at the CURRENT (pre-step) truth state -- the look-at trigger
        R_cur_ned = _R_from_quat(np.asarray(st.quat, dtype=np.float64))
        geom = FS.geometry(np.asarray(st.pos, dtype=np.float64), R_cur_ned, emu.gates[gate])
        if lookat and (r_lo <= geom.range_m <= r_hi) and (geom.t_cam[2] > 0):
            rate_frd = np.clip(rate_frd + lookat_frd(geom.t_cam, g_yaw, g_pitch),
                               -rate_lim, rate_lim)
            n_applied += 1

        collective = last_normed * _HOVER_THRUST
        action = np.concatenate([rate_frd, [collective]])

        st_prev_obs = st
        prev_pos = st.pos.copy()
        st = plant_step(st, action, dt, params)

        # ---- scoring (mirror run_episode) ----
        coll_other = False
        for g in range(N_GATES):
            if g == gate:
                continue
            v, _ = _score_gate(prev_pos, st.pos, g, BODY_RADIUS_NOM, FRAME_DEPTH_NOM, gp, None)
            if v == "collision":
                coll_other = True
                break
        if coll_other:
            outcome = "COLLISION_OTHER"
            break
        v, _ = _score_gate(prev_pos, st.pos, gate, BODY_RADIUS_NOM, FRAME_DEPTH_NOM, gp, None)
        if v == "pass":
            max_gate_reached = gate
            if gate == N_GATES - 1:
                outcome = "FINISHED"
                break
            gate += 1
            continue
        if v == "collision":
            outcome = f"COLLISION_g{gate}"
            break
        if v == "miss":
            outcome = f"MISS_g{gate}"
            break
        if np.any(st.pos * _FLIP < oob_lo) or np.any(st.pos * _FLIP > oob_hi):
            outcome = f"OOB_g{gate}"
            break

    tr = emu.trace
    tg = np.asarray(tr.target_gate)
    fx = np.asarray(tr.fix_accepted, dtype=bool)
    im = np.asarray(tr.in_image_target, dtype=bool)
    rg4 = np.asarray(tr.range_gate4)
    # pointing/fix in the surrogate accept band [12,28] m to the TARGET gate
    rt = np.asarray(tr.range_target)
    band = (rt >= 12.0) & (rt <= 28.0)
    g4band = (tg == 4) & (rg4 <= 28.0)
    reached_g4 = bool((tg == 4).any())
    return {
        "lookat": lookat, "seed": emul_seed, "outcome": outcome, "obs_mode": obs_mode,
        "max_gate": max_gate_reached, "reached_g4_target": reached_g4,
        "n_steps": len(tg), "n_applied": n_applied,
        "pointing_rate_band": float(im[band].mean()) if band.any() else float("nan"),
        "fix_rate_band": float(fx[band].mean()) if band.any() else float("nan"),
        "fix_rate_all": float(fx.mean()) if fx.size else float("nan"),
        "n_fix": int(fx.sum()),
        "g4_fix_rate": float(fx[g4band].mean()) if g4band.any() else float("nan"),
        "g4_lock": float(im[(tg == 4) & (rg4 <= 5.0)].mean()) if ((tg == 4) & (rg4 <= 5.0)).any()
        else float("nan"),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="rl/checkpoints/inc8_ws1_seed2_actor.pth")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--g-yaw", type=float, default=-3.0)
    ap.add_argument("--g-pitch", type=float, default=3.0)
    args = ap.parse_args()

    actor = load_actor(args.ckpt)
    obs_dim = actor_obs_dim(actor)
    print(f"ckpt={Path(args.ckpt).name}  obs_dim={obs_dim}  n_episodes={args.n}  "
          f"lookat g_yaw={args.g_yaw} g_pitch={args.g_pitch}")

    # ---- BISECTION 1: TRUTH obs (no KF, no look-at) -- isolates policy/harness/plant from the emul ----
    print(f"\n########## BISECTION: TRUTH obs (perfect pose, no look-at) ##########")
    print("  (if this also misses gate-0, the eval policy can't fly the course at all -> not a KF/emul issue)")
    rows = [run_one(actor, obs_dim, s, False, obs_mode="truth", dump=(s == 0)) for s in range(args.n)]
    reach = sum(r["reached_g4_target"] for r in rows)
    fin = sum(r["outcome"] == "FINISHED" for r in rows)
    maxg = np.array([r["max_gate"] for r in rows])
    print(f"  TRUTH-OBS: finished={fin}/{args.n}  reached_g4={reach}/{args.n}  "
          f"max_gate: min={maxg.min()} med={int(np.median(maxg))} max={maxg.max()}")
    for r in rows:
        print(f"    seed{r['seed']:2d}  {r['outcome']:16s}  max_gate={r['max_gate']:2d}")

    for lookat in (False, True):
        rows = [run_one(actor, obs_dim, s, lookat, g_yaw=args.g_yaw, g_pitch=args.g_pitch,
                        obs_mode="emul")
                for s in range(args.n)]
        tag = "WITH look-at" if lookat else "NO look-at (== current eval)"
        reach = sum(r["reached_g4_target"] for r in rows)
        maxg = np.array([r["max_gate"] for r in rows])
        pr = np.array([r["pointing_rate_band"] for r in rows])
        frb = np.array([r["fix_rate_band"] for r in rows])
        nfix = np.array([r["n_fix"] for r in rows])
        print(f"\n=== {tag} ===")
        print(f"  reached_g4(target): {reach}/{args.n}   "
              f"max_gate_passed: min={maxg.min()} med={int(np.median(maxg))} max={maxg.max()}")
        print(f"  pointing_rate (target in [12,28] m): mean={np.nanmean(pr):.3f}")
        print(f"  fix_rate (target in [12,28] m):      mean={np.nanmean(frb):.3f}")
        print(f"  n_fix per episode: mean={nfix.mean():.1f}  total={int(nfix.sum())}")
        for r in rows:
            print(f"    seed{r['seed']:2d}  {r['outcome']:16s}  max_gate={r['max_gate']:2d}  "
                  f"point_band={r['pointing_rate_band']:.2f}  fix_band={r['fix_rate_band']:.2f}  "
                  f"n_fix={r['n_fix']:3d}  applied={r['n_applied']:3d}")


if __name__ == "__main__":
    main()
