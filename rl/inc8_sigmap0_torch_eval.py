"""rl/inc8_sigmap0_torch_eval.py -- inc8 GO-gate instrument, measured IN THE TORCH TRAINING ENV.

WHY THIS EXISTS (and why it is NOT rl/inc8_sigmap0_eval.py).
The inc8 GO gate is the PHYSICAL terminal-centering spread at the binding gate-4 plane:
``sigma_p0_lat <= ~0.08 m`` (miss p99 ~= 3*sigma_p0). The companion numpy tool
``rl/inc8_sigmap0_eval.py`` flies the policy on the NUMPY plant (contact_true_eval.run_episode) +
the numpy estimator emulator. The first FLYING inc8 policy ("rc1", the recenter re-train job
3276449) flies the full 2-axis camera-pointing primitive IN TRAINING (torch), but DIES 0/200 on
the numpy tool's emulated obs (the train/eval obs-fidelity gap #37, amplified by the aggressive
g_pitch=3 look-at). So the faithful place to measure the 2-axis sigma_p0 is the environment where
rc1 PROVABLY flies: the TORCH training env (PeregrineRacingInc8). This tool rolls a trained actor in
that env under DETERMINISTIC inference (the policy MEAN, no PPO noise) with the in-loop estimator
emulation (the sigma_p0 spread source) and the look-at primitive ON, and records the GROUND-TRUTH
gate-4 plane crossing (y, z) -- never the KF estimate -- to compute the GT-anchored sigma_p0.

WHAT sigma_p0 IS (GT-anchored, identical definition to the numpy tool).
The TRUE drone position relative to the gate-4 centre at the gate-4 plane crossing, in the gate frame:
  - lateral  y = rel[1]  (horizontal cross-track)  -> sigma_p0_LAT  (the binding GO axis)
  - vertical z = rel[2]  (altitude cross-track)     -> sigma_p0_vert (the dominated axis, reported)
The crossing is the SAME interpolation the env's own crossing_events uses: world_to_gateframe of the
TRUTH position (env._p, Z-up) about the gate-4 frame, interpolated to the gate plane x=0. sigma_p0 is
the spread of that crossing across an ENSEMBLE whose ONLY per-episode variation is the estimator DR
(inc8_estimator_emul.EmulConfig: per-fix sigma_lat ~U[0.05,0.15], one-signed PnP bias ~U[0,0.19],
IMU/accept noise), drawn fresh on every reset_idx. estim_err (|KF - truth|, floored ~0.115 m
regardless of policy) is NOT sigma_p0 and is never substituted.

APPLES-TO-APPLES WITH THE NUMPY 0.1768 CROSS-CHECK (the design that makes the spread the RIGHT spread).
The numpy tool's yaw-only baseline (rc1 seed0, start=trainreset = a single FIXED gate-0+1m pose,
identity attitude, v=w=0, NO jitter) = sigma_p0_lat 0.1768. To isolate the SAME spread here -- the
estimator DR, not the spawn -- this tool:
  * defaults --standing-frac 1.0 so EVERY reset targets gate-0 (a FULL gate-0->gate-4 course, not a
    "1 m up-course of a random gate" partial spawn which would contaminate the ensemble with cheap
    near-gate-4 crossings -- peregrine_racing.reset_idx:776-788);
  * FILTERS the ensemble to gate-0-spawned crossings only (the spawn-gate provenance guard), so even
    at a lower standing-frac the number stays a full-course measurement (n_rejected_nonfull reports
    the discards);
  * runs a FIXED horizon budget and accepts EVERY crossing (it does NOT early-exit on a crossing
    count, which would truncate the slow-to-reach tail that drives lat_p99 -- the numpy tool runs a
    fixed n_episodes, every episode to completion).
RESIDUAL CAVEAT: the env applies an UNCONDITIONAL +-0.25 m xy / +-0.15 rad spawn jitter
(peregrine_racing.reset_idx:792-795) the numpy trainreset does not. Over the ~135 m gate-0->gate-4
course that gate-0 spawn jitter largely washes out (the policy returns to the racing line), so its
residual at the gate-4 crossing is small (sub-threshold, within the 0.04 cross-check band). If the yaw
cross-check DISAGREES with 0.1768 by more than the band, the spawn jitter (and #37 obs fidelity) are
the first suspects -- report it, do not silently trust the 2-axis number (the prompt's escape hatch).

GO RULE.  sigma_p0_lat <= 0.08 m AND |lateral miss| p99 (~3*sigma_p0) <= ~0.24 m, on a sufficient
ensemble of gate-4 crossings (a low reach-rate / tiny ensemble is itself a NO-GO/NO-DATA signal).

WHERE THIS RUNS.  The torch env subclasses diffaero's env and uses our torch plant; it needs the
diffaero clone + a CUDA-class torch. It CANNOT run on the dev laptop (no diffaero -- confirmed
2026-06-18: ModuleNotFoundError, torch CPU-only). Run it on Adroit (conda activate diffaero, GPU
node) -- see rl/inc8_sigmap0_torch.sbatch. The build + static validation are laptop work; the NUMBER
is one Adroit run away.

FOOTGUN GUARDED HERE.  The look-at gain has a PPO-update warmup ramp
(inc8_reward.lookat_warmup_factor): in inference the env's ``_ppo_update`` stays 0, so a trained
config with ``lookat_warmup_updates>0`` (rc1 trained with 200) would scale the look-at gain to ZERO
-> no pointing -> an UNFAITHFUL rollout. We FORCE ``lookat_warmup_updates=0`` (full gain from step 0,
== the deployed policy) and re-assert it on the built env.

PROVISIONAL.  The obs the policy flies on here comes from the estimator EMUL (fix_surrogate ->
LinearKF), NOT a real detector->PnP->KF chain (#37). Any GO/NO-GO is PROVISIONAL until the
real-detector spike confirms emul fidelity.

Usage (on Adroit, diffaero env active, PYTHONPATH = repo/rl:repo/src:diffaero):
  # 2-axis (the deliverable): full look-at, the trained gains
  python inc8_sigmap0_torch_eval.py --ckpt outputs/train/inc8_recenter_seed0_rc1/best --lookat auto
  # yaw-only (instrument cross-check vs the numpy tool's ~0.177)
  python inc8_sigmap0_torch_eval.py --ckpt outputs/train/inc8_recenter_seed0_rc1/best --lookat yaw
``--ckpt`` is the FULL agent-save dir (actor.pth + critic.pth + actor.json); the training cfg is read
from its run's .hydra/config.yaml (walked up from --ckpt), exactly as peregrine_eval / diffaero
script/test do.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

# Register OUR injections (zero diffaero-clone edits), identical to peregrine_eval.py /
# peregrine_train_inc8.py. The env alias points at the RECORDING subclass below so build_env
# constructs it; the transition is byte-identical to PeregrineRacingInc8 (the subclass only records
# gate-4 crossings AROUND an unmodified super().step(), and tags each reset's spawn gate).
import diffaero.dynamics as _dyn          # noqa: E402
import diffaero.env as _env               # noqa: E402
from diffaero_dynamics import PeregrinePlantDynamics  # noqa: E402
from peregrine_racing import PeregrineRacing, world_to_gateframe  # noqa: E402
from peregrine_racing_inc8 import PeregrineRacingInc8  # noqa: E402

# Nominal-plant param construction (reuse peregrine_eval's measured-plant assembly so the torch
# plant here is the SAME nominal plant the numpy sigma_p0 tool uses with --plant mixer; the sigma_p0
# spread then comes PURELY from the estimator DR, apples-to-apples with the numpy cross-check).
from racer.rl_plant import (ALPHA_MAX_RPS2_MEASURED, PlantParams,  # noqa: E402
                            SUPER_RATE_S_MEASURED, QUAD_DRAG_C2_MEASURED,
                            COLL_MAP_THR_MEASURED, COLL_MAP_ACCEL_MEASURED,
                            LAPSE_SPEED_MEASURED, LAPSE_FACTOR_MEASURED,
                            MIXER_IDLE_MEASURED, MIXER_KAPPA_ERR_MEASURED,
                            MIXER_KAPPA_HOLD_MEASURED, MIXER_ZETA_YAW_MEASURED)

GATE4_IDX = 4
SPAWN_GATE_FULLCOURSE = 0    # only gate-0-spawned episodes are full gate-0->gate-4 courses
MIN_ENSEMBLE_DEFAULT = 30    # below this, a GO/NO-GO verdict is not statistically meaningful

# Trained look-at config for the rc1 recenter seeds (peregrine_inc8_recenter.sbatch COMMON +
# RC_OVERRIDES; mirrored in contact_true_eval.LOOKAT_*_DEFAULT). SIGN FOOTGUN: g_yaw and g_pitch have
# OPPOSITE signs and BOTH are empirical (the analytical sign was wrong both times) -- do not "fix".
LOOKAT_G_YAW = -3.0
LOOKAT_G_PITCH = 3.0
LOOKAT_R_LO = 8.0
LOOKAT_R_HI = 30.0


class _Sigmap0Env(PeregrineRacingInc8):
    """PeregrineRacingInc8 + a read-only GT gate-4-crossing recorder. The transition is UNCHANGED
    (super().step() / super().reset_idx() are called verbatim; we only record around them), so this is
    byte-identical to the training env for flight/reward/obs. It exposes the ground-truth (y, z) at
    every FULL-COURSE (gate-0-spawned) gate-4 plane crossing for the sigma_p0 ensemble."""

    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self._g4_lat: list[float] = []      # lateral y at the gate-4 crossing (gate frame, TRUTH)
        self._g4_vert: list[float] = []     # vertical z at the gate-4 crossing (gate frame, TRUTH)
        self._spawn_tg = torch.zeros_like(self.target_gates)  # per-env spawn gate of the CURRENT episode
        self._n_episodes = 0                # episodes ENDED (reset events) over the rollout
        self._n_finished = 0                # episodes that FINISHED the course (success)
        self._n_rejected_nonfull = 0        # gate-4 crossings discarded because spawn gate != 0 (partial)
        # running sums of the per-step batch-mean pointing diagnostics (loss_components)
        self._diag_sum = {k: 0.0 for k in (
            "inc8_fix_rate", "inc8_pointing_rate", "inc8_terminal_pointing",
            "inc8_lockband_pointing", "inc8_band_az_abs_deg", "inc8_band_el_abs_deg",
            "inc8_estim_err_inplane_m")}
        self._diag_steps = 0

    def reset_idx(self, env_idx):
        super().reset_idx(env_idx)          # the unmodified inc7/inc8 spawn + emulator re-init
        # tag the spawn gate of the new episode (target_gates is the post-spawn target). Guarded so a
        # reset_idx call during super().__init__ (before _spawn_tg exists) is a no-op.
        if hasattr(self, "_spawn_tg"):
            self._spawn_tg[env_idx] = self.target_gates[env_idx]

    def step(self, action, *args, **kwargs):
        if not getattr(self, "_inc8_on", False):
            return super().step(action, *args, **kwargs)
        prev_p = self._p.clone()                       # Z-up TRUTH position BEFORE the dynamics step
        prev_tg = self.target_gates.clone()            # target gate index BEFORE the step
        spawn_tg = self._spawn_tg.clone()              # the CURRENT episode's spawn gate (pre-reset)
        out = super().step(action, *args, **kwargs)    # the UNMODIFIED inc8 transition (incl. reset_idx)
        info = out[3]
        reset_mask = info["reset"]

        # An env passed gate-4 THIS step iff its target advanced 4 -> 5 (gates advance one at a time)
        # and it was NOT reset this step. For a non-last clean pass the env is NOT terminated, so
        # self._p is still the post-step TRUTH (the env's curr_pos). Restrict to FULL-COURSE episodes
        # (spawn gate 0) so the ensemble is one clean gate-0->gate-4 approach per episode -- the same
        # population the numpy tool's fixed trainreset start measures.
        passed_g4 = (prev_tg == GATE4_IDX) & (self.target_gates == GATE4_IDX + 1) & (~reset_mask)
        full = passed_g4 & (spawn_tg == SPAWN_GATE_FULLCOURSE)
        self._n_rejected_nonfull += int((passed_g4 & ~full).sum())
        if bool(full.any()):
            idx = full.nonzero().flatten()
            g4_pos = self.gate_pos[idx, GATE4_IDX]     # (m,3) Z-up, fixed course geometry
            g4_yaw = self.gate_yaw[idx, GATE4_IDX]     # (m,)
            # SAME gate frame + SAME plane-crossing interpolation as crossing_events (gate plane x=0).
            rel_prev = world_to_gateframe(prev_p[idx] - g4_pos, g4_yaw)
            rel_curr = world_to_gateframe(self._p[idx] - g4_pos, g4_yaw)
            denom = rel_curr[..., 0] - rel_prev[..., 0]
            denom = torch.where(denom.abs() < 1e-9, torch.full_like(denom, 1e-9), denom)
            f = -rel_prev[..., 0] / denom
            y = rel_prev[..., 1] + f * (rel_curr[..., 1] - rel_prev[..., 1])
            z = rel_prev[..., 2] + f * (rel_curr[..., 2] - rel_prev[..., 2])
            self._g4_lat.extend(y.detach().cpu().tolist())
            self._g4_vert.extend(z.detach().cpu().tolist())

        # episode accounting (mirrors peregrine_eval): success_rate in stats_raw is success[reset].
        sr = info.get("stats_raw", {})
        self._n_episodes += int(reset_mask.sum())
        if "success_rate" in sr:
            self._n_finished += int(sr["success_rate"].sum().item())

        # batch-mean pointing diagnostics (logging-only scalars the env already computes)
        lc = info.get("loss_components", {})
        for k in self._diag_sum:
            if k in lc:
                self._diag_sum[k] += float(lc[k])
        self._diag_steps += 1
        return out

    def sigma_p0_report(self) -> dict:
        lat = np.asarray(self._g4_lat, dtype=np.float64)
        vert = np.asarray(self._g4_vert, dtype=np.float64)
        inplane = np.hypot(lat, vert) if lat.size else np.array([])

        def pct(a, q):
            return float(np.percentile(a, q)) if a.size else float("nan")

        diag = {k: (v / self._diag_steps if self._diag_steps else float("nan"))
                for k, v in self._diag_sum.items()}
        return {
            "n_reached_g4": int(lat.size),
            "n_episodes": int(self._n_episodes),
            "n_finished": int(self._n_finished),
            "n_rejected_nonfull": int(self._n_rejected_nonfull),
            "reach_rate": (lat.size / self._n_episodes) if self._n_episodes else float("nan"),
            "success_rate": (self._n_finished / self._n_episodes) if self._n_episodes else float("nan"),
            "sigma_p0_lat": float(np.std(lat)) if lat.size else float("nan"),
            "bias_lat": float(np.mean(lat)) if lat.size else float("nan"),
            "lat_abs_p90": pct(np.abs(lat), 90),
            "lat_abs_p99": pct(np.abs(lat), 99),
            "sigma_p0_vert": float(np.std(vert)) if vert.size else float("nan"),
            "bias_vert": float(np.mean(vert)) if vert.size else float("nan"),
            "vert_abs_p90": pct(np.abs(vert), 90),
            "vert_abs_p99": pct(np.abs(vert), 99),
            "inplane_p90": pct(inplane, 90),
            "inplane_p99": pct(inplane, 99),
            "diag": diag,
        }


def _go_verdict(m: dict, sigma_target: float = 0.08, min_n: int = MIN_ENSEMBLE_DEFAULT) -> tuple[str, str]:
    """The numpy tool's GO rule (sigma_p0_lat <= target AND lat_p99 <= 3*target), with a NO-DATA guard
    for an empty / too-small ensemble. Returns (token, human_string)."""
    n, s, p99 = m["n_reached_g4"], m["sigma_p0_lat"], m["lat_abs_p99"]
    if n == 0 or np.isnan(s):
        return "NO-DATA", "NO-DATA (no gate-4 crossings reached)"
    if n < min_n:
        return "NO-DATA", f"NO-DATA (ensemble n={n} < {min_n}; raise --horizons)"
    ok_sigma = s <= sigma_target
    ok_p99 = (not np.isnan(p99)) and p99 <= 3.0 * sigma_target
    token = "GO" if (ok_sigma and ok_p99) else "NO-GO"
    return token, (f"{token} (sigma_p0_lat={s:.3f}{'<=' if ok_sigma else '>'}{sigma_target} ; "
                   f"lat_p99={p99:.3f}{'<=' if ok_p99 else '>'}{3 * sigma_target:.2f})")


def _build_plant_params(plant: str):
    """Nominal measured plant params (no per-episode plant DR), matching peregrine_eval.py."""
    _aero = dict(super_rate_s=SUPER_RATE_S_MEASURED, alpha_max_rps2=ALPHA_MAX_RPS2_MEASURED.copy(),
                 linear_drag=0.0, quad_drag_c2=QUAD_DRAG_C2_MEASURED.copy(),
                 coll_map_thr=COLL_MAP_THR_MEASURED.copy(), coll_map_accel=COLL_MAP_ACCEL_MEASURED.copy())
    if plant in ("mixer", "lapse"):
        _lapse = (dict(lapse_speed=LAPSE_SPEED_MEASURED.copy(), lapse_factor=LAPSE_FACTOR_MEASURED.copy())
                  if plant == "lapse" else {})
        return PlantParams(**_aero, mixer_idle=MIXER_IDLE_MEASURED, mixer_kappa_err=MIXER_KAPPA_ERR_MEASURED,
                           mixer_kappa_hold=MIXER_KAPPA_HOLD_MEASURED, mixer_zeta_yaw=MIXER_ZETA_YAW_MEASURED,
                           **_lapse)
    if plant == "aero":
        return PlantParams(**_aero)
    if plant == "map":
        return PlantParams(super_rate_s=SUPER_RATE_S_MEASURED, alpha_max_rps2=ALPHA_MAX_RPS2_MEASURED)
    return None


def find_run_root(ckpt: Path) -> Path:
    p = ckpt
    while p != p.parent:
        if (p / ".hydra" / "config.yaml").exists():
            return p
        p = p.parent
    raise SystemExit(f"no .hydra/config.yaml found walking up from {ckpt}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="FULL agent-save dir (actor.pth + critic.pth + actor.json)")
    ap.add_argument("--lookat", default="auto", choices=["auto", "yaw", "off"],
                    help="auto = full 2-axis (g_yaw=-3,g_pitch=3, the deliverable); yaw = yaw-only "
                         "(g_pitch=0, the numpy-tool cross-check); off = no look-at (racing-line baseline)")
    ap.add_argument("--g-yaw", type=float, default=None, help="override look-at yaw gain (default -3.0)")
    ap.add_argument("--g-pitch", type=float, default=None, help="override look-at pitch gain (default +3.0)")
    ap.add_argument("--n-envs", type=int, default=256)
    ap.add_argument("--horizons", type=float, default=2.5,
                    help="run horizons*max_time/dt control steps, accepting EVERY gate-4 crossing (no "
                         "early-exit-on-count, which would bias against the slow-to-reach tail). Raise if "
                         "the gathered ensemble is small.")
    ap.add_argument("--max-time", type=float, default=25.0,
                    help="per-episode truncation horizon (s); must comfortably exceed a full gate-0->gate-4 "
                         "flight (~8 s lap) so slow-but-valid approaches are not truncated short")
    ap.add_argument("--plant", default="mixer", choices=["map", "flat", "aero", "mixer", "lapse"],
                    help="nominal plant (default mixer == numpy tool's default; sigma spread = estimator DR)")
    ap.add_argument("--standing-frac", type=float, default=1.0,
                    help="standing_start_frac; default 1.0 = every reset targets gate-0 (full course == the "
                         "numpy trainreset regime). A lower value still measures full-course only (the "
                         "spawn-gate filter discards partial spawns) but gathers crossings slower.")
    ap.add_argument("--dynamics-dr", action="store_true",
                    help="enable per-episode PLANT DR too (default OFF: isolate the estimator-DR spread, "
                         "matching the numpy tool's nominal-plant cross-check)")
    ap.add_argument("--seed", type=int, default=0, help="torch seed (spawn + estimator DR reproducibility)")
    ap.add_argument("--sigma-target", type=float, default=0.08)
    ap.add_argument("--min-ensemble", type=int, default=MIN_ENSEMBLE_DEFAULT,
                    help="minimum gate-4 crossings before a GO/NO-GO verdict is issued (else NO-DATA)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    # resolve look-at gains for the chosen mode (explicit -> no reliance on the recorded cfg value)
    g_yaw = LOOKAT_G_YAW if args.g_yaw is None else args.g_yaw
    g_pitch = LOOKAT_G_PITCH if args.g_pitch is None else args.g_pitch
    if args.lookat == "yaw":
        g_pitch = 0.0
    elif args.lookat == "off":
        g_yaw, g_pitch = 0.0, 0.0

    params = _build_plant_params(args.plant)
    _dyn.DYNAMICS_ALIAS["peregrine_plant"] = lambda cfg, device: PeregrinePlantDynamics(
        cfg, device, backend="torch", params=params)
    _env.ENV_ALIAS["peregrine_racing"] = PeregrineRacing
    _env.ENV_ALIAS["peregrine_racing_inc8"] = _Sigmap0Env   # the recording subclass

    from diffaero.env import build_env       # noqa: E402  (after alias registration)
    from diffaero.algo import build_agent    # noqa: E402

    ckpt = Path(args.ckpt).resolve()
    if not (ckpt / "actor.pth").is_file():
        raise SystemExit(f"--ckpt {ckpt} has no actor.pth (point it at a FULL agent-save dir, e.g. "
                         f"<run>/best or <run>/periodic; this tool loads actor+critic via PPO.load).")
    if not (ckpt / "critic.pth").is_file():
        print(f"[sigmap0-torch] WARNING: {ckpt} has no critic.pth -- diffaero PPO.load loads actor+critic; "
              f"a deploy-only actor dir will fail. (Deterministic inference needs only the actor, but the "
              f"loader requires both.)")
    run_root = find_run_root(ckpt)
    cfg = OmegaConf.load(run_root / ".hydra" / "config.yaml")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[sigmap0-torch] run_root={run_root}  ckpt={ckpt}  device={device}")

    # ---- cfg overrides: small batch, nominal plant, the chosen look-at, warmup OFF (the footgun) ----
    OmegaConf.update(cfg, "n_envs", args.n_envs, force_add=True)
    OmegaConf.update(cfg, "env.course_mode", "vq1", force_add=True)
    OmegaConf.update(cfg, "env.inc8", True, force_add=True)
    OmegaConf.update(cfg, "env.max_time", float(args.max_time), force_add=True)
    OmegaConf.update(cfg, "env.standing_start_frac", float(args.standing_frac), force_add=True)
    OmegaConf.update(cfg, "dynamics.dr", bool(args.dynamics_dr), force_add=True)
    OmegaConf.update(cfg, "env.lookat_g_yaw", float(g_yaw), force_add=True)
    OmegaConf.update(cfg, "env.lookat_g_pitch", float(g_pitch), force_add=True)
    OmegaConf.update(cfg, "env.lookat_r_lo", float(LOOKAT_R_LO), force_add=True)
    OmegaConf.update(cfg, "env.lookat_r_hi", float(LOOKAT_R_HI), force_add=True)
    # FOOTGUN: with _ppo_update==0 in inference, a positive warmup scales the look-at gain to 0.
    OmegaConf.update(cfg, "env.lookat_warmup_updates", 0, force_add=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    env = build_env(cfg.env, device=device)
    # belt-and-braces: hold the look-at gain at full strength regardless of any warmup the cfg carried.
    if hasattr(env, "_lookat_warmup_updates"):
        env._lookat_warmup_updates = 0
    if hasattr(env, "_ppo_update"):
        env._ppo_update = 10 ** 9
    agent = build_agent(cfg.algo, env, device)
    agent.load(ckpt)

    dt = float(OmegaConf.select(cfg, "env.dt") or 0.0333)
    n_steps = int(args.horizons * args.max_time / dt)
    lookat_on = (g_yaw != 0.0) or (g_pitch != 0.0)
    print(f"[sigmap0-torch] lookat={args.lookat} (g_yaw={g_yaw} g_pitch={g_pitch} on={lookat_on}) "
          f"plant={args.plant} dynamics_dr={args.dynamics_dr} n_envs={args.n_envs} "
          f"standing_frac={args.standing_frac} dt={dt} n_steps={n_steps} seed={args.seed}")
    assert bool(OmegaConf.select(cfg, "env.inc8")), "inc8 must be ON (estimator emul = the sigma_p0 spread)"

    obs = env.reset()
    with torch.no_grad():
        for _ in range(n_steps):                          # FIXED budget; accept EVERY crossing (no bias)
            action, _ = agent.act(obs, test=True)         # DETERMINISTIC policy MEAN (no PPO noise)
            action = env.rescale_action(action)            # [-1,1] -> physical [act_lo, act_hi]
            obs, _loss, _term, _info = env.step(action)    # look-at + estimator-emul happen inside

    m = env.sigma_p0_report()
    d = m["diag"]
    token, verdict = _go_verdict(m, args.sigma_target, args.min_ensemble)
    print(f"\n{'=' * 80}")
    print(f"inc8 TORCH-ENV GT-ANCHORED sigma_p0  ckpt={ckpt.parent.name}/{ckpt.name}  "
          f"lookat={args.lookat}  steps={n_steps}")
    print(f"{'=' * 80}")
    print(f"  episodes_ended={m['n_episodes']}  reached_g4(full-course)={m['n_reached_g4']} "
          f"(reach_rate={m['reach_rate']:.3f})  finished={m['n_finished']} "
          f"(success_rate={m['success_rate']:.3f})  partial_g4_discarded={m['n_rejected_nonfull']}")
    print(f"  (reach_rate/success_rate denominators are COMPLETED episodes; a small in-flight tail at the "
          f"horizon end is excluded.)")
    print(f"  -- GT-anchored TERMINAL CENTERING at the gate-4 plane (TRUE trajectory) --")
    print(f"  LATERAL  sigma_p0_lat = {m['sigma_p0_lat']:.4f} m   bias = {m['bias_lat']:+.4f} m   "
          f"|miss| p90 = {m['lat_abs_p90']:.4f}  p99 = {m['lat_abs_p99']:.4f}")
    print(f"  VERTICAL sigma_p0_vert= {m['sigma_p0_vert']:.4f} m   bias = {m['bias_vert']:+.4f} m   "
          f"|miss| p90 = {m['vert_abs_p90']:.4f}  p99 = {m['vert_abs_p99']:.4f}")
    print(f"  in-plane (combined) |miss| p90 = {m['inplane_p90']:.4f}  p99 = {m['inplane_p99']:.4f}")
    print(f"  -- pointing diagnostics (batch-mean over steps; course-wide, not gate-4-specific) --")
    print(f"  fix_rate={d['inc8_fix_rate']:.3f} pointing={d['inc8_pointing_rate']:.3f} "
          f"terminal_pointing={d['inc8_terminal_pointing']:.3f} lockband_pointing={d['inc8_lockband_pointing']:.3f}")
    print(f"  band_az_abs={d['inc8_band_az_abs_deg']:.2f}deg band_el_abs={d['inc8_band_el_abs_deg']:.2f}deg "
          f"estim_err_inplane={d['inc8_estim_err_inplane_m']:.3f}m")
    print(f"  >> GO RULE (sigma_p0_lat<={args.sigma_target}): {verdict}")
    print(f"  >> NOTE: estimator-DR-only spread (standing_frac={args.standing_frac}, plant DR "
          f"{'ON' if args.dynamics_dr else 'OFF'}); residual gate-0 spawn jitter washes out over the course "
          f"but is NOT zero (vs the numpy zero-jitter trainreset).")
    print(f"  >> PROVISIONAL: emul obs, not a real detector->PnP->KF (#37). Crown only after the spike.\n")

    # machine-readable summary (one grep-able line; the sbatch tee's these). Verdict via _go_verdict.
    print(f"SIGMAP0_TORCH_SUMMARY ckpt={ckpt.parent.parent.name} lookat={args.lookat} "
          f"reach={m['n_reached_g4']}/{m['n_episodes']} reach_rate={m['reach_rate']:.3f} "
          f"success_rate={m['success_rate']:.3f} "
          f"sigma_p0_lat={m['sigma_p0_lat']:.4f} bias_lat={m['bias_lat']:+.4f} "
          f"lat_p90={m['lat_abs_p90']:.4f} lat_p99={m['lat_abs_p99']:.4f} "
          f"sigma_p0_vert={m['sigma_p0_vert']:.4f} bias_vert={m['bias_vert']:+.4f} "
          f"fix_rate={d['inc8_fix_rate']:.3f} term_point={d['inc8_terminal_pointing']:.3f} "
          f"verdict={token}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
