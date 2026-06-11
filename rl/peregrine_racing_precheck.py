"""Login-node (CPU) precheck for the Stage-1.4 racing run.

Composes the EXACT training cfg via hydra (same overrides as peregrine_racing_s14.sbatch),
registers our dynamics (TORCH backend, as in training) + env + GuardedPPO, builds PeregrineRacing
in BOTH course modes, resets, and steps -- so a PASS here means the real run's cfg, wiring, obs
width, course sampling, spawn poses, per-env DR, and the reward/stat machinery are all valid
before spending a GPU SLURM job. Ends with PRECHECK_DONE.
"""
import math

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import diffaero.dynamics as D
import diffaero.env as E
from diffaero_dynamics import PeregrinePlantDynamics
from peregrine_racing import PeregrineRacing

D.DYNAMICS_ALIAS["peregrine_plant"] = lambda cfg, device: PeregrinePlantDynamics(
    cfg, device, backend="torch")          # torch = the training backend (DR lives there)
E.ENV_ALIAS["peregrine_racing"] = PeregrineRacing

OVERRIDES = [
    "env=racing", "env.name=peregrine_racing",
    "+env.course_mode=random", "+env.standing_start_frac=0.3",
    "dynamics=quad", "dynamics.name=peregrine_plant", "dynamics.g=9.80665", "+dynamics.dr=true",
    "dynamics.controller.max_normed_thrust=3.765",
    "algo=ppo", "n_envs=64", "n_updates=2", "headless=True", "device=-1",
]

with initialize_config_dir(version_base="1.3", config_dir="/scratch/network/fl3689/diffaero_repo/cfg"):
    cfg = compose(config_name="config_train", overrides=OVERRIDES)

OmegaConf.set_struct(cfg, False)
cfg.env.render.headless = True
dev = torch.device("cpu")

env = E.build_env(cfg.env, dev)
print("BUILD_OK class=%s dyn=%s backend=%s obs_dim=%d state_dim=%d action_dim=%d n_gates=%d max_steps=%d"
      % (type(env).__name__, type(env.dynamics).__name__, env.dynamics.backend, env.obs_dim,
         env.state_dim, env.action_dim, env.n_gates, env.max_steps))
print("course_mode=%s standing_start_frac=%.2f half_inner=%.2f half_outer=%.2f"
      % (env.course_mode, env.standing_start_frac, env.gate_half_opening_m, env.gate_half_outer_m))
print("reward weights: %s" % env.rw)
print("env0 gate_pos(zup)=\n%s" % env.gate_pos[0].cpu().numpy().round(2))
print("env0 gate_yaw(deg)=%s" % torch.rad2deg(env.gate_yaw[0]).cpu().numpy().round(1))
print("env0 oob_box min=%s max=%s" % (env.box_min[0].cpu().numpy().round(1),
                                      env.box_max[0].cpu().numpy().round(1)))
print("min_action=%s max_action=%s"
      % (env.dynamics.min_action.tolist(), env.dynamics.max_action.tolist()))
assert abs(env.dynamics.max_action[0].item() - 3.765) < 1e-6, "thrust ceiling not applied"

# course randomization is real: envs see distinct courses
spread = (env.gate_pos.std(dim=0).mean()).item()
print("course spread across envs (std of gate_pos): %.2f m  distinct=%s" % (spread, spread > 1.0))
assert spread > 1.0, "course_mode=random but all envs share one course"

obs = env.reset()
print("RESET obs.shape=%s match_obs_dim=%s" % (tuple(obs.shape), obs.shape[-1] == env.obs_dim))
assert obs.shape[-1] == env.obs_dim
state = env.get_state()
print("STATE state.shape=%s match_state_dim=%s" % (tuple(state.shape), state.shape[-1] == env.state_dim))
assert state.shape[-1] == env.state_dim

# standing starts: ~30% of envs at their pad (at rest, tail-first, tilted)
at_pad = (torch.linalg.norm(env.p[:, :2] - env.spawn_pos[:, :2], dim=-1) < 1.0).float().mean()
print("standing-start fraction at reset: %.2f (expect ~0.30)" % at_pad.item())

dyn = env.dynamics
if getattr(dyn, "_dr_enabled", False):
    s = dyn._dr_s
    am = dyn._dr_alpha_max
    print("DR ON  super_rate_s per-env min=%s max=%s (band [0.25,0.35]; G0 FIXED at nominal)"
          % (s.amin(0).cpu().numpy().round(3), s.amax(0).cpu().numpy().round(3)))
    print("DR alpha_max per-env min=%s max=%s (r/p band [200,320]; yaw scaled 80/260)"
          % (am.amin(0).cpu().numpy().round(1), am.amax(0).cpu().numpy().round(1)))
    print("DR hover [%.3f,%.3f] drag [%.3f,%.3f] tau [%.4f,%.4f] (band [0.015,0.030]; distinct=%s)"
          % (dyn._dr_hover.min(), dyn._dr_hover.max(), dyn._dr_drag.min(), dyn._dr_drag.max(),
             dyn._dr_rate_tau.min(), dyn._dr_rate_tau.max(),
             bool((dyn._dr_hover.std() > 0).item())))
    print("DR latency enabled=%s max_steps=%d" % (dyn._latency_enabled, dyn._latency_max))
else:
    print("DR OFF (unexpected)")
    raise SystemExit(1)

# Step with a mid-range action and confirm the course/reward/stat machinery runs.
act = (env.dynamics.min_action + (env.dynamics.max_action - env.dynamics.min_action) * 0.5
       ).unsqueeze(0).expand(64, 4).clone()
z0 = env.p[:, 2].mean().item()
for _ in range(10):
    obs, (loss, reward), term, extra = env.step(act)
print("STEP10 obs.shape=%s reward.mean=%.3f z0=%.2f z10=%.2f n_passed_mean=%.2f"
      % (tuple(obs.shape), reward.mean().item(), z0, env.p[:, 2].mean().item(),
         env.n_passed_gates.float().mean().item()))
expect_stats = {"success_rate", "survive_rate", "l_episode", "n_passed_gates", "collision_rate",
                "miss_rate", "oob_rate", "peak_tilt_deg", "peak_roll_deg", "mean_speed",
                "finish_time_s", "pass_offset_m"}
got_stats = set(extra["stats_raw"].keys())
print("stats_raw keys: %s  missing=%s" % (sorted(got_stats), sorted(expect_stats - got_stats)))
assert expect_stats <= got_stats

# VQ1 hold-out mode builds too (the eval path)
cfg2 = OmegaConf.merge(cfg, {})
cfg2.env.course_mode = "vq1"
cfg2.env.standing_start_frac = 1.0
env2 = E.build_env(cfg2.env, dev)
import numpy as np
gp0 = env2.gate_pos[0].cpu().numpy()
print("VQ1 mode: gate0=%s gate5=%s (expect [-23.3, 0.4, 1.39] / [-159.2, 4.4, -24.61])"
      % (gp0[0].round(2), gp0[5].round(2)))
assert abs(gp0[0][0] + 23.298) < 0.01 and abs(gp0[5][2] + 24.608) < 0.01
env2.reset()
q0 = env2.dynamics._q[0].cpu().numpy()
print("VQ1 standing spawn quat (xyzw, jittered) = %s (nominal ~ [0, -0.155, 0, 0.988])"
      % q0.round(3))

# GuardedPPO registration smoke (the launcher's algo injection)
from peregrine_train_racing import GuardedPPO     # noqa: E402  (registers aliases again, harmless)
import diffaero.algo as A
agent = A.build_agent(cfg.algo, env, dev)
print("AGENT %s nan_guard=%s" % (type(agent).__name__, hasattr(agent, "nan_skipped")))
assert type(agent).__name__ == "GuardedPPO"

print("PRECHECK_DONE")
