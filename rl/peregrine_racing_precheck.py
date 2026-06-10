"""Login-node (CPU) precheck for the Stage-1 racing run.

Composes the EXACT training cfg via hydra (same overrides as peregrine_racing.sbatch), registers our
dynamics + env, builds PeregrineRacing, resets, and steps -- so a PASS here means the real run's cfg,
wiring, obs width, injected course, and per-env DR are all valid before spending a GPU SLURM job.
Heavy imports (pytorch3d/open3d) load on the login node. Ends with PRECHECK_DONE.
"""
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import diffaero.dynamics as D
import diffaero.env as E
from diffaero_dynamics import PeregrinePlantDynamics
from peregrine_racing import PeregrineRacing

D.DYNAMICS_ALIAS["peregrine_plant"] = PeregrinePlantDynamics
E.ENV_ALIAS["peregrine_racing"] = PeregrineRacing

OVERRIDES = [
    "env=racing", "env.name=peregrine_racing",
    "dynamics=quad", "dynamics.name=peregrine_plant", "dynamics.g=9.80665", "+dynamics.dr=true",
    "algo=ppo", "n_envs=64", "n_updates=2", "headless=True", "device=-1",
]

with initialize_config_dir(version_base="1.3", config_dir="/scratch/network/fl3689/diffaero_repo/cfg"):
    cfg = compose(config_name="config_train", overrides=OVERRIDES)

OmegaConf.set_struct(cfg, False)
cfg.env.render.headless = True
dev = torch.device("cpu")

env = E.build_env(cfg.env, dev)
print("BUILD_OK class=%s dyn=%s obs_dim=%d state_dim=%d action_dim=%d n_gates=%d max_steps=%d"
      % (type(env).__name__, type(env.dynamics).__name__, env.obs_dim, env.state_dim,
         env.action_dim, env.n_gates, env.max_steps))
print("gate_pos(zup)=\n%s" % env.gate_pos.cpu().numpy().round(2))
print("gate_yaw(deg)=%s  half_opening=%.2f m"
      % (torch.rad2deg(env.gate_yaw).cpu().numpy().round(1), env.gate_half_opening_m))
print("oob_box min=%s max=%s  passage_bonus=%.1f finish_bonus=%.1f"
      % (env.box_min.cpu().numpy().round(1), env.box_max.cpu().numpy().round(1),
         env.passage_bonus, env.finish_bonus))
print("min_action=%s max_action=%s"
      % (env.dynamics.min_action.tolist(), env.dynamics.max_action.tolist()))

obs = env.reset()
print("RESET obs.shape=%s match_obs_dim=%s" % (tuple(obs.shape), obs.shape[-1] == env.obs_dim))

dyn = env.dynamics
if getattr(dyn, "_dr_enabled", False):
    rg = dyn._dr_rate_gain
    print("DR ON  rate_gain per-env min=%s max=%s (nominal [2.50,2.50,2.23])"
          % (rg.amin(0).cpu().numpy().round(3), rg.amax(0).cpu().numpy().round(3)))
    print("DR hover [%.3f,%.3f] drag [%.3f,%.3f] tau [%.4f,%.4f]  (distinct=%s)"
          % (dyn._dr_hover.min(), dyn._dr_hover.max(), dyn._dr_drag.min(), dyn._dr_drag.max(),
             dyn._dr_rate_tau.min(), dyn._dr_rate_tau.max(),
             bool((dyn._dr_hover.std() > 0).item())))
else:
    print("DR OFF (unexpected)")

# Step with a mid-range action (~hover-ish after rescale) and confirm the course/reward machinery runs.
act = (env.dynamics.min_action + (env.dynamics.max_action - env.dynamics.min_action) * 0.5).unsqueeze(0).expand(64, 4).clone()
z0 = env.p[:, 2].mean().item()
for _ in range(10):
    obs, (loss, reward), term, extra = env.step(act)
print("STEP10 obs.shape=%s reward.mean=%.3f z0=%.2f z10=%.2f n_passed_mean=%.2f stats=%s"
      % (tuple(obs.shape), reward.mean().item(), z0, env.p[:, 2].mean().item(),
         env.n_passed_gates.float().mean().item(), list(extra["stats_raw"].keys())))
print("PRECHECK_DONE")
