"""Lightweight precheck (login CPU): register + build PeregrinePlantDynamics through diffaero's
factory using the quad cfg with name overridden, then take a few hover steps + a reset_idx. Catches
wiring / cfg / interface errors before spending a GPU SLURM job. Does NOT import the heavy env."""
import torch
from omegaconf import OmegaConf
import diffaero.dynamics as D
from diffaero_dynamics import PeregrinePlantDynamics

D.DYNAMICS_ALIAS["peregrine_plant"] = PeregrinePlantDynamics

# quad.yaml has ${...} interpolations for n_envs/n_agents/dt -> set concrete values (no parent cfg here)
cfg = OmegaConf.load("/scratch/network/fl3689/diffaero_repo/cfg/dynamics/quad.yaml")
cfg.name = "peregrine_plant"
cfg.n_envs = 8
cfg.n_agents = 1
cfg.dt = 0.0333
cfg.alpha = 1.0
cfg.g = 9.80665

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dyn = D.build_dynamics(cfg, dev)
print("BUILD_OK class=%s type=%s plant=%s state_dim=%d action_dim=%d"
      % (type(dyn).__name__, dyn.type, getattr(dyn, "plant_name", "?"), dyn.state_dim, dyn.action_dim))
print("min_action", [round(x, 3) for x in dyn.min_action.tolist()],
      "max_action", [round(x, 3) for x in dyn.max_action.tolist()])

# mimic the reset state racing.py writes (pos + identity xyzw quat at index 6), then hover a few steps
state = torch.zeros(8, 13, device=dev)
state[:, 6] = 1.0
dyn._state = state
u_hover = torch.zeros(8, 4, device=dev)
u_hover[:, 0] = 1.0                        # normed thrust == hover -> a_up == g
for _ in range(10):
    dyn.step(u_hover)
print("HOVER10 pos0=%s vel0=%s (expect ~0: hover holds)"
      % ([round(x, 4) for x in dyn.p[0].tolist()], [round(x, 4) for x in dyn.v[0].tolist()]))

# a sink step (normed thrust 0) should accelerate DOWN in DiffAero Z-up (-Z)
dyn._state = state.clone()
u_fall = torch.zeros(8, 4, device=dev)
dyn.step(u_fall)
print("FALL1 vel0_z=%.4f (expect < 0: falls in Z-up)" % dyn.v[0, 2].item())

dyn.reset_idx(torch.tensor([0, 1], device=dev))
print("RESET_IDX_OK")
print("PRECHECK_DONE")
