"""Peregrine smoke-train launcher.

Registers our system-ID'd plant in diffaero's dynamics factory, then hands off to diffaero's OWN
hydra train entrypoint unchanged. The diffaero clone stays pristine (no repo edits): the monkeypatch
runs at import time, and ``build_dynamics`` looks up ``DYNAMICS_ALIAS`` at call time (inside build_env),
so the registration is visible by then.

Select our plant by reusing the existing quad config with a name override (no new cfg file needed):

  python peregrine_train.py env=racing dynamics=quad dynamics.name=peregrine_plant \
         dynamics.g=9.80665 algo=ppo n_envs=2048 headless=True device=0

build_dynamics(cfg.dynamics) then resolves DYNAMICS_ALIAS["peregrine_plant"] -> PeregrinePlantDynamics,
which reads n_envs/n_agents/dt/alpha/g/n_substeps + the controller bounds from the quad cfg (it ignores
quad's m/J/D/arm_l/c_tau -- our plant abstracts the inner loop as a measured rate lag + thrust map).
"""
import diffaero.dynamics as _dyn
from diffaero_dynamics import PeregrinePlantDynamics

_dyn.DYNAMICS_ALIAS["peregrine_plant"] = PeregrinePlantDynamics

from diffaero.script.train import main  # noqa: E402  (must follow the registration above)

if __name__ == "__main__":
    main()
