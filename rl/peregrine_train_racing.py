"""Stage-1 launcher: train a state->action CTBR policy on OUR VQ1 course + OUR plant (given pose, no vision).

Registers BOTH our injections into diffaero's factories at import time (zero edits to the pristine clone),
then hands off to diffaero's own hydra train entrypoint:
  * dynamics: DYNAMICS_ALIAS["peregrine_plant"] -> PeregrinePlantDynamics  (our system-ID'd plant, + DR)
  * env:      ENV_ALIAS["peregrine_racing"]     -> PeregrineRacing          (our 6-gate descending course)

Both are resolved at call time inside build_dynamics/build_env, so registering before importing `main` is
sufficient. Typical invocation (see peregrine_racing.sbatch):

  python peregrine_train_racing.py \
      env=racing env.name=peregrine_racing \
      dynamics=quad dynamics.name=peregrine_plant dynamics.g=9.80665 +dynamics.dr=true \
      algo=ppo n_envs=2048 n_updates=4000 headless=True device=0
"""
import diffaero.dynamics as _dyn
import diffaero.env as _env
from diffaero_dynamics import PeregrinePlantDynamics
from peregrine_racing import PeregrineRacing

# Register with backend="torch": build_dynamics(cfg, device) passes no backend kwarg, so the class
# would otherwise default to the slow, NON-differentiable, DR-IGNORING numpy path (_step_numpy reads
# only the scalar self.params -- the per-env DR tensors + latency apply ONLY in _step_torch/step()).
# The DR (super-rate map s/rate_tau/alpha_max bands + control-latency; the disproven asymmetric
# rate-gain band was REPLACED by the static map, characterize-sweep 2026-06-10) therefore REQUIRES
# the torch backend to take effect. The torch mirror is parity-proven to the numpy plant
# (check_against_rl_plant), so this changes throughput + DR-awareness, NOT the physics.
_dyn.DYNAMICS_ALIAS["peregrine_plant"] = lambda cfg, device: PeregrinePlantDynamics(
    cfg, device, backend="torch")
_env.ENV_ALIAS["peregrine_racing"] = PeregrineRacing

from diffaero.script.train import main  # noqa: E402  (must follow the registrations above)

if __name__ == "__main__":
    main()
