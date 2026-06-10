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

_dyn.DYNAMICS_ALIAS["peregrine_plant"] = PeregrinePlantDynamics
_env.ENV_ALIAS["peregrine_racing"] = PeregrineRacing

from diffaero.script.train import main  # noqa: E402  (must follow the registrations above)

if __name__ == "__main__":
    main()
