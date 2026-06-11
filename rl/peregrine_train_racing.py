"""Stage-1 launcher: train a state->action CTBR policy on OUR courses + OUR plant (given pose, no vision).

Registers our injections into diffaero's factories at import time (zero edits to the pristine clone),
then hands off to diffaero's own hydra train entrypoint:
  * dynamics: DYNAMICS_ALIAS["peregrine_plant"] -> PeregrinePlantDynamics  (system-ID'd plant + DR)
  * env:      ENV_ALIAS["peregrine_racing"]     -> PeregrineRacing          (S1.4 coherent env)
  * algo:     AGENT_ALIAS["ppo"]                -> GuardedPPO               (NaN-guarded PPO, S1.4)
plus two launcher-level lifelines monkeypatched onto diffaero's TrainRunner (S1.4): unconditional
periodic checkpoints and an emergency save on any training exception. Rationale: the S1.3
5000-update run NaN'd at update ~2243 (PPO ``Normal(loc)`` validation) and wrote NO artifact --
the stock runner only saves in ``close()`` (skipped on exceptions) and on a success-rate
high-water mark gated to ``i %% save_freq == 0``.

Typical invocation (see peregrine_racing_s14.sbatch):

  python peregrine_train_racing.py \
      env=racing env.name=peregrine_racing +env.course_mode=random +env.standing_start_frac=0.3 \
      dynamics=quad dynamics.name=peregrine_plant dynamics.g=9.80665 +dynamics.dr=true \
      dynamics.controller.max_normed_thrust=3.765 \
      algo=ppo n_envs=2048 n_updates=6000 headless=True device=0
"""
import json
import os
import traceback

import torch

import diffaero.algo as _algo
import diffaero.dynamics as _dyn
import diffaero.env as _env
from diffaero.algo.PPO import PPO
from diffaero.utils.runner import TrainRunner
from diffaero_dynamics import PeregrinePlantDynamics
from peregrine_racing import PeregrineRacing

# Register with backend="torch": build_dynamics(cfg, device) passes no backend kwarg, so the class
# would otherwise default to the slow, NON-differentiable, DR-IGNORING numpy path (_step_numpy reads
# only the scalar self.params -- the per-env DR tensors + latency apply ONLY in _step_torch/step()).
# The DR (super-rate map s/rate_tau/alpha_max bands + control-latency) therefore REQUIRES the torch
# backend. The torch mirror is parity-proven to the numpy plant (check_against_rl_plant; V100 gate
# job 3265870 worst 1.8e-15), so this changes throughput + DR-awareness, NOT the physics.
_dyn.DYNAMICS_ALIAS["peregrine_plant"] = lambda cfg, device: PeregrinePlantDynamics(
    cfg, device, backend="torch")
_env.ENV_ALIAS["peregrine_racing"] = PeregrineRacing


class GuardedPPO(PPO):
    """PPO + a non-finite-gradient guard. After each minibatch backward, if ANY parameter gradient
    is non-finite, zero all grads and SKIP that optimizer step (counted + reported). One poisoned
    minibatch then costs one skipped step instead of NaN'ing the weights and killing the run."""

    NAN_ABORT_AFTER = 200    # a persistently-poisoned run burns no more than ~6 updates of steps

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nan_skipped = 0
        orig_step = self.optim.step

        def guarded_step(*a, **k):
            for group in self.optim.param_groups:
                for p in group["params"]:
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        self.nan_skipped += 1
                        self.optim.zero_grad()
                        print(f"[nan-guard] non-finite gradient -- optimizer step SKIPPED "
                              f"(total skipped: {self.nan_skipped})")
                        if self.nan_skipped >= self.NAN_ABORT_AFTER:
                            raise RuntimeError(
                                f"[nan-guard] {self.nan_skipped} skipped steps -- the run is "
                                f"persistently poisoned; aborting (emergency save follows)")
                        return None
            return orig_step(*a, **k)

        self.optim.step = guarded_step

    @staticmethod
    def build(cfg, env, device):
        return GuardedPPO(cfg=cfg, obs_dim=env.obs_dim, action_dim=env.action_dim,
                          n_envs=env.n_envs, l_rollout=cfg.l_rollout, device=device)


_algo.AGENT_ALIAS["ppo"] = GuardedPPO


def _weights_finite(agent) -> bool:
    return all(torch.isfinite(p).all() for p in agent.agent.parameters())


_orig_run = TrainRunner.run


def _run_with_lifelines(self):
    agent, logger, cfg = self.agent, self.logger, self.cfg

    # SIDECAR AT SAVE TIME (review finding F1, critical): every actor.pth gets a .json sidecar
    # carrying the action bounds it was TRAINED with -- fly_rl.load_actor reads it so the deploy
    # rescale follows the checkpoint instead of a hand-synced constant. Wrapping agent.save
    # covers ALL save paths in one place: the runner's high-water "best", our periodic and
    # emergency saves, and the final close() save.
    orig_save = agent.save
    cc = cfg.dynamics.controller

    def save_with_sidecar(path):
        orig_save(path)
        sidecar = {"act_max_thrust": float(cc.max_normed_thrust),
                   "act_max_rate": float(cc.max_roll_rate)}
        with open(os.path.join(path, "actor.json"), "w") as f:
            json.dump(sidecar, f)

    agent.save = save_with_sidecar

    orig_step = agent.step
    counter = {"i": 0}

    def step_with_periodic_save(*a, **k):
        out = orig_step(*a, **k)
        counter["i"] += 1
        if counter["i"] % max(int(cfg.save_freq), 1) == 0:
            if _weights_finite(agent):
                # rotate so a kill mid-write never destroys the only copy
                pdir = os.path.join(logger.logdir, "periodic")
                prev = os.path.join(logger.logdir, "periodic_prev")
                if os.path.isdir(pdir):
                    import shutil
                    shutil.rmtree(prev, ignore_errors=True)
                    os.replace(pdir, prev)
                agent.save(pdir)
            else:
                print("[lifeline] weights non-finite at update %d -- periodic save SKIPPED"
                      % counter["i"])
        return out

    agent.step = step_with_periodic_save
    try:
        _orig_run(self)
        if getattr(agent, "nan_skipped", 0):
            print(f"[nan-guard] run completed with {agent.nan_skipped} skipped optimizer steps")
    except Exception:
        traceback.print_exc()
        try:
            if _weights_finite(agent):
                agent.save(os.path.join(logger.logdir, "emergency"))
                print("[lifeline] EMERGENCY checkpoint saved to %s"
                      % os.path.join(logger.logdir, "emergency"))
            else:
                print("[lifeline] weights non-finite -- emergency save skipped "
                      "(use the last periodic/ checkpoint)")
        except Exception:
            traceback.print_exc()
        raise


TrainRunner.run = _run_with_lifelines

from diffaero.script.train import main  # noqa: E402  (must follow the registrations above)

if __name__ == "__main__":
    main()
