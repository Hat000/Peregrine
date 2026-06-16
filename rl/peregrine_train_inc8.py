"""inc8 launcher: train the case-C camera-pointing CTBR policy (in-loop estimator emulation + the
2-axis terminal-lock perception reward) on OUR plant + the VQ1 course.

Mirrors rl/peregrine_train_racing.py (zero edits to the pristine diffaero clone) but registers the
inc8 env and writes the checkpoint's obs-dim + r5 arm into the sidecar so deploy/load gates on it
(S7: inc7 17-dim checkpoints still load; inc8 ships 20-dim).

  * dynamics: DYNAMICS_ALIAS["peregrine_plant"] -> PeregrinePlantDynamics  (system-ID'd plant + DR)
  * env:      ENV_ALIAS["peregrine_racing_inc8"] -> PeregrineRacingInc8     (inc8 case-C env)
  * algo:     AGENT_ALIAS["ppo"]                -> GuardedPPO               (NaN-guarded PPO)

Typical inc8 L0 invocation (BSR3 widened per d5 4.1; map-ON measured plant; arm A default):

  python peregrine_train_inc8.py \
      env=racing env.name=peregrine_racing_inc8 +env.inc8=true +env.r5_arm=A \
      +env.standing_start_frac=0.3 +env.spin_rate_abort=10.0 +env.spin_time_abort=3.0 \
      dynamics=quad dynamics.name=peregrine_plant dynamics.g=9.80665 +dynamics.dr=true \
      +dynamics.dr_aero=true +dynamics.dr_mixer=true \
      dynamics.controller.max_normed_thrust=3.765 \
      algo=ppo n_envs=2048 n_updates=6000 headless=True device=0

This worker BUILDS + smokes the env (1 seed, arm A, short budget). The {L0,L1,L2}x{A>=5,B3,C1}
ladder is the NEXT worker -- do NOT launch it here.
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
from peregrine_racing_inc8 import PeregrineRacingInc8
from inc8_critic_width import maybe_widen_critic

# torch backend (DR-aware, differentiable mirror; parity-proven to the numpy plant). Same rationale
# as peregrine_train_racing.py: the numpy backend ignores the per-env DR tensors.
_dyn.DYNAMICS_ALIAS["peregrine_plant"] = lambda cfg, device: PeregrinePlantDynamics(
    cfg, device, backend="torch")
_env.ENV_ALIAS["peregrine_racing"] = PeregrineRacing            # keep inc7 available
_env.ENV_ALIAS["peregrine_racing_inc8"] = PeregrineRacingInc8   # the inc8 case-C env


class GuardedPPO(PPO):
    """PPO + a non-finite-gradient guard (verbatim from peregrine_train_racing.GuardedPPO): one
    poisoned minibatch costs one skipped optimizer step instead of NaN'ing the weights."""

    NAN_ABORT_AFTER = 200

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # CRITIC-WIDTH A/B (capacity for PPO stability): rebuild ONLY the critic value-net wider when
        # +algo.critic_hidden_dim=[...] is set; unset -> no-op (byte-identical). Done BEFORE the nan-guard
        # wraps self.optim.step, so the re-bound guard wraps the (possibly rebuilt) optimizer.
        maybe_widen_critic(self, kwargs.get("cfg", args[0] if args else None))
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

    orig_save = agent.save
    cc = cfg.dynamics.controller
    env = getattr(self, "env", None)
    obs_dim = int(getattr(env, "obs_dim", 0)) if env is not None else 0
    r5_arm = str(getattr(getattr(cfg, "env", object()), "r5_arm", "")).upper() or None
    inc8_on = bool(getattr(getattr(cfg, "env", object()), "inc8", False))

    def save_with_sidecar(path):
        orig_save(path)
        # S7: the sidecar carries the OBS-DIM the checkpoint was trained with, so deploy/load gates on
        # it (inc7 17-dim loads unchanged; inc8 ships 20-dim) + the action bounds (review F1).
        sidecar = {"act_max_thrust": float(cc.max_normed_thrust),
                   "act_max_rate": float(cc.max_roll_rate),
                   "obs_dim": obs_dim, "inc8": inc8_on, "r5_arm": r5_arm}
        with open(os.path.join(path, "actor.json"), "w") as f:
            json.dump(sidecar, f)

    agent.save = save_with_sidecar

    orig_step = agent.step
    counter = {"i": 0}

    def step_with_periodic_save(*a, **k):
        # LOOK-AT GAIN-WARMUP: expose the 0-based PPO update index to the env BEFORE the rollout so it can
        # ramp the look-at gain over the first lookat_warmup_updates updates. No-op when warmup is OFF
        # (the env's factor is forced to 1.0 and ignores this) and skipped for non-inc8 envs (no attr).
        if env is not None and hasattr(env, "_ppo_update"):
            env._ppo_update = counter["i"]
        out = orig_step(*a, **k)
        counter["i"] += 1
        if counter["i"] % max(int(cfg.save_freq), 1) == 0:
            if _weights_finite(agent):
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
                print("[lifeline] weights non-finite -- emergency save skipped")
        except Exception:
            traceback.print_exc()
        raise


TrainRunner.run = _run_with_lifelines


# ---------------------------------------------------------------------------------------------
# Cosmetic ONNX-export RC guard (memory:434 / project_phase2_rl_vision_decisions.md:434).
#
# diffaero's runner.close() runs, as its LAST step, an ONNX export of the trained policy
# (close() -> agent.export() -> PolicyExporter.export() -> torch.onnx.export()). The committed
# cfg/dynamics/quad.yaml ships action_frame="body" (body-frame CTBR rates -- intentional; see
# rl/diffaero_dynamics.py), but diffaero's utils/exporter.py post_process only implements
# "local"/"world" and RAISES `ValueError: Unknown action frame: body` when torch.onnx.export
# re-traces through it. The checkpoint save AND the TorchScript/.pt2 export both SUCCEED first;
# only this trailing ONNX re-trace raises -- poisoning the process exit code (RC=1) AFTER a fully
# successful train + checkpoint (observed: inc8 GPU smoke job 3273374, 300/300 updates, no NaN).
#
# We keep the diffaero clone PRISTINE and guard ONLY this one export failure here in the wrapper,
# mirroring the existing monkeypatch-lifeline pattern: a SUCCESSFUL train MUST exit 0. The guard
# is deliberately NARROW -- it re-raises anything that is not unambiguously the action-frame
# ONNX-export ValueError (matched on BOTH the unique diffaero message AND an export/onnx call-site
# in the traceback). A real failure (NaN-abort, a construction/shape error, an emergency-save
# failure, any other close()-time error) therefore still propagates a non-zero RC.
# ---------------------------------------------------------------------------------------------

def _is_onnx_action_frame_export_error(exc: BaseException) -> bool:
    """True ONLY for diffaero's `ValueError: Unknown action frame: <frame>` raised on the trailing
    ONNX export. Requires BOTH the unique message and an export/onnx frame in the traceback, so an
    unrelated ValueError (or a real training failure) can never be swallowed."""
    if not isinstance(exc, ValueError):
        return False
    if "action frame" not in str(exc).lower():
        return False
    tb_text = "".join(traceback.format_tb(exc.__traceback__)).lower()
    return any(marker in tb_text for marker in ("onnx", "export", "exporter"))


_orig_close = TrainRunner.close


def _close_guarding_onnx_export(self, *a, **k):
    try:
        return _orig_close(self, *a, **k)
    except Exception as exc:  # noqa: BLE001 -- re-raised below unless it is the known cosmetic crash
        if _is_onnx_action_frame_export_error(exc):
            print('[onnx-guard] diffaero\'s trailing ONNX export does not support '
                  'action_frame="body" (ValueError: %s). Train + checkpoint + TorchScript/.pt2 '
                  'export already SUCCEEDED; treating this cosmetic export-only failure as '
                  'non-fatal so a successful run exits 0. '
                  '(memory:434 / project_phase2_rl_vision_decisions.md:434)' % exc)
            return None
        raise


TrainRunner.close = _close_guarding_onnx_export

from diffaero.script.train import main  # noqa: E402  (must follow the registrations above)

if __name__ == "__main__":
    main()
