"""VQ2 egocentric launcher: register + train the ego env (component C) via the ENV_ALIAS / monkeypatch
pattern, exactly like rl/peregrine_train_inc8.py. ZERO edits to the pristine diffaero clone.

  * dynamics: DYNAMICS_ALIAS["peregrine_plant"] -> PeregrinePlantDynamics  (system-ID'd plant + DR)
  * env:      ENV_ALIAS["peregrine_racing_ego"] -> PeregrineRacingEgo       (VQ2 egocentric env)
  * algo:     AGENT_ALIAS["appo"]               -> the privileged-critic APPO (asymmetric)

Typical invocation (default OFF == byte-identical inc7; turn ON with +env.ego=true):

  python peregrine_train_ego.py \
      env=racing env.name=peregrine_racing_ego +env.ego=true \
      +env.standing_start_frac=1.0 \
      dynamics=quad dynamics.name=peregrine_plant dynamics.g=9.80665 \
      algo=appo n_envs=2048 n_updates=6000 headless=True device=0

The ladder is a separate module (rl/vq2_ego_curriculum.py) driven by the sbatch (DESIGN.md §E).

======================================================================================================
FOLDED-FORWARD FIX #1 -- the LOAD-BEARING appo / privileged-critic assert, wired LIVE here.
======================================================================================================
The assert lives in ego_reward.assert_privileged_critic_connected (exposed on the env as
env.assert_appo_critic) but is INERT unless something CALLS it at runtime. This launcher wraps
TrainRunner.run (the inc8 lifeline pattern) and, right AFTER the agent is built (self.agent exists),
reads the BUILT critic's input_dim and calls env.assert_appo_critic(algo_name, critic_input_dim). A
run launched with algo=ppo (GuardedPPO's SYMMETRIC obs-only critic -- the inc8 seed-collapse root
cause) or with an accidentally-symmetric critic therefore raises AssertionError at startup, BEFORE any
compute is burned, instead of silently training a damped seed-collapsed policy.

======================================================================================================
FOLDED-FORWARD FIX #2 -- the WARM-START LIFELINE, wired LIVE here (the ego ladder's stage->stage chain).
======================================================================================================
The ego curriculum (rl/vq2_ego_curriculum.py) is a stage->stage WARM-STARTED ladder: each stage
CONTINUES from the previous stage's checkpoint via ``+init_from=<prev>/checkpoints`` (the runner's
end-of-run save dir, which holds actor.pth + critic.pth + actor.json -- the exact set maybe_warmstart
needs). WITHOUT wiring the lifeline this launcher would leave +init_from a DEAD hydra key (hydra's '+'
adds it in struct mode, then NOTHING reads it), so every ladder stage would train FRESH -- silently
wasting the whole warm-start chain. This wrapper mirrors the PROVEN inc8 lifeline
(rl/peregrine_train_inc8.py::_run_with_lifelines) for the ego generation:

  * PERIODIC SAVE (every save_freq updates) -> ``<logdir>/periodic`` (+ periodic_prev) + EMERGENCY save
    on any exception: the crash-resilience half. (The runner ALSO writes its own ``<rundir>/checkpoints``
    end-of-run save -- that is the dir the NEXT stage's +init_from points at.)
  * WARM-START LOAD: maybe_warmstart(agent, env, cfg) loads actor+critic from ``cfg.init_from`` into the
    freshly-built agent (weights-only; optimizer + rollout buffer stay fresh). UNSET => byte-identical
    fresh init (the off-path touches nothing). The ego env has no look-at gain-warmup, so the inc8
    _force_lookat_warmup_off inside maybe_warmstart is a safe no-op here.
  * LOGSTD RESET (``+warmstart_reset_logstd`` [+ _std]): after the load, reset the exploration logstd to
    an intermediate level so the transferred MEANS carry over but the ground-down exploitation schedule
    does NOT (the audit-A3 collapse fix). Applied here in the launcher so it fires regardless of which
    inc8_warmstart.py version is on PYTHONPATH (the newer maybe_warmstart also applies it -- an
    idempotent double-set of the same raw value; harmless).
  * CRITIC WARMUP (``+critic_warmup_updates=N``): zero every actor-named gradient for the first N PPO
    updates so ONLY the critic adapts to the new stage's return scale before the policy moves (the
    audit-A1 boundary-divergence fix). Unset/0 => byte-identical.

These are exactly the keys rl/peregrine_vq2_ego.sbatch's BOUNDARY_OV passes
(+warmstart_reset_logstd=true +warmstart_reset_logstd_std=0.18 +critic_warmup_updates=100).
"""
import os
import traceback

import torch

import diffaero.algo as _algo          # noqa: E402
import diffaero.dynamics as _dyn        # noqa: E402
import diffaero.env as _env             # noqa: E402
from diffaero.utils.runner import TrainRunner
from diffaero_dynamics import PeregrinePlantDynamics
from peregrine_racing import PeregrineRacing
from peregrine_racing_ego import PeregrineRacingEgo
from inc8_warmstart import maybe_warmstart
# NOISE-CEILING lever (RC2 fix): resolve_noise_anneal is a pure getattr (None when +algo.noise_anneal is
# unset -> byte-identical off-path, no torch); apply_noise_schedule clamps actor_logstd to the scheduled
# std ceiling BEFORE each rollout. Without this the constant entropy bonus drives actor_logstd to the
# LOG_STD_MAX ceiling (std=exp(2)=7.39, bang-bang) with nothing capping it -- the ego single_gate collapse.
from inc8_noise_anneal import resolve_noise_anneal, apply_noise_schedule

try:                                    # newer inc8_warmstart.py exports the logstd-reset helper
    from inc8_warmstart import reset_actor_logstd
except ImportError:                     # older inc8_warmstart.py predates it -> self-contained fallback
    import math as _math

    # tanh-squash bounds -- PINNED to diffaero network/agents.py (LOG_STD_MAX=2, LOG_STD_MIN=-5); the
    # squash is logstd = MIN + 0.5*(MAX-MIN)*(tanh(raw)+1). Verbatim from the newer inc8_warmstart.py so
    # +warmstart_reset_logstd fires the same whether the synced inc8_warmstart.py is old or new.
    _LOG_STD_MIN, _LOG_STD_MAX = -5.0, 2.0

    def _raw_logstd_for_std(target_std: float) -> float:
        x = (_math.log(target_std) - _LOG_STD_MIN) / (0.5 * (_LOG_STD_MAX - _LOG_STD_MIN)) - 1.0
        x = max(min(x, 1.0 - 1e-9), -1.0 + 1e-9)
        return _math.atanh(x)

    def reset_actor_logstd(agent, target_std: float = 0.18) -> int:
        """Set every ``actor_logstd`` RAW parameter so the squashed per-dim std equals ``target_std``.
        Returns the count reset (0 -> loud warning: layout changed, reset silently missed)."""
        raw = _raw_logstd_for_std(target_std)
        module = getattr(agent, "agent", agent)
        n_reset = 0
        for name, p in module.named_parameters():
            if "actor_logstd" in name:
                p.data.fill_(raw)
                n_reset += 1
                print(f"[warmstart] RESET exploration: {name} -> raw {raw:.4f} (squashed std "
                      f"~{target_std:.3f}/dim -- intermediate exploration on transferred means).")
        if n_reset == 0:
            print("[warmstart] WARNING: warmstart_reset_logstd requested but NO actor_logstd parameter "
                  "found -- network layout changed? Exploration was NOT reset.")
        return n_reset

_dyn.DYNAMICS_ALIAS["peregrine_plant"] = lambda cfg, device: PeregrinePlantDynamics(
    cfg, device, backend="torch")
_env.ENV_ALIAS["peregrine_racing"] = PeregrineRacing            # keep inc7 available
_env.ENV_ALIAS["peregrine_racing_ego"] = PeregrineRacingEgo     # the VQ2 egocentric env


# ================================================================================================
# FOLDED-FORWARD FIX #1: fire the appo / privileged-critic assert right after the agent is built.
# ================================================================================================
def _algo_name(cfg) -> str:
    """Best-effort extraction of the selected algo name from the resolved cfg. The hydra group choice
    (algo=appo) populates cfg.algo.name; fall back to the group node's _target_ / str."""
    algo_cfg = getattr(cfg, "algo", None)
    name = getattr(algo_cfg, "name", None)
    if name:
        return str(name)
    tgt = getattr(algo_cfg, "_target_", None)
    return str(tgt) if tgt else str(algo_cfg)


def _built_critic_input_dim(agent):
    """Read the input_dim of the BUILT critic value-net (the same backbone path inc8_critic_width uses:
    agent.agent.critic.critic.input_dim). Returns None if the diffaero network API has moved (the assert
    then falls back to the env's own state_dim vs obs_dim check on a None-safe path -> loud, not silent)."""
    try:
        critic_backbone = agent.agent.critic.critic
        return getattr(critic_backbone, "input_dim", None)
    except Exception:                               # pragma: no cover - API drift
        return None


def _weights_finite(agent) -> bool:
    return all(torch.isfinite(p).all() for p in agent.agent.parameters())


_orig_run = TrainRunner.run


def _run_with_ego_lifelines(self):
    """Wrap TrainRunner.run for the ego generation. AFTER the agent is built and BEFORE the training
    loop it (in order):
      (fix #1) fires env.assert_appo_critic so a ppo / symmetric-critic run raises AssertionError
               instead of silently seed-collapsing;
      (fix #2) installs the WARM-START LIFELINE -- periodic/emergency saves, +init_from load,
               +warmstart_reset_logstd, +critic_warmup_updates -- so the ego ladder's stage->stage
               warm-start chain is LIVE (mirrors rl/peregrine_train_inc8.py::_run_with_lifelines).
    Non-ego runs (``+env.ego`` unset/false) take the byte-identical fresh-init path (no lifeline)."""
    env, agent, cfg = getattr(self, "env", None), getattr(self, "agent", None), self.cfg
    ego_on = bool(getattr(getattr(cfg, "env", object()), "ego", False))

    # ---- FIX #1: appo / privileged-critic assert (only meaningful on the ego path) ----
    if env is not None and agent is not None and hasattr(env, "assert_appo_critic") and ego_on:
        critic_input_dim = _built_critic_input_dim(agent)
        if critic_input_dim is None:
            raise RuntimeError(
                "[ego-launcher] could not read the built critic input_dim (agent.agent.critic.critic."
                "input_dim) -- the diffaero network API may have moved; reconcile peregrine_train_ego.py "
                "against the Adroit clone before training. The appo/privileged-critic assert MUST run.")
        env.assert_appo_critic(_algo_name(cfg), int(critic_input_dim))
        print(f"[ego-launcher] appo/privileged-critic assert PASSED: algo={_algo_name(cfg)!r}, "
              f"critic_input_dim={int(critic_input_dim)} == state_dim={env.state_dim} "
              f"(!= obs_dim={env.obs_dim}).")

    # ---- non-ego (OFF) path: byte-identical fresh-init trainer, no lifeline ----
    if not ego_on or agent is None:
        return _orig_run(self)

    # ================================ FIX #2: the warm-start lifeline ================================
    logger = self.logger

    # NOISE CEILING (RC2): resolve the std-ceiling schedule (None when +algo.noise_anneal unset ->
    # byte-identical). Applied per-update BELOW, BEFORE each rollout, so actor_logstd cannot run away to
    # the LOG_STD_MAX bang-bang ceiling. Mirror of rl/peregrine_train_inc8.py's proven wiring.
    noise_sched = resolve_noise_anneal(cfg)
    if noise_sched is not None:
        print(f"[noise-anneal] ON: {noise_sched}")

    # PERIODIC SAVE every save_freq updates -> <logdir>/periodic (+ periodic_prev). This is the
    # crash-resilience save; the runner ALSO writes its own <rundir>/checkpoints end-of-run save (the
    # dir the NEXT stage's +init_from points at). Wrap agent.step (the same hook inc8 uses).
    orig_step = agent.step
    counter = {"i": 0}

    def step_with_periodic_save(*a, **k):
        # NOISE CEILING (RC2): clamp actor_logstd to the scheduled std ceiling BEFORE this rollout so the
        # sampled actions obey it (OFF -> noise_sched None -> skipped, byte-identical). Mirrors inc8.
        if noise_sched is not None:
            nv = apply_noise_schedule(agent, counter["i"], noise_sched)
            if counter["i"] % max(int(cfg.log_freq), 1) == 0:
                print(f"[noise-anneal] update {counter['i']}: std_ceil={nv['std_ceil']:.4f} "
                      f"entropy_weight={nv['entropy_weight']:.5f} (progress={nv['progress']:.2f})")
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

    # WARM-START: load actor+critic from cfg.init_from into the freshly-built agent (weights-only;
    # optimizer + rollout buffer stay fresh). UNSET => no-op, BYTE-IDENTICAL fresh init. The ego env has
    # no look-at gain-warmup, so the inc8 _force_lookat_warmup_off inside maybe_warmstart is a safe
    # no-op. Called AFTER the agent + env are built, BEFORE the first training step (inc8 contract).
    warmstart_from = maybe_warmstart(agent, env, cfg)
    if warmstart_from is not None:
        print(f"[lifeline] CONTINUING ego training from warm-started weights: {warmstart_from}")
        # LOGSTD RESET after the load (+warmstart_reset_logstd): transfer the MEANS, reset the
        # ground-down exploitation schedule to an intermediate level so the new stage keeps escape
        # velocity. Applied here so it fires regardless of the synced inc8_warmstart.py version (the
        # newer maybe_warmstart also applies it -> an idempotent double-set of the same raw value).
        if bool(getattr(cfg, "warmstart_reset_logstd", False)):
            target = float(getattr(cfg, "warmstart_reset_logstd_std", 0.18))
            reset_actor_logstd(agent, target_std=target)

    # CRITIC-ONLY WARMUP (+critic_warmup_updates=N): zero every actor-named gradient for the first N PPO
    # updates so ONLY the critic adapts to the new stage's return scale before the policy moves (the
    # audit-A1 boundary-divergence fix). Unset/0 => byte-identical. Applies to fresh starts too (a
    # random critic's advantages are equally garbage). counter["i"] is the same 0-based PPO-update index.
    critic_warmup = int(getattr(cfg, "critic_warmup_updates", 0) or 0)
    if critic_warmup > 0:
        _cw_state = {"released": False}
        _pre_cw_step = agent.optim.step

        def _critic_warmup_step(*a, **k):
            if counter["i"] < critic_warmup:
                for _n, _p in agent.agent.named_parameters():
                    if "actor" in _n and _p.grad is not None:
                        _p.grad.zero_()
            elif not _cw_state["released"]:
                _cw_state["released"] = True
                print(f"[critic-warmup] RELEASED at update {counter['i']}: actor learning resumes "
                      f"(critic adapted alone for the first {critic_warmup} updates).")
            return _pre_cw_step(*a, **k)

        agent.optim.step = _critic_warmup_step
        print(f"[critic-warmup] ON: actor gradients ZEROED for the first {critic_warmup} PPO updates "
              f"(critic-only adaptation to the new return scale).")

    try:
        return _orig_run(self)
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


TrainRunner.run = _run_with_ego_lifelines

from diffaero.script.train import main    # noqa: E402  (must follow the registrations above)

if __name__ == "__main__":
    main()
