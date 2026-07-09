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


def _cross_zero_schedule(update_idx: int, n_updates: int, start: float, end: float,
                         hold_frac: float) -> float:
    """PURE geometric (log-linear) decay of the parabola ZERO RADIUS (rw_cross_zero_m) from ``start`` to
    ``end`` over the back ``1-hold_frac`` of training; hold ``start`` for the first ``hold_frac``. Mirrors
    inc8_noise_anneal's std schedule. Shrinking the zero WITHIN a run sharpens the near-centre crossing
    gradient smoothly as the policy centres -- avoiding the discrete warm-start-into-tighter-zero collapse
    (vglp3/vglp3b: a 4->3 STEP detonated the warm-start; a gradual anneal has no discontinuity)."""
    N = max(int(n_updates), 1)
    i0 = hold_frac * N
    span = max(N - i0, 1.0)
    p = min(max((update_idx - i0) / span, 0.0), 1.0)
    if start > 0.0 and end > 0.0:
        return start * (end / start) ** p
    return end


def _resolve_cross_zero_anneal(cfg):
    """Parse the parabola-zero anneal from cfg.env, or None when OFF (byte-identical default). Gated by
    ``+env.cross_zero_anneal`` (truthy); ``+env.cross_zero_start/end/hold_frac`` optional. PURE getattr."""
    env = getattr(cfg, "env", None)
    if env is None or not bool(getattr(env, "cross_zero_anneal", False)):
        return None
    return dict(
        start=float(getattr(env, "cross_zero_start", 5.0)),
        end=float(getattr(env, "cross_zero_end", 0.75)),
        hold_frac=float(getattr(env, "cross_zero_hold_frac", 0.1)),
        n_updates=int(getattr(cfg, "n_updates", 0) or 0),
    )


def _unwrap_env_with(env, attr, max_depth=12):
    """Drill the ``.env`` wrapper chain to the object whose OWN __dict__ holds ``attr`` (e.g. the raw
    peregrine_racing_ego holding ``_egorw``/``_estimator``). DiffAero's TrainRunner wraps the env in
    RecordEpisodeStatistics (utils/runner.py), and the wrapper chain's ``__getattr__`` forwarding does
    NOT reliably reach these attributes from the top -- so ``hasattr(self.env, "_egorw")`` returns False
    and the in-run anneals silently SKIP (the cross-zero anneal was inert the whole 2026-07-08 campaign).
    This walks ``vars(e)['env']`` (instance dict, NOT getattr -> no __getattr__ recursion) to find the
    real holder. Returns the holder object, or None if not found within max_depth."""
    e = env
    for _ in range(max_depth):
        if e is None:
            return None
        d = getattr(e, "__dict__", {})
        if attr in d:
            return e
        e = d.get("env", None)
    return None


def _noise_scale_schedule(update_idx: int, n_updates: int, start: float, end: float,
                          hold_frac: float) -> float:
    """LINEAR ramp of the estimator global noise multiplier (ego_noise_scale) from ``start`` to ``end``
    over the back ``1-hold_frac`` of training; hold ``start`` for the first ``hold_frac``. A NOISE
    CURRICULUM: learn to centre on a clean (or low-noise) signal first, then ramp the measured vision
    noise in so the policy adapts that centring skill to be noise-ROBUST -- the data-motivated response to
    the vglpns0 finding (perfect estimator -> 64% thread / 0.32 m; the ~0.88 m floor is perception-noise-
    limited). LINEAR (not geometric) because start can be 0 (geometric can't cross 0). Continuous -> no
    discontinuity -> avoids the discrete-warm-start collapse (a FIXED intermediate noise detonated: vglpns5
    0.5 -> collapsed basin; the continuous ramp is the L6-safe cure)."""
    N = max(int(n_updates), 1)
    i0 = hold_frac * N
    span = max(N - i0, 1.0)
    p = min(max((update_idx - i0) / span, 0.0), 1.0)
    return start + (end - start) * p


def _resolve_noise_scale_anneal(cfg):
    """Parse the estimator noise-scale anneal from cfg.env, or None when OFF (byte-identical default).
    Gated by ``+env.noise_scale_anneal`` (truthy); ``+env.noise_scale_start/end/hold_frac`` optional."""
    env = getattr(cfg, "env", None)
    if env is None or not bool(getattr(env, "noise_scale_anneal", False)):
        return None
    return dict(
        start=float(getattr(env, "noise_scale_start", 0.0)),
        end=float(getattr(env, "noise_scale_end", 1.0)),
        hold_frac=float(getattr(env, "noise_scale_hold_frac", 0.1)),
        n_updates=int(getattr(cfg, "n_updates", 0) or 0),
    )


def _run_det_eval(self, env, agent, cfg):
    """FAITHFUL deterministic box-exit on the LIVE training env, AFTER training completes.

    Motivation: the training box-exit is STOCHASTIC, and the standalone offline harness
    (ego_render_rollout.py) diverges badly from training (reports ~46% oob where training logs 0.05%),
    so we had NO trustworthy measurement of the DEPLOYED (deterministic-mean) policy. This runs the
    policy with test=True on ``self.env`` -- the exact env instance that produced the trusted training
    metrics -- so it is faithful by construction. It runs post-training (checkpoint already saved), so
    it cannot affect the result. Prints a greppable ``DET_EVAL[...]`` line. ``+eval_det_steps=0`` skips.
    """
    steps = int(getattr(cfg, "eval_det_steps", 300) or 0)
    if steps <= 0 or env is None or agent is None:
        return
    import torch
    try:
        agg, n_ep = {}, 0
        obs = env.reset()
        with torch.no_grad():
            for _ in range(steps):
                action, _ = agent.act(obs, test=True)          # DETERMINISTIC mean (deployed policy)
                obs, _l, _t, info = env.step(env.rescale_action(action))
                sr = info.get("stats_raw", {})
                m = info.get("reset")
                if m is None:
                    continue
                n = int(m.sum().item())
                if not n:
                    continue
                n_ep += n
                for k in ("success_rate", "collision_rate", "miss_rate", "oob_rate"):
                    if k in sr:
                        agg[k] = agg.get(k, 0.0) + float(sr[k].sum().item())
        if n_ep:
            r = {k: agg.get(k, 0.0) / n_ep for k in ("success_rate", "collision_rate", "miss_rate", "oob_rate")}
            print(f"DET_EVAL[{getattr(cfg, 'runname', '?')}] n_ep={n_ep} "
                  f"thread={r['success_rate']:.4f} collision={r['collision_rate']:.4f} "
                  f"miss={r['miss_rate']:.4f} oob={r['oob_rate']:.4f}  (test=True, LIVE env, {steps} steps)")
        else:
            print(f"DET_EVAL[{getattr(cfg, 'runname', '?')}] no episodes completed in {steps} steps")
    except Exception as e:  # never let the post-hoc eval fail a completed run
        print(f"DET_EVAL: FAILED ({type(e).__name__}: {e})")


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

    # PARABOLA-ZERO anneal (Fengyou 2026-07-08): shrink env._egorw.cross_zero_m gradually WITHIN the run
    # (the in-run cure for the discrete-shrink warm-start collapse). Requires the live racing env + parabola
    # on; None (OFF) is byte-identical. env._egorw IS the exact object the reward reads (self._egorw), so
    # mutating cross_zero_m before each rollout takes effect the next step.
    cz_sched = _resolve_cross_zero_anneal(cfg)
    cz_env = _unwrap_env_with(env, "_egorw") if cz_sched is not None else None
    if cz_sched is not None and cz_env is not None:
        print(f"[cross-zero-anneal] ON: {cz_sched} (from {getattr(cz_env._egorw, 'cross_zero_m', '?')})")
    else:
        if cz_sched is not None:
            print("[cross-zero-anneal] requested but _egorw holder not found in env chain -- SKIPPED")
        cz_sched = None

    # NOISE CURRICULUM (Fengyou 2026-07-09): anneal the estimator global noise multiplier ego_noise_scale
    # from start->end WITHIN the run (the data-motivated response to vglpns0 = perception-noise-limited).
    # _estimator.set_noise_scale mutates the frozen EgoEstimatorConfig live; None (OFF) is byte-identical.
    ns_sched = _resolve_noise_scale_anneal(cfg)
    ns_env = _unwrap_env_with(env, "_estimator") if ns_sched is not None else None
    if ns_sched is not None and ns_env is not None:
        print(f"[noise-scale-anneal] ON: {ns_sched} (from {getattr(ns_env._estimator.cfg, 'noise_scale', '?')})")
    else:
        if ns_sched is not None:
            print("[noise-scale-anneal] requested but _estimator holder not found in env chain -- SKIPPED")
        ns_sched = None

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
        if cz_sched is not None:
            czv = _cross_zero_schedule(counter["i"], cz_sched["n_updates"],
                                       cz_sched["start"], cz_sched["end"], cz_sched["hold_frac"])
            cz_env._egorw.cross_zero_m = czv
            if counter["i"] % max(int(cfg.log_freq), 1) == 0:
                print(f"[cross-zero-anneal] update {counter['i']}: cross_zero_m={czv:.3f}")
        if ns_sched is not None:
            nsv = _noise_scale_schedule(counter["i"], ns_sched["n_updates"],
                                        ns_sched["start"], ns_sched["end"], ns_sched["hold_frac"])
            ns_env._estimator.set_noise_scale(nsv)
            if counter["i"] % max(int(cfg.log_freq), 1) == 0:
                print(f"[noise-scale-anneal] update {counter['i']}: ego_noise_scale={nsv:.3f}")
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
        result = _orig_run(self)
        _run_det_eval(self, env, agent, cfg)   # faithful deterministic box-exit on the live env
        return result
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
