"""inc8 WARM-START knob -- continue training from a converged checkpoint instead of fresh init.

WHY: inc8 S3 (rw_centering=1.0, FRESH seeds) was NOT-GO from two COLD-START artifacts -- 2/3 seeds died
in the look-at gain-warmup (value_loss 432/572 @ step100) and the 1 seed that broke out hit a reward
CLIFF at centering-onset (value_loss 650 @ step1200) and collapsed. A warm-start from the converged S2
seed-2 checkpoint (already past warmup, already pointing at full look-at gain) removes the regime change
that drives BOTH: the policy starts inside the pointing basin, so there is no warmup ramp left to
survive and no "KF suddenly has fixes" discontinuity for centering to slam into.

WHAT: a ROOT-level Hydra key ``+init_from=<ckpt dir>`` that, when set, loads BOTH the actor and critic
state_dicts from the checkpoint into the freshly-built PPO agent BEFORE the training loop, then
continues. The load delegates to diffaero's OWN ``PPO.load(path)`` (-> StochasticActorCriticV.load ->
StochasticActor.load + CriticV.load), the EXACT inverse of the periodic-save ``agent.save(path)``: it
reads ``<path>/actor.pth`` ({"actor_mean": state_dict, "actor_logstd": nn.Parameter}) and
``<path>/critic.pth`` (the critic backbone state_dict). The optimizer (Adam moments) and the rollout
buffer are LEFT FRESH -- the checkpoint carries no optimizer state, and fresh moments are the right
choice for a reward re-pilot (a changed landscape). This is a WEIGHTS-only transfer.

NAMESPACE: ``init_from`` is ROOT-level -- a training-loop concern, exactly like ``n_updates`` /
``seed`` / ``save_freq`` / ``runname`` (all root-level in diffaero; there is NO ``train`` config
group, so a ``train.init_from`` key would land in a dead namespace). The ``+`` prefix is required
because the pristine diffaero config has no such key (same as ``+env.inc8`` / ``+dynamics.dr``).

WARMUP INTERACTION: a warm-started policy already points at full look-at gain. Re-ramping the gain from
0 over the first ``lookat_warmup_updates`` PPO updates (peregrine_racing_inc8.step ->
inc8_reward.lookat_warmup_factor) would DISRUPT that established pointing. So on warm-start we FORCE the
env's look-at gain-warmup to 0 (hold full gain from update 0 == ``lookat_warmup_factor == 1.0`` for
every update), logging loudly if a positive warmup had been requested -- warm-start + ramp is
contradictory, and there is no case where re-ramping a loaded pointing policy is desirable. The
warmstart sbatch also passes ``+env.lookat_warmup_updates=0`` belt-and-braces. The write uses the SAME
``self.env`` reference the trainer already uses to advance ``env._ppo_update`` each update (the proven
S2 warmup path), so it reaches the underlying env's ``_lookat_warmup_updates``.

OFF == BYTE-IDENTICAL: with no ``+init_from`` the off-path returns None on its first effective statement
BEFORE importing torch or touching agent / env / cfg -- a pure, side-effect-free getattr. No checkpoint
read, no weight change, no warmup change, no RNG draw, no I/O. The trainer is bit-for-bit the current
fresh-init trainer. Heavy work is deferred into the on-path so this module imports clean on the dev
laptop (no torch / diffaero needed to import).

NB the SYMMETRIC critic is unchanged: this knob loads weights into whatever agent was built; it does NOT
switch to the appo asymmetric critic (a separate decision) and does NOT widen the critic. If you combine
warm-start with ``+algo.critic_hidden_dim`` the source ``critic.pth`` MUST match that width or
``agent.load`` will shape-error -- warm-start the SAME architecture you trained.

OVERRIDE STRING (dispatch):  +init_from=/scratch/network/fl3689/<run>/periodic
"""
from __future__ import annotations

import os
from typing import Optional


def resolve_init_from(value) -> Optional[str]:
    """Normalize the cfg value for ``init_from`` to a path string, or None when unset (the OFF
    sentinel). ``None`` / ``""`` / ``"none"`` / ``"null"`` (any case) -> None; anything else -> str.
    Pure (no torch / omegaconf), so the off-path stays import-clean and side-effect-free."""
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() in ("none", "null"):
        return None
    return s


def maybe_warmstart(agent, env, cfg) -> Optional[str]:
    """Load actor+critic from ``cfg.init_from`` into the freshly-built PPO ``agent`` before training.

    Returns the resolved checkpoint dir if a warm-start happened, or ``None`` when ``init_from`` is
    unset -- the byte-identical fresh-init default, which touches NOTHING. ``cfg`` is the ROOT config
    (``cfg.init_from``); ``env`` is the training env (its look-at gain-warmup is forced to 0 on a
    warm-start). Call AFTER the agent + env are built and BEFORE the first training step (the trainer
    calls it at the top of the run wrapper, before ``TrainRunner.run``'s ``env.reset()``).

    The load is diffaero's verified inverse-of-save: ``PPO.load(path)`` -> ``agent.load`` ->
    ``actor.load`` (actor.pth: actor_mean state_dict + actor_logstd Parameter) + ``critic.load``
    (critic.pth). Optimizer + rollout buffer stay FRESH (weights-only transfer)."""
    init_from = resolve_init_from(getattr(cfg, "init_from", None))
    if init_from is None:
        return None                       # OFF: byte-identical fresh init -- touch NOTHING.

    # ---- on-path only past here (the off-path above stays import-clean + side-effect-free) ----
    if not os.path.isdir(init_from):
        raise FileNotFoundError(
            f"[warmstart] +init_from={init_from!r} is not a directory. Point it at a checkpoint dir "
            f"holding actor.pth + critic.pth (a trainer 'periodic' / 'periodic_prev' / 'emergency' "
            f"save dir, e.g. <run logdir>/periodic).")
    missing = [fn for fn in ("actor.pth", "critic.pth")
               if not os.path.isfile(os.path.join(init_from, fn))]
    if missing:
        raise FileNotFoundError(
            f"[warmstart] +init_from={init_from!r} is missing {missing}. Warm-start needs a FULL "
            f"agent.save dir (BOTH actor.pth and critic.pth). A deploy-only actor.pth alone is not "
            f"enough -- the critic must resume too, else PPO restarts value learning from scratch "
            f"(which reintroduces the very value_loss spike warm-start exists to avoid).")

    # obs-dim compatibility gate (S7 sidecar): a 17-dim inc7 checkpoint cannot warm-start a 20-dim inc8
    # run (actor_mean first-layer shape mismatch). Fail LOUD + actionable here rather than with a
    # cryptic load_state_dict size error inside agent.load. The sidecar is advisory: absent -> skip the
    # check (agent.load still shape-checks the tensors); present + mismatched -> stop early.
    env_obs_dim = int(getattr(env, "obs_dim", 0) or 0) if env is not None else 0
    ckpt_obs_dim = _sidecar_obs_dim(os.path.join(init_from, "actor.json"))
    if ckpt_obs_dim and env_obs_dim and ckpt_obs_dim != env_obs_dim:
        raise ValueError(
            f"[warmstart] checkpoint obs_dim={ckpt_obs_dim} (from actor.json) != env obs_dim="
            f"{env_obs_dim}. Warm-start needs a matching obs layout (inc8 20-dim from an inc8 20-dim "
            f"checkpoint); you cannot warm-start a 20-dim inc8 run from a 17-dim inc7 checkpoint.")

    agent.load(init_from)                 # PPO.load -> agent.load -> actor.load + critic.load (weights)
    print(f"[warmstart] LOADED actor+critic from {init_from} (ckpt obs_dim={ckpt_obs_dim or '?'}, "
          f"env obs_dim={env_obs_dim or '?'}; optimizer + rollout buffer FRESH -- weights-only "
          f"transfer for the reward re-pilot).")

    _force_lookat_warmup_off(env)
    return init_from


def _sidecar_obs_dim(sidecar_path: str) -> int:
    """obs_dim from a checkpoint's actor.json sidecar (S7), or 0 if absent / unreadable / no key."""
    if not os.path.isfile(sidecar_path):
        return 0
    try:
        import json
        with open(sidecar_path) as f:
            return int(json.load(f).get("obs_dim", 0) or 0)
    except Exception:                     # pragma: no cover - a corrupt sidecar must not block the load
        return 0


def _force_lookat_warmup_off(env) -> None:
    """Hold the look-at gain at full strength from update 0 on a warm-start (the loaded policy already
    points; re-ramping from 0 would disrupt it). The env captured ``lookat_warmup_updates`` at
    construction into ``self._lookat_warmup_updates``; we overwrite it to 0 so
    ``lookat_warmup_factor(idx, 0) == 1.0`` at every update. Same ``env`` reference the trainer uses to
    advance ``env._ppo_update`` (the proven S2 warmup write path). Loud if a positive warmup had been
    requested -- warm-start + ramp is contradictory."""
    if env is None or not hasattr(env, "_lookat_warmup_updates"):
        return
    requested = int(getattr(env, "_lookat_warmup_updates", 0) or 0)
    if requested > 0:
        print(f"[warmstart] OVERRIDING look-at gain-warmup {requested} -> 0: a warm-started policy "
              f"already points at full look-at gain; re-ramping from 0 would disrupt established "
              f"pointing. (Do not set a warmup when warm-starting.)")
    env._lookat_warmup_updates = 0
