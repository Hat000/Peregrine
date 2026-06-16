"""inc8 CRITIC-WIDTH knob -- A/B the critic value-net capacity for PPO stability (the 2/3-seed
early-collapse: entropy collapses + value_loss spikes to 246-412 at steps ~100/800). A wider critic
gives lower-variance value targets, which can damp that instability. The ACTOR is never touched (a
wider actor tends to WORSEN PPO stability -- the opposite of the goal).

WHY A REBUILD (not a plain Hydra override): our launcher registers GuardedPPO(diffaero PPO) and builds
it with obs_dim only, i.e. the SYMMETRIC StochasticActorCriticV, whose actor AND critic are both built
from the SAME `cfg.network` (shared `hidden_dim`). A bare `network.hidden_dim=[...]` override would
therefore widen BOTH nets. To widen ONLY the critic we rebuild `agent.critic` after construction with a
private network cfg (hidden_dim replaced), reusing diffaero's own `CriticV` so it constructs identically
to the stock path. The critic input dim is READ BACK from the already-built critic, so this is correct
whether the critic is symmetric (obs-input) or, if a future run switches to AsymmetricPPO, state-input.

DEPLOY-SAFE: the critic is training-only (it saves to `critic.pth`; deploy loads `actor.pth`). Widening
it cannot change the shipped policy.

🚩 Unset == byte-identical: with no `+algo.critic_hidden_dim=...` the off-path returns BEFORE importing
omegaconf/diffaero or touching `agent`/`optim` -- a true no-op. Heavy imports are deferred into the
on-path so this module imports clean on the dev laptop (no torch/omegaconf/diffaero required to import).

OVERRIDE STRING (dispatch):  +algo.critic_hidden_dim=[512,256]
  (`+` because the key is not in cfg/algo/ppo.yaml; `algo.` because the agent cfg is `cfg.algo`.)
"""
from __future__ import annotations

from typing import List, Optional


def normalize_hidden_dims(value) -> Optional[List[int]]:
    """Normalize the cfg value for ``critic_hidden_dim`` to a list of positive ints (or None when unset).

    None -> None (the OFF sentinel). A bare int N -> [N]. A list/tuple/OmegaConf ListConfig -> [int, ...].
    Raises ValueError on an empty or non-positive spec (an obvious mis-typed override should fail loud,
    not silently build a degenerate net). Pure (no torch/omegaconf needed for the list/int paths)."""
    if value is None:
        return None
    try:                                            # OmegaConf ListConfig -> plain list (deferred import)
        from omegaconf import ListConfig
        if isinstance(value, ListConfig):
            value = list(value)
    except Exception:                               # pragma: no cover - omegaconf absent (laptop import)
        pass
    seq = list(value) if isinstance(value, (list, tuple)) else [value]
    dims = [int(x) for x in seq]
    if not dims or any(d <= 0 for d in dims):
        raise ValueError(f"critic_hidden_dim must be a positive int or list of positive ints, got {value!r}")
    return dims


def maybe_widen_critic(ppo, cfg) -> bool:
    """Rebuild ONLY the critic value-net wider when ``cfg.critic_hidden_dim`` is set; else no-op.

    Returns True if the critic was rebuilt, False if left untouched (the byte-identical default). MUST be
    called inside the PPO ctor AFTER ``super().__init__`` (so ``agent``/``optim`` exist) and BEFORE any
    optimizer-step wrapping (so a re-bound nan-guard wraps the rebuilt optimizer). ``cfg`` is the agent
    cfg (== ``cfg.algo``), carrying ``.network`` / ``.lr`` / ``.eps`` and the optional ``.critic_hidden_dim``."""
    dims = normalize_hidden_dims(getattr(cfg, "critic_hidden_dim", None))
    if dims is None:
        return False                                # byte-identical: stock critic + optimizer stand

    import torch
    from omegaconf import OmegaConf
    from diffaero.network.agents import CriticV

    critic_backbone = ppo.agent.critic.critic       # CriticV.critic == the backbone net (MLP for our cfg)
    input_dim = getattr(critic_backbone, "input_dim", None)
    if input_dim is None:                           # diffaero network API moved -> fail loud + actionable
        raise RuntimeError(
            f"[critic-width] could not read critic input dim from {type(critic_backbone).__name__}."
            "input_dim -- the diffaero network API may have moved; reconcile rl/inc8_critic_width.py "
            "against the Adroit clone /scratch/network/fl3689/diffaero_repo before using this knob.")
    old_hidden = list(getattr(cfg.network, "hidden_dim", []))
    device = next(ppo.agent.parameters()).device

    # private critic cfg = a copy of the network cfg with hidden_dim replaced (every other field --
    # name, any rnn/cnn keys -- is preserved, so the rebuild matches the stock backbone for any name).
    net_cfg = OmegaConf.create(OmegaConf.to_container(cfg.network, resolve=True))
    net_cfg.hidden_dim = dims
    ppo.agent.critic = CriticV(net_cfg, input_dim).to(device)

    # the Adam optimizer was built over the OLD critic params; rebuild it so it tracks the new ones. At
    # construction there is no optimizer state yet, so this equals a single from-scratch build (and the
    # actor params, unchanged, re-enter with empty state exactly as before).
    ppo.optim = torch.optim.Adam(ppo.agent.parameters(), lr=cfg.lr, eps=cfg.eps)

    n_params = sum(p.numel() for p in ppo.agent.critic.parameters())
    print(f"[critic-width] critic value-net REBUILT: hidden_dim {old_hidden} -> {dims} "
          f"(input_dim={input_dim}, params={n_params}); ACTOR UNCHANGED; optimizer rebuilt over new params.")
    return True
