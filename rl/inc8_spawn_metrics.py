"""B2/M2a spawn-class success telemetry -- pure helpers (laptop-testable, no diffaero).

WHY (diagnosis RC1, docs/vq2-handoff-diagnosis-2026-07-06): 70% of resets near-spawn 1 m before a
uniformly-random gate INCLUDING the last -> for G=2, 35% of episodes scored 'success' via a
perception-free 1 m dash, inflating every stage verdict. M1 removes the last-gate draw for G>=2;
these metrics PROVE it (class-2 mass -> ~0) and give the uncontaminated verdict signal
(inc8_success_standing) the FLIGHTCHECK gates on.

CLASSES: 0 = standing start (pad) · 1 = near-spawn at a NON-last gate · 2 = near-spawn at the
LAST gate (the trivial-success class; for G=1 every near-spawn is class 2 by definition).

AGGREGATION: per-step-decayed accumulators (EMA over terminated episodes, window ~1/(1-DECAY)
env-steps ~ a few thousand episodes at n_envs=2048). The logged ratio is a valid class success
estimate at EVERY step, independent of how the trainer aggregates loss_components per update
(last-step or rollout-mean -- unverifiable from the laptop, diffaero clone absent). 0/0 is
structurally impossible: empty class -> 0.0 exactly, never NaN.
METRICS-ONLY: no RNG draws, no reward/obs/gradient effect.
"""
from __future__ import annotations

try:
    import torch
except Exception:                       # pragma: no cover - torch absent in some tooling contexts
    torch = None

CLS_STANDING, CLS_NEAR, CLS_NEARLAST = 0, 1, 2
N_CLASSES = 3
DECAY = 0.999   # per-env-step EMA decay; ~1000-step window ~= ~31 PPO updates at l_rollout 32


def spawn_class_of(standing, tg_new, n_gates: int):
    """Class tensor from the FINAL reset_idx spawn draw (post ``tg_new[standing] = 0``). Agnostic to
    the M1 last-gate exclusion: it reads whatever tg_new was actually drawn."""
    cls = torch.ones_like(tg_new, dtype=torch.long)
    cls[standing] = CLS_STANDING
    cls[(~standing) & (tg_new.long() == (n_gates - 1))] = CLS_NEARLAST
    return cls


def update_class_accumulators(ep_w, succ_w, npass_w, spawn_class, reset_mask,
                              success, n_passed, decay: float = DECAY) -> None:
    """In-place decayed accumulation over THIS step's terminated episodes. All-GPU, sync-free."""
    oh = (spawn_class.unsqueeze(1)
          == torch.arange(N_CLASSES, device=spawn_class.device).unsqueeze(0)).to(ep_w.dtype)
    r = reset_mask.to(ep_w.dtype).unsqueeze(1)                    # (N,1)
    ep_w.mul_(decay).add_((oh * r).sum(0))
    succ_w.mul_(decay).add_((oh * r * success.to(ep_w.dtype).unsqueeze(1)).sum(0))
    npass_w.mul_(decay).add_((oh * r * n_passed.to(ep_w.dtype).unsqueeze(1)).sum(0))


def class_rates(ep_w, succ_w, npass_w, eps: float = 1e-6):
    """(success_rate(3,), npass_mean(3,), class_mass_frac(3,)). Empty class -> 0.0, never NaN."""
    denom = ep_w.clamp(min=eps)
    return succ_w / denom, npass_w / denom, ep_w / ep_w.sum().clamp(min=eps)
