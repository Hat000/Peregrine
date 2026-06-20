"""inc8 NOISE-ANNEAL knob -- collapse the PPO exploration noise onto the policy MEAN over training so
the DETERMINISTICALLY-inferred policy (``act(test=True)`` == the deployed/eval policy == the bare
``action_mean``) flies the course, not just the noise-averaged stochastic rollout.

WHY (the rc1 failure, quantified). diffaero's StochasticActor samples rollout actions as
``tanh(mean + std * N(0,1))`` with a STATE-INDEPENDENT learnable ``actor_logstd`` Parameter, squashed
``std = exp(LOG_STD_MIN + 0.5*(LOG_STD_MAX-LOG_STD_MIN)*(tanh(actor_logstd)+1))`` (network/agents.py).
The first FLYING inc8 ("rc1", job 3276449) trained to ~0.50-0.54 STOCHASTIC success but CRASHES ~1 s
after spawn under deterministic inference (sigma_p0 job 3278151: 2-axis reach=0 all 3 seeds). The cause,
measured from the rc1 checkpoints (handoff probe_logstd.py): rc1 seed0's actor_logstd saturated TWO
action channels to ``std = exp(2) = 7.39`` (the LOG_STD_MAX ceiling; raw params 9.0 / 4.3) -- after the
tanh squash those channels are near-random bang-bang in the rollout. The policy learned to fly *in
expectation over that chaos* (feedback on the other channels compensates), so its MEAN is an unvisited,
unstable operating point. Seeds 1/2 sit at a tamer ~0.25 but their means also crash. The entropy bonus
(``entropy_weight=0.01``, constant, no schedule) with no std ceiling is what drove the saturation.

THE FIX. Two composed levers, scheduled per PPO update, applied to the LIVE agent BEFORE each rollout:
  1. CLAMP ``actor_logstd`` to a decaying std CEILING. A loose hold ceiling (``std_hold``, default 0.6)
     from update 0 kills the exp(2)=7.39 saturation pathology while preserving healthy ~0.25
     exploration; then a geometric (log-linear) decay ``std_hold -> std_floor`` (default 0.03) over the
     back ``1-hold_frac`` of training forces the rollout distribution onto the mean, so the policy is
     trained -- and selected -- on near-deterministic actions. CLAMP (not set): PPO may still drive std
     LOWER, never above the schedule.
  2. ANNEAL ``entropy_weight`` linearly ``entropy_hold -> entropy_floor`` (default 0.01 -> 0.0) over the
     same window, so the entropy bonus stops fighting the clamp.

Deterministic deployment/eval uses ONLY the mean (test=True), so ``actor_logstd`` never ships -- this
knob purely changes the TRAINING exploration distribution, which is exactly the lever that decides
whether the mean is flyable. Pair with rl/inc8_snapshots.py (dense retention) +
rl/inc8_select_ckpt.py (deterministic selection) -- selection alone cannot help if training never
produces a deterministic-flyable mean.

OFF == BYTE-IDENTICAL. With ``algo.noise_anneal`` unset/false, ``resolve_noise_anneal`` returns None on
a pure getattr BEFORE importing torch or touching the agent -- no clamp, no entropy change, no RNG draw.
The trainer is bit-for-bit the current inc8 trainer. Heavy work (torch) is deferred into the on-path so
this module imports clean on the dev laptop (no torch/diffaero needed).

🚩 The squash constants below MUST match diffaero network/agents.py StochasticActor.forward
(LOG_STD_MIN=-5, LOG_STD_MAX=2). Pinned by tests/test_inc8_noise_anneal.py round-trip.

OVERRIDE STRINGS (dispatch; ``algo.`` because the agent cfg is cfg.algo, ``+`` because not in ppo.yaml):
  +algo.noise_anneal=true
  [+algo.noise_hold_frac=0.5 +algo.noise_std_hold=0.6 +algo.noise_std_floor=0.03
   +algo.noise_entropy_floor=0.0 +algo.noise_entropy_hold=<default cfg.algo.entropy_weight>]
"""
from __future__ import annotations

import math
from typing import Dict, Optional

# diffaero StochasticActor.forward squash constants (network/agents.py). The map from the raw learnable
# actor_logstd param to the sampling std. KEEP IN SYNC with the clone (round-trip-pinned in tests).
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
_HALF_RANGE = 0.5 * (LOG_STD_MAX - LOG_STD_MIN)   # 3.5

# defaults (calibrated to rc1: healthy std ~0.25, pathological saturation at exp(2)=7.39)
DEFAULT_HOLD_FRAC = 0.5
DEFAULT_STD_HOLD = 0.6
DEFAULT_STD_FLOOR = 0.03
DEFAULT_ENTROPY_FLOOR = 0.0


def param_to_std(p: float) -> float:
    """Forward squash: raw actor_logstd param -> sampling std (for tests / logging)."""
    action_logstd = LOG_STD_MIN + _HALF_RANGE * (math.tanh(p) + 1.0)
    return math.exp(action_logstd)


def std_to_param(std: float) -> float:
    """Inverse squash: target sampling std -> the raw actor_logstd param p with param_to_std(p)==std.
    Used to express a std CEILING as a clamp on the raw parameter. tanh is clipped to (-1,1) so a std
    outside (exp(LOG_STD_MIN), exp(LOG_STD_MAX)) saturates the param instead of NaNing (atanh(+-1)=inf).
    """
    if std <= 0.0:
        return -math.inf
    action_logstd = math.log(std)
    tanh_p = (action_logstd - LOG_STD_MIN) / _HALF_RANGE - 1.0
    tanh_p = min(max(tanh_p, -0.999999), 0.999999)
    return math.atanh(tanh_p)


def schedule_values(update_idx: int, n_updates: int, sched: dict) -> Dict[str, float]:
    """PURE schedule. Given the 0-based update index and total updates, return this update's std ceiling
    and entropy weight. Hold for the first ``hold_frac`` of training, then over [hold_frac*N, N]:
      - std ceiling: geometric (log-linear) decay std_hold -> std_floor
      - entropy weight: linear decay entropy_hold -> entropy_floor
    progress is clamped to [0,1] (so update_idx>=N gives the floor; the hold region gives p=0). No torch.
    """
    N = max(int(n_updates), 1)
    i0 = sched["hold_frac"] * N
    span = max(N - i0, 1.0)
    p = (update_idx - i0) / span
    p = min(max(p, 0.0), 1.0)
    sh, sf = sched["std_hold"], sched["std_floor"]
    # geometric decay in std-space (linear in log-std), guarded for non-positive inputs
    std_ceil = sh * (sf / sh) ** p if (sh > 0.0 and sf > 0.0) else sf
    eh, ef = sched["entropy_hold"], sched["entropy_floor"]
    entropy_weight = eh + (ef - eh) * p
    return {"std_ceil": std_ceil, "entropy_weight": entropy_weight, "progress": p}


def resolve_noise_anneal(cfg) -> Optional[dict]:
    """Parse the noise-anneal config from the ROOT cfg, or None when OFF (the byte-identical default).

    Gated by ``cfg.algo.noise_anneal`` (truthy). Optional overrides ``cfg.algo.noise_*`` fall back to the
    module DEFAULTS; ``entropy_hold`` defaults to the live ``cfg.algo.entropy_weight``. ``n_updates`` (the
    schedule denominator) is read from the root cfg. PURE -- no torch -- so the OFF path stays
    import-clean and side-effect-free."""
    algo = getattr(cfg, "algo", None)
    if algo is None or not bool(getattr(algo, "noise_anneal", False)):
        return None
    entropy_hold = getattr(algo, "noise_entropy_hold", None)
    if entropy_hold is None:
        entropy_hold = getattr(algo, "entropy_weight", 0.01)
    return dict(
        hold_frac=float(getattr(algo, "noise_hold_frac", DEFAULT_HOLD_FRAC)),
        std_hold=float(getattr(algo, "noise_std_hold", DEFAULT_STD_HOLD)),
        std_floor=float(getattr(algo, "noise_std_floor", DEFAULT_STD_FLOOR)),
        entropy_hold=float(entropy_hold),
        entropy_floor=float(getattr(algo, "noise_entropy_floor", DEFAULT_ENTROPY_FLOOR)),
        n_updates=int(getattr(cfg, "n_updates", 0) or 0),
    )


def _actor_logstd(agent):
    """Locate the StochasticActor.actor_logstd Parameter on a (Guarded)PPO / AsymmetricPPO agent.
    agent.agent = Stochastic[Asymmetric]ActorCriticV; .actor = StochasticActor; .actor_logstd = Param."""
    try:
        return agent.agent.actor.actor_logstd
    except AttributeError as exc:
        raise RuntimeError(
            "[noise-anneal] could not find agent.agent.actor.actor_logstd -- the diffaero agent API may "
            "have moved; reconcile rl/inc8_noise_anneal.py against the Adroit clone "
            "/scratch/network/fl3689/diffaero_repo/network/agents.py.") from exc


def apply_noise_schedule(agent, update_idx: int, sched: dict) -> Dict[str, float]:
    """Mutate the LIVE PPO agent for this update: set ``entropy_weight`` and CLAMP ``actor_logstd`` to the
    scheduled std ceiling (clamp_max -> PPO may go lower, never higher than the schedule). Call BEFORE the
    rollout each update (so the SAMPLED actions obey the ceiling). Returns the applied values for logging.
    torch is imported here (on-path only) so the OFF path stays laptop-import-clean."""
    import torch

    vals = schedule_values(update_idx, sched["n_updates"], sched)
    agent.entropy_weight = vals["entropy_weight"]
    p_ceil = std_to_param(vals["std_ceil"])
    logstd = _actor_logstd(agent)
    with torch.no_grad():
        logstd.data.clamp_(max=p_ceil)
    vals["p_ceil"] = p_ceil
    return vals
