"""Diagnostic: read rc1's converged actor_logstd and report the rollout action_std it flew under.

No diffaero import needed -- actor.pth is a plain dict {"actor_mean": sd, "actor_logstd": Parameter}.
Reproduces diffaero StochasticActor.forward's tanh-squash map (LOG_STD_MIN=-5, LOG_STD_MAX=2) to turn
the raw learnable param into the actual sampling std. A large std => the policy explored far from its
mean => the deterministic mean is an unvisited, possibly-unstable operating point (the rc1 hypothesis).
"""
import sys, math
import torch

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

def std_from_param(p):
    al = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (math.tanh(p) + 1.0)
    return math.exp(al)

for path in sys.argv[1:]:
    try:
        d = torch.load(path, map_location="cpu", weights_only=True)
        ls = d["actor_logstd"].flatten().tolist()
        stds = [std_from_param(p) for p in ls]
        print(f"{path}")
        print(f"  actor_logstd(raw) = {[round(x,4) for x in ls]}")
        print(f"  action_std        = {[round(x,4) for x in stds]}")
        print(f"  mean_std={sum(stds)/len(stds):.4f}  max_std={max(stds):.4f}  min_std={min(stds):.4f}")
    except Exception as e:
        print(f"{path}  ERROR: {e}")
