# 00 — CORRECTION (read before [07]): the offline eval harness is unfaithful

**Added 2026-07-08, shortly after the rest of the dossier.** Re-deriving the reach-rate claim from
the *authoritative* training metrics exposed a serious discrepancy that invalidates parts of the
hit-map diagnosis in [07] (and the companion artifact). Recording it here rather than silently
rewriting, because the discrepancy is itself an important lesson (it's law [06] L12 — "renders/eval
harnesses lie" — biting again).

## The discrepancy

For the same checkpoint (`vglpan`), the **in-training box-exit** (logged inside the real training env,
the number we have trusted all along and verified via triple-counting, [06] L13) and the **standalone
offline rollout** (`ego_render_rollout.py`, the basis of the [07] hit-map) disagree by ~4× on thread
and by ~1000× on out-of-bounds:

| metric | training (in-loop, trusted) | offline rollout, stochastic DR-on |
|---|---|---|
| success / thread | **19.9%** | 5.5% |
| collision (frame-clip) | **71.3%** | 34.8% |
| plane-miss | 8.7% | 13.7% |
| out-of-bounds | **0.05%** | **46.1%** |

A stochastic offline eval of the same checkpoint should reproduce the training number (~20%); it reads
5.5% with 46% OOB. **`ego_render_rollout.py` is not faithfully reproducing the training environment.**
The drone flies out of bounds ~46% of the time in the offline harness and essentially never (0.05%) in
training. The cause is not yet found (candidates: a config/override not restored from `.hydra`, the
noise-anneal logstd clamp not applied, an arena/oob-bound difference, or a frame/action-scaling
mismatch in the standalone step loop).

## What this invalidates

Everything in [07] and the artifact that is **derived from the offline rollout** is unreliable:
- the "~46% never reach the gate" reach-rate finding — **false**; training oob≈0, floor≈0.
- the "+3.2 m vertical high-bias" — an artifact of the (broken, DR-off) offline harness.
- the specific 2×2 thread/vert-bias numbers.

My offline *reconstruction* also had its own bug (extrapolating a crossing the reset had overwritten),
compounding the harness problem. Treat [07]'s figures as **not trustworthy** until the harness is fixed
and re-validated against the in-loop metrics.

## What is TRUE (the trusted training picture for vglpan)

```
success = exit_thread = 19.9%    collision (frame-clip) = 71.3%
plane-miss = 8.7%                oob = 0.05%    floor ≈ 0    timeout ≈ 0
mean crossing offset (xoff) = 0.88 m
```

**It is a pure centring problem, not a reach-rate problem.** ~100% of drones reach the gate and cross
the plane; **71% clip the frame** (cross at L-inf 0.75–1.36 m, just outside the aperture); only the
tail beyond 1.36 m is a wide "miss". The whole task is to pull the crossing distribution from a ~0.88 m
mean into the ~0.42 m body-effective clean-pass window — which would convert most of that 71% frame-clip
into threads. The [06]/[08]/[09] discussions of an effective ~0.42 m aperture and of the centring
mechanism (parabola + in-run zero-anneal) **stand**; the parts framing reach-rate as a co-equal front
**do not**.

## Consequence for the plan

1. **We currently have no faithful *deterministic* evaluation.** The only offline harness is broken and
   the training box-exit is stochastic. Establishing a trustworthy deterministic-policy metric (an
   in-training periodic `test=True` rollout logged from the real env, or a fixed `ego_render_rollout`)
   is a **prerequisite** to the "improve the deployed policy" goal — you cannot optimise what you cannot
   measure.
2. The forward work is single-front: **centring** (anneal the parabola zero toward the ~0.5 m effective
   aperture + sharpen the endgame noise floor), evaluated on the trusted in-loop box-exit until a
   faithful deterministic metric exists.
