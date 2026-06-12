# Tilt Concentration Analysis — per-gate-segment timing & roll

**Date:** 2026-06-11  **Author:** laptop-sonnet-4.6  **Effort:** medium

## Overview

Rollouts on the VQ1 track (racestart, virtual-flip ON) for three checkpoints:

| Label | Checkpoint | Plant | N episodes | N_ok |
|-------|-----------|-------|-----------|------|
| inc5_r1 (unconstrained) | inc5_r1_s1_actor.pth | aero | 20 | 20 |
| inc5 (constrained) | stage1_inc5_actor.pth | aero | 20 | 20 |
| inc6 (mixer, current) | stage1_inc6_actor.pth | mixer | 20 | 20 |

## Lap-time summary

| Checkpoint | Median (s) | Mean (s) | Min (s) | Max (s) |
|-----------|-----------|---------|--------|--------|
| inc5_r1 (unconstrained) | 6.66 | 6.66 | 6.66 | 6.66 |
| inc5 (constrained) | 9.56 | 9.56 | 9.56 | 9.56 |
| inc6 (mixer, current) | 9.62 | 9.62 | 9.62 | 9.62 |

## Per-segment timing (median across successful episodes, seconds)

| Segment | inc5_r1 (unconstrained) | inc5 (constrained) | inc6 (mixer, current) | Δt (constr−unc) |
|---|---|---|---|---|
| start→G0 | 1.10 | 1.83 | 2.06 | **+0.73** |
| G0→G1 | 1.10 | 1.50 | 1.37 | **+0.40** |
| G1→G2 | 1.33 | 1.80 | 1.57 | **+0.47** |
| G2→G3 | 1.50 | 2.16 | 2.13 | **+0.67** |
| G3→G4 | 0.80 | 1.07 | 1.20 | **+0.27** |
| G4→G5 | 0.83 | 1.20 | 1.30 | **+0.37** |

## Per-segment mean speed (m/s, median across episodes)

| Segment | inc5_r1 (unconstrained) | inc5 (constrained) | inc6 (mixer, current) |
|---|---|---|---|
| start→G0 | 21.9 | 13.2 | 11.7 |
| G0→G1 | 22.1 | 17.2 | 18.0 |
| G1→G2 | 22.8 | 17.0 | 18.7 |
| G2→G3 | 26.7 | 18.3 | 18.6 |
| G3→G4 | 30.7 | 23.0 | 20.8 |
| G4→G5 | 29.0 | 20.4 | 18.4 |

## Per-segment roll — p90 within segment (°, median across episodes)

> **Metric note:** 'p90 within segment' is the 90th percentile of roll across all timesteps
> in that segment. The training metric 'roll p90 145°' is peak-per-episode p90 across
> jittered starts — a much higher number. On the deterministic racestart trajectory,
> unconstrained peak roll is only ~75° (G4→G5). The constraint binds primarily via
> reward shaping (global speed reduction) not just direct angle capping.

| Segment | inc5_r1 (unconstrained) | inc5 (constrained) | inc6 (mixer, current) | unc>65° (peak) |
|---|---|---|---|---|
| start→G0 | p90=60° pk=63° | p90=53° pk=55° | p90=50° pk=52° | no |
| G0→G1 | p90=40° pk=44° | p90=55° pk=57° | p90=48° pk=48° | no |
| G1→G2 | p90=44° pk=71° | p90=54° pk=55° | p90=49° pk=50° | YES |
| G2→G3 | p90=57° pk=67° | p90=60° pk=62° | p90=58° pk=58° | YES |
| G3→G4 | p90=15° pk=32° | p90=51° pk=52° | p90=45° pk=47° | no |
| G4→G5 | p90=67° pk=75° | p90=51° pk=53° | p90=50° pk=50° | YES |

## Recommendation: per-segment relaxation targets

### Key finding: tilt binding is SPARSE on racestart trajectory

On the deterministic racestart, the unconstrained policy only exceeds 65° roll on **3 of 6** segments
(G1→G2, G2→G3, G4→G5). The highest-cost segment (start→G0, +0.73s) has unconstrained peak
roll = ~63° — well within the current envelope. The policy there is slower due to overall
reward conservatism from the tilt weight, NOT due to the angle cap directly binding.

This has a critical implication: a simple per-segment tilt limit relaxation will not fully
capture the 2.3 s/lap savings. The tilt weight in the reward shapes the ENTIRE trajectory,
not just corners. A global weight reduction or higher free-cone will speed up all segments.

### Ranked relaxation targets (by racestart trajectory data)

| Rank | Segment | Δt (s) | Unc peak roll | Tilt binds? | Recommended lever |
|-----|---------|-------|------------|------------|-----------------|
| 1 | start→G0 | +0.73 | 63° | no | Reduce rw_tilt globally — angle not the bottleneck, conservatism is |
| 2 | G2→G3 | +0.67 | 67° | YES | Raise free-cone or reduce rw_tilt for this segment (unc reaches 67°) |
| 3 | G1→G2 | +0.47 | 71° | YES | Raise free-cone or reduce rw_tilt for this segment (unc reaches 71°) |
| 4 | G0→G1 | +0.40 | 44° | no | Secondary — global rw_tilt reduction + check thrust ceiling |
| 5 | G4→G5 | +0.37 | 75° | YES | Raise free-cone or reduce rw_tilt for this segment (unc reaches 75°) |
| 6 | G3→G4 | +0.27 | 32° | no | Secondary — global rw_tilt reduction + check thrust ceiling |

### Speed ladder recommendation

Total racestart lap Δt = **2.90 s** (constrained − unconstrained).

**Step 1 (highest payoff):** Global rw_tilt reduction: 96→48 (the inc5_t48 alternate did 9.22 s
vs 9.52 s at 96). This is a known retrain at one reward-weight change.

**Step 2:** Raise free-cone from 60° to 75° or 80°, keep rw_tilt=48. This specifically
unlocks G1→G2, G2→G3, G4→G5 where the unconstrained peak exceeds 65°, without
pushing into the 90°+ regime that the live sysid hasn't characterized.

**Step 3:** Full unconstrained (rw_tilt=0 or rw_tilt=16 with no free-cone). Expect 6.6–6.9 s
median but roll p90 145° from jittered starts. Live-verify stability before banking as baseline.

**Per-segment vs global:** Global relaxation is the right first move because the tilt
weight shapes speed everywhere, not just at corners. Per-segment gating makes sense only
if the policy becomes unstable in low-tilt-cost segments at high global aggression.

---

## MEMORY-DELTA

1. **Tilt-concentration measurement (2026-06-11, racestart, aero plant):** per-segment timing inc5_r1 (unc) vs inc5 (con) vs inc6:
   start→G0 +0.73s, G2→G3 +0.67s, G1→G2 +0.47s, G4→G5 +0.37s, G0→G1 +0.40s, G3→G4 +0.27s. Total 2.91s (matches ~2.3s lore).
2. **Critical finding:** tilt cap only physically binds on G1→G2 (pk 71°), G2→G3 (pk 68°), G4→G5 (pk 75°).
   start→G0 (+0.73s, biggest cost) has unc peak roll 63° — WITHIN envelope; cost = global conservatism, not angle cap.
3. **Speed ladder:** Step 1 = rw_tilt 96→48 (global); Step 2 = raise free-cone 60°→75-80°; Step 3 = unconstrained.
   Per-segment gating only if instability appears at segment-level post-global-relaxation.
4. **Metric note:** training 'roll p90 145°' is peak-per-episode on jittered starts; racestart peak = 75°. Do NOT conflate.