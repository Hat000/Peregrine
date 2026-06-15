# inc8 Arm-A 6000-Update Pilot Report — Job 3273437
**Date:** 2026-06-14/15  
**Node:** adroit-h11g3  
**Config:** NUPD=6000 / NENVS=2048 / SEED=0 / arm A / --time=14:00:00  
**Harness commit:** 1a7f9a0 (all 3 pre-checks passed)  
**Wall time:** ~59 min (ran to completion at 6000 updates)  
**Outcome:** COMPLETE — RC 0/0, full trajectory captured

---

## (a) GPU Util — 28.6% (unchanged; accepted for this run)

```
mean_util=28.6%  samples=178  (~59 min of data, full run)
```

Consistent with prior runs (27.9% re-smoke, 27.7% aborted pilot). The metric_vec
sync-fix (1a7f9a0) produced no measurable improvement. The dominant bottleneck
remains `compute_reward_terms` ~10 `.item()` calls/step. Accepted for this run per
the commander's instruction; parent-reward fix still needed before the L0 ladder.

---

## (b) Pointing Trajectory — EMERGES at step ~1000, then PLATEAUS

```
    step   total_reward  pointing_rate  terminal_pointing   fix_rate    entropy  value_loss
       0       -0.32463        0.00000            0.00000    0.00000    0.35913    17.81984
     100       -0.86217        0.00000            0.00000    0.00000    0.53857     1.87057
     200       -0.88391        0.01562            0.00000    0.01074    1.58562    21.81857
     300       -0.75250        0.00000            0.00000    0.00000    1.65858     2.02298
     400       -0.71088        0.00049            0.00000    0.00049    2.01682     1.66905
     500       -0.75357        0.00049            0.00000    0.00049    2.21059     1.94998
     600       -0.55751        0.00000            0.00000    0.00000    2.14164     1.49860
     700       -0.49378        0.00000            0.00000    0.00000    1.82320     2.22940
     800       -0.51600        0.00000            0.00000    0.00000    1.92835     1.18617
     900       -0.48014        0.00000            0.00000    0.00000    2.13092     1.84526
    1000       -0.40884        0.00195            0.00111    0.00098    1.68329     6.23963  ← INFLECTION
    1100       -0.45980        0.04004            0.02617    0.00000    1.67795     1.70310  ← JUMP
    1200       -0.52255        0.01074            0.00478    0.00049    1.69955     1.72114
    1300       -0.46408        0.03174            0.06548    0.00049    1.23075     2.69394
    1400       -0.49681        0.02051            0.04481    0.00049    1.07355     1.27272
    1500       -0.55533        0.03564            0.07777    0.00049    0.73644     2.55175
    1600       -0.47650        0.02979            0.06740    0.00098    0.81769     1.51454
    1700       -0.58225        0.03320            0.07424    0.00000    0.28951     2.49681
    1800       -0.58713        0.03076            0.06992    0.00000    0.50986     1.36915
    1900       -0.48903        0.02344            0.05257    0.00000    0.15481     1.38977
    2000       -0.45761        0.01660            0.03802    0.00000    0.32715     1.30133
    2100       -0.60014        0.03125            0.06760    0.00000    0.07355     1.99738
    2200       -0.68767        0.02686            0.05927    0.00000    0.00961     1.89225
    2300       -0.42410        0.02588            0.05791    0.00098   -0.32139     3.50234  ← entropy goes negative
    2400       -0.65697        0.03076            0.06718    0.00000   -0.48561     2.00469
    2500       -0.54896        0.03125            0.06534    0.00000   -0.49813     2.37869
    2600       -0.49014        0.02148            0.04955    0.00000   -0.79799     1.85231
    2700       -0.60064        0.04004            0.08894    0.00000   -1.00666     3.06033
    2800       -0.61978        0.03418            0.07030    0.00049   -0.96924     1.50884
    2900       -0.58614        0.02832            0.06517    0.00049   -0.68052     4.07563
    3000       -0.53597        0.02441            0.05436    0.00000   -0.75672     2.46190
    3100       -0.68941        0.03369            0.07913    0.00098   -0.75237     2.02028
    3200       -0.46582        0.02148            0.04724    0.00098   -0.71820     1.63632
    3300       -0.50744        0.01904            0.04413    0.00000   -0.71404     1.76711
    3400       -0.49159        0.03076            0.06885    0.00000   -0.93384     2.39871
    3500       -0.57615        0.02344            0.05735    0.00000   -0.76110     2.08289
    3600       -0.60683        0.03369            0.07814    0.00049   -0.57703     1.75791
    3700       -0.53593        0.02832            0.06346    0.00000   -0.79231     2.18152
    3800       -0.51082        0.03174            0.07025    0.00000   -0.77860     1.40280
    3900       -0.36065        0.02490            0.05822    0.00049   -0.67658     1.68689
    4000       -0.52096        0.04004            0.08962    0.00049   -0.60398     1.40426
    4100       -0.60668        0.02246            0.04772    0.00000   -0.89261     2.55179
    4200       -0.49898        0.02295            0.05059    0.00000   -0.96341     2.33178
    4300       -0.50723        0.03564            0.08410    0.00049   -0.83793     2.43658
    4400       -0.62695        0.02490            0.05385    0.00049   -0.81598     2.70261
    4500       -0.51648        0.01953            0.04447    0.00098   -0.55439     2.12905
    4600       -0.55295        0.02832            0.06492    0.00049   -0.88217     1.94606
    4700       -0.59538        0.02344            0.05363    0.00000   -0.84326     2.05102
    4800       -0.52739        0.02002            0.04790    0.00000   -0.63359     2.35332
    4900       -0.55910        0.02930            0.07160    0.00049   -0.65359     1.88120
    5000       -0.49049        0.02930            0.06944    0.00049   -0.59094     2.51920
    5100       -0.51331        0.01318            0.03222    0.00049   -0.58628     1.23443
    5200       -0.59982        0.02246            0.05343    0.00000   -0.45545     1.93962
    5300       -0.54787        0.02490            0.05958    0.00000   -0.71217     1.65517
    5400       -0.58122        0.02539            0.05991    0.00000   -0.66606     2.03139
    5500       -0.53777        0.02637            0.05804    0.00098   -0.67323     2.04380
    5600       -0.60372        0.02881            0.06884    0.00049   -0.84704     1.57849
    5700       -0.59856        0.02393            0.05487    0.00000   -0.64223     1.88497
    5800       -0.56118        0.02979            0.06829    0.00146   -0.85556     2.35450
    5900       -0.51606        0.03125            0.07749    0.00049   -0.74001     1.34184
    5990       -0.63046        0.02441            0.05889    0.00000   -0.74060     2.13861
```

### Trajectory interpretation

**Three phases:**

**Phase 1 (steps 0–1000): Pointing FLAT.**  
Policy explores (entropy rises 0.36→2.2). Pointing metrics are zero — the reward
hasn't shaped orientation yet. total_reward recovers from nadir −0.88 (step 200) to
−0.41 (step 1000) as basic course-following partially emerges.

**Phase 2 (steps 1000–1500): Pointing EMERGES.**  
At step 1000 a sharp inflection: pointing_rate 0.000→0.040 and terminal_pointing
0.000→0.078 within 500 updates. The explicit R5 arm-A 2-axis terminal-lock reward
is clearly shaping behavior. terminal_pointing consistently exceeds pointing_rate
(~0.06 vs ~0.03 at peak), confirming the terminal-lock component is working —
the policy learns to point specifically near gates, not just anywhere.

**Phase 3 (steps 1500–5990): PLATEAU.**  
Both metrics oscillate around a plateau:
- pointing_rate: ~0.020–0.040 (mean ~0.028)
- terminal_pointing: ~0.040–0.090 (mean ~0.065)
- fix_rate: near-zero throughout (~0.001, single blips)
- total_reward: noisy −0.36 to −0.69, no upward trend

The plateau is wide and stable — 4500 updates without further improvement. The policy
is stuck: it learned to orient toward gates but cannot convert that into actual vision
fixes (fix_rate ≈ 0) or positive reward.

**Notable signals:**
- **Entropy goes negative at step ~2300** and stays negative (down to −1.01 at step
  2700). Negative entropy = entropy_coeff × (−H) added to the loss → policy is
  deterministic. PPO entropy regularization has been overwhelmed. The policy has
  collapsed to a near-deterministic behavior that points but doesn't progress.
- **fix_rate ≈ 0 despite pointing_rate ~0.03**: The camera is pointing at gates but
  the estimator surrogate is not registering fixes. This gap (pointing ≠ fixing) may
  indicate the pointing angle is insufficient for the detector threshold, or the
  fix_surrogate's gate-visibility criterion requires tighter alignment than the arm-A
  reward achieves.

---

## (c) RC and Sidecar

| Check | Result |
|---|---|
| PRECHECK_RC | **0** ✅ |
| SMOKE_RC | **0** ✅ |
| obs_dim | **20** ✅ |
| inc8 | **true** ✅ |
| r5_arm | **A** ✅ |

---

## Summary for Commander

**The load-bearing answer: arm A DOES produce camera-pointing, but plateaus.**

- Pointing emerges at ~step 1000 (not from the start — the policy first needs to
  learn basic course-following before the pointing reward can shape behavior).
- terminal_pointing peaks ~0.09, plateaus ~0.065. pointing_rate peaks ~0.04,
  plateaus ~0.028. Neither continues trending upward after step 1500.
- The target from margin-closure is fix_rate ≥ ~0.25 with terminal gate-lock ≥ 60%.
  Current fix_rate ≈ 0.001 — three orders of magnitude below target. Pointing is
  oriented but not converting to fixes.
- Entropy collapse to negative by step 2300 suggests the policy is premature-
  convergence limited, possibly starved of gradient diversity at 28% GPU util.
- The plateau may reflect: (a) insufficient gradient throughput at 28% util,
  (b) arm A's terminal-lock reward being too weak relative to the course-completion
  loss, or (c) the fix_surrogate requiring tighter boresight correction before it
  activates.

**Artifact paths** (on /scratch):
- Log: `/scratch/network/fl3689/peregrine_inc8_smoke.out`
- GPU util: `/scratch/network/fl3689/peregrine_inc8_gpu_util.log`
- Run dir: `/scratch/network/fl3689/diffaero/outputs/train/2026-06-14/23-14-55/`
- Decorated logdir: `…/quad__racing__ppo__mlp__inc8_smoke_A_seed0__0/`
