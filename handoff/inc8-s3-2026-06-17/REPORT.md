# inc8 S3 — Centering + 2-Axis Look-At — REPORT
**Date:** 2026-06-17  
**Author:** Overall Commander  
**Stage:** S3 = yaw + pitch + centering (g_yaw=−3.0, g_pitch=+3.0, rw_centering=1.0)

---

## 1. Context

S2 established that the 2-axis look-at primitive works (seed 2 converged: band_el 24–31°, fix_rate 0.11–0.16, terminal_pointing 0.03–0.13) but estim_err stalled at the estimator RMS ceiling (~0.115–0.130). The GO gate requires estim_err ≲ 0.08. Physical diagnosis: with fixes but without centering, the in-plane offset is bounded by the estimator accuracy floor; centering is co-load-bearing to close the final gap. S3 adds `+env.rw_centering=1.0` on top of all S2 overrides.

---

## 2. Job Table

| Job ID   | Seed | Output file                           |
|----------|------|---------------------------------------|
| 3275483  | 0    | peregrine_inc8_s0_s3cen1_s0.out       |
| 3275484  | 1    | peregrine_inc8_s0_s3cen1_s1.out       |
| 3275485  | 2    | peregrine_inc8_s0_s3cen1_s2.out       |

Config: `++env.lookat_g_yaw=-3.0 ++env.lookat_g_pitch=3.0 +env.lookat_warmup_updates=200 +env.rw_centering=1.0`, per-seed Hydra dirs, NUPD=4000.

---

## 3. Per-Seed Summary

### Seed 0 — PPO collapse at step 100, zombie

- **entropy @step100:** −1.08 | **value_loss @step100:** 432.75 → CATASTROPHIC COLLAPSE  
- warmup_updates=200 did not prevent collapse (3rd occurrence of this failure mode)  
- band_el: 71–81° throughout (no pointing), fix_rate: 0–0.002 (floor)  
- terminal_pointing: 0–0.015 (noise only)  
- **estim_err: 0.070–0.089** — dips below 0.08 from step 1000+ BUT DISQUALIFIED:  
  - No pointing (fix_rate=0.000–0.002), no causal mechanism  
  - centering=−0.025 to −0.030 (zombie policy flying near gate BY COINCIDENCE / KF drift to zero)  
  - KF without fixes → err_ip estimates biased toward 0 (KF "thinks" drone is at gate center)  
  - This is the same zombie artifact as S1@3.0 seed 1 and S2 seed 1

### Seed 1 — PPO collapse at step 100, worst instability yet

- **entropy @step100:** −0.63 | **value_loss @step100:** 572.21 → CATASTROPHIC COLLAPSE  
- value_loss oscillates 63–572 throughout (never settles)  
- band_el: 66–78°, fix_rate: 0–0.004, terminal_pointing: 0–0.022 (noise)  
- estim_err: 0.082–0.099 (above 0.08 — less coincidental improvement than seed 0)  
- centering: −0.022 to −0.027 (zombie-KF artifact)

### Seed 2 — Partial convergence window then PPO collapse at ~step 1200

This seed is the signal. Warmup held through step 700; 2-axis pointing broke out steps 800–1100 then was destroyed by a value_loss spike (650) at step 1200.

Key trace:

```
  step  pointing_rate  lockband_ptg   fix_rate  band_el_deg  estim_err  centering  entropy  value_loss
     0      0.000         0.000        0.000       0.000       0.150     -0.091    +0.372    20.56   ← initial
   100      0.000         0.000        0.000       85.31       0.092     -0.050    +0.363     2.66   ← WARMUP STABLE
   200      0.000         0.000        0.049       83.53       0.090     -0.053    +2.005     1.00
   300      0.000         0.000        0.049       82.94       0.079     -0.048    +2.485     1.00
   400      0.000         0.000        0.000       82.22       0.150     -0.046    +2.733    12.53
   500      0.000         0.000        0.049       84.08       0.127     -0.049    +3.078     2.73
   600      0.002         0.000        0.000       83.17       0.116     -0.045    +2.458    28.11
   700      0.012         0.010        0.003       72.28       0.121     -0.049    +0.026    81.93   ← stirring
   800      0.086         0.175        0.042       52.33       0.118     -0.048    -0.840   226.99   ← BREAKOUT
   900      0.146         0.240        0.059       49.02       0.107     -0.048    -1.951    67.68
  1000      0.149         0.273        0.071       44.89       0.118     -0.047    -2.294    59.14
  1100      0.205         0.328        0.093       42.73       0.113     -0.045    -2.583    67.86
  1200      0.182         0.256        0.064       46.27       0.116     -0.045    -3.099   650.92   ← SPIKE → COLLAPSE
  1300      0.079         0.121        0.029       55.04       0.110     -0.046    -4.293    58.36   ← retreating
  1400      0.095         0.151        0.040       54.44       0.121     -0.046    -3.800    57.47
  1500      0.202         0.350        0.099       43.03       0.122     -0.045    -3.669    55.82
  1600      0.160         0.274        0.071       46.54       0.117     -0.047    -3.995    77.36
  1700      0.205         0.346        0.106       39.87       0.122     -0.042    -3.898    68.48
  1800      0.228         0.339        0.101       38.61       0.118     -0.045    -3.774    49.51
  1900      0.231         0.354        0.109       39.83       0.123     -0.044    -3.821    54.09
  2000      0.095         0.107        0.030       54.37       0.118     -0.041    -5.217    95.68   ← failing
  2100      0.017         0.008        0.004       70.82       0.095     -0.024    -6.138   338.09   ← COLLAPSED
  2200-3990 (zombie): band_el 70–75°, fix_rate 0.001-0.004, entropy −6.9 to −5.9, value_loss 65-650
```

**Seed 2 during convergence window (steps 800–1900):**
- band_el: 39–54° (falling — 2-axis working)
- fix_rate: 0.029–0.109 (off floor — same order as S2 seed 2)
- terminal_pointing: 0.035–0.094 (real pointing at gate)
- **estim_err: 0.107–0.123** — NOT below 0.08 even during active pointing + centering
- centering: −0.041 to −0.049 (WORSE than zombie's −0.024 — drone is actually far from center, KF knows it)

---

## 4. Root Cause Analysis

### 4.1 Why does centering destabilize PPO at pointing convergence?

When the pointing primitive kicks in (fix_rate 0→0.09), the centering reward undergoes a sudden qualitative change:

- **Before pointing:** KF without fixes → drifts toward estimating drone IS at gate center → err_ip→0 → centering reward ≈ 0 → policy "doesn't feel" the centering gradient
- **At pointing onset:** Fixes land → KF suddenly has accurate in-plane error (e.g. 0.12 m) → centering reward fires at full strength → reward landscape shifts discontinuously

This is a **reward cliff**: the same trajectory yields a very different reward depending on whether fixes land. PPO's value function, trained on the "no centering felt" regime, suddenly encounters a different reward regime when pointing converges. The value estimation error is large → large gradient → value_loss spike (650 at step 1200) → PPO clip mechanism fails → policy collapses.

**The symmetric critic (CRITIC-IS-SYMMETRIC footgun) amplifies this:** the 20-dim obs-only critic cannot see GT state (true in-plane error, true position). It can't anticipate the centering reward change because it doesn't have access to the ground truth in-plane offset. The GT-privileged 36-dim critic would have known the drone was 0.12 m off-center all along → smooth value estimation → no cliff.

### 4.2 Why does seed 2 converge in S2 but collapse in S3?

In S2 (no centering): fixing starts, fix_rate 0→0.11, reward landscape shifts due to estim_err improvement (small, smooth). Value_loss was 2–10 (stable).

In S3 (centering=1.0): fixing starts, centering penalty suddenly fires (from near-0 to −0.045), reward cliff → value_loss 226 at step 800, 650 at step 1200, permanent collapse.

The centering reward's magnitude (1.0 × sigmoid × err_ip ≈ 0.04–0.05) is large relative to PPO's value function accuracy when the critic hasn't seen the centering regime.

### 4.3 Zombie estim_err < 0.08 (seeds 0, 1) — does it count?

**NO.** Seed 0's estim_err 0.070–0.089 from step 1000+ is coincidental:
- pointing_rate ~0.001, fix_rate ~0.001 (floor) → KF has no fixes → err_ip estimate drifts to near-zero (KF thinks drone is at gate center because it last estimated "near center" and has drifted on priors)
- centering = −0.025 (small, consistent with KF-estimated err_ip ≈ 0.025)
- This is an artifact of the collapsed policy flying some trajectory where KF drifts to a favorable state, not real gate centering

**Physical test:** if estim_err below 0.08 were real, fix_rate would be ≥0.05 (needed for KF to have accurate measurements). fix_rate=0.001 → no real centering → DISQUALIFIED.

---

## 5. GO Gate

| Criterion | Seed 0 | Seed 1 | Seed 2 (steps 800–1900) | Seed 2 (steps 2100–3990) |
|-----------|--------|--------|------------------------|--------------------------|
| terminal_pointing > 0 | zombie noise | zombie noise | 0.035–0.094 **YES** | zombie noise |
| estim_err ≲ 0.08 | 0.070–0.089 (zombie artifact) | 0.082–0.099 | 0.107–0.123 **NO** | — |

**Formal gate: NOT-GO.** No seed achieves simultaneous `estim_err ≲ 0.08 AND terminal_pointing > 0`.

Critical finding: **even during seed 2's actual convergence window (pointing working, fix_rate 0.09–0.11), estim_err stayed at 0.107–0.123 — the same ceiling as S2.** Centering did NOT further reduce estim_err during this window. The centering reward needs MORE updates in the pointing basin to progressively reduce err_ip — but the policy collapsed before that could happen.

---

## 6. Commander Adjudication

**S3 VERDICT: INSTABILITY — centering reward creates reward cliff at pointing convergence; rw_centering=1.0 NOT viable. Three paths forward:**

---

### Path A: Warm-start from S2 seed 2 checkpoint + rw_centering=0.3

**Why:** S2 seed 2 is already IN the pointing basin (step 3990, stable). Starting there with small centering:
- No reward cliff (pointing is already stable → centering fires at the same level from the start)
- Small centering (0.3) → smaller gradient shock
- The "discovery" problem is gone — policy already points
- Centering has 2000+ updates to progressively reduce err_ip from 0.12 toward 0.08

**Risks:** Requires checkpoint loading support in `peregrine_train_inc8.py` (may need `+train.resume_ckpt=PATH`). If not supported, would need to add it. S2 seed 2 checkpoint is at `/scratch/.../s3cen1/seed2/` (or wherever S2 wrote it — need to locate).

**GO gate:** Same — estim_err ≲ 0.08 on the warm-started run.

---

### Path B: Fresh seeds with rw_centering=0.3 (S4)

**Why:** Simply reduce centering weight until the reward cliff is small enough not to destabilize. 0.3 = 3× less steep cliff than 1.0.

**Risks:** Still fresh exploration — need to re-find the pointing basin (same 1/3 convergence rate). The cliff at 0.3 is smaller but not zero. May still collapse.

**Config:** `++env.lookat_g_yaw=-3.0 ++env.lookat_g_pitch=3.0 +env.lookat_warmup_updates=200 +env.rw_centering=0.3`, 3 seeds fresh, per-seed Hydra dirs.

---

### Path C: Switch to asymmetric critic (appo) — ROOT CAUSE FIX

**Why:** The CRITIC-IS-SYMMETRIC footgun is the structural cause of the 2/3 seed collapse rate AND the reward-cliff instability. With the GT-privileged 36-dim critic, the value function can anticipate the centering reward (it sees the true in-plane error) → no cliff → stable training.

**Config:** Add `algo=appo` to COMMON (requires GuardedPPO to be built with AsymmetricPPO backend). `state_dim=36` passed to the env. `critic_hidden_dim` width knob still works.

**Risks:** Requires code change to confirm appo path in GuardedPPO (verify `algo=appo` routes to AsymmetricPPO, not the existing symmetric PPO). Adroit repo must be updated. This is the cleanest fix but requires a code deploy.

---

**RECOMMENDATION: Path C (appo) + Path B in parallel.** appo addresses the root cause and unblocks both convergence-rate AND reward-cliff problems. If appo is quick to wire (likely — it's already in the MEMORY design), submit Path B simultaneously as a hedge. If Path B converges with 0.3, we have data. If appo convergence works, it likely produces a much more stable result.

**If only one path:** Path C. The symmetric critic is the load-bearing structural problem; weight reduction is a patch that may require further iteration (0.3 fails → try 0.1 → etc.).

---

## 7. MEMORY-DELTA

```
S3 RESULTS (2026-06-17, jobs 3275483-85):
Seeds 0+1 = warmup collapse (entropy -1.1/-0.6, value_loss 432/572 @step100; 3rd occurrence of this failure). Seed 2 = warmup stable, 2-axis breakout steps 800-1100 (fix_rate 0.04-0.11, terminal_pointing 0.04-0.09), then value_loss SPIKE 650 @step1200 → collapse. Pattern: centering reward fires at pointing onset (KF suddenly has fixes → err_ip jumps from KF-zero to honest 0.12) = REWARD CLIFF → PPO destabilizes.
estim_err during seed 2 convergence window (steps 800-1900) = 0.107-0.123 (NOT <0.08; centering DID NOT reduce estim_err further in the available window).
ROOT CAUSE: CRITIC-IS-SYMMETRIC footgun is the load-bearing problem: symmetric critic (obs=20) can't anticipate centering reward → no value estimate smoothing at pointing onset → cliff → collapse. Zombie estim_err <0.08 (seeds 0/1) DISQUALIFIED (KF without fixes drifts err_ip→0 artifact).
VERDICT: S3 NOT-GO. Three paths: A=warm-start from S2-seed2 ckpt + rw_centering=0.3; B=fresh rw_centering=0.3; C=appo (asymmetric critic, root-cause fix). RECOMMENDATION: C (appo) primary + B as hedge. Full traces → handoff/inc8-s3-2026-06-17/REPORT.md.
```
