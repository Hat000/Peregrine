# PEREGRINE FAILURE-MODE PROFILE

One instrument, calibrated against ground truth, run over the whole flight corpus.

**Corpus.** 616 session directories carrying both `meta.json` and `ego_obs.jsonl` across the five
run trees (9 further directories had `meta.json` but no log; 8 duplicate names were de-duplicated).
615 have >8 ticks. **583 ended in CRASH**, 16 STALLED, 16 SIM_RESET. Not one of the 616 finished the
course. After removing 11 pilot aim-probe flights (below), the census is **N = 572 deaths**.

**Read this first.** The single most important number in this document is not a mode count. It is
this: **the median at-gate death log ends 1.43 m / 0.222 s before the gate plane, and 69% of at-gate
deaths cannot be resolved to a kill axis at all.** Every mode share below is bounded by that.

---

## 1. THE INSTRUMENT

### 1.1 Why the old ones could not work

Both prior instruments measured a *position series* that is not the drone's motion:

* the **vision series** (`y_gate = -(R·rel_flu)·l̂`) is the ESTIMATE's position — it moves when the
  estimate moves, and the estimate's own noise is the same size as the signal;
* the **KF track** is dead-reckoned with no vision fusion (`tsv=null`, RewindKF off), so it moves
  with its own drift.

Neither can separate "the drone moved" from "the number moved", and you cannot fix that by choosing
one of them — you need a third measurement that is anchored to something physically fixed.

### 1.2 The VG (vision + gyro) instrument

The gate is a **fixed world object**. Let `r(t)` be drone→gate in the TRUE body-FLU frame (the
logged `rel_flu`, unflipped). For a fixed point,

```
    ṙ = −ω × r − v_body
```

so the body-frame motion of the gate mixes drone ROTATION (gyro — trusted) with drone TRANSLATION
(the channel `obs[0:3]` lies about). De-rotate with a gyro-propagated attitude `R_wb`:

```
    q(t) := R_wb(t) · r(t) = G − p(t)          (G fixed)
 ⇒  −q(t) = p(t) − G  =  THE DRONE'S OWN TRAJECTORY RELATIVE TO THE GATE
```

No KF, no `obs[0:3]`. Writing `r = ρ·b̂`:

| channel | what carries it | measured noise (rms, terminal window) |
|---|---|---|
| **transverse** (⊥ to the line of sight) | the BEARING; range enters only as a scale | **0.07 m** (p90 0.11) |
| **radial** (along the line of sight) | the RANGE itself | **0.43 m** (p90 1.49) |

That 6× asymmetry is the whole instrument. Corpus-wide the radial/transverse residual ratio has
median **4.6** (p90 13.1) over 580 terminal fits.

**Separation of drone motion from estimate motion.** A real trajectory is acceleration-bounded
(≤ ~40 m/s² = 3.765 g rail + g), so over a ≤0.6 s window a quadratic fits real motion to centimetres.
The fit uses anisotropic Huber weighting (transverse-tight σ = max(0.04, 0.008ρ); radial-loose
σ = max(0.25, 0.12ρ)), further de-weighted by perception quality (×4 radial for `range_src="bbox"` —
the deploy code itself calls that "a floor, not a measurement", centre_emit.py:135-145; ×2.5 for
≤2 corners). **The fitted trajectory is drone motion; the residual is estimate motion.**

Terminal geometry (`terminal.py`) uses a *linear* robust fit over ~1.0 s for the direction of travel,
a separate robust linear fit of RANGE for the closing speed, and carries the transverse offset to the
gate plane over the *known* blind interval only (capped at 0.40 s).

### 1.3 Frame handling — and one load-bearing correction

`rel_flu` lives in the **tilted body frame**. At the flown median pitch of **+23.8°** at the last
fresh fix, a level gate 2.5 m ahead reads ~1.0 m "above" in the body frame. Any vertical geometry
must be gravity-levelled first. Measured on 102 at-gate deaths:

| gate-above-drone, read three ways | p50 | fraction reading >0.75 m ABOVE |
|---|---|---|
| `rel_flu[2]` RAW body frame (pitch-coupled) | **+1.07 m** | **73.5%** |
| gravity-levelled, same tick | −0.09 m | 12.7% |
| VG trajectory carried to the plane | −0.12 m | 20.6% |

Median discrepancy 1.00 m, p90 2.27 m, corr(raw−levelled, pitch) = +0.47.

`z_bias` sign was verified empirically, not assumed: with the correction applied, the confirmed-pass
vertical miss is centred at −0.07…−0.17 m for every `z_bias` group (0.00 / 0.20 / 0.25 / 0.30 / 0.40);
without it the biased groups shift systematically to +0.06…+0.24 m.

### 1.4 VALIDATION — 1151 confirmed gate passes as ground truth

Every RACE_STATUS advance is a certified fact: the drone went through a 1.5 m square opening without
touching it (contact invalidates the run). So at a pass the true miss is |y| < 0.75 m and |z| < 0.75 m.
That makes the corpus's passes a **labelled validation set**, and the readings above 0.75 m are pure
instrument error.

**Frame validation (n = 400 passes):** levelled vertical miss mean −0.114, sd **0.392 m**;
corr(y_v, ρ·sin(pitch)) = **+0.112** (uncorrelated). Control — the same passes read RAW: mean −0.939,
sd 0.604, corr(y_v, ρ·sin(pitch)) = **−0.861**, and **66.2% of CONFIRMED PASSES would be called a
vertical strike**. A 17.8° levelling error would inject 0.63 m of pitch-correlated error at the
median 2.1 m range; the observed sd is 0.39 m and shows no pitch coupling. The frame is right.

**Threshold calibration** (false-positive rate on confirmed passes):

| data-quality gate | pass coverage | crash coverage | FP(&#124;y_v&#124;>0.75) | FP(y_h>0.75) | FP(y_h>1.0) | FP(y_h>1.5) |
|---|---|---|---|---|---|---|
| rms_r≤0.35, s_end≤2.0, extrap≤0.6 | 22% | 19% | 3.1% | 19.6% | 10.2% | 2.0% |
| **rms_r≤0.8, s_end≤3.0  ← adopted** | **51%** | **63%** | **6.3%** | 25.7% | **14.8%** | **2.9%** |
| no gate at all | 100% | 100% | 15.5% | 31.7% | 23.5% | 14.2% |

Adopted: **vertical strike |y_v| > 0.75 m (6.3% FP)**; **lateral strike y_h > 1.50 m (2.9% FP,
conservative)**; the **1.0–1.5 m band is refused a mode name** — 14.8% of confirmed passes read that
high, so it is not separable and is carried as an explicit bracket.

Blackout thresholds were calibrated the same way, on the last second before 291 confirmed advances:
healthy `max_fix_gap` p99 = 0.54 s, `blind_frac` p99 = 0.61. Adopted **max_fix_gap ≥ 0.60 s** or
**blind_frac ≥ 0.70** — both **0.0% FP on healthy flight**.

Separation achieved on the adopted tier: AUC(y_h, crash>pass) = **0.721**, AUC(|y_v|) = **0.731**.

### 1.5 HAND-LABEL VALIDATION

24 deaths were drawn stratified by (gate at death) × (near/far) plus lineage top-ups, and labelled by
hand from the raw tick traces before the classifier was consulted (`handlabels.json` carries the
per-case reasoning). One label was **corrected after the fact and is recorded as such**:
`20260719_075831` — my first read cited a stale v1-instrument number printed by the trace tool rather
than the raw ticks; re-reading the ticks (lat_lev +2.15→+1.07 over the whole approach, never
converging) makes it a lateral miss.

**Agreement: 19/24 = 79%.** Counting the two aim-probe cases as concordant (I labelled them "other /
not a clean baseline"; the classifier labels them "excluded aim probe" — the same judgment,
different word): **21/24 = 88%.**

The five disagreements, in full:

| run | hand | classifier | verdict |
|---|---|---|---|
| `20260724_225312_v19` | other (aim probe) | excluded_aim_probe | same judgment, better word — classifier right |
| `20260725_001004_v19` | at_gate_undetermined | excluded_aim_probe | I had not seen the aim flag — classifier right |
| `20260718_160639_v1` | corner_both | other (mid-leg) | genuine boundary: died 3.9 m out, threshold 3.5 m |
| `20260714_043212_vtrackA` | lateral_diverge | other (mid-leg) | genuine boundary: 3.2 m off line but died 4.7 m out |
| `20260719_224943_v16` | lateral_diverge | at_gate_undetermined | **real instrument limitation, see below** |

The last one matters. Body-frame lateral sat at −1.4…−1.68 m for the whole approach, but the VG
ballistic miss reads 0.30 m — because the drone was **crabbing**: its nose pointed off the line while
its velocity pointed at the gate. `y_h` is the miss the drone's *current velocity* would produce; it
is not the same as body-frame lateral offset, and for a turning or crabbing drone the two legitimately
disagree. Neither reading is wrong; they answer different questions, and this log cannot say which one
the gate frame met.

An earlier classifier version scored **54%**. The two fixes that took it to 79% are both about
refusing false precision: (a) the blackout rule was firing at `max_fix_gap ≥ 0.35 s`, which 26.8% of
*healthy* passes also reach — retightened to the 0%-FP value; (b) the 1.0–1.5 m lateral band was
being given a mode name — now refused.

---

## 2. THE 10-vs-41 SPLIT — DIAGNOSED AND ADJUDICATED

Reproduced on the same detector (`lmin ≤ 0.45 ∧ reopen ≥ 0.30`) over the same 87 analysable v19
fatal approaches:

| instrument | what it measures | fires |
|---|---|---|
| **A** vision-estimate series | where the ESTIMATE was | **17 (20%)** |
| **B** KF dead-reckoned track | where the KF BELIEVED it was | **11 (13%)** |
| **C** VG drone-motion series | **where the DRONE was** | **9 (10%)** |

A∩C = 8, A-only = 9, C-only = 1. (An independent reproduction of the legacy pipeline on its own
pool definition gave A = 36 / B = 14 / A∩B = 9 of 112 — same structure, different pool scope.)

**Why A over-counts.** Injecting each flight's OWN measured estimate noise into its OWN true (VG)
trajectory and re-running the detector: for the 9 **A-only** flights the signature is manufactured by
noise alone on **36% of draws**, against a 6% baseline for flights whose true trajectory does not
fire. The reason is blunt — the estimate's radial noise is **0.43 m (p50), 1.49 m (p90)**, while the
detector's thresholds are `lmin ≤ 0.45 m` and `reopen ≥ 0.30 m`. **The measurement noise is the same
size as the thing being measured.** And the statistic is doubly vulnerable: `lmin` is the *minimum*
of a noisy series (noise biases it down, manufacturing the "was centred" half) while `reopen` is
end-minus-min (the same noise inflates it).

**Why B is not the corrective.** Over one approach the KF displacement disagrees with the VG
displacement by a median **3.90 m** after the best single yaw alignment — **38% of a 10.3 m path**.
Against thresholds of 0.45 and 0.30 m, the KF track carries roughly **ten times** the error of the
signal it is being asked to measure. It does not fire less because it is smoother: its `reopen` p50
is **0.87 m against a true 0.18 m** — it reports *4.8× more* lateral excursion than really happened.
It fires less only because the same drift also inflates `lmin` (1.00 vs a true 0.78), so the
"was-centred" gate fails. **B's apparent agreement with the truth at ~10 was a coincidence of two
errors, not corroboration.**

**Verdict.** Both prior instruments are **RETIRED**. Neither measured drone motion; they disagreed
because their errors have different shapes, not because one was closer to the truth. The adjudicated
count for this signature on the v19 pool is **9/87 = 10%** — numerically near the low end of the old
bracket, arrived at for a completely different reason.

---

## 3. CENSUS

### Data hazards, handled explicitly

1. **`aim_off` (pilot manual offset).** 15 sessions carry non-zero `aim_off`, **360 ticks total**:
   12 on the gate-5 leg (up to [0,10] m, applied from 26.9 m down to 12.2 m range — this is the
   pilot's workaround for the invisible obstacle) and 3 on gate 1 (probes, up to [3,3]).
   **All 360 ticks are excluded from every motion computation.** Where the aim leg is also the fatal
   leg (11 sessions) the whole session is **excluded from the census** as `excluded_aim_probe`; the
   other 4 are retained with the aim ticks removed.
2. **Terminal blindness.** Quantified below, per mode and in aggregate. Nothing is extrapolated
   beyond the known blind interval (cap 0.40 s), and every mode carries the blind distance.

### TABLE 1 — mode × confidence × lineage (N = 572)

| mode | N | % | high | med | low | v1 | vpef | vtrackA | v15 | v16 | v18 | v19 | v20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| launch / release window | **115** | **20.1%** | 93 | 22 | 0 | 2 | 3 | 0 | 18 | 49 | 19 | 21 | 3 |
| vertical undershoot | 9 | 1.6% | 1 | 8 | 0 | 0 | 0 | 0 | 3 | 2 | 3 | 1 | 0 |
| vertical overshoot | 16 | 2.8% | 2 | 14 | 0 | 0 | 1 | 1 | 1 | 8 | 3 | 2 | 0 |
| lateral diverge (conservative) | 44 | 7.7% | 7 | 37 | 0 | 1 | 1 | 5 | 11 | 14 | 2 | 9 | 1 |
| corner (both axes out) | 16 | 2.8% | 0 | 16 | 0 | 0 | 0 | 0 | 1 | 11 | 1 | 3 | 0 |
| **at-gate, AXIS UNDETERMINED** | **193** | **33.7%** | 0 | 25 | 168 | 6 | 13 | 4 | 41 | 76 | 16 | 36 | 1 |
| gate-to-gate blackout | 64 | 11.2% | 8 | 56 | 0 | 1 | 3 | 3 | 15 | 36 | 2 | 4 | 0 |
| obstacle leg (gate-5 leg) | 22 | 3.8% | 0 | 22 | 0 | 0 | 0 | 0 | 10 | 8 | 1 | 3 | 0 |
| other / unclassifiable | 93 | 16.3% | 0 | 0 | 93 | 2 | 6 | 6 | 20 | 30 | 6 | 22 | 1 |
| **TOTAL** | **572** | | | | | 12 | 27 | 19 | 120 | 234 | 53 | 101 | 6 |

### TABLE 2 — share of each lineage's own deaths (%)

| mode | v1 | vpef | vtrackA | v15 | v16 | v18 | v19 | v20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| launch / release | 17 | 11 | 0 | 15 | **21** | **36** | **21** | 50* |
| vertical undershoot | 0 | 0 | 0 | 2 | 1 | 6 | 1 | 0 |
| vertical overshoot | 0 | 4 | 5 | 1 | 3 | 6 | 2 | 0 |
| lateral diverge | 8 | 4 | 26 | 9 | 6 | 4 | 9 | 17* |
| corner | 0 | 0 | 0 | 1 | 5 | 2 | 3 | 0 |
| at-gate undetermined | 50 | 48 | 21 | 34 | 32 | 30 | 36 | 17* |
| blackout | 8 | 11 | 16 | 12 | **15** | **4** | **4** | 0 |
| obstacle leg | 0 | 0 | 0 | 8 | 3 | 2 | 3 | 0 |
| other | 17 | 22 | 32 | 17 | 13 | 11 | 22 | 17* |
| **n** | 12 | 27 | 19 | 120 | 234 | 53 | 101 | **6*** |

\* v20 is 6 flights. Report it, do not conclude from it.

**The only clean lineage trend in the table** is the **blackout collapse: v15 12% → v16 15% → v18 4%
→ v19 4%** (v16+v15 51/354 = 14.4% vs v18+v19 6/154 = 3.9%; Fisher p < 0.001). That is the
vision-emission fix `b13a07f` landing, visible in the death census.

The **launch/release share rose** across the same transition (v15 15% → v16 21% → v18 36% → v19 21%),
and v18/v19 die *earlier* overall (median max_gate 1 and 2 vs v15's 3). Fixing the blackout did not
move the deaths to later gates; it moved them to the launch window and to at-gate.

### TABLE 3 — severity / tails

| mode | n | t_death p50 | p95 | kill-axis magnitude | log ends short of plane, p50 / p90 |
|---|---:|---:|---:|---|---|
| launch / release | 115 | 2.1 s | 2.3 s | — | 1.59 / 4.07 m |
| vertical undershoot | 9 | 3.3 s | 21.1 s | \|Δz\| 0.98 / p90 1.25 m | 1.81 / 2.55 m |
| vertical overshoot | 16 | 5.6 s | 7.1 s | \|Δz\| 0.96 / p90 1.33 m | 1.39 / 2.91 m |
| lateral diverge | 44 | 6.9 s | 10.1 s | lat **1.84** / p90 **2.88 m** | 1.37 / 2.10 m |
| corner | 16 | 4.6 s | 17.6 s | lat 1.81 / p90 3.09 m | 1.21 / 2.19 m |
| at-gate undetermined | 193 | 5.9 s | 13.1 s | lat 0.87 / p90 1.39 (unresolved) | 1.47 / 2.66 m |
| blackout | 64 | 6.6 s | 11.7 s | died 9.9 / p90 19.0 m out | — |
| obstacle leg | 22 | 11.3 s | 13.4 s | died 14.3 / p90 16.3 m out | — |
| other | 93 | 6.4 s | 12.7 s | died 12.8 / p90 19.7 m out | — |

The launch window is extraordinarily tight — **t_death p50 2.1 s, p95 2.3 s** across 115 deaths and
seven checkpoint generations. That is not a distribution of causes; that is one repeatable event.

### TABLE 4 — TERMINAL BLINDNESS (the bound on everything above)

278 deaths (49% of the census) are at-gate. For those:

* distance from the **last observable state** to the gate plane: p25 **1.00**, p50 **1.43**,
  p75 **2.11**, p90 **2.60 m**;
* as time: p50 **0.222 s**, p90 **0.406 s**;
* logs ending >0.5 m short: **90.3%**; >1.0 m: **74.8%**; >1.5 m: **46.0%**; >2.0 m: **28.4%**.

**193 of 278 at-gate deaths (69%) are not resolvable to a kill axis.** Of those, **109 read INSIDE
the 1.5 m opening at the last observable moment** — the whole kill happened in the window the log
never sees. The prior figure of "31 of 78" understated this considerably.

### TABLE 5 — the honest brackets

| family | conservative | + not-separable band | + corner | BRACKET |
|---|---:|---:|---:|---|
| **lateral** | 44 (7.7%) | +67 (y_h 1.0–1.5 m, 14.8% pass FP) | +16 | **44 … 127  (7.7% … 22.2%)** |
| **vertical** | 25 (4.4%) | — | +16 | **25 … 41  (4.4% … 7.2%)** |

**These two modes are NOT cleanly separable with current instrumentation.** The 16 corner cases sit
in both brackets, and the 67-flight lateral band cannot be told from a clean crossing. What would
separate them is stated in §5.

### TABLE 6 — mid-leg deaths by gate: there is probably a SECOND invisible obstacle

Leg lengths measured from the corpus (range at the advance): →g1 17.2 m, →g2 7.7 m, →g3 11.2 m,
**→g4 20.4 m**, **→g5 20.8 m**, →g6 15.1 m.

| leg | n (mid-leg, vision healthy) | died this far short of the gate | IQR | ⇒ position past the previous gate |
|---|---:|---|---:|---|
| **→ gate 5** (KNOWN obstacle) | 22 | p50 **14.3 m** | **2.7 m** | ~6.5 m past gate 4 |
| **→ gate 4** | **30** | p50 **14.1 m** | **1.9 m** | **~6.3 m past gate 3** |
| → gate 1 | 27 | p50 9.8 m | 4.9 m | diffuse |
| → gate 2 | 13 | p50 4.7 m | 3.4 m | diffuse |

The gate-4 cluster is **tighter than the known obstacle's** and its dynamics are indistinguishable
from it:

| group | n | gyro-roll p90 | \|roll\|max | blind frac | fix gap | range rms | speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| gate-5 cluster (KNOWN obstacle) | 20 | 1.28 | 23.8° | 0.23 | 0.26 s | 0.26 m | 8.2 m/s |
| **gate-4 cluster** | **24** | **1.79** | **21.9°** | **0.11** | **0.20 s** | **0.35 m** | **7.1 m/s** |
| gate-blackout (contrast) | 64 | 3.16 | 48.3° | 1.00 | 1.20 s | 1.25 m | 5.6 m/s |

A **calm, level, vision-healthy** drone dying at a tightly clustered distance is the signature of
hitting a fixed invisible object, not of a control departure. **30 deaths (5.2% of the census)
currently filed as "other" are very likely a second obstacle ~6.3 m past gate 3 on the 20.4 m leg to
gate 4.** Both legs are ~20.5 m and both clusters sit ~6.4 m past the previous gate — either two
obstacles at the same relative position, or something systematic that happens 6.4 m after a gate on a
long leg. This is a **hypothesis with a cheap test** (§5), not an established mode; it stays in
"other" in Table 1.

### A note on "launch/release-DIVE"

The bucket is real and tightly clustered, but the word *dive* is only half supported. Across the 115:
the commanded pitch rate goes nose-DOWN in the first 1.2 s on essentially every flight
(min p50 **−1.07 rad/s**, p90 −0.93 — i.e. even the mildest asks for 0.93 rad/s of nose-down), and
16% of early ticks are pinned to +0.000 by the pitch fence. But the **realized** vertical velocity at
t=1.2 s is p50 **+0.42 m/s (climbing)**, and only 11% descend faster than 1 m/s (control: 6%).
**The policy asks for a dive; the fence and the thrust mostly prevent it, and the drone dies anyway.**
Call it the *launch/release-window* mode; the release-pitch command is its most consistent marker,
not the trajectory.

---

## 4. TARGETING TABLE

Shares are **exposure ceilings** — the fraction of deaths in which the named defect is present and
plausibly load-bearing. They are not recovery promises, and they overlap.

| fix | modes addressed | exposure | evidence |
|---|---|---|---|
| **D1 truthful lateral velocity** | lateral diverge + corner + part of at-gate-undetermined | **7.7% … 22.2%** (44…127) | The only velocity channel the policy has is measured wrong: obs[1] vs vision-truth slope −0.110, corr −0.057, 53% sign disagreement. Lateral tail is the worst of any resolved mode (p50 1.84 m, p90 2.88 m). |
| **close-in coast + bbox-range preference** | at-gate undetermined (primary), blackout (secondary) | **up to 33.7%** at-gate + 11.2% blackout | 75% of at-gate deaths end >1.0 m short of the plane; 35% have a ≥0.25 s no-fix gap in the last 2 s; 9% have an unusable terminal range channel (rms > 0.8 m). This is the largest single bucket in the census and coast attacks it directly. |
| **gate-seam (pass_drop teleport) fix** | obstacle leg + the gate-4 "other" cluster + early-leg blackout | **3.8% … 13.9%** | 20% of deaths occur within 1.0 s of a RACE_STATUS advance. Both mid-leg clusters sit ~6.4 m past the previous gate on the two ~20.5 m legs — exactly where the seeker has just dropped the passed gate and must acquire the next at ~20 m. |
| **release-dive retrain** | launch / release window | **20.1%** (115) | Second-largest bucket, present in every lineage (v15 15% → v18 36%), t_death p50 2.1 s / p95 2.3 s. Universal nose-down command at release (min pitch cmd p50 −1.07 rad/s). Highest confidence bucket in the census (93/115 high). |
| **vertical-structure training (v2.0, shipped)** | vertical undershoot + overshoot + corner | **4.4% … 7.2%** (25…41) | **Smaller than expected, and the expectation was an artifact:** the "vertical undershoot dominates" reading comes from `rel_flu[2]` unlevelled, which calls **66.2% of CONFIRMED PASSES** a vertical strike. Levelled, undershoot is 9 and overshoot is 16 — the vertical error is roughly symmetric, not a systematic low bias. |
| **obs fix-gain** | at-gate undetermined + lateral (via estimate noise) | **up to 33.7%**, low confidence | `fix_gain = 1.0` snaps to every fix, so the full 0.43 m (p90 1.49 m) radial noise enters obs[11:14] each tick. A gain < 1 attacks exactly the channel that is 6× noisier than the bearing. But it also adds lag, and the noise is *radial* — it is not obviously what kills. |
| **loop-rate (`vq2_ego_lean`)** | all at-gate modes (margin) | **39% of deaths flew < 30 Hz** | Median loop 35.6 Hz but p10 = 24.1 Hz; the blind terminal window is 0.222 s p50, so 3–5 lost ticks are a real slice of it. A margin-eater, not a cause. |

**Ranking by exposure × confidence:**

1. **release-dive retrain** — 20.1%, highest-confidence bucket, unambiguous, and it is the one mode
   whose classification does not depend on the blind window at all.
2. **close-in coast + bbox-range preference** — the largest exposure (33.7% at-gate undetermined),
   and it is the only fix on the list that attacks the *measurement* blindness rather than working
   around it. It would also shrink every other bracket in this document.
3. **D1 truthful lateral velocity** — 7.7…22.2%, worst tail, mechanism convicted independently.
4. **gate-seam fix** — 3.8…13.9%, and it is cheap; the two clustered mid-leg families are the
   tightest, most repeatable non-launch signatures in the corpus.
5. **vertical-structure training** — 4.4…7.2%. Already shipped; the census says do not expect more.
6. **loop-rate** — margin only.
7. **obs fix-gain** — speculative on this evidence.

---

## 5. WHAT REMAINS AMBIGUOUS

Stated plainly, because a clean-looking table here would be the exact failure this work exists to
escape.

1. **The kill axis is unknown for 69% of at-gate deaths (193 of 278; 33.7% of the whole census).**
   Not "uncertain" — unknown. The median at-gate log ends 1.43 m / 0.222 s before the plane, and 109
   of those deaths read *inside the opening* at the last observable moment.
   **Resolver:** keep the recorder alive ~0.5 s past collision, or log the endpoint's contact point.
   This is a one-line recorder change and it would re-score every past flight in this corpus.

2. **Lateral and vertical are not separable for 16 corner cases + a 67-flight lateral band.** The
   lateral channel's false-positive rate on confirmed passes is 14.8% at 1.0 m and only falls to 2.9%
   at 1.5 m; the vertical channel is 6.3% at 0.75 m. So vertical is measurable and lateral is
   half-measurable. **Resolver:** the same recorder change, plus a ground-truth odometry stream on a
   ~10-flight cohort (which would end the KF-vs-vision dispute for every past and future flight).

3. **`y_h` is a ballistic miss, not an impact point.** For a crabbing or turning drone, body-frame
   lateral offset and the velocity-projected miss legitimately disagree (hand-label case
   `20260719_224943`: −1.5 m body lateral, 0.30 m ballistic miss). Neither is wrong; the log cannot
   say which one the gate frame met. This is the single largest residual weakness of the instrument.

4. **The second obstacle is a hypothesis, not a finding.** 30 deaths cluster at 14.1 m from gate 4
   (IQR 1.9 m) with dynamics indistinguishable from the known gate-5 obstacle. **Resolver:** the
   pilot already owns it — fly ~6 flights with `aim_off` on the gate-4 leg, exactly as on gate 5. If
   the cluster disappears, it is an obstacle; if not, it is a post-gate acquisition failure and the
   gate-seam fix is the right lever.

5. **"other" (93, 16.3%) is honest, not lazy.** It is mid-leg deaths with vision *healthy* — no
   blackout, no gate strike. 30 are the gate-4 cluster; the rest are diffuse (gate-1 IQR 4.9 m,
   gate-2 IQR 3.4 m). Without a world map or an obstacle inventory these cannot be attributed
   further from flight logs alone.

6. **v20 is 6 flights.** Every v20 number in this document is reported for completeness and should
   not carry a decision.

7. **What this profile does NOT establish:** that the modes are causally independent (a late lateral
   correction and a close-in blackout co-occur constantly), or that fixing a mode recovers its share.
   The exposure column is a ceiling.

---

## 6. ARTIFACTS

All under `…/scratchpad/profile/`. Read-only with respect to the repo and both deploy trees.

| file | contents |
|---|---|
| `vg.py` | the VG instrument: loader, true-frame builder, gyro/AHRS attitude, anisotropic Huber fit, LOESS smoother, shape statistics |
| `terminal.py` | v2 terminal geometry (robust axis, range-derived closure, bounded plane carry) |
| `extract.py` → `sessions.pkl` | per-session features, all 616 sessions |
| `events2.py` → `events2.pkl` | 1151 pass + 531 crash gate events (the ground-truth validation set) |
| `classify.py` → `census.pkl` | the calibrated classifier and the census |
| `validate_pass.py`, `validate_frame.py` | the pass ground-truth and frame validations |
| `diagnose_10v41.py` → `diag10v41.pkl` | the A/B/C adjudication + Monte-Carlo false-positive test |
| `report.py` → `census_tables.txt` | the tables above |
| `handlabels.json`, `handlabel_sample.json` | the 24 hand labels with per-case reasoning |
| `trace.py` | per-flight tick trace printer used for hand-labelling |
| `old_instruments/` | independent reproduction of the two legacy instruments |
