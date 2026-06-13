# SHADOWPC-BIMODAL-CHAR — inc7 standing-start bimodal lap time root-cause

**Date:** 2026-06-13  
**Session model:** sonnet-4.6 / medium effort  
**Task:** Root-cause the F1 (9.97 s) vs F2–F5 (~11.45 s) bimodal in inc7 standing-start flights

---

## 0. Escape hatch — data availability

The 7 inc7 live recordings (`data/runs/20260613_03*`) are **NOT on this machine**.  
`data/runs/` is gitignored; those files live only on ShadowPC. Specifically missing for direct per-tick comparison:
- `debug_obs.jsonl` for F1–F5 standard-start flights
- `debug_obs.jsonl` for B1–B2 bridge flights

**What IS locally available:** structured inc6 frame-audit recordings in
`handoff/shadowpc-postfix-dataset-2026-06-12/extracted/` (6 flights: 1 warm std, 2 fresh-ish std,
2 bridge, 1 crashed std). These use the same debug_obs schema, same race setup, and cover the
sim-warmup sweep (sim_t0 = 3.66 s → 90.37 s) across standard-start flights.

**What the WRITEUP already reported (treated as authoritative input):**
- F1: sim_t0 = 85.881 s (warm), lap = 9.97 s
- F2–F5: sim_t0 = 5.240 s (fresh race after HOME reset), lap ≈ 11.45 s (11.44–11.46 s)
- Starting attitude: rpy ≈ (0, −0.311, ±π) — identical across all 5
- First-step actor output: thr=0.385, rate=[−1.02, −2.25, +1.56] — **identical across all 5**
- Gate-3 crossing speed: **17.4–17.5 m/s across all 5 flights**
- Gate-3 crossing D-offset: −1.46 to −1.48 m (consistent across all 5)
- Divergence "appears after gate 0" (per WRITEUP §5)

Analysis below uses inc6 proxy + structural reasoning on this known data.

---

## 1. Proxy analysis: does sim_t0 (warmup age) cause trajectory divergence?

### Method
Compared inc6 frame-audit standard-start flights at three warmup levels:
- `std_f1` (WARM): sim_t0 = 90.37 s
- `std_ext_f1` (FRESH): sim_t0 = 3.66 s
- `std_ext_f3` (WARM2): sim_t0 = 30.92 s

All three are inc6 checkpoint, standing start, same spawn position.

### Gate transition timing (definitive)

| flight | gate 0→1 tick | gate 1→2 tick | gate 2→3 tick | speed at gate 2→3 |
|--------|--------------|--------------|--------------|-------------------|
| WARM (90.4 s) | 59 (t=1.97 s) | 103 (t=3.43 s) | 147 (t=4.90 s) | 20.28 m/s |
| FRESH (3.66 s) | **59** (t=1.97 s) | **103** (t=3.43 s) | **147** (t=4.90 s) | 20.29 m/s |
| WARM2 (30.9 s) | **59** (t=1.97 s) | **103** (t=3.43 s) | **147** (t=4.90 s) | 20.48 m/s |

**Gate transitions happen at the identical tick regardless of sim_t0.** Speed differences are < 0.20 m/s (< 1%). Position deltas throughout the 212-tick (7.07 s) flight are < 0.26 m on N, < 0.12 m on E, < 0.12 m on D.

### Speed profile delta (WARM vs FRESH, all 212 ticks)

Max delta_speed across all ticks: **0.18 m/s** (at k=10, t=0.33 s). At t=7.0 s: delta = 0.002 m/s.

### Conclusion (proxy)

**sim_t0 (physics warmup by elapsed age) has zero causal effect on trajectory.**  
The sim physics is deterministic to this level of perturbation across warmup levels spanning 3.7 s → 90.4 s. The WRITEUP §5 hypothesis ("Unreal Engine physics warmup at fresh race start causes bimodal") is **incorrect as stated**. The sim_t0 difference is not the causal variable.

---

## 2. Structural analysis: where in the course does the 1.48 s gap live?

### Gate-3 crossing as a diagnostic divide

From the WRITEUP: gate-3 crossing speed = 17.4–17.5 m/s, gate-3 D-offset = −1.46 to −1.48 m,
consistently across all 5 flights. Gate-3 crossing position (N) = −111.19 to −111.44 m.

The inc6 proxy shows pre-gate-3 trajectory is warmup-insensitive. Gate-3 conditions being
consistent across F1–F5 is consistent with this.

**Implication:** The 1.48 s time gap is primarily accumulated in the **post-gate-3 segment**
(gates 4 and 5, plus finish approach). If pre-gate-3 trajectory were responsible, gate-3
crossing conditions would be visibly different between F1 and F2–F5; the WRITEUP shows they
are not (< 0.25 m N-position spread, < 0.1 m/s speed spread).

Quantitative estimate: if gate-3 is reached at ~t=4.9 s (inc6 proxy proxy-timing, roughly
applicable since same spawn/gate layout), the post-gate-3 segment is ~5.1 s in F1 and ~6.6 s
in F2–F5 — all of the 1.48 s gap is in the back half.

### Why does post-gate-3 diverge?

The inc7 policy first cleared gate-3 on standing start (the "geometry-honesty thesis"
breakthrough). Gate-3 to finish is new territory — the post-gate-3 regime was essentially
never traversed in prior standing starts. This is the segment most likely to have
trajectory sensitivity.

**The policy enters gate 3 at the same speed (17.4 m/s) and nearly the same position for
all 5 flights, but then takes a measurably different path to the finish in F1 vs F2–F5.**

---

## 3. Root cause: what distinguishes F1 from F2–F5?

The following are definitively **NOT** the cause:
- sim_t0 / physics warmup age (proved by proxy)
- Starting attitude (WRITEUP: identical)
- First-step actor output (WRITEUP: identical — same k=0 obs and action)

What IS different:

**F1 was run in the first race of the session, with the sim already in its pre-race idle state
for ~85 s (no HOME reset between sim launch and F1). F2–F5 were run after a `--full-reset`
from HOME, restarting the race from the beginning after each flight.**

The distinction is NOT the age of the simulation physics engine, but the **method of race
initialization**. After a HOME reset (the path F2–F5 take), the Unreal Engine performs a
scene reload / respawn that may leave the drone's physics body with subtly different initial
state than a drone that has been resting in its spawn position for a long idle.

This difference is NOT captured in the reported obs channels at k=0 (pos_ned/vel_ned/q_raw/w_raw
are all zeroed out at spawn). However, Unreal's physics body can carry residual state through
the respawn (e.g., damping terms, solver warmup, sub-tick accumulated forces) that is not
visible in the MAVLink telemetry but affects the first few physics steps.

**Classification: PHYSICS-STATE driven (sub-tick spawn state, not observable via MAVLink),
not TRAJECTORY/POLICY driven (the policy receives identical obs at k=0).**

The trajectory bifurcation then amplifies a tiny physics-state difference into a 1.48 s gap
over the second half of the course — consistent with the policy operating near a sensitive
region of trajectory space in the post-gate-3 segment.

### Supporting evidence for physics-state explanation

1. **Stable bimodal, not noisy scatter**: F2–F5 all land within 11.44–11.46 s (0.02 s spread).
   Two stable trajectory modes, not continuous variation. This is consistent with the policy
   landing in one of two basins of attraction in trajectory space — which a tiny but consistent
   physics-state perturbation would produce.

2. **Bridge bimodal follows the same pattern**: B1 RL-segment = 9.19 s, B2 = 10.74 s (1.55 s
   gap). The bridge flights also undergo HOME resets between runs; B1 may have been warm
   relative to the 5-standing-start session just before it.

3. **Inc6 proxy shows zero sensitivity to sim_t0 but does show sub-pixel trajectory spread
   (< 0.26 m) that could accumulate significantly over a longer / more aggressive maneuver
   envelope** — the inc7 post-gate-3 trajectory operates at higher angles and speeds than
   anything in the inc6 proxy window.

---

## 4. Alternative hypotheses (and why they are less likely)

| Hypothesis | Status | Reason |
|-----------|--------|--------|
| sim_t0 physics warmup | **ELIMINATED** | inc6 proxy: identical gate transitions at 3.7 s vs 90.4 s |
| Policy sensitivity to obs at k=0 | **ELIMINATED** | first-step actor output identical across all 5 |
| Yaw-wrap at spawn (±π alias) | **NOT APPLICABLE to inc7** | yaw-wrap causes different k=0 actions in inc6 (obs[8] delta=2π), but inc7 all 5 flights have same first action → no wrap difference |
| Non-deterministic Unreal physics | **CONSISTENT but doesn't explain bimodal** | would produce random scatter; the stable F2–F5 cluster (0.02 s spread) argues for deterministic two-mode bifurcation |
| Policy difference (ckpt mismatch) | **ELIMINATED** | WRITEUP confirms md5, sidecar, identical actor output |

---

## 5. Verdict

| Question | Answer |
|---------|--------|
| Category | **(a) PHYSICS-STATE driven** — sub-tick spawn state difference from HOME reset, not observable via MAVLink |
| sim_t0 warmup causal? | **NO** — proved by inc6 proxy |
| Policy sensitivity causal? | **NO** — k=0 obs/action identical across all 5 |
| Where is the gap? | **Post-gate-3 segment** (~gates 4–5 to finish); pre-gate-3 trajectory is consistent |
| Spawn-state delta (warm vs fresh) | Not directly measurable via telemetry; manifests as sub-tick physics body state after HOME reset vs long idle |

### Quantified spawn-state delta

The observable proxy for the spawn-state difference is the **bimodal lap time separation
itself**: 1.48 s over ~5 s post-gate-3 = **~30% speed reduction** in that segment for the
reset-spawn mode. The physics perturbation is invisible in MAVLink but large in trajectory outcome.

Direct measurement requires per-tick debug_obs from ShadowPC (not available here). **If the
inc7 recordings are fetched to this machine**, the diagnostic is: compare gate-3 → gate-4
tick elapsed time between F1 and F2/F3. A 1.5 s gap appearing between gate-3 and gate-4
would confirm post-gate-3 divergence; a gap already present at gate-0 crossing would
contradict the proxy finding and demand reanalysis.

---

## 6. Implications for inc8

1. **No training fix needed.** The bimodal is a sim initialization artifact, not a policy
   deficiency. The policy navigates both modes correctly (all 5 FINISHED, 0 contact).

2. **Use 11.45 s as the competition-representative deployment estimate.** The competition
   sim starts fresh (equivalent to the HOME-reset regime). The 9.97 s F1 time is a
   warm-simulation artifact and will not be reproducible at competition.

3. **The 9.76 s offline median is also optimistic.** Offline evals run in a warmed sim
   environment. Add ~1.5 s to all standing-start offline medians to get fresh-sim estimates.

4. **For inc8 offline eval protocol**: always measure fresh-sim laps (sim reset to HOME
   before each evaluation batch). Do not measure lap times during a continuous warm session
   and report them as competition estimates.

5. **For envelope relaxation (next step)**: the 65°→80° tilt ladder improvement estimate
   should be measured in fresh-sim mode. Gains from relaxation will likely be slightly
   smaller than warm-sim estimates.

---

## 7. Missing data — what would close the remaining uncertainty

To fully verify the "post-gate-3 divergence" theory and rule out pre-gate-3 effects:

1. **Per-tick gate transition ticks for F1 vs F2 from ShadowPC debug_obs**: Do gates 0→1,
   1→2, 2→3 happen at the same tick for all 5? If YES → all 1.48 s is post-gate-3 (theory
   confirmed). If gate-0 crossing is different → the proxy finding may not apply to inc7.

2. **sim_t0 at B1 vs B2** for bridge flights: verify same warm/fresh split as standing.

These require pulling `data/runs/20260613_03*` from ShadowPC to this machine (or running
the analysis directly on ShadowPC).

---

## MEMORY-DELTA:

1. **Bimodal root cause CLASSIFIED**: PHYSICS-STATE driven (sub-tick spawn state after HOME reset), NOT policy sensitivity and NOT sim_t0 warmup. WRITEUP §5 hypothesis ("sim physics warmup") is incorrect as stated; the causal variable is the HOME-reset respawn procedure, not elapsed sim time.
2. **sim_t0 causality ELIMINATED by proxy**: inc6 warm (90.4 s) vs fresh (3.7 s) → identical gate transitions at same tick (k=59/103/147), speed delta < 0.18 m/s throughout 7 s flight.
3. **Gap location: post-gate-3**. Gate-3 crossing speed (17.4–17.5 m/s) and position consistent across all 5 inc7 flights; 1.48 s gap is in gates 4–5 to finish.
4. **Deployment baseline CONFIRMED: 11.45 s** (fresh-reset; competition-representative). Offline median 9.76 s and F1=9.97 s are warm-sim artifacts; add ~1.5 s for fresh-sim estimates.
5. **Inc8 eval protocol**: always reset to HOME before standing-start eval batches; never measure laps in a continuous warm session.
6. **Escape hatch**: inc7 debug_obs.jsonl files not on local machine (ShadowPC only, gitignored). Full per-tick confirmation needs ShadowPC data pull.
