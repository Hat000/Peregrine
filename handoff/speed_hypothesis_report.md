# Does forward SPEED drive gate survival? — adjudication

**Worker report, 2026-07-27. Corpus: 680 flights → 1910 per-gate approaches (658 flights with usable
logs), 1316 survived / 594 died, 92 (checkpoint × commanded-rate × gate) strata.**

---

## VERDICT

> **NO. "The drone flies too fast to see" is REFUTED as a mechanism, and no `ego_speed_gov`
> setting is supported by this corpus. Do not fly one as a survival fix.**
>
> The mechanism is broken at its first link: **forward speed does not move vision at all.**
> `corr(approach speed, last-sighted range) = +0.006` (n = 1818). Across speed quintiles spanning
> **2.56 → 16.52 m/s**, the median last-sighted range is flat at **1.86 – 2.08 m**. The blind onset
> is FOV geometry, exactly as [[vision-horizon-fov-2026-07-27]] says — it is **range-only, and
> speed-invariant**. There is no a-path, so there is nothing to mediate.
>
> **Worse, slowing down makes the blind segment strictly worse.** The blind DISTANCE is fixed at
> ~2.0 m by geometry, so slowing only stretches the *time* spent dead-reckoning through it
> (corr(speed, blind time) = **−0.36 to −0.51**), and blind TIME is an independently significant
> killer: **b = −2.31, z = −2.67, p = 0.0075** even after controlling for blind distance. Going
> 8.0 → 6.0 m/s adds **+34% open-loop time** (0.249 s → 0.333 s) through a blind distance that does
> not shrink.
>
> **The one lever that is real is VISION FRESHNESS, not speed.** In the same stratified model:
> fresh **b = +2.84 to +3.89, z = +5.2 to +8.5, p < 1e-5**, while speed is null
> (**z = −1.63 to +0.52**). Per +0.10 of fresh-fix fraction, **OR ≈ 1.33 – 1.48** — which is the
> whole size of the rate-30 win.

**One qualified caveat, stated up front:** there IS a real raw association between a *high reading
on `hypot(obs[0],obs[1])`* and death (OR 0.53 above 8.5 m/s). It is a **second-order interaction
with estimator degradation**, not a speed effect, and it fails to reproduce on 2 of 3 independent
measurements of the same physical quantity. Details in §3. It is not a basis for a governor.

---

## Instrument, conventions, and one bug I found in my own extractor

**Frames — stated once, applied throughout.**
* `obs[0:3]` is body-FLU velocity **virtual-flipped** by `diag(-1,-1,1)` ⇒ true `v_fwd = -obs[0]`,
  true `v_left = -obs[1]`, true `v_up = +obs[2]`.
* **Primary speed variable = `hypot(obs[0], obs[1])`** — a magnitude, therefore flip-agnostic, and
  *exactly* the quantity `ego_speed_gov` keys off (`rl/fly_rl.py:791`). Chosen deliberately so the
  analysis variable is the intervention variable.
* `obs[11:14]` slot-0 lever is virtual-flipped with the aim offset and z-bias **baked in**;
  true FLU lever = `diag(-1,-1,1) @ obs[11:14]`, then de-injected
  `rel_true = rel_logged + [0, +lat, −vert]` (`src/racer/ego_obs.py:279-281`). Range = its norm.

**Guards applied.** MIS-LOCK seam killed as a **distance** gate (consecutive-fix lever jump > 1.5 m,
never a speed gate); ADVANCE seam killed by construction (±2 ticks either side of every
`gate_index` change); the aim-offset **release tick excluded**; pre-acquisition masked-lever ticks
dropped. Unit of analysis is the **per-gate approach**, so exposure is conditioned on by
construction. `STALLED` / `SIM_RESET` terminal approaches are censored out (not gate outcomes).

**Speed is measured in a fixed RANGE window, never whole-flight.** Primary window `[5, 8) m`
("apr") — ends **0.76 s** before the gate plane at median speed. Supporting windows
`band [12,18)`, `far [8,12)`, `mid [3,5)`, `near [1.19,3)`.

**Instrument check (passed before anything was believed):**
* rate 30 n=190 mean max gate **2.447**, rate 40 n=490 **1.737** — reproduces memory's 2.460/1.769.
* `last_sighted_m` **minimum 1.20 m, zero approaches inside 1.19 m**, median 1.89 m on confirmed
  passes — reproduces the FOV blind-zone floor and the 1.78 m median.

**Bug found and fixed in my own extractor:** `int(d.get("gate_index", -1) or -1)` maps
`gate_index == 0` to `-1` (falsy zero), silently deleting **every gate-0 approach** — 658 of 1910,
34% of the corpus. Caught because the survival-by-gate table started at gate 1. This matters
beyond bookkeeping: gate 0 turned out to carry the launch confound that drove the first (wrong)
answer. **This is instrument error #11 of the genre; the corpus never lied.**

---

## (a) Does speed predict per-gate survival, properly controlled?

**The pooled answer is a Simpson's-paradox artifact — it inverts.**

| model | D(survivor − died), apr window | verdict |
|---|---|---|
| **Pooled** (all 1818 approaches) | **−0.218 m/s**, t=−2.70, p=0.007 | survivors look SLOWER |
| **Inverse-variance pooled within (ckpt×rate×gate)** | **+0.173 m/s**, z=+3.46, p=0.0005 | survivors look FASTER — sign flip |
| **Logistic, speed + stratum fixed effects** | β = −0.105, **z=−1.63, p=0.102** | **null** |

The within-stratum "faster is safer" result is itself an artifact: **41.6% of its pooled weight
comes from a single stratum, (v19Ws0, rate 30, gate 0)** — the launch/takeoff gate, where "slow"
means "never got going", the textbook reverse-causality trap. Excluding gate 0 it collapses:
`+0.173 (z=+3.46) → +0.108 (z=+1.42, p=0.155)`; at the mid window `+0.379 → +0.080 (p=0.316)`.

**The linear model was the wrong functional form.** Adding a quadratic term makes the shape
appear: apr `speed² = −0.092, z=−3.86, p=0.0001` (and −0.112, z=−3.29 excluding gate 0). The
relationship is an **inverted U**, which a linear coefficient averages to zero.

**Effect size and n, the only form worth quoting** — threshold, stratum FE, gate ≥ 1:

| trigger | n above | β | z | p | OR |
|---|---|---|---|---|---|
| `sp_apr > 7.0` | 675 | +0.310 | +1.60 | 0.109 | 1.36 |
| `sp_apr > 7.5` | 455 | −0.180 | −0.94 | 0.345 | 0.84 |
| **`sp_apr > 8.0`** | 290 | **−0.522** | **−2.67** | **0.008** | **0.59** |
| **`sp_apr > 8.5`** | 182 | **−0.576** | **−2.60** | **0.009** | **0.56** |
| `sp_apr > 9.0` | 107 | −0.580 | −2.17 | 0.030 | 0.56 |

Cluster bootstrap over **flights** (400 reps, the correct unit — approaches within a flight are not
independent): β median −0.654, **95% CI [−1.129, −0.193]**, 1.0% of mass ≥ 0. So the *association*
on the obs channel is statistically solid. Whether it is **speed** is §3.

It survives conditioning on approach alignment (`|lat|`, `|vert|` in-window): −0.638 → −0.672
(z=−3.06). It is not misalignment.

## (b) Is there an optimum, or is it monotone down?

**There is an interior optimum at ≈ 7.0 – 8.0 m/s, and the fleet already sits at or just below it.**
Dose-response, gate ≥ 1, apr window (gate 0 excluded to remove the launch confound):

| `sp_apr` band | n | p(survive) |
|---|---|---|
| [0, 5.5) | 41 | 0.512 ± 0.078 |
| [5.5, 6.0) | 73 | 0.658 ± 0.056 |
| [6.0, 6.5) | 130 | 0.731 ± 0.039 |
| [6.5, 7.0) | 208 | 0.764 ± 0.029 |
| **[7.0, 7.5)** | **161** | **0.839 ± 0.029** ← peak |
| [7.5, 8.0) | 134 | 0.806 ± 0.034 |
| [8.0, 8.5) | 96 | 0.677 ± 0.048 |
| [8.5, 9.0) | 69 | 0.638 ± 0.058 |
| [9.0, ∞) | 81 | 0.593 ± 0.055 |

**It is emphatically NOT monotone down.** The corpus median `sp_apr` is **6.58 m/s** — already
*below* the empirical peak. Both tails are worse than the middle.

**This is the concrete danger in the knob's own help text.** `ego_speed_gov` help suggests
"try 5,6.5". At SOFT = 5.0 the ramp engages on **99.6%** of approaches; HARD = 6.5 sits in the
**0.731** bin while the peak is **0.839**. Setting 5,6.5 would drag the entire fleet from its
optimum down into a demonstrably worse band. **If any governor is ever flown, 5,6.5 is the wrong
number and would be actively harmful on this corpus.**

## (c) Does speed act THROUGH vision freshness / last-sighted range?

**No. There is no a-path to mediate — this is the core refutation.**

| a-path | correlation | n |
|---|---|---|
| corr(speed [5,8) m, **last-sighted range**) | **+0.0061** | 1818 |
| corr(speed [3,5) m, last-sighted range) | −0.0110 | 1807 |
| corr(speed [1.19,3) m, last-sighted range) | +0.0163 | 1740 |
| corr(speed [5,8) m, **fresh-fix fraction**) | **+0.0043** | 1818 |

Last-sighted range by speed quintile — quintile 1 mean speed 2.56–5.48 m/s, quintile 5
7.78–16.52 m/s:

| Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---|---|---|
| 1.946 m | 1.984 m | 2.083 m | 1.966 m | 1.857 m |

Flat across a **6×** speed range. Speed does not buy a single centimetre of extra sight. This is
precisely what the FOV solution `|el| + atan(0.75/R) = VFOV/2` predicts: the cutoff is a function
of **range only**.

Because there is no a-path, the b-path is moot, and indeed adding the mediators barely moves the
speed coefficient (−0.638 → −0.606 with fresh; → −0.527 with last-sighted). **Speed and freshness
are independent channels, and freshness is the big one:**

| window | fresh β (per 1.0 fraction) | z | p | speed β (per 1 m/s) | z | p |
|---|---|---|---|---|---|---|
| far  | **+3.891** | +5.46 | <1e-5 | +0.041 | +0.52 | 0.607 |
| apr  | **+2.842** | +5.18 | <1e-5 | −0.105 | −1.63 | 0.102 |
| mid  | **+3.510** | +8.46 | <1e-5 | +0.009 | +0.14 | 0.887 |

**The rate-30 win is a freshness win, and it is NOT a speed win — the sign is wrong.** Within the
only checkpoint with both rates at usable n (v19Ws0, n30=593 / n40=25), rate 30 is **FASTER**
(+0.571 m/s, t=+2.50, p=0.019), survives better (0.742 vs 0.640), and is fresher (0.990 vs 0.896).
Corpus-wide, rate 30 is faster in **every** window (far +0.662, apr +0.453, mid +0.360, near
+0.249; all p ≤ 2.4e-4). If "slower = safer" were the mechanism behind signal 1, rate 30 would have
to be slower. It is not. Freshness moves the same way in both checkpoints that have both rates;
speed moves in **opposite** directions (v19Ws0 +0.571, vpeffs0 −0.809). **Freshness is the
consistent channel; speed is not.**

### The blind segment: slowing is a net negative

| quantity | finding |
|---|---|
| blind DISTANCE (last-sighted range) | **FOV-fixed, median 2.00 m**, speed-invariant |
| blind TIME (= distance / speed) | median 0.284 s; **corr with speed −0.36 to −0.51** |
| blind DISTANCE → survival | β = −0.867, **z = −8.68** |
| blind TIME → survival, controlling for distance | β = **−2.31**, **z = −2.67, p = 0.0075** |

Open-loop transit at the median 2.00 m blind distance: 8.5 m/s → 0.235 s; 8.0 → 0.249 s;
7.0 → 0.285 s; 6.5 → 0.307 s; **6.0 → 0.333 s (+34% vs 8.0)**. Against `ego_stale_horizon = 0.5 s`
and memory's ~0.18 s design point, a governor spends its entire budget buying *more* dead-reckoning
through a window that does not get shorter. **Slowing down is the wrong sign on the one thing it
actually changes about vision.**

## (d) Concrete recommendation

> ### The data does not support an `ego_speed_gov` value. Do not arm it.

Grounds, in order of weight:

1. **The mechanism is refuted** (§c): zero a-path to vision, and the only vision quantity speed
   *does* move — blind time — it moves in the harmful direction.
2. **The surviving association is not established to be speed** (§3): 2 of 3 independent
   measurements of approach speed fail to reproduce it, one reverses it.
3. **The fleet is already at or below the empirical optimum** (6.58 m/s median vs a 7.0–8.0 m/s
   peak). A governor is a one-sided cap: it can only push down, i.e. **away** from the peak.
4. **The upside is capped by exposure.** A HARD = 8.5 governor can touch the `[5,8) m` mean of only
   **15.4%** of gate ≥ 1 approaches; HARD = 8.0 only 24.7%. Even if the OR 0.56 were fully causal,
   the reachable gain is ~2–4 points of per-gate survival, against a per-gate target of
   **p ≥ 0.9659** for a 50% lap ([[campaign-20-gates-2026-07-27]]) versus today's 0.8075. **A
   governor cannot close that gap even in its best case.**
5. **The knob's own suggested setting is harmful** (§b): "5,6.5" engages on 99.6% of approaches and
   lands the fleet in the 0.731 band instead of the 0.839 peak.

**Provenance for the record: `ego_speed_gov` has NEVER been flown.** All 655 corpus `meta.json`
that carry the field record `""`; the other 25 predate it. Every `tools/pilot_panel_logs/*.log`
line reads `speed_gov=off`. Any claim about its effect is at present a pure extrapolation.

**If it is flown anyway** (pilot's call — the mechanism is refuted, not the knob), the only
defensible setting is a **pure tail-clip that cannot touch the fleet's operating point**:
`ego_speed_gov 8.5,9.5`. That leaves the 7.0–8.0 peak completely untouched, clips only the
declining tail, and uses the ramp (not the abrupt single-value hard cap) so it cannot step-change
thrust mid-approach. **It is a bounded-risk experiment, not a recommendation** — I estimate its
true expected effect is ≈ 0.

**Where the effort should go instead: vision freshness.** It is 10–30× the effect size of speed in
the identical model, it is robust to every cut I applied, and it is the channel that actually
explains the rate-30 win. Memory's standing item — *commanding SLOWER STILL (25, 20 Hz) may be
better again* — is the right experiment, and this analysis strengthens it: the payoff is freshness,
and it is being pursued for the right reason. Note it is **not** a speed intervention; the loop
rate does not govern the airspeed.

## (e) The dodge: "works by slowing" vs "works by dodging"

**Existing data CANNOT separate them. The proposed arm is required.**

Dodge armed read **from the log** (`aim_off` non-null AND `|vertical| > 0.5`), never the config
string: **39 approaches** (gate 5: 24, gate 4: 10, gate 1: 5), 38 of 39 on v19Ws0, 37 of 39 at
rate 30. Matched pool = gates 4–5, same checkpoint and rate: n = 141.

| gate | dodge | no-dodge | Fisher p |
|---|---|---|---|
| **4** | **8/10 (0.800)** | 37/88 (0.420) | **0.0404** |
| 5 | 9/24 (0.375) | 5/19 (0.263) | 0.523 |

**The confound is real and large.** At gate 4 the dodge flights are slower in every window:
band −0.469 (t=−1.91), **far −1.175 (t=−4.33)**, **apr −0.938 (t=−3.20)**, mid −0.689 (t=−2.29).
The gate-4 dodge and a ~1 m/s slowdown are the same flights.

**Two readings, and the corpus cannot choose between them:**
* *For "it works by slowing":* the pattern across the two gates lines up — gate 4 slows (−0.94 apr)
  and works (p=0.040); gate 5 does **not** slow (+0.666 apr, +1.315 mid — if anything faster) and
  does **not** work (p=0.523). That is a 2-point correlation. n = 2. It is a coincidence-grade
  observation, not evidence.
* *Against:* the dodge coefficient does not shrink when speed is controlled — bare β = +1.164,
  +sp_apr **+1.187**, +sp_band +1.134, +both +1.355. But **the test has no power**: n = 99 across
  4 strata, and the dodge coefficient is not significant even bare (z = +1.38, p = 0.168). A
  coefficient that never reaches significance cannot be shown to survive or to collapse.
* Separately, the dodge **does** move the drone: mean true `v_up` in the far window is +3.323 vs
  +2.802 at gate 4 (t=+3.56, p=0.002) and +2.622 vs +1.179 at gate 5 (t=+3.98, p<0.001). So it is
  not a pure speed intervention. (This is the *lever/velocity* channel, a different quantity from
  the sister analysis's absolute in-band altitude Δ = −0.028 m — the two are not in conflict, they
  measure different things.)

### The arm that would settle it

The standing proposal — **`ego_speed_gov` armed with NO aim offset** — is the right design and is
a **true single-variable test**: `_V1_RECIPE` pins `ego_aim_offsets` to `""`, so a model-pick
clears it by construction (**pick the model FIRST, then type the knob**). Specification:

* **Arm A (control):** current deploy config, no aim offsets, no governor.
* **Arm B (test):** `ego_speed_gov 7.4,8.4`, no aim offsets. Chosen to reproduce the gate-4 dodge's
  *measured* in-band slowdown (−0.94 m/s from a no-dodge mean of 8.78 m/s at apr) **without any
  lateral or vertical aim change**. Do not use 5,6.5 — see §b.
* **Read-out:** gate-4 and gate-5 survival, plus `gov` / `gov_engaged` in `ego_obs.jsonl` to
  confirm the governor actually engaged in the band (a knob that never fires is not a null result).
* **Everything else pinned:** v19Ws0, rate 30, `ego_pitch_clamp` fixed, `ego_assist_thrust`
  recorded (memory's provenance hole — a drifted assist silently confounds the cohort).
* **Decision rule:** if Arm B reproduces a meaningful part of the gate-4 dodge benefit, the dodge
  works by slowing. If it does not, the dodge works by dodging.

**Power — read this before flying it.** Baseline per-gate survival (gate ≥ 1) = 0.654.
To detect **+0.10** absolute at 80% power / α=0.05 needs **≈326 approaches per arm ≈ 130 flights
per arm**. To detect **+0.15**: ≈137 approaches ≈ **55 flights per arm**. The gate-4 dodge's
observed effect (0.800 vs 0.420) is +0.38 and would need only ~15 flights per arm — **but that
0.38 is an n=10 point estimate and is certainly inflated.** Budget for **≥55 flights per arm** and
treat anything smaller as a pilot probe that cannot conclude. The existing 39 dodge approaches are
roughly **1/8** of the n needed for the modest effect sizes that are actually plausible.

---

## 3. SUSPECT THE INSTRUMENT — why I do not believe the obs-speed association is speed

I named the artifact that could produce this exact sign and magnitude — *`hypot(obs[0],obs[1])` is
the KF's velocity estimate, not the drone's speed* — and then went and measured the same physical
quantity three other ways. **The effect does not travel.**

| instrument | source (independent of `obs[0:3]`?) | high-tail effect on survival |
|---|---|---|
| `hypot(obs[0],obs[1])` in [5,8) m | **no** — KF velocity, the governor's own key | **OR 0.53, z=−2.91, p=0.004** |
|  …after dropping 21/1910 approaches with physically absurd `spmax` (>11 m/s) | same | OR 0.77, z=−1.10, **p=0.271** |
|  …restricted to `fresh ≥ 0.95` | same | OR 0.88, z=−0.48, **p=0.632** |
| vision range-closing rate, [5,8) m | **yes** — detector/PnP lever derivative | OR 0.99, z=−0.06, p=0.952 |
| vision range-closing rate, [3,5) m | **yes** | OR 1.33, z=+1.64, p=0.101 (**wrong sign**) |
| sim-clock shell transit 8→5 m | mostly — clock, lever only for bounds | OR 1.47, z=+2.04, p=0.041 (**wrong sign**) |
| sim-clock shell transit 12→8 m | mostly | OR 1.59, z=+2.37, p=0.018 (**wrong sign**) |
| **previous-leg traversal time**, top 30% fastest | **fully — sim clock + advance events only** | OR 0.69, z=−2.29, p=0.022 |
| previous-leg traversal time, linear | fully | n.s., z=−0.81, p=0.420 |

Reading:
1. **Fragility.** Removing **21 approaches out of 1910 (1.1%)** whose window max exceeds 11 m/s —
   the corpus max is **54.25 m/s**, which no VQ2 airframe does — cuts β by 60% and takes p from
   0.004 to 0.271. An effect carried by 1.1% of the rows is a tail-of-the-instrument effect.
   *(Leave-one-stratum-out is by contrast stable: z ∈ [−3.92, −2.20]. The fragility is to
   individual outlier approaches, not to any one stratum.)*
2. **It does not reproduce, and twice it reverses.** The two lever-derived and clock-derived
   measures say fast approaches survive *better*. Only the fully estimator-free previous-leg-time
   agrees in sign — and only at 1 of 4 thresholds tested, with the linear form null. That is
   threshold-shopped, not a result.
3. **The four "speed" measures correlate at only +0.09 to +0.35 with each other.** Four
   measurements of one physical quantity that barely co-vary means at least three are
   noise-dominated. **I do not have a trustworthy per-approach speed measurement in this corpus**,
   and I am not going to build a flight recommendation on the one channel that happens to be the
   knob's trigger.
4. **What the association actually is: an interaction with estimator degradation.**

   | | fresh ≥ 0.95 | fresh < 0.95 |
   |---|---|---|
   | `sp_apr ≤ 8.5` | 0.766 (n=862) | 0.703 (n=774) |
   | `sp_apr > 8.5` | 0.664 (n=122) | **0.433 (n=60)** |

   Speed costs −0.102 when vision is healthy and **−0.270 when it is degraded**. Speed and
   staleness compound; speed alone does little. Note this is *not* "stale estimator reads hot" —
   high obs speed is associated with slightly **better** freshness (0.942 in the >9.5 m/s bin vs
   0.921 below 7). I checked and discarded that hypothesis.

**Honest limit on the refutation.** The `fresh ≥ 0.95` null is power-limited: n falls 1636 → 984,
and its 95% CI still admits effects up to about the original size. It **attenuates** the estimate
by ~80%; it does not prove absence. Likewise the previous-leg-time hint (OR 0.69) is not nothing.
The correct summary is not "speed provably does nothing" but: **speed is at best a weak,
non-robust, second-order effect that appears only in combination with degraded vision, and the
proposed causal pathway through vision is affirmatively refuted.** That is more than enough to say
do not fly a governor, and nowhere near enough to pick a value for one.

## What would change this verdict

* A flight arm (Arm A/B above) is the only clean test — the governor has never been flown, so
  every number here is observational.
* A trustworthy independent speed channel. If GT velocity is ever available in a deploy log, redo
  §3 in one pass; all four of my witnesses are compromised in different ways.
* If `ego_obs_coast` training lands and lifts freshness at the wire, re-run the interaction table —
  the whole speed effect lives in the degraded-vision cell and may simply disappear.

---

## Reproduction

Scripts (scratchpad, not committed — read-only against the corpus):
`work/extract.py` (per-approach extraction, all guards), `work/st.py` (numpy-only stats: Welch,
Fisher exact, IRLS logistic, AUC), `work/fe.py` (stratum-FE helper), `work/a1_core.py` …
`work/a8_blind.py`. Corpus untouched.

---

## MEMORY-DELTA

- 🛑🛑 **"THE DRONE FLIES TOO FAST TO SEE" IS REFUTED.** `corr(speed, last-sighted range) = +0.006` (n=1818); across quintiles 2.56→16.52 m/s the last fix is flat at 1.86–2.08 m. **The blind onset is FOV geometry — range-only, speed-INVARIANT.** No a-path ⇒ nothing to mediate.
- 🛑 **SLOWING MAKES THE BLIND SEGMENT WORSE.** Blind DISTANCE is FOV-fixed (med 2.00 m); slowing only stretches blind TIME (corr −0.36…−0.51), and blind time kills independently (**b=−2.31, z=−2.67**). 8.0→6.0 m/s = **+34% open-loop** (0.249→0.333 s).
- 🚩 **`ego_speed_gov` HAS NEVER BEEN FLOWN** (655 metas `""`, every panel log `speed_gov=off`). **DO NOT ARM IT** — mechanism refuted, fleet median 6.58 m/s already BELOW the 7.0–8.0 m/s survival peak, and a governor is a one-sided cap. 🛑 **The knob help's suggested "5,6.5" would ENGAGE ON 99.6% of approaches and drag the fleet from p=0.839 to p=0.731 — actively harmful.** Bounded-risk tail-clip only if flown anyway: **8.5,9.5**.
- 🟢🟢 **FRESHNESS IS THE LEVER, SPEED IS NOT** — same stratified model, n=1910 approaches / 92 strata: fresh **b=+2.84…+3.89, z=+5.2…+8.5**; speed **z=−1.63…+0.52 (null)**. Per +0.10 fresh, **OR 1.33–1.48** = the whole rate-30 win. 🛑 **Rate 30 is FASTER not slower** (v19Ws0 within-ckpt +0.571 m/s, p=0.019, AND survives better) ⇒ the rate-30 win cannot be a speed effect.
- 🚩 **PER-GATE SURVIVAL vs SPEED IS AN INVERTED U, peak 7.0–8.0 m/s** (p=0.839 at [7.0,7.5) vs 0.731 at [6.0,6.5) and 0.593 above 9.0). **A LINEAR MODEL AVERAGES IT TO ZERO** — that is why the FE logistic read null (z=−1.63) while `speed²` is z=−3.86.
- 🛑 **THE obs-SPEED TAIL EFFECT DOES NOT TRAVEL.** `I(hypot(obs[0],obs[1])>8.5)` → OR 0.53 (z=−2.91), but **2 of 3 independent speed witnesses REVERSE it** (vision closing rate OR 1.33; sim-clock shell transit OR 1.47–1.59, p≈0.02–0.04). Only estimator-free prev-leg-time agrees (OR 0.69) and only at 1 of 4 thresholds. **The four speed measures inter-correlate at just +0.09…+0.35 — there is NO trustworthy per-approach speed channel in this corpus.**
- 🚩 **WHAT THE obs-SPEED ASSOCIATION REALLY IS: an INTERACTION WITH DEGRADED VISION.** speed>8.5 costs −0.102 when fresh≥0.95 but **−0.270 when fresh<0.95**. Also **1.1% fragile**: dropping 21/1910 approaches with `spmax`>11 m/s (corpus max **54.25 m/s** — impossible) takes p 0.004→0.271.
- 🛑 **GATE 0 IS A LAUNCH CONFOUND — EXCLUDE IT FROM EVERY SPEED/SURVIVAL VERDICT.** It alone carried **41.6%** of the inverse-variance pooled weight and flipped the within-stratum sign; excluding it, +0.173 (z=3.46) → +0.108 (p=0.155). "Slow at gate 0" = "never got going".
- 🛑🛑 **INSTRUMENT ERROR #11 (mine): `int(d.get("gate_index",-1) or -1)` maps gate 0 → −1 (FALSY ZERO) and silently deleted 658/1910 approaches (34%).** Caught only because the by-gate table started at gate 1. **NEVER `or`-default a numeric log field.**
- 🚩 **THE DODGE: existing data CANNOT separate "works by slowing" from "works by dodging."** Gate 4 dodge 8/10 vs 37/88 (Fisher **p=0.040**) AND slower in every window (far −1.175 t=−4.33, apr −0.938 t=−3.20); gate 5 neither slows nor works. Controlling for speed does not shrink the dodge β (+1.164→+1.187) **but the test has NO POWER** (n=99, β n.s. even bare). Dodge DOES move `v_up` (far +0.521 t=+3.56 g4; +1.443 t=+3.98 g5). **ARM = `ego_speed_gov 7.4,8.4` with NO aim offset** (reproduces the measured −0.94 m/s, not 5,6.5); **≥55 flights/arm** for +0.15, **~130 flights/arm** for +0.10 (baseline p=0.654).
