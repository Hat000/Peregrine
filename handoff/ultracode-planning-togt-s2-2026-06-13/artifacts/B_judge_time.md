# Phase B — JUDGE PANEL, lens = ACHIEVABLE LAP TIME (corrected-aero, tracking-realizable)

**Author:** Phase-B judge (opus-4.8), ultracode S2-planning fan-out. **Date:** 2026-06-13. **For:** Fengyou.
**Scope:** analysis only; no source edits, no live sim, no SLURM, no network. All numbers re-derived
natively (`.venv`, `PYTHONPATH=src`) from repo data or read from frozen TOGT CSVs / the Phase-A/B artifacts
in this directory.

**The lens (single axis):** *what lap time does this approach actually deliver as a flyable line on the
REAL corrected-aero plant, including tracking realizability?* I am adversarial about point-mass optimism and
I penalize any approach whose quoted time assumes a tracker that does not exist or has never been measured.

---

## 0. The one fact that collapses the comparison

I re-ran `minsnap_topp_proto.py` natively. The **IDEAL point-mass lap by tilt cap reproduces exactly**:

| tilt cap | minsnap IDEAL | togt_cpc honest TOPP | hybrid TOPP | mpcc coupled TOPP |
|---|---|---|---|---|
| 60° | 9.72 s | 9.64 s | 5.05 s* | 5.39 s* |
| 65° | 8.81 s | 8.67 s | 4.93 s* | — |
| 75° | 6.79 s | 6.57 s | 4.78 s* | 4.76 s* |
| 80° | 5.55 s | 5.33 s | 4.73 s* | — |

\* hybrid/mpcc quote a *thrust-magnitude-capped* (not tilt-cone) budget, so their "point-mass" floor is the
drag-wall ~4.7 s regardless of cone — they relax the a_lat=g·tan(θ) cone into a_h=78.3·sinθ. minsnap and
togt_cpc keep the honest g·tan(θ) cone, which is the binding physical constraint on a near-level descending
line. **The two families disagree by ~4.5 s at the 60° cone purely on how they model the tilt constraint.**

This disagreement is the crux of the TIME lens. Which is right?

**The g·tan(θ) cone (minsnap/togt_cpc) is the honest one for a course that must hold altitude / descend
gently.** To corner OR push against drag while keeping the thrust vector's vertical component ≥ g, the usable
horizontal accel is bounded by g·tan(θ), not 78.3·sinθ. The 78.3·sinθ budget is only available if you let the
vertical component drop below g — i.e. you trade altitude. The course descends 26 m total (mean slope 10.7°),
which gives a *small* vertical-thrust credit (~1.7%, per togt_cpc §4), nowhere near enough to license the
full 78.3·sinθ at 60°. **So the hybrid/mpcc 5.4 s "60°-cone" floor silently spends altitude it doesn't have;
the honest 60°-cone floor is ~9.6 s.** This matters enormously: it means *at the doctrine-allowed 60° cone,
EVERY planning approach lands at ~9.6 s — identical to inc7's already-live 9.76 s.*

**Corollary (the headline of this whole panel): on corrected aero the lap time is a TILT story, not a planner
story.** No trajectory optimizer manufactures the envelope relaxation. The relaxation is a reward/doctrine
decision (the rw_tilt ladder), independently gated on LAPTOP-INC8-BINDING-GATE-VERIFY. Every approach's
"fast" number (5–6 s) is conditioned on opening the cone to ~75–80°, and the SAME relaxation would let the
already-live monolithic RL go fast too (inc5 unconstrained already hit 6.9 s with NO planner).

---

## 1. The realizability filter (the adversarial core)

A point-mass lap is not a lap time. The lens explicitly demands I penalize unrealized-tracker optimism. The
evidence on tracker realizability is brutal and symmetric across the field:

- **minsnap_topp's own bundled geometric tracker FAILS.** I ran it: `tracked lap = None`, all six gate misses
  = **inf**, at BOTH the 60° and 75° plans. The author's note ("my closed-loop tracker had a sign bug I
  declined to chase") is not a footnote — it means the 7.0 s estimate has **zero native realizability
  evidence**; the 15–30% tracking pad is borrowed from the k=1.85 geometric datum, not measured.
- **hybrid's DF+PD tracker FAILS:** gate misses 5–35 m, divergence to 142 m at high attitude gain
  (`hybrid_tracker.py`, per B_approach_hybrid_rl.md §3).
- **mpcc's toy SLSQP/PD tracker reached only ~18–21 m/s terminal** vs the 38 m/s wall — "naive tracking
  leaves half the envelope on the floor" (B_approach_mpcc.md §4.5).
- **The ONLY measured tracker datum on this plant is k=1.85** (geometric tracker on the shipped 4.55 s line →
  8.3 s; 54-combo gain sweep found nothing faster — a STRUCTURAL gap, not a tuning gap).

So three of three native tracker prototypes could not fly *any* of these lines to a valid finish. Every
"fast" number in this panel rests on the assumption that an UNBUILT MPCC or RL-tracker recovers 85–90% of the
point-mass speed — an assumption with **no supporting measurement and one contradicting datum (k=1.85)**.

**The only approach family with a MEASURED realizable lap on the corrected plant is the monolithic RL**
(inc5 unconstrained 6.9 s twin; inc7 constrained 9.76 s offline / 11.45 s live). It is the empirical anchor,
and it is the comparator I score all four against, because it is what they must beat.

---

## 2. Per-approach time scoring (0–10; higher = better achievable-time story)

### minsnap_topp — claimed 7.0 s — **SCORE 4.5/10**

- IDEAL lap reproduces (9.72/8.81/6.79/5.55 by cone) — geometry/timing arithmetic is sound and native.
- **The 7.0 s is the weakest-supported number in the panel.** It = (80° ideal 5.55 s) × (1.15–1.30 pad). The
  pad is borrowed from k=1.85; the author's OWN tracker produced `None`/inf. There is no realizability floor
  under it. Adversarially, replace the broken-tracker pad with the only measured datum (k=1.85) and the 80°
  plan flies at 5.55×1.85 ≈ **10.3 s** — i.e. the honest realizable minsnap number could be *slower than
  inc7*, not 7.0 s.
- It is also **explicitly above** the corrected-aero point-mass optimum (4.71 s) by the separability tax
  (geometry-then-timing ≠ joint optimization) AND above inc5 unconstrained RL (6.9 s). On the pure time lens
  it offers no frontier.
- Honest realizable band: **6.5 s (optimistic, good unbuilt tracker, 80° cone) to ~10 s (k=1.85 reality).**
  Midpoint ~8 s. The 7.0 s sits at the optimistic edge. At the doctrine 60° cone it's ~9.7 s ideal → ~11+ s
  realized = no better than inc7.
- **Time verdict:** a clean, free, deterministic *line generator* whose achievable TIME is dominated by an
  unmeasured tracker and a separability tax. Not a time frontier. Score reflects: correct floor, badly
  optimistic ceiling, no realizability evidence.

### HYBRID (decomposed: offline line + RL/MPCC progress-reward tracker) — claimed 8.0 s — **SCORE 5.5/10**

- **Most intellectually honest of the four on this lens.** It refuses the point-mass floor (calls 4.7 s
  "fiction"), measures the slew demand (169° tilt, 90 rad/s axis slew, 14.3 g decel on the shipped geometry),
  and quotes a k-dilated 7.5–8.5 s offline band with explicit ±uncertainty. This is the right *shape* of
  estimate.
- **But on the TIME lens specifically, its own number indicts it:** 8.0 s offline / ~9–10 s deployed is
  **slower than inc5 unconstrained monolithic (6.9 s)** and barely faster than inc7 (9.76 s). The
  decomposition's lower ceiling is the whole story: it caps achievable time *below* learn-the-line, and
  learn-the-line is already live. Its self-assessment §3 says exactly this.
- The 8.0 s rests on k≈1.5–1.7, **interpolated, not measured** (the measured datum is k=1.85, which gives
  ~8.8 s; a mediocre RL-tracker could be worse). Its native tracker also failed (5–35 m miss). So even the
  8.0 s carries an unbuilt-tracker assumption — but at least the assumption is conservative and bracketed by
  the k=1.85 datum, unlike minsnap's optimistic 1.15–1.30.
- Genuine time advantage it claims: drag-wall-limited TOPP means tilt buys little (65°→4.93, 80°→4.73), so it
  can live near the 60° cone at only ~0.2–0.3 s cost — IF you accept the thrust-magnitude (78.3·sinθ) budget.
  Per §0 that budget over-spends altitude, so this "cheap cone" advantage is partly illusory; the honest cone
  cost is closer to the g·tan(θ) curve.
- **Time verdict:** honest, conservative, and *honestly mediocre on time* — the decomposition trades ceiling
  for safety/determinism, and the lens here is time. Score reflects: best-calibrated estimate in the panel,
  but the calibrated answer is "slower than the already-live monolith."

### togt_cpc (time-optimal point-mass-through-gate, regenerated corrected-aero) — est 4.9 s — **SCORE 4.0/10**

- **Highest CEILING of the four** — it is the gold-standard joint geometry+timing optimizer, and the honest
  corrected-aero alt-tracking TOPP cross-validates to THREE independent measured anchors (inc5 rw_tilt=96
  9.52 ↔ 60° 9.64; inc7 9.76; inc5 unconstrained 6.6–6.9 ↔ 75° 6.57; refined CSV 4.71 ↔ 83° 4.51 + rate tax).
  The 4.9 s is the most-defensible *point-mass-plus-rate-tax* number in the panel.
- **But the lens penalizes exactly its weakness.** The 4.9 s is conditioned on (a) relaxing the cone to
  ~78–80° AND (b) a tracker good enough to fly a ~30 m/s line riding the thrust ceiling 84% of the lap with
  zero actuation headroom — and the ONLY tracker datum is k=1.85 (→ would dilate the 4.51 s 83°-plan to
  ~8.3 s). The refined "feasible" line demands **17 rad/s body rates** the ~11.2 rad/s super-rate plant
  CANNOT produce, so even the 4.71 s CSV is not flyable as-is.
- At the doctrine 60° cone, **togt_cpc = 9.6 s = inc7**. Zero marginal value at the allowed envelope.
- It is a **plan-line generator, not a closed loop** — least robust standalone artifact (plans to the
  ceiling, no disturbance rejection), and needs WSL/C++ to even regenerate faithfully.
- **Time verdict:** the best *ceiling* but the worst *realizability-to-ceiling ratio*. On a lens that
  penalizes "time assumes a tracker that does not exist," togt_cpc is the most-penalized: its headline 4.9 s
  is doubly conditional (cone relaxation + unbuilt good tracker) and its "feasible" line violates the rate
  ceiling. Score reflects: real ceiling, but the achievable-NOW time is ~9.6 s (60°) and the fast number is
  the most speculative in the panel.

### mpcc (Model Predictive Contouring Control as decomposed-S2 tracker) — est 5.8 s — **SCORE 6.0/10**

- **The strongest achievable-time story of the four, conditional on a build.** Unlike the other three, MPCC
  is the *tracker itself*, not a planner needing a separate (missing) tracker — it directly attacks the
  realizability gap that sinks minsnap/togt. Its honest 60°-cone number 5.6–6.4 s (midpoint 5.8) uses an 88%
  speed-keep grounded in the racing-MPCC literature (Romero 2022, within ~10% of time-optimal on hardware) —
  a *defensible* tracker assumption, not a borrowed pad.
- **It is the only approach whose fast number could beat inc5 unconstrained (6.9 s)** while staying inside or
  near the doctrine cone — because MPCC discovers speed online and can ride the corrected-aero envelope a
  fixed offline profile cannot. 5.8 s @ 60° → ~5.4 s @ 75°.
- Adversarial penalty (and it's real): the 5.8 s is a **projection gated on an unbuilt acados/WSL tracker**;
  its own toy tracker hit only ~18–21 m/s (half the wall). The author flags this as "the single biggest
  unknown" — MPCC might recover 88% (→5.8 s) or limp to 60% (→~8 s, no better than k=1.85). So the time has a
  fat lower tail. It also amplifies model error (plans to its internal model) and is more latency/perception-
  brittle than inc7's margin-absorb doctrine; the 0.23 m gate-4 tube < 0.73 m N-axis world-fix σ.
- The 88% speed-keep is the most *evidence-backed* tracker assumption in the panel (external literature +
  a clear mechanism for why MPCC beats geometric/DF trackers), which is why it scores above minsnap/togt
  despite the same "unbuilt tracker" caveat.
- **Time verdict:** best ceiling-AND-realizability combination, because it is the tracker, not a plan needing
  one. Penalized for being a projection (5.8 s is unmeasured, fat lower tail) and for the highest integration
  cost. But on the pure time lens — *fastest defensible flyable number inside the doctrine cone* — it leads.

---

## 3. Ranking on the TIME lens

| rank | approach | score | one-line time verdict |
|---|---|---|---|
| 1 | **mpcc** | 6.0 | Fastest *defensible* flyable number (5.8 s @60°, 5.4 @75°); it IS the tracker, so it attacks realizability directly. Penalized: unbuilt, fat lower tail, highest integ cost. |
| 2 | hybrid | 5.5 | Best-calibrated estimate, but the honest answer (8.0 s offline) is *slower than the already-live monolith* — decomposition caps the ceiling. |
| 3 | minsnap_topp | 4.5 | Correct ideal floor, but 7.0 s rests on a borrowed pad over a tracker that returned `None`/inf natively; honest band 6.5–10 s. No time frontier (above 4.71 bound AND 6.9 RL). |
| 4 | togt_cpc | 4.0 | Highest ceiling, worst realizability-to-ceiling ratio: 4.9 s needs cone-relaxation AND an unbuilt good tracker, and its "feasible" line breaks the 11.2 rad/s rate ceiling (demands 17). At 60° cone = inc7. |

**Why mpcc > hybrid** despite hybrid's more honest self-assessment: the lens is *achievable time*, and MPCC's
defensible time (5.8 s) is materially faster than hybrid's defensible time (8.0 s), for the same decomposed-S2
structural cost. MPCC is essentially "the good tracker that makes hybrid/togt/minsnap's lines worth flying."

**Why togt_cpc is LAST despite the best ceiling:** the lens explicitly penalizes time that "assumes a tracker
that does not exist." togt_cpc is the purest example — it is a plan with no closed loop, its fast number is
doubly conditional, and its nominal "feasible" line is itself infeasible on the real rate plant. Its true
*achievable-now* time at the allowed cone is ~9.6 s.

---

## 4. The comparator that beats most of the field: MONOLITHIC RL

This is the load-bearing context for the TIME lens. The four candidates are measured against what already
exists:

- **inc7 monolithic (constrained, 60° cone): 9.76 s offline / 11.45 s live — MEASURED, LIVE-CONFIRMED.**
- **inc5 unconstrained monolithic: 6.9 s twin — MEASURED.**

On the pure time lens, the monolith is the only approach with a *measured realizable* number on the corrected
plant, and unconstrained it (6.9 s) already beats minsnap (7.0 s claimed / ~8–10 s honest), hybrid (8.0 s),
and ties the envelope that togt/mpcc need a from-scratch tracker to reach. The corrected-aero point-mass
bound is 4.72 s; the residual from inc7 to that bound is ~5.0 s, of which ~2.6 s is the style cone (a
relaxation lever that helps the monolith too) and ~2.2 s is planning+tracking+policy slack (A3 §3).

**The decisive time-lens conclusion:** none of the four trajectory-opt approaches is *clearly faster, as a
flyable line at the doctrine-allowed envelope, than retraining the already-live monolith*. Their fast numbers
(togt 4.9, mpcc 5.8) all require (a) the same cone relaxation that speeds the monolith and (b) an unbuilt
tracker. The monolith already removed the inversion pathology (plant fix + DR) that was the decomposition's
main reason to exist.

---

## 5. Best for VQ2 on the TIME lens

**MPCC (as the decomposed-S2 tracker), IF the S2 decision goes decomposed — but the time lens alone does NOT
justify switching off the monolith.**

- MPCC is the best *achievable-time* candidate among the four because it is the tracker that makes any
  offline line flyable near the point-mass envelope (5.4–5.8 s defensible), and it makes contact-free an
  explicit constraint (the only approach that does).
- BUT the honest time-lens read is that the **fastest realized-NOW path is "relax the cone + retrain the
  monolith"** (already live, 6.9 s unconstrained measured), not any of the four planners. MPCC/togt only win
  *after* a high-cost unbuilt tracker materializes and only *with* the same cone relaxation.
- The natural division: **togt_cpc or minsnap_topp generate the offline contact-free line; MPCC tracks it.**
  That decomposed stack's honest ceiling is ~5.4–5.8 s — genuinely faster than inc7's 9.76 s — but it is a
  three-subsystem build whose realizability is unproven, versus a monolith that is one subsystem and live.

**Recommendation feeding the S2 adjudication (time lens only):** crown MPCC as the time winner *of the four*,
but flag that on pure achievable-time-NOW the monolithic-RL-with-cone-relaxation dominates all four, and the
decomposed stack (line-gen + MPCC) is the *fallback* whose time ceiling (~5.5 s) only justifies its
integration cost if monolithic style/inversion pathologies re-emerge under relaxation.

---

## 6. Caveats / confidence

- IDEAL lap-by-cone numbers: re-run natively, reproduce exactly — **HIGH**.
- Tracker-failure evidence (minsnap `None`/inf, hybrid 5–35 m, mpcc ~20 m/s): re-run / read from artifacts —
  **HIGH** that naive trackers fail; these are lower bounds on tracker quality, not verdicts on a *good*
  tracker.
- The g·tan(θ) vs 78.3·sinθ cone disagreement (§0): the g·tan(θ) cone is the honest binding constraint for an
  altitude-holding line; the descent gives only ~1.7% credit. **MEDIUM-HIGH** — this is the single biggest
  modeling judgment and it drives the 60°-cone spread (9.6 vs 5.4 s). If a future analysis shows the policy
  can profitably trade altitude on the descent, the hybrid/mpcc thrust-budget cone gets more generous and
  their 60°-cone numbers improve.
- k-dilation realizability (the 85–92% / 88% speed-keep for MPCC, k=1.5–1.7 for hybrid): **borrowed/projected,
  the dominant uncertainty.** The only measured datum is k=1.85 (geometric). MPCC's 88% is literature-backed
  but unmeasured on this plant. Every fast number has a fat slow tail under this.
- expl_corrected_aero CSV: the *refined* CSV peaks 39.26 m/s (= drag wall), NOT 60 m/s as some artifacts
  claim — the refined lap is genuinely drag-wall-limited (~4.71/5.64 s). The "60 m/s fiction" warning applies
  to the linear-plant bound_* cases (51–55 m/s), not the corrected refined CSV. Minor correction; does not
  move the verdict.
- All four scores are on TIME ONLY; integration cost, determinism, and robustness are noted only where they
  bound achievable time (e.g. "unbuilt tracker" → time is a projection). A different lens reorders them.
