# Phase B — Judge Panel, Lens = ROBUSTNESS + VALIDITY

**Date:** 2026-06-13 · **Judge:** Phase-B robustness adjudicator (opus) · **For:** Fengyou

Lens definition (binding): contact-free guarantee strength (gate contact = INVALID run; gate-4
binding margin 0.155 m simstart @ r=0.38), sensitivity to plant error / 67 ms latency / perception
noise / fresh-sim respawn, and failure mode (graceful vs catastrophic). NOT speed. Speed claims are
in scope ONLY to expose where a robustness story rests on an unjustified achievable-time.

This panel scores the FOUR proposals (two were supplied verbatim; I additionally place the two
incumbents — pure MONOLITHIC and pure TOGT-collocation — on the same lens for ranking context, as
the prompt's "four trajectory-opt approaches" is the planning ladder and the adjudication target is
S2 = decomposed vs monolithic vs hybrid). Where the prompt supplied exactly two JSON proposals
(`minsnap_topp`, `HYBRID`), those are scored in full; the monolithic and TOGT poles are scored as
the comparison frame the S2 decision actually needs.

---

## 1. What the code + data actually say (evidence base, re-derived)

### 1.1 The shipped 4.55 s reference line is a near-INVERSION, fully-saturated line
Measured directly from `rl/reference_line_vq1.json` (368 samples, quat_wxyz → R33 tilt):

| metric | value |
|---|---|
| thrust-vector tilt p50 | **74.3°** |
| tilt p90 | **111.1°** |
| tilt max | **170.4°** (essentially inverted) |
| inverted samples (tilt > 90°) | **95 / 368 = 26 %** |
| thrust_norm saturated (>0.95·max) | **92 % of lap** |
| peak speed | 51.1 m/s (the FALSIFIED linear plant) |

Consequences for the lens:
- The shipped line is **catastrophic as geometry** on the corrected plant — it commands a 170°
  inversion and 51 m/s the corrected plant cannot reach (drag wall ~36–39 m/s). The HYBRID's
  stated fallback ("reuse the shipped spatial path, discard timing") is **not available**; the
  HYBRID's own key_risks admit this. The geometry MUST be re-planned. Net: the HYBRID does NOT get
  a free, audited, shipped geometry asset — that part of its determinism/audit story is fiction
  until a new line is built.
- It confirms WHY the monolithic inc1 produced the "backflip-dive": a time-optimal line on a
  high-T/W plant genuinely wants to point thrust backward. The monolith's fix was plant-correction
  + DR + the R4 tilt hinge (free-cone 60°, quadratic beyond), NOT a structural ban. So the
  monolith's inversion-safety is a SOFT reward guarantee; the decomposed line's is a HARD planner
  guarantee — but only if the PLANNER doesn't demand inversion (see 1.2).

### 1.2 A gentle min-snap line through the gate centres does NOT demand inversion
`probe_robustness.py` (clamped-cubic through start+6 gates, the proposal's own conservative seed):

```
line length to gate-5 plane = 164.65 m
kappa_max = 0.0334  ->  r_min = 30.0 m   (geometrically gentle — matches the 23–73 m claim)

tilt_cap  topp_t  v_max  v_wall  geomtilt_p50  geomtilt_max
   60      5.53    36.0   36.1      54.1         60.0
   65      5.22    36.9   36.9      56.5         65.0
   75      4.75    38.1   38.1      59.5         75.0
   80      4.63    38.5   38.5      60.5         78.8
```

Findings (adversarial):
- **The ideal coupled-TOPP lap is reproducible** (~5.5 s @ 60°, ~4.6 s @ 80°) and sits ABOVE the
  TOGT collocation optimum (4.71 s) — the geometry-then-timing separability tax is real, as
  `minsnap_topp` honestly states.
- **The ideal lap is DRAG-WALL-limited, not cornering-tilt-limited.** v_max ≈ v_wall at every cap;
  the gentle geometry (r_min 30 m) means centripetal demand (kappa·v² ≤ ~30 m/s²) never approaches
  the lateral cap. So the proposal's claim "the tilt cone is binding at every cap" is **misleading
  for cornering** — what actually binds is (a) the v² drag wall and (b) geometry-forced tilt to
  hold the descent. This is the SAME binding structure the HYBRID names, and it is correct: high
  tilt buys little (5.53 → 4.63 s for 60° → 80°), so a min-snap/decomposed line can also be planned
  near the 60° cone at small time cost. `minsnap_topp` UNDER-credits itself here.
- **Geometry-forced tilt p50 = 54–60°, max ≤ 79°, ZERO inversion.** This is the crux: a gentle
  min-snap line does NOT force the 88–107° tilt the proposal quotes (that number was lifted from
  the SATURATED shipped TOGT line, 1.1). The honest figure is p50 54–60°, i.e. the line sits right
  at the 60° style cone from descent geometry alone — relaxation to ~65–70° is needed but NOT the
  drastic 75–80° the proposal fears. The proposal is internally inconsistent (it cites both 54° via
  its own TOPP and 88–107° via the wrong line); the correct reading STRENGTHENS its contact-free /
  no-inversion story relative to its own writeup.

### 1.3 Latency (67 ms) is validity-benign; perception is the real validity threat
From `probe_robustness.py` margin-erosion block (gate-4 binding margin 0.155 m, simstart, r=0.38):

- A held 2–5° attitude error over the 67 ms (2-tick) latency produces **0.77–1.93 mm** of
  cross-track displacement at ANY speed up to 37.7 m/s — i.e. **<1.3 % of the 155 mm margin.**
  Latency is a time/along-track effect (folds into the k-dilation), not a validity threat. Both
  proposals are correct to fold it into k.
- **Perception is the binding validity risk for any world-frame line-tracking approach.** VISION-PKG2
  world-fix East σ ≈ 0.47 m = **3.0× the 0.155 m gate-4 binding margin, unfiltered.** A KF must reach
  <0.05 m 1-σ (to keep 3-σ inside 0.155 m) at the post-gate-3 37 m/s window. Every approach that
  tracks a WORLD line inherits this; the monolith — which consumes the SAME world fix through the
  same KF — is no better OR worse. Decomposition does NOT help perception robustness, contrary to
  any implied "cleaner line = safer" intuition. The shared limiter is the estimator, not the planner.

### 1.4 The metric instrument confirms the binding gate and its fragility
`rl/contact_true_eval.py` is the canonical inc8 selection metric: contact-true margin =
(0.75 − r) − L∞, slab/frame volumetric, gate-3 isolated, gate-3/4/5 D-offset fragility probe. The
prompt's binding-gate datum (gate-4 simstart L∞ 0.215 → margin 0.155 m @ r=0.38, tightest of 6;
gate-3 D-offset metric flips at ~1.5 m) is the lens's hard floor. KEY: the offline twin is **blind
to fresh-respawn post-gate-3 sensitivity** (the bimodal-lap-time / HOME-reset sub-tick spawn-state
finding) — so EVERY approach's offline margin must be discounted by the WINNER-VALIDATION RIDER
(crown only after a ≥3–5-lap fresh-reset live batch). This rider applies equally to all four and is
therefore rank-neutral, but it CAPS the confidence any offline-only robustness claim can earn.

---

## 2. Per-approach scoring on the ROBUSTNESS + VALIDITY lens

### A. `minsnap_topp` — offline min-snap geometry + coupled-TOPP timing — **SCORE 6.5 / 10**

**Contact-free guarantee (strong).** Crossings pinned dead-centre (miss 0.00 m → full 0.37–0.42 m
contact-true band free for the tracker). TOPP curvature cap means the PLANNED line never demands
more lateral accel than the envelope supplies → contact-free *by construction at the plan level*.
Geometry-forced tilt p50 54–60°, ZERO inversion (1.2) → the plan itself is structurally validity-safe.
This is the strongest PLAN-level contact-free story of the four.

**Plant-error robustness (strong).** TOPP is parameterised by (a_up_max, c2), both DR-banded; plan at
conservative values → robustness for free via per-track determinism. Pure numpy/scipy, bit-identical
per track, <1 s on the laptop — the ideal offline line-iteration engine (LEGAL flywheel).

**Where it's weak / where the claim is unjustified.**
- **It is only a PLANNER.** Its robustness is the robustness of a *reference*, not of a closed loop.
  The entire validity burden moves to a tracker it does not specify and did not build (its own
  prototype "had a sign bug I declined to chase"). On THIS lens, an unspecified tracker is the
  dominant unmodeled risk — the 8.3 s k=1.85 datum proves the existing geometric tracker needs huge
  dilation and STILL we have no evidence it holds a 0.155 m cross-track budget at 37 m/s. The
  contact-free-by-construction guarantee is at the PLAN; the realized run can still strike if the
  tracker lags. Score is capped here.
- **The 7.0 s est_lap is a planner-ideal + hand-waved 15–30 % pad.** Not load-bearing for the lens,
  but it signals the same gap: no closed-loop evidence.
- **Requires envelope relaxation** to be competitive (its own envelope_validity). On robustness this
  is neutral-to-slightly-negative: a 65–70° cone is mild (1.2), but ANY relaxation widens the live
  attitude excursion the perception/KF must survive.

**Failure mode: GRACEFUL.** A non-saturated min-snap line (unlike the 92%-saturated shipped line)
leaves the tracker headroom to recover lag → degradation is smooth, not a cliff. If the tracker
falls behind it slows/widens rather than inverting. This is its single best lens property.

**Net:** Best PLAN-level contact-free guarantee, fully deterministic/offline, graceful failure — but
its closed-loop validity is UNDEMONSTRATED and entirely outsourced to an unspecified tracker. 6.5.

---

### B. `HYBRID` (decomposed): offline plan-line + RL/MPCC tracker, arc-length progress reward — **SCORE 6.0 / 10**

**Contact-free guarantee (strong-in-principle).** Plan crossings ≤0.13 m from centre; progress
reward over the planned contact-safe line → the tracker is NEVER rewarded for cutting inside the
frame (this is the genuine, real robustness win of decomposition: it removes the corner-cut
incentive that a pure progress-to-gate-centre monolith reward can create). Inversion structurally
impossible IF the planned line doesn't demand it — and a drag-wall-limited line at 36–38 m/s does
not (1.2). So the "inversion impossible" claim is TRUE, but it is a property of the PLANNER it shares
with `minsnap_topp`, plus a tracker tilt-clamp — not a unique architectural magic.

**Plant-error robustness (good).** Feedforward from the planned line + feedback absorbs mismatch;
line re-iterable offline; the cleanest plant-error story of the closed-loop options.

**Where it's weak / where the claim is unjustified (adversarial).**
- **The 8.0 s est rests on a BORROWED k-dilation (1.5–1.7).** That k is interpolated from the
  geometric-tracker k=1.85 datum, NOT measured for any RL/MPCC tracker. The HYBRID itself flags this;
  on the lens it means the realizable band (7.5–8.5 s offline, ~9–10 s deployment) has real
  uncertainty AND, worse, we have NO evidence the chosen tracker holds the 0.155 m gate-4 budget. A
  tracker that needs k=1.7 to track is by definition lagging — lag at 37 m/s is exactly the
  validity-eroding regime, even though latency-per-se is benign (1.3). The robustness claim and the
  speed claim share the same unproven tracker.
- **Shipped-line reuse is FICTION (1.1).** The HYBRID's "take the shipped spatial path" is invalidated
  by the 170° inversion measurement; geometry must be re-planned, so its "static shipped asset /
  cleanest audit story" is not yet real — it inherits min-snap's planner-build cost PLUS a tracker
  campaign.
- **World-frame line couples perception σ directly to cross-track (1.3): 0.47 m East σ = 3× the
  binding margin.** The HYBRID names this honestly. It is NOT worse than the monolith, but
  decomposition buys NOTHING here — a key over-claim to neutralize ("not helped by decomposition
  either," correctly).
- **Integration cost HIGH, ceiling LOW.** acados/MPCC does not build on Windows; an RL-tracker needs a
  fresh line-relative obs layout + a NEW Adroit campaign (monolith checkpoints NOT reusable). And the
  inc5 unconstrained monolith already does 6.9 s twin — likely FASTER than the HYBRID's 8.0 s. On a
  pure robustness lens speed is secondary, but "slower AND a full new campaign AND no perception
  benefit" is a weak trade unless style pathologies actually recur (they have not — inc7 is
  live-confirmed clean).

**Failure mode: GRACEFUL-to-MIXED.** Tracker clamps tilt → inversion impossible (good). BUT if the
tracker lags at the post-gate-3 high-speed approach, it lags into the binding gate-4 — the failure
is a quiet margin-erosion, not a loud crash, but it lands exactly at the tightest gate. Graceful in
attitude, fragile in cross-track at the worst place.

**Net:** Strongest *incentive-level* contact-free guarantee (no corner-cut reward) and best
plant-error story, but its validity rests on an unbuilt, unmeasured tracker; reuse-of-shipped-line is
fiction; high integration cost; no perception benefit; probably slower than the live-confirmed
monolith. 6.0.

---

### C. MONOLITHIC RL (inc7-class, learn-the-line) — **SCORE 7.5 / 10** *(incumbent pole; the comparison frame)*

**Contact-free guarantee (EMPIRICAL, strong).** inc7 is LIVE-CONFIRMED: standing 5/5 FINISHED,
gate-3 barrier GONE (0 collisions offline AND 0/5 live), contact-true gate-4 margin 0.155 m simstart
holds in the metric instrument. The guarantee is SOFT (R4 tilt hinge + contact-true DR geometry,
not a hard ban) but it is the ONLY guarantee BACKED BY LIVE FLIGHT. On a lens that rewards
demonstrated validity over asserted validity, this is decisive: every other option's contact-free
claim is a plan-level or incentive-level argument; this one has 7 live sessions of zero contact.

**Plant-error robustness (strong, demonstrated).** The geometry-honesty thesis is LIVE-VALIDATED:
contact-true volumetric training + structured DR absorbed the +3.5 m/s² N+D climb-bin residual via
MARGIN, fixed gate-3 on the first try. Robustness from honest geometry + DR + diversity — the
binding doctrine — is PROVEN here, not hoped for.

**Where it's weak.**
- **Narrow convergent basin (2/3 seeds viable).** S_stable risk; inc8-class retrains need more seeds
  budgeted. A robustness debit, but a TRAINING-process risk, not a deploy-time one.
- **Soft inversion guarantee** could in principle regress under a reward/plant change — but inc1's
  root cause (wrong plant + no DR + reward/termination conflict) is understood and fixed, and inc7
  shows no inversion. The decomposed options' HARD guarantee is genuinely stronger IN PRINCIPLE; the
  monolith's is genuinely stronger IN EVIDENCE.
- **Fresh-respawn post-gate-3 sensitivity (bimodal laps)** — but this is the WINNER-VALIDATION RIDER
  that caps ALL four equally, and the monolith is the only one that has ALREADY passed a live fresh
  batch.
- **Perception:** same world-fix dependence as everyone (1.3); no better, no worse.

**Failure mode: GRACEFUL-AND-OBSERVED.** Live evidence shows it slows/widens, never tumbles;
command saturation NONE (max |tanh|≈0.547); crab 63° ≈ predicted. We have actually SEEN how it fails.

**Net:** The only option with LIVE-CONFIRMED contact-free validity and live-validated plant-error
robustness. Soft (not hard) inversion guarantee and narrow basin are the only debits. 7.5.

---

### D. TOGT collocation (full nonlinear time-optimal, e.g. corrected-aero refine) — **SCORE 4.5 / 10** *(pole; the ceiling engine)*

**Contact-free guarantee (NOMINAL strong, REALIZED weak).** TOGT can plan contact-free with explicit
gate margins (the 4.27 s bound was inscribed-circle CONTACT-FREE). BUT the only TOGT lines we have
are FALSIFIED: the shipped 4.55 s line is a 170° inversion at 51 m/s (1.1); the corrected-aero
`expl_corrected_aero` CSV peaks 60 m/s with no drag wall (fiction, per the prompt). A line that
demands inversion + impossible speed is contact-free ON PAPER and catastrophic in transfer.

**Plant-error robustness (POOR).** Full collocation rides the plant model to saturation (92% of lap)
→ it is maximally sensitive to any plant error: a wrong drag wall or thrust map and the line is
infeasible, not just slow. No feedback in the plan; the realizability dilation (k=1.85) IS the
measure of how badly the open-loop optimum transfers.

**Where it's catastrophic.** Saturated + inverted + on a falsified plant + no Windows toolchain (WSL,
do-not-re-run) + needs a separate tracker anyway. As a STANDALONE S2 it is a non-starter on
robustness. Its legitimate role is a BOUND/datum and a geometry seed for a decomposed line — NOT a
deployable approach.

**Failure mode: CATASTROPHIC.** A saturated inverted line has zero headroom; any lag or plant
mismatch → frame strike or inversion → INVALID run.

**Net:** Best ceiling, worst robustness; valuable only as an offline bound + geometry seed. 4.5.

---

## 3. Ranking (robustness + validity lens, best first)

1. **MONOLITHIC RL (inc7-class) — 7.5** — only LIVE-CONFIRMED contact-free validity; live-validated
   plant-error robustness; graceful, observed failure. Soft inversion guarantee + narrow basin are
   the debits.
2. **minsnap_topp — 6.5** — best PLAN-level contact-free guarantee, fully deterministic/offline,
   graceful failure, no inversion in the geometry; but closed-loop validity is undemonstrated and
   outsourced to an unspecified tracker.
3. **HYBRID (decomposed) — 6.0** — strongest incentive-level contact-free guarantee (no corner-cut
   reward) + best plant-error story; but validity rests on an unbuilt/unmeasured tracker, shipped-line
   reuse is fiction, high integration cost, no perception benefit, likely slower than the live monolith.
4. **TOGT collocation — 4.5** — best ceiling, catastrophic robustness; the only lines we have are
   falsified (inversion, impossible speed); role is bound + geometry seed, not a deployable S2.

---

## 4. Best for VQ2 on THIS lens

**MONOLITHIC RL (inc7-class), retrained on the corrected plant for inc8.** On the robustness +
validity lens specifically, demonstrated validity dominates asserted validity: it is the ONLY option
with live-confirmed zero-contact flight and live-validated plant-error robustness, its failure mode
is graceful AND observed, and the only thing the decomposed options offer that it lacks — a HARD
(vs soft) inversion ban — addresses a pathology (inc1 backflip) whose ROOT CAUSE is already fixed and
which inc7 does not exhibit. The prior reviewer's read ("retrain monolithic first; decomposition is
the fallback if style pathologies persist") is correct on this lens.

**Caveat / where I'd hedge:** if envelope relaxation (rw_tilt 96→48, free cone 60→75–80°) is pushed
hard for speed, the monolith's SOFT tilt guarantee weakens exactly as the attitude envelope widens —
that is the one regime where decomposition's hard tilt-clamp earns its keep. So the robust play is:
**monolith as primary, with `minsnap_topp` as the offline geometry/line-iteration flywheel kept warm**
(it is ~free to maintain, fully deterministic, and is the natural seed for a decomposed fallback or a
HYBRID arc-length progress reward if the relaxed-envelope monolith starts cutting corners). The
HYBRID is the right STRUCTURE for that fallback but should not be built speculatively — its tracker
is unproven and its perception exposure is identical to the monolith's.

**Hard floor that caps everyone:** perception East σ 0.47 m = 3× the 0.155 m gate-4 binding margin;
no planning architecture fixes this — the KF/estimator must reach <0.05 m 1-σ at the post-gate-3
window or NO approach is valid at VQ2 race speed. Robustness work should prioritise the estimator
over the planner.

---

## 5. Method / caveats

- Coupled-TOPP and geometry-tilt numbers from `probe_robustness.py` (this dir): clamped-cubic line
  through start+6 gate centres (the proposal's own conservative seed), forward-backward TOPP with
  a_up_max=78.3, c2=0.052, the proposal's own lateral cap min(a_h, g·tan(tilt)). Reproduces the
  proposal's ideal-lap bracket; corrects its tilt-binding and 88–107° geometry-tilt claims.
- Shipped-line tilt/thrust/speed from `rl/reference_line_vq1.json` directly (368 samples).
- Latency cross-track = 0.5·g·tan(eps)·dt² with dt=0.067 — first-order; a true closed loop would be
  smaller (feedback corrects within the next tick). Conservative upper bound; still <1.3% of margin.
- Scores are on the ROBUSTNESS+VALIDITY lens ONLY; speed-frontier or integration-velocity lenses
  would re-rank (TOGT/monolith rise on ceiling; min-snap rises on integration cost).
- The two incumbents (monolith C, TOGT D) are scored as the comparison frame the S2 decision needs;
  the two supplied JSON proposals (A, B) are scored in full as requested.
