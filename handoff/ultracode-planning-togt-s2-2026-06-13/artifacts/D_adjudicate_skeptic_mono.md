# Phase D — S2 Architecture Adjudication (lens: SKEPTIC-OF-MONOLITHIC)

**Date:** 2026-06-13 · **Author:** Phase-D adjudicator (opus-4.8), ultracode S2 fan-out · **For:** Fengyou
**Scope:** analysis + offline numpy only. No tracked source edited; no live sim, SLURM, or network.
All load-bearing numbers re-derived natively (`.venv`, `PYTHONPATH=src`) from repo data + the frozen
Phase-A/B/C artifacts in this directory. New verification script: `verify_honest_tilt_topp.py`.

**Lens charge:** stress-test the case FOR decomposition / hybrid. Argue the strongest possible
skeptic-of-monolithic position — the monolith's freedom PRODUCED the inc1 backflip-dive; an explicit
plan-line makes inversion structurally impossible; the shipped `reference_line_vq1.json` + `progress()`
already exist; the offline line-iteration flywheel exploits determinism-per-track; MPCC/RL-as-tracker
over a contact-free line guards the gate-4 binding margin (0.155 m) by construction. Then be honest if
the evidence does not support switching off the live monolith.

---

## 0. Verdict in one paragraph

**Recommendation: STAGED — retrain the MONOLITH on the corrected-aero plant for inc8 (one ladder step of
cone relaxation, rw_tilt 96→48), grafting an arc-length progress reward over a *rebuilt* contact-free
min-snap reference line (the HYBRID / MPCC-cast 3rd option), and build the fully DECOMPOSED stack only as
a warm fallback gated on a measured trigger.** I went in trying to crown decomposition and could not, on
the evidence. The skeptic case has exactly one durable win — **inversion / corner-cut becomes structurally
impossible** — and that win is real and worth importing. But the two pillars the skeptic case needs to
*beat* the monolith both collapse under verification: (1) **the speed prize evaporates at the legal
envelope** — my independent honest tilt-capped point-mass TOPP puts the 60° doctrine-cone bound at **~10.6
s**, AT or ABOVE inc7's already-live 9.76 s twin, so at the allowed cone NO planner (decomposed, hybrid, or
monolithic) is faster than where the monolith already sits; speed below ~9 s is a *cone-relaxation* lever
that speeds the monolith equally, not an architecture lever; and (2) **decomposition's central enabling
assumption — a good tracker — is the single largest UNPROVEN risk in the project**: the only measured datum
is k=1.85 (8.3 s to track the optimal line, 54-combo sweep finds nothing faster), and three independent
native trackers have failed. The honest graft (progress-reward-cast monolith) captures decomposition's one
real win at near-zero architectural cost without betting the program on an unbuilt tracker.

---

## 1. Method — what I verified independently (not just inherited)

I treated the Phase-A/B/C artifacts as claims to be checked, and re-derived the four load-bearing numbers
the skeptic case lives or dies on. All reproduced; constants read live from `src/racer/rl_plant.py`.

| # | claim under test | my independent check | result |
|---|---|---|---|
| V1 | corrected aero constants (78.28 m/s² full-stick; drag wall ~38.6 m/s) | read `COLL_MAP_ACCEL_MEASURED`, `QUAD_DRAG_C2_*` live; closed-form `v=√(√(A²−g²)/c2)` | **78.2828 m/s² = 7.98 g; drag wall 38.65 m/s.** EXACT. |
| V2 | **the 60° tilt-cap dispute** (C1 5.35 s "cornering-only" vs verifier ~9.8 s "honest total-tilt") | wrote `verify_honest_tilt_topp.py` — same centre line, both cap definitions, friction-circle braking, descent-gravity credit | **honest 60° = 10.62 s** (cornering-only = 5.53 s). Verifier UPHELD. |
| V3 | k=1.85 / 8.3 s structural tracker gap | grep `memory/project_phase2_rl_vision_decisions.md` §TOGT-BOUND | **confirmed**: geometric tracker first valid 6/6 at k=1.85, twin lap ≈8.3 s, 54-combo sweep finds nothing faster, "gap is STRUCTURAL". |
| V4 | shipped line reuse is fiction (inverted + drag-infeasible) | loaded `reference_line_vq1.json`, computed R33-tilt + speed | **max tilt 170.4°, 25.8% inverted, peak 51.1 m/s vs 38.6 m/s drag wall.** Reuse impossible. |

`verify_honest_tilt_topp.py` output (the decisive run):
```
 cap   A:cornering-only   B:honest-total-tilt
  60     5.53s v36.0       10.62s v18.7
  65     5.22s v36.9        9.57s v21.1
  75     4.75s v38.1        7.27s v29.4
  80     4.63s v38.5        5.91s v36.7
  90     4.59s v38.8        4.57s v39.2   <- converge (no cap to mis-apply)
```
The two readings converge at 90° (no cap) and diverge monotonically as the cap tightens — the exact
signature of a mis-placed cap in reading A. Reading B is physically correct: body tilt is the angle of the
*total* specific force `f = a_des − grav`, and braking from ~38 m/s into a corner tilts the body just as
cornering does. The skeptic case must use reading B (≈10.6 s @ 60°), and reading B is what kills the speed
argument for decomposition.

---

## 2. The skeptic case, stated at its strongest — then tested

I owe the strong form before I dismantle the weak parts. The case FOR decomposition / hybrid:

- **S-1 (structural inversion ban).** The inc1 backflip-dive is the existence proof that a monolith's
  freedom can point thrust backward. An offline plan-line that never demands >cone tilt makes inversion
  *geometrically impossible*; the tracker only clamps to it. The monolith's R4 tilt-hinge is a SOFT reward
  guarantee — it can regress under reward/plant change, exactly the regime envelope relaxation enters.
- **S-2 (no corner-cut incentive).** A progress-to-gate-*centre* monolith reward can pay the policy to cut
  inside the frame at the 0.155 m binding gate-4. An arc-length progress reward over a contact-safe line
  *removes the incentive* by construction. This is the genuine, real robustness win of the decomposed
  reward structure.
- **S-3 (offline line-iteration flywheel).** Determinism-is-per-track ⇒ iterating the line offline across
  attempts is LEGAL. Decomposition turns the per-track line into a free, deterministic, auditable asset the
  monolith's entangled policy cannot expose.
- **S-4 (assets already exist).** `reference_line_vq1.json` + `progress()` are shipped; the path is half-built.
- **S-5 (higher ceiling).** Learn-the-line's ceiling is bounded by what RL discovers; an explicit optimal
  line + good tracker can in principle ride closer to the point-mass optimum.

Now the tests.

### Test of S-5 (higher ceiling) — FAILS at the legal envelope (V2)
The ceiling argument only bites if the decomposed line is *faster than the monolith can be*. At the
doctrine-legal 60° cone, the honest point-mass bound is **~10.6 s** (V2) — i.e. the absolute best ANY
trajectory optimizer can plan at 60° is slower than inc7's already-live **9.76 s twin / 11.45 s deployment**.
The decomposed line has no speed headroom to sell at the legal cone. Speed below ~9 s requires *opening the
cone* (65°→9.6 s, 75°→7.3 s, 80°→5.9 s) — and cone relaxation is a reward/doctrine lever (rw_tilt 96→48,
free-cone 60→75–80°) that speeds the **monolith equally** (inc5 unconstrained monolith already measured 6.9 s
twin with NO planner). So S-5 is not an *architecture* advantage; it is a *cone* advantage available to both.
**S-5: refuted at the legal envelope.**

### Test of S-4 (assets exist) — FAILS (V4)
`reference_line_vq1.json` is built on the FALSIFIED linear plant: max tilt **170.4°**, **25.8% of samples
inverted**, peak **51.1 m/s** against a corrected drag wall of **38.6 m/s**. It is simultaneously inverted
(violates the cone the decomposed line is supposed to *guarantee*) and drag-infeasible (a tracker cannot
follow it at planned speed). The decomposition's "free shipped geometry asset" does not exist — the line
must be **rebuilt on corrected aero** before decomposition is even a valid target. `progress()` survives as
machinery, but the geometry it indexes is fiction. **S-4: refuted.**

### Test of S-1 (structural inversion ban) — UPHELD but de-fanged
S-1 is TRUE: a gentle min-snap line through the gate centres forces only p50 54–60° tilt, ZERO inversion
(`probe_robustness.py`), so a decomposed line genuinely cannot demand a backflip. BUT the *pathology it
guards against is already fixed*: inc1's root cause (wrong plant + no DR + reward/termination conflict) is
understood, and inc7 across 7 live zero-contact sessions exhibits **zero inversion**, command saturation
NONE (|tanh|≤0.547). So S-1 is a hard guarantee against a failure mode that no longer occurs in the live
policy. It is insurance, not a present need — its value is conditional on the SOFT guarantee actually
regressing under cone relaxation, which has not happened. **S-1: real but conditional.**

### Test of S-2 (no corner-cut incentive) — UPHELD, and it is the one to IMPORT
S-2 is the durable skeptic win. It is a property of the *reward structure*, not of the architecture — and
that is precisely why it can be grafted onto the monolith WITHOUT decomposing: cast the monolith as MPCC
with an arc-length progress reward over a contact-safe line (the task's 3rd / HYBRID-monolithic option).
The tracker is then never paid to cut inside the frame, importing decomposition's incentive-level
contact-free guarantee at the cost of a reward change, not a new subsystem. **S-2: upheld → graft it.**

### Test of S-3 (offline flywheel) — UPHELD, and ALSO graftable
S-3 is real and valuable, but it is a property of *having an offline line generator*, not of *being
decomposed*. The hybrid monolith consumes the same offline-iterated min-snap line as its progress
reference, so it inherits the flywheel. The min-snap + coupled-aero TOPP line generator (pure numpy,
deterministic, <1 s — `minsnap_topp_proto.py` is half-built) should be built and kept warm REGARDLESS of
the architecture choice, because it feeds both the hybrid monolith's reward AND any future decomposed
tracker. **S-3: upheld → build the generator, it is architecture-neutral.**

**Net of the skeptic test:** of the five skeptic pillars, the two that would justify *switching off the
monolith* (S-4 assets, S-5 ceiling) are refuted; the two that are real (S-2 incentive, S-3 flywheel) are
graftable onto the monolith without decomposing; and the one unique-to-decomposition win (S-1 hard
inversion ban) guards an already-fixed pathology and is best held as conditional insurance.

---

## 3. The decomposition-killer the skeptic case cannot answer: the tracker (V3)

The skeptic case is built on a load-bearing *assumption* it never discharges: that a good tracker exists.
The evidence on this is the most decision-relevant in the entire panel, and it is bad:

- **The only MEASURED tracker datum is k=1.85** (memory §TOGT-BOUND, V3): the existing geometric tracker,
  fed the optimal line, first achieves a valid 6/6 only at time-dilation k=1.85 ⇒ twin-tracked lap ≈ **8.3
  s**, and a 54-combo gain sweep finds nothing faster. The gap is explicitly STRUCTURAL.
- **Why structural, not tuning (A2 headroom argument, re-derived):** the optimal line rides the thrust
  ceiling ~91% of the lap. A 1× follower that falls behind needs accel ABOVE the ceiling to catch up — there
  is none. Tracking error diverges by design. k=1.85 drops demand ≈1/k² ≈ 0.29× and frees ~70% of the
  ceiling for correction — that is *why* 1.85 specifically restores feasibility.
- **Three of three native trackers have FAILED:** geometric → needs k=1.85; hybrid DF+PD → 5–35 m miss;
  scipy-SLSQP toy MPCC → only ~18–21 m/s (half the 38 m/s wall). C1's own feedforward+PD twin diverges 37.9 m.

The skeptic's own A2 number is decisive here: **DECOMPOSED-with-a-naive-tracker = 8.3 s, which is SLOWER
than even the style-ON inc7 monolith (9.76 s twin would be the comparator, but at the *same legal cone* the
geometric tracker's 8.3 s is on the optimal line whose 60° honest bound is ~10.6 s — the geometric tracker
is riding a faster-than-legal line and still only hits 8.3 s, and it does so by paying a 3.87 s structural
tax the monolith never pays).** Decomposition only beats the monolith if an UNBUILT MPCC/RL tracker recovers
85–90% of point-mass speed instead of limping to k≈1.85. That single number — the real k on the corrected
plant for a *good* tracker — is unmeasured, and the entire decomposed ceiling (~5.4–5.8 s) is a paper number
until it is measured. **You do not switch a live, validated subsystem off to chase a paper number gated on an
unbuilt tracker.**

---

## 4. Where the skeptic case is RIGHT about the monolith (honest debits)

I will not whitewash the monolith. The genuine monolithic risks the skeptic correctly identifies:

- **D-1 SOFT inversion guarantee.** Real. Under aggressive cone relaxation (toward 75–80°) the R4 tilt-hinge
  could in principle license attitudes that drift toward corner-cutting. This is the ONE regime where S-1's
  hard ban earns its keep — and it is the trigger condition for building the fallback (§6).
- **D-2 Corner-cut reward incentive.** Real if the monolith reward stays progress-to-gate-centre. This is
  exactly what graft S-2 (arc-length progress over a contact-safe line) neutralizes — so the hybrid monolith
  closes D-2 by construction. A pure monolith does not; the hybrid cast is therefore *strictly better* than a
  naive monolith on this axis, which is why I recommend the hybrid cast, not a bare retrain.
- **D-3 Narrow convergent basin (2/3 seeds).** Real training-process risk; budget more seeds for inc8 (already
  in memory). A deploy-time non-issue.
- **D-4 Fresh-respawn post-gate-3 sensitivity (bimodal laps).** Real, but the WINNER-VALIDATION RIDER caps ALL
  four options equally — and the monolith is the ONLY one that has already PASSED a live fresh batch (5/5).

The skeptic's strongest *honest* point is D-1+D-2 together: if you relax the cone hard for speed, the SOFT
monolith could start cutting corners at the binding gate-4, and that is exactly where decomposition's hard
guarantee would have saved you. My answer is the staged plan: the hybrid cast (graft S-2) closes D-2 now,
and the warm decomposed fallback (§6) closes D-1 *if and when it actually regresses* — measured, not feared.

---

## 5. The hard floor that caps EVERY architecture (and demotes the whole S2 question)

Both judge lenses converge on a number that should reframe Fengyou's priorities: **perception East σ ≈ 0.47 m
= 3.0× the 0.155 m gate-4 binding margin, UNFILTERED.** Every world-frame line-tracking approach — decomposed,
hybrid, AND monolithic — consumes the same world fix through the same KF. Decomposition buys **NOTHING** on
perception (a key over-claim to neutralize: a "cleaner line" does not clean the estimator). The KF must reach
<0.05 m 1-σ at the post-gate-3 37 m/s window or NO approach is valid at VQ2 race speed. Latency (67 ms) is
validity-BENIGN: 0.77–1.93 mm cross-track at ≤37.7 m/s, <1.3% of the margin — fold into k, do not chase.

**Implication:** the binding VQ2 validity risk is the ESTIMATOR, not the planner. Robustness investment should
prioritize the KF over the S2 architecture choice regardless of which architecture wins. This is the single
most important cross-cutting finding, and it is architecture-independent.

---

## 6. Recommendation, integration plan, and the fallback trigger

**Primary (no Phase-C blocker): STAGED monolithic-then-conditional-decompose.**

1. **Retrain the monolith for inc8 on the corrected-aero plant**, cone relaxed ONE ladder step (rw_tilt
   96→48, free-cone held at 60° initially), grafting the **arc-length progress reward over a rebuilt
   contact-free min-snap line** (graft S-2; the hybrid / MPCC-cast). Budget MORE seeds (≥4) for the narrow
   basin. This closes D-2 by construction and keeps the learn-the-line ceiling.
2. **Build the min-snap + coupled-aero TOPP line generator** (graft S-3; pure numpy, deterministic, <1 s).
   Emit a corrected-aero feasible, contact-free, NON-saturated line in the `peregrine.reference_line.v1`
   schema, crossings dead-centre (full 0.37 m contact-true band free at r=0.38), planned at DR-conservative
   envelope params (a_up_max ~70 not 78.3, pooled c2). This is architecture-neutral: it feeds the hybrid
   monolith's reward now AND the decomposed fallback's geometry rung later. Keep it as the legal per-track
   flywheel.
3. **Run the scipy-SLSQP toy MPCC tracker on `rl_plant.step` as the realizability PROBE.** This measures the
   real k on the corrected plant for a constraint-aware tracker (currently one bad datum, k=1.85). It is the
   single most decision-relevant native run available — it tells you whether the decomposed fallback's ~5.5 s
   ceiling is real BEFORE you pay any acados/WSL/Adroit cost.
4. **Crown the inc8 winner only after a ≥3–5-lap fresh-reset live batch** (WINNER-VALIDATION RIDER; the offline
   twin is blind to fresh-respawn post-gate-3 sensitivity).
5. **Validate every speed/margin win against the gate-4 contact-true metric** (`rl/contact_true_eval.py`,
   margin = (0.75−r)−L∞) before shipping — gate-4 binds at +0.155 m flown / +0.046 m planning; pushing corner
   speed trades against the 0.75 m aperture.

**Fallback (build ONLY on a measured trigger): fully DECOMPOSED stack** = corrected-aero min-snap line + MPCC
(or RL) tracker. **Trigger conditions (BOTH required):**
- (a) the toy-MPCC probe (step 3) shows a constraint-aware tracker recovers ≥~80% of envelope speed (k ≲ ~1.25)
  — i.e. the decomposed ceiling is real, not a paper number; AND
- (b) the relaxed-envelope hybrid monolith actually starts cutting corners at gate-4 (the SOFT-guarantee
  regression D-1 the fallback exists to catch), observed in the contact-true metric or a live batch.
Do NOT build the acados/WSL production tracker or a fresh RL-tracker Adroit campaign speculatively — both
trigger conditions must fire first.

---

## 7. Confidence and decision-dependencies

**Confidence: HIGH** on the recommendation structure (monolith-primary, hybrid-cast, decomposed-as-triggered-
fallback), because every number that would flip it has been independently reproduced:
- The speed-prize-evaporates-at-60° finding (V2, 10.6 s honest bound) is the linchpin and is reproduced by my
  own clean-room TOPP, not inherited.
- The tracker-gap (V3, k=1.85, 3/3 native trackers fail) is the decomposition-killer and is in memory + four
  independent failed prototypes.
- The shipped-line-reuse-is-fiction (V4) is measured directly off the JSON.

**MEDIUM** confidence only on the *magnitudes* that don't change the ranking: the exact honest 60° bound
(10.6 s is point-mass, instant-attitude, ~0-airspeed convex-map over-fit ⇒ a touch optimistic; true ≥10.6 s,
which only strengthens the conclusion); the inc5-vs-inc7 lineage style-tax offset (2.63 vs 2.87 s); the
gate-4 absolute margin (registration-uncertain pending SHADOWPC-VISION-CAL, but the binding-gate RANKING is
robust).

**Decision-dependencies (what live data would change the call):**
1. **The toy-MPCC k on the corrected plant** — the one unmeasured number the whole decomposed-fallback decision
   hinges on. If k ≲ 1.25 for a constraint-aware tracker, the fallback's ceiling is real and worth pre-building.
2. **inc7-with-envelope-OFF retrain** — resolves whether the clean inc7 style tax is 2.63 or ~2.87 s, and
   whether the relaxed-cone monolith starts cutting corners (the D-1 fallback trigger).
3. **inc7 per-tick gate-transition ticks from ShadowPC** — confirms the post-gate-3 localization the waterfall
   leans on (currently PROXY-reconstructed).
4. **KF 1-σ at the post-gate-3 37 m/s window** — the architecture-independent hard floor; <0.05 m or no
   approach is valid at race speed. This dominates the S2 choice and should be measured first.

---

## 8. The honest bottom line for Fengyou

I was charged to argue FOR decomposition, and the strongest honest version of that case is: *graft its two
real wins (S-2 incentive-level contact-free reward, S-3 offline flywheel) onto the live monolith, and hold its
one unique win (S-1 hard inversion ban) as a triggered fallback.* Decomposition does NOT win the primary seat,
because the two things it would need to beat the monolith both fail verification: at the legal 60° cone the
honest point-mass bound (~10.6 s) is at/above where inc7 already flies, so there is no speed prize to sell; and
its enabling assumption — a good tracker — is the largest unproven risk in the project (k=1.85 measured, 3/3
native trackers failed). The fastest realized-NOW path is "relax the cone + retrain the monolith," and the
binding VQ2 validity risk is the estimator (East σ 0.47 m = 3× the gate-4 margin), which no planner
architecture fixes. Build the line generator and the toy-MPCC probe — they are cheap, native, decision-
relevant, and serve both paths — but do not switch the live monolith off to chase a paper ceiling gated on an
unbuilt tracker.

---

### Appendix — reproduction
```
PYTHONPATH=src .venv/Scripts/python.exe handoff/ultracode-planning-togt-s2-2026-06-13/artifacts/verify_honest_tilt_topp.py
# line len 164.65 m, kappa_max 0.0334 (r_min 30.0 m)
# A_UP_MAX 78.28 m/s2 (7.98 g), drag wall ~38.6 m/s
#  cap   A:cornering-only   B:honest-total-tilt
#   60     5.53s v36.0       10.62s v18.7
#   65     5.22s v36.9        9.57s v21.1
#   75     4.75s v38.1        7.27s v29.4
#   80     4.63s v38.5        5.91s v36.7
#   90     4.59s v38.8        4.57s v39.2
```
Constants live from `src/racer/rl_plant.py`: `COLL_MAP_ACCEL_MEASURED[-1]=78.2828`, `QUAD_DRAG_C2_POOLED=0.052`,
`g=9.80665`. Shipped line (`rl/reference_line_vq1.json`): max R33-tilt 170.4°, 25.8% inverted, peak 51.1 m/s.
k=1.85 / 8.3 s structural datum: `memory/project_phase2_rl_vision_decisions.md` §TOGT-BOUND (lines 323–326).
