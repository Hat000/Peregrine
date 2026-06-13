# Phase D — S2 ARCHITECTURE ADJUDICATION (lens: SKEPTIC-OF-DECOMPOSITION)

**Author:** Phase-D adjudicator (opus-4.8), ultracode S2-planning fan-out · **Date:** 2026-06-13 · **For:** Fengyou
**Scope:** analysis + reasoning only. No tracked source edited, no live sim, no SLURM, no network. All numbers taken
from the frozen Phase-A/B/C artifacts in this directory (each re-derived natively from repo data by its author) and
two source-fact checks I ran (`reference_line.py::progress`, `peregrine_racing.py` reward terms).

**Lens mandate:** stress-test the case FOR monolithic. inc7 monolithic is ALREADY live-confirmed working (5/5
standing, gate-3 barrier gone, ~11.45 s fresh). The backflip-dive root cause is fixed. The offline line-iteration
flywheel can also be applied to a monolithic policy. Argue whether decomposition's structural-inversion-impossibility
is worth its lower ceiling + the REQUIREMENT of a good tracker — but be honest if the evidence favors decomposition
or hybrid.

---

## 0. Verdict in one paragraph

**Recommendation: STAGED — retrain the live MONOLITHIC policy on the corrected-aero plant for inc8 with the style
cone opened one ladder rung, cast as a HYBRID-monolithic (arc-length progress reward over an offline contact-safe
reference line); keep the DECOMPOSED stack (offline line generator + a real MPCC/RL tracker) as an explicitly-built
*warm fallback* triggered only if the relaxed-envelope monolith starts cutting corners.** The skeptic lens is upheld
by the evidence, not merely asserted: (1) the monolith is the ONLY architecture with a measured, live-confirmed,
zero-contact realizable lap on the corrected plant — every decomposition number is a planning bound times an unbuilt
tracker; (2) three of three native trackers FAILED and the only measured tracker datum is k=1.85 (the geometric
tracker needs 8.3 s to fly the 4.55 s line — *slower than the style-ON monolith's 9.76 s*), so "a good tracker IS the
hard part" is empirically true, not rhetorical; (3) the decomposition's one unique structural win — a HARD inversion
ban — addresses the inc1 backflip whose root cause (wrong plant + no DR + reward/termination conflict) is understood
and fixed, and which inc7 does not exhibit; (4) the corrected-aero realizability verifier showed the speed gap is a
TILT/cone story, not a planner story (at the doctrine 60° cone EVERY approach lands ~9.6 s ≈ inc7's 9.76 s), and the
cone-relaxation lever speeds the monolith too — so decomposition does not own the prize it appears to offer. The
hybrid-monolithic graft imports decomposition's ONE genuine reward-level advantage (no corner-cut incentive) at
near-zero architectural cost. I label the recommendation **hybrid** in the structured object because the concrete
inc8 build IS the hybrid-monolithic (3rd option), and **staged_monolithic_then_decomposed** is the program shape.

---

## 1. The gap waterfall, re-read through the skeptic lens (where is the recoverable time?)

The Phase-A waterfall is a clean three-rung ledger summing 35.30 → 4.72 s (Σ = 30.58). But Phase-C's
envelope-realizability verifier (`C_verify_envelope_realizability.md`) materially **re-attributes rung 2 vs rung 3**,
and that re-attribution is the single most decision-relevant correction for S2. I carry the corrected reading:

| rung | from→to | Δ | who closes it | skeptic reading |
|---|---|---|---|---|
| 1 | 35.30 → 9.76 | −25.54 s | **ALREADY CLOSED** (inc7 live) | Pathology removal (k=1.85 dilation, alt-relay, start-transient, descent-caution). RL lands +1.46 s ABOVE the 8.3 s best-dilated-geometric floor → this is NOT raw speed, it is the monolith *replacing* the geometric stack. **This rung is the strongest single fact against decomposition-with-the-existing-tracker:** the only tracker we have, dilated, is *slower* than the live monolith. |
| 2 | 9.76 → ~6.9 | −2.6 to −2.9 s | **envelope/cone relaxation** | The style-cap tax. **CRITICAL CORRECTION:** Phase-C proved this is overwhelmingly KINEMATIC, not reward-shaping. Honest tilt-capped TOPP (cap on the TOTAL specific-force direction) gives 60° ≈ 9.8 s ≈ inc5-measured rw_tilt=96 9.52 s. So the 2.6 s is unlocked by OPENING THE CONE (free-cone 60°→75–80°), which is a reward/doctrine lever that **speeds the monolith too**. Decomposition does not manufacture it. |
| 3 | ~6.9 → ~4.7 | −2.2 s | better planner + **a good tracker** | Planning/policy under-driving + point-mass idealization. ~1.5–1.9 s is pure policy-vs-point-mass slack; the rest is point-mass fiction. This is the ONLY rung where a planner/tracker architecture could in principle out-perform a retrained monolith — and it is exactly the regime gated on the unmeasured tracker k. |

**Skeptic conclusion from the waterfall:** the live prize (rungs 2+3 ≈ 5 s) is dominated by rung 2 (cone relaxation,
a monolith-friendly reward lever) and a rung-3 fraction that REQUIRES the one thing the whole field is blocked on (a
good tracker). Decomposition's claim to rung 3 is conditional on building the tracker that does not exist; the
monolith already owns rungs 1 and most of 2 and attacks rung 3 with the same cone lever. **The waterfall does not
contain a "decomposition-only" recoverable block.**

### 1.1 The double-count that kills decomposition-with-naive-tracker (carry forward)
Phase-A's 3.87 s "structural tracker gap" = (k=1.85 − 1) × 4.551 s is deliberately NOT a waterfall rung — it is the
decomposed-geometric-tracker tax that the monolith never pays (the monolith's tracking work is already inside rungs
1+3). It lives as the **S2 adjudication number**: 8.3 s to track the shipped line is slower than even the style-ON
inc7 (9.76 s). Decomposition with the existing tracker is strictly dominated. Decomposition only becomes interesting
with a NEW, BETTER, UNBUILT tracker (MPCC 88% speed-keep → ~5.5 s) — i.e. the entire decomposition case rests on a
single unmeasured number.

---

## 2. The achievable-time evidence (prototypes), honestly de-rated

The prototypes give point-mass PLANNING bounds, not flight times. After the Phase-C corrections:

| quantity | honest figure | source / status |
|---|---|---|
| corrected-aero UNCONSTRAINED (90°) ceiling | **4.57–4.71 s** | C1 4.574 ↔ C++ TOGT 4.714 (3%); v_max 39.2≈39.26. UPHELD, robust. |
| style-respecting **60°** planning bound | **~9.8 s** (NOT 5.35) | C-envelope: C1 mis-applied the tilt cap to cornering only; honest cap on total specific force → 9.8 s ≈ inc5 9.52 ≈ inc7 9.76. **The single biggest correction.** |
| 75° / 80° planning bound | ~6.7 / ~5.4 s | honest tilt-capped TOPP. To go fast you must OPEN THE CONE. |
| corrected-aero CONTACT-VALID @ r=0.38 (margined TOGT line) | **~4.43 s** (bound_free; gate-4 +0.046 m) | C-contact: ref_margin is INVALID at r=0.38; only bound_free is valid, razor-thin at the binding gate. |
| shipped 4.55 s line | drag-INFEASIBLE (plans 46–55 m/s; wall 37–39 m/s) | C2 + both verifiers. **Geometry must be REBUILT on corrected aero** — "reuse the shipped line" is fiction. |
| only MEASURED realizable lap on corrected plant | inc5 unconstrained 6.9 s twin; inc7 9.76 s offline / 11.45 s live | the empirical anchor every approach must beat. |

**Three of three native trackers FAILED:** minsnap geometric → None/inf (all six gate misses inf); hybrid DF+PD →
5–35 m miss, 142 m divergence at high gain; mpcc toy SLSQP → ~18–21 m/s vs the 38 m/s wall. The ONLY measured tracker
datum is k=1.85 (bad, structural — 54-combo sweep found nothing faster). **"A good tracker is required, and a good
tracker is the hard part" is therefore an empirical finding, not a prior.**

---

## 3. The case FOR monolithic (stress-tested, upheld)

1. **DEMONSTRATED > ASSERTED validity.** inc7: standing 5/5 FINISHED, gate-3 barrier GONE (0 collisions offline AND
   0/5 live), command saturation NONE (|tanh|≤0.547), graceful AND observed failure (slows/widens, never tumbles),
   7 live zero-contact sessions. Every rival's contact-free claim is plan-level or incentive-level. On a validity
   lens this is decisive (robustness judge: monolith 7.5 vs hybrid 6.0 vs minsnap 6.5 vs TOGT 4.5).

2. **Fastest realized-NOW.** inc5 unconstrained monolith = 6.9 s twin MEASURED, already beats minsnap's claimed 7.0 s
   (honest band 6.5–10 s), hybrid's 8.0 s, and ties the envelope TOGT/MPCC need a from-scratch unbuilt tracker to
   reach. The time judge's own verdict: "none of the four trajectory-opt approaches is clearly faster, as a flyable
   line at the doctrine-allowed envelope, than retraining the already-live monolith."

3. **The decomposition's reason-to-exist is neutralized.** inc1 backflip root cause (wrong plant + no DR +
   reward/termination conflict) is understood and FIXED; inc7 exhibits ZERO inversion. The HARD-vs-SOFT
   inversion-ban advantage addresses a pathology that no longer occurs. The shipped 4.55 s line's 170° inversion /
   26% inverted samples is a property of the FALSIFIED-linear-plant TOGT optimizer — and a gentle min-snap line
   through the gate centres on the CORRECTED plant has geometry-forced tilt p50 54–60°, max ≤79°, ZERO inversion
   (robustness judge §1.2). So neither the monolith NOR a corrected-plant decomposed line wants to invert. The
   "structural inversion impossibility" is solving a problem the corrected plant already doesn't have.

4. **Lowest integration cost of the viable options:** one subsystem, already live, retrain-only. No acados (does not
   build on Windows/py3.13, `io.h`), no WSL, no fresh line-relative obs layout, no three-subsystem bring-up.

5. **The cone lever is monolith-friendly.** The biggest recoverable block (rung 2, ~2.6 s) is unlocked by relaxing
   R4's free cone — a reward change to the EXISTING monolith (`peregrine_racing.py` `tilt_free_rad` / `rw_tilt`),
   already MEMORY's next-queued action (envelope ladder step 1), now unblocked by inc7 live-confirm.

---

## 4. The honest case where decomposition / hybrid EARNS its keep (not dismissed)

The skeptic lens does not mean "monolith wins on every axis." Three decomposition properties are genuine:

1. **Incentive-level contact-free guarantee (the ONE real reward-level win).** A pure progress-to-gate-CENTER reward
   (which is exactly what `peregrine_racing.py` R1 currently is: `rw_progress * (d2g_prev − d2g_curr)`) can in
   principle reward a policy for cutting *inside* the frame to shorten distance-to-center — corner-cutting is a
   rules-INVALID run. An **arc-length progress reward over a planned contact-safe line** removes that incentive: the
   tracker is rewarded for advancing ALONG Γ, never for shortening the gap to a center it could clip. This is the
   genuine decomposition win, and it grafts onto the monolith as a reward change, NOT a new subsystem (§6).

2. **HARD inversion ban under aggressive cone relaxation.** inc7's guarantee is SOFT (R4 hinge). If the cone is
   opened hard for rung-3 speed, the SOFT guarantee weakens exactly as the attitude envelope widens — this is the
   ONE regime where decomposition's planner-clamp earns its keep. It is a fallback trigger, not a default.

3. **Deterministic offline line-iteration flywheel.** Per-track the plan-line is a frozen, auditable feedforward —
   the cleanest determinism/audit story. BUT this is NOT exclusive to decomposition: a min-snap/TOPP line generator
   feeds the monolith's hybrid progress reward AND the fallback's geometry rung. The flywheel is free to keep warm
   regardless of architecture.

**Where decomposition does NOT help (over-claims neutralized):**
- **Perception buys NOTHING.** A world-frame line couples perception East σ 0.47 m DIRECTLY to cross-track = 3.0×
  the 0.155 m gate-4 binding margin. The monolith consumes the SAME world fix through the same KF. The binding VQ2
  validity risk is the ESTIMATOR (KF to <0.05 m 1-σ at the post-gate-3 37 m/s window), not the planner — and no
  S2 architecture fixes it. This is the hard floor that caps all four equally.
- **Shipped-line reuse is FICTION** (drag-infeasible, 170° inversion). Decomposition does NOT get a free audited
  geometry asset; it inherits a min-snap planner build PLUS a tracker campaign.
- **Ceiling is LOWER and probably slower than the live monolith** unless the cone is relaxed AND a good unbuilt
  tracker materializes.

---

## 5. Failure modes (BOTH architectures, honest)

### 5.1 MONOLITHIC (incl. hybrid-monolithic graft) — failure modes
- **SOFT inversion guarantee regresses under aggressive cone relaxation.** Opening free-cone past ~75–80° for
  rung-3 speed could re-admit the inc1 pathology. *Mitigation:* relax one ladder rung at a time; keep R4 hinge
  intact; gate on contact-true gate-4 metric (0.155 m @ r=0.38) before shipping.
- **Corner-cut incentive in the legacy R1 (progress-to-gate-CENTER) reward.** A speed-pushed policy could clip the
  frame to shorten distance-to-center → INVALID. *Mitigation:* the hybrid-monolithic graft (arc-length progress
  over Γ) directly removes this — the highest-value single change.
- **Narrow convergent basin (2/3 seeds viable).** inc8-class retrains need MORE seeds budgeted (already in MEMORY).
- **Fresh-respawn post-gate-3 sensitivity (bimodal laps).** The offline twin is BLIND to fresh-respawn post-gate-3
  state → WINNER-VALIDATION RIDER: crown only after ≥3–5-lap fresh-reset live batch. (Caps all four equally;
  monolith is the ONLY one that has already PASSED a live fresh batch.)
- **Gate-4 margin erosion at speed.** Pushing corner speed (rung 3) trades against the 0.75 m validity aperture;
  gate-4 simstart margin is only 0.155 m @ r=0.38. *Mitigation:* re-verify any speed win against the contact-true
  metric; the binding gate is the inc8 risk zone.

### 5.2 DECOMPOSED (offline line + tracker) — failure modes
- **The tracker does not materialize at the assumed quality.** The ENTIRE case rests on a good tracker recovering
  85–90% of point-mass speed (→~5.5 s). Three native trackers failed; the only measured datum is k=1.85 (→~8.3 s,
  slower than the live monolith). If the real tracker limps to k≈1.7+, decomposition is a slower, costlier monolith.
- **Tracker lags into the BINDING gate at the worst place.** A tracker needing k≈1.7 is by definition lagging; lag
  at the post-gate-3 37 m/s approach erodes cross-track at gate-4 (0.155 m budget) — a quiet margin-erosion failure
  landing exactly at the tightest gate. Graceful in attitude, fragile in cross-track at the worst location.
- **Geometry must be REBUILT on corrected aero** (shipped line drag-infeasible) — so the "static shipped asset"
  determinism win is not real until a new line + tracker are built.
- **acados/MPCC does not build on Windows; RL-tracker needs a fresh line-relative obs layout + new Adroit campaign.**
  Monolith checkpoints NOT reusable. HIGH integration cost, three-subsystem bring-up, before any time is realized.
- **No perception benefit** — same world-fix/KF dependence as the monolith.
- **TOGT-collocation standalone is catastrophic:** its only lines are falsified (170° inversion, 51–60 m/s, and its
  "feasible" refined line demands 17 rad/s body rates the ~11.2 rad/s super-rate plant CANNOT produce). Role =
  offline BOUND + geometry SEED only, never deployable.

---

## 6. Integration plan (the recommended STAGED / hybrid-monolithic path)

Concrete, ordered. Source seams confirmed: `rl/peregrine_racing.py` R1 = `rw_progress*(d2g_prev−d2g_curr)`
(distance-to-gate-center — the corner-cut-prone term to REPLACE); R4 tilt hinge `tilt_free_rad`(60°)/`rw_tilt`(4.0)
(the cone lever); `rl/reference_line.py::progress(position_ned)` already exists as the arc-length seam;
`rl/contact_true_eval.py` is the canonical gate-4 selection metric.

1. **[native, cheap, no blocker] Build the corrected-aero line generator.** min-snap geometry through gate
   waypoints (entry/exit along each gate normal) + coupled-aero forward/backward TOPP, planned at DR-CONSERVATIVE
   envelope params (a_up_max ~70 not 78.3; pooled c2=0.052), crossings DEAD-CENTER (full 0.37 m contact-true band
   free at r=0.38), emitted in the `peregrine.reference_line.v1` schema. `speed_profile.py` is half-built; the
   coupled-thrust envelope (~40 lines) is already prototyped in `minsnap_topp_proto.py`; min-snap QP = one
   `scipy.linalg.solve_banded`. This feeds BOTH the monolith's progress reward AND the fallback's geometry rung;
   it is the LEGAL per-track flywheel. (Phase-C native prototype #1.)

2. **[reward change, not a new subsystem] Graft the hybrid-monolithic arc-length progress reward.** Replace R1's
   progress-to-gate-CENTER with progress-ALONG-Γ via `reference_line.progress()` over the line from step 1. Keep the
   learn-the-line ceiling; remove the corner-cut incentive (imports decomposition's one genuine reward-level win at
   near-zero architectural cost). This is the highest-value graft.

3. **[doctrine lever] Open the style cone one ladder rung** (rw_tilt 96→48 first; free-cone 60°→~70° next if the
   metric holds), gated on LAPTOP-INC8-BINDING-GATE-VERIFY (gates 4,5 probe at r=0.38). The honest tilt tax is
   KINEMATIC, so this is the load-bearing speed lever — and it speeds the monolith directly.

4. **[retrain] inc8 on the corrected-aero plant** (convex collective map + quad drag, LAPSE OFF), with steps 2–3,
   MORE seeds budgeted (narrow basin). Validate map-ON (peregrine_eval/offline_rollout default to legacy flat plant —
   footgun). Crown the winner ONLY after a ≥3–5-lap fresh-reset live batch (WINNER-VALIDATION RIDER).

5. **[realizability probe, native, decision-relevant] Run the scipy-SLSQP toy MPCC tracker on `rl_plant.step`.** This
   MEASURES the real k on the corrected plant (currently one bad datum, k=1.85). It adjudicates whether the decomposed
   fallback's ~5.5 s ceiling is real BEFORE paying any acados/Adroit cost. (Phase-C native prototype #2.)

6. **[fallback, do NOT build speculatively] Build the production decomposed stack (line + acados-MPCC or RL-tracker +
   estimator) ONLY IF** (a) prototype #2 shows a tracker recovers ≥~80% of envelope speed AND (b) the relaxed-cone
   monolith actually starts cutting corners (the SOFT-guarantee failure the fallback exists to catch). Gate, do not
   pre-build.

7. **[cross-cutting, prioritize over any planner choice] Estimator work.** Perception East σ 0.47 m = 3.0× the gate-4
   margin; the KF must reach <0.05 m 1-σ at the post-gate-3 37 m/s window or NO approach is valid at race speed.
   Latency 67 ms is validity-benign (fold into k, do not chase). This is the binding VQ2 validity risk regardless of
   S2 choice.

---

## 7. Decision dependencies (what live data / organizer answers would change the call)

1. **The measured tracker k on the corrected plant** (Phase-C prototype #2). If a good native MPCC recovers ≥85% of
   point-mass speed (k≈1.2–1.3 → ~5.5 s), the decomposition ceiling becomes genuinely attractive and the call could
   shift toward building the decomposed stack sooner. If it limps to k≈1.7+, decomposition is dominated and the
   recommendation hardens to monolith-only-plus-graft.
2. **Whether the relaxed-cone monolith cuts corners** (inc8 fresh-reset live batch, contact-true gate-4 metric). If
   the SOFT tilt/corner guarantee holds under relaxation, the fallback never triggers. If it regresses, build the
   decomposed fallback (its HARD ban earns its keep).
3. **inc7 per-tick gate-transition ticks from ShadowPC** (debug_obs, gitignored) to confirm the G2→G3 +1.18 s
   localization and de-risk the bimodal post-gate-3 split — tightens the rung-2/rung-3 split.
4. **VQ2 estimator achievable 1-σ at 37 m/s** — if the KF cannot reach <0.05 m at the post-gate-3 window, the
   race-speed validity ceiling binds BEFORE any planner choice matters; the practical target time rises for ALL
   architectures and the estimator becomes the program's critical path.
5. **Organizer answers on stack-carry-state between attempts and per-track determinism scope** — confirms the
   offline line-iteration flywheel is legal as assumed (it is, per current reading) and that per-track frozen lines
   are an acceptable determinism contract.
6. **A convex+anisotropic TOGT re-run** (climb drag 0.076 on −z loaded by the 26 m descent) would tighten the
   ~4.6–4.9 s ceiling band; second-order, would not move the S2 verdict but sharpens the rung-3 target.

---

## 8. Bottom line for Fengyou

The skeptic-of-decomposition lens holds up under the evidence. The monolith is the only architecture with a measured,
live, zero-contact realizable lap; the decomposition's entire speed case rests on one unmeasured number (the tracker
k) against one bad measured datum (k=1.85, slower than the live monolith); its unique structural win (HARD inversion
ban) solves a pathology the corrected plant already doesn't exhibit; and the biggest recoverable time block (the cone)
is a monolith-friendly reward lever, not a planner property. The right move is to retrain the monolith on corrected
aero with the cone opened one rung and the hybrid-monolithic arc-length progress reward grafted (importing
decomposition's one genuine reward-level advantage at near-zero cost), while keeping the decomposed stack as an
explicitly-gated warm fallback. Do NOT build the acados/Adroit production tracker speculatively — gate it on the
native MPCC k-measurement and on the monolith actually starting to cut corners. And regardless of S2 choice, the
binding VQ2 validity risk is the estimator (perception East σ = 3× the gate-4 margin), not the planner — prioritize
the KF.
