# Phase D — S2 FINAL SYNTHESIS + COMPLETENESS CRITIC

**Author:** Phase-D final synthesizer (opus-4.8), ultracode S2 fan-out · **Date:** 2026-06-13 · **For:** Fengyou
**Scope:** analysis + offline reconciliation only. No tracked source edited; no live sim, SLURM, or network.
All load-bearing numbers re-derived natively or re-run from the frozen Phase-A/B/C/D artifacts in this directory.

---

## 0. The one-paragraph verdict

**Retrain the live MONOLITHIC policy on the corrected-aero plant for inc8, cast as a HYBRID-monolithic
(arc-length progress reward over a rebuilt, offline, contact-safe min-snap reference line), with the style
cone opened one ladder rung; keep the fully DECOMPOSED stack (offline line generator + a real MPCC/RL
tracker) as an explicitly-built but UNBUILT-until-triggered warm fallback.** Program shape =
`staged_monolithic_then_decomposed`; the concrete inc8 build IS the hybrid (3rd option). The two
adjudication lenses — one charged to argue FOR monolithic, one charged to argue FOR decomposition — were
run independently and **converge on the identical recommendation at HIGH confidence**. They do not diverge;
they agree on structure and disagree only on which UNBUILT thing to keep warm, and both subordinate that to
the live monolith. The decisive evidence is the Phase-C envelope-realizability correction (independently
reproduced by me and by both Phase-D adjudicators): at the doctrine-legal 60° cone the honest point-mass
bound is ~10.6 s — AT or ABOVE inc7's already-live 9.76 s twin — so the recoverable time lives in the
ENVELOPE (a reward/doctrine cone-relaxation lever that speeds the monolith equally), NOT in
planning/tracking (the architecture-relevant axis). Decomposition therefore owns no speed prize at the legal
envelope; its one unique win (a HARD inversion/corner-cut ban) guards an already-fixed pathology and is best
held as triggered insurance; and its two real wins (no corner-cut incentive, offline flywheel) graft onto
the monolith for near-zero cost.

---

## 1. Reconciliation of the two lenses — do they agree?

| axis | SKEPTIC-OF-DECOMPOSITION (lens A) | SKEPTIC-OF-MONOLITHIC (lens B) | reconciled |
|---|---|---|---|
| `recommendation` | staged_monolithic_then_decomposed | staged_monolithic_then_decomposed | **CONSENSUS** |
| `confidence` | high | high | **high** |
| concrete inc8 build | hybrid-monolithic (graft arc-length progress) | hybrid-monolithic (graft arc-length progress) | **CONSENSUS** |
| decomposition's status | warm fallback, gated, do not pre-build | warm fallback, gated, do not pre-build | **CONSENSUS** |
| where is the recoverable time | rung-2 cone (envelope, monolith-friendly) + rung-3 (gated on tracker) | rung-2 cone (envelope) + rung-3 (gated on tracker) | **CONSENSUS** |
| decomposition's unique win | HARD inversion ban — fixed pathology, conditional insurance | HARD inversion ban — fixed pathology, conditional insurance | **CONSENSUS** |
| graftable wins | S-2 no-corner-cut-incentive; S-3 offline flywheel | S-2 no-corner-cut-incentive; S-3 offline flywheel | **CONSENSUS** |
| binding VQ2 risk | ESTIMATOR (East σ 0.47 m = 3× gate-4 margin), not planner | ESTIMATOR (East σ 0.47 m = 3× gate-4 margin), not planner | **CONSENSUS** |
| the one unmeasured number | toy-MPCC k on corrected plant for a *good* tracker | toy-MPCC k on corrected plant for a *good* tracker | **CONSENSUS** |

**The lenses agree on every structural point.** This is a strong signal: two adversarially-charged
reviewers, each trying to break the other's preferred architecture, both landed on the same staged plan
because the underlying evidence forced it. The only nominal difference is the `recommendation` enum label —
lens A noted it would also accept `hybrid` to name the concrete build — but both explicitly call the program
shape `staged_monolithic_then_decomposed`. I carry that enum.

---

## 2. The adjudicating evidence: where is the recoverable time? (gap waterfall, corrected)

The task's adjudication rule: *if the recoverable time is mostly in the ENVELOPE (a reward knob,
architecture-agnostic) the call is monolithic; if it is mostly in PLANNING/TRACKING (architecture-relevant)
the call leans decomposed.* The corrected three-rung waterfall settles this decisively.

| rung | from→to | Δ | closes it | ENVELOPE or PLANNING/TRACKING? |
|---|---|---|---|---|
| 1 | 35.30 → 9.76 | −25.54 s | ALREADY CLOSED (inc7 live) | Pathology removal (k=1.85 dilation, alt-relay, start-transient, descent-caution). RL lands +1.46 s ABOVE the 8.3 s best-dilated-geometric floor — NOT raw speed; the monolith *replacing* the geometric stack. |
| 2 | 9.76 → ~6.9 | −2.6 to −2.9 s | **cone relaxation** | **ENVELOPE.** Phase-C proved this is overwhelmingly KINEMATIC: honest tilt-capped TOPP at 60° ≈ 9.8–10.6 s ≈ inc5-measured rw_tilt=96 9.52 s. Unlocked by OPENING the free-cone (60°→75–80°), a reward/doctrine lever that speeds the monolith too. |
| 3 | ~6.9 → ~4.7 | −2.2 s | better planner + a good tracker | PLANNING/TRACKING — but ~1.5–1.9 s is pure policy-vs-point-mass slack and the rest is point-mass fiction. The ONLY rung where decomposition *could* out-perform a retrained monolith — and it is gated on the unmeasured tracker k. |

**Verdict on the adjudication rule:** the largest recoverable, *measured*, end-to-end block (rung 2, ~2.6 s)
is in the ENVELOPE and is architecture-agnostic. Rung 3 is the only architecture-relevant block, it is
smaller, half of it is policy slack a retrained monolith also captures, and its decomposition-favoring
fraction is **conditional on a tracker that does not exist** (k=1.85 measured; 3/3 native trackers failed).
**The recoverable time is mostly in the envelope ⇒ the call is monolithic-first.** The waterfall contains no
"decomposition-only" recoverable block.

### 2.1 The double-count that kills decomposition-with-the-existing-tracker
Phase-A's 3.87 s "structural tracker gap" = (k=1.85 − 1) × 4.551 s is **not** a waterfall rung — it is the
decomposed-geometric-tracker tax the monolith never pays. It lives as the S2 adjudication number: 8.3 s to
track the shipped line is *slower than even the style-ON inc7 (9.76 s)*. Decomposition with the existing
tracker is strictly dominated. Decomposition only becomes interesting with a NEW, BETTER, UNBUILT tracker —
the entire case rests on one unmeasured number.

---

## 3. The linchpin, independently reproduced (`verify_honest_tilt_topp.py`)

I re-ran the honest tilt-cap TOPP from scratch. Reproduced bit-for-bit:
```
 cap   A:cornering-only   B:honest-total-tilt
  60     5.53s v36.0       10.62s v18.7
  65     5.22s v36.9        9.57s v21.1
  75     4.75s v38.1        7.27s v29.4
  80     4.63s v38.5        5.91s v36.7
  90     4.59s v38.8        4.57s v39.2   <- converge (no cap to mis-apply)
```
**Why reading B is correct:** body tilt is the angle of the TOTAL specific force `f = a_des − grav`, not the
lateral component alone. Braking from ~38 m/s into a κ-limited corner tilts the body just as cornering does.
C1's reading A capped only centripetal accel and left tangential (brake/forward) free to use the full thrust
ball — so its "60° plan" realizes max body tilt 113.8° (inverted), which does not respect a 60° cone. The
two readings converge exactly at 90° (no cap to mis-apply) and diverge monotonically as the cap tightens —
the signature of a mis-placed cap. **The honest 60° bound (10.62 s) ≈ inc5-measured rw_tilt=96 (9.52 s)** —
an independent point-mass optimum corroborating a real RL flight. This three-way agreement (clean-room TOPP
10.6 s ≈ inc5 RL 9.52 s ≈ inc7 9.76 s) is the strongest single piece of evidence in the panel: at the
doctrine cone, EVERY approach lands ~9.5–10.6 s, so lap time is a CONE story, not a planner story.

**Source-confirmed lever:** `rl/peregrine_racing.py` R4 = `rw_tilt(4.0) * relu(cos(tilt_free_rad) − R33)²`,
`tilt_free_rad = 1.0471976 rad = 60°`. The cone-relaxation lever is literally `tilt_free_rad` (60°→75–80°);
`rw_tilt` is only the *weight* on violations beyond the cone. Phase-C's correction implies: **lowering
rw_tilt alone (ladder step 1) cannot buy most of rung 2 — you must OPEN tilt_free_rad.** This refines the
memory's "envelope-ladder step 1" framing: step 1 (rw_tilt 96→48) is necessary-but-not-sufficient; the
load-bearing speed lever is widening the free cone, gated on the gate-4 contact-true metric.

---

## 4. Failure modes — both paths (the honest debits)

### Monolithic (the recommended path)
- **M-1 SOFT inversion guarantee (not HARD).** R4 is a soft reward hinge; under aggressive cone relaxation
  toward 75–80° it could license attitudes drifting toward corner-cutting — the ONE regime where
  decomposition's hard ban earns its keep. Mitigate: relax one rung at a time, keep R4 intact, gate on the
  contact-true gate-4 metric. This is the fallback's trigger condition.
- **M-2 Corner-cut reward incentive.** Legacy R1 is progress-to-gate-CENTER (`rw_progress·(d2g_prev −
  d2g_curr)`) — a speed-pushed policy could clip the frame to shorten distance-to-center → INVALID. Mitigate:
  the hybrid arc-length-progress graft removes this BY CONSTRUCTION (highest-value change). This is why the
  recommendation is the hybrid cast, not a bare retrain — the hybrid is strictly better on this axis.
- **M-3 Narrow convergent basin (2/3 seeds viable).** Budget ≥4 seeds for inc8. Training-process risk only.
- **M-4 Fresh-respawn post-gate-3 sensitivity (bimodal laps).** Offline twin is BLIND to it → crown only
  after a ≥3–5-lap fresh-reset live batch (WINNER-VALIDATION RIDER). Note: the monolith is the ONLY option
  that has already PASSED a live fresh batch (5/5 standing).
- **M-5 Gate-4 margin erosion at speed.** Pushing rung-3 corner speed trades against the 0.155 m simstart
  margin @ r=0.38 (binding gate, post-gate-3 inc8 risk zone). Re-verify every speed win against
  `rl/contact_true_eval.py`.

### Decomposed (the fallback path)
- **D-1 Tracker realizability UNPROVEN (dominant).** Only measured datum k=1.85 (8.3 s, structural, 54-combo
  sweep finds nothing faster); 3/3 native trackers failed (geometric → k=1.85; hybrid DF+PD → 5–35 m miss;
  scipy-SLSQP toy MPCC → ~18–21 m/s vs the 38 m/s wall; C1 feedforward+PD diverges 37.9 m). If real k ≥ 1.7,
  decomposition is a slower, costlier monolith. The ~5.4–5.8 s ceiling is a paper number gated on this.
- **D-2 Tracker lags INTO the binding gate at the worst place.** A tracker needing k~1.7 is by definition
  lagging; lag at the post-gate-3 ~37 m/s approach erodes cross-track at gate-4 (0.155 m budget). Graceful in
  attitude, fragile in cross-track at the tightest gate.
- **D-3 Geometry must be REBUILT.** The shipped `reference_line_vq1.json` is FICTION: max R33-tilt 170.4°,
  25.8% inverted, peak 51.1 m/s vs the 38.6 m/s corrected drag wall. It violates the very cone the decomposed
  line is supposed to guarantee AND is drag-infeasible. The "free shipped audited asset" advantage does not
  exist until a new corrected-aero line is built.
- **D-4 HIGH integration cost.** Three-subsystem build (line + tracker + estimator); acados/MPCC does not
  build on Windows/py3.13 (io.h); an RL-tracker needs a fresh line-relative obs layout + a NEW Adroit campaign
  (monolith checkpoints NOT reusable). Large cost for an unproven ceiling.
- **D-5 No perception benefit.** A world-frame line couples perception East σ 0.47 m DIRECTLY to cross-track
  = 3.0× the 0.155 m gate-4 margin — same world-fix/KF dependence as the monolith. Decomposition buys NOTHING
  on the binding VQ2 validity risk.
- **D-6 Even the contact-valid corrected TOGT line is razor-thin.** bound_free clears gate-4 by only +0.046 m
  planning margin (entirely consumed by tracking error); ref_margin is INVALID at r=0.38 (gate-4 −0.110 m).
  Any decomposed line MUST be rebuilt WITH a gate-4 contact-true guard.

---

## 5. Concrete ordered integration plan

1. **[reward change, not a new subsystem] Graft the hybrid arc-length progress reward.** Replace
   `peregrine_racing.py` R1 (progress-to-gate-CENTER) with progress-ALONG-Γ via `reference_line.progress()`
   over a rebuilt contact-safe line. Closes M-2 by construction; keeps the learn-the-line ceiling. Highest-
   value graft.
2. **[native, no blocker] Build the corrected-aero min-snap + coupled-TOPP line generator.** Pure numpy,
   deterministic, <1 s (`speed_profile.py` half-built; envelope ~40 lines already in `minsnap_topp_proto.py`;
   min-snap QP = one `scipy.linalg.solve_banded`). Emit a corrected-aero feasible, contact-free, NON-saturated
   line in the `peregrine.reference_line.v1` schema, crossings DEAD-CENTER (full 0.37 m contact-true band free
   @ r=0.38), planned at DR-conservative envelope params (a_up_max ~70 not 78.3, pooled c2=0.052). Architecture-
   neutral: feeds the hybrid reward NOW and the decomposed fallback's geometry LATER; serves as the legal
   per-track flywheel (S-3).
3. **[doctrine lever — THE load-bearing speed knob] Open the style cone one rung.** rw_tilt 96→48 FIRST,
   then **free-cone `tilt_free_rad` 60°→~70°** if the gate-4 metric holds (Phase-C: rw_tilt alone is
   insufficient; the cone is the binding kinematic constraint). Gated on LAPTOP-INC8-BINDING-GATE-VERIFY
   (gates 4,5 @ r=0.38). The tilt tax is KINEMATIC, so this is the biggest recoverable block and it speeds the
   monolith directly.
4. **[retrain] inc8 on the corrected-aero plant** (convex collective map + quad drag, **LAPSE OFF**,
   linear_drag=0, super-rate + mixer) with steps 1+3, ≥4 seeds (narrow basin). Validate **map-ON**
   (`peregrine_eval`/`offline_rollout` default to legacy flat plant — footgun). Crown ONLY after a ≥3–5-lap
   fresh-reset live batch (M-4 WINNER-VALIDATION RIDER).
5. **[realizability probe — native, decision-relevant] Run the scipy-SLSQP toy MPCC tracker on
   `rl_plant.step`** to MEASURE the real k on the corrected plant for a constraint-aware tracker (currently one
   bad datum, k=1.85). Adjudicates whether the decomposed fallback's ~5.5 s ceiling is real BEFORE paying any
   acados/WSL/Adroit cost. Single most decision-relevant native run available.
6. **[fallback — build ONLY on BOTH triggers] Production decomposed stack** (corrected-aero line + acados-MPCC
   or RL tracker + estimator). Trigger (a): step-5 probe shows a constraint-aware tracker recovers ≥~80% of
   envelope speed (k ≲ ~1.25); AND (b): the relaxed-cone hybrid monolith actually starts cutting corners at
   gate-4 (the M-1 SOFT-guarantee regression). Gate, do not pre-build.
7. **[cross-cutting — PRIORITIZE OVER ANY PLANNER CHOICE] Estimator work.** Drive the KF to <0.05 m 1-σ at
   the post-gate-3 ~37 m/s window — perception East σ 0.47 m = 3.0× the gate-4 margin; latency 67 ms is
   validity-benign (fold into k). This is the binding VQ2 validity risk regardless of S2 choice and should be
   measured first.

---

## 6. Decision dependencies (what live data would flip or sharpen the call)

1. **The toy-MPCC k on the corrected plant** (step 5). k ≲ 1.25 for a constraint-aware tracker → the
   decomposed ceiling (~5.5 s) is real and worth pre-building, shifting toward building the decomposed stack
   sooner. k ≥ 1.7 → hardens the recommendation to monolith-only-plus-graft. **The single number the whole
   field is blocked on.**
2. **inc7-with-envelope-OFF retrain.** Resolves whether the clean inc7 style tax is 2.63 or ~2.87 s, AND
   whether the relaxed-cone monolith starts cutting corners at gate-4 (the M-1 fallback trigger).
3. **inc7 per-tick gate-transition ticks from ShadowPC** (gitignored debug_obs). Confirms the post-gate-3 /
   G2→G3 +1.18 s localization the waterfall leans on (currently PROXY-reconstructed from inc6 transitions +
   bimodal localization). Tightens rung-2/rung-3 attribution.
4. **KF 1-σ at the post-gate-3 37 m/s window.** The architecture-independent hard floor; <0.05 m or NO
   approach is valid at race speed. Dominates the S2 choice — measure first.
5. **Organizer answers** on stack-carry-state between attempts and per-track determinism scope — confirms the
   offline line-iteration flywheel is legal as assumed and per-track frozen lines are an acceptable
   determinism contract.
6. **Gate-4 absolute contact-true margin** pending SHADOWPC-VISION-CAL (track_map registration). The binding-
   gate RANKING is robust (gate-4 confirmed independently: bound_free +0.046 m planning, inc7 +0.155 m flown);
   the absolute margin carries registration uncertainty that gates how aggressively corner speed can be pushed.

---

## 7. Completeness critic — what is still unverified / would change the call

This is the honest catalogue of what the panel did NOT verify, organized by how much it could move the call.

**COULD CHANGE THE CALL (high leverage):**
- **C-1 No tracked-flight time on the corrected plant exists for ANY non-monolith approach.** Every
  decomposition number is a point-mass PLANNING bound × an unbuilt tracker. The toy-MPCC k (dep #1) is the
  one number that would convert a paper ceiling into a measured one. Until it is run, the decomposed ceiling
  is unfalsified, not validated.
- **C-2 The honest 60° bound (10.6 s) is itself a point-mass bound** (instant attitude, no body-rate
  transient, ~0-airspeed convex-map over-fit that over-predicts thrust ~22% at 4–12 m/s). It is a mild LOWER
  bound — true cost ≥10.6 s — which only STRENGTHENS the monolith-first call, but the exact rung-2 magnitude
  (and thus the speed prize from cone relaxation) carries ±~0.5 s.
- **C-3 The gate-4 absolute margin is registration-uncertain** (SHADOWPC-VISION-CAL pending; track_map
  gate-3 D-offset ~1.46 m is a confirmed fragility). The RANKING (gate-4 binding) is robust across two
  independent methods, but the absolute 0.155 m could shift, changing how hard corner speed can be pushed in
  step 3/4. The contact-true metric flips at ~1.5 m D-shift.

**MODALITY/CLAIM NOT CHECKED (medium leverage):**
- **C-4 The KF achievable 1-σ at 37 m/s was NEVER measured** — only the *requirement* (<0.05 m) and the
  *raw* perception East σ (0.47 m) are known. The 9.4× gap between raw σ and requirement is assumed closable
  by filtering; no filter has been run at the post-gate-3 window. This is the architecture-independent hard
  floor and it is UNVERIFIED. If the KF cannot reach <0.05 m, the race-speed validity ceiling binds BEFORE any
  S2 choice matters — the estimator becomes the program's critical path for ALL architectures. **This is the
  single biggest unverified claim in the panel.**
- **C-5 No corrected-aero line was actually BUILT.** Step 2's generator is asserted feasible (envelope ~40
  lines exist) but the corrected-aero contact-free min-snap line — the asset BOTH the hybrid reward and the
  fallback depend on — has not been produced or audited. Its existence/quality is a prototype assumption.
- **C-6 The hybrid arc-length-progress graft was not prototyped in-loop.** It is argued to remove the corner-
  cut incentive "by construction," but no training run confirms it does not introduce a new pathology (e.g.
  reward hacking via progress-rate gaming, or over-tracking a sub-optimal line that caps the ceiling below the
  learn-the-line monolith). The graft's *ceiling cost* (does pinning to Γ leave speed on the table vs free
  learn-the-line?) is unquantified.
- **C-7 The convex+anisotropic TOGT re-run was not done.** The corrected ceiling band (~4.6–4.9 s) uses
  pooled isotropic c2=0.052 and models 8 g as linear T/W 8; the true map is convex sub-linear below hover and
  drag is anisotropic (0.076 body-down, loaded by the 26 m descent). Second-order — would tighten the rung-3
  target, would NOT move the S2 verdict.

**LOWER LEVERAGE / KNOWN-AND-ACCEPTED:**
- **C-8 The rung-1 internal split** (alt-relay vs k=1.85-dilation vs start-transient vs descent-caution) is
  reasoned apportionment, never A/B-measured at VQ1 level. MEDIUM on the split, HIGH on the 25.54 s total and
  its pathology-removal attribution (the 8.3 s geometric floor brackets it). Does not affect S2.
- **C-9 The 0.24 s ball→box margin recovery and rung-3 planning wins** must be re-verified against the gate-4
  contact-true metric before shipping — already in the plan (step 4/5), flagged here for completeness.
- **C-10 inc7 per-gate crossings are PROXY-reconstructed** (inc6 standing transitions + bimodal localization),
  not inc7 debug_obs. The two-block pre/post-G3 split is HIGH confidence; intra-block per-segment carries
  ~±0.3 s. Dep #3 closes this.

**WHAT LIVE DATA WOULD FLIP THE CALL (the short list):**
1. A toy-MPCC k ≲ 1.25 on the corrected plant → makes decomposition's ceiling real; would justify building
   the decomposed stack in parallel rather than as a gated fallback. (Currently k=1.85, the opposite.)
2. A relaxed-cone hybrid monolith that demonstrably cuts corners at gate-4 → fires the M-1 trigger; promotes
   the decomposed hard-ban fallback from "warm" to "build now."
3. A KF that CANNOT reach <0.05 m 1-σ at 37 m/s → demotes the entire S2 question; the estimator becomes the
   critical path and race-speed validity binds before any planner choice. (Most likely to actually bind.)

None of these is currently observed; all three point the same way today (monolith-first). The recommendation
is robust to everything verified and is gated, not bet, on everything unverified.

---

## 8. Bottom line for Fengyou

The two adversarial lenses converge: **retrain the monolith on corrected aero as a hybrid-monolithic (arc-
length progress over a rebuilt contact-safe line), open the cone one rung, and keep the decomposed stack as a
triggered warm fallback.** The recoverable time is in the ENVELOPE (cone relaxation, architecture-agnostic),
not in planning/tracking, so the architecture question is largely demoted — and the binding VQ2 risk is the
ESTIMATOR (East σ 0.47 m = 3× the gate-4 margin), which no planner architecture fixes and which is the single
most under-verified claim. Build the line generator and the toy-MPCC probe (cheap, native, serve both paths),
measure the KF at race speed FIRST, and do not switch the live, 7-session-validated monolith off to chase a
~5.5 s paper ceiling gated on a tracker that three native prototypes have already failed to build.
