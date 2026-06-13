# TOGT Bound + S2 Architecture - Draft Report

**Fengyou** - draft for commander refinement. Peregrine / AI Grand Prix. Date 2026-06-13.
Scope: offline analysis + prototypes only (no live sim, no SLURM, no source edits). All numbers reproduced natively in .venv (PYTHONPATH=src).

**TL;DR.** The 35.3 s -> corrected-bound gap is a clean three-rung monolithic ledger whose rung 1 (the model-based->RL collapse) is already closed by the live-confirmed inc7. The remaining recoverable time lives almost entirely in the style/tilt cone (a reward/doctrine lever, architecture-agnostic, that speeds the live monolith equally), not in planning/tracking. Recommendation: staged monolithic-then-decomposed - retrain the live monolithic policy on the corrected-aero plant for inc8, cast as a hybrid (arc-length progress reward over a rebuilt contact-safe line), open the style cone one rung; keep the fully decomposed stack as a triggered warm fallback. Confidence HIGH (two independently-charged adversarial lenses converged). The binding VQ2 risk is the estimator, not the planner.

> CRITICAL FRAMING CORRECTION (load-bearing). Two distinct tilt models give wildly different style-cone numbers. The Phase-C prototype (proto_envelope_topp.py) applies the tilt cap to cornering accel only -> 60deg = 5.35 s. The Phase-C verifier and the S2-final reproduction apply the cap to the total specific-force direction (the physically correct definition - body tilt points along f = a_des - g, which braking dominates) -> 60deg ~= 9.8-10.6 s. The honest reading is the latter. This single modeling judgment is what makes the verdict "the lap time is a TILT story, not a planner story": at the doctrine-legal 60deg cone every trajectory-opt approach lands ~= 9.6-10.6 s ~= inc7 already-live 9.76 s. Where a single number is needed below, the honest (total-force) tilt model is used and the cornering-only prototype number is shown in parentheses as the optimistic edge.

---

## Commander review note (2026-06-13, opus-4.8)

Workflow `wf_4a86f0e5-dd8` (21 agents, 2.48M tok, ~76 min). Three agents hit transient API errors mid-run — `B:design:togt_cpc`, `B:design:mpcc` (rate-limited returns), `B:judge:integration` (socket) — but each wrote its full artifact before failing (`artifacts/B_approach_{togt_cpc,mpcc}.md`, 18–23 KB) and the judges scored all four approaches from context, so the panel is substantively complete. All numbers reproduced natively in `.venv`. The adversarial verification layer functioned as designed: it **refuted** C1's most consequential headline (the "0.77 s tilt tax / mostly reward-shaping" claim) and the S2-final independently reproduced the correction — that self-correction is the report's most valuable output.

**Rung-3 reconciliation (commander, supersedes the row-3 "policy under-driving" framing).** The Phase-A waterfall was authored *before* the honest-tilt correction, so its rung-3 attribution (2.17 s "planning + policy under-driving; RL flies ~50% of optimal") compares inc7/inc5 against the **90°-unconstrained** point-mass bound — apples-to-oranges, since neither policy flies a 90°/inverted envelope. Under the honest total-force tilt model:
- inc7 (rw_tilt=96, free-cone 60°, ~55–75° effective tilt) at 9.76 s sits right at the honest 60°-cone point-mass bound (~9.8–10.6 s) → **near point-mass-optimal for its envelope, not under-driving.**
- inc5-R1 unconstrained (6.89 s) sits at the honest ~75–78°-cone bound (~6.7 s) → **also near-optimal for its effective envelope.**
- ⟹ On the **cornering** segments the recoverable time is *envelope* (cone relaxation), not policy. The one genuine architecture-relevant residual is **straight-line under-driving** (straights are drag-wall-limited, NOT cone-limited — the G2→G3 41 m straight is the suspect), but that estimate rests on PROXY-reconstructed inc7 crossings (real per-tick debug_obs is ShadowPC-gitignored) and is the lower-confidence, smaller piece (order tenths-to-~1 s, not 2.17 s).

**Net effect: this *strengthens* the verdict.** Nearly the entire ~5 s live prize (rungs 2+3) is the **tilt envelope** (free-cone + rw_tilt weight) plus a small straight-line speed-reward term — both architecture-agnostic. The decomposed-vs-monolithic choice owns almost no speed prize at the legal envelope; the binding VQ2 validity risk is the **estimator** (East σ 0.47 m = 3.0× the 0.155 m gate-4 margin). Commander concurs with `staged_monolithic_then_decomposed` at HIGH confidence. The straight-line speed-reward lever is exactly what the recommended hybrid arc-length-progress graft captures *inside* the monolith — no architecture change needed to chase it.

---

## 1. Quantified GAP ATTRIBUTION (the waterfall)

The gap from the old model-based VQ1 finish to the honest corrected-aero point-mass bound is one three-rung ledger. Sum of deltas = 30.58 s = 35.30 - 4.72 (verified).

| Rung | From -> To (s) | Delta (s) | What it is | Closable by | Status |
|---|---|---|---|---|---|
| 1 | 35.30 -> 9.76 | 25.54 | Model-based -> RL collapse. Pathology removal, NOT raw speed: k=1.85 global time-dilation + alt-relay limit cycle + start-transient + descent-caution + reactive non-optimal line. | (done) RL replaced the geometric stack. | CLOSED - inc7 live-confirmed (5/5 standing FINISHED, gate-3 barrier gone, 0/5 live collisions). |
| 2 | 9.76 -> 6.89 | 2.87 | R4 tilt-cap style tax (2.63 s clean inc5 A/B + 0.24 s inc7-vs-inc5 lineage offset; the 0.24 s is NOT envelope-recoverable). | Envelope relax: rw_tilt 96->48 (~1.4 s on non-binding segments) + free-cone 60deg->75-80deg (~1.5 s on 3 high-kappa corners G1->G2, G2->G3, G4->G5). R4 hinge stays intact; high-value 65->80deg band avoids the inc1-inversion regime. | OPEN - biggest recoverable lever. Already MEMORY next-queued action (envelope ladder step 1), now unblocked by inc7 live-confirm. |
| 3 | 6.89 -> 4.72 | 2.17 | Planning + policy under-driving vs the corrected-aero point-mass bound. RL flies ~50% of optimal speed every segment, worst on the 41 m G2->G3 straight. | Better planner + tracker: ride the thrust ceiling ~84% of lap (TOGT optimum) via a progress/speed reward (hybrid MPCC-cast); offline line-iteration incl. ball->box margin frees ~0.24 s. ~1.5-1.9 s is pure policy-vs-point-mass slack; remainder is point-mass idealization. | OPEN - lower confidence. This is exactly the regime the inc1 backflip-dive lived in. |

Endpoint = 4.72 s, honest band 4.7-4.9 s (the OPTIMISTIC edge). The shipped 4.55 s line and 4.27 s bound are falsified-linear-plant fictions (planned v_max 51-52 m/s; real v_max = 39.3 m/s, v^2 drag wall). Corrected-aero point-mass bound cross-checks two ways: prototype 90deg unconstrained TOPP = 4.574 s vs independent C++ TOGT corrected-aero refined 4.714 s (delta 0.14 s / 3%; v_max 39.17 vs 39.26 - agree <0.3%).

### Biggest lever
Rung 2 - style/envelope tilt-cap relaxation (~2.6-2.9 s/lap). It is (a) the largest single recoverable block, (b) the ONLY rung measured end-to-end (inc5 A/B -> bankable, not modeled), (c) the lowest-risk (the high-value 65deg->75-80deg band recovers ~1.3-1.5 s WITHOUT entering the inc1-inversion regime), and (d) it splits into two composable, offline-verifiable sub-levers: rw_tilt weight 96->48 (non-binding segments) + free-cone 60deg->75-80deg (high-kappa corners). Bank the style lever first, in the monolithic architecture, before touching the S2 architecture question. Secondary lever = rung 3 (~1.5-1.9 s pure policy slack) - higher ceiling but lower confidence and it is the backflip-dive regime.

### Two non-rungs (double-count guards - carry forward)
- A2 3.87 s structural-tracker gap is NOT a waterfall rung. It is the decomposed-geometric-tracker tax (= (k=1.85-1) x 4.551) that monolithic RL never pays (its work is already inside rungs 1+3). Inserting it would double-count the dilation already removed in rung 1. It lives as the S2-architecture adjudication number that kills decomposition-with-a-naive-tracker (8.3 s is slower than even style-ON inc7).
- A2 0.278 s margin tax and +0.44 s plant revision are denominator decompositions (linear-fiction 4.27 -> honest 4.72), not additive taxes.

Live prize = rungs 2+3 ~= 5 s.

---

## 2. TRAJECTORY-OPT JUDGE PANEL

Four approaches scored under two independent lenses (TIME, ROBUSTNESS+VALIDITY). Scores are 0-10 within each lens. Full derivations: artifacts/B_judge_time.md, artifacts/B_judge_robustness.md, artifacts/B_synthesis.md.

| Approach | TIME | ROBUSTNESS+VALIDITY | One-line read |
|---|---|---|---|
| Monolithic RL (inc7-class) | (overall time winner - measured) | 7.5 | ONLY live-confirmed contact-free validity; fastest realized-NOW (inc5 unconstrained 6.9 s twin MEASURED). |
| MPCC tracker | 6.0 | (~ hybrid) | Fastest defensible of the four (5.4-5.8 s @ cone) because it IS the tracker; but unbuilt (acados/WSL), fat slow tail. |
| HYBRID (decomposed line + RL/MPCC tracker) | 5.5 | 6.0 | Best-calibrated estimate (8.0 s offline) - but its own calibration says slower than the live monolith. |
| min-snap + coupled-TOPP | 4.5 | 6.5 | Free deterministic LINE GENERATOR, not a time frontier; closed-loop validity outsourced to an unbuilt tracker. |
| TOGT collocation | 4.0 | 4.5 | Best ceiling, worst realizability-to-ceiling ratio; only lines are FALSIFIED. Role = offline BOUND + geometry SEED only. |

### Recommended approach and its VALIDATED achievable lap time
Recommended: MONOLITHIC RL (inc7-class), retrained on corrected aero for inc8, hybrid-cast (arc-length progress reward over a rebuilt contact-safe line), style cone opened one rung.

- Validated achievable (MEASURED, not projected): inc5 unconstrained monolithic twin = 6.9 s on the identical corrected-aero plant; inc7 monolithic offline = 9.76 s / live = 11.45 s fresh-sim. The monolith is the only option with a measured realized lap; every decomposition number is a point-mass planning bound x an unbuilt tracker.
- The natural decomposed fallback stack (line-gen + MPCC tracker) has an honest ceiling ~5.4-5.8 s - genuinely faster than inc7 9.76 s - but it is a projection gated on a tracker that does not yet exist (the only measured tracker datum on this plant is k=1.85, which is bad; 3 of 3 native tracker prototypes failed: min-snap geometric -> None/inf, hybrid DF+PD -> 5-35 m miss, scipy-SLSQP toy MPCC -> ~18-21 m/s vs the 38 m/s wall).

### Tilt -> lap-time curve (corrected-aero point-mass bound)

| tilt cap | cornering-only TOPP (optimistic, proto_envelope_topp.py) | honest total-force TOPP (verify_honest_tilt_topp.py) | open-loop feasible? |
|---|---|---|---|
| 60deg | 5.35 s | ~9.8-10.6 s | NO - 2.2% of lap demands >78.3 m/s2 (max req 111.6) |
| 65deg | 5.08 s | ~8.8 s | NO - 1.4% over ceiling (max req 101.4) |
| 75deg | 4.69 s | ~6.7 s | YES (max req 77.8 <= 78.3) |
| 80deg | 4.57 s | ~5.4 s | YES (max req 70.5) |
| 90deg (cap off) | 4.57 s | 4.60 s | YES - both models agree (no cap to mis-apply) |

Reading. At the doctrine-legal 60deg cone the honest bound is at or above inc7 already-live 9.76 s - so the recoverable time is in the envelope (cone relaxation, a reward/doctrine lever that speeds the monolith equally), NOT in planning. The same cone relaxation that unlocks 5-6 s would also speed the already-live monolith (inc5 unconstrained measured 6.9 s). 80deg and 90deg are identical (4.574 s) because on this gentle course (R_min ~ 30 m) the sharpest corner at the 39 m/s drag wall needs only a_lat ~ 51 m/s2 = g*tan(79deg); above ~79deg the v^2 drag wall is the sole ceiling.

> TIME-lens adversarial caveat (carry forward): all fast numbers (togt 4.9, mpcc 5.8) are unbuilt-tracker projections with a fat slow tail; the only measured tracker is k=1.85. The TIME lens alone does not justify switching off the monolith - the fastest realized-NOW path is "relax the cone + retrain the already-live monolith."

---

## 3. S2 ARCHITECTURE RECOMMENDATION

Recommendation: staged_monolithic_then_decomposed. Confidence: HIGH. Full derivation: artifacts/D_s2_final.md; adversarial lenses artifacts/D_adjudicate_skeptic_mono.md (charged to argue FOR monolithic) and artifacts/D_adjudicate_skeptic_decomp.md (charged to argue FOR decomposition) - both independently converged on this exact call, same inc8 build, same fallback trigger logic, same "estimator is the binding risk" finding.

Retrain the live MONOLITHIC policy on the corrected-aero plant for inc8, cast as a HYBRID-monolithic (arc-length progress reward over a rebuilt, offline, contact-safe min-snap reference line), with the style cone opened one ladder rung. Keep the fully DECOMPOSED stack (offline line generator + a real MPCC/RL tracker) as an explicitly-specced but unbuilt-until-triggered warm fallback.

### Why monolithic-first
1. Recoverable time is in the envelope, not the architecture. The largest MEASURED recoverable block (rung 2, ~2.6-2.9 s) is the style-cone tilt tax, which Phase-C proved is overwhelmingly kinematic, not reward-shaping: honest tilt-capped TOPP at 60deg = 10.62 s (reproduced bit-for-bit, verify_honest_tilt_topp.py) and matches inc5-measured rw_tilt=96 -> 9.52 s. It is unlocked by opening the cone - a reward/doctrine lever that is architecture-agnostic and speeds the monolith too.
2. The monolith is the only MEASURED realizable lap. inc7 is live-confirmed (5/5 standing FINISHED, gate-3 barrier gone, 0/5 live collisions, |tanh| <= 0.547, 7 live zero-contact sessions, graceful+observed failure).
3. A good tracker is empirically the hard part. Only measured datum is k=1.85 (8.3 s - slower than the style-ON monolith; 54-combo sweep finds nothing faster -> STRUCTURAL gap). 3/3 native trackers FAILED. Decomposition-with-the-existing-tracker is strictly dominated.
4. Decomposition assets are fiction, its wins graft free. The shipped reference_line_vq1.json is built on the falsified linear plant (max R33-tilt 170.4deg, 25.8% inverted samples, peak 51.1 m/s vs 38.6 m/s drag wall) -> geometry must be REBUILT before decomposition is even a valid target. Its two real wins (no-corner-cut incentive via arc-length progress; offline line-iteration flywheel) import to the monolith at near-zero cost. Its one unique win (HARD inversion ban) guards inc1 backflip whose root cause is already FIXED and which inc7 never exhibits.
5. The binding VQ2 risk is the ESTIMATOR, not the planner (architecture-independent -> demotes the whole S2 question): perception East sigma = 0.47 m = 3.0x the 0.155 m gate-4 binding margin, unfiltered. Every world-frame line-tracking approach consumes the same world fix through the same KF - decomposition buys NOTHING on perception. Latency 67 ms is validity-benign (0.77-1.93 mm cross-track, <1.3% of margin; fold into k).

### Failure modes
Monolithic: M-1 SOFT (not HARD) inversion guarantee via R4 hinge - could drift to corner-cutting under aggressive cone relaxation toward 75-80deg (the one regime decomposition hard ban would save -> this is the fallback trigger). M-2 legacy R1 progress-to-gate-CENTER could pay a speed-pushed policy to clip the frame -> removed BY CONSTRUCTION by the hybrid arc-length graft. M-3 narrow basin (2/3 seeds) -> budget >=4 seeds. M-4 fresh-respawn post-gate-3 sensitivity (bimodal laps) -> crown only after >=3-5-lap fresh-reset live batch (WINNER-VALIDATION RIDER); the monolith is the only option that has ALREADY passed one. M-5 gate-4 margin erosion at speed -> re-verify every speed win against rl/contact_true_eval.py.

Decomposed: D-1 tracker realizability UNPROVEN (dominant; only k=1.85, 3/3 native trackers failed - if real k>=1.7 it is a slower, costlier monolith). D-2 a k~1.7 tracker is by definition lagging, and lag into the post-gate-3 ~37 m/s gate-4 approach erodes the 0.155 m cross-track budget at the worst place. D-3 geometry must be REBUILT (shipped line drag-infeasible + 170deg inverted). D-4 HIGH integration cost (3-subsystem; acados/MPCC does not build on Windows/py3.13; RL-tracker needs a new line-relative obs layout + fresh Adroit campaign, monolith checkpoints NOT reusable). D-5 no perception benefit. D-6 even the contact-valid corrected TOGT line is razor-thin (bound_free clears gate-4 by only +0.046 m planning margin; ref_margin is INVALID at r=0.38, gate-4 -0.110 m). D-7 TOGT-collocation standalone is catastrophic as a deployable (its "feasible" refined line demands 17 rad/s body rates the ~11.2 rad/s super-rate plant CANNOT produce) -> role = offline BOUND + geometry SEED only.

### Ordered integration plan
1. [reward change] Graft the hybrid arc-length progress reward: replace peregrine_racing.py R1 (progress-to-gate-CENTER) with progress-ALONG-Gamma via reference_line.progress() over a rebuilt contact-safe line. Closes M-2 by construction. Highest-value graft.
2. [native, no blocker] Build the corrected-aero min-snap + coupled-TOPP line generator (pure numpy, deterministic, <1 s; speed_profile.py half-built, envelope ~40 lines exist in proto_envelope_topp.py, min-snap QP = one scipy.linalg.solve_banded). Emit a corrected-aero feasible, contact-free, NON-saturated line in peregrine.reference_line.v1 schema, crossings DEAD-CENTER (full 0.37 m contact-true band free @ r=0.38), planned at DR-conservative params (a_up_max ~ 70 not 78.3, pooled c2=0.052). Architecture-neutral: feeds the hybrid reward NOW and the decomposed fallback LATER.
3. [doctrine lever - THE load-bearing speed knob] Open the style cone one rung: rw_tilt 96->48 FIRST, then free-cone 60deg->~70deg if the gate-4 metric holds. Phase-C proved rw_tilt alone is necessary-but-insufficient - the free cone is the binding kinematic constraint. Gated on LAPTOP-INC8-BINDING-GATE-VERIFY (gates 4,5 @ r=0.38).
4. [retrain] inc8 on corrected aero (convex collective map + quad drag, LAPSE OFF, linear_drag=0, super-rate + mixer) with steps 1+3, >=4 seeds. Validate map-ON (peregrine_eval/offline_rollout default to legacy flat plant - footgun). Crown ONLY after a >=3-5-lap fresh-reset live batch.
5. [realizability probe - native, most decision-relevant] Run the scipy-SLSQP toy MPCC tracker on rl_plant.step to MEASURE the real k for a constraint-aware tracker (currently one bad datum). Adjudicates whether the decomposed fallback ~5.5 s ceiling is real BEFORE paying any acados/WSL/Adroit cost.
6. [fallback - build ONLY on BOTH triggers] Production decomposed stack. Trigger (a): step-5 probe shows a constraint-aware tracker recovers >=~80% of envelope speed (k <~ 1.25); AND (b): the relaxed-cone hybrid monolith actually starts cutting corners at gate-4 (the M-1 regression). Gate, do not pre-build.
7. [cross-cutting - PRIORITIZE OVER ANY PLANNER CHOICE] Estimator: drive the KF to <0.05 m 1-sigma at the post-gate-3 ~37 m/s window. Binding VQ2 validity risk regardless of S2 choice; measure FIRST.

### Decision dependencies
- The toy-MPCC k on the corrected plant for a CONSTRAINT-AWARE tracker (step 5) - the single number the whole field is blocked on. k <~ 1.25 -> decomposed ceiling (~5.5 s) is real, build sooner; k >= 1.7 -> hardens to monolith-only-plus-graft.
- inc7-with-envelope-OFF retrain - resolves whether the clean inc7 style tax is 2.63 or ~2.87 s, AND whether the relaxed-cone monolith starts cutting corners at gate-4 (the M-1 fallback trigger).
- inc7 per-tick gate-transition ticks from ShadowPC (gitignored debug_obs) - confirms the post-gate-3 / G2->G3 +1.18 s localization the waterfall leans on (currently PROXY-reconstructed).
- KF 1-sigma at the post-gate-3 37 m/s window - the architecture-independent hard floor; <0.05 m or NO approach is valid at race speed. Dominates the S2 choice; measure first.
- Organizer answers on stack-carry-state between attempts and per-track determinism scope - confirms the offline line-iteration flywheel is legal as assumed.
- Gate-4 absolute contact-true margin pending SHADOWPC-VISION-CAL - the binding-gate RANKING is robust (gate-4 confirmed two ways: bound_free +0.046 m planning, inc7 +0.155 m flown), but the absolute margin carries registration uncertainty (metric flips at ~1.5 m D-shift).

---

## 4. PROTOTYPES + VALIDATION

All under handoff/ultracode-planning-togt-s2-2026-06-13/. Pure numpy/scipy, constants read live from src/racer/rl_plant.py, no RNG / live sim / SLURM / network. All reproduce bit-for-bit on re-run.

### C1 - Envelope-aware corrected-aero point-mass TOPP
proto_envelope_topp.py -> envelope_topp_result.json. Forward-backward TOPP over a cubic-spline centre line through the 6 gates; accel budget = thrust ball radius A_UP_MAX = 78.283 m/s2 (~8 g, full-stick convex collective map) centred at gravity in NED; cornering coupled to the style envelope via a_lat_max = g*tan(tilt_cap); top speed capped by the v^2 quad-drag wall (c2=0.052) ~ 39 m/s.

- Measured: line length 164.65 m, max spline-to-centre deviation 0.0125 m (asserted <=0.5 m), min radius 29.97 m. Tilt sweep (cornering-only model): 60deg=5.348 / 65deg=5.076 / 75deg=4.686 / 80deg=4.574 / 90deg=4.574 s. v_max 38.1->39.17 m/s. Per-gate miss 0.0 m at all 6 gates; contact-free TRUE at r in {0.38, 0.33, 0.28} (full +0.37 m margin @ r=0.38, including binding gate-4).
- Cross-check: 90deg unconstrained 4.574 s vs independent C++ TOGT corrected-aero refined 4.714 s (delta 0.14 s / 3%); v_max 39.17 vs 39.26 (agree <0.3%). Corrected ceiling ROBUST ~4.5-4.7 s - the doubled thrust (T/W~8) does NOT buy sub-4 s; the v^2 drag wall caps ~39 m/s and THRUST binds (TOGT 91% collective saturation).
- Adversarial verdict (C_verify_envelope_realizability.md): PARTIALLY UPHELD. Unconstrained ceiling SOUND. CENTRAL FLAW: the cap is applied to cornering accel only; body tilt is the direction of the total specific force (braking-dominated). Honest total-force recompute: 60deg = 9.83 s (not 5.35), realized tilt a correct 54.5deg; 60deg tilt tax = ~5.2 s not 0.77 s. C1 downstream "ladder step 1 needs no cone relaxation" claim collapses - the cone IS the binding kinematic constraint. Validity verdict (C_verify_contact_validity.md): centre line genuinely contact-free, L-inf <= 0.0008 m to every centre, no corner-cutting - UPHELD. 60deg/65deg laps are open-loop INFEASIBLE lower bounds (2.2%/1.4% of lap demands >78.3 m/s2); 75deg/80deg/90deg are clean feasible bounds.
- Realizability: the naive feedforward+PD twin probe DIVERGES 37.9 m on the 35+ m/s plan -> realizing this ~5 s-class ceiling requires a capable tracker (learn-the-line RL or MPCC). The corrected prize (~4.6-5.4 s honest) is ~2x faster than inc7 9.76 s twin -> a STRUCTURAL gap, not a residual.

### C2 - Re-evaluate the shipped TOGT trajectories under corrected aero
proto_togt_corrected.py. Pure-numpy re-analysis of the shipped TOGT CSVs (no C++ re-run) under three lenses (thrust feasibility / drag re-accounting / fixed-geometry TOPP re-time).

- THRUST: the old TOGT line is FEASIBLE under corrected aero with ~8.5x headroom (refined demands <=36.92 m/s2 body-up; full-stick is 78.28 m/s2; ZERO nodes exceed authority).
- DRAG: the old line is INFEASIBLE above ~37 m/s. It plans 46-55 m/s; at the refined peak 55.2 m/s corrected quad drag = 158.6 m/s2 vs old linear 11.6 m/s2 - a 13.7x under-modeling. Drag wall ~ 37-39 m/s.
- NET: corrected achievable time for the EXISTING geometry = ~4.7-5.0 s, NOT 4.27 s (geometry-fixed TOPP re-time = 4.996 s; IPOPT geometry-reopt = 4.714 s, VALIDATED). The 4.27 s planning bound was 0.4-0.7 s OPTIMISTIC - a falsified-linear-plant artifact (phantom speed from under-modeled drag). Margin tax is real and plant-independent (~0.18-0.30 s/lap).
- Adversarial verdict (C_verify_plant_honesty.md + C_verify_contact_validity.md): PARTIALLY UPHELD. Plant constants + v^2 drag wall UPHELD exactly (full-stick 78.283 m/s2 = 7.98 g; pooled wall 38.65 m/s vs C++ 39.26; thrust headroom 8.48x). Falsified-linear 4.27 s / 4.55 s line REFUTED. VALIDITY CORRECTION: the 4.714 s corrected-TOGT cross-check anchor is CONTACT-INVALID (gate misses up to 0.766 m > 0.75 half-opening even at r=0; all 6 gates violated at r=0.38) - a margin-0 ball-gate planning bound, must NOT be quoted as a flyable valid lap. The C2 claim that ref_margin (4.311 s) is contact-valid is REFUTED at r=0.38 (L-inf 0.41-0.48 vs band 0.37 -> gate-3 -0.111 m, gate-4 -0.110 m). The ONLY contact-valid shipped TOGT line at r=0.38 is bound_free (4.432 s), clearing gate-4 by just +0.046 m. Pooled drag c2=0.052 is CONSERVATIVE on the dominant nose-first axis (c2=0.042) -> bounds under-claim straight-line speed, not over-claim.

### Limitations (both prototypes)
- PLANNING bounds, not tracked times. No body-rate/tilt-rate dynamics (instant attitude); TOGT shows rates cost ~0.04 s but the multiple-shooting lap is ~0.14 s slower -> the 90deg 4.574 s is mildly optimistic. No prototype produces a tracked, contact-valid, corrected-aero FLIGHT time - C1 tracker diverges 37.9 m; the only measured realized lap is inc7 (9.76 s twin / 11.45 s fresh).
- Pooled isotropic drag c2=0.052 (measured table is per-axis/per-sign 0.042-0.076; the 0.076 body-down brakes harder on the 26 m descent -> second-order shift).
- Convex collective map is a ~0-airspeed fit (over-predicts thrust ~22% at 4-12 m/s; lapse correctly VOIDED/OFF) -> the 78.3 m/s2 ceiling is the ~0-airspeed value -> bound a touch optimistic at race speed (consistent with TOGT being 0.14 s slower).
- 60deg/65deg laps are mild lower bounds (corner-entry braking 1.4-2.2% over the thrust ball -> true ~+0.1-0.2 s slower); 75deg/80deg/90deg are clean.
- Bare centre-line geometry (no through-gate normal entry/exit min-snap shaping, no tracker-error margin offset).

---

## OPEN / REMAINING
1. KF achievable 1-sigma at 37 m/s post-gate-3 NEVER measured - only the requirement (<0.05 m) and raw perception East sigma (0.47 m) are known. The 9.4x gap is ASSUMED closable by filtering; no filter run at that window. Architecture-independent critical path for ALL approaches - measure first.
2. Toy-MPCC k on the corrected plant for a constraint-aware tracker - the one number the whole field is blocked on (currently one bad datum, k=1.85). Converts the decomposed ~5.5 s ceiling from paper to measured.
3. No corrected-aero contact-free min-snap line actually BUILT - step-2 generator asserted feasible (~40 lines exist) but the asset both the hybrid reward and the fallback depend on is unproduced/unaudited.
4. Hybrid arc-length-progress graft not prototyped in-loop - argued to remove the corner-cut incentive "by construction"; no run confirms no new pathology (progress-rate gaming) or quantifies its ceiling cost vs free learn-the-line.
5. inc7 per-gate crossings are PROXY-reconstructed (inc6 transitions + bimodal localization), NOT inc7 debug_obs (ShadowPC-only, gitignored) - pull per-tick ticks to confirm the G2->G3 +1.18 s localization; intra-block per-segment carries ~+/-0.3 s.
6. Style-tax basis offset (2.63 clean vs 2.87 ledger): the +0.24 s inc7-vs-inc5 lineage cost is NOT envelope-recoverable; re-measure the tax on the inc7 lineage (inc7 envelope-OFF) to resolve.
7. Gate-4 absolute contact-true margin is registration-uncertain (SHADOWPC-VISION-CAL pending; track_map gate-3 D-offset ~1.46 m confirmed fragility, metric flips at ~1.5 m shift). RANKING robust (two methods); absolute 0.155 m could shift.
8. rung-1 internal split (alt-relay vs k=1.85-dilation vs start-transient vs descent-caution) is reasoned apportionment, never A/B-measured at VQ1 level (MEDIUM on split, HIGH on the 25.54 s total + pathology-removal attribution - the 8.3 s geometric floor brackets it).
9. Second-order (would NOT move the verdict): a convex+anisotropic TOGT re-run (climb drag 0.076 on -z, loaded by the descent) would tighten the ~4.6-4.9 s ceiling band and sharpen the rung-3 target.
