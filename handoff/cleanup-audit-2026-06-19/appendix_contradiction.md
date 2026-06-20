# Cleanup Audit Appendix — Cross-cutting Contradictions

**Bucket:** `contradiction` · **Date:** 2026-06-19 · **Scope:** cross-cutting contradictions, stale verdicts, and the propagation gaps surrounding them across `memory/` and `handoff/` in the canonical repo (`C:/Users/Fengy/Downloads/Projects/Anduril-cmdr`).

## Area state (summary)

This area is dominated by ONE root event: the **2026-06-19 gate-4 σ_p0 bar correction** (commit `511e85c`, captured ONLY in `MEMORY.md:8`). Fengyou caught that the σ_p0 ≲ 0.08 m "closure bar" was a **double-count** — `margin_envelope.py` subtracted the drone radius TWICE (`W_EFF = 0.75 − 0.215 chassis`, then `MARGIN = W_EFF − r`; `0.08 = 0.235/3`). The SIM was always correct (pass = offset < 0.75 − r → real clearance 0.37–0.47 m); the real bar is **σ_p0_lat ≲ 0.15 (p99 ≲ 0.45)**, the measured 0.15–0.20 is **MARGINAL-PASSING not NO-GO**, and the **near-field-gate-estimator pivot is RETRACTED** (RL stays primary; gate-4 is a REACH/PASS-RATE problem, not sub-8 cm centering).

Because the correction commit touched ONLY `MEMORY.md`, the fix did NOT propagate: the double-counted 0.08/0.235/0.155 budgets and the "NO-GO vs 0.08 / near-field-estimator next-lever" verdict still read as current truth across two sub-indices, three topic files, six handoff REPORTs, the design-doc origin, and the `margin_envelope.py` source — and `MEMORY.md:8` even **self-contradicts** (retracts the near-field pivot in its head, re-asserts it as "2nd lever" in its tail). A SECOND, independent contradiction cluster concerns **case-C / self-localization** ("Q① resolved / position not streamed / case-C confirmed" stated as settled in three places, contradicted by the canonical `MEMORY.md:18` open status). A THIRD concerns the **stale "main local-only / do-not-push" directive** (main is in fact fully synced with origin). Plus speed/timing (37 m/s → corrected ~30 m/s; 4.72 s → ~8 s) staleness and several confirmed-consistent (no-defect) checks recorded for provenance.

---

## ISSUE TYPE: contradiction

### [HIGH] `margin_envelope.py` mislabels linf0 as chassis radius — the SOURCE of the wrong 0.08 bar
- **location:** `handoff/margin-closure-envelope-2026-06-14/margin_envelope.py:33,77,78,101,102` (mislabel also at `:31-33`)
- **issue_type:** contradiction
- **severity:** high
- **evidence:**
  - `:78` — `W_EFF = MARGIN_G4_AT_R038 + 0.38  # 0.535 m = 0.75 gate-clear-halfwidth - 0.215 chassis half-diag`
  - `:33` (docstring) — `MARGIN(r) = W_EFF - r, W_EFF = 0.535 m (= 0.75 gate-clear-halfwidth - 0.215 chassis half-diag)`
  - `:102` — `def margin_at(r): return W_EFF - r`
  - This labels `0.215` as the CHASSIS half-diagonal radius and then ALSO subtracts the body radius `r` again in `MARGIN(r)=W_EFF-r` — the double-count. The CANONICAL meaning (`CORRECTED_MARGIN.md:49-59`, `g3_margin_sim.py:14`, `d3_margin_closure.py:59`) is that `0.215` is **linf0**, the RADIUS-INVARIANT gate-4 crossing offset, NOT a chassis radius: `budget(r) = (0.75 - r) - linf0(0.215)`.
  - `MEMORY.md:8` (canonical) — "margin_envelope.py subtracted the drone TWICE (W_EFF=0.75-0.215 chassis, then MARGIN=W_EFF-r=0.535-0.30=0.235; 0.08=0.235/3)". File last committed at `3542fd1` (pre-correction); never fixed.
- **recommended_action:** Add a correction header comment to `margin_envelope.py` (lines 31-33 and 77-78): the `0.215` term is linf0 (radius-invariant gate-4 crossing offset), NOT the chassis half-diagonal, so `MARGIN(r)=(0.75-r)-0.215` must NOT be read as a second radius subtraction; and the 0.08 m closure bar this script feeds was retracted 2026-06-19 (`MEMORY.md:8`, commit `511e85c`). Do NOT silently re-run/overwrite the numbers — preserve as the historical artifact that produced the now-debunked bar. (A second agent also flagged this same file as the named ROOT of the double-count, noting re-derived verdicts like `index_vision_estimator.md:296` "p99 0.297 vs MARGIN 0.235" are affected; preserve the 268-cell envelope sweep machinery and the σ=0.10-vs-0.265 finding — flag only the threshold.)
- **info_loss_risk:** none for the corrected interpretation (already in `MEMORY.md:8`, `CORRECTED_MARGIN.md:49-65`, `g3_margin_sim.py:14`); this only flags the un-annotated mislabel that still reads as current truth. The 268-cell envelope-sweep machinery is valuable and must be preserved — flag the threshold, keep the sweep.

### [HIGH] `index_vision_estimator.md:285` states the double-counted 0.08 m closure bar as current truth
- **location:** `memory/index_vision_estimator.md:285` (flagged by 3 agents)
- **issue_type:** contradiction (also reported as `stale` by other agents — same line, same defect)
- **severity:** high
- **evidence:**
  - `:285` — "CLOSURE BOUNDARY (gate-difficulty sweep): r=0.30 (B=0.235) CLOSES IFF σ_p0_lat ≲ 0.08 m; r=0.38 (B=0.155) ≲ 0.05; tight-VQ2 (B=0.12) ≲ 0.04."
  - This is the ORIGIN of the 0.08 number (`0.235/3`) AND is the `§TERMINAL-LOCK`/closure-boundary section `MEMORY.md:8` points the reader to as the home of the binding σ_p0 number → the stale bar is authoritative-by-pointer.
  - CONTRADICTED by `MEMORY.md:8` — "the σ_p0 ≲ 0.08 'closure bar' was a DOUBLE-COUNT ... REAL bar ≈ σ_p0_lat ≲ 0.15 (p99 ≲ 0.45)."
- **recommended_action:** Add a 2026-06-19 CORRECTION note at `:285` (and to the `§COAST-DRIFT RESOLVED` header at `:282`): mark the 0.08/0.05/0.04 boundaries as a `margin_envelope.py` double-count (W_EFF=0.75−0.215 chassis, then MARGIN−r subtracts the drone again); real clearance 0.37–0.47 m, real bar σ_p0_lat ≲ 0.15 / p99 ≲ 0.45; measured 0.15–0.20 = MARGINAL-PASSING. Keep the original numbers with a strikethrough + pointer to `MEMORY.md:8`; preserve the closure-boundary sweep methodology (miss p99 ≈ 3·σ_p0, B-axis difficulty sweep) — only the 0.08/0.05/0.04 thresholds are wrong.
- **info_loss_risk:** none if the B=0.235/0.155/0.12 budget arithmetic and the achievable-σ derivation are kept (still useful inputs); only the bar VALUE is corrected. Annotate-as-superseded; do not delete.

### [HIGH] `index_strategy_meta.md:10` asserts "vision self-loc = case-C confirmed / position almost-certainly NOT streamed"
- **location:** `memory/index_strategy_meta.md:10` (note internal tension with `:32` which correctly keeps Q① open)
- **issue_type:** contradiction
- **severity:** high
- **evidence:**
  - `:10` (FAQ 2026-06-15) — "Interface CONSISTENT across both VQ rounds + COURSE DOWNLOADABLE offline ... position almost-certainly NOT streamed live (vision self-loc = case-C confirmed); test the official VQ1 wire ONCE → holds for VQ2."
  - CONTRADICTED by canonical `MEMORY.md:18` (2026-06-17) — "Official VQ1 STREAMS LOCAL_POSITION_NED + ODOMETRY → our VQ1 PASSES USED GIVEN POSITION (case-A...)" and `reference_sim_interface.md:64` — "Our ShadowPC sim ... streams LOCAL_POSITION_NED (97 Hz) + ODOMETRY (75 Hz) — position IS on the wire." The empirical VQ1 wire DID stream position → "position almost-certainly NOT streamed live" and "case-C confirmed" are falsified for VQ1. SAME file at `:32` correctly keeps it open ("Q① (VQ2 streams pose?)") — an internal tension.
- **recommended_action:** Update `:10` to: "... position NOT streamed on the SPEC'd scored wire (§4.3); BUT our official VQ1 sim streamed LOCAL_POSITION_NED+ODOMETRY (case-A) → case-C is the DESIGN assumption for deploy, NOT yet 'confirmed' — VQ2 is the resolver (see MEMORY.md WIRE-SPEC, parked_backlog A1)." Remove the unqualified "case-C confirmed".
- **info_loss_risk:** none — the load-bearing FAQ facts (interface consistent across rounds, course downloadable, map known offline, test-VQ1-wire-once) are preserved; only the falsified "position not streamed / case-C confirmed" clause is corrected to the empirically-observed case-A-on-VQ1 status (`MEMORY.md:18`, `reference_sim_interface.md:64-67`).

### [MED] `MEMORY.md:8` self-contradicts — retracts the near-field-estimator pivot in its head, re-asserts it as "2nd lever" in its tail
- **location:** `memory/MEMORY.md:8` (flagged by 3 agents — internal contradiction within the canonical correction bullet)
- **issue_type:** contradiction
- **severity:** med
- **evidence:**
  - HEAD (retraction) — "🚩 RL IS the right tool — the 'pivot off RL to a near-field-gate estimator' recommendation is RETRACTED; next lever = RL to lift reach/pass-rate; vision = SUPPORT, do NOT lean on it."
  - TAIL (surviving old fragment, same line, in the SIGN-FOOTGUN clause) — "σ_p0 = per-fix σ_lat ... ⊕ σ_b (boresight bake −0.25 DEPLOYED 5764291). 2nd lever = near-field GATE estimator (lower the 12 m floor). r=0.38 NEVER; 'fix-rate ≥0.50 through the last 6 m' GEOMETRICALLY UNACHIEVABLE (the 12 m PnP floor)."
  - The retracted near-field-estimator pivot survives verbatim inside the very bullet that retracts it (the correction edit re-wrote the first half but left the trailing footgun sub-note).
- **recommended_action:** Reconcile the tail with the head: either delete the bare "2nd lever = near-field GATE estimator (lower the 12 m floor)" fragment, or re-label it as "near-field gate estimator = SUPPORT/insurance only (RETRACTED as the primary lever — see σ_p0 RESULT above; primary lever = RL reach/pass-rate)". **PRESERVE** the LIVE launch-config SIGN FOOTGUN (`++env.lookat_g_yaw=-3.0 ++env.lookat_g_pitch=3.0`, warmup_updates=0, rw_centering=1.0, per-seed hydra.run.dir) and the geometric facts (r=0.38 NEVER; 12 m PnP floor; "fix-rate ≥0.50 through last 6 m" geometrically unachievable) — only the "lever" framing changes.
- **info_loss_risk:** none — the near-field-estimator idea is preserved in `index_vision_estimator.md:286-287` and the coast-drift REPORT; the 12 m PnP floor geometry and the SIGN FOOTGUN launch config are load-bearing and must stay. Only the contradictory phrasing needs tightening.

### [MED] `reference_competition_materials.md:75` "Q① resolved" contradicts `:80` "Open gaps (wire-spec discrepancy)" in the same file
- **location:** `memory/reference_competition_materials.md:75` vs `:80`
- **issue_type:** contradiction
- **severity:** med
- **evidence:**
  - `:75` — "self-localize REQUIRED (estimator LOAD-BEARING; Q① resolved)" (treats it closed).
  - `:80` — "Open gaps (verify when official sim drops): ... the wire-spec discrepancy (our sim has position/ODOMETRY, official doesn't)." (lists the position-on-wire question as an OPEN gap → directly undercuts `:75`'s "Q① resolved").
  - Canonical `MEMORY.md:18` sides with `:80`: the question is OPEN pending VQ2.
- **recommended_action:** Make `:75` consistent with `:80`: change "Q① resolved" to "Q① resolved AT THE SPEC LEVEL (scored wire lists no position) but OPEN empirically — our VQ1 sim streams position; see Open gaps below and MEMORY.md WIRE-SPEC."
- **info_loss_risk:** none — both lines and their facts are retained; the edit only removes the in-file contradiction, aligning to canonical `MEMORY.md:18`.

---

## ISSUE TYPE: stale

### [HIGH] `MEMORY.md:4` — stale "main LOCAL-ONLY / do NOT push" directive (auto-loaded, reads as current truth)
- **location:** `memory/MEMORY.md:4` (NOW section)
- **issue_type:** stale
- **severity:** high
- **evidence:**
  - `:4` — "🚩 `main` LOCAL-ONLY, far ahead of origin — do NOT push / rewrite history (ShadowPC main may have diverged → reconcile via origin deliberately)."
  - CONTRADICTED by authoritative git state: `git rev-parse origin/main` == `git rev-parse main` == `c5d60a09c8138a6ce75d7d041b4fa8790e6856f7`; `git rev-list --left-right --count origin/main...main` == `0 0` (byte-identical → main IS fully pushed; no divergence remains). Task statement confirms origin/main == c5d60a0 is on GitHub.
- **recommended_action:** Replace the "main LOCAL-ONLY ... do NOT push" clause with a statement that main is now PUSHED and synced (origin/main == main == c5d60a0) and drop the standing do-not-push directive. If a ShadowPC-divergence caution is still warranted, re-scope it narrowly to "fetch+inspect before merging ShadowPC work" rather than a blanket "do NOT push". **Highest-priority** because `MEMORY.md` is auto-loaded and this line will actively misdirect the next commander into NOT pushing already-pushed work.
- **info_loss_risk:** none — the historical fact that main WAS local-only at the Gen-3→4 handoff is preserved in `COMMANDER.md:65` and `handoff/commander-handoff-2026-06-18.md:6,38`; only the now-false "currently still unpushed" assertion is corrected.

### [HIGH] `index_vision_estimator.md:52,58,63,65` — double-counted budget formula `(0.75−r)−0.215` stated as identity
- **location:** `memory/index_vision_estimator.md:52,58,63,65`
- **issue_type:** stale
- **severity:** high
- **evidence:**
  - `:52` — "The 0.155 m budget = budget(r=0.38) = (0.75 − 0.38) − 0.215, where linf₀=0.215 m is the radius-invariant gate-4 simstart crossing offset."
  - `:58` — "Budget table (budget(r) = (0.75−r) − 0.215):"
  - `:63` — "0.30 (central) | 0.235 m | +0.080 (~1.5×)"
  - `:65` — "0.38 (worst-case) | 0.155 m"
  - The formula subtracts the chassis `0.215` (≈ chassis half-diag 0.2135, per `:54`) AND the contact radius `r` — the same double-count flagged in `MEMORY.md:8`.
- **recommended_action:** Annotate the budget table (lines 52–65) as DOUBLE-COUNTED per the 2026-06-19 correction: correct clearance is `0.75 − r` (subtract the drone once), so r=0.30 clearance ≈ 0.45 m and r=0.38 ≈ 0.37 m, NOT 0.235/0.155. Add pointer to `MEMORY.md:8`. Note the SIM's `(0.75−r)−linf` is correct WHERE linf is a measured trajectory crossing offset; the bug is treating the fixed 0.215 chassis term as if it were that offset while ALSO subtracting r.
- **info_loss_risk:** MED — the radius-band reporting discipline and the linf crossing-offset concept are valuable; preserve them, only correct the budget arithmetic. Do not delete the table — annotate it.

### [HIGH] `index_vision_estimator.md:286-287` — retracted near-field-estimator pivot reads as current recommendation
- **location:** `memory/index_vision_estimator.md:286-287` (flagged by 2 agents; also reported as `contradiction`)
- **issue_type:** stale
- **severity:** high (one agent rated this med — kept at high to match the load-bearing index pointer)
- **evidence:**
  - `:286` — "The centering lever = a near-field GATE estimator (lower the 12 m floor — a gate ANCHOR, not odometry) + per-fix accuracy."
  - `:287` — "INVEST: ① inc8 per-fix accuracy + band fix-density ...; ② near-field gate estimator. NOT ESKF-for-coast..., NOT VIO, NOT speed."
  - CONTRADICTED by `MEMORY.md:8` — "RL IS the right tool — the 'pivot off RL to a near-field-gate estimator' recommendation is RETRACTED; next lever = RL to lift reach/pass-rate; vision = SUPPORT, do NOT lean on it."
- **recommended_action:** Re-rank the levers so RL reach/pass-rate is PRIMARY and the near-field gate estimator is demoted to secondary/insurance; note gate-4 is now understood as a REACH/PASS-RATE problem (gate off the racing line), not a sub-8 cm centering problem. Keep the VIO-is-wrong-lever reasoning (still valid).
- **info_loss_risk:** none — the near-field-estimator concept and the 12 m PnP-floor analysis remain valid as a candidate insurance arm; only the "this is THE lever" framing changes.

### [HIGH] `index_rl_training.md:40,50,52` — double-counted budgets (0.235/0.155) as the live inc8 SELECT criterion
- **location:** `memory/index_rl_training.md:40,50,52`
- **issue_type:** stale
- **severity:** high
- **evidence:**
  - `:40` — "Central planning radius = 0.30 m (budget 0.235 m); worst-case stress knob = 0.38 m (budget 0.155 m)."
  - `:50` — "SELECT on p90 gate: gate-4 SIMSTART in-plane p90 < 0.155 m @ r=0.38".
  - `:52` — "central = r=0.30 (budget 0.235); select at r=0.38 worst-case (budget 0.155)."
  - CONTRADICTED by `MEMORY.md:8` — "real clearance 0.37–0.47 m ... REAL bar ≈ σ_p0_lat ≲ 0.15."
- **recommended_action:** Correct the budgets to single-count clearance (r=0.30 → ~0.45 m, r=0.38 → ~0.37 m), OR re-frame the SELECT criterion against the corrected σ_p0_lat ≲ 0.15 bar. Add pointer to `MEMORY.md:8`. Note gate-4 is now judged a REACH/PASS-RATE problem, not a tighter-gate centering problem.
- **info_loss_risk:** MED — the p90-AND-p99 radius-band reporting discipline is sound and must be kept; only the budget magnitudes are wrong.

### [HIGH] `index_vision_estimator.md:285` (the closure-boundary origin) — see contradiction section above
- **location:** `memory/index_vision_estimator.md:285`
- **issue_type:** stale (duplicate surface of the contradiction-section entry; same line flagged by multiple agents under both labels)
- **severity:** high
- **evidence:** Same `:285` "CLOSURE BOUNDARY ... r=0.30 (B=0.235) CLOSES IFF σ_p0_lat ≲ 0.08 m" stale text; full evidence and action in the contradiction-section entry above. Consolidated here for completeness of the stale-by-issue-type listing.
- **recommended_action:** See the contradiction-section entry for `:285` (CORRECTION note + strikethrough + pointer to `MEMORY.md:8`).
- **info_loss_risk:** Low if the B=0.235/0.155/0.12 budget arithmetic and the achievable-σ derivation are kept; relabel the 0.08 conclusion as the corrected double-count.

### [HIGH] `project-fullstack-burn.md:17` — stale `std ≲ 0.08` as the operative burn GO metric
- **location:** `memory/project-fullstack-burn.md:17` (flagged by 2 agents)
- **issue_type:** stale
- **severity:** high
- **evidence:**
  - `:17` — "GO = bias-inclusive p99 within the gate margin AND std ≲ 0.08 (inc7 had a −0.215 m gate-4 lateral bias — centering must shrink it ...)."
  - CONTRADICTED by `MEMORY.md:8` — "REAL bar ≈ σ_p0_lat ≲ 0.15 (p99 ≲ 0.45) ... Measured σ_p0 0.15–0.20 / p99 0.37–0.40 = MARGINAL-PASSING ... NOT the NO-GO we mis-called vs the bad 0.08 bar." This is an ACTIVE execution doc → the wrong `std≲0.08` GO would mis-gate the next campaign.
- **recommended_action:** Update `std ≲ 0.08` → `std/σ_p0_lat ≲ 0.15 (p99 ≲ 0.45)`; keep the bias-inclusive p99-within-margin framing and the inc7 −0.215 m bias note (both still valid). The σ_p0 definition ("across-episode std + bias + p90/p99 via `rl/inc8_sigmap0_eval.py`" — NOT per-fix σ_lat, NOT floored estim_err) is correct and stays; only the threshold changes.
- **info_loss_risk:** none — the σ_p0 definition is a correct and important anti-footgun; change only the numeric threshold.

### [HIGH] `index_vision_estimator.md:80-86` — ε_vert framed as gate-4 BINDING / ESKF "co-equal margin lever"
- **location:** `memory/index_vision_estimator.md:80,82,84,86`
- **issue_type:** stale
- **severity:** med (kept in HIGH block per evidence weight; one agent rated med)
- **evidence:**
  - `:80` — "The binding factor = VERTICAL BORESIGHT BIAS ε_vert ≈ 0.215 m (~0.56° camera-pitch mismatch, CALIBRATABLE) + effective systematic attitude/accel bias, NOT the contact radius and NOT per-fix σ."
  - `:82` — "ESKF attitude-bias estimation + boresight calibration (co-equal margin levers — PROMOTED by δ_map finding)".
  - `:84` — "ESKF attitude-bias estimation (bias state) = CO-EQUAL MARGIN LEVER".
  - `:86` — "Closure rests on boresight calibration (ε_vert→0) + camera pointing/fix-density + attitude/accel-bias estimation + uncertainty-aware speed-down".
  - CONTRADICTED by `MEMORY.md:8` — "VIO / ESKF / velocity / speed ALL ruled out ... gate-4 is a REACH/PASS-RATE control problem ... NOT sub-8cm centering." The "binding factor / co-equal ESKF margin lever" framing was derived against the now-retracted 0.08 bar.
- **recommended_action:** Annotate `:80-86` with a SUPERSEDED banner → `MEMORY.md:8`: keep the ε_vert magnitude facts (0.215 m / ~0.56° raw, refined/deployed −0.25) but mark the "binding factor" / "ESKF co-equal margin lever" / "closure rests on attitude-bias estimation" conclusions as superseded — gate-4 is now reach/pass-rate; ESKF/attitude-bias ruled out as a centering lever; boresight bake retained for DEPLOY accuracy only. Cross-link to `src/racer/frames.py:58`.
- **info_loss_risk:** preserve the ε_vert measurement (0.215 m/0.56° raw pin; −0.25 m production value), the δ_map discriminator result, and the boresight-bake-for-deploy-accuracy fact — all valid. Do not delete; mark superseded.

### [HIGH] `inc8-deterministic-retrain-2026-06-19/REPORT.md` — NO-GO-vs-0.08 verdict + near-field-estimator next-lever (the report MEMORY re-judges)
- **location:** `handoff/inc8-deterministic-retrain-2026-06-19/REPORT.md:89,101,106,132-135,144,149-150` (flagged by 4 agents)
- **issue_type:** stale
- **severity:** high
- **evidence:**
  - `:101` — "2-axis σ_p0_lat ≈ 0.15–0.20 m = NO-GO vs 0.08 (lat_p99 0.37–0.40 vs 0.24)".
  - `:132` — "Gate-4 NOT closed by the look-at bet: 2-axis σ_p0_lat 0.16–0.20 m = NO-GO".
  - `:134-135` — "RECOMMENDED NEXT LEVER (not more look-at RL): the NEAR-FIELD-GATE ESTIMATOR".
  - `:144` (MEMORY-DELTA) — "Measured 2-axis gate-4 σ_p0 = 0.15–0.20 m = NO-GO vs 0.08".
  - `:149-150` — "NEXT LEVER = near-field-gate estimator ... NOT more look-at-gain RL."
  - CONTRADICTED by `MEMORY.md:8` — "Measured σ_p0 0.15–0.20 ... = MARGINAL-PASSING ... NOT the NO-GO we mis-called vs the bad 0.08 bar ... the 'pivot off RL to a near-field-gate estimator' recommendation is RETRACTED." `MEMORY.md:8` explicitly cites and overturns this report; correction commit `511e85c` never touched the file.
- **recommended_action:** Prepend a dated `SUPERSEDED 2026-06-19` banner at the top + at the CONCLUSION/MEMORY-DELTA pointing to `MEMORY.md:8` / `511e85c`: the 0.08 bar was a double-count; real bar σ_p0_lat ≲ 0.15 / p99 ≲ 0.45; 0.15–0.20 is MARGINAL-PASSING not NO-GO; the near-field-estimator recommendation is RETRACTED — RL stays the right tool (reach/pass-rate lever). Strike/annotate the overturned lines in place rather than editing the body. Do NOT alter the data tables.
- **info_loss_risk:** Must preserve (all referenced as KEEP by `MEMORY.md:8`): det reach 0.467; the σ_p0 0.15–0.20 tables; the noise-anneal/std-cap root cause (rc1 actor_logstd saturation at exp(2)=7.39, fix = std-ceiling clamp); the 5-run g_pitch bracket; the lever-1 std-ceiling-vs-anneal refinement. Only the GO/NO-GO verdict and the next-lever recommendation are superseded.

### [HIGH] inc8 deterministic-retrain result + gate-4 bar correction ABSENT from the sub-index layer (propagation gap appears as stale sub-indices)
- **location:** `memory/index_rl_training.md` (inc8 thread ends ~line 130, lines 117-123 dated 2026-06-14/15) and `memory/index_vision_estimator.md` (`§TERMINAL-LOCK`/closure-boundary)
- **issue_type:** stale / gap (listed in gap section too)
- **severity:** high
- **evidence:** Verified by grep — `0.467`, `det reach`, `noise-anneal`, `std-cap`, `deterministic-stability` appear in ZERO sub-index/topic files (only `MEMORY.md:8` and the handoff REPORT). `index_rl_training.md`'s inc8 thread ends at the band-pass/fix-driven NO-GO saga (lines 117-123) and never reaches the 2026-06-18 recenter-flew-3/3, the 2026-06-17 S4 warm-start, the 2026-06-19 deterministic retrain, OR the gate-4 bar correction. `git show --stat 511e85c` = `memory/MEMORY.md | 2 +-` (touched ONLY MEMORY.md).
- **recommended_action:** Propagate BOTH the corrected gate-4 bar (real ~0.15/p99 0.45; 0.08 was a double-count) AND the deterministic-retrain headline (std-cap lever = first det-flyable 2-axis inc8, det reach 0.467, rc1 logstd-saturation root cause, σ_p0 0.15–0.20 = MARGINAL) down into `index_rl_training.md` and `index_vision_estimator.md`. Thin-index discipline requires the sub-index to carry the CURRENT verdict, not lag MEMORY.md by ~5 events.
- **info_loss_risk:** none — additive propagation; preserves all existing saga detail while bringing the sub-index current. Without it, anyone following the `MEMORY.md [[index-rl-training]]` pointer lands on a stale NO-GO/0.08 picture.

### [MED] `index_rl_training.md:119` — stale `σ_p0 ≲0.08 (the GO)` in the S1 readout note
- **location:** `memory/index_rl_training.md:119` (flagged by 2 agents)
- **issue_type:** stale
- **severity:** med
- **evidence:** `:119` (S0/S1 log) — "READ S1: band_az LOW (sign ok) → estim_err-near-gate → σ_p0 ≲0.08 (the GO, NOT fix_rate) → line survives". CONTRADICTED by `MEMORY.md:8` — "REAL bar ≈ σ_p0_lat ≲ 0.15 (p99 ≲ 0.45)". File head (lines 1-66) carries NO mention of the 2026-06-19 correction or the deterministic-retrain result.
- **recommended_action:** Dated episodic log → annotate rather than rewrite. Update `≲0.08 (the GO)` → `≲0.15 (corrected bar)` and add a one-line 🚩 at the inc8 section head (or file head) pointing to `MEMORY.md:8`: the `σ_p0 ≲0.08` GO referenced throughout the S0–S5 log was a double-count; real bar ≲0.15.
- **info_loss_risk:** none — the historical S0/S1 reasoning stays intact; only the bar value is flagged superseded.

### [MED] `project_phase2_rl_vision_decisions.md:1531,1532,1545,1550,1552,1710` — double-count budget identity in the decisions log
- **location:** `memory/project_phase2_rl_vision_decisions.md:1531,1532,1545,1550,1552,1710`
- **issue_type:** stale
- **severity:** med
- **evidence:**
  - `:1532` — "Budget identity: `budget(r) = (0.75 − r) − 0.215`, where 0.215 m = radius-invariant gate-4 simstart crossing offset (back-out from d3: (0.75−0.38)−0.155; confirmed bit-for-bit)."
  - `:1550` — "0.30 (central) | 0.235 m | +0.080 (~1.5×)".
  - `:1710` — "central on 0.30 (budget 0.235 m), worst-case on 0.38 (budget 0.155 m)".
  - CONTRADICTED by `MEMORY.md:8` — "pass = offset < 0.75−r ... real clearance 0.37–0.47 m."
- **recommended_action:** Add a 2026-06-19 CORRECTION banner to the `§contact-radius-reconciliation`/budget-table sections (≈ lines 1531–1552, 1710) flagging the `(0.75−r)−0.215` identity as a double-count; cite `MEMORY.md:8` for the corrected clearance (0.37–0.47 m). Keep historical text (dated decision log); mark superseded.
- **info_loss_risk:** none — historical decision log; the crossing-offset linf concept is preserved for provenance.

### [MED] `project_rl_increment_history.md:748-750` — stale NO-GO-vs-0.08 + missing the 2026-06-19 result (canonical inc8 lineage log lags by 2 events)
- **location:** `memory/project_rl_increment_history.md:748-750` (also `:736`; flagged by 3 agents — overlaps the gap-section entry on the SAME ending)
- **issue_type:** stale
- **severity:** med
- **evidence:**
  - `:749` — "Provisional σ_p0 (seed0, trainreset, YAW-ONLY, 200 ep) = 0.177 m (bias −0.095, p99 0.375) → NO-GO vs 0.08".
  - `:750` — "Best-available FAITHFUL σ_p0 = yaw-only seed0 0.177 m (bias −0.095, p99 0.375) → NO-GO vs 0.08, pessimistic".
  - This is the durable §inc8 lineage `MEMORY.md` points to via `[[project-rl-increment-history]] §inc8`; it ENDS at the S5 yaw-injection-fix bullet and never received the deterministic-retrain (det reach 0.467) result NOR the bar correction. CONTRADICTED by `MEMORY.md:8` — "REAL bar ≈ σ_p0_lat ≲ 0.15 (p99 ≲ 0.45) ... 0.15–0.20 = MARGINAL-PASSING."
- **recommended_action:** Append a new dated §inc8 entry after `:750` capturing (1) the deterministic-stability retrain (root cause = saturated actor_logstd exp(2)=7.39; fix = std-ceiling clamp; FIRST det-flyable 2-axis inc8, gp1.0 det reach 0.467; merged `42ddb0b`); (2) the gate-4 bar correction (0.08 was a double-count; real bar ≲0.15; measured 0.15–0.20 = MARGINAL-PASSING; near-field pivot RETRACTED; `511e85c`). Re-contextualize the trailing `→ NO-GO vs 0.08` labels on `:749-750` as "judged against the now-CORRECTED bar (≲0.15/0.45) = MARGINAL ... the 0.08 target was a double-count".
- **info_loss_risk:** none — the 0.177 m measurement, the yaw-injection bug history, and the `_FLIP/_ACT_FLU_TO_FRD/_RATE_SIGN_LIVE` sign analysis are retained; only the 0.08 comparison is re-contextualized and the missing milestones added.

### [MED] `commander-handoff-2026-06-18.md:21,28` — stale GO rule `σ_p0_lat ≤ 0.08 AND lat_p99 ≤ 0.24` in the LIVE-flagged handoff
- **location:** `handoff/commander-handoff-2026-06-18.md:21,28` (also `:6,:38` for git state — see push-state entry; flagged by multiple agents)
- **issue_type:** stale
- **severity:** med (one agent rated low — file self-flags transient)
- **evidence:**
  - `:21` — "GO rule: σ_p0_lat ≤ 0.08 AND lat_p99 ≤ 0.24".
  - `:28` — "Best-available faithful σ_p0 = yaw-only 0.177 m → NO-GO vs 0.08 (pessimistic; pitch/elevation off)."
  - CONTRADICTED by `MEMORY.md:8` corrected bar ≲0.15. MITIGATION: file header (`:3`) says "This file is the transient 'what is happening RIGHT NOW' snapshot at retirement. Delete/supersede it once you've absorbed it." This handoff is flagged LIVE in `MEMORY.md` NOW, so its NO-GO-vs-0.08 reads as current truth.
- **recommended_action:** Add a one-line CORRECTION note at the top: "GO rule σ_p0_lat≤0.08/p99≤0.24 SUPERSEDED 2026-06-19 — that bar was a double-count; real bar ≲0.15/≲0.45 (see MEMORY.md:8)." Do not rewrite the body of a historical handoff. Alternatively archive per the file's own header once absorbed.
- **info_loss_risk:** none — content is a transient snapshot duplicated into `MEMORY.md` and topic files; safe to annotate or archive.

### [MED] `sigmap0-adroit-2026-06-19/REPORT.md:9-10,150-152,171,191` — NO-GO-vs-0.08 asserted as binding verdict
- **location:** `handoff/sigmap0-adroit-2026-06-19/REPORT.md:9-10,150-152,171,191` (flagged by 3 agents)
- **issue_type:** stale
- **severity:** med (one agent rated low)
- **evidence:**
  - `:9-10` — "The best faithful σ_p0 in hand is the yaw-only seed0 number 0.198 m → NO-GO vs 0.08 ... Gate-4 verdict: NO-DATA on 2-axis; NO-GO on the measurable yaw-only fallback".
  - `:150-152` — "σ_p0_lat = 0.198 m, σ_p0_vert = 0.191 m, lat_p99 = 0.399 → NO-GO vs 0.08 / 0.24 ... This is the binding gate-4 number today".
  - `:191` (MEMORY-DELTA) — "Binding gate-4 σ_p0 = 0.198 m (yaw/racing-line) = NO-GO vs 0.08".
  - CONTRADICTED by `MEMORY.md:8` — real bar ≲0.15/p99≲0.45; 0.198/p99 0.399 is within the corrected p99 envelope → MARGINAL not a clean NO-GO.
- **recommended_action:** Prepend a dated 🚩 SUPERSEDED banner → `MEMORY.md:8`: the "NO-GO vs 0.08" verdicts are retracted; 0.198 m / p99 0.399 = MARGINAL. Keep the instrument-validity result (yaw cross-check torch 0.1981 vs numpy 0.1768 AGREE), the g_pitch stability-cliff data (flyable ≤0.5 / crash ≥1.0), and the 5th-harness-bug fix unchanged.
- **info_loss_risk:** none — the instrument validation and the g_pitch sweep stability cliff are reusable findings; only the verdict is superseded.

### [MED] `reference_competition_materials.md:75` — "Q① resolved / estimator LOAD-BEARING" stated as settled
- **location:** `memory/reference_competition_materials.md:75`
- **issue_type:** stale
- **severity:** high (case-c agent rated high; consolidated here with its in-file contradiction at `:80`)
- **evidence:** `:75` (COWORK-1 intel, 2026-06-14) — "NO position on wire (§3.3/§4.3): ... NO LOCAL_POSITION_NED/GLOBAL_POSITION_INT/ODOMETRY → self-localize REQUIRED (estimator LOAD-BEARING; Q① resolved)." CONTRADICTED by canonical `MEMORY.md:18` (2026-06-17): "Official VQ1 STREAMS LOCAL_POSITION_NED + ODOMETRY → case-A ... VQ2 honest wire ... = THE resolver (decides case-C = LOAD-BEARING vs INSURANCE) — WAIT for it before concluding."
- **recommended_action:** Qualify `:75`: change "self-localize REQUIRED (estimator LOAD-BEARING; Q① resolved)" → "the SPEC §4.3 scored wire lists no position → DEPLOYED stack must self-localize (case-C) IF the spec holds; but our official VQ1 sim DID stream LOCAL_POSITION_NED+ODOMETRY (case-A) → whether case-C is LOAD-BEARING vs INSURANCE is NOT yet resolved — VQ2 honest wire is THE resolver (see MEMORY.md WIRE-SPEC + parked_backlog A1)." Add `[[pointer]]` to `reference_sim_interface.md §WIRE-SPEC DISCREPANCY`.
- **info_loss_risk:** none — the §4.3-spec fact (scored wire lists no position) is preserved; only the over-confident "resolved" verdict is softened to match `MEMORY.md:18`, `reference_sim_interface.md:64-67`, `parked_backlog.md:91`.

### [MED] `MEMORY.md:17` — "SELF-LOCALIZE REQUIRED → gate-relative pivot VALIDATED" reads as closed adjacent to the :18 correction
- **location:** `memory/MEMORY.md:17`
- **issue_type:** stale
- **severity:** med
- **evidence:** `:17` (COMPETITION INTEL, 2026-06-14) — "(2) NO position on the scored wire (§4.3 = HEARTBEAT/ATTITUDE/HIGHRES_IMU/TIMESYNC) → SELF-LOCALIZE REQUIRED → gate-relative pivot VALIDATED." Next bullet `:18` (2026-06-17) supersedes: "Official VQ1 STREAMS LOCAL_POSITION_NED + ODOMETRY → case-A ... VQ2 ... = THE resolver — WAIT for it before concluding."
- **recommended_action:** Edit `:17` to defer to `:18`: append "(NOTE: VQ1 actually streamed position = case-A; whether self-loc is load-bearing on the SCORED wire pends VQ2 — see next bullet.)" or soften "pivot VALIDATED" → "pivot JUSTIFIED for the spec'd scored wire (VQ2 confirms)." Keep both bullets but make `:17` visibly subordinate to `:18`.
- **info_loss_risk:** none — the spec §4.3 message list and the rationale for building gate-relative are retained; the correction is already present one line below at `:18`.

### [MED] `project_phase2_rl_vision_decisions.md:1365` — stale "~4.6–4.7 s honest band" lap bound (no local supersession pointer)
- **location:** `memory/project_phase2_rl_vision_decisions.md:1365` (section header `§PLANNING-TOGT-S2` at `:1357`)
- **issue_type:** stale
- **severity:** med
- **evidence:** `:1365` — "| 4.27 s TOGT planning-valid bound | FALSIFIED — linear-plant fiction | ~4.6–4.7 s honest band |". CANONICAL `index_rl_training.md:88` (DOCTRINE REVISION 2026-06-14, RATIFIED) — "Supersedes: the gap-decomposition '→ ~4.72 s honest bound' (inverted fiction) ... Honesty trajectory: 4.27/4.55 (linear) → 4.6–4.7 (inverted/TOGT) → ~8 s upright-feasible." `MEMORY.md:21` — "the old ~4.6 s 'bound' was the rate-infeasible TOGT optimum."
- **recommended_action:** Add a local supersession banner at the top of `§PLANNING-TOGT-S2` (`:1357`): "⚠️ SUPERSEDED 2026-06-14 by index_rl_training.md §DOCTRINE REVISION — the ~4.6–4.7 s 'honest band' is the FULL-ATTITUDE/TOGT inverted (rate-infeasible) optimum; the honest UPRIGHT lap ≈ 8 s. The corrected-aero feasibility-table physics in this section remain valid; only the 4.6–4.7 s lap-endpoint framing is superseded."
- **info_loss_risk:** none — the corrected-aero TOPP feasibility tables (tilt-vs-laptime, 1377-1383) and the falsification of the 4.27 s linear bound stay intact; only a pointer is added.

### [MED] `project_phase2_rl_vision_decisions.md:1395` — stale "4.72 s / 4.43 s honest endpoint" (no local supersession)
- **location:** `memory/project_phase2_rl_vision_decisions.md:1395`
- **issue_type:** stale
- **severity:** med
- **evidence:** `:1395` — "Sum = 30.58 s = 35.30 − 4.72 (verified). Honest endpoint = 4.72 s; contact-valid = 4.43 s (bound_free, gate-4 planning margin only +0.046 m — razor-thin)." CANONICAL `index_rl_training.md:88` — "→ ~8 s upright-feasible"; `build_reference_line.py:568` — "the fastest UPRIGHT profile ... lap ~8.5 s ... the absolute upright feasible edge ... is ~7.9 s."
- **recommended_action:** Annotate `:1395` inline: append "[SUPERSEDED 2026-06-14: 4.72 s endpoint is the inverted/rate-infeasible TOGT fiction; honest upright-feasible endpoint ≈ 8 s — see index-rl-training §DOCTRINE REVISION]". Covered by the `§PLANNING-TOGT-S2` banner if applied.
- **info_loss_risk:** none — the 35.30→9.76 closed rung and the gap-decomposition arithmetic are preserved; the inverted-bound endpoint is retained as a labeled (superseded) datum.

### [MED] VQ2 detector parked-backlog terminal decision not updated for the 2026-06-19 clean-ensemble supersession
- **location:** `memory/project_parked_backlog.md:35` (entry #30/#58, dated 2026-06-16)
- **issue_type:** stale
- **severity:** med
- **evidence:** `:35` — "Round-1 best.pt = CHAMPION on everything measurable. ... DECISION (commander, detector-lane): DEPLOY Round-1 (YOLO11s, ~4 ms real; TensorRT-INT8 on-target validate before commit). Aug recipe is correct-but-unadoptable-on-current-evidence (a hedge)". CANONICAL `MEMORY.md:15` (2026-06-19) — "VQ2 DETECTOR = CLEAN ENSEMBLE (282abb9, 2026-06-19; supersedes Round-1): beats champion on the 2 stable metrics — course 76→82% (p=0.0005), deployed 62→82%; win = a TRAINING-LOSS lever (inner-4 OKS + abs-pixel, cluster/vq2_precision_loss.py) NOT data ... 2-model ensemble, ~26 ms ... GO-LIVE HELD on inc8". `MEMORY.md` points to `[[project-parked-backlog]] #30/#58` but the backlog's last sub-entry terminates at the superseded "DEPLOY Round-1" decision (backlog file last touched `4c105c8`, pre-ensemble).
- **recommended_action:** Append a dated sub-bullet under #30/#58: "CLEAN ENSEMBLE (282abb9) beats champion (course 76→82% p=0.0005, deployed 62→82%); win is the TRAINING-LOSS lever (cluster/vq2_precision_loss.py), confirmed NOT data (confound-kill A/B p=1.0); 2-model yolo11s ~26 ms / measured p95 43.7 ms; deploy step-1 opt-in EnsembleGateDetector MERGED 06876bc; champion UNTOUCHED; GO-LIVE HELD on inc8." Re-mark the older "DEPLOY Round-1 / Round-1 best.pt = CHAMPION" lines as "SUPERSEDED 2026-06-19 → see clean ensemble sub-bullet".
- **info_loss_risk:** none — the superseded Round-1 ledger text is preserved; only a forward supersession note is added (full detail already in `MEMORY.md:15` and the VQ2 handoff REPORTs).

### [MED] `handoff/commander-handoff-2026-06-18.md:6,38` — stale "main local-only / do-not-push" retirement snapshot
- **location:** `handoff/commander-handoff-2026-06-18.md:6,38`
- **issue_type:** stale
- **severity:** med
- **evidence:** `:6` (TL;DR) — "`main` is local-only (ahead of origin by 50, not pushed), and a ShadowPC clone may be on its own `main` — do not push or rewrite history until that's reconciled." `:38` (Git state) — "`main` = `dceaeba`, LOCAL-ONLY, ahead of origin by 50, NOT pushed." Both contradict authoritative git state (origin/main == main == c5d60a0, 0/0; `dceaeba` is itself a now-superseded commit). File self-describes as a transient retirement snapshot ("Delete/supersede it once you've absorbed it").
- **recommended_action:** Per the file's own instruction, supersede/archive now that contents (incl. local-only main) are absorbed and main is pushed. First confirm all unique content (the σ_p0 saga ledger lines 24-35, resources-built list, deferred-cleanup notes) is captured in `MEMORY.md`/index files; then archive (move under an archive dir or mark RESOLVED at top). Do NOT silently delete until the σ_p0 saga + resources-built lists are verified banked in `[[project-rl-increment-history]] §inc8` and the index files.
- **info_loss_risk:** MEDIUM if deleted outright — holds the most detailed point-in-time σ_p0 saga ledger and the resources-built list. Bank first, then archive.

### [MED] `index_control_sim.md:68` — queued full-lap gate-4 recording pinned to 37 m/s (corrected to ~30 m/s)
- **location:** `memory/index_control_sim.md:68` (also identically-worded `project_parked_backlog.md:14` #4 and `:50` #29)
- **issue_type:** stale
- **severity:** low (grouped under med block; agent rated low)
- **evidence:** `:68` — "Remaining queued items (... full-lap ~37 m/s gate-4 recording): DEFERRED to post-merge phase." CANONICAL `MEMORY.md:21` / `index_rl_training.md:89` — binding gate-4 speed = ~30 m/s, NOT 37–55; `index_vision_estimator.md:287` — "Best gate-4 speed = 30 m/s".
- **recommended_action:** Change "~37 m/s gate-4 recording" → "~30 m/s gate-4 recording" in `:68`; same change applies to `project_parked_backlog.md:14` (#4) and `:50` (#29).
- **info_loss_risk:** none — the queued-item list and DEFERRED status are preserved; only the speed figure is corrected.

### [LOW] `index_rl_training.md:33` — gap-decomposition "→ ~4.72 s" leads with stale value (already carries inline SUPERSEDED tag)
- **location:** `memory/index_rl_training.md:33`
- **issue_type:** stale
- **severity:** low
- **evidence:** `:33` — "Gap decomposition: 35.3 s (VQ1) → 9.76 s (inc7) → 6.89 s (cone tax) → ~4.72 s. [🆕 SUPERSEDED 2026-06-14: the ~4.72 s is the INVERTED/rate-infeasible TOGT bound; UPRIGHT-feasible ≈ 8 s, gate-4 ~30 m/s — see §DOCTRINE REVISION.]". Internally consistent (self-marks SUPERSEDED) → NOT a contradiction, but the stale 4.72 s leads the line. Canonical anchor same file `:88` "→ ~8 s upright-feasible".
- **recommended_action:** Optional tidy: rewrite the waterfall terminus to "... → 6.89 s (cone tax) → ~8 s upright-feasible (NOT ~4.72 s — that was the inverted/rate-infeasible TOGT bound; gate-4 ~30 m/s)". Low priority — the inline tag already prevents misreading.
- **info_loss_risk:** none — the SUPERSEDED annotation and the historical datum are retained inline.

### [LOW] `project_phase2_rl_vision_decisions.md:1353` — inc8 estimator target pinned to ~37 m/s
- **location:** `memory/project_phase2_rl_vision_decisions.md:1353`
- **issue_type:** stale
- **severity:** low
- **evidence:** `:1353` ("Ordered integration plan (inc8)" item 6) — "Estimator: drive KF to <0.05 m 1-sigma at post-gate-3 ~37 m/s window. THE binding VQ2 validity risk; measure first." CANONICAL `index_rl_training.md:89` — "binding gate-4 speed = ~30 m/s, NOT 37–55 → ... the σ-gate's 37 m/s targets were over-pessimistic on the speed axis."
- **recommended_action:** Update "~37 m/s window" → "~30 m/s window" (or append "(corrected to ~30 m/s, 2026-06-14 — see index-rl-training §DOCTRINE REVISION)"). NOTE: the separate `<0.05 m` bar in this item is the double-counted σ_p0 closure bar corrected 2026-06-19 — flag for the σ_p0 audit (out of this claim's scope).
- **info_loss_risk:** none — only the speed figure is updated; the estimator-as-binding-risk content is preserved.

### [LOW] `index_control_sim.md:65` — latency-geometry illustration uses 37 m/s
- **location:** `memory/index_control_sim.md:65`
- **issue_type:** stale
- **severity:** low
- **evidence:** `:65` — "At 37 m/s: 115 ms = ~4.3 m ⇒ CONFIRMS RewindKF-as-default (covers CPU case)." Corrected binding gate-4 speed is ~30 m/s (`MEMORY.md:21`); at 30 m/s the figure would be ~3.45 m. The conclusion (RewindKF-as-default) is unchanged; 37 m/s here is a conservative worst-case illustration.
- **recommended_action:** Optional: note 37 m/s here is a conservative upper bound and the corrected binding gate-4 speed is ~30 m/s (~3.45 m at 115 ms). Do NOT alter the RewindKF conclusion. Lowest priority.
- **info_loss_risk:** none — the latency budget and RewindKF-default conclusion are preserved; only a clarifying note on the speed assumption is added.

### [LOW] `handoff/coast-drift-2026-06-15/REPORT.md:36,95,241-252` — root-source of the double-counted MARGIN formula + near-field lever
- **location:** `handoff/coast-drift-2026-06-15/REPORT.md:36,95,241-252`
- **issue_type:** stale
- **severity:** low
- **evidence:**
  - `:95` — "MARGIN(r) = W_EFF − r = (0.75 − r) − 0.215: ... r0.30→0.235, ... r0.38→0.155" (the double-count).
  - `:36` — "r=0.30 (B=0.235) closes IFF the terminal centering σ_p0 ≲ 0.075–0.08 m".
  - `:248-252` — "Centering lever = near-field GATE estimator (lower the 12 m floor) ... sig_p0_lat<=0.08 at 12 m."
  - CONTRADICTED by `MEMORY.md:8` — "margin_envelope.py subtracted the drone TWICE ... real clearance 0.37–0.47 m ... REAL bar ≈ σ_p0_lat ≲ 0.15."
- **recommended_action:** Prepend a dated 🚩 SUPERSEDED banner: the `MARGIN(r)=(0.75−r)−0.215` formula double-counts the drone radius; correct clearance is 0.75−r (0.37–0.47 m for r 0.28–0.38), so the σ_p0 ≲0.08 boundary and the near-field-estimator lever derived from it are superseded (see `MEMORY.md:8`). **KEEP** the coast-drift physics result (informed coast <0.02 m over 0.4 s = NOT the blocker) — explicitly preserved as canonical in `MEMORY.md:8`.
- **info_loss_risk:** none — the coast-drift core result is canonical and kept; only the margin-budget formula and the σ_p0/near-field conclusions built on it are superseded.

### [LOW] `handoff/boresight-closure-2026-06-14/REPORT.md:95` — same double-counted MARGIN formula
- **location:** `handoff/boresight-closure-2026-06-14/REPORT.md:95`
- **issue_type:** stale
- **severity:** low
- **evidence:** `:95` — "MARGIN(r) = W_EFF − r = (0.75 − r) − 0.215: r0.21→0.325 ... r0.30→0.235, r0.33→0.205, r0.38→0.155" — same double-counted formula. CONTRADICTED by `MEMORY.md:8` — "real clearance 0.37–0.47 m."
- **recommended_action:** Add a one-line 🚩 note at `:95` that `MARGIN(r)=(0.75−r)−0.215` double-counts the drone (see `MEMORY.md:8`); the boresight bake DELTA result (−0.25 removes +0.08–0.15 m off p99) is independent and stays valid.
- **info_loss_risk:** none — the boresight bake delta is a real, still-valid lever; only the margin-budget arithmetic is flagged.

### [LOW] `d3_margin_findings.md:50,80,157` + `closure/COLD_VERDICT.md:61-63` — design-doc origin of the 0.235/0.155 budgets
- **location:** `handoff/ultracode-gate-relative-pipeline-design-2026-06-13/d3_margin_findings.md:50,80,157` and `.../closure/COLD_VERDICT.md:61-63`
- **issue_type:** stale
- **severity:** low
- **evidence:** `d3_margin_findings.md:50` — "cold (case-C reality ...) | 0.154 | 0.235 | 0.322 | 37% | NO"; `:80/:157` use p90 0.235 / p99 0.32 as the gate-4 budget over-run; `COLD_VERDICT.md:61-63` table uses 0.235 as the r=0.30 margin. These 0.235/0.155 budgets are the double-counted MARGIN per `MEMORY.md:8`.
- **recommended_action:** Add a dated 🚩 note at the top of `d3_margin_findings.md` and `COLD_VERDICT.md`: the 0.235 (r=0.30) / 0.155 (r=0.38) margin budgets reflect the drone-double-count; corrected clearance is 0.75−r ≈ 0.37–0.47 m (see `MEMORY.md:8`). 2026-06-13 design docs → annotate-as-superseded, do not delete.
- **info_loss_risk:** none — the case-C cold-vs-warm estimator RMS/p90/p99 measurements are reusable; only the margin-budget the pass/fail verdict is compared against is wrong.

### [LOW] inc8 σ_p0 eval reports — NO-GO-vs-0.08 verdict (cluster, single index-level note suffices)
- **location:** `handoff/inc8-eval-lookat-2026-06-18/REPORT.md:127`; `handoff/inc8-eval-pitch-2026-06-18/REPORT.md:8,148`; `handoff/inc8-sigmap0-torch-2026-06-18/REPORT.md:85,105,156,178`
- **issue_type:** stale
- **severity:** low
- **evidence:**
  - `inc8-eval-lookat/REPORT.md:127` — "σ_p0 (seed 0, ... YAW-ONLY ...) = lat 0.177 ... → NO-GO vs 0.08".
  - `inc8-eval-pitch/REPORT.md:8` — "seed0 σ_p0_lat = 0.177 m → NO-GO vs 0.08"; `:148` — "p99 0.370 → NO-GO vs 0.08."
  - `inc8-sigmap0-torch/REPORT.md:85` — "GO rule identical to the numpy tool (σ_p0_lat ≤ 0.08 AND lat_p99 ≤ 0.24)"; `:105` — "reach 133/200 (0.67) ... NO-GO (0.177 > 0.08)".
  - CONTRADICTED by `MEMORY.md:8` — real bar ≲0.15/≲0.45; the 0.08 GO rule these reports apply is the double-counted bar.
- **recommended_action:** These are frozen dated eval artifacts. A single dated correction note in the parent memory index (`index_rl_training §inc8` or `project_rl_increment_history §inc8`) stating that ALL "NO-GO vs 0.08" verdicts in the 2026-06-18/19 inc8 σ_p0 reports are superseded (0.08 was a double-count; ~0.177 is MARGINAL-PASSING) suffices; no need to edit each report. Cite `MEMORY.md:8`. (If editing individually, prepend a top banner to each — "σ_p0 GO rule 0.08/0.24 SUPERSEDED 2026-06-19 (double-count); real bar ≲0.15/≲0.45".)
- **info_loss_risk:** none — measurements (0.177/0.198 yaw-only) and the torch-instrument design are preserved; verdicts corrected by a single index-level note covering the cluster.

### [LOW] `project_parked_backlog.md:78 (#70), :82 (#74)` — near-field-estimator framed as "the centering lever"
- **location:** `memory/project_parked_backlog.md:78` (#70) and `:82` (#74)
- **issue_type:** stale
- **severity:** low
- **evidence:** `:78` (#70) — "The centering lever = a near-field GATE estimator (lower the 12 m floor — a gate anchor), NOT odometry." `:82` (#74) frames "σ_p0 = the real binding term". CANONICAL `MEMORY.md:8` — gate-4 is a "REACH/PASS-RATE control problem ... NOT sub-8cm centering"; near-field-estimator pivot RETRACTED. The #70 framing is a VIO-vs-near-field comparison (correctly rules out VIO) so partly valid; its "the centering lever = near-field estimator" clause now over-states the estimator's priority.
- **recommended_action:** Add a one-line revive-trigger update to #70/#74 noting the gate-4 reframe (reach/pass-rate problem; RL primary; near-field estimator = support/insurance). Keep the VIO-wrong-lever reasoning and the KF-floor finding (#74) intact — both still valid.
- **info_loss_risk:** none — parked-conditional items; the VIO back-solve and σ_v0 trigger analysis are unchanged; only the priority caveat is added.

### [LOW] `COMMANDER.md:65` — Gen-4 lineage entry asserts unpushed main without a superseded marker
- **location:** `COMMANDER.md:65`
- **issue_type:** stale
- **severity:** low
- **evidence:** `:65` (Gen-4 LINEAGE, 2026-06-18) — "(c) retired mid-burn with TWO live goals ... and `main` local-only ahead of origin by 50 — handed off via `handoff/commander-handoff-2026-06-18.md` without disturbing either." Now false as present-tense fact — main == origin/main == c5d60a0 (0/0). Correctly framed as a dated historical lineage note (lower severity than `MEMORY.md:4`) but skimmable as still-current.
- **recommended_action:** Leave the historical narration intact (it correctly records the Gen-4 retirement state); if touched, append "(since reconciled + pushed; origin/main == main as of c5d60a0)". Do NOT delete — legitimate lineage record.
- **info_loss_risk:** none — this IS the canonical historical record of the Gen-4 retirement git state; preserve it, only add a forward-reference.

### [LOW] `margin_envelope.py:33,77,78,101,102` — see contradiction-section root entry
- **location:** `handoff/margin-closure-envelope-2026-06-14/margin_envelope.py:33,77,78,101,102`
- **issue_type:** stale (a second agent labeled the same `margin_envelope.py` double-count as `stale`/computational-module; see the HIGH contradiction-section root entry for full evidence and action)
- **severity:** med
- **evidence:** Same `W_EFF`/`margin_at` double-subtract code; re-derived envelope verdicts (e.g. `index_vision_estimator.md:296` "p99 0.297 vs MARGIN 0.235") are affected. Full evidence in the contradiction-section entry.
- **recommended_action:** See the contradiction-section root entry (note at top of `margin_envelope.py` or its REPORT.md; corrected pass criterion offset < 0.75 − r; do not silently re-run).
- **info_loss_risk:** MED — the 268-cell envelope sweep machinery and σ=0.10-vs-0.265 finding are valuable; preserve the sweep, flag the threshold.

---

## ISSUE TYPE: gap

### [HIGH] inc8 deterministic-retrain result + 2026-06-19 correction missing from the canonical inc8 lineage log AND sub-index layer
- **location:** `memory/project_rl_increment_history.md` (§inc8 ends ~line 750, 2026-06-18 `ec4cb03`) and `memory/index_rl_training.md` (inc8 thread ends ~line 130) / `memory/index_vision_estimator.md`
- **issue_type:** gap
- **severity:** high (med per the increment-history-only agent; raised to high given the sub-index gap compounds it)
- **evidence:** `project_rl_increment_history.md §inc8` (lines 733-752) ends at the 2026-06-18 "S5 σ_p0 EVAL — YAW-INJECTION BUG FIXED" entry (`:750`) whose tail reads "Best-available FAITHFUL σ_p0 = yaw-only seed0 0.177 m ... → NO-GO vs 0.08, pessimistic". The 2026-06-19 deterministic-retrain result (noise-anneal/std-cap, det reach 0.467, merged `42ddb0b`) and the 0.08-bar correction (`511e85c`) appear NOWHERE in this file (grep for `deterministic-retrain|noise-anneal|42ddb0b|0.467|511e85c` returns no hits). Grep also confirms `0.467`, `det reach`, `noise-anneal`, `std-cap`, `deterministic-stability` appear in ZERO sub-index/topic files (only `MEMORY.md:8` and the handoff REPORT). The canonical inc8 lineage log both ENDS on the stale "NO-GO vs 0.08" note AND omits the latest two milestones.
- **recommended_action:** Append a new §inc8 entry after `:750` capturing (1) the deterministic-stability retrain (root cause = saturated actor_logstd exp(2)=7.39; fix = std-ceiling clamp; FIRST det-flyable 2-axis inc8, gp1.0 det reach 0.467; merged `42ddb0b`); (2) the gate-4 bar correction (0.08 was a double-count; real bar ≲0.15; measured 0.15–0.20 = MARGINAL-PASSING; near-field pivot RETRACTED; `511e85c`). Propagate the same two facts down into `index_rl_training.md` and `index_vision_estimator.md §TERMINAL-LOCK`. Closes the log gap and removes the stale terminal "NO-GO vs 0.08" as the last word.
- **info_loss_risk:** none — additive; existing S0–S5 entries preserved. The deterministic-retrain detail currently lives only in `MEMORY.md:8` (dense) and the handoff REPORT, so logging it in the topic + sub-index files improves durability. (This entry subsumes the earlier-listed stale `project_rl_increment_history.md:748-750` and the stale-sub-index entries — same underlying propagation gap.)

---

## ISSUE TYPE: duplication

### [LOW] `MEMORY.md:9` and `:50` — near-identical OBS SIGN = +L blocks
- **location:** `memory/MEMORY.md:9` and `:50`
- **issue_type:** duplication
- **severity:** low
- **evidence:** `:9` (NOW/contract block) and `:50` (Cross-cutting footguns) are the SAME block verbatim except trivial wording ("The d1/d2", "The future C2"): "OBS SIGN = +L (CORRECT IN CODE; obs_from_zup:348 = R_w2g@(gate−pos), localization:86). ... d1/d2 spec PROSE 'estimator delivers −L' is wrong ... pinned by tests/test_obs_sign_faithfulness.py (+L 4.77e-7 / −L control breaks 24 m). Future C2 estimator→obs MUST deliver +L."
- **recommended_action:** Collapse to ONE. Keep the `:50` copy (Cross-cutting footguns = the durable home for a forget=disaster invariant); replace the `:9` copy with a one-line pointer ("🚩 OBS SIGN = +L — see Cross-cutting footguns below"). No content change; both copies are correct (+L).
- **info_loss_risk:** none — content is identical and fully preserved in the retained copy plus `tests/test_obs_sign_faithfulness.py` and code `obs_from_zup:348` / `localization.py:86`.

### [LOW] case-C "build gate-relative regardless; Q① = load-bearing vs insurance" restated in three layers (CONSISTENT)
- **location:** `memory/project_phase2_rl_vision_decisions.md:1645-1646` vs `memory/MEMORY.md:7` vs `memory/index_vision_estimator.md:9`
- **issue_type:** duplication
- **severity:** low
- **evidence:** All three AGREE (no contradiction): `project_phase2_rl_vision_decisions.md:1646` — "CURRENT: Build gate-relative regardless. Q① only decides whether it's load-bearing (case C, no pose streamed) or free insurance (case A/B, pose given)."; `MEMORY.md:7` — "Build gate-relative REGARDLESS (Q① only decides load-bearing-vs-insurance)."; `index_vision_estimator.md:9` — "ORGANIZER-PIVOT: build gate-relative REGARDLESS — Q① = load-bearing vs free-insurance only." Flagged only as a consolidation opportunity (same routing rule restated at three index layers).
- **recommended_action:** Keep `MEMORY.md:7` and `index_vision_estimator.md:9` as thin-index pointers; treat `project_phase2_rl_vision_decisions.md:1645-1646` as the canonical detailed home. No correctness change. Optionally add an explicit `[[pointer]]` from `index_vision_estimator.md:9` to the decisions §the-master-gate block so the three stay synced on future edits.
- **info_loss_risk:** none — all three mutually consistent; do NOT prune (they live at different index layers by design).

### [LOW] obs_dim=20 (d5 layout) restated in MEMORY/index/decisions (CONSISTENT, no contradiction)
- **location:** `memory/MEMORY.md:11`; `memory/index_vision_estimator.md:106` (block 105-111); `memory/project_phase2_rl_vision_decisions.md:1617` (block 1612-1631); `tests/test_deploy_obs20.py`; `handoff/data-staging-2026-06-17/MANIFEST.md:16,148`
- **issue_type:** duplication
- **severity:** low
- **evidence:** `obs_dim` stated uniformly as 20 with [0:17] bit-exact + [17:20] confidence triple; every "17-dim" reference is correctly scoped to the inc7/VQ1 case-A path (NOT asserted as the current inc8 contract). `tests/test_deploy_obs20.py:16` — "inc7 17-dim path BYTE-IDENTICAL (the JUDGED VQ1 path -- 20-dim is strictly opt-in)"; `MANIFEST.md:148` — "obs_dim:20 inc8:true r5_arm:A; actor/critic both 20-dim". No location asserts obs_dim=17 as current inc8 truth; no −L-as-current anywhere.
- **recommended_action:** No correctness change required. If pruning for thinness, keep the topic-file copy (`project_phase2_rl_vision_decisions.md §OBS-CONTRACT`, line 1612+) as SSOT and reduce the index/MEMORY copies to pointers — housekeeping only.
- **info_loss_risk:** none — restatements mutually consistent; SSOT copy preserves full detail.

### [LOW] Contact-radius doctrine restated across canonical locations (CONSISTENT, no contradiction)
- **location:** `memory/MEMORY.md:51`; `memory/index_vision_estimator.md:10,52,56,80`; `memory/index_rl_training.md:40`; `memory/project_phase2_rl_vision_decisions.md:1531-1552`; `src/.../peregrine_racing.py:153-169`; `handoff/body-contact-reconcile-2026-06-13/CORRECTED_MARGIN.md`; `empirical_radius.md`
- **issue_type:** duplication
- **severity:** low
- **evidence:** The radius reconciliation (0.2135 m chassis geometry cap / 3D half-diag; 0.30 m central reporting radius, band 0.26-0.33; 0.38 m worst-case stress knob; 0.198 flat half-diag floor; near-level gate-0 r_eff 0.18-0.20 flagged WRONG posture) is stated CONSISTENTLY. `MEMORY.md:51` matches `CORRECTED_MARGIN.md:96` ("Reconciled best estimate = 0.30 m central, band 0.26-0.33 m, worst-case tail 0.38 m") and `peregrine_racing.py:163` ("Doctrine band [0.28, 0.38] m"). No value contradictions among these. (Positive confirmation reported per the prime directive.)
- **recommended_action:** No change required. MINOR reader-confusion note: `MEMORY.md:52` uses 0.215 for the boresight ε_vert, numerically near 0.2135 (chassis geom) and 0.215 (linf0) — THREE distinct quantities that happen to be ~equal; not an error, but worth a parenthetical if the file is ever edited.
- **info_loss_risk:** none — confirmation only.

### [LOW] look-at launch config (signs/values) cross-source check (CONSISTENT, no defect)
- **location:** `rl/contact_true_eval.py:72-73` vs `memory/MEMORY.md:8` vs `rl/peregrine_inc8_recenter.sbatch:111` vs `rl/peregrine_inc8_warmstart.sbatch` + `tests/test_inc8_lookat.py:153-155,213`
- **issue_type:** duplication
- **severity:** low
- **evidence:** CONSISTENT (no sign/value contradiction). `contact_true_eval.py:72-73` — `LOOKAT_G_YAW_DEFAULT = -3.0 # empirical S0 yaw sign` / `LOOKAT_G_PITCH_DEFAULT = 3.0 # empirical S2 pitch sign (opposite)`. `MEMORY.md:8` launch config matches; `recenter sbatch:111` override matches; `test_inc8_lookat.py:153` "validated gain is NEGATIVE (g_yaw=-3.0)". NOTE the sbatch COMMON block default is `+env.lookat_g_yaw=3.0` (POSITIVE), DELIBERATELY overridden to -3.0 by the double-plus `++` — documented in-file (recenter sbatch:97 vs :111; comments :106-107) as the SIGN FOOTGUN, not an error.
- **recommended_action:** No change required — internally consistent across code, tests, MEMORY.md, and the sbatch overrides. The COMMON `+3.0` default vs `++ -3.0` override is intentional and documented; retain the footgun annotation. Recorded to note the assigned config-consistency check passed.
- **info_loss_risk:** none — informational confirmation only.

### [LOW] boresight ε_vert magnitudes (0.215 m vs −0.25 m) coexist and are EXPLICITLY reconciled (no contradiction)
- **location:** `memory/index_vision_estimator.md:239,251,258` vs `src/racer/frames.py:49-58`
- **issue_type:** duplication
- **severity:** low
- **evidence:** Two ε_vert magnitudes coexist and are EXPLICITLY reconciled (NOT a contradiction): `index_vision_estimator.md:239` — "ε MAGNITUDE PINNED = −0.25 m @ 23 m ≈ −0.61° ... (Consistent with the prior 0.56°/0.215 m pin; the dedicated head-on gate-0 reads the higher end)."; `:251` — "FORM = METRIC ... ε_vert = −0.25 m (refined/production)/−0.27 m (raw)." Code `frames.py:49-50/58` — "RESOLVED eps_vert = METRIC, range-INDEPENDENT, magnitude -0.25 m ... BORESIGHT = BoresightCorrection(vert_offset_m=-0.25)". The 0.215 m/0.56° (early raw pin) and −0.25 m/−0.61° (refined, DEPLOYED) are a documented refinement of the same quantity.
- **recommended_action:** No correction needed — values internally consistent and match deployed code (`vert_offset_m=-0.25`). Optional clarity: in the SHADOWVISION/binding-factor bullets (`index_vision_estimator.md:80,:195`) that still lead with "0.215 m", add "(early raw pin; production bake = −0.25 m, src/racer/frames.py:58)".
- **info_loss_risk:** none — flagged only to confirm the two magnitudes are a reconciled refinement; both values and provenance retained.

---

## Notes on dedup

- The `index_vision_estimator.md:285` closure-boundary line was flagged by 3 agents under both `contradiction` and `stale` labels — merged into ONE primary entry (contradiction section) with a cross-reference in the stale section.
- The `inc8-deterministic-retrain-2026-06-19/REPORT.md` NO-GO/near-field verdict was flagged by 4 agents at overlapping line sets (89/101/106/132-135/144/149-150) — merged into ONE entry preserving every cited line.
- The `MEMORY.md:8` internal self-contradiction (retract head / re-assert tail) was flagged by 3 agents (severities med/med/low) — merged into ONE contradiction entry at the higher (med) severity.
- The `margin_envelope.py` double-count was flagged by 2 agents (one as `contradiction` = the root, one as `stale` = computational module) — kept as the primary HIGH contradiction entry with a cross-reference stub in the stale section.
- The stale `project_rl_increment_history.md:748-750` (stale) and the inc8/sub-index propagation gap (gap) describe the SAME ending-on-stale-NO-GO + missing-milestones condition from two angles — the gap-section entry is primary and subsumes the stale duplicate (noted inline).
- `project-fullstack-burn.md:17`, `index_vision_estimator.md:286-287`, and `reference_competition_materials.md:75` were each flagged by 2 agents — merged, preserving the union of evidence and actions.
- All four "CONSISTENT / no-defect" confirmations (contact-radius doctrine, look-at config, obs_dim=20, ε_vert reconciliation, case-C three-layer restatement) are retained under `duplication` per the prime directive to record positive confirmations.
