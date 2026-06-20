# Cleanup Audit Appendix — Area: Completeness follow-ups (bucket: `gap`)

**Date:** 2026-06-19 · **Audit root (canonical):** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr` @ `main` `c5d60a0` (HEAD verified `c5d60a09...`). **Live working tree:** `C:/Users/Fengy/Downloads/Projects/Anduril` @ branch `inc8-deterministic-retrain` `05809ba`.

## Area state (headline)
This area covers two intertwined classes of completeness issues. **(1) A premise correction for the follow-up itself:** the audit root and the live branch DIVERGED at merge-base `387c4f3` — the live branch is **6 ahead / 7 behind** main (a divergent *older* sibling, NOT a newer descendant). The follow-up's worry that "6 live commits superseded the audit's view" has the diff direction INVERTED: it is `main @ c5d60a0` that already holds the 2026-06-19 gate-4 correction (`511e85c`), the VQ2 ensemble, and the corrected MEMORY; the live branch holds the *stale pre-correction* state. **No audit finding computed against `c5d60a0` is invalidated.** **(2) The actual gap/staleness backlog on the canonical tree:** the 2026-06-19 gate-4 σ_p0 bar correction (0.08 → ~0.15) has NOT yet propagated into ~5 code instruments and ~15 memory/handoff locations; the artifact-pipe "NEVER git-add binaries" rule is unenforced (~38 MB of tracked zips/PNGs/tfevents/npz in `handoff/` + retired-increment checkpoints/launchers) with no `.gitignore` coverage; several dangling `§`-pointers, stale README/docs, a drifted `run_parity.sh` md5, and a now-resolved `fly_rl.py` footgun round it out. The prime directive for every item below: **propagate/banner/re-home, do NOT delete** — preservation is paramount, and historical dated records get forward-pointers, not rewrites.

---

# Premise correction (audit-vs-live divergence) — issue_type: gap / stale / contradiction

> These six findings (agent `gap-r1-0`) collectively REFUTE the follow-up's framing. They are grouped here because they share one root: the diff direction between `main @ c5d60a0` and live `inc8-deterministic-retrain @ 05809ba` is INVERTED in the follow-up prompt. Net verdict: NO audit finding is stale; the stale artifacts are all on the LIVE branch (outside the audit root, correctly NOT read).

## [gap · high] Audit-root-vs-live diff direction (premise correction)
- **Location:** `git diff --stat main..inc8-deterministic-retrain` (run from `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr`)
- **Evidence:** `git merge-base main inc8-deterministic-retrain = 387c4f3` (NOT main's tip). `rev-list`: `main..inc8` = 6 (live ahead) AND `inc8..main` = 7 (live BEHIND) → the branches DIVERGED at `387c4f3`; the live branch does NOT contain the 7 commits on main (incl. the 2026-06-19 gate-4 correction `511e85c`). Diffstat: `detector.py |125 --`, `tests/test_ensemble_detector.py |309 ---`, `handoff/vq2-ensemble-support-2026-06-19/REPORT.md |83 --`, `.../measure_latency.py |67 --`, `memory/MEMORY.md |4 +-`, `COMMANDER.md |2 -`. The `-` lines are content that exists on MAIN (`c5d60a0`) and is ABSENT on the live branch.
- **Recommended action:** Treat `main @ c5d60a0` as the CANONICAL, NEWER, corrected state and live `inc8-deterministic-retrain @ 05809ba` as a DIVERGENT, OLDER sibling (6 ahead / 7 behind). The follow-up's framing ("those 6 commits DELETE the ensemble / bump the sentinel / REWRITE the bar 0.08→0.15, therefore the audit reasoned about a superseded state") has the diff direction INVERTED: it is the LIVE branch that holds the stale/pre-correction state, and MAIN that already holds the corrections. The audit was computed against the canonical commit; its findings about the canonical project are NOT invalidated.
- **info_loss_risk:** none

## [stale · high] Claim 1 — VQ2 ensemble detector presence
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/src/racer/vision/detector.py:189` (class `EnsembleGateDetector`); `tests/test_ensemble_detector.py` (whole file)
- **Evidence:** On AUDIT ROOT (`main @ c5d60a0`): `detector.py` = 274 lines, `class EnsembleGateDetector:` at line 189 plus `_dedup` (line 233) / `_fuse` (line 258) and the `++` opt-in route at `GateDetector.load:166`; `tests/test_ensemble_detector.py` PRESENT (309 lines). On LIVE branch (`inc8-deterministic-retrain @ 05809ba`): `detector.py` = 151 lines, `EnsembleGateDetector` REMOVED (class/`_dedup`/`_fuse`/`++` route gone); `git ls-tree inc8-deterministic-retrain -- tests/test_ensemble_detector.py` returns EMPTY → the test file is DELETED on the live branch.
- **Recommended action:** ANSWER to claim 1: On the LIVE branch the ensemble is DELETED (both `EnsembleGateDetector` and `test_ensemble_detector.py`). But the audit's `code:src` finding (saw `detector.py` at 274 incl. ensemble) and `code:tests+greengate` finding (saw `test_ensemble_detector` still present) are CORRECT for the audit's canonical root (`main @ c5d60a0`) where they DO exist. No correction needed to the audit's findings; scope any ensemble-pruning/keeping recommendation explicitly to main, and note the live branch already removed it (a divergence to reconcile at merge, not an audit error).
- **info_loss_risk:** none

## [contradiction · high] Claim 2 — green_gate test-count sentinel (1077 bump REFUTED)
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/scripts/green_gate.py:67`
- **Evidence:** PROMPT CLAIM: "bump the green_gate sentinel from 933 to 1077". ACTUAL: `git diff main..inc8-deterministic-retrain -- scripts/green_gate.py` returns EMPTY (zero diff). `green_gate.py:67` reads `DEFAULT_BASELINE = 933` on BOTH `main @ c5d60a0` AND `inc8 @ 05809ba` (verified via `git show inc8:scripts/green_gate.py`). grep for `1077` in `scripts/` and `memory/` on BOTH trees returns NOTHING. The `1077` appears ONLY in commit message `fe31d62` ("inc8 detstab: green_gate GREEN (1077 passed, invariants clean)") = a RUNTIME pytest pass-count, not the hardcoded constant. `green_gate.py:64-66` comment confirms `DEFAULT_BASELINE=933` was "measured 2026-06-17". *(Auditor re-confirmed on audit root: `green_gate.py` reads `DEFAULT_BASELINE = 933`.)*
- **Recommended action:** ANSWER to claim 2: The sentinel is 933 on BOTH branches; it was NOT bumped to 1077. The audit's `code:tests+greengate` finding ("933 sentinel") STILL HOLDS on the live branch and on main. The follow-up's assertion that the live commits bumped the sentinel is FALSE (it confused a one-off green_gate runtime pass count in a commit message with the `DEFAULT_BASELINE` constant). No audit finding is stale on this point. **NOTE:** finding `gap-r2-1` ("README staleness") separately recommends README cite "947" as the canonical count per MEMORY.md — `green_gate.py`'s on-disk sentinel is 933 and is the SSOT; 947 is the MEMORY-stated collection count. This 933-vs-947 discrepancy is itself a minor stale/gap to reconcile (MEMORY says "bump to 947" but the constant is still 933).
- **info_loss_risk:** none

## [stale · high] Claim 3 — gate-4 σ_p0 bar (MEMORY.md), direction INVERTED
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/memory/MEMORY.md:8`
- **Evidence:** AUDIT ROOT (`main @ c5d60a0`) `MEMORY.md:8` holds the CORRECTED bar: "GATE-4 BAR CORRECTED (2026-06-19, Fengyou caught a real error): the σ_p0 ≤ 0.08 closure bar was a DOUBLE-COUNT ... REAL bar ≈ σ_p0_lat ≤ 0.15 (p99 ≤ 0.45) ... Measured σ_p0 0.15-0.20 / p99 0.37-0.40 = MARGINAL-PASSING ... NOT the NO-GO we mis-called vs the bad 0.08 bar ... RL IS the right tool — the pivot off RL to a near-field-gate estimator recommendation is RETRACTED". LIVE branch (`inc8 @ 05809ba`) `MEMORY.md:8` holds the OLD bar: "GATE-4 CLOSURE REDUCES TO ONE NUMBER ... terminal CENTERING σ_p0_lat ≤ 0.08 m ... BINDING GATE-4 σ_p0 = 0.198 m (yaw/racing-line) = NO-GO vs 0.08 ... 2nd lever = near-field GATE estimator". The diff (−corrected/+old) confirms main has the correction; the live branch never received it.
- **Recommended action:** ANSWER to claim 3: On the LIVE branch it is the OLD "≤0.08 NO-GO" bar; on the AUDIT ROOT (main) it is the CORRECTED "~0.15 marginal-passing, RL stays". The audit's `x:gate4-sigma-bar` cross-check ran against main's `MEMORY.md`, which ALREADY holds the correction → the audit cross-checked against the CORRECT (corrected) text, contrary to the follow-up's worry that it "cross-checked against a MEMORY.md that still held the OLD 0.08 bar". The audit finding is NOT stale. Separately the live-branch `MEMORY.md:8` and its `handoff/inc8-deterministic-retrain-2026-06-19/REPORT.md` commit subject ("gate-4 NO-GO definitive, next lever = near-field estimator") ARE the stale OLD view; flag them as branch-local staleness to reconcile when the live branch is merged/rebased onto main — do NOT edit (read-only).
- **info_loss_risk:** none — the corrected analysis is preserved verbatim on `main @ c5d60a0` `MEMORY.md:8`; the live-branch older text is preserved in git history at `05809ba` and in `handoff/inc8-deterministic-retrain-2026-06-19/REPORT.md`.

## [stale · med] Claim 4 — GO-LIVE framing (MEMORY.md), direction INVERTED
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/memory/MEMORY.md:15`
- **Evidence:** AUDIT ROOT (`main @ c5d60a0`) `MEMORY.md:15`: "GO-LIVE HELD on inc8 (no case-C flight controller yet → closed-loop flight validation BLOCKED until inc8 flies on its mean; inc7 is case-A/brittle). Deploy STEP-1 DONE (06876bc): opt-in EnsembleGateDetector ...". LIVE branch (`inc8 @ 05809ba`) `MEMORY.md:15`: "GO-LIVE GATE = closed-loop flight validation (AUTHORIZED 2026-06-19, sim/no-risk) → then OPT-IN ensemble support (VQ1 BYTE-ID) + point at weights." Also `COMMANDER.md` diff `main..inc8` REMOVES a Gen-5 lesson on main ("Verify a validation PREREQUISITES exist before authorizing it ... I authorized a VQ2 closed-loop FLIGHT validation ... without checking that a flight controller for that loop exists; it doesn't yet ... Fengyou caught it") — i.e. main RETRACTED the "AUTHORIZED" framing and recorded the retraction lesson; the live branch predates that retraction.
- **Recommended action:** ANSWER to claim 4: On the LIVE branch the framing is "GO-LIVE GATE = closed-loop flight AUTHORIZED"; on the AUDIT ROOT (main) the framing is the LATER "GO-LIVE HELD on inc8 / Deploy STEP-1 DONE: opt-in EnsembleGateDetector", and `COMMANDER.md` on main carries the Gen-5 lesson explaining the authorization was RETRACTED because the flight-controller prerequisite did not exist. So main's "HELD" is the newer, corrected position and the live branch's "AUTHORIZED" is the older one. The audit's `x:vq2-detector` finding was reasoning about main's tree (ensemble exists, GO-LIVE HELD) — the canonical state — so it is NOT invalidated. Treat the live-branch "AUTHORIZED" line as branch-local staleness for merge-time reconciliation; do not edit.
- **info_loss_risk:** none — main's HELD framing + the `COMMANDER.md` Gen-5 retraction lesson preserve the corrected reasoning; the live-branch AUTHORIZED line persists in git at `05809ba`.

## [gap · high] Net verdict on whether the audit reasoned about a superseded state
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr` (audit root) vs the live branch `inc8-deterministic-retrain @ 05809ba`
- **Evidence:** The audit root IS `main @ c5d60a0` (HEAD verified = `c5d60a09...`). The follow-up's claimed sequence ("6 commits ahead that DELETE ensemble / bump sentinel / rewrite bar to corrected") is contradicted by the data: (a) ensemble delete = real but it's MAIN that has the ensemble, live that deleted it; (b) sentinel bump = FALSE (both 933, `green_gate.py` zero diff); (c) bar rewrite = INVERTED (main has corrected, live has old 0.08). The live branch is 6 ahead / 7 behind → a divergent sibling, NOT a descendant carrying newer truth.
- **Recommended action:** ANSWER to the close-out question: NO audit finding computed against `c5d60a0` is rendered stale by `05809ba`, because `05809ba` is an OLDER divergent branch, not a newer superseding state. The four findings the follow-up worried about (`code:src`, `code:tests+greengate`, `x:vq2-detector`, `x:gate4-sigma-bar`) were all computed against the canonical, corrected main and remain valid for main. The genuinely stale artifacts are on the LIVE branch (its `MEMORY.md:8` 0.08-NO-GO bar, `MEMORY.md:15` AUTHORIZED framing, deleted ensemble, missing `COMMANDER.md` Gen-5 lesson, and `handoff/inc8-deterministic-retrain-2026-06-19/REPORT.md` "NO-GO definitive / near-field estimator" conclusion) — pre-correction, to be reconciled (rebase/merge onto main) before the live branch is trusted, but OUTSIDE the audit root and correctly NOT read.
- **info_loss_risk:** none

---

# Gate-4 σ_p0 bar correction not yet propagated — issue_type: contradiction / stale (the 0.08 → ~0.15 double-count fix)

> The 2026-06-19 correction (`511e85c`, ancestor of audit-root HEAD) is landed in `MEMORY.md:8` but has NOT propagated into the executable instruments or several memory/handoff docs. These are the highest-leverage *code* items in this area: the hardcoded 0.08 default emits a false NO-GO for a marginal-passing policy. Auditor re-confirmed the 0.08 default is live in BOTH eval tools (see evidence). Prime directive: edit thresholds/prose in code; banner (do not rewrite) dated historical reports.

## [contradiction · high] inc8 σ_p0 GO-rule — stale 0.08 bar in CODE (numpy eval)
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/rl/inc8_sigmap0_eval.py:5, :27, :174, :199`
- **Evidence (auditor re-confirmed):** `:5` ``sigma_p0_lat <= ~0.08 m``; `:27` `GO RULE.  sigma_p0_lat <= 0.08 m AND |lateral miss| p99 ~= 3*sigma_p0 <= ~0.24 m`; `:174` `def _go_verdict(m: dict, sigma_target: float = 0.08)`; `:199` `ap.add_argument("--sigma-target", type=float, default=0.08)`. CONTRADICTS the landed correction (`MEMORY.md`, `511e85c` ancestor of HEAD `c5d60a0`): "the σ_p0 0.08 m closure bar was a DOUBLE-COUNT ... REAL bar ≈ σ_p0_lat ≤ 0.15 (p99 ≤ 0.45). Measured σ_p0 0.15-0.20 = MARGINAL-PASSING." The hardcoded 0.08 default emits a false NO-GO for a marginal-passing policy.
- **Recommended action:** Update default `sigma_target`/`--sigma-target` from 0.08 to 0.15 and the p99 bar from ~0.24 to ~0.45; rewrite the GO-RULE docstring (lines 4-5, 27-28) to cite the corrected clearance (real clearance 0.37-0.47 m; bar σ_p0_lat ~0.15 / p99 ~0.45 per `511e85c`). Do NOT delete the tool.
- **info_loss_risk:** none — only a numeric threshold and prose change; the measurement machinery and #37-provisional caveat are preserved.

## [contradiction · high] inc8 σ_p0 GO-rule — stale 0.08 bar in the TORCH eval (LIVE deliverable instrument)
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/rl/inc8_sigmap0_torch_eval.py:5, :47, :223, :292`
- **Evidence (auditor re-confirmed):** `:5` ``sigma_p0_lat <= ~0.08 m``; `:47` `GO RULE.  sigma_p0_lat <= 0.08 m AND |lateral miss| p99 (~3*sigma_p0) <= ~0.24 m`; `:223` `def _go_verdict(m: dict, sigma_target: float = 0.08, ...)`; `:292` `ap.add_argument("--sigma-target", type=float, default=0.08)`. This is the LIVE instrument wired into `inc8_select_ckpt.sbatch` and `peregrine_inc8_detstab.sbatch`. Same contradiction with `511e85c`'s corrected bar (real ~0.15 / p99 ~0.45).
- **Recommended action:** Update the torch eval default `sigma_target`/`--sigma-target` to 0.15 and p99 bar to ~0.45; rewrite the GO-RULE docstring (lines 4-5, 47-48) per the `511e85c` correction. Keep the `MIN_ENSEMBLE` / reach-rate NO-DATA logic unchanged. **Fix BOTH eval tools together** so the numpy cross-check and torch deliverable stay consistent (see duplication finding below).
- **info_loss_risk:** none — threshold + prose only; the cross-check and ensemble-statistics logic are untouched.

## [contradiction · high] gate-4 σ_p0 closure bar — stale double-counted bar in decisions topic file
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/memory/project_phase2_rl_vision_decisions.md:1507`
- **Evidence:** STALE (§ESTIMATOR-RACESPEED, dated 2026-06-13): "Achievable in-plane miss: **0.11-0.21 m RMS — CONDITIONAL-GO on the 0.155 m contact margin** at 37 m/s. Does NOT clear the strict 0.05 m bar (that needs per-fix lateral ≤ ~0.08 m)." (line 1507) — CONTRADICTS corrected SSOT `MEMORY.md:8`: "the σ_p0 ≤ 0.08 'closure bar' was a DOUBLE-COUNT ... REAL bar ~ σ_p0_lat ≤ 0.15 (p99 ≤ 0.45) ... real clearance 0.37-0.47 m ... Measured σ_p0 0.15-0.20 = MARGINAL-PASSING, NOT the NO-GO we mis-called vs the bad 0.08 bar." The "0.05 m strict bar" and "per-fix lateral ≤ ~0.08 m" here are exactly the double-counted analysis bar that was retracted.
- **Recommended action:** Do NOT delete (load-bearing referenced section). Add a dated supersession rider immediately under this line: "SUPERSEDED 2026-06-19 (gate-4 bar correction, `MEMORY.md` / commit `511e85c`): the 0.05 m / 0.08 m per-fix-lateral bar was a double-count (`margin_envelope.py` subtracted the drone radius twice). Real gate-4 clearance is 0.37-0.47 m; real bar σ_p0_lat ≲0.15 (p99 ≲0.45). Measured σ_p0 0.15-0.20 is MARGINAL-PASSING, not NO-GO." Then point to `MEMORY.md` + `index_vision_estimator.md` TERMINAL-LOCK.
- **info_loss_risk:** none — the 0.11-0.21 m RMS gate-relative measurement and the velocity-prior fork are independent of the bar error and stay valid; only the threshold it is judged against changes.

## [contradiction · high] inc8 / KF accuracy build directive (<0.05 m 1-sigma) — stale build target
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/memory/project_phase2_rl_vision_decisions.md:1353`
- **Evidence:** STALE (§S2-DECISION "Ordered integration plan" step 6): "[cross-cutting — PRIORITIZE OVER ANY PLANNER CHOICE] Estimator: drive KF to <0.05 m 1-sigma at post-gate-3 ~37 m/s window. THE binding VQ2 validity risk; measure first." — the <0.05 m target is the retracted double-counted bar. CONTRADICTS `MEMORY.md:8` (real bar σ_p0_lat ≲0.15, measured 0.15-0.20 is marginal-passing) AND the retraction "RL IS the right tool ... next lever = RL to lift reach/pass-rate; vision = SUPPORT, do NOT lean on it". Reads as a current top-priority build directive pointing engineering at the wrong (10×-too-tight) target.
- **Recommended action:** Edit the "<0.05 m 1-sigma" target to "<~0.15 m σ_p0_lat (corrected 2026-06-19; was <0.05 m double-counted bar)" and append: "and gate-4 is now understood as a REACH/PASS-RATE problem, not a sub-decimetre centering problem — RL stays the primary lever." Do not delete the step.
- **info_loss_risk:** none — the "measure the estimator first / estimator is the binding risk" instinct is preserved; only the numeric target is corrected.

## [contradiction · high] gate-4 contradiction — index_vision_estimator.md lacks the correction
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/memory/index_vision_estimator.md:285, :286-287, :258, header:11`
- **Evidence:** `:285` "CLOSES IFF σ_p0_lat ≤ 0.08"; `:286-287` "centering lever = near-field GATE estimator"; the gate-4 0.08 correction is ABSENT (grep finds only unrelated double-count hits at 212/234/332). Contradicts `MEMORY.md:8` REAL bar ≤ 0.15 (p99 ≤ 0.45), RETRACTED.
- **Recommended action:** Propagate `511e85c` into this vision/estimator sub-index: dated note at 285-287/header — budgets double-counted (clearance 0.37-0.47 m), binding bar 0.15, near-field-estimator DEMOTED to support, RL primary. Keep the σ/fix-density analysis.
- **info_loss_risk:** none if reframed — σ physics and boresight bake stay; only threshold/priority change.

## [contradiction · med] Gate-4: MEMORY.md NOW bullet internal residual contradiction
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/memory/MEMORY.md:8`
- **Evidence:** Headline: near-field pivot RETRACTED, next lever = RL. Later in the SAME bullet: "2nd lever = near-field GATE estimator (12 m floor); fix-rate ≥0.50 through last 6 m GEOMETRICALLY UNACHIEVABLE." The 2nd-lever line re-asserts the retracted pivot.
- **Recommended action:** Consolidate: delete/qualify the "2nd-lever near-field" phrase so the bullet is internally consistent. PRESERVE the SIGN FOOTGUN launch config (`++lookat` gains) and σ_b boresight bake.
- **info_loss_risk:** none — launch-config footgun must be preserved; only the near-field restatement is stale.

## [stale · high] case-C estimator absolute-path verdict (NO-GO) — judged against retracted budget
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/memory/project_phase2_rl_vision_decisions.md:1481, :1484, :1487`
- **Evidence:** STALE (§ESTIMATOR-RACESPEED Verdict): "Case-C ABSOLUTE world-frame KF nav = NO-GO at race speed, by a wide and speed-flat margin." (1481); "Deployable (b1-corrected): ~0.55 m RMS — ~3.5× the 0.155 m contact margin, ~11× the 0.05 m bar." (1484); "18-cell sweep ... NO cell clears 0.05 m." (1487). These NO-GO judgements are computed against the retracted 0.155 m margin and 0.05 m bar. `MEMORY.md:8` now states real clearance 0.37-0.47 m and "gate-4 is a REACH/PASS-RATE problem ... NOT sub-8cm centering"; the RECENTER policy FLEW 3/3 — the "NO-GO definitive" framing is superseded.
- **Recommended action:** Add the same dated supersession rider at the top of the §ESTIMATOR-RACESPEED "Verdict" block: the absolute-vs-gate-relative architecture finding stands, but every "/0.05 m bar", "/0.155 m margin", and "NO-GO/clears 0.05 m" magnitude judgement must be re-read against the corrected 0.37-0.47 m clearance and the 0.15 m bar. Do not delete the analysis.
- **info_loss_risk:** none — the gate-relative-beats-absolute architectural conclusion and the variance/bias decomposition survive; only the pass/fail labels flip from NO-GO to marginal-pass.

## [stale · high] Gate-4 stale — det-retrain REPORT asserts the RETRACTED view
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/handoff/inc8-deterministic-retrain-2026-06-19/REPORT.md:101, :105-106, :113-116, :125, :134, :135, :144-150` (MEMORY-DELTA items 2-4; committed at `42ddb0b`, on main)
- **Evidence:** REPORT asserts the OLD view as the verdict: `:144` "Measured 2-axis gate-4 σ_p0 = 0.15–0.20 m = NO-GO vs 0.08 (lat_p99 0.37–0.40; σ_vert ~0.20). The un-measurable gap is CLOSED — but the verdict is NO-GO."; `:150` "⇒ NEXT LEVER = near-field-gate estimator ... — NOT more look-at-gain RL."; `:106` "the binding wall is now fix-seating-vs-flight."; `:101` NO-GO vs 0.08; `:134`/`:149` NEXT LEVER = near-field-gate estimator, NOT more look-at RL. BUT the correction `511e85c` (CONFIRMED ancestor of HEAD): "CORRECT the gate-4 σ_p0 bar — 0.08 was a double-count ... real bar ~0.15; measured 0.15-0.20 = MARGINAL-PASSING not NO-GO; RETRACT the pivot-off-RL." `MEMORY.md` NOW-block likewise.
- **Recommended action:** Do NOT silently edit the historical REPORT — prepend a dated CORRECTION banner at the top pointing to `511e85c` / `MEMORY.md` NOW: "SUPERSEDED 2026-06-19: the 0.08 bar was a double-count; real bar ~0.15; measured 0.15-0.20 = MARGINAL-PASSING not NO-GO; the near-field-estimator pivot is RETRACTED — RL stays the right tool (gate-4 is a reach/pass-rate problem)." This preserves the deterministic-stability/std-cap lever (items 1 & 5 — GOOD and current) while flagging items 2-4 as retracted.
- **info_loss_risk:** none if banner-corrected (not deleted) — the std-cap/noise-anneal lever finding (det reach 0.467, actor_logstd saturation root cause) is still valid and load-bearing; only the σ_p0 VERDICT framing (NO-GO / pivot-to-estimator / NOT-more-RL) is stale.

## [stale · med] Gate-4 stale — sigmap0-adroit carries NO-GO-vs-0.08
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/handoff/sigmap0-adroit-2026-06-19/REPORT.md:9, :191`
- **Evidence:** `:191` and `:9` "gate-4 σ_p0 0.198 m = NO-GO vs 0.08". Superseded by `MEMORY.md:8` (marginal vs real 0.15 / p99 0.45).
- **Recommended action:** Prepend the same dated SUPERSEDED banner; keep validation tables.
- **info_loss_risk:** none — validation (torch 0.1981 vs numpy 0.1768 AGREE) and harness-bug fix unaffected.

## [stale · med] Gate-4 stale — commander-handoff-2026-06-18 states 0.08 GO rule live
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/handoff/commander-handoff-2026-06-18.md:21, :28`
- **Evidence:** `:21` "GO rule σ_p0_lat ≤ 0.08 AND lat_p99 ≤ 0.24"; `:28` "yaw-only 0.177 m NO-GO vs 0.08". Predates `511e85c`, reads as a live forward instruction.
- **Recommended action:** One-line dated note at `:21`: "GO rule SUPERSEDED 2026-06-19 (`511e85c`) — real bar 0.15 / p99 0.45, see `MEMORY.md` NOW."
- **info_loss_risk:** none — dated snapshot; correction supersedes only the threshold.

## [stale · med] pivot off RL / near-field estimator as primary lever
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/memory/project_phase2_rl_vision_decisions.md:1495-1502, :1690`
- **Evidence:** STALE: §ESTIMATOR-RACESPEED "Dead ends confirmed (do not spend on these)" table (1495-1502) frames "Better detector accuracy (c3) ... 0.000 m — detector already sub-pixel" and higher-cadence/lower-cov-floor as dead ends, concluding the fix must come from the estimator/gate-relative side; §GATE-RELATIVE-BLUEPRINT headline #3 (1690) "d4v cheap velocity lever REFUTED, demoted to P2 insurance." The whole file's thrust is "estimator/vision is the load-bearing fix for gate-4." CONTRADICTS the corrected retraction `MEMORY.md:8`: "the 'pivot off RL to a near-field-gate estimator' recommendation is RETRACTED; next lever = RL to lift reach/pass-rate; vision = SUPPORT ... the fix_rate=0-at-flyable-gains 'wall' is moot — the policy nearly passes WITHOUT vision fixes."
- **Recommended action:** Add a §ESTIMATOR-RACESPEED rider noting the post-2026-06-19 reframe: gate-4 is a REACH/PASS-RATE RL problem; the near-field-gate-estimator is a SUPPORT lever (2nd), not the primary fix. Keep the dead-end table (c2/c3/c4 dead-ends are still genuine within the estimator branch) but flag that the binding lever moved to RL reach/pass-rate. Frame as demotion-from-primary, NOT deletion (the near-field estimator IS retained as the "2nd lever" in `MEMORY.md`).
- **info_loss_risk:** none if the rider preserves the near-field-estimator idea as the secondary lever; deleting the section would lose the c2/c3/c4 dead-end analysis.

## [stale · med] gate-4 margin DOES-NOT-CLOSE / CANNOT-SETTLE-OFFLINE verdicts
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/memory/project_phase2_rl_vision_decisions.md:1559-1562, :1572, :1689, :1731`
- **Evidence:** STALE (§BODY-RADIUS-RECONCILE + §GATE-RELATIVE-BLUEPRINT): "cold @ 1.4° att-bias (REALISTIC) ... FAIL −0.103 ... FAILS at EVERY admissible radius at p90 AND p99 ... To clear it at p90 requires r < 0.197 m — below the chassis geometry floor; physically impossible." (1559-1562); "Gate-4 0.155 m worst-case margin DOES-NOT-CLOSE offline in ANY regime. CANNOT-SETTLE-OFFLINE." (1689); ranked-risk #1 "Gate-4 p90/p99 margin DOES NOT clear offline for the realistic case-C tail (cold p90@1.4-bias = 0.338 m)" (1731). All computed against the 0.155/0.235 m double-counted budgets. `MEMORY.md:8` corrects: real clearance 0.37-0.47 m, coast-drift RESOLVED ("informed coast <0.02 m over 0.4 s = NOT the blocker"), VIO/ESKF/velocity/speed ALL ruled out.
- **Recommended action:** Add a dated rider to §BODY-RADIUS-RECONCILE and the §GATE-RELATIVE-BLUEPRINT ranked-risk list: the p90/p99-fail-everywhere verdict used the double-counted 0.155/0.235 m budgets; at corrected clearance 0.37-0.47 m the realistic-tail cells largely PASS (marginal), coast-drift is resolved, and CANNOT-SETTLE-OFFLINE is downgraded (the only surviving offline-unsettleable item is the physical-drone at-speed blur σ, per `index_vision_estimator.md`). Preserve the radius-reconciliation math (0.2135 m geom floor, 0.30 central).
- **info_loss_risk:** none — the budget(r) identity, the 0.2135 m geometry floor, and the 0.30 m central radius are retained verbatim in the corrected `index_vision_estimator.md:52-65`; only the close/no-close conclusion against the bad budget changes.

## [stale · med] ESKF / boresight promoted as binding margin lever
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/memory/project_phase2_rl_vision_decisions.md:1564-1566, :1711, :1731-1732`
- **Evidence:** STALE: "ESKF attitude-bias estimation = BINDING MARGIN LEVER ... PRIMARY lever for closing the cold@1.4 case" (1566); ranked-risk #1/#2 (1731-1732) "BINDING FACTOR = VERTICAL BORESIGHT BIAS ... ESKF attitude-bias state = CO-EQUAL MARGIN LEVER ... HIGH". CONTRADICTS `MEMORY.md:8` "VIO / ESKF / velocity / speed ALL ruled out" and `index_vision_estimator.md:287` "NOT ESKF-for-coast (0.008 m), NOT VIO, NOT speed." ESKF was demoted from binding/primary lever once the bar was corrected and coast-drift resolved.
- **Recommended action:** Add a rider noting ESKF/boresight were demoted from "binding/co-equal margin lever" to data-conditional/minor by the 2026-06-19 correction (coast-drift resolved, ESKF-for-coast worth only ~0.008 m). Keep the boresight ε_vert ~0.215 m bias finding as a real DEPLOY-accuracy item (still deployed as `frames.BORESIGHT` −0.25 per `index_vision_estimator.md:258`).
- **info_loss_risk:** none — the ε_vert boresight bias and the deployed −0.25 bake remain valid for deploy accuracy; only the "binding/primary margin lever" priority claim is superseded.

## [stale · med] inc8 σ_p0 launcher GO-rule comment (stale 0.08)
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/rl/inc8_sigmap0_torch.sbatch:18, :127`
- **Evidence:** `:18` "vs the GO rule (σ_p0_lat ≤ 0.08 AND lat_p99 ≤ 0.24)." and `:127` "is suspect — report the discrepancy). Then the 2-axis σ_p0_lat vs the 0.08 GO rule." — both assert the double-counted 0.08 bar as the GO criterion, contradicting `511e85c`.
- **Recommended action:** Update both comment lines to the corrected bar (σ_p0_lat ~0.15 / p99 ~0.45) referencing the 2026-06-19 correction. Note that gate-4 is a reach/pass-rate problem, not a sub-8cm centering problem.
- **info_loss_risk:** none — comment text only; the yaw cross-check / 2-axis run logic is unaffected.

## [stale · med] inc8 warmstart launcher embeds the double-counted closure bar
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/rl/peregrine_inc8_warmstart.sbatch:49`
- **Evidence:** `:49` "miss σ_p0 ≲ 0.08 m (gate-4 closure, r=0.30). The TB inc8_centering trace shows the reward is" — states the honest-objective physical-centering bar as 0.08 m at r=0.30, exactly the double-count `511e85c` identified (gate clearance is ~0.37-0.47 m; 0.08 = 0.235/3 from subtracting the drone twice).
- **Recommended action:** Update the comment to "physical centering miss σ_p0 ~0.15 m (gate-4 clearance 0.37-0.47 m after the 2026-06-19 double-count correction)". The surrounding logic (crown on `contact_true_eval` physical miss, not `estim_err`) stays valid.
- **info_loss_risk:** none — the "crown on physical miss not estim_err" guidance is preserved; only the numeric bar is corrected.

## [stale · low] RL-portfolio gate-4 selection metric tied to old margin
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/memory/project_phase2_rl_vision_decisions.md:1671, :1679, :1710`
- **Evidence:** STALE (§RL-PORTFOLIO + §GATE-RELATIVE-BLUEPRINT inc8 spec): selection metric "Contact-valid (gate-4 margin ≥ 0 ...)" (1671); "Every cone-relaxation rung worsens gate-4 margin/σ ratio → re-verify gate-relative per-fix lateral against achieved σ" (1679); SELECT-on-p90/p99 "central on 0.30 (budget 0.235 m), worst-case on 0.38 (budget 0.155 m)" (1710). The 0.235/0.155 budgets are the double-counted values. The reporting-across-r discipline survives, but the absolute budget numbers are stale.
- **Recommended action:** Update the budget numbers (0.235/0.155) to the corrected clearance (0.37-0.47 m across r) OR add a one-line note "budgets here are pre-2026-06-19 double-counted values; corrected clearance 0.37-0.47 m — see `MEMORY.md`". Keep the report-across-r p90/p99 selection discipline (still current per `index_vision_estimator.md:10`).
- **info_loss_risk:** none — the multi-radius p90/p99 reporting discipline is unaffected; only the budget magnitudes shift.

## [stale · low] cross-reference / supersession ledger uses stale magnitudes
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/memory/project_phase2_rl_vision_decisions.md:1334, :1409, :1603`
- **Evidence:** STALE supersession entries re-asserting the NO-GO magnitude as settled fact: §S2-DECISION #5 "East σ 0.47 m = 3.0× the 0.155 m gate-4 margin. Decomposition buys NOTHING on perception." (1334); §PLANNING-TOGT-S2 (1409) "East σ 0.47 m ... = 3.0× the gate-4 0.155 m contact-true margin. Deployable in-plane ~0.55 m (absolute NO-GO, speed-flat)."; §ESTIMATOR-RACESPEED cross-refs (1603) "0.55 m deployable (absolute) vs 0.155 m margin → NO-GO." All anchored to the retracted 0.155 m margin.
- **Recommended action:** When the §ESTIMATOR-RACESPEED rider lands, append a back-pointer here noting these NO-GO ratios use the pre-correction 0.155 m margin; the absolute-vs-gate-relative architectural point (decomposition buys nothing on perception) stands, but the NO-GO label is now marginal-pass. Low priority — downstream echoes of the high-severity items above.
- **info_loss_risk:** none — downstream echoes; correcting the primary §ESTIMATOR-RACESPEED block plus a back-pointer covers these.

## [stale · low] Gate-4: parked-backlog #70 cites near-field estimator as centering lever
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/memory/project_parked_backlog.md:78`
- **Evidence:** `:78` "The centering lever = a near-field GATE estimator (lower the 12 m floor), NOT odometry." Tension with `MEMORY.md:8` (RL primary, estimator support).
- **Recommended action:** Dated note to #70: gate-4 re-framed 2026-06-19 (`511e85c`) as reach/pass-rate, RL primary, near-field estimator now optional SUPPORT. Keep the VIO back-solve note (×9 margin).
- **info_loss_risk:** none — VIO reasoning and parked status unaffected; only the centering-lever phrasing needs the note.

## [stale · low] Gate-4: historical inc8 ladder records use the 0.08 bar (point-in-time)
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/memory/project_rl_increment_history.md:734, :736, :749, :750`; inc8-s1/s2/s3 REPORTs; `inc8-architecture/DESIGN.md:82, :86`; `inc8-eval-lookat:127`; `inc8-eval-pitch:148`; `inc8-recenter-run:139`; `coast-drift:153, :158, :165, :196, :252`; `inc8-sigmap0-torch:178`
- **Evidence:** `history:734` "estim_err 0.115 vs ≤0.08 NOT-GO"; `inc8-s3:129` "NOT-GO estim_err ≤ 0.08"; `coast-drift:165` "CONDITIONAL CLOSE gated on σ_p0 ≤ 0.08." Dated records where 0.08 WAS the working bar.
- **Recommended action:** Do NOT rewrite. Add ONE forward-pointer at the inc8-ladder header in `project_rl_increment_history.md`: "the ≤0.08 bar was CORRECTED 2026-06-19 (`511e85c`) to 0.15 / p99 0.45 — re-judged in `MEMORY.md` NOW." Leave per-stage handoff REPORTs as-is.
- **info_loss_risk:** none — valid history; one forward pointer prevents mistaking the dated bar for current truth without erasing provenance.

---

# Artifact-pipe violations & tracked binaries — issue_type: prune (NEVER-git-add rule unenforced)

> The MEMORY artifact-pipe rule ("binaries ship as `burn-artifacts-*` GitHub release assets, NEVER git-add") has NO enforcing `.gitignore` entry for the binary types that actually landed: ~38 MB of tracked zips/PNGs/tfevents/npz/npy across `handoff/` + retired-increment checkpoints under `rl/checkpoints/`. The root-cause fix (a gitignore block) is filed at the end of this section. Across ALL of these: re-home to a release asset FIRST, leave an md pointer, then `git rm --cached` — do NOT plain-delete until the release asset exists.

## [prune · high] committed binary — 9 MB zip dataset (shadowpc-refit 17-run)
- **Location:** `handoff/shadowpc-refit-dataset-2026-06-12/debug_obs_17runs.zip` (added commit `c846054`)
- **Evidence:** `git ls-files` shows TRACKED. `stat` size = 9,465,597 bytes (9.0 MB) — the single largest file in `handoff` and ~16% of the whole 59 MB handoff tree. MEMORY artifact-pipe rule: "binaries ... NEVER git-add". `.gitignore` covers `*.pt`/`*.pth` but NOT `*.zip`, so this slipped through.
- **Recommended action:** Re-home `debug_obs_17runs.zip` to a `burn-artifacts-*` release asset, leave a one-line pointer in the dir's REPORT/HANDOFF md (or a STAGED.md), add `*.zip` under `handoff/` to `.gitignore`, then `git rm --cached` the file. Do NOT plain-delete until the release asset exists.
- **info_loss_risk:** MED — the raw 17-run debug_obs dataset for the translational refit; the refit RESULTS are narrated in the dir's md but the raw jsonl runs are only in this zip. Preserve by re-homing (not deletion) before un-tracking.

## [prune · high] committed binaries — TensorBoard event files (~4.83 MB)
- **Location:** `handoff/inc8-intraining-coursecheck-2026-06-17/tb/ws_seed0.tfevents, ws_seed1.tfevents, ws_seed2.tfevents, s2full_seed2_yawprim.tfevents` (added commit `604831d`)
- **Evidence:** All 4 TRACKED; sizes 1,208,081 / 1,208,081 / 1,208,081 / 1,207,731 bytes = ~4.83 MB combined (4th-largest handoff dir by bytes). Raw training-run event logs — exactly the "recordings" the artifact-pipe rule says must ship as release assets, not git-add.
- **Recommended action:** Re-home the 4 `*.tfevents` to a `burn-artifacts-*` release asset, add `*.tfevents` to `.gitignore`, then `git rm --cached`. The in-training coursecheck REPORT.md narrates the diagnosis, so the curves can be regenerated/reattached on demand.
- **info_loss_risk:** MED — raw TB curves for the inc8 in-training coursecheck seeds; the policy-gap diagnosis is in REPORT.md, but the per-step curves are only in these files. Preserve via release asset before un-tracking.

## [prune · high] heavy PNG artifacts — vq2-blender-render previews (~10.2 MB)
- **Location:** `handoff/vq2-blender-render-2026-06-15/preview/` (28 PNGs, 9,385,040 b) + `samples/` (13 PNGs, ~813 KB)
- **Evidence:** Dir is 10.2 MB total, 41 of 45 files are PNG (10,198,036 b = 99.8% of the dir). Top files: `15_gate_allhue_lit.png` 414,761 b, `07_people_hangar.png` 413,309 b, `18_combinedV2_2.png` 403,723 b — 28 preview renders >320 KB each. The VQ2 detector WIN is a TRAINING-LOSS lever narrated in REPORT.md + COMMANDER_REPORT.md (both present in the dir).
- **Recommended action:** Re-home `preview/` and `samples/` PNGs to a `burn-artifacts-*` release asset (illustrative renders, regenerable from the Blender pipeline — the `.ps1` render scripts are in the same dir), add a pointer in REPORT.md, add `handoff/**/*.png` to `.gitignore`, then `git rm --cached`. Reclaims ~10 MB. Do not delete the two md narratives.
- **info_loss_risk:** LOW-MED — regenerable preview renders; the appearance-broadening conclusion is in REPORT.md. Re-home rather than delete so the visual evidence remains retrievable.

## [prune · high] heavy PNG artifacts — shadowpc-followups PnP frame dumps (~12.9 MB)
- **Location:** `handoff/shadowpc-followups-2026-06-05/task2_frames/` (40 PNGs, 10,820,388 b) + `scratch/` (8 PNGs, 2,080,261 b)
- **Evidence:** Dir is 13.7 MB total (largest handoff dir by bytes), 48 of 65 files are PNG (12,900,649 b = 94% of the dir). `task2_frames` holds 40 range-tagged frame PNGs (e.g. `05_id12713_3.4m.png` 296,162 b). The PnP verdict is narrated in `TASK2_PNP_VERDICT.md` (3,075 b) and `UNDERSTANDING.md` (9,565 b), both present.
- **Recommended action:** Re-home `task2_frames/` and `scratch/` PNGs to a `burn-artifacts-*` release asset, add a pointer to `TASK2_PNP_VERDICT.md`, gitignore handoff PNGs, `git rm --cached`. Reclaims ~12.9 MB. Keep the 3 md narratives.
- **info_loss_risk:** LOW — per-detection frame images supporting the PnP verdict, which is fully captured in `TASK2_PNP_VERDICT.md`. Re-home to retain the raw frames for future re-analysis.

## [prune · med] committed binary — 374 KB zip dataset (shadowpc-postfix 8-run)
- **Location:** `handoff/shadowpc-postfix-dataset-2026-06-12/debug_obs_8runs.zip` (added commit `6dcd62b`)
- **Evidence:** TRACKED; `stat` size = 374,403 bytes. Same class as the 17-run zip — a packaged debug_obs dataset committed in violation of the NEVER-git-add rule.
- **Recommended action:** Re-home to a `burn-artifacts-*` release asset, add `*.zip` under `handoff/` to `.gitignore`, `git rm --cached`. Pointer in the dir's md.
- **info_loss_risk:** MED — raw 8-run post-fix dataset for the laptop dynamics refit; results narrated in md, raw runs only in zip. Re-home, do not delete.

## [prune · med] raw trajectory CSVs — TOGT/refined dumps (~3.17 MB)
- **Location:** `handoff/laptop-togt-bound-2026-06-10/cases/*/togt_traj.csv and refined_traj.csv` (22 CSVs, 3,165,286 b)
- **Evidence:** Dir is 3.34 MB, 22 CSVs = 3,165,286 b = 95% of the dir. Per-case trajectory dumps (`sens_thr50/togt_traj.csv` 205,543 b, etc.). `WRITEUP.md` (15,069 b) present, references togt/refined/trajectory/bound 34 times. `results_table.md` (1,182 b) holds the summary table.
- **Recommended action:** Confirm `WRITEUP.md` + `results_table.md` capture the per-case bound/sensitivity CONCLUSIONS (they do — 34 mentions), then re-home the raw `togt_traj`/`refined_traj` CSVs to a release asset and `git rm --cached`, OR keep if cheap (3.3 MB plain text). Lower priority than binaries/PNGs since CSVs diff/compress well. Add `*.csv` under `handoff/` to `.gitignore` only if a policy decision is made.
- **info_loss_risk:** MED — raw optimizer trajectories per sensitivity case; the bound/sensitivity verdicts are in `WRITEUP.md` but the point-by-point trajectories are only in the CSVs. Re-home (do not delete) if reclaiming.

## [prune · med] retired-increment actor checkpoints tracked despite .gitignore artifact-pipe policy
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/rl/checkpoints/stage1_inc1_actor.pth, stage1_inc5_actor.{pth,json}, stage1_inc6_actor.{pth,json}`
- **Evidence:** `.gitignore:39` `*.pth` and `:40` `rl/checkpoints/*.json` — yet `git ls-files rl/checkpoints` returns 11 tracked `.pth`/`.json` (inc1/inc3/inc4/inc5/inc6/inc7). MEMORY: "binaries ... NEVER git-add". Reference scan: inc1.pth = prose-only (`fly_rl.py:465`, `offline_rollout.py:261` docstrings; `peregrine_racing.sbatch` trains it but that launcher is itself retired); inc5.pth/.json + inc6.pth/.json referenced ONLY by orphan analysis scripts `replay_obs.py`/`tilt_segment_analysis.py` (themselves orphaned). inc3.json is LOAD-BEARING (pinned by `tests/test_offline_rollout_events.py:88` `test_inc3_sidecar_committed`). inc4.pth is the default of `offline_rollout.py:256` (a still-referenced analysis tool). inc7.{pth,json} = LIVE current best.
- **Recommended action:** KEEP inc7.{pth,json} (live), inc3.json (test-pinned), inc4.pth/.json (offline_rollout default + sidecar test). For inc1.pth, inc5.{pth,json}, inc6.{pth,json}: migrate to the `burn-artifacts` release-asset pipe and untrack, OR leave as a deliberate grandfathered exception — this needs the commander's call (they predate the ignore rule; untracking is a history/availability decision). Flag as UNCERTAIN for the human on untrack-vs-keep.
- **info_loss_risk:** MEDIUM — these are the only in-repo copies of retired actor weights; do NOT delete. If untracked, first publish as `burn-artifacts` release assets so the weights remain retrievable, then remove from git tracking.

## [prune · low] committed binaries — numpy arrays (npz/npy)
- **Location:** `handoff/laptop-training-doctrine-2026-06-12/scripts/q1_trace.npz` (48,707 b, commit `7ecc575`); `handoff/shadowpc-twin-falsify-2026-06-10/vert_fit_coef.npy` (640 b, commit `9684ae1`)
- **Evidence:** Both TRACKED. `*.npz`/`*.npy` not in `.gitignore`. Small (49 KB and 640 B) so byte-reclaim is minor, but they are committed binary artifacts — same policy class as the zips/tfevents.
- **Recommended action:** Add `*.npz` and `*.npy` under `handoff/` to `.gitignore` for future hygiene. These two are small; if the parent WRITEUP/analysis already records the fit coefficients (`vert_fit_coef` is a tiny coefficient array; `q1_trace` is a probe trace), re-home to a release asset OR confirm the numbers are quoted in the md and then `git rm --cached`. Low priority.
- **info_loss_risk:** LOW for `vert_fit_coef.npy` (a few coefficients — verify quoted in twin-falsify WRITEUP.md). MED for `q1_trace.npz` (a probe trace) — confirm Q1 findings in laptop-training-doctrine WRITEUP.md before un-tracking; otherwise re-home.

## [prune · low] zero-byte SLURM error logs (simops-mastery)
- **Location:** `handoff/simops-mastery-2026-06-13/logs/char_std{1-5}_g{0-5}.err and related` (30 files, 0 bytes total)
- **Evidence:** `find handoff -type f -empty`: ALL 30 empty files in handoff are `.err` logs in this one dir. Zero bytes (no error output = clean runs). They reclaim file-COUNT clutter (30 of the dir's 117 files), not bytes. The char_g* run RESULTS are narrated in REPORT.md (14,092 b) and the cov-sweep JSONs.
- **Recommended action:** `git rm --cached` the 30 zero-byte `*.err` logs and add `handoff/**/*.err` to `.gitignore`. Empty SLURM stderr carries no information (runs are clean; results are in the JSON bundles + REPORT.md).
- **info_loss_risk:** NONE — files are 0 bytes (no error content). The clean-run fact is implied by the successful JSON results already tracked.

## [prune · low] retired-increment sbatch launchers (inc1/s13/s14/inc5/inc6 era)
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/rl/peregrine_racing.sbatch, peregrine_racing_s13.sbatch, peregrine_s13_sweep.sbatch, peregrine_racing_s14.sbatch, peregrine_racing_inc5.sbatch, peregrine_racing_inc6.sbatch, peregrine_smoke.sbatch`
- **Evidence:** All seven target the RETIRED lineage per `index_rl_training.md:13` "inc4 (RETIRED) → inc5 (RETIRED) → inc6 (fallback) → inc7 = CURRENT BEST". `peregrine_racing.sbatch` trains `peregrine_stage1_inc1`; racing_s13/s14/sweep = S1.3/S1.4 stages; `inc5.sbatch` header "S1.5 (inc5) retrain"; `inc6.sbatch` header "S17 (inc6) retrain"; `smoke.sbatch` = inc1-era 240s smoke. Last-touched 2026-06-08 to 2026-06-11; none invoked by any live path. Each carries UNIQUE reward-weight/DR-band provenance in its header comments (e.g. inc6 S17 DR bands `idle[0.04,0.08]`/`kappa_err[0.060,0.085]`, the mixer-corner R7 tax derivation).
- **Recommended action:** Consolidate into an `rl/archive/` subdir (or `rl/legacy_launchers/` via `git mv`) rather than delete — the header comments are the only record of each stage's exact DR bands and reward weights. Before archiving, confirm the DR-band/reward provenance in each header is mirrored in `project_rl_increment_history.md` (§S14/§S15/§INC5/§S17); if any band is NOT in memory, bank it first. `peregrine_racing_inc7.sbatch` is LIVE — KEEP it in `rl/` root.
- **info_loss_risk:** MEDIUM if deleted outright — per-stage DR bands and reward-weight rationale live in these headers. Preserve by archiving (not deleting) and/or banking any header detail not already in `project_rl_increment_history.md`.

## [prune · low] orphaned analysis scripts (zero non-self references)
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/rl/tb_parse.py, rl/tilt_segment_analysis.py, rl/peregrine_smoke_precheck.py, rl/replay_obs.py, rl/local_gate_harness.py`
- **Evidence:** Repo-wide grep (src/scripts/tests/cluster/rl, excluding self): `tb_parse` → 0 refs; `tilt_segment_analysis` → 0 refs (imports retired inc5/inc6 ckpts); `peregrine_smoke_precheck` → 0 refs (companion to the retired inc1-era `peregrine_smoke.sbatch`, oldest rl/ file); `replay_obs` → referenced only in a comment (`fly_rl.py:136`); `local_gate_harness` → referenced only in a test DOCSTRING (`test_measured_aero.py:25`, not imported/executed). None imported by any module/test/sbatch.
- **Recommended action:** Consolidate into `rl/archive/` (`git mv`), not delete — one-off debug/analysis utilities that may be re-run. `peregrine_racing_precheck.py` is NOT in this list (pinned by `tests/test_peregrine_racing_core.py`, KEEP it). Verify no MEMORY topic file documents a workflow that still calls these before archiving.
- **info_loss_risk:** LOW — debug tooling; archive (do not delete) to preserve re-run ability; `tilt_segment_analysis` additionally encodes the inc5-vs-inc6 tilt comparison method.

## [prune · low] inc1-era smoke trainer entrypoint (peregrine_train.py)
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/rl/peregrine_train.py`
- **Evidence:** Docstring: "Peregrine smoke-train launcher ... hands off to diffaero's OWN hydra train entrypoint unchanged." Invoked ONLY by `rl/peregrine_smoke.sbatch` (itself an inc1-era orphan). LIVE trainers are `peregrine_train_racing.py` (racing/s13/s14/inc5/inc6/inc7 sbatch) and `peregrine_train_inc8.py` (all six inc8 sbatch). No test imports `peregrine_train.py`.
- **Recommended action:** Archive alongside `peregrine_smoke.sbatch` (a matched inc1-era smoke pair) into `rl/archive/`. Do NOT touch `peregrine_train_racing.py` (shared by the LIVE inc7 launcher) or `peregrine_train_inc8.py` (active dev).
- **info_loss_risk:** none — the plant-registration monkeypatch pattern is reproduced in `peregrine_train_racing.py` and `peregrine_train_inc8.py`.

---

# Stale / drifted documentation & comments — issue_type: stale / contradiction

## [contradiction · high] footgun doc drift — fly_rl.py default ckpt (memory says inc4, code says inc7)
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/rl/fly_rl.py:1201-1207` vs `memory/MEMORY.md` and `memory/index_rl_training.md:14, :26, :576`
- **Evidence:** CODE (`fly_rl.py:1202-1207`): `default=str(Path(__file__).resolve().parent / "checkpoints" / "stage1_inc7_actor.pth") ... DEFAULT = inc7 (LIVE-CONFIRMED current best ... The retired inc4 default was a footgun)`. MEMORY (`MEMORY.md` cross-cutting footguns): "fly_rl.py default ckpt = retired inc4" and `index_rl_training.md:14` "fly_rl.py default still inc4 — pass inc7 explicitly", `:576` "fly_rl.py default still points at inc4". The code default was FIXED to inc7; the memory footgun is now STALE and over-warns. `submit_rl.py:37` correctly pins `_CKPT = _HERE / "checkpoints" / "stage1_inc7_actor.pth"` (the submit_rl half is accurate).
- **Recommended action:** Update `MEMORY.md` and `index_rl_training.md` (lines 14/26/576) to: "fly_rl.py default is NOW inc7 (fixed); submit_rl.py still pins inc7 explicitly." Preserve the historical note that the inc4 default WAS a footgun (mark RESOLVED, do not drop the lesson). This is a memory-file edit, not an rl/ change.
- **info_loss_risk:** none if the historical "inc4-was-a-footgun" lesson is retained in the rewrite.

## [stale · med] run_parity.sh hardcodes a stale md5 for diffaero_dynamics.py
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/rl/run_parity.sh` (md5 line + the echoed md5)
- **Evidence (auditor re-confirmed):** `run_parity.sh:11` `echo "=== diffaero_dynamics md5 (should be cdf79613267644f94d3e89f5277bcf5f) ==="`. Actual `md5sum rl/diffaero_dynamics.py` = `019969d3bcb4022a24c42a9b7dff4dde`. The expected md5 no longer matches the tracked file, so the parity-gate runner's self-check prints a mismatch warning on every run.
- **Recommended action:** Either update the expected md5 in `run_parity.sh` to `019969d3bcb4022a24c42a9b7dff4dde`, or replace the brittle hardcoded-md5 guard with a comment that the md5 is informational only. Verify against the current `diffaero_dynamics.py` before pinning.
- **info_loss_risk:** none — the parity check itself (`check_diffaero_gate.py`) is the real gate; the md5 line is only an advisory fingerprint.

## [stale · med] README.md staleness (test count + Python version)
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/README.md:15, :28`
- **Evidence (auditor re-confirmed):** `README.md:15` `## Setup (Python 3.13)` and `README.md:28` `pytest   # expect 125 passed`. The expected count 125 is far below the current suite: MEMORY.md states "947 tests collected (...prior 723/884/933 stale; +14 spike)". README last touched 2026-05-29 (`d6c47d4`), before the suite grew ~8×. (The Python 3.13 choice is internally consistent with `pyproject.toml` requires-python ">=3.13,<3.14", so that part is current.)
- **Recommended action:** Update `README.md:28` from `# expect 125 passed` to the canonical count, OR replace the brittle hardcoded number with a pointer to the gate ("run `scripts/green_gate.py`; the sentinel baseline is the source of truth") so it stops drifting. **NOTE the 933-vs-947 discrepancy:** `green_gate.py` sentinel is 933 on disk; MEMORY says 947 — reconcile which number is canonical before pinning either into README. Do not touch the Python 3.13 lines.
- **info_loss_risk:** none — the 125 number carries no unique information; the live source of truth is `scripts/green_gate.py`'s sentinel.

## [stale · low] README.md staleness (retired pipeline description)
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/README.md:55-65`
- **Evidence:** `README.md:63-65` "VQ1 uses a color/contour gate detector + position-target controller. VQ2 swaps in a learned detector and (optionally) attitude-target control without changing the rest of the stack." This is the early VQ1/VQ2 plan; current reality per MEMORY is inc7 (CTBR/SET_ATTITUDE_TARGET) live-confirmed, a learned YOLO-pose detector + VQ2 ensemble already built, and a case-C self-localizing estimator stack. The README does NOT name any specific retired increment (no "inc4"), so this is mild drift, not a hard contradiction.
- **Recommended action:** Light refresh: add a one-line "see `memory/index_rl_training.md` and `memory/index_vision_estimator.md` for current pipeline state" pointer, or refresh the two sentences to note CTBR/attitude-target is the live control path. Low priority.
- **info_loss_risk:** none — the current pipeline is fully captured in the memory index files; the README paragraph is a superset-stale summary, not unique knowledge.

## [stale · low] README.md open-questions already resolved
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/README.md:67-72`
- **Evidence:** `README.md:69-72` lists as still-open: "Spec lists 'linear velocities' ... but no velocity-bearing message ... Which message actually carries it?" and "Whether LOCAL_POSITION_NED is emitted even though it isn't listed." Both RESOLVED in memory: `reference_competition_materials.md:52` (velocity §4.5 prose only; §4.3 carries none — derive it) and the wire-spec facts (the practice/ShadowPC sim DOES stream LOCAL_POSITION_NED + ODOMETRY; official scored wire does not — MEMORY.md WIRE-SPEC block).
- **Recommended action:** Replace the bullet list with a pointer ("resolved — see `memory/reference_competition_materials.md` velocity-telemetry + WIRE-SPEC notes") rather than leaving them phrased as live unknowns. Low priority.
- **info_loss_risk:** none — answers preserved in `reference_competition_materials.md:52` and the WIRE-SPEC memory block.

## [stale · low] docs/first_contact.md staleness (forward-dated, pre-contact runbook)
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/docs/first_contact.md:1-8`
- **Evidence:** Title `# First-contact runbook (sim drop 2026-05-30)` and line 4 "sim interface — CONFIRMED from the shipped PyAIPilotExample (2026-06-01)"; body in future tense ("the ordered sweep to run the moment the official sim is up", "MUST-VERIFY live ... settle them in the first session"). Last touched 2026-06-01 (`5501d43`). First contact has since happened: MEMORY documents inc7 live-confirmed, ShadowPC live deploy 2026-06-11, system-id DONE 2026-06-18 — nearly all R1/R2/clock/attitude-bias questions this runbook poses were settled weeks ago.
- **Recommended action:** KEEP but mark historical: add a one-line top banner ("HISTORICAL — first contact occurred ~2026-06-01..06-11; see `memory/index_control_sim.md` (sim interface) + `memory/reference_sim_ops.md` for settled answers; retained for the master measurement checklist §A-§J which is still a useful template"). The §A-§J "INFORMATION TO EXTRACT" checklist retains reuse value.
- **info_loss_risk:** The §A-§J master checklist and the R1-R11 decision table are a thorough measurement template not fully duplicated elsewhere; do NOT delete. Preserve by retaining the file and adding only a historical banner.

## [stale · low] docs/elodin.md vs memory (status framing outdated)
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/docs/elodin.md:1-32`
- **Evidence:** Last touched 2026-05-28 (`c1c12e1`); describes the Elodin open-source rig as a "develop the whole stack against today" surrogate while "the official DCL/Anduril sim isn't released yet" (lines 2-3). The official sim has since dropped and been used extensively (ShadowPC live deploy 2026-06-11), so the "isn't released yet" framing is stale. Pins `elodin==0.17.2` and Python deps; `reference_competition_materials.md:66` says run the ML stack on Python 3.12 for CUDA while README/pyproject standardize on 3.13 — elodin.md itself asserts no Python version, so no direct contradiction, but its "today / not released yet" premise is outdated.
- **Recommended action:** KEEP as Elodin-surrogate reference (the gotchas — camera far=0.65 patch, GT discipline, baro=altitude, VFoV mislabel — remain valid). Add a one-line status note that the official sim has since released and Elodin is now a secondary/optional surrogate. Do not delete.
- **info_loss_risk:** The Elodin integration gotchas (far-plane patch, ENU/FLU/scalar-last adapter conventions, GT-hiding discipline) are not fully duplicated in memory; preserve the file and only soften the "not released yet" framing.

---

# Duplication — issue_type: duplication

## [duplication · low] duplicated 0.08-bar GO-gate framing — numpy eval vs live torch eval
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/rl/inc8_sigmap0_eval.py` vs `rl/inc8_sigmap0_torch_eval.py`
- **Evidence:** Two near-parallel GO-gate instruments. The numpy tool (`inc8_sigmap0_eval.py`, `--estim-emul` path, laptop-runnable) is the predecessor; the torch tool (`inc8_sigmap0_torch_eval.py`) is the LIVE deliverable wired into `inc8_select_ckpt.sbatch`/`detstab.sbatch` and is what MEMORY treats as the gate-4 σ_p0 instrument ("measure-in-torch confirmed by system-id `bc182f9`"). The numpy tool is still referenced by the torch tool and by `tests/test_inc8_sigmap0_torch_crossing.py` (cross-check), so NOT fully orphan.
- **Recommended action:** KEEP both (the numpy tool is the laptop-runnable cross-check the torch tool validates against), but when fixing the 0.08 bar do so in BOTH so they stay consistent. Add a one-line header note in `inc8_sigmap0_eval.py` pointing to `inc8_sigmap0_torch_eval.py` as the deliverable instrument to avoid future confusion about which is canonical.
- **info_loss_risk:** none — both retained; the change is the threshold fix already captured in the two high-severity findings above.

## [duplication · low] superseded reference-line JSON (vq1, drag-infeasible)
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/rl/reference_line_vq1.json`
- **Evidence:** `reference_line_inc8.json` header field: `"supersedes": "rl/reference_line_vq1.json (TOGT on linear-drag plant; drag-infeasible)"` and `"note": "T/W 3.765 in reference_line_vq1.json is the LINEAR plant's full-stick ... the measured convex map delivers 7.98 g."` `index_rl_training.md:91` confirms `rl/reference_line_inc8.json + rl/build_reference_line.py (vq1 json untouched)`. The vq1 line is the drag-infeasible predecessor of the live inc8 reference line.
- **Recommended action:** KEEP for now — `reference_line_inc8.json`'s provenance explicitly cites vq1 as what it supersedes (the comparison is informative), and `build_reference_line.py` is the documented generator of the inc8 line. If pruning later, confirm no eval/test loads `reference_line_vq1.json` first; safe to archive only after that check. Treat as UNCERTAIN.
- **info_loss_risk:** LOW — the vq1 line's infeasibility lesson is captured in the inc8 JSON's supersedes/note fields, but keep the file unless a reference check confirms nothing loads it.

## [duplication · low] superseded scratch script duplicated from scripts/ (simops-mastery)
- **Location:** `handoff/simops-mastery-2026-06-13/verify_bundle.py` (10,298 b) vs canonical `scripts/verify_bundle.py` (10,913 b)
- **Evidence:** Same basename in both places; diff = DIFFERENT. The `scripts/` header: `"""verify_bundle.py — recording-bundle integrity validator (PROMOTED to scripts/ 2026-06-17)."` The handoff copy header: `"""verify_bundle.py — PART C harness validator (SIMOPS-MASTERY-SHADOWVISION)."` I.e. `scripts/verify_bundle.py` is the productionized descendant; the handoff one is the superseded scratch original.
- **Recommended action:** Leave the handoff copy as historical scratch (documents the SIMOPS-MASTERY PART-C provenance) OR replace with a one-line pointer comment to `scripts/verify_bundle.py` if dir slimming is wanted. The other 4 simops `.py` (`analyze_shadow_vision`, `bearing_diag`, `partA_cycle`, `partA_recovery`) are UNIQUE (no scripts/ or rl/ counterpart) — KEEP them.
- **info_loss_risk:** NONE for the duplication — canonical lives at `scripts/verify_bundle.py`. The provenance note in the handoff header is the only unique content; preserve it in the dir's REPORT.md if the file is collapsed to a pointer.

## [duplication · low] _spec.txt vs 260508_Technical_Spec_0002.pdf (intentional text mirror — KEEP)
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/_spec.txt:1-566` (vs tracked binary `260508_Technical_Spec_0002.pdf`)
- **Evidence (auditor re-confirmed):** `_spec.txt` header lines 1-6: "AI Grand Prix Virtual Qualifier / Technical Specification / Document ID: VADR-TS-002 / Issue: 00.02 / Date: 2026-05-08" — matches the PDF identity in `reference_competition_materials.md:12`. COMPLETE extraction (§1 through final §8.3 "Maximum run duration: 8 minutes."), not partial. LOAD-BEARING grep source memory cites: `project_parked_backlog.md:91` "SPEC-DIVE CONFIRMED (2026-06-15, _spec.txt §4.3/§4.5)"; the §3.8 VFoV mislabel MEMORY relies on is verifiable at `_spec.txt:344` `VFoV= 90°` next to `[fx,fy] =[320,320]` (auditor confirmed this exact line). `project_master_plan.md:66` registers it: "Spec PDF + extracted _spec.txt at root."
- **Recommended action:** KEEP. NOT a stale/redundant duplicate to prune — it is the intentional, current (Issue 00.02, same as the PDF) plain-text mirror of a 1.4 MB tracked binary, kept so the spec is greppable/citable. Verified identical document identity and full section coverage. No action beyond noting the leading-underscore filename is a minor presentability nit (separate finding).
- **info_loss_risk:** Deleting `_spec.txt` would remove the only text-searchable copy of the authoritative spec (the PDF is binary, skipped by grep), breaking the section-citation workflow memory depends on. Do not delete; if regenerated, re-extract from the same PDF and confirm §3.8/§4.3/§4.5 survive.

## [duplication · low] docs/sim_ops.md vs memory/reference_sim_ops.md (complementary — KEEP both)
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/docs/sim_ops.md:1-166` vs `memory/reference_sim_ops.md:1-83`
- **Evidence:** Both cover unattended ShadowPC FlightSim control. Overlapping facts: "One sim instance, ever" / zombie dual-instance (`docs:11,40-43` vs `memory:32`); MAV_CMD 31000 in-race-only no-op (`docs:16,103` vs `memory:20`); fullscreen auto-minimize + verified foreground (`docs:51-67` vs `memory:36`); autoreset/spin guards (`docs:118-128` vs `memory:33-34`). COMPLEMENTARY, not byte-identical: `docs/sim_ops.md` is the polished operator runbook (canonical `_force_foreground`, `sim_focus.py` CLI, screen-map table, telemetry-trust table) last touched 2026-06-11 (`972cebb`); `reference_sim_ops.md` is the commander's banked-facts + the 2026-06-13 autonomy-readiness DQ audit (MAV_CMD 31000 on judged wire = DQ) which `docs/sim_ops.md` does NOT contain.
- **Recommended action:** KEEP both (in-repo operator runbook vs memory fact/footgun trail). No content contradiction. Optionally add a cross-pointer: `docs/sim_ops.md` → `reference_sim_ops.md` §AUTONOMY-READINESS for the DQ-risk audit, and the memory file → `docs/sim_ops.md` as the canonical procedure. Do not merge or prune.
- **info_loss_risk:** none if both kept. If ever consolidated, the memory file's §AUTONOMY-READINESS DQ audit (MAV_CMD 31000 = §7 DQ) is unique and must be preserved — NOT in `docs/sim_ops.md`.

## [duplication · low] docs/training_doctrine.md vs memory/index_rl_training.md (overlap, current — KEEP)
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/docs/training_doctrine.md:1-113`
- **Evidence:** Last touched 2026-06-12 (`a1ad554`); durable in-repo RL training doctrine (reward composition, contact geometry, DR policy, checkpoint gauntlet). Consistent with current memory; contains NO stale gate-4/σ_p0 material — grep for "sigma_p0|0.08|no-go|near-field|gate-4|inc8|inc7|inc4" across `docs/` returned ZERO hits, so the 2026-06-19 gate-4 correction does not touch any doc. Contact-geometry numbers (r_body in [0.28,0.38], pass requires L-inf ≤ 0.75 − r_body, lines 36-39) MATCH MEMORY's corrected contact-radius footgun.
- **Recommended action:** KEEP — current and canonical durable RL training doctrine; `index_rl_training.md` is the live index that should point to it. No contradiction with the gate-4 correction. No action beyond confirming `index_rl_training.md` links to it.
- **info_loss_risk:** none — load-bearing doctrine, not a duplicate to prune.

---

# Loose / repo-root files & config — issue_type: prune / presentability

## [prune · low] repo-root loose file simops_helper.py (KEEP — has live importers)
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/simops_helper.py:1-5` (consumers `handoff/at-speed-sigma-2026-06-15/sweep_runner.py:27`, `run_inc7_lap.py:13`)
- **Evidence:** Docstring: "simops_helper.py — boresight mission helpers: probe + key-send + zombie-guard. Derived from `handoff/simops-mastery-2026-06-13/partA_cycle.py` (proven chain). NOT modifying src/racer, rl/, scripts/, or memory/. Disposable helper only." IMPORTED by two handoff scripts: `sweep_runner.py:27` `from simops_helper import drive_to_waiting, send_keys, n_sim_procs, classify` and `run_inc7_lap.py:13` same — both add repo ROOT to `sys.path` (`ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))`) precisely to resolve this top-level import. Last touched 2026-06-15 (`2f53ae1`).
- **Recommended action:** KEEP in place but not canonical. NOT orphaned — two `handoff/at-speed-sigma-2026-06-15` scripts import it from repo root. Deliberately placed at root (self-described "disposable helper", avoiding `scripts/`) so as not to touch canonical trees. Do NOT move it into `scripts/` without also editing both handoff importers' `sys.path` (silent breakage). Safe relocation path if root tidy is wanted: move to `handoff/at-speed-sigma-2026-06-15/` (co-located with its only two consumers) and re-run both. Functional overlap with `scripts/sim_focus.py` (window-find + force_foreground + key-send) but `simops_helper` adds a MAVLink probe/classify/drive-to-waiting state machine `sim_focus.py` lacks — not a pure duplicate.
- **info_loss_risk:** Moving/deleting without fixing the two importers' `sys.path` breaks `sweep_runner.py` and `run_inc7_lap.py` at import time. Keep resolvable on the path those scripts set; if relocating, co-locate with the importers and re-run to confirm.

## [presentability · low] _spec.txt filename leading-underscore
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/_spec.txt`
- **Evidence:** Filename leads with an underscore (`_spec.txt`) and sits at repo root alongside the PDF. Memory references it by this exact name (`project_parked_backlog.md:91` "_spec.txt §4.3/§4.5", `project_master_plan.md:66` "extracted _spec.txt at root").
- **Recommended action:** KEEP the name as-is. The leading underscore looks untidy but RENAMING is net-negative: two memory files cite it by literal name. If presentability ever matters more than the citation cost, rename to `spec_text.txt` AND update `project_parked_backlog.md:91` + `project_master_plan.md:66` in the same change. Otherwise leave it.
- **info_loss_risk:** Renaming without updating the two memory citations turns them into dangling references. Preserve by editing both citations atomically with any rename.

## [prune · low] conftest.py + pyproject.toml staleness check (no staleness — KEEP)
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/conftest.py:1-4` and `pyproject.toml:1-35`
- **Evidence:** `conftest.py:1-4` is a 4-line src/ path shim (`sys.path.insert(0, str(Path(__file__).parent / "src"))`) — duplicated by `pyproject.toml:36` `pythonpath = ["src"]` but harmless/standard. `pyproject.toml` references only live modules: name='racer', requires-python ">=3.13,<3.14" (matches README), packages=['src/racer'], optional deps detector (ultralytics/albumentations) and planning (toppra) — all current in memory. No removed-module references. Last touched in the May scaffold era (conftest `b011009` 2026-05-27; pyproject `01ec973` 2026-05-31).
- **Recommended action:** KEEP both unchanged. Reference no retired modules/increments. The conftest/pyproject src-path duplication is conventional (conftest covers ad-hoc invocation, pyproject pytest.ini_options covers pytest). No staleness found.
- **info_loss_risk:** none.

---

# Structural / pointer gaps & root-cause fixes — issue_type: gap / presentability

## [gap · high] Git UNMERGED branches; one is the user live checkout (do NOT prune)
- **Location:** `claude/reverent-shirley-b94605`; `inc8-deterministic-retrain` (= live tree `Anduril`)
- **Evidence:** `branch --no-merged main` lists both; `git -C Anduril rev-parse HEAD` = `inc8-deterministic-retrain` (active checkout, not orphan); `reverent-shirley` cherry `+1/-0` = one unmerged handoff-doc commit.
- **Recommended action:** Exclude both from any prune sweep. `inc8-deterministic-retrain` backs the live working tree; leave alone. `reverent-shirley`: cherry-pick its handoff doc or retain until banked.
- **info_loss_risk:** HIGH if a cmdr-rooted sweep mistakes `inc8-deterministic-retrain` for an orphan; it is the live checkout.

## [gap · high] Live-only untracked artifacts invisible to -cmdr audit
- **Location:** live: `handoff/inc8-deterministic-retrain-2026-06-19/probe_logstd.b64`; `handoff/inc8-reach-rate-2026-06-19/`
- **Evidence:** `git -C live status`: both untracked; in `-cmdr` `inc8-reach-rate-2026-06-19` absent and `probe_logstd.b64` absent; live REPORT.md dated Jun 20, post-correction.
- **Recommended action:** `probe_logstd.b64` = b64 binary, never git-add (artifact-pipe), delete-in-place from live only. `inc8-reach-rate-2026-06-19` is NEW post-correction work, live only; commit/bank from live before consolidation. Do not action from `-cmdr`.
- **info_loss_risk:** HIGH for `inc8-reach-rate-2026-06-19` — untracked live-only post-correction follow-up. Commit from live first.

## [gap · med] all handoff files tracked — no artifact-pipe gitignore coverage (root-cause fix)
- **Location:** `.gitignore:38-43` (covers `*.pt`, `*.pth`, `rl/checkpoints/*.json`, `data/runs/` — but NOT `*.zip`, `*.npz`, `*.npy`, `*.tfevents`, handoff PNGs)
- **Evidence:** `git ls-files handoff` = 1082; `find handoff -type f` = 1082 → 100% tracked, including the 9 MB zip, 4.8 MB of tfevents, 22.8 MB of PNGs (93 files, 83 over 100 KB). `.gitignore:38-43` only ignore `*.pt`/`*.pth` and `rl/checkpoints`. The artifact-pipe rule has no enforcing gitignore entry for the binary types that actually landed in handoff.
- **Recommended action:** Add a handoff-artifacts block to `.gitignore`: `handoff/**/*.png`, `handoff/**/*.zip`, `handoff/**/*.npz`, `handoff/**/*.npy`, `handoff/**/*.tfevents`, `handoff/**/*.err` (and `*.out`). Pair with re-homing the existing tracked binaries (prune findings above) so future handoff dirs cannot re-violate the rule. This is the root-cause fix that prevents recurrence.
- **info_loss_risk:** none — forward-looking ignore rule; does not touch existing content. Re-homing of already-committed binaries is handled by the individual prune findings.

## [gap · med] dangling section pointer §INC8-DESIGN (3 internal + 2 external refs)
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/memory/project_phase2_rl_vision_decisions.md:1657, :1702` (+ external `memory/index_rl_training.md:65`)
- **Evidence:** Line 1657 "...layered on **§INC8-DESIGN** + §S2-DECISION." and 1702 "C5 — inc8 retrain spec (see **§INC8-DESIGN BLUEPRINT below**)." both reference a section that DOES NOT EXIST (grep `^##`/`^###` returns no INC8-DESIGN header; nearest is §GATE-RELATIVE-BLUEPRINT "inc8 retrain spec" at 1705). Worse, external `index_rl_training.md:65` also points to "[[project-phase2-rl-vision-decisions]] §INC8-DESIGN" and `:127` lists "§INC8-DESIGN" among this file's sections — a dangling cross-file pointer.
- **Recommended action:** Either (a) rename the §GATE-RELATIVE-BLUEPRINT "inc8 retrain spec" subsection (line 1705) to "§INC8-DESIGN" so the 3 internal + 2 external index refs resolve, or (b) fix the 5 pointers to read "§GATE-RELATIVE-BLUEPRINT > inc8 retrain spec". Verify against `index_rl_training.md` lines 65 and 127.
- **info_loss_risk:** none — the inc8 retrain spec content exists at 1705-1713; this is a label/pointer mismatch, not missing content.

## [gap · low] dangling section pointer §GATE-3-FALSIFICATION
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/memory/project_phase2_rl_vision_decisions.md:1283`
- **Evidence:** Line 1283 "Gate-3 1.46 m mis-registration FALSIFIED (see **§GATE-3-FALSIFICATION**)." references a header that does not exist; the content is the "### Gate-3 falsification (terminal, compact)" heading at line 1285. Heading-name vs pointer-name mismatch (capitalization/slug).
- **Recommended action:** Change the reference to "(see ### Gate-3 falsification below)" or rename the 1285 heading to "### GATE-3-FALSIFICATION" for an exact match. Trivial.
- **info_loss_risk:** none — target content present at 1285-1287.

## [presentability · low] trailing block not under any parent ## section
- **Location:** `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/memory/project_phase2_rl_vision_decisions.md:1730-1733`
- **Evidence:** The final "### Ranked residual risks (top 3) — UPDATED 2026-06-14 by delta_map discriminator" block (1730-1733) sits under §GATE-RELATIVE-BLUEPRINT but its three risk items (#1 gate-4 p90/p99 does-not-close; #2 boresight binding lever; #3 in-loop latency) are the same stale-magnitude claims flagged above, and the file simply ends there with CRLF line endings (whole file is CRLF, confirmed via `cat -A`). It is the file's final word and reads as the current top-3 risk list.
- **Recommended action:** Fold the 2026-06-19 supersession into these three ranked risks (risk #1 and #2 are downgraded per the gate-4 correction; risk #3 latency is unaffected). As the file's concluding "current top risks" list, it is the highest-leverage place to land the correction rider so a reader does not take stale risks #1/#2 as live.
- **info_loss_risk:** none — risk #3 (eval-HW latency) survives unchanged; risks #1/#2 need the same dated rider as the body sections, not deletion.

## [prune · med] Git content-merged branches safe-to-prune
- **Location:** `p2-inc8-rl, p2-inc8-rl-harvest, p1-calib-v2, calib-v2-apply2, oneoff-regsuite, worktree-wf_43064dee-ce7-1, p2-inc8-warmstart, p2-substrate-diagnose, p2-substrate-greengate, p2-system-id, claude/stoic-cray-33a854`
- **Evidence:** `cherry main`: six `+0/-0` merged-ancestry; warmstart/substrate-diagnose/stoic-cray `+0/-1` (tip patch in main); substrate-greengate `+1/-0` but `scripts/green_gate.py` present in main; p2-system-id `+10/-0` but squash `bc182f9` IS in main. All 17 worktree dirs present; prune dry-run empty.
- **Recommended action:** Content-captured in main; worktree dirs safe-to-prune after confirming no live agent. Prune merged branch refs at the gate; only `git worktree prune` the harness pool. Verify system-id and substrate-greengate by file presence, not ancestry.
- **info_loss_risk:** none — content verified in main; branch-ref deletion is reflog-reversible.

---

## Cross-cutting notes & recurring patterns
- **One dominant theme:** the 2026-06-19 gate-4 σ_p0 bar correction (`511e85c`: 0.08 was a double-count → real bar ~0.15, p99 ~0.45; near-field-estimator pivot RETRACTED, RL stays primary). It is landed in `MEMORY.md:8` but un-propagated into **5 code/launcher instruments** (`inc8_sigmap0_eval.py`, `inc8_sigmap0_torch_eval.py`, `inc8_sigmap0_torch.sbatch`, `peregrine_inc8_warmstart.sbatch`) and **~15 memory/handoff locations** (`project_phase2_rl_vision_decisions.md` ×9 blocks, `index_vision_estimator.md`, 3 handoff REPORTs, `project_parked_backlog.md`, `project_rl_increment_history.md`). Discipline: EDIT thresholds/prose in live code & sub-indices; BANNER (do not rewrite) dated historical handoff REPORTs and the inc8-ladder history; one forward-pointer suffices for point-in-time records.
- **Second theme:** artifact-pipe enforcement gap. ~38 MB of tracked binaries in `handoff/` (9 MB zip, 22.8 MB PNGs, 4.8 MB tfevents, npz/npy) + retired-increment checkpoints, with NO gitignore coverage. The root-cause fix is a single `.gitignore` handoff-artifacts block; every binary re-homes to a `burn-artifacts-*` release asset FIRST (never plain-delete) then `git rm --cached`.
- **Premise-correction caveat:** all `-cmdr` findings are valid for the canonical `main @ c5d60a0`. The live branch `inc8-deterministic-retrain @ 05809ba` is an OLDER divergent sibling (6 ahead / 7 behind) holding the pre-correction state; its staleness is a merge-time reconciliation item, NOT an audit error, and was correctly NOT read.
- **Numeric reconciliation flagged twice:** `green_gate.py` sentinel is **933** on disk (both branches; the "1077" was a runtime pass-count in a commit message, not the constant), while MEMORY states the suite is **947**. Decide which is canonical before pinning either into README.
- **Resolved-footgun to retire:** `fly_rl.py` default ckpt is NOW inc7 (fixed in code); MEMORY/`index_rl_training.md` still warn "default = inc4" — mark RESOLVED, keep the historical lesson.
