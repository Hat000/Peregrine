# Peregrine Cleanup Plan (audit 2026-06-19, main @ c5d60a0)

> Master assembler note: this plan is CURATED (HIGH-severity first, contradictions first, deduped across the seven audit areas). The per-area appendices under `handoff/cleanup-audit-2026-06-19/` hold the exhaustive med/low detail — each work-list section cross-references its appendix. Audit root = `main @ c5d60a0` (canonical). The live working tree (`inc8-deterministic-retrain @ 05809ba`) is an OLDER divergent sibling (6 ahead / 7 behind), so its stale pre-correction text is a merge-time concern, NOT an audit error — do not "fix" the live branch from the audit root.

---

## Executive summary

**Counts by issue_type (across all 7 areas):**

| issue_type     | count |
|----------------|-------|
| contradiction  | 45    |
| stale          | 154   |
| duplication    | 83    |
| gap            | 23    |
| prune          | 48    |
| presentability | 21    |
| **total**      | **375** |

**The single biggest theme — ONE root error, fanned out everywhere.** On 2026-06-19 (commit `511e85c`) Fengyou caught that the gate-4 `sigma_p0` "**0.08 m closure bar**" was a **double-count**: `margin_envelope.py` subtracted the drone's extent TWICE (`W_EFF = 0.75 − 0.215 chassis`, then `MARGIN = W_EFF − r`), and 0.08 = 0.235/3 inherited that. The real gate clearance is **~0.37–0.47 m**; the real bar is **sigma_p0_lat ~0.15 (p99 ~0.45)**; measured **sigma_p0 0.15–0.20 is MARGINAL-PASSING, NOT a NO-GO**. The "pivot off RL to a near-field-gate estimator" recommendation was **RETRACTED** — RL stays the right tool; gate-4 is a **reach/pass-rate** problem (gate sits off the racing line), not a sub-cm centering problem. **This correction landed ONLY in `MEMORY.md:8`.** It was never banked down-layer, so the retracted 0.08-bar / NO-GO / near-field-pivot doctrine still reads as CURRENT TRUTH across ~30+ locations: 2 sub-indices (`index_vision_estimator`, `index_rl_training`), 3 topic files (`project_phase2_rl_vision_decisions`, `project_rl_increment_history`, `project-fullstack-burn`), ~14 handoff REPORTs, the design-doc origin, **and — most dangerously — 5 LIVE code/launcher instruments** (`rl/inc8_sigmap0_eval.py`, `rl/inc8_sigmap0_torch_eval.py`, `rl/inc8_sigmap0_torch.sbatch`, `cluster` selectors) that still default `sigma_target=0.08` and will SILENTLY re-stamp a false NO-GO if re-run.

**The second theme — stale auto-loaded directives + un-enforced artifact pipe.** `MEMORY.md:4` still asserts "**main LOCAL-ONLY, far ahead of origin — do NOT push**"; this audit VERIFIED FALSE (`local main == origin/main == c5d60a0`, left-right count 0/0). And the "binaries NEVER git-add" rule is unenforced: ~38 MB of tracked `.zip`/`.png`/`.tfevents` artifacts sit in `handoff/`.

**State of the mess.** The memory ROOT (`MEMORY.md`) is current and correct; everything BELOW the root lags by a phase. The mandated fix ORDER is non-negotiable: **(1) bank the correction DOWN-LAYER first** (the newest facts — det reach 0.467, std-cap lever, corrected bar — exist ONLY in `MEMORY.md:8` + the handoff REPORT; trimming before banking = real info loss), **(2) propagate correction banners** into indices/topic files and banner-annotate (never delete) the frozen handoff REPORTs, **(3) fix the 5 live code instruments**, **(4) THEN** compress the 4.5 KB `MEMORY.md:8` megaline and split the 185 KB `project_phase2` monolith. Info-loss risk is near-zero because every measured number, root-cause, and build artifact is already banked-or-merged — the cleanup is overwhelmingly *banner-correct-then-archive*, with a small set of HARD ACTIVE FIXTURES that must stay in place.

---

## Top ~10 must-fixes

Highest-leverage first. Contradictions and the gate-4/0.08 stale-survivors lead; the false "do-not-push" directive is near the top because it is auto-loaded into every commander boot.

1. **Bank the gate-4 correction DOWN-LAYER, THEN collapse the megaline — action:** bank the det-retrain + bar-correction from `MEMORY.md:8` into `project_rl_increment_history.md §inc8` + `index_rl_training.md` + `index_vision_estimator.md`; verify pointer targets exist; THEN collapse the 4.5 KB `MEMORY.md:8` to the corrected verdict + LIVE SIGN FOOTGUN config only. **— location:** `MEMORY.md:8` (4531 B, 18.3% of an over-cap file). **— why:** `det reach`/`0.467`/`std-cap`/`MARGINAL-PASS` appear ONLY in `MEMORY.md` — the load-bearing newest facts are stuck UP in the over-limit index while topic files narrate the SUPERSEDED verdict. Corrected value: real bar `sigma_p0_lat ~0.15 (p99 ~0.45)`, measured 0.15–0.20 = MARGINAL-PASSING. **HIGH info-loss risk if trimmed before banking** — bank FIRST.

2. **Fix the 2 LIVE sigma-eval instruments (code, not doc) — action:** change `_go_verdict(sigma_target=0.08)` and `--sigma-target default=0.08` → `0.15`, and p99 `~0.24` → `~0.45`, in BOTH `rl/inc8_sigmap0_eval.py` (`:174,:199`, docstring `:5,:27-28`) and `rl/inc8_sigmap0_torch_eval.py` (`:223,:292`, docstring `:47-48`); add a "`0.08 was a pre-2026-06-19 double-count, corrected @ 511e85c`" note; keep the two tools in sync. **— location:** the two evals are the laptop-validated instruments actually run on Adroit (the torch one is wired into `inc8_select_ckpt.sbatch` / `peregrine_inc8_detstab.sbatch`). **— why:** CONFIRMED live on disk this audit (`sigma_target: float = 0.08`); a re-run today would silently brand a marginal-PASSING policy NO-GO. Corrected bar: `sigma_p0_lat <= 0.15 AND p99 <= 0.45`.

3. **Remove the false "main LOCAL-ONLY / do-not-push" directive — action:** replace the `MEMORY.md:4` clause with "`main PUSHED & == origin/main (c5d60a0)`; ShadowPC-main reconciliation via origin still deliberate (fetch+inspect before merging)." KEEP the ShadowPC-divergence caution. **— location:** `MEMORY.md:4` (auto-loaded). **— why:** VERIFIED FALSE this audit — `git rev-list --left-right --count main...origin/main` = `0  0`; `HEAD(main) == origin/main == c5d60a0`. The directive misdirects the next commander into NOT pushing already-pushed work.

4. **Propagate the correction into `index_vision_estimator.md` — action:** add correction banners — `:285` "`0.15 / p99 0.45 [CORRECTED 2026-06-19]`" (this is the ORIGIN of 0.08 = 0.235/3); `:286-287` demote the near-field estimator from "THE centering lever / INVEST-2" to vision SUPPORT/insurance; `:52-65` budget-identity annotate "double-counted, real clearance = 0.75−r (~0.45 @ r=0.30, ~0.37 @ r=0.38)"; §TERMINAL-LOCK + §COLD-MARGIN heads marked SUPERSEDED/RE-JUDGED. Do NOT blanket-reverse CANNOT-SETTLE-OFFLINE (its narrow core — at-speed sigma needs a physical drone — stays true). **— location:** `index_vision_estimator.md:52-65,:80-86,:258-259,:264,:285,:286-287`. **— why:** index last touched `d688923` (2026-06-15), predating the correction; the whole 0.08 boundary + "fix-rate ≥0.50 through last 6 m GEOMETRICALLY UNACHIEVABLE" wall rests on the retracted bar. Gate-4 "nearly passes WITHOUT vision fixes."

5. **Banner-correct the inc8 det-retrain handoff REPORT (do NOT delete) — action:** prepend a dated SUPERSEDED banner at top + beside HEADLINE/CONCLUSION/MEMORY-DELTA: "SUPERSEDED 2026-06-19 (511e85c): 0.08 bar was a double-count; real ~0.15/p99~0.45; 0.15–0.20 is MARGINAL-PASSING not NO-GO; near-field-estimator pivot RETRACTED, RL stays the tool." Keep all engineering sections verbatim. **— location:** `handoff/inc8-deterministic-retrain-2026-06-19/REPORT.md:89,101,113-135,144-150` (ON main @ c5d60a0). **— why:** the REPORT asserts the OLD view ("Objective B = NO-GO vs 0.08", "NEXT LEVER = near-field-gate estimator … NOT look-at RL") and `MEMORY.md:8` even cites this exact path while contradicting it. KEEP: rc1 actor_logstd saturation (exp(2)=7.39 on 2 channels), det reach 0.467, the sigma_p0 table, the std-ceiling lever.

6. **Correct the false "36-dim asymmetric PRIVILEGED critic" claim — action:** edit all sites to state the inc8 critic is SYMMETRIC (obs-input 20) under `algo=ppo`; the 36-dim privileged critic is an UNBUILT `algo=appo` option (`get_state` 33→36 is NEVER consumed in the active path). **— location:** `index_rl_training.md:47,:81,:100` + `project_phase2_rl_vision_decisions.md:479,:1624,:1698`. **— why:** MEMORY footgun, CONFIRMED ×2 (static launcher evidence + the S3 rw=1.0 cliff-collapse) — this is the ROOT-CAUSE of the seed collapse; the stabilizer was never connected. Preserve the get_state DESIGN intent (value IF built via appo) at `project_rl_increment_history.md:738/752`.

7. **Un-park RL in the master plan — action:** update `project_master_plan.md:99,:157,:170` from "RL = Track B, parked" → "RL COMMITTED as the VQ2 path (2026-05-29); inc8 is the live workstream → [[index-rl-training]]"; reframe `:157` SWIFT-ablation as "why RL was INITIALLY parked (later overturned)." **— location:** `project_master_plan.md:99,:157,:170` (contradicts its OWN `:121` "NO LONGER PARKED. COMMITTED" and `index_strategy_meta.md:15`). **— why:** the plan frames the LIVE workstream as parked; SWIFT-ablation rationale kept as historical "why."

8. **Close the canonical inc8-ledger gap — action:** append a new dated (~2026-06-19) `§inc8` entry to `project_rl_increment_history.md` (after L750): det-stability retrain (logstd-saturation root cause, std-ceiling clamp, FIRST det-flyable 2-axis inc8, det reach 0.467, `42ddb0b`; bar correction `511e85c`); add "BAR CORRECTED 2026-06-19" to the older 0.08 ladder bullets (`:734-750`); mirror into `index_rl_training.md` (inc8 block frozen at 2026-06-16, missing S2-confirmed / RECENTER-flew-3/3 / det-reach-0.467). **— location:** `project_rl_increment_history.md` (after L750) + `index_rl_training.md:99-124`. **— why:** the canonical episodic ledger ends on a STALE NO-GO and omits the 2 decisive milestones; a reader is left at a stale NO-GO. Additive — do NOT prune the irreplaceable S0→S5 ladder.

9. **Add a SUPERSEDED guard to the margin engine (do NOT archive standalone) — action:** add a header comment / `SUPERSEDED.md` to `handoff/margin-closure-envelope-2026-06-14/margin_envelope.py`: "`W_EFF` double-counts the drone (0.215 is the linf0 radius-invariant crossing offset, NOT a second chassis-radius subtraction); real clearance = 0.75−r (~0.37–0.47); real bar sigma_p0_lat ~0.15. DO NOT reuse `margin_at()`/`W_EFF`." **— location:** `margin_envelope.py:33,:77-78,:101-103,:352-353` (CONFIRMED this audit). **— why:** this is the EXACT double-count `MEMORY.md:8` names; it is an ACTIVE engine reused by boresight-closure/coast-drift and cited BY NAME in the correction — keep it as the worked example, annotated, not deleted.

10. **Bump the green_gate test-count sentinel — action:** update `DEFAULT_BASELINE = 933` → the confirmed live collect (full-stack `pytest --collect-only` = **1091**, re-confirmed this audit on disk; MEMORY independently flags 933 as stale-low and cites 947); update the 3 docstring/comment mentions; keep `723/884/933/1077` as lineage history. **— location:** `scripts/green_gate.py:67` (docstring `:27-29,:40,:66`). **— why:** the floor lags real count by ~158 tests, so the gate passes even if up to 158 tests silently vanish — the EXACT silent-test-loss class the sentinel exists to catch. (Reconcile the 933-vs-947-vs-1091 discrepancy before pinning a number; `1077` in commit `fe31d62` is a runtime pass-count, not the constant.)

---

## DO NOT TOUCH — load-bearing warnings

The cleanup must NOT "fix" any of these. They look like bugs/stale/duplication but are deliberate or active dependencies.

- **`MEMORY.md:8` LIVE SIGN FOOTGUN launch config** — `++env.lookat_g_yaw=-3.0 ++env.lookat_g_pitch=3.0` (DOUBLE-plus; both empirical, OPPOSITE signs — analytical was WRONG both times) + `+env.lookat_warmup_updates=0` + `+env.rw_centering=1.0` + per-seed `hydra.run.dir`. This MUST stay at top level through any megaline collapse.
- **CTBR/VQ1 legacy sign config** is a self-consistent alias — DO NOT "fix" it.
- **Contact radius 0.38** = EMPIRICAL crash halo / worst-case stress knob, NOT a geometry bug (rigid-body cap is 0.2135 m). Central reporting radius = 0.30 m; keep 0.38 as the stress knob.
- **OBS SIGN = +L is CORRECT in code** (`obs_from_zup:348`, `localization:86`; pinned by `tests/test_obs_sign_faithfulness.py`). The d1/d2 spec PROSE saying "estimator delivers −L" is the wrong one — do NOT flip the code.
- **ODOMETRY quat is R_y(π)-CONJUGATED** — this is the known convention; run `scripts/frame_residual_report.py` after every live session. Do NOT "un-conjugate."
- **Harness-managed `.claude/worktrees/*` pool** — RECYCLES; NEVER manually prune (racy, can knife a launching session). Only `git worktree prune` (gone dirs) + remove confirmed-DONE MANUAL worktrees.
- **Live working-tree branch `inc8-deterministic-retrain @ 05809ba`** backs the active checkout — EXCLUDE from any prune sweep; it is the live checkout, not an orphan.
- **HARD ACTIVE FIXTURES — KEEP IN PLACE, do NOT archive/move:**
  - `handoff/shadowpc-firstcontact-2026-06-02/track_map.json` — the ONLY copy of the canonical 6-gate course map; loaded by **26** `.py` consumers (`tests/test_navigator.py`, `src/racer/gate_mapper_synth.py`, `rl/peregrine_course.py`, …).
  - `handoff/shadowpc-followups-2026-06-05/sysid/` + `task2_frames/` — committed fixtures consumed by `tests/test_twin_fit.py`, `scripts/fit_twin.py`, `scripts/characterize_perception.py`, `task2_gate_pnp.py`, `blender_gen/contract.py`.
  - `handoff/shadowpc-postfix-dataset-2026-06-12/extracted/` — git-tracked R_y(π)-mirror positive-control fixture for `tests/test_diagnose_session.py` + `tests/_audit_io.py`.
- **Named source-provenance WRITEUPs** cited by `rl_plant.py`/`twin_fit.py`/`test_*.py` — keep in place.
- **`margin_envelope.py`** — annotate-as-superseded, do NOT archive standalone (active engine, cited by name — see must-fix #9).
- **Live untracked post-correction work** `handoff/inc8-reach-rate-2026-06-19/` (HIGH info-loss risk) — commit/bank FROM THE LIVE TREE first; do NOT action from the audit root.

---

## Prioritized work-list (by AREA → ISSUE_TYPE, HIGH first)

Every HIGH `high_item` from the seven area summaries appears below. For med/low items, see the cited appendix.

### AREA: memory — appendix_memory.md (contradiction 12, stale 58, duplication 22, gap 6, prune 4, presentability 15; total 117)

**Contradictions (HIGH)**
- **Gate-4 0.08-bar survivors** — `MEMORY.md:8` + `index_rl_training.md:119` + `index_vision_estimator.md:285` + `project-fullstack-burn.md:17` + `project_phase2_rl_vision_decisions.md:1484/1487/1491/1507/1591` + `project_rl_increment_history.md:734-750`. *Action:* propagate correction banner + inline supersession into each canonical home; replace "0.08 (the GO)" with "0.15 / p99 0.45 (0.08 was a double-count, RETRACTED 2026-06-19)"; preserve original derivation as labeled-historical. *Evidence:* `index:285` "r=0.30 CLOSES IFF sigma_p0_lat ≤ 0.08 m"; `history:750` "NO-GO vs 0.08". *info_loss:* none — corrected bar + rationale fully in `MEMORY.md:8` + REPORT; keep r-ladder relationship + pointed-vs-raw sigma decomposition.
- **Near-field estimator mis-ranked as THE lever** — `index_vision_estimator.md:286-287` + `project_parked_backlog.md:78(#70),:83(#75),:94` + `project_phase2_rl_vision_decisions.md:1566,:1731` + `project-fullstack-burn.md:56`. *Action:* re-rank from "THE centering lever/INVEST" to vision SUPPORT/insurance (parked secondary); state RL reach/pass-rate is the primary lever; in #70/#75/L94 keep the surviving true point (VIO/twin world-loc does NOT help gate-4). *Evidence:* `MEMORY.md:8` "pivot off RL to a near-field-gate estimator … RETRACTED; next lever = RL". *info_loss:* none — survives as banked insurance ("2nd lever").
- **36-dim asymmetric privileged critic claimed active** — `index_rl_training.md:47,:81,:100` + `project_phase2_rl_vision_decisions.md:479,:1624,:1698`. *Action:* state critic is SYMMETRIC (obs-input 20) under `algo=ppo`; 36-dim is an UNBUILT `algo=appo` option. *Evidence:* footgun CONFIRMED ×2; "GuardedPPO extends symmetric PPO; `get_state(33→36)`/`state_dim` NEVER consumed". *info_loss:* none — get_state DESIGN intent preserved (`history:738/752`).
- **RL framed as parked** — `project_master_plan.md:99,:157,:170` (vs own `:121`). *Action:* update to "RL COMMITTED as the VQ2 path (2026-05-29); inc8 live". *info_loss:* none — SWIFT-ablation kept as historical "why".
- **CTBR Sec-7 altitude-balloon reads as live OPEN BLOCKER** — `project_ctbr_control_sysid.md:231-251` (vs own `:3,:27-38`) + `index_control_sim.md:15`. *Action:* insert inline "[SUPERSEDED 2026-06-07 — balloon re-diagnosed as delay-driven relay, position-harmless, accepted for VQ1]" at `:231,:240`. *info_loss:* none — diagnostic + next-levers stay as history.
- **Adroit scoped "VQ2 ML training only"** — `reference_adroit_princeton.md:3,:19-22` (vs own body `:29-46` DiffAero RL bake-off) + `index_rl_training.md:69-75`. *Action:* broaden frontmatter+Role to "ML training substrate: YOLO-pose detector AND inc8 RL (DiffAero PPO)"; add RL workflow at `:22`. *info_loss:* none — additive.

**Stale (HIGH)**
- **`MEMORY.md:4`** "main LOCAL-ONLY … do NOT push" — VERIFIED FALSE (`== origin/main == c5d60a0`). *Action:* replace with "main PUSHED & == origin/main"; keep ShadowPC caution. *info_loss:* none.
- **`index_rl_training.md` inc8 block (99-124)** frozen at 2026-06-16 ("S1 DISPATCHED"); grep for recenter|0.467|deterministic = ZERO. *Action:* add a dated top-of-inc8 summary (S2 confirmed; recenter flew 3/3; det reach 0.467) + pointer; mirror `MEMORY.md:8`. *info_loss:* none — additive.
- **`project_rl_increment_history.md` (after L750)** — ledger ends at 2026-06-18 yaw fix; det-retrain chapter exists ONLY in `MEMORY.md:8` + REPORT. *Action:* bank a new ~2026-06-19 closing bullet; add "BAR CORRECTED 2026-06-19" to the 0.08 ladder bullets; do NOT prune the S0→S5 ladder. *info_loss:* none — resolves a gap.
- **`index_vision_estimator.md:285,:286-287,:264,:258-259,:13/:183/:294/:324/:337`** — last touched `d688923`; 0.08 boundary + "fix-rate ≥0.50 GEOMETRICALLY UNACHIEVABLE" + "CANNOT-SETTLE-OFFLINE SURVIVES" (×5) rest on the retracted bar. *Action:* add banners; demote near-field; do NOT blanket-reverse CANNOT-SETTLE (narrow core stays). *info_loss:* none.
- **`project_phase2_rl_vision_decisions.md:1481,:1485-1486,:1562,:1689,:1700,:1731,:1334,:1353,:1407,:1409`** — case-C absolute-nav NO-GO / `<0.05 m` KF target / "0.155 m margin DOES-NOT-CLOSE" all rest on the voided budget chain. *Action:* single SUPERSEDED header at head of §ESTIMATOR-RACESPEED; re-anchor on 0.37–0.47 m; drop `<0.05`; best path = collapse to a pointer into `index_vision_estimator.md` once that index carries the correction. *info_loss:* none — durable sub-findings mirrored.
- **`project_master_plan.md` (whole; `:2,:10`)** — self-describes as "single source of truth, update in place" but last substantive edit 2026-05-31. *Action:* demote `:2/:10` to "FROZEN Phase-1 architecture baseline (2026-05-31); LIVE truth in MEMORY.md + domain indexes" + dated FREEZE banner. *info_loss:* none — demotion, not removal.
- **`project_detector_training_pipeline.md:93-109`** — terminal "SHIP v2 … wire v2 into navigator.py"; superseded by VQ2 ensemble (`282abb9`, `06876bc`). *Action:* top supersession banner; keep v2 saga as history. *info_loss:* none.
- **`project_hardware_constraint.md:31-38,:15`** — "TensorDock RTX 3090 / Azure A10 = the sim host"; canonical sim host is now ShadowPC. *Action:* dated banner; mark Azure/TensorDock rows historical (keep as fallback). *info_loss:* none.
- **`reference_sim_ops.md:70` (+49-82); `:29`; `index_control_sim.md:47,:82`** — "NOT merged to main / NOT yet on main" — FALSE (merged `6876f44`, `1b54cb8`). *Action:* update to MERGED; drop the "pending" clauses. *info_loss:* none — diagnostic detail stays.
- **`index_control_sim.md:4-8 (esp :5)`** — "pos+vel GIVEN (LPN 97 Hz + ODOMETRY 75 Hz)" stated flat. *Action:* add a one-line flag: these are OUR practice-sim facts; the OFFICIAL scored wire has NO position (§4.3) → deployed stack MUST self-localize (case-C). *info_loss:* none.
- **`project_parked_backlog.md`** — 6 self-CLOSED items (#3/#15/#16/#37/#40/#62) sit in open FOLD-NOW; #74 (DONE) bundles an open carry-forward; #58 duplicated (`:30` vs `:69`); #32/#33 trigger fired; A1 (`:91`) date elapsed; register lags the gate-4 correction. *Action:* one consolidation pass — relocate 7 closed→DEAD; SPLIT #74; collapse #58; re-triage #32/#33; A1→"as of 2026-06-20 wire NOT confirmed dropped, checklist armed"; propagate gate-4 reach/pass-rate correction into #70/#75/L94. *info_loss:* none for relocations; MEDIUM for #74 (SPLIT, don't collapse).

**Gap (HIGH)**
- **`project_master_plan.md:72,:108,:126,:25(c)`** — treats self-localization as an OPEN hole; now resolved doctrine. *Action:* add a pointer under Sim-interface: "POSITION DOCTRINE RESOLVED — deployed stack must self-localize; gate-relative obs is the binding fix; absolute world-frame nav = NO-GO. See [[index-vision-estimator]] + MEMORY.md." *info_loss:* none.

**Prune (HIGH)**
- **`MEMORY.md:8`** (4531 B megaline, ≥8 distinct items) — *Action (SEQUENCE):* (1) bank det-retrain + bar-correction DOWN; (2) verify pointer targets; (3) THEN collapse line 8 to the corrected gate-4 verdict + LIVE SIGN FOOTGUN config, route the rest to one-line pointers. *info_loss:* **HIGH if trimmed before banking** — newest facts are NOT yet in any down-layer file. Bank FIRST.
- **`project_phase2_rl_vision_decisions.md` (185 KB monolith)** — *Action:* SPLIT/route — collapse each retired session to a one-line `[[pointer]]` + its `handoff/` path; keep only the unique 2026-06-07 planning decisions; freeze banner first; confirm each section's handoff path + canonical entry exist before collapsing; spot-check unique data (e.g. per-segment tilt table `1015-1023`) and MOVE first if unique. *info_loss:* most content duplicated in canonical files — verify before collapsing.

**Presentability (HIGH)**
- **`project_phase2…:1-14` + `project_master_plan.md` (no freeze marker)** — *Action:* add dated freeze/status banners (phase2 "current as of 2026-06-14; later verdicts in MEMORY.md + indexes"; master_plan "FROZEN 2026-05-31 Phase-1 baseline …"). *info_loss:* none — additive.
- **`project_parked_backlog.md:3-6` (F/T/D legend) + section placement** — *Action:* the consolidation pass above restores F/T/D membership to match the legend. *info_loss:* none.

*Med/low memory items → `handoff/cleanup-audit-2026-06-19/appendix_memory.md`.* Verified clean: wiki-pointer hyphen/underscore "broken links" are false positives; no orphan files; reference/feedback layer carries none of the stale gate-4 language.

---

### AREA: handoff — appendix_handoff.md (contradiction 13, stale 30, duplication 44, gap 1, prune 11; total 99)

**Contradictions (HIGH) — banner-correct-then-archive unless noted**
- **`inc8-deterministic-retrain-2026-06-19/REPORT.md:89,101,132-135,144-150`** — "Objective B = NO-GO … NEXT LEVER = near-field-gate estimator". *Action:* prepend dated SUPERSEDED banner → `MEMORY.md:8`/`511e85c`; keep measured numbers / rc1-saturation / std-ceiling lever. *info_loss:* none.
- **`rl/inc8_sigmap0_eval.py:174,199,237` + `rl/inc8_sigmap0_torch_eval.py:223,292,394`** (default `sigma_target=0.08`) — CONFIRMED live. *Action:* **FLAG UP to commander as a CODE change** (not archive): default 0.08→0.15, p99 multiple/comment/docstring; or a loud retracted-0.08 comment. *info_loss:* none.
- **`inc8-eval-pitch-2026-06-18/REPORT.md:8,107,115,148`** — "seed0 0.177 → NO-GO vs 0.08". *Action:* SUPERSEDED-VERDICT banner ("0.177 marginal per MEMORY.md:8; the YAW-injection fix below stands"). *info_loss:* none — injection fix `ec4cb03` banked.
- **`sigmap0-adroit-2026-06-19/REPORT.md:9-10,36,150-152,171,191`** — "0.198 → NO-GO vs 0.08". *Action:* annotate as JUDGED-AGAINST-WRONG-BAR (now marginal); keep instrument-validation + rc1 + g_pitch cliff. *info_loss:* MED — durable mechanism. **CARRY-FORWARD:** the "5th harness bug" fix to `rl/inc8_sigmap0_torch.sbatch` was "uncommitted, not pushed" (`:140,:193`) — **verify it landed; flag to commander.**
- **`coast-drift-2026-06-15/REPORT.md:36-41,142-168,184-199,241-256`** — 0.08 closure + near-field-as-lever. *Action:* dated CORRECTION banner (real bar ~0.15; near-field retracted); keep coast-is-cheap / sigma_v-non-binding. *info_loss:* MED — banked at `index_vision_estimator.md:284-287`. Note that index STILL carries the stale text — flag to fix memory too.
- **`boresight-closure-2026-06-14/REPORT.md:13-36,95,98-112,143-155,180-190`** — "NO-CLOSE", `W_EFF=0.535`. *Action:* CORRECTION banner; KEEP −0.25 boresight bake (DEPLOYED `5764291`) + anisotropic sigma-recal. *info_loss:* LOW.
- **`coast-drift…:189,199,248-249`** (near-field-as-THE-lever) — covered by the coast-drift banner; ensure RETRACTION reflected. *Note:* `MEMORY.md:8` itself still carries a residual "2nd lever = near-field GATE estimator" inside the SIGN FOOTGUN — surface to memory owner to reconcile. *info_loss:* none.
- **`body-contact-reconcile-2026-06-13/CORRECTED_MARGIN.md:32-35,52-65,113-170` + `BODY_RECONCILE_VERIFY.md:15-34,156-182`** — `budget(r)=(0.75-r)-0.215`. *Action:* CORRECTION banner to BOTH; KEEP geometry half (0.2135 ceiling, 0.38 halo, 0.30 central). *Note:* footers cite the STALE `C:/…/Anduril` tree (`CORRECTED_MARGIN.md:217-221`) — update to `Anduril-cmdr` if retained. *info_loss:* none for geometry.
- **`margin-closure-envelope-2026-06-14/margin_envelope.py:33,77-78,101-103,352-353`** — CONFIRMED. *Action:* **KEEP (annotate-superseded); do NOT archive standalone** — active engine, cited by name. Add header comment / `SUPERSEDED.md`. *info_loss:* none.
- **`_commander-xcheck/CROSSCHECK_FINDINGS.md:30-65`** — DOUBLY stale (0.155 margin AND the sigma_v-binding thesis, refuted by coast-drift). *Action:* annotate SUPERSEDED then archive. *info_loss:* LOW.

**Duplication (HIGH) — KEEP (load-bearing data fixtures), prose is banked**
- `shadowpc-firstcontact-2026-06-02/` (`track_map.json`, 26 consumers) — KEEP.
- `shadowpc-followups-2026-06-05/` (`sysid/`, `task2_frames/`) — KEEP.
- `shadowpc-reverify-2026-06-07/rung23/REPORT.md` — bank-then-archive (no hard code dep; confirm per-gate-miss numbers survive in `project_ctbr_control_sysid.md` first).
- `shadowpc-postfix-dataset-2026-06-12/` (`extracted/`) — KEEP.

*Med/low handoff items (the other ~85, incl. the 27 shadowpc dirs confirmed CLEAN of gate-4 staleness) → `appendix_handoff.md`.* Two non-archival exposures must not be missed: (1) the 2 live evaluators default 0.08; (2) the hard fixtures above + named source-provenance WRITEUPs MUST stay in place.

---

### AREA: commander — appendix_commander.md (contradiction 1, stale 2, duplication 1, gap 0, prune 0, presentability 2; total 7)

No HIGH items. The single MED need: re-tense `COMMANDER.md:49` ("inc8 isn't deterministically flyable" went stale after `42ddb0b`) to past tense, point live state to `memory/` — this also resolves the §3-rule-4 STATE-leak contradiction. Everything else low (intentional MEMORY overlap, historical Gen-4 lineage state, 2 cosmetic density notes). No deletions. *Detail → `appendix_commander.md`.*

---

### AREA: code — appendix_code.md (contradiction 3, stale 14, duplication 4, gap 4, prune 6, presentability 1; total 32)

**Contradictions (HIGH)**
- **`rl/inc8_sigmap0_torch_eval.py:4-5,:47-48,:223,:292`** — docstring/GO-RULE/default/argparse all hardcode 0.08 (this is the instrument run on Adroit). *Action:* set docstring + `_go_verdict` default + argparse default to 0.15 / p99 0.45; add "0.08 was a pre-2026-06-19 double-count, corrected @ 511e85c"; keep in sync with the numpy companion. *info_loss:* none.
- **`rl/inc8_sigmap0_eval.py:4-5,:27-28,:174,:199`** — numpy companion, same stale bar; emits a FALSE NO-GO. *Action:* 0.08→0.15 (`:174,:199`), p99 prose 0.24→~0.45; same note; keep both tools in sync. *info_loss:* none.
- **`cluster/vq2_pose_train.sbatch:33` vs `cluster/vq2_pose_train.py:57`** — sbatch launches `--batch 32`; the trainer warns "`32 silently changes the run`" (champion = 16, F-REPRO-1). *Action:* change sbatch `--batch 32`→`16` to match the champion, OR add an inline comment documenting the deliberate deviation. *info_loss:* none.

**Stale (HIGH)**
- **`scripts/green_gate.py:67`** `DEFAULT_BASELINE=933` (live collect = 1091; ~158-test silent-loss gap). *Action:* bump to 1091 after confirming collect; update the 3 mentions; keep `723/884/933` lineage. *info_loss:* none.
- **`rl/inc8_sigmap0_torch.sbatch:18,:127`** — encodes "sigma_p0_lat ≤ 0.08 AND lat_p99 ≤ 0.24" as the deliverable. *Action:* update to "≤ ~0.15 AND lat_p99 ≤ ~0.45"; optional double-count note. Commit msg `36b7730` is immutable; only in-file prose corrected. *info_loss:* none — 0.198 measurement stays valid.

*Med/low code items → `appendix_code.md`.* Everything else is low-severity stale doc pointers, lineage sbatch to archive (not delete), and benign cross-referenced duplication.

---

### AREA: git — appendix_git.md (contradiction 3, stale 6, duplication 0, gap 4, prune 12, presentability 1; total 26)

**Contradictions (HIGH) — banner-correct, do NOT delete/merge-as-current**
- **`inc8-deterministic-retrain-2026-06-19/REPORT.md` (ON main @ c5d60a0)** — HEADLINE/CONCLUSION/MEMORY-DELTA assert the OLD view. *Action:* prepend a dated CORRECTION banner beside HEADLINE/CONCLUSION/MEMORY-DELTA; keep engineering sections verbatim; annotate verdict/recommendation lines only. Branch ref prunable separately (code already on main via `42ddb0b`). *info_loss:* none from annotating; deleting would lose rc1 saturation + det reach 0.467 + sigma_p0 table — so annotate, don't delete.
- **branch `claude/charming-jemison-b111c5` (tip `7654e99`)** — `margin-closure-envelope` REPORT.md + 27 result files (NOT on main; only `margin_envelope.py` merged, byte-same via `3542fd1`) assert "MARGIN=0.235 / does NOT close / CANNOT-SETTLE-OFFLINE survives". *Action:* do NOT merge the REPORT/results as current truth; PRUNE the branch; if the 268-cell raw sweep has archival value, archive the dir with a one-line SUPERSEDED header (not a live merge). *info_loss:* low — methodology echoed in main memory; the stale VERDICT must NOT be preserved as current.

*Notes:* the redundant branches/worktrees are overwhelmingly prune-safe (12 prune entries, all zero-unique-content via ancestor/blob-SHA checks). Before pruning, bank two MED gaps: the p1-eskf-design observability verdict and the VQ2 F-SPLIT-1 staged dataset-split bug fix; preserve the ~5 memory-sanctioned artifact branches (blissful-kalam L3, dmap-discriminator, hardcore-lehmann rows.json, vq2-data orphan, shadowpc-artifact-push) whose raw data is reproducible only by re-recording. *Med/low → `appendix_git.md`.*

---

### AREA: contradiction (cross-cutting sweep) — appendix_contradiction.md (contradiction 5, stale 24, duplication 6, gap 1; total 36)

**Contradictions (HIGH)** — all trace to the single `511e85c` correction landing in `MEMORY.md` only:
- **`margin_envelope.py:33,77,78,101,102`** — `W_EFF` labels 0.215 "chassis half-diag" then subtracts the drone AGAIN; canonical meaning is 0.215 = linf0 radius-invariant offset. *Action:* correction header comment (0.215 is linf0 NOT chassis radius; the 0.08 bar this feeds was retracted `511e85c`); preserve as historical artifact; do not silently re-run. *info_loss:* none.
- **`index_vision_estimator.md:285`** — ORIGIN of 0.08 (0.235/3). *Action:* 2026-06-19 CORRECTION note at `:285` + §COAST-DRIFT header at `:282`; strikethrough 0.08/0.05/0.04; state real clearance 0.37–0.47 / bar ≤0.15 / measured 0.15–0.20 = MARGINAL-PASSING; pointer to `MEMORY.md:8`; keep sweep methodology. *info_loss:* low.
- **`index_strategy_meta.md:10`** — "case-C confirmed" (falsified for VQ1: official streamed LPN+ODOMETRY = case-A). *Action:* change to "position NOT on the SPEC'd scored wire (§4.3); BUT our VQ1 sim streamed position (case-A) → case-C is the DESIGN assumption, NOT confirmed — VQ2 is the resolver"; remove unqualified "case-C confirmed". *info_loss:* none.

**Stale (HIGH)** — `MEMORY.md:4` (do-not-push, covered above); `index_vision_estimator.md:52-65,:80-86,:286-287`; `index_rl_training.md:40,:50,:52`; `project-fullstack-burn.md:17` ("std ≤0.08" GO → "≤0.15 (p99 ≤0.45)"); `inc8-deterministic-retrain REPORT.md:89,101,106,132-135,144,149-150`; `reference_competition_materials.md:75` ("Q① resolved" → "resolved AT THE SPEC LEVEL but OPEN empirically — our VQ1 sim streams position; VQ2 is the resolver"). *Actions/evidence per item → `appendix_contradiction.md`.* *info_loss:* none-to-med (preserve budget arithmetic, linf concept, eps_vert measurement, delta_map discriminator, boresight bake — relabel only conclusions).

**Gap (HIGH)** — `project_rl_increment_history.md §inc8` ends ~L750 on the stale NO-GO; `deterministic-retrain|noise-anneal|42ddb0b|0.467|511e85c` ABSENT from this file AND all sub-index/topic files. *Action:* append a new §inc8 entry + propagate into `index_rl_training.md` + `index_vision_estimator.md §TERMINAL-LOCK`. *info_loss:* none — additive (same as memory must-fix #8).

---

### AREA: gap (forward-looking sweep) — appendix_gap.md (contradiction 8, stale 20, duplication 6, gap 7, prune 15, presentability 2; total 58)

**Branch-divergence framing (HIGH gap) — corrects an inverted premise:** `main @ c5d60a0` is the canonical NEWER corrected state; live `inc8-deterministic-retrain @ 05809ba` is an OLDER divergent sibling (6 ahead / 7 behind, diverged at `387c4f3`). The "live commits superseded the audit" framing has the diff direction INVERTED. NO finding computed against `c5d60a0` is invalidated. Corollaries verified: the EnsembleGateDetector exists on main (`detector.py:189`, `test_ensemble_detector.py` 309 lines) and is DELETED on live — scope any ensemble recommendation to main. `green_gate.py` sentinel is **933 on BOTH branches** (the "bumped to 1077" claim is FALSE; 1077 = a runtime pass-count in `fe31d62`). `MEMORY.md:8` corrected on main / old on live. **EXCLUDE live `inc8-deterministic-retrain` + untracked `handoff/inc8-reach-rate-2026-06-19/` + `probe_logstd.b64` from any cmdr-rooted sweep** — the reach-rate dir is HIGH-info-loss live-only post-correction work (commit/bank from live first); `probe_logstd.b64` is a b64 binary (never git-add, delete-in-place from live only).

**Contradictions (HIGH)** — the 2 live evals (`rl/inc8_sigmap0_eval.py`, `rl/inc8_sigmap0_torch_eval.py`) default 0.08 (covered, must-fix #2); `project_phase2…:1507` ("0.05 m bar / per-fix lateral ≤ ~0.08") + `:1353` ("drive KF to <0.05 m … THE binding VQ2 risk") + `index_vision_estimator.md:285-287` + `fly_rl.py:1201-1207` vs `MEMORY.md`/`index_rl_training.md:14,26,576` (code default is NOW inc7 — the inc4-footgun memory warning is STALE/over-warning; mark RESOLVED, don't drop the lesson). *Actions per item → `appendix_gap.md`.*

**Stale (HIGH)** — `project_phase2…:1481,1484,1487` (case-C absolute verdict, judged vs the retracted 0.155/0.05 — add a dated supersession rider, architecture stands but pass/fail labels flip); `inc8-deterministic-retrain REPORT.md:101,105-106,113-116,125,134-135,144-150` (banner, preserve std-cap lever items 1&5). 

**Prune (HIGH) — artifact-pipe enforcement, RE-HOME before un-tracking:**
- `shadowpc-refit-dataset-2026-06-12/debug_obs_17runs.zip` (9.0 MB, largest handoff file) — re-home to a `burn-artifacts-*` release asset, md pointer, add handoff `*.zip` to `.gitignore`, `git rm --cached`. *info_loss:* MED (raw 17-run jsonl only in this zip) — re-home FIRST.
- `inc8-intraining-coursecheck-2026-06-17/tb/*.tfevents` (4 files, ~4.83 MB) — re-home, gitignore `*.tfevents`, `git rm --cached`. *info_loss:* MED — re-home first.
- `vq2-blender-render-2026-06-15/preview/` (28 PNG) + `samples/` (13 PNG) (~10.2 MB) — re-home, gitignore handoff PNGs, `git rm --cached`. *info_loss:* LOW-MED (regenerable).
- `shadowpc-followups-2026-06-05/task2_frames/` (40 PNG) + `scratch/` (8 PNG) (~13.7 MB) — re-home (keep the 3 md narratives + the fixture files used by tests). *info_loss:* LOW.

*Note:* a single `.gitignore` handoff-artifacts block (`*.zip`, `*.tfevents`, `handoff/**/*.png`, `*.b64`) + re-home-then-`git rm --cached` fixes the whole ~38 MB class at root. *Med/low → `appendix_gap.md`.*

---

## Preservation ledger

**Prime directive:** this is a BANK-DOWN-FIRST, banner-correct, archive-never-delete cleanup. No finding deletes irreplaceable content. Order is mandatory: **(1) bank the correction down-layer → (2) propagate banners + annotate frozen REPORTs → (3) fix the 5 live code instruments → (4) THEN compress `MEMORY.md:8` and split `project_phase2`.** The newest facts (corrected bar, det reach 0.467, std-cap lever) currently live ONLY in `MEMORY.md:8` + `handoff/inc8-deterministic-retrain-2026-06-19/REPORT.md` — both stay intact until banked.

**Every prune/merge in this plan carries a captured-elsewhere citation:**
- `MEMORY.md:8` megaline collapse — captured in `project_rl_increment_history.md §inc8` + indices AFTER step 1 (NOT before).
- `project_phase2` monolith split — duplicated in `project_rl_increment_history` / `index_vision_estimator` / `index_rl_training` / `index_control_sim` + `handoff/*/WRITEUP.md`; verify both paths exist before each collapse.
- All 14 stale handoff REPORTs — banner-correct, NOT delete; every measured number/root-cause is banked-or-merged.
- `charming-jemison` branch prune — methodology echoed in main memory; archive raw sweep with SUPERSEDED header if kept.
- Tracked binaries (zip/tfevents/png/b64) — re-home to `burn-artifacts-*` release asset BEFORE `git rm --cached`.

**Items with NON-TRIVIAL (MED/HIGH) info-loss risk — treat with extra care:**
1. **HIGH — `MEMORY.md:8` collapse:** trimming before banking loses the only copy of the newest facts. Bank FIRST.
2. **HIGH — live-only `handoff/inc8-reach-rate-2026-06-19/`:** untracked post-correction follow-up; commit from the live tree before consolidation.
3. **HIGH — would-be orphan confusion:** live `inc8-deterministic-retrain` is the active checkout, not an orphan — exclude from prune sweeps.
4. **MED — `#74` parked item:** SPLIT (closed code→DEAD, emul-floor-mirror trigger→T), do NOT collapse.
5. **MED — `debug_obs_17runs.zip` / `*.tfevents`:** raw datasets/curves only in these binaries — re-home before un-tracking.
6. **MED — `sigmap0-adroit` / `coast-drift` / `body-contact` REPORTs:** preserve instrument-validation, sigma_v-non-binding, cold@1.4°-bias phantom-accel mechanism, geometry ceiling.
7. **CARRY-FORWARD to verify (not info-loss, but don't drop):** the "5th harness bug" fix to `rl/inc8_sigmap0_torch.sbatch` was reported uncommitted — confirm it landed.

---

## Appendix index

| Appendix file | bucket | contradiction | stale | duplication | gap | prune | presentability | total |
|---|---|---|---|---|---|---|---|---|
| `handoff/cleanup-audit-2026-06-19/appendix_memory.md` | memory | 12 | 58 | 22 | 6 | 4 | 15 | 117 |
| `handoff/cleanup-audit-2026-06-19/appendix_handoff.md` | handoff | 13 | 30 | 44 | 1 | 11 | 0 | 99 |
| `handoff/cleanup-audit-2026-06-19/appendix_commander.md` | commander | 1 | 2 | 1 | 0 | 0 | 2 | 7 |
| `handoff/cleanup-audit-2026-06-19/appendix_code.md` | code | 3 | 14 | 4 | 4 | 6 | 1 | 32 |
| `handoff/cleanup-audit-2026-06-19/appendix_git.md` | git | 3 | 6 | 0 | 4 | 12 | 1 | 26 |
| `handoff/cleanup-audit-2026-06-19/appendix_contradiction.md` | contradiction | 5 | 24 | 6 | 1 | 0 | 0 | 36 |
| `handoff/cleanup-audit-2026-06-19/appendix_gap.md` | gap | 8 | 20 | 6 | 7 | 15 | 2 | 58 |
| **TOTAL** | | **45** | **154** | **83** | **23** | **48** | **21** | **375** |
