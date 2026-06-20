# Cleanup Audit Appendix — Area: Memory (SSOT)

**Bucket:** `memory`
**Date:** 2026-06-19 (audit) / written 2026-06-20
**Scope:** The full `memory/` SSOT corpus under `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr/memory/` — MEMORY.md (root index), the four domain sub-indices (index_rl_training, index_vision_estimator, index_control_sim, index_strategy_meta), the project topic files (project_phase2_rl_vision_decisions, project_rl_increment_history, project_master_plan, project_parked_backlog, project-fullstack-burn, project_ai_grand_prix, project_ctbr_control_sysid, project_detector_training_pipeline, project_estimator_robustness, project_hardware_constraint, project_red_team_pass_2/3, project_tooling_recommendations_eval), the reference files (reference_sim_ops, reference_sim_interface, reference_adroit_princeton, reference_competition_materials, reference_prior_art), the feedback files, and README.md.

## Area state (summary)

The memory layer is **substantively current in MEMORY.md but badly out-of-sync below the root**. The single most important recent event — the **2026-06-19 gate-4 σ_p0 "0.08 m double-count" correction (commit 511e85c)** plus the **deterministic-retrain result (42ddb0b, det reach 0.467, RL-stays-the-tool, near-field-estimator pivot RETRACTED)** — landed **only in MEMORY.md** and was **never propagated down** into the sub-indices, topic files, or parked backlog. As a result the corpus contains a dense cluster of high-severity contradictions/stale verdicts where the down-layer files still teach the retracted 0.08 m bar, the "near-field gate estimator = THE centering lever" recommendation, the "CANNOT-SETTLE-OFFLINE / NO-GO" verdict, and the false "36-dim asymmetric privileged critic" claim. Second axis: **stale status/dates** — `main` is actually pushed (== origin/main c5d60a0) yet MEMORY says LOCAL-ONLY-do-not-push; the NOW header and "VQ2 wire ~tomorrow" date have elapsed; several "NOT yet on main / pending" merge-status lines describe work that is in fact merged. Third axis: **bloat/routing-violations** — MEMORY.md:8 is a 4.5 KB megaline (18% of an over-cap file), and project_phase2_rl_vision_decisions.md (185 KB) is an un-pruned monolith duplicating the lineage/vision content of the canonical files. The prime fix sequence is: **bank the 2026-06-19 detail DOWN first → then propagate correction banners → then compress MEMORY.md:8 and the monolith.** Wiki-pointer hyphen/underscore "broken links" are confirmed FALSE POSITIVES (consistent repo-wide normalization; all real targets resolve; no orphan files).

---

# CONTRADICTIONS (high → low)

### C-1 [HIGH] Gate-4 σ_p0 ≲ 0.08 m "closure bar" still asserted as the live GO criterion across the down-layer files (the KNOWN 2026-06-19 correction not propagated)
**Locations (all carry the retracted bar):**
- `index_rl_training.md:119` — "READ S1: band_az LOW (sign ok) → estim_err-near-gate → σ_p0 ≲0.08 (the GO, NOT fix_rate)"
- `index_vision_estimator.md:285` — "CLOSURE BOUNDARY (gate-difficulty sweep): r=0.30 (B=0.235) CLOSES IFF σ_p0_lat ≲ **0.08 m**; r=0.38 (B=0.155) ≲ 0.05; tight-VQ2 (B=0.12) ≲ 0.04." (canonical home: §COAST-DRIFT RESOLVED, block 282-287)
- `project-fullstack-burn.md:17` — "GO = bias-inclusive p99 within the gate margin AND std ≲ 0.08 (inc7 had a −0.215 m gate-4 lateral bias …)"
- `project_phase2_rl_vision_decisions.md:1484,1487,1491,1507,1591` — "≈11× the 0.05 m bar" / "NO cell clears 0.05 m" / "only per-fix σ ≤ ~0.03 m clears the 0.05 m bar" / "Does NOT clear the strict 0.05 m bar (that needs per-fix lateral ≤ ~0.08 m)" / "Per-fix 0.05 m → 0.05 m bar holds only to ~26 m/s"
- `project_rl_increment_history.md:734,736,743,745,746,748,749,750` — e.g. ":749 σ_p0 … 0.177 m … → NO-GO vs 0.08"; ":750 Best-available FAITHFUL σ_p0 = yaw-only seed0 0.177 m … → NO-GO vs 0.08, pessimistic"

**issue_type:** contradiction · **severity:** high
**Evidence (corrected truth, MEMORY.md:8):** "GATE-4 BAR CORRECTED (2026-06-19 …): the σ_p0 ≲ 0.08 'closure bar' was a DOUBLE-COUNT — … margin_envelope.py subtracted the drone TWICE (W_EFF=0.75−0.215 chassis, then MARGIN=W_EFF−r=0.535−0.30=0.235; 0.08=0.235/3). The SIM is CORRECT … REAL bar ≈ σ_p0_lat ≲ 0.15 (p99 ≲ 0.45). Measured σ_p0 0.15–0.20 = MARGINAL-PASSING, NOT the NO-GO we mis-called." The correction landed only in MEMORY.md (511e85c); every down-layer file still presents the 0.08/0.05 bar as the current GO/closure criterion.
**recommended_action:** Propagate one correction banner + inline supersession marker into each canonical home: replace/annotate "σ_p0 ≲ 0.08 (the GO)" with the corrected bar "σ_p0_lat ≲ 0.15 / p99 ≲ 0.45 (the 0.08/0.05 bar was a margin_envelope.py double-count, RETRACTED 2026-06-19, commit 511e85c)"; in project_phase2 add the supersession header at the head of §ESTIMATOR-RACESPEED (before L1479); in project_rl_increment_history add "BAR CORRECTED 2026-06-19" to the affected ladder bullets; preserve the original derivation text below the marker (it documents the double-count mechanism). Cross-ref handoff/inc8-deterministic-retrain-2026-06-19/REPORT.md.
**info_loss_risk:** none — the corrected bar and the double-count rationale are fully captured in MEMORY.md:8 and the handoff REPORT; preserve the original 0.08-derivation math as a labeled-historical/erratum note (the r-ladder relationship "tighter radius/VQ2 needs tighter σ" and the pointed-vs-raw σ_lat decomposition 0.075–0.11 pointed / 0.13–0.17 raw remain valid — re-label only the absolute thresholds).

### C-2 [HIGH] "Near-field gate estimator = THE centering lever" — the RETRACTED pivot still reads as current truth (multiple files)
**Locations:**
- `index_vision_estimator.md:286-287` — ":286 The centering lever = a **near-field GATE estimator** (lower the 12 m floor — a gate ANCHOR, not odometry) + **per-fix accuracy.**"; ":287 INVEST: ① inc8 per-fix accuracy + band fix-density …; ② near-field gate estimator. NOT ESKF-for-coast …, NOT VIO, NOT speed."
- `project_parked_backlog.md:78` (#70) — "The centering lever = a near-field GATE estimator (lower the 12 m floor — a gate anchor), NOT odometry."
- `project_parked_backlog.md:94` (WORST-CASE, Unifying invariant) — "gate-4's binding term is gate-RELATIVE centering (vision); NO drone-world-localization (IMU/VIO/twin) addresses it unless A1 or A2 relaxes."
- `project_parked_backlog.md:83` (#75) — "SAME blindness as VIO #70: gives drone WORLD-position, BLIND to gate-RELATIVE centering σ_p0 (the binding gate-4 term)."
- `project_phase2_rl_vision_decisions.md:1566,1731` — ":1566 ESKF attitude-bias estimation = BINDING MARGIN LEVER … PRIMARY lever"; ":1731 BINDING FACTOR = VERTICAL BORESIGHT BIAS … + ATTITUDE/ACCEL BIAS … Path to close = boresight calibration + ESKF attitude-bias estimation."
- `project-fullstack-burn.md:56` (Cut list) — "Dual near/far vision + runtime selector (… build the SINGLE near-field candidate as a gated arm + MEASURE the 12m floor first …)"

**issue_type:** contradiction · **severity:** high (index/backlog) / med (phase2/fullstack)
**Evidence (retraction, MEMORY.md:8):** "RL IS the right tool — the 'pivot off RL to a near-field-gate estimator' recommendation is RETRACTED; next lever = RL to lift reach/pass-rate; vision = SUPPORT, do NOT lean on it (the fix_rate=0-at-flyable-gains 'wall' is moot — the policy nearly passes WITHOUT vision fixes)." "VIO / ESKF / velocity / speed ALL ruled out." Gate-4 is a "REACH/PASS-RATE control problem (gate off the racing line), NOT sub-8cm centering."
**recommended_action:** Annotate each location: re-rank the near-field gate estimator from "THE centering lever / INVEST-②" to **vision SUPPORT/insurance** (a parked secondary lever); state the primary lever is now **RL reach/pass-rate** (recenter flew 3/3); in project_parked_backlog #70/#75/line-94 keep the surviving true point (VIO/twin world-localization does NOT address gate-4) and correct only the "centering = THE binding term / near-field estimator = THE lever" clause; in project_phase2 demote the "ESKF/boresight = PRIMARY/BINDING lever" framing to historical/secondary. Cite MEMORY.md:8 as authority. Keep the "measure the 12 m floor first" instinct (still valid, consistent with MEMORY's "12 m PnP floor").
**info_loss_risk:** none — the near-field estimator survives as a banked insurance lever in MEMORY.md:8 ("2nd lever = near-field GATE estimator") and the project-fullstack-burn cut-list; boresight −0.25 (deployed 5764291) and ESKF remain documented accuracy levers; only the PRIORITY/binding-status changes.

### C-3 [HIGH] inc8 critic asserted as 36-dim / asymmetric privileged — confirmed FALSE (it is symmetric, 20-dim obs, under algo=ppo)
**Locations:**
- `index_rl_training.md:47` ("Critic 36-dim."), `:81` ("asymmetric PRIVILEGED critic (critic sees true alpha/beta/gate pose…)"), `:100` ("critic 36")
- `project_phase2_rl_vision_decisions.md:479` ("asymmetric actor-critic: privileged critic sees truth"), `:1624` ("get_state: 33 → **36** dims … appended for critic"), `:1698` ("Critic 33→36")

**issue_type:** contradiction · **severity:** high (index) / med (phase2)
**Evidence (MEMORY.md cross-cutting footgun, CONFIRMED ×2):** "inc8 CRITIC IS SYMMETRIC (obs-input 20), NOT the asymmetric/privileged 36-dim critic the SSOT implied … GuardedPPO extends diffaero's symmetric PPO; GuardedPPO.build passes obs_dim ONLY; algo=ppo ⇒ get_state(33→36)/state_dim are NEVER consumed in training … The 'critic 36 / truth-seeing asymmetric-critic' claims describe the get_state DESIGN, not the active training path. ROOT-CAUSE of the seed collapse." To get the privileged critic = algo=appo + GuardedPPO(AsymmetricPPO) + env state_dim=36 (separate decision). Confirmable on Adroit: `agent.agent.critic.critic.input_dim` (==20 ⇒ symmetric).
**recommended_action:** Edit index_rl_training.md:47/81/100 and project_phase2 L479/1624/1698 to state the critic is SYMMETRIC (obs-input 20) under algo=ppo, and that the 36-dim privileged critic is an UNBUILT algo=appo option (get_state(33→36)/state_dim never consumed in training). This is a cross-file propagation gap — fix all sites. Cross-ref the MEMORY footgun + project_rl_increment_history.md:752.
**info_loss_risk:** none — the 36-dim get_state DESIGN intent (its latent value if built via appo) is preserved (project_rl_increment_history.md:738/752); the correction only removes the false "it is 36-dim/asymmetric today" claim.

### C-4 [HIGH] Master plan: RL framed as "parked / Track B" while it is the COMMITTED VQ2 path (live inc8) — internal + cross-file contradiction
**Locations:** `project_master_plan.md:99` ("RL = Track B, parked."), `:157` ("Track A (model-based) primary; RL parked."), `:170` ("Surrogate sim + deep RL (Track B) — parked; trigger above.") — vs the plan's OWN `:121` ("C1: RL / surrogate-sim … NO LONGER PARKED. COMMITTED as THE VQ2 path (user decision 2026-05-29).")
**issue_type:** contradiction · **severity:** high
**Evidence:** Contradicted by current truth: index_strategy_meta.md:15 "RL = COMMITTED VQ2 path." and MEMORY.md:21 "inc8 = CASE-C DEPLOYABILITY … Reward spine = explicit arm A". RL is the live center of gravity (inc8), not parked.
**recommended_action:** Update :99/:157/:170 to "RL COMMITTED as the VQ2 path (2026-05-29); inc8 is the live workstream — see [[index-rl-training]]", consistent with the plan's own :121. Reframe the :157 SWIFT-ablation reasoning as "why RL was initially parked (later overturned)" rather than deleting it.
**info_loss_risk:** none — the SWIFT-ablation rationale at :157 is kept as historical "why".

### C-5 [HIGH] Alt-balloon §7 still reads as an OPEN BLOCKER while the file's own TL;DR resolves it (VQ1 passed)
**Location:** `project_ctbr_control_sysid.md:231-251` (§7 ⛔ OPEN BLOCKER + "## Next levers for the altitude balloon")
**issue_type:** contradiction · **severity:** high
**Evidence:** Body :231 "⛔ OPEN BLOCKER — altitude BALLOON / sim auto-thrust … NOT solved. This is the one thing blocking a gate-0 thread"; :240 "## Next levers for the altitude balloon (resume here)". Contradicted by the same file's TL;DR :3 "🏁 VQ1 PASSED — this CTBR stack threads the full 6-gate course" and :27-38 "[2026-06-07 UPDATE — supersedes the §7 'altitude balloon' blocker below] The balloon/limit-cycle is OURS … position-harmless … USER DECISION: accept it for VQ1". Also index_control_sim.md:15 "Alt balloon is OURS".
**recommended_action:** Insert inline "[SUPERSEDED 2026-06-07 — see TL;DR / §2026-06-07 UPDATE: balloon re-diagnosed as a delay-driven relay, position-harmless, accepted for VQ1]" at the head of §7 (L231) and at "## Next levers" (L240).
**info_loss_risk:** none — diagnostic content + "next levers" list stay as historical derivation; only a supersession pointer is added.

### C-6 [HIGH] Adroit role scoped as "VQ2 ML training only (gate detector)" — contradicts its now-primary role as the RL training substrate
**Location:** `reference_adroit_princeton.md:3` (frontmatter), `:19-22` (Role section)
**issue_type:** contradiction · **severity:** high
**Evidence:** :3 "used by this project for VQ2 ML training (gate detector)"; :19 "**Role in this project (VQ2 ML training only):**"; :22 lists only detector training. Contradicted by the SAME file's body :29-46 ("Ops learned driving the DiffAero RL bake-off (2026-06-08)", A100/V100, PPO must saturate GPU) AND index_rl_training.md:69-75 (DiffAero/Adroit substrate, 2048-env PPO) AND the entire inc8 RL campaign on Adroit SLURM (index_rl_training.md:104-123).
**recommended_action:** Broaden the frontmatter + "Role" header from "VQ2 ML training only (gate detector)" to "ML training substrate: the YOLO-pose gate detector AND inc8 RL (DiffAero PPO)"; add the RL training workflow at :22.
**info_loss_risk:** none — additive; the detector-pipeline pointer stays.

### C-7 [MED] Autonomy-hardening merge status contradicts the index (NOT merged vs MERGED 6876f44)
**Location:** `reference_sim_ops.md:70` (and §AUTONOMY-READINESS block 49-82)
**issue_type:** contradiction · **severity:** med
**Evidence:** sim_ops.md:70 "DONE: LAPTOP-FLYRL-AUTONOMY-HARDENING … commit 7210c1d; suite 657 green; NOT merged to main (held for ShadowPC live-verify)." + "MERGE GATE: ShadowPC §4 checklist must pass". Contradicted by index_control_sim.md:55 "Autonomy-hardening (DONE; MERGED to main 6876f44; … full suite 687 passed/0 skips; pushed …)" and :62 "§4 LIVE-VERIFY PASS 5/5 (2026-06-13) … CLEARED TO MERGE."
**recommended_action:** Update reference_sim_ops.md §AUTONOMY-READINESS to the merged state (MERGED 6876f44; §4 live-verify 5/5 2026-06-13; submission entry = rl/submit_rl.py); convert the forward-looking MERGE-GATE framing to past-tense/resolved or a one-line pointer to index_control_sim.md §Autonomy-hardening.
**info_loss_risk:** none — the diagnostic detail (R1/R2/D1/D2 root causes, F-A..F-D fixes) is valuable history and stays KEPT; only the merge-status line + merge-gate framing are stale.

### C-8 [MED] §OBS-CONTRACT get_state 33→36 implies an active privileged critic (same symmetric-critic issue, phase2-local)
**Location:** `project_phase2_rl_vision_decisions.md:1623-1624, :1698`
**issue_type:** contradiction · **severity:** med
**Evidence:** ":1624 get_state: 33 → **36** dims (same 3-dim gate-relative ground-truth state appended for critic)"; ":1698 Critic 33→36." Implies a privileged 36-dim asymmetric critic is the active training path — contradicted by the MEMORY footgun (see C-3).
**recommended_action:** Annotate L1624/1698: "get_state 33→36 is the DESIGN; under algo=ppo it is NEVER consumed — the active critic is SYMMETRIC (20-dim obs). The privileged critic requires algo=appo + state_dim=36 (separate decision)." (The obs-dim=20 freeze itself is correct; only the "appended for critic" active framing is wrong.) Covered jointly by C-3.
**info_loss_risk:** none — design intent preserved; annotation only flags it dormant.

### C-9 [MED] Master plan: "vision-in-loop keeps the run legitimate/valid" refuted twice in-file but the original claim still stands in two other places
**Location:** `project_master_plan.md:86` and `:152` — vs the corrections at `:18` and `:181`
**issue_type:** contradiction · **severity:** med
**Evidence:** :18 "CORRECTION: 'vision-in-loop keeps the run legitimate' is UNFOUNDED (§7 only bans human interaction) … don't treat it as a compliance constraint"; :181 "'vision-in-loop keeps run legitimate' → unfounded (not a rule)". Yet :86 "Vision stays in the loop during timed runs (re-anchors position; keeps the run legitimate)" and :152 "Pre-planned trajectory replay is legal (autonomous; vision-in-loop keeps the run valid)" retain the refuted clause.
**recommended_action:** Edit :86 and :152 to drop the "keeps the run legitimate/valid" clause, keeping the true parts (vision re-anchors position; replay is legal because autonomous). The resolution already exists at :18/:181.
**info_loss_risk:** none — substantive facts retained; the refutation is logged at :18/:181.

### C-10 [LOW] CTBR ff_gain value (2.5 vs deployed 2.6)
**Location:** `index_control_sim.md:11` ("… / ff_gain=2.5")
**issue_type:** contradiction · **severity:** low
**Evidence:** Contradicts project_ctbr_control_sysid.md:148 "Undo it with ff_gain ~ 2.6" and :180 deployed fly command "--ff-gain 2.6". Corpus tally: 2.5 appears ONLY at index_control_sim.md:11; 2.6 appears twice in the topic file.
**recommended_action:** Change ff_gain=2.5 → 2.6 at index_control_sim.md:11 to match the measured/deployed value.
**info_loss_risk:** none — 2.6 is the deployed value; 2.5 has no supporting source.

### C-11 [LOW] Master plan: Python 3.12 vs 3.13 internal inconsistency
**Location:** `project_master_plan.md:139` ("Python 3.12 ML stack") — vs `:22` ("use 3.13 (DONE)") and `:69` (".venv rebuilt on Python 3.13.2 … requires-python >=3.13,<3.14")
**issue_type:** contradiction · **severity:** low
**Evidence:** :139 P0 build-sequence still says "Python 3.12 ML stack"; current canonical is 3.13 (index_strategy_meta.md:49). The 3.12→3.13 rationale (MS-Store-build footgun) is at :22.
**recommended_action:** Change :139 "Python 3.12 ML stack" → "Python 3.13 ML stack (DONE — see Codebase state)".
**info_loss_risk:** none — the 3.12→3.13 rationale preserved at :22.

### C-12 [LOW] Spike −L break magnitude (16 m) vs canonical obs-sign magnitude (~24 m) — unreconciled
**Location:** `project-fullstack-burn.md:22` (carry-forward (5))
**issue_type:** contradiction · **severity:** low
**Evidence:** ":22 +L holds end-to-end (−L control breaks 16 m)." Canonical: MEMORY.md footgun "−L control breaks 24 m"; tests/test_obs_sign_faithfulness.py:45 "the -L negative control must miss by at least this (observed ~24 m)". The 16 m may be the spike's deploy z-fix path (a different test), not the full obs-sign flip — not certainly an error, just unreconciled.
**recommended_action:** Clarify at :22 whether 16 m is the deploy z-fix-path break (distinct from the ~24 m full obs-sign flip in test_obs_sign_faithfulness.py); if the same end-to-end pos_g flip → reconcile to ~24 m; if distinct z-only path → label it as such. Do NOT overwrite 16 m blindly.
**info_loss_risk:** none if labeled — confirm which path it measures first. Canonical 24 m preserved in MEMORY.md + pinned test.

---

# STALE (high → low)

### S-1 [HIGH] `main` "LOCAL-ONLY, far ahead of origin — do NOT push / rewrite history" is false (main IS pushed; HEAD == origin/main c5d60a0)
**Location:** `MEMORY.md:4`
**issue_type:** stale · **severity:** high
**Evidence:** MEMORY.md:4 "🚩 **`main` LOCAL-ONLY, far ahead of origin — do NOT push / rewrite history** (ShadowPC main may have diverged → reconcile via origin deliberately)." VERIFIED FALSE: `git rev-parse HEAD` == `git rev-parse origin/main` == c5d60a09c8138a6ce75d7d041b4fa8790e6856f7 (confirmed live this audit). main is pushed; origin/main is at the audited commit.
**recommended_action:** Replace the "LOCAL-ONLY, far ahead of origin — do NOT push" clause with current truth, e.g. "main PUSHED & == origin/main (c5d60a0); ShadowPC-main reconciliation via origin still deliberate." KEEP the ShadowPC-divergence caution (still live).
**info_loss_risk:** none — the only live sub-fact (ShadowPC main may have diverged, reconcile via origin deliberately) is preserved in the rewrite; the push-prohibition is simply false now.

### S-2 [HIGH] index_rl_training.md frozen at 2026-06-16 — entire 2026-06-17/18/19 inc8 saga missing (S2 confirmed, S5 recenter FLEW 3/3, det reach 0.467)
**Location:** `index_rl_training.md` (inc8 block, lines 99-124; last substantive entry = S0 RUN #1 + GO-to-S1 dispatch at L119, dated 2026-06-16)
**issue_type:** stale (gap-shaped) · **severity:** high
**Evidence:** Index ends at L119 "S1 DISPATCHED: rw_centering=3.0 + g_yaw=-3.0 + 3 seeds." The canonical history (project_rl_increment_history.md:736-750) ran S1/S2/S3/S4/warm-start/S5-recenter through 2026-06-18, and MEMORY.md:8 records the 2026-06-19 deterministic retrain. grep for recenter|0.467|deterministic|RECENTER in the index = ZERO hits. None of "S2 2-AXIS PRIMITIVE CONFIRMED", "S5 RECENTER FLEW (job 3276449, success ~0.50-0.54)", "det reach 0.467" appear in the index.
**recommended_action:** Add a NEW dated top-of-inc8 summary block (3-6 lines): S2 2-axis primitive confirmed; S5 recenter FLEW 3/3 (job 3276449, first flying inc8, success ~0.50-0.54); deterministic retrain det reach 0.467 @ gp1.0; with a [[project-rl-increment-history]] §inc8 pointer for the blow-by-blow. Mirror MEMORY.md:8. Do NOT delete older detail before the pointer exists.
**info_loss_risk:** none — superseding detail already in project_rl_increment_history.md:728-753 and MEMORY.md:8; purely additive.

### S-3 [HIGH] inc8 deterministic-retrain chapter ABSENT from the lineage ledger (the canonical episodic home stops at a stale NO-GO)
**Location:** `project_rl_increment_history.md` (after L750/L752 — end of inc8 saga)
**issue_type:** gap/stale · **severity:** high
**Evidence:** The inc8 saga ends at the 2026-06-18 yaw-injection fix (L750). The deterministic-stability RETRAIN chapter — noise-anneal/std-cap lever MERGED 42ddb0b, FIRST det-flyable 2-axis inc8 (gp1.0 det reach 0.467; rc1 0.000), rc1 root cause (2 actor_logstd channels saturated at exp(2)=7.39 → std-ceiling clamp fix), measured σ_p0 0.15-0.20 / p99 0.37-0.40 re-judged MARGINAL-PASSING, and the RETRACTED near-field-estimator pivot — exists in MEMORY.md:8 + handoff/inc8-deterministic-retrain-2026-06-19/REPORT.md but is ABSENT here. git log: last commit touching this file is dddf5d9 (yaw-injection fix); 42ddb0b/511e85c never updated it. Also note: ladder bullets L734/736/743/745/746/748/749/750 narrate the 0.08 GO bar as live (see C-1).
**recommended_action:** Bank a new dated (~2026-06-19) inc8 closing bullet summarizing: std-cap/noise-anneal lever (42ddb0b, reusable KEEP); first det-flyable 2-axis inc8 (det reach 0.467 @ gp1.0, rc1 0.000); rc1 saturation root-cause + std-ceiling fix; measured σ_p0 0.15-0.20 MARGINAL-PASSING under the corrected bar; RETRACTED "pivot off RL to near-field estimator" (RL stays the right tool; gate-4 = reach/pass-rate). Add "BAR CORRECTED 2026-06-19 — see closing bullet / MEMORY.md:8" to the older 0.08 ladder bullets. Source: MEMORY.md:8 + the handoff REPORT.
**info_loss_risk:** none — this is additive (resolves a gap, not a duplication); the irreplaceable episodic ladder (S0→S5, sign footguns, warm-start, recenter) must NOT be pruned. Transcribe from MEMORY.md:8 + handoff REPORT.

### S-4 [HIGH] index_vision_estimator.md: the 2026-06-19 gate-4 correction entirely absent; the whole §COAST-DRIFT / §margin / §TERMINAL-LOCK verdict cluster reads as current
**Locations:**
- `index_vision_estimator.md:285` — "CLOSURE BOUNDARY … σ_p0_lat ≲ **0.08 m**" (see C-1)
- `index_vision_estimator.md:286-287` — near-field estimator INVEST framing (see C-2)
- `index_vision_estimator.md:264` (§TERMINAL-LOCK REFRAMED) + `:258-259` — "'fix-rate ≥0.50 through the last 6 m / 0.15 s' is GEOMETRICALLY UNACHIEVABLE"; ":258 Gate-4 STILL NO_CLOSE @ fr0.07 → blocker RELOCATED to TERMINAL GATE-LOCK + fix-rate ≥0.50 = the inc8 job"
- `index_vision_estimator.md:13, :183, :294, :324, :337` — "CANNOT-SETTLE-OFFLINE SURVIVES" repeated as the current verdict

**issue_type:** stale · **severity:** high
**Evidence:** git log: index last touched d688923 (2026-06-15), predating the correction. The 0.08 m bar / "GEOMETRICALLY UNACHIEVABLE fix-seating wall" / "CANNOT-SETTLE-OFFLINE SURVIVES" all rest on the retracted 0.08 m bar + case-(b) ε_vert exhausting the budget. Post-correction (MEMORY.md:8): measured σ_p0 0.15-0.20 is MARGINAL-PASSING (sim success 0.65); gate-4 "nearly passes WITHOUT vision fixes"; gate-4 is a REACH/PASS-RATE problem (off the racing line). The terminal-fix-rate ≥0.50 requirement is no longer the binding closure condition.
**recommended_action:** (a) §285: replace "σ_p0_lat ≲ 0.08 m" with "σ_p0_lat ≲ ~0.15 m (p99 ≲ 0.45)" + "[CORRECTED 2026-06-19, 511e85c: the 0.08 m was a margin_envelope.py double-count]". (b) §286-287: per C-2. (c) §TERMINAL-LOCK head: SUPERSEDED note — the ≥0.50-fix-rate "wall" followed from the double-counted bar; under the corrected ~0.15 bar gate-4 is reach/pass-rate-bound, not fix-seating-bound. (d) §COLD-MARGIN head (near L12/13): ONE banner — "RE-JUDGED 2026-06-19: the CANNOT-SETTLE-OFFLINE / margin-OPEN verdict below was computed against the wrong 0.08 m bar; against the corrected ~0.15 bar measured σ_p0 0.15-0.20 is MARGINAL-PASSING (sim success 0.65). Read the cold-margin sections as HISTORICAL analysis." Do NOT blanket-reverse "CANNOT-SETTLE-OFFLINE": its narrow surviving core (at-speed σ at 30 m/s needs a physical drone — MEMORY.md:8 "at-speed σ = a PHYSICAL-DRONE item only") stays true; scope the correction to the closure VERDICT.
**info_loss_risk:** none — the boresight/σ/fix-rate decomposition, the 12 m PnP floor geometry, the informed-coast analysis, and the bias-axis finding (ε_vert ~0.56° nearly exhausts a 0.6° budget) are all still valid and preserved; only the top-line GO/NO-GO is re-judged (correct conclusion in MEMORY.md:8).

### S-5 [HIGH] project_phase2_rl_vision_decisions.md: case-C "absolute NO-GO / CANNOT-SETTLE / DOES-NOT-CLOSE" verdict cluster reads as current — re-judged MARGINAL-PASSING
**Locations:** `:1481` ("Case-C ABSOLUTE world-frame KF nav = NO-GO at race speed, by a wide and speed-flat margin"), `:1485` ("Both decisively over both thresholds"), `:1486` ("SPEED-FLAT: invalid even at 8 m/s"), `:1524` (rel E_bias), `:1562` ("CANNOT-SETTLE-OFFLINE SURVIVES, sharpened"), `:1689`, `:1700`, `:1731` ("Gate-4 0.155 m worst-case margin DOES-NOT-CLOSE … fails at EVERY admissible radius at p90 AND p99")
**issue_type:** stale · **severity:** high
**Evidence:** The whole verdict rests on the 0.155 m budget @ r=0.38 + the 0.05/0.08 bar the 2026-06-19 correction VOIDED. MEMORY.md:8: real clearance 0.37–0.47 m; measured σ_p0 0.15–0.20 = "MARGINAL-PASSING … NOT the NO-GO we mis-called"; binding gate-4 speed ~30 m/s (not 37–55) → fix-rate/bias/latency/terminal-lock conditions ALL relax.
**recommended_action:** Mark the §ESTIMATOR-RACESPEED + §COLD-MARGIN-CLOSURE + §GATE-RELATIVE-BLUEPRINT "NO-GO / CANNOT-SETTLE / DOES-NOT-CLOSE" verdicts as SUPERSEDED by the 2026-06-19 reframe (one supersession header at the head of §ESTIMATOR-RACESPEED, before L1479, per the file's own convention cf. L893/934/1156). Re-anchor on the corrected real clearance 0.37–0.47 m. Best path = collapse the whole block to a pointer into index_vision_estimator.md COLD-MARGIN (which must first carry the corrected reading).
**info_loss_risk:** preserve the durable sub-findings that survive: gate-relative obs drops map bias EXACTLY (rel E_bias −0.000); RewindKF solves latency; the 0.30 m central contact radius vs 0.38 stress knob; the budget identity budget(r)=(0.75−r)−0.215 and r-band table. All mirrored in index_vision_estimator.md + MEMORY footguns — confirm before pruning.

### S-6 [HIGH] project_phase2: Binding-VQ2-risk = ESTIMATOR framing (with <0.05 m KF target) superseded by RL-reach/pass-rate
**Location:** `project_phase2_rl_vision_decisions.md:1334, :1353, :1407, :1409`
**issue_type:** stale · **severity:** high
**Evidence:** ":1334 Binding VQ2 risk = ESTIMATOR … East σ 0.47 m = 3.0× the 0.155 m gate-4 margin"; ":1353 Estimator: drive KF to <0.05 m 1-sigma … THE binding VQ2 validity risk"; ":1407/1409 BINDING VQ2 RISK = ESTIMATOR … Deployable in-plane ~0.55 m (absolute NO-GO, speed-flat)." Superseded by MEMORY.md:8 (RL is the right tool; vision = SUPPORT).
**recommended_action:** Re-cast the "<0.05 m 1-sigma" KF target (L1353) and "binding risk = ESTIMATOR" (L1334/1407) as superseded — the binding lever is now RL reach/pass-rate. Route to index_vision_estimator.md + index_rl_training.md (which already reflect ~30 m/s + the retraction). Drop the "<0.05 m" KF target.
**info_loss_risk:** preserve the still-valid quantified vision-fix numbers (East σ 0.47 m unfiltered, gate-relative 0.11–0.21 m) as historical measurements in index_vision_estimator.md; only the NO-GO INTERPRETATION is stale.

### S-7 [HIGH] Master plan is no longer the operative SSOT (frozen 2026-05-31; ~3 weeks + a phase of drift)
**Location:** `project_master_plan.md` (whole file; :2 description, :10-11 "How to apply"; git log: last substantive edit 21b221f 2026-05-31)
**issue_type:** stale · **severity:** high
**Evidence:** :2 "Living architecture + execution plan … current best state"; :10 "the single source of truth … update IN PLACE." But the entire inc1-inc8 RL lineage, case-C self-localization pivot, gate-4 σ_p0 saga, C2 estimator chain (05ed750), VQ2 detector ensemble (282abb9), and the 2026-06-19 correction are ALL absent.
**recommended_action:** Demote in place: change :2/:10 from "the single source of truth" to "the FROZEN Phase-1/pre-RL architecture baseline (current as of 2026-05-31); LIVE truth now lives in MEMORY.md + the domain indexes." Add a dated top banner pointing to [[index-rl-training]], [[index-vision-estimator]], [[index-control-sim]] for anything post-2026-05-31. Do NOT delete the architecture/scaffold content. (See also the presentability finding P-3 — same fix.)
**info_loss_risk:** none — demotion + banner, not removal.

### S-8 [HIGH] Position telemetry / case-C self-localization doctrine absent from the master plan (treated as an open hole; it is resolved doctrine)
**Location:** `project_master_plan.md:72, :108, :126, :25(c)`
**issue_type:** gap · **severity:** high
**Evidence:** :72 "position NOT provided (§3.3, no GPS) ⇒ derive both; vision is the sole position sensor"; :25(c) "if a position msg exists the problem collapses" — treat self-loc as an open hole. Resolved into doctrine: MEMORY.md:18 "Official VQ1 STREAMS LOCAL_POSITION_NED + ODOMETRY → … case-A; DEPLOYED stack must SELF-LOCALIZE"; MEMORY.md:7 "Case-C absolute world-frame nav = NO-GO at any speed; FIX = gate-relative obs". The case-A/B/C framing + gate-relative-obs decision is missing.
**recommended_action:** Add a pointer under the Sim-interface section: "POSITION DOCTRINE RESOLVED — deployed stack must self-localize (no pose on scored wire); gate-relative obs is the binding fix (absolute world-frame nav = NO-GO). See [[index-vision-estimator]] + MEMORY.md."
**info_loss_risk:** none — full doctrine banked in MEMORY.md + index_vision_estimator.md.

### S-9 [HIGH] project_detector_training_pipeline.md terminal verdict = "SHIP v2" — superseded by the VQ2 ensemble
**Location:** `project_detector_training_pipeline.md:93-109` (## Status / VERDICT 2026-06-02 — SHIP v2)
**issue_type:** stale · **severity:** high
**Evidence:** "Status / VERDICT (2026-06-02) — SHIP v2 … v2 gate_yolo11s_curriculum_v2.pt WINS … NEXT: wire v2 into racer/navigator.py". MEMORY.md:15: "VQ2 DETECTOR = CLEAN ENSEMBLE (282abb9, 2026-06-19; supersedes Round-1): beats champion … course 76→82% … 2-model ensemble, ~26 ms … Deploy STEP-1 DONE (06876bc): opt-in EnsembleGateDetector". File last commit 383a06b (2026-06-11), predating the ensemble campaign (#30/#58, d64e1dc/282abb9).
**recommended_action:** Add a dated supersession banner at the TOP: "[SUPERSEDED for DEPLOY] v2 was the VQ1-era champion; the live VQ2 detector is the 8-kpt photoreal-trained 2-model ENSEMBLE (282abb9, opt-in EnsembleGateDetector 06876bc) — see [[index-vision-estimator]] §VQ2 DETECTOR / [[project-parked-backlog]] #30/#58." Keep the v2 training-saga body as history.
**info_loss_risk:** none — v2 methodology (curriculum, augmentation, multi-gate, weighted-PnP) preserved as useful history; current ensemble detail is in MEMORY.md:15 and #30/#58.

### S-10 [HIGH] Hardware: sim host identity stale (TensorDock/Azure A10 vs the canonical ShadowPC)
**Location:** `project_hardware_constraint.md:31-38` (Windows-GPU-host decision) + title row `:15`
**issue_type:** stale · **severity:** high
**Evidence:** :32 "✅ PROVISIONED 2026-05-29 — TensorDock RTX 3090 is the sim host … This is now the primary sim host; Azure … stays a durable headless fallback."; the machine table :15 lists "Azure NV12ads A10 v5 (cloud) | DCL simulator host". Canonical sim host = ShadowPC: index_strategy_meta.md:63 "ShadowPC (sim)"; MEMORY.md:4 treats ShadowPC as the live sim/eval host; reference_sim_interface calls ShadowPC "the OFFICIAL VQ1 eval sim".
**recommended_action:** Add a dated banner: "[UPDATED 2026-06-1x] Sim host is now ShadowPC (persistent Windows PC) — TensorDock RTX 3090 and the Azure A10/Free-Trial plan are SUPERSEDED for the sim-host role and kept as cost/provisioning history + headless fallback." Mark the Azure machine-table row + TensorDock bullet historical.
**info_loss_risk:** none — TensorDock/Azure provisioning lessons (N-edition cv2 DLL gotcha, quota-ticket path, VM SKU comparison) remain useful fallback knowledge; only the "X is the sim host" designation needs correcting.

### S-11 [HIGH] reference_sim_ops.md §AUTONOMY-READINESS merge status stale (see also C-7) + simops-mastery tooling "NOT yet on main"
**Locations:** `reference_sim_ops.md:70` (autonomy-hardening "NOT merged"); `reference_sim_ops.md:29` (simops-mastery tooling "NOT yet on main (pending commit+push)")
**issue_type:** stale · **severity:** high (C-7 covers the autonomy line; this entry adds the :29 tooling line)
**Evidence:** :29 "🚩 Tooling in handoff/simops-mastery-2026-06-13/ on ShadowPC — NOT yet on main (pending commit+push from ShadowPC)." VERIFIED FALSE: git log shows commit 1b54cb8 "simops-mastery: 10/10 clean autonomous sim-ops cycles …" on main and handoff/simops-mastery-2026-06-13/partA_cycle.py IS tracked. (Mirrors index_control_sim.md:47/:82, finding S-14.)
**recommended_action:** Update :29 to "Tooling MERGED to main (commit 1b54cb8) at handoff/simops-mastery-2026-06-13/ (partA_cycle.py et al.)." Remove the "NOT yet on main (pending commit+push)" clause. (Fix :70 per C-7.)
**info_loss_risk:** none — the orchestrator-pattern content is preserved; only on-main status corrected.

### S-12 [HIGH] project_parked_backlog.md A1 date elapsed + 6 DONE items still in the open FOLD-NOW section (legend-membership broken)
**Locations:** `project_parked_backlog.md:13 (#3), :18 (#15), :19 (#16), :24 (#37), :26 (#40), :29 (#62)` (DONE items in FOLD-NOW); `:82 (#74)` (DONE item in DORMANT-T); `:30/#58 vs :69/#58` (duplicate); `:52 (#32)/:53 (#33)` (fired triggers in DORMANT); `:91 (A1)` (date elapsed); `:68 (#57)` (generic trigger fired)
**issue_type:** stale · **severity:** high (the cluster) — itemized:
**Evidence:**
- DONE-in-FOLD-NOW: e.g. L13 "#3 ✅ DONE (P1, a1620b3, 2026-06-14) … (Closed.)"; L24 "#37 ✅ RESOLVED (system-id, 2026-06-18, bc182f9) … (Closed.)"; L29 "#62 ✅ RESOLVED (2026-06-14) … (Closed.)". The file's own rule (L109): "when an item is actioned, move it to DEAD with the closing commit/branch." Not moved.
- #74 (L82, DORMANT-T) self-labeled "✅ DONE (worker/kf-pfloor-2026-06-15 @ fe4b152, merged; 826 green)" — but BUNDLES an open dormant carry-forward (mirror the floor into the emul IF a future config produces a denser-than-G3 fix stream).
- #58 duplicate: L30 "#30 / #58 🟢 ACTIVATED (2026-06-14)" (delivered L31-35) vs L69 standalone DORMANT-T "#58 full VQ2 photoreal detector campaign …". Both gating conditions of the L69 trigger fired (pilot #30 landed d437507/d64e1dc; COWORK-1 confirmed appearance gap L8); L35 records the deliberate DEFER.
- #32/#33 (L52-53) trigger "FIRES once photoreal dataset A lands" — dataset A LANDED (L31 d437507, L33 d64e1dc) and the recipe session ran (L34: Round-1 transfer CONFIRMED 39/40; 8-kpt permanent; stay YOLO11s) — yet they remain in DORMANT-T as if un-fired.
- A1 (L91): "honest VQ2 truth expected ~2026-06-18 (TOMORROW) → run the A1/A2/A3 checklist THEN." + L90 "(~2026-06-29 …)". Today 2026-06-20 → "~2026-06-18 (TOMORROW)" is past; no repo evidence the wire dropped.
- #57 (L68): "Adroit non-lapse config-matrix V100 parity gate — [next Adroit contact]" — extensive Adroit contact occurred (inc8 GPU smoke + ladder + σ_p0 evals + handoff/sigmap0-adroit-2026-06-19), so the generic trigger literally fired, with no status note.
**recommended_action:** One consolidation pass: (1) relocate the 7 closed items (#3/#15/#16/#37/#40/#62 + the closed-code half of #74) to DEAD with their inline closing commits; (2) SPLIT #74 — closed code → DEAD, the "mirror floor into emul if denser fix stream" trigger → keep as a T entry; (3) collapse #58 to ONE entry (update L69 to "campaign DEFERRED pending VQ2 arena ~2026-06-29 per L35", or fold into the #30/#58 block); (4) re-triage #32/#33 — mark the session actioned (L34/35 results), demote the still-open residue (heteroscedastic per-corner σ / YOLO26 NOT adopted) to "[next detector retrain after the 4-gate re-render]"; (5) A1: change "~2026-06-18 (TOMORROW)" to "as of 2026-06-20 the VQ2 honest wire is NOT confirmed dropped (no commit/handoff evidence); checklist stays armed"; (6) #57: add a status line / sharpen to "[next Adroit V100 multi-config run]". Per the prime directive: relocate, do not delete.
**info_loss_risk:** none for relocations (each closed item carries its full closing record inline); MEDIUM for #74 if naively moved (it bundles a closed deliverable with an open carry-forward — split, don't collapse); preserve the per-corner-σ / YOLO26 / 10–50k-scene scale-up intents.

### S-13 [MED] project_phase2 frontmatter / "How to apply" describes a pre-build 2026-06-07 planning doc; file contains executed results through 2026-06-14
**Location:** `project_phase2_rl_vision_decisions.md:1-14`
**issue_type:** stale · **severity:** med
**Evidence:** ":9-10 Captures the strategy decisions from the 2026-06-07 … planning session (post-VQ1, pre-build). NOT yet executed — these are the agreed directions + open bake-offs for the NEXT session to act on." The file now contains inc7 LIVE-CONFIRMED, S17 mixer, frame-audit, estimator-racespeed verdicts. L11-12 "First-session-next: wire a pointer in MEMORY.md … update project-master-plan C1 (it still says Isaac Lab …)" is a long-completed one-time bootstrap task.
**recommended_action:** Update the frontmatter/How-to-apply to "phase-2 execution journal spanning 2026-06-07..2026-06-14 (now superseded by the indices), not a pre-build planning doc." (Combine with the freeze-banner P-4.)
**info_loss_risk:** none (frontmatter is metadata; the Isaac→DiffAero decision is captured in the body L26-52 and in project_tooling_recommendations_eval.md).

### S-14 [MED] index_control_sim.md: simops tooling "NOT yet on main / pending commit+push" (false — commit 1b54cb8 on main)
**Location:** `index_control_sim.md:47` and `:82`
**issue_type:** stale · **severity:** med
**Evidence:** L47 "ShadowPC tooling (partA_cycle.py + recorder) in handoff/simops-mastery-2026-06-13/ on ShadowPC — NOT YET on main (pending commit+push from ShadowPC)."; L82 same. Refuted by git: commit 1b54cb8 on main; handoff/simops-mastery-2026-06-13/partA_cycle.py present.
**recommended_action:** Update both lines to "tooling IS on main (commit 1b54cb8; handoff/simops-mastery-2026-06-13/ tracked)"; drop the "NOT YET on main / pending" clauses.
**info_loss_risk:** none — the location fact preserved.

### S-15 [MED] Test-count sentinel stale (692/692) in the strategy index — canonical is 947
**Location:** `index_strategy_meta.md:49`
**issue_type:** stale · **severity:** med
**Evidence:** ":49 main CANONICAL. 692/692 tests green (P4-C05 f50b9b4 adds 5 new tests). .venv Python 3.13." Current: MEMORY.md:49 "947 tests collected (.venv; prior 723/884/933 stale; +14 spike)"; scripts/green_gate.py DEFAULT_BASELINE=933 (MEMORY notes bump to 947). 692 is several generations stale.
**recommended_action:** Replace "692/692 tests green" with "947 tests collected (green_gate sentinel baseline 933→947; prior 692/723/884/933 stale)" and cite scripts/green_gate.py as the authoritative gate, OR collapse the Git/env block to a pointer to MEMORY.md:49.
**info_loss_risk:** none — current count in MEMORY.md:49.

### S-16 [MED] Banking protocol rev drift (strategy index at rev 4; MEMORY at rev 5 with the /consolidate-memory prohibition + route-detail-down mechanic)
**Location:** `index_strategy_meta.md:34`
**issue_type:** stale · **severity:** med
**Evidence:** ":34 ## Banking protocol (rev 4 — thin-index structure)." MEMORY.md:36 carries rev 5: "… NEVER use the /consolidate-memory skill (Fengyou, repeated 2026-06-15) — maintain the thin index BY HAND … route the bloated/superseded detail DOWN a layer …". The sub-index (the detail home for banking) is a rev behind and omits both the prohibition + the route-down mechanic.
**recommended_action:** Bump :34 to rev 5 and add the rev-5 content from MEMORY.md:36 (the /consolidate-memory prohibition + route-detail-down-a-layer mechanic).
**info_loss_risk:** HIGH if left as-is — the sub-index workers consult for banking omits the standing "never consolidate-memory" rule (MEMORY states it only in condensed form). Preserve by copying the rev-5 directive in.

### S-17 [MED] Organizer open-questions Q(1)/Q(5) listed as open — both resolved by COWORK-1
**Location:** `index_strategy_meta.md:32`
**issue_type:** stale · **severity:** med
**Evidence:** ":32 Q(1)+Q(5)+Q-A+Q-B+Q-C+Q-D still useful but NOT blocking. Q(1) (VQ2 streams pose?); Q(5) (eval-HW GPU?)". Resolved: reference_competition_materials.md:75 "self-localize REQUIRED … Q(1) resolved" and :74 "VQ eval = Windows 11 + … ~8 GB-VRAM GPU"; MEMORY.md:17 COWORK-1.
**recommended_action:** Mark Q(1) and Q(5) RESOLVED with answers (Q(1): no position on wire → self-localize; Q(5): VQ eval Windows 11 + ~8 GB-VRAM GPU); reduce the open set to Q-A/Q-B/Q-C/Q-D; add a pointer to [[reference-competition-materials]] FAQ INTEL/COWORK-1.
**info_loss_risk:** none — questions preserved as resolved with cited answers.

### S-18 [HIGH] index_control_sim.md wire-facts header: "pos+vel GIVEN" stated flat (case-A) without the official-scored-wire case-C caveat
**Location:** `index_control_sim.md:4-8` (esp. :5)
**issue_type:** stale · **severity:** high
**Evidence:** ":4-5 ## Sim interface wire facts (all 5 RESOLVED) — pos+vel GIVEN (LPN 97 Hz + ODOMETRY 75 Hz)." stated flat. Current doctrine: this is OUR ShadowPC practice sim only; the OFFICIAL scored wire streams NO position — reference_sim_interface.md:64 "the OFFICIAL public spec VADR-TS-002 4.3 lists ONLY HEARTBEAT/ATTITUDE/HIGHRES_IMU/TIMESYNC … the official eval does NOT stream position"; MEMORY.md:17 "NO position on the scored wire … SELF-LOCALIZE REQUIRED"; index_strategy_meta.md:10. The control-sim index carries no discrepancy/self-localize flag.
**recommended_action:** Add a one-line flag to the wire-facts header: these are OUR ShadowPC practice-sim facts; the OFFICIAL scored wire has NO position (§4.3) → deployed stack MUST self-localize (case-C); LPN/ODOMETRY tooling is offline-calibration-only. Point to [[reference-sim-interface]] WIRE-SPEC DISCREPANCY.
**info_loss_risk:** none — practice-sim facts stay; the flag prevents treating "position given" (case-A) as the deployment contract.

### S-19 [MED] reference_sim_interface.md §WIRE-SPEC DISCREPANCY frames the wire question as STILL-OPEN provenance — resolved by the 2026-06-17 WIRE-SPEC correction
**Location:** `reference_sim_interface.md:67` (and block 63-67)
**issue_type:** stale · **severity:** med
**Evidence:** ":67 The open question is PROVENANCE, not availability: is our sim the official GRADED build, or the early Elodin/permissive practice rig? … DEFINITIVE TEST (doable NOW): (a) check our sim's provenance/version; (b) is the official package even RELEASED yet". MEMORY.md (WIRE-SPEC CORRECTED 2026-06-17): "ShadowPC = the OFFICIAL VQ1 eval sim … Official VQ1 STREAMS LOCAL_POSITION_NED + ODOMETRY → case-A … the SCORED wire has NO position → DEPLOYED stack must SELF-LOCALIZE; VQ2 honest wire = THE resolver."
**recommended_action:** Add a dated 2026-06-17 update at the top of §WIRE-SPEC DISCREPANCY (ShadowPC VQ1 streams position = a case-A pass; the scored/deployed wire has no position so self-loc is mandatory; VQ2 honest wire is the binding resolver). Keep the original open-question text as superseded history. Do NOT delete the "re-verify the entire wire when the official sim drops" action (:66 — still correct).
**info_loss_risk:** low but real — the file's core caution ("build self-loc regardless = robust superset", "LPN/ODOMETRY tooling is offline-calibration-only") is STILL correct and load-bearing — preserve it; only the "provenance unresolved / ONE test settles it" status needs the update layered on top.

### S-20 [MED] RL substrate recommendation in prior-art still names JAX-MuJoCo/Brax/Isaac/Elodin — superseded by DiffAero
**Location:** `reference_prior_art.md:24` and `:26`
**issue_type:** stale · **severity:** med
**Evidence:** ":24 Library choices locked-ish: … JAX MuJoCo-MJX/Brax or Isaac Lab (RL surrogate training on Adroit)"; ":26 the recommended SPEED-UPGRADE is now RL-on-surrogate … the deterministic Apache-2.0 Elodin surrogate on Adroit." Superseded: project_phase2_rl_vision_decisions.md:26-53 "DiffAero/Crazyflow OVER Isaac … STAGE-0 BAKE-OFF … DiffAero CONFIRMED (PASS)"; index_rl_training.md:69-70 "DiffAero / Adroit substrate". The RL substrate is DiffAero, not JAX-MuJoCo/Brax/Isaac, and not the Elodin surrogate.
**recommended_action:** Update :24 to name DiffAero as the chosen RL substrate (Isaac/JAX-MuJoCo demoted) with a [[project-phase2-rl-vision-decisions]] pointer. Reframe :26's "Elodin surrogate on Adroit" — Elodin is a dev/validation rig (per the table :22), NOT the Adroit RL training substrate; DiffAero is. Add "(SUPERSEDED: DiffAero chosen 2026-06-08)".
**info_loss_risk:** none — the bake-off rationale + Elodin's dev-rig role are in project_phase2 and the prior-art table itself (:22).

### S-21 [MED] project_ai_grand_prix stale-strategy header is itself doubly stale (names master plan as SSOT; says "RL parked")
**Location:** `project_ai_grand_prix.md:12` (and the :43-50 "Strategic angle" it flags)
**issue_type:** stale · **severity:** med
**Evidence:** ":12 ⚠️ STALE STRATEGY SECTIONS (flagged 2026-05-28): [[project-master-plan]] is the single source of truth and SUPERSEDES … the old 'hybrid CasADi-prior + RL-refines' strategy is replaced by Track-A model-based (RL parked …)". (a) names the master plan as "the single source of truth" — itself now stale (S-7); (b) "RL parked" — but RL is now COMMITTED/live (index_strategy_meta.md:15).
**recommended_action:** Update the :12 header to point readers to MEMORY.md + [[index-rl-training]]/[[index-vision-estimator]] as the live SSOT (not the frozen master plan); change "RL parked" to "RL is now the COMMITTED VQ2 path (inc8 live)". Keep the :18-39 competition-fact/technical-envelope lines (still valid).
**info_loss_risk:** none — the competition facts at :18-39 remain.

### S-22 [MED] README test-count expectation grossly stale (125)
**Location:** `README.md:28` ("pytest   # expect 125 passed")
**issue_type:** stale · **severity:** med
**Evidence:** Canonical suite is 947 collected (MEMORY.md:49; green_gate.py baseline 933→947). 125 is grossly stale (README last touched d6c47d4, 2026-05-29).
**recommended_action:** Update README.md:28 to the current expected count (947, or a soft "~900+") or replace the hardcoded number with a reference to scripts/green_gate.py as the authoritative gate. (User-facing — round/maintainable phrasing preferable to a brittle exact count.)
**info_loss_risk:** none.

### S-23 [MED] README open-questions already resolved at first contact (velocity / LOCAL_POSITION_NED)
**Location:** `README.md:67-71` (Open questions to confirm on first connection)
**issue_type:** stale · **severity:** med
**Evidence:** ":68-71 Spec lists 'linear velocities' (4.5) but no velocity-bearing message in the supported table (4.3)… / Exact MAVLink host:port … / Whether LOCAL_POSITION_NED is emitted." Resolved: reference_sim_interface.md:23 (LPN x,y,z+vx,vy,vz AND ODOMETRY streamed), :44 "Position+velocity = GIVEN, pristine ground-truth (LPN 97 Hz + ODOMETRY 75 Hz)", :64-66 (official-vs-our-sim reconciliation).
**recommended_action:** Mark the velocity/LPN questions answered (our sim streams both; official scored wire §4.3 does NOT — pointer to reference_sim_interface.md); resolve or retain the host:port question if still genuinely open. Preserve the official-vs-practice-wire nuance (self-localization required on the scored wire).
**info_loss_risk:** none — resolved answers + the distinction are in reference_sim_interface.md:23,44,64-66 and MEMORY.md.

### S-24 [HIGH] project-fullstack-burn obs[17:20] carry-forward (2) + wiring-blocker (2) read as open — both DONE (dc56ce7)
**Location:** `project-fullstack-burn.md:22` (carry-forward (2)) and `:30` (THREE VERIFIED WIRING BLOCKERS, #2)
**issue_type:** stale · **severity:** high
**Evidence:** ":22 (2) obs[17:20] has NO production builder (estimator_obs stops at 17) → promote deploy_confidence_triple into estimator_obs.py."; ":30 estimator_obs.py:67 explicitly does NOT build obs[17:20] → no inc8 (20-dim) checkpoint can be flown today." BOTH DONE: commit dc56ce7 (ancestor of HEAD) added confidence_triple / confidence_triple_from_sigmas / estimator_obs20 to src/racer/estimator_obs.py (lines 100/129/171); MEMORY.md:6 marks carry-forward (b) "✅ DONE dc56ce7 … estimator_obs20 … parity 2.93e-8".
**recommended_action:** Mark carry-forward (2) and wiring-blocker (2) RESOLVED (commit dc56ce7) with a pointer to MEMORY.md:6 / src/racer/estimator_obs.py:estimator_obs20. Do not delete the historical context (the ÷√2 footgun + SIGMA_REF/TAU_STALE constants stay relevant).
**info_loss_risk:** none — resolution + parity figure banked in MEMORY.md:6; the ÷√2 footgun preserved in MEMORY.md:6 + fullstack carry-forward (1).

### S-25 [LOW] project-fullstack-burn wiring-blocker #3 path stale (rl/detector.py); self-corrected at :20
**Location:** `project-fullstack-burn.md:31`
**issue_type:** stale · **severity:** low
**Evidence:** ":31 blocker #3 against 'detector.py:33 hardcodes N_CORNERS=4'", but ":20 the verified detector path is src/racer/vision/detector.py, NOT rl/detector.py — the old blocker-#3 memory path was stale" and "All 3 wiring blockers bridged" (spike 95dcc93). The blocker list at :31 wasn't updated.
**recommended_action:** Update the wiring-blocker list (:28-31) header to note all three were bridged by the spike (95dcc93) per :20; fix the blocker-#3 path to src/racer/vision/detector.py. Keep the IPPE corner-order verification caution (corner_to_center 180° bug — still load-bearing).
**info_loss_risk:** none — bridge recorded at :20 + MEMORY.md:6; IPPE caution preserved.

### S-26 [LOW] project-fullstack-burn green_gate baseline 933 (spike bumped collected to 947)
**Location:** `project-fullstack-burn.md:13` (and :40)
**issue_type:** stale · **severity:** low
**Evidence:** ":13 a test-count sentinel baseline 933 … MERGED." MEMORY.md:49 "sentinel baseline 933 → bump to 947"; green_gate.py DEFAULT_BASELINE=933 but the spike added +14 (947 collected).
**recommended_action:** Annotate :13 "933" with "(historical; spike bumped collected to 947 — see MEMORY.md test-count line / green_gate DEFAULT_BASELINE)." Low priority (STATUS-snapshot section).
**info_loss_risk:** none — both figures in MEMORY.md:49.

### S-27 [LOW] project-fullstack-burn RL-lane status (warm-start running / not launched) overtaken by deterministic retrain + recenter 3/3
**Location:** `project-fullstack-burn.md:14` and `:64`
**issue_type:** stale · **severity:** low
**Evidence:** ":14 RL lane: warm-start running on Adroit (job 3276071 …)"; ":64 Warm-start build (BUILT + merged 11cb840; … NOT launched)." Since then the deterministic-stability RETRAIN (42ddb0b) produced the first det-flyable 2-axis inc8 (det reach 0.467) and recenter flew 3/3 (job 3276449).
**recommended_action:** Add a one-line "SUPERSEDED 2026-06-18/19" pointer in the RL-lane bullet (:14) to MEMORY.md:8 (recenter 3/3 + det reach 0.467). Date-stamp the in-flight-prompts fit-check (:64) as a 2026-06-17 snapshot.
**info_loss_risk:** none — current RL state in MEMORY.md:8 + handoff REPORT.

### S-28 [MED] project-fullstack-burn CRITICAL-PATH-ZERO ("no inc7/inc8 checkpoint staged") contradicts the later STATUS "Data staging DONE"
**Location:** `project-fullstack-burn.md:34` (CRITICAL-PATH-ZERO) vs `:15` (STATUS: Data staging DONE)
**issue_type:** contradiction (stale-status) · **severity:** med
**Evidence:** ":34 VERIFIED by the critic: the checkout has NO inc7 actor + NO inc8 checkpoint … Before ANY fan-out, stage …" vs the later-dated ":15 Data staging — DONE via the artifact pipe (2026-06-17): inc8-best + inc7 checkpoints LOCAL; detector best.pt … published to GitHub release burn-artifacts-2026-06-17."
**recommended_action:** Add a forward-pointer at the top of the CRITICAL-PATH-ZERO section (:33-34) noting it was RESOLVED 2026-06-17 (see STATUS :15); retain the section as rationale. The 8-kpt best.pt sub-item is still genuinely open (carry-forward 3, :22) — keep that distinction explicit.
**info_loss_risk:** none — resolution recorded at :15; the still-open 8-kpt best.pt gap is independently preserved at :22.

### S-29 [LOW] project-fullstack-burn "MEMORY-corrections surfaced by the survey" subsection now historical (~900 → 947)
**Location:** `project-fullstack-burn.md:67-70`
**issue_type:** stale · **severity:** low
**Evidence:** ":68 Test suite is ~900 (collect-only), not 723 (stale) — sentinel fixed in MEMORY.md." (now 947); ":70 MEMORY.md over the 24.4KB limit — the full line-6 (inc8 saga) compression remains pending the VQ2 re-bank." (the broken-pointer fix at :69 is done).
**recommended_action:** Demote :67-70 to a dated "survey audit log (2026-06-17, actions complete)" note or fold into STATUS; update "~900" lineage to point at 947. Retain the line-6-compression-pending note only if still true (verify against current MEMORY.md size — confirmed 24798 bytes, still over).
**info_loss_risk:** none — the test-count lineage (723→884→933→947) is in MEMORY.md:49.

### S-30 [MED] NOW header date lags (2026-06-18) the content it contains (2026-06-19 work)
**Location:** `MEMORY.md:3`
**issue_type:** stale · **severity:** med
**Evidence:** "## NOW (2026-06-18)" while today is 2026-06-20 and the section already contains 2026-06-19 work (line 8 "GATE-4 BAR CORRECTED (2026-06-19…)", line 15 "CLEAN ENSEMBLE (282abb9, 2026-06-19…)").
**recommended_action:** Bump to "## NOW (2026-06-20)" (or the latest substantive-update date) so the timestamp matches the freshest content.
**info_loss_risk:** none — pure timestamp update.

### S-31 [MED] MEMORY.md:18 VQ2 honest-wire prediction now in the past ("expected ~2026-06-18 (TOMORROW)")
**Location:** `MEMORY.md:18`
**issue_type:** stale · **severity:** med
**Evidence:** ":18 🚩 VQ2 honest wire expected ~2026-06-18 (TOMORROW) = THE resolver … WAIT for it before concluding (Fengyou)." Today is 2026-06-20 → 2 days past. No statement of arrival; project-fullstack-burn.md:45 still says "Hold behind the VQ2 wire (~tomorrow)". (Matches project_parked_backlog A1, S-12.)
**recommended_action:** Update to current status: "VQ2 honest wire was due ~2026-06-18; status as of 2026-06-20 = [arrived/still pending]" — confirm arrival before editing; if still pending, change "(TOMORROW)" to "OVERDUE since 2026-06-18". Keep the resolver semantics + the WAIT directive.
**info_loss_risk:** none — resolver meaning + WAIT directive preserved; only the stale relative date corrected.

### S-32 [LOW] MEMORY.md:6 "2-DAY BURN (~2026-06-18)" window elapsed
**Location:** `MEMORY.md:6`
**issue_type:** stale · **severity:** low
**Evidence:** ":6 🚩 2-DAY BURN (~2026-06-18): CENTER OF GRAVITY = the full stack …" — a 2-day burn anchored at ~2026-06-18 has elapsed by 2026-06-20. The spike result (95dcc93, golden tuple 7/7) it reports remains valid.
**recommended_action:** Drop/update the time-boxed "2-DAY BURN (~2026-06-18)" label; retain the durable spike result (loop-closed 95dcc93, golden tuple 7/7, #37 de-risked, carry-forwards a-d). Consider routing the spike narrative to [[project-fullstack-burn]] and keeping a one-line result here.
**info_loss_risk:** none — the spike outcome + carry-forwards retained (detail already pointed to project-fullstack-burn.md).

### S-33 [LOW] MEMORY.md:13 CRITICAL PATH first step "σ_p0 GO (Adroit, above)" reads pending while line 8 reports it measured
**Location:** `MEMORY.md:13`
**issue_type:** stale · **severity:** low
**Evidence:** ":13 lists 'σ_p0 GO (Adroit, above) → POC flight …' as the forward path", but line 8 reports the σ_p0 measurement DONE (det reach 0.467, MARGINAL-PASSING). The first step reads as still-pending.
**recommended_action:** Update the first step to reflect the σ_p0 single-seed measurement is DONE (MARGINAL-PASSING, seed1/2 pending), so the actionable next step is the POC flight (case-C emulator-fidelity check #37).
**info_loss_risk:** none — the downstream path (POC flight, #37, C6/POC HELD) retained.

### S-34 [MED] Master plan speed/timing strategy silent on the corrected ~8 s / ~30 m/s ceiling (4.6 s TOGT / 37-55 m/s superseded)
**Location:** `project_master_plan.md:159, :163`
**issue_type:** stale · **severity:** med
**Evidence:** :159 frames the speed path around CPC/Foehn TOGT + Bayesian-opt time-allocation, no corrected ceiling. Current truth index_rl_training.md:86-89: "the banked ~4.6–4.7 s / 4.72 s … was the FULL-ATTITUDE/TOGT optimum — RATE-INFEASIBLE … honest UPRIGHT … ≈ 7.9–8.5 s … binding gate-4 speed = ~30 m/s, NOT 37–55."
**recommended_action:** Add a one-line pointer: "SPEED CEILING SUPERSEDED — upright-feasible lap ≈8 s, binding gate-4 ≈30 m/s (not 37-55); the ~4.6s TOGT bound was rate-infeasible inverted fiction. See [[index-rl-training]] §DOCTRINE REVISION." Keep the classical-planner ladder as a valid alternative record.
**info_loss_risk:** none — corrected numbers in index_rl_training.md:86-89; pointer suffices.

### S-35 [MED] Master plan Codebase-state grossly stale (119 tests; self-contradicting NOT-BUILT list)
**Location:** `project_master_plan.md:65-69` (Codebase state 2026-05-29) and `:68` (NOT BUILT list)
**issue_type:** stale · **severity:** med
**Evidence:** :67 "BUILT ✅ (119 tests pass on 3.13 …)"; :68 "NOT BUILT ⬜: gate_pose, detector, state_estimator (ESKF), mapper/orderer, planner, controller, main loop, logger/recorder, replay harness, first-contact toolkit, eval harness …". Current canonical 947 tests; many "NOT BUILT" items now built — the SAME file marks first-contact toolkit "✅ COMPLETE" at :115 and planner/controller "🔨 BUILT" at :110-111.
**recommended_action:** Stamp the Codebase-state section as a 2026-05-29 SNAPSHOT and point to MEMORY.md:49 + the component checklist; reconcile the :68 NOT-BUILT list against the in-document :110-115.
**info_loss_risk:** none — snapshot-stamp + pointer; the 2026-05-29 build record preserved as history.

### S-36 [LOW] Master plan Adroit "A10 / Azure VM" compute topology superseded (Adroit V100 SLURM + ShadowPC)
**Location:** `project_master_plan.md:25(d), :69, :84, :99`
**issue_type:** stale · **severity:** low
**Evidence:** :25(d) "GPU contention sim+detector share the A10 → … Azure VM (Adroit offline-only)"; :69 "Azure VM + Adroit"; :99 "OFFLINE (Adroit/Azure)". Current ops: Adroit SLURM on V100/Tesla (index_rl_training.md "Tesla V100") + ShadowPC as the official VQ1 eval sim (MEMORY.md:18).
**recommended_action:** Add a note: "COMPUTE TOPOLOGY UPDATED — training = Adroit SLURM (V100), official VQ1 eval = ShadowPC; the Azure/A10 framing is unrealized. See [[reference-adroit-princeton]] + MEMORY.md." Do not delete (Azure may still be a physical-qualifier deployment path).
**info_loss_risk:** confirm the Azure-VM-for-real-time-inference idea is captured in project_hardware_constraint.md or project_parked_backlog before pruning; if not, MOVE it there.

### S-37 [LOW] Master plan build-sequence P0-P6 numbering collides with the live Phase-2 P1-P5 path taxonomy
**Location:** `project_master_plan.md:138-145`
**issue_type:** stale · **severity:** low
**Evidence:** :138-145 "P0 (now) … P6 (VQ2 speed): RL racing policy on the Elodin surrogate". The executed structure is the Phase-2 path taxonomy with DIFFERENT meaning per label: MEMORY.md:15 "PHASE-2 PATHS … P1 vision-accuracy … P2 inc8-RL … P5 VQ2 detector". Navigation hazard.
**recommended_action:** Stamp :138-145 as the "original 2026-05-28 plan-of-record (Phase-1)"; note the live execution follows the Phase-2 P1-P5 taxonomy (different meaning) per MEMORY.md/index_rl_training. Keep the original as history.
**info_loss_risk:** none — original sequence kept as history.

### S-38 [LOW] Master plan: Elodin as the training/surrogate rig superseded by DiffAero
**Location:** `project_master_plan.md:20, :27, :45, :61, :121, :145`
**issue_type:** stale · **severity:** low
**Evidence:** :20 "Use ELODIN as a NOW-available surrogate/dev/validation rig"; :121 "Training = a massively-parallel twin (Isaac Lab on Adroit …) → Elodin as the spec-exact-camera eval oracle"; :145 "RL racing policy on the Elodin surrogate". Current RL training is on DiffAero (index_rl_training.md: DiffAero racing.yaml :73, PeregrineRacingInc8 env).
**recommended_action:** Add a note: "TRAINING RIG SUPERSEDED — RL trains on DiffAero on Adroit, not Elodin/Isaac-Lab; Elodin remains a possible spec-camera eval oracle only. See [[index-rl-training]]." Keep the Elodin Apache-2.0/adapter record.
**info_loss_risk:** confirm the Elodin-adapter build + Apache-2.0 license verdict is captured in reference_prior_art or the codebase before pruning; keep as history.

### S-39 [LOW] index_control_sim gate-4 target speed 37 m/s vs current binding 30 m/s
**Location:** `index_control_sim.md:65` and `:68`
**issue_type:** stale · **severity:** low
**Evidence:** :65 "At 37 m/s: 115 ms = ~4.3 m"; :68 "full-lap ~37 m/s gate-4 recording". Current: MEMORY.md:21 "binding gate-4 speed ~30 m/s [not 37-55]"; MEMORY.md:8 "best gate-4 speed = 30 m/s".
**recommended_action:** :65 is a dated measurement — keep as historical. Update the forward-looking queued-item at :68 to "~30 m/s gate-4 recording" (or add "(target now ~30 m/s)").
**info_loss_risk:** none — :65's historical computation intact; only the actionable queued recording corrected.

### S-40 [LOW] index_rl_training S2 gap-decomposition leads with the refuted 4.72 s number
**Location:** `index_rl_training.md:33`
**issue_type:** stale · **severity:** low
**Evidence:** ":33 Gap decomposition: 35.3 s (VQ1) → 9.76 s (inc7) → 6.89 s (cone tax) → ~4.72 s. [SUPERSEDED 2026-06-14: the ~4.72 s is the INVERTED/rate-infeasible TOGT bound; UPRIGHT-feasible ~= 8 s…]". The corrected ~8 s is also at :85-91 (DOCTRINE REVISION). The leading "→ ~4.72 s" is the refuted number presented first.
**recommended_action:** Rewrite :33 to lead with the corrected upright-feasible ~8 s bound, demoting 4.72 s to a parenthetical "(superseded inverted/TOGT fiction — see DOCTRINE REVISION)". Point to :85-91 (canonical).
**info_loss_risk:** none — the honesty-trajectory is preserved at :85-91; presentation-ordering only.

### S-41 [MED] index_rl_training inc8 SELECT criterion cites the pre-correction p90<0.155 budget chain
**Location:** `index_rl_training.md:50, :40, :97`
**issue_type:** stale · **severity:** med
**Evidence:** :50 "SELECT on p90 gate: gate-4 SIMSTART in-plane p90 < 0.155 m @ r=0.38"; :40 "worst-case stress knob = 0.38 m (budget 0.155 m)"; :97 "cold@bias0 p90 0.392 / bias-ON 0.457, both over budget = CANNOT-SETTLE reproduced". These budgets descend from the margin_envelope.py double-count (MEMORY.md:8); the 0.392/0.457 "both over budget" verdict is the very NO-GO re-judged to MARGINAL-PASSING.
**recommended_action:** Annotate :40/:50/:97 "SUPERSEDED 2026-06-19 — see gate-4 bar correction (MEMORY.md / [[index-vision-estimator]]): the 0.155/0.235 budget chain double-counts the drone radius; real clearance 0.37-0.47 m, real bar σ_lat ≲ 0.15." Do not delete the radius-reconciliation reporting discipline at :40 (still valid) — flag the budget numbers only.
**info_loss_risk:** none — corrected clearance/bar in MEMORY.md:8; the r-function reporting discipline unaffected.

### S-42 [LOW] index_rl_training inc7 baseline figures (9.76 s warm vs 11.45 s fresh) compressed — see also presentability
**Location:** `index_rl_training.md:6`
**issue_type:** presentability/stale · **severity:** low
**Evidence:** ":6 t_med=9.76 s (warm) … deployment baseline ~11.45 s (fresh-reset = deployment estimate)." The 9.76 vs 11.45 (warm vs fresh-reset cold-start bimodal artifact) is correct but compressed; a reader could quote 9.76 s as the deployment number. Cold-start artifact explanation at :9 + project_rl_increment_history.md:558.
**recommended_action:** Leave the numbers (both correct) but tighten phrasing so the deployment-relevant figure (~11.45 s fresh) is unambiguous vs warm-lap 9.76 s; optionally cross-ref the cold-start note at :9.
**info_loss_risk:** none — clarity tweak only.

### S-43 [MED] project_phase2 inc8 retrain spec ("SUPERSEDES prior inc8 spec") is itself superseded by the look-at/recenter pivot
**Location:** `project_phase2_rl_vision_decisions.md:1705-1714`
**issue_type:** stale · **severity:** med
**Evidence:** ":1705 ### inc8 retrain spec (Adroit-ready; SUPERSEDES prior inc8 spec)"; ":1709 R5' camera-pointing reward … NEW TOP LEVER … Reward the OUTCOME … let pointing + fix timing EMERGE. Do NOT hand-reward a specific yaw setpoint". Superseded by MEMORY.md:8: "4 reward-WEIGHT iterations ALL NO-GO ⇒ reward ARCHITECTURE not weights … ARCHITECTURE PIVOT (2026-06-16). The LOOK-AT PRIMITIVE is the load-bearing fix (ENGINEER the pointing)" and "RECENTER FLEW 3/3 = FIRST FLYING inc8".
**recommended_action:** Add a forward-pointer at :1705 noting this spec (emergent-pointing R5') is SUPERSEDED by the 2026-06-16 pivot to the engineered LOOK-AT PRIMITIVE + through_centering_reward (RECENTER, MERGED 9cecf64). Canonical = [[project-rl-increment-history]] §inc8 + MEMORY.md NOW. Preserve the spec as the record of the emergent-reward attempt.
**info_loss_risk:** none — the abandoned emergent-pointing approach is itself a load-bearing lesson ("let pointing EMERGE failed → engineer it"); preserve in place with the supersession pointer.

### S-44 [MED] project_phase2 VQ2 data-stream §Open item resolved by COWORK-1
**Location:** `project_phase2_rl_vision_decisions.md:1001-1006` (## Open)
**issue_type:** stale · **severity:** low (org. resolution)
**Evidence:** ":1006 - VQ2 data-stream answer (gates the map/SLAM + vision-load-bearing question)." listed OPEN. Resolved per MEMORY.md COMPETITION INTEL (COWORK-1, 2026-06-14): "NO position on the scored wire (§4.3) → SELF-LOCALIZE REQUIRED → gate-relative pivot VALIDATED" + WIRE-SPEC (2026-06-17).
**recommended_action:** Mark the §Open "VQ2 data-stream answer" RESOLVED with a one-line pointer to MEMORY.md COMPETITION-INTEL/WIRE-SPEC. Leave the substrate item (already struck/RESOLVED) as-is.
**info_loss_risk:** none — resolution captured in MEMORY.md.

### S-45 [LOW] project_phase2 STACK-REVIEW-VQ2 + YOLO26 plan frozen pre-ensemble
**Location:** `project_phase2_rl_vision_decisions.md:490-569, :526-528`
**issue_type:** stale · **severity:** low
**Evidence:** §STACK-REVIEW-VQ2 (:490) + ":526 YOLO26-pose at the photoreal v4 retrain (NOT before)" present the detector as un-upgraded with a forward YOLO26 plan. MEMORY.md:15: "VQ2 DETECTOR = CLEAN ENSEMBLE (282abb9, 2026-06-19) … win = a TRAINING-LOSS lever … 2-model ensemble." (Plan now executed differently.)
**recommended_action:** Annotate §STACK-REVIEW-VQ2 + the YOLO26 line as the 2026-06-10 review baseline, now advanced by the 2026-06-19 ensemble (282abb9). Route current detector status to project_detector_training_pipeline.md / index_vision_estimator.md.
**info_loss_risk:** preserve the "examined and REJECTED" detector ledger (DA3, RT-DETR/D-FINE, PVNet, tightly-coupled VIO, etc., :538-558) — confirm mirrored in project_detector_training_pipeline.md before collapsing.

### S-46 [LOW] project_phase2 SUBSTRATE-AUDIT (P4-C05) operational status may have advanced
**Location:** `project_phase2_rl_vision_decisions.md:1425-1471`
**issue_type:** stale · **severity:** low
**Evidence:** §SUBSTRATE-AUDIT (:1425) lists P4-C05 (obs_from_zup hardcoded gate yaw=pi, "ACTION: GO-BEFORE-VQ2") + a regression-suite "DISPATCH QUEUED". P4-C05 was resolved (index_vision_estimator.md:163 "P4-C05 DONE (f50b9b4, 2026-06-13; suite 687→692 green)"), so the "GO-BEFORE-VQ2 / QUEUED" status is stale-open.
**recommended_action:** Cross-check each SUBSTRATE-AUDIT action item against index_vision_estimator.md (P4-C05 DONE f50b9b4) + the test count (947 now); update or collapse to a pointer to the handoff report + DONE markers.
**info_loss_risk:** preserve the 8-test regression-suite intent + the dormant-VQ2-hazard catalog (non-pi-course obs drift up to 4.22 m); confirm the tests were promoted (suite grew to 947) before marking the queue closed.

### S-47 [MED] project_estimator_robustness per-fix σ value stale (0.35 m / 0.265 m modeled vs ~0.10 m measured)
**Location:** `project_estimator_robustness.md:18` (layer-1 adaptive R)
**issue_type:** stale · **severity:** med
**Evidence:** ":18 R already = PnP pixel-noise cov + attitude lever-arm term (range-dependent: 1° attitude err @20 m ≈ 0.35 m …)". Same family as the modeled σ=0.265 m that SHADOWVISION measured to ~0.10 m lateral (index_vision_estimator.md:17-21) and refined to anisotropic gate-4 [lat 0.19, vert 0.10] (:175).
**recommended_action:** Append a dated note to layer-1 at :18: "[2026-06-13+ UPDATE] modeled per-fix σ has since been MEASURED — pooled lateral ~0.10 m, gate-4 anisotropic [lat 0.19, vert 0.10] (not the ~0.35 m / 0.265 m model); see index_vision_estimator.md §SHADOWVISION / §L3 AT-SPEED." Keep the design reasoning.
**info_loss_risk:** none — the 3-layer scheme is architecturally valid; only the illustrative σ magnitude is superseded (preserved in the index).

### S-48 [LOW] project_estimator_robustness layer status (layer-2 deferred / layer-3 not built) vs the shipped C2 chain
**Location:** `project_estimator_robustness.md:28` (Status) + `:39`, `:41`
**issue_type:** stale · **severity:** low
**Evidence:** ":28 Status: layer 1 BUILT … layer 2 DEFERRED … layer 3 NOT BUILT (needs the mapper)." Since this 2026-05-31 note the C2 chain shipped much of layer 2/3: index_vision_estimator.md:330-337 "C2-ESTIMATOR-CHAIN DONE & MERGED (05ed750) … relinnov χ²(2,.999)=13.82" (layer-2 Mahalanobis gate); gate_mapper.py cases A/B/C complete (:123).
**recommended_action:** Add a dated pointer to the Status line: "[2026-06-14 UPDATE] layer-2 Mahalanobis innovation gate now BUILT (relinnov χ²(2,.999)=13.82) in the C2 chain (05ed750); gate mapper A/B/C complete; RewindKF productionized — see index_vision_estimator.md §C2-ESTIMATOR-CHAIN. This note remains the design rationale."
**info_loss_risk:** none — design rationale preserved; build detail in the index.

### S-49 [MED] index_vision_estimator parent-index detector pointer "SHIP v2" stale (VQ2 ensemble)
**Location:** `index_vision_estimator.md:115` (## VISION-PKG2 specs)
**issue_type:** stale · **severity:** med
**Evidence:** ":115 Detector: SHIP v2 (models/gate_yolo11s_curriculum_v2.pt, multi-gate). v3 --hard = NEGATIVE." Superseded by the VQ2 ensemble (MEMORY.md:15).
**recommended_action:** Update :115 to "Detector: VQ1 champion = v2 (gate_yolo11s_curriculum_v2.pt); VQ2 deploy candidate = 8-kpt photoreal 2-model ENSEMBLE (282abb9, opt-in EnsembleGateDetector 06876bc, course 82% vs 76%). v3 --hard = NEGATIVE (VQ1-era)."
**info_loss_risk:** none — v2/v3 facts retained; adds the current ensemble champion.

### S-50 [MED] index_vision_estimator VISION-PKG2 world-fix σ [0.73,0.47,0.29] / accept ~47% — modeled values, superseded by measured later in the same file
**Location:** `index_vision_estimator.md:116-117`
**issue_type:** stale · **severity:** med
**Evidence:** ":116 World-fix σ≈[0.73,0.47,0.29] m (N,E,D), range-flat to ~24 m; acceptance ~47%, leak 0.53%."; :117 cites the "0.40 m cov floor". Overturned elsewhere: :21 "per-fix lateral σ = 0.10 m (MEASURED), NOT 0.265 m"; :294 "σ=0.10 (measured) makes closure POSSIBLE"; :21/27 measured accept ~7% un-pointed (not 47%).
**recommended_action:** Add a note to the VISION-PKG2 block: "σ/acceptance here are MODELED (2026-06-10); SUPERSEDED by measured per-fix σ_lat 0.10 (gate-4 0.19), σ_vert 0.10, accept ~7% un-pointed / 67–77% pointed — see §SHADOWVISION, §L3 AT-SPEED, §FORM RESOLVED." Keep the modeled table as provenance.
**info_loss_risk:** none — measured values captured in §SHADOWVISION (:17), §L3 (:175), §accept-geometry (:272-273); modeled table retained as the original spec.

### S-51 [MED] index_vision_estimator boresight ε_vert early sections cite 0.215 m / 0.56° as binding; superseded by resolved −0.25 m / −0.61°
**Location:** `index_vision_estimator.md:80, :83, :195, :327` (0.215/0.56°) vs `:239, :253, :258` (−0.25)
**issue_type:** stale · **severity:** med
**Evidence:** Early: :80 "VERTICAL BORESIGHT BIAS ε_vert ≈ 0.215 m (~0.56°…)"; :195 "ε_vert ≈ 0.215 m (= 0.56°…)"; :327 "δ_map pins ε_vert at 0.56°". Resolved later same file: :239 "ε MAGNITUDE PINNED = −0.25 m @ 23 m ≈ −0.61°"; :253 "BoresightCorrection(vert_offset_m = −0.25)"; :258 "frames.BORESIGHT = … DEPLOYED". MEMORY.md confirms −0.25 deployed (5764291).
**recommended_action:** On the first occurrence (:80) append "(provisional; PINNED+RESOLVED to −0.25 m / −0.61° metric @ §BORESIGHT FORM RESOLVED, DEPLOYED frames.BORESIGHT=−0.25 5764291 — use that value)." Chronological-supersession within one file → a forward-pointer suffices.
**info_loss_risk:** none — both values kept (0.215 m = early discriminator estimate, −0.25 m = deployed production constant); forward-pointer prevents mis-citation.

### S-52 [LOW] index_vision_estimator stale dated open-action items presented as live (overtaken)
**Location:** `index_vision_estimator.md:83, :96, :208-210, :216-220, :241`
**issue_type:** stale · **severity:** low
**Evidence:** :83 "Airtight lock test pending (morning ShadowPC static head-on fix at g2/g4)" — DONE/superseded by §P3 HEAD-ON (:238, ε pinned −0.25); :210 marks the g2/g4 framing "[SUPERSEDED 2026-06-14]". :241 "DO NOT lock calibration TYPE until the IPPE both-flip … pass resolves it" — RESOLVED at :250 "BORESIGHT FORM RESOLVED = METRIC." :96 escape-hatch + :220 "AUTHORIZED LIVE ARBITER" overtaken by FORM RESOLVED + deployed bake.
**recommended_action:** Mark each overtaken open-action with its resolution pointer (e.g. :83 → "RESOLVED: gate-0 head-on, ε=−0.25, §P3 HEAD-ON"; :241 → "RESOLVED §BORESIGHT FORM RESOLVED"). Prefer in-place "✅ RESOLVED →" annotations over deletion.
**info_loss_risk:** none — every resolution exists later in the same file.

### S-53 [MED] index_vision_estimator ESKF attitude-bias "CO-EQUAL margin lever" promoted early, DEMOTED later — see also contradiction
**Location:** `index_vision_estimator.md:82, :84, :206, :327` (CO-EQUAL) vs `:226, :234, :247` (DEMOTED)
**issue_type:** contradiction (within-file chronological) · **severity:** med
**Evidence:** :84 "ESKF attitude-bias estimation … = CO-EQUAL MARGIN LEVER (PROMOTED from secondary…)"; :206 "PROMOTED … to co-equal margin lever." Later :226 "DEMOTES the banked 'ESKF attitude-bias estimation = CO-EQUAL margin lever' framing … The boresight is an OFFLINE constant, not an online lever."; :247 "boresight DROPPED as an online state … build the accel-bias estimator ONLY if first-contact … shows a real accel-bias ABOVE budget."
**recommended_action:** Append to :82/:84/:206/:327 a forward-pointer: "(SUPERSEDED — boresight is an OFFLINE constant not an online lever; ESKF accel-bias is DATA-CONDITIONAL/secondary → see §BORESIGHT = OFFLINE EXTRINSIC CORRECTION and §P1 PATHWAY COMPLETE)." Resolved-within-file supersession — flag, don't delete.
**info_loss_risk:** none — the demotion + rationale at :224-248; the early "co-equal" framing preserved as the superseded prior.

### S-54 [LOW] reference_adroit detector status stale ("detector v2 shipped 2026-06-02")
**Location:** `reference_adroit_princeton.md:21`
**issue_type:** stale · **severity:** low
**Evidence:** ":21 Used for: training the YOLO-pose gate detector (detector v2 shipped 2026-06-02)." MEMORY.md NOW: detector advanced to the VQ2 ensemble (282abb9, 2026-06-19).
**recommended_action:** Drop the parenthetical version stamp or update to "detector now at the VQ2 ensemble (282abb9, 2026-06-19) — see [[index-vision-estimator]]". The Adroit-role point needs no frozen version number.
**info_loss_risk:** none — detector lineage lives in index_vision_estimator.md + project_detector_training_pipeline.md.

### S-55 [LOW] reference_competition_materials compute/latency-cap "VERIFY" already resolved in-file
**Location:** `reference_competition_materials.md:57`
**issue_type:** stale · **severity:** low
**Evidence:** ":57 Onboard edge compute ~100 TOPS … VERIFY whether the virtual-qual sim enforces a compute/latency cap." Resolved at :74 (COWORK-1): "VQ eval = Windows 11 + standard PC + discrete ~8 GB-VRAM GPU (§5.1) … LATENCY CLOSES for VQ (15–25 ms ≪ 50 ms)"; ~100 TOPS = the physical drone, not the VQ box (MEMORY.md).
**recommended_action:** Annotate the :57 "VERIFY" as RESOLVED with a pointer to the COWORK-1 block (:74) / [[reference-sim-interface]]: VQ eval = Windows + ~8 GB-VRAM GPU, latency axis closed; ~100 TOPS = physical drone only.
**info_loss_risk:** none — resolution already in the same file (:74) + the indices.

### S-56 [LOW] reference_competition_materials VQ1 submission-interface "not yet published" open items (dated 2026-06-07)
**Location:** `reference_competition_materials.md:34, :38`
**issue_type:** stale · **severity:** low
**Evidence:** ":34 The submission interface is NOT yet published — registered teams receive it … by EMAIL."; ":38 OPEN list: the exact submission interface … whether eval re-runs are unlimited; the precise VQ1 deadline …". Dated 2026-06-07; today 2026-06-20. Cannot confirm resolution from the audited files; parked-backlog (:80 "submission/entrypoint packaging (LOW)") suggests partial movement.
**recommended_action:** Re-verify against the latest organizer email / parked-backlog; if still open, add "still-open as of <date>"; if resolved, mark resolved with a pointer. At minimum date-stamp the OPEN list. (Flag to the commander — not auto-editable from these files alone.)
**info_loss_risk:** none — tracking items; verify-and-restamp rather than remove.

### S-57 [LOW] reference_sim_interface R8 sim-reset (MAV_CMD 31000) "still MURKY" superseded by reference_sim_ops resolved semantics
**Location:** `reference_sim_interface.md:56` vs `reference_sim_ops.md:20-23`
**issue_type:** stale · **severity:** low
**Evidence:** sim_interface.md:56 (first-contact 2026-06-02): "R8 sim-reset (MAV_CMD 31000) still MURKY — did NOT visibly reset race state (started stayed True) …". Refined in sim_ops.md:20 (2026-06-13): "MAV_CMD 31000 restarts race once race context exists — NO-OP from HOME" + the full idle-state state machine (:21-24).
**recommended_action:** Add a one-line pointer at sim_interface.md:56 ("RESOLVED — see [[reference-sim-ops]] §Race restart semantics: 31000 restarts once race context exists, NO-OP from HOME").
**info_loss_risk:** none — resolved semantics in sim_ops.md:20-24; forward pointer only.

### S-58 [LOW] feedback_checkpoint_transfer gitignore mechanism partially superseded by the artifact pipe
**Location:** `feedback_checkpoint_transfer.md:12`
**issue_type:** stale · **severity:** low
**Evidence:** ":12 either (a) unignore the specific checkpoint path in .gitignore … or (b) add a dedicated rl/checkpoints/ directory NOT covered by *.pt." MEMORY.md footgun: "*.pt/*.pth + rl/checkpoints/*.json … gitignored" + ARTIFACT PIPE: "binaries … ship as GitHub release assets (burn-artifacts-* tags), NEVER git-add". The "commit small checkpoints to git" advice is partially superseded by the release-asset pipe for binaries.
**recommended_action:** Reconcile: note that the small-checkpoint-in-git path (e.g. tracked stage1_inc7_actor.pth) still holds for tiny policy .pth, but large binaries now go via the burn-artifacts-* release pipe (MEMORY.md ARTIFACT PIPE / [[project-fullstack-burn]]). Add the pointer so the conventions don't read as conflicting.
**info_loss_risk:** low — the rule (small→git, large→out-of-git) is directionally correct; align the "large→base64/manual-copy" tail with the release-asset pipe.

---

# DUPLICATION (high → low)

### D-1 [HIGH] project_phase2_rl_vision_decisions.md duplicates the full inc-lineage saga (S1.2..inc7, frame-audit, crab-diag, S18, training-doctrine) of the canonical history file
**Location:** `project_phase2_rl_vision_decisions.md:407-466, 579-648, 865-1156, 1205-1256`
**issue_type:** duplication · **severity:** high
**Evidence:** Phase2 carries full writeups of S1.2/S1.3/inc4 (:407-466), inc5 (:579-607), SHADOWPC-LIVE-DEPLOY-DIAG/INC6-DIAG (:612-648, 891-928), S18-THRUST-LAPSE (:932-997), FRAME-AUDIT (:1041-1131), CRAB-DIAG (:1135-1156), TRAINING-DOCTRINE/inc7 (:1205-1256). project_rl_increment_history.md has the canonical, already-compressed versions of EVERY one (headers: S1.2, S1.3/inc3, S15/inc4, Inc5, SHADOWPC-INC6-DIAG, S18 LAPSE, LAPTOP-FRAME-AUDIT, CRAB-DIAG, TRAINING-DOCTRINE + INC7 SPEC, INC7 COMPLETE, INC7 LIVE CONFIRMED).
**recommended_action:** Collapse the phase2 inc-lineage blocks to a single pointer to project_rl_increment_history.md (the designated lineage SSOT). Each retired increment keeps only its handoff/WRITEUP path if not already in the history file.
**info_loss_risk:** low — project_rl_increment_history.md is the explicit canonical lineage home and already contains these (verified via section headers). Spot-check the phase2 versions for any unique number absent from the history file (e.g. the per-segment tilt-concentration table :1015-1023) before collapsing; if unique, MOVE it first.

### D-2 [HIGH] project_phase2 estimator-racespeed + cold-margin + body-radius + gate-relative-blueprint (4 stacked blocks) duplicate index_vision_estimator
**Location:** `project_phase2_rl_vision_decisions.md:1475-1608, 1522-1583, 1529-1569, 1683-1733`
**issue_type:** duplication · **severity:** high
**Evidence:** §ESTIMATOR-RACESPEED (:1475), §COLD-MARGIN-CLOSURE (:1522), §BODY-RADIUS-RECONCILE (:1529), §GATE-RELATIVE-BLUEPRINT (:1683) are full-length reproductions. Canonical: index_vision_estimator.md:4 §ESTIMATOR-RACESPEED VERDICT, :12 §COLD-MARGIN CLOSURE, :51 §Contact-radius reconciliation. The contact-radius central=0.30 / stress=0.38 + the 0.2135 m geometry floor are ALSO in MEMORY.md footguns. (Verbatim overlap e.g. phase2 :1550 "| 0.30 (central) | 0.235 m | +0.080 (~1.5×) |" ≡ index :63; phase2 :1524 "rel E_bias −0.000 … RMS 0.139" vs index :337.)
**recommended_action:** Collapse the four blocks to pointers into index_vision_estimator.md (+ the handoff/ULTRACODE-ESTIMATOR-RACESPEED + body-contact-reconcile + gate-relative-pipeline-design paths). BECAUSE these blocks carry the now-corrected 0.05/0.08 bar + NO-GO verdicts (C-1/S-5), the canonical sections must FIRST be confirmed to carry the 2026-06-19 reframe before collapsing the phase2 duplicates.
**info_loss_risk:** preserve the budget identity budget(r)=(0.75−r)−0.215, the r-band table {0.21,0.26,0.30,0.33,0.38}, the 0.2135 m geometry floor, gate-relative-drops-bias-exactly, and the ESKF-attitude-bias lever — all mirrored in index_vision_estimator.md + MEMORY footguns; verify each before collapsing.

### D-3 [MED] project_phase2 plant-sysID campaign (characterize-sweep, S14, TWIN-FALSIFY/S16, TOGT, coast-replay) duplicated
**Location:** `project_phase2_rl_vision_decisions.md:163-333, 671-676`
**issue_type:** duplication · **severity:** med
**Evidence:** Full §CHARACTERIZE-SWEEP (:189), §S14 STATIC-MAP (:234), §TWIN-FALSIFY (:271), §TOGT-BOUND (:302), §COAST-REPLAY (:671). project_rl_increment_history.md headers include "CHARACTERIZE-SWEEP (510da24)", "S14 (static super-rate map)", "TWIN-FALSIFY → S16 (aero)"; index_control_sim.md / project_ctbr_control_sysid.md own the CTBR plant conventions. The static-map params (s∈[0.25,0.35], τ∈[0.015,0.030], alpha_max∈[200,320]) + quad-drag c2≈0.052 recur verbatim.
**recommended_action:** Collapse to pointers to project_rl_increment_history.md (campaign milestones) + project_ctbr_control_sysid.md / index_control_sim.md (durable plant conventions) + the handoff WRITEUP paths. Keep the one-line DR-band nominals only if NOT already in the control-sim topic file (verify).
**info_loss_risk:** preserve the measured DR bands + the quad-drag/convex-collective anchors — confirm in project_ctbr_control_sysid.md before pruning; if absent, MOVE.

### D-4 [MED] project_phase2 30 Hz speed-ceiling + parallel-systems + advisor-triage blocks duplicate index_vision_estimator
**Location:** `project_phase2_rl_vision_decisions.md:700-745, 749-769, 773-859, 1160-1203`
**issue_type:** duplication · **severity:** med
**Evidence:** §POLICY-DECISION-RATE 30 Hz (:700), §SPEED-CEILING-ANALYTIC (:708), §PARALLEL-ONBOARD-SYSTEMS-LEDGER (:749), §ADVISOR-TRIAGE (:773), §ADVISOR-REPORT-2-TRIAGE (:1160). Canonical: index_vision_estimator.md:131 §30 Hz speed-ceiling analytic, :144 §Parallel onboard systems, :156 §Advisor-triage open queue. The advisor-triage ledger is operational queue state that belongs in an index, not a frozen journal.
**recommended_action:** Collapse §SPEED-CEILING-ANALYTIC, §PARALLEL-ONBOARD-SYSTEMS, §ADVISOR-TRIAGE, §ADVISOR-REPORT-2-TRIAGE to pointers into index_vision_estimator.md. The 30 Hz verdict ("binds only at ≥30 m/s × last-fix ≤10 m") sits in the speed-ceiling section; note MEMORY's "binding gate-4 speed ~30 m/s" relaxes the urgency.
**info_loss_risk:** preserve the REJECTED ledgers (HSV pre-filter re-open conditions, policy-ensemble, wind-estimator, in-race map mutation) — confirm in index_vision_estimator.md advisor-triage before collapsing.

### D-5 [MED] project_phase2 §S2-DECISION block duplicated (canonical = index_rl_training §S2)
**Location:** `project_phase2_rl_vision_decisions.md:1320-1354`
**issue_type:** duplication · **severity:** med
**Evidence:** §S2-DECISION (:1320 "S2 architecture DECIDED: staged_monolithic_then_decomposed") reproduced in full. Canonical: index_rl_training.md:28 "## S2 architecture — DECIDED (2026-06-13)". The phase2 block also embeds the stale "Binding VQ2 risk = ESTIMATOR … East σ 0.47 m = 3.0× the 0.155 m gate-4 margin" (:1334, flagged in S-6).
**recommended_action:** Collapse §S2-DECISION (:1320-1354) to a one-line pointer to index_rl_training.md §S2 + the handoff path (handoff/ultracode-planning-togt-s2-2026-06-13/REPORT.md). Strip the embedded stale gate-4-margin sub-bullet on the way (covered by C-1/S-6).
**info_loss_risk:** none — index_rl_training.md:28 carries the decision; the handoff REPORT carries the two-adversarial-lens detail.

### D-6 [MED] project_phase2 §OBS-CONTRACT block duplicates index_vision_estimator + MEMORY
**Location:** `project_phase2_rl_vision_decisions.md:1612-1635`
**issue_type:** duplication · **severity:** med
**Evidence:** §OBS-CONTRACT (:1612 "FROZEN — d5 layout, 20 dims") is a full duplicate of index_vision_estimator.md:105 "## OBS CONTRACT (2026-06-13, FROZEN — d5 layout wins…)" + summarized in MEMORY.md (obs_dim=20). Same layout text, σ_ref≈0.05/TAU_STALE≈0.10, same 33→36 critic note.
**recommended_action:** Collapse phase2 §OBS-CONTRACT to a pointer to index_vision_estimator.md §OBS CONTRACT. The contained "33→36 critic" line should ALSO be corrected per the symmetric-critic finding (C-3/C-8) — the obs-dim=20 freeze is correct; the "appended for critic" framing is the get_state design, not active training.
**info_loss_risk:** none — d5 layout is canonical in index_vision_estimator.md + MEMORY.md; nothing unique except the (separately-flagged) critic note.

### D-7 [MED] index_rl_training inc8 WEIGHT-tuning blow-by-blow duplicated; canonical home = project_rl_increment_history
**Location:** `index_rl_training.md:104-123` (RE-SMOKE/PILOT#1/#2/CONVERSION-GAP/RE-PILOT/CONVERGENCE/BAND-PASS/FIX-DRIVEN/SYNC-FIX/GPU-SMOKE) and `:111-117`
**issue_type:** duplication · **severity:** med
**Evidence:** index :117 "META: inc8 WEIGHT-tuning is CLOSED (window→entropy→shape→incentive, 4 clean NO-GOs each ruling out a layer)" + per-job logs (jobs 3273400/3273431/3273437/3273552/3273701/3274167/3274179, util 27.9%/28.5%/28.2%, RUNTAGs, branch/commit hashes, entropy_weight values, d_lock=18/d_acq=28). project_rl_increment_history.md:730 compresses the same arc to one line. The "Topic file pointers" section (index :127) explicitly delegates episodic detail to [[project-phase2-rl-vision-decisions]] / [[project-rl-increment-history]].
**recommended_action:** Designate project_rl_increment_history.md as CANONICAL for the inc8 reward-tuning episodic detail. Move the :104-123/:111-117 blow-by-blow into a single dated inc8 sub-section there; collapse the index copy to a 1-2 line pointer ("inc8 reward-WEIGHT saga (window→entropy→band-pass→fix-driven, 4 NO-GOs) → [[project-rl-increment-history]] §inc8"). Verify each fact transcribed before pruning.
**info_loss_risk:** the index entries contain UNIQUE operational detail (job IDs 3273552/3273701/3274167/3274179, hashes, entropy values, d_lock=18/d_acq=28) — do NOT delete from the index until each is confirmed present in the history file; consolidate then trim to a pointer.

### D-8 [MED] index_rl_training inc8 sub-section is topic-file-grade forensics (index over-length)
**Location:** `index_rl_training.md:99-123` (25 lines, many 600+ chars; file 57910 bytes)
**issue_type:** prune/duplication · **severity:** med
**Evidence:** Lines 99-123 contain full paragraph-length entries with /scratch artifact paths, .item() line numbers (peregrine_racing_inc8.py:305-316), tarball sha256s ("b0277fbd…cf40"), Hydra dir-collision forensics — topic-file-grade detail, not index-grade (e.g. :104 is a single ~1900-char bullet). Banking doctrine: route bloated detail DOWN a layer.
**recommended_action:** Route the per-run forensics (sync-fix .item() line numbers, util-profiling history, tarball hashes, /scratch paths, GPU-smoke harness bugs) DOWN into project_rl_increment_history.md / project_phase2_rl_vision_decisions.md with 1-line pointers. Keep in the index only: standing doctrine, current verdict, binding footguns. Cite the destination before removing each datum.
**info_loss_risk:** several forensic data (e.g. the GPU-smoke harness-bug triad at :123, the deferred parent compute_reward_terms sync source at :122) may be UNIQUE to the index — verify each is in a topic file first; if not, MOVE it down rather than drop.

### D-9 [LOW] MEMORY.md OBS SIGN = +L duplicated near-verbatim (NOW vs footguns sections)
**Location:** `MEMORY.md:9` and `:50`
**issue_type:** duplication · **severity:** low
**Evidence:** :9 "🚩 OBS SIGN = +L (CORRECT IN CODE; obs_from_zup:348 = R_w2g@(gate−pos), localization:86). d1/d2 spec PROSE 'estimator delivers −L' is wrong (would be a 24 m flip). NOT a past code bug — pinned by tests/test_obs_sign_faithfulness.py (+L 4.77e-7 / −L control breaks 24 m). Future C2 estimator→obs MUST deliver +L. → [[index-vision-estimator]]" duplicated at :50 (identical except "The d1/d2", "The future C2"). ~600 bytes repeated under cap pressure.
**recommended_action:** Keep the single canonical copy in the "Cross-cutting footguns" section (:50, the durable home) and collapse the NOW-section copy (:9) to a one-line pointer (or vice-versa) — not both full copies. Saves ~600 bytes against the cap.
**info_loss_risk:** none — semantically identical; one full copy + a pointer preserves all content.

### D-10 [LOW] MEMORY.md camera mount / VFoV-mislabel duplicated (SPEC CAMERA REVIEW vs footgun)
**Location:** `MEMORY.md:14` and `:54`
**issue_type:** duplication · **severity:** low
**Evidence:** :14 (SPEC CAMERA REVIEW: "Camera 20° SPEC-EXACT … VFoV=90° MISLABELED (it is HFoV); true VFoV≈58.7° — ALWAYS use fy=320…") and :54 (CAMERA MOUNT FIXED AT 20° … mount-uptilt lever DEAD … VFoV=90° MISLABELED … TRUE VFoV≈58.7° (fy=320, H=360) … Boresight ε_vert≈0.56° calibratable seam) carry the same four facts. Both point to [[index-vision-estimator]] §Spec facts.
**recommended_action:** Merge into one entry (keep the footgun copy at :54 as canonical — it is the actionable "mount-uptilt lever DEAD / always decode fy=320" rule; collapse :14 to a pointer to §Spec facts). Saves ~600 bytes.
**info_loss_risk:** none — union preserved in the retained copy + index_vision_estimator.md §Spec facts.

### D-11 [LOW] Cross-file gate-4 budget table (the double-count source) duplicated verbatim
**Location:** `index_vision_estimator.md:58-65`; `project_phase2_rl_vision_decisions.md:1545-1552`; `project_master_plan.md` (same identity referenced, :1558 clear-matrix)
**issue_type:** duplication · **severity:** med
**Evidence:** The identical budget table "budget(r) = (0.75−r) − 0.215" with rows 0.213/0.26/0.30/0.33/0.38 → budgets 0.322/0.275/0.235/0.205/0.155 appears verbatim in index_vision_estimator.md:58-65 AND project_phase2:1545-1552 (and the −0.215 identity underlies master_plan's clear-matrix). MEMORY.md:8 identifies this exact "−0.215 subtracted after r already accounts for the drone" as the double-count.
**recommended_action:** Canonical home = index_vision_estimator.md §Contact-radius reconciliation. Add the 2026-06-19 double-count correction THERE once (the "budget" column double-subtracts the chassis; real clearance = 0.75 − r, i.e. 0.37-0.47 m for r∈[0.28,0.38]); collapse the project_phase2:1545-1552 copy to a [[index-vision-estimator]] pointer; annotate the master_plan clear-matrix as derived-from-the-superseded-budget. Keep the r-band reasoning (0.26-0.38, independently valid).
**info_loss_risk:** the contact-radius band (central 0.30, worst-case 0.38, geom floor 0.2135) + crash-halo derivation are load-bearing — preserve in the canonical copy; only the "budget"/"margin" column is the double-counted artifact to re-label, and only the duplicate copies to collapse.

### D-12 [LOW] Cross-file contact-radius 0.38 crash-halo derivation duplicated (≥4 places) — acceptable footgun
**Location:** `index_vision_estimator.md:54`; `project_phase2_rl_vision_decisions.md:1535`; `MEMORY.md:51` (footgun); `index_rl_training.md:40`
**issue_type:** duplication · **severity:** low
**Evidence:** "0.38 = empirical crash halo, NOT geometry; rigid-body chassis caps at 0.2135 m (3D half-diagonal √(0.14²+0.14²+0.08²)); central 0.30, band 0.26-0.33" in near-identical form in ≥4 places. Independent of the gate-4 bar correction (the radius is unaffected; only the budget COLUMN derived from it is double-counted).
**recommended_action:** Acceptable as a deliberate cross-cutting footgun (MEMORY.md:51) + canonical derivation (index_vision_estimator.md:54); the index_rl_training.md:40 + project_phase2:1535 copies can stay as in-context reminders. No collapse required; if pruning later, keep MEMORY.md:51 + index_vision_estimator.md:54 as the two canonical homes and reduce the others to pointers.
**info_loss_risk:** none — consistent across all copies, unaffected by the 2026-06-19 correction.

### D-13 [LOW] project_phase2 cross-references a2→b2 "carries same NO-GO direction" — NO-GO language corrected upstream
**Location:** `project_phase2_rl_vision_decisions.md:1600-1604`
**issue_type:** duplication/stale · **severity:** low
**Evidence:** :1603 "§BINDING VQ2 RISK = ESTIMATOR: … 0.55 m deployable (absolute) vs 0.155 m margin → NO-GO. Gate-relative CONDITIONAL-GO."; :1604 "a2's 0.52 m constant bias → SUPERSEDED by b2's range-collapsing 0.19 m near-band (favorable direction, carries same NO-GO direction)."
**recommended_action:** Covered by the §ESTIMATOR-RACESPEED supersession header (C-1/S-5): the absolute-path "→ NO-GO" is retained as historical but the standing gate-4 verdict is MARGINAL-PASSING. No separate edit beyond the header; flagged for completeness so the auditor of the live index does not re-bank "NO-GO" as current.
**info_loss_risk:** none — covered by the header; the a2→b2 bias-magnitude correction itself is unaffected.

### D-14 [LOW] index_control_sim sim-ops state machine duplicates reference_sim_ops verbatim
**Location:** `index_control_sim.md:70-76` (vs `reference_sim_ops.md:19-24`)
**issue_type:** duplication · **severity:** low
**Evidence:** index :71 "HOME→Enter→waiting-room (started=False…) → Enter AGAIN → GO (… cold-launch waiting room takes TWO Enters)." and :72 "From a FINISHED race: ESC+Down x3+Enter RESTARTS the race…" reproduce reference_sim_ops.md:22-23.
**recommended_action:** Collapse index :70-76 to a one-line digest ("cold-launch takes TWO Enters; from FINISHED, ESC+Down x3+Enter restarts to waiting-room — full chain in [[reference-sim-ops]]"); rely on the existing pointer.
**info_loss_risk:** none — identical detail in reference_sim_ops.md:19-24.

### D-15 [LOW] project_phase2 sim-ops facts scattered in the RL journal (belong in reference_sim_ops)
**Location:** `project_phase2_rl_vision_decisions.md:226-232, 294-300, 680-696`
**issue_type:** duplication · **severity:** low
**Evidence:** Durable sim-ops facts embedded mid-journal: ":226 THIRD sim idle state … recovery = Win32-focus + Enter ×2"; ":294 ZOMBIE DUAL-INSTANCE mode"; §NEW SIM OPS autoreset-latched-throttle / spin-detection / between-flight reset / SIM LAUNCH PATH (:680-696). These mechanics belong in reference_sim_ops.md per the MEMORY index.
**recommended_action:** Move the sim-ops facts (idle states, zombie dual-instance, autoreset hazard, spin detection, between-flight reset, SIM LAUNCH PATH) to reference_sim_ops.md if not already there; leave a pointer.
**info_loss_risk:** these are load-bearing operational footguns — verify each exists in reference_sim_ops.md BEFORE removing; if absent, MOVE. The SIM LAUNCH PATH + autoreset-latched-throttle hazard are especially important to preserve.

### D-16 [LOW] project_ctbr COLL_MAP entry overlaps index_control_sim + project_phase2 §SUBSTRATE-AUDIT (3-way)
**Location:** `project_ctbr_control_sysid.md:98-114` (twin harness + canonical-gains) and `:116-117` (COLL_MAP)
**issue_type:** duplication · **severity:** low
**Evidence:** :116-117 "§COLL_MAP-RECONCILIATION … COLL_MAP table CLEARED … Full substrate audit: [[project-phase2-rl-vision-decisions]] §SUBSTRATE-AUDIT" duplicates index_control_sim.md:50 "COLL_MAP NOT A BUG". The twin-harness/canonical-gains block (:98-114) is also summarized in index_control_sim.md (correct index→topic layering).
**recommended_action:** Leave the deep derivations in the topic file (correct layering). Optionally collapse the COLL_MAP paragraph (:116-117) to a one-line pointer: "COLL_MAP over-prediction = NOT A BUG (reconstruction artifact); do NOT refit QUAD_DRAG. → [[project-phase2-rl-vision-decisions]] §SUBSTRATE-AUDIT."
**info_loss_risk:** none IF collapsed to the cited pointer — full COLL_MAP reasoning is in project_phase2 §SUBSTRATE-AUDIT + index_control_sim.md:50.

### D-17 [LOW] reference_competition_materials velocity-on-wire fact repeated three times
**Location:** `reference_competition_materials.md:44, :52, :75` (with related :58, :63)
**issue_type:** duplication · **severity:** low
**Evidence:** The "is velocity on the wire?" question restated near-verbatim: :44 "Whether velocity is in the telemetry stream … biggest swing on VQ2 difficulty."; :52 "Velocity-telemetry check (2026-05-29, re-verified): … velocity is NOT confirmed-available; design estimator to derive it…"; :75 (COWORK-1) "§4.5 lists 'linear velocities'; … NO LOCAL_POSITION_NED/GLOBAL_POSITION_INT/ODOMETRY → self-localize REQUIRED". The practice-sim answer (velocity IS given via LPN vx,vy,vz) lives in reference_sim_interface.md:23,44.
**recommended_action:** Consolidate the three restatements into a single dated bullet (official §4.3 scored wire = NO carrying velocity message → derive; our practice sim DOES stream it via LPN — offline-only) + a pointer to [[reference-sim-interface]]. Collapse :44/:52 into a pointer to the COWORK-1 :75 statement.
**info_loss_risk:** low — preserve the methodological note ("derive velocity via IMU integ + vision finite-diff + drag, confirm via msg_audit") from :52 when consolidating.

### D-18 [LOW] reference_sim_interface VFoV-is-HFoV fact restated within a bullet + across files
**Location:** `reference_sim_interface.md:39` (twice within the bullet) and `reference_competition_materials.md:78, :65`
**issue_type:** duplication · **severity:** low
**Evidence:** sim_interface.md:39 states the "VFoV=90° is the HFoV; true VFoV≈58.7°; ALWAYS use fy=320" fact, then RESTATES it in the same bullet with a 🚩. The identical fact also at competition_materials.md:78 ("spec 'VFoV=90°' = HFoV bug (true VFoV≈58.7°)") and :65 ("Elodin independently confirms VFoV≈58.72°").
**recommended_action:** Within sim_interface.md:39 collapse the two intra-bullet restatements into one. Cross-file duplication is acceptable (sim_interface = authoritative wire facts; competition_materials = spec/Elodin context) — leave one canonical statement in sim_interface.md and let the others point to it.
**info_loss_risk:** none — fact fully preserved; only the intra-bullet restatement trimmed.

### D-19 [LOW] reference_sim_interface vs reference_sim_ops dual sim-package launch-path facts (no cross-ref)
**Location:** `reference_sim_interface.md:10` vs `reference_sim_ops.md:15`
**issue_type:** duplication · **severity:** low
**Evidence:** sim_interface.md:10 "AI-GP Simulator v1.0.3364.zip … Extracted on the dev laptop at C:\Users\Fengy\Downloads\AIGP_sim\"; sim_ops.md:15 "C:\Users\Shadow\Downloads\AI-GP Simulator v1.0.3364\AIGP_3364\FlightSim.exe". Same version (3364), different host paths (laptop vs ShadowPC). Not a contradiction (two machines) but no cross-reference.
**recommended_action:** No change to the facts; optionally add a cross-pointer noting the laptop path (sim_interface) vs the ShadowPC launch path (sim_ops) are the same v3364 package on different hosts.
**info_loss_risk:** none.

### D-20 [LOW] index_strategy_meta VQ-mechanics + budget/banking/canary digests restate MEMORY/competition-materials — INTENDED layering
**Location:** `index_strategy_meta.md:5-6` and `:23-46`
**issue_type:** duplication (intentional) · **severity:** low
**Evidence:** :5 "VQ1 = PASS/FAIL (spec 8, 8-min cap). VQ2 = fastest-valid." restates reference_competition_materials.md:33; budget/banking/canary (:23-46) restate MEMORY.md:34-37. This is the intended MEMORY→sub-index→topic layering (digest + pointer), NOT a contradiction.
**recommended_action:** No collapse — correct thin-index design. Action: keep digests but ensure they track their detail homes (apply the Banking rev-5 (S-16) and Q(1)/Q(5)-resolved (S-17) fixes so the digest does not silently lag).
**info_loss_risk:** none — intentional layered restatement; pointers present.

### D-21 [LOW] Duplication: boresight −0.25 metric finding + σ_p0 GO-metric stated in 3 places
**Location:** `index_vision_estimator.md:253-258` vs `project-fullstack-burn.md:23` and `MEMORY.md:8`
**issue_type:** duplication · **severity:** low
**Evidence:** index :253-258 details "BoresightCorrection(vert_offset_m=−0.25) into the +L lever … DEPLOYED 5764291"; fullstack-burn.md:23 restates "keep frames.BORESIGHT=-0.25 … binding bias is metric/range-flat (−0.27 m, N=13,308)"; MEMORY.md:8 carries "boresight bake −0.25 DEPLOYED 5764291".
**recommended_action:** Keep the index as the canonical home for the boresight DERIVATION (full FORM RESOLVED analysis); reduce fullstack-burn.md:23 + MEMORY.md to a one-line "−0.25 deployed → [[index-vision-estimator]] §BORESIGHT FORM RESOLVED" pointer. Do NOT delete the derivation from the index.
**info_loss_risk:** none — one canonical derivation with pointers; the deployed-constant fact retained everywhere as a one-liner.

### D-22 [LOW] project_phase2 §Cross-references / a2→b2 — see D-13 (recorded under cross-references)

---

# GAPS (high → low)

### G-1 [MED] index_rl_training nowhere reflects the RETRACTION of the "pivot off RL / near-field estimator" recommendation
**Location:** `index_rl_training.md` (absent — grep for near-field|RETRACTED|"RL stays" = 0 hits)
**issue_type:** gap · **severity:** med
**Evidence:** MEMORY.md:8 "RL IS the right tool — the 'pivot off RL to a near-field-gate estimator' recommendation is RETRACTED; next lever = RL to lift reach/pass-rate; vision = SUPPORT, do NOT lean on it." The RL domain sub-index — the natural home — contains no statement of it.
**recommended_action:** Add one line to the inc8 summary block: "Gate-4 = REACH/PASS-RATE problem (gate off the racing line), NOT sub-8cm centering; RL is the right lever (the near-field-estimator pivot is RETRACTED); vision = support. → MEMORY.md / [[index-vision-estimator]]." (Pairs with S-2.)
**info_loss_risk:** none — statement preserved in MEMORY.md:8; surfaces it in the correct domain index.

### G-2 [MED] Thin-index ROUTING violation — the newest result (det-retrain) stranded in MEMORY.md:8 (over size limit)
**Location:** `MEMORY.md:8` (file 24798 bytes, self-described "over the 24.4KB limit" at project-fullstack-burn.md:70)
**issue_type:** gap (routing violation) · **severity:** med/high
**Evidence:** MEMORY.md:8 is a single ~4.5 KB megaline (18.3% of the 24798-byte file at the ~25 KB cap) carrying BOTH the gate-4 bar correction AND the full deterministic-retrain result (det reach 0.467, std-cap lever, σ_p0 0.15-0.20 re-judgement, near-field-estimator retraction). grep confirms "det reach", "0.467", "std-cap", "MARGINAL-PASS" appear ONLY in MEMORY.md (the retraction also in project-fullstack-burn.md:8). Per the BANKING doctrine (MEMORY.md:36) detail belongs DOWN in the topic file + sub-index with a one-line pointer; here the load-bearing detail is stuck UP in the over-limit index while the topic files still narrate the superseded verdict. (This is the prime-mover finding — line 8 also bundles ≥8 distinct items: gate-4 correction, ARCHITECTURE PIVOT, LOOK-AT PRIMITIVE, the S0→S3 ladder/obs-gap saga, through_centering_reward (9cecf64), RECENTER FLEW 3/3 (job 3276449), the σ_p0 retrain (42ddb0b), the SIGN FOOTGUN launch config.)
**recommended_action:** **Sequence (do NOT reverse):** (1) bank the det-retrain result DOWN into project_rl_increment_history.md §inc8 (closing bullet, S-3) and the bar-correction + σ re-judgement into index_vision_estimator.md §margin/§TERMINAL-LOCK (correction banners, S-4); route the episodic inc8 ladder/recenter/std-cap detail to [[index-rl-training]] + [[project-rl-increment-history]] §inc8; (2) verify the pointer targets exist; (3) THEN compress MEMORY.md:8 to keep at top level ONLY (a) the corrected gate-4 verdict (real clearance 0.37–0.47, bar σ_p0_lat ≲0.15, MARGINAL-PASSING, RL stays the tool, reach/pass-rate problem) and (b) the LIVE SIGN FOOTGUN launch config (`++env.lookat_g_yaw=-3.0 ++env.lookat_g_pitch=3.0` double-plus opposite-sign, warmup=0, rw_centering=1.0, per-seed hydra.run.dir, boresight bake −0.25). This both fixes the routing violation and brings MEMORY.md back under its limit.
**info_loss_risk:** HIGH if trimmed before banking — grep confirms RECENTER-FLEW-3/3, std-cap/noise-anneal lever, det reach 0.467, and MARGINAL-PASSING are NOT yet present in index_rl_training.md or index_vision_estimator.md (recent commits 42ddb0b/511e85c/c5d60a0 edited MEMORY.md only). Preserve by banking FIRST (handoff/inc8-deterministic-retrain-2026-06-19/REPORT.md holds the source detail, exists on disk), THEN collapse line 8 to pointers. The LIVE SIGN FOOTGUN launch config MUST be preserved at top level (load-bearing for the next launch).

### G-3 [MED] index_strategy_meta "Topic file pointers" missing the live domain indexes (RL/vision/control)
**Location:** `index_strategy_meta.md:59-70`
**issue_type:** gap · **severity:** med
**Evidence:** :59-70 "Topic file pointers" lists strategy/competition/feedback topic files but NOT [[index-rl-training]], [[index-vision-estimator]], or [[index-control-sim]] — the three domain indexes that now carry the live engineering truth (MEMORY.md:64-67 Library index). A reader on the strategy sub-index cannot navigate to the live RL/vision/control state.
**recommended_action:** Add [[index-rl-training]], [[index-vision-estimator]], [[index-control-sim]] to the pointer list at :59-70 (or add a "sibling domain indexes" note pointing to MEMORY.md's Library index).
**info_loss_risk:** none — these indexes exist and are linked from MEMORY.md; this restores cross-navigation only.

### G-4 [LOW] index_control_sim simops.py promotion DISPATCHED but unverified (file absent under that name)
**Location:** `index_control_sim.md:67`
**issue_type:** gap · **severity:** low
**Evidence:** ":67 simops.py promotion to scripts/ DISPATCHED (LAPTOP-PROMOTE-SIMOPS)." Repo check: scripts/ has sim_focus.py but NO scripts/simops.py; find . -name simops.py returns nothing. The promotion has not landed under that name (possibly subsumed by scripts/sim_focus.py per commit 972cebb "promote sim_focus.py").
**recommended_action:** Reconcile: confirm whether simops.py was subsumed into scripts/sim_focus.py and update the note, or flag LAPTOP-PROMOTE-SIMOPS as still-open. Do not leave it as "DISPATCHED" if subsumed.
**info_loss_risk:** none — clarifying the true state preserves intent; the stale "DISPATCHED" currently hides that scripts/simops.py is absent.

### G-5 [INFO] Wiki-pointer hyphen↔underscore convention — verified clean (no broken links, no orphans)
**Location:** `memory/` (all 29 *.md; e.g. [[project-phase2-rl-vision-decisions]] → project_phase2_rl_vision_decisions.md; inline [[index-rl-training]] etc.)
**issue_type:** gap (verification) · **severity:** low
**Evidence:** Every [[target]] in the corpus resolves to exactly one real file under hyphen→underscore normalization — verified across all audited files (index/project/reference/feedback). The hyphen/underscore mismatch is the project-wide convention, not a defect. Computed inbound-link count for every file: ZERO orphans except MEMORY.md (root index, expected) and README.md (human-facing, outside the [[]] graph, expected). project-fullstack-burn.md IS reachable (MEMORY.md:6, :49). NOTE: several index/project files have EMPTY 'name:' frontmatter (index_strategy_meta, index_rl_training, index_control_sim, index_vision_estimator, project_detector_training_pipeline, project_ctbr_control_sysid, project_parked_backlog) — pointers still resolve by filename convention but the missing frontmatter name could break a stricter automated resolver.
**recommended_action:** No broken pointers to fix. Optional hygiene: backfill the empty 'name:' frontmatter on the index_* + a few project_* files so [[slug]] resolution does not depend solely on filename munging. Flag to commander; not a required edit. If a stricter resolver is ever adopted, a global hyphen→underscore normalization is the fix — do not hand-edit individual links.
**info_loss_risk:** none — informational.

### G-6 [INFO] Gate-4 σ_p0 correction confirmed ABSENT from all reference_* and feedback_* files (clean — no propagation needed there)
**Location:** `memory/` (all reference_* and feedback_* audited)
**issue_type:** gap (negative result) · **severity:** low
**Evidence:** grep for "0.08", "near-field", "NO-GO", "fix-seating", "pivot off RL", "sigma_p0/σ_p0", "closure bar" across all reference_* and feedback_* files returned NO matches. The stale gate-4 pre-correction language is NOT present in any reference/feedback file — it lives only in MEMORY.md and the vision/RL/phase2/backlog files. The correction-propagation work is confined to those files, not the reference/feedback layer.
**recommended_action:** No edit needed to the reference/feedback files for the gate-4 correction. Recorded as a NEGATIVE result so the commander knows the reference/feedback layer is clean on this axis.
**info_loss_risk:** none — informational.

---

# PRUNE (high → low)

### PR-1 [HIGH] MEMORY.md:8 over-cap megaline (4531 bytes = 18.3% of the file; ≥8 bundled items)
**Location:** `MEMORY.md:8`
**issue_type:** prune · **severity:** high
**Evidence:** Line 8 is 4531 bytes — 18.3% of the 24798-byte file (at the ~25 KB cap). It bundles ≥8 distinct items behind one bullet: gate-4 bar correction, ARCHITECTURE PIVOT, LOOK-AT PRIMITIVE, the full inc8 S0→S3 ladder/obs-gap saga, the through_centering_reward build (9cecf64), RECENTER FLEW 3/3 (job 3276449), the σ_p0 deterministic-stability RETRAIN result (42ddb0b, det reach 0.467, std-cap lever), and the SIGN FOOTGUN launch config. The BANKING directive (:36) mandates routing bloated/superseded detail DOWN a layer with a one-line pointer. Line 8 is the prime violation.
**recommended_action:** Split line 8 — keep at top level ONLY (a) the corrected gate-4 verdict and (b) the LIVE SIGN FOOTGUN launch config; route the episodic inc8 ladder/recenter/std-cap/det-reach detail to [[index-rl-training]] + [[project-rl-increment-history]] §inc8 with one-line pointers. **CAUTION: do this AFTER banking the new detail down (see G-2 sequence).**
**info_loss_risk:** HIGH if trimmed before banking — see G-2. Bank into index_rl_training.md + project_rl_increment_history.md FIRST (handoff REPORT holds the source), THEN collapse to pointers.

### PR-2 [HIGH] project_phase2_rl_vision_decisions.md is an over-cap monolith (185 KB / 1733 lines; un-pruned accretion log)
**Location:** `project_phase2_rl_vision_decisions.md:1-1733`
**issue_type:** prune/presentability · **severity:** high
**Evidence:** 185198 bytes / 1733 lines — BY FAR the largest memory file (next: project_rl_increment_history.md 89786 bytes, ~2.06× smaller). A single chronological journal of EVERY phase-2 session 2026-06-07..2026-06-14 (S1.2, S14, S16, S17, S18, frame-audit, crab-diag, training-doctrine, inc4/5/6/7, TOGT, estimator-racespeed, gate-relative blueprint, obs-contract). Not a thin topic file — an un-pruned accretion log. Banking directive: route bloated detail DOWN a layer.
**recommended_action:** SPLIT/route this monolith. The RL-substrate + plant-sysID + inc-lineage half (~lines 26-1037) belongs under project_rl_increment_history.md / index_rl_training.md (D-1, D-3, D-5); the vision/estimator/gate-4/obs-contract half (~89-161, ~490-569, ~1258-1733) belongs under index_vision_estimator.md (D-2, D-4, D-6). Collapse each retired session to a one-line [[pointer]] + its handoff/ writeup path. Keep only the genuinely-unique 2026-06-07 planning-session decisions (substrate bake-off rationale, RL I/O doctrine, pixel-to-control rejection) not captured elsewhere. Add the freeze banner (P-4) FIRST.
**info_loss_risk:** most content is duplicated in canonical files (cited per D-1..D-6) + in handoff/*/WRITEUP.md paths embedded in each section. Before collapsing any section, verify its handoff path + canonical-index entry both exist; preserve the handoff/ pointer in the collapsed one-liner.

### PR-3 [MED] project_phase2 within-file supersession churn (inline ❌/⚠️/strikethrough kept in full prose)
**Location:** `project_phase2_rl_vision_decisions.md:163-188, 181-187, 302-318, 420, 449-452, 628-629, 886-887, 932-997, 1088-1090, 1156`
**issue_type:** prune · **severity:** med
**Evidence:** Layered self-superseding records kept inline: §2nd-order re-sysID "STRUCTURALLY SUPERSEDED" (:163), "+30% rate_gain band DR proxy DISPROVEN" (:181), TOGT "CORRECTS the §TWIN-FALSIFY CONSERVATIVE note" (:314), inc4 "RETIRED … VOID" (:581,628), inc5 "formally retired" (:886), §S18-THRUST-LAPSE "VOIDED by §FRAME-AUDIT" (:932), "Lapse voided" (:1088). Many carry both the wrong claim AND its retraction in full prose.
**recommended_action:** For each superseded sub-block, consolidate to a single line: "<claim> — SUPERSEDED by <successor §/commit>, see <canonical file/handoff>". The detailed why-it-was-wrong belongs in project_rl_increment_history.md "Bug histories (root-caused; do not re-litigate)" (header at :22) — route the rationale there, leave a pointer here (or delete here once the file is collapsed per PR-2).
**info_loss_risk:** preserve the do-NOT-re-litigate lessons (PI-windup DISPROVEN, lapse code stays in repo defaults OFF, "internal consistency cannot catch a proper-rotation conjugation"). project_rl_increment_history.md §Bug histories is the designated home — verify each lesson is recorded there before pruning the long-form version.

### PR-4 [LOW] index_strategy_meta stale ops residue (branch cleanup names / fix_surrogate "reconfirm pending")
**Location:** `index_strategy_meta.md:56-57`
**issue_type:** prune · **severity:** low
**Evidence:** :56 "fix_surrogate MERGED to main (ffb2c74, 2026-06-14) … from-root full-count reconfirm pending"; :57 "Branch cleanup (2026-06-14): 2 stale local-only branches+worktrees pruned (claude/inspiring-tereshkova-9bd0f7, claude/magical-sutherland-546d7e) … 3 real-artifact branches KEPT". Point-in-time 2026-06-14 ops snapshots; the "reconfirm pending" is resolved (947 collected) and the named ephemeral branches are long gone.
**recommended_action:** Consolidate: drop the resolved "from-root reconfirm pending" clause + the dead branch names (the WORKTREE HYGIENE doctrine at :55 captures the generalizable rule). Keep the fix_surrogate-merged fact as a one-liner only if not already in index_rl_training.
**info_loss_risk:** before pruning the fix_surrogate-merged line, confirm ffb2c74 + merge scope is recorded in index_rl_training.md or project_rl_increment_history.md; if absent, MOVE it rather than delete.

---

# PRESENTABILITY (high → low)

### P-1 [HIGH] project_phase2 has no freeze/"current-as-of" marker — all post-2026-06-14 supersessions invisible to a reader who lands here
**Location:** `project_phase2_rl_vision_decisions.md:1-14` (front-matter + "How to apply") and absence of any freeze marker
**issue_type:** presentability · **severity:** high (combined with med per agents)
**Evidence:** :9-10 "Captures the strategy decisions from the 2026-06-07 … planning session (post-VQ1, pre-build). NOT yet executed …" The newest dated content is :1730 "### Ranked residual risks (top 3) — UPDATED 2026-06-14". Nothing tells a reader the file stops at 2026-06-14 and that inc8-flight, look-at, VQ2-ensemble, and the 2026-06-19 gate-4 correction all post-date it.
**recommended_action:** Add a one-line banner under the front-matter (~:14): "STATUS: append-only historical decisions log, current as of 2026-06-14. Verdicts after 2026-06-14 (inc8 flight, look-at primitive, VQ2 ensemble, 2026-06-19 gate-4 bar correction) live in MEMORY.md NOW + [[index-vision-estimator]] + [[index-rl-training]] + [[project-rl-increment-history]]. Sections flagged ⚠️ SUPERSEDED below." Makes the whole-file staleness explicit without rewriting individual sections.
**info_loss_risk:** none — purely additive banner; orients the reader.

### P-2 [HIGH] project_master_plan 53 KB reads as the active LIVING SSOT but is a frozen baseline — needs a dated freeze banner (single highest-leverage edit)
**Location:** `project_master_plan.md` (entire 184-line / 53 KB file)
**issue_type:** presentability · **severity:** high (med per agent; elevated as the single-edit fix for S-7)
**Evidence:** Presents as the active LIVING SSOT (:2 "current best state", :10 "update IN PLACE") with no freeze-date banner, yet nearly every operational section (RL status, speed ceiling, position doctrine, test count, build status, training rig, compute topology) is overtaken (S-7, S-4, S-8, S-15, S-34, S-35, S-36, S-37, S-38). Stale and still-valid content (frozen data contracts, IPPE convention lock :107, gate-ordering :79, failure-handling :165) interleaved without dating.
**recommended_action:** Add a dated FREEZE BANNER at the very top: "FROZEN 2026-05-31 as the Phase-1 architecture baseline. Sections on RL status, speed ceiling, position/self-loc doctrine, test count, build status, training rig, and compute topology are SUPERSEDED — see MEMORY.md + [[index-rl-training]]/[[index-vision-estimator]]/[[index-control-sim]]. Still-current: frozen data contracts, frame/IPPE conventions, gate-ordering, failure-handling principles." Single highest-leverage edit to stop reality-drift confusion without losing content. (Encapsulates the S-7 demotion.)
**info_loss_risk:** none — banner only; nothing removed.

### P-3 [MED] project_parked_backlog F/T/D triage currency — section membership no longer matches the legend
**Location:** `project_parked_backlog.md:3-6` (disposition legend) and section placement overall
**issue_type:** presentability · **severity:** med
**Evidence:** The legend (:4-6) defines F=actionable/open, T=dormant-until-trigger, D=closed. But FOLD-NOW (F) holds 6 CLOSED items (#3/#15/#16/#37/#40/#62, S-12); DORMANT-T holds 1 done item (#74) + 2 fired-trigger items (#32/#33, #58-dup) + a date-lapsed trigger (A1). The most-recent dated entries are 2026-06-19 (#79) while the gate-4 correction of the same date (511e85c) was NOT propagated here (#70/#75/line-94, C-2). So section membership cannot be trusted to convey status, and the register lags the canonical memory by the single most important recent correction.
**recommended_action:** One consolidation pass: (1) relocate the 7 closed items to DEAD (S-12); (2) re-triage the 3 fired-trigger items (#32/#33, #58, A1, S-12); (3) propagate the 2026-06-19 gate-4 reach/pass-rate correction into #70/#75/line-94 (C-2). After that the F/T/D membership matches the legend and the register is current to 2026-06-19.
**info_loss_risk:** none — reorganization moves existing lines between sections; no content dropped (each relocation cites the inline closing commit).

### P-4 [LOW] project_parked_backlog COWORK-1 cluster-key index — verified clean (NOT duplication)
**Location:** `project_parked_backlog.md:8` (cluster key) vs `:43,:60,:61,:62,:66,:75` (#12/#47/#48/#49/#53/#67 T-entries)
**issue_type:** presentability (verification) · **severity:** low
**Evidence:** Cluster-key :8 names #12/#47/#48/#49/#53/#58/#67/#13 and each also has its own T-entry (each item appears exactly twice). This is an INTENTIONAL index (the key un-gates them, the T-entry holds the detail), NOT harmful duplication.
**recommended_action:** No change. Leave the cluster-key-as-index pattern intact; it aids the "when COWORK-1 answers land, un-gate these" workflow. (Recorded so a future auditor does not mistake it for duplication.)
**info_loss_risk:** none.

### P-5 [LOW] MEMORY.md `[[pointers]]` / `[[pointer]]` — FALSE-POSITIVE broken links (doctrinal prose, not navigation)
**Location:** `MEMORY.md:1` ([[pointers]]) and `MEMORY.md:36` ([[pointer]])
**issue_type:** presentability · **severity:** low
**Evidence:** :1 "...detail lives in the domain sub-indices + topic files; follow the `[[pointers]]`."; :36 "...route the bloated/superseded detail DOWN a layer (into the sub-index) and leave a one-line `[[pointer]]`...". Both are INSIDE backticks and are doctrinal references to the wiki-link SYNTAX, not navigational links to files named 'pointers'/'pointer'. No file slug 'pointers' or 'pointer' exists (verified). Every OTHER [[target]] resolves via hyphen→underscore normalization (12 real targets verified OK).
**recommended_action:** Treat both as FALSE POSITIVES (not real broken links) — no semantic fix needed. To silence a future link-checker, optionally render the literal words without double brackets (e.g. "`[[...]]` pointers"); do NOT create pointers.md and do NOT alter the doctrine text's meaning.
**info_loss_risk:** none — wording-only; meaning identical.

### P-6 [LOW] project-fullstack-burn:69 `[[rl-increment-history]]` — FALSE-POSITIVE (quoted changelog artifact, already fixed)
**Location:** `project-fullstack-burn.md:69`
**issue_type:** presentability · **severity:** low
**Evidence:** ":69 Broken pointer fixed: `[[rl-increment-history]]` → `[[project-rl-increment-history]]` (project_phase2_rl_vision_decisions.md)." The bare `[[rl-increment-history]]` is a QUOTED historical artifact (the old broken target, recorded as already-fixed), not a live link. The same line carries the correct `[[project-rl-increment-history]]`; project_phase2_rl_vision_decisions.md:1154 confirms it now uses the correct slug.
**recommended_action:** No fix — false positive. The only live (resolvable) link on the line is the correct project-rl-increment-history slug.
**info_loss_risk:** none.

### P-7 [LOW] project_red_team_pass numbering vs date labels inverted (pass_3 dated earlier than pass_2 it builds on)
**Location:** `project_red_team_pass_2.md:3,10` vs `project_red_team_pass_3.md:3,10`
**issue_type:** presentability · **severity:** low
**Evidence:** pass_2 dated 2026-05-30 ("received 2026-05-29/30"), ends at 159 tests; pass_3 description "2026-05-29 (eve of sim drop) 3rd red-team pass … 169 tests", body "169 tests green (was 159, +10)". pass_3 BUILDS ON pass_2's 159-test state yet carries an EARLIER date label (05-29 vs 05-30). Test-count chain (154→159 in pass_2, 159→169 in pass_3) is self-consistent; only the human-readable date labels appear inverted.
**recommended_action:** Reconcile the date labels: either correct pass_3's "2026-05-29" to its actual triage date (consistent with building on pass_2's 159 state), or add a one-line note that the dates are report-ARRIVAL dates while the numbering reflects triage order.
**info_loss_risk:** none — the test-count lineage already disambiguates ordering; cosmetic date-label clarification only.

### P-8 [LOW] index_strategy_meta organizer open-questions presentability — covered by S-17 (Q(1)/Q(5) resolved)
**Location:** `index_strategy_meta.md:32`
**issue_type:** presentability · **severity:** low
**Evidence:** See S-17 — the open-questions list reads as live while Q(1)/Q(5) are resolved by COWORK-1.
**recommended_action:** Covered by S-17 (mark Q(1)/Q(5) resolved; reduce the open set).
**info_loss_risk:** none.

### P-9 [LOW] index_vision_estimator outdated test-count / merge-status sentinels embedded inline (provenance markers)
**Location:** `index_vision_estimator.md:8, :38, :44, :115-119, :163, :230, :330, :342`
**issue_type:** stale/presentability · **severity:** low
**Evidence:** Point-in-time green-suite counts: :8 "692 green", :38 "15 green", :44 "703 passed / 35 skip", :163 "687→692 green", :230 "786 green", :330 "700→723 green", :342 "692→700 green". MEMORY current canonical is "947 tests collected" (and project-fullstack-burn.md:68 notes "~900 … not 723 (stale)"). These per-merge counts are historical provenance for each landed change, not current totals.
**recommended_action:** Leave the per-merge counts attached to their specific commits (legitimate provenance for when each feature landed), but do NOT treat any as the current suite size — the live sentinel is MEMORY.md "947". If trimming for bloat, these counts travel with their blocks into the topic file.
**info_loss_risk:** none — counts are commit-anchored provenance; preserved with their blocks.

### P-10 [LOW] project_detector_training_pipeline / project_ctbr / project_hardware stale test-count / branch markers (provenance)
**Location:** `project_detector_training_pipeline.md:107` (+ :106-109); `project_ctbr_control_sysid.md:122, :186`; `project_hardware_constraint.md:32`
**issue_type:** stale (provenance) · **severity:** low
**Evidence:** detector:107 "Committed red-team-tier-a fc29287 … 211 tests green." MEMORY footgun "main CANONICAL · 947 tests collected"; the red-team-tier-a branch was long merged to main. Similar dated branch/test snapshots: ctbr:122 ("Branch red-team-tier-a, 258 tests green"), :186 ("265 green"), hardware:32 ("159 tests green").
**recommended_action:** Point-in-time provenance stamps; leave the historical numbers but, if refreshing, append "(branch since merged to main; current suite 947)". Low priority — they read as dated snapshots, not current claims. Do not delete (they document which commit a finding landed on).
**info_loss_risk:** none — provenance value retained.

### P-11 [LOW] project_hardware_constraint front-matter "decision pending" framing doubly stale
**Location:** `project_hardware_constraint.md:3` (YAML description) + `:31` header
**issue_type:** stale (front-matter) · **severity:** low/med
**Evidence:** :3 'description: "Three-machine architecture: laptop (dev), Azure Windows GPU VM (sim host, decision pending Free Trial vs Student+Upgrade), Adroit Princeton (VQ2 ML training)."' The Free-Trial-vs-Student decision is long RESOLVED (:33 "Azure for Students = DEAD END for GPU"; :31 TensorDock provisioned 2026-05-29) and the host is now ShadowPC (S-10).
**recommended_action:** Update the YAML description to "Three-machine architecture: laptop (dev), ShadowPC (sim host; TensorDock/Azure = historical fallback), Adroit Princeton (VQ2 ML training)."
**info_loss_risk:** none — metadata description refresh.

### P-12 [LOW] CTBR sign conventions — four superseded layers presented sequentially without an up-front canonical pointer
**Location:** `project_ctbr_control_sysid.md:50-96, :135-156, :184-197`
**issue_type:** stale/presentability · **severity:** med
**Evidence:** The file stacks four successive sign verdicts without an up-front pointer to the final one: :50 "[2026-06-10 … rate sign [−1,−1,1]] (SUPERSEDED)"; :56 "[2026-06-12 … TRUE = [+1,−1,+1]] (bcc93f9)"; :78 "[2026-06-12 UPDATE #2 … q_true=q_raw·[1,−1,1,−1]; ω_true=−w_raw; cmd→rate [+1,+1,+1]] (93023cf — supersedes the above)". The "## Inner-rate system-ID" (:135-149) + Gate-0 saga (:184-197) still state the OLD measured signs (e.g. :136 body_rate_sign=[-1,1,-1]; :140 odo_rate_sign=[+1,-1,+1]) as if current. index_control_sim.md:27-31 carries only the final 93023cf TRUE conventions.
**recommended_action:** Add a one-line pointer at the top of "## Inner-rate system-ID" (:135) + Gate-0 saga (:184): "SIGNS HERE ARE THE LEGACY/AS-MEASURED ALIAS — the TRUE physical convention is q_raw·[1,−1,1,−1] / ω_true=−w_raw / cmd→rate [+1,+1,+1] per the 2026-06-12 UPDATE #2 block (93023cf); the CTBR/VQ1 LEGACY stack keeps the old alias intentionally — see index_control_sim.md §CTBR/VQ1 LEGACY." Do not delete the measured-sign detail.
**info_loss_risk:** none — all four convention layers are load-bearing history (the legacy alias is DELIBERATELY kept VQ1-proven); only an orientation pointer is added so a reader knows canonical-true vs legacy-alias.

### P-13 [LOW] inc8 SPEC DR σ values differ across files without a canonical-source note (index vs phase2)
**Location:** `index_rl_training.md:48` vs `project_phase2_rl_vision_decisions.md:1708`
**issue_type:** contradiction (mild)/presentability · **severity:** low
**Evidence:** index :48 "per-fix lateral σ = U[0.05,0.15] m/axis (MEASURED 0.10 m; supersedes modeled 0.265 m)"; phase2 :1708 "per-fix lateral σ~U[0.08,0.30] m/axis, N_eff~U[4,9]". Two different DR σ ranges for the same inc8 estimator-DR spec, neither cross-referencing the other as canonical.
**recommended_action:** Reconcile: the index's U[0.05,0.15] is annotated "MEASURED 0.10 m, supersedes modeled 0.265" and looks newer; annotate the stale one in phase2:1708, or add a "canonical = index" note. Confirm against rl/fix_surrogate.py calibration (σ_lat 0.104 per index:61) before editing.
**info_loss_risk:** low but real — if the U[0.08,0.30] range is a deliberately wider stress-DR (not stale), do not overwrite; reconcile by adding a note distinguishing measured-nominal vs stress-wide rather than deleting either.

### P-14 [INFO] False-positive stale "0.08" matches (verification, no action)
**Location:** `project_red_team_pass_3.md:22` (and similar unrelated 0.08 occurrences)
**issue_type:** presentability (verification) · **severity:** low
**Evidence:** red_team_pass_3.md:22 matched the "0.08" stale-scan but is unrelated to gate-4: "The report's `e1=0.01 vs e2=0.08 = '8× unambiguous'` mistakes sub-noise numerical noise for signal." This is a PnP reprojection-ambiguity epsilon example, not the gate-4 closure bar.
**recommended_action:** No action — confirmed unrelated to the gate-4 correction; recorded so the stale-scan is auditable.
**info_loss_risk:** none.

### P-15 [INFO] Red-team pass 2/3 "NOT yet merged to main" framing stale (fixes ARE in main) — see also stale
**Location:** `project_red_team_pass_2.md:12` and `project_red_team_pass_3.md:3,10`
**issue_type:** stale · **severity:** low
**Evidence:** pass_2:12 "branch red-team-tier-a, 154 tests green …; NOT yet merged to main"; pass_3:10 "Branch red-team-tier-a, 169 tests green …; NOT yet merged to main, NOT yet committed (pass-3 edits in the working tree pending user OK)." The fixes ARE now in main: verified P3P_FIX_COV_INFLATION=9.0 (localization.py:217), attitude_noise_std/σθ² term (localization.py:128-129), _takeoff_origin relative-climb (mission.py:75,101-108), KF max_dt_s clamp (kf_rewind.py:181), AMBIGUITY_EPS_PX=0.1 (gate_pose.py:68,198), fit_thrust_curve (sysid.py:66), sample_attitude_bias (firstcontact.py:106).
**recommended_action:** These two files are NOT superseded scratch — they are the load-bearing RATIONALE/provenance for tunable constants + the "don't re-litigate the rejected claims" record (all constants survive in live code). Update the "NOT yet merged" clause in both to "MERGED to main" (or drop it), so the status line isn't misread as live-pending work. Keep all reasoning intact.
**info_loss_risk:** none — only the merge-status clause is stale; the technical content maps 1:1 to live code constants and must be preserved as the provenance-of-record.

---

## Cross-cutting note (fix sequencing)

The high-severity contradictions C-1, C-2, C-3 and the gaps G-1, G-2 / prune PR-1, PR-2 are all facets of the SAME root cause: **the 2026-06-19 gate-4 correction + deterministic-retrain result landed only in MEMORY.md and was never banked down.** The correct order of operations (to lose nothing):
1. **Bank DOWN first** — det-retrain result → project_rl_increment_history.md §inc8 (S-3) + index_rl_training.md inc8 summary (S-2, G-1); bar-correction + σ re-judgement → index_vision_estimator.md §margin/§TERMINAL-LOCK banners (S-4).
2. **Propagate correction banners** into the remaining stale carriers (project_phase2 §ESTIMATOR-RACESPEED header S-5/S-6; index_rl_training :40/:50/:97/:119 S-41/C-1; project_rl_increment_history ladder bullets S-3/C-1; project_parked_backlog #70/#75/line-94 C-2/P-3; project-fullstack-burn:17/:56 C-1/C-2).
3. **THEN compress** MEMORY.md:8 to pointers (PR-1/G-2) and split the project_phase2 monolith (PR-2). Never compress/collapse before the down-bank lands and the pointer targets are verified to exist.
