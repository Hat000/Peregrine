# Cleanup Audit Appendix — Git Hygiene (branches / worktrees)

**Bucket:** `git`
**Date:** 2026-06-19
**Source agents:** `git:claude-branches`, `git:named-branches`, `git:worktree+origin`

## Area state (summary)

The branch/worktree landscape is dominated by **prune-safe redundancy**: the large majority of local
`claude/*`, `p2-*`, `worktree-agent-*`, and origin worker branches carry **zero unique content** — their
tips are strict ancestors of `main`, or their files are byte-identical blobs already merged (selectively or
in bundles). These are safe to delete with no information loss. The genuinely load-bearing issues are NOT
the redundant branches but **two HIGH-severity stale-verdict contradictions living ON MAIN** (the inc8
deterministic-retrain REPORT.md and the charming-jemison margin-closure report both still assert the
retracted gate-4 "NO-GO / 0.08-bar / pivot-off-RL-to-near-field-estimator" doctrine that `511e85c`
corrected in `MEMORY.md:8`). A handful of origin/worktree branches are **intentionally KEPT artifact
stores** (raw recordings, design sims) whose verdicts are banked but whose raw data is reproducible only by
re-recording — these must NOT be pruned. One MED gap (`p1-eskf-design` observability sims) and one MED gap
(VQ2 stack-audit with a STAGED dataset-split bug fix) carry unmerged content worth banking/merging before
any prune.

**Headline cleanup needs:** (1) Banner-correct the two on-main stale reports + annotate the on-main
margin_envelope.py/boresight-closure double-count; (2) prune the ~40 redundant branches; (3) preserve the
~6 memory-sanctioned artifact branches; (4) bank the two MED gaps (ESKF observability verdict; VQ2
F-SPLIT-1 dataset-split bug) before pruning their branches.

---

## CONTRADICTION (high → med)

### [HIGH] inc8 deterministic-retrain REPORT.md — stale gate-4 NO-GO / pivot-off-RL verdict living verbatim ON MAIN
- **Location:** `handoff/inc8-deterministic-retrain-2026-06-19/REPORT.md` (ON main @ c5d60a0; identical copy
  also on local branch `inc8-deterministic-retrain`). Affected lines: HEADLINE (~88/89), CONCLUSION
  (~113–135), MEMORY-DELTA (~124–150); specifically the σ_p0 table (~95–101).
  Merged onto main via `42ddb0b`; NOT corrected by `511e85c`.
- **Issue type:** contradiction · **Severity:** high
- **Evidence (exact quotes, CONFIRMED in-tree by this auditor):**
  REPORT.md asserts the OLD view as CURRENT truth:
  - `## ⭐ HEADLINE — Objective A ACHIEVED, Objective B MEASURED = NO-GO`
  - `**2-axis σ_p0_lat ≈ 0.15–0.20 m = NO-GO vs 0.08** (lat_p99 0.37–0.40 vs 0.24).`
  - `❌ **Gate-4 NOT closed by the look-at bet: 2-axis σ_p0_lat 0.16–0.20 m = NO-GO**, because fix_rate=0 at every deterministically-flyable gain. Definitive across the bracket.`
  - `🚩 **RECOMMENDED NEXT LEVER (not more look-at RL):** the NEAR-FIELD-GATE ESTIMATOR`
  - MEMORY-DELTA #2 `**Measured 2-axis gate-4 σ_p0 = 0.15–0.20 m = NO-GO vs 0.08**` and #4 `⇒ **NEXT LEVER = near-field-gate estimator** ... — NOT more look-at-gain RL.`

  DIRECTLY contradicts `main:memory/MEMORY.md:8` (correction `511e85c`):
  - `the σ_p0 ≲ 0.08 "closure bar" was a DOUBLE-COUNT ... REAL bar ≈ σ_p0_lat ≲ 0.15 (p99 ≲ 0.45)`
  - `Measured σ_p0 0.15–0.20 / p99 0.37–0.40 = MARGINAL-PASSING vs the real ~0.15/0.45 clearance (sim success 0.65) — NOT the NO-GO we mis-called vs the bad 0.08 bar`
  - `🚩 RL IS the right tool — the "pivot off RL to a near-field-gate estimator" recommendation is RETRACTED; next lever = RL to lift reach/pass-rate`

  `MEMORY.md:8` even cites this exact REPORT path while contradicting its verdict. The merge commit
  `42ddb0b` (of which `511e85c` is an ancestor) notes the conclusion is superseded, but the REPORT.md body
  was never corrected.
- **Recommended action:** Prepend a dated CORRECTION banner at the top of the on-main REPORT.md, and beside
  the HEADLINE / CONCLUSION / MEMORY-DELTA, e.g.:
  `>>> SUPERSEDED 2026-06-19 (511e85c): the 0.08 bar was a double-count; real bar ~0.15 / p99 ~0.45; measured 0.15–0.20 is MARGINAL-PASSING, NOT NO-GO. The near-field-estimator pivot is RETRACTED — RL stays the tool; gate-4 is a reach/pass-rate problem.`
  Keep the engineering sections verbatim (root-cause: `actor_logstd` saturation at exp(2)=7.39 on 2
  channels; the noise-anneal/std-cap lever; gp1.0 det reach 0.467; measured σ_p0 table). Strike-through /
  annotate ONLY the verdict/recommendation lines (~89, ~101, ~129–135, ~144–150). The branch carries zero
  unique content and is prunable separately, but pruning the branch does NOT fix the on-main stale report.
- **Info-loss risk:** none from annotating (corrected verdict already authoritative in `MEMORY.md:8`; the
  reusable lever is captured by `42ddb0b`). If the report were *deleted* instead of annotated, the durable
  wins would be lost: the rc1 `actor_logstd` saturation root cause (exp(2)=7.39 on 2 channels), the
  noise-anneal/std-cap lever result (first det-flyable 2-axis inc8, gp1.0 det reach 0.467), and the measured
  σ_p0 table (~95–101). Preserve by annotating in place, not deleting.

### [HIGH] charming-jemison (margin-closure-envelope) branch — carries the refuted gate-4 NO-GO doctrine
- **Location:** branch `claude/charming-jemison-b111c5` (tip 7654e99);
  `handoff/margin-closure-envelope-2026-06-14/REPORT.md` (NOT on main) + 27 results/adversarial files (NOT on
  main); only `margin_envelope.py` is on main.
- **Issue type:** contradiction · **Severity:** high
- **Evidence (exact quotes):** Branch REPORT.md states the double-count formula (line 36) and the refuted
  verdict (lines 80–82, 292–293, 338):
  - `Margin model: MARGIN(r) = W_EFF − r, W_EFF = 0.535 m (= 0.75 gate-clear-halfwidth − 0.215 chassis half-diagonal; the memory footgun budget(r)=(0.75−r)−0.215)`
  - `At the measured operating point the gate-4 margin does NOT close at the central r=0.30 ... CANNOT-SETTLE-OFFLINE SURVIVES.`

  Directly contradicts `main MEMORY.md:8` (`511e85c`, 2026-06-19): the W_EFF=0.535 / MARGIN=0.235 chain is the
  exact "drone subtracted TWICE" double-count it names; REAL bar ≈ σ_p0_lat ≲ 0.15; measured 0.15–0.20 =
  MARGINAL-PASSING, NOT the NO-GO. Of the branch's 29 files, only `margin_envelope.py` (byte-SAME on main via
  boresight-closure `3542fd1`) is merged; the REPORT.md + 27 result/adversarial files are unmerged.
- **Recommended action:** Do NOT merge the REPORT.md / results onto main as current truth — its headline
  (MARGIN=0.235, "does NOT close", CANNOT-SETTLE-OFFLINE survives, fix-rate ≥0.25 floor) is the exact analysis
  `511e85c` retracted. Prune the branch. If the raw 268-cell sweep data has archival value, archive the dir
  under a clearly-dated SUPERSEDED note (e.g. one-line header `SUPERSEDED 2026-06-19 by 511e85c — the
  W_EFF=0.535 MARGIN=0.235 chain is a double-count; do not cite as current`) rather than merging it live. The
  corrected doctrine is already on main (`MEMORY.md:8`).
- **Info-loss risk:** low — the substantive sweep methodology and the (now-correct-direction) lever-moved-to-
  fix-rate insight are echoed in main memory; the stale VERDICT must NOT be preserved as current. To keep the
  raw numbers, archive the dir with the SUPERSEDED banner so no future reader treats "does NOT close" as live.

### [MED] On-main stale artifact — the double-count survives in main's margin_envelope.py + boresight-closure REPORT
- **Location:** `main:handoff/margin-closure-envelope-2026-06-14/margin_envelope.py:77-78,102` and
  `main:handoff/boresight-closure-2026-06-14/REPORT.md:16,23,95,154,179`.
- **Issue type:** contradiction · **Severity:** med
- **Evidence (exact quotes):**
  - `margin_envelope.py:78` — `W_EFF = MARGIN_G4_AT_R038 + 0.38  # 0.535 m = 0.75 gate-clear-halfwidth - 0.215 chassis half-diag`; `:102` — `return W_EFF - r` (the exact double-count `511e85c` names).
  - `boresight-closure/REPORT.md:95` — `MARGIN(r) = W_EFF − r = (0.75 − r) − 0.215`: ... r0.30→0.235 ... r0.38→0.155`; and `:16/:154/:179` — `CANNOT-SETTLE-OFFLINE does NOT flip to CLOSE`.
  - No correction/SUPERSEDED note exists next to either on main
    (`git grep '2026-06-19|double-count|CORRECTED|RETRACT'` over `handoff/margin-closure-envelope-2026-06-14/`
    returns nothing). These contradict `main MEMORY.md:8` / commit `511e85c`.
- **Recommended action:** (Outside the strict branch scope but flagged per task.) Add a dated SUPERSEDED
  header to `handoff/margin-closure-envelope-2026-06-14/margin_envelope.py` and
  `handoff/boresight-closure-2026-06-14/REPORT.md` pointing to `MEMORY.md:8` / `511e85c` (W_EFF=0.535 /
  MARGIN=0.235 is a drone-subtracted-twice double-count; real clearance 0.37–0.47 m; "does NOT close" verdict
  retracted). Do NOT delete the code/data; annotate it stale.
- **Info-loss risk:** none from annotating — the corrected bar is in `MEMORY.md:8`; adding a SUPERSEDED
  pointer only prevents future mis-citation. Deleting the files is NOT recommended (sweep machinery has reuse
  value once the W_EFF formula is fixed).

---

## STALE (med → low)

### [MED] inc8-deterministic-retrain local branch — merged-by-cherry-pick + stale report
- **Location:** branch `inc8-deterministic-retrain` (local, tip `05809ba`); 6 commits ahead of main;
  `handoff/inc8-deterministic-retrain-2026-06-19/REPORT.md`.
- **Issue type:** stale · **Severity:** med
- **Evidence:** `git rev-list --count main..inc8-deterministic-retrain` = 6; the reusable code
  (`rl/inc8_noise_anneal.py`, `inc8_snapshots.py`, `inc8_select_ckpt.sbatch`, `peregrine_inc8_detstab.sbatch`,
  the 2 tests) was SELECTIVELY merged to main as `42ddb0b` (verified `rl/inc8_noise_anneal.py` exists on main;
  `42ddb0b` IS ancestor of main). The branch's `REPORT.md`/`monitor.sh`/`probe_logstd.py` are ALSO already on
  main (identical, `git diff` empty). The branch's 6 raw commits carry the stale "gate-4 NO-GO definitive,
  next lever = near-field estimator" subjects (`05809ba`, `da1a751`). NOTE: this is the branch currently
  checked out at the stale sibling worktree `C:/Users/Fengy/Downloads/Projects/Anduril`.
- **Recommended action:** Archive/retire the local branch — its KEEP content (lever code + report) is on main
  via the selective merge `42ddb0b`; the report banner-fix is tracked separately (see HIGH contradiction
  above). Do NOT force-merge its 6 raw commits (they carry the stale NO-GO framing in their subjects).
- **Info-loss risk:** none — all unique content (code + report) verified present on main (`42ddb0b` for code;
  identical `REPORT.md`/`monitor.sh`/`probe_logstd.py` paths on main).

### [MED] L3 at-speed gate-4 "margin CLOSES" (case-a) report — REFUTED conclusion on unmerged origin branch
- **Location:** `origin/claude/blissful-kalam-75f52b` : `handoff/l3-atspeed-recording-2026-06-14/REPORT.md`
  (1 commit `ee62f00`; NOT on main — verified absent).
- **Issue type:** stale · **Severity:** med
- **Evidence:** Report headline: `the margin-binding attitude/accel CHAIN bias ε ≈ 0 on both axes ... With
  ε≈0 + tight σ this is the G3 BIAS-FREE regime → margin CLOSES vs central r=0.30 (0.235 m)`. This case-a
  hypothesis was REFUTED by the `d7c592e` δ_map discriminator; `main:memory/index_vision_estimator.md:171`
  records it: `## L3 AT-SPEED GATE-4 BIAS PINNED — 'CLOSE' NOW REFUTED (2026-06-14; branch
  claude/blissful-kalam-75f52b; discriminator refutes case-a)`. Memory (`index_strategy_meta.md:57`)
  deliberately KEEPS this branch on origin: `3 real-artifact branches KEPT on origin
  (claude/blissful-kalam-75f52b L3 ... handoff artifacts only, findings banked, NOT merged)`.
- **Recommended action:** Leave the branch as-is per the existing memory decision (KEEP on origin as raw
  artifact). No branch correction needed (memory already flags 'CLOSE' as REFUTED; the L3 KEPT numbers gate-4
  σ_vert 0.10 / σ_lat 0.19, N=81 are banked at `index_vision_estimator.md:256`). If consolidating, add a
  one-line pointer in the report header to the δ_map refutation; do NOT delete (carries the only raw L3 shadow
  recording: `shadow_gate4.py` + 3 large rows.json).
- **Info-loss risk:** If pruned, lose the raw L3 at-speed shadow recording (`shadow_gate4.py` analysis script
  + ~225k-line gate2/3/4 rows.json captures). The load-bearing σ numbers are banked; the raw rows.json are
  reproducible-only-by-re-recording, so keep the origin branch as the artifact store (already
  memory-sanctioned).

### [LOW] stupefied-fermi (fix-surrogate) — branch code is SUPERSEDED by main (do NOT re-merge)
- **Location:** branch `claude/stupefied-fermi-20126c` (tip `c0d3f49`); `rl/fix_surrogate.py:110-116` and
  `tests/test_fix_surrogate.py:64-72`.
- **Issue type:** stale · **Severity:** low
- **Evidence:** All 11 `handoff/fix-surrogate-2026-06-14/*` files are byte-identical on main (SAME blob SHAs).
  The 2 code files DIFFER and main is NEWER: `git diff main..stupefied-fermi -- rl/fix_surrogate.py` shows the
  branch REVERTS main's evolution — branch has `accept_rlo: float = 16.191103551308505`; main has
  `accept_rlo: float = 12.0` with comment `accept_rlo lowered 16.191 ... -> 12.0 = the POINTED accurate floor
  (accept-geometry-2026-06-15)`. Branch tip dated 2026-06-14; main's change is `8cf23c9` dated 2026-06-15
  (after the branch was cherry-picked in via `ffb2c74`).
- **Recommended action:** Prune branch `claude/stupefied-fermi-20126c` — its handoff dir is byte-identical on
  main and its `rl/fix_surrogate.py` / tests are the OLDER pre-accept-geometry-2026-06-15 versions main has
  superseded (accept_rlo 16.191 → 12.0). Do NOT re-merge the branch's code (it would regress accept_rlo).
- **Info-loss risk:** none — the unique handoff content is on main; the branch's code is a strict predecessor
  of main's (re-merging would LOSE main's improvement, not the reverse).

### [LOW] p2-system-id — branch carries a stale obs[17:20]-gap test main has since CLOSED
- **Location:** branch `p2-system-id` (ahead=10, behind=23); `tests/test_sysid_production_obs.py`.
- **Issue type:** stale (prune-safe) · **Severity:** low
- **Evidence:** All 9 sysid test files + `handoff/system-id-2026-06-18/REPORT.md` + `wf_investigate.js`
  verified IDENTICAL to main. The lone differing file `tests/test_sysid_production_obs.py` is OLDER on the
  branch: branch R2 says `there is NO production obs[17:20]/confidence-triple builder in src/racer ... This
  test PINS the gap` (148 lines); main's R2 says `the obs[17:20] gap CLOSED -- present + faithful ... promoted
  into src/racer/estimator_obs.py (confidence_triple / estimator_obs20)` (181 lines, asserts byte-faithful).
  Main is the newer/correct version (gap closed per `dc56ce7`).
- **Recommended action:** Prune branch `p2-system-id` — all system-id registration content merged; the one
  differing test is stale (pins a gap main has since CLOSED). No unique unmerged content.
- **Info-loss risk:** none — the gap-closed test on main supersedes the branch's gap-open pin; the closure is
  itself the captured information.

### [LOW] inc8 re-pilot reward-window NO-GO report (ON main) — historically consistent, superseded code
- **Location:** `handoff/inc8-repilot-window-2026-06-15/REPORT.md` (ON main; `origin/worker/inc8-repilot-
  window-2026-06-15` has 9 commits, branch code superseded by main's newer `rl/inc8_reward.py`).
- **Issue type:** stale · **Severity:** low
- **Evidence:** Report concludes `## PART 2d ... NO-GO → STOP (architecture, not weights)` and `### STOP per
  the task -> ARCHITECTURE change (commander's call; NOT launched, no more weight sweeps)`. This is NOT the
  stale gate-4/0.08-bar conclusion — it is the reward-WEIGHT-vs-ARCHITECTURE finding MEMORY still endorses
  (`MEMORY.md:8`: `4 reward-WEIGHT iterations ALL NO-GO ⇒ it's reward ARCHITECTURE not weights ... →
  ARCHITECTURE PIVOT (Fengyou-approved)`). The branch's code (`rl/inc8_reward.py` reward-window experiment)
  differs from main and was NOT adopted (main evolved to the look-at primitive); branch is 9 commits ahead but
  its report is already on main and its code is superseded.
- **Recommended action:** No correction needed — the report's conclusion is consistent with current memory (it
  motivated the look-at pivot). The origin branch (9 commits, superseded reward-window code) is a candidate to
  archive/prune since its report is on main and its code path was abandoned. NOTE-only: verify no unique
  unbanked detail before pruning; the REPORT.md is already on main (identical).
- **Info-loss risk:** none for the report (identical copy on main). The branch's superseded reward-window code
  (`rl/inc8_reward.py` band-pass/fix-driven variants) is intentionally abandoned and its lesson ('weights
  NO-GO → architecture') is banked in `MEMORY.md:8`.

### [LOW] Residual stale fragment inside the corrected MEMORY.md gate-4 line (on main)
- **Location:** `memory/MEMORY.md:8` (within the GATE-4 BAR CORRECTED bullet).
- **Issue type:** stale · **Severity:** low
- **Evidence:** The same corrected line that RETRACTS the near-field pivot still ends with a residual
  pre-correction clause: `σ_p0 = per-fix σ_lat ... 2nd lever = near-field GATE estimator (lower the 12 m
  floor). r=0.38 NEVER; 'fix-rate ≥0.50 through the last 6 m' GEOMETRICALLY UNACHIEVABLE (the 12 m PnP
  floor).` This reasserts the near-field-estimator lever the earlier part of the SAME bullet just RETRACTED
  (`the pivot-off-RL to a near-field-gate estimator recommendation is RETRACTED; next lever = RL to lift
  reach/pass-rate; vision = SUPPORT`).
- **Recommended action:** Reconcile the tail of `MEMORY.md:8` with its own correction — demote the 'near-field
  GATE estimator' from '2nd lever' to a SUPPORT/optional note (or move the 12 m PnP-floor detail down to
  `[[index-vision-estimator]]` as a vision-support item), so the line consistently states RL-reach/pass-rate
  is the primary lever. Pre-existing main content (touched by `511e85c`) — flag for the commander's
  hand-maintained index, not a branch action.
- **Info-loss risk:** none if reconciled in place — the 12 m PnP-floor fact and r=0.38 geometry are retained,
  just re-framed as vision-support rather than the primary pivot. Do not delete the geometry facts.

---

## GAP (med → low)

### [MED] P1 ESKF boresight/attitude bias-state design + observability sims — UNMERGED unique design
- **Location:** `origin/p1-eskf-design` : `handoff/p1-vision-accuracy-2026-06-14/ESKF_DESIGN.md` +
  `scratch-eskf/` (`eskf_geometry.py`, `eskf_jacobian_check.py`, `eskf_observability.py`, results json/txt) —
  1 commit `c7059a9`, 1334 lines; NOT on main (main has a DIFFERENT `ESKF_RESCOPE_DESIGN.md` +
  `scratch-eskf-rescope/` for accel-bias, `524d4b2`).
- **Issue type:** gap · **Severity:** med
- **Evidence:** This is a SEPARATE design from main's merged one. `origin/p1-eskf-design` carries the
  boresight/attitude-bias OBSERVABILITY verdict: `at the realized L3 fix distribution the boresight bias is
  NOT observable from the terminal gate-4 window or any single gate (posterior σ_β ≈ 4.6–7.5°) ... observable
  only by pooling fixes across the whole multi-gate lap ... ESKF-alone needs roughly a full lap-plus (≈100+
  fixes) to reach ≤0.6° ... Therefore the design is static-calibration-PRIMARY, ESKF-secondary`. Main memory
  captures the QUALITATIVE 'ESKF attitude-bias is CALIBRATABLE / co-equal lever'
  (`index_vision_estimator.md:83`, `index_rl_training.md:63`) but the QUANTITATIVE observability verdict
  (σ_β 4.6–7.5° terminal, 100+ fixes to budget, static-PRIMARY ordering, `H_beta=-skew(L)R_wc` validated
  0.49% @ 0.56°) does NOT appear in main memory.
- **Recommended action:** Bank the observability verdict to memory (`index_vision_estimator.md` §ESKF or a new
  bullet): `P1-ESKF observability (c7059a9, origin/p1-eskf-design, NOT merged): boresight bias σ_β 4.6–7.5° at
  terminal/single gate → static-calib-PRIMARY, ESKF-secondary (drift-track only); needs ~full lap (100+ fixes)
  for ESKF-alone to reach 0.6°; H_beta=-skew(L)R_wc validated.` THEN keep the origin branch as the design
  artifact store (do NOT prune).
- **Info-loss risk:** MED if pruned — the quantitative observability sims (`eskf_observability.py` + results)
  and the static-PRIMARY/ESKF-secondary architectural decision are unique to this branch and would directly
  inform whether/how to build the attitude-bias ESKF (a co-equal inc8 lever per memory). Preserve by banking
  the verdict (above) and retaining the origin branch.

### [MED] VQ2 stack-audit findings — live-bug forensics + STAGED dataset-split fix on unmerged branch
- **Location:** `origin/claude/vq2-cleandata-2026-06-19` : `handoff/vq2-stack-audit-2026-06-18/
  AUDIT_FINDINGS.md` (336 lines) + `AUDIT_BRIEF.md` + `repro_bugA_split.py`; also handoff/vq2-cleandata
  commander report (`daef178`); 38 commits; NOT on main (verified absent).
- **Issue type:** gap · **Severity:** med
- **Evidence:** AUDIT_FINDINGS.md documents live bugs:
  - `1. 🐞 The champion 8-kpt model cannot be deployed through the flight stack ... observations_from_keypoints hard-raises ValueError on an 8-keypoint model, and fly_vq1.py still defaults to the legacy 4-kpt curriculum_v2. Confirmed. (F-DEP-1)`
  - `2. 🐞 The dataset split bug (Bug A) is still live ... cluster/vq2_pose_dataset.py shuffles every set with one shared RNG ... Fix staged on this branch (per-set seeding). (F-SPLIT-1)`

  The 8-kpt deploy block IS banked qualitatively (`main:MEMORY.md:6` carry-forward (a) `8-kpt best.pt NOT
  staged = binding-CPZ`; `project-fullstack-burn.md:31` `detector.py:33 hardcodes N_CORNERS=4 against the
  merged 8-kpt VQ2 fork`). The clean-ensemble adoption is banked (`MEMORY.md:15`, `282abb9`). But F-SPLIT-1
  (the shared-RNG dataset split bug + the STAGED fix on this branch) and the structured F-DEP-1..F-* audit
  register are NOT in main memory.
- **Recommended action:** Bank to `project_parked_backlog.md` (or `index_vision_estimator.md`): the F-SPLIT-1
  dataset split-RNG bug (`cluster/vq2_pose_dataset.py` shared `sorted()` RNG reshuffles val membership; fix
  STAGED on `origin/claude/vq2-cleandata-2026-06-19` as per-set seeding) + a pointer to AUDIT_FINDINGS.md.
  Decide whether to selectively-merge the staged split-bug fix (a real correctness bug in the retrain
  pipeline). Keep the origin branch until the staged fix is merged or explicitly declined.
- **Info-loss risk:** MED if pruned — the staged split-bug fix (`repro_bugA_split.py` + the per-set-seeding
  patch) and the 336-line audit register are unique to this branch; losing them re-opens a confirmed
  data-pipeline correctness bug. Preserve by banking F-SPLIT-1 + retaining the branch until the fix is
  merged/declined.

### [LOW] reverent-shirley (inc8 GPU smoke #1 report) — verdict banked, long-form report unmerged
- **Location:** branch `claude/reverent-shirley-b94605` (tip `e08a19b`);
  `handoff/inc8-gpu-smoke-2026-06-14/REPORT.md` (NOT on main — dir absent).
- **Issue type:** gap · **Severity:** low
- **Evidence:** `git ls-tree main -- handoff/inc8-gpu-smoke-2026-06-14/REPORT.md` count=0 (file not on main).
  BUT the finding is fully banked in main memory: `index_rl_training.md:123` = `GPU SMOKE #1 = INFRA-GREEN /
  GATE-RED (job 3273374 ... report handoff/inc8-gpu-smoke-2026-06-14, branch claude/reverent-shirley-b94605
  e08a19b): torch port TRAINS END-TO-END ... RED is HARNESS, NOT training`. Also superseded by the re-smoke at
  `index_rl_training.md:104` (job 3273400, util root-caused to ~11 `.item()` host-syncs).
- **Recommended action:** Prune-safe but verify-first. The REPORT.md prose is the only copy of the detailed
  smoke#1 write-up, but its substance (infra-GREEN/gate-RED, RC=1 cosmetic ONNX, 2 sbatch bugs, util 25.8%,
  pointing-flat-INCONCLUSIVE, NO-GO-on-L0) is already banked at `index_rl_training.md:123` and superseded at
  `:104`. Either prune the branch (memory captures the verdict) OR, if the raw report is wanted as an
  artifact, cherry-pick the single REPORT.md onto main before pruning. Recommend prune (memory is the SSOT and
  explicitly cites the branch+commit).
- **Info-loss risk:** low — verdict and root-cause preserved in `index_rl_training.md:123` (and `:104`). Only
  the long-form report body is unique; if it has reference value, cherry-pick the single REPORT.md to main
  first.

### [LOW] δ_map vertical discriminator report — unmerged unique 208-line report (finding banked)
- **Location:** `worktree-agent-a88bc717008e79cd9` (== origin twin, both @ `d7c592e`) :
  `handoff/dmap-vert-discriminator-2026-06-14/REPORT.md` (1 commit, 208 lines; NOT on main — verified absent).
- **Issue type:** gap · **Severity:** low
- **Evidence:** `git rev-list --count main..worktree-agent-a88bc717008e79cd9` = 1; sole content =
  `handoff/dmap-vert-discriminator-2026-06-14/REPORT.md` (208 lines, NOT on main). The FINDING is thoroughly
  banked: `main:memory/index_vision_estimator.md:181` has a full §δ_MAP VERTICAL DISCRIMINATOR section,
  `MEMORY.md:52/55` carry the verdict (δ_map_vert≈0 → ε_vert≈0.215 m perception bias → L3 CLOSE REFUTED), and
  memory explicitly notes `branch worktree-agent-a88bc717008e79cd9, commit d7c592e, NOT merged`.
- **Recommended action:** Keep the branch/worktree as the artifact store (already memory-sanctioned at
  `index_strategy_meta.md:57`). No merge required. If decluttering branches, the report could be consolidated
  onto main as a handoff file, but the analytical conclusion is fully banked so this is optional.
- **Info-loss risk:** low — the 208-line forensic detail (per-gate δ_map_vert numbers +0.044 g2 / −0.067 g4,
  the two reconciled prior map-offset errors, the escape-hatch) lives only in this report; the verdict + key
  numbers are in memory. Preserve by keeping the origin/worktree branch (do not prune).

---

## PRUNE (med → low)

### [MED] inc8-deterministic-retrain — branch prune status (code valuable + already on main)
- **Location:** branch `inc8-deterministic-retrain` (ahead=6, behind=7 vs main).
- **Issue type:** prune · **Severity:** med
- **Evidence:** All 10 files in `git diff --stat main...inc8-deterministic-retrain` are byte-identical to
  main: `rl/inc8_noise_anneal.py`, `rl/inc8_snapshots.py`, `rl/inc8_select_ckpt.sbatch`,
  `rl/peregrine_inc8_detstab.sbatch`, `rl/peregrine_train_inc8.py`, `tests/test_inc8_noise_anneal.py`,
  `tests/test_inc8_snapshots.py` ALL IDENTICAL to main; handoff `REPORT.md`/`monitor.sh`/`probe_logstd.py` ALL
  ON-MAIN. The CODE (noise-anneal/std-cap lever) IS worth keeping and IS already merged (via `42ddb0b`). The
  branch is behind=7 (missing main's later detector ensemble + estimator_obs obs20 + the `511e85c`
  correction).
  (NOTE: this is the same local branch flagged STALE above for the on-main report; here it is the prune
  classification of the redundant ref.)
- **Recommended action:** Prune branch `inc8-deterministic-retrain` — all unique content (code + report)
  verified on main and the branch is behind on the gate-4 correction. The CODE is valuable and retained on
  main; only the redundant branch ref is removed. (Separately handle the on-main stale REPORT.md per the HIGH
  contradiction above.)
- **Info-loss risk:** none — every branch file verified byte-identical to or absent-newer-on main.

### [LOW] stoic-cray (P1 ESKF re-scope design) — all 9 unique files byte-identical on main
- **Location:** branch `claude/stoic-cray-33a854` (tip `dbcf6a1`); files under
  `handoff/p1-vision-accuracy-2026-06-14/ESKF_RESCOPE_DESIGN.md` + `scratch-eskf-rescope/*`.
- **Issue type:** prune · **Severity:** low
- **Evidence:** `git diff --stat main...stoic-cray` lists 9 files (1104 insertions) but per-file blob compare
  `git rev-parse stoic-cray:<f>` == `git rev-parse main:<f>` returns SAME for all 9 (`ESKF_RESCOPE_DESIGN.md`
  and the 8 `scratch-eskf-rescope` files are byte-identical on main under the same path). The reverse two-dot
  `git diff main..stoic-cray` shows 62,735 deletions = main is far ahead of this old branch tip.
- **Recommended action:** Prune branch `claude/stoic-cray-33a854` — all 9 unique files are byte-identical on
  main at `handoff/p1-vision-accuracy-2026-06-14/` (merged via `c740743` P1 vision-accuracy bundle). Content
  on main, verified by identical blob SHAs.
- **Info-loss risk:** none — the ESKF re-scope design doc and scratch sims are byte-identical on main; nothing
  lost.

### [LOW] Empty/merged local memory+work branches (9 of 13) — strict ancestors of main
- **Location:** branches `claude/admiring-leavitt-7c331e`, `claude/festive-mclean-a79c65`,
  `claude/frosty-sanderson-50061e`, `claude/funny-nash-d07d7b`, `claude/interesting-bohr-639309`,
  `claude/jovial-gagarin-65931d`, `claude/laughing-turing-02eabb`, `claude/musing-lichterman-4410c2`,
  `claude/quizzical-davinci-fd2913`.
- **Issue type:** prune · **Severity:** low
- **Evidence:** For all 9: `git rev-list --count main..<b>` = 0 and `git merge-base --is-ancestor <b> main` =
  YES. Tips are exact ancestors of main (e.g. funny-nash tip `c5d60a0` = main HEAD byte-for-byte;
  festive-mclean/frosty-sanderson both tip `dddf5d9`, behind 23; admiring-leavitt tip `67334f1`, behind 147).
  `git diff --stat main...<b>` empty for every one.
- **Recommended action:** Prune all 9 branches — each is a strict ancestor of main with 0 unique commits;
  content fully on main, verified by `git rev-list --count main..<b>`=0 and is-ancestor=YES.
- **Info-loss risk:** none — every commit on these branches is reachable from main.

### [LOW] p2-inc8-warmstart — branch prune status
- **Location:** branch `p2-inc8-warmstart` (ahead=1, behind=55 vs main).
- **Issue type:** prune · **Severity:** low
- **Evidence:** Unique commit `5e4deed` adds `rl/inc8_warmstart.py` + `tests/test_inc8_warmstart.py` + a
  `peregrine_train_inc8.py` hook + handoff REPORT. `rl/inc8_warmstart.py` IDENTICAL to main;
  `tests/test_inc8_warmstart.py` IDENTICAL. The only differing file, `rl/peregrine_train_inc8.py`, differs
  because MAIN is the SUPERSET — main's version retains the warmstart import/call AND adds the later
  noise-anneal+snapshot wiring (lines the branch lacks). Branch carries no content main lacks.
- **Recommended action:** Prune branch `p2-inc8-warmstart` — warmstart code + test verified on main; the
  train_inc8 delta is main being newer (warmstart + noise-anneal both present on main).
- **Info-loss risk:** none — warmstart feature fully merged; main strictly newer.

### [LOW] p2-substrate-diagnose — branch prune status
- **Location:** branch `p2-substrate-diagnose` (ahead=1, behind=71 vs main).
- **Issue type:** prune · **Severity:** low
- **Evidence:** Unique commit `fa96369` adds `scripts/diagnose_session.py`, `scripts/verify_bundle.py`,
  `tests/test_diagnose_session.py` — all three verified IDENTICAL to main. No branch file differs-from or is
  absent-on main.
- **Recommended action:** Prune branch `p2-substrate-diagnose` — all three files byte-identical to main.
- **Info-loss risk:** none — telemetry auto-diagnoser fully merged.
  (NOTE — cross-agent: the worktree-clutter finding observed this branch as ahead=1 carrying
  `diagnose_session.py`/`verify_bundle.py` "worth a merge-review"; the blob-identical compare here confirms it
  is in fact already ON main, so it is prune-safe. If any doubt, merge-review before prune.)

### [LOW] p2-substrate-greengate — branch prune status (stale baseline)
- **Location:** branch `p2-substrate-greengate` (ahead=1, behind=71); `scripts/green_gate.py:67`.
- **Issue type:** prune · **Severity:** low
- **Evidence:** Unique commit `a7ef9e3`'s only file `scripts/green_gate.py` is OLDER than main: branch has
  `DEFAULT_BASELINE = 884`, main has `DEFAULT_BASELINE = 933`. Diff main→branch shows the branch reverting the
  sentinel from 933 to 884 and dropping the 'diagnose + warm-start merged' provenance comment — i.e. main is
  the newer superset. The diff-scoped green-gate feature is fully present on main.
- **Recommended action:** Prune branch `p2-substrate-greengate` — green_gate.py feature merged; branch holds
  only the stale 884 baseline (main's 933 is current).
- **Info-loss risk:** none — main's green_gate.py is strictly newer (higher sentinel, fuller provenance).

### [LOW] p2-system-id — branch prune status (see also STALE entry for the gap-test)
- **Location:** branch `p2-system-id` (ahead=10, behind=23).
- **Issue type:** prune · **Severity:** low
- **Evidence:** All 9 sysid test files + `handoff/system-id-2026-06-18/REPORT.md` + `wf_investigate.js` are
  IDENTICAL to main; the lone differing `tests/test_sysid_production_obs.py` is the stale gap-pin (see STALE
  entry — main's gap-closed version supersedes it). No unique unmerged content.
- **Recommended action:** Prune branch `p2-system-id` — all system-id registration content merged.
- **Info-loss risk:** none — gap-closed test on main supersedes the branch's gap-open pin.

### [LOW] Fully-merged local branches (tips are ancestors of main) — bulk prune (12)
- **Location:** branches `data-staging-2026-06-17`, `p2-eval-lookat`, `p2-eval-pitch`, `p2-inc8-recenter`,
  `p2-inc8-rl`, `p2-inc8-rl-harvest`, `p2-inc8-sigmap0-eval`, `p2-sigmap0-torch`, `p2-spike-vertical`,
  `p1-calib-v2`, `calib-v2-apply2`, `oneoff-regsuite`.
- **Issue type:** prune · **Severity:** low
- **Evidence:** Each has ahead=0 vs main and `git merge-base --is-ancestor <branch> main` returns true for all
  12 (MERGED, ancestor of main). Their entire history is on main.
- **Recommended action:** Prune all 12 branches — each tip is a direct ancestor of main (zero unique commits).
- **Info-loss risk:** none — every commit is reachable from main.

### [LOW] Fully-merged origin branches with zero unique content (3)
- **Location:** `origin/p1-audit-extrinsics`, `origin/worker/at-speed-sigma-2026-06-15`,
  `origin/accept-geometry-2026-06-15` (all 0 commits ahead, verified `git merge-base --is-ancestor` = YES).
- **Issue type:** prune · **Severity:** low
- **Evidence:** `git rev-list --count main..<branch>` = 0 and `git merge-base --is-ancestor <branch> main` =
  YES for all three. `p1-audit-extrinsics` (boresight static-calib, the parallel worker to p1-eskf-design),
  `at-speed-sigma` (tip `2f53ae1` = ancestor of main), `accept-geometry` (ancestor of main) — all content is
  on main.
- **Recommended action:** Prune these three origin branches — all unique content verified on main (0 commits
  ahead, tips are ancestors of main).
- **Info-loss risk:** none — verified by `git merge-base --is-ancestor`: every commit on these branches is
  reachable from main.

### [LOW] Assigned worktree-* local label-branches — empty/merged (3, plus 1 KEEP)
- **Location:** `worktree-agent-a2e0b926c2b864668` (@ `ea714234`, 0 ahead), `worktree-agent-a5e34977321fd2a8e`
  (@ `ea714234`, 0 ahead), `worktree-wf_43064dee-ce7-1` (0 ahead). NOTE
  `worktree-agent-a88bc717008e79cd9` is the ONLY one with unique content (covered in the δ_map GAP entry,
  `d7c592e`).
- **Issue type:** prune · **Severity:** low
- **Evidence:** `git rev-list --count main..worktree-agent-a2e0b926c2b864668` = 0; same for
  `a5e34977321fd2a8e` and `worktree-wf_43064dee-ce7-1`. All three resolve to commits already on main. (The
  worktree DIRECTORIES `.claude/worktrees/agent-a2e0.../agent-a5e3...` are checked out to DIFFERENT branches
  `p2-substrate-diagnose`/`p2-substrate-greengate` — those substrate branches carry their own commits
  `fa96369`/`a7ef9e3`, handled in their own prune entries above.)
- **Recommended action:** Prune the three empty label-branches `worktree-agent-a2e0b926c2b864668`,
  `worktree-agent-a5e34977321fd2a8e`, `worktree-wf_43064dee-ce7-1` (0 ahead of main). KEEP
  `worktree-agent-a88bc717008e79cd9` (carries `d7c592e` δ_map report, NOT merged). Do NOT touch the
  `.claude/worktrees/*` directories themselves (harness-managed — see presentability entry).
- **Info-loss risk:** none for the three empty branches (0 commits ahead). `a88bc717` retained per the δ_map
  finding.

### [LOW] Origin VQ2/blender artifact branches — code merged, only reports + scratch data unique
- **Location:** `origin/shadowpc-artifact-push` (14 commits; superset of `origin/vq2-blender-render-2026-06-15`
  [13 commits, confirmed ancestor]); `origin/vq2-data` (orphan, 6003 files, no merge-base).
- **Issue type:** prune · **Severity:** low
- **Evidence:** `git merge-base --is-ancestor vq2-blender-render shadowpc-artifact-push` = YES (shadowpc is the
  superset). The `blender_gen` photoreal CODE (`bpy_photoreal.py`, `assets.py`, `masks.py`, etc.) is IDENTICAL
  to main (`git diff --stat main shadowpc -- src/racer/vision/blender_gen/` empty; merged as `d64e1dc` per
  `project_parked_backlog.md:33`). Unique-on-branch = handoff REPORTs (`shadowpc-artifact-push-2026-06-17/
  REPORT.md` artifact-pipe doc; `vq2-data-transfer-2026-06-16/REPORT.md`) + scratch label/png sample files.
  `origin/vq2-data` = 6003-file orphan dataset branch (intentional per artifact-pipe doctrine; MEMORY
  'binaries ship as GitHub release assets, NEVER git-add').
- **Recommended action:** `origin/vq2-blender-render-2026-06-15` is fully contained in
  `origin/shadowpc-artifact-push` → consolidate (archive vq2-blender-render, keep shadowpc-artifact-push as
  the artifact-pipe record). Keep `origin/vq2-data` as the sanctioned orphan dataset branch (do NOT merge into
  main — it has no merge-base by design). Bank the shadowpc-artifact-push REPORT.md artifact-pipe recipe
  pointer if not already in memory (artifact-pipe is in `MEMORY.md`/`project-fullstack-burn.md`).
- **Info-loss risk:** low — code is on main; the unique reports are handoff records (artifact-pipe recipe is
  banked in MEMORY artifact-pipe doctrine). Preserve the dataset by keeping `origin/vq2-data` (orphan) and
  `shadowpc-artifact-push` (superset report). Do not delete either; consolidate vq2-blender-render into
  shadowpc only.

### [LOW] VQ2 ensemble-support — code merged, report-only unique branch
- **Location:** `origin/vq2-ensemble-support` : `handoff/vq2-ensemble-support-2026-06-19/REPORT.md` +
  `measure_latency.py` (1 commit `05368ce`; `src/racer/vision/detector.py` + `tests/test_ensemble_detector.py`
  IDENTICAL to main).
- **Issue type:** prune · **Severity:** low
- **Evidence:** `git diff --stat main origin/vq2-ensemble-support -- src/racer/vision/detector.py
  tests/test_ensemble_detector.py` empty (EnsembleGateDetector + test byte-identical on main, merged as
  `06876bc` per `MEMORY.md:15` 'Deploy STEP-1 DONE (06876bc): opt-in EnsembleGateDetector'). Only unique
  content = `handoff/vq2-ensemble-support-2026-06-19/REPORT.md` + `measure_latency.py` (NOT on main).
- **Recommended action:** Consolidate: load-bearing detector code is on main (`06876bc`). Archive
  `origin/vq2-ensemble-support` after confirming the REPORT.md (p95 43.7 ms latency, union+per-gate dedup
  design) is banked — `MEMORY.md:15` already captures the headline. Optionally copy
  REPORT.md/measure_latency.py to a handoff dir on main if the latency-measurement script is wanted as a tool.
- **Info-loss risk:** low — code merged (`06876bc`); the latency number (p95 43.7 ms) and design are banked in
  `MEMORY.md:15`. Preserve `measure_latency.py` by keeping the origin branch until confirmed unneeded as a
  standing tool.

### [LOW] P3 boresight FORM-resolution branch — reports/scripts merged, only raw JSON row-dumps unique
- **Location:** `origin/claude/hardcore-lehmann-1de77c` (2 commits `b064b3c`, `29452b6`); unique-not-on-main =
  `handoff/p3-simops-empirical-2026-06-14/analysis/b1_g0_rows.json` (~170k lines), `b2_g0_rows.json` (~338k
  lines).
- **Issue type:** prune · **Severity:** low
- **Evidence:** All report/script files (REPORT.md, FORM_RESOLUTION.md, `boresight_form.py`,
  `headon_bothflip.py`, etc.) are IDENTICAL to main (`git diff --stat` empty). Only the two large raw
  rows.json captures are unique. The FORM=METRIC conclusion is banked:
  `main:memory/index_vision_estimator.md:251` `FORM = METRIC (range-independent); angular REFUTED ... ε_vert =
  −0.25 m ... FLAT across the 14–26 m band`.
- **Recommended action:** Consolidate/archive `origin/claude/hardcore-lehmann-1de77c` — its analytical content
  (FORM=METRIC, the discriminator scripts) is fully on main and the verdict is banked
  (`index_vision_estimator.md:251-254`). The two raw rows.json are large data captures; keep the origin branch
  as their store OR move them to the artifact-pipe (GitHub release) if disk matters. Do NOT delete the
  rows.json without the artifact-pipe copy.
- **Info-loss risk:** low — scripts/reports on main, verdict banked. The two rows.json (raw boresight head-on
  captures, reproducible-only-by-re-recording) survive on the origin branch; preserve by keeping the branch or
  pushing the JSONs to the artifact-pipe before any prune.

### [LOW] p2-substrate-diagnose / p2-substrate-greengate — cross-listed under worktree clutter (AHEAD, but on main)
- **Location:** worktrees `.claude/worktrees/agent-a2e0.../p2-substrate-diagnose` and
  `.../agent-a5e3.../p2-substrate-greengate`.
- **Issue type:** prune · **Severity:** low
- **Evidence:** The worktree-clutter agent saw these as AHEAD=1 carrying unmerged tooling
  (`diagnose_session.py`+`verify_bundle.py` 974 lines; diff-scoped `green_gate.py` 483 lines) and flagged them
  "worth a merge-review." The dedicated branch-compare agent verified the files are in fact byte-IDENTICAL on
  main (diagnose) / strictly OLDER on the branch (greengate 884<933) — i.e. already merged / superseded. The
  AHEAD=1 is the redundant commit, not unmerged content.
- **Recommended action:** Treat as the same prune actions as the `p2-substrate-diagnose` /
  `p2-substrate-greengate` entries above. If any residual doubt remains for the commander, a 30-second
  blob-SHA compare resolves it before prune. (These two branches were outside the worktree-agent's assignment,
  hence the "flag UP" note; the named-branches agent owns the verdict = prune-safe.)
- **Info-loss risk:** none — blob-identical (diagnose) / strictly-newer-on-main (greengate) confirmed.

---

## PRESENTABILITY (low)

### [LOW] Worktree clutter — 18 worktrees, all harness-managed except the stale sibling checkout
- **Location:** `git worktree list`: 16 dirs under
  `C:/Users/Fengy/Downloads/Projects/Anduril/.claude/worktrees/*` + the primary
  `C:/Users/Fengy/Downloads/Projects/Anduril` (stale pre-correction sibling) + the audit root
  `C:/Users/Fengy/Downloads/Projects/Anduril-cmdr`.
- **Issue type:** presentability · **Severity:** low
- **Evidence:** All 16 worktree dirs are under `.claude/worktrees/*` (HARNESS-MANAGED — per memory 'NEVER
  manually prune it (racy); only `git worktree prune` (gone dirs)'). Merge status of their branches:
  - MERGED / 0-ahead (prunable-on-merge, NOTE-only): `admiring-leavitt-7c331e`, `funny-nash-d07d7b`
    (@ `c5d60a0`==main), `interesting-bohr-639309`, `jovial-gagarin-65931d`, `p2-inc8-rl-harvest`,
    `p2-inc8-rl`, `p1-calib-v2`, `calib-v2-apply2`, `oneoff-regsuite`, `wf_43064dee-ce7-1`.
  - AHEAD (carry unique content, do NOT prune the worktree): `agent-a2e0.../p2-substrate-diagnose` (1),
    `agent-a5e3.../p2-substrate-greengate` (1), `festive-mclean/p2-system-id` (10), `p2-inc8-warmstart` (1),
    `reverent-shirley-b94605` (1), `stoic-cray-33a854` (1). (Branch-level prune verdicts for these are in the
    PRUNE/GAP sections; the worktrees themselves are harness-managed.)
  - The non-`.claude` worktree `C:/Users/Fengy/Downloads/Projects/Anduril` (`inc8-deterministic-retrain`) =
    the STALE pre-correction sibling the task warns to not read from (Fengyou's working tree).
- **Recommended action:** NOTE-only (read-only audit): do NOT manually prune any `.claude/worktrees/*`
  (harness-managed, recycles). Let the harness reclaim merged ones; only run `git worktree prune` for gone
  dirs. The substrate-branch tooling is unmerged-looking but blob-confirmed already on main (see PRUNE
  entries) — no merge-review actually needed, just prune the redundant branch refs. Leave the
  `C:/Users/Fengy/Downloads/Projects/Anduril` sibling (Fengyou's working tree).
- **Info-loss risk:** none — NOTE-only classification; no prune recommended for harness-managed worktrees.

---

## Cross-agent dedup notes

- The `inc8-deterministic-retrain` local branch + its on-main REPORT.md were reported by BOTH
  `git:named-branches` and `git:worktree+origin` (one as the on-main contradiction, one as the branch
  stale/prune status). Merged here: the **on-main report** is the HIGH contradiction (annotate, do not delete);
  the **branch ref** is a MED prune (code already on main via `42ddb0b`).
- The inc8 deterministic-retrain HIGH contradiction was reported by both `git:named-branches` and
  `git:worktree+origin` with near-identical evidence; merged into one HIGH entry preserving both line-number
  sets (HEADLINE ~88/89, CONCLUSION ~113–135, MEMORY-DELTA ~124–150).
- `p2-substrate-diagnose` / `p2-substrate-greengate`: `git:worktree+origin` flagged them as AHEAD "worth a
  merge-review"; `git:named-branches` blob-confirmed they are already on main / strictly older. Both
  perspectives preserved (the prune entries + the cross-listed worktree-clutter entry) so the commander sees
  the apparent-ahead-but-actually-merged reconciliation.
- This auditor independently CONFIRMED the two HIGH on-main contradictions by reading
  `handoff/inc8-deterministic-retrain-2026-06-19/REPORT.md` and `memory/MEMORY.md:8` in the audit root —
  quotes match verbatim.
