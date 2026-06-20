# Cleanup Audit Appendix — Code (src / rl / cluster / tests / scripts)

**Area state (3-line summary).** The code is structurally healthy — no orphaned tests, no
unconditional skips, no leaked "retracted pivot off RL / near-field estimator / fix-seating
wall" language in source, and the look-at-primitive "architecture pivot" tokens are CURRENT
(not the retracted one). The dominant real defect is a single cross-cutting STALE/CONTRADICTION
cluster: the **pre-2026-06-19 gate-4 `sigma_p0_lat <= 0.08 m` GO bar** (a double-counted clearance,
corrected to `~0.15 m` / p99 `~0.45` on main @ 511e85c) is still wired into FIVE rl eval tools +
sbatch as the default and prose success criterion, and would brand a marginal-PASSING policy NO-GO.
Two other high-severity items: a **VQ2 sbatch `--batch 32`** that the trainer itself flags as not
reproducing the champion, and a **green_gate test-count sentinel floor (933)** that now lags the
real full-stack collect count (1091) by ~158 tests. The remainder are low-severity stale doc
pointers, archive-this-lineage prune candidates, and benign duplication with cross-reference comments.

**Dedup note.** Several findings were reported by more than one agent (e.g. the 0.08 bar in
`inc8_sigmap0_torch_eval.py` was filed by both `code:rl` and `code:stalegrep`; the green_gate
baseline by both `code:cluster+scripts` and `code:tests+greengate`). These are merged into single
entries below, preserving every distinct location, line number, and piece of evidence. The
highest severity reported by any agent is used for the merged entry.

---

## CONTRADICTION

### [HIGH] inc8 gate-4 `sigma_p0` GO-gate bar — TORCH instrument
- **Location:** `rl/inc8_sigmap0_torch_eval.py:4-5` (module docstring), `:47-48` (GO RULE), `:223` (`_go_verdict` default `sigma_target=0.08`), `:234-235`, `:270`, `:273`, `:292` (argparse `--sigma-target` default `0.08`)
- **issue_type:** contradiction (also filed as `stale` by `code:stalegrep`) · **severity:** high
- **Evidence (exact quotes):**
  - L4-5: `The inc8 GO gate is the PHYSICAL terminal-centering spread at the binding gate-4 plane: ``sigma_p0_lat <= ~0.08 m`` (miss p99 ~= 3*sigma_p0).`
  - L47-48: `GO RULE.  sigma_p0_lat <= 0.08 m AND |lateral miss| p99 (~3*sigma_p0) <= ~0.24 m ... (a low reach-rate / tiny ensemble is itself a NO-GO/NO-DATA signal).`
  - L223: `def _go_verdict(m: dict, sigma_target: float = 0.08, min_n: int = MIN_ENSEMBLE_DEFAULT) -> tuple[str, str]:`
  - L233/234-235: token = `'GO' if (ok_sigma and ok_p99) else 'NO-GO'`
  - L292: `ap.add_argument("--sigma-target", type=float, default=0.08)`
  - CONTRADICTS the correction landed on main @ 511e85c (2026-06-19, also in MEMORY.md "GATE-4 BAR CORRECTED"): the 0.08 m closure bar was a DOUBLE-COUNT (W_EFF subtracted the drone twice); real gate clearance ~0.37-0.47 m, real bar `sigma_p0_lat ~0.15` (p99 ~0.45); measured 0.15-0.20 = MARGINAL-PASSING, NOT a NO-GO. git confirms this tool (cc86d38, 2026-06-18) predates the correction and was never updated.
  - This is the **laptop-validated instrument actually run on Adroit** for the GO/NO-GO number, so the stale default is load-bearing — it will brand a marginal-passing policy NO-GO.
- **Recommended action:** Update the docstring GO RULE (L4-5, L47-48) and BOTH `0.08` defaults (`_go_verdict` L223, argparse L292) to `sigma_p0_lat <= ~0.15 m AND lat_p99 <= ~0.45 m`. Add a one-line note: `0.08 was a pre-2026-06-19 double-counted bar (W_EFF subtracted the drone twice); corrected on main @ 511e85c — real clearance ~0.37-0.47 m, real bar ~0.15/p99 ~0.45; gate-4 is now a reach/pass-rate problem, not sub-8cm centering.` Keep the verdict MACHINERY (it is parameterized); only the numeric default + prose are stale. The p99=3*sigma coupling still holds (3*0.15 = 0.45). Keep the numbers IDENTICAL across the torch and numpy tools (the docstring explicitly mirrors the numpy tool).
- **info_loss_risk:** none — the 0.08 lineage + its retraction are fully preserved in `handoff/inc8-deterministic-retrain-2026-06-19/REPORT.md` and `memory/MEMORY.md`; changing the default does not erase history.

### [HIGH] inc8 gate-4 `sigma_p0` GO-gate bar — NUMPY companion instrument
- **Location:** `rl/inc8_sigmap0_eval.py:4-5` (module docstring), `:27-28` (GO RULE), `:174` (`_go_verdict` default `sigma_target=0.08`), `:181-183`, `:199` (argparse `--sigma-target` default `0.08`)
- **issue_type:** contradiction (also filed as `stale` by `code:stalegrep`) · **severity:** high
- **Evidence (exact quotes):**
  - L4-5: `... ``sigma_p0_lat <= ~0.08 m`` (miss p99 ~= 3*sigma_p0).`
  - L27-28: `GO RULE.  sigma_p0_lat <= 0.08 m AND |lateral miss| p99 ~= 3*sigma_p0 <= ~0.24 m ... (a low reach-rate is itself a NO-GO signal ...).`
  - L174: `def _go_verdict(m: dict, sigma_target: float = 0.08) -> str:`
  - L181-183: `verdict = 'GO' if (ok_sigma and ok_p99) else 'NO-GO'`
  - L199: `ap.add_argument("--sigma-target", type=float, default=0.08)`
  - Same pre-correction 0.08 bar as the torch tool; CONTRADICTS the corrected ~0.15 / p99 ~0.45 bar (511e85c). At 0.08 this tool emits a FALSE NO-GO on a marginal-passing policy (measured 0.15-0.20).
- **Recommended action:** Same fix as the torch tool: change the `0.08` defaults to `0.15` (L174, L199), the p99 prose `0.24` to `~0.45` (L5, L27-28), add the same double-count correction note. Keep both tools' defaults in sync.
- **info_loss_risk:** none — old value preserved in the correction note in memory + REPORT.md.

### [HIGH] VQ2 detector training — Adroit sbatch batch size contradicts the trainer's champion-repro warning
- **Location:** `cluster/vq2_pose_train.sbatch:33` vs `cluster/vq2_pose_train.py:57`
- **issue_type:** contradiction · **severity:** high
- **Evidence (exact quotes, VERIFIED on disk):**
  - `cluster/vq2_pose_train.sbatch:33` launches: `--epochs 100 --patience 20 --imgsz 640 --batch 32 --device 0`
  - The trainer it calls warns on the same flag: `ap.add_argument("--batch", type=int, default=16)   # champion = 16 (F-REPRO-1); 32 silently changes the run` (`vq2_pose_train.py:57`)
  - The Adroit job therefore launches the VQ2 detector at the exact batch size the trainer flags as NOT reproducing the champion.
- **Recommended action:** Change `vq2_pose_train.sbatch:33` `--batch 32` to `--batch 16` to match the champion (F-REPRO-1), OR add an inline comment on that sbatch line documenting the deliberate deviation and why it is acceptable. Do not silently leave the two in conflict.
- **info_loss_risk:** none

---

## STALE

### [HIGH] green_gate pre-merge test-count sentinel baseline (933) lags real collect count (1091)
- **Location:** `scripts/green_gate.py:67` (`DEFAULT_BASELINE = 933`); rationale + docstring mentions at `:27-29`, `:40`, `:66`
- **issue_type:** stale · **severity:** high (merged: `code:tests+greengate` filed high, `code:cluster+scripts` filed med — both describe the same `:67` line)
- **Evidence (exact quotes, VERIFIED on disk):**
  - `green_gate.py:67`: `DEFAULT_BASELINE = 933` with docstring `:27`: `count must be >= a baseline (default 933, the honest full-stack count; the long-quoted "723" was stale per the burn survey)`.
  - `:66` comment: `merged). 723 was stale (burn survey); 884 was a stale-base worker measurement. Override with --baseline.`
  - Live measurement under the full stack (py3.13 + torch 2.9.1+cpu + scipy + numpy, all importable): `py -3.13 -m pytest --collect-only -q tests` => `1091 tests collected` (0 collection errors). The sentinel floor (933) now lags the real collected count by 158 tests, so the gate would PASS even if up to 158 tests silently vanished — the exact silent-test-loss regression class the sentinel exists to catch.
  - The 27 newest tests (`test_inc8_noise_anneal.py`=16, `test_inc8_stability_knobs.py`=11) were added in HEAD commit 42ddb0b (2026-06-20), AFTER the 933 baseline was measured (2026-06-17). MEMORY.md independently flags `947 tests collected ... prior 723/884/933 stale ... bump to 947` — confirming 933 is stale-low.
- **Recommended action:** Update `DEFAULT_BASELINE` from `933` to `1091` (freshly measured full-stack collect on main @ c5d60a0) after confirming `pytest --collect-only` reports it; optionally set a slightly conservative `1085` if parametrized-test counts may drift, but 1091 is the honest number and matches the gate's own `>= baseline` semantics. Update the three docstring/comment mentions of `933` and append the `933 -> 1091` bump note; keep the 723/884 lineage as documented history.
- **info_loss_risk:** none — pure floor bump; old value recoverable from git history, rationale comment updated alongside.

### [HIGH] inc8 `sigma_p0` torch LAUNCH sbatch — deliverable definition encodes the 0.08 bar
- **Location:** `rl/inc8_sigmap0_torch.sbatch:18`, `:127` (launch commit subject 36b7730)
- **issue_type:** stale · **severity:** high (merged: `code:stalegrep` filed high, `code:rl` filed med — same two lines)
- **Evidence (exact quotes):**
  - L18: `(2) 2-axis (auto) -- THE DELIVERABLE: full look-at (g_yaw=-3, g_pitch=3), GT-anchored sigma_p0_lat vs the GO rule (sigma_p0_lat <= 0.08 AND lat_p99 <= 0.24).`
  - L127: `Then the 2-axis sigma_p0_lat vs the 0.08 GO rule.`
  - The launch commit subject (36b7730, 2026-06-19 19:48, hours before the 23:53 correction): `definitive gate-4 verdict (NO-GO 0.198, 2-axis NO-DATA)` — the exact NO-GO call the correction retracted (0.198 vs the old 0.08 bar; against the corrected ~0.15 bar it is marginal/near-passing, NOT a definitive NO-GO).
- **Recommended action:** Update L18 and L127 to `sigma_p0_lat <= ~0.15 AND lat_p99 <= ~0.45` (corrected gate-4 bar, main @ 511e85c); optionally add the one-line double-count note. The commit MESSAGE is immutable history (do not rewrite); only the in-file prose must be corrected. No behavior change — these are batch-script comments documenting the wrong success threshold.
- **info_loss_risk:** none — the 0.198 measurement itself stays valid; only its GO/NO-GO interpretation changes (marginal-passing). Memory + the deterministic-retrain REPORT already record the re-judgment.

### [MED] inc8 reward dense-centering objective comment — 0.08 honest-objective number
- **Location:** `rl/inc8_reward.py:47-49`
- **issue_type:** stale · **severity:** med
- **Evidence (exact quotes):** `... so "estim_err <= 0.08" is geometrically unachievable and is NOT the gate. The honest objective is the physical centering miss sigma_p0 <~ 0.08 m (gate-4 closure, r=0.30).` The asserted honest-objective number `0.08 m` is the double-counted bar; corrected gate-4 bar is `~0.15 m` (511e85c). (The surrounding point — estim_err is floored ~0.115-0.13 and is NOT the gate — remains CORRECT and must be kept.)
- **Recommended action:** Update `<~ 0.08 m` to `<~ 0.15 m (gate-4 closure; corrected 2026-06-19 — 0.08 was a double-count)`. Leave the estim_err-vs-sigma_p0 distinction intact.
- **info_loss_risk:** none.

### [MED] inc8 warm-start sbatch — honest-objective definition encodes 0.08 bar
- **Location:** `rl/peregrine_inc8_warmstart.sbatch:47-49` (GO-criterion block lines 43-50; L49 is the number)
- **issue_type:** stale · **severity:** med (merged: `code:stalegrep` med, `code:rl` low — same block)
- **Evidence (exact quotes):** L47-49: `... so "estim_err <= 0.08" is geometrically unachievable and is NOT the gate ... The honest objective is the physical centering miss sigma_p0 <~ 0.08 m (gate-4 closure, r=0.30).` Same stale 0.08 bar; CONTRADICTS the corrected real bar `sigma_p0_lat ~0.15` (p99 ~0.45). (The surrounding estim_err-floor argument remains correct and should be kept.) NOTE: this sbatch is ALSO a superseded run-config (see the PRUNE finding below) — if archived, this is subsumed.
- **Recommended action:** If the warmstart sbatch is retained, update `<~ 0.08 m (gate-4 closure, r=0.30)` to `<~ 0.15 m (gate-4 lateral; corrected bar, main @ 511e85c — 0.08 was a double-counted clearance, real clearance ~0.37-0.47 m)`. Leave the estim_err-floor reasoning + r=0.30 intact. Otherwise subsumed by the prune/consolidate finding.
- **info_loss_risk:** none — r=0.30 and the estim_err-floor reasoning retained; old value preserved in REPORT.md + MEMORY.md.

### [MED] green_gate sentinel rationale comment — "honest full-stack count" narrative
- **Location:** `scripts/green_gate.py:27-29` and `:66`
- **issue_type:** stale · **severity:** med
- **Evidence (exact quotes):** L27-28: `count must be >= a baseline (default 933, the honest full-stack count; the long-quoted "723" was stale per the burn survey)`. L66: `merged). 723 was stale (burn survey); 884 was a stale-base worker measurement. Override with --baseline.` Both assert 933 is "the honest full-stack count", but the honest full-stack count is now 1091. This narrative will mislead the next maintainer into trusting 933.
- **Recommended action:** Update both comments to state the honest full-stack count is 1091 (measured 2026-06-20 on main @ c5d60a0 with the noise-anneal/std-cap tests merged); keep the 723/884 lineage as historical context but mark 933 itself as superseded.
- **info_loss_risk:** none — the 723/884/933 lineage stays as documented history; only the "current honest count" label moves to 1091.

### [MED] Detector deployment-status comment — v2 champion vs VQ2 ensemble
- **Location:** `scripts/eval_detector.py:27-30`
- **issue_type:** stale · **severity:** med
- **Evidence (exact quotes):** `# v2 (multi-gate, 6k) is the chosen deployment detector: best corner + pose error on this / fixed-set eval (2026-06-02). v3's --hard tail-oversampling regressed the bulk...` Per MEMORY (2026-06-19, commit 282abb9) the VQ2 8-keypoint ensemble now BEATS the champion (course 76->82%, deployed 62->82%) and `src/racer/vision/detector.py:24-32` documents an opt-in ensemble as the 2026-06-19 deploy gate (single-model v2 is now only the "VQ1-proven default", not the best detector). The flat assertion that v2 is "the chosen deployment detector / best ... error" reads as current truth but is superseded for VQ2.
- **Recommended action:** Update the comment to note v2 is the VQ1-proven single-model baseline for THIS synthetic fixed-set eval, and that the VQ2 photoreal 8-kpt ensemble (282abb9) is the superseding detector for the photoreal track; point to `detector.py`'s `EnsembleGateDetector`. Do NOT delete the v2-vs-v3 finding (it is the only record of why v3 was rejected).
- **info_loss_risk:** Preserve the v3 `--hard` regression finding (high-roll corner-swap belongs in solver/prior) — it is NOT captured elsewhere in scripts/. Keep it in the rewritten comment.

### [MED] green_gate stack-guard illustrative measurements (884 -> 690) stale
- **Location:** `scripts/green_gate.py:17-18` and `:248`
- **issue_type:** stale · **severity:** low (per `code:tests+greengate`; listed under stale)
- **Evidence (exact quotes):** L17-18: `~5 modules hard-error at collection (measured: 884 -> 690 collected). So a torch-less run is RED`. L248: `(measured 884 -> 690). So torch-less is RED, never a green-with-skips.` The 884 with-stack / 690 without-stack figures predate the current 1091 with-stack count, so the absolute numbers are stale (the ~194-test torch-delta ratio and the RED-on-torch-absent logic remain valid).
- **Recommended action:** Refresh the parenthetical measurements to the current with-stack count (1091) and re-measure the torch-absent collect if a precise delta is wanted; the qualitative claim (torch-less => collection errors => RED) needs no change.
- **info_loss_risk:** none — only the illustrative numbers change; the load-bearing logic and its justification are preserved.

### [MED] Default detector weights path across 4 scripts — non-existent in repo
- **Location:** `scripts/fly_vq1.py:328`; `scripts/eval_detector.py:30`; `scripts/characterize_perception.py:60`; `scripts/task2_gate_pnp.py:38`
- **issue_type:** stale · **severity:** med
- **Evidence (exact quotes):** All four default to `models/gate_yolo11s_curriculum_v2.pt` (e.g. `fly_vq1.py:328` `ap.add_argument("--weights", default="models/gate_yolo11s_curriculum_v2.pt")`; `characterize_perception.py:60` and `task2_gate_pnp.py:38` `WEIGHTS = ROOT / "models/gate_yolo11s_curriculum_v2.pt"`). There is NO `models/` directory in the repo (`*.pt` is gitignored; weights ship as GitHub release assets per MEMORY artifact-pipe), so the default path resolves to nothing on a fresh canonical checkout — each script errors unless the user fetches + places the asset or passes `--weights`.
- **Recommended action:** Either (a) document the artifact-pipe fetch step in each docstring (`gh release download --repo Hat000/Peregrine ...` to materialize `models/gate_yolo11s_curriculum_v2.pt`), or (b) point the default at the conventional release-asset cache path. At minimum add a one-line note that the default weights are a gitignored release asset, not in-tree. None is a delete.
- **info_loss_risk:** none

### [LOW] inc8 sigma_p0 torch sbatch GO-rule prose (low-severity duplicate framing)
- **Location:** `rl/inc8_sigmap0_torch.sbatch:18`, `:127` — SAME lines as the [HIGH] launch-sbatch entry above; recorded by `code:rl` at med severity. Merged into that entry (highest severity wins). No separate action.

### [LOW] Clock/timeline epoch reconciliation TODO markers (LEGITIMATE OPEN WORK — keep)
- **Location:** `src/racer/contracts.py:22-24`, `src/racer/mavlink_client.py:333`, `src/racer/recording.py:21`
- **issue_type:** stale (flagged-as-keep) · **severity:** low
- **Evidence (exact quotes):** `contracts.py:22` `TODO(clock): reconcile the epochs via TIMESYNC once ``msg_audit`` characterises the offsets -- master plan hole #10.` `mavlink_client.py:333` `pass  # TODO(clock): use TIMESYNC to reconcile sim_time_ns across streams.` These are the ONLY TODO/FIXME/HACK/XXX tokens in `src/`. COMPETITION INTEL (memory, 2026-06-14) notes §4.3 scored wire includes TIMESYNC, so the reconciliation is now actionable.
- **Recommended action:** KEEP the `TODO(clock)` markers (legitimate open work, not stale comments). Optionally verify against the parked backlog that "master plan hole #10 / TIMESYNC epoch reconciliation" is still tracked; if `msg_audit` has since characterized the offsets, update or close the TODO. No code change needed now.
- **info_loss_risk:** none — live, correctly-described open item; removal would lose tracked work.

### [LOW] fly_vq1 docstring references to non-existent sibling scripts
- **Location:** `scripts/fly_vq1.py:11` and `:13`
- **issue_type:** stale · **severity:** low
- **Evidence (exact quotes):** L11: `Run ``control_mode_probe.py`` FIRST to learn which interface actually moves the drone`. L13: `ATTITUDE mode needs a calibrated ``--hover-thrust`` (run ``innerloop_step.py``)`. Neither `control_mode_probe.py` nor `innerloop_step.py` exists anywhere in the repo (confirmed: not under scripts/, src/, or handoff/). The two SAFETY-procedure pointers are broken.
- **Recommended action:** Update the two references to the surviving equivalents (hover-thrust calibration now lives in `scripts/rate_sysid.py --mode hover` per its docstring; the control-mode question was resolved — replace `control_mode_probe.py` with current guidance or drop the stale pointer). Verify the replacement before editing.
- **info_loss_risk:** none — the SAFETY guidance text itself stays; only the dead script names change.

### [LOW] run_mapper_offline docstring references non-existent extractor
- **Location:** `scripts/run_mapper_offline.py:15`
- **issue_type:** stale · **severity:** low
- **Evidence (exact quotes):** `The extractor that turns a ``data/runs/<session>`` recording into this file follows the ``export_course_bundle.py`` alignment recipe`. No `export_course_bundle.py` exists outside handoff/ (confirmed absent under scripts/ and src/). The closest live tool is `scripts/export_frame_bundle.py`.
- **Recommended action:** Confirm whether the intended reference is `scripts/export_frame_bundle.py` (the live FPV-bundle exporter) and update the name; if `export_course_bundle.py` was a handoff-only one-off, say so explicitly.
- **info_loss_risk:** none

### [LOW] cluster/push_dir uploader — hardcoded sibling-clone ROOT + stale INCLUDE list
- **Location:** `cluster/push_dir.py:16-24`
- **issue_type:** stale · **severity:** low
- **Evidence (exact quotes):** `ROOT = Path(r"C:\Users\Fengy\Downloads\Projects\Anduril")` (`push_dir.py:16`) hardcodes the SIBLING pre-correction clone, not the canonical Anduril-cmdr. Its INCLUDE list (lines 17-24) ships only `src/racer`, `scripts/gen_synthetic_dataset.py`, and the four `cluster/yolo_*` files (the OLD synthetic-YOLO detector infra); it does NOT include the current VQ2 pipeline (`cluster/vq2_pose_dataset.py` / `vq2_pose_train.py` / `vq2_precision_loss.py` / `vq2_pose_train.sbatch`). Last touched 2026-06-02 (fc29287), before the VQ2 work — the uploader is stale for the current detector pipeline.
- **Recommended action:** Update ROOT to the canonical repo path (or make it a CLI arg / repo-root-relative), and extend INCLUDE with the `cluster/vq2_pose_*` files now that VQ2 is the live detector pipeline. Keep the chunked-base64 transfer mechanism (the documented no-token Adroit path).
- **info_loss_risk:** none

### [LOW] TOGT gen_cases references the superseded CLI as if current
- **Location:** `scripts/togt/gen_cases.py:4-6` vs `scripts/togt/README.md:71` and `scripts/togt/traj_planner_peregrine.cpp:10-12`
- **issue_type:** stale · **severity:** low
- **Evidence (exact quotes):** `gen_cases.py:4-6` says the case dir is `consumed by the patched ``planners`` CLI (scripts/togt/traj_planner_peregrine.cpp) and then by the multiple-shooting refine`. But `README.md:71` states `traj_planner_peregrine.cpp -- kept for reference; superseded by the gtest driver` and `traj_planner_peregrine.cpp:10-12` documents the standalone CLI segfaults before main() so the TOGT phase actually runs via `test_peregrine.cpp` (a gtest case), as `run_cases.sh:37-41` confirms. The gen_cases docstring names the dead CLI as the consumer.
- **Recommended action:** Update `gen_cases.py:4-6` to point at the gtest driver (`scripts/togt/test_peregrine.cpp` via `run_cases.sh`'s `tests --gtest_filter=PeregrineTOGT.plan`) as the actual TOGT-phase consumer, matching README.md and run_cases.sh.
- **info_loss_risk:** none

### [LOW] green_gate usage example overrides floor DOWN to a stale value
- **Location:** `scripts/green_gate.py:40`
- **issue_type:** stale · **severity:** low
- **Evidence (exact quote):** USAGE block: `python scripts/green_gate.py --baseline 884  # override the test-count floor`. 884 is a retired stale-base measurement (per the file's own L66), so the worked example demonstrates overriding the floor DOWN to a number below even the old default — confusing for a 1091-reality repo.
- **Recommended action:** Change the example override value to a current-plausible number (e.g. `--baseline 1091`) or a clearly-illustrative round number, so the doc does not model lowering the floor to a stale value.
- **info_loss_risk:** none — purely illustrative text.

### [LOW] test suite — obsolete gate-4 0.08-bar / NO-GO pins (CLEAN NEGATIVE — confirm-and-close)
- **Location:** `tests/` (whole dir; searched `test_inc8_sigmap0_eval.py`, `test_casec_foundation.py:139`, `test_estimator_emul.py` gate-4 block, all 92 files)
- **issue_type:** stale (negative) · **severity:** low
- **Evidence:** Searched for pins of the retracted 2026-06-19 view ("0.08 bar", "NO-GO", "near-field estimator pivot", "fix-seating wall", "RL wrong tool"). RESULT: NONE found in tests. Every literal `0.08` in tests is an unrelated physical/numeric constant — `test_casec_foundation.py:139` `L_const = 0.08` (vision latency seconds), `test_deploy_obs20.py:271` `nav_ip, nav_al = 0.07, 0.08` (sigma metres), `test_mixer.py:92` `measured 0.08 (idle + riding)`, `test_twin_fit.py:62` tau band. `test_inc8_sigmap0_eval.py:57` bounds the gate-4 crossing at `abs(y) < 0.42 and abs(z) < 0.42` — CONSISTENT with the corrected ~0.37-0.47 m clearance, NOT the obsolete 8 cm bar. The obsolete view lived only in prose/memory, never pinned in a test.
- **Recommended action:** No change needed — confirm-and-close. No test edit is required by the 2026-06-19 correction.
- **info_loss_risk:** none.

---

## DUPLICATION

### [LOW] Duplicated `_clip_norm` helper (byte-identical body)
- **Location:** `src/racer/controller.py:41-46` and `src/racer/twin.py:48-52`
- **issue_type:** duplication · **severity:** low
- **Evidence (exact quotes):**
  - `controller.py:41`: `def _clip_norm(v: np.ndarray, max_norm: float) -> np.ndarray:` + docstring `"""Scale ``v`` down so its norm is at most ``max_norm`` (direction preserved)."""` then `n = float(np.linalg.norm(v)); if n > max_norm > 0.0: return v * (max_norm / n); return v`
  - `twin.py:48`: identical body, NO docstring.
  - Byte-identical body; only the docstring differs.
- **Recommended action:** Consolidate: keep one canonical `_clip_norm` (promote the `controller.py` version which carries the docstring into a shared util — e.g. a small `racer.mathutil` or an existing import-safe module) and have `twin.py` import it. Low priority — both are private helpers, behavior identical; only de-duplicate if a shared util module already exists to avoid adding a new import edge into `twin.py`.
- **info_loss_risk:** none — identical implementations; preserve the controller.py docstring as canonical.

### [LOW] Duplicated LOOK-AT default constants across eval tools (drift risk)
- **Location:** `rl/inc8_sigmap0_torch_eval.py:111-114` vs `rl/contact_true_eval.py:72-75`
- **issue_type:** duplication · **severity:** low
- **Evidence (exact quotes):**
  - `inc8_sigmap0_torch_eval.py:111` `LOOKAT_G_YAW = -3.0` / `:112` `LOOKAT_G_PITCH = 3.0` / `:113` `LOOKAT_R_LO = 8.0` / `:114` `LOOKAT_R_HI = 30.0`
  - `contact_true_eval.py:72` `LOOKAT_G_YAW_DEFAULT = -3.0  # empirical S0 yaw sign` / `:73` `LOOKAT_G_PITCH_DEFAULT = 3.0  # empirical S2 pitch sign (opposite)` / `:74-75` R_LO/R_HI `8.0`/`30.0`
  - The four trained-gain defaults are hand-copied in two eval modules (and re-stated in each sbatch's COMMON/overrides). The look-at FUNCTION itself is NOT duplicated — it is the single canonical `inc8_reward.lookat_correction`, imported by both `contact_true_eval` (thin wrapper `_lookat_dlook_flu`) and `peregrine_racing_inc8` (via R8) — only these magic-number defaults drift-risk.
- **Recommended action:** Consolidate the four rc1 trained-gain defaults into ONE module-level home (e.g. add `LOOKAT_*_DEFAULT` to `inc8_reward.py` next to `lookat_correction`) and import them in both `inc8_sigmap0_torch_eval.py` and `contact_true_eval.py`, so a future gain change (e.g. the detstab `g_pitch` sweep landing a value != 3.0) cannot leave one tool stale. (NOTE: ties to the gate item below — when the sweep lands a winner != 3.0, this consolidated home is the single update point.)
- **info_loss_risk:** none — pure refactor; the empirical-sign comments must be carried to the new single home.

### [LOW] Duplicated gate-plane-miss / gate-crossing geometry (3 implementations)
- **Location:** `scripts/twin_fly_course.py:38` (`gate_plane_miss`); `scripts/togt/analyze.py:38` (`gate_crossings`); `scripts/twin_track_reference.py:80-82` (inline)
- **issue_type:** duplication · **severity:** low
- **Evidence (exact quotes):** Three implementations of the same "minimum-in-plane-miss crossing among all plane crossings after the previous gate" convention. They are aware of each other: `analyze.py:42` `(same convention as twin_fly_course.gate_plane_miss)` and `twin_track_reference.py:81-82` `(same convention as twin_fly_course.gate_plane_miss and scripts/togt/analyze.py)`. `twin_launch_phase_sweep.py:47` correctly IMPORTS the shared `twin_fly_course.gate_plane_miss` rather than re-implementing.
- **Recommended action:** Low priority: consider hoisting the shared crossing/plane-miss helper into one module (e.g. a small `src/racer` geometry util) that all three import, as `twin_launch_phase_sweep` already does. The docstring cross-references make this safe-but-not-urgent; the togt copy intentionally lives in the self-contained togt pipeline.
- **info_loss_risk:** none — behaviour identical and documented; consolidation, not a correctness fix.

### [LOW] test suite — duplicate test-function names across modules (INTENTIONAL mirror)
- **Location:** `tests/test_estimator_emul.py:139,229` vs `tests/test_inc8_estimator_emul_torch.py:246,289`
- **issue_type:** duplication (intentional) · **severity:** low
- **Evidence:** `test_kf_calibration_nees_in_band_bias_off` and `test_obs1720_bounds_and_responsiveness` are each defined in BOTH `test_estimator_emul.py` (numpy reference) and `test_inc8_estimator_emul_torch.py` (torch port). These are INTENTIONAL numpy<->torch parity mirrors (torch file docstring confirms the "numpy == torch <= 1e-4" mirror doctrine); pytest disambiguates by module path, so NOT a real collision.
- **Recommended action:** No action — confirm-and-close. The shared names are the deliberate numpy/torch mirror pattern. If desired for clarity only, the torch copies could be suffixed `_torch`, but this is cosmetic and risks breaking the intentional 1:1 mirror readability.
- **info_loss_risk:** none.

---

## GAP

### [LOW] Elodin solver adapter — placeholder stick mapping + latent quat double-conjugation (DEV SCAFFOLDING — keep)
- **Location:** `src/racer/elodin_adapter.py:138` (and `sensorupdate_to_state` quat handling)
- **issue_type:** gap · **severity:** low
- **Evidence (exact quotes):** `elodin_adapter.py:138` `"""Stick mapping. ALL VALUES ARE PLACEHOLDERS — calibrate against the rig."""` and the prior substrate audit (`handoff/ultracode-substrate-audit-2026-06-13/REPORT.md:90-93`) records that `sensorupdate_to_state` should conjugate the wire quat by `frames.ODO_QUAT_TRUE_CONJ_WXYZ` (matching `twin.py:368` / `mavlink_client.py:266`); the bare true-attitude quat is double-conjugated (~53.5 deg error). Module is referenced only by `docs/elodin.md` + `tests/test_elodin_adapter.py` (no live-flight importer).
- **Recommended action:** Do NOT prune — intentional dev/eval scaffolding (Elodin surrogate rig, unit-tested). Leave a one-line in-code pointer to the substrate-audit finding so the quat-conjugation gap + placeholder stick mapping are fixed if/when an Elodin->Navigator runner is built. Confirm the carry-forward is filed in the parked backlog (it is: project_master_plan / ultracode-substrate-audit handoff).
- **info_loss_risk:** none — nothing to delete; flags a latent bug + placeholder, both already documented in `handoff/ultracode-substrate-audit-2026-06-13/REPORT.md:90-93`. Preserve by keeping that REPORT and (optionally) cross-referencing it in the docstring.

### [LOW] select_ckpt eval default g_pitch=3.0 — CORRECT now, FORWARD-WATCH on sweep landing
- **Location:** `rl/inc8_select_ckpt.sbatch:25`, `:42-44`; `rl/inc8_sigmap0_torch_eval.py:111-112,:273`; `rl/contact_true_eval.py:73`
- **issue_type:** gap (forward-looking watch item) · **severity:** low
- **Evidence (exact quote):** `inc8_select_ckpt.sbatch:42-44` `GPITCH=${GPITCH:-3.0}  # MUST match the policy's TRAINED look-at pitch gain (eval default is 3.0 = rc1 gain; a detstab policy trained at g_pitch=1.0/1.5 MUST be evaluated at its OWN gain or it crashes like rc1-at-3 -> false NO-GO).` This is CORRECT and well-guarded. HOWEVER the detstab lever (42ddb0b) is actively SWEEPING g_pitch DOWN (default `GPITCH=1.0`, `peregrine_inc8_detstab.sbatch:70`) because rc1's deterministic ceiling was 0.5. If the sweep lands a NEW canonical g_pitch, the `3.0` eval-tool defaults (`torch_eval LOOKAT_G_PITCH=3.0`, `contact_true_eval LOOKAT_G_PITCH_DEFAULT=3.0`) become stale.
- **Recommended action:** No change now — the override knobs work and the comment is accurate. FLAG for follow-up: when the detstab g_pitch sweep selects a deterministic-flyable winner != 3.0, update the eval-tool `LOOKAT_G_PITCH` defaults (consolidated per the duplication finding above) to the new value. Track as a dependency of the detstab outcome.
- **info_loss_risk:** none — forward-looking watch item, not a current error.

### [LOW] No TODO/FIXME/XXX/HACK and no retracted-pivot text in rl/ source (CLEAN NEGATIVE)
- **Location:** `rl/` (whole tree: code + sbatch + sh; excludes `*.json` reference lines and `handoff/`)
- **issue_type:** gap (negative) · **severity:** low
- **Evidence:** Grep for `TODO|FIXME|XXX|HACK` => No matches. Grep for `near-field|fix-seating|pivot off RL|RL is the wrong tool|wrong tool|RETRACT|the wall|fix_rate=0` across rl/ => No matches in source (the "pivot" hits are all the legitimate "architecture pivot S0/S1" look-at-primitive lineage in `inc8_reward.py`/`peregrine_racing_inc8.py`/the s0 sbatch — a DIFFERENT, still-accurate pivot — NOT the retracted "pivot off RL to a near-field estimator"). The retracted near-field-estimator / fix-seating-wall language lives only in memory/handoff, NOT in rl/ code, so nothing to correct there.
- **Recommended action:** No action — recorded as a clean negative so the commander knows the retracted-pivot language did not leak into rl/ source; the only code-level stale residue of the old gate-4 view is the 0.08 bar (findings above).
- **info_loss_risk:** none.

### [LOW] green_gate diff->test import mapping over-maps leaf module to parent package
- **Location:** `scripts/green_gate.py:155-157` (`_module_names_for` parent-package token) and `:231` (`imps & mods` match)
- **issue_type:** gap · **severity:** low
- **Evidence (exact quotes):** L155-157 adds the parent package token: `# also the parent package ... if len(parts) > 1: names.add('.'.join(parts[:-1]))`. For `src/racer/estimator_obs.py` this yields the bare token `racer`, and L231 `importers = [tf for tf, imps in test_import_index.items() if imps & mods]` then matches EVERY test that does `from racer.* import ...`. The deploy-obs20 handoff `REPORT.md:156` documents the live consequence: `green_gate over-maps ``estimator_obs``->parent pkg ``racer``->63 modules ... Worth a separate cleanup task.` Over-inclusive (conservative for safety, but defeats diff-scoping and pulls in the unrelated `test_measured_aero.py` diffaero-stub flake).
- **Recommended action:** Tighten the parent-package token: only add the parent package name when the changed file IS the package `__init__` (or restrict the bare top-level `racer` token), so a leaf-module edit maps to its true importers rather than the whole `racer` package. Do NOT change the safety fallback (unmapped -> `--full`). Track as the cleanup task the deploy-obs20 report already flagged.
- **info_loss_risk:** none for safety — the change only narrows over-matching; the conservative widen-to-full fallback for genuinely unmapped changes is untouched.

---

## PRUNE

### [LOW] Superseded inc8 reward-ladder run sbatch (lineage artifacts — archive, do NOT delete)
- **Location:** `rl/peregrine_inc8_repilot.sbatch` (v3 fix-driven, 2026-06-15), `rl/peregrine_inc8_s0.sbatch` (S0 yaw-only, 2026-06-16), `rl/peregrine_inc8_warmstart.sbatch` (S3 warm-start probe, 2026-06-17)
- **issue_type:** prune · **severity:** low
- **Evidence (exact quotes):** These three encode the historical reward-tuning/architecture-pivot ladder SUPERSEDED by the root-cause fix. `peregrine_inc8_recenter.sbatch:12-20` records the resolution (`the inc8 lineage NEVER flew a lap ... ROOT ... the inc8 reward redesign DROPPED inc7 through-approach centering ... The whole ladder optimised a PROXY (camera pointing) and never gated the OBJECTIVE`); detstab (42ddb0b) is the now-active lever. `repilot.sbatch:30-32` self-documents its STOP CONDITION (`iteration 4 ... the problem is reward ARCHITECTURE ... STOP and flag for the commander`). None of the three is referenced by the active detstab/select_ckpt pipeline.
- **Recommended action:** Do NOT delete. Consolidate: move the three superseded sbatch into an archive subdir (e.g. `rl/archive_inc8_ladder/`) OR confirm their narrative is captured in `[[project-rl-increment-history]] §inc8` and then archive. Verify the lineage prose (4-NO-GO exploration-wall finding, S0/S1/S2/S3 stage definitions, warm-start cliff diagnosis) is banked there BEFORE archiving; if any detail is sbatch-only, bank it first.
- **info_loss_risk:** Risk: these sbatch carry stage-specific empirical findings (repilot's `pooled fix_rate >= 0.50 was never achievable -- terminal 6 m is a geometric dead zone`; warmstart's S3 reward-cliff `value_loss 400-650` diagnosis; s0's FLU/FRD sign-gate procedure). Preserve by confirming each is in `[[index-rl-training]]`/`[[project-rl-increment-history]] §inc8` before moving; cite the memory location in the archive commit.

### [LOW] Superseded pre-inc7 racing increment sbatch (lineage artifacts — archive, do NOT delete)
- **Location:** `rl/peregrine_racing_inc5.sbatch` (2026-06-11), `rl/peregrine_racing_inc6.sbatch` (2026-06-11), `rl/peregrine_racing_s13.sbatch` (2026-06-10), `rl/peregrine_racing_s14.sbatch` (2026-06-11), `rl/peregrine_s13_sweep.sbatch` (2026-06-10), `rl/peregrine_smoke.sbatch` + `rl/peregrine_gate.sbatch` (2026-06-08)
- **issue_type:** prune · **severity:** low
- **Evidence:** All invoke the older `peregrine_train_racing.py` / `peregrine_train.py` and predate inc7 (the documented current-best baseline: MEMORY "inc7 LIVE-CONFIRMED = current best"). They are the s13/s14/inc5/inc6 reward-lineage runs superseded by inc7 (`rl/peregrine_racing_inc7.sbatch`, 2026-06-12, which is KEPT — the standing baseline). Not referenced by any active pipeline.
- **Recommended action:** Do NOT delete (reproducibility lineage). Consolidate into an archive subdir alongside the inc8-ladder archive, keeping `peregrine_racing_inc7.sbatch` in place as the live baseline. Verify the increment history is captured in `[[project-rl-increment-history]]` before moving.
- **info_loss_risk:** Risk: DR-band definitions embedded in these headers (inc5 `residual d1~U[0,0.08]`; inc6 `S17 DR bands idle [0.04,0.08], kappa_err [0.060,0.085]...`). These DR bands ALSO live in `rl/diffaero_dynamics.py:449-460` as code defaults (`dr_d1_hi=0.08`, `dr_mix_idle_hi=0.08`, `dr_mix_kerr_hi=0.085`), so the numbers are not sbatch-only — but confirm the lineage rationale is in memory before archiving.

### [LOW] task2_gate_pnp — resolved one-off geometry probe still in scripts/
- **Location:** `scripts/task2_gate_pnp.py` (whole file)
- **issue_type:** prune · **severity:** low
- **Evidence (exact quotes):** `task2_gate_pnp.py:1` `Task 2 -- resolve gate-0 map anchor (bottom-CENTER vs bottom-CORNER) by vision PnP.` A sibling states the question is ANSWERED: `characterize_perception.py:5-6` `It is NOT Task 2: ``task2_gate_pnp.py`` answered a geometry question (is the map anchor a corner or centre?) with a rotation-FIXED solver`. It hardcodes a handoff bundle path (`BUNDLE = handoff/shadowpc-followups-2026-06-05/task2_frames`) and the absent `models/` weights. Single-use diagnostic whose verdict has shipped (`corner_to_center` in the live chain).
- **Recommended action:** Consolidate: move `task2_gate_pnp.py` into the handoff/ tree alongside the bundle it consumes (closed one-off probe, not standing tooling), OR leave it but add a one-line header noting the verdict is settled and the script is archival. Do NOT delete — it is the reproducible record of the anchor-resolution geometry. Confirm the verdict is banked in the vision-estimator memory before any move.
- **info_loss_risk:** The corner-vs-center verdict and the rotation-FIXED-prior technique are load-bearing. Verify they are captured in `index-vision-estimator` before relocating; if only the script records the method, keep it in scripts/ with an archival header instead of moving.

### [LOW] TOGT refinement upstream TODOs (vendored third-party — do NOT edit)
- **Location:** `scripts/togt/refine/quadrotor.py:216`, `:414`; `scripts/togt/refine/optimization.py:125`; `scripts/togt/refine/trajectory.py:131`, `:176`
- **issue_type:** prune · **severity:** low
- **Evidence (exact quotes):** TODO markers in `refine/`, e.g. `quadrotor.py:216` `# TODO: us RK4 integration`, `optimization.py:125` `#TODO: Problem: the second u could be wrong!!!!`, `trajectory.py:131` `#TODO: find the closest t in self._t and return its index`. `README.md:67-68` marks `refine/{optimization,trajectory}.py` as `verbatim copies (MIT, (c) 2024 FSC Lab)` and quadrotor.py as a copy plus the linear_drag term — so these TODOs are UPSTREAM, not Peregrine debt.
- **Recommended action:** No action on the TODOs (do not edit vendored upstream code). Optionally note in README.md's provenance section that the `refine/` TODOs are upstream FSC-Lab markers, so a future cleanup pass does not mistake them for local debt.
- **info_loss_risk:** none — preserving verbatim-copy fidelity is the point; leave as-is.

### [LOW] test suite — conditional skip whose guard is now always-true
- **Location:** `tests/test_calib_dualform.py:167-168`, `:171`, `:186`, `:192`
- **issue_type:** prune · **severity:** low
- **Evidence (exact quotes):** L167-168: `_HAS = hasattr(F, 'BORESIGHT')` / `_skip = pytest.mark.skipif(not _HAS, reason='frames.BORESIGHT not present until proposed_calib_dualform.patch lands')`, guarding 3 tests (L171/186/192). But `src/racer/frames.py:58` now ships `BORESIGHT = BoresightCorrection(vert_offset_m=-0.25)` (DEPLOYED), so `_HAS` is always True and the skip never fires — the "until the patch lands" framing is obsolete (the patch landed).
- **Recommended action:** Consolidate: drop the now-vacuous `_skip` marker (or update the reason to a present-tense defensive note), since `frames.BORESIGHT` is permanently present on main. Verify the 3 tests pass un-skipped before removing the guard. Low priority — currently harmless (tests do run).
- **info_loss_risk:** none — tests already execute (guard always-true); removing/relabelling does not change collection or coverage. Confirm by running the 3 tests un-skipped first.

### [LOW] inc8 look-at-primitive "architecture pivot" tokens — NOT-A-FINDING / disambiguation (do NOT flag)
- **Location:** `rl/inc8_reward.py:66,167-168`; `rl/peregrine_inc8_s0.sbatch:10-13`; `rl/peregrine_racing_inc8.py:118`; `rl/inc8_select_ckpt.sbatch:44`; `rl/peregrine_inc8_recenter.sbatch:27`
- **issue_type:** prune (negative / disambiguation) · **severity:** low
- **Evidence (exact quotes):** These contain "architecture pivot" (`inc8_reward.py:167` `===== active-perception LOOK-AT primitive (architecture pivot S0)`) and "NO-GO" (`inc8_reward.py:168` `After 4 weight-tuning NO-GOs (pointing->fix is sparse...`; `select_ckpt.sbatch:44` `...or it crashes like rc1-at-3 -> false NO-GO`; the GO/NO-GO verdict machinery in the eval tools).
- **Recommended action:** NO CHANGE — explicitly do NOT flag these. The "pivot" here is the look-at-primitive architecture pivot, which MEMORY.md confirms is the CURRENT load-bearing fix (distinct from the RETRACTED "pivot off RL to a near-field estimator"). The "NO-GO" tokens are either the generic GO/NO-GO eval vocabulary or the still-valid reward-architecture history. Listed only so a future sweep does not mis-flag them.
- **info_loss_risk:** none — nothing removed; documentation of correct-and-current content.

---

## PRESENTABILITY

### [LOW] test suite — skip/xfail/importorskip inventory (orphan check — HEALTHY)
- **Location:** `tests/` (`test_inc8_lookat.py:55`, `test_inc8_sigmap0_eval.py:28`, `test_diagnose_session.py:105/118/164/266`, `test_confirmed_cr1-4` pytestmark skipifs, ~14 module-level `pytest.importorskip('torch')`)
- **issue_type:** presentability · **severity:** low
- **Evidence:** All skips are CONDITION-GUARDED on legitimately-absent dependencies/data, not blanket disables: `pytest.importorskip('torch')` (~14 files), `skipif(not _INC8.exists())` (inc8 actor checkpoint, gitignored), `skipif(... diffaero not importable ...)` (`test_inc8_lookat.py:55`, "GPU/cluster only"), `skipif(not _REAL.is_dir())` (recorded bundles absent), `skipif(not _HAVE_SCIPY)`. No `@pytest.mark.xfail`, no unconditional `pytest.mark.skip`, no orphaned test imports a missing module (`tests/_audit_io.py` is a live shared helper imported by 10 test modules). On the canonical laptop (torch present) these guards mostly pass through.
- **Recommended action:** No action — confirm-and-close. The skip surface is healthy (environment/artifact-gated, self-documenting reasons). Note: the green_gate stack-guard (forces torch present => RED if absent) already prevents the importorskip traps from masking the load-bearing invariants.
- **info_loss_risk:** none.

---

## Per-issue-type counts (after dedup)

| issue_type      | count |
|-----------------|-------|
| contradiction   | 3     |
| stale           | 14    |
| duplication     | 4     |
| gap             | 4     |
| prune           | 6     |
| presentability  | 1     |
| **TOTAL**       | **32** |

**Merges applied (raw 39 -> 32 deduplicated):**
- `inc8_sigmap0_torch_eval.py` 0.08 bar: `code:rl`(contradiction) + `code:stalegrep`(stale) -> 1 (contradiction/high)
- `inc8_sigmap0_eval.py` 0.08 bar: `code:rl`(contradiction) + `code:stalegrep`(stale) -> 1 (contradiction/high)
- `inc8_sigmap0_torch.sbatch` deliverable 0.08: `code:rl`(stale/med) + `code:stalegrep`(stale/high) -> 1 (stale/high)
- `peregrine_inc8_warmstart.sbatch` 0.08: `code:rl`(stale/low) + `code:stalegrep`(stale/med) -> 1 (stale/med)
- `green_gate.py:67` baseline 933: `code:cluster+scripts`(stale/med) + `code:tests+greengate`(stale/high) -> 1 (stale/high)
- (The duplicated low-severity "inc8 sigma_p0 torch sbatch GO-rule prose" entry is folded into the merged launch-sbatch [HIGH] entry; the disambiguation NOT-A-FINDING entries from `code:rl` and `code:stalegrep` are the same content, kept once.)
