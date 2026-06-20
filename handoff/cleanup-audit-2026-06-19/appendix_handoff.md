# Cleanup Audit Appendix — Area: Handoff reports (bucket: `handoff`)

**Generated:** 2026-06-19 · **Scope:** the entire `handoff/` tree (inc8-* / sigmap0-* / shadowpc-* / laptop-* / ultracode-* / p0..p5 / margin & boresight & coast / VQ2 / misc ops) plus the binding live-code consumers and adjacent within-memory hygiene lines those reports point to.

---

## Area state (headline)

The handoff tree is the **stale-conclusion epicenter** for one specific error: the **gate-4 σ_p0 ≲ 0.08 m "closure bar" was a DOUBLE-COUNT** (the drone extent was subtracted twice — `W_EFF = 0.75 − 0.215 chassis`, then `MARGIN = W_EFF − r`; `0.08 = 0.235/3`). The 2026-06-19 correction (commit 511e85c per findings; the corrected SSOT lives at **memory/MEMORY.md:8** as held by the commander) sets the **real bar ≈ σ_p0_lat ≲ 0.15 (p99 ≲ 0.45)**, re-judges the measured σ_p0 0.15–0.20 / p99 0.37–0.40 as **MARGINAL-PASSING (not NO-GO)**, and **RETRACTS** the "pivot off RL to a near-field-gate estimator" recommendation (RL stays the lever; vision = support). A large cohort of reports issue NO-GO / NO-CLOSE / does-not-close verdicts judged against the old 0.08-derived bar and must be banner-corrected before archive so a future reader is not misled. The **measured numbers, root-causes, and durable build artifacts are all banked or merged** — the cleanup is almost entirely *neutralize-the-verdict-prose, preserve-the-substance, then archive*. Two non-archival exposures stand out: (1) **two LIVE evaluator tools still default `sigma_target=0.08`** and will silently re-stamp the wrong verdict on any re-run; (2) the **shadowpc set (27 dirs) carries ZERO gate-4 stale conclusions** — confirmed by exhaustive grep — so the correction risk lives ONLY in the later inc8/sigmap0/margin/boresight/coast dirs. A scatter of older shadowpc/laptop/ultracode dirs are pure duplication (banked verbatim) or prune-able scratch, but several contain **HARD ACTIVE FIXTURES** (track_map.json, sysid/, task2_frames/, extracted/, debug_obs_*.zip, WRITEUP provenance citations) that MUST be kept in place.

### IMPORTANT on-disk reconciliation note (surfaced for the commander)
The findings repeatedly cite **memory/MEMORY.md:8** as the home of the *corrected* bar (0.15 / double-count / near-field-RETRACTED). The MEMORY.md present in THIS audit worktree still shows the **pre-correction NOW block at line 8** ("GATE-4 CLOSURE REDUCES TO ONE NUMBER … σ_p0_lat ≲ 0.08", "BINDING GATE-4 σ_p0 = 0.198 m = NO-GO vs 0.08", "2nd lever = near-field GATE estimator"). The commander's canonical `main` carries the corrected line; this worktree's copy is behind. **The banner corrections below assume the corrected SSOT.** If memory/MEMORY.md:8 in the canonical tree still carries any residual stale clause (esp. the "2nd lever = near-field GATE estimator" sub-bullet inside the SIGN FOOTGUN, and the same stale 0.08 boundary + near-field-lever text reportedly still at index_vision_estimator.md:285-287), reconcile memory FIRST — else archiving the handoffs loses nothing but the stale memory survives.

---

# ISSUE TYPE: CONTRADICTION (live-truth conflicts — fix first)

## [HIGH] Gate-4 verdict — THE stale-conclusion epicenter (det-retrain REPORT)
- **Location:** `handoff/inc8-deterministic-retrain-2026-06-19/REPORT.md:89,101,132-135,144-150` (incl. MEMORY-DELTA #4)
- **Issue type:** contradiction · **Severity:** high
- **Evidence:** REPORT says "Objective B MEASURED = NO-GO"; "2-axis σ_p0_lat ~ 0.15-0.20 m = NO-GO vs 0.08 (lat_p99 0.37-0.40 vs 0.24)"; "Gate-4 NOT closed by the look-at bet … Definitive across the bracket"; MEMORY-DELTA #4: "NEXT LEVER = near-field-gate estimator … NOT more look-at-gain RL". CONTRADICTED by memory/MEMORY.md:8 (corrected): "Measured σ_p0 0.15-0.20 / p99 0.37-0.40 = MARGINAL-PASSING vs the real ~0.15/0.45 clearance (sim success 0.65) -- NOT the NO-GO we mis-called vs the bad 0.08 bar … the 'pivot off RL to a near-field-gate estimator' recommendation is RETRACTED; next lever = RL to lift reach/pass-rate."
- **Recommended action:** Do NOT delete (the measured numbers — det-reach 0.467 / σ_p0 0.15-0.20 / the noise-anneal/std-cap lever — are the load-bearing result and ARE banked in MEMORY.md:8). Prepend a dated SUPERSEDED banner to the REPORT header pointing to MEMORY.md:8 / commit 511e85c: "VERDICT REVERSED 2026-06-19: the 0.08 bar was a double-count; measured σ_p0 0.15-0.20 is MARGINAL-PASSING not NO-GO; the near-field-estimator pivot is RETRACTED; RL stays the tool." Then archive the dir (findings fully banked).
- **Info-loss risk:** none — the measured numbers, the rc1 actor_logstd-saturation root cause, and the std-ceiling lever are all banked verbatim in memory/MEMORY.md:8; the banner only neutralizes the wrong verdict prose.

## [HIGH] σ_p0 GO rule baked into LIVE evaluator code (binding consumer, NOT a report)
- **Location:** `rl/inc8_sigmap0_eval.py:174,199,237` and `rl/inc8_sigmap0_torch_eval.py:223,292,394` (default `sigma_target=0.08`)
- **Issue type:** contradiction · **Severity:** high
- **Evidence:** Confirmed live on `main` this audit: `inc8_sigmap0_eval.py:174 def _go_verdict(m, sigma_target: float = 0.08)`; `:199 ap.add_argument("--sigma-target", type=float, default=0.08)`; `:237` prints the GO RULE against `args.sigma_target`. Identical in `inc8_sigmap0_torch_eval.py:223,292,373-394`. (Also docstring `inc8_sigmap0_eval.py:5,27` states the `<=0.08`/`<=0.24` GO rule.) These tools STAMP a GO/NO-GO using the retracted 0.08 bar by default. MEMORY.md:8 corrects the bar to ~0.15. Outside the handoff tree but the binding consumer that the inc8-sigmap0-* reports point to — a re-run today silently re-emits the wrong NO-GO.
- **Recommended action:** FLAG UP to commander (code change, not an archive op): change the default `sigma_target` 0.08 → 0.15 (and the p99 multiple/comment/docstring) in BOTH files, or add a loud retracted-0.08 comment. Out of audit scope to edit (READ-ONLY) but MUST NOT be missed.
- **Info-loss risk:** none — corrected bar is in MEMORY.md:8; this is a code-default that needs updating to match.

## [HIGH] Faithful σ_p0 eval verdict (numpy/yaw path) — NO-GO vs 0.08
- **Location:** `handoff/inc8-eval-pitch-2026-06-18/REPORT.md:8,107,115,148` and MEMORY-DELTA:147
- **Issue type:** contradiction · **Severity:** high
- **Evidence:** "seed0 σ_p0_lat = 0.177 m -> NO-GO vs 0.08"; table row "yaw (faithful) | 0.177 ... NO-GO (>0.08)"; MEMORY-DELTA "yaw-only seed0 = 0.177 ... -> NO-GO vs 0.08". CONTRADICTED by MEMORY.md:8 corrected bar (~0.15/0.45 → 0.177 is marginal, not categorical NO-GO). NOTE: the report's PRIMARY technical finding (the YAW-not-pitch injection bug, fixed at contact_true_eval.py:365/367 = ec4cb03) is CORRECT and banked at project_rl_increment_history.md:750.
- **Recommended action:** Prepend SUPERSEDED-VERDICT banner ("0.08 bar retracted → 0.177 is marginal-passing per MEMORY.md:8; the YAW-injection fix below stands"), then archive. The yaw-injection trace + fix is the keep-worthy substance and is already banked.
- **Info-loss risk:** none — the injection root-cause/fix (ec4cb03, pinned test_sysid_action_sign_maps.py) is banked at project_rl_increment_history.md:750.

## [HIGH] sigmap0-adroit NO-GO-vs-0.08 verdict (the precursor to the merged det-retrain)
- **Location:** `handoff/sigmap0-adroit-2026-06-19/REPORT.md:9-10,36,150-152,171,191` (incl. MEMORY-DELTA #5)
- **Issue type:** contradiction (catalogued by one agent as `stale`; merged here — same 0.08-bar verdict conflict) · **Severity:** high
- **Evidence:** "yaw-only seed0 number 0.198 m -> NO-GO vs 0.08" (TL;DR :9); ":36 verdict=NO-GO"; ":151 σ_p0_lat = 0.198 m, σ_p0_vert = 0.191 m, lat_p99 = 0.399 -> NO-GO vs 0.08 / 0.24"; MEMORY-DELTA #5 ":191 Binding gate-4 σ_p0 = 0.198 m (yaw/racing-line) = NO-GO vs 0.08". Per MEMORY.md:8 these SAME numbers (0.15-0.20 / p99 0.37-0.40) are MARGINAL-PASSING vs the real ~0.15/0.45 bar, NOT NO-GO.
- **Recommended action:** Annotate the NO-GO verdict lines as JUDGED-AGAINST-THE-WRONG-0.08-BAR (now marginal-passing) then bank-then-archive. The DURABLE root-cause IS banked and load-bearing and led directly to the merged retrain (42ddb0b): "torch σ_p0 instrument VALIDATED (yaw 0.1981 vs numpy 0.1768 AGREE); rc1 crashes ~1 s under DETERMINISTIC inference (training success was a stochastic artifact); g_pitch stability cliff between 0.5 and 1.0; LEVER = retrain for deterministic stability, select on deterministic reach." Retire its NO-GO bar, keep its mechanism.
- **Info-loss risk:** MED — instrument-validation, the stochastic-vs-deterministic gap, and the g_pitch cliff are durable and led to the merged retrain. CARRY-FORWARD to verify: the "5th harness bug" fix to `rl/inc8_sigmap0_torch.sbatch` (RUNDECOR best/ path) was "working-tree edit, uncommitted, not pushed" (lines 140,193) — confirm it landed or the next σ_p0 sbatch eval silently re-breaks. **Flag to commander.**

## [HIGH] Gate-4 margin: coast-drift closure boundary σ_p0 ≤ 0.08 (closes-IFF framing)
- **Location:** `handoff/coast-drift-2026-06-15/REPORT.md:36-41,142-168,184-199,241-256` (also :37, :146-153, :165, MEMORY-DELTA :241-245)
- **Issue type:** contradiction (two agents; one tagged `stale`, merged) · **Severity:** high
- **Evidence:** ":36 r=0.30 (B=0.235) closes IFF the terminal centering σ_p0 ≲ 0.075-0.08 m"; ":153 Use σ_p0_lat ≲ 0.08 m as the gate-4 r=0.30 closure boundary"; ":155 B_min ~ 3.0*σ_p0". DIRECTLY CONTRADICTS MEMORY.md:8: "the σ_p0 ≲ 0.08 'closure bar' was a DOUBLE-COUNT … REAL bar ≈ σ_p0_lat ≲ 0.15 (p99 ≲ 0.45)" and "Measured σ_p0 0.15-0.20 … = MARGINAL-PASSING … NOT the NO-GO we mis-called vs the bad 0.08 bar."
- **Recommended action:** Prepend a dated CORRECTION banner at the top of REPORT §0 (real bar ~0.15 not 0.08; σ_p0 0.15-0.20 is MARGINAL-PASS not NO-GO; near-field-estimator-as-lever RETRACTED). Keep the body (coast-is-cheap physics + σ_v-non-binding finding are valid and banked). Then archive. Do NOT delete — the coast<0.02 m decomposition and VIO-trigger back-solve are uniquely captured here and at index_vision_estimator.md:284-287.
- **Info-loss risk:** MED — coast-cheapness / σ_v-IMU-floored / VIO-trigger findings are load-bearing and banked at index_vision_estimator.md:284-287. Only the 0.08-bar FRAMING is wrong; the underlying coast physics survives. NOTE: index_vision_estimator.md:285-287 reportedly STILL carries the stale 0.08 boundary + "near-field GATE estimator" lever as current text — flag to fix memory too.

## [HIGH] Gate-4 boresight-closure NO-CLOSE verdict on the 0.08-derived bar
- **Location:** `handoff/boresight-closure-2026-06-14/REPORT.md:13-36,95,98-112,143-155,180-190` (also :15-19, :31-36, :100-108, MEMORY-DELTA :179-189)
- **Issue type:** contradiction (two agents; merged) · **Severity:** high
- **Evidence:** VERDICT (:15) "NO-CLOSE at the measured operating point"; (:95) "MARGIN(r) = W_EFF - r = (0.75 - r) - 0.215 … r0.30→0.235, r0.38→0.155"; requirements (:31-34) "terminal gate-lock — accepted gate-4 fixes through the last ~6 m" + "effective fix-rate ≥ 0.50"; MEMORY-DELTA (:187) hardcodes "σ_lat ceiling ~0.23-0.245 for r=0.30 @ fr0.50". The whole closure table tests p99 against MARGIN(r=0.30)=0.235 from the W_EFF=0.535 double-count. Per MEMORY.md:8 real clearance 0.37-0.47 m → NO_CLOSE cells re-judged MARGINAL-PASSING.
- **Recommended action:** Prepend a CORRECTION banner to §0/VERDICT (the 0.235/0.245 margin is the double-counted bar; real clearance 0.37-0.47; NO-CLOSE → marginal-pass). KEEP the two genuinely-valid axes this report closed — (1) the −0.25 boresight bake (DEPLOYED 5764291, banked index_vision_estimator.md:258) and (2) the anisotropic-σ-recal [lat 0.19, vert 0.10] pooling-artifact finding (banked) — those are unaffected. Then archive.
- **Info-loss risk:** LOW — boresight bake + σ-recal/anisotropy findings are banked (index_vision_estimator.md:258, :87). The only at-risk durable nugget is the at-speed-blur-extrapolation reasoning, subsumed by at-speed-sigma's blur-free finding; cite that. Only the NO-CLOSE margin verdict framing is stale.

## [HIGH] Gate-4 margin: near-field GATE estimator named as THE load-bearing centering lever
- **Location:** `handoff/coast-drift-2026-06-15/REPORT.md:189,199,248-249` (also accept-geometry, boresight-closure)
- **Issue type:** contradiction · **Severity:** high (catalogued `stale`/`high`; merged into the contradiction group — it is the retracted-recommendation half of coast-drift)
- **Evidence:** ":199 Near-field GATE estimator (lower the 12 m floor) → MEDIUM-HIGH"; ":248-249 Centering lever = near-field GATE estimator … NOT odometry." The 2026-06-19 correction (MEMORY.md:8) RETRACTS this: "the 'pivot off RL to a near-field-gate estimator' recommendation is RETRACTED; next lever = RL to lift reach/pass-rate; vision = SUPPORT, do NOT lean on it."
- **Recommended action:** Covered by the same CORRECTION banner on coast-drift (above). When banking, ensure the RETRACTION is reflected. NOTE: memory/MEMORY.md:8 ITSELF still carries a residual stale clause "2nd lever = near-field GATE estimator (lower the 12 m floor)" inside the SIGN FOOTGUN sub-bullet — reconcile with the RETRACTION two sentences earlier (surface to memory owner).
- **Info-loss risk:** none — the near-field-estimator idea is preserved as a parked/secondary option; only its status as THE current lever is retracted.

## [HIGH] Body-contact gate-4 CLOSES_EXCEPT_WORST_BIAS / QUALIFIED-NO-CLOSE verdict
- **Location:** `handoff/body-contact-reconcile-2026-06-13/CORRECTED_MARGIN.md:32-35,52-65,113-170` and `BODY_RECONCILE_VERIFY.md:15-34,156-182` (also CORRECTED_MARGIN :59,:163; VERIFY :16-34)
- **Issue type:** contradiction (two agents; merged) · **Severity:** high
- **Evidence:** CORRECTED_MARGIN.md:60 derives "budget(r) = (0.75 − r) − 0.215" (linf0 = "radius-invariant gate-4 simstart crossing offset" = a SECOND 0.215 drone-extent subtraction) and :32/:163 concludes "VERDICT: CLOSES_EXCEPT_WORST_BIAS"; BODY_RECONCILE_VERIFY.md:32 upgrades to "QUALIFIED-NO-CLOSE … the realistic case-C tail does not close at any radius at p90 or p99." Both rest on the 0.155/0.235 budget the correction (MEMORY.md:8) shows is a double-count → real clearance 0.37-0.47.
- **Recommended action:** Prepend a CORRECTION banner to BOTH files (budget=(0.75−r)−0.215 is the double-count; verdict re-judged marginal-pass). KEEP the GEOMETRY half — it is DURABLE and banked: 3D box half-diagonal 0.2135 m as the absolute non-geometric ceiling, 0.38 = rotor/prop crash-halo (NOT geometry), posture-matched r_eff 0.26-0.38, central reporting radius 0.30, prop-span adds 0.23-0.29, "keep 0.38 as worst-case knob" (memory/MEMORY.md CONTACT-RADIUS footgun + index_vision_estimator.md:54). Then archive both. NOTE: file footers cite absolute paths under the STALE `C:/Users/Fengy/Downloads/Projects/Anduril` tree (CORRECTED_MARGIN.md:217-221) — update to Anduril-cmdr if retained.
- **Info-loss risk:** none for geometry (banked). The cold@1.4°-attitude-bias p90=0.338 phantom-accel (g·sin) mechanism is a real datapoint — preserve the mechanism note, discard the "fails at every radius" verdict (margin-double-count-driven). Do NOT delete — these are the provenance of the 0.2135 m ceiling.

## [HIGH] margin_envelope.py — the literal source of the double-count the correction names
- **Location:** `handoff/margin-closure-envelope-2026-06-14/margin_envelope.py:33,77-78,101-103,352-353`
- **Issue type:** contradiction (two agents; one tagged `stale`, merged) · **Severity:** high
- **Evidence:** `:78 W_EFF = MARGIN_G4_AT_R038 + 0.38  # 0.535 m = 0.75 gate-clear-halfwidth - 0.215 chassis half-diag` and `:102 def margin_at(r): return W_EFF - r`. This is EXACTLY the formula MEMORY.md:8 names: "margin_envelope.py subtracted the drone TWICE (W_EFF=0.75−0.215 chassis, then MARGIN=W_EFF−r=0.535−0.30=0.235; 0.08=0.235/3)." The docstring presents MARGIN(r)=W_EFF−r as the correct gate-4 margin; MARGIN(0.30)=0.235 / MARGIN(0.38)=0.155 budgets are wrong by one drone-extent subtraction; every downstream NO_CLOSE verdict built on it is invalid.
- **Recommended action:** This dir is the canonical artifact of the corrected bug — KEEP (annotate-as-superseded), do NOT archive standalone (it is an active engine reused by boresight-closure/coast-drift). Add a header comment (or a SUPERSEDED.md in the dir) flagging the W_EFF double-count per commit 511e85c: "W_EFF double-counts the drone; real clearance = 0.75 − r (~0.37-0.47 m); real bar σ_p0_lat ~0.15. DO NOT reuse margin_at() / W_EFF." Do NOT delete — it is cited by name in the correction.
- **Info-loss risk:** none — the bug and its correct replacement are fully banked in memory/MEMORY.md:8. Preserve the script as the worked example the correction refers to.

## [HIGH] Commander cross-check: σ_v-binding + 0.155 margin (DOUBLY stale)
- **Location:** `handoff/_commander-xcheck/CROSSCHECK_FINDINGS.md:30-65`
- **Issue type:** contradiction (stale on two axes) · **Severity:** high
- **Evidence:** Headline table tests "clears 0.155 RMS / p90" with margin 0.155 (=double-counted MARGIN_G4@r=0.38) and concludes (:48) "The verdict pivots on σ_v … at σ_v ~ 0.3 m/s the RMS itself crosses the margin" + (:64) "SELECT THE SPEED RUNG … likely BELOW 37 m/s". DOUBLY stale: (a) the 0.155 margin is the double-count; (b) the σ_v-is-binding thesis was REFUTED by coast-drift (σ_v0~0.02 m/s IMU-floored, NON-binding) and MEMORY.md:8 ("velocity … ALL ruled out").
- **Recommended action:** Annotate as SUPERSEDED (both the 0.155 margin and the σ_v-binding conclusion) then archive. The durable methodological nugget — "RewindKF is NOT the binding term; latency compensates correctly; a margin is a worst-case (p90/p95) gate not an RMS gate" — is preserved in the c2-estimator-chain build and the body-contact p99 discipline. No REPORT.md; one-off commander scratch.
- **Info-loss risk:** LOW — RewindKF-latency-OK and the p90/p99-worst-case-gate discipline are captured elsewhere (c2 chain merged 05ed750; body-contact VERIFY). The σ_v honesty-bound reasoning is obsolete (refuted).

## [MED] Faithful σ_p0 eval verdict (look-at port) — stale on TWO axes
- **Location:** `handoff/inc8-eval-lookat-2026-06-18/REPORT.md:20,96,127`
- **Issue type:** contradiction · **Severity:** med
- **Evidence:** "σ_p0_lat = 0.177 m … -> NO-GO vs the 0.08 target"; "σ_p0_lat 0.177 m ~ 2.2x the 0.08 target -> NO-GO". ALSO: this report's own `_RATE_SIGN_LIVE=[1,1,-1]` / pitch-breaks-plant diagnosis was SUPERSEDED by inc8-eval-pitch (real value [1,1,1], bug was yaw) per that report's MEMORY-DELTA:141-142 and project_rl_increment_history.md:750. Stale on TWO axes: the 0.08 verdict AND its own pitch diagnosis.
- **Recommended action:** Prepend a banner noting BOTH supersessions: "(1) 0.08 bar retracted, 0.177 is marginal per MEMORY.md:8; (2) the pitch-plant / _RATE_SIGN_LIVE=[1,1,-1] diagnosis is WRONG — corrected to yaw-bug / [1,1,1] by inc8-eval-pitch-2026-06-18 = ec4cb03." Then archive.
- **Info-loss risk:** none — both the correct injection fix and the correction of this report's pitch error are banked at project_rl_increment_history.md:750.

## [MED] Cross-report contradiction: gate-3 1.46 m map mis-registration status
- **Location:** `handoff/ultracode-planning-togt-s2-2026-06-13/REPORT.md:115,155` vs `handoff/ultracode-vision-case-c-2026-06-13/REPORT.md:215-220,329,379`
- **Issue type:** contradiction · **Severity:** med
- **Evidence:** planning-togt-s2 :155 (OPEN item 7) "track_map gate-3 D-offset ~1.46 m confirmed fragility, metric flips at ~1.5 m shift" (treated as live). ultracode-vision-case-c :329 records gate-3 "0.018 in-plane (NOT 1.46 m) … memory claim falsified" and :379 recommends correcting commit 82c2d20 / MEMORY.md. Memory has since adopted the FALSIFICATION (index_rl_training.md:8, index_vision_estimator.md:213).
- **Recommended action:** Add a note to planning-togt-s2 item 7 (and the :115 dependency) that the gate-3 1.46 m fragility was subsequently FALSIFIED (bottom-vs-opening-centre artifact, true miss 0.017 m; banked index_vision_estimator.md:213). Both reports archive together → low-risk; the falsification is already canonical in memory.
- **Info-loss risk:** none — the falsification is banked; this is just a stale cross-reference inside an older report.

## [LOW] Within-memory stale line: mixer probe2 capture status
- **Location:** `memory/project_rl_increment_history.md:117` vs `:141`
- **Issue type:** contradiction · **Severity:** low
- **Evidence:** :117 "Probe rows c100_y31 / c60_r31 / zhov_r31 remain uncaptured (~10 min unattended probe in next session)" CONTRADICTS :141 "Mixer probe2 COMPLETE (2026-06-12): all 4 remaining rows captured (c100_r31, c100_y31, c60_r31, zhov_r31) … handoff/shadowpc-bridge-retest-2026-06-12/probe_summary.md". The bridge-retest dir confirms :141 (all 4 rows captured).
- **Recommended action:** MEMORY hygiene adjacent to this set (source = bridge-retest, in scope). Update memory/project_rl_increment_history.md:117 to mark the probe rows CAPTURED (point to :141 / probe_summary.md). Out of strict handoff-dir scope; surfaced because bridge-retest is the resolving artifact.
- **Info-loss risk:** none — both lines exist; :141 is authoritative. Reconcile :117 so a future session does not re-run a complete probe.

---

# ISSUE TYPE: STALE (superseded verdicts / framing — annotate, keep substance)

## [HIGH] At-speed-sigma terminal fix-rate ≥0.50 / 0.245-ceiling framing (BLUR-FREE finding is VALID)
- **Location:** `handoff/at-speed-sigma-2026-06-15/REPORT.md:8-22,109-114,150-160,185-203` (incl. MEMORY-DELTA :190-196)
- **Issue type:** stale (two agents; severities low+med — keeping the higher) · **Severity:** med
- **Evidence:** "σ_lat(30) ≈ 0.15 ± 0.04 m → CLEARS the 0.245 m ceiling (×1.6 headroom)" (:16,109), "Terminal last-6 m fix-rate ≈ 5%, NOT ≥0.50" (:20,114,158,193), NEW PARKED "the ≥0.50 terminal-6 m fix-lock looks geometrically hard" (:201). The 0.245 ceiling and ≥0.50 last-6m requirement are the double-count-derived gate MEMORY.md:8 calls moot ("fix-rate ≥0.50 through the last 6 m GEOMETRICALLY UNACHIEVABLE … the policy nearly passes WITHOUT vision fixes").
- **Recommended action:** KEEP — the HEADLINE is DURABLE and high-value and fully banked (MEMORY.md:8 "sim BLUR-FREE so at-speed σ = a PHYSICAL-DRONE item only"; index_vision_estimator.md:267): "the sim is BLUR-FREE (gate-box Laplacian fast/slow 0.68-1.55, no speed trend); σ_lat FLAT 0.13-0.17 over 6→15 m/s; the linear-blur model is FALSIFIED in-sim → at-speed σ is a SIM-TO-REAL physical-drone item." Only annotate the "0.245 ceiling" / "≥0.50 last-6m" references as pre-correction framing (and note σ_lat~0.15 only gets EASIER vs the corrected ~0.15 real bar). Bank-then-archive.
- **Info-loss risk:** none — the blur-free + flat-σ physics is banked; the association/close-range terminal funnel (5% accept at 6-8 m, 99% detected) is durable and consistent with the 12 m floor (preserved via accept-geometry banking).

## [MED] Architecture DESIGN stage table — 0.08 GO bar
- **Location:** `handoff/inc8-architecture-2026-06-15/DESIGN.md:82,86`
- **Issue type:** stale · **Severity:** med
- **Evidence:** S1 GO signal (:82): "inc8_estim_err_inplane_m near gate down toward σ_p0 <= 0.08 m; pass_offset_m p99 down"; (:86) "Stop at the earliest stage that hits σ_p0 <= 0.08 m without destroying the line." The 0.08 bar is retracted (MEMORY.md:8: "REAL bar ~ σ_p0_lat <= 0.15 (p99 <= 0.45)").
- **Recommended action:** Add a one-line SUPERSEDED note at DESIGN.md top: "GO bar 0.08 → CORRECTED to ~0.15 (MEMORY.md:8, 2026-06-19); design plan otherwise stands (look-at primitive + dense centering reward both MERGED 9cecf64)." Then archive — the architecture itself was adopted and built, so this is historical design rationale.
- **Info-loss risk:** none — the design was implemented and is banked (project_rl_increment_history.md:746 S5 recenter; MEMORY.md architecture-pivot bullet). The 0.08 references are the only stale content.

## [MED] Torch-env σ_p0 evaluator report — built, GO rule documents the 0.08 default
- **Location:** `handoff/inc8-sigmap0-torch-2026-06-18/REPORT.md:85-86,106,178` (+ the live tool it documents)
- **Issue type:** stale · **Severity:** med
- **Evidence:** "GO rule identical to the numpy tool (σ_p0_lat <= 0.08 AND lat_p99 <= 0.24)"; (:106) "NO-GO (0.177 > 0.08)"; MEMORY-DELTA "2-axis GO/NO-GO vs 0.08 = PENDING Adroit". The tool `rl/inc8_sigmap0_torch_eval.py` it describes hardcodes `sigma_target=0.08` (confirmed; see the live-code CONTRADICTION finding). Bar retracted to ~0.15.
- **Recommended action:** Add a SUPERSEDED-bar note to the report header pointing to MEMORY.md:8, AND surface the live-code default-0.08 issue (separate finding) so the Adroit run this report sets up does not re-stamp the wrong verdict. Then archive (the instrument build/validation is the keep-worthy part and the build is sound).
- **Info-loss risk:** none for the report (instrument design banked implicitly via the det-retrain torch path). The live tool's 0.08 default is the real exposure — captured as its own finding.

## [MED] inc8 ladder S2/S3 GO gate — 0.08 used as estim_err/σ_p0 gate (+ ladder subsumed)
- **Location:** `handoff/inc8-s2-2026-06-16/REPORT.md:227-228,238,243-247` and `handoff/inc8-s3-2026-06-17/REPORT.md:10,34,127-131`
- **Issue type:** stale · **Severity:** med
- **Evidence:** S2: "The 0.08 threshold requires …", "GO for S2 = estim_err <~ 0.08 AND terminal_pointing > 0". S3: "The GO gate requires estim_err <~ 0.08", "No seed achieves simultaneous estim_err <~ 0.08 AND terminal_pointing > 0". Two staleness layers: (a) 0.08 retracted; (b) these gates conflated estim_err (estimator RMS floor ~0.13, policy-independent) with σ_p0 — and the WHOLE ladder NOT-GO chain was SUBSUMED (MEMORY.md:8: "course-completion was NEVER a GO criterion; the ladder optimized a POINTING proxy on a non-flying policy, success==0 lineage-wide").
- **Recommended action:** Historical ladder runs whose verdicts are SUBSUMED by the recenter result. Add a single dir-level banner to each (S2,S3): "ladder verdict SUBSUMED: course-completion was never gated; see project_rl_increment_history.md:728-747 and MEMORY.md:8; 0.08 bar retracted." Then archive. Keep the raw TB traces (per-seed convergence data) intact inside.
- **Info-loss risk:** none — the full S0→S3 episodic detail incl. the estim_err-floor-not-policy finding is banked at project_rl_increment_history.md:728-740; subsumption stated in MEMORY.md:8.

## [MED] laptop-s18-thrust-lapse — whole-dir VOIDED verdict
- **Location:** `handoff/laptop-s18-thrust-lapse-2026-06-12/WRITEUP.md:11-14,253-261` (TL;DR + MEMORY-DELTA)
- **Issue type:** stale · **Severity:** med
- **Evidence:** S18 WRITEUP: "1. The thrust lapse is REAL and is now fit, integrated into all three plants … 3. … a residual LATERAL (roll-handedness) sign inversion … our force model computes the East thrust acceleration as -33 m/s² where the sim measured +30." SUPERSEDED by frame-audit/WRITEUP.md:36 "4. S18 THRUST LAPSE VOIDED. Re-fit with the true attitude on the same 17 runs: K_eff/K ratio ~ 1.00 … The '15-25% deficit' was the mirrored-b3 projection error." Banked at index_rl_training.md:24 "S18 LAPSE VOIDED — do NOT use --plant lapse/dr_lapse" and src/racer/rl_plant.py:153 "VOIDED BY FRAME-AUDIT 2026-06-12".
- **Recommended action:** Archive the dir. The VOIDED verdict + lapse-constants-annotated-VOIDED are banked (index_rl_training.md:24, index_control_sim.md:51) and merged (rl_plant.py:153). KEEP the `scripts/` subtree inside the archive — `s18_force_frame_selfcheck.py` + `s18_openloop_replay.py` are the ONLY copies of those diagnostic tools (find returns only this handoff path) and are now historical (the diag they were "next steps" for was completed by frame-audit).
- **Info-loss risk:** the 10 s18_*.py diagnostic scripts exist ONLY here. The lapse fit method/curve (LAPSE_FACTOR=[1.0,0.78,0.80,0.92,1.0]) is preserved in rl_plant.py as VOIDED constants; the open-loop-replay method is captured in frame-audit. Preserve by archiving the dir intact (keep scripts/), not pruning.

## [MED] PnP cov-inflation K=2.0 verdict (UNDER-DELIVERS) — overturned next day
- **Location:** `handoff/shadowpc-cov-finishhold-verify-2026-06-09/HANDOFF.md`
- **Issue type:** stale · **Severity:** med
- **Evidence:** Primary recommendation "do NOT ship K=2.0; K=1.0 lowest leak" was OVERTURNED one day later by assoc-flipfix HANDOFF.md §K=2.0: "With the depth sanity in place that penalty vanishes: K=1.0 and K=2.0 both leak the SAME 2 fixes … no rollback needed." VISION-PKG2 shipped K=PNP_FIX_COV_INFLATION=2.0. Memory reflects the FINAL state (K=2.0 + 0.40 floor + 32 m cap, project_phase2_rl_vision_decisions.md:93,144).
- **Recommended action:** Bank-then-archive. The cov-inflation conclusion is correctly superseded in memory; finish_hold banked (project_phase2); the sim-side flight regression (under gate-1/backflip on build 1.0.3364) was resolved by start-ramp (6/6 recovered). cov_sweep/ holds 30 regenerable per-gate per-K char JSONs. Archive dir.
- **Info-loss risk:** none — the report's own (now-stale) "do not ship K=2.0" verdict is harmless because memory carries the corrected K=2.0 decision; do NOT re-bank this report's recommendation as current.

## [MED] 2nd-order rate-loop re-sysid (cc6921d) — STRUCTURALLY SUPERSEDED
- **Location:** `handoff/shadowpc-2ndorder-resysid-2026-06-10/WRITEUP.md`
- **Issue type:** stale · **Severity:** med
- **Evidence:** Headline (amplitude-dependent 2nd-order wn/zeta + PI-windup) overturned by the characterize-sweep next day. Memory records it explicitly: project_phase2_rl_vision_decisions.md:163 "inc8 … 2nd-order inner-loop re-system-ID (cc6921d, fable, 2026-06-10) — STRUCTURALLY SUPERSEDED by the characterize-sweep (510da24)" and :164 "Kept as the record of WHY the windup/2nd-order picture was wrong (do NOT re-litigate)", :179 cites this dir.
- **Recommended action:** Bank-then-archive. Already explicitly marked superseded in memory with a "do NOT re-litigate" note citing this dir. `fit_2nd_order.py` is cited as the sweep-loader provenance in scripts/rate_sysid.py:8,630 (doc comment, not a load) — if archiving, keep the rate_sysid.py comment pointing wherever it lands.
- **Info-loss risk:** LOW — the WHY-it-was-wrong is the value, preserved in memory; keep the fit_2nd_order.py provenance pointer in rate_sysid.py.

## [MED] bimodal-char — "post-gate-3" inference SUPERSEDED (relocated to start→g0 cold-start)
- **Location:** `handoff/shadowpc-bimodal-char-2026-06-13/WRITEUP.md`
- **Issue type:** stale · **Severity:** med
- **Evidence:** This report concluded the bimodal gap "lives in the POST-gate-3 segment" (§2/§5). Overturned: project_rl_increment_history.md:558 "SUPERSEDED — PER-TICK RE-LOCATION (P2-OFFLINE-ANALYSIS, 2026-06-13, DEFINITIVE): the ENTIRE 1.48 s bimodal gap is in the start→g0 segment (104% of total) … PRE-GATE-0 COLD-START ARTIFACT … The earlier post-gate-3 inference was an error: constant gate-3 SPEED ≠ constant gate-3 TIME." Durable verdict (bimodal = physics-state HOME-reset, NOT sim_t0 warmup; 11.45 s baseline; fresh-reset eval protocol) banked at :554-562.
- **Recommended action:** Bank-then-archive. Its erroneous "post-gate-3" location is explicitly corrected in memory (cited as "(initial)"); the durable findings are banked. Do NOT re-bank the "post-gate-3" inference.
- **Info-loss risk:** none — the corrected (start→g0) location and durable findings are in project_rl_increment_history; the report stands as the "(initial)" record.

## [MED] laptop-inc8-binding-gate-verify — gate-4 0.155 @r=0.38 + 1.46 m registration (refuted)
- **Location:** `handoff/laptop-inc8-binding-gate-verify-2026-06-13/WRITEUP.md:37,92-98` (also :70-73)
- **Issue type:** stale (catalogued `duplication`/med by one agent; the refuted-claim half is stale — kept here) · **Severity:** med
- **Evidence:** :37 "gate-4 … margin r=0.38 (primary) 0.155 ← BINDING"; :70-73 "track_map gate-3 D-center is mis-registered by ~1.46 m." The gate-4 0.155@r=0.38 framing IS banked AND consistent with the correction (it is the r=0.38 worst-case budget identity; index_vision_estimator.md:52). The 1.46 m track_map claim was REFUTED: index_vision_estimator.md:213 "'gate-3 crosses 1.46 m above (map) center' = BOTTOM-vs-OPENING reference-frame confusion; true miss is 0.017 m" and :187 "MAP IS GOOD VERTICALLY: direct δ_map ≈ 0".
- **Recommended action:** Bank-then-archive. The binding-gate=gate-4 ranking + the 0.155@r=0.38 budget + the contact_true_eval per_gate_margin_stats pass_band bug-fix are banked (index_rl_training.md:40,50,52). The Task-4 "1.46 m registration error" conclusion is STALE/refuted — ensure the refutation pointer (δ_map≈0; bottom-vs-opening) is the live truth (it is, index_vision_estimator.md:213) before archiving. Add a one-line stale-banner note. Then archive.
- **Info-loss risk:** the archived WRITEUP asserts "1.46 m map mis-registration" as a live conclusion (now refuted). Rely on the banked refutation as SSOT; the archive remains valid as the historical contact_true_eval tool-build record (the pass_band radius-invariance bug-fix is the durable contribution).

## [MED] Data-staging manifest — stale checkout path + pre-supersede artifact pointers
- **Location:** `handoff/data-staging-2026-06-17/MANIFEST.md:4,28-74,84-99`
- **Issue type:** stale · **Severity:** med (one agent low, one med — keeping med)
- **Evidence:** Manifest targets "Checkout: C:\Users\Fengy\Downloads\Projects\Anduril (main checkout)" (:4, the stale pre-correction tree) and stages "VQ2 Round-1 detector best.pt (champion)" (:47, S4) — but Round-1 was SUPERSEDED by the clean ensemble (282abb9). The inc8-best = S2-seed2 pointer (:14) and the gh artifact-pipe mechanism are durable.
- **Recommended action:** Bank-then-archive. Durable content (inc8-best = S2 seed2 staged+sha-verified; Adroit pull_file.py + MSYS_NO_PATHCONV=1 gotcha; origin/vq2-data = synthetic-only) is captured in its own MEMORY-DELTA (:84-99) and the artifact-pipe is banked in project-fullstack-burn. Archive once its MEMORY-DELTA is confirmed banked. Note the S4 "pull Round-1 champion" instruction is superseded by the clean ensemble.
- **Info-loss risk:** LOW — the gh artifact-pipe recipe and inc8-best provenance are banked. Preserve the paste-and-go ShadowPC pull commands only if the ShadowPC corpus still needs staging.

## [MED] VQ2 overnight — superseded intermediate REPORT.md (retracted data hypothesis)
- **Location:** `handoff/vq2-overnight-goal-2026-06-18/REPORT.md:3,63-81,100-127`
- **Issue type:** stale · **Severity:** med
- **Evidence:** REPORT.md marked "Status: IN PROGRESS" (:3); central §2b hypothesis (:63-81) — "the quarantined low-res/degraded renders are load-bearing precision data … Excluding them is the likely hidden reason every clean lever lost" — REFUTED by same-dir REPORT_CLEAN.md:99 ("removing the dirty data washed; it was never load-bearing. My earlier load-bearing substandard data story was wrong") and COMMANDER_REPORT.md:32 ("a win that turned out to be a mirage … timestamp forensics proved it"). REPORT.md's "NEW CHAMPION (ensemble)" trained on "RECONSTRUCTED champion data" (confounded), superseded by clean_champion.
- **Recommended action:** Within this dir, COMMANDER_REPORT.md + REPORT_CLEAN.md are the AUTHORITATIVE final (match memory exactly: clean ensemble 282abb9, loss-lever NOT data, confound-kill p=1.0). Mark REPORT.md as SUPERSEDED-BY-REPORT_CLEAN (its §2b explicitly retracted) and retire the new_champion/ artifacts as REPORT_CLEAN.md:123 recommends. Keep dir; archive/prune the confounded new_champion/ weights+README per the report's own instruction.
- **Info-loss risk:** none — the final verdict is in COMMANDER_REPORT.md + REPORT_CLEAN.md (same dir) and memory/MEMORY.md. The caught-confound scientific record is preserved in COMMANDER_REPORT.md wall #6. Preserve new_champion/README.md (corrected-provenance record) even if weights are pruned.

## [MED] accept-geometry — 0.245 ceiling / 0.10 budget / fix-rate≥0.50-through-6m framing
- **Location:** `handoff/accept-geometry-2026-06-15/REPORT.md:23-27,73,84-87,107-117,161-164` (incl. MEMORY-DELTA :161-164)
- **Issue type:** stale (two agents; med+low — keeping med) · **Severity:** med
- **Evidence:** ":84 crosses the 0.10 budget at ~12 m and the 0.245 margin ceiling at ~11 m"; ":113 pointing alone — even ideal — cannot satisfy 'fix-rate ≥0.50 through the last 6 m / 0.15 s'"; ":24-27 fix-rate ≥0.50 through the last 6 m is geometrically UNACHIEVABLE"; ":73 ceiling ~0.23-0.245 m for r=0.30 @ fr=0.50". The 0.245 ceiling and ≥0.50-terminal-lock are the pre-correction bar; the geometric 12 m accurate-fix floor finding is independent and valid.
- **Recommended action:** Add a note that the 0.245/0.10 budget and ≥0.50-last-6m framing are the superseded pre-2026-06-19 bar; the ~12 m accurate-fix floor + accept_rlo 16.2→12 re-fit are the valid, banked deliverables (index_vision_estimator.md:273-277, index_rl_training.md:113,117). Archive after the note.
- **Info-loss risk:** none — the 12 m PnP floor, accept_rlo re-fit, near-field-degradation mechanism, 6-8 m dead zone, and the χ²-tightness near-field PARKED item are banked. Only the margin-ceiling/terminal-lock framing is stale.

## [MED] ultracode-estimator-racespeed — 0.05 m (margin/3) and 0.155 margin as hard targets
- **Location:** `handoff/ultracode-estimator-racespeed-2026-06-13/REPORT.md:14,41-46,108-130` (incl. MEMORY-DELTA :269-294)
- **Issue type:** stale · **Severity:** med
- **Evidence:** ":14 the bar ③ derived as margin/3 (<0.05 m 1-σ in-plane)" tests everything against the 0.155 m margin (the r=0.38 double-counted instance). The 0.05 m = 0.155/3 bar is the lineage the correction supersedes. HOWEVER the PRIMARY verdict — absolute world-frame nav NO-GO, gate-relative observation = THE fix — is VALID and banked.
- **Recommended action:** Add a note that the 0.05 m (margin/3) and 0.155 m hard bars are the superseded pre-2026-06-19 framing; the architecture verdict (absolute NO-GO; gate-relative +L = the fix; RewindKF; relinnov gate) is intact and was BUILT/MERGED (C2 chain 05ed750). Archive after the note.
- **Info-loss risk:** the gate-relative architecture, absolute-NO-GO, dead-ends (cadence/cov-floor/detector), and speed-ladder coupling are banked/merged. Only the strict-bar framing is stale.

## [MED] ultracode-gate-relative-pipeline-design — CANNOT-SETTLE-OFFLINE headline + p90<0.155 selection gate
- **Location:** `handoff/ultracode-gate-relative-pipeline-design-2026-06-13/REPORT.md:25-37,49-63,116-120,142` (incl. MEMORY-DELTA :199-217)
- **Issue type:** stale · **Severity:** med
- **Evidence:** ":25 the gate-4 0.155 m WORST-CASE margin does NOT close offline in ANY regime"; ":116 the inc8 SELECT gate 'gate-4 in-plane p90 < 0.155 m @ r=0.38'". The p90<0.155 selection bar is the double-counted r=0.38 margin; per the correction real clearance 0.37-0.47 → the "does-not-close" headline is superseded. BUT the BLUEPRINT (gate-relative +L obs, 20-dim obs, RewindKF, relinnov χ²=13.82, +L sign footgun) is all CURRENT/MERGED.
- **Recommended action:** Add a CORRECTION note to §1 TL;DR (#2/#3) and risk-table risk #1 that the p90<0.155 "does-not-close" verdict used the double-counted bar (corrected 2026-06-19); the BLUEPRINT is intact and merged (C2 chain 05ed750, obs contract frozen). KEEP the dir (BLUEPRINT.md is the design-of-record for the merged estimator chain + inc8 retrain) — classify KEEP (reference) rather than archive.
- **Info-loss risk:** the BLUEPRINT (estimator chain, obs contract, +L sign, retrain spec) is the design-of-record for merged code and live inc8 work — must be preserved. Only the margin-closure headline is stale.

## [LOW] inc8 ladder S0/S1 next-steps reference 0.08 (subsumed-ladder context)
- **Location:** `handoff/inc8-sigmap0-rc1-2026-06-18/REPORT.md:80` (MEMORY-DELTA:110)
- **Issue type:** stale · **Severity:** low
- **Evidence:** NEXT: "measure σ_p0 vs 0.08 and run the RW_TC sweep"; MEMORY-DELTA: "re-measure σ_p0 vs 0.08 + sweep". The 0.08 reference is retracted. The CORE verdict (σ_p0 UN-MEASURABLE because rc1 seeds reach gate-4 ≤1/200 from simstart = an EVAL-faithfulness/start-distribution blocker, not a policy regression) is CORRECT and triggered the eval-lookat/eval-pitch port chain.
- **Recommended action:** Add a one-line note ("vs-0.08 references retracted → ~0.15, MEMORY.md:8") then archive. The un-measurability/start-distribution finding is captured at project_rl_increment_history.md:743-744.
- **Info-loss risk:** none — the un-measurable/instrument-vs-policy finding is banked at project_rl_increment_history.md:743-744.

## [LOW] obs-parity diagnosis — "σ_p0 stays NO-GO/UN-MEASURABLE" ambiguity
- **Location:** `handoff/inc8-obs-parity-2026-06-17/REPORT.md:141,164` (NEXT/MEMORY-DELTA)
- **Issue type:** stale · **Severity:** low
- **Evidence:** "σ_p0 stays UN-MEASURABLE / NO-GO until a policy completes the course." The "UN-MEASURABLE-until-a-lap" part is contextually CORRECT (the recenter run later unblocked it); the bare "σ_p0 … NO-GO" reads ambiguously against the corrected bar. CORE finding (POLICY-GAP not instrument-gap; obs[0:17] parity 7.6e-6; misses gate-0 on truth pose) is CORRECT and banked.
- **Recommended action:** Archive with a light note ("NO-GO here means UN-MEASURABLE pre-flight, resolved by the recenter first-flight; 0.08 bar retracted"). Core finding banked at project_rl_increment_history.md:744.
- **Info-loss risk:** none — the policy-gap diagnosis (3 hypotheses ruled out, parity proof) is banked at project_rl_increment_history.md:744.

## [LOW] in-training course-completion check — same UN-MEASURABLE-vs-NO-GO ambiguity
- **Location:** `handoff/inc8-intraining-coursecheck-2026-06-17/REPORT.md:67,85`
- **Issue type:** stale · **Severity:** low
- **Evidence:** "σ_p0 stays NO-GO until a policy completes a lap" / "σ_p0 NO-GO until a lap completes." Same ambiguity. CORE finding (confound-free PURE POLICY-GAP: success_rate==0 flat across all 4 runs incl. the S2 parent; ladder optimized pointing on a non-flying policy) is CORRECT and decisive evidence behind the recenter fix.
- **Recommended action:** Archive with the same light note. The PURE-POLICY-GAP confirmation + the craft lesson ("gate on the END OBJECTIVE not a PROXY") are banked at project_rl_increment_history.md:745 (craft lesson routed to COMMANDER.md).
- **Info-loss risk:** none — banked at project_rl_increment_history.md:745 incl. the craft lesson.

## [LOW] recenter run — THE current-truth milestone, single stale line
- **Location:** `handoff/inc8-recenter-run-2026-06-17/REPORT.md:139`
- **Issue type:** stale · **Severity:** low
- **Evidence:** Single stale line in an otherwise current report: NEXT item 1 "Gate-4 closure needs σ_p0_lat <~ 0.08 m." Everything else (FLIGHT RECOVERED, 3/3 seeds flew, success_rate ~0.50-0.54, through-centering = root-cause fix) is CURRENT TRUTH and fully banked at project_rl_increment_history.md:747.
- **Recommended action:** KEEP this dir (live milestone report, not yet superseded) but correct the single line to "~0.15 m (MEMORY.md:8); gate-4 is a reach/pass-rate problem." Lowest-priority edit; safe to archive later once the det-retrain follow-on fully lands.
- **Info-loss risk:** none — first-flying-inc8 result banked at project_rl_increment_history.md:747 and MEMORY.md:8.

## [LOW] Weight-iteration NO-GO chain (drove the architecture pivot, now subsumed)
- **Location:** `handoff/inc8-repilot-window-2026-06-15/REPORT.md:255,305,330,348,357-371` and `handoff/inc8-conversion-diag-2026-06-15/REPORT.md` (verdict + MEMORY-DELTA:99-104)
- **Issue type:** stale · **Severity:** low
- **Evidence:** repilot: "Three iterations BRACKET it → NOT a weight problem … STOP per the task → ARCHITECTURE change"; conversion-diag: "REWARD↔ACCEPT RANGE ANTI-ALIGNMENT". Correct for their moment and DROVE the (now-adopted) architecture pivot, but the entire "pointing proxy" framing is SUBSUMED (MEMORY.md:8: "4 reward-WEIGHT iterations ALL NO-GO ⇒ it is reward ARCHITECTURE not weights"). The reports read as live next-steps (band-pass reshape, fix-driven fallback) overtaken by the through-centering recenter fix.
- **Recommended action:** Archive both with a one-line banner: "weight-iteration NO-GO chain → drove the architecture pivot, now SUBSUMED by the recenter through-centering fix 9cecf64; see MEMORY.md:8 / project_rl_increment_history.md:746." Keep the per-job TB traces (provenance for "why weights cannot fix it") inside.
- **Info-loss risk:** none — the 4-iteration NO-GO summary and architecture-pivot rationale are banked (MEMORY.md:8 architecture-pivot bullet; index_rl_training.md:113,116).

## [LOW] laptop-togt-bound — 4.27/4.55 s time-optimal bound FALSIFIED
- **Location:** `handoff/laptop-togt-bound-2026-06-10/WRITEUP.md:1-3,33-44`; results_table.md
- **Issue type:** stale · **Severity:** low
- **Evidence:** :3 "the ceiling is ~4.3 s"; :39 "ref_circle (exported) … 4.55". Flagged FALSIFIED/superseded in memory: index_rl_training.md:31 "4.27/4.55 s bounds FALSIFIED (linear-plant fiction; real v_max ~39 m/s v² drag wall)" and :86 "the banked ~4.6-4.7 s … was the FULL-ATTITUDE/TOGT optimum — RATE-INFEASIBLE … UPRIGHT, rate+collective-feasible lap ≈ 7.9-8.5 s". The §8 corrected-aero 4.71 s is also in the WRITEUP. The TOGT pipeline (scripts/togt/, twin_track_reference.py, rl/reference_line_vq1.json, rl/reference_line.py) is merged.
- **Recommended action:** Bank-then-archive. Bound result + falsification + full §TOGT-BOUND detail are banked (project_phase2_rl_vision_decisions.md:302-332, cites this WRITEUP as "Source of truth"). PRUNE the 167-file `cases/` subtree (88 yaml + 33 json + 22 txt + 22 csv = raw IPOPT per-case solver dumps) — reproducible via scripts/togt/ in main; keep WRITEUP.md + results_table.md. Then archive.
- **Info-loss risk:** the per-case yaml/json/csv dumps are reproducible from scripts/togt/gen_cases.py+run_cases.sh (merged) per WRITEUP §6 "Reproduce". Summary numbers are in results_table.md + WRITEUP tables + banked memory. Preserve the two .md; the case dumps are safe to consolidate-away.

## [LOW] laptop-inc5-aero-retrain — inc5 superseded twice (inc6→inc7)
- **Location:** `handoff/laptop-inc5-aero-retrain-2026-06-11/WRITEUP.md:1-21` (OUTCOME)
- **Issue type:** stale · **Severity:** low
- **Evidence:** Ships "rl/checkpoints/stage1_inc5_actor.pth … inc5 supersedes inc4 as THE transfer candidate". inc5 superseded twice over: inc6 (mixer, S17) then inc7 (LIVE-CONFIRMED best). inc7-eval WRITEUP:225 "SUPERSEDES inc6 as current best". Aero machinery (--plant aero, +dynamics.dr_aero) merged. inc5's roll-p90-145° style-vs-speed lesson banked at project_phase2_rl_vision_decisions.md:600.
- **Recommended action:** Bank-then-archive. Durable lessons (aero retrain = correctness fix; ~2.3 s/lap style-vs-speed cost; inc4 fails 53.9% on corrected plant) banked. The inc5 checkpoint is a retired-lineage artifact. Archive.
- **Info-loss risk:** none material — the rw_tilt sweep lesson and inc4-on-aero failure datum are banked (project_phase2_rl_vision_decisions.md §INC5 / §600). Checkpoint md5s recorded in the WRITEUP which stays in the archive.

## [LOW] live-confirm — Facts 4/5 (0.85 droop, ~70% slowdown) overturned by crab-diag
- **Location:** `handoff/shadowpc-live-confirm-2026-06-12/WRITEUP.md`
- **Issue type:** stale · **Severity:** low
- **Evidence:** Fact 4 ("0.85 rate-gain droop") and Fact 5 ("~70% slowdown") OVERTURNED same day by crab-diag, recorded: project_rl_increment_history.md:345 "Fact 2 — ~70% slowdown DISSOLVED (clock artifact; supersedes live-confirm §Fact 5)" and :373; crab-diag MEMORY-DELTA "Supersedes LIVE-CONFIRM fact 4/5". Durable findings (inc6 bridge 2/2 FINISHED, gate-3 = new barrier, spawn artefact after gate-3 crash) banked.
- **Recommended action:** Bank-then-archive. Two erroneous facts explicitly superseded by crab-diag IN MEMORY; durable bridge-confirm + gate-3-barrier findings banked. Do NOT re-bank facts 4/5 as current.
- **Info-loss risk:** none — durable findings banked; the two stale facts flagged-superseded in project_rl_increment_history.

## [LOW] ultracode-planning-togt-s2 — estimator-binding-risk on 0.47m=3.0×-0.155-margin / KF<0.05m
- **Location:** `handoff/ultracode-planning-togt-s2-2026-06-13/REPORT.md:6,21,93,107,113,149`
- **Issue type:** stale · **Severity:** low
- **Evidence:** ":93 perception East σ = 0.47 m = 3.0x the 0.155 m gate-4 binding margin"; ":107 drive the KF to <0.05 m 1-σ". Rest on the 0.155 m double-counted margin and the 0.05 m=margin/3 bar. The S2 ARCHITECTURE verdict (staged_monolithic_then_decomposed, tilt-envelope-not-planner) is VALID and banked.
- **Recommended action:** Add a note that the 0.155 m / 0.05 m gate-4 numbers used to rank "estimator is the binding risk" are the superseded bar; the S2 architecture decision (banked: index_rl_training.md S2 DECIDED) and the tilt-envelope lap-time story are unaffected. Bank-then-archive.
- **Info-loss risk:** the S2 decision, tilt-vs-lap-time waterfall, and TOGT bound are banked. Only the gate-4-margin-as-binding-risk numbers are stale.

## [LOW] p2-inc8-rl + trainport — binding-vertical-σ thesis (σ_vert 0.28 ≫ σ_lat 0.10) + p90<0.155
- **Location:** `handoff/p2-inc8-rl-2026-06-14/REPORT.md:127-130` (MEMORY-DELTA :176); `handoff/p2-inc8-rl-trainport-2026-06-14/REPORT.md:111-115,155-157`
- **Issue type:** stale · **Severity:** low
- **Evidence:** p2-inc8-rl :176 "Emulated gate-4 in-plane is VERTICAL-σ-dominated (σ_vert 0.28 ≫ σ_lat 0.10) … cold@bias0 p90 0.392 … one-signed bias U[0,0.19] survives the +L fix → p90 0.457 (the case-(b) floor)". Judged against the double-counted ~0.235/0.155 budget + "binding case-(b)" framing. The det-retrain (MEMORY.md:8) later showed measured σ_p0 0.15-0.20 is MARGINAL-PASS and the policy nearly passes WITHOUT vision fixes (fix_rate=0-wall moot).
- **Recommended action:** Add a one-line note that the p90<0.155 selection numbers + binding-case-(b)-floor framing predate the 2026-06-19 correction and the det-retrain result. KEEP both dirs — estimator_emul.py (p2-inc8-rl, MERGED 714-test) and the inc8 torch train-env port (trainport, MERGED b0b322e) are live infrastructure; the reports document merged code. Bank-then-archive the report text; keep the code references.
- **Info-loss risk:** estimator_emul.py and the inc8 torch env are MERGED/live (b0b322e); the GREEN escape-hatch + parity gates are banked. Only the margin-selection numbers are stale.

## [LOW] p3-simops-empirical — boresight δ_map-vs-ε_vert INCONCLUSIVE (later RESOLVED in-dir)
- **Location:** `handoff/p3-simops-empirical-2026-06-14/REPORT.md:9,41-50,80-90` vs `FORM_RESOLUTION.md:1-5`
- **Issue type:** stale · **Severity:** low
- **Evidence:** REPORT.md:9 verdict "INCONCLUSIVE → live static test required" for the δ_map-vs-ε_vert split. Companion FORM_RESOLUTION.md:3 RESOLVES it: "ε_vert is a METRIC vertical offset … Magnitude −0.25 m … sign gate-DOWN" — banked (index_vision_estimator.md:232, boresight DEPLOYED 5764291). The INCONCLUSIVE REPORT reads as live but is superseded by its own dir-mate and by the δ_map discriminator choosing case-(b).
- **Recommended action:** Add a one-line pointer at the top of REPORT.md to FORM_RESOLUTION.md (FORM=METRIC −0.25, RESOLVED) and the δ_map discriminator (case-b). Bank-then-archive — the resolution is banked.
- **Info-loss risk:** none — the FORM=METRIC resolution is banked; the inconclusive offline range-resolve is a documented dead-instrument lesson preserved in FORM_RESOLUTION's caveat.

## [LOW] ultracode-vision-case-c — incidental 0.155 m references
- **Location:** `handoff/ultracode-vision-case-c-2026-06-13/REPORT.md:237,358-366`
- **Issue type:** stale · **Severity:** low
- **Evidence:** Primary findings (gate-3 1.46m FALSIFIED, all-6-gate registration ≤0.37m, RewindKF/OOSM, range-anisotropic R, latency P0-P3 plan) banked/merged. References the "0.155 m geometric margin" as the gate-4 target (now superseded), but those are incidental to the registration/latency deliverables.
- **Recommended action:** Add a one-line note that the incidental "0.155 m gate-4 margin" references are the superseded bar (gate-3 falsification + registration findings unaffected). Bank-then-archive — registration falsification banked (index_vision_estimator.md:213), latency/OOSM merged (C2 chain 05ed750, kf-pfloor #74).
- **Info-loss risk:** none — registration falsification, OOSM, latency split, and P0-P3 build plan are banked/merged.

## [LOW] c2-estimator-chain — G3 margin verdict line (CANNOT-SETTLE-OFFLINE)
- **Location:** `handoff/c2-estimator-chain-2026-06-13/REPORT.md:11-13,71-85,140`
- **Issue type:** stale (one agent `duplication`, one `stale`; the verdict line is stale) · **Severity:** low
- **Evidence:** ":11-13 rel p90 0.197 m > 0.155 m worst-case @ r=0.38 → CANNOT-SETTLE-OFFLINE"; G3 table header (:74) "MARGIN(r)=0.535-r"; MEMORY-DELTA (:140) "rel p90 0.197 > 0.155 @r=0.38 → CANNOT-SETTLE-OFFLINE survives". The 0.535-r / 0.155 budget is the double-count; the margin VERDICT is stale (now a comfortable pass vs the ~0.45 p99 real bar). The build (RewindKF OOSM, gate_relative_inplane_fix +L anisotropic cov, relinnov χ²=13.82, GATE_REL_INPLANE_SIGMA=0.265, estimator_obs seam, NavState cov export, gated-off byte-identical) is DONE+MERGED (05ed750) and banked.
- **Recommended action:** Bank-then-archive. Add a one-line note that the "p90 0.197 > 0.155 → CANNOT-SETTLE" line used the superseded bar. Cite index_vision_estimator.md:332 / MEMORY.md C2 line. KEEP the build content (headline, correct, banked).
- **Info-loss risk:** none — the entire C2 build is merged and banked; the rel E_bias −0.000 / RMS 0.131 measurement is durable, just don't compare it to the 0.155 budget.

## [LOW] MonoRace synthesis — residual 0.08 reference
- **Location:** `handoff/monorace-synthesis-2026-06-18/REPORT.md:100,117`
- **Issue type:** stale · **Severity:** low
- **Evidence:** Adoptable-ideas table row 1 (:100): "raises chance a policy holds σ_p0_lat<=0.08 + completes a lap"; MEMORY-DELTA #2 (:117) references "success_rate=0 pay-back under-centring" tied to the same. The σ_p0_lat<=0.08 target is the retracted double-count bar.
- **Recommended action:** Keep the report (high-value, durable, banked across 9 memory files). Note the single "<=0.08" phrase is the stale bar — the underlying idea (persistent ABSOLUTE cross-track centering penalty layered on the delta-form through_centering_reward, to kill the success_rate=0 pay-back) is correct and bar-independent. No archive; minor annotation only.
- **Info-loss risk:** none — the 5 adoptable MonoRace ideas are banked in project_parked_backlog/index files. Only the 0.08 numeric is stale and non-load-bearing here.

## [LOW] system-id R2 carry-forward now closed
- **Location:** `handoff/system-id-2026-06-18/REPORT.md:27,45,56,278`
- **Issue type:** stale · **Severity:** low
- **Evidence:** Pair-9 carry-forward (:45,56): "Production obs[17:20] builder is absent in src/racer … When promoted it MUST divide NavState.nav_inplane_sigma by sqrt(2)". This GAP was CLOSED by deploy-obs20 (handoff/deploy-obs20-2026-06-19/REPORT.md + memory "DONE dc56ce7"), which productionized confidence_triple with the /√2 in the builder.
- **Recommended action:** Keep (system-id MERGED bc182f9, durable, fully banked, 86 pinning tests). No change to the report (it correctly registered the gap as of 2026-06-18). Note for the commander that this carry-forward is now resolved by deploy-obs20 → should not be re-surfaced as open. Classify: keep.
- **Info-loss risk:** none — registration merged and banked; the resolved carry-forward is tracked in deploy-obs20's banking. wf_investigate.js is scratch (safe to leave).

## [LOW] Stack-review strategic snapshot — aged datapoints
- **Location:** `handoff/stack-review-2026-06-10/REPORT.md:6,16,20`
- **Issue type:** stale · **Severity:** low
- **Evidence:** 2026-06-10 review (flags "fable background agent" provenance) cites aged anchors: "VQ1 35.3 s" lap time (:16), plant authority "~11 rad/s (was modeled 7.85)" (:16, measurement-in-progress), and frames upgrades as future ("build the offline mapper skeleton now", "add a time-optimal line generator") that have since been actioned (TOGT bound done, MonoRace synthesized, photoreal VQ2 detector built).
- **Recommended action:** Keep as a strategic snapshot/rejected-options ledger. Its enduring value is the (c) "examined and rejected — do not re-litigate" list (Depth Anything 3, RT-DETR/D-FINE/RF-DETR, GTSAM-in-loop, OpenVINS VIO, Isaac Lab, Crazyflow, MPPI, G&CNets, pixel-to-control) banked in project_parked_backlog (#33 YOLO26, #64 GTSAM) and reference_prior_art. No archive needed; if pruning, first confirm the full ~12-item rejected-options list is mirrored into the parked register.
- **Info-loss risk:** LOW — the rejected-options ledger is the load-bearing content; confirm all ~12 rejections are in project_parked_backlog before any archive. Aged lap-time/authority numbers are superseded by later sysid and non-load-bearing here.

---

# ISSUE TYPE: DUPLICATION (banked verbatim elsewhere — bank-then-archive, watch for fixtures)

## [HIGH] First-contact wire spec / canonical track map — HARD ACTIVE DEPENDENCY
- **Location:** `handoff/shadowpc-firstcontact-2026-06-02/` (HANDOFF.md + track_map.json)
- **Issue type:** duplication · **Severity:** high
- **Evidence:** HANDOFF.md content (ATTITUDE.pitch sign-inversion → ODOMETRY quat, map broadcast-once-at-level-load, 2.72 outer / 1.5 inner, rates 97/75/120 Hz, ~14× video dedup, ACRO-only) is banked verbatim in memory/reference_sim_interface.md:19-66. BUT `track_map.json` is a HARD ACTIVE DEPENDENCY: `grep firstcontact-2026-06-02/track_map --include=*.py` = 26 files including tests/test_navigator.py, src/racer/gate_mapper_synth.py, rl/peregrine_course.py, scripts/characterize_perception.py, scripts/task2_gate_pnp.py, scripts/twin_fly_course.py — and it is the ONLY copy of the canonical 6-gate course map (`find -name track_map*.json -not -path */handoff/*` = empty).
- **Recommended action:** **KEEP the directory.** The HANDOFF.md prose is fully banked (could be pointer-collapsed) but the dir CANNOT be archived/moved: track_map.json is the canonical course map loaded by 26 scripts+tests. Disposition = KEEP (load-bearing data fixture).
- **Info-loss risk:** Archiving/moving would break tests/test_navigator.py + ~25 scripts (no other copy). Preserve in place; if ever relocated, move track_map.json to a stable data/ path and update all 26 references first.

## [HIGH] Followups — gate-0 PnP map-anchor + balloon verdicts + SYSID FIXTURE
- **Location:** `handoff/shadowpc-followups-2026-06-05/` (TASK2_PNP_VERDICT.md, TASK3_BALLOON.md, UNDERSTANDING.md, sysid/, task2_frames/)
- **Issue type:** duplication · **Severity:** high
- **Evidence:** Task2 (map z = bottom EDGE, corner_to_center z-only correct, bottom-left-corner REFUTED, ~3.6° yaw bias) and Task3 ("we own thrust in CTBR; no sim auto-thrust; balloon = our control law") are banked (corner_to_center in 8 memory files; "own thrust/auto-thrust/balloon" in index_control_sim.md + project_ctbr_control_sysid.md; yaw-bias in index_vision_estimator.md). BUT two SUBDIRS are HARD fixtures: `sysid/` loaded by tests/test_twin_fit.py:16, scripts/fit_twin.py:17, documented as the source in src/racer/twin_fit.py:1-3; `task2_frames/` loaded by scripts/characterize_perception.py:59, scripts/task2_gate_pnp.py:37, src/racer/vision/blender_gen/contract.py:59.
- **Recommended action:** **KEEP the directory** (do NOT archive). The three .md verdicts are fully banked (could be pointer-collapsed) but sysid/ and task2_frames/ are committed fixtures consumed by the test suite (test_twin_fit) and production scripts. Disposition = KEEP (load-bearing fixtures).
- **Info-loss risk:** Archiving would break tests/test_twin_fit.py (loads sysid/) and orphan scripts/task2_gate_pnp.py + characterize_perception.py (load task2_frames/). Preserve in place.

## [HIGH] reVERIFY rung23 — VQ1 FLOOR + canonical per-gate misses + start transient
- **Location:** `handoff/shadowpc-reverify-2026-06-07/rung23/REPORT.md`
- **Issue type:** duplication · **Severity:** high
- **Evidence:** FIRST valid sim-recognized 6/6 finish (~35.3 s) = the VQ1 floor, plus canonical per-gate in-plane misses (g0 0.37, g1 0.05, g2 0.06, g3 0.03, g4 0.10, g5 ~0.71), the non-deterministic start transient (18°→54° bank — later root-caused/fixed by the launch-ramp), and the "always-on relay (not static-only)" refinement. "VQ1 floor" banked across 10 memory files.
- **Recommended action:** Bank-then-archive. VQ1-floor + start-transient + always-on-relay all banked. course_60s_extract.json is the extract of the gitignored canonical recording `20260607_194615_course_60s` (frames.py:65 and fly_rl.py:560 cite the recording, NOT this extract); the extract is consumed only by the dir's own scratch/gate_analyze.py. No hard code dep. Archive dir.
- **Info-loss risk:** Low: the per-gate-miss numbers are the canonical course1 reference — confirm they survive in project_ctbr_control_sysid.md before pruning the extract JSON; the verdict prose is banked.

## [HIGH] postfix-dataset — 8-run debug_obs package (AUDIT TEST FIXTURE)
- **Location:** `handoff/shadowpc-postfix-dataset-2026-06-12/` (MANIFEST.md, debug_obs_8runs.zip, extracted/)
- **Issue type:** duplication · **Severity:** high
- **Evidence:** MANIFEST is a pure data-package index (no novel findings beyond live-confirm). BUT extracted/ is a GIT-TRACKED HARD TEST FIXTURE: tests/_audit_io.py:88 "postfix: ROOT/handoff/shadowpc-postfix-dataset-2026-06-12/extracted" and tests/test_diagnose_session.py:263 load it; _audit_io.py:98-105 notes the postfix recordings are git-tracked (used as the R_y(π)-mirror positive control). extracted/ confirmed populated (8 run dirs).
- **Recommended action:** **KEEP the directory** (do NOT archive). extracted/ is a committed audit fixture consumed by tests/test_diagnose_session.py + tests/_audit_io.py. Disposition = KEEP (load-bearing test fixture). The MANIFEST.md prose adds no memory-worthy finding (its source is live-confirm, banked).
- **Info-loss risk:** Archiving would break tests/test_diagnose_session.py and the audit-suite positive control (_audit_io.py:88). Preserve extracted/ in place.

## [MED] Navigator build + body-rate sign + countdown DQ
- **Location:** `handoff/shadowpc-navigator-2026-06-02/HANDOFF.md`
- **Issue type:** duplication · **Severity:** med
- **Evidence:** body_rate_sign=[-1,1,-1] (later refined to the 3-layer R_y(π) convention), the race countdown/early-start DQ mechanics, ACRO-only position-runaway, and analyze_run.py replay are all banked: grep hits in index_control_sim.md, project_ctbr_control_sysid.md, reference_sim_interface.md, reference_sim_ops.md.
- **Recommended action:** Bank-then-archive. All findings captured. The [-1,1,-1] sign here is the EARLY value, correctly superseded by the inc6-diag 3-layer convention in memory — no contradiction (memory documents the supersession). Archive dir.
- **Info-loss risk:** none — sign evolution and DQ mechanics fully captured with supersession noted.

## [MED] Velocity-setpoint fork (World A) + ODOMETRY body-frame velocity bug
- **Location:** `handoff/shadowpc-velocity-fork-2026-06-04/REPORT.md`
- **Issue type:** duplication · **Severity:** med
- **Evidence:** "World A" easy-mode-closed verdict and the ODOMETRY body-frame velocity bug (velocity_ned interleaved body/world) banked: project_ctbr_control_sysid.md:13-16 ("World A confirmed") and :219-221 (ODOMETRY body-frame velocity bug, FIXED c3b5a8e). 6 trace_*.json + UNDERSTANDING.md pre-registration are supporting raw data.
- **Recommended action:** Bank-then-archive. Verdict + bug both banked in project_ctbr_control_sysid.md with the dir cited by path. Trace JSONs regenerable; no code dependency. Archive dir.
- **Info-loss risk:** none — both the verdict and the highest-value ODOMETRY-frame bug are in project_ctbr_control_sysid.md, which cites this dir.

## [MED] Frame-fix re-validation (c3b5a8e confirmed; §5 KF-4×-lag retired)
- **Location:** `handoff/shadowpc-framefix-revalidation-2026-06-05/` (REVALIDATION.md, UNDERSTANDING.md, revalidate.py)
- **Issue type:** duplication · **Severity:** med
- **Evidence:** c3b5a8e fix confirmation + retirement of the "§5 KF velocity lags 4×" framing banked: project_ctbr_control_sysid.md:18-23 ("FIXED in c3b5a8e … KF now tracks truth to 0.05 m/s; rotation validated to ±12° roll/±20° pitch") and :221. Follow-up #1 ("update §5") executed.
- **Recommended action:** Bank-then-archive. Confirmation + §5 retirement both in project_ctbr_control_sysid.md. No code dependency. Archive dir.
- **Info-loss risk:** none — fix validity and saga retirement captured; revalidate.py self-contained and regenerable.

## [MED] Live VERIFY + reVERIFY — alt-loop relay limit cycle (delay-driven)
- **Location:** `handoff/shadowpc-verify-2026-06-06/` (HANDOFF.md, RUNG1_HOVER.md, VPROBE.md, UNDERSTANDING.md) and `handoff/shadowpc-reverify-2026-06-07/REVERIFY.md`
- **Issue type:** duplication · **Severity:** med
- **Evidence:** The ~6 Hz alt-loop relay limit cycle = delay-driven (~40 ms loop transport delay, NOT vz-source lag), and the raw-vz fix FAIL, banked: project_ctbr_control_sysid.md:5 ("~6 Hz limit cycle is a delay-driven relay (~40 ms loop delay), ALWAYS-ON but VALIDITY-HARMLESS") and :28-33 which cites "handoff/shadowpc-reverify-2026-06-07/". hover≈0.2656 banked across 4 files.
- **Recommended action:** Bank-then-archive (both verify + reverify). The limit-cycle saga + delay-driven verdict + raw-vz-fix-FAIL are in project_ctbr_control_sysid.md, which cites the reverify dir. rung1_hover/vprobe extract JSONs regenerable; no code dep. Archive both dirs.
- **Info-loss risk:** none — the limit-cycle root cause and failed-fix lesson are captured with citation.

## [MED] Robust association + depth-sanity (catastrophic-tail kill at source)
- **Location:** `handoff/shadowpc-assoc-flipfix-2026-06-09/HANDOFF.md`
- **Issue type:** duplication · **Severity:** med
- **Evidence:** The 3-layer fix (predict_gates_in_camera + scale-normalised associate + range_consistent depth sanity) and tail 46%→6.3% / leak 1.6%→1.1% banked: project_phase2_rl_vision_decisions.md:83-86 ("association.py ships … Catastrophic tail 46%→6.3%, leak 1.6%→1.1%, 370 tests green", commit 4673517) and :108. The fit-sizing script reltol_sizing.py is cited as provenance in src/racer/vision/association.py:61.
- **Recommended action:** Bank-then-archive. Verdict + numbers + commit 4673517 banked. Note association.py:61 cites reltol_sizing.py as a comment provenance (not a load) — archiving leaves a dangling doc-path; acceptable, but if archiving, update the comment to the new path. Archive dir.
- **Info-loss risk:** Low — keep the reltol_sizing.py provenance comment in association.py pointing wherever the dir lands.

## [MED] Start-transient launch ramp + LIVE 6/6 recovery + start YAW-spin bug
- **Location:** `handoff/shadowpc-start-ramp-2026-06-09/` (REPORT.md, FOLLOWUPS.md)
- **Issue type:** duplication · **Severity:** med
- **Evidence:** Sim-clock-keyed launch ramp + 4× CLEAN 6/6 recovery on build 1.0.3364 banked: "launch ramp / start transient / 6/6 recover" in index_control_sim.md, project_ctbr_control_sysid.md, project_phase2_rl_vision_decisions.md. The VQ2-fatal start YAW-spin bug (FOLLOWUPS §A) banked: project_phase2, project_rl_increment_history, reference_sim_ops.
- **Recommended action:** Bank-then-archive. Launch-ramp resolution + yaw-spin bug both banked. sweep_full.txt regenerable twin-sweep output. Archive dir.
- **Info-loss risk:** none — the ramp fix and the parked yaw-spin VQ2 item are captured.

## [MED] S1.2 RL deployment — virtual-flip + 4 plant-fidelity gaps + sim automation
- **Location:** `handoff/shadowpc-s12-rl-live-2026-06-10/HANDOFF.md`
- **Issue type:** duplication · **Severity:** med
- **Evidence:** tail-first/virtual-flip, the deployment math corrections, and the 4 gaps (rate-transient overshoot 9.7, ±180 yaw-spin, ~67 ms latency, thrust ceiling) banked: "virtual-flip/tail-first/S1.2" in feedback_checkpoint_transfer.md, project_rl_increment_history.md, project_phase2; MAV_CMD 31000 automation in reference_sim_ops.md + reference_sim_interface.md. inc4/inc5 RETIRED in the supersession chain (project_rl_increment_history.md:10).
- **Recommended action:** Bank-then-archive. S1.2 lineage subsumed by the inc-supersession chain and the §S1.2 section in project_phase2. Archive dir.
- **Info-loss risk:** none — virtual-flip primitive + 4 gaps + automation recipe captured.

## [MED] Characterize-sweep — static super-rate map + NO sim anomaly (the superseder)
- **Location:** `handoff/shadowpc-characterize-sweep-2026-06-10/WRITEUP.md`
- **Issue type:** duplication · **Severity:** med
- **Evidence:** The static amplitude-dependent gain map (super-rate, G0/(1-s|c|/π), s≈0.30) and "THERE IS NO sim anomaly (open-loop) — the spin was a gate-post collision" banked: project_phase2_rl_vision_decisions.md:189-217 ("CHARACTERIZE-SWEEP DONE (510da24) … static gain map; NO sim anomaly"); src/racer/rl_plant.py:122 cites "characterize-sweep 2026-06-10 WRITEUP Section 3"; tests/test_super_rate.py:3 cites WRITEUP Section 1.
- **Recommended action:** Bank-then-archive — BUT **KEEP the WRITEUP.md** (cited as constant-provenance by src/racer/rl_plant.py:122 and tests/test_super_rate.py:3 doc comments; it is the named source-of-truth for shipped super-rate constants + the collision-not-anomaly termination decision). The analysis .py files (fit_sweep/fit_windup/etc.) are regenerable scratch that may be pruned.
- **Info-loss risk:** Keep WRITEUP.md in place so rl_plant.py:122 / test_super_rate.py:3 provenance citations resolve; pruning the dir's analysis scripts is safe.

## [MED] Twin-falsify — quadratic drag + convex collective map + no battery sag
- **Location:** `handoff/shadowpc-twin-falsify-2026-06-10/WRITEUP.md` (+ profiles/, fit_*.py)
- **Issue type:** duplication · **Severity:** med
- **Evidence:** Quad body-frame drag (c2≈0.052), convex K(thr) collective map (2.12× at full stick), super-rate holds at airspeed, no battery sag banked: project_phase2_rl_vision_decisions.md:271-291 ("TWIN-FALSIFY CAMPAIGN COMPLETE … S16 AERO INTEGRATION COMPLETE eba4349"). WRITEUP cited by src/racer/twin_fit.py:378, tests/test_measured_aero.py:3, rl/diffaero_dynamics.py:405; profiles/ cited by scripts/rate_sysid.py:248.
- **Recommended action:** Bank-then-archive findings, but **KEEP the WRITEUP.md + profiles/** (named provenance for shipped aero constants + re-runnable probe profiles, cited by twin_fit.py / test_measured_aero.py / diffaero_dynamics.py / rate_sysid.py). The fit_*.py + predict.py + vert_fit_coef.npy are regenerable scratch.
- **Info-loss risk:** Keep WRITEUP.md + profiles/ in place so the 4 source/test citations resolve; the campaign findings are independently banked in project_phase2.

## [MED] VISION-PKG2 — measured fix-covariance model + 180° frame fix + yaw-bias refuted
- **Location:** `handoff/shadowpc-vision-pkg2-2026-06-10/WRITEUP.md` (+ analysis scripts, stdouts)
- **Issue type:** duplication · **Severity:** med
- **Evidence:** σ_theta=1.4° lever + 0.40 m floor + 32 m cap + K=2 + the 180° corner_to_center frame fix + "no shippable fixed calibration (yaw bias refuted)" banked: project_phase2_rl_vision_decisions.md:114-156 ("VISION-PKG2 COMPLETE … commits 37e7ab1, 1b7e753, 9ccc88c, 4831991"). Source-of-truth path cited at :116. The +0.3 m vertical constant caveat is the seed of the later boresight/ε_vert work.
- **Recommended action:** Bank-then-archive. Model + frame fix + commits all banked. The before/after/fixedframe stdout .txt are regenerable (per-frame JSONs gitignored). No code dependency on this dir's files. Archive dir.
- **Info-loss risk:** none — the full covariance model, the 180° fix, and the calibration-refutation are in project_phase2; the +0.3 m caveat is carried forward into the boresight thread.

## [MED] Live-deploy-diag — SIM MOTOR MIXER coupling (top-tier plant-fidelity discovery)
- **Location:** `handoff/shadowpc-live-deploy-diag-2026-06-11/WRITEUP.md` (+ mixer_probe.json)
- **Issue type:** duplication · **Severity:** med
- **Evidence:** The mixer thrust/rate coupling at saturation corners (corner1 thr=0×yaw=3.14 → 9.4 m/s² uncommanded lift, motors [0.08,0.73,0.73,0.08]; corner2 coll≈1 → rate authority vanishes) and --yaw-scale 0 (14 passes/10 flights) banked: project_rl_increment_history.md:49 (commits ef2605d..8dbd9bf). WRITEUP cited by tests/test_mixer.py:2. mixer_probe.json is the fit data (also referenced as data/runs/...mixer_probe).
- **Recommended action:** **KEEP the WRITEUP.md + mixer_probe.json.** WRITEUP is the named provenance for the shipped mixer model (test_mixer.py:2 cites it); mixer_probe.json is the fit data for kappa_err. Disposition = KEEP (provenance + fit-data fixture). Findings independently banked.
- **Info-loss risk:** Keep WRITEUP.md + mixer_probe.json so test_mixer.py:2 provenance resolves and the mixer fit data is preserved.

## [MED] crab-diag — crab = trained posture (not a bug); slowdown/droop DISSOLVE
- **Location:** `handoff/shadowpc-crab-diag-2026-06-12/WRITEUP.md` (+ scripts/)
- **Issue type:** duplication · **Severity:** med
- **Evidence:** crab = inc6's ~55° trained racing style (twin reproduces roll +41.5/yaw +130.9/tilt +54.8, FINISH 9.62 s); slowdown = clock artifact (RL-segment 8.96 s ≈ twin 8.16 s); rate-droop = wrong canary; gate-3 clip = thrust/drag plant residual — all banked: project_rl_increment_history.md:328-405 ("CRAB-DIAG … commit 301a2cf", writeup path cited).
- **Recommended action:** Bank-then-archive. Comprehensively banked (commit 301a2cf) including the explicit supersession of live-confirm facts and the "no reward change / no virtual-flip fix" doctrine. scripts/ self-contained. Archive dir.
- **Info-loss risk:** none — crab-as-trained-posture and the dissolved deficits fully captured.

## [LOW] bridge-retest — bridge "threads gates" prediction FAILED + mixer probe2 rows
- **Location:** `handoff/shadowpc-bridge-retest-2026-06-12/` (WRITEUP.md, probe_summary.md, sim_exit_home.png)
- **Issue type:** duplication · **Severity:** low
- **Evidence:** Bridge 5/5 gate-0, 0/5 gate-1 (DIAG prediction FAILED, thrust-lapse load-bearing for both modes) + the 4 mixer-probe2 corners (top-rail ~35% clip; bottom-rail ~80% at hover) banked: project_rl_increment_history.md:129-141 (commit e84a18d; cites probe_summary.md). probe2 consistency CONTRADICTION flagged at :211-217.
- **Recommended action:** Bank-then-archive — but **KEEP probe_summary.md** (named source-of-truth for the probe2 mixer corner rows that memory cites at :141). sim_exit_home.png is a 325 KB screenshot (prune). Archive the rest.
- **Info-loss risk:** Keep probe_summary.md so the :141 citation resolves; the PNG is a session-end housekeeping screenshot with no analytic content.

## [LOW] inc4-live — 0/10 transfer failure (RETIRED checkpoint)
- **Location:** `handoff/shadowpc-inc4-live-2026-06-11/WRITEUP.md`
- **Issue type:** duplication · **Severity:** low
- **Evidence:** inc4 RETIRED (project_rl_increment_history.md:10 "inc4 (aero-blind, RETIRED)"). Its 0/10 live failure root-caused later (roll mirror, inc6-diag; mixer, live-deploy-diag) and inc4's heavier rail-riding confirmed-worst on corrected aero (live-deploy-diag §7). Corner-pass geometry (≤0.64 m clean) and coast RMS 0.224 duplicated in inc5-live + corner-pass banking.
- **Recommended action:** Bank-then-archive. Pure live-failure session record, fully subsumed by the retirement chain + diag root-causes. Archive dir.
- **Info-loss risk:** none — all forensics superseded by inc6-diag/live-deploy-diag (banked).

## [LOW] inc5-live — 0/10 transfer failure + LIVE corner-pass aperture confirmation
- **Location:** `handoff/shadowpc-inc5-live-2026-06-11/WRITEUP.md`
- **Issue type:** duplication · **Severity:** low
- **Evidence:** inc5 RETIRED (project_rl_increment_history.md:10,58). §5 corner-pass probe (sim advances active_gate_index for crossings ≤0.64 m; 0.60 m = pass-with-contact; confirms 0.75 m aperture live) banked: project_phase2_rl_vision_decisions.md:652-659 ("CORNER-PASS PROBE (2026-06-11) … ≤0.64 m Gate ADVANCES"); the 0.60 m contact datum feeds the contact-radius band at :1537.
- **Recommended action:** Bank-then-archive. Live-failure subsumed by the retirement chain; the load-bearing corner-pass-aperture result banked in project_phase2 §CORNER-PASS. Archive dir.
- **Info-loss risk:** none — the 0.75 m aperture live-confirmation and the 0.60 m contact datum are in project_phase2.

## [LOW] inc6-live — 0/15 failure (INPUT to the roll-mirror root-cause)
- **Location:** `handoff/shadowpc-inc6-live-2026-06-12/WRITEUP.md`
- **Issue type:** duplication · **Severity:** low
- **Evidence:** The 0/15 failure is the named INPUT to inc6-diag (root-caused as the roll mirror). project_rl_increment_history.md:59 "inc6 … live transfer FAILED 2026-06-12; diag in progress". The mixer_probe2 partial-capture (z00_y31 clean, c100_r31 aborted at Euler singularity) completed later in bridge-retest.
- **Recommended action:** Bank-then-archive. Pure failure-session record subsumed by inc6-diag (root cause) + bridge-retest (probe2 completion), both banked. Archive dir.
- **Info-loss risk:** none — the failure and its root cause captured downstream.

## [MED] inc6-diag — 3-layer roll-mirror convention + thrust-lapse residual (top-tier discovery)
- **Location:** `handoff/shadowpc-inc6-diag-2026-06-12/WRITEUP.md` (+ scripts/diag_*.py)
- **Issue type:** duplication · **Severity:** med
- **Evidence:** ODOMETRY quat = true attitude; raw→true rates [+1,-1,+1]; live command rate_sign [-1,+1,-1] (roll AND yaw inverted); validate via tilted-phase quat-FD; fix bcc93f9; and the residual thrust-lapse 15-25% at 3-12 m/s banked: "roll mirror / [-1,+1,-1] / bcc93f9" in project_ctbr_control_sysid.md, project_rl_increment_history.md, reference_sim_interface.md; "thrust lapse 15-25%" in 9 memory files.
- **Recommended action:** Bank-then-archive. The 3-layer convention + tilted-phase-quat-FD gate + thrust-lapse are comprehensively banked (a cross-cutting footgun referenced throughout). scripts/diag_*.py (14 forensic scripts) are self-contained regenerable scratch. Archive dir.
- **Info-loss risk:** none — the convention discovery is one of the most-cited facts in memory; diag scripts reproducible.

## [LOW] laptop-frame-audit — the ODOMETRY R_y(π) conjugation root-cause
- **Location:** `handoff/laptop-frame-audit-2026-06-12/WRITEUP.md:14-21,207-230` (TL;DR + MEMORY-DELTA)
- **Issue type:** duplication · **Severity:** low
- **Evidence:** :14 "ROOT CAUSE … the sim ODOMETRY quaternion is … R_y(π)-conjugated frame pair. q_true = q_raw * [1,-1,1,-1]". A top-level MEMORY footgun: memory/MEMORY.md (ODOMETRY quat conjugation directive) + index_control_sim.md:28 "ODOMETRY quat R_y(π)-CONJUGATED. True attitude: q_true=q_raw·[1,−1,1,−1]; true rate: ω_true=−w_raw (gain 0.999); cmd→rate sign [+1,+1,+1]". Regression armor (tests/test_frame_conventions.py, scripts/frame_residual_report.py) merged (present in main).
- **Recommended action:** Bank-then-archive (fully banked + armored). Retain scripts/audit_candidates.py etc. inside the archive as the historical derivation (only copies; not in main).
- **Info-loss risk:** The per-layer handedness map (WRITEUP §1 table, 17 rows) is the most complete single artifact of which layers are conjugated/aliased; memory captures the conclusions but not the full table. Preserve by keeping WRITEUP.md in the archive (do not delete).

## [LOW] laptop-vision-frame-fix — 4th seam / east-bias
- **Location:** `handoff/laptop-vision-frame-fix-2026-06-12/WRITEUP.md:141-155` (MEMORY-DELTA)
- **Issue type:** duplication · **Severity:** low
- **Evidence:** :146 "vision-frame-fix SHIPPED (2026-06-12): navigator.py:295 now uses R_world_from_odo_quat_wxyz … fix p50 1.37→0.47m, leak 3.4%→0%". Banked: index_vision_estimator.md:118 "VISION-FRAME-FIX (8d7b0b3): east bias eliminated" and project_phase2_rl_vision_decisions.md:127-130 + :1086.
- **Recommended action:** Bank-then-archive. Fully banked and merged.
- **Info-loss risk:** none — code merged (8d7b0b3), conclusions + before/after table banked, regression tests in tests/test_frames.py + test_navigator.py on main.

## [LOW] laptop-pnp-cov-inflation
- **Location:** `handoff/laptop-pnp-cov-inflation-2026-06-08/HANDOFF.md`
- **Issue type:** duplication · **Severity:** low
- **Evidence:** TL;DR "Fix shipped here: a scalar inflation … PNP_FIX_COV_INFLATION = 2.0". Banked + CLOSED: project_phase2_rl_vision_decisions.md:111 "PNP_FIX_COV_INFLATION task CLOSED: behind the depth-sanity gate K=1.0 and K=2.0 leak identically … No revert needed. Detail: handoff/laptop-pnp-cov-inflation-2026-06-08". Adaptive-R design banked at project_estimator_robustness.md:18.
- **Recommended action:** Archive the dir. Task explicitly CLOSED in memory and the memory entry cites this dir as the detail pointer — keep that pointer valid by archiving rather than deleting.
- **Info-loss risk:** none — verdict CLOSED and banked; the ShadowPC re-measure recipe was superseded by the depth-sanity gate result.

## [LOW] laptop-s14-staticmap-integration
- **Location:** `handoff/laptop-s14-staticmap-integration-2026-06-10/REPORT.md`
- **Issue type:** duplication · **Severity:** low
- **Evidence:** "static super-rate gain map integrated … All gates PASS; local suite 409 green; defaults OFF". Banked: index_rl_training.md:21 "super-rate ✅ S14" and project_phase2_rl_vision_decisions.md:234 ("S14 STATIC-MAP INTEGRATION COMPLETE … Report: handoff/laptop-s14-staticmap-integration-2026-06-10/REPORT.md. Tests 376→409."). Constants SUPER_RATE_S_MEASURED=0.30 etc. merged into rl_plant.py.
- **Recommended action:** Bank-then-archive. Code merged, parity gates passed/banked, memory cites this REPORT. Archive (single REPORT.md).
- **Info-loss risk:** none — measured gain table + parity numbers banked; constants in main.

## [LOW] laptop-s16-aero-integration
- **Location:** `handoff/laptop-s16-aero-integration-2026-06-11/REPORT.md`
- **Issue type:** duplication · **Severity:** low
- **Evidence:** "measured aero (quad body drag + convex collective) integrated … defaults OFF". Banked: index_rl_training.md:20 "Quad drag ✅ S16" and project_phase2_rl_vision_decisions.md:291 ("S16 AERO INTEGRATION COMPLETE … report handoff/laptop-s16-aero-integration-2026-06-11/REPORT.md … 8 notable deviations."). QUAD_DRAG_C2_MEASURED / COLL_MAP_* merged into rl_plant.py.
- **Recommended action:** Bank-then-archive. Code merged, deviations banked, memory cites this REPORT. Archive.
- **Info-loss risk:** none — the 8 deviations are individually banked at project_phase2_rl_vision_decisions.md:291.

## [LOW] laptop-s17-mixer-inc6
- **Location:** `handoff/laptop-s17-mixer-inc6-2026-06-11/WRITEUP.md` (model §1 + fit §2); mixer_probe2.json
- **Issue type:** duplication · **Severity:** low
- **Evidence:** Ships "stage1_inc6_actor.pth" + the mixer model "u_i=clip(c+S·d, idle, 1); κ_err=0.073". Banked: index_rl_training.md:20 "mixer ✅ S17 (u_i=clip(c+S·d, idle, 1); κ_err=0.073)" and project_rl_increment_history.md:52 (full model). MIXER_KAPPA_ERR/HOLD_MEASURED merged in rl_plant.py:173. The mixer_probe2 follow-up (S19 contradiction) banked at parked #45/#56 and project_phase2_rl_vision_decisions.md:987-993. inc6 superseded by inc7.
- **Recommended action:** Bank-then-archive. **KEEP mixer_probe2.json + fit_mixer.py inside the archive** (only copies; not in main; support the still-open parked #45/#56 S19 follow-up; keeper-run path recorded at project_rl_increment_history.md:116). Archive the dir.
- **Info-loss risk:** fit_mixer.py + mixer_probe2.json exist only here; they support the open parked #45/#56 S19 follow-up. Preserve by archiving the dir intact.

## [LOW] laptop-s15-envredesign-retrain
- **Location:** `handoff/laptop-s15-envredesign-retrain-2026-06-11/WRITEUP.md:1-30` (OUTCOME) + §5 review
- **Issue type:** duplication · **Severity:** low
- **Evidence:** Ships inc4 (stage1_inc4_actor.pth) + the C1-C6 coherent-reward env redesign + procedural courses + the SIDECAR deploy-bug fix ("every 3.765-trained checkpoint deployed through a [0,5] thrust rescale"). The env redesign is the foundation of all later increments (merged in peregrine_racing.py docstring per WRITEUP §1). inc4 retired.
- **Recommended action:** Bank-then-archive. Verify the C1-C6 coherent-reward/termination design + the sidecar-overdrive bug are captured in project_rl_increment_history.md (the env-redesign module docstring is the SSOT in peregrine_racing.py on main); if the sidecar [0,5]-rescale footgun is NOT explicitly in memory, bank one line to index_rl_training.md then archive.
- **Info-loss risk:** low — the design doc is the live module docstring in main. The adversarial-review residue (GAE-leak-across-truncation, hover-stall local optimum, time-in-reward aliasing — all "accepted/pre-existing") is unique to this WRITEUP §5; preserve by keeping WRITEUP.md in the archive.

## [LOW] laptop-inc7-env — contact-true geometry env
- **Location:** `handoff/laptop-inc7-env-2026-06-12/WRITEUP.md:188-209` (MEMORY-DELTA); scripts/inc6_geom_eval.py, scripts/launch_inc7.py
- **Issue type:** duplication · **Severity:** low
- **Evidence:** :189 "INC7 ENV SHIPPED (134cc41 …): +env.body_radius_lo/hi + +env.frame_depth_m (EXACT segment-vs-slab) + +dynamics.dr_force_bias". The contact-true env is the merged foundation of inc7 (current best). Also contains the inc6-geometry-honest eval (gate-5 also illegal; g3 true margin 0.06-0.16 m).
- **Recommended action:** Bank-then-archive. Env merged (134cc41), inc7 launched/banked. **KEEP scripts/inc6_geom_eval.py + launch_inc7.py inside the archive** (only copies; inc6_geom_eval is cross-referenced by laptop-inc8-metric-instrument WRITEUP:99). Verify the "gate-5 ALSO illegal under honest contact" finding is banked; if not, bank one line then archive.
- **Info-loss risk:** inc6_geom_eval.py is referenced by another handoff as a cross-check; keep it in the archive. The vision-frame-fix-breaks-4-twin-tests note (WRITEUP:205) is already resolved/superseded.

## [LOW] laptop-inc7-eval — inc7 = current best, LIVE-CONFIRMED foundation
- **Location:** `handoff/laptop-inc7-eval-2026-06-13/WRITEUP.md:218-228` (MEMORY-DELTA)
- **Issue type:** duplication · **Severity:** low
- **Evidence:** :219 "inc7 COMPLETE. Winner = s0 (job 3270605); rl/checkpoints/stage1_inc7_actor.pth; md5 2AFF8D…". Banked: MEMORY.md (inc7 LIVE-CONFIRMED = current best) + index_rl_training.md (9.76 s, sr 1.000). The deploy-matrix lat=2 result + latency-DR behavior banked.
- **Recommended action:** Bank-then-archive. inc7 result fully banked + checkpoint shipped. run_pergate.py is the only copy (one-off eval helper) — keep inside the archive. Archive.
- **Info-loss risk:** none material — the per-gate trainreset/simstart tables are supporting detail; the headline numbers + md5 + selection rationale are banked. run_pergate.py preserved in archive.

## [LOW] inc7-live — FIRST standing-start RL finishes (current-best LIVE confirmation)
- **Location:** `handoff/shadowpc-inc7-live-2026-06-13/WRITEUP.md`
- **Issue type:** duplication · **Severity:** med (one agent med; kept) → grouped low-priority disposition
- **Evidence:** inc7 standing 5/5 FINISHED, gate-3 barrier GONE, bimodal 9.97 vs 11.45 s, mirror canary intact, residual N+D plant gap tolerated — all banked: project_rl_increment_history.md:506+ ("INC7 LIVE CONFIRMED (2026-06-13, commit 7fe90df; writeup handoff/shadowpc-inc7-live-2026-06-13/WRITEUP.md)"); MEMORY.md NOW flags "inc7 LIVE-CONFIRMED = current best".
- **Recommended action:** Bank-then-archive. Comprehensively banked (commit 7fe90df, the live-confirm milestone). Archive dir. (inc7 is standing CURRENT-BEST per MEMORY.md NOW, so keep the banked summary prominent; the handoff dir itself is archivable.)
- **Info-loss risk:** none — the 5/5 milestone, gate-3-cleared, bimodal, and residual-gap are all in project_rl_increment_history + MEMORY.md.

## [LOW] laptop-p4-c05 — per-gate-yaw gate frame
- **Location:** `handoff/laptop-p4-c05-2026-06-13/WRITEUP.md` (MEMORY-DELTA)
- **Issue type:** duplication · **Severity:** low
- **Evidence:** "P4-C05 FIXED on main: fly_rl.obs_from_zup/build_obs now take opt-in gate_map … default = hardcoded yaw=π VQ1 path, BIT-EXACT … old default diverges 4.22 m". Banked: index_rl_training.md:43 "P4-C05 DONE (f50b9b4, 692 green)", index_control_sim.md:56 (the 3 loud guards), index_vision_estimator.md:163. Merged at f50b9b4.
- **Recommended action:** Archive the dir (bank-then-archive). Fully banked (3 memory files) + merged + the loud-guard deploy constraint is a banked footgun. Single WRITEUP.md.
- **Info-loss risk:** none — code merged, the 4.22 m non-π divergence datum + guard semantics banked in 3 memory files.

## [LOW] laptop-flyrl-autonomy-hardening — submit_rl.py judged-run entrypoint
- **Location:** `handoff/laptop-flyrl-autonomy-hardening-2026-06-13/WRITEUP.md` (F-A..F-D); verify/*.py
- **Issue type:** duplication · **Severity:** low
- **Evidence:** F-A "rl/submit_rl.py (NEW): the committed, submission-safe entrypoint. Pins … stage1_inc7_actor.pth --no-bridge --no-auto-reset --no-debug-obs". Banked: index_control_sim.md:55-57 (Autonomy-hardening DONE; MERGED 6876f44; submit_rl.py on main, NO MAV_CMD 31000) + MEMORY.md:46 (judged-runs footgun) + parked #54/#63. Merged at 6876f44.
- **Recommended action:** Archive the dir (bank-then-archive). Merged + judged-runs footgun is a top-level MEMORY directive. verify/*.py offline-verify scripts are one-off; keep inside the archive. The §4 LIVE-VERIFICATION CHECKLIST is the only unbanked content — **confirm whether the ShadowPC live-verify of the hardened arm/disarm path was completed** (the writeup gated merge on it); if not yet done, it is a live action, not archivable content.
- **Info-loss risk:** The §4 ShadowPC live-verification checklist (7 items, arm/late-join/odo-guard/disarm-on-crash) is a pending VERIFICATION procedure, not a finding. If the live-verify was not run, preserve by keeping WRITEUP.md accessible (archive, don't delete) and surfacing it as an open action.

## [LOW] laptop-promote-regression-suite
- **Location:** `handoff/laptop-promote-regression-suite-2026-06-13/WRITEUP.md`
- **Issue type:** duplication · **Severity:** low
- **Evidence:** "Promote the 8 external-invariant regression scripts … into tests/. 30 new test functions … Suite 657→687 passed." The 8 tests/test_*.py + tests/_audit_io.py merged (e.g. test_frame_conventions.py, test_confirmed_p4_c05.py present in main; suite count 687→692 reflected in index_strategy_meta.md:49).
- **Recommended action:** Archive the dir (bank-then-archive). Pure test-infra promotion; all artifacts merged into tests/. Single WRITEUP.md. The "guard the guard" negative-control philosophy is worth one line if not already in index_strategy_meta — optional.
- **Info-loss risk:** none — the promoted tests are on main; the East-axis force-vs-FD discriminator (the shared positive control) is the merged test_frame_force_vs_fd_mirror_canary.py.

## [LOW] laptop-inc8-metric-instrument
- **Location:** `handoff/laptop-inc8-metric-instrument-2026-06-13/WRITEUP.md:104-122,167-185`
- **Issue type:** duplication · **Severity:** low
- **Evidence:** :11 "rl/contact_true_eval.py — the inc8 Phase-0(b) metric instrument"; :179 "contact_true_eval.py LIVE in main (commit 0bb60cc)". Tool merged. The gate-3 ±1.5 m-flips-to-COLLISION probe (:116) shares the SAME 1.46 m claim later refuted (see binding-gate finding). Tightest-gate=gate-4(0.205 m) banked.
- **Recommended action:** Bank-then-archive. contact_true_eval.py merged (0bb60cc), superseded/extended by the binding-gate session + p4-c05 yaw-aware threading. The "~1.46 m map confound → interpret gate-3 with caution" note is stale (δ_map≈0 refuted it). Archive; rely on the banked refutation as SSOT.
- **Info-loss risk:** Same as binding-gate: the archived writeup repeats the refuted 1.46 m confound. The durable content (contact-true metric replaces L-inf<0.75 fiction; per-gate margins) is banked + tool on main. No unique loss.

## [LOW] laptop-tilt-concentration
- **Location:** `handoff/laptop-tilt-concentration-2026-06-11/WRITEUP.md` (MEMORY-DELTA)
- **Issue type:** duplication · **Severity:** low
- **Evidence:** Per-segment Δt (start→G0 +0.73s, G2→G3 +0.67s …) + 3-step speed ladder (rw_tilt 96→48; free-cone 60→75-80°). Banked verbatim: project_phase2_rl_vision_decisions.md:1009-1030 ("✅ TILT-CONCENTRATION (2026-06-11) … Source: handoff/laptop-tilt-concentration-2026-06-11/WRITEUP.md") + index_rl_training.md:38 + parked #38/#44.
- **Recommended action:** Archive the dir (bank-then-archive). Fully banked with the per-segment table reproduced in project_phase2 and the dir cited as source. Single WRITEUP.md.
- **Info-loss risk:** none — per-segment table + revised ladder banked verbatim; the metric-note caveat (training roll-p90-145° ≠ racestart-peak-75°) banked at project_phase2:1009-1030.

## [LOW] laptop-photoreal-recon
- **Location:** `handoff/laptop-photoreal-recon-2026-06-11/WRITEUP.md` (MEMORY-DELTA)
- **Issue type:** duplication · **Severity:** low
- **Evidence:** v2 dataset inventory (data/mix_v2, 6000+800, edge_prob=0, no HDRI/PBR; biggest gaps lighting/background/edge-case density). Banked: project_detector_training_pipeline.md:80-82 ("Source: handoff/laptop-photoreal-recon-2026-06-11/WRITEUP.md. Closes the first-action of ADVISOR-TRIAGE item ① … data/mix_v2/, 6000 train + 800 val …").
- **Recommended action:** Archive the dir (bank-then-archive). Recon findings banked + the VQ2 Blender pipeline that this recon scoped is now landed (project_parked_backlog.md:31, blender_gen on main d437507). Single WRITEUP.md.
- **Info-loss risk:** none — the 6-point gap assessment + the "Blender pipeline CAN reuse to_yolo_pose_label" conclusion banked in project_detector_training_pipeline.md.

## [LOW] laptop-gate-mapper
- **Location:** `handoff/laptop-gate-mapper-2026-06-11/WRITEUP.md` (§4 validation + §5 failure modes); validation_table.md; scratch/, samples/
- **Issue type:** duplication · **Severity:** low
- **Evidence:** "src/racer/gate_mapper.py + gate_mapper_synth.py … cases A/B/C … 31 tests, full suite green (528)". Banked: index_vision_estimator.md:123 "GATE MAPPER COMPLETE: src/racer/gate_mapper.py cases A/B/C" (merged; src/racer/gate_mapper.py present in main). Case-A bias-corrected 0.20 m; case-C SHAPE-only (absolute frame unobservable) + 7 honest failure modes.
- **Recommended action:** Bank-then-archive. Code merged + completion banked. **Verify the 7 honest case-C failure modes (§5) are reflected in the case-C readiness banking** (index_vision_estimator.md case-C section / project_estimator_robustness.md); if the §5 list is not banked, bank a compressed version then archive. PRUNE scratch/debug_c.py + scratch/smoke.py (dev scratch) after confirming.
- **Info-loss risk:** The §5 failure-mode list and §4 A/B/C validation tables are detailed; memory has the headline but may lack the case-C SHAPE-vs-world caveat and the co-visibility-impossible geometry note. Preserve by keeping WRITEUP.md + validation_table.md in the archive; consolidate the §5 list into case-C readiness if absent.

## [LOW] laptop-training-doctrine — the "fictional gate" → inc7 doctrine
- **Location:** `handoff/laptop-training-doctrine-2026-06-12/WRITEUP.md` (TL;DR + §2 ledger); QUESTIONS.md; scripts/q1_trace.npz
- **Issue type:** duplication · **Severity:** low
- **Evidence:** TL;DR "We have been training against a fictional gate … live crashes terminated at L-inf 0.37-0.49 m — positions the training env calls comfortable passes … Fix the geometry in training (body-radius inflation + frame extrusion) … that is inc7." Banked: index_rl_training.md:127 cites §TRAINING-DOCTRINE in project_phase2. The crab-near-optimum (Q1, ~0.15 s/lap, 15× smaller than tilt lever) banked at the training-doctrine section.
- **Recommended action:** Bank-then-archive. The doctrine is captured (project_phase2 §TRAINING-DOCTRINE) and realized as merged inc7-env. **Keep scripts/doctrine_probes.py + q2_live_forensics.py + q1_trace.npz inside the archive** (only copies; q1_trace.npz is a small binary). Archive.
- **Info-loss risk:** low — the 14-question ledger (QUESTIONS.md) is thorough; the answered conclusions (fictional-gate, crab-optimal, force-scale-DR-already-on) are banked. Preserve QUESTIONS.md + WRITEUP.md in the archive. q1_trace.npz reproducible via doctrine_probes.py.

## [LOW] recenter build — the through-centering fix (MERGED)
- **Location:** `handoff/inc8-recenter-build-2026-06-17/REPORT.md` (whole)
- **Issue type:** duplication · **Severity:** low
- **Evidence:** Documents the through_centering_reward build (merged 9cecf64). Fully banked at project_rl_increment_history.md:746 with the same code paths, the R1-to-centre diagnosis, the cross-track-vs-Euclidean rationale, and the GO-criterion lesson. The build is in main code (rl/inc8_reward.py, rl/peregrine_racing_inc8.py).
- **Recommended action:** Archive (content in main code + banked at project_rl_increment_history.md:746). No banner needed — no stale verdict.
- **Info-loss risk:** none — merged to code and banked verbatim at project_rl_increment_history.md:746.

## [LOW] warm-start build + S4 result
- **Location:** `handoff/inc8-warmstart-build-2026-06-17/REPORT.md` (whole)
- **Issue type:** duplication · **Severity:** low
- **Evidence:** Documents the +init_from warm-start capability (merged 11cb840) and the S4 3/3-clean training result. Fully banked at project_rl_increment_history.md:741-742 (key=root-level +init_from, agent.load weights-only, critic stays symmetric, look-at warmup forced 0). The warm-start PATH was later superseded as the route-to-flight by the fresh-seed recenter run (warmstart seeds still died at gate-0 — :743), but the capability remains in code.
- **Recommended action:** Archive with a one-line note ("warm-start CAPABILITY merged + kept in code; as the route-to-flight it was superseded by the fresh-seed recenter run — the ws seeds inherited the gate-0 policy-gap"). Banked at project_rl_increment_history.md:741-743.
- **Info-loss risk:** none — capability + S4 result banked at project_rl_increment_history.md:741-742.

## [LOW] stability knobs (warmup / critic-width / band_el; symmetric-critic footgun source)
- **Location:** `handoff/inc8-stability-knobs-2026-06-16/REPORT.md:92-115`
- **Issue type:** duplication · **Severity:** low
- **Evidence:** The load-bearing finding ("the inc8 critic is SYMMETRIC (obs-input 20), NOT the asymmetric 36-dim critic the doctrine assumes … GuardedPPO extends symmetric PPO, build passes obs_dim only") is banked VERBATIM in MEMORY.md (the "inc8 CRITIC IS SYMMETRIC" cross-cutting footgun) and index_rl_training. The three knobs (warmup, critic-width, band_el) are merged to code.
- **Recommended action:** Archive (knobs in code; the symmetric-critic footgun banked in MEMORY.md's Cross-cutting-footguns). No stale verdict.
- **Info-loss risk:** none — the symmetric-critic footgun (with the Adroit confirmation recipe agent.agent.critic.critic.input_dim==20) is banked in MEMORY.md.

## [LOW] laptop-s0/s1 ladder (yaw sign + centering-weight closure)
- **Location:** `handoff/inc8-s0-2026-06-16/REPORT.md` (MEMORY-DELTA:401-407) and `handoff/inc8-s1-cen3-2026-06-16/REPORT.md` (MEMORY-DELTA tail)
- **Issue type:** duplication · **Severity:** low
- **Evidence:** S0 MEMORY-DELTA pins the empirical yaw sign ("yawneg3 … SIGN=CONFIRMED NEGATIVE … g_yaw=-3.0") and the Hydra dir-collision footgun; S1-cen3 pins "rw_centering=3.0 WORSE than 1.0 … weight sweep CLOSED" + the Hydra collision footgun. Both banked: the -3.0 sign in MEMORY.md (SIGN FOOTGUN bullet) and the ladder detail at project_rl_increment_history.md:728-735 (incl. estim_err-saturation root cause at :735).
- **Recommended action:** Archive both dirs (content duplicated into memory). No banner needed — they carry no live stale verdict beyond the subsumed-ladder context. Keep the raw per-seed traces for provenance.
- **Info-loss risk:** none — yaw sign, Hydra-collision footgun, and the yaw-only→no-elevation-fix→estim_err-floor finding banked (MEMORY.md SIGN FOOTGUN; project_rl_increment_history.md:735).

## [LOW] fix-surrogate — live infrastructure, accept_rlo superseded by accept-geometry re-fit
- **Location:** `handoff/fix-surrogate-2026-06-14/REPORT.md:60,118-128` (also :88-103)
- **Issue type:** duplication · **Severity:** low
- **Evidence:** :60 fits "rlo 16.2" which accept-geometry-2026-06-15 (:118-128) later recommends re-fitting to 12, now adopted (index_rl_training.md:117 "accept_rlo=12 … merged to main @ 8cf23c9"). The module (rl/fix_surrogate.py, merged ffb2c74) + 2-axis elevation lever finding (:88-103, ×8.5 2-axis pointing) banked/merged.
- **Recommended action:** **KEEP (reference)** — rl/fix_surrogate.py is live infrastructure and this is its design-of-record (calibration provenance, the 4 sub-models, the ×8.5 2-axis-pointing lever). Add a one-line pointer that accept_rlo was re-fit 16.2→12 (accept-geometry, merged 8cf23c9). No stale gate-4 framing of concern.
- **Info-loss risk:** none — the module is merged; the elevation-co-binds finding is banked. The rlo update is noted.

---

# ISSUE TYPE: GAP (missing banking — bank then archive cleanly)

## [LOW] Banking completeness gap — det-retrain not in the rl-history §inc8 ledger
- **Location:** `memory/project_rl_increment_history.md` (§inc8 ends at S5 eval-pitch line 750; the det-retrain 42ddb0b result is only in MEMORY.md:8)
- **Issue type:** gap · **Severity:** low
- **Evidence:** The det-retrain final result (noise-anneal/std-cap lever 42ddb0b; first det-flyable 2-axis inc8 det-reach 0.467; rc1 actor_logstd saturation root cause; σ_p0 0.15-0.20 marginal-passing) is fully in MEMORY.md:8 but does NOT appear as an S6/final entry in project_rl_increment_history.md's §inc8 ledger (which stops at S5/eval-pitch:750). The thin-index doctrine routes detail DOWN to the topic file; this detail currently lives only in the layer-1 index.
- **Recommended action:** Bank a new S6 entry to project_rl_increment_history.md §inc8 summarizing the det-retrain (lever 42ddb0b, det-reach 0.467, the rc1 std-saturation root cause, the std-ceiling KEEP, the corrected MARGINAL-PASSING verdict) with a pointer from MEMORY.md:8 — then the det-retrain handoff dir can archive cleanly behind a topic-file home rather than relying on the NOW-block. Low severity (no info loss today — it IS in MEMORY.md).
- **Info-loss risk:** none currently (banked in MEMORY.md:8). The action is hygiene: route the detail to the topic file so MEMORY.md's NOW block can later thin without dropping it.

---

# ISSUE TYPE: PRUNE (scratch / completed-mechanical — safe to consolidate; protect fixtures)

## [LOW] shadowpc set — gate-4 stale-conclusion scan returned CLEAN (recorded explicitly)
- **Location:** `handoff/shadowpc-*` (all 27 dirs)
- **Issue type:** prune (negative finding — no action) · **Severity:** low
- **Evidence:** Exhaustive grep for "sigma_p0|σ_p0|near-field|pivot off RL|fix-seating|0.08 m closure bar|NO-GO definitive" across all 27 shadowpc dirs returned ZERO matches in any REPORT/WRITEUP prose (the only "NO-GO"/"0.08" hits were numeric values inside data .json files and unrelated substrings). All shadowpc dirs are 2026-06-02..13 control/sysid/perception/live work, predating the gate-4 σ_p0 saga entirely.
- **Recommended action:** No action — this set contains NO stale gate-4 0.08/NO-GO/pivot conclusions to flag. Recorded explicitly so the commander knows the gate-4 correction risk lives in the LATER dirs (inc8-*, sigmap0-*, boresight-closure, coast-drift, at-speed-sigma), NOT here.
- **Info-loss risk:** none.

## [LOW] inc8 scratch artifacts (diag scripts, sig dumps, tb event files)
- **Location:** `handoff/inc8-eval-pitch-2026-06-18/{diag_*.py,trace_bodyrate.py,sig_*.txt,before/after_*.txt}`; `handoff/inc8-intraining-coursecheck-2026-06-17/tb/*.tfevents + parse_tb.py`; `handoff/inc8-obs-parity-2026-06-17/diag_parity.py`; `handoff/inc8-repilot-window/check_c_inplane.py`; `handoff/inc8-conversion-diag/reconstruct.py`; `handoff/inc8-deterministic-retrain/{probe_logstd.py,monitor.sh}`
- **Issue type:** prune · **Severity:** low
- **Evidence:** sig_yaw_seed0.txt content = the 0.1768 m run dump that REPORT.md:104 already quotes; diag_*.py / trace_bodyrate.py are one-shot bisection scripts whose RESULTS are in the eval-pitch REPORT (t_cam Δ=[0,0,0], realized-omega trace); tb/*.tfevents are raw protobufs whose parsed numbers are in the coursecheck REPORT table (success_rate 0.0000 ×4). All substantive numbers are in the accompanying REPORT.md.
- **Recommended action:** Safe to prune the raw scratch (sig/before/after .txt, *.tfevents) AND keep-with-report the diag/trace .py (small, reproducible provenance) when archiving each parent dir — i.e. archive the whole dir as a unit; do NOT separately delete the .py reconstruction scripts (they are the only record of HOW each number was derived). The .txt/.tfevents dumps are the prunable layer. (NOTE: probe_logstd.b64 + the inc8-reach-rate dir are untracked working-tree artifacts per git status — they are part of this same scratch layer.)
- **Info-loss risk:** none for the .txt/.tfevents (numbers quoted in the parent REPORT.md). Preserve the diag/trace .py scripts inside the archived dir — they encode the methodology (e.g. the yaw/pitch realized-omega trace) not re-derivable from prose alone.

## [MED] refit-dataset — 17-run debug_obs package (.zip is the tracked source)
- **Location:** `handoff/shadowpc-refit-dataset-2026-06-12/` (MANIFEST.md, debug_obs_17runs.zip, extracted/)
- **Issue type:** prune · **Severity:** med
- **Evidence:** MANIFEST's refit target (thrust lapse 0.74-0.88 at 3-12 m/s) is from inc6-diag §6 (banked). tests/_audit_io.py:89 points at extracted/ BUT :101-105 documents "refit/extracted is gitignored (only its .zip is tracked) … the promoted tests skip (rather than ERROR)". extracted/ confirmed EMPTY on disk; the tracked artifact is debug_obs_17runs.zip (9.4 MB).
- **Recommended action:** **KEEP debug_obs_17runs.zip** (the git-tracked source of the refit dataset + audit fixture; tests unzip it on demand). The empty extracted/ is regenerable. Disposition = KEEP (the .zip is the load-bearing artifact; findings banked, the empty extracted/ may be left as-is since tests skip gracefully).
- **Info-loss risk:** Do NOT delete debug_obs_17runs.zip — it is the only tracked copy of the 17-run refit dataset and an audit fixture; the underlying data/runs recordings are gitignored.

## [LOW] inc8 infra/early-pilot reports (harness fixes, util sync, smoke, pilotA)
- **Location:** `handoff/inc8-smoke-harness-fix-2026-06-14/REPORT.md`, `handoff/inc8-util-syncfix-2026-06-14/REPORT.md`, `handoff/inc8-resmoke-2026-06-14/REPORT.md`, `handoff/inc8-pilotA-2026-06-14/REPORT.md`
- **Issue type:** prune · **Severity:** low
- **Evidence:** smoke-harness-fix: 4 harness defects (ONNX-export RC guard, sbatch RUN glob, TB-trace helper, .gitattributes eol=lf) — ALL merged to code (rl/peregrine_train_inc8.py, rl/inc8_tb_trace.py, .gitattributes). util-syncfix: metric .item() host-sync batching, merged. resmoke/pilotA: early GPU-util ~28% CPU-bound readings + the first pointing-plateau pilot — superseded by every later run. Tooling/early-pilot scaffolding with no surviving live decision.
- **Recommended action:** Archive all four (no banner). The ONNX-guard (memory:434 RESOLVED), the CPU-bound-at-numpy-estimator-boundary finding, and the pointing-plateau are captured in code + the ladder history. Verified captured: ONNX guard in rl/peregrine_train_inc8.py; util/CPU-bound noted across resmoke/pilotA and the inc8 saga.
- **Info-loss risk:** none — harness fixes in main code; the CPU-bound util finding and early pointing-plateau subsumed by the full ladder banking at project_rl_increment_history.md:728+.

## [LOW] ultracode-autonomy-readiness — DONE + fully implemented
- **Location:** `handoff/ultracode-autonomy-readiness-2026-06-13/REPORT.md` (whole) + memory/index_control_sim.md:55-62
- **Issue type:** prune · **Severity:** low
- **Evidence:** The audit's findings (R1 MAV_CMD 31000 DQ, F-A submit_rl.py wrapper, F-B finally-disarm, F-C odo_recv_ns gate, late-join, BSR3) are ALL implemented and merged: index_control_sim.md:55 ("Autonomy-hardening (DONE; MERGED to main 6876f44 … rl/submit_rl.py now ON MAIN … telemetry_health()/odo_recv_ns … finally-disarm, late-join GO all live)") with §4 LIVE-VERIFY PASS 5/5. The ~80 scratch/ probes are intermediate failure-injection dumps.
- **Recommended action:** Bank-then-archive. Findings + fixes fully banked (index_control_sim.md:55-62) and merged. scratch/ probes reproducible-from-source and safe to consolidate; cite index_control_sim.md §Autonomy-hardening + reference_sim_ops.md §AUTONOMY-READINESS as the home. No gate-4 framing present.
- **Info-loss risk:** none — every finding + fix is banked in index_control_sim.md:55-62 and reference_sim_ops.md, and merged (6876f44 / submit_rl.py on main).

## [LOW] ultracode-substrate-audit — DONE, regression suite promoted; ONE re-home check
- **Location:** `handoff/ultracode-substrate-audit-2026-06-13/REPORT.md` (whole) + regression_suite/
- **Issue type:** prune · **Severity:** low
- **Evidence:** Marks contact/margin OUT-OF-SCOPE (:7) so carries NO stale gate-4 framing. Findings (P4-C05 yaw=π VQ2 hazard, refit-dataset stale-provenance, external-invariant regression suite) banked: index_rl_training.md:43 (P4-C05 DONE f50b9b4), index_control_sim.md:56. The suite was promoted (handoff/oneoff-regression-suite-2026-06-14 exists). REPORT.md:63 flags a SLUG COLLISION (test_confirmed_cr4_03.py overwrote a COLL_MAP finding).
- **Recommended action:** Bank-then-archive. P4-C05 + the 61 CLEARED conventions + provenance defects banked. **BEFORE archiving, confirm the REPORT.md:63 SLUG-COLLISION COLL_MAP finding** (rl_plant.py thrust over-prediction at knots 6-9) was either re-homed or is captured in memory (project_ctbr_control_sysid) — the one item flagged at risk of loss. Cite index_rl_training.md:43 + index_control_sim.md:56 + the promoted oneoff-regression-suite as the home.
- **Info-loss risk:** LOW but real: the REPORT.md:63 COLL_MAP finding was OVERWRITTEN by the cr4_03 test and flagged "must be re-homed before it is lost." Verify it is captured in memory/project_ctbr_control_sysid.md or re-home it before archiving.

## [LOW] p0-casec-foundation — DONE (navigator P0-a/b/c + sign pin merged)
- **Location:** `handoff/p0-casec-foundation-2026-06-13/REPORT.md` (whole)
- **Issue type:** prune · **Severity:** low
- **Evidence:** P0-a (case-C cold-start seed gating), P0-b (TIMESYNC delta_epoch + predict-forward), P0-c (velocity coupling), and the +L sign pin (test_obs_sign_faithfulness.py, +L 4.77e-7 / −L 24 m) banked (MEMORY.md OBS SIGN +L footgun; index_vision_estimator.md) and the tests are in-suite. No gate-4 margin framing.
- **Recommended action:** Bank-then-archive. All four fixes + the +L pin banked/merged. Cite MEMORY.md OBS-SIGN footgun + index_vision_estimator.md as the home. Clean report.
- **Info-loss risk:** none — P0-a/b/c and the +L sign pin banked and the tests in-suite.

## [LOW] p1-vision-accuracy + p2-harvest + p2-refline — DONE infrastructure
- **Location:** `handoff/p1-vision-accuracy-2026-06-14/*` ; `handoff/p2-inc8-rl-harvest-2026-06-14/REPORT.md` ; `handoff/p2-inc8-rl-refline-2026-06-14/REPORT.md`
- **Issue type:** prune · **Severity:** low
- **Evidence:** p1-vision-accuracy (AUDIT escape-hatch, FORM=METRIC, dual-form boresight infra APPLIED 5091c88, ESKF re-scoped) banked/deployed (index_vision_estimator.md:232-258, boresight DEPLOYED 5764291). p2-harvest (passive-observer + S_stable instruments) merged into contact_true_eval/estimator_emul (722-test). p2-refline (reference_line_inc8.json, ~8s upright finding, ~7× quad-drag cause) banked (index_rl_training.md ~30m/8s expectation) and the file is the merged R1' Γ.
- **Recommended action:** Bank-then-archive all three. p1 = boresight pathway (DONE c740743/5764291/5091c88, banked). p2-harvest = merged instruments. p2-refline = merged Γ + the ~8s/30m upright-speed expectation (banked). No gate-4 closure-bar framing requiring correction.
- **Info-loss risk:** none — boresight FORM=METRIC + bake, harvest instruments, and the reference-line ~8s/30m finding all banked/merged.

## [LOW] kf-pfloor (#74) — durable insurance build
- **Location:** `handoff/kf-pfloor-2026-06-15/REPORT.md` (whole) + src/racer/state_estimator.py:77-91,153
- **Issue type:** prune · **Severity:** low
- **Evidence:** In-plane P-floor (_apply_inplane_pos_floor, state_estimator.py:91,153) MERGED (fe4b152); KEY finding — over-convergence does NOT reproduce with realistic correlated fixes (σ self-limits 0.074-0.124 > 0.05 floor; floor is insurance not load-bearing) — banked verbatim (project_parked_backlog.md:82 "#74 DONE"). Uses the coast-drift σ_p0⊕σ_b budget framing for context but makes NO gate-4 close/no-close verdict. "Full suite GREEN: 826 passed … 0 regressions"; "floor default 0.0 → byte-identical".
- **Recommended action:** **Keep.** Merged insurance feature, banked across index_rl_training/index_vision_estimator/project_parked_backlog (#74). Single REPORT.md + 2 ripple scripts; no scratch to prune. No staleness in the verdict (it's a build). **Confirm the "mirror the floor into the inc8 training emul if a future config produces a denser-than-G3 stream" caveat (:103-106) is parked.**
- **Info-loss risk:** none — build merged and banked. The denser-than-G3 carry-forward is a real item; confirm it's parked.

## [LOW] durable cluster — fix-surrogate / cowork / blender / simops / perception-char / ensemble-support / deploy-obs20 / spike
- **Location:** `handoff/cowork-2026-06-14/*.md`; `handoff/vq2-blender-dataset-2026-06-14/REPORT.md`; `handoff/vq2-blender-render-2026-06-15/COMMANDER_REPORT.md`; `handoff/simops-mastery-2026-06-13/REPORT.md`; `handoff/perception-char-2026-06-08/README.md`; `handoff/vq2-ensemble-support-2026-06-19/REPORT.md`; `handoff/deploy-obs20-2026-06-19/REPORT.md`; `handoff/spike-vertical-2026-06-18/REPORT.md` (fix-surrogate covered above under duplication)
- **Issue type:** prune · **Severity:** low
- **Evidence:** All durable build/research/ops reports with verdicts consistent with current memory: cowork boresight-calibration (offline-bake ADOPTED → bake −0.25 merged 5764291) + perception-reward (informed the look-at primitive); blender-dataset/render (photoreal generator P5 d64e1dc + the base-2000 that fed the champion); simops-mastery (10/10 clean cycles; σ_lat 0.10 gate-4 band; camera-pointing is the binding constraint; GT velocity available); perception-char (explicitly SUPERSEDED-by-VISION-PKG2 in project_phase2:112); ensemble-support (EnsembleGateDetector STEP-1 06876bc); deploy-obs20 (obs[17:20] productionized dc56ce7); spike-vertical (loop closed 95dcc93, golden frozen green_gate 947).
- **Recommended action:** **Keep all.** Each is merged-or-actioned and banked; no stale gate-4/σ verdicts. Two minor carry-forwards to confirm-parked (both already banked): perception-reward's d_lock~5 m terminal-weight was later superseded by the inc8 band-pass terminal_weight; fix-surrogate's accept_rlo=16.2 refined to 12. perception-char (28 files, mostly JSON/stdout dumps) — findings SUPERSEDED-and-banked, so the raw dumps are safe to archive (cite: project_phase2_rl_vision_decisions.md:89-112).
- **Info-loss risk:** none — all verdicts banked. perception-char raw JSON dumps are intermediate; the distilled twin-one-liner + VISION-PKG2 supersession is in memory. Preserve perception-char/README.md commands.txt provenance if archiving the dumps.

## [LOW] oneoff-regression-suite — completed mechanical task
- **Location:** `handoff/oneoff-regression-suite-2026-06-14/REPORT.md:11-47,99-130`
- **Issue type:** prune · **Severity:** low
- **Evidence:** Pure process record: "3 remaining regression scripts promoted from handoff/ultracode-substrate-audit-2026-06-13/regression_suite/ into tests/ (CR1-01, CR3-02, P3-C06)"; "now ALL 11 in tests/"; "703 passed, 42 skipped … 0 regressions"; branch oneoff-regsuite cc3a25a pushed, awaiting merge. The actual TEST FILES live in tests/ on the canonical suite (now 1077+).
- **Recommended action:** Keep/archive — the deliverable (3 tests/test_confirmed_*.py) is in the canonical suite on main; the handoff REPORT.md is a one-line process record. Confirm the 3 files are merged to main (referenced by system-id pair-5 as GREEN), then archive the handoff dir.
- **Info-loss risk:** none — the promoted tests are the artifact and live in tests/. The handoff dir is process metadata only.

---

## Disposition quick-reference (fixtures & provenance that MUST stay in place)

- **KEEP (active data fixtures, archiving BREAKS tests/scripts):** `shadowpc-firstcontact-2026-06-02/track_map.json` (26 consumers, ONLY copy of course map) · `shadowpc-followups-2026-06-05/{sysid/, task2_frames/}` (test_twin_fit, characterize_perception, task2_gate_pnp) · `shadowpc-postfix-dataset-2026-06-12/extracted/` (test_diagnose_session, _audit_io positive control) · `shadowpc-refit-dataset-2026-06-12/debug_obs_17runs.zip` (audit fixture, only tracked copy).
- **KEEP (named provenance for shipped constants — cited by source/test doc comments):** `shadowpc-characterize-sweep/WRITEUP.md` (rl_plant.py:122, test_super_rate.py:3) · `shadowpc-twin-falsify/{WRITEUP.md, profiles/}` (twin_fit.py:378, test_measured_aero.py:3, diffaero_dynamics.py:405, rate_sysid.py:248) · `shadowpc-live-deploy-diag/{WRITEUP.md, mixer_probe.json}` (test_mixer.py:2) · `shadowpc-bridge-retest/probe_summary.md` (project_rl_increment_history.md:141).
- **KEEP (active engine / design-of-record, annotate-as-superseded but do NOT archive):** `margin-closure-envelope/margin_envelope.py` (the double-count engine, cited by the correction) · `ultracode-gate-relative-pipeline-design/BLUEPRINT.md` (design-of-record for merged C2 chain) · `fix-surrogate/REPORT.md` (rl/fix_surrogate.py design-of-record) · `kf-pfloor/REPORT.md`.
- **KEEP-IN-ARCHIVE (only copies of one-off diagnostic scripts — archive the dir intact, do NOT prune the .py):** `laptop-s18-thrust-lapse/scripts/` (10 s18_*.py) · `laptop-frame-audit/scripts/` · `laptop-s17-mixer-inc6/{mixer_probe2.json, fit_mixer.py}` · `laptop-inc7-env/scripts/inc6_geom_eval.py` (cross-ref'd) · `laptop-training-doctrine/{QUESTIONS.md, scripts/, q1_trace.npz}`.
- **OPEN ACTIONS to surface (not archivable content):** (1) the two LIVE evaluators default `sigma_target=0.08` — fix to ~0.15. (2) sigmap0-adroit "5th harness bug" sbatch edit possibly uncommitted — verify it landed. (3) ultracode-substrate-audit REPORT.md:63 COLL_MAP slug-collision finding — re-home or confirm banked. (4) flyrl-autonomy-hardening §4 ShadowPC live-verify checklist — confirm completed (index_control_sim.md:55 says §4 LIVE-VERIFY PASS 5/5, likely done). (5) memory reconciliation: MEMORY.md:8 residual "2nd lever = near-field GATE estimator" sub-clause + index_vision_estimator.md:285-287 stale 0.08/near-field text + project_rl_increment_history.md:117 mixer-probe2 "uncaptured" line.
