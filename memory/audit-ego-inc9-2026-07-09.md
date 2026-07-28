---
name: audit-ego-inc9-2026-07-09
description: Fresh-eyes adversarial audit (15-agent workflow) of the ego-inc9 single-gate campaign — 5 of 8 load-bearing claims revised/refuted; KF answer; ranked plan to 90%
metadata: 
  node_type: memory
  type: project
  originSessionId: c864e5bb-f53f-4b54-b78a-a2216bb2c906
---

# Ego-inc9 fresh-eyes audit (2026-07-09, gate-threading-review session)

15-agent workflow (5 readers + 8 adversarial verifiers + critic + planner) ordered by Fengyou.
Full agent outputs (verdicts/critic/plan/readers, ~200 KB): session scratchpad
`C:/Users/Fengy/AppData/Local/Temp/claude/C--Users-Fengy-Downloads-Projects-Anduril--claude-worktrees-cool-heyrovsky-e08624/c864e5bb-f53f-4b54-b78a-a2216bb2c906/tasks/audit_extract/` (EPHEMERAL — copy out if needed).
Durable raw-evidence anchors live in `C:/Users/Fengy/Downloads/Projects/Adroit/adroit-connector/command_history.log`
(vglp05 launch :74210-74217 · vglpan SKIPPED banner :76596-76598 · vns01/31 bit-identical DET_EVAL :76584-76592 ·
DiffAero runner.py verbatim cat :37105-37137 · vglpanEV DET_EVAL :74814 · vglpsl launches :74933).

## Verdicts (claim → verdict, conf)

1. **C1 perception-limited → CONFIRMED 0.8**, but RENAME: *estimator-CORRUPTION-limited* (noise_scale=0 also zeroes
   misses/teleports/bias/IMU-drift — availability + bias are bundled with sigma). vglpns0 0.63/0.32 m vs vglpan
   0.20/0.88 m is real (local watcher artifact, dose-divergence from identical step-30 start). n=1 seed.
2. **C2 anneal-inert bug + 4f91bd8 fix → CONFIRMED 0.93.** DiffAero's VENDORED RecordEpisodeStatistics
   `__getattr__` RAISES on underscore names (not gymnasium's forwarding). vglpan.out shows exactly one line:
   `SKIPPED`. Residual hole: NO post-fix `ON` banner ever observed in a real run (vcza2/vnsb died first).
3. **C3 r_perc rejected → REFUTED 0.78.** vperc runs carried a HIDDEN discrete zero 4.0→0.75 jump (stage set
   0.75 "champion settings" but real vglpan trained at 4.0) on the 6/6-fatal vglpan re-warm base; the no-r_perc
   control (vns01/31) collapsed identically. r_perc = UNTESTED-CLEANLY, back on the board (field's biggest
   centring lever, 0.5→0.12-0.22 m). Farmability plausible in OUR terminal-EV structure but not demonstrated.
4. **C4 speed-lever refuted → REFUTED 0.8.** rw_vmax_mps is a reward-CREDIT clip (ego_reward.py:358-370), never
   slows the drone; no crossing speed was ever measured. Only the knob is dead; slow-at-plane is UNTESTED and
   targets exactly the near-centred-collision residual mode.
5. **C5 "36% control-clip residual" → REFUTED 0.8.** vglpns0 trained at effective FIXED zero=4.0 (pre-fix, job-id
   ordering proves it) where a 0.32 m crossing earns 99.4% of parabola max AND frame-clips pay +19.2 (parabola
   bypasses clip terminals, ego_reward.py:797-801). The 0.32 m floor is a reward-indifference reading. The missing
   calibration run = **vczns0** (noise 0 + REAL anneal 4→0.75) — never run.
6. **C6 DET_EVAL faithful → CONFIRMED 0.85** (same live env instance, n_ep~20k, counting verified line-by-line).
   Caveats: 300-step default censors; thread is point-mass geometric, not collision-simulated; offline
   ego_render_rollout harness broken with root cause NEVER found — do not resurrect.
7. **C7 noise contract "measured/valid" → REFUTED 0.87.** Sigmas = VQ1 Track-3 fit 2026-06-14 (126 fixes,
   10-26 m band); 5-10 m endgame band had n=5 showing ~7× WORSE sigma than the floors used there; vert 0.2816
   contested by boresight-closure (pooling artifact folding the −0.215 one-signed calibratable bias; true
   lat/vert ≈ 0.19/0.10); miss=0.10 & teleport=0.002 INVENTED (no provenance) — and miss=0.10 contradicts the
   only measured accept stat (P(accept|in-image)≈0.10, i.e. 90% miss) by ~9×; content latency (70-620 ms,
   0.15-0.6 m effect) unmodeled. Only in-domain piece: the VQ2 IMU model (run 024203).
8. **C8 warm-start-collapse law "any discrete change, 6×" → REFUTED 0.82.** Real tally: 2 clean single-lever
   collapses (vgap8, vglp3); 1 fresh (vgk10a); 2 carried the hidden zero-jump (vglpsub, vglpc40); 4+ discrete-change
   warm-starts SURVIVED (vgctrw, vglp4, vglp3c, vglpsl*, vglpns0-huge-obs-change). Weaker true law: a discrete
   reward change that NEWLY PUNISHES the warm policy's current operating point can detonate, base-dependently.
   "Continuous anneal is safe" has ZERO clean positive evidence (the flagship datum never ran) — vcza2 is its FIRST test.

## New hard discoveries (not in the dossier)

- **vglp05 was an accidental A/A replicate of vglpan** (launch line: `EXTRA=++env.cross_zero_end=0.5` = no-op
  through the inert hook; command_history.log:74210). vglpan 0.20/0.88 vs vglp05 0.07/1.7 ⇒ **single-seed
  run-to-run variance ~2.8× thread / ~2× xoff is DEMONSTRATED**. Every n=1 verdict must be re-graded against it.
  (vglpan6≈vglpan shows the variance is basin-bimodal, not uniform jitter.) vglpshp had NO zero-anneal component
  (live std-floor lever only) — its collapse is clean n=1 evidence against endgame std-sharpening.
- **Obs blackout masking**: peregrine_racing_ego.py ~:300-304 zeroes rel_pos on INSTANTANEOUS non-detectability
  (`keep = valid & det & conf>0`) — the estimator's coast-through-gaps NEVER reaches the policy; the crossing
  endgame is flown on zeroed gate slots, even at noise_scale=0. Part of the 0.32 m "floor" is this cliff.
- **ego_estimator._rel_var is DEAD state** (written, never read) — latency_cov_inflate=1.5 is a silent no-op;
  the emulator has no covariance feedback and NO innovation gating (deploy HAS chi2 gates → teleports pessimistic).
- **[b,0,b] bias bug-in-effect**: one scalar draw sets lat & vert bias identically, sign-randomized; measurement
  says lat −0.034 / vert +0.195 ONE-SIGNED (calibratable). Over-injects lateral ~5×, fabricates correlation, and
  converts a calibratable systematic into irreducible DR (teaches hedging, not compensation).
- **"~0.28 m measurement noise" is the wrong statistic** — it's the 3-D norm incl. 0.33 m smoothed DEPTH (doesn't
  project into head-on crossing offset). Crossing-relevant in-plane ≈ **0.18 m RMS, BIAS-dominated**
  (smoothed white noise only 0.04 lat / 0.11 vert — filter math has little headroom).

## KF answer (Fengyou's direct question)

Three estimators exist; only one is a real KF: (1) DEPLOYED world-NED 6-state LinearKF (05ed750: Joseph-form,
OOSM RewindKF, TWO chi-square innovation gates) — competent but MAP-ANCHORED, unusable on the VQ2 wire (no map);
(2) the LIVE map-free flight path uses a plain EMA (gate_seeker track_ema_alpha=0.5) — no per-gate body-frame
relative KF exists in deploy; (3) training ego_estimator = fixed-gain low-pass EMULATOR, not a filter.
**Filtering is NOT where the error lives** — the budget is bias (unfilterable) + availability/blackout + closed-loop
amplification. A "better KF" buys little; bias calibration + availability + masking fix + in-domain contract
measurement is the leverage. Do NOT relay a generic "better Kalman filter" ask to the vision commander.

## Plan skeleton (agreed positions; full plan in scratchpad plan.md)

- **DP0 (needs Fengyou Duo)**: daemon up → BEFORE launching: grep vcza2/vnsb .out for `ON` banner + death cause;
  pull vglp05 .hydra (confirm replicate); tbdump mean_speed for vglpsl5/sl3.
- **W0 local (no cluster)**: W0.1 annealable frame-clip terminal under parabola + once-per-gate latch ·
  W0.2 blackout-masking fix (feed coasted estimate + decaying conf, flag, default-off) · W0.3 per-axis measured
  bias model · W0.4 DET_EVAL steps 300→1200 + SKIPPED→raise + _unwrap_env_with unit test vs mock wrapper ·
  W0.5 dossier correction propagation · W0.6 fire vision relay.
- **Wave 1 (2 slots)**: R1 = vcza2 relaunch AS-IS (first real anneal test) · R2 = **vczns0** (noise 0 + real anneal
  — THE calibration anchor; expect floor <0.32 m). **vnsb as-is: NO** (curriculum over a wrong contract; vglpns5
  mid-dose collapse is the regime a ramp must traverse; Swift trains at fixed realistic noise). Completes the
  {noise 0,1}×{zero fixed-4, real-anneal} factorial with vglpan/vglpns0.
- **Vision relay asks (ranked)**: (1) measured VQ2 sigma.json-v2 incl. 5-10 m band, SIGNED per-axis bias,
  P(accept|in-image) per range, latency mixture (= their own standing "refit the twin" directive, never executed);
  (2) vertical-bias boresight bake (~0.215 m); (3) close-range fix availability; (4, lower) per-gate body-frame
  relative filter w/ innovation gating for deploy parity.
- **Wave 2 (contingent)**: clean r_perc retest (warm vglp4 @ native zero=4, w=0.05, record oob failure MODE) ·
  masking-fix flight · corrected-bias flight · vglpns0 seed-1 replicate · single-channel ablations (miss-only /
  bias-only). Multi-gate N=2: trigger = single-gate det ≥50-60% at corrected contract OR post-Wave-3 plateau.
- **90% feasibility arithmetic**: need mean xoff ≈0.15-0.20 m with σ≈0.10-0.12 m per axis. With calibrated residual
  bias ~0.1 m → reachable (~99%/axis); with full 0.19 m sign-random bias → likely NOT. The bias story is
  load-bearing, not hygiene.

## EXECUTION (same day, afternoon session — coordinator = Fable, workers opus/sonnet)

- **Overnight "job deaths" were a FALSE ALARM** — the watcher died with the daemon; both jobs COMPLETED (sacct 0:0).
  vcza2 (real anneal, noise 1) = thread 6.5%/DET 0.078 after a 100%-FLOOR-dive DURING THE HOLD (config-identical
  window to vglpan) → reg-audit (opus, airtight): commits 7a26d4a/983faf6/4f91bd8 CLEARED — bit-identical reward
  assembly, zero added RNG draws, hold verified no-op ⇒ **pure GPU-nondeterminism basin variance** (same mechanism
  as vglp05, whose hydra PRIMARY-confirmed the A/A: sole delta `++env.cross_zero_end=0.5`, inert). vnsb collapse =
  live noise-curriculum mechanism + BUNDLED std_hold 0.06/floor 0.02 override (stage comment claimed "isolates the
  noise lever" — false). **4f91bd8 fix verified live (ON banners) — vcza2/vnsb were its first real runs.**
- **Wave-1 vczns0 pair (noise 0 + REAL anneal 4→0.75, warm vglp4, seeds 0/1): DET thread 0.574 / 0.403, xoff
  0.35–0.36 m, collision 0.42/0.57, miss ~0.01–0.03, no detonation (anneal SAFE, 2/2).** Verdict: reward-slack
  explained part of the old floor (threaded crossings centre ~0.23 m) but **a genuine non-reward residual exists
  under perfect perception — noise-0 ceiling ≈0.40–0.64 across n=3, NOT ≥0.85**. Anneal not better than fixed-4
  at noise 0 → anneal question PARKED. Seed 0 still climbing at cutoff (0.32→0.57 back half).
- **W5 offline characterization (report: handoff/audit-ego-inc9-2026-07-09/estimator-characterization/):**
  in-plane 0.23 m (0.18 teleport-free), bias share 0.88–0.90, quadrature closes (bias 0.15 + teleport 0.15 +
  sigma 0.09); miss_prob/IMU = ZERO in-band error; coast through ≤1.5 s blackout costs ~0.01 m; **terminal blind
  onset 0.7–2.2 m in ALL geometries** (steep descents 60–76% blind over final 5 m) ⇒ 0.5 s horizon expires
  pre-crossing at slow-lap; K=1 reacq snap 1.7–2.6×, teleport-as-first-fix up to 32 m ungated (emulator lacks the
  deploy chi² gate — gate it, don't zero the prob); EMA realizes σ/√(2N−1), 1.37× tighter than DESIGN.md text.
  Buy-down: signed-bias calibration (0.23→~0.10) > obs-coast+horizon > teleport gating > σ recal > skip miss/IMU.
- **mean_speed closure (C4-b):** vglpan 13.5 / vglpsl5 12.0 / vglpsl3 6.8 m/s — the cap DID slow the drone and
  centring got WORSE (2× slower → 4× worse). Speed lever demoted hard; slow-AT-PLANE still technically untested.
- **Code landed on chaum (deployed==committed now, sha256-verified): 7dbd69a** (snapshot of the
  deployed-but-uncommitted campaign working set) · **5ea4b16** rw_clip_terminal (graded anti-clip under parabola,
  annealable via `clip_pen_anneal`) + rw_parabola_latch (re-cross farm defense; env `parabola_paid` buffer clears
  on advance+reset+truncation) · **c74853b** `+env.ego_obs_coast` + `+env.ego_stale_horizon_s` + 
  `bias_model='measured'` (independent one-signed per-axis, sigma.json values; legacy RNG stream proven unshifted)
  · **1e44ae7** STRICT anneal hooks (requested-but-unattachable now RAISES; L16 codified) + `_cross_zero_schedule`
  start≤0 fallback fixed (was an unconditional update-0 jump!) + eval_det_steps 1200 · **5df483c** coast0 stage.
  All default-off/byte-identical (golden tests); 34 new + 83 existing tests pass.
- 🎯 **Wave-2 IN FLIGHT: vcoast0/vcoast0s1 (jobs 3299083/3299084)** = vczns0 recipe + coast package
  (`ego_obs_coast=true`, horizon 1.2 s) at noise 0, seeds 0/1, control = vczns0 pair. THE question: does the
  0.42–0.57 collision residual convert once the endgame isn't flown blind on zeros?
- **Vision relay box DELIVERED to Fengyou** (sigma.json-v2 w/ 5–10 m band + signed bias bake + close-range
  availability; explicitly NOT a generic KF ask). Pending his paste to the vision commander.
- 🚩 Ops lessons: serve-daemon = parent+child PAIR (killing either kills both — check ParentProcessId before
  pruning "zombies"; I killed Fengyou's live daemon learning this); tqdm-dominated .out files — never tail raw
  (961 KB single line), grep patterns or read TB via tbdump (works mid-run); DONE-marker `grep -c` returns 2
  (bash -x trace echoes the marker line).

## WAVES 2-4 (2026-07-09 evening) — THE NOISE-0 VERDICT: CONTROL PRECISION

- **Wave-2 vcoast0 (coast+horizon 1.2 s @ noise 0): DET 0.522 / s1 COLLAPSED** (det fly-away oob 0.76 while
  stoch stayed bounded — determinism-gap pathology). Coast converted NOTHING on threads; plane-misses → 0.002,
  xoff 0.33. **Endgame blindness was NOT the binding layer** (policy already extrapolated ballistically).
- **Wave-3 vclip0 (clip-penalty anneal 0→20 @ noise 0): DET 0.478 / 0.322, collision 0.51/0.63, miss 0.01-0.05.**
  A −20 clip cost neither stopped clipping nor converted clips to misses.
- 🛑 **NOISE-0 LEDGER COMPLETE (8 runs, 4 configs): fixed-4 0.641 · anneal 0.574/0.403 · +coast 0.522/collapse ·
  +clip-pen 0.478/0.322 — ALL in the 0.32-0.64 band, collisions ~0.35-0.63 immune to every reward/obs lever.
  VERDICT: at ~13 m/s the binding layer is CONTROL PRECISION (cannot place the crossing inside ±0.42 m
  >~60% of the time even with perfect perception + punished clips). Config spread = basin variance.**
  Untested single-gate lever remaining: slow-AT-PLANE (evidence leans against; conflicts with moonshot).
- **Wave-4 IN FLIGHT (pivot trigger fired): handoff_drill0 (fresh) vs handoff_drill0w (warm-from-vglp4),
  jobs 3299211/12** — 2 gates 10-20 m @ noise 0, EASY reward, 2000 upd (commit dee8c33). Datum: handoff
  mechanics baseline + does filling the always-empty second obs slot detonate a warm policy (H6)?
  Multi-gate rationale: CPC/TOGT sequence-defined crossing; Song N=2-obs crash reduction.
- **Wave-4 FINALS: BOTH drill arms DEAD** — hd0 fresh DET 0.000 (collision 0.53/miss 0.43); hd0w warm DET 0.000
  **oob 1.00** (det fly-away detonation; the arm swapped vglp4's parabola reward for the EASY drill reward AND
  filled the always-empty slot 2 — reward-swap likely dominant). **EASY-reward handoff_drill = falsified vehicle;
  the multi-gate step needs the PROVEN lpara stack ported to 2 gates** (design agent building `dual_gate_fullstack0`:
  racing line spanning gates, per-gate parabola+latch, warm vglp4, noise 0, NO anneal in v1).
- **Overnight pair (3299231/32): vczns0s2** (3rd anneal-arm seed; existing 0.574/0.403) + **vczext** (8000-upd
  extension of the still-climbing seed 0) — zero-design-risk calibration completion.
- New commits: a1d1da2 (clip0 stage) · dee8c33 (handoff_drill0/0w). Deployed==committed throughout.
- 🚩 Det-fly-away pathology (2× now: vcoast0s1, hd0w): collapsed basins show stoch-bounded but det-mean flies away
  — DET_EVAL is the ONLY reliable collapse detector; never trust the stochastic box-exit alone for basin health.

## 🛑🛑 LATE-NIGHT REVERSAL — THE "CONTROL CEILING" WAS A CONVERGENCE ARTIFACT

- **vczext (vczns0 recipe, 8000 updates instead of 4000): DET thread 0.8366, collision 0.162, miss 0.002,
  oob 0.000, xoff 0.24 — and STILL CLIMBING at cutoff** (box 0.643@4630 → 0.786@7930, no plateau).
  The 0.5-0.6 "control-precision ceiling" was the 4000-UPDATE BUDGET, not physics: every ledger run (incl.
  waves 1-3) was an unconverged snapshot. The "not convergence-limited" belief (vglpan6 6000==4000) was
  fixed-4/noise-1/inert-era and does NOT transfer. **90% single-gate @ noise 0 looks budget-reachable.**
- Consequences: (a) wave-2/3 "converted nothing" verdicts hold ONLY at matched 4000-upd budget — asymptotic
  value of coast/clip unknown; (b) **the real-noise champion 20% was ALSO a 4000-upd snapshot** — whether the
  DEPLOYABLE number climbs the same way is now the campaign's #1 open question; (c) anneal-arm seed spread
  (0.574/0.403/0.603 + s2's 0.603) partly reflects where each seed sat on an unconverged curve.
- 🎯 **Queued (behind the dgfs pair): vcz16 (noise 0, 16000 upd, job 3299282 — the 90% attempt) + vn16
  (REAL noise contract, 16000 upd, job 3299283 — the deployable-ceiling probe).**
- Third anneal seed vczns0s2: DET 0.603 (arm now 0.574/0.403/0.603).
- **Wave-5 dgfs pair FINALS — H6 CONFIRMED 2/2, clean attribution:** dgfs0 DET 0.0001 / dgfs0s1 0.0000
  (collision 0.36-0.46, miss 0.54-0.64, oob ~0 — wide-missing from step 0, NOT detonation/fly-away; no reward
  swap this time). **A policy trained with slot-1 always-empty loses gate-0 competence the moment slot-1 fills.**
  Stage machinery itself sound. Fix ladder (next session): (1) SLOT-CURRICULUM warm-start — mask slot-1 in obs,
  anneal it in via the proven hook machinery (small patch); (2) FRESH dual_gate_fullstack0 at 16k budget
  (credible post-reversal); (3) future single-gate policies should train with slot-1 noise-filled for
  forward-compat. vcz16 + vn16 running overnight (~3.5 h each).

## VISION RELAY-BACK (2026-07-09, via Fengyou) — contract-changing facts

- **No GT on the VQ2 wire, EVER** (pose-denied: HIGHRES_IMU + ACTUATOR + HEARTBEAT + video + COLLISION only)
  ⇒ NO deploy recording can yield bias/sigma; the 2026-06-14 contract was fittable only because Track-3 was
  given-pose. The "refit the twin on VQ2 logs" directive was unexecutable as written. Substrate options:
  (A) fresh given-pose capture (task2 40-frame VQ2 bundle w/ drone_position_ned proves the mechanism existed
  2026-06-05; current-sim availability unverified) · (B) Blender-synthetic GT (dense, sidesteps pose-denial;
  domain-gap anchored to task2 frames) · vision cmdr recommends (C) both. RL priority sent: P1 latency/cadence
  mixture (no GT needed, deliverable now) · P2 vert-bias VQ2 transfer on task2 · P3 Blender dense curves
  ([[project_blender_vq2_data_pipeline]] trigger) · P4 M-model accept-rate 5-15 m.
- 🛑 **VERTICAL BIAS ALREADY BAKED IN DEPLOY**: −0.25 m METRIC TRANSLATION at `frames.BORESIGHT.vert_offset_m`
  (NOT R_camera_from_body — the angular form was empirically REFUTED: rel_vert flat over 14-26 m; a tilt would
  scale ×range), applied on the gate_relative_inplane_fix path (= the VQ2 map-free path). VQ1 post-bake residual
  **+0.004 m**. If it transfers to VQ2 (P2 check), the RL training contract's 0.19 sign-random bias models a bias
  the deployed system REMOVES — the biggest single pessimism in the noise budget (bias = 0.15 of 0.23 m). The
  `bias_model='measured'` knob should then get a THIRD mode ~'residual' (≲0.05 m) pending P2's number.
- Vision's new **M detector** trained on partial/clipped gates to push detection past the geometric blind onset;
  their EMA-vs-KF+χ² deploy gap acknowledged, per-gate body-frame filter deferred as agreed.

## STRATEGY REVIEW (2026-07-09 night, Fengyou-ordered 11-agent workflow; extracts → session scratchpad strategy_extract/, full → tasks/wbhvrz0m0.output)

- 🛑🛑 **THE CLOCK: VQ2 closes ~mid/late JULY (~2-3 weeks), NOT November.** Sept = Physical Qualifier (SoCal, real Neros, ~100 TOPS); Nov = Final (Columbus, $500K). TS-003 §9.4 VERBATIM: pure TIME-TRIAL, **UNLIMITED attempts**, best valid time counts, free non-counting TRAINING mode (§9.2). Gate contact = whole-run invalid (FAQ-sourced). Cut line + exact close date SILENT → organizer email.
- 🛑 **STALE MEMORY RULE: the 2026-06-15 "time-lock" HARD RULE (repo memory reference_competition_materials.md:35, index_strategy_meta.md:9) is SUPERSEDED by §9.2/§9.4** — it currently blocks completing full runs in dev. Retire it (verify Training/Competitive UI split once first).
- **BANK-FIRST DOCTRINE (both judges): the moment the stack completes 20 gates at ANY speed, fly Competitive and bank the time — after banking, invalid fast attempts cost nothing → all-in Swift-class speed.** Reliability matters only until the bank.
- **JUDGES CONVERGED (opposing priors, same portfolio): KEEP the trunk** (YOLO-pose M + IPPE/P3P PnP + bias bake = champion-unanimous family per Swift/MonoRace/"On Your Own"; egocentric APPO infra; deterministic-track doctrine). **NO vision swap now** — availability may be a THRESHOLD problem (A6: sweep score_thresh 0.25/kpt_conf 0.5 accept-rate-by-range on A18/A19 recordings BEFORE any detector work); red_glow classical detector (loving-galileo worktree only) parked with revive triggers; YOLO latency ladder (TensorRT/imgsz/crop) before any swap.
- **IMMEDIATE SEQUENCE (72h): vn16 fork read (+1 replicate — n=1!) → compliance package (remove human GO = §7 DQ; FLOSS/AGPL disclosure; organizer email) → FRESH dual-gate 16k ×2 seeds (composition kill-shot; parity ⇒ 20 gates = curriculum) → ONE obs-contract-v2 detonation (prev-action + K=3 frame stack + age/validity flags + masking fix + multi-gate slots — current 21-D obs is BELOW Swift/Geles baseline: no prev-action, no history!) → track-prior table keyed by active_gate_index (the cheap "map") → wire flights counting gates → BANK.**
- **MAPPING (Fengyou's iterative-map idea): validated as real but ~Sept horizon.** ~70% built: src/racer/gate_mapper.py case-C BA (pose-aided validated 0.20 m; case-C failure = graph disconnection which VQ2's 10-20 m spacing REMOVES; scale free from known 1.5 m gate; active_gate_index = free association). Do NOW (cheap): legality question (persistent training-derived map vs §9.2 code audit — SILENT in spec) + offline case-C probe on existing logs. Map enters ONLY as training-course truth + priors — NEVER the world-NED-KF deploy path.
- **RL MEMORY (Fengyou's "GVL"): disambiguated** — GVL=VLM value learner (red herring); Sutton-GVFs (no racing precedent, marginal); internal GVF=racing-line field (name collision — police in relays). Champions (Swift/MonoRace/Geles) ALL fly FEEDFORWARD; memory lives in the FILTER + availability engineering. Adopt: frame stack + prev-action via the v2 bundle; GRU only if stack plateaus; distillation/world-models = reject for VQ2.
- **DROP: noise-0 single-gate polish** (calibration instrument now), slow-at-plane, dense SLAM/mono-depth, end-to-end pixels, pose-tuple regression, SkyDreamer-class world models, generic better-KF.
- **AUGUST (post-bank): sim-to-real audit begins** — the CTBR tail-first alias + R_y(π) quat conjugation are sim-only self-consistencies that DIE on real hardware by construction; GT substrate; sigma.json-v2; red_glow as real-hardware redundancy.

Related: [[vq2-rl-commander-2026-07-05]] · [[ego-inc9-diagnostic-2026-07-07]] · [[index-rl-training]] · [[index-vision-estimator]]
