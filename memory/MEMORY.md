**Peregrine** — Anduril **AI Grand Prix** autonomous drone-racing entry (May–Nov 2026; fun). Repo `github.com/Hat000/Peregrine` at `C:\Users\Fengy\Downloads\Projects\Anduril`. Orientation index; detail in topic files.

## READ FIRST — current state (2026-06-13)
**🏁 VQ1 PASSED** (dashboard-confirmed). CTBR stack threads 6-gate course — valid finish ~35.3 s, in-plane miss `[0.37, 0.05, 0.06, 0.03, 0.10, ~0.71]` m (all < 0.75 m). Phase 2 = **RL + VQ2 vision**.

**✅ INC7 LIVE CONFIRMED (2026-06-13, commit 7fe90df):** Standing 5/5 FINISHED, gate-3 NEVER crashed; F1 9.97 s (warm), F2–F5 ≈11.45 s (fresh-reset = deployment estimate); Bridge 2/2. GEOMETRY-HONESTY THESIS LIVE-VALIDATED. Finding-A (gate-3 D-offset) FALSIFIED (Fengyou-verified): bottom-vs-opening artifact; all 6 gates ≤0.37 m in-plane; track_map trustworthy. Bimodal lap times = physics-state/HOME-reset (benign). 🚩 Budget MORE seeds for inc8-class retrains (2/3 viable; narrow basin). Detail [[rl-increment-history]] §INC7-LIVE-CONFIRMED.

**🚩 ESTIMATOR-RACESPEED VERDICT (2026-06-13; handoff/ultracode-estimator-racespeed-2026-06-13/REPORT.md):** Case-C absolute world-frame nav = **NO-GO at ANY speed** — deployable in-plane ~0.55 m vs 0.155 m gate-4 margin (3.5×); SPEED-FLAT. **FIX = GATE-RELATIVE** (map bias drops out → 0.11–0.21 m RMS = **CONDITIONAL-GO**, velocity-prior-sensitive; straddles 0.155 m). **RewindKF MANDATORY.** Policy obs must be gate-relative → P4-C05 is FOUNDATION. **🚩 ORGANIZER-PIVOT: build gate-relative REGARDLESS** — Q① = load-bearing vs free-insurance only. Detail [[project-phase2-rl-vision-decisions]] §ESTIMATOR-RACESPEED + §ORGANIZER-PIVOT.

**🚩 S2 DECIDED + CROSS-CUTTING REPRIORITIZATION (2026-06-13):** Binding VQ2 risk = ESTIMATOR. **S2 architecture = `staged_monolithic_then_decomposed` (HIGH confidence).** Speed gap = TILT ENVELOPE not architecture. 4.27/4.55 s bounds FALSIFIED (linear-plant fiction); honest corrected-aero bound ~4.6–4.7 s. Detail [[project-phase2-rl-vision-decisions]] §S2-DECISION + §PLANNING-TOGT-S2.

**✅ AUDITS DONE (2026-06-13):** FRAME-AUDIT/CRAB-DIAG settled — ODOMETRY quat R_y(π)-conjugated; crab ~55° = near-optimal; gate-3 barrier root cause = point-mass L-inf training vs volumetric sim contact; S18 VOIDED. Substrate EXTERNALLY CLEAN 68/68; COLL_MAP NOT A BUG. Autonomy hardening DONE (see ④ below). Vision-frame-fix DONE (8d7b0b3). Regression suite DONE (dc4b532). Detail [[rl-increment-history]] §LAPTOP-FRAME-AUDIT; [[project-phase2-rl-vision-decisions]] §SUBSTRATE-AUDIT + §AUTONOMY-READINESS.

**✅ SIM BUILD 1.0.3364 REGRESSION FIXED:** `launch_ramp_s=0.6 s` + `--rate 100`. Detail [[project-ctbr-control-sysid]].

**SIM OPS (full mechanics [[reference-sim-ops]]):** unattended PROVEN. 31000 restart: Win32 foreground + Enter×2; sim AUTORESETS on gate contact → guard cuts RL commands instantly; use raw ODOMETRY rate. **🚩 SPAWN-ARTEFACT:** gate-3 hard-collision reset → next respawn 0-tick crash; NOT policy failure.

## The flyable stack (durable — RL training substrate)
- **Sim interface** (all 5 RESOLVED; full spec [[reference-sim-interface]]): pos+vel GIVEN (LPN 97 Hz + ODOMETRY 75 Hz); **🚩 ATTITUDE.pitch sign-inverted → use ODOMETRY quat**; 6 gates, 2.72 m OUTER / ~1.5 m inner; course DESCENDS 26 m; clocks INDEPENDENT; video dedup by frame_id; sim PAUSES off-race.
- **Fly on CTBR** (`SET_ATTITUDE_TARGET`; sim ACRO). OFFLINE TWIN (`src/racer/twin.py`/`twin_fit.py`): **rate_gain [2.50,2.50,2.23], rate_sign [+1,+1,+1] (vanilla), τ=0.019 s, hover=0.2656, linear_drag≈0.21/s**. **🚩 rate_gain amplitude-DEPENDENT: super-rate map `g(|c|)=G0/(1−s·min(|c|,π)/π)`, s≈0.30 ⇒ full-stick ~11 rad/s (NOT 7.85).** Plant: super-rate ✅ S14 + quad drag ✅ S16 + mixer ✅ S17 — COMPLETE; parity gate CLEARED job 3270602. Detail [[rl-increment-history]].
- **🚩 CTBR/VQ1 LEGACY SIGN CONFIG (VQ1-proven, DO NOT "fix"):** `body_rate_sign=[1,1,−1]`/`odo_att_sign=[−1,1,1]`/`odo_rate_sign=[−1,−1,1]`/`ff_gain=2.5` — self-consistent alias, NOT true physical conventions. Arm = `MAV_CMD_COMPONENT_ARM_DISARM` p1=1. **Live config = `FAITHFUL_TUNED_GAINS` (`kp_alt 3.0/kd_alt 1.75`, raw vz) + `−0.4 m alt_offset` + `launch_ramp_s=0.6 s`.** Detail [[project-ctbr-control-sysid]].
- **🆕 RL DEPLOYMENT RECIPE (93023cf; `rl/fly_rl.py`):** ① tanh(mean)→rescale (thrust [0,5], rates ±3.14); ② quat conjugated `q_true=q_raw·[1,−1,1,−1]`; ③ rate sign **[−1,−1,−1]**; ④ wire cmd = `rate_flu·[+1,−1,+1]`; ⑤ collective obs init 0.0; ⑥ 30 Hz; ⑦ resets at rest 1 m before random gate. Detail [[project-phase2-rl-vision-decisions]] §FRAME-AUDIT.
- **🚩 TRUE PHYSICAL CONVENTIONS (93023cf):** ODOMETRY quat R_y(π)-CONJUGATED. True attitude: `q_true=q_raw·[1,−1,1,−1]`; true rate: `ω_true=−w_raw` (gain 0.999); cmd→rate sign [+1,+1,+1]. ODO twist + accel_body PAIR with raw quat. VALIDATION: run `scripts/frame_residual_report.py` after every live session. Detail [[project-ctbr-control-sysid]] + [[project-phase2-rl-vision-decisions]] §FRAME-AUDIT.
- **Alt balloon is OURS.** **🚩 LIVE LATENCY = 2 ticks (67 ms)** (supersedes ~40 ms); offline twin under-models — do NOT trust offline "holds" claims.
- **🚩 ODOMETRY velocity-frame bug FIXED (`c3b5a8e`):** twist BODY-frame, now rotated via `frames.world_vec_from_body_quat`.

**Detector: SHIP v2** (`models/gate_yolo11s_curriculum_v2.pt`, multi-gate; v3 `--hard` = NEGATIVE). **✅ VISION-PKG2 (2026-06-10):** 1.4° attitude lever + 0.40 m cov floor + 32 m cap; over-rejection 15.7%→1.2%, leak 0.53%. **✅ VISION-FRAME-FIX (8d7b0b3):** east bias eliminated. **🚩 `corner_to_center` 180° flip FIXED (37e7ab1).** Detail [[project-phase2-rl-vision-decisions]] §VISION-PKG2.

**✅ GATE MAPPER COMPLETE:** `src/racer/gate_mapper.py` cases A/B/C. Detail [[project-phase2-rl-vision-decisions]] §GATE-MAPPER.

**Git / env:** `main` CANONICAL. **687/687 tests green** (regression-suite promotion dc4b532). `.venv` Python 3.13. `*.pt` + `data/runs` gitignored. Memory mirrored `memory/` ↔ `~/.claude`; ShadowPC via `handoff/`. **🚩 OPS:** concurrent laptop sessions must use separate git worktrees or stagger suite runs. **🚩 BANKING CONCURRENCY:** banking agents must `git add` only their OWN specific paths (never `git add -A`/`git add .`).

**🚩 GATE CONTACT = INVALID RUN.** **🚩 4.27 s TOGT bound + 4.55 s shipped line = FALSIFIED** (linear-plant fiction; real v_max ~39 m/s v² drag wall). Honest corrected-aero bound ~4.6–4.7 s; contact-valid = 4.43 s (gate-4 +0.046 m margin — razor-thin). `rl/reference_line_vq1.json` drag-infeasible + 170° inverted → must be REBUILT. `rl/reference_line.py` loader valid. Detail [[project-phase2-rl-vision-decisions]] §CORNER-PASS + §PLANNING-TOGT-S2.

**🚩 S17 + TRAINING-DOCTRINE:** Mixer per-motor clip u_i=clip(c+S·d, idle, 1); κ_err=0.073. **🚩 S18 LAPSE VOIDED** — do NOT use `--plant lapse`/`dr_lapse`. **🚩 `fly_rl.py` defaults now SAFE (auto-reset OPT-IN; use `rl/submit_rl.py` for judged runs); pass inc7 (or inc6) ckpt explicitly if not using submit_rl.py.** Doctrine: honest contact geometry + structured DR; NEVER reward damping. Detail [[project-phase2-rl-vision-decisions]] §S17 + §TRAINING-DOCTRINE.

**🚩 STYLE ENVELOPE COST (corrected 2026-06-13):** ≤65° roll = ~2.87 s/lap (ledger; inc5 A/B). 60° cone bound ~9.8–10.6 s ≈ inc7 live 9.76 s → gap = CONE, not policy/architecture. Relaxation ladder: (1) rw_tilt 96→48 (~1.4 s), (2) free-cone 60°→75–80° (~1.5 s — THE binding kinematic lever), (3) unconstrained. **COUPLING: faster speed worsens gate-4 margin/σ ratio → estimator accuracy must be verified at each speed rung. PRIMARY MARGIN GUARD = GATE-4** (0.155 m @ r=0.38). Gate-5 CLEAN (0.314 m). Detail [[project-phase2-rl-vision-decisions]] §PLANNING-TOGT-S2.

## NEXT queue — live/pending items
①–⑤ ✅ ALL DONE (2026-06-13):
- INC7 LIVE CONFIRMED — `stage1_inc7_actor.pth` s0, md5 2AFF8D62569BA5FEC769028D72AF50E3, sidecar {act_max_thrust 3.765, act_max_rate 3.14}. sr=1.000 / t_med=9.76 s / gen=0.924 / deployment ~11.45 s. Detail [[rl-increment-history]] §INC7-LIVE-CONFIRMED.
- ShadowPC INC7-live + audits (substrate CLEAN 68/68, autonomy, frame-audit). Detail [[project-phase2-rl-vision-decisions]] §SUBSTRATE-AUDIT + §AUTONOMY-READINESS.
- Vision-frame-fix (8d7b0b3); S2 decided; regression suite (dc4b532, 657→687 green). **🚩 Guards DATA-DEPENDENT** (skip on clean clone/CI/Adroit).
- **LAPTOP-FLYRL-AUTONOMY-HARDENING DONE (commit 7210c1d; NOT merged — held for ShadowPC live-verify).** `rl/submit_rl.py` = authoritative entry (pins inc7, NO `MAV_CMD 31000` on judged path); auto-reset OPT-IN (`--dev-auto-reset`); `telemetry_health()` → SAFE HOVER on stale/non-finite; `obs[12]` = `prev_normed_thrust`. **🚩 DEV-RIG CHANGE: ShadowPC dev batch now needs `--dev-auto-reset`.** MERGE GATE: ShadowPC §4 checklist. Detail [[reference-sim-ops]] §AUTONOMY-READINESS.
⑥ 🚩 **P4-C05 gate-yaw hardcode (GO-BEFORE-VQ2; FOUNDATION for gate-relative obs rebuild):** `obs_from_zup`/`build_obs` hardcode yaw=pi — silently wrong up to 4.22 m on any VQ2/non-pi course.
⑦ 🚩 **CR1-01 + P1-C06 → WINNER-VALIDATION RIDER:** fold yaw-active segment into inc8 fresh-reset live batch; closes absolute yaw wire sign + obs yaw seam.
⑧ **INC8 sub-tasks:** (1) arc-length progress reward; (2) corrected-aero min-snap+TOPP line; (3) rw_tilt 96→48 + free-cone 60°→~70° GATED on gate-4 metric (BSR3 spin-margin gate MANDATORY — widen spin_rate_abort→~9–10 + spin_time_abort→3.0 s); (4) retrain ≥4 seeds; (5) scipy-SLSQP toy MPCC probe. **INC8 BINDING GATE = GATE-4 (0.155 m @ r=0.38, SIMSTART, registration-confirmed).** RL SPEED-LADDER PORTFOLIO: train deliberate envelope ladder; select fastest contact-valid policy the gate-relative estimator can DELIVER. Detail [[project-phase2-rl-vision-decisions]] §S2-DECISION + §INC8-DESIGN + §RL-PORTFOLIO + [[rl-increment-history]] §Phase-0(b).
⑨ 🚩 **ORGANIZER NON-RESPONSE PIVOT:** 3 emails UNANSWERED — build the robust superset, do NOT gate engineering on organizer answers. Gate-relative = GO. Keep ONE nudge ~weekly. Q①+Q⑤+Q-A–D still useful, not blocking. Detail [[project-phase2-rl-vision-decisions]] §ORGANIZER-PIVOT.
⑩ **ShadowPC agenda (SHADOWPC-VISION-CAL):** in-loop latency L + accel_body logging + hardening live-verify + sigma_theta refit + per-gate last-fix distance + 2-corner PnP + Bayesian-IoU extrinsic + yaw-active debug_obs capture + full-lap case-C sim (~37 m/s gate-4 recording = collapses dominant uncertainty). Detail [[project-phase2-rl-vision-decisions]] §ESTIMATOR-RACESPEED.

**🚩 30 Hz = inherited default (DiffAero racing.yaml).** Binds ONLY at ≥30 m/s × last-fix ≤10 m; fixes to 15 m → NOT bottleneck. **🚩 GATING MEASUREMENT queued (SHADOWPC-VISION-CAL).** Detail [[project-phase2-rl-vision-decisions]] §SPEED-CEILING-ANALYTIC.

**🆕 PARALLEL ONBOARD SYSTEMS:** async detector→KF; ROI zoom; MPC-shadow; critic-as-risk. REJECTED: map mutation; ensemble voting; wind estimator; in-race adaptation. Specs [[project-phase2-rl-vision-decisions]] §PARALLEL-SYSTEMS.

## NEXT — Phase 2 (RL + VQ2); VQ1 banked
**🚩 VQ1 = PASS/FAIL (spec §8, 8-min cap); VQ2 = fastest-valid.** Submission = Python stack, unattended, FULLY AUTONOMOUS (§7: human interaction = DQ). Unlimited attempts, May→~mid/late July. **Adroit summer access CONFIRMED.**
- **RL = COMMITTED VQ2 path.** Stage verdicts [[project-phase2-rl-vision-decisions]]; lineage [[rl-increment-history]].
  - **Substrate = DiffAero on Adroit:** `src/racer/rl_plant.py` + `rl/diffaero_dynamics.py`, parity BIT-IDENTICAL; 2048-env PPO ~88.9 K steps/s. **🚩 GPU PREFERENCE:** prefer A100 but do NOT idle — THROUGHPUT beats waiting.
  - **Checkpoint lineage:** inc4 (RETIRED) → inc5 (RETIRED) → inc6 (fallback, bridge live-confirmed) → **inc7 = CURRENT BEST** (LIVE CONFIRMED; `stage1_inc7_actor.pth` s0; standing 5/5, sr=1.000, gen=0.924). **🚩 `fly_rl.py` default still inc4 — pass inc7 explicitly.**
  - **🚩 Retrain footguns:** evals default legacy flat plant — all evals must run map-ON; `/scratch/network/fl3689/peregrine_repo` is FILE COPY not git clone; parity gate CLEARED job 3270602.
  - **S2 architecture DECIDED: `staged_monolithic_then_decomposed`.** Gap: 35.3 s→9.76 s (inc7)→6.89 s (cone tax, OPEN)→~4.72 s (honest corrected-aero bound). Speed gap = TILT ENVELOPE. Detail [[project-phase2-rl-vision-decisions]] §S2-DECISION.
  - **Perception (VISION-PKG2):** world-fix σ≈[0.73,0.47,0.29] m (N,E,D), range-flat to ~24 m; acceptance ~47%, leak 0.53%. Detail [[project-phase2-rl-vision-decisions]] §VISION-PKG2.
  - **STACK-REVIEW-VQ2:** architecture AFFIRMED. Case-C **GO-WITH-CONDITIONS** (binding = unmeasured in-loop latency L; 3 P0 bugs: _initialize crutch/TIMESYNC/cold-start untested; prototypes in `handoff/ultracode-vision-case-c-2026-06-13/`). Detail [[project-phase2-rl-vision-decisions]] §CASE-C-READINESS.
  - **Vision VQ2:** detector→robust-PnP→KF. 🚩 sigma_theta=1.4° re-fit queued. **RewindKF = DEFAULT (ships regardless; ② already built).** [[project-estimator-robustness]].

## Durable strategy & directives
- **Track A = VQ1 floor + VQ2 baseline.** MVP = min-snap line + slow VALID finish > fast invalid.
- **Overfit GEOMETRY, randomize APPEARANCE.** NEVER fine-tune detector on clean VQ1 frames.
- **Planning + speed = VQ2 differentiator; vision = bounded threshold. RL = COMMITTED VQ2 path.** [[project-master-plan]] C1.
- **User directives:** speed > gate-in-view; vision must be excellent; offline processing LEGAL; SLURM only on Adroit. Re-derive every verdict from data; tune offline, fly to verify; bounded actuation, user owns GUI/risk. **Every new-session prompt carries MODEL (with VERSION) + EFFORT.**
- **BUDGET POLICY (rev 5, 2026-06-13):** WEEKLY POOL sole watch; push limits to MAX. **opus-4.8 ≈ 0.5 × fable cost — opus IS the affordable design tier.** PUSH models UP; do NOT throttle. **opus-4.8 = ALL correctness/judgment-dense work** (commander, env/reward, plant, vision math, S2 arch, selection-metric/eval code) at high/max. **Ultracode** available on opus for hardest fan-out/verify. **sonnet-4.6 = low-stakes mechanizable only** (banking, flight ops, git, routine reruns). **haiku-4.5:** trivial ops.
- **🚩 FABLE REVOKED 2026-06-12 (US-gov, PENDING DISPUTE).** opus-4.8 is top tier until resolved.
- **🚩 CANARY PROTOCOL (MANDATORY):** every session addresses **Fengyou** by name in EVERY message. Missing name = context degradation → Fengyou rotates session.
- **BANKING PROTOCOL (rev 3):** workers end with **"MEMORY-DELTA:"** (≤10 lines); commander triages; sonnet banking agent reads handoff. Bank in BATCHES.
- **PROMPT-EMISSION:** when amending any worker prompt, ALWAYS re-emit COMPLETE prompt — never a splice/delta.
- **Spec facts:** VFoV≈58.7° (spec's "90°" = HFoV; camera tilts up). ~100 TOPS onboard. Obstacles exist but don't map monocularly.

**🆕 ADVISOR-TRIAGE OPEN QUEUE:** 🚩 **ESTIMATOR VERDICT BANKED** (absolute NO-GO; gate-relative = CONDITIONAL-GO; RewindKF = DEFAULT; detail §ESTIMATOR-RACESPEED + §ORGANIZER-PIVOT). ① photoreal detector (Blender/Cycles); ② ONNX/TRT latency; ⑤⑥ SHADOWPC-VISION-CAL (in-loop L + at-speed gate-4 + 2-corner PnP + Bayesian-IoU extrinsic + per-gate last-fix + accel_body + yaw-active). **🚩 P4-C05 gate-yaw hardcode = GO-BEFORE-VQ2 (foundation for gate-relative rebuild). ✅ Regression suite DONE (dc4b532).** K+L+N PLANNED. REJECTED: HSV pre-filter. Detail [[project-phase2-rl-vision-decisions]] §ESTIMATOR-RACESPEED + §ORGANIZER-PIVOT + §RL-PORTFOLIO + §CASE-C-READINESS + §PLANNING-TOGT-S2 + §SUBSTRATE-AUDIT.

## Topic files
- [Master plan (LIVING, SSOT)](project_master_plan.md) — architecture, Track-A strategy, risks R1–R11, build sequence.
- [Phase-2 RL+vision decisions](project_phase2_rl_vision_decisions.md) — **READ when resuming Phase 2.** RL substrate, policy I/O, vision plan, staging, ALL stage verdicts/supersessions.
- [RL increment history](project_rl_increment_history.md) — checkpoint lineage inc1→inc7, stage history S1.1→S17, inc7 job IDs, md5s/commits, NaN/sidecar/OOB bug histories.
- [Sim ops](reference_sim_ops.md) — unattended FlightSim mechanics: launch/login, 31000 semantics, idle states, zombie instance, autoreset/spin guards, footguns.
- [Sim interface](reference_sim_interface.md) — confirmed MAVLink+video wire spec; all 5 must-verify items.
- [CTBR control + sysid](project_ctbr_control_sysid.md) — flyable stack: gains/signs, offline twin, Gate-0 saga, alt-relay.
- [Detector pipeline + PnP](project_detector_training_pipeline.md) — Adroit YOLO-pose training; v2/v3; weighted-PnP; adroit-connector ops.
- [AI Grand Prix context](project_ai_grand_prix.md) — competition overview.
- [Competition materials](reference_competition_materials.md) — spec facts, rules/FAQ, VQ1 mechanics, open unknowns.
- [Hardware](project_hardware_constraint.md) — laptop (dev) · ShadowPC (sim) · Adroit (GPU training).
- [Adroit cluster](reference_adroit_princeton.md) — Slurm/GPU; summer access confirmed.
- [Prior-art survey](reference_prior_art.md) — drone-racing projects + libraries + 8-paper ledger.
- [Tooling eval](project_tooling_recommendations_eval.md) — 3rd-party triage; keepers ADRC/GTSAM/min-snap/Isaac.
- [Estimator robustness](project_estimator_robustness.md) — adaptive R, innovation-gate, map-avg, SEARCH state, optical-flow.
- [Walking-skeleton directive](feedback_walking_skeleton_no_vq1_crutches.md) — VQ1 must be full VQ2 stack under-tuned.
- [Red-team passes #2+#3](project_red_team_pass_2.md) — historical triage; kept to avoid re-litigation.
- [Checkpoint transfer](feedback_checkpoint_transfer.md) — small RL checkpoints go in git; ShadowPC pulls.
