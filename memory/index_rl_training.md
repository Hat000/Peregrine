# RL Training Sub-Index
Mid-level index for RL increments, training doctrine, S2 architecture, plant/sysid, speed-ladder, and Adroit substrate. Deep detail in topic files below.

## inc7 — current best (LIVE-CONFIRMED 2026-06-13, commit 7fe90df)
- `stage1_inc7_actor.pth` seed-0; md5 `2AFF8D62569BA5FEC769028D72AF50E3`; sidecar `{act_max_thrust 3.765, act_max_rate 3.14}`.
- Standing 5/5 FINISHED; bridge 2/2; gate-3 NEVER crashed. sr=1.000 / t_med=9.76 s (warm) / gen=0.924 / deployment baseline **~11.45 s** (fresh-reset = deployment estimate).
- GEOMETRY-HONESTY THESIS LIVE-VALIDATED: contact-true volumetric training fixed gate-3 barrier on first try.
- Finding-A (gate-3 D-offset) FALSIFIED (Fengyou-verified): bottom-vs-opening-centre artifact; all 6 gates ≤0.37 m in-plane; track_map trustworthy.
- Bimodal lap times RE-LOCATED (P2-OFFLINE-ANALYSIS, definitive): ENTIRE 1.48 s gap = PRE-gate-0 cold-start artifact (~1.5 s physics stabilization); inter-gate intervals deterministic ±0.04 s. NOT post-gate-3. See §INC8-DESIGN CONVERGENT SYNTHESIS (VOIDED) in [[project-rl-increment-history]].
- 🚩 **Budget MORE seeds for inc8-class retrains** — 2/3 viable; narrow basin.

## Checkpoint lineage
inc4 (RETIRED) → inc5 (RETIRED) → inc6 (fallback, bridge live-confirmed) → **inc7 = CURRENT BEST**.
🚩 `fly_rl.py` default still inc4 — **pass inc7 explicitly** if not using `submit_rl.py`.

## RL plant / offline twin specs
- Offline twin: `src/racer/twin.py` / `twin_fit.py`
- `rate_gain [2.50,2.50,2.23]`, `rate_sign [+1,+1,+1]` (vanilla), `τ=0.019 s`, `hover=0.2656`, `linear_drag≈0.21/s`
- 🚩 **super-rate map** `g(|c|)=G0/(1−s·min(|c|,π)/π)`, `s≈0.30` ⇒ full-stick ~11 rad/s (NOT 7.85). Rate_gain is AMPLITUDE-DEPENDENT.
- **Quad drag** ✅ S16; **mixer** ✅ S17 (`u_i=clip(c+S·d, idle, 1)`; κ_err=0.073); **parity gate CLEARED** job 3270602.
- Plant: super-rate ✅ S14 + quad drag ✅ S16 + mixer ✅ S17 — COMPLETE.

## S17 + TRAINING-DOCTRINE
- 🚩 **S18 LAPSE VOIDED** — do NOT use `--plant lapse`/`dr_lapse`.
- Doctrine: honest contact geometry + structured DR; **NEVER reward damping**.
- 🚩 `fly_rl.py` defaults now SAFE (auto-reset OPT-IN). Use `rl/submit_rl.py` for judged runs; pass inc7 explicitly if bypassing.

## S2 architecture — DECIDED (2026-06-13)
- **`staged_monolithic_then_decomposed`** (HIGH confidence).
- Speed gap = TILT ENVELOPE, not architecture.
- 4.27/4.55 s bounds **FALSIFIED** (linear-plant fiction; real v_max ~39 m/s v² drag wall).
- Honest corrected-aero bound ~4.6–4.7 s; contact-valid = 4.43 s (gate-4 +0.046 m margin — razor-thin).
- Gap decomposition: 35.3 s (VQ1) → 9.76 s (inc7) → 6.89 s (cone tax) → ~4.72 s. [🆕 SUPERSEDED 2026-06-14: the ~4.72 s is the INVERTED/rate-infeasible TOGT bound; UPRIGHT-feasible ≈ 8 s, gate-4 ~30 m/s — see §DOCTRINE REVISION.]
- `rl/reference_line_vq1.json` drag-infeasible + 170° inverted → must be REBUILT. `rl/reference_line.py` loader valid.

## STYLE ENVELOPE COST + cone relaxation ladder
- 🚩 ≤65° roll = ~2.87 s/lap (ledger; inc5 A/B). 60° cone bound ~9.8–10.6 s ≈ inc7 live 9.76 s → gap = CONE, not policy/architecture.
- Relaxation ladder: (1) rw_tilt 96→48 (~1.4 s); (2) free-cone 60°→75–80° (~1.5 s — **THE binding kinematic lever**); (3) unconstrained.
- **COUPLING:** faster speed worsens gate-4 margin/σ ratio → estimator accuracy must be verified at each speed rung.
- **PRIMARY MARGIN GUARD = GATE-4.** Reporting discipline (body-radius reconciliation 2026-06-13): quote gate-4 margin as a FUNCTION of r ∈ {0.21, 0.26, 0.30, 0.33, 0.38}, ALWAYS p90 AND p99 — NEVER a single number. Central planning radius = **0.30 m** (budget 0.235 m); worst-case stress knob = **0.38 m** (budget 0.155 m). Gate-5 CLEAN (0.314 m @ r=0.30).

## NEXT queue — RL items
⑥ ✅ **P4-C05 DONE (f50b9b4, 692 green):** `GateMap` per-gate-yaw obs wired; yaw-aware path BIT-EXACT; default=VQ1 yaw=π BIT-EXACT; `get_gate_rotmat_w2g` is gate-relative estimator hook. Current stack VQ1-only (loud-aborts non-π). → [[index-vision-estimator]] §P4-C05
⑦ 🚩 **CR1-01 + P1-C06 → WINNER-VALIDATION RIDER:** fold yaw-active segment into inc8 fresh-reset live batch; closes absolute yaw wire sign + obs yaw seam in one capture.
⑧ **INC8 SPEC (gate-relative BLUEPRINT.md, 2026-06-13 — Adroit-ready; SHADOWVISION-updated):**
  - **BSR3 spin-gate MANDATORY FIRST** (spin_rate_abort→9–10, spin_time_abort→3.0 s; super-rate plant legitimately commands ~11 rad/s).
  - Obs: 20-dim (C1); train env populates pos_g as GT +L offset + MEASURED estimator residual (NEVER pristine truth, NEVER R_w2g@(gate_map−p_KF) anti-pattern). Critic 36-dim.
  - DR (estimator, on obs only): **per-fix lateral σ = U[0.05,0.15] m/axis (MEASURED 0.10 m; supersedes modeled 0.265 m)**; **one-signed PnP/extrinsic bias U[0,0.19] m with per-episode constant RANDOM SIGN** (SUPERSEDES ±0.10 zero-mean — cannot cover a one-sided 0.19 m bias). Map bias NOT injected (drops out).
  - Reward: R1' arc-length over REBUILT corrected-aero reference line (reference_line_vq1.json drag-infeasible+170° inverted → MUST rebuild); T4 finish-time KEPT; R4' **fixed relaxed cone = L0 default** (tilt_free_rad is scalar; confidence-gated cone is portfolio ablation); **🆕 R5' camera-pointing / gate-in-FoV reward = NEW TOP LEVER (2-AXIS: azimuth + ELEVATION)** — reward keeping active gate centre in-image; ELEVATION CO-BINDS (after decrab, gate horizontal FoV 100% but vertical FoV only ~10%; |el|~45° vs VFoV 29°); ×8.5 in-window fix-density gain (0.099→0.84) from centering through the 18–28 m window. 🚩 **MOUNT FIXED at 20° by spec §3.8 — mount-uptilt lever DEAD; elevation fix-rate is POLICY-only (inc8 pitch/pointing).** This is the primary reward change vs original spec.
  - **SELECT on p90 gate**: gate-4 SIMSTART in-plane p90 < 0.155 m @ r=0.38 (NOT RMS); design point COLD; ≥5 seeds; BSR3 FIRST. ESTIMATOR sets the ceiling, not the policy.
  - **HARD prereq for G4/G6:** contact_true_eval currently runs obs_from_truth (perfect pose, no noise) → selecting on it crowns a fiction. Must build: estimator-emulation obs wrapper + v* (achieved gate-4 approach speed) extraction instrument FIRST.
  - **INC8 BINDING GATE = GATE-4 (SIMSTART, p90 AND p99, COLD design point).** Report as function of r ∈ {0.21,0.26,0.30,0.33,0.38}; central = r=0.30 (budget 0.235); select at r=0.38 worst-case (budget 0.155). **INC8 LEVERS = TWO, both vertical-axis, now CO-EQUAL:** (1) **2-axis camera-pointing = fix-RATE** (policy R5' reward; surrogate-driven); (2) **BORESIGHT/attitude-bias estimation = fix-ACCURACY** (ε_vert ≈ 0.56°, δ_map discriminator d7c592e finding; CALIBRATABLE). Pointing gets fixes; boresight makes them true. ESKF attitude-bias estimation PROMOTED from secondary to co-equal lever.

## INC8 REWARD-DESIGN LEAN (2026-06-14 brainstorm — NOT frozen; dedicated RL session pending)
Design principle: reward the OUTCOME (estimator error vs GT); let camera pointing + fix timing EMERGE.
- **ANCHOR = GT estimator-error (privileged reward):** actor on noisy 20-dim obs; reward uses GT. Free in sim, ungameable by staring (cannot fake low error against truth). Reducing it REQUIRES fixes = pointing.
- **FLAT weighting (no gate-proximity schedule):** we do NOT know the optimal fix phase. Proximity-weighting prescribes a timing schedule we don't have evidence for. Flat = "information is generically good"; fix TIMING is an emergent training output.
- **Speed/finish DOMINANT:** kills the "slow down to farm fixes" exploit. Camera yaw/pointing is ~free translationally (decoupled from tilt/thrust axis) — inc7's 64° crab is an unconstrained slack DOF, NOT a speed-buying maneuver.
- **Dense shaping = frozen obs channels (c_inplane, c_along, age_norm):** used for fast credit-assignment; ANNEAL late so the final policy is honest to the true speed+finish objective.
- **Sparse-only is a BAD BET alone:** speed-ladder pressure manufactures incentive to learn vision, BUT the easier escape = fly conservatively / lean on cold IMU → converges to a slow, eyes-closed local optimum. Pure sparse throws away free GT signal.
- **✅ FIX-SURROGATE MERGED to main (ffb2c74, 2026-06-14; infra; code + calibrated checkpoints only, branch memory excluded; 15 green):** analytic NO-RENDER fix model `rl/fix_surrogate.py` for inc8 training. geometry→p_accept/fix_sigma/sample_fix; calibrated to Track-3 (accept @ peak 0.826, σ_lat 0.104); NOT overfit (LOFO CV). Reward NOT decided (Fengyou owns). σ single swappable checkpoint for recalibration from L3 at-speed recording. → [[index-vision-estimator]] §FIX-SURROGATE
- **🚩 2-AXIS CAMERA-POINTING (2026-06-14):** elevation co-binds — decrab alone insufficient (el-ok 0.10 after decrab; |el|~45° vs VFoV 29°); ×8.5 fix-density gain from 2-axis centering. 🚩 **MOUNT FIXED at 20° spec-EXACT (§3.8) — mount-uptilt lever DEAD; elevation fix-rate is POLICY-only via pitch/pointing.** R5' reward spec is 2-AXIS (azimuth + elevation via pitch). → [[index-vision-estimator]] §2-AXIS CAMERA-POINTING FINDING
- **🚩 BORESIGHT/ATTITUDE-BIAS = CO-EQUAL INC8 LEVER (2026-06-14; δ_map discriminator d7c592e):** ε_vert ≈ 0.215 m (~0.56°) is a genuine perception/attitude sighting bias that +L does NOT cancel. NOT map, NOT body radius. CALIBRATABLE (boresight cal / ESKF attitude-bias state). ESKF attitude-bias estimation PROMOTED from secondary to CO-EQUAL margin lever alongside 2-axis camera-pointing. Morning lock test: static head-on level fix at g2/g4 (bearing≈0). → [[index-vision-estimator]] §δ_MAP VERTICAL DISCRIMINATOR
- **🚩 TERMINAL GATE-LOCK = LOAD-BEARING INC8 CAMERA-POINTING ITEM (2026-06-14; margin-closure-envelope 7654e99):** MARGIN-CLOSURE-ENVELOPE (268-cell sweep, 4-lens adversarial) shows CLUSTERING is the dominant adversarial lens — terminal fix-DROUGHTS break closure at any mean rate. r=0.30 closure needs fix-rate ≥ ~0.25 WITH TERMINAL gate-lock (camera held ≥60% of final ~5 m approach), NOT just a high pooled mean. The crude global-burst models diverged ~8–12 m (upper bounds); the terminal-drought model is the defensible boundary. **Add TERMINAL gate-lock as an explicit framing in the 2-axis pointing reward spec** (not just a high average fix-rate). Fix-diff velocity channel HARMFUL at sparse fix-rate → keep OFF / P2 insurance. GPU latency ceiling ≤ ~50 ms (CPU-115 ms breaks v=55 cells). → [[index-vision-estimator]] §MARGIN-CLOSURE-ENVELOPE
- Dedicated RL brainstorm + hyperparameter session PENDING (step-6 inc8 input), GATED behind POC + L3 at-speed recording. Seed: this lean + S2 decision (staged_monolithic_then_decomposed) + speed-ladder doctrine. → [[project-phase2-rl-vision-decisions]] §INC8-DESIGN
- **RL SPEED-LADDER PORTFOLIO:** train deliberate envelope ladder; select fastest contact-valid policy the gate-relative estimator can DELIVER. Speed ceiling GATED on L3 at-speed recording — CANNOT be settled offline. Binding factor = systematic attitude/accel bias, not contact radius.
- 🚩 **LSQ VISION-VELOCITY REFUTED — NOT LOAD-BEARING (SUPERSEDES earlier banking).** Honest inter-frame PnP-delta σ_v ≈ 2.81 m/s (NOT assumed 0.3–1.0) + mandated RewindKF L≈115 ms → cold in-plane NO-GO at every smoothing window. Demoted to P2 insurance. Margin closure rests on attitude/accel-bias control (≤~0.6°) + uncertainty speed-down. Position-fix-differencing IS the KF (free baseline); do NOT build a load-bearing vision-velocity channel.

## DiffAero / Adroit substrate
- `src/racer/rl_plant.py` + `rl/diffaero_dynamics.py`, parity BIT-IDENTICAL; 2048-env PPO ~88.9 K steps/s.
- 🚩 **GPU PREFERENCE:** prefer A100 but do NOT idle — THROUGHPUT beats waiting.
- 🚩 **Retrain footguns:** evals default legacy flat plant — all evals must run **map-ON**; `/scratch/network/fl3689/peregrine_repo` is FILE COPY not git clone; parity gate CLEARED job 3270602.
- 🚩 **30 Hz = inherited default (DiffAero racing.yaml).** Binds ONLY at ≥30 m/s × last-fix ≤10 m; fixes to 15 m → NOT bottleneck.

- 🚩 **ADROIT AUP (binding; #62 resolved 2026-06-14):** SLURM only — NO compute on login nodes; NO internet on compute nodes (pre-download git/pip/conda/HF+YOLO weights on login/vis BEFORE submit); output→/scratch not /projects; accurate --mem + 1-core serial (over-alloc / multi-core-serial → SUSPENSION); zero-GPU-util killed at 2h (PPO must saturate GPU); `checkquota` routinely. → [[reference-adroit-princeton]] §Acceptable-Use Policy.

## INC8 PERCEPTION-REWARD CANDIDATE (2026-06-14; COWORK-3; handoff/cowork-2026-06-14/perception-reward.md; NOT frozen — Fengyou owns freeze)
Spine (SWIFT Nature'23 / Geles RSS'24): keep the DOMINANT potential-based gate-progress reward; add ONE small look-at term:
**r_perc = λp · w_term(d) · exp[ −((α/σα)⁴ + (β/σβ)⁴) ] · max(Δs, 0)** (gate centre in camera frame; α=atan2(X,Z), β=atan2(Y,√(X²+Z²))).
- **2-axis** (Qin α/β): σα≈45°, σβ≈29.5° (match 90°H/59°V; elevation tighter). **Terminal-locked:** w_term→full inside ~5 m (Azhari λ(d); floor w0≈0.1–0.2) = our TERMINAL gate-lock. **Progress-gated** ·max(Δs,0): slow-to-farm earns ≈0 (THEIR construction → validate in A4).
- Routing: azimuth→yaw (free → emergent crab reduction); elevation→pitch GENTLE (fights accel axis; 20° tilt partly offsets → keep σβ generous). λp≈5% of progress; **asymmetric PRIVILEGED critic** (critic sees true α/β/gate pose; actor discovers fix-timing). Anneal λp from 0; CAPS + yaw/pitch-RATE penalties for chatter.
- **Ablation:** A0 none → A1 1-axis → A2 2-axis → A3 +terminal → A4 +progress-gate; orthogonal critic on/off + λp∈{2,5,10}%. Metrics: lap-time(true), terminal-window fix-rate, worst-gate fix-rate, rate-RMS; watch a phase-transition cliff (Pan).
- Honesty: progress-gating not in any drone paper (A4 validates); terminal-vs-uniform unproven on localization (A3 generates evidence — we have fix-rate instrumentation nobody published); SWIFT λ-values not public. LATENCY (COWORK-1): VQ eval = desktop GPU → don't over-engineer for CPU latency.

## DOCTRINE REVISION — UPRIGHT-FEASIBLE LAP ≈ 8 s, NOT 4.7 s (2026-06-14; P2 Worker #2, RATIFIED; p2-inc8-refline merged 5a4afa6)
🚩 **The banked ~4.6–4.7 s / 4.72 s "honest corrected-aero bound" was the FULL-ATTITUDE/TOGT optimum — RATE-INFEASIBLE.** It needs sustained INVERTED descent (median tilt ~100°; tilt>90° over ~50% of the course) AND yaw-rate ~13 rad/s > the plant's ~11 rad/s ceiling. The honest **UPRIGHT, rate+collective-feasible lap ≈ 7.9–8.5 s** (emitted Γ = **8.455 s @ 75° cone**; collective 0.91≤1.0, rate 5.4≤11, v 28.9<39, 0% inverted).
- **Cause:** MEASURED quad-drag ~7× linear at 30 m/s → upright racing tops **~27–30 m/s** (level @30 m/s already needs ~78° tilt; faster upright → >90° = inverted). The banked **"39 m/s drag wall" was the INVERTED full-thrust case**, not upright. Two independent checks agree (T/W 3.765 + measured drag → ~27–30 m/s upright).
- **Supersedes:** the gap-decomposition "→ ~4.72 s honest bound" (inverted fiction) and "cone-tax = biggest lever" (the prize is ~9.76→~8 s ≈ 18%, smaller than banked). Honesty trajectory: 4.27/4.55 (linear) → 4.6–4.7 (inverted/TOGT) → **~8 s upright-feasible**.
- 🚩 **NET POSITIVE for margin closure:** binding gate-4 speed = **~30 m/s, NOT 37–55** → fix-rate/bias/latency/terminal-lock conditions ALL relax; the σ-gate's 37 m/s targets were over-pessimistic on the speed axis.
- 🚩 **REFRAME inc8's value:** given Q2 (no pose on wire → self-localize REQUIRED), inc8's headline value = **CASE-C DEPLOYABILITY** (camera-pointing + estimator-robust policy that flies on self-localization without crashing), NOT a big speed jump. ~8 s upright = realistic racing ceiling; the win is flying it at all on the real eval.
- R1' arc-length reward UNAFFECTED (uses Γ geometry only, identical across speed profiles). Γ on main: rl/reference_line_inc8.json + rl/build_reference_line.py (vq1 json untouched).

## INC8 ESTIMATOR-EMUL ESCAPE-HATCH = 🟢 GREEN ×2 + REWARD SPINE A RATIFIED (2026-06-14; P2 Worker #1; p2-inc8-rl merged 5a4afa6; suite 762 green)
- **GREEN ×2** (two independent builds, consistent non-identical numbers → real, not a single-impl artifact): the 20-dim obs contract carries the camera-pointing signal honestly. Frame-seam identity ≤1e-6 / 250 states (+L correct); pointing→fix-rate monotone (~8.6× in-window gain; term-lock 0.91→0.00); KF in-plane error −59% with fix density (monotone), NEES 0.971 (calibrated); obs[17:20] non-degenerate. Escape NOT fired → torch train-env port (Deliverable 2) UNBLOCKED.
- **DISCRIMINATES:** inc7 on emulated obs is non-robust (collides @gate-0; S_stable 0.571 vs 5/5 perfect-pose) = the why of inc8.
- 🚩 **REWARD SPINE = EXPLICIT (arm A), RATIFIED:** emergent pointing gradient is WEAK on a clean/slow course (trusted-IMU + truth-gate → no-fix coast ~0.30 m ≈ biased-fix floor @19 m/s) → implicit-only (arm C) UNDER-produces pointing → the EXPLICIT 2-axis terminal-lock term (A, COWORK-3 shape) is LOAD-BEARING. Commander-ratified training portfolio: **A ≥5 seeds / B 3 / C 1-seed falsifier.** Final reward FREEZE = Fengyou's, post-portfolio + post-σ-recal.
- 🚩 **SELECTION GATE — no crowning a winner until BOTH:** (1) **P3 L3 at-speed σ recalibration + at-speed coast pin** — gate-4 in-plane is **VERTICAL-σ-DOMINATED (σ_vert 0.28 ≫ σ_lat 0.10)**, cold@bias0 p90 0.392 / bias-ON 0.457, both over budget = CANNOT-SETTLE reproduced; (2) the revised ~30 m/s upright / ~8 s ceiling. The vertical-σ dominance reinforces P1's vertical-boresight fix as the binding axis.

## INC8 TORCH TRAIN-ENV MERGED + GPU-SPEND GATES (2026-06-14; P2 Worker #3 / D2 merged b0b322e, 828 green)
- **PeregrineRacingInc8 (ENV_ALIAS subclass) MERGED:** R1' arc-Γ (corrected-aero reference_line_inc8.json) + T4 finish-time + FLAT truth-seen GT-anchor + annealed confidence-shaping + R4 60° cone + **R5' = COWORK-3 2-axis terminal-lock reward (arms A/B/C toggles)**; critic 36; BSR3 spin-gate (realized-ω 10/3s); obs 17→20 sidecar. S1–S4 parity GREEN (torch==numpy ≤1e-4); **OFF==inc7 byte-identical (AST-proven)**; 42 tests; CPU smoke green (pointing→fix→confidence→R5' chain + terminal fix-drought in-loop). Adroit-loadable (numpy+torch core, no scipy).
- 🚩 **PARITY-CATCH (NOT a boresight finding):** the 'R_camera_from_body sin-sign flip' was a bug in the worker's OWN scipy-free torch REIMPL (rl/inc8_estimator_emul.py), CAUGHT by the S1 parity gate vs canonical src/racer/frames.py and fixed there. **main's frames.py is CORRECT — do NOT chase this as an ε_vert source.** A parity-gate SUCCESS.
- **GPU-SPEND GATES (AUTHORIZED/CONFIRMED by overall commander; smoke-GO + L0-triage = commander's call post-cutover):** GPU smoke (Worker #4, sonnet, 1 seed / arm A / 2048 envs + 512 precheck, ~45min/1GPU — GREEN = no-NaN + GPU-saturated ≥70% + pointing-rising + sidecar obs_dim=20/inc8=true/r5_arm=A) → **L0 reward-arm portfolio** (Worker #5, opus, A×5/B×3/C×1 = 9-run SLURM array, train to inc8 budget = inc7 converged length, eval BOTH closed-loop estimator-emulated AND passive-observer) → L1/L2 speed rungs on the WINNING arm ONLY (STAGED — avoids 3× spend before the arm question settles) → [P3 at-speed σ-recal] → CROWN.
- 🚩 **CROWNING HELD on P3's L3 at-speed σ recalibration** (gate-4 in-plane VERTICAL-σ-dominated, σ_vert 0.28 ≫ σ_lat 0.10; pointing-incentive untrustworthy until the at-speed no-fix coast is pinned). ALL evals on ESTIMATOR-EMULATED obs (GREEN-verified), map-ON, NEVER perfect pose. arm C = implicit-only FALSIFIER (predicted to under-produce pointing); arm A = ratified load-bearing spine; B = flat-FoV bridge. Adroit AUP MANDATORY on both workers (SLURM-only, pre-download on login/vis, /scratch output, accurate alloc, GPU-saturate, checkquota).
- 🚩 **GPU SMOKE #1 = INFRA-GREEN / GATE-RED (job 3273374, Tesla V100, 2026-06-14; report handoff/inc8-gpu-smoke-2026-06-14, branch claude/reverent-shirley-b94605 e08a19b):** torch port TRAINS END-TO-END on GPU — 300/300 updates, no NaN, sidecar obs_dim=20/inc8/arm A confirmed, checkpoint + .pt2 export succeed. RED is HARNESS, NOT training: (1) **RC=1 COSMETIC** = diffaero ONNX export of action_frame="body" in runner.close() crashes AFTER success (memory:434 loose end CONFIRMED LIVE; fix in peregrine wrapper, keep diffaero pristine flyingbitac@291ea14); (2) two sbatch post-proc bugs hid signal (RUN-glob misses `quad__…__<run>__0/` layout; inc8 metrics are TB-only but sbatch greps stdout — worker pulled signal from TB event file); (3) **GPU util 25.8%/max 29% CONFOUNDED** (short/export-dominated; suspect numpy-estimator CPU boundary — needs clean longer-run measurement before L0). **SUBSTANTIVE FLAG (NOT a verdict): pointing flat under arm A @300 updates** (pointing_rate 0→0.006, terminal_pointing flat 0.000, total_reward −0.33→−0.64 declining, entropy 0.36→1.64 rising, critic 17.8→2.3 = IS training). 🚩 **300 updates = 5% of budget = pre-competence → INCONCLUSIVE; do NOT bank "arm A fails".** **COMMANDER CALL = NO-GO on L0; fix harness (RC guard + 2 sbatch bugs + .gitattributes *.sbatch eol=lf) + re-smoke @ NUPD=1000 / --time 2.5h → re-adjudicate arm-A pointing + clean util BEFORE the 9-run ladder.** Laptop fix worker DISPATCHED (opus). **OPS: adroit.py serve daemon UP → re-run via `adroit.py x "…"` no new Duo while window stays open; scratch 55.9/93 GiB OK, home 9.3/10 tight (job→scratch only).**

## Topic file pointers
- [[project-rl-increment-history]] — checkpoint lineage inc1→inc7, stage history S1.1→S17, inc7 job IDs, md5s/commits, NaN/sidecar/OOB bug histories, §INC7-LIVE-CONFIRMED, §Phase-0(b).
- [[project-phase2-rl-vision-decisions]] — RL/S2/inc8/doctrine sections: §S2-DECISION, §PLANNING-TOGT-S2, §S17, §TRAINING-DOCTRINE, §CORNER-PASS, §INC8-DESIGN, §RL-PORTFOLIO, §FRAME-AUDIT.
- [[project-tooling-recommendations-eval]] — 3rd-party triage; keepers ADRC/GTSAM/min-snap/Isaac.
- [[feedback-checkpoint-transfer]] — small RL checkpoints go in git; ShadowPC pulls.
