---
name: rl-increment-history
description: "Full RL checkpoint lineage inc1→inc6 + stage history S1.1/S1.2/S1.3/S14/S15/S16/S17 — provenance (md5s, commits), supersession chain, bug histories (NaN, sidecar, OOB-box). Episodic record; CURRENT state lives in MEMORY.md."
metadata:
  type: project
---

# RL increment + stage history (lineage ledger)

**Supersession chain:** inc1 (overfit, no plant DR) → inc3 (wrong flat-2.5 plant) → inc4 (aero-blind, RETIRED) → inc5 (mixer-blind, RETIRED — live failure twin-reproduced) → **inc6 = current best** (mixer+aero+map plant, corner-tax c16, SHIPPED — live transfer pending). Per-stage verdict detail: [[project-phase2-rl-vision-decisions]] (§Staging, §S1.2, §CHARACTERIZE-SWEEP, §S14, §TWIN-FALSIFY, §S16, §S15, §INC5, §SHADOWPC-LIVE-DEPLOY-DIAG, §S17). Facts below moved verbatim from MEMORY.md index (2026-06-11 restructure).

## Inc-1 / S1.1
- **Inc-1 ✅ (0.97 success, `stage1_inc1_actor.pth`)** — 🚩 no plant DR (numpy backend silent), overfit-nominal.
- S1.1 (sonnet) deployment had 3 bugs; corrected by the S1.2 recipe (recipe itself = load-bearing, stays in MEMORY.md).

## S1.2
- **S1.2 ✅ live pipeline VERIFIED, checkpoint NOT transfer-ready** — "backflip-diver" failure (roll 104–126° pre-gate); tail-first spawn fixed by virtual π body-z flip. **🚩 "OOD-at-start" RETRACTED** (artifact of wrong transfer; reset saturation is normal). Detail §S1.2.

## S1.3 / inc3
- **S1.3 ✅ in-twin:** 3.765 ceiling + standing-start + ATT=2.0/JERK=0.5 → 100% 6/6, peak roll 65°; `rl/checkpoints/stage1_inc3_actor.pth` (md5 ea2bf8a2…, commit c05362f). 🚩 S1.3+inc-1 trained WRONG flat-2.5 plant → retrain gated; **S1.3-live DEMOTED to optional datum.**

## Bug histories (root-caused; do not re-litigate)
- **🚩 OOB-box spawn bug (fixed in S15):** standing-start spawn at x_zup=0 sat ~8 m OUTSIDE the gate-bbox+margin OOB box → every standing-start env truncated on step 1 (cause of an earlier 768768-episode / 0%-success eval — the BUG, not the policy). Detail §S15.
- **🚩 S1.3 NaN ROOT-CAUSED (corrects "PPO grad-NaN/seed" lore):** unclamped Euler extraction in `get_observations` — pytorch3d asin returns NaN when float32 entry hits 1+1e-7 (~once per 30M obs). NOT a PPO/gradient issue. Fix = clamp + counted `nan_to_num`; zero NaN events in all post-fix runs (holds through inc5 aero plant). Detail §S15.
- **Sidecar convention (since inc4):** `.json` sidecar beside `.pth` carries `{"act_max_thrust", "act_max_rate"}`; auto-loads via `load_actor` (watch `[load_actor] sidecar` line in `fly_rl.py`).

## CHARACTERIZE-SWEEP (510da24)
- **🚩 SUPERSEDES cc6921d 2nd-order/PI-windup model:** inner loop = static super-rate map + fast ~critically-damped loop; PI-windup DISPROVEN; crash-termination = collision-based. DR spec (s∈[0.25,0.35], τ∈[0.015,0.03], alpha_max∈[200,320]) ✅ SHIPPED S14.

## S14 (static super-rate map)
- **S14 ✅ (static-map, critical-path step 1):** map+slew in all three plants + torch DR, V100 worst 1.8e-15 PASS, 376→409 tests. Map defaults None = legacy bit-identical; `faithful_config(super_rate=True)` = map-ON.
- **🚩 `peregrine_eval.py`/`offline_rollout.py` DEFAULT to legacy flat plant — all evals must run map-ON.** (Also flagged in MEMORY.md.)
- **🚩 `/scratch/network/fl3689/peregrine_repo` is FILE COPY, not git clone** — stage deploy-key clone if sessions grow.

## S15 / inc4
- **✅ S15 ENV-COHERENCE REDESIGN + S1.4 RETRAIN COMPLETE (2026-06-11, 7 jobs/3 rounds Adroit): `rl/checkpoints/stage1_inc4_actor.pth` + `.json` sidecar (md5 766ea71b, commit 66dd3c2).** Held-out VQ1: 100.0% / 3582 eps / 7.03 s median. Generalization 0.845. 528 tests green. Detail §S15.
- **🚩 inc4 KNOWN-BROKEN on corrected aero (sr 0.461, 53.9% collisions) — SUPERSEDED by inc5.** Live verdict: **inc4 RETIRED** (3.1 m rocket, falsified-linear-map 2× thrust).

## TWIN-FALSIFY → S16 (aero)
- **🚩 TWIN-FALSIFY:** ① drag QUADRATIC body-dir-dependent (c2≈0.052/m; 2.2× at 9 m/s → policies ~2× under-braking); ② collective CONVEX, full-stick ≈8 g ("3.77 g" was linear artifact). Survivals: rate map airspeed-invariant, determinism SD ~0.03 m/s.
- **✅ S16 AERO (commit eba4349):** quad drag + convex knot table in all three plants, 497 tests green; CPU-torch 8.9e-16 PASS; twin↔rl_plant ≤5.1e-13. **🚩 V100 gate NOT run — `run_parity.sh` md5 tripwire updated; run FIRST at retrain.** Detail §TWIN-FALSIFY+§S16 + `handoff/laptop-s16-aero-integration-2026-06-11/REPORT.md`.
- **✅ COAST-REPLAY CONFIRMED through S16 plant: 0.224 m/s speed RMS (target ~0.24)** — aero integration reproduces campaign fits.

## Inc5
- **✅ INC5 SHIPPED (2026-06-11; commit 02fcce1; `rl/checkpoints/stage1_inc5_actor.pth` md5 bd1d670f…):** Aero-ON, sr 1.000 / 9.52 s median / gen 0.939 / roll p90 64.5° / in-twin deploy 6/6 +2-step latency. SUPERSEDES inc4. V100 parity worst 2.665e-15. Round 1 unconstrained datum 6.89 s (roll p90 145°). **`fly_rl.py` default still points at inc4 — pass inc5 ckpt explicitly.** Detail §INC5.
- **LIVE: 0 finishes** — blocked by mixer coupling (below); inc5 retired-from-live pending inc6.

## Live-deploy diagnosis → S17 / inc6
- **✅ LIVE-DEPLOY ROOT CAUSE (2026-06-11, commits ef2605d..8dbd9bf): SIM MOTOR MIXER COUPLES THRUST AND RATE AUTHORITY AT SATURATION CORNERS — unmodeled in all three plants.** `rl/replay_obs.py` proves pipeline BYTE-CORRECT (H1/H2/H3 hypotheses ALL FALSE; "all-rail outputs" = normal bang-bang). **Corner 1** (thr=0 × yaw=3.14): motors [0.08,0.73,0.73,0.08] → **9.4 m/s² uncommanded lift**. **Corner 2** (coll≈1.0 × rate): **rate authority vanishes** (gate-2 dy −1.0…−1.8 m). Fit data: `data/runs/20260611_194826_mixer_probe`. **--yaw-scale 0:** 14 clean passes / 10 flights, 0 finishes — ceiling hit. Detail §SHADOWPC-LIVE-DEPLOY-DIAG.

## ✅ S17 COMPLETE (2026-06-11, laptop fable; writeup handoff/laptop-s17-mixer-inc6-2026-06-11/WRITEUP.md)
- **Mixer model:** per-motor clip `u_i = clip(c+S·d, idle, 1)`; κ_err=0.073 (two probes agree 0.2%); S14 slew limits were measured WITH mixer throttling (authority scales by Q=r/r_fit). Top-rail corner unmeasured — `mixer_probe2.json` queued. Integration: all three plants, defaults OFF=bit-identical; 528→562 tests; CPU gate 7.1e-15; V100 GATE_PASS 7.1e-15.
- **Reward-design lesson (durable):** joint corner tax `w·|thr−mid|·‖rate‖` DECOUPLES smoothness from generalization; blunt ‖Δa‖² trades them monotonically (gen 0.727→0.571 as dact 1→16; combo arm collapses to 0.633 when dact=4 reappears).

## ✅ Inc6 SHIPPED (2026-06-11, laptop fable; live transfer PENDING)
- **Winner c16** (joint-penalty only, no dact): VQ1 sr 1.000 / 9.86 s median / gen 0.982 / thr_p95 0.061 / 0% saturation / max pass offset 0.277 m (zero-contact). 16/16 laptop deploy matrix (latency 0–3×every start mode×seams perturbed). No mitigation flags.
- **Checkpoint:** `stage1_inc6_actor.pth` + sidecar; `fly_rl.py` default still inc4 — explicit `--checkpoint` required. **R2 RESOLVED (job 3268876, inc6_c16_s1):** seed 1 reproduces transfer-critical properties exactly (VQ1 sr 1.000, 9.72 s, thr_p95 0.062/yaw_p95 0.010/0% flips) — rail-free style is a property of the corner-tax family, NOT seed luck. Shipped s0 STANDS. Gen is SEED-VOLATILE: 0.741 (s1) vs 0.982 (s0) — the headline was the better draw; even s1 beats every dact arm (0.571–0.727). **SELECTION PROTOCOL (durable):** future candidates (inc7 envelope-ladder, S2 work) must average generalization over ≥3 seeds — never trust single-seed gen numbers.
- **Twin discrimination:** inc5 on mixer-ON plant = 0.000/0.000, yaw_flip 81.7% (live 0/20 reproduced); inc6 passes everything. Inc5 formally RETIRED.
- **Supersession:** inc5 (mixer-blind, retired) → **inc6 (mixer+aero+map plant, corner-tax c16, SHIPPED)** → live transfer FAILED 2026-06-12; diag in progress.

## inc6 live attempt 1 (2026-06-12) — FAILED (root cause found)

**Result: 0/15 finishes.** Same checkpoint + code ran 16/16 on laptop deploy matrix.

**Standing start ×10 (0/10):** deterministic first action → large lateral sweep → SPIN_ABORT/CRASH/TIMEOUT at gates=0.

**Bridge start ×5 (0/5):** 5/5 passed gate 0 (inertia carries), 0/5 finished — all stalled at gate-1 (first banked move) with 50–100 m orbital arc + left-right oscillations → spin.

**Diag artifacts:** `debug_obs.jsonl` in all 15 run dirs on ShadowPC. Forensics: SHADOWPC-INC6-DIAG.

---

## ✅ SHADOWPC-INC6-DIAG — root cause found + fixed (2026-06-12; commits bcc93f9, 325e191; writeup handoff/shadowpc-inc6-diag-2026-06-12/WRITEUP.md)

### Root cause: roll-axis convention mirror in RL deploy layer
**H0–H3 ALL CLEAN:** correct ckpt/md5/sidecar; no machine-local map in RL path; 29.2 Hz loop; sim/wall 1.000; zero stale ticks; step-0 obs matches twin to 7e-4. Not obs encoding, not timing.

**Discovery chain:**
1. Open-loop twin under recorded live actions diverges in z/x within 16–17 ticks — rates roughly match. Physics, not policy.
2. Attitude contradiction: "artifact-undone" roll ≈0 for 1+ s while raw rate channel claims +1.0–1.5 rad/s sustained body roll. Self-contradiction only visible at −55° pitch.
3. **Quat-FD test (diag_h4d.py):** raw ODOMETRY quat finite-differenced body rates match `w_raw` under **[+1,−1,+1]** (corr 0.93–0.98, level AND tilted, 3 flights). The roll-undone model collapses in tilted flight (corr 0.0–0.45). The raw quat correctly rotates v_body onto position derivative — it IS the true attitude.
4. Feedback-free command sign probe (c100_r31): wire roll +3.14 → raw-quat roll −1.34 rad in 0.18 s ≈ −10 rad/s. **S_live = [−1,+1,−1]** (roll AND yaw inverted).
5. `mavlink_client.py` documents the sign-inversion lore belongs to ATTITUDE euler, not ODOMETRY quat.
6. The 2026-06-10 re-verification measured corr(raw, reported-attitude-FD) = [+,−,+] — same data — but inherited "reported roll is inverted" as prior → concluded [−1,−1,+1]. Level-attitude data cannot discriminate.

**Three coupled errors (self-consistent at level attitude, physically impossible when tilted):**
1. `build_obs` applied roll-inversion "artifact undo" (`R(-roll,pitch,yaw)`) — artifact does not exist on the ODOMETRY quat.
2. `_ODO_RATE_SIGN = [−1,−1,+1]` — should be `[+1,−1,+1]`.
3. `_ACT_FLU_TO_FRD = [1,−1,−1]` — did not account for live roll inversion; should be `[−1,−1,−1]`.

**Closed-loop counterfactual (`diag_counterfactual.py`):** buggy mapping → 0 gates, lateral sweep, OOB 2.4–2.6 s (reproduces live signature). Fixed mapping → **6/6, 9.50 s** (lat 0 and 2).

**The laptop matrix could not catch it:** `telemetry_from_truth` applied the same assumed artifact model that `build_obs` undoes — any misidentification round-trips to zero. Only live MAVLink exercises the real convention.

### Fix (bcc93f9)
- `rl/fly_rl.py`: `build_obs` attitude from raw quat; `_ODO_RATE_SIGN=[1,−1,1]`; `_ACT_FLU_TO_FRD=[−1,−1,−1]`.
- `rl/offline_rollout.py`: `telemetry_from_truth` synthesizes TRUE artifacts; all plants get measured live `rate_sign=[−1,+1,−1]`.
- `rl/replay_obs.py`, `rl/rate_sysid.py`: artifact undo corrected.
- Simstart + handoff mixer rollouts still finish 6/6; `check_build_obs` 0.00e+00.

### Post-fix live (2 flights — `inc6_rollfix_f1/f2`)
**No spin. No oscillation. Smooth, coordinated, correct-signed flight.** But 0/2 standing start: ~5 m +y miss at gate-0 plane → OOD wander. Failure mode completely different from pre-fix.

**Residual gap = translational (thrust-map overprediction):** counterfactual gains forward speed ~50% faster than live (vel_gx 8.8 vs 5.7 m/s by tick 12). Collective map fit at near-zero airspeed; live thrust at 3–12 m/s runs **15–25% below the model** (n≈1700 smooth-tick audit, ratio 0.74–0.88). Enough to displace the approach line by ~5 m in the first 2 s from a tilted standing start.

**Rate channel verified end-to-end:** transport d=2 ticks, τ=0.019 s, per-axis gain live ≈0.94–0.97 × model. Convention fix confirmed live.

### Inc5 gate-2 co-attribution
Inc5's uniform gate-2 lateral miss (dy −1.0…−1.8 m, mid-bank) is now co-attributed to the roll mirror (first real banked y-move on the course + roll channel mirrored). The mixer rails and the roll mirror both contributed to inc5-era 0/20. Their relative weight is untestable now that inc6+fix supersedes the stack.

### Pipeline byte-correct ≠ convention-correct (durable lesson)
"The obs/action pipeline was proven byte-correct" (inc5 diag) must be read as *deterministically reproducible*, NOT *convention-correct*. `replay_obs` verifies the code against itself. Only tilted-phase quat-FD (`diag_h4d.py` method) validates conventions against the sim. **Adopt as the permanent convention gate.**

### Mixer probe2 / rate_sysid status
- `rate_sysid.py` Euler-singularity aborts fixed via per-phase `no_tilt_abort` (325e191).
- Keeper run: `20260612_034154_mixer_probe2` (`z00_y31_long` clean).
- **Probe rows `c100_y31` / `c60_r31` / `zhov_r31` remain uncaptured** (~10 min unattended probe in next session).

### Recommended path
1. **Bridge ×5 with fix** (next ShadowPC session): strong prediction: threads gates (bridge bypasses low-speed lapse regime; cheapest discriminator).
2. **Laptop S18:** joint translational refit (collective-vs-airspeed lapse + drag) from 17 2026-06-12 recordings (3–30 m/s, no new flights) → re-run inc6 deploy matrix on refit plant.
3. If inc6 robust to refit → fly as-is; else inc7 retrain with lapse-DR (+ optional sin/cos yaw-wrap encoding for obs[8]).
4. Standing start = deployment target, gated on (2).

**Recordings:** `data/runs/20260612_044219_inc6_rollfix_f1`, `…_044438_inc6_rollfix_f2`. Forensic scripts: `handoff/shadowpc-inc6-diag-2026-06-12/scripts/`.

**Inc6 checkpoint STANDS. No retrain implied by the convention fix itself.**

## inc6 live attempt 2 (bridge retest, 2026-06-12) — prediction FALSIFIED

**Session:** SHADOWPC-BRIDGE-RETEST. **Commit:** e84a18d (writeup). Dataset commit: c846054.

**Result: 5/5 gate-0, 0/5 gate-1, 0/5 finish.** Same gate count as pre-fix bridge attempts.

**Key facts:**
- Roll convention fix (bcc93f9) confirmed live: post-fix bridge sweeps to +y; pre-fix swept to −y (sign inversion matches prediction). No spin guard trips at gate-0.
- DIAG "strong prediction: bridge threads gates" FALSIFIED. Bridge hands off at ~5 m/s — still within the 3–12 m/s thrust-lapse regime (15–25% below model). Gate-1 approach vector is displaced OOD regardless of start mode.
- 180° turn-back bug NOT observed in 5/5 post-fix flights — likely pre-fix artifact or course-geometry dependent.
- **Implication: thrust-lapse refit is load-bearing for BOTH standing and bridge start; no start-mode shortcut.**

**Mixer probe2 COMPLETE (2026-06-12):** all 4 remaining rows captured (c100_r31, c100_y31, c60_r31, zhov_r31) in `data/runs/20260612_133307_mixer_probe2`; `handoff/shadowpc-bridge-retest-2026-06-12/probe_summary.md`. Key findings: top-rail roll clips ~35% authority (motors 0,2 saturate at c=1.0); bottom-rail roll clips ~80% at hover thrust (dominant at standing-start low speeds); mid-band c=0.6 linear. Consistency vs S17 mixer model = optional S18 task.

**Refit dataset:** `handoff/shadowpc-refit-dataset-2026-06-12/debug_obs_17runs.zip` (9 MB; 10 standing pre-fix + 5 bridge pre-fix + 2 rollfix post-fix = 17 runs).

**Updated recommended path:**
1. Laptop S18: joint translational refit (collective-vs-airspeed lapse + drag) from 17 runs — no new flights needed.
2. Re-run inc6 deploy matrix on refit plant; if robust → fly as-is; else inc7 with lapse-DR.
3. Next live session = standing start ×5 (gated on S18). Bridge no longer a useful discriminator until plant gap closed.

---

## ✅ S18 LAPSE FIT+INTEGRATION + PREMISE REVERSAL (2026-06-12, laptop opus; commits fb99636+0fd741b; writeup handoff/laptop-s18-thrust-lapse-2026-06-12/WRITEUP.md)

### 1. Lapse fit — real and well-identified

Fit method: smooth ticks (|ω|<1 rad/s) across 17 runs; pristine vel_ned FD for measured specific force; raw quat as-is (bcc93f9); collective shifted by d=2 transport delay. K_eff ratio vs |v| band:

| |v| (m/s) | 3–6 | 6–9 | 9–12 | 12–15 | 15–18 |
|---|---|---|---|---|---|---|
| ratio | 0.74 | 0.82 | 0.88 | 1.00 | 0.99 |

Locked model (np.interp end-clamped; L(0)=1 hover-anchored):
```
LAPSE_SPEED_MEASURED  = [0.0, 4.0, 8.0, 12.0, 15.0]
LAPSE_FACTOR_MEASURED = [1.0, 0.78, 0.80, 0.92, 1.0]
```
Drag-independent where it matters (spread ≤0.09 across drag 0.5–1.5× at 3–9 m/s). Fast-descent thrust loss (L→−0.9 in vortex-ring regime) excluded — folded into inc7 DR band, not modeled. 3-fold CV reduces out-of-sample b3-accel bias in every low-speed band.

### 2. Integration — all three plants, defaults OFF (`fb99636`)

Multiplicative `a_up *= interp(|vel|, lapse_speed, lapse_factor)` in `rl_plant.py`, `twin.py`, `twin_fit.py` (faithful_config(lapse=True)), `rl/diffaero_dynamics.py` (+dynamics.dr_lapse scales lapse DEPTH 1−L per-env for inc7). Eval: `--plant lapse`. **562→589 tests green** (+20 twin↔rl_plant parity + 7 lapse unit/torch/DR tests). All defaults OFF = bit-identical legacy.

**🚩 V100 config-matrix gate MUST run at next Adroit contact** — CPU DiffAero gate un-runnable on laptop (pre-existing; DiffAero base not importable); lapse/lapse_full configs now in the matrix.

### 3. Verdict: lapse is NOT the live cause — PREMISE REVERSED

**No lapse curve reproduces the +5 m East miss.** Repro sweep across depth/persistence/latency {0,2,3}: shallow lapses finish cleanly centred; deep persistent lapses cause vertical crash (still laterally centred). No thrust-axis curve bends the trajectory into a lateral excursion.

**The live failure is a yaw spin the policy COMMANDS.** Live divergence develops at tick 42–66 at 16–18 m/s — where lapse ≈ 1. Policy commands a hard pitch-up flare + yaw turn; sideslip −72°; heading spins 135°→175°→−115°. Yaw rate prediction vs realized: +2.12 vs +2.08 (tick 60). Rate loop faithfully tracks commands; there is no missing torque.

### 4. Root cause: THIRD convention mirror — thrust→world lateral projection

Open-loop replay seeded at live pre-divergence state (tick 36), driven with exact recorded live wire commands for 27 ticks:

| | live | twin |
|---|---|---|
| roll/pitch/yaw (tick 60) | 57°/7°/−115° | 56°/7°/−113° |
| yaw rate | +2.08 | +2.14 |
| speed | 14.8 | 14.9 m/s |
| **East velocity** | **+11.1** | **−11.1 m/s** |

Attitude, rates, speed reproduced exactly — East velocity OPPOSITE-SIGNED.

Force frame selfcheck using live-recorded true attitude:

| tick | roll | measured a_E | model a_E |
|---|---|---|---|
| 51 | 59° | +30.3 | −35.1 |

Model East acceleration = NEGATIVE of measured; flipping thrust East sign makes model match measurement (−35→+31 ≈ measured +30). Discrepancy present at tick 39–42 in normal banked flight (roll 44–46°). Interpretation: twin thrust→world lateral projection is roll-handedness-mirrored relative to the sim given the same attitude quaternion. Most likely root: raw ODOMETRY quat is roll-mirrored vs true physical attitude (the diag's quat-FD validated level+pitched only; hard-roll gate-0 flare is the first maneuver that exercises lateral handedness).

### 5. 🚩 EVIDENCE VOIDED — FALSE PASSES

**The offline twin SELF-MIRRORS** (closed loop mirrors self-consistently → inc6 finishes 6/6 on every plant offline). Therefore:
- **The laptop 16/16 deploy matrix is a FALSE PASS** — it cannot expose the live failure.
- **The SHADOWPC-INC6-DIAG "counterfactual 6/6 @ 9.50 s" is a FALSE PASS** — same harness.
- **"Inc6 checkpoint STANDS" verdict from INC6-DIAG is UNKNOWN** pending LAPTOP-FRAME-AUDIT.

**DO NOT fly inc6 as-is. DO NOT train inc7 yet** (retraining against self-mirrored twin bakes the mirror deeper).

### 6. mixer_probe2 consistency check — CONTRADICTED, flagged

S17 model (`u_i = clip(c ± d, idle, 1)`) vs new settled-spin rows:
- `c60_r31` (clean, no clip): model d=0.523, measured d=0.132 → **roll κ_hold over-predicted ~4×** (implied κ_hold ≈ 0.012, not the yaw-derived 0.046)
- `c100_y31`: model d=0.122, measured d=0.349 → **yaw top-rail under-predicted ~3×** (zeta over-suppresses at c=1.0)

NOT integrated (structural: per-axis κ_hold + top-rail yaw effectiveness). Policy avoids the regime (thr_p95 0.061, 0% saturation, 0% yaw flips) — impact likely small. Flagged for dedicated follow-up (S19 candidate).

### 7. Next step

**LAPTOP-FRAME-AUDIT (fable):** systemic per-layer handedness audit — the third convention mirror. Diagnostic tools shipped: `handoff/laptop-s18-thrust-lapse-2026-06-12/scripts/s18_force_frame_selfcheck.py` + `s18_openloop_replay.py`. Extend quat-FD method to hard-ROLL phases. Candidate roots: (a) raw ODOMETRY quat roll-mirrored vs true physical attitude; (b) sign in deploy thrust path. Probe: deliberate sustained-roll-bank at moderate speed → measured lateral accel vs attitude-derived prediction.

---

## ✅ LAPTOP-FRAME-AUDIT — one-defect root cause; lapse voided; inc6 salvaged (2026-06-12, laptop fable; commits 93023cf fix, 1bfb936 writeup)

### Root cause: single R_y(π) conjugation in sim telemetry

**ONE defect explains all three mirrors:** the sim's ODOMETRY quaternion is the true attitude expressed in an R_y(π)-conjugated frame pair. `q_true = q_raw·[1,−1,1,−1]` (wxyz; negate x,z). Euler: roll AND yaw negated, pitch intact. Because R_y(π) conjugation is a PROPER rotation, it passes every internal consistency check (quat-FD vs rate, twist round-trip, level flight) — only comparison against an external invariant discriminates.

**Decisive measurement** (`audit_candidates.py`, 17 runs, 28,195 banked ticks): force model from conjugated attitude matches measured world accel on ALL axes (corr +0.97…+0.99, med residual ~1 m/s²); as-is reading anti-correlates East at bank (−0.84, med error 24 m/s²). True body rate = `−w_raw` (all axes; quat-FD gain 0.999). Live cmd→rate sign = **[+1,+1,+1]** — vanilla CTBR, no inversion on any axis. Every historical "roll/yaw inversion" was this single telemetry alias.

### Evidence voided and restored

All prior false-pass evidence (S18's "16/16 FALSE PASS" / "counterfactual 6/6 FALSE PASS") was caused by the same conjugation being applied symmetrically in the emulation harness and deploy code — they mirrored each other. The S18 open-loop replay METHOD was sound; it found the mirror. With fixed tools: **deploy matrix REBUILT 16/16** (mixer plant, latency 0–3×4 start modes, 8.2–9.6 s); counterfactual fixed 6/6 @ 9.50 s; bcc93f9 mapping against corrected emulation diverges before gate-0.

Why bcc93f9's tilted-phase validation failed: it validated the quat against its own rate channel (conjugation-invariant). It never tested force direction against measured world acceleration.

### S18 lapse voided

Re-fit with the true attitude on the same 17 runs: K_eff/K ratio ≈ 1.00–1.10 all speed bands. The "15–25% deficit" was mirrored-b3 projection error (≈20° East bank at early climb gives exactly 0.74; "recovered" as trajectory straightened). No residual translational plant gap: regime-binned residuals ≤~2 m/s² everywhere on the mixer plant, lapse OFF. **Do NOT use `--plant lapse` or `dr_lapse` for any future training. inc7 lapse-DR branch is dead.**

### Fix inventory (93023cf)

- `src/racer/frames.py`: `ODO_QUAT_TRUE_CONJ_WXYZ` + `true_attitude_from_odo_quat_wxyz` + `true_rate_from_odo_angular_rate` + full convention note (single source of truth).
- `rl/fly_rl.py`: conjugation in `build_obs`; `_ODO_RATE_SIGN=[−1,−1,−1]`; `_ACT_FLU_TO_FRD=[+1,−1,+1]`.
- `rl/offline_rollout.py`: emits conjugated quat + negated rates; live plant modeled as `rate_sign=[+1,+1,+1]`.
- `rl/replay_obs.py`: true-state extraction conjugated; `v_artifact` repurposed as deliberate wrong-frame canary.
- `rl/rl_plant.py`: LAPSE constants annotated VOIDED.

### Regression armor (598 tests; +9 vs S18's 589)

`tests/test_frame_conventions.py` — 9 golden-file handedness tests pinning per-axis handedness against recorded live data, including open-loop East-sign replay + mirror canary (must keep FAILING for as-is reading). `scripts/frame_residual_report.py` — standing per-session canary: regime-binned residuals + mirror canary + rate canary + optional open-loop replay. **Re-run after EVERY live session.**

### Inc6 verdict: SALVAGED

Training world (rl_plant/DiffAero, proper-rotation bridges both sides) was internally self-consistent throughout — no mirror was ever inside training. DO-NOT-FLY lifted. NEXT = ShadowPC live confirm: standing ×2 + bridge ×2 (spec: writeup §6). Prediction: standing start clears gate 0 centred (offline E at plane ≈ −0.2 m).

### Per-layer handedness map (final, all layers resolved)

| Layer | Verdict | Evidence |
|---|---|---|
| LPN pos/vel (world NED) | ✅ TRUE anchor | pos-FD ≡ vel through 60° bank |
| ODOMETRY quat | 🚩 R_y(π)-conjugated | force corr +0.99 conj vs −0.84 as-is at bank (28k ticks) |
| ODOMETRY angular_rate | ω_true = −w_raw | quat-FD(conj) = −w_raw, gain 0.999 |
| ODOMETRY twist + accel_body | pair with raw quat | consistent pair; KF safe |
| Sim physics cmd→rate | [+1,+1,+1] vanilla | open-loop replay with conj obs |
| CTBR stack | ✅ closed alias | VQ1-proven; per-axis closure derived §3 of writeup |
| fly_rl (93023cf) | ✅ fixed | golden tests pinned |
| rl_plant/DiffAero | ✅ self-consistent | parity suite |
| Vision chain (navigator.py:295) | 🚩 LATENT 4th seam (bank regime) | queued VISION-FRAME-FIX |

### Queued follow-ups

1. **VISION-FRAME-FIX** (before VQ2 banked vision): `frames.true_attitude_from_odo_quat_wxyz` for all world-geometry projections in navigator; VQ1-replay regression required. May explain part of VISION-PKG2's 1.4° attitude-noise fit.
2. **S19** (mixer_probe2 vs S17 contradiction — sign-invariant, unchanged by audit).
3. V100 config-matrix gate at next Adroit contact (lapse configs present but VOIDED — do not select).
4. inc7 planning: lapse-DR branch dead; if inc6 confirms live, S2 architecture resumes per master plan.

---

## ✅ inc6 LIVE CONFIRMED (2026-06-12, ShadowPC sonnet; writeup handoff/shadowpc-live-confirm-2026-06-12/WRITEUP.md)

**FIRST RL LIVE FINISHES EVER.** Supersedes "NEXT = ShadowPC live confirm" from LAPTOP-FRAME-AUDIT.

### Results

| Mode | Flights | Finish | Lap times |
|------|---------|--------|-----------|
| Bridge ×2 | 2 | **2/2** | **16.24 s, 17.74 s** |
| Standing ×4 (valid) | 4 | 0/4 | — |
| Spawn artefacts | 2 | — | 0-tick crash (not policy) |

**Convention tests:** `pytest tests/test_frame_conventions.py` → 9/9 passed pre-flight.

### Fact 1 — Frame fix verified live (9/9 tests; mirror canary TRUE +0.97 / AS-IS ≤−0.68)
Mirror canary confirmed on all sessions: TRUE East corr +0.97, AS-IS ≤−0.68 at bank. Both prior failure modes (pre-fix spin; post-fix +5 m East miss) are gone. Gate 0 centered in every valid standing-start flight (std_f1 y=−0.5 m at gate-0 plane, well within ±1.5 m). Open-loop vel error ≤1.24 m/s per axis. The falsify→integrate→audit chain is validated end-to-end.

### Fact 2 — Bridge 2/2 FINISHED
brg_f1 gate trace: gate 0 @ x=−20.3 (bridge handoff) → gate 2 → gate 3 @ (−108.5, −4.6, +21.7) → gate 4 → **RACE_STATUS finished 16.24 s.** brg_f2: same course, 17.74 s. Bridge = CTBR launch → RL handoff → full course. If zero-contact confirmed (check WRITEUP), bridge laps are a legitimate submission fallback at ~2× VQ1 speed (35.3 s → 16–18 s).

### Fact 3 — Standing start: new barrier = gate-3 (trajectory, NOT frames)
All 4 valid standing-start flights crash at gate 3 at identical pos ≈ (−110.6, −5.0, +22.5) ± 0.1 m, thr≈0.46. Gate-3 center: (−111.5, +5.1, −23.2). Drone clips south side. Bridge clears gate 3 at (−108.5, −4.6, +21.7) — ~0.4 m north and ~0.8 m lower, clearing the aperture. Gap = trajectory alignment, not convention. Likely resolves via offline gap analysis (fact 5).

### Fact 4 — Rate canary regime dependence
Bridge/high-speed flights: roll/yaw gains ~0.85 vs expected ~1.0. Standing-start at similar tilt: ~0.94–1.00. Sign correct (positive); convention correct. Appears regime-dependent (high speed / high collective). Not a frame defect; flagged for monitoring. LIVE-GAP-ANALYSIS will quantify.

### Fact 5 (commander addition) — ~70% lap slowdown; single-cause hypothesis
Live bridge laps 16–17 s vs twin counterfactual 9.5 s = ~70% slowdown. Hypothesis: under-tracking of the rate loop at speed (fact 4 gain ~0.85) → slower/wider/lower trajectory. One cause may explain BOTH the slowdown AND the gate-3 standing crash (approach line displaced from training). **LIVE-GAP-ANALYSIS queued** (offline, new recordings: twin-vs-live divergence, lap-gap decomposition, rate-gain regime dependence). This is the critical-path next step before any further live work.

### Fact 6 (commander addition) — Bridge as fallback submission posture
Bridge start = our own stack end-to-end (CTBR launch → RL handoff at x≈−20 m) → legitimate competition run if zero-contact. Inc6 bridge laps ~2× faster than VQ1's 35.3 s. Verify contact status in WRITEUP. Standing start is still the deployment target; bridge is the fallback while gap is open.

### Spawn artefact — confirmed pattern (sim-ops gotcha)
Gate-3 HARD COLLISION full-reset → next respawn hits residual collision geometry → 0-tick crash (header only, 0 obs). Pattern: gate-3 crash → artefact → clean spawn → gate-3 crash → repeat. Accounts for ALL gates=0 observations in session. Not a policy failure. Added to sim-ops gotchas.

### Next queue (per commander triage)
① LIVE-GAP-ANALYSIS (offline, new recordings) ② VISION-FRAME-FIX ③ gate-3 standing trajectory (likely resolves with ①) ④ S19 mixer contradiction ⑤ envelope ladder (gated on gap analysis).

---

## ✅ CRAB-DIAG (2026-06-12, ShadowPC opus; commit 301a2cf; writeup handoff/shadowpc-crab-diag-2026-06-12/WRITEUP.md)

**Branch 2b outcome: twin reproduces the crab → no live-only deploy fix needed. No flights flown; no jobs submitted.**

### Fact 1 — Crab = trained posture, not a bug (supersedes live-confirm §Fact 3 "trajectory alignment gap" framing AND commander's "obs-seam rotation error" hypothesis)

The "crab" (live cruise: ~+40° roll, ~+130° heading, ~+54° sideslip) is inc6's genuine trained flight style. Discriminator: extract TRUE attitude from the offline twin, not just lap time. Twin racestart holds **roll +41.5°, yaw +130.9°, tilt +54.8°, sideslip +51°** and **FINISHES 9.62 s**. Training-native `trainreset --no-virtual-flip` flies **+55.5° tilt, FINISHES 9.12 s** — the same style in the native training frame. inc6's `rw_tilt=96` produced a ~55° tilt racing cruise, not a low-tilt one; speed/progress optimum sits there.

Control experiment: `racestart --no-virtual-flip` (wrong frame) tilts to 69° and goes OOB at gate-0 — proving the virtual flip is correctly applied and re-expresses the same ~55° training style in NED at +130° heading. Step-0 obs: live vs twin <0.003; `check_build_obs = 0.00e+00`. Deploy chain verified byte-faithful. **DO NOT touch the virtual flip or conjugation. NO reward change.**

| source | roll | yaw | tilt | outcome |
|---|---|---|---|---|
| Live std_f1 | +42.6° | +129° | ~53° | gate-3 clip |
| Twin racestart | +41.5° | +130.9° | +54.8° | FINISH 9.62 s |
| Live brg_f1 | +38.7° | +130.9° | ~52° | FINISH 8.96 s RL |
| Twin handoff | +39.5° | +134.7° | ~54° | FINISH 8.16 s |

### Fact 2 — "~70% slowdown" DISSOLVED (clock artifact; supersedes live-confirm §Fact 5)

Live bridge RACE_STATUS clock 16.24 s includes ~8 s CTBR launch before RL recording starts. Actual RL segment (gate-0→finish) = **8.96 s at 17.2 m/s median**, matching twin handoff 8.16 s **per-segment within ≤0.15 s** (longest leg gate-2→3 = ~2.1–2.2 s in both). No transfer slowdown exists.

### Fact 3 — "0.85 rate-gain droop" DISSOLVED (wrong canary; supersedes live-confirm §Fact 4)

The 0.85 was the `quat-FD(true) ~ −w_raw` telemetry-consistency canary, not command tracking. Realized/commanded rate gain = **~2.5 (super-rate map), identical live vs twin** across all three bridge/standing sessions. Rate tracking is faithful.

### Fact 4 — ONE genuine residual: collective/drag plant gap on the banked climb

Tick-aligned twin `simstart` vs live std_f1 (identical start state, shared obs/action code → divergence = pure plant gap): E tracks to **0.04 m** throughout (frame-clean); divergence is N+D only, accumulating on the gate-2→3 climb (+11 m altitude). At gate-3 plane: live is **0.5 m high and ~0.5 m short** → clips upper frame; twin threads it (vert-off 0.00). `frame_residual_report` confirms: **+2.0–2.8 m/s² systematic N+D residual in the 12–18 m/s × tilt-35–90 bin**, mirror canary TRUE +0.97. Integrated over ~2 s climb → ~1–2 m N+D divergence. The bridge clears gate 3 because its flatter/faster approach has more vertical margin.

### Fact 5 — Bridge finishes zero-contact; rules-valid fallback

`n_coll = 0` across all 269 (brg_f1) and 267 (brg_f2) ticks → both finishes are **zero-contact**. Bridge-mode inc6 = a legitimate ~9 s RL-segment submission posture (vs VQ1's 35.3 s total).

### Fact 6 — S20 spec (inc7 retrain)

1. **Plant refit (banked-climb regime):** re-identify `COLL_MAP_ACCEL`/`QUAD_DRAG_C2` against pristine-vel_ned FD in the 12–18 m/s × tilt-35–90 × high-collective bin using 8 post-fix zero-contact recordings + 2 bridge finishes. Target: N/D median residual from +2–2.8 → <1 m/s².
2. **DR for gate-3 margin:** ±12% DR on high-collective collective_map_accel (and drag) → policy carries ~0.5–0.7 m vertical margin through gate 3. Do NOT use `--plant lapse`/`dr_lapse` (VOIDED).
3. **Selection:** ≥3-seed generalization averaging. Single-seed gen is volatile (0.741–0.982 for inc6 seeds).
4. **V100 config-matrix gate** bundled at next Adroit contact.

**Predicted outcome (on record):** standing start clears gate 3 with margin; posture unchanged (~55° tilt trained style); bridge parity.

### Supersession notes
- **Supersedes** live-confirm §Fact 3 "trajectory alignment gap (~0.4 m south of center)" framing → reframed as collective/drag plant residual in the banked-climb bin.
- **Supersedes** live-confirm §Fact 4 "0.85 rate-gain droop" → dissolved (wrong canary, not command tracking).
- **Supersedes** live-confirm §Fact 5 "~70% slowdown" → dissolved (clock artifact, RL segment = 8.96 s ≈ twin 8.16 s).
- **Supersedes** live-confirm NEXT-queue "LIVE-GAP-ANALYSIS" as the critical-path item → replaced by S20 refit+inc7.
- **Commander obs-seam hypothesis FALSIFIED** (step-0 obs <0.003, mirror canary +0.97, twin reproduces posture exactly).

---

## ✅ TRAINING-DOCTRINE + INC7 SPEC (2026-06-12, laptop fable; writeup handoff/laptop-training-doctrine-2026-06-12/WRITEUP.md; doctrine doc/training_doctrine.md)

### Root cause reframe (supersedes "climb-bin plant gap = whole story")

Training env scores pass as point-mass L-inf < 0.75 m; sim enforces volumetric body-halo contact. All 4 standing gate-3 crashes terminate at L-inf **0.37–0.49 m** with mid-range commands (rate p95 ≈ 0.5 rad/s vs 3.14 cap, 84% authority unused, zero saturation) — positions the trainer calls comfortable passes. The S20 climb-bin residual (+2–2.8 m/s²) supplies ~0.2–0.5 m displacement; geometry fiction converts it to a crash. Bridge threads the identical funnel 0.2 m lower at 0.25 m.

### Verdicts from doctrine probes

- **Crab = near-optimal:** excess along-track drag 0.43 m/s² (3%) ≈ 0.15 s/lap vs drag-optimal thrust-axis rotation; flat basin; ~0.32 rad/s mean yaw cost. No reward change; 15× smaller than envelope lever.
- **Authority = fine:** 12/12 re-threads from ±1.5 m displaced restarts 1.5 s before gate-3 (3× live error); 48/48 jitter×latency finishes. Recovery curricula not load-bearing.
- **inc6 DR robustness measured:** absorbs ±12% global collective, ±25% drag, full climb-bin residual in closed loop, centered. Premise "DR omitted force scale" was FALSE — dr_aero already randomizes coll ±10%, c2 ±24%.
- **META gap ① stale:** inc6 trains `course_mode=random` (±60°/seg, VQ1 held-out). Remaining exposure is sampler range vs unknown VQ2 course — widen only on VQ2 info.

### Stale premises corrected

- "Standing 0/4 = climb-bin plant gap is the whole cause" → geometry fiction is the primary cause; residual supplies displacement.
- "Single-track training = open gap" → CLOSED at distribution level.
- "DR omitted force-model scale" → FALSE (dr_aero was on; inc6 absorbs the residual in closed loop).

### Doctrine summary (full text `docs/training_doctrine.md`)

Robustness priority: (1) honest contact geometry, (2) structured + global force DR, (3) reset/course diversity — NEVER reward damping. Checkpoint gauntlet adds: per-gate crossing L-inf p95 vs contact-true aperture, corridor clearance at x=−1 m, robustness probes, twin-attitude extraction. Inadmissible evidence list updated (margins against 0.75 point aperture are inflated by body halo).

### Advisor items O and R — REJECTED (doctrine session verdict)

- **(O) Arc-length progress along TOGT reference line as dense RL reward:** REJECTED for inc7. Addresses the 8.3→4.27 s structural gap — a Stage-2 / S2-architecture concern, not a contact-geometry fix. Adding a new dense reward term alongside the geometry fix violates one-change-family-per-increment attribution discipline. Revisit at Stage-2 redesign.
- **(R) Gate-crossing lateral-velocity penalty:** REJECTED. Crab (sideslip ~51°) IS the winning trained style (Q1: 3% drag cost, flat basin); a lateral-velocity penalty suppresses a near-optimal posture for a 0.15 s/lap prize. Confirmed at doctrine level: no style terms for measured-near-optimal styles.

### Inc7 spec (c16 lineage; sbatch `rl/peregrine_racing_inc7.sbatch`, DO NOT SUBMIT until gates pass)

- Reward: **byte-identical c16** (no changes).
- Env additions (LAPTOP-INC7-ENV to implement):
  1. `+env.body_radius_m` ∈ [0.28, 0.38] m per env: pass band 0.75−r, collision band (0.75−r, 1.36+r].
  2. `+env.frame_depth_m=0.30`: volumetric frame collision over gate-frame |x| ≤ 0.30 m.
  3. `+dynamics.dr_force_bias`: per-env random world-frame bias ‖b‖ ≤ 3 m/s² in a random speed×tilt bin.
- Plant: S17 mixer (unchanged). S20 refit folds in if ready (dataset LOCAL `shadowpc-postfix-dataset-2026-06-12/`); NOT a launch blocker.
- DR: all inc6 axes + dr_force_bias.
- **Launch gates:** ① V100 config-matrix parity; ② 598+ tests green incl. new geometry unit tests; ③ deploy matrix rebuilt on contact-true scoring.
- **Prediction on record:** standing clears gate-3 ≥0.3 m corridor margin; tails ≤0.25; posture unchanged (~55° tilt crab); lap cost vs inc6 ≤0.3 s.

### Rejected (no-re-litigate ledger additions)

Anti-crab/sideslip terms; aggression/action damping beyond c16; gate-proximity penalty; recovery curriculum as inc7 blocker; reset-distribution redesign for cold start; 60/100 Hz retrain now; course-sampler widening now; `--plant lapse`/`dr_lapse` (previously voided).
