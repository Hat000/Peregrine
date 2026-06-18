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

## ✅ TWIN HARNESS SEAM FIX (2026-06-12; commit pending)

`CtbrPlant.state()` now emits `orientation_ned_wxyz` in the RAW wire convention (`q_phys·[1,−1,1,−1]`), pinned as literal `_ODO_WIRE_QUAT_CONJ_WXYZ` in `twin.py` — deliberately NOT imported from `frames.py` (training-doctrine §6: harnesses must not share convention constants with deploy decode). The 4 post-8d7b0b3 test_twin* failures were this defect: twin fed true quats, navigator correctly un-conjugated them, KF IMU-predict mis-rotated between given-position updates. Suite: **620/620 green**. Attribution note: prior session wrongly attributed these 4 failures to "uncommitted inc7 rl/"; inc7-env's diagnosis (twin harness, not inc7 code) was CORRECT.

**Test infra note:** `tests/test_twin_fit.py::_sim_run` deliberately records `q` in the legacy euler-alias frame (the legacy fitter's decomposition model), NOT the wire frame — this is correct; do not "fix" it.

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

---

## ✅ INC7 LAUNCHED (2026-06-12, Adroit A100 partition)

### GPU parity gate (job 3270602) — CLEARED
Config-matrix parity gate run on Adroit as job **3270602** (A100 partition). **CLEARS the standing "V100 parity gate not run since S16" footgun.** Note: gate naming is historical ("V100 gate"); always run on whatever GPU is allocated (user directive 2026-06-12 = A100 preferred for all future trainings).

### Three-seed training launch
| Job | Seed | Outputs |
|-----|------|---------|
| 3270605 | 0 | `/scratch/network/fl3689/inc7_run_3270605.out` |
| 3270606 | 1 | `/scratch/network/fl3689/inc7_run_3270606.out` |
| 3270608 | 2 | `/scratch/network/fl3689/inc7_run_3270608.out` |

**Selection protocol:** ≥3-seed generalization average (durable — single-seed gen volatile; inc6 seeds ranged 0.741–0.982). Sbatch: `rl/peregrine_racing_inc7.sbatch`. Env: contact-true geometry (body_radius_m ∈ [0.28,0.38] + frame_depth_m=0.30) + dr_force_bias (regime-binned ≤3 m/s²); reward = c16 byte-identical; plant = S17 mixer.

**Prediction on record (from training-doctrine session):** standing clears gate-3 ≥0.3 m corridor margin; tails ≤0.25; posture unchanged (~55° tilt crab); lap cost vs inc6 ≤0.3 s.

---

## ✅ INC7 COMPLETE + SHIPPED (2026-06-12, commit 7de36af; writeup handoff/laptop-inc7-eval-2026-06-13/WRITEUP.md)

### Checkpoint

**Winner: seed 0 / job 3270605** → `rl/checkpoints/stage1_inc7_actor.pth`
md5 (verified local): **2AFF8D62569BA5FEC769028D72AF50E3**
Sidecar: `{"act_max_thrust": 3.765, "act_max_rate": 3.14}`
Staging dir: `rl/checkpoints/inc7_staging/` — s0 local, s1/s2 Adroit-only.

### 3-seed outcome

| seed | job | status | VQ1 sr | t_med | gen_sr | notes |
|------|-----|--------|--------|-------|--------|-------|
| s0 | 3270605 | COMPLETED | **1.000** | 9.76 s | **0.924** | SHIPPED |
| s1 | 3270606 | COMPLETED | **0.000** | — | 0.000 | COLLAPSED (sr≈0.11 from iter 1000, no recovery) |
| s2 | 3270608 | COMPLETED | **0.669** | 10.96 s | 0.375 | Weak (33% collision, slab_strikes=840) |

2-seed viable gen mean (s0+s2): **(0.924+0.375)/2 = 0.650**. Escape hatch invoked (only 2 viable); proceed with reduced confidence. **🚩 Budget MORE seeds for inc8-class retrains** — narrow convergent basin confirmed.

### VQ1 eval (s0, plant=mixer, r∈[0.28,0.38], 2560 eps standing-start)

| metric | inc7 s0 | inc6 datum | delta |
|--------|---------|-----------|-------|
| sr | 1.000 | 1.000 | 0 |
| t_med | 9.76 s | 9.86 s | **−0.10 s (faster)** |
| PASS_OFFSET max | 0.169 m | 0.277 m | better |
| PASS_MARGIN min | 0.207 m | — | — |
| slab_strikes | 0 | — | clean |
| thr_p95 | 0.030 | 0.061 | smoother |
| crab (tilt max) | 64.3° | 64.3° | unchanged |

### Prediction scoring (on record from training-doctrine)

1. **Gate-3 barrier cleared** — ✅ BINARY PASS (0 collisions/2560 eps); ⚠️ INDETERMINATE quantitative (≥0.30 m sub-claim; sbatch PASS_MARGIN min=0.207 m all gates all eps, bottleneck likely gate 2 at r=0.38; trainreset gate-3 isolation at r=0.38 = 0.101 m cold-start; full-course approach likely higher).
2. **Pass tails ≤0.25 m** — ✅ PASS (max 0.169 m).
3. **G5 tail ≤0.37 m at worst r=0.38** — ✅ PASS (trainreset g5 = 0.292 m, FINISHED).
4. **Crab unchanged** — ✅ PASS (64.3° identical).
5. **Lap cost ≤0.3 s vs inc6** — ✅ PASS (−0.10 s, faster).

### Deploy matrix (s0, lat=2 live operating point, plant=mixer, r=0.33 nominal)

4/4 start modes FINISHED at lat=2 (live point); 10/16 total. Handoff/trainreset fail at lat=0/1 (latency DR pre-compensation; not a deployment concern at live lat=2).

### Pre-flight verification (ultracode 5-lens adversarial fan-out, 2026-06-12, read-only)

Artifact integrity PASS; plant-honesty PASS (eval ran `--plant mixer` + contact geometry r∈[0.28,0.38]/depth 0.30 — NOT legacy flat plant; footgun clean); anti-mirror PASS (deploy matrix built on corrected post-93023cf tools; anti-mirror test keys on LIVE external-reference data — NOT a self-mirror false pass); 9/9 frame-convention tests green. **Verdict: GO for live.** Replacement seed NOT warranted pre-live (s0 shows wide robust sub-basin).

3 sub-blocker concerns on record:
- **(a) Gate-3 margin thin:** ~0.10–0.15 m twin at worst r=0.38 vs known twin→live arrival residual ~0.15–0.25 m → live gate-3 outcome genuinely uncertain. A live gate-3 clip = residual-exceeds-margin (scopes inc8 margin fix), NOT a frame/eval failure.
- **(b) Inc7 config less stable than inc6:** 2/3 seeds viable (1/3 COLLAPSED); two new elements (contact-true volumetric termination + dr_force_bias) plausibly narrowed convergent basin. Budget ≥4 seeds for inc8-class retrains.
- **(c) `fly_rl.py` default-inc4 footgun reaffirmed** — pass `--checkpoint stage1_inc7_actor.pth` explicitly every run.

### Supersession / lineage update

inc6 (`stage1_inc6_actor.pth`) → **inc7 s0 (`stage1_inc7_actor.pth`) = CURRENT BEST (pending live confirm)**.
inc6 remains **LIVE-CONFIRMED FALLBACK** (bridge 2/2 zero-contact ~9 s RL-segment confirmed 2026-06-12).

### Next

SHADOWPC-INC7-LIVE: standing ×5 + bridge ×2 regression. Gate-3 watch (thin margin above).

---

## ✅ INC7 LIVE CONFIRMED (2026-06-13, commit 7fe90df; writeup handoff/shadowpc-inc7-live-2026-06-13/WRITEUP.md)

**FIRST STANDING-START RL FINISHES in project history.** inc7 is the live-confirmed current best.

### Standing start ×5 — results

| flight | result | gates | lap time (s) | gate 3 | n_coll |
|--------|--------|-------|-------------|--------|--------|
| F1 | **FINISHED** | 6/6 | **9.97** | PASS-CLEAN | 0 |
| F2 | **FINISHED** | 6/6 | 11.44 | PASS-CLEAN | 0 |
| F3 | **FINISHED** | 6/6 | 11.46 | PASS-CLEAN | 0 |
| F4 | **FINISHED** | 6/6 | 11.45 | PASS-CLEAN | 0 |
| F5 | **FINISHED** | 6/6 | 11.45 | PASS-CLEAN | 0 |

**5/5 FINISHED, 30/30 gate passes CLEAN, zero contact events.** Gate-3 barrier from inc6 = GONE.

### Bridge ×2 — results

| flight | result | RL-segment (s) | gate 3 | n_coll |
|--------|--------|----------------|--------|--------|
| B1 | **FINISHED** | **~9.19** | PASS-CLEAN | 0 |
| B2 | **FINISHED** | **~10.74** | PASS-CLEAN | 0 |

2/2 FINISHED. Inc6 bridge RL-segment was 8.96 s; B1 +0.23 s (within margin), B2 +1.78 s (bimodal pattern, see below).

### Frame canary (post-session frame_residual_report.py, all 7 recordings)

Mirror canary: **TRUE +0.97 / AS-IS −0.81** across all 7 sessions. Frame conventions intact. Rate canary gain [0.84–0.98] — slight under-read consistent with clock-mixed quat-FD; no axis sign flip. Force residuals: N+D +2.2–3.75 m/s² at high-v/high-tilt (known climb-bin gap, confirmed not fatal — MARGIN absorbed it). E residual: −0.21 to +0.43 m/s² (very small). No new pathology.

### GEOMETRY-HONESTY THESIS VALIDATED

Contact-true volumetric training + body-radius/frame-extrusion DR fixed the standing gate-3 barrier on the FIRST live try. The known +3.5 m/s² N+D climb-bin residual (still present in frame_residual_report) was absorbed by MARGIN rather than eliminated. Confirms the doctrine: robustness from honest geometry + structured DR; do NOT chase every residual. S20 refit stays correctly deferred.

### Prediction scoring (vs training-doctrine predictions)

1. Gate-3 clears standing (binary) → ✅ 5/5, n_coll=0
2. Pass tails ≤0.25 m → ⚠️ INDETERMINATE — see gate-3 D-offset finding below (contact-free ground truth confirmed; track_map offset confounds L-inf metric)
3. Crab ~64° → ✅ live max 63.2° (Δ = −1.1° from offline 64.3°)
4. Lap cost within 0.3 s of inc6 → ✅/⚠️ PARTIAL — F1 9.97 s (+0.11 s vs inc6 9.86 s offline, PASS); F2–F5 ~11.45 s (+1.59 s, bimodal effect); bridge B1 9.19 s (+0.23 s, PASS)

### ~~FINDING (A) — track_map gate-3 D-offset ~1.46 m~~ — FALSIFIED (2026-06-13, Fengyou-verified)

**FALSIFICATION:** The "1.46 m D mis-registration" was a reference-frame artifact. track_map records gate BOTTOM-centre; drone flies through OPENING-centre, ~1.36 m (half outer-height) above it. drone_D − record_bottom_D ≈ −1.36 m at EVERY gate (gate-3's −1.377 is NOT anomalous). Referenced to the opening-centre, gate-3 crosses 0.056 m (3D) — PASS-CLEAN. All 6 gates registered ≤0.37 m in-plane / ≤0.44 m 3D. track_map is trustworthy. Finding-A is dead.

**Lesson (validation-doctrine):** The artifact survived two rounds because both uses (finding-A and the binding-gate D-probe) referenced bottom-centre implicitly. Only independent raw re-derivation (frames.json + track_map.json with the reference convention made explicit) killed it — always cross-check registration claims against the raw bundle with the reference convention stated.

Live gate-3 crossing speed 17.4–17.5 m/s consistent across all 5 flights (unchanged fact).

### ✅ NEW FINDING (B) — bimodal lap times RESOLVED (2026-06-13, commit f4d723b) — RE-LOCATED by P2-OFFLINE-ANALYSIS

**INITIAL CHARACTERIZATION (commit f4d723b, now SUPERSEDED in part):** F1 (9.97 s, sim warmed at sim_t=85.9 s) vs F2–F5 (~11.45 s, fresh race sim_t≈5.2 s). Physics-state/HOME-reset classified; policy sensitivity ELIMINATED (k=0 obs/action identical). Gap initially inferred as POST-gate-3 from gate-3 crossing speed being constant (~17.4–17.5 m/s).

**🚩 SUPERSEDED — PER-TICK RE-LOCATION (P2-OFFLINE-ANALYSIS, 2026-06-13, DEFINITIVE):** Per-tick analysis of all 5 flights reveals the ENTIRE 1.48 s bimodal gap is in the **start→g0 segment (104% of total)**. Every inter-gate interval g0→g1 through g4→fin is identical across all 5 flights to ±0.04 s. The bimodal is a **PRE-GATE-0 COLD-START ARTIFACT** (~1.5 s sim physics stabilization before the first tick), NOT a post-gate-3 trajectory/speed risk. The earlier "gap lives post-gate-3" inference was an error: constant gate-3 SPEED ≠ constant gate-3 TIME.

**PRESERVED FACTS:**
- BENIGN: both modes FINISH clean, 0 contact.
- **Deployment baseline CONFIRMED: 11.45 s** (F2–F5 fresh regime, competition-representative).
- The ~1.5 s cold-start penalty is competition-representative (every fresh judged run pays it; physics-limited; mitigation unclear — note for inc8/deploy picture).
- HOME-reset-before-each-standing-batch eval discipline STANDS (the startup offset is real).

**SOFTENED:** the winner-validation RIDER's rationale of "fresh-respawn → post-gate-3 trajectory-basin sensitivity" is void (inter-gate trajectories are deterministic, ±0.04 s) — live-verifying the inc8 winner fresh remains GOOD PRACTICE (confirm clean finish + startup offset), but it is no longer guarding a trajectory-sensitivity risk.

Detail: `handoff/shadowpc-bimodal-char-2026-06-13/WRITEUP.md` (initial); P2-OFFLINE-ANALYSIS 2026-06-13 (per-tick definitive).

### Supersession

inc6 (`stage1_inc6_actor.pth`) demoted to historical fallback (bridge-only live-confirmed). **inc7 s0 (`stage1_inc7_actor.pth`) = LIVE-CONFIRMED CURRENT BEST** (standing 5/5 + bridge 2/2, 2026-06-13).

### Deployment note

`fly_rl.py` default still points at inc4 — pass `--checkpoint stage1_inc7_actor.pth` explicitly every run (footgun unchanged).

## §INC8-DESIGN (2026-06-13, ultracode 6-lens fan-out + critic; commander-synthesized STAGED program)

### Critic adjudications (THREE overrides — supersede naive "fire 15 arms")

**1. BIMODAL lap-time (finding B) = TOP COMMITMENT RISK — ✅ DEFUSED (2026-06-13); ROOT CAUSE RE-LOCATED (P2-OFFLINE-ANALYSIS, 2026-06-13, DEFINITIVE).**
Competition runs a FRESH sim = the 11.45 s regime, NOT the offline 9.76 s median. Root cause CLASSIFIED: PHYSICS-STATE/HOME-RESET (NOT policy sensitivity, NOT sim_t0 warmup). Policy sensitivity ELIMINATED (k=0 obs/action identical across F1–F5); warmup-age causality ELIMINATED by inc6 proxy. Risk DEFUSED: the bimodal is a sim initialization artifact — the policy is not broken, both modes finish clean. **Inc8 evals must reset to HOME before every standing-start eval batch; never report warm continuous-session laps as competition estimates; add ~1.5 s to offline standing medians for fresh-sim estimates.**
🚩 **PER-TICK RE-LOCATION (P2-OFFLINE-ANALYSIS, DEFINITIVE):** The ENTIRE 1.48 s gap is in the start→g0 segment (104% of total). Every inter-gate interval g0→g1 through g4→fin is identical across all 5 flights to ±0.04 s. The bimodal is a PRE-GATE-0 cold-start artifact (~1.5 s sim physics stabilization before the first tick), NOT a post-gate-3 trajectory or speed risk. The "gap lives post-gate-3" inference in the initial characterization (f4d723b) was an error: constant gate-3 SPEED does not imply constant gate-3 TIME. Inter-gate trajectories are fully deterministic.

**2. rw_progress bump is OVERRATED.**
max|tanh|≈0.547 is IDENTICAL inc6 (loose point-mass geom) vs inc7 (tight contact-true). If reward gradient were the bottleneck the looser inc6 would use MORE authority. Same value ⇒ the binding constraint is the ENVELOPE/GEOMETRY, NOT the reward. **Relax the envelope; do NOT pump progress.**

**3. rw_finish_time bump is FUTILE.**
At γ=0.99 over ~1000 steps the terminal bonus is discounted to ~5e-5; 3× of ~0 is ~0. Use a γ 0.99→0.995 arm instead (if the discount check confirms). Per doctrine §4: SPLIT merged arms for attribution; do NOT bundle.

---

### Staged program

#### Phase 0 — measure-first (cheap, sonnet, ~no Adroit; gates everything)

**(a) Bimodal root-cause** ✅ REPORTED (2026-06-13, commit f4d723b): physics-state/HOME-reset (NOT warmup, NOT policy); baseline 11.45 s confirmed; inc8 Phase-1 top-risk gate CLEARED. 🚩 **SUPERSEDED IN PART (P2-OFFLINE-ANALYSIS, 2026-06-13):** gap is PRE-gate-0 cold-start (NOT post-gate-3 as originally reported — constant gate-3 speed ≠ constant gate-3 time; per-tick definitive re-location). See §INC7-LIVE-CONFIRMED NEW FINDING (B) above.

**(b) METRIC INSTRUMENT** (replaces point-mass proxy metrics):
- Sim-contact-truth: COLLISION id 1001 + active_gate_index as PRIMARY live scoring (kills the gate-3 D-offset dependency on track_map L-inf).
- Per-gate margin DISTRIBUTIONS with gate-3 isolated as its own row.
- Contact-true offline thresholds: pass band 0.75−r, collision band up to 1.36+r — retires the point-mass 0.75 m aperture.
- Seed-stability score S_stable = frac(seeds reaching VQ1 sr≥0.90), gate ≥2/3.
- Map-offset sensitivity probe: ±1.5 m gate-3 D shift (validates whether gate-3 metric is confounded).

**(c) Inc7 BASELINE PROBES:**
- gen_stress triplet: vq1 / nominal / stress course modes.
- Per-gate |tanh| authority profile + 3-line debug_obs logging.
- γ/discount check: read DiffAero ppo.yaml (confirm discount value for terminal-bonus math).
- 30 Hz per-gate last-accepted-fix distance (speed-ceiling gating measurement).

---

#### Phase 0(b) RESULTS + binding-gate triage (2026-06-13, commit 0bb60cc; VERIFIED + CORRECTED commit 82c2d20)

**INSTRUMENT SHIPPED:** `rl/contact_true_eval.py` = canonical inc8 selection API. Contact-true margin = `(0.75 − body_radius) − linf`; reuses `slab_frame_hit_np` / `offline_rollout` geometry verbatim; 27 new tests + 200×6 legacy parity gate; 647/647 green.

**🚩 MEASUREMENT BUG FIXED (82c2d20):** `per_gate_margin_stats` had hard-coded `pass_band = PASS_BAND_NOM` (r=0.33) regardless of `--body-radius` → margins were IDENTICAL at every radius → the prior multi-radius sweep in the original 0bb60cc bank was MEANINGLESS. Now `pass_band = 0.75 − args.body_radius` threads through correctly. Tests 647→**657** green (+10 new: generic-probe keys/gate-id/per-gate-nominal/D-offset-direction/gate3-wrapper-equivalence; pass_band-shift; start_filter).

**Inc7 baseline (scored at r=0.33, plant=mixer, pooled — for reference; use SIMSTART rows for binding-gate decisions):**

| gate | n_pass | linf_med | margin_min | note |
|------|--------|----------|------------|------|
| 0 | 2 | 0.109 | 0.311 | |
| 1 | 3 | 0.079 | 0.328 | |
| 2 | 4 | 0.144 | 0.216 | |
| **3** | **5** | **0.036** | **0.287** | non-binding; simstart linf 0.051 |
| **4** | **6** | **0.161** | **0.205** | **tightest pooled** |
| 5 | 7 | 0.059 | 0.268 | |

S_stable = **1.000 (7/7 eval seeds)** — EVAL-COVERAGE (fraction of start scenarios finishing on deterministic twin), NOT training-seed viability. Inc8 ≥2/3 escape-hatch gate is TRAINING-seed convergence from ≥5-seed Adroit fan-out — DIFFERENT metric; keep distinct.

**BINDING GATE = GATE-4 (VERIFIED, commit 82c2d20; SUPERSEDES "post-gate-3 / gates 4-5" framing from 0bb60cc).**

Multi-radius simstart table (inc7-training-consistent primary = r=0.38; `linf` radius-invariant; margin = `(0.75−r)−linf`):

| gate | simstart linf | margin r=0.28 | margin r=0.33 | **margin r=0.38 (primary)** |
|:----:|:-------------:|:-------------:|:-------------:|:---------------------------:|
| 0 | 0.109 | 0.361 | 0.311 | 0.261 |
| 1 | 0.079 | 0.391 | 0.341 | 0.291 |
| 2 | 0.151 | 0.319 | 0.269 | 0.219 |
| 3 | 0.051 | 0.419 | 0.369 | 0.319 |
| **4** | **0.215** | **0.255** | **0.205** | **0.155 ← BINDING** |
| 5 | 0.056 | 0.414 | 0.364 | 0.314 |

**Gate-4 is binding at EVERY radius** — margins shift by constant `(0.75−r)`, so ranking is RADIUS-INVARIANT. Gate-5 is CLEAN (simstart linf 0.056 / margin 0.314 @ r=0.38). Risk localizes to gate-4 specifically; NOT the whole 4–5 segment. Gate-4 registration confirmed offline (live-crosses ~0.10 m in-plane; map offset not a concern).

**Binding is START-TYPE-DEPENDENT:**

| start type | binding gate | margin @ r=0.38 | runner-up | note |
|---|---|---|---|---|
| **SIMSTART** (full-course, competition-representative) | **gate-4** | **0.155** | gate-2 @ 0.219 | gap 0.064 m |
| TRAINRESET (synthetic, at-rest 1 m-back) | gate-2 | 0.166 | gate-4 @ 0.171 | gate-4 linf drops 0.215→0.199 |

POOLED or trainreset views MIS-RANK the binding gate. Always use SIMSTART for the post-gate-3 binding question. The linf drop (0.215→0.199) is direct evidence the **high-speed post-gate-3 approach is the stressor** that makes gate-4 bind.

**D-offset probe now PARAMETRIC (gates 3/4/5; 82c2d20):** ±1.5 m probe on all three gates — ALL flip pass→collision. Fragility is UNIVERSAL, not gate-4-specific. The probe does NOT single out any gate as map-confounded; it is sensitivity analysis only.

**🚩 GATE-3 TRACK_MAP D-REGISTRATION: FINDING-A FALSIFIED (2026-06-13, Fengyou-verified; supersedes prior "CONFIRMED REAL" / commit 82c2d20 claim):**
- track_map records gate BOTTOM-centre; drone flies through OPENING-centre (~1.36 m above). drone_D − record_bottom_D ≈ −1.36 m at EVERY gate — gate-3's −1.377 m is NOT anomalous. Referenced to opening-centre, gate-3 crosses 0.056 m (3D), PASS-CLEAN.
- All 6 gates registered ≤0.37 m in-plane. track_map is trustworthy. Finding-A dead.
- Gate-3 nominal margin USABLE for ranking (non-binding, live-clean).
- **Lesson:** the artifact survived because both the inc7-live finding and the inc8 D-probe used the bottom-centre reference implicitly. Only independent raw re-derivation (frames.json + track_map.json with convention stated) falsified it.

**GATE-4 REGISTRATION CONFIRMED OFFLINE:** gate-4 live-crosses ~0.10 m in-plane (fid 1020, course_bundle), rules out ≥0.5 m offset. The 0.155 m @ r=0.38 margin is NOT threatened by a map mis-registration. Ranking robust; absolute sub-0.155 m clearance still wants the live winner-validation batch (orthogonal policy/physics-state question — keep the rider).

**🚩 TRACK_MAP REGISTRATION / VISION-CAL EXTRINSIC-CALIBRATION DESIGN folded into the vision case-C ultracode workstream (commander decision, 2026-06-13).**

~~**🚩 CONVERGENT SYNTHESIS (high-value):** bimodal-char (1.48 s gap lives post-gate-3) + contact-true eval (gate-4 binding, also post-gate-3) → **POST-GATE-3 SEGMENT = inc8 risk zone for BOTH speed and validity.**~~
**🚩 VOIDED (P2-OFFLINE-ANALYSIS, 2026-06-13):** The bimodal is a PRE-GATE-0 cold-start artifact (per-tick definitive). The "gap lives post-gate-3" premise is FALSE — it was an inference error. The synthesis's "post-gate-3 = BOTH speed and validity risk zone" framing is VOID. Do NOT design inc8 around a false post-gate-3 speed-risk zone.
**PRESERVED INDEPENDENTLY:** Gate-4 remains the BINDING inc8 gate (0.155 m @ r=0.38, SIMSTART, registration-confirmed) — valid on its own from the contact-true eval, with no bimodal corroboration required. Envelope-ladder primary margin guard: gate-4.

**RIDER (SOFTENED RATIONALE) — offline twin is structurally blind to fresh-respawn cold-start offset.** Offline median 9.76 s ≈ warm-live F1 9.97 s, both ~1.5 s under fresh 11.45 s (pre-gate-0 cold-start). Crown inc8 winner ONLY after fresh-reset live batch (≥3–5 laps); `gen_stress` is best offline proxy. Perturbation-recovery / velocity-anneal curricula STAY rejected. NOTE: the former "fresh-respawn → post-gate-3 trajectory-basin sensitivity" rationale for this rider is void (inter-gate trajectories are deterministic, ±0.04 s) — the rider now guards the ~1.5 s startup offset and confirms a clean finish, not trajectory-sensitivity.

---

#### Phase 1 — measured-physics arms (Adroit, ≥5 seeds, gated on Phase 0)

**HIGHEST-LEVERAGE: the envelope split** (only change backed by a MEASURED 2.3 s/lap physics effect).

- **Arm 3a:** rw_tilt 96→48 (WEIGHT only, cone unchanged) — isolated attribution for tilt-weight change.
- **Arm 3b:** free-cone 60°→75° (CONE only, weight unchanged) — isolated attribution for cone relaxation.
  - 3a and 3b MUST be SPLIT per doctrine §4; do NOT merge.
- **Body-radius A/B:**
  - A: FIXED r=0.38 m (inc7 config, baseline).
  - B: RANDOMIZED U[0.28, 0.38] m (proxy for combined arrival uncertainty ~0.15–0.25 m + halo; NOT physical body variation).
  - ≥4 seeds each. Prediction: randomized B wins ~0.1–0.2 s/lap by calibrating margins vs over-conserving on the 5 easy gates.
- **Bake CRITIC-SAVE into every sbatch:** 3 lines in `_run_with_lifelines` — unblocks Tier-1 critic-as-risk-monitor (parallel systems).

---

#### Phase 2 — conditional (gated on Phase 1 results)

- **γ 0.99→0.995:** if terminal-bonus γ-check (Phase 0c) confirms invisible bonus. Standalone arm.
- **Network-arch ablation 128/64 and 256/256:** if 1/3 seed collapse persists after Phase 1 (narrow basin = candidate cause; never tested).
- **obs yaw sin/cos:** slot 16 ±π discontinuity; free accuracy improvement but backward-incompatible with existing obs layout → standalone arm only, never bundled.
- **Sampler-widening ±60°→±80°:** if inc7 gen_stress (Phase 0c) shows a cliff at ±60°.

---

### Rejected / deferred (no re-litigate)

| Item | Disposition |
|------|-------------|
| Arc-length/MPCC (advisor O) | Stage-2 — doctrine stands |
| Perturbation-recovery curriculum | Proven 12/12 in inc6; narrows basin |
| Velocity-anneal curriculum | Narrows basin; deferred |
| Deploy action-noise | Determinism directive + thin gate-3 margin |
| logstd-as-uncertainty | Global PPO constant [1,4], zero per-obs info |
| Force-bias band widening | S20 refit is the fix; deferred to pre-inc9 |
| Monolithic 15-job batch | → staged batches with escape hatch after each phase |

---

### Seed budget

**≥5 seeds/arm** (inc7 1/3-viable confirmed narrow convergent basin; 2/3 gate for escape hatch).

---

### Body-radius rationale (user critique validated)

Randomization U[0.28,0.38] is a proxy for the combined arrival residual (~0.15–0.25 m) + halo uncertainty, NOT physical body variation. This is the honest framing: it calibrates training margins to reflect actual live uncertainty rather than physical body size. Fixed r=0.38 is the conservative incumbent; randomized is the challenger (hypothesis: reduces over-conservation on 5 easy gates at cost of slightly tighter gate-3 margin).

---

## ✅ inc7 → inc8 (case-C self-localizing racer; staged look-at primitive, 2026-06-16) — episodic detail routed from MEMORY.md

**inc7 = LIVE-CONFIRMED current best** (VQ1 5/5, gate-3 barrier gone). **inc8 = case-C deployability increment:** a 2-axis active-perception LOOK-AT PRIMITIVE (ENGINEER the camera-pointing) + a dense terminal-σ_p0 centering reward. **Pivot rationale:** 4 reward-WEIGHT iterations all NO-GO (low-pass→terminal exploit; band-pass→abandons pointing; fix-driven v1 job 3274179→no gradient: conf_shape ~constant −0.29/step, fix_bonus never fires) ⇒ it is reward ARCHITECTURE not weights (pointing→fix is sparse/all-or-nothing → PPO can't discover it from a non-pointing start at ANY weight; geometric-pointing proxy ≠ accepted fix). → ENGINEER the pointing (Fengyou-approved 2026-06-16).

**Staging — each axis sign is EMPIRICAL (the analytical FLU/FRD sign was WRONG BOTH times; the CTBR legacy alias is not reasonable a priori → trust the band metric in ~100 steps, not the derivation):**
- **S0 (yaw-only, lookat_g_yaw)** — main @ e21d15f, OFF byte-identical, 25 unit tests (look-at matrix PINNED to canonical frames; action 4-dim CTBR FLU, flip diag(1,−1,−1)). #1 at +3.0 = SIGN INVERTED (band_az 0→~56° UNANIMOUS 3/3 = structural). Flipped to −3.0 (RUNTAG=yawneg3) → **1/3 converged** (seed-0 band_az 60→30°, fix_rate 0→0.08-0.09, lockband 0→0.21, line OK; seed-1 local-min stall; seed-2 entropy-collapse @step 600). Yaw sign = **NEGATIVE** confirmed.
- **S1 (+rw_centering on the yaw base):** @1.0 → 1/3 converged (seed-0 band_az 17-35°, fix_rate 0→0.16, lockband 0→0.55), estim_err 0.115 vs ≲0.08 = NOT-GO. @3.0 (jobs 3274993-95) → estim_err 0.13-0.15 = WORSE; rw=3.0 destabilized 1/2 independent runs (value_loss 715 @step100, entropy −7 by step 1100 — larger weight = worse PPO instability). 🚩 **the @3.0 "1/2 independent" was a Hydra dir collision** (seeds 0+2 launched the same start-second → same timestamped checkpoint dir → bit-identical traces; FIX = per-seed `hydra.run.dir`). → **rw_centering sweep CLOSED.**
- 🚩 **ROOT CAUSE of the estim_err floor:** yaw-only → no elevation fix → **terminal_pointing = 0.000 across ALL seeds ALL runs** → the gate misses the vertical FoV in terminal approach → σ_p0 saturates 0.11-0.15 regardless of centering weight. Confirms "azimuth-corrects-but-fix-flat → S2 (+g_pitch)".
- **S2 (+lookat_g_pitch, the elevation axis):** pitch sign = **+3.0** (probe: g_pitch=−3.0 → band_el 83-90° INVERTED both seeds → +3.0 by elimination; OPPOSITE the yaw −3.0 = 2nd empirical inversion). Stage-2 live 2026-06-16 (jobs 3275300-02, ++g_yaw=-3.0 ++g_pitch=3.0 +warmup=200, NO centering, per-seed Hydra dirs). seed 2 CONVERGED (band_el 24-31°, band_az 22-37°, fix_rate 0.11-0.16, terminal_pointing 0.03-0.13); seeds 0/1 non-convergent (warmup). estim_err floor 0.115-0.130 = the estimator's gate-relative RMS ceiling (0.131 from the C2 chain), NOT policy failure. VERDICT = **2-AXIS PRIMITIVE CONFIRMED**; formal GO not cleared (estim_err 0.115 > 0.08).
- **S3 (+rw_centering=1.0 on the 2-axis base, jobs 3275483-85, 2026-06-17):** seeds 0+1 = warmup collapse (value_loss 432/572 @step100, before pointing established — same warmup fragility as S2 seeds 0/1). Seed 2 = warmup stable → 2-axis breakout steps 800-1100 (fix_rate 0.04-0.11, terminal_pointing 0.04-0.09) → value_loss SPIKE 650 @step1200 → permanent collapse. estim_err in the seed-2 window = 0.107-0.123 (NOT <0.08; too few centering updates to bite). Zombie estim_err <0.08 on seeds 0/1 DISQUALIFIED (no-fix KF drifts err_ip→0 = artifact, not real centering). **S3 NOT-GO.** Full traces → handoff/inc8-s3-2026-06-17/REPORT.md.
  - 🚩 **REWARD CLIFF mechanism:** rw_centering ≈ range-weighted err_ip. Cold-start, pre-pointing: no fixes → KF prior dead-reckons to gate-center → err_ip≈0 → centering penalty≈0. At pointing onset, fixes land → KF reveals the drone is ~0.12 m off-center → centering penalty fires at FULL strength *discontinuously* → the symmetric obs-20 critic (CRITIC-IS-SYMMETRIC footgun, **2nd convergent signal**) can't anticipate it → no value smoothing → value_loss spike → cascade. A GT-privileged critic (algo=appo) would price the off-center penalty in from step 0 → smooth advantage.
  - 🚩 **COMMANDER REFINEMENT 1 — the cliff is CASE-C-SPECIFIC:** err_ip is observation-dependent (the no-fix→fix discontinuity) ONLY because position is self-localized. If VQ2 streams position (case-A, like VQ1 did), err_ip is truthful from step 0, there is NO cliff, and appo is unnecessary for centering. ⇒ **the S4/appo decision is GATED on the VQ2-tomorrow wire** (alongside the whole case-C load-bearing-vs-insurance question). Do NOT build appo before VQ2 resolves — the problem it fixes may not exist.
  - 🚩 **COMMANDER REFINEMENT 2 — warm-start is the cheaper pre-appo probe (rank it above the rw=0.3 hedge):** the cliff is a COLD-START transition artifact. WARM-START S3 from the S2-seed2 converged checkpoint (pointing already established, fixes already flowing): the centering penalty is then honest + roughly constant from fine-tune step 0 (no regime change) → dodges BOTH the seed-0/1 warmup collapse AND the seed-2 onset cliff, with NO build. Keep the pointing reward active so the policy can't dodge centering by dropping pointing (the look-at reward pins it on). If warm-start ALSO collapses, that is strong evidence appo is truly mandatory (good diagnostic either way). Use 3 fine-tune RNG seeds off the single seed-2 init for diversity. Risk-order: **warm-start** (small build — see below) → **appo** (GuardedPPO→AsymmetricPPO + env state_dim=36 wire + parity + smoke + rerun, ~worker-day) → **rw_centering=0.3 fresh-seed** (patch; shrinks but doesn't remove the discontinuity; weaker centering).
- **S4-PREP — WARM-START BUILT + MERGED (branch p2-inc8-warmstart → main 11cb840, 2026-06-17; laptop code, NOT launched, Adroit/VQ2-gated):** `rl/inc8_warmstart.py` `maybe_warmstart(agent,env,cfg)` + 1 guarded call in `peregrine_train_inc8.py` (after the step-hook, before `_orig_run`). KEY = **root-level `+init_from=<ckpt dir>`** (single-plus; diffaero has NO `train` group — the briefed `+train.init_from` would've been a dead namespace; worker verified source). LOAD = diffaero's own **`agent.load`** (PPO.load→actor.load+critic.load = the inverse of the lifeline periodic-save; reads actor.pth{actor_mean,actor_logstd}+critic.pth) — **WEIGHTS-only** (optimizer+buffer FRESH = correct for a reward re-pilot); did NOT reuse fly_rl.load_actor (deploy-side _ActorMean, wrong target, no critic). Sidecar obs-dim gate (17-dim inc7 can't warm-start 20-dim inc8); critic stays SYMMETRIC (NOT appo); look-at warmup FORCED→0 (loaded policy already points). **OFF (+init_from unset) = BYTE-IDENTICAL** (returns None on 1st stmt; no torch/agent/env/RNG touch — 3-leg proof incl. `_ExplodingAgent` + AST no-other-agent.load test); 13 new tests, full suite 871 pass. sbatch `rl/peregrine_inc8_warmstart.sbatch` = S0/S2-convergent COMMON verbatim + `++env.lookat_g_yaw=-3.0 ++env.lookat_g_pitch=3.0 +env.lookat_warmup_updates=0 +env.rw_centering=1.0` + per-seed hydra.run.dir + AUP + file-copy pre-sync. GO = physical σ_p0 (contact_true_eval), NOT floored estim_err; headline = NO value_loss cliff @~step1200 (= warm-start solved it; spike = escalate to appo). 🚩 launch caveat: `${CKPT}` assumes S2 seed-2 runname `inc8_s0_yawprim_seed2_s2full_s2` + a `periodic/` save dir — sbatch header has the ls/cat to verify sidecar obs_dim:20/inc8:true BEFORE submit.
- **S4 WARM-START RESULT (3/3 RC=0, 2026-06-17): 3/3 SEEDS CLEAN — both S3 cold-start failure modes DISSOLVED.** value_loss at the S3 cliff zone (S3-fresh was ~650 → collapse): seed0 11.95 / seed1 19.73 / seed2 5.04 @step1200; band (steps 100-3990) ~2-30; only a single step-0 transient (132-162, gone by step 100-500 = fresh optimizer + loaded critic meeting the new centering term for one update — NOT a cliff). Pointing RETAINED + centering ENGAGED all 3: fix_rate 0.14-0.16, terminal_pointing >0 (~0.02-0.07), centering ~−0.04 steady, entropy −2.1 to −3.5 (no −6 zombie collapse). 🚩 **VERDICT: appo NOT needed — the cheap pre-appo probe WON** (warm-start removes the cold-start regime change that caused BOTH S3's warmup deaths and the onset cliff). 3 ws checkpoints on Adroit `outputs/train/inc8_warmstart_seed{0,1,2}_ws1/`. 🚩 **REAL GO still pending = physical σ_p0 (contact_true_eval, GT-anchored)** — 3/3 healthy training w/ retained pointing is the PRECONDITION (green), not the GO; estim_err is floored ~0.13 regardless of policy so it is NOT the gate. 🚩 σ_p0 from the estimator EMUL = **PROVISIONAL** (#37 "selection on obs_from_truth crowns a fiction" — the real-detector vertical spike must confirm emul fidelity before any crown). Ops: stale Adroit serve daemon (the auth-race cause) killed.
- **S4 σ_p0 EVAL = UN-MEASURABLE (2026-06-17): reach_rate 0/200 ALL 3 warm-start seeds.** Pulled via the Adroit serve daemon (md5-verified distinct policies 98f72253/65cea7eb/f8ed4be3; sidecars obs_dim:20/inc8:true). No episode reaches the gate-4 crossing → empty GT-anchored ensemble → worker correctly REFUSED to fabricate (the only in-plane numbers ip_p90≈0.24 = floored estim_err of a no-fix coast, NOT centering). 🚩 **ROOT = TRAIN/EVAL OBS MISMATCH:** through the EVAL estimator_emul all 3 show lock_g4=0.00 / fixrate_g4=0.000 (policy doesn't point → KF coasts → misses every gate) — CONTRADICTS the warm-start TRAINING traces (fix_rate 0.14-0.16, pointing>0). Same policy, opposite behavior → eval obs ≠ training obs = #37 "selection on obs_from_truth crowns a fiction" CONCRETIZED + now BLOCKING. RULED OUT: instrument/loader (inc7 truth-obs gives a real −0.215 m crossing; 20-dim actors load + fly varied multi-gate trajectories, not a crash) + robustness (same reach=0 best as periodic) → failure is UPSTREAM of the metric, in policy behavior under the emulated obs. NEXT = **obs-parity diagnosis** (drive training-env obs vs eval-instrument obs from a common state for one ws ckpt; localize the divergence — prime suspects: the [17:20] _confidence_triple @ estimator_emul.py:307 [survey: estimator_obs.py:67 omits [17:20]], the +L/frame/KF-pose seam, and whether the policy's look-at output is wired into the eval emul's fix-acceptance AT ALL) → INSTRUMENT-GAP (eval mis-builds obs, policy fine, fixable + re-run) vs POLICY-GAP (training pointing was an artifact). 🚩 the warm-start TRAINING win (cliff dissolved, appo dodged) STANDS — this is goal-achievement BLOCKED, not the training undone. σ_p0 un-measurable until parity resolved; PROVISIONAL vs the real chain (#37) even after. Report: handoff/inc8-sigmap0-eval-2026-06-17/RESULTS.md.
- **S4 OBS-PARITY DIAGNOSIS RESOLVED (e9bf3b7, 2026-06-17) = POLICY-GAP (course-flight), NOT instrument-gap.** Bisection: (a) inc8 ws-seed2 on PERFECT pose (truth obs, no KF, no look-at) MISSES gate-0 0/10, drifts ~5 m off-center while approaching (along-track 23→3 m); (b) inc7 in the IDENTICAL harness finishes 3/3 all gates (+ real eval simstart FINISHED, S_STABLE 1.000) → harness faithful, inc8 policy is the problem; (c) systematic across EVERY inc8 ckpt (stage1_inc8, ws seed0/1/2, seed0_BEST) = the LINEAGE not a seed. The 3 obs/wiring hypotheses RULED OUT with proof: obs[0:17] frame/+L/seam = PARITY (obs_zup_torch train inc8_estimator_emul.py:138 vs obs_from_zup eval fly_rl.py:307, max 7.6e-6 over 200 states); [17:20] triple = irrelevant (misses gate-0 for every constant triple); fix-acceptance wiring = works (eval derives accept from truth attitude→geometry estimator_emul.py:275, n_fix≈9, pointing 0.43 — fixrate_g4=0 is downstream of never reaching gate-4). 🚩 **WHY TRAINING LOOKED FINE: fix_rate 0.14-0.16 / terminal_pointing>0 are PER-STEP BATCH MEANS — a pointing-but-resetting policy yields exactly those with ZERO laps. COURSE-COMPLETION WAS NEVER A GO CRITERION anywhere in the ladder** (warm-start GO was only 'value_loss no-cliff + pointing retained'). 🚩 LIKELY ROOT = the inc8 reward REMOVED R1-to-centre (peregrine_racing_inc8.py:329) for arc-Γ + near-gate-only centring → policy under-centres through the approach → drifts off course. SECONDARY (real, not the cause): the eval omits the look-at primitive training composes onto the action (peregrine_racing_inc8.py:233-246 applies it; contact_true_eval.py:307-309 uses the raw action) — port for a faithful σ_p0, but doesn't fix reach (fails on truth obs too; a faithful re-inject went OOB). 🚩 **FIX (ordered):** (1) Adroit: read in-training success_rate/n_passed_gates (peregrine_racing_inc8.py:443) — ~0 ⇒ pure policy-gap, settles it in one query (diffaero not importable on the laptop, so this couldn't be closed there). (2) ADD course-completion to the inc8 GO criteria (every stage). (3) restore through-approach centring pressure. (4) port look-at into the eval. σ_p0 stays NO-GO until a policy completes a lap. The warm-start TRAINING-STABILITY win (cliff dissolved, appo dodged) STANDS — this reframes inc8 as 'points-but-doesn't-fly,' a reward + GO-criteria fix, not the training undone.

**Stability knobs (branch e114727 → main c83d071, all OFF byte-identical, 77 inc8 tests green):** band_el metric (the pitch sign-oracle), look-at gain-warmup (`+env.lookat_warmup_updates`, the 2/3-collapse lever — magnitude-only ramp, sign-safe), critic-width A/B (`+algo.critic_hidden_dim`, actor untouched). 🚩 **Worker finding:** the inc8 critic is SYMMETRIC (obs-input 20), NOT the asymmetric privileged 36-dim critic the SSOT assumed (GuardedPPO extends diffaero symmetric PPO; algo=ppo; get_state(36)/state_dim never consumed) → candidate root-cause of the 2/3 collapse; the privileged critic = a separate `algo=appo` move. Confirm on Adroit (`agent.agent.critic.critic.input_dim`==20).
