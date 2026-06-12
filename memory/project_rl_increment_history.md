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
