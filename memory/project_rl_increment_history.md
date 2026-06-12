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
- **Supersession:** inc5 (mixer-blind, retired) → **inc6 (mixer+aero+map plant, corner-tax c16, SHIPPED)** → live transfer = next step.
