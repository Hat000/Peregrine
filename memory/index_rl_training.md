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
- Gap decomposition: 35.3 s (VQ1) → 9.76 s (inc7) → 6.89 s (cone tax, OPEN — biggest lever) → ~4.72 s (honest bound).
- `rl/reference_line_vq1.json` drag-infeasible + 170° inverted → must be REBUILT. `rl/reference_line.py` loader valid.

## STYLE ENVELOPE COST + cone relaxation ladder
- 🚩 ≤65° roll = ~2.87 s/lap (ledger; inc5 A/B). 60° cone bound ~9.8–10.6 s ≈ inc7 live 9.76 s → gap = CONE, not policy/architecture.
- Relaxation ladder: (1) rw_tilt 96→48 (~1.4 s); (2) free-cone 60°→75–80° (~1.5 s — **THE binding kinematic lever**); (3) unconstrained.
- **COUPLING:** faster speed worsens gate-4 margin/σ ratio → estimator accuracy must be verified at each speed rung.
- **PRIMARY MARGIN GUARD = GATE-4** (0.155 m @ r=0.38). Gate-5 CLEAN (0.314 m).

## NEXT queue — RL items
⑥ 🚩 **P4-C05 gate-yaw hardcode (GO-BEFORE-VQ2; FOUNDATION for gate-relative obs rebuild):** `obs_from_zup`/`build_obs` hardcode yaw=pi — silently wrong up to 4.22 m on any VQ2/non-pi course.
⑦ 🚩 **CR1-01 + P1-C06 → WINNER-VALIDATION RIDER:** fold yaw-active segment into inc8 fresh-reset live batch; closes absolute yaw wire sign + obs yaw seam in one capture.
⑧ **INC8 sub-tasks (ordered):**
  1. Graft arc-length progress reward.
  2. Corrected-aero min-snap+TOPP reference line.
  3. rw_tilt 96→48 + free-cone 60°→~70° **GATED on gate-4 metric** (BSR3 spin-margin gate MANDATORY before retrain — widen spin_rate_abort→~9–10 + spin_time_abort→3.0 s).
  4. Retrain ≥4 seeds.
  5. scipy-SLSQP toy MPCC probe.
  - **INC8 BINDING GATE = GATE-4 (0.155 m @ r=0.38, SIMSTART, registration-confirmed).**
- **RL SPEED-LADDER PORTFOLIO:** train deliberate envelope ladder; select fastest contact-valid policy the gate-relative estimator can DELIVER.

## DiffAero / Adroit substrate
- `src/racer/rl_plant.py` + `rl/diffaero_dynamics.py`, parity BIT-IDENTICAL; 2048-env PPO ~88.9 K steps/s.
- 🚩 **GPU PREFERENCE:** prefer A100 but do NOT idle — THROUGHPUT beats waiting.
- 🚩 **Retrain footguns:** evals default legacy flat plant — all evals must run **map-ON**; `/scratch/network/fl3689/peregrine_repo` is FILE COPY not git clone; parity gate CLEARED job 3270602.
- 🚩 **30 Hz = inherited default (DiffAero racing.yaml).** Binds ONLY at ≥30 m/s × last-fix ≤10 m; fixes to 15 m → NOT bottleneck.

## Topic file pointers
- [[project-rl-increment-history]] — checkpoint lineage inc1→inc7, stage history S1.1→S17, inc7 job IDs, md5s/commits, NaN/sidecar/OOB bug histories, §INC7-LIVE-CONFIRMED, §Phase-0(b).
- [[project-phase2-rl-vision-decisions]] — RL/S2/inc8/doctrine sections: §S2-DECISION, §PLANNING-TOGT-S2, §S17, §TRAINING-DOCTRINE, §CORNER-PASS, §INC8-DESIGN, §RL-PORTFOLIO, §FRAME-AUDIT.
- [[project-tooling-recommendations-eval]] — 3rd-party triage; keepers ADRC/GTSAM/min-snap/Isaac.
- [[feedback-checkpoint-transfer]] — small RL checkpoints go in git; ShadowPC pulls.
