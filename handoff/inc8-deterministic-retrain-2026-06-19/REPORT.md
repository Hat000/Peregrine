# inc8 Deterministic-Stability Re-train — σ_p0 measurable (inc8-deterministic-retrain, 2026-06-19)

**Worker:** inc8-deterministic-retrain (opus-4.8, EFFORT max). **Branch:** `inc8-deterministic-retrain`
(local, NOT pushed). **Status:** LIVE — engineering done + laptop-validated; Adroit training launching.

## Goal
Produce an inc8 policy whose DETERMINISTIC mean (test=True) flies the 2-axis camera-pointing course
(look-at ON, g_pitch high enough to seat fixes), then MEASURE its binding gate-4 terminal-centering σ_p0
with the validated torch instrument. Today σ_p0 is NO-DATA: rc1 flies only STOCHASTICALLY; its
deterministic mean crashes ~1 s post-spawn. Deliverable = a measured 2-axis σ_p0 (GO or NO-GO), or a
supported negative that the look-at magnitude is incompatible with a stable deterministic mean.

## ROOT CAUSE — quantified (the new finding)
rc1's checkpoints, read directly (`handoff/.../probe_logstd.py`, run on Adroit):

| seed | action_std (4 CTBR dims) | note |
|---|---|---|
| **0** | **[7.39, 0.16, 0.18, 7.39]** | dims 0,3 SATURATED at exp(2)=7.39 (LOG_STD_MAX); raw logstd 9.0/4.3 |
| 1 | [0.26, 0.26, 0.24, 0.29] | healthy ~0.25 |
| 2 | [0.25, 0.23, 0.24, 0.28] | healthy ~0.25 |

diffaero samples rollout actions `tanh(mean + std·N(0,1))` with a **state-independent learnable
`actor_logstd`** (network/agents.py). rc1 seed0 (the lone deterministic yaw-flyer, σ=0.198) trained with
**two action channels at the MAX std** → near-random bang-bang in the rollout. The policy learned to fly
*in expectation over that chaos* (feedback masks it), so its **mean is an unvisited, unstable operating
point**. The `entropy_weight=0.01` bonus (constant, no schedule) with no std ceiling drove the
saturation. This is the mechanism behind "stochastic 50% → deterministic crash", now localized to the std.

## The fix — three levers (TRAINING-CONFIG only; reward + look-at primitive untouched)
1. **NOISE-ANNEAL (load-bearing)** — `rl/inc8_noise_anneal.py`. Clamp `actor_logstd` to a decaying std
   CEILING (loose hold `std_hold=0.6` from update 0 to kill the 7.39 saturation; geometric decay
   `std_hold→std_floor=0.03` over the back `1-hold_frac` of training to collapse the rollout onto the
   mean) + anneal `entropy_weight→0` over the same window. The policy is then trained on near-deterministic
   actions ⇒ the mean flies. CLAMP not SET (PPO may still go lower). Wired into the existing
   `step_with_periodic_save` monkeypatch in `peregrine_train_inc8.py`, gated `+algo.noise_anneal=true`;
   OFF == byte-identical inc8. Zero diffaero-clone edits.
2. **DETERMINISTIC SELECTION** — `rl/inc8_snapshots.py` (dense numbered snapshots during training,
   `+snapshot_every=100 +snapshot_from_frac=0.4`) + `rl/inc8_select_ckpt.sbatch` (post-hoc: cheap
   deterministic reach-rank over all snapshots via the VALIDATED σ_p0 instrument, then full σ_p0 on the
   top-K). Replaces the runner's STOCHASTIC high-water "best", which did not transfer to the mean.
3. **g_pitch sweep** — find the highest look-at pitch gain a deterministic mean both FLIES and POINTS at
   (rc1's deterministic ceiling was 0.5, where fix_rate=0; we need ≥1 to seat fixes). `GPITCH` submit knob.

## Verification done (laptop .venv, no diffaero)
- `tests/test_inc8_noise_anneal.py` + `tests/test_inc8_snapshots.py`: **29 passed**. Pins the squash
  round-trip to diffaero's constants (init param 0 ↔ std 0.2231), the schedule shape (hold flat / geometric
  decay / floor), OFF==None resolves, and the REAL `actor_logstd` clamp reproducing the rc1 fix
  (7.39→0.6→0.03 on the saturated channels; healthy channels untouched; clamp only lowers).
- `py_compile` of the trainer; new modules import clean (no torch/diffaero at module level).
- diffaero PPO/agents/runner read from the Adroit clone to wire the lever against the real API.

## Campaign plan
**Batch 1** (seed0, parallel, noise-anneal ON, snapshots on) — brackets g_pitch × cap-strength:
- `ds_gp1p0_h06`: GPITCH=1.0 STD_HOLD=0.6 STD_FLOOR=0.03  (lever test + aggressive cap)
- `ds_gp1p0_h15`: GPITCH=1.0 STD_HOLD=1.5 STD_FLOOR=0.05  (cap-strength bracket)
- `ds_gp1p5_h06`: GPITCH=1.5 STD_HOLD=0.6 STD_FLOOR=0.03  (g_pitch headroom)

Each: precheck-gated (3 updates, exits before the 8 h burn on a wiring bug), stochastic FLIGHTCHECK early
kill, monitored mid-run via the deterministic snapshot probe (scancel non-flyers at the anneal tail).
**Batch 2** (informed): best deterministic config → seeds 1,2 (the ≥3-seed GO) + push g_pitch; then
`inc8_select_ckpt` + the full σ_p0 GO/NO-GO.

## Results
_(pending — Batch 1 launching)_

## MEMORY-DELTA
_(pending final result)_
