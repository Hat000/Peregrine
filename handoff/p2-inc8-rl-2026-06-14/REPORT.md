# P2 INC8-RL WORKER #1 — eval-side estimator-emulation + escape-hatch gate

**Session:** P2 INC8-RL Worker #1 · opus-4.8 · MAX. **Branch:** `p2-inc8-rl` (off main `742dc43`).
**Date:** 2026-06-14.

---

## VERDICT: 🟢 GREEN — the 20-dim obs contract carries the camera-pointing signal honestly.

All four GREEN conditions are met; the escape hatch is **NOT** fired. The held Deliverable-2 torch
train-env port is **UNBLOCKED** (pending the commander's go — I did not touch the train env).

| GREEN condition | result | evidence |
|---|---|---|
| GATE#1 frame-seam identity passes | ✅ | worst \|obs17 − obs_from_truth\| = **0.0** over 250 random states (≤1e-6 required) |
| pointing measurably + monotonically moves fix-rate | ✅ | crab sweep fix_rate **0.368 → 0.000** (gate leaves frame @ ~45–50°); terminal-lock 0.909 → 0.000; monotone |
| KF in-plane error shrinks with fix density | ✅ | with 0.5 m in-plane drift: fixes **0.285 m** vs no-fix **0.695 m** (corrects 59%); single-run decay **0.136 → 0.048 m** as fixes accumulate |
| obs[17:20] channels non-degenerate | ✅ | c_inplane **0.049–0.518** (56 uniq), c_along 0.042–0.134, age_norm **0–1** (cycles); c_inplane rises after a fix (0.398 vs 0.314 mean) |

Plus GATE#2 (KF calibration): pooled NEES = **0.971 ∈ [0.8, 1.3]** (bias-off) — the confidence channel
is honest, not overconfident.

---

## What was built (lane: `rl/` + `cfg/` only; `src/racer/` import-only; train env untouched)

1. **`rl/estimator_emul.py`** (new) — the no-render estimator-emulation core. Runs an ACTUAL
   `racer.state_estimator.LinearKF` driven by the calibrated `rl/fix_surrogate.py`, so
   camera-pointing → fix density → KF accuracy → the obs the policy is scored on.
   - `EmulConfig` — the DR + encoding contract (supersedes d5 per MEMORY NOW): per-fix **lateral σ ~
     U[0.05,0.15]** (overrides `sigma_lateral_floor`, keeps the range-collapse a1 + fitted vertical
     0.28 / depth 0.85); **one-signed in-plane PnP/extrinsic bias ~ U[0,0.19]** with a per-episode
     constant random sign on the lateral+vertical gate-frame axes (the binding case-(b)/OPEN
     assumption; depth bias = 0); **map bias NOT injected** (cancels in gate-relative — the
     anti-pattern). obs[17:20] = `clip(σ_ref/σ_hat,0,1)` (σ_ref=0.05) + `clip(t_since_fix/τ,0,1)`
     (τ=0.10 s), the FROZEN d5 §1.2 layout.
   - `EstimatorEmulator` — KF + FixSurrogate + per-episode DR + staleness clock. `reset` (COLD:
     pos_std 1.0 / vel_std 5.0, IMU `attitude_noise_std=0` = trusted-with-noise so P matches the
     IMU+fix noise → honest NEES), `step` (synthesize body specific force from truth + N(0,0.3),
     predict; sample a fix to the target gate; update + reset the clock), `obs` (the EXACT deploy
     seam — replace ONLY pos/vel with the KF, keep truth attitude/rates; 17 verbatim, 20 = +triple).
   - `ned_gate_frame(yaw)` / `make_ned_gate` — the NED `contracts.Gate` frame `[[s,0,c],[c,0,-s],
     [0,1,0]]` (cols [right, down, downrange]). **This is NOT the Z-up `_R_W2G`** — pinned by the C1
     identity (a pure in-plane NED-gate displacement → 0 obs along-track change; a pure downrange
     displacement → 0 obs in-plane change) to **1.8e-15**.
   - Instrumentation: `v_star`, `gate4_band_fix_rate`, `terminal_gate_lock_frac`,
     `gate4_inplane_error_series`, `confidence_channel`, `nees_inplane_along`.

2. **`rl/contact_true_eval.py`** (extended) — added `--estim-emul` (default OFF = the optimistic
   `obs_from_truth` control), `--obs-dim` (0 = inferred from the actor), `--emul-seed`. `run_episode`
   threads the emulator (`emulator.step` then `emulator.obs` replacing `obs_from_truth`); the
   emulator (+ instrumentation) is attached to `EpisodeResult.emulator`. `CONTACT_TRUE_SUMMARY`
   extended with v*, gate-4 fix-rate, terminal-gate-lock, gate-4 SIMSTART in-plane **p90 AND p99**.
   Runs map-ON (`--plant mixer`).

3. **`tests/test_estimator_emul.py`** (new) — 11 tests, all green; full suite **714 passed, 35
   skipped (pre-existing), 0 regressions**.

---

## Test gates (`tests/test_estimator_emul.py` — 11 passed)

| # | gate | what it pins |
|---|---|---|
| GATE#1 | `test_frame_seam_identity_obs17_equals_obs_from_truth` | KF==truth → obs17 == obs_from_truth ≤1e-6 / 250 states (NED↔Z-up↔gate-frame wiring + the +L sign) |
| GATE#1 | `test_ned_gate_frame_c1_consistency_identity` | the NED gate frame ≠ Z-up `_R_W2G`; in-plane↔along-track decomposition exact to 1e-9 |
| GATE#2 | `test_kf_calibration_nees_in_band_bias_off` | pooled NEES ∈ [0.8,1.3] (honest confidence), bias-OFF |
| GATE#2 | `test_nees_inflates_when_bias_injected` | the one-signed bias is a REAL unobserved systematic (in-plane NEES > 1.3) — WHY GATE#2 is bias-off |
| GATE#3 | `test_pointing_fix_coupling_monotonic_in_crab` | accept rises monotonically as the gate centers; off-pointed → ~0 |
| GATE#3 | `test_pointing_fix_coupling_elevation_axis` | the 2-axis (vertical) pointing lever also gates acceptance |
| GATE#3 | `test_crab_to_fix_rate_table_monotone_and_window_gain` | cross-check vs `crab_to_fix_rate` (monotone) + the ×8.5 in-window gain |
| GATE#4 | `test_obs1720_bounds_and_responsiveness` | [17:20]∈[0,1]; age_norm→1 on dropout; c_inplane rises across a fix |
| GATE#4 | `test_obs20_appends_triple_obs17_unchanged` | obs20[:17] == obs17 (byte-identical) |
| GATE#5 | `test_estim_emul_off_is_byte_identical_to_legacy` | `--estim-emul` OFF == legacy obs_from_truth (no inc7-scoring regression) |
| — | `test_gate_construction_positions_match_course` | NED gate centres == `_GATE_POS_ZUP * _FLIP` |

---

## Escape-hatch probe — the decision evidence

### Machinery verification (controlled gate-4 approach; `probe_machinery.py`)

```
(ii) POINTING -> FIX-RATE lever (crab sweep, bias OFF, mean/6 seeds)
  crab  fix_rate  term_lock  n_fix
     0     0.368      0.909   21.0
    ..     0.368      0.909   21.0   (gate stays in frame through 40deg)
    50     0.000      0.000    0.0   (gate OUT of frame -> zero fixes)
  monotone NON-increasing as the gate de-centers: True

(iii) KF in-plane error SHRINKS with fix density (0.5 m drift entering gate-4)
  CENTERED (fixes)        fix_rate 0.368  ip_terminal 0.285  n_fix 21
  POINTED-AWAY (no fixes) fix_rate 0.000  ip_terminal 0.695  n_fix  0
  => fixes correct 59% of the drift; single-run decay 0.136 -> 0.048 m as fixes accumulate

(iv) obs[17:20] NON-DEGENERACY (centered approach)
  c_inplane  min 0.049 max 0.518 std 0.110 uniq 56   NON-DEGENERATE
  c_along    min 0.042 max 0.134 std 0.028 uniq 56   NON-DEGENERATE
  age_norm   min 0.000 max 1.000 std 0.462 uniq  5   NON-DEGENERATE
  c_inplane right after a fix = 0.398 (vs overall 0.314)  -> responds to the fix stream
```

**Subtle but important (reported honestly):** in a perfectly straight, perfectly-tracked head-on
approach the in-plane axis has NO drift, so fixes only add their noise (no-fix error < with-fix). That
is correct KF behaviour and is itself the memory thesis: the gate-4 **in-plane error floor is set by
the fix model (σ + the one-signed bias), NOT IMU drift** — exactly the "binding factor = vertical
boresight + per-fix σ, not IMU" finding. The (iii) demonstration therefore models the realistic case
(0.5 m of accumulated in-plane drift entering the approach) where fixes are load-bearing, and there
they cut the error 59%.

### inc7 baseline (`contact_true_eval.py --ckpt stage1_inc7_actor.pth --plant mixer --body-radius 0.38 --estim-emul`)

- **SIMSTART = COLLISION @ gate 0** under emulated obs. inc7 (trained on PERFECT pose, zero pointing
  incentive, never trained on noised obs) cannot survive the cold-start estimator transient
  (kf velocity error peaks ~1.5 m/s at the launch as the cold prior recovers via fix-differencing) →
  collides before gate 0. **This is EXPECTED and is the whole reason inc8 exists** — it is the
  cleanest possible demonstration that inc7 has zero robustness to estimator error.
- Gate-4 SIMSTART selection numbers = **N/A** (never reached). `trainreset_g4` (a 1 m at-rest coast
  from truth) gives v*=5.1, gate4_fix_rate **0.000**, terminal_lock **0.000** — confirming inc7's
  ~zero gate-4 fix-rate / pointing (consistent with L3's ~0.014).
- `S_stable` = 0.571 (4/7) on emulated obs (vs the truth-pose 5/5) — the estimator degrades inc7.

### Selection-regime sanity (emulated gate-4 in-plane, cold prior, 12 seeds)

| regime | p50 | p90 | p99 | note |
|---|---|---|---|---|
| cold @ bias 0 | 0.165 | **0.392** | 0.567 | vs banked cold@bias0 p90≈0.234 — same order, ~1.7× (the emulator carries the MEASURED anisotropic in-plane σ: vertical 0.28 dominates lateral 0.10, consistent with the binding-VERTICAL thesis; banked 0.234 used isotropic lateral 0.265) |
| cold @ bias ON (U[0,0.19]) | 0.229 | **0.457** | 0.866 | the one-signed bias SURVIVES the +L fix → floor up — the case-(b)/OPEN binding case the policy must train against |

(Per the reporting discipline: central report r=0.30, select r=0.38; the probe ran r=0.38.)

---

## Reproduce

```
# tests (from repo ROOT, .venv)
.venv\Scripts\python.exe -m pytest tests/test_estimator_emul.py -q          # 11 passed
.venv\Scripts\python.exe -m pytest tests/ -q                                # 714 passed, 35 skipped

# machinery probe (the GREEN evidence)
.venv\Scripts\python.exe handoff/p2-inc8-rl-2026-06-14/probe_machinery.py

# inc7 baseline on emulated obs
.venv\Scripts\python.exe rl/contact_true_eval.py --ckpt rl/checkpoints/stage1_inc7_actor.pth ^
    --plant mixer --body-radius 0.38 --frame-depth 0.30 --estim-emul
```

---

## Notes / carries for the P2 commander

- **The train-env port (Deliverable 2) is unblocked but I did NOT start it** (out of lane). When it
  is built, the DR (`EmulConfig`) + the obs[17:20] encoding here are the reference contract; the
  emulator and the train env must agree (this eval is the SELECTION instrument, the train env the
  learning environment).
- **fix_surrogate σ is a single swappable checkpoint** — when P3's L3 at-speed recording lands, swap
  via `FixSurrogate.from_checkpoints()`; nothing here hardcodes σ except the per-episode
  `sigma_lateral_floor` DR draw (which the prompt specifies).
- **The one-signed bias DR is deliberately the conservative case-(b)/OPEN assumption.** If P3's live
  head-on boresight arbiter (authorized, Fengyou GO) resolves case-(a)/CLOSE, relax `EmulConfig`'s
  `bias_mag_hi` (or set `inject_bias=False`); the GREEN verdict holds under either case (the probe
  tests the POINTING signal, which the bias does not gate).
- **Latency / RewindKF is OUT of v1** (LinearKF predict-forward). A later selection-time refinement;
  the escape hatch does not need it.
- inc7 collides at gate-0 on emulated obs, so the gate-4 SIMSTART numbers come from inc8 policies
  (which will be trained to point) — the eval is ready for them (`--obs-dim` auto-infers 17/20).

---

## MEMORY-DELTA (≤10 lines — do NOT edit memory/ myself)

- **P2 INC8-RL Worker#1 DONE: eval-side estimator-emulation + escape-hatch = 🟢 GREEN.** `rl/estimator_emul.py` (new) runs a real LinearKF off the calibrated fix_surrogate; `rl/contact_true_eval.py` gains `--estim-emul`; `tests/test_estimator_emul.py` 11 green; **full suite 714 pass / 35 skip / 0 regress.** Branch `p2-inc8-rl` (off 742dc43), pushed.
- **ESCAPE HATCH NOT FIRED — the 20-dim obs contract carries the pointing signal honestly.** All 4 GREEN met: frame-seam identity **0.0**; pointing→fix-rate monotone **0.368→0.000**; KF in-plane error shrinks with fixes (0.5 m drift: **0.285 vs 0.695**, −59%); [17:20] non-degenerate (c_inplane 0.049–0.518, age_norm 0–1). KF calibration NEES **0.971** (bias-off). → **inc8 train-env port (Deliverable 2) UNBLOCKED** pending commander go.
- **inc7 on emulated obs = COLLISION @ gate-0** (zero robustness to cold-start estimator error; gate-4 fix-rate/lock ≈0) — the cleanest proof of WHY inc8 must retrain on emulated obs.
- **Emulated gate-4 in-plane is VERTICAL-σ-dominated** (σ_vert 0.28 ≫ σ_lat 0.10): cold@bias0 p90 **0.392** (banked 0.234, ~1.7×, consistent w/ binding-vertical thesis); one-signed bias U[0,0.19] survives the +L fix → p90 **0.457** (the case-(b) floor). Confirms binding axis = fix model (σ+bias), NOT IMU.
- **NED gate frame `[[s,0,c],[c,0,-s],[0,1,0]]` ≠ Z-up `_R_W2G`** — pinned by the C1 in-plane↔along-track identity to 1.8e-15; reuse via `estimator_emul.ned_gate_frame`.
- **DR knobs (EmulConfig, the inc8 train-env reference contract):** lateral σ U[0.05,0.15] (overrides fix_surrogate floor, keeps a1+vertical/depth); one-signed in-plane bias U[0,0.19] (case-b, relax if P3 boresight resolves CLOSE); map bias NOT injected; obs[17:20] σ_ref=0.05 / τ_stale=0.10. KF: attitude_noise=0 (IMU trusted-with-noise).
