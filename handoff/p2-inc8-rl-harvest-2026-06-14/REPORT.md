# P2 INC8-RL Harvest — REPORT
**Branch:** `p2-inc8-rl-harvest`
**Date:** 2026-06-14
**Source:** `claude/affectionate-brattain-08cb15` (commit 947721e)
**Base:** `main` (5a4afa6 / bc71085)

## Status: GREEN

Two instruments ported from `claude/affectionate-brattain-08cb15` onto main's code,
preserving the canonical GREEN behavior (722 pass / 0 fail; +2 new tests).

---

## Instruments Ported

### 1. PASSIVE-OBSERVER mode

**What:** `run_episode(estim_emul=True, emul_passive=True)` runs the
`EstimatorEmulator` alongside a truth-driven policy. The estimator records the
fix stream, KF error, and gate-4 instrumentation **without closing the loop** —
the policy sees `obs_from_truth` and completes the course.

**Why it matters:** Measures the estimator machinery under a course-completing
policy (how L3 measured inc7's fix-rate). Separates the POINTING lever signal
from policy-degradation effects, enabling a clean measurement of
`gate4_band_fix_rate` and `terminal_gate_lock_frac` on a valid trajectory.

**Changes:**
- `rl/contact_true_eval.py` — added `emul_passive: bool = False` to
  `run_episode()`. When `emul_passive=True` and `estim_emul=True`, the emulator's
  `step()` is still called every step (full fix-rate instrumentation), but the
  policy obs is sourced from `obs_from_truth` instead of `emulator.obs()`.

**Parity with source branch:** The passive-observer branch is functionally
identical to the source branch's `emul_passive` path. The `step()` call ordering
(`k > 0` guard retained from main) gives the same fix-rate statistics on
non-trivial trajectories. `result.emulator` holds the full instrumentation.

**Verification:** `test_passive_observer_records_fix_rate_on_truth_trajectory`
— episode FINISHES on simstart (truth obs → no degradation); `gate4_band_fix_rate`
and `terminal_gate_lock_frac` are valid floats in [0,1]; ≥1 fix recorded.

---

### 2. CLOSED-LOOP S_stable on emulated obs

**What:** `--estim-emul` flag in `main()` now collects per-seed emulator
instrumentation in `emul_metrics` dict, prints a full per-seed gate-4 selection
table, and updates `CONTACT_TRUE_SUMMARY` with `estim_emul=ON/OFF`.

The `compute_s_stable()` call in `main()` already runs on the emulated-obs
episodes (S_stable = fraction of seeds that FINISH), so the closed-loop S_stable
is automatically produced when `--estim-emul` is active.

**Reported value (banked from W1 source branch):** inc7 / 4-7 seeds = 0.571.

**Changes:**
- `rl/estimator_emul.py` — added `actor_obs_dim(actor)` function (first
  `nn.Linear` in_features; fallback 17).
- `rl/contact_true_eval.py`:
  - Top-level `from estimator_emul import actor_obs_dim` (removes the try/except
    inline attr access for obs_dim inference).
  - `--obs-dim` CLI arg retained for backward compat; `actor_obs_dim(actor)` used
    as the default.
  - `emul_metrics` dict collected per-seed from `result.emulator` (using existing
    `v_star()`, `gate4_band_fix_rate()`, `terminal_gate_lock_frac()`,
    `gate4_inplane_error_series()` accessor methods).
  - Full per-seed gate-4 selection table printed when any emul_metrics collected.
  - `CONTACT_TRUE_SUMMARY` now includes `estim_emul=ON ...` or `estim_emul=OFF`.

**Parity with source branch:** The source branch uses property-based accessors
(`v_star`, not `v_star()`). Our port uses the current main's method-based API
(`v_star()`, etc.) — functionally identical values.

**Verification:** `test_closed_loop_sstable_inc7_on_emulated_obs` — runs all 7
seeds (simstart + trainreset_g0..g5) with `estim_emul=True, emul_seed=si`,
computes S_stable, checks `3 ≤ n_stable ≤ 5` (reproduces banked ~0.571 / 4-7
with ±1 seed slack). Also verifies `S_stable < 1.0` (degradation) and
`S_stable ≥ 2/7` (viable).

---

## Files Changed

| File | Change |
|------|--------|
| `rl/estimator_emul.py` | +`actor_obs_dim()` function |
| `rl/contact_true_eval.py` | +`emul_passive` param; +`actor_obs_dim` import; +`emul_metrics` dict; improved `main()` reporting; updated summary |
| `tests/test_estimator_emul.py` | +2 tests: passive-observer + closed-loop S_stable |
| `handoff/p2-inc8-rl-harvest-2026-06-14/REPORT.md` | This file |

## Test Results

```
722 passed, 42 skipped (0 failed)   [full suite, 3m04s]
13 passed                           [test_estimator_emul.py alone]
```

All 5 GATE conditions (frame-seam identity, NEES calibration, pointing→fix
coupling, obs[17:20] bounds, no-regression) remain GREEN.

---

## MEMORY-DELTA

```
- Harvested 2 instruments from claude/affectionate-brattain-08cb15 → p2-inc8-rl-harvest
- passive-observer: run_episode(estim_emul=True, emul_passive=True) → policy on truth, emulator records fix stream
- closed-loop S_stable: --estim-emul in main() → compute_s_stable on emulated obs; inc7 ~0.571 / 4-7
- actor_obs_dim() added to estimator_emul.py
- Full suite: 722 pass / 0 fail
- Branch: p2-inc8-rl-harvest
```
