# Worker #74 — LinearKF in-plane STATE-covariance floor (over-convergence on dense fixes)

**Branch:** `worker/kf-pfloor-2026-06-15`  ·  **Date:** 2026-06-15  ·  **Base:** `main` @ de9ff35

## Problem (parked #74, coast-drift 2026-06-15)
The production `LinearKF` over-converges on a dense gate-relative fix stream: the position covariance
`P` shrinks as `R/N` (independent updates accumulate) toward ~mm, so the Kalman gain for new fixes → ~0,
the KF stops trusting fixes and rides the drifting IMU — **"centering-blind" exactly when a well-pointed
inc8 policy makes fixes densest.** The existing MEASUREMENT R-floor (`localization.FIX_COV_FLOOR_STD`)
does NOT fix it: repeated floored-R updates still drive `P → R/N → 0`. The fix is a **STATE** (`P`) floor.

## The fix
Floor the **smallest eigenvalue of the horizontal (N-E) position block** `P[:2,:2]` to `σ_floor²` AFTER
each update, so the KF can never claim better than `~σ_floor` gate-relative centering (it physically
can't — the irreducible systematic bias `σ_b` survives the boresight bake), keeping the gain responsive.

- **Value:** `σ_floor = 0.05 m` = `localization.INPLANE_POS_FLOOR_STD`, aligned to `σ_ref`
  (`rl.estimator_emul.SIGMA_REF_M = 0.05`, the obs confidence reference) and the `σ_b` the coast-drift
  `σ_p0 ⊕ σ_b` budget assumes. **Not a new magic number.**
- **Frame:** world-NED horizontal (N-E) only. **Eigenvalue** floor (not per-axis diag) → catches the
  binding gate-LATERAL/centering direction at any gate yaw, and raises ONLY the small (over-converged)
  eigenvalue — the loose along-track horizontal eigenvalue (var ≫ floor) and the vertical (world-down) +
  velocity blocks are left as-is. (Vertical bias is owned by the boresight/ESKF pathway, per the task.)
- **Site:** `src/racer/state_estimator.py` — `LinearKF.inplane_pos_floor_std` field (default **0.0 =
  DISABLED → byte-identical**) + `_apply_inplane_pos_floor()` called at the end of `update()`. PSD +
  symmetry preserved (raising a principal sub-block's eigenvalues adds a PSD increment to `P`). NO-OP
  (P bit-unchanged, early return) when disabled OR when the block already clears the floor.
- **Gating:** `src/racer/navigator.py` — `NavigatorConfig.use_inplane_pos_floor=True` +
  `inplane_pos_floor_std=0.05`. Activated in `_initialize` **only** when `use_rewind_kf OR
  use_gate_relative` (the case-C estimator chain). VQ1 / case-A (both OFF) → `floor_std=0.0` → bare
  filter → **byte-identical**, matching the existing C2 gated-off invariant.

## Does the over-convergence reproduce? (escape-hatch question)
**In the iid regime (no process noise), YES; with realistic correlated fixes, it is much milder and the
production G3 sim never trips it.**

| regime | in-plane σ after 200 dense fixes (floor OFF) | next-fix gain (OFF) | floor ON |
|---|---|---|---|
| no-Q (pure R/N, the coast-drift iid MC) | **18.7 mm** (collapsed) | 0.005 (decaying →0) | σ pinned 50 mm, gain 0.034 |
| with-Q (IMU predict between fixes, realistic) | **39.7 mm** | 0.022 | σ pinned 50 mm, gain 0.034 |

The IMU predict's process noise `Q` between fixes partially counteracts the `R/N` collapse. The floored
gain asymptotes to `floor²/(floor²+R_lat) ≈ 0.034` (non-decaying) vs the bare gain decaying as `1/N`.

## Ripple (b) — C2 G3 case-C margin: **bit-identical (not regressed)**
Re-ran the validated G3 `rel` arm (`RewindKF` + production cov) with floor OFF vs ON, at the production
14 Hz and dense 60/120 Hz (`handoff/kf-pfloor-2026-06-15/g3_pfloor_ripple.py`):

| fix_hz | floor | n_fix | RMS | p50 | p90 | p99 | E_bias | D_bias | min in-plane σ |
|---|---|---|---|---|---|---|---|---|---|
| 14 | 0.00 | 10 | 0.131 | 0.108 | 0.197 | 0.288 | −0.000 | −0.004 | 0.124 |
| 14 | 0.05 | 10 | 0.131 | 0.108 | 0.197 | 0.288 | −0.000 | −0.004 | 0.124 |
| 60 | 0.00 | 29 | 0.084 | 0.068 | 0.126 | 0.186 | −0.002 | −0.002 | 0.074 |
| 60 | 0.05 | 29 | 0.084 | 0.068 | 0.126 | 0.186 | −0.002 | −0.002 | 0.074 |
| 120 | 0.00/0.05 | 29 | 0.084 | … | … | … | … | … | 0.074 |

Floor OFF == ON to every printed digit. The realistic correlated stream (`RewindKF` predict + bounded
fix count in the 12 m window) self-limits the in-plane σ at **0.074–0.124 m — always above the 0.05
floor**, so the floor is a true NO-OP here. **The over-convergence the iid MC implied does NOT reproduce
in the production G3 scenario** → the floor is pure insurance for a future denser/longer-coast regime,
not load-bearing today. (The OFF/14 Hz row reproduces the documented G3 baseline: RMS 0.131, p90 0.197,
E_bias −0.000 — confirming the default path is unperturbed by the new field.)

## Ripple (a) — obs[17:20] contract: intact + more honest
With the lateral σ over-converged to 13 mm (floor OFF), the **pre-clip** confidence ratio
`σ_ref/σ_lat = 3.77` — i.e. `clip(·,0,1)` was masking a 3.8× over-confidence. With the floor the pre-clip
ratio is exactly **1.00** (honest, not clip-masked). `obs_dim` stays 20, `[0:17]` untouched, and
`c_inplane ∈ [0,1]` holds either way (the clip). The floor IMPROVES honesty, does not break the contract.
Note: `c_inplane` mixes lateral (floored, horizontal) + vertical (world-down, left as-is per scope); full
`c_inplane` honesty additionally needs the vertical floor owned by the boresight/ESKF pathway.

## Ripple (c) — NEES calibration: over-confident → ~1 DOF (honest, not under-confident)
MC (400 runs, dense fixes carrying a constant lateral systematic `b = σ_floor`), lateral
`NEES = mean(e²)/mean(P)` (1-DOF, ~1 calibrated):

| regime | NEES floor ON | NEES floor OFF |
|---|---|---|
| no-Q, 300 fixes | **1.54** (P held at 0.05²) | 11.46 (P collapsed → over-confident) |
| with-Q, 300 fixes | 1.62 | 2.62 |

Floored NEES is consistently order-1 (1.4–1.9) — honest and **not under-confident** (≫0.3); the bare
filter is materially over-confident. (`test_floor_keeps_nees_calibrated_not_overconfident`.)

## Tests added
- `tests/test_state_estimator_pfloor.py` (11 tests): collapse-without-floor; σ pinned at floor;
  responsiveness (gain non-negligible & ≫ bare); eigenvalue-not-axis-aligned; single-application
  isolation (vertical/velocity/cross-blocks untouched); SPD+symmetric; floor-OFF byte-identical;
  no-op-when-above-floor bit-identical; NEES calibration; Q-makes-it-milder.
- `tests/test_navigator_gate_relative.py` (+3): case-C activates the floor; VQ1 keeps floor=0;
  `use_inplane_pos_floor=False` kill-switch → case-C floor=0.

## Gating / non-regression
- **Full suite GREEN** (`PYTHONPATH=src pytest -q`): **826 passed, 42 skipped, 0 failed** (+14 new tests
  in this branch), 0 regressions.
- VQ1 / inc7 spoon-fed path: **byte-identical** (floor default 0.0 → early return; gated off in case-A).
- C2 gated-off (rewind+gate-rel OFF): **byte-identical** (floor inactive).
- numpy↔torch `LinearKF` parity (S1/S2 gate): preserved — the torch `BatchedLinearKF` and the inc8
  training emul construct the KF unfloored (default), and the parity tests pass unchanged.

## Caveats / surfaced
- The floor is **insurance**: in the production G3 correlated regime it never binds (σ stays >0.05 via
  `Q` + bounded fix count). It activates only under a sustained dense-fix / low-Q / long-coast regime.
- The inc8 **training** emul (`rl/estimator_emul.py`, torch `BatchedLinearKF`) does NOT carry the floor.
  No observable train/deploy mismatch today: `c_inplane` saturates to 1.0 identically floored/unfloored,
  and a floored deploy KF only tracks BETTER (higher gain). If a future training config produces a
  denser-than-G3 stream, mirror the floor into the emul for parity. (Out of this worker's scope.)
