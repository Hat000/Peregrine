# C2 ESTIMATOR CHAIN — REPORT (Peregrine, case-C VQ2 gate-relative pipeline, STEP 3)

**Session:** C2-ESTIMATOR-CHAIN · laptop · opus-4.8 · branch `claude/inspiring-tereshkova-9bd0f7`
(off the P0-CASEC-FOUNDATION base f798d10). **Scope built:** component C2 (estimator chain) + the 17-dim
obs wiring + the offline gauntlet ONLY. NO retrain, NO 20-dim confidence channel, NO live sim, NO
`memory/` edits. Spec: BLUEPRINT §1.2–1.6 + §3.4.

## STATUS: DONE — gauntlet green, full suite green (723 = 700 + 23 new, 0 regressions).

The gate-relative +L observation fix is productionized and its headline numbers REPRODUCE the design
(rel E_bias **−0.000** / RMS **0.131** vs abs **+0.176**/0.283, submap **+0.174**/0.223). The make-or-break
finding survives as designed: rel p90 **0.197 m > 0.155 m** worst-case @ r=0.38 → **CANNOT-SETTLE-OFFLINE**,
escape-hatched to L3. This is the EXPECTED result, NOT a build failure.

---

## WHAT WAS BUILT (all in `src/racer/`, AUGMENTING the absolute KF — it is KEPT)

1. **`kf_rewind.py` — `RewindKF` (productionized from the prototype).** OOSM ring-buffer wrapper around
   `LinearKF`; `update_position_at(t_fix_ns, z, cov)` rewinds to the fix CAPTURE time, applies it, and
   replays the buffered IMU forward (exact, no Bar-Shalom approx). `horizon_s=0.5` (covers CPU-class
   L~125 ms). `assert_horizon_gt(L)` LOUD-guards the horizon≤L trap (drops 100% of fixes → ~21 m
   divergence). Zero-latency / predict-forward-at-now (`t_fix>=now`) is **bit-identical** to bare
   `LinearKF.update_position` (asserted). State proxies (`x/P/position/velocity`) so the navigator reads
   the estimate unchanged.

2. **`localization.gate_relative_inplane_fix(pose, gate, R_wb) -> (z_ned, cov_ned)` — THE core fix.**
   `L = R_wc @ t_cam_gate` (+L, world NED to the SEEN opening); `z_ned = gate.position_ned - L` (same
   world pseudo-fix as the absolute fix). The WIN is the **anisotropic gate-plane cov**: in-plane (X,Y)
   tight PnP lateral `σ = max(0.265, a1·r)` with **NO 0.40 m floor** (the floor is bias-absorption; the
   relative obs has no bias to absorb); along-track (Z, gate normal) **loose** (base 0.50 + attitude
   lever + the floor). `GATE_REL_INPLANE_SIGMA = 0.265` is the **single swappable** calibration constant
   (Track-3 shadow-mode may recalibrate). `a1=0.026` flagged: extrapolates badly past ~24 m, in-band for
   the ~12 m gate-4 window (flat 0.265 there).

3. **Relative-innovation outlier gate (navigator `_apply_gate_relative_fix`).** `nu_ip = B @ (z_rel −
   x_KF)` (in-plane projection via `gate.R_world_gate[:,:2]`), `S_ip = B(P+cov_rel)Bᵀ`,
   `d2_rel = nu_ipᵀ S_ip⁻¹ nu_ip`; **accept iff ≤ chi2(2,0.999)=13.82**. Applied AFTER the absolute fix
   (KEPT); a rejected relative fix leaves the absolute estimate intact. REQUIRED because reproj passes
   93% of depth-flips and the absolute 3-DOF Maha gate (16.27) passes the small/mid in-plane throws this
   2-DOF gate catches.

4. **Estimator → obs wiring (`estimator_obs.py`).** `estimator_state_for_obs(ds, nav)` substitutes the
   estimator's pos/vel into the wire `DroneState` (keeping the trusted given attitude/rates);
   `estimator_obs(...)` runs the **canonical `fly_rl.build_obs`** on it → the 17-dim layout/sign are
   bit-identical to inc7's. In case-C the gate-relative +L estimate sources `pos_g`/`vel_g`; the map bias
   cancels (`pos_g = R_w2g @ (gate_map − p_KF)` with `p_KF` pulled tight to `gate_map − L` == `R_w2g @
   (+L)`). When no fresh in-band fix, this is the absolute-KF continuation (automatic). NO obs[17:20].
   The shim is pure/torch-free; the `build_obs` import is LAZY (G0).

5. **Calibrated gate-frame cov export (`NavState.nav_inplane_sigma` / `nav_along_sigma`).** Navigator
   `_gate_frame_pos_sigma()` projects `P[:3,:3]` into the last-fix gate plane: `inplane =
   sqrt(P_g[ip0,ip0]+P_g[ip1,ip1])` (the §1.6 `sigma_inplane_hat`), `along = sqrt(P_g[along,along])`.
   `inf` until a gate-relative fix anchors a frame. BUILT for the inc8 confidence channel, **UNCONSUMED**
   by the inc7 17-dim obs.

**Wiring into the navigator** is gated by two config flags, both OFF by default → the VQ1 / case-A path
is byte-identical (bare `LinearKF`, in-place fixes, no augment; 58 existing navigator/KF/localization
tests unchanged): `use_rewind_kf` (RewindKF wrap + IMU-clock predict + capture-time fixes) and
`use_gate_relative` (the in-plane augment + relative-innovation gate).

---

## OFFLINE GAUNTLET (BLUEPRINT §3.4) — ALL GREEN

| Gate | Where | Result |
|---|---|---|
| **G0** import torch-free | `tests/test_c2_torch_free.py` (subprocess) | estimator chain imports with `torch` absent. |
| **G1** obs faithfulness + sign | `tests/test_estimator_obs_wiring.py` (+ existing `test_obs_sign_faithfulness`, `test_train_deploy_obs_elementwise`) | estimator wiring == `build_obs` byte-identical; estimator-sourced `pos_g` == `R_w2g@(+L)` ≤1e-5, −L control breaks ~24 m. |
| **G2** estimator units | `tests/test_kf_rewind.py`, `tests/test_gate_relative_fix.py` | `update_position_at(t≥now)` bit-identical; OOSM == in-order oracle (atol 1e-12); P stays SPD through a rewind; horizon>L asserted + horizon<L drops 100%; in-plane σ=0.265 NO floor / along-track loose+floor; relinnov gate keeps 100% clean / rejects 100% flips (≥1.5 m throw). |
| **G3** margin sim | `handoff/.../g3_margin_sim.py` (full report) + `tests/test_g3_margin_invariants.py` (in-suite guard) | rel E_bias −0.000, RMS 0.131, p90 0.197, p99 0.288; **abs 0.283 / submap 0.223 negative controls FAIL** (RMS over 0.155, bias +0.17). frac-over swept r∈{0.21,0.26,0.30,0.33,0.38}; CENTRAL r=0.30 (margin 0.235) rel frac-over 3.5%; worst-case r=0.38 (margin 0.155) 25.5%. |
| **G7** full suite | `pytest` from repo ROOT | **723 passed** (700 baseline + 23 new), 0 regressions. |

**G3 table (n_mc=600, V_RACE=37 m/s, σ_inplane=0.265, MARGIN(r)=0.535−r):**

```
    arm     RMS     p50     p90     p99   E_bias   D_bias  over@0.21 over@0.26 over@0.30 over@0.33 over@0.38
    abs   0.283   0.253   0.414   0.555   +0.176   +0.062   26.5%    44.2%    54.7%    62.7%    78.0%
 submap   0.223   0.204   0.316   0.385   +0.174   +0.063    7.7%    21.5%    34.7%    49.7%    73.7%
    rel   0.131   0.108   0.197   0.288   -0.000   -0.004    0.5%     1.8%     3.5%     8.8%    25.5%
```

This G3 is the bias-FREE baseline (matches c1 part_b). At the measured 1.4° attitude/accel bias the rel
p90 rises further (per `v3_margin_refute`); the bias — not the body radius — is the binding factor
(memory footgun). The escape hatch (L3 at-speed gate-4 recording) is the sole resolver, by design.

---

## KEY DESIGN DECISIONS / DEVIATIONS (re-derived, not trusted from prose)

- **The d2 spec's `e_pred = R_w2g@(gate − x_KF)` is a SIGN TYPO.** The coherent relative innovation is the
  standard KF innovation of the pseudo-fix `z_rel = gate − L` projected in-plane: `nu_ip = B@(z_rel −
  x_KF)`. Implemented the correct (standard-innovation) form; the obs +L convention (`R_w2g@(+L)`) is
  pinned by G1, consistent with `test_obs_sign_faithfulness`.
- **In-plane σ is `max(0.265, a1·r)`, NOT additive `0.265 + a1·r`.** The lateral law is
  `(a1·r)²+sig0²` and the c1-MEASURED 0.265 is the in-band accepted value at ~10 m — `a1·~10.2 = 0.265`,
  so the law and the measurement AGREE there. Additive-on-top would double-count and inflate the rel arm
  off the validated 0.139. The `max` keeps 0.265 flat in the gate-4 window (reproduces 0.131) and lets
  the a1·r law take over only beyond ~10 m. Single swappable constant preserved.
- **Bias cancellation is in the OBS, not the KF.** No single sighting yields a bias-free ABSOLUTE
  position (you lack the gate's true absolute pos). The gate-relative win is: the tight in-plane fix pins
  `p_KF` in-plane to `gate_map − L`, so `pos_g = R_w2g@(gate_map − p_KF)` == `R_w2g@(+L)` and the common
  `db` cancels. The "forbidden submap" arm FAILS not by re-injecting bias into this difference but by
  sourcing the BLURRY loose-absolute `p_KF` (0.24 m) + the bias-absorption floor — G3 reproduces its
  0.223 m.
- **Double-application of `z` (absolute loose, then relative tight in-plane) is intentional** (the §3
  "AUGMENT"): the loose absolute fix barely moves `p_KF` in-plane (small gain), so the relative gate
  still discriminates and the tight in-plane application dominates. Verified the RewindKF second rewind
  does NOT re-apply the absolute fix (it is baked into the replay base snapshot).

---

## NOT BUILT (out of scope — next steps)

- **20-dim confidence channel / inc8 retrain** (§0.1, §2): the cov export exists and is UNCONSUMED;
  obs[17:20] is the inc8 step.
- **Deploy-loop wiring** (`fly_rl._fly_armed` PATH-B insertion, guard replacement §3.2): C2 provides the
  `estimator_obs` SEAM; wiring it into the judged/live loop is the integration step (C6).
- **Full RewindKF capture-time OOSM live** is HARD-blocked on TIMESYNC (P0-b gives IMU-clock stamping +
  predict-forward; the latter ships first, sidesteps TIMESYNC). L on eval HW + `delta_epoch` are
  live-only (L4).
- **Live sim / L3 recording** — the escape hatch that resolves the p90 verdict.

## HOW TO RUN

```
# full report (radius-band frac-over + JSON):
PYTHONPATH=src .venv/Scripts/python.exe handoff/c2-estimator-chain-2026-06-13/g3_margin_sim.py
# the gauntlet (in-suite):
.venv/Scripts/python.exe -m pytest tests/test_kf_rewind.py tests/test_gate_relative_fix.py \
  tests/test_estimator_obs_wiring.py tests/test_navigator_gate_relative.py \
  tests/test_g3_margin_invariants.py tests/test_c2_torch_free.py -q
```

---

## MEMORY-DELTA (≤10 lines)
- **C2-ESTIMATOR-CHAIN DONE (branch `claude/inspiring-tereshkova-9bd0f7`, 723 green = 700+23, 0 regress; pending merge).** Productionized into `src/racer/`: `kf_rewind.RewindKF` (OOSM, horizon 0.5>L assert, zero-lat bit-identical), `localization.gate_relative_inplane_fix` (+L, in-plane σ=0.265 NO floor, along loose+floor), navigator relative-innovation gate (chi2(2,.999)=13.82), `estimator_obs` wiring (reuses `build_obs`, bit-exact, +L), NavState `nav_inplane_sigma`/`nav_along_sigma` cov export (unconsumed by inc7).
- **Gated by config flags `use_rewind_kf` + `use_gate_relative` (BOTH off by default → VQ1/case-A byte-identical, 0 regressions).** Case-C only.
- **G3 REPRODUCES the design headline:** rel E_bias −0.000 / RMS 0.131 / p90 0.197; abs +0.176/0.283, submap +0.174/0.223 (negative controls FAIL). rel p90 0.197 > 0.155 @r=0.38 → CANNOT-SETTLE-OFFLINE survives (escape-hatch L3), BY DESIGN. Central r=0.30 rel frac-over 3.5%.
- 🚩 **σ=0.265 is `GATE_REL_INPLANE_SIGMA` — the SINGLE swappable constant.** In-plane model is `max(0.265, a1·r)` (NOT additive; the law & the c1 measurement agree at ~10 m). a1=0.026 extrapolates badly >24 m (in-band for the ~12 m gate-4 window).
- **d2 spec `e_pred` sign is a TYPO** — implemented the standard-innovation form `nu_ip = B@(z_rel − x_KF)`; +L obs convention pinned (G1, consistent with `test_obs_sign_faithfulness`).
- **NEXT:** inc8 20-dim confidence channel (consume the cov export) + deploy-loop wiring (`fly_rl` PATH-B / guard replacement, C6) + L3/L4 live. Full RewindKF OOSM is TIMESYNC-blocked (predict-forward ships first).
