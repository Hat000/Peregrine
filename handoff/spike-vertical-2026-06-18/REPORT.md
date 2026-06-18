# Vertical-slice spike — close + freeze the full-stack loop on ONE frame

**Date:** 2026-06-18 · **Branch:** `p2-spike-vertical` (off `main` @ ebdbf42) · **Gate:** green_gate GREEN (947 ≥ 933)

## VERDICT: LOOP CLOSES = **YES**

`frame(pixels) → 8→4 slice → PnP GatePose → case-C Navigator (RewindKF + gate-relative +L) → NavState
→ obs[0:20] (build_obs ⊕ d5 confidence triple) → inc8 actor → action` runs end-to-end and yields a
**finite, in-bounds action**. The golden 4-tuple (GatePose, NavState, obs[0:20], action) is frozen and
deterministic (bit-identical run-to-run). **7/7 golden tests pass; green_gate GREEN.** No structural
break (no sign/frame/order/σ mismatch survived a seam assert). The escape hatch did NOT fire.

Success criterion was wiring closure + golden + emul-delta, **NOT lap-completion** — and the known
policy-gap (inc8 can't fly a lap even on truth obs) is visible here as a near-zero-thrust action (see §Action).

## Artifacts
- `rl/spike_vertical_slice.py` — the two new seams (`slice_8kp_to_4`, `deploy_confidence_triple`) + the
  Navigator driver + actor step + emul cross-check. `python -m spike_vertical_slice` prints the goldens.
- `tests/test_spike_golden.py` — 7 frozen/per-seam tests (the regression anchor).
- `handoff/spike-vertical-2026-06-18/_*.py` — reproducible recon/verify scripts (evidence).

## Data reality (shapes the spike)
- **8-kpt `best.pt` (vq2_pose_8kp_r1clean) is NOT staged** — no `gh` on PATH, no `_staged/`, release not
  pulled. Local `models/*.pt` are all **4-kpt legacy** pose models (`kpt_shape=[4,3]`).
- **680 real VQ1 frames** (360×640 BGR, known ranges 1.8–23.3 m) available in a worktree handoff dir.
- ⇒ closed-loop golden uses **synthetic-exact pixels** through the **REAL** stack (the only way to a
  deterministic golden — there is no synchronized odometry for the captured frames). The **real-detector
  arm is the SENSE seam only** (see §Real detector).

## Per-seam verification

| Seam | Check | Result |
|---|---|---|
| **SLICE** (blocker#3) | 8-kpt `(1,8,2)` → inner-4 slice == feeding inner-4 directly | **bit-identical** ✓ |
| **IPPE order** | project KNOWN pose → 4px (LL,LR,UR,UL) → PnP → recover | `|Δt|~1e-15`, reproj~1e-14, cheirality ✓ |
| **+L sign** | localization +L lever recovers true pose (BORESIGHT zeroed) | `+L` err 9e-16; **−L control breaks 16.06 m** ✓ |
| **KF (case-C)** | real Navigator (RewindKF + relinnov gate) → converged NavState | x,y exact; **z −1.75 = true −2.0 + BORESIGHT −0.25** ✓ |
| **obs[0:17]** | deploy `build_obs` vs emul `obs_from_zup`, same state | **max\|Δ\| = 0.0** (parity, ⊂ known 7.6e-6) ✓ |
| **obs[17:20]** | `deploy_confidence_triple` vs REAL `EstimatorEmulator.confidence_channel` | **max\|Δ\| = 9.2e-9** (float32 only) ✓ |
| **ACTOR** | `load_actor`(20-dim, sidecar act_max 3.765) → `policy_step` | rate `[0.53,2.35,0.07]`≤3.14, finite+in-bounds ✓ |

## emul-vs-real obs[0:20] delta (#37 "emul fiction")
The deploy obs the inc8 actor would see **IS** what it trained on, at the encoding level:
- **obs[0:17]: Δ = 0.0** — the deploy `build_obs` path and the emul's `obs_from_zup` are bit-identical for
  the same KF pos/vel + attitude (the inc7 contract is unchanged; this corroborates the banked 7.6e-6).
- **obs[17:20]: Δ = 9.2e-9** — `deploy_confidence_triple` reproduces the emul `confidence_channel`
  byte-for-byte (delta is pure float32 rounding). The deploy [17:20] builder is faithful.
- **Range fan (KF converged at 7.5/10.5/13.5 m):** seam holds at every range (obs17 Δ=0.0, fidelity ≤3e-8,
  action finite+in-bounds); the confidence channel is physically range-dependent: `c_along` 0.75→0.65→0.55
  as range grows (fix cov grows with range), `c_inplane` stays high (in-plane floored ~0.05).
- **inplane floor ON (deploy) vs OFF (emul-train):** identical P at this convergence (one relative fix →
  not over-converged). At dense-fix saturation BOTH regimes clip `c_inplane→1.0`, so the one known
  deploy/emul KF difference **does not materially move obs[17:20]** — a de-risking datum for #37.

## ⚠️ Findings the burn must carry
1. **σ FOOTGUN (CONFIRMED FROM SOURCE + numerically):** `NavState.nav_inplane_sigma` =
   `sqrt(P_E+P_D)` (navigator.py:626, the **sum**); the emul `confidence_channel` =
   `sqrt((P_E+P_D)/2)` (estimator_emul.py:323, the **average**). Ratio = **√2 exactly** (pinned by a test).
   A deploy obs[17:20] built from `NavState.nav_inplane_sigma` would feed the actor `c_inplane=0.70`
   instead of the trained `0.99` — a 1/√2 train/deploy obs bug. **The deploy builder must use the EMUL
   convention** (`deploy_confidence_triple` does; do NOT wire `NavState.nav_inplane_sigma` into obs[17]).
2. **obs[17:20] HAS NO PRODUCTION BUILDER:** `estimator_obs`/`build_obs` stop at 17 dims (the deploy gap,
   estimator_obs.py:67). `deploy_confidence_triple` should be **promoted into `estimator_obs.py`** for the
   real deploy path (a 20-dim actor otherwise gets garbage/zeros for [17:20]).
3. **Navigator RANGE-CONSISTENCY GATE (deploy insight):** the cold origin-seed's expected range to the
   first gate must roughly match the PnP range, or fixes are **rejected** and the estimate never leaves the
   seed (the spike's off-range geometries stayed at the σ≈5 cold prior). Cold-start must seed a plausible
   prior range (or relax `fix_range_*_tol`) for the first gate.
4. **Real-detector ACCURACY is BLOCKED on staging the 8-kpt R1 champion:** the local 4-kpt legacy model on
   real frames produces **correctly-ordered but poorly-localized** keypoints (image quadrants = canonical
   LL,LR,UR,UL ✓; reproj 7–19 px; PnP range 3–4 m regardless of true 2.6–19.3 m). Order is sound; accuracy
   is the weak legacy model. The SENSE **wiring** (detector loads/runs/produces poses; PnP consumes them)
   is confirmed on real pixels — but a faithful real-detector datum needs `best.pt` pulled to `_staged/`.

## Scope / limitations (deliberate, for a wiring spike)
- Closed loop uses **synthetic-exact pixels** (deterministic golden); the **real frame** contribution is
  the SENSE seam (real 4-kpt YOLO → PnP → GatePose runs).
- **Level attitude** scenario → the R_y(π) odo-quat conjugation is exercised but is a no-op at identity; a
  non-level attitude golden would stress it (follow-up).
- **Look-at primitive NOT composed** onto the action (flight-faithfulness; parked with the policy-gap). The
  action is the raw `policy_step` output — finite+in-bounds, which is the success criterion.

## Action (the closed-loop output)
`rate_frd=[0.534, 2.347, 0.071] rad/s` (|·|≤3.14 ✓), `collective≈1.1e-9`, `normed_thrust≈4.0e-9`
(both in-bounds). The near-zero thrust is the policy's response to a static/somewhat-OOD obs and is
**consistent with the banked lineage policy-gap** (inc8 points but cannot fly a lap), NOT a wiring defect.
The criterion — a finite, in-bounds action emerges from the closed seam — is met.

## MEMORY-DELTA (≤10 lines, for the commander to bank)
- 🚩 inc8 VERTICAL-SLICE SPIKE CLOSED THE LOOP (p2-spike-vertical, green_gate GREEN 947): frame→8/4-slice→PnP→case-C Navigator→NavState→obs[0:20]→inc8 actor = finite in-bounds action; golden frozen `tests/test_spike_golden.py` (7/7). #37 wiring de-risked.
- 🚩 σ FOOTGUN CONFIRMED FROM SOURCE: `NavState.nav_inplane_sigma`=sqrt(P_E+P_D) (navigator.py:626) = **√2 ×** emul `confidence_channel`=sqrt((P_E+P_D)/2) (estimator_emul.py:323). Deploy obs[17:20] MUST use the EMUL convention (`rl/spike_vertical_slice.deploy_confidence_triple`), NOT NavState's field (else c_inplane off by 1/√2).
- 🚩 obs[17:20] has NO production builder (estimator_obs stops at 17) — promote `deploy_confidence_triple` into `estimator_obs.py` for deploy. obs[0:17] deploy `build_obs`≡emul `obs_from_zup` Δ=0.0; [17:20] deploy≡emul Δ=9e-9 (builder faithful).
- 🚩 8-kpt `best.pt` NOT staged (no gh/_staged; only 4-kpt legacy local) → real-detector ACCURACY arm blocked (legacy: order-correct, localization-poor on real frames); SENSE wiring confirmed. CRITICAL-PATH-ZERO data-staging still binding.
- Navigator range-consistency gate: cold origin-seed expected-range must ≈ PnP range or fixes rejected (estimate stuck at seed) — deploy cold-start consideration.
- BORESIGHT −0.25 confirmed live in the deploy z-fix (NavState z −1.75 vs true −2.0; zeroed → −2.0001). +L sign holds end-to-end (−L control breaks 16 m).
