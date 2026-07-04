# VQ2 A36 — Turn Convergence: yaw-sign root cause + slow/switch-lanes/physical-plane turn (2026-07-03)

A35 (Fixes 2 & 3) cleared the vertical channel (no ceiling balloon, no floor dip). Run
`20260704_032626_rl_s1_f1` (40.7 s, closest-to-gate-2 yet) then EXPOSED the horizontal killer:
after passing gate 0 the drone **orbited** gate 2 — track_range stayed 15-19 m for ~70 ticks and
never closed. This spec diagnoses the DOMINANT root (a yaw-command sign inversion), corrects it, and
builds the operator's slow + switch-lanes + physical-plane-turn refinements. All flag-gated,
VQ1/case-A byte-identical, `vq2_case_c` only.

---

## 0. Executive summary

| # | Item | Mechanism | Fix |
|---|------|-----------|-----|
| **0** | **Yaw command INVERTED (the root)** | The VQ2 wire NEGATES the emitted yaw-rate command. A22 (07-02) set `body_rate_sign` yaw to +1 off ONE run that predated its own commit; every one of the 16 flights since has flown an inverted yaw loop — the SENT command aims at the gate but the drone yaws AWAY = positive feedback = the orbit. | `body_rate_sign (1,1,1) -> (1,1,-1)` (the VQ1-proven default). |
| 1 | Too fast into the turn | cos^pow forward still pushes hard while off-axis, so the drone arcs wide instead of turning. | POINTING GATE: cut forward to ~0 until pointed (`fwd_point_gate_az_rad`). Default OFF. |
| 3 | Turn never commits / reverts too soon | The wire leads the physical plane ~9 m and self-cancels (H-1b 0.0 coast + acquire-next re-locks the still-ahead gate); the real vis pass coasts only 0.3 s, far short of a ~95° slew. | PHYSICAL-PLANE commit (`pass_wire_requires_near`) + HOLD until pointed (`pass_turn_hold_until_pointed`). |
| 4 | Yaw-to-face + carried past = orbit | Cross-track corrected only by yawing + forward drive (momentum carries past); lateral term capped at 1.5 and clipped behind forward. | LATERAL-FIRST "switch lanes": bank sideways onto the approach line (`use_lateral_first_budget` + raised `image_lat_cap_mps2`). Default OFF. |
| 2 | (superseded) | The "reacquire yaw-inversion" the task scoped was a SYMPTOM of Item 0 (the global yaw inversion), not a separate low-confidence-frame transient. | **No separate fix** — dissolves once Item 0 lands. |

**Item 0 is the headline** and by itself flips the yaw loop from positive to negative feedback.
Items 1/3/4 make the turn SHARP once the drone is turning the right way. **Flight plan:** the first
A36 re-fly is **Item 0 + Item 3 ON, Items 1&4 OFF** (read the sign cleanly), then enable 1&4.

---

## 1. Item 0 — the yaw command is inverted (proven, dominant)

### Evidence
Run `20260704_032626` (whole flight): the seeker computes `yaw_des` correctly toward the gate, but
the drone physically turns the OPPOSITE way to the emitted body-rate yaw command.
* `cmd_yaw_rate` (post-sign `body_rate[2]`) vs `raw_gyro_yaw`: **114 opposed / 0 agree**.
* `cmd_yaw_rate` vs `d(yaw_estimate)/dt`: **120 / 2 opposed**.
* `raw_gyro_yaw` vs `d(yaw_estimate)`: **136 / 2 AGREE** — the estimator faithfully integrates the
  gyro; the estimate is trustworthy (re-confirms A22's T2/T3 estimator+vision exoneration).
* SENT command points TOWARD the gate **132/5** (the seeker aims correctly) yet realized yaw goes
  toward the gate only **23/141** — the drone yaws AWAY from where it is correctly commanded.

**Cross-run:** ALL 16 post-A22 flights with real yaw commands are inverted (opp >> agree, most 0
agree). Not one agrees. This is a positive-feedback yaw loop and is the root of the never-turns-to-
gate-2 orbit, the "yaws left for no reason," and even the 203° whip that motivated the A31
orbit-breaker.

### Why A22 got it backwards
A22 (`d1bca4b`, 07-02 09:01 -0700) set `(1,1,1)` from run `20260702_152528`. That run was created
15:25:28 UTC = **~36 min BEFORE** the fix commit and had NO `vq2_case_c` `body_rate_sign` override,
so it flew on the seeker default **-1**. A22's T1 measured `corr(PRE-sign FRD cmd, realized) = -0.79`
and reasoned "realized tracks the POST-sign wire value → the wire honors plain FRD → set +1." But
the run already flew -1 and was honoring the sent (-1) command correctly; A22 read the pre/post-sign
relationship backwards and flipped a WORKING sign. T2 (estimator) and T3 (vision) only ever
exonerated the estimator + vision — they never proved the actuation sign.

Code path A22→HEAD (`458223d`) is UNCHANGED (`controller.py:513 omega = omega * body_rate_sign`,
logged post-sign as `cmd.body_rate`); `git log d1bca4b..HEAD` on controller.py is all vertical-
channel (A23-A28), touching no yaw/omega/sign line; no gyro/SEEKER_SIGNS change. The ONLY thing that
changed yaw behavior since 07-02 is A22's own override value. The A9 gyro-negation is upstream of the
estimate and consistent in both runs; mixer coupling is out (roll_cmd magnitude identical on
opposed vs agree ticks, 0.17/0.17). **The sim did not change; A22 was a mis-conclusion.**

### Fix
`deploy_profile.py` `vq2_case_c().controller_overrides["body_rate_sign"] = (1,1,-1)` (the VQ1
default). VQ1/case-A untouched (seeker default already -1, no vq1 controller overrides).

### Offline check (measured plant, run 20260704_032626)
Least-squares plant gain sent→realized = **-0.83** (negative ⇒ the wire inverts). Replaying the
logged geometry with the corrected sign through the measured inverting plant: realized yaw goes
**TOWARD the gate 132/5** (was 23/141 with +1). The yaw loop flips positive→negative feedback.

---

## 2. Item 1 — pointing gate on the forward drive ("point before you push")

`_compose_image_servo_accel`: after the cos^pow `fwd_scale`, multiply `a_fwd` by a SMOOTHSTEP gate
`g_point` that is 1.0 for `|az| <= fwd_point_gate_full_az_rad` and ramps to 0.0 by
`|az| >= fwd_point_gate_az_rad`. Purely a further REDUCTION of forward (never an increase). So on the
pass→turn the drone spends ~no forward energy until re-pointed and turns in place instead of arcing
wide. `fwd_point_gate_az_rad = None` (default) ⇒ no gate (byte-identical). vq2_case_c ready value:
0.35 rad (ramp 0.05→0.35). This is the operator's dominant "slow is smooth" ask applied at the
turn.

---

## 3. Item 3 — physical-plane turn commit + hold-until-pointed (operator ruling: physical plane)

Two coupled knobs in the pass state machine (`_update_pass_state`, `_pass_acquired_next`,
`_in_pass_dead_reckon`):

1. **`pass_wire_requires_near`**: a wire `index_advanced` while the tracked gate is still WELL AHEAD
   (`range > pass_arm_range_m`, i.e. NOT at the plane — the wire leads ~9 m) no longer commits; it
   records `_pass_wire_pending`. The PHYSICAL commit (degenerate-close / lost-after-arm) then fires
   and HONORS the pending intent (upgrades to the fast wire window). A wire advance already within
   `pass_arm_range_m` still commits immediately. This enforces the A33-spec H-1 "defer the wire while
   the gate is in front" that the shipped code never actually enforced.
2. **`pass_turn_hold_until_pointed`**: with a blind turn target latched (`pass_turn_through`), HOLD
   the dead-reckon coast (and DEFER acquire-next) until the commanded heading is within
   `pass_turn_point_tol_rad` (~10°) of the turn target OR the bounded `pass_turn_coast_s` (1.5 s)
   elapses — instead of reverting on the fixed 0.3 s `pass_coast_s`, far short of a ~95° slew. So the
   turn COMPLETES rather than orbiting.

Both default OFF ⇒ index_advanced commits unconditionally and the fixed window governs
(byte-identical). vq2_case_c: both ON (first re-fly).

---

## 4. Item 4 — lateral-first "switch lanes" (operator insight; image-based visual servoing)

A quad translates holonomically, so cross-track error should be closed by BANKING sideways onto the
gate's approach line, not only yaw-to-face + forward (which gets carried past). The A30 image servo
already has `a_lat = clip(k_az·dead(az), ±image_lat_cap)` along `e_right`, but (a) the cap 1.5 limits
it to ~8.7° and (b) it is summed with a_fwd then norm-capped at total_accel_cap, so a large forward
CLIPS the lateral. `use_lateral_first_budget` ON allocates the accel budget LATERAL-FIRST when
off-axis: |a_lat| takes priority within total_accel_cap and forward gets only the remaining radial
budget `sqrt(cap² - a_lat²)`. Pure proportional-on-bearing, NO derivative (the A29 LOS-rate
differentiation of a noisy bearing went unstable — NOT reintroduced), bounded by the same
total_accel_cap, and the A31 `image_lat_slew` rate-limit still caps the per-tick lateral change.
Pairs with a raised `image_lat_cap_mps2` (vq2_case_c ready: 3.0 ≈ 17° bank). Default OFF ⇒ the A30
sum-then-norm-cap composition (byte-identical).

---

## 5. Item 2 — SUPERSEDED

The task's Item 2 (fix the "reacquire yaw-inversion" where `chase_dpsi` went negative while az_err
stayed positive during a low-bearing-weight transient) was a MISREAD: `chase_dpsi_rad` is the
orbit-integrator (logging-only), not the yaw command, and the "yaws left" was the GLOBAL yaw
inversion (Item 0), constant all flight (114/0), not a 5-tick transient. Once Item 0 lands the
symptom dissolves. **No separate fix built.**

---

## 6. Build result (2026-07-03, HEAD 458223d, branch vq2-gate2-turn-dive)

**Files changed:**
* `src/racer/deploy_profile.py` — Item 0 `body_rate_sign -> (1,1,-1)`; A36 seeker overrides (Item 3
  ON, Items 1&4 wired OFF-but-tuned, individually toggleable).
* `src/racer/gate_seeker.py` — config fields (Items 1/3/4) + `_pass_wire_pending` state; Item 1
  pointing gate + Item 4 lateral-first budget in `_compose_image_servo_accel`; Item 3
  `pass_wire_requires_near` gating in `_update_pass_state` + `_turn_hold_active` helper used by
  `_pass_acquired_next` / `_in_pass_dead_reckon`; cleanup in `_end_pass` / `reset`.
* `tests/test_vq2_yaw_actuation_sign.py` — REWRITTEN to the corrected inverting-wire convention
  (gate-right ⇒ wire-negative ⇒ turn right; closed-loop plant = -2.1× wire; counterfactual pin now
  flags +1 as divergent).
* `tests/test_vq2_a36_turn_convergence.py` — NEW: 13 focused tests + regression pins for Items 1/3/4.

**Test suite:** 1743 passed, 73 skipped, **11 failed — all PRE-EXISTING** (verified failing on the
clean baseline tree: `test_ahrs_bench::test_R_quat_roundtrip`, `test_diagnose_session` [missing
bundles], `test_sysid_camera_geometry` + 8× `test_sysid_plant_parity_aggressive`, all bit-exactness
at ~1e-15). Zero new failures. `EXPECTED_KEYS = 46` (unchanged — A36 adds no nav-log field; the
fixes are observable via existing `body_rate` / `az_err_rad` / `fwd_scale` / `alat_mps2` /
`track_range_m` / `pass_*` fields).

**Regression pins (all pass):** each new flag default-off reproduces the pre-A36 path byte-for-byte
(pointing gate None, lateral-first False, pass_wire_requires_near False, pass_turn_hold False); the
A22 +1 sign is still constructible for the counterfactual divergence pin.

**Offline check (run 20260704_032626):** corrected sign through the measured inverting plant
(gain -0.83) ⇒ realized yaw TOWARD gate 132/5 (was 23/141). Turn now commits at the physical plane
(unit-pinned) not the wire.

**NOT flown** — first re-fly is Item 0 + Item 3, Items 1&4 OFF, per the flight plan.
