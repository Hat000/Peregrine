# VQ2 Gate-1 Turn Readiness Audit — 2026-07-05

**Scope:** design/audit only (no src/rl/tests edits). CPU-only. HEAD `f1c5db9`, branch `vq2-gate2-turn-dive`.
**Frontier under audit:** the gate-1 TURN. Operator eyes: direction now correct on both axes, but the
drone turns TOO FAST + UNDER-ROLLED → overshoots → gate-1 skews to a parallelogram → vision starves.
Operator prescription: SLOWER approach, EARLIER turn trigger, MORE ROLL.

Data: nav_estimate.jsonl / commands.jsonl for the three eyes-relevant runs
(`20260704_183049` STALLED@g1, `20260704_231155` TIMEOUT@g1, `20260705_012253` TIMEOUT@g1 — the most
recent). `20260704_182621` is a dead capture (command_records=0, no nav_estimate) — excluded.
All three stall at **gate index 1** — the turn is exactly the wall.

---

## 1. CURRENT TURN PATH ON HEAD (state machine + file:line)

Approach → pass → coast/slew → reacquire, as it actually runs in `vq2_case_c`:

1. **Pursuit toward gate 0** — `_visual_pursuit_command` (`gate_seeker.py:2363`). Image-servo lateral
   `a_lat=clip(k_az·az·bearing_w, ±image_lat_cap)` (`:2194`), lateral-first budget
   (`use_lateral_first_budget`, `:2229`) so forward takes only `sqrt(cap²−a_lat²)`. Yaw setpoint slewed
   by `_slew_heading` (`:2665`) at `pursuit_yaw_slew_rps=0.9`, close-range-tapered (Fix B, floor 0.35).

2. **Pass ARM** — `_update_pass_state` (`:1555`): arms once tracked range ≤ `pass_arm_range_m=4.5`
   (`:1575`). Confirmed in data: first pass fires at rng ≈ 4.4–4.6 m across all three runs.

3. **Pass COMMIT** — three triggers (`:1586-1605`):
   - `degenerate_close` (rng ≤ `pass_degenerate_range_m=3.0`), or `lost_after_arm`, or `wire_commits`.
   - **Physical-plane gating IS live:** `pass_wire_requires_near=True` (`:1596`) → a wire index-advance
     while the gate is still far ahead (rng > 4.5) only sets `_pass_wire_pending`; the physical commit
     (degenerate/lost) fires the turn. **This is A35 Fix1 — it landed** (was DEFERRED at A35). Data
     confirms first `pass_wire` at rng 4.4 m, i.e. AT the plane, not the ~9 m-early wire.

4. **`_begin_pass`** (`:1607`) freezes `_pass_heading=_last_yaw` (`:1616`). With `pass_turn_refine=True`
   it leaves `_pass_turn_yaw=None` + `_pass_turn_pending=True` (`:1637-1638`) — coast straight, aim later.

5. **Coast/slew** — `_pass_coast_command` (`:1789`): each tick `_maybe_refine_turn_target` (`:1748`)
   re-aims `_pass_turn_yaw` at a valid downrange gate-2 pose (bounded ±`pass_blind_turn_cap_rad` from the
   pass heading; needs rng > 3.0, ≤ 35 m, not-the-passed-gate, bearing_w ≥ 0.3). `_slew_heading` slews
   the coast yaw toward the target at 0.9 rad/s (`:1812`). Vertical is a **flat alt-hold** here:
   `_feedforward_command(..., vz_cmd=0.0)` and **no `gate_pd_scale`** passed (`:1817`).

6. **Hold-until-pointed** — `_turn_hold_active` (`:1690`) keeps acquire-next deferred and the coast alive
   until `|_last_yaw − _pass_turn_yaw| ≤ pass_turn_point_tol_rad=0.17`, bounded by
   `pass_turn_coast_s=1.5` (`:1697-1713`). Gates `_pass_acquired_next` (`:1726`) and `_in_pass_dead_reckon`
   (`:1743`).

7. **Reacquire** — once pointed (or past the coast window) a downrange pose ends the pass
   (`_end_pass`, `:1658`), forward re-ramps (`reramp_forward_after_pass`), pursuit resumes on gate 1.

**Trigger-timing verdict (operator "earlier"):** the turn already commits AT the physical plane
(rng ≈ 4.5 m). The knob that moves it earlier is **`pass_arm_range_m`** (currently 4.5) — raising it arms
+ enables the degenerate/lost commit sooner. But the true late-ness is NOT the commit tick; it is that the
**hold releases on a stale/too-small target** (below), handing a barely-begun turn to a rate-limited chase.

---

## 2. SELF-CANCEL BUG VERDICT (A35 pass_wire_coast=0.0 same-tick revert)

**FIXED / superseded — not live.** Structural fix = `pass_turn_hold_until_pointed`. Evidence: on HEAD the
first `pass_wire` persists **4 consecutive pass ticks (~0.21 s)** in both 183049 and 012253 before the
regime changes — not a 1-tick self-cancel. `pass_wire_coast_s=0.0` no longer causes an instant revert
because `_turn_hold_active` / `_in_pass_dead_reckon` keep the coast alive independent of the 0.0 window.

**BUT a NEW, subtler failure replaces it (Latent Bug #1 below):** the hold releases after only ~0.2 s
because `_pass_turn_yaw` refined to a *small, close, sometimes wrong-signed* target (~+0.35 rad), the
"pointed" test `|yaw−0.35|<0.17` is satisfied trivially, and the pass ends pointing at a stale residual —
NOT at the real gate 1, which only appears ~0.9 s later at rng ≈ 16 m and yaw_des climbing to +0.6…+1.7.
So the pass-turn "completes" against the wrong target and dumps into a rate-limited pursuit tail-chase.

---

## 3. LATENT BUG HUNT (ranked by flight-risk)

### Bug #1 (HIGHEST) — hold-until-pointed releases on a stale/too-small refine target → rate-limited tail-chase
`_maybe_refine_turn_target` (`:1786`) latch-FOLLOWS the *first qualifying* gate-2 pose. Near the plane the
first qualifying pose is a **close residual** (rng just over 3.0 m, small world-bearing, and in 012253
even *opposite-signed*: yaw_des flips −0.33 at tick 54). `pass_turn_yaw` latches ≈ +0.35 rad; the drone
already points near there, so `_turn_hold_active` returns False at ~0.2 s and the pass ends
(`_pass_acquired_next`). The REAL gate 1 (rng 16 m, world-bearing +0.6…+1.7) is acquired 0.9 s later, in
**pursuit**, where the setpoint slew is rate-capped at 0.9 rad/s. `yaw_des` then runs away from `yaw`
(012253 ticks 65–88: yaw_des 0.58→1.67 while yaw only −0.29→−1.54; yaw-error grows monotonically to
**3.0 rad**, `wz` pinned at **0.90** the whole time). This is the "turns too fast then can't keep up"
signature — but it is really a *too-short hold* + a *rate-limited chase*, not raw over-speed.
**Fix direction:** the hold should not release until the target is BOTH pointed AND at a plausible
next-gate range (not a <5 m residual), and the refine should prefer the FARTHEST qualifying pose or
require a min range, so the latched target is the real gate not a close residual.

### Bug #2 (HIGH) — omega norm-clip STARVES roll during the turn (the brief's suspected contest, CONFIRMED)
`controller.py:815-818`: one rotvec combines roll+pitch+yaw error, then `_clip_norm(omega,
max_body_rate_rps=4.0)` scales **all three axes proportionally**. Per-axis caps (yaw→0.9 `:2548`,
roll→1.5, pitch→1.5) apply AFTER. Measured: in **55–69 % of pursuit ticks** the yaw term ALONE
(`kp_att·|yaw_err| = 4.0·|yaw_err|`) exceeds 4.0 (76–83 % of ticks have |yaw_err| > 0.5 rad). When the
yaw term saturates the norm, any roll demand is scaled by `4.0/‖omega‖` **before** the roll cap — so
`_cap_roll_rate(1.5)` binds on **0 %** of pursuit ticks (roll is crushed upstream, not cap-limited).
While yaw is pinned at 0.9, roll |wx| median is only 0.25–0.30 and is **< 0.5 on 76–91 %** of those
ticks. This is exactly "banked barely any roll while yawing hard." Cutting yaw already helped (0.9 caps),
but the yaw-error can still be 3 rad, so `kp_att·err` still saturates the norm and eats roll.
**Fix direction:** either raise `max_body_rate_rps` for the seeker (give the norm headroom so roll
survives), or decouple yaw from the roll/pitch clip budget, or (cheapest) keep the yaw SETPOINT from
running 3 rad ahead (Bug #1 fix shrinks yaw_err, which un-starves roll for free — same coupling the yaw
fix already exploited once).

### Bug #3 (MEDIUM) — parallelogram-skew → bearing_w collapse → roll cut (positive feedback), during reacquire
`_compose_image_servo_accel` (`:2192`): `az_eff *= bearing_w`. As the overshoot skews gate-1 to a
parallelogram, `theta_g` grows (corr(|theta_g|,|az_err|)=+0.27 in pursuit) and the Cauchy bearing weight
collapses: **bearing_w < 0.1 on 20 % of reacquire-pursuit ticks** (183049 & 012253), p10 ≈ 0.04. A
crushed weight zeroes the lateral (roll) demand exactly when the drone most needs to bank onto the line —
the skew→low-bw→less-roll→more-skew loop the TURN PACKAGE comment already named. `track_max_loww_ticks=4`
bounds the *drop*, but the per-tick weight crush still cuts roll on the surviving frames.
**Not a crash bug**, but it caps the achievable roll independent of Bug #2.

### Non-bugs / cleared hypotheses (checked, benign):
- **Vertical taper going garbage mid-turn:** NO. `_gate_pd_scale` (`:2699`) only runs on the PURSUIT path
  and is passed to the controller only there; `_pass_coast_command` passes NO `gate_pd_scale` and
  `vz_cmd=0.0`, so during the coast the vertical is a flat alt-hold — the terminal taper is simply
  inert, not fed stale/garbage z_off. When pursuit resumes it recomputes from live range. No stale leak.
- **Stale `_pass_*` across gate-index changes:** `_end_pass` (`:1658`) clears `_pass_turn_yaw`,
  `_pass_turn_pending`, `_pass_wire_pending`, `_pass_heading`, `_pass_prev_*`, re-arms bookkeeping.
  `_begin_pass` re-freezes fresh. No cross-gate latch leak found.
- **NaN/degenerate-pose guards:** `_unit` has a fallback (`:2392`); refine gates on rng bands and
  bearing_w; `_slew_heading` and `_gate_pd_scale` guard `hi>lo` and None-range. No div-by-zero surface
  found on the turn path.
- **"Yaw physically inverted" (would contradict the SOLVED yaw):** the estimator-frame yaw-realization
  gain is a consistent **−0.8** (corr −0.88) across all three runs — but this is the **benign estimator
  vs sim-actuation sign convention** (the whole point of body_rate_sign/yaw_steer). Physical camera-truth
  test: mean d|az_err| is NEGATIVE (drone nets toward centering), so yaw is NOT physically inverted.
  **The brief's "yaw is SOLVED" holds.** The −0.8 log-frame gain is expected; do not chase it.

---

## 4. TUNING PROPOSAL (prefer existing flags; file:line, current→proposed, effect, risk)

All in `src/racer/deploy_profile.py::vq2_case_c` ("dp:") unless noted. Line numbers exact on the
E1-implemented tree (2026-07-05). "Slower / Earlier / More-roll" grouped.

| # | Lever (file:line) | Current | Proposed | Predicted effect | Risk |
|---|---|---|---|---|---|
| **SLOWER** |
| S1 | `forward_accel_mps2` (dp:502) | 0.35 | 0.25 | Entry ~3.2→~2.7 m/s at plane; less carried tangential to whip. Data: approach into gate-0 median 3.1–3.9 m/s vs ~2.5 comfortable at realized lateral. | Low. Could under-drive; egress-slide residual (~2.4 m/s) still floors it, so limited gain alone — pair with S2. |
| S2 | `fwd_point_gate_az_rad` (dp:494) | 0.25 | 0.18 | Cuts forward to ~0 beyond ~10° az (was 14°), so the drone TURNS before translating past the gate. | Low. Too tight → creeps when it should push; 0.18 keeps centered pace. |
| **EARLIER** |
| E1 | **IMPLEMENTED 2026-07-05** — `pass_refine_prefer_far` flag: `gate_seeker.py:839` (field), refine re-aim bound + acquire-next bound (see §2a below); **staged OFF at dp:462** | False (staged) | **True at GO** | Both the refine re-aim AND acquire-next require rng > `pass_arm_range_m` (a plausible NEXT gate): the 3.0–4.5 m residual can neither aim the turn nor end the pass; the turn stays pending until the real gate is seen. Kills the 012253 stale-target early release + the tick-54 residual hand-off. | Low (was Med as raw code change): flag-gated, default-off byte-identical, 10 new tests pin ON/OFF paths. Residual: if gate 1 never shows > 4.5 m during the 1.5 s coast, fallback = straight-coast timeout (unchanged). |
| E2 | `pass_arm_range_m` (dp:551) | 4.5 | 5.5 | Arms + enables the physical commit ~1 m sooner → turn begins earlier. NOTE: with E1 ON this also raises the plausible-next-gate bound to 5.5 (coherent — the residual band and the arm band are the same physical territory). | Low-med. Too early arms on a still-usable pose; keep ≤ ~6 (usable PnP). |
| E3 | `pass_turn_point_tol_rad` (dp:440) | 0.17 (~10°) | 0.10 (~6°) | Hold releases only when more precisely pointed → turn completes further before pursuit handoff. | Low. Interacts with E1; tighten only after E1 fixes the *target*, else it just holds longer on the wrong target. |
| **MORE ROLL** |
| R1 | `max_body_rate_rps` (seeker factory default `gate_seeker.py:149`; GO change = **add** `"max_body_rate_rps": 6.0` to `controller_overrides`, dp:675, so VQ1 stays untouched) | 4.0 | 6.0 | Gives the omega norm-clip headroom so a saturating yaw term stops scaling roll down. Directly attacks Bug #2 (clip bites 55–69 % of ticks). Test-verified: raising 4→6 frees roll by exactly 1.5× while the clip binds (`test_vq2_e1_prefer_far.py::test_b_*`). | Med. Raises the safety ceiling on ALL axes; roll/pitch already per-axis-capped at 1.5, yaw at 0.9, so realized rates stay bounded — the clip was the ONLY thing coupling them. |
| R2 | `image_lat_cap_mps2` (dp:515) + `total_accel_cap_mps2` (dp:540) | 4.0 / 4.0 | 4.5 / 4.5 | More lateral (roll) authority. Data: alat railed the 4.0 cap only 7 % (012253), 0 % (others) — cap is **not** the main binder, so this is secondary to R1/E1. | Low. Modest bank increase (~22°→~25°). |
| R3 | E1 — *indirect* | — | — | Shrinks yaw_err → `kp_att·err` stops saturating the norm-clip → roll un-starved "for free" (same coupling the yaw fix exploited). | — (covered by E1). |
| **Optional** |
| O1 | `pass_turn_coast_s` (dp:439) | 1.5 | 2.0 | If E1 holds the turn to the real (far) gate, the slew at 0.9 rad/s over ~90° needs ~1.7 s — 1.5 s may time out mid-turn. Raise to give the completed turn room. | Low. Bounded; only extends the blind coast if a far target is latched. |

### §2a E1 implementation (landed 2026-07-05, UNCOMMITTED, flag staged OFF)
- `src/racer/gate_seeker.py:839` — `pass_refine_prefer_far: bool = False` (field + evidence comment).
- `src/racer/gate_seeker.py` `_maybe_refine_turn_target` — the downrange bound becomes
  `pass_arm_range_m` when the flag is ON (else the legacy `pass_degenerate_range_m`).
- `src/racer/gate_seeker.py` `_pass_acquired_next` — same bound swap on the pass-ending re-acquire.
- `src/racer/deploy_profile.py:462` — `"pass_refine_prefer_far": False` staged in vq2_case_c.
- `tests/test_vq2_e1_prefer_far.py` — 10 tests (see §5).

### GO-time mechanical flips (exact, one value each)
- **E1-enable:** dp:462 `"pass_refine_prefer_far": False` → `True`
- **S1:** dp:502 `"forward_accel_mps2": 0.35` → `0.25`
- **E2:** dp:551 `"pass_arm_range_m": 4.5` → `5.5`
- **R1:** dp:675 `controller_overrides={...}` → add key `"max_body_rate_rps": 6.0,`
- (secondary: S2 dp:494 `0.25→0.18`; E3 dp:440 `0.17→0.10`; R2 dp:515/dp:540 `4.0→4.5`; O1 dp:439 `1.5→2.0`)

**Recommended first fly:** E1-enable (dp:462) + S1 + E2 + R1 — the root fix is now flag-staged, so the
whole package is four mechanical profile edits, each one-value-revertible.

---

## 5. TEST COVERAGE

**`tests/test_vq2_a36_turn_convergence.py` already pins:** Item-1 pointing-gate (kills fwd off-axis,
identical centered, monotone, off-path byte-identity); Item-4 lateral-first (prioritizes lateral, fwd =
sqrt(cap²−a_lat²), off-path identity); Item-3 physical-plane wire gating (defers far, commits at plane,
off-path unconditional); hold-until-pointed (holds then releases, bounded by coast_s, off = no hold);
Fix-B yaw taper (reduces close-range authority, off = no taper); refine-to-real-gate (pending-not-blind,
re-aims at downrange, ignores passed-gate/low-bw/degenerate, bounded to cap, off = blind); yaw-steer
selector A/B/off; coordinated-turn arms-at-vision-floor + orbit-guard-demoted + fwd-not-ballooned; and
`test_profile_flight3_full_turn_package` pins the shipped profile (note: its *docstring* still says
`(1,1,-1)` but the body correctly asserts `yaw_steer_mode=="off"` + `(1,1,1)` — stale comment only,
test passes).

**NEW (2026-07-05): `tests/test_vq2_e1_prefer_far.py` — 10 tests, all passing (the three recommended
pins, landed with the E1 implementation):**
- **(a) hold vs residual:** `test_a_hold_does_not_release_on_close_residual` (a 4.3 m residual cannot
  re-aim, the turn stays PENDING, the hold defers acquire-next, the pass does not end);
  `test_a_hold_completes_against_the_real_far_gate` (the ~16 m gate re-aims, the hold keeps holding
  until pointed at THAT target, then the far pose ends the pass);
  `test_a_acquire_next_range_bound_without_the_hold` (the tick-54 hand-off seam in isolation).
- **(b) omega-clip roll-survival (Bug #2 / R1, controller-level, flag-independent):**
  `test_b_norm_clip_binds_during_the_turn_and_starves_roll` (2.5 rad yaw error → |omega| == the cap;
  roll crushed vs its unstarved no-yaw-contest value);
  `test_b_raising_the_ceiling_frees_roll_proportionally` (max_body_rate_rps 4.0 vs 6.0: clip still
  binds, every axis — roll included — freed by exactly 1.5×; documents the R1 flip).
- **(c) far-gate selection:** `test_c_residual_first_far_gate_still_wins` (residual leaves no residue
  on the latched target) + `test_c_far_first_residual_cannot_drag_the_target_back`.
- **(d) OFF byte-identity:** `test_d_default_off_and_profile_staged_off` (field default False +
  vq2_case_c stages False) + `test_d_flag_off_residual_latches_exactly_as_before` (the 012253 legacy
  latch) + `test_d_flag_off_acquire_next_accepts_the_residual` (the legacy tick-54 hand-off; the
  degenerate band stays rejected on both paths).

**REMAINING gap (deferred):** a full scripted `command_visual` end-to-end kinematic replay
(approach→pass→coast→reacquire with the gate-1 geometry) asserting post-handoff yaw-error stays
< ~1 rad — the (a)-tests pin the state machine at the unit level; the integrated replay would also
catch regressions in the regime plumbing around it.

---

## OPEN QUESTIONS FOR THE OPERATOR
1. **Bug #1 is the root:** the turn hold releases pointing at a *close residual* (~0.35 rad), not gate 1
   (~1.5 rad, 16 m). Confirm from eyes whether the drone visibly "finishes a small turn then slowly
   swings further" — that would corroborate the stale-target early-release read vs a raw over-speed read.
2. **R1 (`max_body_rate_rps` 4.0→6.0)** is the cleanest fix for roll-starvation but raises the seeker's
   attitude-rate safety ceiling. Per-axis caps (roll/pitch 1.5, yaw 0.9) still bound realized rates —
   OK to raise the norm ceiling, or prefer the E1/yaw-error-shrink route that un-starves roll indirectly?
3. ~~Priority order for the next fly: E1 (code, root) first, or a flag-only S1+E2+R1 fly?~~ RESOLVED
   2026-07-05: E1 is implemented flag-gated and staged OFF (dp:462), so the whole package —
   E1-enable + S1 + E2 + R1 — is four mechanical one-value profile edits at GO (see the GO-time
   flips list in §4).
4. Is the ~2.4 m/s egress/spawn-tilt slide (flagged earlier as a follow-up, contributes most of the
   "entry too fast") in scope for this turn work, or a separate vertical/egress fix?
