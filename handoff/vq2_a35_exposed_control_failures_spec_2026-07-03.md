# VQ2 A35 — A34-Exposed Control Failures: Diagnosis + Fix (2026-07-03)

A34 (commit `80902d4`) **worked**: 0 ticks locked track_range >35 m across 3 flights, max range 31 m
— the garbage-lock that blinded A33 is GONE, the vision feed is clean. With a clean feed the REAL
control failures are now exposed. This diagnoses the three the operator saw and designs the fixes.
**Review BEFORE building.**

**Evidence base:** FLIGHT 1 ONLY (`data/runs/20260704_024434_rl_s1_f1`, clean cold-launch, full 15 s
/ 349 ticks). Flights 2 & 3 (`_024629`, `_024747`) are **NON-EVIDENCE** — 75 ticks / ~3 s each, both
after a vq2ctl restart, renders truncated before the startup dip even landed (operator). They are
identical stubs (both reach gate 0, both 10.8 m max range, same regime counts) = a restart-degraded
spawn/run artifact, NOT a control signal. Flagged for a separate sim-ops look; **no control
conclusion is drawn from them.**

Operator eyes (flight 1, ground truth): *"startup dip HITS THE GROUND; continues up and forward
with NO yaw change; climbs and HITS THE CEILING."*

---

## 0. Executive summary — three failures, two roots

| # | Failure (operator) | Proven mechanism | Root |
|---|---|---|---|
| 1 | No turn at all | The wire index leads the physical gate plane by ~9 m; H-1c's turn commits on the wire but H-1b's 0.0-coast + acquire-next INSTANTLY re-locks the still-9m-ahead gate-1 and cancels the turn. The real pass (rng 3.9 m) commits as vis, latches the turn target (−1.65 rad) but only coasts 5 ticks (0.3 s) then reverts. | **Pass/turn sequencing** (independent) |
| 2 | Climbs into ceiling; V-2 floor never used | As the drone rises above the gate the +20° camera loses it → bearing_w collapses to 0.04 → the A32 soft weight crushes the honest +4.7 m z_off innovation to weight 0.02 and flags it rejected → the estimator ignores vision and PROPAGATES z_off to the WRONG SIGN (−3.9 m = "gate above, climb more") on a drifting IMU vz → PD commands hover/up, never the down-authority V-2 opened. | **Vertical estimator under FOV loss** |
| 3 | Startup dip hits ground | Drone spawns ~−18° nose-down; the cold AHRS takes ~0.5 s to level the estimate; during settle/anchor the hold thrust band [0.90, 1.12]×hover lets collective sit at 0.239 (below hover 0.2656) — 50% of settle+anchor ticks are sub-hover → net sink → ground tap BEFORE egress (whose thrust floor would have held it). | **Cold-start altitude hold** |

**Shared-root assessment (the coordinator's read, refined):** (2) and (3) are BOTH
vertical-altitude-hold failures but do NOT share a code root — (3) is the *settle/anchor hold band*
(cold-start, pre-estimator), (2) is the *A28/A32 estimator's vision-distrust divergence* (mid-flight,
gate out of FOV). They are fixed in different places. (1) is fully independent (the pass/turn state
machine). So: **three fixes, largely independent.** Recommended sequencing in §4.

---

## 1. Failure 1 — the turn never fires (proven)

`gate_index` advanced 0→1 at **cmd 80, t 3.51 s, while the tracked gate-1 was still 9.3 m ahead,
dead-center** (az≈0, bw 0.99). `pass_wire` logged False and the regime stayed `pursuit` — the wire
pass committed and INSTANTLY ended:

* H-1b set `pass_wire_coast_s = 0.0`. On the wire advance, `_begin_pass(wire=True)` fires, but the
  very next check `_pass_acquired_next` computes `elapsed (0) ≥ _effective_pass_coast_s (0.0)` →
  True, and the still-visible gate at 9.3 m > `pass_degenerate_range_m` (2.5) → returns True →
  `_end_pass` SAME TICK → `pass_wire` reset to False, regime logged `pursuit`, track keeps gate-1
  (rng continues 9.3 → 8.8, same dead-center gate). **The wire-triggered turn evaporated because the
  drone was 9 m short of the physical plane.**
* The A34 offline work already noted the wire leads the plane (~3 m in prior runs); **this flight it
  led by ~9 m** — even wider. So H-1c's "turn on the wire" is structurally premature.
* The REAL physical pass finally commits at **cmd 151 (t 6.61, rng 3.9 m)** as `pass_vis` (armed +
  degenerate/lost), latches `pass_turn_yaw_rad = −1.65` (turn LEFT ~95°), and the turn-through slew
  starts — but `pass_coast_s = 0.3` (~9 ticks) expires and it reverts to `pursuit` at cmd 156 after
  only 5 pass_vis ticks, re-acquiring a 17.9 m gate. **Total yaw movement all flight: 37° (−32 to
  +5) — the turn never completed.**

### Root
Two coupled bugs:
1. **The wire pass fires ~9 m early and self-cancels** (H-1b 0.0-coast + acquire-next re-locks the
   still-ahead gate). H-1a's geometric exclusion did NOT save it because the re-locked candidate IS
   the near gate dead ahead (not excluded — it's the gate we're pursuing, not one we passed).
2. **When the real pass does commit (vis, rng 3.9), the coast is too short (0.3 s) for the ~95°
   turn** — the slew needs ~1.0-1.3 s at 1.5 rad/s, but the coast reverts in 0.3 s.

### Fix 1 — arm the turn on RANGE, not on the premature wire; hold the turn coast long enough
* **Do NOT commit the pass on `index_advanced` while the tracked gate is still well ahead.** Gate the
  wire commit on range: only let the wire advance commit a pass when `track_range_m <= a
  near-plane threshold` (e.g. `pass_arm_range_m` = 3.0, i.e. the drone is actually AT the plane), OR
  treat the wire advance as *arming intent* that fires the turn only once the physical pass
  (degenerate/lost-after-arm) confirms. New field `pass_wire_requires_near: bool` (default False =
  legacy; vq2_case_c True): `index_advanced` alone no longer commits — it sets `_pass_wire_pending`,
  and the real commit (degenerate/lost) then uses the FAST turn path. This is the A33-spec H-1
  "defer the wire while the gate is in front" idea, which the shipped code did NOT actually enforce
  (the commit still fired on `index_advanced` unconditionally).
* **Extend the turn coast so the ~95° slew completes.** With `pass_turn_through` on, the coast must
  last at least the blind-turn duration: raise the effective pass coast for a turn-through commit to
  ~1.3 s (a new `pass_turn_coast_s`, or make `_in_pass_dead_reckon` hold while `|yaw −
  pass_turn_yaw| > tol` and the acquire hasn't cleanly re-locked the NEXT gate). The turn should end
  when the drone is POINTED (yaw within ~10° of target) or a fresh next-gate pose clears, whichever
  first — not on a fixed 0.3 s clock.

> **[OPERATOR DECISION — Fengyou, 2026-07-03: turn-commit policy = PHYSICAL PLANE.** "there was
> absolutely no turn this flight, but physical plane should be right." **Fix 1 is DEFERRED (build
> after Fixes 2 & 3 re-fly).** When built: commit the turn at the physical gate plane (rng ≈ 3 m /
> the arm range), NOT the wire (which leads ~9 m and self-cancels), AND hold the turn coast until
> POINTED (yaw-within-tol, ~1.3 s slew) so the turn completes instead of reverting at 0.3 s. The
> `pass_wire_requires_near` + `pass_turn_coast_s` (or yaw-within-tol) design above is the shape.
> DO NOT build Fix 1 in the A35 build — Fixes 2 & 3 only, re-flown to isolate their effect first.]**

---

## 2. Failure 2 — vertical estimator diverges to the WRONG SIGN under FOV loss (proven, most damaging)

The most damaging failure and the reason V-2's open floor was moot. Trace (cmd 195-307):

* The drone climbs above the gate. `offset_z_world` (RAW vision) correctly reads +3.5 → **+4.7 m**
  (gate is 4.7 m BELOW the drone → the PD must DESCEND HARD).
* But the +20°-up camera loses the now-below gate → `bearing_w` collapses **0.11 → 0.04**. The A32
  soft weight `zoff_w = huber × bearing_w` crushes to **0.02** — the honest, sustained **+2.99 to
  +4.7 m** z_off innovation is applied at ~2% and flagged **`accepted=False`** (nu > miss threshold).
* With vision throttled to ~nothing, the A28 predict step propagates `z_off -= vz_est·dt`. `vz_est`
  drifts to **+1.9 m/s** (the washout, unfed by vision, integrates a wrong-signed IMU residual), so
  `z_off_est` runs **−0.15 → −3.9 m** — i.e. the PD input says **"gate is 3.9 m ABOVE, climb MORE"**
  while vision screams "+4.7 m below, descend." The PD holds thrust ~0.24-0.29, **never demands the
  down-authority V-2 opened** (thrust min 0.229, floor 0.05 never approached).

### Root
**When the gate leaves the FOV during a real vertical excursion, the A32 bearing-weight
down-weighting throttles the honest LARGE offset to ~0, and the estimator then diverges to the
wrong sign on a drifting IMU propagate.** This is the exact failure mode the A32 spec's "weight
don't discard" principle protects against for *small noisy* offsets — but a **+4.7 m offset
sustained for 100 ticks is not noise; it is the single most important signal in the flight**, and
the weighting discards it precisely when it matters most. `zoff_reseed_min_w` (A33 V-1) makes this
WORSE here: it (correctly) blocks a reseed onto low-weight frames, but that means the wrong-signed
divergence is never re-locked to the honest +4.7 m either — the state just drifts.

### Fix 2 — a large, persistent offset must OVERRIDE bearing distrust (magnitude-gated trust floor)
The principle: bearing-weight distrust is for *plausibly-noisy* poses. A z_off innovation that is
**both large AND persistent** (many consecutive frames all saying "you are several metres off, same
sign") is a real excursion the loop MUST act on, regardless of bearing weight. Concrete design
(all vq2_case_c-gated, default-off byte-identical):

* **(a) Magnitude trust floor on the z_off correction:** when `|innov|` exceeds a "this is real, not
  noise" threshold (e.g. `zoff_big_innov_m` ≈ 2.5 m) AND the sign is consistent across the last N
  frames, apply a MINIMUM correction weight (e.g. `zoff_big_innov_min_w` ≈ 0.5) that FLOORS `zoff_w`
  from below — the bearing weight can no longer crush a large honest offset to 0.02. A +4.7 m
  offset then corrects the state toward +4.7, killing the wrong-sign divergence.
* **(b) Persistence reseed on a large consistent offset:** if `|innov| > zoff_big_innov_m` with a
  consistent sign for `reseed_after` frames, RE-LOCK `z_off = z_meas` even from low bearing weight —
  a sustained "you are 4.7 m off" IS a retarget-grade signal (this is the sign-flip case
  `zoff_reseed_min_w` currently blocks; the magnitude gate is the exception that lets a genuinely
  large offset through). Bounds: only when the sign is stable (not oscillating), so a noisy jumpy
  offset still can't reseed.
* **(c) Freeze the propagate when vision is distrusted AND the offset is large:** while
  `bearing_w < floor` and `|last z_meas| > zoff_big_innov_m`, do NOT propagate `z_off` on the
  unfed vz (that is what drove −3.9). Hold the last honest z_meas instead of drifting — a large
  known offset held is strictly better than the same offset drifted to the opposite sign.

This is the operator's principle applied correctly at the extreme: **weight-don't-discard for small
noise, but a large-and-persistent offset is trusted signal, not an outlier to weight down.** V-2's
open floor then becomes live — once z_off_est reads the true +4.7, the PD commands the −6 to −8
m/s² descent the floor now permits.

**Alternative / complementary (prevent the climb at the source):** the +2.3 m/s² translational-lift
up-bias (diagnosed in A33 §V-2, un-fixed) is what lifts the drone above the gate in the first place.
S-1's cos⁴ speed cut mitigates it but did not eliminate it. If Fix 2 proves insufficient, the
next lever is reducing the up-bias (lower `forward_accel_mps2` further, or the flagged
low-collective sysid). **Recommend Fix 2 first** (it makes the loop SEE the excursion and correct
it, which is the missing capability), then re-assess the residual climb.

---

## 3. Failure 3 — startup dip to ground (proven)

* The drone spawns **~−18° nose-down** (cmd 1: pitch −17.8°, body_rate command 0 → it is the cold
  AHRS's initial attitude, not a commanded lean). The AHRS takes ~0.5 s to find "down" (theta_g
  spikes to 14° then settles) and the pitch estimate decays −17.8° → −10.8° over the settle.
* During settle+anchor (cmd 0-23) the `_bound_hold_thrust` band `[hold_thrust_lo_frac,
  hold_thrust_hi_frac]×hover = [0.90, 1.12]×0.2656 = [0.239, 0.297]` lets collective sit at **0.239
  (below hover)** — **50% of settle+anchor ticks are sub-hover**, anchor-phase mean 0.2435 < hover
  0.2656 → net sink. The egress thrust floor (1.0×hover) would hold altitude but only engages at
  egress (cmd 24+), AFTER the sink.

### Root
The settle/anchor hold allows **sub-hover collective while the cold estimator can't hold altitude**,
so the drone sinks before egress lifts it. (The A31 startup-swell fix TIGHTENED this band to
[0.90, 1.12] to stop the cold estimator railing the collective UP; but the 0.90 lower bound now
lets it sink.)

### Fix 3 — floor the settle/anchor collective at hover (no sink before egress)
Raise `hold_thrust_lo_frac` from 0.90 to **1.0** for vq2_case_c (or apply the egress thrust floor
during settle/anchor too): the cold-start hold can HOLD or gently climb, never sink. The upper
bound stays 1.12 (the A31 anti-swell cap). This is a one-value profile change:
`hold_thrust_lo_frac: 0.90 → 1.00`. Bias deliberately toward NOT sinking (a small climb off the
line is recoverable; a ground tap is not — mirrors the A34 cap philosophy).

> **[Minor open question — the −18° spawn pitch:] is the drone physically spawned nose-down ~18°, or
> is it the cold AHRS mis-reading level as −18° for the first ~0.5 s? If physical, Fix 3 (hold
> hover) is enough — the drone holds while the seeker levels it in egress. If it's an AHRS
> transient, the fix still holds (thrust at hover regardless of the mis-read pitch), but worth
> knowing for the estimator. Not blocking.**

---

## 4. Recommended fix sequencing

All three are largely independent; **Fix 2 is the highest-value single fix** (it is the "hits the
ceiling" killer and unlocks V-2). But **Fix 3 is the cheapest and gates everything** (if we tap the
ground on the line, nothing downstream matters). Recommended order:

1. **Fix 3 (one-value: `hold_thrust_lo_frac` 0.90 → 1.00)** — cheapest, stops the ground tap, get a
   clean full-altitude launch every flight. Ship first.
2. **Fix 2 (magnitude-gated trust floor in the vertical estimator)** — the ceiling killer; makes the
   loop see + correct a large vertical excursion, unlocks V-2. The real substance of A35.
3. **Fix 1 (rng-gate the wire pass + extend the turn coast)** — the turn; pending the turn-commit
   policy confirmation (§1). Independent of 2 & 3.

Fixes 2 and 3 can ship together (both vertical, both low-risk, one-value + estimator-gated). Fix 1
is separable and needs the policy call first. **All flag-gated, VQ1/case-A byte-identical.**

---

## 5. What I need from Fengyou

1. **Turn-commit policy (§1):** commit the turn at the PHYSICAL plane (rng≈3 m, ~1 s later) vs at
   the wire (~9 m early)? Recommend physical-plane. This decides Fix 1's shape.
2. **(Minor, non-blocking) the −18° spawn pitch (§3):** physical spawn attitude or cold-AHRS
   transient? Fix 3 works either way.

Nothing else is blocked — Fixes 2 & 3 are fully specified and ready to build on your go.

---

## 6. Evidence appendix (flight 1, `nav_estimate.jsonl`)

* Turn: gate_index 0→1 at cmd 80 / t 3.51 / rng 9.3 m, az≈0, bw 0.99; pass_wire 0 ticks all flight;
  pass_vis cmd 151-155 (rng 3.9 m) with pass_turn_yaw_rad −1.65; reverted to pursuit cmd 156 (17.9 m
  re-acquire); total yaw 37°.
* Vertical: offset_z_world +4.7 m (cmd 219+) while z_off_est drifted −0.15 → −3.9 m; bearing_w 0.04,
  zoff_w 0.02, zoff_innov +2.99 flagged accepted=False for ~100 ticks; thrust min 0.229 (floor 0.05
  never approached); integrated altitude peak 4.3 m.
* Startup: pitch −17.8° at cmd 1 (body_rate cmd 0 = estimator attitude), decaying to −10.8° over
  settle; theta_g spike 14°; settle+anchor thrust 50% sub-hover, anchor mean 0.2435 < hover 0.2656.
* Flights 2 & 3: 75 ticks / ~3 s, identical stubs, gate 0 only — restart artifacts, non-evidence.

*Analysis scripts (session scratchpad): `a35_diag.py` (3-flight char + flight-1 detail on all three
axes), plus the focused turn/vertical/startup follow-ups + `a35_offline_gate.py`.*

---

## 7. BUILD RESULT (A35, Fixes 2 & 3 only — Fix 1 deferred, 2026-07-03)

**Scope built (operator: "build 2&3 now, turn later"):** Fix 2 + Fix 3. Fix 1 DEFERRED (§1).

* **Fix 3** — `deploy_profile.py` vq2_case_c: `hold_thrust_lo_frac` 0.90 → 1.00 (one value; the
  1.12 anti-swell cap kept). Field default (0.6) unchanged → byte-identical off-path.
* **Fix 2** — `vertical_estimator.py`: new `use_zoff_big_trust` (default False) + `zoff_big_innov_m`
  (2.5), `zoff_big_innov_min_w` (0.5), `zoff_big_sign_consec` (3). In `_zoff_soft_correct`: track
  consecutive same-sign LARGE innovations; when large+persistent+consistent → (a) floor the applied
  correction weight, (b) allow reseed even from low bearing weight (the A33 V-1 exception), (c) set
  `_big_persistent_offset` so the predict step FREEZES the z_off propagate. All under
  `use_zoff_big_trust`; vq2_case_c turns it on via `vertical_estimator_overrides`.

**Offline gate (flight 1 latch stream, A34 vs A35 estimator):**
* z_off_est in the climb window (offset_z_world > +2 m, drone above gate): A34 **wrong sign**
  (median −25.7 m in the standalone replay; the flight itself reached −3.5 m — the replay's
  fabricated vz amplifies the drift, but the SIGN is the point), A35 **+21 m (correct sign)**.
* Wrong-sign tick count in the climb window: **A34 = 142 → A35 = 3** (target ~0). ✔
* Clean unit repro (sustained +4.7 m at bearing_w 0.04, drifting vz): A34 z_off = **−1.61**
  (matches the flight's −3.5 direction) → PD gate-term **+0.0645 (CLIMB, wrong)**; A35 z_off =
  **+5.52 (tracks +4.7)** → PD gate-term **−0.12 (DESCEND, floor now live)**. ✔
* Freeze (c): once engaged, 20 pure-predict ticks with a strong wrong-signed vz move z_off < 1.5 m
  (held, not drifted). ✔
* Fix 3: `_bound_hold_thrust` floors every raw thrust ≥ hover (0.2656) — no sub-hover settle. ✔

**Regression pins (all pass):**
* `use_zoff_big_trust=False` reproduces the pure-A34 estimator byte-for-byte on a mixed
  small+large latch stream. ✔
* Small/noisy offsets never trip the big-trust path → byte-identical to A34 (A32 frame-starvation
  fix + A28 calm vertical preserved). ✔
* `hold_thrust_lo_frac` field default 0.6 unchanged; vq1_case_a has no overrides. ✔
* EXPECTED_KEYS = **45** (unchanged — A35 adds NO nav log field; the fix is observable via the
  existing z_off_est / offset_z_world / thrust fields). ✔

**Full suite:** [reported in the hand-back message with the commit hash].
**Not flown** — separate handoff after the coordinator clears the offline gate.
