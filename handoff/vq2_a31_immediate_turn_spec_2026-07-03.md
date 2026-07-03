# A31 — Immediate turn on the wire gate-pass + overfly kill (diagnosis + fix spec)

**Run analyzed:** `data/runs/20260703_160715_rl_s1_f1` (A30, engine + commit 3dcbcc3).
**Baseline compare:** `20260703_024023` (A28). Operator ground truth: smoother overall (A30 win),
threaded gate 0 (first ever), but late/whipping turn to gate 1 → faced backwards → chaos; plus a
startup vertical swell → floor tap, and an unexplained left roll through the gate-0 pass.
**Verdict up front:** the commander's overfly frame is CONFIRMED with one precision — the yaw never
chased a gate whose *camera bearing* slipped behind (az stayed small, in-FOV, +0.17 rad mean); the
whole **line of sight physically rotated 203° in 4.9 s** because the drone, carrying ~3.9 m/s of
commanded build-up, orbited/overflew gate 1 while its yaw servo lag-followed the sweep all the way
to backwards. The guard therefore must fire on **cumulative LOS rotation**, not instantaneous
bearing. Every contributing latency and both secondary anomalies are traced to specific lines below.

---

## 1. Root cause, from the log (all times relative to first command)

### 1.1 The turn was structurally late: `pass_coast_s` is a 1.2 s blind glide

- `gate_index` flips 0→1 at **t=3.502** (tick 57) — the wire event was delivered same-tick
  (fly_rl.py:1516 reads `RACE_STATUS.active_gate_index` with no debounce; no lag upstream of the
  seeker).
- The seeker enters regime `pass` at that exact tick (`index_advanced` → `_begin_pass`,
  gate_seeker.py:1054) — so far so good. But `_pass_acquired_next` (gate_seeker.py:1079)
  **unconditionally ignores every pose until `elapsed >= pass_coast_s` (1.2 s)**. Pursuit of gate 1
  resumes at **t=4.72 = 3.50 + 1.22** — the first eligible tick — and that tick already has a
  usable gate-1 pose with capture age 0.32 s (captured ~t=4.4, i.e. **during** the blind window).
  Gate 1 was in view during the coast; the machinery refused to look.
- During the whole 1.22 s glide the heading is FROZEN (yaw ≈ 0.04) **and the seeker keeps
  accelerating**: `_pass_coast_command` (gate_seeker.py:1102) commands
  `pass_coast_accel_mps2 = 1.2` at full authority. Dead-reckoned from the logged commanded-accel
  schedule, the drone carries **~2.4 m/s at the flip and ~3.9 m/s at t=4.72** when it finally
  starts turning.

**Measured turn latency (pre-registered metric): flip → first yaw command toward gate 1 = 1.22 s.**
Target < 0.5 s.

### 1.2 The whip-around is a tail-chase orbit, not a yaw bug

Chase window t=4.72–9.7 (74 pursuit + 19 bridge ticks):

| quantity | value |
|---|---|
| true yaw (−nav.yaw) sweep | −0.04 → +3.34 rad = **193°** at mean **0.67 rad/s**, then keeps going past ±π (backwards) to ~−2.5 |
| yaw_des (world LOS to gate 1) sweep | +0.48 → +4.01 unwrapped = **203°** at **0.70 rad/s**, same direction, never reverses (monotone) |
| az_err (LOS − yaw) | pinned **+0.3…+0.5 rad** early chase (mean +0.17 whole window) — the yaw lag-follows the sweep, never converges, never loses the gate to the side |
| A30 lateral brake `alat` | **saturated at +1.5 m/s² (cap) on 63% of ticks** — correct sign (anti-tangential), insufficient magnitude vs the carried velocity |
| forward drive | `fwd_scale = cos²(az)` stayed 0.55–0.99 → a_fwd ≈ 0.7–1.2 m/s² kept feeding closing speed that the rotating LOS continuously converts to fresh tangential velocity |
| the "1205°" instrumentation number | reproduced exactly: it is the **total path length** of unwrapped yaw_des over ALL gate-1 ticks — 203° of clean whip + ~1000° of post-whip chaos hunting. It is the whip + its aftermath, **not** a clean multi-revolution orbit |

Yaw authority audit (exonerates the actuation chain): over the chase,
`median(realized_gyro_yaw / (0.4 × commanded_body_rate_z)) = 2.29` — the 2.5× realization /
`cmd_rate_scale=0.4` compensation holds for yaw on this wire. Commanded yaw rate averaged only
0.72 rad/s (32% of ticks at the 1.5 cap); realized/commanded ≈ 0.92. **Authority was available;
the geometry (v_t/r ≈ 0.7 rad/s LOS sweep) was the disease, the az lag just its symptom** — which
is also exactly the operator's "yaw NOT centered on the gate this time": in A28 the drone orbited
with the nose ON the gate; in A30 the faster sweep + pose age (~0.3 s) + slew produced a standing
~+0.35 rad lag.

Causal chain: 1.22 s blind accelerating glide → 3.9 m/s carried into a gate ~30° off-axis at short
range → LOS rate v_t/r ≈ 0.7 rad/s ≥ the closed-loop yaw-following rate → stable tail-chase → LOS
(and the obedient yaw) sweeps through 180° → gate mechanically unresolvable → chaos.
**Cure = (a) start the turn ~1 s earlier (more range, less speed, smaller sweep rate), (b) stop
accelerating through the pass, (c) a cumulative-LOS orbit-breaker so a forming whip is aborted
instead of followed to backwards.**

### 1.3 The left roll through gate 0: a close-range track hop drove the A30 lateral to the −1.5 rail

t=3.05–3.21: gate 0 centered (az +0.015…+0.04, alat ≈ 0, offset_z ≈ +0.68). Then:

| t | az_err | alat | offset_z_world | body_rate roll cmd | nav.roll (true = −nav.roll) |
|---|---|---|---|---|---|
| 3.21 | +0.015 | 0.00 | +0.675 | −0.01 | −0.002 |
| **3.25** | **−0.313** | **−1.50** | **+1.504** | −0.29 | −0.003 |
| 3.29 | −0.318 | −1.50 | +1.502 | **−0.62** | −0.002 |
| 3.32 (bridge) | −0.318 | −1.50 | +1.502 | −0.43 | +0.017 |
| 3.50 (pass) | stale | stale | stale | +0.62 (recovering) | **+0.116 → true −6.6° LEFT bank at the crossing** |

One fresh frame at t=3.25 hopped the tracked gate-0 solution **0.33 rad left + 0.83 m down** in a
single frame — a degenerate close-range PnP shift as the gate filled the image. It passed the
track-continuity gate because `track_max_bearing_jump_rad = 0.35` (gate_seeker.py:226) admits a
0.33 rad step; the A30 lateral (no slew of its own) snapped to the −1.5 cap the same tick; the A13
bridge then **re-issued the poisoned composed vector for 0.18 s more** (t=3.32→3.50) until the wire
flip froze the pass. Nothing is over there — the operator's read is exactly right; the roll chased
a phantom. The same hop **latched offset_z +1.50 (was +0.68) into the A28 vertical estimator**
(0.78 m innovation < the 2 m gate → accepted) — see §1.5.

**Why the range-based guards all missed it:** the pass never ARMED — commit came only from
`index_advanced`, which means **no gate-0 pose ever reported `range_m ≤ pass_arm_range_m = 3.0`,
even with the drone inside the gate.** Close-range PnP on this detector over-reports range, so any
fix keyed on `pass_arm_range_m` / `pass_degenerate_range_m` is unreliable on this wire. The fix
must live at the track jump gate + a lateral-demand slew (§2.4), not at a range threshold.

### 1.4 Startup vertical swell: seeded in the SETTLE hold, amplified by the PD transient

- t=0.0–0.75 (settle): the hold's collective **rails between both hold bounds** — 0.159 =
  hover×`hold_thrust_lo_frac`(0.6) at t=0.27 and 0.372 = hover×`hold_thrust_hi_frac`(1.4) pinned
  t≈0.5–0.7 — the alt-hold is chewing on a cold, garbage z estimate (A28 filter not yet seeded:
  `z_off_est=None` until t=1.06). ~0.3 s at +3.9 m/s² ≈ **+1.2 m/s upward velocity injected before
  pursuit even begins**.
- t=1.06: first latch — opening is 1.4–2.0 m ABOVE (z_off −1.4→−2.0): the climb itself is correct.
  But pre-loaded with the settle impulse the filter's vz rails at −2.5 (the export clip) at t=1.68
  and the drone **overshoots ~1 m above the opening** (z_off crosses to +0.95 by t=2.78) despite
  the on-paper ζ=0.92, then descends onto the opening.
- The descent then over-runs into the floor because of §1.3's poisoned latch: the hop wrote
  z_off +1.50 (vs +0.68 honest) at t=3.25, and through the 1.2 s pass glide there is no fresh
  latch, so `term_gate ≈ −0.030` (a standing descend bias on a ~2× exaggerated offset) rides the
  whole pass → the "brought it down in time … but tapped the floor" the operator saw.

The A28 law itself behaved (thrust tracked hover −kp·z_off +kd·vz throughout; no limit cycle) —
**keep it**. The fixes are the settle band and the hop rejection, not the gains.

### 1.5 Gate-1 pose pipeline degradation (context, not a new fix)

`track_range_m` is NULL all flight — that is an **instrumentation gap**, not a dead range channel:
`_last_track_range_m` is only written inside the A29 branch (`_compose_los_damped_accel`,
gate_seeker.py:1414), which A30 made unreachable. valid_empty=140 (detected-but-no-pose) is real —
the known negv1 poor-pose weakness — but 56 fresh yaw_des updates landed in the 4.9 s chase
(~11 Hz fresh-pose cadence): **the turn was not starved of bearings; it was starved of time and
fed too much speed.** Fix the logging (§2.6); detector pose quality stays on the vision
workstream.

---

## 2. The A31 design

All changes are new `GateSeekerConfig` fields defaulting to today's behaviour (byte-identical
flag-off), activated only via `vq2_case_c()` `seeker_overrides`. A28 vertical law, A29 clock
reconciliation, A30 image servo: **kept, untouched**.

### 2.1 FIX 1 (operator directive): immediate target-switch + turn on the wire pass event

**Change A — wire-pass fast window.** New config:

```python
# gate_seeker.py — GateSeekerConfig
# When the pass was committed (or confirmed while already committed) by the AUTHORITATIVE wire
# signal (RACE_STATUS.active_gate_index increment), the drone is PAST the gate plane -- the
# 1.2 s blind dead-reckon glide is pure lost time (run 20260703_160715: gate 1 was detected
# DURING the glide; pursuit resumed at exactly coast expiry, 1.22 s late, carrying ~3.9 m/s).
# Use this much shorter acquire-eligibility window instead (just enough to physically clear the
# frame the camera is inside). None => pass_coast_s (legacy, byte-identical).
pass_wire_coast_s: float | None = None
```

Implementation:
1. `_begin_pass(sim_time_ns)` gains a `wire: bool = False` arg; `_update_pass_state` passes
   `wire=index_advanced`. New state `_pass_wire: bool`.
2. In `_update_pass_state` (gate_seeker.py:1048), the early-return branch becomes an upgrade
   seam: `if self._passing: self._pass_wire = self._pass_wire or index_advanced; return` — a
   vision-committed pass (degenerate/lost-after-arm) that is then confirmed by the wire flips to
   the fast window mid-glide.
3. New helper `_effective_pass_coast_s()` → `pass_wire_coast_s if (_pass_wire and
   pass_wire_coast_s is not None) else pass_coast_s`; used by `_pass_acquired_next` (line 1088)
   and the ACQUIRE-NEXT track-reset gate (line 919–922). `_in_pass_dead_reckon`'s total bound
   becomes `_effective_pass_coast_s() + acquire_next_s`.
4. `_end_pass` / `reset()` clear `_pass_wire`.

**Override:** `"pass_wire_coast_s": 0.25`.

Expected: flip at ~t_p → track cleared and poses eligible at t_p+0.25 → first fresh pose ~t_p+0.3
→ yaw begins slewing toward gate 1 at **≤0.4 s** (was 1.22 s), from a longer range with ~1 m/s
less speed. `_pass_acquired_next`'s existing `range_m > pass_degenerate_range_m` filter stays: the
just-passed gate is behind the image plane (`t_cam z > 0.05` rejects it), so a 0.25 s window
cannot re-lock gate 0.

**No blind yaw pre-rotation** is added (case-C has no map — the drone cannot know gate 1's
direction before seeing it; this run proves it doesn't need to: gate 1 was in frame at eligibility
on the frozen heading). If a later track section hides the next gate off-FOV, a scan behaviour is
a separate, pre-registered follow-up — do not bundle it here.

### 2.2 FIX 2a: stop accelerating through the pass

**Override (existing field, value change only):** `"pass_coast_accel_mps2": 0.0` (was 1.2 default).

The pass glide should spend momentum, not build it (+1.5 m/s was added across the old 1.2 s
window). With the 0.25 s wire window the forward-lean release is a 0.25 s, slew-limited
(`body_rate_slew_max_rps2=8`) transition — nothing like the A5 snap-re-level (that bug was the
`velocity=0 + launch_ramp=0` hold, which this does not touch). For a vision-committed pass
(pre-plane, up to the full 1.2 s if the wire never confirms) the drone still carries ≥2 m/s
through a ~4 m glide — clears the opening in <2 s with no drive.

### 2.3 FIX 2b: re-ramp the forward drive out of every pass ("point before pushing", enforced)

New config + state:

```python
# Re-ramp the forward feedforward from zero over forward_ramp_s again after EVERY pass ends
# (next gate acquired), not just from the spawn release: the post-pass geometry is a fresh
# acquisition (bearing typically 30-60 deg off) and the A30 orbit data shows feeding forward
# drive while the yaw converges is what sustains the tail-chase. False => legacy (ramp measured
# from spawn release only; byte-identical).
reramp_forward_after_pass: bool = False
```

Implementation: new `_fwd_ramp_t_ns` (init = `_release_t_ns` at release); `_end_pass` sets it to
the current sim time when the flag is on; `_forward_accel_ramp` (gate_seeker.py:1748) measures
from `_fwd_ramp_t_ns` instead of `_release_t_ns` when the flag is on. Composed with the existing
`cos²(az)` scale, the first second after re-acquisition now averages a_fwd ≈ 1.2×0.5×0.8 ≈
0.5 m/s² while the lateral brake holds 1.5 — a 3:1 brake:feed ratio (was ~1:0.9).

**Override:** `"reramp_forward_after_pass": True`.

Deliberately NOT changing `forward_accel_mps2` (1.2) or the cos² exponent: the gate-0 leg was
healthy, and 2a+2b remove the specific speed feeds the data indicts. A dead-reckoned speed
governor (leaky ∫a_cmd with a cap) is pre-registered as the NEXT lever if a future run still
overflies on a long leg — not bundled now (open-loop speed estimates deserve their own
validation pass).

### 2.4 FIX 3: orbit-breaker (cumulative-LOS guard) + hard yaw-excursion clamp

The guard fires on **cumulative unwrapped LOS rotation since acquisition** — the correct
observable per §1.2 (instantaneous az stays small during a whip).

```python
# --- A31 ORBIT-BREAKER: abort a forming whip-around (2026-07-03) ---
# Run 20260703_160715: after overflying gate 1 the world LOS rotated 203 deg in 4.9 s and the
# yaw lag-followed it all the way to BACKWARDS. A pursuit whose LOS has rotated this far since
# acquisition is orbiting its gate, not approaching it -- following further is always wrong.
# 0.0 => guard off (byte-identical).
orbit_guard_rad: float = 0.0          # trip when |unwrapped psi_world - psi_at_acquire| exceeds this
orbit_break_s: float = 1.0            # bounded brake regime length
orbit_break_exit_az_rad: float = 0.15 # early exit: gate re-centered ...
orbit_break_exit_rate_rps: float = 0.3  # ... AND fresh-pose LOS drift below this
# Hard clamp on the slewed pursuit yaw setpoint, relative to the yaw at (re)acquisition of the
# current gate: the "never turn to backwards chasing a gate" pin. 0.0 => off.
orbit_yaw_clamp_rad: float = 0.0
```

Mechanics (all inside the pursuit path, vq2-gated):
1. On `_end_pass` (a fresh gate acquired) and on first-acquisition after a track drop: record
   `_chase_psi0 = yaw_des` (unwrapped baseline) and `_chase_yaw0 = current slewed yaw`; maintain
   `_chase_dpsi` by accumulating the shortest-path delta of yaw_des per fresh pose (reuse the
   pose-stamp dedupe; reset with the track).
2. `_slew_heading` result is clamped to `_chase_yaw0 ± orbit_yaw_clamp_rad` (when > 0) — the
   pursuit yaw physically cannot rotate past ~140° from where the chase began, whatever the
   bearing does.
3. When `|_chase_dpsi| > orbit_guard_rad` → regime `"orbit_break"` for ≤ `orbit_break_s`:
   forward accel 0, lateral = `image_lat_cap_mps2` toward the CURRENT apparent gate (the brake,
   same `e_right(yaw_now)` composition), yaw setpoint HELD (stop following the sweep — let the
   LOS come back as v_t dies). Early exit when `|az| < exit_az` AND the per-fresh-pose LOS drift
   < `exit_rate`; on exit, rebase `_chase_psi0/_chase_yaw0` and resume pursuit.
4. Second trip on the same gate acquisition → drop the track (`_track_* = None`) and fall to the
   no-detection hold: refuse the spin, wait level, reacquire clean.

**Overrides:** `"orbit_guard_rad": 1.75` (≈100°), `"orbit_break_s": 1.0`,
`"orbit_yaw_clamp_rad": 2.4` (≈137°). With FIX 1+2 the guard should never trip (LOS rate at the
earlier/slower turn ≈ 0.7×(v_t/2.4)×(r_old/r_new) ≈ 0.15–0.25 rad/s, converging); it is the
pre-registered safety net that turns a re-formed whip into a 1 s brake instead of a 180° spin.

### 2.5 FIX 4 (secondary): startup swell + hop-poison (the floor tap)

- **Settle thrust band** — the hold is supposed to be a conservative hover, not an alt-hold on an
  unseeded estimator. Override the existing fields:
  `"hold_thrust_lo_frac": 0.90, "hold_thrust_hi_frac": 1.12` (band 0.239–0.297 vs 0.159–0.372).
  Worst-case injected velocity over the 0.75 s settle drops from ~1.2 m/s to ~0.3 m/s; the A28 PD
  then closes the honest −1.5…−2 m opening offset with margin instead of pre-loaded (the ~1 m
  overshoot at t=2.8 and the deep descent that followed both shrink). A7's free-fall concern is
  covered: the floor is 0.90×hover (near-hover), and the egress thrust floor (1.0×hover) is
  untouched.
- **Track hop rejection — AS BUILT (commander-directed upgrade, deviates from this spec's
  original design).** The original fix here was `"track_max_bearing_jump_rad": 0.25` (tighten the
  fixed threshold 0.35 → 0.25). The operator directed a ROBUST replacement instead: the
  **IMU-consistency bearing gate** (`use_imu_bearing_gate`) — the horizontal analog of the A28
  vertical complementary filter. Rationale: a fixed threshold of ANY size can be defeated by a
  right-sized hop; the t=3.25 hop slid under 0.35 and a future one can slide under 0.25.
  Principle: gates are STATIC — over one inter-frame dt the gate's bearing can only change as
  fast as the drone's own MEASURED motion, dominated by rotation (AHRS attitude delta, measured
  cleanly) plus a translation-parallax term (∝ motion/range, larger at close range). AS BUILT:
    * **Frame choice:** gate in the WORLD frame. Each accepted pose's camera lever is rotated
      with the attitude AT ITS CAPTURE TIME (the A30 `_att_hist` ring buffer via `_rpy_at`), so
      the measured rotation between the two frames is compensated EXACTLY (this IS the
      IMU-predicted bearing, expressed as a world direction) and a static gate's world direction
      is quasi-constant — the only honest residual is translation parallax, bounded explicitly.
    * **Predict:** expected world direction this frame = the last ACCEPTED tracked pose's world
      direction (`_bg_prev_dir_world`, stamped `_bg_prev_pose_ns` on the camera epoch — both
      stamps same clock, so the A29 epoch-rate skew cancels in dt).
    * **Reject:** deviation = angle(candidate world dir @ its capture attitude, prediction);
      allowance = `bearing_gate_noise_rad` (0.06 ≈ 3.4°: PnP bearing noise + AHRS-delta margin)
      + `bearing_gate_trans_mps` (4.0) × dt / max(track range, `bearing_gate_min_range_m`=1.0).
      The gate widens ∝ dt (predicts THROUGH short pose gaps — also helps the gate-1 dropout) and
      ∝ 1/range (a legitimate close-range sweep is never falsely rejected: honest gate-0 crossing
      sweep 2.4 m/s × 0.045 s / 3 m = 0.036 rad ≪ the 0.12 rad allowance). Close-range PnP
      OVER-reports range (§1.3), which UNDER-sizes the allowance — the strict, safe direction.
      The t=3.25 hop: dev 0.33 rad vs allowance ~0.105 rad at its dt/range → rejected 3× over,
      REGARDLESS of any fixed threshold. Rejection routes to the existing continuity-coast
      (bridge covers the gap). Removes BOTH the left-roll injector and the +0.82 m garbage
      vertical latch (§1.3/§1.4) at their shared source.
    * **Replaces** the fixed `track_max_bearing_jump_rad` check for vq2_case_c (the profile does
      NOT tighten it; it stays 0.35 as the legacy fallback when no capture-time attitude is
      available and as VQ1's untouched path). The range-jump check (6 m) is unchanged.
    * **Pinned by tests** (test_vq2_a31_immediate_turn.py §5): the ACTUAL bad frame replayed —
      a 0.33 rad close-range hop with a <0.3° measured attitude delta is ACCEPTED by the legacy
      0.35 gate and REJECTED by the IMU gate; a legitimate parallax sweep (consistent with
      bounded translation, incl. across a 0.15 s gap) is ACCEPTED; and the prediction SIGN is
      pinned (a bearing that moves OPPOSITE a measured +0.15 rad yaw — static-gate-consistent —
      is accepted; one that moves WITH it is rejected; a flipped sign would swap the outcomes).
  Range-gated guards were considered and rejected: §1.3 proves close-range PnP range never reads
  ≤3 m on this wire.
- **Slow the approach (commander directive, addition to this spec's original scope)** — override
  `"forward_accel_mps2": 0.8` (was 1.2; field default untouched). §1.2's disease is geometric:
  LOS sweep rate v_t/r ≈ 0.7 rad/s outran the closed-loop yaw follow. Speed build-up scales
  ~sqrt(a) (~0.82×), compounding with the zero coast accel (−1.5 m/s) and the ~1 s earlier turn
  to roughly HALVE the carried speed at gate-1 acquisition → sweep ~0.2–0.3 rad/s, inside what
  the yaw servo + the A30 ±1.5 lateral brake (which saturated 63% of the A30 chase) can kill.
- **Lateral-demand slew** (defense in depth for hops that pass any gate): new config
  `image_lat_slew_mps3: float = 0.0` (0 = off, byte-identical); when > 0, `a_lat` in
  `_compose_image_servo_accel` is rate-limited per tick toward its target. Override
  `"image_lat_slew_mps3": 6.0` — full-scale reversal (−1.5→+1.5) takes 0.5 s; an honest az ramp
  (≤0.7 rad/s × 8 = 5.6 m/s³ worst) is never limited, a one-frame rail-snap is.

### 2.6 Instrumentation (logging only)

1. **Fix the track_range NULL:** write `self._last_track_range_m = (self._track_range_m if
   self._track_range_m is not None else pose.range_m)` at the top of `_visual_pursuit_command`
   every pursuit tick (it is currently A29-branch-only) — flight-over-flight range history is how
   the overfly radius gets measured next run.
2. New regime strings: `"orbit_break"`; suffix the pass regime with its commit source
   (`"pass_wire"` / `"pass_vis"`) via `_last_regime` so the fast-window path is auditable.
3. Stash `_last_chase_dpsi` (the guard integrator) + `_pass_wire`; fly_rl logs them as
   `chase_dpsi_rad` / `pass_wire` alongside the existing A30 fields.

---

## 3. Byte-identity plan

| change | mechanism | VQ1/case-A effect |
|---|---|---|
| `pass_wire_coast_s=None` default | None → `pass_coast_s` used everywhere | identical |
| `_pass_wire` state + upgrade seam | only changes behaviour when `pass_wire_coast_s is not None` | identical (state written, never read) |
| `pass_coast_accel_mps2` 0.0 | **profile override only**, field default stays 1.2 | identical |
| `reramp_forward_after_pass=False` default | `_fwd_ramp_t_ns` == `_release_t_ns` when off | identical |
| `orbit_guard_rad=0.0` / `orbit_yaw_clamp_rad=0.0` defaults | guard/clamp code unreachable at 0 | identical |
| `image_lat_slew_mps3=0.0` default | slew bypassed at 0 | identical |
| hold band / bearing-jump values | profile overrides only, field defaults untouched | identical |
| instrumentation stashes | logging only, never consumed by control | identical commands |

Verification: existing unit suite + the A30 sign/orbit pins must pass unchanged; add pins for
(a) wire-committed pass uses the fast window, vision-committed without wire uses the legacy one,
(b) orbit guard trips at >1.75 rad cumulative LOS and the break command has zero forward + capped
lateral toward the gate + held yaw, (c) yaw clamp holds at baseline±2.4, (d) lateral slew limits a
rail-step to 6 m/s³, (e) flag-off config produces bit-identical commands on a recorded A30 tick
stream (replay harness as in A28/A29 specs).

## 4. vq2_case_c seeker_overrides after A31 (full dict, for the wiring diff)

```python
seeker_overrides={"egress_freeze_attitude": True, "hold_last_demand_s": 0.6,
                  "true_attitude_from_ahrs": True,
                  "use_image_servo_lateral": True,
                  "total_accel_cap_mps2": 2.0,
                  "pursuit_yaw_slew_rps": 1.5,
                  # --- A31 (as shipped) ---
                  "pass_wire_coast_s": 0.25,          # immediate turn on the wire pass
                  "pass_coast_accel_mps2": 0.0,       # spend momentum through the pass, don't build it
                  "reramp_forward_after_pass": True,  # point before pushing, enforced post-pass
                  "forward_accel_mps2": 0.8,          # slow the approach (commander directive, §2.5)
                  "orbit_guard_rad": 1.75,            # cumulative-LOS whip abort
                  "orbit_break_s": 1.0,
                  "orbit_yaw_clamp_rad": 2.4,         # never chase a gate to backwards
                  "use_imu_bearing_gate": True,       # IMU-consistency bearing gate (§2.5 as-built;
                                                      # REPLACES the tightened 0.25 fixed threshold —
                                                      # track_max_bearing_jump_rad stays 0.35 fallback)
                  "image_lat_slew_mps3": 6.0,         # no one-frame lateral rail-snap
                  "hold_thrust_lo_frac": 0.90,        # settle = conservative hover, not garbage alt-hold
                  "hold_thrust_hi_frac": 1.12},
```

Controller overrides: **unchanged** (A28 vertical law kept intact per constraint).

## 5. Pre-registered flight checks (next fly)

1. **Turn latency:** first pursuit tick on gate 1 within **0.5 s** of the `gate_index` flip
   (was 1.22 s); regime log shows `pass_wire`.
2. **Thread gate 0 AND gate 1.** Gate-1 approach: az_err converges below 0.15 rad and STAYS
   (no standing +0.35 lag); cumulative `chase_dpsi_rad` on gate 1 < 1.75 (guard never trips). If
   the guard does trip: exactly one bounded `orbit_break` (≤1 s), then converging pursuit — and
   in no case |true yaw − yaw at gate-1 acquisition| > 2.4 rad.
3. **Startup:** settle thrust stays inside [0.239, 0.297]; z_off overshoot past the opening
   < 0.4 m (was ~1.0 m); no floor contact during the gate-0 pass.
4. **Left roll gone:** no |alat| ≥ 1.4 m/s² tick within 1 s before a pass commit on gate 0 unless
   az has been ≥ 0.15 rad for ≥ 3 consecutive fresh poses (i.e. no single-frame rail-snap); true
   roll through the pass |φ| < 3°.
5. **Speed sanity (new instrumentation):** logged track range at gate-1 acquisition ≥ 6 m and
   monotonically closing thereafter (no through-minimum-and-out signature = no overfly).
6. **Regression:** vertical channel matches A28's behaviour (operator: "calm vertically"); no
   bang-bang thrust; motion smoothness at least A30's (operator judgment).

Revert criteria: if the fast window re-locks the just-passed gate (log: gate-1 acquisition range
< 3 m falling), raise `pass_wire_coast_s` 0.25 → 0.5 before touching anything else. If bridge
coverage collapses under the IMU-consistency bearing gate (none_continuity_reject share > 50% of
gate-1 ticks — audit `bearing_dev_rad` vs `bearing_allow_rad` in the log to see WHICH side is
wrong), first raise `bearing_gate_noise_rad` 0.06 → 0.10; if honest rejects persist at close
range, raise `bearing_gate_trans_mps` 4.0 → 6.0; only then fall back to
`use_imu_bearing_gate: False` + `track_max_bearing_jump_rad: 0.25` + the lateral slew alone.

## 6. What was checked and exonerated

- **RACE_STATUS delivery:** same-tick at fly_rl.py:1516/1797 — no wire-side lag.
- **Yaw actuation chain:** realized/(0.4×cmd) = 2.29 ≈ the modeled 2.5×; A22's identity
  `body_rate_sign` correct on this wire; commands averaged half the cap — authority not binding.
- **A30 lateral sign:** saturated brake pointed anti-tangential throughout the whip (correct);
  its cap, not its sign, was outmatched. It does not couple into heading (yaw setpoint comes only
  from the slewed LOS; the roll term is a translational accel) — the operator's "banks/rolls left
  then whips yaw" sequencing is the hop transient (§1.3) followed by the lag-following whip
  (§1.2), two separate mechanisms.
- **A28 vertical law:** tracked its inputs faithfully all flight; the floor tap traces to a
  poisoned INPUT (the t=3.25 latch), not the law. Watch item only: vz_est touched the ±2.5 export
  clip at t=1.68 during the settle-seeded climb; expected to vanish with the tightened hold band.
