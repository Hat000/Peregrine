# VQ2 A33 — Gate-2 Coordinated Intercept: Diagnosis + Fix Spec (2026-07-03)

Run analyzed: `data/runs/20260703_210632_rl_s1_f1` (A32 stack, branch `vq2-gate2-turn-dive` @ `75213c8`).
485 commands, 1:1 with `nav_estimate.jsonl` ticks — the operator's "cmd N" = tick index N.
Attitude is TRUSTWORTHY this flight (A32; theta_g median 3.3°), so every signal below is read at
face value except where explicitly flagged. `hover_thrust = 0.2656` (FIT, recovers g to 2.3%); the
controller's LINEAR thrust model (~37 m/s²/collective) UNDER-states the measured convex plant map at
the floor — see V-2 for the reconciliation.

**Operator rulings folded in (2026-07-03, via commander):**
* **Q1 — turn timing:** credit assumed at the gate-plane pass; the wire advance is the turn trigger.
  "TURN AS FAST AS POSSIBLE, momentum will carry us through the gate." The 1.2 s frozen-yaw blind
  coast must DIE, and the good next-gate frames arriving during it must be USED, not discarded.
* **Q2 — gate heights:** HIGH GATES EXIST (gate 2 itself; higher later). Climb authority stays
  FULLY SYMMETRIC — the overshoot is fixed at the SOURCE (reseed/smear), never by starving climb.
* **Amendment 1 — pass occlusion:** the gate-1 frame occludes the camera at the pass, so the turn
  must begin on the PASS TRIGGER, not on gate-2 acquisition, and ride blind through the occlusion
  (→ H-1c).
* **Amendment 2 — active down-authority:** "we can push the drone down FASTER than gravity." Real
  net-down authority exists below free-fall, in a region our sysid never measured (→ V-2 re-derived:
  open the floor to reach it, conservatively, with a watchdog + a flagged sysid gap).

---

## 0. Executive summary

The failure is a **sequencing cascade, not an authority or gain problem**. Reconstructed
tick-by-tick, aligned with the operator's narrative:

1. **cmd 49 (t 3.21): the RACE_STATUS index advances ~3 m BEFORE the physical gate-1 plane.**
   The tracked gate-1 is still dead ahead (az ≈ 0.01, same height, range EMA 5.4 → raw ~3.8–3.4 m
   by cmd 54–56) and the drone physically threads it 1.2–1.5 s LATER (operator: thread + land
   after the advance). **A31's premise — "index increment = past the gate plane" — is FALSE on
   this sim.** The A31 fast window fires prematurely, the acquire-next re-locks the SAME gate-1
   (nearest, dead ahead), and the one-shot `_pass_wire` fast window is SPENT. Per the operator's
   Q1 ruling this early trigger is fine — the drone should have started the gate-2 turn RIGHT
   THERE, riding momentum through gate-1's opening. Instead it burned 2.9 s (points 2–5).
2. **cmd 62–70: the gate-track SMEARS from gate-1 onto gate-2 instead of handing off.** The A32
   soft bearing weight removed the HARD range-jump reject on the soft path, so gate-2 candidates
   8+ m off the track's range prediction (bw 0.11–0.31 — correctly flagged suspect) still nudge
   the track EMA every frame: the range EMA walks 2.8 → 11.4 m across ~10 frames and the
   bearing-gate reference walks with it. No clean track drop, no clean re-acquisition — a silent
   identity swap.
3. **cmd 67: the vertical estimator's reseed-on-persistence TELEPORTS z_off onto the smear**
   (−0.13 → −5.00 m "gate above"): 4 consecutive big innovations, each from a LOW-WEIGHT frame
   (bw 0.11–0.31). The A32 reseed never consults the weight — 4 garbage frames re-lock the state
   exactly like 4 honest ones. (Gate 2 IS high — a real climb was owed — but the smear-frame −5 m
   was the unvetted version of it, latched while the drone was still short of gate-1.)
4. **cmd 72–90 (t 4.83–6.12): the PHYSICAL gate-1 pass commits as a VISION pass and gets the
   LEGACY 1.2 s frozen-yaw blind coast** (`_pass_wire` False — spent at cmd 49). Yaw FROZEN at
   +47° for 1.3 s while gate-2 sits at +86°; well-detected gate-2 poses (bw 0.85–1.0!) are
   DISCARDED by the acquire-eligibility window the entire time; the drone coasts forward on
   momentum. This is the operator's *"we yaw to face the second gate, once again NOT ENOUGH, we
   fly forward."* Meanwhile the vertical loop climbs at its full +0.12-collective rail toward the
   teleported offset: **+2.3 m altitude and ~2 m/s climb momentum built during the stall.**
5. **cmd 90–108: the second turn** (*"~cmd 100 we realize we didn't turn enough"*). Once pursuit
   resumes, the machinery is HEALTHY: yaw at ~80°/s (the 1.5 rad/s slew), a_lat 100% sign-correct,
   az closes 0.73 → 0.03 rad in ~1.2 s. Total wire advance → pointed-at-gate-2: **4.1 s**, nearly
   all of it the two stalls.
6. **cmd 108–156: the vertical overshoot.** The drone reaches gate-2 height still climbing
   ~0.5–1 m/s (stall-built momentum). The PD does the right thing — it drives the collective to the
   0.150 floor and WANTS to go lower (its bounded worst-case demand is ~0.0) — **but the FLOOR CAPS
   it**, and the climb does not arrest: `vz_est` stays ~−0.5 to −1.0 m/s the whole window. Realized
   net vertical accel at the floor ticks was ≈ −0.55 m/s² (near-neutral, still rising) vs the
   measured plant map's −7.68 prediction → **~+7 m/s² of translational lift AND large unused
   down-authority margin** (the plant was nowhere near a limit; the FLOOR, not the plant, was the
   binding constraint). The up-bias is NOT a hover offset (map recovers g to 2.3%) and NOT
   rate-coupling (corr(|ω|,residual) −0.23) — it is speed-driven lift (see V-2). The drone rises
   above the gate; the +20°-up camera drops it out of frame (*"climbing and yawing right until the
   gate is out of view"*). **Fix: V-2 opens the floor so the PD reaches its designed descent; V-1
   stops the phantom-climb into the lift regime; S-1 cuts the speed that drives the lift.**
7. **cmd 158–164: garbage endgame.** A +27° pitch-up excursion (t 10.2), PnP garbage (bw
   0.02–0.04), and the reseed TELEPORTS z_off again (+2.82 → −2.97 on frame weight 0.00–0.04) —
   flipping the vertical command from descend to CLIMB exactly when the drone is high and must
   come down. Bridge/ZOH freezes the garbage demand for 1.5 s. Floor impact ~cmd 170, chaos
   after. All |offset_z_world| > 8 m episodes are POST-crash (cmd 196+).

**Verdict on the briefed hypotheses:**

* *"A32 Huber soft-weight down-weights REAL vertical offsets (zoff_w median 0.32)"* — **REFUTED
  as the mechanism.** Pre-crash HONEST latches (|innov| < 1.4 m, n=81) carry w median **0.87**;
  the Huber factor is 1.0 for 100% of them (median |innov| 0.52 m « k·sigma = 1.4 m). The 0.32
  whole-flight median averages over chaos-phase garbage. The A32 vertical regression is real but
  lives elsewhere: **(a) the weight-blind reseed teleports** (cmd 67, 164) and **(b) the soft
  track path feeding a gate-identity SMEAR into the innovation stream** (the removed hard range
  reject). Fix those; leave sigma/k/alpha/beta alone.
* *"offset_z_world ±23 m = vertical regressed"* — the ±23 m swings are post-crash chaos; the
  flight-relevant span was −5.1..+2.6 m. The regression vs A28 is the reseed + smear chain, plus
  the overshoot physics in (6).
* *"Startup balloon"* — liftoff clean (operator; A31 hold bounds worked). The gate-1 approach
  climbed +1.9 m and threaded near the TOP edge — mild climb-overshoot, same weak-brake signature
  as (6) at small scale.
* *"Lateral/bank under-powered"* — **REFUTED.** 31% saturation, 100% sign agreement, turn at the
  slew cap when free to run. The "not banking" feel is the two SEQUENCING stalls.
* *"Speed"* — partially confirmed: ~2 m/s carried through a 1.3 s frozen coast made the stall
  expensive, and off-axis forward drive stayed high (cos² at az 42° still pushes 55%). Fix the
  stall first (H-1), trim off-axis drive second (S-1).

---

## 1. THE FIX — five changes, priority order

All changes are opt-in config fields defaulting to today's behavior; `vq2_case_c()` flips them.
VQ1/case-A byte-identity holds because every touched path is behind a new default-off flag or
inside an existing vq2-only branch.

### H-1. TURN-NOW pass handoff: kill the blind coast, exclude the just-passed gate by GEOMETRY, not by TIME — `gate_seeker.py`

The acquire-eligibility windows (0.25 s wire / 1.2 s vision) exist only to keep the seeker from
re-locking the gate it is passing. Replace that TIME discrimination with a GEOMETRIC one, then
collapse the windows to ~zero — the operator's "turn as fast as possible, use the good frames."

**(a) Old-gate exclusion.** New config:

```python
pass_exclude_prev_gate: bool = False       # vq2_case_c: True
pass_prev_gate_excl_rad: float = 0.35      # world-direction cone around the pass heading
pass_prev_gate_excl_margin_m: float = 3.0  # range slack beyond the last tracked range
```

In `_begin_pass`, BEFORE the `_reset_bearing_gate()` call, snapshot the passed gate:
`_pass_prev_dir_world = self._bg_prev_dir_world` (capture-time-rotated world direction; fallback
`[cos(_pass_heading), sin(_pass_heading), 0]`), `_pass_prev_range_m = self._track_range_m`
(fallback `pass_arm_range_m`). While `self._passing` and the flag is ON, in `detect_gate_lever`
filter the candidate list BEFORE `_first_acquisition` (the call site at ~line 939 — the only
acquisition path while passing, since `_begin_pass`/the acquire-next reset cleared the track):

```python
def _is_prev_gate(p):   # built where rpy context exists (detect_gate_lever)
    rpy = self._rpy_at(int(p.sim_time_ns)) or self._att_rpy(nav_proxy)   # same source as A30
    d = self._gate_dir_world_rpy(p, rpy)
    ang = arccos(clip(d @ _pass_prev_dir_world, -1, 1))
    return (ang < pass_prev_gate_excl_rad
            and p.range_m <= _pass_prev_range_m + pass_prev_gate_excl_margin_m)
poses = [p for p in poses if not _is_prev_gate(p)]     # empty -> normal no-candidate coast tick
```

Replay check on this run: at cmd 49 the snapshot is (dead-ahead dir, range 5.4). Gate-1 residue
(0.0 rad off, 2.8–3.8 m ≤ 8.4) → EXCLUDED. Gate-2 (+1.48 rad off) → passes regardless of range. A
straight-section next gate dead ahead at 10–12 m → passes on range (> 8.4). Exclusion state clears
in `_end_pass`/`reset`.

**(b) Collapse the eligibility windows** (profile `seeker_overrides`, field defaults untouched):
`pass_wire_coast_s: 0.25 → 0.0` and `pass_coast_s: 1.2 → 0.3` (vis-committed passes are at/near
the plane by construction — degenerate/lost-after-arm — so 0.3 s ≈ 0.6 m of glide; the wire path
needs no glide at all since the exclusion, not the clock, now rejects the old gate).
`_pass_acquired_next` needs no change: with the acquisition filtered upstream, its
`range > pass_degenerate_range_m` check is only a backstop.

**(c) TURN-THROUGH-OCCLUSION — begin the slew on the PASS TRIGGER, not on gate-2 acquisition
(operator amendment 1, 2026-07-03).** Fengyou: at the pass the gate-1 frame occludes much of the
camera, so there may be NO gate-2 pose for a beat — the turn must NOT wait for one.

*Verified against the code:* as originally written H-1 **does stall the turn** during occlusion.
In `command_visual`, while `self._passing` and `_pass_acquired_next` is False (no gate-2 pose yet),
control routes to `_pass_coast_command`, which sets `yaw0 = self._pass_heading` (the FROZEN pre-pass
heading) and holds it — the yaw slew only starts after `_end_pass`, which requires an acquired pose.
So an occluded pass beat = a frozen-heading beat. This is exactly the amendment's concern; fixed
here.

*Fix:* the coast heading becomes a BLIND TURN TARGET that the coast slews toward, so the turn
begins on the pass commit itself and only refines (not starts) when gate-2 clears into view. New
config:

```python
pass_turn_through: bool = False        # vq2_case_c: True
pass_blind_turn_cap_rad: float = 1.6   # max blind heading change during the coast (~92°; a bounded
                                       #  open-loop turn, never a full spin toward "backwards")
```

At `_begin_pass`, latch a blind turn target `_pass_turn_yaw`:
* **preferred** — if a fresh pre-pass gate-2 sighting exists (a valid pose this frame that
  `_is_prev_gate` (b) marks as NOT the passed gate — i.e. the off-axis next gate already glimpsed),
  aim at its world bearing `atan2(los_y, los_x)`, clamped to `_pass_heading ± pass_blind_turn_cap_rad`;
* **fallback** — no next-gate glimpse yet: aim at `_pass_heading + sign * pass_blind_turn_cap_rad`
  where `sign` is the sign of the last pursuit `az_err` (the direction the gate was already drifting
  off-nose), or the last `_chase_rate` sign; if neither is known, hold heading (degrades to today).

`_pass_coast_command` then slews `_last_yaw` from `_pass_heading` toward `_pass_turn_yaw` at
`pursuit_yaw_slew_rps` (the same 1.5 rad/s cap the pursuit uses — one turn-rate limit everywhere)
instead of freezing it; forward drive stays the coast's `pass_coast_accel_mps2` (0.0 here, per A31)
so it is a pure pointing turn on momentum, no lunge. When gate-2 poses clear the occlusion,
`_pass_acquired_next` fires → `_end_pass` → normal pursuit takes over the now-nearly-aligned yaw
seamlessly (the blind target was already in the right direction, so there is no snap). If gate-2
never clears within the coast+acquire window, the bounded blind turn simply completes and the
gentle no-detection hold catches it — never a spin.

Default OFF (`pass_turn_through=False`) → `_pass_coast_command` freezes the heading exactly as
today (byte-identical). This is strictly additive to (a)/(b).

**Effect replayed on this flight:** wire advance cmd 49 → pass commit → the blind turn target
latches toward the last-known gate-2 drift direction (right, +az) and the yaw slew begins
**immediately on the trigger** rather than at acquisition; gate-2 poses (bw 0.85–1.0) refine it the
instant they clear (~0.1–0.3 s later), and the second spurious 1.2 s pass_vis stall at cmd 72
cannot occur. Expected wire→pointed latency: ~1.0 s of 85° yaw slew at 1.5 rad/s, started at the
trigger ≈ **~1.2 s vs 4.1 s flown**, with ZERO frozen-yaw beats even if the view is occluded for
0.2–0.5 s (flight-checked FC-9). The forward re-ramp (A31 `reramp_forward_after_pass`, kept) plus
S-1's cos⁴ means the drone barely drives forward until the nose comes around — momentum alone
carries it through gate-1's opening, exactly the operator's intent. **Watched risk (accepted per
Q1 ruling, flight-checked FC-2):** lateral image-servo accel toward gate-2 while still short of
gate-1's plane can displace ~1 m before the plane; the `image_lat_slew_mps3=6` ramp and the
az-saturated `fwd_scale≈0` deceleration both soften it. **Second watched risk (FC-9):** a blind
turn toward a WRONG direction if the pre-pass drift sign is misleading — bounded by
`pass_blind_turn_cap_rad` (92°, never backwards) and corrected within ~2 detector frames once
gate-2 clears; on this course the next-gate glimpse is available at commit (gate-2 was tracked/seen
before the pass), so the preferred branch dominates.

### V-1. Weight-qualified reseed (the teleport killer) — `vertical_estimator.py`

**Change:** in `_zoff_soft_correct`, the miss/reseed machinery only engages on frames whose
applied weight is credible:

```python
# new field, default 0.0 == byte-identical (any weight counts, today's behavior)
zoff_reseed_min_w: float = 0.0        # vq2_case_c vertical_estimator_overrides: 0.3

# in _zoff_soft_correct, the nu > zoff_miss_nu branch:
if nu > self.zoff_miss_nu:
    self._zoff_last_accepted = False
    if w >= self.zoff_reseed_min_w:            # a garbage frame is NOT retarget evidence
        self._zoff_miss += 1
        if self._zoff_miss >= self.reseed_after:
            self._z_off = z_meas               # re-lock only ever onto a credible frame
            self._zoff_miss = 0
            self._zoff_last_accept_t_s = now_t
    return
```

Rationale: a REAL retarget (gate handoff) arrives on frames the bearing gate scores well — after
a track drop the bearing reference resets and the new gate's first frames carry w = 1.0 (code:
`_last_bearing_w` is 1.0 when no live IMU-gate reference exists), so an honest handoff still
reseeds in 4 frames, and the legitimate gate-2 CLIMB command still fires — at the vetted
magnitude. A garbage stretch (bw 0.00–0.04, cmd 158–164) can no longer teleport the state; it
coasts (blind-coast leak after 1 s — today's bounded behavior). Both flight-ending teleports
(cmd 67: frames 0.11–0.31; cmd 164: frames 0.00–0.04) die at the 0.3 threshold — verified against
the log. This is the operator-endorsed fix shape for the overshoot: *"the climb command is only as
large as the TRUE offset"* — full climb authority retained (Q2), garbage magnitudes rejected.

### H-3. Hard range-jump reject on the A32 soft track path (the smear killer) — `gate_seeker.py`

**Change:** the soft path (`detect_gate_lever`, `if soft: cands = poses`) reinstates the RANGE leg
of the continuity check as a hard candidate filter, keeping the bearing leg soft:

```python
# new GateSeekerConfig field, default False == byte-identical A32 soft path
soft_range_hard_reject: bool = False   # vq2_case_c seeker_overrides: True

if soft:
    if self.config.soft_range_hard_reject:
        cands = [p for p in poses
                 if abs(p.range_m - pred_r) <= self.config.track_max_range_jump_m]
        if not cands:
            # identical bookkeeping to the binary continuity_reject branch:
            # coast tick, drop track after track_max_coast_ticks, return None
            ...
    else:
        cands = poses
```

Rationale: a candidate 8+ m off the track's range prediction is a DIFFERENT PHYSICAL OBJECT — a
noisy measurement of the same gate cannot move 8 m between frames — so Cauchy-weighting it is a
category error (it nudged the EMA 11%/frame and walked the track gate-1→gate-2 in ~10 frames,
dragging the bearing reference along). The A31→A32 starvation (59.9% rejects) came from the
BEARING gate's binary consequence, which STAYS soft — this restores only the range wall. The
sub-threshold `nu_r` term in the Cauchy weight is unchanged. With H-1 in place this fix's main
job is protecting the track (and the z_off latch stream) from cross-gate contamination in all the
OTHER moments: overfly geometries, re-acquisition after loss, the chaos tail.

### V-2. OPEN the floor to reach real net-down authority (operator amendment 2) — `deploy_profile.py` `controller_overrides`

Fengyou (amendment 2): *"the drone can thrust down if needed … we are at thrust floor but still
continuing up — consider a little bit of thrust to push the drone downwards if need be."* Follow-up
ruling: **"We can push the drone down FASTER than gravity"** — the real platform has NET-DOWNWARD
authority below free-fall, in a region our sysid never measured.

**Honest log analysis FIRST (his required caveat) — RECONCILED against the plant maps + the live
log.** Commander-supplied plant facts, all folded in:
* `hover_thrust=0.2656` is a FIT recovering g to 2.3% (~0.23 m/s²) — **10× too small to be the +2.3
  up-bias; hover recalibration is REFUTED as the fix.**
* The MEASURED convex map (`rl_plant.py:144-151`, 4812-sample fit) is only trustworthy in its fit
  band **0.15–0.20** (net a_up−g: 0.15 → −7.68, 0.20 → −5.11 at level). **Below 0.15 it is NOT
  measured** — the 0.0/0.10 knots were the linear extrapolation of the 0.15–0.20 segment FLOORED AT
  0 (`rl_plant.py:139-142`: the raw extrapolation −5.60/−0.45 was "worse-than-free-fall" and
  discarded). So the "free-fall ceiling at 0.10" in my prior draft was an ASSUMPTION baked into the
  map, not a measurement. Fengyou confirms the real plant produces net-down force there.
* **Reconciliation (the commander's standing task, done):** at the clean pinned-floor ticks
  (cmd 149–156, before the cmd-158 pitch excursion) the drone's REALIZED net vertical accel was
  **≈ −0.55 m/s² (median) — essentially neutral, still drifting UP** (independent cross-check:
  vision `offset_z_world` slope +1.4 m/s rising; `vz_est` −0.65..−1.06 climb). The measured map
  predicts −7.68 at that collective; realized ≈ 0. Two conclusions: (i) the mid-range measured map
  is NOT cleanly live — the drone carried **~+7 m/s² of translational lift** that no collective map
  captures (the +2.3 whole-flight median / +8.8 p90 up-bias — a speed-driven lift, NOT mixer
  rate-coupling: corr(|ω|,residual)=−0.23, flat); (ii) crucially, **the plant had huge unused
  down-authority MARGIN** — it was near NEUTRAL at floor collective, nowhere near a down-authority
  limit. The overshoot persisted because the FLOOR CAPPED the command, not because the plant ran out
  of push.

**Verdict — the floor cap, not the plant, was the binding constraint.** The gate-PD's OWN worst-case
demand (z_off at +3 → −0.12, `vz_lp` at the −2.5 climb rail → −0.15) is
`hover − 0.12 − 0.15 ≈ 0.00` — **the PD was ALREADY asking for ~0.0 collective and being denied
0.15 of it by the floor.** Opening the floor lets the bounded PD reach its own designed maximum
descent, and — key safety property — **the collective can never go below what the bounded PD asks**
(z_off is clipped ±3, gains are fixed), so there is no wide-open slam surface even in the unmeasured
region. The fix:

**(a) Open the floor 0.15 → 0.05** (`controller_overrides`, `alt_thrust_lo`): exposes the net-down
region the PD's worst-case (~0.0) demand needs, while staying just above a fully-open 0.0 (a hair of
margin against any non-PD path). The PD's ±3/gain bounds cap the WORST commanded descent at ~0.0
collective; the A27 slew (b) ramps INTO it. Not lower, because below the PD's own worst-case demand
there is nothing for the extra range to do except expose an unmeasured slam surface to a
hypothetical future path.
* **Worst-case survivability check (commander-required (b)):** the PD reaches 0.0 collective only
  when z_off=+3 AND vz_lp=−2.5 simultaneously (above the gate AND climbing hard) — precisely the
  overshoot we are trying to arrest. Realized descent there is bounded by the plant's net-down at
  ~0.0 collective, which is UNMEASURED — hence FC-10 (below) and the slew ramp (c). The steady-state
  descent the PD settles to (z_off=+3, vz→steady) is `(Kp/Kd)·3 = 2.0 m/s` by the A28 sizing —
  survivable and unchanged from A28's design intent; only the transient authority to REACH it widens.

**(b) NOT a hover recalibration (REFUTED).** `hover_thrust=0.2656` stays; the +2.3 up-bias is
translational lift, addressed upstream by S-1 (less speed) and V-1 (never over-climb into it). FC-7
measures the lift-vs-speed relationship for the next iteration.

**(c) Collective slew — KEEP the A27 up-limit, ramp INTO the new down-authority (commander-required
(a)).** The A27 `alt_thrust_slew_per_s` (2.0/s, symmetric) is RETAINED unchanged: it is now
load-bearing as the guard that we **ramp into the unmeasured net-down region rather than STEP to
it** — an uncommanded dip to the opened floor can no longer slam in one tick; it ramps over
~(0.15/2.0)=75 ms+. Do NOT add the down-fast asymmetry I floated in the prior draft: in an
UNMEASURED down-region, faster-into-the-floor is the wrong direction — conservative ramp wins.
(If a future measured low-collective sweep characterizes the region as gentle, revisit.)

**(d) FLAGGED SYSID GAP (future task, non-blocking):** we have ZERO measured data on collective→
net-down below 0.15. A dedicated low-collective sysid sweep (hold 0.12, 0.08, 0.05, 0.02 at level,
measure realized a_up) should characterize this region properly — the real max down-authority, the
knee, and whether it is gentle or violent. Until then V-2 operates conservatively (ramp + FC-10
watch) inside it. This gap is why FC-10 exists and why the floor is 0.05 not 0.0.

### S-1. Sharpen the off-axis forward cut: `fwd_scale = cos(az)^4` — `gate_seeker.py`

**Change:** new `GateSeekerConfig.fwd_scale_pow: float = 2.0` (vq2_case_c: 4.0);
`_compose_image_servo_accel` uses `max(cos(az), 0.0)**fwd_scale_pow`.

Rationale (operator, twice: too fast): at az 42° the cos² scale still drives 55% forward while
badly mis-pointed; cos⁴ gives 30%. Near-centered flight is barely touched (az 0.2 rad: 0.92 vs
0.96) so lap pace is preserved; the change bites exactly in the point-before-pushing regime — and
under H-1's turn-now it is what keeps the immediate turn from ALSO being an immediate lunge.
`forward_accel_mps2` stays 0.8.

**S-1 ↔ V-2 linkage (added per amendment-2 analysis):** the overshoot's +2.3 m/s² (p90 +8.8)
up-bias is TRANSLATIONAL LIFT — it scales with carried forward speed. So S-1's off-axis speed cut is
NOT merely a pointing fix; it is also the primary UPSTREAM mitigation for the vertical up-bias that
V-2's floor fights downstream. Less speed through the turn → less lift → less overshoot to arrest.
The two changes compound: S-1 reduces the disturbance, V-2 gives the loop authority to reject what
remains.

### Explicitly NOT changed (with reasons)

* **Asymmetric climb clamp — CONSIDERED AND DROPPED (operator Q2 ruling):** high gates exist
  (gate 2 itself; significantly higher later), so climb authority stays fully symmetric at
  ±`gate_pd_z_off_clip_m`=3. The overshoot is fixed at the source (V-1 + H-3 + V-2). Future note:
  the 3 m clip / 2 m/s steady climb may need RAISING for the later high gates — revisit when one
  is reached, not now.
* **Huber sigma/k, zoff alpha/beta, dt floor** — honest-latch weights already 0.87/1.0; garbage
  never reaches the filter once V-1 + H-3 land. Re-tuning now would mask whether the structural
  fixes worked.
* **`image_kaz`, `image_lat_cap`, yaw slew 1.5** — measured healthy (sign 100%, saturation 31%,
  turn at the slew cap when free to run).
* **kp_gate / ff_vertical_kd_alt** — the PD commanded the floor correctly during the overshoot;
  the miss was entry momentum + authority/physics, both addressed upstream.
* **bridge/ZOH machinery** — the cmd 166–184 frozen garbage demand is starved at the source by
  H-3/V-1 (the pose would have been range-rejected; the z_off state never teleported).

---

## 2. deploy_profile.py delta (single audit point)

```python
# vertical_estimator_overrides += {"zoff_reseed_min_w": 0.3}                    (V-1)
# seeker_overrides += {"pass_exclude_prev_gate": True,                          (H-1a)
#                      "pass_wire_coast_s": 0.25 -> 0.0,                        (H-1b)
#                      "pass_coast_s": 0.3,          # field default stays 1.2  (H-1b)
#                      "pass_turn_through": True,                               (H-1c, amendment 1)
#                      "soft_range_hard_reject": True,                          (H-3)
#                      "fwd_scale_pow": 4.0}                                    (S-1)
# controller_overrides: "alt_thrust_lo": 0.15 -> 0.05                            (V-2a, amendment 2)
#   A27 alt_thrust_slew_per_s stays 2.0 (symmetric) — ramps INTO the opened down-region (V-2c)
#   hover_thrust UNCHANGED (0.2656) — refuted as the +2.3 up-bias source                (V-2b)
#   NOTE: 0.05 exposes an UNMEASURED net-down region (see V-2d flagged sysid gap); FC-10 watches it
```

## 3. Byte-identity plan

* Every new field defaults to the legacy value (`0.0`, `False`, `2.0`); VQ1/case-A never construct
  the estimator, never enable the soft path, never set the flags — bit-identical. `pass_coast_s`
  and `pass_wire_coast_s` change ONLY in the vq2 profile override dict, never at the field.
* The A32 soft path exists only under `use_soft_bearing_weight` (vq2-only); H-3 gates further
  inside it. `pass_exclude_prev_gate` is checked only while `self._passing` and only filters the
  candidate list — flag-off is a no-op filter. `pass_turn_through=False` leaves `_pass_coast_command`
  freezing the heading exactly as today; the blind-turn slew only exists on the flag-on branch.
  V-2 touches NO code — it is a single override value (`alt_thrust_lo` 0.15 → 0.05); the A27 slew
  stays symmetric at 2.0, so `_apply_thrust_slew` is unchanged.
* Regression pins to add: (1) `zoff_reseed_min_w=0` reproduces A32's reseed byte-for-byte on this
  run's recorded latch stream; (2) `soft_range_hard_reject=False` reproduces the A32 track trace
  on this run's pose stream; (3) `fwd_scale_pow=2` == today's cos²; (4) `pass_exclude_prev_gate=
  False` == today's acquisition on a replayed pass sequence; (5) sign/geometry pin for the
  exclusion cone (a gate 85° off the pass heading at any range is NEVER excluded; a dead-ahead
  gate at snapshot-range−1 ALWAYS is); (6) `pass_turn_through=False` reproduces today's frozen-heading
  `_pass_coast_command` on a replayed pass; and a sign pin that the blind-turn target latches TOWARD
  the last pre-pass gate-2 bearing (right glimpse → right turn), clamped to ±`pass_blind_turn_cap_rad`.
  (V-2 needs no regression pin — it is a value override, not a code path.)
* Offline replay before flight: re-run this run's pose/latch streams through the patched seeker +
  estimator and confirm: no cmd-67/164 teleports; gate-2 track lock within ~0.2 s of cmd 49;
  exactly one pass commit for gate-1; no pass_vis stall.

## 4. Pre-registered flight checks (next flight, in order)

* **FC-1 (liftoff regression guard):** smooth liftoff, no dip — unchanged from A32.
* **FC-2 (gate-1 thread under turn-now):** the drone clears gate-1's opening while already
  turning — no contact with the gate-1 frame (operator's eyes). Log check: no contact-freeze
  event within 1 s after the gate-1 wire advance.
* **FC-3 (one pass per gate):** exactly ONE pass commit per gate index in the log (was: wire at
  cmd 49 + spurious vis at cmd 72 for the same physical gate).
* **FC-4 (turn latency — the operator's headline ask):** the yaw slew BEGINS within 0.15 s of the
  pass commit (`seeker_regime` leaves the frozen-heading coast into a slewing turn — measured as
  |d(yaw)/dt| > 0.3 rad/s starting ≤ 0.15 s after commit), NOT gated on gate-2 acquisition; wire
  advance → |az| < 0.2 rad ≤ 2.0 s (was 4.1 s); no frozen-yaw stretch > 0.3 s anywhere after the
  wire advance.
* **FC-5 (no garbage teleport):** zero z_off steps > 1.2 m/tick whose latch frame carried
  w < 0.3 (was: 2 pre-crash, both flight-shaping).
* **FC-6 (altitude sanity / no overfly):** during gate-2 pursuit, z_off_est within ±3 m ≥ 90% of
  ticks; no |offset_z_world| > 8 m before any contact event; bearing_w ≥ 0.3 on ≥ 70% of pursuit
  ticks until range < 4 m (the gate STAYS IN VIEW = not flying over it). Note: gate 2 is a HIGH
  gate (operator) — a sustained early climb is EXPECTED and correct; the check is that the climb
  ends at gate height, not above it.
* **FC-7 (arrest + translational-lift characterization — operator amendment 2):** TWO measurements.
  (i) *Arrest:* on the next overshoot, does the opened floor (0.05) now arrest the climb? Log check:
  `vz_est` reaches ≥ 0 (climb stopped) within 1.0 s of the PD first commanding ≤ 0.10 collective. If
  it does → the opened floor sufficed. (ii) *Lift-vs-speed:* the +2.3 up-bias is translational lift,
  not a hover offset (measured map recovers g to 2.3%) and not rate-coupling (corr −0.23). Correlate
  the collective-unexplained a_up residual (`−d(vz_est)/dt − (measured-map a_up at the commanded
  collective)`) against a forward-speed proxy (the integrated forward feedforward, or pitch-lean
  magnitude). A positive slope confirms translational lift → the lever is S-1 (less speed), already
  applied; log the coefficient to size whether a further speed cut is warranted. **Do NOT touch
  `hover_thrust`** (refuted).
* **FC-8 (the mission):** reach AND thread gate 2 (operator's eyes are the criterion); altitude
  stays gate-relative within a few m; no overfly-orbit.
* **FC-9 (occlusion no-stall — operator amendment 1):** across the ~0.2–0.5 s the gate-1 frame
  plausibly occludes the camera at the pass, the commanded yaw rate never drops to zero for
  > 0.15 s (the blind turn keeps slewing through the occlusion). Log check: no `seeker_regime` in
  {pass_wire, pass_vis} tick has |commanded yaw rate| < 0.1 rad/s while `_pass_turn_yaw` differs
  from the current yaw by > 0.1 rad. Operator's eyes: the nose keeps coming around through the
  gate, no visible "hang".
* **FC-10 (net-down watchdog — the UNMEASURED-region guard, operator amendment 2 caution):** V-2
  opens the floor into a collective region (< 0.15) our sysid never characterized. Watch the
  realized descent whenever the PD commands low collective: on any tick with commanded collective
  ≤ 0.10, if realized net vertical accel `−d(vz_est)/dt` exceeds a HARD-DOWN threshold (< −12 m/s²,
  i.e. faster than free-fall by > 2 m/s² — the plant is delivering strong net-down we didn't model)
  OR the drone over-descends past gate height (z_off crosses from + to strongly − while commanding
  down), FLAG IT. Two consequences: (1) it VALIDATES the net-down authority Fengyou described (good —
  we can then trust it); (2) if the descent is violent (< −15 m/s²) or drives a floor-smack, the
  floor was opened too far → raise `alt_thrust_lo` back toward 0.10 and prioritize the V-2d
  low-collective sysid sweep before opening it again. This is the pre-registered tripwire for the
  unmeasured region; the A27 slew (retained) is its first-line defense.

## 5. Evidence appendix (all from `nav_estimate.jsonl`, this run)

* Wire-leads-plane: cmd 49 advance with track range 5.4 EMA / raw 3.8–3.4 by cmd 54–56, az ≈
  0.01, ozw ≈ +0.06 (dead ahead, same height — cannot be gate-2, which needed an ~85° right turn
  and sat 4–5 m higher); operator threads + lands AFTER the advance.
* Track smear: rng EMA 2.8 → 3.6 → 4.7 → 7.0 → 8.3 → 10.8 → 11.4 (cmd 60–72) with bw 0.11–0.31;
  the bearing reference walks over (bw recovers 0.59 → 0.98 by cmd 74–88 on the SAME new gate).
* Reseeds (|Δz_off| > 1.2 m/tick): cmd 67 (−0.13→−5.00, frame w 0.28), cmd 164 (+2.82→−2.97,
  frame w 0.00, during the +27° pitch garbage stretch), post-crash cmd 220/390/396/404 — cmd 396
  re-locked at −21.97 on a w 0.01 frame.
* pass_vis stall: cmd 72–90 = 1.29 s (pass_coast_s 1.2 + tick), yaw frozen +47.2° vs yaw_des
  +86°, gate-2 bw 0.85–1.0 discarded throughout; vz −1.55..−2.03 (climb) all window;
  ∫vz ≈ +2.3 m altitude gained during the stall.
* Healthy turn: cmd 92–98 yaw_true 45° → 80° (up to 81°/s); az/alat sign agreement 100%
  (cmd 90–150); alat saturation 31% of finite ticks.
* Overshoot: z_off crosses 0 at ~cmd 135 still climbing; collective ≤ 0.20 from cmd 136, pinned
  0.150 cmd 150–162; mean commanded −3.7 m/s². `vz_est` (IMU-integrated) held ~−0.5..−1.0 m/s the
  whole cmd 100–158 window, overall slope +0.01 m/s² (climb did NOT decay under the floor brake).
  Collective-unexplained a_up residual: median +2.3 m/s², p90 +8.8; corr(|body_rate|, residual)
  = −0.23 (constant up-bias, NOT rate-coupled) — high-rate ticks +2.9 vs low-rate +2.0. The "+6"
  first read was a single-tick `offset_z_world` pose-noise artifact (ozw column swings ±5–15 m/s
  tick-to-tick). Honest-latch weights (pre-crash, |innov| < 1.4 m, n=81): w median 0.87, Huber
  factor 1.0 for 100% — the A32 soft weight did NOT throttle honest vertical corrections.
* Plant-map reconciliation (V-2): at clean pinned-floor ticks cmd 149–156, realized net vertical
  accel ≈ −0.55 m/s² (median) — vs the measured map's −7.68 prediction → ~+7 m/s² translational
  lift AND large unused down-authority margin (drone near-neutral, not at a plant limit). vz_est is
  ±2.5-clipped/CF-smoothed so the +7 is order-of-magnitude; the SIGN (rising at floor) is confirmed
  independently by vision ozw slope +1.4 m/s. Gate-PD worst-case commanded collective =
  hover − 0.12 − 0.15 ≈ 0.0 (bounded by ±3 z_off clip + gains) — the floor 0.15, not the plant, was
  the binding constraint.
* Chaos containment: all |ozw| > 8 m episodes start ≥ cmd 196 (post-floor-impact).

**Known plant gap (flagged, non-blocking — see V-2d):** collective→net-down below 0.15 is
UNMEASURED (`rl_plant.py:139-142`: sub-0.15 knots are floored-at-0 extrapolation, not sysid).
Fengyou confirms real net-down-faster-than-g exists there. Recommend a dedicated low-collective
sweep (hold 0.12/0.08/0.05/0.02 at level, measure realized a_up) as a future task; until then V-2
operates conservatively in this region (A27 slew ramp + FC-10 watchdog).

*Analysis scripts: session scratchpad `analysis/a33_analyze.py`, `a33_deep.py`, `a33_reseed.py`,
`a33_residual.py`, `a33_residual2.py`, `a33_plantmap.py` (the last three are the amendment-2
down-authority / plant-residual / measured-map reconciliation; all rerunnable against the run
directory).*
