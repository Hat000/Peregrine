# VQ2 A34 — A33 Pose-Feed Regression: Diagnosis + Absolute-Range-Cap Fix (2026-07-03)

A33 (commit `ac11e63`) flew run `data/runs/20260703_223956_rl_s1_f1` and REGRESSED vs the A32
baseline `20260703_210632`. Operator eyes (ground truth): *"video laggy / choking frames to vision
again; hit the ground after takeoff again; threaded the gate but went STRAIGHT FOR CEILING, NO TURN
AT ALL. Step in the wrong direction."*

**Review BEFORE building.** This is a resume of the A33 context; all analysis below was run this
session against the two run directories.

---

## 0. Executive summary

The regression is **NOT pose-age starvation and NOT a loop choke.** It is a **garbage far-range
track lock** created by A33's H-3 relative range wall interacting with first-acquisition's permissive
fallback, which then **strangled recovery**. Three proven facts:

1. **pose_age_s = 1.0 is a LOGGING/CLOCK ARTIFACT, not real starvation.** The real pose age is
   ~100 ms (the async-detect `age_ms()` the flight console prints), video frames flow at 17 fps with
   no mid-flight choke, and A33 **did not touch `navigator.py`** where the clock reconciliation
   lives. The `pose_age_s` field is computed via the camera↔IMU epoch reconciliation, which
   silently railed to the 1.0 s cap this flight — same code as A32 (0.13 s), opposite result,
   because A33 changed the *flight dynamics* (the drone stalled/climbed) enough to degrade the
   epoch-delta pairing. **Do not chase pose_age; it lies this flight.** (This is the exact
   ambiguity flagged in A28.)

2. **The real control failure (logging-independent): the drone acquired gate-1 dead-center at ~10 m
   with a PERFECT lock (bearing_w=1.0, az≈0, fwd_scale≈1.0) but never translated forward.** Range
   crept 10.6 → 6.4 m then FROZE at exactly 6.4279 m for 4.4 s (cmd 162→264, offset_z_world frozen
   too) while the drone rode at ~4 m altitude, then the track JUMPED to 52 m (cmd 264) and smeared
   48-52 m for the rest of the flight. It never closed on gate-1; no turn ever fired (0 pass_wire /
   0 pass_vis regimes, 0 blind-turn latches — H-1c never executed).

3. **The 52 m jump is the culprit event, and it is an ABSOLUTE-range problem the RELATIVE cap made
   worse.** track_range read **>32 m on 42% of ticks (up to 52 m)** — physically impossible on this
   course (max inter-gate spacing 38.5 m, usable PnP detection ~24-32 m; `gate_mapper.py:36`). Once
   that garbage entered the track, A33's H-3 relative wall (`|range − pred_r| > 6 m`) REJECTED every
   fresh close 6 m pose (`|6 − 49| = 43 m > 6 m`) — **permanent starvation, cannot recover.** The
   absolute cap accepts the 6 m pose (`6 ≤ 32`) and re-locks the near gate.

**Verdict on the coordinator's leading hypothesis (H-3 over-rejects relative to a smeared pred_r):
CONFIRMED as the *recovery-strangling* mechanism** — but with an important refinement (below): H-3
did not *cause* the smear, and it never could have caught the *original* A32 smear either.

---

## 1. The mechanism, proven step by step

Timeline (run `20260703_223956`, cmd = tick):

* cmd 45-150: healthy pursuit of gate-1, dead center (az≈0.00, bw 0.83-1.0, fwd_scale≈1.0), range
  closing 10.6 → 6.4 m. Thrust ~0.27 (just above hover 0.2656), pitch −4°. **The drone barely
  leans forward and closes slowly.**
* cmd 87: RACE_STATUS index advances 0→1 while the drone is at range 10.46 m (the wire leads the
  physical plane, as in A32). `_pass_armed` is False (range never < `pass_arm_range_m`=3.0), and
  the pass either never commits or commits-and-instantly-ends under H-1b's `pass_wire_coast_s=0.0`;
  either way `pass_wire` logs False and the regime stays `pursuit`. **No pass, no turn.**
* cmd 150-264 (t 7.1-11.5 s, 4.4 s): **offset_z_world and track_range FREEZE** at exactly
  −1.6996 m / 6.4279 m — a Zero-Order-Hold. Regime goes pursuit(frozen) → bridge → hold. **75
  fresh video frames arrived during this window (17 fps)** — the freeze is NOT a frame choke. The
  detector went dark or its candidates were rejected: the drone sat at ~4 m altitude with the
  +20°-up camera, gate-1 opening below the flight path, plausibly out of the up-tilted FOV.
* cmd 264: track JUMPS 6.4 → 52.2 m. The near track had coasted out (dropped after
  `track_max_coast_ticks`), so `_first_acquisition` ran on whatever the detector then saw — a lone
  distant object at 52 m — and its **permissive fallback** (`if not admissible: admissible = poses`,
  gate_seeker.py:1205-1206) LOCKED it despite `max_acquire_range_m`=22. From here the track is
  garbage; the relative wall rejects every close pose; the drone never recovers and env-collides
  at ~19.8 s.

### 1.1 Refinement the coordinator should know (changes the fix rationale)

**H-3's relative range wall could NEVER have caught the original A32 gate-1→gate-2 smear.** That
smear (run 210632, cmd 60-72) walked the range EMA 2.8 → 11.4 m in per-frame steps of only
**0.8-2.5 m — all UNDER the 6 m jump threshold.** The A32 smear was the *soft-bearing-weight EMA
walking the track across gates*, not a range JUMP. So H-3 as shipped:
* did NOT stop the problem it was built for (the gradual within-30 m smear), and
* DID introduce the far-range starvation that killed this flight.

This is why the fix is not "tune H-3" — it is "replace H-3's relative wall with an absolute cap,
and address the gradual smear separately (it needs a different guard, see §2.3)."

### 1.2 H-1a (geometric exclusion) — secondary suspect: REFUTED as the cause

H-1a only ever runs while `_passing` with a snapshot present. This flight had **0 committed pass
regimes** (0 pass_wire, 0 pass_vis ticks), so H-1a's `_is_prev_gate` filter never fired in flight.
It did not cause the freeze or the 52 m lock. (It remains wired for when a real pass commits; no
change proposed to it here.)

---

## 2. THE FIX — absolute course-informed range cap (A34)

Replace the fragile pred_r-RELATIVE range wall (H-3) with an ABSOLUTE cap at candidate admission,
applied on BOTH acquisition paths (first-acquisition AND continuity), as a HARD discard. A reading
beyond the cap is physically impossible on this course = a different object / mis-depthed garbage —
the category where hard-reject is correct (this does NOT contradict Fengyou's weight-don't-discard
principle, which governs *plausible* measurements).

### 2.1 The cap value — course geometry

> **[CONFIRM WITH FENGYOU: true max gate spacing on the VQ2 course.]** The repo documents the
> **VQ1** course (`gate_mapper.py:36`): inter-gate spacing **23.7-38.5 m**, usable PnP detection
> **~24-32 m**. Fengyou's estimate for VQ2 is ~30 m max. These mostly agree, but the VQ1 g2→g3 leg
> is 38.5 m (beyond usable detection — "possibly never co-visible"). Two safe readings:
> * If VQ2 max spacing ≈ 30 m (Fengyou): cap at **35 m** (his estimate + margin, still at the
>   usable-detection ceiling).
> * If VQ2 inherits VQ1's 38.5 m leg: a legitimate far gate could read ~35 m, but a pose >32 m is
>   beyond reliable PnP range anyway (close-range over-reporting + far-range noise), so **35 m** is
>   still the right cap — it discards the 48-52 m garbage while admitting anything the PnP can
>   actually trust.

**Proposed: `track_abs_range_cap_m = 35.0`** (a config field, default `None` = OFF = byte-identical;
vq2_case_c sets 35.0). 35 clears every legitimate close pose this flight (6.4-10.7 m) with 3x
margin and discards 100% of the >48 m garbage. If Fengyou wants it tighter to 32, that is a
one-value change and still passes every legit pose.

### 2.2 Where it applies — `gate_seeker.py`

New `GateSeekerConfig` field:
```python
# A34: absolute course-informed range cap (m). A candidate whose PnP range exceeds this is
# physically impossible on the course (max gate spacing ~30-38 m, usable PnP ~32 m) -> a different
# object / mis-depthed garbage: HARD-discard at candidate admission, on BOTH acquisition paths.
# Replaces A33's pred_r-RELATIVE range wall (which strangled recovery once pred_r smeared far and
# never caught the gradual within-range smear anyway). None (default) = OFF = byte-identical (VQ1 /
# case-A). vq2_case_c: 35.0.
track_abs_range_cap_m: float | None = None
```

Applied ONCE, at the top of `detect_gate_lever` right after `poses = self._valid_poses(frame)`
(before the H-1a exclusion, before the first-acquisition / continuity split), so EVERY downstream
path — first-acquisition, the permissive fallback, continuity, the soft path — only ever sees
in-range candidates:
```python
if self.config.track_abs_range_cap_m is not None:
    poses = [p for p in poses if float(p.range_m) <= self.config.track_abs_range_cap_m]
    # empty -> normal no-candidate coast tick (valid_poses_empty), never locks garbage
```
This closes the first-acquisition permissive-fallback hole (`if not admissible: admissible = poses`
can no longer resurrect a >35 m garbage candidate — there are none in `poses`).

### 2.3 REMOVE H-3, address the gradual smear separately

**Turn H-3 OFF** in vq2_case_c (`soft_range_hard_reject: True → False`, i.e. drop it from the
override dict). Rationale: it never caught the gradual smear it was built for (§1.1) and it caused
the far-range starvation. The absolute cap (§2.2) is the correct range guard.

The **original A32 gradual within-30 m smear** (the gate-1→gate-2 track walk) still needs a guard —
the absolute cap does NOT catch it (both gates < 35 m). Options, in order of preference:
* **(A) rely on H-1a's geometric old-gate exclusion + a wire-pass commit.** In A32 the smear
  happened *because the pass never cleanly committed* and the acquire-next never re-locked cleanly.
  With the A33 pass machinery (H-1a exclusion + turn-through) working on a *correctly localized*
  approach (which the absolute cap now enables), the gate-1→gate-2 handoff goes through the pass
  path, not a silent EMA smear. **This is the preferred path: fix localization first, re-fly, and
  check whether the smear even recurs.**
* **(B) if the smear recurs, add a bounded SECONDARY relative check INSIDE the cap** — but note it
  must be a *cumulative-drift* guard, not a per-frame jump (the smear's per-frame steps were
  0.8-2.5 m, under any sane jump threshold). E.g. reject a candidate whose range differs from the
  track EMA by more than K meters *and* whose bearing is also drifting — the two-signal version.
  DEFER this until a re-fly shows it is needed; do not pre-build a guard for a problem the
  localization fix may dissolve.

**Recommendation: ship the absolute cap + drop H-3, re-fly, and decide on (B) from the data.** A
working pose feed beats a clever filter that starves it (the coordinator's own framing).

### 2.4 What else to REVERT or KEEP from A33

* **V-1 (weight-qualified reseed): KEEP.** It held perfectly — 0 z_off teleports, max jump
  0.554 m. No downside observed.
* **H-1a (geometric exclusion): KEEP** (never fired this flight; needed when a real pass commits).
* **H-1b (window collapse) + H-1c (turn-through) + V-2 (open floor): KEEP but INCONCLUSIVE.** The
  turn never fired (no localization) and thrust never went below 0.238 (V-2 floor never exercised).
  Do NOT judge or revert them on this flight; they get a fair test once the pose feed is fixed.
* **S-1 (cos⁴): KEEP** (harmless; not implicated).
* **H-3 (relative range wall): REMOVE** (§2.3) — the one A33 change that regressed.

So A34 is a **surgical one-in-one-out**: add the absolute cap, remove the relative wall. Everything
else A33 shipped stays for a fair re-test on a healthy feed.

---

## 3. Offline verification (all run this session against the two run dirs)

* **Real-vs-artifact:** offset_z_world changes 273 times (fresh poses every ~1.7 ticks); async
  console age ~100 ms; video 17 fps no choke; navigator.py untouched by A33. → pose_age=1.0 is a
  clock artifact. ✔
* **(a) absolute cap discards the 52 m garbage:** admission test — 52.2 m and 68 m REJECTED by the
  35/32 cap; 6.5 / 11 / 30 m ACCEPTED. Flight: 189/408 finite-range ticks (42%) read >32 m — all
  discarded by the cap; the 219 legit ticks (6.4-10.7 m) all kept. ✔
* **(b) does NOT reject the close poses the relative cap killed:** once the track smeared to 49 m, a
  fresh 6 m pose is `|6−49|=43 m > 6 m` → RELATIVE rejects (starves), ABSOLUTE accepts (`6 ≤ 35`) →
  re-locks the near gate → pose feed recovers to A32-like freshness. ✔
* **(c) within-30 m smear:** the absolute cap does NOT catch it (both gates < 35 m); the original
  A32 smear was gradual (per-frame 0.8-2.5 m, sub-jump-threshold) so H-3 never caught it either.
  Handled by §2.3 option (A) [re-fly and observe] with (B) as a deferred fallback. ✔ (characterized,
  not yet re-verified in flight — flagged honestly)
* **Culprit isolation:** first-acquisition's permissive fallback (`if not admissible: admissible =
  poses`) is the code that locked the 52 m; the absolute cap upstream removes all >35 m candidates
  so the fallback has nothing garbage to lock. ✔

Analysis scripts (session scratchpad, rerunnable): `a33fly_char.py` (real-vs-artifact + A32/A33
compare), `a33fly_trace.py` (timeline), `a33fly_repro.py` (seeker repro), `a33_abscap_replay.py`
(admission test + distribution), plus the frame-cadence checks.

---

## 4. Byte-identity + build plan (for after review)

* `track_abs_range_cap_m` defaults `None` → the filter line is skipped → VQ1/case-A byte-identical.
* Removing `soft_range_hard_reject` from vq2_case_c's overrides reverts to the A32 soft path (the
  field default is already `False`, so the code path is unchanged; only the profile stops setting
  it True). The `soft_range_hard_reject` field + code can STAY in the tree (dormant) for A/B, or be
  removed — reviewer's call; leaving it dormant is lower-risk.
* Tests: add a pin that `track_abs_range_cap_m=None` is a no-op; a pin that a 52 m candidate is
  discarded and a 10 m one kept; update the vq2_case_c profile-wiring assertions (soft_range_hard_
  reject no longer in the dict; track_abs_range_cap_m=35.0 is). Full suite green (the 11 pre-existing
  baseline failures excepted, as in A33).
* Offline gate before flight: re-run this run's range stream through the patched admission and
  confirm 0 ticks lock >35 m; confirm the close-pose acceptance rate returns to A32 levels.

---

## 5. DEFERRED (operator: do NOT build now) — bounded TWO-AXIS gate-direction memory for H-1c

Fengyou's idea: at flight start gate-2 is visible off to one side (the second-largest gate);
REMEMBER its direction so the post-pass turn commits correctly even through occlusion. **Marked
DEFERRED per operator — do NOT build in A34.** Scope recorded here for when it is approved:

* **What it is:** a bounded latched **TWO-AXIS** direction memory — **left/right AND up/down** (is the
  next gate higher or lower) — captured once from the pre-race multi-gate view, consumed by H-1c's
  blind-turn-target as the direction of last resort. **Two-axis is required** (operator, 2026-07-03):
  gate 2 is a HIGH gate, so a side-only memory is half-blind — the drone must know to CLIMB toward it
  as well as which way to yaw. Today H-1c uses only the last pre-pass sighting's horizontal drift
  (degrading to hold-heading if passed dead-center) and has no vertical anticipation at all.
* **Map-free-roadmap tension (flagged honestly):** this is NOT VIO and NOT a map — it is two bits of
  coarse direction memory (a yaw sign + a pitch/altitude sign), well inside the map-free doctrine
  (no world positions, no landmark graph). But it IS a small step toward remembering *something*
  between gates, so it needs Fengyou's explicit OK rather than being slipped in.
* **Honest framing: this would NOT have fixed this flight** — we never reached the turn (no
  localization). It is a robustness upgrade to H-1c for a *future* healthy flight, orthogonal to
  the A34 pose-feed fix. Decide separately; do not couple to A34.

---

## 6. Open items

1. **VQ2 max gate spacing — RESOLVED (operator, 2026-07-03):** Fengyou "no idea" on the exact value,
   so we use the repo-derived cap **35 m** (PnP detection tops ~24-32 m → no legit pose beyond ~32 m;
   35 m discards this flight's 48-52 m garbage with margin and never rejects a legit close pose; bias
   deliberately toward NOT rejecting legit poses). **Cap is finalized at 35.0 m.**
2. **§5 two-axis gate-direction memory — DEFERRED** (operator); build only on a separate go.

---

## 7. Build result (A34, 2026-07-03) — one-in-one-out, offline-gated

* **ADDED** `GateSeekerConfig.track_abs_range_cap_m` (default `None` = OFF = byte-identical;
  vq2_case_c = 35.0), applied once in `detect_gate_lever` right after `_valid_poses`, BEFORE the
  H-1a exclusion and the first-acq/continuity split — so no downstream path (incl. first-acq's
  permissive fallback) can lock a >cap candidate.
* **REMOVED** `soft_range_hard_reject` from vq2_case_c (field stays dormant at its `False` default
  for A/B; code untouched).
* **KEPT** V-1, H-1a/b/c, V-2, S-1 (all A33 keepers).

**Offline gate on run 20260703_223956 (measured):**
* (a) 189/408 finite-range readings were >35 m (42%, all the 48-52 m garbage) → ALL discarded.
* (b) all 219 legit close readings (6.4-10.7 m) KEPT, 0 rejected.
* (c) driving the real seeker over the flight's range stream: max track prediction **10.7 m with cap
  ON** (tracks the real approach) vs **52.2 m with cap OFF**; far-locks (>cap accepted) **0 (ON) vs
  180 (OFF)**.
* (d) lone 52 m candidate through first-acquisition: **discarded, no lock (cap ON)**; **LOCKED
  (cap OFF)** — reproduces the bug.

**Recovery mechanism (surfaced during build, flagged honestly):** the cap's primary protection is
that pred_r can NEVER smear beyond the cap (every >cap candidate discarded at admission), so the
43 m relative-wall starvation gap the flight hit cannot form. If a *within-cap* held smear ever
forms, recovery is via track coast-out (`track_max_coast_ticks`=8, ~0.3 s) → first-acquisition
re-lock (which has NO relative wall, only max_acquire_range_m + the cap) — a bounded ~8-tick
recovery, not permanent starvation. The legacy continuity `track_max_range_jump_m` wall still exists
inside the cap but can no longer produce the far-gap that starved this flight. Within-30 m gradual
smear (original A32): handled by re-fly-and-observe (§2.3 option A); the deferred cumulative-drift
secondary (§2.3 option B) remains available if a re-fly shows it is needed.
