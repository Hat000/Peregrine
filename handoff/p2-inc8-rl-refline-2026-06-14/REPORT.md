# P2 INC8-RL — corrected-aero reference line rebuild for R1′ (Γ)

**Worker #2 · 2026-06-14 · branch `p2-inc8-refline` · model opus-4.8**
Deliverables: `rl/build_reference_line.py`, `rl/reference_line_inc8.json`, `tests/test_reference_line_inc8.py` (6/6 green; full suite 709 passed / 35 skipped / 0 fail). `rl/reference_line_vq1.json` UNTOUCHED.

---

## HEADLINE (read first — refutes a brief premise, with data)

The brief's **"honest lap bound ≈ 4.6–4.7 s" is the FULL-ATTITUDE (TOGT/proto "ball") bound, and it is NOT achievable upright or within the measured rate envelope.** Re-derived from data on the measured plant:

- Hitting ~4.6 s requires **sustained INVERTED thrust on the descent** — median body tilt **≈100°** through gates 1–3, **~50 % of the course at tilt > 90°** (thrust vector pointing below horizontal).
- It also requires **yaw body-rate ≈ 13.1 rad/s > the ~11 rad/s super-rate envelope** (yaw angular-accel ≈108 > 80 rad/s² envelope). So it fails the brief's own **test-4 rate criterion**.
- Root cause: **the MEASURED quad-drag is ~7× the linear plant at racing speed** (`c2·30² = 0.052·900 = 4.8 g` of drag at 30 m/s). Holding even *level* flight at 30 m/s needs ~78° tilt; the 16–19° descent then needs near-knife-edge or inverted thrust to sustain >30 m/s.

**The honest UPRIGHT, rate-feasible, collective-feasible lap is ≈ 7.9–8.5 s, not 4.6–4.7 s.** The emitted Γ is the margin-preserving upright line at **lap = 8.455 s**.

**Why this is safe for R1′:** the GEOMETRY (gate-centred cubic spline) is **identical** across every speed profile — ball or cone, 4.6 s or 8.5 s. R1′ is *arc-length progress along the geometry*; it never reads the speed/attitude profile. So the choice of certificate does **not** change the reward signal. What changes is only the honest `lap_time_s` + feasibility metadata and the implied inc8 speed expectation. **Action for commander: revise the inc8 upright speed expectation (~30 m/s / ~8 s), not 39 m/s / 4.6 s.**

I did **not** invoke the escape hatch: the geometry is reachable, contact-safe, and drag-feasible (a feasible upright profile exists). The infeasibility is specific to the *4.6 s aggressive profile*, which is surfaced here rather than shipped.

---

## What was built

`rl/build_reference_line.py` (offline, pure-numpy; no GPU/SLURM/sim):
1. **Canonical gates** — `_GATE_POS_ZUP * _FLIP` AST-sourced from `rl/fly_rl.py` (single source of truth, no torch import; never the json's `pos_ned`).
2. **Geometry** — natural cubic-spline (min-curvature) through `[spawn, G0..G5]` + an 8 m straight run-out, dense arc-length sampling. Interpolating *through* the centres ⇒ dead-centre crossings.
3. **TOPP** — corrected-aero forward-backward speed profile adapted from `proto_envelope_topp.py`, with the MEASURED envelope: convex collective ceiling (full-stick **78.28 m/s² = 7.98 g**, NOT the linear plant's 3.765 g), v²-quad-drag wall, and the body-tilt constraint. Two models:
   - **`cone`** (emitted): exact upright body-tilt cone `tilt(f_thrust) ≤ tilt_max` (no inverted thrust). On level flight it reduces *exactly* to the proto's `a_lat = g·tan(tilt)`; on slopes it correctly forbids the inverted-thrust the proto's ball admits.
   - **`ball`** (recorded for comparison): the proto/TOGT full-attitude model.
4. **Flatness inverse** — attitude (quat), body rates, collective from `f_thrust = acc − g + drag`; **acceleration is analytic Frenet** (`a_tan·t̂ + κv²·n̂`), finite-jerk smoothed — *not* `np.gradient(vel)`, which rang at the bang-bang accel↔brake kinks and injected spurious 30–170 rad/s rate spikes.
5. **Emit** — uniform-time (100 Hz) resample; `peregrine.reference_line.v1` schema with HONEST provenance (measured plant constants, T/W 7.98, the finding, and the ball-bound comparison block).

Run: `PYTHONPATH=src .venv/Scripts/python.exe rl/build_reference_line.py --emit` (also `--sweep`, `--model {cone,ball}`).

---

## Achieved metrics — emitted Γ (cone, tilt cone 75°)

| quantity | value | limit | margin |
|---|---|---|---|
| **lap_time_s** (last gate crossing) | **8.455 s** | — | honest upright |
| total_duration_s (incl. run-out) | 8.780 s | — | |
| peak speed | 28.92 m/s | < 39 m/s drag wall | ✓ |
| peak collective (normed) | 0.912 | ≤ 1.0 | ✓ 8.8 % |
| peak required \|f_thrust\| | 66.04 m/s² | ≤ 78.28 m/s² | ✓ |
| peak body rate — roll / **pitch** / yaw | 1.63 / **5.35** / 3.90 rad/s | ≤ ~11 (per axis) | ✓ |
| peak tilt | 74.98° | ≤ 75° cone | ✓ |
| frac inverted (tilt > 90°) | **0.0 %** | — | upright ✓ |
| samples (N) @ 100 Hz | 879 | — | |

**Per-gate crossing miss vs canonical centre** (in-plane Y–Z; target < 0.05 m, band < 0.37 m @ r=0.38):

| gate | miss_m | miss_h (E) | miss_v (D) |
|---|---|---|---|
| G0 | 0.0001 | −0.0000 | +0.0000 |
| G1 | 0.0001 | +0.0001 | +0.0000 |
| G2 | 0.0000 | −0.0000 | +0.0000 |
| G3 | 0.0000 | +0.0000 | −0.0000 |
| G4 | 0.0002 | −0.0002 | +0.0000 |
| G5 | 0.0000 | −0.0000 | +0.0000 |

**DEAD-CENTRE** (≤ 0.0002 m, ~5000× inside the contact band) ⇒ the full contact-true band is free at every gate.

**Orientation fix CONFIRMED** (the ~170° vq1 inversion is gone): X strictly decreasing (course runs −X); `velocity_ned · path_tangent > 0` at every sample (min 0.50 = the start v-floor); yaw ∈ [166.6°, 195.0°] (i.e. ≈ ±180°, the −X hemisphere — NOT ≈ 0°); `yaw ≡ atan2(vE,vN)` to < 0.001°; gate-approach velocity faces travel at all 6 gates.

---

## The data behind the headline (ball vs cone, finite-jerk-smoothed, per-axis rates)

| model | tilt | lap (s) | v_max | rate r/p/**y** (rad/s) | coll | tilt median (descent) | inverted % | verdict |
|---|---|---|---|---|---|---|---|---|
| **cone** (emitted) | 75° | **8.455** | 28.9 | 1.6/5.4/**3.9** | 0.91 | 74° | 0 % | ✓ feasible, margin |
| cone (upright edge) | 78° | 7.901 | 33.4 | 1.9/9.2/**5.5** | 1.00 | 78° | 0 % | ✓ feasible, no margin |
| **ball** (TOGT/proto) | 75–90° | 4.58–4.69 | 39.2 | 1.7/3.4/**13.1** | 1.00 | **~100°** | **~50 %** | ✗ yaw>11; inverted |

The collective ceiling binds hard at cone tilt ≥ 76° (coll → 1.0); 75° is the last cap with full thrust + rate margin, hence the emitted operating point. The C++ TOGT corrected-aero refined 4.714 s ≈ the ball-90° bound (4.58 s) — confirming 4.7 s *is* the full-attitude bound, and that it carries the inverted-descent + yaw-rate infeasibility.

---

## Footgun checks honored
- Did **not** trust vq1's numbers — schema layout only; vq1 file untouched (provenance/comparison).
- Did **not** use `linear_drag 0.21` — feasibility built on `rl_plant.py` MEASURED super-rate + quad-drag + convex collective.
- Gate centres from `_GATE_POS_ZUP*_FLIP` (AST), not the json; pinned by an independent AST check in the tests.
- Lap **8.455 s > 4.7 s** — not re-importing the infeasibility (a test asserts `lap > 4.7 s`).
- "T/W 3.765" recognized as the LINEAR plant's full-stick (g/hover); provenance records the measured T/W ≈ 7.98 and the feasibility ceiling 78.28 m/s².

## Suggested follow-ups (commander's call)
- **Decide the inc8 speed target** against the ~8 s upright reality (or accept aggressive near-knife-edge ≥78° tilt for ~7.9 s). The geometry is unaffected either way.
- If R1′ wants the file's profile to advertise the aggressive bound, `--model ball` re-emits it — but it is rate-infeasible (yaw 13>11) and inverted; not recommended as a tracking target.
- Minor: a true min-snap (vs natural cubic) would trim the 19.5° descent overshoot toward the 15.6° gate-mandated mean — small lap gain, does not change the finding.

---

## MEMORY-DELTA (≤10 lines — commander banks; I did not touch memory/)
- 🚩 **inc8 R1′ reference line REBUILT on measured plant → `rl/reference_line_inc8.json`** (branch `p2-inc8-refline`, NOT merged). Generator `rl/build_reference_line.py`, tests green (6/6; suite 709/35skip/0fail). vq1 untouched.
- 🚩 **FINDING refutes brief premise:** the "honest 4.6–4.7 s lap" is the **full-attitude/TOGT bound** — needs **sustained INVERTED descent (median tilt ~100°, ~50% course tilt>90°) + yaw-rate ~13 > 11 rad/s** (rate-infeasible). NOT a sane/feasible upright reference.
- 🚩 **Honest UPRIGHT, rate+collective-feasible lap ≈ 7.9–8.5 s** (emitted Γ = 8.455 s, tilt cone 75°, margin: coll 0.91, pitch-rate 5.4, |f| 66/78, 0% inverted, v_max 28.9). Upright edge 7.9 s @78° (rides collective ceiling).
- 🚩 **Cause = MEASURED quad-drag ~7× linear at 30 m/s** → upright racing tops ~30 m/s on this descent course (level-flight at 30 m/s already needs ~78° tilt). **Revise inc8 speed expectation (~30 m/s / ~8 s, not 39 / 4.6).**
- ✅ Geometry (gate-centred cubic spline, dead-centre ≤0.0002 m, orientation fixed: yaw≈±180°, vel down-course) is **identical across all speed profiles** → R1′ arc-length reward unaffected by the speed choice. Canonical gates AST-sourced from `fly_rl._GATE_POS_ZUP*_FLIP`.
- NEW parked candidate: min-snap vs natural-cubic descent-overshoot trim (minor; does not change the finding).
