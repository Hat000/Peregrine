# Live VERIFY flight (roadmap D) — UNDERSTANDING + pre-registration

**Session:** ShadowPC, co-located with `FlightSim.exe`, 2026-06-06. Branch `red-team-tier-a`
(merged `origin/red-team-tier-a` → HEAD contains `0425679`+: faithful twin + live config).
**Goal:** fly `FAITHFUL_TUNED_GAINS` on the real sim and answer three questions —
(a) does it **transfer**? (b) does it **hold altitude** (the balloon)? (c) does it **thread gate 0**?

This is the convergence point: an offline sim-faithful twin was fitted to the ShadowPC sysid data
(Task B) and the decoupled CTBR controller was re-tuned on it (Task C) → a live-ready config that
threads all 6 gates **offline**. Task 3 proved the altitude balloon is **OURS** (our alt-PD /
vz-damping / tilt-comp), not a sim auto-thrust — so if it shows up live, the fix is an **offline
re-tune on the twin**, not a live fight.

---

## 1. The config I will fly (single source of truth — no hand-copying)

Wired via `make_controller(signs=_FAITHFUL_SIGNS, **FAITHFUL_TUNED_GAINS)` +
`ReactivePlanner(**FAITHFUL_TUNED_PLANNER, yaw_mode="course")`
(`scripts/twin_fly_course.py` + `scripts/twin_tune.py`). **Verified offline this session** — the
constructed `Controller` prints exactly:

```
mode BODY_RATE (3) | decoupled True
hover_thrust 0.2656 | kp_pos 0.6  kd_vel 2.0  max_speed 6.0
kp_att 10.0  kd_att 0.15  ff_gain 2.5  max_body_rate_rps 8.0  max_accel_mps2 12.0
kp_alt 4.0  kd_alt 2.0  tilt_comp True  alt_thrust_lo 0.05  alt_thrust_hi 0.6  alt_offset_m 0.0
body_rate_sign [1, 1, -1]  odo_att_sign [-1, 1, 1]  odo_rate_sign [-1, -1, 1]
planner: lookahead_m 5.0  cruise_speed 8.0  yaw_mode course
```

This matches the handoff target line-for-line. `_FAITHFUL_SIGNS` carries `hover_thrust=0.2656`
and `ff_gain=2.5`; `make_controller` keeps the canonical `max_accel_mps2=12.0`,
`max_body_rate_rps=8.0`, clamp `[0.05, 0.6]`, `alt_offset_m=0.0` (defaults).

**The faithful plant** (`racer.twin_fit.faithful_config`, what the prediction is computed against):
`hover_thrust=0.2656`, `rate_tau_s=0.0190`, `rate_gain=[2.501,2.504,2.231]`,
`rate_sign=[+1,+1,−1]` (only yaw command physically inverted), `linear_drag=0.2111/s`,
`odo_att_report_sign=[−1,1,1]`, `odo_rate_report_sign=[−1,−1,1]`. Vertical model is one-parameter:
`a_up = g·(thrust/hover)` ⇒ at level `az_NED = g·(1 − thrust/0.2656) − drag·vz`.

### Live wiring (DONE this session — `scripts/fly_vq1.py`)
- **`--faithful`** builds the controller via `make_controller(signs=_FAITHFUL_SIGNS,
  **FAITHFUL_TUNED_GAINS)` and the planner via `ReactivePlanner(**FAITHFUL_TUNED_PLANNER,
  yaw_mode="course")` — imported from the sibling scripts, **no hand-copied gains** (single source
  of truth). `--print-config` prints the wired law and exits (the wiring-check artifact; verified to
  match the target line-for-line). 285 tests still green.
- **`--alt-thrust-hi 0.40`** caps the alt clip for rung 1 (the takeoff gentling). It overrides the
  faithful 0.6 only when given; `--alt-thrust-lo` similarly. Legacy path defaults unchanged (0.18/0.36).
- **`--hover-hold`** = rung-1 fixed-target hold (no gates). **`--max-tilt-deg`** = the >60° tilt
  auto-abort (the hard guard for the attitude rungs, per the GO). cmd-log now also records roll/pitch;
  meta.json records the full `controller_config` + bounds (self-describing for the laptop).
- **`--rate 100`** — the faithful gains were tuned at the live 100 Hz (dt=0.01); the rate-loop
  τ≈0.019 s is under-resolved at 50 Hz. Fly at 100 Hz.
- **Gate map:** the default `--map data/runs/track_map_20260602_114630.json` is **gitignored/absent**.
  The twin flew `handoff/shadowpc-firstcontact-2026-06-02/track_map.json` with `corner_to_center=True`.
  To match the twin exactly: `--map handoff/shadowpc-firstcontact-2026-06-02/track_map.json
  --force-saved-map --gate-corner-to-center` (the live TRACK_INFO reassembly has been corrupt;
  the deterministic saved map is ground truth). Confirm `corner_to_center` lift semantics at the
  gate-0 rung (Task 2: z-only lift to the opening, **no** half-width shift).
- **Recording (mandatory):** `--cmd-log` (+ the always-on `Recorder`: MAVLink + video + meta) so the
  laptop can overlay live-vs-twin and re-tune offline. Extract a compact cmd+telemetry JSON per rung.

---

## 2. Pre-registered twin prediction (computed + reproduced this session, deterministic)

### Gate course (faithful twin, dt=0.01) — reproduced via the exact `make_controller` wiring
- **FINISHED 6/6**, every plane crossed, **t = 31.9 s**.
- per-gate in-plane opening miss `[0.134, 0.61, 0.135, 0.264, 0.141, 0.053]` m (worst **0.61** at g1
  < 0.75 half-opening ⇒ valid). Matches the recorded Task-C result to 3 decimals.
- **Gate 0: closest approach 0.128 m, in-plane miss 0.134 m** → the "~0.13 m" pre-registration.
- g1 (first cross-track) is the marginal one (0.61 m) — a known VQ2 racing-line/RL target, not
  tightenable with this PD. **Honest caveat:** the faithful optimum is ~9× looser than the canonical
  twin (0.61 vs 0.069); the reactive lateral loop is delicate on the draggy/fast-τ faithful plant.

### Altitude / hover (the balloon) — twin holds, but **this is the genuinely uncertain part**
The twin's hover **equals** the controller's `hover_thrust=0.2656` by construction, and the twin does
**NOT** model the balloon → it holds **tautologically**. So the offline altitude prediction is a
*floor* (it can't reveal a live balloon), not a guarantee. Rung 1 is flown via the new `--hover-hold`
path (fixed position target at 1.5 m above start, no gates; see §1). Driving the real
Controller+CtbrPlant with that exact loop (100 Hz, **clip capped at 0.40** per the GO):
- **Climbs gently and settles exactly at 1.500 m** (the fixed target; `kp_alt` restores to it).
  Peak climb rate **+2.07 m/s**, reaches 1.45 m in ~1.9 s, **thrust → 0.2656 = hover** at steady
  state, vz→0. No overshoot, no bound trip. (Uncapped 0.60 contrast: peak climb +2.47 m/s, same
  1.500 m / 0.2656 hold — confirming the clip is a saturation limit that only bites the takeoff
  transient, not the hold we're verifying.)
- **The GO's gentling:** `--alt-thrust-hi 0.40` caps the takeoff at +5 m/s² instead of +12. **No live
  data exists above 0.27** (Task 3 probed 0.27→0.21 *down*), so even 0.40 is mild extrapolation of the
  2-point thrust slope; the 6 m bound + GUI guard a runaway. Raise toward 0.6 on a later rung once the
  real thrust response is seen.
- **Sign sanity (watch the GUI):** a wrong `body_rate_sign`/`odo_att_sign` shows as roll/pitch/yaw
  **drift or tumble even at hover**, before altitude matters. Expected: level, no rotation; the
  fixed-xy target also self-corrects any lateral drift (a lateral positive-feedback sign error would
  instead diverge → geofence abort).

### What would CONTRADICT a premise (→ hard stop + report)
- Live altitude **balloons or sags** at hover ⇒ `hover_thrust`/`kp_alt`/`kd_alt` or the 2-point
  thrust slope is off → **STOP, do not go to the gate**, re-tune the alt loop / re-fit the slope
  offline on the twin (Task-3 verdict: it's ours, fixable offline).
- Gate-0 live miss ≫ 0.13 m (e.g. the saga's "got close, then oscillate and miss") ⇒ the lateral
  loop doesn't transfer → report, re-tune offline.
- Anything that refutes "we own thrust in CTBR" (motors pin high / climb on a collective dip).

---

## 3. The VERIFY ladder (one rung at a time; HARD STOP + report between each)

**Bounds (abort + force-disarm):** altitude |z−start| > 6 m · tilt > 60° · any hard collision ·
time cap. `fly_vq1` enforces climb (`--max-climb-m 6`), geofence (`--geofence-m`), collision, and
`--max-seconds`; **tilt has no auto-abort** → teammate watches the GUI for >60°. Control only after GO.

1. **Hover-hold (balloon test).** `--faithful --hover-hold --alt-thrust-hi 0.40` (gentled clip):
   arm → gentle takeoff to a fixed 1.5 m target → hold for the duration. Bounds:
   `--max-climb-m 6 --geofence-m 4 --max-tilt-deg 60 --max-seconds ~12`. **Pass =** holds altitude
   (vz≈0, GUI level, thrust ≈0.266) AND no attitude drift/tumble. **Fail =** balloons/sags (alt loop /
   thrust slope) OR drifts/tumbles (sign error) → STOP, report, offline re-tune. *Twin
   pre-registration:* settles 1.500 m, thrust→0.2656, vz→0, peak climb +2.07 m/s.
2. **Gate-0 approach** (only on GO after rung 1 holds). Run the mission toward gate 0
   (`--max-gates 1`) with the faithful controller. Report **closest approach + pass/miss**
   (GUI-corroborated + the in-plane crossing). *Twin pre-registration:* closest **0.128 m**,
   in-plane **0.134 m**, pass.
3. **Course** (only if gate 0 threads). Continue the 6 gates, or stop and hand back — teammate's call.
   *Twin pre-registration:* 6/6, worst 0.61 m at g1, ~31.9 s.

**Discipline:** separate OBSERVATION from INTERPRETATION per rung; the GUI is the only mode
ground-truth; reproduce any interpreted result ≥2×; record everything; push to `red-team-tier-a`;
do not edit `memory/`.

---

## 4. Live-vs-twin gaps to keep in mind (where transfer could break)
- **Vertical velocity feedback:** live `fly_vq1` feeds the alt loop the **KF** vz (lagged), not the
  raw given vz (raw vz floors the thrust → balloon, per memory). The twin used the clean **true** vz.
  So the live `kd_alt` damping sees a slightly-lagged vz — a real difference at the alt loop.
- **Horizontal velocity:** live uses the **raw given** horizontal velocity (pristine; KF lags ~4×);
  twin used true world velocity. These should be close (both ≈ truth).
- **Thrust slope above hover:** unvalidated live > 0.27; the takeoff commands 0.6 (extrapolated).
- **Hover constant:** twin hover ≡ controller hover (0.2656); live hover may differ slightly →
  exactly what rung 1 measures.
- **Gate geometry:** `corner_to_center` lift + the gate-0 map anchor (bottom-centre, no half-width
  shift) — re-confirm at rung 2; the saga's gate-0 misses were the frame bug + gains, not the map.

**Status:** orientation + live wiring complete; offline prediction reproduced; `--print-config`
matches the target; 285 tests green. Awaiting GO to arm rung 1 (HARD STOP before arming).
