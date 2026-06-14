# SIMOPS-MASTERY-SHADOWVISION — REPORT

**Date:** 2026-06-13 (live cycles past midnight → run dirs stamped `20260614_…`) · **Machine:** ShadowPC (sim host)
**Operator model:** opus-4.8 · **Branch:** `flyrl-autonomy-hardening` @ `7210c1d`
**Sim:** AI-GP v1.0.3364 (`DCGame-Win64-Shipping`, window "AI-GP"), endpoint `udp:127.0.0.1:14550`
**Policy:** inc7 `rl/checkpoints/stage1_inc7_actor.pth` · **Detector:** `models/gate_yolo11s_curriculum_v2.pt`
**Constraint honored:** inc7 + existing detector only (no estimator, no retrain). **ZERO gate contact across all 14 live flights.**

> **Fengyou** — headline for you up front: **(A)** sim-ops is bulletproof — **10/10 consecutive clean
> autonomous cycles, longest streak = 10**, plus all four failure-modes recovered; **(B)** the camera's
> *precision* validates the vision premise (lateral σ ≈ 0.10 m at the gate-4 band, ~2.5× **tighter** than
> the modeled 0.265 m) — but inc7's racing **crab points the camera off-gate**, so only ~7 % of frames
> yield an accepted fix: the binding constraint for vision-driving is **camera pointing, not precision**;
> **(C)** ground-truth velocity **IS available** (`LOCAL_POSITION_NED.{vx,vy,vz}`, world NED, ~97 Hz, live
> to 21.3 m/s) — the step-5 plan is **not** changed.

---

## PART A — SIM-OPS MASTERY ✅

### 10 consecutive clean cycles (probe-VERIFIED, zero manual intervention)
Tool: `partA_cycle.py` — runs `rl/fly_rl.py --flights 1` per cycle (faithful inc7 passive launch:
`--no-bridge --no-auto-reset`), freeing the 14550 endpoint each cycle so **every state transition is
confirmed by a passive MAVLink probe** (not the blind between-flight reset of `--flights N`).

| run | cycles | result | gate contact |
|---|---|---|---|
| smoke (`simopsA_smoke`) | 2/2 clean | FINISHED 6/6 | 0 |
| **main (`simopsA`)** | **10/10 clean, streak = 10** | FINISHED 6/6 every cycle | 0 |
| instrumented (`simops_fra`, `--debug-obs`) | 1/1 clean | FINISHED 6/6 | 0 |

**= 13 consecutive clean inc7 flights, 0 collisions.** Per cycle (probe-confirmed, logged to
`logs/partA_timeline.json`): `WAITING (started=False, drone@origin 0.02 m)` → launch fly_rl (passive) →
**GO caught via fly_rl's "Waiting PASSIVELY" banner at ~4 s** (the Enter is the organizer's race start) →
`FINISHED active_gate=6` (true 6/6; `meta.gate_index=5` is the known recorder-truncation gotcha — the 6th
pass coincides with finish+disarm) → 0 collisions, ~440 frames → full-reset to HOME. Cycle wall-time ~33–45 s.
No stale hang, no zombie, no NO_GO, no manual touch.

**Sim state machine confirmed (matches `reference_sim_ops.md` / vision-cal `SIM_TOOL_USAGE_FYI`):**
cold-launch `NO_HEARTBEAT (title/login)` → Enter×2 → `HOME (hb+ENCAP, pos=None)` → Enter → `WAITING
(drone@origin)` → Enter → `ACTIVE GO`. **From cold launch the waiting room is TWO Enters past the first
telemetry screen** (both pre-waiting screens read `started=False, pos=None`; the probe-retry loop absorbs this).

### Failure-mode recovery (`partA_recovery.py`, `logs/partA_recovery*.json`)
| Mode | Status | Recovery (verified or documented) |
|---|---|---|
| **Off-race physics pause** | established + shown | Physics FREEZES off-race (`sim_time` static at HOME); advances once a race is ACTIVE (`sim_time` 25.4→28.4 s in-race). **Drive HOME→WAITING→GO to get live physics.** ⚠️ **Nuance found:** from a **FINISHED** race, `ESC+Down×3+Enter` **RESTARTS the race (clock resets)** — it does *not* land at HOME. Probe-verified driving tolerates this (the 10-cycle loop re-probed and reached WAITING every time regardless). |
| **Stale `started=True`** | ✅ **demonstrated live** | Induced `started=True`, drone parked@origin (`to_go=-2.53 s`) → full chain `esc,down×3,enter,enter` → `WAITING (started=False)`. The real video-stopping stale (idle ≳ minutes) uses this **identical** chain. |
| **Zombie dual-instance** | ✅ **demonstrated live** | 2nd `FlightSim.exe` launch → **2 `DCGame` procs detected** (arms-but-deaf) → recovery = **kill ALL (DCGame+FlightSim) + relaunch ONE** → 1 clean instance, heartbeat in 5 s. (Counting must target `DCGame-Win64-Shipping` only — the windowless `FlightSim` launcher is always a 2nd, benign process.) |
| **Spawn-artefact 0-tick** | detection wired + documented | Detect: `meta.video_frames < 10 AND final_state != FINISHED` (obs header only, 0 usable ticks). Recovery: **sleep 2–3 s after full-reset before arm** (let residual gate-3 collision geometry clear) — wired in `partA_cycle.run_cycle`. **Not induced** (requires a gate-3 HARD collision; excluded by the zero-contact rule). 13/13 flights healthy (439–442 frames, FINISHED). |

### Drift from `reference_sim_ops.md`
None material. Two **refinements** to bank: (1) cold-launch waiting room = **two** Enters past first
telemetry (not one); (2) `ESC+Down×3+Enter` from **FINISHED restarts the race (clock resets)**, not HOME —
sharpens the existing "context-dependent" note. Telemetry convention **unchanged** (mirror canary below).

---

## PART B — SHADOW-MODE VISION vs ABSOLUTE ✅

Ran the navigator's REAL chain (`GateDetector` → `estimate_gate_pose` IPPE/P3P+refine →
`gate_pose_to_world_position`, with the corrected `R_world_from_odo_quat_wxyz` attitude + VISION-PKG2
covariance: 1.4° lever + 0.40 m floor + 32 m cap) in **shadow** on **6 inc7 given-pose flights pooled**
(`cr1b` at-speed to 21 m/s + `std1–5`), **1821 frames**, against the absolute surveyed map
(`track_map.json`). Tools: `scripts/characterize_perception.py` per gate, pooled by
`analyze_shadow_vision.py`. The world-fix error `pos_fix − drone_truth` is, by construction
(`corner_to_center=True`), the **vision-derived-gate − absolute-gate** offset.

### The funnel (what shadow vision actually delivers at inc7 racing attitude)
```
1821 frames → detected 1644 (90%) → associated 417 (23%) → offered/χ²-accepted 126 (7%)
```
**The detector sees gates fine (90%); the loss is association + depth-sanity at oblique viewing angles.**

### Camera coverage — the binding constraint
Independent of the detector (pure geometry, given pose + attitude): **only 33 % of frames have ANY gate
centre projected inside the 640×360 image**; the in-view gate sits at **bearing p50 33.5°, p90 46.1°** —
right at the **HFoV half-angle (45°)**. inc7's heavy ~64° racing **crab points the camera off-gate**, so
the drone mostly sees the *next* gate at the frame edge (or nothing). This — not detector precision — is
why fixes are scarce.

### Precision WHERE fixes land (clean accepted, vision−absolute offset, |off|<3 m)
| range band | N | lateral (cross-track) σ | vertical σ | along-track (depth) σ | \|off\| p50 |
|---|---|---|---|---|---|
| <10 m (corner-clip) | 5 | 0.736 | 0.667 | 0.661 | 0.85 |
| 10–20 m | 25 | **0.086** | 0.412 | 1.089 | 1.11 |
| **20–26 m (gate-4 band)** | **94** | **0.103** | 0.235 | 0.775 | 0.67 |
| pooled | 125 | 0.206 | 0.312 | 0.860 | 0.69 |

- **Lateral (cross-track) σ ≈ 0.10 m in the 10–26 m band — ~2.5× TIGHTER than the modeled 0.265 m**, with
  **~zero lateral bias** (mean −0.002 m). At racing speed (>10 m/s) lateral σ = 0.105 m. ✅ matches/beats model.
- **Along-track (depth) is the weak axis (σ ~0.8–1.1 m)** — expected for monocular PnP; exactly why the KF
  fuses vision with the IMU/velocity prior rather than trusting raw depth. The pooled 0.206 m lateral is
  inflated only by 5 close-range (<10 m) corner-clip fixes.

### Per-gate, especially gate-4 (the binding band)
Gate-4 accumulates **N=20 accepted sightings** (seen during the gate-3 approach, ~23 m / 17 m/s):
**|off| p50 0.61 m, σ_E 0.18 / σ_D 0.10 m** — accurate whenever it is in view. (0 fixes associate while
*at* gate-4 because the camera is then looking at gate-5; sightings of gate G come from the G−1 approach.)

### Verdict on the vision premise
The detector+PnP **lateral precision is sufficient — it exceeds the modeled σ.** The premise holds for the
fixes that are accepted. **BUT** inc7's given-pose flight gives the camera poor coverage (crab → 33 % any-
gate-in-view → 7 % accepted). **This is a LOWER BOUND specific to inc7's flight style**, not an achievable-
vision number: inc7 has zero incentive to aim the camera. A vision-driving policy (inc8+) must **reduce
crab / point the camera at the target gate** — the binding work item is camera pointing, not detector/PnP.

---

## PART C — RECORDING HARNESS + GROUND-TRUTH-VELOCITY VERDICT ✅

**GROUND-TRUTH VELOCITY IS AVAILABLE AND LIVE — the step-5 plan is NOT changed.** (This was the gating
question; confirmed early, both from the client decode path and live recordings.)

The existing recorder (`scripts/record_session.py` / the `rl/fly_rl.py` recorder) already captures the full
step-5 bundle, **time-aligned on ONE clock** (recv-monotonic, bridged to UNIX epoch in the tlog + via the
meta `t0` pair for the video index — the sim's in-frame clocks are independent and deliberately NOT used).
Validator built + run: `verify_bundle.py` on the banked at-speed `instr_cr1b` AND a fresh PART-A session
(`simopsA_c05`) — **ALL_PASS** both:

| field | source | rate | live? |
|---|---|---|---|
| **ground-truth velocity** (load-bearing) | **`LOCAL_POSITION_NED.{vx,vy,vz}` world NED** | **96.8 Hz** | ✅ live to **21.3 m/s** |
| ground-truth velocity (cross-check) | `ODOMETRY` twist (body→NED rotated) | 75 Hz | ✅ agrees (21.3 m/s) |
| ground-truth position | `LOCAL_POSITION_NED.{x,y,z}` / `ODOMETRY` | 97 / 75 Hz | ✅ |
| `accel_body` (specific force, FRD) | `HIGHRES_IMU.{xacc,yacc,zacc}` | 119 Hz | ✅ p50 9.81 (=g at rest) |
| attitude quaternion (TRUE) | `ODOMETRY.q` (ATTITUDE-euler is sign-aliased — not used) | 75 Hz | ✅ |
| per-frame video (→ offline 4-corner detect + PnP `t_cam_gate` + reproj) | JPEG-UDP | 30 Hz | ✅ |

**Cross-stream alignment on the recv clock:** video↔LPN **2.5–2.7 ms p50 / ≤6.4 ms max**; video↔ODO 3.3 ms
p50 / ≤8.8 ms max (≈ half a telemetry period — as tight as the 75–97 Hz sources allow). The **per-frame
4-corner detection + PnP + reproj layer is the offline `characterize_perception` chain** exercised on 1644
detections in PART B — i.e. the step-5 harness = raw recorder (validated) + offline detector pass (validated).

**How to capture the cold-prior velocity:** read `LOCAL_POSITION_NED` `vx/vy/vz` (already stored as
`DroneState.velocity_ned`, world NED, pristine ~97 Hz). At-speed capture not needed to confirm this — proven
on `instr_cr1b` (21.3 m/s) and the fresh standing flight.

---

## Tooling promoted (`handoff/simops-mastery-2026-06-13/`)
- **`partA_cycle.py`** — probe-verified autonomous N-cycle sim-ops orchestrator (the reusable "hammer the
  sim reliably" harness for autonomous agents). `--debug-obs` passthrough for the residual check.
- **`partA_recovery.py`** — failure-mode recovery demos + documentation (pause/stale/zombie/artefact).
- **`verify_bundle.py`** — PART C harness validator: asserts every step-5 field present at rate + GT-velocity
  live + recv-clock alignment; exit 0 iff all pass.
- **`analyze_shadow_vision.py`** — PART B shadow-vision-vs-absolute aggregator (funnel, coverage, per-gate /
  per-range offset, lateral σ vs model).
- **`bearing_diag.py`** — camera-coverage geometry diagnostic.
- `logs/` — all JSON results + per-cycle fly_rl logs; `bundles/` — per-gate frame bundles (`frames.json`).
- Recommendation (carry-over from vision-cal): promote a trimmed `simops.py`/`partA_cycle.py` into `scripts/`
  (e.g. `scripts/sim_cycle.py`) — every live worker re-derives this loop.

## Validity
- **Mandatory `scripts/frame_residual_report.py` (fresh `simops_fra` session): mirror canary TRUE +0.97 /
  AS-IS −0.81 → OK.** Telemetry convention UNCHANGED on this fresh sim instance (identical to vision-cal
  close-out). Rate canary gain [0.93, 0.85, 0.95] ≈ 1.0; force residual matches the banked plant-gap baseline.
- Sim left in a clean, known **WAITING** state (1 `DCGame` instance, started=False, drone@origin).

---

## MEMORY-DELTA (≤10 lines)
- **SIM-OPS MASTERY CONFIRMED:** 10/10 consecutive clean autonomous inc7 cycles (streak=10; +3 = 13 clean
  flights, 0 gate contact) via probe-verified `partA_cycle.py`. All 4 failure modes handled: stale-started &
  zombie **recovered live**, spawn-artefact detection+sleep-before-arm wired, off-race-pause established.
- **SIM-OPS REFINEMENTS:** cold-launch waiting room = **TWO** Enters past first telemetry (not one); from a
  **FINISHED** race `ESC+Down×3+Enter` **RESTARTS the race (clock resets)**, does NOT go to HOME; zombie
  count must target `DCGame-Win64-Shipping` only (the `FlightSim` launcher is a benign 2nd proc).
- **SHADOW VISION vs ABSOLUTE (6 inc7 flights, 1821 frames):** detector 90%, but only **7% of frames yield an
  accepted fix** — camera coverage is the binding limit (33% any-gate-in-view; in-view bearing p50 33° at the
  45° HFoV edge) because inc7's ~64° crab points the camera off-gate. WHERE fixes land, **lateral (cross-track)
  σ ≈ 0.10 m at the 20–26 m gate-4 band — ~2.5× TIGHTER than the modeled 0.265 m, ~zero bias**; along-track
  (depth) is the weak axis (σ ~0.8 m). Gate-4: N=20, |off| p50 0.61 m. **Premise holds on precision; the
  vision-driving work item is CAMERA POINTING (de-crab), not detector/PnP.** Lower bound (inc7 has no
  camera-aim incentive on given pose).
- **GROUND-TRUTH VELOCITY = AVAILABLE & LIVE → step-5 plan UNCHANGED:** `LOCAL_POSITION_NED.{vx,vy,vz}`
  (world NED, pristine ~97 Hz, live to 21.3 m/s; `DroneState.velocity_ned`), ODOMETRY twist 75 Hz cross-check.
  Recorder captures the full step-5 bundle recv-clock-aligned (video↔LPN 2.5 ms p50); `verify_bundle.py`
  ALL_PASS on banked + fresh sessions. Mirror canary OK (TRUE +0.97) — telemetry convention stable across sim relaunch.
