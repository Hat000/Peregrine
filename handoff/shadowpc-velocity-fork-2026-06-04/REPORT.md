# REPORT — velocity-setpoint fork (ShadowPC, 2026-06-04/05)

**Binary question:** does the sim's velocity-setpoint controller hold altitude and track a commanded
NED velocity in **ANGLE** mode? **ANSWER: NO — World A.** At the reference command rate (250 Hz) the
velocity controller engages but its vertical/thrust channel is broken: a `vz=0` setpoint drives
collective to **max (~0.98)** and the drone **climbs away**, and a hard corrective `vz=+2.0` is
**ignored** (motors stay pinned at max). **⇒ the hand-built CTBR cascade is justified.**

This re-opened the memo's "easy-mode is CLOSED" claim under instruction, with validated instruments
and a recording. The conclusion lands in the same place as the memo — **but for a sharper, recorded
reason**, and the run surfaced a likely-important latent bug (see the ODOMETRY section).

> Pre-registration: predictions were locked in `UNDERSTANDING.md` (commit 4e09daa) **before** any
> data. World A was pre-registered as "HOVER climbs regardless AND DESCEND/CLIMB don't respond" — which
> is what the data show.

---

## CONDITIONS
- **Sim:** AI-GP Simulator **v1.0.3364**, `FlightSim.exe` on ShadowPC (co-located client). Race
  RUNNING (physics active, sim clock advancing) for every measured run.
- **Client tap:** `send_heartbeats=False`, `send_timesync=False` (clean tap). Did **not** arm (race
  auto-arms). Refused to start unless drone at origin (|pos|<1 m) + clock advancing.
- **Start state (per run):** pos ≈ `[0,0,0.02]` (origin), initial rpy ≈ `[0°, −17.8°, −179.9°]`
  (resting nose-down tilt, facing −X down-course). `reset_counter=0`.
- **Command:** `SET_POSITION_TARGET_LOCAL_NED`, velocity-only, **yaw-ignore**, type_mask **3527**
  (byte-identical to the reference client's `VELOCITY_POSITION_MASK` — verified on the wire each run).
- **GUI mode is NOT on the wire** (`HEARTBEAT.base_mode` constant) → the teammate's GUI is the sole
  mode ground-truth, reported per run below.

## RUN LEDGER
| # | run / file | rate | what it tested | GUI (teammate) | result |
|---|---|---|---|---|---|
| — | `trace_gate1_nullgate_ACRO.json` | — | null/mode gate | **ACRO** (stale) | instruments pass; mode ACRO ⇒ not usable |
| — | `trace_gate3_nullgate_ANGLE.json` | — | null/mode gate | **ANGLE**, held | instruments pass; **clean ANGLE start** |
| A | `trace_yaw_ignore_25hz_run1.json` | 25 Hz | HOVER/DESC/CLIMB/FWD | *not watched* | **UNCORROBORATED** — drone pinned; reframed as pre-control pin |
| 1a | `trace_yaw_ignore_250hz_first.json` | 250 Hz | HOVER | ANGLE, climbed | climb onset; aborted at 82 ms on a v_fd artifact (instrument fixed) |
| 1 | `trace_hover_250hz_long.json` | 250 Hz | HOVER (long) | ANGLE, climbed | **RUNAWAY** to motors-max; aborted on frame artifact (instrument fixed) |
| 2 | `trace_descend_arrest_250hz.json` | 250 Hz | DESCEND vz=+2 vs climb | ANGLE, kept climbing | **World A** — vz ignored, motors pinned |

---

# OBSERVATIONS (raw)

## O1 — Phase-1 instrument validation (the hard gate, run before interpreting anything)
- **Null test** (stationary drone, sending nothing): all three velocity derivations read **exactly
  0.0** (mean & max), n≈70–890 samples, across every run (`gate1`, `gate3`, and the start of `hover_250_long`,
  `descend_arrest`). Unbiased pipeline. ✔
- **Telemetry liveness** with heartbeats **and** timesync **both OFF**: LOCAL_POSITION_NED ≈ **95–97 Hz**,
  ODOMETRY ≈ **74–75 Hz**, ACTUATOR_OUTPUT_STATUS ≈ **95–97 Hz** → **LIVE**. (Resolves the
  `mavlink_client.py:148-152` hedge: no keepalive is needed; `angle_vel_test.py` was right.) ✔
- **Command echo** (captured from the actual wire send): mask **3527**, vel only, pos/accel/yaw
  ignored — **byte-identical to the reference client**, every run. ✔
- **Actuator witness** (parser-independent): rest = **[0.05, 0.05, 0.05, 0.05]** at a clean race
  start; modulates under control (see traces). ✔
- **Triangulation:** three independent velocity derivations — `v_lpn`=LOCAL_POSITION_NED (world NED),
  `v_odo`=ODOMETRY (**body FRD**, see bug section), `v_fd`=Δpos/Δt (world). After frame reconciliation
  they agree (O4). The raw 3-channel tap was necessary because `mavlink_client` collapses
  LOCAL_POSITION_NED + ODOMETRY into a single `velocity_ned` field (interleaved 95/75 Hz).
- **GUI corroboration:** mode is finicky — `gate1` (Home→Race) came up **ACRO**, `gate3` (Home→Race)
  came up **ANGLE**. So ANGLE was re-confirmed by the teammate on **every** interpreted run.

## O2 — Run #1: HOVER @ 250 Hz, v=(0,0,0)  (`trace_hover_250hz_long.json`)
Downsampled; vz from `v_lpn` (world), motors = mean of 4:

| t (ms) | z rel (m, −=up) | vz (m/s) | motors | pitch |
|---|---|---|---|---|
| 0 | 0.000 | 0.0 | **0.05** (pin) | −18° |
| 69 | −0.034 | −0.93 | **0.475** (wakes) | −6° |
| 146 | −0.160 | −2.46 | 0.67 | −2° |
| 194 | −0.332 | −4.46 | 0.89 | −1° |
| 264 | **−0.828** | **−8.18** | **0.98** | 0° |

vz monotonic & **accelerating**; z climbs accelerating; **motors ramp to 0.98 (near full throttle)** —
the *runaway* signature, opposite of *settle* (vz→0, motors→hover). Motor-wake onset (~69 ms) matches
Run #1a → **crux reproduced ×2 independent races** (and a 3rd time in Run #2 Phase A). GUI: ANGLE,
"climbed upward, then held upward velocity" (the "held" = post-disarm ballistic coast).

## O3 — Run #2: DESCEND-ARREST @ 250 Hz  (`trace_descend_arrest_250hz.json`) — the discriminator
Phase A (v=0): climbed to 1.52 m, vz→**−11.35 m/s**, motors→**0.99** (reproduction #3). Then snap to
**vz=+2.0 (hard DOWN), held**:

| phase | z rel (m) | vz (m/s) | motors | cmd vz |
|---|---|---|---|---|
| A→B **switch** | −1.52 | **−11.35** | 0.991 | → +2.0 |
| B +0.10 s | −2.6 | −15.3 | 0.991 | +2.0 |
| B +0.25 s | −5.9 | −20.6 | 0.987 | +2.0 |
| 8 m abort | −8.30 | **−23.1** | 0.982 | +2.0 |

- **vz kept growing more negative** (−11 → **−23 m/s**) — climbed *faster*, never arrested.
- **motors stayed pinned at max: min 0.982, max 0.992** across all of Phase B (a 1 % drift, not a
  control response). `reversed=False, motors_dropped=False`.
- **Net vertical acceleration** (per-tick): first/mid/last third = **−39.8 / −33.5 / −24.0 m/s²** —
  always strongly upward. The taper (−40→−24) is **aerodynamic drag** (∝v² at −23 m/s), **not** the
  command: thrust never dropped. The teammate's "it stopped accelerating" = this drag taper + the
  post-disarm coast (records end at the 8 m abort; everything after is unrecorded ballistic).
- Floor-safety never triggered (it climbed, never descended). Aborted on the 8 m altitude limit.

## O4 — Triangulation numbers (the frame catch)
`max |v_lpn − v_odo_world|` per axis, **after** rotating ODOMETRY twist body→world:
- Run #2 Phase A: vx **0.04**, vy 0.01, vz ≤0.68 · Phase B: vx **0.03**, vy 0.00, vz ≤0.52.
- The residual vz gap (~0.5–0.7) is 97 Hz-vs-75 Hz **rate-stagger during 30+ m/s² acceleration**, not
  a fault. **Before** the frame rotation, the raw `|v_lpn − v_odo|` was **vx = 0.81** (equal-and-
  opposite) — that was the clue (see bug section). Conclusion: **the two independent sim sources agree
  once in a common frame ⇒ telemetry is trustworthy.**

---

# 🚩 THE ODOMETRY BODY-FRAME VELOCITY BUG (likely the most valuable finding here)

**Claim:** `ODOMETRY.vx/vy/vz` are reported in the **BODY (FRD) frame**, but
`mavlink_client._handle` (`src/racer/mavlink_client.py:255-268`) stores them straight into
`DroneState.velocity_ned` as if they were **world NED** — and that field is **interleaved** with
LOCAL_POSITION_NED's genuinely-world velocity (the two messages overwrite the same field at 75/95 Hz).
So `velocity_ned` has been a mix of world (from LOCAL_POSITION_NED) and body-frame (from ODOMETRY)
samples whenever the drone is moving while yawed/tilted.

**How it was found (triangulation, not luck):** during the runaway the two sim velocity sources
agreed on **vz** but were **equal-and-opposite on vx** (`v_lpn.vx=−0.41`, `v_odo.vx=+0.40`) while the
drone sat at **yaw=−180°, level**. A 180° yaw maps body→world as `(bx,by,bz)→(−bx,−by,bz)`, which
predicts exactly `v_lpn.vx = −v_odo.vx`, `vy` flipped, `vz` unchanged — all three matched. Rotating
`v_odo` by the ODOMETRY quaternion collapsed the vx disagreement from **0.81 → 0.03 m/s**. (Sanity-
checked offline: `Rotation.from_quat` at yaw −180° sends body `(0.4,0,−8)` → world `(−0.4,0,−8)`.)

**Why this may matter a lot (hypothesis — to investigate, not yet proven):** the navigator/CTBR stack
consumes `DroneState.velocity_ned` for the KF, the controller's velocity damping, and the planner.
If that field has been silently mixing a **body-frame** velocity in (wrong sign on x/y at the course's
−X/−180° heading), it is a strong **candidate root cause** for two prior unexplained gremlins recorded
in `project_ctbr_control_sysid.md`:
- **"KF velocity LAGS the truth ~4×"** (saga §5) — a frame-mismatched velocity fed as a tight KF
  measurement would corrupt the velocity estimate, not just lag it.
- **the lateral oscillation / positive-feedback feel** — a sign-flipped horizontal velocity in the
  damping term is exactly the kind of thing that turns damping into anti-damping on one axis.

This was **not chased further this run** (out of scope; logged per the teammate). **Follow-up:** decode
a recorded `ODOMETRY` vs `LOCAL_POSITION_NED` velocity offline across a yawed move to confirm the
frame, then fix `mavlink_client` to either rotate ODOMETRY twist body→world or take velocity solely
from LOCAL_POSITION_NED — and re-test the KF-lag and lateral findings against that fix.

---

# INTERPRETATION

## Verdict: **World A** (per the decision matrix)
> *World A — HOVER climbs regardless of vz sign AND DESCEND/CLIMB don't respond.* ✔ observed:
> HOVER (v=0) climbs to motors-max; DESCEND (vz=+2 corrective) is ignored, motors stay pinned.

The sim's ANGLE-mode velocity controller does **engage** at the reference 250 Hz rate (motors leave
the pre-control pin), but its **vertical/thrust law is broken**: instead of regulating to the
commanded vertical velocity, collective saturates upward and the drone runs away, **indifferent to the
commanded `vz`**. Plain velocity setpoints therefore **cannot** fly the course (the drone leaves the
corridor vertically regardless of command) ⇒ **we must own thrust ⇒ CTBR.** This agrees with the
memo's destination, but the memo's *route* (a GUI-glance generalization of an ACRO result to "every
mode") is now replaced by a recorded, frame-validated ANGLE-mode measurement.

## Evidence chain, each as (instrument) + (independent corroboration) + (what would falsify it)
- **C1 — World B is false (HOVER does not hold).** *Instrument:* `v_lpn` world vz 0→−8.2 accelerating,
  motors 0.05→0.98, z climbing (Run #1). *Corroboration:* teammate GUI "climbed", ANGLE; **reproduced
  ×3** independent races. *Falsifier:* vz→0 with motors settling to hover (~0.26) and z stable — did
  not occur.
- **C2 — World A, not Partial (vz has no authority).** *Instrument:* under `vz=+2.0` DOWN, motors
  pinned 0.982–0.992 (never dropped), vz −11→−23 (never arrested), climb to ceiling (Run #2).
  *Corroboration:* teammate GUI "kept climbing", ANGLE. *Falsifier:* motors dropping / vz arresting or
  reversing on the down command ⇒ Partial — did not occur.
- **C3 — telemetry is trustworthy.** *Instrument:* null exactly 0.0 on all three channels; the two sim
  sources agree to ~0.03 m/s after frame reconciliation. *Corroboration:* the body-frame hypothesis
  *predicted* the vx equal-and-opposite signature, then the rotation removed it. *Falsifier:* gross
  `v_lpn` vs `v_odo_world` disagreement after frame correction — did not occur.

## Reproduction
HOVER climb-away reproduced across **3 independent races** (Run #1a 250 Hz, Run #1 long, Run #2 Phase
A), with matching onset (motors leave 0.05 and ramp to ~0.5 within ~60–70 ms, then to max). The verdict
is **not** from a single run.

## Scope caveats (honest limits)
- **Horizontal velocity authority was not isolated** at 250 Hz (no clean FORWARD-only-with-held-z run —
  impossible, since z can't be held). It is **moot for the verdict**: vertical is uncontrollable, so
  the course cannot be flown on velocity setpoints regardless of horizontal behavior. (Weak signal:
  horizontal also drifted off `vx=0` during the climbs, but that's confounded with the vertical
  runaway and the ODOMETRY frame bug, so it is **not** claimed.)
- **Yaw A/B not run.** Only yaw-ignore (mask 3527, reference-exact) was used. The yaw-hold pass was
  unnecessary: the failure is in the vertical/thrust channel, where a yaw setpoint is irrelevant; and
  World A was settled before a yaw comparison was needed.
- **The 25 Hz "HOVER held" observation is UNCORROBORATED** (teammate did not watch
  `trace_yaw_ignore_25hz_run1.json`) and is **disregarded** for the verdict. Reframed (per teammate)
  as the **pre-control pin**: at a sparse 25 Hz the velocity controller does not engage and the sim
  holds the drone; at 250 Hz it engages and the runaway appears. Behavior is **rate-dependent**, and
  only the engaged (≥~reference) rate is representative.
- **Mode is finicky and off-wire:** a Home→Race reset landed ACRO once and ANGLE once. Any future
  velocity work must re-confirm ANGLE on the GUI per run; it cannot be read from telemetry.

## Implication
Proceed on **CTBR** (the existing `project_ctbr_control_sysid.md` stack). The open gate-0 "altitude
balloon" blocker is consistent with this finding — the same broken vertical auto-thrust that climbs
away on a velocity setpoint is what "takes over and balloons" when the CTBR collective dips. The
ODOMETRY frame bug (above) is the highest-value follow-up: it may explain prior KF/lateral gremlins.

---

## ARTIFACTS (all committed to `red-team-tier-a`, this handoff dir)
- `UNDERSTANDING.md` — pre-registered model + predictions (committed **before** data, 4e09daa).
- Per-tick JSON traces (downsampled per tick; meta has null/echo/actuator/verdict; records have
  `cmd_v, v_lpn, v_odo, v_odo_world, v_fd, pos_lpn, pos_odo, rel_pos, rpy_deg, actuators,
  tri_disagree, lpn_odo_disagree` — enough to re-derive the verdict):
  `trace_gate1_nullgate_ACRO.json`, `trace_gate3_nullgate_ANGLE.json`,
  `trace_yaw_ignore_25hz_run1.json`, `trace_yaw_ignore_250hz_first.json`,
  `trace_hover_250hz_long.json`, `trace_descend_arrest_250hz.json`.
- Scripts (in `scripts/`): `velocity_ladder.py` (ladder + Phase-1 gate + frame-aware triangulation),
  `descend_arrest.py` (the World-A discriminator), `analyze_ladder.py` (trace/agreement analyzer).
- Raw `mavlink.tlog` per run + the rest of `data/runs/*` are **gitignored on ShadowPC** (not pushed).

## FOLLOW-UPS (for the laptop / memory consolidation — NOT edited here)
1. **ODOMETRY body-frame velocity bug** — confirm offline + fix `mavlink_client`, then re-test the
   KF-lag-4× and lateral-oscillation findings against the fix. (Highest value.)
2. Easy-mode is now **closed with a recording** (not a glance) — the memo can cite this report.
3. Velocity-controller behavior is **rate-dependent** (sparse → pin, dense → engaged runaway) — note
   for anyone tempted to re-probe at low rate.
