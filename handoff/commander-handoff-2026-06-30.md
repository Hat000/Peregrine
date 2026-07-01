# Peregrine — COMMANDER HANDOFF (Gen 6 → Gen 7), 2026-06-30

*You are booting as the OVERALL COMMANDER of Peregrine, my successor. This is your complete
situation briefing. Read in this order: (1) `COMMANDER.md` (repo root) — the durable commanding
craft; (2) `memory/MEMORY.md` — the SSOT index; (3) THIS FILE — the live state, the resource map,
and your immediate next action. Then follow the `[[pointers]]` into `memory/index_vision_estimator.md`
(the VQ2 slow-lap saga lives there in full detail). Address Fengyou by name in every message.*

---

## 0 · THE ONE-PARAGRAPH SITUATION
We are bringing up the **first working VQ2 slow lap** — a self-localizing autonomous drone flying
slowly and cleanly through gates in the Anduril AI Grand Prix qualifier sim ("Now You See Me, Now
You Don't", a dark warehouse, gates glow red). We work as a **tight footage-verification loop**:
I (commander) predict behavior from telemetry + code, Fengyou flies it on ShadowPC and reports what
the onboard footage shows, I root-cause and dispatch the next fix. Over this session that loop
walked the drone from "tumbles on tick 1" to "passes the first gate" (milestone **A5**) to "the
estimator is trustworthy" (milestone **A10**) and just now cleared **two independent blockers at
once**: the case-C yaw steering frame was mirrored (fixed, `337c554`), and the trained gate detector
was never actually wired into the flight loop (fixed, `2ae5a71`). **The next fly (A15) is the first
attempt where the seeker has both enough gate-poses AND correct steering** — the milestone to watch
is *sustained pursuit through a gate*. Nothing is blocked on you except handing that fly to ShadowPC
and continuing the loop.

## 1 · THE MISSION (what we're doing and why)
- **VQ2 qualifier**: fly the official ShadowPC sim (build 1.0.3379) through the gate course, zero
  gate contact (contact = invalid run). The eval sim IS our development sim — no sim-to-real gap.
- **Doctrine: SLOW IS SMOOTH, SMOOTH IS FAST.** Close the loop SLOW first (more frames/metre, no
  motion blur, vision self-localization works), THEN ramp speed. `[[feedback_slow_is_smooth]]`.
- **Moonshot** (`[[project_moonshot_speed]]`): target is *championship-fast*, not just finishing.
  North star = Swift (Nature 2023). Arc = slow bootstrap → perception hardening → RL speed ramp.
- **The binding constraint that shapes everything:** the VQ2 wire is STATE-DENIED. No position, no
  attitude, no ODOMETRY, no gate map. `HIGHRES_IMU` = accel + gyro ONLY (🚩 NO MAG, NO BARO). So
  **yaw + altitude must come from VISION**; attitude roll/pitch from an IMU AHRS. The drone
  self-localizes or it doesn't fly. This is "case-C". Full wire recon → `[[index-vision-estimator]]`
  §VQ2-WIRE-RECON-2026-06-29.

## 2 · HOW FAR WE'VE COME — the A5→A15 fix chain (each its own diagnose→implement→fly cycle)
The slow-lap bring-up ("A#" = attempt/fly number). Each was root-caused from telem+footage, fixed
behind a default-off / vq2_case_c-only flag (VQ1 byte-identical), TDD/green_gate-verified, flown.

- **The 2 Hz vision choke** → vectorized the vanishing-point RANSAC + dropped iters 2000→256. Loop
  ~2.3 Hz → ~16 Hz, video drops 55%→0. (`de9d226`, `f1587b3`, `6733f50`.) *Lesson banked: fixing the
  framerate EXPOSED latent bugs the slow loop had masked — speed is a bug detector.*
- **A7 spawn-inside-gate collapse** → VQ2 spawns the drone INSIDE gate 0 → point-blank → thrust
  floors → free-fall. Fixed: egress thrust floor + point-blank elevation guard. (`dcc1d7f`.)
- **A8/A9 globally-inverted live gyro** → the live VQ2 `HIGHRES_IMU` gyro is SIGN-NEGATED on all 3
  axes (`gyro_reported ≈ -2.1·ω`; the 2.1 is the cmd→realized gain, only the SIGN is the bug). Fixed
  via `MavlinkClient.gyro_sign = VQ2_GYRO_SIGN = (-1,-1,-1)`. (`5d7215e`, `a5c5e76`.) **A10 = the
  breakthrough: 28 s stable flight, self-localization TRUSTWORTHY for the first time.**
- **A10 egress acquisition trap** → egress ramps forward-demand from ~0 while the spawn tilt is
  frozen → the controller commands a saturated nose-up re-level → camera swings off gate → pursuit
  never acquires. Fixed: `egress_freeze_attitude` (hold the spawn attitude through egress). (`31aca88`.)
- **A11 stiff-control handoff overshoot + bang-bang** → `kp_att=10` vs the ±1.5 rad/s pitch cap
  saturates on any >8.6° error → the egress→pursuit handoff whips nose-up + 15/18 command bursts slam
  the clip. Fixed: soften the attitude loop — `kp_att 10→4` + a body-rate slew limiter — via a new
  `DeployProfile.controller_overrides` seam. (`8b702e7`.)
- **A12/A13 pose-gap zero-coast (polygonal motion)** → on a tick with no usable pose the seeker
  emitted an all-axes-zero hold → "line segments, not a curve." Fixed: `hold_last_demand_s=0.6`
  bridge (re-issue the last pursuit demand through gaps → continuous per-tick command) + `[seeker-diag]`
  instrumentation. (`ec35a2b`.)
- **A13/A14 root data**: the `[seeker-diag]` counter revealed the REAL sparsity root is
  `valid_empty` (the perception pipeline returns NO usable pose on ~74% of in-view ticks), NOT the
  continuity gate we'd guessed. Two independent blockers isolated from here:
  - **Lead 3 — case-C yaw steering was MIRRORED.** Under `use_ahrs` the navigator re-encodes the
    TRUE AHRS attitude into the legacy ODOMETRY conjugation (`q_true*[1,-1,1,-1]`, roll+yaw negated),
    and the case-C gate-seeker — a NEW consumer — read that conjugated `NavState` to build its
    world-frame gate direction → for a gate on the RIGHT it steered the nose LEFT. **NOT a sign knob
    (gyro/body_rate all confirmed correct)** — a frame-convention mismatch. Fixed: `true_attitude_from_ahrs`
    makes the seeker recover the TRUE euler `(-nav.roll, nav.pitch, -nav.yaw)` for all geometry +
    pass the controller a yaw-negated nav so `R_cur=R_true`. (`337c554`.) *This one took FOUR agent
    deaths to land — see §6.*
  - **Vision — the trained detector was never deployed.** The live gate-seeker defaulted to the
    CLASSICAL `red_glow` OpenCV detector; the champion-beating trained YOLO ensemble (`282abb9`) was
    never wired in because `--checkpoint` was overloaded (loaded as the RL actor before the detector
    could use it). Offline on the real A13 frames the YOLO STRICTLY DOMINATES red_glow (`valid_empty`
    74% → 35% single / 27% ensemble). Fixed: `--seeker-weights` splits the detector weights from the
    RL actor + skip the actor load under `--gate-seeker`. (`2ae5a71`.)

**Milestones:** A5 = FIRST GATE PASSED (footage-confirmed, map-free, no mag/baro). A10 = FIRST
trustworthy self-localization (28 s). A15 (next) = first fly with poses + correct steering together.

## 3 · THE TWO ACTIVE THREADS
1. **Flight-controller bring-up** (the A# loop). Owner: this thread. Next = the A15 combined fly,
   then Lead 1 (see §4).
2. **Vision-training** (Fengyou opened it explicitly). The gate-perception system. Step 1 was the
   near-free deployment-gap win (deploy the trained YOLO — DONE, arg-split shipped). The REAL vision
   work is the residual: the trained YOLO still leaves ~27% `valid_empty` (out-of-frame / far /
   oblique / motion-blur gates). That's the perception-hardening effort — see §4.

## 4 · WHAT'S PENDING / THE PRIORITY QUEUE
1. **A15 combined fly** (immediate — box in §7). YOLO detector + yaw-frame fix, single-model YOLO
   first (21 ms, kinder to the ~12 Hz choke). Read the two independent signals: `[seeker-diag]`
   `valid_empty` (detector) and yaw-steer direction (frame fix). Watch for SUSTAINED pursuit.
2. **Re-measure, then decide vision scope.** The A15 `valid_empty` (with the yaw fix keeping gates
   in frame) tells you how much of the 74% was self-inflicted by the old steer-away vs a genuine
   detector recall hole. If a real hole remains → the perception/data-engine effort:
   - Blender data pipeline is SCAFFOLD-ONLY: `tools/blender_pipeline/` has `APPEARANCE_SPEC.md`
     (the dark-warehouse/red-glow look quantified) + a `MockRenderer` only — the photoreal backend
     is UNBUILT. `[[project_blender_vq2_data_pipeline]]`.
   - Sim-direct auto-labeling is BLOCKED by the state-denied wire (no pose → no label); the
     practical path is semi-auto labeling of real sim frames. Training-loss lever =
     `cluster/vq2_precision_loss.py`. Early deliverable = a flight-REPRESENTATIVE eval set (the
     existing detector benchmark doesn't capture the flight failure regime).
3. **Lead 1 — the egress→pursuit handoff +28° nose-up.** Still untouched (regime-3, unrelated to
   the yaw/detector fixes). Hypothesis: the egress-freeze holds the ~-18° dive, SINKING the drone so
   the gate ends up ABOVE it → pursuit computes a steep climb → nose-up. Needs its own diagnostic;
   likely fix = bound egress descent or clamp the transition target. Do it AFTER A15 (it may present
   differently once poses + steering work).
4. **The ~12 Hz loop choke** (target 30). The YOLO adds ~21 ms; watch the loop self-report. Latent
   lever if pursuit is throughput-limited.

## 5 · RESOURCE MAP (everything you need to operate)
**Repo / git.** `github.com/Hat000/Peregrine`, local `C:\Users\Fengy\Downloads\Projects\Anduril`.
You are in a WORKTREE: `.claude\worktrees\loving-galileo-92f020`, branch
`claude/loving-galileo-92f020`, **HEAD = `2ae5a71`** (== origin; pushed). Untracked
`handoff/vq2-recon-2026-06-29/` is fine — leave it. `main` is behind this branch; the VQ2 slow-lap
work lives on this feature branch (not merged to main yet — Fengyou's call when/if).

**Python / tests.** 🚩 **The `.venv` is in the MAIN checkout, NOT the worktree:**
`C:/Users/Fengy/Downloads/Projects/Anduril/.venv/Scripts/python.exe` (the worktree's `.venv` does
not exist — a footgun that cost me a confused minute). Pre-merge gate = `scripts/green_gate.py`
(load-bearing invariants + test-count sentinel ~1581). 🚩 **green_gate carries ONE KNOWN PRE-EXISTING
RED: `tests/test_diagnose_session.py::test_real_bundles_diagnose_end_to_end`** — the handoff-slimming
purge removed its data fixtures but left the dir, so its skipif no longer fires. It is UNRELATED to
any current work — do NOT chase it (a `spawn_task` chip is filed to fix the guard). "GREEN-except-that"
is the pass condition.

**ShadowPC (the flight relay).** ShadowPC runs the OFFICIAL VQ2 eval sim. It is a **copy-paste
relay** (no connector yet — building one is the standing task to retire the last paste,
`[[index-strategy-meta]]` §SHADOWPC-CONNECTOR). You emit a paste-ready fly box; Fengyou runs it;
Fengyou (or the ShadowPC worker) reports back + pushes a handoff branch. ShadowPC repo =
`C:\Users\Shadow\Peregrine`; run dirs `C:\Users\Shadow\Peregrine\data\runs\<timestamp>_...\`
(onboard.mp4 + nav_estimate.jsonl + seeker_diag.txt). **Flight ritual:** start the sim; ARM; the
race stays `started=False` until Fengyou presses GO/Enter; `cmd_rate_scale` "Down ×2"; after any
crash the race drops to `started=False` and needs Fengyou's FRESH RELOAD (a single GO-Enter won't
restart) — lean on him for reloads between attempts. Sim flights = NO physical risk → authorize
directly. Prior flight handoffs are pushed branches: `vq2-slowlap-a13-2026-06-30`,
`vq2-slowlap-a14-2026-06-30` (fetch + read their FINDINGS.md for the exact base fly command + numbers).

**VQ2 wire facts (live-confirmed, build 1.0.3379).** Blocked: LOCAL_POSITION_NED, ODOMETRY,
ATTITUDE, GATE_INFO. On the wire: HEARTBEAT(10Hz); `HIGHRES_IMU`(117Hz, accel+gyro only — NO
mag/baro); 30 Hz JPEG cam (udpin:14550 / video:5600); `RACE_STATUS`(4Hz, `active_gate_index` present
→ gate ordering solved); `ACTUATOR_OUTPUT_STATUS`(95Hz, 4 motors); TIMESYNC response-only. **Control
handshake:** ARM (MAV_CMD 400 p1=1) + stream `SET_ATTITUDE_TARGET` BODY-RATE (type_mask=0b10000000,
FRD rates + collective thrust [0,1]) ≥30 Hz; sim default = ACRO = CTBR; `fly_rl.py` already drives
this. Command→realized body-rate gain ≈2.5× (compensated by `cmd_rate_scale=0.4`). Gyro globally
sign-negated (compensated by `gyro_sign=(-1,-1,-1)`).

**The stack + the SEAMS (how a fix reaches the flight).** The case-C flight is one command:
`rl/fly_rl.py --gate-seeker --deploy-profile vq2_case_c`.
- `src/racer/deploy_profile.py` — the ONE curated bundle. `vq2_case_c()` sets `nav_config` (case-C
  estimator flags), `cmd_rate_scale=0.4`, `gyro_sign=(-1,-1,-1)`, and TWO override dicts that are
  your primary tuning seams (both default-off/None → VQ1 byte-identical):
    - `seeker_overrides` → `GateSeekerConfig(**...)` — currently `{egress_freeze_attitude:True,
      hold_last_demand_s:0.6, true_attitude_from_ahrs:True}`.
    - `controller_overrides` → `make_seeker_controller(**...)` — currently `{kp_att:4.0,
      body_rate_slew_max_rps2:8.0}`.
  Threaded in `fly_rl.py` `_build_casec_seeker` (~L986). **To add a vq2-only knob: add the field to
  GateSeekerConfig/Controller default-off, then set it in the profile dict. VQ1 stays byte-identical.**
- `src/racer/gate_seeker.py` — `GateSeeker.command_visual` (the map-free visual-servo pursuit law;
  fly_rl calls it at ~L1248). Regimes: settle → anchor → egress → pass-coast → no-pose-hold →
  pursuit. Helpers `_att_rpy`/`_att_yaw`/`_controller_nav` (the A14 true-attitude recovery).
- `src/racer/navigator.py` — the case-C estimator: AHRS (`use_ahrs`), `use_vp_yaw` +
  `use_gate_bearing_yaw` (vision yaw, no mag), `use_floor_height` (z), the gate-relative +L chain.
  🚩 It re-encodes the AHRS true attitude into the ODO conjugation before writing `NavState` (the
  root of the A14 bug — the seeker now un-conjugates).
- `src/racer/controller.py` — decoupled CTBR (`_decoupled_body_rate`). Sign config
  `odo_att_sign=[-1,1,1]` / `odo_rate_sign=[-1,-1,1]` / `body_rate_sign=[1,1,-1]` is a self-consistent
  VQ1-tuned alias — **DO NOT "fix" it.**
- `src/racer/mavlink_client.py` — the ONLY wire origin of `gyro_body`; applies `gyro_sign` before the
  AHRS. Also stashes `gyro_body_raw` (pre-sign) for the yaw probe.
- **Detector**: `--seeker-detector {red_glow|yolo|none}` (default red_glow). YOLO weights via
  `--seeker-weights "<a.pt>"` or `"<a.pt>++<b.pt>"` (ensemble). Weights are gitignored release assets
  (`burn-artifacts-*` tags, `gh release download --repo Hat000/Peregrine`) — ShadowPC has them.

**Instrumentation (read these, don't eyeball).** `nav_estimate.jsonl` (per tick: ahrs_quat,
roll/pitch/yaw, position_ned, body_rate, thrust, time_since_vision_s, `yaw_des_rad`, `raw_gyro_yaw`).
`[seeker-diag]` one-line summary at loop exit: `pursuit / valid_empty (valid_poses_empty /
continuity_reject / first_acq) / bridged / held_legacy`.

**Memory.** `memory/MEMORY.md` (SSOT index — over its size cap; keep it lean) → the domain
sub-indices. The FULL A5→A15 saga + every root cause + the resolved footguns are in
`memory/index_vision_estimator.md` (§VQ2 SLOW-LAP BRING-UP SAGA / the A#-lettered bullets). Feedback
files: `[[feedback_slow_is_smooth]]`, `[[feedback_telem_ambiguous_without_footage]]` (the
measurement-vs-belief trust hierarchy — read this, it encodes my biggest mistake). Mirror
`memory/` ↔ `~/.claude` after banking.

## 6 · HAZARDS SPECIFIC TO RIGHT NOW
- 🚩 **Background agents keep dying** — the Claude-Code process exited repeatedly this session,
  killing every in-flight agent (4 deaths: 3 process-exits + 1 API-stall). When one dies: `git
  status`/`git diff` FIRST — twice a dead agent left a COMPLETE, correct, uncommitted implementation
  I recovered + verified + landed inline in minutes. For a load-bearing task that's already failed
  once to infra, implement it INLINE (can't be killed by a process exit). See COMMANDER.md §2.
- 🚩 **Don't trust estimator-derived logs on this denied wire.** `position_ned` in the log is
  dead-reckoned FICTION; reconstructed attitude can wrap. Anchor to raw IMU / actuator / camera +
  footage. (This burned me — the A8 "nose-down dive" was actually nose-up-and-backward.)
- 🚩 **Never double-fix a sign** — localize offline where provable, else probe with an
  estimator-independent cross-check. The A14 yaw looked like a `body_rate_sign` flip; it was an
  estimator-frame mirror. See COMMANDER.md §2.
- 🚩 **NO sign knob is broken:** `gyro_sign=(-1,-1,-1)`, `body_rate_sign`, `vp_yaw`,
  `gate_bearing_yaw`, `odo_att_sign` are ALL confirmed correct. Future yaw/frame issues are consumer
  bugs, not sign bugs.

## 7 · YOUR IMMEDIATE NEXT ACTION — hand this A15 fly box to Fengyou
```
A15 combined fly — YOLO detector (deployment-gap fix) + case-C yaw-frame fix. HEAD = 2ae5a71.

PULL: git fetch && git checkout claude/loving-galileo-92f020 && git pull   (HEAD should be 2ae5a71)
Confirm: git log --oneline -2
   -> 2ae5a71 feat(vq2): split detector weights ... --seeker-weights
   -> 337c554 fix(vq2): case-C seeker reads TRUE AHRS attitude ...

WEIGHTS: locate the trained YOLO. Start with a SINGLE model (21ms, kinder to the ~12Hz choke) —
one .pt from the 282abb9 clean ensemble (models/ or `gh release download --repo Hat000/Peregrine`
from the burn-artifacts-* tag). Keep the 2nd model for an ensemble follow-up if recall needs it.

FLY: same vq2_case_c slow gate-seeker run as A13, detector now on --seeker-weights:
   ... --gate-seeker --deploy-profile vq2_case_c --seeker-detector yolo --seeker-weights "<single_model.pt>"
   (--checkpoint is NO LONGER the detector — leave it at default; the actor load is skipped under
    --gate-seeker now. Ensemble later: --seeker-weights "a.pt++b.pt".)
Same launch ritual (ARM; race started=False until GO/Enter; cmd_rate_scale Down x2; fresh reload
between crashes). Capture onboard.mp4 + nav_estimate.jsonl + the [seeker-diag] line.

READ — TWO independent things, both should improve (separable in the logs):
  1. DETECTOR ([seeker-diag] valid_empty %): A13 red_glow ~74%. Expect a big drop (offline ~35% single).
  2. YAW STEERING: does it now yaw TOWARD gates? gate on the RIGHT should pull the nose RIGHT
     (A13/A14 it went LEFT). Footage + yaw_des_rad are the tell.
  3. LOOP RATE self-report with the 21ms YOLO.

MILESTONE TO WATCH: with poses flowing AND steering un-mirrored, the seeker should finally SUSTAIN
pursuit — track a gate, commit nose-down forward, approach/thread it. Even partial (closes + lines
up) is the win; note where it breaks next.

Push (onboard.mp4 + nav_estimate.jsonl + [seeker-diag] + valid_empty-vs-A13 + which-way-it-yaws +
loop rate + 1-line outcome) to a handoff dir; main untouched.
```

When the A15 result returns: attribute each signal (detector vs steering), bank it to
`index_vision_estimator.md`, and pick the next lever from §4. Welcome to the chair. — Gen 6
