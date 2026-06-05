# Followups 2026-06-05 — UNDERSTANDING (pre-registration)

**Session:** ShadowPC, fresh, co-located with `FlightSim.exe`. Three follow-ups inherited after the
ODOMETRY velocity-frame bug was found+fixed (`c3b5a8e`) and re-validated (CONFIRMED,
`handoff/shadowpc-framefix-revalidation-2026-06-05/`). Discipline: validate instruments before
trusting them, separate OBSERVATION from INTERPRETATION, one variable at a time, GUI is the only
mode ground-truth, "blocked/inconclusive honestly" is a success.

---

## Orientation note — the worktree was on the WRONG base (resolved, non-destructive)

`git pull` said "up to date" but I was missing everything the brief referenced. Cause: this
worktree branch (`claude/blissful-sutherland-c8ebd9`) sat at `cbf2e6e`, whose tree is **identical
to the merge-base `1ffd414`** (the 4 commits above it are empty merges) and which **tracks
`origin/main`**, not red-team-tier-a. All the referenced work — the frame fix `c3b5a8e`, the
velocity-fork, the re-validation, the twin (`src/racer/twin.py`), **and the updated memory** — lives
on `origin/red-team-tier-a` @ `f137432`, a clean descendant of `1ffd414`. `src/racer/twin.py` and
the two handoff dirs did not exist in my checkout until I moved.

**Action:** `git reset --hard origin/red-team-tier-a` (clean tree, no unique commits, fully
recoverable via reflog + the `nervous-payne` worktree which also points at `cbf2e6e`). Now at
`f137432` with the code, handoffs, and memory the brief assumes. The local `red-team-tier-a` checked
out in the MAIN worktree (`C:/Users/Shadow/Peregrine`) is itself stale at `8d0890f` (7 behind
origin) — the laptop/teammate should `git pull` there. I will commit on top of `f137432` and push
`HEAD:red-team-tier-a`.

---

## The three tasks (in my words)

### Task 1 — sysid extract (OFFLINE, FIRST; unblocks the laptop's twin fit) — **DONE this session**
The laptop will fit the CTBR plant twin (`CtbrPlant`: `rate_gain`, `rate_sign`, `rate_tau`, thrust
`hover`/`slope`) and validate it, but it can't run the sim. So I hand it the data: small downsampled
(~50 Hz) JSON per run with, per sample, `{t_ns, odo_q_wxyz, odo_angular_rate, odo_vel_body,
odo_vel_world, odo_pos, lpn_pos, lpn_vel, actuators[4], cmd_body_rate, cmd_thrust, cmd_age_ms,
ctx}`. Open-loop `*_rate*`/`*_hsweep*` runs (cleanest input→response) for identification, plus
`gate0_course1` (clean, dense, FINISHED) for closed-loop validation. Commands WERE logged
(`commands.jsonl`), so no controller-replay is required, but I include the recorded actuators too.

### Task 2 — gate-center via vision-PnP (OFFLINE) — resolves bottom-left-CORNER vs bottom-CENTER
The map gives gate 0 at `(-23.30, -0.40, -0.03)`. The corner→center code currently applies a
**vertical-only** lift of half the (outer 2.72 m) height → center `(-23.30, -0.40, -1.39)`, i.e. it
assumes the map point is the **bottom-CENTER**. Memory now flags this as suspect: the "no width
offset" verdict came from a **mis-signed width-axis test**, and the z-anchor being the bottom EDGE
hints the map may give a **bottom-left CORNER** that also needs a ±half-width (~±1.36 m along the
gate's +Y width axis) shift. Resolve empirically: run the detector on FPV frames where gate 0 is
well-framed (≥3 corners) → corners → `estimate_gate_pose` (inner 1.5 m PnP) → gate center in camera
→ transform to world NED with the given drone position + ODOMETRY attitude + camera extrinsics
(`frames.R_camera_from_body`), reusing `localization.gate_pose_to_world_position` /
`navigator._process_observation`. Median over several frames. Compare PnP center to the map.

### Task 3 — characterize the §7 altitude balloon (LIVE) — the open gate-0 blocker
In CTBR we send `SET_ATTITUDE_TARGET` (body-rate + explicit collective). The velocity-fork showed
the sim's *velocity* controller drives thrust to max in ANGLE mode — but that's a different control
path. Question: when the drone balloons during a run, is it the **SIM** (an auto-thrust overriding
our collective) or **OUR** controller (vz-damping/alt-PD cratering thrust)? Decisive test: command
zero body-rate (hold level) + a fixed collective, confirm level on the GUI, then step the collective
DOWN (0.27→0.25→0.23→0.21…, ~2 s each), recording commanded collective, actual
`ACTUATOR_OUTPUT_STATUS` (parser-independent witness), world vz (fixed velocity), and GUI behavior.

---

## Inventory — what I have vs what is missing (checked on disk this session)

Recordings live in the MAIN checkout `C:/Users/Shadow/Peregrine/data/runs/` (gitignored; this
worktree shares `.git` but not the working files — I read them by absolute path).

| Need | Status |
|---|---|
| **Task 1** open-loop: `rate1`, `rate2` (doublets), `hsweep1/2/3` (thrust) | ✅ present (commands.jsonl logged) |
| **Task 1** closed-loop: `20260604_025347_gate0_course1` | ✅ present (FINISHED, dense, clean) |
| **Task 2** weights: `models/gate_yolo11s_curriculum_v2.pt` | ✅ present (main repo, 19.7 MB) |
| **Task 2** recording: `20260602_000538_firstcontact_race1` (FPV + given pos) | ✅ present (24.8k frames) |
| **Task 2** runtime: `ultralytics` + `torch` in the ShadowPC venv | ❌ **MISSING** (only `cv2` present) — **BLOCKER** |
| **Task 3** live race + teammate at the GUI | ⏳ needs the teammate (LIVE; hard stop first) |

**Task-1 caveat (honest):** the `hsweep` runs are NOT clean multi-level thrust sweeps — they aborted
early on the ±8 m altitude limit, each holding ~one collective level: `hsweep1`=0.48, `hsweep2`=0.25
(the 0.6 s anti-damping tumble), `hsweep3`=0.25+0.30. So the thrust line must be fit by POOLING the
discrete levels {0.25, 0.30, 0.48} (hsweeps) with the rate runs (0.25) and course1 (~hover 0.26–0.27)
rather than from one clean sweep. Rate doublets (rate1/rate2) are good: roll/pitch/yaw, ~2.4–2.7×
steady gain recoverable.

**Task-2 blocker detail:** weights + recording are here; the detector RUNTIME is not. Options for the
teammate: (a) `pip install ultralytics` (pulls torch; CPU-only is fine for a few frames) into the
ShadowPC venv, or (b) the laptop runs Task 2 itself (it has ultralytics) if the firstcontact
recording is moved over. I did NOT install a multi-GB torch into the shared venv unprompted. The
Task-2 pipeline code (`estimate_gate_pose`, `gate_pose_to_world_position`, `R_camera_from_body`,
`_process_observation`) all exist and are ready to wire.

---

## Pre-registered expectations (locked BEFORE interpreting results)

### Task 1
- Per-run, the rotated ODOMETRY velocity (`odo_vel_world`, the fix) must agree with the independent
  `lpn_vel` (world) to a small RMS on clean runs. **[Confirmed in extraction: course1 0.006 m/s RMS,
  rate1/rate2 ~0.011; hsweeps 0.02–0.06 inflated by 75/97 Hz stagger during fast vertical accel.]**
- The laptop's rate-gain fit should recover **|gain| ≈ 2.4–2.7**, **sign inverted on roll+yaw, not
  pitch**; raw ODOMETRY angular_rate sign-inverted on pitch. **[Coherence-checked from my JSON:
  rate2 roll −2.04 / pitch +1.93 / yaw −1.56 — signs correct; magnitude reads low only because my
  quick fit includes the rise transient, the settled-tail fit will give ~2.7.]**
- Thrust: `a_up = g·(thrust/hover)` with **hover ≈ 0.26**; sparse coverage (see caveat).

### Task 2 — falsifiable decision rule (I do NOT know the answer; memory leans "corner is plausible")
- Report the RAW 3-vector offset `PnP_center_world − map(-23.30,-0.40,-0.03)` (OBSERVATION).
- **CENTER verdict** if the offset is ≈ vertical-only `[0, 0, −1.36]` (|horizontal| ≲ 0.4 m) ⇒ the
  current `corner_to_center` (z-only) is correct, no gate-loader change.
- **CORNER verdict** if there's an extra horizontal ≈ **±1.36 m along world Y** ⇒ map point is a
  bottom-left/right corner; the gate loader needs a ±half-width shift on the +Y width axis (sign per
  which way the offset points). This would mean we've been aiming ~1.36 m off-center (grazing the
  edge — consistent with the user's "miss the gate").
- Instrument discipline: PnP is noisy ⇒ ≥ several frames + MEDIAN; cross-check apparent gate size vs
  range; the detected corners must visually match the gate; keep raw offset (obs) separate from the
  corner/center verdict (interp).

### Task 3 — falsifiable discriminator (genuinely uncertain; prior leans "ours", memo §7 says "sim")
- **OUR-controller verdict** if `ACTUATOR_OUTPUT_STATUS` TRACKS the commanded collective (motors fall
  as we command down; vz responds) ⇒ no sim auto-thrust in CTBR; the balloon was our vz-damping/alt-PD
  cratering thrust. (My prior: in CTBR the memo says "we own thrust," so I lean here.)
- **SIM-auto-thrust verdict** if the actuators DIVERGE upward / stay pinned high while we command low
  (motors don't follow down) ⇒ the sim overrides thrust in CTBR; then find the collective where it
  engages so we can floor just above it.
- Bounded: abort + force-disarm on altitude/tilt/collision; respect the countdown (control before GO
  = DQ); reproduce ≥ 2 races; GUI-corroborate every interpreted run.

---

## Deliverables status
- ✅ Task 1: `sysid/*.json` (6 runs) + `sysid/manifest.json` + `scripts/extract_sysid.py`.
- ⏳ Task 2: blocked on ultralytics/torch (reported); pipeline ready.
- ⏳ Task 3: LIVE; awaits teammate + go.
- Throwaway probes in `scratch/` (clock + extract-coherence checks). Memory NOT edited (laptop
  consolidates).
