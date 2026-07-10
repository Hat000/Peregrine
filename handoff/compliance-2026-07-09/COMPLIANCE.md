# VQ2 Compliance Package — Team Peregrine

- **Date:** 2026-07-10 (drafted against branch `claude/ego-deploy-2026-07-09`, HEAD `e45398f`)
- **Governing documents:** VADR-TS-003 (`260624_Technical_Spec_0003.pdf`, repo root; 12 pp, quotes verified by direct read 2026-07-10) + AI Grand Prix Official Rules (theaigrandprix.com/official-rules; grounded facts banked in `memory/reference_competition_materials.md`).
- **Status:** DRAFT for Fengyou's review. Nothing here has been sent to the organizers. The disclosure paragraph in §B is ready-to-send once reviewed.

---

## A. §7 Autonomy checklist

VADR-TS-003 §7 (p. 11, verbatim):

> "Participants must ensure their implementation conforms to this specification. Specifically, human interaction during the flight which the participants submit as a timed run is grounds for immediate disqualification."

Every requirement mapped to the code fact that satisfies it (all facts re-verified against `rl/fly_rl.py` on this branch, 2026-07-10; independently confirmed by the 5-agent adversarial audit of 2026-07-09, `handoff/ego-deploy-2026-07-09/REPORT.md`, zero flight-blocking findings):

| # | Requirement | Code fact | Where |
|---|---|---|---|
| 1 | No human GO — the run starts on the organizer's race GO, not an operator action | Default posture is a passive wait: `">>> Waiting PASSIVELY for the race GO (no sim-control command will be sent; this is the submission-safe posture)."` | `rl/fly_rl.py:950-951` (`wait_fresh_go`) |
| 2 | No sim-control commands on the submitted path | The ONLY sim-control command in the codebase, MAV_CMD 31000 (sim restart), lives inside `wait_fresh_go` and is gated behind `--dev-auto-reset` (`action="store_true"`, **default OFF**); `--no-auto-reset` is a hard override that always wins | `rl/fly_rl.py:941-951` (gate), `:2212-2219` (flags) |
| 3 | Self-arming, no human hand on the stick | Client arms itself via MAV_CMD_COMPONENT_ARM_DISARM (400) p1=1, with bounded re-send + ACK branching (F-D/AR1) | `rl/fly_rl.py:963-972` |
| 4 | Fully autonomous flight | Control = streamed `SET_ATTITUDE_TARGET` BODY-RATE (type_mask=0b10000000, FRD rates + thrust); observations from the permitted wire only; policy = onboard RL actor + YOLO-pose vision | `rl/fly_rl.py` header (lines 8-14) + control loop |
| 5 | No human input paths during a run | No stdin/keyboard polling anywhere in the flight loop; nothing blocks on operator input between GO and disarm | 5-agent audit 2026-07-09 (`handoff/ego-deploy-2026-07-09/REPORT.md`) |
| 6 | Clean, autonomous termination on every exit path | `finally`-owned force-disarm wraps the entire armed section (F-B); per-flight backstop + top-level backstop disarm | `rl/fly_rl.py:998-1004`, `:2523-2542`, `:2568-2578` |

**THE ONE OPERATIONAL RULE for Competitive submits:** launch `fly_rl.py` with **defaults** — never pass `--dev-auto-reset` — confirm the "Waiting PASSIVELY" banner prints, and the operator touches **nothing** after the race GO. (This is the existing launch ritual: `fly_rl` FIRST → banner → THEN GO.)

---

## B. FLOSS inventory + written disclosure

Obligation source — Official Rules (grounded 2026-05-28/2026-06-14, `memory/reference_competition_materials.md:37,85`): external libraries and gen-AI coding tools are permitted; **FLOSS is allowed BUT must be disclosed in writing to the organizers and must not violate third-party license terms.** (The disclosure clause is in the Official Rules, not VADR-TS-003 — a full-text search of TS-003 for FLOSS/open-source/disclosure terms returns nothing.)

### Inventory

Scanned: `pyproject.toml` (sole dependency manifest — no requirements*.txt / setup.py in the repo) + the import graph of the deploy stack (`rl/fly_rl.py`, `src/racer/**`). Licenses verified 2026-07-10 from installed package metadata (`importlib.metadata`) in the main venv `C:/Users/Fengy/Downloads/Projects/Anduril/.venv` — not from memory.

| Package | Installed version | Manifest pin | License (from metadata) | Role |
|---|---|---|---|---|
| ultralytics | 8.4.57 | `>=8.3.0` (detector extra) | **AGPL-3.0** | YOLO-pose gate detector — training AND runtime inference |
| torch | 2.12.0 | out-of-band CUDA wheel (per-platform) | BSD-3-Clause | NN inference (RL policy + detector backbone) |
| torchvision | 0.27.0 | pulled with torch | BSD | torch companion (detector pipeline) |
| opencv-python | 4.13.0.92 | `>=4.9` | Apache-2.0 | JPEG decode, image ops |
| numpy | 2.4.6 | `>=2.0` | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | numerics throughout |
| scipy | 1.17.1 | `>=1.13` | BSD-3-Clause | rotations, filtering |
| pymavlink | 2.4.49 | `>=2.4.40` | LGPL-3.0 | MAVLink client (sim wire) |
| albumentations | 2.0.8 | `>=1.4` (detector extra) | MIT | training-only data augmentation |
| pytest | 9.0.3 | `>=8.0` (dev extra) | MIT | dev/test only, not in the deploy stack |

Not verified (flagged, not guessed):

- **toppra** (`>=0.6.0`, planning extra, `platform_system != 'Windows'` only) — NOT installed in the main venv, so its license could not be read from package metadata. It is not in the Windows deploy stack. Verify from metadata before disclosure **if** it is ever installed/used on an eval-relevant box.
- **Blender (`bpy`/`bpy_extras`/`mathutils`)** — imported by the training-data render pipeline (`src/racer/**/bpy_*.py`), used as an external application, not a pip dependency of the entry; GPL-licensed upstream. Training-data tooling only; nothing Blender-derived executes at race time.

**AGPL-3.0 note (ultralytics — the load-bearing one):** AGPL's source-provision obligations attach on *distribution* or on offering the software as a *network service* to third parties. Competition use is internal/non-distributed — the organizers receive a judging-period run/review license under the Official Rules (IP clause, `reference_competition_materials.md:38`), which is not public distribution. The concrete obligation for this entry is therefore the **written disclosure below** plus staying license-compliant (unmodified ultralytics used as a library; if we ever publicly distribute the stack or serve it over a network, full AGPL source obligations attach and must be revisited).

### Ready-to-send written disclosure (per the Official Rules' FLOSS clause)

> **FLOSS disclosure — Team Peregrine (AI Grand Prix, Round 2 virtual qualifier).**
> Per the Official Rules' requirement that free/libre/open-source software be disclosed in writing, Team Peregrine discloses that its entry is original work that uses the following open-source components as libraries, unmodified: Ultralytics YOLO (AGPL-3.0) for gate detection; PyTorch and torchvision (BSD-3-Clause/BSD) for neural-network inference; OpenCV (opencv-python, Apache-2.0) for image processing; NumPy and SciPy (BSD-family) for numerics; pymavlink (LGPL-3.0) for MAVLink communication with the simulator; and, in offline training tooling only (not executed at race time), albumentations (MIT) and Blender (GPL) for training-data generation. Generative-AI coding tools (Anthropic Claude) were used in development, disclosed per the same clause. All components are used in compliance with their license terms; the entry is not publicly distributed, and no third-party license term is violated by its use in this competition.

---

## C. §9.2 code-audit readiness

VADR-TS-003 §9.2 (p. 12, verbatim):

> "⚠ Important Note on Code Integrity: Once you submit a time-tracked race for qualification, DCL reserves the right to review your codebase. If the Race Director suspects any form of cheating or manipulation of the simulator constraints, a formal code audit will be triggered."

Our position, point by point — the codebase is auditable as-is:

**1. No simulator-constraint manipulation anywhere.** The client consumes only the permitted wire — HIGHRES_IMU (accel+gyro), the 30 Hz JPEG vision stream (UDP:5600), RACE_STATUS, HEARTBEAT/TIMESYNC — and sends only ARM (MAV_CMD 400), `SET_ATTITUDE_TARGET`, heartbeats, and TIMESYNC. The §9.3 blocked messages ("ATTITUDE, LOCAL_POSITION_NED, ODOMETRY, GATE_INFO", p. 12) are neither requested nor required: the VQ2 stack derives attitude, yaw, and altitude from vision + IMU (ESKF leveler + YOLO-pose/PnP + KF), by design, because the VQ2 wire carries no mag/baro and no position (live-confirmed 2026-06-29, `memory/reference_sim_interface.md`). Legacy code paths that *read* ODOMETRY/LOCAL_POSITION_NED exist for the practice/VQ1 sim, where those messages are broadcast by the sim itself; they are passive consumers, inert on the VQ2 wire, and manipulate nothing. **No ground truth ever rides the VQ2 wire** (vision-commander relay, 2026-07-09).

**2. The one sim-control command is dev-gated, default OFF.** MAV_CMD 31000 (sim restart, an organizer-documented interface) appears in exactly one place — `wait_fresh_go` in `rl/fly_rl.py` — behind `--dev-auto-reset` (default OFF) with a `--no-auto-reset` hard override. The submitted/competitive launch posture is passive (§A rows 1-2). Nothing in the codebase touches sim state, files, memory, clocks, or physics.

**3. Track-knowledge provenance policy.** All track knowledge deployed at race time derives exclusively from (a) **Training-mode flights** (§9.2's explicitly free, non-counted mode) and (b) **offline processing of our own flight logs** between runs — which the organizers confirmed is legal (COWORK Q④ resolution: "offline log-based retuning BETWEEN runs ALLOWED", `memory/reference_competition_materials.md:85`; standing user directive "offline processing legal"). The deploy path contains **no simulator-file extraction**: nothing reads, parses, or introspects the simulator's installation files or assets. (The FAQ itself notes teams can download the simulator package including the course; regardless, our race-time stack learns the course only by flying it.)

**4. Determinism is observed, not manipulated.** The course is deterministic per load — established by flying it (VQ1 first contact 2026-06-02: "deterministic course (capture once, reuse)", `memory/reference_sim_interface.md`; VQ2 recon 2026-06-29: track "Now You See Me, Now You Don't", stations 01-20, deterministic per-load). Exploiting run-to-run repeatability through repeated attempts is squarely within §9.4's "unlimited number of attempts to set their fastest time."

**5. §9.4 scoring (for the record, p. 12 verbatim):** "Leaderboard (Qualification Event) ranking is strictly determined based on the timing results of valid, completed runs. • Faster times rank higher. • Attempts: Contestants are permitted an unlimited number of attempts to set their fastest time. • Team Scoring: ... The best single timing result from a team dictates that team's overall rank..." — i.e., pure time-trial, best valid time, unlimited attempts. This supersedes the 2026-06-15 "time-lock" precaution (a completed run can only *improve* our standing); Training completions additionally never count (§9.2), pending the email confirmation in §D.

---

## D. Open items

| Item | Status | Action / owner |
|---|---|---|
| Exact VQ2 close date/time + timezone | UNKNOWN (estimate: mid/late July 2026) | Organizer email Q1 (`organizer-email-draft.md`) — note: 3 prior emails unanswered |
| Teams advancing to the September Physical Qualifier | UNKNOWN | Organizer email Q2 |
| Training vs Competitive separation (Training completions never post to leaderboard) | Spec §9.2 says Training is "not counted"; belt-and-suspenders confirmation pending | Organizer email Q3 + **one look at the Training/Qualification event-block UI before the FIRST Competitive submit** (verify the two "manually selectable event blocks" behave as §9.2 describes) |
| Cut line (how fast is fast enough to advance) | UNKNOWN | Monitor leaderboard; BANK-FIRST doctrine (bank a completed time early, then iterate) |
| Competitive submission mechanics (run live from our box vs uploaded stack, per the VQ1-era "integrated into the simulator" language) | Partially known; §9.2 implies self-serve event selection | Verify at the Training→Competitive transition; parked backlog #61 |
| toppra license verification | Not installed in main venv; not in the deploy stack | Verify from package metadata before disclosure if ever used on an eval box |
| Send the FLOSS disclosure (§B) | Draft ready | Fengyou reviews + sends before (or with) the first Competitive submit |
