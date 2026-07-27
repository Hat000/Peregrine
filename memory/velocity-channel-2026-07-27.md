---
name: velocity-channel-2026-07-27
description: "SSOT for the 2026-07-26/27 cycle — the obs lateral-velocity channel is systemically uninformative (root cause of the at-gate side strike), v2.0 shipped conditionally, the 9-gate arithmetic, and the failure-profile plan."
metadata:
  node_type: memory
  type: project
  originSessionId: b85130f9-ac88-4db2-8c68-0e28b966cf80
  modified: 2026-07-27T01:56:22.043Z
---

# Velocity channel + v2.0 cycle — 2026-07-26/27

Companion to [[deploy-emission-2026-07-22]]. Every claim names its instrument.

## 🛑🛑 ROOT CAUSE — THE OBS LATERAL-VELOCITY CHANNEL CARRIES NO INFORMATION

**Measured (commander, independently, over the corpus).** If the velocity channel were honest,
regressing the vision-measured lateral motion of the gate on `obs[1]` gives slope ≈ **+1.0**. Actual:

| lineage | n ticks | slope | corr | sign disagreement |
|---|---|---|---|---|
| v1.9 (v19Ws0) | **4945** | **−0.110** | −0.057 | **53% (coin flip)** |
| case flight | 97 | −0.466 | −0.161 | 48% |
| v2.0 (v20Vs0) | 312 | +0.216 | +0.161 | 45% |
| v1.6 (record9) | 331 | +0.383 | +0.319 | 58% |

**No lineage is near +1.** The policy's only lateral-velocity input is uncorrelated with actual lateral
motion, and has been across every generation. Caveat: the `d(lat)/dt` proxy is noisy and un-corrected
for yaw rate, so regression dilution biases slope toward 0 — but at n=4945 a strong true relation would
still show. Disagreement is real and systemic.

🟢 **CHANNEL SPLIT CLOSED (Fengyou, 2026-07-27): "vision is pretty bang on, no need to doubt that."**
⇒ the error is in the **KF / dead-reckoning**, NOT vision. Do not re-open the vision-bias branch.
Mechanism: KF dead-reckons with **no lateral reference** (no GPS/mag/baro); the case study attributes
~1.4 m/s of error to a 4–5° AHRS lean during the bank reversal out of the preceding gate.

**Consequence (26-agent case study, 15 claims survived / 3 refuted, flight `20260726_181345_v19pick_Ws0_f1`):**
the policy is told it is sliding TOWARD the gate while it is sliding AWAY, so it under-corrects and
strikes the gate side. 🚩 **Peak roll demand in the failure was 33% of the rail; across 113 flights
NOTHING ever approached the rails.** Authority is abundant — the policy does not ask for it, because it
does not believe it needs to.
🚩 **The famous "centred at 6.7 m (lat −0.21 m)" was a ZERO-CROSSING of a moving error, not a capture.**
The drone was sweeping laterally through the centreline at ~2 m/s, never settled on it.

**EXONERATED by the case study — do not re-investigate:** frame/sign chain end-to-end (machine-precision,
3 independent ways) · policy/wire replay identity (158 ticks to 5.7e-5) · actuation authority/saturation
· the ~2.5× plant gain (matches trained model) · persistent left yaw (trained coordinated-turn, yaw is
bearing-blind by design) · obs builder · staleness as CAUSE (only ~0.07 m of the 0.78 m divergence) ·
course-geometry OOD (that turn-then-short-leg config is 38.6% of training episodes) · EMA.

## 🚩🚩 FRAME GOTCHA — MISREAD THIS AND EVERY LOG ANALYSIS IS WRONG

* Logged **`rel_flu` in ego_obs.jsonl is TRUE body FLU [fwd, left, up], UNFLIPPED.** Negative lat =
  gate to the drone's RIGHT.
* **`obs[0:3]` velocity, `obs[5:8]` rates, `obs[11:14]` slot0 rel_pos are ALL VIRTUAL-FLIPPED**
  (`_RZ_PI_BODY = diag(-1,-1,1)`; `src/racer/ego_obs.py:36`, `:124-127`). **So true v_left = −obs[1].**
* The two live in DIFFERENT conventions in the same log line. The commander initially read them in the
  same convention and got the opposite conclusion.

## 🚩 THE 9-GATE ARITHMETIC — RECORDS ARE LUCK, NOT CAPABILITY

Per-gate conditional survival over the **whole 573-flight corpus**:

```
gate:    0     1     2     3     4     5     6     7
P(pass) 71%   66%   67%   56%   46%   20%   44%   50%
```

Compounding ⇒ **P(reach gate 9) = 0.35% ≈ 1 in 285.** Over 573 flights that predicts **2**; we observed
**2**. (Tautological as a fit — survival was derived from the same data — but the MAGNITUDE is the
point: two 9-gate runs in 573 attempts is unremarkable.) **The 9-gate flights happened DESPITE the
flaw.** Consistent with the older "a record is not a baseline / 1-in-11 tail" note, now corpus-wide.

🚩 **WHY THE FLAW DOESN'T KILL EVERY GATE — IT IS CONDITIONAL, NOT CONSTANT.** The policy has two
lateral inputs: POSITION (vision, good) and VELOCITY (broken). Position feedback alone threads a gate
you are already lined up on; the velocity term only becomes load-bearing when carrying real **crossing
velocity**, i.e. coming out of a turn. Hence survival degrades monotonically with depth (71→66→67→56→
46→20%): deeper = faster + more turn-exit states = more exposure. **Net effect: the broken channel
converts a ~1-in-3 clean course into ~1-in-285.**

## 🏁 v2.0 — SHIPPED CONDITIONALLY 2026-07-26 (`ego-ckpts-v20-2026-07-26`, asset v20Vs0, sha 05ac7ec3)

**Pilot authorised shipping over the commander's stated objection ("ship it anyways, with a warning").**
v19Ws0 remains the SAFE lead.

**Head-to-head (job 3324162, 3 policies × 2 envs, 2048 envs each, eval env replayed from each run's own
recorded `.hydra/overrides.yaml` — no hand transcription):**

| policy | VERT env gates | FLAT env gates |
|---|---|---|
| **v20Vs0** | **5.884** | **5.973** |
| v20Vs1 | 5.363 | 5.373 |
| v19Ws0 | 4.893 | 5.880 |

🚩 **v19Ws0 is COURSE-FRAGILE (5.88 → 4.89 on climbs); v20Vs0 is COURSE-ROBUST (5.97 → 5.88).** v20Vs0
also has the lowest roll signflips in both envs. v20Vs1 not shipped (worse on flat and on the dive).

🛑 **BUT THE RELEASE DIVE REGRESSED ~2×** (451-flight replay, `scratchpad/replay_v20.py`): v20Vs0 mean
**−1.064** / MAX −1.963, **dives harder than v19Ws0 on 450/451 flights**; v20Vs1 451/451. Same signature
that got the v1.9 Q arms rejected. **First 6 wire flights: 3/6 died at gate 0 in ~2 s** = the regression
showing on the wire (n=6, 3 different clamps, soft).
🚩 **CORRECTION TO A PRIOR CLAIM:** the v2.0 course change was verified to leave spawn pose + gate-0
geometry bit-identical, and that was reported as preserving launch competence "by construction". True
of the COURSE, **FALSE of the POLICY** — 18k updates changed the learned pitch temperament. Bit-identical
geometry ≠ bit-identical behaviour.
**Hypothesis for v2.1:** existing pitch penalties (`rw_pitch_jerk`, `pitch_duty_free_band`) price the
RATE of pitch change, not commanded direction at release; climbing training widens the pitch envelope
both ways.

**The vertical fix itself (branch `train-vertical-2026-07-25`, `rl/launch_v20.sh`):**
🛑 `course_gates_above_spawn` was **NOT** the cause of vert-sector starvation (refuted by bit-identity
test — the floor clamp only SHRINKS descents, can never create a climb). Real cause: descent-biased
band `drop_m=(-3,+12)` vs the 0.20 rad deadband needing `dz > tan(0.20)·L`. Fix =
`course_drop_lo/hi=-10/+10` + NEW `course_gates_ceiling=16.0` ⇒ **`+1` 1.6% → 30.0%** (8-gate courses),
consecutive-UP courses 1.6% → 56.9%. `exit_ceiling` stayed 0.0000 (policy flies through, not out).
🚩 **Control tick is 30.03 Hz, not 40** (stale comment corrected). `ego_vision_frame_hz=30` fired on
99.9% of ticks = a near-NO-OP; **25 Hz is exact** (6/5 → 5 frames per 6 ticks), fresh-fix 10.5→8.75 Hz.
IMU cadence IS modelled (n_substeps=5 ≈150 Hz, last-substep sample once per control tick) and must NOT
be "fixed" — it reproduces the wire leveler to median 0.0003°.
🛑 **NEVER move `ego_coarse_vert_thresh_rad` (0.20) to raise +1** — deploy reads a HAND-AUTHORED coarse
map on that convention. **Move the GEOMETRY, never the label.**

## 📋 THE FAILURE-PROFILE PLAN (in flight at compaction)

**Why it starts with instruments:** two prior detectors disagreed **4× (10 vs 41 of 114)** on the
lateral family, because the vision-based one measured **ESTIMATE motion, not DRONE motion**. Also
🚩 **31 of 78 at-gate deaths have logs ending 0.1–0.15 s BEFORE impact** — the kill axis is unobservable
for 40% of our best events.

* **Phase 0a** (agent `p0-classifier`, running): one definitive classifier separating real vs estimate
  motion using **gyro + vision bearing-rate** (never `obs[0:3]`), hand-validated on ~20 labelled
  approaches, then diagnose the 10-vs-41 split and census all ~650 sessions by mode × lineage.
* **Phase 0b** (agent `p0-recorder`, running): record ~0.5 s PAST the terminal condition, post-terminal
  ticks flagged; must not touch control timing/commands/termination.
* **Phase 1**: roll-authority ceiling (probe LOGGED obs perturbed, never synthetic — OOD trap),
  was-it-ever-centred, kill-axis bounds.
* **Phase 2** (cluster, gates a 6 GPU-h decision): does SIM ever produce the precursor state
  (`|y|<0.5, |vy|>1.5` at crossing)? One-line logging + rollout census. **If rare in sim, skip the
  training-side change entirely.**
* **Phase 3**: 🛑 **MIRRORED COURSE IS DEAD — the course is FIXED** (Fengyou). Replacement probes:
  **offline video re-detection** (needs Fengyou to push the ~3 MB video blob for
  `20260726_181345_v19pick_Ws0_f1`; only the .jsonl index reaches git) — though with vision now cleared
  this is DEMOTED — and an `aim_off = (±2, 0)` semi-truth anchor probe.
* **Phase 4**: merge census × fix-map ⇒ expected flights recovered per fix.

**D1** (agent `d1-velocity`, running, branch `d1-velocity-2026-07-27`): make obs lateral velocity
truthful by fusing seeker vision innovations into nav velocity — **estimator-faithful, NO GT**,
default-OFF, validated OFFLINE on the corpus before flying. Acceptance: slope moves toward +1.0,
sign-disagreement well under 53%, and roll-correction onset moves 0.3–0.7 s earlier on failing
approaches. **A negative result must be reported plainly, not laundered.**

## 🚀 CLUSTER OPS ADDENDA
* Adroit took a `Reboot ASAP` maintenance drain 2026-07-26; all GPU nodes recovered by evening.
* 🚩 **Backfill beats pinning:** requesting `--gres=gpu:1` (any type) with a SHORT wall got a queued
  sweep through hours earlier than a V100-pinned 3 h job. Check per-node TURNOVER (`squeue -t RUNNING`
  TIME_LEFT), not just idle CPUs — V100 jobs ran 8–12 h while A100 jobs turned over in 3–4 h. Idle CPU
  counts are a red herring; GPUs are the binding resource.
* 🚩 SLURM refuses `--dependency=afterok:<job>` on a job that has already COMPLETED and left the active
  list ("Job dependency problem"). Use the launcher's `SMOKE_JID=none` escape hatch after adjudicating
  the smoke by hand.
* 🚩 `launch_v20.sh` carries a default `--exclude=adroit-h11g3`; pass empty `SBATCH_EXCLUDE=` to submit
  to V100.
* 🚩 Daemon: a stale `serve` from a previous day can hold port 8765 while its SSH session is dead —
  symptom is `ConnectionResetError`/`WinError 10054`, NOT `unauthorized`. `netstat -ano | findstr 8765`,
  kill the old PID, restart. `scratchpad/tb_scalars.py` must be pushed to `/scratch/network/fl3689/`.
* 🚩 The adroit connector strips quotes — avoid parentheses and nested quotes in remote `echo` strings.

## Open at compaction
* C control arm **3323864** (v20C_s0_v100) RUNNING — the clean "did vertical structure cause the gain"
  control. Adjudicate V vs C when done.
* Three agents running: `d1-velocity`, `p0-classifier`, `p0-recorder`.
* v2.1 (release-dive fix on top of v2.0's vertical structure) not yet specified.
