---
name: velocity-channel-2026-07-27
description: "SSOT for the 2026-07-26/27 cycle — the lateral-velocity root cause REFUTED then RE-DERIVED as a train/deploy parity gap + GT leak in the training actor obs, v2.0 shipped conditionally, the 9-gate arithmetic, and the failure-profile plan."
metadata:
  node_type: memory
  type: project
  originSessionId: b85130f9-ac88-4db2-8c68-0e28b966cf80
  modified: 2026-07-27T03:13:34.315Z
---

# Velocity channel + v2.0 cycle — 2026-07-26/27

Companion to [[deploy-emission-2026-07-22]]. Every claim names its instrument.

## 🛑 REFUTED — "the obs lateral-velocity channel carries no information"

**The 2026-07-27 ROOT CAUSE (slope −0.110 / corr −0.057 / 53% sign-flip over 4945 v1.9 ticks) was a
ROTATION ARTIFACT and is WITHDRAWN.** It came from differencing the gate lever
(`d(rel_flu[1])/dt`) **without gyro de-rotation**. A gate is a world-fixed landmark: when the drone
ROTATES, the lever sweeps without the drone translating. At 8 m, **0.3 rad/s of yaw = 2.4 m/s of fake
lateral motion**, against a real lateral-velocity std of only ~1.7 m/s. And because the controller
**yaws to correct lateral drift**, the fake term is **anti-correlated with the truth by construction** —
which is precisely the negative slope that was reported.

**Independent confirmation (commander, own instrument, 563 flights / 24 799 fresh-fix same-gate tick
pairs, stratified by the contaminant instead of compensating for it):**

| \|yaw rate\| bin (rad/s) | n | slope | corr | sign-disagree |
|---|---|---|---|---|
| 0.00 – 0.10 | 6140 | **+0.496** | +0.203 | 42.9% |
| 0.10 – 0.25 | 7431 | +0.494 | +0.182 | 52.2% |
| 0.25 – 0.50 | 7695 | +0.414 | +0.153 | 52.9% |
| 0.50 – 1.00 | 3298 | +0.281 | +0.099 | 53.2% |
| **> 1.00** | **235** | **−0.767** | −0.188 | 69.8% |

**Monotone in the contaminant, and the negative slope lives ENTIRELY in the top bin — 235 of 24 799
pairs (0.9% of the data).** Pooled slope on a clean filter is **+0.366**, not −0.110. The D1 agent's
de-rotated measurement puts it at **+0.93 / corr 0.76 at 4–8 m** and **+0.99 / 0.91 inside 4 m**; the
commander's cruder single-tick instrument retains residual coupling and reads a floor of +0.50. Both
instruments agree on direction. **The channel is not blind.**
Repro: `scratchpad/verify_rotation_artifact.py`, and `scripts/d1_velocity_replay.py --no-rotcomp`
(which reproduces the withdrawn number exactly — 54.2% sign-flip).

🚩 **Any other case-study claim derived from un-de-rotated lever differencing is suspect and must be
re-measured before use.**

## 🛑🛑 ROOT CAUSE (CORRECTED) — TRAIN/DEPLOY PARITY GAP ON obs[0:3], *AND* A GT LEAK

Read structurally from both code paths — **no statistics required**:

* **DEPLOY**: actor velocity is `nav_state.velocity_ned` (`rl/fly_rl.py:2857`) from the **map-free**
  navigator. `rl/fly_rl.py:1920` states it outright: *"the live VQ2 wire the navigator runs map-free and
  `nav.time_since_vision_update_s` never goes finite."* Confirmed: `tsv` is `None` on **100%** of ticks.
  No GPS, no mag, no baro, no world fix ⇒ **obs[0:3] is OPEN-LOOP IMU strapdown with no velocity
  reference of any kind.**
* **TRAINING**: `vel_model` defaults to **`"legacy"`** unless `env.ego_faithful=true`
  (`rl/peregrine_racing_ego.py:1353`). Legacy **seeds `_vel_body` at exact truth**
  (`rl/ego_estimator.py:471`, `R_wb^T @ drone_vel`) and **pulls it 15% toward TRUE body velocity at
  every accepted fix** (`:815-819`, `vel_correct_gain=0.15`). Gate fixes land on **92.3% of ticks** ⇒ a
  truth-tracking servo with **τ ≈ 0.23 s** and steady-state error **~0.01 m/s**.
* 🛑 **NO launcher in the v16→v20 lineage arms `ego_faithful`** — `launch_v20.sh` sets
  `faithful_rate` + `ego_vision_cadence` only. Every shipped policy trained on `legacy`.

⇒ **~100× parity gap** (training ~0.01 m/s vs deploy ~1.4 m/s rms lateral error), **and** it is
**ground truth inside the actor observation** — a direct violation of [[feedback-no-gt-actor-obs]].

🚩 **The legacy model emulates a deploy mechanism THAT DOES NOT EXIST.** Its own comment justifies the
gain as modelling *"the deploy KF corrects velocity SOLELY through gate-relative POSITION fixes via the
pos-vel cross-covariance."* On the map-free wire there is no such correction — the gate-relative path
feeds obs directly and never bleeds into velocity.

**Sufficiency**: ~1–1.4 m/s of lateral error over a 1–2 s approach ⇒ 1–2 m of belief error against a
gate opening of order 1.5 m. **Enough to strike a side on its own.** The failure mechanism survives
intact; only its explanation changed — the policy does not get a *sign-flipped* channel, it gets an
*honest-but-un-referenced* one that it was trained to trust as near-truth.

🟢 **CHANNEL SPLIT CLOSED (Fengyou, 2026-07-27): "vision is pretty bang on, no need to doubt that."**
⇒ error is in the **KF / dead-reckoning**, NOT vision. Do not re-open the vision-bias branch.

**Consequence (26-agent case study, flight `20260726_181345_v19pick_Ws0_f1`):** the policy under-corrects
and strikes the gate side. 🚩 **Peak roll demand in the failure was 33% of the rail; across 113 flights
NOTHING ever approached the rails.** Authority is abundant — the policy does not ask for it.
🚩 **The famous "centred at 6.7 m (lat −0.21 m)" was a ZERO-CROSSING of a moving error, not a capture.**

**EXONERATED by the case study — do not re-investigate:** frame/sign chain end-to-end (machine-precision,
3 independent ways) · policy/wire replay identity (158 ticks to 5.7e-5) · actuation authority/saturation
· the ~2.5× plant gain (matches trained model) · persistent left yaw (trained coordinated-turn, yaw is
bearing-blind by design) · obs builder · staleness as CAUSE (only ~0.07 m of the 0.78 m divergence) ·
course-geometry OOD (that turn-then-short-leg config is 38.6% of training episodes) · EMA.

## ▶️ THE ARM THIS PRESCRIBES — `ego_faithful` / `vel_model='kf'`

`vel_model='kf'` already exists, is DEFAULT-OFF, and **has never been used in any run**. Its contract:
*"the translated deploy LinearKF … strapdown predict through the EMULATED attitude + position-fix-only
corrections … velocity projected world→body through the SAME emulated attitude — both deploy error
injections reproduced."* It **never touches truth**, so it closes the GT-leak rule violation and most of
the parity gap in one arm. 🚩 Caveat before launching: it still applies *position-fix* corrections that
the map-free wire does not have, so it remains optimistic — verify how much, and consider a pure-
strapdown variant. `att_model='eskf'` + `rate_model='sampled'` come with the same master switch;
arm them per-channel for ablation, not blind.

**D1 (`--ego-vel-fuse`, branch `d1-velocity-2026-07-27`, `b5fc6bfd`) is the DEPLOY-side half-measure**
and is shipped **DEFAULT-OFF**: real but small (sign-disagree 12.9→11.6%, corr +0.02, rms −4%, −13%
inside 4 m; placebo-verified) and the actor roll onset moved only **+0.167 s** vs the 0.3–0.7 s the case
study predicted. **Do NOT fly it as the gate-3 fix.**

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

## 🎥 POST-IMPACT RECORDER — `c0d3f6ef`, branch `recorder-postimpact-2026-07-27`

**The impact was IMPOSSIBLE to log, not merely missed.** Every `_fly_ego` stop condition `break`s at the
**TOP** of the tick body (`rl/fly_rl.py:2631-2634`) while the forensics row is appended at the **BOTTOM**
(`:2925`, ~290 lines later) ⇒ the detecting tick is discarded WHOLE and recording ends forever.

* **533 banked CRASH sessions end with a median 0.223 s of time-to-contact still remaining.**
* `video_index.jsonl` runs **70–130 ms LONGER** than `ego_obs.jsonl` ⇒ **the impact is probably already
  ON VIDEO in flights we already hold.** Recoverable retroactively; validates the 0.5 s window.
* **2nd gap closed:** the ego path persisted only an integer collision COUNT ⇒ **gate (id 1001) vs
  environment (1002) was unrecoverable corpus-wide** — most of what failure classification needs.
  Now persisted with impulse, per-tick `gyro_frd`/`accel_frd` through impact (the axis discriminator),
  and `frame_id` to join to the impact video frame.
* 🚩 **No contact POINT exists on the wire** — MAVLink `COLLISION` has no position field. A true contact
  point would require the sim to publish one.
* Rows go to a **SEPARATE `ego_postimpact.jsonl`** — merging into `ego_obs.jsonl` would hard-error
  `tape_extract.analyze()` (missing command keys) or replay a crashed tumble as a `--sysid-replay` tape.
* Default **ON at 0.5 s, CRASH-only**; `--ego-post-terminal-s 0` restores the old teardown byte-for-byte.
  19 new tests; suite 1922 passed / 1 pre-existing failure.
* **Commander ACCEPTED the residual**: during the window the vehicle stays armed with its last command
  latched. Sim-only, CRASH-only, run already INVALID, collision ledger restored ⇒ no downstream metric
  moves. → [[feedback-sim-no-physical-risk]]

## 🛑🛑 THE FIVE WITHDRAWN PREMISES — 2026-07-26/27, full record

**Not one was a wrong number.** Each was a correct number carrying a mechanism claim its artifact never
measured. Craft lesson lives in `COMMANDER.md` §2 (Gen 7).

| # | premise | what actually produced the number |
|---|---|---|
| 1 | "obs lateral-velocity carries no information" (slope −0.110, 53% sign-flip) | **rotation contamination** — de-rotated it reads **+0.93** @4-8 m |
| 2 | "vertical undershoot dominates" | **pitch coupling** — 66.2% of CONFIRMED PASSES misclassify |
| 3 | "launch/release window = 20.1%, the biggest mode" | **classifier RULE ORDER** — a temporal rule pre-empting the geometric ones; those flights die at range p50 **1.50 m** = at-gate death at gate 0 |
| 4 | "speed kills at gate 0" (AUC 0.723) | **gate-seam teleport** — seam-free AUC **0.565**, sign reverses |
| 5 | "GT leak / 100× parity gap on obs[0:3]" | **read the LAUNCHER, not the run that ran** |

### #5 in full — refuted four independent ways

* `.hydra/overrides.yaml` **and** resolved `config.yaml` carry `ego_faithful: true` — v19Ws0 and both v20V seeds.
* The **STAGE dict** `dual_gate_fullstack_floor_pef16` arms it (`rl/vq2_ego_curriculum.py:1390`). **`_pef` = _percept + estimator-faithful** — the name said so all along.
* The `kf` path **RAISES without `sf_body`** (`ego_estimator.py:652`); the stage supplies `capture_specific_force` + `n_substeps=5`. **A run that trained to completion cannot have been on `legacy`.** (A proof, not a correlation.)
* `env_loss/kf_vel_err_mean` is **PRESENT, n=1800, in all three runs** — the tag is gated on `if self._est_vel_model == "kf"` (`peregrine_racing_ego.py:2393`).

🛑🛑 **ADJUDICATE FROM `.hydra` + THE STAGE DICT, NEVER LAUNCHER SOURCE. "No launcher sets X" is a NULL
RESULT, not evidence.** 🚩 `ego_vel_model` is **ABSENT** from these configs — it resolves from
`ego_faithful` via `"kf" if _faithful else "legacy"` (`:1353`). **Absence of the key is not absence of
the path.**

📏 **AUTHORITATIVE training velocity error = 0.28–0.30 m/s at convergence** (v19Ws0 **0.2970** · v20Vs0
**0.2761** · v20Vs1 **0.2937**), falling from ~0.72–0.83; **max == first in every run** ⇒ it never rose,
and last is within 3–8% of min ⇒ converged. **v2.0-vs-v1.9 gap is smaller than the v2.0 seed spread —
read NO estimator improvement into it.** Replaces the 0.01 m/s figure, which was never re-derivable.

🚩 **THE PARITY GAP SURVIVES, RESTATED AND SHARPER:** training obs[0:3] is a KF **corrected** by a stream
of vision POSITION fixes (`ego_estimator.py:777-806`, via the pos/vel cross-covariance); deploy folds in
**zero** (measured 0 of 94 991 ticks). ⇒ **~0.29 m/s vs ~1.4 m/s rms ≈ 5×**, not 100×. And the wire's
inability is **structural, not a bug**: the fix needs `z_datum = gate_pos_datum − R_datum @ fix_body`,
which requires **a MAP**, and the deploy profile is self-localizing/map-free (`fly_rl` builds
`gates = []`). ⇒ **D1's gate-RELATIVE displacement is the map-free analog of exactly that correction** —
which motivates D1 better than its original premise did, even though its measured gain stayed small.

🚩 **TB/helper footgun:** `tb_scalars.py` lives at **`/scratch/network/fl3689/tb_scalars.py`**, NOT under
`peregrine_repo/scratchpad/` (that copy has no `scratchpad/`). ~2 s on a 10 MB event file, login-node
safe. It prints n/first/last/min/max with **NO step numbers** — derive the axis from `n_updates` in
`.hydra/config.yaml` (18000 updates ÷ 1800 `env_loss` points = every 10). v1.9→v2.0 instrumentation
delta is **4 vertical-course tags only**; estimator instrumentation is identical ⇒ apples-to-apples.

## 🛑 OPS — ONE WORKTREE PER AGENT (violated 2026-07-27, near miss)

`d1-velocity` and `p0-recorder` were both pointed at `wt-arrest`. The recorder's `git checkout -b` moved
HEAD out from under the live D1 agent, and separately reverted `rl/fly_rl.py` mid-edit, destroying D1's
changes once. Both agents self-detected and repaired; nothing was lost. **The commander caused this by
sharing a checkout.** Rule: one worktree per concurrent agent, and agents commit early when sharing.

## Open at compaction
* C control arm **3323864** (v20C_s0_v100) RUNNING — the clean "did vertical structure cause the gain"
  control. Adjudicate V vs C when done.
* `p0-classifier` still running (real-vs-estimate-motion classifier + 650-session census).
* **v2.1 spec** — release-dive fix on top of v2.0's vertical structure — not yet written.
* **The `ego_faithful` arm** (above) — the highest-value open lever, not yet launched.
