---
name: durables-operational
description: "Standing operational durables from Track A and the v1→v1.8 campaign — flight/recipe ritual, run-selection rules, training-diagnosis rules, and the small hard-won facts that have no other home."
metadata: 
  node_type: memory
  type: project
  originSessionId: b85130f9-ac88-4db2-8c68-0e28b966cf80
  modified: 2026-07-27T02:40:57.283Z
---

# Operational durables (relocated from MEMORY.md, 2026-07-27 compaction)

Nothing here is stale — it is standing practice. It lives one layer down because MEMORY.md is a thin
index. Thread history: [[track-a-beat-vpeffs0-2026-07-13]] · [[replay-ratchet-2026-07-17]] ·
[[trackA-stall-forensic-2026-07-13]] · [[ego-deploy-contract-2026-07-09]] · [[audit-ego-inc9-2026-07-09]].

## Flight + recipe ritual

* 🚩 **PIN "GO ≥3 s post-reset"** — wake-0 variance drops ~1000×.
* 🚩 **LAUNCH RITUAL: `fly_rl` FIRST → wait for the "Waiting PASSIVELY" banner → THEN GO.**
* 🚩 **Recipe TRUTH = the PANEL LOG, not `meta.json`.** RECIPE (`.hydra`): `ticks_hi=1`, yaw 0.7,
  `blur=false`.
* 🚩 **The pitch clamp is PER-RELEASE — mixing releases is OOD.**
* 🚩 **RE-FLY RULE = ‖gyro‖ > 2.5 for ≥3 CONSECUTIVE ticks.**
* 🚩 **GATE CONTACT = INVALID RUN.**
* venv = the **MAIN checkout** `.venv`.

## Run selection + adjudication

* 🚩 **Select on `exit_frame` / HAZARD / `n_passed` — NEVER on value.**
* 🚩 **Multi-seed is MANDATORY** (~2.8× seed variance).
* 🚩 **Monitor TensorBoard, not stdout.**
* 🚩 **DIAGNOSE FROM TRAINING METRICS**, not from anecdote.
* 🚩 **L16: absent hook prints on a LIVE job ≠ inert** — SLURM buffers. Adjudicate only from a
  COMPLETED log or TB.
* 🚩 **Telemetry is the DEFENDANT, not the witness** — when it disagrees with the video, ask human eyes.
* 🚩 **Fengyou counts gates 1-BASED.**

## Deploy discipline

* 🛑 **ARRESTOR / tape / speed-governor are RETIRED** → [[feedback-no-deploy-bandaids]].
* 🚩 **Fence ONLY the axis the course does NOT use.**
* 🚩 **Fly the TRAINING action space** → [[feedback-training-faithful-deploy]]. **SIM `YAW_EVAL`
  TRANSFERS 1:1 TO WIRE** when flips ≤4 ∧ absmean ≤0.3 ∧ saturation ≤10%.

## Settled campaign verdicts — do not re-litigate

* 🟢 The v1 ceiling was **SPEED ACCUMULATION** (closed).
* 🟢 **"flat plant = the wall" is OVERTURNED** — no more plant tuning.
* 🟢 The runaway was **DISCOUNT arithmetic**, not a controller defect.
* 🚩 The **billboard is a death-zone obstacle**, not scenery.
* 🚩 **RENDER-OVERLAY GOTCHA:** `render_vision_video.py` overlays **red_glow**, NOT the YOLO that flew.
* 🚩 A **0-byte agent transcript ≠ a dead agent** — check TREE mtime. Keep ≤2 heavy agents.

## Track A framing (goal still standing)

🎯 **BEAT vpeffs0 on the full 20-gate course.** 🚩 **PRESERVE the yaw → gate-in-view → altitude
coupling** — penalize yaw JITTER only → [[feedback-beat-dont-reproduce]] ·
[[feedback-preserve-yaw-altitude-hold]].

Reference constants: **camera FOV 90° H / 58.7° V (fx = fy = 320)** · **EGO obs 21-dim, critic 16**
(SSOT `docs/vq2-egocentric-gen/DESIGN.md`) · **VQ2 wire** → [[reference-sim-interface]]: HIGHRES_IMU
117 Hz accel+gyro only · 30 Hz JPEG · RACE_STATUS 4 Hz · dark warehouse.
