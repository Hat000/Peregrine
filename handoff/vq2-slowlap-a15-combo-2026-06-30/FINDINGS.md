# VQ2 SLOW-lap — ATTEMPT 15 (combined: YOLO detector + case-C yaw-frame fix) — 2026-07-01

**One-line outcome:** **Recall SOLVED** (`valid_empty` 74%→**0%**) — but the **onboard camera FROZE** for the
YOLO first-inference warmup (~2.9 s) and dropped ~19% of frames, so **the drone's actual flight behavior was
UNOBSERVED.** The onboard camera is the ONLY view (pilot watches the same live stream — it froze; the
recording is full of the same freezes). **The #1 next fix is the frame-freeze itself** (pre-warm the detector
+ stop the drops); until the view is live neither the pilot nor the recording can judge whether the drone
flew. What IS verifiable looks good ("it was doing so well"): recall solved, and the log shows steering/
attitude no longer thrashing.

> **RETRACTION.** My first write-up said "the drone sat in the start gate / didn't move." That was NOT
> observable — the camera was frozen — and it leaned on the unreliable dead-reckoned estimate. **Do not treat
> A15's flight outcome as known.** (Two bad reads in a row: first trusting the estimator's fake "climb," then
> a frozen/mis-aligned clip. Corrected here.)

**Branch:** `claude/loving-galileo-92f020` @ `2ae5a71` (`--seeker-weights` split) + `337c554` (case-C TRUE-AHRS
yaw-frame fix). `--gate-seeker --deploy-profile vq2_case_c --seeker-detector yolo --seeker-weights
models/gate_clean_ens_course_L110.pt` (single model). VQ2 confirmed. gates=0, 67 nav ticks (~6 s loop).

## What is genuinely verifiable
- **DETECTOR / recall — SOLVED (real, detector-level).** `[seeker-diag] cmds=56 pursuit=56 none=0
  valid_empty=0`. Every tick got a pose (A13 red_glow was 96/130 = 74% empty). The deployment-gap fix works
  in the loop; pose starvation is gone.
- **Steering/attitude no longer thrashing (in the log).** No +22–28° nose-up whip and no yaw-away command
  pattern this run. Consistent with the yaw-frame + softening fixes working — but NOT visually confirmed
  (frozen view), so treat as promising-not-proven.

## THE #1 FIX (fold into the commander report) — the camera/loop frame-freeze
- `[loop-rate] worst work **2873 ms**` = the YOLO **first-inference warmup** (CUDA/JIT). ~2.9 s with the
  onboard camera **frozen** — right over the launch/egress window, so the pilot (and recording) see NOTHING
  during the most important moment. Video dropped **108/561 frames (~19%)**.
- The rest of the loop kept up (only 4.5% ticks over budget, 10.6 Hz avg dragged down by that one warmup
  tick) — so this is a **startup spike + frame drops**, not sustained choke.
- **Fixes:** (a) **pre-warm the detector** — run one dummy inference at load, BEFORE ARM, so tick-1 isn't a
  ~3 s stall; (b) chase the ~19% frame drops (GPU contention between YOLO inference and the video receiver /
  loop) so the live view stays smooth. Re-check achieved Hz + drop-rate after. **This unblocks OBSERVATION
  itself** — we can't evaluate the flight (or the yaw fix) until the camera is live through the launch.

## Still open (assess AFTER the freeze is fixed and we can watch)
- Does the drone lift off + translate forward toward the (correctly-detected, centred) gate, or stall near
  the start? The forward-pitch probe flagged a no-liftoff/no-sustained-forward-flight risk; unknown here
  because it wasn't observable.
- Dynamic confirmation of the yaw-frame fix (needs an off-axis gate + a live view).

## Artifacts
- `nav_estimate.jsonl` (incl. yaw_des_rad) — note: position is dead-reckoned (VQ2 pos=NO), NOT ground truth.
  `seeker_diag.txt`. Recording (gitignored, ShadowPC): `data/runs/20260701_033242_..._f1/onboard.mp4` — but
  it is freeze-riddled; not a reliable record of the flight. Flight stack untouched; handoff dir only.
