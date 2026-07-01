# VQ2 SLOW-lap — ATTEMPT 15 (combined: YOLO detector + case-C yaw-frame fix) — 2026-07-01

**One-line outcome:** **Recall SOLVED** (`valid_empty` 74%→**0%**) and the yaw/pitch thrashing is **gone**
(the fixes calmed it) — but per pilot ground-truth **the drone never flew**: it sat in the start gate
(collision id 1001 = gate) and didn't move. Estimator "motion" was dead-reckoned fiction. Two separable
next blockers: (a) the **YOLO first-inference warmup (~2.9 s) freezing the launch window** + frame drops, and
(b) the persistent **no-liftoff / no forward flight** (now that control is calm, it doesn't thrash out of the
start gate either).

**Branch:** `claude/loving-galileo-92f020` @ `2ae5a71` (`--seeker-weights` split) + `337c554` (case-C TRUE-AHRS
yaw-frame fix). `--gate-seeker --deploy-profile vq2_case_c --seeker-detector yolo --seeker-weights
models/gate_clean_ens_course_L110.pt` (single model). VQ2 confirmed. CRASH gates=0, 67 ticks.

## What genuinely improved
- **DETECTOR / recall — SOLVED.** `[seeker-diag] cmds=56 pursuit=56 none=0 valid_empty=0`. Every tick got a
  pose (A13 red_glow was 96/130 = 74% empty). The deployment-gap fix works in the loop: pose starvation is
  gone. Gate detected all flight at ~11 m, dead-centre (`gate_cx ≈ 0.50`).
- **Yaw/pitch chaos — GONE.** No nose-up whip to +22–28° and no yaw-away this run; egress-end pitched only to
  +3° then settled gentle nose-down. The yaw-frame + softening fixes removed the A11–A14 thrashing.

## The correction (estimator-is-defendant, applied)
The `nav_estimate` position (x→+0.7 m, z→−1.7 m "climb") is DEAD-RECKONED (VQ2 pos=NO) and is **fiction** —
pilot (Fengyou): *"the drone didn't move at all ... toward the end we got frame drops again, but still no
drone movement."* Collision id **1001 = gate** = the START gate the drone is spawned inside. So: it sat in
the start gate, never lifted off / translated, and the persistent gate contact tripped the HARD-COLLISION
abort. (I initially mis-read the estimator's climb as real forward/vertical motion — corrected here.)

## Two separable next blockers
1. **YOLO warmup freezes the launch window.** `[loop-rate] worst work 2873 ms` = the first YOLO inference
   (CUDA/JIT warmup) — ~2.9 s with the drone frozen in the start gate right when egress should fire. Video
   dropped 108/561 frames (~19%) under GPU load. The rest of the loop kept up (only 4.5% ticks over budget,
   10.6 Hz avg dragged down by that one warmup tick). **Fix: pre-warm the detector (run one dummy inference
   BEFORE arming)** so tick-1 isn't a 3 s stall; re-check achieved Hz + frame drops after.
2. **No-liftoff / no forward flight.** With control now calm (no thrashing), the drone doesn't build the
   forward flight to leave the start gate either — same root the forward-pitch probe flagged (the seeker
   HOLD/egress doesn't produce sustained liftoff+translation). Now that recall + steering are fixed and the
   chaos is gone, THIS is the clean next target: get the drone to actually lift off and translate forward
   toward the (correctly-detected, centred) gate.

## Not yet testable
The yaw-frame fix couldn't be dynamically confirmed: the drone didn't move/turn, so "gate stayed centred" just
means a stationary drone was spawned facing it — not an off-axis yaw test. Re-test once it flies.

## Artifacts
- `nav_estimate.jsonl` (incl. yaw_des_rad), `seeker_diag.txt`. Recording (gitignored, ShadowPC):
  `data/runs/20260701_033242_vq2_slow_seeker_a15_combo_f1/onboard.mp4` (2×). Flight stack untouched; handoff only.
