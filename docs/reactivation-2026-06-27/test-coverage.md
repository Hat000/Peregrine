# Test Coverage — every ALIVE fork → the test that resolves it (2026-06-27)

*Audit (Fengyou's catch): 3 named spikes can't resolve ~25 alive forks. They don't have to — most forks collapse onto a few SHARED HARNESSES (one rig A/Bs a whole cluster), and many forks STACK rather than compete (no binary decision). Taxonomy: 3 strategic disposable spikes + 5 shared harnesses + the load-check battery.*

## Two reframes that dissolve the apparent mismatch
1. **Estimator forks STACK, they aren't either/or.** Strong-tracking + cubature + square-root + robocentric can all coexist → one bench, turn each lever on, keep what measurably helps. Not 9 decisions — incremental measure-and-keep.
2. **Harnesses are permanent build infra, not throwaway spikes.** The 3 named spikes (A/B/C) are disposable (they test whether a RISKY new direction is worth committing). The harnesses (estimator bench, RL ablation loop) we build anyway to measure the stack.

## Coverage table
| Test | Kind | Resolves (alive forks) |
|---|---|---|
| Spike A — gate-4 | strategic, disposable | SITT co-training · horizontal-FoV idea · D4 W-reward-shaper · calibration-vs-reach |
| Spike B — hybrid control | strategic, disposable | D3a AC-MPC · D3b RL+MPPI |
| Spike C — recon-map | strategic, disposable | A2 moat (dense/sparse/dead) · A3 learned VO |
| **T1 — AHRS bench** (NEW — previously hidden behind C2/C8) | shared harness | classical ESKF · invariant/polar EKF · vision-aided AHRS — all vs learned RIANN/Brossard, on sim-GT IMU at high-g |
| T2 — detector A/B | shared harness | Swift-YOLO small-target levers · 4→8–12 keypoints (distant-gate mAP, corner precision) |
| **T3 — estimator bench** (the big one) | shared harness | strong-tracking fading · cubature · square-root/UD · invariant+bias · metric-depth gap-fill · per-corner covariance→R · bearing-angle range channel — and validates the LEADING ones (ADR-VINS, FEJ2/robocentric, Cioffi LIO, blind-zone coast) in the same rig. Replay sim trajectories + injected noise; measure drift/consistency(NEES/NIS)/compute |
| T4 — RL ablation suite | build-loop A/B | bootstrapping (teacher→student) · GRU actor · layout-DR generalist · horizontal-FoV reward · D4 W-shaper (+ confirm the 4 certain wins) — each toggled in the normal training loop, scored on reach/pass/contact |
| T5 — generalization test | build-loop A/B | E1 layout-DR generalist · E2 recon-lap adaptation — train on layout-DR, eval on held-out layouts |
| Load checks C0–C10 (+ 2-load hash) | applicability gates | do NOT resolve forks — tell us WHICH are relevant (mag→AHRS, track-stability→moat, gap-frequency→gap-filler, legality→offline-calib) |

## Why it felt lopsided
- **T3 alone resolves ~9 estimator/pose forks** because they're a stackable menu tested by one rig.
- **Load checks gate applicability, not selection** — they prune which forks to bother testing.
- The 3 disposable spikes are only the BIG, expensive, direction-setting bets; the routine cluster-resolvers (T1–T5) were implicit in "A/B vs RewindKF" and the RL build loop.

## The one genuinely new test this audit surfaced
**T1 — AHRS bench.** Picking learned-vs-classical-vs-invariant attitude was sitting behind load-checks C2/C8, but those only reveal the *sensor conditions* (is mag usable? does the IMU saturate?). Choosing the *filter* needs its own offline bench on sim-GT IMU sequences at high-g. Add it to the build plan.

## Build-order implication
Cheapest-first still holds: load checks + the 2-load hash (applicability) → the 4 certain RL wins (T4, no gating needed) → the estimator bench T3 (resolves the biggest cluster) → the 3 strategic spikes (commit/park the risky directions) → T1/T2/T5 alongside.
