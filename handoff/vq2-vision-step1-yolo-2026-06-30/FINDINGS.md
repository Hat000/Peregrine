# VQ2 Vision Step-1 — YOLO detector recall vs red_glow — 2026-06-30

**One-line outcome:** The live YOLO-in-the-loop run is **BLOCKED by an arg-wiring bug** (`--checkpoint` is
overloaded), so I answered the core question **offline on the same real live-sim frames**: the trained YOLO
**collapses `valid_empty` from 74% → 27%** (ensemble) / 35% (single), and **strictly dominates** red_glow
(`yolo_only=74, rg_only=0`). **Verdict: DEPLOYMENT GAP — a free recall win once wired in**, not an appearance/
retraining gap.

## BLOCKER — the live run can't be invoked as specced (needs a commander fix)
`fly_rl.main()` hard-loads `--checkpoint` as the **RL actor** (line 1762) before the gate-seeker path reuses
the SAME `args.checkpoint` for `GateDetector.load` (line 969). There is no separate detector-weights flag and
`load_actor` has no `++` handling, so `--checkpoint "modelA.pt++modelB.pt"` dies with `FileNotFoundError` in
the actor load. The `--seeker-detector yolo` path was "never deployed," and this is why.
**Fix (commander): split the arg** — add `--seeker-weights` (or skip the actor load entirely under
`--gate-seeker`, since the seeker is the controller and the RL actor is unused there). Then the live fly +
loop-rate check can run.

## The experiment (offline, faithful): red_glow vs YOLO on identical A13 frames
`valid_empty` = fraction of frames where `detector.detect → estimate_gate_pose` yields NO finite-range pose —
exactly what `[seeker-diag]` counts. Run on the A13 red_glow baseline recording (312 frames, 156 sampled).
**Cross-validation: offline red_glow `valid_empty` = 74%, identical to the live A13 seeker-diag (96/130 = 74%)
— so this offline proxy is faithful.**

| detector | valid-pose | **valid_empty** | latency (median / p90) |
|---|---|---|---|
| red_glow (A13/A14 baseline) | 26% | **74%** | classical, fast |
| YOLO single (course L110) | 65% | **35%** | 21 / 26 ms |
| YOLO ensemble (course+precision) | 73% | **27%** | 39 / 46 ms |

Overlap (ensemble vs red_glow): **both=40, yolo_only=74, rg_only=0, neither=42.**
- **YOLO strictly dominates:** every frame red_glow poses, YOLO also poses (`rg_only=0`), plus 74 more.
- `neither=42` (27%) = the truly hard frames (gate out-of-frame / very far / oblique / motion-blur) — the
  residual real-vision gap, but small next to the deployment win.

## Read (matches the commander's decision tree)
`valid_empty` **COLLAPSES** (74% → 27%/35%) → **DEPLOYMENT gap, free win**: the trained ensemble already fixes
recall; it was just never in the loop. NOT an appearance/off-axis gap requiring a data-engine (that would be
`valid_empty` staying high). The remaining 27% is a smaller, separate follow-on.

## Latency / loop-rate note
The loop was already choked at ~12 Hz with the (fast) red_glow. Adding the **ensemble (39 ms)** would choke it
harder; the **single model (21 ms)** captures most of the recall win (74→35 vs 74→27) at ~half the cost and is
the better first live deployment given the choke. Confirm achieved Hz on the live fly once the arg is split;
re-check whether the recall win offsets the added latency in closed loop.

## Recommendation
1. **Commander: split `--checkpoint`** so the gate-seeker can take detector weights independently of the actor.
2. **Deploy the single-model YOLO first** (latency), then live-fly to confirm `valid_empty` drops in the loop
   and whether pursuit finally sustains (this directly attacks the A13/A14 pose-starvation root).
3. The yaw-frame fix (A14) and this detector win are complementary: recall gives poses; the yaw fix makes the
   seeker steer toward them. Both likely needed before a gate is threaded.

## Artifacts
- `detector_recall_compare.py` — reproduces the table (`python detector_recall_compare.py [<run_dir>]`).
- No new recording (live run blocked); analysis is on the A13 recording (`data/runs/20260630_225052_...`, on ShadowPC).
- Flight stack untouched; handoff dir only.
