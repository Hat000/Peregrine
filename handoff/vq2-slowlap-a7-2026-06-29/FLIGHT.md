# VQ2 self-localized SLOW-lap — ATTEMPT 7 (navigator perf fix + AHRS free-fall guard) — 2026-06-29

**Branch flown:** `claude/loving-galileo-92f020` @ `6733f50`. **Sim:** AI-GP 1.0.3379, **R2 - TRAINING =
VQ2** — lit warehouse + position-denied wire (`pos=NO`) both confirmed before flight. **Controller:**
map-free visual-servo gate-seeker, `vq2_case_c`, `red_glow`, cmd_rate_scale=0.4. Command exactly as
specified: `--gate-seeker --deploy-profile vq2_case_c` (default `--rate 30`, `--seeker-speed 3.0`,
`--flights 1`). 3 runs flown via in-sim RESTART: run1/run2 @3.0, run3 @2.0.

## TL;DR scoreboard
| Question | Result |
|---|---|
| **Did the loop speed up? (the A7 ask)** | **YES — decisively, at the pipeline + vision-core level.** Camera frame drops **55% → 0.0%**; navigator vision core **~448 ms → ~33 ms/frame** (~13×). See below. |
| 30 Hz control loop confirmed *live*? | **NOT YET** — every run crashed at spawn in <1.5 s (15–18 ticks), so the live self-report is **cold-start-dominated** (~11–13 Hz). Steady-state could not be sampled live. Offline the vision core sits **at** the 33 ms/30 Hz budget. |
| Gates passed? | **0** — all 3 runs **HARD COLLISION at gate 0 within ~1 s of GO**. Worse than A6 (which reached the pass). |
| Failure mode | **Close-range THRUST COLLAPSE → free-fall → start-gate contact**, fired **immediately at spawn**. The A6 bug, unmasked. |
| AHRS free-fall guard (ec10dc5) | Landed; did **not** save the run — it guards the *estimate* during free-fall, but nothing stops the free-fall itself (thrust collapse was NOT patched). |

## ✅ The perf fix works — this is the headline result
The two things the A6 report flagged as the bottleneck are **fixed and verified twice (run1 + run2):**

**1. Camera frame pipeline — drops eliminated.** `scripts/video_timing_report.py --latest`:
```
run1: frames kept 711/711  DROPPED 0 (0.0%)   eff 30.1 fps   surges 0   gaps 0   over 23.6s
run2: frames kept 673/673  DROPPED 0 (0.0%)   eff 30.0 fps   surges 0   gaps 0   over 22.4s
```
A6 dropped ~55% of frames with multi-frame holes. **A7: 0.0%, zero surges, zero gaps.** The "video
lag/surges" the user observed are gone.

**2. Navigator vanishing-point fit — ~13× faster.** Measured **offline on the run1 recorded frames**
(143 sampled, 5-frame warmup skipped — so this is clean steady-state, free of the live cold-start):
```
estimate_heading iters=2000 (A6-era):  mean 105.1 ms   p95 111.7   (already 2.5–3× faster than A6's
                                                                     unvectorized 230–325 ms via de9d226)
estimate_heading iters= 256 (A7 flight): mean  31.5 ms   p95  33.9   <- f1587b3 (2000→256 iters)
red_glow detect:                          mean   1.8 ms
detect + heading@256 (dominant vision):   mean  33.3 ms   p95  35.7
```
Net: the per-frame vision core went from **~448 ms (A6, ~2.2 Hz) to ~33 ms (~30 Hz)** — the vectorized
RANSAC (de9d226, bit-identical) × the 2000→256 iters cut (f1587b3). The vision core now lands **on the
33.3 ms / 30 Hz budget.**

## ⚠ Why the *live* loop still self-reports CHOKED (and why that's not the real story)
The in-loop self-report (new in 6733f50) printed:
```
run1: [loop-rate] 10.9 Hz over 15 ticks; worst work 165 ms; 100% over budget -> CHOKED
run2: [loop-rate] 12.7 Hz over 18 ticks; worst work  89 ms; 100% over budget -> CHOKED
run3: [loop-rate] 13.1 Hz over 17 ticks; worst work  88 ms; 100% over budget -> CHOKED
```
This is **cold-start-dominated and unrepresentative**: each run is a *fresh* `fly_rl` process that crashes
in <1.5 s, so the sample is only 15–18 ticks and the **first nav.update pays one-time cv2/numpy/JIT
warmup** (the 88–165 ms worst tick). The flight dies before the loop reaches steady state. Honest read:
- **The vision core is fixed** (33 ms offline, 0% drops live) — this is solid.
- **A clean 30 Hz LIVE confirmation is still pending** a flight that survives past cold-start. "100% over
  budget" on warm ticks 6–17 hints the *full* live tick (nav.update + seeker + send, under live sim load
  on this VM) is somewhat above 33 ms — likely **~15–25 Hz warm**, i.e. **~6–10× better than A6's 2.3 Hz**
  but not yet a clean 30 Hz. Can't be pinned down until the spawn crash is fixed.

## ❌ The blocker: close-range thrust collapse now fires AT SPAWN
All 3 runs: `gate-seeker HARD COLLISION -> abort`, **gates=0**, within ~1 s of GO. tlog HIGHRES_IMU
(run1, flight = last ~1.4 s of the recording; VQ2 denies LOCAL_POSITION_NED so 0 NED rows, as expected):
```
pre-arm: rests at pitch +18deg, |a|=9.8 (stationary, wedged in the start gate)
t=+0.3s: arm, attitude levels to ~(0,0)
t=+0.6s: |a| -> 0.1  m/s^2   <- FREE FALL (thrust collapsed to the 0.05 floor; matches live thr=0.050)
t=+1.2s: peak gyro [+10.0,+17.7,-1.4] rps  <- impact/tumble into the start-gate structure
         -> 18–200 env contacts (1002), no gate contacts (1001) -> abort
```
**Chain:** VQ2 spawns *inside* the start gate → the gate fills the frame at point-blank range → its
`trk_el` goes negative immediately → the altitude controller cuts thrust to the **0.05 floor** → free-fall
→ the drone drops/tumbles into the start-gate frame before it can egress.

**Why it's a regression from A6 (which egressed and reached the pass):** A6's loop ran at ~2.3 Hz, so in
the first 1–2 ticks (~600–900 ms) the drone applied egress thrust and **drifted forward out of the start
gate before the seeker "saw" the close gate and cut thrust.** A7's fast loop sees the point-blank gate on
**tick 1** and cuts thrust immediately → it never clears the gate. **The perf fix unmasked the latent
thrust-collapse bug by letting it fire sooner.** Confirmed speed-independent (n=3, incl. 2.0 m/s — same
spawn crash), so it is the alt-controller logic, not cruise dynamics.

## Recommended next steps (flight-stack — commander's call; stack was NOT touched)
1. **Fix the close-range thrust collapse (A6 rec #2, still unimplemented — now the #1 blocker).** Don't
   let the altitude controller cut thrust to the floor when a gate's `trk_el` goes negative at short
   range / when z is degenerate. **Critically, hold hover thrust through the START-GATE egress** — at GO
   the drone is inside the start gate, so the very first ticks must own thrust (the dead-reckon/egress
   regime should hold ≥hover, not defer to the alt controller). This alone should let it clear spawn.
2. **Then re-measure the live loop** — with a flight that survives cold-start, the `[loop-rate]` self-
   report will read true steady-state (expect a large jump from the ~12 Hz cold sample). If warm ticks
   still exceed 33 ms, profile the *rest* of nav.update (ESKF/PnP/KF) — the VP fit is no longer the cost.
3. Keep the perf fix as-is — it's verified and bit-identical; no further VP tuning needed for now.

## Artifacts (this dir)
- `navtime.py` — offline steady-state navigator timing (the 105 ms→31.5 ms table). Run:
  `python navtime.py <session_dir>`.
- `crash_analysis.py` — tlog HIGHRES_IMU reconstruction of the spawn crash (free-fall → tumble). Run:
  `python crash_analysis.py <session_dir>`.
- `timing_run1.txt`, `timing_run2.txt` — `video_timing_report --latest` output (0% drops, both runs).
- Recordings (gitignored, on ShadowPC): `data/runs/20260629_234501_*` (run1 @3.0),
  `_235500_*` (run2 @3.0), `_235630_*_s2_*` (run3 @2.0).
- Flight stack untouched (clean working tree); only this handoff dir is committed.
