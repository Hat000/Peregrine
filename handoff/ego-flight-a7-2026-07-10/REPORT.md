# EGO a7 (2026-07-10) — clean >20 Hz loop-rate run (TRT engine benefit confirmed in-flight)

Branch `claude/ego-deploy-2026-07-09`. vn16 policy + **TRT engine** detector, takeoff-assist ON.
This is the ">20 Hz" run: **23.9 Hz**, cleared the bar. Focused loop-rate writeup; the full
diagnosis + crash analysis is in the sibling `handoff/ego-flight-a5-commander-2026-07-10/REPORT.md`.

## Result

`[loop-rate] 23.9 Hz over 27 ticks; worst work 79 ms; 40.7 % over budget` — best loop rate of the
session. Clean fresh GO, video flowed (177 frames, gaps>1s=0), IMU captured (36,760 HIGHRES_IMU;
armed window ~1.1 s — filter the tlog to the armed segment). Crashed as expected (handover 0.121 s
trigger=rates, env collision, gates=0) — the policy tumble, an RL/OOD issue independent of loop rate.

## The engine benefit finally shows clean

`nav.detect` per-call cost across the session's real GO'd runs (all orphans-killed):

| run | detector | nav.detect avg | loop |
|---|---|---|---|
| a3 | `.pt` | 24.4 ms | 20.1 Hz |
| a5 | engine | 20.7 ms | 17.2 Hz (flew 2× longer → more vp_yaw/floor_height fired) |
| **a7** | **engine** | **10.5 ms** | **23.9 Hz** |

a7 caught a **low-GPU-contention window** — detect landed at 10.5 ms, right at the ~12.7 ms
uncontended parity number. This is the clean confirmation the engine buys ~1.7× on detect in-flight.
**Caveat (honest):** the engine's detect swings 10.5–20.7 ms run-to-run because of GPU-context
serialization with the sim's render (the session's core finding — the passthrough RTX 2000 serializes
sim-render vs detector CUDA). So >20 Hz is *achievable* with the engine but not *guaranteed* every
run; a5's 17.2 Hz was a higher-contention window, not a regression. Loop rate on this hardware is
inherently variable, bounded below by whatever the sim's render is doing that instant.

## Remaining path to 30 Hz (unchanged from a5 — RL/nav side)

With detect down to ~10 ms, the loop is gated by the other two per-call costs:
- `nav.vp_yaw` (VP-RANSAC): 34.5 ms avg (n=5). Profile sets `vp_yaw_async=True`; verify the async
  worker actually decouples it from the loop thread on the ego path.
- `nav.floor_height`: 22.4 ms avg (n=9) — **still running despite `use_floor_height=False` in the
  `vq2_case_c` profile.** ~200 ms/flight of pure waste; the profile flag isn't honored on the ego path.

Kill those two and the loop should sit at/near 30 Hz consistently.

## Bundle

`data/runs/20260711_001220_ego_vn16_a7_f1/` (video.bin 7.7 MB local-only). Self-contained copy in
this handoff dir: REPORT + ego_obs (27 ticks) + ego_timing + meta + mavlink.tlog (185 Hz IMU).

## MEMORY-DELTA

- Engine benefit confirmed in-flight: a7 detect 10.5 ms (vs .pt 24.4 ms) → 23.9 Hz, best of session.
  Detect swings 10.5–20.7 ms run-to-run = GPU-serialization variance (sim render vs detector CUDA on
  the shared passthrough GPU), so >20 Hz achievable-not-guaranteed. Loop now gated by vp_yaw (~34 ms)
  + floor_height (~22 ms, running despite profile=off) — the two RL/nav levers left for 30 Hz.
