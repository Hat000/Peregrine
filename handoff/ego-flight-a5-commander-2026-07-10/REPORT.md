# EGO commander data run (a5, 2026-07-10) — clean fully-logged failed flight + loop-rate diagnosis

Branch `claude/ego-deploy-2026-07-09` @ `e9ce54f`. vn16 egocentric policy (`vn16_final_actor.pth`,
sha `8f1d679e…`) + **TRT engine** detector (`vq2_darkred_negreal42_2026-07-05_fp16_384x640.engine`),
takeoff-assist ON. Flown on the pilot side (sim side driven by the operator). Run is a **deliberate
data-collection flight — expected to crash**; the value is the fully-logged bundle + the loop-rate
root-cause.

## TL;DR

- **Clean flight, fully logged.** Fresh GO (not a stale late-join), video flowed (3120 frames,
  gaps>1s=0), 48 control ticks, IMU captured (12,199 HIGHRES_IMU @ ~185 Hz). Ended in an env
  collision, gates=0 — as expected.
- **Loop-rate root cause SOLVED** (the session's headline): the catastrophic 2.8–3.0 Hz chokes seen
  in earlier flights were **orphaned pilot processes** (stale `fly_rl` python procs from
  crashed/torn-down sessions, each holding a YOLO on the shared passthrough GPU), NOT the idle
  Claude sessions and NOT the detector choice. Killing 2 orphans took the loop 2.8 → 20.1 Hz with
  nothing else changed. Confirmed by operator Task-Manager read: **CPU 38 %, GPU 41 % during flight
  — nothing saturated** → the bottleneck is GPU-context *serialization latency* (sim render +
  detector sharing one passthrough RTX 2000 Ada), the documented ShadowPC failure mode, not resource
  exhaustion.
- **The TRT engine works** (I was wrong earlier to call it broken — that was this branch missing the
  `task=pose` loader fix, now ported + committed). Parity-verified vs the `.pt`: corner median
  0.01 px, exact detection-count/score match, 1.65–1.78× faster.
- **Policy flight quality is a genuine RL/OOD issue, not a loop artifact.** Same crash signature at
  every loop rate tested (2.8, 20.1, 17.2 Hz): lift → handover → punch to max thrust (3.76 g) →
  lose the gate → tumble into the environment in ~1.5 s.

## Bundle manifest

Full run: `data/runs/20260710_235926_ego_vn16_a5_commander_f1/` (on ShadowPC).

| file | size | contents |
|---|---|---|
| `mavlink.tlog` | 1.3 MB | **IMU**: 12,199 HIGHRES_IMU (accel+gyro ~185 Hz) · 9,900 ACTUATOR_OUTPUT_STATUS · 5 COLLISION · HEARTBEAT · video chunks |
| `ego_obs.jsonl` | 22 KB | per control-tick (30 Hz): 21-dim obs, `rate_frd` command, `collective`, `normed_thrust`, `assist` flag, `kf_pos_ned`, pose/conf/area |
| `ego_timing.jsonl` | 4 KB | per-tick phase wall-times (`work_ms` / `nav_ms` / `detect_ms`) — the loop-rate instrument |
| `meta.json` | — | run config + summary (final_state, collisions, durations) |
| `video.bin` + `video_index.jsonl` | 133 MB | onboard FPV frames (in-sim HUD included). **Stays local — >100 MB, cannot go in git.** |

Small logs (ego_obs / ego_timing / meta) are copied alongside this REPORT for portability and are
committed to the branch; `video.bin` + `tlog` remain in the run dir on ShadowPC (pull on request).

## Flight outcome + trajectory (from ego_obs.jsonl)

`[ego-assist] ACTIVE` → `HANDOVER at t=0.212s trigger=rates` (measured-rate handover, well inside the
1.5 s cap — the A2 takeoff-assist fix continues to work). Then:

```
k  t_rel pose conf area  thr   nt    rate_frd
0  0.00   T  1.00 0.96 0.292 1.10  [-1.3,-1.2,+3.1]   assist floor, lifting
3  0.35   F  0.72 0.93 1.000 3.76  [-3.0,+1.5,+3.1]   punched to max thrust
6  0.60   F  0.24 0.93 1.000 3.76  gate fading
9  0.79   F  0.00 0.93 0.955 railed rates, gate LOST
…  tumbling at near-max thrust, brief reacquires at k30 (conf .72) / k45 (conf 1.0)
→ HARD COLLISION (env, id 1002), gates=0, ~2.6 s after GO
```

Failure mode is identical to A2 and the a3 diagnostic run: the trained opener lifts, hands over,
immediately punches to the thrust ceiling and rails its body rates, loses the gate out of frame
within ~0.6 s, and tumbles into the warehouse structure. 79 % blackout duty (38/48 masked-slot0
ticks) reflects the gate being out of view for most of the tumble. **This is the RL-side flight-
quality question — it reproduces at 17–20 Hz, so it is not caused by the loop rate.**

## Loop-rate / timing (the instrument)

`[loop-rate] 17.2 Hz over 48 ticks; worst work 126 ms; 52.1 % over budget → CHOKED`

Per-phase medians (ego_timing.jsonl + nav sub-timers):

| phase | this run (engine) | a3 (.pt) | note |
|---|---|---|---|
| `nav.detect` (YOLO) | **20.7 ms** avg | 24.4 ms | **engine is ~15 % faster on-GPU, as expected** |
| `nav.vp_yaw` (VP-RANSAC) | 50.3 ms avg (n=9) | 38.2 ms (n=5) | **now the single most expensive call** |
| `nav.floor_height` | 24.7 ms avg (n=16) | 25.2 ms (n=8) | **running despite profile `use_floor_height=False`** |
| non-vision tick | ~2 ms | ~2 ms | estimator floor is trivial |

**Key finding for tuning:** the engine did its job (detect 24 → 21 ms), but the detector is no
longer the loop bottleneck. The loop is now gated by (1) `vp_yaw` VP-RANSAC at ~50 ms/call and (2)
`floor_height` at ~25 ms/call — and floor_height **should not be running at all** on this profile
(`use_floor_height=False` in `vq2_case_c`), so ~16×25 ms ≈ 400 ms/flight is pure waste. The loop-Hz
comparison a3(20.1) vs a5(17.2) is confounded (a5 flew ~2× longer → more of the decimated
vp_yaw/floor_height fired, plus run-to-run GPU-serialization variance), so read the per-phase costs,
not the headline Hz.

## Code changes committed this session (branch `claude/ego-deploy-2026-07-09`)

- `d8e23a4` — `detector.py`: port `_load_yolo_model` `task='pose'` for `.engine`/`.onnx` (without it,
  ultralytics guesses `task=detect` and silently drops all keypoints → 0 gates: the reason I wrongly
  called the engine broken). `fly_rl.py`: `_looks_like_detector_weights` now accepts `.engine`/`.onnx`
  (startup-guard weights-gating only, no obs/action-contract change). Plus an **additive,
  contract-neutral `[DIAG]` per-phase timing logger** in the ego loop (touches `rl/fly_rl.py`, RL-owned
  — flagged; changes no obs/command value).
- `e9ce54f` — vendored `scripts/vq2ctl.py` + the a3 diagnostic data.

## RL / nav-side action items (ranked)

1. **`floor_height` runs despite `use_floor_height=False`** — the profile flag isn't honored on the
   ego path (or floor_height fires for another reason). ~400 ms/flight of wasted GPU/CPU. Easiest win.
2. **`vp_yaw` (VP-RANSAC) ~50 ms/call is now the top per-call cost.** Profile sets `vp_yaw_async=True`
   (worker-thread), yet the timing still charges ~50 ms on the loop thread — verify the async
   decouple actually fires on the ego path, or the timer is catching submission/retrieval.
3. **Policy post-handover behavior** (punch-to-max + tumble + gate-loss) — the flight-quality blocker,
   reproducible independent of loop rate. Owner: RL.

## Operational lessons (now folded into pilot startup hygiene)

Before every launch: (1) kill orphaned `fly_rl` pilots — they accumulate silently across crashed
sessions and are the real GPU-contention source; (2) require a genuine `WAITING` (`started=False`) —
the post-race menu streams stale `started=True` residue and the pilot will late-join a dead race (cost
one run, a4: 0 frames); (3) after GO, confirm `frames>0` before trusting the flight.

## MEMORY-DELTA (for commander banking)

- Loop-choke ROOT CAUSE = orphaned `fly_rl` pilots (stale procs holding a YOLO on the shared GPU),
  NOT idle Claude sessions. Kill-before-launch is mandatory; 2 orphans = 2.8 Hz, 0 orphans = 20 Hz.
- Bottleneck is GPU-context SERIALIZATION on the passthrough RTX 2000 (sim render vs detector CUDA),
  NOT saturation — operator confirmed CPU 38 %/GPU 41 % during a choked flight. Fix = fewer CUDA
  contexts sharing the GPU + cheaper detector calls, not more compute.
- negreal42 TRT engine WORKS (parity: corner 0.01 px, 1.65–1.78× faster). Earlier "engine broken"
  was WRONG — the ego branch lacked main's `task=pose` `.engine` loader fix; now ported + committed
  (`d8e23a4`). Engine cut detect 24 → 21 ms; loop is now gated by vp_yaw (~50 ms) + floor_height
  (~25 ms, running despite profile=off).
- vn16 flight quality unchanged across loop rates (2.8/20/17 Hz): lift → handover (measured-rate,
  works) → punch to 3.76 g → lose gate → tumble into env ~1.5 s. Flight-quality is an RL/OOD issue,
  not loop-rate. Data: `data/runs/20260710_235926_ego_vn16_a5_commander_f1/` (tlog=185 Hz IMU,
  ego_obs, ego_timing, video.bin local-only).
- Pilot ops split (2026-07-10): pilot session owns launch + data + smooth-startup hygiene; operator
  owns the sim (menu/GO). See [[vq2ctl-tool]].
