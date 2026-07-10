# EGO-A2 flight report (2026-07-10) — takeoff-assist fix verification

Branch `claude/ego-deploy-2026-07-09` @ `4343aba` ("feat(ego-deploy): autonomous takeoff assist
for the ego opener (A2 unblocker)"), pulled clean fast-forward from `3662227`. Only `rl/fly_rl.py`
(+154) and a new `tests/test_ego_takeoff_assist.py` (+288) changed — nothing in `ego_obs.py`, as
promised. Read `EgoTakeoffAssist` (fly_rl.py:637) and its call site (fly_rl.py:1811) before flying:
implementation matches the design description exactly — floor only ever raises the emitted
collective (`max(policy_normed, assist_g)`), rate commands pass from `policy_step` straight to
`ControlCommand` untouched, handover keys off *measured* `gyro_frd`/`nav_z` (not commanded rate),
permanent one-shot latch, provably no-op when disabled or after handover (same formula as
`policy_step`'s own). No code changes made.

## Pre-flight: machine load

Flagged before launch: 12 concurrent `claude` processes running (more than the 5 that choked
A1-flight-2), though CPU accumulation looked modest/fresh at the time, not obviously heavy. Could
not selectively close other sessions myself — proceeded and watched loop-rate live as the ground
truth instead of guessing. **In hindsight this was likely the dominant factor in the outcome
below** — see Verdict.

## Attempt 1 — crashed while waiting (sim not up)

First pilot start hit `ConnectionResetError: [WinError 10054]` inside `wait_fresh_go` -> `client.pump()`
after ~20s of `waiting: started=False pos=n` with 0 video frames received. `vq2ctl.py status`
confirmed `flightsim_procs: 0, dcgame_procs: 0` — the sim executable wasn't running (an Edge window
titled "AI Grand Prix 2026 — Team Portal" was up instead). Not a code issue — purely the pilot
attaching before the sim existed. Stopped per the task's escape hatch, waited for confirmation the
sim was actually up (`vq2ctl.py probe` -> `WAITING`), then restarted clean.

## Flight 1 — `ego_vn16_a2` (`data/runs/20260710_225745_ego_vn16_a2_f1`)

```
[ego] takeoff-assist ON: thrust floor 1.1 g (collective 0.292); rate commands pass through
  untouched; disarms PERMANENTLY on |gyro_body|>1.0 rad/s OR climb>0.5 m OR 1.5s.

  [ego-assist] ACTIVE (thrust floor 1.100 g)
  t=66.52s gi=0 conf=1.00 area=0.95 thr=0.292 rate=[-1.25,-1.21,+3.14]
  [ego-assist] HANDOVER at t=0.716s trigger=rates
  t=67.64s ... thr=1.000 rate=[-2.89,-0.22,+3.14]
  t=68.82s ... thr=0.995 rate=[-1.54,-1.13,+3.14]
  t=69.96s ... conf=0.00 thr=0.859 rate=[-1.73,+0.53,+3.13]
  [ego] HARD COLLISION -> abort.

  [ego-log] wrote 13 ticks -> ego_obs.jsonl
  [loop-rate]   2.8 Hz over 13 ticks (target 30); worst work 481 ms; 100.0% ticks over budget -> CHOKED
  [ego-diag] fresh gate levers=8  masked-slot0 ticks=5/13 (38% blackout duty)
  flight 1: CRASH  gates=0  session=data\runs\20260710_225745_ego_vn16_a2_f1
  [video-thread] frames=630 max_gap=144ms@fid=997 gaps>200ms=0 gaps>1s=0 gap_sum=23159ms max_pub=0.3ms
    reconnects=0(idle=0,exc=0) | wire: datagrams=656794 completed=630 evicted=0 dup=628322
```
No `[vision-timing]`/`[seeker-diag]` line — same as A1, not emitted by this build's `_fly_ego`.

**[ego-assist] lines — both present, exactly as the task asked to watch for:**
`ACTIVE (thrust floor 1.100 g)` immediately post-GO; `HANDOVER at t=0.716s trigger=rates` —
well inside the 1.5s cap, fired via the *intended* primary mechanism (measured body rate crossing
1.0 rad/s), not the timeout fallback.

### Trajectory (from ego_obs.jsonl, all 13 ticks)

- k=0-1 (assist active, t=0 - 0.38s): thrust floored at collective 0.292 (1.1g), rates railed
  similarly to A1 ([-1.2,-1.2,+3.14]), attitude essentially at spawn (roll~0, pitch~-0.31 — same
  tilted-pad value as A1, confirmed in-distribution per the RL report).
- k=2 (t=0.72s, first handed-over tick): roll_pitch obs jumps from [-0.0003,-0.31] to
  [0.12, 0.95] in one tick — the real rotation the assist was waiting to feel. `assist=false` from
  here on (confirmed per-tick in the obs record — the new field the RL side added).
  normed_thrust jumps 1.1 -> 3.66 g in this same tick.
- k=3-7 (t=0.72s - 2.69s): thrust sustained near-max (collective 0.97-1.00, normed_thrust
  3.6-3.76g, i.e. pegged at/near the hardcoded ceiling) — matches the RL side's offline-replay
  prediction almost exactly. BUT roll/pitch swing violently and don't settle: roll ranges
  -0.61 to +1.48 rad, pitch -0.44 to +1.01 rad, tick to tick, no visible convergence toward a
  stable attitude pointed at the gate.
- k=8-12 (t=3.11s - 4.30s): `pose_seen` flips to `false` and stays false (age_s climbs 0.42 ->
  1.61s, conf 0.15 -> 0.0) — vision lost the gate entirely, consistent with the drone tumbling/
  rotating too fast or too far off-axis for the gate to stay in frame. Thrust stays pegged near
  max (0.86-1.00) the whole time on stale-obs commands.
- `[ego] HARD COLLISION -> abort` fires at k=12/t=4.30s. `meta.json`: `collisions: 2`,
  `final_state: CRASH`, `gate_index: 0` (never advanced).

### Visual check

Extracted frames via `RecordingReader` (same technique as A1). Proportional-index sampling was
unreliable here (the recording is short and choke-distorted, frames aren't evenly spaced in time)
so this isn't a tick-by-tick visual trace like A1's — but the very last frame in the recording
(`frame_lastframe_impact.png`) is unambiguous: a close-up of an overhead grid/truss structure,
not the corridor-and-gate view from every earlier frame (`frame_preliftoff_85pct.png`, still
showing the ground-level gate view). Consistent with a nose-up/pitch-up excursion into the
warehouse ceiling grid — an environment collision, not a gate collision, matching
`meta.json`'s collision-type note (1001=gate, 1002=env; this run logged env).

## Verdict: the FIX WORKED. The CRASH is a separate, likely-contention-driven problem.

Two genuinely different questions, and this flight answers them differently:

1. **Does the takeoff-assist fix solve the A1 ground-freeze?** YES, cleanly. Liftoff happened,
   the handover fired via the correct measured-rate mechanism well inside its time budget, and the
   post-handover thrust punch matches the RL side's own offline prediction almost exactly. This is
   a clean, direct confirmation of both the root-cause diagnosis and the fix.
2. **Did the drone then fly well toward gate 1?** No — hard collision with the ceiling ~4.3s after
   GO, gates=0. But the control loop was running at 2.8 Hz against a 30 Hz target (100% of ticks
   over budget, worst tick 481 ms) — roughly 10x slower than intended, at exactly the moment
   (near-max thrust, aggressive rates, first seconds of real flight) where tight control matters
   most. A body-rate-controlled quad punching to ~3.7g of commanded thrust with ~350ms between
   control updates has a lot of room to diverge before the next correction lands. This is the same
   contention signature flagged before launch (12 concurrent Claude-session processes) — worse
   than A1-flight-2's 3.0 Hz, itself already identified as a contention artifact separate from the
   adapter. [video-thread] stayed healthy again (gaps>1s=0, evicted=0) — frame supply isn't the
   bottleneck, general host contention is.

**Net: this flight does not cleanly test "does the policy fly well post-liftoff," because the loop
never ran fast enough to give it a fair shot.** Recommend a clean repeat once the machine is
actually quiet before drawing conclusions about the policy's post-handover behavior — the
takeoff-assist mechanism itself doesn't need to be re-verified, it already worked as designed.

## Why no further repeats this round

Neither of the task's two branch conditions matched what happened: it did not "settle back" after
handover (thrust stayed pegged near-max the whole time, never dropped) — so the
`--ego-assist-max-s 3.0` retry doesn't apply — and it did not "freeze on the pad with assist
active" — so the "stop, don't iterate knobs" instruction doesn't strictly apply either, but the
same spirit (don't blind-tune parameters into a result that's actually explained by something
else) argues against inventing a new retry recipe. Also declined to auto-fly the "2 more repeats
if healthy" branch — CRASH + gates=0 doesn't read as healthy. Flagging this specific outcome
(worked-then-crashed-under-contention) for a decision rather than picking a branch that wasn't
written for it.

## Preserved artifacts

- `data/runs/20260710_225745_ego_vn16_a2_f1/` — `ego_obs.jsonl` (13 ticks, now carries a per-tick
  `assist` bool field), `mavlink.tlog`, `video.bin`, `video_index.jsonl`, `meta.json`.
- `handoff/ego-flight-a2-2026-07-10/frame_preliftoff_85pct.png`,
  `frame_lastframe_impact.png` — the two extracted verification frames.
- Raw stdout: `flight_ego_vn16_a2.log` (worktree root; contains both the crashed cold-start
  attempt and the successful reconnect+flight, in that order).
- All left in place under `C:\Users\Shadow\Peregrine-ego-flight`.

## MEMORY-DELTA

- EGO-A2 (2026-07-10, commit 4343aba EgoTakeoffAssist): FIX CONFIRMED WORKING — liftoff, handover
  via measured-rate trigger at t=0.716s (well inside 1.5s cap), thrust punch to ~3.7g matching
  RL's offline-replay prediction almost exactly. A1's ground-freeze root cause is resolved.
- Same flight then hit a HARD COLLISION with the ceiling/environment ~4.3s after GO (gates=0,
  collisions=2), coincident with loop-rate choked to 2.8 Hz (100% over budget, worst tick 481ms) —
  worse than A1-flight-2's 3.0Hz. 12 concurrent Claude-session processes were running pre-flight
  (flagged before launch, couldn't self-mitigate). [video-thread] stayed healthy again — this is
  general host contention, not a frame-supply or adapter problem.
- Roll/pitch obs swung violently post-handover (roll -0.61..+1.48 rad, pitch -0.44..+1.01 rad)
  with no visible convergence; vision lost the gate (`pose_seen=false`) from k=8 onward, consistent
  with the drone tumbling/rotating too far for the gate to stay in frame — plausibly a symptom of
  the 10x-too-slow control loop rather than a policy or obs-pipeline defect.
- Did NOT fly further repeats — outcome didn't match either pre-defined branch (didn't settle back;
  didn't freeze on pad). Recommend one clean repeat once the machine is actually quiet before
  drawing any conclusion about post-handover flight quality.
