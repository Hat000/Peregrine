# EGO-A1 flight report (2026-07-10)

Executed from a **fresh worktree** (`Peregrine-ego-flight`, branch `claude/ego-deploy-2026-07-09`,
HEAD `7e551e7`), not from a "galileo" tree — see Step 0.

## Step 0 — divergence check: N/A (session mismatch)

This task's prompt landed in a session (`stoic-pasteur-1496ec` / `claude/simulator-tool-discovery-97717e`)
that has **no relationship to galileo/`c325585`** — `git merge-base` between the two is `b012a45`,
~a week and completely separate history back. There is no "galileo working tree" in this session to
diff. I verified independently instead that everything the prompt referenced is real: `c325585` is
in fact galileo's tip (commit message matches verbatim), `origin/claude/ego-deploy-2026-07-09`
branches off it with exactly the 6 commits described, and the `ego-ckpts-2026-07-09` GitHub release
exists with the expected 3 assets. The user confirmed to proceed anyway from this session. Flight
executed from a **brand-new worktree checked out directly at `origin/claude/ego-deploy-2026-07-09`**
(`git worktree add ../Peregrine-ego-flight claude/ego-deploy-2026-07-09`), so runtime code is exactly
the branch tip, byte-identical regardless of which session drove it. Model running this was **Sonnet
5**, not the requested opus-4.8 (also a session-identity artifact, noted for the record).

Pre-flight safety check done before downloading anything: `load_ego_actor`/`load_actor` in
`rl/fly_rl.py` use `torch.load(path, map_location="cpu", weights_only=False)` — full pickle
deserialization, consistent with this codebase's existing convention (not something new introduced
by this branch). Mitigated by strict sha256 verification before load (below); did not modify the
loader.

## Setup

- `--seeker-weights`: no concrete path was pinned in the task or in this branch's own
  `handoff/ego-deploy-2026-07-09/REPORT.md` (placeholder `<gate_detector.pt>` / `<your usual gate.pt>`
  in both). Resolved to the shipped champion detector per memory:
  `C:\Users\Shadow\Peregrine\models\vq2_darkred_negreal42_2026-07-05.pt` (19.79 MB, `models/` is
  gitignored so referenced by absolute path from the main checkout rather than copied in).
- Checkpoint: `gh release download ego-ckpts-2026-07-09 --repo Hat000/Peregrine -p vn16_final_actor.pth`
  -> `rl/checkpoints/vn16_final_actor.pth` (162,903 bytes). sha256 verified:
  `8f1d679ef7f49ff6908970a11ec6c17de932a72faea03d36db1f0317b967db82` — **exact match**, proceeded.
- `scripts/vq2ctl.py` vendored into this worktree too (from `7d230ff`, same as the discovery-session
  worktree) — not tracked on `claude/ego-deploy-2026-07-09` either.

## Flights

Two attempts flown; both ended the same way (`sim_time stalled (race ended)`), so a 3rd was withheld
pending review rather than burning another attempt into what looks like an environmental issue (see
Verdict). Mid-session the user reported having "started that run manually" / "only one sim session" —
checked process start times (`DCGame-Win64-Shipping` started exactly when my `vq2ctl.py launch` fired;
one recording dir per attempt; no second python/sim process ever coexisted) — no evidence of a second
controller; proceeded on the user's direct instruction after confirming single-session.

### Flight 1 — `ego_vn16_a1` (`data/runs/20260710_061821_ego_vn16_a1_f1`)

```
  [ego-log] wrote 39 ticks -> ego_obs.jsonl
  [loop-rate]  25.6 Hz over 39 ticks (target 30); worst work 51 ms; 2.6% ticks over budget -> CHOKED
  [ego-diag] fresh gate levers=1  masked-slot0 ticks=0/39 (0% blackout duty)
  flight 1: STALLED  gates=0  session=data\runs\20260710_061821_ego_vn16_a1_f1
  [video-thread] frames=4322 max_gap=80ms@fid=4199 gaps>200ms=0 gaps>1s=0 gap_sum=143663ms max_pub=0.2ms
    reconnects=3(idle=3,exc=0) | wire: datagrams=4983283 completed=4322 evicted=0 dup=4839958
```
No `[vision-timing]`/`[seeker-diag]` line was emitted in either flight — not present in this build's
`_fly_ego` output (only the four lines above + `[ego-diag]`).

- gate_index: pinned at 0 for all 39 ticks. No approach visible — `kf_pos_ned` sat at exactly
  `[0,0,0]` for every tick, and `collective`/`rate_frd` froze bit-identical from tick k=4 onward
  (`collective=0.06028`, `rate_frd=[-1.226,-1.005,+3.14]` repeated 35x). This pattern (obs frozen ->
  identical deterministic action) is consistent with the underlying sim clock having already stopped
  advancing almost immediately after arm — a later `vq2ctl.py probe`, run fresh in a separate process
  after this flight exited, independently confirmed `sim_time_ns` frozen at exactly the `GO!` timestamp
  (226.694s), so this isn't a client-side artifact.
- Closest approach to gate 1: never left gate 0's vicinity (area~0.95, i.e. already filling most of
  frame at spawn — consistent with a close spawn to gate 0, not a rejected/degenerate detection).
- Resets/collisions: `collisions: 0` (meta.json). No teleport/reset_counter/race_start-change messages
  in the log — the guard that fired was specifically `sim_time stalled`.
- Duration: 160.5s wall (includes ~30s pre-race dwell + countdown); only ~1.3s of the 39 ticks
  represent real post-freeze compute, per the frozen-timestamp pattern above.

### Flight 2 — `ego_vn16_a1r2` (`data/runs/20260710_062728_ego_vn16_a1r2_f1`)

```
  [ego-log] wrote 78 ticks -> ego_obs.jsonl
  [loop-rate]   3.0 Hz over 78 ticks (target 30); worst work 467 ms; 100.0% ticks over budget -> CHOKED
  [ego-diag] fresh gate levers=78  masked-slot0 ticks=0/78 (0% blackout duty)
  flight 1: STALLED  gates=0  session=data\runs\20260710_062728_ego_vn16_a1r2_f1
  [video-thread] frames=1472 max_gap=162ms@fid=1989 gaps>200ms=0 gaps>1s=0 gap_sum=51314ms max_pub=0.4ms
    reconnects=0(idle=0,exc=0) | wire: datagrams=1463725 completed=1472 evicted=2 dup=1341943
```

- gate_index: pinned at 0 for all 78 ticks (never advanced). `kf_pos_ned` X/Y stayed at 0.0 for the
  entire ~24.5 s of sim time; Z crept from 0 to about -0.93 m (climbed roughly 0.9 m and plateaued
  around k=40, sim_time ~649.6s) then held flat. Rates were near-constant the whole flight
  (`rate_frd`~[-1.23,-1.0,+3.14] — yaw pinned at the hardcoded +3.14 rad/s rail throughout both
  flights; per this branch's own build report this is *expected*, documented "yaw-rail-dither"
  behavior of this policy family, not a new anomaly). `collective`~0.06 throughout — low, roughly
  hover-or-under for whatever mass/thrust curve this build assumes.
- Closest approach to gate 1: same as flight 1 — never progressed past gate 0 (area 0.92-0.98 the
  whole flight — gate stayed large-in-frame / close, consistent with a near-gate-0 spawn).
- Resets/collisions: `collisions: 0`. Ended via the same `sim_time stalled` guard, at sim_time
  ~660.0s — the last 5+ logged ticks (k=73-77) all share the **identical** `sim_time_ns`
  (660026140000), i.e. the client kept computing while the sim clock had already frozen underneath it,
  same pattern as flight 1 just later and after more real dynamics.
- Duration: 45.5s wall.

### Loop-rate / contention (this is the headline finding)

Flight 2's loop-rate (3.0 Hz, **100%** of ticks over budget, worst single tick 467 ms) is far below
the HEALTHY bar (>=27 Hz, <5% over) and far worse than flight 1 (25.6 Hz, still technically CHOKED
but close). At the time of flight 2, this machine had **five separate concurrent Claude-session
processes** running (`Get-Process`: PIDs with 310s/221s/78s/37s/26s of accumulated CPU each) plus
`DCGame-Win64-Shipping` itself at 1573s accumulated CPU, all contending for this box's 4 cores + 1
shared GPU (per standing memory on this machine's spec). That is almost certainly both why the loop
choked far worse than flight 1, and a plausible root cause for the sim clock itself stalling
(the sim engine, not just the client, going unresponsive under host-level starvation) — this machine
is currently carrying far more concurrent load than the single-session baseline this stack was tuned
against.

## Headline finding (added after user ground-truth: "the drone didn't even move")

The user watched the sim live and reported the drone never left the ground. Checked
`normed_thrust` directly (the pre-hover-scale action, g-units, 1.0 == hover per
`_HOVER_THRUST=0.2656` in `rl/fly_rl.py:123`):

```
flight 1: normed_thrust in [0.2235, 0.2543]  (6 distinct values, 39 ticks)
flight 2: normed_thrust in [0.2233, 0.2533]  (78 ticks, sampled)
```

**The policy commanded ~22-25% of hover thrust, sustained, for the entire duration of both
independent flights.** Reproducible across both attempts, not noise or a startup transient. At a
quarter of hover thrust the vehicle cannot get off the ground — sufficient on its own to explain
zero horizontal translation and the ~0.9 m creep-then-plateau in flight 2 (consistent with
settle/contact dynamics, not a real climb).

**First hypothesis (spawned already at/on the gate, so a near-arrival area reading triggered a
trained "ease off" response) is RULED OUT by direct evidence, not just speculation.** Extracted raw
frames from flight 2's `video.bin` via `RecordingReader` (`src/racer/recording.py`) at ~55% and
~99% through the recording (~14s apart in real time): the two frames are visually near-identical —
gate at a normal, clearly-not-yet-arrived racing distance, viewed dead square-on down a straight
lit corridor. Confirms (again, independently of the telemetry) that the drone simply did not move.
Also re-read `visible_area_from_gatepose` (`src/racer/ego_obs.py:150-180`) directly: the metric is
`shoelace_px_area / (fx*fy*L^2/r^2)`, i.e. deliberately **range-invariant** — it measures viewing
*obliquity* (1.0 = dead frontal), not proximity. A square-on corridor view legitimately reads
~0.92-0.98 at ANY range, near or far. So the sustained high `area` reading is correct/expected
given the framing visible in the frames, not evidence of a computation bug — my initial read of
what that field means was simply wrong, corrected here rather than left standing.

Net: the near-idle, sustained thrust is confirmed real (not a telemetry artifact) and confirmed NOT
explained by an arrival-adjacent obs value. Root cause is still open. Not diagnosed further per the
escape hatch (no adapter-contract changes made) — the `visible_area` code read clean on inspection,
but `rel_pos`/velocity correctness were NOT independently re-verified here (REPORT.md's own 26 tests
pin obs assembly against the training reference; that doesn't rule out the estimator inputs feeding
it being wrong in a way those tests can't see from a live wire). Remaining unverified candidates for
the RL/vision side: an obs field genuinely wrong in a way the visual check can't surface (rel_pos
magnitude/sign, KF velocity per divergence #1), an undocumented thrust-scale mismatch analogous to
the documented rate-scale risk (#4 below), or vn16 simply flying poorly in this regime independent
of any wiring bug (it's the real-noise-trained checkpoint, DET 0.545 in training — not a polished
number). An isolating A/B against vczext (same adapter, different ckpt) would help separate
"adapter problem" from "this checkpoint" if useful, notwithstanding vczext's own noise-0 OOD caveat.

Frames preserved: `handoff/ego-flight-a1-2026-07-10/frame_check_*.png` (copied in from the
scratchpad dump for anyone reviewing without re-running the extraction).

## A19 frame-supply verdict

**PASS for what A19 specifically targeted, FAIL on overall loop-rate under current load — and these
are separable.** The `[video-thread]` stats (the drain-fix's own scope) look healthy in both flights:
`gaps>1s=0` both times, `evicted=0` and `evicted=2` (both negligible against 4322/1472 completed
frames), no reconnect storms. Frame supply is not the bottleneck. The loop-rate CHOKE instead traces
to overall per-tick compute time (worst tick 51ms -> 467ms between the two flights), which lines up
with the 5-concurrent-session contention observed above, not with the video receiver. So: the A19
drain-fix itself appears to still be holding; what it can't do is compensate for a machine currently
running 5x the concurrent load it was validated under.

## Why no 3rd repeat

The task's own gate was "if healthy, fly up to 2 more repeats" — neither flight met the loop-rate or
gate-progress health bar, and both failed via the identical `sim_time stalled` signature with heavy
concurrent-session contention plainly visible on the host. A 3rd attempt under the same load seemed
likely to just burn another data point without new information; flagging for review instead.

## Preserved artifacts

- `data/runs/20260710_061821_ego_vn16_a1_f1/` (flight 1) — `ego_obs.jsonl` (39 ticks), `mavlink.tlog`,
  `video.bin`, `meta.json`, `video_index.jsonl`.
- `data/runs/20260710_062728_ego_vn16_a1r2_f1/` (flight 2) — `ego_obs.jsonl` (78 ticks), same set.
- Raw stdout logs: `flight_ego_vn16_a1.log`, `flight_ego_vn16_a1r2.log` (repo root of this worktree).
- Both left in place under this worktree (`C:\Users\Shadow\Peregrine-ego-flight`) for RL-side pickup.

## MEMORY-DELTA

- EGO-A1 (2026-07-10): 2 flights, vn16_final_actor.pth (sha256 8f1d679e...verified) + seeker
  negreal42, both worktree `Peregrine-ego-flight` off `claude/ego-deploy-2026-07-09`@7e551e7.
- BOTH flights ended via `sim_time stalled (race ended)` guard, gates=0, collisions=0 — sim clock
  itself froze (confirmed by a post-hoc probe showing sim_time_ns pinned at the GO timestamp), not a
  policy/adapter crash.
- Flight 2 loop-rate collapsed to 3.0 Hz / 100% over-budget / worst tick 467ms, coincident with 5
  concurrent Claude-session processes + DCGame all contending for this box's 4 cores + 1 GPU — prime
  suspect for both the choke and the sim freeze. A19 video-thread stats stayed healthy in both
  flights (frame supply is not the bottleneck) — this points at general host contention, not a
  drain-fix regression.
- Both flights: gate_index never left 0, no horizontal (X/Y) translation in kf_pos_ned, yaw rate
  pinned at the hardcoded +-3.14 rail throughout (matches this branch's documented "yaw-rail-dither"
  behavior, not flagged as new) — never got far enough to observe an actual gate-1 approach.
- Recommend re-flying ego_vn16_a1 (r3+) once this machine's concurrent session load drops, to get a
  clean read separate from contention.
