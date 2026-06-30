# VQ2 self-localized SLOW-lap — ATTEMPT 8 (start-gate egress thrust floor + point-blank elevation guard) — 2026-06-30

**Branch flown:** `claude/loving-galileo-92f020` @ `dcc1d7f`. **Sim:** AI-GP 1.0.3379, **R2 - TRAINING =
VQ2**. Command exactly as specified: `--gate-seeker --deploy-profile vq2_case_c` (default `--rate 30`,
`--seeker-speed 3.0`). 2 clean VQ2 runs (run1 fresh GO, run2 in-sim RESTART late-join). New gate_seeker
tests pass (78/78).

> **VQ2 verification note (important):** `pos=NO` on the wire is **NOT** a VQ1/VQ2 discriminator — both
> tracks deny position on the 3379 wire. (A false belief from earlier attempts; corrected here after a
> first run accidentally loaded **R1 = VQ1** by pressing FLY on the *default* event.) VQ1/VQ2 must be
> confirmed **visually**: VQ1 = grey wireframe, VQ2 = lit warehouse with glowing **red** gates. Both A8
> runs verified VQ2 from the **onboard camera** (red-glow pixels 4586 / 9105; run1 frame shows the lit
> warehouse, red gate, cyan floor lines, ANDURIL wall). Menu: must deliberately select **R2-TRAINING**
> in ACTIVE EVENTS (orange border) before FLY — do not press Enter on the default (R1/VQ1).

## TL;DR scoreboard
| Question | Result |
|---|---|
| **Spawn crash (A7) fixed?** | **YES ✓✓ (n=2).** The egress thrust floor + point-blank elevation guard work — the drone **clears the start gate and climbs out** (brief thr=0.05 dip at spawn, but the floor catches it → altitude goes UP, not the A7 free-fall). |
| Gates passed? | **0** — but it now flies **~110 m down the course** before crashing (A7 died at the start gate in <1.5 m). |
| New failure mode | **Runaway forward speed + altitude instability → flies BELOW the gates → dives into the environment at ~110 m.** |
| Loop rate (live, real sample) | **~16 Hz over 94–95 ticks** — the FIRST steady-state live samples (flights survived 5–6 s, not A7's 15 cold-start ticks). Still CHOKED vs 30 Hz. |
| Frame drops | **0.0%** (both runs, 30 fps) — perf fix still holds. |

## ✅ The A7 spawn fix WORKS
Both runs clear the start gate that killed all 3 A7 runs. run1 NED trajectory (x=fwd, z=down):
```
t+0.0  pos=(  0.0, 0.0,  0.0) thr=0.266   <- spawn, inside start gate
t+1.0  pos=(  1.3, 0.0, -0.7) thr=0.159   <- EGRESS: climbs OUT (z=-0.7 up). A7 free-fell here (thr 0.05).
t+2.0  pos=(  9.3, 0.1, +0.3) thr=0.372
```
tlog: `min|a|=0.1 @t+0.3s` (a brief spawn thrust dip survives) then `|a|=22` (forward accel) — the egress
floor recovered the dip instead of letting it free-fall into the gate. **A7 recommendation #2 landed and
is verified.**

## ❌ New failure: forward-speed runaway + altitude dive (flies under the gates)
Past egress, the drone accelerates forward UNBOUNDED at full feedforward thrust (0.372) and loses altitude.
run1 (NED): the clean dive; run2 (RESTART): porpoising, same end.
```
run1:  t+2  x= 9.3  z=+0.3      run2:  t+2  x= 9.3  z=+1.5
       t+3  x=31.9  z=+4.0             t+3  x=31.3  z=+6.0
       t+4  x=66.8  z=+12.7            t+4  x=64.8  z=-7.3   (porpoise)
       t+5  x=111.3 z=+28.4  CRASH     t+5  x=109.4 z=+2.6   CRASH
```
- **Forward speed runs away to ~40–45 m/s** (Δx ≈ 44 m in the last second) — NOT the 3 m/s cruise. The
  forward feedforward is not velocity-limited in pursuit, so it just keeps tilting/accelerating.
- **Altitude is not held.** run1 tlog realized pitch marches nose-DOWN: `-2 -> -8 -> -15 -> -21 -> -34deg`
  — a deepening dive — until impact at ~110 m (`|a|=1101`, roll flips to 179deg inverted, peak gyro 68 rps).
- **gates=0 despite flying 110 m down the course:** the drone is sinking below / overshooting the gate
  openings at 40 m/s, so it never registers a pass (env contacts 1002 only, no gate 1001). It flies UNDER
  or PAST the gates and hits the floor/wall.

**Read:** the A7 fix peeled back the spawn layer and exposed the **pursuit speed/altitude controller** as
the next blocker. The drone can now leave the start gate and acquire the course visually (the onboard
frames clearly show the red gate ahead with the cyan lead-in lines), but it dives through the floor at
runaway speed instead of cruising at 3 m/s and holding the gate-opening height.

## Loop rate — first real LIVE steady-state read: ~16 Hz
With flights now surviving 5–6 s, the `[loop-rate]` self-report has a meaningful sample:
```
run1: 16.2 Hz over 95 ticks; worst work 92 ms; 100% over budget -> CHOKED
run2: 15.7 Hz over 94 ticks; worst work 88 ms; 100% over budget -> CHOKED
```
Context vs the campaign: **A6 ~2.3 Hz -> A8 ~16 Hz (~7x).** Frame drops 0% (perf fix holds). BUT it's not
the clean 30 Hz the offline vision-core timing (33 ms) predicted: the **full live tick is ~62 ms**
(16 Hz), i.e. the rest of `nav.update` (ESKF + PnP + KF + gate-bearing yaw) + the seeker + live sim load
on this VM adds ~30 ms on top of the 33 ms VP/detect core. The VP RANSAC is no longer the bottleneck — to
reach 30 Hz the *non-VP* path needs profiling. (Note: at 16 Hz and 40 m/s the drone travels ~2.5 m between
commands, which itself worsens the dive — the speed runaway and the sub-30 Hz loop compound.)

## Recommended next steps (flight-stack — commander's call; stack NOT touched)
1. **Cap forward speed to the cruise setpoint.** The pursuit forward feedforward runs away to ~44 m/s; it
   must saturate at `--seeker-speed` (3 m/s). This is the #1 cause of both the dive and the gate misses.
2. **Hold altitude / track the gate-opening height in pursuit.** Realized pitch marches to -34deg with no
   restoring vertical command — the drone needs to hold height (or follow `trk_el` to the gate centre)
   rather than trading all thrust for forward tilt. Together with (1) this should let it fly THROUGH a gate.
3. **Then re-check gate registration:** once it flies at gate height and 3 m/s, confirm a pass increments
   gi (it flew 110 m past the line of gates at 40 m/s with gates=0 — partly geometry, possibly also a
   pass-detection range/speed gate worth checking).
4. **(Lower priority) profile the non-VP nav tick** to claw back the last ~16->30 Hz; the VP fix already
   did its job (0% drops, 33 ms core).

## Artifacts (this dir)
- `crash_analysis.py` — tlog realized IMU/attitude reconstruction (the nose-down dive). `python crash_analysis.py <session>`.
- `navtime.py` — offline navigator timing (carried from A7; the 33 ms vision core).
- `timing_run1.txt` — `video_timing_report` (0% drops). `onboard_run1_frame303.png` — the VQ2 lit-warehouse proof frame.
- Recordings (gitignored, on ShadowPC): `data/runs/20260630_005945_*` (run1, fresh GO),
  `20260630_010404_*` (run2, RESTART late-join). Earlier `005808`/`005231` were the VQ1-mistake / stall lead-ups.
- Flight stack untouched (clean tree); only this handoff dir committed.
