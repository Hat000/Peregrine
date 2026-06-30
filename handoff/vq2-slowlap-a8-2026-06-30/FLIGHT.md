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
| Gates passed? | **0** — it approaches gate 0 briefly, then pitches up and flies AWAY (see corrected failure below). |
| New failure mode | **Attitude/self-localization estimate INVERTS → the drone pitches NOSE-UP, throttle up, and flies UP + BACKWARD into the back wall/ceiling.** (NOT a forward dive — that was my error trusting the diverged estimator; corrected from the user's direct view of the video + the onboard camera.) |
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

## ❌ New failure: attitude/estimate INVERSION → drone pitches UP and flies UP + BACKWARD
**This section is corrected.** My first write-up called it a "nose-down forward dive into the floor at
110 m." That was WRONG — it trusted the onboard NED position, which on a position-denied VQ2 wire is the
**self-localization estimator's belief, not ground truth.** The user watched the video (and the onboard
camera confirms it): the drone **pitches NOSE-UP, runs the throttle up, and flies UP + BACKWARD** — run1
goes up-and-back, run2 (a touch of spawn sag first) pitches up HARD, holds the angle, and goes mostly
BACKWARD (less up than run1). It crashes into the back wall / ceiling, not the floor.

**Onboard-camera proof (run1, the only trustworthy onboard signal):** the red gate is acquired and grows
as the drone approaches (frames ~300→460, gate large & close), then its glow slides DOWN the frame
(y-centroid 174→246) and **vanishes** as the camera tilts up — by frame ~500 the camera is staring
straight at the **warehouse ceiling grid** (gate gone, red-pixels=0). Camera pitched up = nose up. ✓

**The estimator reported the INVERSE of reality:**
```
                 estimator said        reality (user + camera)
  forward/back   +111 m FORWARD    ->  actually BACKWARD
  vertical       +28 m DOWN        ->  actually UP (camera ends on the ceiling)
  pitch          (I mislabeled it "nose-down -34deg"; the same accel solution is NOSE-UP)
```
So BOTH the along-track and vertical estimates are sign-inverted relative to truth. The diverged attitude
then (a) feeds the controller backwards — it thrusts up-and-back believing it is driving forward — and
(b) integrates into the bogus "forward+down dive" telemetry I first reported. The real impact spike
(`|a|=1101`, roll→179°, peak gyro 68 rps) is hitting the back wall/ceiling while inverted, not the floor.

**Candidate root causes (commander's call):**
1. **Attitude-estimate (AHRS/ESKF) pitch inversion** in the self-localization — the most likely single
   cause, since it explains the wrong-way control AND the inverted position integration at once. Connects
   to the A6/A7 "attitude inverts" theme; may start inverted or flip right after egress.
2. A **pitch-command / body-rate sign error** in pursuit (gyro_y rode the +1.50 rps cap from the first
   tick — if that sign is wrong the nose goes up when chasing a gate ahead).
Either way the drone is driven the WRONG WAY, so gates=0 is a direction failure, not a speed/altitude one.

**Retraction:** ignore the earlier "runaway forward speed to 44 m/s / altitude dive" framing — that Δx was
the diverged estimator, not real ground speed. The verified failure is the inversion above.

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
reach 30 Hz the *non-VP* path needs profiling.

## Recommended next steps (flight-stack — commander's call; stack NOT touched)
1. **Find the attitude/estimate inversion (the #1 blocker).** The drone flies UP+BACK while the estimate
   says forward+down — verify the AHRS/ESKF pitch sign and the body-rate (gyro_y) command sign on the live
   VQ2 wire. A bench check: at egress, does the estimated attitude match the realized accel tilt, or is it
   sign-flipped? Until the drone is driven toward the gate it sees, speed/altitude tuning is moot.
2. **Then (downstream of the inversion) re-check pursuit speed + altitude hold.** Only meaningful once the
   drone heads the right way; the "44 m/s runaway / -34deg" figures were the diverged estimator and can't be
   trusted as control targets until (1) is resolved.
3. **(Lower priority) profile the non-VP nav tick** to claw back the last ~16->30 Hz; the VP fix already
   did its job (0% drops, 33 ms core).

## Artifacts (this dir)
- `crash_analysis.py` — tlog realized IMU/attitude reconstruction. NOTE its `acc_pitch` sign reads nose-up
  as negative — the run1 "-34deg" is NOSE-UP. Cross-check against the onboard camera (frame ~500 = ceiling).
- `onboard_run1_frame500_ceiling.png` — camera staring at the ceiling = nose-up proof of the inversion.
- `navtime.py` — offline navigator timing (carried from A7; the 33 ms vision core).
- `timing_run1.txt` — `video_timing_report` (0% drops). `onboard_run1_frame303.png` — the VQ2 lit-warehouse proof frame.
- Recordings (gitignored, on ShadowPC): `data/runs/20260630_005945_*` (run1, fresh GO),
  `20260630_010404_*` (run2, RESTART late-join). Earlier `005808`/`005231` were the VQ1-mistake / stall lead-ups.
- Flight stack untouched (clean tree); only this handoff dir committed.
