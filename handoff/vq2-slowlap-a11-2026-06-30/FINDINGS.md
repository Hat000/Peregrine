# VQ2 self-localized SLOW-lap — ATTEMPT 11 (egress attitude-freeze) — 2026-06-30

**One-line outcome:** Egress-freeze fix CLEARED the A10 acquisition trap (held the −18° nose-down spawn
attitude, flew forward, gate 0 stayed framed and it closed to **5.2 m**) — but at egress-end a **+1.50
nose-UP** command fires (not the validated −0.77 nose-down pursuit), pitching the camera up; the drone then
spirals backward into the environment. **CRASH, gates=0.**

**Branch flown:** `claude/loving-galileo-92f020` @ `31aca88` (egress attitude-freeze: zeroes egress roll/pitch
rate, holds spawn attitude instead of leveling). **Sim:** AI-GP 1.0.3379, **R2-TRAINING = VQ2** (red-glow
5443–37752 px = lit warehouse). `--gate-seeker --deploy-profile vq2_case_c --label vq2_slow_seeker_a11`. Loop
**12.1 Hz / CHOKED** (80 ticks, worst 121 ms, 100% over budget); frame drops 0% (211/211).

## The fix worked where it was aimed: egress acquisition
Pilot (Fengyou): *"it held forward flight slowly near the ground, then all of a sudden pitched up, which
caused it to start accelerating backwards, collided with something, that made it roll a bit."* Matches the
trace exactly:
```
 t(s) roll pitch yaw  pos(x,y,z)        cmd[r,p,y]            note
 0.4   -0   -18   -0  ( +0.1, +0.0, +0.0) [ 0.00, 0.00, 0.00]  egress: HELD -18deg nose-DOWN (fix working)
 1.0   -0   -18   -1  ( +1.3, +0.0, -0.8) [ 0.00, 0.00,-0.01]  flying FORWARD, gate 0 framed (~8-10 m)
 1.3   -0   -18   -0  ( +2.3, +0.0, -0.8) [-0.06,+1.50,-0.01]  <-- egress END: +1.50 NOSE-UP cmd fires
 2.0   -0   +18   +2  ( +5.1, +0.1, -1.1) [ 0.00, 0.00, 0.00]  pitched UP +18deg; coasting fwd on momentum
 3.1   -6   +20  +12  ( +6.5, +0.3, -2.0) [ 0.00, 0.00, 0.00]  closest: detector range 5.2 m to gate 0
 3.8   -8   +21  +11  ( +5.4, +1.7, -1.7) [-1.50,-1.50,-1.50]  saturated burst; rolling/yawing off
 4.9  +25  -11  +32  ( +1.0, +4.4, -2.3) [ 0.00, 0.00, 0.00]  spiraling: roll +25, yaw +32
 6.4  +23  -20  +35  ( -5.7, +0.6, -1.4) [+1.50,-1.50,-1.50]  backward to -5.7 m -> environment collision
```
Detector timeline confirms: gate 0 in frame from spawn (range 10.2→8.5 m through t2), **closest 5.2 m at
t≈3 s**, then range grows back to 24–38 m as it pitches up and spirals, lost by t6. So this is NOT "gate
drifts out despite steady attitude" and NOT "no forward commit" — it committed forward and got close. The
break is the **egress→pursuit handoff**.

## Two targets for the next fix
1. **Egress-end fires +1.50 nose-UP instead of the −0.77 nose-down pursuit.** The egress HOLD is correct now,
   but the very first post-egress command is `+1.50` pitch (nose-up) — the same bogus value A10's egress
   emitted, and the OPPOSITE of the validated forward feedforward (see `vq2-forward-pitch-sign-2026-06-30`:
   a real forward `accel_ned` gives −0.77 nose-down). Pursuit is still not engaging; a +1.50 nose-up
   (anchor-release re-level? raw rate?) fires in its place and breaks acquisition right at 5 m. Trace the
   command source on the egress→pursuit tick (t≈1.3 s here).
2. **NEW — commands arrive as sparse SATURATED bursts, not sustained pursuit (Fengyou's catch).** Quantified
   over the flight: **only 22% of ticks emit a command (78% are zero/coast), with zero-gaps up to ~1.2 s,
   and 15 of 18 active bursts are slammed to the rate clip (±1.50).** So even when it acquires, it fires one
   maxed-out burst then coasts blind until the next detection/regime tick — bang-bang with long dead periods,
   which is itself destabilizing (the spiral). Likely compounded by the choked 12 Hz loop + intermittent
   detections + the "no detection → zero-rate coast" policy. A sustained pursuit needs a command every tick
   (hold the last good demand through detection gaps), not a spike-then-silence.

## Artifacts
- `nav_estimate.jsonl` — per-tick estimate (roll/pitch/yaw, dead-reckoned pos, cmd body_rate, thrust). Copied
  here (run dir is gitignored).
- Recording (gitignored, 24 MB, on ShadowPC): `data/runs/20260630_201218_vq2_slow_seeker_a11_f1/onboard.mp4`
  (2× slow-mo, gate overlay) + `video.bin`/`video_index.jsonl`. Render:
  `python scripts/render_vision_video.py <run_dir> <out.mp4> --slowmo 2`.
- Flight stack untouched; handoff dir only.
