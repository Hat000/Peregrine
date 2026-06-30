# VQ2 self-localized SLOW-lap — ATTEMPT 12 (seeker control-softening) — 2026-06-30

**One-line outcome:** Softening ENGAGED (kp_att 10→4 + body-rate slew 8 rad/s², both confirmed live) but
INEFFECTIVE — the egress→pursuit handoff still whips **nose-UP to +22°** and the drone flies a **polygonal
bang-bang path** (commands on only 25% of ticks, ~0.5–1.2 s straight coasts between saturated bursts), yaws
left, climbs, and crashes into gate 1. **CRASH, gates=0, ~4 s.** ≈A11 (no improvement; arguably a touch worse).

**Branch flown:** `claude/loving-galileo-92f020` @ `8b702e7`. **Sim:** AI-GP 1.0.3379, **R2-TRAINING = VQ2**
(red-glow 4.5k–43k px). `--gate-seeker --deploy-profile vq2_case_c --label vq2_slow_seeker_a12`. Loop **13.2
Hz / CHOKED** (59 ticks, worst 110 ms); frame drops 0% (144/144).

## The fix was engaged — verified, not assumed
`_build_casec_seeker` (fly_rl.py:993) builds `controller=make_seeker_controller(**profile.controller_overrides)`
and `vq2_case_c.controller_overrides = {kp_att:4.0, body_rate_slew_max_rps2:8.0}`. Live-confirmed engaged
controller: **kp_att=4.0, slew=8.0, max_body_rate=4.0 → max non-saturating attitude error = 1.5/4 = 21.5°.**

## Why softer gains didn't help: the handoff TARGET is nose-up (direction, not stiffness)
```
 t(s) pitch  pos(x,y,z)        cmd[r,p,y]            note
 0.3   -18  ( +0.0,+0.0,-0.7) [ 0.00, 0.00, 0.00]  egress: HELD -18deg nose-down, flew fwd (fix #A11 still good)
 1.2   -12  ( +1.7,+0.0,-0.8) [-0.01,+1.50,-0.04]  <-- egress END: +1.50 NOSE-UP again (kp=4 STILL saturates)
 1.4    +8  ( +2.5,+0.0,-0.9) [+0.01,+1.50,-0.06]  pitch crossing level, still commanding nose-up
 1.6   +22  ( +3.2,+0.0,-0.8) [ 0.00, 0.00, 0.00]  reached +22deg NOSE-UP, cmd->0  => desired attitude WAS +22deg up
 2.4   +17  ( +3.9,+0.0,-0.6) [+0.09,-1.50,-1.36]  saturated correction; yaw starts winding
 2.9   -16  ( +3.5,-0.3,-1.4) [ 0.00, 0.00, 0.00]  yaw +32, rolling, drifting (coast)
 4.3   +19  ( +5.5,-1.9,-1.0) [ 0.00, 0.00, 0.00]  yaw +45 -> into gate 1
```
The controller drove pitch −12° → **+22° nose-up** then commanded zero — so the *desired* attitude at the
handoff is ~+22° nose-up, a ~34° error that saturates the rate cap **even at kp=4** (21.5° threshold). The
slew only delayed reaching the clip; it can't bend a wrong-DIRECTION target. The validated forward feedforward
is nose-DOWN (−0.77, see `vq2-forward-pitch-sign-2026-06-30`), so the egress→pursuit transition is computing a
nose-UP desired attitude. **Lowering kp further won't fix direction — smooth/correct the transition TARGET
directly** (the post-egress desired should be the nose-down cruise lean, not nose-up).

## DOMINANT lead (pilot, emphasized again): polygonal bang-bang motion
Fengyou: *"every movement was separate, line polygons, not a curve ... commands sent VERY sparsely."*
Quantified: **active 15/59 ticks = 25%**, zero-coast runs of **7–15 ticks (~0.5–1.2 s)** between bursts, and
the active bursts are nearly all large (1.17–1.50, 8/15 at the ±1.50 clip). So the drone gets one saturated
shove, then coasts blind on a straight glide for up to 1.2 s, then another shove — the path is a polygon, not
a pursuit curve. Critically, **detections were present on far more ticks than 25%** (range 6–35 m through the
flight), so the sparsity is NOT only detection gaps — `command_visual` is returning hold/zero on many ticks
where a gate IS seen. This polygonal control is its own crash cause (can't thread a gate without continuous
steering) and is independent of the handoff-direction bug.

## Next-fix recommendation (commander's call)
1. **A13 = hold-last-demand-through-gaps + continuous command** (matches the pre-registered branch). Stop
   zeroing the command between detections; hold/decay the last good pursuit demand so motion is a curve. AND
   trace why `command_visual` emits zero on ticks where the gate IS detected (command-issuance gating, not just
   the no-detection coast) — that's why active << detection rate.
2. **Smooth/correct the egress→pursuit transition TARGET** so the post-egress desired is nose-down cruise, not
   +22° nose-up. (Softening kp/slew alone is proven insufficient — it's direction, not stiffness.)
3. Loop still choked **13 Hz** — perf/detection cadence remains a background contributor to the sparsity.

## Artifacts
- `nav_estimate.jsonl` — per-tick estimate + cmd (committed; run dir gitignored).
- Recording (gitignored, on ShadowPC): `data/runs/20260630_222041_vq2_slow_seeker_a12_f1/onboard.mp4` (2×).
- Flight stack untouched; handoff dir only.
