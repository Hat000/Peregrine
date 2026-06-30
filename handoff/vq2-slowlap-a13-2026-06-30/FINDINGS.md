# VQ2 self-localized SLOW-lap — ATTEMPT 13 (hold-last-demand bridge) — 2026-06-30

**One-line outcome:** Bridge helped but is insufficient (active ticks 25%→32%, flight 4s→**~10s**); the
`[seeker-diag]` line localizes the bottleneck to **`valid_poses_empty=96` (74% of ticks the pose pipeline
returns NO pose even when the gate is in view)** — the bridge only covers 29% of those gaps. **NEW Lead 3
(pilot + data): the gate-bearing YAW steers the WRONG way** — it yawed LEFT, away from a gate on the right.
**CRASH, gates=0.**

**Branch flown:** `claude/loving-galileo-92f020` @ `ec35a2b` (hold-last-demand bridge, `hold_last_demand_s=0.6`).
**Sim:** AI-GP 1.0.3379, **R2-TRAINING = VQ2**. `--gate-seeker --deploy-profile vq2_case_c --label
vq2_slow_seeker_a13`. Loop **12.7 Hz / CHOKED** (130 ticks); frame drops 0% (312/312).

## THE DATA THAT DECIDES A14 — the [seeker-diag] line
```
[seeker-diag] cmds=118 pursuit=13 none=105 (valid_empty=96 continuity=9 first_acq=0 other=0)
              -> bridged=30 (29% of none) held_legacy=75
```
- **pursuit=13 / 130 ticks = only 10% of ticks had a usable pose.** Real pursuit commands are that rare.
- **none=105, valid_empty=96** — on **74% of ticks the pose pipeline returns EMPTY**, vs only 9 continuity-
  rejects. So it is NOT a gating/rejection problem — PnP/pose comes back with nothing even though the detector
  sees gates. Pilot-confirmed: *"the gate is still in view in the very right side of the vision model, but not
  being detected."* (Edge-of-frame / oblique gate → no valid pose.)
- **bridged=30 (29%) / held_legacy=75** — the bridge covered only 29% of the gaps; **71% still zero-coast**,
  because valid poses are so sparse that most gaps exceed the 0.6 s bridge window (one zero-coast run was ~3.5 s).

**A14 (the diag's verdict): fix the POSE/DETECTION pipeline (why `valid_poses_empty` on 74% of in-view ticks),
NOT just lengthen the bridge.** The bridge is a bandaid over a pipeline that gives a pose only 1 tick in 10.

## NEW Lead 3 — gate-bearing YAW is inverted (turns AWAY from the gate)
Pilot: *"it pitched back down and the gate was in frame ... instead of flying forward it yawed LEFT — no
reason, nothing over there; the next gate is on the right ... yaws even more left to face absolutely nothing."*
The trace confirms it is COMMANDED, not drift: yaw jumps land exactly on the yaw-command bursts, which are
**consistently negative** (cmd_yaw −0.4…−1.50, **sum −30.2** over the flight) while the drone turns toward
+100° = physically LEFT = **away** from the gate it can see (on the right). To chase a gate on the right it
must yaw the OTHER way. **The gate-bearing-yaw / vision-yaw steering sign is backwards** — consistent with the
never-verified flag that the full gyro-vector negation `(-1,-1,-1)` also flipped the yaw axis
(`vq2-gyro-sign-probe-2026-06-30`: "also flips yaw → recheck vision-yaw/gate-bearing-yaw"). **This is high
priority: even with a perfect pose, inverted yaw-to-gate means pursuit can never converge on an off-axis gate.**
(The forward-pitch probe `vq2-forward-pitch-sign-2026-06-30` validated forward-along-heading, which is immune
to a consistent yaw inversion; gate-bearing yaw is the first thing that ISN'T.)

## Trajectory (estimate; dead-reckoned, direction trustworthy)
```
 t(s) pitch  yaw   pos(x,y,z)         note
 0.6   -18    -1  ( +0.4,+0.0,-0.8)  egress held nose-down, fwd (A11 fix still good)
 1.1   -12    -1  ( +1.7,+0.0,-0.7)  egress END: +1.50 nose-UP (Lead 1, untouched/expected)
 1.6   +28    +1  ( +3.3,+0.0,-1.0)  whipped to +28 nose-up -> gates leave frame (pilot)
 2.7    -4   +33  ( +3.9,+0.1,-1.3)  pitched back down, gate back in frame, but YAWING LEFT (Lead 3)
 5.6    +3   +64  ( +7.0,-0.6,-2.3)  long ~3.5s zero-coast glide, yawed ~63 deg off
 8.9    -1  +100  (+13.8,+2.5,-0.6)  yaw cmds drive it to +100 deg; sinks, floor bounce, crash
```

## Leads, prioritized for the commander
1. **Lead 3 — gate-bearing-yaw sign inverted (NEW, high).** Verify + fix the vision-yaw/gate-bearing-yaw sign
   post gyro yaw-negation. Without it pursuit yaws away from gates. Cheapest high-leverage fix.
2. **Pose pipeline `valid_poses_empty=96` (the diag's A14 decider).** Why does PnP/pose return empty on 74% of
   in-view ticks (esp. edge-of-frame / oblique gates)? This is the sparsity root; the bridge can't paper over
   a 1-in-10 pose rate.
3. **Lead 1 — egress→pursuit handoff still whips +1.50 nose-up to +28°** (untouched by A12/A13). Smooth the
   transition TARGET direction (not loop stiffness).
4. Loop still choked **12.7 Hz** (background contributor to pose sparsity).

## Artifacts
- `seeker_diag.txt` — the [seeker-diag] + [loop-rate] lines. `nav_estimate.jsonl` — per-tick estimate + cmd.
- Recording (gitignored, on ShadowPC): `data/runs/20260630_225052_vq2_slow_seeker_a13_f1/onboard.mp4` (2×).
- Flight stack untouched; handoff dir only.
