# VQ2 self-localized SLOW-lap — ATTEMPT 10 (full gyro sign negation) — 2026-06-30

**Branch flown:** `claude/loving-galileo-92f020` @ `a5c5e76` (`VQ2_GYRO_SIGN=(-1,-1,-1)` full angular-rate
negation at the wire + per-tick `nav_estimate.jsonl`). **Sim:** AI-GP 1.0.3379, **R2-TRAINING = VQ2**
(deliberately selected; lit-warehouse verified, red-glow 5489 px). `--gate-seeker --deploy-profile
vq2_case_c`. All prior fixes ride along (egress floor, accel-reject, ~30 Hz VP). test_mavlink_client 21/21.

## TL;DR — the gyro fix WORKS; the attitude estimate is finally trustworthy
| | Result |
|---|---|
| **Gyro sign fix effective?** | **YES — validated by ground truth.** The pilot watched the flight and confirmed *"the estimator is correct."* No more nose-up-to-ceiling runaway. The attitude estimate is **stable** (pitch +9±14°, roll +6±3°, yaw +12±9° — no divergence). |
| Flight duration | **~28 s / 367 ticks** — by far the longest of the campaign (A8 ~5 s, A9 ~2–5 s). |
| Gates passed | **0** — see the newly-exposed blocker below. |
| Loop / frames | 12.9 Hz / CHOKED (367 ticks, worst 144 ms); frame drops 0.0% (2 minor 65 ms surges over 46 s). |
| **New blocker (exposed)** | **The seeker does not pursue the visible gate.** After a short egress it pitches nose-UP and slowly drifts BACKWARD, away from a gate 9.3 m ahead, then bounces near the floor and hits the back wall. |

## The win: ground-truth-validated estimate
For the first time the live attitude estimate matches what the pilot sees. Pilot's live account:
*"flew forward for a little, pitched up, slowly flew back, then leveled out, bounced up and down near the
ground, hit the back wall and collapsed."* The `nav_estimate.jsonl` trace agrees exactly (estimate, now
trustworthy):
```
 t(s) est_roll est_pitch est_yaw  pos(x,y,z)           cmd_rate[r,p,y]   thr
  0.0    +0      +0      +0   (  +0.0, +0.0, +0.0)  [ 0.00, 0.00, 0.00] 0.27
  1.3    +0      -3      -0   (  +2.0, +0.0, -0.7)  [ 0.00,+1.50,-0.11] 0.60   <- egress (forward+climb)
  2.5    +1     +26      +3   (  +2.1, +0.1, -1.2)  [ 0.00, 0.00, 0.00] 0.37   <- pitches nose-UP, stalls fwd
  5.0    +8      +7     +11   ( -13.3, +0.6, -1.3)  [ 0.00, 0.00, 0.00] 0.16   <- now going BACKWARD
 ...     +8     +13     +12   ( steadily -x, -y )   [ 0.00, 0.00, 0.00]        <- holds tilt, drifts back ~25s
 25.1    +9     +13     +12   (-593.5,-203.0,+3.3)  [ 0.00, 0.00, 0.00] 0.37
 27.6    +4     -36     +47   (-728.7,-255.0,-2.5)  [ 0.00, 0.00,-0.00] 0.16   <- bounce / back-wall hit
```
The attitude HOLDS (no runaway) — the gyro fix did its job. The dead-reckoned position magnitude (−728 m)
is unbounded (no absolute fix; see below) but its DIRECTION (backward) is pilot-confirmed.

## The newly-exposed blocker: the seeker won't pursue the gate it can see
- **A gate is clearly visible during early flight.** Offline detector on the recorded frames: gates
  detected with `nearest_range` 9–36 m through frame ~558 (first ~6 s of flight), including **2 gates at
  9.3 m** (frames 310–496). After ~10 s the drone has drifted away and detections go to 0 (no gate in FOV).
- **But the seeker doesn't fly toward it.** `cmd_rate ≈ [0,0,0]` for ~25 s of the flight (only the egress
  pulse at t=1.3 and a late flail at t=26.4 are non-zero). It egresses forward a little, pitches nose-UP
  (+26° then settles +13°), and drifts BACKWARD — the opposite of pursuing a gate ahead.
- **Note on `time_since_vision=inf`:** this is EXPECTED, not the bug — the navigator is map-free (no
  TRACK_INFO gate map), so its absolute *position* fix legitimately never fires. The seeker's `command_visual`
  runs its OWN detect+PnP to chase the SEEN gate; that visual servo is what isn't steering.

**Read:** the gyro fix removed the attitude runaway that was masking everything. Now the next layer is the
**seeker's forward pursuit**: after egress it holds a nose-up tilt and drifts back instead of pitching
toward the visible gate. Candidate causes (commander's call):
1. **Seeker pursuit pitch / forward-command sign or attitude-target**, newly exposed now that the attitude
   estimate is correctly signed — it should pitch nose-DOWN to a gate ahead but ends nose-UP.
2. **The yaw-axis negation's effect on gate-bearing / vision-yaw** (flagged in the gyro-probe handoff: a
   full-vector negation also flips yaw — re-check the gate-bearing-yaw and vp-yaw branch sign).
3. Acquisition/regime: the seeker may never leave egress/hold into active pursuit (cmd_rate stays ~0).

## Recommended next steps (flight-stack — commander's call; stack NOT touched)
1. **Trace the seeker `command_visual` post-egress on this run** — does it see the 9.3 m gate (it's in the
   frames) and what pitch/forward command does it emit? `nav_estimate.jsonl` shows it emitting ~0 — find why
   (no gate handed in, vs gate seen but command ~0, vs wrong-sign pursuit).
2. **Re-verify the yaw/gate-bearing sign** after the full gyro negation.
3. This is the first run where the estimate is trustworthy enough that the SEEKER can be debugged fairly —
   prior attempts never got here.

## Artifacts
- `analyze_nav_estimate.py` — dumps the estimate trajectory + per-frame detection/range over the flight
  (the tables above). `python analyze_nav_estimate.py <run_dir>`.
- `nav_estimate_trace.txt` — saved output for this run (estimate trajectory + detection timeline + the
  "5% of ticks actively steering" stat).
- Recording (gitignored, on ShadowPC): `data/runs/20260630_042447_vq2_slow_seeker_a10_f1` (incl.
  `nav_estimate.jsonl`).
- Flight stack untouched; handoff dir only.
