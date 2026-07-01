# VQ2 Slow-Lap A15b — pre-warmed YOLO (launch-freeze fix) — 2026-07-01

**OUTCOME (grounded in the PILOT's live eyes, not the estimate):** Pre-warm WORKS — live
observation is UNBLOCKED (pilot saw both flights clearly, no camera freeze). **Flight 1 flew
up+forward and PASSED gate 1**, then flew into the ceiling. The dynamically-confirmed diagnosis:
**the pursuit vertical handoff is ACTIVELY HARMFUL — the instant it engages (flight 2) it commands
a big nose-UP pitch rate and balloons up into the ceiling.** Blocker #3 (egress→pursuit nose-up
overshoot) is the #1 killer. The yaw un-mirror fix (337c554) is UNTESTABLE until #3 is fixed — both
flights die at the nose-up overshoot before ever reaching the off-axis gate 2.

**PILOT'S REVISED READ OF FLIGHT 1 (do NOT over-credit it):** the gate-1 pass was most likely the
controller issuing an *initial* fly command that then **dropped out**, leaving the drone on a fixed
nose-down attitude that **coincidentally** threaded gate 1 — luck, not robust pursuit. Treat flight 1
as "an initial command + dropout that happened to work," NOT as proof a frozen egress attitude is a
reliably good trajectory. The load-bearing finding is flight 2's nose-up overshoot.

HEAD flown: `142b6bc` (fly_rl.py byte-identical to `12f366b` — the pre-warm commit; +2 memory/docs
commits don't touch the flight stack). Config: `--gate-seeker --deploy-profile vq2_case_c
--seeker-detector yolo --seeker-weights models/gate_clean_ens_course_L110.pt` (single model, ~21ms).

---

## seeker-diag / loop-rate (both flights)

    FLIGHT 1 (A15b, LATE-JOIN — see caveat):
      [prewarm] YOLO detector warmed in 7.81s
      LATE-JOIN GO!  to_go=-48.40s   armed sim_t=51.914s
      gate 0 PASSED -> targeting 1 ; HARD COLLISION -> abort
      [loop-rate]  24.9 Hz / 259 ticks / worst work 468 ms / 1.5% over -> CHOKED
      [seeker-diag] cmds=0 pursuit=0 none=0 (valid_empty=0)   gates=1

    FLIGHT 2 (A15b2, CLEAN GO — the normal path):
      [prewarm] YOLO detector warmed in 7.49s
      GO!  armed sim_t=177.179s        (to_go~0, no late-join)
      HARD COLLISION -> abort
      [loop-rate]  18.1 Hz / 71 ticks / worst work 515 ms / 5.6% over -> CHOKED
      [seeker-diag] cmds=12 pursuit=12 none=0 (valid_empty=0)  gates=0

**MILESTONE — pre-warm confirmed:** worst-work tick **515 ms (flight 2) / 468 ms (flight 1)** vs
A15's **~2873 ms**. The tick-0 warmup stall is GONE from the control loop, and — per the pilot —
the LIVE onboard stream did NOT freeze on either flight. `valid_empty=0` (recall still solid).

---

## PILOT'S EYEBALL (ground truth — overrides the estimate)

- **Flight 1 (A15b):** "flew up and forward PERFECTLY. It hit the first gate as well as I could
  hope. However it failed to recognize the second gate is to its RIGHT and instead kept flying
  forward and up until it hit the ceiling."
- **Flight 2 (A15b2):** "flew forward close to the ground, and at some point — almost as if the
  flight controls finally warmed up — it flew up, but instead of up-and-through the gate the system
  seemed to lag out and after that initial up command continued to fly up and up, over the top of
  the gate into the ceiling."

Both flights ended by climbing into the CEILING.

---

## NAV DATA (attitude + commands are TRUSTWORTHY post gyro-fix; position_ned is DEAD-RECKONED FICTION)

### Flight 2 (CLEAN, the normal path) — the smoking gun for blocker #3
| phase | ticks | pitch | pitch-rate cmd | vision | note |
|---|---|---|---|---|---|
| egress freeze | 0–56 (~3.1s) | held **−16° nose-DOWN** | 0 (frozen) | tsv=none, yaw_des=none | correct forward-intent hold |
| pursuit handoff | 60–70 (~0.6s) | **−14° → +15° nose-UP** | **+1.24 → +1.50 rad/s** | yaw_des +1.4°→+0.3° (gate ~centered) | over-pitch UP, thrust→0.6, CEILING |

### Flight 1 (LATE-JOIN) — an initial command + dropout that coincidentally threaded gate 1
- The data shows the drone on a **fixed ~−16° nose-down attitude for ALL 259 ticks**: thrust
  ~0.16–0.37, `yaw_des=none` / `tsv=none` / zero *sampled* rate commands. No active pursuit ever ran
  (a late-join artifact — see caveat).
- **Pilot's reading (preferred):** an *initial* fly command was issued that then **dropped out**,
  and the resulting fixed nose-down attitude **coincidentally** threaded gate 1 (gate_index 0→1 at
  tick ~72) before the ballistic climb into the ceiling. This is LUCK, not a robust trajectory — do
  not treat "frozen egress attitude" as a reliable good path.
- With pursuit/vision-yaw never active, there was no steering at all → "didn't turn right for gate 2"
  is NOT a yaw-sign test; steering was simply OFF.

**Cross-flight conclusion:** the load-bearing evidence is flight 2 — the moment pursuit engages it
commands nose-UP (+1.5 rad/s) and flies into the ceiling. Flight 1 (pursuit never active) only tells
us a bare nose-down attitude can, by luck, thread gate 1. The pursuit vertical/pitch target is wrong-
DIRECTION (drives nose-UP); softening (A12, kp_att 10→4) can't fix a wrong-direction target.

---

## PRIORITIZED NEXT-FIX LEADS

1. **[PRIMARY] Fix the pursuit→pitch handoff nose-up overshoot (blocker #3).** The moment pursuit
   engages it commands +1.5 rad/s nose-UP (flight 2, ticks 60–70), pitching −14°→+15° and flying up
   into the ceiling. The pursuit vertical target should keep the drone tracking the gate forward, not
   command a large nose-up. Suspect the pursuit pitch/altitude target sign or the egress→pursuit
   blend. This gates everything downstream. (Caveat: flight 1's coincidental gate-1 thread on a bare
   nose-down attitude is NOT proof the egress attitude is the right target — don't over-fit to it.)

2. **[BLOCKED on #1] Confirm the yaw un-mirror (337c554) on an off-axis gate.** Neither flight
   reached the right-side gate 2 with pursuit active, so the fix is still unconfirmed dynamically.
   Once #1 lets the drone survive past gate 1 into a real off-axis pursuit, re-run the gate-RIGHT →
   nose-RIGHT test.

3. **[SEQUENCING — operator note] Do NOT pre-GO the race.** Launch fly_rl FIRST against a
   `started=False` waiting room; it pre-warms + imports (~48s) then sits at `waiting: started=False`;
   send GO (Enter) only then, so it arms at `to_go≈0`. Pre-GO'ing makes fly_rl LATE-JOIN
   (`to_go=-48s`), which also appears to break the egress-release (flight 1 never left freeze) and
   zeroes the seeker-diag counters. Flight 2 used the correct sequence.

4. **[COMMANDER — investigate] Recorder frame-drops are REALLY BAD (~95%: 25/938 flight 2, 3/259
   flight 1).** Distinct from the tick-0 warmup freeze that the pre-warm fixed. The LIVE stream is
   fine (pilot observed both flights), so this doesn't block *flying* — but it makes post-hoc mp4
   review useless, and worst-work ticks were still 468–515 ms (over the 33 ms budget → CHOKED).
   **Pilot recalls we have hit this exact bad-drop symptom before, and it was traced to a per-frame
   stack element running too long and holding up the pipeline — he thinks it was the "Manhattan /
   vanishing-point" module (a per-frame vanishing-point / Manhattan-world estimate) blocking the loop.
   Commander: please check whether that module (or an equivalent per-frame vision step) is still in
   the live path and time-boxed — it's the prime suspect for both the ~95% recorder starvation and
   the 468–515 ms worst-work ticks.** The pilot did NOT have the exact name; verify against current
   code.

5. **[FUTURE VIDEOS — rendering request from the pilot] Overlay a per-frame "command sent" indicator
   in rendered onboard videos**, so we can see the command cadence relative to the video frame rate
   (how fast commands update vs. how fast frames arrive). `nav_estimate.jsonl` carries `sim_time_ns` +
   `body_rate` per tick and `video_index.jsonl` carries per-frame `sim_time_ns`, so each frame can be
   tagged with the nearest tick's command (mark frames whose tick issued a nonzero/pursuit command).
   NOT applied retroactively to the A15b clips; to be added to the render workflow for future runs.
   (Flight-test agent will handle this in a handoff-dir render tool; flagged here so commander is
   aware of the diagnostic.)

---

## ARTIFACTS
- `nav_estimate_flight1_latejoin.jsonl` (259 ticks), `nav_estimate_flight2_clean.jsonl` (71 ticks)
- `seeker_diag.txt` (both flights' key log lines)
- Onboard mp4s exist but are near-empty (3 / 25 frames) — NOT worth reviewing; rely on the pilot's
  live read. Paths on ShadowPC:
  - `C:\Users\Shadow\Peregrine\data\runs\20260701_041856_vq2_slow_seeker_a15b_f1\onboard.mp4`
  - `C:\Users\Shadow\Peregrine\data\runs\20260701_042347_vq2_slow_seeker_a15b2_f1\onboard.mp4`
