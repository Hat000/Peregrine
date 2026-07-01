# VQ2 Slow-Lap A15b — pre-warmed YOLO (launch-freeze fix) — 2026-07-01

**OUTCOME (grounded in the PILOT's live eyes, not the estimate):** Pre-warm WORKS — live
observation is UNBLOCKED (pilot saw both flights clearly, no camera freeze). Best result of the
campaign: **flight 1 flew up+forward and PASSED gate 1**, then flew into the ceiling. The
diagnosis is now sharp and dynamically confirmed: **the frozen egress attitude is a GOOD
trajectory (it threads gate 1 with zero pursuit help); the pursuit vertical handoff is ACTIVELY
HARMFUL — the instant it engages it commands a big nose-UP pitch rate and balloons up into the
ceiling.** Blocker #3 (egress→pursuit nose-up overshoot) is the #1 killer. The yaw un-mirror fix
(337c554) is UNTESTABLE until #3 is fixed — both flights die at the nose-up overshoot before ever
reaching the off-axis gate 2.

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

### Flight 1 (LATE-JOIN) — the gift: pure frozen-egress ballistic
- **Held the egress freeze for ALL 259 ticks**: pitch ~−16° nose-down, thrust ~0.16–0.37,
  `yaw_des=none` / `tsv=none` / zero rate commands **the entire flight**. The egress-freeze
  **NEVER released into pursuit** (a late-join artifact — see caveat).
- gate_index 0→1 at tick ~72: the frozen nose-down + thrust trajectory **threaded gate 1**.
- With pursuit/vision-yaw never active, it could not steer right for gate 2 → continued the
  ballistic climb into the ceiling. (So "didn't turn right" ≠ a yaw-sign test; steering was OFF.)

**Cross-flight conclusion:** pursuit-OFF (flight 1) passed gate 1; pursuit-ON (flight 2) over-pitched
and missed. The pursuit vertical/pitch target is wrong — it drives nose-UP when the working egress
attitude is nose-DOWN. Softening (A12, kp_att 10→4) can't fix a wrong-direction target.

---

## PRIORITIZED NEXT-FIX LEADS

1. **[PRIMARY] Fix the pursuit→pitch handoff nose-up overshoot (blocker #3).** The moment pursuit
   engages it commands +1.5 rad/s nose-UP (flight 2, ticks 60–70), pitching −14°→+15° and flying up
   into the ceiling. The frozen egress attitude (−16° nose-down, thrust ~0.37) is a GOOD trajectory
   — the pursuit vertical target should hold/continue that, not command a large nose-up. Suspect the
   pursuit pitch/altitude target sign or the egress→pursuit blend. This gates everything downstream.

2. **[BLOCKED on #1] Confirm the yaw un-mirror (337c554) on an off-axis gate.** Neither flight
   reached the right-side gate 2 with pursuit active, so the fix is still unconfirmed dynamically.
   Once #1 lets the drone survive past gate 1 into a real off-axis pursuit, re-run the gate-RIGHT →
   nose-RIGHT test.

3. **[SEQUENCING — operator note] Do NOT pre-GO the race.** Launch fly_rl FIRST against a
   `started=False` waiting room; it pre-warms + imports (~48s) then sits at `waiting: started=False`;
   send GO (Enter) only then, so it arms at `to_go≈0`. Pre-GO'ing makes fly_rl LATE-JOIN
   (`to_go=-48s`), which also appears to break the egress-release (flight 1 never left freeze) and
   zeroes the seeker-diag counters. Flight 2 used the correct sequence.

4. **[LOW] Recorder frame-drops (~95%: 25/938 flight 2, 3/259 flight 1).** Separate from the tick-0
   freeze (which the pre-warm fixed). The LIVE stream is fine (pilot observed both flights), so this
   only makes post-hoc mp4 review useless — not a flight blocker. If post-hoc video matters later,
   the video-recorder thread is starving under YOLO GPU load.

---

## ARTIFACTS
- `nav_estimate_flight1_latejoin.jsonl` (259 ticks), `nav_estimate_flight2_clean.jsonl` (71 ticks)
- `seeker_diag.txt` (both flights' key log lines)
- Onboard mp4s exist but are near-empty (3 / 25 frames) — NOT worth reviewing; rely on the pilot's
  live read. Paths on ShadowPC:
  - `C:\Users\Shadow\Peregrine\data\runs\20260701_041856_vq2_slow_seeker_a15b_f1\onboard.mp4`
  - `C:\Users\Shadow\Peregrine\data\runs\20260701_042347_vq2_slow_seeker_a15b2_f1\onboard.mp4`
