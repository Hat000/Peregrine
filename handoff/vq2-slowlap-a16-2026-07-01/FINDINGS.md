# VQ2 Slow-Lap A16 — ff_owns_horizontal (pursuit nose-up ROOT fix) — 2026-07-01

**OUTCOME (footage-grounded; attitude+commands trustworthy, position_ned is dead-reckon fiction):**
✅ **The nose-up-into-the-ceiling is GONE — `ff_owns_horizontal` works.** Pursuit now holds a sane
nose-DOWN attitude (eases −17° → −8° forward lean) with a small decaying pitch-rate command, instead
of the A15b saturated +1.5 rad/s nose-up slam. **BUT the flight still fails, and the two remaining
blockers are now the story: (1) the drone barely leaves the START SHELF — mostly sits, little/no
liftoff-translation; (2) the moment it DOES move, frame-drops black out the onboard view, so the
movement phase is unobservable.** Crashed at ~3.4s, gates=0.

HEAD flown: `d5b8fed` (flight stack byte-identical to `35561c7` — the ff_owns_horizontal commit; later
commits are memory-only). Config: `--gate-seeker --deploy-profile vq2_case_c --seeker-detector yolo
--seeker-weights models/gate_clean_ens_course_L110.pt`. Launch ritual correct (fly_rl first → waited
for "Waiting PASSIVELY" → GO): clean `GO!` at to_go≈0, **no late-join**.

---

## seeker-diag / loop-rate

    [prewarm] YOLO detector warmed in 12.16s
    GO!  armed sim_t=111.361s  (clean, no late-join)
    detector=yolo   HARD COLLISION -> abort
    [loop-rate]  13.8 Hz / 47 ticks / worst work 504 ms / 8.5% over -> CHOKED
    [seeker-diag] cmds=18 pursuit=17 none=1 (valid_empty=0 continuity=1) bridged=1  gates=0

Pursuit engaged strongly (17 pursuit ticks, pose EVERY tick, valid_empty=0). Loop badly CHOKED at
**13.8 Hz** (worse than A15b's 18.1) — see frame-drop lead below (same root suspect).

---

## THE FIX WORKED — pitch arc before vs after

| | A15b2 (before fix) | **A16 (ff_owns_horizontal)** |
|---|---|---|
| pitch-rate cmd at pursuit handoff | **+1.5 rad/s (SATURATED)** | **+0.47 → +0.08 rad/s (gentle, decaying)** |
| pitch trajectory | −14° → **+15° nose-UP → CEILING** | −17° → **−8.3° (eases toward ~−7° forward lean)** |
| result | ballooned up into ceiling | stays nose-down; no ceiling slam ✅ |

A16 pursuit arc (ticks 30–46): pitch −14.9 → −8.3° (monotonic ease toward forward lean), pitch-rate
cmd +0.47→+0.08, yaw_des +0.8→+0.4° (gate ~centered), thrust ~0.6. Exactly the predicted behavior.

---

## PILOT'S EYEBALL (ground truth — overrides estimate)

1. "Not getting a good fix on the first gate / looks like the wrong vision model." → **See the
   red_glow-overlay caveat below — the flight DID use YOLO; the video overlay is misleading.**
2. "For the majority of the video there's NO movement — it just sits on the start shelf."
3. "When we DO get movement (lights green / pursuit), the camera lags out, I lose telemetry — really
   bad frame drops."

So: sane nose-down attitude (per data) but **little actual liftoff off the shelf**, and the movement
phase is **blacked out by frame-drops** → we still can't watch it thread (or miss) gate 1.

---

## ⚠️ RENDER-OVERLAY CAVEAT (important — resolves the pilot's "wrong model" read)
The flight detector was **YOLO** (`gate_clean_ens_course_L110.pt`), confirmed: `[prewarm] YOLO`,
`detector=yolo`, no fallback, `valid_empty=0` (a YOLO pose every tick). **BUT
`scripts/render_vision_video.py` overlays the classical `RedGlowGateDetector`, NOT YOLO** — so the
boxes/range drawn on the onboard.mp4 are red_glow, unrelated to the flight's actual YOLO fixes. The
pilot's "bad gate fix" impression is the red_glow overlay, not YOLO. **ACTION (flight-test agent):
the planned handoff-dir render tool will overlay the REAL YOLO detections + the per-frame command
indicator so this confusion goes away.**

Model inventory (for the "use the best version" request): `gate_clean_ens_course_L110.pt` (course,
flown), `gate_clean_ens_precision_L107.pt` (precision-lever, best <0.5m fixes, same ~21ms), older
`gate_yolo11s_curriculum_v2.pt`. Ensemble = course++precision (best fix, ~2× cost → worse choke).

---

## PRIORITIZED NEXT-FIX LEADS

1. **[#1 OBSERVABILITY — commander already investigating] Kill the frame-drops (heading_vp +
   manhattan_lines, per-frame in `_maybe_run_vision`).** Confirmed suspect (commit 5b3c131). They
   black out the movement phase (pilot loses view exactly when it starts moving) AND drive the
   13.8 Hz choke / 504 ms worst-work. Until fixed, we CANNOT observe whether the drone lifts off and
   threads gate 1 — this gates the whole diagnosis. (Drop rate is variable run-to-run: A16 recording
   378/1062 ≈ 61%, A15b was ≈95%.)

2. **[#2 NO-LIFTOFF / no shelf-egress] The drone sits on the start shelf and barely translates.**
   With the nose-up fixed, this is the next flight blocker: sane nose-down attitude (−17°→−8°) +
   thrust ~0.6 but the pilot sees little/no forward liftoff off the shelf. Suspect insufficient
   climb/thrust to leave the shelf, or the egress→pursuit forward feedforward is too weak to build
   translation. Re-diagnose once #1 makes the movement observable.

3. **[BLOCKED on #1, #2] Yaw un-mirror (337c554) STILL untested** — A16 never reached the off-axis
   gate 2 (crashed at ~3.4s on the shelf/near gate 1); yaw_des stayed ~+0.4° (gate centered). Same as
   A15b: blocked behind survival past gate 1.

4. **[MODEL — pilot request] Consider deploying the precision model / ensemble for a better gate
   fix.** valid_empty=0 already (recall fine), so this is about pose PRECISION, not recall. Precision
   single (precision_L107) is same-speed; the ensemble worsens the #1 choke. Recommend precision
   single AFTER the frame-drops are fixed (so the added-precision effect is observable and not
   choke-confounded).

---

## ARTIFACTS
- `nav_estimate.jsonl` (47 ticks), `seeker_diag.txt`
- Onboard mp4 (378 frames, red_glow overlay — see caveat) on ShadowPC:
  `C:\Users\Shadow\Peregrine\data\runs\20260701_161958_vq2_slow_seeker_a16_f1\onboard.mp4`
