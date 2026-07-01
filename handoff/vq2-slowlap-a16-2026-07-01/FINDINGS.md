# VQ2 Slow-Lap A16 — ff_owns_horizontal (pursuit nose-up ROOT fix) — 2026-07-01

**OUTCOME (footage-grounded; attitude+commands trustworthy, position_ned is dead-reckon fiction):**
✅ **The nose-up-into-the-ceiling is GONE — `ff_owns_horizontal` works.** Pursuit now holds a sane
nose-DOWN attitude (eases −17° → −8° forward lean) with a small decaying pitch-rate command, instead
of the A15b saturated +1.5 rad/s nose-up slam (confirmed from the nav DATA — trustworthy). **The
flight still fails (crash ~3.4s, gates=0), but the actual flight PHASE is UNOBSERVED: frame-drops
black out the view the instant the race goes GREEN.** See the corrected pilot read below — the
"sitting on the shelf" was mostly the pre-GO WAITING ROOM (red side-lights), NOT a confirmed liftoff
failure.

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
2. **[CORRECTED] "Most of the recording the SIDE-LIGHTS were RED = still in the WAITING ROOM, not
   racing. The moment the light turns GREEN and the drone activates is exactly when the frame-drops
   hit."** So the long "sitting on the shelf" stretch was mostly the pre-GO waiting room (by design),
   NOT a confirmed liftoff failure. (The recorder attaches during the passive-wait, so the recording
   spans waiting-room + the ~3.4s armed flight; nav_estimate's 47 ticks are only the armed flight.)
3. The actual flight phase (post-GREEN) is **blacked out by frame-drops** → we could NOT watch whether
   it lifts off, translates, or how it threads/misses gate 1.

**Key correlation for the commander:** frame-drops onset EXACTLY at GO / green-light / drone-activate
— i.e., when the armed flight loop + the full per-frame vision pipeline (heading_vp + manhattan_lines)
starts running under load. In the static red-light waiting room the frames are fine. That the drop is
*state-triggered by flight-active*, not gradual, points straight at a per-frame compute step that only
runs (or only saturates) once flying.

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
   manhattan_lines, per-frame in `_maybe_run_vision`).** Confirmed suspect (commit 5b3c131). NEW
   diagnostic: the drops onset **exactly at GO/green-light/drone-activate** — fine in the static
   red-light waiting room, saturate the instant the armed flight loop + full per-frame vision runs.
   So it's state-triggered by flight-active, not gradual — a per-frame step that only runs/saturates
   once flying. They black out the entire flight phase AND drive the 13.8 Hz choke / 504 ms
   worst-work. Until fixed we CANNOT observe liftoff/translation/gate-1 — this gates the whole
   diagnosis. (Drop rate variable run-to-run: A16 378/1062 ≈ 61%, A15b ≈ 95%.)

2. **[#2 UNKNOWN — flight phase unobserved, was overstated as "no-liftoff"] Whether the drone lifts
   off / threads gate 1 is NOT yet known.** CORRECTION: the "sits on the shelf" the pilot saw was
   mostly the pre-GO WAITING ROOM (red side-lights), not a confirmed liftoff failure — and the actual
   post-green flight is blacked out by #1. What we DO know from the nav DATA: 47 armed ticks, sane
   nose-down attitude (−17°→−8°), thrust ~0.6, crash (HARD COLLISION) at ~3.4s, gates=0. The crash
   cause is undetermined until #1 restores observation. Do NOT assume no-liftoff.

3. **[BLOCKED on #1] Yaw un-mirror (337c554) STILL untested** — A16 crashed at ~3.4s before reaching
   the off-axis gate 2; yaw_des stayed ~+0.4° (gate centered). Same as A15b: blocked behind surviving
   the (now unobserved) flight phase past gate 1.

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
