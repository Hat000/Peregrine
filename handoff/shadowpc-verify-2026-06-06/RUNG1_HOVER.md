# Rung 1 — hover-hold (the balloon test). LIVE, ShadowPC 2026-06-06

**Config:** the faithful live controller (`make_controller(signs=_FAITHFUL_SIGNS, **FAITHFUL_TUNED_GAINS)`)
flown via `fly_vq1.py --faithful --hover-hold --alt-thrust-hi 0.40 --rate 100` — a FIXED position target
1.5 m above the start, no gates, isolating the vertical alt-hold + attitude-sign sanity. Bounded
`--max-climb-m 6 --max-tilt-deg 60 --geofence-m 4 --max-seconds 12`, force-disarm on exit.
**Reproduced ×2** (run1 `20260606_210716_rung1_hover`, run2 `20260606_212923_hover_reproduce`).
Pre-registered twin prediction: clean settle at 1.500 m, thrust→0.2656, vz→0.

---

# OBSERVATION (raw — telemetry + motor witness + GUI)

| signal (settle, last 5–7 s) | run 1 | run 2 (reproduce) |
|---|---|---|
| altitude hold mean | 1.399 m | 1.394 m |
| altitude net drift (slope) | −0.002 m/s | +0.003 m/s |
| **TRUE vz** (LOCAL_POSITION_NED) | mean 0.00, \|max\| 0.61, std 0.23 | mean −0.01, \|max\| 0.49, std 0.23 |
| vz peak-to-peak | 1.05 m/s | 0.93 m/s |
| **thrust** (commanded) | bang-bang 0.05↔0.40, 68% @hi | 0.05↔0.40, 66% @hi |
| **motors** (ACTUATOR_OUTPUT_STATUS) | mean 0.325, [0.17,0.40] | mean 0.324, [0.17,0.39] |
| **limit-cycle period** (vz zero-cross / thrust rail-flip) | 0.167 s / 0.146 s (~6 Hz) | 0.160 s / 0.150 s (~6.2 Hz) |
| attitude settle (pitch, roll) | 0.0°, 0.0° | 0.0°, 0.0° |
| horizontal drift max | 0.14 m | 0.17 m |

Both runs: armed at a fresh GO at the origin; the drone leveled the −17.8° resting pitch (a +1.24 rad/s
pitch-rate correction), climbed 0→~1.4 m in ~3 s, then a **sustained ~6 Hz vertical limit cycle around
~1.39 m** until the time cap; no collision, no abort, force-disarmed.

**GUI (teammate, given BEFORE this analysis):** "Yes, I saw bobbing" (amplitude not quantifiable by eye);
"the drone first pitched up and then thrusted up and hovered"; "no visible roll and yaw"; "pitch or
vertical height may have been climbing a little bit"; "nothing major different between the runs."
→ corroborates the telemetry: vertical bob, clean roll/yaw, level-up transient, ~flat net altitude.

---

# INTERPRETATION

## The wins (transfer CONFIRMED live)
- **No balloon, no sag, no runaway.** Altitude is *held* (net drift ≤0.003 m/s). The §7 balloon does NOT
  occur with the faithful config.
- **We own thrust in CTBR — confirmed live again.** Motors track our commanded collective (mean 0.324–0.325
  ≈ commanded mean), never pinned high. Task-3's verdict holds.
- **The faithful sign set transfers.** The drone leveled the −17.8° resting tilt to 0° and held it with **no
  roll/yaw** and ≤0.17 m horizontal drift — `body_rate_sign=[1,1,-1]` / `odo_att_sign=[-1,1,1]` /
  `odo_rate_sign=[-1,-1,1]` are correct on the real sim. (The lateral loop also self-corrected drift back to
  the start point: no positive-feedback sign error.)

## The lone issue: the alt loop is a relay limit cycle (CONTRADICTS the twin's clean-hold premise)
- True vz oscillates ±0.5 m/s at **~6 Hz**, thrust bang-bangs against BOTH clips (0.05 and 0.40). The altitude
  excursion is small (~0.1 m p-p) because the oscillation is fast; it's a **velocity** limit cycle.
- **Cause = `kp_alt=4.0` is far too stiff for the live plant** + the loop's true latency. At 1.39 m the alt
  demand is 0.2656 + 4·0.11 ≈ 0.71 → saturates the 0.40 ceiling; with the real loop delay this becomes a
  relay (bang-bang) oscillation. The 0.40 cap I added for the takeoff REDUCED the amplitude (a smaller relay
  swing); it is **not** the cause — kp_alt slamming both rails is. Holds ~0.1 m below the 1.5 m target because
  the demand saturates (can't servo the last 0.1 m).
- **Do NOT read hover from the mean thrust (~0.32).** That is a clip-duty-cycle artifact of the asymmetric
  relay (66% at the 0.40 ceiling), not the hover thrust. The open-loop vertical probe (`VPROBE`) de-confounds
  this with a clean hover + thrust-slope at the operating point.

## Verdict & next step
Rung-1 **STOP** (as designed): altitude holds but not cleanly (vz ≠ 0) — the alt loop needs an **offline
re-tune on the twin**, NOT a live fight. The ideal twin holds *tautologically* (its hover ≡ the controller's,
clean vz), so it cannot reproduce this; the offline loop must first add the missing live vertical dynamics —
**(a)** the real hover + thrust-slope (from `VPROBE`), and **(b)** the loop latency implied by the **~0.16–0.20 s
(~6 Hz) limit-cycle period** — to the twin's vertical channel, THEN re-tune `kp_alt`/`kd_alt` (much lower
`kp_alt`) until it holds. Do not proceed to gate 0 until the alt loop holds.

**Recordings (compact extracts):** `rung1_hover_run1_extract.json`, `rung1_hover_run2_extract.json`
(commanded thrust/body-rate + given pos/att, TRUE LOCAL_POSITION_NED vz, ACTUATOR motor witness; downsampled
+ summary). Reproduced ×2.
