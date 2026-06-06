# Task 2 — gate-0 map-anchor verdict (laptop vision-PnP, 2026-06-06)

Ran `scripts/task2_gate_pnp.py` on ShadowPC's 40-frame course1 bundle (`task2_frames/`).
Detector = `gate_yolo11s_curriculum_v2.pt`; 38/40 frames detected gate 0, 26 confidently associated.

## Method (and two corrections made along the way)
Detect → 4 corners → gate origin in camera → world via the GIVEN drone pose + ODOMETRY attitude +
`R_camera_from_body` → offset vs map `(-23.3, -0.4, -0.03)`, per axis N/E/D.
- **Correction 1:** single-frame IPPE is 2-fold AMBIGUOUS on a near-frontal gate (the mirror flips
  the translation laterally). Fixed by solving **translation-only with the rotation FIXED** to the
  known gate orientation + given attitude (`solve_t_known_R`) — no ambiguity.
- **Correction 2:** all 6 gates share height+quaternion, so "highest-score" sometimes locked onto a
  background gate seen through gate 0 (same D, wrong N/E). Fixed by **associating to gate 0** = the
  detection whose solved centre is nearest the gate-0 map point (other gates are 4 m+ away → unbiased
  for the sub-metre question).

## OBSERVATION (linear fit of offset vs range over the 26 gate-0 frames)
| axis | intercept (true offset) | slope | as angle |
|---|---|---|---|
| N (approach) | +0.13 m | −0.023 m/m | −1.3° |
| E (cross-track / width) | −0.68 m | −0.063 m/m | −3.6° |
| D (vertical) | −1.18 m | +0.005 m/m | +0.3° (flat) |

Near-range (1.8–5 m, most reliable) horizontal offset = **0.50 m**; range-0 extrapolated = 0.69 m.
Vertical D is flat at −1.2 m across all ranges (MAD 0.14).

## INTERPRETATION / VERDICT
1. **z-anchor CONFIRMED:** map z is the gate **bottom edge**; true centre ~1.3 m above. The gate
   loader's z-only `corner_to_center` (lift ~1.36 m) is **correct** (we measure 1.2 m; ~0.16 m of
   over-lift, negligible vs the 0.75 m half-opening).
2. **NOT a 1.36 m bottom-corner — the bottom-left-CORNER hypothesis is REFUTED.** Horizontal offset
   is 0.5 m near / 0.7 m at range-0, nowhere near 1.36 m. **Do NOT add a half-width horizontal shift.**
   So gate-0 misses were the ODOMETRY frame bug + gains, NOT a mis-anchored map.
3. **NEW calibration finding:** the cross-track offset GROWS with range at **~−3.6°/range** — the
   fingerprint of a small **yaw bias** somewhere in {camera mount, given-attitude yaw, map gate yaw}.
   It puts a range-dependent lateral error into vision PnP fixes. **Irrelevant to the current
   model-based nav** (which flies the GIVEN pristine position, not vision), but a real **VQ2 target**
   to isolate before leaning on vision for lateral aiming at range.

## Caveat
Single-frame monocular PnP at range is noisy (scatter beyond ~10 m; depth ill-conditioned). The
verdict rests on the near-range frames + the range-decomposition, not the raw median (which the
yaw artifact inflates to a misleading 1.24 m). A cleaner sub-metre number would need multi-frame
triangulation or resolving the yaw bias first — not needed for the loader decision (not a corner).
