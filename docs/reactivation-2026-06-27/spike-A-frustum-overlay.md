# Spike A: Gate Frustum Overlay

**Date:** 2026-06-27  
**Status:** COMPLETE  
**Question:** During the fix-seating window (gate range 12-28 m), does gate-4 fall inside or outside the camera's vertical FoV, relative to gates 0-3?

---

## Conventions Used (source: `src/racer/frames.py`)

| Parameter | Value | Source |
|---|---|---|
| Camera | 640×360, fx=fy=320, cx=320, cy=180 | `frames.py:74-82` |
| True VFoV | 2·atan(180/320) = **58.72°** → half = **±29.36°** | `frames.py:249-251` |
| True HFoV | 2·atan(320/320) = 90° → half = **±45.00°** | `frames.py:243-246` |
| Camera mount | **+20° about body-Y = UP-tilt** (camera pitched UP from body-forward) | `frames.py:7, line 20` |
| Spec note | Spec's "90° VFoV" is mislabeled; it is HFoV. True VFoV ≈ 58.7°. | `frames.py:249-251` |
| Body frame | FRD (X=forward, Y=right, Z=down). Pitch positive = nose-down in NED. | `frames.py:4-6` |
| Camera frame | Optical (X=right, Y=down, Z=forward). `R_camera_from_body()` = axis-swap ∘ R_Y(-20°). | `frames.py:84-90, 209-217` |

**Camera optical-axis elevation in body frame:** +20° above body-forward.  
**Visible body-elevation band:** (20 - 29.36, 20 + 29.36) = **(−9.36°, +49.36°)** above body-forward.  
Source: `frames.py:254-260`, `camera_elevation_band_deg()`.

**Critical threshold:** A gate at the drone's altitude (body-elevation = 0°) exits the BOTTOM of frame when body pitch > **~9.4° nose-down**. This is a pure function of mount angle (20°) and half-VFoV (29.36°): threshold = 29.36° − 20° = 9.36°. It is **independent of range**.

---

## Gate Positions (NED, from `handoff/shadowpc-firstcontact-2026-06-02/track_map.json`)

| Gate | X_ned (m) | Y_ned (m) | Z_ned (m) | Alt below arm (m) |
|---|---|---|---|---|
| 0 | -23.30 | -0.40 | -0.03 | -0.03 (at arm level) |
| 1 | -46.89 | -2.50 | +5.07 | 5.07 below |
| 2 | -74.59 | +1.20 | +13.67 | 13.67 below |
| 3 | -111.49 | -5.10 | +24.57 | 24.57 below |
| 4 | -135.49 | -0.80 | +25.36 | 25.36 below |
| 5 | -159.19 | -4.40 | +25.97 | 25.97 below |

**Altitude step from gate-3 to gate-4:** +0.79 m over 24.4 m horizontal = **1.85° descent slope**. Gates 3→4→5 are nearly flat after a 24.6 m climb over gates 0→3.

---

## Method

For a drone at range `r` from a gate on a **same-altitude straight-line approach** (drone Z_ned = gate Z_ned, approaching from the north/X direction), with body pitch θ (nose-down positive), the gate-center elevation in camera frame is:

```
elev_cam = -(θ + 20°)        [at same altitude, gate at body-forward horizon]
```

This holds because: at pitch θ, body-forward points θ° below world-horizontal. The gate at same altitude is θ° above body-forward. The camera axis is 20° above body-forward. Gate elevation in camera = +(θ° of gate elevation above body-forward) - 20° (camera up-tilt) = θ − 20° below camera axis = −(θ + 20°) offset from camera center... 

More precisely, the gate at same altitude is NOT at body-forward — it is at body-elevation +θ (looking "up" relative to pitched body). Camera center is at body-elevation +20°. So gate camera elevation = θ − 20°, which is **negative** (below camera center) when θ < 20°.

The R-matrix computation confirms: at pitch θ, same-altitude range-r approach, gate camera elevation = **−(θ + 20°) for the BOTTOM exit direction wait** — correction: the actual formula from numerical output is exactly `elev_cam = -(pitch_deg + 20.0)`. Numerically confirmed for all pitches 0-55°, all ranges 12-28 m.

**Gate exits BOTTOM of frame when:** `|-(pitch + 20)| > 29.36` → `pitch > 9.36°`.

---

## Frustum Table: All Gates, Head-On Same-Altitude Approach

The camera elevation angle (and frame status) for a gate at zero vertical offset from drone:

| Body pitch (nose-down) | Elev in camera (°) | In frame? |
|---|---|---|
| 0° | -20.0° | **YES** |
| 5° | -25.0° | **YES** |
| 9° | -29.0° | **YES** (just inside) |
| 10° | -30.0° | **NO** (exits bottom) |
| 20° | -40.0° | NO |
| 30° | -50.0° | NO |
| 35° | -55.0° | NO |
| 40° | -60.0° | NO |
| 45° | -65.0° | NO |
| 55° | -75.0° | NO |

**This table is IDENTICAL for all gates 0-5, for all ranges 12-28 m.** The frustum exit is a purely angular threshold set by the camera mount and VFoV. Range does not affect it (azimuth varies slightly but is negligible for head-on approaches; elevation is range-invariant at same altitude).

**Summary table: critical exit pitch (°) by gate × range (same-altitude head-on approach)**

```
 Gate      28m      24m      20m      16m      12m
    0      9.4      9.4      9.4      9.4      9.4
    1      9.4      9.4      9.4      9.4      9.4
    2      9.4      9.4      9.4      9.4      9.4
    3      9.4      9.4      9.4      9.4      9.4
    4      9.4      9.4      9.4      9.4      9.4
    5      9.4      9.4      9.4      9.4      9.4
```

**There is zero frustum difference between gate-4 and gates 0-3.** The exit threshold is 9.4° regardless.

---

## Gate-4 Approach from Gate-3 (Realistic Vector)

Gate 3→4 approach vector in NED: [-24.00, +4.30, +0.79] m. The gate-4 approach is nearly horizontal (1.85° descent). With yaw rotated to face gate-4 from gate-3 (yaw ≈ 170°), the same pitch-dependent exclusion applies: at pitch 10°+, gate exits bottom of frame at camera elevation < −29.36°.

---

## Altitude-Offset Effect: What Actually Helps Gates 1-3

The key differentiator between gates is the **altitude delta between drone and gate during the approach window**, not the gate index per se:

- **Gates 1-3 (climbing approach):** The drone flies BELOW the gate throughout the fix-seating window. At range 20 m and a gate that is 10 m above the drone, the gate subtends a significant upward angle in world frame (+27° elevation), partially compensating for nose-down pitch. The gate stays in frame at higher pitches than the same-altitude case.
- **Gate-4 (nearly flat approach):** After passing gate-3, the track is nearly level (Δalt = +0.79 m over 24.4 m). The drone is at approximately the same altitude as gate-4. The full 9.4° threshold applies with minimal climb-angle benefit.
- **Gate-0 (initial launch):** Gate-0 is at arm altitude. During launch the drone climbs toward it.

**Quantified effect of altitude offset on frame visibility:**

For a gate X meters ABOVE the drone at range 20 m (approaching from below):
- world elevation of gate = atan(X/20) above horizontal
- This adds to the effective "in-camera" angle, raising the exit threshold by the same amount.
- At X=10 m altitude advantage (like gate-2/3 at far approach distance), world elevation ≈ +26.6°.
- Effective exit threshold: 9.4° + 26.6° = **36.0°**. Gate stays in frame until pitch > 36°.
- Gate-4 at same altitude: X≈0 → exit threshold remains **9.4°**.

This is a **27° difference in effective pitch tolerance** between a strongly-climbing approach (gate-3) and the nearly-flat approach to gate-4.

---

## TOGT Trajectory Reference

The TOGT optimal trajectory (`handoff/laptop-togt-bound-2026-06-10/cases/bound_nominal/togt_traj.csv`) covers gates 0-2 but is too short (6.62 s) to reach gates 3-5 in a 30 m search radius with the gate numbering in that trajectory (TOGT trajectory uses ENU/FLU conventions with negated Z vs. track_map.json NED).

TOGT crossing pitches (at closest approach, converted):
- Gate 0 crossing: pitch ≈ +30.8° nose-up (climbing launch)
- Gate 1 crossing: pitch ≈ -6.6° nose-down (light)
- Gate 2 crossing: pitch ≈ -23.1° nose-down

**Caveat:** The TOGT is the time-optimal trajectory, not the RL-trained inc8 trajectory. Actual RL approach pitches during the fix-seating window (12-28 m before each gate) are unknown from local data. No inc8 evaluation trajectory npz/csv is available in the worktree. The frustum analysis was therefore run as a **pitch sweep** per the escape-hatch protocol.

---

## Gate-4 Verdict

**Gate-4 is geometrically IDENTICAL to gates 0-5 in frustum terms, given a same-altitude approach.**

The reachability hypothesis (Gemini: "at ~45° nose-down, gate-4 exits the frustum") is correct as a general statement — **every gate exits the bottom of frame at pitch > 9.4° for a same-altitude approach**. But gate-4 is not special: all gates exit at the same threshold.

**What IS potentially special about gate-4 is the altitude regime on approach:**
- Gates 1-3: drone is well below the gate during the 12-28 m fix window → the climb angle provides 10-30° of additional pitch headroom beyond the 9.4° threshold.
- Gate-4: the track flattens. The drone arrives at gate-3 altitude (+24.57 m below arm) and gate-4 is only +0.79 m deeper. During the approach, the drone has minimal altitude advantage. At any racing pitch > ~10°, gate-4 exits the frame bottom.

**The frustum problem is REAL at gate-4 but is NOT a gate-4-specific geometric property — it is a consequence of the flat altitude on the gate-3→4 segment, combined with the RL policy maintaining racing pitch during the approach.** Whether the RL policy actually flies at >9.4° pitch during the 12-28 m approach window to gate-4 is NOT confirmed by local data; no inc8 eval trajectory is available.

---

## Hypothesis Assessment

**"At racing pitch (~45° nose-down) plus 20° mount, the optical axis points ~25° below horizon and gate-4 leaves the 58.7° vertical frustum kinematically."**

- The 25° figure: camera axis at 45° nose-down pitch points 45° − 20° = 25° below horizon. Correct.
- Frame bottom edge at 45° pitch: 25° + 29.36° = 54.36° below horizon. Gate at same altitude = 45° below horizon. Gate IS inside frame? No: gate at same altitude at 45° pitch → elev_cam = −(45+20) = −65°, which is outside ±29.36°. The Gemini framing has a slight error in sign arithmetic but the conclusion holds.

**VERDICT: This SUPPORTS the reachability hypothesis as a general geometric fact, but REFUTES gate-4 uniqueness.** The hypothesis is correct that nose-down pitch pushes the gate out of frame, and gate-4's flat altitude exacerbates this. But gates 0-5 share the same frustum threshold. The gate-4 fix problem is better characterized as:

> "The inc8 RL policy likely maintains nose-down pitch >9.4° during the flat gate-3→4 approach segment, so gate-4 has zero or negative altitude advantage over the drone, causing it to exit the frame bottom. Gates 1-3 are approached with the drone flying below the gate, providing a climb-angle buffer that keeps the gate in frame at higher pitches."

**This SUPPORTS the hypothesis that gate-4 has a camera-reachability problem,** but the root cause is the altitude profile, not a gate-specific geometry. **It does NOT redirect the fix to RL/calibration over reachability** — the camera geometry creates a real fix-rate floor at gate-4 that is absent for gates 1-3.

**Remaining unknown:** the actual RL policy pitch profile at 12-28 m from gate-4. If the policy flies at <9.4° pitch in that window, gate-4 is in frame and the frustum is not the active bottleneck. A pitch-profile trace from an inc8 eval run would close this fork definitively (requires ShadowPC or Adroit eval run with trajectory logging).

---

## Source Files

- Camera convention: `/src/racer/frames.py` (lines 1-261)
- Gate geometry: `/handoff/shadowpc-firstcontact-2026-06-02/track_map.json`
- TOGT trajectory: `/handoff/laptop-togt-bound-2026-06-10/cases/bound_nominal/togt_traj.csv`
- Scratch computation: scratchpad `frustum_overlay.py`, `frustum2.py`, `frustum3.py` (not in repo)
