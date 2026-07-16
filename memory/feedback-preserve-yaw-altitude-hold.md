---
name: feedback-preserve-yaw-altitude-hold
description: "Fengyou: preserve the yaw→gate-in-view→altitude coupling (the crown jewel) — controls must not suppress functional gate-tracking yaw or camera-pointing pitch."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: c864e5bb-f53f-4b54-b78a-a2216bb2c906
---

**Fengyou 2026-07-13:** "make sure not to forget about yaw control and gate in view to support altitude control. we worked so hard for a successful yaw and altitude hold."

**Why (the mechanism):** the drone has NO absolute-altitude sensor (no baro; wire is accel+gyro only). Altitude is held ONLY by keeping the gate in view — the estimator gives gate-relative vertical, so the policy holds altitude by keeping the gate centered in its observable rel_pos. Yaw is what keeps the gate in view. So yaw control → gate-in-view → altitude hold is one coupled chain, and it was hard to get working.

**How to apply (constrains every Track A control):**
- **Anti-dither** (`rw_yaw_dither`) penalizes yaw-command JITTER (Δ post-clamp yaw action across steps) — high-freq reversals — NOT sustained gate-tracking yaw (a smooth, low-jerk command). Arm gentle + annealed; watch that DET-thread + altitude hold. If it suppresses functional yaw → gate lost → altitude lost → do NOT ship.
- **Attitude cap** is a LOOSE 60° backstop. Pitch's real failure lever is **rw_perception**, NOT the cap: the gate-losing head-down dive (~−50°) sits INSIDE any 60° cone (a camera-sees-floor perception loss), so keep the pitch cap loose and lean on perception.
- **Velocity cap** is safe here — it doesn't touch yaw/pitch and is zero below 9 m/s, so normal ~8 m/s yaw+altitude flight is untouched.

Pairs with [[feedback-beat-dont-reproduce]] · [[feedback-no-gt-actor-obs]] (the gate-relative-only obs is WHY altitude depends on the gate). Full → [[track-a-beat-vpeffs0-2026-07-13]].
