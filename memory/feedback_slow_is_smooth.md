---
name: feedback-slow-is-smooth
description: Flight curriculum directive — close the loop SLOW first, then ramp speed. Slow is the bootstrap for self-localized VQ2 flight.
metadata:
  type: feedback
---

Fengyou 2026-06-29: "Slow is smooth, smooth is fast."

**FLIGHT CURRICULUM DIRECTIVE:** close the loop SLOW first, then incrementally ramp speed.

**Why:**
(a) De-risks the self-localizing loop — slow flights are easier to diagnose if the estimator drifts or the control mode misbehaves.
(b) TECHNICALLY easier — with VQ2's no-mag/no-baro wire, yaw+Z come from VISION, and vision-pinned estimation is far easier slow: more frames/meter, minimal motion blur, denser gate sightings, vanishing-point structure cleanly resolvable. Slow is the regime where self-localization actually works.

**How to apply:** the FIRST closed-loop milestone = a SLOW, zero-gate-contact, fully self-localized lap — NOT a fast time. Only after that is locked do we ramp speed via the RL substrate (time-optimal line, rw_line_progress, speed-profile reward, appo critic). Slow is the BOOTSTRAP, not the destination.

Links [[project_vq2_stack_research]] [[feedback-vq1-deprecated]].
