---
name: feedback-walking-skeleton-no-vq1-crutches
description: VQ1 must run the FULL VQ2 stack under-tuned — never a simpler crutch (no hardcoded waypoints; vision + KF always in-loop).
metadata: 
  node_type: memory
  type: feedback
  originSessionId: ac1fef09-8e82-4891-a523-1597e2cd51d5
---

The VQ1 system must BE the full VQ2 system operating in an under-tuned / lower-performance envelope — a "walking skeleton" — not a simpler, task-specific variant. Concretely: the vision pipeline AND the Kalman filter must be ACTIVE and driving the vehicle in VQ1, even when the desaturated/aided VQ1 environment would let a "blind" hardcoded-waypoint run succeed.

**Why:** VQ1-only crutches (hardcoded waypoint following without perception; bypassing the estimator because VQ1 is visually easy) incur massive technical debt for VQ2 and mean a VQ1 pass de-risks nothing for VQ2. A walking skeleton makes every VQ1 run a genuine end-to-end test of the VQ2 architecture, exercised on the real sim.

**How to apply:** This SHARPENS the [[project-master-plan]] "position easy-mode" item. Still run `control_mode_probe` at first contact to characterize the control interface (does it honor SET_POSITION_TARGET?), but the DEPLOYED VQ1 system computes its setpoints from the live vision→PnP→KF estimate, NOT from hardcoded or ground-truth waypoints. The frozen data contracts are the integration seam, so VQ1→VQ2 is a tuning/swap change inside a fixed skeleton, never a rewrite. Source: external code-review feedback (2026-05-29) — the same pass that flagged and got 6 stack anomalies fixed on `main` (sim_time_ns clock jitter, IPPE prior blind-trust, 4-point-PnP too strict for clipped gates at transit, missing attitude-error process noise, JPEG frame-eviction leak, smoke_video infinite-block) and added the `ultralytics` `[detector]` extra.
