---
name: feedback-no-spin-hard-requirement
description: "Fengyou hard directive (2026-07-10) — the drone must NOT spin, ever, regardless of how the sim vision model behaves; enforce by construction in training, not by incentive shaping"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: c864e5bb-f53f-4b54-b78a-a2216bb2c906
---

**Fengyou (2026-07-10, verbatim intent):** "I don't want the system to be spinning at all. Even if it works in sim, it's not good. We should have it not spin, regardless of how the vision system behaves in the simulator."

Context: vn16 (real-noise champion) turned out to be a SPINNER — corkscrew gait, ~9.3 rad/s cruise, spin-scan vision (see [[ego-deploy-contract-2026-07-09]] §A2). My first fix plan led with perception-honesty (blur-gated detection + framing reward) to make spin *unprofitable*. Fengyou overruled the ranking: non-spin is a **hard behavioral requirement**, not an incentive to shape.

**Why:** a gait that only survives because the sim tolerates it is a sim artifact waiting to fail on the wire; 9 rad/s near gate frames = contact risk (contact = INVALID run); loop-rate-existential fragility (A2); estimator/detector degradation under spin. Also plain unacceptable behavior for the system regardless of scores.

**How to apply:**
- Training MUST make spin fatal-by-construction: tight all-axes spin aborts with collision-class terminal penalty (the old spin_rate_abort=10/3s was toothless) + hard realized-yaw clamp at command application. Perception honesty (blur model, rw_perception) stays but is SECONDARY.
- Judge every future policy/gait on this before DET numbers: a high-DET spinner is NOT a champion.
- Existing spinner ckpts (vn16 lineage, likely dgbf0f/vdff1 lineage) are not flight-worthy as final answers; flights of spinners only for diagnostics Fengyou explicitly wants.
- Related standing constraint: NO energy-optimal penalties either ([[ego-deploy-contract-2026-07-09]]) — anti-spin = threshold/abort form, not effort costs.
