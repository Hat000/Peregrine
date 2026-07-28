---
name: feedback-beat-dont-reproduce
description: "Fengyou: Track A must BEAT vpeffs0, not reproduce it — warm from its ckpt + add improvements; reproduction runs are the wrong target."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: c864e5bb-f53f-4b54-b78a-a2216bb2c906
---

**Fengyou 2026-07-13:** "it sounds like you're retraining vpeffs0, when in reality we're trying to beat it."

The goal is a policy BETTER than the champion vpeffs0 (no velocity runaway, no deploy walls, 20-gate), not a re-derivation of it. I had launched byte-exact reproduction runs (vta0/vta40) as a "tree-health anchor" — Fengyou flagged that as the wrong target and I canceled them.

**Why:** vpeffs0 is the hard-won base (the successful yaw + altitude hold). Spending compute re-making it is wasted; the value is in ADDING what it lacks.

**How to apply:** warm-start every beat-run FROM vpeffs0's existing ckpt (`/scratch/.../ego_dual_gate_fullstack_floor_pef_seed0_vpeffs0/checkpoints`) and each run must ADD a goal-improvement (annealed, preserving the core), not just re-train the champion. Reproduction/tree-health can be inferred from the beat-run's early behavior — no dedicated repro run. Full plan → [[track-a-beat-vpeffs0-2026-07-13]]. Pairs with [[feedback-preserve-yaw-altitude-hold]].
