---
name: feedback-vq1-deprecated-track-agnostic
description: VQ1 is DEPRECATED — never train/test/optimize against the VQ1 track; all pre-VQ2 work must be track-agnostic
metadata:
  type: feedback
---

Fengyou, 2026-06-28: "VQ1 is deprecated, forget anything about VQ1, we should not be testing against it." CONFIRMED OPERATING RULE: VQ1 allowed ONLY as a throwaway code-path smoke fixture (prove a tool RUNS), NEVER as a performance / training / optimization target.

**Why:** VQ2 has a different (and possibly per-load-randomized) track; any policy or metric optimized against the VQ1 gate layout won't transfer and actively misleads.

**How to apply:**
1. **TRACK vs PLANT distinction (load-bearing):** the **TRACK** (gate layout) is VQ1-deprecated. The **PLANT** (drone dynamics) is the SAME drone across VQ1/VQ2 — so plant-level work on the diffaero plant (distillation/system-id, AHRS-on-IMU, control tuning) is LEGITIMATE and track-agnostic.
2. **The diffaero TWIN uses the VQ1 track** → therefore NO track-specific RL training/eval until VQ2 drops. Pre-VQ2 RL must train on **RANDOMIZED / synthetic layouts** (layout-DR generalist), never the VQ1 track.
3. **Tools already built on the VQ1 substrate STAND** (they're track-agnostic): the planner tool, the optimal-line code, the distillation pipeline, the AHRS bench, the estimator, the load-day probes. Their VQ1 numbers (e.g. the 4.62 s line) are DRY-RUN validations, NOT targets — the tools take VQ2 inputs on drop.
4. **Everything new must be track-agnostic by construction.** When a task needs a track substrate, use randomized/synthetic layouts, not VQ1. Links: [[feedback-divergent-research-stance]] · [[project_vq2_stack_research]].
