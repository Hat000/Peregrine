---
name: feedback-training-faithful-deploy
description: "Deploy recipe for a NEW policy lineage must match its TRAINING action space (pitch free, yaw 0.7); the champion's recipe reproduces the CHAMPION only. Also: the 4.0 flips/s sim gate + cmd_absmean transfer 1:1 to deploy — never override on normalized arguments."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: cd451162-1e4a-450b-987c-7c21ee7a5592
---

# Deploy recipe = training action space, not the champion's band-aids (Fengyou, 2026-07-18)

**What happened:** the v1 pick-flight box carried the champion ground-truth recipe verbatim — including `--ego-pitch-clamp 1.0` (a vpeffs0-era deploy band-aid; training had pitch FREE). Fengyou: "why are we flying with pitch clamps, isn't the whole point of these training to get rid of pitch clamp and still be able to fly without velocity runaway?" He is right — Track-A's goal is NO deploy walls; a new lineage must be flown in its training action space (yaw clamp 0.7 stays because 0.7 was IN training; pitch clamp does not, because training pitch was free). Champion recipe = for reproducing the champion, nothing else.

**Why:** (1) goal-level — the campaign exists to make deploy fences unnecessary; carrying them forward hides whether the goal was met. (2) mechanism-level — v1 trained 8-gate + vision-cadence learned to USE pitch for gate framing; freezing pitch at deploy removes a channel the policy relies on and forces yaw/roll compensation (plausibly amplifying the very hunt being tested).

**Also learned (same flights):** the sim YAW_EVAL transferred to the wire almost exactly (v1 sweep flips 6.3/6.0 + cmd_absmean 0.66 → deploy 4.4–6.8 + 0.60–0.67; champion sim 2.45/0.23 → deploy 2.2/0.15; champion sim n_passed 1.99 → deploy median 2). **The 4.0 flips/s hard gate was RIGHT; my per-gate-normalized override was WRONG as a deploy predictor — reinstate it, and ADD a cmd_absmean gate (~≤0.3; it separated champion from v1 by 4×, sharper than flips).** v1's n_passed 4.65 did NOT transfer (deploy ~1) — high sim throughput with hot yaw = sim-gap exploitation, not deployable skill.

**How to apply:** new-lineage pick-flights fly training-faithful flags (pitch free/unset, yaw 0.7, no governor, no floor). Selection gates: flips ≤4.0 HARD + cmd_absmean ≤~0.3 HARD, then rank n_passed. Champion-recipe flights only for champion reproduction. [[replay-ratchet-2026-07-17]] · [[feedback-no-deploy-bandaids]] · [[track-a-beat-vpeffs0-2026-07-13]]
