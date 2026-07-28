---
name: feedback-no-deploy-bandaids
description: "No deploy-side bandaid clamps (speed governor, pitch limit) — root-cause via the reward/asym-critic in training; the speed governor was rejected + removed."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: c864e5bb-f53f-4b54-b78a-a2216bb2c906
---

Fengyou 2026-07-14, on the deploy speed governor I built (commit 293ffee, `--ego-speed-gov`): "the speed gov you implemented does not work and that goes against our design direction. we want less bandaid fixes, like pitch limit or speed governor. it appears applying speed governor causes the drone to hit the gate sides." **REJECTED + being removed. He also demanded: everything traced to root cause, not bandaided.**

**Why:** A deploy-side clamp treats the symptom, not the cause, and fights the policy's own control authority. The governor cuts collective when observed horizontal speed > cap — but the drone NEEDS collective in the turn (exactly where speed peaks) to generate the lateral accel that holds its line → it drifts wide → clips the gate side; and the overridden collective puts the policy OOD. The correct design is ALREADY in training: `rw_v_cap` penalizes **GT/TRUE speed** (ego_reward.py:387), the **privileged/asymmetric critic** sees it, the policy learns to fly slow INTRINSICALLY. m8b (soft9) proves it transfers: train mean 7.66 → deploy ~8, no governor. There is NO "speed clamp" flag in training (only the soft reward penalty). The "flies slow in training, fast in sim" was NOT a train/deploy bug — it was the NO-CAP models (aw0/am8, `rw_v_cap=0`, never penalized) + the short 2-gate training course masking their top speed; give them runway and they run.

**How to apply:** Handle speed (and pitch/perception, floor descent) via TRAINING root-fixes — reward terms the asym critic sees, or curriculum — NOT deploy clamps. Every deploy fence (pitch/roll/floor clamp, hand-authored coarse map) is a bandaid whose root fix belongs in training; migrate them back (e.g. the coarse-map fix that kills the hand-authored-map crutch). Keep ONLY clamps that are part of the trained action space — the yaw-rate clamp, matched per-ckpt to its training value. Trace every symptom to root cause before adding any deploy-side override. Links: [[feedback-no-gt-actor-obs]] · [[feedback-no-spin-hard-requirement]] · [[trackA-stall-forensic-2026-07-13]].
