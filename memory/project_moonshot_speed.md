---
name: project-moonshot-speed
description: North-star strategy — championship-fast autonomous flight, not merely finishing. Shoot for the moon via slow bootstrap → perception hardening → RL speed ramp.
metadata:
  type: project
---

Fengyou 2026-06-29: "Shoot for the moon."

**TARGET:** championship-fast autonomous flight, not merely finishing.

**North star:** Swift (Kaufmann et al., Nature 2023) beat human champions, proving it is achievable. Also MonoRace/A2RL lineage (TU Delft, arXiv 2601.15222). These are existence proofs — our architecture is built for the same speed ramp.

**Our architecture advantage:** offline-exact distill+plan ⊕ online RL, vision-pinned case-C self-localization → the design already supports the speed escalation path.

**Strategy (three-phase):**
1. **Bootstrap slow:** close the loop slow first — ESKF+EqVIO stable, control handshake resolved, self-localized lap complete, zero gate contact. See [[feedback_slow_is_smooth]].
2. **Harden perception:** train learned detector on VQ2-matched Blender synthetic data to extend the detection envelope (far gates, oblique angles, motion blur). See [[project_blender_vq2_data_pipeline]].
3. **Ramp speed via RL:** with a locked self-localization loop, push pace via time-optimal reward, rw_line_progress, speed-profile shaping, appo asymmetric critic. Slow is the BOOTSTRAP; fast is the destination.

**Reminder:** VQ1 passing + inc8-appo training are not the goal — beating the leaderboard time is. Every architectural decision should be pressure-tested against "does this get us faster?"
