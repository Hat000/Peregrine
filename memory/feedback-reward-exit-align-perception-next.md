---
name: feedback-reward-exit-align-perception-next
description: "Fengyou: reward exit_align + perception_next to shape the next-gate approach; do NOT reward area (it's an obs input the policy learns as a confidence signal — a reward would fight the angled entry line)."
metadata:
  node_type: memory
  type: feedback
  originSessionId: c864e5bb-f53f-4b54-b78a-a2216bb2c906
---

**Fengyou 2026-07-13** (after the m8 rollout trace showed the coarse map points right 96–97% but the drone can't set up the next gate): the coarse map is richer than a "turn" signal — "the RL policy has enough to learn that entering a gate at an angle will allow it to continue." So the fix is to **reward the trajectory shaping**, not the map.

**Decision — turn ON two reward levers, leave a third OFF:**
- ✅ **`exit_align` (`rw_exit_align`)** — `exit_line_reward` fires ONCE per gate pass = `rw_exit_align · cos(exit velocity, curr→next-gate bearing)`. Rewards LEAVING gate k already headed at gate k+1 (sets up the next acquisition — directly attacks the accumulating cross-offset the trace found). Chosen weight **4.0** (~20% of the +20 crossing parabola: shapes the exit without overriding the centered-crossing incentive).
- ✅ **`perception_next` (`rw_perception_next`)** — Swift/Geles `exp(−δ⁴)` bonus on the NEXT gate's optical-axis angle (env-gated on the next gate being detectable), rewarding acquire/hold of the upcoming gate.
- 🛑 **Do NOT reward `area` (`area_dist` coupling stays OFF).** Fengyou: "rewarding area might not be best as it fights our entry line; as long as area is in the input state, it shall learn to use area as an indicator of how confident it is in flying into that gate." Area IS in the 21-dim obs (slot0 conf+area for the current gate, slot1 for the next). Rewarding square-on approach would pull against the angled entry `exit_align` wants. **Reward the OUTCOME (exit heading), let the policy OBSERVE area and decide when to square up.**

**Footgun — farm-neutrality assert (`EgoRewardWeights.__post_init__`):** with `perception_next > 0` the code asserts **`perception + perception_next ≤ rw_time` (0.02)**. `rw_perception` was already 0.02 = the ceiling, so you must **SPLIT** the perception budget, not add. Chosen split: `rw_perception=0.012` + `rw_perception_next=0.008` (=0.02, 60/40 favoring the load-bearing current gate). The PRECHECK runs `__post_init__` and fails fast if violated.

First test = **vtrackArs0** (m8b recipe + these, warm vtrackAw0) as a matched pair vs m8b to isolate the shaping delta. Pairs with [[track-a-beat-vpeffs0-2026-07-13]] (§reward design + §root-cause) · [[feedback-preserve-yaw-altitude-hold]] (perception is also the altitude/gate-in-view lever).

**🛑 OUTCOME (2026-07-13, vtrackArs0 12k): REFUTED at n=1 — the bundle REGRESSED vs m8b** (success 0.326→0.231, exit_frame 0.631→0.733, spin ~doubled). Knobs fired mechanically but rewarding exit-heading + holding k+1 did NOT improve acquisition. CAVEAT: rs0 bundled FOUR changes (exit_align + perception split + dither 0.5→0.125 + centering 0.4→0.3) on a single seed, so this is not a clean indictment of exit_align/perception_next specifically — it's "didn't help as a bundle, and m8b (no shaping) won." The **rationale above stays valid as Fengyou's reasoning**; the empirical verdict is "needs single-variable isolation + multi-seed to salvage" (DEFERRED under BANK-FIRST). Do NOT re-apply this bundle as a known-good lever. Full → [[track-a-beat-vpeffs0-2026-07-13]] §rs0.
