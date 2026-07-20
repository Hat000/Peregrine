---
name: feedback-tails-not-medians
description: "Fengyou directive — judge flights by tail events (max/p99/worst-event), not medians/means; one strong roll kills a flight"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: cbe568b8-e0d0-4d37-9b22-0120c088aea4
---

**Fengyou (2026-07-19):** "if we keep looking at the median and mean, we miss where the damage happens. usually the max and min, the tails are what gets us and ruins a flight. one strong roll can kill a flight, even though the median is nice and stable."

**Why:** Flight outcomes are min-over-events, not average-over-time: a single tail event (one saturated roll, one wrong-gate tick, one blind-panic yaw spike) invalidates the run regardless of how quiet the other 99% was. The v16 wire-miner independently confirmed the structure: yaw median 0.08 vs p99 2.66 (the damage), wrong-gate locks = single-tick transients invisible in aggregates, pass-shake = a 4.61 gyro PEAK.

**How to apply:** Every flight/census adjudication leads with per-flight worst-event metrics — max/p99 per channel, tail-event counts (saturation bursts, track snaps, blind ticks), and a "what single event ended it" attribution — before any mean/median. Selection gates and A/B verdicts must include a tail column. Training-side: shape/penalize the TAIL (duty above threshold, jerk, abort-by-construction), which the lineage already does — keep it that way. Related: [[feedback-no-deploy-bandaids]], [[perception-sim2sim-gap-2026-07-15]].
