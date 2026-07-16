# Feedback: Offline Harness Realism

**Type:** craft lesson
**Date:** 2026-06-29
**Source:** VQ2 slow-lap attempt 1 failure (gate_seeker dry-run PASS → live flight 0/6 gates)

## Lesson

An offline dry-run / test harness MUST model the REAL deployment conditions, especially the adversarial ones — not a friendly synthetic.

The gate_seeker dry-run PASSED (threaded 3 gates, 74 vision fixes, self-loc bounded 0.27 m).
The live flight FAILED 0/6 because the dry-run used:
- a CONSISTENT synthetic map (3 gates placed sensibly)
- an estimator that converged cleanly because the carrot started in the right direction
- no stale-map fallback path exercised

Whereas live VQ2 had:
- NO map on the wire (VQ2 blocks all position telemetry)
- a stale VQ1 map as fallback (gate0 at world (-23.3,-0.4,0))
- an origin-seeded estimator (pose 0,0,0 yaw 0) with no vision fix yet
- tick-1 guidance demanding ~180 deg yaw from the visible start gate

Result: drone spun away from the gate the camera was ALREADY SEEING, camera went dark, estimator never anchored.

## Why It Matters

A harness that omits the binding real-world condition gives false confidence and lets a fatal bug ship to the expensive (flight) stage. The dry-run was testing the happy path; the live flight hit the only adversarial path.

## How to Apply

When building any offline validator:
1. Ask: "What does the real environment DENY or get wrong that my synthetic provides for free?"
2. Explicitly model hostile deployment conditions: missing/blocked inputs, wrong/stale fallback data, cold-start/uninitialized state, the actual sensor denials.
3. Add a regression that the OLD code FAILS and the NEW code PASSES under those conditions.
4. For VQ2 specifically: dry-run must exercise the no-map + origin-seed + gate-visible-at-spawn scenario, not a placed-sensibly synthetic map.

## Note for COMMANDER.md

This craft lesson belongs in COMMANDER.md on the next handoff: "offline harness = adversarial, not friendly; model what the deployment denies."

## Links
- [[feedback_slow_is_smooth]]
- [[index-vision-estimator]] §VQ2 SLOW-LAP ATTEMPT 1 (FAILED 2026-06-29)
