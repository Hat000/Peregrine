# Feedback: Telem alone is ambiguous on close-range outcomes
Type: feedback
Date: 2026-06-29

## Lesson
A telemetry contact-signature "range closes to small value → contact spike → estimator inverts" is AMBIGUOUS between:
- (a) CLIPPED the gate (didn't pass) — hit the structure on the approach side
- (b) PASSED then hit something just after (e.g. the backside)

Both failure modes produce a near-identical range-then-contact trace. Only the onboard FOOTAGE disambiguates which occurred.

## What went wrong
In A5 bring-up (2026-06-29), the commander observed the telem signature "range→2.78m→contact spike→estimator inversion" and committed to failure mode (a): concluded the drone CLIPPED the top bar and fixed vertical-align accordingly. The footage subsequently showed the drone PASSED through the gate opening cleanly, then hit the BACKSIDE wall after. The fix was applied to the wrong failure mode.

## How to apply
When diagnosing a close-range flight outcome:
1. Do NOT commit to a specific failure mode (clip vs pass-then-hit vs overshoot) from telem/range alone.
2. Get footage confirmation FIRST, or hold the conclusion as PROVISIONAL until footage is reviewed.
3. Only after footage confirms the failure mode, commit the fix.

This is exactly why Fengyou's footage-confirmation loop exists: predict behavior → user reviews footage → confirm/correct before acting.

## Reinforces
→ [[feedback_offline_harness_realism]] (dry-runs can miss live failure modes)

## Note for COMMANDER.md
This lesson belongs in COMMANDER.md on the next handoff under a "close-range outcome diagnosis" banner: telem alone can mislead — footage is the ground truth for near-contact events.
