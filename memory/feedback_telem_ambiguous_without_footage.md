# Feedback: Telem alone is ambiguous on close-range outcomes
Type: feedback
Date: 2026-06-29 (A15 reinforcement 2026-06-30)

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

## A15 reinforcement (2026-06-30) — the trap has TWO deeper layers
Gen-7 commander walked straight into this on A15, TWICE in a row, and had to be corrected by Fengyou:
1. **The estimate is FICTION on the denied wire — never read motion from it.** On VQ2 there is NO position/velocity/attitude truth (state-denied wire). `nav_estimate.jsonl` `position_ned` is a pure DEAD-RECKONED integral. The commander read the A15 estimate's "climb + forward" and the commanded thrust trace as "the drone sat in the start gate / bang-banged thrust / never lifted off" — a confident physical conclusion built on a fictional position. WRONG: motion was never observable this run.
2. **Footage can itself be CORRUPTED — then you have NO ground truth, so manufacture none.** The A15 onboard camera FROZE for the ~2.9 s YOLO first-inference warmup and dropped ~19% of frames right over the launch window. Fengyou (watching that same frozen stream) initially reported "didn't move," then RETRACTED it: the launch was never observable. The correct state was "UNKNOWN whether it lifted off," not a guess dressed up with the estimate. When the ONE ground-truth channel is degraded, the honest output is "we cannot tell yet" + a fix that RESTORES observation (here: pre-warm the detector so the launch view survives) — not a diagnosis.

**Meta-lesson (logged hard):** two over-confident wrong reads in a row from unreliable data. The discipline: (a) on a denied wire, estimator-derived quantities are NOT measurements — anchor only to raw IMU / actuator / camera; (b) if the footage is degraded, fixing OBSERVABILITY is the prerequisite step, and it OUTRANKS any behavioral fix you're tempted to ship on the corrupted look; (c) state UNKNOWN as UNKNOWN.

## Reinforces
→ [[feedback_offline_harness_realism]] (dry-runs can miss live failure modes)
→ [[feedback_slow_is_smooth]] (the slow-lap bring-up this keeps biting)

## Note for COMMANDER.md
This lesson belongs in COMMANDER.md on the next handoff under a "close-range outcome diagnosis" banner: telem alone can mislead — footage is the ground truth for near-contact events.
