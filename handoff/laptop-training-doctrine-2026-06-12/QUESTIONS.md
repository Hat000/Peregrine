# LAPTOP-TRAINING-DOCTRINE — Part 1: the question ledger (2026-06-12)

Every reasonable question current evidence raises about whether inc7 would train the RIGHT
THING. Each tagged: **[DATA]** answerable-now-from-data (answered by computation in
`scripts/doctrine_probes.py` + WRITEUP §2), **[PROBE]** needs a new measurement,
**[JUDGMENT]** design judgment. Seed questions Q1–Q8 from the session brief; Q9–Q14 are the
questions we hadn't asked.

## Seeded

- **Q1 [DATA]** Is the ~55° tilt / +51°-sideslip crab OPTIMAL in the measured plant or a PPO
  local optimum? Decide by computation: along-track drag of the observed orientation vs the
  drag-optimal rotation about the thrust axis (trajectory-preserving DOF), on the measured
  body-direction-dependent quad-drag table; plus the yaw-rate budget and reward cost a
  coordinated (zero-sideslip) alternative would pay.
- **Q2 [PROBE, bounded by DATA]** Crash forensics on the 4 standing gate-3 clips: saturated
  corrective commands losing a fight (authority gap) vs mid-range commands drifting passively
  (coverage hole)? Live `debug_obs` for std_f1–f4 NOT packaged locally → provisional; bounded
  offline by injecting the measured climb-bin residual (additive AND multiplicative, with live
  latency) and reading terminal command statistics.
- **Q3 [DATA]** Margin doctrine: per-gate pass-offset distribution under start-jitter × latency
  vs the margin a policy SHOULD hold so that any plant error within measurement bounds cannot
  produce contact. Is gate-edge proximity efficient or reckless given zero-contact validity?
- **Q4 [DATA + JUDGMENT]** DR doctrine: did inc6's DR really omit force-model scale (session
  premise)? What force-scale errors does the trained policy actually absorb? Should force DR be
  standing policy and at what width; what does it cost?
- **Q5 [DATA]** Recovery curriculum: does the policy recover from "1.5 m off-trajectory at gate
  approach" states the twin nominally never visits, or do we need state-perturbation injection?
- **Q6 [DATA]** Cold start: quantify what differs standing vs bridge (residual-bin exposure,
  regime traversal) and whether reset-distribution coverage is the deficit.
- **Q7 [JUDGMENT, fold-in only if binding]** Does 30 Hz interact with any of the above at VQ1
  speeds? (§SPEED-CEILING-ANALYTIC verdict stands unless contradicted.)
- **Q8 [DATA premise-check + JUDGMENT]** Single-track overfit (META gap ①): does inc7 stay
  VQ1-specialized or is this the moment for procedural tracks?

## Found (the unasked half)

- **Q9 [DATA]** Does the training env's gate-pass geometry match the MEASURED contact geometry?
  The env scores a point-mass L-inf < 0.75 m as a clean pass; the corner-pass probe (2026-06-11)
  logged a sim CONTACT event at 0.60 m offset, and the rules-valid aperture is ≈0.5 m. If the
  env over-promises aperture, every "margin" number we quote is inflated by the drone's body
  extent — and contact-on-passing is invisible to training.
- **Q10 [DATA]** Is the env's plane-band collision model blind to VOLUMETRIC frame strikes?
  Live std_f1 hit the frame ~0.5 m high AND ~0.5 m SHORT of the plane. A plane-crossing test
  cannot price approach-slope risk: a late-converging steep approach turns upstream displacement
  into frame strikes before any plane is crossed. Measure the trained style's approach-corridor
  geometry (offset vs distance-to-plane, slope) per gate.
- **Q11 [JUDGMENT]** Training obs are pristine truth; no estimation noise is injected. When does
  that become load-bearing (VQ2 case C vision-only), and is it an inc7 blocker?
- **Q12 [JUDGMENT]** Selection metrics: sr/lap-time/gen (gen seed-volatile) don't measure the
  transfer-relevant quantities (offset tails, corridor clearance, robustness-probe outcomes).
  What belongs in the checkpoint gauntlet?
- **Q13 [DATA]** Does the c16 corner tax suppress the corrective authority needed in the climb
  regime (the operator's "corners are the most efficient places" worry, inverted: does the tax
  bind where we need corrections)? Measure tax exposure along the lap.
- **Q14 [JUDGMENT]** Where must robustness come from, given we cannot penalize aggression
  (S17: blunt action damping trades away generalization)? Candidate sources: DR structure,
  contact-true geometry, reset/course diversity, reward shaping (in order of suspicion).

Answers with numbers: `WRITEUP.md` §2. Probe code: `scripts/doctrine_probes.py`
(subcommands q1, q2q6, q3, q4, q5, corridor).
