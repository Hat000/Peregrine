# 09 — Open Questions for the Oracle (ranked)

The core ask: **how do we get a single egocentric gate from ~16% (realistic-regime) thread to ≥90%
for the deployed deterministic policy, zero contact?** Broken into the questions we most want answered.

## Q1 — Reach rate (the biggest, least-explored lever)

~46% of drones never reach the gate (crash / floor / timeout) in the realistic regime. **Why, and what
fixes it?** Specifically:
- Is it dominated by floor-dives, out-of-bounds, or timeouts/stalls? (We have not split this — do it
  first.)
- Is the persistent **vertical high-bias** the *cause* of low-gate floor-dives (can't push down enough
  when the gate is below spawn)?
- What reward/curriculum change lifts arrival without re-triggering the warm-start collapse (L1)? Is a
  **two-stage** "reach-first, then centre" curriculum the right structure?

## Q2 — The 0.75 m-reward vs 0.42 m-body-aperture tension

The reward geometry centres to the 0.75 m aperture, but the **body-effective clean-pass is ~0.42 m**,
and the anneal end must stay ≥ 0.75 m or it punishes valid edge-threads (L7). So the reward can't
simply target 0.42 m. **How should the reward be shaped to drive the *centre-of-mass* crossing inside
0.42 m without penalising geometrically-valid threads near the 0.75 m edge?** (Body-inflated pass
bonus? A crossing bowl whose *shape* — not just zero-radius — has a strong near-centre gradient, e.g.
Gaussian? A separate "arrive-head-on / low-lateral-velocity" term?)

## Q3 — Warm-start fragility (L1) — is it inherent or fixable?

Essentially every discrete reward-semantic change to a warm-started policy floor-dives; only
warm-from-champion or continuous in-run anneals survive, and re-warming an *anneal-produced* policy
seems even more fragile. **Is this a fundamental property of narrow racing attractors, or an artifact of
our PPO setup (critic_warmup too short, no trust-region/KL constraint, value scale shift)?** If fixable,
what changes make warm-start fine-tuning robust — so we can iterate levers without starting fresh each
time?

## Q4 — Deployment fidelity / DR calibration

DR-off shifts the policy +3.2 m high (leans on modeled aero + trained latency). We don't know how the
**real competition sim** differs from our nominal `peregrine_plant`. **Which evaluation regime should we
trust as "deployment", and is the policy's DR-dependence a real risk?** Concretely: should we (a) match
DR to the real sim (needs its dynamics), (b) train for robustness *including* nominal (so DR-off doesn't
break it), or (c) is DR-off simply irrelevant because the real sim has aero+latency? What's the right
**nominal-dynamics eval gate**?

## Q5 — The centring floor: control-limited or reward-limited?

The 6000-upd anneal (`vglpan6`) landed identically to 4000 (~0.9 m / 20%) → **not convergence-limited**.
Is ~0.9 m a **control-precision floor** (the plant + CTBR + vision-noise can't do better at these
speeds), a **reward-shape limit** (the parabola is flat near centre), or an **exploration-noise floor**
(deterministic mean can't sharpen below the held std, though the gap looked modest)? Each implies a
different fix (slow down / better perception vs sharper reward vs lower noise floor). **How would you
decide which, and what's the fix for the dominant one?**

## Q6 — Is the egocentric obs *sufficient* to thread to 0.42 m?

The actor sees body-frame `rel_pos` (metric, from a noised estimator), velocity, attitude, rates,
`visible_area`, and a coarse sector — **no world position, no gate normal**. **Is this observation
information-sufficient for sub-0.42 m centring at approach speed, or is there a missing observable**
(e.g. a better head-on/normal cue, a range-rate, a plane-time-to-contact) whose absence caps precision?
If a cue is missing, which one, and can it be estimated from what the wire provides (30 Hz camera +
accel/gyro, no mag/baro)?

## Q7 — What's the right *global* strategy?

Given all of the above — is incremental reward-shaping on this stack the right path to 90%, or does
this evidence argue for a **structurally different** approach (e.g. a different action space, a
predictive/model-based inner loop, an explicit two-phase reach-then-centre policy, a different network,
imitation from an optimal-control expert on the racing line, a curriculum that grows the aperture in
*time-to-gate* rather than metres)? **If you were to bet on one plan to reach 90%+ single-gate, what is
it, in order of moves?**

---

## What we would most like handed back

1. A ranked, concrete **action plan** (which knob/curriculum/architecture change, in what order, with
   the expected failure mode of each so we can detect it early).
2. A **falsifiable explanation** of the warm-start collapse (L1) and the reach-rate loss (Q1).
3. The **right deployment-eval definition** (Q4) so we stop optimising a number the drone won't see.
4. A verdict on **Q5/Q6**: is 90% reachable on this obs+plant at all, or does something structural have
   to change first?
