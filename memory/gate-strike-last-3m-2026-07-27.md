---
name: gate-strike-last-3m-2026-07-27
description: "SSOT for the gate-SIDE strike with healthy vision (the pilot's standing target) — the lateral loop is proven sound to 3 m, the tail is created inside the last 3 m, the aperture yardstick was wrong, and the reward never priced lateral separately from vertical."
metadata: 
  node_type: memory
  type: project
  originSessionId: b85130f9-ac88-4db2-8c68-0e28b966cf80
  modified: 2026-07-27T04:42:17.119Z
---

# The gate-SIDE strike — 2026-07-27

**Fengyou's standing target, restated by him after the commander drifted off it:** *"the drone flying
into the side of the gate even though the vision was fine — that was the goal the entire time."* Not the
invisible obstacles (he can see those live → [[failure-census-2026-07-27]]), not the launch window.

Instruments committed on branch `lateral-capture-refute-2026-07-27`: `a2_null.py` (the reusable
null model) and `a7_transverse_guard.py` (the seam guard).

## 🛑 THE COMMANDER'S "8→5 m CAPTURE GAP" IS A SELECTION ARTIFACT — WITHDRAWN

Claimed: passes cut lateral error 62% between 8 m and 5 m while deaths cut only 25%, therefore failures
under-correct in a narrow capture window.

**Refuted.** A **pooled linear-Gaussian null** — one control law, no differential behaviour, labels never
read, calibrated to p99 — **reproduces the whole table** (PASSED −55%, DIED −22%) and **out-separates
the real data at 3 m at every guard cut**. The clincher is a conditional-independence test: a real
defect must show up early, so β(|lat@8 m| given |lat@3 m|) has to be positive; observed is
**−0.37 to −0.51**, the same sign as the null, at every cut.

🛑 **The roll sub-claim dies with it.** Scored as the quantity roll produces, the null matches every bin
to two significant figures — **including the negative "they roll the wrong way" ratio in the 0.5–1.0 m
band (null −0.19, observed −0.18)**. That appeared in a model **with no roll axis in it**.

🚩 **Outcome geometry alone is tautological**: a flight off-centre at 5 m dies *because* it is off-centre.
Only a difference in the **control at matched state**, or a sim/wire contrast, can carry the claim.

## 🟢 WHAT IS TRUE, AND IT RELOCATES THE TARGET

* **The lateral loop WORKS.** Slope |lat@8 m| → |lat@2 m| = **0.000 ± 0.03**. Terminal |lat| regressed on
  *all* 4–8 m state gives **R² = 0.004**. Where the drone is on approach tells you **nothing** about
  where it ends up — the loop fully rejects the entry error.
* ⇒ **THE TAIL IS CREATED INSIDE THE LAST ~3 m.** Not the approach, not the entry, not the turn.
* **Lateral exposure is 4.5× vertical**, concentrated at **gate 3** (23% outside 0.75 m vs 9% / 6%).

## 🛑🛑 0.75 m IS THE WRONG YARDSTICK — EVERY MARGIN NUMBER BEFORE THIS IS OPTIMISTIC

Training threads `0.75 − body_radius` with **r ~ U[0.28, 0.38]** ⇒ the effective half-width is
**0.37–0.47 m**, not 0.75 m. Against 0.45 m, wire lateral exposure is **~30% of ALL approaches** and
**41% at gate 3**. Re-measure any clearance claim against ~0.45 m.

## 🛑 NO REWARD TERM ISOLATES LATERAL OFFSET (v19/v20) — verified in code by the commander

* The crossing reward takes **`cross_offset = pass_linf`** — **L-infinity, `max(|lateral|, |vertical|)`**.
  Lateral and vertical are priced **identically**.
* `rw_centering` / `rw_corridor` price **3-D perpendicular** distance to the racing line, not lateral.
* **`rw_passage`** — the only literally-lateral formula — is **hard-zeroed by `rw_parabola_crossing`**.

⇒ **Priced symmetrically, failing 4.5× asymmetrically.** This is the first *training-side* defect this
cycle that survived contact, and it is the leading candidate for a v2.1 arm.

## 🚩 THE SIM-vs-WIRE EXPERIMENT IS NOT RUNNABLE AS FRAMED

The commander's decisive experiment ("does the same policy capture laterally in sim as on the wire?")
cannot be run as specified: the **sim course is RE-SAMPLED EVERY RESET** (8 gates, spacing U[10,20] m,
heading ±60°), so **there is no gate 1/2/3 to compare against**; there is no per-step emitter; and the
v19/v20 weights are cluster-only (actor-only locally). Any sim/wire contrast needs a fixed-course
harness built first.

## 🚩 SEAM-GUARD CORRECTION (the commander's spec was wrong, twice)

Gating on **total apparent speed is the wrong channel**: at 30 Hz, σ_r = 0.12ρ makes *range flap* read
~30 m/s from noise alone, so a 20 m/s total gate flagged **7.2% of ticks / 77% of approaches** — nothing
like the true 1.57%. **A re-lock moves the BEARING** (transverse rms 0.07 m vs radial 0.43 m). Gating
**transverse** gives **1.77% at 12 m/s**, matching the independent count. See also the *distance* form
(>1.5 m) in [[velocity-channel-2026-07-27]], which beat both speed gates.
🚩 **Never TRUNCATE an approach at the jump** — that biases range coverage (keeps far ticks, drops near
ones). **Drop the whole approach.**
🟢 Deaths re-lock only mildly more than passes (D−P = +0.2/+7.3/+12.4/+4.6/−1.0 pts), and swept 3→20 m/s
the capture effect was robust — **the seam is not what killed the finding; the null is.**

## Where this leaves the target

Eliminated: vision · the lateral-velocity channel · actuation authority · the turn out of the previous
gate · the approach itself · the launch window · course-geometry OOD · staleness as a cause.
Remaining: **lateral, inside the last 3 m, on a reward that never priced lateral separately** — and
**we have never observed that interval**, because the median at-gate log ends **1.43 m short**. That is
exactly what the post-impact recorder (`c0d3f6ef`) exists to close.
