# 01 — Problem & Goal

## The competition

Anduril **AI Grand Prix**, an autonomous drone-racing challenge (May–Nov 2026). "VQ2" is the
second qualification round: track *"Now You See Me, Now You Don't"* — a dark warehouse with
glowing-red gates, stations 01–20, deterministic layout per load (not randomized). The drone must
fly the course autonomously from its own sensors and pass through each gate's aperture.

**Validity rule (hard):** any gate contact invalidates the run. Zero-contact is the constraint.
Speed matters (the north star is Swift, *Nature* 2023 — championship-fast, not just finishing), but
the current phase is a **slow, correct bring-up**: close the loop reliably first, ramp speed later.

## The specific problem this dossier is about

We are training the flight policy with reinforcement learning in a simulated twin. We adopted an
**egocentric, position-free** observation this generation (call it "inc9"): the policy sees only
what a drone could actually perceive from its own body frame — relative gate geometry from vision,
its own IMU-derived velocity/attitude/rates — and **no absolute world position**. (Rationale in
[02](02-system-and-stack.md); the short version is that the prior world-frame Kalman filter drifted
and laundered a position-immunity we never actually had.)

Before scaling to the full multi-gate course, we must prove the policy on the **simplest possible
sub-problem**: a **single gate**, placed at a spec-plausible distance and height, approached from a
standing start. If it cannot reliably thread one gate, multi-gate is hopeless.

### Definition of "thread" (success)

The drone crosses the gate plane (the plane x=0 in the gate's local frame, +x = the exit / down-course
direction) with its in-plane offset inside the aperture, **without contacting the frame**. Concretely
the environment classifies each forward plane crossing by the **L-infinity** off-centre distance
`Linf = max(|lateral|, |vertical|)` at the interpolated crossing point:

- `Linf < 0.75 m` → **thread / pass** (inside the 1.5 m inner opening; half-opening 0.75 m).
- `0.75 ≤ Linf ≤ 1.36 m` → **frame-clip** (hits the 2.72 m physical frame band; a contact = invalid).
- `Linf > 1.36 m` → **wide miss** (flies past the frame entirely; no contact but no pass).
- Never crossing the plane (crashes / floor / out-of-bounds / timeout first) → **no-reach**.

**Crucial subtlety we only nailed late (see [07]):** the drone is not a point. The body radius is
0.28–0.38 m, and the environment inflates the contact bands by it. So the **effective clean-pass
window is ~0.75 − 0.38 ≈ 0.42 m**, not 0.75 m. A crossing centred to 0.5–0.7 m still clips. The real
centring target is **sub-0.42 m**.

### The goal, stated precisely

> **≥90% single-gate thread rate**, over the training distribution of spawn distances/heights (and,
> as of this session, lateral bearings), for the **deployed (deterministic-mean)** policy, with zero
> gate contact — as the gate to begin multi-gate training.

Fengyou's words: *"we need a ~90 percent thread before I'm confident to move on to multigate."*

## Why this is hard (our current understanding)

1. **The aperture is tiny relative to control precision.** ~0.42 m effective, approached at speed
   from 8–15 m away, with vision-only relative sensing and IMU-integrated velocity.
2. **Sparse, cliff-like success signal.** A binary pass/clip/miss at the plane gives almost no
   gradient to *centre* — most reward-shaping work in this dossier is about manufacturing a dense,
   non-degenerate centring gradient without introducing a pathology.
3. **The policy is fragile.** Once a policy is warm-started and specialised, essentially any
   structural change to its reward collapses it into a floor-dive (see [06]).
4. **Two independent failure modes** (reach-rate ~46%, and centring against a ~0.42 m window) that a
   single reward lever cannot both fix.

## Scope notes

- Everything here is **simulation**. Sim flights carry no physical risk; they are authorised freely.
- Training runs on **Adroit** (Princeton SLURM cluster; GPU nodes have no internet, output to
  `/scratch`, SLURM-only compute). See [02] for the harness.
- The prior generations (VQ1 passed; inc7 is the last live-confirmed best; inc8 was a speed increment)
  are **not** the subject here — they are referenced only where a mechanism carries over.
