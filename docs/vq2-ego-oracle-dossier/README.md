# VQ2 Egocentric Single-Gate Thread-Rate — Oracle Dossier

**Purpose.** This folder is a complete, self-contained hand-off of one problem: *get an
egocentric (position-free, drone-eye-view) RL policy to thread a single drone-racing gate
~90% of the time, so we can trust it before scaling to a multi-gate course.* It is written
for an imagined entity that can solve the problem **fully** — but only if we hand over
**everything** we know. So everything is here: what we saw, what we tried, what worked, what
back-fired, what we deliberately rejected, the empirical "laws" we inferred, and — just as
important — what we **never measured** and **never tried**.

Written 2026-07-08 by the RL commander (Claude, Opus 4.8) for Fengyou. Repo: `Peregrine`
(Anduril AI Grand Prix entry). Branch: `claude/vq2-ego-single-gate-rl-0b78fb`.

---

## TL;DR

- **Goal:** ≥90% single-gate *thread* rate (clean pass through the aperture, zero contact),
  as a gate before multi-gate work.
- **Best policy to date:** `vglpan` — **~16% thread** (as-trained regime, stochastic + domain
  randomization), mean crossing offset **~0.9 m** Euclidean. Deterministic (deployed-mean) ~10%.
- **The last honest diagnostic (the gate-plane "hit map", see [07](07-diagnosis-hitmap.md)) split
  the failure into two independent problems we had been conflating:**
  1. **~46% of drones never reach the gate at all** (crash / floor / timeout before the plane).
  2. The crossings that *do* happen scatter ~1–2 m around the aperture with a mild **vertical
     high-bias**, against an **effective clean-pass window of only ~0.42 m** (the 0.28–0.38 m body
     radius shrinks the 0.75 m geometric aperture). We had been targeting sub-0.75 m; the real
     target is sub-0.42 m.
- **Central unsolved question:** how to get from 16% → 90% given those two problems, a policy that
  **collapses whenever a warm-started reward is changed structurally** (observed 6×), and an
  exploration-noise / domain-randomization sensitivity that muddies which number is "real".

---

## The single most important thing we learned

Progress came almost entirely from **one mechanism that finally worked**: a **smooth parabolic
crossing reward whose "zero radius" is annealed continuously, within a single training run, from
wide (≈gate distance) down toward the aperture.** That took the crossing offset from ~6 m → ~0.9 m
*with lateral gate variation switched on* — after a dozen other centering ideas plateaued or
collapsed. The reason it worked where discrete steps failed: **any discrete change to an
already-warm-started policy detonates it** (see [06](06-empirical-laws.md)), and a continuous
in-run schedule has no discontinuity to detonate on.

It is *not enough* — 0.9 m ≫ 0.42 m effective aperture, and it does nothing for the 46%
never-reach problem. But it is the thread we were pulling when this dossier was written.

---

## How to read this folder

| File | What's in it |
|---|---|
| [01-problem-and-goal.md](01-problem-and-goal.md) | The task, the definition of "thread", validity rules, why single-gate first, the moonshot context. |
| [02-system-and-stack.md](02-system-and-stack.md) | The full ML stack: observation, privileged critic, algorithm, environment, every reward term, the estimator, the course sampler, the racing line, the training harness, deployment. |
| [03-physics-frames-dr-geometry.md](03-physics-frames-dr-geometry.md) | The simulated plant, domain randomization (every component), coordinate frames & the tail-first / camera-flip conventions, gate & body geometry, the sensor wire. |
| [04-experiment-log.md](04-experiment-log.md) | **The core data.** Every run we launched, its config, its result, and what we concluded. Chronological. |
| [05-fixes-applied.md](05-fixes-applied.md) | Every fix/lever we built, what it did, whether it held. |
| [06-empirical-laws.md](06-empirical-laws.md) | The repeatable patterns / footguns we inferred (warm-start collapse, bundling trap, PBRS-telescoping, direction<distance, etc.). |
| [07-diagnosis-hitmap.md](07-diagnosis-hitmap.md) | The gate-plane hit-map diagnostic: where crossings land, the 2×2 noise×DR attribution, the reach-rate + effective-aperture reframe. Figures in `figures/`. |
| [08-ideas-rejected-and-untried.md](08-ideas-rejected-and-untried.md) | Every idea we shot down (with why), and — separately — everything we **never tried** and everything we **never measured**. |
| [09-open-questions-for-the-oracle.md](09-open-questions-for-the-oracle.md) | The specific questions we want answered, ranked. |

Figures live in [`figures/`](figures/). The analysis is reproducible from the scripts described in
[07](07-diagnosis-hitmap.md).

---

## Caveats on our own numbers (read before trusting anything below)

- **Renders have repeatedly misled us; we diagnose from training metrics.** The one exception is
  the offline hit-map in [07], which is computed from logged trajectories, not a viewer.
- **"Thread %" is measured in the training regime** (stochastic actions + domain randomization) unless
  a row says otherwise. The deployed policy is the *deterministic mean*; the gap is modest (~16%→10%)
  but real.
- **Turning domain randomization fully off is NOT "deployment"** — it removes *modeled* aerodynamics
  and drops actuator latency out-of-distribution. See [03](03-physics-frames-dr-geometry.md) and [07].
- Some early runs predate a terminal-reward change, so absolute thread numbers are not always
  comparable across the whole log; each [04] row notes the code era where it matters.
