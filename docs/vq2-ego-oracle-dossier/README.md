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

> ⚠️ **READ [00-CORRECTION-eval-harness.md](00-CORRECTION-eval-harness.md) FIRST.** After writing the
> rest of this dossier we found the offline eval harness that produced the [07] hit-map is
> **unfaithful** (it reports 46% out-of-bounds where the real training env reports 0.05%). The
> "reach-rate / 46%-never-reach / +3.2 m high-bias" findings are **artifacts** and are corrected there.
> The TL;DR below is already corrected; [07] is left as-was with a correction banner.

## TL;DR

- **Goal:** ≥90% single-gate *thread* rate (clean pass through the aperture, zero contact),
  as a gate before multi-gate work.
- **Best policy to date:** `vglpan` — **~20% thread** (trusted in-training metric), and **22.6%
  deterministic** (faithful `_run_det_eval` — see below), mean crossing offset **~0.88 m** L-inf.
- **The trusted diagnosis (from the in-loop box-exit, not the broken offline harness): a pure
  centring problem.** ~100% of drones reach the gate and cross the plane; **71% clip the frame**
  (cross at L-inf 0.75–1.36 m, *just* outside the aperture); ~9% miss wide; oob≈0, floor≈0. There is
  **no reach-rate problem** — the earlier "46% never reach" was an eval-harness artifact.
- **NO determinism gap:** the deployed (deterministic-mean) policy threads **22.6%** ≥ the stochastic
  20% — the held exploration noise was mildly *hurting*. The "deployed policy is much worse" fear (from
  the broken harness's 0%) is false. Every run now self-reports a faithful `DET_EVAL[...]`.
- **The centring floor is CONTROL-limited at ~0.88 m — reward shaping is spent.** More training, a
  tighter reward zero (`vglp05`→worse), and a sharper noise floor (`vglpshp`→collapse) all fail; at the
  0.75 m end zero a 0.88 m crossing already earns a *negative* reward and the policy still can't tighten.
- **The real target is sub-0.42 m** (body radius shrinks the 0.75 m aperture), so the fix is to *lower
  the control floor*, not tighten the reward past what control can hit.
- **Central live question:** is the ~0.88 m floor caused by **speed** (crossing ~8 m/s, too fast to
  thread 0.42 m — under test via `rw_vmax_mps` caps, `vglpsl*`), by **perception** (estimator noise),
  or does it need an **arrive-head-on / low-lateral-velocity** term? Plus the standing puzzle: the
  policy **collapses whenever a warm-started reward is changed structurally** (observed 6×).

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
| [00-CORRECTION-eval-harness.md](00-CORRECTION-eval-harness.md) | **Read first.** The offline eval harness is unfaithful; corrects the [07] reach-rate/high-bias findings and states the trusted training numbers. |
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
