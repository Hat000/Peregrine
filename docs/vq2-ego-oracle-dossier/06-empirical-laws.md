# 06 — Empirical Laws / Footguns We Inferred

Repeatable patterns from the log. These are *inferences from behaviour*, not proofs — but each is
backed by ≥2 observations. An oracle should either exploit or explain/refute them.

### L1. Any large *structural* change to a warm-started policy collapses it into a floor-dive.

> 🛑 **RETRACTED/CORRECTED (2026-07-09 audit)** — C8 (REFUTED, conf 0.82) re-checked all 6 claimed
> collapses against the actual stage configs: **1 of 6 was not a warm-start at all** (`vgk10a`, fresh),
> **2 carried the same hidden discrete 4.0→0.75 zero jump** as the `r_perc` runs (`vglpsub`,
> `vglpc40`) — leaving only **2 clean single-lever collapses** (`vgap8`, `vglp3`). Counter-evidence:
> **4 discrete-reward-change survivals** (`vgctrw`, `vglp4`, `vglp3c`, `vglpsl5`/`vglpsl3`) plus one
> massive obs-change survival (`vglpns0`); the *same* zero-3→4 change survives from one base (`vglp3c`,
> direct-from-champion) and detonates from another (`vglp3`, re-warm) — the "any" quantifier fails, and
> the "warm-from-champion survives" corollary below also fails (`vgap8` was warm-from-champion and
> collapsed). **Restated law:** *a discrete reward change that newly punishes the policy's CURRENT
> operating point can detonate it, base-dependently* — 2 clean supporting observations, not 6, all
> single-seed. See [11-audit-2026-07-09.md](11-audit-2026-07-09.md) C8. **Original (now-narrowed) claim
> below.**

Observed **6×**: `frame_clip_is_miss` (warm), `k=10`+anneal, wide-aperture (`vgap8`), discrete zero
shrink (`vglp3`/`vglp3b`), sub-aperture re-warm (`vglpsub`), double-centre re-warm (`vglpc40`). The
policy sits in a narrow attractor; changing the reward semantics faster than the critic re-adapts
sends the actor into the floor, and it recovers (if at all) to a *worse* basin. **Corollary:** the
things that *did* survive were (a) warm-start **from the champion** (a clean single jump from a stable
base), or (b) a **continuous in-run anneal** (no discontinuity). `critic_warmup=100` is not enough to
absorb a discrete reward-semantic change.

### L2. A dense penalty ("I term") back-fires when applied *fresh*, works *warm*.

Fresh magnitude centring (`vgctr`) → 100% floor; the same term warm-started (`vgctrw`) → 4.4→3.4 m.
Fresh, the penalty bleeds reward before the field is learned and the policy gives up / dives. **Apply
dense shaping only after the coarse task is solved.**

### L3. Fresh multi-lever bundles fail — no attribution *and* usually a collapse.

`vgpsm`/`vgpsh` (align + parabola + lateral, fresh) → dead-stuck. Change **one** lever per run.

### L4. Telescoping PBRS (potential-based, "D") gives zero gradient on a *sustained* offset.

`vganl`'s offset plateaued at 4.4 m and the noise anneal could not tighten it → a **mean-policy
limit, not noise**. The contouring rewards *reducing* perp (Δpotential); a drone flying *parallel* to
the line at a constant offset earns nothing. You need an **integral term** (`−k∫perp dt`, the
magnitude penalty) to nag a standing offset. D-term guides the approach; I-term closes the last bit.

### L5. "Reward pointing the right way" < "reward being closer."

A pure direction-alignment vector field (`dot(v̂, F̂)`, `vgv2*`) centres to 10–12 m — far worse than
the distance-reducing contouring + magnitude terms (3.4 m). Direction reward is speed-blind and
satisfied by a roughly-aligned velocity while crossing wide. Elegant, empirically weaker.

### L6. ⚠️ PARTLY RETRACTED (2026-07-09) — the "continuous anneal" was never actually running.

The in-run cross-zero anneal hook **silently no-op'd the entire 2026-07-08 campaign** (the trainer's
`env._egorw` did not reach the raw env through DiffAero's `RecordEpisodeStatistics` wrapper → `hasattr`
False → SKIP; caught 2026-07-09, see [04]/[05]). So `vglpan` trained at a **FIXED `cross_zero=4.0`**, not
the 4→0.75 anneal — its centering to ~0.88 m came from fixed zero=4 + corridor + magnitude centering, and
the "continuous anneal beat the discrete step" claim below is **unproven** (the discrete `vglp3` collapse
was real, but the surviving alternative was *fixed zero=4*, not a working anneal). The actual 4→0.75
anneal is now fixed and under test (`vcza2`). The parts of L6 still standing: a fixed zero=6 is too flat
(drifted worse) and zero=4 pulls (both observed at fixed values); the near-centre gradient scaling
argument is analytic. **Original (now-suspect) claim below.**

> 🛑 **RETRACTED/CORRECTED (2026-07-09 audit, extends the retraction above)** — C8 (REFUTED, conf 0.82)
> makes this explicit: **"continuous anneal is safe" now has zero clean positive evidence.** The one
> run this claim leaned on (`vglpan`) never ran a continuous anneal at all (it trained at fixed
> zero=4.0); the only genuinely continuous in-run anneals that *did* execute during the campaign
> (the noise-scale curricula `vns01`/`vns31`, once the hook bug was understood as inert-then-fixed)
> collapsed. `vcza2` (the real 4→0.75 anneal) is therefore the **first actual test** of this claim, not
> a replication — treat any result from it as new information. See
> [11-audit-2026-07-09.md](11-audit-2026-07-09.md) C8.

**[SUSPECT] For a smooth crossing bowl, the "zero radius" must be tight enough to make a gradient — and
must be *shrunk continuously*, never stepped.**

Parabola zero=6 (too flat near centre) drifted worse; zero=4 pulled (`vglp4`); a **continuous** anneal
4→0.75 rode the offset down (`vglpan`); a **discrete** 4→3 step collapsed (`vglp3`). The gradient near
centre scales like `−2·center·e/R²`, so a wide R is flat at the top (no final tightening) and a narrow
R punishes valid edge-crossings — hence anneal from wide→aperture.

### L7. The end must stay ≥ the aperture, or you punish valid threads.

An anneal end below 0.75 m makes a crossing at 0.6 m (a *valid* thread) earn a negative parabola →
the policy avoids it. (This is *why* `vglpsub`'s 0.75→0.5 was suspect; it also collapsed for L1.)

### L8. Isotropic homing floor-dives on a flat gate; height variation fixes it geometrically.

A level same-height gate → zero vertical gradient in `−‖pos−centre‖` → 100% floor. Varying gate height
±6 m un-buries it (gate above/below gives the norm a vertical component). Aniso reward-weighting also
un-buries it but over-corrects (side/back divergence) on a varied gate. → **Move the vertical signal
into geometry, not the reward weight.**

### L9. The head-on racing-line arc-length is a *better* altitude keeper than isotropic homing.

`vgvf` (line progress) beat `vgvfh` (isotropic homing + line contouring): the line arcs to gate height
and arrives horizontal; isotropic homing pulls diagonal-to-centre and permits the sink.

### L10. There is essentially NO determinism gap (corrected).

We initially feared the deployed (deterministic-mean) policy was far worse than the stochastic training
number. The faithful `_run_det_eval` refuted this: vglpan deterministic **22.6%** ≥ stochastic **20%**
(oob 0.03%, matching training). The held exploration noise was mildly *hurting*, not helping. So the
noise floor is **not** a blocker to close a deployment gap — and sharpening it further *collapses* the
endgame ([L14], `vglpshp`). (The earlier "16%→10% modest gap" was itself a broken-harness artifact.)

### L11. Turning DR fully off is not "nominal deployment."

`dynamics.dr=False` removes *modeled* aero and drops latency to an OOD 0. It shifts the policy +3.2 m
high. Use DR-ON as the deployment proxy; the DR-off number is a fragility probe.

### L12. Renders lie; diagnose from metrics.

Nearly every confusing flip in this project traced to trusting a render/viewer. The box-exit
classifier and the offline hit-map (from logged trajectories) are trustworthy; the deterministic
render was the source of the ~"successes" that were actually noise.

### L13. Counting is sound (we checked).

`success == exit_thread == n_passed_gates` agree exactly; `pass_offset_m < 0.75`. A 0% is genuine
non-arrival, not a blind classifier. (We *did* have a reconstruction bug in the *offline* hit-map — we
sliced off the crossing step and had to extrapolate — but the *in-training* counting is verified.)

### L14. ⚠️ CORRECTED — the ~0.88 m crossing is PERCEPTION-limited, not control-limited.

**2026-07-09 correction (supersedes the "control-limited" claim below).** The `vglpns0` ablation
(`ego_noise_scale=0`, a *perfect estimator*, else the vglpan recipe) drove the crossing **0.88 → 0.52 m
and still falling, thread 0.20 → 0.32 rising** — so removing the *estimator* noise substantially
tightens the floor. The original "control-limited" reading was wrong because all three of its failing
pushes (more training, tighter reward zero, sharper *actor-exploration* noise) left the **estimator
(observation) noise untouched** — the one thing that actually mattered. The effect (~0.4 m+) exceeds the
raw ~0.28 m measurement error because the **closed loop amplifies** it: the policy acts on a noisy
`rel_pos` every step → hedged/jittery control → a systematic offset larger than the instantaneous noise.
**Implication:** centering is gated by *perception*, so the deployable levers are (a) a perception reward
`r_perc` that keeps the gate detectable/observed near the plane (Swift/Geles; built 2026-07-09), (b) a
better estimator/detector (vision-commander domain), or (c) a policy more robust to the noise — **not**
further crossing-reward shaping. `vglpns0` is a *ceiling* (we lack a perfect estimator at deploy), not a
deployable number. The pre-correction reasoning is kept below for the record.

> 🛑 **RETRACTED/CORRECTED (2026-07-09 audit)** — two problems in the superseded block below, on top of
> it already being superseded. (1) **The `vglp05` citation is invalid**: `vglp05`'s zero-anneal
> `EXTRA` was a no-op (C2) — it ran config-identical to `vglpan`, so "a tighter reward zero (`vglp05`)
> → worse" is really a ~2.8× thread / ~2× xoff **single-seed replicate**, not an independent
> reward-tightening data point (see [11-audit-2026-07-09.md](11-audit-2026-07-09.md), the `vglp05`
> finding). (2) **The "~0.28 m measurement error" bound is the wrong summary statistic**: it is the
> 3-D `rel_pos` error norm, dominated by the ~0.33 m smoothed *depth* channel, which does not project
> into a head-on crossing offset; the crossing-relevant **in-plane** error is ≈0.18 m and is
> **bias-dominated** (a mostly-calibratable ≤0.19 m per-episode offset, not scatter) — so this
> paragraph's perception-vs-control bound is built on the wrong number. See C1/C7 in
> [11-audit-2026-07-09.md](11-audit-2026-07-09.md).

**[SUPERSEDED] Original L14 claim: "a CONTROL-precision floor, not a reward/noise/convergence limit."**

Three independent pushes all fail to tighten it: more training (`vglpan6` == `vglpan`), a tighter
reward zero (`vglp05` 4→0.5 → **worse**, 1.7 m), a sharper noise floor (`vglpshp` → **collapse**).
Decisive: at the working end zero of 0.75 m, a 0.88 m crossing *already* earns a **negative** parabola
(`≈ −7.6`) — the reward is already pushing tighter and the policy **cannot comply**. → Further reward
shaping is spent; the lever is now **control / speed / perception**.

**Speed sub-lever REFUTED (`vglpsl5`/`vglpsl3`, 2026-07-09).** Capping the rewarded closing rate
(`rw_vmax_mps` 39 → 5 → 3) made centring *monotonically worse* (0.88 → 1.57 → 3.29 m), the opposite of
the "too fast to thread" prediction. Caveat: `rw_vmax_mps` flattens the far-field progress gradient (the
drone dawdles and arrives worse) rather than forcing a slower *crossing*, so this kills the *lever*, not
the physics — a speed-at-plane penalty is still untried. But the standing speed reward is not the cause.

**Perception sub-lever under decisive test (`vglpns0`/`vglpns5`, 2026-07-09).** New `+env.ego_noise_scale`
knob (multiplies all injected estimator noise; 0 = perfect estimator). The realized per-gate perception
error is only ~0.28 m (+ ≤0.19 m per-episode bias), bounding perception to ≤~0.3 m of the 0.88 m offset —
so the expectation is the offset stays ~0.88 m at scale 0, which would **prove control-limited decisively**
and remove perception from the suspect list. If instead it drops toward ~0.28 m, perception was the cap.

### L15. The reward geometry (0.75 m) and the real target (0.42 m) disagree — but you can't just retarget.

The body radius (0.28–0.38 m) makes the *effective* clean-pass ~0.42 m, so a 0.6–0.7 m crossing clips
even though it's inside the 0.75 m geometric aperture. Intuition says anneal the reward zero to ~0.42 m —
but `vglp05` shows that *back-fires* when the policy is control-limited above that (it just punishes
achievable crossings and destabilises). Closing the reward-vs-body gap requires first *lowering the
control floor* (slow down / better perception), not tightening the reward past what control can hit.

### L16. VERIFY A HOOK ACTUALLY FIRES — a silent `hasattr`/wrapper skip can invalidate a whole campaign.

The cross-zero anneal (L6) and the first noise curriculum both *silently skipped* for want of an env
attribute behind a wrapper, and it went unnoticed for weeks because the runs still produced plausible
(fixed-config) results. Two tells that caught it: (1) a **dose-response pair that came back bit-identical**
(`vns01` vs `vns31`, different `noise_scale_start`, identical DET_EVAL) — impossible if the knob were
live; (2) grepping the run log for the hook's own `ON`/`SKIPPED` line. **Rule:** every in-run hook must
print `ON` with a live value each log interval, and every experiment must confirm that line (and/or a
dose-response that *must* differ) before trusting the result. `r_perc` (a reward term read in the env's
compute path) was NOT affected — only train-loop hooks reaching through the env wrapper were.
