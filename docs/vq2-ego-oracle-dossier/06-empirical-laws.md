# 06 — Empirical Laws / Footguns We Inferred

Repeatable patterns from the log. These are *inferences from behaviour*, not proofs — but each is
backed by ≥2 observations. An oracle should either exploit or explain/refute them.

### L1. Any large *structural* change to a warm-started policy collapses it into a floor-dive.

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

### L6. For a smooth crossing bowl, the "zero radius" must be tight enough to make a gradient — and
must be *shrunk continuously*, never stepped.

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

### L14. The ~0.88 m crossing is a CONTROL-precision floor, not a reward/noise/convergence limit.

Three independent pushes all fail to tighten it: more training (`vglpan6` == `vglpan`), a tighter
reward zero (`vglp05` 4→0.5 → **worse**, 1.7 m), a sharper noise floor (`vglpshp` → **collapse**).
Decisive: at the working end zero of 0.75 m, a 0.88 m crossing *already* earns a **negative** parabola
(`≈ −7.6`) — the reward is already pushing tighter and the policy **cannot comply**. → Further reward
shaping is spent; the lever is now **control / speed / perception**. Leading hypothesis: the policy
crosses too fast (~8 m/s) to thread a ~0.42 m window because progress pays closing-rate up to
`rw_vmax_mps = 39 m/s`. Under test (`vglpsl5`/`vglpsl3`, capped rewarded speed).

### L15. The reward geometry (0.75 m) and the real target (0.42 m) disagree — but you can't just retarget.

The body radius (0.28–0.38 m) makes the *effective* clean-pass ~0.42 m, so a 0.6–0.7 m crossing clips
even though it's inside the 0.75 m geometric aperture. Intuition says anneal the reward zero to ~0.42 m —
but `vglp05` shows that *back-fires* when the policy is control-limited above that (it just punishes
achievable crossings and destabilises). Closing the reward-vs-body gap requires first *lowering the
control floor* (slow down / better perception), not tightening the reward past what control can hit.
