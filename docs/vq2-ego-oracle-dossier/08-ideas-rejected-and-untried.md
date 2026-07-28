# 08 — Ideas Rejected, and Everything We Never Tried / Never Saw

> 🛑 **RETRACTED/CORRECTED (2026-07-09 audit).** Two corrections to the UPDATE block immediately below:
> (1) **the `vglp05`-based sub-aperture rejection is invalid** — `vglp05`'s zero-anneal override was a
> no-op (the anneal hook was inert), so it ran config-identical to `vglpan`; the "newly rejected,
> zero→0.5 → worse" reading is really a ~2.8× thread / ~2× xoff single-seed replicate, not an
> independent test of over-tightening (see [11-audit-2026-07-09.md](11-audit-2026-07-09.md), the
> `vglp05` finding). (2) **`r_perc` moves back to UNTESTED-CLEANLY** (C3, REFUTED conf 0.78) — its
> rejection (referenced later in Part A) was confounded by a hidden discrete 4.0→0.75 reward-zero jump
> plus a fragile re-warm base whose no-`r_perc` control collapsed identically; see C3 in
> [11-audit-2026-07-09.md](11-audit-2026-07-09.md).

> **UPDATE (2026-07-09) — several items below have since moved.** Newly **rejected** (tested, failed):
> the sub-aperture anneal (`vglp05`, zero→0.5 → *worse*, 1.7 m) and the sharper endgame noise floor
> (`vglpshp`, 0.015 → collapse). Newly **measured/resolved**: the deterministic policy (faithful
> `_run_det_eval` — deployed 22.6%, **no determinism gap**); the reach-rate "gap" was an **eval-harness
> artifact** (training oob 0.05%, not 46%), so Part B's "reach-rate breakdown" is moot. Now **being
> tried**: the speed cap (`rw_vmax_mps` → 5/3, `vglpsl*`). See [04] Era 6 and [00-CORRECTION].

## Part A — Ideas we deliberately rejected (with why)

- **Gate-normal in the observation.** We *had* a gate-normal channel and **removed it**: perceiving the
  normal from vision was far too inconsistent. Replaced by `visible_area` (a coarse, range-free
  foreshortening proxy for "am I head-on"). Re-proposing gate-normal is a known dead end unless
  perception improves.
- **Anisotropic vertical-weight progress (for the varied gate).** Fixes the flat-gate floor-dive but
  over-corrects on a height-varied gate (side/back divergence, `vaniso` 97% side). Kept only as a
  *flat-section* fallback. (See [06] L8.)
- **MPCC-style along-track-lag progress.** Dropping isotropic/line homing for segment-lag lost the
  plane-reach (`vmpcc` 60% floor). Rejected as the *primary* progress term.
- **`frame_clip_is_miss` (frame-moat fix).** Floor-dives (removes the caution the frame penalty
  implicitly enforced). Abandoned; the parabola achieves no-moat *without* removing the penalty. (L1.)
- **Pure vector-field `dot(v̂, F̂)` as the centring mechanism.** Speed-blind, centres to 10–12 m.
  Rejected. (L5.)
- **k≥10 contouring.** Collapses with anneal. k=4 ceiling. (L1/L6.)
- **Fresh dense magnitude centring.** Floor-dives fresh; warm-only. (L2.)
- **Fresh lateral-FOV jitter.** Too hard from scratch (`vglat`/`vgpsm`). Lateral must come via warm-start
  or a jitter curriculum.
- **Discrete zero-shrink ladder (rung-by-rung re-warm).** Detonates the warm-start every rung
  (`vglp3`, `vglpsub`, `vglpc40`). Replaced by the continuous in-run anneal. (L1/L6.)
- **Wide fixed aperture as a standalone target.** Warm-start collapse (`vgap8`); the continuous anneal
  is the surviving form.
- **Bundling multiple new levers in one fresh run.** No attribution + collapse (`vgpsm`). (L3.)

## Part B — Things we NEVER MEASURED (data gaps)

- **The reach-rate breakdown for the current best.** We know ~46% never reach the gate but have **not**
  split it into floor-dive vs out-of-bounds vs timeout/stall for vglpan. This is one cheap box-exit read
  and would directly aim front (A). *Highest-value missing measurement.*
- **Per-axis error attribution over training.** How much of the residual offset is lateral vs vertical,
  as a function of update, for the *deterministic* policy. (We have it only at the final hit-map.)
- **A "dr_nominal" evaluation** (DR code-path ON, params pinned at nominal, force-bias zeroed). This is
  the honest "nominal deployment" proxy that keeps the aero structure — we ran only full-DR and full-off.
- **The actual competition-sim dynamics vs our nominal plant.** We do not know how the real VQ2 sim
  differs (aero, latency, thrust) from `peregrine_plant`. The whole DR-fidelity question hinges on this.
- **Whether the vertical high-bias causes the floor-dives.** The DR-off policy flies high; does the
  *low-gate* subset (gate below spawn) preferentially floor-dive because the policy can't push down
  enough? Not separated.
- **Deterministic eval was only run at the very end.** For most of the campaign we optimised the
  stochastic training number without watching the deployed mean.
- **Sensitivity to `γ`, entropy weight, network size, n_envs, seed.** We fixed γ=0.9975 (load-bearing
  by one earlier check) and never swept it or the others for the ego generation. **No seed ensemble** —
  every result is a single seed=0.
- **The effect of the 8-keypoint visibility / estimator noise on centring.** We never ablated the
  estimator noise (accel/gyro/visible-area σ) to see how much of the offset is perception vs control.

## Part C — Levers we NEVER TRIED

- **Anything targeting reach rate directly** — approach-stability shaping, a smoother speed profile, a
  floor-proximity penalty separate from the terminal, an altitude-hold assist that fades near the gate.
  (We ran an altitude-hold *probe* once, in a different context, but never as a reach-rate lever here.)
- **A non-parabola crossing bowl.** A cone (linear, constant gradient to centre — no flat top) or a
  Gaussian. The parabola is flat at the top (weak final tightening); a shape with a stronger near-centre
  gradient might close the last 0.9→0.4 m. (Fengyou preferred smooth, so a Gaussian, not a cone-kink.)
- **A body-radius-aware reward.** The reward geometry uses 0.75 m; the *real* clean-pass is 0.42 m.
  We never made the reward target the body-effective window (e.g. zero-radius annealed to ~0.42 m with
  the valid-thread caveat handled, or an explicit body-inflated pass bonus).
- **A jitter *curriculum*** (ramp lateral jitter 0 → ±7° → ±14° → ±20° across warm-started stages), vs
  the all-at-once ±14° we used. Gentle lateral onset might avoid the reach-rate hit.
- **A fresh in-run anneal straight to sub-aperture** (rather than re-warming vglpan, which collapses) —
  one run, fresh or warm-from-champion, zero annealed 4→0.4 with the L7 valid-thread handling.
- **Endgame noise floor below 0.02** (e.g. 0.01) to sharpen the deterministic mean, paired with the
  zero-anneal endgame.
- **Velocity-alignment-at-crossing reward** (reward crossing the plane *moving head-on* / low lateral
  velocity), distinct from position centring — to arrive square and not overshoot laterally.
- **A vertical-specific term** to kill the persistent +vertical bias (a signed vertical penalty, not the
  isotropic L-inf), if the bias survives the fidelity fix.
- **PPO/critic hyperparameter changes** to make warm-starts less fragile (longer critic warmup, KL
  trust-region, value-clip) — we only ever used critic_warmup 100/400.
- **Two-stage decouple:** train reach-to-plane first (reward = reach the plane at any offset), then
  freeze/curriculum into centring — instead of jointly.
- **Multi-gate anything** — not touched. Gate-ID / ByteTrack / misassociation handling are designed but
  deferred.

## Part D — Things we saw but don't understand

- **Why re-warming vglpan collapses even for a *continuous* anneal start** (`vglpsub` started at
  zero=0.75 = vglpan's end, yet detonated). The lower noise floor and/or sub-aperture were also changed,
  so it's confounded — but it suggests the anneal-produced policy is *more* fragile than a
  champion-produced one.
- **Why DR-off shifts specifically +high** (vertical), not some other direction. Latency-0 overshoot is
  the leading hypothesis but unproven; the aero-off lift change is an alternative.
- **Why ~46% never reach the gate in DR-on but ~0% fail to reach in DR-off** (DR-off they all barrel to
  the plane fast and miss high). The randomized dynamics evidently cause early crashes that nominal
  doesn't — but we haven't localised it.
