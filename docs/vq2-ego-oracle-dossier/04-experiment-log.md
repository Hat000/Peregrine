# 04 — Experiment Log (the core data)

Every run we launched for the VQ2-ego single-gate problem, chronological, grouped by era. Each entry:
**tag** · config (deltas from the working recipe) · **result** · conclusion. Metrics are the box-exit
classifier during training (**stochastic actions + DR on**) unless stated. `xoff` = mean L-inf/Euclidean
crossing offset (m); `plane` = fraction reaching & crossing the plane off-aperture; `floor` = fraction
that dove into the floor; `thread` = clean pass. All single gate, depth 8–15 m, height ±6 m, standing
start, n_envs 2048, γ=0.9975.

Legend for outcome: 🟢 kept/advanced · 🟡 partial · 🔴 failed/abandoned.

---

## Era 1 — Which progress/homing form even reaches the gate plane?

The flat-vs-varied-gate saga. Baseline problem: a level same-height gate gives an isotropic
distance-to-centre potential a **zero vertical gradient** → 100% floor-dive.

- 🟡 **`varied`** — isotropic 3D homing (`progress_to_center`, w=1), gate height varied ±6 m. Result:
  **70% reach the plane**, floor 50%→20% (decaying, still improving at cutoff ~1450 upd), xoff 8.7→7.0.
  Best *partial* — the height variation un-buries the vertical gradient **geometrically** (gate
  above/below → the isotropic norm already has a vertical component). Conclusion: **isotropic homing is
  the only form that reaches the plane; don't replace it, augment it.**
- 🔴 **`vaniso`** — anisotropic vertical weight (couple the vertical into the progress potential, w=6).
  Floor **solved (0.00)** but **97% exit_SIDE**, xoff 11.4. The coupling rotated the *forward* gradient
  into lateral divergence. Conclusion: aniso destabilises at every weight (25→80% back, 6→97% side).
  Stashed as a *flat-section* fallback only.
- 🔴 **`vmpcc`** — decoupled MPCC-style along-track lag + cross-track corridor, k=4. 60% floor + 35%
  side, xoff 13.8. Dropping isotropic homing for segment-lag lost the plane-reach; k=4 too weak.
- 🟢 **`vgvf`** — **pure GVF**: progress from the curved head-on **racing line** arc-length +
  contouring k=4. Result: **plane 83%, floor 3%**, xoff 10→5.0 (≈3.9 vert + 3.1 lat), frame-clip 11%
  (committing), side 0.7% (terminal equalisation killed the bail). **WON the form race.**
- 🔴 **`vgvfh`** — isotropic homing + decoupled line-contouring (`racing_line_progress=false`). Stuck
  51% floor. Conclusion (we predicted the opposite): the head-on **line arc-length progress encodes the
  altitude profile** (arc to gate height, arrive horizontal) so it holds altitude; isotropic homing
  pulls diagonal-to-centre and permits the sink. **Winning form = `use_racing_line=true` +
  `racing_line_progress=true` + contouring.**

**Aniso↔isotropic reconciliation (this is *not* a flip-flop, it's a stage change):** a **flat** gate
needs aniso (zero vertical grad); a **height-varied** gate un-buries the vertical geometrically so
isotropic suffices and aniso over-corrects. Same gravity fight, signal moved from reward *weight* to
*geometry*.

---

## Era 2 — Centring the ~5 m crossing offset (still NO lateral gate variation)

- 🟢 **`vganl`** — GVF + **noise anneal** (k=4, hold 0.30 first half → floor 0.05). **Best pre-centring
  policy:** floor **0.7%**, plane **81%**, frame-clip 17% (genuinely close), xoff 5.0→**4.4** (≈3.4
  vert + 3.2 lat), thread 0.5%. 🚩 **The anneal did NOT tighten the offset** — it plateaued 4.4 m from
  upd 2000→3990. So **4.4 m is a mean-policy limit, not noise scatter.**
- 🔴 **`vgk10a`** — k=10 contouring + anneal. **Collapsed to 99.6% floor.** (Was floor 1.5% at upd 900
  during the hold, then the anneal **sharpened it into the floor**.) 🚩 **k=4 is the ceiling for PBRS
  contouring weight; strong contouring + anneal = floor-dive.**
- 🔴 **`vgctr`** — **fresh** magnitude centring (`rw_centering=0.4`, clamp 6 m). **100% floor.**
  🚩 **A magnitude (dense) centring penalty from scratch back-fires** — it bleeds reward before the
  field is learned. Apply only via warm-start.
- 🟢 **`vgctrw`** — **warm-start** vganl + magnitude centring 0.4 (clamp 6 m), low warm noise. xoff
  4.4→**3.38**, thread 0.5%→**1.3–1.6%**, frame-clip 20%. **Became "the champion"** (still no lateral).
  Diagnosis of *why centring was needed*: the PBRS contouring **telescopes** (rewards *reducing* perp),
  so a *sustained* 4.4 m offset earns zero gradient and the policy parks there. The magnitude term is
  the **integral ("I") term** — `−k∫perp dt` — that nags a standing offset. (D-term = telescoping PBRS.)
- 🔴 **`vgctrf` / `vgctrf2`** — champion + **frame-moat fix** (`frame_clip_is_miss`: a frame-clip
  forfeits no banked progress = nets == a wide miss, so the aperture ring isn't a moat). `vgctrf` 100%
  floor; `vgctrf2` (+critic_warmup 400) recovered to 24% floor but **worse** than the no-fix control.
- 🔴 **`vgff`** — **fresh** `frame_clip_is_miss`. 99.6% floor (vs the same stage without it, vganl,
  0.7%). 🚩 **`frame_clip_is_miss` ABANDONED:** removing the frame penalty makes the drone commit
  aggressively toward low gates → dives into the floor; the frame penalty was implicitly enforcing
  caution near the gate. Fengyou's moat theory was sound but this implementation's side effect dominates.

---

## Era 3 — The vector field, the parabola, and lateral (this session's opening)

- 🔴 **`vgalsm`/`vgalsh`, then `vgv2sm`/`vgv2sh`** — the **true vector field**: replace contouring with
  a direction-alignment reward `rw_align·dot(v̂, F̂)`, `F = cos θ·tangent + sin θ·inward`,
  `θ = atan(align_gain·perp)`. `align_gain` = convergence tightness. Clean sweep (`vgv2sm` gain 0.5,
  `vgv2sh` gain 3.0): plane **44% / 19%**, xoff **12.3 / 10.4** — both **far worse** than champion
  (74% / 3.4 m). 🚩 **Direction-only reward is speed-blind — satisfied by "roughly pointing right"
  while crossing 12 m wide. "Reward being closer" (contouring + magnitude) ≫ "reward pointing the right
  way".** Vector field **abandoned** as the centring mechanism. (For the record: `F =
  normalize(t̂ + k·(c−p))`, flow lines `y = C·e^(−kx)`.)
- 🔴 **`vgpsm`/`vgpsh`** — **bundled** align + parabola + lateral FOV jitter, fresh. Plane-reach **0**,
  dead-stuck. 🚩 **Two lessons:** (a) lateral FOV jitter is **too hard fresh** (gate ±20° off-axis → a
  fresh policy never turns to reach it); (b) **don't bundle** 3 new levers — no attribution (the "ERA0
  trap").
- 🔴 **`vgap8`** — champion + **wide aperture curriculum** (`gate_inner_opening_m=8`, so the passage
  reward has a gradient across the 3.4 m dead-zone), warm from champion. **Collapsed** the warm-start →
  99.7% floor @ upd 170, clawed back only to floor 42% / xoff 6.4 (worse than champion). 🚩 **Wide
  aperture is a large structural change → warm-start floor-dive** (pattern, see [06]).
- 🔴 **`vgcp`** — champion + parabola crossing (zero=0.75) + critic_warmup 400. **Never ran** — a Hydra
  append-collision (`++critic_warmup_updates=400` in the stage `_raw` vs the sbatch `BOUNDARY_OV`'s
  always-appended `+critic_warmup_updates=100`) → RC=1, no tfevents. Fixed (inherit the 100). The
  parabola-at-0.75 was thus never actually tested (0.75 zero is a dead-zone at a 3.4 m offset anyway).

### The lateral A/B — the "x-y movement extremely important" finding

Fengyou flagged that the champion lineage trained with `spawn_yaw_jitter=0` → the gate was **always
dead-centre horizontally**, so the lateral obs channel `rel_pos[1]` was ≈0 in 100% of training. We
turned on ±14° jitter:

- 🔴 **`vglat`** — **fresh** vganl recipe + jitter 0.25. Plane 79%, xoff 5–6, thread ~0; anneal-collapsed
  at the very end (plane 79%→11%, floor 42% @ upd 3970).
- 🔴 **`vglatw`** — **warm** champion + jitter 0.25. **Plane-reach stayed 75% (== champion), but xoff
  grew 3.4→6.0 m** ≈ exactly the ±14° lateral offset. 🚩 **The drone reaches the plane and lets the
  whole lateral offset pass UNCORRECTED — it ignores `rel_pos[1]`.** Side-miss only 1%, floor 12%.

**Unified diagnosis:** the lateral miss and the 3.4 m no-jitter plateau are the **same disease** — the
centring reward is too weak to beat the "fly-forward-fast" progress incentive; the drone reaches the
plane and crosses wherever momentum leaves it.

---

## Era 4 — The parabola-anneal breakthrough

Fix = the smooth **parabola crossing** as a *strong* terminal centring gradient, but the **zero radius
must be wide enough to span the current offset** (at 0.75 m it's a dead-zone at 6 m), then shrunk.

- 🔴 **`vglp6`** — warm champion + jitter + parabola **zero=6.0**. xoff → 10 m. **Too flat near centre →
  no gradient → drifted worse.**
- 🟢 **`vglp4`** — warm champion + jitter + parabola **zero=4.0**. xoff **6→1.9 m**, thread **0→6.5%**.
  🚩 **BROKE the plateau, with lateral on.** Zero radius is the knob: too wide = no gradient, tight = pulls.
- 🔴 **`vglp3`/`vglp3b`** — **re-warm vglp4** into a *discrete* zero=3.0 (`b` also center 40). **Jumped
  to 8 m, recovered only to ~6 m.** 🚩 A discrete zero step IS a reward-semantic change → detonates the
  warm-start (critic_warmup 100 too short). (5th warm-start collapse.)
- 🟢 **`vglp3c`** — zero=3.0 warm **directly from champion** (a clean single jump, like zero=4 was).
  xoff **1.57 m**, thread **7%**, no collapse. 🚩 **Direct-from-champion walks tighter fine; re-warming
  an already-shifted policy is what collapses.**
- 🟢🟢 **`vglpan`** — **the winner.** Warm from vglp4 + **in-run zero anneal 4.0→0.75** (continuous,
  mutated per-update in the training loop — see [05]), 4000 upd. xoff rode **2.76→1.9→1.5→1.2→1.04→0.88**
  (monotone) as the zero annealed; thread **3%→20%**. Blew past every prior plateau. **Best policy to
  date.**
- 🟡 **`vglpan6`** — same anneal, **6000 upd** (hedge: convergence- or reward-limited?). xoff 0.90,
  thread 20.9% — **≈ identical to the 4000 run.** 🚩 **Not convergence-limited** — more time doesn't help;
  the ~0.9 m / 20% is a genuine plateau of this recipe.

---

## Era 5 — Endgame fine-tunes (both failed) + the hit-map diagnostic

- 🔴 **`vglpsub`** — **re-warm vglpan** + sub-aperture anneal (zero 0.75→0.5, lower noise floor 0.02).
  Collapsed: xoff → 8 m, thread → 0.
- 🔴 **`vglpc40`** — **re-warm vglpan** + double centre bonus (cross_center 20→40). Collapsed: xoff →
  9.7 m, thread → 0. 🚩 **6th warm-start collapse** — re-warming an already-specialised policy fails
  even for a *continuous* anneal start, once noise/centre are also changed. Only warm-**from-champion**
  or a *fresh* in-run anneal survive.

### The hit-map 2×2 (offline, from logged trajectories — see [07])

Rolled out **vglpan** deterministically and stochastically × DR-on and DR-off, 256 episodes each,
and reconstructed the gate-frame crossing point per episode:

| | DR-ON (as trained ≈ deploy) | DR-OFF (aero/latency removed) |
|---|---|---|
| **stochastic** (training metric) | thread **16%**, vert-bias +1.2 m | thread 1%, vert-bias +3.2 m |
| **deterministic** (deployed mean) | thread **10%**, vert-bias +1.6 m | thread **0%**, vert-bias +3.2 m |

Findings (AS WE BELIEVED THEM AT THE TIME): the training "~20%" is real; DR-off is a sensitivity probe;
in DR-on ~46% never reach the gate; crossings scatter with a vertical high-bias.

> 🛑 **RETRACTED — see [00-CORRECTION](00-CORRECTION-eval-harness.md).** The rollout harness that
> produced this table is unfaithful (it reports 46% oob where the real training env logs 0.05%). The
> "46% never reach", "+3.2 m high-bias", and these 2×2 numbers are **artifacts**. The trusted picture
> is the in-loop box-exit (Era 6): **20% thread, 71% frame-clip, ~0% oob/floor, xoff 0.88 m — a pure
> centring problem.**

---

## Era 6 — Faithful evaluation, the "no determinism gap" result, and the control-limited pivot

After Era 5 we stopped trusting the offline harness and re-derived from the authoritative in-training
box-exit. That changed the diagnosis and the strategy.

**Trusted training truth for `vglpan`** (in-loop, stochastic, DR-on, step 3990):
`success = exit_thread = 19.9%`, `collision (frame-clip) = 71.3%`, `plane-miss = 8.7%`,
`oob = 0.05%`, `floor ≈ 0`, `timeout ≈ 0`, `xoff = 0.88 m`. → **pure centring problem; no reach-rate
problem** (the Era-5 "46%" was a harness artifact).

- 🟢 **`_run_det_eval` (fix, not a run):** added a post-training deterministic eval to
  `peregrine_train_ego.py` — runs the policy `test=True` on the LIVE env and prints `DET_EVAL[...]`.
  Faithful by construction; auto-on for all future runs; evaluate an old ckpt via a 5-update warm run
  (actor frozen by the critic-warmup). See [05].
- 🟢 **`vglpanEV`** — faithful deterministic eval of vglpan (5-update frozen-actor warm run).
  **`DET_EVAL: thread 22.6%, frame-clip 68.4%, miss 8.9%, oob 0.03%`** (~20k episodes). 🚩 **oob 0.03%
  matches training (0.05%), NOT the broken harness's 46%** — confirms `_run_det_eval` faithful and
  `ego_render_rollout` broken. 🚩 **deterministic (22.6%) ≥ stochastic (20%): essentially NO
  determinism gap** — the held noise was mildly *hurting*. The deployed policy is as good as training.
- 🔴 **`vglp05`** — fresh anneal (warm vglp4), zero **4→0.5** (target the effective aperture, contra the
  earlier L7 worry, since a 0.6 m crossing clips anyway). Result: xoff **1.7 m, thread 7%** — *worse*
  than vglpan (0.88 m, 20%). **Over-tightening the reward zero back-fires** (crossings the policy can't
  achieve get strongly-negative parabola → destabilises → crosses wider).
- 🔴 **`vglpshp`** — fresh anneal + sharper endgame noise floor **0.03→0.015**. **End-collapsed**
  (xoff 9.9, thread 0). Over-sharpening destabilises the endgame (anneal-into-worse).

**Conclusion — the ~0.88 m crossing is a CONTROL-PRECISION floor, not a reward/noise/convergence
limit** (answers [09] Q5):
- more training doesn't help (`vglpan6` == `vglpan`);
- a tighter reward zero makes it worse (`vglp05`);
- a sharper noise floor collapses it (`vglpshp`);
- **decisive:** at vglpan's end zero of 0.75 m, a 0.88 m crossing *already* earns a **negative**
  parabola (`20·(1−(0.88/0.75)²) ≈ −7.6`) — the reward is already punishing it and the policy still
  cannot tighten. Reward shaping is spent.

**Pivot to control/speed/perception.** Leading hypothesis: the drone crosses at ~8 m/s (≈12 m in
~1.5 s) — too fast to thread a ~0.42 m window. The progress reward pays closing-rate up to
`rw_vmax_mps = 39 m/s`, a standing speed incentive.

- ❌ **`vglpsl5` / `vglpsl3` (DONE, 2026-07-09) — SPEED LEVER REFUTED, and in the *wrong* direction.**
  The vglpan recipe with the rewarded-speed cap `rw_vmax_mps` lowered to **5** and **3 m/s**. Same
  warm-start, same 4000 updates, same anneal. Result (final box-exit):

  | run | `rw_vmax_mps` | thread | xoff |
  |---|---|---|---|
  | vglpan (champion) | 39 (none) | ~0.20 | **0.88 m** |
  | vglpsl5 | 5 | 0.069 | 1.57 m |
  | vglpsl3 | 3 | 0.027 | 3.29 m |

  Capping the rewarded closing rate made centring **monotonically worse** (3.29 > 1.57 > 0.88 m as the
  cap tightened). The "crosses too fast to thread 0.42 m" hypothesis predicted slower→tighter; we got
  slower→looser. **Read carefully:** `rw_vmax_mps` does not *force* slower flight — it flattens the
  far-field progress gradient, so the drone dawdles far out and arrives at a *worse* offset. So this
  refutes the *lever* (`rw_vmax_mps` is not the knob), not the physics of "a slower crossing threads
  better" — a term that penalises *speed at the crossing plane* is still untried. But the standing
  speed-reward is not the cause of the floor.

- 🎯 **`vglpns0` / `vglpns5` (IN FLIGHT, 2026-07-09) — the PERCEPTION ablation (the decisive fork).**
  The vglpan recipe with a new single knob `+env.ego_noise_scale` = **0.0** (a perfect truth estimator)
  and **0.5** (half the measured noise), warm from vglp4, DET_EVAL on. `ego_noise_scale` multiplies ALL
  injected estimator noise/corruption (vision fix σ, per-episode in-plane bias, teleport/miss/normal-flip
  probs, IMU accel white + residual-bias drift, colored gyro, visible-area σ); default 1.0 is numerically
  identical to every prior run. Smoke-verified: scale 0 → rel_pos error 0.000 m; scale 1 → 0.281 m (the
  measured contract). Jobs 3298802 / 3298803, watcher `bginllvbb`. **Prior (quantitative):** the realized
  per-gate perception error is ~0.28 m (vertical-dominated) + a ≤0.19 m per-episode bias → perception can
  account for **at most ~0.25–0.3 m** of the 0.88 m offset, so I *expect* the offset to stay ~0.88 m
  (control-limited confirmed). **Fork:** offset stays ~0.88 m at scale 0 ⇒ **control-limited, decisively**
  (perception removed from the suspect list; next lever is the action space or an arrive-head-on term);
  offset drops toward the ~0.28 m perception floor ⇒ **perception-limited** (harden the estimator / add a
  perception-aware cue). The 0.5 point gives the dose-response slope either way.
