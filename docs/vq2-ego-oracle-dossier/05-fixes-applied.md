# 05 — Fixes / Levers We Built (and whether they held)

Each entry: what it is, why we built it, outcome.

## Held / in the current recipe

- **Egocentric position-free obs + privileged asymmetric critic (`algo=appo`).** The whole generation.
  Why: the prior world-frame KF drifted and laundered a position-immunity we never had; Swift's policy
  sees velocity + relative gate geometry, not absolute pos. Held — the privileged critic is what lets
  the policy learn at all (the asymmetric information is load-bearing; `ppo` seed-collapses).

- **GVF racing line + arc-length progress (`use_racing_line=true`, `racing_line_progress=true`).**
  Online Hermite head-on line, contouring perp. Why: isotropic homing floor-dives; the head-on line
  encodes the altitude profile so the drone arrives horizontal. **Won the form race (`vgvf`).** Held.

- **Terminal equalisation** (`_COMMON`): contact 200→**100** (Fengyou-authorised, DQ-scale), miss
  8→**100** (match, to kill the cheap-bail-teaches-gate-avoidance incentive), oob 200. Why: a
  miss<contact asymmetry taught gate-avoidance / bailing. Held — killed the `side` bail (`vgvf` side
  0.7%).

- **Noise anneal** (`inc8_noise_anneal.py`): clamp `actor_logstd` to a decaying std ceiling +
  anneal entropy. Why: constant entropy drives std to the bang-bang ceiling → the deterministic mean
  is an unvisited crash point. Held — but the **held floor is a precision cap** (the mean can't sharpen
  below it).

- **Magnitude centring — the "I term"** (`rw_centering`, clamp `rw_centering_max_m` widened to 6 m).
  Dense `−k·clamp(perp,0,max)`. Why: the telescoping PBRS contouring (D-term) gives *zero* gradient on
  a *sustained* offset, so the policy parks at ~4.4 m; the integral term nags it. Held **only via
  warm-start** (`vgctrw` 4.4→3.4 m); **fresh it floor-dives** (`vgctr`).

- **Smooth parabola crossing** (`rw_parabola_crossing`, `crossing_parabola_reward`): `r =
  clamp(cross_center·(1 − (Linf/R)²), −cap, cross_center)`. Fengyou's design ("large reward in the
  middle, 0 at the edges, negatives that grow fast outside … the policy reacts better to smooth things").
  Why: replace the discontinuous {pass/clip/miss} cliff with one smooth monotone bowl (no moat →
  closer is *always* better). Held **once the zero radius R is tight enough** (`vglp4` zero=4 broke the
  plateau; zero=6 was too flat).

- **In-run zero anneal** (`peregrine_train_ego._cross_zero_schedule` + `_resolve_cross_zero_anneal`,
  `+env.cross_zero_anneal/start/end/hold_frac`). Mutates `env._egorw.cross_zero_m` per PPO update
  (geometric start→end over the back `1−hold_frac`), mirroring the noise-anneal hook. The reward reads
  `w.cross_zero_m` live each step, so the mutation takes effect. Why: a **discrete** zero shrink
  detonates a warm-start; a **continuous** in-run shrink has no discontinuity. **This is the
  breakthrough** — `vglpan` rode 4.0→0.75 and took xoff 2→0.9 m, thread → 20%.

- **Hydra dup fix** — removed the duplicate `critic_warmup_updates` from the parabola stage `_raw`
  (the sbatch `BOUNDARY_OV` already appends it). Why: `vgcp` died RC=1 on the collision. Held.

- **appo/privileged-critic assert + camera-flip (RC1) + warm-start reset-logstd + critic-warmup.**
  Infrastructure guards, all held (see [02]).

## Built but ABANDONED (back-fired)

- **`frame_clip_is_miss` (frame-moat fix).** Make a frame-clip forfeit no banked progress (nets == a
  wide miss) so the aperture ring stops punishing getting-close. Sound theory. **Back-fired both fresh
  and warm** (`vgff`, `vgctrf`) → floor-dive: removing the frame penalty makes the drone commit
  aggressively toward low gates and dive. The frame penalty was implicitly enforcing caution.
  Abandoned. (The **parabola** achieved the intended no-moat property *without* removing the penalty —
  it keeps a *growing* penalty outside the aperture.)

- **Wide aperture curriculum (`gate_inner_opening_m=8`).** Give the passage reward a gradient across the
  dead-zone, then shrink. **Collapsed the warm-start** (`vgap8`, 42% floor). The *continuous parabola
  zero-anneal* is the surviving realisation of the same intent.

- **k=10 / k=15 contouring.** Stronger cross-track pull. `vgk10a` collapsed (99.6% floor with anneal).
  k=4 is the ceiling.

## Built but NEVER FAIRLY TESTED

- **`rw_area_dist_ref_m` (square-on / area-distance coupling)** — scale positive progress by how
  square-on the gate is when close (an *angle* signal complementing position centring). Deployed as a
  backup stage (`squareon`) but never cleanly run under the current recipe.
- **Parabola at the true 0.75 m zero (fixed, no anneal)** — only ever hit via `vgcp` (which never ran)
  or as the anneal *endpoint*. Never run standalone.
