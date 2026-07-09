# 07 — Diagnosis: the gate-plane hit-map

> 🛑 **CORRECTION — this section's numbers are UNRELIABLE.** See
> [00-CORRECTION-eval-harness.md](00-CORRECTION-eval-harness.md). The offline harness
> (`ego_render_rollout.py`) that produced these figures reports 46% out-of-bounds where the real
> training env reports 0.05% — it does not faithfully reproduce training. The "46% never reach",
> "+3.2 m high-bias", and the 2×2 thread numbers below are **artifacts**. The trusted picture (from
> the in-loop box-exit) is: **20% thread, 71% frame-clip, ~0% oob/floor, xoff 0.88 m — a pure centring
> problem.** The section is kept as-written for the record and as the reproduction of the bug.

This was our deepest look at *where* the crossings land. We believed it was trustworthy (computed
offline from logged trajectories, not a render) — but it turned out the *rollout that generated those
trajectories* was itself unfaithful (see the banner above). Read it as a cautionary tale.

Figures (in [`figures/`](figures/)):
- `ehm_stoch_dron.png` — **realistic regime** (stochastic + DR-on, the training conditions). The main
  answer.
- `ego_hitmap_vglpan.png` — **sensitivity probe** (deterministic + DR-off). Not deployment.
- `ehm_det_dron.png`, `ehm_stoch.png` — the other two cells of the 2×2.

Each figure has three panels: (1) the **gate-plane hit map** — a scatter of every crossing point in
gate-frame `(lateral y, vertical z)`, with the aperture (0.75 m), effective clean-pass (~0.42 m), and
outer-frame (1.36 m) boxes, plus overlaid lateral/vertical offset histograms; (2) **top-down** flight
paths (approach axis × lateral) with the optimal head-on line dotted; (3) **side** paths (approach ×
vertical), which exposes the vertical bias.

## How it was produced (reproducible)

1. `rl/ego_render_rollout.py --ckpt <run>/checkpoints --n-envs 256 --max-steps 700 [--stochastic]
   [--keep-dr] --out X.npz` on Adroit (a GPU node). It loads the checkpoint's **own** `.hydra/config.yaml`
   (so the spawn distribution matches training), forces standing start, and dumps `pos (T,N,3)`,
   `reset_step (N)`, `gate_pos`, `gate_yaw`, `spawn_pos`, `w_g_half`, and per-env outcome flags.
2. Offline analyzer (`ego_hitmap_analyze.py`, in the session scratchpad — a copy of its logic is in
   [06 L13] and the code is short): for each env, transform the trajectory to gate frame
   (`world_to_gateframe`), find the forward crossing of x=0, **extrapolate the last approach segment to
   x=0** (the reset overwrites the actual crossing step), classify by L-inf, render.
   - ⚠️ The first pass had a bug: slicing `pos[:reset_step]` drops the crossing step and the naive
     detector found *zero* crossings (contradicting the env's 92% miss). Fixed by extrapolating the last
     two approach points to the plane. **If you re-derive, extrapolate.**

## The 2×2 (vglpan, 256 episodes/cell)

| | DR-ON (as trained ≈ deploy) | DR-OFF (aero/latency removed) |
|---|---|---|
| **stochastic** (training metric) | **thread 16%**, vert-bias +1.2 m, lat 0.76 m | thread 1%, vert-bias +3.2 m, lat 2.58 m |
| **deterministic** (deployed mean) | thread 10%, vert-bias +1.6 m, lat 0.90 m | thread 0%, vert-bias +3.2 m, lat 2.55 m |

(vert-bias = mean signed vertical offset; +'ve = high. lat = mean |lateral|.)

## What it says

1. **The training "~20%" is real** — 16% at stoch+DR-on. Not a logging artifact. The **determinism
   gap is modest** (16%→10%), so the deployed *mean* is close to the training number. This corrected an
   earlier wrong belief that the deployed policy was ~0%.

2. **DR-OFF is a sensitivity probe, not deployment.** `dr=False` removes the modeled convex-collective
   aero and drops actuator latency to an OOD 0 (training was 1–3 steps). The whole crossing cloud jumps
   **+3.2 m high** and thread → 0%. Read this as *"the policy leans on the DR dynamics"* — a fidelity
   risk to check, **not** a claim that the deployed drone threads 0%. (We do **not** have an eval at the
   true competition-sim dynamics, nor a "dr_nominal" run.)

3. **In the realistic (DR-on) regime the failure splits into two independent problems** we had been
   conflating:
   - **Reach rate: ~46% of drones never reach the gate** (crash / floor / timeout before the plane —
     the grey trajectories). The thread ceiling is capped by *arrival*, not just centring. We had been
     spending almost all effort on centring.
   - **Centring vs a tiny real aperture:** arriving crossings scatter ~1–2 m with a mild vertical
     high-bias, but the target is the **~0.42 m body-effective window**, not 0.75 m. A 0.5–0.7 m crossing
     still clips. So even "perfect" sub-0.75 m centring wouldn't reach 90%.

## The reframe for the 90% target

The centring ladder (parabola + in-run zero anneal) is genuinely working (6 m → 0.9 m) but touches
**neither** the 46% never-reach **nor** the fact that the true target is 0.42 m. The path to 90% is
therefore **two fronts**:

- **(A) Reach rate** — why do ~46% not arrive? (floor-dive? oob? timeout/stall?) We have *not* pulled
  the floor/oob/timeout split for vglpan (it's one cheap read away in the box-exit). Lever candidates:
  approach-stability shaping, floor/altitude penalties, the vertical high-bias itself may be causing
  low-gate floor-dives.
- **(B) Sub-0.42 m centring** — continue the zero-anneal toward the aperture, but the end must stay ≥
  0.75 m (L7) while the *effective* target is 0.42 m — i.e. the reward geometry and the body-radius
  geometry disagree; this tension is unresolved (see [09]).
- Plus a **nominal-dynamics eval gate** so we optimise a number deployment will actually see.
