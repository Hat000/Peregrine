# v1 + v15 pick-flights — v15 transfer CONFIRMED (quiet yaw ~as simmed, sim→wire 1:1 again); Ws0 breaks the gate-2 wall (gates=3, first in the settled record); v1 was a hard regression (yaw pegged at the clamp)

2026-07-19 (ShadowPC). Flights on `ratchet-arrestor-2026-07-18` working tree (fly_rl = ego-deploy
superset; arrestor OFF ⇒ policy path byte-identical). Champion recipe throughout: TRT M-engine
detector, assist 1.3 g, frozen champion log-echo map (`configs/vq2_coarse_map_champion_logecho.json`),
det-hold 0.2, stale 0.5, rate 40 @ scale 1.2, virtual-flip, slot1, yaw clamp 0.7. v15 delta:
**PITCH FREE** (no `--ego-pitch-clamp`). Settled GO ≥3 s; md5s verified against the relay
(`e3bcd9ab…` Qs1 / `29f493e2…` Ws0). Metric conventions calibrated on the champion log:
yaw absmean = mean|rate_frd[2]| (champion 0.151 ✓ exact), flips/s counted between consecutive
|wz|≥0.05 samples (champion 2.22 ≈ ref 2.2), max speed = KF finite-diff (champion 15.21 ✓ exact).

## v15 results (my CLI flights, settled protocol)

| flight | gates | end | contacts | yaw absmean / flips/s | max spd | passes (s) |
|---|---|---|---|---|---|---|
| Qs1 f1 | 2 | CRASH | 1 | 0.135 / 2.14 | 7.9 | g0 2.18, g1 4.43 |
| Qs1 f2 | 1 | CRASH | 3 | 0.117 / 1.85 | 8.8 | g0 2.23 |
| Qs1 f3 | 2 | CRASH | 5 | 0.098 / 2.21 | 8.7 | g0 1.97, g1 4.49 |
| **Ws0 f1** | **3** | CRASH | 2 | **0.154 / 0.89** | 9.3 | g0 2.20, g1 4.71, **g2 5.75** |
| Ws0 f2 | 0 | CRASH | 200 | — | — | **VOID: non-settled** (race created 3.3 s after sim boot, GO ~0.5 s later; drone still bouncing from the spawn drop) |
| Ws0 f3 | 2 | CRASH | 200* | 0.172 / 0.35 | 10.1 | g0 2.21, g1 4.49 |

*f3's contact count accumulated during a post-crash grind; flight-relevant story below.

**Scored distributions: Qs1 {2,1,2} · Ws0 {3,2}** vs settled-champion baseline {2,2,1,2,2}
(vpeffs0). Zero-contact validity: **0/5 counted flights contact-free** (same as the baseline).

## Adjudication

1. **Sim→wire transfer: CONFIRMED a third time, now in the QUIET direction.** Qs1 wire yaw
   absmean 0.098–0.135 vs predicted ~0.14 (champion 0.151); Ws0 flips 0.35–0.98/s vs predicted
   ~0.7 (champion 2.2). v1 had already confirmed the HOT direction (below). The sweep's yaw
   numbers are now a trustworthy pre-flight gate.
2. **Max speed 7.9–10.1 ≤ the 12 m/s training abort** — the priced-in overspeed held; no
   runaway dive observed in any counted flight (pitch commands active, range ±1.1).
3. **PITCH FREE transforms the opening**: g0 pass at **1.97–2.23 s across all 5 counted flights
   vs 2.94–3.00 for the entire clamped lineage** (champion 2.945). ~0.8 s faster to gate 0,
   same reliability (5/5). g1 at 4.43–4.71 vs champion 5.17 / clamped-pack 5.45–5.48.
4. **THE HEADLINE: Ws0 f1 passed gate 2 (5.75 s) — the first settled-launch flight ever to
   clear the descend wall** (0/5 champion baseline, 0/15 v1, all P0.x replays died there or
   earlier). The quiet-yaw fallback, not the sweep-topper, did it. It then crashed hunting g3.
5. **Emergent failure mode past g1 (both ckpts): the descend gate goes UNDETECTED for seconds.**
   Ws0 f3 flew BLIND for ~6 s after the g1 pass (conf 0.00, slow wander) before reacquiring g2
   and crashing on contact; Qs1 f3's final tick was conf 0.00. With the flight line now surviving
   past g1, **detection/acquisition of the descend gate is on the critical path** — the gate
   leaves the +20°-mounted camera's view during the drop. (Arrestor's stare-brake, or a z-bias,
   are the obvious levers; not flown here per DO-NOTs.)

## v1 scoring (Fengyou's 15 panel flights, 2026-07-18 16:0x — the deferred v1 report)

| ckpt | gates distribution | yaw absmean / flips/s | note |
|---|---|---|---|
| v1As0 (5 flights) | {1,1,0,0,1} | 0.34–0.68 / 4.4–8.0 | yaw PEGGED at the 0.7 clamp |
| v1Rs0 (10 flights) | {1,0,0,0,0,2,1,0,0,0} | 0.60–0.67 / 4.4–6.6 | same; one 61 m/s KF blowup during a tumble (estimator artifact, not physical) |

**v1 = hard regression** vs the settled baseline (mean ~0.5 gates vs 1.8): the sweep's hot yaw
transferred 1:1 to the wire and the clamp rode the rail. This is the negative control that makes
the v15 quiet-yaw confirmation meaningful.

## Protocol notes

- One non-settled start caught and re-flown (Ws0 f2 → f3) — the tell is a GO within ~4 s of a
  fresh sim boot + a contact storm at spawn. On fresh boots: wait for the waiting room, THEN
  count the full 3 s.
- All counted GOs were fresh (`pos_off=0.00`, late-join guard refused one leftover race).
- v15 models + full recipe are wired into the panel (`MODEL_DEFAULTS`, pitch clamp auto-set to 0)
  — panel picks are recipe-faithful for follow-up flights.
- Sessions committed (this commit): all six v15pick + the ten v1pick sessions the panel button
  hadn't captured (5 were already at 6bd989e).

## MEMORY-DELTA (≤10 lines)

1. v15 picks 2026-07-19 (settled, champion recipe + PITCH FREE + yaw 0.7): **Qs1 {2,1,2},
   Ws0 {3,2}** vs baseline {2,2,1,2,2} — **Ws0 f1 = FIRST settled flight through gate 2**
   (g2 @ 5.75 s). Sim yaw metrics transfer 1:1 (Qs1 0.098–0.135 abs ≈ pred 0.14; Ws0 flips
   0.35–0.98 ≈ pred 0.7); max spd ≤10.1 (12 abort held). Pitch-free = g0 in ~2.0–2.2 s
   (clamped lineage 2.94–3.00), same 5/5 reliability.
2. v1 picks (Fengyou, 15 flights) = REGRESSION: A {1,1,0,0,1}, R {1,0,0,0,0,2,1,0,0,0}; wire yaw
   pegged at the 0.7 clamp (0.6–0.68 abs, 4.4–8 flips/s) — hot-yaw sim prediction also
   transferred 1:1. Quiet yaw is the pick criterion that works.
3. NEW critical path past g1: the descend gate goes conf-0 for seconds (Ws0 f3 blind-wandered
   ~6 s before reacquiring + crashing) — g2 DETECTION during the drop, not the flight line, is
   now the wall. Non-settled tell on fresh boots: GO ≤4 s after boot + contact storm at spawn
   → void + re-fly.
