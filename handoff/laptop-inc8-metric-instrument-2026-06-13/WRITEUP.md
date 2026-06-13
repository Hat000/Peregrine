# LAPTOP-INC8-METRIC-INSTRUMENT — 2026-06-13

**Session:** laptop-inc8-metric-instrument  
**Commit:** 0bb60cc  
**Tests:** 647/647 green (620 prior + 27 new)

---

## What was built

`rl/contact_true_eval.py` — the inc8 Phase-0(b) metric instrument.  
`tests/test_contact_true_eval.py` — 27 unit tests covering all key components.

### Replaces

The legacy **point-mass L-inf<0.75 pass proxy** previously used in the eval path is now
replaced by **contact-true scoring** via `_score_gate()`, which:

1. Calls `slab_frame_hit_np` (imported from `offline_rollout`) to detect pre-plane and
   steep-approach frame strikes (the exact mechanism that crashed inc6 standing at gate 3).
2. Applies inflated bands `[_HALF_OPEN - r, _HALF_OUTER + r]` for the body DR radius `r`.
3. Uses the same `_HALF_OPEN=0.75`, `_HALF_OUTER=1.36` constants — **no re-derivation**.
4. Accepts a `gate_pos_zup` override for the map-offset sensitivity probe.

### Architecture

```
contact_true_eval.py
  _score_gate(prev_ned, cur_ned, gate, body_radius, frame_depth, gate_pos_zup)
    -> (verdict, linf)   # reuses slab_frame_hit_np from offline_rollout
  run_episode(actor, start_state, target_gate, virtual_flip, params, ...)
    -> (EpisodeResult, pos_trajectory_ned)
  per_gate_margin_stats(results) -> dict gate -> {linf/margin stats}
  compute_s_stable(results_by_seed, threshold=0.90) -> (s_stable, per_seed_sr)
  gate3_d_offset_probe(pos_traj, delta_d_m, ...) -> dict
```

### Legacy parity verified

`test_all_gates_parity_random`: 200 random gate-frame segments per gate, all 6 gates,
`_score_gate(r=0, depth=0)` == `gate_event()` (from offline_rollout) on every case.
This is the negative-control: the new module does not regress the legacy classifier.

---

## inc7 baseline results

Run: `python rl/contact_true_eval.py --ckpt rl/checkpoints/stage1_inc7_actor.pth --plant mixer`

### Configuration

| Parameter | Value |
|-----------|-------|
| body_radius (nominal) | 0.33 m |
| frame_depth | 0.30 m |
| pass_band (contact-true) | 0.420 m (`0.75 - 0.33`) |
| plant | mixer (fully measured, inc6+ default) |

### S_stable

**S_stable = 1.000 (7/7 seeds)** — All 7 start conditions (simstart + trainreset at gates 0-5)
finish cleanly under contact-true geometry.

Note: training-time S_stable for inc7 = 1/3 (only 1 of 3 training seeds converged — the narrow
basin finding). The eval-time S_stable here measures start-condition robustness for the s0 checkpoint;
it does not retroactively measure training convergence breadth.

### Per-gate contact-true margin distributions (7 seeds, 1 episode each)

| gate | n_pass | linf_p10 | linf_med | linf_max | margin_p10 | margin_med | margin_min | note |
|------|--------|----------|----------|----------|------------|------------|------------|------|
| 0    | 2      | 0.109    | 0.109    | 0.109    | 0.311      | 0.311      | 0.311      |      |
| 1    | 3      | 0.035    | 0.079    | 0.092    | 0.330      | 0.341      | 0.328      |      |
| 2    | 4      | 0.112    | 0.144    | 0.204    | 0.232      | 0.276      | 0.216      |      |
| **3** | **5** | **0.021** | **0.036** | **0.133** | **0.320** | **0.384** | **0.287** | historically binding |
| 4    | 6      | 0.153    | 0.161    | 0.215    | 0.213      | 0.259      | 0.205      |      |
| 5    | 7      | 0.054    | 0.059    | 0.152    | 0.315      | 0.361      | 0.268      |      |

**Gate-3 isolated row:** linf ranges 0.021–0.133 m. Contact-true margin min = **0.287 m** (vs
legacy fiction ~0.699 m at simstart). The training doctrine change (contact-true geometry in
training env) worked: inc7's gate-3 paths are cleanly within the pass band.

**Tightest gate offline**: Gate 4 (margin_min 0.205 m), not gate 3. Gate 2 also tight (0.216 m).
These are the candidates for inc8 envelope-ladder scrutiny.

### Legacy vs contact-true comparison (simstart, gate-3)

| Metric | Legacy (L-inf<0.75) | Contact-true (L-inf<0.420) |
|--------|---------------------|---------------------------|
| simstart g3 linf | 0.051 m | 0.051 m (same measurement) |
| simstart g3 margin | 0.699 m (fiction) | 0.369 m (truth) |
| verdict | PASS | PASS |
| inc6 standing (historical) | ~0.44 m fiction | ~0.06-0.16 m truth (crashed) |

The gap the doctrine fixed is now visible: inc7's 0.369 m true margin vs inc6's 0.06-0.16 m.

### Cross-check against inc6_geom_eval

`handoff/laptop-inc7-env-2026-06-12/scripts/inc6_geom_eval.py` (inc6 policy, standing nominal):
reported `g3 true margin 0.06-0.16 m at r=0.33`. Our module produces the same geometry (same
`slab_frame_hit_np` + same constants) — verified by `test_legacy_parity.test_all_gates_parity_random`
and `test_slab_numpy_mirror_agrees_with_torch` (existing test from `test_contact_geometry.py`).

### Gate-3 D-offset sensitivity probe

Trajectory from simstart re-scored with gate-3 shifted ±1.5 m in NED Down:

| delta_D | outcome | g3_pass | g3_coll | g3_linf | g3_margin |
|---------|---------|---------|---------|---------|-----------|
| 0.0 m (nominal) | FINISHED | 1 | 0 | 0.051 m | +0.369 m |
| -1.5 m (gate 1.5 m higher NED) | **COLLISION** | 0 | 1 | N/A | N/A |
| +1.5 m (gate 1.5 m lower NED) | **COLLISION** | 0 | 1 | N/A | N/A |

**Key finding:** Both ±1.5 m D-shifts convert gate-3 from PASS to COLLISION. With the nominal
0.051 m linf, any offset > 0.42 - 0.051 = 0.369 m in either D axis changes the verdict.
Since live finding A reports ~1.46 m D-offset in the track_map, the offline metric **is indeed
confounded** — the drone flew cleanly live but the probe confirms that a ~1.5 m mis-calibration
would reverse the offline verdict.

**Implication for inc8 selection:** Inc8 gate-3 offline margins should be interpreted with
caution until SHADOWPC-VISION-CAL resolves the track_map offset. The contact-true instrument
makes this confound explicit (the legacy metric hides it behind 0.699 m of fictional margin).

---

## What changed vs the legacy metric

| Aspect | Before (legacy L-inf<0.75) | After (contact-true) |
|--------|---------------------------|----------------------|
| inc7 simstart g3 pass margin | 0.699 m (fiction) | 0.369 m (truth) |
| Pre-plane slab strikes | invisible | scored as collision |
| Gate-4 tightest margin | hidden | 0.205 m (tightest, watch for inc8) |
| Map-offset sensitivity | invisible | explicit ±probe |
| Selection grounding | wrong metric | what the sim enforces |

---

## Footguns respected

- **plant=map footgun:** The module defaults to `--plant mixer` (fully measured). `flat` is still
  available for legacy evals but no inc6+ eval should use it. Verified: the module prints the
  plant name in the summary line.
- **inc4 default footgun:** The CLI requires `--ckpt` (no default); no silent fallback to inc4.
- **620/620 legacy tests preserved:** 647 total (27 new).

---

## Using the instrument for inc8 selection

```bash
# Score a new checkpoint with the contact-true metric
python rl/contact_true_eval.py --ckpt rl/checkpoints/<new_ckpt>.pth --plant mixer

# To test at DR boundary (body_radius=0.28 conservative, 0.38 aggressive)
python rl/contact_true_eval.py --ckpt ... --body-radius 0.28
python rl/contact_true_eval.py --ckpt ... --body-radius 0.38

# Gate-3 offset probe at the measured mis-calibration distance
python rl/contact_true_eval.py --ckpt ... --g3-probe-range 1.46
```

For inc8 arm selection, compare `margin_min` across gates; flag any arm where `margin_min < 0.10 m`
on gate-3 or gate-4 (tightest in inc7 baseline: g4=0.205, g2=0.216).

---

## MEMORY-DELTA

(For commander banking agent — triage before writing to memory/)

1. **inc7 contact-true baseline ESTABLISHED**: g3 margin 0.287-0.384 m (truth, contact-true),
   vs legacy fiction 0.699 m. Gate-4 tightest at 0.205 m min — flag for inc8 envelope ladder.
2. **S_stable = 1.000 (eval-time)**: All 7 start seeds finish under contact-true geometry.
   Distinct from training-time S_stable (1/3 viable seeds); eval robustness is healthy.
3. **Map-offset ±1.5 m both flip gate-3 verdict to COLLISION**: offline metric confounded by
   track_map mis-cal ~1.46 m. Confirmed empirically. Gate-3 margins must be interpreted with
   caution until SHADOWPC-VISION-CAL.
4. **contact_true_eval.py LIVE in main (commit 0bb60cc)**: replaces legacy L-inf<0.75 proxy for
   offline inc8 selection; `per_gate_margin_stats` + `compute_s_stable` + `gate3_d_offset_probe`
   are the canonical APIs. 647/647 tests.
5. **SUPERSESSION**: Memory entry for "inc7 pass offsets [0.37, 0.05, 0.06, 0.03, 0.10, ~0.71] m"
   (from live VQ1 recording) is the live oracle — NOT superseded (these are different: live RACE_STATUS
   vs offline contact-true scoring). Both are valid. Contact-true offline g3_linf=0.051 m (simstart)
   vs live g3 pass offset ~0.06 m — consistent.
