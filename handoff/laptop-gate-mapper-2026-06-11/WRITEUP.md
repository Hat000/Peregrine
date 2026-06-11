# Offline gate mapper â€” laptop session, 2026-06-11 (LAPTOP-GATE-MAPPER)

**Mission:** the one missing infra piece for VQ2 cases B (pose + rough map) and C (no pose,
vision-only): after a conservative exploration lap, build or refine the gate map OFFLINE
from logged detections, between attempts. Built tonight, validated on synthetic data
generated from the MEASURED perception-noise model (perception-char 2026-06-08), on the
real VQ1 course geometry, with the real camera/visibility model.

**Status: COMPLETE.** `src/racer/gate_mapper.py` + `gate_mapper_synth.py` +
`scripts/run_mapper_offline.py` + `scripts/validate_gate_mapper.py`, 31 new tests, full
suite green. Validation tables below; honest failure modes in Â§5.

---

## 1. What was built

| piece | role |
|---|---|
| `src/racer/gate_mapper.py` | the mapper: pose-aided robust averaging (A), rough-prior fusion (B), no-pose linear gate-landmark adjustment (C); sightings-file I/O; capture-schema map output |
| `src/racer/gate_mapper_synth.py` | synthetic validation: measured noise model, real VQ1 geometry (verbatim from the 2026-06-02 capture), real camera visibility, exploration-path generator, scoring |
| `scripts/run_mapper_offline.py` | CLI: `racer.mapper_sightings/v1` JSON in â†’ `track_map_*.json` out (+ per-gate report) |
| `scripts/validate_gate_mapper.py` | the sweep harness that produced the tables below (`--quick` for a fast re-run) |
| `tests/test_gate_mapper.py` | 31 tests: primitives, all three cases, schema round-trips, failure modes, CLI end-to-end |

Pure offline: numpy + scipy only; no imports from navigator/localization/state_estimator
(ShadowPC owns those tonight â€” nothing wired into the live loop, by design).

### The seam (consume-interchangeably contract)
- **Input** `racer.mapper_sightings/v1` (one JSON, two modes):
  - `pose_aided`: `{gate_world_ned, gate_id?, yaw_world?, t, frame_id?, range_m?}` where
    `gate_world_ned = drone_position_ned + R_world_camera @ t_cam_gate` â€” the trusted pose
    plus the PnP lever, both already present per-fix in the live chain.
  - `relative`: `{frame_id, rel_position_frd, quat_wxyz, gate_id?, rel_yaw?, t}` â€” PnP
    lever in body FRD + the GIVEN attitude quat; no drone position anywhere.
- **Output** `GateMapEstimate.to_track_records()` emits **exactly the
  `capture_track_map.py` schema** (bottom-centre position + authored col0=+width /
  col1=normal / col2=+height(down) quaternion), so `navigator.load_track_map(...,
  corner_to_center=True)` reads a mapper map and a given map identically. Round-trip is
  asserted in tests to 1e-6 (positions AND through-yaws), including a full
  noisy-sightings â†’ mapper â†’ JSON â†’ live-loader loop.
- Gates assumed upright (only position + through-YAW estimated); per-gate covariance,
  yaw std, sighting counts and honesty flags ride in a `mapper` provenance block the live
  loader ignores.

A future `extract_mapper_sightings.py` for real recordings = the `export_course_bundle.py`
alignment recipe (dedup frames by `frame_id`, nearest ODOMETRY quat / LOCAL_POSITION_NED by
recv clock) + detector + PnP â†’ one of the two row shapes above. The synthetic generator
writes the same schema today (`samples/` here), so the CLI is exercised end-to-end now.

---

## 2. The measured noise model (and the sign trap)

From `handoff/perception-char-2026-06-08` (canonical 6/6 course recording, N=312 fixes,
KF-accepted subset):

- world-fix error: bias **[âˆ’0.42, +0.06, âˆ’0.28] m**, Ïƒ **[0.73, 0.47, 0.29] m** (N,E,D),
  range-flat to ~24 m
- ~Â±3Â° angular term â†’ modelled as a per-gate-per-flight yaw bias U(âˆ’3Â°,3Â°) + 2Â° white
  (the per-gate part does NOT average away â€” that is the yaw error floor)
- catastrophic leak (post-Ï‡Â² residual): **1.1%** default, 3â€“8 m random direction, half
  with a 90/180Â° yaw flip
- association errors: **10%** default (measured 85â€“95% reliable at 5â€“15 m), modelled by
  the REAL mechanism: the label says a neighbour gate while the geometry stays true to the
  gate actually seen (live, a good relative pose anchored to the wrong map gate)
- visibility: real intrinsics + the +20Â° up-tilted mount via `racer.frames`; detection
  range 2â€“28 m default (measured flat-to-24, live cap 32 â€” both swept), p(detect)=0.85

**Sign trap (now pinned by a test):** those are FIX-error statistics
(`fix = gate_true âˆ’ lever`), and a gate measurement is `drone_pos + lever`, so the gate
measurement error is **âˆ’e_fix**: a mapper estimate built from raw sightings sits at
`truth + [+0.42, âˆ’0.06, +0.28]`. `bias_correction_ned` therefore **adds** the measured fix
bias. Correction is **opt-in (default off)** â€” the bias was measured on one course flown
one direction; it is a calibration, not a constant of nature. Both rows appear in the
table. Notably the bias projects mostly **along the through-axis** (gates face âˆ’X, bias is
mostly N): the **in-plane** error â€” what the 0.75 m validity half-opening actually budgets
â€” stays ~0.35 m even uncorrected.

---

## 3. Algorithms (short version, details in module docstrings)

**A (pose-aided):** per gate, iterative median/MAD outlier rejection on the normalized 3D
residual norm (k=4 â‰ˆ âˆšÏ‡Â²â‚ƒ,â‚€.â‚‰â‚‰â‚‰ â€” the same convention as the live KF innovation gate),
then inlier mean + statistical covariance. Yaw: robust circular (circular-median seed, MAD
rejection â€” kills 90/180Â° flips). Association: upstream labels if present (wrong labels
land ~an inter-gate spacing away and die in rejection); else greedy spatial clustering
(course-ordered by first-seen time). Clustering is immune to association errors by
construction â€” the geometry is true to the gate actually seen. Labeled mode additionally
**geometry-cross-checks the label groups**: a group whose robust centre coincides with a
bigger group's is folded in (`n_label_groups_merged`) â€” with partial coverage, a
yet-unseen gate's label can be 100% neighbour mislabels, a tight phantom parked at the
WRONG gate that no internal outlier screen can catch (quarter-lap sweeps measured 24â€“40 m
map errors before this fix; the validation table's partial-lap rows now stay sane).

**B (rough prior + refinement):** per-axis information fusion, measurement variance =
statistical + `map_systematic_std`Â² (0.25 m: the per-gate offsets averaging can't remove),
prior trust configurable (`prior_sigma_pos`, default 2 m / 15Â°). Below `min_sightings` the
measurement is hard-downweighted (an unscreenable 2-sighting group can hide a leak
outlier), and a Ï‡Â² consistency gate keeps the prior outright when the measurement
contradicts it beyond kÂ·Ïƒ of the combined uncertainty (a 100%-mislabel group sits ~24 m
off; dragging 20% toward garbage is worse than ignoring it). Unseen gates pass through
flagged `prior_only`.

**C (no pose):** with attitude GIVEN, the residual `p_gate âˆ’ x_pose âˆ’ R(q)Â·rel_frd` is
LINEAR in all unknowns â†’ sparse exact-Jacobian least-squares (scipy `trf`+`lsmr`), robust
`soft_l1` loss, translation gauge anchored on the first pose. Yaw decouples entirely
(each rel-yaw is a direct gate-yaw measurement) â†’ same robust circular estimator. Scale is
inherent (PnP against the known 1.5 m inner square). Pipeline hardening that the
calibration runs forced (each was a measured failure first):

1. **Velocity dead-reckoned chain init** â€” holding a stale pose across a blind transit gap
   founded every post-gap gate ~15 m off as one rigid block.
2. **Two-stage solve (linear â†’ robust)** â€” soft_l1 saturates on a metres-off init seam
   (vanishing gradient â†’ premature ftol stop, components stranded at init); the convex
   linear pass drags every component through the bridge first, the robust pass then only
   re-weights outliers locally. This took case C from ~54 m to ~1.3â€“4.9 m.
3. **Virtual coast poses** in detection gaps, so the const-accel smoothness prior chains
   through the gap instead of extrapolating one noise-amplified instantaneous velocity.
4. **Duplicate-gate merge** post-solve (a gate re-founded across a gap coincides after
   adjustment; real gates are â‰¥23.7 m apart, merge radius 8 m) + spurious-speck drop
   (`min_sightings`, the leak's debris).
5. **Smoothness prior Ïƒ_a = 1 m/sÂ² default** â€” honest FOR THE EXPLORATION LAP (we fly it
   deliberately smooth, near-constant velocity); corner impulses at gates land where
   observations dominate and the robust loss absorbs them.

Labels in case C are used for output **naming only** (majority vote per spatial gate) â€”
never for geometry association, so a mislabeled sighting still lands on the gate it
actually saw (immune to first-occurrence-mislabel poisoning, and mislabels get *used*
rather than rejected).

---

## 4. Validation (the deliverable proof)

Full sweep output: `validation_table.md` here (regenerate:
`python scripts/validate_gate_mapper.py --out ... --dump-sightings ...`). Cells are
**mean (worst)** over seeds; "in-plane" is the error component in the gate plane â€” what
the 0.75 m validity half-opening actually budgets.

### Case A â€” pose-aided (5 seeds, ~61 sightings/gate/lap)

| config | gates | err mean m | err max m | in-plane max m | yaw max deg |
|---|---|---|---|---|---|
| baseline (labeled, 1 lap) | 6/6 | 0.52 (0.58) | 0.63 (0.70) | 0.36 (0.39) | 2.47 (2.74) |
| + measured-bias correction | 6/6 | 0.13 (0.15) | **0.20 (0.25)** | 0.13 (0.17) | 2.47 (2.74) |
| unlabeled (clustering) | 6/6 | 0.54 (0.58) | 0.62 (0.68) | 0.33 (0.34) | 2.78 (2.92) |
| noise x0.5 | 6/6 | 0.26 (0.29) | 0.31 (0.35) | 0.18 (0.20) | 2.47 (2.74) |
| noise x2 | 6/6 | 1.03 (1.17) | 1.27 (1.41) | 0.72 (0.79) | 2.47 (2.74) |
| leak 0% / 5% | 6/6 | 0.52 / 0.54 | 0.64 / 0.62 | 0.35 / 0.33 | ~2.5 |
| assoc errors 5% / 15% | 6/6 | 0.53 / 0.52 | 0.63 / 0.64 | 0.34 / 0.36 | ~2.5 |
| 0.25 lap / 0.5 lap | 2/6, 3/6 | 0.54 / 0.54 | 0.61 / 0.62 | 0.33 / 0.33 | ~2.2 |
| 2 laps | 6/6 | 0.51 (0.55) | 0.57 (0.62) | 0.32 (0.34) | 2.50 (3.06) |

**Reading:** every robustness axis is FLAT â€” leak 0â†’5% and association errors 5â†’15%
change nothing (the rejection is doing its job; cluster mode rejects almost nothing
because mislabels can't exist there), and partial coverage maps only what it saw, at full
accuracy. The error budget is the measured BIAS, full stop: correction takes 0.63â†’0.20 max.
**Acceptance vs the 0.75 m half-opening: total 0.63 < 0.75 (uncorrected; bias-dominated),
in-plane 0.36 = well under, bias-corrected 0.20 total = well under.** The one threatening
row is noiseÃ—2 (total 1.27, in-plane 0.79 â‰ˆ the budget) â€” if a real VQ2 lap measures 2Ã—
the VQ1 noise, map from two laps and/or enable correction measured on that course.

### Case B â€” rough prior + refinement (5 seeds)

| config | prior err max m | fused err max m | gain |
|---|---|---|---|
| offset 2 m, 1 lap, prior_sigma 2 | 2.00 | 0.62 (0.69) | 3.2x |
| offset 2 m, 0.25 lap (starved) | 2.00 | 2.00 (2.00) | 1.0x â€” **no harm** |
| offset 3 m, 1 lap | 3.00 | 0.62 (0.70) | 4.8x |
| offset 1 m, 1 lap, prior_sigma 1 | 1.00 | 0.59 (0.68) | 1.7x |

**Reading:** one lap pulls any 1â€“3 m rough prior to the ~0.6 m measurement floor (= the
uncorrected bias; add `--bias-correct` for ~0.2). The starved row is the designed
failsafe firing: Ï‡Â² consistency gate + low-n downweight keep the prior untouched rather
than dragging it toward unscreenable debris. Fusion never made anything worse in any run.

### Case C â€” no pose (3 seeds; aligned = translation gauge removed)

| config | gates (+extra) | covis comps | aligned mean m | aligned max m | raw max m | in-plane max m |
|---|---|---|---|---|---|---|
| baseline (scan, range 28, sig_a 1.0) | 6/6 (+0) | 3 | 1.64 (2.57) | 2.14 (3.30) | 4.25 (5.82) | 0.85 (1.37) |
| no scan nod | 6/6 (+0) | 4 | 2.80 (3.13) | 3.40 (3.80) | 6.67 (7.78) | 3.08 (3.71) |
| range 24 (measured flat-to) | 6/6 (+0) | 6 | 2.53 (3.53) | 2.84 (3.86) | 5.76 (7.37) | 2.00 (3.48) |
| range 32 (live cap) | 6/6 (+0) | 2 | 1.46 (2.22) | 1.82 (2.55) | 3.85 (5.08) | 1.56 (2.51) |
| sig_a 0.5 / 6.0 | 6/6 (+0) | 4 / 3 | 1.58 / 2.09 | 2.23 / 2.40 | 4.41 / 4.72 | 0.98 / 1.13 |
| NO smoothness bridge | 6/6 (+0) | 3 | 1.88 (3.12) | 2.23 (3.43) | 4.87 (6.75) | 1.27 (1.58) |
| leak 5% | 6/6 (+0) | 3 | 2.21 (3.70) | 2.50 (4.15) | 5.33 (8.18) | 2.15 (3.65) |
| labels given (naming) | 6/6 (+0) | 3 | 2.91 (4.58) | 3.77 (6.11) | 6.54 (11.16) | 1.75 (2.20) |
| 2 passes | 6/6 (**+6**) | 7 | 1.57 (2.51) | 2.06 (3.27) | 4.12 (5.74) | 0.82 (1.23) |

**Reading:** degraded but BOUNDED, as targeted â€” every single-pass config sits in a
1.8â€“3.4 m aligned band (yaw everywhere â‰¤ ~3Â°, solve 2â€“22 s). Detection range is the
dominant lever (32 m â†’ 1.82; 24 m â†’ 2.84 with the graph in 6 pieces, still bounded). The
scan nod buys most of its value in-plane (3.08 â†’ 0.85 m). Striking: the velocity
dead-reckoned chain init carries even the NO-bridge case (2.23 m with the observation
graph in 3 pieces) â€” the smoothness bridge is refinement + connectivity accounting, not
survival; before the init fix this exact config was 54 m. The 2-pass row is the honest
loop-closure gap: per-pass geometry is great (2.06 aligned) but pass 2 re-founds the
course (+6 phantom gates) â€” see failure mode #7; single-pass per solve, then pose-aided
refinement. The labels-given row differs from baseline only through rng coupling (labels
never enter geometry); its spread is seed noise.

---

## 5. Honest failure modes (read before trusting a case-C map)

1. **Consecutive-gate co-visibility is geometrically IMPOSSIBLE on this course.** Both
   gates are within detection range only *between* them (spacing 23.7â€“38.5 m vs 24â€“32 m
   range), where one is behind the forward camera (90Â° HFOV). A deliberate pre-transit
   **scan nod** (~25Â° pitch-down within ~8 m of each gate) buys a few co-visible frames on
   the â‰¤24 m legs only; the 39 m g2â†’g3 leg has none at any plausible range. Without the
   smoothness bridge, the observation graph SHATTERS (4â€“6 components) and inter-component
   placement is pure init â€” the `NO smoothness bridge` table row shows the wreckage, and
   that is exactly what the `disconnected` flags + `covis_components` +
   `warning_connectivity` diagnostics are for. **The no-pose map's backbone is the motion
   prior, not co-visibility.** Corollary: fly the exploration lap SMOOTH and SLOW; lap
   count adds sightings but does NOT fix connectivity (same pattern every lap).
2. **The bridge has a noise-INDEPENDENT error floor.** The path *turns at gates*, i.e.
   inside the blind gap; a const-accel bridge cuts that corner (~1.2â€“1.6 m on this
   geometry even with near-zero sensor noise â€” measured in the straight-course control
   test where the floor vanishes). Bridge drift accumulates down-course like a random
   walk; the gauge anchor means raw (world) error grows while aligned (shape) error stays
   bounded. **The case-C map is a SHAPE estimate**: consumers must localize against it
   (which is what the vision stack does anyway), not treat it as world-registered.
3. **Case-C absolute frame is unobservable by construction** (3 translation DoF gauge).
   Raw-vs-truth error includes the anchor pose's unknowable position; score and consume it
   translation-aligned. If ANY world anchor exists in VQ2 (start pose, one surveyed gate,
   a rough prior), register the shape to it â€” or better, use B-mode.
4. **Yaw error floor â‰ˆ the per-gate Â±3Â° bias** (it does not average away). Yaw outliers
   (flips) are handled; the bias is not separable offline from one lap. Validity impact at
   1.5 m opening: negligible.
5. **The measured bias correction is course/direction-specific** â€” default OFF. Uncorrected
   case-A maps carry ~0.5 m along-track offset (in-plane stays ~0.35 m). If VQ2 lets state
   carry between attempts, the FIRST racing attempt against the mapped course measures the
   residual bias for the next one (progressive refinement is the natural loop here).
6. **Spurious/duplicate gates** are real failure modes at the measured leak rate
   (leak-spawned specks; gap-drift re-founds): handled by `min_sightings` drop +
   post-solve merge; `n_dropped_gates` / `n_merged_gates` in diagnostics tell you it
   happened. A gate seen <3 times stays low-confidence (flagged) â€” fly closer next lap.
7. **Multi-pass case C has NO loop closure.** The high return leg is a ~60 s blind gap;
   dead-reckoning across it drifts far beyond the 8 m duplicate-merge radius, so pass 2
   re-founds the whole course (+6 phantom gates in the `2 passes` row â€” the per-pass
   geometry stays good, which is why the matched errors still look fine, but the OUTPUT
   has 12 gates). Re-visit recognition is real SLAM scope. The workflow instead: one pass
   per case-C solve; once the first map exists, later laps localize against it and feed
   **pose-aided** mode (the progressive-refinement loop) â€” which also dissolves the gauge
   problem.
8. **Partial-coverage labeled mode** can manufacture phantom gates out of pure mislabel
   groups (no true sightings to outvote them). Fixed by the geometry cross-check (Â§3 A);
   the residual risk is a phantom whose members straddle TWO neighbour gates (needs both
   neighbours mislabeling toward the same unseen id in one window â€” the robust iteration
   collapses it to the denser side and the merge then catches it).
9. **Where scipy stops being enough:** if attitude must FLOAT (real-IMU yaw drift instead
   of the sim's given quat), the problem stops being linear â†’ GTSAM (SO(3) manifold, IMU
   preintegration, iSAM2 incremental) is the upgrade path. At â‰¤10 landmarks and a few
   hundred poses, scipy solves the given-attitude problem in seconds; conditioning is
   reported (`jtj_cond`, `jtj_min_eig`) so a sick solve is visible, and the per-gate
   marginals come from the exact normal-matrix pseudo-inverse (meaningful only within the
   anchor component â€” pinv assigns ~0 pseudo-variance to unobservable directions).

## 6. Operational recommendations (the VQ2 case-C exploration lap)

- Fly it **slow (~5 m/s), smooth (near-constant velocity), and known-in-advance** â€” the
  smoothness prior is the map's backbone and Ïƒ_a is a promise about OUR flying, not the
  platform envelope. Add the **pre-transit scan nod** on short legs (free co-visibility).
- Prefer B-mode the moment anything map-like exists (the FAQ suggests "rough" data is
  likely). B-mode with a 2â€“3 m prior is dramatically better-conditioned than pure C and
  immune to the gauge problem.
- Between attempts, re-run the mapper with the new lap's sightings appended (more
  sightings per gate; for case C also re-anchor against the previous map as a prior via
  pose-aided mode once the first map exists â€” the racing attempts themselves become
  mapping laps, since by then the navigator localizes against the map and pose-aided mode
  applies).

## 7. Repo state

- Commits (this session): `46a4cc4` core+synth+tests Â· `6584ec3` CLI+sweep harness Â·
  `5303e9e` phantom-mislabel-group merge Â· plus the handoff commit (this directory).
  All pushed to origin/main (pull --rebase'd around the concurrent RL/TOGT sessions).
- Test suite: **528 passed** full-suite final run â€” 31 are this session's
  (`tests/test_gate_mapper.py`); baseline at session start was 457 (other overnight
  sessions added the rest concurrently).
- `scratch/` here: `smoke.py` (magnitude calibration), `debug_c.py` (the case-C 54 m
  dissection that motivated hardening #1/#2). `samples/`: example sightings files in the
  CLI input schema. `validation_table.md`: the full sweep output (regenerate via
  `python scripts/validate_gate_mapper.py --out ...`).
- Not touched: `memory/`, `src/racer/{localization,state_estimator,navigator}.py`,
  `vision/` (ShadowPC ownership respected; navigator imported in tests/CLI read-only as
  the round-trip oracle).
