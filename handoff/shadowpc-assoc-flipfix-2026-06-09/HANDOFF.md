# Catastrophic-tail fix at the SOURCE — robust association + depth sanity (ShadowPC, 2026-06-09)

Repo pulled @ `2e7df71`. venv `.venv\Scripts\python.exe` (torch CPU + ultralytics from 2026-06-08).
Task: kill the measured 46% raw catastrophic world-fix tail (87 wrong-gate + 56 depth-flip fixes,
`handoff/perception-char-2026-06-08`) in the solver/association/estimator logic — no detector changes.

## TL;DR — measured before/after (course per-gate bundles, 360 frames, `20260607_194615_course_60s`)

| course-level (6 bundles pooled) | BEFORE (naive nearest-centre) | AFTER (robust assoc + depth sanity) |
|---|---|---|
| solved+associated fixes | 312 | 189 (junk refuses to associate) |
| **raw catastrophic (\|fix\|≥3 m) of solved** | **143 = 46%** | **12 = 6.3%** |
| catastrophic OFFERED to the χ² gate | 143 (46%) | **2 (1.1%)** (depth sanity drops 10 more) |
| **residual leak past χ²₀.₉₉₉** | **5 = 1.6% of solved** (1.4% of frames) | **2 = 1.1% of solved** (**0.6% of frames**) |
| good fixes (\|fix\|<1 m) | 103 (91 accepted) | **107** (94–95 accepted) — *recovered*, not lost |
| KF-accepted \|fix\| p50 / p90 | 0.84 / 1.74 m | 0.82 / 1.69 m |

Baseline reproduced bit-for-bit first (312/143/5 = the published perception-char numbers) via the new
`--naive-assoc` flag, so the deltas are apples-to-apples on the same frames/weights/map.
**All targets met: raw tail 46%→6.3% (≪46%), leak 1.1% ≤ 1.6%.** 370 tests green (suite was 357).

## What was actually wrong (from the per-frame dumps, before writing any code)
Decomposition (`tail_decomposition.py` on the prior session's k1.0 dumps):
- **Wrong-gate (87):** the course is a receding COLLINEAR line of gates, so every gate's predicted
  image centre clusters near the image centre. The fixed 150 px nearest-centre gate then matched a
  near detection to a map gate 40–160 m downrange (fix error ≈ the inter-gate distance, p50 16 m).
  Signature: solved PnP depth 2–35 m vs associated-gate range 40–160 m.
- **"Depth flips" (56):** NOT IPPE branch flips. They are detections whose apparent scale implies a
  depth 1.2–13× (p50 2.3×) the true range — close-range partial/garbage boxes and small spurious
  squares — solved cleanly (reproj p50 1.0 px!) but at the wrong metric scale. The characterize
  harness was ALREADY feeding the predicted prior, so prior-based flip disambiguation was already at
  its ceiling in that baseline; reprojection quality cannot catch these. Geometry consistency can.
- Good fixes agree with the map+given-pose prediction in depth to ~3% of range (p90 11%) — a huge,
  clean separation to cut on. **Overfit GEOMETRY, randomize APPEARANCE** — this is the geometry half.

## The fix (3 layers, all in `src/racer/vision/association.py`, shared by navigator + harness)
1. **`predict_gates_in_camera`** — every map gate projected through the state prior (KF pos + given
   attitude): predicted pose, centre AND the 4 predicted corner pixels (the apparent shape).
2. **`associate` (replaces nearest-centre):** a detection matches a gate only if the apparent-size
   ratio is within [1/1.6, 1.6] and the centre offset is ≤2.5 predicted-size units (scale-NORMALISED
   — a fixed pixel gate is meaningless across 2–160 m candidates); score = normalised centre distance
   + 2·|log size-ratio|. A far gate predicts a tiny shape → a near detection can no longer match it;
   junk that matches no gate's shape produces NO fix. This also RE-associates the near detections
   that used to lock onto background gates → good<1m fixes went UP 103→107.
3. **`range_consistent` (post-PnP depth sanity, the known-gate-size lever):** reject a fix whose
   solved depth disagrees with the predicted range to the associated gate by more than
   max(1.0 m, 0.15·range). Sized from the data (`reltol_sizing.py`): 0.15 costs ZERO sub-metre fixes,
   drops 10 of the 12 remaining catastrophic, leak 5→2; 0.25 left the long-range depth-noise tail
   through; 0.10 starts eating the good-fix noise band. The 1 m floor keeps a ~1 m VQ2 prior error
   harmless at close range.

**Flip disambiguation:** the navigator now ALWAYS passes the fresh map+attitude+KF-predicted pose as
the PnP prior (IPPE 2-fold tie-break + P3P selection). It previously preferred the LAST POSE ESTIMATE
(`_gate_priors`) — a sticky-flip bug: one accepted flip endorsed the next. The fresh prediction is the
motion/temporal consistency term (the KF propagates it between frames); the predicted ROTATION from
the given attitude breaks the frontal ambiguity (unit-tested across noise seeds); the known gate size
backs it with the depth sanity. The χ²₀.₉₉₉ innovation gate stays downstream as the final backstop.

### Code changes (committed to main)
- `src/racer/vision/association.py` — NEW: the 3 layers above + tunables, sized from course data.
- `src/racer/navigator.py` — uses the shared association; fresh predicted prior (sticky `_gate_priors`
  removed); depth sanity before the world fix; `NavigatorConfig.assoc_max_px` REPLACED by
  `assoc_max_size_ratio` / `assoc_max_center_units` + `fix_range_rel_tol` / `fix_range_abs_tol_m`;
  `_VisionDiag.n_rejected_range` counts depth-sanity drops.
- `scripts/characterize_perception.py` — now imports the SAME association/verify code (no more
  hand-copied logic), `--naive-assoc` reproduces the retired baseline, report adds DEPTH-SANITY +
  OFFERED accounting and computes the GATE TRADE-OFF on offered fixes (+ explicit RESIDUAL LEAK line).
- `tests/test_association.py` — NEW, 12 tests: collinear-course association by apparent size,
  wrong-scale refusal, scale-normalised centre gating (the 150 px case), 3-corner like-for-like,
  mid-transit skip, depth-sanity bounds, frontal-IPPE disambiguation by the predicted prior across
  noise seeds, navigator-level wrong-depth rejection.
- `tests/test_navigator.py` — the old innovation-gate test now splits into association-refusal vs a
  genuinely geometry-consistent-but-state-inconsistent fix (χ² still exercised + still anchors).

## The 2 residual leaked fixes (honest accounting)
`g1→g2 @38 m (|fix| 4.9, depth err −4.9)` and `g2→g3 @25 m (|fix| 3.0, −2.9)`: correctly-associated
NEXT-gate fixes at long range whose depth noise is just inside the 0.15 band — the genuine PnP
depth-noise tail (1.5 m gate ≈ 13 px at 38 m), not an association/flip failure. Their real issue is
the analytic cov UNDERSTATING long-range depth noise (maha 3–11 ⇒ they pass χ²); the honest-covariance
work (attitude_noise_std / cov modelling) is the right lever if 0.6%-of-frames must shrink further.

## K = PNP_FIX_COV_INFLATION note (ties off the prior session's open item)
The 2026-06-09 cov sweep (handoff/shadowpc-cov-finishhold-verify-2026-06-09) recommended NOT shipping
K=2.0 because it pushed leak 1.6%→2.2% for little good-fix gain. **With the depth sanity in place that
penalty vanishes:** K=1.0 and K=2.0 both leak the SAME 2 fixes; K=2.0 keeps one extra good fix (yield
89% vs 88%). The shipped K=2.0 default is now harmless-to-mildly-positive on this data — no rollback
needed. (χ² good-fix over-rejection itself is unchanged ~12% at <1 m — attitude-lever-dominated, as
that session diagnosed; separate lever, untouched here.)

## RL-twin perception-model update (Stage-2 input)
With the fixed front-end the measured in-loop model becomes: same noise floor (accepted |fix| p50
0.82 / p90 1.69 m, σ/bias per perception-char unchanged), **fix rate ~40% of frames in the ≤26 m
race window** (174 offered/360; the naive 312 "fixes" included the junk), **catastrophic leak ~0.6%
of frames / 1.1% of solved** (was 1.4% / 1.6%), leaked-fix magnitude now bounded ≈3–5 m (the 16 m
teleports are gone at the source).

## Repro
`commands.txt` has everything; headline = `run_measure.ps1` (naive baseline) + `run_measure_final.ps1`
(robust, final tolerances) + `aggregate_results.py` → `aggregate_stdout.txt` (committed). Per-frame
JSONs/stdouts are regenerable and gitignored. The pre-existing uncommitted race_outcome trio +
the 06-09 cov/finishhold handoff were left untouched (separate work).

## Surprises / for the commander
1. **The "56 frontal-PnP depth flips" were mostly NOT solver flips** — they're wrong-scale detector
   boxes solved at the wrong metric depth. Prior-based disambiguation could never have fixed them;
   geometry consistency does. The flip MECHANISM (sticky last-pose prior) was real in the navigator
   though — live flips were stickier than the harness showed; fixed.
2. Robust association is a net GOOD-fix generator (+4 sub-metre fixes): the near detections that used
   to lock onto background gates re-associate correctly.
3. The solved-fix count drops 312→189: those 123 rows were almost all junk that should never have
   been fixes. Detector detection rate is unchanged (349/360 frames).
4. K=2.0's leak penalty evaporates behind the depth sanity (see above) — the cov-inflation open item
   can close with K=2.0 kept.
5. Model-friction note (first run on this model): none blocking — the prior sessions' per-frame JSON
   dumps made it possible to design + size every threshold from data before touching code, which is
   why the first measured run already cleared the targets.
