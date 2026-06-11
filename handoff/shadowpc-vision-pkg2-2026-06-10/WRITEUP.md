# Vision pkg-2: measured attitude/fix-covariance model (ShadowPC, 2026-06-10)

Repo @ `d05796e` pulled, work committed as `37e7ab1` (map-frame fix) + `1b7e753` (covariance
model) (hashes post-rebase onto the concurrent laptop S14 commits). venv `.venv` (torch CPU + ultralytics, unchanged). Task: replace the guessed
`attitude_noise_std = 1.0 deg` with a measured model — (1) split the published ~3.6 deg yaw
bias into fixed CALIBRATION vs noise, (2) re-derive `attitude_noise_std` from residuals, one
value, all three call sites. Data: the canonical 6/6 recording `20260607_194615_course_60s`
via the 6 per-gate bundles (`perception-char-2026-06-08/pg/course_g0..5`, 360 frames) +
`task2_frames` (gate-0, 2026-06-05 — a DIFFERENT flight, used as cross-validation).

## TL;DR

| course-level (6 bundles, chi2_0.999 gate) | BEFORE (1.0 deg, no floor) | AFTER (shipped) | target |
|---|---|---|---|
| good<1 m fixes rejected | 12/107 = **11.2%** | 0/105 = **0.0%** | — |
| good<3 m fixes rejected | 27/172 = **15.7%** | 2/172 = **1.2%** | meaningfully <17% ✅ |
| catastrophic leak (of 189 solved) | 2 = **1.06%** | 1 = **0.53%** | ≤1.1% ✅ (stretch ½-met) |
| fixes ACCEPTED by the KF | 147 | **171** (+16%) | yield UP ✅ |
| accepted-fix \|err\| p50 / p90 | 0.83 / 1.68 m | 0.85 / 1.72 m | (quality held) |

Per-gate good<1m rejection: g0 6/17→0, g1 1/14→0, g2 1/13→0, g3 0→0, g4 3/27→0, g5 1/19→0.
**Full suite green: 379 passed** (incl. 4 new tests; **412 passed** after the rebase onto the
concurrent laptop S14 commits — re-verified on the integrated tree). Given-pose navigation untouched
(`use_given_position` default True, `fly_vq1` unchanged, vision stays out-of-loop for VQ1);
`twin.py` / `rl_plant.py` / `diffaero_dynamics.py` not touched.

**What shipped** (all measured, none guessed):
1. `frames.ATTITUDE_NOISE_STD_RAD = 1.4 deg` — ONE constant now feeds all three former
   1.0-deg sites: `localization.gate_pose_to_world_position`/`apply_gate_pose_update`
   defaults, `NavigatorConfig.attitude_noise_std`, `LinearKF.attitude_noise_std`.
2. `localization.FIX_COV_FLOOR_STD = 0.40 m` — NEW isotropic floor on every vision fix
   covariance (the second term the data demanded; see below).
3. `gates_from_track_records` corner_to_center frame fix — the in-plane axes were 180 deg
   off vs the detector corner convention (a real calibration bug found en route; see below).
4. `NavigatorConfig.vision_max_range_m` 40 → 32 (kills the worst residual leak at the source,
   measured zero cost).
5. `scripts/characterize_perception.py`: `--attitude-noise-std-deg`, `--fix-cov-floor`,
   `--max-range-m` (mirrors the navigator cap; pass large to reproduce pre-cap baselines),
   `--dump-extras` (odo quat/euler, solved+predicted gate rotations, t_cam vectors, analytic
   world cov, corner px+conf per row → every analysis below re-runs offline, no detector).

## Sub-question 1 — split bias from noise: there is NO shippable fixed-rotation calibration

The ~3.6 deg story does not survive contact with per-frame data. What the investigation found,
in order (analysis scripts in this dir, all re-runnable):

1. **Global rotation fit** (`fit_calibration_quick.py`, off = [L]×(R e) linear LS, world vs
   body frame, ± constant): |e| ≈ 1–1.7 deg only, components flip sign with the good-fix
   threshold, and PER-GATE fits swing **−6…+8 deg with both signs** — not one rotation. The
   published "≈ −3.8…+4.1 deg" per-gate slopes were already both-signed; part of the 3.6 deg
   was the ALONG-TRACK depth-scale error read through a per-axis slope as if it were an angle,
   part was gate-0's leg-specific lateral bias (Task-2 only looked at gate 0).
2. **Offset-vs-rotation discriminator** (`offset_vs_rotation_discriminator.py`): in the lever
   frame, the VERT channel is a **constant +0.3 m (fix-high) offset, flat across 8–30 m** (a
   rotation would grow with range) — map opening-centre vs true centre, or a camera-height
   offset; not attitude. HORIZ pooled slope +0.2±0.4 deg ≈ no global yaw. ALONG: close-range
   (<8 m) depth-short bias ~+0.5 m, no global scale.
3. **Roll coupling** (`delta_vs_attitude_check.py`): the per-frame angular error δ is
   deterministically roll-coupled WITHIN the flight: δ_yaw ≈ **−0.52 deg/deg · roll, r² 0.87**
   (survives per-gate centering; body rates and bearing do NOT explain it; image-vs-attitude
   timing refuted by ω-regression r²≤0.02). A tilt-split composition fit
   (`composition_fit.py`: uptilt partially applied BEFORE roll, τ_pre ≈ 29 deg) nails it
   in-flight (residual r² 0.00, perp-RMS 0.37→0.15 m) — **but fails cross-validation**: on the
   2026-06-05 task2 flight the coupling has the OPPOSITE sign (+2.7 deg/deg) and the fitted
   correction makes that data WORSE. A constant image/telemetry latency (off = −Δ·v) also
   fails: the velocity matrix is far from −Δ·I and task2's own Δ flips sign (`latency_fit.py`).
   → The coupling is real but FLIGHT/TRAJECTORY-SPECIFIC (roll proxies something about each
   flight's approach geometry); shipping any of these "calibrations" would help one flight and
   hurt the next.
4. **The one true fixed calibration found — and it was 180 deg, not 3.6**: comparing PnP
   SOLVED gate rotations against the model-predicted ones (`rotation_error_analysis.py` /
   `convention_180_check.py`) showed |E| p50 **174 deg**, collapsing to 27 deg only under an
   in-plane 180 deg flip: the corner_to_center map frame built right/down from the raw gate
   quaternion columns, which are authored for the OPPOSITE facing. The position chain never
   noticed (4-corner translation is symmetric to in-plane 180) but the PnP disambiguation
   PRIOR was anti-aligned — the IPPE frontal tie-break and P3P branch selection were
   effectively random on real detections, and 3-corner association compared against the
   diagonally-opposite predicted corners. **Fixed** in `gates_from_track_records` (in-plane
   axes now from the approach-view `_frame_from_through` convention; `through_dir` and
   `mission._passed` invariant; convention test added). Population effect on this data: nil
   (verified — same 189 solved, same leaks) — the value is correctness of the prior-dependent
   paths (gate-transit P3P, genuinely-ambiguous frontal cases) and of any future use of the
   solved rotations.

**Conclusion:** the systematic error is constants + flight-specific wander, not a fixed chain
rotation. The honest treatment for the LIVE gate (and the RL twin) is a zero-mean covariance
model that COVERS them. That is sub-question 2.

## Sub-question 2 — the measured covariance model

Model per fix (what `gate_pose_to_world_position` now produces):

    cov = K·R_wc Σ_pnp R_wcᵀ + σ_θ²(|L|²I − LLᵀ) + σ_0²·I,   K = PNP_FIX_COV_INFLATION = 2.0

Joint MLE of (σ_θ, σ_0) over the 165 offered (depth-sane) non-catastrophic 4-corner fixes
(`mle_sigma_theta.py`, exact replication of the chain cov verified to 0 ulp against dumps):

    σ_θ = 1.37 deg,  σ_0 = 0.407 m   (shipped 1.4 deg / 0.40 m)
    2-term nll 0.090  vs  σ_θ-only 1.700 (which distorts to σ_θ = 3.21 deg)

The floor is NOT optional: the constant systematics dominate at short range where the lever
term vanishes and the analytic PnP cov is mm-tight — that combination is exactly why gate-0's
close fixes carried maha in the hundreds and 35% of its good fixes were rejected (the
"attitude-lever-dominated" diagnosis of 06-09 was right that PnP-K was the wrong lever, but
the missing term at short range is the FLOOR; the attitude term then covers the range-growing
remainder). A σ_θ-only fit would have shipped 3.2 deg — over-trusting close fixes AND
over-widening far ones.

`vision_max_range_m` 40→32: of the two residual leaks (the honest long-range depth tail,
maha 1.7 and 9.4 — wider covariance only makes them pass MORE comfortably, so the gate can
never catch them), the 38 m one (|fix| 4.9 m) is removed by the cap at zero measured cost
(0 good sub-metre fixes beyond 30 m in the data; no course point further than ~30 m from its
next gate). The 25.4 m one (|fix| 3.00 m — 1 cm over the catastrophic threshold, depth err
−2.9 m ≈ −12%) remains: catching it needs `fix_range_rel_tol` 0.15→~0.115, which starts
cutting the good-fix depth-noise band (p90 = 11%) — not taken, documented.

The KF predict-Q (`state_estimator`) uses the same 1.4 deg: the vision-chain measurement is an
upper bound on the pure given-attitude error, so the Q inflation is honest-to-conservative;
under VQ1's 5 cm given-position anchoring the effect on the estimate is nil (suite green,
given-pose tracking tests unchanged).

## Acceptance evidence

- BEFORE run reproduces the published assoc-flipfix numbers bit-for-bit at the old defaults
  (189 solved / 174 offered / 107 good<1m / 12 rejected / leak 2 = 1.06%) — `before_g*.stdout.txt`.
- AFTER (`after_g*.stdout.txt`, aggregate via `aggregate_before_after.py`): table at top.
  The χ² gate now keeps 171/173 offered fixes; rejection work has moved upstream into the
  geometry gates (association, depth sanity, range cap) — which is the intended architecture:
  hard geometric refusal at the source, the statistical gate as backstop (it still kills any
  multi-metre teleport: maha at 16.27 ≈ 4σ ≈ 2–3 m disagreement).
- task2 bundle on the new chain (`after_task2.stdout.txt`): 6/6 good fixes kept (0 rejected);
  its own long-range catastrophic (1) leaks there too — same depth-tail mechanism, N small.
- Full suite: **379 passed** (3 new localization tests incl. constants-consolidation guard +
  1 navigator convention test; innovation-gate test scenario moved 1 m→2.2 m because 1 m is
  ~2σ of the REAL chain noise and is now correctly kept — the gate still rejects genuine
  inconsistency, exercised at 2.2 m ≈ χ² 22).

## RL-twin perception-model update (Stage-2 input)

With the measured covariance: in the ≤26 m race window the chain now ACCEPTS ~47% of frames
(171/360, was ~41%) at unchanged quality (accepted |err| p50 0.85 / p90 1.72 m, per-axis
σ ≈ [0.7, 0.5, 0.3] m N/E/D as before); catastrophic leak 0.53% of solved (≈0.3% of frames),
magnitude now bounded ≈3 m (the 38 m/4.9 m event is capped out; 16 m teleports were already
gone). Twin fix-covariance should mirror: K=2 analytic + 1.4 deg lever + 0.40 m floor, 32 m cap.

## Caveats / follow-ups for the commander

1. **The +0.3 m-high vertical constant** is real on the canonical flight (flat 8–30 m) but its
   SOURCE is ambiguous: map opening-centre too high (corner_to_center lift = full half-height
   2.72/2 = 1.36 m vs true opening centre) vs a camera-above-origin lever offset. Fixing the
   map moves planner carrots (given-pose behavior!) so it was deliberately NOT touched; the
   floor covers it for vision. Disambiguation needs frames at varied attitude/height near one
   gate — worth doing before VQ2 if the 0.3 m matters there.
2. **The roll/trajectory-coupled wander** (±0.5 deg/deg within a flight, sign flips across
   flights) is the dominant residual angular term inside the 1.4 deg. Mechanism unidentified
   (mount/composition/latency/decode all refuted by cross-day transfer); re-measure on the
   NEXT fresh 6/6 recording (post sim-regression) — `composition_fit.py` runs as-is on new
   `--dump-extras` dumps; if τ_pre reproduces WITH SIGN across two flights, revisit shipping it.
3. The remaining leak is a borderline 3.0 m next-gate fix at 25 m (depth tail). Either accept
   (0.53%, VQ1-harmless, VQ2 χ² breathing-S still bounds it) or trade `fix_range_rel_tol`
   0.15→0.115 knowing it cuts into the good band.
4. The 180-deg frame fix changes predicted-corner IDENTITIES; anything downstream that starts
   consuming predicted corner order per-corner (none today beyond P3P subsetting) inherits the
   corrected convention.
5. Per-frame dumps (`*_g*.json`, ~1.4 MB total) are gitignored as regenerable; the stdouts +
   all analysis scripts are committed. The MLE/replay (`mle_sigma_theta.py`) accepts any
   `--dump-extras` dump pattern → future re-derivations are a few CPU-minutes, no detector.

## Repro

```
# instrumented runs (per gate g in 0..5; same for task2_frames):
.venv\Scripts\python.exe scripts\characterize_perception.py ^
  --bundle handoff\perception-char-2026-06-08\pg\course_g<G> ^
  --weights models\gate_yolo11s_curriculum_v2.pt ^
  --map handoff\shadowpc-firstcontact-2026-06-02\track_map.json ^
  --dump-extras --json handoff\shadowpc-vision-pkg2-2026-06-10\after_g<G>.json
# pre-pkg2 baseline equivalence: add --attitude-noise-std-deg 1.0 --fix-cov-floor 0 --max-range-m 999
# analyses (this dir): fit_calibration_quick / angular_error_decomposition /
#   offset_vs_rotation_discriminator / delta_vs_attitude_check / rotation_error_analysis /
#   convention_180_check / composition_fit / latency_fit / mle_sigma_theta /
#   aggregate_before_after / leak_range_check / postfix_analysis / ab_extrapolation
```

## Files

- `before_g*.stdout.txt` / `fixedframe_g*.stdout.txt` / `after_g*.stdout.txt` /
  `after_task2.stdout.txt` / `task2_fixedframe.stdout.txt` — the measured runs (JSONs regenerable).
- analysis scripts as listed above; `aggregate_before_after.py` prints the acceptance table.
- Source commits: `37e7ab1` (map frame), `1b7e753` (covariance model + harness + tests).
