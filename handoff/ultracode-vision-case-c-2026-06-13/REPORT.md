# Case-C (vision-only pose) readiness — synthesis & handoff

Fengyou — this is the synthesis-lead deliverable for the `ultracode-vision-case-c-2026-06-13`
workstream. It reconciles the A-audit (verified), the four adversarially-verified prototype pieces
(B rewind buffer, C range-anisotropic R, D in-loop latency, F registration/VISION-CAL), and the two
judge lenses (skeptic readiness + build prioritizer) into one decision document. Canary acknowledged.

Every load-bearing claim below was re-derived from the raw artifacts (not just trusted from verifier
prose); the spots where adversarial verification changed or weakened a prototype claim are flagged
explicitly. All prototype + verifier files are under
`handoff/ultracode-vision-case-c-2026-06-13/` and are cited inline.

---

## 1. Case-C readiness verdict + GO / NO-GO

### Verdict: GO-WITH-CONDITIONS

Case C is structurally sound, and the four pieces are real engineering — not vaporware. All four
prototypes reproduce their headline numbers, compose the real `LinearKF` / `localization` /
`gate_pose` code (no re-implemented physics), and survived independent adversarial verification with
only honest, non-fatal corrections. The track-map registration that case-C world-fixes depend on is
better than memory believed (see Section 4 — the gate-3 mis-registration scare is falsified).

But case C is NOT yet flight-ready, and the conditions below are HARD GATES, not nice-to-haves. The
single load-bearing operating point — true vision-only pose x VQ2 speed (15-30 m/s) x real eval-HW
latency — has never been exercised end-to-end on anything. Every coefficient that earns case C its
keep is extrapolated from one slow VQ1 recording (median 5.35 m/s, <=24 m range) on a CPU-only laptop
with no accelerator. One of the conditions (eval-HW detector latency) can flip the entire speed
conclusion to NO-GO if the eval box is CPU-class.

### The single biggest unresolved risk — UNMEASURED in-loop vision latency L

L (frame-in -> fix-applied compute time) is unmeasured on representative hardware, and it gates
everything downstream. Three separate failure modes hang off it:

1. The 67 ms datum is the WRONG number. 67 ms is command-side actuation latency. The vision
   capture-to-apply latency (frame transport + jpeg decode + detect + PnP + queue) was never measured.
   Piece D estimates 5-15 ms on an assumed-accelerated edge box, but this laptop is `torch+cpu` (CUDA
   unavailable, independently confirmed) where the detector runs ~112-139 ms. At that latency v*L =
   2.3-4.2 m at 20-30 m/s — which breaks the 10 m last-fix rule and inverts the whole case-C speed
   thesis.
2. The rewind buffer (piece B) catastrophically inverts if its horizon is sized below the true L.
   Piece B's own verifier (TEST 2) showed: horizon < L -> rewind DROPS every fix, dead-reckons, and
   diverges to ~21 m RMSE vs naive ~6 m — the "safety feature" becomes far worse than doing nothing.
   You cannot size the horizon without measuring L.
3. Predict-forward (the cheap piece-D fallback) needs a calibrated constant age, which is also L. Both
   compensation paths are blocked on the same measurement.

One ShadowPC/eval-HW recording with capture-and-apply timestamps resolves all three. Until it exists,
case C ships with an unbounded latency assumption.

### Scope of the GO

- GO for building the case-C pieces as offline-validated artifacts and for the cheap, low-risk
  integrations (predict-forward, the `_initialize` crutch fix, TIMESYNC).
- GO-WITH-CONDITIONS for the rewind buffer and range-anisotropic R, behind the L measurement.
- NO-GO at VQ2 speed if eval HW turns out to be CPU-class (the 2.3-4.2 m staleness eats the gate
  budget whole). This is a HW-conditional NO-GO, resolvable by one timing run.

---

## 2. Per-piece findings

Each piece states: what it claims, its validation result against the canonical VISION-PKG2 noise model
(world-fix sigma ~ [0.73,0.47,0.29] m, bias [-0.42,+0.06,-0.28] m, range-flat to ~24 m), and its
adversarial-verification verdict (what survived, what was corrected).

### Piece B — KF rewind buffer / OOSM (`kf_rewind_buffer.py`, `validate_rewind.py`)

Claim. A ring buffer of `(state, P, accel_body, R_wb, dt, sim_time_ns)` lets a late/out-of-order
vision fix rewind the KF to its capture time, apply `update_position` there, and re-propagate buffered
IMU forward — textbook out-of-sequence-measurement (OOSM) handling. Removes the v*L staleness bias the
naive in-place update injects.

Validation vs noise model. The headline reproduces: at 20 m/s, L=2 ticks -> naive 0.644 m bias driven
to 0.438 m (32% cut). The benefit is noise-model-independent (a timing fix), confirmed identical for
flat vs heavy-tailed t(3) fix noise (22.0% vs 22.1% cut). The RMSE-floor numbers (0.27/0.44/0.60 m)
are tied to the harness range-invariant flat fix-cov model and are NOT the true case-C floor (real
depth error is range-dependent — that is piece C's job); rewind correctness is unaffected because the
timing fix is orthogonal to the noise distribution.

Adversarial verdict: CONFIRMED (strengthened).
- Strongest independent proof (verifier TEST C): on a noise-free constant-velocity track, naive lags
  truth by exactly -v*L*dt (-0.444 m at L=2, -0.891 m at L=4, matching analytic to <0.3%) and rewind
  drives it to +0.000 m — a confound-free, deterministic demonstration that the win is genuine latency
  removal, not a synthetic-track artifact.
- OOSM exactness is bit-identical against a from-scratch in-order oracle even with mixed
  predict+pos+vel ops and time-varying attitude replayed (|dx|=|dP|=0). The wrapper forwards to the
  real `LinearKF` (no re-implemented filter physics).
- P stays SPD through repeated rewinds (Joseph form preserved; min eig >0, max|P-P^T| ~ 2e-17).
- Too-old fixes (older than the horizon) are correctly dropped; off-by-one boundary handling correct.

Three honest corrections (none fatal):
1. NEES is mildly UNDER-confident (~2.75 mean / ~2.16 median in an independent N=42000 MC vs chi2(3)
   mean 3.0), NOT "neither over- nor under-confident" as the docstring claims. Conservative is the
   safe direction; naive is the dangerous one (~8.9, 3.2x over-confident). Wording nit.
2. The SUMMARY-line "~0.03 ms/fix" is ~6x optimistic (a timeit fill-vs-fill artifact); direct
   steady-state timing is ~0.19 ms median / ~0.49 ms p95 — still ~170x under the 33 ms period, so
   "negligible" holds. The report's own `residual_cost` range (0.11-0.18 ms) is the honest figure.
3. Empirical accepted-fix bias [-0.44,+0.08,-0.26] matches canonical to mm; bulk/robust sigma
   (race-window MAD) [0.59,0.46,0.15] confirms the canonical model is representative-to-conservative.

SHARPEST RISK (sharper than the prototype conveyed): horizon < L -> total divergence (~21 m RMSE),
worse than naive. L is unmeasured. Horizon sizing is HARD-blocked on measuring L, and the buffer is
HARD-blocked on TIMESYNC (the capture timestamp must share the IMU master clock or the buffer is
indexed by a garbage epoch). Default 0.5 s horizon is safe ONLY if real L < ~0.45 s.

### Piece C — range-anisotropic measurement covariance R (`range_anisotropic_R.py`, `fit_range_R.py`, `validate_R.py`, `range_R_coeffs.json`)

Claim. Replace the partly-isotropic fix covariance with an analytic LOS-frame model: depth (radial)
variance grows as r^4 (std ~ r^2, the PnP Fisher law std(r)=2*sigma_px*r^2/(f*s*sqrt(N))), lateral
~ r, attitude lever tangential, isotropic floor. Coeffs: c2=0.003125 (depth, pinned to physics),
a1=0.026 (lateral, band-fit), sig0_rad=0.40, sig0_tan=0.282, sigma_theta=0.0244 rad (1.4 deg).

Validation vs noise model. Reproduces exactly: held-out NIS — LIVE-analytic 3.07 / CURRENT 2.94 /
iso-floor CONTROL 6.83 / NEW R_aniso 2.84. LOS-frame whitened variance NEW (1.31,0.40,0.77) vs control
(4.52,0.52,1.03). Per-axis off_ned bias [-0.39,+0.03,-0.30] matches canonical [-0.42,+0.06,-0.28]
(same dataset, correct sign: radial = -range_err, corr -0.9989). Held-out split is by source recording
(train {0,2,4} / test {1,3,5}), disjoint, train-vs-test NIS gap 0.47 — no overfit. Every residual-sigma
table value verified to <0.05 m.

Adversarial verdict: CONFIRMED as math, WEAKENED as a deliverable.
- The r^4 depth law is textbook-correct and MC-confirmed (independent fresh-seed slope 1.89). R_aniso
  is PSD on 5000 random geometries (0 failures); the largest-eigenvalue eigenvector at head-on = the
  LOS axis; reproj inflation monotonic. The model is numerically sound.
- Verification STRENGTHENED the report's own caveat into a near-disqualifier of in-range value: the
  SHIPPED analytic-Fisher localization cov (`gate_pose_to_world_position`) already carries the
  identical r^4 depth law at every range (depth_std@40m identical to R_aniso to 3 d.p.). So R_aniso is
  a re-parameterization with ~zero in-range win — its value is confined to case-C long range (>24 m,
  which the VQ1 recording never reaches) and robustness to off-nominal detector pixel noise.

Corrections / caveats the report soft-pedaled:
1. The lateral coefficient a1=0.026 is an 11x band-floor over the VQ1 range, NOT the physics r^1 Fisher
   law (~0.0023) — it EXTRAPOLATES BADLY to long range, the exact regime R_aniso is for. The depth
   axis extrapolates faithfully; the lateral axis does not. Asymmetry not called out.
2. The pinned c2=0.003125 is ~34% looser than the true MC Fisher value (~0.0023) — conservative (safe)
   but means the depth R is not tightly calibrated to the sim PnP.
3. R_aniso structurally CANNOT improve catastrophic depth-flip rejection through the chi2 gate — its
   12 leaked fixes (8.4% vs CURRENT 7.7%) are all long-range depth errors that leak because the
   correct depth law is loose at range. Upstream defenses (association + depth-sanity + S=P_pos+R)
   bound the live 0.53% leak; R_aniso is orthogonal to them.
4. off_ned carries a ~0.5 m systematic bias (map/height); NIS~3 is partly bias-absorption by the
   0.40 m floor, not a clean variance calibration. Ranking still fair (all models share the bias).
   Minor: the r^2 term overtakes the floor at r~11.3 m, not the report ~16 m.

Net: a bounded-threshold tweak (consistent with project doctrine that vision is a threshold, not the
differentiator), correctly de-prioritized. Do NOT integrate for VQ1/cases A-B — the live model is
provably equivalent in-range.

### Piece D — in-loop vision latency budget + compensation (`latency_harness.py`, `latency_design.md`, `latency_results.json`)

Claim. Measure the per-tick vision compute chain (detector -> PnP -> associate -> world-fix -> KF
update), keep it distinct from the 67 ms actuation latency and the content-offset Delta, and design
predict-forward vs rewind compensation.

Validation (measured numbers, HW-tagged).
- PnP->KF chain MEASURED on laptop CPU = 0.77 ms p50 / 0.90 ms p90 (PnP dominates ~85%; assoc /
  worldfix / KF each sub-0.1 ms). Tiny dense linear algebra -> ports to edge as a low-uncertainty upper
  bound. Negligible at any speed (0.83 ms x 30 m/s = 0.025 m).
- Detector CPU = ~112-139 ms (run-to-run), labelled UPPER BOUND (this laptop has no accelerator;
  cuda_available=False, torch 2.12.0+cpu). Reported only to bracket the top end.
- Detector edge = 5-15 ms ESTIMATE (yolo11s, 9.72 M params, 13.5 GFLOPs@640x384, ~100 TOPS box). FLOP
  floor 0.39-1.35 ms proves it is not compute-bound, but the empirical bracket (5-15 ms) is used for
  the budget. Explicitly an ESTIMATE, never measured on eval HW.
- L_total budget: edge ~ 6 ms p50 / 16 ms p90; CPU ~ 112-140 ms. v*L_total: edge 0.11-0.47 m at
  20-30 m/s (comparable to the gate-4 margin -> not ignorable, but an order of magnitude below CPU);
  CPU 2.3-4.2 m (catastrophic, breaks the 10 m rule).

Adversarial verdict: CONFIRMED.
- The three latency species (67 ms actuation / content-offset Delta / in-loop compute L_total) are
  kept properly distinct — fixing the conflation the A-audit flagged.
- v*L_total edge arithmetic verified to mm. The compensation analysis is correct: predict-forward
  removes the first-order v*age bias to ~0; rewind subsumes it; the residual gap is exactly
  0.5*a*age^2 (negligible <3 mm at edge, material ~0.2 m only at CPU latency) — independently
  KF-confirmed, not just trusted.
- Honesty contract intact: no fabricated GPU number; CPU never masquerades as eval-HW; PnP-chain timed
  on the live path (compute_covariance=True) on reprojection-valid corners.

Corrections / caveats:
1. The "predict-forward beats naive" magnitude needs nuance: it removes the first-order bias to ~0, but
   the realized posterior reduction at edge latency (16 ms) is modest (~0.42->0.39 m) because the
   v*age bias is comparable to the 0.40 m fix-noise floor; the win is dramatic only at large age. The
   bias removal is what matters and it is exact.
2. v*L_total slightly OVER-states the actual injected state error (KF gain <1 -> realized bias ~ 0.6x
   v*L_total) — conservative, favorable to the conclusion.
3. CPU absolute time is machine-load-dependent (re-runs 112-139 ms) — treat as an order-of-magnitude
   upper bound, not a fixed 130 ms. Edge numbers are deterministic.

The single load-bearing weakness (which the prototype flags loudest): the edge detector time is an
ESTIMATE on hardware that does not exist on this laptop. The CPU upper bound shows the entire case-C
speed conclusion flips if eval HW lacks an NPU/GPU. Get one real eval-HW detector timing. Also: the
content-offset Delta and raw transport/jpeg age are additive and NOT re-measured here — total v*age >
v*L_total.

### Piece F — track-map registration re-survey / VISION-CAL (`vision_cal.py`, `validate_registration.py`)

Claim. A robust per-gate median/MAD re-survey of the track-map world gate-centres from KF-accepted
fixes, recovering the global registration offset and per-gate residuals, plus a falsification test of
the memory claim that gate-3 is mis-registered ~1.46 m in D.

Validation vs noise model. Reproduces bit-for-bit: global offset [0.391,-0.054,0.309] =
-MEASURED_FIX_BIAS_NED on all 3 axes (reproduced 3 ways: flat-fix-mean, gate-equal-weight, median).
gate-3 D re-survey residual +0.346 m; gate-3 transit miss-vs-opening 0.017 m; across-gate sigma
[0.198,0.381,0.056]. Sign convention verified against source (`localization.py:86-87` lever sign,
`characterize_perception.py:189` off = pos_fix - drone).

Adversarial verdict: PARTIALLY-CONFIRMED — and the prototype was TOO CONSERVATIVE in a way that
improves the readiness picture. I re-ran the all-six-gate transit cross-check from the raw
`course_bundle/frames.json` against the live navigator loader (load_track_map(corner_to_center=True))
to confirm the load-bearing corrections independently (numbers in Section 4).

WHAT HOLDS (high confidence):
- SIGN is correct, not flipped (offset = -mean(off_ned) = -MEASURED_FIX_BIAS_NED, three ways).
- The gate-3 "1.46 m D mis-registration" (commit 82c2d20 / MEMORY.md) is FALSIFIED, non-circularly.
  The bundle range_m field IS the distance to the OPENING centre (independent of the fix chain). The
  drone physically crosses gate-3 ~0.056 m (3D) from the opening centre, PASS-CLEAN. The "1.46 m" was
  drone_D - record_BOTTOM_D ~ -1.36 m = the half-height lift of the bottom-referenced map
  position_ned vs the opening centre, measured against the wrong reference. Referenced to the opening
  centre (what flies), gate-3 D is off by ~0.02 m. This is an immediate memory correction.
- Depth-vs-registration disentanglement is sound: corr(range_err, off_N)=+0.998 (PnP depth bias maps
  onto N), off_D is depth-free, so the small D offsets (~0.3 m) are genuine but tiny vertical
  systematics.
- The prototype map_opening_centres is bit-identical to the live navigator loader, and off_ned is
  opening-referenced — no bottom-vs-opening contamination leaks into the re-survey.

WHERE THE PROTOTYPE IS WRONG / TOO CONSERVATIVE (the main finding):
- Its claim that gate-4 lateral E offset is "range-FLAT ~+0.5 m constant" (claim 4) is a polyfit
  artifact of far-range depth-flipped outliers. A clean fit (range <=20 m, outliers removed) gives a
  range-DEPENDENT ~+4 deg/range bearing/yaw bias (R^2~0.88-0.94), small at the transit-relevant short
  range (E@2m ~ -0.29 m). It is a global attitude calibration term, not a gate-4-specific constant map
  error. (The two judge lenses got opposite slope signs from different outlier filters; the
  load-bearing conclusion — range-DEPENDENT, modest at transit — is direction-independent.)
- Its open-risk "gates 4/5 have NO offline transit cross-check" is FALSE — course_bundle has clean
  transit frames for ALL 6 gates, and the gate-3-style discriminator resolves them favorably.
- Consequence: all 6 gates are in-plane-registered to ~0.06-0.37 m; gate-4 specifically to ~0.1 m. The
  inc8 gate-4 0.155 m geometric margin is therefore NOT threatened by a true map offset (Section 4).

Net: the robust re-survey tool is validated; ship-with-conditions stands. The case-C R-inflation
(~0.4 m lateral / 0.06 m vertical) is correct and conservative — and ~already covered by the shipped
FIX_COV_FLOOR_STD=0.40 m — but it guards the ESTIMATOR against a consistent fix bias, NOT against a
real map error (there isn't one).

---

## 3. Prioritized P0-P3 build plan

Framing constraint honored: vision is a bounded threshold; planning + speed is the differentiator.
Anything that is a re-parameterization with ~zero in-range benefit is de-prioritized; everything
expensive is gated behind one cheap measurement.

### P0 — BLOCKERS (case C is unshippable / silently wrong without these)

- P0-1 — Fix the `_initialize()` crutch + true cold-start path (effort: S, 1 file). `navigator.py:255-271`
  reads `ds.position_ned` with NO `use_given_position` guard. Flipping the config flags off does NOT
  produce a vision-only cold-start if the sim still streams LPN — you get a hidden ground-truth seed on
  tick 1, and every case-C "test" is secretly case A. True case C (position_ned=None) seeds origin at
  pos_std=5.0 (P[0,0]=25). This is the gate to even validating anything else. Depends on: nothing.
- P0-2 — Measure in-loop vision latency L on eval HW (effort: S — one ShadowPC recording with
  capture->apply timestamps). The single most load-bearing UNMEASURED number. 67 ms is ACTUATION, not
  vision. L gates P1 horizon-sizing, the predict-forward age constant, AND the "30 Hz binds only at
  >=30 m/s & last-fix <=10 m" speed conclusion. The CPU upper bound (2.3-4.2 m error) shows the verdict
  FLIPS if eval HW lacks an accelerator. Pair with P0-3 in one ShadowPC session. Depends on: nothing.
- P0-3 — TIMESYNC: reconcile video epoch <-> IMU epoch (effort: M). `frame.sim_time_ns` is the server
  UNIX epoch; `DroneState.sim_time_ns` is the IMU sim-boot epoch — distinct and unreconciled. Two
  consequences: (i) HARD prereq for the P1 rewind buffer (indexed by a garbage epoch otherwise); (ii)
  ALREADY corrupts `time_since_vision_update_s` (`navigator.py:426` subtracts the two epochs) — benign
  at VQ1 (tsv unused) but load-bearing in the case-C coast/abort policy. Depends on: nothing.

### P1 — HIGH VALUE for true case C at speed (prototypes built + verified)

- P1-1 — Ship predict-forward compensation (constant calibrated age) (effort: S — design done, D
  verified). Removes the first-order v*age staleness bias to ~0; residual gap to full rewind is exactly
  0.5*a*age^2 (<3 mm at edge L, ~0.2 m only at CPU L). Sidesteps TIMESYNC (uses a constant age, not
  capture timestamps) -> the cheap 80% of the rewind win. Ship this before the rewind buffer. Depends
  on: P0-2 (the calibrated age = measured L).
- P1-2 — KF rewind buffer / OOSM (piece B) (effort: S-integration — prototype verified bit-exact).
  Handles VARIABLE latency and out-of-order fixes exactly; benefit over P1-1 is variable/late fixes.
  HARD-blocked on P0-3 (TIMESYNC) and P0-2 (L to size horizon). SHARPEST RISK: horizon < L -> drops all
  fixes, diverges to ~21 m. Bank corrections: NEES mildly under-confident (~2.75, conservative); cost
  ~0.19 ms/fix (not 0.03), still ~170x under budget. Depends on: P0-3, P0-2, P1-1.

### P2 — MEDIUM, case-C long-range only

- P2-1 — Range-anisotropic R (piece C) (effort: S — prototype verified). NUANCE: the shipped
  analytic-Fisher cov ALREADY carries the r^4 depth law in-range -> in-range benefit ~ZERO. Value is
  case-C >24 m + robustness to off-nominal pixel noise. Caveats to carry: pinned c2 ~34% conservative
  (safe); calibrated lateral a1 is an 11x band-floor that EXTRAPOLATES BADLY past VQ1 range (depth axis
  OK, lateral not); R_aniso cannot improve long-range depth-flip rejection. Do NOT integrate for
  VQ1/cases A-B. Depends on: re-fit after VISION-CAL; a >24 m / higher-speed ShadowPC recording to
  confirm the depth law end-to-end.

### P3 — DEFER / do not build (over-building per directive)

- P3-1 — vision-velocity measurement channel for case C. Velocity is unobservable in true case C
  (vision is position-only; vel = IMU integration only). Real limitation, but the drift is bounded by
  P1; building a vision-velocity measurement is over-build. DEFER.
- P3-2 — per-gate registration coefficients in R. Across-gate residual is small and isotropic-floor
  covered (Section 4); a per-gate model is over-build. DEFER.
- P3-3 — `reset()` stale `_reset_counter` (`navigator.py:273-278`). One-line robustness note; works
  today via `_initialize` re-read. Fold into P0-1 if touching the file, else DEFER.

| item | priority | effort | depends_on | benefit |
|---|---|---|---|---|
| P0-1 fix `_initialize` crutch + true cold-start | P0 | S | — | unblocks ALL case-C validation (else every test is secretly case A) |
| P0-2 measure in-loop vision L on eval HW | P0 | S | — | gates horizon-sizing, predict-forward age, AND the speed thesis |
| P0-3 TIMESYNC epoch reconciliation | P0 | M | — | unblocks rewind buffer; fixes the case-C coast/abort tsv bug |
| P1-1 predict-forward (constant calibrated age) | P1 | S | P0-2 | removes first-order v*age bias (0.1-0.5 m @VQ2), zero buffer, no TIMESYNC |
| P1-2 KF rewind buffer / OOSM | P1 | S | P0-3, P0-2, P1-1 | exact variable/late-fix handling; horizon<L -> ~21 m divergence risk |
| P2-1 range-anisotropic R | P2 | S | P0-2 + >24 m recording | case-C long-range depth/lateral cov; ~zero in-range value |
| P3-1 vision-velocity channel | P3 | M | — | DEFER — over-build; drift bounded by P1 |
| P3-2 per-gate registration R coeffs | P3 | S | — | DEFER — over-build; floor already covers residual |
| P3-3 reset() `_reset_counter` note | P3 | XS | — | DEFER / fold into P0-1 |

---

## 4. Per-gate REGISTRATION verdict + residual sigma + inc8 de-provisionalization

I independently re-ran the all-six-gate transit cross-check from raw `course_bundle/frames.json`
against the live navigator loader (load_track_map(corner_to_center=True)). Closest-approach frame per
gate; in-plane miss = component of (drone - opening-centre) perpendicular to the gate normal;
inner_half = 0.75 m is the PASS-CLEAN threshold.

| gate | fid | 3D dist (m) | offset / in-plane miss (m) | axis | drone_D - record_bottom_D (m) | mis-registered? | recoverable by VISION-CAL? | residual sigma |
|---|---|---|---|---|---|---|---|---|
| 0 | 332 | 0.373 | 0.368 in-plane | N,E (lateral) | -1.345 | No (within inner-half) | Yes (global de-bias) | covered by 0.40 m floor |
| 1 | 475 | 0.134 | 0.062 in-plane | N,E | -1.393 | No | Yes | covered by 0.40 m floor |
| 2 | 647 | 0.120 | 0.070 in-plane | N,E | -1.404 | No | Yes | covered by 0.40 m floor |
| 3 | 877 | 0.056 | 0.018 in-plane (NOT 1.46 m) | D claim FALSIFIED | -1.377 | No — memory claim falsified | Yes (was never off) | covered by 0.40 m floor |
| 4 | 1020 | 0.103 | 0.096 in-plane | N,E (~0.1 m) | -1.293 | No (~0.1 m in-plane) | Yes | covered by 0.40 m floor |
| 5 | 1158 | 0.436 | 0.093 in-plane | N,E (0.43 m is along-track) | -1.301 | No | Yes | covered by 0.40 m floor |

Notes on the table:
- drone_D - record_bottom_D ~ -1.29 to -1.40 m for EVERY gate ~ the 1.36 m half-height lift. Gate-3
  value (-1.377) is NOT anomalous; it is the same uniform lift every gate shows. This is the decisive
  evidence that the "1.46 m gate-3 D mis-registration" is a reference-frame artifact (record-bottom vs
  opening-centre), not a true registration error.
- The ~0.4-0.5 m per-axis "offset" sometimes seen in fixes is a RANGE-DEPENDENT bearing/yaw FIX
  systematic (gate-4 E fit R^2~0.88-0.94), NOT a map error. It is modest at transit range and a 2nd
  calibration lap averages it down. PnP depth bias maps almost entirely onto N (corr +0.998); D is
  depth-free.
- Number-spread honesty: my independent in-plane misses (gate-3 0.018, gate-4 0.096) differ slightly
  from the judge lenses (gate-3 0.056, gate-4 0.077) because of which closest-approach frame /
  axial-vs-in-plane decomposition is used. The spread is sub-decimeter and direction-independent — all
  six gates are registered to well inside inner_half 0.75 m, and gate-3 "1.46 m" is falsified, under
  every decomposition.

### Residual registration sigma feeding the case-C R model

After global de-bias, the per-gate residual registration sigma ~ [0.21, 0.24, 0.03] m (N, E, D)
(max |resid| ~ [0.45, 0.34, 0.10] m). This is already covered by the shipped FIX_COV_FLOOR_STD = 0.40 m
isotropic floor — no new R term is needed for registration. The floor guards the estimator against a
consistent fix bias; it does not imply any real map error. (P3-2, per-gate R coefficients, is correctly
deferred.) The residual is recording-specific bearing wander that a 2nd calibration lap removes; the
global bias offset = -MEASURED_FIX_BIAS = [0.42, -0.08, 0.26] is fully recoverable by the validated
robust re-survey tool (`vision_cal.py`).

### Does this de-provisionalize the inc8 gate-4 0.155 m margin?

YES — w.r.t. map registration, offline, and that is the right scope. The inc8 gate-4 margin was flagged
"PROVISIONAL — shares registration risk with no live cross-check." That premise is now FALSE: an
offline GT-pose transit cross-check for gate-4 EXISTS (fid 1020) and resolves favorably — the clean
trajectory crosses ~0.10 m (3D) / ~0.096 m in-plane from the mapped opening centre, PASS-CLEAN. A true
>=0.5 m gate-4 map offset would force a PASS-CLEAN drone to cross ~0.5 m off-centre; it crosses ~0.1 m.
Gate-4 map is registered to ~0.1 m in-plane and the 0.155 m geometric margin is NOT eaten by a map
mis-registration. The "shared registration risk" with the (now-falsified) gate-3 offset is dissolved.

What this does NOT do (scoping):
- It de-provisionalizes the map-registration leg of "provisional" (the stated reason). It does NOT
  replace the inc8 live winner-validation fresh-reset batch (>=3-5 laps) — that guards the
  POLICY/physics-state question, which is orthogonal to map registration. Keep that rider.
- The transit argument bounds gate-4 in-plane registration to ~0.1 m (rules out a >=0.5 m offset,
  confirms PASS-CLEAN); it is not by itself a sub-0.155 m absolute-clearance proof, partly because in
  case A the drone aimed at the given map centre (crossing-near-centre is mildly self-fulfilling). The
  clean-pass + ~0.1 m GT-vs-map combination is what does the bounding. So: margin RANKING and "not
  threatened by a true map error" = de-provisionalized offline; absolute sub-0.155 m clearance under
  the live policy still wants the winner-validation batch.

Recommended memory action: correct commit 82c2d20 / MEMORY.md READ-FIRST (gate-3 1.46 m FALSIFIED),
re-label the inc8 gate-4 margin "registration-confirmed, awaiting live winner-validation" (drop "shares
registration risk"), and fold L-measurement + TIMESYNC + a 2nd cal lap into SHADOWPC-VISION-CAL.

---

## 5. What remains UNMEASURED / requires a live ShadowPC cross-check (escape-hatch)

### DONE (offline, this workstream)
- Piece B OOSM correctness + v*L removal (bit-exact + confound-free), SPD-preservation, drop logic.
- Piece C r^4 depth law (analytic + MC), PSD, held-out NIS, sign convention, no overfit.
- Piece D PnP->KF chain timing (0.77 ms, MEASURED on laptop CPU); compensation gap 0.5*a*age^2 (KF-confirmed).
- Piece F robust re-survey + sign convention; the all-six-gate transit cross-check (re-derived from raw
  data here) -> gate-3 1.46 m FALSIFIED, all gates registered <=0.37 m in-plane, gate-4 ~0.1 m.
- The A-audit ground truth: predict() runs on accel+given-attitude with pos/vel OFF; KF stays linear
  (attitude given); the dual-epoch clock does NOT corrupt the KF today (obs time never enters
  predict/update) but DOES corrupt `time_since_vision_update_s`.

### REMAINING — requires a live ShadowPC / eval-HW cross-check (bundle into SHADOWPC-VISION-CAL)
1. In-loop vision latency L on eval HW — capture-to-apply timestamps. THE binding measurement. Sizes
   the rewind horizon AND the predict-forward constant age. (67 ms is actuation, the wrong number.)
2. Eval-HW detector timing — is there an NPU/GPU? CPU-class = ~112-139 ms = the speed thesis flips. No
   measurement exists on any representative box. One timing run (or an organizer latency spec) resolves it.
3. TIMESYNC epoch reconciliation — video UNIX epoch vs IMU boot epoch. HARD prereq for the rewind
   buffer; also fixes the latent `navigator.py:426` coast/abort tsv bug. Build + verify vs a live wire trace.
4. True vision-only cold-start — `_initialize()` reads `ds.position_ned` regardless of
   `use_given_position`. The real case-C launch (position_ned=None -> origin, 5 m sigma) is a distinct,
   UNTESTED path. Exercise it end-to-end (the config-flag path is a hidden ground-truth crutch).
5. Velocity unobservability in true case C — vision is position-only; velocity = IMU integration of
   accel_body only. Accel bias integrates into both position AND the planner velocity. Never exercised.
6. The VQ2 15-30 m/s regime — every coefficient (depth law, fix sigma, latency) is extrapolated from a
   <=9.8 m/s recording. The physics is proven; the operating point is not.
7. Live cross-check of gates 4/5 absolute clearance under the inc8 winner — the offline transit check
   rules out a >~0.1 m in-plane map error but does not substitute for a fresh-reset live
   winner-validation batch at the inc8 flight envelope. (Orthogonal to map registration.)
8. Content-offset Delta + raw transport/jpeg frame age — additive to L_total; not re-measured here.
   Total fix age the compensation must cover = frame-age (Delta + transport) + L_total.

### Prototype + verifier files (all under `handoff/ultracode-vision-case-c-2026-06-13/`)
- Piece B: `kf_rewind_buffer.py`, `validate_rewind.py`; verifiers `verify_b_independent.py`,
  `verify_b_nees_and_data.py`, `verify_b_robustness.py`.
- Piece C: `range_anisotropic_R.py`, `fit_range_R.py`, `validate_R.py`, `range_R_coeffs.json`; verifier
  `verify_C_independent.py`.
- Piece D: `latency_harness.py`, `latency_design.md`, `latency_results.json`; verifier `verify_D_latency.py`.
- Piece F: `vision_cal.py`, `validate_registration.py`; verifier `verify_F_independent.py`.
- This report: `REPORT.md`.

---

## Confidence flags (honesty)

- HIGH confidence: gate-3 1.46 m falsification (re-derived from raw data here); all-six-gate
  registration <=0.37 m in-plane; piece-B OOSM exactness; piece-D PnP-chain timing; piece-C r^4 math.
- MEDIUM confidence: the inc8 gate-4 map-registration de-provisionalization (sound offline, but the
  absolute sub-0.155 m clearance still wants the live winner-validation batch).
- LOW confidence / extrapolated (did NOT survive as eval-HW fact): the edge detector 5-15 ms is an
  ESTIMATE; the VQ2 15-30 m/s coefficients are extrapolated from a <=9.8 m/s recording; L is
  unmeasured. The piece-C lateral coefficient a1 does NOT extrapolate to long range. These are the
  claims to treat as conditional until the ShadowPC/eval-HW cross-check in Section 5 lands.
