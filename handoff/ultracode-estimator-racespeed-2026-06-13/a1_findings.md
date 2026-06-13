# a1 — Closed-loop estimator sim: achieved filtered position 1-sigma at the gate-4 plane crossing

Agent a1 (estimator/perception, opus-4.8). Offline sim only. Composes the REAL
`racer.state_estimator.LinearKF` + the REAL `RewindKF` (kf_rewind_buffer.py) + the REAL
`racer.localization.gate_pose_to_world_position` fix-cov model. Residuals RESAMPLED from the
MEASURED `off_ned` pool (perception-char-2026-06-08), range-band conditioned.

Files: `a1_sim.py` (sweep), `a1_results.json` (36 cells), `a1_floor_probe.py` +
`a1_floor_probe.json` (the "can the floor come down" experiment). Reproducible: numpy RNG
seeded per-seed/per-cell; re-running reproduces every number below.

---

## HEADLINE (the core mission measurement)

**NO cell reaches <0.05 m filtered in-plane 1-sigma at the gate-4 ~37 m/s window — not the
claimed P, not the empirical RMS, in any latency / acceptance / error-model combination.**

Best achievable (const-v, edge L=6 ms, 47% acceptance ~14 Hz, DEBIASED — the most favourable
cell): filtered in-plane **1-sigma ≈ 0.23 m per axis (E 0.239 / D 0.239)**, in-plane RMS
**0.33 m**. That is **~5x over the 0.05 m bar and ~2x over the 0.155 m gate-4 contact-true
margin**. The RAW (bias-included) best cell is worse: in-plane RMS **0.46 m** (D-axis carries a
−0.33 m un-filterable bias).

**Verdict: ABSOLUTE world-fix localization (case C) CANNOT thread gate-4 at race speed on the
1-sigma criterion. The binding term is VARIANCE (the per-fix measurement floor), with a BIAS
floor on D making RAW additionally fail.** This confirms and quantifies the cross-cutting
reprioritization: the estimator, not the planner, is the binding VQ2 validity risk.

---

## What binds, and the variance-vs-bias split (the crux)

### (1) VARIANCE — the binding wall (DEBIASED arm = pure noise)
With the per-gate bias removed, the in-plane error is pure zero-mean scatter, and the filter
P is HONEST (NEES ≈ 3.3–4.3 vs the χ²(3) target 3.0; see below). The filtered in-plane
1-sigma is set by **how deeply the KF can average the ~0.50 m per-fix measurement floor**:

| acceptance / rate | DEBIASED filtered in-plane 1-sigma (E / D) | in-plane RMS |
|---|---|---|
| 47% ~14 Hz | 0.239 / 0.239 | 0.33 |
| 25% ~7 Hz  | 0.319 / 0.320 | 0.45 |
| 10% ~3 Hz  | 0.456 / 0.456 | 0.62 |

(edge L; CPU L barely shifts the DEBIASED rewind numbers — see latency section.)

**Why it floors at ~0.24 m and not lower:** the real `gate_pose_to_world_position` assigns
each accepted fix a per-axis 1-sigma of **~0.50 m** at gate-4 approach ranges (FIX_COV_FLOOR_STD
0.40 m ⊕ the 0.30 m default PnP block, growing with range via the 1.4° attitude lever:
0.50 m @ 2 m → 0.64 m @ 16 m). In case C **velocity is unobservable** (position-only vision),
so position information decays quickly between fixes and the KF averages only **~2–4 effective
fixes**. Empirically the filter scales as `filtered_sigma ≈ 0.49–0.65 × per_fix_sigma`.

**"Can the floor come down?" — the decisive probe (`a1_floor_probe.json`):** holding the best
cell and sweeping the per-fix isotropic 1-sigma down:

| per-fix 1-sigma | filtered in-plane 1-sigma @14 Hz | clears 0.05 m? |
|---|---|---|
| 0.50 m (realistic) | 0.323 | NO |
| 0.30 m | 0.226 | NO |
| 0.20 m | 0.175 | NO |
| 0.10 m | 0.098 | NO |
| 0.07 m | 0.075 | NO |
| **0.05 m** | **0.051** | NO (just misses) |
| **0.03 m** | **0.032** | YES |

To clear the 0.05 m in-plane bar at the realistic ~14 Hz the **per-fix accuracy must improve
~16x (0.50 m → ~0.03 m)**. Even at an idealized 30 Hz / 100% acceptance, per-fix must reach
~0.07 m (0.10 m per-fix → 0.065 m filtered, still over). **You cannot rate-average from 0.50 m
to 0.05 m; the per-fix accuracy itself must be near 0.03–0.07 m.** The 0.40 m floor is
explicitly BIAS-ABSORPTION (it covers the +0.3 m vertical, per-gate lateral, close-range depth
systematics) — taking it down without first removing those biases re-opens the chi² over-
rejection / catastrophic-leak problem it was added to fix. So the floor cannot simply be lowered.

### (2) BIAS floor — why RAW fails worse on D (un-filterable within one gate approach)
The measured per-fix bias [N,E,D] = **[−0.285, +0.064, −0.346] m** is a near-constant offset
within one track for one gate; a KF tracks it as real drift and does NOT average it out. In the
RAW arm the filtered estimate settles at bias ≈ **[−0.25, +0.01, −0.33] m** — the D-axis carries
a −0.33 m vertical systematic (the map opening-centre / camera-height offset) that is the single
largest in-plane contributor. RAW in-plane RMS (0.46–0.75 m) ≫ DEBIASED (0.33–0.62 m).
**De-bias (VISION-CAL global re-survey) is mandatory and removes the E/D bias to ~±0.02 m**
(DEBIASED filtered bias ≈ [+0.01, 0.00, −0.03] m). But de-bias only attacks the GLOBAL offset;
the case-C per-track residual registration sigma (~[0.21, 0.24, 0.03] m N,E,D across gates) is a
per-track constant that de-bias cannot touch lap-to-lap unless a 2nd calibration lap is run
(determinism is per-track — a 2nd-lap re-survey is the lever). Even then the residual VARIANCE
wall (1) remains binding.

### (3) Latency staleness — mostly along-track at gate-4; rewind is decisive only at CPU L
v·L is along the −N approach (along-track), so it perturbs WHEN the plane is crossed, not the
in-plane miss directly. Rewind-vs-naive in-plane RMS delta:

| L | rewind in-plane RMS | naive in-plane RMS | naive N (along-track) bias | naive NEES |
|---|---|---|---|---|
| 6 ms (edge p50)  | 0.33 (DE) | 0.34 | +0.41 m | 7.9 |
| 16 ms (edge p90) | 0.34 (DE) | 0.36 | +0.81 m | 18.3 |
| 112 ms (CPU UB)  | 0.35 (DE) | **0.83** | **+4.30 m** | **411** |

**At edge latency (6–16 ms) rewind barely matters** (Δ ≤ 0.03 m in-plane) — the staleness is
sub-resolution against the 0.24 m variance wall. **At CPU latency (112 ms) the naive arm is
catastrophic** (in-plane RMS 0.83 m, along-track bias +4.3 m = 37 m/s × 112 ms, NEES 411 =
wildly overconfident) while the rewind arm holds at ~0.35 m. So: **RewindKF is MANDATORY iff the
detector runs on CPU; on ~100 TOPS edge HW it is nearly free insurance.** (Buffer horizon
0.5 s > 112 ms — confirmed: mean dropped-fixes 0.1–0.5/run, i.e. only the very first pre-buffer
fix, never a horizon-driven divergence. If horizon < L the FACTS-flagged ~21 m divergence
would appear; it does not here.)

---

## NEES / honesty (is the filter P trustworthy?)

| arm / model | rewind NEES (target 3.0) | reading |
|---|---|---|
| DEBIASED, rewind | 3.27 – 4.28 | near-CONSISTENT, marginally overconfident |
| RAW, rewind | 4.06 – 7.54 | OVERCONFIDENT (P does not model the per-fix bias) |
| any model, naive @112 ms | 86 – 411 | BROKEN — do not trust naive P at CPU latency |

**The deployed path (rewind + de-biased fixes) has an HONEST covariance** at the gate-4 window
(NEES ~3.5–4.3): the claimed ~0.24 m 1-sigma is real, not a paper number. That honesty is what
makes the verdict load-bearing — the filter is correctly reporting that it is ~5x over the bar,
not hiding the error. The RAW arm's overconfidence (NEES up to 7.5) is the chi² flag that the
unmodeled per-fix bias is real error — another reason de-bias is mandatory.

---

## Assumptions — MEASURED vs ASSUMED (flagged)

- **Trajectory (ASSUMED):** const velocity 37 m/s along g3→g4 (24.4 m, 0.8 m descent ⇒ treated
  level), from track_map gate centres (MEASURED) + the post-gate-3 ~37 m/s speed (memory/plan
  ASSUMED — INC7 live cruise; not re-measured here). Sensitivity: a mild +5 m/s² along-track
  case shifts in-plane RMS by <0.02 m — the verdict is insensitive to the accel profile.
- **Attitude (ASSUMED, stated):** drag-hold pitch ≈ 38.4° (atan(0.21·37 / 9.807), linear_drag
  0.21/s MEASURED twin-fit), roll 0, yaw along segment. This is the realistic high-speed cruise
  posture and makes the predict-step attitude-skew injection non-trivial (worst-ish). A near-
  level 5° variant changes nothing material (position process-growth over a fix interval is
  only ~3 mm regardless — process noise is NOT the binding term).
- **IMU noise (MEASURED-as-configured):** accel_noise_std 0.3 m/s² (the KF's own value),
  attitude_noise_std 1.4° (MEASURED, frames.ATTITUDE_NOISE_STD_RAD). IMU at 90 Hz (ASSUMED,
  within the 75–97 Hz MEASURED band).
- **Fix stream (MEASURED):** off_ned residuals RESAMPLED from the 109 good fixes
  (|off|<3 m), range-band conditioned on `range_m` (the field that reproduces the FACTS per-band
  bias/std exactly: [0,8) n=37 bias[−.376,−.003,−.255] std[.99,.55,.58]; [8,16) n=60
  bias[−.177,+.017,−.422] std[.74,.57,.29]). Near the gate-4 crossing range→0 so [0,8) dominates.
- **Fix covariance (REAL model):** `gate_pose_to_world_position` with `covariance=None`
  (⇒ 0.30 m default PnP block ⊕ 1.4° attitude lever ⊕ 0.40 m floor). This is the shipped model.
- **Acceptance (MEASURED-anchored):** 47% canonical; 25%/10% are degraded-rate sensitivity rungs.
- **CASE CONDITIONALITY:** this whole measurement is the **case-C worst case** (vision-only
  pose). If VQ2 streams LOCAL_POSITION_NED/ODOMETRY (case A/B) the pose is pristine and this
  risk is MOOT. Verdict is conditional on organizer Q① (does VQ2 stream pose?).

### Honesty caveat — what is NOT measured
- The measured residual pool was recorded at **~5.35 m/s median, range ≤ 23.3 m** — there is
  **NO measured per-fix data at 37 m/s**. Motion blur, rolling-shutter, and detector latency-
  jitter at 37 m/s could WIDEN the per-fix noise beyond the resampled 0.50 m (this sim assumes
  the per-fix error distribution is speed-invariant — an OPTIMISTIC assumption). So 0.24 m
  filtered is a **best-case / lower bound**; real race-speed filtered sigma is ≥ this.
- In-loop latency L is an ESTIMATE (edge 6/16 ms never measured on eval HW; CPU 112 ms is a
  laptop upper bound). SHADOWPC-VISION-CAL must measure L in-loop.

---

## Implications / recommendation (for the commander)

1. **Absolute world-fix case-C cannot meet the 0.05 m in-plane 1-sigma bar at 37 m/s.** The
   binding term is the **per-fix VARIANCE floor (~0.50 m)**; a KF averages it to ~0.24 m, ~5x
   over the bar. Lowering the 0.40 m cov floor is not free (it is bias-absorption; needs the
   biases removed first) and even per-fix→0.05 m only barely clears.
2. **The FACTS-flagged highest-leverage fix is correct: switch to a GATE-RELATIVE observation**
   (close the loop on the SEEN gate-4 corners for centering) — that sidesteps BOTH the per-track
   absolute BIAS floor (2) AND much of the per-fix absolute VARIANCE, because PnP corner
   reprojection at close range is mm-pixel-tight and the quantity that matters for "thread the
   gate" is the drone position RELATIVE to the gate, not in absolute NED. This is the next probe
   I'd build (a2/b-class): re-run this sim with a gate-relative measurement model.
3. **De-bias (VISION-CAL) is mandatory** (removes the −0.33 m D bias; RAW is overconfident).
4. **RewindKF: mandatory on CPU detector (naive NEES 411, in-plane 0.83 m); ~free on edge HW.**
5. **Speed couples adversely:** inc8 cone-relax that raises speed will widen per-fix error
   (motion blur, unmeasured here) and shrink the gate-4 margin — verify the estimator at each
   speed rung, as the cross-cutting note already says.

---

## MEMORY-DELTA:
- a1 estimator-racespeed sim DONE: filtered in-plane 1-sigma at gate-4 37 m/s window = **0.24 m**
  best (DEBIASED, edge L, 47%/14Hz), in-plane RMS 0.33 m — **5x over 0.05 m bar; NO cell clears**
  (raw or debiased, any L/acceptance). Confirms ESTIMATOR is binding VQ2 risk.
- BINDING TERM = **VARIANCE** (per-fix ~0.50 m cov floor; KF averages only ~2–4 effective fixes
  since vel unobservable in case C). Need per-fix ~0.03 m (16x) to clear at 14Hz — cannot rate-
  average there. 0.40 m floor is bias-absorption, can't just lower it.
- BIAS floor: D-axis −0.33 m (RAW) un-filterable within one approach → de-bias MANDATORY.
- Filter P is HONEST in deployed path (rewind+debiased NEES 3.3–4.3); RAW overconfident (≤7.5);
  naive @112 ms CPU BROKEN (NEES 411, +4.3 m along-track bias).
- RewindKF MANDATORY on CPU detector, ~free on edge. Horizon 0.5 s > 112 ms confirmed safe.
- HIGHEST-LEVERAGE NEXT: gate-RELATIVE observation (sidesteps absolute bias+variance) — re-run
  this sim with a gate-relative measurement model. Caveat: per-fix pool is ~5 m/s data, 37 m/s
  blur UNMEASURED ⇒ 0.24 m is a best-case lower bound.
