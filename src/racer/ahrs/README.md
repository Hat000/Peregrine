# AHRS — self-attitude estimation for the VQ2 wire (T1 bench)

**Why this exists.** VQ2's competitive wire *blocks* the `ATTITUDE` MAVLink message
(VADR-TS-003 §9.3). The `LinearKF` in `state_estimator.py` trusts a *given* attitude and
therefore cannot run on the VQ2 wire. This package is the missing layer: estimate
orientation from `HIGHRES_IMU` (raw gyro + accel, confirmed available on the scored wire,
§4.3). This is the **#1 VQ2 frontier build**.

This is a **bench**, not yet a production estimator: it picks the filter and pins the
correctness invariants. Nothing here modifies the live `src/racer/*.py` stack.

## Filters

| filter | module | summary |
|---|---|---|
| **ESKF** | `eskf.py` | Error-State KF: gyro propagation + accel tilt update, two-gate high-g robustness (+ optional mag yaw). |
| **IEKF** | `iekf.py` | Left-invariant EKF on SO(3) (Barrau–Bonnabel / van Goor EqF line). Geometrically-consistent contender. |
| **Madgwick** | `classical.py` | Gradient-descent complementary filter (`beta`). |
| **Mahony** | `classical.py` | Nonlinear complementary filter on SO(3) (`kp`, `ki`). |

Metric: geodesic (angle-axis) attitude error, `metrics.geodesic_error_rad` —
`theta = 2·atan2(‖vec‖, |dot|)` with **`abs(dot)` for quaternion double-cover** and inputs
renormalised. Range `[0, π]`.

## Corrected high-g comparison (synthetic, 8 s, dt=5 ms, 200 Hz)

Geodesic attitude error **p90 (deg)**, after a 0.5 s warmup skip:

| scenario | ‖a‖ (g) | ESKF | IEKF | Madgwick | Mahony |
|---|---|---|---|---|---|
| STATIC_GRAVITY | 1.0 | **0.18** | **0.18** | 0.23 | 0.17 |
| CONSTANT_SPIN | 1.0 | **0.18** | **0.18** | 0.23 | 0.17 |
| ROLLING_MANEUVER | ~1.06 | **6.59** | 6.59 | 10.52 | 18.40 |
| HIGH_G_PULL | ~2.9 (max 4.5) | **6.53** | 6.53 | 27.01 | 82.28 |
| HIGH_G_RANDOM | ~2.2 (max 5.5) | **0.67** | 0.67 | 0.88 | 5.84 |

Reproduce: `python scripts/benches/ahrs_bench.py --duration 8`.

## Which filter wins, and **why**

**The model-based filters (ESKF / IEKF) win under high-g — but the mechanism is the
*innovation* gate, not magnitude gating.** This is the load-bearing finding and it was
*not* obvious:

- Classical Madgwick/Mahony feed the raw accelerometer into the tilt correction with no
  principled rejection. Under sustained high-g (`HIGH_G_PULL`, ‖a‖≫g) the specific-force
  vector points far from −g, so they tilt toward a wrong "down" and blow up (Mahony p90
  82°, Madgwick 27°).
- A **magnitude gate** alone — down-weighting accel when `‖a‖` deviates from g — is
  *insufficient*. On `HIGH_G_RANDOM`, **~43% of samples have ‖a‖ within 10% of g** while
  their *direction* is random-corrupted. They sail through the magnitude gate and poison
  the estimate (magnitude-gate-only ESKF p90 blew up to **67°**).
- The fix is a **direction-aware innovation gate**: a Mahalanobis / chi-square consistency
  test on the innovation, `innovᵀ S⁻¹ innov ≤ χ²(3, 0.95)=7.815`. This is the same
  `relinnov` χ² gate family already used in the C2 vision-estimator chain. With it, the
  ESKF/IEKF recover to **0.65°** on `HIGH_G_RANDOM` and beat both classicals everywhere.

Ablation on `HIGH_G_RANDOM` (p90, deg): no gates **55.9** · magnitude-only **66.7** ·
innovation-only **0.86** · both **0.65**. The innovation gate is the mechanism; the
magnitude gate adds a small complementary improvement on sustained pulls.

### ESKF vs IEKF: a deliberate null result

The IEKF **matches the ESKF to ~1e-11 deg on every scenario**, including fast recovery from
an 80° initial tilt error. This is expected and correct: for the *pure SO(3)
attitude-from-gravity* problem the left-invariant output Jacobian `H = +skew(R̂ᵀg)` and the
ESKF's `H = −skew(ĝ)` are algebraically equivalent (`ŷ = −ĝ` for gravity-down vs
specific-force-down). The IEKF's celebrated **consistency** advantage (no covariance
collapse under large excursions, Barrau–Bonnabel 2017) materialises on the **coupled
SE₂(3)/INS** problem (attitude **+ velocity + position**), not attitude-only. Carrying the
IEKF therefore (a) cross-validates the ESKF and (b) leaves the geometrically-consistent
propagation in place for the eventual extension to a full visual-inertial estimator.

**Recommendation:** ship the **ESKF** as the VQ2 AHRS (simplest correct filter that wins),
keep the **IEKF** as the growth path toward EqVIO/SE₂(3). Mahony is a poor choice for this
regime; Madgwick is acceptable on mild maneuvers but loses badly on sustained g.

## Bugs fixed making this bench trustworthy (debug log)

1. **Geodesic metric** — inputs were not renormalised, so a slightly off-norm but *correct*
   quaternion read ~0.7° instead of 0°. Now renormalises both inputs and uses the
   numerically-robust `2·atan2` form. (`abs(dot)` double-cover handling was already present;
   pinned by `test_q_and_neg_q_give_zero_distance`.)
2. **Mahony sign** — fed raw specific force `[0,0,−1]` against a gravity-down prediction
   `[0,0,+1]`, putting the filter at the unstable 180° equilibrium (~30° static error). Now
   negates: `v_meas = normalise(−accel)`. → 0.07° static.
3. **ESKF Jacobian scale** — `H` used `−skew(g_body)` (g-scaled, ~9.8×) against a *unit*
   innovation, miscalibrating the Kalman gain and corrupting attitude during rotation
   (15° on a gentle roll). Now `H = −skew(ĝ)` (unit gravity direction). → 0.07° static.
4. **ESKF high-g robustness** — added the chi-square innovation gate (above).

## Plugging in real twin / VQ2 IMU (later)

`imu_gen.IMUSequence` is the integration seam. Build one directly from recordings:

```python
seq = IMUSequence(
    t=time_s,            # (N,) seconds, resampled to constant dt
    q_wxyz_gt=q_gt,      # (N,4) body->world (w,x,y,z); None disables scoring
    gyro=gyro_frd,       # (N,3) rad/s body FRD
    accel=accel_frd,     # (N,3) m/s^2 body FRD specific force (rest = [0,0,-g])
    name="real_twin",
)
q_est = run_filter_on_sequence(ESKFAHRS(...), seq)
```

- Twin/VQ2 emit `HIGHRES_IMU` at ~200 Hz — resample to a constant `dt` first
  (`frame_residual_report.py`'s IMU stream is the right starting point).
- GT attitude for offline scoring comes from `ODOMETRY` via
  `true_attitude_from_odo_quat_wxyz` — **offline calibration only** (`ODOMETRY` is blocked on
  the VQ2 scored wire). Watch the **R_y(π) ODOMETRY quat conjugation** footgun.
- **Determinism angle:** VQ2's IMU is *deterministic*. The synthetic noise envelope here is
  for filter *selection*; final AHRS tuning will be against the *exact captured sequences*
  from the VQ2 load (not a noise model). Re-tune `accel_noise_std`, `accel_gate_alpha`, and
  `accel_chi2_thresh` against real captures once available.

## Next step: learned AHRS (needs training data)

The classical/model ceiling here is set by **gyro bias random walk** and the fundamental
2-DOF observability of gravity (yaw is unobservable from accel alone). The research path
beyond it is **learned gyro denoising / learned AHRS**:

- **RIANN** (Weber et al., a GRU that maps IMU → attitude directly, robust to accel
  disturbance) and **Brossard et al. 2020 "AI-IMU Dead-Reckoning"** (learned gyro
  denoising feeding a classical filter) are the two reference designs.
- Integration point: `ESKFAHRS._predict()` — replace the raw gyro with
  `omega_clean = denoiser(gyro_window)` before propagation. The filter structure is
  unchanged.
- **Blocker: training data.** This needs labelled IMU↔attitude sequences. Since VQ2's IMU
  is deterministic, the highest-value training set is *captured VQ2/twin sequences with
  ODOMETRY-derived GT attitude* (offline) — i.e. the same loader described above, used to
  build a dataset rather than to score. Defer until VQ2 loads and we can capture.

## References

- Barrau, Bonnabel (2017). *The Invariant Extended Kalman Filter as a Stable Observer.*
  IEEE TAC 62(4).
- van Goor, P. (2023). *Equivariant Filters for Visual Spatial Awareness.* ANU PhD thesis
  (EqVIO / EqF line; same family cited by the C2 vision-estimator design).
- Madgwick, Harrison, Vaidyanathan (2011). *Estimation of IMU and MARG orientation using a
  gradient descent algorithm.* IEEE ICORR.
- Mahony, Hamel, Pflimlin (2008). *Nonlinear complementary filters on the special
  orthogonal group.* IEEE TAC 53(5).
- Brossard, Barrau, Bonnabel (2020). *AI-IMU Dead-Reckoning.* IEEE T-IV.
- Solà (2017). *Quaternion kinematics for the error-state Kalman filter.* (ESKF reference.)
