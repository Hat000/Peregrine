# VQ2 A32 — Robust Estimation Spec: "Always Find Down" + Soft-Weighted Vision Fusion

**Date:** 2026-07-03 · **Branch:** `vq2-gate2-turn-dive` @ `f96962d` · **Evidence run:** `data/runs/20260703_172104_rl_s1_f1`
**Status:** BUILT + OFFLINE-GATE PASSED (this commit). §3.1(1)-(5) + §3.2 implemented, all
vq2_case_c-gated, flag-off byte-identity pinned against a pristine f96962d checkout
(tests/test_a32_robust_estimation.py + tests/data/vq2_a32_byteid_reference.npz). Offline gate
(§3.5.1, `handoff/tools/a32_eskf_replay.py --v2 [--imu-rate]`): the 20260703_172104 tail
recovers to 180 deg roll at t=28.34 s / final +175.4 deg (latest-sample) and t=28.41 s / final
+180.0 deg (imu-rate) — was pinned at ~150 deg forever. §3.3 (vertical balloon over gate 0)
is QUEUED — post-A32 tuning item, deliberately NOT in this pass.
**Operator directives honored:** (1) "We should ALWAYS be able to identify which way is down — every flight computer does it rock-solid." (2) "Don't throw away gates that don't match — take every measurement in and WEIGHT them. This has been solved; find the references."

---

## 0. Executive summary

**The AHRS root cause is NOT a missing accel-magnitude gate — we have three gates. It is that one of
them (the A8 `accel_motion_reject` covariance inflation) is UNBOUNDED and ATTITUDE-REFERENCED, so it
is a self-locking distrust loop:** once the attitude estimate is wrong, gravity no longer cancels in
its linear-acceleration residual, so the filter concludes "the accel is lying" *forever* and the
Kalman gain on the gravity correction goes to ~0. Replaying the flight's actual HIGHRES_IMU through
the deployed `ESKFAHRS` config (instrumented, this pass) shows: after the t=27.3 s contact the
estimate sits at ~150° roll while the accelerometer reads **|a| = 9.81 m/s² exactly** (the drone at
rest, inverted) and cleanly implies 180° roll — and every one of those accel updates is "accepted"
with a covariance inflation factor of **4,950–10,000×** (the 1e4 cap), i.e. accepted with zero
weight. The filter never comes back. Worse: even in *normal* powered flight the median inflation is
**671×** — the accel correction (and with it gyro-bias observability) is effectively OFF for the
whole flight, which is exactly the "cannot find down" the operator called out.

Secondary causes: (a) the AHRS is stepped at the ~18 Hz nav tick with the **latest** IMU sample ×
a ~55 ms dt while ~185 Hz of HIGHRES_IMU is on the wire — sample-and-hold aliasing injects the
initial large error at contact (a +45° single-tick attitude step at t=25.47); (b) the χ² gate is a
hard reject whose readmission time after a large error is minutes (P grows at only
1e-4 rad²/s), though in this config it never fires because the inflation already neutered S.

**The fix (§3.1)** is the proven pattern every production flight stack uses (Mahony 2008 explicit
complementary filter; PX4 EKF2 / ArduPilot accel weighting): keep the |a|≈g magnitude weighting,
make **every** rejection mechanism *bounded and soft*, give the accel correction a **floor** so its
pull toward gravity never reaches zero when |a|≈g, time-limit the gyro-anchored reference, and add a
**gravity-recovery watchdog** (the ESKF-native equivalent of PX4/ArduPilot's attitude reset): if the
accel steadily disagrees with the estimate by >25° while |a|≈g, bump P and re-level. Guaranteed
recovery < 1 s, by construction.

**The vision-fusion root cause (§2.3):** the A31 bearing gate hard-rejected **59.9%** of evaluated
frames this run (median deviation 9.85° vs median allowance 5.39°; the median rejected frame was
only **1.87×** over the line — information thrown away wholesale). The A28 vertical innovation gate
rejected 130 ticks with a hard 2 m cliff. **The fix (§3.2)** is standard robust estimation
(Huber/Cauchy M-estimator weighting, adaptive R on normalized innovation): every frame contributes,
scaled continuously by its consistency — the median "rejected" frame now contributes ~22% weight
instead of 0%.

---

## 1. Canonical methods + references (don't reinvent)

### 1.1 Gravity-referenced attitude estimation that always recovers "down"

The physics all of these lean on: the accelerometer measures specific force = f = a_kinematic − g.
Whenever the vehicle is not sustained-accelerating (|f| ≈ g), f points opposite gravity, so the
accel is a noisy but **unbiased, absolute** reference for "down." A correct filter integrates the
gyro at high frequency and applies a *never-zero* low-frequency pull toward the accel-implied down
whenever |a| ≈ g. That never-zero pull is the recovery guarantee; any mechanism that can drive it to
exactly zero for unbounded time forfeits the guarantee.

- **Mahony explicit complementary filter (ECF) on SO(3)** — Mahony, Hamel, Pflimlin, "Nonlinear
  Complementary Filters on the Special Orthogonal Group," *IEEE Trans. Automatic Control* 53(5),
  2008 ([HAL open copy](https://hal.science/hal-00488376/document);
  [reference implementation docs](https://ahrs.readthedocs.io/en/latest/filters/mahony.html)).
  Correction is `ω_corr = ω_gyro − b̂ + kP·e + kI·∫e`, with `e = v_meas × v̂_pred` (cross product of
  measured and predicted gravity direction in the body frame). **Almost-globally asymptotically
  stable**: from any initial error except the exact antipode, the estimate converges to the
  gravity reference with time constant ~1/kP. The kI integral term estimates gyro bias online.
  Two knobs. This is the filter in Pixhawk-class FCs (ArduPilot DCM fallback, Betaflight/SpeedyBee
  Mahony/`imuMahonyAHRSupdate`) that the operator correctly observes is "rock-solid no matter
  what" — its accel pull is *weighted*, never *switched off permanently*.
- **Madgwick gradient-descent filter** — S. Madgwick, "An efficient orientation filter for inertial
  and magnetic sensor arrays," Univ. Bristol tech report, 2010 / *IEEE ICORR* 2011. Same
  complementary structure, one gain β; equivalent recovery property.
- **Error-state / multiplicative EKF (ESKF/MEKF)** — J. Solà, "Quaternion kinematics for the
  error-state Kalman filter," arXiv:1711.02508, 2017 (the standard modern reference; our
  `eskf.py` follows it); Markley & Crassidis, *Fundamentals of Spacecraft Attitude Determination
  and Control* (MEKF). Statistically optimal, but the recovery guarantee is **not automatic** — it
  holds only if the accel update's effective gain stays bounded away from zero; naive robust
  add-ons (hard χ² gates, unbounded R inflation) can silently destroy it. That is precisely our bug.
- **PX4 ECL EKF2** — innovation consistency gating via `EKF2_*_GATE` (a Mahalanobis check in
  *standard deviations*, applied per-observation with recovery logic, not a permanent lockout) and
  explicit **state/attitude resets** when a fault persists (yaw resets, velocity resets, lane
  switching in ArduPilot EKF3). Docs: [Using PX4's Navigation Filter (EKF2)](https://docs.px4.io/main/en/advanced_config/tuning_the_ecl_ekf),
  [ArduPilot EKF2 overview](https://ardupilot.org/dev/docs/ekf2-estimation-system.html).
  The load-bearing pattern for us: **when a measurement stream is persistently inconsistent, the
  production answer is to RESET the offending state to the measurement, not to distrust the
  measurement forever.**
- **ArduPilot DCM/AHRS accel weighting** — `AP_AHRS_DCM` scales the accel correction gain by how
  far |a| is from 1 g (`_ra_scale`, "GPS-denied" accel trust) — a *bounded, continuous* weight in
  [w_min, 1], never a latch. Same idea as our Gaussian gate weight; the difference is the bound.

### 1.2 Robust measurement fusion — soft-weight, never hard-reject

The standard pattern: compute the **normalized innovation** ν = innovation / expected-1σ (or NIS
d² = yᵀS⁻¹y), then scale the measurement's influence *continuously* as a function of ν — by
inflating its covariance R (equivalently down-weighting its Kalman gain) — instead of a binary
accept/reject. A binary gate throws away 100% of the information in a measurement that is 1.05×
over the line; a robust weight keeps ~all of it and still suppresses a 10× outlier to ~1% influence.

- **M-estimators / Huber robust Kalman filtering** — Huber (1964); applied to KFs: Karlgaard &
  Schaub, "Huber-Based Divided Difference Filtering," *J. Guidance, Control & Dynamics* 30(3),
  2007 ([AIAA](https://arc.aiaa.org/doi/10.2514/1.27968)); Chang et al., "Robust derivative-free
  Kalman filter based on Huber's M-estimation methodology," 2012
  ([ResearchGate](https://www.researchgate.net/publication/259138260)). Huber weight:
  `w(ν) = 1 if |ν| ≤ k, else k/|ν|` (k ≈ 1.345 for 95% Gaussian efficiency). Implemented as
  R ← R/w (IRLS: iteratively reweighted least squares).
- **Cauchy / redescending weights** — `w(ν) = 1/(1 + (ν/c)²)`: heavier suppression of gross
  outliers, never exactly zero. The pragmatic choice when outliers can be 10σ+ (our ±8 m
  offset_z steps).
- **Adaptive / innovation-based covariance inflation (NIS χ² scaling)** — inflate R by
  `max(1, d²/χ²_thresh)` so a measurement exactly at the consistency boundary passes at full
  weight and worse ones degrade gracefully; survey of the family:
  [Adaptive Kalman Filter with Measurement Trust](https://www.emergentmind.com/topics/kalman-filter-with-adapting-measurement-trust).
- **Maximum-correntropy KF (MCKF)** — Chen et al. 2017; robust M-estimation MCC hybrids
  ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0019057822005547),
  [adaptive robust UKF via MCC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6112039/)): Gaussian-
  kernel weight `w = exp(−ν²/2σ_k²)` — same soft-weight family, information-theoretic derivation.
- **IMM (interacting multiple models)** — overkill here; noted for completeness. Our two regimes
  ("gravity reference valid" / "dynamic") are already handled by the |a|-weight.
- **PX4 gate precedent** — `EKF2_*_GATE` innovation gates exist, but each has bounded consequences
  (skip *this* sample) plus timeout-triggered resets; none can permanently sever a sensor.

**The pattern to copy, in one line:** `R_eff = R_base / w(ν)` with `w` continuous, `w > 0`
everywhere, and any hard consequence (reset/reseed) triggered by *persistence*, never by one sample.

---

## 2. Diagnosis of our actual code

### 2.1 What we fly (verified against `deploy_profile.py` vq2_case_c + `navigator.py`)

- Attitude: `ESKFAHRS` (`src/racer/ahrs/eskf.py`) wrapped by `AHRSAttitudeSource`
  (`src/racer/ahrs/ahrs_adapter.py`), owned by the Navigator behind `use_ahrs=True`
  (`navigator.py:594-604`), stepped in `_step_ahrs` (`navigator.py:700-728`).
- Flight config: `gyro_noise_std=0.01`, `accel_gate_alpha=10.0`, **`use_accel_motion_reject=True`**
  (`deploy_profile.py:163`), gyro sign `(-1,-1,-1)` applied at the wire
  (`mavlink_client.py:286`, `VQ2_GYRO_SIGN`).
- The ESKF accel update already has THREE protection mechanisms (`eskf.py:_update_accel`):
  1. free-fall/high-g **magnitude band** (skip when |a| outside [0.25 g, 10 g]) — hard skip, fine
     (a spike is transient; skipping is correct there);
  2. Gaussian **magnitude gate weight** `exp(-10·((|a|-g)/g)²)` dividing into R — soft, correct
     (this *is* the ArduPilot-style |a|≈g trust weight the task asks for — we have it);
  3. **χ² innovation gate** (hard skip at md > 7.815) — hard, lockout-capable;
  plus 4. the A8 **accel-motion inflation** `R ← R · (1 + (|a_lin|/0.1)²)`, capped at **1e4** —
  soft in form, but unbounded-in-effect and attitude-referenced. This one is the killer.

### 2.2 AHRS: why "down" is lost and never recovered (replay evidence)

Method: the run's `mavlink.tlog` HIGHRES_IMU stream (5,541 samples ≈ 185 Hz) was replayed through
the exact flight `ESKFAHRS` config at nav-tick cadence (latest-sample-and-hold, dt from consecutive
nav `sim_time_ns` — faithfully reproducing `_step_ahrs`), with the accel-update decision path
instrumented. Replay fidelity vs the flight log: pitch error median 0.35° (the replay IS the flight
filter); roll diverges only *after* the second contact — to +150° in the replay vs −146° in flight,
i.e. the post-spike integration is *chaotically sensitive* (finding F3). Scripts (checked in):
`handoff/tools/a32_eskf_replay.py` (the instrumented replay) and `handoff/tools/a32_log_analysis.py`
(nav_estimate timeline/gate stats). Run with the vq2yolo venv python, PYTHONUTF8=1.

Flight timeline (nav_estimate.jsonl): contact #1 at t=25.47 s (attitude steps **+45° pitch / +49°
roll in ONE 55 ms tick**), partial re-level by t≈26.1; contact #2 at t=27.33; from t=27.6 to the
end (2.4 s) the estimate is pinned at roll ≈ −146°, pitch ≈ −31°, thrust railed at 0.6, while the
vehicle is actually at rest — |pitch|>30° on 11% of all ticks.

**F1 — PRIMARY: the A8 motion-reject inflation is a self-locking distrust loop with no recovery
path.** `_accel_motion_inflation` (`eskf.py:412-438`) computes `a_lin = R_ref @ f + g_ned` and
inflates R by `1 + (|a_lin|/0.1)²` (cap 1e4). The residual is *attitude-referenced*: when the
attitude (hence `R_ref`) is wrong by angle θ, gravity fails to cancel and |a_lin| ≈ 2g·sin(θ/2)
**even for a perfectly clean, resting accelerometer**. At θ=150°, |a_lin| ≈ 1.9 g → inflation rails
at the 1e4 cap → Kalman gain ~0 → θ never shrinks → inflation stays railed. Replay, tail
(t=28.0–29.9, drone at rest inverted): |a| = 9.81 m/s² **exactly**, accel-implied roll = 180.0°,
estimate ≈ +150° — every update path says "accept", median inflation **4,950×**, and the estimate
moves only ~10° in 1.4 s (pure gyro drift, not correction). The operator's invariant — 1.5 s of
|a|=g pointing steadily at "down" MUST re-level the filter — is structurally violated.

**F2 — the inflation also disables the accel in NORMAL flight.** `accel_motion_scale=0.1 m/s²`
means any honest maneuvering (~2.5 m/s² of real lateral/vertical acceleration) inflates R by
~600×. Replay, pre-contact phase (0–25.4 s): median inflation **671×**. Consequences: (a) the
gravity pull is ~0 for the *whole flight*, so attitude is gyro dead-reckoning + occasional
near-hover corrections; (b) gyro-bias observability is starved (the bias state only learns through
the accel update); (c) the χ² gate is blinded — with R inflated, S is huge and md ≈ 0.1 always
(replay: **zero** χ² rejections in the entire flight — the gate designed to catch direction-liars
never fires because the inflation already swallowed it). Related: `accel_motion_anchor_thr=0.3
m/s²` means `R_ref` re-anchors to the live estimate only in near-perfect equilibrium — in flight
essentially never — so `R_ref` free-runs on the gyro indefinitely and its own accumulated drift
inflates |a_lin| further. Positive feedback, no time bound.

**F3 — 18 Hz sample-and-hold gyro integration (aliasing) injects the initial error.** The Navigator
ingests only the LATEST HIGHRES_IMU sample per ~55 ms nav tick (`_step_ahrs`; `DroneState` holds one
sample) — ~90% of the 185 Hz gyro stream is discarded. In smooth flight this is tolerable; through a
contact spike it integrates one instantaneous extreme rate sample across 55 ms: the +45°-in-one-tick
step at t=25.47 (a ~14 rad/s sample held for a tick), and the replay-vs-flight sign flip of the
post-contact roll divergence (+150° vs −146° from meV-level numeric differences) proves the spike
integration is aliased garbage, not tracking. The true rotation is unrecoverable from our sampling —
which is *fine* iff the gravity pull afterwards is alive (it is not, per F1).

**F4 — the χ² hard gate is independently lockout-capable (latent).** With the inflation fixed, a
large post-spike innovation meets a small P (P_φ grows at only `gyro_noise_std²=1e-4 rad²/s`); a
0.7-magnitude innovation over S ≈ 1e-2 gives md ≈ 44 ≫ 7.8 → reject, and P must grow ~60× before
readmission — **minutes**. Any hard gate + slow-growing P = the same "smug filter" failure mode.
It must become soft (§3.1) or persistence-bounded.

**F5 — gyro sign convention is CORRECT (A9 fix verified standing).** With `gyro_sign=(-1,-1,-1)`
the replayed filter tracks the flight log to 0.35° median and agrees with the accel-implied tilt
through all smooth-flight segments (e.g. t=25.40: est (5.7, −4.9)° vs accel (2.1, −1.1)°). No sign
regression.

**F6 — no accelerometer-bias state.** Minor on this wire (sim accel is clean: on-pad |f| = 9.810
exactly), noted for completeness; not a cause of this failure. Do not add state this pass.

### 2.3 Vision fusion: where we hard-reject

**A31 IMU-consistency bearing gate** (`gate_seeker.py:975-996` `_imu_bearing_consistent`, consumed
in the track-continuity filter at `gate_seeker.py:927-944`): binary `dev <= allow` with
`allow = 0.06 + 4.0·dt/range`. This run: **536 evaluated frames, 59.9% rejected**; deviation median
9.85°, p90 27.9°; allowance median 5.39°; the median rejected frame is only **1.87×** over the
line (p90 3.61×). Two compounding problems: (a) binary — a frame at 1.05× allowance carries nearly
full information and gets zero weight; at ~18 Hz with real per-frame bearing motion ~10°, most
honest frames die and the seeker starves (the A31 observed failure); (b) the allowance's rotation
compensation uses the AHRS attitude at capture time — with the AHRS accel-starved (F2), the
attitude error directly inflates `dev`, so **the AHRS defect masquerades as vision inconsistency
and the gate amplifies it**. Rejected frames route to `continuity_reject` → coast → track drop
after `track_max_coast_ticks` → re-acquisition churn.

**A28 vertical innovation gate + reseed** (`vertical_estimator.py:_zoff_filter_correct`,
lines 473-526): binary `|innov| <= innov_gate_m` (2 m); reject leaves the state untouched;
`reseed_after=4` consecutive rejects re-locks `z_off = z_meas`. This run: 380 accepted ticks, **130
rejected**, |innov| median 0.60 m but p90 **7.31 m** / max 13.6 m — the reseed fired repeatedly and
`z_off` teleported (visible as the ±5 m z_off swings in the log). A hard 2 m cliff means a 2.1 m
innovation (possibly honest after a blind stretch) contributes nothing while a 1.9 m one
contributes fully.

**Coupled tuning item — the vertical "balloon" (operator: over gate 1).** Confirmed cheap from this
log (gate 0 here, same mechanism): at t≈14.5 the estimator believed vz = +1.09 m/s (descending —
fiction) while z_off was already +1.0 (above the gate), so the damper term `+0.06·vz_lp` ADDED
thrust (0.311) while above the gate; upward momentum built; from t=16.5–18.5 thrust sat at its 0.150
floor (max ~4 m/s² net downward authority) for 2.0 s while z_off grew +3.2 → +5.7 and vz_est railed
at the −2.5 export clip (blind to the true climb magnitude). Diagnosis: momentum + railed/fictional
vz arming the damper the wrong way — an estimator-honesty + clip-tuning problem on the A28 loop, NOT
an attitude problem. Note also the +5.7 m z_off excursion is partly *measurement* drift (accepted
innovations walking z_off up during the same window) — the robust weighting of §3.2 addresses the
same root. Treat as a tuning item AFTER A32 lands (§3.3).

---

## 3. The fix design

### 3.1 AHRS (LEAD): bounded-trust ESKF + gravity-recovery watchdog
**Named method match:** Mahony-2008 weighted gravity pull (the never-zero correction) grafted into
our Solà-style ESKF, with PX4/ArduPilot-style bounded innovation gating and persistence-triggered
attitude reset. We KEEP the ESKF (its structure is right and bench-validated; the IEKF/EqVIO stubs
remain future work) and fix the four defects. All new behavior behind ONE new flag
`use_accel_trust_v2: bool = False` on `ESKFAHRS` (plus the navigator/profile plumbing), so the
bench cross-validation and every non-vq2 path are byte-identical.

**(1) Bound the motion-reject inflation and time-limit its reference** (kills F1/F2):
```
accel_motion_scale:        0.1  -> 1.0   m/s^2   # knee where R doubles: real maneuvers ~2-3 m/s^2
                                                 # now inflate 5-10x, not 400-1600x
accel_motion_max_inflate:  1e4  -> 25.0          # NEVER more than 25x distrust from this mechanism
accel_motion_anchor_thr:   0.3  -> 0.75  m/s^2   # re-anchor in realistic quasi-steady flight
accel_ref_max_freerun_s:   NEW  =  1.0   s       # R_ref free-run time bound: if not re-anchored
                                                 # within 1 s, force re-anchor to the live estimate
```
Rationale for the cap: with gate=1 (|a|≈g) and inflation at the 25× cap, R_eff = (0.3/9.81)²·25 ≈
2.3e-2 (unit-sphere units). The correction can be slowed, never severed. The free-run bound is the
"no permanent grudge" rule: `R_ref`'s job is to expose a *sustained* contamination window (the A8
signature, ~4 s); a reference that has free-run for >1 s is itself drifted and no longer evidence.

**(2) χ² gate → Huber-soft** (kills F4): replace the hard skip in `_update_accel` (and in
`update_yaw`) with NIS-scaled inflation:
```
md = innov' S^-1 innov                      # as today
if md > chi2_thresh:  R_meas *= (md / chi2_thresh)   # then redo S, K — one IRLS pass, no reject
```
One reweighting pass is the standard Huber-KF implementation (Karlgaard & Schaub); a measurement at
the boundary passes untouched, a 10× outlier is ~10×-damped, nothing is discarded, and readmission
is instant when consistency returns.

**(3) Accel-trust floor — the recovery guarantee** (the operator's invariant, made structural):
whenever the magnitude band says |a| ∈ [0.5 g, 1.5 g] *(tighten tol_lo from 0.75 for v2: the
gravity direction is meaningful well inside that band)*, the TOTAL effective down-weighting of the
accel tilt update is capped:
```
total_deweight = (1/gate_weight) * motion_inflate * huber_factor
total_deweight = min(total_deweight, deweight_cap = 100.0)
```
Recovery math at the cap: R_eff = 9.4e-4·100 = 9.4e-2; per-tick gain K ≈ P/(P+R). Even with P_φ at
its 1e-2 seed, K ≈ 0.096/tick at 18 Hz → error time constant ≈ 0.6 s. **"Down" recovers within ~1 s
of any spike no matter what the trust heuristics believe**, while a 100× deweight still keeps honest
high-g maneuver contamination to <1% influence per tick. This single clamp is the load-bearing line
of A32.

**(4) Gravity-recovery watchdog — attitude reset on persistent gross disagreement** (PX4/ArduPilot
reset pattern; belt-and-suspenders over (3)): track `theta_g = angle(a_hat, -g_hat_pred)` each
in-band tick (this is also NEW instrumentation → log it). If `theta_g > 25°` for `> 0.5 s` of
consecutive in-band ticks (10+ ticks at 18 Hz — a genuine maneuver cannot hold a *steady* 25°
gravity residual with |a|≈g for 0.5 s):
```
P[:3,:3] += diag( (20 deg)^2 )        # confess attitude ignorance (covariance bump, not a hard set)
R_ref     = R(level_seed_from_accel)  # re-anchor the motion-reject reference to the accel
watchdog cooldown 1.0 s
```
The next ordinary accel updates then close the error in 2–3 ticks (K ≈ 0.84 at the bumped P). Roll
and pitch only — yaw is untouched (unobservable from gravity; vision yaw path unchanged).

**(5) IMU-rate ingestion (fixes F3, separable sub-task, strongly recommended in the same pass):**
buffer ALL HIGHRES_IMU samples in `MavlinkClient` (ring, ~64 deep) instead of latest-only; the
Navigator drains the buffer each tick and steps the ESKF per-sample (dt from consecutive
`time_usec`). ~10 extra 6×6 filter steps per tick — trivial CPU. This kills the 55 ms spike
aliasing at the source AND makes the contact-tick attitude honest enough that (3)/(4) rarely have
work to do. It also upgrades `_rpy_at` capture-time attitude quality for the A31 gate. Keep
`level_seed_from_accel` seeding unchanged. (If schedule forces a cut, cut this LAST — (1)-(4) alone
already restore recovery; but this is the difference between "recovers from" and "barely gets
wrong".)

**Explicitly rejected alternative:** replacing the ESKF with a plain Mahony filter. The ESKF's
covariance machinery is what lets vision yaw (`update_yaw`), per-axis uncertainty reporting
(`yaw_uncertainty_rad`), and the alignment gating work; the failure was tuning/robustness policy,
not structure. We graft Mahony's *guarantee* (never-zero weighted gravity pull) into the ESKF
rather than discard the statistical framework.

### 3.2 Vision fusion: hard gates → robust soft weights

**A31 bearing gate → Cauchy-weighted track update** (`gate_seeker.py`). Keep the deviation and
allowance computation EXACTLY as-is (`dev`, `allow` — the physics is right); change the
*consequence*. New config `use_soft_bearing_weight: bool = False` (vq2_case_c sets True; replaces
the binary use of the gate, `use_imu_bearing_gate` machinery/instrumentation retained):
```
nu = dev / allow                       # normalized innovation (allow ≈ the honest-1σ scale)
w  = 1 / (1 + nu^2)                    # Cauchy weight: w(0)=1, w(1)=0.5, w(1.87)=0.22, w(4)=0.06
```
- Candidate selection: unchanged (rank by closeness to prediction).
- Track update: `alpha_eff = track_ema_alpha · w` for both range and bearing EMAs — every frame
  updates the track, wild ones barely. The A13 hold-last-demand bridge still covers w≈0 stretches,
  but "no usable pose" ticks now only occur when there are NO detections at all: **the
  `continuity_reject` starvation path (59.9% this run) is gone by construction.**
- Control consumption: the accepted pose this tick is the chosen candidate; downstream demands
  (image-servo lateral, vz_t latch) scale their per-tick step by the same `w` where they consume
  the fresh pose (image servo: multiply the az error term; z_off latch: pass `w` through — see
  below). One weight, computed once, logged once (`bearing_w` replaces/joins
  `bearing_dev_rad`/`bearing_allow_rad` instrumentation).
- Hard consequences only on persistence: keep the existing coast/track-drop counter but tick it
  ONLY when `w < 0.1` (ν > 3), i.e. a track drop still happens if every frame is grossly
  inconsistent for `track_max_coast_ticks` — that is a real track loss, not noise.
- Range-jump check: convert identically: `nu_r = |range - pred_r| / track_max_range_jump_m`,
  fold into the weight as `w = 1/(1 + nu^2 + nu_r^2)` (one combined consistency weight).

**A28 vertical innovation gate → Huber-weighted alpha-beta correction**
(`vertical_estimator.py:_zoff_filter_correct`). New config `use_soft_innov_weight: bool = False`
(vq2_case_c True):
```
sigma_z = 0.7 m                        # honest pose sigma incl. latency comp (median innov 0.60 m)
nu = |innov| / sigma_z
w  = 1            if nu <= 2           #  <=1.4 m: full weight (today's typical innovations)
w  = 2 / nu       if nu >  2           #  Huber tail: 4 m innov -> w=0.35, 8 m -> w=0.18, never 0
z_off  += zoff_alpha * w * innov
vz     -= zoff_beta  * w * innov / dt_div
```
- Reseed logic RETAINED unchanged (a real gate handoff still needs re-lock): the miss counter
  increments when `nu > 4` (2.8 m at σ=0.7 — near-today's 2 m cliff), reseed at 4 consecutive as
  today. Difference: those 4 "miss" latches each still nudged the state by their (small) weight, so
  the filter degrades gracefully into the reseed instead of freezing then teleporting.
- Multiply `w` by the seeker's `bearing_w` for the same frame (a bearing-inconsistent frame's
  z_off measurement is suspect for the same reason) — one line, plumbed through `latch_offset`'s
  signature as an optional `weight: float = 1.0`.

### 3.3 Vertical-overshoot / balloon — coupled tuning item (do AFTER flight-checking A32)

Pre-registered hypothesis from §2.3: fictional +vz (post-spike / railed washout) adds thrust via the
damper while above the gate; momentum then carries the drone over on the 0.15 thrust floor. Expected
effect of A32 alone: the AHRS fixes make `a_up` honest (attitude error was corrupting the gravity
subtraction, F2), and §3.2 keeps the innovation stream flowing — the vz fiction shrinks at the
source. THEN tune, in order, only if the balloon persists: (a) `export_clip_mps` 2.5 → 3.5 (the
damper must SEE a 3 m/s climb to brake it); (b) asymmetric climb authority: cap commanded climb
`vz_t` (or thrust above hover in the gate-PD law) near a gate so momentum can't build (e.g. thrust
≤ hover+0.06 when z_off < −0.5); (c) revisit kp_gate/kd (currently 0.04/0.06, ωn=1.2 rad/s, ζ≈0.9
nominal — the *linear* design is fine; the failure is saturation, so fix saturations first). No code
this pass; logged here so the tuning session starts from the hypothesis, not from scratch.

### 3.4 Byte-identity plan

- `ESKFAHRS`: all §3.1 changes behind `use_accel_trust_v2=False` default (new fields; existing
  fields' defaults untouched). Watchdog + deweight cap + Huber-χ² + new bounds all inside that
  flag. Bench (`scripts/benches/ahrs_bench.py`, `tests/test_ahrs_bench.py::TestIEKF` cross-checks)
  runs the flag OFF ⇒ byte-identical; add a v2 bench variant.
- `MavlinkClient` IMU ring buffer: additive field; latest-sample behavior byte-identical for every
  consumer that doesn't drain it; only the case-C Navigator drains (behind `use_ahrs` +
  a new `ahrs_imu_rate_ingest` NavigatorConfig flag).
- `GateSeeker`: `use_soft_bearing_weight=False` default; when False the A31 binary path runs
  byte-identically. `VerticalEstimator`: `use_soft_innov_weight=False` default likewise.
- Only `deploy_profile.py` vq2_case_c flips: `use_accel_trust_v2=True`, `ahrs_imu_rate_ingest=True`,
  `use_soft_bearing_weight=True`, `use_soft_innov_weight=True`. VQ1/case-A: zero new objects, zero
  RNG, byte-identical (same discipline as use_ahrs/use_zoff_filter today).
- New instrumentation to `nav_estimate.jsonl` (additive keys, null off-path): `theta_g_deg`,
  `accel_deweight_total`, `ahrs_watchdog_fired`, `bearing_w`, `zoff_w`, `imu_samples_ingested`.

### 3.5 Pre-registered flight checks (define pass/fail BEFORE the flight)

1. **Down-recovery (the headline):** for every `contact_frozen` onset (and any |a|>20 m/s² spike):
   `theta_g_deg < 10°` within **1.0 s** of the spike end, on every event. Also: no tick outside
   ±30° roll/pitch estimate error lasting >0.5 s while |a|∈[0.85g, 1.15g] (proxy: theta_g).
   Offline gate before flying: replay `20260703_172104` tlog through v2 — the tail (t≥28.16,
   inverted at rest) must converge to roll 180°±10° by t≤29.2 (it currently sits at 150° forever).
2. **No starvation:** `bearing_w` mean > 0.5 over pursuit ticks; fraction of pursuit ticks with a
   usable pose (w>0.1) > 85% (was: 40.1% survived the binary gate); zero `continuity_reject`
   track-drops during continuous visibility of one gate.
3. **Vertical honesty / no balloon at gate 0:** z_off excursion above +3 m lasting >1 s: none
   during the gate-0 approach; thrust pinned at the floor >1.5 s continuous: none; `zoff_w` median
   > 0.8.
4. **No regression on the solved channels:** vertical calm (A28 operator check) still holds;
   startup: no attitude transient >10° in the first 2 s on the pad (watchdog must NOT fire on the
   tilted spawn — the level seed covers it).
5. **Byte-identity:** VQ1 smoke replay hash-identical with all flags default; existing AHRS bench
   suite green untouched.

---

## 4. Findings ledger (for the memory pass)

- ROOT CAUSE of "forgets which way is down": A8 `accel_motion_reject` unbounded attitude-referenced
  R-inflation (cap 1e4, hit; median 671× even pre-contact) → accel correction ≈ 0 always → no
  gravity pull, no recovery, gyro-bias starvation; χ² gate blinded (0 fires all flight).
- Gyro sign (-1,-1,-1) confirmed correct by replay (0.35° median pitch fidelity).
- 18 Hz latest-sample AHRS stepping discards ~90% of the 185 Hz IMU and aliases contact spikes
  (+45°/tick step; replay/flight diverge to opposite roll signs post-spike).
- A31 bearing gate measured 59.9% hard-reject; median rejected frame only 1.87× over allowance.
- A28 innov gate: 130 rejected ticks, innov p90 7.3 m, reseed-teleports.
- Fix = bounded trust + accel-gain floor + persistence watchdog (Mahony/PX4 pattern) + Cauchy/Huber
  soft weights; all vq2_case_c-gated.

## Sources

- Mahony, Hamel, Pflimlin, *Nonlinear Complementary Filters on the Special Orthogonal Group*, IEEE TAC 2008 — https://hal.science/hal-00488376/document
- Mahony filter reference implementation notes — https://ahrs.readthedocs.io/en/latest/filters/mahony.html
- Solà, *Quaternion kinematics for the error-state Kalman filter*, arXiv:1711.02508 (2017)
- PX4, *Using PX4's Navigation Filter (EKF2)* (innovation gates `EKF2_*_GATE`, resets) — https://docs.px4.io/main/en/advanced_config/tuning_the_ecl_ekf
- ArduPilot, *EKF2 Estimation System* — https://ardupilot.org/dev/docs/ekf2-estimation-system.html
- Karlgaard & Schaub, *Huber-Based Divided Difference Filtering*, JGCD 2007 — https://arc.aiaa.org/doi/10.2514/1.27968
- Chang et al., *Robust derivative-free Kalman filter based on Huber's M-estimation methodology* — https://www.researchgate.net/publication/259138260
- *Robust M-estimation-based maximum correntropy Kalman filter* — https://www.sciencedirect.com/science/article/abs/pii/S0019057822005547
- *Adaptive Robust UKF via Fading Factor and Maximum Correntropy Criterion* — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6112039/
- Adaptive measurement-trust KF survey — https://www.emergentmind.com/topics/kalman-filter-with-adapting-measurement-trust
