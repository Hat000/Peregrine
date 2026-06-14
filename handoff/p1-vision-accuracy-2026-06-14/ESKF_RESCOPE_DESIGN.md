# ESKF RE-SCOPE — boresight goes offline; the ESKF estimates the observable IMU bias (P1-ESKF-RESCOPE)

**Fengyou — headline up front.** Per COWORK-2 the boresight is now an **OFFLINE EXTRINSIC BAKE**
(`frames.BoresightCorrection`, calib-v2), so I **DROP the online pitch-bias corrector** (a constant the
filter shouldn't chase — gauge-ambiguous + double-counts the bake; corroborated by p1-eskf-design's own
1.0° at the first gate-4 pass). Re-scoped, the ESKF earns its keep on the **time-varying dead-reckoning
bias** the *given-attitude* architecture leaves on the table. The honest, validated outcome:

- **GYRO-bias = DEAD state → NOT built.** Attitude is GIVEN (ODOMETRY quat), never gyro-integrated in the
  `[p,v]` LinearKF, so a gyro-bias has **zero coupling** to pos/vel and no measurement (confirmed: its
  posterior never leaves its prior). Any upstream gyro drift shows up as a slowly-tilting given-attitude
  → gravity leakage → **absorbed by the accel-bias state** below. No separate gyro state.
- **ACCEL-bias = LIVE, but only the observable 2-axis subspace.** Estimate body **lateral (Y) + vertical
  (Z)** accel bias — exactly the **gate-plane in-plane** axes the margin's "effective bias ≤ 0.6°"
  (g·sin = 0.103 m/s²) binds on. Validated honest (MC **NEES 1.97 / target 2, unbiased**); converges
  under budget in **~2 laps at fr0.07, ~1 lap at fr0.15+**. **DROP body along-track (X)**: it is
  **unobservable** (aliased with along-track velocity; MC NEES **18**, estimate stuck at the full injected
  bias) and maps to the gate-NORMAL (depth) — margin-irrelevant. This is the COWORK-2 "accel-bias stalls
  at sparse fixes" caution, localised: it's the *along-track* axis that stalls; the cross-track axes are
  fine.
- **WATCHDOG = kept, the robust primary deliverable.** A **read-only** tight-prior monitor on the baked
  boresight that FLAGS (warn/abort, never silently corrects) a mount shift / mis-bake — covering BOTH
  calib-v2 forms (angular = range-proportional residual; metric = range-flat residual). Mechanism is
  **ε-independent**; only its prior VALUE comes from the bake.

This is **DESIGN ONLY**: no `src/` edits, baseline `pytest` green (703 passed / 35 skipped), feature gated
**OFF** behind a new `use_imu_bias_eskf` (default False → inc7/VQ1/case-A byte-identical), the **20-dim obs
contract is NOT reopened**, **+L preserved**, and the bake's mean is **never re-applied** (no double-count).

> Net: boresight → offline (calib-v2). ESKF → **2-axis accel-bias (live, margin-relevant) + camera-pitch
> watchdog (read-only)**. Gyro → dropped (dead). The ESKF is no longer the boresight lever; it is the
> dead-reckoning-bias lever, which is a genuine, non-redundant job.

---

## 0. Scope, inputs, baseline

Re-scopes `origin/p1-eskf-design` (ESKF_DESIGN.md + scratch-eskf/) given COWORK-2's boresight change and
coordinates with `origin/p1-calib-v2` (CALIB_V2_DESIGN.md + the `proposed_calib_dualform.patch` that defines
`frames.BoresightCorrection`). COWORK-2 boresight memo located at `ae58507:handoff/cowork-2026-06-14/
boresight-calibration.md` and read in full. Baseline `pytest` from repo root on `claude/stoic-cray-33a854`
@8c28c84 = **703 passed, 35 skipped** (escape-hatch satisfied); I change no `src/`, so it stays green.

Scratch sims (no `src/`) under `handoff/p1-vision-accuracy-2026-06-14/scratch-eskf-rescope/`:
- `accelbias_realistic.py` — **the trustworthy instrument**: a continuous race trajectory, CONSTANT-rate
  fixes (fr·30 Hz), 90 Hz sub-stepped predict (production Q), per-axis observability + NEES + drift +
  gyro-dead. → `accelbias_realistic_console.txt` / `_results.json`.
- `accel_bias_observability.py` + `validate_accelbias.py` — the geometry-generator study (gyro-dead +
  illustration). **Superseded for the RATE verdict** by `accelbias_realistic.py`: the generator's
  inter-gate gaps are *geometric*, not rate-realistic, and accel-bias observability is acutely
  gap-sensitive (signal ∝ τ²). Kept for the gyro-dead confirmation and the H_β re-target.

---

## 1. DROP the online boresight corrector (and preserve the proven result, re-targeted)

**Rationale (three independent reasons, all pointing the same way):**
1. **It's a CONSTANT → belongs offline.** ε_vert ≈ 0.56° is a constant render-vs-decode convention offset.
   calib-v2 bakes −ε once into `R_camera_from_body` (angular) and/or the +L lever (metric). Zero runtime
   cost, exact. An online state to re-learn a known constant is strictly worse.
2. **Gauge ambiguity / double-count.** A constant pitch offset is unidentifiable from a slowly-varying
   attitude bias at sparse, narrow-bearing fixes; and if −ε is baked AND an unanchored pitch-bias state is
   free, the state drifts to re-absorb ε and the two corrections fight (COWORK-2 §3). calib-v2 removes the
   mean at the bake site, so the ESKF must **NOT** re-apply it (§5).
3. **The data already said so.** p1-eskf-design measured the online pitch corrector at **σ_β = 1.0°
   entering the first gate-4 pass (28 fixes), OVER the 0.6° budget** → STATIC-PRIMARY. Dropping the online
   corrector is consistent with that.

**The "estimate-not-inflate" result is PRESERVED and RE-TARGETED.** p1-eskf-design proved (bit-exact vs the
real `localization`): the boresight enters the **fix** as `H_β = −skew(L)·R_wc`, which is the **MEAN** of
the zero-mean lever-arm term `localization.py` already inflates for, `σ_θ²(|L|²I − LLᵀ) = skew(L)σ_θ²skew(L)ᵀ`.
That relationship is now documentation (the boresight is offline). The **same principle re-targets to the
PREDICT side**: `state_estimator.predict` inflates for the attitude error via
`attitude_noise_std²·(S@Sᵀ)`, `S = skew(specific_force_world)` (state_estimator.py:117-118). The
dead-reckoning bias is the **MEAN of that predict term** — `g·sinθ = skew(s)·δθ` for the tilt part, plus a
true accelerometer bias. The ESKF **estimates that mean** instead of merely inflating Q for it (the exact
"covariance inflation masks drift, it does not remove a systematic bias" rebuild the deferred note asked
for) — now on the **specific-force skew (predict)** rather than the **lever skew (fix)**.

---

## 2. The IMU-bias ESKF (what survives the re-scope)

### 2.1 GYRO-bias = DEAD state → not built (honest assessment)

The LinearKF state is `[p, v]` only; attitude is GIVEN via `frames.R_world_from_odo_quat_wxyz` and used in
`predict` (`a_world = R_wb @ accel_body + g`) and the fix lever — it is **never integrated from the gyro**.
A gyro-bias `b_g` therefore has **∂(p,v)/∂b_g = 0** (no F-coupling) and no measurement touches it. Building
it = a dead state whose covariance never moves. **Confirmed** (`accelbias_realistic.py` (5),
`accel_bias_observability.py` (C)): with a `b_g` block added, its posterior 1-σ stays at its 0.5°/s prior
(0.500 → 0.500/0.525) while the accel-bias shrinks. **Do not build it.** The only path by which an upstream
gyro bias matters — it slowly tilts the *given attitude* → gravity leaks into the predicted accel — is
**subsumed by the accel-bias state** (§2.2), which is the catch-all for the net body-frame specific-force
bias. No separate attitude-bias state in a given-attitude architecture.

### 2.2 ACCEL-bias = the live state — 2-axis observable subspace

**State / process / measurement (error-state augmentation behind the LinearKF/RewindKF interface):**
```
state   x = [ p(3), v(3), b_a(2) ]      b_a = BODY accel bias on the LATERAL (Y) + VERTICAL (Z) axes
predict  a_world = R_wb @ (accel_body - B@b_a) + g ,  B = R_camera... no: B = body-axis selector [e_y e_z] (3x2)
         F = [[I, dt I, -0.5 dt^2 R_wb B],[0, I, -dt R_wb B],[0,0,I]]    (EXACT const-accel discretisation)
         Q = blkdiag( Q_pv (unchanged production B accel_cov B^T) ,  q_ba^2 dt I_2 )   slow random walk
fix      z = p + nu ,  H = [ I3 | 0 | 0 ]      b_a observed THROUGH the predict coupling (aided-INS), not H
update   joint Kalman update -> estimates AND removes the in-plane dead-reckoning bias from the coast
given    given pos/vel updates: H = [I|0|0]; they do NOT inform b_a beyond the dynamics
```
- **Why body Y,Z and not 3-axis.** During a head-on gate approach body-X≡gate-normal (depth), body-Y≡gate
  in-plane horizontal (lateral), body-Z≡gate in-plane vertical. The MARGIN binds on the **in-plane**
  centering (Y,Z); the along-track (X) is depth, which the absolute fix + IMU own. And X is **unobservable**:
  a forward accel bias is degenerate with along-track velocity (both push along v̂), so estimating it is the
  classic accel-bias-vs-velocity ambiguity + the EKF spurious-info-gain trap COWORK-2 flagged.
- **Validated observability (`accelbias_realistic.py`, fr0.07, 6 laps, true b_a=[0.10,−0.08,0.05] m/s²):**

  | design | per-axis err_rms (m/s²) | NEES (target) | verdict |
  |---|---|---|---|
  | 3-axis X,Y,Z | **[0.103, 0.025, 0.005]** | **18.0** (3) | X unobservable + OVERCONFIDENT |
  | **2-axis Y,Z (recommended)** | **[0.025, 0.005]** | **1.97** (2) | observable, unbiased, **honest** |

- **Rate to budget** (2-axis, σ_inplane < 0.103 m/s², constant bias): fr0.07 → **lap 2**; fr0.15/fr0.25 →
  **lap 1**. (σ_inplane: fr0.07 1-lap 0.106 → 2-lap 0.049 → 6-lap 0.026.)
- **Drift tracking** (2-axis, fr0.07, slow random walk): q_ba ≤ 0.02 m/s²/√s keeps ss σ_inplane ≤ 0.083
  (< budget); q_ba = 0.05 → 0.155 (over). Recommend q_ba ≈ 0.005–0.01.
- **Margin/accuracy benefit (concrete).** The margin's binding term is the "effective attitude/accel bias
  ≤ 0.6° (g·sin ≤ 0.103 m/s²)" entering the terminal gate. The 2-axis accel-bias **is** that in-plane
  dead-reckoning bias; estimating+removing it drives the term to σ ≤ 0.03–0.05 m/s² from lap 2 onward. It
  is **distinct from the boresight** (a vision sighting bias, offline-baked) — it catches a *true*
  accelerometer bias AND a given-attitude *tilt* bias's gravity leakage. **Value is conditional**: (a) a
  real systematic accel/tilt bias must EXIST (data-dependent — pin it at first-contact by running the KF
  open-loop on recorded IMU vs GT velocity, the margin-closure §7.2 method), and (b) ~2 laps to converge
  at fr0.07 (a lap-2+ lever, not lap-1). If first-contact shows the dead-reckoning bias is already ≤ budget
  (good given attitude + small IMU bias), the accel-bias state adds little and the ESKF reduces to the
  watchdog — a clean simplification, reported as such.

---

## 3. The camera-pitch WATCHDOG (read-only, both calib-v2 forms)

**Purpose:** the boresight is baked (calib-v2), not estimated. The watchdog detects when the bake becomes
WRONG — a physical mount shift (crash/thermal) or a mis-bake — and FLAGS it, **without** continuously
estimating-and-correcting (that is exactly the gauge-ambiguity trap §1).

**Statistic (form-agnostic, reuses calib-v2's #16 discriminator as a monitor).** Over a sliding window of
accepted vision fixes, fit the gate-vertical fix residual (gate-down component of `z − p_pred`, i.e. the
innovation projected on `R_world_gate[:,1]`) against range:
```
    m_v(r) = a + b·r            a = intercept (METRIC residual, range-FLAT)   b = slope (ANGULAR residual, ∝ range)
```
held with a **tight prior at the baked value** (post-bake both a,b ≈ 0). A residual back-out angle
`θ_dev = −atan(b)` reuses the proven `H_β = −skew(L)·R_wc[:,0]` as the angular detection model (read-only).

**Both selected-form cases (per the calib-v2 coordination facts):**
- **Form = ANGULAR** (pitch baked in `R_camera_from_body`): a mount-pitch shift Δθ adds **slope b =
  −tan(Δθ)** (range-proportional). PRIMARY watch = `|θ_dev| = |atan(b)|`; flag at `> 0.6°` (the budget).
- **Form = METRIC** (vert_offset baked in the lever): a camera-translation shift Δz adds **intercept a = Δz**
  (range-FLAT, pixels ∝1/r). PRIMARY watch = `|a|`; flag at `> ~0.15 m` (beyond the 0.05 m acceptance).
- The watchdog runs BOTH a,b regardless (post-bake both ≈ 0); the selected form just names the PRIMARY. A
  shift of the *unexpected* type still trips the other parameter — and tells you which kind occurred.

**Thresholds + trigger (warn/abort, NEVER a silent correction):**
- **WARN** when the windowed `|θ_dev| > 0.6°` OR `|a| > 0.15 m` is sustained over ≥1 full lap (the window
  must span a lap to get the ≥~10 m range spread calib-v2 §identifiability requires; per-gate range is too
  narrow to fit the slope).
- **ABORT / mark-vision-untrusted** when ≥2–3 consecutive lap-windows trip, or a single window exceeds 2×
  budget (a gross knock). On abort the navigator falls back to the existing coast/innovation-gate behaviour
  (vision down-weighted) — it does not silently re-aim.
- **Honest limit:** at fr0.07 + narrow per-gate range the windowed slope is noisy (σ ~ several tenths of a
  degree per lap), so the watchdog reliably catches only **gross** shifts (≳1°, ≳0.15 m) — which is its job
  (detect a mount knock / mis-bake), NOT refine the calibration. Sub-budget drift stays within budget anyway.

---

## 4. Contract + non-regression (carried from p1-eskf-design)

- **obs_dim STAYS 20 — not reopened.** `b_a` is internal; its effect on the obs is (1) a bias-corrected
  `p_est/v_est` through the existing `estimator_obs` seam into bit-exact `[0:17]`, and (2) a tightened/honest
  `σ_hat` via the existing `_gate_frame_pos_sigma` projection of the augmented `P[:3,:3]` into `[17:19]`. No
  new slot. The watchdog flag is diagnostics/telemetry, **off the NavState contract**.
- **+L preserved.** We touch the estimator's internal fix handling, never `obs_from_zup`/`build_obs`. Flag
  OFF ⇒ byte-identical; flag ON with `b_a = 0` ⇒ fix unchanged ⇒ `tests/test_obs_sign_faithfulness.py`
  (+L 4.8e-7 / −L 24 m) intact. A new test pins +L under a non-zero `b_a` estimate.
- **OFF-by-default flag `use_imu_bias_eskf` (default False)**, same discipline as
  `use_rewind_kf`/`use_gate_relative` ⇒ **inc7/VQ1/case-A byte-identical** (pinned by a test mirroring
  `test_vq1_path_unchanged_when_c2_off`).
- **Touch-points (re-used from p1-eskf-design, updated for accel-bias):**
  `state_estimator.py:155` (`velocity` → `x[3:6]`), `:135` (Joseph `np.eye(6)` → `np.eye(self.dim)`),
  `:179` (`make_nav_state` `pos_vel_covariance = kf.P[:6,:6].copy()`); `kf_rewind.py:281` (replay carries
  the accel-bias predict coupling — no new fix-op kind needed since `b_a` enters PREDICT, not the fix H, so
  the **existing** predict replay already propagates it once `predict` records `R_wb`, which it does at
  kf_rewind.py:177-197); `navigator.py:500/538` (fix routing UNCHANGED — `b_a` needs no per-fix Jacobian).
  Net: the accel-bias is LESS invasive than the boresight would have been — it rides the existing predict
  replay and the existing position-fix H.

---

## 5. Interface with calib-v2 `frames.BoresightCorrection` (no double-count)

- **The ESKF consumes the bake as a FIXED extrinsic, never re-applies it.** The angular mean is already in
  `R_camera_from_body()` (so the ESKF's `R_wc = R_wb @ R_camera_from_body().T` inherits it) and the metric
  mean is in the +L lever (so the fix `z` is already de-metric-biased). The ESKF reads `frames.BORESIGHT`
  **only** to set the WATCHDOG's tight prior: **mean 0** about the baked value, **width = residual
  uncertainty**. It does **not** add the correction to `R_wc` or `z` (that double-counts — calib-v2 §6.1).
- **The accel-bias is ORTHOGONAL to the bake** (different physics + different range-signature): the bake is
  a vision *sighting* term (range-proportional if angular, range-flat if metric); the accel-bias is a
  *dead-reckoning* term in the predict. No aliasing. (Consistent with the ESKF §3.4 bias-vs-map
  separability corr 0.004 in the prior design.)
- **Form-choice is load-bearing for the WATCHDOG, not the accel-bias.** The accel-bias is rotational/dynamic
  and independent of whether the boresight is angular or metric. The WATCHDOG's signature DOES depend on it
  (§3) — angular shift = range-proportional, metric shift = range-flat — which is why it monitors both a,b.
- **Interface request UP (do NOT edit calib-v2):** `BoresightCorrection{pitch_rad, roll_rad, vert_offset_m}`
  has no field for the watchdog prior WIDTH (the post-calibration residual σ). **Default:** the ESKF
  hardcodes a conservative `watchdog_prior_deg ≈ 0.13°` (= calib-v2's acceptance ceiling) and
  `watchdog_prior_m ≈ 0.05 m`. **Optional request:** calib-v2 add `pitch_residual_rad` / `vert_residual_m`
  so the watchdog prior is data-driven from the actual fit uncertainty. Flagged to the P1 commander; not
  edited.

---

## 6. Phased build plan (test-pinned, ε-independence stated)

**ε-INDEPENDENCE:** the accel-bias ESKF design + the watchdog MECHANISM are **entirely ε-independent** —
they need no boresight value to build or test. Only the watchdog's **prior VALUE** (mean = baked ε; width)
and the **final on-hardware validation** need ε (and the calib-v2 struct populated). Phases A–C can be built
and merged before P3's arbiter selects the form/value.

**Phase A — accel-bias-augmented filter (numpy, torch-free).** Add optional `n_bias`/`bias_axes=(1,2)` to
`LinearKF` (or sibling `AccelBiasKF`): state `[p,v,b_a]`, predict extends F/Q with the body Y,Z coupling
`−0.5dt²R_wb[:,1:3] / −dt R_wb[:,1:3]`; the position/given/velocity updates use `H=[I,0,0]` (b_a rides the
predict). `n_bias=0` ⇒ unchanged. Honor the three byte-identical guards (§4). *Tests:* `n_bias=0`
byte-identical; 2-axis convergence to an injected in-plane bias on a maneuvering trajectory; SPD-preserve;
**NEES ≈ 2** (the observability-honesty pin — guards against the 3-axis overconfidence).
**Phase B — RewindKF carries the accel-bias through replay.** `kf_rewind.py` predict already records
`R_wb`; extend the wrapped state to the augmented dim so the replay re-propagates `b_a` (no new fix-op kind
— `b_a` is predict-coupled, not fix-coupled). *Tests:* OOSM replay with `b_a` == in-order oracle (atol
1e-12); zero-latency degeneracy; SPD.
**Phase C — navigator wiring + watchdog, behind `use_imu_bias_eskf` (OFF).** `NavigatorConfig`:
`use_imu_bias_eskf=False`, `bias_axes=(1,2)`, `accel_bias_prior`, `accel_bias_rw` (≈0.005–0.01), watchdog
params. Build the augmented filter when ON (composing with `use_rewind_kf`). The WATCHDOG: accumulate the
windowed `m_v(r) = a+b·r` from accepted-fix vertical innovations, expose `watchdog_flag` on `_VisionDiag`
(off-contract). *Tests:* OFF ⇒ VQ1 byte-identical; ON ⇒ in-plane bias converges + terminal coast de-biased;
`test_obs_sign_faithfulness` +L survives non-zero `b_a`; watchdog flags an injected 1.5° / 0.2 m shift and
does NOT flag the baked (zero-residual) case.
**Phase D — calibration/consistency pins + compose seam.** Pin the gyro-DEAD assertion (a `b_g` block gains
no information). Pin no-double-count (ESKF reads `frames.BORESIGHT` only for the watchdog prior; `R_wc`/`z`
unchanged by it). Document `bias_axes=(1,2)` rationale (the 3-axis NEES-18 pathology).

**Order A→B→C→D.** Contract-watch: none expected (obs_dim/+L preserved by construction); surface UP the
moment anything touches `[17:19]` semantics, the obs builder, or NavState's field set, or if calib-v2's
struct gains the residual-width field.

---

## 7. Risks / escape-hatch findings

- **Accel-bias value is data-conditional.** It is observable + honest (2-axis) and converges in ~2 laps,
  but only BUYS margin if a real dead-reckoning bias exists. **First-contact must measure it** (KF
  open-loop vs GT velocity). If it's already ≤ budget, the ESKF legitimately reduces to **just the
  watchdog** — a clean simplification, not a failure (the escape-hatch outcome).
- **Gyro is dead — do not be tempted.** Adding it for "completeness" injects an uninformed state; it is
  architecturally dead until/unless we switch to gyro-integrated attitude (we do not, and should not for VQ).
- **3-axis accel-bias is a trap.** The along-track axis is unobservable and makes the filter overconfident
  (NEES 18). The 2-axis (Y,Z) restriction is load-bearing — do not "generalise" to 3-axis.
- **Sim caveat (carried):** the geometry-generator accel-bias study (`accel_bias_observability.py`) is
  over-optimistic on the RATE (artificial inter-gate gaps); trust `accelbias_realistic.py` (constant-rate,
  continuous trajectory) for the rate verdict. A debugged bug (missing `t_prev` update collapsing the gap)
  is fixed in both; the realistic instrument is authoritative.
- **Double-count guard (carried):** once `b_a` is estimated, the predict's `attitude_noise_std²(S Sᵀ)`
  inflation double-counts the systematic part; a later tightening can reduce it to the residual random
  attitude noise. The conservative analysis keeps the full term, so this is a future optimisation, not a
  bug. Flag UP if changing `attitude_noise_std`.

---

## MEMORY-DELTA (≤10 lines)
- **branch: claude/stoic-cray-33a854** (design-only, no src; baseline 703 pass / 35 skip green; gated OFF
  behind new `use_imu_bias_eskf` → inc7/VQ1/case-A byte-identical; 20-dim obs + +L preserved). Report:
  `handoff/p1-vision-accuracy-2026-06-14/ESKF_RESCOPE_DESIGN.md` (+ scratch-eskf-rescope/).
- **DROPPED the online boresight pitch-bias state** (constant → calib-v2 OFFLINE bake; gauge-ambiguous +
  double-counts; corroborated by p1-eskf-design's 1.0° first-pass). The proven `H_β=−skew(L)·R_wc` =
  MEAN-of-lever-term result is RE-TARGETED to the predict's `attitude_noise²·skew(s)·skew(s)ᵀ` (estimate
  the dead-reckoning bias mean, don't inflate).
- **GYRO-bias = DEAD → not built** (attitude GIVEN, never gyro-integrated → zero coupling to pos/vel;
  posterior never leaves prior). Upstream gyro drift → tilt → gravity leak → ABSORBED by accel-bias.
- **ACCEL-bias = LIVE, 2-axis BODY lateral(Y)+vertical(Z) ONLY** = the gate-plane in-plane = margin axes.
  Observable + honest (MC NEES 1.97/2, unbiased); under-budget (g·sin0.6°=0.103 m/s²) in ~2 laps @fr0.07,
  ~1 lap @fr0.15+; drift q_ba≤0.02 stays under. **DROP along-track(X): unobservable (NEES 18, aliased w/
  velocity), maps to gate-normal/depth** (margin-irrelevant).
- **WATCHDOG = read-only tight-prior monitor** of windowed vertical-fix residual `m_v(r)=a+b·r`: ANGULAR
  form → flag slope `|atan(b)|>0.6°` (range-∝); METRIC form → flag intercept `|a|>0.15 m` (range-flat).
  WARN (≥1 lap-window) → ABORT/vision-untrusted (≥2-3 windows or 2×budget); NEVER silently corrects.
  Catches gross shifts only (sparse fixes). Reuses `H_β` as the angular detector.
- **NO DOUBLE-COUNT interface:** ESKF reads `frames.BoresightCorrection` ONLY to set the watchdog prior
  (mean 0 about baked, width=residual); never re-applies the mean to `R_wc`/`z`. Accel-bias is orthogonal
  to the bake (dynamic vs sighting; different range-signature). Form-choice is load-bearing for the
  WATCHDOG signature (not the accel-bias). **Interface request UP:** add `pitch_residual_rad`/`vert_residual_m`
  to the struct for a data-driven watchdog prior width (else ESKF hardcodes 0.13°/0.05 m). calib-v2 NOT edited.
- **ε-INDEPENDENCE:** accel-bias design + watchdog MECHANISM need no ε; only the watchdog prior VALUE +
  final validation need ε/the populated struct. Phases A–C buildable before P3 selects the form. → [[index-vision-estimator]]
