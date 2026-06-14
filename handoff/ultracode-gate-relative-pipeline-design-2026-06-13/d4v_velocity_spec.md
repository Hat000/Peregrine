# d4v — VELOCITY CHANNEL (component 4): case-C velocity acquisition for the gate-relative pipeline

**Component:** `velchannel`. **Scope:** how the inc8 policy's `obs[3:6] = vel_g = R_w2g @ vel` is
acquired in TRUE case C (vision position-only), and whether to add a direct vision-velocity measurement.
**Why it matters:** velocity is the **swing variable** for the gate-4 0.155 m in-plane margin
(estimator-racespeed REPORT §2 commander refinement: warm vs cold velocity prior = in-plane 0.11 m vs
0.21 m, straddling the margin).

All load-bearing numbers below were **re-derived this run** by running the REAL `racer.state_estimator.LinearKF`
+ `RewindKF` over the g3→g4 transit. Prototype + JSON:
`handoff/ultracode-gate-relative-pipeline-design-2026-06-13/d4v_velocity_channel.py`
(`d4v_velocity_channel_results.json`). Reproduces bit-stable across runs (seed 20260613, deterministic).

---

## TL;DR — the verdict in five lines

1. **The KF already differences position fixes optimally** — you do NOT build a separate differencer.
   Naive single-pair differencing is useless at the fix cadence (σ_v = √2·σ_p/dt = **5.2 m/s** lateral at
   14 Hz); the KF/LSQ over the 9-fix g3→g4 window gives **σ_v ≈ 0.42 m/s lateral / 0.78 m/s radial**.
2. **The cold↔warm fork is REAL and reproduced:** cold (wrong velocity prior, position-fix-diff only) →
   gate-4 in-plane pos RMS **0.215 m (OVER the 0.155 m margin)**; warm (lap-converged velocity) → **0.119 m
   (clears it)**. The position-fix-difference *during the 0.66 s transit alone* does NOT recover the warm
   number from a cold start — there isn't enough baseline.
3. **A WEAK vision-velocity assist flips cold to GO.** Even a crude direct vel measurement (σ_v = 1.0 m/s)
   pulls cold from 0.215 m → **0.144 m (clears the margin)**; σ_v = 0.5 m/s → 0.124 m; σ_v = 0.3 m/s →
   recovers warm-prior quality (~0.12 m). **It moves the gate-4 margin — so it is NOT an over-build in
   the cold-start case.**
4. **Accel-bias drift couples LINEARLY into vel_g and QUADRATICALLY into position.** Between fixes (0.07 s)
   it is negligible (dv≈0.01 m/s, dp≈0.4 mm at 1° attitude bias). It only bites on a **terminal coast**
   (gate exits FoV at r<~1.3 m) or a vision dropout: at 0.3 s coast, 1° attitude bias → dv≈0.05 m/s,
   dp≈8 mm — still sub-margin. Over a **full lap with no vision** it is unbounded (dp ≈ 11 m at 1°), but
   that is the vision-dropout failure mode, not the nominal gate-4 case.
5. **RECOMMENDATION: build position-fix-differencing (free — it's the KF) as the baseline; ADD the
   vision-velocity channel as a CONDITIONAL P1, gated on the full-lap velocity-prior measurement.** If the
   full-lap sim shows the velocity prior entering gate-4 is **warm** → vision-velocity is insurance (DEFER,
   matching vision-case-c P3-1). If it shows **cold** → vision-velocity is **load-bearing** (it is the
   cheapest lever that flips the margin). This is the same Q①-style "load-bearing vs insurance" fork the
   organizer pivot imposes — **build it regardless if cheap, decide priority on the full-lap measurement.**

---

## (1) POSITION-FIX DIFFERENCING — the math, the noise, and why the KF already does it

### The arithmetic
The naive velocity from two consecutive gate-relative position fixes is
`v_hat = (p_fix[k] − p_fix[k−1]) / dt`. Each fix carries per-axis noise σ_p (gate-relative **lateral
σ_p = 0.265 m**, the measured value reproduced bit-exact in c1; radial/along-track **σ_p ≈ 0.50 m**). The
difference of two independent fixes has noise √2·σ_p, so

> **σ_v(naive single-pair) = √2 · σ_p / dt**

This **amplifies** noise catastrophically at the fix cadence (MEASURED, part B):

| dt | lateral σ_v (σ_p=0.265 m) | radial σ_v (σ_p=0.50 m) |
|---|---|---|
| 1/14 s (fix gap) | **5.25 m/s** | 9.90 m/s |
| 0.10 s | 3.75 m/s | 7.07 m/s |
| 0.50 s | 0.75 m/s | 1.41 m/s |
| 0.66 s (whole transit) | 0.57 m/s | 1.07 m/s |

**You never difference raw fixes.** The right operator is a least-squares slope (= what the KF does
optimally, fusing the IMU velocity prior): for N fixes uniformly spaced over a window W,
`σ_v(LSQ) = σ_p · √(12 / (N·(N²−1))) / dt_fix`. Over the g3→g4 window (N≈9 fixes, W=0.66 s, MEASURED):

> **σ_v(LSQ) ≈ 0.42 m/s lateral / 0.78 m/s radial** — a ~12× improvement over the single-pair difference.

### Does it beat pure IMU integration entering gate-4?
**Yes, but only marginally, and it depends on the prior.** Pure IMU velocity drifts at `dv = b_a·T`
(accel-bias × time since the last velocity-informative event). The KF's position-fix-differencing *bounds*
this drift to the LSQ slope variance above — i.e. position fixes ARE the only thing that observes velocity
in case C, so the KF velocity is exactly "IMU integration, periodically re-sloped by the fix stream." The
binding question is the **starting velocity error**:

- The KF run (part C, REAL LinearKF) shows the position-fix-difference over the **0.66 s transit alone**
  pulls a cold prior's velocity E-σ from 1.5 m/s down to **0.49 m/s** with a residual **+0.28 m/s bias**
  (the LSQ slope is biased low when the prior is wrong and the window is short). That residual velocity
  error integrates into the gate-4 in-plane pos miss → **0.215 m RMS, OVER the margin.**
- A **warm** prior (lap-converged velocity, the c1 assumption) gives velocity E-σ **0.04 m/s** → in-plane
  pos **0.119 m, clears the margin.**

**Conclusion:** position-fix-differencing (= the KF) is necessary and sufficient *if the velocity prior
entering gate-4 is already warm*. From a cold start it does NOT converge fast enough within the 0.66 s
gate-4 window. This is the fork the velocity channel must resolve.

---

## (2) THE REOPENED VISION-VELOCITY CHANNEL — observation, covariance, where it enters, decision

### The observation
A weak **direct** vision-velocity measurement of the world-NED velocity, from one of:
- **(a) inter-frame PnP translation delta:** `v_vis ≈ (t_cam_gate[k] − t_cam_gate[k−1]) / Δt_frame`,
  rotated to world `R_wc`. This is a *gate-relative* velocity (relative to the seen gate, which is
  static) → map bias drops out exactly, same as the position channel. Per-frame it is noisy (it is a
  position-difference over the 1/30 s frame gap → σ ≈ √2·σ_pnp/Δt_frame ≈ several m/s raw), so it must be
  **smoothed over a few frames** before it is useful: a 3–5-frame regression gives σ_v ≈ 0.3–1.0 m/s.
- **(b) gate-corner optical flow:** the apparent corner motion (px/frame) × range → world velocity. Lower
  latency than PnP-delta, but range-coupled and only the cross-LOS (lateral) component is well-observed.

Both are **lateral-dominant** (the radial/depth velocity is poorly observed — the same depth-degeneracy
that limits the position fix), which is fortunate: the gate-4 binding axes are **E (lateral) + D
(vertical)**, exactly where vision-velocity is strongest.

### The covariance
`R_vel = σ_v² · I` (or anisotropic: tight on E/D, loose on N, mirroring the position fix-cov structure).
The candidate σ_v is set by the smoothing window: **σ_v ∈ [0.3, 1.0] m/s** is the plausible band (part C
sweep). It must be HONEST — a vision-velocity that claims σ_v=0.1 m/s it cannot deliver makes the KF
overconfident and re-injects the same disease as the cov-floor trap. Calibrate σ_v from the **same
ShadowPC at-speed recording** that calibrates the position fix-cov (fold into SHADOWPC-VISION-CAL).

### Where it enters
- **KF:** through the existing `LinearKF.update_velocity(z, cov)` (state_estimator.py:145, `_H_VEL`
  observes velocity directly). **It already exists** — the channel is a new *measurement source*, not a
  new filter path. It enters at the fix cadence, AFTER the position update, on the same accepted frames.
  Gate it with the same relative-innovation outlier test as the position fix (a depth-flip corrupts both).
- **Policy obs:** it does NOT add a new obs dimension. It improves the *quality* of the existing
  `vel_g = R_w2g @ vel` (obs[3:6], fly_rl.py:349) the policy already consumes — the KF velocity feeds
  `vel_g` 1:1. So **no retrain is needed to consume it**; it just makes obs[3:6] less wrong in case C.
  (It DOES interact with the inc8 measured-error DR: train inc8 on the *cold* velocity-error class so the
  policy is robust whether or not vision-velocity lands — see §3 recommendation.)

### Decision criterion — does it move the gate-4 margin?
**MEASURED (part C, REAL LinearKF, with a 0.5° given-attitude bias injected):**

| regime | vis-vel σ_v | vel E-σ (m/s) | vel E-bias (m/s) | vel in-plane σ (m/s) | **gate-4 in-plane pos RMS** | p90 | < 0.155 m margin? |
|---|---|---|---|---|---|---|---|
| **cold** (no vis-vel) | — | 0.49 | +0.28 | 0.70 | **0.215 m** | 0.324 | **NO** |
| **warm** (no vis-vel) | — | 0.04 | +0.05 | 0.05 | **0.119 m** | 0.182 | YES |
| visvel | 1.0 | 0.27 | +0.07 | 0.38 | **0.144 m** | 0.213 | **YES** |
| visvel | 0.5 | 0.15 | +0.04 | 0.22 | **0.124 m** | 0.185 | YES |
| visvel | 0.3 | 0.10 | +0.03 | 0.14 | **0.124 m** | 0.187 | YES (≈ warm) |

**The criterion is met:** the vision-velocity channel **moves the gate-4 in-plane pos RMS from 0.215 m
(NO-GO) to ≤0.144 m (GO)** from a cold start, even at the crudest σ_v=1.0 m/s. It is the **cheapest lever
that flips the margin** when the prior is cold. **Coordinate with component 3's margin sweep:** the
threshold that decides "build vs defer" is whether the full-lap velocity prior entering gate-4 sits on the
cold or warm side — that is the single number that converts this from insurance to load-bearing.

**Worth building? CONDITIONAL — but lean BUILD.** vision-case-c P3-1 deferred it as over-build because it
assumed the position-fix-difference + IMU drift bound was sufficient. The estimator-racespeed refinement
*reopened* it precisely because velocity turned out to be the swing variable. This sim **confirms the
reopening was right**: the channel is decision-relevant in the cold case. It is also genuinely cheap
(the `update_velocity` path already exists; the only new code is the inter-frame PnP-delta smoother + its
σ_v calibration + reusing the position outlier gate). **Recommend: DESIGN it now, build it as a
CONDITIONAL P1, decide go/defer on the full-lap velocity-prior measurement.**

---

## (3) ACCEL-BIAS DRIFT BOUND + the cold/diff/vision recommendation

### The drift bound (MEASURED, part A — re-derived analytically; matches `_audit_numbers3.py`)
The KF predict-step Q models **zero-mean white** accel (0.3 m/s²) + attitude (1.4°) noise — it does NOT
model a **systematic** bias. A constant horizontal accel bias `b_a` (from a residual IMU bias OR, the
dominant source, a systematic given-attitude tilt `b_θ` → phantom accel `g·sin(b_θ)`) drifts:

> **velocity error  dv(T) = b_a · T**  (LINEAR → couples into `vel_g`, obs[3:6])
> **position error  dp(T) = ½ · b_a · T²**  (QUADRATIC → couples into the gate-4 in-plane miss)

| bias source | effective b_a (m/s²) | dv @ 0.3 s | dp @ 0.3 s | dv @ 0.66 s | dp @ 0.66 s | dp @ 11.45 s (lap) |
|---|---|---|---|---|---|---|
| accel_bias 0.03 m/s² | 0.030 | 0.009 | 0.001 | 0.020 | 0.006 | 1.97 m |
| accel_bias 0.10 m/s² | 0.100 | 0.030 | 0.004 | 0.066 | 0.022 | 6.56 m |
| attitude 0.5° | 0.086 | 0.026 | 0.004 | 0.057 | 0.019 | 5.61 m |
| attitude 1.0° | 0.171 | 0.051 | **0.008** | 0.113 | 0.037 | 11.22 m |
| attitude 1.4° (meas 1σ) | 0.240 | 0.072 | 0.011 | 0.158 | **0.052** | 15.71 m |

**Interpretation:**
- **Between fixes (0.07 s gap):** drift is negligible — dv≈0.012 m/s, dp≈0.4 mm at 1° bias. The fix
  cadence keeps the position channel from ever coasting long. Accel bias is NOT a between-fix problem.
- **Terminal coast (gate exits FoV at r<~1.3 m → ~0.04 s at 37 m/s; a dropout could reach 0.3 s):** at
  0.3 s, 1° bias → dv≈0.05 m/s, dp≈8 mm — still well sub-margin on position. **But the velocity error
  (dv) it injects persists into vel_g and is NOT reset by the next position fix** (case C: position-only);
  it is only slowly re-observed by position-fix differencing. This is why the **vel-prior bias** (+0.28 m/s
  in the cold KF run) is the real coupling — it is the accumulated, un-reset velocity error.
- **Full lap with NO vision (11.45 s):** dp blows up to **6–16 m** — but this is the **vision-dropout
  failure mode**, not the nominal gate-4 case. It bounds the abort/coast policy, not the margin.

The attitude-bias term lands on the **horizontal (E in-plane + N along-track)** axes (gravity tilts
sideways, benign on D) — so it hits the binding E axis, consistent with the cold KF run's +0.28 m/s
**E-velocity bias**.

### The three options, with margin consequence

| option | cost | gate-4 in-plane pos RMS | vel_g quality (E-σ) | verdict |
|---|---|---|---|---|
| **(a) cold IMU-only** (no vision-vel, cold prior) | **free** (cheapest) | **0.215 m → OVER margin** | 0.49 m/s + 0.28 bias | **insufficient from cold start** |
| **(b) position-fix-difference** (= the KF, warm prior) | **free** (already built) | **0.119 m → clears** | 0.04 m/s | **sufficient IFF prior is warm** |
| **(c) vision-velocity** (σ_v≤1.0 m/s, from cold) | **low** (P1, update_velocity exists) | **0.144 m → clears** | 0.27 m/s | **the lever that makes cold-start safe** |

**RECOMMENDATION (ranked):**
1. **Ship position-fix-differencing as the baseline — it is free (it IS the KF).** It is necessary and it
   suffices *when the velocity prior is warm*. This is the default and must be in regardless.
2. **Run the full-lap case-C sim FIRST** (the commander follow-up) to measure whether the velocity prior
   entering gate-4 is warm or cold. **This is the decision gate.** Pure offline, no new data.
3. **If the full-lap prior is COLD → build the vision-velocity channel (P1, load-bearing).** It is the
   cheapest lever that moves the margin from NO-GO to GO (0.215 → 0.144 m at σ_v=1.0). Reuse
   `update_velocity` + the inter-frame PnP-delta smoother + the position outlier gate.
4. **Either way, train inc8 on the COLD velocity-error class** (measured-error DR, CONTEXT input ii) so
   the policy is robust to a cold prior whether or not vision-velocity lands. Do NOT train assuming a warm
   prior — that is the brittle assumption that makes the margin straddle.

---

## Confidence — MEASURED / EXTRAPOLATED / ASSUMED

**MEASURED (re-derived this run):**
- Naive-difference and LSQ velocity-noise propagation (part B) — closed-form, exact.
- Accel-bias drift dv=b_a·T, dp=½b_a·T² (part A) — analytic, matches `_audit_numbers3.py` exactly.
- The cold/warm fork and the vision-velocity recovery (part C) — REAL LinearKF + RewindKF, 600-MC,
  deterministic and bit-stable across runs. cold 0.215 m / warm 0.119 m / visvel-1.0 0.144 m.
- Per-fix gate-relative lateral σ_p = 0.265 m (reproduced bit-exact from c1, the measured pool).

**EXTRAPOLATED (medium-low — the dominant residual uncertainty):**
- The **velocity prior actually entering gate-4 in a full case-C lap** (cold vs warm). My part-C cold/warm
  are the two ENDPOINTS; the real prior sits between them and is the swing variable. The full-lap sim
  resolves it — **it is the single number that decides build-vs-defer for the vision-velocity channel.**
- The achievable vision-velocity σ_v (0.3–1.0 m/s band) — set by the smoothing window and the at-speed
  PnP-delta noise, neither measured at 37 m/s. The position fix-cov extrapolation (best-case lower bound,
  all data ≤8.4 m/s) applies equally here.
- The systematic given-attitude bias magnitude (0.5–1.4°) — bounded by the measured attitude-noise 1σ but
  the *systematic* component is unmeasured; I used 0.5° as the injected nominal.

**ASSUMED (stated):** const-v 37 m/s straight g3→g4 (perfectly centered truth), 14 Hz landed fixes,
0.66 s transit, 90 Hz IMU, the +1.5 m/s lateral cold-prior offset (a deliberate cold-start stress). The
margin (0.155 m) and per-fix lateral (0.265 m) are inherited measured facts.

**THE BIGGEST RESIDUAL UNCERTAINTY:** the full-lap velocity prior. If warm → position-fix-differencing
suffices and vision-velocity is insurance (defer). If cold → vision-velocity is load-bearing and is the
cheapest margin lever in the whole pipeline. One offline full-lap sim + the ShadowPC σ_v calibration
resolve it. The whole question is gated upstream by organizer Q① (if case A/B, velocity is given and none
of this matters).
