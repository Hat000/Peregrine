# VQ2 RL Controller — Launch-Ready Spec (2026-07-04)

**The "second controller":** an RL policy trained in DiffAero on Adroit, deployed on the VQ2
self-localizing (case-C) wire. This doc is the launch package: the twin-fidelity transfer gate
(T1), the observation-model spec reconciled with the obs-20 contract + research (T2), and the
curriculum + Adroit launch plan (T3).

Author: DESIGN+GATE-CHECK dive, 2026-07-04. READ-ONLY pass on src/rl (probe script + this doc are
the only writes). **Every hypothesis is flagged AS a hypothesis. No commits.**

Scope note: this is Fengyou's "second controller" — a *vision-pose + IMU → CTBR* policy that
subsumes today's hand-built gate_seeker classical servo (the "first controller", inc7-on-case-C,
which stalls at gate 0/1 in all 26 flights in the run pile). The RL controller is the same S2
architecture as inc8 (perception-module → engineered-state → policy), VQ2-ized.

---

## 0. TL;DR for the commander / Fengyou

- **T1 TWIN GATE = CONDITIONAL-GO.** On the two channels the competitive wire lets us measure
  *attitude-free*, the twin is faithful to VQ2: one-step gyro-response residual **p90 ≈ 0.024
  rad/s** on clean flights (the rate loop — the transfer-critical axis — transfers essentially
  exactly), and hover specific-force **9.812 m/s² vs g=9.807 (+0.005)** (hover point exact). The
  CONDITION: three channels are **UNVALIDATED by this tier** (thrust map above hover, drag at
  speed, sub-0.15 collective) because they cannot be separated from attitude without GT. Verdict:
  **GO to train the SLOW-first curriculum now; hold the high-speed rungs until a command-sweep or
  a BEST-tier (GT-leaking Training-mode) capture closes the above-hover thrust/drag gap.**
- **T2 OBS MODEL** = the frozen inc8 **obs-20** contract, VQ2-ized: it already carries exactly
  Fengyou's list (filtered gate-relative pose + derivatives from the FILTER that flies, IMU-derived
  attitude/rates, last command, confidence triple). The **"gate normal" is DELIBERATELY ABSENT** —
  research says discard PnP rotation (planar-flip ambiguity); Fengyou's own ~20° cone estimate
  confirms it's too imprecise to feed. Approach geometry comes from bearing + known-gate-size range,
  not the normal. Add: **asymmetric privileged critic** (already wired, `algo=appo`), a **GRU** for
  the terminal blind-zone coast, and **measured-from-data** no-info/staleness encoding.
- **T3 CURRICULUM** = hover → single-gate → blackout-pass → multi-gate lap, on the random-track
  generator VQ2-ized to the gate priors. Derive the launch from **`peregrine_inc8_recenter.sbatch`**
  (the only inc8 recipe that has ever flown a lap). Reward spine exists in code (progress-dominant +
  R5' 2-axis terminal-lock + through-centering); **the reward FREEZE is Fengyou's** — options
  presented, not decided.
- **🚩 ADROIT BRIDGE IS DOWN ON SHADOWPC.** The `adroit-connector` (paramiko `serve`+`x` daemon)
  that drove all training lived on the OLD laptop and is **not in the ShadowPC checkout.** Fengyou
  must re-establish SSH+Duo access and re-stage the connector before ANY GPU spend. Details in §T3.4.

---

# T1 — TWIN-FIDELITY GATE (the transfer risk)

## T1.1 What we could actually measure — and why it's the FALLBACK tier

The 2026-06-11 VQ1 transfer failed on **unmodeled mixer rate↔thrust coupling**. The twin-fidelity
probe (`twin_fidelity_probe_DESIGN.md`) was SPEC-ONLY and never ran against VQ2. I built and ran it.

**Wire reality (verified from the tlogs of all 26 runs):** the VQ2 competitive wire carries ONLY
`HIGHRES_IMU` (accel + gyro, ~185 Hz), `ACTUATOR_OUTPUT_STATUS` (all-zero — useless), and
`COLLISION`. There is **NO `ATTITUDE` / `ODOMETRY` / `LOCAL_POSITION_NED`** → no ground-truth
world position, velocity, OR attitude, and mag/baro are NaN. So the BEST-tier "commanded-input
replay vs GT velocity" is impossible on these captures. We are in the **FALLBACK tier** and I built
two *attitude-free body-frame residuals* that need no world-frame GT:

**(A) Gyro-response residual (rate loop — the transfer-critical axis).** The twin's rate loop maps
a commanded body rate → realized rate via `gain·rate_sign·cmd_rate`, lag τ, super-rate, slew clip.
On the wire the realized body rate IS the HIGHRES_IMU gyro (after the deploy-profile `gyro_sign`
correction). Seed the twin's ω at tick *k* from the measured ω, drive it one control step with the
**actual sent command** (`body_rate_sent = 0.4·body_rate_frd`), compare the twin's predicted ω at
*k+1* to the measured ω at *k+1*. This is **one-step-ahead** → never integrates attitude → no AHRS
contamination. Directly tests `rate_gain` (the ~2.5×), `rate_tau`, super-rate, mixer slew.

**(B) Specific-force-magnitude residual (thrust map + hover).** `|a_body|` from HIGHRES_IMU is
attitude-INVARIANT (rotation preserves norm). The twin predicts an upward specific force `a_up` from
the commanded collective. Near hover, `|a_body| ≈ a_up`. Comparing them tests the thrust map + hover
WITHOUT an attitude estimate.

Probe: `scripts/vq2_loadday/twin_fidelity_probe.py` (new; reuses `racer.rl_plant.step` +
`racer.frames` conventions; runs on the .venv). It reports BOTH the *measured* plant (map ON:
super-rate + convex coll-map + mixer + quad-drag) and the *flat* plant (map OFF — the config every
shipped inc7/inc8 checkpoint was actually trained against).

## T1.2 Results (26 runs; the probe's JSON is in scratch, headline numbers below)

**Gyro-response residual (rate loop):**

| flight class | one-step gyro resid norm p90 | active-maneuver p90 | interpretation |
|---|---|---|---|
| 3 cleanest long flights (130314, 120357, 032626) | **0.024 rad/s** (median) | 0.35–0.73 | rate loop transfers ~exactly |
| all 12 healthy runs (≥500 cmds) | 0.50 (median), 0.02 best | 0.71 | contaminated by contact spikes |
| short/crash runs | up to 3.5 | up to 7.5 | drone hitting gates/floor |

The per-axis clean p90 is **0.005–0.013 rad/s** — a <1% error on typical 0.3–1.5 rad/s commands.
The elevated median across all runs is **NOT a rate-loop error**: it is the +45°/tick contact-spike
aliasing (a real physical event — the drone striking a gate/floor — that the twin correctly does not
reproduce, and that the deploy-profile A32 already calls out as "spec F3"). Gating to the flying
(non-contact) portion of clean flights, the rate loop is GO by a wide margin. **This is the axis the
last transfer died on, and it now transfers — because `cmd_rate_scale=0.4` exactly cancels the ~2.5×
realization, and the twin's `rate_gain≈2.5` + `rate_tau=0.019` match the wire's inner loop.**

**Specific-force / hover:**

| metric | value | interpretation |
|---|---|---|
| hover-band `\|a_body\|` median (across healthy runs) | **9.812 m/s²** (g = 9.80665; err **+0.005**) | hover point EXACT — confirms twin hover 0.2656 |
| hover-band SF resid, flat plant | +0.005 to +0.07 m/s² | thrust map at hover is exact |
| above-hover "clean" resid | 0.1–1.5 m/s² (CONFOUNDED) | see below — not a twin error |

**🚩 Key measurement caveat (a finding, not a defect):** the above-hover specific-force residual is
**CONFOUNDED and must NOT be read as a twin thrust-map error.** A tick with low body-rate is NOT
near-level: a drone in steady forward flight (pursuit/cruise) has low rate but ~18–30° tilt and
non-trivial speed, so `|a_body|` there mixes thrust + drag + a tilted specific-force direction we
cannot separate attitude-free. Diagnostic (run 120357): as commanded collective rose 0.26→0.31,
`|a_body|` stayed pinned near g (9.81–9.83) while both thrust maps predicted a_up rising to
~11–12 m/s² — because the drone was tilted forward and NOT accelerating hard upward. This is a
limitation of the *fallback tier*, not evidence of a twin gap.

**Secondary finding (real, and useful):** on the *valid* (hover-band) SF check, the **flat
`g·coll/hover` map matches the wire slightly better than the convex measured coll-map** (measured-map
hover resid +0.23 vs flat +0.005). Since the shipped checkpoints trained against the flat map, this
is *reassuring for those checkpoints* — but it is only a hover-point statement; the two maps'
above-hover divergence (the whole reason the convex map exists) is exactly the region the fallback
tier can't adjudicate. HYPOTHESIS: the convex measured coll-map (VQ1-calibrated) may slightly
over-predict thrust vs VQ2; only a BEST-tier or command-sweep capture can confirm.

## T1.3 Known gaps — where each convention lives, and what's UNVALIDATED

- **Sub-0.15 collective: UNMEASURED (confirmed).** No clean (near-hover-rate) tick commanded
  collective below ~0.16; the plant's sub-0.15 knots are the floored linear-extrapolation guesses
  (`COLL_MAP_ACCEL_MEASURED[0:2] = 0`). If the curriculum lets the policy command deep net-down
  collective (hard braking / descent), it trains against an unvalidated floor. **Mitigation:** cap
  the training collective floor at ~0.15 for the slow curriculum, OR capture a deliberate
  descent-collective sweep to fit the low end. (Ties to the memory's "VQ2 plant CAN thrust below
  free-fall … sub-0.15 unmeasured, floor-smack risk".)
- **Translational lift +2.3 m/s² at speed:** the memory notes a +2.3 up-bias at speed
  (translational lift, not hover miscal). The twin's quad-drag/lift model has a body-frame quadratic
  term but **the fallback tier cannot verify equivalence at speed** (same attitude-free confound as
  above). UNVALIDATED — needs BEST tier. HYPOTHESIS: this is inside the twin's DR-coverable envelope
  (the drag coefficients have measured DR bands), but that is an assumption until measured.
- **Wire-layer sign conventions — where they belong in the training actuation path:** the plant
  (`rl_plant`) is telemetry-free physics. The wire conventions live OUTSIDE it, and the RL actuation
  path must apply them at the deploy boundary, NOT in the trained plant:
  - `cmd_rate_scale = 0.4` — a WIRE multiply (`MavlinkClient.send_command`). The twin's `rate_gain`
    already encodes the ~2.5× realization, so **the training env applies the raw FRD body-rate to
    the plant; the 0.4 is a deploy-time uplink pre-scale only.** Do NOT bake 0.4 into training.
  - `gyro_sign = (-1,-1,-1)` — a WIRE read correction (`MavlinkClient`, before AHRS). Training never
    sees a wire gyro (it has true ω), so this is deploy-only. The `rate_sign` in the plant
    (`(1,1,-1)` trained-world = telemetry artifact; TRUE live sign is `(1,1,1)`, FRAME-AUDIT
    2026-06-12) is the ONLY sign in the plant, and it is the *trained-world* convention the FLU→FRD
    action adapter composes with — leave it as the shipped default.
  - `body_rate_sign` / `yaw_steer_mode` (A36/A37) — these are SEEKER/CONTROLLER wire conventions for
    the *classical* controller, not the RL plant. The RL controller emits CTBR directly and its
    action→wire sign map is the inc8 `_FLIP`/`_ACT_FLU_TO_FRD` composition (already parity-pinned).
    **The A36/A37 yaw saga (resolved: mode "off" + brs (1,1,1) = the A14-correct state, per the yaw
    memory) is a first-controller finding and does NOT transfer to the RL actuation path** — the RL
    policy learns its own steering; only the CTBR sign map matters, and that is already pinned.

## T1.4 T1 VERDICT

**CONDITIONAL-GO.** The twin is faithful enough to train the *slow-is-smooth* curriculum against
NOW: the rate loop (last transfer's failure axis) and the hover point are validated exact by
attitude-free residuals on real VQ2 flights. Three channels (above-hover thrust map, drag at racing
speed, sub-0.15 collective) are UNVALIDATED by the fallback tier and gate the *high-speed* rungs.

**To lift the condition (recommended, cheap):** capture ONE deliberate **command-sweep** flight on
the VQ2 wire — a scripted sequence (hover hold → collective steps up to full → single-axis roll/
pitch/yaw steps → a descent-collective sweep to sub-0.15) rather than an incidental race trajectory.
Because the sim is deterministic, a sweep pins the thrust map and drag cleanly even without GT
(the gyro-response + SF-magnitude residuals become clean at the sweep's known operating points).
**Better:** if C5 (load-day) reveals Training mode leaks `ODOMETRY`/`LOCAL_POSITION_NED`, run the
BEST tier (velocity-residual replay + `fit_twin.py` calibration) — that closes all three gaps and
lets `fit_twin` re-estimate `PlantParams` for a VQ2 `_MEASURED` constant set.

**🚩 Fork for commander/Fengyou:** train the slow curriculum on the current (flat-map) plant now, or
wait for a command-sweep capture first? My recommendation: **start the slow curriculum now** (the
validated channels dominate slow flight; the unvalidated channels bind only at speed), and capture
the sweep in parallel so the high-speed rungs are unblocked when the slow policy is ready.

---

# T2 — OBSERVATION-MODEL SPEC

## T2.1 Reconciling Fengyou's input list with the frozen obs-20 contract

Fengyou's intent: *vision pose + derivatives (gate center + normal), IMU + derivatives (gravity-
derived attitude, per-axis felt accel).* The inc8 **obs-20 contract already IS this list**, in the
form research says to use. It is FROZEN and parity-pinned (`tests/test_deploy_obs20.py`,
`inc8_estimator_emul.py` d5), and — critically — **fed from the FILTER states that fly**
(`racer.estimator_obs.estimator_obs20` sources it from the live `NavState`), not from raw PnP. That
satisfies Fengyou's "should be handled by gate-position smoothing" directly: the policy sees the
*same smoothed gate estimate the deployed stack produces*, so train==deploy distribution holds.

**The input-state vector (name, source, units, deployment provenance):**

| idx | name | source (train) | source (deploy) | units | notes |
|---|---|---|---|---|---|
| 0:3 | `pos_g` | GT gate-rel pos + measured estimator residual (NEVER pristine truth) | RewindKF gate-relative +L fix | m, gate frame | Fengyou's "filtered gate center position" |
| 3:6 | `vel_g` | KF vel, gate frame | KF vel (fix-differencing = the KF, free) | m/s | the *derivative* Fengyou wants; from the KF, not a load-bearing vision-velocity channel (research: LSQ vision-velocity REFUTED) |
| 6:9 | `rpy_g` | AHRS attitude, gate frame | **ESKF** (gravity-derived — Fengyou's "gravity-derived pitch/roll") | rad | attitude from AHRS, NOT from PnP rotation |
| 9:12 | `w_flu` | plant ω (FLU) | HIGHRES_IMU gyro (post gyro_sign) | rad/s | Fengyou's per-axis rates |
| 12 | `last_normed` | last collective cmd | last collective cmd | — | action history (1 step) |
| 13:16 | `gate_rel_pos_nxt` | next-gate rel pos | KF + known map / next-gate track | m | next-gate lookahead |
| 16 | `gate_yaw_rel_nxt` | next-gate yaw | known per-gate yaw | rad | gate-relative yaw (upright prior) |
| 17 | `c_inplane` | `clip(σ_ref/σ_inplane_hat,0,1)` | live KF cov | — | in-plane fix confidence |
| 18 | `c_along` | `clip(σ_ref/σ_along_hat,0,1)` | live KF cov | — | along-track (range) confidence |
| 19 | `age_norm` | `clip(t_since_fix/τ_stale,0,1)` | pose age | — | **staleness** (see T2.3) |

`σ_ref = 0.05 m`, `τ_stale = 0.10 s` (frozen d5 constants, `estimator_obs.SIGMA_REF_M/TAU_STALE_S`).

**"Felt acceleration per axis" (Fengyou explicitly wants it):** NOT currently in obs-20. HYPOTHESIS:
add a **`a_body` channel (obs[20:23], specific force FRD from HIGHRES_IMU)** for the VQ2 policy — it
is deployment-available (the raw IMU accel), reliable, and gives the policy the "felt acceleration"
signal Fengyou names + a lead indicator for the terminal blind-zone. This is a **VQ2 obs extension
(23-dim)**, gated behind a fresh train (breaks obs-20 parity by design). Recommend as a portfolio
arm, not the default first run. (Rationale for caution: the design-philosophy memory — "every actor
input must be deployment-available + reliable + train==deploy; too much unreliable info → spurious
correlations". `a_body` is reliable, so it's a good candidate, but validate it earns its keep.)

## T2.2 The gate "normal" — DISCARD for attitude, use bearing+size for geometry

Fengyou: *"the normal is extremely imprecise, maybe a ~20° cone; lots of decisions, figure it out."*
The research is decisive and agrees with his instinct:

- **Conclusion 5 (BINDING):** DISCARD PnP rotation — the planar-square FLIP ambiguity is worst at
  distance; attitude comes from AHRS/filter, not the gate normal.
- So the gate **normal is NOT an attitude source** and NOT an obs channel for orientation.
- **For APPROACH GEOMETRY**, two options (Fengyou's "figure it out"):
  - **(RECOMMENDED) Bearing-only + known-size range.** The gate is a known 1.5/2.72 m concentric
    square → its *apparent size* gives range without needing the normal (research conclusion 7:
    bearing-angle / known-size observability = range without lateral maneuver). The obs already
    carries `pos_g` (which encodes bearing + range from the +L fix); the normal adds nothing
    reliable. This is the map-free, flip-immune path. **Do this.**
  - (Alternative, only if a normal is ever wanted) A **coarse confidence-weighted normal channel**:
    feed the *angle* of the estimated normal AND a confidence scalar (low, reflecting the ~20° cone),
    so the policy can learn to discount it. HYPOTHESIS: this underperforms bearing+size and adds an
    unreliable channel the design-philosophy memory warns against. **Reject unless a specific need
    appears** — the upright-gate prior (roll=pitch=0, gate-relative yaw known from the map) already
    supplies the orientation the normal would.

**Net:** no normal in the obs. Geometry = bearing (in `pos_g`) + known-size range (in `pos_g`/
`c_along`). This matches conclusion 7's Gap-Traversal-Guidance framing (through-gate from vertex
bearings, no PnP).

## T2.3 No-info / staleness encoding — MODELED FROM MEASURED VQ2 REALITY

I measured the real distributions across all 26 runs (16,702 ticks). These are the numbers the
injection spec uses (Fengyou's "exploit determinism" — exact-match measured channels, DR only the
genuinely stochastic one):

**Vision staleness (`age_norm`, obs[19]):**
- Live field is `pose_age_s`. **`time_since_vision_s` is 100% null in the data — do NOT use it.**
- `pose_age_s`: 16% null (pre-first-fix), non-null p50 **0.253 s**, p90 **0.552 s**, **clamped at
  1.0 s** (never exceeds; 6.3% of ticks sit exactly at the 1.0 clamp).
- **Fresh-fix cadence when fed: ~28.7 Hz (median gap 0.035 s)** — the async-detect decouple works;
  the detector produces a fresh pose essentially every control tick in healthy runs.
- **Two regimes** (sample BOTH in training): ~15 runs *fed* (pose_age 0.13–0.29 s median), **5 runs
  fully vision-starved** (pose_age pinned at 1.0 s — GPU-contention stall), 4 early runs never got
  vision up.
- **Injection model:** baseline pose-age lag ~U[0.15, 0.30] s (the fed-regime latency floor) +
  **intermittent stall events** that ramp pose_age to the 1.0 s clamp (the starved regime). The lag
  itself is GPU-contention-driven = genuinely stochastic → **this is the one channel that gets DR**
  (per determinism principle). `age_norm = clip(pose_age/0.10, 0, 1)` already saturates at ~0.10 s,
  so most fed ticks read age_norm≈1 — the finer signal for the *coast* is the GRU (T2.4), not this
  clipped scalar. HYPOTHESIS: consider raising `τ_stale` from 0.10 to ~0.5 s for VQ2 so `age_norm`
  is not saturated across the whole fed range (it would then actually discriminate 0.15 vs 0.55 s).
  Flag for Fengyou — it changes the frozen d5 constant.

**Gate-blackout inside ~4.3 m (terminal no-info):** measured — `track_range_m` floors at ~4.3 m on
the pass (gate fills/exits FOV), matching the memory's "gate blackout inside ~4.3 m". Model:
**inside ~4.5 m the vision fix stops updating** (age_norm → ramps to clamp) → the policy must coast
on IMU + belief state. This is the **terminal blind-zone** the GRU (T2.4) exists for.

**Detector good-fix + garbage-lock (fix quality):**
- Detector good-fix ~82% (negv1, from memory) — the *accepted* fix rate. In the data, fix weights:
  **`bearing_w` median 0.06 with 57% of non-null frames below the 0.1 garbage threshold** (heavily
  bimodal toward zero), `zoff_w` median 0.50 (healthier, 34% below 0.1).
- `track_range_m`: 31% null; p50 **10.8 m**; **cap pileup at 25.0 m (2.7%)** — this is the *navigator
  range cap*, not 35; only **2.5% exceed 35 m** (the A34 garbage threshold); max 92 m (rare genuine
  mis-depth). Pre-A34, garbage-lock >35 m mis-depth was the failure mode; A34's abs-range-cap fixed
  the feed.
- **Injection model for the FIX itself (the `fix_surrogate`):** the calibrated surrogate
  (`rl/fix_surrogate.py`) already produces `p_accept(geometry)` + `σ(geometry)` — a range BAND-PASS
  (~0 below 16 m, ~0 above 28 m, peak ~0.83 in 18–26 m) with per-fix σ: lateral 0.10 m, vertical
  **0.28 m** (the binding axis — σ_vert ≫ σ_lat), depth 0.85 m. This is the MEASURED Track-3 noise.
  **VQ2-recalibration is a one-file swap** (`FixSurrogate.from_checkpoints`) — recalibrate from a
  VQ2 at-speed detector recording when available. For no-info regimes, feed the surrogate the
  measured `p_accept` AND inject the **garbage-lock modes** as a low-probability tail: a mis-depth
  fix at >35 m with low weight (`bearing_w < 0.1`) that the policy must learn to distrust via the
  confidence channel.
- **Vision dropout / inconsistency:** modeled as fix-miss (Bernoulli `1 - p_accept`) + the age ramp.
  The smoothing (RewindKF/complementary filter) is IN THE LOOP in both train and deploy — feed the
  FILTER output, so a dropout shows up as rising `age_norm` + widening `c_inplane`, exactly as it
  does live. **Gate jitter** = the per-fix σ + the occasional 25/35 m range jump; model it as the
  surrogate's fix noise, NOT as an independent jitter injection (double-counting).

**Determinism discipline (per the research memory):** exact-match the deterministic channels
(the fix geometry, the known map, the plant), DR ONLY the GPU-contention latency (pose_age) — it is
the one genuinely stochastic channel (host-level GPU arbitration, confirmed not fixable guest-side).

## T2.4 Architecture adds (research conclusions, all pre-decided)

- **Asymmetric privileged critic — ALREADY WIRED.** `algo=appo` + `GuardedAPPO(AsymmetricPPO)`,
  productionized on main (`0392b64`). Critic sees the 37-dim privileged `get_state` (true gate pose,
  GT error); actor sees the 20(+)-dim deploy obs. On VQ1-clean it was insurance; **its payoff is
  exactly this VQ2 obs** (harder partial observability: self-loc + AHRS + noisier vision). Turn it
  ON. (Research conclusion 2 — the #1 fix.)
- **GRU actor for the terminal blind-zone coast.** Research conclusions 4 + 9: a small GRU for
  gate-occlusion memory + the terminal blind-zone coast (gate fills FoV in the final ~4.5 m → vision
  → belief state). This is the memory that carries the drone through the measured ~4.3 m blackout.
  **Add a small GRU to the actor** (not raw-pixel — over the engineered obs). HYPOTHESIS: 1-layer,
  ~64-hidden is sufficient; validate against a no-GRU control arm.
- **Age-of-last-fix channel:** obs[19] `age_norm` IS this (see T2.3 for the τ_stale caveat).
- **Feed FILTER states not raw PnP — the SAME smoothing code that flies.** Satisfied:
  `estimator_obs20` sources from `NavState` (the live filter). For VQ2, the decomposed-fidelity path
  (memory: "surrogate ONLY detection, run REAL PnP + REAL estimator") is the fidelity target — run
  the real RewindKF/complementary filter in training over surrogate detections, so train==deploy on
  the *fusion* side, not just the detection side.

## T2.5 T2 deliverable summary

- **Input-state vector:** obs-20 (table T2.1), optionally +`a_body` (23-dim VQ2 arm).
- **No normal channel** (bearing + known-size range instead).
- **No-info/staleness encoding:** `age_norm` (pose_age, τ_stale caveat), terminal blackout <4.5 m,
  fix-miss via surrogate `1-p_accept`, garbage-lock tail >35 m at low weight.
- **Noise/dropout injection spec + provenance:** pose-age DR U[0.15,0.30]+stall (MEASURED, the only
  DR channel); fix σ [lat 0.10 / vert 0.28 / depth 0.85] (MEASURED Track-3, one-file recal);
  p_accept range band-pass (MEASURED); everything else exact-match deterministic.

---

# T3 — CURRICULUM + LAUNCH PLAN

## T3.1 VQ2-ize the track family

The random-track generator already exists (`rl/peregrine_course.py` `sample_courses` +
`DIFFICULTY_PRESETS`, committed `3d3c917`) and is the pre-VQ2 substrate. VQ2-ize its ranges to the
gate priors (from the VQ2 geometry memory):
- Upright gates (roll=pitch=0), **inner 1.5 m / outer 2.72 m concentric squares, 8 keypoints**.
- Spacing **23.7–38.5 m**; **no next gate > ~30 m** (the usable-PnP / range-cap horizon).
- Height deltas including a **HIGH gate-2-like climb** (the memory's "high gate-2" that the classical
  stack kept climbing into) — this is a load-bearing curriculum case.
- Warehouse scale + the dark-red appearance (detector already trained: `vq2_darkred_negv1`).
- **Add a `vq2_like` difficulty preset** alongside `vq1_like`/`easy`/`medium`/`hard` (the generator
  is designed for exactly this — one preset dict, byte-identical default preserved).

## T3.2 Curriculum stages (slow-is-smooth first, per mission)

1. **Hover / station-keep** (validated twin channels only — the safest ground). Confirms the CTBR
   action map + AHRS + hover point transfer. Gate: stable hover, no divergence.
2. **Single gate, head-on, slow.** No blackout yet (start beyond 4.5 m and stop the metric before the
   blind zone). Gate: pass-rate on one gate with the estimator-emulated obs (NEVER perfect pose).
3. **Single gate WITH the terminal blackout pass** (drive through the <4.5 m blind zone). This is
   where the GRU + IMU-coast earn their keep. Gate: pass-rate holds through the blackout.
4. **Multi-gate lap** (2–3 gates, then full), including the high-climb gate-2 case + a turn between
   gates. Gate: lap completion on estimator-emulated obs.

Slow throughout first (low `forward_accel`, generous cone) — the high-speed rungs are GATED on T1's
above-hover thrust/drag validation (§T1.4). This mirrors the mission's "slow is smooth" and the
classical stack's own A36 "even a tad slower" operator directive.

## T3.3 Reward spine (options — the FREEZE is Fengyou's)

The spine exists in code (`rl/inc8_reward.py`). Present, do not decide:
- **Progress-dominant** (R1' arc-Γ over the corrected-aero reference line + T4 finish-time) — the
  SWIFT/Geles spine, KEEP.
- **Through-approach centering** (`through_centering_reward`) — the cross-track restoring pull that
  was the root-cause fix for "inc8 never flew a lap" (the recenter recipe). **LOAD-BEARING — the only
  inc8 that ever flew had this on.** Include.
- **R5' 2-axis terminal-lock perception reward** (arm A, `perc=0.5`, band-pass w_term peak 12–28 m,
  σ_a 45° / σ_b 29.5°) — exists, arms A/B/C toggle. Camera-pointing / gate-in-FoV.
- **GT-anchored terminal-σ_p0 centering** (`centering`) — the honest un-gameable objective (reward
  the true lateral error at gate crossing).
- **Options for Fengyou's freeze:**
  - (a) **recenter-faithful:** progress + through-centering + R5'-A, `rw_centering` OFF (the proven
    flying config). Safest first launch. **My recommendation for the FIRST run.**
  - (b) **+ dense centering:** add GT-σ_p0 centering (the honest objective) — the design the research
    converged on but never got a clean flight (needs the look-at primitive; see the inc8 saga).
  - (c) **look-at primitive + learned gain** (the S0 architecture pivot) — mechanically breaks the
    pointing exploration wall; the only path that produced pointing→fixes in the whole inc8 saga.
  - **DECISION OWNER: Fengyou.** The 4-iteration inc8 weight-tuning saga (window→entropy→shape→
    incentive, all NO-GO) is the binding lesson: do NOT blind-tune weights; the look-at primitive was
    the load-bearing fix. Whatever freeze, keep the through-centering (course-completion is PRIMARY).

## T3.4 Bootstrap plan if pure RL stalls

Pure vision-RL = 0% (research conclusion 4). The bootstrap ladder (Xing 2403.12203):
**state-teacher (inc7/inc8, perfect-pose) → DAgger student (on estimator-emulated obs) → adaptive-RL
fine-tune (performance-gated).** The state-teacher already exists (inc7/inc8 checkpoints). If the
from-scratch curriculum stalls at stage 2/3, distill inc7 (the flying VQ1 teacher) into the VQ2
student obs via DAgger, then RL-fine-tune. This is the research's pre-decided fallback — hold it in
reserve; don't build it until the pure-RL curriculum demonstrates a stall.

## T3.5 Adroit launch package

- **Derive from `rl/peregrine_inc8_recenter.sbatch`** — the ONLY inc8 recipe that has flown a lap
  (3/3 seeds, symmetric + appo both). It has the FLIGHTCHECK early-stop gate (scancel if
  success_rate~0 & no gates passed by ~step 800-1000) — reuse it. Change to VQ2-ize: `course_mode=
  random` + `track_difficulty=vq2_like`, `algo=appo` (privileged critic), add the GRU flag, feed the
  VQ2 obs (23-dim if `a_body` arm), recalibrated fix_surrogate σ.
- **GPU-spend gate ladder (documented doctrine):**
  1. **SMOKE** (1 seed, arm A, 2048 envs, ~45 min/1 GPU) — GREEN = no-NaN + GPU-sat ≥70% + sidecar
     dims correct + FLIGHTCHECK not-immediately-dead. Commander's GO.
  2. **PORTFOLIO** (reward arms ×seeds, the recenter 3-seed shape) — eval BOTH closed-loop
     estimator-emulated AND passive-observer, map-ON, NEVER perfect pose.
  3. **RUNGS** (speed ladder on the WINNING arm ONLY) — staged, gated on T1 above-hover validation.
- **AUP compliance (binding, from the Adroit memory):** SLURM only (NO compute on login nodes); NO
  internet on compute nodes (pre-download git/pip/conda/YOLO weights on login/vis BEFORE submit);
  output → `/scratch/network/fl3689` NOT `/projects`; accurate `--mem` + 1-core serial (over-alloc
  or multi-core-serial → SUSPENSION); zero-GPU-util killed at 2 h (PPO must saturate the GPU);
  `checkquota` routinely (home 9.3/10 GiB tight → jobs to /scratch); `module load anaconda3/2024.10`
  (no default); envs pre-built. `/scratch/network/fl3689/peregrine_repo` is a FILE COPY not a git
  clone. Evals default the legacy flat plant → all evals must run **map-ON**.

## T3.6 🚩 ADROIT BRIDGE — what needs re-establishing on ShadowPC (FLAG)

We migrated laptop→ShadowPC (Gen 8 handoff). The Adroit access path used an **SSH `serve` daemon +
Duo** via the **`adroit-connector`** tool. **That connector is NOT in the ShadowPC checkout** (only
`memory/reference_adroit_princeton.md` documents it; no `adroit.py` / connector venv on ShadowPC).

**What the bridge WAS (from the memory):**
- `adroit-connector/` — its OWN venv with `paramiko`; SSH key is **passphrase-encrypted** → only
  **`serve`** works (caches the key after ONE passphrase + ONE Duo, then `x "cmd"` reuses it).
  `run`/`submit`/`upload` each re-prompt the passphrase → unusable non-interactively → drive
  everything through `serve` + `x`.
- File transfer = `adroit.py upload <local> <full-remote-FILE-path>` (paramiko SFTP; pins
  `hmac-sha2-256` because the campus VPN corrupts the default MAC → **raw scp FAILS**). Run in
  PowerShell via the connector venv. `pull_file.py` = download.
- The daemon **idle-drops often** (each drop → a reconnect Duo in the serve window).

**What Fengyou MUST do (only he can):**
1. **Verify Adroit access is still active** (the memory flags summer-recess access loss for
   non-enrolled accounts — confirm with Research Computing).
2. **Approve the Duo push(es)** — every `serve` start + reconnect needs his phone.
3. **Enter the SSH key passphrase** in the `serve` window (getpass — interactive only).
4. **Confirm the VPN** is up (needed for the HMAC-pinned transfer + login).

**What WE can do (once Fengyou has serve up):**
1. **Re-stage the `adroit-connector` on ShadowPC** — copy/re-create the connector + its paramiko
   venv (the tool itself is small; needs paramiko + the pinned HMAC connect logic). HYPOTHESIS: it's
   recoverable from an old-laptop backup or the git history of a sibling repo; if not, it's a
   ~1-file paramiko wrapper to rebuild from the memory's spec. **Confirm with Fengyou whether the
   connector source exists to copy, or must be rebuilt.**
2. Upload the repo tarball + VQ2 obs/reward/curriculum code + the recalibrated fix_surrogate.
3. Pre-download the env deps on the login/vis node (AUP).
4. Submit the smoke `sbatch` and read the TB trace (via `x "cat …"` or `pull_file.py`).
5. Drive the gate ladder (smoke → portfolio → rungs) with commander adjudication at each gate.

**Sequence:** Fengyou re-establishes access → we re-stage the connector → smoke → ladder.

---

## Appendix — probe reproduction

```
.venv\Scripts\python.exe scripts\vq2_loadday\twin_fidelity_probe.py \
    --glob ".../data/runs/*_rl_s1_f1" --json twin_probe.json
```
Reports per-run + aggregate gyro-response and specific-force residuals, `measured` and `flat` plants.
FALLBACK tier only (competitive wire). The 3 cleanest long flights (130314, 120357, 032626) are the
best estimate of true (contact-free) fidelity. JSON verdict schema: `{aggregate, per_run, thresholds}`.

## Appendix — open questions surfaced (for commander/Fengyou)

1. **τ_stale = 0.10 s** saturates `age_norm` across the whole fed regime (pose_age 0.15–0.55 s all
   read ~1). Raise to ~0.5 s for VQ2 so the channel discriminates? (Changes a frozen d5 constant.)
2. **Add `a_body` (felt-accel) obs channel** — Fengyou explicitly wants it; it's a 23-dim VQ2 arm.
   Default first run, or a portfolio arm? (Recommend: portfolio arm, validate it earns its keep.)
3. **Reward freeze** (T3.3 options a/b/c) — Fengyou owns.
4. **Command-sweep capture** to close T1's above-hover thrust/drag gap before the high-speed rungs —
   schedule it? (Recommend: yes, in parallel with the slow curriculum.)
5. **Adroit connector source** — does it exist to copy from a laptop backup, or rebuild from spec?
