# Vision & Estimator Sub-Index
Mid-level index for the estimator race-speed verdict, VISION-PKG2 specs, gate mapper, case-C readiness, estimator robustness, and advisor-triage queue. Deep detail in topic files below.

## ESTIMATOR-RACESPEED VERDICT (2026-06-13; handoff/ultracode-estimator-racespeed-2026-06-13/REPORT.md)
- **Case-C absolute world-frame nav = NO-GO at ANY speed:** deployable in-plane ~0.55 m vs 0.155 m gate-4 margin (3.5×); SPEED-FLAT.
- **FIX = GATE-RELATIVE** (map bias drops out → 0.11–0.21 m RMS = **CONDITIONAL-GO**, velocity-prior-sensitive; straddles 0.155 m).
- **RewindKF = DEFAULT** (ships regardless; already built).
- Policy obs must be gate-relative → **P4-C05 DONE (f50b9b4, 692 green) — see §P4-C05 below.**
- 🚩 **ORGANIZER-PIVOT: build gate-relative REGARDLESS** — Q① = load-bearing vs free-insurance only (3 emails UNANSWERED; do NOT gate engineering on organizer answers; keep ONE nudge ~weekly).
- COUPLING: faster speed worsens gate-4 margin/σ ratio → estimator accuracy must be verified at each speed rung. **PRIMARY MARGIN GUARD = GATE-4** (0.155 m @ r=0.38; reporting discipline: always quote margin as function of r ∈ {0.21,0.26,0.30,0.33,0.38} at p90/p99 — never a single number; central planning radius = 0.30 m, worst-case stress knob = 0.38).

## COLD-MARGIN CLOSURE (2026-06-13 offline; L3 at-speed 2026-06-14; δ_map discriminator 2026-06-14; BLUEPRINT.md + REPORT.md in handoff/ultracode-gate-relative-pipeline-design-2026-06-13/; SHADOWVISION + L3 + DMAP UPDATES below)
**VERDICT: CANNOT-SETTLE-OFFLINE SURVIVES — L3 "CLOSE" REFUTED by δ_map discriminator (case b, high-confidence; 2026-06-14; branch worktree-agent-a88bc717008e79cd9, commit d7c592e, NOT merged; handoff/dmap-vert-discriminator-2026-06-14/REPORT.md).** Offline (tri-confirmed): d3 full-lap sim + v3 verifier + commander cross-check; margin does NOT close offline. L3 at-speed (2026-06-14): gate-4 chain bias lateral ≈0 / vertical −0.215 m (common-mode). FORMER HYPOTHESIS (case a, now REFUTED): δ_map offset → +L CANCELS → ε≈0 → close. ACTUAL (case b): no map offset → ε_vert ≈ +0.15…+0.26 m = genuine PERCEPTION/ATTITUDE sighting bias → +L does NOT cancel → margin stays OPEN. L3's own rel in-plane p90 = 0.492 m does NOT close at r=0.30 (budget 0.235) or r=0.38 (budget 0.155). The gate-relative OBSERVATION fix is SOLID and MUST be built (map bias drops EXACTLY: rel E_bias −0.000 vs abs/submap +0.176; RMS 0.139 m). → §δ_MAP VERTICAL DISCRIMINATOR below; §L3 AT-SPEED above.

### SHADOWVISION UPDATE (2026-06-13; shadow mode on inc7 given-pose flights at gate-4 band, range 20–26 m)
**MEASURED vision precision — BETTER than modeled:**
- **Accepted-fix lateral (cross-track) σ ≈ 0.10 m** (was modeled 0.265 m in DR/cov; ~2.5× tighter); ~zero bias.
- **Along-track (depth) σ ≈ 0.8 m** — weak axis; this is WHY the KF fuses velocity prior (correct design).
- Detector: ~90% detection in-image; FINE.
- **🚩 NEW BINDING LEVER = camera pointing / fix DENSITY (NOT per-fix σ):** only ~7% of frames yield an ACCEPTED FIX (vs ~47% modeled). Root cause: inc7 64° racing crab aims camera off-gate; only ~33% of frames have any gate centre in-image (gate sits near ~45° FoV edge). This is a TRAINING/POLICY-addressable problem, not a sensor limit. Inc7 on given pose has ZERO incentive to aim the camera — a vision-trained policy should do markedly better.
- **Update DR/cov table: per-fix lateral σ = 0.10 m (MEASURED), NOT 0.265 m.** Fix-RATE is the new binding variable for the gate-4 margin (was treated as ~47%; measured ~7% un-pointed; improvable via de-crab/camera-pointing reward).

**HOPEFUL SYNTHESIS:**
- With measured σ=0.10, the old speed-gate (σ≤0.10 → >55 m/s viable) implies the margin VERY LIKELY CLOSES at race speed IF fix density is adequate.
- Denser, tighter fixes attack the WHOLE problem at once: per-fix error (0.10 solved), cold velocity prior (faster convergence), AND attitude-bias drift (corrected more often).
- **Unified path to close gate-4 margin = camera pointing/fix-density (primary lever) + ESKF attitude-bias estimation (secondary, residual worst-case).** This SOFTENS the CANNOT-SETTLE-OFFLINE picture: σ is great; live confirmation now hinges on fix density via pointing.
- CAVEAT: 7% fix-rate is a LOWER BOUND — inc7 given-pose has no pointing incentive; vision-trained policy expected much higher.
- Source: handoff/simops-mastery-2026-06-13/ (on ShadowPC — pending commit+push from ShadowPC to main).

### 2-AXIS CAMERA-POINTING FINDING (2026-06-14; FIX-SURROGATE calibration; handoff/fix-surrogate-2026-06-14/REPORT.md)
**ELEVATION CO-BINDS — decrab is NECESSARY but NOT SUFFICIENT.**
- After de-crabbing (yaw — translationally ~free), the active gate is in horizontal FoV **100%** but vertical FoV only **~10%** (|el| ~45° vs VFoV 29° at +20° mount). Centering the gate through its 18–28 m window = **×8.5 in-window fix-density gain** (accept_density: 0.099 → 0.84).
- This PROMOTES elevation from a flagged "second-order" concern to a **co-binding axis**, refining the earlier "decrab ≈ free solves pointing" framing. Elevation couples to PITCH = the accel knob (harder axis than yaw).
- 🚩 **MOUNT-UPTILT LEVER DEAD — SPEC §3.8 CONFIRMS MOUNT IS FIXED AT 20° EXACTLY (no tolerance, no range).** The camera origin is at the body origin (zero translational offset / no lever arm). Therefore the "mount-uptilt sweep / cheap mount knob" (banked from FIX-SURROGATE finding 2026-06-14) is OFF THE TABLE. The elevation half of camera-pointing (gate out of the 58.7° VFoV at race pitch) MUST be solved by the POLICY (inc8 pitch/pointing — pitch-coupled axis), NOT by hardware tilt.
- crab→fix-rate map is a callable + table; its accept_rate is a range-MARGINAL LOWER BOUND — use accept_density_in_window for the real in-window number.
- **Inc8 camera-pointing reward spec: 2-AXIS (azimuth + elevation)**, not azimuth only.

### FIX-SURROGATE MERGED (2026-06-14; merged to main ffb2c74 — code + calibrated checkpoints only, branch memory excluded; 15 green; green-by-construction. C6/POC hold INTACT — offline training infra, touches nothing live. Branch claude/stupefied-fermi-20126c commits 0610591+c0d3f49)
- **What:** analytic NO-RENDER fix model for inc8 training. `rl/fix_surrogate.py` + `tests/test_fix_surrogate.py` (15 green) + `handoff/fix-surrogate-2026-06-14/REPORT.md`. CPU-only, torch-free. src/ UNTOUCHED.
- **Interface:** geometry(drone_pos, R_world_body, gate) → GT relative geometry, mirroring the DEPLOYED camera chain (same intrinsics, +20° mount, predict_gates_in_camera). `p_accept(geom)` (range band-pass × in-image), `fix_sigma`/`fix_covariance` (gate-plane anisotropic SPD, mirrors `gate_relative_inplane_fix`), `sample_fix(geom,rng)→(z,cov)|None` feeds LinearKF.update_position directly; `crab_to_fix_rate` + `accept_density_in_window`. INFRA ONLY — reward NOT decided (Fengyou owns brainstorm).
- **Calibration vs Track-3 (1821 frames / 126 fixes), within tolerance:** accept @ peak(18–24 m) 0.835→0.826; accept @ inc7-crab 0.069 (anchor); in-FoV 0.332; σ lateral 0.104 (CI68 .094–.114) / vertical 0.282 / depth 0.852. Floors carry ±15–18% small-n CI (~16–24 rows/flight).
- **Validated NOT overfit:** leave-one-flight-out CV (held-out accept AUC .862 vs .867; lateral floor .098 vs .104) + 4-agent adversarial workflow (concerns addressed). CPU smoke: sample_fix→LinearKF mean Mahalanobis 3.00 = dof, 20000/20000 SPD/finite.
- **σ is a SINGLE swappable checkpoint** — `fix_surrogate_fit.py` recalibrates from an L3 at-speed recording.
- Full suite: 703 passed / 35 skip / 0 fail = no regression.
- 🚩 **SURROGATE σ vs L3 GATE-4 σ RECONCILIATION REQUIRED BEFORE LOCKING THE MARGIN σ:** surrogate POOLED σ (lat 0.104 / vert 0.282) ≠ L3 GATE-4 at-speed σ (lat 0.19 / vert 0.10) axis-for-axis. NOT a bug — different purposes (surrogate = training DR distribution; L3 = gate-4 margin number). Reconcile axis-convention vs per-gate/speed variation BEFORE locking the σ the margin verdict uses.

### Corrected framing (supersedes any "c1 0.139 clears margin" language)
- c1 0.139 = **RMS** (clears on RMS); p90 = **0.203 m** (does NOT clear 0.155 m). The mandated worst-case read is **p90**. Stop citing c1-warm RMS 0.139 or d4v-visvel RMS 0.144 as "clears" — both are p90-FAILS (0.203 / 0.213).
- d3 anchor reproduces c1 exactly (0.139 RMS / 0.203 p90, N=600) — machinery validated.

### Contact-radius reconciliation (2026-06-13; handoff/body-contact-reconcile-2026-06-13/)
**The 0.155 m budget = budget(r=0.38) = (0.75 − 0.38) − 0.215, where linf₀=0.215 m is the radius-invariant gate-4 simstart crossing offset [X].** Budget identity confirmed bit-for-bit vs d3's MARGIN_G4=0.155.

**Reconciled contact radius:** 0.38 is NOT geometric — rigid-body chassis geometry caps at **0.2135 m** (3D half-diagonal √(0.14²+0.14²+0.08²); tilt adds only +0.015 m; posture-free absolute ceiling). The extra ~0.17 m is an unmodeled rotor-wash/blade-strike halo (props UNDOCUMENTED in spec). Posture-matched empirical data (gate-3 steep crashes, tilt ~55°, 12–18 m/s) logs contacts at L-inf 0.37–0.49 m → r_eff ≥ 0.26–0.38 (LOWER BOUNDS). **Central best-estimate = 0.30 m (band 0.26–0.33); worst-case tail = 0.38 m** [X].

🚩 **DO NOT adopt 0.18–0.20 m (gate-0, near-level corner-pass probe) as the gate-4 radius** — that is the WRONG POSTURE (~3 m/s CTBR bridge, near-level; not gate-4 steep). DO NOT re-litigate 0.38 as a geometry bug.

**Budget table (budget(r) = (0.75−r) − 0.215):**
| r | budget | vs incumbent |
|---|---|---|
| 0.213 (geom floor) | 0.322 m | +0.167 |
| 0.26 (best-est lo) | 0.275 m | +0.120 |
| **0.30 (central)** | **0.235 m** | **+0.080 (~1.5×)** |
| 0.33 (best-est hi) | 0.205 m | +0.050 |
| 0.38 (worst-case) | 0.155 m | — |

### Cold verdict @ gate-4, 37 m/s — p90/p99 full matrix
| mode | p90 | vs budget@0.30 | vs budget@0.38 | p99 clears? |
|---|---|---|---|---|
| **warm (vel=truth — NOT case C)** | 0.203 | PASS (+0.032) | FAIL | Y(r≤0.38) |
| **cold @ bias 0 (case-C ideal)** | 0.234 | PASS (+0.001) KNIFE-EDGE | FAIL | FAIL (p99=0.322; clears only at r≤0.213) |
| **cold @ 1.4° attitude bias (REALISTIC)** | 0.338 | **FAIL (−0.103)** | **FAIL** | **FAIL at ALL r** |
| cold@1.4° in-plane worst | 0.348 | FAIL | FAIL | FAIL at ALL r |

**Verdict (QUALIFIED-NO-CLOSE):** With r=0.30 (central), warm and cold@bias0 p90 clear — but cold@bias0 is KNIFE-EDGE (+0.001 m) and **FAILS at p99** (p99=0.322 > budget 0.235). The realistic case-C tail (cold@1.4°) **fails at every admissible radius at p90 AND p99** — needs budget > 0.338 → r < 0.197 m, **below the chassis geometry floor (0.2135 m)**. Radius correction is REAL but operationally inert for the binding case.

EVERY {σ_v × accel-bias} cell in the grid is `p90_clear=false` (v3b re-run). RewindKF latency 15→115 ms ~equal once capture-timed — **NOT binding.**

### Binding factor (SHARPENED by radius reconciliation + SHADOWVISION + δ_MAP DISCRIMINATOR)
- **The binding factor = VERTICAL BORESIGHT BIAS ε_vert ≈ 0.215 m (~0.56° camera-pitch mismatch, CALIBRATABLE) + effective systematic attitude/accel bias, NOT the contact radius and NOT per-fix σ.** The radius correction (~1.5× budget) does not move the binding case. The cold@1.4° p90 (0.338 m) fails at every physically admissible radius.
- **🆕 δ_MAP DISCRIMINATOR (2026-06-14):** δ_map_vert ≈ 0 across all 6 gates — L3 "CLOSE" was REFUTED. L3's vertical −0.215 m = ε_vert (genuine perception/attitude sighting bias; +L does NOT cancel). See §δ_MAP VERTICAL DISCRIMINATOR above. Path to close: boresight calibration + ESKF attitude-bias estimation.
- **SHADOWVISION SHARPENING:** per-fix lateral σ IS SOLVED (0.10 m measured). The remaining binding factors are **camera pointing / fix DENSITY** (primary lever, policy-addressable via inc8 gate-in-FoV reward) + **ESKF attitude-bias estimation + boresight calibration** (co-equal margin levers — PROMOTED by δ_map finding). See §SHADOWVISION UPDATE and §δ_MAP VERTICAL DISCRIMINATOR above.
- **HOPE FLAG:** ε_vert ≈ 0.56° is CALIBRATABLE (boresight calibration / ESKF attitude-bias state). After calibration, the bias-free regime is the correct design point. Airtight lock test pending (morning ShadowPC static head-on fix at g2/g4).
- **ESKF attitude-bias estimation (bias state) = CO-EQUAL MARGIN LEVER** (PROMOTED from secondary by δ_map finding; now co-equal with 2-axis camera-pointing/fix-rate). Confirms d3's "ESKF/attitude pipeline in the critical path for the MARGIN, not just absolute nav."
- Cold-prior risk = **HIGH** (variance/window-driven, NOT init-driven). "Warm by gate-4" is FALSE — a full lap does NOT pre-converge velocity to warm quality. (But denser fixes via pointing ACCELERATE cold-prior convergence.)
- Closure rests on **boresight calibration (ε_vert→0) + camera pointing/fix-density (fix-rate) + attitude/accel-bias estimation + uncertainty-aware speed-down**, NOT the velocity channel and NOT the contact radius and NOT per-fix σ (solved).

### Vision-velocity channel — REFUTED, demoted to P2 insurance (SUPERSEDES "LOAD-BEARING" framing)
- ~~**LSQ-over-window (σ_v 0.3–0.4, d4v visvel):** p90 cold 0.34→**0.18 m**. Build this. NECESSARY.~~
- **REFUTED (verify_velchannel_v6.py, re-run):** honest inter-frame PnP-delta σ_v ≈ **2.81 m/s** (NOT assumed 0.3–1.0); mandated RewindKF L≈115 ms → cold in-plane RMS 0.19–0.35 m **NO-GO at every smoothing window**. d4v ran at L=0 with GT-cheat velocity — those "0.18 m" numbers were fiction.
- **DEMOTED to P2 insurance.** Position-fix-differencing IS the KF (free baseline). Direct vision-velocity is NOT the margin lever. Do NOT make it load-bearing.
- **Naive fix-difference (σ_v~5.2):** p90 ≈ cold or worse. Same verdict.

### Speed-ladder and escape hatch
- **Speed-ladder selection metric = p90/p99 gate** (gate-4 SIMSTART in-plane — report across r ∈ {0.21,0.26,0.30,0.33,0.38}, ALWAYS p90 AND p99; central on 0.30; 0.38 as worst-case stress knob; NEVER a single number tied to one radius), NOT RMS.
- **ESCAPE HATCH = ShadowPC at-speed (~37 m/s) gate-4 recording (L3):** ≥5 laps, all fields on ONE clock, GT position+velocity. GT velocity = **LOCAL_POSITION_NED.{vx,vy,vz}** (world NED, ~97 Hz, live-verified to 21.3 m/s; stored as DroneState.velocity_ned); ODOMETRY twist as cross-check. Pins: **true effective attitude/accel bias** (secondary binding factor), realized cold velocity-prior distribution, per-fix σ at real blur (σ=0.10 is current lower bound — measured at inc7 speed; may tighten at 37 m/s with blur), one-signed PnP bias magnitude+sign. **Key open question: what fix DENSITY does a vision-trained policy achieve? (primary binding factor).** If measured bias ≤ ~0.6°, margin CLOSES at r=0.30. Plus one eval-HW latency run (L4) + TIMESYNC wire trace.
- **🆕 Recording harness VALIDATED:** verify_bundle.py ALL_PASS (video↔LPN 2.5 ms p50 recv-clock-aligned). Tooling in handoff/simops-mastery-2026-06-13/ on ShadowPC — **pending commit+push from ShadowPC to main.** frame_residual_report.py mirror canary TRUE +0.97 / AS-IS −0.81 = OK (telemetry convention stable).

### Obs sign — +L CORRECT IN CODE; spec PROSE wrong; pinned by test (2026-06-13)
- 🚩 **Obs sign = +L and was ALWAYS correct in code.** `obs_from_zup:348 = R_w2g @ (gate_pos − pos)`; `localization.py:86` builds the +L lever. There was NEVER a code bug.
- The d1/d2 spec PROSE "estimator delivers −L" was wrong prose, not wrong code. The 24 m "flip" is the cost a FUTURE C2 estimator→obs interface would pay IF it implemented −L.
- **Pinned by `tests/test_obs_sign_faithfulness.py`:** +L end-to-end identity 4.77e-7 (≤1e-5 pass) AND a −L sign-flip negative control that BREAKS at 24.0 m. The future C2 estimator→obs MUST deliver +L — the test will catch any regression.
- d1's "0.0 bit-exact unification proof" was a tautology (tested identity within one frame, never exercising the NED↔Z-up conversion). The adversarial faithfulness test supersedes it.

## OBS CONTRACT (2026-06-13, FROZEN — d5 layout wins; resolves d1/d5/d6 seam)
- **obs_dim = 20.** [0:17] unchanged, bit-exact (d1 faithfulness proof holds). [17] c_inplane, [18] c_along, [19] age_norm.
- [17]/[18] = `clip(σ_ref/σ_hat, 0, 1)` bounded **confidence ratios** (σ_ref ≈ 0.05 m). **NOT raw σ in metres** (raw σ has no natural scale → explodes PPO obs-normalizer).
- [19] = `clip(t_since_last_accepted_fix / TAU_STALE, 0, 1)`, TAU_STALE ≈ 0.10 s.
- `σ_hat` = **CALIBRATED gate-frame KF covariance (NEES≈3)** supplied by RewindKF — calibration is the binding interface requirement on the estimator.
- **Critic get_state 33→36** (same triple + ground-truth gate-relative state added for critic).
- **Deploy:** `fly_rl` gates new dims on checkpoint sidecar obs-dim → inc7 (17-dim) still runs unchanged.
- **WHY d5's 20 over d1(18/19)/d6(18):** cold verdict confirms cold/coast regimes are real; confidence + staleness channels are load-bearing (not cosmetic). The encoding choice (bounded ratio vs raw) is locked — do not reopen.

## VISION-PKG2 specs (2026-06-10)
- **Detector: SHIP v2** (`models/gate_yolo11s_curriculum_v2.pt`, multi-gate). v3 `--hard` = NEGATIVE.
- World-fix σ≈**[0.73, 0.47, 0.29] m (N,E,D)**, range-flat to ~24 m; acceptance ~47%, leak 0.53%.
- Improvements: 1.4° attitude lever + 0.40 m cov floor + 32 m cap; over-rejection 15.7%→1.2%.
- ✅ **VISION-FRAME-FIX (8d7b0b3):** east bias eliminated.
- 🚩 **`corner_to_center` 180° flip FIXED (37e7ab1).**
- sigma_theta=1.4° re-fit **INCONCLUSIVE (P2-OFFLINE-ANALYSIS, 2026-06-13):** at-speed MLE = 0.46° on N=28 fixes (681 frames → 36 depth-sane → 28 MLE-eligible) — UNDERPOWERED (prior used N=165); floor collapsed to 0.00 m (inconsistent); +0.67 m systematic N-bias (unmodeled intercept) deflates estimate. **1.4° + 0.40 m floor PRODUCTION CONSTANTS STAND** (escape hatch invoked). Directional: σ_theta likely smaller at speed (PnP-dominated at ~15–18 m/s); refit DEFERRED pending N≥100 at-speed fixes (awaits inc8-class faster flights). 🆕 **+0.67 m N-bias corroborates gate-relative plan** (per-track world-frame bias drops out in gate-relative obs; cross-ref gate-relative rebuild).

## Gate mapper
- ✅ **GATE MAPPER COMPLETE:** `src/racer/gate_mapper.py` cases A/B/C.

## Case-C readiness (STACK-REVIEW-VQ2)
- Architecture **AFFIRMED**. Case-C **GO-WITH-CONDITIONS** (binding = unmeasured in-loop latency L).
- **3 P0 bugs FIXED (P0-CASEC-FOUNDATION, pending merge):** _initialize seed gating / IMU-clock timestamping / cold-start guard-tested. See §P0-CASEC-FOUNDATION below.
- Prototypes in `handoff/ultracode-vision-case-c-2026-06-13/`.
- Full-lap case-C sim at ~37 m/s gate-4 recording = collapses dominant uncertainty (queued §⑩).

## 30 Hz speed-ceiling analytic
- 30 Hz = inherited default (DiffAero racing.yaml). Binds ONLY at ≥30 m/s × last-fix ≤10 m; fixes to 15 m → NOT bottleneck.
- 🚩 GATING MEASUREMENT queued (SHADOWPC-VISION-CAL).

## Spec facts (§3.8 camera — REVIEWED 2026-06-14)
- **Camera tilt = 20° UP from body +X (MAV_FRAME_BODY_NED). SPEC-EXACT — stated ONCE, NO tolerance/range. Mount is FIXED; mount-uptilt lever is DEAD.**
- **Zero translational offset** — camera origin = body origin (no lever arm).
- Pinhole, NO lens distortion. Resolution 640×360. [cx,cy]=[320,180] (dead-centre, no principal-point offset). [fx,fy]=[320,320] (square pixels).
- Vision stream: 30 Hz, 640×360 JPEG, UDP port 5600. Sensors: HIGHRES_IMU, ATTITUDE, TIMESYNC. Body↔IMU = identity. Physics 120 Hz, command <100 Hz.
- 🚩 **VFoV=90° IN SPEC IS MISLABELED — IT IS HFoV. TRUE VFoV≈58.7° (±29.35°); HFoV=90.0° exactly.** Derivation: fy=320, H=360 → 2·atan(180/320)=58.7°; fx=320, W=640 → 2·atan(320/320)=90.0°. ALWAYS decode vertical from fy=320 — NEVER from a literal 90° VFoV. Audit that render-prediction AND PnP both use fy=320.
- **Boresight ε_vert ≈ 0.56° reconciled with spec:** 20° is spec-EXACT, so the 0.56° boresight is NOT a wrong mount angle — it is a CONVENTION/PROJECTION seam (spec explicitly offloads the body→camera→image-library frame rotation) and/or fy/VFoV mishandling. Code-side and CALIBRATABLE (0.56° ≈ 3.1 px vertical). NEXT ACTION: vision-extrinsics audit (body→cam→image rotation + fy=320 consistency).
- ~100 TOPS onboard. Obstacles exist but don't map monocularly.

## Parallel onboard systems
- **APPROVED:** async detector→KF; ROI zoom; MPC-shadow; critic-as-risk.
- **REJECTED:** map mutation; ensemble voting; wind estimator; in-race adaptation.

## CR1-01 settled (2026-06-13)
- Navigator KF IMU-predict frame NOT the case-C error source: `navigator.py:313` correct (HIGHRES_IMU accel_body↔TRUE-conjugated, live-verified). Removes one estimator-path concern. Gate-relative rebuild remains the path.

## In-loop latency L (measured 2026-06-13)
- L ≈ 115 ms median CPU-only ShadowPC (detect 109 + transport 4 + PnP 1.6 ms). GPU host ≈ 15–25 ms. CONFIRMS RewindKF default.
- inc7 live top speed ≈ 21.3 m/s gate-4 approach (median 20.1); 681 vision frames recorded at speed → extends vision data from ≤8.4 m/s toward race speed. Full 37 m/s CONDITIONAL-GO validation AWAITS inc8 envelope-relaxed policy.
- P2 DEFERRED (data banked: instr_cr1b = video + tlog + debug_obs at speed): sigma_theta 1.4° refit + per-tick gate-crossing/bimodal residual → dispatched as sonnet offload (P2-OFFLINE-ANALYSIS).

## Advisor-triage open queue
- 🚩 **ESTIMATOR = CRITICAL PATH** (RewindKF + gate-relative rebuild, gated on estimator report).
- ① Photoreal detector (Blender/Cycles next).
- ② ONNX/TRT latency.
- ⑤⑥ Post-merge: at-speed gate-4 recording + 2-corner PnP + Bayesian-IoU extrinsic + per-gate last-fix distance + sigma_theta refit + yaw-active debug_obs capture.
- K+L+N PLANNED. REJECTED: HSV pre-filter.

## P4-C05 DONE (f50b9b4, 2026-06-13; suite 687→692 green)
- **Fix:** opt-in `GateMap` (`make_gate_map`) threads runtime per-gate yaw through `fly_rl.obs_from_zup` / `build_obs` + `offline_rollout.py` + `contact_true_eval.py`.
- **Yaw-aware path** reproduces `peregrine_racing.get_observations` BIT-EXACTLY (0.0, all 17 dims).
- **Default `gate_map=None`** keeps exact VQ1 yaw=π specialization (bit-exact 0.0 over 2000 tilted states, both virtual_flip modes); old hardcoded-π diverged by exactly the audit's 4.22 m on `nxt_rely`.
- **`get_gate_rotmat_w2g(gate_yaw[tg])`** is THE FOUNDATION HOOK for the gate-relative estimator rebuild (same code path, same rotation matrix).
- `tests/test_confirmed_p4_c05.py` expanded 5→10 tests.
- 🚩 **DEPLOY CONSTRAINT (current stack VQ1-only):** three loud guards now ship — `_assert_vq1_constants_consistent` (import-time, half-migrated constants), `_assert_live_course_is_vq1` (deploy-time, aborts if any gate yaw ≠ π before RL loop starts), `assert_gate_map_allpi` (public). **The current deployable stack fail-loud-aborts on any non-π/VQ2 course until gate-relative path is wired + `gate_map` passed.** Correct fail-loud behavior; removed by the gate-relative rebuild. → [[index-control-sim]]

## L3 AT-SPEED GATE-4 BIAS PINNED — "CLOSE" NOW REFUTED (2026-06-14; branch claude/blissful-kalam-75f52b; discriminator refutes case-a → see §δ_MAP VERTICAL DISCRIMINATOR)
- **Recording:** 13 clean inc7 laps, 0 gate contact, FINISHED 6/6, fully unattended. GT vel live to 21.4 m/s. R_y(π) canary OK (mirror +0.97 / as-is −0.81 → bias REAL, not a frame flip). Offline shadow analysis replayed production C2 chain vs GT, gate-4 focus, N=81 sightings @ 22.3 m / 17.4 m/s.
- **Gate-4 effective in-plane FIX-vs-GT bias:** lateral −0.04 ≈ 0; VERTICAL −0.215 m. Vertical is COMMON-MODE (gate-2 −0.211 ≡ gate-4 −0.215).
- **L3 FORMER HYPOTHESIS (case a — REFUTED by δ_map discriminator):** −0.215 vertical = track_map MAP offset δ_map → +L CANCELS → close. REFUTED: δ_map_vert ≈ 0 (max 0.077 m across 6 gates) → rel_vert −0.215 m = ε_vert (perception/attitude sighting bias) → +L does NOT cancel → margin stays OPEN. L3's own rel in-plane p90 = 0.492 m (NOT 0.197 m claimed under case-a) → does NOT close at any r. See §δ_MAP VERTICAL DISCRIMINATOR for full analysis.
- **σ (gate-4 specific):** lateral 0.19 m / vertical 0.10 m (< modeled 0.265 m); along-track +0.40 loose. NOTE: gate-4 σ_lat 0.19 is ABOVE the pooled headline 0.10 — use gate-4 value for gate-4 margin; the "σ=0.10" pooled figure was pooled across all gates.
- **Relinnov gate:** accepts 100% of offered gate-4 fixes (d2 p90 0.94 ≪ 13.82) — NOT the limiter.
- **BINDING LIMIT = FIX-RATE 1.4% of frames at gate-4** (gate-4 is worst case; pooled ~7%): detector 92%; gate-4-in-FoV 42%; crab p50 66°; at >60° crab (51% of frames) gate-4-in-FoV 5% / 0 fixes; ALL 81 fixes from the crab-30–45° gate-3-approach window. Root cause = camera POINTING **2-AXIS** (azimuth + ELEVATION co-bind — see §2-AXIS CAMERA-POINTING FINDING; mount FIXED → elevation fix-rate is POLICY-only via inc8 2-axis gate-in-FoV reward).
- **Depth blow-up:** clean fixes only in the ~20–22 m band; gate-3 @ 31.8 m showed +2.05 m along-track depth blow-up (beyond 32 m cap) — consistent with the a1·r >24 m extrapolation warning.
- 🚩 **FOOTGUN: COMMON-MODE-ACROSS-GATES ≠ PROOF OF +L CANCELLATION.** Identical vertical offset at multiple gates is consistent with BOTH (a) track_map representation offset [+L CANCELS → close] AND (b) detector/camera-EXTRINSIC sighting bias [+L does NOT cancel → fails] — one camera sees every gate, so extrinsic bias is also common-mode. Only the DIRECT δ_map pin distinguishes them. **The discriminator chose (b).**

## δ_MAP VERTICAL DISCRIMINATOR — CASE (b) CONFIRMED; L3 "CLOSE" REFUTED (2026-06-14; branch worktree-agent-a88bc717008e79cd9, commit d7c592e, NOT merged; handoff/dmap-vert-discriminator-2026-06-14/REPORT.md)

**VERDICT: CASE (b) — the L3 gate-4 "CLOSE" is REFUTED. CANNOT-SETTLE-OFFLINE SURVIVES.**

### Direct δ_map measurement
- δ_map_vert (track_map opening-z − true-opening-z) = gate-2 **+0.044 m**, gate-4 **−0.067 m**; MAX across 6 gates **0.077 m**. There is **NO −0.2 m map offset**.
- MAP IS GOOD VERTICALLY: direct δ_map ≈ 0; piece-F (ultracode-vision-case-c) GT-transit resurvey D-residual σ = 0.03 m (max 0.10); contact-free GT crossings within ±0.07 m of track_map OPENING centres at all 6 gates.

### Decomposition of L3 vertical bias
- L3's rel_vert −0.215 m decomposes: rel_vert = δ_map_vert − ε_vert → ε_vert ≈ **+0.15…+0.26 m** = a genuine **PERCEPTION/ATTITUDE sighting bias**.
- The +L gate-relative fix does **NOT cancel ε_vert** — it removes only the map-position term. ε_vert survives.
- **L3's OWN shadow_gate4.py: rel in-plane p90 = 0.492 m → does NOT close at r=0.30 (budget 0.235) or r=0.38 (budget 0.155).** The L3 report flipped to "CLOSE" only by ASSUMING case (a).

### Binding factor identified
🚩 **BINDING FACTOR = VERTICAL BORESIGHT BIAS ε_vert ≈ 0.215 m (= 0.56° apparent camera pitch at 22 m; matches L3 |b|=0.218 exactly).** NOT the map, NOT the body radius.

### Non-circular signature confirming case (b)
- Vertical bias is **DEPTH-FREE** (corr range_err vs off_D = +0.11).
- Vertical bias is **BEARING-CORRELATED** (corr off_D vs bearing = +0.44).
- A world-frame map offset would be bearing-INVARIANT; the bearing-dependence is a camera/perception-geometry signature.
- Likely mechanism: **sim-render-vs-decode MOUNT MISMATCH** — estimator decodes with `CAMERA_PITCH_RAD = 20.0°` exactly; currently uncalibrated, hiding inside the 1.4° attitude / 0.40 m cov floor.

### Silver lining: CALIBRATABLE
- ε_vert is **CALIBRATABLE** via boresight calibration / ESKF attitude-bias state → drives ε_vert → 0 → back to the bias-free regime where the margin CLOSES.
- Path to close = **calibrate boresight + solve 2-axis camera-pointing/fix-rate**.
- ESKF attitude-bias estimation PROMOTED from secondary to **co-equal margin lever** (alongside 2-axis camera-pointing/fix-rate).

### Residual caveat + airtight lock test
- "GT crosses near map centre" evidence is mildly SELF-FULFILLING on the vertical axis (inc7/calibration laps aim at map centre); depth-free + bearing + resurvey lines break it to case (b) but not airtightly.
- **AIRTIGHT LOCK (morning, ShadowPC):** a STATIC, head-on, LEVEL vision fix at g2 & g4 (bearing ≈ 0) isolates boresight ε from any map term. Expect δ_map_vert ≈ 0, small head-on ε. Alternatively: bearing-resolve the ShadowPC-local data/runs/*_l3atspeed_* via shadow_gate4.py. [🆕 SUPERSEDED 2026-06-14 → gate-0/1 CTBR head-on, ≥2 ranges (boresight common-mode → transfers to g4); Fengyou GO. See §P3 OFFLINE RANGE-RESOLVE.]

### 🚩 Two stale map-offset citations — CORRECTED
1. **"gate-3 crosses 1.46 m above (map) center"** = BOTTOM-vs-OPENING reference-frame confusion; true miss is **0.017 m**. NOT evidence of a map vertical offset. Do NOT cite as case-(a)/map-offset support.
2. **piece-F "+0.3 m registration_D"** = ε − δ_map MISLABELLED as registration; its own GT-transit resurvey σ=0.03 m contradicts it. NOT a real map offset. Do NOT cite as case-(a)/map-offset support.

## P3 OFFLINE RANGE-RESOLVE = INCONCLUSIVE; LIVE HEAD-ON FIX AUTHORIZED (2026-06-14; gate-0/1 CTBR, Fengyou GO; SUPERSEDES the g2/g4 lock-test framing)

- **P3 independent discriminator** (rel_vert vs true_range over L3 shadow_gate{2,4}_rows.json; map offset ⇒ constant metres slope≈0, a 0.56° boresight ⇒ constant angle slope≈−0.0098 m/m): all-accepted pooled (N=315, 4.5–25.9 m) slope **+0.001 m/m CI[−0.007,+0.009]** (flat/map-like, marginally excludes the boresight slope); clean 4-corner subset (N=172, 18.6–25.2 m) slope **−0.043** → non-physical 2.4° + 0.71 m intercept = a **23–25 m PnP/3-corner band artifact**, non-monotonic; **ZERO accepted fixes <30° bearing** at g2 AND g4 (inc7 66° crab → oblique-only) → no head-on data to resolve ε(bearing→0).
- **Does NOT refute the DIRECT δ_map≈0 (d7c592e):** over the ~20–26 m gate-4 band a 0.56° boresight moves rel_vert only ~0.06 m (below scatter) → the slope test is UNDERPOWERED there; the pooled 'flat' mildly tensions case-(b) toward map/CLOSE but is PnP/3-corner-confounded. The −0.215 m is PINNED; its δ_map(CLOSE)-vs-ε_vert(OPEN) split is NOT determinable offline. **Margin verdict pivots on it.**
- 🚩 **AUTHORIZED LIVE ARBITER (Fengyou GO, 2026-06-14):** STATIC, level, HEAD-ON (bearing≈0) vision fix at **gate-0/1 via the proven CTBR bridge, ≥2 ranges.** Reveals the bias FUNCTIONAL FORM (ε∝range ⇒ angular boresight → calibrate the angle; ε flat-in-metres ⇒ constant offset → calibrate a metric term) + magnitude + sign → pins P1's calibration. Boresight is **COMMON-MODE across gates** → a gate-0/1 measurement transfers to gate-4. **SUPERSEDES the prior "static head-on at g2/g4" framing** (the g2/g4 ANGLE position-hold is untested/higher-risk). Conservative case-(b)/OPEN posture HELD meanwhile — P1 builds boresight-cal + ESKF regardless. Artifacts: handoff/p3-simops-empirical-2026-06-14/ (boresight_bearing_resolve.py, boresight_resolve_all.json), committed for review.

## BORESIGHT = OFFLINE EXTRINSIC CORRECTION, NOT ONLINE ESKF (2026-06-14; COWORK-2; handoff/cowork-2026-06-14/boresight-calibration.md)

**DECISION (adopted): remove ε_vert OFFLINE as a static extrinsic constant; do NOT estimate it online.** ε_vert ≈ 0.56° ≡ 3.1 px @ fy=320 ≡ 0.215 m @ 22 m = a CONSTANT, known-sign, sub-degree pitch offset from a sim-render-vs-PnP-decode convention mismatch. Fix = align the PnP decoder's extrinsic to the renderer → **bake −ε into the camera↔body/IMU extrinsic** (one-time, zero runtime cost, gate-4 bias term → 0).
- **Do NOT add an online residual pitch attitude-bias state for that DOF.** At ~7% fix density it converges slowly, confounds with gyro/accel biases AND the gate offset, and DOUBLE-COUNTS the offline correction (gauge ambiguity — pick ONE home for the constant).
- 🚩 **DEMOTES the banked "ESKF attitude-bias estimation = CO-EQUAL margin lever" framing.** ESKF keeps the TIME-VARYING gyro/accel-bias states (different physics) + optional TIGHT-PRIOR camera-pitch watchdog (monitor-only). The boresight is an OFFLINE constant, not an online lever. Margin levers now = fix-rate/terminal-gate-lock (P2) + offline boresight measure→bake (P3→P1).
- **Precedent (convergent):** Agilicious (Kalibr offline), AlphaPilot (constant body↔camera; EKF only VIO-misalignment + gate map), Swift (offline-learned residual obs model); MonoRace (arXiv 2601.15222) offline-calibrates from onboard logs using known gate geometry. NONE estimate camera–IMU rotation online in the racing loop.
- **DISCRIMINATOR (= P3's range-slope, run HEAD-ON):** static known-gate at **≥3 ranges, camera gravity-leveled** → angular boresight = constant ~3.1 px / metric ∝ range; map offset = constant metric / pixels ∝ 1/R; single fixed range is ambiguous. Use cv::SOLVEPNP_IPPE_SQUARE (inspect both flips). Keep FEJ/OC-VINS consistency given sparse fixes.

## P1 BORESIGHT AUDIT + CALIB-V2 (2026-06-14; ✅ MERGED to main c740743 — apply 5091c88, default-zero byte-identical, 786 green; see §P1 PATHWAY COMPLETE)
- **AUDIT (closes the vision-extrinsics-audit open item #3):** boresight bake site = **frames.R_camera_from_body()** (the camera↔IMU extrinsic; body↔IMU identity per spec §3.8). **NO code bug** — round-trip 1.48e-8 m; **fy=320 / cy=180 consistent in BOTH render-prediction AND PnP decode.** The 0.56° is purely the render-vs-decode CONVENTION constant, as COWORK-2 predicted.
- **CALIB-V2 (mechanism):** unified **frames.BoresightCorrection{pitch_rad, roll_rad, vert_offset_m}**, default-zero **byte-identical** (np.array_equal on mount + +L lever). Two forms: **ANGULAR** (vbias ∝ range; angle-flat +0.560°; correction in R_camera_from_body) vs **METRIC** (vbias flat −0.215 m; angle ∝ 1/range; vert_offset in the +L lever). Identical at 22 m; each form's decode correction zeroes residual ≤1e-5 m. Dual-form patch supersedes the audit frames-only patch; **NOT applied yet** (infra-first apply-worker → default-zero, then ε is a one-line {form,value}). Pattern mirrors C2 / fix_surrogate (gated-off, cannot regress live path).
- **Calibration estimators validated vs the REAL chain:** #15 MonoRace IoU-BO recovers pitch+roll 0.000° noiseless / ≤0.008° @0.5 px; #16 level-hover WLS splits intercept(metric) vs slope(angular) → an ANGULAR/METRIC/MIXED classifier.
- 🚩 **ESKF RE-SCOPED:** DROP the online boresight corrector (own analysis: 1.0° / lap-1 insufficiency). KEEP accel-bias [**gyro-bias LIKELY DEAD — attitude is GIVEN (ATTITUDE on wire), not gyro-integrated**; verify on official-sim wire] + a tight-prior camera-pitch WATCHDOG. The watchdog reads frames.BoresightCorrection ONLY to set its prior (mean 0, width = residual) — must NOT re-apply the baked mean (double-count). Its rotational δB CANNOT absorb a METRIC (translation) offset → **the form choice is load-bearing for the ESKF too.**
- 🚩 **cv2 SOLVEPNP_IPPE_SQUARE NaN edge** on near-frontal + uniform-shift quads (opencv numerical edge, NOT a chain bug) — mitigate with frame-averaging + a geometry-nudge (P3 lock test) + navigator prior/coast (flight).
- 🚩 **FORM-DISCRIMINATION needs WIDE-SEPARATED ranges:** ≥2 ranges with Δrange ≥ ~10 m (12 m & ≥24 m); "≥3 ranges" alone is INSUFFICIENT (closely-spaced 20/22/24 = degenerate → magnitude only, NOT form). g2/g4 at ~4 m only if per-fix σ_vert ≤ ~0.005 m. → routed to P3.

## P3 BORESIGHT HEAD-ON: ε MAGNITUDE PINNED + CAMERA-POINTING VALIDATED; FORM PENDING (2026-06-14; gate-0 CTBR head-on; N=13,308 accepted; tooling boresight_form.py + b1/b2_g0_rows.json)
- 🚩 **ε MAGNITUDE PINNED = −0.25 m @ 23 m ≈ −0.61°.** Sign: rel_vert NEGATIVE = gate-DOWN (vision places the gate ~0.25 m BELOW truth; estimator reads the drone ~0.25 m HIGH relative to the gate). cross ≈ 0 (no azimuth/attitude error). ROBUST — 4 independent measurements agree at ~22 m: gate-0 head-on −0.248, L3 gate-4 −0.215, gate-2 −0.211, static-hover ≡ moving-approach. (Consistent with the prior 0.56°/0.215 m pin; the dedicated head-on gate-0 reads the higher end.)
- 🚩 **CAMERA-POINTING THESIS EMPIRICALLY VALIDATED:** head-on fix-rate **67–77% vs inc7's 1.4%** (~50× fix density from pointing the camera at the gate). The inc8 2-axis terminal-lock premise is confirmed on REAL data, not just the surrogate.
- **FORM (angular vs metric) NOT YET RESOLVED — gated on ANALYSIS, not data (no re-fly; N=13k):** range-signal is 17σ (Δ(far−near) = −0.114 m; 12 m −0.134 vs 23 m −0.248 ≈ range-ratio → ANGULAR-leaning). Confounds: (a) the discriminating ranges sit in the SOLVEPNP_IPPE_SQUARE planar-FLIP zone, range-dependent the WRONG way — the far 23 m anchor (gate subtends ~3.7°, weak perspective) is MORE flip-ambiguous, so the −0.248 anchor may be flip-tainted; (b) a 14–20 m overshoot to −0.27 (beyond either model); (c) N-imbalance (12k at the 23 m hover anchor). Provisional steer = leans ANGULAR/extrinsic-rotation (bias scales down at near range) but the flip could flip it. **DO NOT lock calibration TYPE until the IPPE both-flip + regime-separated (hover el+20° vs approach el−1°) + range-balanced pass resolves it (~30–45 min).**
- **P1 deliverable NOW:** ε magnitude + sign in BOTH representations — angular −0.61° in R_camera_from_body, metric −0.25 m @ 23 m in the +L lever. HOLD the TYPE; calib-v2 is already dual-form. 🚩 The IPPE planar-flip is now a CONFIRMED form-discrimination confound (range-dependent, worse at far/weak-perspective range) — refines the banked cv2 IPPE known-issue.

## P1 PATHWAY COMPLETE — BORESIGHT INFRA MERGED + ESKF RE-SCOPE DESIGNED (2026-06-14; P1 prunable; infra c740743, ESKF design 524d4b2)
- **BORESIGHT INFRA MERGED (c740743, apply 5091c88):** frames.BoresightCorrection{pitch_rad,roll_rad,vert_offset_m} default ALL-ZERO → composed into R_camera_from_body (mount np.array_equal byte-identical) + localization._apply_camera_vert_offset (guarded !=0) into BOTH +L lever sites. tests/test_calib_dualform.py (22; 3 byte-identity pins active+pass). 786 green, 0 regressions, +L preserved, 20°-mount / 0.38-radius / 20-dim-obs untouched. **ε (P3's measure) drops in ONE LINE: `frames.BORESIGHT = BoresightCorrection({form, value})`** — applied by the OVERALL COMMANDER once P3's FORM lands (P1 pruned).
- **RATIFIED DEVIATION (Option A):** localization reads `frames.BORESIGHT` LIVE (single-source-of-truth) not a snapshot import — fixed a latent footgun (the ε-rebind would not have reached the lever, and broke the artifact's own test). Byte-identical at default, production-identical.
- **ESKF RE-SCOPE DESIGNED (design preserved handoff/p1-vision-accuracy-2026-06-14/ESKF_RESCOPE_DESIGN.md; code NOT built — data-conditional):** boresight DROPPED as an online state (constant→offline bake; gauge-ambiguous). **GYRO-bias DEAD** (attitude is GIVEN, never gyro-integrated → zero coupling; 🚩 flagged for official-sim WIRE RE-VERIFY — our ShadowPC streams ODOMETRY/LPN the official spec may not). **2-AXIS ACCEL-bias body-Y,Z** (lateral+vertical = gate-plane margin axes) LIVE design (MC NEES 1.97/2 unbiased; under 0.6° budget in ~2 laps @fr0.07); body-X DROPPED (unobservable, velocity-aliased — 3-axis = overconfidence trap). Watchdog read-only, both forms (angular=slope shift, metric=intercept shift; warn→abort, never silently corrects). Contract-safe: obs_dim 20, +L preserved, gated OFF behind `use_imu_bias_eskf` → inc7/VQ1/case-A byte-identical; rides existing RewindKF predict-replay.
- **ESKF DECISIONS (P1):** residual/prior-width fields (pitch_residual_rad/vert_residual_m) ACCEPTED but DEFERRED to next struct-touch (ESKF hardcodes conservative 0.13°/0.05m interim). 🚩 **ESKF BUILD = DATA-CONDITIONAL:** build the accel-bias estimator ONLY if first-contact (POC / at-speed recording) shows a real accel-bias ABOVE budget; else it reduces cleanly to watchdog-only. **Boresight bake = PRIMARY bias lever (done); accel-bias = SECONDARY, data-gated.**

## MARGIN-CLOSURE-ENVELOPE DONE (2026-06-14; branch claude/charming-jemison-b111c5, commit 7654e99, pushed, NOT merged; handoff/margin-closure-envelope-2026-06-14/REPORT.md)

**268-cell sweep (fix-rate × attitude-bias × speed × latency), nmc=5000 bootstrap CIs, 0 errors, 4-lens adversarial pass.**

### Core verdict
**σ=0.10 relocates the blocker from σ → FIX-RATE. CANNOT-SETTLE-OFFLINE SURVIVES** (consistent with the δ_map refutation). The old σ=0.265 modeled value made closure impossible everywhere; σ=0.10 (measured) makes closure POSSIBLE — but only if fix-rate AND bias AND latency all clear their thresholds simultaneously.

- **At the MEASURED operating point (cold case-C, σ=0.10, bias≈0, fix-rate=0.07, 37 m/s):** r=0.30 is OPEN — p99 0.297 [CI 0.288, 0.304] vs MARGIN 0.235 (CI sits ENTIRELY over the wall — not noise).
- **At modeled σ=0.265:** closure envelope is EMPTY everywhere.
- **σ=0.10 fix-rate 0.07 fails at every bias and speed.** r=0.30 closure needs fix-rate ≥ 0.15–0.25 (CI-clean: fr=0.15 → p99 0.196 [0.190, 0.203]).

### Closure envelope (r=0.30)
Closes IFF **ALL** of:
1. **fix-rate ≥ ~0.25** (≥ 0.35 at v=55 m/s)
2. **attitude bias ≤ ~0.6°** (in-plane-conservative)
3. **GPU latency ≤ ~50 ms** (CPU-115 ms breaks the v=55 cells; RewindKF handles OOSM but CPU strands post-plane fixes)
4. **TERMINAL GATE-LOCK: camera held through ≥ 60% of the final approach** — pooled mean fix-rate is NOT sufficient

r=0.38 stress: fr ≥ ~0.5, bias ≤ ~0.3–0.6°. Speed-gate: at fr≥0.25, r=0.30 holds across the whole 25→55 m/s ladder.

### 4-lens adversarial pass (majority adjudication)
1. **LATENCY = real third axis.** CPU-115 ms breaks the v=55 cells; stranding post-plane fixes is correct physics, not a bug.
2. **CLUSTERING = STRONGEST LENS.** Terminal fix-DROUGHTS break closure at any mean rate. Global-burst models diverged ~8–12 m (upper bounds); the terminal-drought model is the defensible boundary. **Closure needs TERMINAL gate-lock, NOT just a high pooled mean.**
3. **IN-PLANE BIAS confirmed** (boundary tightens ~one step; operating verdict unchanged).
4. **FIX-RATE optimism:** the 0.07 bottleneck is CAMERA COVERAGE, not χ² (gate-4 band 94/95 accepted once associated); even fr=0.10 leaves r=0.30 OPEN.

### L3 ShadowPC target — 3 conditions needed for race-the-cap
1. Gate-4-band fix-rate ≥ ~0.25 WITH TERMINAL VISIBILITY (camera holds gate through final ~5 m) = load-bearing inc8 camera-pointing item.
2. Effective attitude bias ≤ ~0.6° (KF open-loop vs GT vel LOCAL_POSITION_NED).
3. Vision latency ≤ ~50 ms (GPU-class).

### Operational call
**Race-the-cap is gated on CAMERA POINTING (terminal gate-lock), NOT σ.** Do NOT spend on per-fix precision — spend on TERMINAL GATE-LOCK / fix density. A naive fix-difference velocity channel is HARMFUL at sparse fix-rate (corroborates the already-banked velocity-channel REFUTED → P2 insurance).

### 🚩 3-Way overnight convergence (2026-06-14)
All three overnight sessions (L3 shadow, δ_map discriminator, margin-closure-envelope) agree: CANNOT-SETTLE-OFFLINE SURVIVES, and the path to CLOSE is now fully quantified:
- σ NOT the blocker (0.10 < modeled 0.265); body radius NOT the blocker (central r=0.30).
- **Dominant lever = FIX-RATE via camera pointing: need ≥ ~0.25 with TERMINAL gate-lock; measured 0.07 pooled / 1.4% @ gate-4.**
- **Bias axis:** ceiling ≤ ~0.6°; δ_map pins ε_vert at 0.56° — nearly EXHAUSTS the budget alone → boresight calibration / ESKF attitude-bias estimation REQUIRED, not optional.
- **Latency axis:** ≤ ~50 ms GPU-class (CPU-115 ms breaks the high-speed cells).

## C2-ESTIMATOR-CHAIN DONE & MERGED (2026-06-14; merge commit 05ed750; feature 2d85d7e; 700→723 green, 0 regressions)
- **RewindKF (`kf_rewind.py`):** productionized OOSM. `update_position_at` rewinds to capture-time + replays buffered IMU exactly. horizon=0.5 s with `assert_horizon_gt(L)` guarding the horizon≤L "drop 100% of fixes" trap. t_fix≥now = bit-identical to bare update. Full live OOSM TIMESYNC-blocked → predict-forward fallback ships first.
- **`localization.gate_relative_inplane_fix`:** core +L fix. Anisotropic gate-plane cov: in-plane from law `max(σ_ref, a1·r)` (NOT additive — additive double-counts off validated c1 0.139; law & c1 agree ~10 m). Along-track loose + explicit floor. `GATE_REL_INPLANE_SIGMA = 0.265` = the SINGLE swappable constant. a1=0.026 extrapolates badly >24 m (in-band for the ~12 m gate-4 window).
- **Navigator relinnov gate:** d2_rel ≤ χ²(2,.999)=13.82, applied AFTER the kept absolute fix; a rejected relative fix leaves absolute estimate intact.
- **estimator_obs:** thin seam reusing canonical build_obs; bit-exact 17-dim, +L; pos_g/vel_g from estimate, attitude from wire; NO obs[17:20].
- **state_estimator/contracts:** NavState `nav_inplane_sigma` / `nav_along_sigma` cov export (built, UNCONSUMED by inc7).
- **Gating:** `use_rewind_kf` + `use_gate_relative`, BOTH OFF by default → VQ1/case-A BYTE-IDENTICAL (cannot regress inc7). Case-C only.
- **G3 result:** rel E_bias −0.000 / RMS 0.131 / p90 0.197; abs +0.176/0.283, submap +0.174/0.223 (neg controls FAIL). rel p90 0.197 > 0.155 @ r=0.38 → CANNOT-SETTLE-OFFLINE survives → escape-hatch L3, by design.
- 🚩 **d2 spec e_pred sign = TYPO** — C2 used STANDARD-innovation form `nu_ip = B@(z_rel − x_KF)`; +L pinned by G1 (identity ≤1e-5, −L control breaks ~24 m). Consistent with banked "+L is correct in code" footgun.
- 🚩 **OPEN follow-up (do NOT lose):** G3 ran at σ_ref=0.265 (design value), but Track-3 MEASURED σ=0.10. The 0.265 floor dominates the gate-4 band → offline verdict is PESSIMISTIC. Re-run G3 with `GATE_REL_INPLANE_SIGMA=0.10` for the OPERATIONAL margin verdict (plan = L3 either way; only the honesty of the offline number changes). G3 also has NO fix-RATE sweep → fold into the POC (camera-pointing is the binding lever).
- simops-mastery Track-3 tooling ON ORIGIN (commit 1b54cb8) → prior "ShadowPC-only, needs push" caveat CLOSED.

## P0-CASEC-FOUNDATION DONE (2026-06-13; merged to main; 692→700 green, 0 regressions)
- **P0-a:** `_initialize` now gates position/velocity seed on `config.use_given_position` / `use_given_velocity` flags (mirroring per-tick guards at :315/:320). True case-C seeds origin @ pos_std=5.0 (P[0,0]=25) even with LOCAL_POSITION_NED broadcasting. This closed the leak that made every case-C test secretly case-A.
- **P0-b:** `time_since_vision_update_s` measured on IMU master clock (delta_epoch learned once recv-paired, re-learned on reset()), plus predict-forward fallback (`reconcile_vision_clock=False`, `vision_latency_const_s`). delta_epoch=0 on same-clock data → back-compat exact. **OOSM capture-time rewind DEFERRED to C2** — implemented at navigator STAMPING level only; LinearKF has no `update_position_at`. tsv-observable effect identical.
- **P0-c:** case-C cold-velocity = KF pos/vel coupling; already exported via `NavState.velocity_ned` + `pos_vel_covariance[3:6,3:6]`. Confirmed + guard-tested; no behavior change.
- New test files: `tests/test_obs_sign_faithfulness.py` + one P0 integration test file.

## Topic file pointers
- [[project-phase2-rl-vision-decisions]] — vision/estimator/case-C sections: §VISION-PKG2, §ESTIMATOR-RACESPEED, §ORGANIZER-PIVOT, §CASE-C-READINESS, §SPEED-CEILING-ANALYTIC, §GATE-MAPPER, §PARALLEL-SYSTEMS, §ADVISOR-TRIAGE-2026-06-11.
- [[project-detector-training-pipeline]] — Adroit YOLO-pose training; v2/v3; weighted-PnP; adroit-connector ops.
- [[project-estimator-robustness]] — adaptive R, innovation-gate, map-avg, SEARCH state, optical-flow, RewindKF.
