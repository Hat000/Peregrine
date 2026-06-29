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
🚩 **RE-JUDGED 2026-06-19 (bar correction):** every "p90 fails 0.155 / CANNOT-SETTLE / σ_p0 ≲0.08" verdict in this section was judged against the DOUBLE-COUNTED budget (clearance−0.215; see the §Contact-radius reconciliation banner). Real clearance ≈0.37–0.47 m, real bar **σ_p0_lat ≲0.15** → the cold-margin "failures" re-judge to MARGINAL-PASSING. The NARROW CANNOT-SETTLE-OFFLINE CORE (at-speed σ needs a physical drone — sim is blur-free) STAYS true; the margin-NO-GO it implied does NOT. → §inc8-2026-06-19 in [[index-rl-training]].
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
🚩 **BUDGET IDENTITY CORRECTED 2026-06-19 (Fengyou caught a real error) — the `−0.215` SECOND subtraction in this whole block is the DOUBLE-COUNT.** `budget(r) = (0.75−r) − 0.215` subtracts the drone TWICE: `r` (the empirical crash halo) ALREADY includes the chassis, and `linf₀=0.215` IS the chassis half-diagonal (0.2135) — NOT an independent "crossing offset." The SIM's real pass test is `|offset| < 0.75 − r` → **real gate clearance ≈ 0.45 m @ r=0.30, ≈ 0.37 m @ r=0.38** (every budget row below UNDERSTATES the real clearance by 0.215 m; the 0.235→/3=0.08 σ_p0 bar derived from it is RETRACTED). The CONTACT-RADIUS reconciliation (0.30 central / 0.38 worst-case / don't-adopt-0.18) STANDS — only the budget=clearance−0.215 step and any 0.08-derived verdict are wrong. Real centering bar: **σ_p0_lat ≲0.15 (p99 ≲0.45)**; measured 0.15–0.20 = MARGINAL-PASSING. → §inc8-2026-06-19 in [[index-rl-training]] · [[project-rl-increment-history]].
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
- Vision stream: 30 Hz, 640×360 JPEG, UDP port 5600, 24-byte header. VQ2 wire: HEARTBEAT(10Hz) + HIGHRES_IMU(117Hz, **accel+gyro ONLY** — 🚩 NO MAG/NO BARO confirmed 2026-06-29; fields_updated=63) + ENCAPSULATED_DATA=RACE_STATUS(4Hz, `active_gate_index`) + ACTUATOR_OUTPUT_STATUS(95Hz) + TIMESYNC(response-only). ATTITUDE/LOCAL_POSITION_NED/ODOMETRY/GATE_INFO ALL BLOCKED. Body↔IMU = identity. Physics 120 Hz, command <100 Hz.
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
- ① ~~Photoreal detector (Blender/Cycles next)~~ — **RETRACTED 2026-06-29**: VQ2 live appearance = glowing-red/low-light, NOT photoreal → the GS/NeRF/Blender-Cycles photoreal-hardening direction is DEPRIORITIZED; re-scope to low-light/red-glow data after recon (see §VQ2 APPEARANCE below).
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

## P3 BORESIGHT HEAD-ON: ε MAGNITUDE PINNED + CAMERA-POINTING VALIDATED; FORM RESOLVED = METRIC [see §BORESIGHT FORM RESOLVED] (2026-06-14; gate-0 CTBR head-on; N=13,308 accepted; tooling boresight_form.py + b1/b2_g0_rows.json)
- 🚩 **ε MAGNITUDE PINNED = −0.25 m @ 23 m ≈ −0.61°.** Sign: rel_vert NEGATIVE = gate-DOWN (vision places the gate ~0.25 m BELOW truth; estimator reads the drone ~0.25 m HIGH relative to the gate). cross ≈ 0 (no azimuth/attitude error). ROBUST — 4 independent measurements agree at ~22 m: gate-0 head-on −0.248, L3 gate-4 −0.215, gate-2 −0.211, static-hover ≡ moving-approach. (Consistent with the prior 0.56°/0.215 m pin; the dedicated head-on gate-0 reads the higher end.)
- 🚩 **CAMERA-POINTING THESIS EMPIRICALLY VALIDATED:** head-on fix-rate **67–77% vs inc7's 1.4%** (~50× fix density from pointing the camera at the gate). The inc8 2-axis terminal-lock premise is confirmed on REAL data, not just the surrogate.
- **FORM (angular vs metric) NOT YET RESOLVED — gated on ANALYSIS, not data (no re-fly; N=13k):** range-signal is 17σ (Δ(far−near) = −0.114 m; 12 m −0.134 vs 23 m −0.248 ≈ range-ratio → ANGULAR-leaning). Confounds: (a) the discriminating ranges sit in the SOLVEPNP_IPPE_SQUARE planar-FLIP zone, range-dependent the WRONG way — the far 23 m anchor (gate subtends ~3.7°, weak perspective) is MORE flip-ambiguous, so the −0.248 anchor may be flip-tainted; (b) a 14–20 m overshoot to −0.27 (beyond either model); (c) N-imbalance (12k at the 23 m hover anchor). Provisional steer = leans ANGULAR/extrinsic-rotation (bias scales down at near range) but the flip could flip it. **DO NOT lock calibration TYPE until the IPPE both-flip + regime-separated (hover el+20° vs approach el−1°) + range-balanced pass resolves it (~30–45 min).**
- **P1 deliverable NOW:** ε magnitude + sign in BOTH representations — angular −0.61° in R_camera_from_body, metric −0.25 m @ 23 m in the +L lever. HOLD the TYPE; calib-v2 is already dual-form. 🚩 The IPPE planar-flip is now a CONFIRMED form-discrimination confound (range-dependent, worse at far/weak-perspective range) — refines the banked cv2 IPPE known-issue.

## P1 PATHWAY COMPLETE — BORESIGHT INFRA MERGED + ESKF RE-SCOPE DESIGNED (2026-06-14; P1 prunable; infra c740743, ESKF design 524d4b2)
- **BORESIGHT INFRA MERGED (c740743, apply 5091c88):** frames.BoresightCorrection{pitch_rad,roll_rad,vert_offset_m} default ALL-ZERO → composed into R_camera_from_body (mount np.array_equal byte-identical) + localization._apply_camera_vert_offset (guarded !=0) into BOTH +L lever sites. tests/test_calib_dualform.py (22; 3 byte-identity pins active+pass). 786 green, 0 regressions, +L preserved, 20°-mount / 0.38-radius / 20-dim-obs untouched. **ε (P3's measure) drops in ONE LINE: `frames.BORESIGHT = BoresightCorrection({form, value})`** — applied by the OVERALL COMMANDER once P3's FORM lands (P1 pruned).
- **RATIFIED DEVIATION (Option A):** localization reads `frames.BORESIGHT` LIVE (single-source-of-truth) not a snapshot import — fixed a latent footgun (the ε-rebind would not have reached the lever, and broke the artifact's own test). Byte-identical at default, production-identical.
- **ESKF RE-SCOPE DESIGNED (design preserved handoff/p1-vision-accuracy-2026-06-14/ESKF_RESCOPE_DESIGN.md; code NOT built — data-conditional):** boresight DROPPED as an online state (constant→offline bake; gauge-ambiguous). **GYRO-bias DEAD** (attitude is GIVEN, never gyro-integrated → zero coupling; 🚩 flagged for official-sim WIRE RE-VERIFY — our ShadowPC streams ODOMETRY/LPN the official spec may not). **2-AXIS ACCEL-bias body-Y,Z** (lateral+vertical = gate-plane margin axes) LIVE design (MC NEES 1.97/2 unbiased; under 0.6° budget in ~2 laps @fr0.07); body-X DROPPED (unobservable, velocity-aliased — 3-axis = overconfidence trap). Watchdog read-only, both forms (angular=slope shift, metric=intercept shift; warn→abort, never silently corrects). Contract-safe: obs_dim 20, +L preserved, gated OFF behind `use_imu_bias_eskf` → inc7/VQ1/case-A byte-identical; rides existing RewindKF predict-replay.
- **ESKF DECISIONS (P1):** residual/prior-width fields (pitch_residual_rad/vert_residual_m) ACCEPTED but DEFERRED to next struct-touch (ESKF hardcodes conservative 0.13°/0.05m interim). 🚩 **ESKF BUILD = DATA-CONDITIONAL:** build the accel-bias estimator ONLY if first-contact (POC / at-speed recording) shows a real accel-bias ABOVE budget; else it reduces cleanly to watchdog-only. **Boresight bake = PRIMARY bias lever (done); accel-bias = SECONDARY, data-gated.**

## BORESIGHT FORM RESOLVED = METRIC; σ-RECAL INHERITED; CLOSE/NO-CLOSE RE-RUN ✅ DONE 2026-06-15 (2026-06-14 P3 closeout; FORM_RESOLUTION.md on main d99ddf6; re-run = boresight-closure merge 5764291)
- 🚩 **FORM = METRIC (range-independent); angular REFUTED.** The IPPE both-flip overturned the earlier angular-lean. ε_vert = −0.25 m (refined/production)/−0.27 m (raw) @ ref ~22 m; **FLAT at −0.27 m across the 14–26 m band** (slight decrease −0.275→−0.270→−0.266); an angular boresight pinned @23 m would GROW ×1.9 → −0.51 m @26 m — refuted. L3 cross-gate (g2 −0.211@20m ≈ g4 −0.215@22m) corroborates. Sign gate-DOWN (chain places gate ~0.25 m BELOW truth); cross≈0.
- **IPPE both-flip IMMATERIAL:** rel_vert spread <0.024 m at EVERY range incl. the ambiguous 23 m anchor (e2/e1≈1.0); rv_loreproj ≈ rv_GTcorrect → NOT flip-tainted, deploy-robust WITHOUT a GT prior. Regime hover(el+20°) ≡ approach(el−1°) within 0.007 m → rules out cy/distortion. 10–14 m bin drops to −0.09 (low-N, fast near-field; outside the binding band, 2nd-order).
- 🚩 **CALIBRATION = `BoresightCorrection(vert_offset_m = −0.25)` into the +L LEVER** (constant; NO range schedule over 14–26 m; NOT a pitch/extrinsic rotation — angular refuted). P1's dual-form mechanism (frames.BoresightCorrection, main c740743) handles it; apply is ONE LINE `frames.BORESIGHT = ...`.
- 🚩 **RECONCILIATION (P3 caveat b — RESOLVED): metric-ε is CONSISTENT with δ_map≈0.** A constant-metre offset COULD be δ_map (cancels in +L) OR ε (doesn't); the δ_map discriminator (d7c592e) DIRECTLY measured δ_map≈0 (map good) → the −0.25 m is **ε (perception), NOT map → does NOT cancel → the bake is LOAD-BEARING + correct, not redundant.** After bake, ε→0, the bias term is removed.
- 🚩 **STRATEGIC: CANNOT-SETTLE may FLIP to CLOSE.** With boresight RESOLVED+calibratable (this) + camera-pointing VALIDATED (67–77% head-on) + latency CLOSED (desktop GPU, COWORK-1), the three CANNOT-SETTLE factors are now measured/proven. The OPEN question collapses to: does gate-4 cold margin clear once the −0.25 bake is applied AND the real σ is used? → the σ-recal + margin RE-RUN.
- **σ-RECAL INHERITED (analysis, NOT a recording):** L3 at-speed gate-4 is DONE (13 inc7 laps ~17–21 m/s, blissful-kalam ee62f00; gate-4 σ_vert 0.10 / σ_lat 0.19, N=81). σ-recal = reconcile surrogate pooled σ_vert 0.28 vs L3 gate-4 0.10 (axis-convention vs per-gate/speed). 🚩 If real gate-4 σ_vert is 0.10 (not 0.28), gate-4 is LESS vertical-σ-dominated → margin BETTER. CAVEAT: L3 is inc7-speed (~20 m/s); a ~30 m/s gate-4 recording is TODO (gated on the held inc8 envelope-relaxed policy). [parked #4 → active]
- **NEXT (overall-commander pathway, AUTHORED): boresight APPLY (−0.25 into +L) + validate (gate-4 bias → ~0) + σ-recal + gate-4 cold-margin RE-RUN (corrected ε + real σ) → the close/no-close VERDICT.** Data: P3 b1/b2_g0_rows (hardcore-lehmann b064b3c), L3 shadow_gate4_rows (blissful-kalam ee62f00), surrogate + frames.BoresightCorrection (main). This is the highest-leverage offline analysis remaining.
- ✅ **RE-RUN DONE (2026-06-15; boresight-closure MERGED 5764291, full suite 856 green; handoff/boresight-closure-2026-06-14/REPORT.md). CANNOT-SETTLE NOT flipped — TWO axes CLOSED:** (1) **BIAS:** `frames.BORESIGHT = BoresightCorrection(vert_offset_m=−0.25)` DEPLOYED into the +L lever (−0.08..0.15 m off gate-4 p99; byte-identical VQ1 path; gated). (2) **σ-FLOOR:** the real gate-4 σ is ANISOTROPIC [lat 0.19, vert 0.10] — the 0.28/0.265 pooled value was a POOLING artifact, NOT the floor. **Gate-4 STILL NO_CLOSE @ fr0.07** → blocker RELOCATED to (a) **TERMINAL GATE-LOCK + fix-rate ≥0.50 = the inc8 job** (🚩 BOTH threads converge: this margin verdict AND the inc8 conversion-gap fix) + (b) **a 30 m/s gate-4 AT-SPEED σ recording = the ONE genuine offline-unsettleable risk** (blur ×1.67–2.06 from 18→30 m/s, past the σ_lat ceiling ~0.24). r=0.30 closes IFF residual-bias ≤0.6° (ESKF, data-conditional); **r=0.38 NEVER.** 🚩 **Boresight axis CLOSED — and a RED HERRING for the inc8 TRAINING conversion gap** (boresight is real for DEPLOY accuracy only; the inc8 gap is reward-window-vs-p_accept-band, p_accept=f(range,in_image) — see [[index-rl-training]]).
  - **PRECISE r=0.30 closure spec (post-bake, anisotropic σ [lat 0.191, vert 0.101], COLD, bootstrap-90%-CI-honest; supersedes the §margin-closure-envelope thresholds):** (a) **terminal gate-lock** = accepted gate-4 fixes through the **last ~6 m / 0.15 s** of approach (NOT a high pooled mean); a 0.15 s terminal fix-DROUGHT breaks closure at any mean rate. (b) **effective fix-rate ≥ 0.50 IN THAT TERMINAL WINDOW** — 🚩 **SUPERSEDES the envelope's fr≥0.25/≥0.35**, which used the over-optimistic iso/pooled σ=0.10; the honest anisotropic σ_lat=0.191 raises the floor to 0.50 (naive iso-0.10 over-claims closure by 2.3–3.3×; lateral axis dominates). (c) **at-speed σ_lat ≤ ~0.23–0.245 m** (measured 0.191 @18 m/s = only ×1.25 headroom; motion-blur ×1.67–2.06 to 30–37 m/s → 0.32–0.39 = FAR past the ceiling = the at-speed σ-recording risk). (d) **bias ≤ ~0.6°** — even at fr=0.50 a 0.6° accel-bias breaks r=0.30 → the ESKF 2-axis accel-bias lever stays relevant (data-conditional).
  - 🚩 **ACCEPT-GEOMETRY TENSION (open, affects inc8 reward-window design):** closure wants accepted fixes through the **last ~6 m**, but the fix-surrogate's p_accept ≡ 0 below ~14 m (dead zone, calibrated on UN-POINTED inc7 Track-3). P3 head-on showed 67–77 % fix-rate at close range when POINTED → the dead zone is likely a pointing artifact, not a sensor floor (a pointed camera gets fixes closer in). RESOLVE before trusting the inc8 lock-window range: does pointing extend the fixable band inward toward the last 6 m, or is the surrogate dead-zone real? Determines whether the inc8 lock window should reach the terminal 6 m or stop at ~14–18 m. → [[index-rl-training]]
  - ✅ **RESOLVED 2026-06-15 → see §TERMINAL-LOCK REFRAMED below.**

## TERMINAL-LOCK REFRAMED + AT-SPEED σ — TWO CONVERGING WORKERS (2026-06-15; main @ 0484b74; handoff/at-speed-sigma-2026-06-15/ + handoff/accept-geometry-2026-06-15/REPORT.md)
🚩 **RE-JUDGED 2026-06-19 (bar correction):** the σ_p0 ≲0.08 terminal-lock target throughout this section is WRONG (a double-count); the real bar is **σ_p0_lat ≲0.15 (p99 ≲0.45)**, and measured 0.15–0.20 = MARGINAL-PASSING. Gate-4 is a REACH/PASS-RATE problem (the gate sits OFF the racing line), NOT sub-cm centering; the "fix-rate ≥0.50 through the last 6 m = GEOMETRICALLY UNACHIEVABLE" wall is MOOT (the policy nearly passes WITHOUT near-field fixes). Near-field-estimator pivot RETRACTED — RL stays the tool. The 12 m PnP floor + accept-geometry mechanisms below STAND; only the σ_p0 bar + the NO-GO verdict are corrected. → §inc8-2026-06-19.
🚩 **The gate-4 terminal-gate-lock blocker is REFRAMED: camera-pointing extends ACCURATE fixes inward only to ~12 m, NOT the terminal 6 m → "fix-rate ≥0.50 through the last 6 m / 0.15 s" is GEOMETRICALLY UNACHIEVABLE. Terminal-lock = deepest accurate fix ~12 m + RewindKF COAST the final ~12 m.**

### Worker A — at-speed σ + terminal-window recording (branch worker/at-speed-sigma-2026-06-15 @ 2f53ae1; merged 0484b74)
- 🚩 **SIM IS BLUR-FREE** (pinhole/SceneCapture): at 14.5 m/s edges razor-sharp ≡ hover (gate-box Laplacian ratio 0.68–1.55, no speed trend). → in-sim σ is RANGE-determined, NOT speed-determined; the linear motion-blur model (σ_lat ×1.67–2.06 → 0.32–0.39 @30 m/s) is **FALSIFIED in-sim.**
- **σ_lat vs speed (range-controlled) FLAT 0.13–0.17 m over 6→15 m/s, bias≈0 → projects flat to 30 m/s ≈ 0.15 m → CLEARS the 0.245 ceiling ×1.6.** The at-speed-σ/blur risk is **SIM-TO-REAL only → a PHYSICAL-DRONE item (Sept/Nov), unsettleable offline** (the anticipated branch — NOT a sim/VQ blocker).
- Terminal last-6 m fix-rate ≈5% (fixes die ~8 m, head-on CTBR). PnP depth bias positive, range-dependent (+2.1@26 m→+0.25@14 m), speed-independent. Max clean head-on CTBR = 14.5 m/s (faithful CTBR drag-limited at the 45° tilt cap; inc7 reaches 21.4 m/s but oblique-only; a true 30 m/s race-posture recording still needs the inc8 envelope-relaxed policy). Canaries PASS (verify_bundle p50 2.51 ms; frame_residual R_y(π) mirror +0.97/−0.81 OK → measurements real). Head-on CTBR (crab ~5°) = a clean σ-vs-range rig for future calibration. Raw video/rows gitignored (data/runs/*_atspd_*).

### Worker B — accept-geometry (pooled head-on POINTED gate-0, N=18,166 frames / 13,308 accepted, MOVING ≤7 m/s; merged @ cb46d14; handoff/accept-geometry-2026-06-15/REPORT.md)
- **Accept|in-FoV holds a ~0.78 plateau from ~24 m down to ~10.5 m** (surrogate band-pass rlo=16.2 predicts ~0 at 12–14 m → emp/surrogate **22–160×**) → the surrogate's <16 m roll-off is **MOSTLY a POINTING ARTIFACT** of the un-pointed inc7 fit (pointing takes 10–18 m accept ~0.18→~0.78, ≈4×).
- 🚩 **But the ACCURATE-fix floor = ~12 m:** lateral MAD flat 0.05–0.075 down to 12 m, then **0.31 @10–12 m / 0.22 @8–10 m** with +0.21→+0.55 m bias; crosses the 0.245 ceiling at ~11 m. Driver = near-field weak-perspective PnP + multi-detection (n_det 3–4 as the gate fills the frame → mis-association → biased-long depth). NOT flips (vertical MAD tight at all ranges, confirms FORM_RESOLUTION §2) and NOT corner-clip (n_corners=4 down to 6 m). 6–8 m = dead zone (association collapse).
- **χ² NON-BINDING here** (accept|offered=100%) → the funnel narrows at detection/association, not the gate. 🚩 **NEW PARKED:** confirm the DEPLOYED relinnov χ²(2,.999)=13.82 actually rejects the inflated 10–12 m fixes (MAD 0.31) — else a bad ~11 m fix could seed the terminal coast. Revive-trigger: any inc8 terminal-gate-lock / near-field-fix work.

### The convergent verdict (both workers agree)
- 🚩 **inc8 LOCK-WINDOW = harvest accurate fixes through ~12–18 m, then RewindKF COAST the final ~12 m (≈0.4 s @30 m/s).** 🚩 **CORRECTION (terminal_weight mechanism, 2026-06-15): KEEP d_lock=18 / d_acq=28 (the re-pilot's config).** terminal_weight SATURATES w_term=1.0 for range ≤ d_lock, so d_lock=18 rewards the FULL 12–18 m accurate band at max; **d_lock=12 would UNDER-reward it** (12–18 m becomes a ramp 0.68→1.0) AND saturate the <12 m dead zone. The earlier "d_lock≈12" was a mechanism-misread (d_lock is the saturation edge, NOT a band inner-edge — Worker B gave the conceptual target, not the mechanism). Actively DIS-rewarding the <12 m dead zone needs a LOWER-EDGE taper to terminal_weight (a code change), NOT a d_lock cut; the surrogate accept_rlo→12 fix + the coast reframe already keep the policy from chasing sub-12 m fixes. 🚩 **SUPERSEDED 2026-06-15 by convergence run 3273701: d_lock=18 ALSO FAILS — terminal_weight is a LOW-PASS, so d_lock=18 rewards the easy <5 m terminal zone equally, and a CONVERGED policy exploits it (lockband=0, fix_rate≈0). The lower-edge taper I called "optional" is LOAD-BEARING. FIX = a BAND-PASS terminal_weight (peak ~12–28 m, suppress below the accept floor), NOT any d_lock value. → [[index-rl-training]] §CONVERGENCE RUN.**
- 🚩 **SHARPENS the closure blocker — gate-4 r=0.30 @30 m/s now HINGES ON COAST QUALITY, not on closer fixes.** A 12 m accurate floor @30 m/s ⇒ a 0.4 s terminal coast, which EXCEEDS the closure model's 0.30 s drought tolerance (Lens A broke all closing cells at 0.30 s) → **pointing ALONE cannot satisfy terminal-lock.** ⚠️ CAVEAT (hopeful): the drought model is WORST-CASE (no info); an INFORMED coast with a good velocity estimate from dense 12–18 m fixes dead-reckons with BOUNDED drift, not a drought → the **RewindKF terminal-coast-drift analysis (offline, NEXT) settles gate-4 feasibility.** Levers if it fails: lower the coast DRIFT (sharper velocity estimate from the dense 12–18 m fixes + low IMU bias — the "informed coast" itself), or push the floor inward via a near-field estimator (hard — weak-perspective geometry, not corner count). 🚩 NOTE: speed is a TRADEOFF, NOT a clean lever — SLOWER lengthens the coast TIME (12 m / v) = MORE drift; FASTER shortens the coast but worsens the velocity estimate (fewer 12–18 m fixes) + at-speed blur. The coast-drift analysis weighs it; do NOT assume "speed-down" helps the terminal coast.
- **SURROGATE FIDELITY FIX (for inc8 training): `accept_rlo` 16.19 → 12.0** (one constant, keep wlo≈1.0; conservative = the accurate floor). Do NOT go below 12 without a companion close-range σ_lat inflation (flat σ would over-credit 10–12 m fixes; full fix = rlo≈10 + σ_lat ramp 0.10→~0.3 below 12 m). NOT applied yet.
- **CAVEAT:** the accept-geometry data is ≤7 m/s; the 12 m floor is GEOMETRIC (transfers) but at-speed only RAISES it → 12 m is OPTIMISTIC; the 30 m/s gate-4 at-speed recording (physical drone) pins it and is now MORE load-bearing, not retired.

### COAST-DRIFT RESOLVED — coast NOT the blocker; binding = terminal centering σ_p0 (2026-06-15; merged @ 1b34974; handoff/coast-drift-2026-06-15/REPORT.md)
- 🚩 **The informed terminal coast is NOT the gate-4 blocker.** Production LinearKF+RewindKF covariance propagated PREDICT-ONLY over the 0.4 s coast (real gate-4 posture/Q, 90 Hz) adds **<0.02 m**: process noise 0.009 / accel-bias 0.6° 0.008 / velocity-drift 0.008 (σ_v0≈0.02 m/s). 54-cell robustness grid (v∈{22,30,37}×floor×bias×σ) p99 ∈ [0.068, 0.114] — ALL clear r=0.30 AND r=0.38. **The boresight 0.30 s "drought" was the WRONG analogy** (a drought is zero-info; the informed coast dead-reckons on a velocity the dense pointed [12–24 m] stream pins tight).
- 🚩 **σ_v HYPOTHESIS REFUTED-as-binding: σ_v0≈0.02 m/s (IMU owns short-term velocity, fix-rate-robust). BINDING = terminal CENTERING σ_p0** — the last accurate fix's gate-relative lateral accuracy @~12 m, coasted RIGIDLY. **miss p99 ≈ 3.0·σ_p0.** (Caveat: fed iid fixes the production KF OVER-CONVERGES to σ_p≈1 mm and rides the IMU → MC centering-blind → A1 with the MEASURED per-fix σ is the centering tool. → NEW PARKED #74.)

🚩 **BAR CORRECTED 2026-06-19 (Fengyou caught a real error) — the σ_p0 ≲0.08 "closure bar" in the 3 bullets below is WRONG (a `margin_envelope.py` DOUBLE-COUNT).** B=0.235 subtracts the drone TWICE (W_EFF=0.75−0.215 chassis, then MARGIN=W_EFF−r → the drone again; 0.08=0.235/3). The SIM is correct: pass = offset < 0.75−r, r∈[0.28,0.38] → real clearance **0.37–0.47 m**; the REAL bar is **σ_p0_lat ≲0.15 (p99 ≲0.45)**. Measured σ_p0 0.15–0.20 / p99 0.37–0.40 = **MARGINAL-PASSING (sim ~0.65), NOT the NO-GO this block implies.** ⇒ gate-4 is a REACH/PASS-RATE problem (the gate sits OFF the racing line; clearance is the SAME as every gate), not sub-cm centering. 🚩 **The "near-field GATE estimator" framed below as "the centering lever" / "INVEST ②" is DEMOTED to SUPPORT/insurance — RL (lift reach/pass-rate) is the load-bearing lever; the policy nearly PASSES without a vision-side near-field fix. Do NOT lean on vision. The "pivot off RL" recommendation is RETRACTED.** The closure-boundary MATH below stands (it correctly maps B→σ_p0); only the B value (0.235→ the real ~0.45 gate-relative clearance) and the verdict are corrected. → [[index-rl-training]] §inc8-2026-06-19 · [[project-rl-increment-history]] §inc8-2026-06-19.
- **CLOSURE BOUNDARY (gate-difficulty sweep):** r=0.30 (B=0.235) CLOSES IFF σ_p0_lat ≲ **0.08 m**; r=0.38 (B=0.155) ≲ 0.05; tight-VQ2 (B=0.12) ≲ 0.04. Achievable σ_p0 ≈ σ_lat/√N_eff ⊕ σ_b: **0.05–0.07 m (pointed/accept-geom σ_lat 0.075–0.11 → CLOSES r=0.30 w/ margin) to 0.08–0.13 m (raw at-speed σ_lat 0.13–0.17 → KNIFE-EDGE/FAILS).** → **CONDITIONAL-CLOSE gated on inc8 per-fix σ + band fix-DENSITY @12 m, NOT the coast.** Sharpens boresight's "σ_lat≤0.245 + terminal gate-lock" to ONE number.
- 🚩 **VIO #70 back-solve (nothing built): trigger σ_v0 ≳ 0.18 m/s (g4) / 0.11 (r0.38) / 0.08 (B=0.12); IMU ~0.02 (×9 margin) → STAYS PARKED.** VIO shrinks σ_v0 (already fine) but is BLIND to gate-relative centering → WRONG lever. The centering lever = a **near-field GATE estimator** (lower the 12 m floor — a gate ANCHOR, not odometry) + **per-fix accuracy.**
- **INVEST: ① inc8 per-fix accuracy + band fix-density (the σ_p0 knob — highest value = the band-pass reshape + boresight bake); ② near-field gate estimator. NOT ESKF-for-coast (0.008 m), NOT VIO, NOT speed.** Best gate-4 speed = 30 m/s (coast flat 22–37, marginally favours faster; faster only re-opens the sim-to-real blur-σ risk).

## MARGIN-CLOSURE-ENVELOPE DONE (2026-06-14; branch claude/charming-jemison-b111c5, commit 7654e99, pushed, NOT merged; handoff/margin-closure-envelope-2026-06-14/REPORT.md)
🚩 **RE-JUDGED 2026-06-19 (bar correction):** this envelope's closure boundary is built on `margin_envelope.py`, which DOUBLE-COUNTS the drone (see the SUPERSEDED guard prepended to that file). The 0.08/0.155 budgets understate the real clearance (`0.75−r` ≈0.37–0.47 m); the σ_p0 ≲0.08 boundary is RETRACTED (real ≲0.15). The 4-D sweep METHOD + the clustering/terminal-drought lens insight STAND; the budget VALUES + the NO-GO verdict do not. → §inc8-2026-06-19.

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

## §FRUSTUM-REACHABILITY (Spike A, 2026-06-27) — gate-4 reframe, FORK STILL OPEN
Kinematic frustum overlay (no training/sim; `docs/reactivation-2026-06-27/spike-A-frustum-overlay.md`).
- **Convention CONFIRMED:** camera mounted **+20° UP** about body-Y (`src/racer/frames.py:7,20`; `R_camera_from_body = _R_CAMERA_FROM_TILTED_BODY @ R_Y(-20°)`). Optical axis 20° ABOVE body-forward; visible body-elevation band ≈ (−9.36°, +49.36°).
- **EXIT THRESHOLD (range-invariant 12–28 m):** a same-altitude gate falls out the **BOTTOM** of frame once body pitch > **9.4°** nose-down (= half-VFoV 29.36° − mount 20°). At racing pitch 40–45° the gate sits at camera-elev −60° to −65° — far outside the ±29.36° VFoV. So **nose-down racing pitch + up-mount = gate leaves frame**: reachability-as-visibility is REAL.
- **GATE-4 IS NOT GEOMETRICALLY UNIQUE:** identical frustum to gates 0–5. The difference is the **altitude PROFILE** — gates 1–3 are approached flying 10–25 m BELOW the gate (10–30° of pitch headroom → effective exit threshold 20–36°); gate-4 sits after the track flattens at gate-3 (Δ +0.79 m over 24.4 m ≈ 1.85° slope) → ~zero buffer → the bare 9.4° threshold bites.
- **VERDICT:** SUPPORTS the reachability hypothesis as a general fact; **REFUTES the "gate-4 uniquely excluded by geometry" framing** — the driver is approach altitude, not a gate-4-specific frustum.
- 🚩 **FORK STAYS OPEN** (per doors-open rule — a model prediction + an analytic overlay don't close it). The actual inc8 RL **pitch trace at 12–28 m from gate-4 is UNKNOWN** locally (no eval trajectory in worktree). CHEAP CLOSER: pull/run an inc8 eval on ShadowPC/Adroit, read the pitch profile near gate-4.
- **STRATEGY KICK:** strengthens two existing forks — (1) **horizontal-FoV approach shaping** (90° HFoV ≫ 58.7° VFoV → yaw-to-gate where geometry allows, dodging the vertical exit); (2) **approach-from-below / climb-angle reward** to manufacture the pitch headroom gate-4 lacks. Both are POLICY levers (mount is spec-fixed). → [[index-rl-training]]

## SE_2(3) RIGHT-INVARIANT EKF + SOT(3) LANDMARK PROTOTYPE (committed e4447e9, 2026-06-28)
`src/racer/ahrs/eqvio.py` + `traj6dof.py` + `eqvio_landmark.py`; tests 33/33; full AHRS suite 60/60.

**CONSISTENCY THESIS CONFIRMED at propagation level, deterministic:** noiseless RIEKF NEES CONSTANT (rel-spread ~1e-13, covariance transport group-affine EXACT) vs standard-EKF NEES DRIFTING 15→151. **This explains the earlier SO(3)-only null** (IEKF tied ESKF on attitude-only) — the invariant consistency edge needs the COUPLED att+vel+pos problem.

**Closed-loop (landmark fixes):** RIEKF in-band avg-NEES 9.01 on HIGH_G_LOOP vs std-EKF under-confident ~4.8. 🚩 **HONEST CAVEAT:** RIEKF NEES HIGH (~26) on very-excursive AGGRESSIVE_S after 8° init error — edge is CALIBRATION quality (not point-accuracy); reported, not tuned away.

**SOT(3) = SO(3) × R⁺ inverse-depth gate-corner bearing DONE+tested** (group / bearing / analytic Jacobians / triangulation FD-verified; bearing scale-invariant → depth from parallax). NEXT: joint pose+landmark covariance + Schur-marginalization = full EqVIO. → [[project_vq2_stack_research]]

### FULL EqVIO COMPLETE (committed 26e25ea, 2026-06-28)
`EqVIOJointEKF` (`src/racer/ahrs/eqvio.py`) joins SE_2(3) RIEKF pose + per-corner SOT(3) inverse-depth in ONE joint covariance [pose(9) | lmk(4)×K] — coupled bearing update (`update_landmark_joint`) + Schur-complement marginalization (`drop_landmark`, PSD-preserving, == info-form to 5.7e-15). Anchored joint Jacobians in `eqvio_landmark.py` (pose 3×9, landmark 3×4 with NON-ZERO parallax scale column → depth observable), FD-verified ~1e-10.

**VALIDATION:** depth recovers via parallax (world err <0.15 m noiseless); **JOINT BEATS SEPARATE** (decoupled dead-reckon+triangulate) 1.53 vs 2.24 m = 1.47×, wins 90% trials → cross-covariance is load-bearing.

**CONSISTENCY REGIME-DEPENDENT (honest):** forward-parallax pose NEES ~2.1–2.6 (consistent-to-conservative, never overconfident) — forward-parallax IS the racing regime (flying toward a gate); near-constant-range ORBIT NEES HOT ~100–900 (frozen-anchor-pose-error not modeled; pinned as known limitation; next-tier fix = FEJ/null-space).

Tests +18 (`test_eqvio_joint.py`); 78 AHRS+eqvio green. **STILL UNWIRED to the nav loop** (offline/synthetic only). → [[project_vq2_stack_research]]

## BEARING-RANGE CHANNEL B1 (committed af88b2c, 2026-06-28; perception-l2)
`apparent_range_from_gate_span` + `gate_range_sigma` + `gate_range_fix` in `localization.py`; wired via `NavigatorConfig.use_range_channel=False` (default OFF, byte-identical to VQ1) + `_apply_range_fix` (1-DOF chi2=10.83 along-track innovation gate, requires `use_gate_relative`).

**Motivation:** range from the gate inner-square apparent span is ATTITUDE-INDEPENDENT → robust to ε_vert boresight bias; replaces the loose `GATE_REL_ALONG_SIGMA=0.5` prior on the weak along-track axis.

**FIX vs scope-doc:** span half-angle = per-axis RMS half-WIDTH, not radial (scope-doc sketch was √2 too large).

**NUMBERS:** 0.000% range err head-on 5–30 m; tilt drift 0.17 m @ 15°/10 m (vs ~1.7 m for the +L lever); under +0.6 m depth bias: along-track err 0.600 → 0.149 m (−75.2%).

**+L invariant GREEN** (uses `t_cam_gate` lever, not PnP rotation; `test_obs_sign_faithfulness` passes). Tests `test_gate_range_channel.py` 12/12.

🚩 **NOTE: discard-PnP-rotation (square-flip) upgrade = ZERO code-change** confirmed (`R_cam_gate` already off the fix path) → see `docs/reactivation-2026-06-27/perception-l2-scope.md`. → [[project_vq2_stack_research]]

## CASE-C INTEGRATION (self-localizing loop, #37) — 2026-06-28

### Closed-loop scope (committed a5d232b; `docs/reactivation-2026-06-27/case-c-integration-scope.md`)
vision → localization (C2 chain + bearing-range) → obs[0:20] → inc8 policy = **WIRED + pinned byte-faithful** (golden-tuple 7/7, #37 emulator-fidelity ≤1e-6, +L green). IMU → AHRS → attitude = **MISSING from loop** (7 AHRS filters prototyped, ESKF selected, all UNWIRED).

**CENTRAL GAP = attitude:** `navigator.py:416` reads ODOMETRY quat into R_wb (predict + PnP lever) + `state_estimator.py:215` copies euler/rates into NavState obs[6:12] — both VQ2-blocked. ESKF must replace both via ONE new `use_ahrs` seam.

**6-step build plan:** 2 parallel (gyro audit, adapter shim) DONE, then 4 serial (wire behind `use_ahrs=False` → route into NavState → case-C deploy profile → dual-Navigator fidelity harness).

### Gyro plumbing audit (committed 077eefd; `docs/.../gyro-plumbing-audit.md`)
`angular_rate_body` sourced ONLY from `ODOMETRY.{roll,pitch,yaw}speed` (`mavlink_client.py:275-277`) = VQ2-blocked; `HIGHRES_IMU` gyro NEVER parsed (no `gyro_body` field on DroneState).

🚩 **`angular_rate_body` carries ODOMETRY-convention sign = −1 × true-FRD** (`_ODO_RATE_SIGN=[-1,-1,-1]`); raw `HIGHRES_IMU` gyro is body-FRD (no rotation).

**Minimal fix (option A — zero downstream recal):** add `gyro_body` to DroneState, parse `xgyro/ygyro/zgyro`, feed raw FRD gyro to ESKF, write AHRS out as `−(gyro_body−bias)` into `NavState.angular_rate_body` (true-FRD → ODOMETRY-convention so obs/controller unchanged). Gate behind `use_ahrs=False`.

### AHRS adapter shim (committed 2d927a7; `src/racer/ahrs/ahrs_adapter.py`)
`AHRSAttitudeSource` wraps `ESKFAHRS`, standalone / UNWIRED / additive (transparent-wrapper bit-identical 1e-15).

**Contract:** `R_wb` / `q_wxyz` TRUE FRD→NED wxyz (drop-in for OUTPUT of `R_world_from_odo_quat_wxyz` @ `navigator.py:416`); `euler_rpy` via `euler_from_quat_wxyz` (== `state_estimator.py:215`); `body_rate` = gyro−bias = `R_i2b@w` (TRUE FRD, NOT raw/quat-FD).

**Validation:** convention parity 1e-12; tracking p90 STATIC 0.084°/SPIN 0.183°/ROLLING 5.846°; 19 tests, AHRS suite 70 pass. → [[project_vq2_stack_research]]

### 🚩 DOUBLE-CONJUGATION FOOTGUN (case-C wiring)
The ESKF emits TRUE attitude DIRECTLY — there is **NO** `R_y(π)` ODOMETRY telemetry-frame conjugation to undo. The current `estimator_state_for_obs` keeps the RAW ODOMETRY quat precisely so `build_obs` can conjugate it (`_ODO_QUAT_TRUE_CONJ`) + applies `_ODO_RATE_SIGN` to the rate. An AHRS source **must BYPASS both** — wiring GAP#2 must NOT re-apply `_ODO_QUAT_TRUE_CONJ` / `_ODO_RATE_SIGN` or it double-conjugates. (Adapter emits +TRUE-FRD rate; consumers expect ODOMETRY-sign → the `use_ahrs` seam applies the −1 flip per the gyro audit.)

### CASE-C STEP 3 DONE — use_ahrs WIRING (committed da280e2)
ESKF attitude source wired into the nav loop behind `NavigatorConfig.use_ahrs` (**DEFAULT FALSE = byte-identical**).
- **contracts.py:** `gyro_body` field on `DroneState` (raw HIGHRES_IMU gyro, body FRD).
- **mavlink_client.py:** parses `HIGHRES_IMU xgyro/ygyro/zgyro` → `gyro_body`; DEFENSIVE (`hasattr` guard → `None` when a message variant/fake omits gyro; only consumed under `use_ahrs` so OFF stays byte-id + ingest never crashes).
- **navigator.py:** `use_ahrs` seam feeds `AHRSAttitudeSource(accel,gyro,dt)` → its `R_wb` for KF predict + PnP lever.
- **state_estimator.py:** routes adapter `euler_rpy`/`body_rate` into NavState, handling the double-conjugation footgun (ESKF emits TRUE attitude → NOT re-conjugated) + the rate sign flip (+TRUE-FRD → ODOMETRY-convention) so `build_obs`/controller unchanged.
- **Verified:** 77 targeted tests (use_ahrs OFF-byte-id + ON-smoke, +L green, firstcontact/vq2_loadday/mavlink green).
- 🚩 **GOTCHA CAUGHT:** wiring agent's first cut parsed gyro UNCONDITIONALLY (`msg.xgyro` direct) → crashed 8 tests whose HIGHRES_IMU fakes omit gyro (AttributeError); fixed with the `hasattr` guard.
- **REMAINING case-C steps:** 5 (case-C deploy profile turning the `use_*` flags ON together) + 6 (dual-Navigator fidelity harness = the actual #37 answer: ESKF case-C obs vs truth-attitude oracle obs).

## VQ2 APPEARANCE (live 2026-06-29) — glowing-red/low-light, NOT photoreal

**CONFIRMED by Fengyou on VQ2 load 2026-06-29.** VQ2 actual visual environment = LOW-LIGHT, HIGH-CONTRAST scene with GLOWING RED (emissive) gates. "Photorealistic" was an overstatement; the BINDING visual reality is dark background + glowing/emissive red gate borders.

**OVERTURNS:** the standing assumption that VQ2 visuals = photorealistic renders (which had justified P5 photoreal detector + GS/NeRF hardening direction). P5 as originally scoped (Blender Cycles photoreal-match, GS/NeRF domain randomization toward daylight renders) is WRONG for this appearance and DEPRIORITIZED until recon confirms otherwise.

**IMPLICATIONS (commander analysis; mark as hypotheses-to-verify-in-recon):**
- **Detection likely EASIER, not harder:** glowing red on dark background = HIGH-SNR, color-segmentable target. A red-glow segmentation + corner extractor is a strong simple baseline; heavy photoreal modeling may be unnecessary. VERIFY in recon.
- **GS/NeRF / photoreal-hardening WRONG direction:** do NOT pursue Blender-Cycles/GS/NeRF photoreal rendering until recon confirms it is relevant to this appearance domain.
- 🚩 **REAL CAVEAT = GLOW BLOOM (geometry-relevant):** emissive gate blooms/saturates → can SMEAR the exact corner pixels. PnP + the bearing-range channel key off the corner SPAN; bloom that inflates the apparent square biases range and pose. **MUST measure bloom extent in recon and re-validate gate-corner→localization→bearing-range chain against it.** This is the primary appearance-specific risk for the estimator.
- **DETERMINISM helps:** sim is deterministic → this appearance is FIXED and repeatable. Characterize glow/contrast/bloom/motion-blur ONCE and bake it; NO appearance domain randomization needed for VQ2.
- **Training imagery is OFF-DISTRIBUTION:** current training images rendered as photoreal daylight are wrong-domain for the detector. Re-capture/re-render in the true low-light/glowing-red appearance before any VQ2 detector fine-tune.
- **What is UNAFFECTED (appearance-agnostic):** plant/distillation, estimator math (EqVIO, bearing-range channel, AHRS), case-C self-localization wiring (`use_ahrs`), RL substrate, obs contract (+L), gate geometry (1.5 m inner square = unchanged).

**LOAD-DAY RECON CONFIRMED 2026-06-29 (12 labeled 640x360 frames, handoff/vq2-recon-2026-06-29/frames/curated/):**
- Scene mean gray ~36/255 (very low-light). Glowing-RED (emissive) gate borders on near-black structure.
- Clutter: blue direction-chevrons INSIDE gates, orange floor light-beams, blue floor lane-lines (converge toward next gate), green/red pole markers, white ceiling-truss + floor grid, numbered station pillars 01-20, AND multiple distant red gates in view simultaneously.
- 🚩 **DETECTOR THRESHOLD RULE: gate corners on SATURATED RED CORE (R≥250, red-dominant) ONLY.** Bloom halo (~12px outer, ~3px crisp inner core edge) + low-amplitude red ambient wash INFLATE apparent square → bias range NEAR if thresholded too loosely. Extreme-near approach = corner BLOW-OUT (saturates white) → dead-reckon final approach on IMU.
- Structured grid/truss/lane-lines = rich VIO features between gates (helps vision-pinned yaw/z).
- 🚩 **P5 photoreal/GS-NeRF direction RETRACTED** (see APPEARANCE section above). Re-scope to low-light/red-glow training data.

## VQ2-WIRE-RECON-2026-06-29 (live build 1.0.3379; handoff/vq2-recon-2026-06-29/RECON.md)

### Wire confirmed (live)
- **ALL §9.3 blocks confirmed live** in BOTH training AND competition (byte-identical message sets). The "training may expose more" hedge REFUTED. VQ1 (build 3364) streamed LPN+ODOMETRY+TRACK_INFO; VQ2 does NOT.
- Present: HEARTBEAT(10Hz), HIGHRES_IMU(117Hz), 30Hz JPEG cam (udpin:14550/video:5600, 24-byte header unchanged from VQ1), ENCAPSULATED_DATA=RACE_STATUS(4Hz, data_type=1), ACTUATOR_OUTPUT_STATUS(95Hz, 4 motor outputs idle 0.05), TIMESYNC(response-only).
- ARM via MAV_CMD 400 ACCEPTED (result 0).

### 🚩🚩 NO MAGNETOMETER, NO BAROMETER
- `HIGHRES_IMU.fields_updated=63` (bits 0-5) = accel + gyro ONLY. Mag (xmag/ymag/zmag) + baro (abs_pressure/pressure_alt/temperature) = NaN/absent.
- **CONSEQUENCE: YAW has NO inertial reference** (gravity gives roll/pitch only; gyro yaw drifts unbounded). **ALTITUDE has no baro.**
- YAW + Z MUST COME FROM VISION. ESKF/AHRS demotes to a roll/pitch leveler. EqVIO joint filter (built, unwired) becomes the estimator SPINE — gate-corner + structured-grid bearings pin yaw+z.
- AHRS adapter's mag param is moot (always None in VQ2).

### RACE_STATUS.active_gate_index (gate ordering SOLVED)
- `ENCAPSULATED_DATA` data_type=1 @4Hz carries `active_gate_index` = current target gate index.
- Gate ORDERING/sequencing is SOLVED on the wire — the old "gate ordering without a map" open question is ANSWERED. Perception needs to DETECT+LOCALIZE the active gate among multiple red markers + build a LOCAL gate map from vision (no global map on wire).

### ACTUATOR_OUTPUT_STATUS (sysid gift)
- @95Hz, 4 motor outputs; idle 0.05.
- DISTILL pillar can identify the inner-loop plant from REAL (actuator→IMU accel/gyro) sim data, not only synthetic.

### 🚩 OPEN BLOCKER — CONTROL-MODE HANDSHAKE (CRITICAL PATH)
- `SET_POSITION_TARGET_LOCAL_NED` velocity setpoints in sim default ACRO mode → drone tumbled + env COLLISIONs id=1002.
- ARM accepted (MAV_CMD 400 result 0). Controllable angle/position mode needs a mode-switch/handshake (possibly TIMESYNC-only no-heartbeat regime). **UNRESOLVED; needs focused ShadowPC investigation. BLOCKS all closed-loop VQ2 flight.**

### docs/first_contact.md now STALE
- Describes VQ1 pose+map wire; VQ2 has neither. Do NOT use as VQ2 wire reference.

## VQ2 RED-GLOW DETECTOR BUILT (5f7d842, 2026-06-29; `src/racer/vision/red_glow_detector.py`)
Classical no-model detector for glowing-red VQ2 gates; additive/opt-in; emits the SAME `GateObservation` contract as `detector.py` (corners_px/corner_ids/conf/score/bbox, IPPE_SQUARE order 0=LL,1=LR,2=UR,3=UL) → drop-in for gate_pose/localization; YOLO path untouched + default.

**Approach:** segment SATURATED CORE (R≥250 & R−G≥80 & R−B≥80, NOT a low red threshold) → morph-close bloom notches → contour + reject clutter by red-dominance (blue chevrons/lane-lines, green markers, white truss/grid) + shape/aspect (orange floor beams); extract inner-square corners from red ring's inner HOLE (primary), outer-quad-inset fallback (×0.7 conf) when hole fragmented; DROP blow-outs + border-clipped (no fabricated pose → final approach dead-reckons on IMU). Multi-gate: all returned, primary = highest score (size × squareness).

**Validation on recon frames:** near 01/02 DETECT (PnP 10.4/9.3 m, reproj ≤0.5 px); no-gate 07/08/10 REJECT; blow-out 06/09 graceful 0.

🚩 **CORE-vs-NAIVE span inflation +31% (f01)/+17% (f02):** a low threshold biases PnP range ~31% NEAR; core threshold avoids it.

🚩 **RANGE FLOOR ~15 m:** far gates (>~15 m) have NO saturated core → detection dead zone far side; <1 s lookahead at speed; may need a far-gate centroid/bearing channel later.

Tests: `test_red_glow_detector.py` (9 tests) + fixtures `tests/fixtures/vq2_recon/`.

## MAG-FREE VISION-YAW/Z DESIGN (f1fd597, 2026-06-29; `docs/reactivation-2026-06-27/magfree-vision-yaw-scope.md`)
Driven by VQ2-WIRE-RECON `fields_updated=63` = accel+gyro ONLY — no mag/no baro → yaw+z MUST come from vision.

**Root cause:** ESKF accel update is RANK-2 YAW-BLIND (`eskf.py:301`, H=−skew(g_hat), zero yaw column) → yaw drifts on gyro-z bias. ESKF/AHRS demotes to a roll/pitch leveler.

**Yaw fix:** from the WELL-CONDITIONED gate bearing/+L lever, NOT PnP rotation R_cam_gate (resolves the square-flip tension). Map-free ABSOLUTE backstop = **Manhattan-world VANISHING POINTS** (floor grid/ceiling truss/blue lane-lines → drift-free heading with NO gate in view). 🚩 gate-bearing-yaw needs a SELF-BUILT local gate map (wire gives only active_gate_index, no positions) → VP-yaw is the true map-free anchor.

**Z fix:** existing gate-relative vertical fix + new floor-plane height channel.

**Architecture = Option C (ship B, hold C as escalation):** SHIP Option B = light vision-yaw + vision-z scalar corrections into existing ESKF+C2 chain, composes with `use_ahrs` seam, byte-identical OFF. HOLD full `EqVIOJointEKF` (`eqvio.py:552`) as gated escalation if B's budget fails (3–5× cost).

**Build plan (3 parallel → 4 serial):** ESKF `update_yaw` scalar pseudo-msmt FD-pinned body-dphi Jacobian; `vision/heading_vp.py`; `vision/floor_height.py` → wire behind `use_gate_bearing_yaw`/`use_vp_yaw`/`use_floor_height` flags threading `RACE_STATUS.active_gate_index` → case-C profile.

🚩 **Footguns:** yaw Jacobian body-dphi vs world-yaw (FD-pin required); angle-wrap innovation; head-on yaw degeneracy (gate in boresight = yaw unobservable from bearing); VP 90° lattice ambiguity (floor grid symmetric → need lane-lines or truss asymmetry to break).
