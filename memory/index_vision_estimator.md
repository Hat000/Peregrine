# Vision & Estimator Sub-Index
Mid-level index for the estimator race-speed verdict, VISION-PKG2 specs, gate mapper, case-C readiness, estimator robustness, and advisor-triage queue. Deep detail in topic files below.

## ESTIMATOR-RACESPEED VERDICT (2026-06-13; handoff/ultracode-estimator-racespeed-2026-06-13/REPORT.md)
- **Case-C absolute world-frame nav = NO-GO at ANY speed:** deployable in-plane ~0.55 m vs 0.155 m gate-4 margin (3.5×); SPEED-FLAT.
- **FIX = GATE-RELATIVE** (map bias drops out → 0.11–0.21 m RMS = **CONDITIONAL-GO**, velocity-prior-sensitive; straddles 0.155 m).
- **RewindKF = DEFAULT** (ships regardless; already built).
- Policy obs must be gate-relative → **P4-C05 DONE (f50b9b4, 692 green) — see §P4-C05 below.**
- 🚩 **ORGANIZER-PIVOT: build gate-relative REGARDLESS** — Q① = load-bearing vs free-insurance only (3 emails UNANSWERED; do NOT gate engineering on organizer answers; keep ONE nudge ~weekly).
- COUPLING: faster speed worsens gate-4 margin/σ ratio → estimator accuracy must be verified at each speed rung. **PRIMARY MARGIN GUARD = GATE-4** (0.155 m @ r=0.38; reporting discipline: always quote margin as function of r ∈ {0.21,0.26,0.30,0.33,0.38} at p90/p99 — never a single number; central planning radius = 0.30 m, worst-case stress knob = 0.38).

## COLD-MARGIN CLOSURE (2026-06-13; BLUEPRINT.md + REPORT.md in handoff/ultracode-gate-relative-pipeline-design-2026-06-13/; SHADOWVISION UPDATE below)
**TRI-CONFIRMED VERDICT: CANNOT-SETTLE-OFFLINE.** Source: d3 full-lap sim + v3 verifier + commander independent cross-check (3 independent constructions; all agree on load-bearing numbers). The gate-relative OBSERVATION fix is SOLID and MUST be built (map bias drops EXACTLY: rel E_bias −0.000 vs abs/submap +0.176; RMS 0.139 m). But the margin DOES NOT close offline in any regime.

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

### Binding factor (SHARPENED by radius reconciliation + SHADOWVISION)
- **The binding factor = effective systematic attitude/accel bias entering gate-4, NOT the contact radius and NOT per-fix σ.** The radius correction (~1.5× budget) does not move the binding case. The cold@1.4° p90 (0.338 m) fails at every physically admissible radius.
- **🆕 SHADOWVISION SHARPENING:** per-fix lateral σ IS SOLVED (0.10 m measured). The remaining binding factor is **camera pointing / fix DENSITY** (primary lever, policy-addressable via inc8 gate-in-FoV reward) + **ESKF attitude-bias estimation** (secondary, residual worst-case). See §SHADOWVISION UPDATE above.
- **HOPE FLAG:** The cold@1.4° design point is likely PESSIMISTIC — it treats the ATTITUDE_NOISE_STD_RAD (1.4°, σ) as if it were a constant systematic bias. The true systematic drift-causing bias is probably smaller. **If the live recording pins it ≤ ~0.6°, the margin CLOSES at r=0.30.** This is the decisive open question.
- **ESKF attitude-bias estimation (bias state) = SECONDARY MARGIN LEVER** (elevated from raise-ceiling stash; residual worst-case after pointing is solved). Confirms d3's "ESKF/attitude pipeline in the critical path for the MARGIN, not just absolute nav."
- Cold-prior risk = **HIGH** (variance/window-driven, NOT init-driven). "Warm by gate-4" is FALSE — a full lap does NOT pre-converge velocity to warm quality. (But denser fixes via pointing ACCELERATE cold-prior convergence.)
- Closure rests on **camera pointing/fix-density (primary) + attitude/accel-bias control ≤~0.6° (secondary) + uncertainty-aware speed-down**, NOT the velocity channel and NOT the contact radius and NOT per-fix σ (solved).

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

## Spec facts
- VFoV≈58.7° (spec's "90°" = HFoV; camera tilts up).
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

## P0-CASEC-FOUNDATION DONE (2026-06-13; branch pending merge; 692→700 green, 0 regressions)
- **P0-a:** `_initialize` now gates position/velocity seed on `config.use_given_position` / `use_given_velocity` flags (mirroring per-tick guards at :315/:320). True case-C seeds origin @ pos_std=5.0 (P[0,0]=25) even with LOCAL_POSITION_NED broadcasting. This closed the leak that made every case-C test secretly case-A.
- **P0-b:** `time_since_vision_update_s` measured on IMU master clock (delta_epoch learned once recv-paired, re-learned on reset()), plus predict-forward fallback (`reconcile_vision_clock=False`, `vision_latency_const_s`). delta_epoch=0 on same-clock data → back-compat exact. **OOSM capture-time rewind DEFERRED to C2** — implemented at navigator STAMPING level only; LinearKF has no `update_position_at`. tsv-observable effect identical.
- **P0-c:** case-C cold-velocity = KF pos/vel coupling; already exported via `NavState.velocity_ned` + `pos_vel_covariance[3:6,3:6]`. Confirmed + guard-tested; no behavior change.
- New test files: `tests/test_obs_sign_faithfulness.py` + one P0 integration test file.

## Topic file pointers
- [[project-phase2-rl-vision-decisions]] — vision/estimator/case-C sections: §VISION-PKG2, §ESTIMATOR-RACESPEED, §ORGANIZER-PIVOT, §CASE-C-READINESS, §SPEED-CEILING-ANALYTIC, §GATE-MAPPER, §PARALLEL-SYSTEMS, §ADVISOR-TRIAGE-2026-06-11.
- [[project-detector-training-pipeline]] — Adroit YOLO-pose training; v2/v3; weighted-PnP; adroit-connector ops.
- [[project-estimator-robustness]] — adaptive R, innovation-gate, map-avg, SEARCH state, optical-flow, RewindKF.
