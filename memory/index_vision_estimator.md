# Vision & Estimator Sub-Index
Mid-level index for the estimator race-speed verdict, VISION-PKG2 specs, gate mapper, case-C readiness, estimator robustness, and advisor-triage queue. Deep detail in topic files below.

## ESTIMATOR-RACESPEED VERDICT (2026-06-13; handoff/ultracode-estimator-racespeed-2026-06-13/REPORT.md)
- **Case-C absolute world-frame nav = NO-GO at ANY speed:** deployable in-plane ~0.55 m vs 0.155 m gate-4 margin (3.5×); SPEED-FLAT.
- **FIX = GATE-RELATIVE** (map bias drops out → 0.11–0.21 m RMS = **CONDITIONAL-GO**, velocity-prior-sensitive; straddles 0.155 m).
- **RewindKF = DEFAULT** (ships regardless; already built).
- Policy obs must be gate-relative → **P4-C05 is FOUNDATION** (see [[index-rl-training]]).
- 🚩 **ORGANIZER-PIVOT: build gate-relative REGARDLESS** — Q① = load-bearing vs free-insurance only (3 emails UNANSWERED; do NOT gate engineering on organizer answers; keep ONE nudge ~weekly).
- COUPLING: faster speed worsens gate-4 margin/σ ratio → estimator accuracy must be verified at each speed rung. **PRIMARY MARGIN GUARD = GATE-4** (0.155 m @ r=0.38).

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
- **3 P0 bugs:** _initialize crutch / TIMESYNC / cold-start untested.
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

## Topic file pointers
- [[project-phase2-rl-vision-decisions]] — vision/estimator/case-C sections: §VISION-PKG2, §ESTIMATOR-RACESPEED, §ORGANIZER-PIVOT, §CASE-C-READINESS, §SPEED-CEILING-ANALYTIC, §GATE-MAPPER, §PARALLEL-SYSTEMS, §ADVISOR-TRIAGE-2026-06-11.
- [[project-detector-training-pipeline]] — Adroit YOLO-pose training; v2/v3; weighted-PnP; adroit-connector ops.
- [[project-estimator-robustness]] — adaptive R, innovation-gate, map-avg, SEARCH state, optical-flow, RewindKF.
