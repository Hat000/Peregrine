# Parked Backlog Register — Peregrine

STANDING register of banked-but-parked threads — the anti-"banked-then-forgot" mechanism. **REVIEW every session AND whenever a trigger/gate fires** (POC unfreeze, COWORK-1 organizer answers, an Adroit contact, a fresh recording). Sweep origin: 2026-06-14 (70 items surfaced across all memory). Dispositions:
- **F = FOLD-NOW** — actionable; in/into an active path (P1/P2/P3) or a one-off dispatch.
- **T = DORMANT-TRIGGERED** — parked until [trigger] fires; do not spend before then.
- **D = DEAD/SUPERSEDED** — closed; recorded to prevent re-litigation.

🔑 **COWORK-1 CLUSTER KEY — ✅ LANDED 2026-06-14** (handoff/cowork-2026-06-14/competition-intel.md): Q① = NO pose streamed → estimator LOAD-BEARING; Q⑤ = eval is desktop GPU → LATENCY axis CLOSES (#13 ONNX/TRT now lower-urgency for VQ, still the physical-round portability play); Q④ = offline retune BETWEEN runs ALLOWED → #48/#53 between-attempt refinement LEGAL. UN-GATED (blocked→ALLOWED; triage on merit, NOT all critical-path): #12, #47, #48, #49 (vision-only confirmed), #53, #58 (Round-2 3D-scan → appearance gap REAL), #67. #13 demoted (VQ latency closed).

---

## 🔼 FOLD-NOW (F)
- **#3** ✅ **DONE (P1, a1620b3, 2026-06-14):** bake site = frames.R_camera_from_body; NO code bug (round-trip 1.48e-8 m; fy=320/cy=180 consistent render+PnP). The 0.56° = render-vs-decode convention constant. (Closed.)
- **#4** ⏳ **INHERITED by overall commander (2026-06-14):** surrogate σ vs L3 gate-4 σ reconciliation (surrogate pooled σ_vert 0.28 vs L3 gate-4 0.10). P3 confirmed the L3 at-speed recording is DONE → this is ANALYSIS, folded into the boresight-closure + margin-re-run pathway. Full 37 m/s recording TODO (gated on the held inc8 envelope-relaxed policy).
- **#5** bearing-resolve ShadowPC `data/runs/*_l3atspeed_*` via shadow_gate4.py → **P3** boresight-lock alternative.
- **#6 / #28** 30 Hz gating + per-gate last-accepted-fix distance from L3 → **P3** at-speed recording (feeds speed-ceiling analytic + #43).
- **#13** ONNX/TensorRT export of detector v2 + in-loop latency benchmark → **NEW dispatch**; CLOSES the ≤50 ms LATENCY axis if eval=GPU (COWORK-1). Export laptop-doable now; on-target benchmark folds into P3.
- **#15** ✅ **VALIDATED + IN USE (P1, 2026-06-14):** MonoRace IoU-BO recovers pitch+roll 0.000° noiseless / ≤0.008° @0.5 px = the boresight ANGULAR-form estimator. (Closed.)
- **#16** ✅ **VALIDATED + IN USE (P1, 2026-06-14):** level-hover WLS splits intercept(metric) vs slope(angular) = the ANGULAR/METRIC/MIXED form classifier. (Closed.)
- **#22** sim regression-suite promotion (8 scripts handoff/ultracode-substrate-audit → tests/; resolve test_confirmed_cr4_03 slug-collision) → one-off hygiene (prevents silent regressions).
- **#23** CR1-01 + P1-C06 yaw-active capture rider → fold into **P2** inc8 fresh-reset live batch (closes abs-yaw wire sign + obs yaw seam in one capture).
- **#25** ShadowPC simops tooling push to main → **P3 #3** (partly on origin 1b54cb8; verify complete).
- **#36** inc8 reward design (lean banked, NOT frozen) → **P2** prepares ablations; **Fengyou owns freeze**.
- **#37** estimator-emulation obs wrapper + v* (gate-4 approach-speed) instrument → **P2 HARD PREREQ** (selection on obs_from_truth crowns a fiction).
- **#39** scipy-SLSQP toy MPCC probe (measures real tracking-k for the decomposed-architecture ceiling) → optional cheap one-off, P2-adjacent.
- **#40** ✅ **DONE (P2 W#2, merged 5a4afa6, 2026-06-14):** rl/reference_line_inc8.json + build_reference_line.py — dead-centre, orientation-fixed, drag-feasible UPRIGHT, R1'-ready. (Closed.)
- **#55** BSR3 spin-gate (spin_rate_abort→9–10, spin_time_abort→3.0 s) → **P2** prereq before ANY inc8 run.
- **#60** organizer Qs (6 open: Q①④⑤ + Q-A/B/C/D) → **COWORK-1** attacks; keep ONE weekly nudge.
- **#62** ✅ **RESOLVED (2026-06-14):** Adroit acceptable-use PERMITTED with rules — Fengyou provided the Princeton RC AUP; banked in reference_adroit_princeton.md §Acceptable-Use Policy + MEMORY.md footgun. (Closed.)
- **#30 / #58** 🟢 **ACTIVATED (2026-06-14) — ONE overnight session DISPATCHED (B CUT as redundant, Fengyou-caught):** (A) VQ2 photoreal Blender/Cycles dataset GENERATOR [ultracode] (bpy, runnable on ShadowPC — intrinsics==K[[320,0,320],[0,320,180]] @640x360 +20° mount, appearance-randomized incl. VQ1-red+wide-hue to kill gambling-red overfit, Cycles photoreal + motion-blur/flares, keypoint-aware albumentations, YOLO-pose labels matching src/racer/vision/{synthetic,detector,gate_pose}.py, 5 scenario presets, viz-verify, OUTPUTS a standard ultralytics data.yaml; handoff/vq2-blender-dataset-2026-06-14). 🚩 **NO new training harness needed — cluster/yolo_train_v3.sbatch IS the harness (standard `yolo pose train` on a data.yaml; ultralytics does val metrics + early-stop; trained v2/v3).** Morning training = copy yolo_train_v3.sbatch, point data= at the Blender data.yaml, drop the gen step (A pre-renders) = ONE-LINE edit. Only genuinely-new residue = sim-to-real eval on REAL VQ1 frames (existing pipeline validates synthetic-only) — SMALL, post-training, not a session. Morning: ShadowPC runs A → upload → reuse yolo_train_v3.sbatch. A reports branch UP for merge. → was [[index-vision-estimator]] P5 photoreal.
  - ✅ **LANDED on main (d437507, 2026-06-15, 28 tests green, pure additions): src/racer/vision/blender_gen/ + 6 presets + VQ2_VISION_TRAINING_PLAN.md.** 🚩 **MonoRace (A2RL'25 winner, near-1:1 match) THESIS: photorealism is the BASE not the lever — the transfer levers are APPEARANCE DIVERSITY + MOTION BLUR + a GEOMETRY-KEYED LABEL + ~30% HARD NEGATIVES; gate color = NUISANCE axis (real DCL gate = SOLID PURPLE frame NOT LED/orange; orange was drone-LED+VQ1-styling); venue = DARK INDOOR ARENA (Sept SoCal + Nov Columbus indoor) not daylight.** Generator: optical-frame PnP-exact labels (<1e-9px vs frames.py; live ≤1px self-check), real-course viewpoints, ProceduralBackend (laptop, no Blender) + BlenderBackend (bpy/Cycles ShadowPC, doc-verified-not-render-verified → intrinsics self-check + viz overlay = live arbiters on 1st ShadowPC run). **COMMANDER ANSWERS (2026-06-15): Q-color = color-as-nuisance CONFIRMED (VQ1-orange anchor retained; hard-negs make it discriminative). Q-arch = EMIT MASKS (cheap hedge, keeps segmentation open w/o re-gen) + keypoint-pose stays BASELINE but segmentation+geometric-corners (MonoRace's appearance-robust path) = FIRST-CLASS A/B in the detector-recipe session (#32/#33, now un-gated; corners feed the same PnP→KF → small downstream cost); sim-to-real eval picks winner.** 🚩 **MonoRace→BORESIGHT loop: MonoRace's biggest accuracy lever = offline IoU-based extrinsic calibration from flight logs = EXACTLY our ε_vert −0.25 boresight (frames.BoresightCorrection ready) → champion validates the approach; connects to the inc8 conversion-gap diagnostic.** 🚩 PROCESS: worker committed to main DIRECTLY (2nd branch-collision instance; landed safe but luck) → reinforce worktree isolation. → [[index-vision-estimator]] · [[reference-prior-art]]

## ⏳ DORMANT-TRIGGERED (T) — [trigger]
- **#38** ⭐ **speed-ladder envelope relaxation** (rw_tilt 96→48 ~1.4 s; free-cone 60→75–80° ~1.5 s) — **THE biggest lap-time lever** (cone tax 6.89→4.72 s) — [POC unfrozen + inc8 live-confirm, one step at a time].
- **#2** fix-rate sweep fold into POC — [POC unfrozen].
- **#7** sigma_theta 1.4° refit — [N≥100 at-speed fixes from inc8-class flights].
- **#10** TIMESYNC epoch reconciliation (navigator.py:426) — [full LIVE OOSM needed; C2 predict-forward fallback ships now].
- **#11** range-anisotropic R (depth ∝ r²) — [>24 m fixes start mattering].
- **#12** K+L+N stage-2 estimator bundle (reprojection-EKF / temporal-parallax / Fisher-info KF) — [Q①=vision-only + stage-2] · COWORK-1.
- **#14** 2-corner translation-only PnP fallback — [terminal close-range fix-drought confirmed] · relevant to TERMINAL gate-lock margin.
- **#17** roll-wander τ_pre re-measure — [fresh uncontaminated 6/6 recording].
- **#19** LK sub-pixel corner tracking (M) + rolling-Mahalanobis innovation monitor (Q) — [SHADOWPC-VISION-CAL].
- **#20** SEARCH/acquisition state in mission.py (yaw/alt scan when no gate visible) — [blind-segment robustness needed].
- **#21** optical-flow translation-velocity assist (long blind gaps) — [inter-gate blind-gap data shows need].
- **#27** hard-coast + map-averaging layer-3 — [mapper built = #64].
- **#29** full-lap case-C sim at ~37 m/s gate-4 — [inc8 envelope-relaxed policy exists].
- **#31** adaptive state-based ROI cropping (extends >32 m range) — [case-C co-visibility wall].
- **#32** heteroscedastic per-corner σ (NLL/RLE detector head) — [next detector retrain = P5]. 🟡 **QUEUED: this + #33 (stronger/newer arch) + a better aug recipe = the "detector-recipe improvement" session Fengyou offered (2026-06-14); FIRES once photoreal dataset A lands (tune the recipe ON that data, not before). Per-corner σ feeds weighted-PnP → fix ACCURACY → estimator + emulation-fidelity (see index_rl_training §EMULATION-FIDELITY). Commander to write the prompt when A's data exists.**
- **#33** YOLO26-pose retrain + A/B — [photoreal v4 = P5].
- **#34** kpt_conf_thresh 0.5 relaxation — [real confidence distribution measured].
- **#41** TOGT re-run with mixer coupling — [post-inc8 bound refresh].
- **#42** inc8 per-tick gate crossings from debug_obs — [rung-2/3 gap attribution].
- **#43** 60/100 Hz retrain — [last-fix ≤10 m at speed confirmed; analytic says likely UNNEEDED].
- **#44** per-segment tilt-cost concentration analysis — [if global rw_tilt step insufficient].
- **#46** learned residual dynamics on twin (parity-gated, off by default) — [experiment desired].
- **#47** MPC-shadow safety floor + critic-as-risk monitor — [Q①/Q⑤] · COWORK-1.
- **#48** between-attempt trajectory refinement (exploits determinism) — [Q④ state-persistence] · COWORK-1.
- **#49** stage-2 case-C noise-model extension (1.8–3.4 m lateral) — [Q①=vision-only] · COWORK-1.
- **#50** runtime validity supervisor / watchdog demotion — [submission mechanics known].
- **#51** multi-frame far-gate PnP (3–5 frame accumulate) — [>32 m gates matter].
- **#52** IMM filter bank (hover/cruise/transit) — [KF divergence on sharp maneuvers observed].
- **#53** ILC-style feedforward correction across laps — [Q④] · COWORK-1.
- **#54** fly_rl.py default ckpt inc4→inc7 — [P4 unfreeze / next fly_rl touch]. DEV footgun only (submit_rl.py pins inc7 on judged path).
- **#57** Adroit non-lapse config-matrix V100 parity gate — [next Adroit contact] (the lapse configs are DEAD — S18 voided).
- **#58** full VQ2 photoreal detector campaign (Adroit, 10–50k scenes) — [pilot #30 justifies + COWORK-1 shows appearance gap].
- **#59** PVNet dense-corner-voting escalation — [P3P fallback failing at gate transit].
- **#61** submission interface details + FLOSS disclosure doc — [organizer email / pre-submission].
- **#63** fully-autonomous submission stack (remove human GO/disarm) — [pre-submission; partly done via submit_rl.py autonomy-hardening].
- **#64** GTSAM offline mapper (per-gate marginal cov → localization R) — [case-B/C mapping needed; scipy least_squares first].
- **#66** min-snap geometry rung (rpg_trajectory_generation numpy port) — [VQ2 quality; folds with #40].
- **#67** async heavyweight detector ~5 Hz full-res → rewind-corrected KF — [Q①/Q⑤] · COWORK-1.
- **#68** TVM / NVIDIA TAO INT8 edge-deploy toolchain — [Sept+ physical round].
- **#69** ⏳ **CONFIRMED LIVE (2026-06-14, inc8 GPU smoke):** peregrine_repo on /scratch IS a stale flat copy (no cluster GitHub creds) → refresh needs a `git archive` tarball of the target commit + a manual upload per code change = real iteration friction (one upload per re-smoke/ladder). Workaround works; PROMOTE to deploy-key git clone (replace /scratch file copy) when Adroit iteration cadence justifies — [Adroit sessions grow / next multi-round GPU phase].
- **#70** VIO escalation (VINS-Fusion / OpenVINS) — [attitude biased/missing OR blind-segment drift too large].
- **#71** min-snap vs natural-cubic descent-overshoot trim on the inc8 reference line — [if descent overshoot is shown to hurt; minor] (P2 W#2, 2026-06-14).

## ⛔ DEAD / SUPERSEDED (D) — do not re-litigate
- **#1** G3 re-run at σ=0.10 — SUPERSEDED by margin-closure-envelope (already re-derived at σ=0.10, 7654e99).
- **#8** predict-forward (constant calibrated age) — SHIPPED in C2/P0-b (reconcile_vision_clock=False).
- **#9** full KF rewind/OOSM — PRODUCTIONIZED as RewindKF (C2 05ed750; gated off by default).
- **#18** vision-velocity LSQ channel — REFUTED (honest σ_v 2.81 m/s; P2 insurance only, NOT load-bearing).
- **#24** KF accel_body convention — CR1-01 SETTLED (navigator.py:313 rotate-by-TRUE-conjugated is correct).
- **#26** absolute-loop Mahalanobis innovation gate — largely SUPERSEDED by C2 relinnov gate χ²(2,.999)=13.82 (re-confirm coverage if a wrong-gate-teleport is ever seen).
- **#35** Elodin adapter `orientation_ned_wxyz` assert — unreachable (only if Elodin solver-glue + Navigator eval runner ever built).
- **#45** S19 mixer per-axis κ contradiction — low-impact (policy avoids the saturation regime).
- **#56** S19 mixer-probe row completion (c100/c60/zhov_r31) — same; low-impact.
- **#65** ADRC / Extended State Observer — RL supersedes (retain only as model-based safety-net reference).

---
*Maintenance: when an item is actioned, move it to DEAD with the closing commit/branch. When a trigger fires, move the item to FOLD-NOW and assign an owner. Keep this file the SINGLE register — do not re-scatter parked items back into topic files.*
