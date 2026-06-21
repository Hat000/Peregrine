> 🚩 **SUPERSEDED 2026-06-19 (511e85c) re: the σ_p0 bar.** Any "σ_p0 ≲ 0.08 / NO-GO vs 0.08 / near-field-gate-estimator pivot" conclusion in this report is OVERTURNED — the 0.08 bar was a `margin_envelope.py` double-count of the drone (real gate clearance 0.75−r ≈ 0.37–0.47 m). The REAL bar is **σ_p0_lat ≲ 0.15 (p99 ≲ 0.45)**; measured 0.15–0.20 = MARGINAL-PASSING, NOT NO-GO. Gate-4 is a REACH/PASS-RATE problem; RL stays the tool (the "pivot off RL" is RETRACTED). The engineering + measurements below STAND; only the bar and its GO/NO-GO verdict are corrected. → MEMORY.md:8 + memory/project_rl_increment_history.md §inc8-2026-06-19.
# MonoRace SYNTHESIS — race-winning deltas vs the Peregrine stack

**Session:** MonoRace synthesis (ultracode fan-out: 4 dimensions → synthesis) · **Date:** 2026-06-18 · **Model:** Opus 4.8
**Source:** *MonoRace: Winning Champion-Level Drone Racing with Robust Monocular AI* — Bahnam, Ferede, Blaha, Lang, Lucassen, Missinne, Verraest, De Wagter, de Croon (TU Delft MAVLab). arXiv **2601.15222**, won A2RL 2025 (beat all AI teams + 3 human world champions, up to 100 km/h).
**Escape hatch:** NOT invoked — `arxiv.org/html/2601.15222` fetched cleanly across all four dimensions. ⚠️ All MonoRace numbers/equations below are LLM-summarized from the arXiv HTML; any number tagged **[verify vs PDF]** must be confirmed against the source before it becomes load-bearing for a decision. No equations were fabricated.
**Method:** commander pre-extracted the paper per-dimension (consistent content, no fetch race), then 4 analyst agents compared each dimension against our stack while reading the actual repo files, then 1 synthesis agent ranked + deduplicated. 5 agents, ~462k tokens.

---

## BLUF — the three findings that matter

1. **MonoRace's headline calibration contribution (offline IoU + Bayesian-opt extrinsic recovery) is REDUNDANT for us, and our method DOMINATES it.** We independently built a near line-for-line port (`handoff/p1-vision-accuracy-2026-06-14/scratch-calib-v2/iou_bo_calib.py`, GP+EI ≈40 evals, recovers pitch+roll to ≤0.008°). Worse for adoption: MonoRace's objective is **angle-only**, but our binding bias is **metric / range-flat** (−0.27 m over 14–26 m, P3 N=13,308) on a **spec-fixed 20° mount** — an angle-only BO would mis-fit it. Keep `frames.BORESIGHT(vert_offset_m=-0.25)` as shipped; IoU-BO is at most a Round-2 drift-watchdog for the *physical* Neros mount.

2. **Most of our architecture divergences from MonoRace are CORRECT and forced by the AI-GP wire, not gaps.** MonoRace runs a 16-state EKF (estimates attitude+IMU-bias) feeding a 500 Hz direct-**motor** G&CNet. We run a 6-state LinearKF (attitude is *given* on the wire) feeding **CTBR rates at <100 Hz onto a stabilized inner loop** — because the scored wire gives no raw-motor channel. Our 20-dim obs is at parity-or-better (next-gate lookahead present; fix-confidence triple replaces their wire-illegal motor-speed channels). Net new adoptable architecture: small.

3. **Exactly one idea lands on the live critical path, and exactly one is a clean adopt-now code change.** Critical-path: layer a *persistent absolute* cross-track centering penalty onto our delta-form `through_centering_reward` to kill the inc8 lineage's `success_rate=0` "pay-back" under-centring. Adopt-now: a pre-calibrated fixed-delay **fallback** for the RewindKF timestamp path. Everything else is Round-2 or interesting-but-no.

### TOP-3 ADOPTABLE IDEAS (returned)
1. **Persistent absolute cross-track centering penalty** on `rl/inc8_reward.py`, layered on the delta-form `through_centering_reward` — the only synthesis idea on the live critical path (attacks inc8 `success_rate=0`). *Effort S; tag: Round-2 (gate behind the recenter run 9cecf64 / job 3276449 so it doesn't confound the single-variable test).*
2. **Pre-calibrated fixed-delay fallback + age-clamp** in `navigator._vision_fix_time_imu_ns` so a TIMESYNC epoch-reconciliation failure degrades to a constant (~17 ms edge / ~115 ms CPU) instead of mis-timing a fix. *Effort S; tag: adopt-now; error-path-only, byte-identical default.*
3. **Corruption-robustness benchmark of the EXISTING YOLO11s detector** (not a new model) under MonoRace-style heavy per-frame corruption — converts "segmentation *might* be more robust" into a measured number that retires or justifies the GateNet/QuAdGate A/B before any L-effort. *Effort M; tag: Round-2.*

---

## The reproduced IoU extrinsic-calibration recipe (MonoRace) + our verdict

**Objective.** Use the logged **state estimate** + the **flight plan (known map of gate geometry)** to **re-project** each gate's mask into every logged image under a *candidate* camera→body extrinsic, and compare the predicted gate mask to the **actually observed NN gate segmentation** via **Intersection-over-Union (IoU)**. Maximize **mean IoU** across the flight log. Presented as general ("refine any state-estimation parameter") but used **mainly for the angular extrinsic offsets** (camera→body pitch/roll/yaw), motivated by a flexible TPU camera mount whose angles drift between flights.

**Data.** Onboard flight logs only — gate segmentation images + logged state estimates + the known gate map. No GT pose, no dedicated calibration flight, no motion-capture.

**Optimizer.** **Bayesian Optimization** (GP surrogate), chosen for good results in few samples under race-day time pressure. **≈40 optimizer calls** [verify vs PDF].

**Reported results** [all verify vs PDF]:
- *Sim*, 2° initial error: IoU **0.83 → 0.91** after 40 iters; recovered extrinsics **(1.9, 54.6, 2.3)°** vs GT (2, 55, 2)°.
- *Real*, 40 calls: IoU **0.64 → 0.78**; sub-degree accuracy — pitch **0.14°** (σ 0.09), roll **0.49°** (σ 0.20), yaw **0.71°** (σ 0.28).

**Step-by-step (as we would reproduce it):**
1. Collect a normal flight log: detector segmentation/corners per frame + logged NavState + the known gate map.
2. Define a candidate extrinsic θ (camera→body pitch/roll[/yaw], and — for us — the metric `vert_offset_m`).
3. For each frame: project map gate corners/mask using (logged pose ⊕ θ); compute IoU vs the observed detector mask/quad; average over frames → score(θ).
4. GP surrogate + acquisition (Expected Improvement) proposes the next θ; ≈40 evals.
5. Return the θ maximizing mean IoU as the refreshed extrinsic.

**VERDICT — REDUNDANT, do not adopt as a replacement.**
- **Method:** we already have it. `iou_bo_calib.py` is a self-contained numpy GP(RBF)+EI BO (≈40 evals) over `cv2.intersectConvexConvex` IoU of reprojected-vs-detected corners, recovering injected pitch+roll to **0.000° noiseless / ≤0.008° @ 0.5 px / 8 frames** (`iou_bo_OUT.txt`). MonoRace offers nothing new on the recipe.
- **Fit to our error:** MonoRace's objective is **angle-only**; our dominant bias is **metric / range-flat** (`FORM_RESOLUTION.md`, N=13,308: rel_vert flat −0.27 m over 14–26 m, refuting angular which would grow ~1.9×). An angle-only BO backs out a spurious pitch that mis-scales our offset across range. Our mount is **spec-fixed at 20°** (no drifting TPU clamp), so the per-flight motivation doesn't apply in the sim qualifier.
- **vs our head-on pin:** our gate-0 CTBR head-on campaign (−0.25 m @ 23 m) is **strictly more capable** — it *discriminated metric-vs-angular* via multi-range structure, which a single-objective IoU-BO at one range is *degenerate* on (`frames.py:39`).
- **The one non-redundant use (Round-2):** productionize `iou_bo_calib.py` into a wired **log-replay QA tool** that *jointly* fits (pitch, roll, **vert_offset_m**) and flags drift from the baked −0.25 m — an advisory watchdog for the **physical Neros mount** that *can* shift on crash/thermal. **Never an auto-bake** (the deep-research memo ruled out online per-flight extrinsic estimation: gauge ambiguity / ESKF double-count at 7% fix density).

---

## Per-dimension deltas

### Dimension 1 — Architecture (mono PnP + IMU + estimator + control)

**MonoRace:** camera → adaptive state-based crop → GateNet (U-Net) segmentation → QuAdGate (LSD) corners → OpenCV PnP → **16-state EKF** `[pos,vel,quat,accel_bias,gyro_bias]` (IMU is the process model; estimates attitude+biases) → G&CNet 500 Hz direct-motor. Distance-/evidence-dependent measurement noise `σ²_pos = 0.02·d²/(N_c²·N_g)` [verify]; **multi-gate (2) single-PnP**; translation-only PnP + KF-attitude beyond 5 m; trace-test outlier rejection `‖x−x_PnP‖² < 16·N_c²·trace(P_pos)`; **pre-calibrated fixed-delay** time-sync (17 ms img / 0.5 ms IMU → RMS 0.289→0.103 m); **IMU-saturation fallback** (residual >22 m/s² → model-based accel + inflate cov).

**Ours:** 6-state LinearKF `[pos,vel]` world NED (`state_estimator.py:64`) — attitude *given* on wire so no quat/gyro-bias states; C2 chain adds RewindKF OOSM (`kf_rewind.py:97`, horizon 0.5 s) + `gate_relative_inplane_fix` (+L anisotropic cov, `localization.py:160`). **Single-gate** PnP (IPPE_SQUARE/P3P, `gate_pose.py:303`; multi-gate flagged as future, `localization.py:22`). Layered outlier rejection: range+reproj pre-gate → 3-DOF Mahalanobis χ²=16.27 → **2-DOF in-plane relinnov χ²=13.82** (`navigator.py:66`, depth-flip-aware). Control = inc7 CTBR at 30 Hz (`fly_rl.py`).

| Aspect | MonoRace | Ours | Who's better / significance |
|---|---|---|---|
| Estimator state | 16-state EKF (estimates attitude+bias) | 6-state LinearKF (attitude given) | **Correct divergence** — their cost buys what our wire gives free. Their bias machinery = a reference for our *deferred* ESKF fallback only. |
| Multi-gate PnP | 2-gate single PnP | single-gate | **MonoRace better** — tightens absolute translation + our weak ~0.8 m depth on co-visible frames. Round-2. |
| Distance-dependent R | `σ²∝d²/(N_c²N_g)` | lateral `a1·r` linear + lever-arm + floors | **Mixed** — our lever-arm already gives r-growth; their N_c term smooths our hard 3-vs-4 P3P cliff. |
| Outlier rejection | trace test ×N_c² | 2-stage χ², depth-flip-aware | **Ours better.** |
| Time-sync | pre-calibrated constant | RewindKF exact per-frame | **Ours more general**; their constant = a cheap *fallback/sanity-bound*. |
| IMU saturation | model-fallback >22 m/s² | **none** (`predict()` trusts accel) | **MonoRace better** — genuinely missing; cheap guard. |
| Control tail | 500 Hz direct-motor | CTBR <100 Hz on inner loop | **Wire conflict — non-adoptable.** |

### Dimension 2 — The IoU extrinsic-calibration recipe
Fully covered above. **REDUNDANT + angle-only mis-fit.** Our `iou_bo_calib.py` + head-on metric pin dominate. Only Round-2 value = productionized joint-DOF drift watchdog for the physical mount.

### Dimension 3 — Appearance-robust detector (GateNet U-Net seg + QuAdGate vs our 8-kpt YOLO11s)

**MonoRace:** U-Net (enc 64-128-256-512-512, ×f=4) → binary mask @384×384 → **QuAdGate** geometric corners (LSD lines → intersections → handcrafted descriptors → RANSAC). Synth+real **3500:500**, augs = HSV + Gaussian noise + motion-blur (square kernel 5–15 px). Headline: completed a full track with **50% of frames corrupted** [verify vs PDF]. 90 Hz on Orin NX.

**Ours:** single-stage 8-keypoint YOLO11s pose (`detector.py`, `gate_yolo11s_curriculum_v2.pt`), ~4 ms INT8/TRT, won Round-1 sim-to-real 39/40 + legacy-crush 47% vs 28% on 240 real frames. Aug pipeline (`cluster/vq2_pose_train.py`, `blender_gen/augment.py`) is **already a strict superset** of MonoRace's: HSV + GaussNoise + MotionBlur(3–15 px, same range) **plus** JPEG/ISONoise/Defocus/Downscale targeting the actual 640×360 jpeg VQ stream, fresh per epoch on a clean photoreal Blender base.

- **Decisive:** the "adopt their augs into our YOLO" idea is **already done and exceeded** — re-porting is a no-op/regression.
- **The one place segmentation might genuinely win:** robustness to heavy per-frame corruption (mask + geometric RANSAC vs hallucinated keypoint regression). Our robustness is distributed (detector aug + Tukey-PnP + χ² gate) but **unbenchmarked under corruption** → measure before any swap.
- **Highest data lever (architecture-independent):** a modest **manually-labeled real Round-2-arena anchor** once the official VQ2 3D-scanned arena drops (~2026-06-29), folded into the existing auto-label flywheel.

### Dimension 4 — Estimator/Control vs our open gaps (cold-start, 12 m floor, terminal centering)

**MonoRace:** PnP range-tiered (full 6-DOF **2–5 m**, translation-only beyond 5 m); 24-dim G&CNet obs (current-gate pos/vel + Euler + ang-vel + **motor speeds** + next-gate rel pos/yaw); reward = progress + gate-pass + **absolute** off-center penalty `λ_offset‖p_k − p_gk‖`; **assumes initial gate visibility — no no-position cold-start solution**.

**Ours:** gate-relative +L pseudo-fix with anisotropic cov → LinearKF/RewindKF → 20-dim obs. Next-gate lookahead **already present** (`nxt_rel`+`nxt_relyaw`, `fly_rl.py:354-355`); we substitute a **fix-confidence triple** `obs[17:20]` for their wire-illegal motor channels. Terminal centering = `centering_reward` (near-gate) + **delta-form** `through_centering_reward` (9cecf64). Fixes modeled by `fix_surrogate.py`: accept ≈0 below 12 m, peak ~0.84 in 18–26 m — our floor is **near-range**, the **inverse** of MonoRace's 2–5 m full-PnP regime.

- **Centering reward form is the load-bearing delta:** their *absolute* penalty gives a persistent gradient that does NOT cancel; our *delta/telescoping* form can be "paid back" by a late re-centre — exactly the inc8 under-centring failure (`success_rate 0`). → **adds a small persistent absolute term** (top-3 #1).
- **Cold-start:** MonoRace does **not** solve it — it sidesteps by assuming first-gate visibility (same assumption we make). Nothing adoptable; honest no-op.
- **Range floor:** their tiering is the *inverted* regime (near-range full PnP); weak support for relaxing our 12 m cutoff (HIGH risk it's real PnP-flip instability, not conservatism).
- **Obs / control:** parity-or-better; direct-motor control is wire-illegal.

---

## Ranked adoptable-ideas list

| # | Idea | Gain | Effort | Tag | Concrete change | Source dim |
|---|---|---|---|---|---|---|
| 1 | **Persistent absolute cross-track centering penalty** layered on delta-form `through_centering_reward` | Kills inc8 `success_rate=0` "pay-back" under-centring; raises chance a policy holds σ_p0_lat≤0.08 + completes a lap | S | **Round-2** (live critical path; gate behind recenter run) | `rl/inc8_reward.py` add `through_centering_abs(curr_inplane,rw)=-rw*curr_inplane`, range-gated; weight 0.0 OFF in `Inc8RewardWeights`; wire at `peregrine_racing_inc8.py:344`. Single-var A/B after job 3276449 reads out. | 4 |
| 2 | **Pre-calibrated fixed-delay fallback + age-clamp** for RewindKF timestamp path | Robustness on deployed case-C stack: TIMESYNC failure degrades to a constant vs mis-timing a fix (their sync cut RMS 0.289→0.103) | S | **adopt-now** | `navigator.py:_vision_fix_time_imu_ns`: when `delta_epoch` None/non-finite → `now_ns - vision_latency_const_s` (~0.017 s edge / 0.115 s CPU), clamp t_fix∈[0,horizon] w/ `kf_rewind.assert_horizon_gt`. Default = exact behavior (byte-identical). | 1 |
| 3 | **IMU-saturation detector → model-predict + Q-inflate** in LinearKF | Removes high-G coast-drift predict bias the gate-4 σ_p0 budget is sensitive to | S | **Round-2** | `state_estimator.py:LinearKF.predict:110`: residual vs constant-accel model > ~22 m/s² → substitute model accel + inflate Q; `sat_threshold` default inf (byte-identical). Tune from at-speed HIGHRES_IMU. | 1, 4 |
| 4 | **Corruption-robustness benchmark of existing YOLO11s** (no new model) | Converts "seg might be more robust" into a measured number; retires or justifies the GateNet A/B | M | **Round-2** (top-3) | New offline harness reusing `detector.py:GateDetector` + corruption transforms in `blender_gen/augment.py`, scored vs `handoff/vq2-blender-render-2026-06-15` + 240 real frames; report fix-rate vs corruption-fraction through the relinnov χ² gate. | 3 |
| 5 | **Attitude-fixed multi-gate (2-gate) single-PnP** | Tighter absolute translation + weak ~0.8 m along-track depth on co-visible frames; denser obs fix stream | M | **Round-2** | `gate_pose.py:303` → new `estimate_multigate_pose` stacking ≥2 gates' 3D points into one `cv2.solvePnPGeneric` (fixed-attitude translation, map yaw via `gate_mapper`); batch in `navigator._process_observation:456`. Gate `use_multigate_pnp`, OFF=byte-identical. | 1 |
| 6 | **Productionize `iou_bo_calib.py`** into a joint-(pitch,roll,vert) log-replay drift watchdog (advisory) | Cheap ε re-pin riding race logs; safety net for physical-mount crash/thermal shift | M | **Round-2** | Promote to `scripts/boresight_iou_recal.py`; feed 8-kpt corners + `track_map.json` + logged state; add `vert_offset_m` as 3rd BO axis via `localization.py:84` +L path; cross-check metric/angular w/ `level_hover_regression.py`. **Advisory only — never auto-bake `frames.BORESIGHT`.** | 2 |
| 7 | **Confirm-and-document no-op:** 20-dim obs at parity-or-better | Avoids a wasted obs-redesign + wire-illegal dead end | S | **adopt-now** (decision) | No code. Record: next-gate lookahead at `fly_rl.py:354-355`; confidence triple per `estimator_obs.py`/d5 contract; motor-speed channels rejected as wire-illegal. | 4 |
| 8 | Smooth N_c cov scaling vs hard 3-vs-4 P3P cliff | Graceful fix-trust degradation near gates | S | interesting-but-no | `localization.py:217` replace binary `P3P_FIX_COV_INFLATION` with bounded `(4/N_c)²`-style; N_c=4 byte-identical. | 1 |
| 9 | Soft in-plane-only degraded fix in 9–12 m guard band | Could push fix density below 12 m floor toward gate-4 | M | interesting-but-no | **MEASURE first** (`navigator._apply_gate_relative_fix` / `fix_surrogate.accept_rlo`); HIGH risk the cutoff is real PnP-flip instability; MonoRace evidence is inverted-regime (2–5 m). | 1, 4 |
| 10 | Direct-motor 500 Hz G&CNet | None for us | L | interesting-but-no | N/A — violates the scored wire (no raw motor channel, <100 Hz, stabilized inner loop). | 1, 4 |
| 11 | d² range-squared cov tail (>24 m) + verbatim aug/U-Net swap | Near-zero | S | interesting-but-no | (a) `localization.py` blend `a1·r`/`c·r²` past ~24 m [verify 0.02 vs PDF] — fixes rarely consumed >12 m. (b) NO aug change (already superset); seg swap deferred to Round-2 A/B gated on rank-4 + VQ2 arena frames. | 1, 3 |

---

## MEMORY-DELTA (≤10 lines, for the commander to bank)

1. **MonoRace IoU-BO calibration = REDUNDANT** — we already built+validated `iou_bo_calib.py` (GP+EI ≈40 evals, recovers pitch+roll ≤0.008°) AND it's angle-only (mis-fits our METRIC range-flat −0.27 m bias on the spec-fixed 20° mount); our head-on −0.25 m pin DOMINATES (it discriminated metric-vs-angular; IoU-BO at one range is degenerate). Keep `frames.BORESIGHT=-0.25`; IoU-BO only a Round-2 drift watchdog for the *physical* mount.
2. **TOP synthesis idea (live critical path) = add a PERSISTENT absolute cross-track penalty to `inc8_reward.py`** layered on the delta-form `through_centering_reward`; targets the `success_rate=0` "pay-back" under-centring. Gate behind recenter job 3276449/9cecf64; single-var A/B; symmetric obs-20 critic may still struggle (footgun); risk = S3 `rw_centering` cliff.
3. **Only true adopt-now CODE idea = pre-calibrated fixed-delay FALLBACK** (~17 ms edge / 115 ms CPU) + age-clamp in `navigator._vision_fix_time_imu_ns` when TIMESYNC `delta_epoch` fails; error-path-only, byte-identical default, pairs with `kf_rewind.assert_horizon_gt`.
4. **IMU-saturation detector GENUINELY MISSING** from `LinearKF.predict` (`state_estimator.py:110` trusts accel unconditionally) — cheap Round-2 guard vs high-G coast-drift bias; needs at-speed HIGHRES_IMU to tune (residual>22 m/s² → model-predict + Q-inflate; `sat_threshold` default inf).
5. **Multi-gate 2-gate single-PnP** adoptable Round-2 (`gate_pose.py:303` → `estimate_multigate_pose`, batch `navigator.py:456`, OFF=byte-identical); tightens weak ~0.8 m along-track depth on straights, not gate-4 terminal; `localization.py:22` already flags it.
6. **Decision-value Round-2 spike: benchmark EXISTING YOLO11s under heavy per-frame corruption** (`detector.py`+`augment.py` vs 240 real frames) BEFORE any GateNet/QuAdGate seg A/B. MonoRace 50%-corrupted-frame claim is LLM-summarized — verify vs PDF.
7. **CONFIRMED no-op:** our 20-dim obs already has next-gate lookahead (`nxt_rel`/`nxt_relyaw` `fly_rl.py:354-355`); MonoRace's 4 motor-speed channels are WIRE-ILLEGAL; our `obs[17:20]` confidence triple is the correct case-C substitute. Do not chase "add lookahead" / "add motor speeds".
8. **MonoRace aug set (HSV+GaussNoise+MotionBlur 5–15px) is a STRICT SUBSET of `cluster/vq2_pose_train.py`** (we add JPEG/ISONoise/Defocus/Downscale for the 640×360 jpeg stream). Re-porting = no-op/regression; Round-2 aug retrain b0yz1s3oz was already neutral on clean data.
9. **Direct-motor 500 Hz G&CNet = interesting-but-no by wire** (confirmed dim 1 & 4). The 9–12 m soft-degraded-fix idea is interesting-but-no (HIGH risk the 12 m cutoff is real PnP-flip instability; MonoRace's 2–5 m evidence is inverted-regime).
10. **MonoRace is mostly CONFIRMATORY, not new:** their estimator/control divergences from us are forced by *their* harder problem (no attitude on wire, direct-motor control). The wire-legal transferable kernel = reward shaping (centering) + time-sync fallback + IMU-sat guard + multi-gate PnP. Their flagship (IoU cal) we already had; their detector augs we already exceed.

---

## Provenance / caveats
- Paper fetched from `arxiv.org/html/2601.15222`; numbers/equations are LLM-summarized — **[verify vs PDF]** tags flag the load-bearing ones (50%-corruption claim, IoU 0.64→0.78, σ²=0.02·d² constants, 40-call BO budget, λ_offset value).
- Per-dimension analyst agents read the live repo and cited real file:line anchors (`state_estimator.py:64/110`, `localization.py:22/84/160/217`, `navigator.py:66/456/546`, `gate_pose.py:303`, `kf_rewind.py:97`, `frames.py:39/58/215`, `fly_rl.py:354-355`, `inc8_reward.py`, `peregrine_racing_inc8.py:344`, `fix_surrogate.py:17,113-118`, `detector.py`, `cluster/vq2_pose_train.py`, `blender_gen/augment.py`, `iou_bo_calib.py`).
- This report is the deliverable; **no `memory/` files were touched** and no commit was made (commander's banking/merge gate).
