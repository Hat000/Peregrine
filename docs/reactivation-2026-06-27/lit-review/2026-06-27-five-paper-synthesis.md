# Five-Paper Lit Review + Synthesis — 2026-06-27

Papers digested (full agent digests in session transcript 2ed2ec3c):
1. **MonoRace** — Bahnam/Ferede et al., arXiv:2601.15222 (Jan 2026). *Winner, A2RL×DCL Grand Challenge Apr 2025, TU Delft MAVLab.* Our near-1:1 prior art.
2. **Swift** — Kaufmann et al., Nature 620 (2023). Champion-level vision+IMU racing.
3. **Reaching the Limit** — Song/Romero/Müller/Koltun/Scaramuzza, Sci. Robot. (2023). RL vs optimal control.
4. **DPVO** — Teed/Lipson/Deng, NeurIPS 2023. Learned sparse-patch monocular VO.
5. **Swift-YOLO** — Sui/Hosoda/Lee, IEEE Sensors J. (Feb 2026). Small-target detector levers.

---

## A. The convergent spine — what ALL champion racers agree on (de-risks our core)

| Design choice | MonoRace | Swift | Reaching-Limit | Our stack |
|---|---|---|---|---|
| Action space = **CTBR** (or motors) | motors @500Hz | thrust+bodyrate | thrust+bodyrate | CTBR ✓ |
| **Gate-progress reward** (Δdist-to-next-gate-center, dense, NO reference trajectory) | yes | r_prog | Eq.5, b=0.01, ±10 | partial |
| **Perception-aware reward** (keep next gate in FoV) | >60° off-axis penalty | exp(λ·δ_cam⁴) | — | inc8 look-at ✓✓ |
| **Gate-relative observation** (4 corners as rel-positions R¹², next N=2 gates) | gate-frame obs | R¹² corners | δp∈R¹², N=2 | ✓ |
| **Tiny MLP policy** | 3×64 | 2×128 | 2×256 | ✓ |
| **Metric scale from known gate size via PnP** | inner 1.5m, IPPE-like | IPPE 4-corner | (mocap) | ✓ |
| **Estimator trusts attitude, corrects translation** | 16-state EKF | drift-KF (trans only) | (mocap) | ESKF/RewindKF ✓ |
| **Asymmetric privileged critic** | — | yes (GT state to critic) | — | currently SYMMETRIC ✗ |
| **Rotation matrix in obs** (not quaternion) | — | yes | yes | check |

**Takeaway:** our architecture is on the champion path. Two independent champion systems use the **perception-aware look-at reward** — strong external validation of the inc8 thesis (our memory had this as a hard-won internal finding; it's now corroborated prior art). The one clear gap vs Swift: our critic is symmetric; Swift confirms privileged-critic is intended.

## B. RL vs OC — CORRECTS my earlier two-mode proposal

Reaching-the-Limit is decisive and counterintuitive: **RL does not optimize better than MPC — it optimizes a *better objective*.** Given the SAME quadratic tracking cost, MPC beats RL. RL wins only because gate-progress (no reference trajectory) lets it re-time/re-shape the path online to whatever is *feasible now*; a precomputed optimal-line reference bakes in a time-allocation the real drone can't hit under model/perception error → tracking it crashes (0% success on realistic model; crashed in the real world).

**Implication for our "map-then-fly-optimal-line on hot laps" idea:** a known map does **NOT** rescue optimal-line *tracking* — that's the exact failure mode this paper exposes. **Revised two-mode design:**
- **Recon lap(s):** conservative flight → build metric gate map + ordering + structure.
- **Hot laps:** the map feeds the RL policy a *longer-horizon gate observation* (N gates ahead, known rel-positions) and provides *relocalization anchors* — but the **executed controller stays RL/CTBR**, not an MPC optimal-line tracker.
- **OC/TOGT used OFFLINE ONLY** to compute the lap-time bound (measure how close RL is to optimal), never as the deployed law.
- ⚠️ **Open: both champions TRAIN PER-TRACK.** Randomized-per-load breaks per-track training. Either (a) generalist policy + map-conditioned obs, or (b) online per-load adaptation (likely infeasible in-competition). KEY UNRESOLVED QUESTION.

## C. Estimator steals — the binding VQ2 risk

MonoRace's estimator is the most directly transferable asset (same sensor suite, minus their known map):
1. **Multi-gate PnP** — fuse corners from the next TWO gates in one solve (even 1 corner from gate-2 helps); the non-coplanar 2nd gate breaks the single-gate translation/rotation ambiguity → ~2° heading gain. Directly attacks our heading-observability risk.
2. **De-rotated fallback** — trust full PnP world-pose ONLY at 2–5 m gate-dist with ≥6 corners; else position = EKF-attitude ⊕ PnP-relative-translation. Decouples our state from noisy far-gate PnP rotation.
3. **IMU-saturation model-substitution** — if ‖measured−model accel‖ > ~22 m/s², feed the dynamics-model accel into the EKF + inflate covariance. Converted 50%→100% finish rate. **We WILL saturate a ~16g accel at 7g maneuvers + vibration.** Near-free to add.
4. **Covariance from corner-perturbation sampling** (Swift + MonoRace) — run PnP on ~20 jittered corner sets, use the spread as R. Sets detection noise from YOLO-pose corner uncertainty.
5. **PnP noise model** σ²_pos = k·d²_gate/(N_c²·N_g) — grows with distance, shrinks with corner & gate count.
6. **AHRS is ours to build** (no ATTITUDE on wire) — MonoRace self-estimates attitude in a 16-state EKF (pos, vel, quat, accel-bias, gyro-bias); mag gives the yaw anchor (verify sim mag fidelity).

## D. DPVO + detector roles (the parallel-fuse branches)

- **DPVO offline recon-lap mapper = PRIMARY use, low risk.** Best-in-class trajectory + sparse map (beats DROID-SLAM by 40% ATE at 8.9× less compute); Fast config 2.5 GB / Default 4.9 GB (fits 8 GB). Monocular = no metric scale → anchor via one-shot Sim3 to gate-PnP (or IMU). Conservative recon-lap speed keeps inter-frame flow inside its 16–72 px training band.
- **DPVO onboard gap-filler = PILOT, don't commit.** Constant-FPS-under-motion is ideal, BUT racing-speed inter-frame flow at 30 Hz may exceed its 7×7 correlation search / 16–72 px training band → tracking loss. Plus Windows port of its custom CUDA BA/correlation is a real risk. Test on our fast-flight footage first.
- **DPVO ships NO IMU fusion** — that's our task: loosely-coupled ESKF over DPVO rel-poses (fast path) or IMU-preintegration factor in its BA (VI-DSO is the blueprint).
- **Swift-YOLO levers for small-gate-at-range corners:** shared-DCNv2 keypoint head (deformable sampling reaches corners of a ~20px gate), SALD attention-downsampling (preserves sub-10px corner info), BFusion learnable fusion (stops near-scene swamping rare far-gate signal). Earlier detection → more look-ahead. Modest but real; do NOT put LEMSC at P3.

## E. Parallel-fuse perception graph (user's compute principle, concretized)

Run concurrently, fuse in the EKF (latency = max branch, not sum):
- **B1:** YOLO-pose detector → multi-gate PnP → metric gate-relative fixes (sparse, low-latency, primary).
- **B2:** learned VO (DPVO-class) → relative motion + structure (occlusion gaps + recon mapping).
- **B3:** AHRS (IMU) → attitude @ high rate.
- (opt **B4:** monocular depth prior for gap scale.)

## F. GEM-HUNT — next pulls (prioritized; niche/recent = competitive edge)

**TOP 3 — pull first:**
1. **Bosello et al., "On Your Own: Pro-level Autonomous Drone Racing in Uninstrumented Arenas"** — arXiv:2510.13644 (2025). *A competitor's full stack for racing with NO instrumentation/map — our exact no-map problem, independent solution. Possibly the single most relevant paper after MonoRace.*
2. **Ferede et al., "One Net to Rule Them All: Domain Randomization in Quadcopter Racing Across Different Platforms"** — arXiv:2504.21586 (2025). *The control net + 40–50% DR recipe that IS MonoRace's G&CNet. The engine room.*
3. **De Wagter, Paredes-Vallés, Sheth, de Croon, "...winning entry to the 2019 AIRR competition"** — Field Robotics 2 (2022), arXiv:2109.14985. *GateNet→QuAdGate→PnP→EKF lineage, full implementation detail, no mocap. Estimator goldmine. (Flagged by 2 digests.)*

**SECOND WAVE:**
4. **Bahnam et al., "Self-Supervised Monocular Visual Drone Model Identification through Improved Occlusion Handling"** — arXiv:2504.21695 (2025). No-mocap self-calibration; root of the reprojection-IoU extrinsic calibration (attacks our ε_vert boresight footgun).
5. **Li & de Croon, "Unsupervised tuning of filter parameters without ground-truth"** — IEEE RA-L 4(4), 2019. Tune our EKF/ESKF noise with no GT.
6. **DPV-SLAM / DPV-SLAM++** — Lipson/Teed/Deng, CVPR 2024. DPVO + loop closure + global opt; the recon-mapper upgrade.
7. **Foehn/Romero/Scaramuzza, "Time-optimal planning for quadrotor waypoint flight" (CPC)** — Sci. Robot. 6, eabh1221 (2021). The gold-standard lap-time bound tool (compare to our TOGT).
8. **Kaufmann/Bauersfeld/Scaramuzza, action-space benchmark** — ICRA 2022. Empirical CTBR justification.
9. **CUAHN-VIO** — Xu & de Croon, RAS 185 (2025). Learned mono VIO w/ uncertainty (fallback localization).
10. **NeuroBEM** — Bauersfeld et al., RSS 2021. High-fidelity BEM aero model for our sim/DR. **LEAP-VO** (CVPR 2024) — dynamic-scene VO, DPVO competitor.

## G. Fork status after this read
- **Fork A (VIO or not):** mapping REOPENED by multi-lap-within-load → DPVO recon mapper YES (offline), onboard VO = pilot. Gate-PnP stays primary.
- **Fork B (learned front-end + filter):** CONFIRMED by all.
- **Fork C (modular vs monolithic):** CONFIRMED modular — every champion is perception→state→tiny-policy.
- **Fork D (RL vs MPC):** RESOLVED — RL/CTBR executed; OC offline-bound only. Known map ≠ track-optimal-line.
- **NEW Fork E (per-track vs generalist policy under per-load randomization):** OPEN, high-stakes.

---

# Round 2 — On Your Own / AIRR-2019 / One Net (2026-06-27 cont.)

## THE HEADLINE REFRAME: nobody published solves our constraint set
Every champion system we've read leans on at least one crutch we are DENIED at VQ2:
| System | Map? | Attitude/scale source | Our denial |
|---|---|---|---|
| Swift (Nature 2023) | known track map | MoCap (train) + T265 VIO | no map, no VIO box, no mocap |
| Reaching-Limit (2023) | known track | MoCap 400 Hz | "" |
| One Net (2025) | fixed single track | MoCap + EKF GT state | "" + track randomizes |
| MonoRace (2025) | known gate flight-plan map | self-EKF (mono+IMU) ✓ | **no map** |
| On Your Own (2026) | **total-station surveyed map** | **stereo T265 VIO box** | **no map, mono-only** |
| AIRR-2019 (2022) | organizer gate map | complementary-filter + laser alt | no map, no altimeter |

**"On Your Own" is mislabelled for our purposes:** "uninstrumented" = no MoCap for fine-tuning; they STILL survey gates with a total station and run a stereo-VIO box. **Mono-only + no-map + per-load-randomized is genuinely uncharted in the published record.** That is simultaneously our biggest risk (no proven recipe) and our biggest edge (competitors copying papers inherit the map/VIO assumptions we can't use — and neither can they at VQ2).

## FORK C REOPENED HARD — pixels-to-control without an estimator
One Net cites **Geles et al., "Demonstrating agile flight from pixels without state estimation," arXiv:2406.12505 (2024)** — UZH, end-to-end pixels→control, NO explicit state estimator. The modular "perception→state→policy" consensus is consensus *among systems that already have good state* (VIO/map/mocap). We don't — our estimator IS the binding risk. A pixels-to-control policy **sidesteps the exact thing we're worst at.** Do NOT treat modular as settled. Fork C is live and arguably more relevant to us than to any paper's authors.

## KEY NEGATIVE RESULTS to respect (don't repeat their dead ends)
- **One Net: online adaptation FAILED.** Adding last-N actions, state history, and GT/noisy param inputs to fight partial observability — none beat the plain net (their Fig. 4). Tempers our recon-lap fast-adaptation enthusiasm; validate before committing architecture.
- **One Net axis caveat:** their "one net" generalizes over PLATFORM DYNAMICS, not TRACK GEOMETRY. It does NOT prove a single net handles layout randomization — that's our untested hypothesis (the axis-swap).
- **Reaching-Limit: tracking a precomputed optimal line fails under model error** (already logged Fork D).
- **DR has a speed tax** (One Net: generalist 15–25% slower than specialist) — on a leaderboard, over-randomizing costs lap time. Tune DR breadth against measured lap time, not just pass-rate.

## TRANSFERABLE GOLD (mono+IMU, map-free compatible)
- **AIRR self-consistency principle:** within one lap, map-offset and state-error are indistinguishable and need NOT be separated — "merge odometry with vision without a calibrated metric representation." Deep validation of gate-relative; formalize it for our multi-lap fixed-track case.
- **AIRR gate-prior corner gating** (project expected gate, accept corners only if side-length/vertex-angle error <25%, reconstruct occluded corners from the projected quad with ≥2 valid) — kills banner/jumbotron/distractor false-positives (our photoreal threat), ~0.14 ms. Substitute recon-map / last-good gate-relative pose as the projection source.
- **Attitude-aided PnP** (seed the PnP solve with our AHRS attitude) + **per-axis RANSAC moving-horizon estimator** (200 iters, 80% subset, δ-regularized toward small velocity corrections, 2 s window) — robust outlier rejection layered on the rewind-KF.
- **AIRR risk-aware speed schedule** (fast far → slow to align → punch through blind when gate exits FoV) lifted their success 50→100%; ready-made PPO reward/curriculum shaping for gate-4 reach.
- **PnP ROTATION is discarded by everyone** (On Your Own + AIRR both throw it away). With no VIO attitude, gate-PnP yaw is a FREE signal to bound AHRS yaw drift. Unused-by-prior-art lever.
- **80-frame Grounding-DINO distillation** (On Your Own) retargets a detector to a new appearance domain with ~80 hand-labels; we have unlimited sim GT → go further (zero hand-labels).

## EXPANDED FORK TREE (killing NONE — all live until proven terrible)
**Perception/estimation arch:**
- C1 Modular detector→PnP→filter→state→policy (consensus; weak link = estimator = our binding risk)
- C2 **Pixels-to-control, no explicit estimator** (Geles 2406.12505) — sidesteps our hardest problem
- C3 Hybrid: learned perception → recurrent policy that implicitly estimates state

**Mapping:**
- A1 Mapless gate-relative reactive (detect next gate, fly)
- A2 **Online gate-graph SLAM on recon lap → exploit hot laps** (the multi-lap edge nobody published exploits — they all had surveyed maps)
- A3 Learned VO (DPVO-class) for inter-gate structure, gate-PnP/IMU scale anchor

**Policy generalization (Fork E sharpened):**
- E1 **Layout-DR generalist** — randomize gate count/spacing/angle/ordering in training (axis-swap of One Net; supported-in-principle, unproven)
- E2 Per-load online adaptation (recon embedding / single-throw ID / meta-RL) — ⚠ One Net negative result
- E3 Per-load retrain (likely infeasible in-competition)

**Control:**
- D1 RL/CTBR executed (consensus + Reaching-Limit) — OC offline for bound only
- D2 End-to-end pixels→motor (ties C2)

**Attitude (new scope):** AHRS from IMU; sub-fork = also pin yaw with gate-PnP rotation (free, unused by prior art).

## NOVEL / UNDOCUMENTED-EDGE SEEDS (where we could leap past the frontier)
1. Online gate-graph SLAM during recon lap (multi-lap exploit nobody published does).
2. Layout-DR generalist policy (axis-swap hypothesis).
3. Pixels-to-control for the no-state regime (push Geles further with our compute/sim).
4. PnP-rotation yaw pinning.
5. Per-load recon embedding done right (clean recon signal vs One Net's failed in-episode history).
6. Formalized self-consistency across laps (AIRR principle × multi-lap).

## NEXT PULLS (prioritized)
1. **Geles et al., "Demonstrating agile flight from pixels without state estimation," arXiv:2406.12505 (2024)** — TOP; the no-estimator fork for our hardest constraint.
2. **Hanover et al., "Autonomous Drone Racing: A Survey," IEEE T-RO 40 (2024)** — full field map for systematic gem-mining.
3. **Ferede et al., "End-to-end RL for time-optimal quadcopter flight," ICRA 2024** — parent of One Net; full obs/dynamics/motor derivation.
4. **RATM dataset** — github.com/tii-racing/drone-racing-dataset (real annotated gate-corner + IMU + champion-pilot trajectories; +6 MCK flights).
5. Blaha, "Control of Unknown Quadrotors from a Single Throw," arXiv:2406.11723 (2024); Zhang/Loquercio, "Single near-hover controller for vastly different quadcopters" (2023) — the adaptation fork (E2).
6. Li et al., "Visual model-predictive localization ... 72-g drone," J. Field Robotics (2020) — the MHE/RANSAC estimator math.

---

# Round 3 — Geles / Hanover survey / VSLAM survey / PAMPC+HPA-MPC / PA-MPPI / E2E-RL / GS-nav (2026-06-27 cont.)

## Decisive finding per paper
- **Geles "pixels without state estimation" (RSS 2024):** the "no estimator" only works because the **track is memorized in the weights** (gate ordering in the privileged critic + per-gate init buffer); per-track trained; #1 failure = **gates-out-of-view collapse (no recurrence)**. ⇒ For our randomized-per-load track, **dropping the estimator is a TRAP** (Fork C2 is NOT a free win). GOLD to steal regardless: (a) **analytic-mask-as-bottleneck** — project gate inner-edges analytically, train RL with NO rendering (<100 µs/frame), decouples detector from policy; (b) **asymmetric critic is load-bearing — symmetric critic = 0% on every track** (independent confirmation of our live footgun: inc8 critic is symmetric); (c) **exp(−δ⁴_cam) look-at reward**; (d) **per-gate initial-state buffer** for gate-4 reach. Their failure mode (no gate in view) is exactly where OUR IMU+estimator is the cure — they threw IMU away; we have it.
- **Hanover "Autonomous Drone Racing: A Survey" (T-RO 2024):** our regime (mono, no-map, per-load-random) is **literally 2 of their 5 named open challenges**; "**no competitive real-world system is map-free**" (Swift uses a prior gate map+VIO). Strongest claim: "**state estimation is the most resilient [classical] part; geometric VIO backends beat end-to-end learned VIO**" ⇒ **stay modular with learned blocks; do NOT go pixels-to-CTBR.** The reframe: **build the per-load gate map on the recon lap, then race against it** = fuse the ONLY proven racing-estimator (known-map localization) with the per-load constraint that breaks everyone's pre-baked maps → **per-load randomization becomes our moat, not our threat.**
- **VSLAM-for-UAVs survey (Sensors 2024):** dated/classical, thin on 3DGS, but: build-once-localize-thereafter is **endorsed prior art** (ORB-SLAM3 Atlas relocalization, ORB-SLAM-VI map-reuse → zero-drift). **Multi-lap-within-load REVIVES loop closure** (laps 2..N = revisits — the mature, best-validated machinery applies). DBoW2 = cheap relocalization baseline. Our gate-PnP metric anchor is white-space the classical lit doesn't use.
- **PAMPC (IROS 2018) + HPA-MPC (RA-L 2025):** perception cost = `‖(fx·xc/zc, fy·yc/zc)‖²` (project POI to image center) + `‖ṡ‖²` (anti-motion-blur). **Verdict: mine these as RL REWARD terms, NOT as the controller** (PAMPC needs known POI position = downstream of the estimator anyway; C++ codegen; validated only 1–3 m/s). The centering term is a principled upgrade to our look-at (bakes in real fx=fy=320 + 20° tilt); the **anti-blur term is a NEW reward channel we lack**, valuable at 30 m/s. **Best novel seed: Fisher-information / observability-aware reward** — reward trace/log-det of PnP information over the 4 corners (prefer poses where corners are spread & resolvable), not mere centroid-centering → directly attacks gate-4 vertical-σ.
- **PA-MPPI (RA-L 2026) — the surprise high-value fork:** GPU-batched sampling MPC, **17,500 samples × 15-step horizon at 50 Hz on a 6 GB laptop GPU**. **Zero training; re-plans online every step ⇒ inherently per-load with NO retrain ⇒ candidate SOLUTION to generalist-vs-per-track.** We already have GPU-batched dynamics (**DiffAero**) = near-1:1 infra reuse; native CTBR output; camera-pointing cost = keep-gate-in-view. Weak: assumes depth/occupancy (we drop that, keep MPPI core + gate-pointing cost from PnP), validated only ≤1.3 m/s, assumes perfect state (mocap).
- **E2E-RL (Ferede, arXiv:2311.16948, 2023 — NOT the ICRA24 we guessed):** gate-relative 24-dim obs, progress + gate-center reward, **γ=0.999 induces time-optimality with no explicit time term**, first-order motor lag. Direct-motor beats CTBR on peak agility BUT **6× worse sim-to-real** ⇒ validates our CTBR choice (trades peak agility for the robustness we need under noisy perception). Gate-center reward is built-in suboptimality (discourages edge-cutting).
- **GS-for-navigation scout (web):** **VERDICT — onboard hot-lap splat-relocalization = research-risky distraction** (30 Hz + 8 GB + motion blur all exceed validated envelopes at once; the whole drone-GS field AVOIDS putting splats in the hot loop — only Splat-Nav closes the loop, at 1.5 m/s for collision avoidance). **GS-as-offline-photoreal-simulator = genuine low-risk edge** (SOUS VIDE / FalconGym recipe: build 3DGS of the training track → photoreal sim to train/harden the RL policy). Cheaper relocalizers (ORB-SLAM3+DBoW2, DPVO+PnP) likely beat GS for our case. **FalconGym** (NeRF/GS gate-racing sim, 95.8% gate success) is the most directly relevant — pull it.

## ⭐ EMERGENT CANDIDATE STACK (leading thesis — alternatives stay alive)
A differentiated assembly that maximally reuses our DiffAero batched-dynamics infra:
1. **Perception:** YOLO-pose gate detector (+ Swift-YOLO small-target levers) → multi-gate PnP (MonoRace) → metric gate-relative fixes; AHRS from IMU (+ gate-PnP yaw pinning, the unused-by-prior-art lever).
2. **Estimator (KEEP — do not drop):** gate-relative ESKF/RewindKF + IMU-saturation model-substitution + de-rotated fallback. This is the cure for Geles's gates-out-of-view collapse.
3. **Map (the moat):** build a per-load **gate-graph map online during the recon lap** (the known-map localization the whole field relies on, but we construct it). Exploit on hot laps 2..N (revisit → loop-closure machinery applies).
4. **Control — HYBRID RL-prior + MPPI:** one generalist RL policy (trained with heavy **layout domain-randomization**) supplies the agile racing-line PRIOR; **MPPI on the DiffAero model samples around that prior online**, adapting to the per-load track with ZERO retraining. Best-of-both: RL's learned line × MPPI's instant per-load adaptation. One codebase (DiffAero) serves training AND deployed rollouts.
5. **Reward/perception-awareness:** PAMPC projection-centering + anti-blur + **Fisher-information-over-corners**, per-gate init buffer, γ=0.999 time-optimality.
6. **GS = OFFLINE ONLY:** 3DGS of the training track as a photoreal sim to harden the policy (FalconGym/SOUS VIDE recipe).

## TRIPLE-CONFIRMED: fix the symmetric critic
Swift + Geles + our own SSOT all say a **symmetric critic catastrophically fails** (Geles: 0% on every track). Our live footgun = inc8 critic is symmetric (obs-input 20-dim, never wired to the privileged 36-dim). **Highest-confidence, on-critical-path fix when we resume building.**

## Updated fork tree (still killing none, but with leanings)
- **Control:** D1 RL/CTBR · **D3 NEW: Hybrid RL-prior + GPU-MPPI (leading)** · D2 pixels→motor (Geles — trap under randomization, gold to mine) · OC offline for bound only.
- **Perception arch:** C1 modular (survey-favored, KEEP) · C2 pixels-no-estimator (TRAP for randomized track; mine the analytic-mask + asym-critic) · C3 hybrid recurrent-implicit.
- **Mapping:** A2 online gate-graph SLAM on recon → exploit (the moat, leading) · A1 reactive mapless (robust fallback) · A3 learned-VO structure.
- **Generalization (E):** **MPPI-online-replanning (D3) may dissolve this fork** — no per-load policy needed · E1 layout-DR generalist (still needed as the MPPI prior) · E2 recon adaptation (⚠ One Net negative) · E3 retrain (infeasible).

## Round-3 GEM-HUNT (next pulls, prioritized)
1. **Cioffi et al., "Learned inertial odometry for autonomous drone racing"** (RA-L) — IMU-only learned odometry; the cure for gate-occlusion gaps.
2. **FalconGym 1.0/2.0** (arXiv:2503.02198 / 2510.02248) — NeRF/GS gate-racing photoreal sim; the safe GS use, directly applicable.
3. **Romero/Song/Scaramuzza, "Actor-Critic Model Predictive Control"** (ICRA 2024) — differentiable-MPC layer on an RL actor = the principled form of our hybrid RL+MPPI fork.
4. **Minařík et al., "MPPI Control for Agile UAVs"** (IROS 2024) + **Williams et al., "MPPI: theory to parallel computation"** (JGCD 2017) — the racing-MPPI + canonical GPU-MPPI to implement D3.
5. **Xing et al., "Contrastive learning for robust scene transfer in vision-based agile flight"** (ICRA 2024) — photoreal-sim→track appearance-gap for the detector.
6. **CUAHN-VIO** (Xu & de Croon, RAS 2025) — learned mono VIO w/ uncertainty (gap-filler + confidence obs).
7. **SOUS VIDE** (arXiv:2412.16346) — GS-as-training-sim → onboard policy, the offline-GS recipe.
8. **Lamberti et al., "Sim-to-real DL for nano-drone racing"** (RA-L 2024) — only stack explicitly engineered for ~8 GB-class compute.

## Open questions still live
- Does MPPI hold at 20–30 m/s racing speed (only validated ≤1.3 m/s)? — spike needed.
- Can recon-lap online gate-graph mapping be built robustly mono+IMU under our narrow FoV?
- Layout-DR generalist as MPPI prior: how much DR before the speed tax bites?
- VQ2-load checks unchanged (mag fidelity, Training-mode GT exposure, occlusion-gap frequency).

---

# Round 4 — MPPI-agile / AC-MPC / Learned-Inertial-Odometry / FalconGym (2026-06-27 cont.)

## Decisive findings
- **MPPI-for-Agile-UAVs (Minařík, IROS 2024) — partially deflates the MPPI fork.** Tops out at **~12 m/s on smooth references, NOT racing speed**; it's a *tracker* (beaten by SE(3) on accuracy), no gates, no time-optimality. BUT proves compute: **100 Hz, 896 samples × 15-step (1.5 s) horizon, fully onboard on an 8 GB Jetson Orin Nano**, CTBR output, full nonlinear model. Never validated under noisy/monocular state — rollouts amplify state error over the 1.5 s horizon (our binding risk). ⇒ **No paper shows MPPI at 30 m/s racing.** Use as an online-planner-around-RL-prior, not a standalone racer. Reusable: dq=1−⟨q1,q2⟩² attitude cost; in-rollout motor clipping; 10 ms/0.1 s interpolation decoupling.
- **AC-MPC (Romero, T-RO 2026 / ICRA 2024) — the better-fit hybrid, nearly our exact stack.** A **differentiable MPC (iLQR) is the last layer of the PPO actor**; a neural cost-map emits diagonal Q,p; PPO backprops through the solver. Critic-Hessian ↔ MPC terminal-cost equivalence (short-horizon MPC + long-horizon critic). **Zero-shot 21 m/s; deploys 50 Hz / 13.5 ms in Python; ONE deterministic solve at inference (leaves VRAM for the detector — unlike MPPI's parallel rollouts).** Buys OOD + **dynamics-change robustness without retraining** (mass +27%, inertia ±80%, wind 1.5×). Reuses **DiffAero** as the required differentiable model. LIMITS: (a) adapts to *dynamics*, **NOT new tracks online** — doesn't beat a well-trained gate-relative PPO on per-load track variation; (b) **NO state constraints** → can't hard-enforce zero-gate-contact; (c) robustness shown to dynamics, **NOT estimator noise** (our actual binding risk); (d) 30× train cost (amortized by DiffAero); diagonal-Q mandatory.
- **Learned Inertial Odometry / "IMO" (Cioffi, RA-L 2023) — ADOPT as the gap-filler, high-value/low-risk.** EKF propagated by IMU + a **TCN that regresses 3-DoF Δp from 0.5 s of (commanded collective-thrust + gyro)**. Beats TLIO by **54%**, **matches camera-VIO-with-gate-map using IMU only**, at 70 km/h (ATE 0.56 m). Nearly 1:1 with our RewindKF/ESKF — TCN = an additive update branch, gated ON in occlusion gaps. KEY: needs the **thrust signal (NOT on the VQ2 wire) — but we know our OWN commanded collective thrust** (exactly what they used on the real drone). We **train in sim with known dynamics → dissolves their data-collection + calibration cost.** Their train-time attitude/gyro-bias noising maps onto our no-ATTITUDE AHRS reality. Limit: doesn't generalize to unseen trajectories → train on a track-randomized distribution; uses mag = our free unused signal.
- **FalconGym (Miao, arXiv:2503.02198v2) — confirms+tempers the GS-as-sim verdict.** NeRF (not 3DGS) photoreal gate-racing sim, zero-shot **95.8% / 10 cm but only at 1 m/s**; needs ~1500 mocap-posed real images; ~15 min NeRF on a 4090. Verdict: **a NeRF/GS buys NOTHING over Blender for a track we can already model — its only unique value is reconstructing the REAL competition appearance we cannot author.** ⇒ Build a 3DGS of real VQ2 imagery ONLY IF Training mode exposes it, to harden the detector. Do NOT rebuild control around their DAGGER/NPE (absolute-pose, case-A; 1 m/s; unvalidated at speed). Medium priority, gated on real-image access.

## Updated control fork (with evidence)
- **D1 RL/CTBR (the spine — keep).** Gate-relative generalist PPO already generalizes across layouts if trained with layout-DR; it is the lowest-risk, proven path.
- **D3a Hybrid AC-MPC (differentiable MPC in actor)** — best architectural fit (our PPO+CTBR+gate-relative+DiffAero ≈ their setup), deploy-light. Buys dynamics/OOD robustness, NOT track-online-adaptation, NOT estimator-noise robustness, NO hard contact constraint.
- **D3b Hybrid RL-prior + sampling MPPI** — gradient-free (arbitrary costs incl. gate-contact), re-plans per-load, but racing-speed UNPROVEN and parallel-rollout VRAM fights the detector on 8 GB.
- **D2 pixels→motor (Geles)** — trap under randomization; mine the analytic-mask + asym-critic only.

## HONEST DEFLATION (anti-jumpy)
The hybrid's headline appeal was "online per-load adaptation without retraining." Evidence says that's **overstated**: AC-MPC adapts *dynamics* not *tracks*; MPPI re-plans per-load but isn't shown at racing speed and competes for VRAM; and a **well-trained gate-relative generalist PPO already absorbs layout variation**. Neither hybrid addresses the *estimator-noise* binding risk. ⇒ The hybrid is a **Phase-2 robustness/constraint lever to spike, NOT a foundational commitment.** Keep RL as the spine; the highest-certainty wins remain: fix the symmetric critic, add perception-aware reward, add the IMO gap-filler.

## Round-4 GEM-HUNT
- Amos et al., "Differentiable MPC for End-to-End Planning and Control," NeurIPS 2018 — the iLQR-backprop tech (`mpc.pytorch`) if we build D3a.
- Sacks et al., "Deep Model Predictive Optimization (DMPO)," ICRA 2024 — RL *learns the MPPI update rule*, beats SOTA MPPI on a real quad; the D3b upgrade.
- TLIO (Liu et al., RA-L 2020) + VIMO (Nisar et al., RA-L 2019) — the IMO base architecture + thrust-as-position-factor model.
- Brossard et al., "Denoising IMU Gyroscopes with Deep Learning," RA-L 2020 — learned gyro denoising for our AHRS.
- SOUS VIDE (arXiv:2412.16346) — 3DGS-as-sim (the GS analog of FalconGym).

---

# Round 5 — BREADTH sweep (web): 6-DoF object pose / monocular depth+3D / RL-methods (2026-06-27 cont.)

## The convergence (independent scouts → same shortlist)
Three separate web-scouts (object-pose, depth/3D, RL-methods) independently pointed at the SAME high-certainty, mostly-low-risk upgrades to our existing stack:

**HIGH-CERTAINTY WINS (evidence-backed, additive to current stack):**
1. **Privileged asymmetric critic** — now QUADRUPLE+ confirmed (Swift, GT Sophy, Learning-to-Fly-in-Seconds, Geles all say symmetric critic = catastrophic). `algo=appo`, state_dim=36. Cheapest, highest-confidence fix; resolves the live inc8 symmetric-critic footgun.
2. **Small GRU (64–128) in the actor**, BPTT 16–32 — bridges the gate-occlusion gap (UZH names this exact fix for the "no gate in view → success collapses" failure). KF stays primary belief; <1 ms. Recurrent-PPO beats transformers on low-dim obs (POPGym).
3. **Camera-aware / perception-aware reward** — PAMPC projection-centering + anti-blur + **Student-Informed Teacher co-training** (2412.09149: reframes gate-4 as an *unrealizable-teacher* problem → camera-aware teacher, 5–8%→46%). Strong gate-4 lever.
4. **Velocity-weighted contact/gate-edge penalty** (Fuchs/GT Sophy) — fixes γ-discounted-progress under-braking; better gate-4 pass-rate than a flat crash penalty.
5. **Soft-collision → annealed curriculum** (2602.24030) — lift gate-4 reach without the S3 rw=1.0 cliff-collapse.
6. **More keypoints + uncertainty-weighted PnP + real EKF covariance** — go 4 corners → 8–12 keypoints (corners + mid-edges + inner/outer frame, PVNet-style voting) for occlusion/truncation robustness; per-keypoint covariance → weighted PnP → EKF R. Start with MonoRace's analytic σ²_pos = 0.02·d²/(N_c²·N_g); graduate to EPro-PnP.

**NEW LEVERS (genuinely new capability):**
7. **Offline recon-map via feed-forward 3D (THE new capability):** VGGT / MASt3R-SfM on recon-lap keyframes → **Sim(3)-align the point cloud to gate-PnP metric anchors** → frozen metric track map → cheap 2D-3D relocalization on hot laps. Builds the map everyone else surveys; **also solves gate-ordering** (order = geometry along the map). Runs offline on A100/H100 — fits the recon-lap/exploit-lap structure exactly.
8. **Onboard metric depth for the gaps:** Depth Anything V2-S / UniDepthV2-S (TensorRT, ~30 Hz on 8 GB), **gate-PnP-rescaled** → metric range/structure prior fused into the ESKF where we currently coast blind between gates. Keep PnP authoritative for the gate rim (thin-structure risk); use depth only for gap structure.

## NEW SHARP RISK: square-symmetry / planar-flip ambiguity
A 1.5 m **square** has (a) 90° rotational ambiguity about its normal (4 corner labelings) and (b) IPPE/`SOLVEPNP_IPPE_SQUARE` **always returns a two-fold planar flip** — **worst exactly at distance** (near-affine projection). Both leading 2026 racing papers (MonoRace, Drift-Corrected VIO 2512.20475) **discard PnP rotation and take attitude from the EKF/VIO** to dodge this. ⇒ Our future C2 estimator MUST: use **PnP position, attitude from AHRS**; resolve flip via **temporal + multi-gate (non-coplanar) constraints**; break corner-labeling via temporal tracking. (Aligns with the new AHRS scope.)

## VALIDATIONS / DE-RISKS
- **Corner-PnP core is VALIDATED by SOTA** — do NOT swap to FoundationPose/MegaPose onboard (RGB-D-native, ~50–66 ms/iter, 8 GB/RGB doesn't close). Reserve render-and-compare for **offline pseudo-labeling + reprojection-confidence oracle** (we have exact gate geometry + cluster GPU).
- **"Next gate" = known ordinal index + corner-prior association across ALL champions**, NOT visual discovery. Truly reactive map-free IBVS exists only at ~1.9 m/s. ⇒ map-free gate ordering at speed remains genuinely open; recon-map (#7) is our answer.
- **Per-load track adaptation: One-Net failed because static geometry leaves NO proprioceptive signature → can't be inferred, must be OBSERVED** (gate-relative obs) + shaped by **auto-curriculum** ("Environment as Policy" 2410.22308: 100% on 6 unseen tracks). A learned recon-lap *encoder* is justified only for per-load *dynamics*, not layout.
- **Pure vision-RL = 0%** (Xing 2403.12203) → must bootstrap: privileged/state teacher → student → RL fine-tune (~60/40).
- **DiffAero's own paper: differentiable/BPTT "struggled with racing"** → keep the teacher on PPO; use diff-sim gradients cautiously.

## Round-5 GEM-HUNT (next pulls, prioritized)
1. **Student-Informed Teacher Training** — arXiv:2412.09149 (ICLR 2025) — gate-4 as unrealizable-teacher; camera-aware co-training.
2. **VGGT** — arXiv:2503.11651 (CVPR'25 Best Paper) — the offline recon-map engine.
3. **Learning to Fly in Seconds** — arXiv:2311.13081 — asym-critic + curriculum-annealing recipe; indicts the symmetric-critic footgun.
4. **Bootstrapping RL with Imitation** (Xing) — arXiv:2403.12203 (CoRL 2024) — vision-RL=0% without IL bootstrap; corner-noise obs model.
5. **EPro-PnP** — arXiv:2203.13254 (CVPR'22) — differentiable probabilistic PnP → SE(3) covariance for the EKF.
6. **Drift-Corrected Monocular VIO + Perception-Aware Planning** — arXiv:2512.20475 (2026) — near-1:1; YOLOv8-Pose+IPPE_SQUARE, confirms discard-PnP-rotation + latency budget.
7. **Environment as Policy** — arXiv:2410.22308 — auto-curriculum for track-layout generalization.
Runners-up: MASt3R-SLAM (2412.12392), Depth Anything V2 (2406.09414), PVNet (1812.11788), Contrastive Scene Transfer (2309.09865), RMA (2107.04034), GT Sophy (Nature 2022), Fuchs (2008.07971).

---

# Round 6 — Gemini Deep Research scan: monocular VI state estimation (our BINDING risk) (2026-06-27)

Source: Gemini Deep Research report (300+ source scan) pasted by Fengyou. ⚠️ Gemini = RECALL tool, not truth tool — several citations look future-dated/confabulated ("authors withheld", "approximate reference"); VERIFY each via deep-dive before treating as load-bearing.

## Actionable estimator-design takeaways (directly feed the C2 estimator + new AHRS scope)
1. **AHRS (NEW scope) → use LEARNED attitude, not a hand-tuned classical ESKF.** Classical (Madgwick/Mahony/ESKF) fails when sustained 4–5 g linear accel masks gravity. **RIANN** (GRU, parameter-free, hardware-general) and **Brossard dilated-CNN gyro denoising → open-loop integration** both rival VIO-grade attitude. We have GT attitude in sim to train them. **MoE-Gyro** extends measurable rate ±450→±1500°/s by reconstructing clipped peaks. → AHRS becomes a learned module trained on diffaero/sim GT.
2. **Invariant / Equivariant filtering (IEKF / EqVIO).** State on the Lie group SE₂(3) → group-affine error dynamics → linearization **independent of the current state estimate** → exact Jacobians through >1000°/s flips → no divergence. EqVIO adds an SOT(3) landmark symmetry. **Learned IMU bias predicted externally** (seq2seq) preserves the group-affine symmetry → 46% ATE-RMSE improvement vs MSCKF. → principled robustness upgrade to RewindKF for the high-dynamics regime.
3. **ADR-VINS (arXiv:2603.02742)** — tightly-coupled ESKF that ingests **2D gate-corner pixels as innovations** (not a solved PnP pose); valid with **as few as 2 visible corners**; preserves geometric covariance; **dodges the planar flip ambiguity by construction**. RMS 0.134 m @ 20.9 m/s. → near-blueprint for our ESKF re-scope; pairs with our +L obs contract.
4. **Translation-only PnP + attitude-from-filter** — confirmed AGAIN (3rd independent source). Lock it: PnP→position, AHRS/filter→attitude.
5. **OOSM / latency** — D-GVIO uses a **Left-Invariant EKF** for *exact* backward-forward buffer re-propagation (state-independent transition matrix). → upgrade path for RewindKF (we already do horizon=0.5 OOSM; L-IEKF makes the rewind exact).
6. **Gap-fill LIO** — Cioffi (commanded-thrust TCN, OUR fit) < **AI-IO** (Transformer + DShot rotor telemetry, +51.2% ATE — but needs rotor speeds we DON'T have on the VQ2 wire) ; **DIVE** (IMU-only velocity). Keep Cioffi commanded-thrust variant.
7. **Motion-blur survival** (physical-drone item; sim is blur-free) — IMU-aided feature tracking (predict feature loc, search a narrow Mahalanobis ellipse) + **mathematical drift models** (artificial friction on velocity during blackouts; A2RL finalist 2602.01860, Saska lab). Direct/photometric frontends (SVO/DPVO) more blur-robust than KLT.

## Paywalled / closed-source EDGE domains identified (arXiv = no edge)
Defense/aerospace GNC (AIAA JGCD, IEEE TAES, SPIE Defense+Commercial Sensing, NATO STO, DTIC, ProQuest dissertations, patents): **terminal guidance / proportional navigation** ≈ our gate-approach; **bearings-only/angle-only nav** ≈ our camera-as-angle-seeker; high-g interceptor filtering; seeker-FoV-aware guidance ≈ keep-gate-in-view. The drone-racing CS community does NOT cite this → asymmetric moat. Next Gemini prompt targets it; Fengyou unblocks flagged sources. → [[index-strategy-meta]] (research-channel division: Gemini=breadth/recall incl. paywalled, Claude=depth/verify/design).

## Estimator-relevant pulls to verify (deep-dive candidates from Gemini's list)
ADR-VINS (2603.02742), EqVIO (2205.01980), Learned-IMU-bias-for-IEKF (2505.06748), Brossard gyro-denoise (2002.10718), AI-IO (verify ID), Vision-only A2RL-finalist artificial-friction (2602.01860), AirIMU (2310.04874), Deep-IMU-bias-factor-graphs (2211.04517).

---

# Round 8 — Gemini Deep Research scan: DEFENSE/AEROSPACE GNC (the asymmetric, paywalled, robotics-doesn't-cite edge) (2026-06-27)

Source: Gemini Deep Research (defense-GNC prompt) pasted by Fengyou. ⚠️ RECALL tool — verify each via deep-dive before load-bearing.

## THE REFRAME (the edge)
**Drone-through-a-sequence-of-gates ≡ strapdown-seeker missile bearings-only terminal intercept of a sequence of waypoints.** Unlocks decades of analytically-guaranteed, SOLVER-FREE guidance + high-g bearings-only filtering the robotics drone-racing community does NOT cite. Recasts VIO+MPC (iterative, latency-bound, covariance-collapse-prone) as terminal-guidance (closed-form, O(1), provably FOV-compliant).

## Top transferable artifacts
1. **Gap Traversal Guidance (Midhun & Ratnoo, JGCD 2022; 3D "Quadrotor Guidance for Window Traversal: A Bearings-Only Approach", JGCD 2025)** — CLOSED-FORM law flies through a gate using ONLY bearing angles of the gate vertices (elliptic shaping on the bisector); NO depth/PnP/VIO. 2025 3D version ≈ 1:1 our problem. → new control fork **D4 (guidance-law)**: deployed controller / RL warm-start prior / reward shaper / perceptual-desert fallback.
2. **Bearing-Angle / bounding-box observability (Ning et al., IJRR/arXiv 2024, OPEN ACCESS)** — treat the gate's apparent SIZE as a pseudo-angle measurement → RANGE observable WITHOUT a lateral maneuver (even with unknown size; we have KNOWN 1.5 m → stronger). ⇒ fly DIRECT/energy-efficient, not weaving for observability. Observability-theory foundation under our gate-PnP-from-known-size.
3. **Modified Polar Coordinates EKF (Aidala & Hammel, IEEE TAC 1983) + Hybrid-Coordinate EKF** — Cartesian EKF collapses covariance under high-g bearings-only; MPC state `[θ,θ̇,ψ,ψ̇cosθ,1/r,ṙ/r]` makes the measurement update LINEAR → no collapse, stable under high-g. → high-dynamics estimator hardening; pairs with IEKF thread (both kill linearization-induced divergence).
4. **FOV-constrained trajectory shaping (Hong et al., JGCD 2021 — tanh look-angle; Hirwani et al., JGCD 2022 — cubic polynomial)** — CLOSED-FORM, GUARANTEED keep-target-in-FOV (tanh natively bounds look-angle to ±σ_max), O(1), no solver. → the *guaranteed* version of our reward-shaped perception-aware flight; could be a hard FOV layer or guidance prior.
5. **Stochastic-projection tight fusion (Veth, AFIT/DTIC dissertation 2006, OPEN ACCESS)** — fuse gate corners as pixel-residual innovations into inertial error states, NO PnP. = ADR-VINS' direct-corner-innovation idea with 2006 aerospace roots (independent confirmation).
6. **Strapdown-seeker latency / parasitic loop (Wang et al., JGCD 2021)** — image-processing delay forms an unstable parasitic feedback loop; model delay (Padé) as a state, PIDN restores phase margin. → CONTROL-side treatment of the 30 Hz vision delay (complements our estimator-side RewindKF OOSM).
7. **Active-perception guidance (AOPN = PN + FIM-trace term; Battistini & Shima differential-game, TAES 2014)** — augment guidance with an observability-maximizing term. NOTE: #2 (known-size bounding-box observability) partially OBVIATES the observability-maneuver for us → prefer direct flight.
8. **Advanced filters (Kim et al., TAES 2012)** — RBPF (marginalize linear states, particles only on nonlinear range/aero-params) + bias-compensated pseudo-linear KF (MMSE at low cost) for joint state+unknown-aero estimation. Lower priority.

## Impact on the stack
- **New control fork D4 (bearings-only guidance law)** — lightweight, analytically guaranteed, no solver, no map; candidate as RL prior / fallback / reward source. CAVEAT (Reaching-the-Limit): a rigid reference law alone likely sub-time-optimal vs RL — use as PRIOR/fallback, not sole deployed controller.
- **Estimator:** MPC-EKF / Hybrid-Coordinate EKF as a high-g-stable alternative/complement to the invariant-filter (IEKF) thread; stochastic-projection confirms direct-corner-innovation (ADR-VINS) design.
- **Perception-aware:** closed-form tanh/polynomial FOV shaping = a guaranteed alternative to reward-shaped look-at; bearing-angle observability says known-size lets us fly direct.

## DEEP-DIVE SHORTLIST (defense-GNC)
OPEN ACCESS (grab free): Ning et al. Bearing-Angle (arXiv 2024); Veth stochastic-projection (DTIC 2006). PAYWALLED (Fengyou unblock, ranked): Midhun & Ratnoo "Window Traversal Bearings-Only" (JGCD 2025) [#1, ≈1:1]; Hong et al. "Trajectory Shaping FOV-Constrained" (JGCD 2021); Aidala & Hammel "Modified Polar Coords" (TAC 1983); Wang et al. "Strapdown-seeker time-delay" (JGCD 2021).

---

# Round 9 — Defense-GNC deep-dives (3) + open dissertations sweep (2026-06-27 cont.)

## ⚠️ PROVENANCE CORRECTION
The "Veth 2006" PDF is actually **Giebner, AFIT MS thesis 2003** (the AFIT *ancestor*, not Veth's stochastic-projection method). Giebner fuses 2 SCALAR ANGLES (az/el) to a single KNOWN ground target — NOT pixel-residuals, NOT unknown-depth features; its attitude-coupling block is DESIGNED BUT NEVER RUN. ⇒ To cite the real pixel-residual tight fusion, pull **Veth, "Fusion of Imaging and Inertial Sensors for Navigation," AFIT PhD 2006**. Do NOT cite the current file as the method.

## Bearing-Angle (Ning et al., arXiv 2024) — the high-value estimator/doctrine result
- **OBSERVABILITY: with a size/subtended-angle measurement, RANGE is observable WITHOUT a lateral maneuver** — the observer may fly STRAIGHT at the target. (Classical bearings-only needs an orthogonal maneuver; Theorem 1 removes that.) For us with **KNOWN size (1.5 m): range is INSTANT closed-form r = 0.75/tan(θ/2)** per frame, no maneuver, no convergence horizon — the "augment-and-converge" problem collapses to a direct measurement.
- **The subtended ANGLE is attitude-INVARIANT** (unlike raw box size) → the size→range channel is DECOUPLED from boresight/mount-rotation bias → **robust to exactly the ε_vert boresight error we fight on the vertical channel.**
- **TRANSFER:** (a) theoretical LICENSE for "fly direct, don't weave" (speed > observability-maneuver) — kills the weave-for-observability worry; (b) add a **subtended-angle range pseudo-measurement** into the gate-relative ESKF as a cheap redundant/robust range channel (esp. when a PnP corner is occluded); single-frame cold-start range (helps the cold-start/range-consistency footgun).
- **CAVEATS:** their box-size pipeline assumes a FILLED blob — our gate is a HOLLOW frame → take span from **PnP corners, not a YOLO box**; small-angle θ≈ℓ/r breaks at **r < ~4.5 m** → use exact arctan near the gate; known-size means **size-error maps 1:1 to range-error** (corner precision becomes binding); scale R with range (noise is range-coupled, worse up close); validated only ≤3 m/s.

## Window-Traversal (Midhun & Ratnoo, arXiv:2410.14367 2024 — NOT a JGCD-2025 extended ver.) — refines D4
- Closed-form γ_des, χ_des from the **4 window-vertex bearings** (angular bisector + elliptic shaping S_γ,S_χ) → Lyapunov-converge to the window-NORMAL through the centroid. **No depth/PnP/range**, O(1). Needs the 4 vertices visible + correctly ORDERED (E1..E4).
- **VALIDATED ONLY 0.1–1 m/s, bearings from MOCAP (not onboard vision), no FOV guarantee (explicit future work), asymptotic (not finite-time/time-optimal), single-window (no sequencing).** Miss ~1.4–6.5 cm at σ≤7° bearing noise.
- **D4 VERDICT (refined):** NOT a standalone 20–30 m/s racer. Best uses, ranked: **(c) REWARD SHAPER** — its Lyapunov potential **W = ½(D_x²+D_z²)** (lateral+vertical offset from the centroid-normal line, in gate-relative coords) is a drop-in potential-based shaping reward that **directly attacks gate-4** (off-line centroid-tracking); **(b) RL WARM-START / RESIDUAL base** — analytic correct expert maps {4 vertex bearings}→{γ,χ}; policy outputs a DELTA on top → small exploration burden + safe init (prevents the inc8 success≡0 policy-gap); **(d) perceptual-desert traversal SHIELD** when KF confidence collapses but vertices remain. Gem: **Sharma & Ratnoo, "Bearings-only trajectory shaping with LOOK-ANGLE CONSTRAINT," IEEE T-AES 2019** = the FOV-constrained variant this paper lacks (pull next).

## Vision-aided AHRS (from the Giebner attitude-Jacobian structure)
With 4–8 gate corners (vs Giebner's 2 angles to 1 target) the **attitude/tilt block becomes observable** → gate-corner PIXEL innovations can CORRECT attitude error states, i.e. vision tightens the IMU-only AHRS exactly when a gate is in view. Seed: quantify how many corners / what spread makes δθ observable. (= the ADR-VINS tight-corner-fusion idea, with a route to also fixing attitude, not just position.)

## Open dissertations sweep (mostly FULLER versions of papers we've read — confirmatory)
TOP OPEN reads (download): **Loquercio 2021** (agile autonomy + sim-to-real appearance-gap, ✅ rpg PDF); **Tal 2022 MIT/Karaman** (INDI tracks aggressive trajectories WITHOUT an accurate model — directly our CTBR-on-imperfect-model, ✅); **Murali 2024 MIT/Karaman** (perception-aware planning / keep features in FoV, ✅); **Shen 2014 GRASP** (mono+IMU GPS-denied systems backbone, ✅). BROWSER-OPEN (ZORA bot-blocked): **Song 2024 UZH** (PPO/CTBR/gate-relative + OC-vs-RL — our RL depth); **Kaufmann 2022 UZH** (Swift lineage, sim-to-real via gate-detector abstraction). **AirIO/AirIMU (CMU, ✅ papers+code)** = the open AHRS-from-IMU match (body-frame IMU beats global 66.7%, learned uncertainty into EKF). Restricted-license but on-point: **Shuo Li 2020 TU Delft** (snake-gate+P3P+Visual-MPC-localization+G&CNet — closest whole-pipeline ancestor of MonoRace). **PAYWALLED to UNBLOCK:** **Svacha (GRASP 2019)** "IMU-Based State Estimation Exploiting Aerodynamic Effects" = highest-value AHRS-from-IMU-at-high-g unblock (the derivation behind MonoRace's estimator); **SkyDreamer** (TU Delft MSc 2025, embargoed→arXiv 2510.14783) end-to-end vision-RL world-model, no extrinsic calib, 21 m/s. Hard-problem coverage: no-map gate ordering = STILL a gap (no open thesis; closest ARPL HUNT arXiv 2509.19452 instantaneous relative frames — pull).

---

# Round 10 — Gemini scans: Chinese-language GNC + high-dynamics estimation GAPS (2026-06-27) — CLOSES RESEARCH BREADTH

Source: two Gemini Deep Research scans pasted by Fengyou. ⚠️ RECALL — Chinese/aerospace claims + RMSE deltas unverified; treat as DESIGN CANDIDATES to validate in spikes, not proven. Both scans CONVERGE on a richer estimator than our ESKF.

## THE CONVERGENT ESTIMATOR DESIGN (binding-risk core, both scans agree)
A high-dynamics gate-relative estimator that beats a vanilla ESKF, assembled from:
1. **Error-state + DIRECT REPROJECTION INNOVATION (ADR-VINS)** — raw gate-corner pixels as innovations, valid update from as few as 1–2 corners, no PnP pseudo-pose. (Already in our plan.)
2. **Multiple-Fading-Factor STRONG TRACKING (MSTF/STCKF)** — adaptive λ_k≥1 inflates prediction covariance the instant the innovation grows → forces the Kalman gain to trust the camera over the lagging aero model when a gust/maneuver hits. **Per-state (diagonal) fading** so a velocity spike doesn't trash the clean gyro-bias axes. 🚩 **BOTH scans flag Strong-Tracking fading-factor filters as ABSENT from Western drone racing** (which uses static process noise + RL) = a real asymmetric estimator edge. UNSOLVED trap: classical STF can't tell a real maneuver from a false visual fix (both = innovation spike) → fix with **Variational-Bayesian STF (VB-STF)** (joint noise estimation + Mahalanobis test decouples process-mismatch from measurement-outlier).
3. **CUBATURE (positive-weight, mixed-degree)** over UKF/EKF — spherical-radial rule guarantees positive weights (no negative-weight covariance collapse in 15-DoF); mixed-degree runs 3rd-degree cheap in smooth flight, escalates to 5th in turbulence; Robust-CKF (Huber M-estimation) for outliers.
4. **SQUARE-ROOT / UD factorization** — propagate the Cholesky factor → guaranteed positive-definite covariance under **32-bit single-precision embedded** math (prevents catastrophic-cancellation collapse). **sqrtVINS** = 2× faster than MSCKF on Jetson-Nano-class hardware; structure-aware LLT to avoid fill-in.
5. **OBSERVABILITY CONSISTENCY: FEJ2 / OC-VINS / ROBOCENTRIC VIO** — monocular VINS has 4 unobservable DoF (global xyz + yaw); a standard EKF re-linearizes at the noisy newest estimate → "hallucinates" observability → overconfident covariance → divergence. **FEJ2** compensates the first-estimate truncation; **Robocentric VIO** reformulates wrt a moving local frame → no global-yaw mismatch + safe init from arbitrary pose. 🚩 **Robocentric ≈ our gate-relative pivot** — they're the same idea; FEJ2/robocentric gives 32–47% drift reduction vs MSCKF. HIGH-value, directly our binding risk.
6. **(if bearings-only/D4) Tracking-Differentiator (Han ADRC) / Super-Twisting ESO** — extract a lag-free LOS-rate derivative from noisy 30 Hz pixels WITHOUT the phase lag of low-pass filtering; implicit-Euler discretization kills chattering; fixed-time fractional ESO for sub-second gate approaches.

⇒ Gemini's "optimal" pick = **MSTMCKF (Multiple-Fading-Factor Strong-Tracking Mixed-Degree Cubature KF) in error-state + direct reprojection.** Plausible but unproven for us — a design candidate to A/B vs our RewindKF, not a mandate.

## NEW DOCTRINE PIECES (beyond the estimator)
- **Guaranteed FoV via Barrier Lyapunov Function (BLF)** — energy fn →∞ at the camera FoV boundary → control law NATIVELY prevents the gate leaving frame (a hard inequality constraint, deterministic — vs RL's implicit reward-shaped FoV-keeping). Complements the defense-GNC tanh/polynomial shaping. ABSENT from Western drone racing.
- **TERMINAL BLIND-ZONE coast (phased guidance 分段制导, 盲区段)** — in the final fraction-second the 1.5 m gate EXPANDS beyond the FoV → switch from vision-closed-loop to **inertial / belief-state propagation** to coast through the aperture on the velocity+attitude established during the constrained approach. Directly our terminal problem; **pairs with the GRU actor**; ZJU RNN belief-state (tilted-gap blind-zone traversal, arXiv 2024) independently confirms recurrent belief survives the blind zone + auto-rolls to align the longest axis.
- **IMM (Interacting Multiple Model)** for CV/CA/CT regime switching — peak theory but **compute-hostile** (parallel 15-DoF CKF banks; +92–148% compute; mode-switch latency). VERDICT: SKIP for embedded; the adaptive single-filter (MSTF) gets most of the benefit. Promising-if-revisited: GRU feeds maneuver likelihood into mixing weights; Maximum-Correntropy IMM for blur-robustness.

## RESEARCH BREADTH = COMPLETE
Domains swept: drone-racing canon · VIO/SLAM · perception-aware control (PAMPC/MPPI/AC-MPC) · Gaussian-splat · detectors · RL methods (memory/distillation/adaptation) · 6-DoF object pose · monocular metric depth + feed-forward 3D · defense/aerospace GNC (bearings-only/terminal guidance) · patents · PhD dissertations · Chinese-language GNC · high-dynamics nonlinear estimation. ⇒ Remaining value is in SPIKES + VQ2-load checks, NOT more reading.

## Top estimator papers IF deep-diving (else implement from the design above)
FEJ2 (Chen/Yang/Huang, ICRA 2022) · sqrtVINS (ICRA 2025) · Huang/Mourikis/Roumeliotis FEJ (ISER 2009) + OC-rules (IJRR 2010) · Li & Mourikis robocentric high-precision EKF-VIO (IJRR 2013) · a VB Strong-Tracking robust KF (DSP 2026) · Implicit-Euler super-twisting (IEEE 2019). Plus pull the REAL Veth AFIT PhD 2006 for stochastic-projection.
