# Peregrine VQ2 — Candidate Stack v2 (synthesis of 24 papers + 2 Gemini scans, 2026-06-27)

*Design synthesis, not a build commitment. Confidence tags: ◆◆◆ overwhelming / ◆◆ strong / ◆ promising-unproven. Forks stay open until a spike kills them.*

## The single biggest strategic insight
**The multi-lap-within-a-load rule converts our hardest constraint — "no-map monocular racing" (UNSOLVED in the published record) — into "build-then-exploit-known-map racing" (SOLVED).** Recon lap → VGGT feed-forward 3D → Sim(3)-anchor to gate-PnP metric fixes → frozen metric track map. That map simultaneously dissolves THREE open problems: (a) metric localization between gates, (b) gate ordering (= geometry along the recon trajectory), (c) it puts us in the known-map regime where the proven Swift / Drift-Corrected-VIO estimators work. **Every published champion either surveys a map offline or carries a VIO box; we build the map online from the recon lap — that's the moat.** (Gated on: track fixed within a load — verify on VQ2 load.)

## The most-confirmed finding in the entire corpus
**Privileged ASYMMETRIC critic — confirmed by SIX independent sources** (Swift, GT Sophy, Learning-to-Fly-in-Seconds, Geles pixels-no-estimation, Xing bootstrapping, + our own SSOT). Symmetric critic = catastrophic (Geles 0%; LtFiS "most policies crashing at 3M steps"). **Our live inc8 footgun (symmetric 20-dim critic, the privileged 36-dim never wired) is THE highest-confidence, on-critical-path fix.** `algo=appo` + `state_dim=36`; put deploy-unobservable signals (true pose/attitude/velocity, episode-constant disturbances) in the critic only.

---

## The stack (layered)

**L0 — Sensing:** mono 30 Hz JPEG + raw IMU (accel/gyro/mag) + TIMESYNC.

**L1 — AHRS (NEW scope) ◆◆:** learned attitude from IMU — RIANN (GRU) or Brossard dilated-CNN gyro-denoise → open-loop integration, trained on sim GT attitude; mag = yaw anchor. Don't hand-tune a classical ESKF (fails when 4–5 g masks gravity).

**L2 — Perception ◆◆◆:** YOLO-pose detector (Swift-YOLO small-target levers for distant gates) → **8–12 keypoints** (corners + mid-edges + inner/outer frame, PVNet-style voting for occlusion robustness) → **IPPE_SQUARE PnP, DISCARD rotation, keep translation** (planar-flip ambiguity — confirmed by 3 sources). Per-keypoint covariance → weighted PnP → EKF R (50×corner-perturbation recipe or EPro-PnP). Detection filters: distance 1–13 m, skew max(w/h,h/w)>2 reject, occlusion. Square symmetry resolved by temporal + multi-gate non-coplanar fusion.

**L3 — Estimator (gate-relative, the binding risk) ◆◆:** ESKF/RewindKF, with a convergent high-dynamics upgrade menu (both Gemini GNC scans agree — A/B vs RewindKF, don't adopt blind):
- **Direct reprojection innovation (ADR-VINS)** — raw gate corners as innovations, valid from 1–2 corners, no PnP pseudo-pose, dodges the flip.
- **Multiple-Fading-Factor Strong Tracking (MSTF)** — per-state adaptive covariance inflation when innovation spikes → trusts the camera over the lagging aero model during gusts/maneuvers (ABSENT from Western drone racing = edge); use **VB-STF** to separate a real maneuver from a false visual fix.
- **Cubature (positive-weight, mixed-degree)** over UKF/EKF for high-dim nonlinearity at >1000°/s.
- **Square-root / UD factorization** (sqrtVINS) for 32-bit embedded numerical stability.
- **Observability consistency: FEJ2 / ROBOCENTRIC VIO** — robocentric ≈ our gate-relative pivot; kills the EKF "hallucinated observability → overconfidence → divergence" failure (32–47% drift cut).
- Alternatives still live: invariant (IEKF/EqVIO) + learned IMU bias; Modified-Polar EKF (high-g, no Cartesian collapse). IMU-saturation model-substitution; OOSM rewind (L-IEKF exact).
- Gap-filler: **Cioffi commanded-thrust LIO TCN**. Bearing-angle known-size **redundant range channel** (attitude-invariant → ε_vert-robust). Optional onboard metric depth (Depth Anything V2-S, gate-PnP-rescaled) for gap structure — keep PnP authoritative for the thin gate rim.
- **Terminal blind-zone coast:** in the final fraction-second the gate expands beyond FoV → switch vision→belief-state/inertial propagation (pairs with the GRU actor) to coast the aperture. ⇒ lit-review Round 10.

**L3.5 — Map / the moat (offline) ◆:** recon lap → VGGT (chunked via VGGT-Long) → Sim(3)-anchor to gate-PnP → frozen metric map → gate ordering from trajectory → hot-lap 2D-3D PnP relocalization. VGGT runs offline on A100/H100 only (8 GB can't). VGGT also auto-labels gate poses for detector training. Validate VGGT on sim renders first (sim-domain gap untested).

**L4 — Control (RL/CTBR spine) ◆◆◆ recipe:**
- Privileged asymmetric critic (above).
- **Bootstrapping (Xing): state teacher (inc7/inc8) → DAgger vision student → adaptive RL fine-tune** with performance-gated LR/clip. Pure vision-RL = 0%; DAgger-then-adaptive-RL = 76–85%. ~60/40 IL/RL, history H≈32.
- Reward: progress + **perception-aware** (PAMPC projection-centering + anti-blur, OR SITT emergent pointing) + **velocity-weighted contact penalty** (Fuchs/GT Sophy, for gate-4 under-braking) + gate-pass + crash.
- **Curriculum: lenient→strict reward-weight annealing** + soft-collision→annealed (gate-4 reach without the rw=1.0 cliff).
- **Small GRU actor (64–128)** for gate-occlusion memory.
- Gate-relative obs (no map) + confidence channels (richer than Xing's (−1,−1) sentinel) + 6-D continuous rotation rep.
- Generalist via **layout domain-randomization** (Environment-as-Policy auto-curriculum) — observe the track, don't infer it (One-Net's failure: static geometry leaves no proprioceptive signature).

**L4-alt — Hybrid control (Phase-2 spike) ◆:** RL-prior + MPPI (hard contact constraint, online per-load replan) or AC-MPC (dynamics-robustness, deploy-light). Parked pending the noisy-state-at-racing-speed spike.

**L4-D4 — Bearings-only guidance law (defense-GNC, new fork) ◆ [refined R9]:** Window-Traversal Guidance (closed-form γ,χ from the 4 gate-vertex bearings, no depth/PnP, O(1)) — validated only ≤1 m/s w/ mocap bearings, so NOT a standalone racer. **Best uses (ranked):** (c) **REWARD SHAPER** — its Lyapunov potential W = ½(D_x²+D_z²) (offset from the centroid-normal line) is a drop-in shaping reward that directly attacks gate-4; (b) **RL warm-start / RESIDUAL base** (policy outputs a delta on the analytic γ,χ → small exploration, safe init, prevents the success≡0 policy-gap); (d) **perceptual-desert traversal shield**. FOV-constrained variant to pull: Sharma & Ratnoo T-AES 2019. → lit-review Round 9.

**Estimator companions from defense-GNC (◆, fold into L3):** **bearing-angle known-size observability** → range observable flying STRAIGHT at the gate (no weave); with known 1.5 m, **range = 0.75/tan(θ/2) instant**, and the subtended angle is **attitude-INVARIANT → range channel robust to ε_vert boresight bias** (add as a redundant ESKF range pseudo-measurement; use PnP corner span not a YOLO box; exact arctan at r<4.5 m). **Modified-Polar / Hybrid-Coordinate EKF** (no high-g covariance collapse). **Vision-aided AHRS:** with 4–8 gate corners the attitude block is observable → corner pixel-innovations can CORRECT tilt (tighten IMU-only AHRS when a gate is in view; = ADR-VINS tight fusion, route to fix attitude not just position). Method citation = Veth AFIT PhD 2006 (NOT the Giebner-2003 file). → lit-review Round 9.

**Offline tooling:** GS/NeRF of training track for appearance hardening (FalconGym/SOUS VIDE) IF VQ2 exposes real imagery; everything heavy (VGGT map, teacher training, AHRS/LIO training, gate-pose labeling) runs offline on Adroit/ShadowPC/RunPod; onboard runs lean inference only.

---

## Action ladder (when we resume BUILDING — by confidence × leverage)
**Tier 1 — certain, do first (all inside the existing RL stack):**
1. Wire the privileged asymmetric critic (`algo=appo`, state_dim=36). [6× confirmed]
2. Perception-aware reward (PAMPC image-plane centering + anti-blur term).
3. Velocity-weighted contact penalty (gate-4 under-braking).
4. Curriculum reward-weight annealing (lenient→strict).

**Tier 2 — high-confidence:**
5. Bootstrapping for the VQ2 vision policy (teacher→DAgger→adaptive-RL).
6. Small GRU actor (occlusion memory).
7. Learned AHRS module (trained on sim GT).
8. Multi-keypoint detector + real PnP covariance + discard-rotation.

**Tier 3 — new capability, scoped:**
9. Recon-map (VGGT + gate-PnP Sim(3) anchor) → relocalization + gate ordering.
10. Cioffi LIO gap-filler in the ESKF.
11. Onboard metric depth for gap structure.

**Tier 4 — spikes/forks (uncertain, gated):**
12. Hybrid RL+MPPI (after the spike).
13. SITT for gate-4 (after the realizability pre-check).
14. Invariant (IEKF) estimator upgrade.

## Spikes that gate the uncertain forks
- **Gate-4 realizability pre-check** (SITT proxy-student KL-spike oracle): is gate-4 a FoV/realizability problem (→ SITT) or a boresight/vertical-σ calibration problem (→ ε_vert calibration, NOT SITT)? Our own memory leans calibration — settle empirically before investing in SITT.
- **MPPI-at-noisy-state-at-racing-speed**: inject measured σ_p0≈0.15 into the policy, compare RL-only vs RL+MPPI on gate-4 pass/contact at 25–30 m/s (the regime no paper tested).
- **VGGT-on-sim-renders validation**: does the feed-forward geometry hold on our photoreal renders?
- **VQ2-load checks**: mag fidelity; track fixed within a load; Training-mode GT exposure; occlusion-gap frequency.

## What we DELIBERATELY ruled out (with evidence)
- Pixels→control with NO estimator (Geles): works only because the track is memorized in weights; trap under per-load randomization. Mine the analytic-mask + asym-critic, keep the estimator.
- FoundationPose/MegaPose onboard: RGB-D-native, too slow at 8 GB/RGB. Reserve render-and-compare for offline pseudo-labeling.
- GS/NeRF onboard hot-lap relocalization: exceeds every validated envelope (30 Hz + 8 GB + blur). GS is an offline sim tool only.
- Optimal-line MPC tracking as the deployed controller (Reaching-the-Limit): infeasible-reference failure; OC offline for the bound only.
- Transformers/in-context-RL onboard, event cameras (no sensor), full SLAM/loop-closure (single-lap — but multi-lap revives map-reuse).
