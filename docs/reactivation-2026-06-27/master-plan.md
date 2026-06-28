# Peregrine VQ2 — Master Plan & Fork Map (2026-06-27)

*The whole decision tree in one place: where we started this morning, every fork today's research opened, and the spikes that resolve them. Status tags: ✅ CERTAIN WIN (build regardless) · ⭐ LEADING (best current bet) · 🔵 ALIVE (kept open, a cheap test decides) · ❌ RULED OUT (with evidence, reversible). Doors-open rule: predictions don't close forks — cheap tests do.*

---

## 0. Where we started this morning — the ORIGINAL VQ2 plan
A single-track pipeline (the inc8 / C2-estimator state, pre-research):
- **Sense:** mono 30 Hz JPEG + raw IMU. (VQ2 spec blocks position/attitude/odometry/gate-info.)
- **Perceive:** YOLO-pose 8-keypoint gate detector → PnP → gate-relative pose.
- **Estimate:** gate-relative ESKF/RewindKF (C2 chain, merged, gated off) — case-C self-localizing.
- **Control:** inc8 PPO RL policy → CTBR; the planned next lever = RL to lift gate-4 reach/pass-rate.
- **Posture:** NO map, reactive (detect next gate, fly through it), gate-relative.
- **Newly-opened gap (the spec itself):** must now SELF-ESTIMATE ATTITUDE (AHRS) — the old stack trusted given attitude.
- **Nagging unknowns:** gate-4 reach / σ_p0 / ε_vert boresight bias; the symmetric-critic question (then unresolved); how to handle a possibly per-load-randomized track.

That was the whole plan: **detector → gate-relative estimator → RL policy, one path, several unknowns.** Research kept that spine and turned most layers into forks.

---

## 1. The stack today — layer by layer, every fork

### L1 — Attitude / AHRS  *(NEW layer — the spec opened it)*
Original: trusted given attitude (now blocked).
- ⭐ **Learned AHRS** — RIANN (GRU) or Brossard gyro-denoise, trained on sim GT; mag = yaw anchor.
- 🔵 Classical ESKF / Madgwick / Mahony — simple, but degrades when 4–5 g masks gravity.
- 🔵 Invariant (IEKF/EqVIO) or Modified-Polar EKF — high-g, no covariance collapse.
- 🔵 **Vision-aided AHRS** — gate-corner pixel innovations correct *tilt* when a gate is in view (free attitude fix).
- **Resolver:** load-check C2 (mag fidelity), C8 (IMU rate/saturation).

### L2 — Perception (detector + pose)
Original: YOLO-pose 8-kpt → PnP.
- Detector: ⭐ keep YOLO-pose · 🔵 + Swift-YOLO small-target levers (shared-DCNv2 head, SALD, BFusion) for distant gates · 🔵 4→8–12 keypoints (PVNet voting) for occlusion robustness.
- Pose: ✅ **DISCARD PnP rotation** (square-flip ambiguity) → attitude from filter · ⭐ IPPE_SQUARE · 🔵 per-keypoint covariance → real EKF R (50×-perturb → EPro-PnP) · 🔵 **bearing-angle known-size range channel** (attitude-invariant → ε_vert-robust; fly direct, no weave) · 🔵 render-and-compare OFFLINE pseudo-labeling (never onboard).
- 🔵 Detection filters (distance 1–13 m / skew>2 reject / occlusion).
- **Resolver:** C6 (appearance gap), C9 (boresight).

### L3 — Estimator  *(the binding VQ2 risk)*
Original: gate-relative ESKF/RewindKF. The convergent high-dynamics upgrade menu — A/B each vs RewindKF, keep ALL alive:
- ⭐ **Direct reprojection innovation (ADR-VINS)** — 1–2 corners, no PnP pose, dodges flip.
- 🔵 **Multiple-fading-factor Strong Tracking (MSTF)** + VB-STF (maneuver-vs-outlier) — the "absent from Western racing" edge.
- 🔵 Cubature (positive-weight, mixed-degree).
- 🔵 Square-root / UD factorization (32-bit embedded stability).
- ⭐ **FEJ2 / Robocentric** observability consistency — robocentric ≈ our gate-relative pivot; kills the hallucinated-observability → overconfidence → divergence failure.
- 🔵 Invariant (IEKF/EqVIO) + learned IMU bias.
- ⭐ **Gap-filler: Cioffi commanded-thrust LIO TCN** (we know our own commanded thrust).
- 🔵 Onboard metric depth (Depth Anything V2-S, gate-PnP-rescaled) for gap structure.
- ⭐ **Terminal blind-zone coast** (vision→belief-state through the final fraction-second; pairs with the GRU actor).
- **Resolver:** C7 (gap frequency), C8 (IMU), C10 (clock/OOSM); the estimator A/B.

### L3.5 — Map  *(NEW layer — the biggest strategic fork)*
Original: NO map (reactive; gate_mapper demoted to training/dev-only).
- 🔵 **A1 Reactive mapless** — the original; robust fallback.
- ⭐ **A2 Online recon-lap map → exploit hot laps (THE MOAT):** recon lap builds a metric map (VGGT dense OR DPVO sparse) → Sim(3)-anchor to gate-PnP → relocalization + gate ordering. *Legal as ONLINE recon, not a pre-shipped map.* Turns no-map → known-map; dissolves gate-ordering + desert-localization.
- 🔵 A3 Learned VO structure between gates.
- **Resolver:** C3 (track-fixed-within-load) + C4 (randomization) + the 2-load hash; Spike C (DPVO-first → VGGT).

### L4 — Control (executed controller)
Original: inc8 PPO RL → CTBR, reactive, single policy.
- ⭐ **D1 RL/CTBR** — the spine, keep.
- 🔵 **D3a Hybrid AC-MPC** (differentiable MPC in actor) — dynamics/OOD robustness, deploy-light (1 iLQR solve).
- 🔵 **D3b Hybrid RL-prior + MPPI** (sampling) — hard contact constraint, online replan; VRAM/latency risk.
- 🔵 **D4 Bearings-only Gap-Traversal guidance** — as reward-shaper (Lyapunov W-potential), RL warm-start, or occlusion fallback; NOT standalone.
- ❌ D2 pixels→motor (Geles) — trap under randomization; mine only the analytic-mask + asym-critic.
- *(OC/TOGT offline for the lap-time bound only — never the deployed controller.)*
- **Resolver:** Spike A (gate-4), Spike B (D3a/D3b vs D1).

### L4-recipe — RL training levers  *(inside the policy — additive, not exclusive)*
- ✅ **Asymmetric privileged critic** — THE #1 fix (6 sources; our live footgun).
- ✅ Perception-aware reward (PAMPC image-plane centering + anti-blur).
- ✅ Velocity-weighted contact penalty (gate-4 under-braking).
- ✅ Curriculum reward-annealing.
- ⭐ Bootstrapping: state-teacher → DAgger student → adaptive RL fine-tune (pure vision-RL = 0%).
- ⭐ Small GRU actor (occlusion memory).
- 🔵 SITT camera-aware co-training (gate-4) — pending Spike A.
- 🔵 **Horizontal-FoV-aware approach shaping** (NEW idea — attack reachability via yaw-to-gate without touching the fixed 20° mount).
- 🔵 Layout domain-randomization generalist.

### Fork E — Generalization (a per-load-randomized track)
- ⭐ E1 Layout-DR generalist (observe the track, don't infer it).
- 🔵 E2 Recon-lap adaptation (One-Net warns naive conditioning fails).
- ❌ E3 Per-load retrain (infeasible in-competition).
- *(MPPI online replanning could dissolve this fork entirely.)*
- **Resolver:** C4.

---

## 2. NEW capability ideas surfaced today (cross-cutting)
- **Track-wide perception-reachability map** — for every gate, when is it in-frame along the racing line (pitch/roll + fixed 20° mount + FoV)? Sizes the blind-coast, shapes reward.
- **Horizontal-FoV approach shaping** — 90° HFoV ≫ 58.7° VFoV; prefer yaw-to-gate where geometry allows.
- **GS/NeRF of the training track as an OFFLINE photoreal sim** (FalconGym/SOUS VIDE) — appearance-hardening only, never onboard.

---

## 3. The spikes — how we resolve the ALIVE forks (cheapest-first)
- **Spike A — Gate-4 diagnosis:** ① kinematic frustum overlay (15 min, no training — is the gate even in-frame during the 12–28 m fix-seating window?) → ② emulator counterfactuals (force-accept = realizability, bias-off = calibration, reach-only = reach) → ③ SITT proxy-KL (only if realizability). *Resolves: SITT fork, calibration-vs-reach, the horizontal-FoV idea.*
- **Spike B — Hybrid control at speed:** ① B1 plant-mismatch (MPPI's home turf) → ② B2 estimator-noise; independent 8 GB VRAM/latency kill-switch. *Resolves: D3a/D3b vs D1.*
- **Spike C — Recon-map viability:** ① DPVO-first (sparse, 2.5 GB, edge/corner tracking) → ② VGGT (dense, offline A100). *Resolves: the A2 moat — dense vs sparse vs dead.*
- **VQ2 load-checks C0–C10 (+ the 2-load hash, run FIRST):** resolve the moat (C3/C4), AHRS (C2/C8), distillation/calib + legality (C5), detector (C6), gap-filler (C7), gate-4 calib (C9), OOSM/ordering (C10).

---

## 4. RULED OUT (evidence-backed, reversible)
- Pixels-no-estimator as a *standalone* (track memorized in weights → trap under randomization).
- FoundationPose/MegaPose onboard (RGB-D-native, too slow on 8 GB).
- GS/NeRF onboard relocalization (exceeds every validated envelope → offline sim tool only).
- Optimal-line MPC tracking as the deployed controller (infeasible-reference failure; OC offline for the bound only).
- IMM filter banks onboard (compute-hostile; MSTF single-filter gets most of it).
- Pre-shipped track map (circumvention → DQ; the ONLINE recon-map is fine).

---

## 5. One-paragraph summary
We began with a single track — detector → gate-relative estimator → RL policy, no map. Today's research kept that spine but: **(a)** handed us a *certain-win RL recipe* (asymmetric critic + perception reward + velocity-contact penalty + curriculum + bootstrapping + GRU actor); **(b)** opened a *map fork* — the online recon-map moat — that could convert no-map into known-map; **(c)** gave the estimator a *high-dynamics upgrade menu* (Strong-Tracking + robocentric/FEJ2 + cubature + square-root + Cioffi gap-filler + blind-zone coast); **(d)** surfaced two *hybrid-control* experiments (AC-MPC, MPPI) and a *bearings-only guidance* reward-shaper; and **(e)** reframed gate-4 as possibly a *camera-reachability geometry* problem with a 15-minute test. Every speculative fork stays open behind a cheap spike; nothing is closed on prediction.
