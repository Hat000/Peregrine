---
name: reference-prior-art
description: "Survey of autonomous drone racing prior art and libraries (Agilicious, Flightmare, gym-pybullet-drones, Swift, time-optimal, CasADi/acados) with license + reuse verdicts."
metadata: 
  node_type: memory
  type: reference
  originSessionId: 125b63a1-bf43-422b-b637-98394ed461bc
---

Surveyed 2026-05-28. Meta-finding: **no reusable racing 'brain' exists** — best systems are academic-licensed or closed. Reuse the building blocks, replicate the methods.

| Project | License | Lang | Maintained | Verdict |
|---|---|---|---|---|
| Agilicious (UZH) | Academic-only | C++/ROS | Active | Learn from (papers), can't reuse code. Best full-pipeline algorithmic reference. |
| Swift (Nature 2023) | Closed | — | — | Approach reference only. Pseudocode on Zenodo (10.5281/zenodo.7955278). RL: camera+IMU → gate-detect NN → control NN. |
| Flightmare (UZH) | MIT | C++/Python | Stale (2020) | Surrogate-sim candidate but may not build in 2026; Unity dep heavy. |
| gym-pybullet-drones | MIT | Python | Active 2026 | Usable RL prototyping (Gymnasium+SB3). Hover-focused, weak vision/aero, not racing-tuned. |
| rpg_time_optimal (UZH) | **GPL-3.0** | Python | Static | Time-optimal waypoint trajectories (CasADi). Closest to RACE planner. GPL → prefer reimplementing the method from the Science Robotics 2021 paper to avoid entanglement. |
| CasADi | LGPL | Py/C++ | Active | USE. Trajectory opt + MPC foundation. Context7: /casadi/casadi |
| acados | BSD | Py/C | Active | USE for real-time NMPC tracking. Context7: /acados/acados |
| L4CasADi | MIT | Python | Active | PyTorch-learned models inside CasADi MPC. Stretch (learned-dynamics MPC). |
| **Elodin AI-GP harness** | **Apache-2.0** | Rust+Py | New 2026 | **USE as dev/surrogate/estimator-validation rig.** Betaflight SITL, 1 kHz lockstep deterministic, camera matches official spec exactly, runs WSL/Adroit. NOT the scoring sim (Betaflight-UDP/ENU/simple-aero; need MAVLink shim). Details in [[reference-competition-materials]]. |

**Library choices locked-ish:** CasADi + acados (plan/control), PyTorch + YOLOv8/11-pose (perception, AGPL OK for our case), pymavlink or MAVSDK (comms), JAX MuJoCo-MJX/Brax or Isaac Lab (RL surrogate training on Adroit).

**Methodological consensus across UZH work:** camera+IMU → learned gate keypoints → state estimation → {time-optimal trajectory + MPC} OR {end-to-end RL}. We're pursuing **Track A (model-based: map → optimal line → tracker)** as the VQ1 floor + VQ2 baseline; the recommended SPEED-UPGRADE is now **RL-on-surrogate** (fits the user's RL strength; viable given ~unlimited attempts + the deterministic Apache-2.0 Elodin surrogate on Adroit), with **MPCC** as the classical alternative (see [[project-master-plan]] Review-validation #12).

## Academic paper ledger (ingested 2026-05-28, full texts cached in `Academic/_txt/`)
Triage of the 20 PDFs in `Academic/`. SIGNAL = shaped the plan; see Cycle 11 in [[project-master-plan]] for techniques adopted.

| Paper (file) | Verdict | What we take |
|---|---|---|
| **SWIFT** Kaufmann 2023 Nature (`document (1)`) | ★★★ SIGNAL | RL→CTBR; KF correcting VIO *translational* drift via gate detections; IPPE + sampling-based covariance; **ablation: model-based beats RL under good state, RL only wins under domain shift** (decides Track A vs B for us). |
| **CPC time-optimal** Foehn 2021 SciRob (`scirobotics.abh1221`) | ★★★ SIGNAL | True time-optimal at single-rotor limits; min-snap is provably suboptimal; **plan at a TWR margin (3.3 vs ~4.0)**; offline-only (min–hrs) but fine for a static course; point-mass init. |
| **MPCC** Romero 2022 T-RO (`Model_Predictive_Contouring…`) | ★★★ SIGNAL | **Primary tracker.** Tracks any C¹ 3D path (even point-mass) → near-time-optimal in ~2 ms, closed-loop; **dynamic Gaussian contour-weights at gates** guarantee passage; beats CPC+MPC & humans in real flight. |
| **Min-snap** Mellinger & Kumar 2011 (`Minimum_snap…`) | ★★★ SIGNAL | Differential flatness (σ=x,y,z,ψ); geometric SE(3) controller; **time-scaling = single speed/safety knob**. VQ1 floor + fallback tracker. |
| **ADR survey** Moon 2019 (`s11370-018`) | ★★★ SIGNAL | Our exact setting (known course, map given). **Winning teams used waypoint-tracking, not end-to-end.** Yaw-only odom→world frame align (gravity fixes roll/pitch); **velocity from drag-model+accel between gates**. |
| **Jung 2018** Perception/Guidance (`Perception_…`) | ★★★ SIGNAL | CNN gate detect (ADRNet/SSD, 29fps TX2) + LOS guidance → center error <0.1 m; **color ~50% vs CNN ~90%** detection. Robust VQ1/EXPLORE reactive template. |
| **Bry 2015** aggressive flight, known env (`bry-et-al…`) | ★★ SIGNAL | EKF in exponential coords (no Euler singularities); GPF+KF; confirms trust-roll/pitch, correct-yaw/pos. |
| **Tang & Kumar 2018** Autonomous Flight (`annurev…`) | ★★ grounding | Field survey for the FLOSS write-up; flatness/planning/control taxonomy. |
| Rezende 2021 integrated ADR (`s11370-021`) | ★ marginal | Full-pipeline integration sanity reference. |
| Minervini 2025 VIO (`document (6)`) | ★ conditional | Only if sim exposes an IMU stream (R1). |
| DeepPilot 2020 (`document (2)`) | ★ marginal | End-to-end CNN; keep only as RL imitation warm-start. |
| ETRI Hong 2021 A*+min-snap (`ETRI…`); Singh 2023 PD+min-snap (`Attitude_…`); Rendón 2020 PSO (`s10846`) | ½ marginal | Min-snap redundant w/ Mellinger; **A* now relevant — spec confirms real obstacles, so collision-free path search IS needed**; PSO ≈ our CMA-ES tuning analogue. |
| Elamin 2025 event-VIO (`document (5)`); Liu 2022 MUSAK anti-drone (`document (3)`); Qu 2024 D3L-SLAM (`document (4)`); Cui 2022 target-pursuit (`document.pdf`); Springer books (`978-3-031`, `978-3-319`) | ✗ NOISE | No event camera; not racing; heavy SLAM unneeded on known course; off-topic. |

Related: [[project-ai-grand-prix]], [[reference-adroit-princeton]]
