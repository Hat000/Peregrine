---
name: project-phase2-rl-vision-decisions
description: Phase-2 (post-VQ1) planning decisions — RL substrate landscape (DiffAero/Crazyflow over Isaac), vision investment plan, RL policy I/O + rewards, gate-map/SLAM cases, the authoritative collision signal, Adroit-as-edge. From the 2026-06-07 planning session.
metadata:
  type: project
---

## How to apply
Captures the strategy decisions from the 2026-06-07 Commander+user planning session (post-VQ1, pre-build).
NOT yet executed — these are the agreed directions + open bake-offs for the NEXT session to act on.
First-session-next: wire a pointer in MEMORY.md + mirror to ~/.claude; update [[project-master-plan]] C1
(it still says "Isaac Lab is the natural scale choice" — SUPERSEDED, see below). Related:
[[project-master-plan]], [[project-detector-training-pipeline]], [[project-estimator-robustness]],
[[reference-competition-materials]], [[reference-adroit-princeton]], [[reference-sim-interface]].

## Context
- **VQ1 banked; planning Phase 2.** No VQ2 sim access (not released to teams) → **VQ1 sim = our "faithful
  truth" for VQ2** (no other option). Discipline: randomize GENEROUSLY over the unmeasurable VQ1→VQ2
  dynamics/latency delta (twin already under-models live latency ~25%); handle the appearance gap by a
  state-policy (not pixels) + a domain-randomized detector.
- **Adroit summer access CONFIRMED** (resolves the long-standing open item). User directive: **Adroit is a
  competitive EDGE** (free GPU other teams may lack) — spend it on BREADTH (wide domain randomization =
  the robustness moat; VQ2-photoreal detector training; ensembles/seeds/sweeps), not just raw steps
  (differentiable methods are sample-efficient).

## RL substrate — DiffAero/Crazyflow OVER Isaac (decision pending a cheap bake-off)
The recurring "just use Isaac Lab, it's batteries-included" tension RESOLVES via lighter, drone-native,
differentiable sims that ALSO supply the infra:
- **DiffAero** (arXiv 2509.10247, github flyingbitac/diffaero) = **leading candidate**: GPU, fully
  DIFFERENTIABLE, lightweight, **multiple dynamics models** (inject our system-ID'd plant), built-in
  sensor stacks (IMU/depth/LiDAR) + racing/obstacle tasks (likely the collision detector we'd otherwise
  hand-roll — verify). Batteries-included WITHOUT Isaac's weight/wrong-plant. "Policies in hours."
  **CONFIRMED 2026-06-08 (paper + repo read):** PyTorch (torch>=2.0), standalone `pip install -e .` (NO
  Isaac/Omniverse), BSD-3, headless. Dynamics take BODY-RATE+COLLECTIVE = our exact CTBR action;
  `dynamics/base_dynamics.py` = the inject point for our system-ID'd plant; `env/racing.py` = ordered-gate
  racing+collision (the env we'd otherwise build); `algo/` = PPO+SHAC+APG(BPTT)+DreamerV3, and SHA2C
  handles MIXED dense-differentiable + sparse-non-differentiable rewards (= our gate-pass reward — solves
  DiffRacing's "gate-pass isn't differentiable" problem); models control-latency + per-episode DR of
  drag/latency/action-range (the VQ1→VQ2 robustness lever). Deps light EXCEPT pytorch3d + open3d
  (cluster-install pain; both rendering/3D → likely SKIPPABLE for a STATE policy). **VERDICT: adopt,
  pending only the Adroit install smoke test.**
- **Crazyflow** (arXiv 2606.01478, utiasDSL/crazyflow) = JAX sibling, ~100M steps/s @ 1M envs.
- **DiffRacing** (vector-field-augmented differentiable policy learning) = a METHOD, not a sim; its
  **Delta Action Model closes dynamics mismatch WITHOUT system-ID** → directly relevant to the
  unmeasurable VQ1→VQ2 gap. Layer on a differentiable sim.
- **Isaac (OmniDrones / Aerial Gym) = DEMOTED**: heavy SLURM install (container/driver/MIG), PhysX ≠ our
  plant, rendering unused for a state policy. Its batteries don't outweigh the wrong-plant + install cost.
- **KEY INSIGHT: differentiable-sim policy learning is the modern fast path** (beats Isaac+PPO,
  sample-efficient, lightweight). Our `twin.py` (~50 lines analytical) is differentiable for free when ported.
- **Decisive criterion = PLANT FIDELITY.** The twin MUST match the system-ID'd VQ1 plant (inverted-yaw
  cmd, rate_gain ~2.5, τ=0.019, hover 0.2656, drag 0.219, ~40 ms latency) for transfer. The substrate is a
  vehicle; **our plant is the payload, injected regardless of tool.**
- **STAGE-0 BAKE-OFF — RUN LIVE ON ADROIT 2026-06-08 → DiffAero CONFIRMED (PASS).** Installs standalone
  (NO Isaac), torch cu121 + CUDA verified on a GPU node (the `gpu` partition has BOTH A100 [4/node,
  adroit-h11g1..3] and V100-32GB), `import diffaero` works, configs `cfg/env/racing.yaml` +
  `cfg/algo/{ppo,shac,sha2c,apg}.yaml` present, action space = **body-rate + collective = our exact CTBR**,
  **plant-injection point = `dynamics/base_dynamics.py`** (+ `quadrotor.py`, `controller.py`). BSD-3.
  - **ONE remaining env step: `pytorch3d`** (required by `diffaero/env` transforms — NOT skippable; the
    "skip rendering deps" guess was wrong). **🚩 Adroit COMPUTE NODES HAVE NO INTERNET → install on the
    LOGIN node.** pytorch3d builds from source → needs the CUDA toolchain ON PATH: the naive
    `module load cudatoolkit/12.6` did NOT put `nvcc`/`ninja`/proper `gcc` on PATH in a non-login shell, so
    the build failed fast. NEXT: set `CUDA_HOME`+PATH explicitly, `pip install ninja`, ensure a compatible
    gcc — OR (preferred) grab a prebuilt wheel (miropsota/torch_packages_builder) matching torch-cu121-py311
    to skip compiling entirely. (Consider asking Princeton RC for the sanctioned CUDA-build module combo.)
  - **Flat-layout gotcha:** clone dir is `diffaero_repo` but code imports `diffaero` → fix with
    `ln -sfn diffaero_repo diffaero` + `PYTHONPATH=<parent>` (or repair the editable install). Footprint:
    `~/.conda/envs/diffaero` (6.4 GB, home), code/logs in `/scratch/network/fl3689` (our dirs only).
  - THEN: tiny racing PPO/SHAC for a throughput number, then write our system-ID'd plant as a
    `base_dynamics` subclass. Smoke sbatch + install scripts are staged in `/scratch/network/fl3689/`.

## Vision = serious investment (user: "train it into something unbeatable") — TWO distinct workstreams
Do NOT conflate them:
- **(a) Detector training** (Adroit + domain randomization) — for the VQ2 PHOTOREAL/appearance gap (the
  real VQ2 vision challenge; "bad in VQ1 → worse in VQ2" is true on THIS axis). This is the Adroit spend.
- **(b) Engineering fixes** — the VQ1 issues measured this session are NOT training problems: the 2-fold
  IPPE flip (solver/prior; a BLANK square is worse than an AprilTag — no internal pattern to break corner
  symmetry → needs the attitude+map+temporal prior, currently firing too weakly), the +3.5°/range yaw
  CALIBRATION bias, wrong-gate association. Training won't fix (b); solver/calibration/logic will.
  **✅ DONE (commit 4673517, ShadowPC, 2026-06-09): `association.py` ships; wrong-gate association killed
  (87 catastrophic fixes eliminated); depth-sanity flip disambiguation kills the scale-error mode (not a
  solver-flip mode — key insight: geometry consistency, not solver tie-breaking, is the lever). Catastrophic
  tail 46%→6.3%, leak 1.6%→1.1%, 370 tests green. Residual 2 leaks = long-range depth noise → next lever
  = `attitude_noise_std` (Stage-2 / vision pkg 2). Vision workstream (b) COMPLETE for VQ1+Stage-1.**

### Perception characterization (this session) — `scripts/characterize_perception.py`
Ran the navigator's ACTUAL chain (detector v2 → estimate_gate_pose → gate_pose_to_world_position) on the
VQ1 gate-0 bundle (handoff/shadowpc-followups-2026-06-05/task2_frames) vs given pose. First time vision was
measured on REAL frames in-loop (v2 had only been eval'd on synthetic; the VQ1 PASS flew vision-OFF):
- **Detection 100%** (40/40, 1.8–34 m; reproj p50 1.6 px) — detector is NOT the bottleneck.
- **Depth IS recoverable** (clean-subset N-bias ~0 — user was right: known size + intrinsics = AprilTag-like).
- Real errors: the **flip tail (31% of fixes, 3–24 m)** + the **+3.5°/range yaw bias** (cross-validates
  Task-2's −3.6° via the full PnP) + the **analytic PnP covariance ~30× too optimistic**.
- VQ1-irrelevant (flies given pose; the Mahalanobis gate rejects the whole tail). **RL-perception-model
  lesson: position-fix noise is anisotropic + heavy-tailed + range-growing + has a discrete flip mode + a
  systematic bias → inject THIS into the RL twin, NOT isotropic Gaussian** (master-plan C1 "measured
  perception model = speed ceiling"). Detail: [[project-estimator-robustness]].

### Perception-noise model — MEASURED full-course (2026-06-08, T1) → the RL twin input
Ran the REAL YOLO→PnP→KF chain on the canonical VQ1 6/6 recording `data/runs/20260607_194615_course_60s`
(1159 frames; per-gate bundles pooled, N=312 solved fixes). **This is the measured speed-ceiling model.**
- **Per-axis bias ± σ** (KF-ACCEPTED fixes, the in-loop cut): N −0.42 ± 0.73 · E +0.06 ± 0.47 · D −0.28 ± 0.29 m. |fix| p50 0.82 / p90 1.66 m. Vertical (D) tightest.
- **Range-FLAT to 24 m — NO growth** (⚠️ CORRECTS the earlier "range-growing" guess above). Per-axis slope ≈ −3.8°…+4.1° = the ~3.6°/range yaw bias, now confirmed through the full chain.
- **Catastrophic tail:** raw |fix|≥3 m = 46%→6.3% (143→12) with association+depth-sanity (commit 4673517, 2026-06-09). The KF χ²₀.₉₉₉ gate rejects further → **residual leak ≈ 1.1% (0.6% of frames) = UPDATED bad-fix rate the twin MUST model.** Good <1 m fixes went UP (103→107).
  - **Key insight:** the "56 frontal depth flips" were NOT solver flips — wrong-scale detector boxes (solved depth 1.2–13× true, p50 2.3×, reproj p50 1.0 px). Geometry/depth-sanity kills them; better solver tie-breaking couldn't have.
  - **2 residual leaks** = honest long-range depth noise (25–38 m range, errors 3–5 m) on correctly-associated next-gate fixes. Next lever = `attitude_noise_std` (queued, Stage-2), NOT tighter geometry gates.
- **PNP_FIX_COV_INFLATION task CLOSED:** behind the depth-sanity gate K=1.0 and K=2.0 leak identically (prior objection — K=2.0 worsens leak — evaporates). No revert needed.
- **TWIN ONE-LINER:** world-fix σ≈[0.73, 0.47, 0.29] m (N,E,D), bias [−0.4, +0.06, −0.28] m, **range-flat to ~24 m**, ±~3° angular term, assoc ~85–95% at 5–15 m, **~1.1% catastrophic leak** after χ²₀.₉₉₉ + association + depth-sanity (commit 4673517, 2026-06-09; prior figure was ~1.6%). **Use 1.1% as the Stage-2 twin input.** Detail: `handoff/perception-char-2026-06-08/`.

## Depth model / ~100 TOPS budget (user) — OFFLINE flywheel
After a recorded run (legal between-runs processing): **multi-view triangulation** from logged poses
(metric, no scale ambiguity — preferred over real-time depth) OR a learned depth model → map the gray-wall
**OBSTACLES** → regen a collision-free line for the next run. Real-time learned depth = backstop for
UNMAPPED obstacles only. Depth's value = obstacles/free-space, **not gate-depth** (known-size PnP has that).

## Gate map / SLAM (user Q) — depends on the VQ2 stream
- **VQ1: map is GIVEN** (TRACK_INFO broadcast type-2 + `capture_track_map.py` + given pose) → **NO SLAM.**
- **VQ2 cases:** (A) gives pose like VQ1 → average gate sightings vs known pose = precise map, no SLAM;
  (B) rough map + pose → refine; (C) **NO pose → gate-landmark SLAM, but BOUNDED** (metric scale from the
  known 1.5 m gate + given IMU attitude = pose-graph over gate landmarks, NOT dense visual SLAM), as an
  offline exploration-lap step.
- **Does NOT block RL** (decoupled; map source swappable behind the seam) — user's own point, affirmed.

## RL policy I/O + rewards (SWIFT-style state→action, NOT pixels)
- **INPUT:** platform state (velocity, attitude/gravity-direction, body rates [, collective]) + **gate-
  RELATIVE geometry** (next gate(s) relative position + normal, body frame — translation-invariant)
  [+ optional reference-line lookahead]. Vision→map→KF supplies the gate-relative geometry; pixels never
  enter the policy.
- **OUTPUT:** **CTBR** = body rates (3) + collective (1) — drop-in for the `SET_ATTITUDE_TARGET` /
  `Setpoint`+`ControlCommand` seam (policy REPLACES planner+controller).
- **REWARD:** dense gate-PROGRESS + gate-PASSAGE bonus + collision/crash penalty (twin geometry) +
  action-smoothness + finish bonus; optional SMALL visibility term (speed > keeping gate in view → minimal).
  Racing line: feed a pre-built reference (safer first) OR let the policy learn it (higher ceiling, SWIFT-style).
- **Pixel-to-control REJECTED** (why, since it recurs): (1) forces rendering into the RL loop → needs
  photoreal = Isaac = heavy + kills parallel-dynamics speed; (2) bakes the VQ1→VQ2 appearance gap into the
  policy; (3) discards the robust supervised detector. SWIFT avoided it.

## Collision detection (user's catch: "our software is terrible at pass-vs-collision")
The SIM emits **AUTHORITATIVE** signals — `COLLISION` (id 1001=gate / 1002=env, threat_level) +
`RACE_STATUS.active_gate_index` — both captured in `mavlink_client.py` and logged (a rejected run shows
"5 collisions"). The unreliability was our GEOMETRIC pass-heuristic (`fly_vq1` gate-0-saga false-passes) +
AI eyeballing the GUI, **NOT the sim.** FIX: an offline `race_outcome` analyzer (active_gate_index
transitions + COLLISION(1001) events → per-gate clean/contact/miss). For the TWIN reward, collision = our
own clean, unit-tested geometry (gate plane + 1.5 m opening + frame thickness). NOT infra we lack.

## Staging (walking-skeleton for RL — addresses "too many components to get right")
- **Stage 0 ✅ COMPLETE (2026-06-08)**: `race_outcome` analyzer SHIPPED + ShadowPC-validated (T1: correctly
  scores the canonical 6/6 recording as 5-clean/0-contact — see the truncation gotcha). DiffAero substrate
  GREEN on Adroit (V100). **Our plant injected + PROVEN**: `rl_plant.py` parity bit-identical to the twin;
  the DiffAero adapter's torch backend = machine-epsilon identical (gate `check_against_rl_plant` DIV
  **4.4e-16** on V100); smoke-train racing-PPO n_envs=2048 **~88.9K env-steps/s**, success 0→0.79, ep-len
  0.5→32.6 s. Wiring = a launcher (`peregrine_train.py`) monkeypatching `DYNAMICS_ALIAS` (ZERO edits to the
  diffaero clone); zero physics/interface drift vs the clone. Stage-1 scaffolding on Adroit:
  `/scratch/network/fl3689/peregrine_repo/{src/racer,rl}` + `peregrine_{gate,smoke}.sbatch`.
  **🚩 race_outcome TRUNCATION GOTCHA:** `fly_vq1` force-disarms AT the final-gate pass → the terminal
  RACE_STATUS never reaches the tlog → recordings self-certify only 5/6 + finished=False (true 6/6 GUI-only).
  Fix queued (hold ~1 s past finish).
- **Stage 1 — ✅ UNBLOCKED = NEXT**: state→action policy on the KNOWN map + GIVEN pose, twin-trained,
  VQ1-sim-validated — **ZERO vision dependency**; subsumes the 3 known VQ1-stack issues (start transient,
  alt relay, descents). Build off the proven `peregrine_train.py`: real RL I/O (§RL policy I/O) + reward
  shaping + DR on rate_gain/hover/linear_drag (g=9.80665); validate the trained policy in VQ1 sim via
  `race_outcome`.
- **Stage 2**: layer the MEASURED perception-noise model (asymmetric actor-critic: privileged critic sees
  truth, actor sees noisy perception-state) + eval the policy driven by REAL YOLO→PnP→KF with **given-pose
  OFF** in VQ1 sim ← the right home for the user's "test the control policy with real YOLO vision."
  **Use updated 1.1% leak figure (commit 4673517)** for the twin injection, not the prior 1.6%. Next
  vision-engineering lever at this stage = `attitude_noise_std` (see perception-char findings above).
- **Vision engineering workstream (b) COMPLETE for VQ1+Stage-1** (commit 4673517, 2026-06-09). Residual
  lever for Stage-2 / vision pkg 2: `attitude_noise_std` (currently inert for given-pose nav).
- **Vision engineering (a) runs PARALLEL** (detector training, Adroit, decoupled by design).

## #1 ORGANIZER ASK (user offered to email info@theaigrandprix.com)
**"In Round Two (VQ2), does the sim still stream LOCAL_POSITION_NED / ODOMETRY (drone position+velocity),
or is position vision-only?"** — architecture-defining: if VQ2 gives pose, the vision→map→path→RL risk
largely evaporates (RL flies on given pose; vision just confirms gates). The spec is SILENT (grepped: says
"GPS not available / no absolute global position" but nothing on VQ2 LOCAL_POSITION_NED). Secondary: the
submission interface spec + VQ1 deadline + confirm registration active.

## Open
- ~~Substrate bake-off verdict~~ ✅ RESOLVED: **DiffAero** — proven on Adroit (plant injected, gate PASS, trains our plant).
- Structural pilot-stack changes (user brainstorming — the Setpoint/ControlCommand seam keeps a
  planner+controller→policy swap low-risk).
- VQ2 data-stream answer (gates the map/SLAM + vision-load-bearing question).
