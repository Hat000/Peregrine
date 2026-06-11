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

### 2nd-order inner-loop re-system-ID (cc6921d, fable, 2026-06-10) — ❌ STRUCTURALLY SUPERSEDED by the characterize-sweep (510da24, next section)
Kept as the record of WHY the windup/2nd-order picture was wrong (do NOT re-litigate). cc6921d re-ID'd
the inner loop from the **n=1** S1.2 tumble and concluded: amplitude-dependent **PI-windup**, 20–26%
transient overshoot at saturation, **DC gain anchored at the steady 2.5 gains (G=3.8 explicitly
rejected)**, a two-regime ωₙ/ζ envelope (saturated ωₙ 21–28 / ζ 0.39–0.47; small-signal ωₙ ~67 / ζ
0.7–0.8), and an ωₙ/ζ-envelope DR plan. The controlled 15-run sweep DISPROVED the structure:
- **PI-windup DISPROVEN** — cmd 3.14 holds 11.1 rad/s for 1.2+ s with ZERO decay (a wound integrator
  must bleed back toward the 2.5-gain target; the windup fit predicts exactly that decay).
- **"DC anchored at 2.5 / G=3.8 rejected" was the WRONG CALL** — the optimizer was measuring the REAL
  saturated gain (sweep: sustained 3.50/3.53 at full stick). Saturated steady gain was unobservable in
  the tumble (commands flip every ~100 ms) — n=1 bit us exactly there.
- The **ωₙ∈[21,68]/ζ∈[0.39,0.85] two-regime envelope was the LTI shadow of the wrong DC anchor**:
  per-regime LTI fits are 10–25× worse on each other's data (sweep-3.14 params on the tumble RMSE 6.5
  vs own-fit 0.25) — nonlinearity misattributed to dynamics lands on whatever maneuver you fit.
- The S1.2 "9.7 rad/s vs 7.85 ceiling" was **unmodeled DC gain, NOT transient overshoot**.
- §5 integration plan (amplitude-scheduled ωₙ/ζ, `omega_dot` state) — ❌ DO NOT USE; replaced by the
  static map (sweep WRITEUP §3). Writeup: `handoff/shadowpc-2ndorder-resysid-2026-06-10/`.

### 🚩 The "+30% rate_gain band" DR proxy is DISPROVEN — replacement = the STATIC-MAP DR (updated 2026-06-10)
The asymmetric rate_gain DR band **[0.90, 1.30]** (item ④, in `rl/diffaero_dynamics.py`, used by S1.3) is
the wrong model. ~~Replace with the ωₙ/ζ-envelope DR~~ — **that replacement is ITSELF superseded** (the
envelope was the LTI shadow of cc6921d's wrong anchor). **At retrain, replace with the static-map DR:
s ∈ [0.25, 0.35], τ ∈ [0.015, 0.03], alpha_max ∈ [200, 320]** (next section).

### ✅ CHARACTERIZE-SWEEP DONE (510da24, fable, 2026-06-10/11) — the static gain map; NO sim anomaly
The "NEXT CHEAP EXPERIMENT" ran: 15 controlled runs (magnitude sweep + anomaly probes, n=1→n=11),
unattended on `scripts/rate_sysid.py`. Source of truth: `handoff/shadowpc-characterize-sweep-2026-06-10/WRITEUP.md`.
- **THE PLANT MODEL (supersedes cc6921d): a STATIC amplitude-dependent gain map ("super-rate" style) in
  front of a fast ~critically-damped loop.** `g(|c|) = G0/(1 − s·min(|c|,π)/π)`, **G0 = shipped
  [2.501, 2.504, 2.231], s ≈ 0.29–0.30** (roll/pitch, ±0.02; roll ≡ pitch to 3 digits; one-param form
  fits within ~3%). Sustained gains **2.50 → 3.50/3.53** (roll/pitch) from |cmd| 0.3 → 3.14.
- **Transient on top:** rise t80 ≈ 31–52 ms at ALL magnitudes; slew ≈ 250–280 rad/s² (r/p), ~78 (yaw
  level); saturated step ~critically damped (ζ 0.84–1.04, 0–1% overshoot); small-signal keeps a
  real-but-minor ~10% overshoot (ωₙ≈73, ζ≈0.5 — the 2nd-order refinement is optional, ~10% fidelity);
  input delay 5–15 ms → belongs in the transport-delay mechanism, NOT in τ.
- **Integration spec = sweep WRITEUP §3** (~4 lines in the plant: g(|cmd|) target + existing first-order
  lag τ≈0.020 + alpha_max slew clamp), **parity-gated twin → rl_plant → torch adapter.** **🚩 The
  twin/plant max body-rate clamp must be raised ≥ ~11.5 rad/s (roll/pitch)** or it re-introduces the old
  ceiling artifact. **DR: s ∈ [0.25, 0.35], τ ∈ [0.015, 0.03], alpha_max ∈ [200, 320]** — replaces both
  the +30% band and the ωₙ/ζ-envelope plan. **UPSIDE: real authority at full stick is ~11 rad/s, not
  7.85 — a HIGHER speed ceiling for VQ2.**
- **THERE IS NO SIM ANOMALY (open-loop).** Sustained rotations through full inversion (up to 11.2 rad/s,
  multiple revolutions, multi-axis combos including the exact S1.2 tumble command [+3.14,−3.14,+3.14],
  tilt abort off) are ALL clean rigid-body; uncommanded axes flat zero. The S1.2 "±180° yaw-spin anomaly"
  decomposes into: (a) unmodeled super-rate gain (the real divergence driver), (b) an Euler yaw-flip
  representation artifact near inversion (quaternion smooth), (c) a **gate-post COLLISION (id 1001,
  threat 2) at sim_t≈11.145 imparting −63 rad/s roll** — the "spin" was contact dynamics. **VQ2 open
  item (A) "untwinned yaw-spin anomaly = RL-killer" DISSOLVES.**
- **Env-redesign consequence: crash-termination = COLLISION-based** (DiffAero's racing env already has
  it); do NOT encode a tilt-threshold pseudo-anomaly (it would forbid attitudes the sim handles fine and
  fast racing may legitimately visit). Tilt/jerk regularization stays justified for smoothness/VQ2 style
  + keeping the policy in well-modeled regimes — NOT anomaly avoidance. **The env/reward redesign is
  therefore UNGATED** (its gate was the anomaly-boundary measurement — there is none to encode).
- **Checkpoint impact: S1.3 + inc-1 trained on the WRONG plant** (2.5-flat; under-predict their own
  authority by up to 42% at full stick) → **retrain after the static-map integration.** The S13-LIVE
  transfer test is **DEMOTED from disambiguator to optional cheap datum** (fly opportunistically,
  non-blocking — the sweep answered the plant-fidelity question directly). **Critical path:
  static-map integration (parity-gated) → env/reward redesign → retrain.**
- **Durable sim/ops facts:** (1) **🚩 THIRD sim idle state** — after ~90 min idle post-race the sim parks
  on an off-race screen where MAV_CMD 31000 is a NO-OP while telemetry still streams (stale RACE_STATUS
  `started=True`, frozen sim_time); recovery = Win32-focus the AI-GP window + Enter ×2 — wired as
  automatic escalation in `scripts/rate_sysid.py` (15+ runs chained, zero GUI touching). (2) **Yaw gain
  is maneuver-dependent**: ~2.35 level (looks like a ~7.4 rad/s cap) vs ~3.1 tumbling — caveated, own
  pass only if it ever matters (racing yaw cmds are small). (3) All probes near hover; aero at racing
  airspeed unmeasured (the DR band covers; live course RMSE 0.22–0.41 at ≲3 rad/s says small-signal is
  unaffected). (4) `rate_sysid.py` gained `--vel-damp`/`--pos-pull` station-keeping + `--mode anomaly`.

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
  **✅ TRUNCATION GOTCHA RESOLVED (2026-06-09):** finish_hold.py (commit 52f075d) ships the post-finish drain; 57f287f adds residue-handling to race_outcome. VQ1 6/6 now self-certifies from tlog.
- **Stage 1 — increment 1 DEPLOYED (S1.2 done: pipeline VERIFIED, checkpoint NOT transfer-ready; S1.3 retrain = NEXT)**: state→action policy on the KNOWN map + GIVEN pose, twin-trained,
  VQ1-sim-validated — **ZERO vision dependency**; subsumes the 3 known VQ1-stack issues (start transient,
  alt relay, descents). Build off the proven `peregrine_train.py`: real RL I/O (§RL policy I/O) + reward
  shaping + DR on rate_gain/hover/linear_drag (g=9.80665); validate the trained policy in VQ1 sim via
  `race_outcome`.
  **✅ Start-transient issue ① RESOLVED at the control level (2026-06-09, commit 260972e):** `launch_ramp_s=0.6 s` authority ramp eliminates the rate-clamp saturation / tick-phase dice-roll — 4× deterministic 6/6 confirmed on 1.0.3364. RL subsumption (smoother racing line, faster transitions) remains the VQ2 path; the VQ1 blocker is gone.
  **Stage 1 increment 1 ✅ COMPLETE (2026-06-09):** CTBR policy threads 6-gate course on our plant, given pose, zero vision. success_rate 0→0.97 (A100, 30:48 wall). obs_dim=17, DR on. Checkpoint `stage1_inc1_actor.pth`. Caveats: (1) over-aggressive (l_ep ~1.9s, no speed shaping — VQ2 target); (2) real test = live sim transfer.

  **✅ S1.2 — increment-1 LIVE DEPLOYMENT (2026-06-10, fable session): pipeline VERIFIED end-to-end, checkpoint NOT transfer-ready (flight 1: 0 gates, tumbled into gate-0 post). Detail: `handoff/shadowpc-s12-rl-live-2026-06-10/`.**
  - **🚩 CORRECTED DEPLOYMENT RECIPE (fixes 3 sonnet S1.1 bugs — the verified-correct way to deploy a DiffAero-trained actor live, in `rl/fly_rl.py`):** ① actor output = **tanh(mean) → rescale** to thrust [0,5] / rates ±3.14 rad/s (NOT raw actor mean); ② FLU→FRD action sign = **[1,−1,−1]** (sonnet's [−1,+1,+1] wrong on all 3 axes; the plant applies rate_gain·rate_sign identically in train + live — no ff/rate_gain algebra); ③ collective obs init = **0.0** with **rescaled** feedback (NOT 1.0); ④ training control rate = **30 Hz** (racing.yaml dt 0.0333; NOT 100); ⑤ training **resets at rest** 1 m in front of a random gate (NOT "racing velocity"). Verified-correct from sonnet: obs layout (17), `R_W2G=diag(−1,−1,1)`, gate yaws all π, final-gate clamp, Euler-ZYX, actor arch (NormedLinear [256,128]).
  - **🚩 "OOD-at-start = root cause" RETRACTED** — an artifact of sonnet's wrong action transfer function, not a real diagnosis; reset saturation is the policy's NORMAL launch behavior (it saturates in training too, then modulates).
  - **Tail-first spawn bug + fix:** policy trained identity-reset + all gates at yaw π ⇒ flies the course tail-first; sim spawns nose-first on a 17.8°-tilted pad ⇒ ~180° attitude-OOD. Fixed with a **virtual π body-z flip in fly_rl.py** (`--virtual-flip`, default ON; exact rigid-body symmetry). Offline: handoff-state rollouts 0/6 → 6/6 under training physics.
  - **🚩 THE REAL transfer failure (the key finding) = the policy is a "backflip-diver":** its NOMINAL twin maneuver rolls through **104–126° before every gate** (fine offline). Live the maneuver diverges — realized rates hit **9.7 rad/s vs the twin's first-order 7.85 ceiling** (=3.14·2.5; ❌ diagnosis SUPERSEDED 2026-06-10: unmodeled static super-rate DC gain, NOT transient overshoot — see the characterize-sweep section), it blows through ±180° tilt (the then-suspected "yaw-spin anomaly" — **DISSOLVED by the sweep**: super-rate gain + Euler yaw-flip artifact + a gate-post COLLISION), and tumbles into the gate post within 0.35 s. NOT reproducible in the twin (replay from the exact live handoff state passes gate 0 at any latency ≤100 ms and gain ×1.24) → **no deployment-side knob fixes it.** The BRIDGE worked perfectly: CTBR delivered the drone dead-centre (dy +0.04, dz +0.05 m) at 5.1 m/s, 3 m before gate 0; the policy's first action = exactly the offline-twin prediction.
  - **OOD verdicts (offline, measured via `rl/offline_rollout.py`):** training reset (1 m, rest) = **6/6 finish 3.3 s** (validates ALL deployment math); raw standing start (23.3 m, rest) ≈ 0–1/6 (arrives at gate 0 at 33–44 m/s, unrecoverable); **bridge handoff + virtual flip = 6/6 under training physics, 4/6 under the live collective ceiling** — the gate-4 wall is purely the thrust clip (trained max_normed_thrust 5.0 ≈ 5 g vs live collective≤1.0 ≈ 3.765 normed ≈ 3.77 g), unfixable at deployment, fixed in retrain.
  - **🆕 UNATTENDED SIM CONTROL ACHIEVED here (the user's standing ask; durable capability for ALL future sim work):** **MAV_CMD 31000** (`client.send_sim_reset()`) restarts the race once a race context exists (fresh ~3 s countdown) — **NO-OP from HOME** (no telemetry there). From HOME: Win32 `SetForegroundWindow` to focus the `AI-GP` window (`WScript.Shell.AppActivate` alone returns False) + Enter twice (home → waiting room → race+countdown). The session **cold-launched FlightSim.exe and raced with NO human.** `rl/fly_rl.py --flights N` chains attempts (never resets into a ticking countdown).
  - **✅ S1.3 RETRAIN SPEC (Path C, ~30 min A100, `rl/peregrine_racing_s13.sbatch` staged):** ① `dynamics.controller.max_normed_thrust=3.765` (live ceiling; one-line override); ② **standing-start resets** (`+env.standing_start_frac`, reset at ~23 m from gate 0 at rest = real race start; **implemented** cfg-gated in `peregrine_racing.py`, spawn pose mapped through the deployment virtual flip → fly_rl.py needs no change; removes the CTBR bridge, bridge stays as fallback); ③ **tilt/jerk regularization** (`reward_weights.quadrotor.attitude≈2.0`/`jerk≈0.3`, sweep — kills the >90°-roll style; then framed as anomaly avoidance, NOW (510da24) justified for smoothness/VQ2 style + staying in the well-modeled envelope only — no anomaly exists; + the VQ2 aggression fix); ④ **latency DR** (port the transport-delay ring buffer to the torch backend of `rl/diffaero_dynamics.py` — rl_plant already supports it, parity-checkable via `check_against_rl_plant` with `transport_delay_steps>0`; DR per-env delay ∈ {0,1,2}) + asymmetric rate-gain DR band **[−10%,+30%]** (cheap transient-overshoot proxy — ❌ later DISPROVEN; replace with the static-map DR, see above). **④ needs an Adroit session on diffaero_dynamics.py FIRST + must re-pass the parity gate** (not yet implemented). Reward shaping (smoothness/time) folds in here.

  Reward shaping (smoothness/time) = part of S1.3, AFTER the transfer fixes.

  **✅ S1.3 RETRAIN COMPLETE IN-TWIN (2026-06-10) — all four items done + validated; an INTERIM TRANSFER-TEST checkpoint, NOT the final policy.**
  - **① max_normed_thrust=3.765** (live collective ceiling) — active in all runs.
  - **② standing-start resets** — active; **FIXED A REAL BUG:** the spawn at x_zup=0 sat ~8 m OUTSIDE the gate-bbox+margin OOB box ⇒ every standing-start env truncated on step 1. This was the cause of an earlier eval's 768768-episodes / 0%-success — that was the BUG, NOT the policy. Spawn is now folded inside the OOB box.
  - **③ tilt/jerk regularization tuned → ATT=2.0, JERK=0.5: 100% 6/6 in-twin, peak roll 65°** (< 80° target), down from inc-1's 104–126° — **the backflip-dive is GONE in-twin.**
  - **④ adapter robustness** (latency + asymmetric rate-gain DR) implemented; parity preserved (`check_against_rl_plant` DIV **2.22e-16**).
  - **🚩 KEY CORRECTION to inc-1's record — inc-1 had NO PLANT DR.** The training launcher defaulted to the numpy backend, which **silently ignores per-env DR** (item ④'s DR only fires on the torch backend). So inc-1's success_rate 0.97 was trained on the NOMINAL plant with ZERO domain randomization (overfit-to-nominal — partly explains its brittleness / non-transfer). S1.3 switched the launcher to the **torch backend → DR is actually active for the first time.**
  - **🚩 CHECKPOINT PROVENANCE — CORRECTED (supersedes the prior f2347db bank, which had it INVERTED).** The S1.3 deliverable is the **2000-update ATT=2.0 (att2) checkpoint** — committed as **c05362f** at `rl/checkpoints/stage1_inc3_actor.pth`, md5 **ea2bf8a24eb4d149ff5dde2fda9432d9**, 158807 bytes (force-added past the `*.pth` gitignore). It hit **success 1.000 by update 2000 and plateaued** — the converged, correct deliverable. The **5000-update run NaN'd** in the PPO policy at update ~2243 (`Expected parameter loc ... to satisfy Real()`) and **NEVER wrote a checkpoint**; with seed=0 a blind re-run would likely re-diverge (deterministic). A true 5000-update policy would need a different seed and/or PPO grad-NaN guards — a cheap follow-up, NOT needed for transfer (the 2000-update already plateaued). **Tilt-reg sweep:** ATT swept {2,4,8} → **2.0 won** (success 1.000 + peak roll 65°); 4 and 8 over-regularize and converge worse.
    **S1.3 SOURCE changes (committed separately):** `rl/diffaero_dynamics.py` (item ④ adapter robustness), `rl/peregrine_racing.py` (OOB-box / standing-start fix), `rl/peregrine_train_racing.py` (torch backend), `rl/peregrine_eval.py` (new), `rl/peregrine_s13_sweep.sbatch`, `rl/run_parity.sh`.
    Loose end: cosmetic `TRAIN_RC=1` from diffaero's post-checkpoint export not supporting `action_frame="body"` — it fires AFTER `agent.save()`, so the checkpoint + eval are unaffected.
  - **🚩 CHECKPOINT CAVEAT (updated 2026-06-10, sweep 510da24): S1.3 + inc-1 were trained on the WRONG plant** (2.5-flat; they under-predict their own authority by up to 42% at full stick — and ④'s +30% band perturbs the wrong mechanism). **Retrain after the static-map integration.**
  - **🚩 The S13-LIVE transfer test is DEMOTED from disambiguator to optional cheap datum** (fly opportunistically via `fly_rl.py --flights N`, non-blocking) — the sweep answered the plant-fidelity question directly. **Critical path: static-map integration (parity-gated twin→rl_plant→torch) → env/reward redesign (UNGATED) → retrain.**

  **🆕 NEXT RETRAIN = an ENV-COHERENCE REDESIGN, not just a policy re-tune (user directive 2026-06-10) — now UNGATED (sweep 510da24 done; there is no anomaly boundary to encode).**
  The user flagged that the **reward CONFLICTS with the termination**, and we need **more reward terms with
  more explicit logic on how each works.** Reframe the next retrain as redesigning the ENV: reward +
  termination designed TOGETHER + the **static-map DR** (s/τ/alpha_max, see the characterize-sweep section)
  + **COLLISION-based crash-termination** (DiffAero's racing env already has it; NOT a tilt threshold —
  ~~crash-termination at the measured anomaly boundary~~ superseded, no anomaly exists). Hand to a dedicated
  **fable + adroit-connector** session. Non-trivial, objectively-checkable → good fable fit.

  **🆕 PENDING ARCHITECTURE DECISION (2026-06-10 discussion) — S2 = decomposed plan-line + RL-tracker?**
  Consider making S2 a **decomposed plan-line + RL-tracker** (plan an explicit smooth/feasible racing line;
  RL only TRACKS it) INSTEAD of another monolithic racer. Rationale: the monolithic policy's freedom to choose
  the trajectory is what produced the **backflip-dive** — an explicit line makes inversion STRUCTURALLY
  impossible. Tradeoff: lower speed ceiling than SWIFT-style learn-the-line, bought back via the offline
  line-iteration flywheel. **DECISION PENDING the post-static-map retrain result** (the S1.3 live datum is
  now optional — see the characterize-sweep section). (Mapping is PERCEPTION — conservative lap +
  detector→PnP→KF — NOT an RL task, and only needed if VQ2 hides the map; open organizer question.)
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
