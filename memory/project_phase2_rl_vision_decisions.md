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
  - **✅ pytorch3d RESOLVED (2026-06-08): prebuilt wheel `0.7.8+pt2.5.1cu121-cp311`** (miropsota/
    torch_packages_builder) + torchvision/open3d — env GREEN; smoke racing-PPO n_envs=64 ≈ **3,220
    env-steps/s on a V100, success 0.97** (a FLOOR — scales to thousands of envs; 2048-env smoke hit
    ~88.9K steps/s, see Stage 0 below). Historical note kept below:
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
  = `attitude_noise_std` — **✅ DONE (VISION-PKG2, 2026-06-10; see that section below)**. Vision workstream (b) COMPLETE for VQ1+Stage-1.**

### Perception characterization (this session) — `scripts/characterize_perception.py`
Ran the navigator's ACTUAL chain (detector v2 → estimate_gate_pose → gate_pose_to_world_position) on the
VQ1 gate-0 bundle (handoff/shadowpc-followups-2026-06-05/task2_frames) vs given pose. First time vision was
measured on REAL frames in-loop (v2 had only been eval'd on synthetic; the VQ1 PASS flew vision-OFF):
- **Detection 100%** (40/40, 1.8–34 m; reproj p50 1.6 px) — detector is NOT the bottleneck.
- **Depth IS recoverable** (clean-subset N-bias ~0 — user was right: known size + intrinsics = AprilTag-like).
- Real errors: the **flip tail (31% of fixes, 3–24 m)** + the **+3.5°/range yaw bias** (cross-validates
  Task-2's −3.6° via the full PnP — ❌ later REFUTED as a fixed calibration, VISION-PKG2 2026-06-10, see
  below) + the **analytic PnP covariance ~30× too optimistic**.
- VQ1-irrelevant (flies given pose; the Mahalanobis gate rejects the whole tail). **RL-perception-model
  lesson: position-fix noise is anisotropic + heavy-tailed + range-growing + has a discrete flip mode + a
  systematic bias → inject THIS into the RL twin, NOT isotropic Gaussian** (master-plan C1 "measured
  perception model = speed ceiling"). Detail: [[project-estimator-robustness]].

### Perception-noise model — MEASURED full-course (2026-06-08, T1) → the RL twin input
Ran the REAL YOLO→PnP→KF chain on the canonical VQ1 6/6 recording `data/runs/20260607_194615_course_60s`
(1159 frames; per-gate bundles pooled, N=312 solved fixes). **This is the measured speed-ceiling model.**
- **Per-axis bias ± σ** (KF-ACCEPTED fixes, the in-loop cut): N −0.42 ± 0.73 · E +0.06 ± 0.47 · D −0.28 ± 0.29 m. |fix| p50 0.82 / p90 1.66 m. Vertical (D) tightest.
- **Range-FLAT to 24 m — NO growth** (⚠️ CORRECTS the earlier "range-growing" guess above). Per-axis slope ≈ −3.8°…+4.1° = the ~3.6°/range "yaw bias" (❌ VISION-PKG2 2026-06-10: NOT a fixed calibration — the both-signed per-gate slopes were the tell; it decomposes into a depth-scale artifact + per-gate lateral offsets + flight-specific roll wander, see §VISION-PKG2 below).
- **Catastrophic tail:** raw |fix|≥3 m = 46%→6.3% (143→12) with association+depth-sanity (commit 4673517, 2026-06-09). The KF χ²₀.₉₉₉ gate rejects further → **residual leak ≈ 1.1% (0.6% of frames)** (⚠️ SUPERSEDED by VISION-PKG2 2026-06-10: leak 0.53% of solved — see §VISION-PKG2). Good <1 m fixes went UP (103→107).
  - **Key insight:** the "56 frontal depth flips" were NOT solver flips — wrong-scale detector boxes (solved depth 1.2–13× true, p50 2.3×, reproj p50 1.0 px). Geometry/depth-sanity kills them; better solver tie-breaking couldn't have.
  - **2 residual leaks** = honest long-range depth noise (25–38 m range, errors 3–5 m) on correctly-associated next-gate fixes. Next lever = `attitude_noise_std` — ✅ DONE (VISION-PKG2, 2026-06-10): the 38 m leak is killed by the 32 m range cap, the 25 m one remains (documented trade-off), NOT tighter geometry gates.
- **PNP_FIX_COV_INFLATION task CLOSED:** behind the depth-sanity gate K=1.0 and K=2.0 leak identically (prior objection — K=2.0 worsens leak — evaporates). No revert needed. Detail: `handoff/laptop-pnp-cov-inflation-2026-06-08`.
- **TWIN ONE-LINER:** world-fix σ≈[0.73, 0.47, 0.29] m (N,E,D), bias [−0.4, +0.06, −0.28] m, **range-flat to ~24 m**, ±~3° angular term, assoc ~85–95% at 5–15 m, **~1.1% catastrophic leak** after χ²₀.₉₉₉ + association + depth-sanity (commit 4673517, 2026-06-09; prior figure was ~1.6%). **⚠️ leak/acceptance SUPERSEDED by VISION-PKG2 (2026-06-10): leak 0.53% of solved (bounded ≈3 m), acceptance ~47% of race-window frames, covariance = K2·analytic + 1.4° lever + 0.40 m floor + 32 m cap — see §VISION-PKG2 below.** Detail: `handoff/perception-char-2026-06-08/`.

### ✅ VISION-PKG2 COMPLETE (2026-06-10, ShadowPC fable; commits 37e7ab1, 1b7e753, 9ccc88c, 4831991; suite 412 green) — measured attitude/fix covariance; yaw bias REFUTED as calibration
Vision pkg 2 / the `attitude_noise_std` lever is DONE — and it overturned two banked beliefs.
Source of truth: `handoff/shadowpc-vision-pkg2-2026-06-10/WRITEUP.md`.
- **🚩 THE ~3.6°/RANGE YAW BIAS IS NOT A FIXED CALIBRATION — REFUTED.** It decomposes into
  (a) an along-track depth-scale artifact misread as an angle, (b) gate-specific lateral offsets of
  BOTH signs (global rotation fit |e|≈1–1.7° only; per-gate fits swing −6…+8°), and (c) a roll-coupled
  wander (δ_yaw ≈ −0.52°/° roll, r²=0.87 in-flight) whose tilt-before-roll composition fit (τ_pre≈29°,
  nails it in-flight) was REFUTED by cross-validation on the June-05 task2 flight — the coupling
  FLIPS SIGN (+2.7°/°) and the correction makes that data WORSE; constant image/telemetry latency
  also refuted. Trajectory-specific ⇒ belongs in COVARIANCE, not calibration. **Follow-up:** re-measure
  the roll-wander on the NEXT fresh 6/6 flight recorded with `--dump-extras` (`composition_fit.py`
  runs as-is); if τ_pre reproduces with consistent sign across two flights, a composition correction
  becomes shippable.
- **🚩 REAL BUG FOUND+FIXED (37e7ab1): `corner_to_center`'s gate frame was rotated 180° in-plane vs
  the detector's corner convention** — invisible to position (square symmetry) but the PnP
  disambiguation prior fed to IPPE/P3P was ANTI-ALIGNED, making the frontal tie-break + P3P branch
  selection effectively random on real detections (and 3-corner association compared against
  diagonally-opposite predicted corners). Fixed in `gates_from_track_records` (axes from the
  approach-view `_frame_from_through` convention); `through_dir` + `mission._passed` invariants
  verified; convention test added. Population effect on this data nil — the value is correctness of
  the prior-dependent paths + any future use of solved rotations (predicted-corner IDENTITIES changed;
  downstream per-corner consumers inherit the corrected convention).
- **Shipped noise model (joint MLE over 165 offered fixes; exact-replication harness to 0 ulp):**
  `frames.ATTITUDE_NOISE_STD_RAD = 1.4°` — ONE constant now feeding all three former 1.0° sites
  (`localization.py`, `navigator.py`, `state_estimator.py`) — plus NEW `FIX_COV_FLOOR_STD = 0.40 m`
  isotropic floor in `localization.py` (covers the measured constant systematics: +0.3 m-high
  vertical, per-gate lateral, close-range depth bias; a σθ-only fit distorts to 3.21° and mis-shapes
  the covariance — the floor is NOT optional), plus `vision_max_range_m` 40→32 (kills the 38 m
  depth-tail leak at zero measured cost). Per-fix cov = K2·analytic(R Σ_pnp Rᵀ) + 1.4° attitude
  lever + 0.40 m floor + 32 m range cap.
- **Acceptance (canonical 6/6 recording, before→after):** good-fix over-rejection <3 m
  **15.7%→1.2%** (<1 m: 11.2%→0.0%; gate-0's 35% close-range rejection →0); catastrophic leak
  **1.06%→0.53%**; KF-accepted fixes 147→171; accepted quality held (p50 0.85 / p90 1.72 m).
  Before-baseline reproduced the published numbers bit-for-bit first.
- **UPDATED STAGE-2 RL-TWIN PERCEPTION MODEL (supersedes the 1.1% leak figure):** fix acceptance
  ~47% of race-window frames (was ~41%), leak 0.53% of solved (bounded ≈3 m), covariance =
  K2·analytic + 1.4° attitude lever + 0.40 m floor + 32 m range cap.
- **Caveats/queued:** the +0.3 m vertical constant is ambiguous (map opening-centre height vs
  camera-height offset) — disambiguating needs varied-attitude frames near one gate; a map fix would
  move planner carrots (given-pose behavior!), so it deliberately stays in covariance. Remaining
  single leak = a borderline 3.00 m fix at 25 m; documented optional trade-off
  `fix_range_rel_tol` 0.15→0.115 (cuts into the good depth-noise band — not taken).

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

### 🚩 The "+30% rate_gain band" DR proxy is DISPROVEN — ✅ REPLACED by the STATIC-MAP DR (S14, 2026-06-10)
The asymmetric rate_gain DR band **[0.90, 1.30]** (item ④, in `rl/diffaero_dynamics.py`, used by S1.3) was
the wrong model. ~~Replace with the ωₙ/ζ-envelope DR~~ — **that replacement was ITSELF superseded** (the
envelope was the LTI shadow of cc6921d's wrong anchor). **✅ DONE (S14): the band is REMOVED from
`rl/diffaero_dynamics.py` (G0 fixed at nominal under DR) and the static-map DR ships — per-env per-axis
s ∈ U[0.25, 0.35], τ ∈ U[0.015, 0.030] abs, alpha_max r/p ∈ U[200, 320] (yaw scaled ×80/260)** (see the
S14 section below).

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
  lag τ≈0.020 + alpha_max slew clamp), **parity-gated twin → rl_plant → torch adapter — ✅ IMPLEMENTED
  (S14, 2026-06-10, next section).** The max-clamp concern resolved without raising it: `max_omega_rps`
  25.0 already sits above the map's DC ceiling (~11.2 rad/s/axis) — never bites. **DR: s ∈ [0.25, 0.35],
  τ ∈ [0.015, 0.03], alpha_max ∈ [200, 320]** — replaces both the +30% band and the ωₙ/ζ-envelope plan
  (✅ shipped in S14). **UPSIDE: real authority at full stick is ~11 rad/s, not 7.85 — a HIGHER speed
  ceiling for VQ2.**
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
  authority by up to 42% at full stick) → retrain on the map-ON plant — **gating now SATISFIED (S14
  integration done)**. The S13-LIVE transfer test is **DEMOTED from disambiguator to optional cheap
  datum** (fly opportunistically, non-blocking — the sweep answered the plant-fidelity question
  directly). **Critical path: static-map integration ✅ DONE (S14, 2026-06-10) → env/reward redesign +
  retrain = S1.4, NEXT.**
- **Durable sim/ops facts:** (1) **🚩 THIRD sim idle state** — after ~90 min idle post-race the sim parks
  on an off-race screen where MAV_CMD 31000 is a NO-OP while telemetry still streams (stale RACE_STATUS
  `started=True`, frozen sim_time); recovery = Win32-focus the AI-GP window + Enter ×2 — wired as
  automatic escalation in `scripts/rate_sysid.py` (15+ runs chained, zero GUI touching). (2) **Yaw gain
  is maneuver-dependent**: ~2.35 level (looks like a ~7.4 rad/s cap) vs ~3.1 tumbling — caveated, own
  pass only if it ever matters (racing yaw cmds are small). (3) All probes near hover; ~~aero at racing
  airspeed unmeasured~~ — ✅ since MEASURED (TWIN-FALSIFY 2026-06-11): the rate map is airspeed-INVARIANT
  but the twin's linear drag + linear collective are FALSIFIED at speed/full-stick — see §TWIN-FALSIFY. (4) `rate_sysid.py` gained `--vel-damp`/`--pos-pull` station-keeping + `--mode anomaly`.

### ✅ S14 STATIC-MAP INTEGRATION COMPLETE (2026-06-10, fable; commits f730bd6, b14ca2e, 77a0186) — critical-path step 1 DONE
Sweep WRITEUP §3 implemented across all three plant implementations, parity-gated at every seam.
Report: `handoff/laptop-s14-staticmap-integration-2026-06-10/REPORT.md`. **Tests 376→409.**
- **twin.py**: `CtbrPlantConfig` gains `super_rate_s` + `alpha_max_rps2`; g = rate_gain/(1−s·min(|cmd|,π)/π),
  per-step omega increment clamped ±alpha_max·dt. Defaults None ⇒ bit-identical legacy (test-pinned).
  `max_omega_rps` stays 25.0 — never bites (the map's DC ceiling is ~11.2 rad/s/axis).
- **twin_fit.py**: `faithful_config(super_rate=True)` = the measured map-ON config (s=0.30,
  alpha_max=[260,260,80]); default False = the exact legacy faithful twin.
- **rl_plant.py**: mirror (omega bit-identical to twin); exports `SUPER_RATE_S_MEASURED` /
  `ALPHA_MAX_RPS2_MEASURED` as the canonical nominals (DR centers).
- **diffaero_dynamics.py** (torch): mirror; the disproven +30% rate_gain band REMOVED (G0 fixed at nominal
  under DR), replaced with per-env per-axis s∈U[0.25,0.35], τ∈U[0.015,0.030] abs, alpha_max
  r/p∈U[200,320] (yaw scaled ×80/260 ⇒ U[61.5,98.5] — same relative width around yaw's own nominal);
  map+slew ALWAYS ON under DR. Also ports the params-level `transport_delay_steps` ring buffer to BOTH
  backends (**S12 item 4a closed** — it had been a silent no-op: the adapter passed `act_buf=None`).
  Wrapper latency DR untouched; measured input delay 5–15 ms < one 33 ms control step — do NOT
  double-count delay into τ.
- **Gates:** gain-table anchor within +1.6–2.9% of measured at all 8 r/p points (within the documented
  ~3% one-param fit); twin↔rl_plant omega BIT-identical (60-case parity battery); **V100 gate (job
  3265870): legacy 8.9e-16 / super_rate 1.8e-15 / delay2 8.9e-16 / map_delay 1.8e-15 — PASS** (acceptance
  ≤1e-6); negative controls prove teeth (1e-6 s-bias caught at 8e-5; dropped delay caught at 13.7).
  `check_diffaero_gate.py` is now a config-matrix gate ({legacy, super_rate, delay2, map_delay} ×
  {float64 gate, float32 advisory}); NEW `rl/local_gate_harness.py` dry-runs it on laptop CPU-torch
  BEFORE any Adroit roundtrip.
- **Resolved ambiguities:** s/alpha_max DR are per-axis (n,3); the level-attitude ~7.4 rad/s yaw cap is
  comment-documented in all three files (NOT modeled; own measurement pass only if racing yaw cmds grow).
  The optional 2nd-order refinement (~10% small-signal overshoot fidelity) was SKIPPED per WRITEUP §3's
  own "static map is the load-bearing fix"; τ default stays 0.019, the DR band [0.015,0.03] covers the
  saturated τ_eq 25–33 ms.
- **🚩 CAVEAT:** eval/rollout scripts (`peregrine_eval.py`, `offline_rollout.py`) still DEFAULT to the
  legacy FLAT plant — correct for the flat-trained checkpoints (inc-1/S1.3), but **the S1.4 retrain + all
  future evals must switch to map-ON** (construct `PlantParams(super_rate_s=SUPER_RATE_S_MEASURED,
  alpha_max_rps2=ALPHA_MAX_RPS2_MEASURED)`, or just enable DR — DR forces the map on).
- **🚩 ADROIT OPS:** `/scratch/network/fl3689/peregrine_repo` is a FILE COPY, not a git clone (the GitHub
  repo is private) — S14 pushed via md5-verified base64 over the x daemon; stage a real clone with a
  deploy key if Adroit sessions grow.

### ✅ TWIN-FALSIFY CAMPAIGN COMPLETE (2026-06-11, ShadowPC fable) — quad drag + convex collective FALSIFY the twin's aero; S16 integration queued
7 probes, predictions committed BEFORE any probe flew (9c240df/cbe9215) — the falsification discipline
held. Source of truth: `handoff/shadowpc-twin-falsify-2026-06-10/WRITEUP.md` (commits cbe9215+0a52bc2+9684ae1).
**TWO FALSIFICATIONS, four survivals:**
- **🚩 LINEAR DRAG FALSIFIED.** The twin's `linear_drag 0.2111/s` world-isotropic is WRONG: real sim drag
  is **QUADRATIC, body-direction-dependent** — c2≈0.052/m (0.042 nose-first / 0.058 tail-first /
  0.055 lateral / 0.076 climb / 0.054 descend). At 9 m/s the sim brakes ~4.2 m/s² where the twin says 1.9
  (**2.2× wrong, growing with speed**). Coast-replay speed RMS 0.81→0.24 m/s with the measured model.
  **Policies trained on linear drag have learned ~2× UNDER-BRAKING at speed.**
- **🚩 LINEAR COLLECTIVE MAP FALSIFIED.** The real thrust curve is **CONVEX**: full-stick vertical accel
  ≈78 m/s² = 2.12× the linear model's 37; sub-linear below hover; motor witness 1:1 ⇒ thrust physics, not
  motor lag. **CORRECTS the banked "live collective≤1.0 ≈ 3.765 normed ≈ 3.77 g ceiling" — that figure
  was a LINEAR-MODEL artifact; real full-stick is ~8 g.** Implications: (a) the S1.3/S1.4 training thrust
  model (max_normed_thrust=3.765 linear) under-states top-end authority ~2×; (b) ~~the TOGT time-optimal
  bound was computed with T/W≈3.77 ⇒ it is CONSERVATIVE~~ — **❌ CORRECTED (TOGT 2026-06-11): that was
  thrust-only reasoning; the corrected-aero case (quad drag + T/W 8) gives 4.71 s — the v² drag wall eats
  the doubled thrust, ceiling ROBUST ~4.3–4.7 s (§TOGT-BOUND). Authoritative re-run still queued post-S16.**
- **Survivals (4):** rate map **airspeed-INVARIANT** (ratios 0.95–0.98 of the hover map at 6 m/s, all
  magnitudes); **control-rate invariant** 50/100/200 Hz (±0.2%); **NO battery sag** (−0.02% over 8 min);
  **determinism** run-to-run SD ~0.03 m/s.
- **Ready-to-port `CandidatePlant`** (quadratic body-frame drag + collective knot table) in the handoff's
  `replay_twin.py`; proposed DR bands in WRITEUP §7. **Integration = S16, a separate parity-gated session
  (same twin→rl_plant→torch recipe as S14). Retrain gating now extends to AERO: any checkpoint trained
  pre-S16 under-brakes ~2× at speed.**
- **Caveats:** drag above 7.6 m/s extrapolated (quad form unverified there; arena corridor proven clear
  to ~43 m — a longer pass is possible); fwd/back anisotropy rests on 2 forward runs.
- **Durable sim-ops facts:** (a) **🚩 ZOMBIE DUAL-INSTANCE mode** — two sim instances both streaming
  MAVLink to 14550 → pymavlink send-peer flaps between them → drone ARMS but IGNORES all commands;
  killing the zombie kills BOTH (shared launcher) → relaunch fresh via the documented login chain
  (WRITEUP §0). (b) **Measurement footgun:** the harness's quaternion-finite-difference rate channel
  ALIASES against the LPN/ODO telemetry stagger (produced a false "rate gain +20% at airspeed/100 Hz" —
  refuted); use raw ODOMETRY rate + euler-slope cross-check (WRITEUP §6). (c) `rate_sysid.py` gained
  `--mode profile` (JSON phase schedules) + a stale-collision fix.

### ✅ TOGT TIME-OPTIMAL BOUND COMPLETE (2026-06-11, laptop fable) — ceiling ~4.3–4.7 s; THRUST binds; reference line shipped
Stack-review meta-gap ② CLOSED. TOGT-Planner + multiple-shooting refine on WSL; pipeline `scripts/togt/`
(re-runs in minutes for new tracks/plants). Source of truth: `handoff/laptop-togt-bound-2026-06-10/WRITEUP.md`
(commits 35f451d+c4a8134+00d7cfe). Tests 457 green.
- **THE BOUND: ~4.27 s** (standing start → gate-5 plane, crossings within the inscribed circle = Euclidean
  miss <0.75 m, the validity rule as we measure it); **4.13 s** if the full 1.5 m square counts. **Our VQ1
  35.3 s is 8× off the ceiling.** Even the twin-TRACKABLE lap (8.3 s, below) is 4.2× faster than VQ1 —
  the VQ2 rank war happens far below 35 s.
- **THRUST BINDS, overwhelmingly:** collective rides the ceiling ~84% of the lap; 75%/50% thrust costs
  +0.77/+2.29 s. Body rates barely matter (the old wrong 7.85 rad/s model costs only +0.05 s; unbounded
  rates buy −0.04 s) — **the super-rate map's value is control fidelity, NOT lap time.** Drag costs
  0.085 s on the linear model; vmax≈52 m/s.
- **🚩 CORRECTS the §TWIN-FALSIFY "TOGT bound is CONSERVATIVE" note (thrust-only reasoning — WRONG):**
  the exploratory corrected-aero case (quadratic drag c2≈0.052 + T/W 8) gives **4.71 s — the v² drag wall
  (~39 m/s top speed) eats the doubled thrust almost exactly; the ceiling is ROBUST at ~4.3–4.7 s** to the
  plant revision. Authoritative re-run queued post-S16 (v² extrapolated beyond 7.6 m/s; the pipeline
  re-runs in minutes).
- **REFERENCE LINE SHIPPED: `rl/reference_line_vq1.json`** (margined circle-gate solution, lap 4.551 s,
  crossings ≤0.14 m from centre = 0.6 m tracking budget) + `rl/reference_line.py` loader with arc-length
  `progress()` for the RL progress reward.
- **TWIN-TRACKED REALITY CHECK:** the existing geometric controller CANNOT track a thrust-saturated
  reference at 1× (zero recovery headroom, diverges by design); time-dilating the geometry, first valid
  6/6 at **k=1.85 ⇒ twin-tracked lap ≈8.3 s**, stable across 0–40 ms latency; a 54-combo gain sweep finds
  nothing faster — **the 4.55→8.3 s gap is STRUCTURAL**, the tracking architecture's to close (MPCC /
  RL-as-tracker / monolithic RL with progress reward). Directly informs the pending S2
  monolithic-vs-decomposed decision.
- **OPEN VALIDITY QUESTION → cheap live probe queued (ShadowPC):** the optimum clips every gate corner
  when allowed (~1.06 m Euclidean, per-axis ≤0.76 m) — whether the sim's race_outcome accepts corner
  passes is UNVERIFIED; a one-off live corner-pass probe closes the 4.13-vs-4.27 bracket and decides how
  aggressively lines may cut corners.
- **Ops keepers** (in `scripts/togt/README`): CRLF segfaults TOGT's YAML parser; the standalone CLI
  crashes pre-main → use the gtest driver; MSYS path mangling when driving WSL from Windows.

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
  **Stage 1 increment 1 ✅ COMPLETE (2026-06-09, job 3261393):** CTBR policy threads 6-gate course on our plant, given pose, zero vision. racing-PPO n_envs=2048, 5000 updates → success_rate 0→0.97, ~89K env-steps/s (A100, 30:48 wall). obs_dim=17 (vel+quat+gate_relpos+gate_normal+body_rates+collective), action=CTBR, DR configured (rate_gain±10%/hover±5%/drag±30%/τ±30% — 🚩 but see the S1.3 correction below: the numpy backend silently ignored it, inc-1 effectively had NO plant DR). Checkpoint `stage1_inc1_actor.pth`. Caveats: (1) over-aggressive (l_ep ~1.9s, no speed shaping — VQ2 target); (2) real test = live sim transfer.

  **✅ S1.2 — increment-1 LIVE DEPLOYMENT (2026-06-10, fable session): pipeline VERIFIED end-to-end, checkpoint NOT transfer-ready (flight 1: 0 gates, tumbled into gate-0 post). Detail: `handoff/shadowpc-s12-rl-live-2026-06-10/`.**
  - **🚩 CORRECTED DEPLOYMENT RECIPE (fixes 3 sonnet S1.1 bugs — the verified-correct way to deploy a DiffAero-trained actor live, in `rl/fly_rl.py`):** ① actor output = **tanh(mean) → rescale** to thrust [0,5] / rates ±3.14 rad/s (NOT raw actor mean); ② FLU→FRD action sign = **[1,−1,−1]** (sonnet's [−1,+1,+1] wrong on all 3 axes; the plant applies rate_gain·rate_sign identically in train + live — no ff/rate_gain algebra); ③ collective obs init = **0.0** with **rescaled** feedback (NOT 1.0); ④ training control rate = **30 Hz** (racing.yaml dt 0.0333; NOT 100); ⑤ training **resets at rest** 1 m in front of a random gate (NOT "racing velocity"). Verified-correct from sonnet: obs layout (17), `R_W2G=diag(−1,−1,1)`, gate yaws all π, final-gate clamp, Euler-ZYX, actor arch (NormedLinear [256,128]).
  - **🚩 "OOD-at-start = root cause" RETRACTED** — an artifact of sonnet's wrong action transfer function, not a real diagnosis; reset saturation is the policy's NORMAL launch behavior (it saturates in training too, then modulates).
  - **Tail-first spawn bug + fix:** policy trained identity-reset + all gates at yaw π ⇒ flies the course tail-first; sim spawns nose-first on a 17.8°-tilted pad ⇒ ~180° attitude-OOD. Fixed with a **virtual π body-z flip in fly_rl.py** (`--virtual-flip`, default ON; exact rigid-body symmetry). Offline: handoff-state rollouts 0/6 → 6/6 under training physics.
  - **🚩 THE REAL transfer failure (the key finding) = the policy is a "backflip-diver":** its NOMINAL twin maneuver rolls through **104–126° before every gate** (fine offline). Live the maneuver diverges — realized rates hit **9.7 rad/s vs the twin's first-order 7.85 ceiling** (=3.14·2.5; ❌ diagnosis SUPERSEDED 2026-06-10: unmodeled static super-rate DC gain, NOT transient overshoot — see the characterize-sweep section), it blows through ±180° tilt (the then-suspected "yaw-spin anomaly" — **DISSOLVED by the sweep**: super-rate gain + Euler yaw-flip artifact + a gate-post COLLISION), and tumbles into the gate post within 0.35 s. NOT reproducible in the twin (replay from the exact live handoff state passes gate 0 at any latency ≤100 ms and gain ×1.24) → **no deployment-side knob fixes it.** The BRIDGE worked perfectly: CTBR delivered the drone dead-centre (dy +0.04, dz +0.05 m) at 5.1 m/s, 3 m before gate 0; the policy's first action = exactly the offline-twin prediction.
  - **OOD verdicts (offline, measured via `rl/offline_rollout.py`):** training reset (1 m, rest) = **6/6 finish 3.3 s** (validates ALL deployment math); raw standing start (23.3 m, rest) ≈ 0–1/6 (arrives at gate 0 at 33–44 m/s, unrecoverable); **bridge handoff + virtual flip = 6/6 under training physics, 4/6 under the live collective ceiling** — the gate-4 wall is purely the thrust clip (trained max_normed_thrust 5.0 ≈ 5 g vs live collective≤1.0 ≈ 3.765 normed ≈ "3.77 g" — ❌ that ceiling FALSIFIED 2026-06-11: a linear-model artifact, real full-stick ≈8 g via a CONVEX thrust curve; see §TWIN-FALSIFY), unfixable at deployment, fixed in retrain.
  - **🆕 UNATTENDED SIM CONTROL ACHIEVED here (the user's standing ask; durable capability for ALL future sim work):** **MAV_CMD 31000** (`client.send_sim_reset()`) restarts the race once a race context exists (fresh ~3 s countdown) — **NO-OP from HOME** (no telemetry there). From HOME: Win32 `SetForegroundWindow` to focus the `AI-GP` window (`WScript.Shell.AppActivate` alone returns False) + Enter twice (home → waiting room → race+countdown). The session **cold-launched FlightSim.exe and raced with NO human.** `rl/fly_rl.py --flights N` chains attempts (never resets into a ticking countdown).
  - **✅ S1.3 RETRAIN SPEC (Path C, ~30 min A100, `rl/peregrine_racing_s13.sbatch` staged):** ① `dynamics.controller.max_normed_thrust=3.765` (live ceiling; one-line override); ② **standing-start resets** (`+env.standing_start_frac`, reset at ~23 m from gate 0 at rest = real race start; **implemented** cfg-gated in `peregrine_racing.py`, spawn pose mapped through the deployment virtual flip → fly_rl.py needs no change; removes the CTBR bridge, bridge stays as fallback); ③ **tilt/jerk regularization** (`reward_weights.quadrotor.attitude≈2.0`/`jerk≈0.3`, sweep — kills the >90°-roll style; then framed as anomaly avoidance, NOW (510da24) justified for smoothness/VQ2 style + staying in the well-modeled envelope only — no anomaly exists; + the VQ2 aggression fix); ④ **latency DR** (port the transport-delay ring buffer to the torch backend of `rl/diffaero_dynamics.py` — rl_plant already supports it, parity-checkable via `check_against_rl_plant` with `transport_delay_steps>0`; DR per-env delay ∈ {0,1,2}) + asymmetric rate-gain DR band **[−10%,+30%]** (cheap transient-overshoot proxy — ❌ later DISPROVEN; ✅ REMOVED + replaced with the static-map DR in S14, see above). Reward shaping (smoothness/time) folds in here.

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

  **🆕 NEXT = S1.4: ENV-COHERENCE REDESIGN + RETRAIN on the map-ON plant (user directive 2026-06-10) — now FULLY UNBLOCKED (sweep 510da24 done = no anomaly boundary to encode; S14 done = measured static-map DR in place).**
  The user flagged that the **reward CONFLICTS with the termination**, and we need **more reward terms with
  more explicit logic on how each works.** Reframe the next retrain as redesigning the ENV: reward +
  termination designed TOGETHER + the **static-map DR** (s/τ/alpha_max — ✅ shipped in S14, just enable it)
  + **COLLISION-based crash-termination** (DiffAero's racing env already has it; NOT a tilt threshold —
  ~~crash-termination at the measured anomaly boundary~~ superseded, no anomaly exists). Hand to a dedicated
  **fable + adroit-connector** session (one session, scope already banked). Non-trivial,
  objectively-checkable → good fable fit. Remember the S14 caveat: training configs + evals must run
  map-ON (DR forces it; eval scripts still default flat).
  **🆕 Scope now ALSO includes procedural track randomization + VQ1-course-as-held-out-eval** (STACK-REVIEW-VQ2
  meta-gap ①, see that section — potentially binary for the unseen VQ2 course).
  **🚩 2026-06-11 (TWIN-FALSIFY): S16 AERO integration (quadratic body-frame drag + convex collective knot
  table, §TWIN-FALSIFY) inserts into the critical path BEFORE the final retrain — parity-gated separate
  session, same twin→rl_plant→torch recipe as S14. Retrain gating now extends to aero: any pre-S16
  checkpoint under-brakes ~2× at speed and trains under a phantom 3.77 g ceiling (real ~8 g).**

  **🆕 PENDING ARCHITECTURE DECISION (2026-06-10 discussion) — S2 = decomposed plan-line + RL-tracker?**
  Consider making S2 a **decomposed plan-line + RL-tracker** (plan an explicit smooth/feasible racing line;
  RL only TRACKS it) INSTEAD of another monolithic racer. Rationale: the monolithic policy's freedom to choose
  the trajectory is what produced the **backflip-dive** — an explicit line makes inversion STRUCTURALLY
  impossible. Tradeoff: lower speed ceiling than SWIFT-style learn-the-line, bought back via the offline
  line-iteration flywheel. **DECISION PENDING the post-static-map retrain result** (the S1.3 live datum is
  now optional — see the characterize-sweep section). **🆕 TOGT datum (2026-06-11, §TOGT-BOUND): the existing
  geometric tracker needs k=1.85 time-dilation to track the optimal line (twin lap ≈8.3 s vs the 4.55 s
  reference; 54-combo gain sweep finds nothing faster) — the gap is STRUCTURAL ⇒ whichever S2 wins must close
  it (MPCC / RL-as-tracker / monolithic RL with progress reward over the shipped reference line).** (Mapping is PERCEPTION — conservative lap +
  detector→PnP→KF — NOT an RL task, and only needed if VQ2 hides the map; open organizer question.)
- **Stage 2**: layer the MEASURED perception-noise model (asymmetric actor-critic: privileged critic sees
  truth, actor sees noisy perception-state) + eval the policy driven by REAL YOLO→PnP→KF with **given-pose
  OFF** in VQ1 sim ← the right home for the user's "test the control policy with real YOLO vision."
  **Use the VISION-PKG2 model (2026-06-10) for the twin injection — leak 0.53% of solved, acceptance
  ~47%, cov = K2·analytic + 1.4° lever + 0.40 m floor + 32 m cap** (supersedes both the 1.6% and 1.1%
  figures). The `attitude_noise_std` lever is ✅ SHIPPED (see §VISION-PKG2 above).
- **Vision engineering workstream (b) COMPLETE for VQ1+Stage-1** (commit 4673517, 2026-06-09).
  **✅ Vision pkg 2 (`attitude_noise_std`) DONE 2026-06-10 — measured covariance model shipped; see
  §VISION-PKG2 above** (still inert for given-pose nav; given-pose tracking untouched, vq1 flow unchanged).
- **Vision engineering (a) runs PARALLEL** (detector training, Adroit, decoupled by design).

## STACK-REVIEW-VQ2 (2026-06-10, fable, report-only — full report: handoff/stack-review-2026-06-10/REPORT.md)
Adversarial whole-stack architecture review (user ask: "is YOLO-pose corners the best we can field? Depth
Anything 3? VSLAM?"); 2025–26 landscape swept, settled-decision ledger honored, verdicts grounded in our
measured anchors. **HEADLINE: architecture AFFIRMED — no component needs replacement.** The 2025 A2RL
champion stack (**MonoRace**, TU Delft/MAVLab, arXiv 2601.15222) validates our exact class (monocular gate
perception + known-geometry calibration refinement + learned controller). The rank-relevant exposure is not
in any component but in **3 META gaps**:
- **① Single-track training vs unseen VQ2 course — potentially BINARY; rank-impact #1.** We train on the
  ONE VQ1 course; VQ2 is an unseen track run unattended in THEIR eval (a one-track policy may score zero).
  Obs are already gate-relative (translation-invariant) — the architecture is ready, only the training
  distribution isn't; literature is unambiguous that randomized tracks generalize (2411.04246, 2512.09571,
  Environment-as-Policy 2410.22308). **FIX = procedural track randomization** (sample spec-plausible 6-gate
  layouts) **+ VQ1-course-as-HELD-OUT-eval** — first experiment: 1k random courses, S1.3 recipe, report
  held-out-VQ1 success vs the single-track baseline. **FOLDED INTO the env-coherence redesign scope.**
- **② No time-optimal bound for the VQ1 course** (35.3 s vs WHAT denominator?). **Upgrade #2 =
  TOGT-Planner** (FSC-Lab + Run-TOGT-Planner Python wrapper; plans through the gate OPENING — frees the
  crossing point, where corner-cut time lives) **+ CPC (Foehn 2021) as the true offline bound.** Serves
  three masters: the gap meter, the RL progress-reward reference, and the explicit line if decomposed S2
  wins. **✅ CLOSED (2026-06-11): bound COMPUTED — ~4.27 s, ROBUST ~4.3–4.7 s under the corrected aero
  (the "T/W≈3.77 ⇒ conservative" worry resolved: the v² drag wall eats the doubled thrust; thrust binds,
  rates barely matter); our 35.3 s is 8× off; reference line + progress() shipped
  (`rl/reference_line_vq1.json` / `rl/reference_line.py`); twin-tracked lap ≈8.3 s = a STRUCTURAL tracking
  gap. See §TOGT-BOUND.**
- **③ Vision-only (VQ2 case C) readiness pieces UNBUILT** (cheap, champion-validated, useful in A/B too):
  **delayed-fix KF rewind ring buffer** (apply fix at capture time — exact + cheap for a linear KF; at
  15 m/s a 50 ms stale fix mis-applied "at now" = 0.75 m; buys more than any factor graph);
  **range-anisotropic R** (depth-axis ∝ r² in the gate-bearing frame — dilution-of-precision is the correct
  model for the 2 residual long-range leaks, NOT learned depth, NOT tighter gates); ~~offline yaw-bias
  calibration solve~~ — **✅ RESOLVED-AS-REFUTED (VISION-PKG2, 2026-06-10): there is NO fixed yaw rotation to
  solve for** (both-signed per-gate offsets + flight-specific roll wander that fails cross-validation) — the
  shipped covariance model covers it; the MonoRace-style solve only becomes live again if the roll-wander
  τ_pre reproduces with consistent sign across two flights (see §VISION-PKG2 follow-up);
  **in-loop perception latency measurement** (🚩 ShadowPC torch is **CPU-ONLY** — decode→detect→PnP→KF has
  NEVER been measured in-loop; YOLO11s@640 on CPU plausibly 30–80 ms; first-order unknown for vision-only VQ2).
  Rewind buffer + anisotropic R + latency measurement remain OPEN; the yaw-calibration sub-piece is closed.

**Secondary keepers:**
- **Learned residual dynamics on the twin** (UZH Learning-on-the-Fly, 2508.21065): fit a small
  (state,action)→accel-residual model on course recordings — closes the ~25% latency under-model + the
  ~~unmeasured~~ racing-airspeed aero (since measured directly — TWIN-FALSIFY 2026-06-11; S16 closes it
  analytically, residual model = the mop-up), the last transfer-gap term. Additive module BEHIND the parity gate, OFF
  by default; first experiment = one-step prediction RMSE vs the analytic twin on a held-out run.
- **YOLO26-pose at the photoreal v4 retrain** (NOT before): RLE per-keypoint σ feeds weighted-PnP directly
  (replacing conf-derived σ), NMS-free deterministic latency; same pipeline/sbatch, near-zero marginal cost —
  train v4 as YOLO26-m AND YOLO11s, A/B on the fixed eval set. Doctrine stands: never fine-tune on clean VQ1 frames.
- **Runtime validity supervisor**: watchdog demoting policy→model-based floor on divergence
  (tilt/track-error threshold) for the unattended they-run eval; ~1 day; optional pending submission mechanics.
- **Offline gate-landmark mapper skeleton** (scipy `least_squares`, GTSAM only if conditioning bites):
  serves VQ2 cases B+C, the one missing infra piece used in 2 of 3 cases; a weekend, zero risk — pre-build now.
- **Reviewer's read on the pending S2 architecture decision** (decided on live data, not this review):
  retrain MONOLITHIC first on the corrected plant — the backflip-dive root cause (wrong plant + no DR +
  reward/termination conflict) is understood and fixed; decomposition = the fallback if style pathologies
  persist. The TOGT line is cheap and needed either way.

**Examined and REJECTED (do not re-litigate without new evidence):**
- **DA3 for gate depth** — re-verified, HOLDS: metric mono-depth error at 25–38 m ≥ the PnP noise it would
  replace; known-size+intrinsics PnP is the geometrically correct tool. DA3 = offline OBSTACLE flywheel only.
- **RT-DETR / D-FINE / RF-DETR** — bbox-first, no 4-keypoint pose path beating ultralytics-pose; YOLO26 is the in-family upgrade.
- **Foundation-model distillation for corners** — solves an open-vocabulary problem we don't have.
- **PVNet-class learned 6DoF gate pose** — parked; measured failures (scale-error boxes, yaw calibration) aren't what dense voting fixes.
- **Tightly-coupled VIO (OpenVINS/VINS-Fusion)** — attitude is given; case-C bounded gate-SLAM covers the rest.
- **GTSAM in-loop** — offline mapper only; linear KF + rewind buffer is near-optimal for a linear problem.
- **Isaac Lab re-promotion** — no 2026 development changes the calculus vs DiffAero+injected-plant (88.9K steps/s, machine-epsilon parity).
- **Crazyflow switch** — JAX throughput we don't need; re-pays the whole plant-injection + parity cost.
- **In-loop MPPI reference-free racing** (2509.14726) — needs GPU rollouts inside the live control loop on
  unknown eval hardware; our RL policy is the same objective baked offline. Revisit only if RL transfer fails twice.
- **Direct motor commands / G&CNets** (the MonoRace control layer) — interface-impossible: we command CTBR
  into a black-box stabilizer; take their calibration practice, not their actuator level.
- **Pixel-to-control RL** — re-affirmed; even 2025–26 "vision-based" racing papers (2512.09571) feed DEPTH,
  not RGB, and still need privileged→visual curricula.
- **Conditional EXPERIMENT (not rejected):** tightly-coupled corner-reprojection EKF (arXiv 2603.02742 —
  handles 2-corner transit frames without PnP) ONLY if VQ2 is vision-only AND transit dropout is measured to hurt.

**New measured-data gaps exposed:**
1. **In-loop perception latency** — no number exists anywhere (ShadowPC torch CPU-only); blocks honest Stage-2 latency injection.
2. **Submission-eval state persistence between attempts** (map, tuned line carried over?) — decides whether
   track-randomized generalization is MANDATORY (no persistence) vs insurance (persistence). **ADD TO THE ORGANIZER EMAIL.**
3. **Eval-hardware compute envelope** (does our stack get a GPU?) — gates detector sizing + any in-loop
   learned component; do NOT size to 100 TOPS for virtual quals (that governs the physical round). **ADD TO THE ORGANIZER EMAIL.**
4. ~~**Aero at racing airspeed**~~ — ✅ MEASURED (TWIN-FALSIFY 2026-06-11): quadratic body-frame drag c2≈0.052/m, rate map airspeed-invariant; linear drag + linear collective FALSIFIED (§TWIN-FALSIFY; >7.6 m/s extrapolated).
5. **Zero VQ2 photoreal frames exist** — the detector's true VQ2 axis (appearance) is unmeasurable until
   organizers release anything — exactly why the DR doctrine must not be diluted.

## #1 ORGANIZER ASK (user offered to email info@theaigrandprix.com)
**"In Round Two (VQ2), does the sim still stream LOCAL_POSITION_NED / ODOMETRY (drone position+velocity),
or is position vision-only?"** — architecture-defining: if VQ2 gives pose, the vision→map→path→RL risk
largely evaporates (RL flies on given pose; vision just confirms gates). The spec is SILENT (grepped: says
"GPS not available / no absolute global position" but nothing on VQ2 LOCAL_POSITION_NED). Secondary: the
submission interface spec + VQ1 deadline + confirm registration active. **🆕 (STACK-REVIEW-VQ2): + ④ can the
stack carry state (map, tuned line) BETWEEN attempts in the controlled eval? (decides track-randomization
mandatory-vs-insurance) + ⑤ the eval-hardware compute envelope — does the submitted stack get a GPU?**

## Open
- ~~Substrate bake-off verdict~~ ✅ RESOLVED: **DiffAero** — proven on Adroit (plant injected, gate PASS, trains our plant).
- Structural pilot-stack changes (user brainstorming — the Setpoint/ControlCommand seam keeps a
  planner+controller→policy swap low-risk).
- VQ2 data-stream answer (gates the map/SLAM + vision-load-bearing question).
