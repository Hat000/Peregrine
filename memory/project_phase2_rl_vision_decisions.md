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
  **⚠️ SUPERSEDED (VISION-FRAME-FIX 2026-06-12, 8d7b0b3):** The +3.5°/range east bias was primarily
  the navigator's aliased yaw — `navigator.py:295` used `R_world_from_body(ds.roll, ds.pitch, ds.yaw)`
  where `ds.yaw` is the negated true yaw (R_y(π) alias). Fixing to
  `R_world_from_odo_quat_wxyz(ds.orientation_ned_wxyz)` eliminates east bias to 0.0°/range and reduces
  north bias −2.0°→−0.4°/range. The 1.4° `ATTITUDE_NOISE_STD_RAD`, 0.40 m cov floor, and 32 m range
  cap remain correct. sigma_theta re-fit **INCONCLUSIVE (P2-OFFLINE-ANALYSIS, 2026-06-13):** at-speed MLE = 0.46° on N=28 fixes (681 frames → 36 depth-sane → 28 MLE-eligible) — underpowered (prior used N=165); floor collapsed to 0.00 m; +0.67 m systematic N-bias deflates estimate. **1.4° + 0.40 m floor STAND** (escape hatch invoked). Directional: σ_theta likely smaller at speed; refit DEFERRED pending N≥100 at-speed fixes (inc8-class faster flights). 🆕 +0.67 m N-bias corroborates gate-relative plan (per-track world-frame bias drops out in gate-relative obs).
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
- **✅ S16 AERO INTEGRATION COMPLETE (2026-06-11, laptop fable; commit eba4349; report handoff/laptop-s16-aero-integration-2026-06-11/REPORT.md):** quad body-frame drag + convex collective knot table integrated through all three plants (twin→rl_plant→DiffAero adapter), S14 recipe, defaults OFF ⇒ legacy bit-identical (pinned by inline-reference test). Suite 457→497 green. Parity: local CPU-torch gate worst 8.9e-16; twin↔rl_plant 80-case battery: omega+thrust exactly 0.0, pos/vel ≤5.1e-13. Anchors reproduce measurements exactly: per-direction decel 3.40/4.70/4.46 m/s² at 9 m/s; all 12 collective knots exact; §3 ratio column 0.38→2.12 within ±0.005. Notable deviations (8 total; key 3): bottom knots FLOORED at 0 (extrapolation gave −5.6 m/s² at zero stick, contradicting free-fall); c2 DR band per-slot RELATIVE; `dr_aero` is a separate OPT-IN flag (S15 trained concurrently — folding would have changed their plant mid-campaign). 🚩 Adroit V100 gate NOT yet run (connector contention; `run_parity.sh` md5 tripwire updated — the next Adroit session MUST run it FIRST). Successor enables aero training with one flag: `+dynamics.dr_aero=true`. Proposed DR bands in WRITEUP §7. **Retrain gating: the definitive retrain is now gated ONLY on S15 finishing.**
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

## ✅ GATE MAPPER COMPLETE (2026-06-11, laptop fable; commits 46a4cc4→75fa42a; 528 tests green, 31 new) — closes stack-review keeper "offline mapper skeleton (cases B/C)"
Source of truth: `handoff/laptop-gate-mapper-2026-06-11/WRITEUP.md`.

### Implementation
- **`src/racer/gate_mapper.py`**: unified mapper for all three VQ2 cases.
  - **Case A / B (pose-aided)**: per-gate robust averaging (median centre + MAD), χ² rejection (k=4, matches live KF gate), per-gate covariance. B-mode: rough-prior fusion with configurable trust + consistency gate + low-n down-weighting.
  - **Case C (no pose, attitude given ⇒ LINEAR)**: sparse exact-Jacobian least-squares (~10 landmarks + few hundred poses, 2–22 s solve), yaw by robust circular averaging. Two-stage solve: linear first (56 m→2 m unlock for the robust-loss saturation pathology), then robust re-weight.
  - **Output**: serializes to the EXACT `capture_track_map.py` schema — `navigator.load_track_map` consumes mapper maps and given maps identically (round-trip 1e-6 m; navigator untouched). Also ships: `gate_mapper_synth.py` (measured-noise synthetic validation), `scripts/run_mapper_offline.py` (CLI), `scripts/validate_gate_mapper.py` (parameter sweep).

### Validation vs 0.75 m validity half-opening (mean/worst over seeds)
- **Case A**: 1 lap = 0.63/0.70 m max; in-plane 0.36 m (bias along through-axis); with measured-bias correction 0.20/0.25 m. Flat across leak 0→5% and association-error 5→15%.
- **Case B**: 1–3 m rough prior → ~0.6 m in one lap (3–5× gain over prior). Starved quarter-lap: correctly no-harm (failsafe holds prior).
- **Case C**: 1.8–3.4 m aligned; yaw ≤3°; bounded; pathologies flagged.

### Three durable findings
1. **Consecutive-gate co-visibility IS GEOMETRICALLY IMPOSSIBLE** on this course (23.7–38.5 m spacing vs 24–32 m camera range) — case-C backbone = motion prior + velocity dead-reckoned init; recommended exploration maneuver = a deliberate pre-transit scan nod.
2. **Case C has a noise-independent ~1.5 m floor** (blind-transit corner-cut) — a case-C map is a SHAPE ESTIMATE to localize against, not survey-grade; use case A/B once pose is available.
3. **Two failure modes found+fixed**: (a) robust-loss saturation stranding far components → linear-first two-stage solve (the 54 m→2 m unlock); (b) phantom gates from pure-mislabel groups → geometry cross-check merging.

### Deviations from original spec
- CLI input = new `racer.mapper_sightings/v1` format (`extract_run.py` emits no detections; recording→sightings extractor recipe documented in WRITEUP).
- Multi-pass case C has **NO loop closure** — documented workflow sidesteps it (one pass per solve, then pose-aided refinement once a first map exists).

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
  - **🚩 CORRECTED DEPLOYMENT RECIPE (fixes 3 sonnet S1.1 bugs):** ① actor output = **tanh(mean) → rescale** to thrust [0,5] / rates ±3.14 rad/s; ② FLU→FRD action sign = **[1,−1,−1]** (S1.2 value — ❌ SUPERSEDED 2026-06-12 by bcc93f9; current correct wire = `rate_flu·[−1,−1,−1]` — see §SHADOWPC-INC6-DIAG); ③ collective obs init = **0.0** rescaled; ④ training rate = **30 Hz**; ⑤ resets **at rest** 1 m from random gate. Other verified-correct items: obs layout (17), `R_W2G=diag(−1,−1,1)`, gate yaws all π, final-gate clamp, Euler-ZYX, actor arch (NormedLinear [256,128]).
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
  **opus-4.8 + adroit-connector** session (one session, scope already banked). Non-trivial,
  objectively-checkable → good opus fit. Remember the S14 caveat: training configs + evals must run
  map-ON (DR forces it; eval scripts still default flat).
  **🆕 Scope now ALSO includes procedural track randomization + VQ1-course-as-held-out-eval** (STACK-REVIEW-VQ2
  meta-gap ①, see that section — potentially binary for the unseen VQ2 course).
  **🚩 2026-06-11 (TWIN-FALSIFY): S16 AERO integration (quadratic body-frame drag + convex collective knot
  table, §TWIN-FALSIFY) inserts into the critical path BEFORE the final retrain — parity-gated separate
  session, same twin→rl_plant→torch recipe as S14. Retrain gating now extends to aero: any pre-S16
  checkpoint under-brakes ~2× at speed and trains under a phantom 3.77 g ceiling (real ~8 g).**

  **✅ S15 ENV-COHERENCE REDESIGN + S1.4 RETRAIN COMPLETE (2026-06-11, laptop fable; 7 jobs / 3 rounds on Adroit; source `handoff/laptop-s15-envredesign-retrain-2026-06-11/WRITEUP.md`; checkpoint commit 66dd3c2).**
  - **Deliverable: `rl/checkpoints/stage1_inc4_actor.pth` + sidecar `stage1_inc4_actor.json`** (md5 `766ea71b9136a65ed9c3b211a0aeea4e`; sidecar = `{"act_max_thrust": 3.765, "act_max_rate": 3.14}`). TRANSFER-CANDIDATE — in-twin ≠ transfer; next session flies via `fly_rl.py --flights N` (zero code changes).
  - **Headline in-twin (map-ON plant, standing starts, held-out VQ1):** sr **1.000 / 3582 eps**, median lap **7.03 s** (model-based 35.3 s; TOGT bound 4.55 s); roll med 72.8°, p90 78.1° (< 80° envelope); roll/pitch saturation ~1–2% (was ~94%); **generalization 0.845 on random unseen courses** (meta-gap ① addressed). Laptop deploy-pipeline rollouts 6/6 from all realistic starts (standing, ±1/2-step latency, handoff, live-clip).
  - **Part A — six audited reward/termination incoherences (C1–C6) fixed:** C1: OOB was free-bootstrapped escape → now terminal −25; C2: one "collision" class + post-crossing test → interpolated crossing point: PASS (<0.75 m) / FRAME-COLLISION −25 terminal (0.75,1.36] / CLEAN-MISS −15 terminal (>1.36), all 6 gates both dirs; C3: ATT taxed all tilt → hinge free below 60°, quadratic above (w=4); C4: jerk=‖ω‖ taxed cornering → action-rate ‖Δa‖² (w=1.0) + residual ‖ω‖ (w=0.05); C5: no time pressure → −0.02/step + finish-time bonus +0.25/s left; C6: survive_rate counted OOB → distinct outcome stats. Winner reward weights: collision 75, oob 75, miss 40, finish_time 0.25, dact 1.0, tilt 16. Every term formula/weight/units/what-breaks in env docstring.
  - **Part B — procedural course sampler** (`rl/peregrine_course.py`): VQ1-derived ranges (U[15,45] m segments, ±60° heading, U[−3,+12] m descent, ±12° yaw jitter); VQ1 held out + interior to every range ⇒ real generalization test. Spawns tail-first vs gate yaw → `fly_rl.py` unchanged. `+env.course_mode=random` enables training; `vq1` (default) = held-out eval + backward-compatible.
  - **🚩 S1.3 NaN ROOT-CAUSED (corrects "PPO grad-NaN/seed" lore):** unclamped Euler extraction in `get_observations` — pytorch3d `matrix_to_euler_angles`/`asin` returns NaN when float32 entry hits 1+1e-7 (once per ~30M obs; faster with random courses + high-tilt). NOT a PPO/gradient issue. Fix: same clamp as parent's loss path + counted `nan_to_num` lifeline. Zero NaN events in all 5 post-fix runs.
  - **🚩 CRITICAL DEPLOY BUG FOUND+FIXED — THE SIDECAR:** `fly_rl.py` had no per-checkpoint action-scale metadata; every 3.765-trained checkpoint (incl. S1.3) deployed through a [0,5] thrust rescale = **×1.33 thrust overdrive + corrupted obs[12] feedback**, invisible to eval (eval uses training cfg). Fix: launcher wraps `agent.save` to emit `<ckpt>.json` at every save point; `stage1_inc3_actor.json` committed retroactively; loud warning in `load_actor` when absent. No sidecar → behavior identical to before (backward-compatible).
  - **Training provenance:** job 3267360 (`s14_valid_s1`), Adroit, 6000 PPO updates (196.6M env-steps), seed 1; config `RW_TILT=16 RW_COLL=75 RW_MISS=40 RW_OOB=75 RW_FTIME=0.25 RW_DACT=1.0`, map-ON DR always-on, `max_normed_thrust=3.765`, g=9.80665. Alternates on Adroit: `s14_valid_s0` (VQ1 0.999/6.73 s, faster); `s14_seed0/emergency@848` (tilt4: VQ1 1.000/8.52 s, gen 0.927, median tilt 97° — style outside envelope, in `.s15_ref/`).
  - **Other review findings (14 confirmed → fixed):** offline_rollout graded backward-open-aperture crossing as collision (wrong); OOB box ~10 m tighter than training behind pad; `--virtual-flip` defaulted OFF vs deployment ON; stale checkpoint paths; sampler ignored misspelled overrides; periodic saves non-atomic; NaN-guard could run forever; sbatch lexicographic sort vs newest-by-mtime; eval peak-tilt read post-reset states. Four pre-existing issues accepted (GAE truncation leak, terminal magnitudes vs value-clip, time aliasing, lateral grazes undetected).
  - **🚩 IMPORTANT CAVEAT: inc4 is map-ON but AERO-OFF** — `dr_aero` was opt-in to avoid mid-campaign plant change; inc4 still carries ~2× under-braking at speed. inc4 = live test of the env-redesign hypothesis; **definitive VQ2 candidate = inc5 (one more retrain, `+dynamics.dr_aero=true`, first action = deferred V100 parity gate).**
  - **Deployment notes:** `fly_rl.py --flights N` zero changes needed; sidecar auto-applies 3.765 bound (watch for `[load_actor] sidecar` line; if WARNING appears, json didn't travel with pth). Expected live: standing-start, ~7–8 s lap, peak roll ≲80°, smooth r/p cmds, thrust/yaw at bounds (normal). CTBR bridge (`--bridge`) + `--max-rate`/`--max-thrust` remain as safety caps. Failure triage: (a) sidecar applied? (b) virtual flip on? (c) replay live handoff through `offline_rollout.py` — passes offline but fails live → suspect telemetry/timing.
  - **Tests: 528 green** (409→528 this session +47 + aero session +72; parity gate re-passed 8.9e-16).
  - **NEXT (CRITICAL PATH):** ① **ShadowPC: `fly_rl.py --flights N`** with inc4 + corner-pass probe folded in; ② **Adroit: inc5 (aero-ON retrain)** — first action = run deferred V100 parity gate (`run_parity.sh`); use `+dynamics.dr_aero=true`, same env/reward config as inc4.

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
- **① ✅ CLOSED AT DISTRIBUTION LEVEL (TRAINING-DOCTRINE 2026-06-12 — supersedes "single-track training open gap"):** inc6 already trains `course_mode=random` (turns ±60°/segment, yaw-jitter ±12°, VQ1 held out); gen 0.982/0.741 across seeds IS the random-course success rate. Remaining honest residual = sampler-range adequacy vs unknown VQ2 course (widen only on actual VQ2 info; widening costs gen variance now for unmeasurable benefit). See §TRAINING-DOCTRINE for verdict. Original framing: single-track training vs unseen VQ2 course — potentially BINARY; rank-impact #1. FIX was procedural track randomization + VQ1-held-out. **FOLDED INTO the env-coherence redesign scope** (completed inc6 training cycle).
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
- **Contact-tolerant racing lines** — rules-invalid (gate contact = invalid run, 2026-06-11). TOGT 1.06 m corner-clip optimum is illegal; plan to ≤~0.5 m from centre.
- **Graded/non-terminal contact reward in RL training** — S15's terminal-collision-penalty-75 design is EXACTLY right for these rules; do not soften.
- **Recovery-from-contact curriculum** — DEMOTED to near-miss/disturbance recovery only; covered better by the queued MPC-shadow watchdog. Contact itself = run over, no recovery to train.
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

## ✅ INC5 SHIPPED — AERO-ON TRANSFER CANDIDATE (2026-06-11, laptop fable + Adroit; commit 02fcce1; writeup handoff/laptop-inc5-aero-retrain-2026-06-11/)

**SUPERSEDES inc4. inc4 = KNOWN-BROKEN on corrected physics: aero-ON eval sr 0.461, 53.9% collisions, dies at +1-step latency. Its "transfer-candidate" status rested on the falsified linear drag — VOID.**

### V100 parity gate (prerequisite; deferred from S16)
Job 3267545 + re-verified. Worst DIV 2.665e-15 across all 6 configs (legacy / super_rate / delay2 / map_delay / aero / aero_delay). PASS (acceptance ≤1e-6). `run_parity.sh` md5 tripwire confirmed.

### Round 1 — inc4 warm-start (unconstrained speed datum)
Inc4 weights verbatim → aero-ON plant → 5 jobs round 1. Outcome: 6.89 s median but roll p90 145° (with real ~8 g convex thrust, the tilt-16 hinge makes aggressive flight too cheap → policy tilts hard). **This is the UNCONSTRAINED SPEED DATUM** (fastest the current reward allows on corrected physics). Not shipped.

### Round 2 — tilt-weight sweep rw_tilt ∈ {48, 96}
Winner: **inc5_t96_s1** (seed 1, job 3267805, 6000 PPO updates):
- VQ1 held-out: sr **1.000** / 2560 eps / 0 collisions / median **9.52 s**
- Generalization: **0.939** (random unseen courses)
- Style: roll p90 **64.5°** (envelope met)
- Deploy: **6/6** incl. +2-step latency (50 ms extra) — robust
- Shipped: `rl/checkpoints/stage1_inc5_actor.pth` md5 **bd1d670f7878eb938119ea173379e7d5** + sidecar `stage1_inc5_actor.json`; sidecar auto-loads via `load_actor`

Alternate banked (not shipped): **t48_s0** (9.22 s median, gen 0.960) — fails +2-step latency → not robust enough for the unattended eval. Kept for reference.

### Key findings
1. **Style envelope cost is now MEASURED:** rw_tilt=96 → 9.52 s; unconstrained (Round 1) → 6.89 s. **~2.3 s/lap cost on real aero.** This is a physics cost, not a tuning knob. The envelope's original anomaly-avoidance rationale is DISSOLVED (no anomaly; plant measured to full stick + full inversion). Envelope relaxation = candidate speed ladder (tilt 96→48→…, one live-verified step at a time) post live-transfer.
2. **Zero NaN** (S15 obs-clamp holds on aero plant across all seeds/jobs).
3. **`fly_rl.py` default checkpoint still points at inc4** — all live sessions must pass `--checkpoint rl/checkpoints/stage1_inc5_actor.pth` explicitly until the default is flipped.

### Deployment
Same recipe as inc4. `fly_rl.py --checkpoint rl/checkpoints/stage1_inc5_actor.pth --flights N`. Sidecar auto-applies correct thrust bound (watch `[load_actor] sidecar` line). Virtual flip ON by default. Expected: standing start, ~9–10 s lap, peak roll ≲65°, smooth commands.

### NEXT (superseded — see §SHADOWPC-LIVE-DEPLOY-DIAG)
Live transfer attempted; mixer coupling found as blocking issue. NEXT = S17 mixer integration + inc6 retrain.

---

## ✅ SHADOWPC-LIVE-DEPLOY-DIAG — MIXER ROOT CAUSE (2026-06-11, commits ef2605d..8dbd9bf; handoff/shadowpc-live-deploy-diag-2026-06-11/WRITEUP.md + Appendices A/B in handoff/shadowpc-inc5-live-2026-06-11/WRITEUP.md)

**SUPERSEDES the §SHADOWPC-INC5-LIVE obs-corruption hypotheses (H1/H2/H3 ALL FALSE).**

### Root cause
**THE SIM'S MOTOR MIXER COUPLES THRUST AND RATE AUTHORITY AT SATURATION CORNERS — unmodeled in all three plants.**

**Evidence:** `rl/replay_obs.py` replay forensics prove the live obs/action pipeline BYTE-CORRECT (step-0 obs/action match offline element-for-element; no NaN/staleness). The "all-rail outputs" are the policy's NORMAL bang-bang style (collective pulses 0↔1.0, yaw dither ±3.14 every tick — it flies the twin this way). Characterization sweep only measured symmetric low-to-moderate inputs; it never probed the saturation corners.

**Corner 1 — low-thrust × high-rate:** at (thr=0 × yaw=3.14) the mixer clips motor pairs at idle → motors `[0.08, 0.73, 0.73, 0.08]` → **9.4 m/s² of UNCOMMANDED hover lift** (caused powered climbs and top-board strikes; earlier "bobbing-parabola" and "hover-substitution" explanations both REFUTED; at thr=0 × rates=0 the sim free-falls honestly).

**Corner 2 — full-thrust × any-rate:** at collective ≈1.0 there is no differential headroom → **rate authority vanishes during thrust pulses** (caused uniform gate-2 lateral misses dy −1.0…−1.8 m). Fit data recorded at `data/runs/20260611_194826_mixer_probe`.

### Mitigation (deploy-side ceiling)
`--yaw-scale 0` (twin-validated 6/6 at latency 0–2): inc5 went from 0 clean passes in 20 flights → **14 clean passes in 10 flights, 0 finishes**. Remaining failures = the two mixer rails (train-side) + 3 flights aborted at step-0 by bridge-seam contacts (strengthens `--no-bridge` default). Deploy-side patching has **hit its ceiling** — both rails require train-side fix.

### ✅ Twin aero LIVE-CONFIRMED, inc4 RETIRED
Inc4 probe ×3: all crash gate-0 in 11 steps via sustained full collective rocketing 3.1 m above the gate = the falsified-linear-map 2× thrust error, LIVE. The corrected-aero twin's inc5≫inc4 discrimination is confirmed; falsify→integrate→retrain chain validated end-to-end. **inc4 is RETIRED** (its "transfer-candidate" status was based on the falsified linear drag — void).

### 🚩 Live latency MEASURED
**2 ticks (67 ms)** by command-vs-realized cross-correlation. Supersedes the ~40 ms lore.

### INC6 SPEC (NEXT — one opus-4.8 session)
1. **Mixer-aware plant:** motor-level clip model, fit data at `data/runs/20260611_194826_mixer_probe`; integrate twin→rl_plant→DiffAero adapter parity-gated (S14/S16 recipe).
2. **Action-rate regularization:** bang-bang/dither is free in-twin, lethal live; penalize `‖Δa‖²` aggressively (build on S15's dact term — raise weight or add a yaw-dither-specific penalty).
3. **Train transport delay ≥2 steps:** 67 ms live = 2 × 33 ms control steps; DR over {0,1,2,3} steps.
4. **Standing-start deployment default:** Appendix B shows all four start modes 6/6 in-twin with inc5; the `fly_rl.py` "0/6 raw standing start" comment is STALE inc4-era (the OOB-box spawn fix + S15 redesign resolved it); bridge adds a non-deterministic contact-abort seam.

Queue note: TOGT bound eventually re-solved with mixer coupling (current bound assumes independent thrust/rate limits → slightly optimistic).

### Appendix facts
- Gate map EXONERATED: model-based runs cross gates 4–9 cm LOW uniformly = CTBR cruise-drag artifact, no fix needed.
- VISION-PKG2 +0.3 m vertical constant is NOT map-z (likely camera-optical-centre vs body-centre offset or anchor mismatch — needs dedicated calibration flights if ever relevant).
- Run 200505 has a frozen-telemetry stretch at gate 5 (identical timestamps) — analysis footgun; exclude from timing studies.

### NEXT
**S17 mixer integration + inc6 retrain (one opus-4.8 session) → live re-test standing-start A/B.**

---

## ✅ CORNER-PASS PROBE (2026-06-11) — closes live validity bracket for TOGT

Probe goal: does the sim's `race_outcome` accept a gate pass that clips the corner (as the TOGT-optimal line does at ~1.06 m Euclidean miss)?

### Results
| Offset from gate center | Outcome |
|---|---|
| ≤0.64 m | Gate ADVANCES (clean accept) |
| 0.60 m | Gate ADVANCES + contact event |
| 0.74 m | Inconclusive — nav overshot into outer frame (CTBR precision limit, NOT a validity reject) |

### Implications (REVISED 2026-06-11 — rules context)
- **🚩 CONTACT = INVALID RUN (rules-confirmed):** "0.60 m WITH contact" is a RULES-INVALID pass. The usable planning aperture = ≤~0.5 m offset from center (0.6 m logged a contact event; 0.64+ is moot — contact invalidates regardless of gate-advance).
- **TOGT corner-clipping optimum (~1.06 m Euclidean clips) is definitively ILLEGAL.** The shipped margined reference line (crossings ≤0.14 m from centre = 0.6 m tracking budget) is the correct planning artifact.
- **🚩 SUPERSEDED NUMBERS (2026-06-13, §PLANNING-TOGT-S2):** The 4.27 s inscribed-circle bound AND the 4.55 s shipped reference line are LINEAR-PLANT FICTIONS. Real v_max ~39 m/s (v² drag wall). Honest corrected-aero bound ~4.6–4.7 s; contact-valid bound ~4.43 s (bound_free, clears gate-4 by only +0.046 m). The 4.13 s corner-clip bound is DOUBLY invalid: ILLEGAL + drag-infeasible. `rl/reference_line_vq1.json` is also drag-infeasible + ~170° inverted as geometry → must be REBUILT before any hybrid-cast use.
- The 0.74 m inconclusive result is a nav-precision artefact; moot given contact rule.

---

## ✅ COAST-REPLAY CONFIRMED through S16 aero plant (2026-06-11)

Replay of the real-flight coast segment through the integrated S16 plant:
- **Speed RMS: 0.224 m/s** (target from campaign fits ~0.24 m/s)
- Result: **PASS** — the S16 aero integration faithfully reproduces the campaign-measured drag on real recordings.
- Confirms twin aero is correct; no further recalibration needed.

---

## 🚩 NEW SIM OPS (2026-06-11, user-observed — MANDATORY for all future live harness work)

### (a) Autoreset-latched-throttle hazard
The sim **AUTORESETS** on sustained gate contact (not just crash). If the RL pilot keeps commanding through an autoreset, the drone enters an **uncontrolled spin with throttle latched**. Rule: **NEVER leave the RL pilot commanding without a clean sim reset**. Harness must detect reset/contact event → cut commands immediately.

### (b) Spin detection
Add unrecovered-spin detection (angular rate magnitude > threshold for N consecutive steps) + auto-flag to the harness; treat as a terminal failure state.

### (c) Between-flight reset protocol
Between flights, do a **FULL escape→Enter reset** so every run starts from a fresh countdown. Observed weird autoreset-from-crash states when reusing the same post-crash context without a full reset.

### (d) Sticky-contact physics artifact (low priority)
During broken-pipe flights the drone was observed **PINNED against a vertical gate surface with zero commanded thrust** — probable sticky contact physics. One-look check in recordings during diagnosis; not an anomaly to model, just a harness edge case.

### SIM LAUNCH PATH (all future ShadowPC prompts)
`exe = "C:\Users\Shadow\Downloads\AI-GP Simulator v1.0.3364\AIGP_3364\FlightSim.exe"`
Launch sequence: any button → login page → Enter → cached logins → homepage (existing mechanics from there).

---

## 🚩 POLICY DECISION RATE: 30 Hz WAS NEVER A CHOICE (2026-06-11 finding)

The 30 Hz control rate was inherited from DiffAero `racing.yaml` `dt=0.0333` — not a deliberate decision. Confirmed non-issue for the plant (control-rate probe: rate map + dynamics identical at 50/100/200 Hz ±0.2%; inference microseconds). **However at VQ2 speeds (20–39 m/s) 30 Hz = 0.7–1.3 m between decisions vs 0.75 m gate half-opening.** This is a non-trivial precision risk at the top of the speed envelope.

**QUEUED experiment (after live transfer verdict, not before):** retrain at 60 Hz then 100 Hz — dt override + PPO horizon retune (keep same env-steps per episode) + latency-DR step rescale (delay steps scale with control freq). Compare: median lap, gate precision (in-plane miss distribution), +2-step latency robustness vs 30 Hz baseline. If 60 Hz wins without latency regression → flip.

---

## §SPEED-CEILING-ANALYTIC (2026-06-11, ADVISOR assignment F — analytic; assumptions twin-verified if ever disputed)

**Question:** does the 30 Hz policy rate bind at VQ2 speeds, or does the 60/100 Hz retrain stay queued?

### Method
3-phase slew-limited roll model:
- Phase 1: slew to max rate — α_max = 260 rad/s², ω_max = 11.2 rad/s (super-rate map full-stick); T_slew = ω_max/α_max ≈ 43 ms
- Phase 2: constant-rate rotation — θ(t) = ω_max · t
- Phase 3: decelerate back — symmetric to phase 1
- Maximum lateral acceleration during correction: a_lat = g · tan(θ), tilt cap θ_max = 65° (inc5 style envelope) → a_lat_max = g · tan(65°) ≈ 21.0 m/s²
- Dead time = 1 decision tick + 2-tick transport delay: Δt_30 = 100 ms, Δt_60 = 50 ms, Δt_100 = 33 ms
- Available correction time: T_avail = d/V − T_dead, where d = distance of last accepted fix, V = approach speed
- Closed-form lateral error correctable (T_avail > T_slew+T_coast): **e_max = 0.024 + 0.763·Δt + 10.52·Δt²** (quadratic in dead time)
- Conservative factor: 15–30% at high V (quadratic drag + roll-induced decel both add time; ignored in bound — adds conservatism)

### Ceiling table (lateral error correctable vs apertures: 0.5 m raw-fix / 0.2 m KF-converged)

| Last fix distance | 30 Hz ceiling | 60 Hz ceiling | 100 Hz ceiling |
|---|---|---|---|
| d = 10 m | ~24–30 m/s (scenario-dep.) | ~29–36 m/s (+5–6) | ~33–40 m/s (+4 more) |
| d = 15 m | ~38 m/s (fine for envelope) | well above | well above |
| d = 20 m | no ceiling anywhere | no ceiling | no ceiling |

### VERDICT
**30 Hz binds ONLY in the (≥30 m/s × last-fix ≤10 m) corner.** With fixes accepted to d=15 m (the 32 m range cap + pristine-velocity KF propagation makes this very achievable in practice), 30 Hz is NOT the bottleneck at inc5's style envelope.

Given-pose VQ2 (case A) has no detection horizon at all — the concern is exclusively in vision-only (case C) operation.

### Coupling note
Envelope relaxation from 65°→80° gives a_lat 9.8·tan(65°)→9.8·tan(80°) ≈ 21→55.7 m/s² (2.6×) — shifts ALL ceilings up substantially. **The envelope ladder is the bigger speed lever and stays first in order of operations:** inc6 live transfer → envelope relaxation → 60/100 Hz retrain only if the gating measurement demands it.

### 🚩 GATING MEASUREMENT (queued — decides the 60/100 Hz retrain priority)
Script over existing recordings (e.g. `data/runs/20260607_194615_course_60s`): extract per-gate-approach distance of the LAST KF-ACCEPTED lateral fix for each gate crossing.

- If consistently ≥15 m → 60/100 Hz retrain STAYS QUEUED (after envelope relaxation)
- If ≤10 m at speed → jumps to CRITICAL PATH ahead of envelope relaxation

**Bundle into the SHADOWPC-VISION-CAL session** (queued post-inc6 anyway; same recording needed for roll-wander re-measure + CPU latency benchmark + extrinsic calibration). One script pass, no new flights needed.

---

## PARALLEL ONBOARD SYSTEMS LEDGER (2026-06-11 brainstorm)

Design rule: all parallel threads feed the sacred 30 Hz control loop **only at tick boundaries**, preserving determinism. Keystone enabler = **KF rewind buffer** (makes any slow perception pipeline usable regardless of compute latency; a 50 ms stale fix mis-applied at 15 m/s = 0.75 m error; the rewind costs nothing for a linear KF). **Gated on organizer answers ① (VQ2 stream) and ⑤ (eval GPU).**

### Tier 1 (high value, known feasible)
- **Async heavyweight detector ~5 Hz full-res → rewind-corrected KF:** run YOLO11s at full resolution on every N-th frame in a background thread; correct into KF at frame timestamp via rewind buffer. Buys better detections at cost of batched latency.
- **ROI re-detection "zoom":** after a full-frame detection, crop a tight ROI around predicted gate position for the next K frames → potentially extends effective range past 32 m cap → could crack the case-C co-visibility wall (24–32 m gate spacing vs ~32 m range cap). **Free offline experiment** on existing recordings.
- **MPC-shadow safety floor + watchdog demotion:** run analytic twin one-step ahead in parallel; if predicted trajectory diverges from RL policy output beyond threshold → demote to model-based controller for that tick. Upgrades the validity supervisor from alarm to actuator.
- **Critic-as-risk-monitor:** run the learned value function alongside the policy; low V(s) = anomaly signal; can trigger watchdog demotion. Free (critic already trained). Needs one recording-replay calibration check to set the threshold.

### Tier 2 (conditional on live data / organizer answers)
- **Multi-frame far-gate PnP:** accumulate detector keypoints across 3–5 frames before solving PnP → lower noise at long range; helpful if >32 m gates matter.
- **Optical-flow velocity witness:** monocular flow cross-checks ODOMETRY velocity during blind transits (zero fixes).
- **Between-attempts residual fitting:** fit a small (state,action)→accel-residual on the just-flown recording, update the live twin for next attempt. Gated on organizer Q④ (can stack carry state between attempts?).
- **IMM filter bank:** multiple motion models (hover / cruise / gate-transit) → auto-switch; helpful if the KF diverges on sharp maneuvers.

### REJECTED (do not re-litigate without new evidence)
- **In-race map mutation:** determinism risk + planner-carrot hazard (changing gate positions mid-flight destabilizes the RL obs → policy OOD).
- **Policy-ensemble voting:** chattering on disagreement; disagreement-as-uncertainty-signal is OK (= critic risk monitor, already Tier 1).
- **Wind estimator:** sim has none; adds complexity for zero modeled benefit.
- **In-race policy adaptation:** gradient updates in the live loop on unknown eval hardware; risk >> reward.

---

## §ADVISOR-TRIAGE-2026-06-11 (first advisor-session batch, items A–J)

**Process note:** ADVISOR session (sonnet, read-only sparring partner) is live. Batches arrive as TO COMMANDER blocks; commander triages → bank/queue/reject. Results below.

### QUEUED (opus-4.8 candidates)

**VQ2 photoreal detector data pipeline (items A+B):**
- Blender/Cycles procedural renders: HDRI env, PBR gate materials, realistic lighting, motion blur, partial occlusion. Hue randomization **WIDER than ±15°** — do NOT assume the gate is red; appearance doctrine = overfit geometry, randomize appearance.
- Include OFF-FRAME corners (partial-gate / close-range scenes); per-keypoint visibility loss — YOLO-pose supports natively via the keypoint-visibility flag.
- Target 10–50k scenes. Can be built BEFORE VQ2 spec clarity (appearance gap is axis-independent of the VQ2 stream question).
- **First action:** check whether v2's existing synthetic set already has partial-gate scenes — avoid redundant generation.

**ShadowPC CPU in-loop perception latency + ONNX/TensorRT export sanity (do-now half of item):**
- YOLO11s@640 on ShadowPC CPU is completely unmeasured; plausibly 30–80 ms; it is a first-order unknown for Stage-2 and vision-only VQ2. Measure it. Export to ONNX, verify TRT if applicable.
- Jetson-class benchmark stays **gated on organizer Q⑤** (eval hardware GPU?).

**Tilt-cost concentration analysis:**
- Pull Round-1 unconstrained trajectory logs (6.89 s, roll p90 145°) and per-segment timing vs the constrained inc5 trajectory (9.52 s, roll p90 64.5°). Identify WHICH segments/gates eat the 2.3 s/lap cost.
- Informs **per-segment envelope relaxation** (relax tilt weight on the expensive segments first) vs. global tilt weight reduction — a finer-grained speed ladder than the current 96→48→… proposal.

**Stage-2 noise model / case-C coverage (gated on organizer Q①):**
- If VQ2 is vision-only, the Stage-2 RL-twin perception injection must cover case-C noise magnitudes (1.8–3.4 m lateral per §GATE-MAPPER). The current VISION-PKG2 twin model uses σ≈[0.73,0.47,0.29] m — a 3–5× underestimate in the case-C scenario.
- Noise model extension = a second training distribution or curriculum (pose-aided early laps → vision-only laps). Blocked until Q① confirmed.

### REJECTED (add to no-re-litigate ledger)

**HSV color pre-filter / color-keyed ROI:**
- Bakes an appearance assumption (gate color) into the runtime detection path — exactly the brittleness the randomize-appearance doctrine guards against.
- Detector false-positives are NOT a measured problem; `association.py` + geometry consistency already kills wrong-gate fixes (§VISION-PKG2).
- Speed benefit duplicates the MAP-DRIVEN ROI-zoom already in the Parallel Systems Tier 1 ledger; the map-driven version is strictly more robust.
- **DO NOT re-litigate without: (a) a measured FP rate problem in VQ2 photoreal conditions AND (b) evidence that the gate color is stable across VQ2 environments.**

### S17 AMENDMENT (sent to dispatched S17 session)
Inc6 regularizer: sweep **BOTH** blunt ‖Δa‖² AND a targeted joint corner penalty `~w·|thr−mid|·‖rate_cmd‖` (taxes exactly the two mixer saturation rails without suppressing mid-range thrust corrections). Ship the faster at equal live-compatibility. The joint-penalty form is a hypothesized improvement over blunt dact; the sweep settles it empirically.

### ASSIGNED TO ADVISOR (results return as TO COMMANDER digests)
1. **MonoRace deep-read** (arXiv 2601.15222, TU Delft/MAVLab): focus on <4-corner handling (partial gate in transits), calibration refinement pipeline, and how they maintain gate identity during blind transits. Findings feed Stage-2 vision design.
2. **Analytic speed-ceiling computation:** 30 Hz policy + 67 ms latency → lateral authority vs 0.75 m aperture across 10–40 m/s range. Does the 30→60/100 Hz retrain jump the queue over inc6? Decide from the math, not heuristics.

### Organizer email addition
**Add Q⑥: does a physical round follow VQ2 (timeline)?** If yes, flag the second sim-to-real layer (motor mix, ESC, vibration, prop wash — scope only if confirmed). Add to the email along with Q①–⑤.

---

---

## §MONORACE-DIGEST (2026-06-11, ADVISOR assignment E — arXiv 2601.15222, TU Delft/MAVLab)

Source: MonoRace paper deep-read. Three findings triaged.

### Partial gates / <4-corner PnP — GEOMETRICALLY INAPPLICABLE but produces one queued task
MonoRace pools corners across 2 co-visible gates in a multi-gate PnP fusion; this occurs in ~27% of frames on their fastest run. **This technique is GEOMETRICALLY INAPPLICABLE to our course:** consecutive gate co-visibility is IMPOSSIBLE (23.7–38.5 m spacing vs 24–32 m effective range — proven in §GATE-MAPPER). There is nothing to adopt from their multi-gate fusion directly.

**What IS applicable:** their off-screen / partial-corner handling. When a gate is partly outside the FOV, they use attitude (from IMU) + known gate orientation from map to solve translation from fewer corners.

**NEW QUEUED TASK — 2-corner translation-only PnP fallback (~1-day):**
- ODOMETRY gives attitude precisely; gate orientation from map is known.
- With attitude and gate normal fixed, 2 adjacent corners (collinear along one gate side) uniquely pin 3D translation.
- This is the minimum-corner analog to MonoRace's "de-rotated IMU-attitude" fallback.
- Use case: close-range transit frames where 2 corners exit the 58.7° VFoV; currently these frames yield no fix.
- Implementation: add a `solve_translation_2corner` branch in `localization.py` behind a `USE_2CORNER_FALLBACK` flag; parity-testable against full 4-corner on frames where all 4 are visible.
- **Fit to the existing off-screen-corner training already queued in the photoreal pipeline (item ①)** — the detector needs to reliably report partial gates for this to fire.
- Priority: post-inc6, bundle with SHADOWPC-VISION-CAL session.

### Calibration — pitch+roll extrinsics NEVER estimated; camera-body translation offset unresolved
Our refutation of the yaw extrinsic (§VISION-PKG2: both-signed per-gate slopes + flight-specific roll wander ⇒ covariance, not calibration) was correct and complete FOR YAW. However:
- **Pitch and roll camera-body extrinsics were never estimated.** MonoRace uses Bayesian IoU maximization (~40 BO iterations): reproject map gate corners through the current state estimate, compute IoU against detector output, optimize the extrinsic offset. Their result shows stable mount geometry.
- **Our 1.4° uniform `ATTITUDE_NOISE_STD_RAD` absorbs pitch+roll extrinsic error implicitly** — if those offsets are stable (plausible for a fixed camera mount), estimating them tightens the lever-arm covariance directly.
- **The VISION-PKG2 +0.3 m vertical systematic** is almost certainly a camera-body TRANSLATION offset (optical center vs body CoM), NOT a map height error (we verified map is exonerated). Regression from a level-hover recording (vary altitude, regress fix-z vs geometry prediction) isolates this cleanly.

**NEW QUEUED TASK — Bayesian-IoU pitch+roll extrinsic calibration:**
- Adapt MonoRace's IoU-BO method to pitch+roll ONLY (yaw treated as noise per our refutation; do NOT re-open the yaw calibration question without two flights showing consistent sign on τ_pre — see §VISION-PKG2).
- Run on the next fresh 6/6 recording with `--dump-extras`.
- Expected payoff: replace 1.4° uniform lever arm with a tighter per-axis value; may reduce false rejections in close-range transit frames.
- **BUNDLE into one SHADOWPC-VISION-CAL session post-inc6** alongside: roll-wander re-measure (`composition_fit.py` as-is), +0.3 m vertical regression from level hover, and in-loop CPU perception latency measurement.
- This session is gated on inc6 being live-tested (need a fresh uncontaminated 6/6 recording on the corrected policy).

### Gate identity / blind transits — NO CHANGE, verdict BANKED
MonoRace maintains gate identity through blind transits using: (a) a gate-sequencing prior (progress through the course in order), (b) velocity dead-reckoning between fixes, and (c) appearance-keyed re-identification when the gate re-enters view.

**Our stack is equivalent or stronger on every layer:**
- `RACE_STATUS.active_gate_index` = an EXPLICIT gate-sequence signal from the sim (MonoRace infers it from their own tracking — we have the ground-truth stream).
- `association.py` wrong-gate kill = the appearance-keyed re-identification analog (geometry consistency rather than appearance; more robust to lighting variation).
- Given `LOCAL_POSITION_NED` + `ODOMETRY` at 97/75 Hz = pristine dead-reckoning during blind transits (MonoRace relies on a noisier IMU integration).
- If VQ2 drops `active_gate_index`, our fallback IS their architecture (sequencing prior + dead-reckoning + association). **Risk CLOSED.**

No action items from this sub-topic.

---

---

## ✅ S17 MIXER INTEGRATION + INC6 COMPLETE (2026-06-11, laptop fable; commits per ffe3cdb+ab634c8; writeup handoff/laptop-s17-mixer-inc6-2026-06-11/WRITEUP.md)

**Critical-path step DONE. NEXT = ShadowPC live transfer (standing start, no mitigation flags) after mixer_probe2 errand + R2 seed-variance check (job 3268876).**

### Mixer model
Per-motor clip: `u_i = clip(c + S·d, idle, 1)`; differential demand `d = κ_err·(target−ω) + κ_hold·ω`; yaw effectiveness `ζ/(ζ+c)`; **κ_err = 0.073** (two independent probes agree to 0.2%); every measured probe row reproduces. S14 slew limits were measured WITH the mixer throttling, so authority scales by `Q=r/r_fit` keeping the fit point exact.

**KEY VALIDATION:** inc5 through the mixer-ON twin reproduces the live failure QUANTITATIVELY (collective 0.000, yaw rail, powered climb 5.1→9.8 m/s by tick 12 = the diag's exact live figure). Top-rail corner (collective≈1.0 × rate) NOT discriminably reproduced — unmeasured; `mixer_probe2.json` committed (10-min probe profile, run before/with the live session, documented rather than guessed).

### Integration
All three plants; defaults OFF = bit-identical legacy; tests 528→562; local CPU gate 7.1e-15; V100 GATE_PASS 7.1e-15; negative controls caught both corruption modes; 25-agent adversarial review, 2 minors fixed.

### Reward-design lesson (durable)
**Joint corner tax `w·|thr−mid|·‖rate‖` DECOUPLES smoothness from generalization, where blunt ‖Δa‖² trades them monotonically** (gen 0.727→0.571 as dact 1→16; combo arm: gen collapses to 0.633 the moment dact=4 reappears). This is a structural result — taxes exactly the mixer saturation rails without suppressing mid-range corrections.

### Inc6 deliverable
Winner **c16** (joint-penalty, no dact): simultaneously fastest (9.86 s median), smoothest (thr_p95 0.061, 0% saturation all axes), best generalizer (0.982 — above inc5's 0.939 on the HARDER mixer-ON plant). Yaw dither dead in ALL seven arms. Checkpoint `stage1_inc6_actor.pth` + sidecar; `fly_rl.py` default still inc4 — explicit `--checkpoint` required. **R2 RESOLVED:** shipped s0 stands; style/validity robust across seeds (s1: VQ1 sr 1.000/9.72 s/thr_p95 0.062/0% flips ≈ s0); gen SEED-VOLATILE (0.741 s1 vs 0.982 s0). **≥3-seed gen averaging required for all future candidate selection** (inc7, S2).

### Deploy matrix
16/16 laptop: latency 0–3 × every start mode × seams perturbed to 10 m/s. NO mitigation flags needed (no `--yaw-scale`). VQ1 held-out: **sr 1.000 / 9.86 s median / max pass offset 0.277 m** (zero-contact margins everywhere — rules-compliant). Twin now discriminates BOTH ways: inc5 on mixer plant = 0.000/0.000 success, yaw_flip 81.7% (reproduces live 0/20 at eval scale); inc6 passes everything.

### Inc5 retirement
Inc5 formally retired. Supersession chain: inc5 (mixer-blind, live failure now twin-reproduced) → **inc6 (mixer+aero+map plant, corner-tax c16, SHIPPED — live transfer pending)**.

---

## ✅ SHADOWPC-INC6-DIAG — roll-mirror root cause (2026-06-12; commits bcc93f9, 325e191; writeup handoff/shadowpc-inc6-diag-2026-06-12/WRITEUP.md)

**⚠️ SUPERSEDED by §FRAME-AUDIT (2026-06-12):** bcc93f9 conventions (quat AS-IS, rate [+1,−1,+1], wire [−1,−1,−1]) were themselves a second mirror — a proper-rotation alias of the one true defect (R_y(π) conjugation). The "counterfactual 6/6 @ 9.50 s" was a false pass. See §FRAME-AUDIT for the definitive per-layer map and corrected deploy recipe (93023cf).

**Supersedes the §SHADOWPC-LIVE-DEPLOY-DIAG hypothesis that obs-encoding or timing was the gap.**

### Evidence chain summary

**H0–H3 ALL CLEAN:** verified ckpt md5 + sidecar, no machine-local map in RL path, 29.2 Hz loop, sim/wall 1.000, zero stale ticks, step-0 obs matches twin to 7e-4. Root cause is NOT obs encoding, NOT timing, NOT artifact skew.

**Physical contradiction at tilt (the discovery):** at −55° pitch (inc6's first banked maneuver), the "artifact-undone" roll state from `build_obs` reads ≈0 for a full second while the raw ODOMETRY rate channel (as `build_obs` reads it) claims +1.0–1.5 rad/s sustained body roll. This is a self-contradiction that is invisible at near-level attitude.

**Quat-FD proof (`diag_h4d.py`):** raw ODOMETRY quat body-rate finite-difference matches `w_raw` under **[+1,−1,+1]** (gain 1.00, corr 0.93–0.98) in BOTH level AND tilted phases across 3 flights. The roll-inverted model collapses in tilted flight (corr 0.0–0.45). The raw quat also correctly rotates v_body onto the position derivative — it is the true attitude.

**Feedback-free command probe (`c100_r31`):** wire roll +3.14 → raw-quat roll −1.34 rad in 0.18 s ≈ −10 rad/s (super-rate ×3.2, inverted sign). Confirms live `S_live = [−1,+1,−1]`.

**Why the laptop matrix could not catch it:** `telemetry_from_truth` applies the same assumed artifact model that `build_obs` undoes — any misidentification round-trips to zero error. Only live MAVLink exercises the real convention.

**Counterfactual (`diag_counterfactual.py`):** buggy mapping → 0 gates, lateral sweep, OOB 2.4–2.6 s. Fixed mapping → **6/6, 9.50 s**.

### True conventions (durable — see MEMORY.md + [[project-ctbr-control-sysid]] for three-layer distinction)
- ODOMETRY quat = TRUE attitude AS-IS (no roll inversion; roll inversion belongs to ATTITUDE euler only).
- Raw `angular_rate` → true = **[+1,−1,+1]** (pitch only inverted). Supersedes S1.2 [−1,−1,1].
- Live command→rate sign = **[−1,+1,−1]** (roll AND yaw inverted).
- RL deploy wire: `rate_flu·[−1,−1,−1]`.

### Post-fix live results
Two standing-start flights (the authorized budget): no spin, no oscillation, smooth coordinated flight. But 0/2 — ~5 m +y miss at gate-0 plane → OOD wander. Rate channel verified (d=2 ticks, τ=0.019, gain 0.94–0.97). Residual = translational: thrust overprediction 15–25% at 3–12 m/s (airspeed lapse, unmodeled at near-zero-airspeed fit). Displaces approach line ~5 m in first 2 s from tilted standing start.

### Durable validation discipline
**ANY future attitude/rate convention change MUST be validated with tilted-phase quat-FD consistency.** Level-flight correlation cannot see a roll mirror. The `diag_h4d.py` method is the permanent convention gate.

### Next path
1. Bridge ×5 with fix (bypasses the thrust-lapse regime; cheapest gate-threading discriminator).
2. S18 joint translational refit from 17 2026-06-12 recordings (3–30 m/s, no new flights).
3. Re-run inc6 deploy matrix on refit plant; fly as-is if robust, else inc7 retrain with lapse-DR.

**Inc6 checkpoint STANDS. No retrain implied by the convention fix.**

---

## §S18-THRUST-LAPSE (2026-06-12, laptop opus; commits fb99636+0fd741b; writeup handoff/laptop-s18-thrust-lapse-2026-06-12/WRITEUP.md)

**⚠️ VOIDED by §FRAME-AUDIT (2026-06-12):** the "15–25% thrust deficit" was an artifact of the mirrored-b3 projection (≈20° East bank at early climb gives exactly the 0.74 ratio; "recovered" as trajectory straightened). True-attitude refit ratio ≈1.00–1.10 all bands. There is NO residual translational plant gap ≥~2 m/s² in any regime (speed×tilt×collective, mixer plant, lapse OFF). Do NOT use `--plant lapse` or `dr_lapse` for training. The lapse code stays in repo (defaults OFF, harmless). Inc6 is SALVAGED (see §FRAME-AUDIT). The evidence-voiding finding ("twin self-mirrors → false passes") that this section discovered was correct; the root cause it pointed at (thrust→world mirror = a third independent mirror) was the R_y(π) conjugation, resolved in §FRAME-AUDIT.

**Headline: the lapse is real and integrated, but it is NOT the live cause. A THIRD convention mirror found (thrust→world lateral projection). The laptop 16/16 deploy matrix and the INC6-DIAG counterfactual 6/6 are BOTH FALSE PASSES. Inc6 validity is UNKNOWN.**

### Fit (§1 in writeup)

17-run dataset (c846054). Smooth-tick (|ω|<1 rad/s) specific-force ratio K_eff/K(collective) vs |vel|:

| |v| band (m/s) | 3–6 | 6–9 | 9–12 | 12–15 | 15–18 |
|---|---|---|---|---|---|---|
| ratio | 0.74 | 0.82 | 0.88 | 1.00 | 0.99 |

Locked model `LAPSE_SPEED_MEASURED=[0,4,8,12,15]` / `LAPSE_FACTOR_MEASURED=[1.0,0.78,0.80,0.92,1.0]`. Conservative floor 0.78–0.80 at 4–12 m/s; drag-independent where it matters (spread ≤0.09 at 3–9 m/s); not separable above 12 m/s (but lapse ≈1 there). Fast-descent thrust loss (vortex-ring-like, L→−0.9) excluded — folded into inc7 DR band. 3-fold CV reduces out-of-sample bias in all low-speed bands.

### Integration (§2)

`fb99636`: multiplicative `a_up *= interp(|vel|, lapse_speed, lapse_factor)` in all three plants. `faithful_config(lapse=True)` = fully sim-faithful twin (super_rate+measured_aero+mixer+lapse). `+dynamics.dr_lapse` for inc7 (scales lapse DEPTH per-env [0.5,1.5]×). `--plant lapse` eval shorthand. **562→589 tests green** (+20 parity +7 unit/torch/DR). Defaults OFF = bit-identical legacy. **🚩 V100 config-matrix gate MUST run at next Adroit contact** (DiffAero base not importable on laptop; lapse/lapse_full configs now in the matrix).

### Verdict reversal chain (§3)

**Step 1 — repro sweep:** inc6 on `--plant lapse` finishes 6/6 @ 9.49 s (same as `--plant mixer`). Swept depth/persistence at latency {0,2,3}: shallow lapses finish cleanly centred; deep persistent → vertical crash (still laterally centred). No lapse curve produces the live +5 m East miss. **Lapse falsified as cause.**

**Step 2 — yaw decomp:** live divergence at ticks 42–66 at 16–18 m/s (where lapse ≈ 1). Policy commands hard pitch-up flare + yaw turn; sideslip −72°; heading 135°→175°→−115°. Yaw rate prediction vs realized: +2.12 vs +2.08 (tick 60). **Rate loop faithfully tracks commands — no missing torque; policy commands the spin.**

**Step 3 — open-loop replay** seeded at live tick 36 with exact recorded live wire commands for 27 ticks:

| | live | twin |
|---|---|---|
| roll/pitch/yaw (tick 60) | 57°/7°/−115° | 56°/7°/−113° |
| speed | 14.8 m/s | 14.9 m/s |
| **East velocity** | **+11.1 m/s** | **−11.1 m/s** |

Attitude, rates, speed reproduced exactly — East velocity OPPOSITE-SIGNED. Same attitude + same thrust → opposite lateral force.

**Step 4 — force frame selfcheck** using live-recorded true attitude:

| tick | roll | measured a_E | model a_E | model thrust_E |
|---|---|---|---|---|
| 42 | 46° | +5.4 | −6.8 | −6.9 |
| 51 | 59° | +30.3 | −35.1 | −33.2 |
| 60 | 57° | +2.9 | −17.3 | −12.4 |

Model East = negative of measured; dominated by the thrust term. Flipping thrust East sign → model matches (tick 51: −35→+31 ≈ measured +30). Discrepancy present at tick 39–42 in normal banked flight (roll 44–46°, not a degenerate case). **Root cause: twin thrust→world lateral projection is roll-handedness-mirrored relative to the sim, given the same attitude quaternion.** Most likely root: raw ODOMETRY quat is roll-mirrored vs true physical attitude (the INC6-DIAG quat-FD only validated level+pitched phases; hard-roll gate-0 flare is the first maneuver exercising lateral handedness).

### 🚩 Evidence voided — false passes

**Twin self-mirrors in closed loop** → inc6 finishes 6/6 on every plant offline (mixer, lapse, all start modes/latencies/perturbed seams). Therefore:
- **Laptop 16/16 deploy matrix = FALSE PASS** (harness mirrors self-consistently).
- **SHADOWPC-INC6-DIAG counterfactual 6/6 @ 9.50 s = FALSE PASS** (same harness).
- **"Inc6 checkpoint STANDS" (INC6-DIAG verdict) = UNKNOWN** pending LAPTOP-FRAME-AUDIT.

**DECISION (commander-accepted):** DO NOT fly inc6 as-is; DO NOT train inc7 (would bake the mirror deeper). Lapse-DR infra built and ready for after the convention is fixed.

### mixer_probe2 contradiction (§4)

New settled-spin rows vs S17 model:
- `c60_r31` (clean, no clip): model predicts d=0.523, measured 0.132 → **roll κ_hold over-predicted ~4×** (implied ≈0.012 vs yaw-derived 0.046)
- `c100_y31`: model predicts d=0.122, measured 0.349 → **yaw top-rail under-predicted ~3×** at c=1.0

Not integrated (structural: per-axis κ_hold + yaw top-rail effectiveness). Impact likely small (policy avoids the regime: thr_p95 0.061, 0% saturation). Flagged as S19 candidate follow-up.

### Next step

**LAPTOP-FRAME-AUDIT (fable):** systemic per-layer handedness audit. Diagnostic scripts in `handoff/laptop-s18-thrust-lapse-2026-06-12/scripts/` (`s18_force_frame_selfcheck.py`, `s18_openloop_replay.py`). Extend quat-FD to hard-ROLL phases. Candidate roots: (a) raw ODOMETRY quat roll-mirrored vs true physical attitude; (b) sign in deploy thrust path. One live probe would settle it (deliberate sustained-roll-bank at moderate speed; compare measured lateral accel to attitude-derived prediction).

---

## Open
- ~~Substrate bake-off verdict~~ ✅ RESOLVED: **DiffAero** — proven on Adroit (plant injected, gate PASS, trains our plant).
- Structural pilot-stack changes (user brainstorming — the Setpoint/ControlCommand seam keeps a
  planner+controller→policy swap low-risk).
- VQ2 data-stream answer (gates the map/SLAM + vision-load-bearing question).

---

## ✅ TILT-CONCENTRATION (2026-06-11) — per-segment cost, revised speed ladder

**Supersedes** the "tilt 96→48→…, per-segment candidate ladder" phrasing in §INC5 and the §SPEED-CEILING-ANALYTIC coupling note. Source: `handoff/laptop-tilt-concentration-2026-06-11/WRITEUP.md`. Rollouts were deterministic from racestart (20 identical episodes ≡ single trajectory) — all figures are single-trajectory, not distributional.

### Per-segment timing table (inc5_r1 unconstrained vs inc5 constrained vs inc6; Δt = constrained − unconstrained)

| Segment | Δt (s) | Unc peak roll | Tilt cap binds? |
|---|---|---|---|
| start→G0 | **+0.73** | 63° | no — cost = global conservatism |
| G2→G3 | **+0.67** | 67° | YES |
| G1→G2 | **+0.47** | 71° | YES |
| G0→G1 | **+0.40** | 44° | no |
| G4→G5 | **+0.37** | 75° | YES |
| G3→G4 | **+0.27** | 32° | no |
| **Total** | **2.91 s** | | |

### Key findings
- Tilt cap physically binds on only **3 of 6 segments**: G1→G2 (pk 71°), G2→G3 (pk 68°), G4→G5 (pk 75°).
- **Biggest-cost segment start→G0 (+0.73 s) has unconstrained peak roll 63° — within the current 65° envelope.** Cost = global reward conservatism from rw_tilt weight, not the angle cap binding. Global weight reduction is the right first lever.
- 🚩 **Do NOT conflate**: training "roll p90 145°" is peak-per-episode across jittered starts (distributional); racestart peak roll = 75° (single trajectory). Two different metrics.

### Revised 3-step speed ladder (supersedes per-segment-first phrasing)
1. **Step 1 — global rw_tilt 96→48** (existing datum: inc5_t48 = 9.22 s; one reward-weight change).
2. **Step 2 — raise free-cone 60°→75–80°, keep rw_tilt=48** — unlocks specifically G1→G2, G2→G3, G4→G5 (the 3 segments where the cap physically binds), without entering the 90°+ regime.
3. **Step 3 — unconstrained** (rw_tilt=16, no cone; potential ~6.6 s). Live-verify stability before banking.

**Per-segment relaxation DEMOTED to contingency** — only if instability appears in low-cost segments at high global aggression. Global reduction is the right first move because the tilt weight shapes speed everywhere, not just at corners.

Ladder remains **gated on inc6 live transfer, one live-verified step at a time.**

---

## §FRAME-AUDIT — R_y(π) conjugation root cause; all mirrors resolved (2026-06-12, laptop fable; commits 93023cf fix, 1bfb936 writeup)

**Single defect explains every historical sign inversion:** the sim's ODOMETRY quaternion is the true attitude expressed in an R_y(π)-conjugated frame pair. This is a proper rotation → passes every internal-consistency test. Only comparison against an external invariant (FD of pristine `vel_ned`) catches it. Full data + scripts in `handoff/laptop-frame-audit-2026-06-12/`.

### Conjugation math

`q_true = q_raw * [1, −1, 1, −1]` (wxyz; negate x and z components). Euler effect: roll AND yaw negated, pitch intact. True angular rate: `ω_true = −w_raw` (all axes; quat-FD gain 0.999/0.999/0.996). Live cmd→rate sign: **[+1,+1,+1]** — vanilla CTBR, NO inversion on any axis. ODO twist (body vel) and accel_body PAIR with the RAW quat and are self-consistent — do not "fix" them.

### External-invariant method (the discriminating test)

Level-flight and internal-consistency tests (quat-FD vs rate, twist round-trip, tilted-phase kinematic coherence) ALL pass for a proper-rotation conjugation — they are conjugation-invariant. The ONLY discriminating test: force model from the candidate attitude vs FD of pristine `vel_ned` (which is NOT affected by the telemetry conjugation). Corpus: 17 runs, 28,195 banked smooth ticks.

- Conjugated attitude: corr +0.97…+0.99 all three force axes, median residual ~1 m/s².
- As-is attitude: corr −0.84 on East at bank, median error 24 m/s².
- R_x(π)/R_z(π) alternatives: fail North (pitch flip) — ruled out.
- Yaw confirmed independently by turn direction (course rate from velocity matches only conjugated yaw rate).

**DURABLE VALIDATION DOCTRINE:** internal consistency CANNOT catch a proper-rotation conjugation. ALWAYS validate conventions with the external invariant (force vs vel_ned FD). Run `scripts/frame_residual_report.py` after every live session.

### Why prior verifications failed

| Verification | Why it failed |
|---|---|
| bcc93f9 quat-FD tilted validation | Validated quat against its own rate channel — conjugation-invariant |
| "Counterfactual 6/6 @ 9.50 s" (bcc93f9 harness) | Emulation applied same mirror as deploy — false pass |
| "16/16 matrix" (bcc93f9 harness) | Same self-consistent mirroring |
| S18 lapse K_eff ratio 0.74–0.88 | Mirrored-b3 projection; true-attitude refit ratio = 1.00–1.10 |
| S1.2 [−1,−1,1] verdict | Level-attitude measurement; roll-mirror invisible near-level |
| bcc93f9 three "coupled corrections" | A second self-consistent mirror; fixed the seam, not the root |

### Per-layer map (complete, 2026-06-12)

| Layer | Status | Notes |
|---|---|---|
| LPN pos/vel (world NED) | ✅ TRUE | pos-FD ≡ vel through 60° bank all axes |
| ODOMETRY quat | R_y(π)-conjugated | `q_true = q_raw·[1,−1,1,−1]` |
| ODOMETRY angular_rate | `ω_true = −w_raw` | gain 0.999 all axes |
| ODOMETRY twist + accel_body | pair with raw quat | consistent pair; KF predict safe as-is |
| Sim cmd→rate physics | [+1,+1,+1] vanilla | open-loop replay; no inversion anywhere |
| mavlink_client velocity_ned | ✅ CORRECT (keep) | raw-quat rotation of twist |
| CTBR stack (all sign configs) | ✅ closed alias | per-axis closure derived; VQ1-proven; do not touch |
| fly_rl (93023cf, fixed) | ✅ TRUE | `_ODO_QUAT_TRUE_CONJ=[1,−1,1,−1]`; `_ODO_RATE_SIGN=[−1,−1,−1]`; `_ACT_FLU_TO_FRD=[+1,−1,+1]` |
| offline_rollout (93023cf, fixed) | ✅ matches wire | emits conjugated quat + negated rates; plant `rate_sign=[+1,+1,+1]` |
| rl_plant / DiffAero | ✅ self-consistent | proper-rotation bridges; V100 parity 7.1e-15 |
| Training world (inc6) | ✅ internally consistent | no mirror ever inside training; `rate_sign` = trained-world convention, keep |
| Vision chain (navigator.py:295) | ✅ FIXED (8d7b0b3) | `R_world_from_odo_quat_wxyz`; east bias 0.0°/range; fix p50 1.37→0.47 m; 616 green |

### S18 lapse voided (detail)

True-attitude refit on the same 17 runs: K_eff/K ≈ 1.00 at 3–6 m/s, 1.04–1.10 above. The apparent 0.74–0.88 ratio at low speed was the mirrored East-component of thrust projection (≈20° East bank at early-climb → projection error gives exactly the observed ratio, then "recovers" as trajectory straightens). Regime-binned residuals (speed × tilt × collective) ≤~2 m/s² everywhere on the mixer plant, lapse OFF. **Lapse code stays in repo (defaults OFF), but do NOT enable it for training or evaluation.**

### Fix inventory (93023cf)

- `src/racer/frames.py`: `ODO_QUAT_TRUE_CONJ_WXYZ=[1,−1,1,−1]`; `true_attitude_from_odo_quat_wxyz()`; `true_rate_from_odo_angular_rate()`; full convention note (single source of truth for the whole stack).
- `rl/fly_rl.py`: conjugation applied in `build_obs`; `_ODO_RATE_SIGN=[−1,−1,−1]`; `_ACT_FLU_TO_FRD=[+1,−1,+1]`.
- `rl/offline_rollout.py`: `_RATE_SIGN_LIVE=[1,1,1]`; `telemetry_from_truth` emits conjugated quat + negated rates; `--plant lapse` marked voided-historical.
- `rl/replay_obs.py`: true-state extraction conjugated; `v_artifact` = deliberate wrong-frame canary.
- `rl/rl_plant.py`: LAPSE constants annotated VOIDED; `rate_sign` documented as trained-world convention.
- **Armor:** `tests/test_frame_conventions.py` (+9 golden-file tests, incl. East-sign open-loop replay and mirror canary that must keep FAILING for as-is reading); `scripts/frame_residual_report.py` (standing per-session residuals + canaries + optional open-loop replay). **598 tests green.**

### Inc6 verdict: SALVAGED

Training world was internally self-consistent throughout (rl_plant and DiffAero use proper rotations, single frame). Only the deploy mapping was wrong. With fixed tools: deploy matrix REBUILT 16/16 (mixer plant, latency 0–3 × 4 start modes, 8.2–9.6 s median); counterfactual fixed mapping 6/6 @ 9.50 s; bcc93f9 mapping against corrected emulation → OOB pre-gate-0 at 2.2 s. No retrain needed.

### ShadowPC live-confirm spec (next session)

4 flights in order: ① standing ×2 (`--checkpoint rl/checkpoints/stage1_inc6_actor.pth --no-bridge --flights 2 --label rl_inc6_frameaudit_std`), ② bridge ×2 (`--bridge --flights 2 --label rl_inc6_frameaudit_brg`). Pull main first; verify `git log -1 ≥ 93023cf`; run `pytest tests/test_frame_conventions.py -q`. After each run: `scripts/frame_residual_report.py --replay 36 <session>`. Abort criteria: spin-guard trip on flight 1; mirror canary TRIPPED (re-audit, do not iterate flags); 0/2 standing AND 0/2 bridge with canaries green → not frames, capture and hand back to laptop.

**Prediction:** standing start clears gate 0 centred (offline E at plane ≈ −0.2 m); prior failure modes (pre-fix spin; post-fix +5 m East miss) both explained and removed.

### ✅ VISION-FRAME-FIX COMPLETE (2026-06-12, sonnet-4.6; commit 8d7b0b3; 616 tests green)

`navigator.py:295` now uses `frames.R_world_from_odo_quat_wxyz(ds.orientation_ned_wxyz)` — true body→world rotation from the R_y(π)-conjugated ODO quat. Helper added to `frames.py`; `characterize_perception.py` updated to match. CTBR control path untouched.

**VQ1-replay regression (task2_frames, 40 near-level frames):**

| Metric | BEFORE | AFTER |
|---|---|---|
| Detection | 40/40 | 40/40 |
| Association | 29/40 | 29/40 |
| Fix p50 | 1.37 m | **0.47 m** |
| Good-fix yield (<1 m) | 6/27 (22%) | 23/27 (85%) |
| East bias slope | +3.5°/range | **0.0°/range** |
| North bias slope | −2.0°/range | −0.4°/range |
| Catastrophic leak | 1/29 (3.4%) | 0/29 (0%) |

**Golden tests:** `test_R_world_from_odo_quat_wxyz_gives_true_rotation_at_bank`, `test_R_world_from_odo_quat_wxyz_level_is_identity`, `test_vision_fix_correct_at_banked_attitude` — all PASSED.

**MLE sigma_theta re-fit:** QUEUED for ShadowPC — do NOT change `ATTITUDE_NOISE_STD_RAD=1.4°` until done.

Detail: `handoff/laptop-vision-frame-fix-2026-06-12/WRITEUP.md`.

---

## §CRAB-DIAG (2026-06-12, ShadowPC opus; commit 301a2cf; writeup handoff/shadowpc-crab-diag-2026-06-12/WRITEUP.md)

**Dissolves the live-confirm "70% slowdown", "rate-gain droop", and "trajectory alignment gap" readings. Confirms deploy chain faithful. Identifies one real deficit and specifies the S20 fix.**

### Durable lesson: extract TWIN ATTITUDE, not just lap time, before interpreting live posture

The key discriminator that falsified the obs-seam hypothesis: run `crab_twin_rollout.py` to extract the offline twin's TRUE attitude (via conjugated quat, same as fly_rl) at the live start state, then compare posture axis-by-axis. If twin and live share the same attitude envelope AND the twin finishes, the deploy chain is faithful and the posture is trained style. **Never interpret live attitude as a "bug" without first checking whether the twin reproduces it.**

### Dissolution table

| Live-confirm reading | Status | Explanation |
|---|---|---|
| "~70% slowdown" (16.24 s vs 9.5 s) | **DISSOLVED** — clock artifact | RACE_STATUS includes ~8 s CTBR launch; RL segment = 8.96 s ≈ twin 8.16 s (≤0.15 s/segment) |
| "0.85 rate-gain droop" at bridge speed | **DISSOLVED** — wrong canary | quat-FD-vs-w_raw telemetry-consistency canary, not command tracking; realized gain = ~2.5 super-rate, identical live vs twin |
| "trajectory alignment gap, ~0.4 m south" (gate-3 crash) | **REFRAMED** — climb-bin plant gap | +2.0–2.8 m/s² N+D force residual in 12–18 m/s × tilt-35–90 bin; E tracks to 0.04 m (frame-clean); accumulated divergence = ~1–2 m after 2 s climb |
| Crab posture — was it a deploy bug? | **CONFIRMED = trained style** | Twin reproduces roll +41.5°/tilt +54.8° and finishes; training-native `trainreset` flies +55.5° tilt; rw_tilt=96 trained a ~55°-tilt racing cruise |

### S20 spec pointer

See [[rl-increment-history]] §CRAB-DIAG for full S20 spec (refit collective/drag in 12–18 m/s × tilt-35–90 bin + ±12% DR + ≥3-seed + V100 gate → inc7). Predicted outcome: standing start clears gate 3, posture unchanged.

**⚠️ Supersession (TRAINING-DOCTRINE 2026-06-12):** "S20 refit → inc7" framing REVISED — the climb-bin residual is real but the PRIMARY barrier is the geometry fiction (see §TRAINING-DOCTRINE). S20 refit folds into inc7 if ready but is NOT the launch blocker. Inc7 spec replaces the "S20+inc7" pipeline here.

---

## §ADVISOR-REPORT-2-TRIAGE (2026-06-12)

Commander triage of Advisor Report 2 (2026-06-12). All rulings are final unless a re-open condition is explicitly met and documented.

### Ruling 1 — RE-REJECTED: HSV red-mask pre-filter (advisor item C)

Ledger entry from ADVISOR-TRIAGE-2026-06-11 stands verbatim. Advisor re-proposed item C in Report 2 WITHOUT meeting the banked re-open conditions (measured FP problem + evidence of stable gate color in VQ2). Re-proposed 2026-06-12 Report 2, re-rejected — conditions unmet; advisor reminded of ledger. Do not re-litigate without both conditions met.

### Ruling 2 — STALE / no-action

- **Item G** (joint throttle×rate penalty): already shipped as c16 in inc6 (§S17). No action.
- **Item H** (tilt concentration): already complete (§TILT-CONCENTRATION). No action.
- **Items E** (MonoRace analog) + **F** (speed ceiling): already banked (§MONORACE-DIGEST, §SPEED-CEILING-ANALYTIC). Verdicts unchanged.

### Ruling 3 — ROUTED TO LAPTOP-TRAINING-DOCTRINE (inputs, NOT decisions)

Two reward proposals are routed to the doctrine session for evaluation on merits — not accepted or rejected here:

- **(O) Arc-length progress along TOGT reference line as dense RL reward** (MPCC-cast): targets the 8.3 s → 4.27 s structural gap; `rl/reference_line_vq1.json` + `rl/reference_line.py` with `progress()` already exist. Doctrine evaluates fit to current env/reward structure.
- **(R) Gate-crossing lateral-velocity penalty**: tension flagged — CRAB-DIAG confirms sideslip crab IS the winning trained style (roll +41.5°, tilt +54.8°, finishes). A lateral-velocity penalty may conflict. Doctrine must resolve whether the penalty helps generalization or suppresses the trained posture.

### Ruling 4 — ACCEPTED: Stage-2 estimator target package (K + L + N)

Promoted from conditional to PLANNED Stage-2 targets. Bundle into Stage-2 estimator design session.

- **(K) Reprojection-error EKF / constellation navigation** (arXiv 2603.02742): pixel-space innovation `h(x) = K·[R|t]·p_gate`; every visible corner contributes independently; partial gates handled naturally; long-range bearing preserved without a hard range cap. Replaces the current world-frame position update.
- **(L) Temporal parallax multi-frame depth triangulation**: at 20 m/s, 5 frames = 3.3 m baseline → σ~0.15 m at 20 m; uses pristine ODOMETRY motion to anchor the baseline; removes the need for a hard range cap; pairs naturally with K (bearing tracks improve triangulation conditioning).
- **(N) Fisher-information observability-weighted KF updates**: principled per-update weighting by the local Fisher information; replaces the 32 m hard range cap with a continuous, geometry-aware gain schedule.

### Ruling 5 — ACCEPTED: bundle into SHADOWPC-VISION-CAL (M + Q + D)

These items join the planned SHADOWPC-VISION-CAL session (which already carries ADVISOR-TRIAGE ②⑤⑥ + last-fix gating measurement):

- **(M) Lucas-Kanade sub-pixel corner tracking**: detector acquires corners, LK tracks them at sub-pixel precision (0.1–0.3 px error), re-detect every N frames; handles close-approach dropout gracefully. Feeds weighted PnP continuously between full detections.
- **(Q) KF innovation monitoring as early-warning**: rolling Mahalanobis window on KF innovations; rising trend → slow down; persistent elevation → abort attempt; single spike → discard fix. Complements (but does not replace) the critic-as-risk-monitor.
- **(D) ONNX/TensorRT INT8 export of SHIP v2 + ShadowPC CPU in-loop latency benchmark**: export NOW on ShadowPC; benchmark latency in the live loop. Real eval-hardware benchmark IMPOSSIBLE until organizer Q⑤ answers. Do NOT claim "perception is fast enough" until measured on actual eval hardware.

### Ruling 6 — ACCEPTED (with noted gating)

- **(S) Heteroscedastic per-corner σ from detector** (NLL/RLE loss): feeds weighted PnP + KF covariance directly; replaces current heuristic confidence→σ mapping. Bundle with next detector retrain (photoreal v4 / Blender/Cycles build).
- **(P) Between-attempt trajectory refinement**: REMAINS gated on organizer Q④ answer (can stack carry state between attempts?). No action until answered.
- **(B) Off-screen corner training**: already part of the banked photoreal spec (§v2-inventory in [[project-detector-training-pipeline]]). Advisor confirmation noted; no new action required.

---

## §TRAINING-DOCTRINE (2026-06-12, laptop fable; writeup handoff/laptop-training-doctrine-2026-06-12/WRITEUP.md; doctrine `docs/training_doctrine.md`)

**Headline:** We have been training against a fictional gate. Fix = honest contact geometry in the env (inc7). Reward and DR are validated; no other changes needed for the gate-3 barrier.

### Root cause of standing gate-3 barrier (supersedes §CRAB-DIAG "S20+inc7" as whole story)

Training env scores point-mass L-inf < 0.75 m as a clean pass. Sim enforces volumetric body-halo contact (rotor halo ~0.3 m, body strikes frame before the plane on steep approaches). Four standing gate-3 crashes: L-inf **0.37–0.49 m** (4/4), mid-range commands (rate p95 ≈ 0.5 rad/s, cap 3.14; 84% authority unused, zero saturation), deterministic to ±0.1 m. The policy flies its trained funnel with high confidence and crosses exactly where trained geometry says is safe — and trained geometry is wrong by the body radius. The climb-bin residual (S20, ~2–2.8 m/s² N+D) supplies ~0.2–0.5 m displacement; the geometry fiction converts it to a crash. Bridge threads the same funnel 0.2 m lower at L-inf 0.25 m — passes fine.

### Operator questions resolved (Q1–Q14)

- **Q1 Crab = near-optimal:** excess along-track drag 0.43 m/s² (3%) ≈ 0.15 s/lap vs drag-optimal thrust-axis rotation; flat basin. No reward change.
- **Q2 Crash forensics:** not saturated-corrective, not passive-drift OOD — confident mid-range flight into fictional aperture. Deterministic 4/4.
- **Q3 Margin doctrine:** margin is a geometry property, not a reward property. Gate-3/5 have near-zero to negative true margin vs contact-true aperture in the tail.
- **Q4 DR premise stale:** dr_aero already randomizes coll ±10%, c2 ±24%; policy absorbs ±12% coll / ±25% drag / full climb-bin residual in closed loop. Add structured regime-binned force-bias DR to cover "sysid wrong in one regime."
- **Q5 Recovery:** 12/12 from ±1.5 m displaced restarts at 18.6 m/s (3× live error). Recovery curricula not load-bearing.
- **Q6 Cold start:** standing vs bridge = 1.9× residual-bin exposure + same funnel → fixes are S20 (shrink displacement) + contact geometry (make funnel hold margin). Reset-distribution redesign does not fix it.
- **Q7 30 Hz:** nothing implicates decision quantization; queue intact.
- **Q8 Single-track premise stale:** inc6 trains random courses; remaining = sampler range.
- **Q9 Aperture fiction (central finding):** see root-cause above.
- **Q10 Corridor blindness:** gates 3/5 approach from outside aperture cone at 3 m out; frame extrusion prices this.
- **Q11 Obs noise:** not an inc7 blocker (VQ1 case A given-pose; VQ2 case C = Stage-2 queued).
- **Q12 Selection metrics:** add per-gate crossing L-inf p95 vs contact-true aperture + corridor clearance at x=−1 m + robustness probes + twin-attitude extraction.
- **Q13 Corner tax exposure:** small (mean factor 0.013 along lap); final-approach commands mid-band; no suppression of corrections.
- **Q14 Robustness priority:** (1) honest contact geometry, (2) structured + global force DR, (3) reset/course diversity — NEVER reward damping.

### Advisor items O and R — REJECTED at doctrine level

- **(O) Arc-length progress along TOGT reference line as dense RL reward:** REJECTED for inc7. Addresses the 8.3→4.27 s structural gap (a Stage-2/S2-architecture concern). Adding a new dense reward alongside the geometry fix violates one-change-family-per-increment attribution discipline. Revisit at Stage-2.
- **(R) Gate-crossing lateral-velocity penalty:** REJECTED. Crab IS the winning trained style (Q1: 3%, flat basin); a lateral-velocity penalty suppresses a near-optimal posture for 0.15 s/lap. No style terms for measured-near-optimal styles.

### Inc7 spec (c16 lineage)

Sbatch: `rl/peregrine_racing_inc7.sbatch` (DO NOT SUBMIT until gates pass). Three additions to inc6 env/DR:
1. `+env.body_radius_m` ∈ [0.28, 0.38] per env: pass band 0.75−r, collision band (0.75−r, 1.36+r].
2. `+env.frame_depth_m=0.30`: volumetric collision over gate-frame |x| ≤ 0.30 m.
3. `+dynamics.dr_force_bias`: per-env world-frame bias ‖b‖ ≤ 3 m/s² in a random speed×tilt bin.
S20 refit folds in if ready (dataset LOCAL `handoff/shadowpc-postfix-dataset-2026-06-12/`); NOT launch blocker. Launch gates: ① V100 config-matrix parity; ② 598+ tests green incl. new geometry unit tests; ③ deploy matrix on contact-true scoring. ≥3 seeds.

**Prediction on record:** standing clears gate-3 ≥0.3 m corridor margin; crossing tails ≤0.25; posture unchanged (~55° tilt crab); lap cost vs inc6 ≤0.3 s.

### Rejected ledger additions (no-re-litigate)

Anti-crab/sideslip terms; aggression/action damping beyond c16; gate-proximity penalty; recovery curriculum as inc7 blocker; reset-distribution redesign for cold start; 60/100 Hz retrain now; course-sampler widening now; `--plant lapse`/`dr_lapse` (voided by frame-audit).

### Ranked next queue

1. **LAPTOP-INC7-ENV** (opus-4.8): implement Q9/Q10 geometry + dr_force_bias + tests; optional S20 refit same session; V100 gate; 3-seed Adroit launch.
2. **VISION-FRAME-FIX** (parallel, navigator.py:295).
3. Envelope ladder step 1 (rw_tilt 96→48) — gated on inc7 standing live confirm.
4. S19 mixer contradiction; SHADOWPC-VISION-CAL; 60/100 Hz (evidence-gated).

---

## §CASE-C-READINESS (2026-06-13, ultracode-vision-case-c workstream; synthesis lead; report handoff/ultracode-vision-case-c-2026-06-13/REPORT.md)

### Verdict: GO-WITH-CONDITIONS

Case C (vision-only pose) is structurally sound. All four prototypes (B rewind buffer, C range-anisotropic R, D latency budget, F registration re-survey) survived adversarial verify with only non-fatal corrections. The track-map registration scare is resolved (see gate-3 falsification below). However, case C is NOT flight-ready — conditions are HARD GATES, not nice-to-haves.

### Binding risk: UNMEASURED in-loop vision latency L

L (frame-in → fix-applied compute time) is unmeasured. Three failure modes hang off it:
1. The 67 ms datum is WRONG for vision — that is actuation latency (command cross-correlation). Edge HW estimate 5–15 ms → v·L 0.1–0.5 m (comparable to gate-4 margin, not catastrophic). Laptop CPU-only 112–139 ms → v·L 2.3–4.2 m at 20–30 m/s — breaks the 10 m last-fix rule. Verdict FLIPS if eval HW is CPU-class.
2. KF rewind buffer (piece B): horizon < L → drops ALL fixes, diverges to ~21 m RMSE (worse than naive). Buffer is HARD-blocked on measuring L to size the horizon.
3. Predict-forward needs a calibrated constant age = L. Both compensation paths blocked on the same measurement.
One ShadowPC/eval-HW recording with capture-to-apply timestamps resolves all three. **This is the #1 SHADOWPC-VISION-CAL item.**

### Three P0 blockers (case C unshippable without these)

- **P0-1 — Fix `_initialize()` crutch** (`navigator.py:255-271`): reads `ds.position_ned` with NO `use_given_position` guard → if sim streams LPN, every "case C" test is secretly case A (hidden ground-truth seed on tick 1). True case C (position_ned=None) seeds origin at pos_std=5.0. Effort: S, 1 file.
- **P0-2 — Measure in-loop vision L on eval HW**: one ShadowPC recording with capture→apply timestamps. Gates horizon-sizing, predict-forward age, and the speed thesis. Pair with P0-3 in one ShadowPC session.
- **P0-3 — TIMESYNC epoch reconciliation**: `frame.sim_time_ns` = server UNIX epoch; `DroneState.sim_time_ns` = IMU boot epoch — unreconciled. Already corrupts `time_since_vision_update_s` (`navigator.py:426`), benign at VQ1 but load-bearing for case-C coast/abort. HARD prereq for the rewind buffer. Effort: M.

### Per-piece findings (all survived adversarial verify)

- **B — KF rewind buffer / OOSM** (`kf_rewind_buffer.py`): bit-exact OOSM confirmed (|dx|=|dP|=0 vs oracle); SPD-preserved; v·L bias removal verified confound-free (analytic to <0.3%). SHARPEST RISK: horizon < L → diverges to ~21 m. Cost ~0.19 ms/fix (not 0.03 ms as the docstring claims), still ~170× under budget. Blocked on P0-2 + P0-3.
- **C — Range-anisotropic R** (`range_anisotropic_R.py`): r⁴ depth law math-correct (MC-confirmed slope 1.89); PSD on 5000 geometries. WEAKENED: shipped analytic-Fisher cov ALREADY carries the identical r⁴ law in-range → in-range benefit ~ZERO. Value only at >24 m range + off-nominal pixel noise. Lateral coefficient a1 extrapolates BADLY past VQ1 range. Do NOT integrate for VQ1/cases A-B. Deprioritized to P2.
- **D — In-loop latency budget** (`latency_harness.py`): PnP→KF chain measured 0.77 ms p50 (negligible). Detector: laptop CPU 112–139 ms (upper bound, no GPU); edge 5–15 ms ESTIMATE (never measured on eval HW). The "30 Hz binds only at ≥30 m/s × last-fix ≤10 m" conclusion is conditional on edge-class eval HW.
- **F — Registration re-survey** (`vision_cal.py`): robust re-survey tool validated. **Gate-3 1.46 m mis-registration FALSIFIED** (see §GATE-3-FALSIFICATION). All 6 gates registered ≤0.37 m in-plane. Residual registration sigma ~[0.21, 0.24, 0.03] m (N,E,D) — already covered by shipped FIX_COV_FLOOR_STD=0.40 m floor. Gate-4 in-plane ~0.10 m (fid 1020); 0.155 m inc8 margin not threatened by map error.

### Gate-3 falsification (terminal, compact)

gate-3 1.46 m D mis-registration FALSIFIED (2026-06-13, Fengyou-verified from raw course_bundle/frames.json + track_map.json). **Reference-frame artifact:** track_map records gate BOTTOM-centre; drone flies through OPENING-centre ~1.36 m (half outer-height) above it. drone_D − record_bottom_D ≈ −1.36 m at EVERY gate (gate-3 = −1.377 m, NOT anomalous). Referenced to opening-centre, gate-3 crosses 0.056 m (3D), PASS-CLEAN. All 6 gates registered ≤0.37 m in-plane. Finding-A dead. track_map is trustworthy.

### Gate-4 de-provisionalization

Gate-4 binding margin 0.155 m @ r=0.38 is registration-confirmed offline: course_bundle fid 1020 transit crosses ~0.10 m in-plane from the mapped opening-centre; a true ≥0.5 m gate-4 offset is ruled out. Drop "shares registration risk" / "provisional." Keep fresh-reset live winner-validation rider (guards the orthogonal policy/physics-state question).

### P0-P3 build plan (prioritized)

| item | priority | effort | depends_on | benefit |
|---|---|---|---|---|
| P0-1 fix `_initialize` crutch | P0 | S | — | unblocks ALL case-C validation |
| P0-2 measure in-loop L on eval HW | P0 | S | — | gates horizon-sizing, predict-forward, speed thesis |
| P0-3 TIMESYNC epoch reconciliation | P0 | M | — | unblocks rewind buffer; fixes navigator.py:426 tsv bug |
| P1-1 predict-forward (constant calibrated age) | P1 | S | P0-2 | removes first-order v·age bias; no TIMESYNC; ~80% of rewind win |
| P1-2 KF rewind buffer / OOSM | P1 | S | P0-2, P0-3, P1-1 | exact variable/late-fix handling; horizon<L → ~21 m divergence risk |
| P2-1 range-anisotropic R | P2 | S | P0-2 + >24 m recording | case-C long-range only; ~zero in-range value |
| P3-1 vision-velocity channel | P3 | M | — | DEFER — over-build; **mildly re-opened as margin lever by §ESTIMATOR-RACESPEED** (velocity prior = swing variable in gate-relative CONDITIONAL-GO) |
| P3-2 per-gate R coeffs | P3 | S | — | DEFER — 0.40 m floor already covers residual |

### SHADOWPC-VISION-CAL additions from this workstream

1. In-loop vision latency L on eval HW (capture→apply timestamps) — THE binding measurement; fold into organizer Q⑤ (eval-HW GPU).
2. Eval-HW detector timing (CPU-class = verdict FLIPS).
3. TIMESYNC epoch reconciliation build + verify vs live wire trace.
4. True vision-only cold-start (P0-1 fix + end-to-end case C exercise).
5. (Pre-existing) 2-corner PnP fallback, Bayesian-IoU extrinsic, per-gate last-fix distance, roll-wander re-measure.

### Prototypes (NOT in src/)

All under `handoff/ultracode-vision-case-c-2026-06-13/`: `kf_rewind_buffer.py`, `range_anisotropic_R.py`, `latency_harness.py`, `vision_cal.py` + verifiers + `REPORT.md`.

---

## §S2-DECISION (2026-06-13) — S2 architecture DECIDED: staged_monolithic_then_decomposed

**Decision: HIGH confidence. Two adversarial lenses (skeptic-mono + skeptic-decomp) independently converged. Source of truth: `handoff/ultracode-planning-togt-s2-2026-06-13/REPORT.md` + `artifacts/D_s2_final.md`.**

### Decision: staged_monolithic_then_decomposed

Retrain the live MONOLITHIC policy on the corrected-aero plant for inc8, cast as a **hybrid-monolithic** (arc-length progress reward over a REBUILT contact-safe min-snap reference line), with the style cone opened one ladder rung. Keep the fully DECOMPOSED stack (offline line generator + a real MPCC/RL tracker) as an explicitly-specced but **unbuilt-until-triggered** warm fallback.

### Why monolithic-first (summary)

1. **Recoverable time = envelope, NOT architecture.** Honest total-force tilt model: 60° cone bound ~9.8–10.6 s ≈ inc7 live 9.76 s → inc7/inc5 are near point-mass-OPTIMAL for their envelope. Architecture-agnostic cone relaxation is the speed lever.
2. **Monolith = only MEASURED realized lap.** inc7 live-confirmed (5/5 standing, gate-3 gone, 0/5 contact). Decomposed ceiling ~5.4–5.8 s is a paper number — 3/3 native trackers failed; only measured datum k=1.85 (bad).
3. **Decomposition assets are fiction.** `rl/reference_line_vq1.json` is drag-infeasible (plans 51 m/s vs 39 m/s drag wall) + ~170° inverted (25.8% inverted samples). Must be REBUILT on corrected aero before decomposition is a valid target.
4. **Decomposition's wins graft free.** No-corner-cut incentive (arc-length progress over reference line) + offline flywheel = adopted into the hybrid-monolithic. Its unique win (HARD inversion ban) guards inc1 backflip whose root cause is FIXED and inc7 never exhibits.
5. **Binding VQ2 risk = ESTIMATOR, not planner** (architecture-independent → demotes the whole S2 question). East σ 0.47 m = 3.0× the 0.155 m gate-4 margin. Decomposition buys NOTHING on perception.

### Decomposed fallback trigger (build ONLY on BOTH)
- (a) scipy-SLSQP toy MPCC probe on `rl_plant.step` shows constraint-aware tracker k <~1.25 (decomposed ~5.5 s ceiling is real); AND
- (b) relaxed-cone hybrid monolith starts cutting corners at gate-4 (M-1 regression, the one regime the HARD inversion ban would save).

### Monolith failure modes to manage
- **M-1** SOFT inversion guarantee via R4 hinge — could drift to corner-cutting at 75–80° → fallback trigger.
- **M-2** progress-to-gate-CENTER → speed-pushed corner-cut incentive → REMOVED by arc-length graft.
- **M-3** narrow convergent basin (2/3 seeds viable) → budget ≥4 seeds.
- **M-4** bimodal cold-start offset (~1.5 s in start→g0, pre-gate-0 only; inter-gate trajectories deterministic ±0.04 s) → WINNER-VALIDATION RIDER (≥3–5 fresh-reset live laps to confirm clean finish + measure startup offset; NOT guarding trajectory-basin sensitivity).
- **M-5** gate-4 margin erosion at speed → re-verify every speed win against `rl/contact_true_eval.py`.

### Ordered integration plan (inc8)
1. **[reward]** Graft hybrid arc-length progress: replace R1 (progress-to-gate-CENTER) with progress-along-Γ via `reference_line.progress()` over a REBUILT contact-safe line.
2. **[native]** Build corrected-aero min-snap + coupled-TOPP line generator (pure numpy, <1 s; ~40 lines exist in `handoff/ultracode-planning-togt-s2-2026-06-13/proto_envelope_topp.py`). Emit contact-free, non-saturated, dead-centre line. Architecture-neutral: feeds hybrid reward NOW + fallback LATER.
3. **[doctrine lever — THE speed knob]** Open style cone one rung: rw_tilt 96→48 FIRST, then free-cone 60°→~70° if gate-4 metric holds. Gated on LAPTOP-INC8-BINDING-GATE-VERIFY.
4. **[retrain]** inc8 on corrected aero (lapse OFF, steps 1+3, ≥4 seeds). Validate map-ON. Crown ONLY after ≥3–5-lap fresh-reset live batch.
5. **[probe]** scipy-SLSQP toy MPCC on `rl_plant.step` → measure real k. Adjudicates decomposed fallback before paying acados/WSL/Adroit cost.
6. **[cross-cutting — PRIORITIZE OVER ANY PLANNER CHOICE]** Estimator: drive KF to <0.05 m 1-sigma at post-gate-3 ~37 m/s window. THE binding VQ2 validity risk; measure first.

---

## §PLANNING-TOGT-S2 (2026-06-13, ULTRACODE-PLANNING-TOGT-S2 workstream) — corrected-aero bounds + gap waterfall

**Source of truth: `handoff/ultracode-planning-togt-s2-2026-06-13/REPORT.md`. Workflow 21 agents, 2.48M tok. All numbers reproduced natively in `.venv` (PYTHONPATH=src). Prototypes: `handoff/ultracode-planning-togt-s2-2026-06-13/{proto_envelope_topp.py, proto_togt_corrected.py, verify_honest_tilt_topp.py}`.**

### 🚩 FALSIFICATIONS (supersede §TOGT-BOUND + §CORNER-PASS shipped numbers)

| Claim | Status | Corrected value |
|---|---|---|
| 4.27 s TOGT planning-valid bound | **FALSIFIED — linear-plant fiction** | ~4.6–4.7 s honest band |
| 4.55 s shipped reference line | **FALSIFIED — linear-plant fiction** | drag-infeasible at planned 51 m/s; real wall ~39 m/s |
| 4.13 s contact-tolerant bound | **DOUBLY INVALID** | ILLEGAL (contact) + drag-infeasible |
| 0.24 s style-cone envelope cost (cornering-only model) | **REFUTED by adversarial verify** | ~5.2 s honest at 60° cone (total-force model) |
| "Inc7/inc5 under-driving" | **REFUTED** | Near point-mass-optimal for their envelope; gap = cone, not policy |

**Why falsified:** the old TOGT planned v_max 51–52 m/s; real quad-drag wall v² caps ~39 m/s. At 55 m/s corrected drag = 158 m/s² vs old linear 11.6 m/s² (13.7× under-model). The ceiling is ROBUST to the plant revision (~4.3–4.7 s) because doubled T/W~8 is eaten almost exactly by the v² drag wall.

**Two independent cross-checks on the corrected bound:** `proto_envelope_topp.py` 90° unconstrained TOPP = 4.574 s vs independent C++ TOGT corrected-aero refined = 4.714 s (delta 0.14 s / 3%, v_max 39.17 vs 39.26 agree <0.3%).

### Corrected-aero tilt-vs-laptime (honest total-force TOPP — `verify_honest_tilt_topp.py`)

| Tilt cap | Honest total-force TOPP | Open-loop feasible? |
|---|---|---|
| 60° | ~9.8–10.6 s | NO (2.2% of lap demands >78.3 m/s²; max req 111.6) |
| 65° | ~8.8 s | NO (1.4% over ceiling) |
| 75° | ~6.7 s | YES (max req 77.8 ≤ 78.3) |
| 80° | ~5.4 s | YES (max req 70.5) |
| 90° | 4.60 s | YES; 80°/90° agree (R_min ~30 m, drag wall = sole ceiling above ~79°) |

**Critical framing:** cornering-only model (proto_envelope_topp.py) gives ~5.35 s at 60° — OPTIMISTIC WRONG MODEL. The physically correct definition = tilt of total specific force (braking dominates). Honest model puts inc7 9.76 s right at the 60° bound → tilt-cone = the binding kinematic constraint, not planning.

### Gap waterfall (corrected, 2026-06-13)

| Rung | From→To (s) | Delta | What it is | Status |
|---|---|---|---|---|
| 1 | 35.30→9.76 | 25.54 | Model-based→RL collapse (k=1.85 dilation + alt-relay + start-transient + descent-caution) | **CLOSED — inc7 live-confirmed** |
| 2 | 9.76→6.89 | 2.87 | R4 tilt-cap style tax (2.63 s clean A/B + 0.24 s lineage offset; the 0.24 s NOT envelope-recoverable) | **OPEN — biggest lever: rw_tilt 96→48 (~1.4 s) + free-cone 60°→75–80° (~1.5 s on 3 high-kappa corners)** |
| 3 | 6.89→4.72 | 2.17 | Policy vs corrected-aero point-mass bound (straight-line under-driving + corners); proxy-reconstructed | **OPEN — lower confidence; ~1.5–1.9 s pure policy slack + small straight residual** |

Sum = 30.58 s = 35.30 − 4.72 (verified). Honest endpoint = 4.72 s; contact-valid = 4.43 s (bound_free, gate-4 planning margin only +0.046 m — razor-thin).

### Trajectory-opt panel (2-lens judge)

| Approach | TIME | ROBUSTNESS+VALIDITY | Read |
|---|---|---|---|
| Monolithic RL (inc7-class) | winner (measured) | 7.5 | ONLY live-confirmed contact-free validity |
| MPCC tracker | 6.0 | ~hybrid | Fastest paper ceiling (~5.4–5.8 s); unbuilt (acados/WSL); k=1.85 is the only measured datum |
| Hybrid (decomposed line + RL/MPCC) | 5.5 | 6.0 | Best-calibrated est. (8.0 s offline) — calibration says SLOWER than live monolith |
| min-snap + coupled-TOPP | 4.5 | 6.5 | Free deterministic line generator, not a time frontier |
| TOGT collocation | 4.0 | 4.5 | Best ceiling; worst realizability ratio; demands 17 rad/s body rates vs ~11.2 rad/s plant ceiling → role = offline BOUND + geometry SEED only |

### 🚩 BINDING VQ2 RISK = ESTIMATOR (converges with §CASE-C-READINESS ②+③; VERDICT BANKED in §ESTIMATOR-RACESPEED)

**✅ ULTRACODE-ESTIMATOR-RACESPEED COMPLETE (2026-06-13).** Vision world-fix East σ 0.47 m (UNFILTERED) = 3.0× the gate-4 0.155 m contact-true margin. Deployable in-plane ~0.55 m (absolute NO-GO, speed-flat). **FIX = gate-relative observation** (CONDITIONAL-GO, velocity-prior-sensitive, straddles margin). **Q① = MASTER GATE.** Full quantified verdict: §ESTIMATOR-RACESPEED.

### Organizer Q① + Q⑤ = CRITICAL PATH (updated priority)

- **Q① (does VQ2 stream pose?)** determines whether the estimator risk is binding or moot.
- **Q⑤ (eval-HW GPU?)** determines whether in-loop vision latency L flips from edge-class (~5–15 ms, benign) to CPU-class (~112–139 ms, breaks case C).
- These two answers gate the entire ESTIMATOR vs PLANNER priority ordering. Send email immediately.

### Cheap follow-ups (architecture-neutral)

1. Native corrected-aero min-snap + coupled-TOPP line generator (feeds hybrid reward NOW + fallback LATER; proto exists).
2. Pull inc7 per-tick gate crossings from ShadowPC (gitignored debug_obs) to firm rung-2/3 attribution.
3. scipy-SLSQP toy MPCC on `rl_plant.step` — the one number the decomposed ceiling depends on.

---

## §SUBSTRATE-AUDIT (2026-06-13, ultracode adversarial correctness audit; HEAD c2af65e; full report handoff/ultracode-substrate-audit-2026-06-13/REPORT.md)

**68/68 claims dispatched; 0 UNVERIFIED; 61 CLEARED; 4 CONFIRMED_BUG; 2 NEEDS_LIVE; 1 INCONCLUSIVE.**

### Summary verdict
No live-code train/deploy-corrupting bug survived escalation on the deployed VQ1 course. The deployed convention chain — ODOMETRY quat R_y(π) conjugation, rate sign [-1,-1,-1], FLU→FRD wire [+1,-1,+1], velocity single-rotation, super-rate gain map, gate-frame lift, 17-dim obs seam, DiffAero↔rl_plant parity, camera mount, MAVLink layouts — was adversarially attacked with fresh external invariants anchored on pristine vel_ned/pos_ned and **held**. Clean bill is **SCOPED TO VQ1**.

### Confirmed bugs (none blocking VQ1 deploy)

**P4-C05 — obs_from_zup/build_obs hardcode gate yaw = pi (VQ2 hazard, DORMANT on VQ1):** `_R_W2G = diag(-1,-1,1)` and `_GATE_YAW_REL = 0` hardcoded in `rl/fly_rl.py` and `offline_rollout.py`. VQ1 all-pi course: train vs deploy max|diff| = 4.44e-15 (bit-exact — completely invisible). Non-pi course (VQ2): consumed-obs max|diff| up to **4.22 m** (pos_gx 1.11, vel_gx 1.61, nxt_rely 3.29 m). **ACTION: GO-BEFORE-VQ2 (gated on ULTRACODE-ESTIMATOR-RACESPEED — entangled with its gate-relative-obs exploration).** Fix: thread per-gate yaw into `obs_from_zup`/`build_obs`; add a loud deploy-time assert `all gate_yaws == pi` when hardcoded path runs. `rl/contact_true_eval.py` likely shares this hardcode — flag for the same fix when next worked (OUT-OF-SCOPE for this audit).

**CR2-01/CR4-01/CR4-03 — refit dataset encodes bcc93f9-era wire map (provenance defects; shipped code CORRECT):** The `refit` recording set was captured under the SUPERSEDED `bcc93f9` code. Its `obs[8]`/`obs[11]`/`rate_frd[2]` yaw channels encode the OLD convention (HYBRID roll/yaw split). `refit` yaw corr vs CURRENT wire = -0.935 to -0.992; vs OLD bcc wire = +0.979. **DATA-HYGIENE RULE: use POSTFIX dataset (`handoff/shadowpc-postfix-dataset-2026-06-12`) for any current-code obs/rate_frd yaw clearance; use refit ONLY as AS-IS/superseded-yaw positive control.** Recommend a producer-commit field on recordings.

### NEEDS_LIVE items (both pending a single yaw-active live capture)

**P1-C06 — obs yaw seam (obs[8] rpy_g_y, obs[11] w_fluz):** only current-convention recording is inc6 postfix; inc7 seam not directly captured. Predicted to pass (obs construction is checkpoint-independent); needs one inc7 debug_obs run with yaw excursions.

**CR1-01 — absolute yaw wire sign (_ACT_FLU_TO_FRD[2] = +1):** every offline closed-loop yaw lens self-correlates regardless of sign — cannot falsify offline. The yaw sign rests on live-confirmed inc6/inc7 flights (5/5 standing, 0/5 contact). **Live probe: pure-yaw-step segment in inc8 fresh-reset batch.** Bundle P1-C06 + CR1-01 into the WINNER-VALIDATION RIDER's pre-crown live batch.

### INCONCLUSIVE

**P5-C04 — Elodin adapter omits orientation_ned_wxyz:** would cause 57.9° attitude error → 17.9 m/s² specific-force error if ever wired. Currently UNREACHABLE (no Elodin→Navigator path exists). Becomes deploy-corrupting only when/if an Elodin solver-glue + Navigator eval runner is built. Action: add assert before `navigator.py:298` so it fails LOUD if ever connected.

### COLL_MAP reconciliation (supersedes any "train-corrupting bug" framing)
Audit's first-pass "COLL_MAP +2.83 m/s² body-up over-prediction at knots 6-9" RE-ADJUDICATED **NOT A BUG** — bare offline `force_model` over-predicts ~13% uniformly (reconstruction artifact, not a table error). COLL_MAP table CLEARED (P2-C04). **Do NOT refit QUAD_DRAG** (CR5-01 confirmed: the ~1.1 m/s² Down residual lives in the THRUST column, not drag; zeroing drag leaves a +5.88 m/s² along-thrust deficit). Consistent with inc7 margin doctrine.

### Operational notes
- `rl/checkpoints/inc7_staging/s0_actor.pth` ships **NO sidecar** (clamp-safe only via the 3.14 legacy default); ship its sidecar OR confirm `stage1_inc7_actor.pth` (which HAS the sidecar) is canonical and drop the staging copy.
- Super-rate `|cmd|=pi` discontinuity is real but **UNREACHABLE**: grand max |act_rate| = 2.717 < pi by 0.42 rad/s.
- Durable clarification: `obs[12]` = previous RESCALED normed_thrust `[0, act_max]` g-units, NOT the [0,1] collective fraction. Deploy matches training; in-code label being corrected in the hardening pass.

### Regression suite (promote → tests/)
8 external-invariant scripts in `handoff/ultracode-substrate-audit-2026-06-13/regression_suite/`. All exit 0 + pass pytest standalone. **DISPATCH QUEUED (GO-next): promote all to `tests/` after resolving the slug-collision note (test_confirmed_cr4_03.py previously held a COLL_MAP test — re-home that under `test_collmap_overpredict.py` before promoting).** Suite encodes the invariants internal-consistency checks are structurally blind to (R_y(π) bug class that bit 4x).

| Test file | Catches |
|---|---|
| `test_frame_force_vs_fd_mirror_canary.py` | Re-introduced R_y(π) conjugation (deploy + vision) |
| `test_train_deploy_obs_elementwise.py` | 17-dim obs seam drift (layout, gate-frame, rate sign, virtual-flip, obs[12] memory) |
| `test_twin_diffaero_extreme_parity.py` | Twin/rl_plant↔DiffAero divergence at extreme states |
| `test_mavlink_velocity_single_rotation.py` | Body-vs-world velocity-frame mix (c3b5a8e) |
| `test_confirmed_p4_c05.py` | obs_from_zup hardcoded yaw=pi (VQ2 hazard) |
| `test_confirmed_cr2_01.py` | Yaw-about-vertical rate sign alias |
| `test_confirmed_cr4_01.py` | Yaw-channel R_y(π) + refit stale-yaw provenance |
| `test_confirmed_cr4_03.py` | refit recorded wire validates OLD bcc93f9 map |

### KF accel_body convention (CR1-01 rider — fold into SHADOWPC-VISION-CAL)
`navigator.py:313` KF-predict rotates raw `accel_body` by the TRUE-conjugated attitude vs doctrine "accel_body pairs with RAW quat" (7.5 m/s² East error @ bank). Recordings lack accel_body. **Action: add HIGHRES_IMU accel_body logging to SHADOWPC-VISION-CAL, adjudicate from live data. Do NOT fix the convention blind.**

---

## §ESTIMATOR-RACESPEED (2026-06-13, ULTRACODE-ESTIMATOR-RACESPEED; 12 opus agents, 1.44M tok; report handoff/ultracode-estimator-racespeed-2026-06-13/REPORT.md)

**Answers the binding VQ2 validity question converged on by §CASE-C-READINESS (②+③) + §PLANNING-TOGT-S2.** Offline analysis only (no src edits, no SLURM, no live sim). Every load-bearing number reproduced by adversarial verifier + commander re-run; verifier corrections flagged where they changed the headline.

### Verdict

**Case-C ABSOLUTE world-frame KF nav = NO-GO at race speed, by a wide and speed-flat margin.**

Measured filtered in-plane error at gate-4 (~37 m/s, 0.66 s window, ~9 fixes):
- **Deployable (b1-corrected): ~0.55 m RMS** — ≈3.5× the 0.155 m contact margin, ≈11× the 0.05 m bar.
- De-biased idealized (variance-only floor): ~0.33 m (commander re-run: 0.28 m). Both decisively over both thresholds.
- **SPEED-FLAT: invalid even at 8 m/s.** No inc8 cone rung is estimator-valid on the absolute path at any speed.
- 18-cell sweep (L ∈ {6,16,112 ms} × acceptance {47,25,10%} × {raw,de-biased}): **NO cell clears 0.05 m.**

### Binding terms — two independent sources, not one

**(i) VARIANCE floor ~0.50 m/axis** — per-fix noise is large AND velocity is unobservable in case C (KF averages only ~3–9 fixes in the 0.66 s window; cannot reach steady-state). Floor probe: only per-fix σ ≤ ~0.03 m clears the 0.05 m bar — a ~16× per-fix accuracy improvement the absolute path cannot reach.

**(ii) Un-filterable per-track BIAS** (range-collapsing). ⚠️ **SUPERSEDES the old a2 0.52 m constant figure**: b2 confirmed direction but corrected magnitude — **≤27 m: 0.52 m; ≤12 m: 0.34 m; ≤9 m: 0.19 m (CI [0.11,0.25]); last accepted fix ~8.25 m: 0.109 m.** Bias survives global de-bias (per-gate residual is not global); UN-REMOVABLE in pure case C (chain-circularity — a cal lap cannot close the loop without pose). Carry **0.19 m** as the defensible near-band per-track bias floor.

### Dead ends confirmed (do not spend on these)

| branch | attacks | best-achievable in-plane | verdict |
|---|---|---|---|
| Higher cadence (c2) | VARIANCE only | floors 0.106 m at impossible 240 Hz; 0.34 m at 30 Hz | DEAD END: bias-blind, saturates above both bars |
| Lower cov floor (c4) | (intended VARIANCE) | NET-NEGATIVE: ≤15 mm gain, +51 mm bias regrowth, 14–36% over-rejection | TRAP: cov floor is correctly MLE-sized to 0.407 m |
| Better detector accuracy (c3) | VARIANCE (pixel) | **0.000 m** — detector already sub-pixel (reproj 0.50 px) | DEAD END: in-plane noise is attitude-lever + floor limited, not pixel limited |

### THE FIX — Gate-relative observation (rank-1 branch)

**Observe the offset to the SEEN opening (−L from PnP).** Map bias `db` drops out of the arithmetic EXACTLY — converts absolute-NED variance to close-range corner-reprojection noise.

Achievable in-plane miss: **0.11–0.21 m RMS — CONDITIONAL-GO on the 0.155 m contact margin** at 37 m/s. Does NOT clear the strict 0.05 m bar (that needs per-fix lateral ≤ ~0.08 m).

**"subtract gate_map pos" = ANTI-PATTERN** — relocates the bias to the control target (estimator E-bias still +0.17 m), does NOT remove it.

**Implementation notes:**
- AUGMENT, do not replace, the absolute LinearKF (keep for planning/feedforward/g4→g5 hand-off).
- Gate-relative term owns only the terminal in-plane miss.
- Observe the offset to the SEEN opening (−L from PnP); the "subtract gate_map" form is proven anti-pattern.
- Add one relative-innovation outlier gate (reproj alone does NOT separate depth-flips; flip p50 0.71 px < clean 1.02 px).
- **Architecture implication (MONOLITH-consistent):** policy obs must be gate-relative, NOT absolute world-frame — an obs-formulation change on the inc8 MONOLITH, built on P4-C05 (per-gate-yaw) as its foundation.

### Commander refinement: the margin-clear is velocity-prior-sensitive (CONDITIONAL, not comfortable)

c1 reported 0.11–0.14 m (per-fix/~2.8 with warm velocity prior). Commander cold-prior re-run: **0.17–0.21 m (per-fix/~1.6), OVER the 0.155 m margin.** Fork is real: case-C velocity is observable only through position-fix differencing. Effective velocity prior entering the gate-4 window sits between cold and warm → **margin-clear STRADDLES the 0.155 m line.** Resolved by: (a) a full-lap case-C sim (pure offline); (b) at-speed recording. Mildly **reopens the deferred vision-velocity channel** (P3-1 from §CASE-C-READINESS) as a margin lever — the velocity prior is the swing variable.

### Latency

**In-plane BENIGN at edge HW** (5 mm leak; latency cost is along-track). CPU-class naive update is CATASTROPHIC (NEES 411, +4.3 m along-track, in-plane 0.83 m) → **RewindKF MANDATORY on CPU-class HW.** Ship it regardless — cheap on edge, decisive on CPU. No horizon<L divergence risk at these L (≥0.37 s margin even at CPU p90). See §CASE-C-READINESS piece B.

### Speed-ladder coupling (c5)

Per-fix lateral ≤ 0.10 m → 0.155 m margin held past 55 m/s (full inc8 ladder valid on the margin). Per-fix 0.05 m → 0.05 m bar holds only to ~26 m/s; 0.03 m → ~50 m/s. **To fly ~37 m/s inside the margin: gate-relative per-fix lateral must reach ≤ ~0.10–0.13 m** (vel-prior fork sets exact threshold). **Re-verify every inc8 cone-relaxation rung against the achieved gate-relative per-fix σ at that rung's speed.**

### Dependencies

- **Q① (case A/B vs C) = MASTER GATE.** If VQ2 streams `LOCAL_POSITION_NED`/`ODOMETRY`, pose is pristine and this entire risk is MOOT. Resolve Q① BEFORE building gate-relative.
- **Q⑤ (eval-HW class)** flips latency posture: edge = free; CPU = ruling, RewindKF mandatory.
- **ONE ShadowPC at-speed (~37 m/s) gate-4 recording** — the single decisive measurement. All current data ≤8.4 m/s; 37 m/s motion-blur multipliers MODELED (best-case lower bounds). Collapses: blur/exposure pivot (×1 vs ×2), per-fix accuracy at speed, gate-relative per-fix-lateral noise, acceptance, first-accept range, AND tests for the one-signed unmeasured gate-relative extrinsic systematic (n=3). **Add to SHADOWPC-VISION-CAL agenda.**
- **Full-lap case-C sim** (offline, pure computation): pins the velocity-prior quality entering gate-4 — resolves the 0.11 vs 0.21 m fork.

### Cross-references (supersessions)

- §CASE-C-READINESS (this file): "binding risk = unmeasured in-loop L" — REFINED. L remains binding for the rewind buffer. The estimator's absolute path is now NO-GO; gate-relative path is the forward architecture. P3-1 (vision-velocity) mildly re-opened as margin lever.
- §PLANNING-TOGT-S2 §BINDING VQ2 RISK = ESTIMATOR: dispatch confirmed with quantified verdict: 0.55 m deployable (absolute) vs 0.155 m margin → NO-GO. Gate-relative CONDITIONAL-GO.
- a2's 0.52 m constant bias → SUPERSEDED by b2's range-collapsing 0.19 m near-band (favorable direction, carries same NO-GO direction).

### Artifacts

All under `handoff/ultracode-estimator-racespeed-2026-06-13/`: `a1_sim.py` / `a1_results.json` (closed-loop KF sim, reproduced bit-for-bit by b1); `a2_bias.py` / `b2_verify.py` (bias decomposition + verifier); `a3_realism.py` (37 m/s model); `a4_latency.py` (latency geometry split); `c1_*` (gate-relative THE FIX); `c2_*`–`c5_*` (dead ends/traps/speed-ladder); `synth_synthesis.md` + `REPORT.md`.

---

## §ORGANIZER-PIVOT (2026-06-13) — stop gating engineering on organizer answers

**Context:** Three organizer emails (sent ~5 days, ~2 days ago, and 2026-06-13) UNANSWERED as of 2026-06-13.

**DIRECTIVE (durable): Build the ROBUST SUPERSET that handles every plausible case — do not gate engineering on organizer answers.**

### Supersession of Q①-gated framing
**SUPERSEDED:** "Q① is the master gate; build gate-relative ONLY after Q① confirms case C."
**CURRENT:** Build gate-relative regardless. Q① only decides whether it's **load-bearing** (case C, no pose streamed) or free insurance (case A/B, pose given). Rationale: case A/B is covered by given-pose; ⑤'s verdict proved gate-relative is the ONLY case-C path → building it is free insurance either way. STOP treating Q① as a build gate; treat it as a routing signal.

### Other unknowns — already handled defensively
- **RewindKF = DEFAULT** (free at edge HW, mandatory at CPU HW — covers both Q⑤ branches; already built).
- **Autonomy unknowns** (start sequence, telemetry reliability) handled by late-join gate + arm-retry + freshness gates from hardening (④ above). No organizer answer needed.
- **Keep ONE more polite nudge in ~a week.** Plan as if no answer comes; organizer email is no longer a blocking gate.

---

## §RL-PORTFOLIO (2026-06-13) — policy speed-ladder as deliberate portfolio

**Commander-assessed, GO as a refinement of the inc8 plan. NOT a new track — layered on §INC8-DESIGN + §S2-DECISION.**

### Framing
Train a deliberate LADDER of policies along the reliability↔speed (tilt/cone-envelope) spectrum; SELECT the fastest one that is VALID at race conditions. Right response to ⑤'s estimator-gates-speed reality (we don't yet know the valid-speed ceiling).

### Portfolio rungs (already in plan, now explicitly framed)
- **inc7** = the conservative guaranteed-valid floor (11.45 s deployment, sr=1.000, gate-4 confirmed).
- **inc8 rung 1:** rw_tilt 96→48 (~1.4 s recovery, cone unchanged).
- **inc8 rung 2:** free-cone 60°→~70° (~1.5 s recovery, 3 high-κ corners — THE binding kinematic lever), gated on gate-4 metric.
- **inc8 rung 3+:** further relaxation if gate-4 metric and estimator σ permit.

### Selection metric
Fastest policy where:
1. Required localization is achievable by the gate-relative estimator at that rung's race speed (ties to §ESTIMATOR-RACESPEED margin/σ analysis).
2. Contact-valid (gate-4 margin ≥ 0 with adequate clearance; re-verify via `rl/contact_true_eval.py` per rung).
Keep inc7 as the **guaranteed-valid floor fallback** ("slow valid > fast invalid" doctrine concretized).

### Rejected end of the axis
- **"Crazy flips / aerobatics":** ANTI-SPEED AND ANTI-VALIDITY for this race — rejected. The productive axis is TILT CONE, not maneuver complexity.
- **Perturbation-recovery curricula:** REJECTED (basin-narrowing, confirmed prior).

### Coupling to estimator
Every cone-relaxation rung worsens gate-4 margin/σ ratio → **re-verify gate-relative per-fix lateral against achieved σ at that rung's speed before crowning any rung as valid.**
