# 02 — System & Stack

The training code lives in `rl/` (key files: `peregrine_racing_ego.py`, `ego_reward.py`,
`ego_estimator.py`, `gate_visibility.py`, `peregrine_course.py`, `racing_line.py`,
`vq2_ego_curriculum.py`, `peregrine_train_ego.py`, `peregrine_vq2_ego.sbatch`). It builds on
DiffAero (a GPU racing-RL library; a clone lives on Adroit at `/scratch/network/fl3689/diffaero_repo`).

## Observation (actor) — 21-dim, position-free, body-frame

Defined in `peregrine_racing_ego.py`. `WINDOW = 2` (current gate + next), `PER_SLOT = 5`.
`EGO_OBS_DIM = 9 + 2 + WINDOW*PER_SLOT = 21`:

- **9** dynamics/state: body-frame velocity (3) + roll,pitch (2) + body angular rates (3) + last
  collision flag (1).
- **2** coarse sector: a very coarse discretised bearing hint to the current target (a static coarse
  map, not a position).
- **2 gate slots × 5 each = 10**: per gate `rel_pos(3)` + `confidence(1)` + `visible_area(1)`.
  - `rel_pos` is the **drone→gate vector in the body frame (FLU: Forward-Left-Up), metric** — it
    carries **range**, not just bearing. E.g. a gate seen up-and-to-the-right ~6 m ahead reads
    `rel_pos ≈ [+6.0, −1.5, +1.2]` (forward +6, right = −left = −1.5, up +1.2).
  - `confidence ∈ [0,1]` — staleness + fix quality; 0 = masked (gate not currently detectable).
  - `visible_area ∈ [0,1]` — normalised apparent opening area (1 = square-on). A **range-free,
    coarse foreshortening proxy** for "how head-on am I", used *instead of* a gate-normal channel
    (we tried gate-normal; rejected — see [08]).
  - A masked slot (past the last gate, or gate stale/undetectable) zeros all 5.

The obs is **heading-invariant** (egocentric) — a global course rotation is a redundant DOF, which is
why the course sampler pins spawn heading (see below).

## Privileged critic — 16-dim, ground-truth, asymmetric

The critic sees a **privileged** state the actor cannot: true body-frame velocity (3) + true
roll,pitch (2) + true body rates (3) + `WINDOW × (true rel_pos 3 + confidence 1) = 8`. Total 16.
(An older variant computed 20; the shipped one is 16.)

🚩 **`algo=appo` is MANDATORY (GuardedAPPO).** With plain `ppo`, DiffAero wires a *symmetric*
obs-only critic and the privileged state is silently never connected → **seed-collapse** (this was
the inc8 root cause). The launcher asserts `critic.input_dim == state_dim (16) != obs_dim (21)`;
a mismatch raises rather than trains a broken run.

## Algorithm & exploration

- **GuardedAPPO**, `γ = 0.9975` (load-bearing — long horizon needed for the sparse crossing reward).
- **Noise anneal** (`rl/inc8_noise_anneal.py`, wired in `peregrine_train_ego.py`): DiffAero's
  `StochasticActor` samples `tanh(mean + std·N(0,1))` with a learnable state-independent
  `actor_logstd`. Left alone with a constant entropy bonus, std runs to the `exp(2)=7.39` ceiling and
  the *mean* becomes an unvisited, unstable operating point (policy flies "in expectation over chaos"
  but its deterministic mean crashes). The anneal **clamps `actor_logstd` to a decaying std ceiling**
  (`std_hold → std_floor` geometrically over the back `1−hold_frac` of training) and anneals the
  entropy weight to 0. Applied per-PPO-update, before each rollout.
  - 🚩 **The held noise floor is a structural precision cap:** the deterministic mean cannot sharpen
    below the noise it was trained under. Typical values used: `std_hold` 0.12–0.30, `std_floor`
    0.02–0.05, `hold_frac` 0.35–0.5.
- **Warm-start lifeline** (`peregrine_train_ego.py`): load actor+critic from `+init_from`; reset the
  exploration `logstd` after load (`warmstart_reset_logstd_std=0.18`) so we transfer *means* not the
  ground-down schedule; optional **critic warmup** (`critic_warmup_updates=N`, default 100) zeros
  actor gradients for the first N updates so the privileged critic re-adapts to a new return scale
  before the policy moves. Periodic + emergency checkpoint saves.

## Environment & reward (`peregrine_racing_ego.py`, `ego_reward.py`)

Reward is "refined-B" (`EgoRewardWeights.from_cfg`). The crossing / contact contract is inherited
verbatim from inc7 (`crossing_events`, `world_to_gateframe` in `peregrine_racing.py`). Terms:

- **Progress** (`rw_progress`): either isotropic 3D homing toward the gate centre
  (`rw_progress_to_center=True`, potential `−‖pos−centre‖` with an optional vertical weight), **or**
  arc-length progress along a curved racing line (`use_racing_line=True`, `racing_line_progress=True`).
  The racing-line form won the "which homing" race (see [04]) because the head-on line encodes the
  altitude profile (arrive horizontal), whereas isotropic homing pulls diagonal-to-centre and permits
  a floor sink.
- **Passage** (`rw_passage`): sparse bonus for crossing inside the aperture (+ an increment per gate).
- **Corridor / contouring** (`rw_corridor`): a **PBRS (potential-based, telescoping)** cross-track
  pull onto the racing line — `k·(perp_prev − perp_curr)`. This is the *"D" term* (rewards *change* in
  offset). k=4 is the working weight; k≥10 collapses under anneal (see [06]).
- **Magnitude centring** (`rw_centering`, clamp `rw_centering_max_m`): a **dense** penalty
  `−k·clamp(perp, 0, max)` on the line-perp. This is the *"I" term* — it nags a **sustained** standing
  offset continuously (the gradient the telescoping PBRS lacks). Only safe via warm-start (fresh → floor).
- **Parabola crossing** (`rw_parabola_crossing`, `crossing_parabola_reward`): the smooth centring
  reward that finally worked. At a forward plane crossing, `r = clamp(cross_center·(1 − (Linf/R)²),
  −cross_neg_cap, cross_center)` — `+cross_center` dead-centre, 0 at the "zero radius" `R =
  rw_cross_zero_m`, growing negative outside, capped. Replaces the passage + frame/miss terminal cliff
  with one smooth monotone bowl (no moat). **We added an in-run anneal of `R`** (see [05]).
- **Terminal** (`terminal_penalty`, `rw_terminal_base/miss/oob`, progress-scaled): contact 100
  (Fengyou-authorised DQ-scale), miss 100 (equalised to remove a cheap-bail incentive), oob 200. A
  `forfeit_mask` optionally forfeits banked progress on floor-only.
- Small regularisers: `rw_rate`, `rw_dact`, `rw_tilt_free_rad`, `rw_altitude_hold` (used in a probe).

## Perception model (`ego_estimator.py`, `gate_visibility.py`)

A **batched relative-state estimator** — deliberately *not* a world Kalman filter. Per gate it
maintains a smoothed body-frame `rel_pos`, a staleness/quality `confidence`, a `normal_conf`, and the
`visible_area` foreshortening ratio. It propagates through no-fix gaps (rotate by inter-frame body
rotation, translate by `−v·dt`), masks when stale, and injects IMU-like noise (accel white ~0.008,
gyro AR(1) ρ0.75 bias-drift). Gate visibility uses an 8-keypoint ≥4-of-8 + occlusion model. **No world
position, no gate-normal in the obs.**

## Course sampler (`peregrine_course.py`)

`sample_courses` draws spec-plausible layouts. For the single-gate stages:
- Gate at **depth 8–15 m** (`course_spawn_dist`), **height ±6 m** (`course_spawn_below_g0`).
- `spawn_heading` **pinned to 0.0** for ego (a global rotation is redundant for the heading-invariant
  obs; leaving it random just scatters the world-frame render into a confusing circle).
- `spawn_yaw_jitter_rad` (`course_spawn_yaw_jitter`): jitters the **drone's** spawn yaw ±jitter off the
  gate bearing so the gate appears **left/right across the FOV** (real lateral training signal — it
  changes the body-frame `rel_pos[1]`). **Default 0** — which means the champion lineage saw the gate
  *always dead-centre horizontally*. Added this session (see [04], [07]).
- Spawn is **tail-first**: `spawn_yaw = gate_yaw + π` (the drone trains flying tail-first; the deploy
  wrapper `fly_rl.py` runs a virtual π body-z flip to compensate — a self-consistent legacy convention,
  do NOT "fix").

## Racing line (`rl/racing_line.py`)

Online, batched, **non-optimal** guiding line: a cubic Hermite through `[spawn, gate centres]` with
the tangent at each gate = the gate normal, giving **head-on crossings** (G1-continuous, not
time-optimal). `query()` returns `(arc-length s, cross-track perp, tangent, inward_unit)`. Equivalent
to a guiding vector field `F = normalize(tangent + k·(centre − pos))`; flow lines `y = C·e^(−kx)`,
`k = align_gain` = convergence tightness. Used only by the *reward* (progress s + contouring perp),
never by the position-free obs.

## Curriculum, harness, deployment

- **Curriculum** (`vq2_ego_curriculum.py`): named stages → hydra overrides. `_COMMON` holds shared
  knobs (terminal weights, etc.). Stages set course ranges + reward knobs + a `_raw` block for
  `env.max_time`, `algo.gamma`, `+init_from`, and `++algo.noise_*` overrides.
- **SLURM** (`peregrine_vq2_ego.sbatch`): `sbatch --export=ALL,SEED=0,RUNTAG=<tag>,STAGES=<stage>,
  UPD_<stage>=<n> …`. A `BOUNDARY_OV` array always appends `+warmstart_reset_logstd=true
  +warmstart_reset_logstd_std=0.18 +critic_warmup_updates=100`. n_envs 2048. Metrics → tfevents;
  the box-exit classifier logs `exit_{thread,plane,frame,floor,ceiling,side,back,front,timeout}` +
  `cross_offset_m` (L-inf of every plane crossing) + `pass_offset_m`.
- **Deploy** (not exercised in this dossier's runs, but the target): ARM (MAV_CMD 400 p1=1) then
  stream `SET_ATTITUDE_TARGET` body-rate (CTBR, FRD rates, thrust[0,1]) ≥30 Hz. The competition wire
  has **no magnetometer, no barometer** — yaw and altitude must come from vision; the ESKF is a
  roll/pitch leveller only.
