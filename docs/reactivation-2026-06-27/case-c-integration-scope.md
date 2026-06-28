# Case-C End-to-End Integration Scope (VQ2 self-localizing loop)

**Author:** background recon/scoping agent · **Date:** 2026-06-28 · **Status:** SCOPE ONLY (no code touched)
**Audience:** the future build agent who wires case-C. Act from this doc cold.

---

## 0. The one-paragraph verdict

The **vision → localization → obs → policy** half of the case-C loop is **WIRED and pinned byte-faithful** (golden-tuple 7/7, `#37` emulator-fidelity sweep at ≤1e-6 / 2.93e-8). The **IMU → attitude** half is **NOT wired**: the navigator still reads attitude from the ODOMETRY quaternion (`navigator.py:416`) and copies `roll/pitch/yaw/angular_rate_body` straight off the wire into `NavState` (`state_estimator.py:215-216`). VQ2 §9.3 blocks `ATTITUDE` / `LOCAL_POSITION_NED` / `ODOMETRY` / `GATE_INFO`, so on the scored wire `orientation_ned_wxyz` is `None` and those Euler/rate fields are zero → the loop has **no attitude source**. **THE central gap is a single seam: an AHRS (ESKF picked) must produce `R_wb` + Euler + body-rates from `HIGHRES_IMU`, replacing the ODOMETRY read at exactly two consumption sites.** Seven standalone AHRS filters exist and are validated in tests but **none is plugged into the nav loop**. Everything downstream of attitude is ready and gated-OFF-byte-identical.

---

## 1. End-to-end data-flow (case-C)

Legend: **[EXISTS+WIRED]** code path is live in `Navigator.update`; **[EXISTS/UNWIRED]** code exists, validated in tests, but not in the live nav loop; **[MISSING]** no code.

```
                          ┌──────────────────────────────────────────────────────────────┐
   HIGHRES_IMU            │  L1  ATTITUDE / AHRS                                          │
  (accel,gyro,mag) ──────►│  [MISSING in loop / filters EXIST/UNWIRED]                     │
   30..?Hz, FRD           │  ESKFAHRS.step(gyro,accel,dt,mag)->q_wxyz                      │
                          │    src/racer/ahrs/eskf.py:191  (picked filter)                 │
                          │  outputs: q_wxyz(168), gyro_bias(173), attitude_unc(178)       │
                          └───────────────┬──────────────────────────────────────────────┘
                                          │ R_wb (body->world), euler(r,p,y), w_body
                                          ▼
   ┌───────────────────────────── CURRENT WIRED SOURCE (VQ2-illegal) ──────────────────────┐
   │ navigator.py:416  R_wb = R_world_from_odo_quat_wxyz(ds.orientation_ned_wxyz)           │ ◄── GAP #1
   │   (frames.py:167)  ← ds.orientation_ned_wxyz is the BLOCKED ODOMETRY quat              │
   │ state_estimator.py:215-216  NavState.roll/pitch/yaw, angular_rate_body = ds.*          │ ◄── GAP #2
   │   ← these come from ODOMETRY too (contracts.py:73-81)                                  │
   └───────────────────────────────────────────────────────────────────────────────────────┘
                                          │ R_wb
        accel_body (FRD) ─────────────────┤
                                          ▼
   ┌────────────────────────── L3  POSITION/VELOCITY KF ──────────────────────────┐
   │ LinearKF.predict(accel_body, R_wb, dt)   state_estimator.py:110  [EXISTS+WIRED]│
   │   a_world = R_wb @ accel + g ; 6-state [pos,vel] world-NED                     │
   │   (RewindKF OOSM wrapper when cfg.use_rewind_kf, navigator.py:_rewind)         │
   └───────────────┬──────────────────────────────────────────────────────────────┘
                   ▲ position fix(es)                         ▲ given pos/vel
                   │                                          │ (VQ1 case-A ONLY;
   ┌───────────────┴─────────────── L2 VISION ────────────┐   │  cfg.use_given_position,
   │ JPEG Frame                                            │   │  navigator.py:436 — OFF for VQ2)
   │  └► detector.detect(frame)->[GateObservation]         │   └── (no wire pos in VQ2)
   │       vision/detector.py:171 (GateDetector)           │
   │       vision/detector.py:223 (EnsembleGateDetector)   │   [EXISTS+WIRED via cfg.use_vision]
   │       corners_px (4 inner kpts; 8->4 slice)           │
   │  └► estimate_gate_pose(obs)->GatePose  [PnP]          │
   │       (camera-frame R_cam_gate,t_cam_gate,cov)        │
   │  └► associate to map Gate (navigator._associate)      │
   │  └► localization → +L world fix:                      │
   │       gate_pose_to_world_position()  localization.py:91   (absolute)           │
   │       gate_relative_inplane_fix()    localization.py:161  (cfg.use_gate_relative)│
   │       gate_range_fix()/apparent_range_from_gate_span localization.py:230,275    │
   │                                          (cfg.use_range_channel)                │
   │     position_ned = gate.position_ned - (R_wc @ t_cam_gate)   ← +L convention    │
   │       NOTE: R_wc = R_wb @ R_camera_from_body().T  ← also depends on R_wb (GAP)  │
   │  └► innovation-gated kf.update_position(z, cov)        │
   └───────────────────────────────────────────────────────┘
                   │
                   ▼
   ┌────────────────────────── NavState assembly ────────────────────────────────┐
   │ make_nav_state(kf, ds, tsv, nav_inplane_sigma, nav_along_sigma)              │
   │   state_estimator.py:196  [EXISTS+WIRED]                                      │
   │   pos/vel  ← kf  (GOOD)                                                       │
   │   roll/pitch/yaw, angular_rate_body ← ds  (BLOCKED in VQ2 — GAP #2)           │
   │   nav_inplane_sigma/along_sigma ← Navigator._gate_frame_pos_sigma navigator.py:677│
   └───────────────┬──────────────────────────────────────────────────────────────┘
                   ▼
   ┌────────────────────────── L4  OBS[0:20] BUILDER ────────────────────────────┐
   │ estimator_obs20(ds, nav_state, target_gate, last_normed_thrust, ...)         │
   │   src/racer/estimator_obs.py:171  [EXISTS+WIRED, pinned]                      │
   │   obs[0:17]  via estimator_obs()->fly_rl.build_obs/obs_from_zup (fly_rl.py:307)│
   │     [0:3] pos_g=R_w2g@(gate-pos) (+L, pinned) ; [3:6] vel_g ; [6:9] rpy_g     │
   │     [9:12] w_flu  ← derived from attitude+rates (GAP #2 feeds here)           │
   │   obs[17:20] confidence_triple(nav_state) estimator_obs.py:129               │
   │     c_inplane,c_along=clip(σ_ref/σ̂,0,1) ; age_norm=clip(t/τ,0,1)             │
   │     SIGMA_REF_M=0.05 (:91) TAU_STALE_S=0.10 (:92) ; /√2 reconcile (:163)      │
   └───────────────┬──────────────────────────────────────────────────────────────┘
                   ▼
   ┌────────────────────────── L5  INC8 POLICY ──────────────────────────────────┐
   │ load_actor(path) rl/fly_rl.py:479  (obs_dim inferred from head.0 weight:488)  │
   │ policy_step(actor, obs, ...) rl/fly_rl.py:505 -> rate_frd, collective         │
   │   [EXISTS+WIRED in fly path; estimator_obs_auto dispatches 17 vs 20 (:194)]   │
   └───────────────┬──────────────────────────────────────────────────────────────┘
                   ▼
            SET_ATTITUDE_TARGET / rate+thrust  →  (control on the wire)
```

### Per-arrow wiring table

| Arrow | Function / file:line | Status |
|---|---|---|
| IMU → attitude (R_wb, euler, rates) | **none in loop**; `ESKFAHRS.step` `ahrs/eskf.py:191` exists | **MISSING (wire it)** |
| attitude → R_wb for predict+PnP | `navigator.py:416` reads ODOMETRY quat | **EXISTS+WIRED but VQ2-illegal source** |
| attitude → NavState euler/rates | `state_estimator.py:215-216` copies `ds.*` | **EXISTS+WIRED but VQ2-illegal source** |
| accel + R_wb → pos/vel | `LinearKF.predict` `state_estimator.py:110` | EXISTS+WIRED |
| JPEG → corners | `GateDetector.detect` `vision/detector.py:171` | EXISTS+WIRED (`cfg.use_vision`) |
| corners → GatePose (PnP) | `estimate_gate_pose` (called `navigator.py` ~:494) | EXISTS+WIRED |
| GatePose → +L world fix | `gate_pose_to_world_position` `localization.py:91` | EXISTS+WIRED |
| gate-relative fix | `gate_relative_inplane_fix` `localization.py:161` | EXISTS/WIRED (`cfg.use_gate_relative`, default OFF) |
| range channel | `gate_range_fix` `localization.py:275`, `apparent_range_from_gate_span:230` | EXISTS/WIRED (`cfg.use_range_channel`, default OFF) |
| fix → KF | `kf.update_position` (innovation-gated) | EXISTS+WIRED |
| KF → NavState | `make_nav_state` `state_estimator.py:196` | EXISTS+WIRED |
| NavState → obs[0:20] | `estimator_obs20` `estimator_obs.py:171` | EXISTS+WIRED, pinned |
| obs → action | `policy_step` `rl/fly_rl.py:505` | EXISTS+WIRED |

---

## 2. THE GAP LIST

### GAP #1 (CENTRAL) — attitude into the KF predict + PnP rotation
**Where:** `navigator.py:416` `R_wb = R_world_from_odo_quat_wxyz(ds.orientation_ned_wxyz)`.
`R_wb` is consumed by (a) `LinearKF.predict(accel_body, R_wb, dt)` for gravity-corrected propagation and (b) the localization lever `R_wc = R_wb @ R_camera_from_body().T` (so PnP world fixes depend on it too). In VQ2 `ds.orientation_ned_wxyz is None`.
**Fix:** run an AHRS each tick from `HIGHRES_IMU` (`ds.accel_body`, gyro, optional `ds.mag_body`) and source `R_wb` from the AHRS quaternion instead of the wire. **ESKF is the picked filter** (`ESKFAHRS`, `ahrs/eskf.py:130`; `step()` :191 → `q_wxyz` :168). Convert `q_wxyz` → rotation matrix for `R_wb`.
**Note on gyro:** `DroneState` currently has no raw gyro field exposed for AHRS (it has `angular_rate_body` from ODOMETRY, and `accel_body`/`mag_body` from HIGHRES_IMU). Confirm the gyro lands on the wire as part of HIGHRES_IMU and is surfaced into `DroneState` (see GAP #4) — the AHRS needs raw gyro.

### GAP #2 — attitude/rates into NavState (obs[6:12] feeder)
**Where:** `make_nav_state` `state_estimator.py:215-216` sets `NavState.roll/pitch/yaw` and `angular_rate_body` from `drone_state.*`. These feed obs[6:9] `rpy_g` and obs[9:12] `w_flu`. In VQ2 they are zero/`None`.
**Fix:** source Euler from the AHRS quaternion (same filter as GAP #1) and body-rates from the **raw gyro** (bias-corrected by the ESKF `gyro_bias` :173 if desired). This is the *same* attitude estimate as GAP #1 — wire it once, consume in both places.

### GAP #3 — AHRS instance ownership + clock
There is no place that constructs/steps an AHRS in the nav loop. Decide ownership: cleanest is the `Navigator` owns an `ESKFAHRS`, steps it on each IMU tick (the `dt>0` branch, `navigator.py:421-434`), and exposes `R_wb` + Euler + rates to both predict and `make_nav_state`. The AHRS `dt` must come from the **same master sim clock** (`ds.sim_time_ns` deltas) used by `LinearKF.predict` — do not mix in wall-clock. TIMESYNC reconciliation already exists for the vision epoch (`test_casec_foundation.py` P0-b); reuse the same clock discipline.

### GAP #4 — raw gyro (and mag) plumbing into DroneState
`contracts.py:81` documents `angular_rate_body` as *"from ODOMETRY"*. For VQ2 the AHRS needs **raw HIGHRES_IMU gyro**. Audit `mavlink_client.py:271,275` (and `elodin_adapter.py:101`) to confirm raw gyro from HIGHRES_IMU is parsed into a field the AHRS can read. If the only gyro path is the ODOMETRY-derived `angular_rate_body`, add/confirm a HIGHRES_IMU gyro field. Mag (`ds.mag_body`, `contracts.py:85`) is already plumbed and `None`-guarded; ESKF `mag_ned` update is optional.

### GAP #5 — case-C config defaults are OFF
`NavigatorConfig` ships `use_given_position=True` (`navigator.py:195`), `use_rewind_kf=False` (:253), `use_gate_relative=False` (:260), `use_range_channel=False` (:282). A case-C deploy profile must flip: `use_given_position=False`, `use_vision=True`, `use_gate_relative=True`, `use_rewind_kf=True`, `use_range_channel=True` (range channel is attitude-independent — valuable when AHRS is cold), and set a new `use_ahrs=True` (the flag GAP #1/#2 introduce). **No such named deploy profile exists yet** — it must be created (gated, OFF by default).

### GAP #6 — AHRS initialization / convergence transient
ESKF needs a startup alignment (accel-gravity levelling for roll/pitch; mag or motion for yaw). On a sim epoch reset (`navigator.py:407`) the AHRS must re-seed alongside the KF. No reset hook exists. Spec: re-`initialize` the AHRS in `Navigator.reset()`/`_initialize`. Until the AHRS converges, attitude is unreliable → the KF predict injects garbage. Mitigation: hold/inflate during the alignment window, and lean on the **attitude-independent range channel** (`gate_range_fix`) for early fixes.

### NON-GAPS (already closed — do not re-build)
- obs[0:20] emulator fidelity (`#37`): pinned ≤1e-6 / 2.93e-8 (`test_deploy_obs20.py`, `test_spike_golden.py`).
- +L sign: pinned (`test_obs_sign_faithfulness.py`).
- /√2 sigma reconciliation: intentional, pinned once at `estimator_obs.py:163` (`test_sysid_production_obs.py` R3).
- obs[0:17] byte-identity vs inc7: 0.0 delta.
- RewindKF OOSM, gate-relative anisotropic cov, range channel, in-plane state floor: all built + tested.

---

## 3. MINIMAL OFFLINE HARNESS (no Adroit/SLURM, no sim, no training)

**Goal:** prove the *new* case-C attitude seam closes faithfully — i.e. the self-localized obs built from AHRS-estimated attitude matches the obs built from ground-truth attitude — extending the existing `#37` check to cover L1.

### Inputs (synthetic preferred; recorded optional)
Generate a short trajectory (the AHRS test infra already exists: `ahrs/traj6dof.py`, `ahrs/imu_gen.py`):
- **Ground truth:** per-tick `R_wb_true(t)`, `pos_true(t)`, `vel_true(t)`, gate poses (use a synthetic gate map; the VQ1 track is allowed **only** as a throwaway geometry fixture, never as a performance target).
- **HIGHRES_IMU stream:** synthesize `accel_body`, `gyro`, `mag_body` from the trajectory via `imu_gen` (add realistic noise/bias).
- **JPEG/Frame stream OR synthetic GatePose:** either render frames, or (cheaper, recommended for unit-level) inject exact `GatePose` from the known geometry through the localization path, bypassing the detector (the golden-tuple already does the detector half).

### The loop under test
Drive **two Navigators** over the identical trajectory:
- **Truth-Nav (oracle):** attitude = `R_wb_true` (the current ODOMETRY-equivalent path).
- **Case-C-Nav (under test):** attitude = `ESKFAHRS` from the synthetic IMU; `use_given_position=False`.
Both produce `NavState` → `estimator_obs20(...)` each tick.

### Comparison metric + pass criteria
1. **Attitude error (L1 isolated):** `angle(R_wb_ahrs, R_wb_true)` after convergence. **Pass:** p90 ≤ a budget tied to `ATTITUDE_NOISE_STD_RAD` (`frames.py`; the KF already inflates Q by this — keep the AHRS error at-or-under that assumption). Suggest p90 ≤ ~1.0° steady-state (tighten with data).
2. **obs delta (end-to-end fidelity, the #37 extension):** per-channel `|obs_casec - obs_truth|` over the post-convergence window.
   - obs[0:6] (pos_g/vel_g): **pass** if within the localization noise floor (these are dominated by the fix, not attitude — should be small).
   - obs[6:9] (rpy_g): **pass** if ≤ the attitude-error budget above (this is the channel the gap directly perturbs).
   - obs[9:12] (w_flu): **pass** if ≤ gyro-bias-residual budget.
   - obs[17:20] (confidence/age): **pass** if the channels stay in [0,1] and track fix density (no NaN/blowup).
3. **Loop liveness:** `policy_step` returns finite, in-bounds rate+thrust for every tick (reuse the golden-tuple action smoke).
4. **OFF-path regression:** with the new `use_ahrs=False`, the whole loop is **byte-identical** to today (run the existing `test_deploy_obs20.py` / `test_spike_golden.py` unchanged → still green).

### Harness shape
A torch-free pytest module `tests/test_casec_ahrs_loop.py` (mirrors `test_casec_foundation.py` style) + an optional driver script `scripts/casec_ahrs_probe.py` for sweeps/plots. No GPU needed for the obs/attitude assertions; the action smoke skips gracefully if the inc8 `.pth` is absent (as the golden-tuple already does). Reuse `ahrs/traj6dof.py` + `ahrs/imu_gen.py` for inputs and `ahrs/metrics.py` for attitude error.

---

## 4. SEQUENCED BUILD PLAN

Each step states files, the gated-OFF strategy, and the test. **[P]** = parallelizable / independent; **[D:n]** = depends on step n.

1. **[P] Gyro/mag plumbing audit & fix (GAP #4).**
   Files: `src/racer/contracts.py`, `src/racer/mavlink_client.py`, `src/racer/elodin_adapter.py`.
   Confirm/add a raw HIGHRES_IMU gyro field on `DroneState` distinct from the ODOMETRY `angular_rate_body`. OFF-strategy: additive field, default `None`; existing fields untouched → byte-identical.
   Test: a contracts/parse unit test that the new field is populated from HIGHRES_IMU and old paths unchanged.

2. **[P] AHRS adapter shim (GAP #1/#2 prep).**
   Files: new `src/racer/ahrs_adapter.py` (or a small helper in `ahrs/__init__.py`).
   Wrap `ESKFAHRS` behind a tiny interface `step(accel,gyro,mag,dt) -> (R_wb, euler_rpy, w_body)` so the navigator depends on one seam, not the filter internals. Pure, torch-free.
   Test: unit-test the adapter against `ahrs/traj6dof` ground truth (attitude error budget from §3).

3. **[D:1,2] Wire AHRS into Navigator behind `use_ahrs` flag (GAP #1, #3, #6).**
   Files: `src/racer/navigator.py` (own an `ESKFAHRS`, step it in the `dt>0` branch :421-434, re-seed in `reset()`/`_initialize`), `NavigatorConfig` (add `use_ahrs: bool = False`).
   OFF-strategy: when `use_ahrs=False`, keep `R_wb = R_world_from_odo_quat_wxyz(...)` exactly as today → byte-identical. When `True`, source `R_wb` from the adapter.
   Test: with flag OFF, all existing navigator/obs tests stay green (regression); with ON, the §3 attitude-error assertion on a synthetic run.

4. **[D:3] Route AHRS attitude into NavState (GAP #2).**
   Files: `src/racer/state_estimator.py` `make_nav_state` (accept optional euler/rates overrides), `src/racer/navigator.py` `_nav_state` (pass AHRS euler/rates when `use_ahrs`).
   OFF-strategy: default args = today's `drone_state.*` → byte-identical when not overridden.
   Test: obs[6:12] now tracks AHRS attitude; OFF path byte-identical (`test_deploy_obs20.py` unchanged).

5. **[D:3,4] Case-C deploy profile (GAP #5).**
   Files: wherever fly/deploy constructs `NavigatorConfig` (`rl/fly_rl.py` / a new `scripts/fly_casec.py` or a config preset).
   Flip: `use_ahrs=True, use_given_position=False, use_vision=True, use_gate_relative=True, use_rewind_kf=True, use_range_channel=True`. Keep the default config untouched (VQ1/case-A path stays).
   Test: smoke that the profile constructs and runs end-to-end on synthetic input (no given position) and emits finite actions.

6. **[D:4,5] The offline fidelity harness (§3).**
   Files: new `tests/test_casec_ahrs_loop.py` (+ optional `scripts/casec_ahrs_probe.py`).
   This is the acceptance gate for the whole effort: dual-Navigator (truth vs case-C), the obs-delta metric, the OFF-byte-identity regression.

**Parallelism:** Steps **1 and 2 are independent** and can run on two agents at once. Steps **3→4→5→6 are a dependent chain** (each needs the prior seam). Net: one parallel pair up front, then a 4-step serial spine. Total **6 steps**, **2 parallel**, **4 serial**.

---

## 5. RISKS / INVARIANTS TO PRESERVE

- **+L obs sign** — `pos_g = R_w2g @ (gate - pos)`. Pinned by `tests/test_obs_sign_faithfulness.py` (≤1e-5 / -L control breaks 24 m). The localization lever `gate.position_ned - R_wc@t_cam_gate` is the same convention. Do not let an attitude-frame change flip it. **Caution:** `R_wc = R_wb @ R_camera_from_body().T` now depends on AHRS `R_wb` — an AHRS frame-convention error (FRD↔FLU, NED↔Z-up, body↔world) will silently corrupt every fix.
- **OFF == byte-identical discipline** — every new flag (`use_ahrs`, the deploy profile) defaults OFF and must leave the VQ1/case-A and existing case-C tests bit-for-bit unchanged. Run `scripts/green_gate.py` (sentinel 1091) after each step.
- **Confidence/age channels [17:20]** — `SIGMA_REF_M=0.05`, `TAU_STALE_S=0.10`, the **single** `/√2` reconciliation at `estimator_obs.py:163` (sum-sqrt NavState vs per-axis-RMS encoder). Do not double-apply or move it. Pinned by `test_sysid_production_obs.py` R3 + `test_spike_golden.py`.
- **Master clock** — AHRS `dt`, KF `dt`, and the vision epoch must all ride `ds.sim_time_ns` (TIMESYNC-reconciled), never wall-clock. Reuse the P0-b discipline (`test_casec_foundation.py`).
- **Frame conventions** — FRD body / NED world / Z-up+FLU obs frame / gate frame. ESKF outputs body(FRD)→world(NED) `q_wxyz`; obs builder converts to Z-up/FLU in `obs_from_zup`. Keep the AHRS on the same FRD/NED contract as `R_world_from_odo_quat_wxyz`.
- **AHRS convergence transient** — garbage attitude during alignment poisons the KF predict and PnP rotation. Hold/inflate until converged; prefer the attitude-independent range channel for cold-start fixes; re-seed AHRS on sim epoch reset.
- **Gyro provenance** — the AHRS must consume **raw HIGHRES_IMU gyro**, not the ODOMETRY-derived `angular_rate_body` (which is blocked in VQ2). A silent fallback to the blocked field would pass tests on the wire-fed sim but fail on the scored wire.

---

## Appendix — key file:line index

| Thing | Location |
|---|---|
| ESKF (picked) | `src/racer/ahrs/eskf.py:130` ctor, `:191` step, `:168` q_wxyz, `:173` gyro_bias, `:178` unc |
| Other AHRS (unwired) | iekf `LeftInvariantEKF:139`; eqvio `SE23RightInvariantEKF:154`, `EqVIOJointEKF:552`; classical `Madgwick:63`,`Mahony:159` |
| Attitude read (GAP #1) | `src/racer/navigator.py:416` |
| Attitude→NavState (GAP #2) | `src/racer/state_estimator.py:215-216` |
| ODOMETRY quat helper | `src/racer/frames.py:167` |
| LinearKF predict | `src/racer/state_estimator.py:110` |
| make_nav_state | `src/racer/state_estimator.py:196` |
| Navigator.update | `src/racer/navigator.py:399` |
| NavigatorConfig flags | `navigator.py:195` given_pos, `:253` rewind_kf, `:260` gate_relative, `:272` inplane_floor, `:282` range_channel |
| Detector | `src/racer/vision/detector.py:171` GateDetector.detect, `:223` Ensemble |
| Localization +L | `src/racer/localization.py:91` abs, `:161` gate-relative, `:230/:275` range |
| obs builder | `src/racer/estimator_obs.py:171` obs20, `:129` confidence_triple, `:194` auto-dispatch; `:91`/`:92` constants; `:163` /√2 |
| obs[0:17] core | `rl/fly_rl.py:307` obs_from_zup, `:360` build_obs |
| policy | `rl/fly_rl.py:479` load_actor, `:505` policy_step; inc8 env `rl/peregrine_racing_inc8.py:69` |
| #37 / golden tests | `tests/test_deploy_obs20.py`, `tests/test_spike_golden.py`, `tests/test_sysid_production_obs.py`, `tests/test_casec_foundation.py` |
| AHRS test infra | `src/racer/ahrs/traj6dof.py`, `ahrs/imu_gen.py`, `ahrs/metrics.py` |
| DroneState contract | `src/racer/contracts.py:59` (`:73-81` attitude fields, `:83-86` HIGHRES_IMU) |
| green gate | `scripts/green_gate.py` (sentinel 1091) |
