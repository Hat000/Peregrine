# Internal System-ID — registration of every laptop-runnable estimator/obs/geometry pipeline

**Date:** 2026-06-18 · **Branch:** `p2-system-id` (off `main` @ dddf5d9) · **Env:** laptop arm — numpy 2.4.6 + torch 2.12.0+cpu + scipy 1.17.1, **diffaero ABSENT** (GPU/diffaero arms marked *deferred: Adroit*).

## Executive summary

Registered **9 diffaero-free pipeline pairs**, each driving both sides with identical input and isolating one layer. **Every pair resolved to a new pinning test — 0 code fixes required, 0 load-bearing sign/frame changes.** Every counterpart that has an independent reference agrees to floating-point tolerance; the three "divergences" that exist are **intentional and now pinned as such** (the √2 σ gap; the conditional-vs-unconditional RNG order; the absent production obs[17:20] builder).

**Flagship finding (#1, the live blocker).** A policy flies in the torch trainer but dies 0/200 in the numpy grader. Driving **both real obs emulators end-to-end** under shared injected draws and comparing the **final obs vector the policy consumes**: `obs[0:17]` max|Δ| = **9.5e-7**, `obs[17:20]` max|Δ| = **2.9e-8**, with **no condition-dependent blow-up** (range / fix-accept / staleness bins all ≤ 1e-6). The encoders are byte-faithful → the 0/200 death is **NOT a grader bug**. Verdict: **MEASURE-IN-TORCH, not FIX-THE-GRADER** — an independent, fresh obs-level confirmation of the standing POLICY-GAP diagnosis.

## Method (what makes a registration valid)

Drive both sides with **identical input** and isolate **one layer**. Dynamics → replay the same action sequence open-loop. Obs/estimators/geometry → same ground-truth trajectory + same DR seed + same injected noise. RNG: **bit-exact** per-step where the two can be fed shared draws; **distributional** (K seeds/side, per-channel mean/std) where the two draw in different order/shape. Divergence is reported **vs condition** (range-to-gate, fix-accept, staleness, tilt/rate), never as a bare scalar. **FIX** only when one side is demonstrably wrong vs a pinned test / canonical numpy frame / spec constant *and* the fix is local + inc7-byte-identical; **else document + pin current behavior**. Load-bearing signs/frames are never changed autonomously.

## Inventory & verdicts

| # | Pair | Counterpart (independent sides) | Drive | Worst divergence (vs condition) | Verdict | New pin |
|---|------|--------------------------------|-------|---------------------------------|---------|---------|
| 1 | obs-emulators | `EstimatorEmulator` (np) ↔ `BatchedEstimatorEmulator` (torch), full obs | mixed | obs[0:17] ≤ 9.5e-7, obs[17:20] ≤ 2.9e-8 (no range/fix/stale blow-up) | pin | `test_sysid_obs_emul_parity.py` |
| 2 | confidence-constants | SIGMA_REF/TAU_STALE + σ̂ formulas: np ↔ torch ↔ NavState | bit-exact | 0.0 (constants); √2 gap intentional | pin | `test_sysid_confidence_constants.py` |
| 3 | frames-camera | `frames.R_camera_from_body` (scipy) ↔ torch reimpl + K/geometry | bit-exact | ≤ 1e-12 (f64); ≤ 1e-2 (f32 deploy) | pin | `test_sysid_camera_geometry.py` |
| 4 | gate-frame-plusL | `_gate_rotmat_w2g`/`ned_gate_frame` np ↔ torch + **+L** | bit-exact | ≤ 1.1e-16; −L control breaks ≥ 10 m | pin (load-bearing) | `test_sysid_gate_frames.py` |
| 5 | action-sign-maps | `_FLIP` / `_ACT_FLU_TO_FRD` / `_RATE_SIGN_LIVE` / `_FLIP_FRD_FLU` | static | exact constants; yaw-only difference pinned | pin (load-bearing) | `test_sysid_action_sign_maps.py` |
| 6 | odometry-ry-pi | `frames` ODO R_y(π) conj ↔ independent `from_rotvec` R_y(π) | bit-exact | ≤ 1e-9; wrong-axis control resid > 0.5 | pin (load-bearing) | `test_sysid_odometry_quat.py` |
| 7 | plant-parity | `rl_plant` ↔ `twin` (numpy/numpy), aggressive regime | bit-exact | ω/thrust = 0.0; pos ≤ 1.0e-12, att ≤ 7.9e-15 at 179.6° tilt / |ω|=25 | pin | `test_sysid_plant_parity_aggressive.py` |
| 8 | localization-kf-navigator | `LinearKF` (np) ↔ `BatchedLinearKF` (torch) stress + cov convention | bit-exact | ≤ 1e-7 stress; cov convention 0.0; √2 gap pinned | pin | `test_sysid_kf_localization.py` |
| 9 | production-vs-trained-obs | deploy `estimator_obs` ↔ trained `build_obs` 20-dim contract | bit-exact | obs[0:17] = 0.0; obs[17:20] **builder absent (gap registered)** | pin | `test_sysid_production_obs.py` |

86 new tests across the 9 files; all GREEN locally (`.venv` python). The existing parity suite each pair leans on was re-run and confirmed GREEN (see per-pair sections).

## RNG-mode finding (registered, not a bug)

The numpy `EstimatorEmulator.step` draws per step in the order `standard_normal(3)` [IMU] → `random()` [Bernoulli] → **conditionally** `standard_normal(3)` [fix noise, only on accept]; the torch `BatchedEstimatorEmulator.step` takes `accept_u`/`accel_noise`/`fix_noise` **unconditionally** (env-injected, so the env owns the batched RNG stream). Feeding the two *real* `.step()` APIs one shared seed therefore diverges in the RNG stream the instant they disagree on an accept/reject. **Bit-exact registration is achievable only by injecting shared draws in numpy's unconditional order — which is exactly what proves the math/encoding is identical.** A paired-draw distributional run collapses all channel means to 1e-8…1e-11; the apparent independent-stream velocity-channel offset is pure MC variance of the cold `vel_std=5.0` prior (scales ~1/√K), not an encoding difference.

## Deferred: Adroit arm (diffaero/GPU-gated, cannot run on laptop)

These existing tests `importorskip`/skip without diffaero or a live stream; their laptop-runnable counterparts ARE registered above:
- `test_twin_diffaero_extreme_parity.py` (SKIP — needs diffaero; the numpy-vs-numpy `rl_plant`↔`twin` arm is registered in pair 7).
- `test_train_deploy_obs_elementwise.py`, `test_confirmed_p4_c05.py` (SKIP — diffaero env; the deploy↔trained 17-dim arm is registered in pair 9).
- `test_mavlink_velocity_single_rotation.py` (SKIP — needs a mavlink stream; the pure R_y(π) frame math is registered in pair 6).
- `test_confirmed_cr1_01 / cr2_01 / cr3_02 / p3_c06` (SKIP — diffaero-gated confirmation cases).

## Carry-forwards surfaced (NOT defects — for the commander)

- **Production obs[17:20] builder is absent** in `src/racer` (the `deploy_confidence_triple` lives only in `rl/spike_vertical_slice.py`, un-promoted). Pair 9 pins the *absence* so a future promotion trips a contract review. When promoted it MUST divide `NavState.nav_inplane_sigma` by √2 before the d5 `clip(σ_ref/σ̂)` encoding (else c_inplane is mis-reported by up to ~0.29).
- **The √2 σ gap is intentional and now pinned twice** (formula level + against the live torch emulator helper): NavState exports `sqrt(P_E+P_D)`; the emulator confidence uses `sqrt((P_E+P_D)/2)`. Do not reconcile.
- **RewindKF OOSM rewind has no torch counterpart** (deploy-only); registered self-consistently, no independent twin exists.

## MEMORY-DELTA (≤10 lines — commander is sole memory writer)

```
- SYSTEM-ID DONE (p2-system-id, 2026-06-18): 9 diffaero-free pipeline pairs registered, 86 new tests/test_sysid_*.py, green_gate GREEN. 0 fixes, 0 load-bearing changes.
- #1 LIVE-BLOCKER RESOLVED at the obs layer: numpy EstimatorEmulator vs torch BatchedEstimatorEmulator emit BYTE-FAITHFUL final obs (obs[0:17]<=9.5e-7, obs[17:20]<=2.9e-8, no range/fix/stale blow-up) => 0/200 grader death is a POLICY-GAP, answer = MEASURE-IN-TORCH (independent confirmation).
- RNG-ORDER registered: numpy .step draws fix-noise CONDITIONALLY (on accept); torch .step draws UNCONDITIONALLY (env-injected). Bit-exact only via injected shared draws; distributionally identical (paired-draw means 1e-8..1e-11).
- √2 σ gap PINNED (NavState sqrt(P_E+P_D) = √2 × emul sqrt((P_E+P_D)/2)) at formula level + vs live emul helper; intentional, do NOT reconcile.
- CARRY-FORWARD: production obs[17:20] builder ABSENT in src/racer (deploy_confidence_triple un-promoted); pair-9 pins the absence; on promotion divide NavState.nav_inplane_sigma by √2 before d5 clip.
- Load-bearing conventions PINNED unchanged: +L (gate-frames), _FLIP=[1,-1,-1]/_ACT_FLU_TO_FRD=[1,-1,1]/_RATE_SIGN_LIVE=[1,1,1] (yaw-only difference), ODO R_y(π) conj (vs independent from_rotvec, wrong-axis negative control).
- Plant rl_plant↔twin bit-identical (ω/thrust=0.0) even at 179.6° tilt / |ω|=25 / collective 1.4; LinearKF np↔torch <=1e-7 under near-singular S / large-cov / tiny-dt stress.
- DEFERRED to Adroit: test_twin_diffaero_extreme_parity, test_train_deploy_obs_elementwise, test_confirmed_p4_c05/cr*, test_mavlink_velocity_single_rotation (diffaero/stream-gated; laptop counterparts registered).
```

---

# Per-pair registration detail



## obs-emulators — numpy `EstimatorEmulator` vs torch `BatchedEstimatorEmulator`

**Counterpart.** numpy `rl/estimator_emul.py::EstimatorEmulator` (`.obs` / `.confidence_channel`, driving the real `racer.state_estimator.LinearKF` + `rl/fix_surrogate.py::FixSurrogate`) vs torch `rl/inc8_estimator_emul.py::BatchedEstimatorEmulator` (`.step` / `.confidence_channel` / `kf_pos_zup` / `kf_vel_zup` / `obs_zup_torch`). Both emit the 20-dim policy obs: `[0:17]` gate-frame KF pos/vel + truth attitude/rates + lookahead, `[17:20]` the d5 confidence triple `[c_inplane, c_along, age_norm]`.

**Existing tests (run, GREEN).** `tests/test_inc8_estimator_emul_torch.py` (7 passed, 10.2 s) — pins frame-seam identity, **KF-state** (`kf.x`/`kf.P`/`t_since`) full-step parity, pointing→fix monotone, NEES band, obs[17:20] bounds. `tests/test_estimator_emul.py` (13 passed, 23.3 s). Neither had compared the **final obs vector** end-to-end — that residual gap is what this pair closes.

**Method (mixed).**
- **Bit-exact via shared INJECTED draws.** The two `.step()` APIs cannot be registered with one shared seed: numpy draws per step `standard_normal(3)`[IMU] → `random()`[Bernoulli] → **conditionally** `standard_normal(3)`[fix noise, only on accept], whereas torch takes `accept_u`/`accel_noise`/`fix_noise` **unconditionally**. The streams desync at the first accept/reject mismatch (numpy consumes 3 extra normals only on accept). So I built a numpy reference from the **real** primitives (`LinearKF.predict/update_position`, `FixSurrogate.p_accept/fix_sigma/fix_covariance`, `EstimatorEmulator.obs/.confidence_channel`) fed the injected draws in numpy's unconditional order, and stepped the **real** torch module with the **same** draws. This isolates the encoding/math layer.
- **Distributional.** K=64 seeds each side, independent internal-RNG `.step` paths, per-channel obs mean/std by range bin (the only honest mode for the two real APIs).

**Divergence vs CONDITION.** Bit-exact, 6 approaches (head-on / crab15 / crab30 / pitch10 / crab8+pitch6, gates 0–5, 486 steps): **GLOBAL `max|Δobs[0:17]| = 9.5e-07`, `max|Δobs[17:20]| = 2.9e-08`.** No condition-dependent blowup:

| condition | bin | max|Δobs[0:17]| | max|Δobs[17:20]| |
|---|---|---|---|
| range-to-gate | [12,18) m | 9.48e-7 | 2.94e-8 |
| range-to-gate | [18,24) m | 9.52e-7 | 2.86e-8 |
| range-to-gate | [24,30) m | 9.49e-7 | 1.46e-8 |
| fix accepted | True | 9.52e-7 | 2.94e-8 |
| fix accepted | False | 9.26e-7 | 2.76e-8 |
| staleness | age∈[0,.01) fresh | 9.52e-7 | 2.94e-8 |
| staleness | age∈[.999,1] stale | 8.95e-7 | 2.76e-8 |

The 9.5e-7 residual is uniform float32-class round-off, not a divergence. Distributional (independent streams, 18–24 m, n=960): triple per-bin mean agrees to `c_inplane` 9e-4, `c_along` 3e-4, `age_norm` 1e-3. Velocity channels showed a per-bin mean offset (`vel_along` 0.36 at K=64) that **scales ~1/√K** (0.17 at K=256) — the cold vel-prior (`vel_std=5.0`) heavy-tail MC variance of two *independent* streams. **Proven RNG-order, not encoding:** a paired shared-draw distributional run (K=96) collapses all channel means to 1e-8…1e-11 and global per-step `max|Δobs[0:20]| = 9.5e-7`.

**Verdict.** The two emulators are **FAITHFUL at the obs encoding level** (byte-identical to float32 round-off across every range / fix-accept / staleness condition, both obs[0:17] and obs[17:20]). Therefore the inc8 **0/200 numpy-grader death is a real POLICY-GAP, NOT an emul-fiction / fix-the-grader problem** — the answer is **MEASURE-IN-TORCH where the policy provably flies**, not patch the numpy grader. This is a non-load-bearing registration; pinned (don't change code) by a new test.

**Evidence.** `handoff/system-id-2026-06-18/scratch/obs-emulators/` — `drive_obs_parity.py` (PART-1 bit-exact + PART-2 distributional) + `drive_output.txt`, `distributional_convergence.py` (1/√K bootstrap-SE convergence), `paired_draw_confirm.py` (means collapse under paired draws). New pin: `tests/test_sysid_obs_emul_parity.py` (3 tests; slowest 6.5 s, file 16 s wall; bit-exact obs[0:17]≤1e-5 / obs[17:20]≤1e-6 with per-condition assertions + triple distributional agreement).



## confidence-constants

**Counterpart.** Three definers of the obs[17:20] confidence channel and its sigma-hat math:
- `rl/estimator_emul.py` (numpy escape-hatch reference / SPEC): `SIGMA_REF_M`, `TAU_STALE_S`, `EmulConfig`, `confidence_channel`, `_gate_frame_sigmas`.
- `rl/inc8_estimator_emul.py` (torch production twin): same names, batched.
- `src/racer` NavState sigma export: `navigator._gate_frame_pos_sigma` (`src/racer/navigator.py:613-628`) → `contracts.NavState.nav_inplane_sigma`/`nav_along_sigma` via `state_estimator.make_nav_state`.

**Deploy seam confirmed.** `src/racer/estimator_obs.py` does NOT append obs[17:20] — it is the inc7 17-dim deploy seam (module docstring + `estimator_state_for_obs` replace only pos/vel; "NO obs[17:20] confidence channel is appended here").

**Method.** Constants compared directly (`==`). For the formula pair, drove BOTH emulators with ONE synthetic gate-frame KF covariance — `P[:3,:3] = Rwg @ diag([P_E,P_D,P_along]) @ Rwgᵀ` with `Rwg = ned_gate_frame(π)`, target gate 2 — injected into numpy `kf.P` and torch `kf.P`, compared `_gate_frame_sigmas` + `confidence_channel`. No RNG in these definitions, so bit-exact single-input drive (no distributional fallback). Script: `handoff/system-id-2026-06-18/scratch/confidence-constants/measure.py`.

**Divergence vs condition** (P_E=0.0144, P_D=0.0400, P_along=0.2500):

| Pair | sigma_inplane_hat | sigma_along_hat | triple max |
|---|---|---|---|
| numpy vs torch | 0.00e+00 | 0.00e+00 | 0.00e+00 |

- Scalar constants: ZERO divergence. `SIGMA_REF_M = 0.05`, `TAU_STALE_S = 0.10` equal at module level AND as `EmulConfig` defaults; all 10 shared `EmulConfig` DR scalars equal.
- `navigator.INPLANE_POS_FLOOR_STD = 0.05` aliases `SIGMA_REF_M` (the σ_b state floor; comment in `localization.py:64` asserts the same).
- The numpy↔torch confidence formula is bit-identical: `sigma_inplane_hat = sqrt((P_E+P_D)/2) = 0.16492` both; `sigma_along_hat = sqrt(P_along) = 0.50000` both; triple `[0.30317, 0.10000, 1.0]` both.

**The INTENTIONAL sqrt(2) gap (registered, NOT reconciled).** NavState in-plane (`navigator.py:626`) exports the COMBINED-axis 1-sigma `sqrt(P_E + P_D) = 0.23324`; the emulator confidence channel (`estimator_emul.py:328`, `inc8_estimator_emul.py:549`) uses the per-axis RMS `sqrt((P_E + P_D)/2) = 0.16492`. Ratio `= 1.414213562373 == √2` to `<1e-12`. The along-track sigma is IDENTICAL in both (`sqrt(P_along)`, no gap). This is a documented design choice — the two channels serve different consumers (NavState wire export vs. PPO obs encoding); do not reconcile.

**Verdict.** new-pinning-test. All 5 existing tests that touch this pair are GREEN as run (`test_estimator_emul.py`, `test_inc8_estimator_emul_torch.py`, `test_inc8_fix_surrogate_torch.py`, `test_navigator_gate_relative.py`, `test_state_estimator_pfloor.py`), but none assert the cross-module constant equality, the numpy↔torch formula parity at the constant level, the `INPLANE_POS_FLOOR_STD == SIGMA_REF_M` alias, or the √2 gap as a named invariant. Added `tests/test_sysid_confidence_constants.py` (5 tests, ~5.7 s, GREEN — validated by running from `tests/`). inc7 (obs_dim 17) untouched — these are obs[17:20] / NavState-export definitions only.

**Evidence.** `measure.py` output: scalar equalities all True; numpy↔torch `|sig_ip|=|sig_al|=|triple|max = 0.00e+00`; `navstate_inplane/emul_inplane = 1.414213562373`, `|ratio − √2| < 1e-12`; along-track identical.



## frames-camera (#3a): camera geometry constants + projection

**Counterpart.** numpy scipy reference `src/racer/frames.py` (`R_camera_from_body`, `CAMERA_INTRINSICS_K`, `IMAGE_WIDTH/HEIGHT`, `project_camera_point`, `vertical_fov_deg`) consumed by `rl/fix_surrogate.py::geometry()` vs the self-contained torch reimpl `rl/inc8_estimator_emul.py` (`_r_camera_from_body_np`/`R_CAMERA_FROM_BODY_NP`, `CAMERA_INTRINSICS_K_NP`, `IMAGE_WIDTH/HEIGHT`) consumed by `batched_geometry()`. The torch side deliberately imports no scipy/frames so it loads on the Adroit GPU.

**Method = bit-exact.** Camera geometry carries no RNG (deterministic given drone pose + gate), so identical poses/gates were injected into both sides and every output channel diffed per-sample (4000 random NED poses, seed 12345; float64 and float32). Projection registered scalar `project_camera_point` against the torch in-image K-projection over front-of-camera samples.

**Existing tests (RUN, all GREEN).** `test_inc8_fix_surrogate_torch.py::{test_inc8_constants_match_numpy, test_geometry_parity, test_geometry_parity_float32_within_tol, test_p_accept_parity}`, `test_frames.py`, `test_projection.py` — 27 passed in 6.4s.

**Divergence vs condition (binned by range-to-gate).**
| dtype | range/lever | t_cam | az/el/bearing/view (12-28 m band) | worst angular | in_image (N=4000) |
|---|---|---|---|---|---|
| float64 | EXACTLY 0.0 | <=1.4e-14 | <=5.7e-14 deg | 5.4e-13 deg (az, >35 m only, out of band) | 0 disagreements |
| float32 (deploy) | range <=1.2e-5 m | <=1.3e-5 m | az <=5.4e-4 deg (worst 9-12 m bin), el <=2.3e-4 deg | 5.4e-4 deg | 0 disagreements |

Constants: `R_camera_from_body` max|diff| = 1.1e-16 (1-ulp scipy-vs-hand-matrix), `K` and image dims byte-identical. Projection: `project_camera_point` vs torch K-projection u,v <= 4.5e-13 px over 1034 front-of-camera samples. No channel exceeds the 1e-4 parity contract anywhere in the operating band; `in_image` (the acceptance-gating boolean) never flips at either dtype.

**Registered precondition (the one non-trivial finding).** The torch `R_CAMERA_FROM_BODY_NP` HARDCODES a pure +20deg mount and has no boresight concept, whereas numpy `R_camera_from_body()` composes the live `frames.BORESIGHT` angular fields. They agree today only because the deployed `BORESIGHT` is METRIC-only (`vert_offset_m=-0.25`; `pitch_rad=roll_rad=0`); the metric offset is a translation applied in the localization +L lever, not in the mount rotation. Verified: a hypothetical `pitch_rad=0.01` angular boresight desyncs the mounts by ~9.4e-3 (≈0.5deg), which would bias every emulated fix. Not a bug — intentional and currently safe — but it was unpinned.

**Verdict = new-pinning-test.** Added `tests/test_sysid_camera_geometry.py` (8 tests, GREEN in 6.7s; no global-state leak — the BORESIGHT mutation test restores in `finally`, confirmed by a combined 35-pass run). It pins: K/dims byte-identity, the pure-20deg/no-boresight torch matrix, the METRIC-only precondition (fires if a future angular boresight lands without updating the torch matrix) with a negative control, the FoV derivation, the float64 bit-close + float32 deploy-dtype geometry sweeps (with exact `in_image`), and projection parity. inc7 (obs_dim 17) is untouched — this pair is pure camera geometry, no obs assembly. No load-bearing sign/frame change proposed.



## gate-frame-plusL — Gate-frame rotations (numpy==torch) + the +L obs convention

**Counterpart.** numpy world->gate `_gate_rotmat_w2g` (`rl/fly_rl.py:185`, rows `[[c,s,0],[-s,c,0],[0,0,1]]`) vs torch `_gate_rotmat_w2g_torch` (`rl/inc8_estimator_emul.py:91`); numpy gate->world-NED `ned_gate_frame` (`rl/estimator_emul.py:85`, columns `[right,down,downrange]=[[s,0,c],[c,0,-s],[0,1,0]]`) vs torch `ned_gate_frame_torch` (`rl/inc8_estimator_emul.py:101`). The +L obs sign: `obs_from_zup` pos_g `= R_w2g @ (gate-pos)` (`rl/fly_rl.py:348`) and `src/racer/localization.py:86`.

**Method (bit-exact).** Both sides driven with the IDENTICAL yaw array — 519 yaws over `[-2π,2π]` plus `0, ±π, ±π/2` edges (`scratch/gate-frame-plusL/measure_gate_frames.py`). The helpers are pure (no internal RNG), so per-element `max|diff|` at float64 is the registration; no distributional fallback needed. **Laptop-runnable in full** — both torch helpers are self-contained `torch.cos/sin` builders (no diffaero/GPU), nothing deferred to Adroit.

**Divergence vs condition — ZERO.**
| pair | max\|diff\| (numpy vs torch, float64) |
|---|---|
| `_gate_rotmat_w2g` | **1.11e-16** (batched == per-element) |
| `ned_gate_frame` | **1.11e-16** |

Both are proper rotations across all yaws: `max|det-1|` = 2.22e-16; `ned` `max|RR^T-I|` = 2.22e-16. The cross-frame **C1 identity** (in-plane↔along-track decomposition) holds even when the displacement is built with the *torch* ned frame and pushed through the *numpy* Z-up obs frame: worst in-plane→along leak 1.78e-15, worst downrange→in-plane leak 1.78e-15. **+L convention:** obs `== R_w2g@(+L)` exactly (0.0e0 over 2000 random pairs); the `-L` negative control is a 2·\|L\| flip (worst miss 147 m on the batch, ~24 m at gate range). The torch env runs these in float32 (~1e-7) — still far inside the registration.

**Existing pins (RUN this session, GREEN, not skipped — torch present so `importorskip` passed):**
- `tests/test_obs_sign_faithfulness.py::test_obs_pos_g_is_plus_L_seen_and_minus_L_breaks` — GREEN (+L 4.77e-7 / −L break ~24 m)
- `tests/test_estimator_emul.py::test_ned_gate_frame_c1_consistency_identity` — GREEN
- `tests/test_estimator_emul.py::test_gate_construction_positions_match_course` — GREEN

No existing test pins **numpy==torch** for these two specific rotation helpers (grep for `_gate_rotmat_w2g_torch` / `ned_gate_frame_torch` in `tests/` → no matches).

**Verdict: new-pinning-test.** No code change — both helpers are bit-identical and +L is correct/load-bearing (do NOT touch the sign; `load_bearing_flag=true` only because the registration restates +L). New file `tests/test_sysid_gate_frames.py` (4 tests) pins numpy==torch for both rotations, the torch-side C1 identity, and the +L convention as a pure-numpy assertion (catches a −L regression even if torch is absent). Validated GREEN from the real `tests/` path (4 passed in 3.54s). inc7 (obs_dim 17) untouched: the test adds no code, only asserts existing behavior.

**Evidence:** `handoff/system-id-2026-06-18/scratch/gate-frame-plusL/measure_gate_frames.py` + `measure_output.txt`.



## action-sign-maps (#3c) — RL action / rate sign maps

**Counterpart.** The sign matrices in the RL action->wire / look-at-reconstruction pipeline, across three modules:
- `rl/fly_rl.py`: `_FLIP=[1,-1,-1]` (FLU<->FRD & Z-up<->NED frame flip, training adapter, L71); `_ACT_FLU_TO_FRD=[1,-1,1]` (policy_step **LIVE wire map**, L97, applied L544); `_RZ_PI_BODY=diag(-1,-1,1)` (virtual body flip, L151).
- `rl/offline_rollout.py`: `_RATE_SIGN_LIVE=[1,1,1]` (eval plant rate sign, L64; re-imported by `contact_true_eval` L47).
- `rl/inc8_reward.py`: `_FLIP_FRD_FLU=(1,-1,-1)` (inc8 reward body-rate FRD<->FLU flip, L176; used as `_LOOKAT_FLIP` in `contact_true_eval` L70).
- Bug call site: `rl/contact_true_eval.py` **L365/367** — invert then re-apply `_ACT_FLU_TO_FRD` to reconstruct FRD from the look-at FLU correction.

**Method.** `static-constant` — these are pure constant sign vectors, so registration is exact-value + algebraic relationship, driven by importing the LIVE objects and comparing (no RNG/dynamics; bit-exact and distributional don't apply). Laptop-only, no diffaero/GPU. Script: `handoff/system-id-2026-06-18/scratch/action-sign-maps/measure_action_sign_maps.py` (raw output `measure_output.txt`).

**Divergence vs condition.** None — counterparts agree exactly as designed (a document+pin pair). Registered facts:
- Values: `_FLIP=[1,-1,-1]`, `_ACT_FLU_TO_FRD=[1,-1,1]`, `_RZ_PI_BODY=diag(-1,-1,1)`, `_RATE_SIGN_LIVE=[1,1,1]`, `_FLIP_FRD_FLU=[1,-1,-1]`.
- `_FLIP` and `_ACT_FLU_TO_FRD` **AGREE on roll/pitch (axes 0,1)** and **DIFFER on YAW (axis 2: -1 vs +1)** — the registered yaw-injection bug: reconstructing the look-at FLU via `_FLIP` instead of `_ACT_FLU_TO_FRD` leaves roll/pitch bit-identical but sign-flips the yaw realized rate (FIXED ec4cb03; matched-state trace: pitch=0, yaw diverged ~0.06–0.10 rad/s).
- All three sign-vectors involutory (`v*m*m==v`); `_RZ_PI_BODY^2 == I`.
- `_FLIP == _FLIP_FRD_FLU` (inc8 reward flip = training adapter).
- Trained-semantics identity holds exactly: `_FLIP * trained_plant_rate_sign([1,1,-1]) == _ACT_FLU_TO_FRD` (the live wire map IS the realized FRD of the training closed loop, which is why deploying on the live plant `rate_sign=[1,1,1]` preserves trained semantics).
- Footgun registered, not fixed: the legacy CTBR `_rate_sign` alias and live `rate_sign=[1,1,1]` are self-consistent and intentional.

**Verdict.** `new-pinning-test`, **load-bearing**. Existing maps tests pass — `test_super_rate.py`, `test_mixer.py`, `test_controller.py`, `test_contact_true_eval.py`, `test_inc8_lookat.py`, `test_confirmed_cr4_03.py` all **GREEN**. But the source-constant invariants for these maps (cr1_01/cr2_01/cr3_02/p3_c06's I3) **SKIP** without ShadowPC live recordings (the clean-laptop default) — leaving the constant-level pin unguarded on a fresh checkout. New test `tests/test_sysid_action_sign_maps.py` (11 tests, 5.6s, no data/diffaero dep) imports the LIVE constants and pins every value + relationship above. inc7 obs_dim 17 untouched (no code change; pin-only).

**Evidence.** All 11 new tests GREEN run from `tests/`. Negative control: reintroducing the OLD bcc93f9 `_ACT_FLU_TO_FRD=[1,-1,-1]` (yaw=-1) fires both `test_flip_and_act_agree_rollpitch_differ_yaw` and `test_trained_semantics_identity` — confirming the pin is a genuine guard. No sign/frame change proposed (load-bearing; commander review only).



## odometry-ry-pi -- ODOMETRY R_y(pi) quaternion-conjugation convention (#3d)

**Counterpart.** Pipeline side: `src/racer/frames.py` -- `ODO_QUAT_TRUE_CONJ_WXYZ = [1,-1,1,-1]` (wxyz; negate x & z), applied by `true_attitude_from_odo_quat_wxyz` / `R_world_from_odo_quat_wxyz`; consumed by `scripts/frame_residual_report.py` (`report()` L76, `replay()` L150) and the vision/PnP path (`navigator.py`/`localization.py`). Independent counterpart: a canonical numpy `R_y(pi)` matrix built from `Rotation.from_rotvec([0, pi, 0])` (== `diag(-1, +1, -1)`), used to check the frame-pair-conjugation identity `R_true == Ry @ R_raw @ Ry` -- **not** a self-consistent round-trip.

**What it is / where applied.** The sim's ODOMETRY attitude quaternion is the FRD->NED attitude expressed in an `R_y(pi)`-conjugated frame pair (180 deg rotation of *both* world and body axes about Y). `frames.py` un-conjugates it via the elementwise wxyz scale `[1,-1,1,-1]`. The vision/PnP path and `frame_residual_report.py` apply the conjugation (use the TRUE attitude); the CTBR control path **intentionally does not** -- `mavlink_client.py:246-274` feeds the RAW quat to `euler_from_quat_wxyz` (a self-consistent alias, VQ1-proven, do-not-fix).

**Method (bit-exact).** Drive both sides with identical ground-truth attitudes: build a random TRUE attitude, conjugate to the raw ODOMETRY form, then reconstruct two ways -- (a) production `R_world_from_odo_quat_wxyz`, (b) an independently constructed numpy `R_y(pi)` matrix conjugation. Deterministic frame algebra (no wire RNG); pure numpy+scipy, laptop-runnable.

**Divergence vs condition.** None -- the identity holds for all attitudes (not condition-dependent). Over 20k random attitudes: `max|R_true - Ry@R_raw@Ry|` (incl. helper) = **2.498e-16**; Euler effect (roll->-roll, yaw->-yaw, pitch intact) err **0.0**; involutory err **0.0**; `Ry` matched `diag(-1,+1,-1)` exactly. The one place divergence could hide is a wrong-axis regression: the existing `test_frames.py` round-trip is **self-consistent** -- it recovers `R_true` even with the scale regressed to `R_x(pi) = [1,1,-1,-1]` (round-trip residual 0.0, mirror survives), exactly the footgun "internal consistency cannot catch a conjugation." The independent `R_y(pi)` reference rejects that mirror (**residual 1.08**).

**Existing tests (run GREEN):** `tests/test_frames.py::test_R_world_from_odo_quat_wxyz_gives_true_rotation_at_bank`, `...::test_R_world_from_odo_quat_wxyz_level_is_identity`; `tests/test_frame_conventions.py::test_deploy_constants_pinned`, `...::test_frames_helpers_involutory`, `...::test_force_projection_handedness`, `...::test_rate_channel_is_negated_true_rate`. `tests/test_mavlink_velocity_single_rotation.py` = **SKIP** (ShadowPC postfix+refit recordings absent on this checkout; that test guards the separate body-vs-world *velocity*-mix bug, a different convention).

**Verdict: new-pinning-test.** The convention is correct and already broadly pinned, but every existing check that exercises the conjugation matrix is either golden-data (force/rate handedness) or a self-consistent round-trip with a blind spot to wrong-axis mirrors. New `tests/test_sysid_odometry_quat.py` (5 tests, ~1.2 s, GREEN) closes the gap by pinning the conjugation against an **independent** `R_y(pi)` matrix, with a negative control proving the wrong-axis mirror is caught. inc7 (obs_dim 17) untouched -- pure frame math, no obs.

**Load-bearing flag: TRUE.** This is a load-bearing frame convention (one of the four documented convention bugs). The registration only *adds a test*; it proposes **no** change to the sign/scale. Any future change to `ODO_QUAT_TRUE_CONJ_WXYZ` must go through the commander.

**Evidence:** `handoff/system-id-2026-06-18/scratch/odometry-ry-pi/` -- `verify_ry_pi.py` (20k-attitude identity + single-sided negative control), `gap_check.py` (self-consistent round-trip vs independent-reference discriminating power), `test_sysid_odometry_quat.py` (the drafted pinning test).



## Pair #4 — plant-parity: `racer.rl_plant` vs `racer.twin.CtbrPlant`

**Counterpart.** `racer.rl_plant.step` (pure-numpy batch plant, `src/racer/rl_plant.py`) vs `racer.twin.CtbrPlant.step` (scipy-`Rotation` single-env twin, **ground truth**, `src/racer/twin.py`). Both are pure numpy/scipy — **neither arm is deferred**; verified that `diffaero` is absent from the venv yet both import and run. Comparison is against the twin's TRUE physical state (`pos/vel/q/omega/_thrust`), not `twin.state()` (which re-applies telemetry report-signs `rl_plant` deliberately omits).

**Existing tests (run, not assumed).**
- `tests/test_rl_plant_parity.py` — **GREEN** (125 passed, 51s). Covers 13 configs × 2 dt × 5 random sequences incl. `aggressive` (rates U(−8,8)) and `smooth_aggressive` (sinusoidal 6.0 rad/s); tolerances pos/vel 1e-9, omega/thrust exactly 0.0, att 1e-11. Also pins the quaternion helpers vs scipy.
- `tests/test_twin_diffaero_extreme_parity.py` — **SKIP** (5 skipped). Gated by a module-level `skipif` on the ShadowPC `postfix+refit` recordings (absent on this laptop). Note: its TEST 2 (extreme-state adapter backend parity) does NOT actually need recordings — `diffaero_dynamics` imports here via the `BaseDynamics=object` fallback and torch is present — but the broad module-level skip suppresses it too. Out of scope for this pair (that's the adapter/backend pair); flagging only as an observation.

**Drive method: BIT-EXACT.** Same deterministic action sequence replayed open-loop through both plants from a shared seeded state, isolating the integrator. No RNG runs inside `step` (DR off in these configs), so a true per-step bit-exact comparison is valid at the API.

**Divergence vs CONDITION** (45000-step aggressive battery: 5 configs `faithful/super_rate/measured_aero/aero_full/mixer_full` × 3 dt `0.02/0.01/0.0333` × 5 extreme sequences). Conditions actually reached: **tilt up to 179.6° (fully inverted)**, **|ω| pinned at exactly 25.000 rad/s** (the norm clamp, 7033 steps in the (24,25] bin; an isolated probe hit the clamp 174/200 steps), **collective up to 1.400** (past the knot-table end clamp).

| metric | GLOBAL max | by-tilt | by-rate | tolerance |
|---|---|---|---|---|
| pos   | 1.02e-12 m   | flat ~1e-12 across all bins | worst 1.02e-12 at the (24,25] clamp bin | 1e-10 |
| vel   | 1.30e-13 m/s | flat ~1.3e-13 | flat ~1.3e-13 | 1e-11 |
| omega | **0.0** (bit-identical) | **0.0 every bin** | **0.0 every bin** | 0.0 |
| att   | 7.9e-15 rad  | flat ~8e-15 | flat ~8e-15 | 1e-13 |
| thrust| **0.0** (bit-identical) | **0.0 every bin** | **0.0 every bin** | 0.0 |

`omega` and `thrust` are **bit-identical** in every tilt/rate/collective bin (the rate loop, norm clamp `_clip_to_norm` vs `_clip_norm`, slew limit, mixer, thrust lag, and knot end-clamp run identical arithmetic). `pos/vel/attitude` diverge only by float-rounding in the quaternion path (rl_plant's hand-rolled helpers vs scipy `Rotation`), **invariant to tilt/rate/collective** and ~12 orders of magnitude below physical relevance.

**Verdict: new-pinning-test (no fix needed).** The registration holds at machine precision through the full aggressive envelope; twin.py and rl_plant.py are the same model. The existing random battery covers aggressive randoms but does not *deterministically* pin the rate-rail / full-flip / past-1.0-collective corners — so I added `tests/test_sysid_plant_parity_aggressive.py` (38 cases, 9.6s) which replays deterministic extreme sequences and asserts omega/thrust bit-identity plus the tight pos/vel/att bounds, with two guard-the-guard tests proving the norm clamp is genuinely pinned (>50/250 steps) and the body actually inverts (>135°). Not load-bearing; no sign/frame change proposed. Measurement script + raw output: `handoff/system-id-2026-06-18/scratch/plant-parity/measure_aggressive.py`.



## Pair: localization-kf-navigator (#5) — LinearKF / RewindKF / Navigator / fix-cov

**Counterparts.** numpy `src/racer/state_estimator.py::LinearKF` (predict + update_position) ⇄ torch `rl/inc8_estimator_emul.py::BatchedLinearKF`; `src/racer/localization.py::gate_relative_inplane_fix` cov-shaping ⇄ `rl/fix_surrogate.py::FixSurrogate.fix_covariance` (+ its torch mirror inc8_estimator_emul.py:309); `src/racer/kf_rewind.py::RewindKF` OOSM (deploy-only, **no torch counterpart**) ⇄ analytic in-order full-re-propagation reference; `navigator.py::_gate_frame_pos_sigma` ⇄ `inc8_estimator_emul.py::_gate_frame_sigmas` (the sqrt(2) sigma gap).

**Method.** Bit-exact shared drive isolating one layer: same `(accel, R_wb, dt)` predict stream + same interleaved `(z, cov)` updates into both KFs; identical gate-frame variances into both cov-shapers; identical op-stream replayed for the OOSM check. Torch is on the laptop → nothing deferred to Adroit.

**Existing tests (all RUN GREEN):** test_localization (12), test_kf_rewind (7), test_navigator (17), test_navigator_gate_relative (7), test_gate_relative_fix (5), test_state_estimator (13), test_state_estimator_pfloor (10), test_inc8_linearkf_torch (4), test_casec_foundation (7). Also confirmed `fix_covariance` numpy⇄torch parity already pinned (test_inc8_fix_surrogate_torch.py:140, GREEN).

**Divergence vs condition.**
- **numpy LinearKF == torch BatchedLinearKF**: bit-exact-to-fp at every stress condition — near-singular S (meas cov_floor 1e-9, fix/step) `8.4e-11`; tiny dt 1e-6 s `1.4e-14`; rapid fix every step `~1.8e-13`; borderline dt 0.199 s `1.7e-13`. **Worst = large-covariance init** (pos/vel std 1e3) `2.2e-9` (prior amplifies the fp floor). Realistic regime (rand-R + surrogate-magnitude anisotropic cov, fix/step, 64×60) `6.7e-13`.
- **One adversarial NON-physical regime** (single-env + R=identity + isotropic ultra-tight cov var≤1e-2 + fix every step): the weakly-observable **velocity** channel accumulates `np.linalg.solve` vs `torch.linalg.solve` rounding to `dx≈1.6e-2` (var=1e-2) → `7.6e-2` (var=1e-3). Never reached in training/deploy (Bernoulli-gated fixes, random attitudes, surrogate σ ~0.1–0.85 m) — documented, not pinned as equality.
- **gate_relative_inplane_fix cov == surrogate fix_covariance** under identical gate-frame variances = **EXACTLY 0.0** over 200 random frames → same `R_wg @ diag @ R_wg.T` convention, axis order in-plane=cols 0,1 / along-track=col 2 (along-track lands on the gate normal).
- **RewindKF OOSM == in-order analytic reference = 0.0** (full re-propagation identity, 50 trials); G2 degeneracy `update_position_at(now)` == bare `update_position` = 0.0; too-old fix dropped, state untouched = 0.0.

**Intentional divergences (registered, NOT reconciled).**
- **In-plane position STATE floor**: numpy `LinearKF._apply_inplane_pos_floor` (default 0.0 = byte-identical no-op; deploy wires 0.05 via `NavigatorConfig.use_inplane_pos_floor`) has **no torch mirror**. When enabled and P over-converges, numpy raises the smallest horizontal eigenvalue (to `floor²=0.00250`) while torch stays at `1.4e-4` (max|ΔP_inplane| `4.1e-2`). Faithful because **both run floor=0 in training**; the floor is a deploy-only gain-keep feature.
- **sqrt(2) sigma gap** (KNOWN FOOTGUN): NavState `sqrt(P00+P11)` (SUM, BLUEPRINT §1.6) vs emulator `sqrt((P00+P11)/2)` (MEAN) → ratio **exactly √2**. Currently UN-WIRED into deploy obs (`estimator_obs.py` builds only the 17-dim inc7 obs; no production obs[17:20] builder yet) — matters only once `deploy_confidence_triple` is promoted.

**Verdict.** New pinning test `tests/test_sysid_kf_localization.py` (9 tests, ~9 s, torch-gated via `importorskip`): KF stress parity (5 conditions + case-C), gate-rel/surrogate cov axis-convention identity (+ explicit axis-order pin), and the √2 sigma-gap ratio (cross-checked against the live `BatchedEstimatorEmulator._gate_frame_sigmas`). No code change; inc7 (obs_dim 17) untouched. **No load-bearing sign/frame edits** — the +L lever and CTBR alias were not modified.



## production-vs-trained-obs (#6)

**Counterpart.** PRODUCTION/deploy: `src/racer/estimator_obs.py` -> `estimator_state_for_obs(ds, nav)` (pure pos/vel substitution of `NavState` into the wire `DroneState`) + `estimator_obs(...)` -> `rl/fly_rl.build_obs`. TRAINED: `rl/fly_rl.py` `build_obs`/`obs_from_zup` (17-dim) + the d5 confidence triple `obs[17:20]` produced in training by `rl/estimator_emul.EstimatorEmulator.confidence_channel` (deploy reproduction `deploy_confidence_triple` lives ONLY in the spike `rl/spike_vertical_slice.py`).

**Method (bit-exact).** Both sides driven with IDENTICAL state, isolating the obs layer. For inc7 the estimator (`NavState`) pos/vel are set EQUAL to the wire pos/vel (matched state) so the obs builder is the only variable. The `obs[17:20]` arm is a static existence/scan + closed-form sqrt(2) algebra. No diffaero/GPU needed (torch present, used by `build_obs`).

**Existing tests (run on laptop venv).** `tests/test_estimator_obs_wiring.py` 3/3 GREEN. `tests/test_train_deploy_obs_elementwise.py` (5) and `tests/test_confirmed_p4_c05.py` (10) all SKIP -- gated on ShadowPC audit recordings absent on this laptop (not failures).

**Divergence vs CONDITION.**
- **R1 (inc7 17-dim, matched state):** `max|deploy - trained| = 0.000e+00` over N=2000 random pose/vel/attitude/gate/thrust draws; 2000/2000 byte-identical. CONFIRMS the commander seed (`obs[0:17]` delta 0.0). inc7 (obs_dim 17) is byte-identical -- the deploy seam only swaps `position_ned`/`velocity_ned`, attitude/rates pass through.
- **R2 (the obs[17:20] GAP):** zero production `obs[17:20]`/confidence-triple builders exist in `src/racer`. The only `obs[17` references there are COMMENTS (`contracts.py:242` "FUTURE inc8 confidence channel"; `estimator_obs.py:14/67` "Does NOT append obs[17:20]"). `estimator_obs` public surface = `{estimator_obs, estimator_state_for_obs}`, both 17-dim. `deploy_confidence_triple` (spike-only) operates on a RAW `nav.kf.P[:3,:3]` (a Navigator internal) and needs the last-fix gate frame (`navigator._last_fix_gate_R`, NOT a `NavState` field) -- so it cannot be driven from the `NavState` contract today. **Registered carry-forward, not a defect** (matches memory: "promote `deploy_confidence_triple` -> `estimator_obs.py`; no production obs[17:20] builder").
- **R3 (sigma sqrt(2) footgun -- INTENTIONAL):** `NavState.nav_inplane_sigma` (`navigator.py:626` = `sqrt(P_E+P_D)`) / emul+spike `sigma_inplane_hat` (`estimator_emul.py:328` & `spike:89` = `sqrt((P_E+P_D)/2)`) = `1.414213562 +/- 1.7e-16` = exactly sqrt(2). WORST-CASE: feeding the NavState export RAW into the d5 encoding `clip(sigma_ref/sigma_hat)` mis-reports `c_inplane` (obs[17]) by up to **0.293**; dividing by sqrt(2) first recovers the trained value to 2.2e-16. The spike's reported ~9.2e-9 `obs[17:20]` delta is just the float32 cast of the triple vs a float64 reference -- `deploy_confidence_triple` reproduces the emul math to **0.0** given the same P/gate-frame/clock.

**Verdict: new-pinning-test (document + pin; NOT fix-needed).** The deploy 17-dim seam is correct and inc7-byte-identical; the missing 20-dim production builder is a contract DESIGN decision (gate-frame + sqrt(2) reconciliation) for the commander, and the sqrt(2) gap is a KNOWN-INTENTIONAL footgun (do not reconcile). New `tests/test_sysid_production_obs.py` pins all three (R1 byte-identity / R2 the absent-builder gap / R3 the sqrt(2) relation + its 0.29 deploy impact) -- 3 GREEN in 4.08s. Not load-bearing-flag: no sign/frame change proposed; the +L convention and CTBR alias are untouched.

**Evidence.** `handoff/system-id-2026-06-18/scratch/production-vs-trained-obs/measure.py` (raw run: R1 max 0.0 / 2000-2000; R2 only comment-hits; R3 ratio 1.414213562) + the spike-vs-emul confidence-math check (max 0.0).
