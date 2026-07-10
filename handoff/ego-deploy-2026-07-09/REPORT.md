# fly_ego deploy adapter — build report (2026-07-09)

Branch `claude/ego-deploy-2026-07-09` (worktree `Anduril-ego-deploy`, based on galileo HEAD
`c325585` incl. the A19 drain-to-empty receiver fix). Everything is **additive + flag-gated
(default OFF)**: without `--ego-ckpt` every existing path (inc7 RL loop, gate-seeker, VQ1) is
untouched — verified by the full test suite (below).

## What was built

| Commit | Content |
|---|---|
| `8be4803` | `src/racer/ego_obs.py` — the 21-dim egocentric obs builder (`EgoObsBuilder` + the pure `ego_actor_obs_np` assembly core + geometry helpers). |
| `adce97a` | `rl/fly_rl.py` seam — `load_ego_actor` (hardcoded bounds), `_fly_ego` loop, main() wiring, 7 new `--ego-*` flags. |
| `490503d` | `tests/test_ego_deploy_obs.py` + `tests/test_ego_deploy_actor.py` (26 tests). |

### Obs pipeline (per tick, inside `_fly_ego`)

1. `nav.update(s, frame)` — the unmodified case-C Navigator (vq2_case_c profile: AHRS,
   vp-yaw/floor-height decimated, map-free on the VQ2 wire). `nav.obs_drone_state(s)` yields the
   AHRS attitude in the ODOMETRY-wire convention; the loop re-applies `_ODO_QUAT_TRUE_CONJ`
   (exactly `build_obs`'s contract) to recover the TRUE `R_frd2ned`.
2. `seeker.detect_gate_lever(frame)` — the seeker's temporal-tracked, quality-gated PnP
   `GatePose` (detector shared + `detect_cached` frame-id-idempotent with the navigator; the
   seeker's *controller* is never called). Each `frame_id` is fed ONCE — a repeated pose is not
   a fresh fix.
3. `EgoObsBuilder.update(...)` -> the 21-dim obs:
   - rel_pos: `R_camera_from_body().T @ t_cam_gate + [0, 0, BORESIGHT.vert_offset_m]` (body
     FRD; the -0.25 m METRIC vert bake read LIVE from `frames.BORESIGHT`, the same dual-form
     correction as the localization +L lever) -> FLU (`[1,-1,-1]`) -> virtual flip
     (`diag(-1,-1,1)`).
   - velocity: `R_frd2ned.T @ nav_state.velocity_ned` (KF world velocity) -> FLU -> flip.
   - roll/pitch: training extraction (`pitch=atan2(-R20,hypot(R21,R22))`, `roll=atan2(R21,R22)`)
     on the *flipped* Z-up body->world matrix (obs_from_zup convention: flip THEN extract).
   - body rates: `DroneState.gyro_body` (wire-signed TRUE FRD; VQ2 `gyro_sign=(-1,-1,-1)`
     applied at the client) -> FLU -> flip.
   - obs[8]: the previous tick's RESCALED normed_thrust in g-units `[0, 3.765]` (policy_step's
     third return; 0.0 at flight start).
   - masking: `conf = clamp(1 - age/0.5, 0, 1)`; `det_proxy = age < 0.2 s`; slot masked to
     ZEROS past the hold (champion coast-OFF parity). Between fixes inside the hold the held
     rel_pos is ego-propagated (`exp(-[w]x dt) rel - v dt`, training estimator parity).
   - visible_area: 4 gate-model inner corners projected through the PnP pose -> image shoelace
     area / `fx*fy*L^2/r^2` (the exact training `gate_apparent_area` quantity; works for
     3-corner P3P fits too since the pose defines the quad); held between fixes, masked with
     the slot.
   - sector obs[9:11]: static per-gate bucket at first acquisition (below).
   - slot1 obs[16:21]: PINNED ZEROS (`n_gates=1` through the *training* window logic — the exact
     single-gate champion condition). `--ego-slot1` = rejected-at-startup stub.
   - Final assembly + NaN guard go through `ego_actor_obs_np`, pinned element-exact against the
     verbatim training reference.
4. `policy_step(actor, obs, args.max_rate, virtual_flip, yaw_scale)` — REUSED UNCHANGED
   (tanh -> rescale onto the hardcoded bounds -> Rz(pi) un-flip -> FLU->FRD `[1,-1,1]` ->
   collective = clip(normed*0.2656, 0, 1)) -> `ControlCommand(BODY_RATE)`.

### Safety / threading (harness-map compliance)

- Guards replicated from the RL loop: reset_counter change, race_start change,
  active_gate_index drop, >10 m teleport, hard collision, RACE_STATUS finished, sim-time stall.
- IMU/AHRS-liveness gate (skip tick on missing/degenerate attitude) — NOT the RL loop's
  ODOMETRY-staleness gate (ODOMETRY is blocked on this wire; `telemetry_health` would read
  no_fix forever).
- No sim-control command anywhere (section-7 DQ-safe); clean disarm owned by `fly_once`'s
  finally.
- No heavy compute added to the tick: detector shared + pre-warmed (`_prewarm_detector`
  extended to the ego path), detect frame-id-cached, `client._latest_frame` consumed
  non-blocking, no second receiver, per-tick forensics buffered in memory and written ONCE at
  exit (`<session>/ego_obs.jsonl`).
- On a RACE_STATUS gate advance: `seeker.reset()` (drop the temporal track -> re-acquire the
  new gate) + builder slot reset (the new slot0 starts masked — window-promotion analog).

### Action bounds (hardcoded, never sidecar)

`load_ego_actor` asserts obs width == 21 and mutates `_ACT_MIN/_ACT_MAX` to thrust `[0, 3.765]`
(sbatch `dynamics.controller.max_normed_thrust=3.765` override) / rates `+-3.14`. The ego
launcher writes no sidecar; the generic loader's legacy `[0,5]` fallback (33% thrust overdrive
+ corrupted obs[8]) is unreachable on this path. `cmd_rate_scale` is FORCED to
`--ego-rate-scale` (default **1.0**, loudly logged; the vq2_case_c profile's 0.4 belongs to the
classical pursuit gains — the RL plant was sysid'd at wire scale 1.0). The profile's
`gyro_sign=(-1,-1,-1)` IS applied (wire convention the AHRS needs).

## Deliberate train/deploy divergences (ranked by expected impact)

1. **Velocity source quality** (largest). Training: IMU-primary body velocity with a small
   per-episode bias drift, partially corrected toward truth at every accepted fix
   (`vel_correct_gain=0.15`). Deploy (map-free VQ2 wire): `NavState.velocity_ned` = the KF's
   pure IMU dead-reckoned velocity — **no position fixes ever correct it** (no TRACK_INFO map ->
   the gate-relative KF fixes never fire). Attitude error of ~1 deg => ~0.17 m/s^2 phantom
   accel => several m/s of drift over a lap. Training's drift was bounded; deploy's is not.
   This is the #1 candidate if the policy flies with a systematic speed/lean error.
   (Mitigation candidates, NOT built: EMA of finite-differenced gate-range along the track;
   a builder-internal pseudo-fix velocity clamp.)
2. **det_proxy vs geometric det** (by design, per directive). Training recomputes geometric
   detectability every 33 ms; deploy holds `age < 0.2 s` (`--ego-det-hold`). Consequences:
   (a) the blackout cliff fires up to ~0.2 s LATER than the geometric exit; (b) brief geometric
   occlusions shorter than the detection cadence never mask (training might have). At slow-lap
   speed 0.2 s ~ 0.4-0.6 m of travel — the crossing endgame is still flown on zeros, slightly
   delayed.
3. **No K=1/N_eff smoothing** — a fresh fix SNAPS the held rel_pos (K=1). Training smoothed
   toward fixes with K ~ 1/6.5. Rationale: the deployed detector+PnP chain already carries the
   seeker's temporal-track continuity gating, fixes arrive 2-4x slower than training's, and the
   likely first checkpoint (vczext) is noise-0-trained, where smoothing is a no-op. Divergence:
   deploy rel_pos is noisier per-fix than the smoothed training channel would be at
   noise_scale=1.
4. **Confidence duty cycle**. At 7-15 Hz detections, conf oscillates ~0.7-1.0 between fixes;
   vczext (noise-0) saw conf == 1.0 while detected (miss_prob*0=0, fix every 33 ms). Mild OOD
   on obs[14] for the noise-0 champion; the noise-matched vn16 saw conf dips from misses.
5. **Coarse sector analog** (obs[9:11]). Training: static per-gate bucket from COURSE geometry
   (incoming->outgoing leg turn / outgoing elevation). No course map on the wire, so:
   horiz == 0 (exactly the value a single-gate course produces — turn=0 — i.e. the ONLY value
   the champion ever saw); vert = leveled elevation bucket of drone->gate at FIRST acquisition
   (the incoming-leg stand-in; at acquisition, right after the previous pass, drone->gate ~ the
   leg), then held static per gate. Pre-first-fix: (0,0) neutral (training would have fed the
   true static bucket; on a flat leg that IS (0,0)). `--ego-sector-mode zero` pins (0,0).
6. **Fix timestamping**. A pose is stamped at the consuming tick's IMU sim-time, not the
   frame's capture time (camera/IMU epoch offset is unresolved on this wire) — the fed rel_pos
   is ~1-2 frames (30-100 ms) stale at acceptance and age slightly underestimates. Training had
   zero-latency fixes (FIX-B modeled latency only in covariance).
7. **Bias distribution** (inherited, not a builder choice). Training injected the [b,0,b]
   sign-random correlated bias; the deploy chain's real bias is one-signed with the vertical
   -0.25 m already baked out by the boresight lever the builder inherits. The training-time
   smear models an error deploy largely removes (relay-back item; pending VQ2 transfer check).
8. **Detector availability != geometric detectability**: score/reproj/continuity gates can drop
   frames training's 8-corner-quorum test would have kept (and vice versa close-in). Absorbed
   into the det_proxy hold; measure the real blackout duty from `ego_obs.jsonl`
   (`[ego-diag] masked-slot0 ticks`).

## Unresolved items

None blocking. No frame/sign question was left to guesswork — every transform reuses a pinned
constant (`_FLIP`/`_RZ_PI_BODY`/`_ODO_QUAT_TRUE_CONJ`/`R_camera_from_body`/`BORESIGHT`) already
validated on this wire, and the assembly core is pinned to the verbatim training reference.
(The cluster quad.yaml rate bound +-3.14 was taken as verified per the tasking note.)

## Test results (verbatim)

New tests — venv = MAIN checkout, `python -m pytest tests/test_ego_deploy_obs.py
tests/test_ego_deploy_actor.py -q`:

```
..........................                                               [100%]
26 passed in 2.00s
```

Directional sanity (SOFT, `-s`; vczext_final, gate 10 m ahead, +-2 m lateral, deploy flip
config):

```
[directional-sanity] vczext_final — gate 10 m ahead, 2 m lateral offset:
  gate LEFT : wire FRD rates=[roll +2.264, pitch -0.358, yaw -2.661] rad/s  collective=0.997  slot0(obs)=[-10.0, -2.0, 0.25]
  gate RIGHT: wire FRD rates=[roll +2.969, pitch -0.513, yaw -2.855] rad/s  collective=0.999  slot0(obs)=[-10.0, 2.0, 0.25]
  (human review: LEFT vs RIGHT should steer opposite ways in yaw and/or roll)
```

NOTE for the reviewer: the lateral response is roll-differential (+0.71 rad/s more roll for the
RIGHT gate), not a yaw sign flip — consistent with the known yaw-rail-dither behaviour of this
policy family (net-zero yaw, roll does the steering). Not asserted; flag if it looks wrong.

Existing fly_rl-adjacent suites (25 files: confirmed_cr*/p3/p4, deploy_obs20, estimator_obs
wiring, frame_conventions, gate_seeker, hold_last_demand, nav_estimate_log,
obs_sign_faithfulness, sysid_*, train_deploy_obs_elementwise, use_ahrs_wiring,
vq2_control_*):

```
228 passed, 35 skipped in 14.41s
```

FULL suite (`python -m pytest -q`):

```
FAILED tests/test_diagnose_session.py::test_real_bundles_diagnose_end_to_end
1 failed, 1543 passed, 73 skipped in 186.95s (0:03:06)
```

The single failure is PRE-EXISTING + ENVIRONMENTAL, unrelated to this change: the test skips
only when `handoff/shadowpc-postfix-dataset-2026-06-12/extracted` is absent; here the bundle
DIRS are in git but their `debug_obs.jsonl` payloads (gitignored recorded data) are not on this
machine, so `bundles == []` asserts. Nothing in this change touches diagnose_session or that
data path.

CLI fail-loud checks (manual): `--ego-ckpt` without `--seeker-weights` -> SystemExit
("gate-seeker/ego YOLO needs --seeker-weights ..."); `--ego-slot1` -> SystemExit (H6 stub);
`load_ego_actor` on a 17-wide dict -> SystemExit; vczext_final strict-load + `zeros(1,21)`
forward finite OK; sha256 of the local ckpt matches the remote
(`82959cb9...` = the 8000-upd final snapshot).

## Launch command (sim test, ShadowPC)

From the MAIN checkout (venv = main `.venv`; ckpt + detector weights are gitignored artifacts —
place `vczext_final_actor.pth` under `rl/checkpoints/`, detector via the artifact pipe):

```
.venv\Scripts\python.exe rl\fly_rl.py ^
  --ego-ckpt rl/checkpoints/vczext_final_actor.pth ^
  --seeker-weights "<gate_detector.pt>" ^
  --no-bridge --flights 1 --label ego_vczext_a1
```

Ritual: start fly_rl FIRST -> ">>> Waiting PASSIVELY..." banner -> THEN GO. Optional
first-flight training wheels: `--max-rate 1.5` (OOD cap). Defaults already correct:
`--ego-det-hold 0.2`, `--ego-stale-horizon 0.5`, coast OFF, sector auto,
`--ego-rate-scale 1.0`, `--virtual-flip` ON, 30 Hz.

Health lines to check at exit: `[loop-rate] ... OK` (not CHOKED), `[video-thread]` completed
climbing / gaps>1s = 0 (A19 criteria), `[ego-diag] fresh gate levers` climbing and
`masked-slot0 ticks` duty <~ 60% on approach, `<session>/ego_obs.jsonl` for per-tick obs
forensics.

## Open risks (ranked)

1. **KF velocity drift (map-free)** — divergence #1. If flights show runaway lean/speed, the
   velocity channel is the first suspect; instrument from ego_obs.jsonl (obs[0:3] vs expected
   slow-lap speeds).
2. **vczext = noise-0-trained** — the real vision noise/bias is OOD for it (known, accepted;
   the audit's late reversal made it the convergence champion). Prefer vn16 (noise-matched,
   job 3299283) when it lands: SAME adapter, zero changes — only the ckpt path.
3. **Detection availability duty cycle** — if the wire delivers levers << 7 Hz under load, the
   policy flies mostly-masked (blackout regime) far more than training did; `[ego-diag]` masks
   ratio is the readout. Knob: `--ego-det-hold` (raising it trades staleness for duty).
4. **cmd_rate_scale=1.0 assumption** — directive-fixed (plant sysid'd at wire scale 1.0). If
   the response looks 2.5x hot, `--ego-rate-scale 0.4` is the one-flag experiment.
5. **Sector analog** — horiz==0 is provably in-distribution; the vert bucket at first
   acquisition can differ from the true leg bucket if acquisition is late/off-line.
   `--ego-sector-mode zero` isolates it in one flag.
6. **Fix latency** (~1-2 frames unmodeled) — negligible at slow-lap speed, grows with the
   speed ramp; a rewind-style capture-time propagation is the future fix.
7. **20-gate course = 20 sequential single-gate episodes from the policy's view** — gate-to-
   gate handoffs (slot0 masked until the next gate is acquired) never occurred in single-gate
   training; between gates the policy sees the all-masked obs it only ever saw pre-acquisition.
   Watch the first inter-gate segment closely.
