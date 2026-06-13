# Control, Sim Interface & Ops Sub-Index
Mid-level index for sim wire facts, CTBR/legacy sign config, RL deployment recipe, true physical conventions, autonomy-hardening, and sim ops mechanics. Deep detail in topic files below.

## Sim interface wire facts (all 5 RESOLVED)
- pos+vel GIVEN (LPN 97 Hz + ODOMETRY 75 Hz).
- 🚩 **ATTITUDE.pitch sign-inverted → use ODOMETRY quat.**
- 6 gates, 2.72 m OUTER / ~1.5 m inner; course DESCENDS 26 m.
- Clocks INDEPENDENT; video dedup by frame_id; sim PAUSES off-race.

## CTBR/VQ1 LEGACY SIGN CONFIG (VQ1-proven — DO NOT "fix")
- `body_rate_sign=[1,1,−1]` / `odo_att_sign=[−1,1,1]` / `odo_rate_sign=[−1,−1,1]` / `ff_gain=2.5`
- Self-consistent alias, NOT true physical conventions.
- Arm = `MAV_CMD_COMPONENT_ARM_DISARM` p1=1.
- **Live config = `FAITHFUL_TUNED_GAINS`** (`kp_alt 3.0/kd_alt 1.75`, raw vz) + `−0.4 m alt_offset` + `launch_ramp_s=0.6 s`.
- Alt balloon is OURS. 🚩 **LIVE LATENCY = 2 ticks (67 ms)** (supersedes ~40 ms); offline twin under-models — do NOT trust offline "holds" claims.

## RL deployment recipe (93023cf; `rl/fly_rl.py`)
1. tanh(mean)→rescale (thrust [0,5], rates ±3.14).
2. Quat conjugated: `q_true=q_raw·[1,−1,1,−1]`.
3. Rate sign **[−1,−1,−1]**.
4. Wire cmd = `rate_flu·[+1,−1,+1]`.
5. Collective obs init 0.0.
6. 30 Hz.
7. Resets at rest 1 m before random gate.
- `obs[12]` = `prev_normed_thrust`.

## TRUE PHYSICAL CONVENTIONS (93023cf)
- ODOMETRY quat **R_y(π)-CONJUGATED**. True attitude: `q_true=q_raw·[1,−1,1,−1]`; true rate: `ω_true=−w_raw` (gain 0.999); cmd→rate sign [+1,+1,+1].
- ODO twist + accel_body PAIR with raw quat.
- 🚩 **VALIDATION: run `scripts/frame_residual_report.py` after EVERY live session** — internal consistency CANNOT catch conjugation.
- ODOMETRY crab ~55° = near-optimal posture (frame-audit confirmed 2026-06-12).

## ODOMETRY velocity-frame bug (FIXED)
- Bug: twist BODY-frame. Fix commit `c3b5a8e`: now rotated via `frames.world_vec_from_body_quat`.

## Sim build 1.0.3364 regression fix
- Fix: `launch_ramp_s=0.6 s` + `--rate 100`.

## Sim ops mechanics (unattended PROVEN)
- **31000 restart:** Win32 foreground + Enter×2.
- Sim AUTORESETS on gate contact → guard cuts RL commands instantly.
- Use raw ODOMETRY rate.
- 🚩 **SPAWN-ARTEFACT:** gate-3 hard-collision reset → next respawn 0-tick crash; NOT policy failure; affects per-batch stats.
- **🚩 OPS:** concurrent laptop sessions must use separate git worktrees or stagger suite runs.

## Substrate audit (DONE 2026-06-13)
- VQ1 substrate **EXTERNALLY CLEAN 68/68** (no live-code train/deploy-corrupting bug found). COLL_MAP NOT A BUG.
- S18 VOIDED — do NOT use `--plant lapse`/`dr_lapse`.
- Gate-3 barrier root cause confirmed: point-mass L-inf training vs volumetric sim contact.
- CRAB-DIAG: ODOMETRY quat R_y(π)-conjugated; crab ~55° = near-optimal posture (frame-audit 2026-06-12).

## Autonomy-hardening (DONE; commit 7210c1d; branch `flyrl-autonomy-hardening`; NOT merged — held for ShadowPC §4 live-verify)
- **F-A:** `rl/submit_rl.py` = new authoritative submission entry (pins inc7/--no-bridge/--no-auto-reset/--no-debug-obs/--flights 1). NO `MAV_CMD 31000` on judged wire path. `fly_rl` defaults flipped SAFE; auto-reset OPT-IN via `--dev-auto-reset`.
- **F-B:** `_fly_armed()` extracted; `fly_once` try/finally force-disarm; `main()` except Exception backstop.
- **F-C:** `DroneState.odo_recv_ns` + `telemetry_health()` gate above spin-guard/build_obs → SAFE HOVER on stale/non-finite; thresholds `--odo-stale-s 0.15` / `--odo-recovery-s 0.5`.
- **F-D:** passive late-join GO accept + bounded arm-retry (3 tries, force-arm last).
- 🚩 **DEV-RIG CHANGE:** ShadowPC dev batch now needs `--dev-auto-reset`.
- **MERGE GATE:** ShadowPC §4 checklist (clean finish + no 31000 on wire + late-join catches GO + odo/finite gates silent on healthy run + crash→disarm) must pass before merge.

## ShadowPC-VISION-CAL agenda (queued ⑩)
In-loop latency L + accel_body logging (CR1-01 rider) + hardening live-verify + sigma_theta refit + per-gate last-fix distance + 2-corner PnP + Bayesian-IoU extrinsic + yaw-active debug_obs capture + full-lap case-C sim (~37 m/s gate-4 recording).

## Topic file pointers
- [[project-ctbr-control-sysid]] — flyable stack: gains/signs, offline twin, Gate-0 saga, alt-relay.
- [[reference-sim-interface]] — confirmed MAVLink+video wire spec; all 5 must-verify items.
- [[reference-sim-ops]] — unattended FlightSim mechanics: launch/login, 31000 semantics, idle states, zombie instance, autoreset/spin guards, footguns, §AUTONOMY-READINESS.
