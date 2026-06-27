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
- ODO twist PAIRS with raw quat (unchanged; verified via c3b5a8e velocity-frame fix).
- 🚩 **CR1-01 SETTLED:** HIGHRES_IMU accel_body PAIRS with TRUE-CONJUGATED quat (NOT raw). Live direct test 281 banked ticks: residual TRUE 0.11 m/s² corr +1.00 vs raw 4.10E/7.54D m/s² corr −0.83. `navigator.py:313` (rotates raw accel_body by true-conjugated attitude) IS CORRECT AS-IS — NOT a case-C error source. SCOPE STRICTLY to HIGHRES_IMU-accel_body↔TRUE-conjugated; ODO-TWIST↔raw-quat UNCHANGED.
- 🚩 **VALIDATION: run `scripts/frame_residual_report.py` after EVERY live session** — internal consistency CANNOT catch conjugation.
- ODOMETRY crab ~55° = near-optimal posture (frame-audit confirmed 2026-06-12).

## ODOMETRY velocity-frame bug (FIXED)
- Bug: twist BODY-frame. Fix commit `c3b5a8e`: now rotated via `frames.world_vec_from_body_quat`.

## Sim build 1.0.3364 regression fix
- Fix: `launch_ramp_s=0.6 s` + `--rate 100`.

## Sim ops mechanics (unattended PROVEN — MASTERED 2026-06-13)
- **SIM-OPS MASTERED:** 10/10 consecutive clean autonomous launch→race→reset cycles; 13 clean inc7 flights; 0 contact. Orchestrator pattern: `partA_cycle.py` runs `fly_rl --flights 1` per cycle so every state transition is MAVLink-confirmed (not a blind between-flight reset). All 4 failure modes handled live (stale-started, zombie dual-instance, spawn-artefact detection, off-race-pause).
- **31000 restart:** Win32 foreground + Enter×2.
- Sim AUTORESETS on gate contact → guard cuts RL commands instantly.
- Use raw ODOMETRY rate.
- 🚩 **SPAWN-ARTEFACT:** gate-3 hard-collision reset → next respawn 0-tick crash; NOT policy failure; affects per-batch stats.
- **🚩 OPS:** concurrent laptop sessions must use separate git worktrees or stagger suite runs.
- 🚩 **ShadowPC tooling** (partA_cycle.py + recorder) in handoff/simops-mastery-2026-06-13/ on ShadowPC — **NOT YET on main (pending commit+push from ShadowPC).**

## Substrate audit (DONE 2026-06-13)
- VQ1 substrate **EXTERNALLY CLEAN 68/68** (no live-code train/deploy-corrupting bug found). COLL_MAP NOT A BUG.
- S18 VOIDED — do NOT use `--plant lapse`/`dr_lapse`.
- Gate-3 barrier root cause confirmed: point-mass L-inf training vs volumetric sim contact.
- CRAB-DIAG: ODOMETRY quat R_y(π)-conjugated; crab ~55° = near-optimal posture (frame-audit 2026-06-12).

## Autonomy-hardening (DONE; MERGED to main 6876f44; clean merge, no conflicts; full suite 687 passed/0 skips; pushed. `rl/submit_rl.py` now ON MAIN as authoritative submission entry — pins inc7, no MAV_CMD 31000 on judged path; auto-reset opt-in via --dev-auto-reset. telemetry_health()/odo_recv_ns freshness+finite gate, finally-disarm, late-join GO all live. Worktree Anduril-wt-flyrl + local branch removed; origin/flyrl-autonomy-hardening retained.)
- 🚩 **P4-C05 DEPLOY CONSTRAINT (f50b9b4):** current stack ships 3 loud guards: `_assert_vq1_constants_consistent` (import-time), `_assert_live_course_is_vq1` (deploy-time, LOUD ABORT before RL loop if any gate yaw ≠ π), `assert_gate_map_allpi` (public). **Fails loud on any non-π/VQ2 course until gate-relative pipeline is wired + `gate_map` passed.** This is intentional and correct; guards are removed by the gate-relative rebuild. → [[index-vision-estimator]] §P4-C05
- **F-A:** `rl/submit_rl.py` = new authoritative submission entry (pins inc7/--no-bridge/--no-auto-reset/--no-debug-obs/--flights 1). NO `MAV_CMD 31000` on judged wire path. `fly_rl` defaults flipped SAFE; auto-reset OPT-IN via `--dev-auto-reset`.
- **F-B:** `_fly_armed()` extracted; `fly_once` try/finally force-disarm; `main()` except Exception backstop.
- **F-C:** `DroneState.odo_recv_ns` + `telemetry_health()` gate above spin-guard/build_obs → SAFE HOVER on stale/non-finite; thresholds `--odo-stale-s 0.15` / `--odo-recovery-s 0.5`.
- **F-D:** passive late-join GO accept + bounded arm-retry (3 tries, force-arm last).
- 🚩 **DEV-RIG CHANGE:** ShadowPC dev batch now needs `--dev-auto-reset`.
- **§4 LIVE-VERIFY PASS 5/5 (2026-06-13):** (a) clean standing finish 9.91 s 6/6 exit 0; (b) ZERO MAV_CMD 31000 on wire (independent relay decode, 118 s); (c) late-join GO to_go=−34s→FINISHED; (d) odo/finite gates silent on healthy run; (e) crash→finally-disarm→exit 1, armed=False. CLEARED TO MERGE.

## ShadowPC-VISION-CAL results (2026-06-13; session COMPLETE)
- **In-loop vision latency L:** ~115 ms median CPU-only (detect 109 + transport 4 + PnP 1.6 ms). GPU eval host ≈ 15–25 ms. At 37 m/s: 115 ms = ~4.3 m ⇒ CONFIRMS RewindKF-as-default (covers CPU case).
- **CR1-01:** SETTLED (see TRUE PHYSICAL CONVENTIONS above).
- **Tooling built** (handoff/shadowpc-vision-cal-2026-06-13/): `simops.py`, `wire/mav_relay.py`, `cr1_direct.py`, `latency_harness.py`. `simops.py` promotion to `scripts/` DISPATCHED (LAPTOP-PROMOTE-SIMOPS).
- **Remaining queued items** (sigma_theta refit, per-gate last-fix distance, 2-corner PnP, Bayesian-IoU extrinsic, yaw-active debug_obs, full-lap ~37 m/s gate-4 recording): DEFERRED to post-merge phase.

## Sim-ops state machine (Fengyou-confirmed 2026-06-13; REFINED 2026-06-13 SHADOWVISION)
- HOME→Enter→waiting-room (started=False, drone at origin) → **Enter AGAIN → GO** (extra Enter out of waiting-room is the gotcha; **cold-launch waiting room takes TWO Enters**).
- **From a FINISHED race: ESC+Down×3+Enter RESTARTS the race (does NOT exit to HOME).** Fresh-race chain from finished = ESC+Down×3+Enter (back in waiting-room) + Enter×2 (to GO); confirmed by fresh to_go + new race_start_boot_ms.
- ESC+Down×3+Enter alone is context-dependent: restart from FINISHED (stays in race context), no-op from stale/HOME.
- 🚩 Stale started=True race STOPS streaming video — never trust started=True alone.
- 🚩 computer-use does NOT work for sim (request_access can't resolve DCGame-Win64-Shipping / "AI-GP") — telemetry-only Win32 is the standing method.
- Cross-cutting gotchas: epoch-vs-sim-boot clock; Windows SIO_UDP_CONNRESET; run_in_background+& double-launch.

## Ground-truth velocity (step-5 at-speed recording — CONFIRMED 2026-06-13)
- **GT velocity = `LOCAL_POSITION_NED.{vx,vy,vz}`** (world NED; ~97 Hz; live-verified to 21.3 m/s; stored as `DroneState.velocity_ned`). ODOMETRY twist (~75 Hz) = cross-check.
- Recorder captures full step-5 bundle recv-clock-aligned (video↔LPN 2.5 ms p50). `verify_bundle.py` ALL_PASS on banked at-speed + fresh standing session.
- frame_residual_report.py mirror canary TRUE +0.97 / AS-IS −0.81 = OK (telemetry convention stable across sim relaunch).
- Step-5 at-speed recording plan UNCHANGED. Tooling in handoff/simops-mastery-2026-06-13/ on ShadowPC — **pending commit+push from ShadowPC.**

## Topic file pointers
- [[project-ctbr-control-sysid]] — flyable stack: gains/signs, offline twin, Gate-0 saga, alt-relay.
- [[reference-sim-interface]] — confirmed MAVLink+video wire spec; all 5 must-verify items.
- [[reference-sim-ops]] — unattended FlightSim mechanics: launch/login, 31000 semantics, idle states, zombie instance, autoreset/spin guards, footguns, §AUTONOMY-READINESS.
- [[project-togt-time-optimal-bound]] — VQ1 time-optimal lap bound: frictionless ~4.3–4.7 s (thrust-bound), twin-tracked ~8.3 s realistic; reference line + TOGT pipeline + WSL/CRLF env landmines. (Headline bound is the rate-infeasible optimum — current doctrine ≈8 s upright; see [[index-rl-training]] §DOCTRINE REVISION.)
