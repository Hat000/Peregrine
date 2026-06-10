# S1.2 — Stage-1 RL policy live validation (ShadowPC, 2026-06-10, fable session)

**Verdict: deployment pipeline VERIFIED end-to-end; checkpoint NOT transfer-ready (0 gates live).**
The blockers are four measured plant-fidelity gaps, all fixable at train time —
`rl/peregrine_racing_s13.sbatch` is the proposed Adroit dispatch. The deployment harness
(`rl/fly_rl.py`) is finished, fully autonomous (sim launch → GUI keys → reset → fly → record),
and ready for the next checkpoint with **zero changes**.

Checkpoint under test: `stage1_inc1_actor.pth` (MD5 f9e0a327…, job 3261393, success_rate 0.97).

---

## 1. What the sonnet debrief got wrong (PART 1 — verified against the actual DiffAero source)

| item | sonnet S1.1 deployment | training truth (env/base_env.py, network/agents.py, algo/PPO.py @ flyingbitac/diffaero main) |
|---|---|---|
| action squash | raw actor mean, clipped | `action = tanh(mean)` then `rescale_action`: linear [-1,1] → thrust [0,5], rates ±3.14 |
| FLU→FRD action sign | `[-1,+1,+1]` (wrong on ALL axes) | plain flip `[1,-1,-1]`; the PLANT applies rate_gain·rate_sign identically in training and live — no ff/rate_gain algebra |
| obs[12] collective | init 1.0, raw-clip feedback | init **0.0** (`last_action` zeroed at reset), feedback = **rescaled** normed_thrust |
| odo rate sign | `[+1,-1,+1]` | `[-1,-1,1]` (flight-proven; re-verified, §3) |
| control rate | 100 Hz | **30 Hz** (racing.yaml dt 0.0333, no sbatch override) |
| reset velocity | "racing velocity" | **at rest**, 1 m in front of a random gate, identity Z-up attitude |

Verified CORRECT in the sonnet version: obs layout (17), `R_W2G = diag(-1,-1,1)` (all gate yaws =
3.141592569), `next_relyaw = 0`, final-gate clamp, Euler-ZYX extraction (matches pytorch3d's
`_angle_from_tan` formulas exactly), actor architecture (NormedLinear = Linear→LN→ELU, [256,128]).
The sonnet "policy saturates at race start = root cause" analysis was made through the wrong action
transfer function; saturation at reset is the policy's NORMAL launch behavior (it saturates in
training too, then modulates).

## 2. The tail-first discovery + virtual flip (deployment key, keep it)

Training resets at identity Z-up attitude with all gates at yaw π ⇒ the policy learned to fly the
ENTIRE course **tail-first** (gate-frame yaw ≈ π throughout). The sim spawns the drone **nose-first**
(measured from recordings: NED pos (0,0,+0.02), true rpy (0°, −17.8°, −179.9°) — tilted pad) ⇒
gate-frame yaw ≈ 0 = 180° attitude-OOD.

Fix shipped in `fly_rl.py` (`--virtual-flip`, default ON): run the policy in a body frame rotated π
about body z — obs `R_b2w·Rz(π)`, body rates ·[-1,-1,1], action rates ·[-1,-1,1]. Exact rigid-body
symmetry (plant rate_gain x/y differ 0.1%). Offline: handoff-state rollouts go 0/6 → 4-6/6 with the
flip alone.

## 3. odo_rate_sign reconciled (PART 2 — settled offline, no test flight needed)

Correlating raw ODOMETRY `angular_rate` against quat-finite-difference rates
(`body_rate_from_quats`) on a course recording: corr = [+0.998, −0.998, +0.999], slope ≈ ±1.0.
The quat-derived rate is the derivative of the REPORTED attitude (roll inverted), so:
- raw vs reported-attitude derivative = `[+1,-1,+1]` ← what the old controller.py sysid measured
- raw vs TRUE rate = `[-1,-1,+1]` ← the flight-proven set; what the obs needs

Both memory values were correct against their own references. Bonus finding: the reported quat +
reported body twist reproduce LOCAL_POSITION_NED world velocity to **0.00 m/s** even at high roll —
the sim's reporting frame is internally self-consistent; only the physical roll response is
inverted. `state.velocity_ned` is pristine; no client change needed.

## 4. Offline rollout harness (`rl/offline_rollout.py`) — the session's main tool

Policy + `racer.rl_plant` (parity-tested twin of the training dynamics) closed-loop on the real
course; `build_obs` telemetry path vs truth path asserted equal to 0.0. Results matrix (live
thrust clip = collective ≤ 1.0 as the real sim enforces):

| start state | flip | live clip | outcome |
|---|---|---|---|
| training reset (1 m, rest) | n/a | no | **6/6 FINISH 3.3 s** (validates ALL deployment math) |
| real spawn (23.3 m, rest) | no | either | 0/6 — veers, clips gate 0 (attitude-OOD) |
| real spawn | yes | yes | 1/6 best (rate-cap variants 0–1/6; arrives at gate 0 at 33–44 m/s, unrecoverable) |
| handoff 3–5 m @ 4–12 m/s | yes | no | **6/6 FINISH** |
| handoff 3–5 m @ 4–12 m/s | yes | yes | **4/6** — gate-4 wall (thrust ceiling, §6.2) |

## 5. The live flight (PART 4, flight 1 of 1 — stopped per protocol)

Config: CTBR bridge (canonical-6/6 faithful stack) → handoff <3 m & >4 m/s → policy @30 Hz,
virtual flip ON. Recording `data/runs/20260610_205414_rl_s12_f1`.

- Bridge: TAKEOFF → RUN → **handoff at +2.98 m, 5.1 m/s, dead-centre (dy +0.04, dz +0.05 m)**.
- Policy takeover: first action = exactly the offline prediction ([+3.14,−3.14,+3.14] FRD, thr 1.0).
- Then: rolled through 42° in 0.1 s, blew past ±90–180° tilt, yaw spun >200°, tumbled into the
  left gate post (dy −1.0 m). race_outcome: **0 passes, 4× COLLISION(1001), no advance**.

## 6. Root causes (each measured, none deployment-fixable)

1. **Rate-transient overshoot:** live realized rates peaked **9.7 rad/s** vs the twin's first-order
   ceiling 7.85 (= 3.14·2.5). The twin models steady gain only (the 2026-06-03 sysid note said so).
   Replay from the exact live handoff state passes gate 0 in the twin at any latency ≤ 100 ms and
   gain ×1.24 — the twin **cannot** reproduce the tumble; the divergence lives in the unmodeled
   transient + item 2.
2. **±180°-tilt yaw-spin sim anomaly** (known VQ2 open item A, untwinned): the policy's NOMINAL
   maneuver rolls through **104–126° before every gate** (measured in-twin) — it flies straight
   into that regime live.
3. **Unmodeled latency:** ~40 ms transport + obs staleness ≈ 1–2 control steps; training DR had no
   transport delay (adapter ring buffer was deliberately omitted). Offline: 2 steps costs gate 3+,
   3 steps costs gate 1+.
4. **Thrust ceiling:** trained max_normed_thrust 5.0 (≈5 g); live collective clips at 1.0 ≈ 3.765
   normed (≈3.77 g). Even from the perfect training-reset state, the live clip fails gate 4 by
   ~2 m in-twin (the pull-out of the steep descent needs the missing authority).

## 7. S1.3 retrain spec (`rl/peregrine_racing_s13.sbatch` — ready to dispatch)

1. `dynamics.controller.max_normed_thrust=3.765` (one-line override; fixes §6.4).
2. `+env.standing_start_frac=0.3` — real-spawn resets, **implemented** in `peregrine_racing.py`
   (cfg-gated, default off; spawn pose mapped through the deployment virtual flip so `fly_rl.py`
   needs no change). Drops the CTBR bridge eventually; bridge remains the fallback.
3. `env.reward_weights.quadrotor.attitude=2.0 env.reward_weights.quadrotor.jerk=0.3` — tilt
   regularization (fixes §6.2 exposure + the VQ2 "over-aggressive" caveat). Starting values; sweep.
4. **Adapter edits needed (NOT yet implemented — flag for the Adroit session):**
   `rl/diffaero_dynamics.py`: (a) port the transport-delay ring buffer to the torch backend
   (rl_plant already supports it ⇒ parity-checkable via `check_against_rl_plant` with
   `transport_delay_steps>0`), DR per-env delay ∈ {0,1,2} steps; (b) widen the rate_gain DR band
   asymmetrically to about [−10%, +30%] as a cheap transient-overshoot proxy (measured +24%).
   Optional better fix: re-system-ID the inner loop as 2nd-order from this session's tumble
   recordings (saturated step commands with realized-rate telemetry = ideal data;
   `data/runs/20260610_205414_rl_s12_f1` and the five `20260610_*_rl_s1_v1` runs).

## 8. Sim automation (PART 0 — solved, document for all future sessions)

- `MAV_CMD 31000` (`client.send_sim_reset()`) **works once a race context exists**: it restarts the
  race (fresh ~3 s countdown). It does NOTHING at the HOME page (no telemetry there at all).
- From HOME: focus the `AI-GP` window (Win32 `SetForegroundWindow`; `WScript.Shell.AppActivate`
  alone returned False) and send Enter twice (home → waiting room → race+countdown). Works
  unattended; this session launched `FlightSim.exe` cold and raced with no human.
- `fly_rl.py --flights N` chains attempts: per-flight recorder sessions, auto-reset between
  flights, manual-fallback hint, never fires a reset into a ticking countdown.

## 9. Files

- `rl/fly_rl.py` — corrected deployment (tanh+rescale, FLU→FRD [1,-1,-1], odo_rate_sign [-1,-1,1],
  collective obs 0-init + rescaled feedback, 30 Hz, virtual flip, CTBR bridge + handoff, auto-reset
  multi-flight loop, --max-rate/--max-thrust diagnostics).
- `rl/offline_rollout.py` — the offline twin evaluator (start states: trainreset / racestart /
  simstart / handoff; latency; caps; live-thrust-clip; build_obs consistency check).
- `rl/peregrine_racing.py` — standing-start reset (cfg-gated, default off).
- `rl/peregrine_racing_s13.sbatch` — proposed S1.3 dispatch.
- Flight 1 recording: `data/runs/20260610_205414_rl_s12_f1` (race_outcome: 0 passes, 4 gate hits).
