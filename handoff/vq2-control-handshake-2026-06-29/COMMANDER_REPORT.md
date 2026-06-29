# Commander Report — VQ2 Control Handshake

**Date:** 2026-06-29 · **Machine:** ShadowPC · **Sim:** AI-GP build 1.0.3379 · **Event:** R2-TRAINING (VQ2)
**Author:** control-handshake investigation agent · **Branch:** `vq2-control-handshake-2026-06-29`

---

## BOTTOM LINE

**The control handshake is SOLVED on VQ2. The loop-closing blocker is cleared.** We have stable,
commanded CTBR flight using the uplink path we already own — no new message plumbing, and it is the
same path `rl/fly_rl.py` already drives. **Confidence: HIGH.**

A secondary question — "is the sim lagging on the control end?" — was investigated and answered:
**No. The control/telemetry wire runs at real time; the lag is display-side only.**

---

## THE WORKING RECIPE (one line)

> ARM (`MAV_CMD_COMPONENT_ARM_DISARM`=400, p1=1), then stream `SET_ATTITUDE_TARGET` in **body-rate
> mode** (`type_mask=0b10000000`, attitude quat ignored) at ~100 Hz (30 Hz deploy also works) with
> **FRD body rates (rad/s) + normalized collective thrust [0,1]**. The sim is in ACRO by default.

This is exactly `ControlMode.BODY_RATE` in `src/racer/mavlink_client.py` and what our policy outputs.
**It transfers directly to deploy.**

---

## EVIDENCE (objective; VQ2 wire carries no pose, so judged on IMU accel, motor outputs, collisions,
real-time clock, and HUD)

| Test | Result |
|---|---|
| **Arm** | ACCEPTED (`MAV_RESULT_ACCEPTED`), no force/bypass needed |
| **CTBR hover** (zero rates) | 4 motors balance at the commanded collective; gyro nulls to 0; 0 collisions → stable |
| **CTBR maneuver** (+0.4 rad/s pitch) | perfectly symmetric motor differential `[0.366,0.366,0.334,0.334]` (pure pitch torque); gyro rises to a **steady, bounded ~1.0 rad/s** and holds — predictable, recoverable, not divergent |
| **Lift** | thrust ≥0.4 climbs cleanly (balanced motors) until it contacts geometry |
| **Failure mode (for contrast)** | velocity setpoints in ACRO TUMBLE (no velocity controller in a rate mode); attitude-quat is ignored in ACRO |

---

## THE LAG QUESTION — ANSWERED

Display/GPU (or remote-desktop) lag is **decoupled** from the physics + telemetry path. Measured
on-box, independent of the screen:

| Metric | Idle | Under 100 Hz control | **During active flight** |
|---|---|---|---|
| Real-time factor (sim clock vs wall-clock) | 1.000 | 1.001 | **1.001** |
| HIGHRES_IMU rate | 115 Hz | 117 Hz | **119 Hz** |
| ACTUATOR_OUTPUT_STATUS | 93 Hz | 95 Hz | **96 Hz** |
| IMU inter-arrival jitter (max) | 34 ms (one-off) | 11 ms | **10.5 ms** |
| Our command send rate | — | 100.0 Hz | 100.0 Hz |
| BAD_DATA / dropped | 0 | 0 | 0 |

**Lag is not a cause of the crashes.** The crashes are environmental (see below), confirmed by the
clean, proportional control telemetry right up to the moment of geometry contact.

---

## VQ2-SPECIFIC FACTS THE TEAM MUST KNOW

1. **VQ1 vs VQ2 is the MENU EVENT, not the track name.** Both events use track "Now You See Me, Now
   You Don't" (this caused an early mislabel — see Caveat). Main-menu list: `R1` = VQ1,
   `R2 - SUBMISSION`, `R2 - TRAINING` = VQ2. **From the main menu, spam DOWN** — it clamps on the last
   item (R2-TRAINING = VQ2). VQ1 renders as bare wireframe; VQ2 is a lit indoor warehouse.
2. **The drone spawns INSIDE the start gate** in a confined course. Low thrust sags into the gate;
   high thrust climbs into the ceiling. Gate (id 1001) contacts during a stationary hover are
   **environmental, not control failures.**
3. **A crash ENDS the episode → returns to menu, and the wire then replays a FROZEN state.**
   Live-vs-frozen check: `RACE_STATUS @4 Hz` + `RTF≈1.0` = live; `RACE_STATUS=0` + `RTF≪1` = frozen.
   The probe enforces this with a `live` guard.
4. **`MAV_CMD_SIM_RESET` (31000) does NOT recover a crashed VQ2 episode** (it works on VQ1).
   Reliable reset = relaunch the sim process and re-navigate the menu.

---

## DEPLOY HANDOFF — two deltas for `rl/fly_rl.py` (already on the BODY_RATE path)

1. **Rate feedback must come from `DroneState.gyro_body` (raw HIGHRES_IMU), NOT `angular_rate_body`**
   — ODOMETRY angular rate is blocked/zero in VQ2.
2. **Command→realized rate gain ≈ 2.5×** (VQ2: cmd 0.4 → 1.0 rad/s; VQ1 cross-check: 0.6 → 1.56).
   Proportional and bounded — calibrate the controller's rate scale, or rely on the closed-loop
   policy to absorb it.

---

## CAVEAT / TRANSPARENCY

The **first** version of this work collected its evidence on **R1 = VQ1 by mistake** (identical
track name fooled the HUD read). It was caught, fully re-run on actual VQ2, and corrected; this
report and all `logs/vq2_*` artifacts are the re-verified VQ2 data. `logs/vq1_*` are retained as a
cross-check — the recipe is identical on both, which is itself reassuring.

---

## RECOMMENDED NEXT STEPS

1. **Wire the two deltas into the deploy loop** (gyro_body feedback; calibrate ~2.5× rate gain), then
   run the policy live on VQ2 — the handshake no longer blocks this.
2. (Optional) A longer free-flight-through-gates run to characterize the rate gain across the speed
   envelope and confirm gate-sequencing under the policy.
3. Treat the confined start-gate spawn as a **launch-transient** case for the controller (smooth
   ramp off the start, as the Mission `launch_ramp` already anticipates).

**Artifacts:** branch `vq2-control-handshake-2026-06-29` → `handoff/vq2-control-handshake-2026-06-29/`
(`CONTROL.md` full writeup, `ctbr_probe.py`, `telem_health.py`, `logs/vq2_*` evidence + HUD shots,
`logs/vq1_*` cross-check). Main untouched.
