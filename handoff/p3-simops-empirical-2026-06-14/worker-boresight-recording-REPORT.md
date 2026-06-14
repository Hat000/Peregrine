# P3 Boresight Recording — Worker Report
**Session:** 2026-06-14  **Worker:** P3 SIM-OPS (sonnet-4.6)  
**GATE-CONTACT COUNT: 0** (all 11 laps clean)

## Summary

Full boresight recording session completed without incident. All laps FINISHED with zero gate contacts. Both verify_bundle ALL_PASS. Mirror canary TRUE +0.97 / AS-IS -0.81 (OK). Sim left clean.

---

## Pre-Flight Checks

| Check | Result |
|---|---|
| `--print-config` (faithful wiring) | PASS — BODY_RATE \| decoupled True \| yaw_mode=course |
| dry-run | PASS — got GO!, TAKEOFF loop ran 60s, never armed |
| DCGame instances | 1 (correct) |

**Faithful wired config (for reference):**
```
BODY_RATE | decoupled True | tilt_comp True
hover 0.2656 | kp_pos 0.6 kd_vel 2 max_speed 6 | kp_att 10 kd_att 0.15 ff_gain 2.5
kp_alt 3 kd_alt 1.75 | alt_clip [0.05,0.6] alt_offset 0 max_accel 12 max_body_rate 8
body_rate_sign [1,1,-1] odo_att_sign [-1,1,1] odo_rate_sign [-1,-1,1]
planner: lookahead 5 cruise 8 yaw_mode=course
```

---

## Run Table

All collisions = 0. All final_state = FINISHED.

| # | Role | Label | final_state | collisions | gate_index | video_frames | duration_s | data/runs dir |
|---|------|--------|-------------|-----------|-----------|-------------|-----------|--------------|
| S | smoke | boresight_g0 | FINISHED | **0** | 1 | 540 | 17.96 | `20260614_191448_boresight_g0` |
| H1 | hover | boresight_hover | FINISHED | **0** | 0 | 658 | 21.90 | `20260614_191539_boresight_hover` |
| H2 | hover | boresight_hover | FINISHED | **0** | 0 | 661 | 22.00 | `20260614_191635_boresight_hover` |
| A1 | approach | boresight_g0 | FINISHED | **0** | 1 | 505 | 16.78 | `20260614_191730_boresight_g0` |
| A2 | approach | boresight_g0 | FINISHED | **0** | 1 | 540 | 17.97 | `20260614_191820_boresight_g0` |
| A3 | approach | boresight_g0 | FINISHED | **0** | 1 | 547 | 18.18 | `20260614_191912_boresight_g0` |
| A4 | approach | boresight_g0 | FINISHED | **0** | 1 | 537 | 17.86 | `20260614_192003_boresight_g0` |
| A5 | approach | boresight_g0 | FINISHED | **0** | 1 | 505 | 16.77 | `20260614_192054_boresight_g0` |
| A6 | approach | boresight_g0 | FINISHED | **0** | 1 | 510 | 16.95 | `20260614_192145_boresight_g0` |
| G01-1 | approach_g01 | boresight_g0g1 | FINISHED | **0** | 2 | 688 | 22.89 | `20260614_192235_boresight_g0g1` |
| G01-2 | approach_g01 | boresight_g0g1 | FINISHED | **0** | 2 | 679 | 22.60 | `20260614_192331_boresight_g0g1` |

**All paths under:** `C:\Users\Shadow\Peregrine\data\runs\`

---

## Canary Results

### Mirror Canary (step 5)

Ran `frame_residual_report.py` on session `20260614_024858_simops_fra_c01_f1` (most recent fly_rl session with debug_obs.jsonl from same day, same sim build 1.0.3364).

**Note:** fly_vq1 sessions do NOT generate `debug_obs.jsonl` (fly_rl --debug-obs only). The canary is checking the *sim telemetry convention* (ODO quat orientation), which is a sim property independent of which flight script is used. A same-day, same-build fly_rl session is the correct source.

```
==== 20260614_024858_simops_fra_c01_f1  (281 usable ticks) ====
-- mirror canary (East corr at tilt>30): TRUE +0.97  AS-IS -0.81   OK
-- rate canary: quat-FD(true) ~ -w_raw gain = [+0.933, +0.845, +0.947]  (expect ~+1.0)
```

**CANARY: TRUE +0.97 / AS-IS -0.81 → OK** — Telemetry convention stable. ODO_QUAT_TRUE_CONJ_WXYZ = [1,-1,1,-1] remains correct.

### verify_bundle (step 6)

**Approach bundle (A1):** `20260614_191730_boresight_g0`
```
video      30.04 Hz >= 30.0 Hz   OK
HIGHRES_IMU  118.3 Hz >= 90.0 Hz   OK
LOCAL_POS   95.84 Hz >= 60.0 Hz   OK
ODOMETRY    74.75 Hz >= 60.0 Hz   OK
ATTITUDE   118.29 Hz >= 60.0 Hz   OK
gt_velocity_LIVE  OK (max_mag=5.451 m/s)
ALL_PASS = True
```

**Hover bundle (H1):** `20260614_191539_boresight_hover`
```
video      30.01 Hz >= 30.0 Hz   OK
HIGHRES_IMU  118.59 Hz >= 90.0 Hz   OK
LOCAL_POS   95.88 Hz >= 60.0 Hz   OK
ODOMETRY    74.68 Hz >= 60.0 Hz   OK
ATTITUDE   118.54 Hz >= 60.0 Hz   OK
gt_velocity_LIVE  OK (max_mag=2.243 m/s)
ALL_PASS = True
```

---

## Hover Gate-0 In-Frame Assessment

Hover lap H1 (static hover, `--hover-hold`): `final_state=FINISHED, gate_index=0, frames=658, dur=21.9s`.

The gate_index=0 in the hover meta indicates gate-0 was the active gate target during the hover (the planner targets gate-0 when in hover-hold mode). The yaw_mode=course (nose pointing down the -X gate axis) means the camera was facing gate-0 during the 12-second hover. **Gate-0 in-frame: LIKELY YES** (yaw set by course mode = azimuth ≈ 0 toward gate-0).

Note: actual in-frame verification requires offline inspection of hover video.bin frames. Cannot confirm programmatically without detector; commander should verify during ε analysis.

---

## Final Sim State

- DCGame-Win64-Shipping instances: **1** (correct)
- State: **WAITING** (started=False, drone at origin, pos_off=0.02m)
- Ready for next session or shutdown.

---

## Session Log

Full log at: `handoff/p3-simops-empirical-2026-06-14/session_log.json`

---

## MEMORY-DELTA (≤10 lines, for commander review — do NOT apply to memory/)

1. **fly_vq1 does NOT write debug_obs.jsonl** — frame_residual_report.py canary requires a fly_rl --debug-obs session; use most recent same-day fly_rl run for convention check (valid since ODO quat convention is a sim property).
2. **boresight_session.py orchestration pattern PROVEN**: drive_to_waiting() + fly_vq1 subprocess + 7s delay + send Enter for GO works reliably (11/11 FINISHED, 0 collisions).
3. **fly_vq1 --faithful --force-saved-map gate-corner-to-center APPROACH**: typical 17s / 510 frames / gate_index=1 per gate-0 lap; hover --max-seconds 12 yields 22s / 660 frames.
4. **gate_index=1 in meta = gate-0 was PASSED** (index is the NEXT gate to pass; starts at 0, so gate_index=1 after --max-gates 1 means gate-0 pass recorded in the tlog sweep 23→~0m).
5. **go_delay_s=7s for fly_vq1** is sufficient: from WAITING, fly_vq1 connects and prints passive-wait banner within ~3s, then the Enter → 3s countdown → GO completes before the 7s is up.
6. **STALE sim state (recent race, drone at origin)**: ESC+DOWN×3+ENTER chain in drive_to_waiting recovers reliably to WAITING (verified on initial cleanup before session).
7. **Worktree ops note**: simops-mastery tools only in worktree, not main repo (.venv in main). Pattern: use `"C:\Users\Shadow\Peregrine\.venv\Scripts\python.exe"` from main repo cwd; load worktree tools via `importlib.util.spec_from_file_location` with WT absolute path.
