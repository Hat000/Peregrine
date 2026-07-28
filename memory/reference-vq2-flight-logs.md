---
name: reference-vq2-flight-logs
description: "Where the real VQ2 flight/sysID logs live on GitHub + what each contains + suitability for per-tick plant validation (pose-blind wire: only gyro+accel are GT)."
metadata: 
  node_type: memory
  type: reference
  originSessionId: c864e5bb-f53f-4b54-b78a-a2216bb2c906
---

# VQ2 real-flight & sysID log inventory (GitHub Hat000/Peregrine) — 2026-07-16

Mapped while building the plant-fidelity validation ([[perception-sim2sim-gap-2026-07-15]]). **Pose-blind wire = the governing constraint: VQ2 emits NO ground-truth position/velocity/attitude anywhere** (no ODOMETRY/mag/baro). The ONLY real sensor GT is **HIGHRES_IMU (gyro + accel)**. So plant validation = per-tick IMU match; position is the (uncheckable) integral of accel — do not chase it.

## The instrument (BEST) — controlled sysID captures
- **Branch `sysid-handoff-2026-07-15`, `data/runs/*_sysid_*_f1/sysid_vq2_log.csv`** (20 captures). **SELF-ALIGNED**, columns: `k,phase,sim_time_ns, a_thrust,a_roll,a_pitch,a_yaw (raw cmd), cmd_wx/wy/wz,cmd_thrust,collective, gyro_x/y/z, accel_x/y/z`. Command AND IMU response in one file — same column structure as our DiffAero replay log (`scratchpad/sysid_diffaero_log.csv`). No alignment problem. `boot` phase = on the 17° tilted pad (accel_x≈−3.0, accel_z≈−9.34 = g·sin/cos 17.8°, encodes the IC); `prog` = the maneuver. (Sibling `sysid_vq2_imu_raw.csv` = IMU only, superseded by the `_log.csv`.)
- Captures: `yaw_ampsweep` (STEPPED holds = the clean money run), `roll_sweep`/`pitch_sweep` (continuous → lag-depressed), `roll_dyn`/`pitch_dyn` (step responses → τ), `yaw_bw` (bandwidth), `roll_fill`/`pitch_fill` (amplitude fill), `thrust_curve`/`thrust_ff`/`thrust_hover3g` (collective sweeps), `vzsweep`/`hover_vzsweep`/`vzsweep_lo` (climb-velocity → inflow lapse), `pitch_level`/`pitch_trulevel`/`pitch_30` (tilt-corrected), `sysid_vq2_f1` (generic/full).

## Racing-regime (RICH but CONFOUNDED for plant gain)
- **Branch `claude/ego-deploy-2026-07-09`, `data/runs/*_panel_run_f1/ego_obs.jsonl`** — dozens of real 40 Hz (≈36 Hz decision) deploy flights. Commits: **`2aca009`** (5× vpef8nc OSCILLATION), **`28404fa`** (deep vpeffs0 ACCUMULATION, reaches gate 5 @13 m/s), `c344e9a` (×6), `80dd3a1` (×21). Per tick: `obs` (21-dim: [0:3] body vel, [3:5] roll/pitch, **[5:8] body rates=gyro**, [8] prev thrust, …), `act_raw` (raw policy action), **`rate_frd`** (commanded rate FRD), `collective`, `kf_pos_ned` (KF estimate, vision-anchored), `ego_timing.jsonl` (work_ms/nav_ms vision latency), `video_index.jsonl` (30 Hz frame clock).
- 🛑 **NOT a clean plant instrument.** Per-tick `|achieved|/|cmd|` on fast-flipping racing commands is dominated by closed-loop LAG/PHASE/TIMING, not plant gain (measured VQ2 roll "gain" 2.94→1.59 DECREASING = a lag artifact). Use the controlled sweeps for the plant verdict. (A 1-step-ahead re-init test MIGHT salvage a racing read — being checked.)

## Full-wire tlogs (IMU + actuator only — NO pose)
- **`sysid-handoff-2026-07-15`, `handoff/ego-flight-*/mavlink.tlog`** (a5/a7/a8 bring-up; `vpeffs0`/`vpefwh2` ego-flights). Parse with pymavlink (`.venv` has 2.4.49). Contain **HIGHRES_IMU (gyro+accel) + ACTUATOR_OUTPUT_STATUS + RACE_STATUS**; **NO ODOMETRY/ATTITUDE** (pose-blind confirmed by direct parse). Command must be reconstructed → limited.

## NOT suitable
- **`race-wire-capture-2026-07-05`, `docs/race-wire-2026-07-05/*.race_wire.jsonl`** — RACE_STATUS + COLLISION only (race-outcome extract). No plant state/command. (This is the one to NOT reach for.)

## Fetch idiom
`gh api "repos/Hat000/Peregrine/contents/<path>?ref=<branch>" -H "Accept: application/vnd.github.raw"` → stdout (binary-safe for tlogs). Tree: `gh api "repos/Hat000/Peregrine/git/trees/<branch>?recursive=1" -q '.tree[].path'`.
