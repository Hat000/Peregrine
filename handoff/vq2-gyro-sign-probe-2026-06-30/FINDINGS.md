# VQ2 gyro-frame sign probe — result — 2026-06-30

**Goal:** determine the sign of the live VQ2 HIGHRES_IMU gyro on EACH body axis vs the commanded body-rate,
to know if the A9 pitch inversion is **y-only** or a **full handedness flip** (which would also hit roll/yaw).

**Answer: it is NOT y-only — ALL THREE axes (roll, pitch, yaw) are sign-inverted, with no axis swap.**
The fix must negate the **entire** gyro vector before the AHRS, not just pitch.

## Method (estimator-independent, no stable flight)
Reused the proven flying path (`racer.mavlink_client.MavlinkClient` — same ARM + `SET_ATTITUDE_TARGET`
body-rate uplink, type_mask `0b10000000`, that flew A9), with **`cmd_rate_scale=1.0`** so the commanded
sign is raw. ARM (MAV_CMD 400 p1=1) into a live R2-TRAINING/VQ2 race, then single-axis body-rate pulses
(+1.0 rad/s, 0.4 s, returning to zero between): `+pitch`, `+roll`, `+yaw`. Logged raw HIGHRES_IMU
`x/y/zgyro` (`state.gyro_body`) at ~100 Hz; per pulse, mean realized gyro over the first 0.2 s vs the
commanded sign. Script: `gyro_frame_probe.py`; raw log: `probe_log.json`; table: `result_table.json`.

## Result
```
  axis | cmd sign | realized raw-gyro sign | mean gyro (rad/s) | INVERTED?
 pitch |    +     |          -             |      -2.18        | YES   <- reproduces A9 (sanity check)
  roll |    +     |          -             |      -2.20        | YES   <- new
   yaw |    +     |          -             |      -2.02        | YES   <- new
```
Cross-axis check (mean gyro on ALL axes over each pulse's first 0.2 s) — the commanded axis dominates
cleanly, no coupling, no axis remap:
```
  +pitch  cmd=[0,1,0]  realized gyro = [roll +0.00, pitch -2.18, yaw -0.00]
  +roll   cmd=[1,0,0]  realized gyro = [roll -2.20, pitch -0.02, yaw +0.00]
  +yaw    cmd=[0,0,1]  realized gyro = [roll -0.02, pitch -0.00, yaw -2.02]
```

## Interpretation
- **Every axis: commanded +1.0 rad/s -> realized gyro ~-2.1 rad/s.** Sign is negated on roll, pitch AND
  yaw; magnitude ~2.1x is the known command->realized rate gain (≈ the documented ~2.5x).
- **No axis permutation/swap** — each commanded axis maps to the SAME gyro axis (negated), so this is a
  uniform `gyro_reported = -k * omega` on the whole vector, not a coordinate-axis relabeling.
- **This is NOT a simple FRD<->FLU rotation.** FRD<->FLU (180deg about x) would invert only y/pitch and
  z/yaw and LEAVE roll/x un-inverted. Roll IS inverted here, so the whole angular-rate vector is negated
  (a full sign/handedness convention mismatch on all three rate axes), not the y-only or FLU case.
- For pitch, A9 already anchored the physical direction (pilot watched cmd `+pitch` -> nose physically UP,
  gyro read negative), establishing that the COMMAND produces the intended motion and the GYRO is the
  inverted reporter. The other two axes show the identical cmd-vs-gyro sign disagreement; the consequence
  for the AHRS is the same regardless of attribution: the gyro fed to the estimator has the opposite sign
  to the convention the controller commands in, on all three axes.

## Recommended fix (commander's call; stack NOT touched)
- **Negate the full HIGHRES_IMU gyro vector** (`gyro_body <- -gyro_body`, all 3 components) at the live-wire
  ingest, before it feeds the AHRS — `mavlink_client.py:~249` currently stores it as "TRUE FRD, no sign
  change", which this probe refutes for the live VQ2 wire.
- Equivalently, fix the HIGHRES_IMU body-frame adapter so the live gyro matches the code's FRD/command
  convention. The offline sign-audit passed because synthetic IMU is self-consistent in the code's assumed
  convention and cannot see a live wire mismatch — validate on the live wire (this probe), not synthetic.
- A full-vector negation also flips yaw, so re-check the vision-yaw / gate-bearing-yaw integration after the
  fix (those consume gyro continuity for branch disambiguation).

## Artifacts
- `gyro_frame_probe.py` — the probe (ARM + single-axis body-rate pulses + raw-gyro logging + sign table).
- `result_table.json` — the 3-row table. `probe_log.json` — full ~100 Hz raw log (t, sim_ns, gyro, cmd, label).
- Flight stack untouched; handoff dir only.
