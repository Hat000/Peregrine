# Raw HIGHRES_IMU samples for RL emulated-IMU calibration (v2, CORRECTED 2026-07-05)

## ⚠️ CORRECTION of the v1 delivery
`highres_imu_60s.csv` (now `highres_imu_60s_MISLABELED_resetslam_plus_stationary.csv`) was labeled
"in-flight". It is NOT flight data. Re-segmentation shows: t=[0,4) = the race-RESET teleport slam
(accel std up to 34 m/s² — an artifact, the drone is teleported to the start pad), t=[4,73] = pure
waiting-room STATIONARY ([0,0,-9.81], std ≤0.02). Run 20260702_024203 actually flew only ~6 s before
CRASH (meta.json: race_start→crash ≈ 4-6 s, gate_index 0) — my "107 s active maneuvering" claim was
wrong, and the max-variance window selector locked onto the reset slam.
**Impact on your v1 analysis:** your quiet-axis white floor was measured on genuinely stationary
data — that number SURVIVES as the stationary white floor. Your "big x/z content = real pitch-plane
vibration" does NOT survive — that content was the teleport transient. Re-derive flight noise from
the new flight file below.

## Files (columns: `t_s, xacc,yacc,zacc [m/s²], xgyro,ygyro,zgyro [rad/s]`, t from segment start)

### `highres_imu_stationary_82s.csv` — your bias/noise-split ask
Run `20260702_024203_rl_s1_f1`, tlog t=[9.25,91.63]: post-reset, pre-GO, LIVE physics, armed-idle
LEVEL on the start pad. 10380 samples, 82.4 s, median 143.5 Hz, dt p99 14 ms, max gap 22 ms.
- accel mean `[-0.0000, -0.0001, -9.8099]`, std `[0.0007, 0.0083, 0.0062]`
  → **accel bias ≲ 0.004 m/s² on every axis** (exact z-bias depends on the sim's g constant:
  9.80665 → -0.0033; 9.81 → +0.0001). Your 0.15 m/s² worry → ~0.7 m over a 3 s coast becomes
  **< 2 cm**. The sim models essentially zero accel bias in this run.
- gyro mean `[+4.0e-4, +3.9e-4, +0.1e-4]` rad/s (≈0.023 °/s x/y) — small but resolved above the
  in-window noise; treat as the gyro-bias magnitude scale.
- ONE-RUN CAVEAT: this is the only live ground-stationary segment in any recording on this box
  (see frozen-stream trap below) — whether bias is re-drawn per run is UNANSWERED. The pending
  race-wire capture flight will produce a second waiting-room segment for free.

### `highres_imu_flight_62s.csv` — genuine sustained flight
Run `20260705_012253_rl_s1_f1` (recon mapping sweep, classical controller: sustained varied
maneuvering, aggressive attitude changes), tlog t=[53.25,115.75], 62.5 s CONTINUOUS active flight.
7308 samples, median 143.0 Hz, dt p99 14 ms, max gap 21 ms.
- accel mean `[-0.57, -0.02, -10.13]` (load factor >1g on average), std `[4.89, 6.04, 9.91]`
- gyro std `[0.69, 0.42, 0.63]` rad/s
This is the file to derive in-flight vibration/noise from. 30 s of gate-pursuit flight
(20260702_152528) and a 9 s hover-hold (012253 t=[44.3,52.8]) exist if you want other regimes.

## ⚠️ CONVENTION TRAPS (unchanged from v1 — both are wired-in deploy corrections)
1. **Gyro polarity: raw wire is INVERTED on ALL THREE axes vs code-FRD.** Deploy negates the full
   gyro vector before use (A9 fix). If your emulator matches code-FRD, negate the gyro columns.
2. **Accel is SPECIFIC FORCE incl gravity reaction, body FRD** (level rest ≈ [0,0,-g]; the deploy
   KF adds gravity back via attitude).

## ⚠️ NEW TRAPS found during re-segmentation (emulator-relevant)
3. **Frozen canned stream outside live race physics.** In menu/attract/post-crash states the wire
   keeps publishing HIGHRES_IMU at ~143 Hz with a BITWISE-CONSTANT tuple
   (accel `[-2.999, -0.002, -9.340]`, std exactly 0). Any analysis must drop bitwise-frozen
   stretches; several whole recordings contain nothing else.
4. **`time_usec` resets BACKWARD at race restart** (sim boot-time clock). Segment any log at
   backward jumps before differencing.
5. Timing jitter is real: dt p99 14 ms, occasional ~20+ ms gaps, in BOTH regimes — emulate the
   jitter, not just the 143.5 Hz median rate.
