# Raw HIGHRES_IMU sample for RL emulated-IMU calibration (ASK 3, RL commander 2026-07-05)

`highres_imu_60s.csv` — 60.0 s / 8609 samples of RAW wire HIGHRES_IMU from a LIVE VQ2 flight,
run `20260702_024203_rl_s1_f1` (107 s active near-gate maneuvering, real-commands era), window
auto-selected for max accel variance (starts t+6.4 s, post-liftoff; accel std 5.16 m/s²,
gyro std 0.145 rad/s — genuinely in-flight, not bench/waiting-room).

## Measured stream properties (correct your assumptions)
- **Rate: ~143.5 Hz median** (NOT the 117 Hz you quoted). dt p99 = 14 ms — there ARE timing gaps;
  emulate the jitter, not just the rate.
- Columns: `t_s` (from window start), `xacc,yacc,zacc` (m/s²), `xgyro,ygyro,zgyro` (rad/s).

## ⚠️ CONVENTION TRAPS (your FYI-a class — both are known, wired-in deploy corrections)
1. **Gyro polarity: this CSV is the RAW WIRE, which is INVERTED on ALL THREE axes vs code-FRD.**
   The deploy stack negates the full gyro vector before use (A9 fix, 2026-06-30, validated by A10
   attitude stability). If your emulator matches code-FRD conventions, negate the gyro columns.
2. **Accel is SPECIFIC FORCE incl gravity reaction, body FRD** (hover reads ≈ [0,0,-g]).
   The deploy KF adds gravity back via the given attitude (state_estimator.predict).

Both corrections live in the deploy chain AFTER this raw capture point — so this file is the right
thing for a WIRE-faithful emulator, and the wrong thing to feed a code-FRD consumer unmodified.
