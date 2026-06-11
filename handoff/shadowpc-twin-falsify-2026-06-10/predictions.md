# Twin predictions (map-ON faithful config) -- COMMITTED BEFORE FLIGHT

config: hover=0.2656  drag=0.2111/s (linear, world-frame, isotropic)  G0=[2.501, 2.504, 2.231]  s=0.3  alpha_max=[260.0, 260.0, 80.0]  tau=0.019

## P1 aero/drag at speed
| tilt (deg) | v_term (m/s) | v @3s held | v @5s held |
|---|---|---|---|
| 8 | 6.53 | 3.06 | 4.26 |
| 10 | 8.19 | 3.84 | 5.34 |
| 12 | 9.87 | 4.63 | 6.44 |
| 17 | 14.20 | 6.66 | 9.26 |
| 25 | 21.66 | 10.16 | 14.12 |
| 32 | 29.03 | 13.62 | 18.93 |

- coast decay: v(t)=v0*exp(-0.2111*t), tau = 4.74 s (numeric CtbrPlant check: 4.74 s); a(v) = -0.2111*v EXACTLY LINEAR through origin, no v^2 term
- per-axis: IDENTICAL forward / backward / lateral / vertical (isotropic world drag);
  body-frame drag or any fwd/lat asymmetry falsifies the model form
- attitude during coast does not matter (world-frame drag): a coast at level attitude
  and a coast while tilted (alt-held) decay identically

## P2 rate loop at airspeed (twin: NO airspeed dependence)
| |cmd| (rad/s) | sustained rate roll/pitch (rad/s) | gain |
|---|---|---|
| 0.3 | 0.772 | 2.575 |
| 1.0 | 2.765 | 2.765 |
| 3.14 | 11.216 | 3.572 |
- identical at 0 m/s and 12 m/s; slew 260 rad/s^2; tau 19 ms. Any shift with airspeed
  falsifies the static map's completeness.

## P3 collective map (twin: linear, a_up = g*(T/0.2656 - 1) - d*vz)
| collective | a_up @vz=0 (m/s^2) | climb v_term (m/s) |
|---|---|---|
| 0 | -9.81 | -- |
| 0.1 | -6.11 | -- |
| 0.2 | -2.42 | -- |
| 0.2656 | +0.00 | -- |
| 0.32 | +2.01 | 9.5 |
| 0.4 | +4.96 | 23.5 |
| 0.55 | +10.50 | 49.7 |
| 0.7 | +16.04 | 76.0 |
| 0.85 | +21.58 | 102.2 |
| 1 | +27.12 | 128.5 |
- T=0 is exactly free fall (-g); no thrust floor, no nonlinearity anywhere on [0,1]
- zero collective->attitude coupling (collective is a pure body -Z force)

## P4 long-duration drift (twin: NONE)
- hover collective is 0.2656 at t=0 and at t=8 min; any monotonic drift falsifies

## P5 control-rate sensitivity (twin: discretization only)
- same maneuver at 50/100/200 Hz differs only via integration step of the SAME ODE;
  realized sustained rates / speeds shift < ~1%. A systematic shift (e.g. the 1.0.3364
  tick-phase coupling) falsifies.

## P6 determinism (twin: exact)
- twin is deterministic; live run-to-run spread = the measurement noise floor that all
  other comparisons inherit. No prediction, this CALIBRATES.

## P7 mixed-axis coordinated turn
- predicted offline by replaying the recorded command stream through CtbrPlant
  (replay_twin.py); banked turn at 20 deg / ~6 m/s -> radius v^2/(g*tan20) ~ 10.1 m,
  turn rate ~0.59 rad/s. Error table comes from the replay, same as the sweep style.
