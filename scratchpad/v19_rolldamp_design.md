# v1.9 roll-damped design (ready to wire once the roll-penalty build lands)

## Premise (wire-measured, 21 flights)
Close-in roll limit cycle is THE gate-count ceiling, pre-existing since v1.6:
- BOUNDED roll (peak <1.0 rad/s): n=8, gates mean 2.12, owns the tail [..4,4,6].
- SPIKED roll (peak >=1.4): n=10, gates mean 1.20, capped at 3, NO tail.
- Every flight reaching >=3 gates needed bounded roll. Damping roll should convert spiked->bounded
  and lift the mean toward 2.12 + reopen the tail. NOT a silver bullet (bounded flights still have 0s
  from other causes).

## GATE BEFORE LAUNCH (do NOT skip -- burns 20h x3 if wrong)
Does the limit cycle reproduce in SIM (clean pose) or is it a seeker-EMA-lag deploy artifact?
- Cheap first check (no new job): grep the existing v18 training .out for YAW_EVAL `roll_swing`
  (peak roll ANGLE, deg). High roll_swing in sim => aggressive roll is in-distribution => reward fix bites.
- Clean check (post-build): rollout_only eval emitting the NEW rate-based ROLL_EVAL on v18Qs0.
- If sim is SMOOTH but wire oscillates => the cause is the seeker EMA (alpha 0.5) lateral phase lag
  close-in; pivot to a seeker fix (lower alpha / phase-compensate close-in) INSTEAD of / with the reward.

## Config (v1.9 = M3 + roll rate-penalty), pending the build's exact +env. key names
Base = launch_v19.sh (M3-only) already prepared; ADD the roll block:
- rw_roll_jerk  Q=0.05  W=0.08   (PRIMARY lever; turn-safe -- smooth 60deg turn-in = low |dcmd|,
  limit cycle = high |dcmd|. Mirror rw_yaw_jerk 0.05 / rw_pitch_jerk 0.03.)
- rw_roll_duty  Q=0.10  W=0.20   roll_duty_free_band=0.8   (SECONDARY; free band 0.8 leaves the
  observed normal turning <0.5 rad/s untaxed; deaths spike to +-2, well separated.)
- KEEP M3 (rw_progress_frame_mult 1.0 / floor 0.5 / perc 0.016). M4 reverted (pass_margin_final_m 1.0).
  M1 off, M2 off. Warm from v18Q_s1 (deploy lead, release dive fixed).
- Arms Q(s0,s1) + W(s0), 3 seeds. NO roll ANGLE fence (backfires on 60deg turns; ego_reward.py ~L323).

## Smoke adjudication (the roll-specific reads)
- ROLL_EVAL signflips_per_s + satur_duty must FALL vs the v18 baseline (the mechanism biting).
- n_passed_gates must NOT collapse (over-damped roll => can't turn => gates drop). If it does, back
  rw_roll_jerk toward 0.03 / widen free_band.
- frame_factor (M3) holds/climbs; release-dive tail stays fixed (replay_sweep_v18.py).
- Multi-seed or it doesn't count (~2.8x seed variance).
