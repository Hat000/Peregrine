# Handoff — ShadowPC FIRST CONTACT session (2026-06-02)

Written by the ShadowPC Claude session that ran first contact with the official AI-GP sim
(co-located: FlightSim.exe + DCGame-Win64-Shipping + our client). This is a SEPARATE record so the
main session can track this work without me overwriting `memory/` (the other session's). Fold what
you want into `memory/`. Official sample is ALSO on ShadowPC:
`C:\Users\Shadow\Downloads\AI-GP Simulator v1.0.3364\PyAIPilotExample\` (read it; it's the wire
authority — but its constants are USED-values, not limits, e.g. 250 Hz control != proven max).

## TL;DR
First contact succeeded. Resolved R1/R2/R3/R4, the ATTITUDE-pitch sign-inversion, clock epochs,
video rate/dup, and arming. 5 new/patched probes. Top code TODO before vision/control: drive
orientation from the ODOMETRY quaternion (ATTITUDE.pitch is sign-inverted).

## Resolved live
- R1: position+velocity GIVEN, pristine (LOCAL_POSITION_NED 97Hz + ODOMETRY 75Hz; drone parks at
  origin, v=0; mag yes). baro field present but VALUE=NaN (unusable) -> z from given position. Keep
  vision->KF in-loop anyway (walking-skeleton). VQ2 may turn position off -> then z also vision-only.
- R2 control: 3 interfaces (all already in our client): SET_ACTUATOR_CONTROL_TARGET (8ch
  [FL,FR,BL,BR,0,0,0,0]; scaling UNPROBED), SET_ATTITUDE_TARGET body-rate/CTBR (ATTITUDE_IGNORE mask
  + thrust 0..1), SET_POSITION_TARGET velocity. THE SIM'S MODE FOLLOWS THE SETPOINT TYPE:
  body-rate->ACRO, attitude-quat->ANGLE (UI showed both). Live test (in acro): velocity->no climb;
  position->VIOLENT 57m runaway for a 1m cmd (acro has no position loop -> cmd misinterpreted, NOT a
  position controller); attitude+body_rate respond. => fly on CTBR; RE-TEST position/velocity easy-
  mode in ANGLE mode. Arm = plain MAV_CMD_COMPONENT_ARM_DISARM p1=1, ACCEPTED in 83ms, NO force, NO
  OFFBOARD/GUIDED switch, NO pre-arm stream. TIMESYNC is client-initiated (timesync_send ts1=0 @10Hz;
  sim DOES respond).
- R3 course (VQ1, FPV frames): desaturated/high-contrast; red square gates in a receding sequence
  (active gate brightest); blue racing-line guide aids ON (off in VQ2); gray block-wall corridor =
  obstacles/boundaries. Real frames at data/runs/20260602_000538_firstcontact_race1.
- Rates: ATTITUDE/IMU 120, LOCAL_POS/ACTUATOR 97, ODOMETRY 75, HEARTBEAT 10, RACE_STATUS 4 Hz.
  Video: true ~28.6 fps but the sim RE-SENDS each frame ~14x (~395 completed/s) -> dedup by frame_id.
- Clocks INDEPENDENT: video sim_time_ns = server UNIX-epoch ns; IMU sim_time = sim-boot-relative.
  Align cross-stream on recv_monotonic (delayed-vision = SUSPECT branch; fine VQ1 localhost).
- Sim PAUSES physics off-race (home page / after ~8-min cap): MAVLink keeps streaming, sim_time
  FREEZES -> live control/system-ID needs an ACTIVE running race.

## CRITICAL: ATTITUDE.pitch is SIGN-INVERTED
At rest (drone pitched): ATTITUDE Euler pitch=+17.80 deg, but ODOMETRY-quaternion=-17.80 AND
accel-gravity=-17.80 AND the FPV view (camera ~level, sees course not sky) ALL agree = nose-DOWN.
=> drive orientation from the ODOMETRY QUATERNION, not ATTITUDE Euler (yaw agrees; roll maybe also
flipped, confirm in-flight). The 35.6 deg "attitude bias" the probe flagged = THIS, NOT an ESKF
trigger -- do NOT enable the bias-state.

## R4 gate map -- RESOLVED (the trigger)
The map is broadcast ONCE at race/level LOAD, to whoever is ALREADY connected (single chunk,
packets=1, ~230B, re-sent ~4x; DATA_TRANSMISSION_HANDSHAKE then ENCAPSULATED_DATA data[0]==2). We
kept missing it because every probe connected AFTER the level loaded (and wait_heartbeat eats
connect-time datagrams). PROCEDURE: connect at the HOME PAGE (wait_heartbeat=False), THEN navigate
home->waiting->Race -- the map lands at level-load. Captured 6 gates; width=height=2.72m = the OUTER
square (inner opening ~1.5m is what PnP uses -- do NOT feed 2.72 to inner-corner PnP). Sample gates:
0 ned~(-23.3,-0.4,0), 1 (-46.9,-2.5,+5.1), 2 (-74.6,+1.2,+13.7), ... (descending/curving; NED +z=down).
Deterministic course -> capture once, reuse. scripts/capture_track_map.py saves it to
data/runs/track_map_*.json. NOT YET SAVED -- run it (home->navigate-in) to bank the full 6-gate map.

## Scripts (committed, in scripts/)
- control_mode_probe.py  PATCHED: climb judged from given position_ned.z (baro is NaN); reports map.
- timesync_probe.py      NEW: client TIMESYNC@10Hz (sim responds); NOT the map trigger.
- gatemap_probe.py       NEW: full sample-startup replication; proved arm+control mid-race != trigger.
- track_probe.py         (friend's) raw handshake/chunk logger + phase walk; the CATCHING run was
                         `track_probe.py --phase-s 100` started at the home page while navigating in.
- capture_track_map.py   NEW: connect-at-home -> saves the gate map to JSON. RUN to bank the map.

## Apply-before-vision/control TODOs
1. mavlink_client: capture ODOMETRY.q -> orientation source (retire ATTITUDE-Euler signs); add test.
2. jpeg_receiver: dedup by frame_id (skip ~14x re-sends).
3. (DONE) control_mode_probe baro->given position.
4. Re-test position/velocity easy-mode in ANGLE mode (send an attitude-quat first to enter angle).
5. Probe real max setpoint rate (250 Hz is sample's used value).
6. innerloop_step system-ID (hover thrust, attitude tau/delay) -- needs a stable running race.

## Next steps
Bank the gate map (capture_track_map). Then ODOMETRY-quat orientation fix + frame dedup; system-ID;
clean angle-mode control test; wire navigator; first VQ1 completion. R8 (sim-reset MAV_CMD 31000)
still murky -- did NOT visibly reset race state (started stayed True) and did NOT re-broadcast the map.

Recordings NOT pushed (large): data/runs/20260602_000538_firstcontact_race1 (944MB, 1797 unique
frames), data/_frames/ (extracted FPV frames).
