"""scripts/twin_launch_phase_sweep.py -- prove the takeoff->RUN launch ramp tames the start
transient ROBUSTLY (phase- and rate-invariant), offline on the SIM-FAITHFUL plant. This is the
offline evidence for the sim-build-1.0.3364 start-transient fix (MEMORY.md issue (1)).

WHY a new harness (twin_fly_course.fly is not enough): that runner steps the plant ONE dt per
command -- control reads a FRESH pose every tick and the loop has no transport delay, so it can
never reproduce the live dice-roll (the 6/6 "won the old timing"). The live failure is a PHASE
effect: the control tick, the ~75-97 Hz pose stream, and the ~40 ms loop transport delay beat
against each other, so WHEN the takeoff->RUN handoff fires decides whether the one-tick attitude
STEP saturates the 8 rad/s body-rate clamp. To see (and kill) that we co-simulate three
independent clocks:

  * PHYSICS   -- the CtbrPlant integrated at a fine ``dt_phys`` (ZOH on the actuated command).
  * POSE      -- the controller only SEES a pose sample at ``pose_rate`` Hz, delayed by
                 ``sensor_latency`` (so a fast control loop re-issues on a STALE pose -- exactly
                 the live "control far above the pose stream" note).
  * CONTROL   -- ticks at ``control_rate`` Hz with a PHASE OFFSET vs the pose clock; each command
                 is actuated after ``actuation_latency`` (sensor+actuation ~= the 40 ms loop delay).

The metric that matters is the PEAK commanded body rate (and the peak REALIZED roll rate + tilt)
in the launch window: the broken build saturates the clamp (>= max_body_rate, ~8 rad/s) at SOME
phases -> tumble. The fix must hold the peak well under the clamp at EVERY phase and rate while
still threading 6/6.

Usage:
  python scripts/twin_launch_phase_sweep.py                 # the headline sweep (ramp off vs on)
  python scripts/twin_launch_phase_sweep.py --ramp-s 0.6    # try a ramp duration
  python scripts/twin_launch_phase_sweep.py --full-course   # also score 6/6 + worst plane-miss
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # twin_fly_course (sibling)

import numpy as np

from racer.contracts import ControlCommand, ControlMode
from racer.mission import Mission, MissionConfig, MissionState
from racer.navigator import Navigator, NavigatorConfig, load_track_map
from racer.planner import ReactivePlanner
from racer.twin import CtbrPlant
from racer.twin_fit import faithful_config
from twin_fly_course import _FAITHFUL_SIGNS, _MAP, _level_quat_wxyz, gate_plane_miss, make_controller
from twin_tune import FAITHFUL_TUNED_GAINS, FAITHFUL_TUNED_PLANNER

MAX_BODY_RATE = 8.0       # the controller's body-rate clamp (rad/s) -- saturating it = tumble risk
LAUNCH_WINDOW_S = 3.0     # measure the transient peak over the first N s of RUN
TUMBLE_TILT_DEG = 70.0    # peak tilt above this in the window = a tumble (not a controlled lean)


def simulate(*, control_rate: float, launch_ramp_s: float, pose_rate: float = 85.0,
             sensor_latency_s: float = 0.020, actuation_latency_s: float = 0.020,
             phase_frac: float = 0.0, spawn_yaw_offset_deg: float = 0.0,
             dt_phys: float = 0.002, n_gates: int = 6,
             max_s: float = 40.0, score_course: bool = False) -> dict:
    """Co-simulate the REAL Navigator/Mission/Planner/Controller on the faithful plant with
    decoupled clocks (see the module docstring). Returns the launch-window transient metrics and,
    if ``score_course``, the 6/6 finish + per-gate in-plane miss."""
    gates = load_track_map(_MAP, corner_to_center=True)[:n_gates]
    nav = Navigator(gates=gates, detector=None, config=NavigatorConfig(use_vision=False))
    ctrl = make_controller(signs=_FAITHFUL_SIGNS, **FAITHFUL_TUNED_GAINS)
    plan = ReactivePlanner(yaw_mode="course", **FAITHFUL_TUNED_PLANNER)
    mission = Mission(gates=gates, planner=plan, controller=ctrl,
                      config=MissionConfig(takeoff_altitude_m=1.5, gate_pass_radius_m=0.75,
                                           launch_ramp_s=launch_ramp_s))
    mission.start()
    # Spawn heading. The course faces -X (yaw=pi); the planner builds R_des at exactly atan2(0,-1)=pi.
    # A spawn yaw OFFSET (the live drone is not perfectly aligned at GO) is the memory's prime suspect
    # for the roll spike: a world -X forward-accel demand projects partly onto BODY ROLL when the body
    # is yawed off -X, and the yaw error itself couples into the rotvec roll command. Sweeping it shows
    # whether the step saturates the clamp on ROLL (it does) and that the ramp kills it.
    plant = CtbrPlant(faithful_config(), position_ned=[0.0, 0.0, 0.0],
                      q_wxyz=_level_quat_wxyz(np.pi + np.deg2rad(spawn_yaw_offset_deg)))

    ctrl_period, pose_period = 1.0 / control_rate, 1.0 / pose_rate
    next_ctrl = phase_frac * ctrl_period                          # the tick PHASE vs the pose clock
    next_pose = 0.0
    hover = float(faithful_config().hover_thrust)
    cur_cmd = ControlCommand(mode=ControlMode.BODY_RATE, body_rate=np.zeros(3), thrust=hover)
    pose_buf: list[tuple[float, object]] = []                     # (available_time, DroneState)
    cmd_buf: list[tuple[float, ControlCommand]] = []              # (actuate_time, command)
    latest_pose = plant.state()

    traj: list[np.ndarray] = []
    peak_omega = 0.0           # peak commanded |body_rate| in the launch window
    peak_roll_cmd = 0.0        # peak commanded ROLL rate magnitude (|body_rate[0]|)
    peak_roll_realized = 0.0   # peak realized roll rate (plant.omega[0])
    peak_tilt_deg = 0.0
    run_start_t: float | None = None
    diverged = False

    n_steps = int(max_s / dt_phys)
    for step in range(n_steps):
        t = step * dt_phys
        # -- pose stream: sample the plant at pose_rate, available after sensor_latency --
        while t >= next_pose - 1e-12:
            pose_buf.append((next_pose + sensor_latency_s, plant.state()))
            next_pose += pose_period
        while pose_buf and pose_buf[0][0] <= t + 1e-12:
            latest_pose = pose_buf.pop(0)[1]
        # -- control ticks at control_rate + phase; command actuated after actuation_latency --
        while t >= next_ctrl - 1e-12:
            ns = nav.update(latest_pose, None)
            cmd = mission.step(ns)
            if mission.state == MissionState.RUN and run_start_t is None:
                run_start_t = t
            if (run_start_t is not None and t - run_start_t <= LAUNCH_WINDOW_S
                    and cmd.body_rate is not None):
                peak_omega = max(peak_omega, float(np.linalg.norm(cmd.body_rate)))
                peak_roll_cmd = max(peak_roll_cmd, abs(float(cmd.body_rate[0])))
            cmd_buf.append((next_ctrl + actuation_latency_s, cmd))
            next_ctrl += ctrl_period
        while cmd_buf and cmd_buf[0][0] <= t + 1e-12:
            cur_cmd = cmd_buf.pop(0)[1]
        # -- integrate one fine physics step under the held (actuated) command --
        plant.step(cur_cmd, dt_phys)
        traj.append(plant.pos.copy())
        if run_start_t is not None and t - run_start_t <= LAUNCH_WINDOW_S:
            peak_roll_realized = max(peak_roll_realized, abs(float(plant.omega[0])))
            st = plant.state()
            tilt = np.degrees(np.arccos(np.clip(np.cos(st.roll) * np.cos(st.pitch), -1.0, 1.0)))
            peak_tilt_deg = max(peak_tilt_deg, float(tilt))
        if not np.all(np.isfinite(plant.pos)) or float(np.linalg.norm(plant.pos)) > 500.0:
            diverged = True
            break
        if mission.state == MissionState.FINISHED and not score_course:
            break

    tumbled = diverged or peak_tilt_deg > TUMBLE_TILT_DEG
    out = {"control_rate": control_rate, "phase_frac": phase_frac, "launch_ramp_s": launch_ramp_s,
           "spawn_yaw_offset_deg": spawn_yaw_offset_deg,
           "peak_omega": peak_omega, "peak_roll_cmd": peak_roll_cmd,
           "peak_roll_realized": peak_roll_realized,
           "peak_tilt_deg": peak_tilt_deg, "saturated": peak_omega >= MAX_BODY_RATE - 1e-6,
           "tumbled": tumbled, "final": mission.state.name, "gate_index": mission.gate_index}
    if score_course:
        P = np.asarray(traj)
        pm = [gate_plane_miss(P, g) for g in gates]
        out["plane_miss"] = pm
        out["finished"] = bool(mission.state == MissionState.FINISHED and mission.gate_index == n_gates)
        crossed = [m for m in pm if m is not None]
        out["worst_miss"] = float(max(crossed)) if crossed else float("nan")
    return out


def _row(r: dict) -> str:
    flags = []
    if r["saturated"]:
        flags.append("SAT")
    if r["tumbled"]:
        flags.append("TUMBLE")
    tag = (" ".join(flags)) if flags else "ok"
    extra = ""
    if "finished" in r:
        fin = "6/6" if r["finished"] else f"{r['gate_index']}/6"
        extra = f"  {fin} worst={r['worst_miss']:.2f}m"
    return (f"  rate={r['control_rate']:>5.0f}Hz phase={r['phase_frac']:.2f}  "
            f"peak|omega|={r['peak_omega']:5.2f}  rollcmd={r['peak_roll_cmd']:5.2f}  "
            f"roll_realized={r['peak_roll_realized']:5.2f}  "
            f"tilt={r['peak_tilt_deg']:5.1f}deg  [{tag}]{extra}")


def _yaw_row(r: dict) -> str:
    flags = []
    if r["saturated"]:
        flags.append("SAT")
    if r["tumbled"]:
        flags.append("TUMBLE")
    tag = (" ".join(flags)) if flags else "ok"
    return (f"  yaw_off={r['spawn_yaw_offset_deg']:+5.0f}deg  peak|omega|={r['peak_omega']:5.2f}  "
            f"rollcmd={r['peak_roll_cmd']:5.2f}  roll_realized={r['peak_roll_realized']:5.2f}  "
            f"tilt={r['peak_tilt_deg']:5.1f}deg  [{tag}]")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ramp-s", type=float, default=0.6, help="launch ramp duration to test (s)")
    ap.add_argument("--rates", default="50,100,200,250", help="control rates (Hz), comma-separated")
    ap.add_argument("--phases", default="0,0.25,0.5,0.75", help="tick phase fractions, comma-separated")
    ap.add_argument("--full-course", action="store_true", help="also fly the whole course -> 6/6 + worst miss")
    ap.add_argument("--yaw-offsets", default="-30,-20,-10,0,10,20,30",
                    help="spawn-yaw offsets (deg) for the yaw-coupling sweep (the roll-spike suspect)")
    args = ap.parse_args()

    rates = [float(x) for x in args.rates.split(",")]
    phases = [float(x) for x in args.phases.split(",")]
    yaw_offsets = [float(x) for x in args.yaw_offsets.split(",")]

    # -- yaw-coupling sweep: the memory's prime suspect for the ROLL spike (yaw~=-180 launch geometry).
    # At 100 Hz, vary the spawn heading off the -X course axis; ramp off vs on. A spawn offset turns the
    # forward-lean step into a ROLL command -> the saturation + tumble the live build shows.
    print("=== YAW-COUPLING (roll-spike suspect): spawn heading off the -X course axis, 100 Hz ===\n")
    for ramp_s, title in ((0.0, "RAMP OFF (legacy step)"), (args.ramp_s, f"RAMP ON ({args.ramp_s:g}s)")):
        print(f"-- {title} --")
        for yo in yaw_offsets:
            r = simulate(control_rate=100.0, launch_ramp_s=ramp_s, phase_frac=0.0,
                         spawn_yaw_offset_deg=yo)
            print(_yaw_row(r))
        print()
    print("Co-sim: faithful plant + 3 decoupled clocks (physics 500 Hz, pose 85 Hz +20 ms, "
          f"control N Hz +20 ms, phase offset). Body-rate clamp = {MAX_BODY_RATE:g} rad/s.\n"
          f"Launch-window peak over the first {LAUNCH_WINDOW_S:g}s of RUN.\n")

    for ramp_s, title in ((0.0, "RAMP OFF (legacy step -- the broken 1.0.3364 behaviour)"),
                          (args.ramp_s, f"RAMP ON  ({args.ramp_s:g}s launch ramp -- the fix)")):
        print(f"== {title} ==")
        worst_peak, n_sat, n_tumble = 0.0, 0, 0
        for rate in rates:
            for ph in phases:
                r = simulate(control_rate=rate, launch_ramp_s=ramp_s, phase_frac=ph,
                             score_course=args.full_course)
                print(_row(r))
                worst_peak = max(worst_peak, r["peak_omega"])
                n_sat += int(r["saturated"])
                n_tumble += int(r["tumbled"])
        print(f"  -> worst peak|omega| over the sweep = {worst_peak:.2f} rad/s; "
              f"{n_sat} saturated, {n_tumble} tumbled (of {len(rates) * len(phases)}).\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
