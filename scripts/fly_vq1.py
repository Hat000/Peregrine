"""Fly a VQ1 run: wire the Navigator into Mission.run against the live sim, and record it.

This is the closed loop. Per tick: pump MAVLink -> latest DroneState + latest camera Frame ->
``Navigator.update`` (IMU + given pos/vel -> KF, vision -> KF behind the innovation gate) ->
``Mission.step`` (planner -> controller -> ControlCommand) -> send. The whole session (both
streams + the clock bridge) is recorded -- the dynamics + perception-noise spec for the RL phase.

SAFETY (read before actuating):
- The sim's flight mode FOLLOWS the setpoint type: body-rate -> ACRO, attitude-quat -> ANGLE. In
  ACRO a POSITION setpoint caused a 57 m runaway at first contact. So position/velocity "easy
  mode" is UNCONFIRMED until re-tested in angle mode. Run ``control_mode_probe.py`` FIRST to learn
  which interface actually moves the drone, then pick ``--mode`` accordingly.
- ATTITUDE mode needs a calibrated ``--hover-thrust`` (run ``innerloop_step.py``); the 0.5 default
  is a placeholder and will drift altitude.
- Start with ``--dry-run``: runs the FULL perception/estimation/planning loop against the live
  stream and prints what it WOULD command, but never arms and never sends. Validate the loop, then
  drop ``--dry-run`` to actuate. Setpoints are bounded (``--cruise``), and the script force-disarms
  on every exit / abort.

Map: connect at the HOME PAGE (we use wait_heartbeat=False), then navigate the sim
home -> waiting -> Race; the gate map broadcasts ONCE at level load to already-connected clients
and is captured live. Falls back to ``--map`` (the saved deterministic course) if not seen.

Usage:
  python scripts/fly_vq1.py --dry-run                      # safe: observe the live loop, no actuation
  python scripts/fly_vq1.py --mode position --cruise 2.5   # fly, leaning on the stabilizer
  python scripts/fly_vq1.py --mode attitude --hover-thrust 0.42 --cruise 2.0
  options: --map PATH --vision --weights PATH --takeoff-alt 1.5 --max-seconds 120 --rate 50
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from racer.contracts import ControlMode
from racer.controller import Controller
from racer.firstcontact import backend_summary, telemetry_summary
from racer.mavlink_client import MavlinkClient
from racer.mission import Mission, MissionConfig, MissionState
from racer.navigator import Navigator, NavigatorConfig, gates_from_track_records, load_track_map
from racer.planner import ReactivePlanner
from racer.recording import Recorder, session_stamp
from racer.vision.jpeg_receiver import VIDEO_PORT, JpegUdpReceiver

_MODE = {
    "position": ControlMode.POSITION,
    "velocity": ControlMode.VELOCITY,
    "attitude": ControlMode.ATTITUDE,
    "body_rate": ControlMode.BODY_RATE,
}


class _LatestFrame:
    """Thread-shared most-recent camera frame (video thread writes, control loop reads)."""

    def __init__(self):
        self._frame = None
        self._lock = threading.Lock()

    def set(self, frame):
        with self._lock:
            self._frame = frame

    def get(self):
        with self._lock:
            return self._frame


def _arm_cmd() -> int:
    from pymavlink import mavutil

    return mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM


def _load_map(client: MavlinkClient, args) -> list:
    """Prefer the live TRACK_INFO map; fall back to the saved deterministic course."""
    if client.track_gates:
        print(f"  using LIVE gate map: {len(client.track_gates)} gates (TRACK_INFO).")
        return gates_from_track_records(client.track_gates)
    gates = load_track_map(args.map)
    print(f"  using SAVED gate map: {len(gates)} gates ({args.map}).")
    return gates


def _wait_for_race(client: MavlinkClient, frames: _LatestFrame, args) -> bool:
    """Pump until a FRESH race GO. Observed mechanics (2026-06-02): clicking Race sets
    ``started=True`` and ``race_start_boot_time_ms`` to a FUTURE time = (now + ~2.8 s) = the GO;
    the countdown is ``race_start_boot - sim_boot`` ticking to 0; controlling before GO = DQ. So
    wait until ``sim_boot >= race_start_boot + start_margin``. We only accept a GENUINELY FRESH
    countdown (GO within the last ~2 s, not a stale race whose GO was minutes ago) and REFUSE to
    fly if the drone isn't reset to the origin -- the two traps that bit the early runs. Connect
    at the home page so the level-load map + the countdown are both caught."""
    print(">>> Reset for a FRESH race: home page -> waiting room -> Race.  (~3 s countdown; map loads with the level)")
    deadline = time.monotonic() + args.wait_seconds
    margin_ms = args.start_margin_s * 1000.0
    last_print = 0.0
    while time.monotonic() < deadline:
        client.pump()
        s = client.state
        rs = client.race_status
        live = s.position_ned is not None and s.sim_time_ns > 0
        now = time.monotonic()

        if args.no_wait_start and live:
            print(f"\n  (--no-wait-start) proceeding without the countdown: {telemetry_summary(client)}")
            return True

        if rs and rs["started"] and live:
            to_go_ms = rs["race_start_boot_time_ms"] - rs["sim_boot_time_ms"]   # >0 = countdown remaining
            fresh = rs["race_start_boot_time_ms"] >= 0 and to_go_ms > -2000.0   # a current GO, not a stale race
            if not fresh:
                if now - last_print >= 1.0:
                    print(f"  STALE race (GO was {-to_go_ms / 1000:.0f}s ago) -> waiting for a fresh reset (home->Race)   ",
                          end="\r", flush=True)
                    last_print = now
                time.sleep(0.01)
                continue
            pos_off = float(np.linalg.norm(s.position_ned))
            if to_go_ms <= -margin_ms:                                         # GO elapsed (+ margin)
                if pos_off > args.max_start_offset_m:
                    print(f"\n  REFUSING: at GO the drone is {pos_off:.0f} m from the origin (corrupt start). "
                          f"Reset (home->Race).", file=sys.stderr)
                    return False
                print(f"\n  GO! countdown elapsed, drone at origin (pos_off={pos_off:.2f} m). {telemetry_summary(client)}")
                return True
            if now - last_print >= 0.25:
                print(f"  countdown {to_go_ms / 1000:+.2f}s to GO  pos_off={pos_off:.1f}m "
                      f"map={len(client.track_gates or [])}   ", end="\r", flush=True)
                last_print = now
            time.sleep(0.005)
            continue

        if now - last_print >= 2.0:
            have_map = len(client.track_gates) if client.track_gates else 0
            print(f"  waiting: started={bool(rs and rs['started'])} pos={'yes' if s.position_ned is not None else 'no'} "
                  f"map={have_map} frame={'yes' if frames.get() else 'no'}   ", end="\r", flush=True)
            last_print = now
        time.sleep(0.005)
    print("\n  timed out waiting for a fresh race GO.", file=sys.stderr)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default="udp:127.0.0.1:14550")
    ap.add_argument("--video-port", type=int, default=VIDEO_PORT)
    ap.add_argument("--map", default="data/runs/track_map_20260602_114630.json",
                    help="saved gate map fallback (live TRACK_INFO is preferred if seen)")
    ap.add_argument("--mode", choices=list(_MODE), default="position")
    ap.add_argument("--cruise", type=float, default=2.5, help="planner cruise speed m/s (bounded; start slow)")
    ap.add_argument("--lookahead", type=float, default=2.0, help="carrot distance beyond the gate (m)")
    ap.add_argument("--takeoff-alt", type=float, default=1.5, help="hover altitude above the start (m)")
    ap.add_argument("--gate-radius", type=float, default=0.75, help="proximity gate-pass radius (m); MUST be <= inner half-opening (~0.75) or a wide miss false-scores a pass")
    ap.add_argument("--hover-thrust", type=float, default=0.5, help="attitude mode: calibrated hover throttle (innerloop_step ~0.489)")
    ap.add_argument("--thrust-slope", type=float, default=None, help="attitude: measured up-accel/thrust (innerloop_step ~25.9)")
    ap.add_argument("--max-accel", type=float, default=None, help="attitude: cap |desired accel| m/s^2 (bounds tilt; e.g. 3)")
    ap.add_argument("--max-pos-error", type=float, default=None, help="attitude: clamp position-error (m) fed to kp_pos (e.g. 2)")
    ap.add_argument("--kp-pos", type=float, default=1.5, help="attitude/CTBR: position gain")
    ap.add_argument("--kd-vel", type=float, default=2.0, help="attitude/CTBR: velocity gain")
    ap.add_argument("--kp-att", type=float, default=4.0, help="CTBR: attitude-error -> body-rate gain")
    ap.add_argument("--kd-att", type=float, default=0.0, help="CTBR: rate damping (omega -= kd_att*body_rate); curbs tumble")
    ap.add_argument("--max-body-rate", type=float, default=2.0, help="CTBR: body-rate clamp (rad/s)")
    ap.add_argument("--rate-sign", default="-1,1,-1",
                    help="CTBR body-rate sign (roll,pitch,yaw) for this sim's convention (measured: roll+yaw inverted)")
    # -- decoupled (plant-matched) CTBR, system-ID'd 2026-06-03 --
    ap.add_argument("--decoupled", action="store_true", help="use the plant-matched decoupled CTBR law")
    ap.add_argument("--ff-gain", type=float, default=2.6, help="decoupled: rate feedforward divisor (measured ~2.6x)")
    ap.add_argument("--odo-rate-sign", default="1,-1,1", help="decoupled: ODOMETRY rate sign vs true (pitch inverted)")
    ap.add_argument("--kp-alt", type=float, default=0.010, help="decoupled: alt-hold thrust per metre sink")
    ap.add_argument("--kd-alt", type=float, default=0.025, help="decoupled: alt-hold thrust per m/s descent")
    ap.add_argument("--alt-thrust-lo", type=float, default=0.18, help="decoupled: alt-hold thrust clamp low")
    ap.add_argument("--alt-thrust-hi", type=float, default=0.36, help="decoupled: alt-hold thrust clamp high")
    ap.add_argument("--max-speed", type=float, default=None, help="decoupled: velocity-targeting speed cap (m/s)")
    ap.add_argument("--tilt-comp", action="store_true", help="decoupled: tilt-compensate collective (thrust/cos(tilt)) so leaning forward doesn't sag altitude")
    ap.add_argument("--max-gates", type=int, default=None, help="fly only the first N gates (staged bring-up)")
    ap.add_argument("--geofence-m", type=float, default=None, help="abort if horiz dist from start exceeds (safety)")
    ap.add_argument("--max-climb-m", type=float, default=None, help="abort if |z-start| exceeds (safety)")
    ap.add_argument("--rate", type=float, default=50.0, help="control loop Hz (sets the setpoint rate)")
    ap.add_argument("--max-seconds", type=float, default=120.0, help="hard wall-clock cap on the run")
    ap.add_argument("--wait-seconds", type=float, default=180.0, help="how long to wait for an active race")
    ap.add_argument("--start-margin-s", type=float, default=0.3,
                    help="wait this long PAST the race GO before any control (avoid early-start DQ)")
    ap.add_argument("--max-start-offset-m", type=float, default=5.0,
                    help="refuse to fly if, at GO, the drone is farther than this from the origin (corrupt start)")
    ap.add_argument("--vision", action="store_true", help="run the detector -> KF vision path (needs weights)")
    ap.add_argument("--weights", default="models/gate_yolo11s_curriculum_v2.pt")
    ap.add_argument("--no-given-position", action="store_true", help="VQ2 sim: ignore given pos (vision-only)")
    ap.add_argument("--dry-run", action="store_true", help="run the full loop but NEVER arm or send (safe)")
    ap.add_argument("--no-wait-start", action="store_true", help="fly as soon as position is live (skip 'started')")
    ap.add_argument("--label", default="vq1")
    ap.add_argument("--connect-timeout", type=float, default=15.0)
    args = ap.parse_args()

    # -- detector (optional; the safe bring-up flies on the given state first) --
    detector = None
    if args.vision:
        if not Path(args.weights).exists():
            print(f"ERROR: --vision but weights missing: {args.weights}", file=sys.stderr)
            return 2
        try:
            from racer.vision.detector import GateDetector

            detector = GateDetector.load(args.weights)
            print(f"  vision ON: {args.weights}")
        except Exception as exc:
            print(f"ERROR loading detector ({exc}); install the [detector] extra.", file=sys.stderr)
            return 2
    else:
        print("  vision OFF: flying on GIVEN state (KF predict + given pos/vel). Safe bring-up order.")

    client = MavlinkClient(args.endpoint)
    print(f"connecting {args.endpoint} (wait_heartbeat=False so the map one-shot is not missed) ...")
    client.connect(wait_heartbeat=False, timeout_s=args.connect_timeout)

    frames = _LatestFrame()
    stop = threading.Event()
    session = Path("data/runs") / f"{session_stamp()}_{args.label}"
    recorder = Recorder(session)
    recorder.start()
    recorder.add_meta(endpoint=args.endpoint, mode=args.mode, cruise=args.cruise,
                      dry_run=args.dry_run, vision=bool(detector), label=args.label)
    print(f"recording -> {session}")

    # video thread: latest-frame + record (separate socket, no MAVLink contention)
    def video_loop():
        while not stop.is_set():
            try:
                with JpegUdpReceiver(port=args.video_port) as rx:
                    for fr in rx.frames(max_wait_s=5.0):
                        frames.set(fr)
                        recorder.record_frame(fr)
                        if stop.is_set():
                            break
            except Exception as exc:
                print(f"video thread: {exc}", file=sys.stderr)
            if not stop.is_set():
                time.sleep(0.5)
    vthread = threading.Thread(target=video_loop, name="video", daemon=True)
    vthread.start()

    # record raw MAVLink off the same pump (main thread) without perturbing parsing
    def on_message(msg):
        if msg.get_type() == "BAD_DATA":
            return
        buf = msg.get_msgbuf()
        if buf:
            recorder.record_mavlink(bytes(buf))
    client.on_message = on_message

    nav: Navigator | None = None
    mission: Mission | None = None
    final_state = MissionState.IDLE
    try:
        ok = _wait_for_race(client, frames, args)
        print(f"  backend: {backend_summary(client)}")
        if not ok:
            print("no live telemetry -> aborting before any actuation.", file=sys.stderr)
            return 1

        gates = _load_map(client, args)
        if args.max_gates is not None:
            gates = gates[:args.max_gates]
            print(f"  staged: flying only the first {len(gates)} gate(s).")
        nav = Navigator(gates=gates, detector=detector,
                        config=NavigatorConfig(use_vision=bool(detector),
                                               use_given_position=not args.no_given_position))
        mission = Mission(
            gates=gates,
            planner=ReactivePlanner(cruise_speed=args.cruise, lookahead_m=args.lookahead),
            controller=Controller(mode=_MODE[args.mode], hover_thrust=args.hover_thrust,
                                   thrust_slope_mps2=args.thrust_slope, max_accel_mps2=args.max_accel,
                                   max_pos_error_m=args.max_pos_error, kp_pos=args.kp_pos, kd_vel=args.kd_vel,
                                   kp_att=args.kp_att, kd_att=args.kd_att, max_body_rate_rps=args.max_body_rate,
                                   body_rate_sign=np.array([float(x) for x in args.rate_sign.split(",")]),
                                   decoupled=args.decoupled, ff_gain=args.ff_gain,
                                   odo_rate_sign=np.array([float(x) for x in args.odo_rate_sign.split(",")]),
                                   kp_alt=args.kp_alt, kd_alt=args.kd_alt,
                                   alt_thrust_lo=args.alt_thrust_lo, alt_thrust_hi=args.alt_thrust_hi,
                                   max_speed=args.max_speed, tilt_comp=args.tilt_comp),
            config=MissionConfig(takeoff_altitude_m=args.takeoff_alt, takeoff_tol_m=0.3,
                                 gate_pass_radius_m=args.gate_radius),
        )

        # -- arm (unless dry-run) --
        if not args.dry_run:
            print(f"\n[arm] mode={args.mode} cruise={args.cruise} m/s ...")
            client.last_command_ack = None
            client.arm()
            ack = client.wait_command_ack(_arm_cmd(), timeout_s=3.0)
            armed = client.wait_armed(True, timeout_s=5.0)
            print(f"  COMMAND_ACK: {ack['result_name'] if ack else 'none'}; armed={armed}")
            if not armed:
                print("  arming refused -> not flying. Try control_mode_probe.py / session_lifecycle.py.")
                return 1
        else:
            print("\n[dry-run] NOT arming; printing commands only.")

        # -- the closed loop --
        tick = 1.0 / args.rate
        st = {"next": time.monotonic(), "last_sim_t": int(client.state.sim_time_ns),
              "last_advance_wall": time.monotonic(), "n": 0, "nav": None,
              "last_print": 0.0, "prev_state": None}
        run_deadline = time.monotonic() + args.max_seconds
        max_steps = int(args.max_seconds * args.rate) + 100

        def navigator():
            # pace to the target rate while continuously draining telemetry + heartbeat
            while time.monotonic() < st["next"]:
                client.pump()
                time.sleep(0.001)
            client.pump()
            st["next"] = time.monotonic() + tick
            st["n"] += 1
            ns = nav.update(client.state, frames.get())
            st["nav"] = ns
            return ns

        class _Transport:
            def send_command(self, cmd):
                if args.dry_run:
                    if st["n"] % int(args.rate) == 1:   # ~1 Hz print in dry-run
                        print(f"  [dry] {mission.state.name} mode={cmd.mode.name} "
                              f"pos={None if cmd.position_ned is None else np.round(cmd.position_ned,1)} "
                              f"vel={None if cmd.velocity_ned is None else np.round(cmd.velocity_ned,1)} "
                              f"thr={cmd.thrust}")
                else:
                    client.send_command(cmd)

        def should_stop():
            now = time.monotonic()
            if now >= run_deadline:
                print("\n  max-seconds cap reached -> stopping.")
                return True
            # state-change + ~1 Hz status line
            if mission.state != st["prev_state"]:
                print(f"\n  [state] {mission.state.name}  gate_index={mission.gate_index}")
                st["prev_state"] = mission.state
            ns = st["nav"]
            if now - st["last_print"] >= 1.0 and ns is not None:
                print(f"  t={client.state.sim_time_ns/1e9:7.2f}s gi={mission.gate_index} "
                      f"pos=({ns.position_ned[0]:+6.1f},{ns.position_ned[1]:+6.1f},{ns.position_ned[2]:+6.1f}) "
                      f"vfix={nav.n_vision_fixes} vrej={nav.n_vision_rejected} col={len(client.collisions)}   ",
                      end="\r", flush=True)
                st["last_print"] = now
            # sim paused (race ended / off-race): sim_time stops advancing
            if int(client.state.sim_time_ns) > st["last_sim_t"]:
                st["last_sim_t"] = int(client.state.sim_time_ns)
                st["last_advance_wall"] = now
            elif now - st["last_advance_wall"] > 1.5 and not args.dry_run:
                print("\n  sim_time stalled (race ended / paused) -> stopping.")
                return True
            rs = client.race_status
            if rs and rs["finished"]:
                print("\n  RACE_STATUS finished -> stopping.")
                return True
            if any(c["threat_level"] >= 2 for c in client.collisions):
                print("\n  HARD COLLISION -> abort.")
                mission.abort()
                return True
            if ns is not None and not np.all(np.isfinite(ns.position_ned)):
                print("\n  estimator diverged (non-finite) -> abort.")
                mission.abort()
                return True
            if ns is not None:                                   # safety geofence (bounded bring-up)
                if st.get("start_pos") is None:
                    st["start_pos"] = np.asarray(ns.position_ned, dtype=np.float64).copy()
                rel = np.asarray(ns.position_ned, dtype=np.float64) - st["start_pos"]
                if args.geofence_m is not None and float(np.hypot(rel[0], rel[1])) > args.geofence_m:
                    print(f"\n  GEOFENCE: {np.hypot(rel[0],rel[1]):.0f} m from start -> abort.")
                    mission.abort()
                    return True
                if args.max_climb_m is not None and abs(float(rel[2])) > args.max_climb_m:
                    print(f"\n  ALTITUDE: {rel[2]:+.0f} m from start -> abort.")
                    mission.abort()
                    return True
            return False

        print(f"\n[run] flying up to {args.max_seconds:g}s / {len(gates)} gates "
              f"({'DRY-RUN' if args.dry_run else 'LIVE'}). Ctrl-C to stop.")
        final_state = mission.run(navigator, _Transport(), max_steps=max_steps, should_stop=should_stop)
    except KeyboardInterrupt:
        print("\nstopping (Ctrl-C) ...")
    finally:
        print("\n[safety] force-disarming ...")
        try:
            client.disarm(force=True)
            client.wait_armed(False, timeout_s=3.0)
        except Exception as exc:
            print(f"  disarm error: {exc}", file=sys.stderr)
        stop.set()
        vthread.join(timeout=6.0)
        recorder.add_meta(
            final_state=final_state.name,
            gate_index=(mission.gate_index if mission is not None else None),
            n_vision_fixes=(nav.n_vision_fixes if nav is not None else 0),
            n_vision_rejected=(nav.n_vision_rejected if nav is not None else 0),
            collisions=len(client.collisions),
            race_status=client.race_status,
        )
        recorder.close()

    # -- summary --
    print("\n==== fly_vq1 summary ====")
    print(f"  final state: {final_state.name}")
    print(f"  gate_index:  {mission.gate_index if mission is not None else '-'} / {len(mission.gates) if mission else '-'}")
    print(f"  recording:   {session}")
    rs = client.race_status
    if rs:
        print(f"  race: active_gate={rs['active_gate_index']} started={rs['started']} finished={rs['finished']}")
    if client.collisions:
        ids = sorted({c['id'] for c in client.collisions})
        print(f"  collisions: {len(client.collisions)} (ids {ids}; 1001=gate, 1002=env)")
    return 0 if final_state in (MissionState.FINISHED, MissionState.IDLE) else 1


if __name__ == "__main__":
    raise SystemExit(main())
