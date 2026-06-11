"""Open-loop body-rate STEP sysid for the sim's inner rate loop (the VQ1 control unlock).

The closed-loop CTBR runs (data/runs/*_ctbr_*) showed roll/yaw stable (sign fix [-1,1,-1]) but
pitch oscillating to +/-4.4 rad/s vs a 2.0 clamp -- but a closed loop can't separate a steady
GAIN from a transient OVERSHOOT (the command is always changing). This probe injects a FIXED,
OPEN-LOOP body rate on ONE axis at a time and measures the actual ODOMETRY rate, so the steady
gain, the per-axis sign, and the damping fall out cleanly. Offline analysis:
``handoff/shadowpc-2ndorder-resysid-2026-06-10/fit_2nd_order.py``.

Two modes (same live scaffolding -- race-wait, recording, abort guards, force-disarm):
  --mode rate   per-axis doublets (+r for ~1s, re-level, -r, re-level). Measures sign + steady
                gain + overshoot. The RAW wire value is sent on the probed axis (the sign is
                MEASURED, not assumed); the other two axes + the resting tilt are held LEVEL by a
                damped closed loop (zero rate does NOT re-level -- body-rate is a rate command).
  --mode hover  hold LEVEL and sweep collective thrust -> the clean level hover_thrust (retires
                the 0.489 measured at the -17.8 deg resting tilt).

SAFETY: bounded magnitudes, short per-phase holds, hard abort on collision / position / altitude
/ attitude-runaway / time cap, and force-disarm on every exit. The level-hold keeps the drone
near the origin + level so a multi-axis sweep does not drift into a wall. Start with --dry-run
(prints the schedule, never arms/sends), then --mode hover (gentlest), then --mode rate.

Respects the race countdown: waits for a FRESH GO (fly_vq1 mechanics) and runs UNATTENDED --
if no fresh GO shows up for --reset-after seconds it fires MAV_CMD 31000 (send_sim_reset; a
fresh ~3 s countdown when a race context exists, NO-OP from HOME), and if there is no telemetry
at all after two resets it focuses the AI-GP window and sends Enter twice (home -> waiting room
-> race; the S1.2 cold-start mechanics). Probes chain without touching the sim GUI.

Usage:
  python scripts/rate_sysid.py --dry-run
  python scripts/rate_sysid.py --mode hover  --thrust-levels 0.42,0.45,0.48,0.51 --label hsweep
  python scripts/rate_sysid.py --mode rate   --mag 0.3 --thrust 0.46 --label rate1
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

# reuse the live-session plumbing already validated against the sim
from fly_vq1 import _LatestFrame, _arm_cmd

from racer.contracts import ControlCommand, ControlMode
from racer.controller import level_hold_body_rate
from racer.frames import body_rate_from_quats
from racer.firstcontact import backend_summary, telemetry_summary
from racer.mavlink_client import MavlinkClient
from racer.recording import Recorder, session_stamp
from racer.vision.jpeg_receiver import VIDEO_PORT, JpegUdpReceiver

_AXIS_NAME = {0: "roll", 1: "pitch", 2: "yaw"}
_NAME_AXIS = {v: k for k, v in _AXIS_NAME.items()}


def _kick_sim_from_home(title_substr: str = "AI-GP", n_enter: int = 2,
                        settle_s: float = 1.5) -> bool:
    """HOME-page recovery (S1.2 mechanics): MAV_CMD 31000 is a NO-OP at the home page
    (no telemetry there), so focus the sim window via Win32 SetForegroundWindow
    (WScript AppActivate alone returns False) and send Enter twice
    (home -> waiting room -> race + countdown)."""
    import ctypes

    user32 = ctypes.windll.user32
    found: list[int] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def _cb(hwnd, _lp):
        if user32.IsWindowVisible(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            if n:
                buf = ctypes.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(hwnd, buf, n + 1)
                if title_substr.lower() in buf.value.lower():
                    found.append(hwnd)
        return True

    user32.EnumWindows(_cb, None)
    if not found:
        print(f"  home-kick: no visible window titled *{title_substr}*", file=sys.stderr)
        return False
    user32.SetForegroundWindow(found[0])
    time.sleep(0.5)
    VK_RETURN, KEYEVENTF_KEYUP = 0x0D, 0x0002
    for _ in range(n_enter):
        user32.keybd_event(VK_RETURN, 0, 0, 0)
        user32.keybd_event(VK_RETURN, 0, KEYEVENTF_KEYUP, 0)
        time.sleep(settle_s)
    return True


def _wait_fresh_go(client, frames, args) -> bool:
    """Pump until a FRESH race GO (fly_vq1 mechanics: GO within ~2 s, drone at the
    origin) with UNATTENDED recovery: every --reset-after s without a fresh GO fire
    send_sim_reset(); after two ineffective resets _kick_sim_from_home() (31000 is
    ignored both at HOME and on the off-race/paused screen the sim parks on after
    an old race -- measured 2026-06-10: stale RACE_STATUS, frozen sim_time, 22x
    31000 no-ops; one focus+2xEnter produced a fresh countdown). Never resets into
    a ticking countdown. A corrupt start (drone far from the origin at GO) waits
    for the reset timer instead of refusing."""
    deadline = time.monotonic() + args.wait_seconds
    margin_ms = args.start_margin_s * 1000.0
    auto = not args.no_auto_reset
    next_reset = time.monotonic() + (args.reset_after if auto else 1e18)
    stale_strikes = 0          # reset attempts since the last fresh countdown
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
            to_go_ms = rs["race_start_boot_time_ms"] - rs["sim_boot_time_ms"]
            fresh = rs["race_start_boot_time_ms"] >= 0 and to_go_ms > -2000.0
            if fresh:
                stale_strikes = 0
                pos_off = float(np.linalg.norm(s.position_ned))
                if to_go_ms <= -margin_ms:
                    if pos_off <= args.max_start_offset_m:
                        print(f"\n  GO! countdown elapsed, drone at origin (pos_off={pos_off:.2f} m). "
                              f"{telemetry_summary(client)}")
                        return True
                    if now - last_print >= 1.0:
                        print(f"  corrupt start (pos_off={pos_off:.0f} m) -> waiting on the reset timer   ",
                              end="\r", flush=True)
                        last_print = now
                else:
                    next_reset = max(next_reset, now + args.reset_after)   # never reset into a countdown
                    if now - last_print >= 0.25:
                        print(f"  countdown {to_go_ms / 1000:+.2f}s to GO  pos_off={pos_off:.1f}m "
                              f"map={len(client.track_gates or [])}   ", end="\r", flush=True)
                        last_print = now
                time.sleep(0.005)
                continue

        if auto and now >= next_reset:
            stale_strikes += 1
            if stale_strikes >= 3 and not args.no_home_kick:
                print("\n  31000 ineffective (off-race screen or HOME) -> focusing AI-GP + Enter x2 ...")
                _kick_sim_from_home()
                stale_strikes = 0
            else:
                print("\n  no fresh GO -> requesting sim reset (MAV_CMD 31000) ...")
                try:
                    client.send_sim_reset()
                except Exception as exc:
                    print(f"  sim reset send failed: {exc}", file=sys.stderr)
            next_reset = now + args.reset_after
        elif now - last_print >= 2.0:
            print(f"  waiting: started={bool(rs and rs['started'])} "
                  f"pos={'yes' if live else 'no'} map={len(client.track_gates or [])} "
                  f"frame={'yes' if frames.get() else 'no'}   ", end="\r", flush=True)
            last_print = now
        time.sleep(0.005)
    print("\n  timed out waiting for a fresh race GO.", file=sys.stderr)
    return False


class _Phase:
    """One segment of the schedule. ``kind`` in {hold, step}; a step injects ``value`` (raw wire
    rad/s) on ``axis`` while the other axes are held level. ``thrust`` overrides the collective."""

    __slots__ = ("name", "kind", "axis", "value", "thrust", "dur")

    def __init__(self, name, kind, dur, *, axis=None, value=None, thrust=None):
        self.name, self.kind, self.dur = name, kind, dur
        self.axis, self.value, self.thrust = axis, value, thrust


def _build_rate_schedule(args) -> list[_Phase]:
    """init level-hold, then per axis: +mag (hold) -level- -mag (hold), for each magnitude."""
    mags = [float(m) for m in args.mag.split(",") if m.strip()]
    axes = [_NAME_AXIS[a.strip()] for a in args.axes.split(",") if a.strip()]
    phases = [_Phase("init_level", "hold", args.init_hold_s)]
    for axis in axes:
        nm = _AXIS_NAME[axis]
        for mag in mags:
            phases.append(_Phase(f"{nm}+{mag:g}", "step", args.step_s, axis=axis, value=+mag))
            phases.append(_Phase(f"{nm}_lvl", "hold", args.hold_s))
            phases.append(_Phase(f"{nm}-{mag:g}", "step", args.step_s, axis=axis, value=-mag))
            phases.append(_Phase(f"{nm}_lvl", "hold", args.hold_s))
    return phases


def _build_anomaly_schedule(args) -> list[_Phase]:
    """Payload-2 probe (anomaly boundary): init level-hold, CLIMB for altitude headroom
    (the drone will tumble and fall -- the floor collision is the bounded end of the
    attempt), then ONE sustained open-loop step on ONE axis with ZERO command on the
    other axes (no doublet, no re-level; the level-hold math is meaningless past 90
    deg anyway). Uses the FIRST --axes axis and FIRST --mag magnitude (signed). The
    tilt abort is OFF by default in this mode; collision/position/altitude/time stay."""
    mags = [float(m) for m in args.mag.split(",") if m.strip()]
    axes = [_NAME_AXIS[a.strip()] for a in args.axes.split(",") if a.strip()]
    vec = np.zeros(3)
    for axis, mag in zip(axes, mags):
        vec[axis] = mag                      # paired element-wise: --axes roll,yaw --mag 3.14,3.14
    name = "anom_" + "_".join(f"{_AXIS_NAME[a]}{m:+g}" for a, m in zip(axes, mags))
    return [
        _Phase("init_level", "hold", args.init_hold_s),
        _Phase("climb", "hold", args.climb_s, thrust=args.climb_thrust),
        _Phase(name[:14], "astep", args.step_s, axis=axes[0], value=vec),
    ]


def _build_hover_schedule(args) -> list[_Phase]:
    """init level-hold, then a level-hold at each thrust level (measure the vertical drift)."""
    levels = [float(t) for t in args.thrust_levels.split(",") if t.strip()]
    phases = [_Phase("init_level", "hold", args.init_hold_s)]
    for thr in levels:
        phases.append(_Phase(f"thr_{thr:g}", "hold", args.dwell_s, thrust=thr))
    return phases


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default="udp:127.0.0.1:14550")
    ap.add_argument("--video-port", type=int, default=VIDEO_PORT)
    ap.add_argument("--mode", choices=["rate", "hover", "anomaly"], default="rate")
    # rate-step schedule
    ap.add_argument("--mag", default="0.3", help="rad/s step magnitude(s), comma list (e.g. 0.2,0.35)")
    ap.add_argument("--axes", default="pitch,roll,yaw", help="which axes to probe, in order")
    ap.add_argument("--step-s", type=float, default=0.8, help="hold each rate step this long")
    ap.add_argument("--hold-s", type=float, default=1.0, help="re-level hold between steps")
    ap.add_argument("--init-hold-s", type=float, default=2.0, help="initial level-hold (kills the resting tilt)")
    # hover sweep
    ap.add_argument("--thrust-levels", default="0.22,0.25,0.28", help="hover mode: thrust sweep (~hover 0.25)")
    ap.add_argument("--dwell-s", type=float, default=0.9, help="hover mode: hold per thrust level")
    # anomaly probe
    ap.add_argument("--climb-s", type=float, default=2.5, help="anomaly mode: climb-phase duration")
    ap.add_argument("--climb-thrust", type=float, default=0.33, help="anomaly mode: climb-phase thrust")
    # control
    ap.add_argument("--thrust", type=float, default=0.26, help="base collective thrust (~hover)")
    ap.add_argument("--alt-hold", action="store_true", help="hold altitude via velocity-damped thrust (keeps z bounded)")
    ap.add_argument("--kp-alt", type=float, default=0.010, help="alt-hold: thrust per metre of sink (NED z error)")
    ap.add_argument("--kd-alt", type=float, default=0.025, help="alt-hold: thrust per m/s descent (vz damping)")
    ap.add_argument("--alt-thr-lo", type=float, default=0.18, help="alt-hold thrust clamp low")
    ap.add_argument("--alt-thr-hi", type=float, default=0.34, help="alt-hold thrust clamp high")
    ap.add_argument("--kp-hold", type=float, default=1.0, help="level-hold attitude gain (low BW = delay-tolerant)")
    ap.add_argument("--kd-hold", type=float, default=1.0, help="level-hold rate damping")
    # velocity damping / station-keeping (2026-06-10: the pure attitude hold ratchets horizontal
    # velocity -- each step's tilt adds a kick that only drag (~5 s) removes -> 18 m drift abort.
    # Mapping measured from run 20260610_223428: a_body ~= -g * reported tilt (corr -0.95..-0.99),
    # so a damping tilt target is tilt_des = +(c/g) * (v_body - v_des) subtracted from the
    # reported attitude fed to the hold. Applies to the NON-probed axes only during steps.)
    ap.add_argument("--vel-damp", type=float, default=1.0,
                    help="hold-phase velocity damping c (1/s): decel = c * v_horiz. 0 = off")
    ap.add_argument("--vel-damp-max-deg", type=float, default=12.0,
                    help="clamp on the damping tilt target")
    ap.add_argument("--pos-pull", type=float, default=0.3,
                    help="station-keep: velocity setpoint = -pos_pull * offset_from_origin (1/s)")
    ap.add_argument("--pos-pull-vmax", type=float, default=2.0,
                    help="clamp on the station-keep velocity setpoint (m/s)")
    ap.add_argument("--ff-gain", type=float, default=2.7, help="divide hold cmd by the measured rate scaling (~2.7x)")
    ap.add_argument("--rate-ema", type=float, default=0.5, help="EMA weight on the finite-diff rate (lower=smoother)")
    ap.add_argument("--max-hold-rate", type=float, default=1.5, help="clamp on the level-hold body rate")
    ap.add_argument("--rate-sign", default="-1,1,-1", help="sim body-rate sign for the HOLD (measured)")
    ap.add_argument("--rate", type=float, default=50.0, help="control/log loop Hz")
    # safety bounds / abort
    ap.add_argument("--max-offset-m", type=float, default=18.0, help="abort if horiz dist from origin exceeds")
    ap.add_argument("--max-alt-m", type=float, default=8.0, help="abort if |z| exceeds (climb/sink)")
    ap.add_argument("--max-tilt-deg", type=float, default=70.0, help="abort if |roll|/|pitch| exceeds")
    ap.add_argument("--max-seconds", type=float, default=40.0, help="hard wall-clock cap")
    # race wait (unattended: auto-reset + HOME kick; see _wait_fresh_go)
    ap.add_argument("--wait-seconds", type=float, default=180.0)
    ap.add_argument("--start-margin-s", type=float, default=0.3)
    ap.add_argument("--max-start-offset-m", type=float, default=5.0)
    ap.add_argument("--no-wait-start", action="store_true")
    ap.add_argument("--reset-after", type=float, default=8.0,
                    help="fire MAV_CMD 31000 whenever no fresh GO for this many seconds")
    ap.add_argument("--no-auto-reset", action="store_true",
                    help="never send MAV_CMD 31000; wait for a manual race start")
    ap.add_argument("--no-home-kick", action="store_true",
                    help="never focus the sim window / send Enter (HOME-page recovery)")
    ap.add_argument("--dry-run", action="store_true", help="print the schedule; never arm or send")
    ap.add_argument("--label", default="ratesysid")
    ap.add_argument("--connect-timeout", type=float, default=15.0)
    args = ap.parse_args()

    sign = np.array([float(x) for x in args.rate_sign.split(",")], dtype=np.float64)
    schedule = {"rate": _build_rate_schedule, "hover": _build_hover_schedule,
                "anomaly": _build_anomaly_schedule}[args.mode](args)
    if args.mode == "anomaly" and args.max_tilt_deg == 70.0:
        args.max_tilt_deg = 1e9          # the tumble IS the measurement
        print("[anomaly] tilt abort OFF (collision/position/altitude/time aborts stay)")
    total_s = sum(p.dur for p in schedule)
    print(f"== rate_sysid mode={args.mode} ==  {len(schedule)} phases, ~{total_s:.1f}s actuation")
    for p in schedule:
        if p.kind == "step":
            extra = f" step {_AXIS_NAME[p.axis]}={p.value:+.2f}rad/s"
        elif p.kind == "astep":
            extra = " step " + ",".join(f"{v:+.2f}" for v in np.atleast_1d(p.value))
        else:
            extra = f" thrust={p.thrust:.2f}" if p.thrust is not None else ""
        print(f"   {p.name:14s} {p.kind:5s} {p.dur:4.1f}s{extra}")
    if args.dry_run:
        print("\n[dry-run] not connecting/arming. Schedule above is what WOULD be flown.")
        return 0

    client = MavlinkClient(args.endpoint)
    print(f"\nconnecting {args.endpoint} (wait_heartbeat=False) ...")
    client.connect(wait_heartbeat=False, timeout_s=args.connect_timeout)

    frames = _LatestFrame()
    stop = threading.Event()
    session = Path("data/runs") / f"{session_stamp()}_{args.label}"
    recorder = Recorder(session)
    recorder.start()
    recorder.add_meta(endpoint=args.endpoint, probe="rate_sysid", mode=args.mode,
                      thrust=args.thrust, mag=args.mag, axes=args.axes, label=args.label)
    print(f"recording -> {session}")

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

    def on_message(msg):
        if msg.get_type() == "BAD_DATA":
            return
        buf = msg.get_msgbuf()
        if buf:
            recorder.record_mavlink(bytes(buf))
    client.on_message = on_message

    cmd_log = open(session / "commands.jsonl", "w", encoding="utf-8")
    armed = False
    aborted_reason = None
    n_rows = 0
    try:
        if not _wait_fresh_go(client, frames, args):
            print("no fresh race GO -> aborting before any actuation.", file=sys.stderr)
            return 1
        print(f"  backend: {backend_summary(client)}")
        print(f"  {telemetry_summary(client)}")

        print(f"\n[arm] arming for {args.mode} sysid ...")
        client.last_command_ack = None
        client.arm()
        client.wait_command_ack(_arm_cmd(), timeout_s=3.0)
        armed = client.wait_armed(True, timeout_s=5.0)
        print(f"  armed={armed}")
        if not armed:
            print("  arming refused -> not flying.", file=sys.stderr)
            return 1

        yaw_hold = float(client.state.yaw)        # hold the start heading throughout
        origin = np.asarray(client.state.position_ned, dtype=np.float64).copy()
        print(f"  yaw_hold={np.degrees(yaw_hold):+.1f} deg  origin={np.round(origin,2)}")
        # trusted body rate = finite-diff of the ODOMETRY quaternion (the ODOMETRY angular_rate is
        # sign-inverted vs the true attitude derivative -> using it for damping is anti-damping).
        prev_q = (np.asarray(client.state.orientation_ned_wxyz, dtype=np.float64).copy()
                  if client.state.orientation_ned_wxyz is not None else None)
        prev_t = int(client.state.sim_time_ns)
        trusted_rate = np.zeros(3)

        tick = 1.0 / args.rate
        t0 = time.monotonic()
        run_deadline = t0 + args.max_seconds
        last_print = 0.0

        def abort_check() -> str | None:
            s = client.state
            if any(c["threat_level"] >= 2 for c in client.collisions):
                return "hard collision"
            if not np.all(np.isfinite(s.position_ned if s.position_ned is not None else [0.0])):
                return "non-finite state"
            rel = np.asarray(s.position_ned, dtype=np.float64) - origin
            if float(np.hypot(rel[0], rel[1])) > args.max_offset_m:
                return f"drifted {np.hypot(rel[0], rel[1]):.0f}m from origin"
            if abs(float(rel[2])) > args.max_alt_m:
                return f"altitude {rel[2]:+.0f}m"
            if max(abs(s.roll), abs(s.pitch)) > np.radians(args.max_tilt_deg):
                return f"attitude runaway ({np.degrees(max(abs(s.roll), abs(s.pitch))):.0f} deg)"
            return None

        origin_z = float(origin[2])

        def hold_thrust(s) -> float:
            """Velocity-damped altitude hold around base thrust (the controller's alt channel).
            NED: z+ = down, vz+ = descending; sink/descent -> more thrust."""
            z = float(s.position_ned[2]) if s.position_ned is not None else origin_z
            vz = float(s.velocity_ned[2]) if s.velocity_ned is not None else 0.0
            thr = args.thrust + args.kp_alt * (z - origin_z) + args.kd_alt * vz
            return float(np.clip(thr, args.alt_thr_lo, args.alt_thr_hi))

        print(f"\n[run] {args.mode} sysid, ~{total_s:.0f}s. Ctrl-C to stop.")
        for phase in schedule:
            phase_end = time.monotonic() + phase.dur
            while time.monotonic() < phase_end:
                client.pump()
                s = client.state
                if s.orientation_ned_wxyz is not None and int(s.sim_time_ns) > prev_t and prev_q is not None:
                    new_rate = body_rate_from_quats(prev_q, s.orientation_ned_wxyz,
                                                    (int(s.sim_time_ns) - prev_t) / 1e9)
                    trusted_rate = args.rate_ema * new_rate + (1.0 - args.rate_ema) * trusted_rate
                    prev_q = np.asarray(s.orientation_ned_wxyz, dtype=np.float64).copy()
                    prev_t = int(s.sim_time_ns)
                roll_des = pitch_des = 0.0
                if args.vel_damp > 0.0 and s.velocity_ned is not None and s.position_ned is not None:
                    cpsi, spsi = np.cos(s.yaw), np.sin(s.yaw)
                    rel = np.asarray(s.position_ned, dtype=np.float64) - origin
                    v_des = -args.pos_pull * rel[:2]                  # world-frame pull to origin
                    nv = float(np.hypot(v_des[0], v_des[1]))
                    if nv > args.pos_pull_vmax:
                        v_des *= args.pos_pull_vmax / nv
                    dv = np.asarray(s.velocity_ned, dtype=np.float64)[:2] - v_des
                    dv_bx = cpsi * dv[0] + spsi * dv[1]               # body-horizontal frame
                    dv_by = -spsi * dv[0] + cpsi * dv[1]
                    tmax = np.radians(args.vel_damp_max_deg)
                    kva = args.vel_damp / 9.80665                     # tilt per (m/s): a_b = -g*tilt
                    pitch_des = float(np.clip(kva * dv_bx, -tmax, tmax))
                    roll_des = float(np.clip(kva * dv_by, -tmax, tmax))
                omega = level_hold_body_rate(
                    s.roll - roll_des, s.pitch - pitch_des, s.yaw, yaw_hold, trusted_rate,
                    kp=args.kp_hold, kd=args.kd_hold, body_rate_sign=sign,
                    max_rate=args.max_hold_rate, ff_gain=args.ff_gain,
                )
                if phase.kind == "step":
                    omega = omega.copy()
                    omega[phase.axis] = phase.value          # RAW open-loop override (measures sign)
                elif phase.kind == "astep":                  # anomaly probe: pure open-loop vector
                    omega = np.asarray(phase.value, dtype=np.float64).copy()
                if phase.thrust is not None:
                    thr = phase.thrust                       # hover-sweep: explicit per-phase thrust
                elif args.alt_hold:
                    thr = hold_thrust(s)                     # rate mode: hold altitude
                else:
                    thr = args.thrust
                cmd = ControlCommand(mode=ControlMode.BODY_RATE, sim_time_ns=int(s.sim_time_ns),
                                     body_rate=omega, thrust=float(thr))
                client.send_command(cmd)

                mr = np.asarray(s.angular_rate_body, dtype=np.float64)
                pos = np.asarray(s.position_ned if s.position_ned is not None else [np.nan]*3, dtype=np.float64)
                vel = np.asarray(s.velocity_ned if s.velocity_ned is not None else [np.nan]*3, dtype=np.float64)
                # ACTUATOR_OUTPUT_STATUS = the parser-INDEPENDENT thrust witness (Task-3 discriminator:
                # do the motors TRACK our commanded collective, or stay pinned = sim auto-thrust?).
                act = (None if client.actuator_outputs is None
                       else [round(float(x), 5) for x in client.actuator_outputs["motors"]])
                cmd_log.write(json.dumps({
                    "t": round(time.monotonic() - t0, 4), "sim_time_ns": int(s.sim_time_ns),
                    "phase": phase.name, "kind": phase.kind,
                    "axis": (-1 if phase.axis is None else int(phase.axis)),
                    "cmd": [round(float(v), 5) for v in omega], "thrust": float(thr),
                    "actuators": act,                                     # motor outputs [0..1]
                    "meas_rate": [round(float(v), 5) for v in mr],        # ODOMETRY (sign-suspect)
                    "true_rate": [round(float(v), 5) for v in trusted_rate],  # finite-diff (sign-correct)
                    "rpy": [round(float(s.roll), 5), round(float(s.pitch), 5), round(float(s.yaw), 5)],
                    "pos": [round(float(v), 3) for v in pos],
                    "vel": [round(float(v), 4) for v in vel],             # FIXED world velocity (c3b5a8e)
                }) + "\n")
                n_rows += 1

                reason = abort_check()
                if reason is not None or time.monotonic() >= run_deadline:
                    aborted_reason = reason or "max-seconds cap"
                    break
                now = time.monotonic()
                if now - last_print >= 0.5:
                    mot = (np.mean(act) if act is not None else float("nan"))
                    # Task-3 discriminator line: commanded collective vs motor witness vs world vz.
                    print(f"  {phase.name:12s} t={now - t0:4.1f}s cmd_thr={thr:.3f} "
                          f"mot~{mot:.3f} [{','.join(f'{a:.2f}' for a in (act or []))}] "
                          f"vz={vel[2]:+5.2f} z={pos[2]:+5.2f} "
                          f"rpy=({np.degrees(s.roll):+4.0f},{np.degrees(s.pitch):+4.0f},"
                          f"{np.degrees(s.yaw):+4.0f})   ", end="\r", flush=True)
                    last_print = now
                time.sleep(tick)
            if aborted_reason is not None:
                print(f"\n  *** ABORT: {aborted_reason} -> disarming.")
                break
    except KeyboardInterrupt:
        print("\nstopping (Ctrl-C) ...")
        aborted_reason = "ctrl-c"
    finally:
        print("\n[safety] force-disarming ...")
        try:
            client.disarm(force=True)
            client.wait_armed(False, timeout_s=3.0)
        except Exception as exc:
            print(f"  disarm error: {exc}", file=sys.stderr)
        stop.set()
        vthread.join(timeout=6.0)
        cmd_log.close()
        recorder.add_meta(rows=n_rows, aborted=aborted_reason, yaw_hold_deg=float(np.degrees(yaw_hold))
                          if armed else None)
        recorder.close()

    print("\n==== rate_sysid summary ====")
    print(f"  mode={args.mode}  rows={n_rows}  aborted={aborted_reason}")
    print(f"  recording: {session}")
    print(f"  analyze:   handoff/shadowpc-2ndorder-resysid-2026-06-10/fit_2nd_order.py (sweep loader)")
    return 0 if aborted_reason in (None, "max-seconds cap") else 1


if __name__ == "__main__":
    raise SystemExit(main())
