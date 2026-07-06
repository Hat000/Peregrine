"""scripts/mapping_mission.py -- HARDCODED open-loop MAPPING FLIGHT for VQ2.

Fly a slow, scripted, OPEN-LOOP trajectory through the warehouse arena while the onboard
camera records, so the footage feeds an OFFLINE 3D reconstruction (MASt3R-SfM). There is NO
vision in the control loop and NO gate seeking -- attitude is held level (or at a small
scripted tilt) via an IMU AHRS, thrust is damped on the IMU vertical-velocity washout, and
yaw is slewed at a scripted rate. The point is broad, smooth camera coverage (a takeoff, a
few in-place 360 panoramas, and a couple of short translation legs at ~120 deg apart), not
racing.

WHY THIS IS SAFE / SIGN-CORRECT (no plant re-discovery -- all facts lifted from the flown,
eyes-confirmed vq2_case_c path, 2026-07-05):
  * Control = CTBR: ARM(400) then SET_ATTITUDE_TARGET in BODY-RATE mode at ~100 Hz
    (racer.contracts.ControlMode.BODY_RATE). We reuse the EXACT client construction the
    racing path uses -- ``MavlinkClient(cmd_rate_scale=0.4, gyro_sign=(-1,-1,-1))`` from the
    vq2_case_c deploy profile -- so the ~2.5x command->realized rate gain is compensated at
    the wire and the live-wire gyro convention is corrected before the AHRS.
  * Attitude source = the case-C AHRS (racer.ahrs.ahrs_adapter.AHRSAttitudeSource): VQ2
    BLOCKS ODOMETRY/ATTITUDE, so ``DroneState.roll/pitch/yaw`` are dead. The AHRS ingests raw
    HIGHRES_IMU (accel + the gyro_sign-corrected gyro) on the master sim clock and yields the
    TRUE FRD->NED attitude. It auto-seeds gravity-aligned on the first sample.
  * Attitude->body-rate law = ``racer.controller.level_hold_body_rate`` fed the AHRS's TRUE
    euler with ``body_rate_sign=(1,1,1)`` and ``ff_gain=1.0``. This is byte-equivalent to the
    flown seeker's net law: the case-C Navigator emits an ODO-conjugated NavState
    (nav.roll=-true_roll, nav.yaw=-true_yaw) and the controller's odo_att_sign=[-1,1,1] +
    the seeker's true_attitude_from_ahrs re-negation rebuild R_cur = R_world_from_body(
    +true_roll,+true_pitch,+true_yaw) before the SAME rotvec * body_rate_sign(1,1,1). Feeding
    ``level_hold_body_rate`` the TRUE euler reproduces that R_cur exactly, so the wire sign is
    the eyes-confirmed one -- no new sign assumption is introduced.
  * Vertical = thrust damper ``thrust = hover + kd*(0 - vz_imu)`` on the parallel IMU-only
    washout (racer.vertical_estimator.VerticalEstimator.vz_imu). vz_imu is the pure A24
    recurrence, never touched by vision, structurally bounded. We integrate it ONLY while the
    race is live (RACE_STATUS started==True) and reset it cleanly on a sim clock jump BACKWARD
    (the sim IMU-trap: a frozen canned tuple + a time_usec that resets at race restart).
  * Hover thrust ~= 0.2656; we never rail thrust (the sim mixer couples thrust<->rates near
    the rails), keep rate commands moderate, and clamp the composed body rate.

MODES:
  --dry-run  Print the full segment schedule (name, duration, commanded rates/thrust deltas)
             and exit. No network, no imports of the sim stack beyond the pure scheduler.
  --go       Attach to the live sim (udp:127.0.0.1:14550 by default), WAIT for RACE_STATUS
             started==True (a separate pilot session GOes the race via vq2ctl -- this script
             NEVER sends GO/arm-the-race itself), then execute the mission, log a per-tick
             JSONL, keep streaming hover 5 s after SETTLE, and exit cleanly.

A fresh session dir ``data/runs/<ts>_mapping_m1`` gets video.bin + video_index.jsonl +
mavlink.tlog + commands.jsonl + the mission's mapping_ticks.jsonl.

Usage:
  python scripts/mapping_mission.py --dry-run
  python scripts/mapping_mission.py --go            # pilot session; human GOes the race separately
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

# scripts/ importable both as a module (tests import build_schedule) and as a program.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

_G = 9.80665
# Hover collective (normalized [0,1]) -- MEASURED on the VQ2 sim, good to ~2.3% (a little
# extra lift appears under translation). Never rail; the sim mixer couples thrust<->rates.
HOVER_THRUST = 0.2656


class DeadStreamAbort(RuntimeError):
    """Sim streams are not live for THIS client (video never arrives / canned-frozen IMU).
    Flying on requires working feedback and produces zero mapping value -- abort fast.
    (M1 2026-07-05 flew 100 s blind with frames=0 and a bitwise-frozen AHRS: a second
    attached client had the streams. This abort caps that failure at ~3 s.)"""


# ---------------------------------------------------------------------------
# PURE SCHEDULER (no network, no sim-stack imports) -- the unit-tested core.
# ---------------------------------------------------------------------------
# A segment is one open-loop phase. All four kinds hold roll/pitch toward a scripted
# attitude (level unless a leg is pitching) and hold OR slew yaw; the difference is what
# the per-tick command synthesiser (segment_setpoint) does with the phase parameters.
SEG_TAKEOFF = "TAKEOFF"   # gentle climb ramp -> mid height, then a brief level hover trim
SEG_PANO = "PANO"         # in-place 360: level hold + constant yaw-rate slew of the target
SEG_LEG = "LEG"           # short translation: pitch fwd, coast level, brake, re-trim
SEG_SETTLE = "SETTLE"     # level hover, N seconds


@dataclass(frozen=True)
class Segment:
    """One open-loop mission phase. ``duration_s`` is wall-time in the phase; the remaining
    fields are the (constant) open-loop knobs the per-tick synthesiser reads. All timings are
    open-loop -- there is NO position feedback on this wire."""

    name: str
    kind: str
    duration_s: float
    # PANO: constant yaw-rate slew of the heading target (rad/s, + = nose-right / CW-from-above).
    yaw_rate_rps: float = 0.0
    # LEG: sub-phase pitch schedule. Pitch is NED aerospace: NEGATIVE pitch = nose-DOWN =
    # accelerate FORWARD (body +x). We accelerate for ``accel_s``, coast level for ``coast_s``,
    # then brake with the opposite pitch for ``brake_s`` (accel_s + coast_s + brake_s ==
    # duration_s). ``pitch_mag_rad`` is the |tilt| used in the accel and brake pulses.
    accel_s: float = 0.0
    coast_s: float = 0.0
    brake_s: float = 0.0
    pitch_mag_rad: float = 0.0
    # TAKEOFF: climb thrust delta (added to hover) for ``climb_s``, then a level hover-trim for
    # the remainder (duration_s - climb_s). During the trim the vz damper owns the collective.
    climb_s: float = 0.0
    climb_thrust_delta: float = 0.0


@dataclass(frozen=True)
class MissionConfig:
    """Tunable open-loop timings/params for the M1 mapping mission. Defaults give a ~100-120 s
    flight: takeoff, PANO, LEG, PANO, LEG(~120 deg off the first heading), PANO, SETTLE."""

    # -- TAKEOFF --
    takeoff_climb_s: float = 2.5          # gentle climb duration (~mid height, time-based)
    takeoff_climb_thrust_delta: float = 0.035   # + over hover during the climb ramp
    takeoff_trim_s: float = 3.0           # level hover-trim after the climb (vz damper owns thrust)
    # -- PANO --
    pano_yaw_rate_dps: float = 20.0       # yaw slew rate during a panorama (deg/s; ~25 target,
                                          #   trimmed a touch for smoother footage + ~100 s total)
    pano_revs: float = 1.0                # full turns per panorama
    # -- LEG --
    leg_accel_s: float = 2.0              # forward-pitch accel pulse (-> ~1.2 m/s)
    leg_coast_s: float = 6.0              # level coast (wider translation baseline for recon)
    leg_brake_s: float = 1.5             # reverse-pitch brake pulse
    leg_pitch_deg: float = 5.0            # |pitch| magnitude for accel + brake
    leg_retrim_s: float = 1.5             # level hover re-trim after the brake
    # -- heading offset between the two legs --
    # The PANO that PRECEDES the 2nd leg ends at a chosen yaw offset so the 2nd leg departs
    # ~120 deg off the 1st leg's heading. A panorama is an integer-ish number of revs, so we
    # add a fractional extra turn to the MIDDLE panorama to land on the offset. Expressed as
    # the extra fraction of a turn (0.333 -> +120 deg).
    mid_pano_extra_turn: float = 120.0 / 360.0
    # -- SETTLE --
    settle_s: float = 10.0


def build_schedule(cfg: MissionConfig | None = None) -> list[Segment]:
    """Build the M1 mapping-mission segment list (pure -- no I/O). Sequence:
    TAKEOFF, PANO, LEG, PANO(+120 deg offset), LEG, PANO, SETTLE.

    The middle PANO carries an extra fractional turn so the SECOND leg departs ~120 deg off
    the FIRST leg's heading (broad coverage; the reconstruction likes wide baselines)."""
    cfg = cfg or MissionConfig()
    yaw_rate = _dps(cfg.pano_yaw_rate_dps)

    def takeoff() -> Segment:
        return Segment(
            name="takeoff", kind=SEG_TAKEOFF,
            duration_s=cfg.takeoff_climb_s + cfg.takeoff_trim_s,
            climb_s=cfg.takeoff_climb_s,
            climb_thrust_delta=cfg.takeoff_climb_thrust_delta,
        )

    def pano(revs: float, name: str) -> Segment:
        # A yaw-rate-signed panorama: duration = |revs| turns / rate. yaw_rate_rps carries the
        # sign of ``revs`` so a negative revs turns the other way (unused by M1 but general).
        dur = abs(revs) * 2.0 * 3.141592653589793 / max(yaw_rate, 1e-6)
        signed_rate = yaw_rate * (1.0 if revs >= 0 else -1.0)
        return Segment(name=name, kind=SEG_PANO, duration_s=dur, yaw_rate_rps=signed_rate)

    def leg(name: str) -> Segment:
        return Segment(
            name=name, kind=SEG_LEG,
            duration_s=cfg.leg_accel_s + cfg.leg_coast_s + cfg.leg_brake_s + cfg.leg_retrim_s,
            accel_s=cfg.leg_accel_s, coast_s=cfg.leg_coast_s, brake_s=cfg.leg_brake_s,
            pitch_mag_rad=_deg(cfg.leg_pitch_deg),
        )

    return [
        takeoff(),
        pano(cfg.pano_revs, "pano_1"),
        leg("leg_1"),
        # middle panorama: full turns + the +120 deg fractional turn -> next leg 120 deg off.
        pano(cfg.pano_revs + cfg.mid_pano_extra_turn, "pano_2_turn"),
        leg("leg_2"),
        pano(cfg.pano_revs, "pano_3"),
        Segment(name="settle", kind=SEG_SETTLE, duration_s=cfg.settle_s),
    ]


def _deg(d: float) -> float:
    return d * 3.141592653589793 / 180.0


def _dps(d: float) -> float:
    return _deg(d)


@dataclass(frozen=True)
class TickSetpoint:
    """The OPEN-LOOP command intent for one tick, BEFORE the attitude loop / sensors run.
    Deterministic function of (segment, time-in-segment) only -- this is what the scheduler
    tests assert against. The flight loop turns (roll_des, pitch_des, dyaw) into body rates via
    ``level_hold_body_rate`` and applies the thrust policy."""

    segment: str          # segment name
    kind: str             # segment kind (SEG_*)
    roll_des_rad: float   # desired roll offset from level (always 0 in M1)
    pitch_des_rad: float  # desired pitch offset from level (fwd/back on a LEG, else 0)
    dyaw_rad: float       # yaw-target advance THIS tick (rate * dt), 0 unless PANO
    # thrust policy this tick:
    #   "climb"  -> hover + climb_thrust_delta (open-loop takeoff ramp)
    #   "damp"   -> hover + kd*(0 - vz_imu)    (vz-damped level hover; the default)
    thrust_mode: str
    climb_thrust_delta: float = 0.0


def segment_setpoint(seg: Segment, t_in_seg: float, dt: float) -> TickSetpoint:
    """Deterministic per-tick open-loop intent for a segment (pure). ``t_in_seg`` is seconds
    since the segment started; ``dt`` is the tick period (for the yaw-rate advance)."""
    if seg.kind == SEG_TAKEOFF:
        if t_in_seg < seg.climb_s:
            return TickSetpoint(seg.name, seg.kind, 0.0, 0.0, 0.0,
                                thrust_mode="climb", climb_thrust_delta=seg.climb_thrust_delta)
        # hover-trim: level, vz-damped
        return TickSetpoint(seg.name, seg.kind, 0.0, 0.0, 0.0, thrust_mode="damp")

    if seg.kind == SEG_PANO:
        return TickSetpoint(seg.name, seg.kind, 0.0, 0.0,
                            dyaw_rad=seg.yaw_rate_rps * dt, thrust_mode="damp")

    if seg.kind == SEG_LEG:
        # NED pitch: NEGATIVE = nose-DOWN = accelerate forward; POSITIVE = nose-up = brake.
        if t_in_seg < seg.accel_s:
            pitch = -seg.pitch_mag_rad
        elif t_in_seg < seg.accel_s + seg.coast_s:
            pitch = 0.0
        elif t_in_seg < seg.accel_s + seg.coast_s + seg.brake_s:
            pitch = +seg.pitch_mag_rad
        else:
            pitch = 0.0   # re-trim: level
        return TickSetpoint(seg.name, seg.kind, 0.0, pitch, 0.0, thrust_mode="damp")

    # SEG_SETTLE (and any unknown kind): level, vz-damped hover.
    return TickSetpoint(seg.name, seg.kind, 0.0, 0.0, 0.0, thrust_mode="damp")


def schedule_total_s(schedule: list[Segment]) -> float:
    return float(sum(s.duration_s for s in schedule))


def describe_schedule(schedule: list[Segment], cfg: MissionConfig | None = None) -> str:
    """Human-readable dump of the schedule (name, duration, commanded rates/thrust deltas) --
    the ``--dry-run`` output. Pure (no network)."""
    cfg = cfg or MissionConfig()
    lines: list[str] = []
    lines.append(f"MAPPING MISSION M1 schedule -- {len(schedule)} segments, "
                 f"total {schedule_total_s(schedule):.1f} s")
    lines.append(f"  hover_thrust={HOVER_THRUST:.4f}  "
                 f"pano_rate={cfg.pano_yaw_rate_dps:.0f} deg/s  leg_pitch={cfg.leg_pitch_deg:.0f} deg")
    lines.append("  " + "-" * 84)
    lines.append(f"  {'#':>2}  {'segment':<12} {'kind':<8} {'dur[s]':>7}  detail")
    lines.append("  " + "-" * 84)
    t0 = 0.0
    for i, s in enumerate(schedule):
        detail = _segment_detail(s)
        lines.append(f"  {i:>2}  {s.name:<12} {s.kind:<8} {s.duration_s:>7.2f}  "
                     f"[t={t0:6.1f}->{t0 + s.duration_s:6.1f}]  {detail}")
        t0 += s.duration_s
    lines.append("  " + "-" * 84)
    lines.append("  thrust: climb = hover+delta (open-loop ramp);  damp = hover + kd*(0 - vz_imu)")
    lines.append("  attitude: level hold via AHRS (kp on rotvec err), body_rate_sign=(1,1,1), "
                 "cmd_rate_scale=0.4")
    return "\n".join(lines)


def _segment_detail(s: Segment) -> str:
    if s.kind == SEG_TAKEOFF:
        return (f"climb {s.climb_s:.1f}s @ hover+{s.climb_thrust_delta:+.3f} thrust, "
                f"then hover-trim {s.duration_s - s.climb_s:.1f}s (vz-damped)")
    if s.kind == SEG_PANO:
        dps = s.yaw_rate_rps * 180.0 / 3.141592653589793
        revs = abs(s.yaw_rate_rps) * s.duration_s / (2.0 * 3.141592653589793)
        return f"yaw-rate {dps:+.1f} deg/s for {revs:.2f} rev  (level, vz-damped)"
    if s.kind == SEG_LEG:
        pdeg = s.pitch_mag_rad * 180.0 / 3.141592653589793
        return (f"accel(nose-down -{pdeg:.0f}deg) {s.accel_s:.1f}s, coast {s.coast_s:.1f}s, "
                f"brake(+{pdeg:.0f}deg) {s.brake_s:.1f}s, re-trim "
                f"{s.duration_s - s.accel_s - s.coast_s - s.brake_s:.1f}s")
    if s.kind == SEG_SETTLE:
        return f"level hover {s.duration_s:.1f}s (vz-damped)"
    return ""


# ---------------------------------------------------------------------------
# FLIGHT (only imported/reached under --go; keeps --dry-run + tests network-free)
# ---------------------------------------------------------------------------
def _run_go(args: argparse.Namespace) -> int:
    # Heavy sim-stack imports are LOCAL to --go so --dry-run and the scheduler tests never pull
    # pymavlink/cv2/scipy or open a socket.
    import json
    import threading
    import time

    import numpy as np

    from racer.ahrs.ahrs_adapter import AHRSAttitudeSource
    from racer.contracts import ControlCommand, ControlMode
    from racer.controller import level_hold_body_rate
    from racer.deploy_profile import get_profile
    from racer.mavlink_client import MavlinkClient
    from racer.recording import Recorder, session_stamp
    from racer.vertical_estimator import VerticalEstimator, a_up_from_specific_force
    from racer.vision.jpeg_receiver import VIDEO_PORT, JpegUdpReceiver

    cfg = MissionConfig()
    schedule = build_schedule(cfg)
    print(describe_schedule(schedule, cfg))

    # -- client: EXACT vq2_case_c uplink construction (cmd_rate_scale + gyro_sign) --
    profile = get_profile("vq2_case_c")
    cmd_rate_scale = profile.cmd_rate_scale
    gyro_sign = profile.gyro_sign
    print(f"\n[mapping] deploy profile vq2_case_c -> cmd_rate_scale={cmd_rate_scale:g} "
          f"gyro_sign={tuple(gyro_sign)}")
    client = MavlinkClient(args.endpoint, cmd_rate_scale=cmd_rate_scale, gyro_sign=gyro_sign,
                           parse_actuator_output=False)
    print(f"connecting {args.endpoint} ...")
    client.connect(wait_heartbeat=False, timeout_s=args.connect_timeout)

    # -- session + recorder (video.bin/video_index.jsonl/mavlink.tlog/commands.jsonl) --
    session = Path(args.out_dir) / f"{session_stamp()}_{args.label}"
    recorder = Recorder(session)
    recorder.start()
    recorder.add_meta(endpoint=args.endpoint, label=args.label, mission="M1",
                      cmd_rate_scale=cmd_rate_scale, gyro_sign=tuple(float(x) for x in gyro_sign),
                      hover_thrust=HOVER_THRUST, total_schedule_s=schedule_total_s(schedule))
    print(f"recording -> {session}")

    stop = threading.Event()

    # raw MAVLink -> tlog; outgoing commands -> commands.jsonl (same taps the racing path uses)
    def _on_msg(msg):
        if msg.get_type() == "BAD_DATA":
            return
        buf = msg.get_msgbuf()
        if buf:
            recorder.record_mavlink(bytes(buf))
    client.on_message = _on_msg

    def _on_cmd(info: dict):
        recorder.record_command(info)
    client.on_command = _on_cmd

    # -- video receiver thread: publish freshest frame + record it (mapping's whole point) --
    def _video():
        while not stop.is_set():
            try:
                with JpegUdpReceiver(port=args.video_port) as rx:
                    for fr in rx.frames(max_wait_s=5.0):
                        recorder.record_frame(fr)
                        if stop.is_set():
                            break
            except Exception as exc:
                print(f"video thread: {exc}", file=sys.stderr)
            if not stop.is_set():
                time.sleep(0.5)
    vthread = threading.Thread(target=_video, name="video", daemon=True)
    vthread.start()
    print(f"listening for video on UDP {args.video_port}")

    tick_log = open(session / "mapping_ticks.jsonl", "w", encoding="utf-8")

    # -- estimators owned HERE (self-contained; not the navigator) --
    ahrs = AHRSAttitudeSource()
    # vz washout: same defaults as the vq2 estimator; we read ``vz_imu`` (the vision-free channel).
    vest = VerticalEstimator()
    kd_vz = float(args.kd_vz)          # thrust damper gain on vz_imu (down-positive)
    kp_att = float(args.kp_att)
    kd_att = float(args.kd_att)
    max_rate = float(args.max_rate)
    body_rate_sign = np.array([1.0, 1.0, 1.0])   # vq2_case_c flown value (A36 Item-0)

    rc = 0
    try:
        # ---- 1) WAIT for a live, STARTED race (human GOes it via vq2ctl; we NEVER send GO) ----
        if not _wait_started(client, args.wait_seconds):
            print("\n[mapping] no RACE_STATUS started within "
                  f"{args.wait_seconds:g}s -> aborting (nothing armed).", file=sys.stderr)
            return 2
        print("\n[mapping] race STARTED -> arming + flying the mission.")

        # ---- 2) ARM (this is the drone arm, NOT a race GO) ----
        client.arm()
        if not client.wait_armed(True, timeout_s=args.arm_timeout):
            print("[mapping] arm not confirmed via HEARTBEAT; proceeding (best-effort).",
                  file=sys.stderr)

        # ---- 3) dead-stream watchdog: frames must arrive and the IMU must be LIVE ----
        wd_s = float(args.watchdog_s)
        wd = {"t0": None, "imu_ns": None, "imu_since": None, "gyro": None, "gyro_since": None}

        def watchdog(mono_now, imu_ns, body_rate):
            if wd_s <= 0.0:
                return
            if wd["t0"] is None:
                wd["t0"] = mono_now
            t_flight = mono_now - wd["t0"]
            if t_flight > wd_s and recorder.n_frames == 0:
                raise DeadStreamAbort(
                    f"no video frame {t_flight:.1f}s after race start "
                    f"(port {args.video_port or 5600} hijacked by another client, or stream dead)")
            if imu_ns != wd["imu_ns"]:
                wd["imu_ns"], wd["imu_since"] = imu_ns, mono_now
            elif wd["imu_since"] is not None and mono_now - wd["imu_since"] > wd_s:
                raise DeadStreamAbort(f"IMU clock frozen {wd_s:g}s (no fresh HIGHRES_IMU)")
            g = None if body_rate is None else (float(body_rate[0]), float(body_rate[1]),
                                                float(body_rate[2]))
            if g != wd["gyro"]:
                wd["gyro"], wd["gyro_since"] = g, mono_now
            elif wd["gyro_since"] is not None and mono_now - wd["gyro_since"] > wd_s:
                raise DeadStreamAbort(f"gyro bitwise-frozen {wd_s:g}s (canned IMU tuple)")

        # ---- 4) run the segment state machine at ~control_hz ----
        _fly_schedule(
            client, schedule, ahrs, vest, tick_log,
            control_hz=args.control_hz, kp_att=kp_att, kd_att=kd_att, max_rate=max_rate,
            kd_vz=kd_vz, body_rate_sign=body_rate_sign, stop=stop, np=np,
            ControlCommand=ControlCommand, ControlMode=ControlMode,
            level_hold_body_rate=level_hold_body_rate,
            a_up_from_specific_force=a_up_from_specific_force, time=time, json=json,
            watchdog=watchdog,
        )

        # ---- 5) post-mission: keep streaming LEVEL hover for --hold-after-s, then stop ----
        print(f"\n[mapping] mission complete -> holding hover {args.hold_after_s:g}s, then disarm.")
        _fly_hold(
            client, ahrs, vest, tick_log, hold_s=args.hold_after_s,
            control_hz=args.control_hz, kp_att=kp_att, kd_att=kd_att, max_rate=max_rate,
            kd_vz=kd_vz, body_rate_sign=body_rate_sign, stop=stop, np=np,
            ControlCommand=ControlCommand, ControlMode=ControlMode,
            level_hold_body_rate=level_hold_body_rate,
            a_up_from_specific_force=a_up_from_specific_force, time=time, json=json,
        )
    except DeadStreamAbort as exc:
        print(f"\n[mapping] DEAD-STREAM ABORT: {exc} -> disarming NOW, no mission flown.",
              file=sys.stderr)
        rc = 2
    except KeyboardInterrupt:
        print("\n[mapping] Ctrl-C -> disarming.")
    except Exception:
        import traceback
        traceback.print_exc()
        rc = 1
    finally:
        try:
            client.disarm(force=True)
        except Exception:
            pass
        stop.set()
        vthread.join(timeout=6.0)
        try:
            tick_log.flush()
            tick_log.close()
        except Exception:
            pass
        recorder.add_meta(collisions=len(client.collisions), race_status=client.race_status)
        recorder.close()
        print(f"[mapping] frames={recorder.n_frames} mav={recorder.n_mavlink} "
              f"cmds={recorder.n_commands} dropped={recorder.n_dropped} -> {session}")
    return rc


def _wait_started(client, wait_seconds: float) -> bool:
    """Pump until RACE_STATUS.started==True on a LIVE wire (sim clock advancing). We do NOT
    send GO -- a separate pilot session GOes the race. Returns True on a started race."""
    import time
    deadline = time.monotonic() + wait_seconds
    last_p = 0.0
    while time.monotonic() < deadline:
        client.pump()
        rs = client.race_status
        live = client.state.sim_time_ns > 0
        if rs and rs.get("started") and live:
            return True
        now = time.monotonic()
        if now - last_p >= 2.0:
            print(f"  waiting for GO: started={bool(rs and rs.get('started'))} "
                  f"live={'y' if live else 'n'}   ", end="\r", flush=True)
            last_p = now
        time.sleep(0.005)
    return False


def _race_live(client) -> bool:
    rs = client.race_status
    return bool(rs and rs.get("started")) and client.state.sim_time_ns > 0


def _step_estimators(client, ahrs, vest, prev_imu_ns, np, a_up_from_specific_force):
    """Step the AHRS + vz washout on the freshest IMU sample, on the MASTER sim clock. Returns
    (roll, pitch, yaw, body_rate, vz_imu, new_prev_imu_ns). Only integrates the vz washout while
    the race is live; resets it on a clock jump BACKWARD (the sim IMU restart trap)."""
    s = client.state
    imu_ns = int(s.sim_time_ns)
    accel = s.accel_body
    gyro = s.gyro_body
    # AHRS: ingest on positive master-clock dt. gyro None (pre-first-IMU) -> skip.
    dt = 0.0
    if prev_imu_ns is not None and imu_ns > prev_imu_ns:
        dt = (imu_ns - prev_imu_ns) / 1e9
    if gyro is not None and accel is not None:
        ahrs.ingest(np.asarray(accel, dtype=float), np.asarray(gyro, dtype=float), float(dt))
    roll, pitch, yaw = ahrs.euler_rpy
    body_rate = ahrs.body_rate

    # vz washout: seed once on first live IMU; reset on a BACKWARD clock jump (race restart).
    if not vest.seeded:
        vest.seed()
    if prev_imu_ns is not None and imu_ns < prev_imu_ns:
        # sim_time reset backward (race restart / frozen-canned-tuple wrap): forget stale state.
        vest.seed()
        dt = 0.0
    if 0.0 < dt <= vest.max_dt_s and accel is not None:
        a_up = a_up_from_specific_force(np.asarray(accel, dtype=float), ahrs.R_wb)
        vest.predict(float(a_up), float(dt))
    return roll, pitch, yaw, body_rate, vest.vz_imu, imu_ns


def _thrust_from_vz(vz_imu, kd_vz, np) -> float:
    """thrust = hover + kd*(0 - vz_imu), clamped to a SAFE band (never rail; the mixer couples
    thrust<->rates near the rails). vz_imu NED down-positive: descending (vz>0) -> more thrust."""
    vz = 0.0 if (vz_imu is None or not np.isfinite(vz_imu)) else float(vz_imu)
    thr = HOVER_THRUST + kd_vz * (0.0 - vz)
    return float(np.clip(thr, 0.18, 0.42))


def _fly_schedule(client, schedule, ahrs, vest, tick_log, *, control_hz, kp_att, kd_att,
                  max_rate, kd_vz, body_rate_sign, stop, np, ControlCommand, ControlMode,
                  level_hold_body_rate, a_up_from_specific_force, time, json, watchdog=None):
    """Run the segment state machine at ~control_hz. Yaw target is PERSISTENT and slewed by the
    PANO yaw-rate; roll/pitch held at the segment's scripted offset; thrust vz-damped (or the
    open-loop climb ramp on takeoff)."""
    tick = 1.0 / control_hz
    prev_imu_ns = None
    yaw_target = None                 # set from the AHRS yaw once we have a real attitude
    seg_i = 0
    seg_t0 = time.monotonic()
    next_t = time.monotonic()
    n_tick = 0
    while seg_i < len(schedule) and not stop.is_set():
        while time.monotonic() < next_t:
            client.pump()
            time.sleep(0.0005)
        next_t = time.monotonic() + tick
        now = time.monotonic()
        client.pump()

        seg = schedule[seg_i]
        t_in_seg = now - seg_t0
        if t_in_seg >= seg.duration_s:
            seg_i += 1
            seg_t0 = now
            continue

        roll, pitch, yaw, body_rate, vz_imu, prev_imu_ns = _step_estimators(
            client, ahrs, vest, prev_imu_ns, np, a_up_from_specific_force)
        if watchdog is not None:
            watchdog(now, prev_imu_ns, body_rate)
        if yaw_target is None:
            yaw_target = float(yaw)

        sp = segment_setpoint(seg, t_in_seg, tick)
        yaw_target = float(yaw_target + sp.dyaw_rad)

        # attitude -> body rate: feed the AHRS TRUE euler MINUS the desired tilt offset (same
        # (roll-roll_des, pitch-pitch_des) form rate_sysid uses), hold yaw at yaw_target.
        omega = level_hold_body_rate(
            float(roll) - sp.roll_des_rad, float(pitch) - sp.pitch_des_rad, float(yaw), yaw_target,
            np.asarray(body_rate, dtype=float),
            kp=kp_att, kd=kd_att, body_rate_sign=body_rate_sign, max_rate=max_rate, ff_gain=1.0,
        )

        if sp.thrust_mode == "climb":
            thr = float(np.clip(HOVER_THRUST + sp.climb_thrust_delta, 0.18, 0.42))
        else:
            thr = _thrust_from_vz(vz_imu, kd_vz, np)

        client.send_command(ControlCommand(mode=ControlMode.BODY_RATE,
                                           sim_time_ns=int(client.state.sim_time_ns),
                                           body_rate=omega, thrust=float(thr)))
        n_tick += 1
        _log_tick(tick_log, json, time, seg.name, sp, roll, pitch, yaw, yaw_target,
                  omega, thr, vz_imu, np)
        if n_tick % max(int(control_hz), 1) == 0:
            print(f"  [{seg.name:<12}] t_seg={t_in_seg:5.1f}/{seg.duration_s:4.1f}s  "
                  f"rpy=({np.degrees(roll):+5.1f},{np.degrees(pitch):+5.1f},"
                  f"{np.degrees(yaw):+6.1f})deg  thr={thr:.3f}  vz={_f(vz_imu):+.2f}   ",
                  end="\r", flush=True)
    print()


def _fly_hold(client, ahrs, vest, tick_log, *, hold_s, control_hz, kp_att, kd_att, max_rate,
              kd_vz, body_rate_sign, stop, np, ControlCommand, ControlMode,
              level_hold_body_rate, a_up_from_specific_force, time, json):
    """Stream LEVEL, vz-damped hover for ``hold_s`` seconds (the clean-exit tail), holding the
    final yaw."""
    tick = 1.0 / control_hz
    prev_imu_ns = None
    yaw_target = None
    t_end = time.monotonic() + hold_s
    next_t = time.monotonic()
    while time.monotonic() < t_end and not stop.is_set():
        while time.monotonic() < next_t:
            client.pump()
            time.sleep(0.0005)
        next_t = time.monotonic() + tick
        client.pump()
        roll, pitch, yaw, body_rate, vz_imu, prev_imu_ns = _step_estimators(
            client, ahrs, vest, prev_imu_ns, np, a_up_from_specific_force)
        if yaw_target is None:
            yaw_target = float(yaw)
        omega = level_hold_body_rate(
            float(roll), float(pitch), float(yaw), yaw_target,
            np.asarray(body_rate, dtype=float),
            kp=kp_att, kd=kd_att, body_rate_sign=body_rate_sign, max_rate=max_rate, ff_gain=1.0,
        )
        thr = _thrust_from_vz(vz_imu, kd_vz, np)
        client.send_command(ControlCommand(mode=ControlMode.BODY_RATE,
                                           sim_time_ns=int(client.state.sim_time_ns),
                                           body_rate=omega, thrust=float(thr)))
        _log_tick(tick_log, json, time, "hold", None, roll, pitch, yaw, yaw_target,
                  omega, thr, vz_imu, np)


def _f(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _log_tick(tick_log, json, time, seg_name, sp, roll, pitch, yaw, yaw_target,
              omega, thr, vz_imu, np):
    try:
        rec = {
            "t_mono_ns": time.monotonic_ns(),
            "segment": seg_name,
            "cmd_body_rate": [round(float(v), 5) for v in omega],
            "thrust": round(float(thr), 5),
            "ahrs_rpy": [round(float(roll), 5), round(float(pitch), 5), round(float(yaw), 5)],
            "yaw_target": round(float(yaw_target), 5),
            "vz_imu": (None if (vz_imu is None or not np.isfinite(vz_imu)) else round(float(vz_imu), 5)),
        }
        if sp is not None:
            rec["roll_des"] = round(float(sp.roll_des_rad), 5)
            rec["pitch_des"] = round(float(sp.pitch_des_rad), 5)
            rec["thrust_mode"] = sp.thrust_mode
        tick_log.write(json.dumps(rec) + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true",
                      help="print the segment schedule and exit (no network)")
    mode.add_argument("--go", action="store_true",
                      help="attach to the live sim, wait for RACE_STATUS started, fly the mission")
    ap.add_argument("--endpoint", default="udp:127.0.0.1:14550")
    ap.add_argument("--video-port", type=int, default=None,
                    help="UDP video port (default: racer.vision.jpeg_receiver.VIDEO_PORT)")
    ap.add_argument("--label", default="mapping_m1", help="session dir suffix")
    ap.add_argument("--out-dir", default="data/runs")
    ap.add_argument("--control-hz", type=float, default=100.0,
                    help="CTBR command rate (~100 Hz per the VQ2 control handshake)")
    ap.add_argument("--kp-att", type=float, default=4.0, help="attitude-error -> body-rate gain")
    ap.add_argument("--kd-att", type=float, default=0.0, help="body-rate damping on the AHRS rate")
    ap.add_argument("--max-rate", type=float, default=0.8,
                    help="composed body-rate clamp (rad/s) -- conservative for mapping")
    ap.add_argument("--kd-vz", type=float, default=0.06,
                    help="thrust damper gain on vz_imu (thrust = hover + kd*(0 - vz_imu))")
    ap.add_argument("--wait-seconds", type=float, default=180.0,
                    help="how long to wait for RACE_STATUS started before aborting")
    ap.add_argument("--arm-timeout", type=float, default=5.0)
    ap.add_argument("--watchdog-s", type=float, default=3.0,
                    help="dead-stream watchdog: abort if no video frame OR the IMU is "
                         "frozen for this many seconds after race start (0 disables)")
    ap.add_argument("--hold-after-s", type=float, default=5.0,
                    help="stream level hover this long after SETTLE, then disarm")
    ap.add_argument("--connect-timeout", type=float, default=15.0)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        print(describe_schedule(build_schedule()))
        return 0
    # --go: resolve the default video port only now (avoids importing the sim stack for --dry-run).
    if args.video_port is None:
        from racer.vision.jpeg_receiver import VIDEO_PORT
        args.video_port = VIDEO_PORT
    return _run_go(args)


if __name__ == "__main__":
    raise SystemExit(main())
