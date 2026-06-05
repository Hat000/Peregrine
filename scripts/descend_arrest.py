"""DESCEND-ARREST @ 250 Hz -- the World-A-vs-Partial discriminator (ShadowPC 2026-06-05).

Run #1 settled that World B is dead: in ANGLE @ 250 Hz, v=(0,0,0) drives thrust to MAX (~0.98) and
the drone accelerates upward (reproduced x2, GUI-confirmed). This run asks whether commanding vz has
ANY authority over that climb:

  Phase A (CLIMB):   command v=(0,0,0); let it climb on its own to ~climb_trigger m up (it's fast --
                     minimal headroom by design).
  Phase B (DESCEND): the INSTANT it passes that altitude, snap to a HARD DOWN command
                     v=(0,0,+descend_vz)  (NED vz>0 = descend), yaw-ignore (mask 3527), and HOLD.
                     Run until the climb is clearly arrested/reversed, OR the 8 m alt abort.

Discriminator -- achieved vz(t) + motors(t) across the A->B switch:
  ARRESTS / REVERSES (vz turns from negative toward +descend_vz; motors drop off max)
        => vz HAS authority => PARTIAL: vertical is velocity-controllable, vz=0 is a pathology
           (we'd fly a hover-trim vz; next step a vz sweep to find it).
  KEEPS CLIMBING to 8 m at max thrust, indifferent to the down command
        => vz has NO authority => WORLD A: vertical uncontrollable via velocity => CTBR.

Floor-safety: in Phase B, if it descends back to within `floor_alt_m` of the start altitude, cut the
command to vz=0 so it can't slam the ground (vz=0 re-climbs -- the very pathology -- which is safe
here). Other safety: alt>8 m, offset>12 m, tilt>80 deg, collision>=2 -> force-disarm.

Clean tap: heartbeats + timesync OFF (stay ANGLE). Does NOT arm (race auto-arms). Refuses to start
unless the drone is at origin (|pos|<1 m) + clock advancing. Reuses velocity_ladder's validated
instrument (raw 3-channel tap, frame-aware triangulation, recorder).

Usage: python scripts/descend_arrest.py --label da1   # race must be LIVE + ANGLE; teammate watching
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from racer.contracts import ControlCommand, ControlMode
from racer.mavlink_client import MavlinkClient, _pos_type_mask
from racer.recording import Recorder, session_stamp
from velocity_ladder import REF_MASK, Channels, _dump, _f3, _tri_disagreement


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default="udp:127.0.0.1:14550")
    ap.add_argument("--rate", type=float, default=250.0)
    ap.add_argument("--descend-vz", type=float, default=2.0, help="Phase-B hard DOWN command (NED vz>0)")
    ap.add_argument("--climb-trigger-m", type=float, default=1.5, help="climb this far up on v=0 before switching to DOWN")
    ap.add_argument("--phase-a-max-s", type=float, default=6.0)
    ap.add_argument("--phase-b-max-s", type=float, default=5.0)
    ap.add_argument("--floor-alt-m", type=float, default=0.5, help="cut to vz=0 if it descends back within this of start")
    ap.add_argument("--null-s", type=float, default=4.0)
    ap.add_argument("--null-thresh", type=float, default=0.05)
    ap.add_argument("--fd-window", type=float, default=0.05)
    ap.add_argument("--tri-abort", type=float, default=1.5, help="sustained v_lpn-vs-v_odo_world gap that halts (gross faults only)")
    ap.add_argument("--tri-abort-ticks", type=int, default=25)
    ap.add_argument("--wait-live-s", type=float, default=90.0)
    ap.add_argument("--max-offset-m", type=float, default=12.0)
    ap.add_argument("--max-alt-m", type=float, default=8.0)
    ap.add_argument("--max-tilt-deg", type=float, default=80.0)
    ap.add_argument("--origin-tol-m", type=float, default=1.0)
    ap.add_argument("--label", default="descend_arrest")
    ap.add_argument("--connect-timeout", type=float, default=10.0)
    args = ap.parse_args()

    c = MavlinkClient(args.endpoint)
    c.send_heartbeats = False
    c.send_timesync = False
    c.connect(wait_heartbeat=False, timeout_s=args.connect_timeout)

    session = Path("data/runs") / f"{session_stamp()}_{args.label}"
    recorder = Recorder(session)
    recorder.start()
    channels = Channels(fd_window_s=args.fd_window)

    echo_calls: list[tuple] = []
    _orig_send = c.conn.mav.set_position_target_local_ned_send
    def _echo_send(*a, **k):
        echo_calls.append(a)
        return _orig_send(*a, **k)
    c.conn.mav.set_position_target_local_ned_send = _echo_send

    def _on_msg(m) -> None:
        if m.get_type() != "BAD_DATA" and m.get_msgbuf():
            recorder.record_mavlink(bytes(m.get_msgbuf()))
        channels.on_msg(m)
    c.on_message = _on_msg

    records: list[dict] = []
    meta: dict = {"probe": "descend_arrest", "label": args.label, "yaw_mode": "ignore",
                  "ned_z_is_down": True, "descend_vz": args.descend_vz,
                  "climb_trigger_m": args.climb_trigger_m, "rate_target": args.rate,
                  "safety": {"max_alt_m": args.max_alt_m, "max_offset_m": args.max_offset_m,
                             "max_tilt_deg": args.max_tilt_deg, "floor_alt_m": args.floor_alt_m}}
    recorder.add_meta(**meta)

    print(f"recording -> {session}")
    print(">>> TAP-IN heartbeats=OFF timesync=OFF. TEAMMATE: race LIVE + confirm GUI=ANGLE. I will NOT arm.\n")

    # ---- ACQUIRE ----
    deadline = time.monotonic() + args.wait_live_s
    last_w = 0.0
    while time.monotonic() < deadline:
        c.pump()
        s = c.state
        if (s.position_ned is not None and s.sim_time_ns > 0 and s.armed
                and float(np.linalg.norm(s.position_ned)) < args.origin_tol_m):
            break
        now = time.monotonic()
        if now - last_w >= 1.0:
            pos = None if s.position_ned is None else np.round(s.position_ned, 2)
            print(f"  waiting: pos={pos} armed={s.armed} sim_t={s.sim_time_ns/1e9:.1f}s", end="\r", flush=True)
            last_w = now
        time.sleep(0.01)
    s = c.state
    if not (s.position_ned is not None and s.sim_time_ns > 0 and s.armed
            and float(np.linalg.norm(s.position_ned)) < args.origin_tol_m):
        print(f"\nREFUSE START: need live+armed+origin(|pos|<{args.origin_tol_m}m). Restart the race.", file=sys.stderr)
        recorder.add_meta(aborted="no_clean_start"); recorder.close()
        return 1
    origin = np.asarray(s.position_ned, dtype=np.float64).copy()
    yaw0 = float(s.yaw)
    meta.update({"origin_ned": _f3(origin), "yaw0_deg": float(np.degrees(yaw0)),
                 "initial_rpy_deg": [float(np.degrees(s.roll)), float(np.degrees(s.pitch)), float(np.degrees(s.yaw))]})
    print(f"  ACQUIRED pos={np.round(origin,3)} rpy_deg="
          f"({np.degrees(s.roll):+.0f},{np.degrees(s.pitch):+.0f},{np.degrees(s.yaw):+.0f})")

    # ---- NULL / MODE gate (teammate confirms ANGLE) ----
    print(f"\n>>> NULL TEST ({args.null_s:g}s passive). TEAMMATE: GUI mode = ANGLE? drone holding?")
    channels.reset_counts()
    nmax = {"v_lpn": 0.0, "v_odo": 0.0, "v_fd": 0.0}
    t_end = time.monotonic() + args.null_s
    while time.monotonic() < t_end:
        c.pump()
        for k, v in (("v_lpn", channels.v_lpn), ("v_odo", channels.v_odo), ("v_fd", channels.v_fd())):
            if v is not None:
                nmax[k] = max(nmax[k], float(np.linalg.norm(v)))
        time.sleep(1.0 / args.rate)
    lpn_hz, odo_hz = channels.n_lpn / args.null_s, channels.n_odo / args.null_s
    telemetry_live = lpn_hz > 5 and odo_hz > 5
    null_pass = telemetry_live and all(nmax[k] < args.null_thresh for k in nmax)
    print(f"  telemetry: LOCAL_POS={lpn_hz:.0f}Hz ODOMETRY={odo_hz:.0f}Hz -> {'LIVE' if telemetry_live else 'DRY'}")
    print(f"  null max |v_lpn|={nmax['v_lpn']:.4f} |v_odo|={nmax['v_odo']:.4f} |v_fd|={nmax['v_fd']:.4f}")
    print(f"  actuator rest: {None if channels.act is None else [round(x,3) for x in channels.act]}")
    meta.update({"null_max": nmax, "telemetry_hz": {"local_pos": lpn_hz, "odometry": odo_hz},
                 "telemetry_live": telemetry_live, "null_pass": null_pass,
                 "actuator_rest": (None if channels.act is None else [float(x) for x in channels.act])})
    if not null_pass:
        print("\nHALT: Phase-1 gate failed (telemetry dry or stationary drone reads nonzero). Report + fix.", file=sys.stderr)
        recorder.add_meta(**meta, aborted="null_fail"); recorder.close(); _dump(session, records, meta)
        return 3

    # ---- COMMAND ECHO ----
    echo_calls.clear()
    c.send_command(ControlCommand(mode=ControlMode.VELOCITY, velocity_ned=np.zeros(3), yaw=None))
    c.pump()
    sent = echo_calls[-1] if echo_calls else None
    echo = None if sent is None else {"mask": int(sent[4]), "vel": _f3(sent[8:11])}
    print(f">>> COMMAND ECHO: {echo}  (reference mask={REF_MASK})")
    meta["command_echo"] = echo

    # ---- PHASE A (climb on v=0) -> PHASE B (hard down) ----
    descend_cmd = np.array([0.0, 0.0, +args.descend_vz])
    zero_cmd = np.zeros(3)
    phase = "A_CLIMB"
    t_switch = None
    floor_hold_until = 0.0
    aborted = None
    reversed_ = False
    motors_dropped = False
    min_motor_B = 1.0
    vz_at_switch = None
    diverge_ticks = 0
    print(f"\n>>> PHASE A: command v=0, let it climb to {args.climb_trigger_m:g} m up, then snap to vz=+{args.descend_vz:g}.")
    print("    TEAMMATE: at the moment the DOWN command starts -- does it stop rising / come back, or keep climbing?")
    last_p = 0.0
    t_a_deadline = time.monotonic() + args.phase_a_max_s
    try:
        while True:
            c.pump()
            cmd = descend_cmd if phase == "B_DESCEND" else zero_cmd   # A_CLIMB + FLOOR_HOLD command vz=0
            c.send_command(ControlCommand(mode=ControlMode.VELOCITY, velocity_ned=cmd, yaw=None))
            s = c.state
            v_lpn, v_odo, v_fd = channels.v_lpn, channels.v_odo, channels.v_fd()
            v_odo_w = channels.v_odo_world()
            pos = channels.pos_lpn if channels.pos_lpn is not None else \
                (np.asarray(s.position_ned) if s.position_ned is not None else origin)
            rel = np.asarray(pos, dtype=np.float64) - origin
            climb_up = -float(rel[2])                          # metres above start
            vz_w = float(v_lpn[2]) if v_lpn is not None else float("nan")   # world NED vz (LPN = world)
            mt = channels.act
            mtr = (sum(mt) / len(mt)) if mt else float("nan")
            tilt_deg = float(np.degrees(max(abs(s.roll), abs(s.pitch))))
            tri = _tri_disagreement([v_lpn, v_odo_w, v_fd])
            lpn_odo = _tri_disagreement([v_lpn, v_odo_w])
            records.append({
                "sim_time_ns": int(s.sim_time_ns), "phase": phase, "cmd_v": _f3(cmd),
                "mask": int(REF_MASK), "v_lpn": _f3(v_lpn), "v_odo": _f3(v_odo),
                "v_odo_world": _f3(v_odo_w), "v_fd": _f3(v_fd),
                "pos_lpn": _f3(channels.pos_lpn), "pos_odo": _f3(channels.pos_odo), "rel_pos": _f3(rel),
                "rpy_deg": [float(np.degrees(s.roll)), float(np.degrees(s.pitch)), float(np.degrees(s.yaw))],
                "actuators": (list(mt) if mt else None), "tri_disagree": tri, "lpn_odo_disagree": lpn_odo,
            })

            # ---- phase transitions ----
            if phase == "A_CLIMB":
                if climb_up >= args.climb_trigger_m:
                    phase = "B_DESCEND"
                    t_switch = time.monotonic()
                    vz_at_switch = vz_w
                    print(f"\n   >>> SWITCH to DOWN (vz=+{args.descend_vz:g}) at climb={climb_up:.2f}m, vz={vz_w:+.2f} m/s")
                elif time.monotonic() > t_a_deadline:
                    aborted = "phaseA_no_climb"
            elif phase == "B_DESCEND":
                if v_lpn is not None and vz_w > 0.3:
                    reversed_ = True
                if mt:
                    min_motor_B = min(min_motor_B, mtr)
                    if mtr < 0.5:
                        motors_dropped = True
                if float(rel[2]) > -args.floor_alt_m:                 # descended back near start -> arrest, don't slam
                    reversed_ = True
                    phase = "FLOOR_HOLD"
                    floor_hold_until = time.monotonic() + 0.5
                    print(f"\n   >>> FLOOR-SAFE: back to {-float(rel[2]):+.2f}m up -> cut to vz=0 (arrest, no slam)")
                elif time.monotonic() > t_switch + args.phase_b_max_s:
                    aborted = "phaseB_cap"
            elif phase == "FLOOR_HOLD":
                if time.monotonic() > floor_hold_until:
                    aborted = "floor_cut_arrested"

            # ---- safety (both phases) ----
            if any(cc["threat_level"] >= 2 for cc in c.collisions):
                aborted = "collision>=2"
            elif float(np.hypot(rel[0], rel[1])) > args.max_offset_m:
                aborted = f"offset {np.hypot(rel[0],rel[1]):.1f}m"
            elif abs(float(rel[2])) > args.max_alt_m:
                aborted = f"alt {rel[2]:+.1f}m"
            elif tilt_deg > args.max_tilt_deg:
                aborted = f"tilt {tilt_deg:.0f}deg"
            if lpn_odo is not None and lpn_odo > args.tri_abort:
                diverge_ticks += 1
                if diverge_ticks >= args.tri_abort_ticks:
                    aborted = f"telemetry_divergence lpn_vs_odo_world={lpn_odo:.2f}"
            else:
                diverge_ticks = 0

            now = time.monotonic()
            if now - last_p >= 0.2:
                print(f"   {phase:9s} cmd_vz={cmd[2]:+.1f} climb={climb_up:+.2f}m vz={vz_w:+.2f} "
                      f"mtr={mtr:.3f} tilt={tilt_deg:.0f}   ", end="\r", flush=True)
                last_p = now
            if aborted:
                break
            time.sleep(1.0 / args.rate)
    except KeyboardInterrupt:
        aborted = "ctrl-c"
    finally:
        print("\n[safety] force-disarming ...")
        try:
            c.send_heartbeats = True
            c.disarm(force=True)
            c.wait_armed(False, timeout_s=3.0)
        except Exception as exc:
            print(f"  disarm error: {exc}", file=sys.stderr)
        # ---- classification ----
        reached_ceiling = isinstance(aborted, str) and aborted.startswith("alt")
        if reached_ceiling and not reversed_ and not motors_dropped:
            verdict = "WORLD_A (climbed to ceiling at max thrust, vz command ignored -> CTBR)"
        elif reversed_ or motors_dropped or (isinstance(aborted, str) and aborted.startswith("floor")):
            verdict = "PARTIAL (vz HAS authority -- climb arrested/reversed -> fly a hover-trim vz)"
        else:
            verdict = "INCONCLUSIVE (re-run: switch earlier / more headroom)"
        meta.update({"aborted": aborted, "vz_at_switch": vz_at_switch, "reversed": reversed_,
                     "motors_dropped": motors_dropped, "min_motor_phaseB": (None if min_motor_B == 1.0 else min_motor_B),
                     "suggested_verdict": verdict})
        recorder.add_meta(**meta)
        recorder.close()
        _dump(session, records, meta)
    print(f"\n==== descend_arrest DONE  aborted={aborted} ====")
    print(f"  vz_at_switch={vz_at_switch}  reversed={reversed_}  motors_dropped={motors_dropped}  min_motor_B={meta['min_motor_phaseB']}")
    print(f"  SUGGESTED: {verdict}")
    print(f"  recording={session}  json={session/'ladder.json'}")
    print("  (OBSERVATION, not final -- corroborate against the teammate's GUI read of the switch.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
