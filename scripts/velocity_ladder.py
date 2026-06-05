"""Velocity-setpoint LADDER in ANGLE mode -- the decisive easy-mode measurement.

THE QUESTION (binary): in ANGLE mode, does the sim's velocity-setpoint controller HOLD
ALTITUDE (vz=0) and TRACK a commanded NED velocity? If yes -> World B (fly on velocity, most
CTBR work unnecessary). If HOVER climbs away regardless of vz AND vz commands don't respond ->
World A (CTBR justified). If HOVER climbs but vz commands DO respond -> Partial (small fix).

Discovery chain this re-tests (see handoff/shadowpc-velocity-fork-2026-06-04/UNDERSTANDING.md):
  * NO client heartbeat/timesync -> sim stays ANGLE (a HEARTBEAT or TIMESYNC flips it to ACRO);
  * a velocity SETPOINT does NOT flip the mode;
  * the prior "easy-mode DEAD" verdict was measured with heartbeats ON (= ACRO, never ANGLE),
    via control_mode_probe, whose climb>0.2m "RESPONDED" test was a false positive.
  * the reference client (PyAIPilotExample/controller.py) ships pure velocity (vx=2,vy=0,vz=0,
    no thrust) as its simple control path.
So: tap in passively (heartbeats OFF, timesync OFF) and send ONLY velocity setpoints.

DISCIPLINE -- this is a MEASUREMENT, not a flight. We report DATA, not conclusions:
  PHASE 1 (instrument gate, runs first, every run):
    * NULL TEST (passive): stationary drone at origin -> all THREE velocity channels read ~0
      (|v|<0.05). Nonzero on a stationary drone = biased pipeline -> HALT (no commanding).
    * TELEMETRY LIVENESS (folded into the null window): with HB+TS both OFF, confirm telemetry
      still flows. If it dries up -> rerun with --keepalive timesync and let the teammate report
      whether the GUI stays ANGLE (resolves the mavlink_client vs angle_vel_test contradiction).
    * COMMAND ECHO: capture the ACTUAL transmitted SET_POSITION_TARGET mask + fields (by wrapping
      the pymavlink send) and diff vs the reference client (mask 3527 yaw-ignore / 2503 yaw-hold).
    * ACTUATOR witness: confirm ACTUATOR_OUTPUT_STATUS reads sane; log motors every tick as a
      SECOND, parser-independent signal of whether the sim is actually controlling.
  Three velocity derivations are kept SEPARATE (mavlink_client collapses LOCAL_POSITION_NED +
  ODOMETRY into one velocity_ned field; we tap the RAW messages instead):
      v_lpn = LOCAL_POSITION_NED.(vx,vy,vz)   v_odo = ODOMETRY.(vx,vy,vz)   v_fd = d(pos)/dt
  A live monitor HALTS the run on gross 3-channel divergence (instruments disagree).
  PHASE 2 (the ladder), each phase ~3s @ 25 Hz, every tick logged:
      HOVER   (0,0,0)            does z hold? (the crux)
      DESCEND (0,0,+0.5)         honors vz? right sign? (+z is DOWN)
      CLIMB   (0,0,-0.5)         vz authority the other way
      FORWARD (~1.0 toward g0)   horizontal tracking with z flat
  Run twice with --yaw-mode ignore (cmd.yaw=None, EXACT reference match) and --yaw-mode hold
  (cmd.yaw=yaw0). If they differ, our yaw setpoint was the bug.

SAFETY: abort + force-disarm on offset>12 m, |alt|>8 m, tilt>80 deg, or collision threat>=2.
We do NOT arm (the race auto-arms) and refuse to start unless at origin (|pos|<1 m) + clock live.

Usage:
  python scripts/velocity_ladder.py --null-only --label gate          # Phase-1 gate (no ladder)
  python scripts/velocity_ladder.py --yaw-mode ignore --label yi1     # full ladder, yaw ignored
  python scripts/velocity_ladder.py --yaw-mode hold   --label yh1     # full ladder, yaw held
  python scripts/velocity_ladder.py --keepalive timesync --null-only  # liveness fallback
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from racer.contracts import ControlCommand, ControlMode
from racer.mavlink_client import MavlinkClient, _pos_type_mask
from racer.recording import Recorder, session_stamp

REF_MASK = 3527  # PyAIPilotExample VELOCITY_POSITION_MASK (X|Y|Z|AX|AY|AZ|YAW|YAW_RATE ignored)


def _f3(v) -> list[float] | None:
    return None if v is None else [float(x) for x in v]


class Channels:
    """Raw-message tap. Keeps the THREE velocity derivations SEPARATE (the whole point):
    mavlink_client._handle overwrites a single state.velocity_ned from BOTH LOCAL_POSITION_NED
    (97 Hz) and ODOMETRY (75 Hz), so reading state would give an interleaved mix. We parse the
    raw messages here so each channel is clean and independently checkable."""

    def __init__(self, fd_window_s: float = 0.12):
        self.fd_window_s = fd_window_s
        self.v_lpn: np.ndarray | None = None
        self.pos_lpn: np.ndarray | None = None
        self.v_odo: np.ndarray | None = None
        self.pos_odo: np.ndarray | None = None
        self.act: list[float] | None = None
        self._pos_hist: deque[tuple[int, np.ndarray]] = deque(maxlen=64)  # (recv_ns, pos) from LPN
        self.n_lpn = self.n_odo = self.n_act = 0

    def reset_counts(self) -> None:
        self.n_lpn = self.n_odo = self.n_act = 0

    def on_msg(self, m) -> None:
        t = m.get_type()
        if t == "LOCAL_POSITION_NED":
            self.pos_lpn = np.array([m.x, m.y, m.z], dtype=np.float64)
            self.v_lpn = np.array([m.vx, m.vy, m.vz], dtype=np.float64)
            self._pos_hist.append((time.monotonic_ns(), self.pos_lpn))
            self.n_lpn += 1
        elif t == "ODOMETRY":
            self.pos_odo = np.array([m.x, m.y, m.z], dtype=np.float64)
            self.v_odo = np.array([m.vx, m.vy, m.vz], dtype=np.float64)
            self.n_odo += 1
        elif t == "ACTUATOR_OUTPUT_STATUS":
            self.act = [float(x) for x in list(m.actuator)[:4]]
            self.n_act += 1

    def v_fd(self) -> np.ndarray | None:
        """Finite-difference velocity from LOCAL_POSITION_NED position over ~fd_window_s.
        A longer baseline than one inter-sample gap keeps null-test noise well under 0.05 m/s."""
        if len(self._pos_hist) < 2:
            return None
        now, p_now = self._pos_hist[-1]
        target = now - int(self.fd_window_s * 1e9)
        old = self._pos_hist[0]
        for sample in self._pos_hist:
            if sample[0] >= target:
                old = sample
                break
        dt = (now - old[0]) / 1e9
        if dt <= 1e-4 or old[0] == now:
            return None
        return (p_now - old[1]) / dt


def _tri_disagreement(vs: list[np.ndarray | None]) -> float | None:
    """Max over axes of the max pairwise |difference| across the available velocity channels."""
    present = [v for v in vs if v is not None]
    if len(present) < 2:
        return None
    stack = np.array(present)
    return float(np.max(np.max(stack, axis=0) - np.min(stack, axis=0)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default="udp:127.0.0.1:14550")
    ap.add_argument("--yaw-mode", choices=["ignore", "hold"], default="ignore",
                    help="ignore = cmd.yaw=None (EXACT reference match, mask 3527); hold = cmd.yaw=yaw0 (mask 2503)")
    ap.add_argument("--keepalive", choices=["none", "timesync"], default="none",
                    help="none = HB+TS both OFF (want ANGLE). timesync = liveness fallback if telemetry dries up.")
    ap.add_argument("--speed", type=float, default=1.0, help="FORWARD speed toward gate 0 (m/s)")
    ap.add_argument("--climb", type=float, default=0.5, help="DESCEND/CLIMB vz magnitude (m/s)")
    ap.add_argument("--phases", default="hover,descend,climb,forward")
    ap.add_argument("--phase-s", type=float, default=3.0)
    ap.add_argument("--rate", type=float, default=25.0)
    ap.add_argument("--null-s", type=float, default=3.0, help="passive null-test / liveness window")
    ap.add_argument("--null-only", action="store_true", help="Phase-1 gate only: null + echo + actuator, NO ladder")
    ap.add_argument("--null-thresh", type=float, default=0.05, help="max |v| for a stationary drone (m/s)")
    ap.add_argument("--tri-abort", type=float, default=0.5, help="gross 3-channel divergence that HALTS a run (m/s)")
    ap.add_argument("--tri-abort-ticks", type=int, default=12, help="consecutive divergent ticks before halt")
    ap.add_argument("--wait-live-s", type=float, default=60.0)
    ap.add_argument("--max-offset-m", type=float, default=12.0)
    ap.add_argument("--max-alt-m", type=float, default=8.0)
    ap.add_argument("--max-tilt-deg", type=float, default=80.0)
    ap.add_argument("--origin-tol-m", type=float, default=1.0)
    ap.add_argument("--label", default="ladder")
    ap.add_argument("--note", default="")
    ap.add_argument("--connect-timeout", type=float, default=10.0)
    args = ap.parse_args()

    c = MavlinkClient(args.endpoint)
    c.send_heartbeats = False                          # NO GCS heartbeat -> stay ANGLE
    c.send_timesync = (args.keepalive == "timesync")   # default OFF; fallback keepalive only
    c.connect(wait_heartbeat=False, timeout_s=args.connect_timeout)

    session = Path("data/runs") / f"{session_stamp()}_{args.label}"
    recorder = Recorder(session)
    recorder.start()
    channels = Channels()

    # Capture the ACTUAL transmitted SET_POSITION_TARGET payload (command echo) by wrapping send.
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
    meta: dict = {
        "probe": "velocity_ladder", "label": args.label, "yaw_mode": args.yaw_mode,
        "keepalive": args.keepalive, "note": args.note, "ned_z_is_down": True,
        "speed_mps": args.speed, "climb_mps": args.climb,
        "safety": {"max_offset_m": args.max_offset_m, "max_alt_m": args.max_alt_m,
                   "max_tilt_deg": args.max_tilt_deg},
    }
    recorder.add_meta(**meta)

    print(f"recording -> {session}")
    print(f">>> TAP-IN  heartbeats=OFF timesync={'ON(fallback)' if c.send_timesync else 'OFF'}  "
          f"yaw_mode={args.yaw_mode}")
    print(">>> TEAMMATE: start the race fresh (Home->Race). Confirm GUI mode = ANGLE. I will NOT arm.\n")

    # ---- ACQUIRE: live + armed + at origin + clock advancing -----------------------------
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
        pos = None if s.position_ned is None else np.round(s.position_ned, 2)
        print(f"\nREFUSE START: need live+armed+origin(|pos|<{args.origin_tol_m}m). "
              f"got pos={pos} armed={s.armed} sim_t={s.sim_time_ns/1e9:.1f}. Restart the race.",
              file=sys.stderr)
        recorder.add_meta(aborted="no_clean_start")
        recorder.close()
        return 1

    origin = np.asarray(s.position_ned, dtype=np.float64).copy()
    yaw0 = float(s.yaw)
    fwd = np.array([np.cos(yaw0), np.sin(yaw0), 0.0])
    sim_t0 = s.sim_time_ns
    rpy0_deg = [float(np.degrees(s.roll)), float(np.degrees(s.pitch)), float(np.degrees(s.yaw))]
    meta.update({"origin_ned": _f3(origin), "yaw0_deg": float(np.degrees(yaw0)),
                 "forward_ned": _f3(fwd), "initial_rpy_deg": rpy0_deg,
                 "sim_time_start_ns": int(sim_t0), "reset_counter": int(s.reset_counter)})
    print(f"  ACQUIRED: pos={np.round(origin,3)} rpy_deg={np.round(rpy0_deg,1)} "
          f"forward={np.round(fwd,2)} reset_ctr={s.reset_counter}")

    # ---- PHASE 1a: NULL TEST (passive) + TELEMETRY LIVENESS --------------------------------
    print(f"\n>>> NULL TEST ({args.null_s:g}s passive, sending NOTHING). TEAMMATE: GUI mode? behavior?")
    channels.reset_counts()
    null_samples = {"v_lpn": [], "v_odo": [], "v_fd": [], "act": []}
    t_end = time.monotonic() + args.null_s
    while time.monotonic() < t_end:
        c.pump()  # recv only (+ timesync if fallback); NO setpoint
        if channels.v_lpn is not None:
            null_samples["v_lpn"].append(float(np.linalg.norm(channels.v_lpn)))
        if channels.v_odo is not None:
            null_samples["v_odo"].append(float(np.linalg.norm(channels.v_odo)))
        vfd = channels.v_fd()
        if vfd is not None:
            null_samples["v_fd"].append(float(np.linalg.norm(vfd)))
        if channels.act is not None:
            null_samples["act"].append(list(channels.act))
        time.sleep(1.0 / args.rate)

    lpn_hz = channels.n_lpn / args.null_s
    odo_hz = channels.n_odo / args.null_s
    act_hz = channels.n_act / args.null_s
    telemetry_live = lpn_hz > 5 and odo_hz > 5

    def _stat(name):
        xs = null_samples[name]
        return {"n": len(xs), "mean": (float(np.mean(xs)) if xs else None),
                "max": (float(np.max(xs)) if xs else None)}
    null_stat = {k: _stat(k) for k in ("v_lpn", "v_odo", "v_fd")}
    act_rest = (np.mean(np.array(null_samples["act"]), axis=0).tolist()
                if null_samples["act"] else None)
    null_pass = telemetry_live and all(
        (null_stat[k]["max"] is not None and null_stat[k]["max"] < args.null_thresh)
        for k in ("v_lpn", "v_odo", "v_fd"))

    print(f"  telemetry rates: LOCAL_POS={lpn_hz:.0f}Hz ODOMETRY={odo_hz:.0f}Hz ACTUATOR={act_hz:.0f}Hz "
          f"-> {'LIVE' if telemetry_live else 'DRY'}")
    for k in ("v_lpn", "v_odo", "v_fd"):
        st = null_stat[k]
        print(f"  null |{k}|: mean={st['mean']} max={st['max']} (n={st['n']})")
    print(f"  actuator rest (motors 0..3): {np.round(act_rest,4).tolist() if act_rest else None}")

    meta.update({"null_test": null_stat, "telemetry_hz": {"local_pos": lpn_hz, "odometry": odo_hz,
                 "actuator": act_hz}, "telemetry_live": telemetry_live,
                 "actuator_rest": act_rest, "null_pass": null_pass})

    if not telemetry_live:
        print("\nHALT: telemetry DRY with this keepalive. Rerun: --keepalive timesync "
              "(teammate: does the GUI stay ANGLE?). This resolves the timesync contradiction.",
              file=sys.stderr)
        recorder.add_meta(**meta, aborted="telemetry_dry")
        recorder.close()
        _dump(session, records, meta)
        return 2
    if not null_pass:
        print("\nHALT (Phase-1 gate FAILED): a stationary drone reads nonzero velocity on a channel "
              "-> biased pipeline. Do NOT interpret the ladder. Report and fix.", file=sys.stderr)
        recorder.add_meta(**meta, aborted="null_fail")
        recorder.close()
        _dump(session, records, meta)
        return 3

    # ---- PHASE 1b: COMMAND ECHO -----------------------------------------------------------
    yaw_cmd = None if args.yaw_mode == "ignore" else yaw0
    expected_mask = _pos_type_mask(None, np.zeros(3), None, yaw_cmd, None)
    echo_calls.clear()
    c.send_command(ControlCommand(mode=ControlMode.VELOCITY, velocity_ned=np.zeros(3), yaw=yaw_cmd))
    c.pump()
    sent = echo_calls[-1] if echo_calls else None
    echo = None
    if sent is not None:
        # positional args: (t, sys, comp, frame, mask, px,py,pz, vx,vy,vz, ax,ay,az, yaw, yaw_rate)
        echo = {"mask": int(sent[4]), "pos": _f3(sent[5:8]), "vel": _f3(sent[8:11]),
                "accel": _f3(sent[11:14]), "yaw": float(sent[14]), "yaw_rate": float(sent[15])}
    mask_matches_ref = (echo is not None and echo["mask"] == REF_MASK) if args.yaw_mode == "ignore" \
        else (echo is not None and echo["mask"] == expected_mask)
    print(f"\n>>> COMMAND ECHO (transmitted): {echo}")
    print(f"  expected mask={expected_mask}  reference mask={REF_MASK}  "
          f"{'== reference (exact match)' if (echo and echo['mask']==REF_MASK) else '(yaw-hold differs from ref by the yaw bit, expected)'}")
    meta.update({"command_echo": echo, "expected_mask": int(expected_mask),
                 "reference_mask": REF_MASK, "echo_mask_ok": bool(echo and echo["mask"] == expected_mask)})

    print(f"\n>>> PHASE-1 GATE: null_pass={null_pass} telemetry_live={telemetry_live} "
          f"echo_mask_ok={meta['echo_mask_ok']} actuator_seen={act_hz>0}")

    if args.null_only:
        print("\n--null-only: Phase-1 gate complete. Inspect before running the ladder.")
        recorder.add_meta(**meta, aborted=None)
        recorder.close()
        _dump(session, records, meta)
        return 0

    # ---- PHASE 2: THE LADDER --------------------------------------------------------------
    phase_defs = {
        "hover": ("HOVER", np.zeros(3)),
        "descend": ("DESCEND", np.array([0.0, 0.0, +args.climb])),   # +z is DOWN
        "climb": ("CLIMB", np.array([0.0, 0.0, -args.climb])),
        "forward": ("FORWARD", args.speed * fwd),
    }
    requested = [p.strip() for p in args.phases.split(",") if p.strip()]
    aborted = None
    phase_summaries: list[dict] = []
    try:
        for key in requested:
            if key not in phase_defs:
                print(f"  (skip unknown phase {key!r})")
                continue
            name, vel_cmd = phase_defs[key]
            print(f"\n>>> PHASE {name} cmd_v={np.round(vel_cmd,2)} ({args.phase_s:g}s). "
                  f"TEAMMATE: GUI mode? climbed/held/moved?")
            p_end = time.monotonic() + args.phase_s
            last_p = 0.0
            diverge_ticks = 0
            phase_recs: list[dict] = []
            while time.monotonic() < p_end:
                c.pump()
                c.send_command(ControlCommand(mode=ControlMode.VELOCITY, velocity_ned=vel_cmd, yaw=yaw_cmd))
                s = c.state
                v_lpn, v_odo, v_fd = channels.v_lpn, channels.v_odo, channels.v_fd()
                pos = channels.pos_lpn if channels.pos_lpn is not None else \
                    (np.asarray(s.position_ned) if s.position_ned is not None else origin)
                rel = np.asarray(pos, dtype=np.float64) - origin
                tilt_deg = float(np.degrees(max(abs(s.roll), abs(s.pitch))))
                tri = _tri_disagreement([v_lpn, v_odo, v_fd])
                rec = {
                    "sim_time_ns": int(s.sim_time_ns), "phase": name,
                    "cmd_v": _f3(vel_cmd), "yaw_cmd": (None if yaw_cmd is None else float(yaw_cmd)),
                    "mask": int(expected_mask),
                    "v_lpn": _f3(v_lpn), "v_odo": _f3(v_odo), "v_fd": _f3(v_fd),
                    "pos_lpn": _f3(channels.pos_lpn), "pos_odo": _f3(channels.pos_odo),
                    "rel_pos": _f3(rel), "rpy_deg": [float(np.degrees(s.roll)),
                    float(np.degrees(s.pitch)), float(np.degrees(s.yaw))],
                    "actuators": (list(channels.act) if channels.act is not None else None),
                    "tri_disagree": tri,
                }
                records.append(rec)
                phase_recs.append(rec)

                # ---- safety ----
                if any(cc["threat_level"] >= 2 for cc in c.collisions):
                    aborted = "collision>=2"
                elif float(np.hypot(rel[0], rel[1])) > args.max_offset_m:
                    aborted = f"offset {np.hypot(rel[0],rel[1]):.1f}m"
                elif abs(float(rel[2])) > args.max_alt_m:
                    aborted = f"alt {rel[2]:+.1f}m"
                elif tilt_deg > args.max_tilt_deg:
                    aborted = f"tilt {tilt_deg:.0f}deg"
                # ---- instrument gate: gross 3-channel divergence ----
                if tri is not None and tri > args.tri_abort:
                    diverge_ticks += 1
                    if diverge_ticks >= args.tri_abort_ticks:
                        aborted = f"triangulation_divergence {tri:.2f}m/s"
                else:
                    diverge_ticks = 0
                if aborted:
                    break

                now = time.monotonic()
                if now - last_p >= 0.3:
                    print(f"   {name:7s} cmd={np.round(vel_cmd,2)} v_lpn={np.round(v_lpn,2) if v_lpn is not None else None} "
                          f"v_odo={np.round(v_odo,2) if v_odo is not None else None} "
                          f"v_fd={np.round(v_fd,2) if v_fd is not None else None} rel={np.round(rel,2)} "
                          f"tilt={tilt_deg:.0f} tri={None if tri is None else round(tri,2)}   ",
                          end="\r", flush=True)
                    last_p = now
                time.sleep(1.0 / args.rate)

            # ---- per-phase summary (steady portion = drop first 0.5s of ticks) ----
            steady = phase_recs[int(0.5 * args.rate):] or phase_recs
            summ = _phase_summary(name, vel_cmd, phase_recs, steady)
            phase_summaries.append(summ)
            print(f"\n   END {name}: achieved v_lpn~{summ['mean_v_lpn']} v_odo~{summ['mean_v_odo']} "
                  f"v_fd~{summ['mean_v_fd']} | z_drift={summ['z_drift_m']}m "
                  f"horiz_drift={summ['horiz_drift_m']}m | tri_max(steady)={summ['tri_max_steady']} "
                  f"| motors_modulating={summ['actuators_modulating']}")
            if aborted:
                print(f"   *** ABORT during {name}: {aborted}")
                break
    except KeyboardInterrupt:
        aborted = "ctrl-c"
    finally:
        if aborted:
            print("\n[safety] force-disarming ...")
            try:
                c.send_heartbeats = True
                c.disarm(force=True)
                c.wait_armed(False, timeout_s=3.0)
            except Exception as exc:
                print(f"  disarm error: {exc}", file=sys.stderr)
        meta.update({"phase_summaries": phase_summaries, "aborted": aborted})
        recorder.add_meta(phase_summaries=phase_summaries, aborted=aborted)
        recorder.close()
        _dump(session, records, meta)

    print(f"\n==== velocity_ladder DONE  yaw_mode={args.yaw_mode}  aborted={aborted} ====")
    print(f"  recording={session}")
    print(f"  json={session/'ladder.json'}")
    print("  Per-phase numbers above are OBSERVATIONS. Map to World A/B/Partial only after the")
    print("  Phase-1 gate passed AND a second independent race reproduces the HOVER crux.")
    return 0


def _phase_summary(name, vel_cmd, recs, steady) -> dict:
    def _mean_axis(key, src):
        xs = [r[key] for r in src if r[key] is not None]
        return None if not xs else [round(float(np.mean([x[i] for x in xs])), 3) for i in range(3)]

    rel0 = next((r["rel_pos"] for r in recs if r["rel_pos"] is not None), None)
    rel1 = next((r["rel_pos"] for r in reversed(recs) if r["rel_pos"] is not None), None)
    z_drift = round(rel1[2] - rel0[2], 3) if (rel0 and rel1) else None
    horiz_drift = round(float(np.hypot(rel1[0] - rel0[0], rel1[1] - rel0[1])), 3) if (rel0 and rel1) else None
    tris = [r["tri_disagree"] for r in steady if r["tri_disagree"] is not None]
    acts = [r["actuators"] for r in recs if r["actuators"] is not None]
    act_spread = None
    if acts:
        a = np.array(acts)
        act_spread = round(float(np.max(np.max(a, axis=0) - np.min(a, axis=0))), 4)
    return {
        "phase": name, "cmd_v": _f3(vel_cmd),
        "mean_v_lpn": _mean_axis("v_lpn", steady), "mean_v_odo": _mean_axis("v_odo", steady),
        "mean_v_fd": _mean_axis("v_fd", steady),
        "z_drift_m": z_drift, "horiz_drift_m": horiz_drift,
        "tri_max_steady": (round(max(tris), 3) if tris else None),
        "actuator_spread": act_spread, "actuators_modulating": (act_spread is not None and act_spread > 1e-4),
        "n_ticks": len(recs),
    }


def _dump(session: Path, records: list[dict], meta: dict) -> None:
    out = {"meta": meta, "records": records}
    (session / "ladder.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
