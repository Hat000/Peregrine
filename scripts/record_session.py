"""Record a sim session: raw MAVLink + JPEG video + timestamps -> data/runs/<stamp>_<label>/.

The deterministic course means one good recording replays into many offline iterations
(system-ID, mapping, racing-line fitting, detector auto-labels) -- so record EVERY sim
contact. This also doubles as a lightweight first-contact `msg_audit`: on exit it prints
the MAVLink message-type histogram and which hedge fields (position / velocity / mag /
baro) actually appeared -- the data that resolves risk R1.

Tip: start the simulator FIRST, then this. The video capture self-heals if the stream
appears late or drops, and MAVLink falls back to video-only if no heartbeat arrives.

Usage:
  python scripts/record_session.py [--label NAME] [--seconds N] [--endpoint EP]
                                   [--video-port P] [--no-video] [--no-mavlink]
  --seconds 0 (default) records until Ctrl-C.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from racer.mavlink_client import MavlinkClient
from racer.recording import Recorder, session_stamp
from racer.vision.jpeg_receiver import VIDEO_PORT, JpegUdpReceiver


def _git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=Path(__file__).resolve().parent.parent,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default="session", help="session name suffix")
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = record until Ctrl-C")
    ap.add_argument("--endpoint", default="udp:127.0.0.1:14550")
    ap.add_argument("--video-port", type=int, default=VIDEO_PORT)
    ap.add_argument("--out-dir", default="data/runs")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-mavlink", action="store_true")
    ap.add_argument("--connect-timeout", type=float, default=15.0)
    args = ap.parse_args()

    session = Path(args.out_dir) / f"{session_stamp()}_{args.label}"
    stop = threading.Event()
    type_counts: Counter = Counter()
    hedge = {"position_ned": False, "velocity_ned": False, "mag_body": False, "baro": False}

    recorder = Recorder(session)
    recorder.start()
    recorder.add_meta(
        endpoint=args.endpoint, video_port=args.video_port,
        label=args.label, git_commit=_git_commit(),
    )
    print(f"recording -> {session}")
    threads: list[threading.Thread] = []

    # -- video first: resilient loop that waits for / survives gaps in the stream --
    if not args.no_video:
        def video_loop() -> None:
            while not stop.is_set():
                try:
                    with JpegUdpReceiver(port=args.video_port) as rx:
                        for frame in rx.frames(max_wait_s=5.0):
                            recorder.record_frame(frame)
                            if stop.is_set():
                                break
                except Exception as exc:  # keep recording the other stream
                    print(f"video thread error: {exc}", file=sys.stderr)
                if not stop.is_set():
                    time.sleep(0.5)  # idle timeout / error -> brief pause, then retry
        t = threading.Thread(target=video_loop, name="video", daemon=True)
        t.start()
        threads.append(t)
        print(f"listening for video on UDP {args.video_port}")

    # -- MAVLink: one connect attempt; video-only fallback on timeout --
    client: MavlinkClient | None = None
    if not args.no_mavlink:
        client = MavlinkClient(args.endpoint)
        print(f"connecting MAVLink {args.endpoint} (waiting for heartbeat) ...")
        try:
            client.connect(timeout_s=args.connect_timeout)
            print("MAVLink heartbeat OK.")
        except TimeoutError as exc:
            print(f"WARNING: {exc} -- recording video only.", file=sys.stderr)
            client = None

    if client is not None:
        def on_message(msg) -> None:
            t = msg.get_type()
            if t == "BAD_DATA":
                return
            buf = msg.get_msgbuf()
            if buf:
                recorder.record_mavlink(bytes(buf))
                type_counts[t] += 1
        client.on_message = on_message

        def mav_loop() -> None:
            while not stop.is_set():
                client.pump()  # also emits the >=2Hz heartbeat the sim needs
                s = client.state
                if s.position_ned is not None:
                    hedge["position_ned"] = True
                if s.velocity_ned is not None:
                    hedge["velocity_ned"] = True
                if s.mag_body is not None:
                    hedge["mag_body"] = True
                if s.baro_pressure_hpa is not None:
                    hedge["baro"] = True
                time.sleep(0.002)
        t = threading.Thread(target=mav_loop, name="mavlink", daemon=True)
        t.start()
        threads.append(t)

    if not threads:
        print("nothing to record (both streams disabled/unavailable).", file=sys.stderr)
        recorder.close()
        return 1

    # -- main wait loop: live counters; stop on duration or Ctrl-C --
    t_end = time.monotonic() + args.seconds if args.seconds > 0 else None
    last = 0.0
    try:
        while not stop.is_set():
            if t_end is not None and time.monotonic() >= t_end:
                break
            now = time.monotonic()
            if now - last >= 1.0:
                print(
                    f"  mav={recorder.n_mavlink}  frames={recorder.n_frames}  "
                    f"dropped={recorder.n_dropped}   ",
                    end="\r",
                    flush=True,
                )
                last = now
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nstopping (Ctrl-C) ...")
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=6.0)
        recorder.add_meta(
            mavlink_type_counts=dict(type_counts),
            hedge_fields_seen=dict(hedge),
        )
        recorder.close()

    # -- first-contact summary --
    print()
    print(
        f"done: {recorder.n_mavlink} mavlink records, {recorder.n_frames} frames, "
        f"{recorder.n_dropped} dropped -> {session}"
    )
    if type_counts:
        print("MAVLink message types seen:")
        for name, count in type_counts.most_common():
            print(f"  {name:24s} {count}")
    print("hedge fields present in telemetry (resolves R1):")
    for key, seen in hedge.items():
        print(f"  {key:14s} {'YES' if seen else 'no'}")
    if client is not None and client.unknown_msg_types:
        print(f"unparsed message types: {sorted(client.unknown_msg_types)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
