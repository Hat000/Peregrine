"""First-contact probe: do the VIDEO and TELEMETRY clocks share an epoch, and is the video
stream healthy? (red-team CRIT-1 / CRIT-3 — the delayed-vision + MTU questions.)

Subscribes to BOTH streams over the SPEC interface (JPEG-UDP sec 4.6 + MAVLink sec 4); does
NOT touch Elodin. For every completed frame it compares the frame's capture stamp
(``sim_time_ns``) against the most-recent HIGHRES_IMU ``sim_time_ns`` and reports the offset's
mean / spread, then judges whether the two clocks are the same timeline:

  * If the offset is SMALL + STABLE, video and IMU share a sim clock and the offset is just the
    video pipeline latency (encode + transmit). Then the delayed-vision ring buffer can rewind a
    fix to its capture time on the IMU timeline cleanly -> safe to build for VQ2 speed.
  * If it is LARGE / NEGATIVE / JITTERY, the streams are on independent epochs -> you must
    reconcile via TIMESYNC before fusing vision at speed; do NOT treat frame.sim_time_ns as an
    IMU-timeline capture time.

It also prints stream health from the receiver's metrics (datagram size vs the ~1500 B MTU,
chunks/frame, frames lost to missing chunks) -> the UDP-fragmentation / packet-loss verdict.

SAFE: read-only. It never arms and sends no setpoints (only the >=2 Hz heartbeat MAVLink needs).

Usage:
  python scripts/clock_probe.py [--endpoint udp:127.0.0.1:14550] [--video-port 5600]
        [--seconds 10] [--align-tol-ms 250] [--jitter-tol-ms 30] [--connect-timeout 15]
"""
from __future__ import annotations

import argparse
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from racer.firstcontact import backend_summary, telemetry_summary
from racer.mavlink_client import MavlinkClient
from racer.vision.jpeg_receiver import VIDEO_PORT, JpegUdpReceiver

_MTU = 1500  # standard Ethernet payload; datagrams larger than this get IP-fragmented


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", default="udp:127.0.0.1:14550")
    ap.add_argument("--video-port", type=int, default=VIDEO_PORT)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--align-tol-ms", type=float, default=250.0, help="|mean offset| under this = aligned")
    ap.add_argument("--jitter-tol-ms", type=float, default=30.0, help="offset std under this = stable")
    ap.add_argument("--connect-timeout", type=float, default=15.0)
    args = ap.parse_args()

    # -- MAVLink in a background thread: keeps client.state fresh + heartbeat alive. The
    #    immutable-snapshot DroneState makes client.state safe to read from the main thread.
    client = MavlinkClient(args.endpoint)
    print(f"connecting MAVLink {args.endpoint} (waiting for heartbeat) ...")
    try:
        client.connect(timeout_s=args.connect_timeout)
        print("heartbeat OK.")
    except TimeoutError as exc:
        print(f"WARNING: {exc} -- continuing video-only (no clock comparison).", file=sys.stderr)
        client = None

    stop = threading.Event()
    if client is not None:
        def mav_loop() -> None:
            while not stop.is_set():
                client.pump()
                time.sleep(0.002)
        threading.Thread(target=mav_loop, name="mavlink", daemon=True).start()
        time.sleep(0.5)  # let a few telemetry msgs land
        print(f"  backend:   {backend_summary(client)}")
        print(f"  telemetry: {telemetry_summary(client)}")

    # -- consume frames, sampling the telemetry clock at each frame's arrival -----------
    offsets_ms: list[float] = []        # imu_sim - frame_sim
    recv_ns: list[int] = []             # frame arrival monotonic, for fps
    sim_ns: list[int] = []              # frame capture stamp, for sim-side fps
    n_no_imu = 0
    print(f"\nlistening for video on UDP {args.video_port} for {args.seconds:g}s ...")
    deadline = time.monotonic() + args.seconds
    try:
        with JpegUdpReceiver(port=args.video_port) as rx:
            for frame in rx.frames(max_wait_s=max(3.0, args.seconds + 1.0)):
                recv_ns.append(frame.recv_monotonic_ns)
                sim_ns.append(frame.sim_time_ns)
                if client is not None:
                    imu_sim = client.state.sim_time_ns
                    if imu_sim > 0 and frame.sim_time_ns > 0:
                        offsets_ms.append((imu_sim - frame.sim_time_ns) / 1e6)
                    else:
                        n_no_imu += 1
                if time.monotonic() >= deadline:
                    break
            metrics = rx.metrics
    finally:
        stop.set()
        time.sleep(0.05)

    # -- report -------------------------------------------------------------------------
    print("\n==== clock_probe summary ====")
    n = len(recv_ns)
    if n == 0:
        print("  NO frames received -> is the sim streaming video on this port? "
              "(check --video-port, that the sim is up, and the firewall).")
        return 1
    span_s = (recv_ns[-1] - recv_ns[0]) / 1e9 if n > 1 else 0.0
    arr_fps = (n - 1) / span_s if span_s > 0 else float("nan")
    sim_span_s = (sim_ns[-1] - sim_ns[0]) / 1e9 if (n > 1 and sim_ns[0] > 0) else 0.0
    sim_fps = (n - 1) / sim_span_s if sim_span_s > 0 else float("nan")
    print(f"  frames: {n}  arrival_fps~{arr_fps:.1f}  sim_fps~{sim_fps:.1f}")

    print("  video health (MTU / packet-loss):")
    lost = metrics.partials_evicted
    seen_frames = metrics.frames_completed + lost
    loss_pct = 100.0 * lost / seen_frames if seen_frames else 0.0
    print(f"    datagrams={metrics.datagrams}  completed={metrics.frames_completed}  "
          f"lost_to_missing_chunks={lost} ({loss_pct:.1f}%)  "
          f"decode_fail={metrics.frames_decode_failed}  size_mismatch={metrics.frames_size_mismatch}")
    print(f"    datagram bytes: min={metrics.min_datagram_bytes} max={metrics.max_datagram_bytes} "
          f"(MTU~{_MTU})  chunks/frame max={metrics.max_total_chunks}")
    if metrics.max_datagram_bytes > _MTU:
        print(f"    NOTE: max datagram {metrics.max_datagram_bytes} B > MTU {_MTU} -> the OS IP-fragments "
              "these; a single lost fragment drops the whole datagram. Watch the loss % under load.")
    else:
        print("    OK: every datagram fits within one MTU (no IP fragmentation).")

    print("  clock alignment (video sim_time vs HIGHRES_IMU sim_time):")
    if client is None:
        print("    no MAVLink -> not measured. Re-run with telemetry up.")
        return 0
    if not offsets_ms:
        print(f"    could NOT compare ({n_no_imu} frames had no IMU/video sim-time). "
              "Is HIGHRES_IMU arriving? Are video stamps non-zero?")
        return 0
    mean = statistics.fmean(offsets_ms)
    std = statistics.pstdev(offsets_ms) if len(offsets_ms) > 1 else 0.0
    lo, hi = min(offsets_ms), max(offsets_ms)
    print(f"    offset (imu - frame) ms: mean={mean:+.1f}  std={std:.1f}  min={lo:+.1f}  max={hi:+.1f}  "
          f"n={len(offsets_ms)}")
    aligned = abs(mean) < args.align_tol_ms and std < args.jitter_tol_ms
    if aligned:
        print(f"    => ALIGNED: shared sim clock; offset ~{mean:.0f} ms is the video pipeline latency. "
              "Delayed-vision rewind is well-posed (rewind a fix ~this far on the IMU timeline).")
    else:
        print("    => SUSPECT: the offset is large/jittery -> video & IMU may be on independent epochs. "
              "Reconcile via TIMESYNC before fusing vision at speed; don't treat frame.sim_time_ns as "
              "IMU-timeline capture time. (Thresholds are rough -- eyeball the numbers above.)")
    print("  NOTE: Elodin is NOT the source of truth; this characterised the spec MAVLink+video sim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
