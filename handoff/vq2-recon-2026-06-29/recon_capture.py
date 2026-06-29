"""VQ2 load-day recon: passive wire inventory + raw camera-frame capture.

Fallback for the (nonexistent) scripts/vq2_loadday kit. Binds BOTH the sim's MAVLink
endpoint (udpin:127.0.0.1:14550) and the JPEG-UDP video port (5600) in one process and
records, for a fixed window:

  WIRE  -- every inbound MAVLink message type: count, observed rate (count/window),
           and a field dump of the first instance. Explicit PRESENT/ABSENT verdict for
           the target set {LOCAL_POSITION_NED, ODOMETRY, ATTITUDE, GATE_INFO/TRACK_INFO,
           HIGHRES_IMU (+ gyro/mag sub-fields), TIMESYNC, HEARTBEAT}. Written to a JSON.
  FRAMES -- saves every Nth decoded 640x360 frame as PNG + keeps the raw JPEG, plus a
           per-frame stat row (mean/median brightness, per-channel means, red-dominance,
           saturated-pixel counts) used downstream to pick representative frames + assess
           glow bloom.

Keepalive mirrors the reference client: TIMESYNC@10Hz, NO GCS heartbeat (the heartbeat
appears to flip the sim into ACRO; we are observing, not flying, so stay quiet). Optional
--control nudges a gentle forward velocity setpoint stream to induce motion-blur frames.

Usage:
  python recon_capture.py --mode training --seconds 50 --outdir <dir>
  python recon_capture.py --mode training --seconds 50 --outdir <dir> --control fwd
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import threading
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MAVLINK20", "1")

import cv2
import numpy as np
from pymavlink import mavutil

import sys
# Find the repo src/ (worktree root is parents[2]); fall back to the main clone.
for _cand in (Path(__file__).resolve().parents[2] / "src", Path("C:/Users/Shadow/Peregrine/src")):
    if (_cand / "racer" / "vision" / "jpeg_receiver.py").exists():
        sys.path.insert(0, str(_cand))
        break
from racer.vision.jpeg_receiver import JpegUdpReceiver  # noqa: E402

TARGET_TYPES = [
    "HEARTBEAT", "TIMESYNC", "HIGHRES_IMU", "ATTITUDE",
    "LOCAL_POSITION_NED", "ODOMETRY", "GLOBAL_POSITION_INT",
    "GATE_INFO", "TRACK_INFO", "ENCAPSULATED_DATA", "DATA_TRANSMISSION_HANDSHAKE",
    "COLLISION", "ACTUATOR_OUTPUT_STATUS", "STATUSTEXT", "COMMAND_ACK",
]


def frame_stats(img_bgr: np.ndarray) -> dict:
    """Appearance + bloom-relevant stats for one BGR frame."""
    b, g, r = img_bgr[..., 0].astype(np.int32), img_bgr[..., 1].astype(np.int32), img_bgr[..., 2].astype(np.int32)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    total = img_bgr.shape[0] * img_bgr.shape[1]
    # red-glow proxy: pixels where red strongly dominates green+blue and is bright
    red_glow = ((r > 120) & (r - g > 60) & (r - b > 60))
    sat_red = (r >= 250)
    hist = np.histogram(gray, bins=16, range=(0, 256))[0]
    return {
        "mean_gray": float(gray.mean()),
        "median_gray": float(np.median(gray)),
        "p95_gray": float(np.percentile(gray, 95)),
        "frac_dark_lt32": float((gray < 32).sum() / total),
        "frac_bright_gt224": float((gray > 224).sum() / total),
        "mean_b": float(b.mean()), "mean_g": float(g.mean()), "mean_r": float(r.mean()),
        "red_glow_px": int(red_glow.sum()),
        "red_glow_frac": float(red_glow.sum() / total),
        "sat_red_px": int(sat_red.sum()),
        "gray_hist16": [int(x) for x in hist],
    }


class WireRecorder:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.counts: dict[str, int] = defaultdict(int)
        self.first_field_dump: dict[str, dict] = {}
        self.first_recv: dict[str, float] = {}
        self.last_recv: dict[str, float] = {}
        self.statustexts: list[str] = []
        self.highres_imu_subfields: dict | None = None
        self.heartbeat_meta: dict | None = None
        self.stop = False

    def run(self, seconds: float, keepalive: str = "timesync", control: str = "none"):
        conn = mavutil.mavlink_connection(self.endpoint, source_system=255, autoreconnect=True)
        t0 = time.monotonic()
        last_timesync = 0.0
        last_hb = 0.0
        last_ctrl = 0.0
        armed_sent = False
        target_learned = False
        VEL_MASK = ((1 << 0) | (1 << 1) | (1 << 2) | (1 << 6) | (1 << 7) | (1 << 8) | (1 << 11))
        while not self.stop and (time.monotonic() - t0) < seconds:
            msg = conn.recv_match(blocking=True, timeout=0.2)
            now = time.monotonic()
            if conn.target_system != 0:
                target_learned = True
            # OPTIONAL gentle motion (same socket that learned the sim peer). Best-effort:
            # if the mode ignores velocity setpoints the drone stays put (frames still useful).
            if control == "fwd" and target_learned:
                if not armed_sent:
                    conn.mav.command_long_send(
                        conn.target_system, conn.target_component,
                        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
                    armed_sent = True
                if now - last_ctrl >= 0.02:
                    conn.mav.set_position_target_local_ned_send(
                        0, conn.target_system, conn.target_component,
                        mavutil.mavlink.MAV_FRAME_LOCAL_NED, VEL_MASK,
                        0, 0, 0, 2.0, 0.0, -0.3, 0, 0, 0, 0.0, 0.0)
                    last_ctrl = now
            # keepalive regime: 'timesync' mirrors the reference client (no GCS heartbeat);
            # 'heartbeat' sends a 2Hz GCS heartbeat to test whether position/attitude telemetry
            # is gated on it; 'both' sends each.
            if target_learned and keepalive in ("timesync", "both") and now - last_timesync >= 0.1:
                try:
                    conn.mav.timesync_send(int(time.time_ns()), 0)
                except Exception:
                    pass
                last_timesync = now
            if target_learned and keepalive in ("heartbeat", "both") and now - last_hb >= 0.5:
                try:
                    conn.mav.heartbeat_send(
                        mavutil.mavlink.MAV_TYPE_GCS,
                        mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
                except Exception:
                    pass
                last_hb = now
            if msg is None:
                continue
            t = msg.get_type()
            if t == "BAD_DATA":
                self.counts["BAD_DATA"] += 1
                continue
            self.counts[t] += 1
            if t not in self.first_recv:
                self.first_recv[t] = now
                try:
                    d = msg.to_dict()
                    d.pop("mavpackettype", None)
                    # keep field dump JSON-safe + compact
                    self.first_field_dump[t] = {
                        k: (list(v) if isinstance(v, (list, tuple)) else v)
                        for k, v in d.items()
                    }
                except Exception:
                    self.first_field_dump[t] = {}
            self.last_recv[t] = now
            if t == "HIGHRES_IMU" and self.highres_imu_subfields is None:
                self.highres_imu_subfields = {
                    "xacc": float(msg.xacc), "yacc": float(msg.yacc), "zacc": float(msg.zacc),
                    "xgyro": float(msg.xgyro), "ygyro": float(msg.ygyro), "zgyro": float(msg.zgyro),
                    "xmag": float(msg.xmag), "ymag": float(msg.ymag), "zmag": float(msg.zmag),
                    "abs_pressure": float(msg.abs_pressure),
                    "fields_updated_bitmask": int(getattr(msg, "fields_updated", 0)),
                    "time_usec": int(msg.time_usec),
                }
            if t == "HEARTBEAT" and self.heartbeat_meta is None:
                self.heartbeat_meta = {
                    "autopilot": int(msg.autopilot), "type": int(msg.type),
                    "base_mode": int(msg.base_mode), "custom_mode": int(msg.custom_mode),
                    "system_status": int(getattr(msg, "system_status", 0)),
                }
            if t == "STATUSTEXT":
                txt = msg.text
                if isinstance(txt, (bytes, bytearray)):
                    txt = bytes(txt).decode("ascii", "replace")
                self.statustexts.append(str(txt).rstrip("\x00").strip())
        conn.close()

    def summary(self, window_s: float) -> dict:
        wire = {}
        for t in sorted(self.counts):
            n = self.counts[t]
            span = max(self.last_recv.get(t, 0) - self.first_recv.get(t, 0), 1e-9)
            rate = (n - 1) / span if n > 1 else 0.0
            wire[t] = {
                "count": n,
                "rate_hz": round(rate, 2),
                "first_fields": self.first_field_dump.get(t, {}),
            }
        target = {}
        for t in TARGET_TYPES:
            target[t] = "PRESENT" if self.counts.get(t, 0) > 0 else "ABSENT"
        return {
            "window_s": window_s,
            "target_verdict": target,
            "highres_imu_subfields": self.highres_imu_subfields,
            "heartbeat_meta": self.heartbeat_meta,
            "statustexts": self.statustexts[:50],
            "all_types": wire,
        }


class FrameRecorder:
    def __init__(self, outdir: Path, save_every: int):
        self.outdir = outdir
        self.save_every = save_every
        self.stats: list[dict] = []
        self.saved: list[str] = []
        self.stop = False

    def run(self, seconds: float):
        framedir = self.outdir / "frames"
        framedir.mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()
        idx = 0
        with JpegUdpReceiver() as rx:
            for frame in rx.frames(max_wait_s=3.0):
                if self.stop or (time.monotonic() - t0) >= seconds:
                    break
                st = frame_stats(frame.image_bgr)
                st["frame_id"] = int(frame.frame_id)
                st["sim_time_ns"] = int(frame.sim_time_ns)
                st["t_rel"] = round(time.monotonic() - t0, 3)
                st["idx"] = idx
                if idx % self.save_every == 0:
                    name = f"raw_{idx:04d}_fid{frame.frame_id}.png"
                    cv2.imwrite(str(framedir / name), frame.image_bgr)
                    # also keep the bit-exact JPEG
                    (framedir / f"raw_{idx:04d}_fid{frame.frame_id}.jpg").write_bytes(frame.jpeg_bytes)
                    st["saved_as"] = name
                    self.saved.append(name)
                self.stats.append(st)
                idx += 1
            self.metrics = {
                "frames_completed": rx.metrics.frames_completed,
                "datagrams": rx.metrics.datagrams,
                "duplicate_datagrams": rx.metrics.duplicate_datagrams,
                "partials_evicted": rx.metrics.partials_evicted,
                "max_datagram_bytes": rx.metrics.max_datagram_bytes,
                "max_total_chunks": rx.metrics.max_total_chunks,
                "min_datagram_bytes": rx.metrics.min_datagram_bytes,
            }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="training")
    ap.add_argument("--seconds", type=float, default=50.0)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--save-every", type=int, default=15)
    ap.add_argument("--endpoint", default="udpin:127.0.0.1:14550")
    ap.add_argument("--control", default="none", choices=["none", "fwd"])
    ap.add_argument("--keepalive", default="timesync", choices=["timesync", "heartbeat", "both"])
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    wire = WireRecorder(args.endpoint)
    frames = FrameRecorder(outdir, args.save_every)
    stop_flag = [False]

    tw = threading.Thread(target=wire.run, args=(args.seconds, args.keepalive, args.control), daemon=True)
    tf = threading.Thread(target=frames.run, args=(args.seconds,), daemon=True)
    tw.start()
    tf.start()

    tw.join(timeout=args.seconds + 10)
    tf.join(timeout=args.seconds + 10)
    stop_flag[0] = True

    summary = wire.summary(args.seconds)
    summary["mode"] = args.mode
    summary["control"] = args.control
    summary["video_metrics"] = getattr(frames, "metrics", {})
    summary["frames_saved"] = frames.saved
    (outdir / f"wire_inventory_{args.mode}.json").write_text(json.dumps(summary, indent=2))
    (outdir / f"frame_stats_{args.mode}.json").write_text(json.dumps(frames.stats, indent=2))

    print(f"=== WIRE INVENTORY ({args.mode}, {args.seconds}s) ===")
    for t, v in summary["target_verdict"].items():
        rate = summary["all_types"].get(t, {}).get("rate_hz", 0)
        print(f"  {t:30s} {v:8s} rate={rate}")
    print("--- all observed types ---")
    for t, v in summary["all_types"].items():
        print(f"  {t:30s} n={v['count']:6d} rate={v['rate_hz']}")
    print(f"frames decoded={summary['video_metrics'].get('frames_completed')} saved={len(frames.saved)}")
    if summary["statustexts"]:
        print("STATUSTEXT:", " | ".join(summary["statustexts"][:10]))


if __name__ == "__main__":
    main()
