"""Session recorder + reader — capture raw MAVLink + JPEG video to disk for replay.

The course is deterministic, so one recorded sim contact replays into many offline
iterations (system-ID, mapping, racing-line fitting, detector auto-labels). This module
owns the on-disk format and keeps disk I/O OFF the hot receive/control loop via a
background writer thread fed through a queue.

Session layout (``data/runs/<stamp>_<label>/``):
  ``mavlink.tlog``      standard pymavlink telemetry log: per record an 8-byte big-endian
                        uint64 UNIX-epoch-microsecond timestamp + the raw MAVLink2 frame.
                        Re-readable by pymavlink / MAVExplorer / QGC; the sim master clock
                        (HIGHRES_IMU.time_usec) lives inside the frames, so it is preserved.
  ``video.bin``         raw JPEG frames concatenated — the exact bytes the sim sent.
  ``video_index.jsonl`` one JSON object per frame: frame_id, sim_time_ns, recv_monotonic_ns,
                        and the byte offset + length into ``video.bin`` (seekable).
  ``meta.json``         session metadata + a monotonic<->UNIX clock bridge + close-out stats.

Clocks: every record is stamped at RECEIVE time (``time.monotonic_ns``), not write time,
so queue latency never skews the log. ``meta.json`` stores a (unix, monotonic) pair sampled
at start, so the video index (monotonic), the tlog (synthesised epoch) and the in-frame sim
clock are all mutually correlatable. See the clock TODO in :mod:`racer.contracts`.

Threading: producers (the MAVLink pump thread, the video thread) call ``record_*`` which
only enqueues; a single daemon writer thread performs all disk writes. Producers must stop
before :meth:`Recorder.close`. Data contracts: :mod:`racer.contracts`.
"""
from __future__ import annotations

import json
import os
import struct
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from queue import Full, Queue

os.environ.setdefault("MAVLINK20", "1")

import cv2
import numpy as np

from racer.contracts import Frame

SCHEMA = "racer.recording/v1"
# pymavlink tlog record prefix: uint64 big-endian microseconds since the UNIX epoch.
_TLOG_STAMP = struct.Struct(">Q")


def session_stamp() -> str:
    """UTC timestamp suitable for a session directory name (sorts chronologically)."""
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


class Recorder:
    """Append-only recorder for one sim session. Use as a context manager or
    ``start()`` / ``close()`` explicitly. ``record_mavlink`` / ``record_frame`` are
    non-blocking (enqueue only); a background thread does the disk writes.
    """

    def __init__(self, session_dir: Path | str, *, queue_max: int = 20_000):
        self.dir = Path(session_dir)
        self._q: Queue = Queue(maxsize=queue_max)
        self._writer: threading.Thread | None = None
        self._t0_unix_ns = 0
        self._t0_mono_ns = 0
        self._extra_meta: dict = {}
        self._meta_lock = threading.Lock()
        # writer-thread-only counters (no lock needed: single writer)
        self._n_mav = 0
        self._n_frames = 0
        self._mav_bytes = 0
        self._video_bytes = 0
        self._video_offset = 0
        # producer-side (multiple threads) -> guarded
        self._n_dropped = 0
        self._drop_lock = threading.Lock()
        self._started = False
        self._closed = False
        self._tlog = None
        self._video = None
        self._index = None

    # -- lifecycle ----------------------------------------------------------
    def __enter__(self) -> Recorder:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def start(self) -> None:
        if self._started:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        self._tlog = open(self.dir / "mavlink.tlog", "wb")
        self._video = open(self.dir / "video.bin", "wb")
        self._index = open(self.dir / "video_index.jsonl", "w", encoding="utf-8")
        # Sample the two clocks back-to-back: this pair bridges monotonic<->epoch.
        self._t0_unix_ns = time.time_ns()
        self._t0_mono_ns = time.monotonic_ns()
        self._started = True
        self._write_meta(closing=False)
        self._writer = threading.Thread(
            target=self._writer_loop, name="recorder-writer", daemon=True
        )
        self._writer.start()

    def close(self) -> None:
        """Flush + close. Idempotent. Producers must have stopped calling ``record_*``."""
        if not self._started or self._closed:
            return
        self._closed = True
        self._q.put(None)  # sentinel; blocking put drains past maxsize as the writer empties
        if self._writer is not None:
            self._writer.join()
        for f in (self._tlog, self._video, self._index):
            if f is None:
                continue
            try:
                f.flush()
                os.fsync(f.fileno())
            except OSError:
                pass
            f.close()
        self._write_meta(closing=True)

    # -- recording API (non-blocking) ---------------------------------------
    def record_mavlink(self, raw: bytes, recv_monotonic_ns: int | None = None) -> None:
        """Enqueue one raw MAVLink frame (e.g. ``msg.get_msgbuf()``)."""
        if not raw:
            return
        recv = time.monotonic_ns() if recv_monotonic_ns is None else recv_monotonic_ns
        self._put(("mav", bytes(raw), int(recv)))

    def record_frame(self, frame: Frame) -> None:
        """Enqueue one camera frame. Uses ``frame.jpeg_bytes`` (bit-exact) when present,
        else re-encodes ``frame.image_bgr`` to JPEG as a fallback."""
        jpeg = frame.jpeg_bytes
        if jpeg is None:
            if frame.image_bgr is None:
                return
            ok, buf = cv2.imencode(".jpg", frame.image_bgr)
            if not ok:
                return
            jpeg = buf.tobytes()
        recv = int(frame.recv_monotonic_ns) or time.monotonic_ns()
        self._put(("frame", int(frame.frame_id), int(frame.sim_time_ns), recv, bytes(jpeg)))

    def add_meta(self, **kwargs) -> None:
        """Merge extra key/values into meta.json (e.g. endpoint, git commit, type counts)."""
        with self._meta_lock:
            self._extra_meta.update(kwargs)

    # -- stats --------------------------------------------------------------
    @property
    def n_mavlink(self) -> int:
        return self._n_mav

    @property
    def n_frames(self) -> int:
        return self._n_frames

    @property
    def n_dropped(self) -> int:
        with self._drop_lock:
            return self._n_dropped

    # -- internals ----------------------------------------------------------
    def _put(self, item: tuple) -> None:
        if not self._started or self._closed:
            return
        try:
            self._q.put_nowait(item)
        except Full:
            with self._drop_lock:
                self._n_dropped += 1

    def _mono_to_unix_us(self, mono_ns: int) -> int:
        return (self._t0_unix_ns + (mono_ns - self._t0_mono_ns)) // 1000

    def _writer_loop(self) -> None:
        while True:
            item = self._q.get()
            try:
                if item is None:
                    return
                kind = item[0]
                if kind == "mav":
                    _, raw, recv = item
                    self._tlog.write(_TLOG_STAMP.pack(self._mono_to_unix_us(recv)))
                    self._tlog.write(raw)
                    self._n_mav += 1
                    self._mav_bytes += len(raw)
                elif kind == "frame":
                    _, fid, sim_ns, recv, jpeg = item
                    off = self._video_offset
                    self._video.write(jpeg)
                    self._video_offset += len(jpeg)
                    self._index.write(
                        json.dumps(
                            {
                                "frame_id": fid,
                                "sim_time_ns": sim_ns,
                                "recv_monotonic_ns": recv,
                                "offset": off,
                                "length": len(jpeg),
                            }
                        )
                        + "\n"
                    )
                    self._n_frames += 1
                    self._video_bytes += len(jpeg)
            finally:
                self._q.task_done()

    def _write_meta(self, *, closing: bool) -> None:
        meta = {
            "schema": SCHEMA,
            "t0_unix_ns": self._t0_unix_ns,
            "t0_monotonic_ns": self._t0_mono_ns,
            "created_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._t0_unix_ns / 1e9)
            ),
        }
        with self._meta_lock:
            meta.update(self._extra_meta)
        if closing:
            meta.update(
                {
                    "duration_s": (time.monotonic_ns() - self._t0_mono_ns) / 1e9,
                    "mavlink_records": self._n_mav,
                    "mavlink_bytes": self._mav_bytes,
                    "video_frames": self._n_frames,
                    "video_bytes": self._video_bytes,
                    "dropped": self.n_dropped,
                }
            )
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


class RecordingReader:
    """Read back a recorded session: parsed MAVLink, raw JPEG, or decoded Frames."""

    def __init__(self, session_dir: Path | str):
        self.dir = Path(session_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))

    def iter_mavlink(self) -> Iterator:
        """Yield parsed pymavlink messages from ``mavlink.tlog`` (standard tlog via mavutil)."""
        from pymavlink import mavutil

        conn = mavutil.mavlink_connection(str(self.dir / "mavlink.tlog"))
        try:
            while True:
                msg = conn.recv_match(blocking=False)
                if msg is None:
                    break
                if msg.get_type() == "BAD_DATA":
                    continue
                yield msg
        finally:
            conn.close()

    def iter_video_index(self) -> Iterator[dict]:
        path = self.dir / "video_index.jsonl"
        if not path.exists():
            return
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def iter_jpeg(self) -> Iterator[tuple[dict, bytes]]:
        """Yield ``(index_entry, raw_jpeg_bytes)`` without decoding (cheap; for offline
        detector runs / re-encoding)."""
        vid = self.dir / "video.bin"
        with open(vid, "rb") as f:
            for entry in self.iter_video_index():
                f.seek(entry["offset"])
                yield entry, f.read(entry["length"])

    def frames(self) -> Iterator[Frame]:
        """Yield decoded :class:`Frame` objects in recorded order."""
        for entry, jpeg in self.iter_jpeg():
            img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            yield Frame(
                frame_id=entry["frame_id"],
                sim_time_ns=entry["sim_time_ns"],
                image_bgr=img,
                recv_monotonic_ns=entry["recv_monotonic_ns"],
                jpeg_bytes=jpeg,
            )
