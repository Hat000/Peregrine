"""UDP JPEG vision stream receiver.

The simulator streams a 30 Hz, 640x360 video as JPEG frames fragmented across
multiple UDP datagrams on port 5600. Each datagram has a 24-byte little-endian
metadata header followed by a JPEG slice; chunks are reassembled by frame_id.

DEDUP [first contact 2026-06-02]: the sim re-sends every frame's chunk set ~14x (the
"~395 fps" illusion; the true frame rate is ~28.6 fps). We emit each frame_id exactly
ONCE — re-sends after a frame completes are dropped — so perception runs once per real
frame. Crucially the suppression triggers only AFTER completion, so re-sent copies that
arrive *before* a frame is whole still fill chunks lost to UDP drops (free redundancy).

Spec ref: VADR-TS-002 sec 4.6.
"""
from __future__ import annotations

import select
import socket
import struct
import time
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass

import cv2
import numpy as np

from racer.contracts import Frame

VIDEO_PORT = 5600
# Header: frame_id (u32), chunk_id (u16), total_chunks (u16), jpeg_size (u32),
#         payload_size (u32), sim_time_ns (u64). Little-endian.
HEADER_FMT = "<IHHIIQ"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
assert HEADER_SIZE == 24, "Spec sec 4.6: header is 24 bytes"


@dataclass
class _PartialFrame:
    total_chunks: int
    jpeg_size: int
    sim_time_ns: int
    chunks: dict[int, bytes]
    first_seen_monotonic: float


@dataclass
class ReceiverMetrics:
    """Stream-health counters for first-contact MTU / packet-loss diagnostics [red-team].

    ``max_datagram_bytes`` vs the ~1500 B Ethernet MTU answers "does the sim send chunks the
    OS must IP-fragment?"; ``max_total_chunks`` = how many datagrams per frame; the gap between
    ``frames_completed`` and ``partials_evicted`` = frames lost to missing chunks (UDP drops).
    """

    datagrams: int = 0
    short_datagrams: int = 0
    frames_completed: int = 0
    frames_decode_failed: int = 0
    frames_size_mismatch: int = 0
    partials_evicted: int = 0          # incomplete frames dropped as stale = lost chunk(s)
    duplicate_datagrams: int = 0       # datagrams for an already-emitted frame_id (sim's ~14x re-send)
    min_datagram_bytes: int = 0
    max_datagram_bytes: int = 0
    max_total_chunks: int = 0


class JpegUdpReceiver:
    """Reassembles chunked JPEG frames from the simulator vision stream."""

    def __init__(
        self,
        bind_host: str = "0.0.0.0",
        port: int = VIDEO_PORT,
        stale_after_s: float = 0.5,
    ):
        self.bind_host = bind_host
        self.port = port
        self.stale_after_s = stale_after_s
        self._sock: socket.socket | None = None
        self._partials: dict[int, _PartialFrame] = {}
        # FIFO set of recently-emitted frame_ids, to drop the sim's ~14x re-sends. Capped well
        # above any reorder/re-send window (512 ids ~= 18 s at 28.6 fps); memory is trivial.
        self._completed: OrderedDict[int, None] = OrderedDict()
        self._completed_cap = 512
        self.metrics = ReceiverMetrics()

    def __enter__(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        self._sock.bind((self.bind_host, self.port))
        self._sock.setblocking(False)
        return self

    def __exit__(self, *exc):
        if self._sock is not None:
            self._sock.close()
        self._sock = None

    def frames(self, max_wait_s: float | None = None) -> Iterator[Frame]:
        """Yield reassembled frames as they complete.

        ``max_wait_s`` is an IDLE timeout: if no datagram arrives for that long the
        generator returns, so a downed sim or blocked port fails gracefully instead of
        hanging forever [review 4B]. It resets on every datagram, so a healthy (if slow)
        stream is never cut off mid-capture. ``None`` waits indefinitely.
        """
        assert self._sock is not None
        deadline = None if max_wait_s is None else time.monotonic() + max_wait_s
        # Cap each kernel wait so _evict_stale still runs (and the idle deadline is honoured)
        # during quiet stretches, even when no datagram arrives.
        poll_s = self.stale_after_s if max_wait_s is None else min(self.stale_after_s, max_wait_s)
        poll_s = max(poll_s, 1e-3)
        while True:
            # Evict first, unconditionally: every early-return below would otherwise skip
            # it and leak stale partials under packet loss / decode failures. [review 4A]
            self._evict_stale()
            # Block in the kernel until the socket is readable or poll_s elapses. Replaces a
            # time.sleep(0.001) busy-poll whose ~15 ms granularity on Windows added frame
            # latency + jitter; select wakes the instant a datagram lands. [red-team 2026-05-30]
            ready, _, _ = select.select([self._sock], [], [], poll_s)
            if not ready:
                if deadline is not None and time.monotonic() >= deadline:
                    return
                continue
            try:
                data, _ = self._sock.recvfrom(65535)
            except BlockingIOError:
                continue
            if deadline is not None:
                deadline = time.monotonic() + max_wait_s
            frame = self._ingest(data)
            if frame is not None:
                yield frame

    def _ingest(self, data: bytes) -> Frame | None:
        """Add one datagram to its partial frame; return a Frame iff it completes + decodes.

        Pure of socket I/O and eviction, so it is directly unit-testable: feed crafted
        datagrams, observe the returned Frame / None and ``self._partials``.
        """
        m = self.metrics
        m.datagrams += 1
        dn = len(data)
        m.min_datagram_bytes = dn if m.max_datagram_bytes == 0 else min(m.min_datagram_bytes, dn)
        m.max_datagram_bytes = max(m.max_datagram_bytes, dn)
        if len(data) < HEADER_SIZE:
            m.short_datagrams += 1
            return None
        (
            frame_id,
            chunk_id,
            total_chunks,
            jpeg_size,
            payload_size,
            sim_time_ns,
        ) = struct.unpack_from(HEADER_FMT, data, 0)
        m.max_total_chunks = max(m.max_total_chunks, total_chunks)
        # Drop the sim's re-sends of a frame we've already emitted (dedup). Checked only
        # against COMPLETED frame_ids, so re-sent copies of a still-incomplete frame are not
        # suppressed here — they fall through and may supply chunks lost to UDP drops.
        if frame_id in self._completed:
            m.duplicate_datagrams += 1
            return None
        payload = data[HEADER_SIZE:HEADER_SIZE + payload_size]
        partial = self._partials.get(frame_id)
        if partial is None:
            partial = _PartialFrame(
                total_chunks=total_chunks,
                jpeg_size=jpeg_size,
                sim_time_ns=sim_time_ns,
                chunks={},
                first_seen_monotonic=time.monotonic(),
            )
            self._partials[frame_id] = partial
        partial.chunks[chunk_id] = payload
        if len(partial.chunks) != partial.total_chunks:
            return None
        jpeg_bytes = b"".join(partial.chunks[i] for i in range(partial.total_chunks))
        self._partials.pop(frame_id, None)
        if len(jpeg_bytes) != partial.jpeg_size:
            m.frames_size_mismatch += 1
            return None
        img = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            m.frames_decode_failed += 1
            return None
        m.frames_completed += 1
        # Mark emitted so subsequent re-sends of this frame_id are dropped (FIFO-capped).
        self._completed[frame_id] = None
        if len(self._completed) > self._completed_cap:
            self._completed.popitem(last=False)
        return Frame(
            frame_id=frame_id,
            sim_time_ns=partial.sim_time_ns,
            image_bgr=img,
            recv_monotonic_ns=time.monotonic_ns(),
            jpeg_bytes=jpeg_bytes,   # keep the raw bytes for bit-exact recording/replay
        )

    def _evict_stale(self) -> None:
        cutoff = time.monotonic() - self.stale_after_s
        stale = [fid for fid, p in self._partials.items() if p.first_seen_monotonic < cutoff]
        for fid in stale:
            self._partials.pop(fid, None)
        self.metrics.partials_evicted += len(stale)
