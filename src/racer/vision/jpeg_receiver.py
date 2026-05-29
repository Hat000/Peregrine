"""UDP JPEG vision stream receiver.

The simulator streams a 30 Hz, 640x360 video as JPEG frames fragmented across
multiple UDP datagrams on port 5600. Each datagram has a 24-byte little-endian
metadata header followed by a JPEG slice; chunks are reassembled by frame_id.

Spec ref: VADR-TS-002 sec 4.6.
"""
from __future__ import annotations

import socket
import struct
import time
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
        while True:
            # Evict first, unconditionally: every early-return below would otherwise skip
            # it and leak stale partials under packet loss / decode failures. [review 4A]
            self._evict_stale()
            try:
                data, _ = self._sock.recvfrom(65535)
            except BlockingIOError:
                if deadline is not None and time.monotonic() >= deadline:
                    return
                time.sleep(0.001)
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
        if len(data) < HEADER_SIZE:
            return None
        (
            frame_id,
            chunk_id,
            total_chunks,
            jpeg_size,
            payload_size,
            sim_time_ns,
        ) = struct.unpack_from(HEADER_FMT, data, 0)
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
            return None
        img = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        return Frame(
            frame_id=frame_id,
            sim_time_ns=partial.sim_time_ns,
            image_bgr=img,
            recv_monotonic_ns=time.monotonic_ns(),
        )

    def _evict_stale(self) -> None:
        cutoff = time.monotonic() - self.stale_after_s
        stale = [fid for fid, p in self._partials.items() if p.first_seen_monotonic < cutoff]
        for fid in stale:
            self._partials.pop(fid, None)
