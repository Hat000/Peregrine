"""Tests for the JPEG-UDP reassembler. ``_ingest`` is exercised directly (no socket); the
eviction logic is tested standalone. Together these cover the leak fix [review 4A]."""
import socket
import struct
import time

import cv2
import numpy as np

from racer.vision.jpeg_receiver import (
    HEADER_FMT,
    JpegUdpReceiver,
    _PartialFrame,
)


def _datagram(frame_id, chunk_id, total_chunks, jpeg_size, payload, sim_time_ns):
    header = struct.pack(HEADER_FMT, frame_id, chunk_id, total_chunks, jpeg_size,
                         len(payload), sim_time_ns)
    return header + payload


def test_ingest_reassembles_valid_jpeg_across_chunks():
    rx = JpegUdpReceiver()
    img = np.full((360, 640, 3), 127, np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    jpeg = buf.tobytes()
    half = len(jpeg) // 2
    assert rx._ingest(_datagram(7, 0, 2, len(jpeg), jpeg[:half], 99)) is None   # incomplete
    frame = rx._ingest(_datagram(7, 1, 2, len(jpeg), jpeg[half:], 99))          # completes
    assert frame is not None
    assert frame.frame_id == 7 and frame.sim_time_ns == 99
    assert frame.image_bgr.shape == (360, 640, 3)
    assert frame.recv_monotonic_ns > 0
    assert 7 not in rx._partials                 # completed frame is removed


def test_ingest_corrupt_complete_frame_returns_none_and_pops():
    # [review 4A] A complete-but-undecodable frame returns None AND drops its partial (no
    # leak), without raising. This is the path that previously skipped eviction.
    rx = JpegUdpReceiver()
    payload = b"this is not a jpeg"
    assert rx._ingest(_datagram(5, 0, 1, len(payload), payload, 42)) is None
    assert 5 not in rx._partials


def test_ingest_size_mismatch_returns_none_and_pops():
    rx = JpegUdpReceiver()
    payload = b"abc"
    # Declared jpeg_size (999) won't match the reassembled length -> dropped.
    assert rx._ingest(_datagram(6, 0, 1, 999, payload, 1)) is None
    assert 6 not in rx._partials


def test_ingest_short_datagram_ignored():
    rx = JpegUdpReceiver()
    assert rx._ingest(b"\x00\x01\x02") is None   # shorter than the 24-byte header
    assert rx._partials == {}


def test_evict_stale_removes_only_aged_partials():
    rx = JpegUdpReceiver(stale_after_s=0.5)
    now = time.monotonic()
    rx._partials[1] = _PartialFrame(total_chunks=2, jpeg_size=10, sim_time_ns=0,
                                    chunks={0: b"x"}, first_seen_monotonic=now - 1.0)  # aged
    rx._partials[2] = _PartialFrame(total_chunks=2, jpeg_size=10, sim_time_ns=0,
                                    chunks={0: b"y"}, first_seen_monotonic=now)         # fresh
    rx._evict_stale()
    assert 1 not in rx._partials                 # aged partial evicted
    assert 2 in rx._partials                     # fresh partial kept


# -- select()-based frames() loop over real loopback UDP [red-team 2026-05-30] --------
def test_frames_yields_frame_over_loopback():
    img = np.full((360, 640, 3), 100, np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    jpeg = buf.tobytes()
    with JpegUdpReceiver(bind_host="127.0.0.1", port=0) as rx:
        port = rx._sock.getsockname()[1]                 # OS-assigned ephemeral port
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sender.sendto(_datagram(1, 0, 1, len(jpeg), jpeg, 55), ("127.0.0.1", port))
            frame = next(rx.frames(max_wait_s=2.0))
        finally:
            sender.close()
    assert frame.frame_id == 1 and frame.sim_time_ns == 55
    assert frame.image_bgr.shape == (360, 640, 3)


def test_frames_idle_timeout_returns_without_hanging():
    with JpegUdpReceiver(bind_host="127.0.0.1", port=0, stale_after_s=0.05) as rx:
        t0 = time.monotonic()
        frames = list(rx.frames(max_wait_s=0.1))         # no datagrams ever arrive
        elapsed = time.monotonic() - t0
    assert frames == []                                  # idle timeout -> graceful stop
    assert elapsed < 1.0                                 # returned promptly, did not hang


# -- stream-health metrics (first-contact MTU / packet-loss diagnostics) [red-team] --------
def test_metrics_track_datagrams_chunks_and_completion():
    rx = JpegUdpReceiver()
    img = np.full((360, 640, 3), 127, np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    jpeg = buf.tobytes()
    half = len(jpeg) // 2
    assert rx._ingest(_datagram(7, 0, 2, len(jpeg), jpeg[:half], 99)) is None
    assert rx._ingest(_datagram(7, 1, 2, len(jpeg), jpeg[half:], 99)) is not None
    m = rx.metrics
    assert m.datagrams == 2
    assert m.frames_completed == 1
    assert m.max_total_chunks == 2
    assert 0 < m.min_datagram_bytes <= m.max_datagram_bytes


def test_metrics_count_evicted_incomplete_as_loss():
    rx = JpegUdpReceiver(stale_after_s=0.5)
    rx._ingest(_datagram(8, 0, 3, 999, b"abcdef", 1))    # chunk 1 of 3, never completes
    assert rx.metrics.frames_completed == 0
    rx._partials[8].first_seen_monotonic -= 10.0          # age it past stale_after_s
    rx._evict_stale()
    assert rx.metrics.partials_evicted == 1
