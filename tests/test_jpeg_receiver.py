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


def test_ingest_dedups_resent_frame_id():
    # [first contact] the sim re-sends each frame's chunks ~14x. After a frame_id completes
    # once, further datagrams for it are dropped (not re-emitted, no new partial), so
    # perception runs exactly once per real frame.
    rx = JpegUdpReceiver()
    img = np.full((360, 640, 3), 127, np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    jpeg = buf.tobytes()
    dg = _datagram(7, 0, 1, len(jpeg), jpeg, 99)     # single-chunk frame
    assert rx._ingest(dg) is not None                 # first copy -> emitted
    assert rx._ingest(dg) is None                     # re-send -> dropped
    assert rx._ingest(dg) is None                     # ...every re-send
    assert rx.metrics.frames_completed == 1           # emitted exactly once
    assert rx.metrics.duplicate_datagrams == 2
    assert rx._partials == {}                          # re-sends never created a partial


def test_resent_chunks_complete_frame_before_dedup_kicks_in():
    # Dedup must NOT block re-sends of a still-INCOMPLETE frame: a chunk lost to a UDP drop is
    # recovered from a later re-sent copy. Suppression starts only once the frame has completed.
    rx = JpegUdpReceiver()
    img = np.full((360, 640, 3), 64, np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    jpeg = buf.tobytes()
    half = len(jpeg) // 2
    c0 = _datagram(9, 0, 2, len(jpeg), jpeg[:half], 77)
    c1 = _datagram(9, 1, 2, len(jpeg), jpeg[half:], 77)
    assert rx._ingest(c0) is None                      # copy A: chunk 0 (chunk 1 dropped)
    assert rx._ingest(c0) is None                      # copy B: chunk 0 again -> still incomplete, NOT deduped
    frame = rx._ingest(c1)                             # copy B: chunk 1 -> completes
    assert frame is not None and frame.frame_id == 9
    assert rx.metrics.duplicate_datagrams == 0         # nothing dropped before completion
    assert rx._ingest(c0) is None                      # now a later re-send IS deduped
    assert rx.metrics.duplicate_datagrams == 1


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


def test_ingest_noncontiguous_chunk_id_dropped_not_crash():
    # Complete-by-count (2 chunks) but chunk_id 2 is outside [0, 2) -> index 1 is missing.
    # Must drop + count, NOT raise KeyError (which would kill the frames() generator). The
    # official sample client guards this case; we now do too. [spec/sample cross-check]
    rx = JpegUdpReceiver()
    assert rx._ingest(_datagram(11, 0, 2, 10, b"aaaaa", 1)) is None   # chunk 0 of 2
    assert rx._ingest(_datagram(11, 2, 2, 10, b"bbbbb", 1)) is None   # chunk_id 2 (out of range)
    assert 11 not in rx._partials                                     # cleaned up, no leak
    assert rx.metrics.frames_bad_chunkmap == 1


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


def test_frames_drains_buffer_in_one_wake():
    """DRAIN-TO-EMPTY (A19 frame-supply fix): frames() reads ALL buffered datagrams per select()
    wake in a tight recvfrom loop, not one-recvfrom-per-select. Buffer many complete frames (+ a
    resend of each) BEFORE consuming, then assert: every unique frame is yielded in order (none
    lost), the resends are deduped, and select() was called FAR fewer times than the datagram count
    (proving the drain — the pre-fix loop needed one select PER datagram, which capped the drain at
    ~1/3 of the sim's flood and grew an unbounded backlog under CPU stalls). [workflow wf_03f4a6b8]"""
    import racer.vision.jpeg_receiver as jr
    img = np.full((360, 640, 3), 100, np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    jpeg = buf.tobytes()
    n = 12
    select_calls = {"n": 0}
    real_select = jr.select.select

    def counting_select(r, w, x, t):
        select_calls["n"] += 1
        return real_select(r, w, x, t)

    with JpegUdpReceiver(bind_host="127.0.0.1", port=0, stale_after_s=0.5) as rx:
        port = rx._sock.getsockname()[1]
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        jr.select.select = counting_select
        try:
            for fid in range(1, n + 1):
                dg = _datagram(fid, 0, 1, len(jpeg), jpeg, fid)
                sender.sendto(dg, ("127.0.0.1", port))
                sender.sendto(dg, ("127.0.0.1", port))   # the sim's resend -> must be deduped
            time.sleep(0.05)                              # let all datagrams land in the kernel buffer
            # Collect via the idle-timeout (not an early break) so ALL 2n datagrams are drained +
            # counted before the generator returns (a break would leave the last frame's resend
            # unread). max_wait_s small so the idle stop is quick.
            got = [fr.frame_id for fr in rx.frames(max_wait_s=0.15)]
        finally:
            jr.select.select = real_select
            sender.close()
    assert got == list(range(1, n + 1))                  # all unique frames, in order, none lost
    assert rx.metrics.duplicate_datagrams == n           # each resend deduped (frame already emitted)
    assert select_calls["n"] <= 5, select_calls          # drained many datagrams per wake, not 1/select


def test_frames_fast_drops_completed_multichunk_resends():
    """DUP FAST-PATH [2026-07-13]: once a frame completes, further datagrams for that frame_id are
    rejected in the drain loop by 4-byte frame_id alone (recvfrom_into a reused buffer, no header
    unpack) -- so the sim's ~33x re-send flood does not hold the GIL. Re-sends of a WHOLE multi-chunk
    frame must not re-emit, must not create a partial, and must be counted as duplicates."""
    img = np.full((360, 640, 3), 77, np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    jpeg = buf.tobytes()
    half = len(jpeg) // 2
    c0 = _datagram(5, 0, 2, len(jpeg), jpeg[:half], 42)
    c1 = _datagram(5, 1, 2, len(jpeg), jpeg[half:], 42)
    with JpegUdpReceiver(bind_host="127.0.0.1", port=0, stale_after_s=0.5) as rx:
        port = rx._sock.getsockname()[1]
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sender.sendto(c0, ("127.0.0.1", port))
            sender.sendto(c1, ("127.0.0.1", port))            # completes frame 5
            for _ in range(3):                                 # the sim's re-sends of a whole frame
                sender.sendto(c0, ("127.0.0.1", port))
                sender.sendto(c1, ("127.0.0.1", port))
            time.sleep(0.05)                                   # let all land in the kernel buffer
            got = [fr.frame_id for fr in rx.frames(max_wait_s=0.15)]
        finally:
            sender.close()
    assert got == [5]                                          # emitted exactly once
    assert rx.metrics.duplicate_datagrams == 6                 # 3 re-sends x 2 chunks, all fast-dropped
    assert rx._partials == {}                                  # re-sends never created a partial


def test_frames_dedups_with_fastpath_off():
    """dedup_fastpath=False routes EVERY datagram through _ingest (the pre-fix path); dups are still
    dropped there and each frame is emitted once -- correctness is independent of the perf toggle."""
    img = np.full((360, 640, 3), 100, np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    jpeg = buf.tobytes()
    with JpegUdpReceiver(bind_host="127.0.0.1", port=0, stale_after_s=0.5, dedup_fastpath=False) as rx:
        port = rx._sock.getsockname()[1]
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            for _ in range(3):                                 # original + 2 re-sends of one frame
                sender.sendto(_datagram(9, 0, 1, len(jpeg), jpeg, 9), ("127.0.0.1", port))
            time.sleep(0.05)
            got = [fr.frame_id for fr in rx.frames(max_wait_s=0.15)]
        finally:
            sender.close()
    assert got == [9]                                          # emitted once even with fast-path OFF
    assert rx.metrics.duplicate_datagrams == 2                 # 2 re-sends deduped by _ingest


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
