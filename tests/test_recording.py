"""Round-trip tests for the session recorder/reader.

These prove a recorded session reads back faithfully (raw MAVLink re-parses, JPEG bytes
are bit-exact, the clock bridge + stats are written) -- the recorder is one-shot infra for
a deterministic sim contact, so it must be trustworthy before first use.
"""
from __future__ import annotations

import os

os.environ.setdefault("MAVLINK20", "1")  # ensure v2 dialect before pymavlink import

import cv2
import numpy as np
from pymavlink import mavutil

from racer.contracts import Frame
from racer.recording import SCHEMA, Recorder, RecordingReader


def _mav() -> "mavutil.mavlink.MAVLink":
    return mavutil.mavlink.MAVLink(None, srcSystem=1, srcComponent=1)


def _gradient_img() -> np.ndarray:
    """A smooth (640x360) BGR image -- compresses cleanly so decode round-trips tightly."""
    row = np.linspace(0, 255, 640, dtype=np.uint8)
    plane = np.repeat(row[None, :], 360, axis=0)
    return np.stack([plane, (plane // 2), (255 - plane)], axis=-1)


def _jpeg(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def test_mavlink_roundtrip_reparses(tmp_path):
    mav = _mav()
    raw = [
        mav.attitude_encode(100, 0.1, -0.2, 1.5, 0.01, 0.02, 0.03).pack(mav),
        mav.attitude_encode(200, 0.2, -0.3, 1.6, 0.0, 0.0, 0.0).pack(mav),
        mav.heartbeat_encode(
            mavutil.mavlink.MAV_TYPE_QUADROTOR,
            mavutil.mavlink.MAV_AUTOPILOT_GENERIC,
            mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED,
            0,
            0,
        ).pack(mav),
    ]
    with Recorder(tmp_path / "s") as rec:
        for b in raw:
            rec.record_mavlink(b)

    reader = RecordingReader(tmp_path / "s")
    msgs = list(reader.iter_mavlink())
    assert [m.get_type() for m in msgs] == ["ATTITUDE", "ATTITUDE", "HEARTBEAT"]
    assert abs(msgs[0].roll - 0.1) < 1e-5
    assert abs(msgs[1].pitch - (-0.3)) < 1e-5
    assert msgs[2].base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
    assert reader.meta["mavlink_records"] == 3


def test_video_roundtrip_is_bit_exact(tmp_path):
    img = _gradient_img()
    jpeg = _jpeg(img)
    with Recorder(tmp_path / "s") as rec:
        rec.record_frame(Frame(1, 1000, img, 5, jpeg))
        rec.record_frame(Frame(2, 2000, img, 6, jpeg))

    reader = RecordingReader(tmp_path / "s")
    idx = list(reader.iter_video_index())
    assert [e["frame_id"] for e in idx] == [1, 2]
    assert idx[0]["sim_time_ns"] == 1000 and idx[1]["sim_time_ns"] == 2000
    # offsets are contiguous and lengths match
    assert idx[0]["offset"] == 0
    assert idx[1]["offset"] == len(jpeg)

    # raw JPEG bytes preserved exactly
    raws = [j for _, j in reader.iter_jpeg()]
    assert raws == [jpeg, jpeg]

    # decoded frames have the right identity + shape
    frames = list(reader.frames())
    assert [f.frame_id for f in frames] == [1, 2]
    assert frames[0].image_bgr.shape == (360, 640, 3)
    assert frames[0].image_bgr.dtype == np.uint8
    assert reader.meta["video_frames"] == 2


def test_video_reencode_fallback_when_no_raw(tmp_path):
    img = _gradient_img()
    with Recorder(tmp_path / "s") as rec:
        rec.record_frame(Frame(9, 7, img, 8))  # no jpeg_bytes -> recorder re-encodes

    reader = RecordingReader(tmp_path / "s")
    frames = list(reader.frames())
    assert len(frames) == 1
    assert frames[0].frame_id == 9
    assert frames[0].image_bgr.shape == (360, 640, 3)


def test_meta_clock_bridge_and_extra(tmp_path):
    with Recorder(tmp_path / "s") as rec:
        rec.add_meta(endpoint="udp:127.0.0.1:14550", git_commit="abc123")
        rec.record_mavlink(_mav().attitude_encode(1, 0.0, 0.0, 0.0, 0, 0, 0).pack(_mav()))
        assert rec.n_dropped == 0

    m = RecordingReader(tmp_path / "s").meta
    assert m["schema"] == SCHEMA
    assert m["endpoint"].endswith("14550")
    assert m["git_commit"] == "abc123"
    # clock bridge present and sane
    assert m["t0_unix_ns"] > 0 and m["t0_monotonic_ns"] > 0
    assert m["duration_s"] >= 0.0
    assert m["dropped"] == 0


def test_mixed_session_and_throughput(tmp_path):
    """Many interleaved records exercise the queue + writer thread; nothing dropped."""
    mav = _mav()
    img = _gradient_img()
    jpeg = _jpeg(img)
    n_mav = n_frame = 0
    with Recorder(tmp_path / "s") as rec:
        for i in range(300):
            rec.record_mavlink(mav.attitude_encode(i, 0.01 * i, 0.0, 0.0, 0, 0, 0).pack(mav))
            n_mav += 1
            if i % 3 == 0:
                rec.record_frame(Frame(i, 1000 * i, img, i + 1, jpeg))
                n_frame += 1
        assert rec.n_dropped == 0

    reader = RecordingReader(tmp_path / "s")
    assert reader.meta["mavlink_records"] == n_mav
    assert reader.meta["video_frames"] == n_frame
    assert reader.meta["dropped"] == 0
    assert len(list(reader.iter_mavlink())) == n_mav
    assert len(list(reader.frames())) == n_frame


def test_recorder_no_loss_at_scale(tmp_path):
    """No telemetry record is lost across a realistic multi-second burst, and it all reads back
    -- the 'we capture every message' guarantee at volume. (Sustained logging over a full 8-min
    run further relies on the background writer's throughput >> arrival, which is disk-bound;
    the design keeps disk I/O off the hot receive loop via the queue + writer thread.)"""
    mav = _mav()
    img = _gradient_img()
    jpeg = _jpeg(img)
    n_mav = 6000
    expected_frames = 0
    with Recorder(tmp_path / "s") as rec:
        for i in range(n_mav):
            rec.record_mavlink(mav.attitude_encode(i, 0.0, 0.0, 0.0, 0, 0, 0).pack(mav))
            if i % 12 == 0:
                rec.record_frame(Frame(i, 1000 * i, img, i + 1, jpeg))
                expected_frames += 1
        assert rec.n_dropped == 0

    reader = RecordingReader(tmp_path / "s")
    assert reader.meta["mavlink_records"] == n_mav
    assert reader.meta["video_frames"] == expected_frames
    assert reader.meta["dropped"] == 0
    assert len(list(reader.iter_mavlink())) == n_mav        # every record re-parses
    assert len(list(reader.frames())) == expected_frames


def test_reader_handles_video_only_and_mavlink_only(tmp_path):
    # video-only: empty tlog must iterate to nothing, not raise
    img = _gradient_img()
    with Recorder(tmp_path / "v") as rec:
        rec.record_frame(Frame(1, 1, img, 1, _jpeg(img)))
    rv = RecordingReader(tmp_path / "v")
    assert list(rv.iter_mavlink()) == []
    assert len(list(rv.frames())) == 1

    # mavlink-only: empty video index must iterate to nothing, not raise
    mav = _mav()
    with Recorder(tmp_path / "m") as rec:
        rec.record_mavlink(mav.heartbeat_encode(2, 3, 0, 0, 0).pack(mav))
    rm = RecordingReader(tmp_path / "m")
    assert list(rm.iter_video_index()) == []
    assert len(list(rm.iter_mavlink())) == 1
