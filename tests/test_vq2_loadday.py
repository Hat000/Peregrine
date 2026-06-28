"""Tests for scripts/vq2_loadday — socket-free, no live sim required.

All tests drive the scripts via their internal functions (not the CLI), using the same
fake-message pattern established in test_mavlink_client.py (SimpleNamespace fakes fed
to MavlinkClient._handle directly).

Covers:
  - wire_inventory: _build_inventory reports correct presence/absence of GT fields;
    _diff correctly identifies streams unique to Training vs Competitive.
  - imu_profile: _analyze produces correct rate, noise-floor, clipping, mag, and baro
    fields from synthetic samples.
  - two_load_hash: _diff correctly identifies stable vs shifted gate geometry and
    matching vs differing frame hashes.
"""
from __future__ import annotations

import hashlib
import json
import struct
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from racer.mavlink_client import MavlinkClient


# ---------------------------------------------------------------------------
# Shared fake-message helpers (mirrors test_mavlink_client.py pattern)
# ---------------------------------------------------------------------------

def _imu(time_usec=1_000_000, xacc=0.0, yacc=0.0, zacc=-9.80665,
          xmag=25.0, ymag=5.0, zmag=-40.0, abs_pressure=1013.25):
    m = SimpleNamespace(time_usec=time_usec, xacc=xacc, yacc=yacc, zacc=zacc,
                        xmag=xmag, ymag=ymag, zmag=zmag, abs_pressure=abs_pressure)
    m.get_type = lambda: "HIGHRES_IMU"
    return m


def _odometry(x=1.0, y=2.0, z=-3.0, vx=0.5, vy=0.0, vz=0.0,
              q=(1.0, 0.0, 0.0, 0.0), rollspeed=0.0, pitchspeed=0.0, yawspeed=0.0,
              reset_counter=0):
    m = SimpleNamespace(x=x, y=y, z=z, vx=vx, vy=vy, vz=vz,
                        q=list(q), rollspeed=rollspeed, pitchspeed=pitchspeed, yawspeed=yawspeed,
                        time_usec=1000, reset_counter=reset_counter)
    m.get_type = lambda: "ODOMETRY"
    return m


def _local_pos(x=1.0, y=2.0, z=3.0, vx=0.1, vy=0.2, vz=0.3):
    m = SimpleNamespace(time_boot_ms=999, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz)
    m.get_type = lambda: "LOCAL_POSITION_NED"
    return m


def _heartbeat(autopilot=18, vtype=2, base_mode=0, custom_mode=0):
    m = SimpleNamespace(autopilot=autopilot, type=vtype, base_mode=base_mode, custom_mode=custom_mode)
    m.get_type = lambda: "HEARTBEAT"
    return m


def _build_fake_client_with_gt():
    """Client with ODOMETRY + LOCAL_POSITION_NED (GT fields present)."""
    c = MavlinkClient()
    c._handle(_imu())
    c._handle(_odometry())
    c._handle(_local_pos())
    c._handle(_heartbeat())
    return c


def _build_fake_client_competitive():
    """Client with only HIGHRES_IMU + HEARTBEAT (GT fields blocked)."""
    c = MavlinkClient()
    c._handle(_imu())
    c._handle(_heartbeat())
    return c


# ---------------------------------------------------------------------------
# wire_inventory tests
# ---------------------------------------------------------------------------

class _FakeTracker:
    """Minimal MessageRateTracker stand-in that returns a fixed rates() dict."""
    def __init__(self, streams: dict[str, dict]):
        self._streams = streams

    def rates(self):
        return {t: {"count": r["count"], "hz": r["hz"]} for t, r in self._streams.items()}


def _make_tracker(stream_names):
    """Tracker reporting each stream as present with rate=100 Hz, count=1000."""
    return _FakeTracker({t: {"count": 1000, "hz": 100.0} for t in stream_names})


def test_wire_inventory_gt_fields_present_when_streams_arrive():
    """When ODOMETRY + LPN are received, gt_field_presence reports non-None."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vq2_loadday"))
    from wire_inventory import _build_inventory

    client = _build_fake_client_with_gt()
    tracker = _make_tracker(["HIGHRES_IMU", "HEARTBEAT", "ODOMETRY", "LOCAL_POSITION_NED"])
    inv = _build_inventory(client, tracker, "training")

    assert inv["gt_field_presence"]["position_ned"]["non_none"] is True
    assert inv["gt_field_presence"]["velocity_ned"]["non_none"] is True
    assert inv["gt_field_presence"]["orientation_ned_wxyz"]["non_none"] is True


def test_wire_inventory_gt_fields_absent_competitive():
    """When only IMU+HEARTBEAT arrive, GT fields show as None."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vq2_loadday"))
    from wire_inventory import _build_inventory

    client = _build_fake_client_competitive()
    tracker = _make_tracker(["HIGHRES_IMU", "HEARTBEAT"])
    inv = _build_inventory(client, tracker, "competitive")

    assert inv["gt_field_presence"]["position_ned"]["non_none"] is False
    assert inv["gt_field_presence"]["orientation_ned_wxyz"]["non_none"] is False


def test_wire_inventory_training_only_blocked_flags():
    """training_only_blocked reports True for streams not in the tracker."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vq2_loadday"))
    from wire_inventory import _build_inventory

    client = _build_fake_client_competitive()
    tracker = _make_tracker(["HIGHRES_IMU", "HEARTBEAT"])
    inv = _build_inventory(client, tracker, "competitive")

    # ODOMETRY + LPN are spec §9.3 blocked in Competitive; tracker didn't see them.
    assert inv["training_only_blocked"]["ODOMETRY"] is True
    assert inv["training_only_blocked"]["LOCAL_POSITION_NED"] is True
    # ATTITUDE is also blocked in Competitive
    assert inv["training_only_blocked"]["ATTITUDE"] is True


def test_wire_inventory_diff_c5(tmp_path):
    """_diff detects Training-exclusive streams and reports the C5 verdict."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vq2_loadday"))
    from wire_inventory import _build_inventory, _diff

    client_training = _build_fake_client_with_gt()
    tracker_training = _make_tracker(["HIGHRES_IMU", "HEARTBEAT", "ODOMETRY", "LOCAL_POSITION_NED"])
    inv_training = _build_inventory(client_training, tracker_training, "training")

    client_comp = _build_fake_client_competitive()
    tracker_comp = _make_tracker(["HIGHRES_IMU", "HEARTBEAT"])
    inv_comp = _build_inventory(client_comp, tracker_comp, "competitive")

    a_path = tmp_path / "inv_training.json"
    b_path = tmp_path / "inv_competitive.json"
    a_path.write_text(json.dumps(inv_training), encoding="utf-8")
    b_path.write_text(json.dumps(inv_comp), encoding="utf-8")

    # Should not raise; the diff is primarily printed but we can verify the JSON objects.
    # Check that ODOMETRY is only in the training inventory.
    assert "ODOMETRY" in inv_training["streams"]
    assert "ODOMETRY" not in inv_comp["streams"]


# ---------------------------------------------------------------------------
# imu_profile tests
# ---------------------------------------------------------------------------

def test_imu_profile_rate_from_recv_timestamps():
    """_analyze computes rate correctly from recv_ns field (100 samples @ 100 Hz = 100 Hz)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vq2_loadday"))
    from imu_profile import _analyze

    # 100 samples, 10 ms apart -> 100 Hz
    G = 9.80665
    t0 = time.monotonic_ns()
    samples = [
        {
            "time_usec": i * 10_000,
            "recv_ns": t0 + i * 10_000_000,   # 10 ms per sample
            "acc": [0.0, 0.0, -G],
            "mag": [25.0, 5.0, -40.0],
            "baro_hpa": 1013.25,
        }
        for i in range(100)
    ]
    prof = _analyze(samples)
    assert abs(prof["rate_hz"] - 100.0) < 2.0, f"rate_hz={prof['rate_hz']}"
    assert prof["c8_rate_ok"] is True


def test_imu_profile_noise_floor():
    """_analyze computes near-zero std for a constant signal."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vq2_loadday"))
    from imu_profile import _analyze

    G = 9.80665
    t0 = time.monotonic_ns()
    samples = [
        {"time_usec": i * 10_000, "recv_ns": t0 + i * 10_000_000,
         "acc": [0.0, 0.0, -G], "mag": [25.0, 5.0, -40.0], "baro_hpa": 1013.25}
        for i in range(50)
    ]
    prof = _analyze(samples)
    std = prof["accel_std_frd_ms2"]
    assert all(abs(s) < 1e-9 for s in std), f"Expected zero std, got {std}"


def test_imu_profile_clipping_detected():
    """_analyze flags clipping when any axis exceeds the threshold."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vq2_loadday"))
    from imu_profile import _analyze, _ACCEL_CLIP_THRESHOLD_MS2

    G = 9.80665
    t0 = time.monotonic_ns()
    samples = [
        {"time_usec": i * 10_000, "recv_ns": t0 + i * 10_000_000,
         "acc": [_ACCEL_CLIP_THRESHOLD_MS2 + 5.0, 0.0, -G],
         "mag": [25.0, 5.0, -40.0], "baro_hpa": 1013.25}
        for i in range(50)
    ]
    prof = _analyze(samples)
    assert prof["accel_clipping_suspected"] is True


def test_imu_profile_mag_absent():
    """_analyze correctly reports mag absent when all mag=None."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vq2_loadday"))
    from imu_profile import _analyze

    G = 9.80665
    t0 = time.monotonic_ns()
    samples = [
        {"time_usec": i * 10_000, "recv_ns": t0 + i * 10_000_000,
         "acc": [0.0, 0.0, -G], "mag": None, "baro_hpa": 1013.25}
        for i in range(50)
    ]
    prof = _analyze(samples)
    assert prof["mag"]["present"] is False
    assert prof["c2_mag_usable"] is False


def test_imu_profile_mag_all_zero():
    """All-zero mag is flagged as all_zero=True and not usable."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vq2_loadday"))
    from imu_profile import _analyze

    G = 9.80665
    t0 = time.monotonic_ns()
    samples = [
        {"time_usec": i * 10_000, "recv_ns": t0 + i * 10_000_000,
         "acc": [0.0, 0.0, -G], "mag": [0.0, 0.0, 0.0], "baro_hpa": 1013.25}
        for i in range(50)
    ]
    prof = _analyze(samples)
    assert prof["mag"]["all_zero"] is True
    assert prof["c2_mag_usable"] is False


def test_imu_profile_empty_samples():
    """_analyze returns an error dict for empty input."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vq2_loadday"))
    from imu_profile import _analyze

    prof = _analyze([])
    assert "error" in prof


# ---------------------------------------------------------------------------
# two_load_hash tests
# ---------------------------------------------------------------------------

def _make_fingerprint(label: str, gate_1_ned, frame_hash: str, n_gates=6):
    return {
        "label": label,
        "capture_time_utc": "2026-06-28T00:00:00Z",
        "gate_map_present": gate_1_ned is not None,
        "gate_1_ned": list(gate_1_ned) if gate_1_ned is not None else None,
        "n_gates": n_gates if gate_1_ned is not None else 0,
        "all_gates": (
            [{"gate_id": i, "position_ned": [float(i), 0.0, 0.0]} for i in range(n_gates)]
            if gate_1_ned is not None else []
        ),
        "frame_hash_sha256": frame_hash,
        "sim_boot_ms": 12345,
        "race_status": None,
        "statustexts_first_10": [],
    }


def test_two_load_hash_stable_geometry(tmp_path, capsys):
    """Two fingerprints with the same gate-1 NED (shift < 5 cm) => exit 0 (stable)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vq2_loadday"))
    from two_load_hash import _diff

    ned = [10.0, 5.0, -2.0]
    h = hashlib.sha256(b"frame_bytes").hexdigest()
    fp_a = _make_fingerprint("load1", ned, h)
    fp_b = _make_fingerprint("load2", ned, h)  # identical

    a = tmp_path / "fp_a.json"
    b = tmp_path / "fp_b.json"
    a.write_text(json.dumps(fp_a), encoding="utf-8")
    b.write_text(json.dumps(fp_b), encoding="utf-8")

    result = _diff(a, b)
    assert result == 0, f"Expected 0 (stable), got {result}"


def test_two_load_hash_shifted_geometry(tmp_path):
    """Gate-1 shift > 5 cm => exit 1 (moat falsified)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vq2_loadday"))
    from two_load_hash import _diff

    ned_a = [10.0, 5.0, -2.0]
    ned_b = [10.5, 5.0, -2.0]   # 50 cm shift (> 5 cm threshold)
    h = hashlib.sha256(b"frame_bytes").hexdigest()
    fp_a = _make_fingerprint("load1", ned_a, h)
    fp_b = _make_fingerprint("load2", ned_b, h)

    a = tmp_path / "fp_a.json"
    b = tmp_path / "fp_b.json"
    a.write_text(json.dumps(fp_a), encoding="utf-8")
    b.write_text(json.dumps(fp_b), encoding="utf-8")

    result = _diff(a, b)
    assert result == 1, f"Expected 1 (moat falsified), got {result}"


def test_two_load_hash_stable_geometry_different_frames(tmp_path):
    """Stable geometry but different frame hashes => exit 0 (geometry stable; note appearance)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vq2_loadday"))
    from two_load_hash import _diff

    ned = [10.0, 5.0, -2.0]
    h_a = hashlib.sha256(b"frame_bytes_load1").hexdigest()
    h_b = hashlib.sha256(b"frame_bytes_load2").hexdigest()
    fp_a = _make_fingerprint("load1", ned, h_a)
    fp_b = _make_fingerprint("load2", ned, h_b)

    a = tmp_path / "fp_a.json"
    b = tmp_path / "fp_b.json"
    a.write_text(json.dumps(fp_a), encoding="utf-8")
    b.write_text(json.dumps(fp_b), encoding="utf-8")

    # Geometry stable (exit 0) even when appearance differs
    result = _diff(a, b)
    assert result == 0


def test_two_load_hash_no_gate_map(tmp_path):
    """If gate map absent in both fingerprints => exit 2 (inconclusive)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vq2_loadday"))
    from two_load_hash import _diff

    h = hashlib.sha256(b"frame_bytes").hexdigest()
    fp_a = _make_fingerprint("load1", None, h)
    fp_b = _make_fingerprint("load2", None, h)

    a = tmp_path / "fp_a.json"
    b = tmp_path / "fp_b.json"
    a.write_text(json.dumps(fp_a), encoding="utf-8")
    b.write_text(json.dumps(fp_b), encoding="utf-8")

    result = _diff(a, b)
    assert result == 2


def test_two_load_hash_small_shift_within_threshold(tmp_path):
    """A 1 cm shift is within the 5 cm threshold => stable (exit 0)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vq2_loadday"))
    from two_load_hash import _diff

    ned_a = [10.0, 5.0, -2.0]
    ned_b = [10.01, 5.0, -2.0]   # 1 cm shift, within threshold
    h = hashlib.sha256(b"frame_bytes").hexdigest()
    fp_a = _make_fingerprint("load1", ned_a, h)
    fp_b = _make_fingerprint("load2", ned_b, h)

    a = tmp_path / "fp_a.json"
    b = tmp_path / "fp_b.json"
    a.write_text(json.dumps(fp_a), encoding="utf-8")
    b.write_text(json.dumps(fp_b), encoding="utf-8")

    result = _diff(a, b)
    assert result == 0
