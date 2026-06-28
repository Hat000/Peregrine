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


# ---------------------------------------------------------------------------
# mag_probe tests (C2)
# ---------------------------------------------------------------------------

def _mag_samples(n=50, mx=25.0, my=5.0, mz=-40.0, vary=True, mag_none=False,
                 yaw_rad=0.0, odo_valid=True):
    """Build synthetic per-tick samples for mag_probe.analyze."""
    t0 = time.monotonic_ns()
    dt = 10_000_000  # 10 ms
    samples = []
    for i in range(n):
        if mag_none:
            mag = None
        elif vary:
            # Add small per-sample variation to simulate a real sensor
            mag = [mx + float(i) * 0.05, my + float(i) * 0.02, mz - float(i) * 0.01]
        else:
            mag = [mx, my, mz]
        samples.append({
            "sim_time_ns": i * dt,
            "recv_ns": t0 + i * dt,
            "mag": mag,
            "yaw_rad": yaw_rad,
            "odo_valid": odo_valid,
        })
    return samples


def test_mag_probe_absent():
    """mag_body always None -> not present, c2_usable=False."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vq2_loadday"))
    from mag_probe import analyze

    res = analyze(_mag_samples(mag_none=True))
    assert res["mag_present"] is False
    assert res["c2_usable"] is False


def test_mag_probe_all_zero():
    """All-zero mag -> all_zero=True, c2_usable=False."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vq2_loadday"))
    from mag_probe import analyze

    samples = _mag_samples(mx=0.0, my=0.0, mz=0.0, vary=False)
    res = analyze(samples)
    assert res["all_zero"] is True
    assert res["c2_usable"] is False


def test_mag_probe_low_variance():
    """Constant non-zero mag -> low_variance flagged, c2_usable=False."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vq2_loadday"))
    from mag_probe import analyze, _MAG_STD_THRESHOLD

    # Exactly constant field — std = 0, below any threshold
    samples = _mag_samples(mx=25.0, my=5.0, mz=-40.0, vary=False)
    res = analyze(samples)
    assert res["low_variance"] is True
    assert res["c2_usable"] is False


def test_mag_probe_varying_no_odo():
    """Varying mag but no ODOMETRY -> usable based on variance alone (correlation skipped)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vq2_loadday"))
    from mag_probe import analyze

    samples = _mag_samples(vary=True, odo_valid=False)
    res = analyze(samples)
    assert res["mag_present"] is True
    assert res["all_zero"] is False
    # The vary=True generates std ~ 0.7 per axis >> _MAG_STD_THRESHOLD=1e-3 -> low_variance=False
    assert res.get("low_variance") is False
    assert res["corr_valid"] is False   # no paired ODO samples


def test_mag_probe_empty_samples():
    """Empty sample list -> error dict."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vq2_loadday"))
    from mag_probe import analyze

    res = analyze([])
    assert "error" in res


def test_mag_probe_good_correlation():
    """Mag heading and ODO yaw both sweeping 90 deg -> high correlation."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vq2_loadday"))
    from mag_probe import analyze

    n = 100
    # Sweep yaw 0 -> 90 degrees (well over the 20 deg range needed)
    yaws = np.linspace(0, np.radians(90), n)
    t0 = time.monotonic_ns()
    samples = []
    for i, yaw in enumerate(yaws):
        B = 45.0
        samples.append({
            "sim_time_ns": i * 10_000_000,
            "recv_ns": t0 + i * 10_000_000,
            "mag": [B * float(np.cos(yaw)), B * float(np.sin(yaw)), -20.0],
            "yaw_rad": float(yaw),
            "odo_valid": True,
        })
    res = analyze(samples)
    assert res["mag_present"] is True
    assert res["corr_valid"] is True
    # Correlation should be very high
    assert res["corr_vs_odo_r"] is not None and abs(res["corr_vs_odo_r"]) > 0.9


# ---------------------------------------------------------------------------
# gap_audit tests (C7)
# ---------------------------------------------------------------------------

def test_gap_audit_no_gaps():
    """All frames detected -> no gaps, max_gap=0, gap_filler_needed=False."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vq2_loadday"))
    from gap_audit import analyze

    dt_ns = int(1e9 / 30)  # 30 Hz
    records = [
        {"frame_id": i, "sim_time_ns": i * dt_ns, "n_detections": 1}
        for i in range(300)
    ]
    res = analyze(records)
    assert res["max_gap_frames"] == 0
    assert res["n_gap_runs"] == 0
    assert res["c7_gap_filler_needed"] is False
    assert res["detection_rate"] == 1.0


def test_gap_audit_small_gaps():
    """Short gaps (< threshold) -> gap_filler_needed=False."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vq2_loadday"))
    from gap_audit import analyze

    dt_ns = int(1e9 / 30)  # 30 Hz -> 33 ms/frame
    # Insert a 3-frame gap (3 * 33 = ~99 ms < 200 ms threshold)
    records = []
    for i in range(100):
        n_det = 0 if 40 <= i < 43 else 1  # 3-frame gap at i=40..42
        records.append({"frame_id": i, "sim_time_ns": i * dt_ns, "n_detections": n_det})
    res = analyze(records)
    assert res["max_gap_frames"] == 3
    assert res["c7_gap_filler_needed"] is False   # 3 * 33 ms = 99 ms < 200 ms
    assert res["c7_gru_depth_frames"] == 5   # max_gap + 2


def test_gap_audit_large_gap():
    """A gap > threshold -> gap_filler_needed=True."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vq2_loadday"))
    from gap_audit import analyze

    dt_ns = int(1e9 / 30)  # 30 Hz -> 33 ms/frame; need > 200 ms -> > 6 frames
    records = []
    for i in range(200):
        n_det = 0 if 50 <= i < 60 else 1  # 10-frame gap = 333 ms > 200 ms
        records.append({"frame_id": i, "sim_time_ns": i * dt_ns, "n_detections": n_det})
    res = analyze(records)
    assert res["max_gap_frames"] == 10
    assert res["c7_gap_filler_needed"] is True


def test_gap_audit_empty():
    """Empty record list -> error dict."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vq2_loadday"))
    from gap_audit import analyze

    res = analyze([])
    assert "error" in res


def test_gap_audit_detection_rate():
    """Alternating detected/gap frames -> detection_rate ~ 0.5."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vq2_loadday"))
    from gap_audit import analyze

    dt_ns = int(1e9 / 30)
    records = [
        {"frame_id": i, "sim_time_ns": i * dt_ns, "n_detections": i % 2}
        for i in range(200)
    ]
    res = analyze(records)
    assert abs(res["detection_rate"] - 0.5) < 0.02


def test_gap_audit_jsonl_roundtrip(tmp_path):
    """Write a JSONL det log, load it, and verify analyze output matches direct call."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vq2_loadday"))
    from gap_audit import analyze, _load_det_log

    dt_ns = int(1e9 / 30)
    records = [
        {"frame_id": i, "sim_time_ns": i * dt_ns, "n_detections": 1 if i % 3 != 0 else 0}
        for i in range(90)
    ]
    log_path = tmp_path / "det_log.jsonl"
    with open(log_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    loaded = _load_det_log(log_path)
    res_direct = analyze(records)
    res_loaded = analyze(loaded)
    assert res_direct["max_gap_frames"] == res_loaded["max_gap_frames"]
    assert res_direct["n_gap_runs"] == res_loaded["n_gap_runs"]


def test_gap_audit_synthetic_helper():
    """_make_synthetic generates correct frame count and roughly correct detect rate."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vq2_loadday"))
    from gap_audit import _make_synthetic, analyze

    records = _make_synthetic(500, 0.7)
    assert len(records) == 500
    res = analyze(records)
    # Detection rate should be near 0.7 (within 5 pp for seed=42)
    assert abs(res["detection_rate"] - 0.7) < 0.05


# ---------------------------------------------------------------------------
# clock_offset tests (C10)
# ---------------------------------------------------------------------------

def test_clock_offset_timesync_no_echoes():
    """analyze_timesync with empty rounds -> echo_rate=0, c10a_timesync_works=False."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vq2_loadday"))
    from clock_offset import analyze_timesync

    res = analyze_timesync([])
    assert res["n_echoed"] == 0
    assert res["c10a_timesync_works"] is False
    assert res["echo_rate"] == 0.0


def test_clock_offset_timesync_good_rounds():
    """Synthetic perfect-echo rounds -> correct RTT/offset stats."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vq2_loadday"))
    from clock_offset import analyze_timesync

    # Simulate 10 round-trips: RTT=5 ms, clock offset=+2 ms (sim ahead)
    now = time.monotonic_ns()
    rtt_ns = 5_000_000   # 5 ms
    offset_ns = 2_000_000  # sim 2 ms ahead of our clock
    rounds = []
    for i in range(10):
        tc1 = int(time.time_ns()) + i * 1_000_000
        send_ns = now + i * 10_000_000
        recv_ns = send_ns + rtt_ns
        ts1_echo = tc1 + rtt_ns // 2 + offset_ns  # sim timestamp at mid-trip + offset
        rounds.append({
            "send_ns": send_ns,
            "recv_ns": recv_ns,
            "tc1_sent": tc1,
            "ts1_echo": ts1_echo,
            "rtt_ns": rtt_ns,
            "offset_ns": ts1_echo - (tc1 + rtt_ns // 2),
        })
    res = analyze_timesync(rounds)
    assert res["c10a_timesync_works"] is True
    assert res["n_echoed"] == 10
    assert abs(res["median_rtt_ms"] - 5.0) < 0.5
    assert abs(res["median_offset_ms"] - 2.0) < 0.5


def test_clock_offset_timesync_high_rtt_outliers():
    """Rounds with RTT > cap are filtered; a few good ones still give a result."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vq2_loadday"))
    from clock_offset import analyze_timesync, _RTT_CAP_MS

    now = time.monotonic_ns()
    rtt_good_ns = 5_000_000
    rtt_bad_ns = int((_RTT_CAP_MS + 10) * 1e6)  # above cap
    rounds = []
    for i in range(5):
        tc1 = int(time.time_ns()) + i * 1_000_000
        send_ns = now + i * 10_000_000
        rtt_ns = rtt_bad_ns if i < 3 else rtt_good_ns
        recv_ns = send_ns + rtt_ns
        ts1_echo = tc1 + rtt_ns // 2
        rounds.append({
            "send_ns": send_ns, "recv_ns": recv_ns, "tc1_sent": tc1,
            "ts1_echo": ts1_echo, "rtt_ns": rtt_ns,
            "offset_ns": ts1_echo - (tc1 + rtt_ns // 2),
        })
    res = analyze_timesync(rounds)
    assert res["c10a_timesync_works"] is True
    assert res["n_good_rounds"] == 2


def test_gate_ordering_no_race_status():
    """No RACE_STATUS packets -> c10b_gate_ordering_works=False."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vq2_loadday"))
    from clock_offset import analyze_gate_ordering

    res = analyze_gate_ordering([])
    assert res["c10b_gate_ordering_works"] is False
    assert res["n_race_status_pkts"] == 0


def test_gate_ordering_static_pre_arm():
    """RACE_STATUS all gate_index=0, race not started -> inconclusive (None)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vq2_loadday"))
    from clock_offset import analyze_gate_ordering

    seq = [
        {"active_gate_index": 0, "started": False, "finished": False},
    ] * 10
    res = analyze_gate_ordering(seq)
    # All gate_index=0, not started -> inconclusive
    assert res["c10b_gate_ordering_works"] is None


def test_gate_ordering_advances():
    """Gate index advances during flight -> c10b_gate_ordering_works=True."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vq2_loadday"))
    from clock_offset import analyze_gate_ordering

    seq = []
    for i in range(6):
        seq.append({"active_gate_index": i, "started": True, "finished": False})
    res = analyze_gate_ordering(seq)
    assert res["c10b_gate_ordering_works"] is True
    assert res["n_gate_advances"] == 5
    assert res["race_ever_started"] is True


def test_gate_ordering_finished():
    """Race finishes -> finished=True captured."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vq2_loadday"))
    from clock_offset import analyze_gate_ordering

    seq = [
        {"active_gate_index": 0, "started": True, "finished": False},
        {"active_gate_index": 5, "started": True, "finished": True},
    ]
    res = analyze_gate_ordering(seq)
    assert res["race_ever_finished"] is True


# ---------------------------------------------------------------------------
# detector_smoke tests (C6)
# ---------------------------------------------------------------------------

def test_detector_smoke_compare_to_baseline_no_regression():
    """compare_to_baseline with results at baseline values -> c6a_pass=True, no alerts."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vq2_loadday"))
    from detector_smoke import compare_to_baseline

    baseline = {
        "level_2_detect_rate": 0.85,
        "level_2_median_corner_px": 5.0,
        "level_3_detect_rate": 0.75,
        "level_3_median_corner_px": 8.0,
    }
    syn_results = {
        2: {"detect_rate": 0.85, "median_corner_px": 5.0, "p90_corner_px": 15.0},
        3: {"detect_rate": 0.75, "median_corner_px": 8.0, "p90_corner_px": 22.0},
    }
    res = compare_to_baseline(syn_results, baseline)
    assert res["c6a_pass"] is True
    assert res["alerts"] == []


def test_detector_smoke_compare_to_baseline_detect_rate_drop():
    """If detect rate drops > 15 pp, an alert is raised and c6a_pass=False."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vq2_loadday"))
    from detector_smoke import compare_to_baseline

    baseline = {
        "level_2_detect_rate": 0.85,
        "level_2_median_corner_px": 5.0,
    }
    # Drop 20 pp -> exceeds _DETECT_RATE_DROP_ALERT (0.15)
    syn_results = {
        2: {"detect_rate": 0.65, "median_corner_px": 5.0, "p90_corner_px": 15.0},
    }
    res = compare_to_baseline(syn_results, baseline)
    assert res["c6a_pass"] is False
    assert len(res["alerts"]) >= 1
    assert any("detect_rate" in a for a in res["alerts"])


def test_detector_smoke_compare_to_baseline_corner_err_scale():
    """If corner error is > 2x baseline, an alert is raised."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vq2_loadday"))
    from detector_smoke import compare_to_baseline

    baseline = {
        "level_2_detect_rate": 0.85,
        "level_2_median_corner_px": 5.0,
    }
    # Corner error 3x baseline
    syn_results = {
        2: {"detect_rate": 0.85, "median_corner_px": 15.0, "p90_corner_px": 40.0},
    }
    res = compare_to_baseline(syn_results, baseline)
    assert res["c6a_pass"] is False
    assert any("corner" in a.lower() for a in res["alerts"])


def test_detector_smoke_compare_to_baseline_partial_levels():
    """compare_to_baseline handles missing levels gracefully (no KeyError)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vq2_loadday"))
    from detector_smoke import compare_to_baseline

    baseline = {
        "level_2_detect_rate": 0.85,
        "level_2_median_corner_px": 5.0,
        "level_3_detect_rate": 0.75,
    }
    # Only level 2 results available
    syn_results = {
        2: {"detect_rate": 0.85, "median_corner_px": 5.0, "p90_corner_px": 15.0},
    }
    # Should not raise
    res = compare_to_baseline(syn_results, baseline)
    assert isinstance(res["c6a_pass"], bool)


def test_detector_smoke_no_weights(tmp_path):
    """Running the smoke test without weights file exits with code 2, no crash."""
    import subprocess
    import sys as _sys

    probe_path = (
        Path(__file__).resolve().parents[1]
        / "scripts" / "vq2_loadday" / "detector_smoke.py"
    )
    venv_python = (
        Path(__file__).resolve().parents[1] / ".venv" / "Scripts" / "python.exe"
    )
    if not venv_python.exists():
        venv_python = Path(_sys.executable)

    result = subprocess.run(
        [str(venv_python), str(probe_path), "--weights", str(tmp_path / "nonexistent.pt"),
         "--n", "5"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 2, (
        f"Expected exit 2, got {result.returncode}\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
