"""Tests for the gate-seeker nav-estimate logger (_nav_estimate_record + _write_nav_estimate_jsonl).

Validates that:
  1. _nav_estimate_record produces a dict with all expected keys from synthetic
     nav_state / nav / cmd objects — no live sim required.
  2. _write_nav_estimate_jsonl writes a valid JSONL file when session_dir is set,
     and writes nothing when session_dir is None.
  3. A missing/None field on the inputs logs null rather than raising.

Run: .venv\\Scripts\\python.exe -m pytest tests/test_nav_estimate_log.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# Make rl/ importable without a live ShadowPC or installed package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rl"))

from fly_rl import _nav_estimate_record, _write_nav_estimate_jsonl  # noqa: E402

# ---------------------------------------------------------------------------
# Minimal synthetic stubs (no simulator, no torch, no live connection)
# ---------------------------------------------------------------------------

EXPECTED_KEYS = {
    "sim_time_ns",
    "tick_index",
    "gate_index",
    "ahrs_quat_wxyz",
    "roll_rad",
    "pitch_rad",
    "yaw_rad",
    "position_ned",
    "time_since_vision_s",
    "vert_z_est",       # A21 vertical-channel estimator export (null when off/unseeded)
    "vert_vz_est",
    "z_off_est",        # A25 gate-relative altitude offset export (null when off/unseeded)
    "vz_t",             # A25 seeker's commanded vertical-velocity target (null when no pursuit tick)
    "offset_z_world",   # A25 raw latched gate->drone vertical offset (null when no pose/no align)
    "pose_age_s",       # A25 per-tick pose observation age (null when no pose this tick)
    "contact_frozen",   # A26 contact-gate export (null when off/unseeded)
    "vz_t_consumed",    # A28 §2.6: the vz_t the CONTROLLER actually used this tick (the seeker-side
                        # "vz_t" above is stale on hold-last-demand bridge ticks -- the 2.3 Hz flicker)
    "term_gate",        # A28 §2.6: per-term vertical-law decomposition (pre tilt-comp/clip)
    "term_damp",
    "thrust_pre_clip",
    "zoff_innov",       # A28 §2.6: complementary-filter last-latch innovation (null when off/none)
    "zoff_innov_accepted",
    "zoff_miss",        # A28 §2.6: consecutive innovation-gate rejects (the reseed counter)
    "body_rate",
    "thrust",
    "yaw_des_rad",
    "raw_gyro_yaw",
    "los_rate_rps",     # A29 §4.4: the seeker's filtered world LOS angular rate (null when off)
    "vt_est_mps",       # A29 §4.4: estimated tangential velocity v_t = -r*theta_dot (null when off)
    "alat_mps2",        # A29 §4.4: applied lateral damping accel (0.0 inside the deadband)
    "track_range_m",    # A29 §4.4: the range used in v_t (EMA track, fallback PnP; null when off)
    "az_err_rad",       # A30 §4: the apparent gate azimuth az = wrap(psi_world - yaw_now) (null when off)
    "fwd_scale",        # A30 §4: the cos^2(az) forward-pointing scale in [0,1] (null when off)
    "seeker_regime",    # A29 §4.4: which command_visual regime returned this tick (first-class)
    "chase_dpsi_rad",   # A31: cumulative unwrapped LOS rotation since acquisition (orbit/whip guard; null when off)
    "pass_wire",        # A31: the authoritative wire gate-pass fast-turn signal (null when off)
    "bearing_dev_rad",  # A31: IMU-consistency bearing gate's last measured deviation (null when off)
    "bearing_allow_rad",  # A31: the bearing gate's noise+parallax allowance for that tick (null when off)
    "theta_g_deg",      # A32: angle(measured specific-force dir, predicted) -- "how wrong is down" (null when off)
    "accel_deweight_total",  # A32: total effective accel down-weight, post-cap (null when off)
    "ahrs_watchdog_fired",   # A32: cumulative gravity-recovery watchdog fires (null when no AHRS)
    "imu_samples_ingested",  # A32: HIGHRES_IMU ring samples consumed this tick (null when off)
    "bearing_w",        # A32: soft Cauchy bearing weight of the last accepted pose (null when off)
    "zoff_w",           # A32: vertical filter's last applied Huber*bearing weight (null when off)
}


def _make_nav_state(roll=0.1, pitch=-0.05, yaw=1.2,
                    pos=None, tsv=0.25):
    """Minimal NavState-like object with the fields _nav_estimate_record reads."""
    ns = SimpleNamespace()
    ns.roll  = roll
    ns.pitch = pitch
    ns.yaw   = yaw
    ns.position_ned = np.array([1.0, 2.0, -3.0] if pos is None else pos, dtype=np.float64)
    ns.time_since_vision_update_s = tsv
    return ns


def _make_nav(with_ahrs=True):
    """Minimal Navigator-like object with an optional _ahrs attribute."""
    nav = SimpleNamespace()
    if with_ahrs:
        ahrs = SimpleNamespace()
        ahrs.q_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        nav._ahrs = ahrs
    else:
        nav._ahrs = None
    return nav


def _make_state(sim_time_ns=123_456_789, gyro_body_raw=None):
    """Minimal DroneState-like object."""
    s = SimpleNamespace()
    s.sim_time_ns = sim_time_ns
    s.gyro_body_raw = gyro_body_raw
    return s


def _make_seeker(last_yaw_des=None):
    """Minimal GateSeeker-like object exposing _last_yaw_des (the A14 stash)."""
    sk = SimpleNamespace()
    sk._last_yaw_des = last_yaw_des
    return sk


def _make_cmd(body_rate=None, thrust=0.55):
    """Minimal ControlCommand-like object."""
    cmd = SimpleNamespace()
    cmd.body_rate = np.array([0.1, -0.2, 0.05] if body_rate is None else body_rate,
                              dtype=np.float64)
    cmd.thrust = thrust
    return cmd


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNavEstimateRecord:
    def test_all_expected_keys_present(self):
        rec = _nav_estimate_record(
            _make_nav_state(), _make_nav(), _make_state(), _make_cmd(),
            gate_index=2, tick_index=7,
        )
        assert EXPECTED_KEYS == rec.keys(), (
            f"missing keys: {EXPECTED_KEYS - rec.keys()}; "
            f"extra keys: {rec.keys() - EXPECTED_KEYS}"
        )

    def test_scalar_fields_correct(self):
        ns = _make_nav_state(roll=0.3, pitch=-0.1, yaw=2.5, tsv=0.12)
        s  = _make_state(sim_time_ns=999_000_000)
        rec = _nav_estimate_record(ns, _make_nav(), s, _make_cmd(),
                                   gate_index=5, tick_index=42)
        assert rec["sim_time_ns"] == 999_000_000
        assert rec["tick_index"]  == 42
        assert rec["gate_index"]  == 5
        assert abs(rec["roll_rad"]  - 0.3)  < 1e-9
        assert abs(rec["pitch_rad"] - (-0.1)) < 1e-9
        assert abs(rec["yaw_rad"]   - 2.5)  < 1e-9
        assert abs(rec["time_since_vision_s"] - 0.12) < 1e-9

    def test_numpy_arrays_are_lists(self):
        """Serialised fields must be plain Python lists (no numpy objects)."""
        rec = _nav_estimate_record(
            _make_nav_state(), _make_nav(), _make_state(), _make_cmd(),
            gate_index=0, tick_index=0,
        )
        assert isinstance(rec["position_ned"], list)
        assert isinstance(rec["body_rate"],    list)
        assert isinstance(rec["ahrs_quat_wxyz"], list)

    def test_json_serializable(self):
        """The record must be serialisable to JSON without error."""
        rec = _nav_estimate_record(
            _make_nav_state(), _make_nav(), _make_state(), _make_cmd(),
            gate_index=0, tick_index=0,
        )
        s = json.dumps(rec)          # must not raise
        back = json.loads(s)
        assert back["tick_index"] == 0

    def test_no_ahrs_logs_null(self):
        """When nav._ahrs is None, ahrs_quat_wxyz is null (not a crash)."""
        rec = _nav_estimate_record(
            _make_nav_state(), _make_nav(with_ahrs=False), _make_state(), _make_cmd(),
            gate_index=0, tick_index=0,
        )
        assert rec["ahrs_quat_wxyz"] is None

    def test_inf_tsv_logged_as_null(self):
        """time_since_vision_update_s == inf is serialised as null, not a crash."""
        ns = _make_nav_state(tsv=float("inf"))
        rec = _nav_estimate_record(ns, _make_nav(), _make_state(), _make_cmd(),
                                   gate_index=0, tick_index=0)
        assert rec["time_since_vision_s"] is None

    def test_missing_body_rate_logs_null(self):
        """cmd.body_rate is None -> body_rate field is null."""
        cmd = _make_cmd()
        cmd.body_rate = None
        rec = _nav_estimate_record(
            _make_nav_state(), _make_nav(), _make_state(), cmd,
            gate_index=0, tick_index=0,
        )
        assert rec["body_rate"] is None

    def test_a14_probe_fields_present_and_correct(self):
        """yaw_des_rad (from the seeker stash) + raw_gyro_yaw (from the state stash) are logged."""
        s = _make_state(gyro_body_raw=np.array([0.01, -0.02, 0.33], dtype=np.float64))
        sk = _make_seeker(last_yaw_des=1.7)
        rec = _nav_estimate_record(
            _make_nav_state(), _make_nav(), s, _make_cmd(),
            gate_index=0, tick_index=0, seeker=sk,
        )
        assert "yaw_des_rad" in rec and "raw_gyro_yaw" in rec
        assert abs(rec["yaw_des_rad"] - 1.7) < 1e-9
        assert abs(rec["raw_gyro_yaw"] - 0.33) < 1e-9  # z/yaw axis only

    def test_a14_probe_fields_default_null(self):
        """No seeker + no gyro stash -> both probe fields log null, not a crash."""
        rec = _nav_estimate_record(
            _make_nav_state(), _make_nav(), _make_state(), _make_cmd(),
            gate_index=0, tick_index=0,  # seeker omitted (defaults None)
        )
        assert rec["yaw_des_rad"] is None
        assert rec["raw_gyro_yaw"] is None

    def test_broken_nav_state_does_not_raise(self):
        """A nav_state with no roll attribute should log nulls, not crash."""
        ns = SimpleNamespace()           # intentionally missing all attributes
        rec = _nav_estimate_record(ns, _make_nav(), _make_state(), _make_cmd(),
                                   gate_index=0, tick_index=0)
        # Must not raise; some fields will be None
        assert "roll_rad" in rec
        assert rec["roll_rad"] is None


class TestWriteNavEstimateJsonl:
    def test_writes_jsonl_when_session_dir_set(self, tmp_path):
        records = [
            _nav_estimate_record(_make_nav_state(), _make_nav(), _make_state(),
                                 _make_cmd(), gate_index=i, tick_index=i)
            for i in range(5)
        ]
        _write_nav_estimate_jsonl(records, tmp_path)
        out = tmp_path / "nav_estimate.jsonl"
        assert out.exists(), "nav_estimate.jsonl was not created"
        lines = [l for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 5, f"expected 5 lines, got {len(lines)}"
        for i, line in enumerate(lines):
            obj = json.loads(line)
            assert obj["gate_index"] == i
            assert EXPECTED_KEYS == obj.keys(), f"row {i} has unexpected keys"

    def test_writes_nothing_when_session_dir_none(self, tmp_path):
        """With session_dir=None no file is created (byte-identical to before)."""
        records = [
            _nav_estimate_record(_make_nav_state(), _make_nav(), _make_state(),
                                 _make_cmd(), gate_index=0, tick_index=0)
        ]
        _write_nav_estimate_jsonl(records, None)
        # Nothing should be written
        assert list(tmp_path.iterdir()) == []

    def test_empty_records_writes_nothing(self, tmp_path):
        """An empty record list produces no file."""
        _write_nav_estimate_jsonl([], tmp_path)
        assert not (tmp_path / "nav_estimate.jsonl").exists()

    def test_file_is_valid_jsonl(self, tmp_path):
        """Every line in the output must parse as a JSON object."""
        records = [
            _nav_estimate_record(_make_nav_state(), _make_nav(), _make_state(),
                                 _make_cmd(thrust=0.6), gate_index=k, tick_index=k)
            for k in range(10)
        ]
        _write_nav_estimate_jsonl(records, tmp_path)
        out = tmp_path / "nav_estimate.jsonl"
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                obj = json.loads(line)   # must not raise
                assert isinstance(obj, dict)
