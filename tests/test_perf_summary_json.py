"""Tests for the gate-seeker perf-summary writer (_write_perf_summary_json).

_write_perf_summary_json is the durable twin of the [loop-rate] / [vision-timing] /
[async-detect] / [seeker-diag] prints in _fly_gate_seeker: it copies the same
already-computed fields out of the per-flight ``result`` dict into
``<session_dir>/perf_summary.json`` so the confirm-flight's primary evidence survives
even if stdout is lost. No live sim is needed to exercise it -- it is a pure function
over a plain dict + a path.

Validates that:
  1. Only the known perf/diagnostic keys present on ``result`` are written (extra
     result keys like collisions_at_start are not silently included).
  2. Missing optional keys (e.g. no vision_step_ms on a sync-path result) are simply
     omitted, not written as null/error.
  3. session_dir=None writes nothing (byte-identical to before this feature existed).
  4. A non-writable/broken destination is swallowed -- never raises -- matching the
     try/except contract shared with _write_nav_estimate_jsonl.
  5. DATA-LOSS GUARD: an exception (or Ctrl-C) escaping _fly_gate_seeker's tick loop
     still produces BOTH nav_estimate.jsonl and perf_summary.json (the epilogue runs in
     a finally), the exception still PROPAGATES, and final_state carries main()'s
     existing EXCEPTION/INTERRUPTED vocabulary. Exercised against the REAL
     _fly_gate_seeker with fake client/nav/seeker (only _build_casec_seeker is
     monkeypatched -- no live sim, no GPU).

Run: .venv\\Scripts\\python.exe -m pytest tests/test_perf_summary_json.py -q
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

import fly_rl  # noqa: E402
from fly_rl import _write_perf_summary_json  # noqa: E402


def _full_result() -> dict:
    """A result dict shaped like a completed confirm-flight run (all fields present)."""
    return {
        "flight": 1,
        "final_state": "TIMEOUT",
        "gate_index": 3,
        "collisions": 0,
        "collisions_at_start": 0,   # NOT one of the written keys -- internal bookkeeping only
        "achieved_hz": 27.4,
        "worst_work_ms": 41.2,
        "loop_over_budget_pct": 2.1,
        "async_detect": {
            "vision_fps": 14.8, "n_detects": 210, "detect_mean_ms": 58.0,
            "detect_max_ms": 121.0, "obs_age_mean_ms": 90.0, "obs_age_max_ms": 210.0,
            "worker_errors": 0,
        },
        "vision_step_ms": {
            "detect": {"count": 400, "total_ms": 20.0, "max_ms": 3.0},
            "vp_yaw": {"count": 27, "total_ms": 850.0, "max_ms": 65.0},
        },
        "vision_worst_tick_ms": {"detect": 0.1, "vp_yaw": 65.0},
        "seeker_diag": {"pursuit": 380, "none_total": 20, "bridged": 15},
    }


class TestWritePerfSummaryJson:
    def test_writes_json_when_session_dir_set(self, tmp_path):
        result = _full_result()
        _write_perf_summary_json(result, tmp_path)
        out = tmp_path / "perf_summary.json"
        assert out.exists(), "perf_summary.json was not created"
        obj = json.loads(out.read_text(encoding="utf-8"))
        assert obj["final_state"] == "TIMEOUT"
        assert obj["achieved_hz"] == 27.4
        assert obj["async_detect"]["vision_fps"] == 14.8
        assert obj["vision_step_ms"]["vp_yaw"]["max_ms"] == 65.0
        assert obj["seeker_diag"]["pursuit"] == 380

    def test_only_known_keys_are_written(self, tmp_path):
        """Internal bookkeeping (e.g. collisions_at_start) must not leak into the file."""
        result = _full_result()
        result["some_future_debug_field"] = {"huge": "blob"}
        _write_perf_summary_json(result, tmp_path)
        obj = json.loads((tmp_path / "perf_summary.json").read_text(encoding="utf-8"))
        assert "collisions_at_start" not in obj
        assert "some_future_debug_field" not in obj

    def test_missing_optional_keys_are_omitted_not_errors(self, tmp_path):
        """A sync-path (non-async-detect) result has no vision_step_ms/async_detect/seeker_diag
        -- those keys must simply be absent, not raise or serialize as null noise."""
        minimal = {
            "flight": 1, "final_state": "FINISHED", "gate_index": 5, "collisions": 0,
            "achieved_hz": 30.1, "worst_work_ms": 12.0, "loop_over_budget_pct": 0.0,
        }
        _write_perf_summary_json(minimal, tmp_path)
        obj = json.loads((tmp_path / "perf_summary.json").read_text(encoding="utf-8"))
        assert obj["final_state"] == "FINISHED"
        for absent_key in ("async_detect", "vision_step_ms", "vision_worst_tick_ms", "seeker_diag"):
            assert absent_key not in obj

    def test_writes_nothing_when_session_dir_none(self, tmp_path):
        """With session_dir=None no file is created (no directory to write into)."""
        _write_perf_summary_json(_full_result(), None)
        assert list(tmp_path.iterdir()) == []

    def test_never_raises_on_bad_destination(self, tmp_path):
        """A session_dir that cannot be written to (e.g. missing parent) must be swallowed,
        matching the try/except contract this shares with _write_nav_estimate_jsonl -- a
        logging failure must never break the flight exit path."""
        bad_dir = tmp_path / "does_not_exist" / "still_missing"
        # Must not raise even though bad_dir was never created.
        _write_perf_summary_json(_full_result(), bad_dir)
        assert not bad_dir.exists()

    def test_file_is_valid_json_object(self, tmp_path):
        _write_perf_summary_json(_full_result(), tmp_path)
        out = tmp_path / "perf_summary.json"
        obj = json.loads(out.read_text(encoding="utf-8"))
        assert isinstance(obj, dict)


# ---------------------------------------------------------------------------
# DATA-LOSS GUARD: mid-loop exception still writes both evidence files
# (drives the REAL _fly_gate_seeker; only _build_casec_seeker is monkeypatched)
# ---------------------------------------------------------------------------

class _FakeClient:
    """Just enough of MavlinkClient for _fly_gate_seeker's tick loop: static-but-advancing
    sim clock, no collisions, no reset, no frames. track_gates falsy + the vq2_case_c
    profile's self_localizing=True -> the MAP-FREE branch (no --map file needed)."""

    def __init__(self):
        self.state = SimpleNamespace(
            sim_time_ns=1_000_000_000,
            reset_counter=0,
            position_ned=np.array([0.0, 0.0, -1.0], dtype=np.float64),
        )
        self.race_status = {"active_gate_index": 0, "finished": False}
        self.collisions: list = []
        self.track_gates = None
        self.cmd_rate_scale = 0.4
        self._latest_frame = None
        self.sent: list = []

    def pump(self):
        self.state.sim_time_ns += 1_000_000   # keep the sim clock advancing (stall guard)

    def send_command(self, cmd):
        self.sent.append(cmd)


class _FakeNav:
    """Navigator stand-in: update() returns a NavState-like stub each tick."""

    _ahrs = None   # _nav_estimate_record tolerates a None AHRS (fields log null)

    def update(self, s, frame):
        ns = SimpleNamespace()
        ns.roll, ns.pitch, ns.yaw = 0.1, -0.05, 1.2
        ns.position_ned = np.array([1.0, 2.0, -3.0], dtype=np.float64)
        ns.time_since_vision_update_s = 0.25
        return ns


class _RaisingSeeker:
    """GateSeeker stand-in: first command tick succeeds (one nav-log record lands),
    the second raises the injected exception from INSIDE the tick loop."""

    def __init__(self, exc: BaseException, raise_on_call: int = 2):
        self._exc = exc
        self._raise_on = raise_on_call
        self._n = 0

    def command_visual(self, nav_state, frame, gate_index, is_final_gate=False):
        self._n += 1
        if self._n >= self._raise_on:
            raise self._exc
        cmd = SimpleNamespace()
        cmd.body_rate = np.array([0.1, -0.2, 0.05], dtype=np.float64)
        cmd.thrust = 0.55
        return cmd


def _make_args() -> SimpleNamespace:
    """Only the args attributes _fly_gate_seeker itself reads (the heavy consumers --
    detector build, async worker -- live behind the monkeypatched _build_casec_seeker)."""
    return SimpleNamespace(
        deploy_profile="vq2_case_c",
        seeker_detector="none",
        seeker_speed=3.0,
        max_seconds=10.0,          # generous; the seeker raises on tick 2 long before this
        rate=200.0,                # 5 ms ticks -> the 2-tick test stays fast
        ignore_collisions=True,
        async_detect="off",
        vertical_estimator="off",
    )


class TestMidLoopExceptionStillWritesEvidence:
    @pytest.mark.parametrize("exc, expected_state", [
        (RuntimeError("boom-mid-loop"), "EXCEPTION"),
        (KeyboardInterrupt(), "INTERRUPTED"),
    ], ids=["runtime-error", "ctrl-c"])
    def test_exception_mid_loop_writes_both_files_and_propagates(
            self, tmp_path, monkeypatch, exc, expected_state):
        nav, seeker, client = _FakeNav(), _RaisingSeeker(exc), _FakeClient()
        monkeypatch.setattr(
            fly_rl, "_build_casec_seeker",
            lambda args, gates, frame_source=None: (nav, seeker, None, None))
        result = {"flight": 1, "final_state": "NO_GO", "gate_index": 0,
                  "collisions_at_start": 0}

        # The exception must still PROPAGATE (the finally never swallows it) ...
        with pytest.raises(type(exc)):
            fly_rl._fly_gate_seeker(client, _make_args(), 1, tmp_path, result)

        # ... AND both evidence files must exist despite the mid-loop crash.
        perf = json.loads((tmp_path / "perf_summary.json").read_text(encoding="utf-8"))
        assert perf["final_state"] == expected_state
        assert "achieved_hz" in perf, "loop-rate stats missing from the crash-path summary"
        nav_out = tmp_path / "nav_estimate.jsonl"
        assert nav_out.exists(), "nav_estimate.jsonl was not written on the crash path"
        rows = [json.loads(l) for l in nav_out.read_text(encoding="utf-8").splitlines()
                if l.strip()]
        assert len(rows) == 1, "expected exactly the one pre-crash tick's record"
        # result was labeled with main()'s existing vocabulary -- no new states.
        assert result["final_state"] == expected_state
