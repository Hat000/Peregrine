"""Residual loop-choke fix package (R1 + R2 + Q3 phase-timing) — 2026-07-05.

Pins the three fixes from handoff/vq2_residual_choke_diag_2026-07-05/ANALYSIS.md:

  R1  Worker-thread vp_yaw (racer.vision.async_detect.VpYawWorker + the Navigator wiring behind
      NavigatorConfig.vp_yaw_async). ONLY the estimate_heading VP-RANSAC moves off the control-loop
      thread; the branch disambiguation / quality+branch gates / update_yaw noise stay on-thread so
      acceptance is byte-identical to the synchronous path. Default OFF == byte-identical sync path.
        * worker produces a HeadingEstimate that the loop applies (yaw pulled toward the datum)
        * latest-wins under backlog (a fresh submit overwrites an unconsumed one -> no queue)
        * flag OFF == byte-identical sync behaviour on the touched path (same KF/AHRS state)
        * a worker compute-fn exception is counted, never reaches the loop

  R2  Short-circuit ACTUATOR_OUTPUT_STATUS parse (MavlinkClient.parse_actuator_output). Default True
      == byte-identical (test_mavlink_client + rate_sysid depend on it); False skips the per-msg dict
      build for the flown loop. The tlog is UNAFFECTED (raw bytes tapped via on_message before _handle).
        * flag False: an ACTUATOR msg no longer mutates client.actuator_outputs
        * flag True (default): populated exactly as before
        * other msg types (COLLISION) unaffected by the flag
        * tlog-path independence: on_message fires on the ACTUATOR msg regardless of the parse flag

  Q3  Per-tick phase timing + the work_ms accounting fix + the nav-record wall stamp, exercised end-
      to-end against the REAL _fly_gate_seeker (only _build_casec_seeker monkeypatched -- no sim/GPU).
        * loop_phase_ms buckets populate + land in perf_summary.json
        * each nav_estimate record carries a monotonic t_mono_ns wall stamp

Torch-free throughout (duck-typed fakes; no model, no sim, no GPU).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))

from racer.contracts import DroneState, Frame, Gate, GateObservation  # noqa: E402
from racer.frames import R_camera_from_body, euler_from_quat_wxyz  # noqa: E402
from racer.navigator import Navigator, NavigatorConfig  # noqa: E402
from racer.vision.async_detect import VpYawResult, VpYawWorker  # noqa: E402
from racer.vision.gate_pose import project_gate_corners  # noqa: E402
from racer.vision.heading_vp import HeadingEstimate  # noqa: E402

_LEVEL_Q = np.array([1.0, 0.0, 0.0, 0.0])
_HOVER_ACCEL = np.array([0.0, 0.0, -9.80665])
_DT_NS = 5_000_000          # 5 ms = 200 Hz HIGHRES_IMU


def _heading_estimate(yaw=0.0, quality=0.9):
    """A HeadingEstimate at absolute yaw ``yaw`` (mod-90 branches around it)."""
    branches = np.array([yaw, yaw + np.pi / 2, yaw + np.pi, yaw - np.pi / 2])
    return HeadingEstimate(heading_mod90_rad=float(yaw), quality=float(quality), n_support=40,
                           vp_px=np.array([320.0, 180.0]), horizontality=0.05,
                           branch_headings_rad=branches)


def _frame(frame_id, image=None):
    img = np.zeros((360, 640, 3), np.uint8) if image is None else image
    return Frame(frame_id=frame_id, sim_time_ns=int(frame_id) * _DT_NS,
                 image_bgr=img, recv_monotonic_ns=0)


def _wait_for(predicate, timeout_s=2.0, poll_s=0.002):
    """Spin (with sleeps) until predicate() or timeout. Returns predicate()'s truthiness."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll_s)
    return predicate()


# ======================================================================================
# R1a. VpYawWorker in isolation: computes, publishes latest-wins, swallows errors
# ======================================================================================
class TestVpYawWorkerUnit:
    def test_worker_computes_and_publishes(self):
        seen = []

        def compute(img, roll, pitch):
            seen.append((roll, pitch))
            return _heading_estimate(yaw=0.3)

        w = VpYawWorker(compute)
        w.start()
        try:
            w.submit(_frame(1), roll=0.1, pitch=-0.2)
            assert _wait_for(lambda: w.latest() is not None), "worker never published a result"
            res = w.latest()
            assert isinstance(res, VpYawResult)
            assert res.frame_id == 1
            assert res.estimate is not None and abs(res.estimate.heading_mod90_rad - 0.3) < 1e-9
            # roll/pitch were passed through to the compute fn verbatim (capture-time tilt)
            assert (abs(res.roll - 0.1) < 1e-9) and (abs(res.pitch + 0.2) < 1e-9)
            assert w.n_computes >= 1 and w.n_errors == 0
        finally:
            w.stop()

    def test_latest_wins_under_backlog(self):
        """Many rapid submits with the worker slow: the worker must skip stale frames and end on the
        FRESHEST submitted frame_id (no per-frame backlog / queue). We gate the compute so several
        submits pile up while one is in-flight, then confirm the final published frame is the last one."""
        gate = __import__("threading").Event()
        first_started = __import__("threading").Event()

        def compute(img, roll, pitch):
            first_started.set()
            gate.wait(1.0)                 # hold the FIRST compute so submits 2..N pile up (latest-wins)
            return _heading_estimate(yaw=0.0)

        w = VpYawWorker(compute)
        w.start()
        try:
            w.submit(_frame(1), 0.0, 0.0)
            assert first_started.wait(1.0), "worker never picked up the first frame"
            # While the first compute is blocked, submit a burst -> latest-wins collapses them to #9.
            for fid in range(2, 10):
                w.submit(_frame(fid), 0.0, 0.0)
            gate.set()                     # release: worker finishes #1, then runs ONLY the freshest (#9)
            assert _wait_for(lambda: (w.latest() is not None and w.latest().frame_id == 9))
            # It must NOT have computed all 9 frames (the backlog collapsed) -- 1 (#1) + 1 (#9) = 2-ish.
            assert w.n_computes <= 3, f"backlog not collapsed: {w.n_computes} computes for 9 submits"
        finally:
            w.stop()

    def test_worker_exception_counted_not_raised(self):
        def boom(img, roll, pitch):
            raise ValueError("compute blew up")

        w = VpYawWorker(boom)
        w.start()
        try:
            w.submit(_frame(1), 0.0, 0.0)
            assert _wait_for(lambda: w.n_errors >= 1), "worker error was never counted"
            assert w.latest() is None, "a failed compute must not publish a result"
            assert w.last_error is not None and "compute blew up" in w.last_error
            # The worker thread must still be alive (an exception must never kill it).
            assert w._thread.is_alive()
        finally:
            w.stop()

    def test_submit_same_frame_id_not_duplicated(self):
        n = {"calls": 0}

        def compute(img, roll, pitch):
            n["calls"] += 1
            time.sleep(0.01)
            return _heading_estimate()

        w = VpYawWorker(compute)
        w.start()
        try:
            # Two submits of the SAME frame_id before the worker drains -> at most one compute for it.
            w.submit(_frame(5), 0.0, 0.0)
            w.submit(_frame(5), 0.0, 0.0)
            assert _wait_for(lambda: w.latest() is not None and w.latest().frame_id == 5)
            time.sleep(0.05)
            assert n["calls"] == 1, f"same frame_id computed {n['calls']}x (expected 1)"
        finally:
            w.stop()


# ======================================================================================
# R1b. Navigator wiring: OFF byte-identical; ON bounds yaw drift; worker error safe
# ======================================================================================
def _gate_facing_north(position, gate_id=0, inner=1.5) -> Gate:
    R = np.column_stack([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    return Gate(gate_id=gate_id, position_ned=np.asarray(position, float),
                R_world_gate=R, inner_size_m=inner)


def _project_gate(gate: Gate, drone_pos, R_wb, inner=1.5) -> np.ndarray:
    R_camera_world = (R_wb @ R_camera_from_body().T).T
    t_cam_gate = R_camera_world @ (gate.position_ned - np.asarray(drone_pos, float))
    R_cam_gate = R_camera_world @ gate.R_world_gate
    return project_gate_corners(R_cam_gate, t_cam_gate, inner_size_m=inner)


class _FakeDetector:
    def __init__(self, gate: Gate, drone_pos, inner=1.5):
        self.gate, self.drone_pos, self.inner = gate, np.asarray(drone_pos, float), inner

    def detect(self, frame: Frame):
        corners = _project_gate(self.gate, self.drone_pos, np.eye(3), self.inner)
        return [GateObservation(
            frame_id=frame.frame_id, sim_time_ns=frame.sim_time_ns,
            corners_px=corners, corner_confidence=np.ones(4), score=0.9,
        )]


def _nav_frame(frame_id, sim_time_ns):
    return Frame(frame_id=frame_id, sim_time_ns=int(sim_time_ns),
                 image_bgr=np.zeros((360, 640, 3), np.uint8), recv_monotonic_ns=0)


def _ds(sim_time_ns, *, gyro_body=None, active_gate_index=0):
    return DroneState(
        sim_time_ns=int(sim_time_ns),
        orientation_ned_wxyz=_LEVEL_Q.copy(),
        roll=0.0, pitch=0.0, yaw=0.0,
        angular_rate_body=np.zeros(3),
        accel_body=_HOVER_ACCEL.copy(),
        gyro_body=(np.zeros(3) if gyro_body is None else np.asarray(gyro_body, float)),
        position_ned=None,
        active_gate_index=active_gate_index,
    )


def _make_nav(gate, drone_pos, **flags):
    cfg = NavigatorConfig(
        use_given_position=False, use_given_velocity=False,
        use_vision=True, use_ahrs=True, reconcile_vision_clock=False,
        **flags,
    )
    return Navigator(gates=[gate], detector=_FakeDetector(gate, drone_pos), config=cfg)


class TestVpYawAsyncNavigatorWiring:
    def test_off_is_byte_identical_sync(self):
        """vp_yaw_async DEFAULTS to False; a nav that explicitly sets it False produces bit-identical
        KF (x,P) + AHRS attitude to one that leaves it at default, over a synthetic run with vp_yaw ON
        (stubbed estimate_heading). No worker thread is constructed on either."""
        import racer.navigator as navmod

        def fake_heading(image_bgr, roll, pitch, **kw):
            return _heading_estimate(yaw=0.0)

        gate = _gate_facing_north([30.0, 0.0, 0.0])
        drone = [0.0, 0.0, 0.0]
        orig = navmod.estimate_heading
        try:
            navmod.estimate_heading = fake_heading
            nav_default = _make_nav(gate, drone, use_vp_yaw=True)                    # vp_yaw_async default
            nav_explicit = _make_nav(gate, drone, use_vp_yaw=True, vp_yaw_async=False)
            assert nav_default.config.vp_yaw_async is False
            for k in range(1, 31):
                t = k * _DT_NS
                ds = _ds(t, gyro_body=np.array([0.0, 0.0, np.deg2rad(6.0)]))
                nav_default.update(ds, _nav_frame(k, t))
                nav_explicit.update(ds, _nav_frame(k, t))
        finally:
            navmod.estimate_heading = orig
        np.testing.assert_array_equal(nav_default.kf.x, nav_explicit.kf.x)
        np.testing.assert_array_equal(nav_default.kf.P, nav_explicit.kf.P)
        np.testing.assert_array_equal(nav_default._ahrs.q_wxyz, nav_explicit._ahrs.q_wxyz)
        # OFF path never constructs a worker.
        assert nav_default._vp_yaw_worker is None and nav_explicit._vp_yaw_worker is None

    def test_async_worker_honours_monkeypatched_estimate_heading_and_bounds_drift(self):
        """ON vp_yaw_async: the worker's compute closure resolves the MONKEYPATCHED
        racer.navigator.estimate_heading (the test stub), and over a yaw-drifting run the applied VP
        yaw bounds the drift a no-VP run diverges under. Proves the async path (a) uses the patched fn
        and (b) actually reaches ESKFAHRS.update_yaw on the loop thread."""
        import racer.navigator as navmod

        def fake_heading(image_bgr, roll, pitch, **kw):
            return _heading_estimate(yaw=0.0)   # datum at TRUE yaw 0

        gate = _gate_facing_north([30.0, 0.0, 0.0])
        drone = [0.0, 0.0, 0.0]
        bias_z = np.deg2rad(8.0)
        orig = navmod.estimate_heading
        try:
            navmod.estimate_heading = fake_heading
            nav = _make_nav(gate, drone, use_vp_yaw=True, vp_yaw_async=True, vp_yaw_min_quality=0.3)
            try:
                for k in range(1, 401):
                    t = k * _DT_NS
                    ds = _ds(t, gyro_body=np.array([0.0, 0.0, bias_z]))
                    nav.update(ds, _nav_frame(k, t))
                    # Give the worker a beat so its first result is ready to consume on later ticks.
                    if k < 5:
                        time.sleep(0.01)
                # Ensure at least one VP yaw was actually applied (the async result was consumed).
                assert _wait_for(lambda: nav.n_vp_yaw_total > 0, timeout_s=1.0), \
                    "async VP yaw never applied"
                # Drain a few more ticks so the bounded steady-state is reached.
                for k in range(401, 460):
                    t = k * _DT_NS
                    nav.update(_ds(t, gyro_body=np.array([0.0, 0.0, bias_z])), _nav_frame(k, t))
                    time.sleep(0.003)
                y_on = euler_from_quat_wxyz(nav._ahrs.q_wxyz)[2]
            finally:
                nav.close_vp_yaw_worker()

            # Reference: identical run with vp_yaw OFF diverges.
            nav_off = _make_nav(gate, drone, use_vp_yaw=False)
            for k in range(1, 460):
                t = k * _DT_NS
                nav_off.update(_ds(t, gyro_body=np.array([0.0, 0.0, bias_z])), _nav_frame(k, t))
            y_off = euler_from_quat_wxyz(nav_off._ahrs.q_wxyz)[2]
        finally:
            navmod.estimate_heading = orig

        assert abs(np.rad2deg(y_off)) > 15.0, "yaw should drift far with VP off"
        assert abs(np.rad2deg(y_on)) < 8.0, f"async VP yaw should bound the drift (got {np.rad2deg(y_on):.1f})"
        assert abs(np.rad2deg(y_on)) < abs(np.rad2deg(y_off))

    def test_async_worker_error_does_not_reach_loop(self):
        """A raising estimate_heading on the async path is swallowed by the worker (counted), the
        control loop keeps running, and NO VP yaw is applied (graceful no-VP degrade -- no crash)."""
        import racer.navigator as navmod

        def boom(image_bgr, roll, pitch, **kw):
            raise RuntimeError("estimate_heading exploded")

        gate = _gate_facing_north([30.0, 0.0, 0.0])
        orig = navmod.estimate_heading
        try:
            navmod.estimate_heading = boom
            nav = _make_nav(gate, [0.0, 0.0, 0.0], use_vp_yaw=True, vp_yaw_async=True)
            try:
                for k in range(1, 40):
                    t = k * _DT_NS
                    nav.update(_ds(t, gyro_body=np.zeros(3)), _nav_frame(k, t))  # must NOT raise
                    time.sleep(0.003)
                assert _wait_for(lambda: nav._vp_yaw_worker is not None
                                 and nav._vp_yaw_worker.n_errors >= 1, timeout_s=1.0)
                assert nav.n_vp_yaw_total == 0, "no VP yaw should apply when every compute fails"
                assert np.all(np.isfinite(nav._ahrs.q_wxyz))   # estimate stayed sane
            finally:
                nav.close_vp_yaw_worker()
        finally:
            navmod.estimate_heading = orig


# ======================================================================================
# R2. MavlinkClient.parse_actuator_output
# ======================================================================================
from racer.mavlink_client import MavlinkClient  # noqa: E402


def _actuator_msg(motors=(0.1, 0.2, 0.3, 0.4)):
    m = SimpleNamespace(time_usec=42, actuator=list(motors) + [0.0, 0.0, 0.0, 0.0])
    m.get_type = lambda: "ACTUATOR_OUTPUT_STATUS"
    return m


def _collision_msg(cid=1001, threat=2, impulse=3.5):
    m = SimpleNamespace(id=cid, threat_level=threat, horizontal_minimum_delta=impulse)
    m.get_type = lambda: "COLLISION"
    return m


class TestActuatorParseFlag:
    def test_default_true_parses_actuator(self):
        """DEFAULT (parse_actuator_output=True) is byte-identical: the ACTUATOR msg populates
        actuator_outputs exactly as before (test_mavlink_client + rate_sysid depend on this)."""
        c = MavlinkClient()
        assert c.parse_actuator_output is True
        c._handle(_actuator_msg(motors=(0.1, 0.2, 0.3, 0.4)))
        assert c.actuator_outputs is not None
        assert c.actuator_outputs["motors"] == [0.1, 0.2, 0.3, 0.4]

    def test_flag_false_skips_actuator_parse(self):
        """parse_actuator_output=False: an ACTUATOR msg no longer mutates actuator_outputs (stays
        None) -- the per-msg dict/list-comp build is skipped on the flown loop."""
        c = MavlinkClient(parse_actuator_output=False)
        assert c.parse_actuator_output is False
        c._handle(_actuator_msg())
        assert c.actuator_outputs is None, "actuator parse should be skipped when the flag is False"

    def test_flag_false_leaves_other_msg_types_unaffected(self):
        """The flag ONLY gates the ACTUATOR branch: a COLLISION still parses normally when parsing is off."""
        c = MavlinkClient(parse_actuator_output=False)
        c._handle(_collision_msg(cid=1001, threat=2, impulse=3.5))
        assert len(c.collisions) == 1
        assert c.collisions[0]["id"] == 1001 and c.collisions[0]["threat_level"] == 2
        # ... and the actuator branch is still inert.
        c._handle(_actuator_msg())
        assert c.actuator_outputs is None

    def test_tlog_tap_independent_of_parse_flag(self):
        """SAFETY: the raw-message tap (on_message -> the recorder's tlog write) fires on the ACTUATOR
        msg REGARDLESS of the parse flag, because pump() calls on_message BEFORE _handle. This is the
        R2 tlog-safety invariant: skipping the parse cannot change what lands in mavlink.tlog.

        Driven through the REAL pump() with a fake conn feeding one ACTUATOR msg then draining."""
        tapped: list = []

        class _OneShotConn:
            def __init__(self, msgs):
                self._msgs = list(msgs)

            def recv_match(self, blocking=False):
                return self._msgs.pop(0) if self._msgs else None

        for parse in (True, False):
            c = MavlinkClient(parse_actuator_output=parse)
            c.conn = _OneShotConn([_actuator_msg()])
            c.send_heartbeats = False
            c.send_timesync = False
            got = []
            c.on_message = lambda m: got.append(m.get_type())
            c.pump()
            # on_message saw the ACTUATOR msg no matter the parse flag (tlog would record it either way).
            assert got == ["ACTUATOR_OUTPUT_STATUS"], f"tap missed the msg (parse={parse})"
            # The parse flag only decides whether actuator_outputs was populated.
            if parse:
                assert c.actuator_outputs is not None
            else:
                assert c.actuator_outputs is None
            tapped.append(got)
        assert tapped == [["ACTUATOR_OUTPUT_STATUS"], ["ACTUATOR_OUTPUT_STATUS"]]


# ======================================================================================
# Q3. Phase timing + nav-record wall stamp (end-to-end through the REAL _fly_gate_seeker)
# ======================================================================================
import fly_rl  # noqa: E402


class _FakeClientQ3:
    """Just enough of MavlinkClient for _fly_gate_seeker's tick loop (mirrors the perf_summary test):
    advancing sim clock, no collisions/reset, map-free branch."""

    def __init__(self):
        self.state = SimpleNamespace(
            sim_time_ns=1_000_000_000, reset_counter=0,
            position_ned=np.array([0.0, 0.0, -1.0], dtype=np.float64),
        )
        self.race_status = {"active_gate_index": 0, "finished": False}
        self.collisions: list = []
        self.track_gates = None
        self.cmd_rate_scale = 0.4
        self._latest_frame = None

    def pump(self):
        self.state.sim_time_ns += 1_000_000

    def send_command(self, cmd):
        pass


class _FakeNavQ3:
    _ahrs = None
    _vp_yaw_worker = None

    def update(self, s, frame):
        ns = SimpleNamespace()
        ns.roll, ns.pitch, ns.yaw = 0.0, 0.0, 0.0
        ns.position_ned = np.array([1.0, 2.0, -3.0], dtype=np.float64)
        ns.time_since_vision_update_s = 0.1
        return ns

    def close_vp_yaw_worker(self):
        pass


class _FakeSeekerQ3:
    def __init__(self, max_calls):
        self._n = 0
        self._max = max_calls

    def command_visual(self, nav_state, frame, gate_index, is_final_gate=False):
        self._n += 1
        if self._n >= self._max:
            raise KeyboardInterrupt()   # clean stop after N ticks (evidence still written in finally)
        cmd = SimpleNamespace()
        cmd.body_rate = np.array([0.1, -0.2, 0.05], dtype=np.float64)
        cmd.thrust = 0.55
        return cmd


def _q3_args():
    return SimpleNamespace(
        deploy_profile="vq2_case_c", seeker_detector="none", seeker_speed=3.0,
        max_seconds=10.0, rate=200.0, ignore_collisions=True,
        async_detect="off", vertical_estimator="off",
    )


class TestQ3PhaseTimingAndStamp:
    def test_phase_buckets_and_nav_stamp_land(self, tmp_path, monkeypatch):
        nav, seeker, client = _FakeNavQ3(), _FakeSeekerQ3(max_calls=6), _FakeClientQ3()
        monkeypatch.setattr(
            fly_rl, "_build_casec_seeker",
            lambda args, gates, frame_source=None: (nav, seeker, None, None))
        result = {"flight": 1, "final_state": "NO_GO", "gate_index": 0, "collisions_at_start": 0}

        with pytest.raises(KeyboardInterrupt):
            fly_rl._fly_gate_seeker(client, _q3_args(), 1, tmp_path, result)

        # perf_summary carries the loop_phase_ms buckets.
        perf = json.loads((tmp_path / "perf_summary.json").read_text(encoding="utf-8"))
        assert "loop_phase_ms" in perf, "loop_phase_ms missing from perf_summary"
        phase = perf["loop_phase_ms"]
        for bucket in ("pump", "nav", "seeker", "ctrl_send", "log"):
            assert bucket in phase, f"phase bucket {bucket!r} missing"
        # The work phases actually ran (>=1 tick before the clean stop).
        assert phase["nav"]["count"] >= 1
        assert phase["seeker"]["count"] >= 1
        assert phase["ctrl_send"]["count"] >= 1
        assert phase["pump"]["count"] >= 1

        # Each nav_estimate record carries the Q3 wall stamp (monotonic ns).
        rows = [json.loads(l) for l in
                (tmp_path / "nav_estimate.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(rows) >= 1
        for r in rows:
            assert "t_mono_ns" in r and isinstance(r["t_mono_ns"], int) and r["t_mono_ns"] > 0
        # Stamps are monotonically non-decreasing across ticks (a real wall-clock interval signal).
        stamps = [r["t_mono_ns"] for r in rows]
        assert all(b >= a for a, b in zip(stamps, stamps[1:])), "t_mono_ns not monotonic across ticks"
