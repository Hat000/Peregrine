"""Async-detect decoupling (A20) — the worker/proxy pair that unpins the 30 Hz control loop
from the ~250 ms GPU-arbitration detect stall (racer.vision.async_detect).

THE CONTRACT UNDER TEST:
  * the worker runs the ONE real detector in its OWN thread, once per NEW frame_id, and
    publishes an immutable latest-wins (frame, observations) snapshot — no queue, no backlog;
  * ``latest()`` / the proxy NEVER block on the model: a slow detect leaves the previous
    published result readable throughout;
  * the proxy serves the PUBLISHED observations for the published frame_id (including the
    immediately-previous one, closing the publish-vs-consume race) and returns [] otherwise —
    it must never fall through to real inference;
  * routed through the existing ``detect_cached`` per-frame_id memoization, the navigator and
    the seeker (sharing ONE proxy) receive the identical list object, exactly like the shared
    real detector on the synchronous path;
  * a detector exception is counted + swallowed (stale beats dead — the worker must survive);
  * the deploy-profile seam: vq2_case_c carries async_detect=True, vq1_case_a / the dataclass
    default stay False (the OFF path is the byte-identical synchronous loop).
"""
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from racer.contracts import Frame, GateObservation  # noqa: E402
from racer.deploy_profile import DeployProfile, get_profile  # noqa: E402
from racer.vision.async_detect import (  # noqa: E402
    AsyncDetectorProxy,
    AsyncDetectWorker,
    DetectResult,
)
from racer.vision.detector import detect_cached  # noqa: E402


def _frame(fid: int) -> Frame:
    return Frame(frame_id=fid, sim_time_ns=fid * 33_000_000,
                 image_bgr=np.zeros((4, 4, 3), dtype=np.uint8),
                 recv_monotonic_ns=time.monotonic_ns())


def _obs(fid: int) -> GateObservation:
    corners = np.array([[10.0, 20.0], [30.0, 20.0], [30.0, 5.0], [10.0, 5.0]])
    return GateObservation(frame_id=fid, sim_time_ns=fid * 33_000_000,
                           corners_px=corners, corner_ids=None,
                           corner_confidence=np.ones(4), score=0.9)


class _FakeDetector:
    """Deterministic detector: one observation tagged with the frame_id; optional per-call
    blocking (to emulate the GPU stall) and scripted failures."""

    def __init__(self, block_s: float = 0.0, fail_on: set | None = None):
        self.block_s = block_s
        self.fail_on = fail_on or set()
        self.calls: list[int] = []
        self.in_detect = threading.Event()
        self.release = threading.Event()

    def detect(self, frame: Frame):
        self.calls.append(int(frame.frame_id))
        if int(frame.frame_id) in self.fail_on:
            raise RuntimeError(f"scripted failure on frame {frame.frame_id}")
        if self.block_s:
            self.in_detect.set()
            self.release.wait(self.block_s)   # emulate the CUDA stall (releases the GIL too)
        return [_obs(int(frame.frame_id))]


class _FrameSource:
    """Latest-wins frame slot standing in for client._latest_frame."""

    def __init__(self):
        self.frame: Frame | None = None

    def __call__(self):
        return self.frame


def _wait_for(pred, timeout_s: float = 5.0) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if pred():
            return True
        time.sleep(0.005)
    return False


# ---------------------------------------------------------------------------
# 1. Worker publishes latest-wins results, once per NEW frame_id
# ---------------------------------------------------------------------------
def test_worker_publishes_latest_result_once_per_frame_id():
    det = _FakeDetector()
    src = _FrameSource()
    w = AsyncDetectWorker(det, src, poll_s=0.001)
    w.start()
    try:
        assert w.latest() is None                       # nothing published before any frame
        src.frame = _frame(1)
        assert _wait_for(lambda: w.latest() is not None)
        res = w.latest()
        assert res.frame.frame_id == 1
        assert [o.frame_id for o in res.observations] == [1]
        assert res.detect_ms >= 0.0

        # same frame_id re-published by the source -> NO re-detect (the worker deduplicates)
        time.sleep(0.05)
        assert det.calls == [1]

        src.frame = _frame(2)
        assert _wait_for(lambda: w.latest() is not None and w.latest().frame.frame_id == 2)
        assert det.calls == [1, 2]
        assert w.n_detects == 2
        assert w.fps() > 0.0
    finally:
        w.stop()


# ---------------------------------------------------------------------------
# 2. NEVER blocks: a slow in-flight detect leaves the previous result readable
# ---------------------------------------------------------------------------
def test_latest_never_blocks_on_a_slow_detect():
    det = _FakeDetector(block_s=10.0)                   # a pathological 10 s "GPU stall"
    src = _FrameSource()
    w = AsyncDetectWorker(det, src, poll_s=0.001)
    # seed one instant result FIRST (no blocking on the very first frame)
    det.block_s = 0.0
    w.start()
    try:
        src.frame = _frame(1)
        assert _wait_for(lambda: w.latest() is not None)
        det.block_s = 10.0
        src.frame = _frame(2)                           # the worker dives into the long stall
        assert _wait_for(lambda: det.in_detect.is_set())
        t0 = time.perf_counter()
        res = w.latest()                                # the control-loop read during the stall
        dt_ms = (time.perf_counter() - t0) * 1e3
        assert dt_ms < 50.0                             # non-blocking (lock hold is trivial)
        assert res is not None and res.frame.frame_id == 1   # the previous COMPLETED result
        proxy = AsyncDetectorProxy(w)
        assert proxy.detect(_frame(2)) == []            # frame 2 not completed -> no detections
        assert [o.frame_id for o in proxy.detect(res.frame)] == [1]
    finally:
        det.release.set()                               # un-stall so stop() joins fast
        w.stop()


# ---------------------------------------------------------------------------
# 3. Proxy + detect_cached: both consumers get the IDENTICAL list; prev closes the race
# ---------------------------------------------------------------------------
def test_proxy_through_detect_cached_shares_one_list_and_serves_prev():
    det = _FakeDetector()
    src = _FrameSource()
    w = AsyncDetectWorker(det, src, poll_s=0.001)
    w.start()
    try:
        f1 = _frame(1)
        src.frame = f1
        assert _wait_for(lambda: w.latest() is not None)
        proxy = AsyncDetectorProxy(w)
        nav_obs = detect_cached(proxy, f1)              # the navigator's call
        seeker_obs = detect_cached(proxy, f1)           # the seeker's call, same shared proxy
        assert nav_obs is seeker_obs                    # identical object (the sync-path invariant)
        assert [o.frame_id for o in nav_obs] == [1]

        # publish frame 2; the previous (frame 1) result must STILL be servable (race closure)
        f2 = _frame(2)
        src.frame = f2
        assert _wait_for(lambda: w.latest() is not None and w.latest().frame.frame_id == 2)
        assert [o.frame_id for o in proxy.detect(f1)] == [1]     # served from _prev
        assert [o.frame_id for o in proxy.detect(f2)] == [2]     # served from _latest
        assert proxy.detect(_frame(99)) == []                    # unknown fid -> no detections
        # crucially: the proxy NEVER touched the real model for any of the above
        assert det.calls == [1, 2]
    finally:
        w.stop()


# ---------------------------------------------------------------------------
# 4. A detector exception is swallowed; the worker survives and keeps serving
# ---------------------------------------------------------------------------
def test_worker_survives_detector_exception():
    det = _FakeDetector(fail_on={2})
    src = _FrameSource()
    w = AsyncDetectWorker(det, src, poll_s=0.001)
    w.start()
    try:
        src.frame = _frame(1)
        assert _wait_for(lambda: w.latest() is not None)
        src.frame = _frame(2)                           # scripted failure
        assert _wait_for(lambda: w.n_errors >= 1)
        assert w.latest().frame.frame_id == 1           # stale beats dead: previous result kept
        assert "scripted failure" in (w.last_error or "")
        src.frame = _frame(3)                           # recovery on the next frame
        assert _wait_for(lambda: w.latest() is not None and w.latest().frame.frame_id == 3)
    finally:
        w.stop()


# ---------------------------------------------------------------------------
# 5. Published snapshots are immutable + consistently paired
# ---------------------------------------------------------------------------
def test_detect_result_snapshot_is_immutable_and_consistent():
    res = DetectResult(frame=_frame(7), observations=(_obs(7),),
                       t_done_monotonic=time.monotonic(), detect_ms=1.0)
    assert isinstance(res.observations, tuple)          # cannot be mutated in place
    assert res.observations[0].frame_id == res.frame.frame_id
    assert res.age_ms() >= 0.0


# ---------------------------------------------------------------------------
# 6. Deploy-profile seam: vq2_case_c ON, vq1_case_a / dataclass default OFF
# ---------------------------------------------------------------------------
def test_profile_async_detect_wiring():
    assert get_profile("vq2_case_c").async_detect is True
    assert get_profile("vq1_case_a").async_detect is False
    # the dataclass DEFAULT is OFF -> any existing/other profile construction stays synchronous
    import dataclasses
    fld = {f.name: f for f in dataclasses.fields(DeployProfile)}["async_detect"]
    assert fld.default is False
