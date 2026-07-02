"""Asynchronous gate-detection worker — decouple the GPU detect from the control loop.

THE PROBLEM (A19/A20, ShadowPC 2026-07-01): the gate-seeker control loop calls the YOLO
detector SYNCHRONOUSLY every tick. On the Shadow cloud VM the detector's CUDA calls stall
~250 ms behind the UE sim's GPU rendering (host-level GPU arbitration on the passthrough
RTX 2000 Ada — confirmed NOT fixable guest-side), so the loop runs at ~3-4 Hz instead of
30 Hz. The sim applies ZERO-ORDER-HOLD between commands, so each ~300 ms gap lets one climb
command fly the drone into the ceiling (crash at gate 0 in ~3 s, every flight).

THE FIX (this module): run detection CONTINUOUSLY in ONE daemon worker thread on the
freshest available camera frame, and publish a latest-wins, atomically-swapped
:class:`DetectResult` snapshot. The control loop ticks at its fixed rate and each tick
consumes the freshest COMPLETED result (however old) WITHOUT ever blocking on detect. The
estimator already gyro-propagates attitude between vision fixes and the RewindKF applies a
fix at its CAPTURE time (OOSM), so a stale-but-timestamped observation is exactly the
intended design; the ZOH command-hold shrinks from ~300 ms back to ~33 ms.

Why a thread works: torch's CUDA synchronization points (the blocking ``.cpu()`` D2H
readbacks in ``observations_from_results`` and ultralytics' internal syncs) RELEASE the GIL
while waiting on the GPU (ATen ops drop the GIL at dispatch), so the control-loop thread
keeps running through the ~250 ms stall. The numpy/OpenCV pre/post-processing inside
``predict`` holds the GIL only for ms-scale stretches. (Not empirically re-confirmed on the
live sim — see the FINDINGS risk note — but this is documented torch behaviour.)

Two pieces:

- :class:`AsyncDetectWorker` — owns the ONE detector instance + the ONE worker thread.
  Pulls the freshest frame from an injected ``frame_source`` callable (the video thread's
  ``client._latest_frame``), runs ``detector.detect`` once per NEW frame_id, and publishes
  the result under a lock (single slot, latest wins, no queue). Any detect error is counted
  and swallowed (the worker must never die mid-flight); the previous published result stays.

- :class:`AsyncDetectorProxy` — a duck-typed detector (``.detect(frame) -> [GateObservation]``)
  handed to the Navigator + GateSeeker IN PLACE of the real detector when async mode is ON.
  Its ``detect`` NEVER runs inference: it returns the worker's published observations when the
  asked-for ``frame_id`` matches the published one, else ``[]``. Because the control loop only
  ever feeds the navigator/seeker the worker's own published frame, the ids match by
  construction, and both consumers keep flowing through the existing ``detect_cached``
  per-frame_id cache unchanged (they share the ONE proxy instance, mirroring how they shared
  the one real detector). This keeps the navigator/gate_seeker code byte-identical — only the
  object injected at construction differs, and only when the async flag is ON.

OFF path: nothing here is imported unless the flag is ON (DeployProfile.async_detect /
fly_rl --async-detect), so the synchronous behaviour is byte-for-byte unchanged.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from racer.contracts import Frame, GateObservation


@dataclass(frozen=True)
class DetectResult:
    """One completed detection pass: the frame it ran on + its observations + timing.

    Immutable + atomically swapped into the worker's single slot, so a reader always sees a
    CONSISTENT (frame, observations) pair — no torn reads. ``observations`` is a tuple (not a
    list) so the published snapshot itself can never be mutated in place."""

    frame: Frame
    observations: tuple[GateObservation, ...]
    t_done_monotonic: float   # time.monotonic() when detect() returned (freshness reference)
    detect_ms: float          # wall-clock cost of this detect() call

    def age_ms(self, now_monotonic_ns: int | None = None) -> float:
        """Age of the underlying CAMERA FRAME (ms) — receipt-to-now, i.e. the total staleness
        of the vision this result carries (frame transport + detect latency + shelf time)."""
        now = time.monotonic_ns() if now_monotonic_ns is None else int(now_monotonic_ns)
        return max(0.0, (now - int(self.frame.recv_monotonic_ns)) / 1e6)


class AsyncDetectWorker:
    """ONE detector, ONE daemon thread, ONE latest-wins result slot.

    ``frame_source`` is a zero-arg callable returning the freshest decoded :class:`Frame` (or
    ``None``); the live wiring is ``lambda: client._latest_frame`` (published by the video
    thread). The worker re-detects only on a NEW ``frame_id`` and idles (short waits) between.

    Stats fields (``n_detects`` / ``total_detect_ms`` / ``max_detect_ms`` / ``n_errors``) are
    plain ints/floats written only by the worker thread and read by the control loop — safe
    under the GIL for the logging they feed (mirrors the video thread's ``vstats`` pattern).
    """

    def __init__(self, detector, frame_source, *, poll_s: float = 0.002,
                 name: str = "async-detect"):
        self._detector = detector
        self._frame_source = frame_source
        self._poll_s = float(poll_s)
        self._lock = threading.Lock()
        self._latest: DetectResult | None = None
        # The immediately-previous result, kept ONLY to close a benign race: the control loop
        # snapshots frame N, the worker publishes N+1, and the loop's detect_cached(frame N)
        # then asks the proxy for N -- serving N from _prev instead of dropping that tick's
        # detections. Never consumed as "the latest"; latest() returns _latest alone.
        self._prev: DetectResult | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._t_start: float | None = None
        # -- stats (worker writes, control loop reads; logging only) --
        self.n_detects: int = 0
        self.total_detect_ms: float = 0.0
        self.max_detect_ms: float = 0.0
        self.n_errors: int = 0
        self.last_error: str | None = None

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        self._t_start = time.monotonic()
        self._thread.start()

    def stop(self, join_timeout_s: float = 3.0) -> None:
        """Signal the worker to exit and join briefly. The thread is a daemon, so a detect
        stuck inside a GPU stall past the timeout cannot block process exit."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=join_timeout_s)

    # -- consumer side (control loop) ----------------------------------------
    def latest(self) -> DetectResult | None:
        """The freshest COMPLETED detection result (never blocks; None until the first one)."""
        with self._lock:
            return self._latest

    def fps(self) -> float:
        """Completed detections per second since start() (the 'vision fps' instrumentation)."""
        if self._t_start is None:
            return 0.0
        elapsed = max(time.monotonic() - self._t_start, 1e-6)
        return self.n_detects / elapsed

    def mean_detect_ms(self) -> float:
        return self.total_detect_ms / self.n_detects if self.n_detects else 0.0

    # -- worker thread --------------------------------------------------------
    def _pin_cuda_device(self) -> None:
        """Best-effort: make the detector's CUDA device current IN THIS THREAD before the first
        inference (torch contexts are per-process, but an explicit set_device removes any
        ambiguity on a multi-GPU box). Failures are swallowed — worst case torch resolves the
        device per-op exactly as it does on the synchronous path."""
        try:
            dev = getattr(self._detector, "device", None)
            if dev is not None and str(dev).startswith("cuda"):
                import torch

                torch.cuda.set_device(dev)
        except Exception:
            pass

    def _run(self) -> None:
        self._pin_cuda_device()
        last_fid: int | None = None
        while not self._stop.is_set():
            try:
                frame = self._frame_source()
            except Exception as exc:      # a frame-source bug must not kill the worker
                self.n_errors += 1
                self.last_error = repr(exc)
                frame = None
            if (frame is None or getattr(frame, "image_bgr", None) is None
                    or frame.frame_id == last_fid):
                self._stop.wait(self._poll_s)   # nothing new: idle briefly, stay responsive
                continue
            last_fid = int(frame.frame_id)
            t0 = time.perf_counter()
            try:
                obs = self._detector.detect(frame)
            except Exception as exc:
                # Count + remember, keep the previous published result (stale beats dead).
                self.n_errors += 1
                self.last_error = repr(exc)
                continue
            dt_ms = (time.perf_counter() - t0) * 1e3
            self.n_detects += 1
            self.total_detect_ms += dt_ms
            if dt_ms > self.max_detect_ms:
                self.max_detect_ms = dt_ms
            result = DetectResult(frame=frame, observations=tuple(obs),
                                  t_done_monotonic=time.monotonic(), detect_ms=dt_ms)
            with self._lock:
                self._prev = self._latest
                self._latest = result

    def result_for_frame_id(self, frame_id) -> DetectResult | None:
        """The published result carrying ``frame_id`` (latest, or the immediately-previous one),
        else None. The proxy's lookup — checking _prev closes the publish-vs-consume race."""
        with self._lock:
            for res in (self._latest, self._prev):
                if res is not None and res.frame.frame_id == frame_id:
                    return res
        return None


class AsyncDetectorProxy:
    """Duck-typed detector for the Navigator + GateSeeker in async mode: ``.detect(frame)``
    NEVER runs inference — it serves the worker's published observations for that frame_id.

    The control loop only feeds the navigator/seeker the worker's OWN published frame, so the
    id matches by construction; a mismatch (possible only in a pathological race) returns
    ``[]`` — a no-detection tick, the same graceful degrade the loop already handles — rather
    than ever blocking on the model. Both consumers must share the ONE proxy instance so the
    existing ``detect_cached`` per-frame_id memoization keeps handing them the identical list
    (exactly as they shared the one real detector on the sync path)."""

    def __init__(self, worker: AsyncDetectWorker):
        self._worker = worker

    def detect(self, frame: Frame) -> list[GateObservation]:
        if frame is None:
            return []
        res = self._worker.result_for_frame_id(getattr(frame, "frame_id", None))
        return list(res.observations) if res is not None else []
