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


@dataclass(frozen=True)
class VpYawResult:
    """One completed vanishing-point heading pass: the frame_id it ran on + the HeadingEstimate
    (or None when the CV found no usable VP) + the roll/pitch used + timing.

    Immutable + atomically swapped into the worker's single slot (mirrors DetectResult). The
    ``estimate`` is whatever ``compute_fn`` returned — a ``racer.vision.heading_vp.HeadingEstimate``
    (itself frozen) or ``None`` — carried verbatim so the loop-thread consumer applies the SAME
    acceptance gates / branch disambiguation / update_yaw noise the synchronous path used (only the
    RANSAC compute moved off-thread, nothing else). ``roll``/``pitch`` are the gravity-known tilt at
    CAPTURE (submit time) that back-projected the VP pixels — logged for provenance, not re-used."""

    frame_id: object
    estimate: object            # HeadingEstimate | None (the compute_fn output, carried verbatim)
    roll: float
    pitch: float
    t_done_monotonic: float
    compute_ms: float


class VpYawWorker:
    """ONE vanishing-point-heading estimator, ONE daemon thread, ONE latest-wins result slot.

    THE PROBLEM (residual loop-choke, 2026-07-05): on the flown map-free path ``vp_yaw`` is the SOLE
    absolute-yaw anchor, and its ``estimate_heading`` VP-RANSAC + Manhattan-line extraction costs
    ~30-100 ms ON the control-loop thread — the single worst-tick spike (161 ms tick = 101 ms
    vp_yaw). ``vp_yaw_decimate=15`` cut its FREQUENCY but not its per-call cost, so it still spikes
    the tick it runs on.

    THE FIX (this worker): run the VP compute CONTINUOUSLY on a daemon thread, exactly like
    :class:`AsyncDetectWorker`. The control loop SUBMITS ``(frame, roll, pitch)`` (latest-wins:
    a fresh submit overwrites an unconsumed one, so the worker always runs the FRESHEST frame and
    never builds a backlog) and, on a later tick, CONSUMES the freshest COMPLETED
    :class:`VpYawResult` WITHOUT blocking. The loop thread then does the branch disambiguation
    (against the CURRENT gyro-propagated yaw estimate — fresh, on-thread) + ``ESKFAHRS.update_yaw``
    + attitude-cache refresh, so the acceptance gates, quality threshold, branch cap, and
    update_yaw noise are BYTE-IDENTICAL to the synchronous path. Only the RANSAC moved.

    WHY A ONE-FRAME-STALE MEASUREMENT IS FINE: yaw drifts ~0.5 deg/s in healthy flight, so a heading
    computed from a frame up to ~200 ms old carries ~0.1 deg of extra error — negligible next to the
    ~5 deg VP noise. Critically the branch snap uses the FRESH on-thread yaw estimate, so a stale
    frame can never cause a silent 90-deg branch flip (the disambiguation is done at APPLY time, not
    compute time). This mirrors the async-detect OOSM rationale (a stale-but-timestamped observation
    is the intended design).

    ``compute_fn(frame_bgr, roll, pitch) -> HeadingEstimate | None`` is injected so the worker stays
    generic AND so a test monkeypatch of ``racer.navigator.estimate_heading`` is honoured: the
    Navigator builds ``compute_fn`` as a small closure that looks the module-global up at CALL time.

    Stats (``n_computes`` / ``total_compute_ms`` / ``max_compute_ms`` / ``n_errors``) are plain
    ints/floats written only by the worker thread and read by the control loop (GIL-safe for the
    logging they feed; mirrors AsyncDetectWorker)."""

    def __init__(self, compute_fn, *, poll_s: float = 0.002, name: str = "vp-yaw"):
        self._compute_fn = compute_fn
        self._poll_s = float(poll_s)
        self._lock = threading.Lock()
        # -- submission slot (loop writes, worker drains; latest-wins so no backlog) --
        self._pending: tuple | None = None   # (frame_id, image_bgr, roll, pitch) or None
        self._pending_event = threading.Event()
        # -- result slot (worker writes, loop reads; latest-wins single slot) --
        self._latest: VpYawResult | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._t_start: float | None = None
        # -- stats (worker writes, control loop reads; logging only) --
        self.n_computes: int = 0
        self.total_compute_ms: float = 0.0
        self.max_compute_ms: float = 0.0
        self.n_errors: int = 0
        self.last_error: str | None = None
        # The frame_id most recently ACCEPTED for compute (pending OR already picked up). A re-submit
        # of the same id is dropped so the worker re-computes only on a NEW frame_id (mirrors
        # AsyncDetectWorker.last_fid). None ids (unusable/test frames) are never de-duped against each
        # other -- matching detect_cached's no-frame-id semantics.
        self._last_submitted_fid: object = None

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        self._t_start = time.monotonic()
        self._thread.start()

    def stop(self, join_timeout_s: float = 2.0) -> None:
        """Signal the worker to exit and join briefly (daemon => a compute past the timeout cannot
        block process exit)."""
        self._stop.set()
        self._pending_event.set()   # wake the worker if it is idling on the submission wait
        if self._thread.is_alive():
            self._thread.join(timeout=join_timeout_s)

    # -- producer side (control loop) ---------------------------------------
    def submit(self, frame, roll: float, pitch: float) -> None:
        """Offer a frame + capture-time tilt for the NEXT VP compute. Latest-wins: overwrites any
        unconsumed submission so the worker always runs the FRESHEST frame (never a backlog). A
        submission for a frame_id equal to the one already pending is dropped (no duplicate compute).
        Never blocks; frame image is referenced, not copied (the video thread owns immutable Frames)."""
        if frame is None or getattr(frame, "image_bgr", None) is None:
            return
        fid = getattr(frame, "frame_id", None)
        with self._lock:
            # Drop a re-submit of the most-recently-accepted frame_id (still pending OR already
            # picked up) -> re-compute only on a NEW frame_id. None ids skip the guard (never de-duped).
            if fid is not None and fid == self._last_submitted_fid:
                return
            self._pending = (fid, frame.image_bgr, float(roll), float(pitch))
            self._last_submitted_fid = fid
        self._pending_event.set()

    # -- consumer side (control loop) ---------------------------------------
    def latest(self) -> VpYawResult | None:
        """The freshest COMPLETED VP heading result (never blocks; None until the first one)."""
        with self._lock:
            return self._latest

    def mean_compute_ms(self) -> float:
        return self.total_compute_ms / self.n_computes if self.n_computes else 0.0

    # -- worker thread ------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            # Wait for a submission (event-driven, not a busy spin); short timeout so stop() is
            # responsive even if it raced the event.
            self._pending_event.wait(self._poll_s)
            if self._stop.is_set():
                break
            with self._lock:
                job = self._pending
                self._pending = None
                self._pending_event.clear()
            if job is None:
                continue
            fid, image_bgr, roll, pitch = job
            t0 = time.perf_counter()
            try:
                est = self._compute_fn(image_bgr, roll, pitch)
            except Exception as exc:
                # Count + remember, keep the previous published result (stale beats dead). A
                # compute-fn bug must NEVER kill the worker or reach the control loop.
                self.n_errors += 1
                self.last_error = repr(exc)
                continue
            dt_ms = (time.perf_counter() - t0) * 1e3
            self.n_computes += 1
            self.total_compute_ms += dt_ms
            if dt_ms > self.max_compute_ms:
                self.max_compute_ms = dt_ms
            result = VpYawResult(frame_id=fid, estimate=est, roll=roll, pitch=pitch,
                                 t_done_monotonic=time.monotonic(), compute_ms=dt_ms)
            with self._lock:
                self._latest = result


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
