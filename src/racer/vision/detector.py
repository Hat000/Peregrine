"""Detector adapter — completes the SENSE seam: a camera :class:`Frame` -> zero or more
:class:`GateObservation` (corners in pixels), ready for ``gate_pose.estimate_gate_pose``.

The runtime detector is an ultralytics **YOLO-pose** model that emits, per detected gate, a
box + 4 keypoints (each with a confidence). Two layers here:

- :func:`observations_from_keypoints` — the pure, array-only core (no ultralytics, no model):
  thresholds detections + keypoints and builds the contract. Fully unit-testable today.
- :class:`GateDetector` — the thin runtime wrapper that loads a YOLO-pose model and runs it.
  ``ultralytics`` is imported LAZILY (it's the heavy, GPU-only ``[detector]`` extra), so the
  core stack imports fine without it; only constructing a ``GateDetector`` needs it installed.

Conventions this adapter depends on (keep them in lockstep or PnP silently produces garbage):
- Keypoint index == canonical corner id: 0=lower-left, 1=lower-right, 2=upper-right,
  3=upper-left — the exact order ``synthetic.py`` writes its labels in and ``gate_pose``
  expects. The model is trained on those labels, so its keypoint i IS corner i; we never
  reorder, we only subset.
- When fewer than 4 keypoints clear the confidence threshold (a gate clipping out of frame at
  transit, or occlusion), we emit the visible subset with ``corner_ids`` set — which routes
  straight into ``gate_pose``'s 3-corner P3P fallback. Below 3 visible we drop the detection
  (a pose is unrecoverable). Data association (assigning ``gate_id``) is the mapper's job, not
  the raw detector's, so ``gate_id`` stays ``None`` here.

OPT-IN ENSEMBLE (deploy gate, 2026-06-19). The single-model :class:`GateDetector` path is the
VQ1-proven default and is left BYTE-IDENTICAL. When 2+ weights are configured (an ``a++b`` spec,
or :class:`EnsembleGateDetector` directly), the detections of every member are UNIONED and then
collapsed by a light per-gate dedup (:meth:`EnsembleGateDetector._dedup`) so the navigator's KF
receives ONE fix per gate instead of one-per-model. The dedup lives HERE, at the detector
boundary (before association), precisely so the downstream navigator / association / PnP path is
untouched — the ensemble is a drop-in detector. (The offline ``eval_ensemble`` unions with NO
dedup because its metrics self-select the best obs per gate; the deploy path feeds the union into
the KF, which would double-count, so the dedup is the genuinely new deploy logic.)

Wiring: this is the SENSE step behind ``Mission.run``'s navigator —
``frame -> detect() -> [GateObservation] -> estimate_gate_pose -> localization -> KF``.
"""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np

from racer.contracts import Frame, GateObservation

N_CORNERS = 4


def _resolve_device(requested: str | None) -> str:
    """Resolve the inference device for a YOLO model.

    A15/A17 FRAME-STARVATION FIX (2026-07-01): ultralytics ``YOLO(weights)`` loads to CPU and never
    moves itself to the GPU unless ``.to('cuda')`` is called (or a ``device=`` is threaded into every
    ``predict``); with ``device=None`` (our old default) ``predict`` runs on the model's CURRENT
    device -- i.e. CPU. On ShadowPC (sim + fly_rl co-located, GPU present) that silently ran YOLO on
    the CPU at ~150 ms/frame (~7x the ~21 ms single-YOLO GPU cost), choking the loop to ~10 Hz and
    starving the video receiver. This resolves the device EXPLICITLY:
      * ``requested`` given (e.g. 'cuda:0', 'cpu') -> honoured verbatim (an override / a test).
      * ``requested`` None -> 'cuda:0' when CUDA is available, else 'cpu'.
    Defensive: if torch import / the cuda probe raises (no torch, CPU-only build, driver issue) it
    falls back to today's behaviour (return ``requested`` unchanged, i.e. None -> ultralytics decides),
    so a warmup/probe miss can NEVER abort a run. Returns the resolved device string (or the original
    ``requested`` on the fallback path)."""
    if requested is not None:
        return requested
    try:
        import torch  # lazy: torch is only present with the [detector] extra

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return requested  # torch missing / probe failed -> today's behaviour (ultralytics decides)


def _move_model_to_device(model, device: str | None) -> None:
    """Best-effort move an ultralytics model onto ``device`` (``model.to(device)``) so its weights
    live where inference runs -- the actual GPU placement that ``device=`` alone in predict does not
    guarantee to persist. No-op when ``device`` is None or the model has no ``.to`` (an injected test
    fake); any failure is swallowed (never abort a run over a placement miss)."""
    if device is None or not hasattr(model, "to"):
        return
    try:
        model.to(device)
    except Exception:
        pass


def observations_from_keypoints(
    frame: Frame,
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
    det_scores: np.ndarray,
    *,
    score_thresh: float = 0.25,
    kpt_conf_thresh: float = 0.5,
    bboxes_xywh: np.ndarray | None = None,
) -> list[GateObservation]:
    """Turn raw YOLO-pose output arrays into GateObservations (the pure, model-free core).

    ``keypoints_xy`` is ``(n_det, 4, 2)`` in canonical corner order; ``keypoints_conf`` is
    ``(n_det, 4)``; ``det_scores`` is ``(n_det,)``; optional ``bboxes_xywh`` is ``(n_det, 4)``.
    A detection is kept only if its score clears ``score_thresh``; within it, keypoints clear
    ``kpt_conf_thresh`` to be used. 4 visible -> full (IPPE) observation; exactly 3 -> a
    subset observation with ``corner_ids`` (P3P); fewer than 3 -> dropped.
    """
    keypoints_xy = np.asarray(keypoints_xy, dtype=np.float64)
    keypoints_conf = np.asarray(keypoints_conf, dtype=np.float64)
    det_scores = np.asarray(det_scores, dtype=np.float64)
    if keypoints_xy.ndim != 3 or keypoints_xy.shape[1:] != (N_CORNERS, 2):
        raise ValueError(f"keypoints_xy must be (n_det,{N_CORNERS},2), got {keypoints_xy.shape}")

    out: list[GateObservation] = []
    for i in range(keypoints_xy.shape[0]):
        if det_scores[i] < score_thresh:
            continue
        kxy = keypoints_xy[i]
        kconf = keypoints_conf[i]
        visible = np.where(kconf >= kpt_conf_thresh)[0]
        if visible.size >= N_CORNERS:
            corners, corner_ids, conf = kxy.copy(), None, kconf.copy()
        elif visible.size == 3:
            corners, corner_ids, conf = kxy[visible].copy(), visible.astype(int), kconf[visible].copy()
        else:
            continue  # < 3 corners: pose is unrecoverable, drop the detection
        bbox = None if bboxes_xywh is None else np.asarray(bboxes_xywh[i], dtype=np.float64)
        out.append(
            GateObservation(
                frame_id=frame.frame_id,
                sim_time_ns=frame.sim_time_ns,
                corners_px=corners,
                corner_ids=corner_ids,
                corner_confidence=conf,
                score=float(det_scores[i]),
                bbox_xywh=bbox,
            )
        )
    return out


def _to_numpy(x):
    """Convert a torch tensor (or array-like) to numpy without importing torch."""
    if x is None:
        return None
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return np.asarray(x.numpy())
    return np.asarray(x)


def observations_from_results(
    frame: Frame,
    results,
    *,
    score_thresh: float = 0.25,
    kpt_conf_thresh: float = 0.5,
) -> list[GateObservation]:
    """Extract the arrays from one ultralytics ``Results`` (duck-typed) and adapt them."""
    boxes = getattr(results, "boxes", None)
    kpts = getattr(results, "keypoints", None)
    if boxes is None or kpts is None:
        return []
    xy = _to_numpy(getattr(kpts, "xy", None))
    if xy is None or xy.size == 0:
        return []
    conf = _to_numpy(getattr(kpts, "conf", None))
    if conf is None:
        conf = np.ones(xy.shape[:2])  # pose model without per-keypoint conf: treat all visible
    # 8-keypoint models emit 4 INNER corners (0..3) then 4 OUTER corners (4..7) -- see
    # blender_gen/contract.py N_KEYPOINTS scheme. PnP (gate_pose / task2_gate_pnp) is built on the
    # 4 INNER corners (the gate opening, gate_object_points(1.5)), so subset to the inner-4 here at
    # the model boundary. The pure 4-corner core + deployed inner-1.5 m PnP stay in lockstep; the
    # native 4-keypoint path is unchanged (this branch is a no-op when xy already has 4 keypoints).
    if xy.shape[1] == 8:
        xy = xy[:, :N_CORNERS, :]
        conf = conf[:, :N_CORNERS]
    scores = _to_numpy(getattr(boxes, "conf", None))
    if scores is None:
        scores = np.ones(xy.shape[0])
    bboxes = _to_numpy(getattr(boxes, "xywh", None))
    return observations_from_keypoints(
        frame, xy, conf, scores,
        score_thresh=score_thresh, kpt_conf_thresh=kpt_conf_thresh, bboxes_xywh=bboxes,
    )


# Sentinel attribute names for the shared per-frame_id detection cache (detect_cached). Namespaced
# with a leading underscore so they cannot collide with a detector's own fields.
_CACHE_FID_ATTR = "_detect_cache_frame_id"
_CACHE_OBS_ATTR = "_detect_cache_obs"
_CACHE_MS_ATTR  = "_detect_cache_last_ms"   # [DIAG] wall-time of the LAST real detect() (0.0 on a cache hit)


def detect_cached(detector, frame: Frame) -> list[GateObservation]:
    """Run ``detector.detect(frame)`` AT MOST ONCE per ``frame.frame_id``, caching the result on the
    detector instance so a SECOND consumer of the SAME shared detector reuses it instead of paying a
    second inference.

    A15/A17 DOUBLE-DETECT FIX (2026-07-01): the navigator (``_maybe_run_vision``) and the gate-seeker
    (``command_visual`` -> ``_valid_poses``) hold the SAME detector instance and each called
    ``detector.detect(frame)`` on every new frame -> YOLO ran TWICE per frame (~2x the ~150 ms hog).
    Routing both through this cache computes the detections ONCE and hands the identical list to both,
    ~halving detect cost/frame. The returned list is the SAME object the direct ``detect`` returned
    (no copy, no re-order, no threshold change), so the observations both consumers see are byte-
    identical to today's -- this is a transparent memoization, safe to leave always-on (VQ1/case-A
    unaffected: it only ever RETURNS what ``detect`` would have).

    Cache scope: keyed strictly on ``frame.frame_id``; a different frame_id (or a frame lacking one)
    recomputes. The cache lives on the detector object (two attrs), so distinct detector instances
    never share a cache. Defensive: any bookkeeping failure falls back to a plain ``detect``."""
    fid = getattr(frame, "frame_id", None)
    if fid is not None:
        try:
            if getattr(detector, _CACHE_FID_ATTR, object()) == fid:
                try: setattr(detector, _CACHE_MS_ATTR, 0.0)     # [DIAG] cache hit -> no inference paid here
                except Exception: pass
                return getattr(detector, _CACHE_OBS_ATTR)
        except Exception:
            pass
    _t0 = time.perf_counter()
    obs = detector.detect(frame)
    _ms = (time.perf_counter() - _t0) * 1e3                     # [DIAG] the REAL YOLO/TRT inference wall-time
    if fid is not None:
        try:
            setattr(detector, _CACHE_FID_ATTR, fid)
            setattr(detector, _CACHE_OBS_ATTR, obs)
            setattr(detector, _CACHE_MS_ATTR, _ms)
        except Exception:
            pass
    return obs


class GateDetector:
    """Runtime YOLO-pose gate detector. ``GateDetector.load(weights)`` loads a model (needs the
    ``[detector]`` extra: ``ultralytics``); ``detect(frame)`` returns GateObservations for one
    frame. The model is injectable, so ``detect`` is testable with a fake (no ultralytics)."""

    def __init__(self, model, *, score_thresh: float = 0.25, kpt_conf_thresh: float = 0.5,
                 device: str | None = None):
        self.model = model
        self.score_thresh = score_thresh
        self.kpt_conf_thresh = kpt_conf_thresh
        self.device = device

    @classmethod
    def load(cls, weights, **kwargs):
        # OPT-IN ensemble path (deploy gate, 2026-06-19): an "a++b" weights spec loads an
        # EnsembleGateDetector (union-of-detections + light per-gate dedup before the KF). A plain
        # single-model spec falls through to the UNCHANGED path below -- byte-identical to the
        # legacy behaviour, so the VQ1-proven single-model flight stack is untouched.
        if "++" in str(weights):
            return EnsembleGateDetector.load(str(weights).split("++"), **kwargs)
        from ultralytics import YOLO  # lazy: the heavy, GPU-only [detector] dependency

        # A15/A17 device fix: RESOLVE the device explicitly at load (GPU if available, else CPU) and
        # MOVE the model onto it, so inference does not silently run on the CPU (~7x slower). The
        # resolved device is stored on the instance and threaded into every predict(); it is LOGGED
        # once so the flight console / pre-warm shows GPU-vs-CPU (the #1 frame-starvation diagnostic).
        device = _resolve_device(kwargs.pop("device", None))
        # task=pose hint for exported graphs (ported from main detector._load_yolo_model): .engine /
        # .onnx carry NO pickled task, so YOLO() guesses 'detect' and the detect post-process SILENTLY
        # drops every keypoint -> 0 gate observations, no error. Pin task='pose' for those extensions;
        # a .pt spec takes the legacy YOLO(weights) call unchanged (byte-identical flight path).
        if str(weights).lower().endswith((".engine", ".onnx")):
            model = YOLO(str(weights), task="pose")
        else:
            model = YOLO(str(weights))
        _move_model_to_device(model, device)
        print(f"  [detector] GateDetector on device={device!r} (weights={str(weights)!r})")
        return cls(model, device=device, **kwargs)

    def detect(self, frame: Frame) -> list[GateObservation]:
        results = self.model.predict(frame.image_bgr, verbose=False, device=self.device)
        if not results:
            return []
        return observations_from_results(
            frame, results[0], score_thresh=self.score_thresh, kpt_conf_thresh=self.kpt_conf_thresh
        )


def _obs_centroid(o: GateObservation) -> np.ndarray:
    return np.asarray(o.corners_px, dtype=np.float64).mean(axis=0)


def _obs_span(o: GateObservation) -> float:
    c = np.asarray(o.corners_px, dtype=np.float64)
    return float(np.linalg.norm(c.max(axis=0) - c.min(axis=0)))


class EnsembleGateDetector:
    """OPT-IN deploy path (deploy gate, 2026-06-19): run 2+ YOLO-pose models, UNION their per-frame
    detections, then a light geometric dedup so the KF gets ONE update per gate (not a double-count
    when both members see the same gate). Interface-compatible with the navigator (only ``.detect``).
    The single-model :class:`GateDetector` path is untouched -- this is purely additive + opt-in.

    Why the dedup lives HERE (not in association): the offline ``eval_ensemble`` unioned with NO
    dedup because its metrics self-select the best obs per gate (course = ``any(obs<3m)``, good-fix =
    ``min di``). The deploy path feeds the union into the navigator -> KF, so two members both
    detecting gate-3 would apply TWO gate-3 fixes and the KF would double-count. Collapsing the union
    at the detector boundary, before association, means the navigator / association / PnP path is
    byte-unchanged -- the ensemble is a drop-in detector.

    Dedup (``_dedup``): cluster detections of the SAME gate (4-corner centroid within ``dedup_px``
    AND of comparable span), then FUSE each cluster to one observation (``_fuse``). ``dedup_px=0``
    recovers the pure union (the offline reference / an A/B). The default ``dedup_px=12`` is the
    deploy config (validated: union 6.5 -> 3.6 obs/frame, accuracy preserved)."""

    def __init__(self, models, *, score_thresh: float = 0.25, kpt_conf_thresh: float = 0.5,
                 device: str | None = None, dedup_px: float = 12.0):
        self.models = list(models)
        self.score_thresh = score_thresh
        self.kpt_conf_thresh = kpt_conf_thresh
        self.device = device
        self.dedup_px = float(dedup_px)
        self.model = self.models[0] if self.models else None  # compat shim if a caller reads .model

    @classmethod
    def load(cls, weights_list, **kwargs) -> "EnsembleGateDetector":
        from ultralytics import YOLO  # lazy: the heavy, GPU-only [detector] dependency

        # A15/A17 device fix (mirrors GateDetector.load): resolve the device explicitly + move EVERY
        # member onto it, so no ensemble member silently runs on the CPU. One log line names the device.
        device = _resolve_device(kwargs.pop("device", None))
        models = [YOLO(str(w).strip()) for w in weights_list if str(w).strip()]
        for m in models:
            _move_model_to_device(m, device)
        print(f"  [detector] EnsembleGateDetector ({len(models)} models) on device={device!r}")
        return cls(models, device=device, **kwargs)

    def detect(self, frame: Frame) -> list[GateObservation]:
        obs: list[GateObservation] = []
        for m in self.models:
            results = m.predict(frame.image_bgr, verbose=False, device=self.device)
            if results:
                obs.extend(observations_from_results(
                    frame, results[0], score_thresh=self.score_thresh,
                    kpt_conf_thresh=self.kpt_conf_thresh))
        return self._dedup(obs)

    def _dedup(self, obs: list[GateObservation]) -> list[GateObservation]:
        """Cluster observations of the SAME gate (centroid within dedup_px + comparable span) and FUSE
        each cluster into ONE observation, so the KF gets a single, noise-reduced update per gate
        instead of N double-counts. ``dedup_px<=0`` (or <=1 obs) returns the union unchanged.

        The span term is the safety against the receding-collinear course: a near (big) and a far
        (small) gate can project to NEARBY centroids, so a centroid-only merge would wrongly collapse
        two distinct gates; requiring comparable apparent span keeps them separate (the same
        overfit-GEOMETRY spirit as ``association`` -- a near detection cannot match a far gate's
        shape). Greedy by detection score: the first (highest-score) member seeds each cluster."""
        if self.dedup_px <= 0 or len(obs) <= 1:
            return obs
        clusters: list[list[GateObservation]] = []
        for o in sorted(obs, key=lambda x: float(x.score), reverse=True):
            co, so = _obs_centroid(o), _obs_span(o)
            for cl in clusters:
                if (float(np.linalg.norm(co - _obs_centroid(cl[0]))) < self.dedup_px
                        and abs(so - _obs_span(cl[0])) < max(self.dedup_px, 0.25 * max(so, _obs_span(cl[0])))):
                    cl.append(o)
                    break
            else:
                clusters.append([o])
        return [self._fuse(cl) for cl in clusters]

    @staticmethod
    def _fuse(cluster: list[GateObservation]) -> GateObservation:
        """Score-weighted corner average of a same-gate cluster. Only the full-4-corner members are
        averaged (the IPPE PnP set); if <2 of those, fall back to the single top-score observation.
        Fusing (vs keep-highest-score) gives the KF a noise-reduced corner set -- a member whose PnP
        is better than its box score implies still contributes (offline A/B favoured fuse)."""
        if len(cluster) == 1:
            return cluster[0]
        full = [o for o in cluster
                if o.corner_ids is None and np.asarray(o.corners_px).shape[0] == N_CORNERS]
        if len(full) < 2:
            return max(cluster, key=lambda o: float(o.score))
        w = np.array([float(o.score) for o in full], dtype=np.float64)
        w = w / w.sum() if w.sum() > 0 else np.full(len(full), 1.0 / len(full))
        corners = sum(wi * np.asarray(o.corners_px, dtype=np.float64) for wi, o in zip(w, full))
        conf = np.max([np.asarray(o.corner_confidence, dtype=np.float64) for o in full], axis=0)
        top = max(full, key=lambda o: float(o.score))
        return replace(top, corners_px=corners, corner_confidence=conf)
