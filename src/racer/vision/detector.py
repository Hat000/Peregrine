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

# Per-keypoint confidence needed to USE a corner. Lowered 0.5 -> 0.2 (2026-07-21) after calibrating
# confidence against actual localisation error rather than assuming it: outer-keypoint confidence is
# nearly UNINFORMATIVE between 0.2 and 0.9 (conf 0.2-0.4 -> 1.7-2.0 px median error, no worse than
# the 0.5-0.7 bin at 1.99 px); only below ~0.10 does it degrade (4.4-6.3 px). So 0.5 was discarding
# well-localised corners. Validated on the task2 GT bundle: emissions 2428 -> 2548 while the centre
# error p90 IMPROVED 0.666 -> 0.520 m and good-fix went 34 -> 35/40. CAVEAT: the calibration has no
# data for INNER keypoints below 0.5 (its reference — 4 confident outer corners — only exists on
# fully-visible gates), so this rests on task2's 40 frames, not on the curve. Callers that pin a
# value explicitly (characterize_perception, task2_gate_pnp) keep 0.5 so historical benchmarks stay
# comparable.
KPT_CONF_THRESH_DEFAULT = 0.2

# Derived-corner confidence for the outer->inner rescue: BELOW gate_pose's CONF_FLOOR (0.1) on
# purpose, so the derived inners only SEED the IPPE init while the joint refinement stays anchored
# on the 4 MEASURED outer corners (the derived points are linear functions of those same outers —
# weighting them fully would double-count the measurement).
_RESCUE_INNER_CONF = 0.10

# Derived-corner confidence for the PARTIAL rescue. Calibrated, not guessed: on 1131 real clean
# detections replayed through the 106 visibility masks that actually occur when a gate crops out of
# frame, the recovered centre lands 0.060 m from truth (p90 0.227 m) against 0.008-0.022 m for a
# full 4-corner fix — a ~3x sigma penalty. gate_pose._refine_pose whitens corner i by
# sigma_i = WEIGHTED_SIGMA_PX / conf_i, so conf ~= 1/3 reproduces that 3x directly and the emitted
# covariance comes out approximately calibrated instead of over-confident. Rounded DOWN (more
# conservative) because the closest band (<3 m) has no ablation coverage — a gate that near is
# always cropped, so it never supplies a full-8 reference frame to measure against.
_PARTIAL_RESCUE_CONF = 0.25

# A keypoint sitting on the image border is CLAMPED, not localised: ultralytics' scale_coords ends
# in an unconditional clip_coords(), so anything the net placed off-frame comes back pinned to the
# edge. Measured against homography truth, those pinned points carry 12.3 px median error (vs
# 0.73 px in-frame) — they must never enter a fit. [2026-07-21]
_BORDER_EPS_PX = 1.0


def _unclamped_mask(keypoints_px: np.ndarray, image_wh: tuple[int, int]) -> np.ndarray:
    """Boolean mask of keypoints strictly inside the image (i.e. not border-clamped)."""
    w, h = image_wh
    x, y = keypoints_px[:, 0], keypoints_px[:, 1]
    return ((x > _BORDER_EPS_PX) & (x < w - _BORDER_EPS_PX)
            & (y > _BORDER_EPS_PX) & (y < h - _BORDER_EPS_PX))


def _derive_inner_from_outer(outer_px: np.ndarray) -> np.ndarray | None:
    """Inner corners from the 4 OUTER corners via the exact gate-plane homography (concentric
    coplanar squares, spec 3.7) — the single implementation lives in ``gate_pose``; imported lazily
    so this module's pure-numpy core stays importable without opencv."""
    from racer.vision.gate_pose import inner_from_outer_homography

    return inner_from_outer_homography(outer_px)


def _derive_inner_from_partial(keypoints_px: np.ndarray, usable: np.ndarray, img_w: int):
    """Inner corners from ANY >=4 usable keypoints (see gate_pose.inner_from_partial_keypoints)."""
    from racer.vision.gate_pose import inner_from_partial_keypoints

    return inner_from_partial_keypoints(keypoints_px, usable, img_w=img_w)


def _load_yolo_model(weights):
    """ultralytics ``YOLO`` loader with the exported-graph task hint (TRT engine pipeline).
    ``.engine`` (TensorRT) / ``.onnx`` files carry no pickled task, so ``YOLO()`` falls back to
    ``guess_model_task() -> "detect"`` and the detect post-process silently drops every keypoint
    (zero gate observations, no error). Pin ``task="pose"`` for those extensions; a ``.pt`` spec
    takes exactly the legacy ``YOLO(weights)`` call so the proven flight path is byte-identical."""
    from ultralytics import YOLO  # lazy: the heavy, GPU-only [detector] dependency

    if str(weights).lower().endswith((".engine", ".onnx")):
        return YOLO(str(weights), task="pose")
    return YOLO(str(weights))


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
    kpt_conf_thresh: float = KPT_CONF_THRESH_DEFAULT,
    bboxes_xywh: np.ndarray | None = None,
    outer_xy: np.ndarray | None = None,
    outer_conf: np.ndarray | None = None,
    partial_rescue: bool = True,
    image_wh: tuple[int, int] | None = None,
) -> list[GateObservation]:
    """Turn raw YOLO-pose output arrays into GateObservations (the pure, model-free core).

    ``keypoints_xy`` is ``(n_det, 4, 2)`` in canonical corner order; ``keypoints_conf`` is
    ``(n_det, 4)``; ``det_scores`` is ``(n_det,)``; optional ``bboxes_xywh`` is ``(n_det, 4)``.
    A detection is kept only if its score clears ``score_thresh``; within it, keypoints clear
    ``kpt_conf_thresh`` to be used. 4 visible -> full (IPPE) observation; exactly 3 -> a
    subset observation with ``corner_ids`` (P3P); fewer than 3 -> dropped.

    OUTER corners (8-keypoint models; 2026-07-05): optional ``outer_xy`` (n_det,4,2) +
    ``outer_conf`` (n_det,4) ride along on every emitted observation (gate_pose fuses them into
    the pose). They also enable the OUTER RESCUE: a detection whose inner corners are washed out
    (<3 clearing the threshold — e.g. glare across the opening) but whose 4 outer corners are ALL
    confident is no longer dropped; the inner corners are derived through the exact gate-plane
    homography (concentric coplanar squares) and the observation is emitted with LOW derived-inner
    confidence (see _RESCUE_INNER_CONF) so the pose leans on the measured outer corners.

    PARTIAL RESCUE (``partial_rescue``, needs ``image_wh``; 2026-07-21): the outer rescue above
    demands all 4 outer corners, which on a CROPPED gate are the least likely to be visible — the
    outer square is 1.81x larger, so it leaves frame first. Measured over 998 real failure-mined
    frames it fired 0/251 times: a structurally dead path. The partial rescue generalises it — ANY
    4 usable keypoints (inner and outer are the same plane) determine the homography, so the common
    "gate half out of frame, two corners plus their outer corners visible" case is recovered instead
    of dropped. It fires only where the pipeline previously gave up, so every pre-existing emission
    is bit-identical. Measured recovery: 106 of 428 dropped detections (24.8%), centre error 0.060 m
    median / 0.227 m p90. Keypoints pinned to the image border are excluded (see _BORDER_EPS_PX).
    """
    keypoints_xy = np.asarray(keypoints_xy, dtype=np.float64)
    keypoints_conf = np.asarray(keypoints_conf, dtype=np.float64)
    det_scores = np.asarray(det_scores, dtype=np.float64)
    if keypoints_xy.ndim != 3 or keypoints_xy.shape[1:] != (N_CORNERS, 2):
        raise ValueError(f"keypoints_xy must be (n_det,{N_CORNERS},2), got {keypoints_xy.shape}")
    if outer_xy is not None:
        outer_xy = np.asarray(outer_xy, dtype=np.float64)
        outer_conf = (np.ones(outer_xy.shape[:2]) if outer_conf is None
                      else np.asarray(outer_conf, dtype=np.float64))

    out: list[GateObservation] = []
    for i in range(keypoints_xy.shape[0]):
        if det_scores[i] < score_thresh:
            continue
        kxy = keypoints_xy[i]
        kconf = keypoints_conf[i]
        o_xy = None if outer_xy is None else outer_xy[i].copy()
        o_conf = None if outer_xy is None else outer_conf[i].copy()
        derived_corners = False
        visible = np.where(kconf >= kpt_conf_thresh)[0]
        if visible.size >= N_CORNERS:
            corners, corner_ids, conf = kxy.copy(), None, kconf.copy()
        elif visible.size == 3:
            corners, corner_ids, conf = kxy[visible].copy(), visible.astype(int), kconf[visible].copy()
        elif (o_xy is not None and o_conf is not None
              and bool((o_conf >= kpt_conf_thresh).all())):
            # OUTER RESCUE: derive the washed-out inner corners from the 4 confident outer corners.
            derived = _derive_inner_from_outer(o_xy)
            if derived is None:
                continue
            corners, corner_ids = derived, None
            conf = np.full(N_CORNERS, _RESCUE_INNER_CONF)
            derived_corners = True
        elif partial_rescue and o_xy is not None and o_conf is not None and image_wh is not None:
            # PARTIAL RESCUE: a cropped gate showing e.g. 2 inner corners AND their 2 outer corners
            # still pins the whole gate — inner and outer are the SAME plane, so any 4 keypoints in
            # general position determine the plane-to-image homography and hence the (partly
            # off-frame) inner square. This fires exactly where the pipeline used to give up.
            kp8 = np.vstack([kxy, o_xy])
            usable = (np.concatenate([kconf, o_conf]) >= kpt_conf_thresh) & _unclamped_mask(kp8, image_wh)
            if int(usable.sum()) < N_CORNERS:
                continue
            derived = _derive_inner_from_partial(kp8, usable, img_w=image_wh[0])
            if derived is None:
                continue
            corners, corner_ids = derived, None
            conf = np.full(N_CORNERS, _PARTIAL_RESCUE_CONF)
            derived_corners = True
            o_xy = o_conf = None   # the outer measurements are already baked into ``derived``;
            #                        re-offering them to the pose fit would double-count them
        else:
            continue  # < 3 usable corners and no recoverable geometry: pose is unrecoverable
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
                outer_corners_px=o_xy,
                outer_corner_confidence=o_conf,
                derived_corners=derived_corners,
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
    kpt_conf_thresh: float = KPT_CONF_THRESH_DEFAULT,
    use_outer: bool = True,
    partial_rescue: bool = True,
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
    # blender_gen/contract.py N_KEYPOINTS scheme. PnP is anchored on the 4 INNER corners (the gate
    # opening, gate_object_points(1.5)), so corners_px stays the inner-4 here at the model
    # boundary — but the OUTER 4 now RIDE ALONG (outer_corners_px) and gate_pose fuses them into
    # the same pose fit (use_outer=False restores the discard, the pre-2026-07-05 behaviour).
    # The native 4-keypoint path is unchanged (this branch is a no-op when xy has 4 keypoints).
    outer_xy = outer_conf = None
    if xy.shape[1] == 8:
        if use_outer:
            outer_xy = xy[:, N_CORNERS:, :]
            outer_conf = conf[:, N_CORNERS:]
        xy = xy[:, :N_CORNERS, :]
        conf = conf[:, :N_CORNERS]
    scores = _to_numpy(getattr(boxes, "conf", None))
    if scores is None:
        scores = np.ones(xy.shape[0])
    bboxes = _to_numpy(getattr(boxes, "xywh", None))
    img = getattr(frame, "image_bgr", None)
    image_wh = None if img is None else (int(img.shape[1]), int(img.shape[0]))
    return observations_from_keypoints(
        frame, xy, conf, scores,
        score_thresh=score_thresh, kpt_conf_thresh=kpt_conf_thresh, bboxes_xywh=bboxes,
        outer_xy=outer_xy, outer_conf=outer_conf,
        partial_rescue=partial_rescue, image_wh=image_wh,
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

    def __init__(self, model, *, score_thresh: float = 0.25, kpt_conf_thresh: float = KPT_CONF_THRESH_DEFAULT,
                 device: str | None = None, use_outer: bool = True, partial_rescue: bool = True):
        self.model = model
        self.score_thresh = score_thresh
        self.kpt_conf_thresh = kpt_conf_thresh
        self.device = device
        # Outer-corner fusion kill-switch (2026-07-05): False restores the exact pre-fusion
        # behaviour (outer keypoints of an 8-kpt model discarded at the model boundary).
        self.use_outer = use_outer
        # Partial-corner rescue kill-switch (2026-07-21): False restores the exact pre-rescue
        # behaviour (a detection with <3 confident inner corners is dropped).
        self.partial_rescue = partial_rescue

    @classmethod
    def load(cls, weights, **kwargs):
        # OPT-IN ensemble path (deploy gate, 2026-06-19): an "a++b" weights spec loads an
        # EnsembleGateDetector (union-of-detections + light per-gate dedup before the KF). A plain
        # single-model spec falls through to the UNCHANGED path below -- byte-identical to the
        # legacy behaviour, so the VQ1-proven single-model flight stack is untouched.
        if "++" in str(weights):
            return EnsembleGateDetector.load(str(weights).split("++"), **kwargs)
        # A15/A17 device fix: RESOLVE the device explicitly at load (GPU if available, else CPU) and
        # MOVE the model onto it, so inference does not silently run on the CPU (~7x slower). The
        # resolved device is stored on the instance and threaded into every predict(); it is LOGGED
        # once so the flight console / pre-warm shows GPU-vs-CPU (the #1 frame-starvation diagnostic).
        # The task='pose' hint for exported .engine/.onnx graphs lives in _load_yolo_model.
        device = _resolve_device(kwargs.pop("device", None))
        model = _load_yolo_model(weights)
        _move_model_to_device(model, device)
        print(f"  [detector] GateDetector on device={device!r} (weights={str(weights)!r})")
        return cls(model, device=device, **kwargs)

    def detect(self, frame: Frame) -> list[GateObservation]:
        results = self.model.predict(frame.image_bgr, verbose=False, device=self.device)
        if not results:
            return []
        return observations_from_results(
            frame, results[0], score_thresh=self.score_thresh, kpt_conf_thresh=self.kpt_conf_thresh,
            use_outer=self.use_outer, partial_rescue=self.partial_rescue,
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

    def __init__(self, models, *, score_thresh: float = 0.25, kpt_conf_thresh: float = KPT_CONF_THRESH_DEFAULT,
                 device: str | None = None, dedup_px: float = 12.0, use_outer: bool = True,
                 partial_rescue: bool = True):
        self.models = list(models)
        self.score_thresh = score_thresh
        self.kpt_conf_thresh = kpt_conf_thresh
        self.device = device
        self.dedup_px = float(dedup_px)
        self.use_outer = use_outer
        self.partial_rescue = partial_rescue
        self.model = self.models[0] if self.models else None  # compat shim if a caller reads .model

    @classmethod
    def load(cls, weights_list, **kwargs) -> "EnsembleGateDetector":
        # A15/A17 device fix (mirrors GateDetector.load): resolve the device explicitly + move EVERY
        # member onto it, so no ensemble member silently runs on the CPU. One log line names the device.
        device = _resolve_device(kwargs.pop("device", None))
        models = [_load_yolo_model(str(w).strip()) for w in weights_list if str(w).strip()]
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
                    kpt_conf_thresh=self.kpt_conf_thresh, use_outer=self.use_outer,
                    partial_rescue=self.partial_rescue))
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
        # OUTER corners fuse the same way (score-weighted mean over the members that carry them,
        # conf = element-max) so the fused observation stays internally consistent; members without
        # outer (4-kpt models in a mixed ensemble) simply don't contribute.
        wo = [(wi, o) for wi, o in zip(w, full) if o.outer_corners_px is not None]
        outer = outer_conf = None
        if wo:
            ws = sum(wi for wi, _ in wo)
            outer = sum(wi * np.asarray(o.outer_corners_px, dtype=np.float64) for wi, o in wo) / ws
            outer_conf = np.max([np.asarray(o.outer_corner_confidence, dtype=np.float64)
                                 if o.outer_corner_confidence is not None else np.ones(4)
                                 for _, o in wo], axis=0)
        return replace(top, corners_px=corners, corner_confidence=conf,
                       outer_corners_px=outer, outer_corner_confidence=outer_conf)
