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

Wiring: this is the SENSE step behind ``Mission.run``'s navigator —
``frame -> detect() -> [GateObservation] -> estimate_gate_pose -> localization -> KF``.
"""
from __future__ import annotations

import numpy as np

from racer.contracts import Frame, GateObservation

N_CORNERS = 4


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
    scores = _to_numpy(getattr(boxes, "conf", None))
    if scores is None:
        scores = np.ones(xy.shape[0])
    bboxes = _to_numpy(getattr(boxes, "xywh", None))
    return observations_from_keypoints(
        frame, xy, conf, scores,
        score_thresh=score_thresh, kpt_conf_thresh=kpt_conf_thresh, bboxes_xywh=bboxes,
    )


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
    def load(cls, weights, **kwargs) -> "GateDetector":
        from ultralytics import YOLO  # lazy: the heavy, GPU-only [detector] dependency

        return cls(YOLO(str(weights)), **kwargs)

    def detect(self, frame: Frame) -> list[GateObservation]:
        results = self.model.predict(frame.image_bgr, verbose=False, device=self.device)
        if not results:
            return []
        return observations_from_results(
            frame, results[0], score_thresh=self.score_thresh, kpt_conf_thresh=self.kpt_conf_thresh
        )
