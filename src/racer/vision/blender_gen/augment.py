"""Keypoint-AWARE albumentations post-pipeline for the photoreal dataset.

Applied AFTER the (Blender or procedural) render, on the BGR image + the gate corners, so the
sensor/lighting/lens degradations the real camera adds at racing speed are layered on without
re-rendering. Geometric augs are run through albumentations ``KeypointParams`` so the inner +
outer gate corners move WITH the image; the bbox + per-corner visibility are then recomputed
from the warped corners. Photometric/sensor augs leave the corners untouched.

NO flips / reflections / 90-deg rotations here: a horizontal flip mirrors the gate, which is a
REFLECTION (not a proper camera pose) and would silently break the keypoint->gate-frame-corner
identity that PnP relies on. Left/right flip augmentation is left to ultralytics' train-time
``fliplr`` + ``flip_idx`` (which relabels corners correctly). We only apply orientation-
preserving warps (small translate / scale / rotate / shear / perspective).

Falls back to an OpenCV photometric pipeline (reusing ``synthetic._chaos_cv2``) if
albumentations is not installed -- in that fallback geometric augs are skipped (keypoints are
never moved), so labels stay correct either way.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contract import IMAGE_HEIGHT, IMAGE_WIDTH, V_OCC, V_OFF, V_VIS
from .geometry import GateRender, MIN_LABEL_AREA_PX


@dataclass(frozen=True)
class AugmentConfig:
    """Post-render augmentation intensity (driven by the scenario preset)."""

    enable: bool = True
    photometric_p: float = 0.9     # ONE primary sensor/motion degradation
    lighting_p: float = 0.5        # at most one shadow/flare/fog/dropout
    brightness_p: float = 0.5
    geometric_p: float = 0.0       # mild orientation-preserving warp (keypoint-aware)
    geometric_translate_frac: float = 0.04
    geometric_scale: float = 0.08          # +/- fraction
    geometric_rotate_deg: float = 8.0      # small: keeps corner identity, no reflection
    geometric_shear_deg: float = 4.0
    perspective_scale: float = 0.04
    motion_blur_limit: int = 15            # racing motion blur kernel (px)
    jpeg_quality_min: int = 25             # the VQ stream IS jpeg
    jpeg_quality_max: int = 70


_PIPELINE_CACHE: dict = {}


def _build_pipeline(cfg: AugmentConfig):
    """Build (once per cfg) the albumentations Compose with keypoint params, or None if absent."""
    key = (cfg.photometric_p, cfg.lighting_p, cfg.brightness_p, cfg.geometric_p,
           cfg.geometric_translate_frac, cfg.geometric_scale, cfg.geometric_rotate_deg,
           cfg.geometric_shear_deg, cfg.perspective_scale, cfg.motion_blur_limit,
           cfg.jpeg_quality_min, cfg.jpeg_quality_max)
    if key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[key] or None
    try:
        import albumentations as A
    except Exception:
        _PIPELINE_CACHE[key] = False
        return None

    transforms = [
        A.OneOf([                                                        # one primary sensor/motion effect
            A.MotionBlur(blur_limit=(3, max(3, cfg.motion_blur_limit))),
            A.ImageCompression(quality_range=(cfg.jpeg_quality_min, cfg.jpeg_quality_max)),
            A.Downscale(scale_range=(0.4, 0.85)),
            A.GaussNoise(std_range=(0.05, 0.2)),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),
            A.Defocus(radius=(3, 7)),
        ], p=cfg.photometric_p),
        A.OneOf([                                                        # at most one lighting/occlusion effect
            A.RandomShadow(),
            A.RandomSunFlare(src_radius=80),
            A.RandomFog(fog_coef_range=(0.1, 0.4)),
            A.CoarseDropout(num_holes_range=(1, 4),
                            hole_height_range=(0.04, 0.12), hole_width_range=(0.04, 0.12)),
        ], p=cfg.lighting_p),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=cfg.brightness_p),
    ]
    if cfg.geometric_p > 0.0:
        # Orientation-preserving ONLY (no flip/reflection): keypoints transform with the image.
        transforms.insert(0, A.OneOf([
            A.Affine(
                translate_percent=(-cfg.geometric_translate_frac, cfg.geometric_translate_frac),
                scale=(1.0 - cfg.geometric_scale, 1.0 + cfg.geometric_scale),
                rotate=(-cfg.geometric_rotate_deg, cfg.geometric_rotate_deg),
                shear=(-cfg.geometric_shear_deg, cfg.geometric_shear_deg),
                fit_output=False, keep_ratio=False,
            ),
            A.Perspective(scale=(0.01, cfg.perspective_scale), fit_output=False),
        ], p=cfg.geometric_p))

    compose = A.Compose(
        transforms,
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )
    _PIPELINE_CACHE[key] = compose
    return compose


def _recompute(gr: GateRender, inner: np.ndarray, outer: np.ndarray,
               partial: tuple[int, float] | None = None) -> GateRender:
    """Rebuild a GateRender from warped inner/outer corners: new bbox + visibility + visible.

    ``partial`` is accepted for signature compatibility but no longer gates visibility: a label now
    requires only enough VISIBLE AREA (see the note below and geometry._has_labellable_area)."""
    vis = gr.visibility.copy()
    for c in range(4):
        x, y = float(inner[c, 0]), float(inner[c, 1])
        if not (0.0 <= x <= IMAGE_WIDTH - 1 and 0.0 <= y <= IMAGE_HEIGHT - 1):
            vis[c] = V_OFF
        elif vis[c] == V_OFF:                                            # warped back in-frame
            vis[c] = V_VIS
    # outer keypoints (4..7): in-frame V_VIS / V_OFF on the warped outer corners
    vis_outer = (gr.outer_visibility.copy() if gr.outer_visibility is not None
                 else np.full(4, V_VIS, dtype=int))
    for c in range(4):
        x, y = float(outer[c, 0]), float(outer[c, 1])
        vis_outer[c] = V_VIS if (0.0 <= x <= IMAGE_WIDTH - 1 and 0.0 <= y <= IMAGE_HEIGHT - 1) else V_OFF
    x0, y0 = outer.min(axis=0)
    x1, y1 = outer.max(axis=0)
    x0, y0 = max(0.0, float(x0)), max(0.0, float(y0))
    x1, y1 = min(float(IMAGE_WIDTH), float(x1)), min(float(IMAGE_HEIGHT), float(y1))
    bbox = np.array([x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)])
    # VISIBLE AREA, not corner count -- the same rule geometry._has_labellable_area applies before
    # augmentation. This function re-derived visibility AFTER the warp using the old ">=3 in-frame
    # corners" / ">=3 + centre in frame" tests, which silently UNDID the geometry-side fix: the
    # sampler labelled 2.58 gates per frame and the written dataset still came out at 1.00.
    # ``partial``'s min_area_frac stays where it belongs, as the crop SAMPLER's acceptance test in
    # geometry.sample_partial_frame; applying it here would drop every small co-visible gate.
    visible = bool(bbox[2] * bbox[3] >= MIN_LABEL_AREA_PX)
    return GateRender(
        gate_id=gr.gate_id, R_cam_gate=gr.R_cam_gate, t_cam_gate=gr.t_cam_gate,
        keypoints_px=np.asarray(inner, dtype=float), outer_px=np.asarray(outer, dtype=float),
        bbox_xywh=bbox, visibility=vis, visible=visible, outer_visibility=vis_outer,
    )


def augment_frame(
    image_bgr: np.ndarray, gates: list[GateRender], cfg: AugmentConfig,
    rng: np.random.Generator | None = None, *, partial: tuple[int, float] | None = None,
) -> tuple[np.ndarray, list[GateRender]]:
    """Apply the keypoint-aware pipeline. Returns (augmented image, updated GateRenders).

    Only the labelled gates' corners are tracked; non-labelled gates pass through unchanged.
    ``partial`` routes the crop arm's relaxed positive rule through :func:`_recompute` (None =
    the frozen full-gate rule).
    """
    if not cfg.enable:
        return image_bgr, gates
    labeled = [g for g in gates if g.visible]
    others = [g for g in gates if not g.visible]
    pipeline = _build_pipeline(cfg)
    if pipeline is None:                                                 # OpenCV photometric fallback
        from racer.vision.synthetic import _chaos_cv2
        gen = np.random.default_rng() if rng is None else rng
        return _chaos_cv2(image_bgr, gen), gates

    # 8 keypoints per labelled gate: inner 0..3 then outer 0..3.
    kps: list = []
    for g in labeled:
        kps.extend([tuple(map(float, p)) for p in g.keypoints_px])
        kps.extend([tuple(map(float, p)) for p in g.outer_px])
    out = pipeline(image=image_bgr, keypoints=kps)
    img = out["image"]
    warped = np.asarray(out["keypoints"], dtype=float).reshape(-1, 2) if out["keypoints"] else np.zeros((0, 2))

    new_labeled: list[GateRender] = []
    for i, g in enumerate(labeled):
        block = warped[8 * i: 8 * i + 8]
        new_labeled.append(_recompute(g, block[0:4], block[4:8], partial=partial))
    return img, new_labeled + others
