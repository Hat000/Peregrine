"""Reproduce the camera matrix K exactly with a Blender camera (and verify it).

Blender parameterises a perspective camera by ``lens`` (focal mm), ``sensor_width`` /
``sensor_height`` (mm), ``sensor_fit``, and ``shift_x`` / ``shift_y`` (principal-point
offset, normalised by the fit dimension). This module converts our pixel-space K into those
parameters and BACK, so a unit test can assert the round-trip reproduces K to floating point
WITHOUT a live Blender. The ShadowPC render-time check (project known 3D points through the
real Blender camera vs through K, assert <= 1 px) is the live arbiter; this is the math gate.

Our K (racer.frames): fx = fy = 320, cx = 320 = W/2, cy = 180 = H/2 at 640x360 -> the
principal point is EXACTLY centred, so shift_x = shift_y = 0. fx == fy with W/H = 16/9 means
square pixels (pixel_aspect 1:1). With sensor_fit = HORIZONTAL the vertical focal is derived
from the resolution at square pixels, so it equals fx automatically -> VFoV ~ 58.7 deg falls
out, never set by hand.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contract import (
    BLENDER_SENSOR_FIT,
    BLENDER_SENSOR_WIDTH_MM,
    CAMERA_INTRINSICS_K,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
)


@dataclass(frozen=True)
class BlenderCameraParams:
    """The Blender camera settings that reproduce K. Set these on ``camera.data`` and render
    at ``(render_width, render_height)`` with ``pixel_aspect_x == pixel_aspect_y``."""

    lens_mm: float
    sensor_width_mm: float
    sensor_height_mm: float
    sensor_fit: str          # "HORIZONTAL"
    shift_x: float
    shift_y: float
    render_width: int
    render_height: int


def blender_camera_params(
    K: np.ndarray = CAMERA_INTRINSICS_K,
    width: int = IMAGE_WIDTH,
    height: int = IMAGE_HEIGHT,
    sensor_width_mm: float = BLENDER_SENSOR_WIDTH_MM,
    *,
    principal_tol_px: float = 1e-6,
) -> BlenderCameraParams:
    """Convert K -> Blender camera params (HORIZONTAL sensor fit, square pixels).

    Raises if fx != fy (non-square pixels need pixel_aspect tuning, out of scope) or if the
    principal point is not centred (would need a non-zero shift -- formula in the comments).
    """
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    if abs(fx - fy) > 1e-6:
        raise ValueError(f"non-square pixels (fx={fx} != fy={fy}); set pixel_aspect, unsupported here")
    dx, dy = cx - width / 2.0, cy - height / 2.0
    if abs(dx) > principal_tol_px or abs(dy) > principal_tol_px:
        # Off-centre principal point. Blender shift is normalised by the FIT dimension (W for
        # HORIZONTAL). The convention (verified against the round-trip below):
        #   shift_x = -(cx - W/2) / W ;  shift_y = (cy - H/2) / W
        raise ValueError(
            f"principal point not centred (cx-W/2={dx:.3g}, cy-H/2={dy:.3g}); our K is centred -- "
            "if this changes, use shift_x=-(cx-W/2)/W, shift_y=(cy-H/2)/W and drop this guard"
        )
    if BLENDER_SENSOR_FIT != "HORIZONTAL":
        raise ValueError("this converter assumes HORIZONTAL sensor_fit")
    # HORIZONTAL fit: fx_px = lens / sensor_width * W  ->  lens = fx * sensor_width / W.
    lens_mm = fx * sensor_width_mm / width
    # sensor_height is unused by HORIZONTAL fit, but set it to the matching value (square
    # pixels: mm/px equal on both axes) so the data-block reads sensibly.
    sensor_height_mm = sensor_width_mm * height / width
    return BlenderCameraParams(
        lens_mm=lens_mm,
        sensor_width_mm=sensor_width_mm,
        sensor_height_mm=sensor_height_mm,
        sensor_fit="HORIZONTAL",
        shift_x=0.0,
        shift_y=0.0,
        render_width=int(width),
        render_height=int(height),
    )


def K_from_blender_params(p: BlenderCameraParams) -> np.ndarray:
    """Reconstruct the pixel-space K a Blender render with params ``p`` produces (square
    pixels). The inverse of :func:`blender_camera_params`; the test asserts it equals K."""
    if p.sensor_fit != "HORIZONTAL":
        raise ValueError("only HORIZONTAL fit modelled")
    px_per_mm = p.render_width / p.sensor_width_mm
    fx = p.lens_mm * px_per_mm
    fy = fx  # square pixels (pixel_aspect 1:1); HORIZONTAL fit derives vertical from resolution
    cx = p.render_width / 2.0 - p.shift_x * p.render_width
    cy = p.render_height / 2.0 + p.shift_y * p.render_width
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def reprojection_max_error_px(
    K: np.ndarray = CAMERA_INTRINSICS_K,
    width: int = IMAGE_WIDTH,
    height: int = IMAGE_HEIGHT,
) -> float:
    """Max abs pixel disagreement between K and the K reconstructed from the Blender params.
    The pure-math analogue of the ShadowPC 'project 3D points through Blender vs K' check."""
    p = blender_camera_params(K, width, height)
    return float(np.max(np.abs(K_from_blender_params(p) - K)))


def horizontal_fov_rad(p: BlenderCameraParams) -> float:
    return float(2.0 * np.arctan(p.sensor_width_mm / (2.0 * p.lens_mm)))
