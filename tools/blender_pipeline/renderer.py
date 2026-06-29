"""Renderer interface + a Blender-FREE mock renderer.

The harness is renderer-agnostic: it talks to an abstract :class:`Renderer` that maps a
``(camera_pose, scene_params)`` to an ``H x W x 3`` BGR image. The real Blender renderer will
implement this same interface and slot in unchanged; here we ship a :class:`MockRenderer` that
draws an emissive-red gate frame (with approximate bloom) at the projected corners over a dark
low-light background per the appearance params. That lets the whole pipeline -- sample -> project
-> render -> detect -> label -- run + be tested with NO Blender.

The mock's job is NOT photorealism. It is to be GEOMETRICALLY faithful (the red core ring lands
exactly on the projected corners) and APPEARANCE-plausible enough that the classical
``red_glow_detector`` recovers those same corners. That round-trip is the harness's correctness
proof; the Blender renderer adds the real look on top of the same geometry contract.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

try:
    import cv2
except Exception as exc:  # pragma: no cover
    cv2 = None
    _CV2_ERR = exc

from racer.frames import IMAGE_HEIGHT, IMAGE_WIDTH
from racer.vision.gate_pose import GATE_INNER_SIZE_M, project_gate_corners

from .appearance import AppearanceParams, load_appearance
from .camera_sampler import CameraPose


class Renderer(Protocol):
    """Render one frame: a camera pose (gate-in-camera optical) + scene params -> BGR image.

    The real Blender renderer implements this Protocol. ``render`` returns an
    ``(H, W, 3) uint8`` BGR image (OpenCV order), the same array a live ``Frame.image_bgr``
    carries, so the detector / dataset writer are renderer-agnostic.
    """

    def render(self, pose: CameraPose, params: AppearanceParams) -> np.ndarray:
        ...


def _rgb_to_bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    return (int(rgb[2]), int(rgb[1]), int(rgb[0]))


class MockRenderer:
    """Blender-free mock: draws an emissive red gate ring at the projected corners + bloom.

    Geometry: the gate's INNER corners are projected with the canonical
    ``project_gate_corners`` (the same call the labels use), so the rendered core ring's inner
    edge lands EXACTLY on the label corners. The red ring is drawn between that inner quad and an
    OUTER quad (the inner quad scaled out about its centre by the bar thickness), leaving the
    inner opening dark -- matching the real gate (a hollow red frame). Bloom is a Gaussian glow
    added around the saturated core. Background is the low-light dark structure + a faint floor
    grid so the frame mean/contrast roughly matches the recon (~36/255).

    ``add_bloom`` / ``add_grid`` / ``jpeg_cycle`` toggle the realism layers; geometry is always
    exact. ``seed`` makes the per-frame sensor noise deterministic.
    """

    def __init__(self, *, add_bloom: bool = True, add_grid: bool = True,
                 add_noise: bool = True, inner_size_m: float = GATE_INNER_SIZE_M,
                 seed: int | None = None):
        if cv2 is None:  # pragma: no cover
            raise RuntimeError(f"MockRenderer requires OpenCV (cv2): {_CV2_ERR}")
        self.add_bloom = add_bloom
        self.add_grid = add_grid
        self.add_noise = add_noise
        self.inner_size_m = inner_size_m
        self.seed = seed

    # ---- background ----------------------------------------------------------------
    def _background(self, params: AppearanceParams, rng: np.random.Generator) -> np.ndarray:
        bg = np.empty((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        bg[:] = _rgb_to_bgr(params.background_rgb)
        if self.add_grid:
            grid_bgr = _rgb_to_bgr(params.floor_grid_rgb)
            # faint regular grid over the lower (floor) half -- pure realism / detector negative.
            step = 28
            for x in range(0, IMAGE_WIDTH, step):
                cv2.line(bg, (x, IMAGE_HEIGHT // 2), (x, IMAGE_HEIGHT - 1), grid_bgr, 1)
            for y in range(IMAGE_HEIGHT // 2, IMAGE_HEIGHT, step):
                cv2.line(bg, (0, y), (IMAGE_WIDTH - 1, y), grid_bgr, 1)
            # a few bright neutral ceiling-light points (the main non-gate bright source)
            ceil_bgr = _rgb_to_bgr(params.ceiling_light_rgb)
            for _ in range(8):
                cx = int(rng.integers(0, IMAGE_WIDTH))
                cy = int(rng.integers(0, IMAGE_HEIGHT // 4))
                cv2.circle(bg, (cx, cy), 1, ceil_bgr, -1)
        return bg

    # ---- gate ring -----------------------------------------------------------------
    def _draw_gate(self, img: np.ndarray, pose: CameraPose, params: AppearanceParams) -> None:
        try:
            inner = project_gate_corners(pose.R_cam_gate, pose.t_cam_gate, self.inner_size_m)
        except ValueError:
            return  # gate behind camera -> nothing to draw (a no-gate frame)
        inner = np.asarray(inner, dtype=np.float64)
        ctr = inner.mean(axis=0)
        # OUTER quad: inner scaled out about centre by the bar thickness. inner = outer*(1-2t)
        # (matches red_glow_detector's inset model), so outer = ctr + (inner-ctr)/(1-2t).
        t = float(params.gate_bar_thickness_frac)
        outer = ctr + (inner - ctr) / max(1e-3, (1.0 - 2.0 * t))

        gate_bgr = _rgb_to_bgr(params.gate_emission_rgb)
        # Fill the red RING = outer polygon, then carve the inner opening back to background.
        ring = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(ring, outer.astype(np.int32), 255)
        cv2.fillConvexPoly(ring, inner.astype(np.int32), 0)  # dark inner hole
        img[ring > 0] = gate_bgr

        if self.add_bloom:
            self._add_bloom(img, ring, params)

    def _add_bloom(self, img: np.ndarray, ring_mask: np.ndarray, params: AppearanceParams) -> None:
        """Approximate the gate glow: a Gaussian halo around the saturated core, ADDED (screen-ish)
        so it brightens the surround without erasing the crisp core. Symmetric here (the real VQ2
        bloom is asymmetric, see APPEARANCE_SPEC §4) -- enough for the round-trip core extraction."""
        k = max(3, int(params.bloom_soft_edge_px) | 1)  # odd kernel ~ soft-edge width
        glow = cv2.GaussianBlur(ring_mask, (k, k), 0).astype(np.float32) / 255.0
        gate_bgr = np.array(_rgb_to_bgr(params.gate_emission_rgb), dtype=np.float32)
        wash = np.array(_rgb_to_bgr(params.ambient_red_wash_rgb), dtype=np.float32)
        # additive halo: scale the wash color by the (blurred minus core) glow envelope
        halo_env = np.clip(glow - (ring_mask > 0).astype(np.float32), 0.0, 1.0)[..., None]
        out = img.astype(np.float32) + halo_env * (0.6 * gate_bgr + 0.4 * wash)
        # keep the saturated core EXACTLY at the emissive color (crisp edge preserved)
        out[ring_mask > 0] = gate_bgr
        np.clip(out, 0, 255, out=out)
        img[:] = out.astype(np.uint8)

    # ---- sensor ---------------------------------------------------------------------
    def _add_sensor(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if not self.add_noise:
            return img
        noise = rng.normal(0.0, 2.0, img.shape).astype(np.float32)  # mild low-light grain
        out = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        # JPEG round-trip (the stream is JPEG @ ~q90) -> realistic block/ringing near edges
        ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return out

    def render(self, pose: CameraPose, params: AppearanceParams | None = None) -> np.ndarray:
        params = params or load_appearance()
        rng = np.random.default_rng(self.seed)
        img = self._background(params, rng)
        self._draw_gate(img, pose, params)
        img = self._add_sensor(img, rng)
        return img
