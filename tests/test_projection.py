"""Tests for the camera-geometry helpers + projection conventions (mirror projection_check)."""
from __future__ import annotations

import numpy as np

from racer import frames

_K = frames.CAMERA_INTRINSICS_K
_CX, _CY = _K[0, 2], _K[1, 2]


def _proj(p_world):
    R = frames.R_world_from_body(0.0, 0.0, 0.0)   # level, facing north
    return frames.project_camera_point(frames.world_point_in_camera(np.asarray(p_world, float), R, np.zeros(3)))


def test_fov_values_match_intrinsics():
    assert abs(frames.horizontal_fov_deg() - 90.0) < 0.5     # spec's mislabelled "VFoV"
    assert abs(frames.vertical_fov_deg() - 58.72) < 0.5      # the REAL vertical FoV


def test_elevation_band_looks_up():
    lo, hi = frames.camera_elevation_band_deg()
    assert abs(lo - (-9.36)) < 1.0
    assert abs(hi - 49.36) < 1.0
    assert abs((hi - lo) - frames.vertical_fov_deg()) < 1e-6


def test_level_ahead_projects_below_centre():
    uv = _proj([10.0, 0.0, 0.0])                  # camera tilts up -> level point lands low
    assert uv is not None and uv[1] > _CY


def test_plus20deg_elevation_projects_near_centre():
    up = 10.0 * np.tan(frames.CAMERA_PITCH_RAD)
    uv = _proj([10.0, 0.0, -up])                  # +20 deg elevation = optical axis
    assert uv is not None and abs(uv[1] - _CY) < 5.0


def test_right_point_projects_right_of_centre():
    uv = _proj([10.0, 5.0, 0.0])
    assert uv is not None and uv[0] > _CX


def test_point_behind_does_not_project():
    assert _proj([-10.0, 0.0, 0.0]) is None


def test_pixel_ray_pixel_roundtrip():
    u0, v0 = 500.0, 120.0
    ray = np.linalg.inv(_K) @ np.array([u0, v0, 1.0])
    uv = frames.project_camera_point(ray)
    assert uv is not None and abs(uv[0] - u0) < 1e-6 and abs(uv[1] - v0) < 1e-6
