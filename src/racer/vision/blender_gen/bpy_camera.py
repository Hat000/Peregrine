"""Configure the Blender camera to reproduce K exactly -- and prove it (bpy, ShadowPC only).

The render-leaf counterpart to ``intrinsics.py``: ``intrinsics.blender_camera_params()`` turns our
pixel-space K into Blender camera settings (lens/sensor/shift) with a NO-Blender round-trip test;
THIS module sets those settings on a live ``bpy`` camera and projects known optical points through
it to confirm the live render matches K to <= 1 px (the ShadowPC arbiter the math gate stands in for).

Optical-frame convention (contract.py): the camera sits at the WORLD ORIGIN with
``matrix_world = Matrix.Rotation(pi, 4, 'X')`` so Blender-world coordinates ARE OpenCV optical
coordinates -- a point at optical (x_right, y_down, z_forward) renders at the K-projection of
(x, y, z). Gates are then placed DIRECTLY at their optical pose ``(R_cam_gate, t_cam_gate)`` by the
scene leaf, and project to exactly ``contract.project_gate_corners(...)`` (the labels).

``import bpy`` / ``import bpy_extras`` succeed only inside Blender; this module is imported lazily by
the Blender backend, never by the pure-Python core (which the laptop test suite exercises).
"""
from __future__ import annotations

import bpy
import bpy_extras.object_utils
import mathutils
import numpy as np

from . import contract
from .intrinsics import BlenderCameraParams, blender_camera_params

# Sensible clip planes for the gate envelope (RANGE_MIN/MAX = 2..30 m); generous so nothing in the
# scene (gates, lights, background props) is ever clipped. clip_start > 0 is required by Blender.
CAMERA_CLIP_START = 0.05
CAMERA_CLIP_END = 500.0

_CAM_NAME = "vq2_optical_cam"


def _optical_world_matrix() -> mathutils.Matrix:
    """matrix_world that makes Blender-world == OpenCV optical frame: a pi rotation about world X
    (== diag(1,-1,-1)), camera at the origin. See contract.BLENDER_CAM_ROTATION_X_RAD."""
    return mathutils.Matrix.Rotation(contract.BLENDER_CAM_ROTATION_X_RAD, 4, "X")


def setup_camera(scene, params: BlenderCameraParams | None = None) -> "bpy.types.Object":
    """Build a PERSP camera that reproduces K, place it at the optical-frame origin, return its Object.

    Sets ``camera.data`` (lens / sensor_width / sensor_height / sensor_fit=HORIZONTAL / shift_x=shift_y=0
    / clip) from ``params`` (default ``intrinsics.blender_camera_params()`` -> lens 18 mm on a 36 mm
    sensor), sets ``scene.render`` resolution (640x360 @ 100%, square pixels), links the object to the
    scene master collection, makes it ``scene.camera``, and rolls it pi about world X so the world frame
    is the OpenCV optical frame. Verify with :func:`projection_max_error_px` (<= 1 px).
    """
    if params is None:
        params = blender_camera_params()

    cam_data = bpy.data.cameras.new(_CAM_NAME)
    cam_data.type = "PERSP"
    cam_data.lens_unit = "MILLIMETERS"
    cam_data.lens = float(params.lens_mm)
    cam_data.sensor_fit = params.sensor_fit            # "HORIZONTAL"
    cam_data.sensor_width = float(params.sensor_width_mm)
    cam_data.sensor_height = float(params.sensor_height_mm)
    cam_data.shift_x = float(params.shift_x)           # 0.0 (principal point centred)
    cam_data.shift_y = float(params.shift_y)           # 0.0
    cam_data.clip_start = CAMERA_CLIP_START
    cam_data.clip_end = CAMERA_CLIP_END

    # Resolution + square pixels. HORIZONTAL fit derives the vertical focal from the resolution at
    # square pixels, so fy == fx falls out -- never set by hand (see intrinsics.py).
    rs = scene.render
    rs.resolution_x = int(params.render_width)         # 640
    rs.resolution_y = int(params.render_height)        # 360
    rs.resolution_percentage = 100
    rs.pixel_aspect_x = 1.0
    rs.pixel_aspect_y = 1.0

    cam_obj = bpy.data.objects.new(_CAM_NAME, cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj
    cam_obj.matrix_world = _optical_world_matrix()
    return cam_obj


def world_to_pixel(scene, cam_obj, p_optical) -> tuple[float, float]:
    """Project an optical-frame 3D point to pixel coords (u, v) through the LIVE Blender camera.

    Uses ``bpy_extras.object_utils.world_to_camera_view`` (honours lens/sensor/shift exactly as the
    render does) -> normalized device coords (ndc_x, ndc_y in [0,1], origin BOTTOM-LEFT, ndc.z =
    forward depth; negative z means behind the camera). Converts to image pixels (origin TOP-LEFT,
    +v down) matching K / OpenCV:
        u = ndc_x * W ;  v = (1 - ndc_y) * H.
    p_optical is in the optical/world frame (camera at origin, Rx(pi)) so it is passed straight in.
    """
    ndc = bpy_extras.object_utils.world_to_camera_view(
        scene, cam_obj, mathutils.Vector(tuple(float(c) for c in p_optical))
    )
    w = int(scene.render.resolution_x)
    h = int(scene.render.resolution_y)
    u = float(ndc.x) * w
    v = (1.0 - float(ndc.y)) * h
    return u, v


def _gate_corner_points() -> np.ndarray:
    """A handful of known optical-frame 3D points: the 4 inner-square gate corners (the keypoints,
    contract.gate_object_points order) for a head-on gate at a few ranges across the envelope.
    Head-on => R_cam_gate = I, t_cam_gate = (0, 0, range): the corners land in front of the camera,
    spread across the frame, so the projection check exercises off-axis pixels, not just the centre.
    """
    obj = contract.gate_object_points(contract.GATE_INNER_SIZE_M)        # (4,3) gate-frame corners
    ranges = (
        contract.RANGE_MIN_M,
        0.5 * (contract.RANGE_MIN_M + contract.RANGE_MAX_M),
        contract.RANGE_MAX_M,
    )
    pts = []
    for rng_m in ranges:
        # R_cam_gate = I, t = (0,0,range): gate corners in the optical/world frame.
        pts.append(obj + np.array([0.0, 0.0, float(rng_m)], dtype=np.float64))
    return np.concatenate(pts, axis=0)                                   # (12, 3)


def _project_through_K(points_optical: np.ndarray) -> np.ndarray:
    """Project optical-frame points through contract.CAMERA_INTRINSICS_K (the label/PnP intrinsics).
    Pinhole: u = K @ (x,y,z) / z. Returns (N, 2) pixel coords (origin top-left), the GROUND TRUTH
    the live Blender projection must match."""
    K = np.asarray(contract.CAMERA_INTRINSICS_K, dtype=np.float64)
    cam = np.asarray(points_optical, dtype=np.float64)
    uvw = (K @ cam.T).T                                                  # (N,3)
    return uvw[:, :2] / uvw[:, 2:3]


def projection_max_error_px(scene, cam_obj) -> float:
    """Max abs pixel disagreement between the LIVE Blender camera and K over known optical points.

    Projects the gate inner-corners (a few ranges) through BOTH ``world_to_pixel`` (the real Blender
    camera) AND ``contract.CAMERA_INTRINSICS_K``, returns the worst |Du|,|Dv| over all points. This is
    the live ShadowPC intrinsics arbiter -- it MUST be <= 1 px; > 1 means the camera data-block does
    not match K (a lens/sensor/shift/resolution mistake) and the dataset would be mislabelled.
    """
    points = _gate_corner_points()
    k_px = _project_through_K(points)
    max_err = 0.0
    for i in range(points.shape[0]):
        u, v = world_to_pixel(scene, cam_obj, points[i])
        max_err = max(max_err, abs(u - float(k_px[i, 0])), abs(v - float(k_px[i, 1])))
    return float(max_err)