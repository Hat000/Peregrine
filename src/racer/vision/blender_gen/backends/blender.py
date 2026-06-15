"""bpy/Cycles photoreal render backend (ShadowPC). Drop-in for ProceduralBackend.

Composes the four bpy leaf modules (``bpy_camera`` / ``bpy_scene`` / ``bpy_materials`` /
``bpy_render``) behind the ``dataset.RenderBackend`` protocol. The geometry/labels are owned by
``geometry.py`` (pure Python) and NEVER read back from Blender -- this backend only turns each
FrameSpec's optical gate poses into photoreal pixels, so a Cycles dataset and a procedural dataset
share byte-identical labels.

Everything bpy is imported lazily inside ``__init__`` / ``render`` so the module imports fine on a
Blender-less laptop (the laptop uses ProceduralBackend). Run it via ``render_entry.py`` under
``blender --background --python``.

Per-frame protocol: keep the camera + gate-mesh template across frames (cheap), rebuild the
randomized material / lighting / world / background / gate instances each frame, render to a temp
PNG, read it back as BGR, and delete the per-frame objects so memory does not grow over a long run.
"""
from __future__ import annotations

import numpy as np

from racer.frames import R_camera_from_body, R_world_from_body

from ..config import ScenarioPreset
from ..geometry import FrameSpec


def _optical_velocity(frame: FrameSpec) -> np.ndarray:
    """Body NED velocity expressed in the camera optical frame (for motion-blur streaking)."""
    R_wb = R_world_from_body(frame.roll, frame.pitch, frame.yaw)
    return R_camera_from_body() @ R_wb.T @ np.asarray(frame.body_vel_ned, dtype=np.float64)


class BlenderBackend:
    """Stateful Cycles/Eevee backend. Construct ONCE (sets up camera + gate template), then call
    ``render`` per frame. Must be constructed inside a running Blender (``import bpy`` succeeds)."""

    def __init__(self, preset: ScenarioPreset):
        import bpy  # noqa: F401  (only available inside Blender)

        from .. import bpy_camera, bpy_materials, bpy_render, bpy_scene

        self._bpy = bpy
        self._cam_mod = bpy_camera
        self._scene_mod = bpy_scene
        self._mat_mod = bpy_materials
        self._render_mod = bpy_render
        self.preset = preset

        bpy_scene.clear_scene()
        self.scene = bpy.context.scene
        self.cam = bpy_camera.setup_camera(self.scene)
        self.template = bpy_scene.build_gate_template()
        # hide the bare template from renders (we render duplicated instances of its mesh)
        try:
            self.template.hide_render = True
        except Exception:
            pass
        self._frame_objects: list = []

    # -- intrinsics self-check (the ShadowPC <=1 px gate) --------------------------------
    def intrinsics_error_px(self) -> float:
        return float(self._cam_mod.projection_max_error_px(self.scene, self.cam))

    # -- per-frame cleanup ---------------------------------------------------------------
    def _purge_frame_objects(self) -> None:
        bpy = self._bpy
        for obj in self._frame_objects:
            try:
                bpy.data.objects.remove(obj, do_unlink=True)
            except Exception:
                pass
        self._frame_objects = []
        # drop now-orphan mesh/light/material/image data so memory does not grow over a long run
        try:
            bpy.data.orphans_purge(do_recursive=True)
        except Exception:
            pass

    # -- the RenderBackend protocol ------------------------------------------------------
    def render(self, frame: FrameSpec, preset: ScenarioPreset, rng: np.random.Generator) -> np.ndarray:
        ap, rc = preset.appearance, preset.render

        # 1. world + lighting + background (fresh randomization per frame)
        self._mat_mod.setup_world(rng, ap)
        self._frame_objects += list(self._mat_mod.add_lighting(rng, ap))
        bg_objs = list(self._scene_mod.build_background(rng, ap))
        if bg_objs:
            bg_mat = self._mat_mod.background_material(rng, ap)
            for o in bg_objs:
                self._mat_mod.apply_material(o, bg_mat)
        self._frame_objects += bg_objs

        # 2. one shared gate material per frame (real gates on a course look identical)
        gate_mat = self._mat_mod.gate_material(rng, ap)
        gate_objs = []
        for gr in frame.gates:                                   # place every in-band gate (occlusion renders)
            obj = self._scene_mod.instance_gate(self.template, gr.R_cam_gate, gr.t_cam_gate, gate_mat)
            gate_objs.append(obj)
        self._frame_objects += gate_objs

        # 3. render config (+ camera-motion blur from the optical-frame velocity)
        self._render_mod.configure_render(self.scene, rc, ap)
        if rc.motion_blur:
            self._render_mod.set_motion_blur(
                self.scene, gate_objs + bg_objs, self.cam, _optical_velocity(frame), rc.motion_blur_shutter
            )

        # 4. render -> BGR, then tear down this frame's objects
        image = self._render_mod.render_to_bgr(self.scene)
        self._purge_frame_objects()
        return np.ascontiguousarray(image)
