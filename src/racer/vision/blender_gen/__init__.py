"""VQ2 photoreal gate-course dataset generator (Blender/Cycles + domain randomization).

The PHOTOREAL upgrade to ``racer.vision.synthetic``: same YOLO-pose label contract and
keypoint convention, but the gate images come from a physically-based Cycles render with
wide appearance domain-randomization (gate hue/material, lighting/HDRI, backgrounds,
motion blur, flares) so the detector generalises to the VQ2 / Round-2 3D-scanned arena
appearance gap instead of overfitting the VQ1 "gambling-red" look.

Design that keeps the labels PROVABLY correct (the #1 risk in synthetic data):

  * The whole geometry/label path is pure-Python and unit-tested on the laptop with NO
    Blender (``contract`` / ``intrinsics`` / ``geometry`` / ``labels`` / ``augment`` /
    ``config`` / ``dataset``). Corners are projected through ``racer.frames`` +
    ``racer.vision.gate_pose`` exactly as the deployed estimator consumes them.
  * Rendering is behind a ``RenderBackend`` protocol (``dataset.RenderBackend``). The
    ``ProceduralBackend`` (numpy/cv2, laptop) lets the FULL pipeline run + be tested here;
    the ``BlenderBackend`` (bpy/Cycles, ShadowPC) is the photoreal drop-in. The labels are
    backend-independent: the renderer never feeds them back.
  * Frames are rendered in the CAMERA-OPTICAL frame (camera at origin, rolled 180 deg about
    its X so Blender-world == OpenCV optical frame), so a gate placed at its (R_cam_gate,
    t_cam_gate) projects to EXACTLY ``project_gate_corners(R_cam_gate, t_cam_gate)``.

Entry points: ``scripts/gen_blender_dataset.py`` (laptop, procedural) and
``render_entry.py`` (``blender --background --python`` on ShadowPC, Cycles). See the bundled
``RUN_GUIDE.md`` / ``DATASET_CONTRACT.md`` under handoff/vq2-blender-dataset-2026-06-14/.
"""
from __future__ import annotations
