"""VQ2 Blender data-generation pipeline (Blender-INDEPENDENT foundations).

This package is the renderer-agnostic scaffolding the Blender scene plugs into later:

  * ``appearance.py``  -- loads APPEARANCE_SPEC / appearance_params.json (what the scene matches).
  * ``camera_sampler.py`` -- samples plausible racing camera poses relative to a gate.
  * ``projector.py``   -- projects the 1.5 m gate's 4 corners to pixels (reuses racer.vision.gate_pose).
  * ``renderer.py``    -- the abstract Renderer interface + a Blender-free MockRenderer.
  * ``dataset.py``     -- writes ultralytics YOLO-pose labels (matches racer.vision.synthetic).

The real Blender renderer implements ``renderer.Renderer`` and slots in without touching the
rest of the harness. Everything here runs + is tested on the laptop venv with NO Blender and
NO diffaero. See ``tests/test_blender_pipeline.py``.

NB: this is a standalone tool dir (``tools/blender_pipeline``), NOT importable as ``racer.*``.
It depends ONLY on the canonical ``racer.frames`` / ``racer.vision.gate_pose`` /
``racer.vision.red_glow_detector`` (imported, never forked).
"""
