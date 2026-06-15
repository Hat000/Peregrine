"""Render backends for the VQ2 dataset generator.

``dataset.ProceduralBackend`` (numpy/cv2, laptop) lives in ``dataset`` so the core has no bpy
dependency. ``blender.BlenderBackend`` (bpy/Cycles, ShadowPC) lives here and imports bpy lazily,
so importing this package on a machine without Blender never fails until you actually construct
the Blender backend.
"""
from __future__ import annotations
