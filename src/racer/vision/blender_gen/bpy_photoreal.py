"""Photoreal scene dressing for the VQ2 backend (bpy, ShadowPC only).

The validated 'real-asset' recipe, consolidated: an HDRI environment (image-based lighting + a real
photographic background, oriented CORRECTLY for the optical frame), a PBR-textured floor placed a
good way BELOW the flight line, real photoscanned props + procedural 'mannequin' people scattered OFF
the gate corridor, a plain solid vivid gate material, and a clean render (NO in-render motion blur or
glare -- camera artifacts are added later by ``augment.py``). It also downgrades a gate keypoint to
OCCLUDED when a prop/person actually blocks the line of sight (raycast), so labels stay honest.

This NEVER touches the gate poses or the projected keypoints -- only pixels and (raycast) per-corner
visibility. ``import bpy`` resolves only inside Blender; imported lazily by the backend.
"""
from __future__ import annotations

import glob
import math
import os
import random

import numpy as np

import bpy
import mathutils

from . import bpy_materials as M
from . import bpy_render
from .contract import (
    GATE_INNER_SIZE_M, GATE_OUTER_SIZE_M, V_OCC, V_VIS,
    VQ1_GATE_RED_RGB_LINEAR, gate_object_points,
)
from .geometry import FrameSpec

_OUTER_H = GATE_OUTER_SIZE_M / 2.0


# ----------------------------------------------------------------------------- world (HDRI + fix)
def setup_hdri_world(scene, rng, hdri_path: str, strength: float) -> bool:
    """Light + back the scene with an equirectangular HDRI, oriented for the OPTICAL frame.

    Critical: our world is the optical frame (up = -Y), but an equirect HDRI assumes up = +Z, so its
    zenith (ceiling) would render straight ahead and the back 'wall' would look like a ceiling. We
    rotate the env mapping -90 deg about X so HDRI-up aligns with optical-up, then spin a random
    azimuth about that vertical. Returns False if the image cannot be loaded.
    """
    world = scene.world or bpy.data.worlds.new("VQ2_World")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    bg = nt.nodes.new("ShaderNodeBackground")
    out = nt.nodes.new("ShaderNodeOutputWorld")
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    mp = nt.nodes.new("ShaderNodeMapping")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    try:
        env.image = bpy.data.images.load(hdri_path, check_existing=True)
    except Exception:
        return False
    mp.inputs["Rotation"].default_value[0] = math.radians(-90.0)          # optical-up alignment
    mp.inputs["Rotation"].default_value[2] = float(rng.uniform(0.0, 2.0 * math.pi))  # azimuth spin
    nt.links.new(tc.outputs["Generated"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], env.inputs["Vector"])
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = float(strength)
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    return True


# ----------------------------------------------------------------------------- gate material
def solid_gate_material(rng, appearance):
    """Plain solid vivid gate colour (VQ1 red, or hue-randomized), with a SMALL emission lift so it
    reads bright without washing out the hue, beveled edges. No branding/decals."""
    if appearance.use_vq1_red and float(appearance.gate_hue_jitter) == 0.0:
        base = tuple(VQ1_GATE_RED_RGB_LINEAR)
    else:
        h = M._vq1_red_hue() + float(rng.uniform(-1, 1)) * float(appearance.gate_hue_jitter) * 0.5 \
            if float(appearance.gate_hue_jitter) < 0.5 else float(rng.random())
        base = M._hsv_to_linear_rgb(h, M._u(rng, appearance.gate_sat_range), M._u(rng, appearance.gate_val_range))
    mat = bpy.data.materials.new("VQ2_Gate")
    mat.use_nodes = True
    nt = mat.node_tree
    b = nt.nodes.get("Principled BSDF") or nt.nodes.new("ShaderNodeBsdfPrincipled")
    M._set_socket(b, "Base Color", M._color4(base))
    M._set_socket(b, "Emission Color", M._color4(base))
    M._set_socket(b, ("Emission Strength", "Emission"), M._u(rng, getattr(appearance, "gate_emission_range", (0.25, 0.5))))
    M._set_socket(b, "Metallic", M._u(rng, appearance.gate_metallic_range))
    M._set_socket(b, "Roughness", M._u(rng, appearance.gate_roughness_range))
    bev = nt.nodes.new("ShaderNodeBevel")
    bev.inputs["Radius"].default_value = 0.015
    nt.links.new(bev.outputs["Normal"], b.inputs["Normal"])
    return mat


# ----------------------------------------------------------------------------- PBR floor
def floor_material(texset: dict, rng):
    mat = bpy.data.materials.new("VQ2_Floor")
    mat.use_nodes = True
    nt = mat.node_tree
    b = nt.nodes.get("Principled BSDF") or nt.nodes.new("ShaderNodeBsdfPrincipled")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    s = float(rng.uniform(6.0, 10.0))
    mp.inputs["Scale"].default_value = (s, s, s)
    nt.links.new(tc.outputs["UV"], mp.inputs["Vector"])

    def img(path, noncolor=False):
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = bpy.data.images.load(path, check_existing=True)
        if noncolor:
            n.image.colorspace_settings.name = "Non-Color"
        nt.links.new(mp.outputs["Vector"], n.inputs["Vector"])
        return n

    if texset.get("diff"):
        nt.links.new(img(texset["diff"]).outputs["Color"], b.inputs["Base Color"])
    if texset.get("arm"):
        sep = nt.nodes.new("ShaderNodeSeparateColor")
        nt.links.new(img(texset["arm"], True).outputs["Color"], sep.inputs["Color"])
        nt.links.new(sep.outputs[1], b.inputs["Roughness"])               # arm.G = roughness
    if texset.get("nor"):
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nt.links.new(img(texset["nor"], True).outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], b.inputs["Normal"])
    return mat


def add_floor(scene, floor_y: float, texset: dict, rng):
    mesh = bpy.data.meshes.new("VQ2_FloorMesh")
    mesh.from_pydata([(-60, floor_y, -12), (60, floor_y, -12), (60, floor_y, 90), (-60, floor_y, 90)],
                     [], [(0, 1, 2, 3)])
    mesh.uv_layers.new(name="UV")
    for i, uv in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)]):
        mesh.uv_layers[0].data[i].uv = uv
    mesh.update()
    obj = bpy.data.objects.new("VQ2_Floor", mesh)
    scene.collection.objects.link(obj)
    if texset:
        obj.data.materials.append(floor_material(texset, rng))
    return obj


# ----------------------------------------------------------------------------- props + people
def _stand_into_optical(empty, children, x, z, floor_y, rng, scale):
    """Stand a Blender +Z-up group into the optical frame (up = -Y) at (x, floor, z), drop to floor."""
    yaw = float(rng.uniform(0, 2 * math.pi))
    R = mathutils.Matrix.Rotation(yaw, 4, "Y") @ mathutils.Matrix.Rotation(math.pi / 2, 4, "X")
    empty.matrix_world = mathutils.Matrix.Translation((x, 0.0, z)) @ R @ mathutils.Matrix.Scale(scale, 4)
    bpy.context.view_layer.update()
    maxy = -1e9
    for o in children:
        if o.type == "MESH":
            for c in o.bound_box:
                maxy = max(maxy, (o.matrix_world @ mathutils.Vector(c)).y)
    if maxy > -1e8:
        empty.location.y += (floor_y - maxy)
    bpy.context.view_layer.update()


class PropCache:
    """Import each glTF prop ONCE, then spawn per-frame DUPLICATES that share the mesh data.

    Re-parsing 10-15 glTF models (+ textures) every frame was the throughput killer (~25 s/frame).
    Here each model is imported once into a hidden template hierarchy; a spawn copies the objects
    (``obj.copy()`` shares the mesh datablock -- no re-parse, no texture reload), so per-frame prop
    placement is cheap. Templates are ``hide_render`` so they never appear; spawns are visible and get
    purged each frame by the backend.
    """

    def __init__(self, scene, gltf_paths: list[str]):
        self.scene = scene
        self.entries: list[tuple] = []     # (root_empty, [child_objs]) per model, hidden templates
        for path in gltf_paths:
            before = set(bpy.data.objects)
            try:
                bpy.ops.import_scene.gltf(filepath=path)
            except Exception:
                continue
            new = [o for o in bpy.data.objects if o not in before]
            if not new:
                continue
            root = bpy.data.objects.new("VQ2_PropTmpl", None)
            scene.collection.objects.link(root)
            for o in new:
                if o.parent is None:
                    o.parent = root
                o.hide_render = True
            root.hide_render = True
            root.location = (0.0, -5000.0, 0.0)    # park far away so templates never hit the raycast
            bpy.context.view_layer.update()
            self.entries.append((root, new))

    def available(self) -> bool:
        return bool(self.entries)

    def spawn(self, x, z, floor_y, rng) -> list:
        if not self.entries:
            return []
        root0, children0 = self.entries[int(rng.integers(len(self.entries)))]
        root = root0.copy()
        root.hide_render = False
        self.scene.collection.objects.link(root)
        spawned = [root]
        for c0 in children0:
            c = c0.copy()                      # shares mesh data (no re-parse)
            c.hide_render = False
            self.scene.collection.objects.link(c)
            c.parent = root
            c.matrix_parent_inverse = c0.matrix_parent_inverse.copy()
            spawned.append(c)
        _stand_into_optical(root, [o for o in spawned if o.type == "MESH"], x, z, floor_y, rng,
                            float(rng.uniform(0.85, 1.5)))
        return spawned


def _mat(name, lin, rough=0.7):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes.get("Principled BSDF")
    M._set_socket(b, "Base Color", M._color4(lin))
    M._set_socket(b, "Roughness", rough)
    return m


def add_person(scene, x, z, floor_y, rng) -> list:
    """A simple clothing-tinted humanoid mannequin (legs=trousers, torso+arms=shirt, head=skin)."""
    shirt = _mat("shirt", M._hsv_to_linear_rgb(float(rng.random()), float(rng.uniform(0.5, 1.0)), float(rng.uniform(0.4, 0.85))))
    trousers = _mat("trousers", M._hsv_to_linear_rgb(float(rng.choice([0.6, 0.08, 0.0, 0.33])), float(rng.uniform(0.2, 0.6)), float(rng.uniform(0.12, 0.4))))
    skin = _mat("skin", (float(rng.uniform(0.25, 0.6)), float(rng.uniform(0.12, 0.35)), float(rng.uniform(0.08, 0.25))))
    parts = []

    def cyl(r, depth, loc, mat, sx=1.0, sy=1.0):
        bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=depth, location=loc, vertices=14)
        o = bpy.context.active_object
        o.scale = (sx, sy, 1.0)
        o.data.materials.append(mat)
        parts.append(o)
        return o

    for sx in (-1, 1):
        cyl(0.1, 0.92, (0.12 * sx, 0, 0.46), trousers)
    cyl(0.2, 0.64, (0, 0, 1.22), shirt, sx=1.15, sy=0.62)
    for sx in (-1, 1):
        cyl(0.06, 0.62, (0.30 * sx, 0, 1.18), shirt)
    cyl(0.05, 0.1, (0, 0, 1.58), skin)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.125, location=(0, 0, 1.74), segments=14, ring_count=8)
    head = bpy.context.active_object
    head.data.materials.append(skin)
    parts.append(head)
    empty = bpy.data.objects.new("VQ2_Person", None)
    scene.collection.objects.link(empty)
    for p in parts:
        p.parent = empty
    _stand_into_optical(empty, parts, x, z, floor_y, rng, float(rng.uniform(0.92, 1.12)))
    return parts + [empty]


def _side_xz(rng):
    """A floor position OFF the central gate corridor: a side band (large |x|) or far background."""
    if float(rng.random()) < 0.6:
        return float(rng.choice([-1, 1])) * float(rng.uniform(6.0, 16.0)), float(rng.uniform(3.0, 30.0))
    return float(rng.uniform(-16.0, 16.0)), float(rng.uniform(16.0, 34.0))


# ----------------------------------------------------------------------------- occlusion (raycast)
def _is_obstacle_hit(obj) -> bool:
    """True if a raycast hit object is a PROP/PERSON (a real occluder) rather than a gate instance or
    the floor. Gate instances are named 'VQ2Gate*' (a gate hitting its OWN angled frame/corner is NOT
    occlusion); the floor is 'VQ2_Floor*'. Everything else placed in the scene is a prop or person."""
    if obj is None:
        return False
    name = getattr(getattr(obj, "original", obj), "name", "") or ""
    return not (name.startswith("VQ2Gate") or name.startswith("VQ2_Floor"))


def occlude_blocked_keypoints(scene, frame: FrameSpec) -> None:
    """Raycast camera->corner for each VISIBLE gate keypoint; if a PROP/PERSON blocks it (a hit on a
    non-gate, non-floor object closer than the corner), downgrade that corner V_VIS -> V_OCC. Mutates
    frame.gates in place so the (post-augment) label rows reflect the real occlusion. Props sit off
    the corridor, so this is usually a no-op -- but it keeps the labels honest when they do block.
    Crucially it does NOT treat a gate hitting its own angled frame/corner as occlusion.

    PER-CORNER ONLY. This function marks corners; it does NOT decide whether a gate gets a label.
    It used to end with ``if (visibility == V_VIS).sum() < 3: gr.visible = False`` -- the retired
    ">= 3 in-frame corners" rule surviving in a fourth place after being replaced in three
    (geometry._apply_occlusion_and_visibility, geometry.sample_partial_frame, augment._recompute).
    It ran BEFORE augment_frame splits labelled from unlabelled, so a gate it zeroed could never
    come back: the backend rendered a big obvious gate into the image and training was told it was
    background. Worse, the count included corners that were merely OFF-FRAME, so it re-killed 30%
    (partial arm) / 39% (faithful arm) of the very crops the area rule exists to rescue -- with no
    prop anywhere near them. The whole-gate decision now lives where the honest measurement is:
    the backend runs the object-id pass and applies geometry.has_visible_silhouette.
    """
    deps = bpy.context.evaluated_depsgraph_get()
    obj_pts = np.asarray(gate_object_points(GATE_INNER_SIZE_M), dtype=np.float64)   # (4,3) gate frame
    origin = mathutils.Vector((0.0, 0.0, 0.0))
    for gr in frame.gates:
        if not gr.visible:
            continue
        corners = (np.asarray(gr.R_cam_gate) @ obj_pts.T).T + np.asarray(gr.t_cam_gate)  # (4,3) optical
        for c in range(4):
            if int(gr.visibility[c]) != V_VIS:
                continue
            p = mathutils.Vector((float(corners[c, 0]), float(corners[c, 1]), float(corners[c, 2])))
            d = (p - origin)
            dist = d.length
            if dist < 1e-4:
                continue
            hit, loc, _n, _idx, obj, _m = scene.ray_cast(deps, origin, d.normalized())
            if hit and (loc - origin).length < dist - 0.10 and _is_obstacle_hit(obj):
                gr.visibility[c] = V_OCC
        # NO whole-gate drop here -- see the docstring. Four rays through four corners cannot
        # measure how much of a gate is visible: a pillar splitting a gate down the middle leaves
        # all four corners V_VIS, and a gate cropped to one visible corner has three V_OFF while
        # most of its ring is on screen. Coverage is an AREA question and is answered with the
        # rendered silhouette in backends/blender.py.


# ----------------------------------------------------------------------------- floor height
def floor_below_gates(frame: FrameSpec, rng) -> float:
    """A ground-plane optical-Y (down = +Y) a good way BELOW every gate, so gates float in the air."""
    obj_pts = np.asarray(gate_object_points(GATE_OUTER_SIZE_M), dtype=np.float64)
    lowest = 0.0  # camera height (optical Y = 0) as a floor for the floor
    for gr in frame.gates:
        corners = (np.asarray(gr.R_cam_gate) @ obj_pts.T).T + np.asarray(gr.t_cam_gate)
        lowest = max(lowest, float(corners[:, 1].max()))
    return lowest + float(rng.uniform(1.5, 4.0))


# ----------------------------------------------------------------------------- clean render config
def configure_clean_render(scene, rng, render_cfg, appearance) -> None:
    """Engine + samples + GPU + denoise + AgX + per-frame exposure. NO in-render motion blur/glare."""
    engine = bpy_render._resolve_engine(scene, render_cfg.engine)
    scene.render.engine = engine
    if engine == "CYCLES":
        scene.cycles.samples = int(render_cfg.samples)
        scene.cycles.use_denoising = bool(render_cfg.use_denoise)
        if render_cfg.use_gpu and not bpy_render._enable_cycles_gpu(scene):
            scene.cycles.device = "CPU"
    else:
        try:
            scene.eevee.taa_render_samples = int(render_cfg.samples)
        except Exception:
            pass
    scene.render.use_motion_blur = False          # camera artifacts -> albumentations (augment.py)
    scene.render.use_compositing = False          # no in-render glare/bloom
    try:
        scene.view_settings.view_transform = "AgX"        # filmic tonemap (no blown highlights)
    except Exception:
        pass
    lo, hi = getattr(appearance, "exposure_range", (-0.4, 0.5))
    scene.view_settings.exposure = float(rng.uniform(float(lo), float(hi)))
