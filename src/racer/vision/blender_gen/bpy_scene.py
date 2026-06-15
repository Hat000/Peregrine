"""Gate mesh + scene placement for the VQ2 photoreal backend (bpy, ShadowPC only).

Builds the gate as a SOLID structural frame (a square annulus prism: outer 2.72 m, inner opening
1.5 m, depth 0.26 m) in the GATE frame (X=right, Y=DOWN, Z=through), SYMMETRIC about Z=0 so the
inner-opening corners -- the detector's keypoints -- sit at (+/-0.75, +/-0.75, 0) and project to
exactly ``contract.project_gate_corners(...)``. This matches the real A2RL x DCL autonomous gate
(a thick painted/branded structural frame, NOT a thin ring and NOT an LED light source -- see the
VQ2 plan), so the rendered occlusion/parallax of the deep frame is faithful.

Placement is in the OPTICAL world (contract): the camera is fixed at the origin (rolled pi about X
by bpy_camera), so a gate at optical pose ``(R_cam_gate, t_cam_gate)`` is placed by assigning that
pose straight into ``matrix_world`` -- no extra transform. Backgrounds (floor / walls / clutter)
live in the same optical world (down = world +Y, downrange = world +Z).

``import bpy`` resolves only inside Blender; this leaf is imported lazily by the Blender backend,
never by the laptop test suite.
"""
from __future__ import annotations

import numpy as np

try:                                    # only importable inside Blender
    import bpy
    import mathutils
except Exception:                       # pragma: no cover - exercised only on ShadowPC
    bpy = None
    mathutils = None

from .contract import GATE_DEPTH_M, GATE_INNER_SIZE_M, GATE_OUTER_SIZE_M

_TEMPLATE_NAME = "VQ2Gate"


# --------------------------------------------------------------------------------------------------
# scene reset
# --------------------------------------------------------------------------------------------------
def clear_scene() -> None:
    """Remove every object + orphan mesh/light/material/image so a run starts clean and memory does
    not grow across many frames (the backend rebuilds camera + gate template after this)."""
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.lights, bpy.data.cameras,
                 bpy.data.materials, bpy.data.images, bpy.data.worlds):
        for block in list(coll):
            if getattr(block, "users", 0) == 0:
                try:
                    coll.remove(block)
                except (RuntimeError, ReferenceError):
                    pass


# --------------------------------------------------------------------------------------------------
# gate mesh (square annulus prism, gate frame)
# --------------------------------------------------------------------------------------------------
def _gate_frame_geometry(inner: float, outer: float, depth: float):
    """Verts + quad faces of the gate frame in the GATE frame (X-right, Y-down, Z-through),
    symmetric about Z=0. 16 verts (8 front at z=-d/2, 8 back at z=+d/2): outer 4 then inner 4 on
    each face; 16 quads (front annulus 4, back annulus 4, outer walls 4, inner walls 4)."""
    oh, ih, zf, zb = outer / 2.0, inner / 2.0, -depth / 2.0, depth / 2.0
    # corner (x, y) in gate frame; y is DOWN so +y is the LOWER edge.
    outer_xy = [(-oh, oh), (oh, oh), (oh, -oh), (-oh, -oh)]   # oLL, oLR, oUR, oUL
    inner_xy = [(-ih, ih), (ih, ih), (ih, -ih), (-ih, -ih)]   # iLL, iLR, iUR, iUL
    verts = []
    for z in (zf, zb):
        for (x, y) in outer_xy:
            verts.append((x, y, z))
        for (x, y) in inner_xy:
            verts.append((x, y, z))
    # index helpers: front outer 0..3, front inner 4..7, back outer 8..11, back inner 12..15
    fo, fi, bo, bi = (0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15)
    faces = []
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):            # 4 sides of the square
        faces.append((fo[a], fo[b], fi[b], fi[a]))           # front annulus strip
        faces.append((bo[a], bo[b], bi[b], bi[a]))           # back annulus strip
        faces.append((fo[a], fo[b], bo[b], bo[a]))           # outer wall
        faces.append((fi[a], fi[b], bi[b], bi[a]))           # inner wall
    return verts, faces


def build_gate_template(name: str = _TEMPLATE_NAME) -> "bpy.types.Object":
    """Build the gate-frame mesh once and return a (hidden-by-the-backend) template Object. The
    backend duplicates its mesh per gate via :func:`instance_gate`."""
    verts, faces = _gate_frame_geometry(GATE_INNER_SIZE_M, GATE_OUTER_SIZE_M, GATE_DEPTH_M)
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    try:
        mesh.shade_flat()          # crisp edges -> exact corner geometry (Blender 4.1+)
    except Exception:
        pass
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


# --------------------------------------------------------------------------------------------------
# placement (optical world)
# --------------------------------------------------------------------------------------------------
def _pose_matrix(R_cam_gate, t_cam_gate) -> "mathutils.Matrix":
    """4x4 world matrix from an optical-frame (R, t): world == optical, so it goes in directly."""
    R = np.asarray(R_cam_gate, dtype=np.float64)
    t = np.asarray(t_cam_gate, dtype=np.float64).reshape(3)
    m = mathutils.Matrix.Identity(4)
    for i in range(3):
        for j in range(3):
            m[i][j] = float(R[i, j])
        m[i][3] = float(t[i])
    return m


def place_gate(obj, R_cam_gate, t_cam_gate) -> None:
    """Set ``obj.matrix_world`` from the optical-frame pose (no extra transform; see module docs)."""
    obj.matrix_world = _pose_matrix(R_cam_gate, t_cam_gate)


def instance_gate(template, R_cam_gate, t_cam_gate, material=None) -> "bpy.types.Object":
    """A fresh gate object (its OWN mesh copy, so per-frame material/cleanup never touch the
    template), placed at the optical pose, with ``material`` assigned if given."""
    mesh = template.data.copy()
    obj = bpy.data.objects.new(f"{template.name}_i", mesh)
    bpy.context.scene.collection.objects.link(obj)
    place_gate(obj, R_cam_gate, t_cam_gate)
    if material is not None:
        obj.data.materials.clear()
        obj.data.materials.append(material)
    return obj


# --------------------------------------------------------------------------------------------------
# background geometry (optical world: down = +Y, downrange = +Z)
# --------------------------------------------------------------------------------------------------
def _add_mesh(name: str, verts, faces) -> "bpy.types.Object":
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _box(name: str, cx, cy, cz, sx, sy, sz) -> "bpy.types.Object":
    """An axis-aligned box centred at (cx,cy,cz) with half-extents (sx,sy,sz), in optical world."""
    xs, ys, zs = (cx - sx, cx + sx), (cy - sy, cy + sy), (cz - sz, cz + sz)
    verts = [(x, y, z) for z in zs for y in ys for x in xs]
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
             (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return _add_mesh(name, verts, faces)


def build_background(rng, appearance) -> "list[bpy.types.Object]":
    """A floor + (indoor) back/side walls + random clutter props in the +Z region the camera sees.

    Optical world: 'down' is world +Y, so the floor is a large plane at +Y below the flight path;
    the scene/gates extend to +Z. ``background_mode`` (arena|indoor|outdoor|mixed) toggles walls;
    ``clutter_max`` caps random boxes (confusers + structure). Returns the created objects (the
    backend assigns them one background material and purges them after the frame)."""
    objs = []
    mode = appearance.background_mode
    if mode == "mixed":
        mode = ("arena", "indoor", "outdoor")[int(rng.integers(0, 3))]

    floor_y = float(rng.uniform(2.5, 7.0))                    # depth below the camera (down = +Y)
    far_z = float(rng.uniform(25.0, 60.0))
    # floor: large quad spanning X and Z at y = floor_y.
    fx, fz0, fz1 = 40.0, -5.0, far_z
    floor = _add_mesh("VQ2_Floor",
                      [(-fx, floor_y, fz0), (fx, floor_y, fz0), (fx, floor_y, fz1), (-fx, floor_y, fz1)],
                      [(0, 1, 2, 3)])
    objs.append(floor)

    if mode in ("arena", "indoor"):
        ceil_y = -float(rng.uniform(4.0, 9.0))               # ceiling above (up = -Y)
        objs.append(_add_mesh("VQ2_Ceiling",
                    [(-fx, ceil_y, fz0), (fx, ceil_y, fz0), (fx, ceil_y, fz1), (-fx, ceil_y, fz1)],
                    [(0, 1, 2, 3)]))
        objs.append(_add_mesh("VQ2_BackWall",                # far wall (dark-arena backdrop)
                    [(-fx, ceil_y, far_z), (fx, ceil_y, far_z), (fx, floor_y, far_z), (-fx, floor_y, far_z)],
                    [(0, 1, 2, 3)]))
        wx = float(rng.uniform(8.0, 18.0))                   # side walls
        for sx in (-wx, wx):
            objs.append(_add_mesh(f"VQ2_SideWall_{sx:.0f}",
                        [(sx, ceil_y, fz0), (sx, ceil_y, far_z), (sx, floor_y, far_z), (sx, floor_y, fz0)],
                        [(0, 1, 2, 3)]))

    # random clutter / confuser props scattered through the +Z region.
    n_clutter = int(rng.integers(0, int(appearance.clutter_max) + 1))
    for k in range(n_clutter):
        cz = float(rng.uniform(3.0, far_z * 0.9))
        cx = float(rng.uniform(-12.0, 12.0))
        cy = float(rng.uniform(-3.0, floor_y))
        s = float(rng.uniform(0.3, 2.5))
        objs.append(_box(f"VQ2_Clutter_{k}", cx, cy, cz, s, s, float(rng.uniform(0.3, 3.0))))
    return objs
