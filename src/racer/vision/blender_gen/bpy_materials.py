"""Appearance domain-randomization for the VQ2 photoreal backend (gate/background materials,
lighting, world).  RUNS INSIDE BLENDER (``blender --background --python``); ``import bpy`` only
resolves there.  Every bpy class / socket / enum below is verified against the bundled Blender
4.2+ RST (see the module's API-used list).  Pure appearance only -- geometry placement + camera
intrinsics live in their own leaves; nothing here touches the optical poses or the labels.

Conventions (frozen, from ``contract`` / ``config``):
  * Gate base colour anchors on ``contract.VQ1_GATE_RED_RGB_LINEAR`` (the extracted VQ1 orange-red,
    already sRGB->linear) when ``use_vq1_red`` and no hue jitter; otherwise we sample HSV around the
    VQ1-red hue and convert HSV->sRGB->linear in Python so the Principled ``Base Color`` (a LINEAR
    socket) reads exactly what we intend regardless of the view transform.
  * All ranges (sat/val/metallic/roughness/sun/temp/...) come straight off ``AppearanceConfig``.
  * Optical world (see contract): scene is in FRONT of the camera at world +Z; 'up' is world -Y,
    'right' is world +X.  Lighting is oriented in THIS world so shading is consistent frame to frame.

Socket-name note: the Principled BSDF v2 (Blender 4.x, OpenPBR-based) input sockets are "Base Color",
"Metallic", "Roughness", "Emission Color", "Emission Strength"; Background / Emission expose
"Color" + "Strength".  We set sockets via a name-with-fallback helper so a socket rename across a
point release degrades to a skipped tweak instead of a hard crash mid-render.

API verified against bundled RST (Blender 4.2 LTS+):
  bpy.data.materials.new(name) / .worlds.new(name) / .lights.new(name, type) /
  .objects.new(name, object_data) / .images.load(filepath, *, check_existing) ;
  Material.use_nodes + .node_tree ; World.node_tree (use_nodes deprecated-noop in 5.x, required 4.2) ;
  ShaderNodeBsdfPrincipled / ShaderNodeOutputMaterial / ShaderNodeBackground / ShaderNodeOutputWorld /
  ShaderNodeTexEnvironment(.image) / ShaderNodeTexSky(.sky_type='MULTIPLE_SCATTERING' -- NISHITA does
  NOT exist in 4.2+, enum is SINGLE_SCATTERING|MULTIPLE_SCATTERING|PREETHAM|HOSEK_WILKIE ; .sun_elevation
  /.sun_rotation/.altitude) / ShaderNodeTexNoise / ShaderNodeTexChecker ;
  SunLight.energy/.angle + AreaLight.energy/.size + Light.color (inherited) ;
  rna_enum_light_type_items ('SUN','AREA') ; mathutils.Vector.to_track_quat('-Z','Y') ;
  NodeTree.nodes.new/.clear/.remove + NodeTree.links.new(input, output) ;
  scene.collection.objects.link(obj).
"""
from __future__ import annotations

import colorsys
import math
import os

import numpy as np

try:                                    # only importable inside Blender; keep the module importable
    import bpy                          # for laptop linting / signature checks
    import mathutils
except Exception:                       # pragma: no cover - exercised only on ShadowPC
    bpy = None
    mathutils = None

from .contract import VQ1_GATE_RED_RGB, VQ1_GATE_RED_RGB_LINEAR

# Env-map file extensions we accept for HDRI lighting.
_HDRI_EXTS = (".hdr", ".exr")


# --------------------------------------------------------------------------------------------------
# small colour + socket helpers
# --------------------------------------------------------------------------------------------------
def _srgb_to_linear(c: float) -> float:
    """Single-channel sRGB (0..1) -> linear, matching contract.VQ1_GATE_RED_RGB_LINEAR's formula.

    NOTE: deliberately the SAME simplified curve (c**2.4, no 1.055 offset) as the frozen
    ``contract.VQ1_GATE_RED_RGB_LINEAR`` so a hue-jittered gate stays colour-consistent with the
    banked VQ1-red constant.  Do NOT "correct" to the exact IEC sRGB EOTF here -- that would fork
    the SSOT (the two reds would no longer match).
    """
    return c ** 2.4 if c > 0.04045 else c / 12.92


def _hsv_to_linear_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    """HSV (each 0..1, hue wraps) -> LINEAR RGB for a Principled Base Color socket.

    ``colorsys.hsv_to_rgb`` yields display-referred (sRGB) RGB; we then sRGB->linear so the
    rendered hue/sat match the sampled HSV irrespective of the scene view transform.
    """
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, float(np.clip(s, 0.0, 1.0)), float(np.clip(v, 0.0, 1.0)))
    return (_srgb_to_linear(r), _srgb_to_linear(g), _srgb_to_linear(b))


def _vq1_red_hue() -> float:
    """Hue (0..1) of the extracted VQ1 orange-red, derived from the stored sRGB triple."""
    r, g, b = (c / 255.0 for c in VQ1_GATE_RED_RGB)
    return colorsys.rgb_to_hsv(r, g, b)[0]


def _set_socket(node, names, value) -> bool:
    """Set ``node.inputs[name] = value`` for the first matching socket name; return success.

    Robust to v2 socket renames: tries each candidate name, silently no-ops if none exist.
    ``names`` may be a single str or an iterable of candidates.
    """
    if isinstance(names, str):
        names = (names,)
    for name in names:
        sock = node.inputs.get(name)
        if sock is not None:
            sock.default_value = value
            return True
    return False


def _color4(rgb) -> tuple[float, float, float, float]:
    """RGB triple -> RGBA (alpha 1) for colour sockets that expect 4 components."""
    r, g, b = rgb
    return (float(r), float(g), float(b), 1.0)


def _u(rng, lo_hi) -> float:
    """Uniform sample from an (lo, hi) range tuple (lo==hi -> the constant)."""
    lo, hi = float(lo_hi[0]), float(lo_hi[1])
    return lo if hi <= lo else float(rng.uniform(lo, hi))


def _blackbody_rgb(temp_k: float) -> tuple[float, float, float]:
    """Approximate LINEAR RGB of a blackbody at ``temp_k`` (1000..40000 K).

    Tanner Helland's piecewise sRGB approximation, then sRGB->linear, normalised so the brightest
    channel is 1 (the light's ENERGY carries intensity; colour only tints).  Avoids relying on a
    Blender blackbody node so the sun colour is deterministic and engine-independent.
    """
    t = float(np.clip(temp_k, 1000.0, 40000.0)) / 100.0
    if t <= 66.0:
        r = 255.0
        g = 99.4708025861 * math.log(t) - 161.1195681661
    else:
        r = 329.698727446 * ((t - 60.0) ** -0.1332047592)
        g = 288.1221695283 * ((t - 60.0) ** -0.0755148492)
    if t >= 66.0:
        b = 255.0
    elif t <= 19.0:
        b = 0.0
    else:
        b = 138.5177312231 * math.log(t - 10.0) - 305.0447927307
    srgb = tuple(float(np.clip(c, 0.0, 255.0)) / 255.0 for c in (r, g, b))
    lin = tuple(_srgb_to_linear(c) for c in srgb)
    m = max(lin) or 1.0
    return (lin[0] / m, lin[1] / m, lin[2] / m)


# --------------------------------------------------------------------------------------------------
# gate material
# --------------------------------------------------------------------------------------------------
def gate_material(rng, appearance) -> "bpy.types.Material":
    """Node-based Principled material for the gate ring (the detector's positive class).

    Colour: the frozen VQ1 linear red when ``use_vq1_red`` and ``gate_hue_jitter == 0``; otherwise
    HSV sampled around the VQ1-red hue (+/- ``gate_hue_jitter*0.5`` turn) with sat/val from the
    config ranges, converted HSV->sRGB->linear.  Metallic / roughness sampled from their ranges.
    With prob ``gate_emission_prob`` the gate is mildly self-lit (Emission Color = base, small
    Emission Strength) for the "lit ring" look seen in some real footage.
    """
    mat = bpy.data.materials.new(name="VQ2_Gate")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf.location = (0.0, 0.0)
    out.location = (320.0, 0.0)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    if appearance.use_vq1_red and float(appearance.gate_hue_jitter) == 0.0:
        base_lin = tuple(VQ1_GATE_RED_RGB_LINEAR)
    else:
        h0 = _vq1_red_hue()
        dh = float(appearance.gate_hue_jitter) * 0.5            # jitter fraction -> +/- turns
        h = h0 + float(rng.uniform(-dh, dh))
        s = _u(rng, appearance.gate_sat_range)
        v = _u(rng, appearance.gate_val_range)
        base_lin = _hsv_to_linear_rgb(h, s, v)

    _set_socket(bsdf, "Base Color", _color4(base_lin))
    _set_socket(bsdf, "Metallic", _u(rng, appearance.gate_metallic_range))
    _set_socket(bsdf, "Roughness", _u(rng, appearance.gate_roughness_range))

    if float(rng.random()) < float(appearance.gate_emission_prob):
        _set_socket(bsdf, "Emission Color", _color4(base_lin))
        _set_socket(bsdf, ("Emission Strength", "Emission"), float(rng.uniform(0.5, 3.0)))

    return mat


# --------------------------------------------------------------------------------------------------
# background material
# --------------------------------------------------------------------------------------------------
def background_material(rng, appearance) -> "bpy.types.Material":
    """PBR floor/wall material.  With prob ``floor_wall_texture_prob`` a procedural texture
    (Noise or Checker -- Musgrave was removed in Blender 4.1) drives Base Color + Roughness;
    otherwise a flat desaturated colour.  Kept low-saturation so it never mimics the gate red.
    """
    mat = bpy.data.materials.new(name="VQ2_Background")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf.location = (0.0, 0.0)
    out.location = (400.0, 0.0)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    # neutral-ish base: low saturation, mid value, random hue (sky/floor/wall greys & browns).
    base_lin = _hsv_to_linear_rgb(
        float(rng.random()), float(rng.uniform(0.0, 0.25)), float(rng.uniform(0.25, 0.8))
    )
    _set_socket(bsdf, "Base Color", _color4(base_lin))
    _set_socket(bsdf, "Metallic", float(rng.uniform(0.0, 0.2)))
    _set_socket(bsdf, "Roughness", float(rng.uniform(0.4, 1.0)))

    if float(rng.random()) < float(appearance.floor_wall_texture_prob):
        if float(rng.random()) < 0.5:
            tex = nt.nodes.new("ShaderNodeTexNoise")
            tex.location = (-360.0, 0.0)
            _set_socket(tex, "Scale", float(rng.uniform(2.0, 30.0)))
            _set_socket(tex, "Detail", float(rng.uniform(1.0, 8.0)))
            _set_socket(tex, "Roughness", float(rng.uniform(0.3, 0.8)))
            col_out = tex.outputs.get("Color") or tex.outputs[0]
        else:
            tex = nt.nodes.new("ShaderNodeTexChecker")
            tex.location = (-360.0, 0.0)
            _set_socket(tex, "Scale", float(rng.uniform(3.0, 25.0)))
            # tint both checker colours toward the (desaturated) base so it never reads as a gate.
            alt = _hsv_to_linear_rgb(
                float(rng.random()), float(rng.uniform(0.0, 0.2)), float(rng.uniform(0.2, 0.7))
            )
            _set_socket(tex, "Color1", _color4(base_lin))
            _set_socket(tex, "Color2", _color4(alt))
            col_out = tex.outputs.get("Color") or tex.outputs[0]

        nt.links.new(col_out, bsdf.inputs["Base Color"])
        # also feed the (scalar Fac if present, else Color) into Roughness for surface variation.
        fac_out = tex.outputs.get("Fac") or col_out
        rough_in = bsdf.inputs.get("Roughness")
        if rough_in is not None:
            nt.links.new(fac_out, rough_in)

    return mat


# --------------------------------------------------------------------------------------------------
# material assignment
# --------------------------------------------------------------------------------------------------
def apply_material(obj, material) -> None:
    """Assign ``material`` as ``obj``'s sole material (clear existing slots then append)."""
    obj.data.materials.clear()
    obj.data.materials.append(material)


# --------------------------------------------------------------------------------------------------
# lighting
# --------------------------------------------------------------------------------------------------
def _orient_to_direction(obj, direction) -> None:
    """Aim a light so its emission travels along ``direction`` (a 3-vector, optical world).

    A Blender light emits along its local -Z; ``Vector.to_track_quat('-Z','Y')`` builds the
    rotation whose -Z maps onto ``direction``.  We set ``rotation_mode='QUATERNION'`` then the
    quaternion directly (avoids gimbal surprises from euler ordering).
    """
    v = mathutils.Vector((float(direction[0]), float(direction[1]), float(direction[2])))
    if v.length < 1e-9:
        v = mathutils.Vector((0.0, 1.0, 0.0))                  # arbitrary fallback
    quat = v.to_track_quat("-Z", "Y")
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = quat


def _sun_direction(elevation_deg: float, azimuth_deg: float) -> np.ndarray:
    """Unit *travel* direction of sun light in the OPTICAL world (up = world -Y, ground plane =
    world X-Z, scene downrange = +Z).  ``elevation`` measured from the horizon; the light shines
    DOWNWARD, so the travel direction has a +Y (toward world-down) component.
    """
    el = math.radians(float(elevation_deg))
    az = math.radians(float(azimuth_deg))
    # source position on the upper hemisphere (up = -Y): horizontal spread in the X-Z plane.
    sx = math.cos(el) * math.cos(az)
    sz = math.cos(el) * math.sin(az)
    sy = -math.sin(el)                                          # up is -Y -> source above is -Y
    src = np.array([sx, sy, sz], dtype=np.float64)
    d = -src                                                    # travels from source toward scene
    n = float(np.linalg.norm(d))
    return d / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])


def _resolve_lighting_mode(rng, appearance) -> str:
    """Resolve appearance.lighting_mode; 'auto' picks per frame, biased to the DARK ARENA (the real
    A2RL x DCL venue) over daylight."""
    mode = getattr(appearance, "lighting_mode", "daylight")
    if mode == "auto":
        return "dark_arena" if float(rng.random()) < 0.65 else "daylight"
    return mode


def _add_daylight(rng, appearance) -> list:
    """A SUN (elevation/azimuth/energy/blackbody colour) + a 50%-chance soft AREA fill."""
    objs = []
    elev = _u(rng, appearance.sun_elevation_range_deg)
    azim = float(rng.uniform(0.0, 360.0))
    energy = _u(rng, appearance.sun_intensity_range)

    sun_data = bpy.data.lights.new(name="VQ2_Sun", type="SUN")
    sun_data.energy = float(energy)
    sun_data.color = _blackbody_rgb(_u(rng, appearance.color_temp_range_k))
    sun_data.angle = float(rng.uniform(math.radians(0.5), math.radians(5.0)))   # soft-ish shadows
    sun_obj = bpy.data.objects.new(name="VQ2_Sun", object_data=sun_data)
    _orient_to_direction(sun_obj, _sun_direction(elev, azim))
    bpy.context.scene.collection.objects.link(sun_obj)
    objs.append(sun_obj)

    if float(rng.random()) < 0.5:
        fill_data = bpy.data.lights.new(name="VQ2_Fill", type="AREA")
        fill_data.energy = float(energy) * float(rng.uniform(8.0, 40.0))
        fill_data.size = float(rng.uniform(3.0, 10.0))
        fill_data.color = _blackbody_rgb(_u(rng, appearance.color_temp_range_k))
        fill_obj = bpy.data.objects.new(name="VQ2_Fill", object_data=fill_data)
        side = 1.0 if rng.random() < 0.5 else -1.0
        pos = np.array([side * float(rng.uniform(4.0, 10.0)),
                        -float(rng.uniform(1.0, 4.0)),
                        float(rng.uniform(4.0, 12.0))], dtype=np.float64)
        fill_obj.location = (float(pos[0]), float(pos[1]), float(pos[2]))
        _orient_to_direction(fill_obj, -pos)
        bpy.context.scene.collection.objects.link(fill_obj)
        objs.append(fill_obj)
    return objs


def _add_dark_arena(rng, appearance) -> list:
    """DARK ARENA: N hard SPOT lights (white or coloured -- DCL stage lighting) above/around the
    scene, aimed down-course; low world ambient is set in setup_world. The dominant real-venue look
    (dark hall + bright spots, high dynamic range)."""
    objs = []
    lo, hi = getattr(appearance, "n_spotlights_range", (1, 3))
    n = int(rng.integers(int(lo), int(hi) + 1)) if int(hi) >= int(lo) else int(lo)
    for k in range(max(1, n)):
        data = bpy.data.lights.new(name=f"VQ2_Spot_{k}", type="SPOT")
        data.energy = _u(rng, getattr(appearance, "spotlight_energy_range", (200.0, 3000.0)))
        if float(rng.random()) < float(getattr(appearance, "colored_light_prob", 0.4)):
            data.color = _hsv_to_linear_rgb(float(rng.random()), float(rng.uniform(0.5, 1.0)), 1.0)
        else:
            data.color = _blackbody_rgb(_u(rng, appearance.color_temp_range_k))
        try:                                                    # SPOT-specific cone controls (guarded)
            data.spot_size = float(rng.uniform(math.radians(20.0), math.radians(80.0)))
            data.spot_blend = float(rng.uniform(0.1, 0.6))
        except (AttributeError, TypeError):
            pass
        obj = bpy.data.objects.new(name=f"VQ2_Spot_{k}", object_data=data)
        # place above (up = -Y) + spread in X-Z, in front of the camera; aim at a point down-course.
        pos = np.array([float(rng.uniform(-10.0, 10.0)),
                        -float(rng.uniform(3.0, 9.0)),
                        float(rng.uniform(3.0, 25.0))], dtype=np.float64)
        target = np.array([float(rng.uniform(-3.0, 3.0)), float(rng.uniform(0.0, 3.0)),
                           float(rng.uniform(4.0, 28.0))], dtype=np.float64)
        obj.location = (float(pos[0]), float(pos[1]), float(pos[2]))
        _orient_to_direction(obj, target - pos)
        bpy.context.scene.collection.objects.link(obj)
        objs.append(obj)
    return objs


def add_lighting(rng, appearance) -> "list[bpy.types.Object]":
    """Create the frame's lights per ``appearance.lighting_mode`` (daylight sun/fill OR a dark-arena
    spotlight rig; 'auto' biases to dark arena). Returns the created light OBJECTS."""
    if _resolve_lighting_mode(rng, appearance) == "dark_arena":
        return _add_dark_arena(rng, appearance)
    return _add_daylight(rng, appearance)


# --------------------------------------------------------------------------------------------------
# world / environment lighting
# --------------------------------------------------------------------------------------------------
def _world_output(nt):
    """Return (existing or new) World Output node for a world node tree."""
    for n in nt.nodes:
        if n.bl_idname == "ShaderNodeOutputWorld":
            return n
    return nt.nodes.new("ShaderNodeOutputWorld")


def _list_hdris(hdri_dir):
    """Return absolute paths of .hdr/.exr files in ``hdri_dir`` (empty if missing/not a dir)."""
    if not hdri_dir or not os.path.isdir(hdri_dir):
        return []
    out = []
    for fn in sorted(os.listdir(hdri_dir)):
        if fn.lower().endswith(_HDRI_EXTS):
            out.append(os.path.join(hdri_dir, fn))
    return out


def setup_world(rng, appearance) -> None:
    """Configure the scene world node tree for per-frame-varied ambient light.

    Priority: a random HDRI (``use_hdri`` and ``hdri_dir`` has env maps) -> Environment Texture into
    Background; else a procedural Sky (``ShaderNodeTexSky``, MULTIPLE_SCATTERING -- 'NISHITA' does
    not exist in this Blender) into Background; else a flat coloured Background.  In every branch the
    Background ``Strength`` is randomised so ambient brightness differs frame to frame.
    """
    scene = bpy.context.scene
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("VQ2_World")
        scene.world = world
    world.use_nodes = True       # required on 4.2 LTS; deprecated no-op (always True) on 5.x
    nt = world.node_tree
    nt.nodes.clear()

    bg = nt.nodes.new("ShaderNodeBackground")
    out = _world_output(nt)
    bg.location = (0.0, 0.0)
    out.location = (300.0, 0.0)
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])

    # ambient EV varies per frame; ambient_strength_range lets a dark-arena preset keep it LOW so
    # the spotlights carry the scene (high dynamic range), vs a bright daylight/HDRI fill.
    amb_lo, amb_hi = getattr(appearance, "ambient_strength_range", (0.2, 2.0))
    _set_socket(bg, "Strength", float(rng.uniform(float(amb_lo), float(amb_hi))))

    hdris = _list_hdris(appearance.hdri_dir) if appearance.use_hdri else []
    if hdris:
        env = nt.nodes.new("ShaderNodeTexEnvironment")
        env.location = (-360.0, 0.0)
        path = hdris[int(rng.integers(0, len(hdris)))]
        try:
            env.image = bpy.data.images.load(path, check_existing=True)
            nt.links.new(env.outputs["Color"], bg.inputs["Color"])
            return
        except Exception:
            nt.nodes.remove(env)                               # fall through to procedural sky

    # procedural sky (preferred) -> background colour.
    try:
        sky = nt.nodes.new("ShaderNodeTexSky")
        sky.location = (-360.0, 0.0)
        sky.sky_type = "MULTIPLE_SCATTERING"
        sky.sun_elevation = math.radians(_u(rng, appearance.sun_elevation_range_deg))
        sky.sun_rotation = float(rng.uniform(0.0, 2.0 * math.pi))
        sky.altitude = float(rng.uniform(0.0, 2000.0))
        nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
        return
    except Exception:
        pass

    # last resort: flat sky-ish colour.
    sky_lin = _hsv_to_linear_rgb(
        float(rng.uniform(0.5, 0.65)), float(rng.uniform(0.05, 0.4)), float(rng.uniform(0.4, 0.9))
    )
    _set_socket(bg, "Color", _color4(sky_lin))
