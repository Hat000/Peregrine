"""Render-engine config, motion blur, glare/exposure + render-to-array (THE bpy leaf).

Runs ONLY inside Blender (``blender --background --python``); ``import bpy`` is fine because
this module is never imported by the laptop pytest suite (the pure-Python core owns the
testable logic). Targets Blender 4.2 LTS or newer: engine ids ``CYCLES`` /
``BLENDER_EEVEE_NEXT``, Principled BSDF v2 sockets, and the compositor Glare node. Every bpy
attribute referenced here was checked against the bundled API RST -- where a name shifted
across 4.2->5.x (``scene.use_nodes``/``scene.node_tree`` vs ``scene.compositing_node_group``;
the legacy ``action.fcurves`` vs the 4.4+ layered ``action.layers[*].strips[*].channelbag``;
the Glare node controls moving from node-props to input-sockets) the access is wrapped so it
works on either, with a ``# verified:`` note on the load-bearing identifier.

Exposure/gamma go on ``scene.view_settings`` (ColorManagedViewSettings.exposure is "stops
applied before the display transform, multiplying by 2^exposure"; .gamma is post-transform) --
NOT a Cycles-film attribute, which is the easy mistake. Motion blur is engine-agnostic at
``scene.render.use_motion_blur`` + ``.motion_blur_shutter``; EEVEE_NEXT also needs its
``scene.eevee.motion_blur_steps`` for a multi-sampled streak.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

try:  # only present inside Blender; guarded so static analysis on the laptop doesn't choke
    import bpy
    import mathutils
except Exception:  # pragma: no cover - never hit inside Blender
    bpy = None
    mathutils = None

from .config import AppearanceConfig, RenderConfig


# --- GPU enable (Cycles addon prefs; addon RST is not bundled, pattern is the documented one) -
def _enable_cycles_gpu(scene) -> bool:
    """Switch Cycles to GPU: pick CUDA (else OPTIX), refresh + enable every matching device,
    set ``scene.cycles.device='GPU'``. Returns True if at least one GPU device was enabled.

    The cycles addon preferences (``compute_device_type`` enum + ``get_devices_for_type`` /
    ``get_devices`` + per-device ``.use``) live in the bundled addon's CyclesPreferences class,
    NOT the core ``bpy.types`` RST, so this is wrapped defensively: on a box without a
    CUDA/OptiX GPU it falls back to CPU instead of aborting the whole render job.
    """
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
    except Exception:
        return False
    enabled_any = False
    for dev_type in ("CUDA", "OPTIX"):  # CUDA first: broadest support on the ShadowPC GPU
        try:
            prefs.compute_device_type = dev_type
        except (TypeError, AttributeError):
            continue  # this backend not compiled in / not available
        # Refresh the device list for the chosen backend, then flag every non-CPU device on.
        try:
            devices = prefs.get_devices_for_type(dev_type)  # 4.x+; returns the typed devices
        except Exception:
            try:
                prefs.get_devices()  # older signature: populates prefs.devices in place
            except Exception:
                devices = None
            devices = list(getattr(prefs, "devices", []) or [])
        for dev in devices or []:
            if getattr(dev, "type", "CPU") != "CPU":
                dev.use = True
                enabled_any = True
            else:
                dev.use = False  # leave CPU off so the GPU isn't bottlenecked by it
        if enabled_any:
            break
    if enabled_any:
        scene.cycles.device = "GPU"  # scene.cycles is the cycles-addon CyclesRenderSettings
    return enabled_any


def _set_view_exposure_gamma(scene, appearance: AppearanceConfig, rng: np.random.Generator) -> None:
    """Sample film exposure (EV) + gamma and apply on the colour-managed view transform.

    ``scene.view_settings.exposure`` (verified: bpy.types.ColorManagedViewSettings.exposure) is
    the stops multiplier applied before the display transform; ``.gamma`` (verified:
    ColorManagedViewSettings.gamma) is the post-transform encoding. Both are per-frame
    domain-randomization knobs from AppearanceConfig.
    """
    vs = scene.view_settings  # verified: bpy.types.Scene.view_settings -> ColorManagedViewSettings
    lo, hi = appearance.exposure_range
    vs.exposure = float(rng.uniform(lo, hi))
    glo, ghi = appearance.gamma_range
    vs.gamma = float(rng.uniform(glo, ghi))


def _build_glare_compositor(scene, glare_type: str = "FOG_GLOW") -> None:
    """Insert a Glare node between Render Layers and Composite for bloom/lens flare.

    Compositor access changed across versions:
      * 4.2:  ``scene.use_nodes = True`` auto-created ``scene.node_tree``.
      * 5.x:  ``scene.use_nodes`` is deprecated (set is a no-op / get always returns True) and
              ``scene.node_tree`` is GONE; the tree lives at ``scene.compositing_node_group``
              and compositing is toggled via ``scene.render.use_compositing`` (verified:
              bpy.types.Scene.compositing_node_group, bpy.types.RenderSettings.use_compositing).
    :func:`_ensure_compositor_tree` tries the modern slot first and creates the tree if absent,
    so this works on either line.

    The Glare node id is ``CompositorNodeGlare`` (verified: bpy.types.CompositorNodeGlare). Its
    ``glare_type`` / ``mix`` / ``threshold`` / ``quality`` controls are node-properties in 4.2
    but several became input sockets in the 5.x rewrite, so we set them through a helper that
    tries the attribute then the named input socket and stays silent on a miss.

    The OUTPUT node differs by line: 4.2 ends the tree on ``CompositorNodeComposite``; Blender 5.x
    REMOVED that node (``CompositorNodeComposite undefined``) -- a compositor node-GROUP ends on a
    ``NodeGroupOutput`` instead. We try Composite, then fall back to the group output. Glare/bloom is
    a non-load-bearing nicety (it never touches geometry or labels), so the ENTIRE build is wrapped:
    any failure on an unexpected build degrades to "no glare" (compositing left off) rather than
    aborting the render -- as the module contract above promises.
    """
    try:
        tree = _ensure_compositor_tree(scene)
        if tree is None:
            return
        nodes, links = tree.nodes, tree.links  # verified: NodeTree.nodes (Nodes), NodeTree.links

        rlayers = _find_or_new(nodes, "CompositorNodeRLayers")
        # Output node: Composite (<=4.x) or, on 5.x where it is undefined, the group output.
        out_node = _find_or_new(nodes, "CompositorNodeComposite") or _find_or_new(nodes, "NodeGroupOutput")
        glare = nodes.new("CompositorNodeGlare")  # verified: Nodes.new(type) wants the bl_idname
        glare.location = (300.0, 0.0)

        _set_glare_param(glare, "glare_type", glare_type)  # 'FOG_GLOW'|'GHOSTS'|'STREAKS'|'BLOOM'
        _set_glare_param(glare, "mix", 0.0)        # 0 = balanced glare+image
        _set_glare_param(glare, "threshold", 1.0)  # only pixels brighter than 1.0 bloom
        _set_glare_param(glare, "quality", "MEDIUM")

        # Render Layers[Image] -> Glare[in 0] -> output[in 0]. Use index 0 sockets: the image socket
        # is first on RLayers/Glare; on a NodeGroupOutput the first input is created on first link.
        if out_node is not None and rlayers is not None and glare is not None:
            links.new(rlayers.outputs[0], glare.inputs[0])   # verified: NodeLinks.new (output, input)
            if out_node.inputs:
                links.new(glare.outputs[0], out_node.inputs[0])
    except Exception:
        # Unsupported compositor layout on this build: turn compositing back off so the render runs
        # clean (no bloom) instead of failing. Bloom is cosmetic; labels/geometry are unaffected.
        try:
            scene.render.use_compositing = False
        except Exception:
            pass


def _ensure_compositor_tree(scene):
    """Return the scene's compositor NodeTree, creating/enabling it, version-tolerant."""
    # Make sure compositing actually runs on the result.
    try:
        scene.render.use_compositing = True  # verified: RenderSettings.use_compositing
    except Exception:
        pass
    # Modern slot first (4.x/5.x): verified bpy.types.Scene.compositing_node_group.
    tree = getattr(scene, "compositing_node_group", None)
    if tree is None:
        # Classic <=4.x path: toggling use_nodes auto-created scene.node_tree. On 5.x use_nodes
        # is a deprecated no-op and scene.node_tree no longer exists, so both getattr's miss
        # harmlessly and we fall through to building the tree explicitly below.
        try:
            scene.use_nodes = True  # verified: Scene.use_nodes (deprecated no-op in 5.x)
        except Exception:
            pass
        tree = getattr(scene, "node_tree", None)
    if tree is None:
        # Neither slot populated (5.x with no tree yet): build a compositor tree and attach it.
        # verified: bpy.types.BlendDataNodeTrees.new(name, type='CompositorNodeTree').
        try:
            tree = bpy.data.node_groups.new("VQ2_Compositor", "CompositorNodeTree")
            scene.compositing_node_group = tree
        except Exception:
            return None
    return tree


def _find_or_new(nodes, bl_idname: str):
    """Reuse an existing node of ``bl_idname`` (e.g. the default RLayers/Composite) or add one.
    Returns None if the type is undefined on this build (e.g. CompositorNodeComposite was removed in
    5.x) so the caller can fall back to an alternative output node instead of crashing."""
    for n in nodes:
        if n.bl_idname == bl_idname:
            return n
    try:
        return nodes.new(bl_idname)
    except (RuntimeError, ValueError):
        return None


def _set_glare_param(node, name: str, value) -> None:
    """Set a Glare control by node-attribute (4.2) OR by named input socket (5.x), whichever
    exists. Silent on miss so a renamed control never aborts the render."""
    if hasattr(node, name):
        try:
            setattr(node, name, value)
            return
        except (TypeError, AttributeError, ValueError):
            pass
    try:
        sock = node.inputs.get(name.replace("_", " ").title()) or node.inputs.get(name)
        if sock is not None:
            sock.default_value = value
    except Exception:
        pass


# --- public API (signatures FROZEN; the composition backend depends on them) ------------------
def _resolve_engine(scene, requested: str) -> str:
    """Map a requested render-engine id onto one the RUNNING Blender actually exposes.

    The EEVEE engine id moved across releases: Blender 4.2 used ``BLENDER_EEVEE_NEXT``; 4.3+/5.x
    renamed it back to ``BLENDER_EEVEE`` (the enum on 5.1 is CYCLES / BLENDER_EEVEE / BLENDER_WORKBENCH,
    with NO _NEXT). A preset or the ``--eevee`` shortcut may carry either spelling, so we look up the
    live ``scene.render.engine`` enum and, if the requested id is absent but is an EEVEE variant, swap
    to whichever EEVEE id this build offers. CYCLES (stable across versions) and a genuinely unknown
    engine pass through unchanged (the unknown one then fails loudly -- correct).
    """
    try:
        valid = {it.identifier for it in scene.render.bl_rna.properties["engine"].enum_items}
    except Exception:
        return requested
    if requested in valid:
        return requested
    if "EEVEE" in requested.upper():
        for cand in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
            if cand in valid:
                return cand
    return requested


def configure_render(scene, render_cfg: RenderConfig, appearance: AppearanceConfig, motion=None) -> None:
    """Apply engine + samples + GPU + denoise + exposure/gamma + motion blur + glare to ``scene``.

    ``motion`` is an optional ``np.random.Generator`` used only to sample exposure/gamma from
    their ranges (passing the per-frame rng keeps the dataset reproducible); None -> a default
    rng. It is NOT the motion-blur driver -- that is :func:`set_motion_blur`, called per frame
    by the composition backend once the geometry is placed.
    """
    rng = motion if isinstance(motion, np.random.Generator) else np.random.default_rng()

    engine = _resolve_engine(scene, render_cfg.engine)  # tolerate EEVEE_NEXT<->EEVEE id drift across versions
    scene.render.engine = engine  # verified: RenderSettings.engine ('CYCLES'/'BLENDER_EEVEE[_NEXT]')
    scene.render.film_transparent = bool(render_cfg.film_transparent)  # verified: RenderSettings.film_transparent

    if engine == "CYCLES":
        cy = scene.cycles  # CyclesRenderSettings (cycles addon)
        cy.samples = int(render_cfg.samples)         # path-trace samples per pixel
        cy.use_denoising = bool(render_cfg.use_denoise)
        if render_cfg.use_gpu:
            if not _enable_cycles_gpu(scene):
                cy.device = "CPU"  # no GPU available -> render on CPU rather than fail
        else:
            cy.device = "CPU"
    else:  # EEVEE_NEXT (or legacy EEVEE): map "samples" onto the TAA render samples.
        try:
            scene.eevee.taa_render_samples = int(render_cfg.samples)  # verified: SceneEEVEE.taa_render_samples
        except Exception:
            pass

    _set_view_exposure_gamma(scene, appearance, rng)

    # Motion blur: engine-agnostic master switch + shutter on RenderSettings; EEVEE needs steps.
    scene.render.use_motion_blur = bool(render_cfg.motion_blur)        # verified: RenderSettings.use_motion_blur
    scene.render.motion_blur_shutter = float(render_cfg.motion_blur_shutter)  # verified: RenderSettings.motion_blur_shutter
    # Keep the shutter centred on the labelled frame so the MIDPOINT is unsmeared (see
    # set_motion_blur). 'CENTER' is the RenderSettings default but we pin it explicitly.
    try:
        scene.render.motion_blur_position = "CENTER"  # verified: RenderSettings.motion_blur_position
    except Exception:
        pass
    if engine != "CYCLES" and render_cfg.motion_blur:
        try:
            scene.eevee.motion_blur_steps = 4  # verified: SceneEEVEE.motion_blur_steps (>1 => real streak)
        except Exception:
            pass

    if render_cfg.use_glare:
        _build_glare_compositor(scene)


def set_motion_blur(scene, moving_objects, cam_obj, v_optical, shutter) -> None:
    """Keyframe the scene's translation across the exposure so the render streaks while the
    LABELLED midpoint frame stays geometrically exact.

    Approach: the drone moves at body velocity ``v_optical`` (expressed in the camera OPTICAL
    frame, a 3-vector, m/s). In our optical-frame scene the camera is fixed at the origin, so we
    induce the relative motion by translating every gate + background object by ``-v_optical*dt``
    (scene moves opposite the camera). We anchor the central integer frame F at the labelled pose
    and add two flanking keyframes at F +/- 1 frame offset by ``-/+ v_optical*(shutter/2)/fps``
    -- linear interpolation through F reproduces the labelled placement exactly at the shutter
    centre (motion_blur_position='CENTER'), while Cycles/EEVEE integrate the streak between the
    flanks. Distance uses ``dt = (shutter * 0.5) / fps`` so ``shutter`` is in FRAMES (its
    RenderSettings unit). No-op when motion blur is off or velocity is ~0.
    """
    if not scene.render.use_motion_blur:
        return
    v = np.asarray(v_optical, dtype=float).reshape(3)
    if not np.isfinite(v).all() or float(np.linalg.norm(v)) < 1e-6:
        return

    fps = float(scene.render.fps) / float(getattr(scene.render, "fps_base", 1.0) or 1.0)
    dt = (float(shutter) * 0.5) / max(fps, 1e-6)        # seconds from centre to each flank
    dpos = v * dt                                       # metres the SCENE shifts by -dpos at +flank

    f0 = int(scene.frame_current)                       # labelled (midpoint) frame
    scene.frame_set(f0)                                 # verified: Scene.frame_set(frame)

    objs = [o for o in (moving_objects or []) if o is not None and o is not cam_obj]
    for obj in objs:
        base = mathutils.Vector(obj.location)           # placement at the labelled pose
        # Centre keyframe: exact labelled position.
        obj.location = base
        obj.keyframe_insert(data_path="location", frame=f0)            # verified: bpy_struct.keyframe_insert
        # Leading flank (shutter opens): scene was -dpos behind.
        obj.location = base - mathutils.Vector(dpos.tolist())
        obj.keyframe_insert(data_path="location", frame=f0 - 1)
        # Trailing flank (shutter closes): scene is +dpos ahead.
        obj.location = base + mathutils.Vector(dpos.tolist())
        obj.keyframe_insert(data_path="location", frame=f0 + 1)
        obj.location = base                             # restore so a no-blur read is correct

    # Linear F-curve interpolation makes the streak straight + the centre exact.
    _force_linear_interpolation(objs)


def _iter_location_fcurves(act):
    """Yield every ``location`` F-Curve of an action, across Blender's two animation models.

    Legacy (<=4.2): ``act.fcurves`` is a flat collection. Layered (4.4+/5.x, verified:
    bpy.types.Action.layers -> ActionLayer.strips -> ActionStrip.channelbag(slot).fcurves --
    ``Action.fcurves`` was REMOVED there). We try the legacy attribute first, then walk the
    layered structure; both are wrapped so a model mismatch never raises.
    """
    # Legacy flat collection (still present on 4.2 LTS legacy actions).
    legacy = getattr(act, "fcurves", None)
    if legacy is not None:
        try:
            for fc in legacy:
                yield fc
            return
        except (TypeError, AttributeError):
            pass
    # Layered model (4.4+): action.layers[*].strips[*].channelbag(slot).fcurves.
    try:
        slots = list(getattr(act, "slots", []) or [])
        for layer in getattr(act, "layers", []) or []:
            for strip in getattr(layer, "strips", []) or []:
                cbag_fn = getattr(strip, "channelbag", None)
                if cbag_fn is None:
                    continue
                for slot in slots:
                    try:
                        cbag = cbag_fn(slot)
                    except (TypeError, RuntimeError):
                        cbag = None
                    if cbag is None:
                        continue
                    for fc in getattr(cbag, "fcurves", []) or []:
                        yield fc
    except Exception:
        return


def _force_linear_interpolation(objs) -> None:
    """Set every inserted location keyframe to LINEAR so the motion across the shutter is a
    straight, constant-velocity streak (Bezier would curve + shift the centre).

    Works on both the legacy (<=4.2 ``action.fcurves``) and the layered (4.4+ channelbag)
    animation models via :func:`_iter_location_fcurves`; a nicety, so it stays fully defensive
    and never aborts the render if the action layout is unexpected. ``Keyframe.interpolation``
    is verified: bpy.types.Keyframe.interpolation (enum incl. 'LINEAR').
    """
    for obj in objs:
        ad = getattr(obj, "animation_data", None)
        act = getattr(ad, "action", None) if ad else None
        if act is None:
            continue
        try:
            for fc in _iter_location_fcurves(act):
                if getattr(fc, "data_path", None) != "location":
                    continue
                for kp in fc.keyframe_points:
                    kp.interpolation = "LINEAR"
        except Exception:
            continue


def render_to_bgr(scene, tmp_path=None) -> np.ndarray:
    """Render the scene to a PNG then read it back as an (H, W, 3) uint8 BGR array.

    The robust ``--background`` path: write a real PNG (so colour-management + compositing are
    applied exactly as on disk) and ``cv2.imread`` it -- which already returns BGR uint8 in
    (H, W, 3) -- rather than fishing in ``bpy.data.images['Render Result'].pixels`` (which is
    float RGBA, bottom-up, and unreliable headless). The temp file is removed unless the caller
    supplied ``tmp_path`` (then it is left for them to manage).
    """
    import cv2

    owns_tmp = tmp_path is None
    if owns_tmp:
        fd, tmp_path = tempfile.mkstemp(suffix=".png", prefix="vq2_render_")
        os.close(fd)

    scene.render.image_settings.file_format = "PNG"  # verified: ImageFormatSettings.file_format ('PNG')
    scene.render.image_settings.color_mode = "RGB"   # verified: ImageFormatSettings.color_mode -> 3 chan
    scene.render.filepath = tmp_path                  # verified: RenderSettings.filepath
    try:
        bpy.ops.render.render(write_still=True)       # verified: bpy.ops.render.render(write_still)
        img = cv2.imread(tmp_path, cv2.IMREAD_COLOR)  # BGR uint8, (H, W, 3)
        if img is None:
            raise RuntimeError(f"render produced no readable image at {tmp_path}")
        return img
    finally:
        if owns_tmp:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
