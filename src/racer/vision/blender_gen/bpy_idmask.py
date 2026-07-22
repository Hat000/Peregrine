"""Per-object ID render pass -- the gate's TRUE projected silhouette, read straight out of Blender.

WHY THIS EXISTS. The seg labels used to be two FLAT QUADS (the 1.5 m inner square and the 2.72 m
outer square) projected and clipped. The real gate is a 0.26 m deep prism. Head-on the two agree;
at 2 m and off-axis they do not, because the camera sees the gate's INNER SIDE WALLS: the true
silhouette is the union of the front face, the back face and the connecting walls, its outer
boundary is BIGGER than either flat square and its see-through hole is SMALLER than the projected
inner square. Every close-range synthetic label was therefore systematically wrong in exactly the
regime the segmentation path exists to fix. Blender already knows the answer; this module asks it.

HOW. Not the compositor. Blender 5.x moved the compositor to ``scene.compositing_node_group``,
deleted ``CompositorNodeComposite``, and reading a named pass out of ``Render Result`` headless is
famously unreliable (see the note in ``bpy_render.render_to_bgr``). Instead we take a second,
throwaway WORKBENCH render of the same scene at the same camera, with every object flat-shaded in
its own ``object.color``. Measured on this box (Blender 5.1.2, probe in the commit message): the
output contains EXACTLY the palette colours and nothing else -- no anti-aliasing, no dither, no
world background -- so decoding is a per-channel threshold and cannot mis-assign a pixel.

The five settings that make that true are each load-bearing; if any is dropped the pass silently
starts producing blended edge pixels that decode to the WRONG gate:
  * ``scene.display.render_aa = 'OFF'``  -- otherwise edge pixels blend red+green -> yellow, which
    is a different, valid palette id (a 1 px sliver labelled as a gate that is not there).
  * ``scene.render.dither_intensity = 0`` -- Blender dithers PNG output by default; measured +/-1
    LSB noise on flat colour, harmless under a threshold but pointless risk.
  * ``view_transform = 'Standard'`` + ``look='None'`` + exposure 0 / gamma 1 -- the photoreal path
    leaves AgX on, which renders pure red as RGB(219,57,33). Still decodable, but only by luck.
  * ``film_transparent = True`` + RGBA -- Workbench does NOT draw the world (verified with a 3x
    strength HDRI), but the fallback viewport grey is 0x40 on every channel; transparency gives an
    unambiguous "no object here" instead of a colour we have to reason about.
  * ``use_motion_blur = False`` + ``use_compositing = False`` -- a streaked or bloomed id is not an
    id. (The photoreal path already turns both off for the beauty render; the legacy path does not.)

PALETTE. The 7 non-black corners of the RGB cube. Decoding is ``channel >= 128`` per channel, so
the result is immune to any residual colour management, gamma or 8-bit quantisation. Black is
reserved for "some other object" (occluders) and transparent for "nothing", so a scene may hold
arbitrarily many objects while only 7 are identified per pass; callers batch.

This module runs ONLY inside Blender and is imported lazily by the Blender backend, exactly like
the other ``bpy_*`` leaves; the laptop test-suite never touches it.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

try:                                    # only importable inside Blender
    import bpy
except Exception:                       # pragma: no cover - exercised only on ShadowPC
    bpy = None


# The 7 non-black corners of the RGB cube, in a fixed order. Index i -> palette[i]; the decoded
# 3-bit code is (R>=128) | (G>=128)<<1 | (B>=128)<<2, so code(i) is just the bit pattern.
PALETTE: tuple[tuple[float, float, float], ...] = (
    (1.0, 0.0, 0.0),   # code 1
    (0.0, 1.0, 0.0),   # code 2
    (0.0, 0.0, 1.0),   # code 4
    (1.0, 1.0, 0.0),   # code 3
    (1.0, 0.0, 1.0),   # code 5
    (0.0, 1.0, 1.0),   # code 6
    (1.0, 1.0, 1.0),   # code 7
)
MAX_IDS_PER_PASS = len(PALETTE)

# Every pixel of an OPAQUE object should sit hard against 0 or 255 on each channel. Anything in
# between means anti-aliasing / dither / a tone-map crept back in and the decode is no longer
# trustworthy -- we do not silently continue, we say so (once).
_AMBIGUOUS_LO, _AMBIGUOUS_HI = 32, 223
_ambiguity_warned = False


def _code_of(idx: int) -> int:
    r, g, b = PALETTE[idx]
    return int(r) | (int(g) << 1) | (int(b) << 2)


# --------------------------------------------------------------------------------------------
# scene-settings snapshot / restore
# --------------------------------------------------------------------------------------------
# (dotted-path, attribute). Anything we touch has to come back, byte for byte: the beauty render
# of the NEXT frame reuses the same scene, and a leaked 'BLENDER_WORKBENCH' or a leaked flat view
# transform would quietly turn the whole dataset into grey mush.
_SAVED = (
    ("render", "engine"),
    ("render", "film_transparent"),
    ("render", "use_compositing"),
    ("render", "use_motion_blur"),
    ("render", "dither_intensity"),
    ("render", "filepath"),
    ("render.image_settings", "file_format"),
    ("render.image_settings", "color_mode"),
    ("view_settings", "view_transform"),
    ("view_settings", "look"),
    ("view_settings", "exposure"),
    ("view_settings", "gamma"),
    ("display", "render_aa"),
    ("display.shading", "light"),
    ("display.shading", "color_type"),
    ("display.shading", "show_object_outline"),
    ("display.shading", "show_specular_highlight"),
    ("display.shading", "show_shadows"),
    ("display.shading", "show_cavity"),
    ("display.shading", "show_xray"),
    ("display.shading", "show_backface_culling"),
    ("display.shading", "background_type"),
)


def _resolve(scene, path: str):
    obj = scene
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _snapshot(scene) -> dict:
    out = {}
    for path, attr in _SAVED:
        holder = _resolve(scene, path)
        if holder is None:
            continue
        try:
            out[(path, attr)] = getattr(holder, attr)
        except AttributeError:
            pass
    return out


def _restore(scene, snap: dict) -> None:
    for (path, attr), value in snap.items():
        holder = _resolve(scene, path)
        if holder is None:
            continue
        try:
            setattr(holder, attr, value)
        except Exception:
            pass


def _configure_id_pass(scene) -> None:
    """Put the scene into the flat, deterministic id-render state documented at the top."""
    scene.render.engine = "BLENDER_WORKBENCH"      # verified present on 5.1.2 (settable + readback)
    sh = scene.display.shading
    for attr, value in (("light", "FLAT"), ("color_type", "OBJECT"),
                        ("show_object_outline", False), ("show_specular_highlight", False),
                        ("show_shadows", False), ("show_cavity", False), ("show_xray", False),
                        ("show_backface_culling", False), ("background_type", "VIEWPORT")):
        try:
            setattr(sh, attr, value)
        except Exception:
            pass   # a control renamed on some build is not worth aborting the render for
    try:
        scene.display.render_aa = "OFF"            # see module docstring: AA -> wrong-id slivers
    except Exception:
        pass
    scene.render.dither_intensity = 0.0
    scene.render.film_transparent = True           # background => alpha 0, never a palette colour
    scene.render.use_compositing = False
    scene.render.use_motion_blur = False
    for attr, value in (("view_transform", "Standard"), ("look", "None"),
                        ("exposure", 0.0), ("gamma", 1.0)):
        try:
            setattr(scene.view_settings, attr, value)
        except Exception:
            pass
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"


def _render_rgba(scene) -> np.ndarray:
    """Render the configured scene and read it back as (H, W, 4) uint8 BGRA (cv2 order)."""
    import cv2

    fd, tmp = tempfile.mkstemp(suffix=".png", prefix="vq2_idpass_")
    os.close(fd)
    scene.render.filepath = tmp
    try:
        bpy.ops.render.render(write_still=True)
        img = cv2.imread(tmp, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"id pass produced no readable image at {tmp}")
        if img.ndim == 3 and img.shape[2] == 3:    # film_transparent unsupported -> synth alpha
            img = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
        return img
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def _decode(img: np.ndarray, n_targets: int) -> list[np.ndarray]:
    """BGRA id render -> one boolean mask per target index (0..n_targets-1)."""
    global _ambiguity_warned
    b, g, r, a = img[..., 0], img[..., 1], img[..., 2], img[..., 3]
    opaque = a >= 128
    code = ((r >= 128).astype(np.uint8)
            | ((g >= 128).astype(np.uint8) << 1)
            | ((b >= 128).astype(np.uint8) << 2))
    if not _ambiguity_warned:
        # A pixel of an opaque object that is neither near-0 nor near-255 on some channel means the
        # "no AA / no dither / no tone-map" guarantee broke. Report ONCE, loudly, and keep going --
        # the threshold decode still assigns every pixel, it just may be a pixel or two off.
        mid = ((img[..., :3] > _AMBIGUOUS_LO) & (img[..., :3] < _AMBIGUOUS_HI)).any(axis=2)
        bad = int((mid & opaque).sum())
        if bad > 0.005 * opaque.size:
            _ambiguity_warned = True
            print(f"[vq2] WARNING: id pass produced {bad} ambiguous (blended) pixels -- the "
                  f"anti-aliasing/dither/view-transform guards did not all take on this Blender "
                  f"build. Silhouette edges may be off by a pixel; check the overlay sheet.")
    return [np.asarray(opaque & (code == _code_of(i))) for i in range(n_targets)]


# --------------------------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------------------------
def render_id_masks(scene, targets: list, hidden=()) -> list[np.ndarray]:
    """Boolean silhouette per object in ``targets``, as the CAMERA sees it.

    ``targets``  -- ordered bpy Objects to identify. Everything else in the scene still renders
                    (painted black), so occlusion by props, people, the floor and other gates is
                    resolved by the z-buffer exactly as in the beauty render -- which is the whole
                    point: this is the rendered truth, not a re-derived guess.
    ``hidden``   -- objects to force ``hide_render`` for this pass only. Used for the opening-plane
                    proxies: a solid black proxy sitting in a gate's opening would wrongly occlude
                    a farther gate seen THROUGH that opening.

    Targets are force-UNhidden for the duration (the opening proxies live with ``hide_render`` on so
    they can never leak into the beauty render; without this they would render nothing here and
    every opening mask would come back empty -- a silent, total loss of class 1).

    Batches automatically at 7 targets per render (the palette size). Restores every scene setting
    and every object colour / hide flag it touched.
    """
    if bpy is None:                                 # pragma: no cover - laptop import guard
        raise RuntimeError("bpy_idmask.render_id_masks requires a running Blender")
    targets = list(targets)
    if not targets:
        return []

    objects = [o for o in scene.objects]
    snap = _snapshot(scene)
    saved_colors = [(o, tuple(o.color)) for o in objects]
    saved_hidden = [(o, bool(o.hide_render)) for o in list(hidden) + targets]
    masks: list[np.ndarray] = []
    try:
        _configure_id_pass(scene)
        for o in hidden:
            o.hide_render = True
        for o in targets:
            o.hide_render = False
        for start in range(0, len(targets), MAX_IDS_PER_PASS):
            batch = targets[start:start + MAX_IDS_PER_PASS]
            # Repaint EVERY object black each batch (not just the previous batch's targets): a prop
            # spawned between calls would otherwise keep a stale palette colour and be decoded as a
            # gate. Cheap -- it is a python-level attribute write, not a mesh edit.
            for o in objects:
                o.color = (0.0, 0.0, 0.0, 1.0)
            for i, o in enumerate(batch):
                o.color = (*PALETTE[i], 1.0)
            masks.extend(_decode(_render_rgba(scene), len(batch)))
    finally:
        for o, col in saved_colors:
            try:
                o.color = col
            except ReferenceError:
                pass
        for o, hid in saved_hidden:
            try:
                o.hide_render = hid
            except ReferenceError:
                pass
        _restore(scene, snap)
    return masks


def disjoint_batches(quads_px: list[np.ndarray]) -> list[list[int]]:
    """Group indices so that no two members' projected quads can overlap on screen.

    The opening-plane proxies are SOLID: two of them in the same render would occlude each other,
    and a gate's opening seen through a NEARER gate's opening (the classic down-course racing shot)
    is exactly that case -- the far opening would come back empty. Grouping by non-overlapping
    axis-aligned bounds is conservative (it may split more than strictly necessary) but can never
    let an occlusion through, and in the common 1-2 gate frame it still costs a single render.
    """
    boxes = []
    for q in quads_px:
        q = np.asarray(q, dtype=float)
        boxes.append((q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()))
    batches: list[list[int]] = []
    for i, bi in enumerate(boxes):
        for batch in batches:
            if all(not (bi[0] <= boxes[j][2] and boxes[j][0] <= bi[2]
                        and bi[1] <= boxes[j][3] and boxes[j][1] <= bi[3]) for j in batch):
                batch.append(i)
                break
        else:
            batches.append([i])
    return batches
