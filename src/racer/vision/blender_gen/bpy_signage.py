"""Printed gate SIGNAGE (decal atlas + the gate material that carries it).

WHY THIS MODULE EXISTS
----------------------
The real A2RL x DCL gate is not a bare red square. It is a printed structural frame: big white
"AI-GP" lettering across the top bar, white CHECKERBOARD strips down the verticals, and small
"DCL" / "ANDURIL" / "VQ-01" brand panels around the bottom. Everything synthetic to date rendered
a BARE flat red annulus, and the cost was measured, not guessed:

  * an M+1 keypoint model trained on the current synthetic data and validated on REAL hand-labelled
    frames PEAKED AT EPOCH 10 (pose mAP50 0.503) and decayed to 0.282 by epoch 100 -- centre error
    20.5 -> 35.6 px, coverage 98% -> 84%. The extra epochs learned synthetic APPEARANCE, not gates.
  * a segmentation model scored 28.5 px centre error on synthetic val but 80 px on real frames.
  * the auto-seeder built for the labelling inbox kept locking onto the signage panels AS IF THEY
    WERE GATES -- the detector has never been shown what those white panels are, so it has no way
    to know they are part of the gate rather than another gate.

The decal is therefore not decoration. It is the texture statistic that separates "large uniform
red blob" (what the model currently learns) from "red frame carrying high-contrast white markings"
(what it must survive at deploy time).

DESIGN NOTES
------------
* Built with numpy (+ PIL for glyphs when importable) straight into a Blender image datablock --
  no asset files to ship, no junction to break, and the atlas is reproducible from a seed.
* Mapped through OBJECT coordinates, NOT UVs. The gate mesh (bpy_scene._gate_frame_geometry) has
  16 shared verts and no UV layer; unwrapping it would mean editing the geometry that the frozen
  keypoint contract projects through. Object space is exactly as good here: the front/back annulus
  faces are planar in gate-local XY, so a linear remap of (x, y) -> (u, v) IS a planar decal.
* The gate frame has Y pointing DOWN (contract) while Blender image V points UP, so the mapping
  flips Y. Get this wrong and every glyph renders upside down -- which is not a subtle bug but is
  an easy one to not notice on a checkerboard.
* The pool of atlases is built ONCE and pinned with ``use_fake_user``. Per-frame materials are
  cheap node graphs that reference a pooled image; the backend's ``orphans_purge`` would otherwise
  free the atlas the moment its one material died.

This module is importable WITHOUT bpy (the atlas builder is pure numpy/PIL) so the laptop test
suite can exercise the layout maths; only the material builder needs Blender.
"""
from __future__ import annotations

import colorsys
import math

import numpy as np

try:                                    # only importable inside Blender
    import bpy
except Exception:                       # pragma: no cover - exercised only on ShadowPC
    bpy = None

from .contract import GATE_INNER_SIZE_M, GATE_OUTER_SIZE_M, VQ1_GATE_RED_RGB

# The ring band in NORMALISED outer-square units: the decal only ever shows on the annulus, so all
# content has to live between the outer edge (0 / 1) and the opening (BAND .. 1-BAND).
BAND = (1.0 - GATE_INNER_SIZE_M / GATE_OUTER_SIZE_M) / 2.0      # 0.2243 for 1.5 / 2.72

# Brand strings seen on the real gates in C:/Users/Shadow/vq2_label_batch_2026-07-22/frames.
_MARQUEE = "AI-GP"
_SMALL_MARKS = ("DCL", "ANDURIL", "VQ-01", "A2RL", "TM")

# Candidate bold faces, best first. Arial Black is the closest stock stand-in for the heavy squarish
# AI-GP wordmark; any of these beats no glyphs at all, and no font at all still leaves the
# checkerboards + bars (which carry most of the texture signal anyway).
_FONT_CANDIDATES = ("ariblk.ttf", "arialbd.ttf", "impact.ttf", "segoeuib.ttf", "tahomabd.ttf")


# ---------------------------------------------------------------------------- atlas (pure numpy/PIL)
def _try_font(size: int):
    """A bold TrueType face at ``size`` px, or None if PIL / the fonts are unavailable.

    Never raises: signage must degrade to 'checkerboards but no lettering' rather than kill a
    multi-hour render because a font moved.
    """
    try:
        from PIL import ImageFont
    except Exception:                                    # pragma: no cover - PIL missing
        return None
    for name in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    try:                                                 # pragma: no cover - last resort
        from PIL import ImageFont as _IF
        return _IF.load_default()
    except Exception:
        return None


def _checker_column(alpha: np.ndarray, u0: float, u1: float, v0: float, v1: float,
                    n: int, rng, taper: bool = True) -> None:
    """Paint a DIAGONAL-ish checkerboard strip into ``alpha`` over the normalised box (u0..u1, v0..v1).

    The real gates carry checkers that thin out along the bar (a run of full squares that decays to
    single scattered squares) rather than a regular chess grid -- see the reference frames. ``taper``
    reproduces that: the probability a cell is inked falls off along the strip. That irregularity is
    the point; a perfectly regular grid is a texture the network can memorise as "the synthetic
    gate", which is the exact failure mode this preset exists to fix.
    """
    h, w = alpha.shape
    cols = max(1, int(round((u1 - u0) * w / ((v1 - v0) * h / max(n, 1)))))
    cell_v = (v1 - v0) / max(n, 1)
    cell_u = (u1 - u0) / max(cols, 1)
    phase = int(rng.integers(0, 2))
    for i in range(n):
        for j in range(cols):
            if (i + j + phase) % 2:
                continue
            keep = 1.0 - (i / max(n - 1, 1)) * 0.85 if taper else 1.0
            if float(rng.random()) > keep:
                continue
            a0 = int(round((v0 + i * cell_v) * h))
            a1 = int(round((v0 + (i + 1) * cell_v) * h))
            b0 = int(round((u0 + j * cell_u) * w))
            b1 = int(round((u0 + (j + 1) * cell_u) * w))
            alpha[a0:a1, b0:b1] = 1.0


def _bar(alpha: np.ndarray, u0: float, u1: float, v0: float, v1: float) -> None:
    h, w = alpha.shape
    alpha[int(v0 * h):int(v1 * h), int(u0 * w):int(u1 * w)] = 1.0


def build_decal_alpha(seed: int, size: int = 1024) -> np.ndarray:
    """A (size, size) float32 INK COVERAGE map in [0, 1], laid out on the gate's outer square.

    Row 0 is the TOP of the gate as seen upright; the material's mapping node performs the
    gate-frame Y-down flip. 1.0 = printed white marking, 0.0 = bare frame colour.
    """
    rng = np.random.default_rng(seed)
    a = np.zeros((size, size), dtype=np.float32)

    # --- checkerboard strips on the vertical bars. Which bars carry them varies per gate, as on the
    #     real course (each gate is a separately printed panel).
    n_cells = int(rng.integers(5, 9))
    if float(rng.random()) < 0.9:                        # LEFT bar, checkers running DOWN
        pad = BAND * 0.18
        _checker_column(a, pad, BAND - pad, BAND * 1.05, BAND * 1.05 + 0.42, n_cells, rng)
    if float(rng.random()) < 0.8:                        # RIGHT bar, checkers running UP
        pad = BAND * 0.18
        _checker_column(a, 1.0 - BAND + pad, 1.0 - pad, BAND * 1.05, BAND * 1.05 + 0.42, n_cells, rng)
    # A short run on the BOTTOM bar on some gates. It occupies the CENTRE slot, and _draw_text is
    # told to leave that slot empty -- overlapping ink would read as one illegible smear rather than
    # as the two distinct marking TYPES (geometric checker vs glyph) the detector should see.
    bottom_run = float(rng.random()) < 0.45
    if bottom_run:
        pad = BAND * 0.22
        _checker_column(a, 0.42, 0.58, 1.0 - BAND + pad, 1.0 - pad,
                        max(2, n_cells // 3), rng, taper=False)

    # --- thin white pinstripe along one edge of the top bar on some gates (printing trim line)
    if float(rng.random()) < 0.35:
        _bar(a, 0.02, 0.98, BAND * 0.86, BAND * 0.90)

    # --- glyphs (optional: no PIL/font -> checkers only, which is still a big step up from bare)
    _draw_text(a, rng, size, bottom_run)
    return np.clip(a, 0.0, 1.0)


def _draw_text(a: np.ndarray, rng, size: int, bottom_run: bool = False) -> None:
    """Overlay the wordmarks onto the ink map. Silent no-op when PIL is unavailable."""
    try:
        from PIL import Image, ImageDraw
    except Exception:                                    # pragma: no cover - PIL missing
        return
    img = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(img)

    def text(s, cx, cy, px, anchor="mm", angle=0.0):
        """Draw ``s`` centred at normalised (cx, cy) at ``px`` pixels tall, optionally rotated."""
        f = _try_font(px)
        if f is None:
            return
        if abs(angle) < 1e-6:
            d.text((cx * size, cy * size), s, fill=255, font=f, anchor=anchor)
            return
        # rotated small print (the real gates carry vertical text down the side bars): render into a
        # scratch tile, rotate, then composite -- PIL cannot draw rotated text directly.
        pad = px * (len(s) + 2)
        tile = Image.new("L", (pad, px * 2), 0)
        ImageDraw.Draw(tile).text((pad // 2, px), s, fill=255, font=f, anchor="mm")
        tile = tile.rotate(angle, expand=True, resample=Image.BILINEAR)
        img.paste(tile, (int(cx * size - tile.width / 2), int(cy * size - tile.height / 2)), tile)

    # The MARQUEE across the top bar -- the single most recognisable marking on the real gate, and
    # the biggest non-red feature the detector has to learn belongs to the gate.
    top_px = int(BAND * size * float(rng.uniform(0.62, 0.80)))
    text(_MARQUEE, float(rng.uniform(0.44, 0.56)), BAND * 0.5, top_px)
    if float(rng.random()) < 0.6:
        text("TM", 0.90, BAND * 0.30, max(8, top_px // 5))

    # Bottom bar brand panels, laid left-to-right in non-overlapping slots. When the checker run took
    # the centre slot the glyphs move to the outer two, so ink never lands on ink.
    small_px = int(BAND * size * float(rng.uniform(0.22, 0.32)))
    slots = [0.16, 0.84] if bottom_run else [0.20, 0.50, 0.80]
    marks = list(_SMALL_MARKS[:4])
    rng.shuffle(marks)
    n_marks = len(slots) if bottom_run else int(rng.integers(2, 4))
    for cx, s in zip(slots, marks[:n_marks]):
        text(s, cx + float(rng.uniform(-0.02, 0.02)), 1.0 - BAND * 0.5, small_px)

    # Vertical small print down a side bar (spec plate / scrutineering text on the real gates).
    if float(rng.random()) < 0.55:
        text("VQ-01  GATE  SPEC", BAND * 0.45, 0.66, max(7, small_px // 2),
             angle=float(rng.choice([90.0, -90.0])))

    ink = np.asarray(img, dtype=np.float32) / 255.0
    np.maximum(a, ink, out=a)


def build_decal_rgba(seed: int, size: int = 1024) -> np.ndarray:
    """(size, size, 4) float32 RGBA: white ink, alpha = coverage. Blender-image ready."""
    alpha = build_decal_alpha(seed, size)
    rgba = np.ones((size, size, 4), dtype=np.float32)
    rgba[..., 3] = alpha
    return rgba


# ---------------------------------------------------------------------------- bpy image pool
def _to_blender_image(name: str, rgba: np.ndarray):
    """Push an (H, W, 4) float array into a NEW Blender image datablock (no file on disk).

    ``pixels`` is bottom-up, so the array is flipped vertically here -- the mapping node then only
    has to worry about the gate frame's Y-down convention, not about two flips at once.
    """
    h, w = rgba.shape[:2]
    img = bpy.data.images.new(name, width=w, height=h, alpha=True, float_buffer=True)
    img.colorspace_settings.name = "Non-Color"   # ink coverage is DATA; sRGB-decoding it would
    #                                              gamma-crush the anti-aliased glyph edges
    img.pixels.foreach_set(np.ascontiguousarray(rgba[::-1], dtype=np.float32).ravel())
    img.pack()                    # keep it self-contained; nothing to re-resolve at render time
    img.use_fake_user = True      # survive the backend's per-frame orphans_purge (see module docs)
    return img


class DecalPool:
    """A small pool of pre-built signage atlases, shared across the whole run.

    Rebuilding a 1024x1024 atlas per frame would cost more than the frame. A handful of variants is
    enough: real course gates are a small set of printed panels too, and the per-frame variation
    that matters (pose, lighting, wear, hue) is applied on top in the material.
    """

    def __init__(self, n: int = 8, size: int = 1024, seed: int = 20260723):
        self.images = []
        for k in range(int(n)):
            try:
                self.images.append(_to_blender_image(f"VQ2_Signage_{k:02d}",
                                                     build_decal_rgba(seed + k, size)))
            except Exception as exc:                     # pragma: no cover - defensive on ShadowPC
                print(f"[vq2] WARNING: signage atlas {k} failed ({type(exc).__name__}: {exc})")
        if not self.images:
            print("[vq2] WARNING: no signage atlases built -- gates render BARE (see bpy_signage).")

    def available(self) -> bool:
        return bool(self.images)

    def pick(self, rng):
        return self.images[int(rng.integers(len(self.images)))]


# ---------------------------------------------------------------------------- gate material
def wants_signage_material(appearance) -> bool:
    """True when the preset needs the signage-capable material builder rather than the plain one.

    Routed on either lever so a preset can ask for the hue re-centring WITHOUT signage, and so any
    preset that sets neither keeps the byte-identical legacy material path.
    """
    return (float(getattr(appearance, "gate_signage_prob", 0.0)) > 0.0
            or abs(float(getattr(appearance, "gate_hue_offset", 0.0))) > 1e-9)


def _u(rng, lo_hi) -> float:
    return float(rng.uniform(float(lo_hi[0]), float(lo_hi[1])))


def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def gate_base_rgb_linear(rng, appearance) -> tuple[float, float, float]:
    """Sample the gate's LINEAR base colour, honouring hue jitter AND the measured hue offset.

    ``gate_hue_offset`` exists because the frozen anchor VQ1_GATE_RED_RGB = (255, 50, 0) sits at
    hue 11.8 deg while gate pixels measured on 400 REAL VQ2 frames average 17.8 deg -- the real
    gate reads a touch more orange than the extracted anchor. The anchor is contract-frozen, so the
    correction is a preset-declared offset rather than an edit to the contract.
    """
    r, g, b = (c / 255.0 for c in VQ1_GATE_RED_RGB)
    h0, _s0, _v0 = colorsys.rgb_to_hsv(r, g, b)
    h = h0 + float(getattr(appearance, "gate_hue_offset", 0.0))
    jit = float(getattr(appearance, "gate_hue_jitter", 0.0))
    if jit > 0.0:
        h += float(rng.uniform(-1.0, 1.0)) * jit * 0.5
    h %= 1.0
    s = _u(rng, getattr(appearance, "gate_sat_range", (0.9, 1.0)))
    v = _u(rng, getattr(appearance, "gate_val_range", (0.85, 1.0)))
    return tuple(_srgb_to_linear(c) for c in colorsys.hsv_to_rgb(h, s, v))


def signage_gate_material(rng, appearance, pool: "DecalPool | None"):
    """Emissive glowing-red gate material, optionally carrying a printed signage decal.

    Shading model, and why each piece is there:
      * BASE + EMISSION share one colour, so the gate is SELF-LIT. Real VQ2 gates glow hard enough
        to light the floor under them in a dark hangar; a diffuse-only gate in a dark world renders
        near black and is a different object as far as the detector is concerned.
      * The decal mixes toward WHITE in both base colour and emission colour. White ink at the same
        emission strength carries ~3x the radiance of the saturated red, which is exactly how the
        checkerboards read on the real footage -- brighter than the frame, not merely lighter.
      * ``mix_fac`` is the ink coverage times a per-gate WEAR factor, so some gates carry crisp
        graphics and others are faded. Printed vinyl on a race course is not uniform.
    """
    base = gate_base_rgb_linear(rng, appearance)
    mat = bpy.data.materials.new("VQ2_GateSignage")
    mat.use_nodes = True
    nt = mat.node_tree
    b = nt.nodes.get("Principled BSDF") or nt.nodes.new("ShaderNodeBsdfPrincipled")
    emis = _u(rng, getattr(appearance, "gate_emission_range", (0.25, 0.5)))

    use_decal = (pool is not None and pool.available()
                 and float(rng.random()) < float(getattr(appearance, "gate_signage_prob", 0.0)))
    if use_decal:
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = pool.pick(rng)
        tex.extension = "EXTEND"       # the walls sample the boundary; CLIP would punch black holes
        tex.interpolation = "Linear"
        tc = nt.nodes.new("ShaderNodeTexCoord")
        mp = nt.nodes.new("ShaderNodeMapping")
        # Object space -> decal UV. Gate-local XY spans [-OUTER/2, +OUTER/2]; Y points DOWN in the
        # gate frame while image V points UP, hence the NEGATIVE Y scale. Both offsets are 0.5.
        s = 1.0 / GATE_OUTER_SIZE_M
        mp.inputs["Scale"].default_value = (s, -s, 1.0)
        mp.inputs["Location"].default_value = (0.5, 0.5, 0.0)
        nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
        nt.links.new(mp.outputs["Vector"], tex.inputs["Vector"])

        wear = nt.nodes.new("ShaderNodeMath")
        wear.operation = "MULTIPLY"
        wear.inputs[1].default_value = float(rng.uniform(0.72, 1.0))   # printed-vinyl wear
        nt.links.new(tex.outputs["Alpha"], wear.inputs[0])

        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.blend_type = "MIX"
        _set_mix(mix, base, (1.0, 1.0, 1.0, 1.0))
        nt.links.new(wear.outputs[0], mix.inputs["Factor"])
        out = mix.outputs[2] if len(mix.outputs) > 2 else mix.outputs[0]
        nt.links.new(out, b.inputs["Base Color"])
        _link_emission_color(nt, b, out)
    else:
        _set(b, "Base Color", (*base, 1.0))
        _set(b, "Emission Color", (*base, 1.0))

    _set(b, ("Emission Strength", "Emission"), emis)
    _set(b, "Metallic", _u(rng, getattr(appearance, "gate_metallic_range", (0.0, 0.2))))
    _set(b, "Roughness", _u(rng, getattr(appearance, "gate_roughness_range", (0.3, 0.6))))
    bev = nt.nodes.new("ShaderNodeBevel")
    bev.inputs["Radius"].default_value = 0.015
    nt.links.new(bev.outputs["Normal"], b.inputs["Normal"])
    return mat


def _set(node, names, value) -> bool:
    """Set a socket by name, tolerating the Principled socket renames across Blender point releases."""
    for n in (names if isinstance(names, (tuple, list)) else (names,)):
        if n in node.inputs:
            node.inputs[n].default_value = value
            return True
    return False


def _set_mix(mix, rgba_a, rgba_b) -> None:
    """Seed a ShaderNodeMix(RGBA)'s two colour inputs; index-based because both are named 'A'/'B'
    across several data types and the RGBA pair is the 6th/7th socket in 4.x."""
    cols = [i for i in mix.inputs if i.type == "RGBA"]
    if len(cols) >= 2:
        cols[0].default_value = (*rgba_a, 1.0) if len(rgba_a) == 3 else rgba_a
        cols[1].default_value = rgba_b


def _link_emission_color(nt, bsdf, socket) -> None:
    for n in ("Emission Color", "Emission"):
        if n in bsdf.inputs and bsdf.inputs[n].type == "RGBA":
            nt.links.new(socket, bsdf.inputs[n])
            return
