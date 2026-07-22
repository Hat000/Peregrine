"""Backend-agnostic dataset generation: sample geometry -> render -> augment -> write YOLO-pose.

The render step is the ONLY backend-specific part, behind :class:`RenderBackend`:

  * :class:`ProceduralBackend` -- numpy/cv2, NO Blender. Draws flat gate rings at the exact
    projected corners. Runs + is unit-tested on the laptop; the full pipeline (geometry,
    augment, label writing, dataset layout, data.yaml) is therefore exercised end-to-end here.
  * ``BlenderBackend`` (racer.vision.blender_gen.backends.blender) -- bpy/Cycles photoreal, the
    ShadowPC drop-in. Same interface, so labels are identical -- the renderer never feeds them
    back; geometry.py is the single source of corner positions.

Output layout (ultralytics): ``<out>/images/{train,val}/*.png`` + ``<out>/labels/{train,val}/
*.txt`` + ``<out>/data.yaml``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .config import AppearanceConfig, ScenarioPreset
from .contract import (
    DATA_YAML,
    GATE_INNER_SIZE_M,
    GATE_OUTER_SIZE_M,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    VQ1_GATE_RED_RGB,
)
from .augment import augment_frame_with_masks
from .geometry import (
    FrameSpec, GateRender, ViewpointConfig,
    sample_frames, sample_negative_frames, sample_partial_frames,
)
from .labels import frame_label_rows
from racer.vision.seg_labels import seg_rows_from_corners, seg_rows_from_silhouette
from .masks import gate_ring_mask, instance_mask_from_silhouettes


class RenderBackend(Protocol):
    """Render one FrameSpec to a (H, W, 3) uint8 BGR image. Implementations are stateful (they
    may hold a Blender scene); ``render`` is called once per frame.

    OPTIONAL extension, implemented only by ``BlenderBackend``: an ``emit_silhouettes`` attribute
    plus a ``gate_silhouettes() -> {gate_id: (ring_mask, opening_mask)}`` method returning the
    per-gate masks MEASURED off the frame just rendered. A backend without it falls back to the
    flat-quad seg target -- loudly (see ``_seg_source``), never silently.
    """

    def render(self, frame: FrameSpec, preset: ScenarioPreset, rng: np.random.Generator) -> np.ndarray:
        ...


# --------------------------------------------------------------------------------------
# Procedural (laptop / no-Blender) backend
# --------------------------------------------------------------------------------------
def sample_gate_color_bgr(rng: np.random.Generator, ap: AppearanceConfig) -> tuple[int, int, int]:
    """A gate colour as a BGR uint8 triple: anchored at the extracted VQ1 orange-red, jittered
    in hue/sat/val per the appearance config. hue_jitter=0 + use_vq1_red -> exactly VQ1 red."""
    r, g, b = VQ1_GATE_RED_RGB
    base_hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0, 0].astype(np.float64)
    base_h = base_hsv[0]                                                  # OpenCV hue 0..179
    if ap.use_vq1_red and ap.gate_hue_jitter == 0.0:
        h = base_h
    else:
        h = (base_h + rng.uniform(-ap.gate_hue_jitter, ap.gate_hue_jitter) * 180.0) % 180.0
    s = float(np.clip(rng.uniform(*ap.gate_sat_range) * 255.0, 0, 255))
    v = float(np.clip(rng.uniform(*ap.gate_val_range) * 255.0, 0, 255))
    bgr = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


@dataclass
class ProceduralBackend:
    """Flat-shaded gate rings on a randomized background. Laptop fallback + test vehicle (NOT
    the photoreal output -- that is the Blender backend). Corners come from geometry.py, so the
    rings land exactly under the labels (the viz overlay confirms it)."""

    def render(self, frame: FrameSpec, preset: ScenarioPreset, rng: np.random.Generator) -> np.ndarray:
        ap = preset.appearance
        img = self._background(rng, ap)
        # paint far -> near so a nearer gate occludes a farther one (matches visibility logic)
        for gr in sorted(frame.gates, key=lambda g: -g.range_m):
            self._draw_ring(img, gr, sample_gate_color_bgr(rng, ap))
        return img

    @staticmethod
    def _background(rng: np.random.Generator, ap: AppearanceConfig) -> np.ndarray:
        base = rng.integers(0, 110, size=3).astype(np.int16)
        img = np.tile(base, (IMAGE_HEIGHT, IMAGE_WIDTH, 1))
        grad = np.linspace(rng.uniform(-40, 40), rng.uniform(-40, 40), IMAGE_WIDTH)
        img = np.ascontiguousarray(np.clip(img + grad[None, :, None], 0, 255).astype(np.uint8))
        for _ in range(int(rng.integers(0, ap.clutter_max + 1))):
            p1 = (int(rng.integers(0, IMAGE_WIDTH)), int(rng.integers(0, IMAGE_HEIGHT)))
            p2 = (int(rng.integers(0, IMAGE_WIDTH)), int(rng.integers(0, IMAGE_HEIGHT)))
            cv2.rectangle(img, p1, p2, tuple(int(c) for c in rng.integers(0, 256, size=3)), -1)
        return img

    @staticmethod
    def _draw_ring(img: np.ndarray, gr: GateRender, color_bgr: tuple[int, int, int]) -> None:
        ring = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(ring, [gr.outer_px.round().astype(np.int32)], 1)
        cv2.fillPoly(ring, [gr.keypoints_px.round().astype(np.int32)], 0)
        img[ring == 1] = color_bgr


# --------------------------------------------------------------------------------------
# Dataset writer (backend-agnostic)
# --------------------------------------------------------------------------------------
@dataclass
class GenStats:
    n_images: int
    n_labels: int
    n_gates: int
    n_negatives: int = 0


# --------------------------------------------------------------------------------------
# segmentation target: rendered silhouette (Blender) vs flat quads (procedural)
# --------------------------------------------------------------------------------------
SEG_SOURCE_SILHOUETTE = "rendered-silhouette"
SEG_SOURCE_FLAT_QUAD = "flat-quad-fallback"

_FLAT_QUAD_WARNING = """\
[vq2] WARNING: {backend} cannot measure gate silhouettes, so the seg target falls back to FLAT
      QUADS -- the projected 1.5 m inner and 2.72 m outer squares, clipped. That is a DIFFERENT
      TARGET from the one the Blender backend writes: it ignores the gate's 0.26 m depth, so it
      omits the inner side walls and over-states the opening on every close, off-axis gate. Do not
      mix this output into a dataset with Blender-rendered seg labels. Stamped in seg/SOURCE.txt.
"""


def _seg_source(backend: RenderBackend) -> str:
    """Which seg target this backend can produce -- and say so out loud if it is the fallback."""
    if callable(getattr(backend, "gate_silhouettes", None)):
        return SEG_SOURCE_SILHOUETTE
    print(_FLAT_QUAD_WARNING.format(backend=type(backend).__name__))
    return SEG_SOURCE_FLAT_QUAD


def _silhouette_stack(gates: list[GateRender], silhouettes: dict,
                      h: int, w: int) -> tuple[np.ndarray | None, dict[int, int]]:
    """Pack {gate_id: (ring, opening)} into an (H, W, 2*N) uint8 stack + a gate_id -> slot map.

    One channel per instance rather than one paint value per gate: instances legitimately OVERLAP
    (a far gate seen through a near gate's opening is visible in both), and a single-channel id map
    would give those pixels to whichever gate was painted last. The stack is what goes through the
    augmentation warp, so it has to be lossless.
    """
    slots = {int(g.gate_id): k for k, g in enumerate(gates)}
    if not slots:
        return None, {}
    stack = np.zeros((h, w, 2 * len(slots)), dtype=np.uint8)
    for gid, k in slots.items():
        pair = silhouettes.get(gid)
        if pair is None:
            continue
        ring, opening = pair
        if ring is not None:
            stack[..., 2 * k] = np.asarray(ring).astype(np.uint8)
        if opening is not None:
            stack[..., 2 * k + 1] = np.asarray(opening).astype(np.uint8)
    return stack, slots


def _gate_masks(stack, slots: dict[int, int], gate: GateRender):
    """(ring, opening) for one POST-augment gate, or (None, None) if it has no measured mask.

    Looked up by ``gate_id``: ``augment_frame`` rebuilds and REORDERS the gate list, so positional
    pairing would silently hand gate A's mask to gate B on every augmented frame.
    """
    k = slots.get(int(gate.gate_id))
    if stack is None or k is None or 2 * k + 1 >= stack.shape[2]:
        return None, None
    return stack[..., 2 * k].astype(bool), stack[..., 2 * k + 1].astype(bool)


def _write_split(
    out: Path, split: str, n: int, preset: ScenarioPreset, backend: RenderBackend,
    seed: int, image_ext: str, track_path: str | None, emit_masks: bool = False,
    emit_seg: bool = False,
) -> GenStats:
    img_dir = out / "images" / split
    lbl_dir = out / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = out / "masks" / split
    if emit_masks:
        mask_dir.mkdir(parents=True, exist_ok=True)
    # SEG labels are the gate's RENDERED SILHOUETTE when the backend can measure one, and never a
    # re-derivation of the pose row (which clamps off-frame corners to the border with v=0, so a
    # homography refit to the survivors extrapolates -- 2.8-7.9 px of avoidable error on exactly the
    # cropped gates this dataset exists to teach). The flat-quad path remains for the procedural
    # backend, which really does draw flat quads; the source is stamped on disk either way.
    seg_dir = out / "seg" / split
    seg_source = _seg_source(backend) if (emit_seg or emit_masks) else None
    if emit_seg:
        seg_dir.mkdir(parents=True, exist_ok=True)
        (out / "seg" / "SOURCE.txt").write_text(
            f"{seg_source}\nbackend={type(backend).__name__}\n")
    use_silhouettes = seg_source == SEG_SOURCE_SILHOUETTE
    if use_silhouettes:
        backend.emit_silhouettes = True     # opt the id pass IN; off by default, costs nothing else
    seg_census: dict[str, int] = {}
    rng = np.random.default_rng(seed)
    n_neg = int(round(n * float(preset.negative_fraction)))
    n_pos = n - n_neg
    made = n_labels = n_gates = n_negatives = 0

    # positives: oversample geometry so post-augment label drops don't starve the split. The crop
    # arm (cfg.partial_gates) swaps in the partial-gate sampler -- same signature, same writer --
    # and routes the relaxed positive rule through augment_frame so _recompute keeps the partials.
    vp = preset.viewpoint
    partial = (vp.partial_min_corners, vp.partial_min_area_frac) if vp.partial_gates else None
    pos_sampler = sample_partial_frames if vp.partial_gates else sample_frames
    for fs in pos_sampler(int(n_pos * 3 + 8), vp, seed=seed, track_path=track_path):
        if made >= n_pos:
            break
        image = backend.render(fs, preset, rng)
        # Masks are measured on the CLEAN render, so they must ride through the same geometric warp
        # the image does -- see augment.augment_frame_with_masks for why this is load-bearing.
        stack, slots = ((None, {}) if not use_silhouettes else
                        _silhouette_stack([g for g in fs.gates if g.visible],
                                          backend.gate_silhouettes(), IMAGE_HEIGHT, IMAGE_WIDTH))
        image, gates, stack = augment_frame_with_masks(
            image, fs.gates, preset.augment, rng, partial=partial, mask=stack)
        labeled = [g for g in gates if g.visible]
        rows = frame_label_rows(labeled)
        if not rows:
            continue

        cv2.imwrite(str(img_dir / f"{made:06d}.{image_ext}"), image)
        (lbl_dir / f"{made:06d}.txt").write_text("\n".join(rows) + "\n")
        if emit_masks:    # instance mask from the SAME post-augment labelled gates
            if use_silhouettes:
                rings = [_gate_masks(stack, slots, g)[0] for g in labeled]
                cv2.imwrite(str(mask_dir / f"{made:06d}.png"),
                            instance_mask_from_silhouettes(labeled, rings))
            else:
                cv2.imwrite(str(mask_dir / f"{made:06d}.png"), gate_ring_mask(labeled))
        if emit_seg:
            srows: list[str] = []
            for g in labeled:
                if use_silhouettes:
                    ring, opening = _gate_masks(stack, slots, g)
                    if ring is None:
                        seg_census["no-mask"] = seg_census.get("no-mask", 0) + 1
                        continue
                    r, why = seg_rows_from_silhouette(ring, opening, IMAGE_WIDTH, IMAGE_HEIGHT)
                else:
                    r = seg_rows_from_corners(g.keypoints_px, g.outer_px)
                    why = "ok" if r else "no-visible-area"
                seg_census[why] = seg_census.get(why, 0) + 1
                srows.extend(r)
            seg_text = ("\n".join(srows) + "\n") if srows else ""
            (seg_dir / f"{made:06d}.txt").write_text(seg_text)
        made += 1
        n_labels += len(rows)
        n_gates += len(rows)

    # hard negatives: background/confusers, NO gate -> empty label file (ultralytics background)
    for fs in sample_negative_frames(n_neg, preset.viewpoint, seed=seed + 7919, track_path=track_path):
        if n_negatives >= n_neg:
            break
        image = backend.render(fs, preset, rng)
        image, _, _ = augment_frame_with_masks(image, [], preset.augment, rng)
        cv2.imwrite(str(img_dir / f"{made:06d}.{image_ext}"), image)
        (lbl_dir / f"{made:06d}.txt").write_text("")        # empty => background/negative
        if emit_masks:    # all-zero mask: no gate pixels in a negative
            cv2.imwrite(str(mask_dir / f"{made:06d}.png"), gate_ring_mask([]))
        if emit_seg:      # WRITE the empty file: a missing one means "unlabelled", not "negative"
            (seg_dir / f"{made:06d}.txt").write_text("")
        made += 1
        n_negatives += 1
    if seg_census:
        print(f"[vq2] seg census ({split}, {seg_source}): "
              + ", ".join(f"{k}={v}" for k, v in sorted(seg_census.items())))
    return GenStats(n_images=made, n_labels=n_labels, n_gates=n_gates, n_negatives=n_negatives)


def generate_dataset(
    out_dir,
    preset: ScenarioPreset,
    backend: RenderBackend | None = None,
    *,
    n_train: int = 1000,
    n_val: int = 100,
    seed: int = 0,
    image_ext: str = "png",
    track_path: str | None = None,
    emit_masks: bool = False,
    emit_seg: bool = False,
) -> Path:
    """Generate a YOLO-pose dataset with ``backend`` (default ProceduralBackend). Returns the
    data.yaml path. The Blender entrypoint passes a ``BlenderBackend``; everything else (geometry,
    augment, labels, layout) is shared, so a procedural run and a Cycles run differ ONLY in pixels.

    ``emit_masks`` also writes a per-frame gate instance mask to ``<out>/masks/{split}/*.png``;
    ``emit_seg`` writes 2-class YOLO-seg polygons to ``<out>/seg/{split}/*.txt``. With the Blender
    backend both are the gate's RENDERED SILHOUETTE (true 3D shape, side walls and all); with the
    procedural backend both are the flat-quad approximation and ``<out>/seg/SOURCE.txt`` says so.
    """
    backend = backend or ProceduralBackend()
    out = Path(out_dir)
    train = _write_split(out, "train", n_train, preset, backend, seed, image_ext, track_path,
                         emit_masks, emit_seg)
    _write_split(out, "val", n_val, preset, backend, seed + 10_000, image_ext, track_path,
                 emit_masks, emit_seg)
    yaml_path = out / "data.yaml"
    yaml_path.write_text(DATA_YAML.format(path=str(out.resolve())))
    return yaml_path
