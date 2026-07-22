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
from .augment import augment_frame
from .geometry import (
    FrameSpec, GateRender, ViewpointConfig,
    sample_frames, sample_negative_frames, sample_partial_frames,
)
from .labels import frame_label_rows
from racer.vision.seg_labels import seg_rows_from_corners
from .masks import gate_ring_mask


class RenderBackend(Protocol):
    """Render one FrameSpec to a (H, W, 3) uint8 BGR image. Implementations are stateful (they
    may hold a Blender scene); ``render`` is called once per frame."""

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
    # SEG labels come from the EXACT projected corners, never from the pose row. The pose row clamps
    # off-frame corners to the border (v=0), so deriving from it would refit a homography to the
    # survivors and extrapolate -- 2.8-7.9 px of avoidable error on exactly the cropped gates this
    # dataset exists to teach. The renderer knows the true corners, so use them.
    seg_dir = out / "seg" / split
    if emit_seg:
        seg_dir.mkdir(parents=True, exist_ok=True)
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
        image, gates = augment_frame(image, fs.gates, preset.augment, rng, partial=partial)
        labeled = [g for g in gates if g.visible]
        rows = frame_label_rows(labeled)
        if not rows:
            continue
        cv2.imwrite(str(img_dir / f"{made:06d}.{image_ext}"), image)
        (lbl_dir / f"{made:06d}.txt").write_text("\n".join(rows) + "\n")
        if emit_masks:    # instance mask from the SAME post-augment labelled gates
            cv2.imwrite(str(mask_dir / f"{made:06d}.png"), gate_ring_mask(labeled))
        if emit_seg:
            srows = [r for g in labeled
                     for r in seg_rows_from_corners(g.keypoints_px, g.outer_px)]
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
        image, _ = augment_frame(image, [], preset.augment, rng)
        cv2.imwrite(str(img_dir / f"{made:06d}.{image_ext}"), image)
        (lbl_dir / f"{made:06d}.txt").write_text("")        # empty => background/negative
        if emit_masks:    # all-zero mask: no gate pixels in a negative
            cv2.imwrite(str(mask_dir / f"{made:06d}.png"), gate_ring_mask([]))
        made += 1
        n_negatives += 1
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

    ``emit_masks`` also writes a per-frame gate-ring instance mask to ``<out>/masks/{split}/*.png``
    (see :mod:`racer.vision.blender_gen.masks`) -- the banked segmentation hedge alongside keypoints.
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
