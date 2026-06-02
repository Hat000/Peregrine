"""Procedural synthetic gate dataset for pre-training a YOLO-pose corner detector.

Renders domain-randomized gate images with PIXEL-PERFECT labels (the 4 inner-square
corners, in the locked canonical order) by reusing ``gate_pose.project_gate_corners``,
so the on-disk keypoints round-trip through PnP exactly. The point of this data is to
teach appearance-invariant *geometry* (find a square gate's 4 corners); photoreal realism
+ course-specific accuracy come later from fine-tuning on real sim frames (the auto-label
flywheel). A Blender renderer can be swapped in behind the same label pipeline for
physically-accurate shading + the 260 mm gate depth.

Curriculum (``level`` 1->3): GEOMETRY is maxed at every level (full perspective +
transit/clipping/occlusion), and only APPEARANCE ramps:
  L1  clean: high contrast, low distraction, uniform gate colour (learn the shape).
  L2  realistic: + directional gradient shading (random light direction), colour, noise,
      background clutter, and occluders.
  L3  chaos: + heavy augmentation (motion blur, JPEG artifacts, glare, dropout, ...),
      via Albumentations if installed, else an equivalent OpenCV pipeline.

Edge cases the detector MUST handle (so they are generated from L1 up):
  - gate transit: the drone is so close the OUTER boundary leaves the frame and only the
    inner ring + its 4 corners are visible (the ring colour bleeds to the frame edge);
  - clipping: a corner leaves the frame -> a 3-corner sample, that corner flagged v=0;
  - occlusion: an object covers a corner (in-frame but hidden) -> flagged v=1.
A low-confidence / missing corner is exactly what the gate_pose 3-corner P3P fallback
consumes, so YOLO is trained to emit it.

Labels are ultralytics YOLO-pose rows ``class cx cy w h (x y v)*4`` (normalized). Keypoint
order matches ``gate_pose``: 0=lower-left, 1=lower-right, 2=upper-right, 3=upper-left (gate
frame X-right, Y-down). ``flip_idx=[1,0,3,2]`` keeps horizontal-flip augmentation correct.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from racer.frames import CAMERA_INTRINSICS_K, IMAGE_HEIGHT, IMAGE_WIDTH
from racer.vision.gate_pose import GATE_INNER_SIZE_M, project_gate_corners

GATE_OUTER_SIZE_M = 2.7  # spec 3.7 outer boundary
FLIP_IDX = [1, 0, 3, 2]  # horizontal flip swaps left<->right corners

# YOLO-pose keypoint visibility flags.
V_VIS, V_OCC, V_OFF = 2, 1, 0   # visible / occluded-in-frame / off-frame

# Pose sampling: geometry is maxed across ALL levels (ramp appearance, not geometry).
# Aggressive but within a realistic approach envelope, and biased CLOSE so the transit and
# clipped-corner cases the P3P fallback exists for are well represented.
_MIN_DIST_M, _MAX_DIST_M = 1.0, 14.0
_PITCH_RAD, _YAW_RAD, _ROLL_RAD = 0.5, 0.8, 0.7   # x up/down, y left/right (trapezoid), z roll (diamond)
_OCCLUSION_PROB = 0.35    # chance of an occluder at level >= 2


@dataclass
class SyntheticSample:
    image_bgr: np.ndarray      # (H, W, 3) uint8
    keypoints_px: np.ndarray   # (4, 2) inner corners, canonical order
    bbox_xywh: np.ndarray      # (4,) pixel bbox [x, y, w, h] from the outer square, clipped to frame
    visible: bool              # usable sample: >= 3 inner corners clearly visible + centre in frame
    R_cam_gate: np.ndarray     # (3, 3) ground-truth pose used to render
    t_cam_gate: np.ndarray     # (3,)
    visibility: np.ndarray = field(default_factory=lambda: np.full(4, V_VIS, dtype=int))  # (4,) per-corner flag


def sample_gate_pose(rng: np.random.Generator, min_dist: float = _MIN_DIST_M, max_dist: float = _MAX_DIST_M,
                     *, high_roll_prob: float = 0.0, edge_prob: float = 0.0):
    """A random gate pose in the camera frame whose centre projects inside the image.

    ``edge_prob``: chance of biasing the centre toward a frame edge so a corner clips (the
    near-edge troublemaker). ``high_roll_prob``: chance of forcing a large |roll| (26-49 deg,
    where the rare corner-identity swaps live). Both default 0 -> the original distribution, and
    with both 0 the RNG draw order is unchanged (existing seeded samples reproduce exactly)."""
    if edge_prob and rng.random() < edge_prob:
        if rng.random() < 0.5:                                    # clip on a left/right edge
            lo, hi = (0.02, 0.14) if rng.random() < 0.5 else (0.86, 0.98)
            u, v = rng.uniform(lo, hi) * IMAGE_WIDTH, rng.uniform(0.12, 0.88) * IMAGE_HEIGHT
        else:                                                     # clip on a top/bottom edge
            lo, hi = (0.02, 0.14) if rng.random() < 0.5 else (0.86, 0.98)
            u, v = rng.uniform(0.12, 0.88) * IMAGE_WIDTH, rng.uniform(lo, hi) * IMAGE_HEIGHT
    else:
        u = rng.uniform(0.12 * IMAGE_WIDTH, 0.88 * IMAGE_WIDTH)
        v = rng.uniform(0.12 * IMAGE_HEIGHT, 0.88 * IMAGE_HEIGHT)
    d = min_dist + (max_dist - min_dist) * rng.random() ** 2      # squared -> skew toward close
    ray = np.linalg.inv(CAMERA_INTRINSICS_K) @ np.array([u, v, 1.0])
    ray /= ray[2]
    t = d * ray                                                   # gate centre at depth ~d through (u, v)
    pitch = rng.uniform(-_PITCH_RAD, _PITCH_RAD)
    yaw = rng.uniform(-_YAW_RAD, _YAW_RAD)
    if high_roll_prob and rng.random() < high_roll_prob:
        roll = float(rng.choice((-1.0, 1.0))) * float(rng.uniform(0.45, 0.85))
    else:
        roll = rng.uniform(-_ROLL_RAD, _ROLL_RAD)
    tilt = Rotation.from_euler("xyz", [pitch, yaw, roll])
    return tilt.as_matrix(), t                                    # base orientation I = facing camera (IPPE native)


def _background(rng: np.random.Generator, level: int) -> np.ndarray:
    base = rng.integers(0, 130, size=3).astype(np.int16)
    img = np.tile(base, (IMAGE_HEIGHT, IMAGE_WIDTH, 1))
    grad = np.linspace(rng.uniform(-50, 50), rng.uniform(-50, 50), IMAGE_WIDTH)
    img = np.ascontiguousarray(np.clip(img + grad[None, :, None], 0, 255).astype(np.uint8))
    n_clutter = 0 if level <= 1 else int(rng.integers(0, 7))      # L1 = low distraction
    for _ in range(n_clutter):
        p1 = (int(rng.integers(0, IMAGE_WIDTH)), int(rng.integers(0, IMAGE_HEIGHT)))
        p2 = (int(rng.integers(0, IMAGE_WIDTH)), int(rng.integers(0, IMAGE_HEIGHT)))
        color = tuple(int(c) for c in rng.integers(0, 256, size=3))
        cv2.rectangle(img, p1, p2, color, -1)
    return img


def _gate_color(rng: np.random.Generator) -> np.ndarray:
    bright = int(rng.integers(0, 3))   # one vivid channel -> a distinct gate colour
    return np.array(
        [int(rng.integers(120, 256)) if i == bright else int(rng.integers(0, 160)) for i in range(3)],
        dtype=np.float64,
    )


def _draw_gate(img: np.ndarray, inner_px: np.ndarray, outer_px: np.ndarray,
               rng: np.random.Generator, level: int) -> None:
    """Paint the gate ring (outer minus inner squares). cv2.fillPoly clips polygons that run
    off the frame, so the close-range transit fill (ring bleeding to the edge) is automatic."""
    ring = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(ring, [outer_px.round().astype(np.int32)], 1)
    cv2.fillPoly(ring, [inner_px.round().astype(np.int32)], 0)
    mask = ring == 1
    if not mask.any():
        return
    color = _gate_color(rng)
    if level >= 2:
        # Directional luminance gradient (one side darker), RANDOM direction + strength, so the
        # detector learns invariance to lighting direction instead of expecting a uniform gate.
        ang = rng.uniform(0.0, 2.0 * np.pi)
        yy, xx = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]
        proj = xx * np.cos(ang) + yy * np.sin(ang)
        proj = (proj - proj.min()) / (np.ptp(proj) + 1e-9)
        lo = rng.uniform(0.35, 0.75)
        factor = (lo + (1.0 - lo) * proj)[mask]
    else:
        factor = np.full(int(mask.sum()), rng.uniform(0.75, 1.0))   # L1: uniform
    img[mask] = np.clip(color[None, :] * factor[:, None], 0, 255).astype(np.uint8)


def _draw_occluder(img: np.ndarray, inner_px: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Paint an opaque occluder near a random corner; return its boolean mask (for visibility)."""
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cx, cy = inner_px[int(rng.integers(0, 4))]
    w, h = float(rng.uniform(30, 130)), float(rng.uniform(30, 130))
    x0 = int(cx - w / 2 + rng.uniform(-25, 25))
    y0 = int(cy - h / 2 + rng.uniform(-25, 25))
    pt1, pt2 = (x0, y0), (x0 + int(w), y0 + int(h))
    cv2.rectangle(img, pt1, pt2, tuple(int(c) for c in rng.integers(0, 256, size=3)), -1)
    cv2.rectangle(mask, pt1, pt2, 1, -1)
    return mask.astype(bool)


def _corner_visibility(inner_px: np.ndarray, occ_mask: np.ndarray | None) -> np.ndarray:
    """Per-corner flag: off-frame -> V_OFF, under the occluder -> V_OCC, else V_VIS."""
    vis = np.full(4, V_VIS, dtype=int)
    for i, (x, y) in enumerate(inner_px):
        if not (0.0 <= x <= IMAGE_WIDTH - 1 and 0.0 <= y <= IMAGE_HEIGHT - 1):
            vis[i] = V_OFF
        elif occ_mask is not None and occ_mask[int(round(y)), int(round(x))]:
            vis[i] = V_OCC
    return vis


def _augment(img: np.ndarray, rng: np.random.Generator, level: int) -> np.ndarray:
    if level <= 1:
        return img                                     # L1: clean
    if rng.random() < 0.4:
        k = int(rng.choice([3, 5]))
        img = cv2.GaussianBlur(img, (k, k), 0)
    gain, bias = rng.uniform(0.7, 1.3), rng.uniform(-25, 25)
    img = np.clip(img.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
    if rng.random() < 0.7:
        img = np.clip(img.astype(np.float32) + rng.normal(0, rng.uniform(2, 12), img.shape), 0, 255).astype(np.uint8)
    if level >= 3:
        img = _chaos(img, rng)                          # L3: heavy
    return img


_ALB = None   # cached Albumentations pipeline, or False if unavailable


def _alb_pipeline():
    """Build (once) the L3 Albumentations pipeline (albumentations 2.x), or None if not installed.

    Deliberately NOT a stack of every effect: each image gets ~ONE primary sensor/motion
    degradation (a ``OneOf``), optionally ONE lighting/occlusion effect, plus a mild exposure
    jitter. Across the dataset every effect is covered; within any single image the gate stays
    detectable rather than buried -- and ultralytics layers its own augmentation on top at train
    time, so over-stacking here would double up and overwhelm the keypoint head."""
    global _ALB
    if _ALB is None:
        try:
            import albumentations as A
            _ALB = A.Compose([
                A.OneOf([                                            # one primary sensor/motion effect
                    A.MotionBlur(blur_limit=(3, 15)),
                    A.ImageCompression(quality_range=(25, 70)),      # JPEG artifacts (the stream IS jpeg)
                    A.Downscale(scale_range=(0.4, 0.85)),            # small gate at range
                    A.GaussNoise(std_range=(0.05, 0.2)),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),
                ], p=0.9),
                A.OneOf([                                            # at most one lighting/occlusion effect
                    A.RandomShadow(),
                    A.RandomSunFlare(src_radius=80),
                    A.RandomFog(fog_coef_range=(0.1, 0.4)),
                    A.CoarseDropout(num_holes_range=(1, 4),
                                    hole_height_range=(0.04, 0.12), hole_width_range=(0.04, 0.12)),
                ], p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.ChannelShuffle(p=0.05),
            ])
        except Exception:
            _ALB = False     # not installed / API mismatch -> the OpenCV fallback
    return _ALB or None


def _chaos(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """L3 heavy augmentation. Prefers Albumentations; falls back to an OpenCV pipeline."""
    alb = _alb_pipeline()
    if alb is not None:
        try:
            return alb(image=img)["image"]
        except Exception:
            pass
    return _chaos_cv2(img, rng)


def _chaos_cv2(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """OpenCV fallback for L3 when Albumentations is absent. Mirrors the pipeline's intent: ONE
    primary degradation per image + an optional lighting/occlusion effect + mild exposure."""
    h, w = img.shape[:2]
    primary = int(rng.integers(0, 4))
    if primary == 0:                                    # directional motion blur
        k = int(rng.integers(5, 16))
        kern = np.zeros((k, k), np.float32)
        kern[k // 2, :] = 1.0
        kern = cv2.warpAffine(kern, cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5),
                                                            float(rng.uniform(0, 180)), 1.0), (k, k))
        s = float(kern.sum())
        if s > 1e-6:
            img = cv2.filter2D(img, -1, kern / s)
    elif primary == 1:                                  # JPEG recompression (the stream IS jpeg/udp)
        ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(rng.integers(18, 70))])
        if ok:
            img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    elif primary == 2:                                  # downscale/upscale (small gates at range)
        sc = float(rng.uniform(0.4, 0.85))
        small = cv2.resize(img, (max(1, int(w * sc)), max(1, int(h * sc))), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    else:                                               # sensor noise
        img = np.clip(img.astype(np.float32) + rng.normal(0, rng.uniform(6, 20), img.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.4:                              # at most one lighting / occlusion effect
        if rng.random() < 0.5:                          # glare blob
            overlay = img.copy()
            cv2.circle(overlay, (int(rng.integers(0, w)), int(rng.integers(0, h))),
                       int(rng.integers(20, 90)), (255, 255, 255), -1)
            a = float(rng.uniform(0.15, 0.45))
            img = cv2.addWeighted(overlay, a, img, 1.0 - a, 0.0)
        else:                                           # a few occlusion boxes
            for _ in range(int(rng.integers(1, 4))):
                x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
                x1, y1 = min(w, x0 + int(rng.integers(12, 46))), min(h, y0 + int(rng.integers(12, 46)))
                img[y0:y1, x0:x1] = int(rng.integers(0, 255))
    img = np.clip(img.astype(np.float32) * rng.uniform(0.8, 1.2) + rng.uniform(-22, 22), 0, 255).astype(np.uint8)
    return img


def _bbox(outer_px: np.ndarray) -> np.ndarray:
    x0, y0 = outer_px.min(axis=0)
    x1, y1 = outer_px.max(axis=0)
    x0, y0 = max(0.0, float(x0)), max(0.0, float(y0))
    x1, y1 = min(float(IMAGE_WIDTH), float(x1)), min(float(IMAGE_HEIGHT), float(y1))
    return np.array([x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)])


def _in_frame(pts: np.ndarray) -> bool:
    return bool(
        (pts[:, 0] >= 0).all() and (pts[:, 0] <= IMAGE_WIDTH).all()
        and (pts[:, 1] >= 0).all() and (pts[:, 1] <= IMAGE_HEIGHT).all()
    )


def _in_ring(pt, outer_px: np.ndarray, inner_px: np.ndarray) -> bool:
    """True if ``pt`` falls on a gate's opaque ring (inside the outer square, outside the inner
    hole) -- i.e. a nearer gate at ``pt`` would occlude whatever is behind it there, while a point
    seen through the inner hole stays visible."""
    outer = outer_px.astype(np.float32).reshape(-1, 1, 2)
    if cv2.pointPolygonTest(outer, pt, False) < 0:
        return False
    inner = inner_px.astype(np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(inner, pt, False) < 0


def _sample_n_gates(rng: np.random.Generator, level: int, max_gates: int) -> int:
    """How many gates to place in one scene. Real frames show the next gate(s) beyond the current
    one, so single-gate dominates but multi-gate appears (light at L1, common at L2+) to teach
    instance separation and 'real gate vs gate-coloured clutter'."""
    if max_gates <= 1:
        return 1
    pool = [1, 1, 1, 1, 2] if level <= 1 else [1, 1, 2, 2, 3]
    pool = [k for k in pool if k <= max_gates]
    return int(rng.choice(pool))


def render_gate_sample(rng: np.random.Generator, level: int = 2, pose=None) -> SyntheticSample:
    R, t = sample_gate_pose(rng) if pose is None else pose
    blank = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    try:
        inner_px = project_gate_corners(R, t, GATE_INNER_SIZE_M)
        outer_px = project_gate_corners(R, t, GATE_OUTER_SIZE_M)
    except ValueError:
        return SyntheticSample(blank, np.zeros((4, 2)), np.zeros(4), False, R, t, np.zeros(4, int))

    img = _background(rng, level)
    _draw_gate(img, inner_px, outer_px, rng, level)
    occ_mask = None
    if level >= 2 and rng.random() < _OCCLUSION_PROB:
        occ_mask = _draw_occluder(img, inner_px, rng)
    img = _augment(img, rng, level)

    visibility = _corner_visibility(inner_px, occ_mask)
    centre = inner_px.mean(axis=0)
    centre_in = 0.0 <= centre[0] <= IMAGE_WIDTH and 0.0 <= centre[1] <= IMAGE_HEIGHT
    # Keep 3- or 4-corner gates: at least 3 corners clearly visible, gate centre in view.
    visible = bool(int((visibility == V_VIS).sum()) >= 3 and centre_in)
    return SyntheticSample(img, inner_px, _bbox(outer_px), visible, R, t, visibility)


def render_scene(rng: np.random.Generator, level: int = 2, max_gates: int = 3,
                 *, high_roll_prob: float = 0.0, edge_prob: float = 0.0):
    """Render a scene with 1..N gates sharing one image; return ``(image, [SyntheticSample,...])``.

    Gates are painted far->near so a nearer gate occludes a farther one; each gate's corners are
    flagged V_OCC when they fall on a nearer gate's ring (or the occluder), V_OFF when off-frame.
    Every sample carries the SAME image (write it once, one label row per visible gate).
    ``high_roll_prob`` / ``edge_prob`` oversample the hard configs (see ``sample_gate_pose``)."""
    n = _sample_n_gates(rng, level, max_gates)
    img = _background(rng, level)
    placed = []   # (R, t, inner_px, outer_px)
    attempts = 0
    while len(placed) < n and attempts < 4 * n:
        attempts += 1
        R, t = sample_gate_pose(rng, high_roll_prob=high_roll_prob, edge_prob=edge_prob)
        try:
            inner = project_gate_corners(R, t, GATE_INNER_SIZE_M)
            outer = project_gate_corners(R, t, GATE_OUTER_SIZE_M)
        except ValueError:
            continue
        c = inner.mean(axis=0)
        if any(np.linalg.norm(c - p[2].mean(axis=0)) < 60.0 for p in placed):
            continue   # keep gate centres visually distinct
        placed.append((R, t, inner, outer))
    if not placed:
        return img, []

    placed.sort(key=lambda p: -float(np.linalg.norm(p[1])))   # far first -> near drawn on top
    for _, _, inner, outer in placed:
        _draw_gate(img, inner, outer, rng, level)
    occ_mask = None
    if level >= 2 and rng.random() < _OCCLUSION_PROB:
        _, _, inner, _ = placed[int(rng.integers(0, len(placed)))]
        occ_mask = _draw_occluder(img, inner, rng)
    img = _augment(img, rng, level)

    samples = []
    for i, (R, t, inner, outer) in enumerate(placed):
        vis = _corner_visibility(inner, occ_mask)
        for j in range(i + 1, len(placed)):          # nearer gates occlude this one's corners
            _, _, inner_j, outer_j = placed[j]
            for c in range(4):
                if vis[c] != V_OFF and _in_ring((float(inner[c][0]), float(inner[c][1])), outer_j, inner_j):
                    vis[c] = V_OCC
        centre = inner.mean(axis=0)
        centre_in = 0.0 <= centre[0] <= IMAGE_WIDTH and 0.0 <= centre[1] <= IMAGE_HEIGHT
        visible = bool(int((vis == V_VIS).sum()) >= 3 and centre_in)
        samples.append(SyntheticSample(img, inner, _bbox(outer), visible, R, t, vis))
    return img, samples


def to_yolo_pose_label(sample: SyntheticSample, class_id: int = 0) -> str | None:
    """One ultralytics YOLO-pose row: ``class cx cy w h (x y v)*4`` (normalized). None if not
    usable. Off-frame corners are clamped into [0,1] and carry v=0; occluded corners carry v=1;
    clearly-visible corners v=2 -- training the detector to flag the weak corner the P3P
    fallback then drops."""
    if not sample.visible:
        return None
    x, y, w, h = sample.bbox_xywh
    fields = [class_id, (x + w / 2) / IMAGE_WIDTH, (y + h / 2) / IMAGE_HEIGHT,
              w / IMAGE_WIDTH, h / IMAGE_HEIGHT]
    for (px, py), v in zip(sample.keypoints_px, sample.visibility):
        nx = min(max(float(px) / IMAGE_WIDTH, 0.0), 1.0)
        ny = min(max(float(py) / IMAGE_HEIGHT, 0.0), 1.0)
        fields += [nx, ny, int(v)]
    return " ".join(f"{vv:.6g}" if isinstance(vv, float) else str(vv) for vv in fields)


DATA_YAML = """\
# Synthetic gate dataset for YOLO-pose (ultralytics).
path: {path}
train: images/train
val: images/val
names:
  0: gate
kpt_shape: [4, 3]   # 4 corners, (x, y, visibility)
flip_idx: [1, 0, 3, 2]   # horizontal flip swaps left<->right corners
"""


def write_dataset(out_dir, n_train: int, n_val: int, level=2, seed: int = 0,
                  max_gates: int = 3, high_roll_prob: float = 0.0, edge_prob: float = 0.0) -> Path:
    """Generate a YOLO-pose dataset (images + labels + data.yaml). Returns the data.yaml path.

    ``level`` is the curriculum level (1/2/3). Pass a SEQUENCE of levels (e.g. ``[1, 2, 2, 3, 3]``)
    to generate a MIXED-difficulty set, one level drawn per image -- the model then sees clean
    geometry through full chaos every epoch (a curriculum without catastrophic forgetting).

    ``max_gates`` caps gates-per-image: >1 (default) renders multi-gate scenes (a label row per
    visible gate) so the detector learns instance separation and real-gate-vs-clutter; pass 1 for
    the legacy single-gate behaviour. ``high_roll_prob`` / ``edge_prob`` oversample the high-roll
    and near-edge configs that dominate the corner-error tail."""
    out = Path(out_dir)
    rng = np.random.default_rng(seed)
    levels = list(level) if isinstance(level, (list, tuple, np.ndarray)) else None
    for split, n in (("train", n_train), ("val", n_val)):
        img_dir = out / "images" / split
        lbl_dir = out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        made = 0
        while made < n:
            lvl = int(rng.choice(levels)) if levels else level
            img, samples = render_scene(rng, level=lvl, max_gates=max_gates,
                                        high_roll_prob=high_roll_prob, edge_prob=edge_prob)
            rows = [r for r in (to_yolo_pose_label(s) for s in samples) if r is not None]
            if not rows:
                continue
            cv2.imwrite(str(img_dir / f"{made:06d}.png"), img)
            (lbl_dir / f"{made:06d}.txt").write_text("\n".join(rows) + "\n")
            made += 1
    yaml_path = out / "data.yaml"
    yaml_path.write_text(DATA_YAML.format(path=str(out.resolve())))
    return yaml_path
