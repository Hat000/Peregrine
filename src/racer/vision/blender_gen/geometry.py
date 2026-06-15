"""Camera-viewpoint + gate-geometry sampling for the photoreal dataset (pure-Python, no bpy).

Produces, per frame, the camera body pose + each visible gate's pose in the CAMERA OPTICAL
frame ``(R_cam_gate, t_cam_gate)`` plus its pixel keypoints / bbox / per-corner visibility --
everything the render backend needs to place geometry and everything the label writer needs.

Viewpoints are sampled on the REAL VQ1 course (``gates_from_track_records`` -> the exact gate
world poses the deployed stack uses), so range distribution, the 26 m descent, gate facing and
down-course co-visibility match deployment instead of a synthetic prior. The optical-pose math
mirrors ``frames.world_point_in_camera`` exactly, so labels = ``project_gate_corners`` and the
detector trains on what PnP consumes.

Occlusion + clipping reuse ``racer.vision.synthetic`` helpers (a nearer gate's ring hides a
farther gate's corner -> V_OCC; off-frame -> V_OFF), so the visibility semantics are identical
to the procedural generator the detector was first trained on.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from racer.frames import R_camera_from_body, R_world_from_body
from racer.navigator import gates_from_track_records
from racer.vision.synthetic import _bbox, _corner_visibility, _in_ring
from racer.contracts import Gate

from .contract import (
    GATE_INNER_SIZE_M,
    GATE_OUTER_SIZE_M,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    RANGE_MAX_M,
    RANGE_MIN_M,
    V_OCC,
    V_OFF,
    V_VIS,
    project_gate_corners,
)


@dataclass(frozen=True)
class ViewpointConfig:
    """The envelope a preset samples camera viewpoints over (the realism knobs)."""

    range_min_m: float = RANGE_MIN_M
    range_max_m: float = RANGE_MAX_M
    lateral_sigma_m: float = 1.2          # cross-track camera offset (m, 1-sigma)
    vertical_sigma_m: float = 0.8         # vertical camera offset (m, 1-sigma)
    yaw_jitter_rad: float = 0.30          # heading error vs straight-at-gate
    pitch_jitter_rad: float = 0.25        # body pitch jitter (camera adds its fixed +20 uptilt)
    roll_jitter_rad: float = 0.35         # body roll (racing bank)
    speed_min_mps: float = 4.0            # for motion-blur magnitude
    speed_max_mps: float = 30.0
    range_skew: float = 2.0               # >1 biases CLOSE (d = min + (max-min)*u**skew)
    multi_gate: bool = True               # label every gate in view, not just the target


@dataclass
class GateRender:
    """One gate as the backend should render it + as it should be labelled. Optical frame."""

    gate_id: int
    R_cam_gate: np.ndarray        # (3,3) gate-frame -> camera optical
    t_cam_gate: np.ndarray        # (3,) gate-opening centre in camera optical (m)
    keypoints_px: np.ndarray      # (4,2) inner-square corners, canonical order
    outer_px: np.ndarray          # (4,2) outer-square corners (bbox source)
    bbox_xywh: np.ndarray         # (4,) [x,y,w,h] clipped to frame
    visibility: np.ndarray        # (4,) V_VIS / V_OCC / V_OFF
    visible: bool                 # usable label: >= 3 corners visible AND centre in frame

    @property
    def range_m(self) -> float:
        return float(np.linalg.norm(self.t_cam_gate))


@dataclass
class FrameSpec:
    """All gates considered for one rendered frame + the camera body pose (for the backend's
    motion blur / lighting placement). ``labeled_gates`` are the ones that get a label row."""

    body_pos_ned: np.ndarray
    roll: float
    pitch: float
    yaw: float
    body_vel_ned: np.ndarray
    gates: list[GateRender] = field(default_factory=list)

    @property
    def labeled_gates(self) -> list[GateRender]:
        return [g for g in self.gates if g.visible]


def load_track_records(path: str | None = None) -> list[dict]:
    """Track records (gate_id, position_ned, orientation_ned_wxyz, width/height). Defaults to
    the baked VQ1 course (``gate_mapper_synth.VQ1_TRACK_RECORDS``); pass a track_map.json path
    to use a different/real course."""
    if path is None:
        from racer.gate_mapper_synth import VQ1_TRACK_RECORDS
        return [dict(r) for r in VQ1_TRACK_RECORDS]
    import json
    from pathlib import Path

    data = json.loads(Path(path).read_text())
    return list(data["gates"]) if isinstance(data, dict) else list(data)


def load_course_gates(path: str | None = None) -> list[Gate]:
    """Ordered :class:`Gate` world poses (opening centre + R_world_gate), via the canonical
    ``navigator.gates_from_track_records(corner_to_center=True)`` used by the deployed stack."""
    return gates_from_track_records(load_track_records(path), corner_to_center=True)


def optical_pose(gate: Gate, body_pos_ned: np.ndarray, R_wb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gate world pose -> camera-optical ``(R_cam_gate, t_cam_gate)``. Mirrors
    ``frames.world_point_in_camera``: ``p_cam = R_camera_from_body @ R_wb.T @ (p_world - body)``
    so a gate point p_g -> R_cam_gate @ p_g + t_cam_gate."""
    R_cam_world = R_camera_from_body() @ R_wb.T
    R_cam_gate = R_cam_world @ gate.R_world_gate
    t_cam_gate = R_cam_world @ (gate.position_ned - body_pos_ned)
    return R_cam_gate, t_cam_gate


def _course_heading(gates: list[Gate], i: int) -> np.ndarray:
    """Unit down-course direction at gate ``i`` (segment toward the next gate)."""
    n = len(gates)
    j = i + 1 if i + 1 < n else i
    seg = gates[j].position_ned - gates[i - 1 if j == i else i].position_ned
    nrm = float(np.linalg.norm(seg))
    return seg / nrm if nrm > 1e-6 else np.array([-1.0, 0.0, 0.0])


def _build_gate_render(gate: Gate, R_cam_gate: np.ndarray, t_cam_gate: np.ndarray) -> GateRender | None:
    """Project + bbox + (pre-occlusion) visibility for one gate. None if it cannot project
    (a corner at/behind the camera)."""
    try:
        inner_px = project_gate_corners(R_cam_gate, t_cam_gate, GATE_INNER_SIZE_M)
        outer_px = project_gate_corners(R_cam_gate, t_cam_gate, GATE_OUTER_SIZE_M)
    except ValueError:
        return None
    vis = _corner_visibility(inner_px, None)
    return GateRender(
        gate_id=int(gate.gate_id), R_cam_gate=R_cam_gate, t_cam_gate=t_cam_gate,
        keypoints_px=inner_px, outer_px=outer_px, bbox_xywh=_bbox(outer_px),
        visibility=vis, visible=False,
    )


def _apply_occlusion_and_visibility(renders: list[GateRender]) -> None:
    """In-place: mark a corner V_OCC when a NEARER gate's ring covers it, then set ``visible``
    (>= 3 corners V_VIS and centre in frame). Same far->near logic as synthetic.render_scene."""
    order = sorted(range(len(renders)), key=lambda k: renders[k].range_m)  # near first
    for a, ia in enumerate(order):
        ga = renders[ia]
        for ib in order[:a]:                                              # strictly nearer gates
            gb = renders[ib]
            for c in range(4):
                if ga.visibility[c] != V_OFF and _in_ring(
                    (float(ga.keypoints_px[c, 0]), float(ga.keypoints_px[c, 1])),
                    gb.outer_px, gb.keypoints_px,
                ):
                    ga.visibility[c] = V_OCC
        centre = ga.keypoints_px.mean(axis=0)
        centre_in = 0.0 <= centre[0] <= IMAGE_WIDTH and 0.0 <= centre[1] <= IMAGE_HEIGHT
        ga.visible = bool(int((ga.visibility == V_VIS).sum()) >= 3 and centre_in)


def sample_frame(rng: np.random.Generator, gates: list[Gate], cfg: ViewpointConfig) -> FrameSpec:
    """Sample one camera viewpoint on the course + every gate's optical pose/label for it.

    Camera position = a point ``d`` metres up-course from a randomly chosen target gate (``d``
    skewed close), plus cross-track + vertical jitter; heading roughly toward the target gate
    with yaw/pitch/roll jitter; velocity down-course (for motion blur). Multiple gates can fall
    in view -> realistic co-visibility + the descent.
    """
    n = len(gates)
    ti = int(rng.integers(0, n))
    target = gates[ti]
    heading = _course_heading(gates, ti)                                  # down-course unit (NED)
    d = cfg.range_min_m + (cfg.range_max_m - cfg.range_min_m) * rng.random() ** cfg.range_skew

    # Camera sits up-course of the target; offset laterally (cross-heading horizontal) + vertically.
    horiz = np.array([heading[0], heading[1], 0.0])
    hn = float(np.linalg.norm(horiz))
    horiz = horiz / hn if hn > 1e-6 else np.array([-1.0, 0.0, 0.0])
    left = np.array([-horiz[1], horiz[0], 0.0])                           # horizontal cross-track
    body_pos = (
        target.position_ned - d * horiz
        + rng.normal(0.0, cfg.lateral_sigma_m) * left
        + np.array([0.0, 0.0, rng.normal(0.0, cfg.vertical_sigma_m)])
    )

    to_gate = target.position_ned - body_pos
    yaw0 = float(np.arctan2(to_gate[1], to_gate[0]))
    # pitch toward the gate (NED: +pitch noses DOWN if target below). Camera adds its fixed +20
    # uptilt in R_camera_from_body, so the BODY pitch here is small and jittered.
    pitch0 = float(np.arctan2(-to_gate[2], np.linalg.norm(to_gate[:2])))
    yaw = yaw0 + float(rng.normal(0.0, cfg.yaw_jitter_rad))
    pitch = 0.4 * pitch0 + float(rng.normal(0.0, cfg.pitch_jitter_rad))
    roll = float(rng.normal(0.0, cfg.roll_jitter_rad))
    R_wb = R_world_from_body(roll, pitch, yaw)

    speed = float(rng.uniform(cfg.speed_min_mps, cfg.speed_max_mps))
    body_vel_ned = speed * (to_gate / (np.linalg.norm(to_gate) + 1e-9))

    renders: list[GateRender] = []
    consider = gates if cfg.multi_gate else [target]
    for g in consider:
        R_cg, t_cg = optical_pose(g, body_pos, R_wb)
        if t_cg[2] <= 0:                                                  # behind camera
            continue
        rng_m = float(np.linalg.norm(t_cg))
        if not (cfg.range_min_m * 0.6 <= rng_m <= cfg.range_max_m * 1.4):  # soft envelope band
            continue
        gr = _build_gate_render(g, R_cg, t_cg)
        if gr is not None:
            renders.append(gr)

    _apply_occlusion_and_visibility(renders)
    return FrameSpec(
        body_pos_ned=body_pos, roll=roll, pitch=pitch, yaw=yaw,
        body_vel_ned=body_vel_ned, gates=renders,
    )


def sample_negative_frame(rng: np.random.Generator, gates: list[Gate], cfg: ViewpointConfig) -> FrameSpec:
    """A HARD-NEGATIVE frame: a real on-course camera viewpoint with NO labelled gate (the backend
    still renders the randomized background + clutter/confusers). Used for the ~25-30% negative arm
    that teaches the detector not to fire on gate-coloured rectangles / arena structure."""
    fs = sample_frame(rng, gates, cfg)
    return FrameSpec(
        body_pos_ned=fs.body_pos_ned, roll=fs.roll, pitch=fs.pitch, yaw=fs.yaw,
        body_vel_ned=fs.body_vel_ned, gates=[],
    )


def sample_negative_frames(n_frames: int, cfg: ViewpointConfig, *, seed: int = 0,
                           track_path: str | None = None):
    """Yield ``n_frames`` hard-negative FrameSpecs (no labelled gate)."""
    rng = np.random.default_rng(seed)
    gates = load_course_gates(track_path)
    for _ in range(n_frames):
        yield sample_negative_frame(rng, gates, cfg)


def sample_frames(
    n_frames: int, cfg: ViewpointConfig, *, seed: int = 0, track_path: str | None = None,
    require_label: bool = True,
):
    """Yield ``n_frames`` FrameSpecs with >= 1 labelled gate (when ``require_label``)."""
    rng = np.random.default_rng(seed)
    gates = load_course_gates(track_path)
    made = 0
    guard = 0
    while made < n_frames and guard < 100 * n_frames + 100:
        guard += 1
        fs = sample_frame(rng, gates, cfg)
        if require_label and not fs.labeled_gates:
            continue
        made += 1
        yield fs
