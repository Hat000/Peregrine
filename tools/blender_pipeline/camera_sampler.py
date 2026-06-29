"""Sample plausible racing camera poses relative to a gate, as (R_cam_gate, t_cam_gate).

The output pose is in EXACTLY the convention ``racer.vision.gate_pose`` uses (gate pose in the
camera optical frame): ``p_cam = R_cam_gate @ p_gate + t_cam_gate``. So a sample feeds straight
into ``project_gate_corners`` (labels) and is the ground-truth that ``estimate_gate_pose``
recovers. We sample directly in this frame rather than building a world + drone pose, because the
harness is map-agnostic: the gate-RELATIVE pose is all the projector + PnP need.

Parameterization (each independently sampled, then composed):
  * range_m        -- gate-centre distance along the look direction (camera->gate).
  * bearing_deg    -- horizontal angle of the gate centre off the optical axis (+ = gate to the
                      right in the image). Kept within ~HFoV/2 so the gate centre is in frame.
  * elevation_deg  -- vertical angle of the gate centre off the optical axis (+ = gate DOWN in
                      image, matching optical +Y down). Kept within ~VFoV/2.
  * yaw/pitch/roll of the gate plane relative to the camera (gate obliquity + in-plane roll):
      gate_yaw_deg   -- gate rotated about its own vertical (left/right oblique view of the frame).
      gate_pitch_deg -- gate rotated about its own horizontal (looking up/down at the frame).
      gate_roll_deg  -- in-plane rotation of the square in the image (camera roll about optical Z).

Bias: most weight on the NEAR/MID band (where detection + PnP actually matter), but we
DELIBERATELY include far (> 15 m) and oblique samples to target the classical detector's weak
spots (far = thin core sliver; oblique = foreshortened square). ``visible_only`` re-samples until
all 4 corners project in front of the camera and inside the image (the trainable regime); set it
False to also emit clipped/edge poses (the P3P / drop regime) for stress coverage.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

# Canonical intrinsics/FoV from the deployed stack -- never re-typed here.
from racer.frames import IMAGE_HEIGHT, IMAGE_WIDTH, horizontal_fov_deg, vertical_fov_deg
from racer.vision.gate_pose import GATE_INNER_SIZE_M, project_gate_corners


@dataclass(frozen=True)
class CameraPose:
    """A gate pose in the camera optical frame -- the (R, t) ``gate_pose`` consumes.

    ``p_cam = R_cam_gate @ p_gate + t_cam_gate``. Also carries the scalar sampling params for
    bookkeeping / stratification, so a dataset can record WHY a sample was hard."""

    R_cam_gate: np.ndarray            # (3,3)
    t_cam_gate: np.ndarray            # (3,) gate centre in camera optical coords [x_right, y_down, z_fwd]
    range_m: float
    bearing_deg: float
    elevation_deg: float
    gate_yaw_deg: float
    gate_pitch_deg: float
    gate_roll_deg: float


@dataclass(frozen=True)
class SamplerConfig:
    """Bounds for the racing-camera pose sampler. Defaults bias near/mid but reach far+oblique."""

    range_min_m: float = 2.0
    range_max_m: float = 25.0
    range_bias: float = 1.7          # >1 oversamples NEAR (power-law on U): r = lo + (hi-lo)*U**bias
    far_frac: float = 0.18           # fraction of samples forced into the FAR band (> far_thresh_m)
    far_thresh_m: float = 15.0
    bearing_max_frac: float = 0.42   # |bearing| <= this * (HFoV/2): keep gate centre in frame
    elevation_max_frac: float = 0.42  # |elevation| <= this * (VFoV/2)
    oblique_max_deg: float = 55.0    # gate yaw/pitch magnitude cap (obliquity)
    oblique_frac: float = 0.30       # fraction forced toward high obliquity (detector weak spot)
    roll_max_deg: float = 35.0       # in-plane camera roll
    inner_size_m: float = GATE_INNER_SIZE_M


def _sample_range(rng: np.random.Generator, cfg: SamplerConfig) -> float:
    if rng.random() < cfg.far_frac:
        return float(rng.uniform(cfg.far_thresh_m, cfg.range_max_m))
    u = rng.random() ** cfg.range_bias  # power-law toward the low (near) end
    return float(cfg.range_min_m + (cfg.range_max_m - cfg.range_min_m) * u)


def _signed_capped(rng: np.random.Generator, cap_deg: float, push_frac: float,
                   push_cap_deg: float) -> float:
    """Sample a signed angle in [-cap, cap]; with prob push_frac bias toward |angle| high."""
    if rng.random() < push_frac:
        mag = rng.uniform(0.6 * push_cap_deg, push_cap_deg)
    else:
        mag = rng.uniform(0.0, cap_deg)
    return float(mag * (1.0 if rng.random() < 0.5 else -1.0))


def sample_pose(rng: np.random.Generator, cfg: SamplerConfig | None = None) -> CameraPose:
    """Draw one racing camera pose relative to a gate (gate-in-camera optical frame)."""
    cfg = cfg or SamplerConfig()
    hfov_half = horizontal_fov_deg() / 2.0
    vfov_half = vertical_fov_deg() / 2.0

    rng_m = _sample_range(rng, cfg)
    bearing = float(rng.uniform(-1, 1) * cfg.bearing_max_frac * hfov_half)
    elevation = float(rng.uniform(-1, 1) * cfg.elevation_max_frac * vfov_half)

    # Gate-centre direction in camera optical coords: az=bearing about +Y(down), el=elevation.
    # A point at range r along that direction (optical: x right, y down, z forward).
    az = np.deg2rad(bearing)
    el = np.deg2rad(elevation)
    direction = np.array([
        np.sin(az) * np.cos(el),   # x right
        np.sin(el),                # y down
        np.cos(az) * np.cos(el),   # z forward
    ])
    t_cam_gate = rng_m * direction

    # Gate-plane orientation relative to camera. Start head-on (R = I => gate +Z downrange ==
    # camera +Z, so IPPE sees a normal pointing AWAY, the valid regime), then apply gate obliquity
    # (yaw about gate-Y, pitch about gate-X) and in-plane roll (about gate-Z == optical-Z roll).
    gate_yaw = _sample_signed_oblique(rng, cfg)
    gate_pitch = _sample_signed_oblique(rng, cfg)
    gate_roll = float(rng.uniform(-1, 1) * cfg.roll_max_deg)
    # Intrinsic gate-frame rotation: roll(Z) then pitch(X) then yaw(Y), composed as camera<-gate.
    R_cam_gate = Rotation.from_euler(
        "ZXY", [gate_roll, gate_pitch, gate_yaw], degrees=True
    ).as_matrix()

    return CameraPose(
        R_cam_gate=R_cam_gate, t_cam_gate=t_cam_gate, range_m=rng_m,
        bearing_deg=bearing, elevation_deg=elevation,
        gate_yaw_deg=gate_yaw, gate_pitch_deg=gate_pitch, gate_roll_deg=gate_roll,
    )


def _sample_signed_oblique(rng: np.random.Generator, cfg: SamplerConfig) -> float:
    return _signed_capped(rng, cfg.oblique_max_deg, cfg.oblique_frac, cfg.oblique_max_deg)


def corners_in_frame(pose: CameraPose, cfg: SamplerConfig | None = None,
                     margin_px: float = 0.0) -> bool:
    """True iff all 4 inner corners project in front of the camera AND inside the image
    (optionally inset by ``margin_px``). Uses the canonical projector so 'in frame' here is
    EXACTLY what the labels/PnP see."""
    cfg = cfg or SamplerConfig()
    try:
        px = project_gate_corners(pose.R_cam_gate, pose.t_cam_gate, cfg.inner_size_m)
    except ValueError:
        return False  # a corner at/behind the camera
    if not np.isfinite(px).all():
        return False
    x_ok = (px[:, 0] >= margin_px) & (px[:, 0] <= IMAGE_WIDTH - 1 - margin_px)
    y_ok = (px[:, 1] >= margin_px) & (px[:, 1] <= IMAGE_HEIGHT - 1 - margin_px)
    return bool(np.all(x_ok & y_ok))


def sample_poses(n: int, *, seed: int = 0, cfg: SamplerConfig | None = None,
                 visible_only: bool = True, max_tries_per: int = 64) -> list[CameraPose]:
    """Sample ``n`` poses. With ``visible_only`` (default) each pose has all 4 corners in frame
    (the trainable IPPE regime); a pose that can't be made visible within ``max_tries_per`` draws
    is skipped (so the list may be shorter than ``n`` only in pathological configs). Deterministic
    given ``seed``."""
    cfg = cfg or SamplerConfig()
    rng = np.random.default_rng(seed)
    out: list[CameraPose] = []
    for _ in range(n):
        if not visible_only:
            out.append(sample_pose(rng, cfg))
            continue
        for _try in range(max_tries_per):
            p = sample_pose(rng, cfg)
            if corners_in_frame(p, cfg, margin_px=1.0):
                out.append(p)
                break
    return out
