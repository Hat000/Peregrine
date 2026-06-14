"""rl/fix_surrogate.py -- analytic, NO-RENDER fix surrogate for inc8 camera-pointing training.

WHY. inc8 must train the policy to keep the gate in the camera (camera-pointing is the binding
lever), but the detector cannot be rendered at thousands-of-envs training scale. This module is the
enabler: given GROUND-TRUTH relative gate geometry -- range, bearing, in-FoV, viewing angle, ALL free
in the sim -- it returns P(accept) and a fix covariance sigma(geometry), so a synthetic gate-relative
position fix can be sampled and fed straight to ``racer.state_estimator.LinearKF`` WITHOUT rendering.

It is INFRASTRUCTURE, not a reward. It answers "if the camera is pointed HERE, do we get a fix and how
good is it?" -- nothing about how to reward that. The reward design is owned elsewhere.

CALIBRATION. Fit to the Track-3 vision-at-speed shadow recordings (handoff/simops-mastery-2026-06-13,
6 flights / 1821 frames / 126 accepted fixes); see handoff/fix-surrogate-2026-06-14/REPORT.md and
fix_surrogate_fit.py. The baked defaults below ARE the fitted values (checkpoints under
handoff/fix-surrogate-2026-06-14/models/); ``FixSurrogate.from_checkpoints()`` reloads them so a
recalibration (e.g. an at-speed L3 recording) is a one-file swap. Reproduces the MEASURED numbers:
  * accept = a sharp range BAND-PASS: ~0 below 16 m (frontal-PnP flips / partial gates) and above
    ~28 m (the navigator range cap), peaking ~0.83 in the 18-26 m offered window.
  * per-fix 1-sigma (binding 10-26 m band): lateral(in-plane cross-track) 0.10 m, vertical(in-plane)
    0.28 m, depth(along-track) 0.85 m -- the MEASURED Track-3 noise (NOT the conservative 0.265 the
    estimator ships).
  * P(accept | a gate is in image) ~ 0.10, flat in crab (a measured assumption -- see report).

COVARIANCE SHAPE mirrors the deployed gate-relative fix (racer.localization.gate_relative_inplane_fix):
a gate-PLANE anisotropic diagonal (in-plane lateral/vertical + along-track depth) rotated to world NED,
so ``sample_fix`` -> (z_ned, cov_ned) drops straight into ``LinearKF.update_position`` / RewindKF.

Pure numpy + the torch-free racer geometry helpers (frames/contracts) -> CPU-only, no GPU dependency.
This module does NOT import or modify any estimator/deploy file.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

# racer geometry/contract helpers (torch-free) -- add src/ like the other rl/ entrypoints do, so the
# surrogate's in-image / bearing computation is BYTE-consistent with the deployed camera chain (same
# intrinsics, same +20-deg mount, same R_y(pi) odo convention). A surrogate whose in-FoV disagreed with
# the real detector would teach the policy the wrong pointing.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from racer.contracts import Gate                                       # noqa: E402
from racer.frames import (                                             # noqa: E402
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    R_camera_from_body,
    project_camera_point,
    vertical_fov_deg,
)

_CKPT_DIR = Path(__file__).resolve().parent.parent / "handoff/fix-surrogate-2026-06-14/models"
HFOV_HALF_DEG = 45.0
VFOV_HALF_DEG = float(vertical_fov_deg() / 2.0)          # 29.36 deg


# ============================================================================ geometry
@dataclass(frozen=True)
class GateGeometry:
    """GT relative geometry of one gate from the drone -- everything the surrogate needs, all free in
    the sim. ``in_image`` is the deterministic camera-pointing gate (gate centre projects inside the
    640x360 frame); ``range_m`` drives the accept band-pass; the gate frame shapes the covariance."""
    range_m: float
    azimuth_deg: float        # horizontal off-boresight (+ = gate to the right); the yaw/crab axis
    elevation_deg: float      # vertical off-boresight (+ = gate below boresight); the pitch axis
    bearing_deg: float        # total off-boresight angle
    in_image: bool            # gate centre projects inside the image rect (in front of camera)
    view_angle_deg: float     # angle between the drone->gate ray and the gate normal (0 = head-on)
    t_cam: np.ndarray         # (3,) gate origin in the camera frame
    lever_world: np.ndarray   # (3,) +L = gate_pos - drone_pos, world NED
    R_world_gate: np.ndarray  # (3,3) gate frame -> world NED (cov shaping)
    gate_position_ned: np.ndarray  # (3,) gate centre, world NED


def geometry(drone_position_ned: np.ndarray, R_world_body: np.ndarray, gate: Gate) -> GateGeometry:
    """Compute the GT relative geometry of ``gate`` from the drone pose -- mirrors build_dataset.py /
    the navigator chain exactly (same R_camera_from_body, same projection)."""
    drone = np.asarray(drone_position_ned, dtype=np.float64)
    R_wc = np.asarray(R_world_body, dtype=np.float64) @ R_camera_from_body().T   # world <- camera
    lever = np.asarray(gate.position_ned, dtype=np.float64) - drone              # +L, world NED
    rng = float(np.linalg.norm(lever))
    t_cam = R_wc.T @ lever                                                       # gate in camera frame
    tx, ty, tz = float(t_cam[0]), float(t_cam[1]), float(t_cam[2])
    az = float(np.degrees(np.arctan2(tx, tz)))
    el = float(np.degrees(np.arctan2(ty, tz)))
    bearing = float(np.degrees(np.arctan2(np.hypot(tx, ty), tz)))
    proj = project_camera_point(t_cam)
    in_img = bool(proj is not None and 0.0 <= proj[0] < IMAGE_WIDTH and 0.0 <= proj[1] < IMAGE_HEIGHT)
    normal = np.asarray(gate.R_world_gate, dtype=np.float64)[:, 2]
    view = (float(np.degrees(np.arccos(np.clip(abs(float((lever / rng) @ normal)), 0.0, 1.0))))
            if rng > 1e-9 else float("nan"))
    return GateGeometry(
        range_m=rng, azimuth_deg=az, elevation_deg=el, bearing_deg=bearing, in_image=in_img,
        view_angle_deg=view, t_cam=t_cam, lever_world=lever,
        R_world_gate=np.asarray(gate.R_world_gate, dtype=np.float64),
        gate_position_ned=np.asarray(gate.position_ned, dtype=np.float64),
    )


# ============================================================================ the surrogate
@dataclass(frozen=True)
class FixSurrogate:
    """Calibrated analytic fix model. Defaults = the Track-3-fitted checkpoint values; swap via
    ``from_checkpoints``. All three sub-models are independent and individually swappable."""

    # -- A. accept: range band-pass  p = pmax * sig((r-rlo)/wlo) * sig((rhi-r)/whi) -----------------
    accept_pmax: float = 0.8419703950411151
    accept_rlo: float = 16.191103551308505
    accept_wlo: float = 1.0000000000000007
    accept_rhi: float = 28.137694558909192
    accept_whi: float = 1.0000000000000002
    accept_p_out_of_image: float = 0.0002098635886673662   # rare partial-gate edge catch off-frame
    # Defensive valid-range guard [adversarial range-extrapolation 2026-06-14]: hard-zero accept
    # outside this window so a far/absurd-range query cannot return a spurious non-zero accept. The
    # window is GENEROUS (keeps the calibrated rising/falling edges 10-32 m -- which carry real signal,
    # emp 0.18 at 10-18 m / 0.69 at 24-32 m); it only kills the asymptotic tails (band-pass < ~0.002).
    accept_range_guard_lo: float = 9.0
    accept_range_guard_hi: float = 35.0

    # -- B. sigma(range): per-axis 1-sigma = max(floor, a1*range), gate-frame (m) -------------------
    sigma_lateral_floor: float = 0.10447100671864633
    sigma_lateral_a1: float = 0.0027626940998010047
    sigma_lateral_bias: float = -0.033817701667839546
    sigma_vertical_floor: float = 0.2815829316354898
    sigma_vertical_a1: float = 0.0
    sigma_vertical_bias: float = 0.19488467958140898
    sigma_depth_floor: float = 0.8524006087680642
    sigma_depth_a1: float = 0.0
    sigma_depth_bias: float = -0.33441613167150736
    # a1 (esp. lateral) extrapolates BADLY past ~24 m (localization carry-note); clamp the range used
    # for the growth term so a far-range query cannot inflate sigma unphysically.
    sigma_growth_max_range_m: float = 30.0
    cov_spd_floor: float = 1e-6        # tiny isotropic add so cov_ned is strictly SPD

    # -- C. crab map: (crab_deg -> active_in_fov_frac, accept_rate) ---------------------------------
    crab_inc7_deg: float = 66.0
    p_accept_active_in_image: float = 0.09844559585492228
    crab_table: tuple = field(default=(
        (0.0, 0.144), (5.0, 0.144), (10.0, 0.144), (15.0, 0.144), (20.0, 0.144), (25.0, 0.143),
        (30.0, 0.143), (35.0, 0.142), (40.0, 0.140), (45.0, 0.136), (50.0, 0.132), (55.0, 0.122),
        (60.0, 0.098), (65.0, 0.065), (70.0, 0.014),
    ))
    # provenance (the inc7 anchors the calibration reproduces; not used in sampling)
    any_gate_coverage_inc7: float = 0.3322350356946733
    any_gate_accept_rate_inc7: float = 0.06919275123558484
    pointing_ceiling_accept_in_window: float = 0.8419703950411151   # az&el centred through 18-28 m window
    pointing_gain_in_window: float = 8.539985435417025

    # ---- construction -------------------------------------------------------
    @classmethod
    def from_checkpoints(cls, ckpt_dir: Path | str = _CKPT_DIR) -> "FixSurrogate":
        """Load the fitted sub-model checkpoints (accept.json / sigma.json / crab.json). Missing files
        fall back to the baked defaults, so a partial recalibration still yields a usable surrogate."""
        d = Path(ckpt_dir)
        kw: dict = {}
        a = _maybe(d / "accept.json")
        if a:
            bp = a["bandpass_params"]
            kw.update(accept_pmax=bp[0], accept_rlo=bp[1], accept_wlo=bp[2], accept_rhi=bp[3],
                      accept_whi=bp[4], accept_p_out_of_image=a["p_accept_out_of_image"])
        s = _maybe(d / "sigma.json")
        if s:
            kw.update(
                sigma_lateral_floor=s["lateral"]["floor"], sigma_lateral_a1=s["lateral"]["a1"],
                sigma_lateral_bias=s["lateral"]["bias_band"],
                sigma_vertical_floor=s["vertical"]["floor"], sigma_vertical_a1=s["vertical"]["a1"],
                sigma_vertical_bias=s["vertical"]["bias_band"],
                sigma_depth_floor=s["depth"]["floor"], sigma_depth_a1=s["depth"]["a1"],
                sigma_depth_bias=s["depth"]["bias_band"],
            )
        c = _maybe(d / "crab.json")
        if c:
            kw.update(crab_inc7_deg=c["crab_inc7_deg"],
                      p_accept_active_in_image=c["p_accept_active_in_image"],
                      crab_table=tuple((t["crab_deg"], t["active_in_fov"]) for t in c["table"]),
                      any_gate_coverage_inc7=c["any_gate_coverage"],
                      any_gate_accept_rate_inc7=c["any_gate_accept_rate"],
                      pointing_ceiling_accept_in_window=c["elevation_diag"]["pointing_ceiling_accept_in_window"],
                      pointing_gain_in_window=c["elevation_diag"]["pointing_gain"])
        return replace(cls(), **kw)

    # ---- A. accept ----------------------------------------------------------
    def accept_prob_in_image(self, range_m):
        """P(accept | gate IS in image) -- the range band-pass, hard-zeroed outside the valid-range
        guard. Vectorised (range_m may be an array). NB acceptance here already folds in the in-image
        -> offered (association + depth-sanity + range-cap) drop; it is NOT a competition-among-gates
        effect [adversarial competition-artifact 2026-06-14]."""
        r = np.asarray(range_m, dtype=np.float64)
        z_lo = np.clip((r - self.accept_rlo) / self.accept_wlo, -30, 30)
        z_hi = np.clip((self.accept_rhi - r) / self.accept_whi, -30, 30)
        p = self.accept_pmax / ((1.0 + np.exp(-z_lo)) * (1.0 + np.exp(-z_hi)))
        guard = (r >= self.accept_range_guard_lo) & (r <= self.accept_range_guard_hi)
        return p * guard

    def p_accept(self, geom: GateGeometry) -> float:
        """P(accept) for one gate at this geometry: the band-pass GATED by in-image (the camera-pointing
        lever). A gate the camera is not pointed at almost never yields a fix."""
        if geom.in_image:
            return float(self.accept_prob_in_image(geom.range_m))
        return float(self.accept_p_out_of_image)

    # ---- B. sigma -----------------------------------------------------------
    def fix_sigma(self, geom: GateGeometry) -> tuple[float, float, float]:
        """(sigma_lateral, sigma_vertical, sigma_depth) gate-frame 1-sigma at this range (m)."""
        r = min(float(geom.range_m), self.sigma_growth_max_range_m)
        return (
            max(self.sigma_lateral_floor, self.sigma_lateral_a1 * r),
            max(self.sigma_vertical_floor, self.sigma_vertical_a1 * r),
            max(self.sigma_depth_floor, self.sigma_depth_a1 * r),
        )

    def fix_covariance(self, geom: GateGeometry) -> np.ndarray:
        """3x3 world-NED fix covariance: gate-plane anisotropic diagonal rotated to world (mirrors
        gate_relative_inplane_fix). Strictly SPD."""
        sl, sv, sd = self.fix_sigma(geom)
        cov_gate = np.diag([sl * sl, sv * sv, sd * sd])
        Rwg = geom.R_world_gate
        return Rwg @ cov_gate @ Rwg.T + self.cov_spd_floor * np.eye(3)

    # ---- D. sample ----------------------------------------------------------
    def sample_fix(self, geom: GateGeometry, rng: np.random.Generator,
                   include_bias: bool = True, force: bool = False
                   ) -> tuple[np.ndarray, np.ndarray] | None:
        """Bernoulli(P(accept)); on accept return (z_ned, cov_ned) for LinearKF.update_position, else
        None. ``z`` is the TRUE drone world position + a gate-frame noise draw (in-plane lateral/
        vertical + along-track depth, optional small systematic bias); ``cov`` is the matching SPD
        covariance. The true drone position is reconstructed from the geometry (z = gate - lever),
        so the caller need only pass the geometry. ``force`` bypasses the accept draw (always sample),
        for deterministic covariance-consistency tests."""
        if not force and rng.random() >= self.p_accept(geom):
            return None
        sl, sv, sd = self.fix_sigma(geom)
        sig = np.array([sl, sv, sd], dtype=np.float64)
        bias = (np.array([self.sigma_lateral_bias, self.sigma_vertical_bias, self.sigma_depth_bias])
                if include_bias else np.zeros(3))
        noise_gate = bias + sig * rng.standard_normal(3)
        Rwg = geom.R_world_gate
        drone_true = geom.gate_position_ned - geom.lever_world
        z = drone_true + Rwg @ noise_gate
        cov = Rwg @ np.diag(sig * sig) @ Rwg.T + self.cov_spd_floor * np.eye(3)
        return z, cov

    def sample_fix_from_state(self, drone_position_ned: np.ndarray, R_world_body: np.ndarray,
                              gate: Gate, rng: np.random.Generator, **kw
                              ) -> tuple[np.ndarray, np.ndarray] | None:
        """Convenience: compute the geometry from raw sim state then sample. Returns None if the gate
        is not in image and the (tiny) off-image accept draw fails."""
        return self.sample_fix(geometry(drone_position_ned, R_world_body, gate), rng, **kw)

    # ---- C. crab map --------------------------------------------------------
    def crab_to_fix_rate(self, crab_deg: float) -> tuple[float, float]:
        """(active_gate_in_fov_fraction, accept_rate_to_active_gate) at a target crab, interpolated
        from the fitted table.

        IMPORTANT [adversarial crab-independence 2026-06-14]: the returned ``accept_rate`` uses the
        RANGE-MARGINAL multiplier ``p_accept_active_in_image`` (~0.098) -- the accept rate averaged over
        the WHOLE approach (most of which is far/near, band-pass-rejected). It is a LOWER BOUND and a
        RELATIVE crab->FoV slope, NOT a per-frame fix density. The density that matters is in the
        18-28 m offered window, where accept-given-in-image ~= the band-pass peak ~0.84 (use
        ``accept_density_in_window`` / ``pointing_ceiling_accept_in_window``). The active gate is also
        ELEVATION-co-bound at fixable range (see report): yaw/crab alone is a modest lever; centring the
        gate through the window (2 axes) is the ~8.5x prize (``pointing_gain_in_window``)."""
        tab = sorted(self.crab_table)
        cs = np.array([t[0] for t in tab])
        ff = np.array([t[1] for t in tab])
        frac = float(np.interp(crab_deg, cs, ff))
        return frac, frac * self.p_accept_active_in_image

    @property
    def accept_density_in_window(self) -> float:
        """Per-frame accept probability when the active gate IS centred in-image during its 18-28 m
        offered window -- the band-pass peak. The reward-relevant fix density (NOT the diluted
        marginal in ``crab_to_fix_rate``)."""
        return float(self.pointing_ceiling_accept_in_window)


def _maybe(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


# The default surrogate uses the baked (fitted) constants -- no file IO at import. Call
# ``FixSurrogate.from_checkpoints()`` to reload the on-disk checkpoints (identical unless recalibrated).
DEFAULT = FixSurrogate()
