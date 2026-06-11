"""Synthetic validation for the offline gate mapper — MEASURED noise, real VQ1 geometry.

The mapper (``racer.gate_mapper``) must be proven offline before any real VQ2 recording
exists, so this module generates gate sightings the way the real perception chain errs:

* TRUE map = the actual captured VQ1 course (``VQ1_TRACK_RECORDS``, verbatim from
  ``handoff/shadowpc-firstcontact-2026-06-02/track_map.json``: 6 gates, 23.7-38.5 m apart,
  descending 26 m, all facing -X).
* Flight = a conservative exploration pass: piecewise-linear through the opening centres at
  ~5 m/s, body-X along velocity (a forward-flying quad pitches into its descent — that is
  what keeps a descending gate inside the +20 deg up-tilted camera's -9.4..+49.4 deg
  elevation band), detections sampled at ~14.5 Hz (the recorded FPV rate).
* Visibility = the REAL camera model (``racer.frames`` intrinsics + mount): the gate centre
  must project inside the image with margin, in front, within [min_range, max_range].
* Noise = the MEASURED model (handoff/perception-char-2026-06-08): per-sighting world-fix
  error e ~ N(bias [-0.42,+0.06,-0.28], diag [0.73,0.47,0.29]^2) m, range-flat; the gate
  measurement carries -e (see the sign note in ``gate_mapper``); a per-gate-per-flight yaw
  bias (the ~+-3 deg term, which does NOT average away) + white yaw noise; a catastrophic
  leak (default 1.1%) replacing e with a 3-8 m outlier and possibly flipping yaw; and
  association errors (default 10%) that mislabel a sighting with a NEIGHBOUR gate's id
  while the geometry stays true to the gate actually seen — the real failure mechanism
  (the live chain anchors a good relative pose to the wrong map gate).

``evaluate_map`` scores an estimate against truth: per-gate position error (norm, NED
components, and the validity-relevant IN-PLANE error vs the 0.75 m half-opening), yaw
error, with optional translation alignment for the gauge-anchored no-pose case.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.spatial.transform import Rotation

from racer.frames import IMAGE_HEIGHT, IMAGE_WIDTH, R_world_from_body, world_point_in_camera
from racer.gate_mapper import (
    GateMapEstimate,
    GateSighting,
    MEASURED_CATASTROPHIC_LEAK,
    MEASURED_FIX_BIAS_NED,
    MEASURED_FIX_SIGMA_NED,
    RelativeGateSighting,
    _parse_prior_records,
    wrap_pi,
)

# The captured VQ1 course map, verbatim (handoff/shadowpc-firstcontact-2026-06-02/
# track_map.json; positions are gate BOTTOM-centres, quat cols = [+width, normal(-X),
# +height(down)]). The synthetic truth and every prior in the tests derive from this.
_VQ1_QUAT = [0.7071067094802856, 0.0, -8.657315930804543e-08, 0.7071067690849304]
_VQ1_POSITIONS = [
    [-23.2979679107666, -0.39990234375, -0.03195800632238388],
    [-46.89374923706055, -2.499990224838257, 5.068041801452637],
    [-74.59375, 1.2000097036361694, 13.668041229248047],
    [-111.49374389648438, -5.099989891052246, 24.56804084777832],
    [-135.49374389648438, -0.7999902367591858, 25.355653762817383],
    [-159.19374084472656, -4.399990081787109, 25.968040466308594],
]
VQ1_TRACK_RECORDS = [
    {"gate_id": i, "position_ned": p, "orientation_ned_wxyz": list(_VQ1_QUAT),
     "width_m": 2.7200000286102295, "height_m": 2.7200000286102295}
    for i, p in enumerate(_VQ1_POSITIONS)
]


def true_gates(records: list[dict] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(opening centres (G,3), through-yaws (G,)) — the mapper's own prior parse, which is
    a deliberate mirror of ``navigator.gates_from_track_records(corner_to_center=True)``
    (parity is asserted in the tests, so synth truth == what the live stack would fly)."""
    prior = _parse_prior_records(records or VQ1_TRACK_RECORDS)
    return prior["pos"], prior["yaw"]


# ---------------------------------------------------------------------------
# Noise model (measured)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NoiseModel:
    """The measured perception-noise model + visibility envelope (see module docstring)."""

    sigma_ned: np.ndarray = field(default_factory=lambda: MEASURED_FIX_SIGMA_NED.copy())
    bias_ned: np.ndarray = field(default_factory=lambda: MEASURED_FIX_BIAS_NED.copy())
    yaw_sigma_rad: float = float(np.deg2rad(2.0))        # white part of the angular term
    yaw_gate_bias_rad: float = float(np.deg2rad(3.0))    # per-gate-per-flight bias, U(-b, b)
    leak_rate: float = MEASURED_CATASTROPHIC_LEAK        # catastrophic fixes that slip the gate
    leak_mag_m: tuple[float, float] = (3.0, 8.0)         # measured leaked tail: 3-5+ m
    assoc_error_rate: float = 0.10                       # mislabel rate (85-95% reliable measured)
    detect_prob: float = 0.85                            # per-frame per-visible-gate detection
    min_range_m: float = 2.0
    max_range_m: float = 28.0    # between the measured range-flat 24 and the live 32 cap
    image_margin_px: float = 10.0

    def scaled(self, noise_scale: float = 1.0, **overrides) -> "NoiseModel":
        """A copy with sigma+bias scaled (sweep axis) and/or fields overridden."""
        nm = replace(self, sigma_ned=self.sigma_ned * noise_scale,
                     bias_ned=self.bias_ned * noise_scale)
        return replace(nm, **overrides) if overrides else nm


MEASURED_NOISE = NoiseModel()


# ---------------------------------------------------------------------------
# Flight path + visibility
# ---------------------------------------------------------------------------
def simulate_exploration_path(centres: np.ndarray, speed_mps: float = 5.0,
                              hz: float = 14.5, passes: int = 1,
                              start: np.ndarray | None = None,
                              scan_pitch_down_deg: float = 0.0,
                              scan_dist_m: float = 8.0,
                              ) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Conservative exploration trajectory: ``[(t, position_ned, R_world_body), ...]``.

    Piecewise-linear through the opening centres at constant speed, body-X along velocity
    (roll 0). ``passes > 1`` re-flies the course after a high return leg (+15 m above,
    reversed) — the up-tilted camera cannot see gates ~15 m BELOW it, so the return adds no
    sightings but keeps the timeline physical (no teleports for the smoothness prior).

    ``scan_pitch_down_deg`` > 0 adds a deliberate nose-down nod within ``scan_dist_m`` of
    each gate transit. This is the CO-VISIBILITY maneuver for no-pose mapping: with the
    camera tilted +20 deg UP, level-ish flight on this descending course never holds the
    current gate and the next (12-16 deg below the horizon) in frame together — measured
    co-visible frames ~12/480 and the no-pose graph shatters. A ~25 deg pre-transit nod
    re-centres the elevation band so both gates project at once on the 24-29 m legs (the
    39 m g2->g3 leg stays out of range regardless — that link rides on the smoothness
    bridge). We control the exploration lap, so this is an operational requirement the
    validation enforces, not a simulation convenience.
    """
    centres = np.asarray(centres, dtype=np.float64)
    start = np.array([0.0, 0.0, centres[0][2]]) if start is None else np.asarray(start, float)
    up = np.array([0.0, 0.0, -15.0])
    gate_keys = {tuple(np.round(c, 6)) for c in centres}
    waypoints: list[np.ndarray] = [start]
    for p in range(passes):
        if p > 0:  # return leg, high above the course, then re-enter at the start
            waypoints += [centres[-1] + up, start + up, start]
        waypoints += [c for c in centres]
    poses = []
    t = 0.0
    dt = 1.0 / hz
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        seg = b - a
        length = float(np.linalg.norm(seg))
        if length < 1e-9:
            continue
        v = seg / length
        yaw = float(np.arctan2(v[1], v[0]))
        pitch = float(-np.arcsin(np.clip(v[2], -1.0, 1.0)))   # nose into the descent
        ends_at_gate = tuple(np.round(b, 6)) in gate_keys
        n_steps = max(int(np.ceil(length / speed_mps * hz)), 1)
        for k in range(n_steps):
            dist_to_end = length - k * speed_mps * dt
            p_k = pitch
            if scan_pitch_down_deg > 0.0 and ends_at_gate and dist_to_end <= scan_dist_m:
                p_k = pitch - float(np.deg2rad(scan_pitch_down_deg))
            poses.append((t, a + v * (k * speed_mps * dt), R_world_from_body(0.0, p_k, yaw)))
            t += dt
    poses.append((t, waypoints[-1].copy(), poses[-1][2]))
    return poses


def visible_gate_indices(position: np.ndarray, R_wb: np.ndarray, centres: np.ndarray,
                         noise: NoiseModel) -> list[int]:
    """Gates whose centre projects inside the image (with margin), in front, range in band."""
    out = []
    for g, c in enumerate(centres):
        rng = float(np.linalg.norm(c - position))
        if not (noise.min_range_m <= rng <= noise.max_range_m):
            continue
        p_cam = world_point_in_camera(c, R_wb, position)
        if p_cam[2] <= 0.5:
            continue
        u = 320.0 * p_cam[0] / p_cam[2] + 320.0
        v = 320.0 * p_cam[1] / p_cam[2] + 180.0
        m = noise.image_margin_px
        if m <= u <= IMAGE_WIDTH - m and m <= v <= IMAGE_HEIGHT - m:
            out.append(g)
    return out


# ---------------------------------------------------------------------------
# Sighting generators (the single noise core serves cases A/B and C identically)
# ---------------------------------------------------------------------------
def _draw_fix_error(rng: np.random.Generator, noise: NoiseModel) -> tuple[np.ndarray, bool]:
    """One world-fix error e (gate measurement carries -e). Returns (e, is_leak)."""
    if rng.uniform() < noise.leak_rate:
        d = rng.normal(size=3)
        d /= np.linalg.norm(d) + 1e-12
        return d * rng.uniform(*noise.leak_mag_m), True
    return rng.normal(noise.bias_ned, noise.sigma_ned), False


def _label_for(g: int, n_gates: int, rng: np.random.Generator, noise: NoiseModel) -> int:
    """Apply the association-error mechanism: usually the true id, sometimes a neighbour."""
    if rng.uniform() >= noise.assoc_error_rate:
        return g
    cands = [j for j in (g - 1, g + 1) if 0 <= j < n_gates] or [g]
    return int(rng.choice(cands))


def generate_pose_aided_sightings(poses, records: list[dict] | None = None,
                                  noise: NoiseModel = MEASURED_NOISE,
                                  rng: np.random.Generator | None = None,
                                  labeled: bool = True,
                                  with_yaw: bool = True) -> list[GateSighting]:
    """Case A/B input: implied world gate-centre measurements from a trusted-pose flight."""
    rng = rng or np.random.default_rng(0)
    centres, yaws = true_gates(records)
    gate_yaw_bias = rng.uniform(-noise.yaw_gate_bias_rad, noise.yaw_gate_bias_rad,
                                len(centres))
    out: list[GateSighting] = []
    for fi, (t, pos, R_wb) in enumerate(poses):
        for g in visible_gate_indices(pos, R_wb, centres, noise):
            if rng.uniform() > noise.detect_prob:
                continue
            e, is_leak = _draw_fix_error(rng, noise)
            yaw_meas = None
            if with_yaw:
                yaw_meas = yaws[g] + gate_yaw_bias[g] + rng.normal(0.0, noise.yaw_sigma_rad)
                if is_leak and rng.uniform() < 0.5:
                    yaw_meas += rng.choice([np.pi / 2, -np.pi / 2, np.pi])
                yaw_meas = float(wrap_pi(yaw_meas))
            out.append(GateSighting(
                gate_world_ned=centres[g] - e,
                gate_id=_label_for(g, len(centres), rng, noise) if labeled else None,
                yaw_world=yaw_meas, t=float(t), frame_id=fi,
                range_m=float(np.linalg.norm(centres[g] - pos))))
    return out


def generate_relative_sightings(poses, records: list[dict] | None = None,
                                noise: NoiseModel = MEASURED_NOISE,
                                rng: np.random.Generator | None = None,
                                labeled: bool = False,
                                with_yaw: bool = True) -> list[RelativeGateSighting]:
    """Case C input: body-FRD levers + given attitude, same noise core as pose-aided."""
    rng = rng or np.random.default_rng(0)
    centres, yaws = true_gates(records)
    gate_yaw_bias = rng.uniform(-noise.yaw_gate_bias_rad, noise.yaw_gate_bias_rad,
                                len(centres))
    out: list[RelativeGateSighting] = []
    for fi, (t, pos, R_wb) in enumerate(poses):
        q_xyzw = Rotation.from_matrix(R_wb).as_quat()
        quat_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        drone_yaw = float(np.arctan2(R_wb[1, 0], R_wb[0, 0]))
        for g in visible_gate_indices(pos, R_wb, centres, noise):
            if rng.uniform() > noise.detect_prob:
                continue
            e, is_leak = _draw_fix_error(rng, noise)
            lever_meas = (centres[g] - pos) - e          # lever error = -fix error
            rel_yaw = None
            if with_yaw:
                yaw_meas = yaws[g] + gate_yaw_bias[g] + rng.normal(0.0, noise.yaw_sigma_rad)
                if is_leak and rng.uniform() < 0.5:
                    yaw_meas += rng.choice([np.pi / 2, -np.pi / 2, np.pi])
                rel_yaw = float(wrap_pi(yaw_meas - drone_yaw))
            out.append(RelativeGateSighting(
                frame_id=fi, rel_position_frd=R_wb.T @ lever_meas, quat_wxyz=quat_wxyz,
                gate_id=_label_for(g, len(centres), rng, noise) if labeled else None,
                rel_yaw=rel_yaw, t=float(t)))
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def evaluate_map(est: GateMapEstimate, records: list[dict] | None = None,
                 align_translation: bool = False) -> dict:
    """Score an estimated map against truth.

    Gates are matched greedily by distance (id-agnostic: cluster/no-pose modes renumber).
    ``align_translation`` removes the mean position error over matched gates first — the
    right scoring for the gauge-anchored no-pose case, where the map is only defined up to
    a world translation (consumers localize against the map, so shape is what matters).
    Reports per-gate: |err|, NED components, IN-PLANE error (the component in the gate
    plane — what the 0.75 m validity half-opening actually budgets), yaw error.
    """
    centres, yaws = true_gates(records)
    G = len(centres)
    est_pos = [g.position_ned for g in est.gates]
    # greedy nearest matching (est gate -> true gate), each truth claimed once
    pairs = []
    for ei, p in enumerate(est_pos):
        d = np.linalg.norm(centres - p, axis=1)
        for ti in np.argsort(d):
            pairs.append((float(d[ti]), ei, int(ti)))
    pairs.sort()
    used_e: set[int] = set()
    used_t: set[int] = set()
    match: dict[int, int] = {}
    for d, ei, ti in pairs:
        if ei in used_e or ti in used_t:
            continue
        match[ei] = ti
        used_e.add(ei)
        used_t.add(ti)
    offset = np.zeros(3)
    if align_translation and match:
        offset = np.mean([est_pos[ei] - centres[ti] for ei, ti in match.items()], axis=0)
    per_gate = []
    for ei, ti in sorted(match.items(), key=lambda kv: kv[1]):
        g = est.gates[ei]
        delta = (est_pos[ei] - offset) - centres[ti]
        n_hat = np.array([np.cos(yaws[ti]), np.sin(yaws[ti]), 0.0])
        inplane = delta - (delta @ n_hat) * n_hat
        yaw_err = None
        if g.yaw is not None:
            dy = wrap_pi(g.yaw - yaws[ti])
            if abs(dy) > np.pi / 2:          # gate-plane 180deg fold (plane is what matters)
                dy = wrap_pi(dy + np.pi)
            yaw_err = float(np.rad2deg(abs(dy)))
        per_gate.append({
            "true_gate": ti, "est_gate_id": g.gate_id,
            "err_m": float(np.linalg.norm(delta)),
            "err_ned": [float(v) for v in delta],
            "inplane_m": float(np.linalg.norm(inplane)),
            "yaw_err_deg": yaw_err,
            "n_used": g.n_used, "flags": list(g.flags),
        })
    errs = [pg["err_m"] for pg in per_gate]
    inpl = [pg["inplane_m"] for pg in per_gate]
    yerrs = [pg["yaw_err_deg"] for pg in per_gate if pg["yaw_err_deg"] is not None]
    return {
        "per_gate": per_gate,
        "n_matched": len(match), "n_extra_est": len(est.gates) - len(match),
        "n_missing_true": G - len(match),
        "pos_err_mean_m": float(np.mean(errs)) if errs else float("nan"),
        "pos_err_max_m": float(np.max(errs)) if errs else float("nan"),
        "inplane_err_max_m": float(np.max(inpl)) if inpl else float("nan"),
        "yaw_err_mean_deg": float(np.mean(yerrs)) if yerrs else float("nan"),
        "yaw_err_max_deg": float(np.max(yerrs)) if yerrs else float("nan"),
        "translation_offset_removed_m": float(np.linalg.norm(offset)),
    }
