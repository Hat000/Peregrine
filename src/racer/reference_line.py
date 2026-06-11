"""Reference racing line — loader + samplers for ``peregrine.reference_line.v1`` JSON
(``rl/reference_line_vq1.json``, produced by ``scripts/togt/export_reference.py`` from the
TOGT + multiple-shooting time-optimal plan; handoff laptop-togt-bound-2026-06-10).

Schema (all world NED / body FRD; SI units, radians; yaw = NED heading atan2(E, N)):
  schema           "peregrine.reference_line.v1"
  frame            human-readable frame note
  source           provenance (case name, plant constraint set used by the planner)
  lap_time_s       time of the LAST gate-plane crossing (t=0 = at rest on the pad);
                   the line continues past the finish (virtual endpoint), so the final
                   sample time > lap_time_s -- use lap_time_s for lap arithmetic.
  total_duration_s last sample time (includes the post-finish run-out)
  gate_crossings   per gate: {gate_id, t, pos_ned, miss_m, miss_h_m, miss_v_m}
  t                (N,)   sample times, strictly increasing, ~50 Hz
  pos_ned/vel_ned/acc_ned  (N,3)
  yaw              (N,)   may wrap; sample() interpolates via unwrap
  quat_wxyz        (N,4)  R_world_body, scalar-first
  omega_frd        (N,3)  body rates
  thrust_norm      (N,)   live normalized collective (hover 0.2656 <-> 1 g)

Intended consumers: the twin tracking replay (scripts/twin_track_reference.py), the RL
progress reward (arc-length progress along the line), and any future plan+track planner.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, eq=False)
class ReferenceSample:
    """Interpolated reference state at one time instant (world NED / body FRD)."""

    t: float
    position_ned: np.ndarray
    velocity_ned: np.ndarray
    accel_ned: np.ndarray
    yaw: float
    thrust_norm: float


class ReferenceLine:
    """Time-indexed reference trajectory with linear interpolation + arc-length progress."""

    def __init__(self, data: dict):
        assert data.get("schema") == "peregrine.reference_line.v1", \
            f"unknown schema: {data.get('schema')!r}"
        self.raw = data
        self.lap_time_s: float = float(data["lap_time_s"])
        self.total_duration_s: float = float(data["total_duration_s"])
        self.gate_crossings = data["gate_crossings"]
        self.t = np.asarray(data["t"], dtype=np.float64)
        self.pos = np.asarray(data["pos_ned"], dtype=np.float64)
        self.vel = np.asarray(data["vel_ned"], dtype=np.float64)
        self.acc = np.asarray(data["acc_ned"], dtype=np.float64)
        self.yaw_unwrapped = np.unwrap(np.asarray(data["yaw"], dtype=np.float64))
        self.thrust_norm = np.asarray(data["thrust_norm"], dtype=np.float64)
        seg = np.linalg.norm(np.diff(self.pos, axis=0), axis=1)
        self.arc = np.concatenate([[0.0], np.cumsum(seg)])     # cumulative arc length (m)

    @classmethod
    def load(cls, path: str | Path) -> "ReferenceLine":
        return cls(json.loads(Path(path).read_text()))

    def sample(self, t: float) -> ReferenceSample:
        """Reference state at time ``t`` (clamped to the line's time span)."""
        tc = float(np.clip(t, self.t[0], self.t[-1]))
        return ReferenceSample(
            t=tc,
            position_ned=np.array([np.interp(tc, self.t, self.pos[:, i]) for i in range(3)]),
            velocity_ned=np.array([np.interp(tc, self.t, self.vel[:, i]) for i in range(3)]),
            accel_ned=np.array([np.interp(tc, self.t, self.acc[:, i]) for i in range(3)]),
            yaw=float(np.interp(tc, self.t, self.yaw_unwrapped)),
            thrust_norm=float(np.interp(tc, self.t, self.thrust_norm)),
        )

    def progress(self, position_ned: np.ndarray) -> float:
        """Arc-length progress (m) of the nearest point on the polyline to ``position_ned``
        -- the RL progress-reward primitive. O(N); fine at ~50 Hz x 30 s."""
        p = np.asarray(position_ned, dtype=np.float64)
        a, b = self.pos[:-1], self.pos[1:]
        ab = b - a
        denom = np.einsum("ij,ij->i", ab, ab)
        denom[denom == 0.0] = 1e-12
        s = np.clip(np.einsum("ij,ij->i", p - a, ab) / denom, 0.0, 1.0)
        proj = a + s[:, None] * ab
        d2 = np.einsum("ij,ij->i", p - proj, p - proj)
        i = int(np.argmin(d2))
        seg_len = float(np.sqrt(denom[i]))
        return float(self.arc[i] + s[i] * seg_len)
