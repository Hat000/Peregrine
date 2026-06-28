"""rl/time_optimal_line.py -- the DENSE optimal-line shaping mechanism (bridge offline-exact -> RL).

TRACK-AGNOSTIC: this loads an ARBITRARY precomputed optimal-line JSON (schema
peregrine.reference_line.v1: pos_ned + a speed reference) and exposes a batched projection. The line
FILE is supplied by the env config (`+env.line_progress_file=<path>`), so a new track (e.g. the VQ2
drop) just feeds its own optimal line -- the reward MECHANISM is identical. The shipped
rl/time_optimal_line_inc8.json (the VQ1 4.62 s drag-ON TOGT solution, scripts/planning/out_drag/
trajectory.csv) is a SMOKE FIXTURE to exercise the code path, NOT a baked-in training target.

The inc8 R1' progress reward (rl/inc8_reward.arc_progress_reward) advances arc-length along the
gate-CENTRED contact-safe line Gamma (rl/reference_line_inc8.json). This optimal-line term is
ORTHOGONAL: it projects onto the supplied (fast) line and rewards forward arc-length, densifying the
sparse gate reward toward the time-optimal racing line. The class exposes:

  * ``progress(pos_ned)``  -- forward arc-length s along the time-optimal RACING line (a DENSE
    projected-progress signal: the planner places this line for time-optimality, NOT dead-centre, so
    it densifies the sparse gate reward toward the FAST line the policy must learn). Inherited from
    BatchedReferenceLine (the same op-for-op projector; parity-gated).
  * ``speed_ref(pos_ned)`` -- the optimal SPEED profile v_ref at the drone's projected arc-length
    (1-D interp over the sample arc table). The line already encodes the corner slow-downs (drag wall
    + turn), so the speed-profile term gives the policy the achievable v(s) to chase WITHOUT it having
    to discover the slow-downs.

Both are pure projections of NED position; the env converts Z-up<->NED at the call site (the line is
stored in NED, like Gamma). Self-contained (numpy + torch + json); the reward TERMS that consume these
live in rl/inc8_reward.py (line_progress_reward / speed_profile_reward), gated to default-0 (byte-id).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import torch
    from torch import Tensor
except Exception:                       # pragma: no cover
    torch = None
    Tensor = "Tensor"                   # type: ignore

from reference_line_torch import BatchedReferenceLine                # noqa: E402


class TimeOptimalLine(BatchedReferenceLine):
    """BatchedReferenceLine + a per-arc speed reference v_ref(s). ``progress`` (projected forward
    arc-length) is inherited verbatim from the parent (so it is parity-gated identically); this only
    adds the speed-profile lookup. ``speed_ref(pos)`` = v_ref at the projected arc-length of ``pos``."""

    def __init__(self, pos_ned: np.ndarray, speed_ref_mps: np.ndarray, device, dtype):
        super().__init__(pos_ned, device, dtype)
        sr = np.asarray(speed_ref_mps, dtype=np.float64).reshape(-1)
        assert sr.shape[0] == int(self.arc.shape[0]), (sr.shape, self.arc.shape)
        # store the sample-aligned (arc, v_ref) table on device for a batched 1-D interp.
        self.v_ref_samples = torch.as_tensor(sr, device=device, dtype=dtype)   # (M,)
        self._arc_np = np.asarray(self.arc.detach().cpu().numpy(), dtype=np.float64)  # (M,)
        self._vref_np = sr                                                     # (M,)

    @classmethod
    def load(cls, path: str | Path, device, dtype) -> "TimeOptimalLine":
        data = json.loads(Path(path).read_text())
        assert data.get("schema") == "peregrine.reference_line.v1", data.get("schema")
        pos = np.asarray(data["pos_ned"], dtype=np.float64)
        # speed_ref_mps is preferred; fall back to |vel_ned| so an older line JSON still loads.
        if "speed_ref_mps" in data:
            sr = np.asarray(data["speed_ref_mps"], dtype=np.float64)
        else:
            sr = np.linalg.norm(np.asarray(data["vel_ned"], dtype=np.float64), axis=1)
        return cls(pos, sr, device, dtype)

    def speed_ref(self, positions_ned: Tensor) -> Tensor:
        """Optimal reference speed (m/s) at the projected arc-length of each of ``positions_ned``
        (N,3). Projects to arc-length s via the inherited ``progress``, then 1-D-interpolates v_ref
        over the (arc, v_ref) sample table (clamped at both ends -- s in [0, total_arc])."""
        s = self.progress(positions_ned)                                       # (N,)
        return self._interp_vref(s)

    def _interp_vref(self, s: Tensor) -> Tensor:
        """Batched 1-D linear interpolation of v_ref over the monotone arc table (torch.searchsorted;
        np.interp-equivalent with end clamping). ``arc`` is strictly increasing (segment lengths > 0)."""
        arc = self.arc                                                         # (M,) monotone
        v = self.v_ref_samples                                                 # (M,)
        M = arc.shape[0]
        s_c = torch.clamp(s, float(arc[0]), float(arc[-1]))
        # right index of the bracketing interval, clamped to [1, M-1]
        idx = torch.searchsorted(arc, s_c, right=True).clamp(1, M - 1)
        lo = idx - 1
        a0, a1 = arc[lo], arc[idx]
        v0, v1 = v[lo], v[idx]
        w = (s_c - a0) / (a1 - a0).clamp(min=1e-12)
        return v0 + w * (v1 - v0)
