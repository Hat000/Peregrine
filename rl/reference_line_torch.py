"""rl/reference_line_torch.py -- batched torch arc-length projection onto the inc8 reference line Gamma.

The torch-vectorised twin of ``racer.reference_line.ReferenceLine.progress`` (numpy, O(N) per query):
projects N drone positions onto the polyline Gamma and returns each one's arc-length progress (m).
This is the R1' progress primitive for the inc8 reward (rw_progress * (s_curr - s_prev) along the
REBUILT corrected-aero contact-safe line). Parity-gated against the numpy loader/projector
(tests/test_inc8_reference_line_torch.py, <=1e-4).

Loaded from ``rl/reference_line_inc8.json`` (schema peregrine.reference_line.v1). Self-contained
(numpy + torch only): the JSON is parsed with plain ``json``; only ``pos_ned`` is needed for
``progress``. The env builds ONE BatchedReferenceLine and calls ``progress`` per step on Z-up
positions (it converts Z-up<->NED with _FLIP at the call site -- Gamma is stored in NED).
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


class BatchedReferenceLine:
    """Time-indexed reference trajectory with a batched arc-length ``progress`` -- the inc8 R1'
    reward primitive. Precomputes the segment table (a, ab, |ab|^2, cumulative arc) as tensors on the
    target device/dtype; ``progress`` is a single O(N*M) projection (M ~= 879 samples, N envs)."""

    def __init__(self, pos_ned: np.ndarray, device, dtype):
        assert torch is not None, "BatchedReferenceLine requires torch"
        pos = np.asarray(pos_ned, dtype=np.float64)
        assert pos.ndim == 2 and pos.shape[1] == 3, pos.shape
        seg = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg)])             # cumulative arc length (m), (M,)
        a = pos[:-1]                                              # (M-1, 3)
        ab = pos[1:] - pos[:-1]                                   # (M-1, 3)
        denom = np.einsum("ij,ij->i", ab, ab)                   # (M-1,)
        denom[denom == 0.0] = 1e-12                              # mirror ReferenceLine.progress guard
        self.device, self.dtype = device, dtype
        self.arc = torch.as_tensor(arc, device=device, dtype=dtype)
        self.a = torch.as_tensor(a, device=device, dtype=dtype)
        self.ab = torch.as_tensor(ab, device=device, dtype=dtype)
        self.denom = torch.as_tensor(denom, device=device, dtype=dtype)
        self.seg_arc = self.arc[:-1]                             # (M-1,) arc at each segment start
        self.total_arc_m = float(arc[-1])
        self.n_seg = int(a.shape[0])

    @classmethod
    def load(cls, path: str | Path, device, dtype) -> "BatchedReferenceLine":
        data = json.loads(Path(path).read_text())
        assert data.get("schema") == "peregrine.reference_line.v1", data.get("schema")
        return cls(np.asarray(data["pos_ned"], dtype=np.float64), device, dtype)

    def progress(self, positions_ned: Tensor) -> Tensor:
        """Arc-length progress (m) of the nearest point on Gamma to each of ``positions_ned`` (N,3).
        Mirrors ReferenceLine.progress op-for-op: per segment, project + clamp s in [0,1], pick the
        nearest segment, return arc[i] + s[i]*seg_len. Returns (N,)."""
        p = positions_ned.to(self.dtype)
        if p.dim() == 1:
            p = p.unsqueeze(0)
        pa = p.unsqueeze(1) - self.a.unsqueeze(0)                # (N, M-1, 3)
        t = (pa * self.ab.unsqueeze(0)).sum(-1)                  # (N, M-1)
        s = torch.clamp(t / self.denom.unsqueeze(0), 0.0, 1.0)  # (N, M-1)
        proj = self.a.unsqueeze(0) + s.unsqueeze(-1) * self.ab.unsqueeze(0)   # (N, M-1, 3)
        diff = p.unsqueeze(1) - proj
        d2 = (diff * diff).sum(-1)                              # (N, M-1)
        i = torch.argmin(d2, dim=1)                            # (N,)  first-min (matches np.argmin)
        seg_len = torch.sqrt(self.denom[i])
        s_i = torch.gather(s, 1, i.unsqueeze(-1)).squeeze(-1)
        return self.seg_arc[i] + s_i * seg_len
