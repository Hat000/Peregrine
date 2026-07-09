"""rl/racing_line.py -- a batched, ONLINE (per-episode) NON-OPTIMAL racing-line generator + a
guiding-vector-field (GVF) query, for the VQ2 egocentric generation (inc9).

WHY THIS EXISTS (Fengyou greenlight 2026-07-08)
-----------------------------------------------
``build_reference_line.py`` builds the FIXED VQ1 6-gate line OFFLINE (a JSON emitted once from the
known gate centres). The inc9 ego env spawns a RANDOM gate layout every reset (single_gate_varied:
gate 8-15 m out, +-6 m in height), so the line must be built ONLINE, batched over N envs, on the
training device. This module does that.

THE LINE (Fengyou's spec, verbatim intent)
-------------------------------------------
A SMOOTH, CONTINUOUS, NON-OPTIMAL curve that passes through EVERY gate CENTRE and crosses each gate
HEAD-ON (tangent == the gate's through-normal at the centre), with NO CUSPS. It does NOT need to be
time/energy optimal (min-snap explicitly NOT wanted) -- it only has to give the reward a sane target
geometry the policy can approximate.

Construction: a per-segment cubic HERMITE spline over the knots [spawn, gate_0, ..., gate_{G-1}].
The tangent DIRECTION at each knot is prescribed:
  * spawn knot  -> the chord direction spawn->gate_0 (the line leaves the pad heading at the gate);
  * gate knot k -> the gate's through-normal n_k = [cos yaw_k, sin yaw_k, 0], oriented DOWN-COURSE.
Because the shared-knot tangent DIRECTION is identical for the two segments meeting at a gate (it is
the one gate normal), the curve is G1 (tangent-direction continuous) => no cusp, no tangent flip at a
gate. Tangent MAGNITUDES are chord-length scaled per segment (a cardinal/Catmull-Rom style), which is
all a non-optimal guide needs (magnitude discontinuity at a knot is not a cusp; the direction is what
the GVF reads). Every gate is crossed head-on by construction (tangent == gate normal there).

THE GVF QUERY (the reward's F(position))
----------------------------------------
For a drone at ``pos`` the query returns, by exact point-to-POLYLINE projection over the dense
samples:
  * ``s``       arc length of the NEAREST point on the line (a POTENTIAL -> telescopes -> the along-
                track progress reward clip(s_curr - s_prev) is NON-farmable and bounded by the line
                length; there is NO moving reference point to outrun or fall behind -- F is defined at
                EVERY position, which is exactly why Fengyou preferred a vector field over a lag ref);
  * ``perp``    perpendicular (cross-track) distance to the line (the contouring error; the MPCC/PBRS
                contouring term clip(perp_prev - perp_curr) pulls the drone ONTO the line, ORTHOGONAL
                to and hence DECOUPLED from the along-track progress -- raising the contouring weight
                strengthens line-pull without stealing forward progress);
  * ``tangent`` the unit line tangent at the nearest point (the along-track field direction; the
                cross-track field is (nearest_point - pos) projected off ``tangent``). The full guiding
                vector field is F = tangent + k*(cross-track error); the reward dot(v, F) integrated
                over a step EQUALS the two telescoping potentials above (progress + contouring), which
                is how this module feeds the existing tested reward machinery rather than a raw
                instantaneous dot product (a raw dot(v, F) speed-farms; the potential form cannot).

All GT (privileged): the line is built from the true spawn + true gate centres/normals and is only
ever consumed by the reward / critic, NEVER by the position-free actor obs.
"""
from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import Tensor
except Exception:                       # pragma: no cover - torch absent in some tooling contexts
    torch = None
    Tensor = "Tensor"                   # type: ignore


def _hermite_basis(u: Tensor):
    """Cubic Hermite basis (h00, h10, h01, h11) for parameter u in [0,1] (any shape)."""
    u2 = u * u
    u3 = u2 * u
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = u3 - 2.0 * u2 + u
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = u3 - u2
    return h00, h10, h01, h11


def gate_normals_from_yaw(gate_yaw: Tensor) -> Tensor:
    """Through-normals n_k = [cos yaw, sin yaw, 0] (N,G,3) from gate plane yaw (N,G). The sampler emits
    gate_yaw as the EXIT / down-course direction (gate-frame +x), so this normal already points
    down-course; ``build_racing_line`` re-orients defensively against the incoming chord anyway."""
    assert torch is not None
    cy = torch.cos(gate_yaw)
    sy = torch.sin(gate_yaw)
    z = torch.zeros_like(cy)
    return torch.stack([cy, sy, z], dim=-1)


@dataclass
class RacingLine:
    """Densely-sampled per-env racing line + arc length + unit tangents, with a batched GVF query.

    samples  (N, S, 3)  dense points along the line (Z-up GT world)
    arclen   (N, S)     cumulative arc length at each sample (arclen[:,0] == 0)
    tangent  (N, S, 3)  unit tangent at each sample (down-course)
    """
    samples: Tensor
    arclen: Tensor
    tangent: Tensor

    @property
    def total_len(self) -> Tensor:
        return self.arclen[:, -1]

    def query(self, pos: Tensor, env_idx: "Tensor | None" = None):
        """Project ``pos`` (M,3) onto the polyline of the selected envs by EXACT point-to-segment
        projection over all S-1 segments (batched). Returns:
            s        (M,)   arc length of the nearest point (along-track potential; monotone down-course)
            perp     (M,)   perpendicular distance to the line (cross-track / contouring error)
            tangent  (M,3)  unit line tangent at the nearest point (down-course)
        ``env_idx`` selects which envs' lines to use (defaults to all N, requiring M == N). A degenerate
        zero-length line (spawn == gate, should never happen given min_pair_dist) returns s=0, perp=dist
        to the single point, tangent = 0."""
        assert torch is not None
        if env_idx is not None:
            S = self.samples[env_idx]                              # (M,S,3)
            arclen = self.arclen[env_idx]                          # (M,S)
        else:
            S = self.samples                                       # (N,S,3)
            arclen = self.arclen
        a = S[:, :-1, :]                                           # (M,K,3) segment starts
        b = S[:, 1:, :]                                            # (M,K,3) segment ends
        ab = b - a                                                 # (M,K,3)
        seg_len2 = (ab * ab).sum(dim=-1)                           # (M,K)
        safe = seg_len2.clamp(min=1e-12)
        rel = pos.unsqueeze(1) - a                                 # (M,K,3)
        t = (rel * ab).sum(dim=-1) / safe                          # (M,K) projection param
        t = t.clamp(0.0, 1.0)
        proj = a + t.unsqueeze(-1) * ab                            # (M,K,3) closest point per segment
        d2 = ((pos.unsqueeze(1) - proj) ** 2).sum(dim=-1)          # (M,K)
        jmin = torch.argmin(d2, dim=1)                             # (M,)
        m = torch.arange(pos.shape[0], device=pos.device)
        t_min = t[m, jmin]                                         # (M,)
        seg_start_s = arclen[m, jmin]                              # (M,) arc length at segment start
        seg_len = torch.sqrt(seg_len2[m, jmin]).clamp(min=1e-9)    # (M,)
        s = seg_start_s + t_min * seg_len                          # (M,) continuous arc length
        perp = torch.sqrt(d2[m, jmin].clamp(min=0.0))              # (M,)
        # tangent at the nearest point = the nearest SEGMENT's direction (already ~unit in self.tangent
        # at sample nodes; use the segment direction for an exact between-node tangent)
        seg_dir = ab[m, jmin] / seg_len.unsqueeze(-1)              # (M,3) down-course tangent
        # INWARD unit direction (toward the line) at pos, for the guiding-vector-field alignment reward:
        # the guiding field is F = cos(theta)*tangent + sin(theta)*inward, theta = atan(gain*perp).
        proj_min = proj[m, jmin]                                   # (M,3) nearest point on the polyline
        inward = proj_min - pos                                    # (M,3) points from the drone TO the line
        inward_unit = inward / perp.clamp(min=1e-6).unsqueeze(-1)  # (M,3) unit (0 when already on the line)
        return s, perp, seg_dir, inward_unit


def build_racing_line(spawn_pos: Tensor, gate_pos: Tensor, gate_yaw: Tensor,
                      samples_per_seg: int = 24) -> RacingLine:
    """Build the batched racing line (see module docstring).

    spawn_pos (N,3), gate_pos (N,G,3), gate_yaw (N,G) -- Z-up GT world. Returns a RacingLine with
    ``N*(G) `` Hermite segments densely sampled at ``samples_per_seg`` each (the first segment keeps its
    u=0 endpoint; later segments drop u=0 to avoid a duplicated shared knot). G>=1.
    """
    assert torch is not None
    N, G, _ = gate_pos.shape
    dev, dt = gate_pos.device, gate_pos.dtype
    # knots P = [spawn, gate_0, ..., gate_{G-1}]  -> (N, G+1, 3)
    P = torch.cat([spawn_pos.unsqueeze(1), gate_pos], dim=1)       # (N, G+1, 3)
    M = G + 1
    # knot tangent DIRECTIONS: spawn -> chord to gate_0; gate k -> its through-normal (down-course).
    normals = gate_normals_from_yaw(gate_yaw)                      # (N,G,3)
    chord0 = gate_pos[:, 0, :] - spawn_pos                         # (N,3) spawn->gate0
    chord0 = chord0 / torch.linalg.norm(chord0, dim=-1, keepdim=True).clamp(min=1e-9)
    # orient each gate normal DOWN-COURSE: positive along the incoming chord (prev_knot -> gate)
    incoming = gate_pos - P[:, :-1, :]                             # (N,G,3) prev-knot -> gate k
    flip = (normals * incoming).sum(dim=-1, keepdim=True) < 0.0    # (N,G,1)
    normals = torch.where(flip, -normals, normals)
    D = torch.cat([chord0.unsqueeze(1), normals], dim=1)           # (N, G+1, 3) unit directions at knots

    u = torch.linspace(0.0, 1.0, samples_per_seg, device=dev, dtype=dt)   # (spp,)
    h00, h10, h01, h11 = _hermite_basis(u)                        # each (spp,)
    seg_pts = []
    for j in range(M - 1):
        P0 = P[:, j, :]                                           # (N,3)
        P1 = P[:, j + 1, :]                                       # (N,3)
        L = torch.linalg.norm(P1 - P0, dim=-1, keepdim=True).clamp(min=1e-9)   # (N,1) chord length
        T0 = D[:, j, :] * L                                       # (N,3) chord-scaled tangent
        T1 = D[:, j + 1, :] * L
        # (N, spp, 3): broadcast basis(spp) over the batch/knot vectors
        pts = (h00[None, :, None] * P0[:, None, :]
               + h10[None, :, None] * T0[:, None, :]
               + h01[None, :, None] * P1[:, None, :]
               + h11[None, :, None] * T1[:, None, :])             # (N, spp, 3)
        seg_pts.append(pts if j == 0 else pts[:, 1:, :])          # drop duplicated shared knot
    samples = torch.cat(seg_pts, dim=1)                           # (N, S, 3)

    # cumulative arc length + unit tangent (central differences on the dense samples)
    diffs = samples[:, 1:, :] - samples[:, :-1, :]                # (N, S-1, 3)
    seg_lens = torch.linalg.norm(diffs, dim=-1)                   # (N, S-1)
    arclen = torch.cat([torch.zeros(N, 1, device=dev, dtype=dt),
                        torch.cumsum(seg_lens, dim=1)], dim=1)    # (N, S)
    tangent = torch.zeros_like(samples)
    tangent[:, :-1, :] += diffs
    tangent[:, 1:, :] += diffs
    tangent = tangent / torch.linalg.norm(tangent, dim=-1, keepdim=True).clamp(min=1e-9)
    return RacingLine(samples=samples, arclen=arclen, tangent=tangent)
