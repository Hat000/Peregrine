"""VQ1 course conversion (NED -> DiffAero Z-up) + PROCEDURAL 6-gate course sampling for RL training.

Part 1 (original): convert the VQ1 captured gate map (NED) into a DiffAero-frame fixed gate layout.
Part 2 (S1.4, 2026-06-10): ``sample_courses`` -- draw random spec-plausible 6-gate courses so the
policy trains on a DISTRIBUTION of tracks and holds the exact VQ1 course out for eval
(stack-review meta-gap #1: a one-track policy on the unseen VQ2 course may score zero).

The Peregrine offline twin / RL plant lives in world NED (Z DOWN); DiffAero's world is Z-UP. Our
dynamics adapter (``rl.diffaero_dynamics``) bridges them with the involutory flip
``_FLIP = [1, -1, -1]`` (NED <-> Z-up on x kept, y/z negated). The DiffAero racing env therefore sees
positions in that Z-up frame, so the injected gate layout must be expressed there too.

Per-gate we produce:
  * ``gate_pos`` (Z-up): the gate OPENING CENTRE. ``capture_track_map.py`` stores ``position_ned`` as the
    gate BOTTOM-centre, so we lift by half the (outer) height along NED-down to reach the opening centre
    (the same +height/2 correction ``navigator.gates_from_track_records(corner_to_center=True)`` applies),
    then flip NED -> Z-up.
  * ``gate_yaw`` (Z-up, rad): the gate's horizontal facing. DiffAero's ``get_gate_rotmat_w2g`` is a pure
    yaw, and ``is_passed`` is a plane-crossing on the gate's local +X axis = world ``[cos yaw, sin yaw, 0]``
    (the EXIT/down-course direction). We take the gate normal from the map quaternion (the true gate
    plane), orient it down-course (toward the next gate), flip to Z-up, and read ``atan2(ny, nx)``.

This module is the SINGLE source of the conversion: run as ``__main__`` on the laptop to emit
``peregrine_course_diffaero.json`` (plain ``[x, y, z, yaw]`` per gate) after eyeballing the printout;
``rl.peregrine_racing`` then just loads that JSON on the cluster (no scipy / no map parsing there).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# NED <-> DiffAero Z-up flip (must match rl.diffaero_dynamics._FLIP exactly).
_FLIP = np.array([1.0, -1.0, -1.0])
_OUTER_TO_INNER_LIFT_FRACTION = 0.5  # opening centre = bottom-centre lifted half the (outer) height


def _quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix R_world_gate from a wxyz unit quaternion (Hamilton, matches scipy/racer.frames)."""
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    s = 0.0 if n == 0.0 else 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ])


def load_gates_ned(map_path: str | Path) -> list[dict]:
    data = json.loads(Path(map_path).read_text())
    return data["gates"]


def course_to_diffaero(gates: list[dict]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (gate_pos_zup (N,3), gate_yaw (N,), diag) for the ordered gate records."""
    n = len(gates)
    centres_ned = np.zeros((n, 3))
    normals_ned = np.zeros((n, 3))
    positions = [np.asarray(g["position_ned"], dtype=np.float64) for g in gates]
    for i, g in enumerate(gates):
        pos = positions[i]
        h = float(g.get("height_m") or 2.72)
        # bottom-centre -> opening centre: lift UP = NED-down is +Z, so subtract along +Z to go up.
        centre = pos + np.array([0.0, 0.0, -_OUTER_TO_INNER_LIFT_FRACTION * h])
        centres_ned[i] = centre
        # gate normal from the map quaternion (col 1 of R_world_gate = gate local Y = through-axis), as in
        # navigator.gates_from_track_records(corner_to_center=True).
        q = g.get("orientation_ned_wxyz")
        seg = (positions[i + 1] - positions[i]) if i + 1 < n else (positions[i] - positions[i - 1])
        if q is not None:
            R = _quat_wxyz_to_matrix(np.asarray(q, dtype=np.float64))
            nrm = R[:, 1]
        else:
            nrm = seg / (np.linalg.norm(seg) + 1e-12)
        if float(nrm @ seg) < 0.0:           # orient down-course (toward the next gate)
            nrm = -nrm
        normals_ned[i] = nrm

    # NED -> Z-up flip.
    gate_pos_zup = centres_ned * _FLIP
    normals_zup = normals_ned * _FLIP
    # yaw from the horizontal normal; also compute the segment-derived yaw as a cross-check.
    gate_yaw = np.arctan2(normals_zup[:, 1], normals_zup[:, 0])
    seg_zup = np.diff(gate_pos_zup, axis=0)
    seg_yaw = np.arctan2(seg_zup[:, 1], seg_zup[:, 0])
    diag = {
        "centres_ned": centres_ned,
        "normals_ned": normals_ned,
        "gate_pos_zup": gate_pos_zup,
        "gate_yaw_deg": np.degrees(gate_yaw),
        "seg_yaw_deg": np.degrees(seg_yaw),
    }
    return gate_pos_zup, gate_yaw, diag


# ================================================================================================
# Part 2 -- PROCEDURAL COURSE SAMPLING (S1.4 / session S15, 2026-06-10).
#
# Torch-based (the env consumes tensors on the training device); torch import is GUARDED so the
# numpy json-conversion CLI above still runs anywhere.
#
# VQ1-derived geometry stats (Z-up, computed from peregrine_course_diffaero.json; re-derived and
# pinned by tests/test_peregrine_course.py so drift in the json breaks the test, not the training):
#   horizontal segment length   [23.69, 27.95, 37.43, 24.38, 23.97] m   (gate->gate)
#   per-segment descent         [ 5.10,  8.60, 10.90,  0.79,  0.61] m   (all descending)
#   heading change per segment  ~[12.7, 17.3, 19.9, 18.8] deg          (gentle weave about a line)
#   max grade |dz|/horiz        0.31
#   gate yaw vs segment heading within ~10 deg (all map yaws are exactly pi)
#   spawn                       23.3 m up-course of gate 0; gate-0 centre +1.41 m ABOVE the pad
#
# Sampling ranges = those stats widened to "sane multiples". Turns go to +-60 deg (3x VQ1's max):
# VQ1 is nearly straight, and multiples of ~0 degrees would still be ~0 -- the whole point is
# insurance against an unseen VQ2 layout, so the turn budget comes from "what a 6-gate drone-racing
# course plausibly does", bounded so courses stay flyable (no hairpins beyond 60 deg/segment).
# The VQ1 course is INTERIOR to every range => training on this distribution with VQ1 held out is a
# genuine generalization test (VQ1 itself has measure zero in the sampler).
# ================================================================================================
try:
    import torch
except Exception:                      # pragma: no cover - torch absent in some tooling contexts
    torch = None

DEFAULT_COURSE_RANGES = dict(
    n_gates=6,
    seg_len_m=(15.0, 45.0),       # horizontal gate->gate distance; VQ1 [23.7, 37.4]
    turn_rad=1.047,               # |heading change| per segment <= 60 deg (uniform in +-); VQ1 <= ~20 deg
    drop_m=(-3.0, 12.0),          # descent per segment, +down (z_up DECREASES); VQ1 [0.6, 10.9]
    max_grade=0.45,               # |dz| <= max_grade * horizontal length; VQ1 max 0.31
    yaw_jitter_rad=0.21,          # gate plane yaw = path bisector +- 12 deg; VQ1 deviates <= ~10 deg
    spawn_dist_m=(18.0, 28.0),    # standing-start pad -> gate 0 horizontal; VQ1 23.3
    spawn_below_g0_m=(0.5, 2.5),  # gate-0 centre this far ABOVE the pad; VQ1 1.41
    min_pair_dist_m=10.0,         # reject layouts with any two gates (or a gate and the pad) closer
    spawn_yaw_jitter_rad=0.0,     # (Fengyou 2026-07-08) jitter the DRONE's spawn yaw off the gate bearing by
                                  # U(+-jitter) so the gate lands at varied LEFT/RIGHT positions in the FOV
                                  # (realistic FOV coverage). Distinct from spawn_heading (a GLOBAL rotation =
                                  # redundant for the heading-invariant egocentric obs); this changes the gate's
                                  # RELATIVE bearing -> the body-frame rel_pos varies -> real training signal.
                                  # Keep <= the camera HFOV so the gate stays visible (>=4 keypoints). 0 == OFF.
    spawn_heading=None,           # segment-0 world heading. None -> random in (-pi, pi] (the VQ1/inc8
                                  # behaviour: courses fan around the pad). A FIXED value -> every course
                                  # starts along that heading (Fengyou 2026-07-07: for the EGOCENTRIC
                                  # position-free policy the global course heading is REDUNDANT -- the obs
                                  # is heading-invariant -- so the random fan adds no training signal and
                                  # just scatters the world-frame render into a confusing circle. Fixing
                                  # it (ego sets 0.0) leaves the egocentric distribution IDENTICAL while
                                  # placing every course ahead of the pad. The relative gate placement
                                  # still varies via the spawn attitude jitter + the turn/drop walk.)
    g1_out_of_fov_bearing_rad=None,  # (pefcap, 2026-07-12) when set to (lo, hi), REPLACE the segment-0->1 turn
                                  # (the bend AT gate 0, placing gate 1) with the turn that puts gate 1's CENTRE
                                  # at a target bearing phi_t ~ U(lo, hi) just PAST the camera FOV edge (half-
                                  # HFOV ~ 45deg; the model's HFOV is 90deg) on the gate-0 approach, so gate 1
                                  # sits OUTSIDE the frame in the decision-critical pre-pass window and acquiring
                                  # it REQUIRES the coarse-map horiz turn prior. The turn is COMPUTED PER-ENV
                                  # from the ACTUAL sampled spawn distance L0 and gate0->gate1 spacing L1 (NOT a
                                  # fixed angle): at a pre-pass reference vantage d_ref = _G1_OOFOV_APPROACH_FRAC
                                  # * L0 behind gate 0 the drone->gate1 bearing off the approach heading is
                                  # atan2(L1 sin tau, L1 cos tau + d); setting that == phi_t gives the law-of-
                                  # sines parallax solve tau = phi_t + asin((d_ref/L1) sin phi_t). Larger spawn
                                  # distance -> larger turn; wider spacing -> smaller. Random sign (left/right).
                                  # The coarse map (build_coarse_map, UNCHANGED) then reads horiz=+-1 for gate 0
                                  # and becomes load-bearing. Needs n_gates>=2 (else inert). None == OFF
                                  # (byte-identical: the turn stays U(-turn_rad, turn_rad)).
    gates_above_spawn_m=None,     # (A1 floor fix, 2026-07-10) when set, EVERY gate centre z is kept
                                  # >= spawn_z + this clearance (the pad is z=0 in the sampler frame).
                                  # Enforced SEQUENTIALLY along the walk (z[g] = max(z[g-1]+dz[g], min_z))
                                  # so a leg that would dive below the floor lands ON it and later climbs
                                  # resume from there (walk shape preserved; a descent's |dz| only ever
                                  # SHRINKS, so max_grade still holds). Needed with the floor_at_spawn
                                  # training floor: spawn_below_g0_m only constrains gate 0 -- with the
                                  # default drop_m a LATER gate can sink below the pad, which would be
                                  # undivable-to with the floor on. None == OFF (byte-identical legacy).
    gates_ceiling_m=None,         # (VERTICAL-STRUCTURE fix, 2026-07-25) the SYMMETRIC COUNTERPART of
                                  # gates_above_spawn_m: when set, EVERY gate centre z is kept <=
                                  # spawn_z + this (the pad is z=0 in the sampler frame). Enforced in
                                  # the SAME sequential walk clamp (z[g] = clamp(z[g-1]+dz[g], min_z,
                                  # max_z)) so a leg that would climb through the roof lands ON it and
                                  # later descents resume from there (walk shape preserved; a climb's
                                  # |dz| only ever SHRINKS, so max_grade still holds).
                                  # WHY IT EXISTS: the legacy drop_m band (-3, 12) is DESCENT-BIASED
                                  # (dz = -drop in [-12, +3]) so the vertical coarse-sector +1 bucket
                                  # is realised on only ~1.6% of gates (measured, n=20000). Widening
                                  # drop_m symmetrically (the existing course_drop_lo/hi keys) fixes
                                  # that, but with only a FLOOR the walk is a reflected random walk
                                  # that drifts unboundedly UP (measured max gate z 44 m over 8 gates
                                  # at drop_m=(-8,8)). The ceiling bounds the envelope to a
                                  # warehouse-realistic band AND keeps the structure OSCILLATING (a
                                  # climb must eventually be followed by a descent) instead of a
                                  # one-way climb-out. None == OFF (byte-identical legacy).
    lateral_offset_m=(0.0, 0.0),  # (Track-A course enrichment, 2026-07-13) per-gate NON-CUMULATIVE
                                  # perpendicular JOG -- a slalom/chicane xy-plane shift that is DISTINCT
                                  # from turn_rad's cumulative heading walk. For every gate g>=1 the gate
                                  # centre is displaced by U(lo, hi) (random +/- sign) along the HORIZONTAL
                                  # (Z-up) perpendicular of its BASE incoming leg, then gate_yaw is
                                  # recomputed from the FINAL positions so the gate faces the true
                                  # down-course bisector (and build_coarse_map, which reads the final
                                  # positions, stays sign-correct BY CONSTRUCTION). Gate 0 is NEVER offset
                                  # (its FOV placement is owned by spawn_dist / spawn_yaw_jitter). Because
                                  # the jog is measured off the BASE (un-offset) cumsum walk it does NOT
                                  # accumulate into the heading, so unlike a larger turn_rad it gives rich
                                  # per-gate lateral variety at HIGH gate counts (20) WITHOUT the heading
                                  # random-walk curling back on itself (which the min-separation reject
                                  # loop otherwise collapses to a straight fallback). (0.0, 0.0) == OFF ==
                                  # byte-identical (the ON branch is the ONLY new RNG; OFF draws nothing).
)

# VQ1 standing-start pad (Z-up), from the S1.2 live recordings (see peregrine_racing.py).
VQ1_SPAWN_POS_ZUP = (0.0, 0.0, -0.02)
VQ1_SPAWN_YAW = 0.0          # body yaw at spawn; gates at yaw pi => tail-first (gate-frame yaw ~ pi)
VQ1_SPAWN_PITCH_RAD = -0.31  # -17.8 deg tilted pad (measured)

# GATE-1 OUT-OF-FOV (pefcap 2026-07-12): the pre-pass reference vantage for the parallax turn solve, as a
# FRACTION of the sampled spawn distance L0 BACK from gate 0 (i.e. the drone is ~(1-frac) of the way in).
# 0.25 == require gate 1 out of FOV once the drone is 3/4 of the way to gate 0 (the decision-critical
# pre-pass commit window; measured critical hide-turns 47-56 deg at that vantage, rl/tools calibration).
_G1_OOFOV_APPROACH_FRAC = 0.25


def _circ_mean(a, b):
    """Circular mean of two angles (torch tensors), elementwise."""
    return torch.atan2(torch.sin(a) + torch.sin(b), torch.cos(a) + torch.cos(b))


def sample_courses(n, device="cpu", generator=None, **overrides):
    """Draw ``n`` random 6-gate courses (Z-up frame). Returns a dict of torch tensors:

        gate_pos  (n, G, 3)  gate opening centres
        gate_yaw  (n, G)     gate plane yaw (exit/down-course direction = +x of the gate frame)
        spawn_pos (n, 3)     standing-start pad position
        spawn_yaw (n,)       nominal body yaw at spawn = gate_yaw[0] + pi  (TAIL-FIRST, matching the
                             VQ1 deployment convention: fly_rl.py runs the policy behind a virtual
                             pi body-z flip, so training spawns must stay tail-first w.r.t. gate 0)

    Construction: a heading random walk. Segment 0 is pad->gate0; segments 1..G-1 are gate->gate.
    Headings turn by U(-turn, +turn) per segment; horizontal lengths and per-segment descents are
    uniform in their ranges (descent clamped to ``max_grade``); gate yaw = circular mean of the
    incoming/outgoing segment headings (pure incoming for the last gate) + jitter. Layouts where any
    two gates (or a gate and the pad) come within ``min_pair_dist_m`` horizontally are redrawn
    (vectorized rejection; with these defaults the reject rate is small and redraws only replace the
    offending courses, so the loop terminates fast).
    """
    if torch is None:
        raise RuntimeError("sample_courses requires torch")
    unknown = set(overrides) - set(DEFAULT_COURSE_RANGES)
    if unknown:
        raise TypeError(f"sample_courses: unknown range override(s) {sorted(unknown)} "
                        f"(valid: {sorted(DEFAULT_COURSE_RANGES)})")
    R = {**DEFAULT_COURSE_RANGES, **overrides}
    G = int(R["n_gates"])

    def U(lo, hi, *shape):
        return lo + (hi - lo) * torch.rand(*shape, device=device, generator=generator)

    def draw(m):
        # heading random walk: heading[k] = direction of segment k (segment 0 = pad->gate0). The
        # segment-0 heading is random (VQ1/inc8) unless spawn_heading is fixed (egocentric: heading is a
        # redundant global DOF -> pin it so the world-frame layout is not a confusing circle).
        if R["spawn_heading"] is None:
            h0 = U(-np.pi, np.pi, m, 1)
        else:
            h0 = torch.full((m, 1), float(R["spawn_heading"]), device=device)
        turns = U(-R["turn_rad"], R["turn_rad"], m, G - 1)
        # seg_len drawn BEFORE headings (headings does no RNG, so the OFF-path RNG order is unchanged) because
        # the gate-1 out-of-FOV forcing below needs the sampled spawn distance L0 + spacing L1.
        seg_len = torch.empty(m, G, device=device)
        seg_len[:, 0] = U(*R["spawn_dist_m"], m)
        seg_len[:, 1:] = U(*R["seg_len_m"], m, G - 1)
        # GATE-1 OUT-OF-FOV forcing (pefcap 2026-07-12; OFF when None == byte-identical). REPLACE the FIRST
        # turn (the bend at gate 0, which places gate 1) with the turn that puts gate 1's CENTRE at a target
        # bearing phi_t just past the camera FOV edge on the gate-0 approach, COMPUTED PER-ENV from the ACTUAL
        # sampled spawn distance L0 and gate0->gate1 spacing L1 (not a fixed angle). At a pre-pass reference
        # vantage d_ref = _G1_OOFOV_APPROACH_FRAC*L0 behind gate 0, the drone->gate1 bearing off the approach
        # heading is atan2(L1 sin tau, L1 cos tau + d); setting that == phi_t gives the law-of-sines parallax
        # solve tau = phi_t + asin((d_ref/L1) sin phi_t). Larger spawn distance -> larger turn; wider spacing
        # -> smaller turn. Random sign (left/right). Needs a segment 1 (G>=2). Only the ON branch draws extra
        # RNG (OFF path never enters). All other turns/segments untouched.
        if R["g1_out_of_fov_bearing_rad"] is not None and G >= 2:
            lo, hi = R["g1_out_of_fov_bearing_rad"]
            phi_t = U(float(lo), float(hi), m)                                    # target gate1 bearing past edge
            L0 = seg_len[:, 0]; L1 = seg_len[:, 1].clamp(min=1e-3)                # sampled spawn dist + spacing
            d_ref = _G1_OOFOV_APPROACH_FRAC * L0                                  # pre-pass reference distance
            tau_mag = phi_t + torch.asin(torch.clamp((d_ref / L1) * torch.sin(phi_t), -1.0, 1.0))
            sign = torch.where(torch.rand(m, device=device, generator=generator) < 0.5, -1.0, 1.0)
            turns[:, 0] = sign * tau_mag
        headings = torch.cat([h0, h0 + torch.cumsum(turns, dim=1)], dim=1)          # (m, G)
        dz = torch.empty(m, G, device=device)
        dz[:, 0] = U(*R["spawn_below_g0_m"], m)                                     # gate 0 ABOVE pad
        drop = U(*R["drop_m"], m, G - 1)
        dz[:, 1:] = -torch.clamp(drop, -R["max_grade"] * seg_len[:, 1:],
                                 R["max_grade"] * seg_len[:, 1:])
        seg = torch.stack([seg_len * torch.cos(headings),
                           seg_len * torch.sin(headings), dz], dim=-1)              # (m, G, 3)
        gate_pos = torch.cumsum(seg, dim=1)                                         # pad at origin
        # LATERAL (xy) PER-GATE JOG (Track-A enrichment, 2026-07-13; OFF at (0,0) == byte-identical). A
        # NON-cumulative perpendicular slalom shift: displace each gate g>=1 by U(lo,hi) * (+/- sign)
        # along the HORIZONTAL perpendicular of its BASE incoming leg (gate[g]-prev[g], prev[0]=spawn=0).
        # Measured off the BASE cumsum walk -> the jog does NOT propagate into later gates' base
        # positions (no heading drift -> valid at high gate counts). Gate 0 is never offset. Only this ON
        # branch draws extra RNG (mag then sign) -> the OFF path RNG stream is untouched. gate_yaw is
        # recomputed from the FINAL (offset) positions below so the facing + build_coarse_map stay right.
        lat_lo, lat_hi = R["lateral_offset_m"]
        lat_on = not (float(lat_lo) == 0.0 and float(lat_hi) == 0.0)
        if lat_on:
            prev_base = torch.zeros_like(gate_pos)
            prev_base[:, 1:, :] = gate_pos[:, :-1, :]                               # prev[0] = spawn (origin)
            incoming = gate_pos - prev_base                                        # (m,G,3) base legs
            hx, hy = incoming[..., 0], incoming[..., 1]
            hnorm = torch.sqrt(hx * hx + hy * hy).clamp(min=1e-6)
            perp = torch.stack([-hy / hnorm, hx / hnorm,                            # left-perp unit (Z-up)
                                torch.zeros_like(hx)], dim=-1)                      # (m,G,3)
            mag = U(float(lat_lo), float(lat_hi), m, G)                             # (m,G) jog magnitude
            mag[:, 0] = 0.0                                                        # gate 0 anchored on the approach
            sign = torch.where(torch.rand(m, G, device=device, generator=generator) < 0.5, -1.0, 1.0)
            gate_pos = gate_pos + (mag * sign).unsqueeze(-1) * perp
        # GATES-ABOVE-SPAWN floor clamp (A1 floor fix, 2026-07-10; OFF when None == legacy) + the
        # SYMMETRIC gates_ceiling_m roof (vertical-structure fix, 2026-07-25; OFF when None). Sequential
        # so the walk continues from the clamped height (a post-dip climb actually climbs) instead of a
        # naive cumulative clamp that would pin every later gate to the floor. G <= ~8 -> loop is trivial.
        # torch.clamp(max=None) is a no-op, so the ceiling-OFF path is BYTE-IDENTICAL to the legacy one.
        _floor_z = R["gates_above_spawn_m"]
        _ceil_z = R["gates_ceiling_m"]
        if _floor_z is not None or _ceil_z is not None:
            min_z = None if _floor_z is None else float(_floor_z)
            max_z = None if _ceil_z is None else float(_ceil_z)
            if min_z is not None and max_z is not None and max_z < min_z:
                raise ValueError(f"gates_ceiling_m={max_z} must be >= gates_above_spawn_m={min_z}")
            z_prev = torch.zeros(m, device=device)                                   # pad z = 0
            for g in range(G):
                z_prev = torch.clamp(z_prev + dz[:, g], min=min_z, max=max_z)
                gate_pos[:, g, 2] = z_prev
        # gate yaw: bisector of incoming/outgoing headings; last gate = incoming heading. With the lateral
        # jog ON the heading walk no longer describes the actual legs, so re-derive the arriving headings
        # from the FINAL (offset) positions (prev[0]=spawn=origin) -- identical formula, but consistent
        # with the flown geometry (and hence with build_coarse_map, which reads the same final positions).
        # No RNG in the re-derivation, so the yaw-jitter draw below keeps its stream position.
        yaw = torch.empty(m, G, device=device)
        if lat_on:
            prev_f = torch.zeros_like(gate_pos)
            prev_f[:, 1:, :] = gate_pos[:, :-1, :]
            arr = gate_pos - prev_f                                                # arriving legs (final)
            h_arr = torch.atan2(arr[..., 1], arr[..., 0])                          # (m,G)
            yaw[:, :-1] = _circ_mean(h_arr[:, :-1], h_arr[:, 1:])
            yaw[:, -1] = h_arr[:, -1]
        else:
            yaw[:, :-1] = _circ_mean(headings[:, :-1], headings[:, 1:])
            yaw[:, -1] = headings[:, -1]
        yaw = yaw + U(-R["yaw_jitter_rad"], R["yaw_jitter_rad"], m, G)
        return gate_pos, yaw

    def too_close(gp):
        pts = torch.cat([torch.zeros(gp.shape[0], 1, 3, device=device), gp], dim=1)  # pad+gates
        d = torch.linalg.norm(pts[:, :, None, :2] - pts[:, None, :, :2], dim=-1)     # horizontal
        d = d + torch.eye(G + 1, device=device) * 1e9
        return d.amin(dim=(1, 2)) < R["min_pair_dist_m"]

    gate_pos, gate_yaw = draw(n)
    for _ in range(12):                       # rejection loop: redraw layouts with close pairs
        bad = too_close(gate_pos)
        if not bad.any():
            break
        m = int(bad.sum())
        rp, ry = draw(m)
        gate_pos[bad], gate_yaw[bad] = rp, ry
    bad = too_close(gate_pos)
    if bad.any():
        # exhausted-rejection fallback (vanishingly rare): a straight course ALWAYS satisfies
        # min separation (segment lengths >= seg_len_m[0] > min_pair_dist) -- never return a
        # silently-violating layout.
        saved = R["turn_rad"]
        R["turn_rad"] = 0.0
        rp, ry = draw(int(bad.sum()))
        R["turn_rad"] = saved
        gate_pos[bad], gate_yaw[bad] = rp, ry

    spawn_pos = torch.zeros(n, 3, device=device)
    # spawn yaw = tail-first toward gate 0 (gate_yaw+pi), PLUS an optional off-bearing jitter so the gate
    # appears across the FOV (Fengyou 2026-07-08). The jitter is on the drone's facing, NOT the layout, so
    # it changes the gate's relative bearing (real egocentric signal), unlike a global spawn_heading rotation.
    base_yaw = gate_yaw[:, 0] + np.pi
    if R["spawn_yaw_jitter_rad"] > 0.0:
        base_yaw = base_yaw + U(-R["spawn_yaw_jitter_rad"], R["spawn_yaw_jitter_rad"], n)
    spawn_yaw = torch.atan2(torch.sin(base_yaw), torch.cos(base_yaw))
    return {"gate_pos": gate_pos, "gate_yaw": gate_yaw,
            "spawn_pos": spawn_pos, "spawn_yaw": spawn_yaw}


def load_course_zup(path):
    """Load a fixed course json (e.g. the VQ1 hold-out) -> (gate_pos (G,3), gate_yaw (G,)) numpy."""
    data = json.loads(Path(path).read_text())
    gp = np.array([g["pos_zup"] for g in data["gates"]], dtype=np.float64)
    gy = np.array([g["yaw"] for g in data["gates"]], dtype=np.float64)
    return gp, gy


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", default="handoff/shadowpc-firstcontact-2026-06-02/track_map.json")
    ap.add_argument("--out", default="rl/peregrine_course_diffaero.json")
    args = ap.parse_args()

    gates = load_gates_ned(args.map)
    gate_pos, gate_yaw, diag = course_to_diffaero(gates)
    n = len(gates)

    print(f"loaded {n} gates from {args.map}\n")
    print(" gid |        NED bottom-centre        |   Z-up opening centre (DiffAero)   | yaw(quat) seg(deg)")
    for i, g in enumerate(gates):
        pn = np.asarray(g["position_ned"])
        zp = gate_pos[i]
        sy = diag["seg_yaw_deg"][i] if i < n - 1 else float("nan")
        print(f"  {g['gate_id']}  | [{pn[0]:8.2f},{pn[1]:6.2f},{pn[2]:7.2f}] | "
              f"[{zp[0]:8.2f},{zp[1]:6.2f},{zp[2]:7.2f}] | {diag['gate_yaw_deg'][i]:7.1f}  {sy:6.1f}")
    print(f"\nZ-up bounds: x[{gate_pos[:,0].min():.1f},{gate_pos[:,0].max():.1f}] "
          f"y[{gate_pos[:,1].min():.1f},{gate_pos[:,1].max():.1f}] "
          f"z[{gate_pos[:,2].min():.1f},{gate_pos[:,2].max():.1f}]")
    print(f"altitude (Z-up z of opening centres): g0={gate_pos[0,2]:+.2f} m -> g{n-1}={gate_pos[-1,2]:+.2f} m "
          f"(descends {gate_pos[0,2]-gate_pos[-1,2]:.1f} m)")

    out = {
        "source_map": str(args.map),
        "frame": "diffaero_zup (NED * [1,-1,-1]); gate_pos = opening centre; yaw rad about +Z",
        "n_gates": n,
        "gates": [
            {"gate_id": int(gates[i]["gate_id"]),
             "pos_zup": [float(x) for x in gate_pos[i]],
             "yaw": float(gate_yaw[i])}
            for i in range(n)
        ],
        "inner_opening_m": 1.5,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
