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
)

# VQ1 standing-start pad (Z-up), from the S1.2 live recordings (see peregrine_racing.py).
VQ1_SPAWN_POS_ZUP = (0.0, 0.0, -0.02)
VQ1_SPAWN_YAW = 0.0          # body yaw at spawn; gates at yaw pi => tail-first (gate-frame yaw ~ pi)
VQ1_SPAWN_PITCH_RAD = -0.31  # -17.8 deg tilted pad (measured)

# ---- Difficulty presets (override DEFAULT_COURSE_RANGES for named difficulty tiers) -----------
# Usage in training: +env.track_difficulty=easy   (PeregrineRacing reads this and calls
# sample_courses(**DIFFICULTY_PRESETS[difficulty]) in _assign_courses).
# "vq1_like" = constrained to roughly VQ1 geometry -- useful early in training; "easy" and "medium"
# are the main curriculum presets; "hard" uses the full DEFAULT range + tighter gates.
DIFFICULTY_PRESETS = {
    # Nearly-straight, moderate-length, mostly-descending course ~ VQ1 character.
    "vq1_like": dict(
        seg_len_m=(20.0, 40.0),
        turn_rad=0.35,           # +-20 deg max per segment (VQ1 <= ~20 deg)
        drop_m=(0.0, 11.0),      # only descending (VQ1 is all descent)
        max_grade=0.32,
        yaw_jitter_rad=0.18,
        spawn_dist_m=(20.0, 26.0),
        spawn_below_g0_m=(1.0, 2.0),
        min_pair_dist_m=12.0,
    ),
    # Gentle turns, moderate segments -- for bootstrapping a fresh policy.
    "easy": dict(
        seg_len_m=(20.0, 40.0),
        turn_rad=0.52,           # +-30 deg
        drop_m=(-2.0, 10.0),
        max_grade=0.38,
        yaw_jitter_rad=0.18,
        spawn_dist_m=(18.0, 26.0),
        spawn_below_g0_m=(0.8, 2.2),
        min_pair_dist_m=12.0,
    ),
    # Default ranges (DEFAULT_COURSE_RANGES): no overrides needed, kept as alias.
    "medium": dict(),
    # Full 60-deg turns, short segments, level or climbing segments -- hardest generalisation.
    "hard": dict(
        seg_len_m=(12.0, 30.0),  # shorter => tighter turns at speed
        turn_rad=1.047,          # +-60 deg (full budget)
        drop_m=(-5.0, 12.0),     # climbs allowed too
        max_grade=0.50,
        yaw_jitter_rad=0.21,
        spawn_dist_m=(15.0, 28.0),
        spawn_below_g0_m=(0.5, 2.5),
        min_pair_dist_m=8.0,
    ),
}


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
        # heading random walk: heading[k] = direction of segment k (segment 0 = pad->gate0)
        h0 = U(-np.pi, np.pi, m, 1)
        turns = U(-R["turn_rad"], R["turn_rad"], m, G - 1)
        headings = torch.cat([h0, h0 + torch.cumsum(turns, dim=1)], dim=1)          # (m, G)
        seg_len = torch.empty(m, G, device=device)
        seg_len[:, 0] = U(*R["spawn_dist_m"], m)
        seg_len[:, 1:] = U(*R["seg_len_m"], m, G - 1)
        dz = torch.empty(m, G, device=device)
        dz[:, 0] = U(*R["spawn_below_g0_m"], m)                                     # gate 0 ABOVE pad
        drop = U(*R["drop_m"], m, G - 1)
        dz[:, 1:] = -torch.clamp(drop, -R["max_grade"] * seg_len[:, 1:],
                                 R["max_grade"] * seg_len[:, 1:])
        seg = torch.stack([seg_len * torch.cos(headings),
                           seg_len * torch.sin(headings), dz], dim=-1)              # (m, G, 3)
        gate_pos = torch.cumsum(seg, dim=1)                                         # pad at origin
        # gate yaw: bisector of incoming/outgoing headings; last gate = incoming heading
        yaw = torch.empty(m, G, device=device)
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
    spawn_yaw = torch.atan2(torch.sin(gate_yaw[:, 0] + np.pi), torch.cos(gate_yaw[:, 0] + np.pi))
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
