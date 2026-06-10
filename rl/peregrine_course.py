"""Convert the VQ1 captured gate map (NED) into a DiffAero-frame fixed gate layout for the RL env.

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
