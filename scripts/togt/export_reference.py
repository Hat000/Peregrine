"""Export a refined TOGT trajectory as the reusable VQ1 reference line (NED), for
(a) the RL progress-reward reference and (b) the explicit line of a decomposed
plan+track architecture. Schema doc: src/racer/reference_line.py (the loader).

Frames: the TOGT pipeline runs in the diffaero z-up frame (NED * [1,-1,-1], body FLU);
this exporter converts everything to the stack's world NED / body FRD conventions:
  pos/vel/acc:  [x, -y, -z]
  yaw:          -yaw_zup  (heading vector [cos a, sin a]: y flips)
  quaternion:   q_ned = qD * q * qD with qD = (0,1,0,0)  (conjugation by diag(1,-1,-1),
                a pi-rotation about x, applied world-side and body-side: FLU->FRD)
  body rates:   [wx, -wy, -wz]
Acceleration is recomputed EXACTLY from the refine dynamics (a = c*R@ez - G*ez - drag*v,
c = sum(rotor thrusts)/mass), not finite-differenced. `thrust_norm` is the live normalized
collective (hover 0.2656 <-> 1 g; ceiling 1.0 <-> 3.765 g).

Usage: python scripts/togt/export_reference.py [--case .../cases/ref_margin]
           [--out rl/reference_line_vq1.json]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze import load_csv, gate_crossings, G  # noqa: E402  (sibling module)

HOVER_NORM = 0.2656     # live normalized collective at 1 g


def quat_zup_flu_to_ned_frd(q: np.ndarray) -> np.ndarray:
    """(N,4) wxyz, world zup/body FLU -> world NED/body FRD: q' = qD * q * qD, qD=(0,1,0,0)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # qD*q = (0,1,0,0)*(w,x,y,z) = (-x, w, z, -y)   [Hamilton product]
    w1, x1, y1, z1 = -x, w, z, -y
    # (w1,x1,y1,z1)*qD = (-x1, w1, z1, -y1)
    return np.stack([-x1, w1, z1, -y1], axis=1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="handoff/laptop-togt-bound-2026-06-10/cases/ref_margin")
    ap.add_argument("--course", default="rl/peregrine_course_diffaero.json")
    ap.add_argument("--out", default="rl/reference_line_vq1.json")
    args = ap.parse_args()

    case = Path(args.case)
    meta = json.loads((case / "meta.json").read_text())
    course = json.loads(Path(args.course).read_text())
    d = load_csv(case / "refined_traj.csv")

    t = d["t"]
    P = np.stack([d["p_x"], d["p_y"], d["p_z"]], axis=1)
    V = np.stack([d["v_x"], d["v_y"], d["v_z"]], axis=1)
    Q = np.stack([d["q_w"], d["q_x"], d["q_y"], d["q_z"]], axis=1)
    W = np.stack([d["w_x"], d["w_y"], d["w_z"]], axis=1)
    U = np.stack([d["u_1"], d["u_2"], d["u_3"], d["u_4"]], axis=1)
    c = U.sum(axis=1)                                   # collective specific force (m/s^2, mass 1)
    drag = float(meta.get("linear_drag", 0.0))

    # exact accel from the refine dynamics (z-up frame)
    w, x, y, z = Q.T
    bz = np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), w * w - x * x - y * y + z * z], axis=1)
    A = c[:, None] * bz - np.array([0.0, 0.0, G]) - drag * V

    crossings = gate_crossings(P, t, course["gates"])
    lap = crossings[-1]["t"] if crossings[-1] else None

    flip = np.array([1.0, -1.0, -1.0])
    q_ned = quat_zup_flu_to_ned_frd(Q)
    yaw_zup = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    out = {
        "schema": "peregrine.reference_line.v1",
        "frame": "world NED (X north, Y east, Z down), body FRD; yaw = NED heading (atan2 E,N)",
        "source": {
            "case": meta["case"], "generator": "TOGT-Planner planTOGT + multiple-shooting refine",
            "handoff": "laptop-togt-bound-2026-06-10",
            "plant": {"thrust_to_weight": meta["thrust_to_weight"], "omega_max": meta["omega_max"],
                      "linear_drag": drag, "gate_margin_m": meta["gate_margin_m"],
                      "refine_tol_m": meta["refine_tol_m"]},
        },
        "lap_time_s": lap,
        "total_duration_s": float(t[-1]),
        "gate_crossings": [
            None if cr is None else {
                "gate_id": gi, "t": cr["t"],
                "pos_ned": [cr["pos"][0], -cr["pos"][1], -cr["pos"][2]],
                "miss_m": cr["miss"], "miss_h_m": cr["miss_h"], "miss_v_m": cr["miss_v"],
            } for gi, cr in enumerate(crossings)],
        "t": np.round(t, 5).tolist(),
        "pos_ned": np.round(P * flip, 5).tolist(),
        "vel_ned": np.round(V * flip, 5).tolist(),
        "acc_ned": np.round(A * flip, 4).tolist(),
        "yaw": np.round(-yaw_zup, 5).tolist(),
        "quat_wxyz": np.round(q_ned, 6).tolist(),
        "omega_frd": np.round(W * flip, 5).tolist(),
        "thrust_norm": np.round(c * HOVER_NORM / G, 5).tolist(),
    }

    text = json.dumps(out, separators=(",", ":"))
    Path(args.out).write_text(text + "\n")
    n_kb = len(text) / 1024
    print(f"{args.out}: case={meta['case']} lap={lap:.3f}s total={t[-1]:.3f}s "
          f"samples={len(t)} ({n_kb:.0f} KB)")
    for gc in out["gate_crossings"]:
        print(f"  gate {gc['gate_id']}: t={gc['t']:.3f}s miss={gc['miss_m']:.3f}m "
              f"(h {gc['miss_h_m']:+.3f}, v {gc['miss_v_m']:+.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
