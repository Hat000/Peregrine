"""Analyze TOGT bound cases: lap time at the gate-5 plane crossing, per-gate in-plane miss,
and which constraint binds (thrust vs body rates), per case dir produced by run_cases.sh.

Lap-time definition (matches the VQ1 race clock): t=0 with the drone AT REST on the spawn
pad; the lap ends when the trajectory crosses the LAST gate's plane. The planned trajectory
continues ~25 m past gate 5 to a virtual endpoint so the forced terminal state cannot slow
the finish crossing -- total trajectory duration is therefore NOT the lap time.

Outputs cases/<name>/analysis.json + a combined markdown table on stdout (and optionally
to --table-out). Pure numpy + stdlib.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

G = 9.8066


def load_csv(path: Path) -> dict[str, np.ndarray]:
    raw = np.genfromtxt(path, dtype=float, delimiter=",", names=True)
    return {name: np.asarray(raw[name], dtype=float) for name in raw.dtype.names}


def quat_rotate_z(qw, qx, qy, qz):
    """Body +Z axis expressed in world coords for quaternion (w,x,y,z): R(q)[:,2]."""
    return np.stack([
        2 * (qx * qz + qw * qy),
        2 * (qy * qz - qw * qx),
        qw * qw - qx * qx - qy * qy + qz * qz,
    ], axis=-1)


def gate_crossings(P: np.ndarray, t: np.ndarray, gates: list[dict]) -> list[dict | None]:
    """Sequential gate passages: for each gate (in order), among ALL crossings of its
    (infinite) plane after the previous gate's passage, the one with MINIMUM in-plane miss
    -- a fast line can cut a gate's plane far outside the opening while lining up, so the
    first crossing is not the passage (same convention as twin_fly_course.gate_plane_miss)."""
    out: list[dict | None] = []
    t_prev = -np.inf
    for g in gates:
        c = np.asarray(g["pos_zup"], dtype=float)
        yaw = float(g["yaw"])
        n = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        u_h = np.array([-math.sin(yaw), math.cos(yaw), 0.0])   # horizontal in-plane axis
        u_v = np.array([0.0, 0.0, 1.0])                        # vertical in-plane axis
        d = (P - c) @ n
        idx = np.flatnonzero((np.sign(d[:-1]) * np.sign(d[1:]) < 0) & (t[1:] > t_prev))
        best = None
        for i in idx:
            f = d[i] / (d[i] - d[i + 1])
            pc = P[i] + f * (P[i + 1] - P[i])
            tc = float(t[i] + f * (t[i + 1] - t[i]))
            mh, mv = float((pc - c) @ u_h), float((pc - c) @ u_v)
            cand = {"t": tc, "miss": float(math.hypot(mh, mv)), "miss_h": mh, "miss_v": mv,
                    "pos": [float(x) for x in pc]}
            if best is None or cand["miss"] < best["miss"]:
                best = cand
        out.append(best)
        if best is not None:
            t_prev = best["t"]
    return out


def analyze_traj(csv_path: Path, gates: list[dict], tw: float, omega_max: list[float]) -> dict:
    d = load_csv(csv_path)
    t = d["t"]
    P = np.stack([d["p_x"], d["p_y"], d["p_z"]], axis=1)
    V = np.stack([d["v_x"], d["v_y"], d["v_z"]], axis=1)
    W = np.stack([d["w_x"], d["w_y"], d["w_z"]], axis=1)
    U = np.stack([d["u_1"], d["u_2"], d["u_3"], d["u_4"]], axis=1)
    collective = U.sum(axis=1)                      # N on 1 kg
    coll_max = tw * G
    speed = np.linalg.norm(V, axis=1)
    bz = quat_rotate_z(d["q_w"], d["q_x"], d["q_y"], d["q_z"])
    tilt = np.degrees(np.arccos(np.clip(bz[:, 2], -1, 1)))
    dt = np.diff(t)
    dt[dt <= 0] = np.nan
    wdot = np.abs(np.diff(W, axis=0) / dt[:, None])

    crossings = gate_crossings(P, t, gates)
    lap = crossings[-1]["t"] if crossings[-1] else None
    wxy = np.abs(W[:, :2]).max(axis=1)
    wz = np.abs(W[:, 2])
    n = len(t)
    return {
        "lap_time_s": lap,
        "total_duration_s": float(t[-1]),
        "crossings": crossings,
        "max_miss_m": max((c["miss"] for c in crossings if c), default=None),
        "max_speed_mps": float(speed.max()),
        "speed_at_finish_mps": float(speed[np.searchsorted(t, lap)]) if lap else None,
        "max_tilt_deg": float(tilt.max()),
        "max_collective_g": float(collective.max() / G),
        "frac_nodes_collective_sat": float((collective >= 0.99 * coll_max).mean()),
        "max_omega_xy": float(wxy.max()),
        "frac_nodes_omega_xy_sat": float((wxy >= 0.99 * omega_max[0]).mean()),
        "max_omega_z": float(wz.max()),
        "frac_nodes_omega_z_sat": float((wz >= 0.99 * omega_max[2]).mean()),
        "max_wdot_xy_rps2": float(np.nanmax(wdot[:, :2])) if n > 2 else None,
        "max_wdot_z_rps2": float(np.nanmax(wdot[:, 2])) if n > 2 else None,
        "n_samples": n,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", default="handoff/laptop-togt-bound-2026-06-10/cases")
    ap.add_argument("--course", default="rl/peregrine_course_diffaero.json")
    ap.add_argument("--table-out", default=None)
    args = ap.parse_args()

    course = json.loads(Path(args.course).read_text())
    gates = course["gates"]
    rows = []
    for d in sorted(Path(args.cases).iterdir()):
        if not (d / "meta.json").exists():
            continue
        meta = json.loads((d / "meta.json").read_text())
        tw, omg = meta["thrust_to_weight"], meta["omega_max"]
        entry: dict = {"case": meta["case"], "meta": meta}
        togt_dur = None
        ps = d / "planner_stdout.txt"
        if ps.exists():
            for line in ps.read_text().splitlines():
                if line.startswith("Duration:"):
                    togt_dur = float(line.split()[1])
        entry["togt_init_duration_s"] = togt_dur
        if (d / "togt_traj.csv").exists():
            entry["togt_init"] = analyze_traj(d / "togt_traj.csv", gates, tw, omg)
        if (d / "refine_summary.json").exists():
            entry["refine_ipopt"] = json.loads((d / "refine_summary.json").read_text())
        if (d / "refined_traj.csv").exists():
            entry["refined"] = analyze_traj(d / "refined_traj.csv", gates, tw, omg)
        (d / "analysis.json").write_text(json.dumps(entry, indent=1))
        rows.append(entry)

    hdr = ("| case | T/W | omega_xy | margin | TOGT lap (s) | REFINED lap (s) | max miss (m) | "
           "vmax (m/s) | coll sat % | omega_xy sat % | ipopt |")
    sep = "|" + "---|" * 11
    lines = [hdr, sep]
    for e in rows:
        m = e["meta"]
        ti = e.get("togt_init", {})
        rf = e.get("refined", {})
        ip = e.get("refine_ipopt", {})
        lines.append(
            f"| {m['case']} | {m['thrust_to_weight']:.2f} | {m['omega_max'][0]:g} | "
            f"{m['gate_margin_m']:g} | "
            f"{(ti.get('lap_time_s') or float('nan')):.2f} | "
            f"{(rf.get('lap_time_s') or float('nan')):.2f} | "
            f"{(rf.get('max_miss_m') if rf.get('max_miss_m') is not None else float('nan')):.2f} | "
            f"{(rf.get('max_speed_mps') or float('nan')):.1f} | "
            f"{100 * (rf.get('frac_nodes_collective_sat') or 0):.0f} | "
            f"{100 * (rf.get('frac_nodes_omega_xy_sat') or 0):.0f} | "
            f"{ip.get('ipopt_status', '?')} |")
    table = "\n".join(lines)
    print(table)
    if args.table_out:
        Path(args.table_out).write_text(table + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
