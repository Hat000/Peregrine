"""Full-COURSE frame-bundle exporter for characterize_perception.py (torch-free).

`scripts/export_frame_bundle.py` is hardcoded to GATE 0 (GATE0_MAP + 1.8-24 m gate-0 strata, the
Task-2 single-gate purpose). characterize_perception.py, by contrast, runs the navigator's chain
against the FULL track map (predicts/associates all 6 gates). To characterize perception across the
*whole* 6-gate course on a single session, we need a bundle whose frames span all gates / the full
range cycle -- which this builds.

Same alignment as export_frame_bundle: dedup video by frame_id, align each frame to the nearest
LOCAL_POSITION_NED (given world pos+vel) + ODOMETRY (attitude quat / rate) by RECV clock. Difference:
range_m = distance to the NEAREST gate centre (any of the 6), nearest_gate_id recorded. Selection =
even coverage across the full range span (linspace targets by range value, nearest unused frame each)
so the dense origin-hover / post-finish clusters don't dominate.

frames.json schema matches racer.frame_bundle/v1 (the keys characterize_perception reads:
frame_id, png, sim_time_ns, drone_position_ned, lpn_vel, odo_q_wxyz, odo_angular_rate, range_m,
speed_mps, pair_gap_ms, lpn_gap_ms, odo_gap_ms, sync_pos_err_m) + nearest_gate_id (extra, ignored
by the consumer).

Usage: .venv\\Scripts\\python handoff/perception-char-2026-06-08/scratch/export_course_bundle.py
       <run_dir> -o <out_dir> [--n 240] [--map MAP] [--commit SHA]
"""
from __future__ import annotations

import argparse
import bisect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np

from racer.navigator import load_track_map
from racer.recording import RecordingReader

DEFAULT_MAP = ROOT / "handoff/shadowpc-firstcontact-2026-06-02/track_map.json"


def _nearest(keys, items, k):
    if not keys:
        return None, None
    i = bisect.bisect_left(keys, k)
    cands = [j for j in (i, i - 1) if 0 <= j < len(keys)]
    b = min(cands, key=lambda j: abs(keys[j] - k))
    return items[b], abs(keys[b] - k)


def export(run: Path, out: Path, n: int, map_path: Path, commit: str | None,
           min_speed: float = 1.0, max_range: float = 26.0, per_gate: bool = False) -> list:
    reader = RecordingReader(run)
    meta = reader.meta
    t0u, t0m = int(meta["t0_unix_ns"]), int(meta["t0_monotonic_ns"])

    # gate opening-centres EXACTLY as characterize_perception sees them (corner_to_center=True)
    gates = load_track_map(str(map_path), corner_to_center=True)
    centers = np.array([g.position_ned for g in gates], float)
    gate_ids = [g.gate_id for g in gates]

    lpn, odo = [], []
    for msg in reader.iter_mavlink():
        t = msg.get_type()
        if t == "LOCAL_POSITION_NED":
            lpn.append((float(msg._timestamp), [float(msg.x), float(msg.y), float(msg.z)],
                        [float(msg.vx), float(msg.vy), float(msg.vz)]))
        elif t == "ODOMETRY":
            odo.append((float(msg._timestamp), [float(v) for v in msg.q],
                        [float(msg.rollspeed), float(msg.pitchspeed), float(msg.yawspeed)]))
    lpn.sort(key=lambda r: r[0]); odo.sort(key=lambda r: r[0])
    lpn_t = [r[0] for r in lpn]; odo_t = [r[0] for r in odo]

    seen, cand = set(), []
    for e in reader.iter_video_index():
        fid = e["frame_id"]
        if fid in seen:
            continue
        seen.add(fid)
        recv = (t0u + (e["recv_monotonic_ns"] - t0m)) / 1e9
        (lp, lg) = _nearest(lpn_t, lpn, recv)
        (od, og) = _nearest(odo_t, odo, recv)
        if lp is None or od is None:
            continue
        pos = np.array(lp[1]); vel = np.array(lp[2])
        dists = np.linalg.norm(centers - pos, axis=1)
        ng = int(np.argmin(dists)); rng = float(dists[ng])
        speed = float(np.linalg.norm(vel))
        pair_gap = max(lg, og) * 1000.0
        cand.append({
            "frame_id": fid, "entry": e,
            "pos": lp[1], "vel": lp[2], "q": od[1], "omega": od[2],
            "range_m": round(rng, 3), "nearest_gate_id": gate_ids[ng], "speed_mps": round(speed, 3),
            "lpn_gap_ms": round(lg * 1000, 2), "odo_gap_ms": round(og * 1000, 2),
            "pair_gap_ms": round(pair_gap, 2),
            "sync_pos_err_m": round(speed * pair_gap / 1000.0, 4),
            "sim_time_ns": int(e["sim_time_ns"]),
        })
    cand.sort(key=lambda c: c["range_m"])
    cand_r = [c["range_m"] for c in cand]
    print(f"{len(cand)} unique aligned frames; range-to-nearest-gate [{cand_r[0]:.2f}..{cand_r[-1]:.2f}] m")

    # The recording has a long static origin-hover (speed~0, nearest=gate0 @~23 m) + post-finish
    # drift that swamp the actual race. Keep only MOVING, in-band frames (the race itself).
    race = [c for c in cand if c["speed_mps"] > min_speed and c["range_m"] <= max_range]
    print(f"{len(race)} moving in-band frames (speed>{min_speed} m/s, range<={max_range} m)")

    def _sample_time(subset):
        subset = sorted(subset, key=lambda c: c["sim_time_ns"])
        if len(subset) <= n:
            return subset
        idx = np.linspace(0, len(subset) - 1, n).round().astype(int)
        return [subset[i] for i in dict.fromkeys(idx)]  # even-in-time, dedup preserves order

    bundles = []
    if per_gate:
        # One bundle per gate: keep ONLY frames whose nearest gate is G. This keeps the
        # detector's association unambiguous so characterize_perception's modal-gate robust
        # subset == this gate and its TAIL == genuine flips/wrong-gate (not just "other gate").
        for g in gate_ids:
            sub = _sample_time([c for c in race if c["nearest_gate_id"] == g])
            if not sub:
                print(f"  gate {g}: 0 frames, skipped"); continue
            bundles.append(_write_bundle(sub, Path(f"{out}_g{g}"), run, map_path, commit,
                                         f"per-gate (nearest=gate {g}) approach", primary_gate=g))
    else:
        bundles.append(_write_bundle(_sample_time(race), out, run, map_path, commit,
                                     "full-course (nearest of all 6 gates)", primary_gate=None))
    return bundles


def _write_bundle(chosen, out: Path, run: Path, map_path: Path, commit, what: str, primary_gate):
    chosen = sorted(chosen, key=lambda c: c["range_m"])
    out.mkdir(parents=True, exist_ok=True)
    vidf = open(run / "video.bin", "rb")
    frames_meta, dims = [], set()
    for i, c in enumerate(chosen):
        e = c["entry"]; vidf.seek(e["offset"]); jpeg = vidf.read(e["length"])
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]; dims.add((w, h))
        png = f"{i:03d}_id{c['frame_id']}_g{c['nearest_gate_id']}_{c['range_m']:.1f}m.png"
        cv2.imwrite(str(out / png), img)
        frames_meta.append({
            "frame_id": c["frame_id"], "png": png, "sim_time_ns": c["sim_time_ns"],
            "drone_position_ned": [round(x, 4) for x in c["pos"]],
            "lpn_vel": [round(x, 4) for x in c["vel"]],
            "odo_q_wxyz": [round(x, 6) for x in c["q"]],
            "odo_angular_rate": [round(x, 5) for x in c["omega"]],
            "range_m": c["range_m"], "nearest_gate_id": c["nearest_gate_id"], "speed_mps": c["speed_mps"],
            "pair_gap_ms": c["pair_gap_ms"], "lpn_gap_ms": c["lpn_gap_ms"],
            "odo_gap_ms": c["odo_gap_ms"], "sync_pos_err_m": c["sync_pos_err_m"],
        })
    vidf.close()
    bundle = {
        "schema": "racer.frame_bundle/v1",
        "run": run.name, "source_commit": commit, "primary_gate": primary_gate,
        "purpose": f"Perception characterization ({what}): navigator vision chain (YOLO->PnP) "
                   "world-fix error vs range. Built for characterize_perception.py.",
        "native_resolution_wh": list(dims.pop()) if len(dims) == 1 else sorted(dims),
        "map": str(map_path.relative_to(ROOT)) if str(map_path).startswith(str(ROOT)) else str(map_path),
        "range_to": "nearest of all 6 gate opening-centres (corner_to_center=True)",
        "n_frames": len(frames_meta),
        "range_span_m": [frames_meta[0]["range_m"], frames_meta[-1]["range_m"]],
        "pair_gap_ms_max": max(f["pair_gap_ms"] for f in frames_meta),
        "frames": frames_meta,
    }
    (out / "frames.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"  -> {out}  ({len(frames_meta)} frames, range "
          f"[{frames_meta[0]['range_m']:.1f}..{frames_meta[-1]['range_m']:.1f}] m)")
    return bundle


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--map", default=str(DEFAULT_MAP))
    ap.add_argument("--min-speed", type=float, default=1.0)
    ap.add_argument("--max-range", type=float, default=26.0)
    ap.add_argument("--per-gate", action="store_true",
                    help="write one bundle per gate (<out>_g0..g5); keeps modal-gate/TAIL meaningful")
    ap.add_argument("--commit", default=None)
    args = ap.parse_args()
    export(Path(args.run), Path(args.out), args.n, Path(args.map), args.commit,
           min_speed=args.min_speed, max_range=args.max_range, per_gate=args.per_gate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
