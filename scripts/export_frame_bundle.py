"""Offline: export a small FPV frame bundle (raw PNGs + frames.json) for the laptop's vision-PnP
gate-center verdict (Task 2). The laptop has ultralytics + the PnP pipeline; this just hands it
audited frames + the pristine given pose/attitude per frame.

Selection: dedup video by frame_id, align each frame to the nearest LOCAL_POSITION_NED (given world
pos + vel) and ODOMETRY (attitude quat) by RECEIVE time (video<->telemetry share the recv clock via
the meta monotonic<->unix bridge; the in-frame sim/IMU clocks are independent). Then STRATIFY by
range-to-gate so the bundle spans far (~20 px) -> near (~270 px), weighted to the corner-localizing
sweet spot but keeping both extremes (so the laptop can check the PnP center is range-CONSISTENT --
a center that drifts with range flags wrong intrinsics or inner/outer-size confusion).

Native FPV res is 640x360 (JPEG headers); intrinsics frames.CAMERA_INTRINSICS_K match as-is.

Per frame in frames.json: frame_id, png, sim_time_ns (video frame clock), drone_position_ned (LPN,
given world), lpn_vel (world; to bound motion/sync error = speed*pair_gap), odo_q_wxyz (attitude),
odo_angular_rate (to bound rotation sync error = range*omega*pair_gap), range_m, speed_mps,
pair_gap_ms (max of lpn/odo recv gap), lpn_gap_ms, odo_gap_ms, sync_pos_err_m (= speed*pair_gap).

Usage: python scripts/export_frame_bundle.py <run_dir> -o <out_dir> [--n 40] [--commit <sha>]
"""
from __future__ import annotations

import argparse
import bisect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np

from racer.recording import RecordingReader

GATE0_MAP = [-23.30, -0.40, -0.03]
# range strata (m): (lo, hi, target_count) -- bulk in the 3-12 m sweet spot, handful at each extreme.
# Near floor 1.8 m (below that the inner opening >270 px overflows the frame -> corners clip).
STRATA = [(1.8, 3.0, 4), (3.0, 5.0, 7), (5.0, 8.0, 9),
          (8.0, 12.0, 9), (12.0, 18.0, 6), (18.0, 24.0, 5)]


def _nearest(keys, items, k):
    if not keys:
        return None, None
    i = bisect.bisect_left(keys, k)
    cands = [j for j in (i, i - 1) if 0 <= j < len(keys)]
    b = min(cands, key=lambda j: abs(keys[j] - k))
    return items[b], abs(keys[b] - k)


def export(run: Path, out: Path, n: int, commit: str | None) -> dict:
    reader = RecordingReader(run)
    meta = reader.meta
    t0u, t0m = int(meta["t0_unix_ns"]), int(meta["t0_monotonic_ns"])
    gate = np.array(GATE0_MAP, float)

    lpn, odo = [], []
    for msg in reader.iter_mavlink():
        t = msg.get_type()
        if t == "LOCAL_POSITION_NED":
            lpn.append((float(msg._timestamp),
                        [float(msg.x), float(msg.y), float(msg.z)],
                        [float(msg.vx), float(msg.vy), float(msg.vz)]))
        elif t == "ODOMETRY":
            odo.append((float(msg._timestamp), [float(v) for v in msg.q],
                        [float(msg.rollspeed), float(msg.pitchspeed), float(msg.yawspeed)]))
    lpn.sort(key=lambda r: r[0]); odo.sort(key=lambda r: r[0])
    lpn_t = [r[0] for r in lpn]
    odo_t = [r[0] for r in odo]

    # dedup video by frame_id (keep first), align pose, compute range
    seen = set()
    cand = []
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
        rng = float(np.linalg.norm(gate - pos))
        speed = float(np.linalg.norm(vel))
        pair_gap = max(lg, og) * 1000.0
        cand.append({
            "frame_id": fid, "entry": e, "recv": recv,
            "pos": lp[1], "vel": lp[2], "q": od[1], "omega": od[2],
            "range_m": round(rng, 3), "speed_mps": round(speed, 3),
            "lpn_gap_ms": round(lg * 1000, 2), "odo_gap_ms": round(og * 1000, 2),
            "pair_gap_ms": round(pair_gap, 2),
            "sync_pos_err_m": round(speed * pair_gap / 1000.0, 4),
            "sim_time_ns": int(e["sim_time_ns"]),
        })
    cand.sort(key=lambda c: c["range_m"])
    cand_r = [c["range_m"] for c in cand]
    print(f"{len(cand)} unique frames; range [{cand_r[0]:.2f}..{cand_r[-1]:.2f}] m")

    # stratified pick: within each band, nearest frame to evenly-spaced TARGET RANGES (by value, not
    # index) -> even range coverage + diverse viewpoints even where one range is over-represented
    # (e.g. the static origin cluster at 23.3 m).
    chosen, used = [], set()
    for lo, hi, cnt in STRATA:
        band = [c for c in cand if lo <= c["range_m"] < hi]
        if not band:
            continue
        bmin, bmax = band[0]["range_m"], band[-1]["range_m"]
        targets = np.linspace(bmin, bmax, cnt) if cnt > 1 else [(bmin + bmax) / 2]
        band_r = [c["range_m"] for c in band]
        for tr in targets:
            j = int(np.argmin(np.abs(np.array(band_r) - tr)))
            # nudge off an already-used frame to the nearest unused in-band
            order = sorted(range(len(band)), key=lambda k: abs(band_r[k] - tr))
            for k in order:
                if band[k]["frame_id"] not in used:
                    used.add(band[k]["frame_id"]); chosen.append(band[k]); break
    chosen.sort(key=lambda c: c["range_m"])
    print(f"selected {len(chosen)} frames across strata")

    # write PNGs + frames.json
    out.mkdir(parents=True, exist_ok=True)
    vidf = open(run / "video.bin", "rb")
    frames_meta = []
    dims = set()
    for i, c in enumerate(chosen):
        e = c["entry"]
        vidf.seek(e["offset"]); jpeg = vidf.read(e["length"])
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]; dims.add((w, h))
        png = f"{i:02d}_id{c['frame_id']}_{c['range_m']:.1f}m.png"
        cv2.imwrite(str(out / png), img)
        frames_meta.append({
            "frame_id": c["frame_id"], "png": png, "sim_time_ns": c["sim_time_ns"],
            "drone_position_ned": [round(x, 4) for x in c["pos"]],
            "lpn_vel": [round(x, 4) for x in c["vel"]],
            "odo_q_wxyz": [round(x, 6) for x in c["q"]],
            "odo_angular_rate": [round(x, 5) for x in c["omega"]],
            "range_m": c["range_m"], "speed_mps": c["speed_mps"],
            "pair_gap_ms": c["pair_gap_ms"], "lpn_gap_ms": c["lpn_gap_ms"],
            "odo_gap_ms": c["odo_gap_ms"], "sync_pos_err_m": c["sync_pos_err_m"],
        })
    vidf.close()

    bundle = {
        "schema": "racer.frame_bundle/v1",
        "run": run.name, "source_commit": commit,
        "purpose": "Task 2 vision-PnP gate-0 center: detector->corners->estimate_gate_pose(inner 1.5m)"
                   "->world via given pose + frames.R_camera_from_body->median; compare to gate map.",
        "native_resolution_wh": list(dims.pop()) if len(dims) == 1 else sorted(dims),
        "intrinsics": "frames.CAMERA_INTRINSICS_K as-is (fx=fy=320, cx=320, cy=180, HFoV 90deg) -- "
                      "matches native 640x360, NO scaling needed.",
        "camera_extrinsics": "frames.R_camera_from_body() (camera pitched +20deg up; optical X-right,"
                             " Y-down, Z-fwd).",
        "gate0_map_ned": GATE0_MAP,
        "pose_alignment": "per frame, nearest LPN (pos/vel) + ODOMETRY (quat/rate) by RECV clock "
                          "(video<->telemetry bridge); pair_gap_ms = max recv gap.",
        "verdict_hint": "report full 3-axis offset PnP_center - map (N/E/D): |horizontal| -> "
                        "corner(+-1.36m)-vs-center; vertical -> is the bottom-edge->center z-anchor "
                        "(-1.36) right. CHECK center is consistent across range_m (drift => wrong "
                        "intrinsics / inner-vs-outer size confusion).",
        "error_budget": "motion/sync pos err ~ speed_mps*pair_gap (sync_pos_err_m); rotation sync "
                        "err ~ range_m*|omega|*pair_gap. Both << 1.36 m corner margin; weight frames "
                        "by these + apparent gate px if desired.",
        "n_frames": len(frames_meta),
        "range_span_m": [frames_meta[0]["range_m"], frames_meta[-1]["range_m"]],
        "pair_gap_ms_max": max(f["pair_gap_ms"] for f in frames_meta),
        "frames": frames_meta,
    }
    (out / "frames.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return bundle


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--commit", default=None)
    args = ap.parse_args()
    b = export(Path(args.run), Path(args.out), args.n, args.commit)
    print(f"\nbundle: {b['n_frames']} frames, range {b['range_span_m']} m, "
          f"native {b['native_resolution_wh']}, pair_gap_max {b['pair_gap_ms_max']}ms")
    print(f"  -> {args.out}/frames.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
