"""blur_probe.py — sim motion-blur determination (Phase 0, empirical), gate-edge version.

Motion blur is temporal: at a FIXED gate range, faster closing speed smears the gate edges, dropping
high-frequency content. Every gate on this course is the IDENTICAL red-square prop, so we measure
sharpness over the PROJECTED gate bounding box (predicted from GT pose + ODO attitude + the surveyed
map -- detector-free, fast) and POOL ALL GATES. That makes the content matched: a gate at range R
looks the same whichever gate it is, so binning by range and splitting by crossing-speed isolates the
speed-dependent (motion) blur from scene content and from range/JPEG effects.

Metric: variance-of-Laplacian over the projected gate bbox (focus measure). If the sim renders motion
blur, FAST-crossing gate boxes are blurrier than SLOW ones at matched range (ratio < 1, falling with
speed). Blur-free => ratio ~ 1.

Usage:
  PYTHONPATH=src .venv/Scripts/python.exe handoff/at-speed-sigma-2026-06-15/analysis/blur_probe.py \
      --glob 'data/runs/*_atspd_*' --speed-split 9
"""
from __future__ import annotations

import argparse
import bisect
import glob as globmod
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np

from racer import frames as F
from racer.navigator import load_track_map
from racer.recording import RecordingReader
from racer.vision.association import predict_gates_in_camera

MAP = ROOT / "handoff/shadowpc-firstcontact-2026-06-02/track_map.json"
W_IMG, H_IMG = 640, 360


def _nearest(keys, items, k):
    if not keys:
        return None
    i = bisect.bisect_left(keys, k)
    cands = [j for j in (i, i - 1) if 0 <= j < len(keys)]
    b = min(cands, key=lambda j: abs(keys[j] - k))
    return items[b]


def lap_var(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def process(sess: Path, gates):
    reader = RecordingReader(sess)
    meta = reader.meta
    t0u, t0m = int(meta["t0_unix_ns"]), int(meta["t0_monotonic_ns"])
    lpn, odo = [], []
    for msg in reader.iter_mavlink():
        t = msg.get_type()
        if t == "LOCAL_POSITION_NED":
            lpn.append((float(msg._timestamp), np.array([msg.x, msg.y, msg.z]),
                        np.array([msg.vx, msg.vy, msg.vz])))
        elif t == "ODOMETRY":
            odo.append((float(msg._timestamp), np.array([float(v) for v in msg.q])))
    lpn.sort(key=lambda r: r[0]); odo.sort(key=lambda r: r[0])
    lt = [r[0] for r in lpn]; ot = [r[0] for r in odo]
    rows, seen = [], set()
    with open(sess / "video.bin", "rb") as vf:
        for e in reader.iter_video_index():
            fid = e["frame_id"]
            if fid in seen:
                continue
            seen.add(fid)
            recv = (t0u + (e["recv_monotonic_ns"] - t0m)) / 1e9
            lp = _nearest(lt, lpn, recv); od = _nearest(ot, odo, recv)
            if lp is None or od is None:
                continue
            drone, vel = lp[1], lp[2]
            spd = float(np.linalg.norm(vel))
            R_wb = F.R_world_from_odo_quat_wxyz(od[1])
            pred = predict_gates_in_camera(gates, drone, R_wb)
            img = None
            for gid, pg in pred.items():
                if pg.corners_px is None or pg.t_cam_gate[2] <= 0:
                    continue
                c = np.asarray(pg.corners_px, float)
                x0, y0 = c[:, 0].min(), c[:, 1].min()
                x1, y1 = c[:, 0].max(), c[:, 1].max()
                # require the full projected gate box inside the image with a margin
                if x0 < 4 or y0 < 4 or x1 > W_IMG - 4 or y1 > H_IMG - 4:
                    continue
                if (x1 - x0) < 12 or (y1 - y0) < 12:
                    continue
                rng = float(np.linalg.norm(gates[gid].position_ned - drone))
                if img is None:
                    vf.seek(e["offset"]); jpeg = vf.read(e["length"])
                    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                    if img is None:
                        break
                    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                roi = gray_full[int(y0):int(y1), int(x0):int(x1)]
                if roi.size < 100:
                    continue
                rows.append(dict(gid=gid, rng=rng, spd=spd, lap=lap_var(roi),
                                 box_px=float((x1 - x0))))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", nargs="+", required=True)
    ap.add_argument("--map", default=str(MAP))
    ap.add_argument("--range-bins", default="8,12,16,20,24,28")
    ap.add_argument("--speed-split", type=float, default=9.0)
    args = ap.parse_args()

    gates = load_track_map(args.map, corner_to_center=True)
    sessions = sorted({Path(p) for pat in args.glob for p in globmod.glob(pat)})
    if not sessions:
        print("no sessions"); return 2
    all_rows = []
    for s in sessions:
        r = process(s, gates)
        all_rows += r
    sp = np.array([r["spd"] for r in all_rows])
    print(f"pooled {len(all_rows)} in-FoV gate-box observations from {len(sessions)} laps; "
          f"speed range {sp.min():.1f}-{sp.max():.1f} m/s (split @ {args.speed_split})")

    rb = [float(x) for x in args.range_bins.split(",")]
    thr = args.speed_split
    print(f"\n=== gate-box sharpness (Laplacian var) vs RANGE, pooled all gates ===")
    print(f"  motion blur => FAST < SLOW at matched range (ratio<1, falling). blur-free => ratio~1.")
    print(f"{'range bin':>12} {'N_slow':>7} {'spd_s':>6} {'lap_slow':>9} | {'N_fast':>7} {'spd_f':>6} {'lap_fast':>9} {'ratio':>7}")
    for lo, hi in zip(rb[:-1], rb[1:]):
        band = [r for r in all_rows if lo <= r["rng"] < hi]
        slow = [r for r in band if r["spd"] < thr]
        fast = [r for r in band if r["spd"] >= thr]
        if len(slow) < 4 or len(fast) < 4:
            print(f"  [{lo:>3g},{hi:>3g})  {len(slow):>7} {'--':>6} {'--':>9} | {len(fast):>7} {'--':>6} {'--':>9} {'(thin)':>7}")
            continue
        ss = np.median([r["spd"] for r in slow]); ls = np.median([r["lap"] for r in slow])
        sf = np.median([r["spd"] for r in fast]); lf = np.median([r["lap"] for r in fast])
        ratio = lf / ls if ls else float("nan")
        print(f"  [{lo:>3g},{hi:>3g})  {len(slow):>7} {ss:>6.1f} {ls:>9.1f} | {len(fast):>7} {sf:>6.1f} {lf:>9.1f} {ratio:>7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
