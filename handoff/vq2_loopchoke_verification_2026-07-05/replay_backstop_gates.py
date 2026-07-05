"""replay_backstop_gates.py — measure the REAL acceptance rate of the two CPU-heavy CV backstops
(vp_yaw / floor_height) that run on the flight loop thread, by replaying recorded flight video
through the EXACT live code paths.

WHY. The control loop chokes (16-18 Hz vs 30 Hz). Two suspects run on the loop thread:
  * vp_yaw     (racer.vision.heading_vp.estimate_heading, ~72ms/call, every 5th vision tick)
  * floor_height (racer.vision.floor_height.estimate_floor_height, ~41ms/call, every 3rd tick)
HYPOTHESIS: in the dark VQ2 warehouse both nearly always FAIL their quality gates -> we pay the
compute and almost never apply a correction -> dead weight, safe to cut.

WHAT THIS DOES. For each recorded run it decodes video.bin frames (render-free, reusing the decode
from handoff/tools/render_cmd_indicator.py), feeds each frame the time-nearest nav_estimate
roll/pitch (mirroring what the LIVE navigator feeds the estimators via _current_true_rpy), calls the
two estimators, and replicates the EXACT acceptance-gate logic from navigator._apply_vp_yaw /
_apply_floor_height (navigator.py ~L858 / ~L896). It records, per estimator: n_tried, n_would_apply,
n_rejected broken down by reason, wall-clock ms/call, and (for vp_yaw) the yaw-innovation magnitude
of accepted corrections.

The ESKF/KF update itself is skipped (we only need would-have-applied vs rejected + reason). This is
OFFLINE + CPU-ONLY: no torch, no CUDA, no YOLO. The two estimators are classical OpenCV.

TIME MAPPING (the load-bearing subtlety). nav_estimate.sim_time_ns is on the SIM-BOOT clock and
covers the ARMED FLIGHT ONLY (~120s). video_index.sim_time_ns is on a different (epoch) clock and
covers the FULL recording incl. the pre-GO waiting room (~141s). They cannot be matched directly,
but BOTH streams END at disarm/crash (same as render_cmd_indicator's end-anchor trick). So we map a
video frame to nav-time by seconds-FROM-THE-END:
    target_nav_ns = nav_end_ns - (video_end_ns - frame_video_ns)
then take the nearest nav record. Frames whose target falls before the first armed nav tick are the
pre-GO waiting room and are dropped (the live estimators never ran on them under this config anyway,
but more importantly we have no attitude for them). This is approximate (+/- recorder tail) but is
the same anchor the project's own render tool uses, and roll/pitch move slowly relative to the tail.

USAGE:
    python replay_backstop_gates.py <run_dir> [<run_dir> ...] [--stride N] [--max-frames M]
"""
from __future__ import annotations

import argparse
import bisect
import json
import os
import sys
import time
from pathlib import Path

# HARD: never touch the GPU. A reconstruction job owns it. Belt-and-suspenders even though these
# classical modules import no torch.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# Repo src on path (this file lives at <repo>/handoff/vq2_loopchoke_verification_2026-07-05/).
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from racer.vision.heading_vp import estimate_heading  # noqa: E402
from racer.vision.floor_height import estimate_floor_height  # noqa: E402

# ---- Live config values (navigator.py NavConfig defaults; brief-confirmed) -------------------
VP_YAW_MIN_QUALITY = 0.30
VP_YAW_BRANCH_MAX_RAD = float(np.deg2rad(35.0))
VP_YAW_RANSAC_ITERS = 256
FLOOR_MIN_QUALITY = 0.30
FLOOR_MAX_STD_M = 0.50
FLOOR_GRID_CELL_M = 2.0


def _wrap_pi(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def load_nav(run: Path):
    """nav_estimate.jsonl -> (times_ns sorted, list of records aligned to times)."""
    recs = []
    for line in (run / "nav_estimate.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        recs.append(r)
    recs.sort(key=lambda r: int(r["sim_time_ns"]))
    times = [int(r["sim_time_ns"]) for r in recs]
    return times, recs


def load_video_index(run: Path):
    idx = [json.loads(l) for l in (run / "video_index.jsonl").read_text().splitlines() if l.strip()]
    idx.sort(key=lambda r: int(r["sim_time_ns"]))
    return idx


def nearest_nav(nav_times, nav_recs, target_ns):
    """Nav record whose sim_time_ns is nearest target_ns (mirrors 'time-nearest nav record')."""
    i = bisect.bisect_left(nav_times, target_ns)
    if i <= 0:
        return nav_recs[0]
    if i >= len(nav_times):
        return nav_recs[-1]
    lo, hi = nav_times[i - 1], nav_times[i]
    return nav_recs[i - 1] if (target_ns - lo) <= (hi - target_ns) else nav_recs[i]


def replay_run(run: Path, stride: int, max_frames: int):
    nav_times, nav_recs = load_nav(run)
    idx = load_video_index(run)
    blob = (run / "video.bin").read_bytes()

    nav_start_ns, nav_end_ns = nav_times[0], nav_times[-1]
    video_end_ns = int(idx[-1]["sim_time_ns"])

    # Map each frame to a nav-time by seconds-from-the-end; keep only frames inside the armed window.
    mapped = []  # (frame_rec, target_nav_ns, nav_rec)
    for rec in idx:
        target = nav_end_ns - (video_end_ns - int(rec["sim_time_ns"]))
        # Drop pre-GO waiting-room frames (target before armed flight, small margin like render tool).
        if target < nav_start_ns - 200_000_000:
            continue
        target = max(target, nav_start_ns)
        nav_rec = nearest_nav(nav_times, nav_recs, target)
        mapped.append((rec, target, nav_rec))

    # Subsample by stride, cap at max_frames, but keep n >= 500 if available.
    sampled = mapped[::stride]
    if len(sampled) > max_frames:
        # even-thin to max_frames
        keep_step = len(sampled) / max_frames
        sampled = [sampled[int(k * keep_step)] for k in range(max_frames)]
    if len(sampled) < 500 and len(mapped) >= 500:
        # stride too coarse; fall back to a stride that yields ~min(max_frames, len) frames
        target_n = min(max_frames, len(mapped))
        step = max(1, len(mapped) // target_n)
        sampled = mapped[::step][:max_frames]

    stats = {
        "run": run.name,
        "n_video_frames_total": len(idx),
        "n_frames_in_armed_window": len(mapped),
        "n_frames_sampled": len(sampled),
        "stride_effective": (len(mapped) / len(sampled)) if sampled else None,
        "vp": {
            "n_tried": 0, "n_would_apply": 0,
            "rej_no_estimate": 0, "rej_low_quality": 0, "rej_branch": 0,
            "ms": [], "innov_rad": [], "quality_all": [],
        },
        "floor": {
            "n_tried": 0, "n_would_apply": 0,
            "rej_no_estimate": 0, "rej_low_quality": 0, "rej_high_std": 0,
            "ms": [], "height_m": [], "std_m_all": [], "quality_all": [],
        },
        "frame_mean_gray": [],
        "n_decode_fail": 0,
    }

    for rec, target, nav_rec in sampled:
        jpg = blob[rec["offset"]:rec["offset"] + rec["length"]]
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            stats["n_decode_fail"] += 1
            continue
        stats["frame_mean_gray"].append(float(img.mean()))
        roll = float(nav_rec.get("roll_rad") or 0.0)
        pitch = float(nav_rec.get("pitch_rad") or 0.0)
        yaw_hat = float(nav_rec.get("yaw_rad") or 0.0)

        # ---- vp_yaw: exact gate replica of navigator._apply_vp_yaw --------------------------
        s = stats["vp"]
        s["n_tried"] += 1
        t0 = time.perf_counter()
        est = estimate_heading(img, roll, pitch, ransac_iters=VP_YAW_RANSAC_ITERS)
        s["ms"].append((time.perf_counter() - t0) * 1000.0)
        if est is None:
            s["rej_no_estimate"] += 1
        else:
            s["quality_all"].append(float(est.quality))
            if est.quality < VP_YAW_MIN_QUALITY:
                s["rej_low_quality"] += 1
            else:
                branches = np.asarray(est.branch_headings_rad, dtype=np.float64)
                diffs = np.array([_wrap_pi(b - yaw_hat) for b in branches])
                k = int(np.argmin(np.abs(diffs)))
                if abs(diffs[k]) > VP_YAW_BRANCH_MAX_RAD:
                    s["rej_branch"] += 1
                else:
                    yaw_meas = float(branches[k])
                    innov = abs(_wrap_pi(yaw_meas - yaw_hat))
                    s["n_would_apply"] += 1
                    s["innov_rad"].append(innov)

        # ---- floor_height: exact gate replica of navigator._apply_floor_height --------------
        f = stats["floor"]
        f["n_tried"] += 1
        t0 = time.perf_counter()
        fest = estimate_floor_height(img, roll, pitch, grid_cell_m=FLOOR_GRID_CELL_M)
        f["ms"].append((time.perf_counter() - t0) * 1000.0)
        if fest is None:
            f["rej_no_estimate"] += 1
        else:
            f["quality_all"].append(float(fest.quality))
            f["std_m_all"].append(float(fest.std_m))
            if fest.quality < FLOOR_MIN_QUALITY:
                f["rej_low_quality"] += 1
            elif fest.std_m > FLOOR_MAX_STD_M:
                f["rej_high_std"] += 1
            else:
                f["n_would_apply"] += 1
                f["height_m"].append(float(fest.height_m))

    return stats


def pctl(xs, p):
    if not xs:
        return None
    return float(np.percentile(np.asarray(xs, dtype=np.float64), p))


def summarize(stats):
    def pct(n, d):
        return (100.0 * n / d) if d else 0.0

    lines = []
    lines.append(f"### Run: {stats['run']}")
    lines.append("")
    lines.append(f"- video frames total: {stats['n_video_frames_total']}")
    lines.append(f"- frames in armed-flight window: {stats['n_frames_in_armed_window']}")
    lines.append(f"- frames sampled (replayed): {stats['n_frames_sampled']} "
                 f"(effective stride ~{stats['stride_effective']:.2f})" if stats['stride_effective']
                 else f"- frames sampled: {stats['n_frames_sampled']}")
    if stats["frame_mean_gray"]:
        mg = np.asarray(stats["frame_mean_gray"])
        lines.append(f"- frame mean-gray: median {np.median(mg):.2f}  p95 {np.percentile(mg,95):.2f}  "
                     f"max {mg.max():.2f}  (dark warehouse)")
    lines.append(f"- decode failures: {stats['n_decode_fail']}")
    lines.append("")

    v = stats["vp"]
    n = v["n_tried"]
    lines.append(f"**vp_yaw** (estimate_heading, {VP_YAW_RANSAC_ITERS} RANSAC iters):")
    lines.append(f"- n_tried={n}  n_would_apply={v['n_would_apply']}  "
                 f"acceptance={pct(v['n_would_apply'], n):.2f}%")
    lines.append(f"- rejections: no-estimate={v['rej_no_estimate']} ({pct(v['rej_no_estimate'],n):.1f}%)  "
                 f"low-quality(<{VP_YAW_MIN_QUALITY})={v['rej_low_quality']} ({pct(v['rej_low_quality'],n):.1f}%)  "
                 f"branch-reject(>35deg)={v['rej_branch']} ({pct(v['rej_branch'],n):.1f}%)")
    lines.append(f"- timing ms/call: mean {np.mean(v['ms']):.1f}  p50 {pctl(v['ms'],50):.1f}  "
                 f"p95 {pctl(v['ms'],95):.1f}  max {max(v['ms']):.1f}")
    if v["quality_all"]:
        q = np.asarray(v["quality_all"])
        lines.append(f"- quality of non-None estimates: n={len(q)}  median {np.median(q):.3f}  "
                     f"p95 {np.percentile(q,95):.3f}  max {q.max():.3f}  "
                     f"(gate needs >= {VP_YAW_MIN_QUALITY})")
    else:
        lines.append(f"- quality: (no non-None estimates)")
    if v["innov_rad"]:
        inn = np.rad2deg(np.asarray(v["innov_rad"]))
        lines.append(f"- ACCEPTED yaw innovation |deg|: n={len(inn)}  median {np.median(inn):.2f}  "
                     f"mean {inn.mean():.2f}  p95 {np.percentile(inn,95):.2f}  max {inn.max():.2f}")
    else:
        lines.append(f"- ACCEPTED yaw innovation: (none accepted)")
    lines.append("")

    f = stats["floor"]
    n = f["n_tried"]
    lines.append(f"**floor_height** (estimate_floor_height):")
    lines.append(f"- n_tried={n}  n_would_apply={f['n_would_apply']}  "
                 f"acceptance={pct(f['n_would_apply'], n):.2f}%")
    lines.append(f"- rejections: no-estimate={f['rej_no_estimate']} ({pct(f['rej_no_estimate'],n):.1f}%)  "
                 f"low-quality(<{FLOOR_MIN_QUALITY})={f['rej_low_quality']} ({pct(f['rej_low_quality'],n):.1f}%)  "
                 f"high-std(>{FLOOR_MAX_STD_M}m)={f['rej_high_std']} ({pct(f['rej_high_std'],n):.1f}%)")
    lines.append(f"- timing ms/call: mean {np.mean(f['ms']):.1f}  p50 {pctl(f['ms'],50):.1f}  "
                 f"p95 {pctl(f['ms'],95):.1f}  max {max(f['ms']):.1f}")
    if f["quality_all"]:
        q = np.asarray(f["quality_all"])
        lines.append(f"- quality of non-None estimates: n={len(q)}  median {np.median(q):.3f}  "
                     f"p95 {np.percentile(q,95):.3f}  max {q.max():.3f}  (gate needs >= {FLOOR_MIN_QUALITY})")
        s = np.asarray(f["std_m_all"])
        lines.append(f"- std_m of non-None estimates: median {np.median(s):.3f}  p05 {np.percentile(s,5):.3f}  "
                     f"min {s.min():.3f}m  (gate needs <= {FLOOR_MAX_STD_M}m)")
    else:
        lines.append(f"- quality/std: (no non-None estimates)")
    if f["height_m"]:
        h = np.asarray(f["height_m"])
        lines.append(f"- ACCEPTED heights m: n={len(h)}  median {np.median(h):.2f}  range [{h.min():.2f}, {h.max():.2f}]")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--stride", type=int, default=4, help="replay every Nth armed-window frame")
    ap.add_argument("--max-frames", type=int, default=1200, help="cap replayed frames per run")
    ap.add_argument("--out", default=None, help="write markdown results here")
    args = ap.parse_args()

    all_out = ["# VQ2 loop-choke backstop verification (offline replay)", "",
               f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} on this machine (CPU-only, uncontended).",
               f"Config: vp_yaw_min_quality={VP_YAW_MIN_QUALITY}, vp_yaw_branch_max={np.rad2deg(VP_YAW_BRANCH_MAX_RAD):.0f}deg, "
               f"ransac_iters={VP_YAW_RANSAC_ITERS}; floor_height_min_quality={FLOOR_MIN_QUALITY}, "
               f"floor_height_max_std_m={FLOOR_MAX_STD_M}, floor_grid_cell_m={FLOOR_GRID_CELL_M}.", ""]
    for r in args.runs:
        run = Path(r)
        if not run.is_absolute():
            run = REPO / run
        print(f"[replay] {run.name} ...", flush=True)
        t0 = time.time()
        stats = replay_run(run, args.stride, args.max_frames)
        summ = summarize(stats)
        print(summ)
        print(f"[replay] {run.name} done in {time.time()-t0:.1f}s")
        all_out.append(summ)

    text = "\n".join(all_out)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
