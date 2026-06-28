"""scripts/vq2_loadday/detector_smoke.py — VQ2 load-day detector smoke-test (C6).

Runs the deployed gate detector against a held-out synthetic set and compares detection
rate + reprojection error to a VQ1 baseline stored in ``data/vq1_detector_baseline.json``
(or a built-in reference if the file is absent).  Flags an appearance-gap / photorealism
risk if the gap exceeds threshold.

Two modes:
  (a) Synthetic-only (offline, no live sim): render N synthetic frames at each curriculum
      level, run the detector, compare to the VQ1 baseline.  Always available.
  (b) Live-frame mode (``--live-frames N``): grab N JPEG frames from the live video stream,
      run detection on them, report raw detection rate.  Appearance-gap vs synthetic is the
      C6 signal.  Requires live sim + video port.

The detector weights must be on disk (gitignored; shipped as GitHub release assets).
If the weights file is absent the probe exits with a clear "needs weights" message
rather than crashing — the escape hatch for offline CI.

Checks addressed:
  C6a  Synthetic benchmark: detection_rate / median corner error vs VQ1 baseline.
  C6b  Appearance gap: live-frame detect rate vs synthetic rate (photorealism check).

Usage (synthetic only — offline smoke-test, no sim, no weights needed if not provided):
  python scripts/vq2_loadday/detector_smoke.py [--weights models/best.pt] [--n 50]

Usage (live frames — appearance-gap check):
  python scripts/vq2_loadday/detector_smoke.py --weights models/best.pt --live-frames 30

Usage (save JSON):
  python scripts/vq2_loadday/detector_smoke.py --weights models/best.pt --out data/c6_smoke.json

ESCAPE HATCH: If weights are absent the synthetic benchmark is SKIPPED with exit code 2
and a clear note.  Add --weights to run it, or point to the release-artifact path.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from racer.contracts import Frame
from racer.vision.detector import GateDetector, observations_from_keypoints
from racer.vision.gate_pose import estimate_gate_pose
from racer.vision.synthetic import V_OFF, render_gate_sample

# VQ1 baseline (stored offline; these values are from the eval_detector.py run on
# gate_yolo11s_curriculum_v2.pt, 200 samples / level, level 2, well-framed).
# Used when data/vq1_detector_baseline.json is not found.
_BUILTIN_VQ1_BASELINE = {
    "label": "builtin_reference_vq1",
    "level_2_detect_rate": 0.85,       # fraction of frames with >= 1 detection
    "level_2_median_corner_px": 5.0,   # px
    "level_2_p90_corner_px": 15.0,
    "level_3_detect_rate": 0.75,
    "level_3_median_corner_px": 8.0,
}

# Alert thresholds: if our numbers fall below these relative to the VQ1 baseline,
# flag it as an appearance-gap risk.
_DETECT_RATE_DROP_ALERT = 0.15   # > 15 pp drop from baseline → alert
_CORNER_ERR_SCALE_ALERT = 2.0    # > 2× baseline corner error → alert

# Default weights path (relative to repo root)
_DEFAULT_WEIGHTS = ROOT / "models" / "gate_yolo11s_curriculum_v2.pt"

# Baseline file path
_BASELINE_PATH = ROOT / "data" / "vq1_detector_baseline.json"


def _load_baseline() -> dict:
    """Load VQ1 baseline from disk, or fall back to the builtin reference."""
    if _BASELINE_PATH.exists():
        try:
            with open(_BASELINE_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return dict(_BUILTIN_VQ1_BASELINE)


def _pct(arr: np.ndarray, q: float) -> float:
    return float(np.percentile(arr, q)) if len(arr) > 0 else float("nan")


def run_synthetic_benchmark(
    detector: GateDetector,
    n_per_level: int = 100,
    levels: tuple[int, ...] = (2, 3),
    rng_seed: int = 42,
) -> dict:
    """Run the synthetic benchmark (offline, no sim needed).

    Returns per-level detection stats.  Pure except for detector.detect() calls.
    """
    rng = np.random.default_rng(rng_seed)
    per_level: dict[int, dict] = {}

    for level in levels:
        errs: list[float] = []
        pose_errs: list[float] = []
        n_detected = 0
        made = 0
        while made < n_per_level:
            sample = render_gate_sample(rng, level=level)
            if not sample.visible:
                continue
            made += 1
            obs_list = detector.detect(
                Frame(frame_id=made, sim_time_ns=0, image_bgr=sample.image_bgr.copy())
            )
            if not obs_list:
                errs.append(float("nan"))
                continue
            n_detected += 1
            obs = max(obs_list, key=lambda o: o.score)
            ids = obs.corner_ids if obs.corner_ids is not None else np.arange(4)
            ids = np.asarray(ids).astype(int)
            # error only over corners that are truly in-frame in the GT
            keep = [j for j, cid in enumerate(ids) if sample.visibility[cid] != V_OFF]
            if keep:
                gt = sample.keypoints_px[ids[keep]]
                err = float(np.mean(np.linalg.norm(obs.corners_px[keep] - gt, axis=1)))
                errs.append(err)
            else:
                errs.append(float("nan"))
            gp = estimate_gate_pose(obs)
            if gp is not None:
                pose_errs.append(float(np.linalg.norm(gp.t_cam_gate - sample.t_cam_gate)))

        errs_arr = np.array(errs)
        ok = ~np.isnan(errs_arr)
        per_level[level] = {
            "n_made": made,
            "n_detected": n_detected,
            "detect_rate": round(n_detected / made if made > 0 else 0.0, 4),
            "median_corner_px": round(_pct(errs_arr[ok], 50), 2),
            "p90_corner_px": round(_pct(errs_arr[ok], 90), 2),
            "max_corner_px": round(_pct(errs_arr[ok], 100), 2),
            "median_pose_err_m": round(_pct(np.array(pose_errs), 50), 4) if pose_errs else None,
        }

    return per_level


def compare_to_baseline(synthetic_results: dict[int, dict], baseline: dict) -> dict:
    """Compare synthetic benchmark results to the VQ1 baseline and flag regressions."""
    alerts: list[str] = []
    deltas: dict = {}

    for level in (2, 3):
        if level not in synthetic_results:
            continue
        res = synthetic_results[level]
        b_rate = baseline.get(f"level_{level}_detect_rate")
        b_med = baseline.get(f"level_{level}_median_corner_px")

        if b_rate is not None:
            drop = b_rate - res["detect_rate"]
            deltas[f"level_{level}_detect_rate_delta"] = round(drop, 4)
            if drop > _DETECT_RATE_DROP_ALERT:
                alerts.append(
                    f"L{level} detect_rate dropped {drop*100:.1f} pp vs baseline "
                    f"({res['detect_rate']*100:.0f}% vs {b_rate*100:.0f}%)"
                )

        if b_med is not None and res["median_corner_px"] is not None and not np.isnan(res["median_corner_px"]):
            scale = res["median_corner_px"] / b_med if b_med > 0 else float("inf")
            deltas[f"level_{level}_corner_err_scale"] = round(scale, 3)
            if scale > _CORNER_ERR_SCALE_ALERT:
                alerts.append(
                    f"L{level} corner error {scale:.1f}× baseline "
                    f"({res['median_corner_px']:.1f} px vs {b_med:.1f} px)"
                )

    return {
        "alerts": alerts,
        "c6a_pass": len(alerts) == 0,
        "deltas": deltas,
    }


def run_live_frame_check(
    detector: GateDetector,
    n_frames: int = 30,
    endpoint: str = "udp:127.0.0.1:14550",
) -> dict:
    """Grab N live JPEG frames and run detection on them.

    Returns detection rate + basic per-frame stats.  Requires live sim + video port.
    """
    try:
        from racer.vision.jpeg_receiver import JpegUdpReceiver
    except ImportError as e:
        return {
            "error": f"JpegUdpReceiver not available: {e}",
            "c6b_appearance_gap": None,
        }

    receiver = JpegUdpReceiver()
    try:
        receiver.start()
    except Exception as e:
        return {
            "error": f"video stream not reachable: {e} — start the sim first",
            "c6b_appearance_gap": None,
        }

    collected_frames: list[Frame] = []
    seen_ids: set[int] = set()
    deadline = time.monotonic() + 30.0  # wait up to 30 s for N frames
    while len(collected_frames) < n_frames and time.monotonic() < deadline:
        f = receiver.latest_frame()
        if f is not None and f.frame_id not in seen_ids:
            seen_ids.add(f.frame_id)
            collected_frames.append(f)
        time.sleep(0.01)

    try:
        receiver.stop()
    except Exception:
        pass

    if not collected_frames:
        return {
            "error": "no video frames received in 30 s — video stream absent or sim not running",
            "c6b_appearance_gap": None,
        }

    n_det = 0
    scores: list[float] = []
    for frame in collected_frames:
        obs_list = detector.detect(frame)
        if obs_list:
            n_det += 1
            best = max(obs_list, key=lambda o: o.score)
            scores.append(float(best.score))

    live_rate = n_det / len(collected_frames)

    return {
        "n_frames_collected": len(collected_frames),
        "n_frames_detected": n_det,
        "live_detect_rate": round(live_rate, 4),
        "mean_score": round(float(np.mean(scores)), 4) if scores else None,
        "c6b_appearance_gap": None,  # filled by compare vs synthetic below
    }


def _print_result(res: dict) -> None:
    print(f"\n{'='*65}")
    print(f"  DETECTOR SMOKE-TEST (C6) — detection rate + reprojection error")
    print(f"{'='*65}")

    if "error" in res:
        print(f"  ERROR: {res['error']}")
        return

    bl = res.get("baseline", {})
    print(f"\n  Baseline: {bl.get('label', '?')}")

    syn = res.get("synthetic", {})
    if "error" in syn:
        print(f"\n  Synthetic benchmark: SKIPPED — {syn['error']}")
    elif syn:
        print(f"\n  Synthetic benchmark:")
        for level in (2, 3):
            lv = syn.get(level, {})
            if not lv:
                continue
            print(f"    L{level}  detect={lv['detect_rate']*100:.0f}%  "
                  f"corner_med={lv['median_corner_px']:.1f}px  "
                  f"p90={lv['p90_corner_px']:.1f}px  "
                  f"pose_med={lv.get('median_pose_err_m', '?')}m")

    cmp = res.get("comparison", {})
    if cmp:
        print(f"\n  Comparison vs VQ1 baseline:")
        for k, v in cmp.get("deltas", {}).items():
            print(f"    {k}: {v:+.3f}")
        alerts = cmp.get("alerts", [])
        if alerts:
            print(f"\n  ALERTS ({len(alerts)}):")
            for a in alerts:
                print(f"    ! {a}")
        print(f"\n  C6a PASS: {cmp.get('c6a_pass')}")

    live = res.get("live_frames", {})
    if live and "error" not in live:
        print(f"\n  Live-frame check:")
        print(f"    n_collected: {live['n_frames_collected']}")
        print(f"    live_detect_rate: {live['live_detect_rate']*100:.0f}%  "
              f"mean_score={live.get('mean_score', '?')}")
        gap = live.get("c6b_appearance_gap")
        if gap is not None:
            print(f"    C6b appearance_gap_pp: {gap:+.1f}  "
                  f"({'ALERT' if abs(gap) > _DETECT_RATE_DROP_ALERT * 100 else 'OK'})")
    elif live:
        print(f"\n  Live-frame check: {live.get('error', 'unavailable')}")

    print(f"\n  NOTE: detector weights required; absent → synthetic skipped, live-frame skipped.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--weights", default=None,
        help=f"Path to .pt weights file (default: {_DEFAULT_WEIGHTS.relative_to(ROOT)})"
    )
    ap.add_argument("--n", type=int, default=100, help="Synthetic samples per level (default: 100)")
    ap.add_argument("--live-frames", type=int, default=0,
                    help="If > 0, also grab N live video frames from the sim (C6b)")
    ap.add_argument("--endpoint", default="udp:127.0.0.1:14550")
    ap.add_argument("--out", default=None, help="Write result JSON here")
    args = ap.parse_args()

    weights_path = Path(args.weights) if args.weights else _DEFAULT_WEIGHTS
    baseline = _load_baseline()

    result: dict = {"baseline": baseline}

    # Weights availability check (ESCAPE HATCH)
    if not weights_path.exists():
        result["error"] = (
            f"Weights file not found at {weights_path}. "
            f"Download from the burn-artifacts GitHub release (model/*.pt) and re-run. "
            f"Offline CI: skip by not providing --weights."
        )
        result["synthetic"] = {"error": "weights absent"}
        result["live_frames"] = {"error": "weights absent"}
        _print_result(result)
        if args.out:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        return 2

    print(f"  Loading detector weights: {weights_path}", flush=True)
    try:
        detector = GateDetector.load(weights_path)
    except Exception as e:
        result["error"] = f"Failed to load detector: {e}"
        _print_result(result)
        return 2

    # C6a: synthetic benchmark
    print(f"  Running synthetic benchmark ({args.n} samples × levels 2,3)...", flush=True)
    syn_results = run_synthetic_benchmark(detector, n_per_level=args.n, levels=(2, 3))
    result["synthetic"] = syn_results
    result["comparison"] = compare_to_baseline(syn_results, baseline)

    # C6b: live frames (optional)
    if args.live_frames > 0:
        print(f"  Grabbing {args.live_frames} live frames from {args.endpoint}...", flush=True)
        live_res = run_live_frame_check(detector, n_frames=args.live_frames, endpoint=args.endpoint)
        # Compute appearance gap vs synthetic level-2
        if "error" not in live_res and 2 in syn_results:
            syn_rate = syn_results[2]["detect_rate"]
            live_rate = live_res["live_detect_rate"]
            # Negative = live is worse than synthetic (appearance gap)
            live_res["c6b_appearance_gap"] = round((live_rate - syn_rate) * 100, 1)
        result["live_frames"] = live_res

    _print_result(result)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(f"\n  Result written to {out_path}")

    c6a_pass = result.get("comparison", {}).get("c6a_pass", True)
    return 0 if c6a_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
