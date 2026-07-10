"""RL egocentric gate-observation emitter — the map-free vision->RL bridge.

Per frame, runs the deploy detector and emits the CLOSEST and SECOND-CLOSEST gate as
body-frame egocentric observations (no world pose, no map, no tracker) matching the RL
obs contract (rl-egocentric-obs-contract-2026-07-06):

    rel_pos_body  (3,)  gate centre in BODY FRD [forward, right, down], metres
    range_m             |rel_pos_body|
    visible_area_ratio  range-free foreshortening ~|cos(approach angle)|; None if <4 corners
    inner_area_px       raw apparent inner-opening area (px^2); None if <4 corners
    score               detector confidence
    n_corners           4 => IPPE, 3 => P3P (trust less)
    reproj_px           PnP reprojection error
    ambiguity_ratio     IPPE 2-fold err2/err1 -- LOW (~1) means FLIP RISK; None if n/a

THREE THINGS THAT ARE EASY TO GET WRONG (all measured, see the module notes):

1. BORESIGHT. The camera optical centre sits at body offset [0,0,vert_offset_m] from the
   body origin, so the gate-relative-to-BODY vector is
       p_gate_body = [0,0,vert_offset_m] + R_camera_from_body().T @ t_cam_gate
   OMITTING the offset injects a **+0.30 m vertical bias** (measured on the task2 given-pose
   bundle: z residual +0.303+-0.038 raw -> +0.053+-0.038 with the deployed -0.25 m bake).
   `frames.BORESIGHT` is read LIVE, so this tracks any recalibration.

2. RANKING. "Closest" is ranked by APPARENT SIZE (bbox area), NOT by PnP range. Prior-free
   PnP flips near-frontal close gates to ~25 m (the IPPE 2-fold), which would mis-rank them
   as farthest. Apparent size is a pure image measurement and is flip-immune.

3. FLIPS + PARTIALS are REAL on the map-free path (no map prior to tie-break the 2-fold).
   `ambiguity_ratio` near 1.0 and/or `n_corners == 3` mean "trust the rel_pos less" --
   `visible_area_ratio` is range-free and stays trustworthy where the PnP range does not.
   A 3-corner (cropped) gate has visible_area_ratio/inner_area_px = None -> RL masks on it.

Usage (library):
    from racer.contracts import Frame
    from racer.vision.detector import GateDetector
    from rl_gate_obs import emit_gate_obs
    det = GateDetector.load("models/vq2_partial_m_2026-07-06_fp16_384x640.engine")
    gates = emit_gate_obs(frame, det, k=2)     # [closest, second_closest]

Usage (demo over a recording):
    .venv/Scripts/python.exe scripts/rl_gate_obs.py --run-dir data/runs/<run> --max-frames 40
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from racer import frames as F                                # noqa: E402
from racer.contracts import Frame, GateObservation           # noqa: E402
from racer.vision.gate_pose import estimate_gate_pose        # noqa: E402


def _camera_centre_in_body() -> np.ndarray:
    """Camera optical centre expressed in body FRD. Reads frames.BORESIGHT LIVE (the ONE
    correction instance), so a recalibration or a runtime swap is picked up here."""
    return np.array([0.0, 0.0, float(F.BORESIGHT.vert_offset_m)])


def _apparent_size_px(o: GateObservation) -> float:
    """Flip-immune closeness proxy: YOLO bbox area (always present). Falls back to the
    inner-corner bbox when bbox_xywh is missing (synthesised observations)."""
    if o.bbox_xywh is not None:
        b = np.asarray(o.bbox_xywh, float)
        return float(abs(b[2] * b[3]))
    c = np.asarray(o.corners_px, float)
    wh = c.max(axis=0) - c.min(axis=0)
    return float(abs(wh[0] * wh[1]))


def gate_obs_from_observation(o: GateObservation) -> dict | None:
    """One detection -> the egocentric RL bundle (body FRD, boresight-corrected). None when
    the pose is unrecoverable (<3 usable corners / degenerate solve)."""
    pose = estimate_gate_pose(o)          # prior-free: the map-free deploy regime
    if pose is None:
        return None
    R_body_cam = F.R_camera_from_body().T
    rel = _camera_centre_in_body() + R_body_cam @ np.asarray(pose.t_cam_gate, float)
    return {
        "rel_pos_body": rel,                              # [fwd, right, down] m, body FRD
        "range_m": float(np.linalg.norm(rel)),
        "visible_area_ratio": o.visible_area_ratio,       # None if <4 corners
        "inner_area_px": o.inner_area_px,                 # None if <4 corners
        "score": float(o.score),
        "n_corners": int(pose.n_corners),
        "reproj_px": float(pose.reproj_error_px),
        "ambiguity_ratio": pose.ambiguity_ratio,          # ~1.0 => flip risk
        "apparent_size_px": _apparent_size_px(o),
    }


def emit_gate_obs(frame: Frame, detector, k: int = 2) -> list[dict]:
    """CLOSEST-first list of at most ``k`` egocentric gate observations for one frame.

    Ranked by APPARENT SIZE (flip-immune), not PnP range -- see module note 2. Detections
    whose pose is unrecoverable are dropped. Returns [] when nothing is usable; RL should
    treat a short list as "gate not observed this tick" and mask accordingly."""
    bundles = []
    for o in detector.detect(frame):
        b = gate_obs_from_observation(o)
        if b is not None:
            bundles.append(b)
    bundles.sort(key=lambda b: -b["apparent_size_px"])     # largest = closest
    return bundles[:k]


def _fmt(b: dict) -> str:
    r = b["rel_pos_body"]
    var = "  none" if b["visible_area_ratio"] is None else f"{b['visible_area_ratio']:.3f}"
    amb = " none" if b["ambiguity_ratio"] is None else f"{b['ambiguity_ratio']:.2f}"
    return (f"rel[fwd {r[0]:+6.2f} right {r[1]:+6.2f} down {r[2]:+6.2f}] "
            f"rng {b['range_m']:5.2f}m  var {var}  nc {b['n_corners']}  "
            f"reproj {b['reproj_px']:4.2f}px  amb {amb}  score {b['score']:.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="models/vq2_partial_m_2026-07-06_fp16_384x640.engine")
    ap.add_argument("--run-dir", required=True, help="recorded session dir (video.bin)")
    ap.add_argument("--max-frames", type=int, default=40)
    ap.add_argument("--k", type=int, default=2)
    args = ap.parse_args()

    from racer.recording import RecordingReader
    from racer.vision.detector import GateDetector

    print(f"boresight vert_offset_m = {F.BORESIGHT.vert_offset_m:+.3f} m "
          f"(camera centre in body FRD)")
    det = GateDetector.load(args.weights)
    n_any = 0
    for i, fr in enumerate(RecordingReader(args.run_dir).frames()):
        if i >= args.max_frames:
            break
        gates = emit_gate_obs(fr, det, k=args.k)
        if not gates:
            continue
        n_any += 1
        print(f"\nframe {i:>3}  ({len(gates)} gate(s))")
        for j, b in enumerate(gates):
            print(f"  [{'closest' if j == 0 else f'#{j+1}':>8}] {_fmt(b)}")
    print(f"\nframes with >=1 usable gate: {n_any}/{min(args.max_frames, i + 1)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
