"""A/B the GATE-EMISSION port on the real flight path (2026-07-22).

Answers ONE question: with the deployed engine and the deployed filters, does the port make the
drone see more gates, and are the extra ones any good?

The path replicated here is exactly ``GateSeeker._valid_poses``: detector -> estimate_gate_pose ->
the reproj / range / in-front quality gates. No map, no KF -- a pose that survives here is a pose
the policy would actually steer on.

Truth: the bundle ships the drone's GIVEN pose per frame plus ``gate0_map_ned``, whose meaning
TASK2_PNP_VERDICT.md settled -- it is the gate's BOTTOM EDGE (true centre ~1.36 m above, z-only),
and the bottom-CORNER hypothesis is REFUTED, so no horizontal shift belongs here.

CALIBRATION WARNING -- read before quoting the absolute numbers. This harness scores what the FLIGHT
emits, i.e. IPPE via ``estimate_gate_pose`` with NO rotation prior, and IPPE's planar 2-fold
ambiguity leaves a systematic ~1.2 m in the E axis on this near-frontal bundle. scripts/task2_gate_pnp.py
reports 0.5-0.7 m instead because it sidesteps the ambiguity entirely (translation-only PnP with the
rotation FIXED from the known gate orientation) -- a reference measurement, not the deploy path. So:
COMPARE THE ARMS, do not read the absolute error as detector accuracy. Note also that task2 is 40
mostly-FULL gates at range -- the wrong population for a cropped-gate change; the failure-mined
inbox is where the port actually pays.

Usage:  python scripts/task2_emission_ab.py [--weights <spec>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from racer import frames as F
from racer.contracts import Frame
from racer.vision.detector import GateDetector
from racer.vision.gate_pose import estimate_gate_pose

ROOT = Path(__file__).resolve().parent.parent
# models/ and handoff/ bundles are gitignored, so they exist only in the MAIN worktree -- a linked
# worktree checks out the tracked files and nothing else. Resolve against whichever tree has them.
_ASSET_ROOTS = [ROOT, Path("C:/Users/Shadow/Peregrine")]


def _asset(rel: str) -> Path:
    for r in _ASSET_ROOTS:
        if (r / rel).exists():
            return r / rel
    return ROOT / rel


BUNDLE = _asset("handoff/shadowpc-followups-2026-06-05/task2_frames")
DEFAULT_WEIGHTS = _asset("models/vq2_partial_m_2026-07-06_fp16_384x640.engine")

# The anchor is read from the bundle (``gate0_map_ned``). TASK2_PNP_VERDICT.md settled what it means:
# map z is the gate's BOTTOM EDGE, true centre ~1.36 m above (D is DOWN, so subtract); and the
# bottom-LEFT-CORNER hypothesis is REFUTED -- there is NO half-width horizontal shift.
GATE_CENTRE_LIFT_M = 1.36

# _valid_poses' deploy gates (gate_seeker.GateSeekerConfig; fly_rl sets max_valid_range_m=30 on ego)
MAX_REPROJ_PX = 12.0
MAX_VALID_RANGE_M = 30.0
GOOD_FIX_M = 0.30                  # the standing "good fix" bar used across the detector benchmarks

_RCB = F.R_camera_from_body()
_GATE0_CENTRE = None


def _gate0_centre():
    assert _GATE0_CENTRE is not None, "gate0 centre not initialised (main() reads it from the bundle)"
    return _GATE0_CENTRE


def R_world_camera(q_wxyz) -> np.ndarray:
    """World<-camera from the ODOMETRY quat. Uses the LIBRARY's own euler/rotation pair, exactly as
    the validated scripts/task2_gate_pnp.py does -- a hand-rolled quat->R here silently disagreed
    with the project convention and put a systematic 1.2 m into the E axis."""
    roll, pitch, yaw = F.euler_from_quat_wxyz(q_wxyz)
    return F.R_world_from_body(roll, pitch, yaw) @ _RCB.T


def _centre_world(pose, meta):
    """Gate centre in world NED from a camera-frame pose + the frame's GIVEN drone pose."""
    return (np.asarray(meta["drone_position_ned"], float)
            + R_world_camera(meta["odo_q_wxyz"]) @ pose.t_cam_gate)


# ASSOCIATION. Several gates are in view at once, so scoring every emitted pose against gate 0 is
# meaningless (it was: 57 poses / 40 frames, median 1.9 m, 0 good fixes -- mostly gate 1 and beyond).
# Associate on the frame's GIVEN range to gate 0, which is INDEPENDENT of the centre error being
# measured -- unlike "take the closest centre", which would hand any arm that emits more candidates
# a free pick and flatter exactly the change under test.
_RANGE_REL_TOL = 0.25
_RANGE_ABS_TOL_M = 1.0


def _is_gate0(pose_range_m: float, gt_range_m: float) -> bool:
    return abs(pose_range_m - gt_range_m) <= _RANGE_ABS_TOL_M + _RANGE_REL_TOL * gt_range_m


def run(detector, frames):
    """Every pose that survives the deploy quality gates, with its centre error."""
    rows = []
    for meta, frame in frames:
        for obs in detector.detect(frame):
            pose = estimate_gate_pose(obs, compute_covariance=False)
            if pose is None or not np.isfinite(pose.t_cam_gate).all():
                continue
            if float(pose.reproj_error_px) > MAX_REPROJ_PX:
                continue
            if float(pose.range_m) > MAX_VALID_RANGE_M:
                continue
            if pose.t_cam_gate[2] <= 0.05:
                continue
            rows.append(dict(
                png=meta["png"], gt_range=float(meta["range_m"]), range_m=float(pose.range_m),
                n_corners=int(pose.n_corners), derived=bool(getattr(obs, "derived_corners", False)),
                err_m=float(np.linalg.norm(_centre_world(pose, meta) - _gate0_centre())),
            ))
    return rows


def gate0_rows(rows):
    """One row per frame: the emitted pose whose RANGE matches the frame's true range to gate 0."""
    best = {}
    for r in rows:
        if not _is_gate0(r["range_m"], r["gt_range"]):
            continue
        prev = best.get(r["png"])
        if prev is None or abs(r["range_m"] - r["gt_range"]) < abs(prev["range_m"] - prev["gt_range"]):
            best[r["png"]] = r
    return best


def summarise(tag, rows, n_frames):
    if not rows:
        print(f"{tag:>10s}: NO emissions")
        return
    g = gate0_rows(rows)
    e = np.array([r["err_m"] for r in g.values()]) if g else np.array([])
    good = int((e <= GOOD_FIX_M).sum()) if e.size else 0
    stats = (f"err med {np.median(e):.3f}  p90 {np.percentile(e, 90):.3f}  max {e.max():.3f}"
             if e.size else "no gate-0 association")
    print(f"{tag:>10s}: {len(rows):3d} poses total | gate0 on {len(g):2d}/{n_frames} frames | "
          f"{stats} | good-fix(<={GOOD_FIX_M}m) {good}/{n_frames}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    args = ap.parse_args()

    global _GATE0_CENTRE
    meta_all = json.loads((BUNDLE / "frames.json").read_text())
    anchor = np.asarray(meta_all["gate0_map_ned"], float)
    _GATE0_CENTRE = anchor + np.array([0.0, 0.0, -GATE_CENTRE_LIFT_M])   # bottom edge -> centre
    print(f"gate0 anchor {anchor} -> centre {_GATE0_CENTRE}")
    metas = meta_all if isinstance(meta_all, list) else meta_all.get("frames", [meta_all])
    if not isinstance(metas, list):
        metas = [metas]

    import cv2
    frames = []
    for i, m in enumerate(metas):
        # per-FILE resolution: the bundle dir is tracked (frames.json) but the PNGs are not, so a
        # linked worktree has the directory and none of the images.
        p = _asset(f"handoff/shadowpc-followups-2026-06-05/task2_frames/{m['png']}")
        if not p.exists():
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        frames.append((m, Frame(frame_id=i, sim_time_ns=int(m["sim_time_ns"]), image_bgr=img)))
    print(f"loaded {len(frames)} frames from {BUNDLE.name}\nweights: {args.weights}\n")

    arms = [
        ("PRE-PORT",  dict(use_outer=False, partial_rescue=False, kpt_conf_thresh=0.5)),
        ("+outer",    dict(use_outer=True,  partial_rescue=False, kpt_conf_thresh=0.5)),
        ("+conf0.2",  dict(use_outer=True,  partial_rescue=False, kpt_conf_thresh=0.2)),
        ("PORTED",    dict(use_outer=True,  partial_rescue=True,  kpt_conf_thresh=0.2)),
    ]
    results = {}
    for tag, kw in arms:
        det = GateDetector.load(args.weights, **kw)
        rows = run(det, frames)
        results[tag] = rows
        summarise(tag, rows, len(frames))

    base, port = results["PRE-PORT"], results["PORTED"]
    print(f"\nemissions {len(base)} -> {len(port)}  ({len(port) - len(base):+d})")
    resc = [r for r in port if r["derived"]]
    if resc:
        e = np.array([r["err_m"] for r in resc])
        print(f"rescued   {len(resc)} poses | err med {np.median(e):.3f}  "
              f"range {min(r['gt_range'] for r in resc):.1f}-{max(r['gt_range'] for r in resc):.1f} m")
    g_base, g_port = gate0_rows(base), gate0_rows(port)
    new_frames = sorted(set(g_port) - set(g_base))
    print(f"frames that went from BLIND to SEEING gate 0: {len(new_frames)}")
    for p in new_frames:
        r = g_port[p]
        print(f"   {p:28s} err {r['err_m']:.3f} m  (gt range {r['gt_range']:.1f} m, "
              f"derived={r['derived']})")
    lost = sorted(set(g_base) - set(g_port))
    if lost:
        print(f"REGRESSION -- gate 0 LOST on {len(lost)} frames: {lost}")


if __name__ == "__main__":
    main()
