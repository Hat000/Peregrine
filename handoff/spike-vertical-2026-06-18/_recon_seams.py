"""Workflow-independent seam recon: (1) IPPE round-trip, (2) 8->4 slice shim, (3) real detector->PnP."""
import os, sys, glob, re
import numpy as np

# --- import path ---
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

from racer.contracts import Frame, GateObservation
from racer.vision.gate_pose import (
    estimate_gate_pose, project_gate_corners, gate_object_points, GATE_INNER_SIZE_M,
)
from racer.vision.detector import observations_from_keypoints, N_CORNERS
from racer.frames import CAMERA_INTRINSICS_K

np.set_printoptions(precision=5, suppress=True)
print("CAMERA_INTRINSICS_K =\n", CAMERA_INTRINSICS_K)
print("gate_object_points (IPPE order LL,LR,UR,UL) =\n", gate_object_points())

# =====================================================================
# (1) IPPE ROUND-TRIP: project a KNOWN pose -> 4 px (canonical order) -> solve -> recover.
# This empirically verifies the IPPE object-point order vs the projector. [step-1 order proof]
# =====================================================================
print("\n===== (1) IPPE ROUND-TRIP =====")
def roundtrip(R, t, label):
    px = project_gate_corners(R, t)                       # (4,2), canonical LL,LR,UR,UL
    obs = GateObservation(frame_id=1, sim_time_ns=1000,
                          corners_px=px, corner_ids=None,
                          corner_confidence=np.ones(4), score=0.95)
    gp = estimate_gate_pose(obs, compute_covariance=True, weighted_refine=True)
    if gp is None:
        print(f"  [{label}] PnP FAILED"); return None
    dt = np.linalg.norm(gp.t_cam_gate - t)
    dR = np.degrees(np.arccos(np.clip((np.trace(R.T @ gp.R_cam_gate) - 1)/2, -1, 1)))
    print(f"  [{label}] range_truth={np.linalg.norm(t):.3f} recovered={gp.range_m:.3f} "
          f"|dt|={dt:.2e}m dR={dR:.2e}deg reproj={gp.reproj_error_px:.3e}px t_z={gp.t_cam_gate[2]:.3f} "
          f"ambig={gp.ambiguity_ratio} ncorn={gp.n_corners} cov={'Y' if gp.covariance is not None else 'N'}")
    return gp, dt, dR

I = np.eye(3)
roundtrip(I, np.array([0.0, 0.0, 8.0]), "head-on 8m")
roundtrip(I, np.array([0.3, -0.2, 8.0]), "offset 8m")
# tilted: rotate gate about its Y (yaw-in-gate) by 15 deg
th = np.radians(15.0)
Ry = np.array([[np.cos(th),0,np.sin(th)],[0,1,0],[-np.sin(th),0,np.cos(th)]])
roundtrip(Ry, np.array([0.5, -0.3, 6.0]), "yaw15 6m")

# =====================================================================
# (2) 8->4 SLICE SHIM: synthetic (1,8,2) -> slice inner-4 -> identical obs to feeding inner-4. [blocker#3]
# =====================================================================
print("\n===== (2) 8->4 SLICE SHIM =====")
px4 = project_gate_corners(I, np.array([0.2, -0.1, 7.0]))         # canonical inner-4
outer = project_gate_corners(I, np.array([0.2, -0.1, 7.0]), inner_size_m=2.72)  # outer-4
xy8 = np.concatenate([px4, outer], axis=0)[None, ...]            # (1,8,2)
conf8 = np.ones((1, 8)); scores = np.array([0.9])
frame = Frame(frame_id=2, sim_time_ns=2000, image_bgr=np.zeros((360,640,3), np.uint8))

def slice_8kp_to_4(xy, conf):
    """SHIM (gap#1): YOLO-8kpt (n,8,2)+(n,8) -> inner-4 (n,4,2)+(n,4) for the PnP path."""
    xy = np.asarray(xy); conf = np.asarray(conf)
    assert xy.ndim == 3 and xy.shape[1] >= N_CORNERS, f"need (n,>=4,2), got {xy.shape}"
    return xy[:, :N_CORNERS, :], conf[:, :N_CORNERS]

xy4, conf4 = slice_8kp_to_4(xy8, conf8)
obs_sliced = observations_from_keypoints(frame, xy4, conf4, scores)
obs_direct = observations_from_keypoints(frame, px4[None, ...], np.ones((1,4)), scores)
ok = (len(obs_sliced) == 1 == len(obs_direct)
      and np.allclose(obs_sliced[0].corners_px, obs_direct[0].corners_px, atol=1e-12)
      and np.allclose(obs_sliced[0].corners_px, px4, atol=1e-12))
print(f"  sliced n={len(obs_sliced)} direct n={len(obs_direct)} IDENTICAL+canonical={ok}")
gp_sliced = estimate_gate_pose(obs_sliced[0], compute_covariance=True)
print(f"  sliced->PnP range={gp_sliced.range_m:.3f} (truth 7.003) reproj={gp_sliced.reproj_error_px:.2e}")

# =====================================================================
# (3) REAL DETECTOR -> PnP on real frames (4-kpt model; 8-kpt unavailable locally).
# =====================================================================
print("\n===== (3) REAL 4-KPT DETECTOR -> PnP (real frames) =====")
weights = "models/gate_yolo11s_curriculum_v3.pt"
frames = sorted(glob.glob(".claude/worktrees/*/handoff/shadowpc-followups-2026-06-05/task2_frames/*.png"))
def rng_of(p):
    m = re.search(r"_([\d.]+)m\.png$", p)
    return float(m.group(1)) if m else None
# pick frames near these ranges
targets = [2.6, 5.0, 10.0, 15.0, 20.0]
picks = []
for tr in targets:
    best = min(frames, key=lambda p: abs((rng_of(p) or 1e9) - tr))
    picks.append(best)
try:
    import cv2
    from racer.vision.detector import GateDetector
    det = GateDetector.load(weights)
    for p in picks:
        img = cv2.imread(p)
        fr = Frame(frame_id=3, sim_time_ns=3000, image_bgr=img)
        obss = det.detect(fr)
        truth = rng_of(p)
        if not obss:
            print(f"  {os.path.basename(p):28s} truth~{truth}m  NO DETECTION")
            continue
        obs = max(obss, key=lambda o: o.score)
        gp = estimate_gate_pose(obs, compute_covariance=True, weighted_refine=True)
        if gp is None:
            print(f"  {os.path.basename(p):28s} truth~{truth}m  PnP FAILED (n={obs.corners_px.shape[0]})")
            continue
        print(f"  {os.path.basename(p):28s} truth~{truth}m  PnP range={gp.range_m:.2f}m "
              f"n_corn={gp.n_corners} reproj={gp.reproj_error_px:.2f}px score={obs.score:.2f} "
              f"t_cam={gp.t_cam_gate}")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"  REAL DETECTOR PATH ERROR: {type(e).__name__}: {e}")
print("\nRECON DONE")
