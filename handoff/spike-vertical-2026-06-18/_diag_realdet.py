"""Is the real-detector PnP miss an ORDER bug or LOCALIZATION (weak-model) error?"""
import os, sys, glob, re, itertools
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src")); os.chdir(ROOT)
import cv2
from racer.contracts import Frame, GateObservation
from racer.vision.detector import GateDetector, observations_from_results, _to_numpy
from racer.vision.gate_pose import estimate_gate_pose
np.set_printoptions(precision=2, suppress=True)

frames = sorted(glob.glob(".claude/worktrees/*/handoff/shadowpc-followups-2026-06-05/task2_frames/*.png"))
def rng_of(p):
    m = re.search(r"_([\d.]+)m\.png$", p); return float(m.group(1)) if m else None
det = GateDetector.load("models/gate_yolo11s_curriculum_v3.pt")

for tr in [2.6, 5.0]:
    p = min(frames, key=lambda q: abs((rng_of(q) or 1e9) - tr))
    img = cv2.imread(p)
    res = det.model.predict(img, verbose=False)[0]
    xy = _to_numpy(res.keypoints.xy)[0]      # (4,2)
    conf = _to_numpy(res.keypoints.conf)
    conf = None if conf is None else conf[0]
    print(f"\n{os.path.basename(p)} truth~{tr}m")
    print(f"  emitted kpts (canonical idx0=LL,1=LR,2=UR,3=UL):\n  {xy}")
    print(f"  kpt conf: {conf}")
    # image geometry: LL=large y(down) small x(left); LR=large y large x; UR=small y large x; UL=small y small x
    cx, cy = xy[:,0].mean(), xy[:,1].mean()
    quad = []
    for (x,y) in xy:
        h = "U" if y < cy else "L"           # image y grows DOWN
        v = "L" if x < cx else "R"
        quad.append(h+v)
    print(f"  per-emitted-kpt image quadrant: {quad}  (canonical wants ['LL','LR','UR','UL'])")
    # try ALL 24 permutations; report the lowest-reproj one
    best = None
    for perm in itertools.permutations(range(4)):
        obs = GateObservation(frame_id=1, sim_time_ns=1, corners_px=xy[list(perm)].copy(),
                              corner_ids=None, corner_confidence=np.ones(4), score=0.9)
        gp = estimate_gate_pose(obs, compute_covariance=False, weighted_refine=False)
        if gp is None: continue
        if best is None or gp.reproj_error_px < best[1]:
            best = (perm, gp.reproj_error_px, gp.range_m)
    print(f"  best-of-24-perms: perm={best[0]} reproj={best[1]:.2f}px range={best[2]:.2f}m "
          f"(identity perm=(0,1,2,3); low reproj at a NON-identity perm => order bug; high reproj at ALL => localization error)")
