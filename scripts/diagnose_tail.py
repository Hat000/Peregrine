"""Confirm what the corner-error tail is: identity SWAPS vs plain imprecision.

For every detected corner we compare its assigned-GT distance against the distance to the
NEAREST in-frame GT corner. If the prediction is closest to a *different* corner (and the
assigned error is large), that corner's identity was swapped -- the classic failure on a
near-rotationally-symmetric square under roll. We report the per-level swap rate, whether
swaps correlate with roll, and dump the worst frames (with index-labelled overlays) to look at.

Run:  .venv\\Scripts\\python.exe scripts\\diagnose_tail.py [weights] [n_per_level]
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from racer.contracts import Frame                                   # noqa: E402
from racer.vision.detector import GateDetector                     # noqa: E402
from racer.vision.synthetic import V_OFF, render_gate_sample       # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "models" / "gate_yolo11s_curriculum_v2.pt"  # v2 = chosen deployment detector
N = int(sys.argv[2]) if len(sys.argv) > 2 else 150
OUT = ROOT / "demo"
SWAP_PX = 15.0   # an assigned error this large, landing nearest a different corner, = a swap


def main() -> int:
    det = GateDetector.load(WEIGHTS)
    rng = np.random.default_rng(7)
    OUT.mkdir(exist_ok=True)
    records = []   # (maxerr, swap, roll_deg, level, sample, ids, pred, percorner)

    for level in (1, 2, 3):
        made = 0
        while made < N:
            s = render_gate_sample(rng, level=level)
            if not s.visible:
                continue
            made += 1
            obs = det.detect(Frame(frame_id=made, sim_time_ns=0, image_bgr=s.image_bgr.copy()))
            if not obs:
                continue
            o = max(obs, key=lambda z: z.score)
            ids = np.asarray(o.corner_ids if o.corner_ids is not None else np.arange(4)).astype(int)
            gt, vis = s.keypoints_px, s.visibility
            roll_deg = abs(np.degrees(Rotation.from_matrix(s.R_cam_gate).as_euler("xyz")[2]))
            percorner, swap, maxerr = {}, False, 0.0
            for p, c in enumerate(ids):
                if vis[c] == V_OFF:
                    continue
                d_assigned = float(np.linalg.norm(o.corners_px[p] - gt[c]))
                dists = [np.linalg.norm(o.corners_px[p] - gt[j]) if vis[j] != V_OFF else np.inf
                         for j in range(4)]
                nearest = int(np.argmin(dists))
                percorner[c] = d_assigned
                maxerr = max(maxerr, d_assigned)
                if nearest != c and d_assigned > SWAP_PX:
                    swap = True
            records.append((maxerr, swap, roll_deg, level, s, ids, o.corners_px, percorner))

    # ---- swap rate + roll correlation ----
    print(f"weights: {WEIGHTS.name}   samples: {len(records)}")
    print(f"\n{'lvl':>3} {'detected':>8} {'swaps':>6} {'swap%':>6} {'roll|swap':>10} {'roll|ok':>9}")
    for level in (1, 2, 3):
        lr = [r for r in records if r[3] == level]
        sw = [r for r in lr if r[1]]
        roll_sw = np.mean([r[2] for r in sw]) if sw else float("nan")
        roll_ok = np.mean([r[2] for r in lr if not r[1]]) if len(lr) > len(sw) else float("nan")
        print(f"{level:>3} {len(lr):>8} {len(sw):>6} {100*len(sw)/max(1,len(lr)):>5.1f}% "
              f"{roll_sw:>9.0f}d {roll_ok:>8.0f}d")

    # ---- worst 12: is it one corner blown out (swap) or all corners off (bad detect)? ----
    records.sort(key=lambda r: -r[0])
    print(f"\nworst 12 frames   (per-corner px by canonical id; SWAP=landed on another corner)")
    print(f"{'maxerr':>7} {'roll':>5} {'lvl':>3} {'ncorner':>7}  per-corner-err          swap")
    for maxerr, swap, roll, level, s, ids, pred, pc in records[:12]:
        cstr = " ".join(f"c{c}:{pc[c]:.0f}" for c in sorted(pc))
        print(f"{maxerr:>7.0f} {roll:>4.0f}d {level:>3} {len(ids):>7}  {cstr:<24} {'SWAP' if swap else ''}")

    # ---- dump worst 6 as labelled overlays ----
    for rank, (maxerr, swap, roll, level, s, ids, pred, pc) in enumerate(records[:6]):
        img = s.image_bgr.copy()
        for j, (x, y) in enumerate(s.keypoints_px):
            p = (int(round(x)), int(round(y)))
            cv2.circle(img, p, 5, (0, 200, 0), -1)
            cv2.putText(img, str(j), (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
        for p_i, c in enumerate(ids):
            p = (int(round(pred[p_i][0])), int(round(pred[p_i][1])))
            cv2.circle(img, p, 4, (0, 0, 230), -1)
            cv2.putText(img, str(c), (p[0] + 6, p[1] + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 230), 1)
        cv2.putText(img, f"maxerr={maxerr:.0f}px roll={roll:.0f}deg L{level} {'SWAP' if swap else ''}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.imwrite(str(OUT / f"tail_{rank}.png"), img)
    print(f"\nworst-6 overlays -> {OUT}\\tail_0..5.png  (green=GT w/ index, red=pred w/ canonical id)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
