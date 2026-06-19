"""Deployed-IPPE good-fix (the brief's "do not regress" secondary): same task2 geometry but range comes
from the DEPLOYED estimate_gate_pose (IPPE_SQUARE) instead of the oracle known-R solver. Relative
comparison (same method for both models) tells us whether the ensemble regresses deployed-IPPE.

  python deployed_ippe.py <champion.pt> <wA++wB...>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "C:/Users/Shadow/Peregrine/cluster")
import json  # noqa: E402
import numpy as np  # noqa: E402
import eval_goodfix as E  # noqa: E402
import task2_gate_pnp as T  # noqa: E402
import racer.frames as Fr  # noqa: E402
from racer.vision.detector import GateDetector  # noqa: E402
from racer.vision.gate_pose import estimate_gate_pose  # noqa: E402

_orig_load = GateDetector.load.__func__


class EnsembleDetector:
    def __init__(self, wl, **kw):
        self.dets = [_orig_load(GateDetector, w, **kw) for w in wl]

    def detect(self, fr):
        out = []
        for d in self.dets:
            out.extend(d.detect(fr))
        return out


def smart_load(w, **kw):
    s = str(w)
    return EnsembleDetector(s.split("++"), **kw) if "++" in s else _orig_load(GateDetector, w, **kw)


GateDetector.load = staticmethod(smart_load)
K = np.asarray(Fr.CAMERA_INTRINSICS_K, float)


def ippe_goodfix(weights):
    d = json.loads((T.BUNDLE / "frames.json").read_text())
    gmap = np.asarray(d["gate0_map_ned"], float)
    det = GateDetector.load(weights, score_thresh=0.25, kpt_conf_thresh=0.5)
    errs = []
    for fr in d["frames"]:
        obs = [o for o in det.detect(E._frame(fr, T.BUNDLE)) if o.corners_px.shape[0] == 4]
        gt = float(fr["range_m"]); e = np.inf
        if obs:
            R_wc = T.R_world_camera(fr["odo_q_wxyz"]); dr = np.asarray(fr["drone_position_ned"], float); best = 9e9
            for o in obs:
                pose = estimate_gate_pose(o, camera_matrix=K)
                if pose is None:
                    continue
                t = np.asarray(pose.t_cam_gate, float)
                di = float(np.linalg.norm(dr + R_wc @ t - gmap))
                if di < best:
                    best = di; e = abs(np.linalg.norm(t) - gt) if di < 3.0 else np.inf
        errs.append(e)
    return np.array(errs)


print(f"{'model (deployed IPPE PnP)':44}{'<0.5m':>14}{'<1m':>8}{'<2m':>8}")
for w in sys.argv[1:]:
    e = ippe_goodfix(w); N = len(e)
    print(f"{Path(w.split('++')[0]).name[:30] + (' (ens)' if '++' in w else ''):44}"
          f"{f'{int((e<0.5).sum())}/{N} ({100*(e<0.5).mean():.0f}%)':>14}"
          f"{f'{100*(e<1).mean():.0f}%':>8}{f'{100*(e<2).mean():.0f}%':>8}", flush=True)
