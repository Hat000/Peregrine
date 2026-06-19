"""LEVER 5: checkpoint ENSEMBLE (no retrain). Union the per-frame detections of 2+ models, then score
the standard good-fix + course metrics + a paired McNemar on course vs the champion.

Union is safe for THESE metrics: course valid-fix is any(obs<3m) and task2 good-fix picks the best obs
(min di), so adding a model's detections can only keep or improve both -- no NMS needed.

  python eval_ensemble.py <champion.pt> <wA++wB[++wC...]> [more specs ...]
  (first arg = champion baseline for the paired test; remaining args = '++'-joined ensemble specs)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "C:/Users/Shadow/Peregrine/cluster")
import json  # noqa: E402
import numpy as np  # noqa: E402
import eval_goodfix as E  # noqa: E402
import task2_gate_pnp as T  # noqa: E402
from wilson import wilson  # noqa: E402
from racer.navigator import load_track_map  # noqa: E402
from racer.vision.detector import GateDetector  # noqa: E402

_orig_load = GateDetector.load.__func__  # underlying function of the classmethod


class EnsembleDetector:
    def __init__(self, weights_list, **kw):
        self.dets = [_orig_load(GateDetector, w, **kw) for w in weights_list]

    def detect(self, frame):
        out = []
        for d in self.dets:
            out.extend(d.detect(frame))
        return out


def smart_load(weights, **kw):
    s = str(weights)
    if "++" in s:
        return EnsembleDetector(s.split("++"), **kw)
    return _orig_load(GateDetector, weights, **kw)


GateDetector.load = staticmethod(smart_load)  # now eval_goodfix + paired logic are ensemble-aware


def course_per_frame(weights):
    gates = {g.gate_id: g for g in load_track_map(E.MAP, corner_to_center=False)}
    gp = {gid: np.asarray(g.position_ned, float) for gid, g in gates.items()}
    d = json.loads((E.COURSE / "frames.json").read_text())
    det = GateDetector.load(weights, score_thresh=0.25, kpt_conf_thresh=0.5)
    band = [fr for fr in d["frames"] if 2 <= fr["range_m"] <= 30]
    out = []
    for fr in band:
        g = gates[int(fr["nearest_gate_id"])]
        obs = [o for o in det.detect(E._frame(fr, E.COURSE)) if o.corners_px.shape[0] == 4]
        ok = False
        if obs:
            R_wc = T.R_world_camera(fr["odo_q_wxyz"]); dr = np.asarray(fr["drone_position_ned"], float)
            Rg = np.asarray(g.R_world_gate, float)
            ok = any(min(np.linalg.norm(dr + R_wc @ T.solve_t_known_R(o.corners_px, R_wc.T @ Rg)[0] - p)
                         for p in gp.values()) < 3.0 for o in obs)
        out.append(ok)
    return np.array(out)


def mcnemar_p(b, c):
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, sum(comb(n, i) for i in range(k + 1)) / (2 ** n) * 2)


champ = sys.argv[1]
specs = sys.argv[2:]
print(f"{'spec':50}{'<0.5m':>14}{'<1m':>7}{'<2m':>7}{'course':>20}")
for w in [champ] + specs:
    e = E.task2_goodfix(w); N = len(e)
    n05 = int((e < 0.5).sum()); n1 = int((e < 1).sum()); n2 = int((e < 2).sum())
    v, M = E.course_validfix(w)
    _, lo, hi = wilson(n05, N); _, clo, chi = wilson(v, M)
    name = ("CHAMPION " if w == champ else "") + "+".join(Path(x).parent.parent.name for x in w.split("++"))
    print(f"{name[:50]:50}{f'{n05}/{N}[{100*lo:.0f},{100*hi:.0f}]':>14}"
          f"{f'{100*n1/N:.0f}%':>7}{f'{100*n2/N:.0f}%':>7}{f'{v}/{M}[{100*clo:.0f},{100*chi:.0f}]':>20}", flush=True)

print("\n=== PAIRED course (McNemar) vs champion ===")
va = course_per_frame(champ)
for w in specs:
    vb = course_per_frame(w)
    onlyA = int((va & ~vb).sum()); onlyB = int((~va & vb).sum())
    name = "+".join(Path(x).parent.parent.name for x in w.split("++"))
    print(f"{name[:46]:46} champ={va.sum()} ens={vb.sum()}  champ_wins={onlyA} ens_wins={onlyB}  "
          f"p={mcnemar_p(onlyA, onlyB):.4f}", flush=True)
