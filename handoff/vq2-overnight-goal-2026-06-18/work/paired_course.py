"""Paired (McNemar) comparison of course valid-fix between two models over the SAME 204 band frames.
Wilson CIs on two independent proportions are conservative for paired data; McNemar uses the discordant
pairs and is the correct test for "does model B beat model A on the same frames".

  python paired_course.py <weightsA(champion)> <weightsB>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "C:/Users/Shadow/Peregrine/cluster")
import json  # noqa: E402
import numpy as np  # noqa: E402
import eval_goodfix as E  # noqa: E402
import task2_gate_pnp as T  # noqa: E402
from racer.navigator import load_track_map  # noqa: E402
from racer.vision.detector import GateDetector  # noqa: E402


def per_frame_valid(weights):
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
    # two-sided exact binomial on discordant pairs (n=b+c, k=min(b,c), p=0.5)
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n) * 2
    return min(1.0, p)


A, B = sys.argv[1], sys.argv[2]
va = per_frame_valid(A); vb = per_frame_valid(B)
both = int((va & vb).sum()); onlyA = int((va & ~vb).sum()); onlyB = int((~va & vb).sum())
neither = int((~va & ~vb).sum())
print(f"A={Path(A).name}  valid={va.sum()}/{len(va)}")
print(f"B={Path(B).name}  valid={vb.sum()}/{len(vb)}")
print(f"both={both}  onlyA(champ wins)={onlyA}  onlyB(new wins)={onlyB}  neither={neither}")
print(f"net B-A = {vb.sum()-va.sum()};  McNemar two-sided p = {mcnemar_p(onlyA, onlyB):.4f}")
print("(p<0.05 with onlyB>onlyA => B significantly beats A on the paired course metric)")
