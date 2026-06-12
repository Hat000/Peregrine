"""Does the S18 lapse curve survive the attitude correction? Recompute the
K_eff/K(collective) ratio per speed band, S18 method (smooth ticks |w|<1,
d=2 collective delay, measured-drag attribution), under BOTH attitude readings."""
import json, sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from racer.rl_plant import (COLL_MAP_THR_MEASURED, COLL_MAP_ACCEL_MEASURED,
                            QUAD_DRAG_C2_MEASURED)

G = 9.80665
DATA = ROOT / "handoff/shadowpc-refit-dataset-2026-06-12/extracted"
BANDS = [(3, 6), (6, 9), (9, 12), (12, 15), (15, 18)]


def ratios(conj):
    out = {b: [] for b in BANDS}
    for run in sorted(p.name for p in DATA.iterdir() if p.is_dir()):
        rows = [json.loads(l) for l in open(DATA / run / "debug_obs.jsonl", encoding="utf-8")]
        rows = [r for r in rows if r.get("type") != "header"]
        t = np.array([r["t_mono"] for r in rows])
        v = np.array([r["vel_ned"] for r in rows])
        q = np.array([r["q_raw_wxyz"] for r in rows]) * np.array(conj)[None, :]
        w = np.array([r["w_raw"] for r in rows])
        c = np.array([r["collective"] for r in rows])
        R = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
        for k in range(2, len(t) - 2):
            dt = t[k + 1] - t[k - 1]
            if not (0.05 < dt < 0.09) or np.linalg.norm(w[k]) > 1.0:
                continue
            a = (v[k + 1] - v[k - 1]) / dt
            sp = np.linalg.norm(v[k])
            K = np.interp(c[max(k - 2, 0)], COLL_MAP_THR_MEASURED, COLL_MAP_ACCEL_MEASURED)
            if K < 3.0:
                continue
            vb = R[k].T @ v[k]
            cc = np.where(vb >= 0, QUAD_DRAG_C2_MEASURED[:, 0], QUAD_DRAG_C2_MEASURED[:, 1])
            drag = R[k] @ (-cc * np.abs(vb) * vb)
            b3 = R[k] @ np.array([0.0, 0.0, -1.0])
            keff = (a - np.array([0.0, 0.0, G]) - drag) @ b3
            for lo, hi in BANDS:
                if lo <= sp < hi:
                    out[(lo, hi)].append(keff / K)
    return out


for nm, conj in [("RAW (S18)", [1, 1, 1, 1]), ("TRUE (audit)", [1, -1, 1, -1])]:
    r = ratios(conj)
    cells = "  ".join(f"{lo}-{hi}: {np.median(r[(lo, hi)]):.3f} (n={len(r[(lo, hi)])})"
                      for lo, hi in BANDS)
    print(f"{nm:13s} {cells}")
