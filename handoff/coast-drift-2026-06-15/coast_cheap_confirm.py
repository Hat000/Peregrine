"""COAST-CHEAPNESS confirmation (robust, unbuffered) — the MC over-converges so this is centering-BLIND;
its sole valid claim is that the COAST DYNAMICS (velocity-drift + process + accel-bias over r_floor/v) are
cheap and robust across speed x floor x bias x per-fix sigma. p99 here is the over-converged floor (NOT the
realistic centering miss — that is final.py F2 via A1). Replaces the crashed sweep.py M-section with a
small robust grid. [COAST-DRIFT 2026-06-15]"""
from __future__ import annotations
import json, sys, time, traceback
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import coast_drift as CD  # noqa: E402
import margin_envelope as ME  # noqa: E402

OUT={"cells":[], "errors":[]}; T0=time.time()
print("COAST-CHEAPNESS grid (v x r_floor x accel-bias x sigma; tau=0.10; over-converged floor)")
print(f"  {'v':>4} {'r_fl':>5} {'bias':>5} {'sig':>5} {'t_coast':>7} {'p50':>6} {'p90':>6} {'p99':>6} "
      f"{'clr.30':>7} {'clr.38':>7}")
for v in (22.0, 30.0, 37.0):
    for r_floor in (10.0, 12.0, 14.0):
        for bias in (0.0, 0.3, 0.6):
            for sg in (0.10, 0.15):
                try:
                    bmag=ME.att_deg_to_accel_bias(bias); n=500; miss=np.empty(n)
                    for s in range(n):
                        seed=(CD.SEED+101*s+int(v)*13+int(bias*100)*7+int(sg*1000)*17
                              +int(r_floor*100)*131+7777)
                        miss[s]=CD.fly_lap_coast(np.random.default_rng(seed), v, sg, 0.10, bmag,
                                                 "inplane", 0.60, r_floor, fix_corr_tau=0.10)
                    p50=float(np.percentile(miss,50)); p90=float(np.percentile(miss,90)); p99=float(np.percentile(miss,99))
                    cell=dict(v=v,r_floor=r_floor,bias_deg=bias,sigma_lat=sg,t_coast=r_floor/v,
                              p50=p50,p90=p90,p99=p99,
                              clr30=bool(p99<CD.M(0.30)),clr38=bool(p99<CD.M(0.38)))
                    OUT["cells"].append(cell)
                    print(f"  {v:4.0f} {r_floor:5.1f} {bias:5.2f} {sg:5.2f} {r_floor/v:7.3f} "
                          f"{p50:6.3f} {p90:6.3f} {p99:6.3f} {str(cell['clr30']):>7} {str(cell['clr38']):>7}",
                          flush=True)
                except Exception as e:  # noqa: BLE001
                    OUT["errors"].append(dict(v=v,r_floor=r_floor,bias=bias,sg=sg,err=repr(e),tb=traceback.format_exc()))
                    print(f"  ERROR v={v} r={r_floor} b={bias} sg={sg}: {e!r}", flush=True)
p99s=[c["p99"] for c in OUT["cells"]]
allclr30=all(c["clr30"] for c in OUT["cells"]); allclr38=all(c["clr38"] for c in OUT["cells"])
OUT["meta"]=dict(seed=CD.SEED, n_cells=len(OUT["cells"]), n_err=len(OUT["errors"]),
                 p99_min=min(p99s), p99_max=max(p99s), all_clear_030=allclr30, all_clear_038=allclr38,
                 M_030=CD.M(0.30), M_038=CD.M(0.38), wall_s=round(time.time()-T0,1),
                 note="OVER-CONVERGED floor (centering-blind); confirms COAST DYNAMICS cheap+robust only")
(Path(__file__).resolve().parent/"coast_cheap_results.json").write_text(json.dumps(OUT,indent=1))
print(f"\nDONE {OUT['meta']['wall_s']}s  n={len(OUT['cells'])} err={len(OUT['errors'])}  "
      f"p99 range [{min(p99s):.3f},{max(p99s):.3f}]  all-clear r0.30={allclr30} r0.38={allclr38}")
