"""Probe the terminal mean-offset source + actual coast-start sigmas. [COAST-DRIFT 2026-06-15]"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import coast_drift as CD  # noqa: E402
import margin_envelope as ME  # noqa: E402

# 1) Fully DETERMINISTIC lap (perfect init, no noise, no bias) -> is the offset deterministic?
rng = np.random.default_rng(1)
m_det, d_det = CD.fly_lap_coast(rng, 30.0, 0.10, 0.10, 0.0, "inplane", 0.60, 12.0,
                                noise_off=True, diag=True)
print("DETERMINISTIC lap (perfect init, zero noise/bias):")
print(f"  terminal miss={m_det:.4f}  lat={d_det['miss_lat']:+.4f} vert={d_det['miss_vert']:+.4f}")
if "perr_lat" in d_det:
    print(f"  coast-start(r={d_det['range_m']:.1f}): perr_lat={d_det['perr_lat']:+.4f} "
          f"verr_lat={d_det['verr_lat']:+.4f} perr_vert={d_det['perr_vert']:+.4f} "
          f"verr_vert={d_det['verr_vert']:+.4f}")

# 2) Deterministic lap with NO coast (fix all the way to gate, old 12 m->0 window) to isolate the coast.
#    Use r_floor tiny so the band reaches ~0; this is the "no-coast" control.
m_nocoast, d_nocoast = CD.fly_lap_coast(rng, 30.0, 0.10, 0.10, 0.0, "inplane", 0.60, 0.5,
                                        r_acc_max=24.0, noise_off=True, diag=True)
print(f"\nNO-COAST control (fix to ~0.5 m): terminal miss={m_nocoast:.4f} "
      f"lat={d_nocoast['miss_lat']:+.4f} vert={d_nocoast['miss_vert']:+.4f}")

# 3) latency 0 vs 15 ms (deterministic)
for lat in (0.0, 15.0):
    m, d = CD.fly_lap_coast(rng, 30.0, 0.10, 0.10, 0.0, "inplane", 0.60, 12.0,
                            latency_ms=lat, noise_off=True, diag=True)
    print(f"latency={lat:4.0f}ms  det miss={m:.4f} lat={d['miss_lat']:+.4f}")

# 4) MC: actual coast-start sigmas vs covariance (now that capture is fixed)
print("\nMC actual coast-start error (n=600):")
for fr, sg in [(0.60, 0.10), (0.20, 0.10), (0.60, 0.15)]:
    pl=[]; vl=[]; pv=[]; vv=[]; rr=[]
    for s in range(600):
        rng = np.random.default_rng(CD.SEED + 7*s + int(fr*1000) + int(sg*1000)*3)
        m, d = CD.fly_lap_coast(rng, 30.0, sg, 0.10, 0.0, "inplane", fr, 12.0, diag=True)
        if "perr_lat" in d:
            pl.append(d["perr_lat"]); vl.append(d["verr_lat"]); pv.append(d["perr_vert"]); vv.append(d["verr_vert"]); rr.append(d["range_m"])
    fc = CD.a2_achieved_sigma_v(30.0, sg, 0.10, fr, 12.0)
    print(f"  fr={fr} sg={sg}: r={np.mean(rr):.1f}m  ACTUAL sig_p_lat={np.std(pl):.4f} "
          f"sig_v_lat={np.std(vl):.4f} (mean verr={np.mean(vl):+.4f}) | COV sig_p={fc['sig_p_lat']:.4f} "
          f"sig_v={fc['sig_v_lat']:.4f}  | ACTUAL sig_v_vert={np.std(vv):.4f}")
