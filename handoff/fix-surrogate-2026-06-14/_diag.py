import numpy as np
from pathlib import Path
D = Path("handoff/fix-surrogate-2026-06-14/data")
g = dict(np.load(D / "geom_frames.npz", allow_pickle=True))
HF, VF = 45.0, 29.36
az, el, tz, crab = g["azimuth_g"], g["elevation_g"], g["tz_g"], g["crab_deg"]
rng, inf = g["range_g"], g["in_fov_g"]
fin = np.isfinite(crab)
print("active-gate in_fov_g overall:", round(float(inf.mean()), 3))
for lab, m in [("range<10", rng < 10), ("10-26", (rng >= 10) & (rng < 26)),
               ("26-40", (rng >= 26) & (rng < 40)), (">40", rng >= 40)]:
    if m.sum():
        print(f"  {lab:>8} n={int(m.sum()):4d}  in_fov {inf[m].mean():.3f}  "
              f"|az| p50 {np.percentile(np.abs(az[m]),50):5.1f}  |el| p50 {np.percentile(np.abs(el[m]),50):5.1f}  "
              f"tz<0 {np.mean(tz[m]<0):.2f}")
eps = az + crab
band = (rng >= 10) & (rng < 30) & fin & (tz > 0)
print("\noffered-range[10,30] active frames:", int(band.sum()))
for C in [0, 10, 20, 30, 40, 50, 60, 66]:
    na = eps - C
    f = ((np.abs(na) < HF) & (np.abs(el) < VF))[band].mean()
    print(f"  crab {C:3d}: active_in_fov(offered-range) {f:.3f}")
print("\noffered-range |el| p50/p90:", np.round(np.percentile(np.abs(el[band]), [50, 90]), 1))
print("offered-range eps(az+crab) p10/p50/p90:", np.round(np.percentile(eps[band], [10, 50, 90]), 1))
# how often is elevation the sole blocker at crab=0 in offered band
na0 = eps - 0.0
azok = np.abs(na0) < HF
elok = np.abs(el) < VF
b = band
print("\noffered-range @crab0: azok", round(float(azok[b].mean()),3),
      " elok", round(float(elok[b].mean()),3),
      " both", round(float((azok&elok)[b].mean()),3))
