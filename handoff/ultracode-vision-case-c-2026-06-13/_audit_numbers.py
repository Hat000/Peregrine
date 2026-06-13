import json, numpy as np, glob
base = "handoff/perception-char-2026-06-08"

# --- Load all per-gate characterize rows ---
rows = []
for g in range(6):
    d = json.load(open(f"{base}/characterize_g{g}.json"))
    for r in d["rows"]:
        if r.get("associated") and r.get("n_corners"):
            rows.append(r)
print(f"total associated rows: {len(rows)}")

# --- depth error (range_err_m) vs true_range: the DEPTH/RADIAL error law ---
tr = np.array([r["true_range_m"] for r in rows])
re = np.array([r["range_err_m"] for r in rows])     # pose_range - true_range (signed, along ray = RADIAL)
tce = np.array([r["t_cam_err_m"] for r in rows])    # full t_cam error magnitude
nc = np.array([r["n_corners"] for r in rows])
# lateral (perp to ray) component magnitude = sqrt(t_cam_err^2 - range_err^2)
lat = np.sqrt(np.clip(tce**2 - re**2, 0, None))

# bin by true range
print("\n=== DEPTH(radial) vs LATERAL error by true-range bin (accepted-ish: |range_err|<5) ===")
mask_ok = np.abs(re) < 5.0   # exclude the catastrophic wrong-scale tail to see the in-family law
for lo,hi in [(0,5),(5,10),(10,15),(15,20),(20,25),(25,40)]:
    m = (tr>=lo)&(tr<hi)&mask_ok
    if m.sum()<2: 
        print(f"  [{lo:2d},{hi:2d}) n={m.sum():3d}  (sparse)"); continue
    print(f"  [{lo:2d},{hi:2d}) n={m.sum():3d}  radial(range_err) mean={re[m].mean():+.3f} std={re[m].std():.3f}  "
          f"|radial|med={np.median(np.abs(re[m])):.3f}  lateral med={np.median(lat[m]):.3f}  "
          f"radial/lateral={np.median(np.abs(re[m]))/max(np.median(lat[m]),1e-3):.2f}")

# radial as a fraction of range (the "3% of range" lore check), on in-family fixes
m = mask_ok & (tr>3)
frac = np.abs(re[m])/tr[m]
print(f"\nin-family |radial|/range: median={np.median(frac)*100:.2f}%  p90={np.percentile(frac,90)*100:.2f}%  (lore ~3%)")

# anisotropy ratio overall (in-family)
print(f"in-family radial std={re[mask_ok].std():.3f}  lateral median={np.median(lat[mask_ok]):.3f}")

# --- catastrophic / wrong-scale tail fraction across ALL ranges ---
print(f"\ncatastrophic |range_err|>=3 m fraction (all associated): {np.mean(np.abs(re)>=3.0)*100:.1f}%")
print(f"  among true_range>30 m: {np.mean((np.abs(re)>=3.0)&(tr>30))/max(np.mean(tr>30),1e-9)*100 if (tr>30).any() else 0:.1f}% of >30m rows are bad")
print(f"  n rows with true_range>30: {(tr>30).sum()};  >32 (cap): {(tr>32).sum()}")

# --- per-axis NED off (world fix error) to confirm canonical [0.73,0.47,0.29] ---
off = np.array([r["off_ned"] for r in rows])
m = np.abs(re)<3.0  # in-family
print(f"\nin-family world-fix err per-axis (N,E,D): mean={off[m].mean(0).round(3)}  std={off[m].std(0).round(3)}")
print("  (canonical VISION-PKG2 sigma [0.73,0.47,0.29], bias [-0.42,+0.06,-0.28])")
