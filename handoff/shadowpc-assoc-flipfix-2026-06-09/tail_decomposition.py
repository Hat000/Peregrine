"""Scratch: decompose the catastrophic tail from the k1.0 per-frame characterize dumps."""
import json
from pathlib import Path

import numpy as np

D = Path("handoff/shadowpc-cov-finishhold-verify-2026-06-09/cov_sweep")
rows = []
for g in range(6):
    d = json.loads((D / f"char_g{g}_k1.0.json").read_text())
    for r in d["rows"]:
        if r.get("associated") and "world_fix_err_m" in r:
            r["bundle_gate"] = g
            rows.append(r)

print(f"total solved fixes: {len(rows)}")
cat = [r for r in rows if r["world_fix_err_m"] >= 3.0]
print(f"catastrophic (|fix|>=3m): {len(cat)} = {100*len(cat)/len(rows):.0f}%")

wrong = [r for r in cat if r["gate_id"] != r["bundle_gate"]]
flip = [r for r in cat if r["gate_id"] == r["bundle_gate"]]
print(f"  wrong-gate (assoc != bundle gate): {len(wrong)}")
print(f"  correct-gate (flips etc):          {len(flip)}")

print("\n--- WRONG-GATE breakdown ---")
print(f"{'bund':>4} {'asc':>3} {'trueRng':>7} {'poseRng':>7} {'rngErr':>7} {'bear':>5} "
      f"{'nc':>3} {'reproj':>7} {'|fix|':>6} {'maha':>9} {'score':>5}")
for r in sorted(wrong, key=lambda x: (x["bundle_gate"], x["range_m"])):
    print(f"{r['bundle_gate']:>4} {r['gate_id']:>3} {r['true_range_m']:7.1f} {r['pose_range_m']:7.1f} "
          f"{r['range_err_m']:+7.1f} {r['bearing_deg']:5.1f} {r['n_corners']:3d} {r['reproj_px']:7.2f} "
          f"{r['world_fix_err_m']:6.1f} {r['maha']:9.1f} {r['score']:5.2f}")

print("\n--- CORRECT-GATE tail breakdown ---")
for r in sorted(flip, key=lambda x: (x["bundle_gate"], x["range_m"])):
    print(f"{r['bundle_gate']:>4} {r['gate_id']:>3} {r['true_range_m']:7.1f} {r['pose_range_m']:7.1f} "
          f"{r['range_err_m']:+7.1f} {r['bearing_deg']:5.1f} {r['n_corners']:3d} {r['reproj_px']:7.2f} "
          f"{r['world_fix_err_m']:6.1f} {r['maha']:9.1f} {r['score']:5.2f}")

# Summary stats on the correct-gate tail: is the depth doubled / halved / reflected?
rr = np.array([r["pose_range_m"] / max(r["true_range_m"], 1e-6) for r in flip])
print("\ncorrect-gate tail pose_range/true_range ratio percentiles:",
      np.percentile(rr, [5, 25, 50, 75, 95]).round(2))
nc = [r["n_corners"] for r in flip]
print("correct-gate tail n_corners==3:", sum(1 for x in nc if x == 3), "/", len(nc))
rp = np.array([r["reproj_px"] for r in flip])
print("correct-gate tail reproj px p50/p90:", np.percentile(rp, [50, 90]).round(2))
br = np.array([r["bearing_deg"] for r in flip])
print("correct-gate tail bearing p50/p90:", np.percentile(br, [50, 90]).round(1))

good = [r for r in rows if r["world_fix_err_m"] < 1.0]
rpg = np.array([r["reproj_px"] for r in good])
print("\nGOOD fixes reproj px p50/p90:", np.percentile(rpg, [50, 90]).round(2))
ncg = [r["n_corners"] for r in good]
print("GOOD n_corners==3:", sum(1 for x in ncg if x == 3), "/", len(ncg))
brg = np.array([r["bearing_deg"] for r in good])
print("GOOD bearing p50/p90:", np.percentile(brg, [50, 90]).round(1))
