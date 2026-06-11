import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
rows = []
for gi in range(6):
    d = json.loads((HERE / f"fixedframe_g{gi}.json").read_text())
    rows += [r for r in d["rows"] if r.get("associated") and "world_fix_err_m" in r
             and r.get("range_ok", True)]
print("offered catastrophic fixes:")
for r in rows:
    if r["world_fix_err_m"] >= 3.0:
        print(f"  gate {r['gate_id']}  range {r['true_range_m']:.1f} m  |fix| {r['world_fix_err_m']:.1f}"
              f"  rngerr {r['range_err_m']:+.1f}  maha {r['maha']:.1f}")
gd = [r for r in rows if r["world_fix_err_m"] < 1.0]
print("offered good<1m: total", len(gd),
      " >26m:", sum(r["true_range_m"] > 26 for r in gd),
      " >30m:", sum(r["true_range_m"] > 30 for r in gd))
g13 = [r for r in rows if 1.0 <= r["world_fix_err_m"] < 3.0]
print("offered 1-3m:    total", len(g13),
      " >26m:", sum(r["true_range_m"] > 26 for r in g13),
      " >30m:", sum(r["true_range_m"] > 30 for r in g13))
