"""Size fix_range_rel_tol from the measured robust-mode data: what does each candidate
tolerance cost in good fixes and buy in leak, course-level (robust_k1.0 pooled)?"""
import json
from pathlib import Path

import numpy as np

OUT = Path("handoff/shadowpc-assoc-flipfix-2026-06-09")
CHI2 = 16.27

rows = []
for g in range(6):
    d = json.loads((OUT / f"char_g{g}_robust_k1.0.json").read_text())
    rows += [r for r in d["rows"] if r.get("associated") and "world_fix_err_m" in r]

# in VQ1 the prior anchor == given pos, so predicted range == true_range_m
good1 = [r for r in rows if r["world_fix_err_m"] < 1.0]
good3 = [r for r in rows if r["world_fix_err_m"] < 3.0]
cat = [r for r in rows if r["world_fix_err_m"] >= 3.0]

gr = np.array([abs(r["range_err_m"]) / r["true_range_m"] for r in good3])
print(f"good(<3m) depth-error/range: p50 {np.percentile(gr,50):.3f}  p90 {np.percentile(gr,90):.3f} "
      f" p99 {np.percentile(gr,99):.3f}  max {gr.max():.3f}   (N={len(gr)})")

print(f"\n{'rel_tol':>7} {'goodLost(<3m)':>13} {'goodLost(<1m)':>13} {'catDropped':>10} {'leak':>4}")
for rel in (0.25, 0.20, 0.15, 0.12, 0.10):
    def ok(r):
        return abs(r["range_err_m"]) <= max(1.0, rel * r["true_range_m"])
    gl3 = sum(not ok(r) for r in good3)
    gl1 = sum(not ok(r) for r in good1)
    cd = sum(not ok(r) for r in cat)
    leak = sum(ok(r) and np.isfinite(r["maha"]) and r["maha"] <= CHI2 for r in cat)
    print(f"{rel:7.2f} {gl3:13d} {gl1:13d} {cd:10d}/{len(cat)} {leak:4d}")
