"""LEAN re-run of the d3 verdict cells (reuses d3.run_cell + d3.reproduce_c1_rel VERBATIM).

The full d3_margin_closure.py sweep (~32k full-lap MC) is being CPU-starved by concurrent sessions.
This trim runs ONLY the cells the COLD-MARGIN verdict needs, at the EXACT d3 machinery (imported,
not re-implemented): the c1 anchor + the 37 m/s table warm/cold/weakvel x accel-bias x latency.
Writes JSON only. Run from repo root:
  PYTHONPATH=src .venv/Scripts/python.exe \
    handoff/ultracode-gate-relative-pipeline-design-2026-06-13/closure/d3_trim.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[3] / "src"))
sys.path.insert(0, str(_HERE.parents[3] / "handoff" / "ultracode-vision-case-c-2026-06-13"))
sys.path.insert(0, str(_HERE.parents[3] / "handoff" / "ultracode-estimator-racespeed-2026-06-13"))
sys.path.insert(0, str(_HERE.parents[1]))
import d3_margin_closure as d3  # noqa: E402

N = 300
out = {"n_mc": N, "margin_g4_m": d3.MARGIN_G4}

print("=" * 96)
print("ANCHOR (reproduce c1 rel WARM, single g3->g4 @37) -- expect RMS~0.139 / p90~0.203")
print("=" * 96)
anc = d3.reproduce_c1_rel(n_mc=600)
out["anchor"] = anc
print(f"  WARM anchor: RMS={anc['inplane_rms']:.3f} p50={anc['inplane_p50']:.3f} "
      f"p90={anc['inplane_p90']:.3f} p99={anc['inplane_p99']:.3f}  (c1: RMS 0.139, p90 0.203)")

print("\n" + "=" * 96)
print("37 m/s TABLE -- warm/cold/weakvel x accel-bias x latency  (d3 PROPER perp-projection in-plane)")
print("=" * 96)
print("%-8s %5s %5s | %7s %7s %7s | %9s | %6s %6s" %
      ("velmode", "bias", "lat", "ip_rms", "ip_p90", "ip_p99", "velErr_g4", "rmsOK", "p90OK"))
rows = []
for vm in ("warm", "cold", "weakvel"):
    for bias in (0.0, 0.24, 0.5, 1.0, 2.0):
        for lat in (15.0, 115.0):
            r = d3.run_cell(37.0, bias, vm, lat, n_mc=N)
            rows.append(r)
            print("%-8s %5.2f %5.0f | %7.3f %7.3f %7.3f | %9.3f | %6s %6s" % (
                vm, bias, lat, r["inplane_rms"], r["inplane_p90"], r["inplane_p99"],
                r["vel_err_entering_g4_mean"] or -1,
                "Y" if r["clears_margin_rms"] else "N",
                "Y" if r["clears_margin_p90"] else "N"))
out["table_37"] = rows
(_HERE.parent / "d3_trim_results.json").write_text(json.dumps(out, indent=2))
print(f"\nwrote {_HERE.parent / 'd3_trim_results.json'}")
