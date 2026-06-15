"""Render margin_rerun_results.json -> margin_rerun_table.md (readable closure table) and print the
headline answers. [BORESIGHT-CLOSURE 2026-06-14]"""
from __future__ import annotations
import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
res = json.loads((_HERE / "margin_rerun_results.json").read_text())
meta = res["meta"]
MARGIN = meta["margin_at_r"]
RADII = ["0.21", "0.26", "0.30", "0.33", "0.38"]


def clo(p90, p99, r):
    m = MARGIN[r]
    return "Y" if (p90 < m and p99 < m) else "n"


lines = []
W = lines.append
W("# GATE-4 COLD case-C margin RE-RUN -- post-bake + real L3 anisotropic sigma\n")
W(f"_Engine: margin_driver_v2 (regression-proven == production ME.fly_lap in iso/zero-bias limit). "
  f"COLD case-C, latency 15 ms, seed {meta['seed']}._\n")
W(f"- sigma: iso {meta['sigma_iso']}, **aniso [lat {meta['sigma_aniso'][0]}, vert {meta['sigma_aniso'][1]}]** "
  f"(the honest gate-4 shape), conservative iso {meta['sigma_conservative']}, old-modeled {meta['sigma_modeled']}.")
W(f"- bake: POST = unbiased +L fix (deployed; eps removed); PRE = +{meta['pre_bake_bias_m']} m on e2 "
  f"(= the measured -0.25 m gate-DOWN epsilon_vert).")
W(f"- MARGIN(r) = W_EFF({meta['w_eff']}) - r:  " + "  ".join(f"r{r}->{MARGIN[r]:.3f}" for r in RADII))
W(f"- closure = (p90 < MARGIN(r)) AND (p99 < MARGIN(r)).  A/B/C nmc={meta['a_nmc']}, D nmc={meta['d_nmc']} "
  f"(+ bootstrap {meta['nboot']}x 90% CI; HONEST closure = upper CI of p99 < MARGIN).\n")

def closes_at(p90, p90hi, p99, p99hi, r):
    """CI-honest closure at radius r: upper CI of BOTH p90 and p99 below MARGIN(r)."""
    m = MARGIN[r]
    if p90hi < m and p99hi < m:
        return "CLOSE"
    if p99 - (p99hi - p99) < m <= p99hi:   # p99 lo-CI < m <= p99 hi-CI  -> straddle
        return "KNIFE_EDGE"
    return "NO_CLOSE"


# ---- per-r OPERATING-POINT table (from the high-nmc D grid; aniso sigma, post-bake) ----
W("## PER-RADIUS at the OPERATING POINT (post-bake, aniso sigma, high-nmc D grid + CI)\n")
W("Operating point = the measured terminal fix-rate fr=0.07 (bias 0) AND the closure-boundary "
  "fix-rates, read across ALL radii off the SAME miss distribution. CI-honest closure = upper-CI(p99) < MARGIN(r).\n")
ops = []
for c in res["cells_D"]:
    if c["bake"] == "post" and c["att_bias_deg"] == 0.0 and c["bias_mode"] == "random3d" and c["v"] in (30.0, 37.0):
        ops.append(c)
ops.sort(key=lambda c: (c["v"], c["fix_rate"]))
W("| v | fr | p90 | p99 | " + " | ".join(f"r{r} (M={MARGIN[r]:.3f})" for r in RADII) + " |")
W("|---|---|---|---|" + "---|" * len(RADII))
for c in ops:
    p90, p99 = c["p90"], c["p99"]
    p90hi, p99hi = c["p90_ci"][1], c["p99_ci"][1]
    cells = []
    for r in RADII:
        m = MARGIN[r]
        # point closure + CI-honest flag
        pt = (p90 < m and p99 < m)
        ci = (p90hi < m and p99hi < m)
        cells.append("CLOSE" if ci else ("knife" if pt else "no"))
    W(f"| {c['v']:.0f} | {c['fix_rate']:.2f} | {p90:.3f} | {p99:.3f} | " + " | ".join(cells) + " |")
W("")

# ---- D: decision cells (the verdict) ----
W("## D -- DECISION CELLS (high-nmc + bootstrap 90% CI; aniso sigma; CI-honest verdict)\n")
W("| v | bias_mode | fr | att | bake | p50 | p90 [CI] | p99 [CI] | r=0.30 | r=0.38 |")
W("|---|---|---|---|---|---|---|---|---|---|")
for c in res["cells_D"]:
    W(f"| {c['v']:.0f} | {c['bias_mode']} | {c['fix_rate']:.2f} | {c['att_bias_deg']:.1f} | {c['bake']} "
      f"| {c['p50']:.3f} | {c['p90']:.3f} [{c['p90_ci'][0]:.3f},{c['p90_ci'][1]:.3f}] "
      f"| {c['p99']:.3f} [{c['p99_ci'][0]:.3f},{c['p99_ci'][1]:.3f}] "
      f"| {c['verdict']['0.30']} | {c['verdict']['0.38']} |")
W("")

# ---- C: baseline anchor ----
W("## C -- BASELINE anchor (iso 0.265, post-bake, fr0.07, b0, v37, random3d)\n")
c = res["cells_C"][0]
W(f"p90 {c['p90']:.3f}  p99 {c['p99']:.3f}  (vs banked iso0.10 0.210/0.297 -- 0.265 is HIGHER, as expected; "
  f"sigma-recal sanity probe had 0.496/0.636 @ v30/inplane).\n")

# ---- B vs A: bake delta at key cells ----
W("## A vs B -- the BAKE DELTA (pre eps=+0.25 vs post eps=0), aniso sigma, p90/p99\n")
W("| v | bias_mode | fr | att | POST p90 | PRE p90 | d_p90 | POST p99 | PRE p99 | d_p99 |")
W("|---|---|---|---|---|---|---|---|---|---|")
A = {(c["v"], c["bias_mode"], c["fix_rate"], c["att_bias_deg"], c["sigma"]): c for c in res["cells_A"]}
for b in res["cells_B"]:
    key = (b["v"], b["bias_mode"], b["fix_rate"], b["att_bias_deg"], b["sigma"])
    a = A.get(key)
    if not a:
        continue
    W(f"| {b['v']:.0f} | {b['bias_mode']} | {b['fix_rate']:.2f} | {b['att_bias_deg']:.1f} "
      f"| {a['p90']:.3f} | {b['p90']:.3f} | {b['p90']-a['p90']:+.3f} "
      f"| {a['p99']:.3f} | {b['p99']:.3f} | {b['p99']-a['p99']:+.3f} |")
W("")

# ---- A: full post-bake sweep, per r p90/p99 + closes? ----
W("## A -- POST-bake sweep (per-r p90/p99 + closes?), all sigma x speed x fr x bias x bias_mode\n")
W("| v | sigma | fr | att | bm | p90 | p99 | " + " | ".join(f"r{r}?" for r in ["0.21","0.26","0.30","0.33","0.38"]) + " |")
W("|---|---|---|---|---|---|---|---|---|---|---|---|")
for c in sorted(res["cells_A"], key=lambda c: (c["v"], c["sigma"], c["bias_mode"], c["att_bias_deg"], c["fix_rate"])):
    cl = "".join  # noqa
    cells = " | ".join(clo(c["p90"], c["p99"], r) for r in ["0.21","0.26","0.30","0.33","0.38"])
    W(f"| {c['v']:.0f} | {c['sigma']} | {c['fix_rate']:.2f} | {c['att_bias_deg']:.1f} | {c['bias_mode'][:3]} "
      f"| {c['p90']:.3f} | {c['p99']:.3f} | {cells} |")
W("")

(_HERE / "margin_rerun_table.md").write_text("\n".join(lines))
print("wrote margin_rerun_table.md (", len(lines), "lines )")

# ---- headline answers (printed) ----
print("\n=== HEADLINE ===")
# min fix-rate that closes r=0.30 post-bake at measured bias (b0), aniso, per speed & bias_mode (from A, CI-honest via D where available)
for v in (30.0, 37.0):
    for bm in ("random3d", "inplane"):
        rows = sorted([c for c in res["cells_A"] if c["v"] == v and c["bias_mode"] == bm
                       and c["att_bias_deg"] == 0.0 and c["sigma"] == "aniso0.19/0.10"],
                      key=lambda c: c["fix_rate"])
        minfr = next((c["fix_rate"] for c in rows if c["clears_p90_and_p99"]["0.30"]), None)
        print(f"  v{v:.0f} {bm:8s} aniso b0: min fr closing r0.30 (point p90&p99) = {minfr}")
