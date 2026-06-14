"""boresight_bearing_resolve.py — OFFLINE δ_map-vs-boresight discriminator (P3, 2026-06-14).

Independent, non-self-fulfilling test of the L3 at-speed gate-4/gate-2 vertical bias, run PURELY over
the existing shadow_gate{2,4}_rows.json (no sim, no detector re-run, no src/ edits).

Question: is the COMMON-MODE vertical bias (g2 -0.211 == g4 -0.215 m) a track_map offset delta_map
(constant METRES, cancels in the +L gate-relative obs -> margin CLOSES) or a camera-boresight / perception
pitch bias eps_vert (constant ANGLE delta_theta, does NOT cancel -> margin OPEN)?

DISCRIMINATOR = rel_vert vs true_range (per fix):
  * MAP  (case a): rel_vert ~ const  -> slope d(rel_vert)/d(range) ~ 0 ; intercept ~ -0.21 m
  * BORE (case b): rel_vert ~ -R*tan(delta_theta) -> slope > 0 (more negative vert at larger range);
                   intercept ~ 0 ; angle theta_i = atan2(rel_vert, range) ~ const across range.
The L3 report only compared POOLED MEANS at two similar median ranges (20 vs 22 m) -> no range leverage.
This uses the WITHIN-gate range spread (and the g2+g4 pool) for real leverage. Bootstrap CIs (stdlib only).

NOTE: rel_vert == abs_vert in every row (the gate-relative win is covariance, not a different z) -> one
vertical-error number per fix. Sign convention: +vert = gate-DOWN (NED). Measured ~ -0.215 = fix places the
drone gate-UP of truth == estimator places the gate gate-DOWN of truth.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics as st
from pathlib import Path

random.seed(1234)


def load_accepted(path: Path, clean_only: bool) -> list[dict]:
    rows = json.loads(path.read_text())["rows"]
    acc = [r for r in rows if r.get("accepted")]
    if clean_only:
        acc = [r for r in acc if r.get("n_corners") == 4 and r.get("reproj_px", 9) < 1.0]
    return acc


def ols(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Return (slope, intercept, pearson_r)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    syy = sum((y - my) ** 2 for y in ys)
    slope = sxy / sxx if sxx else float("nan")
    intercept = my - slope * mx
    r = sxy / math.sqrt(sxx * syy) if sxx and syy else float("nan")
    return slope, intercept, r


def boot_slope_intercept(xs, ys, nboot=4000):
    n = len(xs)
    sl, ic = [], []
    idx = list(range(n))
    for _ in range(nboot):
        s = [random.choice(idx) for _ in range(n)]
        bx = [xs[i] for i in s]
        by = [ys[i] for i in s]
        a, b, _ = ols(bx, by)
        sl.append(a)
        ic.append(b)
    sl.sort()
    ic.sort()
    lo, hi = int(0.025 * nboot), int(0.975 * nboot)
    return (sl[lo], sl[hi]), (ic[lo], ic[hi])


def cov(vals: list[float]) -> float:
    m = st.mean(vals)
    return (st.pstdev(vals) / abs(m)) if m else float("nan")


def analyze(label: str, rows: list[dict]) -> dict:
    rng = [r["true_range_m"] for r in rows]
    vert = [r["rel_vert"] for r in rows]
    cross = [r["rel_cross"] for r in rows]
    # metric vs angular description
    angle = [math.degrees(math.atan2(v, R)) for v, R in zip(vert, rng)]  # signed apparent pitch error
    slope, intercept, r = ols(rng, vert)
    (slo_lo, slo_hi), (ic_lo, ic_hi) = boot_slope_intercept(rng, vert)
    # implied boresight angle if case (b): rel_vert = -R*tan(theta) -> theta = atan(-slope)
    theta_from_slope = math.degrees(math.atan(-slope))
    print(f"\n===== {label}  (N={len(rows)}) =====")
    print(f"  range  : min {min(rng):.1f}  p50 {st.median(rng):.1f}  max {max(rng):.1f}  spread {max(rng)-min(rng):.1f} m")
    print(f"  rel_vert (gate-down, m): mean {st.mean(vert):+.3f}  sd {st.pstdev(vert):.3f}  CoV {cov(vert):.2f}")
    print(f"  rel_cross (control, m) : mean {st.mean(cross):+.3f}  sd {st.pstdev(cross):.3f}")
    print(f"  apparent pitch theta=atan2(vert,range) (deg): mean {st.mean(angle):+.3f}  sd {st.pstdev(angle):.3f}  CoV {cov(angle):.2f}")
    print(f"  --- DISCRIMINATOR: rel_vert = a + b*range ---")
    print(f"    slope b   = {slope:+.5f} m/m   95%CI [{slo_lo:+.5f}, {slo_hi:+.5f}]  (pearson r={r:+.2f})")
    print(f"    intercept = {intercept:+.3f} m       95%CI [{ic_lo:+.3f}, {ic_hi:+.3f}]")
    print(f"    => if BORESIGHT: implied delta_theta = atan(-b) = {theta_from_slope:+.3f} deg")
    # decide which model the slope CI supports
    map_ok = slo_lo <= 0.0 <= slo_hi            # slope CI includes 0 -> flat -> map-consistent
    # boresight prediction: slope ~ -tan(0.56deg) = -0.0098 m/m; check CI overlap
    bore_pred = -math.tan(math.radians(0.56))
    bore_ok = slo_lo <= bore_pred <= slo_hi
    print(f"    slope CI includes 0 (MAP-consistent, flat)?         {map_ok}")
    print(f"    slope CI includes -tan(0.56deg)={bore_pred:+.4f} (BORESIGHT)? {bore_ok}")
    # range-binned means
    print(f"  --- range-binned (mean rel_vert | mean apparent-pitch deg | N) ---")
    bins = [(0, 19), (19, 21), (21, 23), (23, 25), (25, 99)]
    for lo, hi in bins:
        b = [(v, a) for v, a, R in zip(vert, angle, rng) if lo <= R < hi]
        if not b:
            continue
        vs = [x[0] for x in b]
        as_ = [x[1] for x in b]
        print(f"    [{lo:>2}-{hi:>2} m] N={len(b):>3}  rel_vert {st.mean(vs):+.3f}  pitch {st.mean(as_):+.3f} deg")
    return dict(label=label, n=len(rows), slope=slope, slope_ci=[slo_lo, slo_hi],
                intercept=intercept, intercept_ci=[ic_lo, ic_hi], pearson_r=r,
                theta_from_slope_deg=theta_from_slope, map_consistent=map_ok, boresight_consistent=bore_ok,
                mean_vert=st.mean(vert), sd_vert=st.pstdev(vert), mean_cross=st.mean(cross),
                rng_min=min(rng), rng_p50=st.median(rng), rng_max=max(rng))


def bearing_coverage(label: str, all_rows_path: Path) -> None:
    rows = json.loads(all_rows_path.read_text())["rows"]
    acc = [r for r in rows if r.get("accepted")]
    print(f"\n  --- {label} ACCEPTED bearing coverage (boresight needs head-on; do we have it?) ---")
    for lo, hi in [(0, 15), (15, 30), (30, 45), (45, 60), (60, 99)]:
        b = [r for r in acc if lo <= r.get("g4_bearing_deg", 999) < hi]
        if not b:
            print(f"    bearing [{lo:>2}-{hi:>2}deg]: N=0")
            continue
        vs = [r["rel_vert"] for r in b]
        print(f"    bearing [{lo:>2}-{hi:>2}deg]: N={len(b):>3}  rel_vert mean {st.mean(vs):+.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--l3dir", default=str(Path(__file__).resolve().parents[4] /
                    "blissful-kalam-75f52b/handoff/l3-atspeed-recording-2026-06-14/analysis"))
    ap.add_argument("--clean", action="store_true", help="restrict to n_corners==4 & reproj<1px")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    d = Path(args.l3dir)
    g2p, g4p = d / "shadow_gate2_rows.json", d / "shadow_gate4_rows.json"
    print(f"L3 rows dir: {d}\nclean-only (4-corner, reproj<1px): {args.clean}")

    g2 = load_accepted(g2p, args.clean)
    g4 = load_accepted(g4p, args.clean)
    res = {}
    res["gate2"] = analyze("GATE-2", g2)
    res["gate4"] = analyze("GATE-4", g4)
    res["pool"] = analyze("GATE-2+GATE-4 POOLED (max range leverage)", g2 + g4)

    bearing_coverage("GATE-4", g4p)
    bearing_coverage("GATE-2", g2p)

    # cross-gate angular vs metric common-mode adjudication
    print("\n========== ADJUDICATION ==========")
    a2, a4 = res["gate2"], res["gate4"]
    print(f"  cross-gate METRIC : g2 {a2['mean_vert']:+.3f} m @ {a2['rng_p50']:.1f} m  vs  "
          f"g4 {a4['mean_vert']:+.3f} m @ {a4['rng_p50']:.1f} m")
    # boresight would predict g4/g2 metric ratio == range ratio
    ratio_pred = a4["rng_p50"] / a2["rng_p50"]
    ratio_obs = a4["mean_vert"] / a2["mean_vert"] if a2["mean_vert"] else float("nan")
    print(f"    boresight predicts metric ratio == range ratio {ratio_pred:.3f}; observed ratio {ratio_obs:.3f}")
    print(f"  POOLED slope b = {res['pool']['slope']:+.5f} m/m  CI {res['pool']['slope_ci']}  "
          f"intercept {res['pool']['intercept']:+.3f} m  CI {res['pool']['intercept_ci']}")
    verdict = ("MAP-consistent (slope~0)" if res["pool"]["map_consistent"]
               and not res["pool"]["boresight_consistent"]
               else "BORESIGHT-consistent (slope<0)" if res["pool"]["boresight_consistent"]
               and not res["pool"]["map_consistent"]
               else "AMBIGUOUS (CI spans both)")
    print(f"  >>> POOLED RANGE-SLOPE VERDICT: {verdict}")

    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2, default=float))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
