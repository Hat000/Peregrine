"""boresight_form.py — FORM discriminator (ANGULAR vs METRIC) for the head-on boresight test (P3, 2026-06-14).

Reads a shadow_gate{N}_rows.json (produced by shadow_gate4.py --gate N --json over the pooled head-on
recordings), bins accepted fixes by range, reports the AVERAGED sigma_vert per range (the commander's
heavy-averaging requirement), fits rel_vert = a + b*range, and emits ε magnitude + sign + FORM:
  * ANGULAR boresight: slope b != 0, intercept a ~ 0  -> ε = atan(-b) deg  -> P1 calibrates an EXTRINSIC ROTATION.
  * METRIC offset:     slope b ~ 0, intercept a != 0  -> ε = a metres      -> P1 calibrates a +L LEVER OFFSET.
The near/far contrast (~12 m vs ~23 m, Δ≈11 m) is the load-bearing read; its significance = Δ / averaged-σ.
"""
from __future__ import annotations
import argparse, json, math, random, statistics as st, sys
from pathlib import Path
random.seed(7)
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

def ols(xs, ys):
    n=len(xs); mx=sum(xs)/n; my=sum(ys)/n
    sxx=sum((x-mx)**2 for x in xs); sxy=sum((x-mx)*(y-my) for x,y in zip(xs,ys)); syy=sum((y-my)**2 for y in ys)
    b=sxy/sxx if sxx else float('nan'); a=my-b*mx
    r=sxy/math.sqrt(sxx*syy) if sxx and syy else float('nan')
    return b,a,r

def boot(xs,ys,nb=5000):
    n=len(xs); idx=list(range(n)); bs=[]; as_=[]
    for _ in range(nb):
        s=[random.choice(idx) for _ in range(n)]
        b,a,_=ols([xs[i] for i in s],[ys[i] for i in s]); bs.append(b); as_.append(a)
    bs.sort(); as_.sort(); lo=int(.025*nb); hi=int(.975*nb)
    return (bs[lo],bs[hi]),(as_[lo],as_[hi])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--rows", required=True, nargs='+', help="one or more shadow rows JSONs (pooled)")
    ap.add_argument("--gate", type=int, default=0)
    ap.add_argument("--clean", action="store_true", help="restrict to 4-corner & reproj<1px")
    ap.add_argument("--rmax", type=float, default=26.0, help="drop fixes beyond this range (vision-cap noise)")
    a=ap.parse_args()
    rows=[]
    for _rp in a.rows: rows += json.loads(Path(_rp).read_text())["rows"]
    acc=[r for r in rows if r.get("accepted") and r["true_range_m"]<=a.rmax]
    if a.clean: acc=[r for r in acc if r.get("n_corners")==4 and r.get("reproj_px",9)<1.0]
    if len(acc)<10: print(f"gate-{a.gate}: only {len(acc)} accepted fixes — insufficient"); return 1
    rng=[r["true_range_m"] for r in acc]; vert=[r["rel_vert"] for r in acc]; cross=[r["rel_cross"] for r in acc]
    nan_pose=sum(1 for r in acc if not math.isfinite(r.get("rel_vert",0)))
    print(f"gate-{a.gate} FORM · N={len(acc)} accepted (clean={a.clean}, rmax={a.rmax}) range[{min(rng):.1f},{max(rng):.1f}] m  nonfinite={nan_pose}")
    print(f"{'bin(m)':>10} {'N':>5} {'rel_vert':>9} {'sd_vert':>8} {'avg_sd':>8} {'app.pitch°':>10}")
    for lo,hi in [(6,10),(10,14),(14,18),(18,22),(22,26)]:
        b=[(R,v) for R,v in zip(rng,vert) if lo<=R<hi and math.isfinite(v)]
        if len(b)<3: print(f"{f'{lo}-{hi}':>10} {len(b):>5}   (sparse)"); continue
        vs=[x[1] for x in b]; Rs=[x[0] for x in b]
        m=st.mean(vs); sd=st.pstdev(vs); avg=sd/math.sqrt(len(vs)); ang=math.degrees(math.atan2(m,st.mean(Rs)))
        print(f"{f'{lo}-{hi}':>10} {len(vs):>5} {m:>+9.3f} {sd:>8.3f} {avg:>8.4f} {ang:>+10.2f}")
    gv=[(R,v) for R,v in zip(rng,vert) if math.isfinite(v)]
    R=[x[0] for x in gv]; V=[x[1] for x in gv]
    b,a0,r=ols(R,V); (blo,bhi),(alo,ahi)=boot(R,V); ang=math.degrees(math.atan(-b))
    print(f"\nFORM FIT rel_vert = a + b*range  (N={len(gv)}):")
    print(f"  slope b   = {b:+.5f} m/m  CI[{blo:+.5f},{bhi:+.5f}]  (pearson {r:+.2f})  -> if angular, ε={ang:+.3f}°")
    print(f"  intercept = {a0:+.3f} m      CI[{alo:+.3f},{ahi:+.3f}]")
    print(f"  control rel_cross mean {st.mean(cross):+.3f} m (≈0 ⇒ no gross azimuth/attitude error)")
    near=[v for RR,v in zip(R,V) if 10<=RR<14]; far=[v for RR,v in zip(R,V) if 22<=RR<26]
    if len(near)>=3 and len(far)>=3:
        dn,df=st.mean(near),st.mean(far); sdn=st.pstdev(near)/math.sqrt(len(near)); sdf=st.pstdev(far)/math.sqrt(len(far))
        dd=df-dn; sig=math.sqrt(sdn**2+sdf**2); nsig=abs(dd)/sig if sig else float('inf')
        print(f"\n  NEAR ~12 m: {dn:+.3f} ± {sdn:.4f} m (N={len(near)})   FAR ~23 m: {df:+.3f} ± {sdf:.4f} m (N={len(far)})")
        print(f"  Δ(far−near) = {dd:+.3f} m  vs averaged-σ {sig:.4f}  →  {nsig:.1f}σ  ({'RESOLVED' if nsig>=3 else 'MARGINAL — re-fly for more near-range N'})")
    slope_nz = not (blo<=0<=bhi); int_nz = not (alo<=0<=ahi)
    print()
    if slope_nz and not int_nz:
        print(f">>> FORM = ANGULAR boresight (slope≠0, intercept≈0). ε ≈ {ang:+.3f}°  →  P1 calibrates an EXTRINSIC ROTATION.")
    elif int_nz and not slope_nz:
        print(f">>> FORM = METRIC offset (intercept≠0, slope≈0). ε ≈ {a0:+.3f} m  →  P1 calibrates a +L LEVER OFFSET.")
    elif slope_nz and int_nz:
        print(f">>> FORM = MIXED (both terms ≠0): angular {ang:+.3f}° + metric {a0:+.3f} m. P1: dominant term wins; flag for review.")
    else:
        print(f">>> FORM = UNRESOLVED (neither CI excludes 0) — need more near-range N or wider baseline. Re-fly.")
    return 0

if __name__=="__main__": raise SystemExit(main())
