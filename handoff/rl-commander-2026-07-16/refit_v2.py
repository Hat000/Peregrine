"""Re-fit the super-rate model to the COMPLETE 3-axis curves (ShadowPC round 2, 2026-07-16).
Corrects the roll/pitch base (single-point inference 2.22 was wrong; direct ampsweep -> ~2.47).
Caveat: roll/pitch held 0.3s, yaw held 0.5s; the plant rate CREEPS +4-9% over a hold, so yaw's
0.5s reads are inflated vs a maneuver-timescale (~0.3s) gain. Fit each axis to its as-measured curve."""
import math
PI=math.pi
def cmd(a): return 3.14*a
data={
 'roll' :[(0.1,2.47),(0.2,2.46),(0.4,2.65),(0.6,2.92),(0.8,3.07)],   # 0.3s holds
 'pitch':[(0.1,2.47),(0.2,2.46),(0.4,2.64),(0.6,2.90),(0.8,3.04)],   # 0.3s holds
 'yaw'  :[(0.1,2.23),(0.2,2.31),(0.4,2.48),(0.6,2.67),(0.8,2.89)],   # 0.5s holds (creep-inflated)
}
def fit(pts):
    best=None; s=0.0
    while s<=0.60:
        xs=[1.0/(1.0 - s*min(cmd(a),PI)/PI) for a,_ in pts]; gs=[g for _,g in pts]
        G0=sum(g*x for g,x in zip(gs,xs))/sum(x*x for x in xs)
        rmse=math.sqrt(sum((G0*x-g)**2 for g,x in zip(gs,xs))/len(gs))
        if best is None or rmse<best[0]: best=(rmse,s,G0)
        s+=0.002
    return best
print(f"{'axis':6s} {'G0':>6s} {'s':>6s} {'rmse':>7s}   per-point model vs meas")
fits={}
for ax,pts in data.items():
    rmse,s,G0=fit(pts); fits[ax]=(G0,s)
    row=" ".join(f"{G0/(1-s*min(cmd(a),PI)/PI):.2f}/{g:.2f}" for a,g in pts)
    print(f"{ax:6s} {G0:6.3f} {s:6.3f} {rmse:7.4f}   {row}")
# shared-s variant (yaw drives less; use roll/pitch which are cleaner 0.3s)
print("\nRECOMMENDED faithful nominals (per-axis G0, per-axis s):")
print(f"  rate_gain     = [{fits['roll'][0]:.3f}, {fits['pitch'][0]:.3f}, {fits['yaw'][0]:.3f}]")
print(f"  super_rate_s  = [{fits['roll'][1]:.3f}, {fits['pitch'][1]:.3f}, {fits['yaw'][1]:.3f}]")
print("\nCompare to CURRENT DiffAero defaults rate_gain=[2.501,2.504,2.231] (flat) and my shipped")
print("faithful=[2.222,2.198,2.164]/s=0.315:")
for ax in data:
    G0,s=fits[ax]
    ship={'roll':2.222,'pitch':2.198,'yaw':2.164}[ax]
    print(f"  {ax:6s} true base {G0:.3f}  vs shipped {ship:.3f} ({100*(ship-G0)/G0:+.0f}%)  vs default 2.50/2.23")
