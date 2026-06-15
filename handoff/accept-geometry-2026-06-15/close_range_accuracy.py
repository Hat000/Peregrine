"""close_range_accuracy.py -- is the <12 m lateral-scatter blow-up a real widening or a few flip outliers?
Robust scatter (MAD/IQR), per-axis bias vs scatter, n_det / n_corners / reproj diagnostics, and the
both-flip cross-check vs FORM_RESOLUTION (vertical spread<0.024). Moving approach, accepted fixes.
"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; DATA = HERE / "data"

def load():
    rows=[]
    for f in ["b1_g0_rows.json","b2_g0_rows.json"]:
        rows+=json.load(open(DATA/f))["rows"]
    return rows
def col(rows,k,d,dt=float): return np.array([r.get(k,d) for r in rows],dtype=dt)

def robust(x):
    x=np.asarray(x,float); med=np.median(x); mad=1.4826*np.median(np.abs(x-med))
    p16,p84=np.percentile(x,[16,84]); return med,mad,(p84-p16)/2,np.std(x),x.min(),x.max()

rows=load()
R=col(rows,"true_range_m",np.nan); sp=col(rows,"speed_mps",np.nan)
acc=col(rows,"accepted",False,bool); mov=sp>0.5
rcross=col(rows,"rel_cross",np.nan); rvert=col(rows,"rel_vert",np.nan); ralong=col(rows,"rel_along",np.nan)
rep=col(rows,"reproj_px",np.nan); d2=col(rows,"d2_rel",np.nan); nC=col(rows,"n_corners",0,int)
nd=col(rows,"n_det",0,int); wfe=col(rows,"world_fix_err_m",np.nan); prng=col(rows,"pose_range_m",np.nan)

print("LATERAL(cross) residual: robust scatter vs std (accepted, moving) -- outliers vs real widening?")
print("  bin(m)  N   med_cr   MAD_cr  p16_84/2  STD_cr   min/max_cr        |  med|cr|  frac|cr|>0.2")
for lo,hi in [(4,6),(6,8),(8,10),(10,12),(12,14),(14,16),(16,18),(18,20),(20,22),(22,24)]:
    m=acc&mov&(R>=lo)&(R<hi); n=m.sum()
    if n<3: continue
    med,mad,iqr,std,mn,mx=robust(rcross[m])
    print("  [%2d,%2d) %4d  %+6.3f  %6.3f  %7.3f  %6.3f  [%+.2f,%+.2f]   |  %5.3f   %4.2f"%(
        lo,hi,n,med,mad,iqr,std,mn,mx,np.median(np.abs(rcross[m])),(np.abs(rcross[m])>0.2).mean()))
print()
print("Same for VERTICAL (cross-check vs FORM_RESOLUTION both-flip spread<0.024):")
print("  bin(m)  N   med_vt   MAD_vt   STD_vt   min/max_vt")
for lo,hi in [(4,6),(6,8),(8,10),(10,12),(12,14),(14,16),(16,18),(18,20),(20,22),(22,24)]:
    m=acc&mov&(R>=lo)&(R<hi); n=m.sum()
    if n<3: continue
    med,mad,iqr,std,mn,mx=robust(rvert[m])
    print("  [%2d,%2d) %4d  %+6.3f  %6.3f  %6.3f  [%+.2f,%+.2f]"%(lo,hi,n,med,mad,std,mn,mx))
print()
print("DIAGNOSTICS at close range (accepted, moving): why does lateral widen <12 m?")
print("  bin(m)  N   med_nd  med_nC  nC<4%  med_reproj  med_d2  med_poseRng  med_trueRng  pose-true_bias")
for lo,hi in [(4,6),(8,10),(10,12),(12,14),(14,16),(18,20),(22,24)]:
    m=acc&mov&(R>=lo)&(R<hi); n=m.sum()
    if n<3: continue
    print("  [%2d,%2d) %4d  %5.1f   %5.1f  %4.0f   %6.3f   %5.2f   %7.2f     %7.2f    %+6.3f"%(
        lo,hi,n,np.median(nd[m]),np.median(nC[m]),100*(nC[m]<4).mean(),np.median(rep[m]),np.median(d2[m]),
        np.median(prng[m]),np.median(R[m]),np.median(prng[m]-R[m])))
print()
# Is the 8-12m cross blow-up driven by the few 3-corner or high-reproj frames? split it.
print("8-12 m accepted lateral scatter, SPLIT by reproj quality (moving):")
m=acc&mov&(R>=8)&(R<12)
for lab,sub in [("reproj<0.6",rep[m]<0.6),("reproj>=0.6",rep[m]>=0.6),("nC==4",nC[m]==4),("nC<4",nC[m]<4)]:
    s=rcross[m][sub]
    if len(s)>=3:
        print("  %-12s N=%3d  med_cr %+6.3f  MAD %6.3f  std %6.3f  med|cr| %5.3f"%(lab,len(s),np.median(s),1.4826*np.median(np.abs(s-np.median(s))),np.std(s),np.median(np.abs(s))))
