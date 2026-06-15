"""fit_and_plot.py -- (a) fit the surrogate band-pass rising edge to the POINTED accept|fov curve to get a
concrete accept_rlo proposal; (b) check for any speed dependence within the (<=7 m/s) data; (c) plot
accept-rate + lateral-accuracy vs range with the surrogate overlay. Moving approach, pooled b1+b2.
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit
HERE=Path(__file__).resolve().parent; DATA=HERE/"data"
import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

def load():
    rows=[]
    for f in ["b1_g0_rows.json","b2_g0_rows.json"]: rows+=json.load(open(DATA/f))["rows"]
    return rows
def col(rows,k,d,dt=float): return np.array([r.get(k,d) for r in rows],dtype=dt)
rows=load()
R=col(rows,"true_range_m",np.nan); sp=col(rows,"speed_mps",np.nan)
fov=col(rows,"g4_in_fov",False,bool); acc=col(rows,"accepted",False,bool)
rcross=col(rows,"rel_cross",np.nan); mov=sp>0.5

# surrogate current
PMAX0,RLO0,WLO0,RHI0,WHI0=0.8419703950411151,16.191103551308505,1.0,28.137694558909192,1.0
def bandpass(r,pmax,rlo,wlo,rhi,whi):
    zl=np.clip((r-rlo)/wlo,-30,30); zh=np.clip((rhi-r)/whi,-30,30)
    return pmax/((1+np.exp(-zl))*(1+np.exp(-zh)))

# build empirical accept|fov per 1m bin (moving, in-fov denominator), 6..25 m
ctr,emp,nfov=[],[],[]
for lo in np.arange(6,25,1.0):
    m=fov&mov&(R>=lo)&(R<lo+1)
    if m.sum()<8: continue
    ctr.append(lo+0.5); emp.append(acc[m].mean()); nfov.append(m.sum())
ctr=np.array(ctr); emp=np.array(emp); nfov=np.array(nfov)

# Fit rising edge only: fix rhi/whi at current, fit pmax,rlo,wlo. Weight by sqrt(N). Use <=24m.
fitmask=ctr<=24.5
p0=[0.82,12.0,1.0]
def f_rise(r,pmax,rlo,wlo): return bandpass(r,pmax,rlo,wlo,RHI0,WHI0)
popt,_=curve_fit(f_rise,ctr[fitmask],emp[fitmask],p0=p0,sigma=1/np.sqrt(nfov[fitmask]),
                 bounds=([0.5,8,0.3],[1.0,18,4.0]),maxfev=20000)
print("POINTED band-pass rising-edge fit (moving, in-fov denom, 6-24 m):")
print("  fitted pmax=%.3f  rlo=%.2f  wlo=%.2f   (surrogate has pmax=%.3f rlo=%.2f wlo=%.2f)"%(
    popt[0],popt[1],popt[2],PMAX0,RLO0,WLO0))
print("  => pointed rising edge midpoint rlo ~ %.1f m  (vs un-pointed inc7 fit 16.2 m)"%popt[1])
print()
# accurate-fix floor: lateral MAD per bin, find where it crosses 0.12 (the ~budget)
print("  range | emp_acc|fov | surrogate(rlo16.2) | refit(rlo%.1f) | lat_MAD(accepted)"%popt[1])
for c,e,n in zip(ctr,emp,nfov):
    m=acc&mov&(R>=c-0.5)&(R<c+0.5)
    mad=1.4826*np.median(np.abs(rcross[m]-np.median(rcross[m]))) if m.sum()>=5 else np.nan
    print("  %4.1f  |   %.3f    |     %.3f          |    %.3f      |   %s"%(
        c,e,bandpass(c,PMAX0,RLO0,WLO0,RHI0,WHI0),bandpass(c,*popt,RHI0,WHI0),
        ("%.3f"%mad) if not np.isnan(mad) else " n/a"))
print()

# speed lever within data: matched range 12-20 m, low (1-3) vs high (5-7) m/s
print("SPEED LEVER (matched range 12-20 m, accepted, lateral MAD) -- weak (data <=7 m/s):")
for lab,smask in [("1-3 m/s",(sp>=1)&(sp<3)),("3-5 m/s",(sp>=3)&(sp<5)),("5-7 m/s",(sp>=5)&(sp<=7))]:
    m=acc&smask&(R>=12)&(R<20)
    if m.sum()<8: print("  %-8s N<8"%lab); continue
    mad=1.4826*np.median(np.abs(rcross[m]-np.median(rcross[m])))
    print("  %-8s N=%3d  acc|fov(12-20)=%.2f  lat_MAD=%.3f"%(lab,m.sum(),
        acc[fov&smask&(R>=12)&(R<20)].mean(),mad))
print()

# ---- PLOT ----
fig,ax=plt.subplots(1,2,figsize=(13,5))
rr=np.linspace(6,28,200)
ax[0].plot(rr,bandpass(rr,PMAX0,RLO0,WLO0,RHI0,WHI0),'r--',label='surrogate (rlo=16.2, un-pointed inc7)')
ax[0].plot(rr,bandpass(rr,*popt,RHI0,WHI0),'g-',label='refit pointed (rlo=%.1f)'%popt[1])
ax[0].plot(ctr,emp,'ko-',ms=5,label='empirical POINTED accept|in-fov (moving)')
ax[0].axvspan(6,12,alpha=0.12,color='red'); ax[0].axvspan(12,16,alpha=0.12,color='orange')
ax[0].set_xlabel('true range (m)'); ax[0].set_ylabel('P(accept | gate in FoV)')
ax[0].set_title('Pointing extends the accept band inward (16->~12 m)'); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
ax[0].set_xlim(6,28); ax[0].set_ylim(0,1)
# right: lateral MAD vs range
mads,cmads=[],[]
for c in ctr:
    m=acc&mov&(R>=c-0.5)&(R<c+0.5)
    if m.sum()>=5: cmads.append(c); mads.append(1.4826*np.median(np.abs(rcross[m]-np.median(rcross[m]))))
ax[1].plot(cmads,mads,'bs-',ms=5,label='lateral MAD (accepted, moving)')
ax[1].axhline(0.245,color='r',ls=':',label='margin lat-σ ceiling 0.245 (r=0.30,fr=0.50)')
ax[1].axhline(0.10,color='gray',ls=':',label='clean op-band σ_lat ~0.10')
ax[1].axvspan(6,12,alpha=0.12,color='red')
ax[1].set_xlabel('true range (m)'); ax[1].set_ylabel('lateral residual MAD (m)')
ax[1].set_title('Lateral accuracy: clean to ~12 m, blows up below'); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
ax[1].set_xlim(6,25); ax[1].set_ylim(0,0.5)
plt.tight_layout(); out=HERE/"accept_geometry_curves.png"; plt.savefig(out,dpi=110)
print("plot ->",out)
