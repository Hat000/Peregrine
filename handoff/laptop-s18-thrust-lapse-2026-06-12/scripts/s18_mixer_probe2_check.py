import numpy as np
# S17 mixer model: d_ax = kerr*(target-omega) + khold*omega; u = clip(c +- d, idle, 1)
G0=np.array([2.501,2.504,2.231]); S=0.30; kerr=0.073; khold=0.046; idle=0.05; zeta=0.34
def g(axis): return G0[axis]/(1-S*np.pi/np.pi)  # full stick |cmd|=pi
# rows: (name, collective, axis 0=roll/2=yaw, measured up/down motor pair, measured settled omega)
rows=[("c100_r31",1.0,0,(0.999,0.644),10.93),
      ("c60_r31", 0.6,0,(0.732,0.467),10.99),
      ("zhov_r31",0.266,0,(0.487,0.053),10.79),
      ("c100_y31",1.0,2,(1.000,0.651),9.21)]
print("S17 model prediction vs measured (settled, full-stick pi):")
print(f"{'row':>9} {'c':>5} {'ax':>4} {'tgt':>5} {'om':>5} {'d_mdl':>6} {'d_meas':>7} {'up_mdl':>7} {'up_meas':>7} {'dn_mdl':>7} {'dn_meas':>7}")
for name,c,ax,(up_m,dn_m),om in rows:
    tgt=g(ax)*np.pi
    d=kerr*(tgt-om)+khold*om
    if ax==2: d*= zeta/(zeta+c)   # yaw effectiveness
    up=np.clip(c+d,idle,1.0); dn=np.clip(c-d,idle,1.0)
    d_meas=(up_m-dn_m)/2 if not (up_m>=0.999 or dn_m<=0.051) else None
    # back out measured d from the UNCLIPPED motor of the pair
    if up_m<0.999 and dn_m>0.051: dmeas=(up_m-dn_m)/2
    elif dn_m<=0.051: dmeas=up_m-c        # down clipped -> use up
    else: dmeas=c-dn_m                    # up clipped -> use down
    print(f"{name:>9} {c:>5.2f} {ax:>4} {tgt:>5.1f} {om:>5.1f} {d:>6.3f} {dmeas:>7.3f} {up:>7.3f} {up_m:>7.3f} {dn:>7.3f} {dn_m:>7.3f}")
print("\nback-out: clean (unclipped) c60_r31 -> roll khold = dmeas/om =", round(0.1325/10.99,4),
      "vs model khold(yaw-derived)=0.046")
