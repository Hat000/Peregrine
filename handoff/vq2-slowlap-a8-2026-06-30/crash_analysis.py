"""Reconstruct the A7 spawn crash from the tlog: realized IMU tilt+gyro+|a|, and any position/vel,
across the ~1.4s flight, to classify the failure (thrust collapse -> climb/fall -> contact)."""
import sys, math
from pathlib import Path
sys.path.insert(0, "src")
import numpy as np
from pymavlink import mavutil

sess = Path(sys.argv[1])
m = mavutil.mavlink_connection(str(sess/"mavlink.tlog"))
imu=[]; pos=[]; att=[]; sp=[]; t0=None
types=["HIGHRES_IMU","LOCAL_POSITION_NED","ATTITUDE","ATTITUDE_TARGET","RACE_STATUS"]
counts={}
while True:
    msg=m.recv_match(type=types, blocking=False)
    if msg is None: break
    typ=msg.get_type(); counts[typ]=counts.get(typ,0)+1
    if typ=="HIGHRES_IMU":
        t=int(msg.time_usec)
        if t0 is None: t0=t
        dt=(t-t0)/1e6
        ax,ay,az=msg.xacc,msg.yacc,msg.zacc
        imu.append((dt, math.degrees(math.atan2(ay,-az)), math.degrees(math.atan2(-ax,math.hypot(ay,az))),
                    msg.xgyro,msg.ygyro,msg.zgyro, float(np.linalg.norm([ax,ay,az]))))
    elif typ=="LOCAL_POSITION_NED":
        pos.append((msg.x,msg.y,msg.z,msg.vx,msg.vy,msg.vz))
    elif typ=="ATTITUDE_TARGET":
        # body rates commanded + thrust
        sp.append((msg.body_roll_rate,msg.body_pitch_rate,msg.body_yaw_rate,msg.thrust))
print("msg counts:",counts)
print(f"\nLOCAL_POSITION_NED rows: {len(pos)}  (VQ2 should be ~0 = position denied)")
if pos:
    for p in pos[:5]: print("  pos",[round(v,2) for v in p])
print(f"\nHIGHRES_IMU rows: {len(imu)} over {imu[-1][0] if imu else 0:.2f}s")
for s in imu[::max(1,len(imu)//25)]:
    print(f"  t={s[0]:5.2f}  tilt(r,p)=({s[1]:+5.0f},{s[2]:+5.0f})  gyro=[{s[3]:+5.2f},{s[4]:+5.2f},{s[5]:+5.2f}]  |a|={s[6]:4.1f}")
if imu:
    g=max(imu,key=lambda s:max(abs(s[3]),abs(s[4]),abs(s[5])))
    amin=min(imu,key=lambda s:s[6])
    print(f"\npeak|gyro| t={g[0]:.2f} -> [{g[3]:+.2f},{g[4]:+.2f},{g[5]:+.2f}]   min|a|={amin[6]:.1f} @t={amin[0]:.2f} (free-fall if <<9.8)")
print(f"\nATTITUDE_TARGET (cmd) rows: {len(sp)}")
for s in sp[::max(1,len(sp)//20)] if sp else []:
    print(f"  cmd rate=[{s[0]:+.2f},{s[1]:+.2f},{s[2]:+.2f}] thrust={s[3]:.3f}")
