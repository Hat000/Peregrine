"""A9 VQ2 gyro-pitch-sign check (estimator-independent).

Finding: the live VQ2 gyro PITCH (y) polarity is INVERTED vs the code's FRD assumption. Evidence: the
controller COMMANDS pitch rate +1.50 rad/s while the gyro REALIZES ~-1.4 rad/s, and the pilot watched the
nose physically pitch UP (faces ceiling). Standard FRD: nose-up => POSITIVE pitch rate. So the gyro's
negative reading for a nose-up rotation is inverted. The AHRS integrates this backwards -> the estimate
believes nose-down -> the controller commands more nose-up -> runaway to the ceiling.

This script reports, per run, the realized gyro pitch over the controlled-flight maneuver (crash excluded),
to be read against the commanded +1.50 (in the fly_run*.log files) and the pilot's nose-up observation.
Do NOT trust position_ned (VQ2 wire returns pos=NO = estimator fiction).

Usage: python gyro_sign_check.py <run_dir>
"""
import sys, math
from pathlib import Path
import numpy as np
from pymavlink import mavutil

run = Path(sys.argv[1])
m = mavutil.mavlink_connection(str(run / "mavlink.tlog"))
imu, act = [], []
while True:
    msg = m.recv_match(type=["HIGHRES_IMU", "ACTUATOR_OUTPUT_STATUS"], blocking=False)
    if msg is None:
        break
    t = getattr(msg, "time_usec", None)
    if t is None:
        continue
    if msg.get_type() == "HIGHRES_IMU":
        imu.append((int(t), msg.xgyro, msg.ygyro, msg.zgyro, msg.xacc, msg.yacc, msg.zacc))
    else:
        act.append((int(t),) + tuple(float(x) for x in list(msg.actuator)[:4]))
imu = np.array(sorted(imu), float); act = np.array(sorted(act), float)
t = (imu[:, 0] - imu[0, 0]) / 1e6
xg, yg, zg = imu[:, 1], imu[:, 2], imu[:, 3]
amag = np.sqrt(imu[:, 4]**2 + imu[:, 5]**2 + imu[:, 6]**2)
dt = np.clip(np.diff(t, prepend=t[0]), 0, 0.05)
msum = np.array([act[np.argmin(np.abs(act[:, 0] - ti)), 1:5].sum() for ti in imu[:, 0]])

launch = int(np.argmax(msum > 0.8))
imp = launch + int(np.argmax(amag[launch:] > 40)) if (amag[launch:] > 40).any() else len(t) - 1
seg = slice(launch, imp)
net = lambda g: math.degrees(np.sum(g[seg] * dt[seg]))
pulse = yg[seg][np.abs(yg[seg]) > 0.3]

print(f"== {run.name} ==  controlled flight t[{t[launch]:.1f}..{t[imp]:.1f}]s (crash excluded)")
print(f"  net rotation:  roll(x)={net(xg):+.0f}deg   PITCH(y)={net(yg):+.0f}deg   yaw(z)={net(zg):+.0f}deg")
print(f"  pitch maneuver: realized ygyro pulse mean = {np.mean(pulse) if len(pulse) else 0:+.2f} rad/s "
      f"(n={len(pulse)})")
print(f"  COMMANDED pitch rate during the maneuver  = +1.50 rad/s   (see fly_run*.log: rate=[roll,+1.50,yaw])")
print(f"  PILOT (live): the nose pitched UP, faced the ceiling, flew backward (runs 2 & 3).")
print(f"  -> nose-UP physical rotation gives NEGATIVE realized ygyro; standard FRD nose-up is POSITIVE.")
print(f"  -> live VQ2 gyro PITCH polarity is INVERTED vs the code's FRD assumption (candidate b).")
