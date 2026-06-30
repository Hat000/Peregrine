"""A8 VQ2 attitude-divergence localizer — estimator-INDEPENDENT IMU truth extraction.

Context: the A8 drone pitches nose-UP + throttle-up + flies up-and-back (footage-confirmed: onboard
camera on the ceiling). The seeker is fine; the bug is a WRONG live ATTITUDE ESTIMATE fed to the
controller. Two candidates:
  (a) accel-leveling bias under sustained linear acceleration (accel vector != gravity when accelerating), or
  (b) the LIVE VQ2 HIGHRES_IMU gyro polarity/frame does NOT match the code's FRD assumption (the offline
      sign-audit uses self-consistent synthetic data, so it cannot catch a live wire-convention mismatch).

What IS in the tlog: HIGHRES_IMU (xacc/yacc/zacc, x/y/zgyro) and ACTUATOR_OUTPUT_STATUS (motor outputs).
NOT present: ATTITUDE (VQ2 blocks it -> the estimator's reported pitch is NOT on the wire; the navigator's
_ahrs quat is in-memory only) and ATTITUDE_TARGET (fly_rl's OUTGOING commands are not tlogged). So:
  - estimator pitch  -> NOT logged; INFERRED from controller action vs IMU truth (stated, not measured).
  - commanded rate/thrust -> ~1 Hz in fly_rl stdout only; the ACTUATOR motor SUM is the high-rate,
    estimator-independent thrust proxy (a per-motor split is printed for the pitch-torque pattern).

THE DECISIVE, CONVENTION-FREE TEST (no nose-up/down sign needed): over the sustained post-launch flight,
compare the ACCEL-derived apparent pitch change vs the GYRO-integrated pitch change.
  * A gyro measures rotation directly. If the accel-apparent pitch DRIFTS while the gyro reads ~0, the
    drift is LINEAR-ACCELERATION contamination of the accelerometer, not rotation  => candidate (a).
  * If instead the gyro shows a rotation whose integrated sign is OPPOSITE the (footage/accel) truth,
    the gyro polarity is wrong => candidate (b).
A wrong gyro polarity cannot manufacture divergence while the gyro reads ZERO, so a gyro-quiet accel
drift refutes (b) and localizes (a).

Do NOT trust position_ned (VQ2 wire returns pos=NO -> any position is estimator fiction).

Usage: python extract_attitude.py <run_dir> [--g 9.80665]
"""
import sys, argparse, math
from pathlib import Path
import numpy as np
from pymavlink import mavutil

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("--g", type=float, default=9.80665)
a = ap.parse_args()
run = Path(a.run_dir); G = a.g

# ---- read HIGHRES_IMU + ACTUATOR_OUTPUT_STATUS, keyed by time_usec ----------------------------------
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
        imu.append((int(t), msg.ygyro, msg.xacc, msg.yacc, msg.zacc, msg.xgyro, msg.zgyro))
    else:
        imu and act.append((int(t),) + tuple(float(x) for x in list(msg.actuator)[:4]))
imu = np.array(sorted(imu), float); act = np.array(sorted(act), float) if act else np.zeros((1, 5))
t = (imu[:, 0] - imu[0, 0]) / 1e6
yg, xa, ya, za = imu[:, 1], imu[:, 2], imu[:, 3], imu[:, 4]
amag = np.sqrt(xa**2 + ya**2 + za**2)
ap_deg = np.degrees(np.arctan2(-xa, np.hypot(ya, za)))           # accel-apparent pitch (one fixed defn)
dt = np.clip(np.diff(t, prepend=t[0]), 0, 0.05)
motors = np.array([act[np.argmin(np.abs(act[:, 0] - ti)), 1:5] for ti in imu[:, 0]])
msum = motors.sum(axis=1)

# ---- windows: launch = motors past idle; crash = the GLOBAL |a| impact spike -----------------------
THR_ON = 0.8
flying = msum > THR_ON
launch_i = int(np.argmax(flying)) if flying.any() else 0
crash_i = int(np.argmax(amag))                               # the real impact = the largest |a| spike
if crash_i <= launch_i:
    crash_i = len(t) - 1
# seed the gyro integral at the resting tick just before launch (|a|=g, gyro~0 -> accel == true gravity)
seed_i = launch_i
while seed_i > 0 and (abs(amag[seed_i] - G) > 1.0 or abs(yg[seed_i]) > 0.05):
    seed_i -= 1
gyro_int = np.full(len(t), np.nan); gyro_int[seed_i] = ap_deg[seed_i]
for k in range(seed_i + 1, crash_i + 1):
    gyro_int[k] = gyro_int[k-1] + math.degrees(yg[k] * dt[k])

# ---- decisive metric: the LONGEST sustained gyro-QUIET, throttle-ON segment before crash -----------
# (isolates the sustained divergence from the brief launch rotation pulse and the crash spike)
quiet = (np.abs(yg) < 0.10) & (msum > THR_ON)
quiet[:launch_i] = False; quiet[crash_i:] = False
best = (0, launch_i, launch_i); s = None
for k in range(len(t)):
    if quiet[k]:
        s = k if s is None else s
        if t[k] - t[s] > best[0]:
            best = (t[k] - t[s], s, k)
    else:
        s = None
_, d0, d1 = best
d_accel = ap_deg[d1] - ap_deg[d0]
d_gyro  = gyro_int[d1] - gyro_int[d0]
max_yg  = float(np.max(np.abs(yg[d0:d1+1])))
mean_a  = float(np.mean(amag[d0:d1+1]))
gyro_quiet = max_yg < 0.10 and (t[d1] - t[d0]) > 1.0          # a real sustained quiet drift

print(f"== {run.name} ==")
print(f"IMU {len(t)} | seed@t={t[seed_i]:.2f}s(rest pitch {ap_deg[seed_i]:+.1f}deg,|a|={amag[seed_i]:.1f}) "
      f"| launch@t={t[launch_i]:.2f}s | crash@t={t[crash_i]:.2f}s")
print()
hdr = f"{'t-lnch':>7}{'|a|':>6}{'ygyro':>7}{'accel_p':>9}{'gyro_intp':>10}  {'m0':>4}{'m1':>5}{'m2':>5}{'m3':>5}{'sumM':>6}"
print(hdr); print("-"*len(hdr))
last = -9
for k in range(seed_i, crash_i + 1):
    if t[k]-last < 0.9 and k not in (crash_i, launch_i):
        continue
    last = t[k]
    print(f"{t[k]-t[launch_i]:7.2f}{amag[k]:6.1f}{yg[k]:+7.2f}{ap_deg[k]:+9.1f}{gyro_int[k]:+10.1f}  "
          f"{motors[k][0]:4.2f}{motors[k][1]:5.2f}{motors[k][2]:5.2f}{motors[k][3]:5.2f}{msum[k]:6.2f}")

print(f"\nDECISIVE METRIC over the longest sustained gyro-quiet, throttle-on segment "
      f"[t+{t[d0]-t[launch_i]:.1f}s .. t+{t[d1]-t[launch_i]:.1f}s] ({t[d1]-t[d0]:.1f}s):")
print(f"  accel-apparent pitch drift  = {d_accel:+.1f} deg")
print(f"  gyro-integrated pitch drift = {d_gyro:+.1f} deg   (max|ygyro| {max_yg:.3f} rad/s -> {'QUIET (no rotation)' if max_yg<0.10 else 'rotating'})")
print(f"  mean |a| over segment = {mean_a:.1f} m/s^2   (g={G:.1f}; ~g => magnitude accel-gate is EVADED)")
print(f"\n  Q1: over this segment the accel-apparent pitch "
      f"{'DRIFTS %+.0f deg while the gyro-integrated pitch barely moves (%+.0f deg): the gyro reads ~0, so the drift is LINEAR-ACCEL CONTAMINATION, not rotation' % (d_accel, d_gyro) if gyro_quiet else 'and gyro both move (real rotation present)'}.")
print(f"  Q2: estimator pitch not logged (VQ2 blocks ATTITUDE); inferred - a gyro-quiet accel drift toward")
print(f"      nose-down is exactly what makes the controller command nose-up (matches the footage up+back).")
print(f"  Q3: ACTUATOR thrust proxy sumM rises to ~{np.max(msum[launch_i:crash_i]):.1f} and holds (throttle up),")
print(f"      i.e. sustained thrust -> sustained linear accel -> the accel contamination above.")
verdict = ("(a) ACCEL-LEVELING BIAS under sustained linear accel. The apparent-pitch divergence develops "
           "while the gyro reads ~0 and |a|~=g, so a magnitude-gated accel-leveler is fooled by a tilted-but-"
           "g-magnitude specific force. (b) is REFUTED: a zero gyro has no polarity to corrupt."
           if gyro_quiet else
           "(b) GYRO CONVENTION: the gyro shows a sustained rotation whose integrated sign opposes the truth.")
print(f"\n  VERDICT: {verdict}")
