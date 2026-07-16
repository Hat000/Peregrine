#!/usr/bin/env python
"""sysID post-pass: extract the raw ~117 Hz HIGHRES_IMU stream from a session's mavlink.tlog into
sysid_vq2_imu_raw.csv -> columns: sim_time_ns, gyro_x/y/z (rad/s, body), accel_x/y/z (m/s^2, body).

The 40 Hz sysid_vq2_log.csv nearest-sample merge blurs the rate rise-time (tau ~19 ms < the 25 ms
tick); this raw stream resolves it. Gyro/accel are the RAW wire values (HIGHRES_IMU, pre gyro_sign --
default identity), so the commander reconciles signs against the logged wire commands.

Usage: python scripts/sysid_tlog_to_imu_raw.py <session_dir | mavlink.tlog> [out.csv]
"""
import sys, csv, os
from pymavlink import mavutil


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    p = sys.argv[1]
    tlog = p if p.endswith(".tlog") else os.path.join(p, "mavlink.tlog")
    if not os.path.exists(tlog):
        sys.exit(f"no tlog at {tlog}")
    out = (sys.argv[2] if len(sys.argv) > 2
           else os.path.join(os.path.dirname(os.path.abspath(tlog)), "sysid_vq2_imu_raw.csv"))
    m = mavutil.mavlink_connection(tlog)
    n = 0
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sim_time_ns", "gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"])
        while True:
            msg = m.recv_match(type="HIGHRES_IMU", blocking=False)
            if msg is None:
                break
            w.writerow([int(msg.time_usec) * 1000,
                        msg.xgyro, msg.ygyro, msg.zgyro,
                        msg.xacc, msg.yacc, msg.zacc])
            n += 1
    print(f"wrote {n} HIGHRES_IMU rows -> {out}")


if __name__ == "__main__":
    main()
