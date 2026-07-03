"""A32: replay run 20260703_172104 HIGHRES_IMU through the flight ESKFAHRS config.

Two modes:
  * default (diagnosis, the evidence behind the A32 spec): the DEPLOYED config
    (use_accel_motion_reject=True, v1 unbounded inflation), stepped exactly as the Navigator
    does (per nav tick, latest IMU sample, dt from consecutive nav sim_time_ns), with the
    accel-update accept/reject path instrumented -- proves the self-locking distrust loop.
  * --v2 (the A32 offline GATE): use_accel_trust_v2=True (bounded inflation + free-run limit +
    Huber-soft chi2 + |a|~g deweight cap + gravity-recovery watchdog); add --imu-rate to step
    the ESKF per IMU sample between nav ticks (the A32 ring-buffer ingestion). PASS CRITERION
    (spec §3.5.1): the tail (t >= 28.16 s, drone at rest INVERTED) must converge to
    roll 180 deg +/- 10 deg by t <= 29.2 s and stay there to the end of the log.

Run with the repo venv python, PYTHONUTF8=1:
    .venv/Scripts/python.exe handoff/tools/a32_eskf_replay.py [--v2] [--imu-rate]
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

parser = argparse.ArgumentParser()
parser.add_argument("--v2", action="store_true", help="use_accel_trust_v2=True (the A32 fix)")
parser.add_argument("--imu-rate", action="store_true",
                    help="step the ESKF per IMU sample (the A32 ring-buffer ingestion)")
parser.add_argument("--run", default=str(ROOT / "data/runs/20260703_172104_rl_s1_f1"))
args = parser.parse_args()
RUN = Path(args.run)

from racer.ahrs.eskf import ESKFAHRS, GRAVITY, _quat_to_R_wxyz  # noqa: E402
from racer.ahrs.ahrs_adapter import AHRSAttitudeSource  # noqa: E402

GYRO_SIGN = np.array([-1.0, -1.0, -1.0])

# ---- load IMU stream from tlog ----
from pymavlink import mavutil  # noqa: E402
conn = mavutil.mavlink_connection(str(RUN / "mavlink.tlog"))
imu_t, imu_acc, imu_gyr = [], [], []
while True:
    msg = conn.recv_match(type="HIGHRES_IMU", blocking=False)
    if msg is None:
        break
    imu_t.append(int(msg.time_usec) * 1000)
    imu_acc.append((msg.xacc, msg.yacc, msg.zacc))
    imu_gyr.append((msg.xgyro, msg.ygyro, msg.zgyro))
imu_t = np.array(imu_t, dtype=np.int64)
imu_acc = np.array(imu_acc)
imu_gyr = np.array(imu_gyr) * GYRO_SIGN
print(f"IMU samples: {len(imu_t)}  rate={(len(imu_t)-1)/((imu_t[-1]-imu_t[0])/1e9):.1f}Hz")
print(f"mode: v2={args.v2} imu_rate={args.imu_rate}")

# ---- nav ticks ----
rows = [json.loads(l) for l in open(RUN / "nav_estimate.jsonl", encoding="utf-8") if l.strip()]
nav_t = np.array([r["sim_time_ns"] for r in rows], dtype=np.int64)
nav_pitch = np.array([r.get("pitch_rad") or 0.0 for r in rows])
nav_roll = np.array([r.get("roll_rad") or 0.0 for r in rows])
# drop the trailing clock-reset tick(s)
good = np.flatnonzero(np.diff(nav_t) < 0)
end = good[0] + 1 if len(good) else len(nav_t)
nav_t, nav_pitch, nav_roll, rows = nav_t[:end], nav_pitch[:end], nav_roll[:end], rows[:end]

# ---- instrumentation: monkeypatch _update_accel to record the decision path (v1 diagnosis
# mode only -- the v2 path carries NATIVE instrumentation: last_theta_g_deg /
# last_deweight_total / watchdog_fires) ----
trace = []
if not args.v2:
    orig_update = ESKFAHRS._update_accel
    def instrumented(self, accel):
        accel_mag = float(np.linalg.norm(accel))
        rec = {"amag": accel_mag, "path": None, "md": np.nan, "gate": np.nan, "inflate": np.nan}
        trace.append(rec)
        if accel_mag < 1e-6:
            rec["path"] = "zero"; return
        if not self._accel_magnitude_in_band(accel_mag):
            rec["path"] = "band_reject"; return
        gate = self._accel_gate_weight(accel_mag)
        rec["gate"] = gate
        if gate < 1e-4:
            rec["path"] = "gate_reject"; return
        R_wb = _quat_to_R_wxyz(self._q)
        g_body = R_wb.T @ np.array([0.0, 0.0, GRAVITY])
        g_hat = g_body / max(np.linalg.norm(g_body), 1e-9)
        h_hat = -g_hat
        a_hat = accel / accel_mag
        innovation = a_hat - h_hat
        H = np.zeros((3, 6)); H[:, :3] = -np.array([[0, -g_hat[2], g_hat[1]], [g_hat[2], 0, -g_hat[0]], [-g_hat[1], g_hat[0], 0]])
        sigma_a = self.accel_noise_std / GRAVITY
        # NOTE: probing the inflation would re-anchor R_ref twice; save/restore.
        R_ref_saved = self._R_ref.copy()
        infl = self._accel_motion_inflation(accel)
        self._R_ref = R_ref_saved
        rec["inflate"] = infl
        R_meas = (sigma_a**2 / gate) * infl * np.eye(3)
        if self.accel_chi2_thresh > 0.0:
            S = H @ self._P @ H.T + R_meas
            md = float(innovation @ np.linalg.solve(S, innovation))
            rec["md"] = md
            if md > self.accel_chi2_thresh:
                rec["path"] = "chi2_reject"
        if rec["path"] is None:
            rec["path"] = "accept"
        orig_update(self, accel)
    ESKFAHRS._update_accel = instrumented

src = AHRSAttitudeSource(eskf=ESKFAHRS(gyro_noise_std=0.01, accel_gate_alpha=10.0,
                                       use_accel_motion_reject=True,
                                       use_accel_trust_v2=bool(args.v2)))

# ---- replay at nav-tick cadence ----
# latest-sample-and-hold (faithful to the flown _step_ahrs), or --imu-rate: every IMU sample in
# (prev_tick, tick] stepped with per-sample dt (the A32 Navigator._ingest_imu_ring behaviour).
t0 = nav_t[0]
rep_pitch, rep_roll, tick_trace, tick_v2 = [], [], [], []
prev_ns = None
imu_wm = None      # imu-rate watermark (index into imu_t)
idx = np.searchsorted(imu_t, nav_t, side="right") - 1
for k, tns in enumerate(nav_t):
    i = idx[k]
    if i < 0:
        rep_pitch.append(0.0); rep_roll.append(0.0); tick_trace.append(None); tick_v2.append(None)
        continue
    dt = 0.0 if prev_ns is None else (tns - prev_ns) / 1e9
    n_before = len(trace)
    if args.imu_rate and dt > 0 and imu_wm is not None:
        js = np.flatnonzero((imu_t > imu_t[imu_wm]) & (imu_t <= tns))
        prev_t = imu_t[imu_wm]
        if len(js) == 0:
            src.ingest(imu_acc[i], imu_gyr[i], float(dt))
        else:
            for j in js:
                dt_j = (imu_t[j] - prev_t) / 1e9
                src.ingest(imu_acc[j], imu_gyr[j], float(dt_j))
                prev_t = imu_t[j]
            imu_wm = int(js[-1])
    else:
        src.ingest(imu_acc[i], imu_gyr[i], float(dt))
        imu_wm = int(i)
    r, p, y = src.euler_rpy
    rep_roll.append(r); rep_pitch.append(p)
    tick_trace.append(trace[-1] if len(trace) > n_before else None)
    tick_v2.append({"theta_g": src.eskf.last_theta_g_deg,
                    "deweight": src.eskf.last_deweight_total,
                    "wd": src.eskf.watchdog_fires} if args.v2 else None)
    prev_ns = tns

rep_pitch = np.array(rep_pitch); rep_roll = np.array(rep_roll)
t = (nav_t - t0) / 1e9
deg = 180 / math.pi

# fidelity: replay vs logged (only meaningful in the faithful v1 latest-sample mode)
err_p = np.abs(rep_pitch - nav_pitch) * deg
err_r = np.abs(rep_roll - nav_roll) * deg
print(f"\nreplay vs flight log: pitch err median={np.median(err_p):.2f} p95={np.percentile(err_p,95):.2f} max={err_p.max():.2f} deg")
print(f"                      roll  err median={np.median(err_r):.2f} p95={np.percentile(err_r,95):.2f} max={err_r.max():.2f} deg")

if not args.v2:
    paths = np.array([tt["path"] if tt else "none" for tt in tick_trace])
    def phase(name, lo, hi):
        w = (t >= lo) & (t < hi)
        tot = w.sum()
        if not tot: return
        counts = {p: int(np.sum(paths[w] == p)) for p in ("accept", "chi2_reject", "band_reject", "gate_reject", "none")}
        amag = np.array([tick_trace[i]["amag"] if tick_trace[i] else np.nan for i in np.flatnonzero(w)])
        mds = np.array([tick_trace[i]["md"] if tick_trace[i] else np.nan for i in np.flatnonzero(w)])
        infl = np.array([tick_trace[i]["inflate"] if tick_trace[i] else np.nan for i in np.flatnonzero(w)])
        print(f"  {name:22s} ticks={tot:3d} accept={counts['accept']:3d} chi2rej={counts['chi2_reject']:3d} "
              f"bandrej={counts['band_reject']:3d} |a|med={np.nanmedian(amag):5.2f} mdmed={np.nanmedian(mds):8.1f} inflmed={np.nanmedian(infl):8.1f}")
    print("\naccel-update decision by phase:")
    phase("pre-contact 0-25.4", 0, 25.4)
    phase("contact-1 25.4-26.4", 25.4, 26.4)
    phase("between 26.4-27.3", 26.4, 27.3)
    phase("event-2 27.3-28.0", 27.3, 28.0)
    phase("tail 28.0-end", 28.0, t[-1] + 0.1)
else:
    dws = np.array([tv["deweight"] if tv else np.nan for tv in tick_v2])
    tgs = np.array([tv["theta_g"] if tv else np.nan for tv in tick_v2])
    pre = (t < 25.4)
    print(f"\nv2 deweight: pre-contact median={np.nanmedian(dws[pre]):.1f} p90={np.nanpercentile(dws[pre],90):.1f}"
          f"   tail median={np.nanmedian(dws[t>=28.0]):.1f}")
    print(f"v2 watchdog fires (total): {src.eskf.watchdog_fires}")

# tail detail
print("\ntail detail: t, est roll/pitch, accel-implied roll/pitch, |a|" + (", theta_g, deweight, wd" if args.v2 else ""))
last = -1
for k in range(len(t)):
    if t[k] < 25.2: continue
    if t[k] - last < 0.15: continue
    last = t[k]
    i = idx[k]
    a = imu_acc[i]
    n = np.linalg.norm(a)
    g_b = -a / max(n, 1e-9)
    a_roll = math.atan2(g_b[1], g_b[2]) * deg
    a_pitch = math.atan2(-g_b[0], math.hypot(g_b[1], g_b[2])) * deg
    extra = ""
    if args.v2 and tick_v2[k]:
        extra = f" tg={tick_v2[k]['theta_g']:6.1f} dw={tick_v2[k]['deweight']:7.1f} wd={tick_v2[k]['wd']}"
    print(f"  t={t[k]:6.2f} est=({rep_roll[k]*deg:+7.1f},{rep_pitch[k]*deg:+6.1f}) accel=({a_roll:+7.1f},{a_pitch:+6.1f}) |a|={n:5.2f}{extra}")

# ---- THE OFFLINE GATE (spec §3.5.1) ----
# tail (t >= 28.16, at rest inverted): converge to roll 180 +/- 10 deg by t <= 29.2 and HOLD.
TAIL_T0, GATE_T, GATE_TOL = 28.16, 29.2, 10.0
tail = np.flatnonzero(t >= TAIL_T0)
if len(tail):
    def wrap180(x):
        return (x + 180.0) % 360.0 - 180.0
    roll_err = np.abs(wrap180(rep_roll[tail] * deg - 180.0))
    inside = roll_err <= GATE_TOL
    # earliest tick from which the estimate STAYS inside the band to the end of the log
    t_rec = None
    for kk in range(len(tail)):
        if inside[kk:].all():
            t_rec = t[tail[kk]]
            break
    final_roll = rep_roll[tail[-1]] * deg
    print(f"\n=== OFFLINE GATE (roll -> 180 +/- {GATE_TOL:.0f} deg by t <= {GATE_T} s) ===")
    print(f"  tail starts t={t[tail[0]]:.2f}s; final roll={final_roll:+.1f} deg "
          f"(err {abs(wrap180(final_roll-180.0)):.1f} deg)")
    if t_rec is None:
        print("  RESULT: FAIL -- never converged into the band")
    else:
        verdict = "PASS" if t_rec <= GATE_T else "FAIL (late)"
        print(f"  RESULT: {verdict} -- recovered (and held) from t={t_rec:.2f}s "
              f"({t_rec - TAIL_T0:.2f}s after tail start)")
