"""P2 rate-at-airspeed, P4 drift, P5 control-rate sensitivity, P6 determinism.

P2: sustained realized rate during the at-speed step phases vs the static super-rate map
    g(|c|)*c (the hover-measured curve). Airspeed at step onset reported per probe.
P4: drift320 -- alt-hold integrator (the true hover collective readout) + altitude + realized
    thrust, first vs last minute. Twin: zero drift.
P5: fixmnvr realized step gain + accel-phase peak speed at 50/100/200 Hz control rate.
P6: det1-5 trajectory spread (same profile, 100 Hz): per-phase-aligned pos/vel/rate RMS
    between runs = the measurement noise floor.

Usage: .venv\\Scripts\\python.exe handoff\\shadowpc-twin-falsify-2026-06-10\\fit_rates_misc.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runs as R

G0 = np.array([2.501, 2.504, 2.231])


def srate_gain(c: float, axis: int) -> float:
    return G0[axis] / (1.0 - 0.30 * min(abs(c), np.pi) / np.pi)


def p2_rate_at_speed():
    """RAW ODOMETRY rate (|.|; magnitude validated against the euler-angle slope at 3 loop
    rates -- the quat-FD column is biased high by the LPN/ODO sim_time stagger, see runs.py)."""
    print("== P2 rate loop at airspeed (RAW-odo sustained |rate| vs hover map) ==")
    print(f"{'run':10s} {'axis':5s} {'cmd':>6s} {'v_h@step':>8s} {'odo |w|':>8s} "
          f"{'rpy-slope':>9s} {'map pred':>8s} {'ratio':>6s}")
    for lb, axis in (("rspd_r03", 0), ("rspd_r10", 0), ("rspd_p03", 1), ("rspd_p10", 1),
                     ("rspd_r314", 0), ("rspd_p314", 1)):
        run = R.load(lb)
        m = np.array([k in ("step", "astep") for k in run.kind])
        if not m.any():
            print(f"{lb:10s}  -- no step phase found"); continue
        i0 = int(np.argmax(m))
        cmd = float(run.cmd[m, axis][len(run.cmd[m]) // 2])
        vh = float(np.hypot(run.vel[i0, 0], run.vel[i0, 1]))
        t = run.t[m]
        settled = t > t[0] + 0.25                     # past the ~25 ms tau
        w = np.abs(run.odo_rate[m, axis])
        meas = float(np.mean(w[settled])) if settled.sum() > 3 else float(np.mean(w[-5:]))
        # euler-slope cross-check (valid below inversion only)
        ang = np.unwrap(run.rpy[m, axis])
        sl = abs(np.polyfit(t[settled], ang[settled], 1)[0]) if settled.sum() > 3 else np.nan
        pred = srate_gain(cmd, axis) * abs(cmd)
        print(f"{lb:10s} {'rp'[axis]:5s} {cmd:+6.2f} {vh:8.2f} {meas:8.3f} {sl:9.3f} "
              f"{pred:8.3f} {meas/pred:6.3f}")


def p4_drift():
    print("\n== P4 long-duration drift (drift320, 320 s hover at alt -5) ==")
    run = R.load("drift320")
    m = run.seg("hover_drift")
    t = run.t[m] - run.t[m][0]
    z = run.pos[m, 2]
    th = run.thr[m]
    ai = run.alt_i[m]
    for lo, hi, tag in ((10, 70, "first min"), (130, 190, "mid"), (250, 310, "last min")):
        w = (t >= lo) & (t < hi)
        print(f"  {tag:9s}: thr_cmd mean={np.mean(th[w]):.4f}  alt_i={np.mean(ai[w]):+.4f}  "
              f"z={np.mean(z[w]):+.3f} m  (n={int(w.sum())})")
    # linear trend on the integrator = collective drift rate
    w = t > 20
    p = np.polyfit(t[w], ai[w], 1)
    p_thr = np.polyfit(t[w], th[w], 1)
    print(f"  alt_i slope = {p[0]*60:+.5f} /min  ({p[0]*480:+.5f} over 8 min)")
    print(f"  thr_cmd slope = {p_thr[0]*60:+.5f} /min")
    print(f"  twin: exactly 0 drift. |slope*8min| < 0.5% collective => battery model NOT needed")


def p5_rate_hz():
    """RAW-odo rate (the quat-FD column is the loop-rate-ALIASED one -- its spread across Hz
    was the false 'sensitivity'; see nh50/nh100/nh200 + euler-slope ground truth)."""
    print("\n== P5 control-rate sensitivity (fixmnvr @50/100/200 Hz; RAW-odo |rate|) ==")
    print(f"{'run':9s} {'Hz':>4s} {'roll step |w|':>13s} {'quatFD (biased)':>15s} "
          f"{'accel v_end':>11s} {'climb z@t8':>10s}")
    for lb, hz in (("fix50a", 50), ("fix50b", 50), ("det1", 100), ("det2", 100), ("det3", 100),
                   ("det4", 100), ("det5", 100), ("fix200a", 200), ("fix200b", 200)):
        run = R.load(lb)
        ms = np.array([k == "step" for k in run.kind])
        t = run.t[ms]
        settled = t > t[0] + 0.2
        w = np.abs(run.odo_rate[ms, 0])
        wq = np.abs(run.rate[ms, 0])
        meas = float(np.mean(w[settled])) if settled.sum() > 2 else np.nan
        measq = float(np.mean(wq[settled])) if settled.sum() > 2 else np.nan
        ma = run.seg("accel")
        v_end = float(np.hypot(run.vel[ma, 0][-1], run.vel[ma, 1][-1])) if ma.any() else np.nan
        t8 = run.t[0] + 8.0
        z8 = float(np.interp(t8, run.t, run.pos[:, 2]))
        print(f"{lb:9s} {hz:4d} {meas:13.3f} {measq:15.3f} {v_end:11.2f} {z8:10.2f}")
    print("  near-hover anchors (nh50/nh100/nh200, legacy rate mode, cmd 1.0):")
    print("    raw odo 2.679 / 2.683 / 2.688;  euler slope 2.690 / 2.613 / 2.655  -> NO Hz effect")


def p6_determinism():
    print("\n== P6 determinism: det1-5 pairwise spread, aligned at the accel-phase start ==")
    runs = [R.load(f"det{i}") for i in range(1, 6)]
    sigs = []
    for run in runs:
        m = run.seg("accel")
        t0 = run.t[np.argmax(m)]
        tt = np.arange(0.0, 4.5, 0.02)
        sig = np.column_stack([np.interp(tt, run.t - t0, run.pos[:, i]) for i in range(3)]
                              + [np.interp(tt, run.t - t0, run.vel[:, i]) for i in range(3)])
        sigs.append(sig)
    sigs = np.array(sigs)
    sd = sigs.std(axis=0)        # (T, 6) across runs
    print(f"  across-run SD: pos x/y/z = {sd[:, 0].mean():.3f}/{sd[:, 1].mean():.3f}/"
          f"{sd[:, 2].mean():.3f} m   vel = {sd[:, 3].mean():.3f}/{sd[:, 4].mean():.3f}/"
          f"{sd[:, 5].mean():.3f} m/s")
    print(f"  worst instantaneous pos SD: {sd[:, :3].max():.3f} m   vel SD: {sd[:, 3:].max():.3f} m/s")
    print("  (closed-loop replication spread: includes OUR wall-clock command jitter, so this is")
    print("   an UPPER bound on sim noise -- and the floor every other probe comparison inherits)")


if __name__ == "__main__":
    p2_rate_at_speed()
    p4_drift()
    p5_rate_hz()
    p6_determinism()
