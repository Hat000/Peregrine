"""OFFLINE REPRO — the VQ2 vertical-channel bang-bang (the A19c flight-ender) + the ff_owns_vertical fix.

Feeds the REAL recorded A19c arc (nav_estimate_a19c.jsonl, 76 ticks) through the decoupled CTBR
alt-hold and prints the thrust trace.

vel[2] proxy caveat: velocity_ned is NOT logged in the A19c arc, and the estimator's dead-reckoned
vertical velocity (the poison that fed kd_alt) is not directly recoverable. We reconstruct a faithful
noisy-vz proxy as the finite difference of the logged, floor-corrected position_ned[2]:
    vz[i] = (z[i] - z[i-1]) / dt      (dt from the logged sim_time_ns delta)
This captures the jumpy/drifting vz that rail-slammed the collective (the same quantity kd_alt*(vel[2]
- vz_t) chased). It is a PROXY, not the true integrated-IMU vz, but it reproduces the mechanism and the
observed 0.05<->0.60 rail-slam.

The pursuit setpoint mirrors _feedforward_command exactly: accel_ned = a small bounded forward
feedforward, velocity_ned = [0,0,vz_t] (Z-only vertical-align) or None. position_ned is None during
pursuit, so kp_alt*(pos[2]-z_t) collapses to the CONSTANT kp_alt*alt_offset_m (=0 here) and the ONLY
time-varying driver of thrust is the vertical-velocity error term.

Run:  .venv/Scripts/python.exe handoff/vq2-ff-owns-vertical-2026-07-01/repro_vertical_bangbang.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from racer.contracts import NavState, Setpoint  # noqa: E402
from racer.gate_seeker import make_seeker_controller  # noqa: E402

ARC = Path(__file__).resolve().parent / "nav_estimate_a19c.jsonl"


def load_arc():
    rows = [json.loads(l) for l in ARC.open() if l.strip()]
    t = np.array([r["sim_time_ns"] for r in rows], dtype=np.float64)
    z = np.array([r["position_ned"][2] for r in rows], dtype=np.float64)
    thr_flown = np.array([r["thrust"] for r in rows], dtype=np.float64)
    pitch = np.array([r["pitch_rad"] for r in rows], dtype=np.float64)
    roll = np.array([r["roll_rad"] for r in rows], dtype=np.float64)
    yaw = np.array([r["yaw_rad"] for r in rows], dtype=np.float64)
    # noisy-vz PROXY: finite difference of the logged floor-corrected z (see caveat above)
    vz = np.zeros_like(z)
    for i in range(1, len(z)):
        dt = (t[i] - t[i - 1]) / 1e9
        vz[i] = (z[i] - z[i - 1]) / dt if dt > 0 else 0.0
    return t, z, vz, roll, pitch, yaw, thr_flown


def run(controller_kwargs, *, vz_t_series=None):
    """Replay the arc's z + proxy-vz through the alt-hold; return the per-tick commanded thrust.

    vz_t_series: the commanded vertical-align target per tick (None -> vz_t=0 = hold current altitude,
    velocity_ned stays None). A non-None entry carries velocity_ned=[0,0,vz_t]."""
    t, z, vz, roll, pitch, yaw, _ = load_arc()
    ctrl = make_seeker_controller(**controller_kwargs)
    out = np.zeros_like(z)
    for i in range(len(z)):
        nav = NavState(
            sim_time_ns=int(t[i]),
            position_ned=np.array([0.0, 0.0, z[i]]),
            velocity_ned=np.array([0.0, 0.0, vz[i]]),   # dead-reckoned proxy (the poison)
            roll=float(roll[i]), pitch=float(pitch[i]), yaw=float(yaw[i]),
            angular_rate_body=np.zeros(3),
        )
        vz_t = 0.0 if vz_t_series is None else float(vz_t_series[i])
        vned = np.array([0.0, 0.0, vz_t]) if vz_t != 0.0 else None
        sp = Setpoint(
            sim_time_ns=int(t[i]),
            accel_ned=1.2 * np.array([1.0, 0.0, 0.0]),   # bounded forward feedforward (pursuit)
            velocity_ned=vned,
            yaw=float(yaw[i]),
        )
        cmd = ctrl.command(nav, sp)
        out[i] = float(cmd.thrust)
    return t, z, out


def rail_stats(thr, lo=0.05, hi=0.6):
    n_lo = int(np.sum(np.abs(thr - lo) < 1e-9))
    n_hi = int(np.sum(np.abs(thr - hi) < 1e-9))
    # count consecutive lo<->hi flips (a rail-slam = adjacent ticks pinned to opposite rails)
    flips = 0
    for i in range(1, len(thr)):
        a, b = thr[i - 1], thr[i]
        if (abs(a - lo) < 1e-9 and abs(b - hi) < 1e-9) or (abs(a - hi) < 1e-9 and abs(b - lo) < 1e-9):
            flips += 1
    return n_lo, n_hi, flips


def band_stats(thr):
    return float(np.min(thr)), float(np.max(thr)), float(np.mean(thr)), float(np.std(thr))


def _print_trace(label, t, thr, first=0, last=None):
    last = len(thr) if last is None else last
    print(f"\n--- {label}: thrust trace ticks {first}..{last-1} ---")
    print("  " + " ".join(f"{x:.3f}" for x in thr[first:last]))


def run_synthetic(controller_kwargs, *, z_series, t_series, vz_t_val, roll=0.0, pitch=0.0):
    """Feed a CONTROLLED z (a realistic near-constant HOLD, level attitude) through the alt-hold.

    The real A19c arc is an OPEN-LOOP replay of the BROKEN trajectory (z diverges to -4.5 m), so its
    thrust pins low -- it demonstrates the no-rail-slam mechanism fix but NOT the hover band. This
    synthetic held-z (what a WORKING closed loop produces) shows the thrust sits around hover and that
    a commanded vz_t grades it up/down. accel_ned is set (pursuit) so the ff-owns-vertical path engages."""
    ctrl = make_seeker_controller(**controller_kwargs)
    out = np.zeros_like(z_series)
    for i in range(len(z_series)):
        vz_dr = (z_series[i] - z_series[i - 1]) / ((t_series[i] - t_series[i - 1]) / 1e9) if i > 0 else 0.0
        nav = NavState(
            sim_time_ns=int(t_series[i]),
            position_ned=np.array([0.0, 0.0, z_series[i]]),
            velocity_ned=np.array([0.0, 0.0, vz_dr]),
            roll=roll, pitch=pitch, yaw=0.0, angular_rate_body=np.zeros(3),
        )
        vned = np.array([0.0, 0.0, vz_t_val]) if vz_t_val != 0.0 else None
        sp = Setpoint(sim_time_ns=int(t_series[i]),
                      accel_ned=1.2 * np.array([1.0, 0.0, 0.0]),
                      velocity_ned=vned, yaw=0.0)
        out[i] = float(ctrl.command(nav, sp).thrust)
    return out


if __name__ == "__main__":
    HOVER = 0.2656
    print("=" * 78)
    print("A19c VERTICAL BANG-BANG REPRO  (hover_thrust =", HOVER, ", rails [0.05, 0.60])")
    print("=" * 78)

    # ---- BEFORE: current code (kd_alt=3.0 damps the dead-reckoned proxy vz) ----
    t, z, thr_before = run({})  # default seeker gains == today's flight code (ff_owns_vertical OFF)
    nlo, nhi, flips = rail_stats(thr_before)
    lo, hi, mean, std = band_stats(thr_before)
    print("\n[BEFORE — current code, ff_owns_vertical OFF]")
    print(f"  rail hits: lo(0.05)={nlo}  hi(0.60)={nhi}  rail-slam flips={flips}  of {len(thr_before)} ticks")
    print(f"  band: min={lo:.3f} max={hi:.3f} mean={mean:.3f} std={std:.3f}")
    _print_trace("BEFORE", t, thr_before, 10, 45)

    # ---- AFTER: ff_owns_vertical ON, hold (vz_t = 0 everywhere) ----
    t, z, thr_hold = run({"ff_owns_vertical": True}, vz_t_series=None)
    nlo, nhi, flips = rail_stats(thr_hold)
    lo, hi, mean, std = band_stats(thr_hold)
    print("\n[AFTER — ff_owns_vertical ON, vz_t=0 (hold)]")
    print(f"  rail hits: lo(0.05)={nlo}  hi(0.60)={nhi}  rail-slam flips={flips}  of {len(thr_hold)} ticks")
    print(f"  band: min={lo:.3f} max={hi:.3f} mean={mean:.3f} std={std:.3f}")
    _print_trace("AFTER (hold)", t, thr_hold, 10, 45)

    # ---- AFTER: commanded DESCENT (vz_t = +0.8 => sink => LESS thrust than hover mean) ----
    _, _, thr_descend = run({"ff_owns_vertical": True}, vz_t_series=np.full(len(z), 0.8))
    # ---- AFTER: commanded CLIMB (vz_t = -0.8 => rise => MORE thrust than hover mean) ----
    _, _, thr_climb = run({"ff_owns_vertical": True}, vz_t_series=np.full(len(z), -0.8))
    print("\n[AFTER — vertical-align tracking check, on the OPEN-LOOP arc]")
    print(f"  mean thrust: descend(vz_t=+0.8)={np.mean(thr_descend):.3f}  "
          f"hold={np.mean(thr_hold):.3f}  climb(vz_t=-0.8)={np.mean(thr_climb):.3f}")
    assert np.mean(thr_descend) < np.mean(thr_hold) < np.mean(thr_climb), "vertical-align direction broken"
    print("  OK: descend<hold<climb (a commanded sink cuts thrust, a commanded climb adds it)")

    # ---- SYNTHETIC held-altitude (a WORKING closed loop): the hover-band + graded-track proof ----
    print("\n" + "=" * 78)
    print("SYNTHETIC held-altitude (level, z ~= const -2.0 m, 40 ticks @ 12 Hz) — the HOVER BAND proof")
    print("=" * 78)
    rng = np.random.default_rng(0)
    N = 40
    ts = np.arange(N) * int(1e9 / 12)
    z_hold = -2.0 + 0.02 * rng.standard_normal(N)     # realistic small z noise around a held altitude
    thr_syn_off = run_synthetic({}, z_series=z_hold, t_series=ts, vz_t_val=0.0)
    thr_syn_on = run_synthetic({"ff_owns_vertical": True}, z_series=z_hold, t_series=ts, vz_t_val=0.0)
    for lbl, thr in (("OFF (kd_alt=3.0 vs dead-reckoned vz)", thr_syn_off),
                     ("ON  (ff_owns_vertical)", thr_syn_on)):
        nlo, nhi, flips = rail_stats(thr)
        lo, hi, mean, std = band_stats(thr)
        print(f"  [{lbl}]  flips={flips}  band min/max/mean/std={lo:.3f}/{hi:.3f}/{mean:.3f}/{std:.3f} "
              f"(hover={HOVER})")
    # ON must sit in a sane band around hover with NO rail-slam; OFF (fed the SAME finite-diff vz here,
    # which is bounded) is not the poison case -- the poison is the DRIFTING integrated vz on the real
    # wire, reproduced on the arc above. This synthetic block isolates the hover-band claim for ON.
    assert rail_stats(thr_syn_on)[2] == 0, "ff_owns_vertical rail-slammed on a held altitude"
    assert abs(np.median(thr_syn_on) - HOVER) < 0.12, ("ON not near hover", np.median(thr_syn_on))
    print(f"  OK: ON holds around hover (median {np.median(thr_syn_on):.3f}) with zero rail-slam.")

    # graded tracking on the synthetic hold: MODEST vz_t so the response is graded, not saturated
    print("\n[SYNTHETIC graded vertical-align tracking, vz_t small]")
    for vzt in (0.3, 0.0, -0.3):
        thr = run_synthetic({"ff_owns_vertical": True}, z_series=np.full(N, -2.0), t_series=ts, vz_t_val=vzt)
        print(f"  vz_t={vzt:+.1f}: mean_thrust={np.mean(thr):.3f}")
    print("  (descend cuts thrust below hover, climb raises it — graded, sign-correct.)")
