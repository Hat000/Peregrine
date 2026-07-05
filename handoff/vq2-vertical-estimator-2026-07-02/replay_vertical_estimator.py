"""OFFLINE REPRO + FIX EVIDENCE — the vertical-channel teleport/blind-climb (the gate-0 flight
ender) vs the A24 WASHOUT ``VerticalEstimator``, on the REAL run 20260702_040036_rl_s1_f1.

The successor to handoff/vq2-ff-owns-vertical-2026-07-01/repro_vertical_bangbang.py: that harness
proved ff_owns_vertical stops damping the drifting dead-reckoned vel[2]; THIS one shows what the
A19c fix still left broken and that the washout closes it. On the real run:

  * the nav est_z the alt-hold reads TELEPORTS -1.435 m in ONE tick when the sparse floor_height
    pin lands (t=1569.41) — the fd-vz the damper reads spikes ~-4.4 m/s -> a thrust kick from the
    kp term (2.0 * 1.4 = 2.8) AND the kd term in the same tick;
  * between pins that z is BLIND to the real climb: integrating the tlog a_up (the VERIFIED
    analyze_flight decode) shows the true upward velocity reaching +4-5 m/s while est_z crept
    ~6 cm — the damper had nothing real to damp, and the drone climbed into the ceiling
    (COLLISION id=1002 at t=1570.499).

WHAT THE WASHOUT DOES NOW (A24, see handoff/vq2_vertical_washout_spec_2026-07-02.md): the floor-pin
KF was ripped out — floor-height is a false premise (no warehouse grid), so there is NO altitude
state and NO pin-correction path. The estimator is a first-order washout (leaky integrator) on
``(vz, b_hat)`` that integrates the IMU-derived ``a_up`` with an exponential leak (tau=2 s), input
clamped to +/-30 m/s^2 and export clipped to +/-3 m/s — structurally cannot diverge. This replay
therefore no longer injects pins; it shows the PURE a_up-integration behaviour: a smooth, bounded vz
that TRACKS the real climb the teleporting est_z never showed, in place of the fd-vz teleport-slam.

Replays the run's HIGHRES_IMU accel + per-tick attitude through the washout ``VerticalEstimator``
and through the ff-owns-vertical alt-hold with ``use_vertical_estimator`` OFF vs ON, and asserts the
BEFORE fd-vz instability + the AFTER smooth-bounded-climb-tracking damping.

Inputs: reads the live run directory when present (extracting + caching the IMU window into
``imu_trace_20260702_040036.jsonl`` + ``nav_estimate_20260702_040036.jsonl`` beside this script,
so the committed harness is self-contained); else replays from the committed caches.

Run:  .venv/Scripts/python.exe handoff/vq2-vertical-estimator-2026-07-02/replay_vertical_estimator.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MAVLINK20", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from racer.contracts import NavState, Setpoint  # noqa: E402
from racer.gate_seeker import make_seeker_controller  # noqa: E402
from racer.vertical_estimator import VerticalEstimator  # noqa: E402

HERE = Path(__file__).resolve().parent
RUN = "20260702_040036_rl_s1_f1"
RUN_DIRS = [ROOT / "data" / "runs" / RUN,
            Path("C:/Users/Shadow/Peregrine/data/runs") / RUN]   # main checkout (worktree has no data/)
IMU_CACHE = HERE / "imu_trace_20260702_040036.jsonl"
NAV_CACHE = HERE / "nav_estimate_20260702_040036.jsonl"

PIN_STEP_M = 0.5        # |d est_z| in one tick above this = a floor-pin teleport (the live one: 1.435)
G = 9.80665


# ---------------------------------------------------------------------------
# ingest (cache-or-extract)
# ---------------------------------------------------------------------------
def _load_or_extract():
    """(imu_t_us, acc(N,3), nav rows) — from the committed caches, extracting them from the live
    run directory first when it is reachable (tlog parse via the analyze_flight loader)."""
    run_dir = next((d for d in RUN_DIRS if (d / "mavlink.tlog").exists()), None)
    if run_dir is not None and not (IMU_CACHE.exists() and NAV_CACHE.exists()):
        from analyze_flight import load_tlog                      # scripts/ (the standard reader)
        tl = load_tlog(run_dir / "mavlink.tlog")
        nav_rows = [json.loads(l) for l in (run_dir / "nav_estimate.jsonl").open() if l.strip()]
        t0 = nav_rows[0]["sim_time_ns"] / 1e3 - 1e6               # us; 1 s pre-roll for the on-pad check
        t1 = nav_rows[-1]["sim_time_ns"] / 1e3 + 2e5
        keep = (tl["imu_t"] >= t0) & (tl["imu_t"] <= t1)
        with IMU_CACHE.open("w") as f:
            for tt, aa in zip(tl["imu_t"][keep], tl["acc"][keep]):
                f.write(json.dumps({"t_us": int(tt), "acc": [float(x) for x in aa]}) + "\n")
        with NAV_CACHE.open("w") as f:
            for r in nav_rows:
                f.write(json.dumps(r) + "\n")
        print(f"[extract] cached {int(keep.sum())} IMU samples + {len(nav_rows)} nav ticks "
              f"from {run_dir}")
    imu = [json.loads(l) for l in IMU_CACHE.open() if l.strip()]
    nav = [json.loads(l) for l in NAV_CACHE.open() if l.strip()]
    imu_t = np.array([r["t_us"] for r in imu], dtype=np.float64) / 1e6
    acc = np.array([r["acc"] for r in imu], dtype=np.float64)
    return imu_t, acc, nav


def _rot_b2n(roll, pitch, yaw):
    """Body FRD -> world NED (analyze_flight convention — the VERIFIED a_up decode path)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


# ---------------------------------------------------------------------------
# the replay
# ---------------------------------------------------------------------------
def replay():
    imu_t, acc, nav = _load_or_extract()
    tick_t = np.array([r["sim_time_ns"] for r in nav], dtype=np.float64) / 1e9
    est_z = np.array([r["position_ned"][2] for r in nav], dtype=np.float64)
    rpy = np.array([[r["roll_rad"] or 0.0, r["pitch_rad"] or 0.0, r["yaw_rad"] or 0.0]
                    for r in nav], dtype=np.float64)
    # drop the frozen-clock tail (the sim halts its physics clock at the crash and re-sends)
    live = np.concatenate([[True], np.diff(tick_t) > 0])
    tick_t, est_z, rpy = tick_t[live], est_z[live], rpy[live]

    # pins = the est_z step-teleport ticks (the tight floor fix stepping the 6-state KF z)
    pin_i = [i for i in range(1, len(est_z)) if abs(est_z[i] - est_z[i - 1]) > PIN_STEP_M]
    pins = {tick_t[i]: float(est_z[i]) for i in pin_i}
    print(f"run {RUN}: {len(tick_t)} live ticks over {tick_t[-1] - tick_t[0]:.2f}s; "
          f"{len(pins)} floor-pin teleport(s): "
          + ", ".join(f"t={t:.3f} dz={est_z[i] - est_z[i - 1]:+.3f}m" for t, i in zip(pins, pin_i)))

    # per-IMU-sample a_up with the nearest at-or-before tick attitude (analyze_flight alignment)
    def a_up_at(j):
        i = max(int(np.searchsorted(tick_t, imu_t[j], side="right")) - 1, 0)
        return -(float(_rot_b2n(*rpy[i])[2] @ acc[j]) + G)

    # --- feed the WASHOUT VerticalEstimator: predict per IMU sample, NO pins -----------------
    # The A24 washout carries no altitude state and no floor-pin correction path (both ripped out --
    # floor-height was a false premise), so there is nothing to inject: it just integrates a_up with
    # an exponential leak. seed() at rest (the first ~1 s is the grounded pre-arm bias-capture window
    # during which vz stays 0), then predict(a_up, dt) per IMU sample; read vz (clipped +/-3 m/s).
    est = VerticalEstimator()
    est.seed()
    vz_ticks = np.zeros_like(tick_t)
    vz_imu, vz_imu_t = [], []      # per-IMU-sample vz (the TRUE smoothness record, ~140 Hz)
    ti = 0
    j0 = int(np.searchsorted(imu_t, tick_t[0]))
    for j in range(j0, len(imu_t)):
        dt = imu_t[j] - imu_t[j - 1] if j > j0 else 0.0
        est.predict(a_up_at(j), dt)
        vz_imu.append(est.vz)
        vz_imu_t.append(imu_t[j])
        while ti < len(tick_t) and tick_t[ti] <= imu_t[j]:
            vz_ticks[ti] = est.vz
            ti += 1
    while ti < len(tick_t):                                    # ticks past the last IMU sample
        vz_ticks[ti] = est.vz
        ti += 1
    vz_imu, vz_imu_t = np.asarray(vz_imu), np.asarray(vz_imu_t)

    # --- BEFORE: what the A19c damper actually read (fd + LP of the teleporting est_z) ---------
    vz_lp, fd_vz = 0.0, np.zeros_like(tick_t)
    for i in range(1, len(tick_t)):
        dt = tick_t[i] - tick_t[i - 1]
        fd = (est_z[i] - est_z[i - 1]) / dt if dt > 0 else 0.0
        vz_lp = 0.5 * fd + 0.5 * vz_lp                          # ff_vertical_vz_lp_alpha = 0.5
        fd_vz[i] = vz_lp

    print("\n-- damper INPUT, BEFORE vs AFTER (NED vz the kd term reads; pin tick marked) --")
    print(f"{'t(s)':>7} {'est_z':>8} {'fd_vz(BEFORE)':>13} {'vert_vz(AFTER)':>14}")
    for i in range(len(tick_t)):
        mark = "  <-- PIN" if tick_t[i] in pins else ""
        print(f"{tick_t[i] - tick_t[0]:7.2f} {est_z[i]:8.3f} {fd_vz[i]:13.2f} "
              f"{vz_ticks[i]:14.2f}{mark}")

    dz_tp = est_z[pin_i[0]] - est_z[pin_i[0] - 1]
    dt_tp = tick_t[pin_i[0]] - tick_t[pin_i[0] - 1]
    spike_before = float(np.abs(fd_vz[pin_i]).max()) if pin_i else 0.0
    # smoothness over the FLIGHT (pre-impact): the ceiling strike is a real ~140 m/s^2 specific-
    # force discontinuity the filter SHOULD step on — and it lands ~0.38 s BEFORE the sim's
    # COLLISION message (the sim reports late). Bound the window at the first impact signature
    # (|f| > 30 m/s^2; in-flight |f| never leaves single digits); post-impact the logged attitude
    # is tumbling-garbage anyway. Everything the damper acted on happened before this instant.
    f_mag = np.linalg.norm(acc, axis=1)
    imp = np.where((imu_t >= tick_t[0]) & (f_mag > 30.0))[0]
    t_impact = float(imu_t[imp[0]]) if len(imp) else float(tick_t[-1])
    pre_impact = vz_imu_t < t_impact
    step_after = float(np.abs(np.diff(vz_imu[pre_impact])).max())
    print(f"\nBEFORE: the est_z the alt-hold read TELEPORTED {dz_tp:+.3f} m in one tick "
          f"(raw fd {dz_tp / dt_tp:+.2f} m/s; the damper's LP read {spike_before:+.2f} m/s) — "
          f"AND showed only {est_z[pin_i[0] - 1] - est_z[0]:+.3f} m of motion over the prior "
          f"{tick_t[pin_i[0] - 1] - tick_t[0]:.2f} s of real climb (blind).")
    print(f"AFTER : the washout takes NO pin at all (no altitude state, no floor-pin path) — so "
          f"there is no teleport to correct; it integrates a_up straight through the pin tick.")
    print(f"AFTER : max per-IMU-sample |d vz| pre-impact = {step_after:.3f} m/s "
          f"(a continuous rate, no steps; the ceiling strike at t={t_impact:.3f} — "
          f"|f| spikes to {f_mag.max():.0f} m/s^2, {tick_t[-1] - t_impact:.2f} s before the sim's "
          f"COLLISION message — is a real discontinuity and is excluded)")
    print(f"AFTER : peak washout climb rate = {vz_ticks.min():+.2f} m/s NED — a bounded, smooth "
          f"read of the REAL upward climb the teleporting est_z (blind, "
          f"{est_z[pin_i[0] - 1] - est_z[0]:+.3f} m) never showed; the tau=2 s leak attenuates the "
          f"sustained part, so it stays well within the +/-3 m/s clip (the ceiling collision at "
          f"flight end confirms the climb was real)")

    # --- controller-level A/B: the same alt-hold, flag OFF vs ON ------------------------------
    def run_ctrl(vertical_on):
        ctrl = make_seeker_controller(ff_owns_vertical=True,
                                      use_vertical_estimator=vertical_on)
        thr = np.zeros_like(tick_t)
        for i in range(len(tick_t)):
            # A24 controller gates on vert_vz_est ALONE (vert_z_est is permanently NaN -- no
            # altitude state), so only the washout vz is fed.
            kw = dict(vert_vz_est=float(vz_ticks[i])) if vertical_on else {}
            nav_s = NavState(sim_time_ns=int(tick_t[i] * 1e9),
                             position_ned=np.array([0.0, 0.0, est_z[i]]),
                             velocity_ned=np.zeros(3),
                             roll=0.0, pitch=0.0, yaw=0.0,
                             angular_rate_body=np.zeros(3), **kw)
            sp = Setpoint(sim_time_ns=int(tick_t[i] * 1e9),
                          accel_ned=1.2 * np.array([1.0, 0.0, 0.0]), yaw=0.0)
            thr[i] = float(ctrl.command(nav_s, sp).thrust)
        return thr

    thr_off, thr_on = run_ctrl(False), run_ctrl(True)
    i_pin = pin_i[0]
    step_off = abs(thr_off[i_pin] - thr_off[i_pin - 1])
    step_on = abs(thr_on[i_pin] - thr_on[i_pin - 1])
    lo, hi = 0.05, 0.6
    rails = lambda thr: (int(np.sum(np.abs(thr - lo) < 1e-9)), int(np.sum(np.abs(thr - hi) < 1e-9)))
    print("\n-- ff-owns-vertical alt-hold on this replay, use_vertical_estimator OFF vs ON --")
    print(f"  OFF: pin-tick thrust step = {step_off:.3f}   rail hits lo/hi = {rails(thr_off)}   "
          f"std = {np.std(thr_off):.3f}")
    print(f"  ON : pin-tick thrust step = {step_on:.3f}   rail hits lo/hi = {rails(thr_on)}   "
          f"std = {np.std(thr_on):.3f}")
    hover = make_seeker_controller().hover_thrust
    tail = tick_t > tick_t[-1] - 1.5                            # the climb stretch before the crash
    print(f"  ON opposes the REAL climb: mean thrust over the last 1.5 s = "
          f"{np.mean(thr_on[tail]):.3f} vs hover {hover} (cut hard against vz "
          f"{vz_ticks[-1]:+.2f} m/s)  [OFF: {np.mean(thr_off[tail]):.3f} — blind]")

    # --- the asserted evidence (the DoD line items) --------------------------------------------
    assert spike_before > 2.0, ("repro lost: no fd-vz spike at the pin", spike_before)
    # the washout vz moves as a continuous rate (no pin step to inject) -- every per-IMU-sample
    # increment is small, vs the -4.39 m/s raw-fd / +2.13 m/s LP'd spike the fd path handed the damper
    assert step_after < 1.0, ("washout vz not smooth", step_after)
    # the washout TRACKS a substantial upward climb (the point: it SAW the climb the teleporting
    # est_z was blind to) while staying BOUNDED within its structural +/-3 m/s export clip (the
    # tau=2 leak attenuates the sustained part, so it reads a bounded fraction of the true +4-5 m/s
    # -- measured peak on this run is -1.50 m/s; the old -2.0 threshold predated the leak)
    assert float(vz_ticks.min()) < -1.0, ("washout missed the real climb", float(vz_ticks.min()))
    assert float(vz_ticks.min()) >= -3.0, ("export clip breached", float(vz_ticks.min()))
    assert step_off > 0.3, ("repro lost: OFF alt-hold no longer slams on the pin", step_off)
    assert step_on < 0.05, ("ON alt-hold still slams on the pin", step_on)
    assert np.mean(thr_on[tail]) < hover - 0.1, "ON alt-hold does not oppose the climb"
    print("\nOK: teleport-slam reproduced with the flag OFF; smooth vz + damped response with it ON.")


if __name__ == "__main__":
    replay()
