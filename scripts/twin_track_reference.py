"""Track the TOGT time-optimal reference line through the OFFLINE TWIN with the real
CTBR Controller — the reality check on the planner bound (laptop-togt-bound-2026-06-10).

The bound (rl/reference_line_vq1.json) assumes an ideal CTBR plant: collective + body-rate
limits, NO inner-loop lag (tau~=0.02 s), NO slew limit (~260 rad/s^2), NO command latency.
This script flies the line on the twin WITH all of those (the measured super-rate static
gain map included) using the stack's existing geometric tracker — Controller in BODY_RATE
mode (non-decoupled): a_des = accel_ff + PD(pos err, vel err) -> tilt + collective via
_accel_to_attitude -> rotvec body-rate law. The TOGT-ideal vs twin-tracked gap (time + gate
miss) is the deliverable.

Twin configuration = faithful PHYSICS, canonical TELEMETRY: twin_fit.faithful_config(
super_rate=True) keeps the measured gain map / slew / tau / drag / hover, but the ODOMETRY
report-sign quirks are disabled and the matching controller-side compensations dropped —
they are exact inverses of each other (reporting artifacts, not dynamics), so including
them is a wash; the live stack runs the full sign circus (twin_fly_course Task C validates
that path). The controller's ff_gain stays at the flat 2.5 the LIVE stack uses, so the
super-rate map's extra gain at large commands is exercised exactly as it would be live.

Usage:  python scripts/twin_track_reference.py [--ref rl/reference_line_vq1.json]
            [--latency 0.0] [--dt 0.01] [--sweep]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scipy.spatial.transform import Rotation

from racer.contracts import ControlMode, NavState, Setpoint
from racer.controller import Controller
from racer.reference_line import ReferenceLine
from racer.twin import CtbrPlant
from racer.twin_fit import faithful_config

_G = 9.80665
HOVER = 0.2656

# Race-tracking controller: the stack's geometric law with clamps opened up to the plant's
# real envelope (the VQ1 cruise config caps speed at 5 m/s by design — useless here).
# kp_pos/kd_vel ~= a critically-damped ~1.6 rad/s position loop; kp_att/kd_att from the
# twin-validated attitude loop, ff_gain = the live flat 2.5.
TRACK_GAINS = dict(
    mode=ControlMode.BODY_RATE, decoupled=False,
    hover_thrust=HOVER, thrust_slope_mps2=_G / HOVER,       # twin throttle map: a_up = g*t/hover
    kp_pos=2.5, kd_vel=3.2, kp_att=10.0, kd_att=0.30,
    ff_gain=2.5, max_body_rate_rps=11.5,
    max_tilt_rad=np.deg2rad(80.0), max_accel_mps2=None, max_pos_error_m=None,
    body_rate_sign=np.array([1.0, 1.0, -1.0]),              # physical yaw-cmd inversion (plant rate_sign)
)


def canonical_telemetry_config(latency_s: float = 0.0):
    cfg = faithful_config(super_rate=True)
    return replace(cfg,
                   odo_att_report_sign=np.ones(3), odo_rate_report_sign=np.ones(3),
                   cmd_latency_s=latency_s)


def gates_ned(course: dict) -> list[dict]:
    out = []
    for g in course["gates"]:
        p = np.asarray(g["pos_zup"], dtype=float)
        out.append({"pos": np.array([p[0], -p[1], -p[2]]),
                    # zup yaw pi (normal -X) -> NED normal -X as well (y flips, x doesn't)
                    "normal": np.array([np.cos(g["yaw"]), -np.sin(g["yaw"]), 0.0]),
                    "half_opening": course["inner_opening_m"] / 2.0})
    return out


def plane_misses(P: np.ndarray, T: np.ndarray, gates: list[dict]):
    """Sequential gate passages: min-miss crossing of each gate's (infinite) plane after the
    previous gate's passage -- a fast line can cut a plane far outside the opening while
    lining up, so the first crossing is not the passage (same convention as
    twin_fly_course.gate_plane_miss and scripts/togt/analyze.py)."""
    res = []
    t_prev = -np.inf
    for g in gates:
        n, c = g["normal"], g["pos"]
        u_h = np.array([-n[1], n[0], 0.0])
        u_v = np.array([0.0, 0.0, 1.0])
        d = (P - c) @ n
        idx = np.flatnonzero((np.sign(d[:-1]) * np.sign(d[1:]) < 0) & (T[1:] > t_prev))
        best = None
        for i in idx:
            f = d[i] / (d[i] - d[i + 1])
            pc = P[i] + f * (P[i + 1] - P[i])
            tc = float(T[i] + f * (T[i + 1] - T[i]))
            cand = {"t": tc, "miss": float(np.hypot((pc - c) @ u_h, (pc - c) @ u_v))}
            if best is None or cand["miss"] < best["miss"]:
                best = cand
        res.append(best)
        if best is not None:
            t_prev = best["t"]
    return res


def track(ref: ReferenceLine, *, dt: float = 0.01, latency_s: float = 0.0,
          gains: dict | None = None, t_extra: float = 1.5,
          timescale: float = 1.0, lead_s: float = 0.0) -> dict:
    """Track the reference played back ``timescale``x slower (same geometry; velocity ff
    scaled 1/k, accel ff 1/k^2 -- exact time dilation). A time-OPTIMAL reference is thrust-
    saturated ~84% of the lap, leaving no control headroom to recover any lag, so k=1 pure
    time-indexed tracking diverges by design; the smallest k the twin flies VALIDLY is the
    twin-tracked lap. ``lead_s`` samples the reference slightly ahead (lag compensation)."""
    cfg = canonical_telemetry_config(latency_s)
    sp0 = ref.sample(0.0)
    q0 = Rotation.from_euler("ZYX", [sp0.yaw, 0.0, 0.0]).as_quat()      # level at ref yaw
    plant = CtbrPlant(cfg, position_ned=sp0.position_ned.copy(),
                      q_wxyz=np.array([q0[3], q0[0], q0[1], q0[2]]))
    ctrl = Controller(**{**TRACK_GAINS, **(gains or {})})

    n_steps = int((ref.total_duration_s * timescale + t_extra) / dt)
    P, T = np.empty((n_steps, 3)), np.empty(n_steps)
    sat_rate = 0
    for k in range(n_steps):
        t = k * dt
        st = plant.state()
        nav = NavState(sim_time_ns=st.sim_time_ns, position_ned=st.position_ned,
                       velocity_ned=st.velocity_ned, roll=st.roll, pitch=st.pitch,
                       yaw=st.yaw, angular_rate_body=st.angular_rate_body)
        s = ref.sample(t / timescale + lead_s)
        cmd = ctrl.command(nav, Setpoint(sim_time_ns=st.sim_time_ns,
                                         position_ned=s.position_ned,
                                         velocity_ned=s.velocity_ned / timescale,
                                         accel_ned=s.accel_ned / timescale ** 2,
                                         yaw=s.yaw))
        if cmd.body_rate is not None and np.linalg.norm(cmd.body_rate) >= 0.99 * 11.5:
            sat_rate += 1
        plant.step(cmd, dt)
        P[k], T[k] = plant.pos, t + dt
    course = json.loads((Path(__file__).resolve().parent.parent
                         / "rl/peregrine_course_diffaero.json").read_text())
    g = gates_ned(course)
    misses = plane_misses(P, T, g)
    lap = misses[-1]["t"] if misses[-1] else None
    ref_xyz = np.stack([np.interp(T / timescale, ref.t, ref.pos[:, i]) for i in range(3)],
                       axis=1)
    track_rmse = float(np.sqrt(np.mean(np.sum((P - ref_xyz) ** 2, axis=1))))
    return {"lap_time_s": lap, "misses": misses,
            "max_miss_m": max((m["miss"] for m in misses if m), default=None),
            "n_crossed": sum(m is not None for m in misses),
            "valid": all(m is not None and m["miss"] < g[i]["half_opening"]
                         for i, m in enumerate(misses)),
            "track_rmse_m": track_rmse, "frac_rate_sat": sat_rate / n_steps,
            "latency_s": latency_s, "timescale": timescale, "lead_s": lead_s,
            "P": P, "T": T}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", default="rl/reference_line_vq1.json")
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--latency", type=float, default=None,
                    help="single cmd_latency_s; default runs {0, 0.02, 0.04}")
    ap.add_argument("--sweep", action="store_true", help="small tracking-gain sweep")
    args = ap.parse_args()

    ref = ReferenceLine.load(args.ref)
    print(f"reference: lap {ref.lap_time_s:.3f}s total {ref.total_duration_s:.3f}s "
          f"({len(ref.t)} samples)\n")

    if args.sweep:
        # Gains x timescale: find the smallest k any gain set flies VALIDLY (latency 20 ms,
        # lead 100 ms -- the middle row of the default scan). Reports the frontier.
        best = None
        for ts in (1.3, 1.4, 1.5, 1.6, 1.7):
            for kp_pos in (2.5, 6.0, 10.0):
                for kd_vel in (3.2, 5.0, 7.0):
                    for kp_att in (10.0, 16.0):
                        r = track(ref, dt=args.dt, latency_s=0.02, timescale=ts, lead_s=0.10,
                                  gains=dict(kp_pos=kp_pos, kd_vel=kd_vel, kp_att=kp_att))
                        if r["valid"]:
                            tag = f"k={ts} kp_pos={kp_pos} kd_vel={kd_vel} kp_att={kp_att}"
                            print(f"  VALID {tag}: lap {r['lap_time_s']:.2f}s "
                                  f"max_miss {r['max_miss_m']:.2f} rmse {r['track_rmse_m']:.2f}")
                            if best is None or r["lap_time_s"] < best[1]["lap_time_s"]:
                                best = (tag, r)
            if best is not None:
                break              # found the frontier timescale; no need to go slower
        print(f"\nBEST: {best[0] if best else 'none valid <= 1.7'}")
        return 0

    latencies = [args.latency] if args.latency is not None else [0.0, 0.02, 0.04]
    for lat in latencies:
        found = None
        for ts in np.arange(1.0, 2.51, 0.05):
            best = None
            for lead in (0.0, 0.05, 0.10):
                r = track(ref, dt=args.dt, latency_s=lat, timescale=float(ts), lead_s=lead)
                if best is None or (r["max_miss_m"] or 9e9) < (best["max_miss_m"] or 9e9):
                    best = r
            r = best
            miss_str = "  ".join(
                ("g%d ----" % i) if m is None else f"g{i} {m['miss']:.2f}m"
                for i, m in enumerate(r["misses"]))
            print(f"[latency {lat * 1000:>4.0f} ms] k={ts:.2f} lead={r['lead_s'] * 1000:.0f}ms lap "
                  f"{(r['lap_time_s'] if r['lap_time_s'] else float('nan')):.2f}s vs ref "
                  f"{ref.lap_time_s:.2f}s | crossed {r['n_crossed']}/6 valid={r['valid']} | "
                  f"rmse {r['track_rmse_m']:.2f}m | rate-sat {100 * r['frac_rate_sat']:.0f}% | {miss_str}")
            if r["valid"]:
                found = r
                break
        if found:
            print(f"  => twin-tracked VALID lap at latency {lat * 1000:.0f} ms: "
                  f"{found['lap_time_s']:.2f}s (timescale {found['timescale']:.2f} of the "
                  f"{ref.lap_time_s:.2f}s ideal)\n")
        else:
            print(f"  => NO valid timescale <= 2.5 at latency {lat * 1000:.0f} ms\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
