"""Reproduce the VERIFY rung-1 altitude limit cycle OFFLINE and preview the alt-loop re-tune.

Rung 1 (2026-06-06, live): the faithful controller HELD altitude (~1.40 m, net drift 0) but in a
relay LIMIT CYCLE -- true vz +-0.6 m/s, thrust bang-bang 0.05<->0.40 (68% at the ceiling). Signs /
attitude / lateral / "we own thrust" all transferred cleanly; only the vertical PD oscillates. The
ideal twin couldn't show this because it actuates instantly -- so this harness adds the missing
LATENCY (`CtbrPlantConfig.cmd_latency_s`) and shows: (a) latency=0 -> kp_alt=4 holds (the offline
illusion); (b) latency>0 -> kp_alt=4 relay-oscillates (the live behaviour); (c) lower kp_alt -> holds
even with latency (the fix). The exact latency/hover/slope get FIT from the live recordings; this
validates the MECHANISM + the re-tune direction first.

Usage:  .venv\\Scripts\\python scripts/twin_hover.py
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.spatial.transform import Rotation

from racer.contracts import NavState, Setpoint
from racer.twin import CtbrPlant
from racer.twin_fit import faithful_config
from twin_fly_course import _FAITHFUL_SIGNS, make_controller
from twin_tune import FAITHFUL_TUNED_GAINS


def _level(yaw: float) -> np.ndarray:
    x, y, z, w = Rotation.from_euler("ZYX", [yaw, 0.0, 0.0]).as_quat()
    return np.array([w, x, y, z])


def run_hover(*, cmd_latency_s=0.0, kp_alt=4.0, kd_alt=2.0, alt_thrust_hi=0.40,
              target_alt=1.5, seconds=10.0, rate_hz=100.0) -> dict:
    """Closed-loop hover-hold of the faithful controller vs the faithful twin (+ added latency).
    Returns settle-window metrics (last 5 s)."""
    dt = 1.0 / rate_hz
    plant = CtbrPlant(replace(faithful_config(), cmd_latency_s=cmd_latency_s),
                      position_ned=[0.0, 0.0, 0.0], q_wxyz=_level(np.pi))
    gains = {**FAITHFUL_TUNED_GAINS, "kp_alt": kp_alt, "kd_alt": kd_alt, "alt_thrust_hi": alt_thrust_hi}
    ctrl = make_controller(signs=_FAITHFUL_SIGNS, **gains)
    target = np.array([0.0, 0.0, -target_alt])
    alt, vz, thr = [], [], []
    for _ in range(int(seconds / dt)):
        s = plant.state()
        nav = NavState(sim_time_ns=s.sim_time_ns, position_ned=s.position_ned, velocity_ned=s.velocity_ned,
                       roll=s.roll, pitch=s.pitch, yaw=s.yaw, angular_rate_body=s.angular_rate_body)
        cmd = ctrl.command(nav, Setpoint(sim_time_ns=s.sim_time_ns, position_ned=target, yaw=np.pi))
        plant.step(cmd, dt)
        alt.append(-float(plant.pos[2])); vz.append(-float(plant.vel[2])); thr.append(float(cmd.thrust))
    alt, vz, thr = map(np.asarray, (alt, vz, thr))
    n5 = int(5.0 / dt)
    a, v, t = alt[-n5:], vz[-n5:], thr[-n5:]
    # limit-cycle metrics on the settle window
    clip_frac = float(np.mean((t >= alt_thrust_hi - 1e-3) | (t <= 0.05 + 1e-3)))
    sv = np.sign(v - np.mean(v))
    crossings = int(np.sum(sv[1:] * sv[:-1] < 0))
    period = (2.0 * len(v) * dt / crossings) if crossings else float("inf")
    return {"alt_mean": float(np.mean(a)), "alt_reached": float(np.max(alt)),
            "vz_rms": float(np.sqrt(np.mean(v**2))), "vz_amp": float(np.max(np.abs(v))),
            "thr_clip_frac": clip_frac, "period_s": period,
            "verdict": "HOLDS" if np.sqrt(np.mean(v**2)) < 0.05 else "LIMIT CYCLE"}


def main() -> int:
    lat = 0.03   # placeholder ~3 ticks @100 Hz; the live recording will pin it
    print("Faithful twin hover-hold (target 1.5 m); metrics on the last 5 s.\n")
    print(f"{'case':38s} {'vz_rms':>7} {'vz_amp':>7} {'clip%':>6} {'period':>7}  verdict")
    print("-" * 84)
    cases = [
        ("ideal twin (lat 0)        kp_alt 4.0", dict(cmd_latency_s=0.0, kp_alt=4.0)),
        (f"+lat {lat*1000:.0f}ms      kp_alt 4.0 kd_alt 2.0", dict(cmd_latency_s=lat, kp_alt=4.0, kd_alt=2.0)),
        (f"+lat {lat*1000:.0f}ms      kp_alt 0.8 kd_alt 2.0", dict(cmd_latency_s=lat, kp_alt=0.8, kd_alt=2.0)),
        (f"+lat {lat*1000:.0f}ms      kp_alt 0.4 kd_alt 1.0", dict(cmd_latency_s=lat, kp_alt=0.4, kd_alt=1.0)),
        (f"+lat {lat*1000:.0f}ms      kp_alt 0.3 kd_alt 0.5", dict(cmd_latency_s=lat, kp_alt=0.3, kd_alt=0.5)),
        (f"+lat {lat*1000:.0f}ms      kp_alt 1.5 kd_alt 0.3", dict(cmd_latency_s=lat, kp_alt=1.5, kd_alt=0.3)),
    ]
    for label, kw in cases:
        r = run_hover(**kw)
        print(f"{label:38s} {r['vz_rms']:7.3f} {r['vz_amp']:7.3f} {r['thr_clip_frac']*100:5.0f}% "
              f"{r['period_s']:7.2f}  {r['verdict']}")
    print("\n=> if latency turns kp_alt=4 into a LIMIT CYCLE (matching rung 1) and a lower kp_alt HOLDS,\n"
          "   the mechanism is confirmed; fit cmd_latency_s/hover/slope to the live recordings, then\n"
          "   pick the gain on the FIT twin and re-VERIFY.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
