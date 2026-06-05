"""Fly the course OFFLINE: the REAL Navigator + Mission + Planner + Controller against the plant
twin (``racer.twin``) — a full VQ1 dry-run with no sim and no hardware. This is the offline tuning
surface for CTBR (roadmap C): iterate gains here, deterministically, in seconds; fly the live sim
only to VERIFY. Velocity feedback is the FIXED client's world velocity (post c3b5a8e).

Uses the captured deterministic gate map. The twin is CANONICAL (a +roll command rolls right), so
the controller runs with NEUTRAL sim-sign compensation; for a SIM-FAITHFUL run set the twin's
``rate_gain``/``rate_sign`` to the measured values and restore the controller's sim signs.

Usage:  python scripts/twin_fly_course.py [n_gates]
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from scipy.spatial.transform import Rotation

from racer.contracts import ControlMode
from racer.controller import Controller
from racer.mission import Mission, MissionConfig, MissionState
from racer.navigator import Navigator, NavigatorConfig, load_track_map
from racer.planner import ReactivePlanner
from racer.twin import CtbrPlant, CtbrPlantConfig

_MAP = Path(__file__).resolve().parent.parent / "handoff/shadowpc-firstcontact-2026-06-02/track_map.json"


def _level_quat_wxyz(yaw: float) -> np.ndarray:
    x, y, z, w = Rotation.from_euler("ZYX", [yaw, 0.0, 0.0]).as_quat()
    return np.array([w, x, y, z], dtype=np.float64)


class TwinTransport:
    """Drives the twin from the mission's command stream (steps the plant per command)."""

    def __init__(self, plant: CtbrPlant, dt: float):
        self.plant, self.dt, self.n = plant, dt, 0

    def send_command(self, cmd) -> None:
        self.plant.step(cmd, self.dt)
        self.n += 1


def _controller() -> Controller:
    return Controller(
        mode=ControlMode.BODY_RATE, decoupled=True, hover_thrust=0.26,
        kp_pos=1.2, kd_vel=3.0, max_speed=5.0, max_accel_mps2=12.0,
        kp_att=10.0, kd_att=0.30, max_body_rate_rps=8.0, ff_gain=1.0,
        kp_alt=2.0, kd_alt=3.0, alt_thrust_lo=0.05, alt_thrust_hi=0.6, tilt_comp=True,
        body_rate_sign=np.ones(3), odo_att_sign=np.ones(3), odo_rate_sign=np.ones(3),
    )


def fly(n_gates: int = 2, *, dt: float = 0.01, max_s: float = 30.0, velocity_mode: str = "clean") -> dict:
    """``velocity_mode``: 'clean' = the fixed client (true world velocity); 'mix' = the ODOMETRY
    frame bug (world/body interleave); 'flip' = pure body-frame velocity (worst case)."""
    gates = load_track_map(_MAP, corner_to_center=True)[:n_gates]
    nav = Navigator(gates=gates, detector=None, config=NavigatorConfig(use_vision=False))
    mission = Mission(
        gates=gates,
        planner=ReactivePlanner(cruise_speed=5.0, lookahead_m=3.0, yaw_mode="course"),
        controller=_controller(),
        config=MissionConfig(takeoff_altitude_m=1.5, gate_pass_radius_m=0.75),
    )
    twin = CtbrPlant(CtbrPlantConfig(rate_tau_s=0.05), position_ned=[0.0, 0.0, 0.0],
                     q_wxyz=_level_quat_wxyz(np.pi))                 # at the pad, facing -X
    transport = TwinTransport(twin, dt)
    traj: list[np.ndarray] = []
    tick = [0]

    def navigator():
        st = twin.state()
        if velocity_mode != "clean":            # simulate the pre-fix ODOMETRY body-frame velocity
            R = Rotation.from_quat([st.orientation_ned_wxyz[1], st.orientation_ned_wxyz[2],
                                    st.orientation_ned_wxyz[3], st.orientation_ned_wxyz[0]]).as_matrix()
            flip = velocity_mode == "flip" or (tick[0] % 9) < 4    # interleave ~44% body samples
            st = replace(st, velocity_ned=(R.T @ st.velocity_ned if flip else st.velocity_ned))
        tick[0] += 1
        ns = nav.update(st, None)
        traj.append(twin.pos.copy())
        return ns

    final = mission.run(navigator, transport, max_steps=int(max_s / dt))
    P = np.asarray(traj)
    # closest approach of the flown path to each gate centre
    closest = [float(np.min(np.linalg.norm(P - g.position_ned, axis=1))) for g in gates]
    return {"final": MissionState(final), "gate_index": mission.gate_index, "gates": gates,
            "traj": P, "twin": twin, "closest": closest, "steps": transport.n,
            "t_s": transport.n * dt, "velocity_mode": velocity_mode}


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print("OFFLINE course flight - twin + real Navigator/Mission/Planner/Controller")
    print("(inner half-opening = 0.75 m; closest approach < 0.75 m = flew through the opening)\n")
    for mode, tag in (("clean", "FIXED client c3b5a8e (true world velocity)"),
                      ("mix", "ODOMETRY frame BUG, realistic world/body interleave"),
                      ("flip", "worst case (pure body-frame velocity)")):
        r = fly(n, velocity_mode=mode)
        g = r["gates"]
        cols = "  ".join(
            f"g{i} {r['closest'][i]:.2f}m {'PASS' if r['closest'][i] < gg.inner_size_m / 2 else 'MISS'}"
            for i, gg in enumerate(g)
        )
        print(f"[{mode:5s}] {tag}")
        print(f"         {r['final'].name}, {r['gate_index']}/{len(g)} gates, {r['t_s']:.1f}s   |   {cols}\n")
    print("=> clean threads the whole course dead-centre; the frame bug threads the near-straight\n"
          "   early gates but MISSES the first real cross-track gate (the lateral loop oscillates)\n"
          "   and stalls -- the saga's 'got close, then oscillate and miss'. Worst case diverges.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
