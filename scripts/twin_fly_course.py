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


def gate_plane_miss(P: np.ndarray, gate) -> float | None:
    """In-plane (opening) miss where path ``P`` crosses ``gate``'s plane: interpolate each
    through-axis sign change and return the smallest offset within the gate's width/height axes --
    the honest 'how centred through the opening'. Distance-to-CENTRE inflates this along the
    through-axis when a pass is fast (sparse samples) or TRUNCATED (mission.run stops at FINISHED,
    ~gate_pass_radius_m before the last plane), so a dead-centre last gate reads ~the radius. The
    in-plane miss is immune to that. ``None`` if the path never crosses the plane. [2026-06-05]"""
    c = np.asarray(gate.position_ned, dtype=np.float64)
    R = np.asarray(gate.R_world_gate, dtype=np.float64)
    through, w_ax, h_ax = R[:, 2], R[:, 0], R[:, 1]
    d = (P - c) @ through                                  # signed through-offset per sample
    sgn = np.sign(d)
    best = None
    for i in np.flatnonzero(sgn[:-1] * sgn[1:] < 0):       # consecutive samples straddling the plane
        f = d[i] / (d[i] - d[i + 1])                       # linear interp to the crossing
        pc = P[i] + f * (P[i + 1] - P[i])
        miss = float(np.hypot((pc - c) @ w_ax, (pc - c) @ h_ax))
        best = miss if best is None else min(best, miss)
    return best


class TwinTransport:
    """Drives the twin from the mission's command stream (steps the plant per command)."""

    def __init__(self, plant: CtbrPlant, dt: float):
        self.plant, self.dt, self.n = plant, dt, 0

    def send_command(self, cmd) -> None:
        self.plant.step(cmd, self.dt)
        self.n += 1


# Canonical-twin controller defaults (NEUTRAL sim-signs -- the twin is a canonical plant). The
# tuner (scripts/twin_tune.py) overrides the gain subset via make_controller; Task C restores the
# measured sim-signs for the faithful twin. Keeping the baseline here (one source of truth) lets
# the tuner search around it and the test re-derive it.
_CANONICAL_GAINS = dict(
    mode=ControlMode.BODY_RATE, decoupled=True, hover_thrust=0.26,
    kp_pos=1.2, kd_vel=3.0, max_speed=5.0, max_accel_mps2=12.0,
    kp_att=10.0, kd_att=0.30, max_body_rate_rps=8.0, ff_gain=1.0,
    kp_alt=2.0, kd_alt=3.0, alt_thrust_lo=0.05, alt_thrust_hi=0.6, tilt_comp=True,
)


# Live sim-sign compensation MEASURED on ShadowPC (the fly_vq1 gate-0 command + sysid extract): the
# faithful twin's quirks require these so the controller transfers to the sim. body_rate_sign undoes
# the plant's roll/yaw COMMAND inversion; ff_gain undoes the ~2.5x rate amplification; odo_att_sign /
# odo_rate_sign undo the ODOMETRY roll-quat / pitch-rate reporting inversions. Task A uses NEUTRAL
# signs (canonical twin); Task C restores these for the faithful twin. See project_ctbr_control_sysid.
_FAITHFUL_SIGNS = dict(
    body_rate_sign=np.array([1.0, 1.0, -1.0]),     # flight-correct (gate-0 saga): roll re-flipped +1
    odo_att_sign=np.array([-1.0, 1.0, 1.0]),       # ODOMETRY-quat roll inverted
    odo_rate_sign=np.array([-1.0, -1.0, 1.0]),     # ODOMETRY rate: roll + pitch inverted (flight set)
    ff_gain=2.5,                                   # ~ the fitted rate gain (undoes the amplification)
    hover_thrust=0.2656,                           # the fitted faithful-plant hover (plant-matched)
)


def make_controller(*, signs: dict | None = None, **overrides) -> Controller:
    """Build the decoupled CTBR controller, applying gain ``overrides`` (the tuner's search variables,
    e.g. ``kp_pos=2.0``) and a sim-sign set. ``signs=None`` -> NEUTRAL (canonical twin); pass
    ``signs=_FAITHFUL_SIGNS`` (Task C) for the measured live compensation the faithful twin needs.
    ``overrides`` must be :class:`Controller` fields (planner params go to the planner)."""
    params = dict(_CANONICAL_GAINS)                                       # incl. ff_gain=1.0
    params.update(body_rate_sign=np.ones(3), odo_att_sign=np.ones(3), odo_rate_sign=np.ones(3))
    if signs is not None:
        params.update(signs)                                             # faithful signs (override ff_gain)
    params.update(overrides)                                             # explicit gain overrides win last
    return Controller(**params)


def _controller() -> Controller:
    return make_controller()


def fly(n_gates: int = 2, *, dt: float = 0.01, max_s: float = 30.0, velocity_mode: str = "clean",
        controller: Controller | None = None, planner: ReactivePlanner | None = None,
        flythrough_s: float = 0.0, plant_config: CtbrPlantConfig | None = None) -> dict:
    """``velocity_mode``: 'clean' = the fixed client (true world velocity); 'mix' = the ODOMETRY
    frame bug (world/body interleave); 'flip' = pure body-frame velocity (worst case).

    ``controller``/``planner`` override the canonical defaults (the tuning seam: scripts/twin_tune.py
    passes candidate gains here); ``None`` uses the canonical baseline.

    ``flythrough_s`` > 0: after the mission FINISHES, keep RUN-style guidance toward the LAST gate
    for this long so the path crosses its plane -- mission.run stops at FINISHED (~gate_pass_radius_m
    before the last plane), truncating the trajectory short, so ``plane_miss`` for the final gate is
    only meaningful with a continuation. ``t_s``/``steps`` still report the time TO FINISH."""
    gates = load_track_map(_MAP, corner_to_center=True)[:n_gates]
    nav = Navigator(gates=gates, detector=None, config=NavigatorConfig(use_vision=False))
    mission = Mission(
        gates=gates,
        planner=planner if planner is not None
        else ReactivePlanner(cruise_speed=5.0, lookahead_m=3.0, yaw_mode="course"),
        controller=controller if controller is not None else _controller(),
        config=MissionConfig(takeoff_altitude_m=1.5, gate_pass_radius_m=0.75),
    )
    twin = CtbrPlant(plant_config if plant_config is not None else CtbrPlantConfig(rate_tau_s=0.05),
                     position_ned=[0.0, 0.0, 0.0], q_wxyz=_level_quat_wxyz(np.pi))   # pad, facing -X
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
    finish_steps = transport.n                       # time-to-finish, BEFORE any flythrough
    # Fly THROUGH the final gate so its centring is measured like the others (see flythrough_s).
    if flythrough_s > 0.0 and final == MissionState.FINISHED and gates:
        for _ in range(int(flythrough_s / dt)):
            ns = navigator()
            transport.send_command(mission.controller.command(ns, mission.planner.plan(ns, gates[-1])))
    P = np.asarray(traj)
    # closest approach of the flown path to each gate centre (legacy; inflated along-through for a
    # truncated final gate) + the honest in-plane opening miss (None if the plane was never crossed).
    closest = [float(np.min(np.linalg.norm(P - g.position_ned, axis=1))) for g in gates]
    plane_miss = [gate_plane_miss(P, g) for g in gates]
    return {"final": MissionState(final), "gate_index": mission.gate_index, "gates": gates,
            "traj": P, "twin": twin, "closest": closest, "plane_miss": plane_miss,
            "steps": finish_steps, "t_s": finish_steps * dt, "velocity_mode": velocity_mode}


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print("OFFLINE course flight - twin + real Navigator/Mission/Planner/Controller")
    print("(metric = IN-PLANE opening miss at the plane crossing; PASS = < 0.75 m inner half-opening;")
    print(" '----' = plane never crossed. The drone flies THROUGH the final gate so it's measured too.)\n")
    for mode, tag in (("clean", "FIXED client c3b5a8e (true world velocity)"),
                      ("mix", "ODOMETRY frame BUG, realistic world/body interleave"),
                      ("flip", "worst case (pure body-frame velocity)")):
        r = fly(n, velocity_mode=mode, flythrough_s=2.0)
        g = r["gates"]
        cols = "  ".join(
            (f"g{i} ----  MISS" if pm is None
             else f"g{i} {pm:.2f}m {'PASS' if pm < gg.inner_size_m / 2 else 'MISS'}")
            for i, (gg, pm) in enumerate(zip(g, r["plane_miss"]))
        )
        print(f"[{mode:5s}] {tag}")
        print(f"         {r['final'].name}, {r['gate_index']}/{len(g)} gates, {r['t_s']:.1f}s   |   {cols}\n")
    print("=> clean threads the whole course dead-centre; the frame bug threads the near-straight\n"
          "   early gates but MISSES the first real cross-track gate (the lateral loop oscillates)\n"
          "   and stalls -- the saga's 'got close, then oscillate and miss'. Worst case diverges.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
