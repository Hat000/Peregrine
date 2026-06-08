"""Reproduce the gate-0 low-frequency lateral oscillation OFFLINE, on the plant twin.

Runs the REAL decoupled CTBR controller (``racer.controller``) against the canonical plant twin
(``racer.twin``) on a gate-0-style cross-track scenario: the drone starts 1 m off the gate axis at
the -180 deg course heading and must correct to centre while flying down-course. We feed the
controller velocity feedback three ways:

  clean  : the twin's true world velocity (== the FIXED mavlink_client, commit c3b5a8e)
  mix    : the world/body interleave the BUG produced (LOCAL_POSITION world + ODOMETRY body, the
           latter sign-flipped at -180 deg) -> weak/erratic cross-track damping
  flip   : the worst case, pure body-frame velocity (vx/vy fully sign-flipped) -> anti-damping

At yaw=-180 deg the body->world rotation flips vx/vy, so the buggy ``velocity_ned`` fed the
controller's cross-track damping a wrong-signed lateral velocity -> the low-frequency left<->right
oscillation the teammate saw (and the saga's "lateral oscillation"). With clean feedback the same
controller/gains converge smoothly. This is the MECHANISM on a canonical plant; the real sim adds
separate quirks (the §1 ODOMETRY-quat roll-sign) on top.

Usage:  python scripts/twin_oscillation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from scipy.spatial.transform import Rotation

from racer.contracts import ControlMode, NavState, Setpoint
from racer.controller import Controller
from racer.twin import CtbrPlant, CtbrPlantConfig


def _level_quat_wxyz(yaw: float) -> np.ndarray:
    x, y, z, w = Rotation.from_euler("ZYX", [yaw, 0.0, 0.0]).as_quat()
    return np.array([w, x, y, z], dtype=np.float64)


def _controller() -> Controller:
    # Decoupled CTBR with NEUTRAL sim-sign compensation (the twin is canonical), gains chosen so the
    # clean run is well damped. The only thing we vary across runs is the velocity FEEDBACK frame.
    return Controller(
        mode=ControlMode.BODY_RATE,
        decoupled=True,
        hover_thrust=0.26,
        kp_pos=2.0, kd_vel=3.5, max_speed=5.0, max_accel_mps2=10.0,
        kp_att=10.0, kd_att=0.30, max_body_rate_rps=8.0, ff_gain=1.0,
        kp_alt=5.0, kd_alt=4.0, alt_thrust_lo=0.0, alt_thrust_hi=0.7, tilt_comp=True,
        # neutral: the twin needs no sign inversion (canonical plant)
        body_rate_sign=np.ones(3), odo_att_sign=np.ones(3), odo_rate_sign=np.ones(3),
    )


def _corrupt(v_world: np.ndarray, q_wxyz: np.ndarray, mode: str, i: int) -> np.ndarray:
    """Return the velocity the controller SEES, given the true world velocity + attitude."""
    if mode == "clean":
        return v_world
    R = Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
    v_body = R.T @ v_world                       # ODOMETRY twist: body-frame, mislabelled as world
    if mode == "flip":
        return v_body
    if mode == "mix":
        # interleave: ODOMETRY (~44% of samples) writes body; LOCAL_POSITION writes world.
        return v_body if (i % 9) < 4 else v_world
    raise ValueError(mode)


def run_scenario(vmode: str, *, seconds: float = 8.0, dt: float = 0.01) -> dict:
    yaw = np.pi                                  # face -X (down-course); == -180 deg
    z0 = -1.4                                    # 1.4 m up (gate-0 opening centre)
    plant = CtbrPlant(CtbrPlantConfig(rate_tau_s=0.05),
                      position_ned=[0.0, 2.0, z0],     # 2 m EAST of the gate axis (cross-track)
                      velocity_ned=[0.0, 0.0, 0.0], q_wxyz=_level_quat_wxyz(yaw))
    ctrl = _controller()
    ys, xs, vys = [], [], []
    n = int(round(seconds / dt))
    for i in range(n):
        st = plant.state()
        v_fed = _corrupt(st.velocity_ned, st.orientation_ned_wxyz, vmode, i)
        nav = NavState(sim_time_ns=st.sim_time_ns, position_ned=st.position_ned, velocity_ned=v_fed,
                       roll=st.roll, pitch=st.pitch, yaw=st.yaw, angular_rate_body=st.angular_rate_body)
        carrot = np.array([st.position_ned[0] - 4.0, 0.0, z0])     # pure-pursuit, gate axis at y=0
        sp = Setpoint(sim_time_ns=st.sim_time_ns, position_ned=carrot, yaw=yaw)
        plant.step(ctrl.command(nav, sp), dt)
        ys.append(float(plant.pos[1])); xs.append(float(plant.pos[0])); vys.append(float(plant.vel[1]))
        if abs(plant.pos[1]) > 15.0:             # diverged
            break
    ys = np.asarray(ys)
    t = np.arange(len(ys)) * dt
    diverged = bool(abs(ys[-1]) > 15.0)
    late = ys[t >= (t[-1] - 2.0)] if len(ys) else ys      # last 2 s
    late_rms = float(np.sqrt(np.mean(late ** 2))) if len(late) else float("nan")
    over = np.flatnonzero(np.abs(ys) > 0.2)               # settle = last exit from the +-0.2 m band
    settle_s = (0.0 if over.size == 0 else
                float("inf") if over[-1] >= len(ys) - 1 else float(t[over[-1]]))
    warm = int(1.0 / dt)                                  # y-axis crossings (overshoots) after warmup
    sy = np.sign(ys[warm:])
    crossings = int(np.sum(sy[1:] * sy[:-1] < 0))
    return {"y": ys, "x": np.asarray(xs), "settle_s": settle_s, "late_rms": late_rms,
            "crossings": crossings, "diverged": diverged}


def main() -> int:
    print("Gate-0 cross-track scenario: start 2 m off-axis at yaw=-180 deg, correct to centre.\n")
    print(f"{'feedback':10s} {'settle s':>9s} {'late RMS m':>11s} {'osc':>5s}  verdict")
    print("-" * 58)
    for mode in ("clean", "mix", "flip"):
        r = run_scenario(mode)
        if r["diverged"]:
            verdict = "DIVERGED (anti-damped)"
        elif r["late_rms"] > 0.20:
            verdict = "OSCILLATES (won't settle)"
        elif r["settle_s"] > 3.0 or r["late_rms"] > 0.05:
            verdict = "slow oscillation"
        else:
            verdict = "converges smoothly"
        settle = "   inf" if r["settle_s"] == float("inf") else f"{r['settle_s']:6.2f}"
        print(f"{mode:10s} {settle:>9s} {r['late_rms']:11.3f} {r['crossings']:5d}  {verdict}")
    print("\nclean = the FIXED client (c3b5a8e); mix = the bug's world/body interleave; "
          "flip = worst case.\nThe frame fix turns the oscillation into a smooth correction "
          "with the SAME controller + gains.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
