"""P7 integrated test: replay recorded command streams through twin variants, compare.

Variants:
  legacy    faithful_config(super_rate=False)  -- the pre-sweep twin
  map_on    faithful_config(super_rate=True)   -- current best (what RL will train on)
  candidate map_on + tonight's aero model:
              horizontal drag  -c2h*|v_h|*v_h        (c2h = 0.052 /m, isotropic, world frame)
              vertical drag    -c_up/-c_dn *|vz|*vz  (0.0756 climb / 0.0539 descend)
              collective map   K(thr) piecewise knots (vert_fit_coef.npy) instead of g*thr/hover

Metric per run (twin_fit.validate style): open-loop attitude RMS, velocity RMS, speed RMS,
final position error; one-step velocity RMS. Runs: turn20 (mixed-axis) + drag_back25 +
drag_lat17p (the drag-dominated traces).

Usage: .venv\\Scripts\\python.exe handoff\\shadowpc-twin-falsify-2026-06-10\\replay_twin.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))

import runs as R
from racer.contracts import ControlCommand, ControlMode
from racer.frames import euler_from_quat_wxyz
from racer.twin import CtbrPlant, _wxyz_from_euler
from racer.twin_fit import faithful_config

VF = np.load(Path(__file__).parent / "vert_fit_coef.npy", allow_pickle=True).item()
C2H = 0.052          # pure-quad pooled coast fit
D1H_MIX = 0.0696     # mixed-form pooled coast fit (fit_aero.py)
C2H_MIX = 0.0374


class CandidatePlant(CtbrPlant):
    """CtbrPlant with the measured aero: quad drag + knot collective map. Only the
    translational force model changes; the rate loop (super-rate map) is inherited.
    Class attrs d1h/c2h pick the horizontal drag form (pure quad vs mixed)."""

    d1h = 0.0
    c2h = C2H

    def step(self, cmd, dt: float) -> None:
        if dt <= 0.0:
            return
        cfg = self.cfg
        cmd_rate = np.zeros(3) if cmd.body_rate is None else np.asarray(cmd.body_rate, float)
        gain = np.asarray(cfg.rate_gain)
        if cfg.super_rate_s is not None:
            s = np.asarray(cfg.super_rate_s, dtype=np.float64)
            gain = gain / (1.0 - s * np.minimum(np.abs(cmd_rate), np.pi) / np.pi)
        target = gain * np.asarray(cfg.rate_sign) * cmd_rate
        alpha = 1.0 - np.exp(-dt / max(cfg.rate_tau_s, 1e-9))
        domega = alpha * (target - self.omega)
        if cfg.alpha_max_rps2 is not None:
            lim = np.asarray(cfg.alpha_max_rps2, dtype=np.float64) * dt
            domega = np.clip(domega, -lim, lim)
        from racer.twin import _clip_norm
        self.omega = _clip_norm(self.omega + domega, cfg.max_omega_rps)
        R_cur = self._from_quat(self.q)
        from scipy.spatial.transform import Rotation
        R_new = R_cur * Rotation.from_rotvec(self.omega * dt)
        self.q = self._to_wxyz(R_new)
        thrust_cmd = 0.0 if cmd.thrust is None else float(cmd.thrust)
        self._thrust = thrust_cmd
        # measured collective map (knots) instead of g*thr/hover
        a_up = float(np.interp(self._thrust, VF["knots"], VF["K"]))
        f_world = R_new.as_matrix() @ np.array([0.0, 0.0, -a_up])
        # measured drag: horizontal isotropic (d1 + c2 form), vertical directional quad
        vh = np.hypot(self.vel[0], self.vel[1])
        f_world[0] -= (self.d1h + self.c2h * vh) * self.vel[0]
        f_world[1] -= (self.d1h + self.c2h * vh) * self.vel[1]
        cz = VF["c_dn"] if self.vel[2] > 0 else VF["c_up"]
        f_world[2] -= float(cz) * abs(self.vel[2]) * self.vel[2]
        accel = f_world + np.array([0.0, 0.0, cfg.g])
        self.vel = self.vel + accel * dt
        self.pos = self.pos + self.vel * dt
        self.accel_body = R_new.as_matrix().T @ f_world
        self.t_ns += int(round(dt * 1e9))


def seed_q(rpy_reported, att_sign=np.array([-1.0, 1.0, 1.0])):
    r, p, y = rpy_reported * att_sign
    return _wxyz_from_euler(r, p, y)


def replay(label: str, plant_cls, cfg) -> dict:
    run = R.load(label)
    # start where the drone first leaves the pad with a live command
    k0 = int(np.argmax(run.pos[:, 2] < -0.2))
    plant = plant_cls(cfg, position_ned=run.pos[k0], velocity_ned=run.vel[k0],
                      q_wxyz=seed_q(run.rpy[k0]))
    att_e, vel_e = [], []
    vel1 = []
    for k in range(k0, len(run.t) - 1):
        dt = run.t[k + 1] - run.t[k]
        cmd = ControlCommand(mode=ControlMode.BODY_RATE, body_rate=run.cmd[k],
                             thrust=float(run.thr[k]))
        plant.step(cmd, dt)
        st = plant.state()
        att_e.append((np.array([st.roll, st.pitch, st.yaw]) - run.rpy[k + 1] + np.pi)
                     % (2 * np.pi) - np.pi)
        vel_e.append(st.velocity_ned - run.vel[k + 1])
        # one-step
        p1 = plant_cls(cfg, position_ned=run.pos[k], velocity_ned=run.vel[k],
                       q_wxyz=seed_q(run.rpy[k]))
        p1.step(cmd, dt)
        vel1.append(p1.state().velocity_ned - run.vel[k + 1])
    att_e, vel_e, vel1 = map(np.asarray, (att_e, vel_e, vel1))
    sp_rms = float(np.sqrt(np.mean(np.sum(vel_e ** 2, axis=1))))
    pos_fin = float(np.linalg.norm(plant.pos - run.pos[len(run.t) - 1]))
    return {"n": len(att_e),
            "att_rms_deg": np.degrees(np.sqrt(np.mean(att_e ** 2, axis=0))),
            "speed_rms": sp_rms,
            "step_vel_rms": float(np.sqrt(np.mean(np.sum(vel1 ** 2, axis=1)))),
            "pos_final_err": pos_fin}


class CandidateMixed(CandidatePlant):
    d1h = D1H_MIX
    c2h = C2H_MIX


def main():
    variants = [
        ("legacy", CtbrPlant, faithful_config(super_rate=False)),
        ("map_on", CtbrPlant, faithful_config(super_rate=True)),
        ("cand_quad", CandidatePlant, faithful_config(super_rate=True)),
        ("cand_mix", CandidateMixed, faithful_config(super_rate=True)),
    ]
    for lb in ("turn20", "drag_back25", "drag_lat17p", "coll_speed"):
        print(f"\n== open-loop command replay: {lb} ==")
        print(f"{'variant':10s} {'att rms r/p/y (deg)':>24s} {'speed rms':>9s} "
              f"{'1-step vel':>10s} {'final pos err':>13s}")
        for name, cls, cfg in variants:
            v = replay(lb, cls, cfg)
            a = v["att_rms_deg"]
            print(f"{name:10s} {a[0]:7.2f} {a[1]:7.2f} {a[2]:7.2f}  {v['speed_rms']:9.3f} "
                  f"{v['step_vel_rms']:10.4f} {v['pos_final_err']:13.2f}")


if __name__ == "__main__":
    main()
