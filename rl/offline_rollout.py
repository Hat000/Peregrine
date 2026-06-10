"""rl/offline_rollout.py -- closed-loop OFFLINE evaluation of the Stage-1 RL policy.

Rolls the trained actor against ``racer.rl_plant`` (the parity-tested numpy twin of
the training dynamics -- machine-epsilon identical to what the policy was trained on)
on the real 6-gate course, using fly_rl's exact obs builder + action pipeline.
This is the offline-twin discipline applied to RL deployment: if the policy cannot
fly from a given start state HERE, no deployment sign-fix will save it live; if it
can, live failures point at telemetry/timing, not the policy.

Start states:
  trainreset  -- the training reset: 1 m in front of --gate (default 0), at rest,
                 identity Z-up attitude (NED yaw 0).  The policy succeeds from here
                 in training (0.97) => this validates the whole deployment math.
  racestart   -- the REAL race start: NED origin, at rest, yaw pi (facing the
                 course).  The OOD case.
  handoff     -- PATH B seam: --handoff-dist m up-course of gate 0, moving down-
                 course at --handoff-speed m/s, level, yaw pi.

Also runs a build_obs consistency check: synthesizes ODOMETRY-style telemetry
(roll-inverted quat, [-1,-1,1] raw rates) from the true plant state and asserts
fly_rl.build_obs reproduces the truth-path observation.

Usage:
  .venv\\Scripts\\python.exe rl\\offline_rollout.py --start trainreset
  .venv\\Scripts\\python.exe rl\\offline_rollout.py --start racestart --max-rate 1.5
  .venv\\Scripts\\python.exe rl\\offline_rollout.py --start handoff --handoff-speed 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from racer.rl_plant import PlantParams, PlantState, quat_rotate, step as plant_step
from fly_rl import (
    N_GATES, _FLIP, _GATE_POS_ZUP, _R_W2G, _HOVER_THRUST, _ODO_RATE_SIGN,
    _TRAIN_DT, build_obs, load_actor, obs_from_zup, policy_step,
)

_GATE_POS_NED = _GATE_POS_ZUP * _FLIP   # opening centres, NED
_HALF_OPEN = 0.75                       # 1.5 m inner opening, L-inf half-width


def _R_from_quat(q: np.ndarray) -> np.ndarray:
    """R_world_body (NED) from a wxyz quaternion via rl_plant.quat_rotate columns."""
    return np.stack([quat_rotate(q, e) for e in np.eye(3)], axis=-1)


def obs_from_truth(st: PlantState, target_gate: int, last_normed: float,
                   virtual_flip: bool = False) -> np.ndarray:
    """TRUE NED/FRD plant state -> the 17-dim training obs (adapter-faithful path)."""
    R_ned = _R_from_quat(st.quat)
    R_zup = (_FLIP[:, None] * R_ned) * _FLIP[None, :]
    return obs_from_zup(st.pos * _FLIP, st.vel * _FLIP, R_zup,
                        st.omega * _FLIP, target_gate, last_normed,
                        virtual_flip=virtual_flip)


def telemetry_from_truth(st: PlantState) -> SimpleNamespace:
    """Synthesize ODOMETRY-style telemetry (reporting artifacts APPLIED) from truth."""
    from scipy.spatial.transform import Rotation
    from racer.frames import euler_from_quat_wxyz
    roll, pitch, yaw = euler_from_quat_wxyz(st.quat)
    q_rep = Rotation.from_euler("ZYX", [yaw, pitch, -roll]).as_quat()  # xyzw
    q_rep = np.array([q_rep[3], q_rep[0], q_rep[1], q_rep[2]])         # wxyz
    return SimpleNamespace(
        position_ned=st.pos.copy(),
        velocity_ned=st.vel.copy(),
        orientation_ned_wxyz=q_rep,
        angular_rate_body=st.omega * _ODO_RATE_SIGN,   # involutory: true -> raw
    )


def check_build_obs(rng: np.random.Generator) -> float:
    """Truth-path obs vs telemetry-path obs over random states; returns max |diff|."""
    worst = 0.0
    for _ in range(200):
        ang = rng.uniform([-3.1, -1.2, -np.pi], [3.1, 1.2, np.pi])  # roll,pitch,yaw
        from scipy.spatial.transform import Rotation
        qx = Rotation.from_euler("ZYX", [ang[2], ang[1], ang[0]]).as_quat()
        q = np.array([qx[3], qx[0], qx[1], qx[2]])
        st = PlantState(
            pos=rng.uniform(-50, 10, 3), vel=rng.uniform(-20, 20, 3),
            quat=q, omega=rng.uniform(-6, 6, 3),
            thrust=np.float64(0.3),
        )
        g = int(rng.integers(0, N_GATES))
        ln = float(rng.uniform(0, 5))
        a = obs_from_truth(st, g, ln)
        b = build_obs(telemetry_from_truth(st), g, ln)
        worst = max(worst, float(np.max(np.abs(a - b))))
    return worst


def gate_event(prev_ned: np.ndarray, cur_ned: np.ndarray, gate: int) -> str | None:
    """'pass' | 'collision' | None -- plane crossing on the gate +X axis (Z-up frame),
    L-inf 0.75 m aperture; mirrors PeregrineRacing.is_passed."""
    prev_rel = _R_W2G @ (prev_ned * _FLIP - _GATE_POS_ZUP[gate])
    cur_rel  = _R_W2G @ (cur_ned * _FLIP - _GATE_POS_ZUP[gate])
    if prev_rel[0] < 0.0 and cur_rel[0] > 0.0:
        inside = (abs(cur_rel[1]) < _HALF_OPEN) and (abs(cur_rel[2]) < _HALF_OPEN)
        return "pass" if inside else "collision"
    return None


def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    qx = Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_quat()
    return np.array([qx[3], qx[0], qx[1], qx[2]])


def make_start(kind: str, args) -> tuple[PlantState, int]:
    if kind == "trainreset":
        g = args.gate
        pos_ned = (_GATE_POS_ZUP[g] + np.array([1.0, 0.0, 0.0])) * _FLIP
        quat = np.array([1.0, 0.0, 0.0, 0.0])          # identity Z-up == identity NED
        vel = np.zeros(3)
        target = g
    elif kind == "racestart":
        pos_ned = np.zeros(3)
        quat = np.array([0.0, 0.0, 0.0, 1.0])          # NED yaw = pi (faces course, -X)
        vel = np.zeros(3)
        target = 0
    elif kind == "simstart":
        # TRUE spawn measured from race recordings (first ODOMETRY at GO):
        # pos (0,0,+0.02) NED, true rpy = (0, -17.8 deg, -179.9 deg) (tilted pad).
        pos_ned = np.array([0.0, 0.0, 0.02])
        quat = _quat_from_rpy(0.0, np.radians(-17.8), np.radians(-179.9))
        vel = np.zeros(3)
        target = 0
    elif kind == "handoff":
        pos_ned = _GATE_POS_NED[0] + np.array([args.handoff_dist, 0.0, 0.0])
        quat = np.array([0.0, 0.0, 0.0, 1.0])
        vel = np.array([-args.handoff_speed, 0.0, 0.0])  # down-course
        target = 0
    else:
        raise SystemExit(f"unknown start '{kind}'")
    thrust = np.float64(_HOVER_THRUST if args.thrust0 < 0 else args.thrust0)
    return PlantState(pos=pos_ned, vel=vel, quat=quat,
                      omega=np.zeros(3), thrust=thrust), target


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=r"C:\Users\Shadow\Downloads\stage1_inc1_actor.pth")
    ap.add_argument("--start", default="simstart",
                    choices=["trainreset", "racestart", "simstart", "handoff"])
    ap.add_argument("--gate", type=int, default=0, help="trainreset: which gate")
    ap.add_argument("--handoff-dist",  type=float, default=3.0)
    ap.add_argument("--handoff-speed", type=float, default=10.0)
    ap.add_argument("--max-rate",   type=float, default=0.0,
                    help="PATH A: cap |rate_flu| rad/s before the plant; 0=off")
    ap.add_argument("--virtual-flip", action="store_true",
                    help="π body-z conjugation (policy flies tail-first; sim spawns "
                         "nose-first — maps the spawn into distribution)")
    ap.add_argument("--max-thrust", type=float, default=0.0,
                    help="cap normed_thrust (training units, hover=1); 0=off")
    ap.add_argument("--cap-gates", type=int, default=99,
                    help="apply --max-rate/--max-thrust only while target gate < N")
    ap.add_argument("--latency-steps", type=int, default=0,
                    help="plant transport delay in control steps (live ~40 ms ≈ 1)")
    ap.add_argument("--thrust0",    type=float, default=-1.0,
                    help="initial realized collective state; <0 = hover (adapter reset)")
    ap.add_argument("--live-thrust-clip", action="store_true",
                    help="clip collective to [0,1] like the live sim (training did NOT)")
    ap.add_argument("--max-time",   type=float, default=40.0)
    ap.add_argument("--dt",         type=float, default=_TRAIN_DT)
    ap.add_argument("--trace",      default="", help="optional .npz trace dump path")
    ap.add_argument("--skip-check", action="store_true")
    args = ap.parse_args()

    actor = load_actor(args.checkpoint)

    if not args.skip_check:
        worst = check_build_obs(np.random.default_rng(0))
        tag = "OK" if worst < 1e-5 else "FAIL"
        print(f"[check] build_obs telemetry-path vs truth-path: max|diff|={worst:.2e}  {tag}")
        if worst >= 1e-5:
            return 1

    params = PlantParams(transport_delay_steps=args.latency_steps)  # faithful plant, g=9.80665
    st, gate = make_start(args.start, args)
    print(f"[start] {args.start}: pos_ned={np.round(st.pos,2).tolist()} "
          f"vel={np.round(st.vel,2).tolist()} target_gate={gate} thrust0={float(st.thrust):.3f} "
          f"max_rate={args.max_rate or 'off'} live_clip={args.live_thrust_clip} "
          f"vflip={args.virtual_flip}")

    last_normed = 0.0             # training reset: last_action = 0
    n_steps = int(round(args.max_time / args.dt))
    passed: list[int] = []
    outcome = "TIMEOUT"
    log = {"t": [], "pos": [], "vel": [], "act": []}

    obs0 = obs_from_truth(st, gate, last_normed, args.virtual_flip)
    r0, c0, n0 = policy_step(actor, obs0, args.max_rate, args.virtual_flip,
                             args.max_thrust)
    print(f"[step0] obs pos_g={np.round(obs0[0:3],2).tolist()} vel_g={np.round(obs0[3:6],2).tolist()} "
          f"rpy_g={np.round(obs0[6:9],2).tolist()}")
    print(f"[step0] action: rate_frd={np.round(r0,3).tolist()} collective={c0:.3f} "
          f"normed_thrust={n0:.3f}")

    for k in range(n_steps):
        capped = gate < args.cap_gates
        obs = obs_from_truth(st, gate, last_normed, args.virtual_flip)
        rate_frd, collective, last_normed = policy_step(
            actor, obs, args.max_rate if capped else 0.0, args.virtual_flip,
            args.max_thrust if capped else 0.0)
        if not args.live_thrust_clip:
            collective = last_normed * _HOVER_THRUST   # un-clipped, exactly training
        action = np.concatenate([rate_frd, [collective]])

        prev_pos = st.pos.copy()
        st = plant_step(st, action, args.dt, params)
        t = (k + 1) * args.dt

        log["t"].append(t); log["pos"].append(st.pos.copy())
        log["vel"].append(st.vel.copy()); log["act"].append(action)

        ev = gate_event(prev_pos, st.pos, gate)
        if ev == "pass":
            passed.append(gate)
            print(f"  t={t:6.2f}s  gate {gate} PASS   speed={np.linalg.norm(st.vel):5.1f} m/s "
                  f"pos={np.round(st.pos,1).tolist()}")
            if gate == N_GATES - 1:
                outcome = "FINISHED"
                break
            gate += 1
        elif ev == "collision":
            rel = _R_W2G @ (st.pos * _FLIP - _GATE_POS_ZUP[gate])
            print(f"  t={t:6.2f}s  gate {gate} COLLISION  off=[{rel[1]:+.2f},{rel[2]:+.2f}] m "
                  f"speed={np.linalg.norm(st.vel):5.1f}")
            outcome = "COLLISION"
            break

        # out-of-bounds box (training: course bbox + 15/12 m margins)
        zup = st.pos * _FLIP
        lo = _GATE_POS_ZUP.min(0) - [15, 15, 12]
        hi = _GATE_POS_ZUP.max(0) + [15, 15, 12]
        # the start pad (x_zup ~ 0) is outside the gate bbox on x; allow it
        hi[0] = max(hi[0], 5.0)
        if np.any(zup < lo) or np.any(zup > hi):
            print(f"  t={t:6.2f}s  OUT OF BOUNDS  pos_ned={np.round(st.pos,1).tolist()} "
                  f"speed={np.linalg.norm(st.vel):5.1f}")
            outcome = "OOB"
            break

    vmax = float(np.max(np.linalg.norm(np.asarray(log["vel"]), axis=1))) if log["vel"] else 0.0
    print(f"\n[outcome] {outcome}   gates passed: {len(passed)}/{N_GATES} {passed}   "
          f"t={log['t'][-1] if log['t'] else 0:.2f}s   vmax={vmax:.1f} m/s")
    if args.trace:
        np.savez(args.trace, **{k: np.asarray(v) for k, v in log.items()})
        print(f"trace -> {args.trace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
