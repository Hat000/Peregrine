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
(R_y(pi)-conjugated quat, all-axes-negated rates -- FRAME-AUDIT 2026-06-12) from
the true plant state and asserts fly_rl.build_obs reproduces the truth-path
observation.

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

from racer.rl_plant import (ALPHA_MAX_RPS2_MEASURED, PlantParams, PlantState,
                            SUPER_RATE_S_MEASURED, QUAD_DRAG_C2_MEASURED,
                            COLL_MAP_THR_MEASURED, COLL_MAP_ACCEL_MEASURED,
                            LAPSE_SPEED_MEASURED, LAPSE_FACTOR_MEASURED,
                            MIXER_IDLE_MEASURED, MIXER_KAPPA_ERR_MEASURED,
                            MIXER_KAPPA_HOLD_MEASURED, MIXER_ZETA_YAW_MEASURED,
                            quat_rotate, step as plant_step)
from fly_rl import (
    GateMap, N_GATES, _FLIP, _GATE_POS_ZUP, _R_W2G, _HOVER_THRUST, _ODO_RATE_SIGN,
    _TRAIN_DT, _gate_rotmat_w2g, build_obs, load_actor, make_gate_map, obs_from_zup,
    policy_step,
)

_GATE_POS_NED = _GATE_POS_ZUP * _FLIP   # opening centres, NED

# The LIVE sim's TRUE command->rate signs (FRAME-AUDIT 2026-06-12): NO inversion
# on any axis. Open-loop replay of recorded wire commands through rate_sign
# [+1,+1,+1] reproduces the live TRUE attitude trajectory and the correctly
# SIGNED East velocity (the bcc93f9 [-1,+1,-1] was the same physics read through
# the conjugated telemetry frame -- a self-consistent mirror that flipped the
# lateral thrust projection). fly_rl's wire map [+1,-1,+1] composed with this
# reproduces the training-plant closed loop ([1,-1,-1] adapter on [+1,+1,-1])
# exactly.
_RATE_SIGN_LIVE = np.array([1.0, 1.0, 1.0])
_HALF_OPEN = 0.75                       # 1.5 m inner opening, L-inf half-width
_HALF_OUTER = 1.36                      # 2.72 m outer frame, L-inf half-width (S1.4 geometry)


def _R_from_quat(q: np.ndarray) -> np.ndarray:
    """R_world_body (NED) from a wxyz quaternion via rl_plant.quat_rotate columns."""
    return np.stack([quat_rotate(q, e) for e in np.eye(3)], axis=-1)


def obs_from_truth(st: PlantState, target_gate: int, last_normed: float,
                   virtual_flip: bool = False, gate_map: GateMap | None = None) -> np.ndarray:
    """TRUE NED/FRD plant state -> the 17-dim training obs (adapter-faithful path).
    ``gate_map`` threads the runtime per-gate yaw through (P4-C05); None = bit-exact VQ1."""
    R_ned = _R_from_quat(st.quat)
    R_zup = (_FLIP[:, None] * R_ned) * _FLIP[None, :]
    return obs_from_zup(st.pos * _FLIP, st.vel * _FLIP, R_zup,
                        st.omega * _FLIP, target_gate, last_normed,
                        virtual_flip=virtual_flip, gate_map=gate_map)


def telemetry_from_truth(st: PlantState) -> SimpleNamespace:
    """Synthesize ODOMETRY-style telemetry (reporting artifacts APPLIED) from truth.
    FRAME-AUDIT convention (2026-06-12): the quat is reported in the R_y(pi)-
    conjugated telemetry frame (negate x,z components) and the rate channel is the
    all-axes-negated true body rate. Both transforms are involutory, so applying
    fly_rl's undo constants here synthesizes them exactly."""
    from fly_rl import _ODO_QUAT_TRUE_CONJ
    return SimpleNamespace(
        position_ned=st.pos.copy(),
        velocity_ned=st.vel.copy(),
        orientation_ned_wxyz=st.quat * _ODO_QUAT_TRUE_CONJ,   # involutory: true -> raw
        angular_rate_body=st.omega * _ODO_RATE_SIGN,          # [-1,-1,-1]: true -> raw
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


def slab_frame_hit_np(prev_rel: np.ndarray, cur_rel: np.ndarray, half_in: float,
                      half_out: float, depth: float) -> bool:
    """Numpy scalar mirror of peregrine_racing.slab_frame_hits (INC7 contact-true geometry):
    does the straight segment prev_rel -> cur_rel (gate frame) touch frame material
    {|x| <= depth, half_in <= Linf(y,z) <= half_out}? Exact via the convexity of Linf(t)
    (cross-validated against the torch helper in tests/test_contact_geometry.py)."""
    px, py, pz = prev_rel
    dx, dy, dz = cur_rel - prev_rel
    if abs(dx) < 1e-12:
        if abs(px) > depth:
            return False
        t0, t1 = 0.0, 1.0
    else:
        ta, tb = (-depth - px) / dx, (depth - px) / dx
        t0, t1 = max(min(ta, tb), 0.0), min(max(ta, tb), 1.0)
        if t1 < t0:
            return False

    def linf(t):
        return max(abs(py + t * dy), abs(pz + t * dz))

    cands = [t0, t1]
    for num, den in ((-py, dy), (-pz, dz), (pz - py, dy - dz), (-(py + pz), dy + dz)):
        if abs(den) > 1e-12:
            cands.append(min(max(num / den, t0), t1))
    lmin = min(linf(t) for t in cands)
    lmax = max(linf(t0), linf(t1))
    return lmin <= half_out and lmax >= half_in


def _w2g_for(gate: int, gate_yaw: np.ndarray | None) -> np.ndarray:
    """world->gate rotation for ``gate``: the EXACT yaw=π diag (_R_W2G, bit-exact VQ1) when
    ``gate_yaw`` is None, else the per-gate yaw-aware rotation (P4-C05)."""
    return _R_W2G if gate_yaw is None else _gate_rotmat_w2g(float(gate_yaw[gate]))


def gate_event(prev_ned: np.ndarray, cur_ned: np.ndarray, gate: int,
               body_radius: float = 0.0, frame_depth: float = 0.0,
               gate_pos_zup: np.ndarray | None = None,
               gate_yaw: np.ndarray | None = None) -> str | None:
    """'pass' | 'collision' | 'miss' | None for the TARGET gate -- the S1.4 classification
    (mirrors peregrine_racing.crossing_events): the plane crossing is INTERPOLATED to the
    crossing point; L-inf < 0.75 m = pass, in (0.75, 1.36] = frame collision, beyond = clean
    miss. Backward crossings through the frame band also collide.

    INC7 contact-true scoring (defaults OFF = legacy): ``body_radius`` inflates the bands
    (pass < 0.75 - r, frame band [0.75 - r, 1.36 + r]); ``frame_depth`` makes the frame band
    MATERIAL over |x| <= depth (slab_frame_hit_np) -- catching strikes BEFORE the plane and
    striking crossings whose interpolated point threads the pass band.

    ``gate_pos_zup``/``gate_yaw`` (P4-C05): None = the hardcoded VQ1 course (_GATE_POS_ZUP +
    yaw=π, bit-exact); pass per-gate arrays to score a non-π course correctly."""
    gp = _GATE_POS_ZUP if gate_pos_zup is None else gate_pos_zup
    w2g = _w2g_for(gate, gate_yaw)
    half_in = _HALF_OPEN - body_radius
    half_out = _HALF_OUTER + body_radius
    prev_rel = w2g @ (prev_ned * _FLIP - gp[gate])
    cur_rel  = w2g @ (cur_ned * _FLIP - gp[gate])
    if frame_depth > 0.0 and slab_frame_hit_np(prev_rel, cur_rel, half_in, half_out,
                                               frame_depth):
        return "collision"
    fwd = prev_rel[0] < 0.0 and cur_rel[0] >= 0.0
    bwd = prev_rel[0] > 0.0 and cur_rel[0] <= 0.0
    if not (fwd or bwd):
        return None
    f = -prev_rel[0] / ((cur_rel[0] - prev_rel[0]) or 1e-9)
    y = prev_rel[1] + f * (cur_rel[1] - prev_rel[1])
    z = prev_rel[2] + f * (cur_rel[2] - prev_rel[2])
    linf = max(abs(y), abs(z))
    if fwd and linf < half_in:
        return "pass"
    if half_in <= linf <= half_out:            # frame band only -- bwd through the OPEN
        return "collision"                     # aperture is a non-event (matches the env)
    return "miss" if fwd else None


def frame_strike_other_gates(prev_ned: np.ndarray, cur_ned: np.ndarray, target: int,
                             body_radius: float = 0.0, frame_depth: float = 0.0,
                             gate_pos_zup: np.ndarray | None = None,
                             gate_yaw: np.ndarray | None = None) -> int | None:
    """S1.4: gates are physical EVERYWHERE -- a frame-band crossing of any non-target gate
    (either direction) is a collision. Returns the struck gate id or None.
    ``gate_pos_zup``/``gate_yaw`` (P4-C05): None = bit-exact VQ1; per-gate arrays for non-π."""
    for g in range(N_GATES):
        if g == target:
            continue
        ev = gate_event(prev_ned, cur_ned, g, body_radius, frame_depth,
                        gate_pos_zup=gate_pos_zup, gate_yaw=gate_yaw)
        if ev == "collision":
            return g
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
    ap.add_argument("--checkpoint",
                    default=str(Path(__file__).resolve().parent / "checkpoints"
                                / "stage1_inc4_actor.pth"))
    ap.add_argument("--plant", default="map", choices=["map", "flat", "aero", "mixer", "lapse"],
                    help="map (DEFAULT, S1.4+): measured super-rate gain map + slew limits "
                         "(SUPER_RATE_S_MEASURED / ALPHA_MAX_RPS2_MEASURED). flat: the legacy "
                         "flat-2.5 plant -- ONLY correct for flat-trained checkpoints "
                         "(stage1_inc1/inc3). aero: map + measured aero (quad body drag + "
                         "convex collective, linear_drag=0) -- for aero-trained checkpoints "
                         "(stage1_inc5). mixer: aero + the measured motor-mixer coupling "
                         "(live-deploy diag 2026-06-11) -- the fully measured plant as of S17, "
                         "for mixer-trained checkpoints (stage1_inc6+) and for reproducing the "
                         "inc5 live failure modes offline; ALSO the fully measured plant as of "
                         "the 2026-06-12 FRAME-AUDIT. lapse: mixer + the S18 airspeed thrust "
                         "lapse -- 🚩 VOIDED by the frame audit (the lapse was an artifact of "
                         "the mirrored attitude reading; see rl_plant.LAPSE_* comment): kept "
                         "only to reproduce historical S18 analyses")
    ap.add_argument("--start", default="simstart",
                    choices=["trainreset", "racestart", "simstart", "handoff"])
    ap.add_argument("--gate", type=int, default=0, help="trainreset: which gate")
    ap.add_argument("--handoff-dist",  type=float, default=3.0)
    ap.add_argument("--handoff-speed", type=float, default=10.0)
    ap.add_argument("--max-rate",   type=float, default=0.0,
                    help="PATH A: cap |rate_flu| rad/s before the plant; 0=off")
    ap.add_argument("--yaw-scale",  type=float, default=1.0,
                    help="scale the policy yaw-rate command (0 = drop yaw): twin "
                         "validation of the S17 live mixer mitigation")
    ap.add_argument("--virtual-flip", action=argparse.BooleanOptionalAction, default=True,
                    help="π body-z conjugation (policy flies tail-first; sim spawns "
                         "nose-first). DEFAULT ON -- matches fly_rl.py's deployment default "
                         "and is correct for the telemetry-frame starts (simstart/racestart/"
                         "handoff). Use --no-virtual-flip with --start trainreset: that start "
                         "is already in the training-native frame, and flipping it lands OOD")
    ap.add_argument("--max-thrust", type=float, default=0.0,
                    help="cap normed_thrust (training units, hover=1); 0=off")
    ap.add_argument("--cap-gates", type=int, default=99,
                    help="apply --max-rate/--max-thrust only while target gate < N")
    ap.add_argument("--latency-steps", type=int, default=0,
                    help="plant transport delay in control steps (live ~40 ms ≈ 1)")
    ap.add_argument("--obs-extrap-steps", type=float, default=0.0,
                    help="latency compensation: build the obs from the state "
                         "extrapolated forward this many control steps (pos += vel*tau, "
                         "attitude integrated by omega*tau). Pair with --latency-steps "
                         "to validate the live compensator")
    ap.add_argument("--thrust0",    type=float, default=-1.0,
                    help="initial realized collective state; <0 = hover (adapter reset)")
    ap.add_argument("--live-thrust-clip", action="store_true",
                    help="clip collective to [0,1] like the live sim (training did NOT)")
    ap.add_argument("--body-radius", type=float, default=0.0,
                    help="INC7 contact-true scoring: inflate the gate bands by this body "
                         "radius (pass < 0.75-r, frame [0.75-r, 1.36+r]); 0 = legacy point")
    ap.add_argument("--frame-depth", type=float, default=0.0,
                    help="INC7 contact-true scoring: frame band is material over gate-frame "
                         "|x| <= depth (volumetric strike test); 0 = legacy plane band")
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

    _aero = dict(rate_sign=_RATE_SIGN_LIVE.copy(),
                 super_rate_s=SUPER_RATE_S_MEASURED,
                 alpha_max_rps2=ALPHA_MAX_RPS2_MEASURED.copy(),
                 linear_drag=0.0,
                 quad_drag_c2=QUAD_DRAG_C2_MEASURED.copy(),
                 coll_map_thr=COLL_MAP_THR_MEASURED.copy(),
                 coll_map_accel=COLL_MAP_ACCEL_MEASURED.copy())
    if args.plant in ("mixer", "lapse"):    # fully measured plant as of S17 (mixer) / S18 (+lapse)
        _lapse = (dict(lapse_speed=LAPSE_SPEED_MEASURED.copy(),
                       lapse_factor=LAPSE_FACTOR_MEASURED.copy())
                  if args.plant == "lapse" else {})
        params = PlantParams(transport_delay_steps=args.latency_steps, **_aero,
                             mixer_idle=MIXER_IDLE_MEASURED,
                             mixer_kappa_err=MIXER_KAPPA_ERR_MEASURED,
                             mixer_kappa_hold=MIXER_KAPPA_HOLD_MEASURED,
                             mixer_zeta_yaw=MIXER_ZETA_YAW_MEASURED, **_lapse)
    elif args.plant == "aero":   # the S16 fixed-params form (pre-mixer)
        params = PlantParams(transport_delay_steps=args.latency_steps, **_aero)
    elif args.plant == "map":
        params = PlantParams(transport_delay_steps=args.latency_steps,
                             rate_sign=_RATE_SIGN_LIVE.copy(),
                             super_rate_s=SUPER_RATE_S_MEASURED,
                             alpha_max_rps2=ALPHA_MAX_RPS2_MEASURED)
    else:
        params = PlantParams(transport_delay_steps=args.latency_steps,   # legacy flat-2.5
                             rate_sign=_RATE_SIGN_LIVE.copy())
    st, gate = make_start(args.start, args)
    print(f"[start] {args.start}: pos_ned={np.round(st.pos,2).tolist()} "
          f"vel={np.round(st.vel,2).tolist()} target_gate={gate} thrust0={float(st.thrust):.3f} "
          f"plant={args.plant} max_rate={args.max_rate or 'off'} "
          f"live_clip={args.live_thrust_clip} vflip={args.virtual_flip}")

    last_normed = 0.0             # training reset: last_action = 0
    n_steps = int(round(args.max_time / args.dt))
    passed: list[int] = []
    outcome = "TIMEOUT"
    log = {"t": [], "pos": [], "vel": [], "act": []}

    obs0 = obs_from_truth(st, gate, last_normed, args.virtual_flip)
    r0, c0, n0 = policy_step(actor, obs0, args.max_rate, args.virtual_flip,
                             args.max_thrust, yaw_scale=args.yaw_scale)
    print(f"[step0] obs pos_g={np.round(obs0[0:3],2).tolist()} vel_g={np.round(obs0[3:6],2).tolist()} "
          f"rpy_g={np.round(obs0[6:9],2).tolist()}")
    print(f"[step0] action: rate_frd={np.round(r0,3).tolist()} collective={c0:.3f} "
          f"normed_thrust={n0:.3f}")

    from racer.rl_plant import quat_multiply, quat_normalize, rotvec_to_quat
    from dataclasses import replace as _dc_replace

    def _extrap(s: PlantState, tau: float) -> PlantState:
        """First-order state prediction for the obs (latency compensation)."""
        if tau <= 0.0:
            return s
        return _dc_replace(
            s, pos=s.pos + s.vel * tau,
            quat=quat_normalize(quat_multiply(s.quat, rotvec_to_quat(s.omega * tau))))

    for k in range(n_steps):
        capped = gate < args.cap_gates
        obs = obs_from_truth(_extrap(st, args.obs_extrap_steps * args.dt),
                             gate, last_normed, args.virtual_flip)
        rate_frd, collective, last_normed = policy_step(
            actor, obs, args.max_rate if capped else 0.0, args.virtual_flip,
            args.max_thrust if capped else 0.0, yaw_scale=args.yaw_scale)
        if not args.live_thrust_clip:
            collective = last_normed * _HOVER_THRUST   # un-clipped, exactly training
        action = np.concatenate([rate_frd, [collective]])

        prev_pos = st.pos.copy()
        st = plant_step(st, action, args.dt, params)
        t = (k + 1) * args.dt

        log["t"].append(t); log["pos"].append(st.pos.copy())
        log["vel"].append(st.vel.copy()); log["act"].append(action)

        ev = gate_event(prev_pos, st.pos, gate, args.body_radius, args.frame_depth)
        struck = frame_strike_other_gates(prev_pos, st.pos, gate,
                                          args.body_radius, args.frame_depth)
        if struck is not None:
            print(f"  t={t:6.2f}s  gate {struck} FRAME STRIKE (non-target)  "
                  f"speed={np.linalg.norm(st.vel):5.1f}")
            outcome = "COLLISION"
            break
        if ev == "pass":
            passed.append(gate)
            # interpolate the crossing point (matches gate_event) for the true in-plane Linf
            _prev_rel = _R_W2G @ (prev_pos * _FLIP - _GATE_POS_ZUP[gate])
            _cur_rel  = _R_W2G @ (st.pos * _FLIP - _GATE_POS_ZUP[gate])
            _f = -_prev_rel[0] / ((_cur_rel[0] - _prev_rel[0]) or 1e-9)
            _iy = _prev_rel[1] + _f * (_cur_rel[1] - _prev_rel[1])
            _iz = _prev_rel[2] + _f * (_cur_rel[2] - _prev_rel[2])
            _linf = max(abs(_iy), abs(_iz))
            print(f"  t={t:6.2f}s  gate {gate} PASS   speed={np.linalg.norm(st.vel):5.1f} m/s "
                  f"off=[{_iy:+.2f},{_iz:+.2f}] Linf={_linf:.3f}m")
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
        elif ev == "miss":
            rel = _R_W2G @ (st.pos * _FLIP - _GATE_POS_ZUP[gate])
            print(f"  t={t:6.2f}s  gate {gate} CLEAN MISS  off=[{rel[1]:+.2f},{rel[2]:+.2f}] m "
                  f"speed={np.linalg.norm(st.vel):5.1f}")
            outcome = "MISS"
            break

        # out-of-bounds box, EXACTLY the training box: bbox of gates + the spawn pad
        # (0,0,-0.02 zup), then +-15/12 m margins (peregrine_racing._update_boxes)
        zup = st.pos * _FLIP
        pts = np.vstack([_GATE_POS_ZUP, [0.0, 0.0, -0.02]])
        lo = pts.min(0) - [15, 15, 12]
        hi = pts.max(0) + [15, 15, 12]
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
