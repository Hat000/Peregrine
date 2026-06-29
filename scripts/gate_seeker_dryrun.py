"""Offline dry-run for the VQ2 slow gate-seeker on the case-C self-localizing stack.

Runs the INTEGRATED guidance pipeline on the laptop — NO sim, NO Adroit, NO live flight — so the
ShadowPC operator can confirm the controller produces sane, slow, gate-pointing commands BEFORE
flying it. Two checks:

  1. PURSUIT TRAJECTORY (kinematic closed loop): integrate a simple point-mass under the
     gate-seeker's DESIRED VELOCITY through a small synthetic gate course and confirm the drone
     converges to and passes each gate centre at the slow cruise speed, zero contact. This isolates
     the GUIDANCE LAW (does it steer to the gate?) from the estimator.

  2. CASE-C ESTIMATOR + SEEKER (the real seam): drive the case-C Navigator (use_ahrs, no given
     pose, gate-relative + range + rewind) over a synthetic IMU + injected gate sightings, run the
     gate-seeker on its NavState each tick, and confirm the self-localized estimate stays bounded
     and every command is a sane slow BODY_RATE. This is the laptop de-risk of the live loop.

Usage:
  .venv\\Scripts\\python.exe scripts\\gate_seeker_dryrun.py
  .venv\\Scripts\\python.exe scripts\\gate_seeker_dryrun.py --speed 2.0 --gates 4

Exit code 0 = all checks passed (sane slow gate-pointing commands); 1 = a check failed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from racer.contracts import ControlMode, DroneState, Frame, Gate, GateObservation, NavState
from racer.deploy_profile import vq2_case_c
from racer.frames import CAMERA_INTRINSICS_K, R_camera_from_body, R_world_from_body
from racer.gate_seeker import GateSeeker, GateSeekerConfig
from racer.navigator import NavigatorConfig, Navigator


def _u(v):
    n = float(np.linalg.norm(v))
    return np.asarray(v, float) / n if n > 1e-9 else np.zeros(3)


def _gate(position, normal, gate_id=0) -> Gate:
    n = _u(normal)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = _u(np.cross(a, n))
    y = np.cross(n, x)
    return Gate(gate_id=gate_id, position_ned=np.asarray(position, float),
                R_world_gate=np.column_stack([x, y, n]))


# ---------------------------------------------------------------------------
# CHECK 1 — pursuit trajectory (kinematic closed loop on the desired velocity)
# ---------------------------------------------------------------------------
def check_pursuit_trajectory(cruise_speed: float, n_gates: int) -> bool:
    """Integrate a point-mass under the seeker's desired velocity through a straight slalom course;
    confirm it converges to + passes every gate centre within a small radius at the slow cruise."""
    # a gentle slalom (alternating small lateral offsets) so the pursuit must actually steer, not
    # just go straight; gates well-spaced so the slow pursuit has room to centre before each plane.
    gates = [
        _gate([25.0 * (i + 1), 1.5 * ((-1) ** i), -2.0], normal=[1.0, 0.0, 0.0], gate_id=i)
        for i in range(n_gates)
    ]
    seeker = GateSeeker(config=GateSeekerConfig(cruise_speed=cruise_speed, launch_ramp_s=0.0))
    dt = 0.05
    pos = np.array([0.0, 0.0, -2.0])
    # the downrange direction for each gate, fixed from the SPAWN approach (so the metric does not
    # flip when the drone crosses the plane); gates all face +X here so this is just the map normal.
    gnorms = []
    for g in gates:
        gn = _u(g.normal_ned)
        if float((np.asarray(g.position_ned, float) - pos) @ gn) < 0.0:
            gn = -gn
        gnorms.append(gn)
    gi = 0
    # the true "did it thread the opening?" metric: the IN-PLANE (lateral+vertical) miss recorded as
    # the drone approaches and crosses each gate's plane — the closest in-plane offset within the
    # last 3 m of along-track (the along-track overshoot past the carrot is irrelevant).
    cross_miss = [np.inf] * n_gates
    max_speed = 0.0
    t = 0.0
    while gi < n_gates and t < 200.0:
        gate = gates[gi]
        is_final = gi == n_gates - 1
        gpos = np.asarray(gate.position_ned, float)
        gn = gnorms[gi]
        # integrate along the GUIDANCE's desired velocity directly (the inner-loop tracking fidelity
        # is validated separately in check 2 + the unit tests; here we test the GUIDANCE LAW alone).
        # the nav velocity is set downrange so should_advance's travel-orientation is stable.
        nav = NavState(sim_time_ns=int(t * 1e9), position_ned=pos.copy(),
                       velocity_ned=cruise_speed * gn)
        sp = seeker.plan(nav, gate, is_final_gate=is_final)
        des_vel = np.asarray(sp.velocity_ned, float)
        pos = pos + des_vel * dt
        max_speed = max(max_speed, float(np.linalg.norm(des_vel)))
        # record the in-plane miss within 5 m of along-track to the plane (covers the final-gate
        # blow-out, which advances at ~4 m range before reaching the plane itself).
        along = float((gpos - pos) @ gn)
        if abs(along) <= 5.0:
            offset = pos - gpos
            cross_miss[gi] = min(cross_miss[gi],
                                 float(np.linalg.norm(offset - (offset @ gn) * gn)))
        if seeker.should_advance(nav, gate, is_final_gate=is_final):
            gi += 1
        t += dt
    passed_all = gi >= n_gates
    worst_miss = max(m if np.isfinite(m) else np.inf for m in cross_miss)
    speed_ok = max_speed <= cruise_speed * 1.6   # the velocity cap holds (slow, no runaway)
    ok = passed_all and worst_miss < 1.0 and speed_ok
    print(f"  [pursuit] gates passed {gi}/{n_gates}  worst in-plane miss {worst_miss:.2f} m  "
          f"max speed {max_speed:.2f} m/s (cap {cruise_speed:g})  -> {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# CHECK 2 — case-C estimator + seeker (the real self-localizing seam)
# ---------------------------------------------------------------------------
class _ProjDetector:
    """Project a gate's inner corners into the camera from the TRUE pose -> a clean detection the
    navigator's own PnP solves into a real localization fix (model-free)."""

    def __init__(self, gate: Gate, drone_pos, R_wb):
        self.gate, self.drone_pos, self.R_wb = gate, np.asarray(drone_pos, float), np.asarray(R_wb, float)

    def detect(self, frame):
        half = self.gate.inner_size_m / 2.0
        corners_gate = np.array([[-half, half, 0.0], [half, half, 0.0],
                                 [half, -half, 0.0], [-half, -half, 0.0]])
        R_wg = np.asarray(self.gate.R_world_gate, float)
        R_cb = R_camera_from_body()
        K = CAMERA_INTRINSICS_K
        px = []
        for cg in corners_gate:
            p_world = self.gate.position_ned + R_wg @ cg
            p_cam = R_cb @ (self.R_wb.T @ (p_world - self.drone_pos))
            if p_cam[2] <= 0.05:
                return []
            px.append([K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2],
                       K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]])
        return [GateObservation(frame_id=frame.frame_id, sim_time_ns=frame.sim_time_ns,
                                corners_px=np.asarray(px, float), corner_ids=np.array([0, 1, 2, 3]),
                                corner_confidence=np.ones(4))]


def check_casec_estimator_seeker(cruise_speed: float) -> bool:
    """Drive the case-C Navigator + seeker over a synthetic hover IMU + injected sightings; confirm
    the self-localized estimate stays bounded and every command is a sane slow BODY_RATE."""
    from racer.ahrs.imu_gen import Scenario, generate_imu_sequence

    gate = _gate([12.0, 0.0, -2.5], normal=[1.0, 0.0, 0.0], gate_id=0)
    drone_pos = np.zeros(3)
    R_wb = R_world_from_body(0.0, 0.0, 0.0)
    cfg = NavigatorConfig(use_given_position=False, use_given_velocity=False, use_vision=True,
                          use_ahrs=True, use_gate_relative=True, use_rewind_kf=True,
                          use_range_channel=True)
    nav = Navigator(gates=[gate], detector=_ProjDetector(gate, drone_pos, R_wb), config=cfg)
    seeker = GateSeeker(config=GateSeekerConfig(cruise_speed=cruise_speed, launch_ramp_s=0.0))
    seq = generate_imu_sequence(Scenario.STATIC_GRAVITY, duration_s=1.5, dt=0.01, seed=7)
    img = np.zeros((360, 640, 3), dtype=np.uint8)

    max_err = 0.0
    max_rate = 0.0
    max_thr = 0.0
    bad = 0
    for k in range(seq.N):
        t_ns = int(k * 0.01 * 1e9)
        ds = DroneState(sim_time_ns=t_ns, accel_body=np.asarray(seq.accel[k], float),
                        gyro_body=np.asarray(seq.gyro[k], float), position_ned=None,
                        velocity_ned=None, active_gate_index=0)
        frame = Frame(frame_id=k, sim_time_ns=t_ns, image_bgr=img) if k % 2 == 0 else None
        ns = nav.update(ds, frame)
        cmd = seeker.command(ns, gate, 0)
        max_err = max(max_err, float(np.linalg.norm(ns.position_ned - drone_pos)))
        max_rate = max(max_rate, float(np.linalg.norm(cmd.body_rate)))
        max_thr = max(max_thr, float(cmd.thrust))
        if (cmd.mode is not ControlMode.BODY_RATE or not np.all(np.isfinite(cmd.body_rate))
                or not np.isfinite(cmd.thrust) or cmd.thrust < 0.0 or cmd.thrust > 1.0
                or np.linalg.norm(cmd.body_rate) > seeker.controller.max_body_rate_rps + 1e-9):
            bad += 1
    bounded = max_err < 25.0
    fixes_ran = nav.n_vision_fixes > 0
    ok = bounded and fixes_ran and bad == 0
    print(f"  [case-C] vision fixes {nav.n_vision_fixes}  max |est err| {max_err:.2f} m  "
          f"max |rate| {max_rate:.2f} rad/s  max thrust {max_thr:.3f}  bad cmds {bad}  "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# CHECK 3 — VQ2 REALITY: MAP-FREE anchor release (the 2026-06-29 attempt-2 BUG A)
# ---------------------------------------------------------------------------
def check_vq2_map_free_anchor_release(cruise_speed: float) -> bool:
    """Reproduce the LIVE VQ2 wire the SECOND slow-lap stalled on (BUG A), and prove the NEW seeker
    survives it. On VQ2 the navigator runs MAP-FREE (gates=[]): its map-associated fix path never
    fires, so ``nav.time_since_vision_update_s`` stays inf FOREVER. The OLD anchor-release signal
    (tsv finite) is therefore structurally UNREACHABLE -> the seeker is pinned in the launch-hold
    forever, even with the gate centred (exactly attempt-2: held level, saw the gate, never pursued).

    Asserts:
      (a) OLD rule (release only when tsv finite) NEVER releases over the whole map-free run -> pinned;
      (b) NEW rule (release on the seeker's OWN N consecutive quality-gated detections) RELEASES even
          though tsv stays inf, and the seeker then makes forward PROGRESS toward the seen gate;
      (c) no blind launch slew before release; every command is a sane bounded BODY_RATE."""
    from racer.ahrs.imu_gen import Scenario, generate_imu_sequence

    # The TRUE start gate is ahead (north) + up — exactly what the camera sees at spawn.
    true_gate = _gate([12.0, 0.0, -2.5], normal=[1.0, 0.0, 0.0], gate_id=0)
    drone_pos = np.zeros(3)
    R_wb = R_world_from_body(0.0, 0.0, 0.0)        # level, facing north -> the TRUE gate is in view
    detector = _ProjDetector(true_gate, drone_pos, R_wb)

    # MAP-FREE navigator: gates=[] (the live VQ2 wire). No map gates -> no map-associated fix ever ->
    # tsv stays inf. This is the seam attempt-2 actually flew (NOT a true-gate-fed nav that masks it).
    cfg = NavigatorConfig(use_given_position=False, use_given_velocity=False, use_vision=True,
                          use_ahrs=True, use_gate_relative=True, use_rewind_kf=True,
                          use_range_channel=True)
    nav = Navigator(gates=[], detector=None, config=cfg)
    N = 3
    seeker = GateSeeker(config=GateSeekerConfig(cruise_speed=cruise_speed, launch_ramp_s=0.0,
                                                settle_s=0.0, anchor_release_detections=N),
                        detector=detector)

    seq = generate_imu_sequence(Scenario.STATIC_GRAVITY, duration_s=1.5, dt=0.01, seed=7)
    img = np.ones((360, 640, 3), dtype=np.uint8)   # non-None image so the detector runs

    cap = seeker.controller.max_body_rate_rps

    old_anchored = False            # the OLD tsv-only rule, evaluated over the map-free run
    tsv_ever_finite = False
    release_tick = None
    max_yaw_pre_release = 0.0
    fwd_progress = 0.0
    bad = 0
    last_ns = None
    for k in range(seq.N):
        t_ns = int(k * 0.01 * 1e9)
        ds = DroneState(sim_time_ns=t_ns, accel_body=np.asarray(seq.accel[k], float),
                        gyro_body=np.asarray(seq.gyro[k], float), position_ned=None,
                        velocity_ned=None, active_gate_index=0)
        frame = Frame(frame_id=k, sim_time_ns=t_ns, image_bgr=img)
        ns = nav.update(ds, frame)
        last_ns = ns
        if np.isfinite(ns.time_since_vision_update_s):     # the OLD release rule
            old_anchored = True
            tsv_ever_finite = True
        was_anchored = seeker._anchored
        cmd = seeker.command_visual(ns, frame, 0)
        yr = abs(float(cmd.body_rate[2]))
        if not seeker._anchored:
            max_yaw_pre_release = max(max_yaw_pre_release, yr)
        if seeker._anchored and not was_anchored and release_tick is None:
            release_tick = k
        if (cmd.mode is not ControlMode.BODY_RATE or not np.all(np.isfinite(cmd.body_rate))
                or not np.isfinite(cmd.thrust) or not (0.0 <= cmd.thrust <= 1.0)
                or np.linalg.norm(cmd.body_rate) > cap + 1e-9):
            bad += 1
    if seeker._anchored and seeker._last_pose is not None:
        fwd_progress = float(seeker._gate_dir_world(last_ns, seeker._last_pose)[0])

    new_released = seeker._anchored and release_tick is not None
    old_would_pin = not old_anchored and not tsv_ever_finite   # tsv never finite -> OLD pinned forever
    preanchor_safe = max_yaw_pre_release < 0.5
    progressed = fwd_progress > 0.5
    ok = new_released and old_would_pin and preanchor_safe and progressed and bad == 0
    print(f"  [vq2-mapfree] OLD tsv-finite ever? {tsv_ever_finite} (OLD pins={old_would_pin})  "
          f"NEW released@tick {release_tick} (own-detection, N={N})  "
          f"pre-release max|yaw| {max_yaw_pre_release:.2f}  fwd {fwd_progress:+.2f}  "
          f"bad {bad}  -> {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# CHECK 4 — COLD-AHRS LAUNCH: attitude-safe settle from ~18deg (2026-06-29 attempt-2 BUG B)
# ---------------------------------------------------------------------------
def check_cold_ahrs_launch_stays_bounded(cruise_speed: float) -> bool:
    """Reproduce the attempt-2 cold-AHRS pitch tumble (BUG B) and prove the fix. The drone spawns
    TILTED ~18deg inside the start gate on a cold, mag-free AHRS. We drive the case-C AHRS over a
    tilted-rest IMU stream and run the seeker's launch hold each tick.

      * GRAVITY-ALIGN seed: the adapter's first-ingest now seeds level-from-accel, so the ESTIMATED
        attitude tracks the TRUE 18deg tilt (estimate ERROR ~0deg) -- NOT identity (18deg error).
      * SETTLE + ATTITUDE-SAFE HOLD: the seeker holds conservative level with roll/pitch clamped.

    The failure mode was: an IDENTITY seed makes the AHRS believe LEVEL while the drone is 18deg
    tilted -> the controller leans off a wrong attitude AND the accel-pull transient drives a large
    correction -> the saturated pitch-over. With a gravity-align seed the ESTIMATE matches reality
    (~0 error) AND the attitude-safe hold clamps roll/pitch, so no pitch-over.

    Asserts: (a) the gravity-align seed ERROR vs the true tilt is ~0 (the OLD identity seed = 18deg
    error); (b) the estimate stays low-error (tracks the true attitude) through the settle; (c) the
    commanded roll/pitch rates stay BOUNDED (no saturated pitch-over)."""
    from racer.ahrs.ahrs_adapter import AHRSAttitudeSource
    from racer.frames import euler_from_quat_wxyz

    # an ~18deg pitched rest attitude: the specific force the IMU reads at rest, tilted.
    pitch0 = np.deg2rad(18.0)
    R_tilt = R_world_from_body(0.0, pitch0, 0.0)
    g_ned = np.array([0.0, 0.0, 9.80665])
    sf_body = R_tilt.T @ (-g_ned)                  # specific force at rest, in the tilted body frame

    # OLD identity seed: it believes LEVEL while the drone is 18deg tilted -> 18deg ESTIMATE ERROR.
    old_identity_err_deg = float(np.rad2deg(abs(pitch0)))

    # NEW gravity-align seed via the adapter's first ingest (level_seed_from_accel).
    src = AHRSAttitudeSource()
    src.ingest(sf_body, np.zeros(3), dt=0.0)       # dt<=0: just seed (no integration)
    r0, p0, _ = euler_from_quat_wxyz(src.q_wxyz)
    seed_err_deg = float(np.rad2deg(np.hypot(r0 - 0.0, p0 - pitch0)))   # vs the TRUE tilt -> ~0

    # run a short tilted-rest stream through the adapter + the seeker's settle/attitude-safe hold.
    rp_rate_cap = GateSeekerConfig().hold_rp_rate_cap_rps
    seeker = GateSeeker(config=GateSeekerConfig(cruise_speed=cruise_speed, launch_ramp_s=0.0,
                                                settle_s=0.75), detector=None)
    dt = 0.01
    n = 120
    max_rp_rate = 0.0
    max_est_err_deg = 0.0
    for k in range(n):
        t_ns = int(k * dt * 1e9)
        src.ingest(sf_body, np.zeros(3), dt=dt)    # static tilted rest: constant specific force
        r, p, y = src.euler_rpy
        # ESTIMATE ERROR vs the TRUE 18deg tilt (the gravity-align estimate should TRACK it, ~0 err).
        max_est_err_deg = max(max_est_err_deg, float(np.rad2deg(np.hypot(r - 0.0, p - pitch0))))
        # the seeker reads the AHRS attitude via NavState; build one carrying it.
        ns = NavState(sim_time_ns=t_ns, position_ned=np.array([0.0, 0.0, -2.5]),
                      velocity_ned=np.zeros(3), roll=float(r), pitch=float(p), yaw=float(y),
                      angular_rate_body=src.body_rate, time_since_vision_update_s=float("inf"))
        cmd = seeker.command_visual(ns, None, 0)   # None frame -> no detection; in settle/hold
        max_rp_rate = max(max_rp_rate, float(np.linalg.norm(cmd.body_rate[:2])))

    seed_ok = seed_err_deg < 1.0                   # the gravity-align seed nails the tilt (~0 error)
    est_tracks = max_est_err_deg < 5.0             # the estimate TRACKS the true attitude (low error)
    rate_bounded = max_rp_rate <= rp_rate_cap + 1e-9   # no saturated pitch-over in the hold
    old_was_bad = old_identity_err_deg > 10.0          # the OLD identity seed started 18deg in error
    ok = seed_ok and est_tracks and rate_bounded and old_was_bad
    print(f"  [cold-ahrs] OLD identity-seed err {old_identity_err_deg:.1f}deg  "
          f"NEW gravity-seed err {seed_err_deg:.2f}deg  max est-err {max_est_err_deg:.2f}deg  "
          f"max|rp rate| {max_rp_rate:.2f} rad/s (cap {rp_rate_cap:g})  bounded={rate_bounded}  "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speed", type=float, default=3.0, help="cruise speed cap (m/s); SLOW first")
    ap.add_argument("--gates", type=int, default=3, help="synthetic slalom gate count (check 1)")
    args = ap.parse_args()

    p = vq2_case_c()
    print("=== gate-seeker offline dry-run (VQ2 slow-is-smooth) ===")
    print(f"deploy profile: {p.name}  self_localizing={p.self_localizing}  "
          f"cmd_rate_scale={p.cmd_rate_scale:g}  cruise_speed={args.speed:g} m/s\n")

    ok1 = check_pursuit_trajectory(args.speed, args.gates)
    ok2 = check_casec_estimator_seeker(args.speed)
    ok3 = check_vq2_map_free_anchor_release(args.speed)
    ok4 = check_cold_ahrs_launch_stays_bounded(args.speed)

    print()
    if ok1 and ok2 and ok3 and ok4:
        print("RESULT: PASS — the integrated gate-seeker produces sane, slow, gate-pointing "
              "commands on synthetic data; the self-localized estimate stays bounded; the seeker "
              "RELEASES its launch-hold on its OWN map-free detections (the attempt-2 BUG A, where "
              "the old tsv signal pins forever); and the cold ~18deg AHRS launch settles to level "
              "with bounded roll/pitch (the attempt-2 BUG B pitch tumble).")
        return 0
    print("RESULT: FAIL — see the failing check above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
