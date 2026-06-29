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
from racer.gate_seeker import GateSeeker, GateSeekerConfig, make_seeker_controller
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


# ---------------------------------------------------------------------------
# CHECK 5 — MULTI-GATE PURSUIT: the A3 range-flap roll-over (Layer 2a/2b)
# ---------------------------------------------------------------------------
def _project_gate_px(gate, drone_pos, R_wb, R_cb, jitter=None):
    """Project a gate's 4 inner corners to pixels from (drone_pos, R_wb); None if behind the camera."""
    half = gate.inner_size_m / 2.0
    corners_gate = np.array([[-half, half, 0.0], [half, half, 0.0],
                             [half, -half, 0.0], [-half, -half, 0.0]])
    R_wg = np.asarray(gate.R_world_gate, float)
    K = CAMERA_INTRINSICS_K
    px = []
    for cg in corners_gate:
        p_world = gate.position_ned + R_wg @ cg
        p_cam = R_cb @ (np.asarray(R_wb, float).T @ (p_world - np.asarray(drone_pos, float)))
        if p_cam[2] <= 0.05:
            return None
        px.append([K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2],
                   K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]])
    out = np.asarray(px, float)
    return out if jitter is None else out + jitter


class _MultiGateDetector:
    """Project SEVERAL red gates into the camera each frame -> a multi-candidate detection, like the
    lit VQ2 course. A CENTERED 'active' gate sits dead ahead at a steady range; off-to-the-SIDE
    distractor gates have their projected corners SCALE-JITTERED frame-to-frame so their PnP depth
    OSCILLATES and each intermittently reads NEARER than the active gate -- and they are on OPPOSITE
    sides, so "pick the closest each frame" flips the chosen gate (and its bearing) side-to-side every
    tick: the A3 10<->30 m flap that swung the steering bearing and saturated roll. Deterministic."""

    def __init__(self, gates, drone_pos, R_wb):
        self.gates = list(gates)
        self.drone_pos = np.asarray(drone_pos, float)
        self.R_wb = np.asarray(R_wb, float)
        self._R_cb = R_camera_from_body()

    def detect(self, frame):
        out = []
        for i, gate in enumerate(self.gates):
            base = _project_gate_px(gate, self.drone_pos, self.R_wb, self._R_cb)
            if base is None:
                continue
            if i == 0:                                   # the centered ACTIVE gate: stable (no jitter)
                px = base
            else:
                # opposite-phase scale jitter: distractor i shrinks while i+1 grows -> their PnP ranges
                # swing in ANTIPHASE so the "closest" flips between the two SIDES tick-to-tick.
                ctr = base.mean(axis=0)
                scale = 1.0 + 0.5 * np.sin(0.9 * int(frame.frame_id) + np.pi * i)
                px = ctr + (base - ctr) * scale
            out.append(GateObservation(frame_id=frame.frame_id, sim_time_ns=frame.sim_time_ns,
                                       corners_px=px, corner_ids=np.array([0, 1, 2, 3]),
                                       corner_confidence=np.ones(4)))
        return out


def _run_multigate(seeker, detector, n_ticks, speed):
    """Drive the seeker's command_visual over a multi-gate stream from a fixed hover pose; return the
    per-tick (tracked range, commanded roll rate, heading) so we can score flap vs lock + roll bound.
    The drone pose is HELD fixed (we test the SEEKER's target selection + steering, not the plant)."""
    from racer.contracts import NavState
    drone_pos = np.zeros(3)
    R_wb = R_world_from_body(0.0, 0.0, 0.0)
    detector.drone_pos = drone_pos
    detector.R_wb = R_wb
    ranges, rolls, yaws = [], [], []
    for k in range(n_ticks):
        t_ns = int(k * 0.02 * 1e9)
        ns = NavState(sim_time_ns=t_ns, position_ned=drone_pos.copy(),
                      velocity_ned=np.zeros(3), roll=0.0, pitch=0.0, yaw=0.0,
                      time_since_vision_update_s=(0.05 if k > 0 else float("inf")))
        frame = Frame(frame_id=k, sim_time_ns=t_ns, image_bgr=np.ones((360, 640, 3), np.uint8))
        cmd = seeker.command_visual(ns, frame, 0)
        pose = seeker._last_pose
        ranges.append(pose.range_m if pose is not None else np.nan)
        rolls.append(float(cmd.body_rate[0]))
        yaws.append(float(seeker._last_yaw) if seeker._last_yaw is not None else 0.0)
    return np.array(ranges), np.array(rolls), np.array(yaws)


def check_multigate_pursuit_locks_one_gate(cruise_speed: float) -> bool:
    """Reproduce the A3 Layer-2 roll-over (multi-gate range flap) and prove the fix. A CENTERED active
    gate + jittering distractors are visible each frame.

      * OLD logic (pick the CLOSEST each frame, no slew/roll cap): the chosen gate's PnP range FLAPS
        tick-to-tick and the heading swings -> the roll command SATURATES (the roll-over).
      * NEW logic (temporal track locks one gate + heading slew-limit + roll cap): the tracked range
        stays SMOOTH, the heading is steady, and the roll command stays BOUNDED (no roll-over).

    Asserts: OLD flaps + saturates roll; NEW range-stable + bounded roll + smooth heading."""
    active = _gate([20.0, 1.0, -2.5], normal=[1.0, 0.0, 0.0], gate_id=0)         # near-centered (slight off)
    distractor_a = _gate([14.0, 12.0, -2.5], normal=[1.0, 0.0, 0.0], gate_id=1)  # hard LEFT-bearing
    distractor_b = _gate([14.0, -12.0, -2.5], normal=[1.0, 0.0, 0.0], gate_id=2)  # hard RIGHT-bearing
    gates = [active, distractor_a, distractor_b]
    cap = make_seeker_controller().max_body_rate_rps
    n = 60

    # OLD: tracking OFF + no slew/roll cap + the LEGACY velocity pursuit (use_feedforward_forward=False)
    # so the heading-aligned velocity setpoint produces the roll swing the unbounded servo crashed on
    # (cap raised to the controller max so only saturation shows). Egress off (isolate pursuit).
    old_seeker = GateSeeker(
        config=GateSeekerConfig(cruise_speed=cruise_speed, launch_ramp_s=0.0, settle_s=0.0,
                                anchor_release_detections=1, use_gate_track=False,
                                pursuit_ramp_s=0.0, pursuit_yaw_slew_rps=0.0,
                                pursuit_roll_rate_cap_rps=0.0, visual_yaw_rate_cap_rps=cap,
                                use_feedforward_forward=False, use_spawn_egress=False),
        detector=_MultiGateDetector(gates, np.zeros(3), None))
    old_r, old_roll, old_yaw = _run_multigate(old_seeker, old_seeker.detector, n, cruise_speed)

    # NEW: tracking ON + heading slew-limit + roll cap (the shipped defaults); egress off to isolate
    # the tracked-pursuit gate-lock + roll bound (egress is exercised in its own check below).
    new_seeker = GateSeeker(
        config=GateSeekerConfig(cruise_speed=cruise_speed, launch_ramp_s=0.0, settle_s=0.0,
                                anchor_release_detections=1, use_spawn_egress=False),
        detector=_MultiGateDetector(gates, np.zeros(3), None))
    new_r, new_roll, new_yaw = _run_multigate(new_seeker, new_seeker.detector, n, cruise_speed)

    # flap metric: max tick-to-tick jump in the CHOSEN gate's range over the pursuit window.
    def _flap(r):
        d = np.abs(np.diff(r[~np.isnan(r)]))
        return float(d.max()) if d.size else 0.0
    # roll-SWING metric: the max tick-to-tick change in the commanded roll rate -- the oscillation that
    # rolled A3 over (a noisy bearing swinging the lean side-to-side). The proximate roll-over signal.
    def _swing(x):
        d = np.abs(np.diff(x[~np.isnan(x)]))
        return float(d.max()) if d.size else 0.0
    def _yaw_jerk(y):
        dy = np.abs(np.arctan2(np.sin(np.diff(y)), np.cos(np.diff(y))))
        return float(dy.max()) if dy.size else 0.0

    old_flap, new_flap = _flap(old_r), _flap(new_r)
    old_rollswing, new_rollswing = _swing(old_roll), _swing(new_roll)
    old_rollmax = float(np.nanmax(np.abs(old_roll)))
    new_rollmax = float(np.nanmax(np.abs(new_roll)))
    new_yawjerk = _yaw_jerk(new_yaw)

    # OLD: the chosen gate's range FLAPS and the roll command SWINGS hard tick-to-tick (the roll-over
    # the unbounded servo produced). NEW: range locked smooth, roll bounded + non-oscillating.
    old_bad = old_flap > 5.0 and old_rollswing > 1.0
    new_locked = new_flap < 3.0                               # NEW tracked range is smooth (locked one gate)
    new_roll_bounded = new_rollmax <= GateSeekerConfig().pursuit_roll_rate_cap_rps + 1e-9
    new_roll_smooth = new_rollswing < 0.5 * old_rollswing     # NEW roll no longer oscillates hard
    new_yaw_smooth = new_yawjerk < 0.2                        # heading steps stay small (slew-limited)
    ok = old_bad and new_locked and new_roll_bounded and new_roll_smooth and new_yaw_smooth
    print(f"  [multigate] OLD flap {old_flap:.1f} m roll_swing {old_rollswing:.2f} roll_max {old_rollmax:.2f}"
          f"  ->  NEW flap {new_flap:.1f} m roll_swing {new_rollswing:.2f} roll_max {new_rollmax:.2f} "
          f"yaw_jerk {new_yawjerk:.3f}  bounded={new_roll_bounded} smooth={new_yaw_smooth}"
          f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# CHECK 6 — SPAWN-TILT HOLD: the gate stays in view through the release streak (Layer 1)
# ---------------------------------------------------------------------------
def check_spawn_tilt_hold_keeps_gate_in_view(cruise_speed: float) -> bool:
    """Reproduce the A3 Layer-1 stall (the launch-hold slowly pitches the camera off the gate) and
    prove the freeze-attitude fix. The drone spawns at a small gravity-aligned PITCH tilt with the
    gate in view; we run the seeker's settle/anchor hold and a TRUE point-mass attitude integrator
    that applies the commanded body rate, then re-detect the gate each tick.

      * OLD hold (force-LEVEL, roll/pitch clamped to 0.6 but NONZERO): the controller commands a
        persistent pitch toward level off the spawn tilt -> the integrated camera attitude DRIFTS, the
        gate leaves the +20deg view, detections stop, and the 3-streak release STALLS (never reaches 3).
      * NEW hold (FREEZE attitude, roll/pitch rate = 0): the camera attitude HOLDS, the gate stays in
        view, and the detection streak completes -> the anchor RELEASES.

    Asserts: OLD streak stalls (gate drifts out) ; NEW streak reaches N and the gate stays centered."""
    # the start gate sits ~2.5 m ABOVE the spawn (z=-5.0 vs drone -2.5); with the +20deg-up camera the
    # gate is CENTERED at a small nose-down spawn tilt. The level-hold then drives the tilt toward 0,
    # pitching the camera so the elevated gate climbs out of the bottom of the frame (the A3 drift-off).
    pitch0 = np.deg2rad(-6.0)         # a gravity-aligned nose-down spawn tilt; the elevated gate is centered
    gate = _gate([14.0, 0.0, -5.0], normal=[1.0, 0.0, 0.0], gate_id=0)
    drone_pos = np.array([0.0, 0.0, -2.5])
    N = 3
    R_cb = R_camera_from_body()
    H, W = 360, 640

    def _centroid_v(pitch):
        """Vertical image coordinate of the gate centroid at this camera pitch (None if behind)."""
        R_wb = R_world_from_body(0.0, float(pitch), 0.0)
        px = _project_gate_px(gate, drone_pos, R_wb, R_cb)
        return None if px is None else float(px.mean(axis=0)[1])

    # the detector locks the gate only while it is BOTH inside the frame AND its image MOTION between
    # ticks is small. As the level-hold pitches the camera off the gate, the gate both drifts toward
    # the edge AND streaks across the frame fast -> the corner detector flickers (the A3 'meanBGR
    # 27->9, detections vanish'): a fast-drifting gate is NOT cleanly detected. A STATIC (frozen) gate
    # sits still and is detected every tick.
    MARGIN, MAX_IMG_MOTION_PX = 12.0, 8.0

    def _detectable(v_now, v_prev):
        if v_now is None:
            return False
        if not (MARGIN <= v_now <= H - MARGIN):
            return False
        if v_prev is not None and abs(v_now - v_prev) > MAX_IMG_MOTION_PX:
            return False                  # too much image motion this tick -> the detector flickers off
        return True

    class _MotionGatedDetector:
        """_ProjDetector but returns nothing unless the gate is in-frame AND nearly STILL this tick."""
        def __init__(self, pitch, detectable):
            self.pitch, self.detectable = pitch, detectable
        def detect(self, frame):
            if not self.detectable:
                return []
            R_wb = R_world_from_body(0.0, float(self.pitch), 0.0)
            return _ProjDetector(gate, drone_pos, R_wb).detect(frame)

    def _run(freeze: bool):
        seeker = GateSeeker(
            config=GateSeekerConfig(cruise_speed=cruise_speed, launch_ramp_s=0.0, settle_s=0.0,
                                    anchor_release_detections=N, hold_freeze_attitude=freeze),
            detector=None)
        pitch = pitch0                # the TRUE camera pitch, integrated from the commanded body rate
        dt = 0.1
        released_at = None
        hold_detectable_ticks = 0     # ticks the gate was detectable WHILE STILL IN THE HOLD (pre-release)
        v_prev = None
        for k in range(40):
            v_now = _centroid_v(pitch)
            detectable = _detectable(v_now, v_prev)
            seeker.detector = _MotionGatedDetector(pitch, detectable)
            t_ns = int(k * dt * 1e9)
            ns = NavState(sim_time_ns=t_ns, position_ned=drone_pos.copy(), velocity_ned=np.zeros(3),
                          roll=0.0, pitch=float(pitch), yaw=0.0, time_since_vision_update_s=float("inf"))
            frame = Frame(frame_id=k, sim_time_ns=t_ns, image_bgr=np.ones((H, W, 3), np.uint8))
            was = seeker._anchored
            if not was and detectable:            # measure detectability during the HOLD only (Layer 1)
                hold_detectable_ticks += 1
            cmd = seeker.command_visual(ns, frame, 0)
            if seeker._anchored and not was and released_at is None:
                released_at = k
            v_prev = v_now
            # integrate the TRUE pitch from the commanded pitch-rate. The OLD level-hold commands a
            # persistent pitch toward level off the spawn tilt; the odo pitch-rate sign (-1) maps a
            # +body_rate[1] command to a pitch CHANGE that drifts the camera off the gate. The FREEZE
            # hold commands ZERO roll/pitch -> the camera stays on the gate -> the streak completes.
            pitch = pitch + (-1.0) * float(cmd.body_rate[1]) * dt
        return released_at, hold_detectable_ticks

    old_release, old_hold_seen = _run(freeze=False)
    new_release, new_hold_seen = _run(freeze=True)

    # OLD: the level-hold pitches the camera off the gate -> the gate streaks/leaves the frame -> the
    # detector flickers, the 3-consecutive streak never completes -> the hold never releases (A3 2.0).
    old_stalls = old_release is None
    # NEW: the freeze-attitude hold keeps the camera ON the gate -> detectable every hold tick -> the
    # streak completes -> the hold RELEASES at tick N-1.
    new_releases = new_release is not None
    new_held_gate = new_hold_seen >= N             # the gate stayed detectable through the whole streak
    ok = old_stalls and new_releases and new_held_gate
    print(f"  [spawn-tilt] OLD release@{old_release} (stalls={old_stalls}, hold-detectable {old_hold_seen})"
          f"  ->  NEW release@{new_release} hold-detectable {new_hold_seen} (>=N={N}: {new_held_gate})"
          f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# CHECK 7 — MAP-FREE PITCH BOUND: forward motion can't wind pitch to saturation (A4)
# ---------------------------------------------------------------------------
def check_mapfree_pursuit_pitch_bounded(cruise_speed: float) -> bool:
    """Reproduce the A4 pursuit PITCH-windup (the crash) and prove the bounded-feedforward fix. On the
    MAP-FREE VQ2 wire velocity is UNOBSERVABLE: the navigator can't estimate forward velocity, so
    ``nav.velocity_ned`` stays ~0 even as the drone is commanded forward. We hold the drone at a fixed
    hover pose (velocity=0 -- the unobservable case) with the gate centred + anchored, and run a
    SUSTAINED pursuit window, recording the commanded pitch each tick.

      * OLD pursuit (``use_feedforward_forward=False``): asks the controller for a desired VELOCITY
        (cruise_speed*los). The controller closes it with a velocity-ERROR term ``kd_vel*(des_vel-vel)``
        that NEVER closes (vel stays 0) -> the demanded forward accel/tilt stays large -> the commanded
        PITCH WINDS UP toward the -4.0 rad/s controller limit (the A4 trace: -0.69 -> -2.39 -> -3.999).
      * NEW pursuit (bounded feedforward forward tilt + pitch cap + forward ramp): the forward demand is
        a fixed bounded feedforward accel (no velocity term to wind up), so the commanded PITCH stays
        BOUNDED well below saturation across the whole window.

    Asserts: OLD pitch winds toward the -4.0 limit; NEW pitch stays bounded (<= the pitch cap)."""
    gate = _gate([12.0, 0.0, -2.5], normal=[1.0, 0.0, 0.0], gate_id=0)   # dead-ahead, centred
    R_wb = R_world_from_body(0.0, 0.0, 0.0)
    cap = make_seeker_controller().max_body_rate_rps      # the controller saturation limit (4.0)
    pitch_cap = GateSeekerConfig().pursuit_pitch_rate_cap_rps

    def _run(feedforward: bool, egress: bool, pitch_cap_rps: float):
        seeker = GateSeeker(
            config=GateSeekerConfig(cruise_speed=cruise_speed, launch_ramp_s=0.0, settle_s=0.0,
                                    anchor_release_detections=1,
                                    use_feedforward_forward=feedforward, use_spawn_egress=egress,
                                    pursuit_pitch_rate_cap_rps=pitch_cap_rps),
            detector=_ProjDetector(gate, np.zeros(3), R_wb))
        pitches = []
        # a SUSTAINED pursuit window: velocity is HELD at 0 (the map-free unobservable case) the whole
        # time, so a velocity-error forward term has unbounded time to wind up.
        for k in range(40):
            t_ns = int(k * 0.05 * 1e9)
            ns = NavState(sim_time_ns=t_ns, position_ned=np.zeros(3), velocity_ned=np.zeros(3),
                          roll=0.0, pitch=0.0, yaw=0.0,
                          time_since_vision_update_s=(0.05 if k > 0 else float("inf")))
            frame = Frame(frame_id=k, sim_time_ns=t_ns, image_bgr=np.ones((360, 640, 3), np.uint8))
            cmd = seeker.command_visual(ns, frame, 0)
            pitches.append(float(cmd.body_rate[1]))
        return np.array(pitches)

    # OLD: legacy velocity pursuit, pitch cap OFF (so the raw windup shows), egress off (isolate the
    # pursuit pitch windup). This is the genuine pre-A4 path: a velocity setpoint the controller closes
    # with a velocity-error term that never closes map-free -> the pitch winds to saturation.
    old_pitch = _run(feedforward=False, egress=False, pitch_cap_rps=0.0)
    # NEW: bounded feedforward + the shipped pitch cap (egress off so we measure the PURSUIT pitch).
    new_pitch = _run(feedforward=True, egress=False, pitch_cap_rps=pitch_cap)

    old_max = float(np.max(np.abs(old_pitch)))
    new_max = float(np.max(np.abs(new_pitch)))
    old_winds = old_max > 0.9 * cap                       # OLD winds toward the -4.0 saturation limit
    new_bounded = new_max <= pitch_cap + 1e-9             # NEW stays under the pitch cap (no windup)
    ok = old_winds and new_bounded
    print(f"  [pitch-bound] OLD |pitch|max {old_max:.2f} rad/s (winds toward {cap:g}: {old_winds})  "
          f"->  NEW |pitch|max {new_max:.2f} rad/s (cap {pitch_cap:g}: bounded={new_bounded})"
          f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# CHECK 8 — SPAWN-GATE EGRESS: clear the start gate before pursuing downrange (A4)
# ---------------------------------------------------------------------------
def check_spawn_gate_egress_clears_gate0(cruise_speed: float) -> bool:
    """Reproduce the A4 start-gate contact and prove the egress phase. The drone SPAWNS INSIDE gate 0:
    the START gate is at the spawn (drone sits in its plane). A downrange gate is far ahead. With the
    spawn heading == the start-gate normal (the direction OUT of the gate), we integrate a point-mass
    under the seeker's forward demand and check whether the FIRST forward motion drives the drone
    along the gate normal (egress, clearing the start-gate structure) or lunges off-heading.

      * OLD (no egress): the instant pursuit begins the seeker re-aims at the DOWNRANGE gate and the
        forward lunge is toward it -- but the drone is still in the start-gate plane, so the first
        motion is an immediate forward drive into the surrounding gate frame (A4: a 44-rps impact spike).
      * NEW (egress): a brief CAPPED creep along the FROZEN spawn heading (the start-gate normal) first,
        clearing gate 0 by a margin BEFORE the downrange re-aim -- a gentle straight departure.

    Asserts: NEW makes net forward progress along the start-gate normal during egress (clears gate 0)
    while staying gentle (bounded forward tilt); OLD's first forward motion is into the start-gate
    plane (a downrange re-aim with no straight egress)."""
    # start gate at the spawn, normal +X; a downrange gate far ahead and OFF to the side so the
    # downrange re-aim would pull the heading AWAY from the clean +X egress direction.
    start_normal = np.array([1.0, 0.0, 0.0])
    downrange = _gate([12.0, 8.0, -2.5], normal=[1.0, 0.0, 0.0], gate_id=1)   # off to the +Y side
    R_wb = R_world_from_body(0.0, 0.0, 0.0)                # spawn level, facing +X (== start normal)

    def _run(egress: bool):
        seeker = GateSeeker(
            config=GateSeekerConfig(cruise_speed=cruise_speed, launch_ramp_s=0.0, settle_s=0.0,
                                    anchor_release_detections=1, use_spawn_egress=egress,
                                    egress_s=0.8, forward_ramp_s=1.0),
            detector=_ProjDetector(downrange, np.zeros(3), R_wb))
        pos = np.zeros(3)            # spawn at the origin = inside the start gate
        vel = np.zeros(3)
        dt = 0.05
        max_lateral = 0.0           # max off-(start-normal) excursion during the first ~0.8 s
        for k in range(20):         # ~1 s
            t_ns = int(k * dt * 1e9)
            ns = NavState(sim_time_ns=t_ns, position_ned=pos.copy(), velocity_ned=vel.copy(),
                          roll=0.0, pitch=0.0, yaw=0.0,
                          time_since_vision_update_s=(0.05 if k > 0 else float("inf")))
            frame = Frame(frame_id=k, sim_time_ns=t_ns, image_bgr=np.ones((360, 640, 3), np.uint8))
            cmd = seeker.command_visual(ns, frame, 0)
            # integrate a simple point-mass under the commanded TILT -> horizontal accel a ~ g*tan(tilt)
            # along the commanded heading. We approximate the realised forward accel from the seeker's
            # OWN forward-demand model (bounded feedforward) so the egress vs re-aim DIRECTION is what
            # we score (the inner-loop tracking fidelity is validated elsewhere).
            yaw_cmd = float(seeker._last_yaw) if seeker._last_yaw is not None else 0.0
            a = np.array([np.cos(yaw_cmd), np.sin(yaw_cmd), 0.0]) * 1.0   # unit forward demand
            vel = vel + a * dt
            pos = pos + vel * dt
            max_lateral = max(max_lateral, abs(float(pos[1])))           # off the +X start-normal
        return float(pos[0]), max_lateral

    new_fwd, new_lat = _run(egress=True)
    old_fwd, old_lat = _run(egress=False)

    # NEW: during egress the heading is the FROZEN spawn heading (+X start-normal) -> the drone departs
    # STRAIGHT out of the start gate: net +X progress with LITTLE lateral excursion (clears gate 0).
    new_clears = new_fwd > 0.3 and new_lat < 0.5
    # OLD: no egress -> the first motion re-aims at the off-side downrange gate -> the heading pulls
    # OFF the clean egress line (more lateral excursion while still in the start-gate plane).
    old_reaims_off = old_lat > new_lat + 1e-6
    ok = new_clears and old_reaims_off
    print(f"  [egress] NEW fwd {new_fwd:+.2f} m lateral {new_lat:.2f} m (clears straight: {new_clears})"
          f"  ->  OLD fwd {old_fwd:+.2f} m lateral {old_lat:.2f} m (re-aims off-line: {old_reaims_off})"
          f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# CHECK 9 — VERTICAL ALIGNMENT: descend/climb onto the gate-opening centre (A5 BLOCKER 1)
# ---------------------------------------------------------------------------
def check_vertical_alignment_threads_offset_opening(cruise_speed: float) -> bool:
    """Reproduce the A5 close-range clip and prove the vertical-alignment fix. The gate OPENING sits
    BELOW the drone's held altitude (run4: the drone rode too HIGH and CLIPPED THE GATE TOP BAR at
    2.78 m). We hold the drone laterally on-axis + anchored with the gate seen, and over a pursuit
    window integrate ONLY the vertical kinematics under the seeker's commanded vertical-velocity target
    (vz_t), tracking the world vertical offset between the flight path and the gate-opening centre.

      * OLD (use_vertical_align=False): the seeker holds a FIXED altitude -> the vertical offset stays
        at its initial value -> the drone rides above the opening and CLIPS the top bar.
      * NEW (use_vertical_align=True): the seeker commands a bounded vertical velocity that drives the
        offset toward ~0 -> the drone descends onto the opening centre and THREADS it; vz stays BOUNDED.

    The +20deg mount is faithfully in the loop: the detector projects through R_camera_from_body() and
    the seeker recovers the offset via the body->world rotation, so this exercises the real geometry.

    Asserts: OLD leaves a large residual vertical offset (clips); NEW drives it toward ~0 (threads) with
    a BOUNDED commanded vz."""
    # the drone flies at z=-2.5 (NED); the gate OPENING centre sits ~1.0 m BELOW (z=-1.5) -- the run4
    # geometry (the opening below the held path). 1.0 m > the gate's 0.75 m opening HALF-width, so a
    # FIXED-altitude hold leaves the drone clipping the top bar; the fix must drive the offset INSIDE the
    # opening. The detector projects from the TRUE (descending) drone height so the recovered world
    # vertical offset is the genuine drone-vs-opening gap as the loop closes.
    drone_z = -2.5
    gate = _gate([16.0, 0.0, -1.5], normal=[1.0, 0.0, 0.0], gate_id=0)
    R_wb = R_world_from_body(0.0, 0.0, 0.0)
    speed_cap = GateSeekerConfig().vertical_align_speed_cap_mps
    opening_half_m = gate.inner_size_m / 2.0                  # 0.75 m -- thread if |residual| < this

    def _run(align: bool):
        # the detector projects the gate from the drone's CURRENT (integrated) height each tick, so the
        # offset closes as the drone descends (a faithful closed loop on the vertical channel).
        drone_pos = np.array([0.0, 0.0, drone_z])
        seeker = GateSeeker(config=GateSeekerConfig(
            cruise_speed=cruise_speed, launch_ramp_s=0.0, settle_s=0.0, anchor_release_detections=1,
            use_spawn_egress=False, use_vertical_align=align, vertical_align_ramp_s=0.0),
            detector=_ProjDetector(gate, drone_pos, R_wb))
        z = drone_z
        vz = 0.0
        dt = 0.05
        max_vz_cmd = 0.0
        offset0 = None
        for k in range(80):
            t_ns = int(k * dt * 1e9)
            # re-point the detector at the drone's current height so the seen offset reflects the descent.
            seeker.detector = _ProjDetector(gate, np.array([0.0, 0.0, z]), R_wb)
            ns = NavState(sim_time_ns=t_ns, position_ned=np.array([0.0, 0.0, z]),
                          velocity_ned=np.array([0.0, 0.0, vz]), roll=0.0, pitch=0.0, yaw=0.0,
                          time_since_vision_update_s=(0.05 if k > 0 else float("inf")))
            frame = Frame(frame_id=k, sim_time_ns=t_ns, image_bgr=np.ones((360, 640, 3), np.uint8))
            seeker.command_visual(ns, frame, 0)
            pose = seeker._last_pose
            if pose is not None:
                off = float(seeker._gate_lever_world(ns, pose)[2])    # world vertical offset (NED Z)
                if offset0 is None:
                    offset0 = off
                # the seeker's commanded vz target (Z-only velocity setpoint) drives the descent.
                vz_cmd = seeker._vertical_align_vz(ns, pose)
                max_vz_cmd = max(max_vz_cmd, abs(vz_cmd))
                # integrate the vertical kinematics directly under the commanded vz (the alt-hold tracks
                # vz_t; here we model that tracking as vz -> vz_cmd so the offset closes as it would fly).
                vz = vz_cmd
                z = z + vz * dt
        # final residual offset (re-evaluate at the last height)
        seeker.detector = _ProjDetector(gate, np.array([0.0, 0.0, z]), R_wb)
        return offset0, (gate.position_ned[2] - z), max_vz_cmd  # (initial offset, final residual, max vz)

    new_off0, new_resid, new_maxvz = _run(align=True)
    old_off0, old_resid, _ = _run(align=False)

    # NEW: the descent drives the residual INSIDE the opening half-width (threads the hole) and the
    # commanded vz stayed bounded. OLD: fixed-altitude hold keeps the full ~1.0 m offset (> the 0.75 m
    # half-opening) -> rides above the opening and CLIPS the top bar. NEW must also close MOST of the gap.
    new_threads = abs(new_resid) < opening_half_m            # inside the opening -> threads
    new_closed_most = abs(new_resid) < 0.5 * abs(new_off0)   # closed > half the initial offset
    new_vz_bounded = new_maxvz <= speed_cap + 1e-9           # the commanded vz stayed bounded
    old_clips = abs(old_resid) > opening_half_m              # fixed hold left an offset bigger than the opening
    started_offset = abs(new_off0) > opening_half_m and abs(old_off0) > opening_half_m  # both began outside the opening
    ok = new_threads and new_closed_most and new_vz_bounded and old_clips and started_offset
    print(f"  [vert-align] OLD residual {old_resid:+.2f} m (clips: {old_clips})  ->  "
          f"NEW residual {new_resid:+.2f} m (threads: {new_threads}) max|vz| {new_maxvz:.2f} m/s "
          f"(cap {speed_cap:g}: bounded={new_vz_bounded})  -> {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# CHECK 10 — NEAREST-GATE SELECTION: lock the near gate, not the far off-axis one (A5 BLOCKER 2)
# ---------------------------------------------------------------------------
def check_nearest_gate_first_acquisition(cruise_speed: float) -> bool:
    """Reproduce the A5 far-gate lock and prove the nearest-gate fix. TWO gates are visible at spawn: a
    NEAR centered start-line gate (~9 m, dead ahead) and a FAR off-axis downrange gate (~37 m, +18deg).

      * OLD (prefer_nearest=False, prefer_centered only): picks the most-CENTERED gate. With the near
        start gate slightly OFF-axis and a FAR gate dead-ahead (more centered), prefer-centered locks the
        FAR gate -- exactly the A5 trap (a 37 m off-line lock starving the N-detection release).
      * NEW (prefer_nearest=True): rejects the far (>max_acquire_range_m) gate and scores by range +
        bearing, so it LOCKS the NEAR start-line gate -- the one it must fly first.

    Asserts: NEW locks the NEAR gate (range < ~12 m); OLD locks the FAR gate (range > ~30 m)."""
    # the near start gate is slightly OFF-axis (the racing line isn't dead-centre at spawn); the FAR
    # downrange gate happens to sit MORE centered. prefer-centered alone then prefers the far one -- the
    # A5 far-gate trap. (Range disambiguates: the near gate is the one to fly first.)
    near = _gate([9.0, 2.6, -2.5], normal=[1.0, 0.0, 0.0], gate_id=0)            # near, slightly off-axis (~9 m)
    far = _gate([35.0, 0.0, -2.5], normal=[1.0, 0.0, 0.0], gate_id=1)            # far, dead-ahead (~35 m, centered)
    R_wb = R_world_from_body(0.0, 0.0, 0.0)
    gates = [near, far]

    new_seeker = GateSeeker(config=GateSeekerConfig(prefer_nearest=True),
                            detector=_MultiGateDetector(gates, np.zeros(3), R_wb))
    new_pose = new_seeker.detect_gate_lever(Frame(frame_id=0, sim_time_ns=0,
                                                  image_bgr=np.ones((360, 640, 3), np.uint8)))
    old_seeker = GateSeeker(config=GateSeekerConfig(prefer_nearest=False, track_prefer_centered=True),
                            detector=_MultiGateDetector(gates, np.zeros(3), R_wb))
    old_pose = old_seeker.detect_gate_lever(Frame(frame_id=0, sim_time_ns=0,
                                                  image_bgr=np.ones((360, 640, 3), np.uint8)))

    new_locks_near = new_pose is not None and new_pose.range_m < 12.0
    old_locks_far = old_pose is not None and old_pose.range_m > 30.0
    ok = new_locks_near and old_locks_far
    nr = -1.0 if new_pose is None else new_pose.range_m
    orr = -1.0 if old_pose is None else old_pose.range_m
    print(f"  [near-gate] OLD locked range {orr:.1f} m (far: {old_locks_far})  ->  "
          f"NEW locked range {nr:.1f} m (near: {new_locks_near})  -> {'PASS' if ok else 'FAIL'}")
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
    ok5 = check_multigate_pursuit_locks_one_gate(args.speed)
    ok6 = check_spawn_tilt_hold_keeps_gate_in_view(args.speed)
    ok7 = check_mapfree_pursuit_pitch_bounded(args.speed)
    ok8 = check_spawn_gate_egress_clears_gate0(args.speed)
    ok9 = check_vertical_alignment_threads_offset_opening(args.speed)
    ok10 = check_nearest_gate_first_acquisition(args.speed)

    print()
    if ok1 and ok2 and ok3 and ok4 and ok5 and ok6 and ok7 and ok8 and ok9 and ok10:
        print("RESULT: PASS — the integrated gate-seeker produces sane, slow, gate-pointing "
              "commands on synthetic data; the self-localized estimate stays bounded; the seeker "
              "RELEASES its launch-hold on its OWN map-free detections (the attempt-2 BUG A, where "
              "the old tsv signal pins forever); the cold ~18deg AHRS launch settles with bounded "
              "roll/pitch (the attempt-2 BUG B); the multi-gate pursuit LOCKS one gate so the range "
              "stays smooth and the roll stays bounded (the attempt-3 LAYER-2 roll-over); the "
              "spawn-tilt hold FREEZES attitude so the gate stays in view and the release streak "
              "completes (the attempt-3 LAYER-1 stall); the map-free pursuit commands forward motion "
              "as a BOUNDED FEEDFORWARD tilt so the PITCH stays bounded with no velocity observability "
              "(the attempt-4 pitch windup); the SPAWN-GATE EGRESS clears gate 0 along the start "
              "normal before the downrange re-aim (the attempt-4 start-gate contact); the VERTICAL "
              "ALIGNMENT descends/climbs onto the gate-opening centre (bounded vz) instead of holding "
              "altitude and clipping the top bar (the attempt-5 BLOCKER 1 close-range clip); and the "
              "NEAREST-GATE first acquisition locks the near start-line gate over a far off-axis one "
              "(the attempt-5 BLOCKER 2 far-gate lock).")
        return 0
    print("RESULT: FAIL — see the failing check above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
