"""A29 — LOS-rate (tangential-velocity) damping + continuous epoch reconciliation.

THE TRAP (live VQ2 A29, run 20260703_024023): after clearing gate 0 the drone carried ~3 m/s of
momentum mostly PERPENDICULAR to the gate-1 line-of-sight. The pursuit law (`accel = 1.2 m/s^2
along the LOS`) has NO term that can remove that tangential component — on this velocity-denied
wire velocity is unobservable and ff_owns_horizontal (correctly) removed the damping of the
FICTIONAL dead-reckoned velocity, putting nothing back. Result: a textbook pure-pursuit ORBIT —
21.1 m/s of commanded delta-v integrated to a net 2.6 m/s (coherence 0.12) while the LOS swept
392 deg and the drone "yawed at the gate while coasting around it".

THE FIX (gate_seeker.use_los_rate_damping): the missing observable is the LOS RATE. For a fixed
gate, ``v_t = v.e_t = -r*theta_dot`` (theta = the world LOS angle yaw_des, pre-slew; r = the
tracked PnP range) — classical proportional navigation from signals that already exist. The
lateral demand ``a_lat = -kd*v_t = +kd*r*theta_dot`` along ``e_t = [-sin, cos, 0]`` BRAKES the
tangential drift. theta_dot is differenced on CAMERA-EPOCH pose stamps (same clock both sides,
so the epoch-rate skew cancels), EMA-filtered, sample-gated, capped, deadbanded, and reset at
pass / track-drop.

THE SIGN IS SAFETY-CRITICAL: a flipped lateral sign FEEDS the tangential velocity — it TIGHTENS
the orbit instead of killing it. ``test_orbit_demands_braking_accel`` pins it with true-kinematics
scenarios in BOTH drift directions.

Also pinned here (§4.3): ``NavigatorConfig.reconcile_vision_clock_continuous`` — the camera epoch
ran at 0.9449x wall under GPU load while the IMU epoch tracked 1.0002x, so the learn-ONCE
delta_epoch drifted ~0.05-0.10 s/s: pose_age_s ramped to the 1.0 s cap, the vertical-latch
latency comp over-corrected, and every OOSM fix rewound past RewindKF.horizon_s=0.5 from ~t=5 s
on (KF position channel silently dead). Continuous EMA re-estimation bounds the error; OFF
(default) is the learn-once path, byte-identical.

[VQ2 slow-is-smooth, A29, 2026-07-03]
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from racer.contracts import (  # noqa: E402
    ControlMode, DroneState, Frame, GatePose, NavState, Setpoint,
)
from racer.deploy_profile import get_profile, vq1_case_a, vq2_case_c  # noqa: E402
from racer.frames import R_camera_from_body  # noqa: E402
from racer.gate_seeker import (  # noqa: E402
    GateSeeker, GateSeekerConfig, _unit, make_seeker_controller,
)
from racer.navigator import Navigator, NavigatorConfig  # noqa: E402

# The camera/server epoch on the live wire is a unix-wall-clock ns epoch, vastly larger than the
# IMU sim-uptime epoch — keep the two synthetic epochs distinct so a test that accidentally mixed
# them would explode loudly.
_CAM_EPOCH0 = 1_700_000_000_000_000_000


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _nav(t_ns, *, yaw=0.0):
    """A level NavState past the cold-start (finite tsv releases the anchor if ever needed)."""
    return NavState(
        sim_time_ns=int(t_ns),
        position_ned=np.array([0.0, 0.0, -2.5]),
        velocity_ned=np.zeros(3),
        roll=0.0,
        pitch=0.0,
        yaw=float(yaw),
        time_since_vision_update_s=0.05,
    )


def _pose(theta, r, cam_t_ns, frame_id=0):
    """A synthetic GatePose whose gate sits at world LOS angle ``theta`` (rad, horizontal) and
    range ``r`` from a LEVEL identity-attitude drone: with R_wb = I the seeker's
    ``_gate_dir_world`` returns exactly ``[cos(theta), sin(theta), 0]`` for this lever, so
    ``yaw_des == theta`` and the vertical channel stays quiet (world lever z == 0)."""
    d_world = np.array([np.cos(theta), np.sin(theta), 0.0])
    t_cam = R_camera_from_body() @ (float(r) * d_world)
    return GatePose(frame_id=int(frame_id), sim_time_ns=int(cam_t_ns),
                    R_cam_gate=np.eye(3), t_cam_gate=t_cam, reproj_error_px=0.0)


def _pursuit_cfg(**overrides):
    """A GateSeekerConfig with every ramp/slew inert so a direct ``_visual_pursuit_command``
    sequence is a pure function of (pose, nav) — the scripted-pursuit harness."""
    cfg = GateSeekerConfig(launch_ramp_s=0.0, pursuit_ramp_s=0.0, forward_ramp_s=0.0,
                           pursuit_yaw_slew_rps=0.0, settle_s=0.0, use_spawn_egress=False)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _drive(seeker, thetas, rs, dt_s=0.1):
    """Feed a scripted pursuit sequence (theta_k, r_k) at dt_s cadence; IMU epoch = k*dt, camera
    epoch = _CAM_EPOCH0 + k*dt (distinct epochs, same rate). Returns the command list."""
    cmds = []
    for k, (th, r) in enumerate(zip(thetas, rs)):
        t = int(k * dt_s * 1e9)
        cmds.append(seeker._visual_pursuit_command(
            _nav(t), _pose(th, r, _CAM_EPOCH0 + t, frame_id=k)))
    return cmds


def _last_a_vec(seeker):
    """The composed pursuit accel vector, reconstructed from the bridge cache
    (``_record_last_demand(_unit(a_vec), yaw, |a_vec|)`` on the A29 path)."""
    return np.asarray(seeker._last_demand_los, dtype=np.float64) * float(seeker._last_demand_accel)


def _e_t(theta):
    """Horizontal unit perpendicular to the LOS at world angle theta (the +e_t of the spec)."""
    return np.array([-np.sin(theta), np.cos(theta), 0.0])


# ===========================================================================
# 1. OFF => byte-identical: the pre-A29 command stream + bridge cache, bit for bit
# ===========================================================================
def test_los_rate_damping_off_is_byte_identical():
    """Flag OFF (the default; VQ1 / case-A): a scripted pursuit sequence produces EXACTLY the
    pre-A29 command stream — reconstructed here by building the legacy Setpoint by hand
    (``accel_ned = forward_accel * los``, launch_ramp = eff_ramp, velocity_ned None) and driving
    an identical controller instance — and the bridge cache holds EXACTLY the legacy contents
    (los, yaw, forward_accel). The LOS-rate filter must never even engage."""
    assert GateSeekerConfig().use_los_rate_damping is False   # ships OFF
    cfg = _pursuit_cfg()
    seeker = GateSeeker(config=cfg)
    geom = GateSeeker(config=_pursuit_cfg())    # stateless geometry/caps helper for the reference
    ref_ctrl = make_seeker_controller()         # same default gains; fed the identical sequence
    thetas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    dt = 0.1
    for k, th in enumerate(thetas):
        t = int(k * dt * 1e9)
        nav = _nav(t)
        pose = _pose(th, 10.0, _CAM_EPOCH0 + t, frame_id=k)
        cmd = seeker._visual_pursuit_command(nav, pose)
        # --- the pre-A29 reference, built from first principles -------------
        gdir = geom._gate_dir_world(nav, pose)
        los = _unit(np.array([gdir[0], gdir[1], 0.0]),
                    fallback=np.array([1.0, 0.0, 0.0]))
        yaw_des = float(np.arctan2(los[1], los[0]))
        sp = Setpoint(sim_time_ns=t, accel_ned=cfg.forward_accel_mps2 * los,
                      velocity_ned=None, yaw=yaw_des, launch_ramp=1.0)
        ref = ref_ctrl.command(nav, sp)
        ref = geom._cap_yaw_rate(ref, cfg.visual_yaw_rate_cap_rps)
        ref = geom._cap_roll_rate(ref, cfg.pursuit_roll_rate_cap_rps)
        ref = geom._cap_pitch_rate(ref, cfg.pursuit_pitch_rate_cap_rps)
        assert cmd.mode is ControlMode.BODY_RATE
        np.testing.assert_array_equal(cmd.body_rate, ref.body_rate)
        assert cmd.thrust == ref.thrust
        # bridge cache: EXACT legacy contents (unit LOS + scalar forward accel), bit for bit.
        np.testing.assert_array_equal(seeker._last_demand_los, los)
        assert seeker._last_demand_yaw == yaw_des
        assert seeker._last_demand_accel == cfg.forward_accel_mps2
        assert seeker._last_demand_t_ns == t
    # the A29 filter never engaged on the OFF path (no state, no instrumentation).
    assert seeker._los_rate_valid is False
    assert seeker._los_prev_pose_ns is None
    assert seeker._last_los_rate is None and seeker._last_vt_est is None
    assert seeker._last_alat is None


def test_on_differs_from_off_when_los_sweeps():
    """Sanity that the byte-identity test has teeth: the SAME scripted sweeping sequence with the
    flag ON produces a DIFFERENT command once the LOS-rate estimate is valid (the lateral brake
    engages), so a silent always-off regression cannot pass both tests."""
    thetas = list(np.arange(0.0, 0.6, 0.05))     # 0.5 rad/s sweep at 0.1 s cadence
    rs = [10.0] * len(thetas)
    off = GateSeeker(config=_pursuit_cfg(use_los_rate_damping=False))
    on = GateSeeker(config=_pursuit_cfg(use_los_rate_damping=True))
    cmd_off = _drive(off, thetas, rs)[-1]
    cmd_on = _drive(on, thetas, rs)[-1]
    assert not np.array_equal(cmd_on.body_rate, cmd_off.body_rate)
    assert on._los_rate_valid and abs(on._last_alat) > 0.0


# ===========================================================================
# 2. THE SIGN TEST (safety-critical): the lateral demand must BRAKE the tangential velocity
# ===========================================================================
def test_orbit_demands_braking_accel():
    """Spec §4.1 derivation pin: synthetic fresh poses sweeping theta_dot > 0 at fixed r =>
    the composed a_vec has a POSITIVE component along +e_t (= +kd*r*theta_dot*e_t, the braking
    direction), and the demand norm respects total_accel_cap_mps2. A flipped sign here TIGHTENS
    the A29 orbit — this test is the abort-criterion firewall."""
    cfg = _pursuit_cfg(use_los_rate_damping=True)
    seeker = GateSeeker(config=cfg)
    omega = 0.5                          # rad/s, theta_dot > 0
    thetas = [k * omega * 0.1 for k in range(8)]
    _drive(seeker, thetas, [10.0] * 8)
    # theta linear => every sample = omega exactly => EMA == omega.
    assert seeker._los_rate_valid
    assert seeker._los_rate_ema == pytest.approx(omega, rel=1e-9)
    # v_t = -r*theta_dot = -5 m/s ; a_lat = -kd*v_t = +4 -> clipped to +2.0 along +e_t.
    assert seeker._last_vt_est == pytest.approx(-10.0 * omega, rel=1e-9)
    assert seeker._last_alat == pytest.approx(+cfg.lateral_accel_cap_mps2)
    a_vec = _last_a_vec(seeker)
    et = _e_t(thetas[-1])
    assert float(a_vec @ et) == pytest.approx(+cfg.lateral_accel_cap_mps2, rel=1e-9), \
        "theta_dot>0 must demand accel along +e_t (the braking direction) — SIGN IS SAFETY-CRITICAL"
    assert float(np.linalg.norm(a_vec)) <= cfg.total_accel_cap_mps2 + 1e-9


def test_braking_opposes_true_tangential_velocity_both_directions():
    """True-kinematics cross-check (the run-20260703_024023 form of the sign pin): a FIXED gate,
    a drone coasting PERPENDICULAR to the LOS — the lateral demand must OPPOSE the coast, in BOTH
    drift directions. The forward term is exactly orthogonal to e_t, so a_vec @ e_t isolates the
    damping term: sign(a_vec @ e_t) must be -sign(v @ e_t)."""
    gate = np.array([10.0, 0.0, 0.0])   # 10 m north of the drone's start, same height
    for v_east in (+3.0, -3.0):         # coast east, then coast west
        seeker = GateSeeker(config=_pursuit_cfg(use_los_rate_damping=True))
        thetas, rs = [], []
        for k in range(8):
            p = np.array([0.0, v_east * (k * 0.1), 0.0])   # the drone's true position
            rel = gate - p
            thetas.append(float(np.arctan2(rel[1], rel[0])))
            rs.append(float(np.linalg.norm(rel)))
        _drive(seeker, thetas, rs)
        assert seeker._los_rate_valid
        et = _e_t(thetas[-1])
        v = np.array([0.0, v_east, 0.0])
        v_t_true = float(v @ et)
        a_vec = _last_a_vec(seeker)
        a_t = float(a_vec @ et)
        assert a_t * v_t_true < 0.0, (
            f"lateral demand ({a_t:+.2f} along e_t) must OPPOSE the true tangential velocity "
            f"({v_t_true:+.2f}) — a same-sign result is the orbit-TIGHTENING flipped sign")
        # and the estimator's v_t agrees in sign with the true v.e_t (the §5.6 abort telemetry).
        assert seeker._last_vt_est * v_t_true > 0.0


def test_caps_and_deadband():
    """(a) DEADBAND: a near-dead bearing (|v_t| < tangential_deadband_mps) commands NO lateral
    term — a_vec is exactly the forward feedforward (no hunting on noise). (b) NORM CAP: with a
    larger forward demand the composed vector is norm-capped at total_accel_cap_mps2, and the
    lateral term alone respects lateral_accel_cap_mps2."""
    # (a) deadband: omega = 0.002 rad/s at r=10 -> |v_t| = 0.02 < 0.3.
    cfg = _pursuit_cfg(use_los_rate_damping=True)
    s = GateSeeker(config=cfg)
    thetas = [k * 0.002 * 0.1 for k in range(6)]
    _drive(s, thetas, [10.0] * 6)
    assert s._los_rate_valid
    assert abs(s._last_vt_est) < cfg.tangential_deadband_mps
    assert s._last_alat == 0.0
    a_vec = _last_a_vec(s)
    los = np.array([np.cos(thetas[-1]), np.sin(thetas[-1]), 0.0])
    np.testing.assert_allclose(a_vec, cfg.forward_accel_mps2 * los, atol=1e-12)
    # (b) norm cap: forward 2.0 + lateral clipped 2.0 -> raw norm 2.83 -> capped to 2.5.
    cfg2 = _pursuit_cfg(use_los_rate_damping=True, forward_accel_mps2=2.0)
    s2 = GateSeeker(config=cfg2)
    _drive(s2, [k * 0.5 * 0.1 for k in range(8)], [10.0] * 8)
    assert abs(s2._last_alat) == pytest.approx(cfg2.lateral_accel_cap_mps2)
    assert float(np.linalg.norm(_last_a_vec(s2))) == pytest.approx(cfg2.total_accel_cap_mps2)


# ===========================================================================
# 3. SAMPLE GATES: ZOH re-feed, long-gap reset, gate-hop rejection
# ===========================================================================
def test_los_rate_sample_gates():
    s = GateSeeker(config=_pursuit_cfg(use_los_rate_damping=True))
    ns = 1_000_000_000
    # seed + first sample: 0.05 rad over 0.1 s = 0.5 rad/s, seeds the EMA directly.
    s._update_los_rate(0.00, _CAM_EPOCH0 + 0)
    assert s._los_rate_valid is False              # one point is not a rate
    s._update_los_rate(0.05, _CAM_EPOCH0 + int(0.1 * ns))
    assert s._los_rate_valid and s._los_rate_ema == pytest.approx(0.5)
    # ZOH re-feed (same pose_ns): NOT a new geometry sample — nothing moves.
    s._update_los_rate(0.10, _CAM_EPOCH0 + int(0.1 * ns))
    assert s._los_rate_ema == pytest.approx(0.5)
    assert s._los_prev_angle == pytest.approx(0.05)
    # gate hop: a 1.0 rad step over 0.1 s = 10 rad/s > los_rate_max_rps=3 -> sample REJECTED
    # (EMA + validity untouched; the prev anchor moves to the hopped angle).
    s._update_los_rate(1.05, _CAM_EPOCH0 + int(0.2 * ns))
    assert s._los_rate_valid and s._los_rate_ema == pytest.approx(0.5)
    assert s._los_prev_angle == pytest.approx(1.05)
    # long gap: dt = 0.7 s > los_rate_max_dt_s=0.5 -> the filter RESTARTS (stale geometry).
    s._update_los_rate(1.10, _CAM_EPOCH0 + int(0.9 * ns))
    assert s._los_rate_valid is False and s._los_rate_ema == 0.0
    # first sample after the restart seeds the EMA directly again.
    s._update_los_rate(1.15, _CAM_EPOCH0 + int(0.95 * ns))
    assert s._los_rate_valid and s._los_rate_ema == pytest.approx(1.0)   # 0.05 rad / 0.05 s


def test_los_rate_ema_filters_and_wraps():
    """The EMA blends samples at los_rate_ema_alpha, and the angle difference is wrap-safe
    (a yaw_des crossing +pi/-pi must not read as a ~2*pi/dt sample)."""
    s = GateSeeker(config=_pursuit_cfg(use_los_rate_damping=True))
    ns = 1_000_000_000
    s._update_los_rate(0.0, _CAM_EPOCH0)
    s._update_los_rate(0.05, _CAM_EPOCH0 + int(0.1 * ns))     # seeds at 0.5
    s._update_los_rate(0.15, _CAM_EPOCH0 + int(0.2 * ns))     # sample 1.0 -> 0.4*1.0+0.6*0.5
    assert s._los_rate_ema == pytest.approx(0.4 * 1.0 + 0.6 * 0.5)
    # wrap: pi-0.02 -> -(pi-0.02) is a +0.04 shortest-path step, NOT ~ -2*pi.
    w = GateSeeker(config=_pursuit_cfg(use_los_rate_damping=True))
    w._update_los_rate(np.pi - 0.02, _CAM_EPOCH0)
    w._update_los_rate(-(np.pi - 0.02), _CAM_EPOCH0 + int(0.1 * ns))
    assert w._los_rate_valid
    assert w._los_rate_ema == pytest.approx(0.04 / 0.1, rel=1e-6)


# ===========================================================================
# 4. RESETS: pass begin/end, track drop, and reset() all drop the one-gate LOS history
# ===========================================================================
def _seed_valid_los(s):
    s._update_los_rate(0.0, _CAM_EPOCH0)
    s._update_los_rate(0.05, _CAM_EPOCH0 + 100_000_000)
    assert s._los_rate_valid


def _assert_los_reset(s, why):
    assert s._los_rate_valid is False, why
    assert s._los_rate_ema == 0.0, why
    assert s._los_prev_angle is None and s._los_prev_pose_ns is None, why


def test_los_rate_resets_on_pass_and_track_drop():
    cfg = _pursuit_cfg(use_los_rate_damping=True)
    s = GateSeeker(config=cfg)
    # pass BEGIN (the dead-reckon commit): the state described the just-passed gate.
    _seed_valid_los(s)
    s._begin_pass(123)
    _assert_los_reset(s, "_begin_pass must drop the LOS-rate state")
    # pass END (next gate re-acquired): a NEW gate's geometry begins.
    _seed_valid_los(s)
    s._end_pass()
    _assert_los_reset(s, "_end_pass must drop the LOS-rate state")
    # full seeker reset.
    _seed_valid_los(s)
    s.reset()
    _assert_los_reset(s, "reset() must drop the LOS-rate state")

    # track DROP (coast-ticks expiry): drive detect_gate_lever with an empty detector until the
    # temporal track is dropped — the LOS state goes with the tracked-gate identity.
    class _EmptyDetector:
        def detect(self, frame):
            return []

    cfg2 = _pursuit_cfg(use_los_rate_damping=True, track_max_coast_ticks=1)
    s2 = GateSeeker(config=cfg2, detector=_EmptyDetector())
    s2._track_range_m, s2._track_bearing = 10.0, np.zeros(2)   # a live track
    _seed_valid_los(s2)
    img = np.zeros((8, 8, 3), np.uint8)
    f1 = Frame(frame_id=1, sim_time_ns=_CAM_EPOCH0, image_bgr=img, recv_monotonic_ns=0)
    f2 = Frame(frame_id=2, sim_time_ns=_CAM_EPOCH0 + 1, image_bgr=img, recv_monotonic_ns=0)
    assert s2.detect_gate_lever(f1) is None        # coast 1 (<= max) -> track survives
    assert s2._track_range_m is not None and s2._los_rate_valid
    assert s2.detect_gate_lever(f2) is None        # coast 2 (> max) -> track DROPPED
    assert s2._track_range_m is None
    _assert_los_reset(s2, "track drop must drop the LOS-rate state")


# ===========================================================================
# 5. CONTINUOUS EPOCH RECONCILIATION (navigator): rate-skew bounded ON, learn-once OFF
# ===========================================================================
_HOVER_ACCEL = np.array([0.0, 0.0, -9.80665])
_LEVEL_Q = np.array([1.0, 0.0, 0.0, 0.0])


def _ds(sim_time_ns, recv_monotonic_ns):
    return DroneState(
        sim_time_ns=int(sim_time_ns), recv_monotonic_ns=int(recv_monotonic_ns),
        orientation_ned_wxyz=_LEVEL_Q.copy(), accel_body=_HOVER_ACCEL.copy(),
        position_ned=None, velocity_ned=None,
    )


class _NoGateDetector:
    """A detector that sees nothing — the epoch learn happens before detection, so this drives
    the delta_epoch bookkeeping without any vision fixes."""

    def detect(self, frame):
        return []


def _run_skewed_epochs(continuous: bool, *, cam_rate=0.95, n=200, dt_s=0.1):
    """Drive n paired (frame, ds) ticks where the camera epoch runs at ``cam_rate`` x the IMU/wall
    rate (the A29 GPU-load skew): IMU t_k = k*dt (== wall/recv), camera stamp = C0 + cam_rate*t_k.
    Returns (nav, t_final_ns, cam_final_ns)."""
    nav = Navigator(gates=[], detector=_NoGateDetector(),
                    config=NavigatorConfig(use_given_position=False, use_given_velocity=False,
                                           reconcile_vision_clock_continuous=continuous))
    t = cam = 0
    for k in range(n + 1):
        t = int(k * dt_s * 1e9)
        cam = _CAM_EPOCH0 + int(round(cam_rate * t))
        img = np.zeros((8, 8, 3), np.uint8)
        nav.update(_ds(t, t), Frame(frame_id=k, sim_time_ns=cam, image_bgr=img,
                                    recv_monotonic_ns=t))
    return nav, t, cam


def test_continuous_epoch_tracks_rate_skew():
    """Camera epoch at 0.95x the IMU epoch (the measured A29 failure mode, exaggerated slightly):
    the learn-once delta drifts UNBOUNDED (~0.05 s per second — 1.0 s over this 20 s run, past
    the 0.5 s RewindKF horizon => every fix dropped), while the continuous EMA keeps the
    reconstructed capture time bounded (steady-state lag ~ (1-a)/a * per-tick drift ~ 45 ms)."""
    # OFF (learn-once): the reconstructed capture instant of the FINAL frame is ~1.0 s wrong.
    nav_off, t_end, cam_end = _run_skewed_epochs(continuous=False)
    err_off = abs(nav_off.camera_epoch_to_imu_ns(cam_end) - t_end) / 1e9
    assert err_off > 0.5, f"learn-once must drift unbounded under rate skew (got {err_off:.3f} s)"
    # ON (continuous EMA): bounded, well under the 0.5 s rewind horizon.
    nav_on, t_end, cam_end = _run_skewed_epochs(continuous=True)
    err_on = abs(nav_on.camera_epoch_to_imu_ns(cam_end) - t_end) / 1e9
    assert err_on < 0.1, f"continuous reconciliation must bound the epoch error (got {err_on:.3f} s)"
    assert err_on < err_off / 5.0


def test_continuous_epoch_converges_to_constant_offset():
    """No skew (1:1 rates, constant offset): the continuous EMA is a fixed point at the true
    delta — it must stay EXACTLY the learn-once value (EMA of a constant, integer-rounded)."""
    nav, t_end, cam_end = _run_skewed_epochs(continuous=True, cam_rate=1.0, n=50)
    assert nav._delta_epoch_ns == _CAM_EPOCH0
    assert nav.camera_epoch_to_imu_ns(cam_end) == t_end


def test_off_learns_delta_exactly_once():
    """Flag OFF (the default): ``_delta_epoch_ns`` is set ONCE from the first paired tick and
    never rewritten — the pre-A29 learn-once behaviour, byte-identical."""
    assert NavigatorConfig().reconcile_vision_clock_continuous is False   # ships OFF
    nav = Navigator(gates=[], detector=_NoGateDetector(),
                    config=NavigatorConfig(use_given_position=False, use_given_velocity=False))
    img = np.zeros((8, 8, 3), np.uint8)
    nav.update(_ds(0, 0))                                       # tick 0: initialize only
    nav.update(_ds(100_000_000, 100_000_000),
               Frame(frame_id=0, sim_time_ns=_CAM_EPOCH0 + 100_000_000, image_bgr=img,
                     recv_monotonic_ns=100_000_000))
    first = nav._delta_epoch_ns
    assert first == _CAM_EPOCH0                                 # learned from the first paired tick
    # subsequent ticks would re-estimate a DIFFERENT delta (skewed stamps) — OFF must ignore them.
    nav.update(_ds(1_000_000_000, 1_000_000_000),
               Frame(frame_id=1, sim_time_ns=_CAM_EPOCH0 + 950_000_000, image_bgr=img,
                     recv_monotonic_ns=1_000_000_000))
    assert nav._delta_epoch_ns == first, "learn-once must not be rewritten when the flag is OFF"


# ===========================================================================
# 6. PROFILE WIRING: vq2_case_c carries A29; vq1_case_a stays byte-identical
# ===========================================================================
def test_profile_wiring_a29():
    ov = get_profile("vq2_case_c").seeker_overrides
    assert ov["use_los_rate_damping"] is True
    assert ov["pursuit_yaw_slew_rps"] == 1.5
    assert vq2_case_c().nav_config.reconcile_vision_clock_continuous is True
    # VQ1 / case-A: no overrides, every A29 flag at its OFF default.
    assert vq1_case_a().seeker_overrides is None
    assert vq1_case_a().controller_overrides is None
    assert vq1_case_a().nav_config.reconcile_vision_clock_continuous is False
    cfg = GateSeekerConfig()
    assert cfg.use_los_rate_damping is False
    assert cfg.pursuit_yaw_slew_rps == 1.0
    # the shipped A29 tunables (the spec §4.1 values) — a silent default drift is a flight change.
    assert cfg.tangential_kd == 0.8
    assert cfg.lateral_accel_cap_mps2 == 2.0
    assert cfg.total_accel_cap_mps2 == 2.5
    assert cfg.tangential_deadband_mps == 0.3
    assert cfg.los_rate_ema_alpha == 0.4
    assert cfg.los_rate_max_rps == 3.0
    assert cfg.los_rate_max_dt_s == 0.5
    assert cfg.los_range_cap_m == 25.0
    assert NavigatorConfig().reconcile_epoch_ema_alpha == 0.10


def test_make_seeker_threads_a29_for_vq2_case_c_only():
    """A seeker built from vq2_case_c overrides has LOS-rate damping ON + the 1.5 slew; one from
    vq1_case_a (no overrides) keeps both at the OFF/legacy defaults (mirrors rl.fly_rl.make_seeker)."""
    def _seeker(name):
        return GateSeeker(config=GateSeekerConfig(
            cruise_speed=3.0, **(get_profile(name).seeker_overrides or {})))
    assert _seeker("vq2_case_c").config.use_los_rate_damping is True
    assert _seeker("vq2_case_c").config.pursuit_yaw_slew_rps == 1.5
    assert _seeker("vq1_case_a").config.use_los_rate_damping is False
    assert _seeker("vq1_case_a").config.pursuit_yaw_slew_rps == 1.0
