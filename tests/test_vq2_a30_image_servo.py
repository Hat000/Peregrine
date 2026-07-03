"""A30 — image-proportional lateral servo (roll toward the APPARENT gate) replaces the
UNSTABLE A29 LOS-rate damping.

THE A29 POST-MORTEM (run 20260703_150755, gates=0): the LOS-rate lateral differentiated a noisy,
self-motion-contaminated bearing (corr(raw sample, own yaw rate) = 0.65) and multiplied it by an
untrusted range (track EMA pinned at the 25 m cap on 62% of ticks) -> vt_est hit |58| m/s while
the drone flew 8-10, alat saturated +/-2.0 on 80% of active ticks flipping sign every ~1.15 s
(a clean ~2.1 s limit cycle), total LOS rotation 1685 deg. The A29 synthetic tests passed because
their harness had NO bearing noise, NO pose-age jitter, NO AHRS transients: derivative-of-noise x
range-gain never appeared. Section 4 below is THE TEST A29 LACKED -- a noise-realistic orbit sim
that the OLD law must FAIL and the NEW law must PASS, pinning both the diagnosis and the cure.

THE A30 LAW (gate_seeker.use_image_servo_lateral; spec
handoff/vq2_a30_pursuit_redesign_spec_2026-07-03.md §3):

    az    = wrap(psi_world - yaw_now)        # apparent gate azimuth, +right, per-frame LEVEL
    a_lat = clip(k_az * dead(az, db), +/-1.5) along e_right(yaw_now)   # k_az=8, db=0.03
    a_fwd = forward_accel * ramps * max(cos(az), 0)^2                  # point before pushing
    a_vec = clip_norm(a_fwd*los + a_lat*e_right, total_cap)            # 2.0 in vq2_case_c

with psi_world built from the attitude AT CAPTURE TIME (a ring buffer of per-tick attitudes),
so the omega*age of an aged pose (8-17 deg during a 1-2 rad/s yaw = a 1.2-2.3 m/s^2 false
lateral at k_az=8) never enters az. NO derivative, NO range, NO filter state.

THE SIGN IS SAFETY-CRITICAL: +az = gate appears RIGHT of the nose and e_right(psi) =
[-sin(psi), cos(psi), 0] is the drone's right, so a_lat > 0 rolls TOWARD the apparent gate.
When an orbit sweeps the LOS the yaw servo LAGS (A28 measured: az = 0.208*psi_dot, corr 0.64),
the gate sits off-center in the SWEEP direction, and this same demand points anti-tangential --
the brake. A flipped sign pushes AWAY from the apparent gate and feeds the orbit.

[VQ2 slow-is-smooth, A30, 2026-07-03]
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from racer.contracts import ControlMode, GatePose, NavState, Setpoint  # noqa: E402
from racer.deploy_profile import get_profile, vq1_case_a, vq2_case_c  # noqa: E402
from racer.frames import R_camera_from_body, R_world_from_body  # noqa: E402
from racer.gate_seeker import (  # noqa: E402
    GateSeeker, GateSeekerConfig, _unit, make_seeker_controller,
)

# The camera/server epoch on the live wire is a unix-wall-clock ns epoch, vastly larger than the
# IMU sim-uptime epoch — keep the synthetic epochs distinct so an accidental mix explodes loudly.
_CAM_EPOCH0 = 1_700_000_000_000_000_000
_NS = 1_000_000_000


class _EpochStub:
    """A minimal nav_owner exposing the Navigator's camera->IMU epoch conversion (the same seam
    ``_maybe_latch_z_off`` uses; on the live wire the A29 continuous reconciliation keeps it
    accurate). The A30 ring-buffer lookup converts the pose stamp through this."""

    def camera_epoch_to_imu_ns(self, ns):
        return int(ns) - _CAM_EPOCH0


def _wrap(a):
    return float(np.arctan2(np.sin(a), np.cos(a)))


def _nav(t_ns, *, yaw=0.0):
    """A level NavState past the cold-start (finite tsv releases the anchor when needed)."""
    return NavState(
        sim_time_ns=int(t_ns),
        position_ned=np.array([0.0, 0.0, -2.5]),
        velocity_ned=np.zeros(3),
        roll=0.0,
        pitch=0.0,
        yaw=float(yaw),
        time_since_vision_update_s=0.05,
    )


def _pose_from_capture(theta_world, r, yaw_capture, cam_t_ns, frame_id=0):
    """A synthetic GatePose AS CAPTURED: the gate sits at world azimuth ``theta_world`` (rad,
    horizontal, range ``r``) and the LEVEL drone's yaw at the capture instant was
    ``yaw_capture`` — the camera lever therefore encodes the gate's bearing IN THAT frame
    (t_cam = R_cam_from_body @ R_wb(yaw_capture)^T @ r*d_world). Rotating it back with the
    capture attitude recovers theta_world exactly; rotating with a LATER attitude injects
    +omega*age — the contamination A30's ring buffer removes."""
    d_world = np.array([np.cos(theta_world), np.sin(theta_world), 0.0])
    d_body = R_world_from_body(0.0, 0.0, float(yaw_capture)).T @ d_world
    t_cam = R_camera_from_body() @ (float(r) * d_body)
    return GatePose(frame_id=int(frame_id), sim_time_ns=int(cam_t_ns),
                    R_cam_gate=np.eye(3), t_cam_gate=t_cam, reproj_error_px=0.0)


def _pursuit_cfg(**overrides):
    """A GateSeekerConfig with every ramp/slew/settle inert so a direct
    ``_visual_pursuit_command`` call is a pure function of (pose, nav)."""
    cfg = GateSeekerConfig(launch_ramp_s=0.0, pursuit_ramp_s=0.0, forward_ramp_s=0.0,
                           pursuit_yaw_slew_rps=0.0, settle_s=0.0, use_spawn_egress=False)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _last_a_vec(seeker):
    """The composed pursuit accel vector, reconstructed from the bridge cache
    (``_record_last_demand(_unit(a_vec), yaw, |a_vec|)`` on the A30/A29 vector paths)."""
    return np.asarray(seeker._last_demand_los, dtype=np.float64) * float(seeker._last_demand_accel)


def _e_right(psi):
    """The drone's RIGHT in world NED at yaw ``psi`` (the +a_lat direction)."""
    return np.array([-np.sin(psi), np.cos(psi), 0.0])


# ===========================================================================
# 1. SIGN PINS (safety-critical) + deadband + caps + cos^2 forward scale
# ===========================================================================
def test_gate_right_rolls_right_and_left_rolls_left():
    """The apparent-azimuth sign pin, BOTH directions: a gate seen 0.3 rad RIGHT of the nose
    (az > 0) demands lateral accel along +e_right (toward it — the braking direction when an
    orbit sweeps the LOS); mirrored for a gate on the LEFT. k_az*|az-db| = 8*0.27 = 2.16 clips
    at the 1.5 lateral cap. A same-magnitude opposite-sign result is the orbit-FEEDING flip."""
    for sgn in (+1.0, -1.0):
        cfg = _pursuit_cfg(use_image_servo_lateral=True)
        s = GateSeeker(config=cfg)
        az_true = sgn * 0.3
        pose = _pose_from_capture(az_true, 10.0, 0.0, _CAM_EPOCH0, frame_id=0)
        s._visual_pursuit_command(_nav(0, yaw=0.0), pose)
        assert s._last_az_err == pytest.approx(az_true, abs=1e-9)
        assert s._last_alat == pytest.approx(sgn * cfg.image_lat_cap_mps2), \
            "gate right must roll RIGHT (a_lat along +e_right) — SIGN IS SAFETY-CRITICAL"
        a_vec = _last_a_vec(s)
        assert sgn * float(a_vec @ _e_right(0.0)) > 0.0
        # composed norm respects the total cap.
        assert float(np.linalg.norm(a_vec)) <= cfg.total_accel_cap_mps2 + 1e-9


def test_centered_gate_no_lateral_hunting():
    """|az| below the 0.03 rad deadband -> a_lat exactly 0.0; the demand is the pure forward
    drive (scaled by cos^2(az) ~ 1): no hunting on PnP bearing noise when centered."""
    cfg = _pursuit_cfg(use_image_servo_lateral=True)
    s = GateSeeker(config=cfg)
    pose = _pose_from_capture(0.02, 10.0, 0.0, _CAM_EPOCH0, frame_id=0)
    s._visual_pursuit_command(_nav(0, yaw=0.0), pose)
    assert s._last_alat == 0.0
    a_vec = _last_a_vec(s)
    los = np.array([np.cos(0.02), np.sin(0.02), 0.0])
    expect_fwd = cfg.forward_accel_mps2 * float(np.cos(0.02)) ** 2
    np.testing.assert_allclose(a_vec, expect_fwd * los, atol=1e-12)


def test_deadband_is_continuous():
    """The deadband subtracts (sign(az)*max(|az|-db,0)) — no step at the deadband edge. At
    az = 0.05: a_lat = 8*(0.05-0.03) = 0.16, NOT 8*0.05 = 0.40."""
    cfg = _pursuit_cfg(use_image_servo_lateral=True)
    s = GateSeeker(config=cfg)
    pose = _pose_from_capture(0.05, 10.0, 0.0, _CAM_EPOCH0, frame_id=0)
    s._visual_pursuit_command(_nav(0, yaw=0.0), pose)
    assert s._last_alat == pytest.approx(
        cfg.image_kaz_mps2_per_rad * (0.05 - cfg.image_az_deadband_rad), rel=1e-9)


def test_cos2_forward_pointing_scale():
    """The forward drive scales by max(cos(az),0)^2: full when centered, half at 45 deg, ZERO
    beyond 90 deg (translate-to-close only when pointed — kills the residual tangential feed)."""
    for az_true, want in ((0.0, 1.0), (np.pi / 4, 0.5), (np.deg2rad(100.0), 0.0)):
        cfg = _pursuit_cfg(use_image_servo_lateral=True)
        s = GateSeeker(config=cfg)
        pose = _pose_from_capture(az_true, 10.0, 0.0, _CAM_EPOCH0, frame_id=0)
        s._visual_pursuit_command(_nav(0, yaw=0.0), pose)
        assert s._last_fwd_scale == pytest.approx(want, abs=1e-12)
        if want == 0.0:
            # beyond 90 deg the demand is PURE lateral (the saturated turn) — no forward feed.
            a_vec = _last_a_vec(s)
            los = np.array([np.cos(az_true), np.sin(az_true), 0.0])
            e_r = _e_right(0.0)
            lat = float(a_vec @ e_r)
            np.testing.assert_allclose(a_vec, lat * e_r, atol=1e-12)
            assert lat == pytest.approx(cfg.image_lat_cap_mps2)


def test_total_norm_cap():
    """A large forward demand + a saturated lateral compose above the total cap -> the vector is
    norm-capped (direction preserved), and the lateral alone respects its own 1.5 cap."""
    cfg = _pursuit_cfg(use_image_servo_lateral=True, forward_accel_mps2=2.5,
                       total_accel_cap_mps2=2.0)
    s = GateSeeker(config=cfg)
    pose = _pose_from_capture(0.3, 10.0, 0.0, _CAM_EPOCH0, frame_id=0)   # lat clips at 1.5
    s._visual_pursuit_command(_nav(0, yaw=0.0), pose)
    assert abs(s._last_alat) == pytest.approx(cfg.image_lat_cap_mps2)
    assert float(np.linalg.norm(_last_a_vec(s))) == pytest.approx(cfg.total_accel_cap_mps2)


# ===========================================================================
# 2. CAPTURE-TIME ATTITUDE (the omega*age fix) + ring-buffer mechanics
# ===========================================================================
def test_capture_time_attitude_kills_omega_age():
    """Yaw ramping at 1 rad/s, fixed gate, a pose aged 0.2 s: the ring-buffer rotation reads the
    TRUE current apparent azimuth (|error| < 2 deg); the UNCORRECTED current-attitude rotation
    would read az + omega*age = +11.5 deg wrong (asserted, so this test has teeth)."""
    omega, age, theta_gate = 1.0, 0.2, 1.2
    cfg = _pursuit_cfg(use_image_servo_lateral=True)
    s = GateSeeker(config=cfg, nav_owner=_EpochStub())
    dt = 1.0 / 30.0
    # attitude history: one entry per tick, yaw = omega*t (what command_visual would append).
    for k in range(31):
        t = k * dt
        s._append_att_hist(int(round(t * _NS)), (0.0, 0.0, omega * t))
    t_now, t_cap = 1.0, 1.0 - age
    yaw_now, yaw_cap = omega * t_now, omega * t_cap
    pose = _pose_from_capture(theta_gate, 10.0, yaw_cap,
                              _CAM_EPOCH0 + int(round(t_cap * _NS)), frame_id=0)
    nav = _nav(int(round(t_now * _NS)), yaw=yaw_now)
    s._visual_pursuit_command(nav, pose)
    az_true = _wrap(theta_gate - yaw_now)                       # the honest apparent offset
    assert s._last_az_err == pytest.approx(az_true, abs=np.deg2rad(2.0))
    # the test is MEANINGFUL: the uncorrected rotation is ~omega*age = 0.2 rad = 11.5 deg off.
    geom = GateSeeker(config=_pursuit_cfg())
    gdir_uncorr = geom._gate_dir_world(nav, pose)
    az_uncorr = _wrap(float(np.arctan2(gdir_uncorr[1], gdir_uncorr[0])) - yaw_now)
    assert abs(az_uncorr - az_true) > np.deg2rad(10.0), \
        "harness sanity: without the capture-time fix the azimuth must be ~omega*age wrong"


def test_att_hist_nearest_lookup_gap_guard_and_depth():
    """_rpy_at returns the entry NEAREST the (epoch-converted) pose stamp; a nearest-gap beyond
    image_att_max_gap_s returns None (-> current-attitude fallback); the buffer is bounded at
    image_att_hist_len entries (oldest dropped)."""
    cfg = _pursuit_cfg(use_image_servo_lateral=True, image_att_hist_len=5)
    s = GateSeeker(config=cfg, nav_owner=_EpochStub())
    for k in range(8):                                       # 8 appends into a 5-deep buffer
        s._append_att_hist(k * 100_000_000, (0.0, 0.0, 0.1 * k))
    assert len(s._att_hist) == 5                             # bounded; oldest (k=0,1,2) dropped
    # nearest: pose at 0.52 s (camera epoch) -> nearest entry 0.5 s (yaw 0.5).
    assert s._rpy_at(_CAM_EPOCH0 + 520_000_000) == (0.0, 0.0, pytest.approx(0.5))
    # gap guard: pose 1.4 s past the newest entry (0.7 s) -> gap 0.7 > 0.5 -> None.
    assert s._rpy_at(_CAM_EPOCH0 + 2_100_000_000) is None
    # empty buffer -> None.
    s2 = GateSeeker(config=cfg)
    assert s2._rpy_at(_CAM_EPOCH0) is None


def test_command_visual_appends_attitude_only_when_flag_on():
    """The production append lives in command_visual (one entry per tick, flag-gated): a flag-ON
    seeker records this tick's attitude even on a no-frame hold tick; a flag-OFF seeker
    allocates NOTHING (zero side effects — the VQ1 byte-identity guarantee)."""
    on = GateSeeker(config=_pursuit_cfg(use_image_servo_lateral=True))
    on.command_visual(_nav(123_000_000, yaw=0.4), None, 0)
    assert on._att_hist is not None and len(on._att_hist) == 1
    t_ns, rpy = on._att_hist[0]
    assert t_ns == 123_000_000 and rpy[2] == pytest.approx(0.4)
    off = GateSeeker(config=_pursuit_cfg())
    off.command_visual(_nav(123_000_000, yaw=0.4), None, 0)
    assert off._att_hist is None


def test_reset_clears_a30_state():
    s = GateSeeker(config=_pursuit_cfg(use_image_servo_lateral=True))
    s._append_att_hist(0, (0.0, 0.0, 0.0))
    pose = _pose_from_capture(0.3, 10.0, 0.0, _CAM_EPOCH0, frame_id=0)
    s._visual_pursuit_command(_nav(0, yaw=0.0), pose)
    assert s._att_hist and s._last_az_err is not None and s._last_fwd_scale is not None
    s.reset()
    assert s._att_hist is None
    assert s._last_az_err is None and s._last_fwd_scale is None and s._last_alat is None


# ===========================================================================
# 3. FLAG OFF => BYTE-IDENTICAL (the pre-A30 command stream, bit for bit)
# ===========================================================================
def test_image_servo_off_is_byte_identical():
    """Flag OFF (the default; VQ1 / case-A): a scripted pursuit sequence produces EXACTLY the
    pre-A30 command stream — reconstructed from first principles (the legacy Setpoint:
    accel_ned = forward_accel * los, velocity_ned None) — and NO A30 state ever exists (no
    ring buffer allocated, no stashes written)."""
    assert GateSeekerConfig().use_image_servo_lateral is False   # ships OFF
    cfg = _pursuit_cfg()
    seeker = GateSeeker(config=cfg)
    geom = GateSeeker(config=_pursuit_cfg())    # stateless geometry/caps helper for the reference
    ref_ctrl = make_seeker_controller()
    for k, th in enumerate([0.0, 0.05, 0.10, 0.15, 0.20]):
        t = int(k * 0.1 * _NS)
        nav = _nav(t)
        pose = _pose_from_capture(th, 10.0, 0.0, _CAM_EPOCH0 + t, frame_id=k)
        cmd = seeker._visual_pursuit_command(nav, pose)
        gdir = geom._gate_dir_world(nav, pose)
        los = _unit(np.array([gdir[0], gdir[1], 0.0]), fallback=np.array([1.0, 0.0, 0.0]))
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
    assert seeker._att_hist is None
    assert seeker._last_az_err is None and seeker._last_fwd_scale is None
    assert seeker._last_alat is None


def test_on_differs_from_off_for_off_center_gate():
    """Sanity that the byte-identity test has teeth: the SAME off-center pose with the flag ON
    produces a DIFFERENT command (the lateral engages + the forward yields), so a silent
    always-off regression cannot pass both tests."""
    pose = _pose_from_capture(0.3, 10.0, 0.0, _CAM_EPOCH0, frame_id=0)
    off = GateSeeker(config=_pursuit_cfg())
    on = GateSeeker(config=_pursuit_cfg(use_image_servo_lateral=True))
    cmd_off = off._visual_pursuit_command(_nav(0), pose)
    cmd_on = on._visual_pursuit_command(_nav(0), pose)
    assert not np.array_equal(cmd_on.body_rate, cmd_off.body_rate)
    assert on._last_alat is not None and abs(on._last_alat) > 0.0


# ===========================================================================
# 4. THE TEST A29 LACKED — the noise-realistic orbit sim (diagnosis + cure pin)
# ===========================================================================
# 2D point-mass in the post-gate-0 ACQUIRE-TURN geometry of the real A28/A29 failures: the drone
# coasts EAST at 3 m/s while the next gate sits 10 m due NORTH (90 deg off the nose). Realism the
# A29 harness lacked, all per spec §6.3: pose cadence 70 ms, pose AGE 140 +/- 60 ms jitter,
# bearing noise sigma = 1.5 deg, AHRS yaw-pin steps +/-2 deg every ~2 s, yaw servo kp=4 with a
# 1.5 rad/s rate cap (the lag that generates the az signal). The seeker's composed accel vector
# is applied to the point mass; the run ends at the pass-degenerate range (r < 1.5, where the
# production pass dead-reckon takes over) or t_max.

def _run_orbit_sim(cfg, *, seed, t_max=14.0):
    rng = np.random.default_rng(seed)
    dt = 1.0 / 30.0
    gate = np.array([10.0, 0.0])
    p = np.array([0.0, 0.0])
    v = np.array([0.0, 3.0])                  # 3 m/s carried tangential at r = 10
    psi = np.pi / 2.0                          # nose EAST: the acquire-turn start
    seeker = GateSeeker(config=cfg, nav_owner=_EpochStub())
    seeker._last_yaw = psi                     # heading bookkeeping starts at the coast heading

    cam_period, next_cap = 0.070, 0.0
    pending, current, frame_id = [], None, 0
    yaw_pin, next_pin_flip = np.deg2rad(2.0), 2.0
    hist = {k: [] for k in ("t", "r", "vt", "alat", "az", "fwd", "vtest")}
    t = 0.0
    while t < t_max:
        rel = gate - p
        r = float(np.hypot(rel[0], rel[1]))
        if r < 1.5:
            break                              # pass-degenerate range: production hands off
        if t >= next_pin_flip:                 # AHRS yaw-pin step
            yaw_pin, next_pin_flip = -yaw_pin, next_pin_flip + 2.0
        yaw_meas = psi + yaw_pin
        if t >= next_cap:                      # camera capture (true geometry + bearing noise)
            theta = float(np.arctan2(rel[1], rel[0]))
            theta_n = theta + rng.normal(0.0, np.deg2rad(1.5))
            age = float(np.clip(rng.normal(0.140, 0.060), 0.02, 0.40))
            pose = _pose_from_capture(theta_n, r, psi, _CAM_EPOCH0 + int(round(t * _NS)),
                                      frame_id=frame_id)
            pending.append((t + age, pose))
            frame_id += 1
            next_cap += cam_period
        arrived = [pp for pp in pending if pp[0] <= t]
        if arrived:                            # freshest COMPLETED detection (latest capture)
            current = max(arrived, key=lambda pp: pp[1].sim_time_ns)[1]
            pending = [pp for pp in pending if pp[0] > t]
        t_ns = int(round(t * _NS))
        nav = _nav(t_ns, yaw=yaw_meas)
        a_vec = np.zeros(3)
        if current is not None:
            if seeker.config.use_image_servo_lateral:   # command_visual's per-tick append
                seeker._append_att_hist(t_ns, seeker._att_rpy(nav))
            seeker._visual_pursuit_command(nav, current)
            a_vec = _last_a_vec(seeker)
        # plant: yaw servo kp=4 rate-capped 1.5 tracks the seeker's slewed setpoint; the accel
        # demand is realized directly (the inner tilt loop is much faster than the guidance).
        yaw_sp = seeker._last_yaw if seeker._last_yaw is not None else psi
        psi += float(np.clip(4.0 * _wrap(yaw_sp - yaw_meas), -1.5, 1.5)) * dt
        v = v + a_vec[:2] * dt
        p = p + v * dt
        theta = float(np.arctan2(rel[1], rel[0]))
        e_t = np.array([-np.sin(theta), np.cos(theta)])
        hist["t"].append(t)
        hist["r"].append(r)
        hist["vt"].append(float(v @ e_t))
        hist["alat"].append(seeker._last_alat)
        hist["az"].append(seeker._last_az_err)
        hist["fwd"].append(seeker._last_fwd_scale)
        hist["vtest"].append(seeker._last_vt_est)
        t += dt
    return hist


def _orbit_metrics(hist):
    t = np.array(hist["t"])
    r = np.array(hist["r"])
    vt = np.array(hist["vt"])
    alat = np.array([a if a is not None else 0.0 for a in hist["alat"]])
    az = np.array([a if a is not None else np.nan for a in hist["az"]], dtype=float)
    vte = np.array([x if x is not None else 0.0 for x in hist["vtest"]])
    fwd = np.array([x for x in hist["fwd"] if x is not None], dtype=float)
    out = {}
    # lateral sign-flip rate after the 2 s acquire transient, counting only substantive demands
    # (|alat| >= 0.15 on both sides of the flip) — the limit-cycle signature, not zero-chatter.
    m = t >= 2.0
    a = alat[m]
    flips, last_sign, last_big = 0, 0.0, False
    for x in a:
        s = np.sign(x) if abs(x) > 1e-9 else 0.0
        big = abs(x) >= 0.15
        if s != 0.0 and last_sign != 0.0 and s != last_sign and big and last_big:
            flips += 1
        if s != 0.0:
            last_sign, last_big = s, big
    out["flip_rate"] = flips / max(t[-1] - 2.0, 1e-9)
    i8 = min(int(np.searchsorted(t, 8.0)), len(vt) - 1)
    out["vt_at_8s"] = abs(vt[i8]) if t[-1] >= 7.9 else None
    idx = int(np.argmax(r < 2.0)) if bool(np.any(r < 2.0)) else None
    out["arrival_vt"] = abs(vt[idx]) if idx is not None else None
    out["closed"] = bool(np.any(r < 2.0))
    azl = az[t >= (t[-1] - 2.0)]
    azl = azl[~np.isnan(azl)]
    out["az_p50_last2s_deg"] = float(np.rad2deg(np.median(np.abs(azl)))) if len(azl) else None
    out["max_alat"] = float(np.max(np.abs(alat)))
    out["max_vtest"] = float(np.max(np.abs(vte)))
    out["fwd_in_range"] = bool(len(fwd) == 0 or (np.min(fwd) >= 0.0 and np.max(fwd) <= 1.0))
    return out


def test_orbit_sim_a30_kills_the_orbit():
    """The A30 image servo, in the noise-realistic acquire-turn harness: the 3 m/s carried
    tangential is decisively braked (< 1.0 m/s by t = 8 s), the drone CLOSES to the pass range
    with a threadable cross-speed (< 1.5 m/s at first r < 2), the gate is centered on final
    approach (|az| p50 < 5 deg over the last 2 s), NO lateral limit cycle (substantive sign
    flips < 0.5/s after the acquire transient), and every internal is bounded by construction
    (|alat| <= 1.5, fwd_scale in [0,1]). Three seeds — noise-robust, deterministic."""
    for seed in (0, 1, 2):
        m = _orbit_metrics(_run_orbit_sim(
            _pursuit_cfg(use_image_servo_lateral=True, total_accel_cap_mps2=2.0,
                         pursuit_yaw_slew_rps=1.5), seed=seed))
        assert m["closed"], f"seed {seed}: never closed to the pass range — the orbit persists"
        assert m["vt_at_8s"] is not None and m["vt_at_8s"] < 1.0, \
            f"seed {seed}: carried tangential not killed (|vt|@8s = {m['vt_at_8s']})"
        assert m["arrival_vt"] < 1.5, \
            f"seed {seed}: arrival cross-speed {m['arrival_vt']:.2f} — a fly-by, not a thread"
        assert m["az_p50_last2s_deg"] < 5.0, \
            f"seed {seed}: gate not centered on final approach (az p50 {m['az_p50_last2s_deg']:.1f} deg)"
        assert m["flip_rate"] < 0.5, \
            f"seed {seed}: lateral limit cycle ({m['flip_rate']:.2f} flips/s)"
        assert m["max_alat"] <= 1.5 + 1e-9        # bounded by construction (P6)
        assert m["fwd_in_range"]
        assert m["max_vtest"] == 0.0              # NO A29 internal state on the A30 path


def test_orbit_sim_a29_law_fails_the_same_harness():
    """THE DIAGNOSIS PIN: the OLD A29 LOS-rate law, in the SAME harness with its shipped
    parameters, FAILS the criteria the A30 law passes — its lateral limit-cycles (substantive
    sign flips > 1.0/s vs A30's < 0.5) and it arrives at the gate carrying a fly-by cross-speed
    (> 1.5 m/s) — and its internal vt_est is GARBAGE (> 8 m/s against a true speed that never
    exceeds ~4.5): derivative-of-noise x range-gain, exactly the flight failure mode
    (run 20260703_150755 measured |58| m/s). If a future change makes THIS test's assertions
    fail, the A29 law has been fixed — re-evaluate A30 vs it, do not just delete the pin."""
    for seed in (0, 1, 2):
        m = _orbit_metrics(_run_orbit_sim(
            _pursuit_cfg(use_los_rate_damping=True, pursuit_yaw_slew_rps=1.5), seed=seed))
        assert m["flip_rate"] > 1.0, \
            f"seed {seed}: A29 law did not limit-cycle ({m['flip_rate']:.2f} flips/s)"
        assert m["arrival_vt"] is None or m["arrival_vt"] > 1.5, \
            f"seed {seed}: A29 law arrived threadable ({m['arrival_vt']}) — diagnosis pin broken"
        assert m["max_vtest"] > 8.0, \
            f"seed {seed}: A29 vt_est stayed honest ({m['max_vtest']:.1f} m/s) — noise model lost"


# ===========================================================================
# 5. PROFILE WIRING: vq2_case_c carries A30 (and drops the A29 damping); vq1 untouched
# ===========================================================================
def test_profile_wiring_a30():
    ov = get_profile("vq2_case_c").seeker_overrides
    assert ov["use_image_servo_lateral"] is True
    assert ov["total_accel_cap_mps2"] == 2.0          # 11.5 deg max composed lean (override only)
    assert ov["pursuit_yaw_slew_rps"] == 1.5          # the A29 delay trim, KEPT
    assert "use_los_rate_damping" not in ov           # A29 damping DISABLED (default False)
    # the effective vq2_case_c seeker config: A30 on, A29 lateral off.
    eff = GateSeekerConfig(**ov)
    assert eff.use_image_servo_lateral is True
    assert eff.use_los_rate_damping is False
    # the A29 clock reconciliation + the A28 vertical chain are KEPT.
    prof = vq2_case_c()
    assert prof.nav_config.reconcile_vision_clock_continuous is True
    assert prof.vertical_estimator is True
    assert prof.nav_config.vertical_estimator_overrides == {"use_zoff_filter": True,
                                                            "export_clip_mps": 2.5,
                                                            "use_soft_innov_weight": True,
                                                            "zoff_reseed_min_w": 0.3}  # A33 V-1
    # VQ1 / case-A: byte-identical (no overrides; every A30 field at its OFF/neutral default).
    assert vq1_case_a().seeker_overrides is None
    assert vq1_case_a().controller_overrides is None
    cfg = GateSeekerConfig()
    assert cfg.use_image_servo_lateral is False
    assert cfg.image_kaz_mps2_per_rad == 8.0
    assert cfg.image_az_deadband_rad == 0.03
    assert cfg.image_lat_cap_mps2 == 1.5
    assert cfg.image_att_hist_len == 36
    assert cfg.image_att_max_gap_s == 0.5
    assert cfg.total_accel_cap_mps2 == 2.5            # field default UNCHANGED (A29's value)


def test_make_seeker_threads_a30_for_vq2_case_c_only():
    """A seeker built the fly_rl way from vq2_case_c has the image servo ON with the shipped
    gains; one from vq1_case_a keeps every default OFF (mirrors rl.fly_rl.make_seeker)."""
    def _seeker(name):
        return GateSeeker(config=GateSeekerConfig(
            cruise_speed=3.0, **(get_profile(name).seeker_overrides or {})))
    c = _seeker("vq2_case_c").config
    assert c.use_image_servo_lateral is True
    assert c.use_los_rate_damping is False
    assert c.total_accel_cap_mps2 == 2.0
    a = _seeker("vq1_case_a").config
    assert a.use_image_servo_lateral is False
    assert a.total_accel_cap_mps2 == 2.5
