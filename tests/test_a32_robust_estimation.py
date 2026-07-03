"""A32 — robust AHRS ("always find down") + soft-weighted vision fusion.

Spec: handoff/vq2_a32_robust_estimation_spec_2026-07-03.md. Evidence run 20260703_172104:
the A8 ``accel_motion_reject`` covariance inflation is an UNBOUNDED, ATTITUDE-REFERENCED
self-locking distrust loop (median 671x in normal flight, 4950-10000x railed in the inverted
tail while |a| = 9.81 m/s^2 exactly implied 180 deg roll) — the gravity pull was effectively
OFF and the filter never recovered "down". The A31 bearing gate hard-rejected 59.9% of
evaluated frames (median rejected frame 1.87x over the allowance); the A28 vertical gate
hard-rejected 130 ticks over a 2 m cliff and reseed-teleported.

THE FIXES (all vq2_case_c-gated, defaults OFF = byte-identical):
  * ESKF ``use_accel_trust_v2``: bounded inflation (25x cap, 1.0 m/s^2 knee), R_ref free-run
    time limit (1 s), chi2 hard gate -> Huber-soft, TOTAL deweight cap 100x whenever |a| ~ g
    (the structural ~0.6 s recovery guarantee), gravity-recovery watchdog (25 deg / 0.5 s
    persistent disagreement -> P bump + R_ref re-anchor, 1 s cooldown).
  * ``NavigatorConfig.ahrs_imu_rate_ingest`` + ``MavlinkClient.imu_ring``: step the ESKF on
    the FULL ~185 Hz HIGHRES_IMU stream (kills the +45 deg/tick contact-spike aliasing, F3).
  * ``GateSeekerConfig.use_soft_bearing_weight``: the A31 binary gate consequence becomes a
    Cauchy weight w = 1/(1 + (dev/allow)^2 + (range jump/max)^2) scaling the track EMA, the
    image-servo az term and the z_off latch — every frame contributes, starvation impossible
    by construction, coast/track-drop only on PERSISTENT w < 0.1.
  * ``VerticalEstimator.use_soft_innov_weight``: Huber-weighted alpha-beta correction
    (sigma 0.7 m, k=2, never 0) with reseed-on-persistence RETAINED (miss at nu > 4).

OFFLINE GATE (passed before commit; handoff/tools/a32_eskf_replay.py --v2 [--imu-rate]): the
20260703_172104 tail (t >= 28.16 s, at rest INVERTED) converges to roll 180 +/- 10 deg by
t <= 29.2 s — latest-sample mode recovers at t=28.34 s (final roll +175.4 deg), imu-rate mode
at t=28.41 s (final roll +180.0 deg). It previously sat at ~150 deg forever.

[VQ2 A32, 2026-07-03]
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from racer.ahrs.ahrs_adapter import AHRSAttitudeSource  # noqa: E402
from racer.ahrs.eskf import ESKFAHRS, GRAVITY, _quat_to_R_wxyz  # noqa: E402
from racer.contracts import DroneState, GatePose  # noqa: E402
from racer.frames import euler_from_quat_wxyz  # noqa: E402
from racer.gate_seeker import GateSeeker, GateSeekerConfig  # noqa: E402
from racer.vertical_estimator import VerticalEstimator  # noqa: E402

_NS = 1_000_000_000
_FIXTURE = Path(__file__).resolve().parent / "data" / "vq2_a32_byteid_reference.npz"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _a32_byteid_scenario as scenario  # noqa: E402


# ===========================================================================
# 0. BYTE-IDENTITY (the flag-off guarantee, bit for bit vs the pre-A32 tree)
# ===========================================================================
def test_flag_off_byte_identical_to_pre_a32_reference():
    """THE OFF-PATH PIN: the post-A32 code, with every A32 flag at its default, reproduces the
    PRE-A32 (f96962d) streams BIT FOR BIT: the shared ESKF (bare VQ1 bench config AND the A8
    motion-reject config), the seeker track-continuity stream (VQ1 default AND the frozen A31
    vq2 bundle with the IMU bearing gate ON), and the VerticalEstimator (legacy A25 AND the A28
    zoff filter). The reference fixture was recorded by running tests/_a32_byteid_scenario.py
    against a pristine f96962d checkout (git archive) before the A32 edits."""
    ref = np.load(_FIXTURE)
    got = scenario.record_all()
    assert set(got) == set(ref.files)
    for key in ref.files:
        np.testing.assert_array_equal(
            got[key], ref[key],
            err_msg=f"{key}: flag-off stream changed vs the pre-A32 reference (byte-identity!)")


def test_a32_defaults_are_off():
    """Every A32 flag ships OFF / at its legacy default — the byte-identity precondition."""
    from racer.navigator import NavigatorConfig
    e = ESKFAHRS()
    assert e.use_accel_trust_v2 is False
    assert e.accel_motion_scale == 0.1                 # A8 v1 values UNTOUCHED
    assert e.accel_motion_max_inflate == 1e4
    assert e.accel_motion_anchor_thr == 0.3
    n = NavigatorConfig()
    assert n.ahrs_accel_trust_v2 is False
    assert n.ahrs_imu_rate_ingest is False
    g = GateSeekerConfig()
    assert g.use_soft_bearing_weight is False
    assert g.use_imu_bearing_gate is False
    v = VerticalEstimator()
    assert v.use_soft_innov_weight is False
    assert v.innov_gate_m == 2.0
    # A32 v2 parameter values as specced (only read under the flag).
    assert e.accel_trust_v2_motion_scale == 1.0
    assert e.accel_trust_v2_max_inflate == 25.0
    assert e.accel_trust_v2_anchor_thr == 0.75
    assert e.accel_ref_max_freerun_s == 1.0
    assert e.accel_trust_v2_deweight_cap == 100.0
    assert (e.accel_trust_v2_g_band_lo, e.accel_trust_v2_g_band_hi) == (0.5, 1.5)
    assert e.accel_wd_theta_deg == 25.0
    assert e.accel_wd_hold_s == 0.5
    assert e.accel_wd_p_bump_deg == 20.0
    assert e.accel_wd_cooldown_s == 1.0


def test_vq2_profile_ships_a32():
    from racer.deploy_profile import vq1_case_a, vq2_case_c
    prof = vq2_case_c()
    assert prof.nav_config.ahrs_accel_trust_v2 is True
    assert prof.nav_config.ahrs_imu_rate_ingest is True
    assert prof.seeker_overrides["use_soft_bearing_weight"] is True
    assert prof.seeker_overrides["use_imu_bearing_gate"] is True   # the dev/allow machinery
    assert prof.nav_config.vertical_estimator_overrides["use_soft_innov_weight"] is True
    v1 = vq1_case_a()
    assert v1.nav_config.ahrs_accel_trust_v2 is False
    assert v1.nav_config.ahrs_imu_rate_ingest is False
    assert v1.seeker_overrides is None


# ===========================================================================
# 1. AHRS mechanism pins
# ===========================================================================
def _rest_accel():
    """Specific force of a body at rest, ANY attitude irrelevant to |a|: [0,0,-g] level."""
    return np.array([0.0, 0.0, -GRAVITY])


def _roll_quat(roll_deg):
    a = np.radians(roll_deg) / 2.0
    return np.array([np.cos(a), np.sin(a), 0.0, 0.0])


def _roll_err_deg(q, target_roll_deg):
    r, p, y = euler_from_quat_wxyz(q)
    d = np.degrees(r) - target_roll_deg
    return abs((d + 180.0) % 360.0 - 180.0)


def test_v2_motion_inflation_bounded():
    """(1) The bound: with the reference attitude 150 deg wrong and a CLEAN resting accel
    (|a| = g exactly — the inverted-tail signature), v1 rails at the 1e4 cap (the self-lock);
    v2 is hard-capped at 25x. The gravity pull can be slowed, never severed."""
    wrong = _quat_to_R_wxyz(_roll_quat(150.0))
    v1 = ESKFAHRS(use_accel_motion_reject=True)
    v1._R_ref = wrong.copy()
    assert v1._accel_motion_inflation(_rest_accel()) == pytest.approx(1e4)
    v2 = ESKFAHRS(use_accel_motion_reject=True, use_accel_trust_v2=True)
    v2._R_ref = wrong.copy()
    assert v2._accel_motion_inflation(_rest_accel()) <= 25.0 + 1e-12
    # and the honest-maneuver regime (~2.5 m/s^2 real accel) inflates ~5-10x, not ~600x:
    v2b = ESKFAHRS(use_accel_motion_reject=True, use_accel_trust_v2=True)
    a = _rest_accel() + np.array([2.5, 0.0, 0.0])
    infl = v2b._accel_motion_inflation(a)
    assert 4.0 < infl < 12.0
    v1b = ESKFAHRS(use_accel_motion_reject=True)
    assert v1b._accel_motion_inflation(a) > 400.0


def test_v2_ref_freerun_time_limit():
    """(1b) The 'no permanent grudge' rule: a gyro-anchored reference that has not re-anchored
    within accel_ref_max_freerun_s is FORCE re-anchored to the live estimate before the residual
    is computed — a wrong reference cannot persist past the bound."""
    f = ESKFAHRS(use_accel_motion_reject=True, use_accel_trust_v2=True)
    f._R_ref = _quat_to_R_wxyz(_roll_quat(150.0))      # a drifted/garbage reference
    f._ref_freerun_s = 1.5                             # ... that has free-run past the bound
    infl = f._accel_motion_inflation(_rest_accel())
    np.testing.assert_allclose(f._R_ref, _quat_to_R_wxyz(f.q_wxyz), atol=1e-12)
    assert f._ref_freerun_s == 0.0
    assert infl == pytest.approx(1.0, abs=0.2)         # residual vs the live estimate ~ 0
    # under the bound, the wrong reference IS still consulted (the A8 detection job).
    f2 = ESKFAHRS(use_accel_motion_reject=True, use_accel_trust_v2=True)
    f2._R_ref = _quat_to_R_wxyz(_roll_quat(150.0))
    f2._ref_freerun_s = 0.5
    assert f2._accel_motion_inflation(_rest_accel()) == pytest.approx(25.0)


def test_v2_chi2_huber_soft_never_rejects():
    """(2) chi2 hard gate -> Huber-soft: an innovation over the chi2 line is DOWN-WEIGHTED
    (R * md/thresh), never discarded. v1: the attitude is untouched by the update (hard skip).
    v2: the attitude moves toward the accel-implied tilt — readmission is instant."""
    tilt = GRAVITY * np.array([0.6, 0.0, -0.8])        # |a| = g, direction 37 deg off level
    v1 = ESKFAHRS()                                     # bare: chi2 gate live, no inflation
    q_before = v1.q_wxyz
    v1._update_accel(tilt)
    np.testing.assert_array_equal(v1.q_wxyz, q_before, "harness: v1 must chi2-HARD-reject this")
    v2 = ESKFAHRS(use_accel_trust_v2=True)
    q_before2 = v2.q_wxyz
    v2._update_accel(tilt)
    assert not np.array_equal(v2.q_wxyz, q_before2), "v2 must apply a (down-weighted) update"
    # the accel a_hat = [0.6, 0, -0.8] implies gravity g_b = [-0.6, 0, 0.8] => pitch_acc =
    # atan2(0.6, 0.8) = +36.9 deg; the soft update must move pitch TOWARD it (positive).
    r, p, y = euler_from_quat_wxyz(v2.q_wxyz)
    assert p > 0.001, "the soft update must move pitch TOWARD the accel-implied tilt"


def test_v2_deweight_cap_is_the_recovery_guarantee():
    """(3) THE LOAD-BEARING LINE: watchdog disabled, attitude seeded 150 deg wrong, clean
    resting accel (|a| = g) at 18 Hz. The total deweight is capped at 100 whenever |a| ~ g, so
    the gravity pull is NEVER severed: the error decreases monotonically (per-second windows)
    and re-levels — vs v1, whose railed inflation self-locks the estimate at ~150 deg forever.
    (The FAST recovery in the shipped config comes from the cap + the watchdog together — the
    watchdog test below pins < 1 s; this test proves the structural never-zero pull alone.)"""
    dt = 1.0 / 18.0
    v2 = ESKFAHRS(use_accel_motion_reject=True, use_accel_trust_v2=True,
                  accel_wd_theta_deg=1e9)               # watchdog OFF: prove the cap alone
    v2.reset(_roll_quat(150.0))
    errs = [_roll_err_deg(v2.q_wxyz, 0.0)]
    for k in range(int(8.0 / dt)):
        v2.step(np.zeros(3), _rest_accel(), dt)
        assert v2.last_deweight_total <= 100.0 + 1e-9   # |a|=g in-band: cap always engaged
        if (k + 1) % 18 == 0:
            errs.append(_roll_err_deg(v2.q_wxyz, 0.0))
    assert all(b < a for a, b in zip(errs, errs[1:])), f"error must shrink every second: {errs}"
    assert errs[-1] < 30.0, f"cap-only recovery too slow: {errs}"
    v1 = ESKFAHRS(use_accel_motion_reject=True)
    v1.reset(_roll_quat(150.0))
    for _ in range(int(8.0 / dt)):
        v1.step(np.zeros(3), _rest_accel(), dt)
    assert _roll_err_deg(v1.q_wxyz, 0.0) > 100.0, "v1 self-lock must persist (the A32 bug)"


def test_v2_watchdog_fires_and_recovers_fast():
    """(4) The watchdog: persistent (>0.5 s) gross (>25 deg) in-band disagreement fires ONCE
    (P bump + R_ref re-anchor, cooldown-limited), after which ordinary accel updates close the
    error in 2-3 ticks — 'down' recovered well inside 1 s of the persistence threshold."""
    dt = 1.0 / 18.0
    f = ESKFAHRS(use_accel_motion_reject=True, use_accel_trust_v2=True)
    f.reset(_roll_quat(150.0))
    t_fire, t = None, 0.0
    for _ in range(int(2.0 / dt)):
        f.step(np.zeros(3), _rest_accel(), dt)
        t += dt
        if t_fire is None and f.watchdog_fires > 0:
            t_fire = t
    assert f.watchdog_fires >= 1
    assert t_fire is not None and t_fire <= 0.8         # ~0.5 s persistence + a couple ticks
    assert _roll_err_deg(f.q_wxyz, 0.0) < 5.0
    # after recovery the disagreement is gone -> no repeat fires beyond the transient window.
    fires_after = f.watchdog_fires
    for _ in range(int(1.0 / dt)):
        f.step(np.zeros(3), _rest_accel(), dt)
    assert f.watchdog_fires == fires_after


def test_v2_watchdog_quiet_on_agreement_and_tilted_spawn():
    """Flight check §3.5.4 pinned offline: a LEVEL filter at rest never fires; a TILTED spawn
    with the level-from-accel seed (the VQ2 launch path) starts theta_g ~ 0 -> never fires."""
    dt = 1.0 / 18.0
    f = ESKFAHRS(use_accel_motion_reject=True, use_accel_trust_v2=True)
    for _ in range(60):
        f.step(np.zeros(3), _rest_accel(), dt)
    assert f.watchdog_fires == 0
    # tilted spawn (~18 deg), seeded from the first accel sample exactly as the Navigator does:
    roll = np.radians(18.0)
    sf = np.array([0.0, GRAVITY * np.sin(roll), -GRAVITY * np.cos(roll)])   # tilted rest accel
    src = AHRSAttitudeSource(eskf=ESKFAHRS(use_accel_motion_reject=True,
                                           use_accel_trust_v2=True))
    src.seed(AHRSAttitudeSource.level_seed_from_accel(sf))
    for _ in range(60):
        src.ingest(sf, np.zeros(3), dt)
    assert src.eskf.watchdog_fires == 0
    assert src.eskf.last_theta_g_deg < 5.0
    assert abs(abs(np.degrees(src.euler_rpy[0])) - 18.0) < 2.0  # holds the true tilt magnitude


# ===========================================================================
# 2. IMU-rate ring ingestion
# ===========================================================================
def _imu_msg(time_usec, acc=(0.0, 0.0, -GRAVITY), gyr=(0.0, 0.0, 0.0)):
    m = SimpleNamespace(time_usec=time_usec, xacc=acc[0], yacc=acc[1], zacc=acc[2],
                        xgyro=gyr[0], ygyro=gyr[1], zgyro=gyr[2],
                        xmag=1.0, ymag=2.0, zmag=3.0, abs_pressure=1013.25)
    m.get_type = lambda: "HIGHRES_IMU"
    return m


def test_client_ring_appends_post_sign_samples():
    """MavlinkClient appends EVERY HIGHRES_IMU sample (stamp, accel, POST-gyro_sign gyro) to the
    shared ring and threads the deque reference onto the snapshot. Latest-sample fields are
    untouched (byte-identical for every non-draining consumer)."""
    from racer.mavlink_client import MavlinkClient
    c = MavlinkClient(gyro_sign=(-1.0, -1.0, -1.0))
    for k in range(5):
        c._handle(_imu_msg(1_000 + 5_405 * k, gyr=(0.1 * k, 0.0, -0.2)))
    assert c.state.imu_ring is c.imu_ring
    assert len(c.imu_ring) == 5
    t4, a4, g4 = list(c.imu_ring)[-1]
    assert t4 == c.state.sim_time_ns
    np.testing.assert_array_equal(g4, c.state.gyro_body)          # post-sign, same array values
    np.testing.assert_array_equal(g4, [-0.4, 0.0, 0.2])
    np.testing.assert_array_equal(a4, c.state.accel_body)
    # ring is bounded (64 deep).
    for k in range(100):
        c._handle(_imu_msg(100_000 + 5_405 * k))
    assert len(c.imu_ring) == 64


def _nav_with_ahrs(**cfg_over):
    from racer.navigator import Navigator, NavigatorConfig
    cfg = NavigatorConfig(use_given_position=False, use_given_velocity=False,
                          use_ahrs=True, **cfg_over)
    return Navigator(gates=[], detector=None, config=cfg)


def _ds(t_ns, acc, gyr, ring=None):
    return DroneState(sim_time_ns=int(t_ns), accel_body=np.asarray(acc, dtype=np.float64),
                      gyro_body=np.asarray(gyr, dtype=np.float64), imu_ring=ring)


def test_navigator_drains_the_ring_per_sample():
    """ahrs_imu_rate_ingest: the Navigator steps the ESKF once per NEW ring sample (dt from
    consecutive stamps) instead of once per tick — bit-identical to a reference AHRS stepped on
    the full stream. The instrumentation counter reports the drained count."""
    from collections import deque
    rest = _rest_accel()
    gyro = np.array([0.5, -0.2, 0.1])
    ring = deque(maxlen=64)
    nav = _nav_with_ahrs(ahrs_imu_rate_ingest=True)
    # tick 0 (init/seed) + tick 1 (first ring tick: legacy single step, watermark set).
    t0, t1 = 0, 55_000_000
    ring.append((t0, rest.copy(), gyro.copy()))
    nav.update(_ds(t0, rest, gyro, ring))
    ring.append((t1, rest.copy(), gyro.copy()))
    nav.update(_ds(t1, rest, gyro, ring))
    assert nav._ahrs_imu_ingested == 1
    assert nav._ahrs_imu_last_ns == t1
    # reference filter mirrors the state so far (seed + one 55 ms step).
    ref = AHRSAttitudeSource(eskf=ESKFAHRS(gyro_noise_std=0.01, accel_gate_alpha=10.0))
    ref.seed(AHRSAttitudeSource.level_seed_from_accel(rest))
    ref.ingest(rest, gyro, (t1 - t0) / 1e9)
    np.testing.assert_array_equal(nav._ahrs.q_wxyz, ref.q_wxyz)
    # tick 2: THREE intermediate samples land in the ring -> all three are stepped per-sample.
    ts = [t1 + 18_000_000, t1 + 36_000_000, t1 + 55_000_000]
    gys = [gyro * 1.0, gyro * 2.0, gyro * 0.5]
    for t, g in zip(ts, gys):
        ring.append((t, rest.copy(), g.copy()))
    nav.update(_ds(ts[-1], rest, gys[-1], ring))
    assert nav._ahrs_imu_ingested == 3
    prev = t1
    for t, g in zip(ts, gys):
        ref.ingest(rest, g, (t - prev) / 1e9)
        prev = t
    np.testing.assert_array_equal(nav._ahrs.q_wxyz, ref.q_wxyz)
    # flag OFF: latest-sample-and-hold, ring never read (counter stays None).
    nav_off = _nav_with_ahrs()
    nav_off.update(_ds(t0, rest, gyro, ring))
    nav_off.update(_ds(t1, rest, gyro, ring))
    assert nav_off._ahrs_imu_ingested is None


def test_navigator_ring_fallback_without_ring():
    """No ring on the snapshot (fabricated states / tests) -> the legacy single-sample step,
    same result as flag-off; the tick is never lost."""
    rest = _rest_accel()
    gyro = np.array([0.2, 0.1, -0.3])
    nav_on = _nav_with_ahrs(ahrs_imu_rate_ingest=True)
    nav_off = _nav_with_ahrs()
    for t in (0, 55_000_000, 110_000_000):
        nav_on.update(_ds(t, rest, gyro))
        nav_off.update(_ds(t, rest, gyro))
    np.testing.assert_array_equal(nav_on._ahrs.q_wxyz, nav_off._ahrs.q_wxyz)
    assert nav_on._ahrs_imu_ingested == 1


def test_v2_imu_rate_kills_spike_aliasing():
    """F3 in one picture: a 1-sample gyro spike (~14 rad/s for ~5.4 ms) inside a 55 ms tick.
    Latest-sample-and-hold integrates the spike across the WHOLE tick (~+45 deg step); per-sample
    ingestion integrates it over its real 5.4 ms (~4.3 deg). Both filters flag-identical
    otherwise."""
    rest = _rest_accel()
    spike = np.array([14.0, 0.0, 0.0])
    dt_tick = 0.055
    hold = ESKFAHRS(accel_gate_alpha=10.0)
    hold.step(spike, rest, dt_tick)                     # the aliased flight behaviour
    r_hold = abs(np.degrees(euler_from_quat_wxyz(hold.q_wxyz)[0]))
    rate = ESKFAHRS(accel_gate_alpha=10.0)
    for k in range(10):                                 # 10 samples @ 185 Hz ~ one 55 ms tick
        g = spike if k == 9 else np.zeros(3)
        rate.step(g, rest, dt_tick / 10.0)
    r_rate = abs(np.degrees(euler_from_quat_wxyz(rate.q_wxyz)[0]))
    assert r_hold > 35.0
    assert r_rate < 6.0


# ===========================================================================
# 3. Soft bearing weight (Cauchy) — the A31 starvation fix
# ===========================================================================
def _soft_seeker(**cfg_over):
    kw = dict(use_gate_track=True, use_imu_bearing_gate=True, use_soft_bearing_weight=True)
    kw.update(cfg_over)
    s = GateSeeker(config=GateSeekerConfig(**kw))
    s.detector = object()
    return s


def _pose_cam(az, el, r, t_ns, fid=0):
    d = np.array([np.tan(float(az)), np.tan(float(el)), 1.0])
    t_cam = float(r) * d / np.linalg.norm(d)
    return GatePose(frame_id=int(fid), sim_time_ns=int(t_ns),
                    R_cam_gate=np.eye(3), t_cam_gate=t_cam, reproj_error_px=0.0)


def _detect(s, poses, fid, t_ns):
    from racer.contracts import Frame
    s._valid_poses = lambda frame: poses
    return s.detect_gate_lever(Frame(frame_id=int(fid), sim_time_ns=int(t_ns),
                                     image_bgr=np.ones((4, 4, 3), dtype=np.uint8)))


def test_soft_weight_returns_the_a31_rejected_frame():
    """THE STARVATION KILLER: the exact geometry the A31 binary gate hard-rejects (dev ~1.9x
    over the allowance — the run's MEDIAN rejected frame) is now RETURNED with the predicted
    Cauchy weight w = 1/(1 + nu^2) ~ 0.22, and the track EMA moves by alpha*w."""
    r, dt_s = 6.0, 0.05
    t0, t1 = 0, int(dt_s * _NS)
    prev = _pose_cam(0.02, 0.01, r, t0, fid=0)
    # allowance = 0.06 + 4*0.05/6 = 0.0933; a 0.175 rad hop is ~1.87x over it.
    hop = _pose_cam(0.02 + 0.175, 0.01, r, t1, fid=1)

    hard = _soft_seeker(use_soft_bearing_weight=False)   # the A31 binary path
    hard._append_att_hist(t0, (0.0, 0.0, 0.0))
    assert _detect(hard, [prev], 0, t0) is prev
    hard._append_att_hist(t1, (0.0, 0.0, 0.0))
    assert _detect(hard, [hop], 1, t1) is None, "harness: A31 must hard-reject this frame"

    soft = _soft_seeker()
    soft._append_att_hist(t0, (0.0, 0.0, 0.0))
    assert _detect(soft, [prev], 0, t0) is prev
    track_before = float(soft._track_bearing[0])
    soft._append_att_hist(t1, (0.0, 0.0, 0.0))
    got = _detect(soft, [hop], 1, t1)
    assert got is hop, "the soft path must RETURN the frame (starvation impossible)"
    w = soft._last_bearing_w
    nu = soft._last_bearing_dev_rad / soft._last_bearing_allow_rad
    assert w == pytest.approx(1.0 / (1.0 + nu * nu), rel=1e-9)
    assert 0.15 < w < 0.35                               # ~0.22 at the median rejected frame
    # track EMA moved by alpha*w of the innovation, not alpha, not zero.
    a_eff = soft.config.track_ema_alpha * w
    b_hop = GateSeeker._pose_bearing(hop)[0]
    expect = (1.0 - a_eff) * track_before + a_eff * b_hop
    assert float(soft._track_bearing[0]) == pytest.approx(expect, rel=1e-9)
    assert soft._track_coast_ticks == 0                  # w > 0.1: no coast tick


def test_soft_weight_folds_in_range_jump():
    """The range-jump hard check folds into the SAME weight: nu_r = jump/max_jump, so a frame
    with a clean bearing but a 6 m range jump (nu_r = 1) halves its weight instead of dying."""
    r, dt_s = 8.0, 0.05
    t0, t1 = 0, int(dt_s * _NS)
    s = _soft_seeker()
    s._append_att_hist(t0, (0.0, 0.0, 0.0))
    prev = _pose_cam(0.02, 0.01, r, t0, fid=0)
    assert _detect(s, [prev], 0, t0) is prev
    s._append_att_hist(t1, (0.0, 0.0, 0.0))
    jump = _pose_cam(0.02, 0.01, r + 6.0, t1, fid=1)     # same bearing, 6 m range jump
    got = _detect(s, [jump], 1, t1)
    assert got is jump
    w = s._last_bearing_w
    nu = s._last_bearing_dev_rad / s._last_bearing_allow_rad
    nu_r = 6.0 / s.config.track_max_range_jump_m
    assert w == pytest.approx(1.0 / (1.0 + nu * nu + nu_r * nu_r), rel=1e-9)
    assert w < 0.55


def test_soft_weight_persistent_garbage_still_drops_the_track():
    """Hard consequences on PERSISTENCE only: a stream of grossly-inconsistent frames (w < 0.1)
    ticks the coast counter each frame and drops the track after track_max_coast_ticks — a real
    track loss still ends the track; a single wild frame does not."""
    s = _soft_seeker(track_max_coast_ticks=3)
    t = 0
    s._append_att_hist(t, (0.0, 0.0, 0.0))
    prev = _pose_cam(0.0, 0.0, 6.0, t, fid=0)
    assert _detect(s, [prev], 0, t) is prev
    drops = 0
    for k in range(1, 6):
        t = int(k * 0.05 * _NS)
        s._append_att_hist(t, (0.0, 0.0, 0.0))
        wild = _pose_cam(0.9 * (1 if k % 2 else -1), 0.3, 6.0, t, fid=k)   # ~10x the allowance
        got = _detect(s, [wild], k, t)
        if got is None:
            drops += 1
            break
        assert s._last_bearing_w < 0.1
    assert drops == 1, "persistent gross inconsistency must still drop the track"
    assert s._track_range_m is None and s._last_none_reason == "continuity_reject"


def test_soft_weight_scales_image_servo_and_zoff_latch():
    """The ONE weight, computed once, consumed everywhere the fresh pose is: the image-servo az
    term is multiplied by w (half-weight frame -> half the lateral demand) and the z_off latch
    passes w through to the vertical estimator (zoff_last_w = huber * bearing)."""
    cfg = GateSeekerConfig(use_image_servo_lateral=True, use_soft_bearing_weight=True,
                           launch_ramp_s=0.0, pursuit_ramp_s=0.0, forward_ramp_s=0.0,
                           pursuit_yaw_slew_rps=0.0, settle_s=0.0, use_spawn_egress=False,
                           image_az_deadband_rad=0.0)
    from racer.contracts import NavState
    nav_state = NavState(sim_time_ns=0, position_ned=np.zeros(3), velocity_ned=np.zeros(3),
                         roll=0.0, pitch=0.0, yaw=0.0, time_since_vision_update_s=0.05)
    az = 0.1
    alats, az_errs = {}, {}
    for w in (1.0, 0.5):
        s = GateSeeker(config=cfg)
        s._last_bearing_w = w
        s._visual_pursuit_command(nav_state, _pose_cam(az, 0.0, 8.0, 0, fid=0))
        alats[w], az_errs[w] = s._last_alat, s._last_az_err
    # the az stash is the RAW apparent azimuth (pre-weight, world frame incl. the camera
    # up-tilt), identical for both; the applied lateral is k_az * az * w (deadband 0, unclipped).
    assert az_errs[1.0] == pytest.approx(az_errs[0.5], rel=1e-12)
    for w in (1.0, 0.5):
        assert alats[w] == pytest.approx(cfg.image_kaz_mps2_per_rad * az_errs[w] * w, rel=1e-9)
    assert alats[0.5] == pytest.approx(0.5 * alats[1.0], rel=1e-9)
    # latch pass-through: the estimator's applied weight is huber(=1 for a small innov) * w.
    v = VerticalEstimator(use_zoff_filter=True, use_soft_innov_weight=True)
    v.seed()
    for _ in range(30):
        v.predict(0.0, 0.05)                            # close the bias-capture window
    v.latch_offset(1.0, 0.0)                            # initial lock
    v.latch_offset(1.5, 0.0, weight=0.5)                # innov 0.5 m: huber w = 1
    assert v.zoff_last_w == pytest.approx(0.5)
    assert v.z_off == pytest.approx(1.0 + v.zoff_alpha * 0.5 * 0.5)


# ===========================================================================
# 4. Soft vertical innovation weight (Huber) — the A28 cliff fix
# ===========================================================================
def _soft_vert(**kwargs):
    v = VerticalEstimator(use_zoff_filter=True, use_soft_innov_weight=True, **kwargs)
    v.seed()
    for _ in range(30):
        v.predict(0.0, 0.05)                            # close the bias-capture window
    return v


def test_soft_innov_huber_weights_never_zero():
    """The 2 m cliff becomes a Huber ramp: <= 2 sigma (1.4 m) full weight; a 2.1 m innovation
    (0 under A28) now contributes w = k/nu = 2/(2.1/0.7) = 2/3; an 8 m gross jump still nudges
    at w = 0.175 — never zero, and vz takes the SAME weighted beta correction."""
    v = _soft_vert()
    v.latch_offset(0.0, 0.0)                            # initial lock at 0
    v.latch_offset(1.0, 0.0)                            # nu = 1.43 <= 2: full weight
    assert v.zoff_last_w == pytest.approx(1.0)
    assert v.z_off == pytest.approx(v.zoff_alpha * 1.0)
    z = v.z_off
    innov = 2.1
    v.latch_offset(z + innov, 0.0)                      # the just-over-the-cliff case
    w = 2.0 / (2.1 / 0.7)
    assert v.zoff_last_w == pytest.approx(w, rel=1e-9)
    assert v.z_off == pytest.approx(z + v.zoff_alpha * w * innov, rel=1e-9)
    z2 = v.z_off
    v.latch_offset(z2 + 8.0, 0.0)                       # gross: tiny but NON-ZERO nudge
    w8 = 2.0 / (8.0 / 0.7)
    assert v.zoff_last_w == pytest.approx(w8, rel=1e-9)
    assert v.z_off == pytest.approx(z2 + v.zoff_alpha * w8 * 8.0, rel=1e-9)
    assert v.z_off != z2


def test_soft_innov_reseed_on_persistence_retained():
    """A REAL retarget (gate handoff) still re-locks: reseed_after consecutive nu > zoff_miss_nu
    latches re-lock z_off = z_meas with vz untouched — but each miss latch still nudged the
    state by its small weight (graceful degradation INTO the reseed, no freeze-then-teleport)."""
    v = _soft_vert()
    v.latch_offset(0.0, 0.0)
    zs = [v.z_off]
    target = 9.0                                        # the track retargeted ~9 m away
    for k in range(v.reseed_after):
        v.latch_offset(target, 0.0)
        zs.append(v.z_off)
        if k < v.reseed_after - 1:
            assert v.zoff_miss == k + 1
            assert v.zoff_last_accepted is False
            assert zs[-1] > zs[-2], "each miss latch must still nudge toward the measurement"
    assert v.z_off == pytest.approx(target)             # the reseed re-lock
    assert v.zoff_miss == 0
    # honest small innovations reset the miss counter (no spurious reseed).
    v2 = _soft_vert()
    v2.latch_offset(0.0, 0.0)
    v2.latch_offset(5.0, 0.0)                           # one gross miss
    assert v2.zoff_miss == 1
    v2.latch_offset(v2.z_off + 0.3, 0.0)                # honest again
    assert v2.zoff_miss == 0 and v2.zoff_last_accepted is True


def test_soft_innov_off_path_untouched():
    """use_soft_innov_weight=False keeps the A28 binary gate bit-exact (also covered by the
    byte-identity fixture; this is the direct pin): a 2.1 m innovation leaves the state
    UNTOUCHED and ticks the miss counter."""
    v = VerticalEstimator(use_zoff_filter=True)
    v.seed()
    for _ in range(30):
        v.predict(0.0, 0.05)
    v.latch_offset(0.0, 0.0)
    z0 = v.z_off
    v.latch_offset(2.1, 0.0, weight=0.123)              # weight must be IGNORED here
    assert v.z_off == z0
    assert v.zoff_miss == 1 and v.zoff_last_accepted is False


# ===========================================================================
# 5. Instrumentation exports (additive keys, null off-path)
# ===========================================================================
def test_nav_estimate_a32_keys():
    sys.path.insert(0, str(ROOT / "rl"))
    from fly_rl import _nav_estimate_record
    ns = SimpleNamespace(roll=0.0, pitch=0.0, yaw=0.0, position_ned=np.zeros(3),
                         time_since_vision_update_s=0.1)
    nav = SimpleNamespace(_ahrs=None, _vert_est=None)
    s = SimpleNamespace(sim_time_ns=0, gyro_body_raw=None)
    cmd = SimpleNamespace(body_rate=np.zeros(3), thrust=0.3)
    rec = _nav_estimate_record(ns, nav, s, cmd, gate_index=0, tick_index=0)
    for k in ("theta_g_deg", "accel_deweight_total", "ahrs_watchdog_fired",
              "imu_samples_ingested", "bearing_w", "zoff_w"):
        assert k in rec and rec[k] is None, k
    # populated path: a v2 ESKF that has processed an in-band sample exports finite values.
    eskf = ESKFAHRS(use_accel_trust_v2=True)
    eskf.step(np.zeros(3), _rest_accel(), 0.055)
    nav2 = SimpleNamespace(_ahrs=SimpleNamespace(eskf=eskf, q_wxyz=eskf.q_wxyz),
                           _vert_est=None, _ahrs_imu_ingested=7)
    seeker = SimpleNamespace(_last_bearing_w=0.42)
    rec2 = _nav_estimate_record(ns, nav2, s, cmd, gate_index=0, tick_index=1, seeker=seeker)
    assert rec2["theta_g_deg"] is not None
    assert rec2["accel_deweight_total"] is not None
    assert rec2["ahrs_watchdog_fired"] == 0
    assert rec2["imu_samples_ingested"] == 7
    assert rec2["bearing_w"] == pytest.approx(0.42)
