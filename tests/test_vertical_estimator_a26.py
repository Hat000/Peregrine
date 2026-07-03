"""VQ2 A26 -- vertical brake fix: de-clip washout + gate-offset-rate fusion, 2026-07-02.

Follows A25 (gate-relative z_off altitude term). A25 flew and worked directionally (reversed the
climb, sought gate height) but OVERSHOT past gate-1 height to the floor -- the washout ``vz`` rails
to its export clip and is BLIND to real sustained descent (reads ~0), so nothing ever braked the
descent. See ``handoff/vq2_a26_vertical_brake_2026-07-02.md`` for the full root-cause + fix design.

This file pins:
  (a) the vz_gate SIGN (descending toward the gate => vz_gate > 0, matching the vz down-positive
      convention) -- the one derivation an implementer is most likely to get backwards.
  (b) the dt accept-window gates (too-small / too-large both reject, leaving vz untouched).
  (c) the |vz_gate|>reject-threshold gate (implausible rate leaves vz untouched).
  (d) the first-ever latch is a no-op for fusion (nothing to difference against yet).
  (e) ``use_gate_vz_fusion=False`` is a pure no-op -- VQ1/case-A byte-identical.
  (f) the export clip is now +/-1.5 (FIX 1, de-saturate the damper).
  (g) the deploy-profile flag (vq2_case_c ON, vq1_case_a OFF).
  (h) the ``pose_age_s`` epoch-mismatch fix (camera-epoch vs IMU-epoch conversion) + the
      ``contact_frozen`` NavState plumbing.

THE pose_age_s BUG (root-caused empirically against the flown run 20260702_221235's
nav_estimate.jsonl -- pose_age_s read EXACTLY 0.0 on all 159 non-null ticks despite
offset_z_world visibly changing almost every tick): ``GatePose.sim_time_ns`` (sourced from the
JPEG-wire header) is on the CAMERA/server epoch -- confirmed against that run's own
video_index.jsonl to be UNIX WALL-CLOCK nanoseconds (~1.78e18, i.e. ~year-2026 unix time) -- while
``NavState.sim_time_ns`` (sourced from ``DroneState.sim_time_ns`` / ``HIGHRES_IMU.time_usec``) is
on the IMU epoch, a small SIM-UPTIME clock (~1.9e13, ~19000 seconds). The raw subtraction
``nav.sim_time_ns - pose.sim_time_ns`` was therefore always a HUGE NEGATIVE number (camera epoch
vastly exceeds IMU epoch), and the very next ``max(0.0, ...)`` staleness guard in
``GateSeeker._maybe_latch_z_off`` silently clamped every such tick to exactly 0.0, masking the
real pose age. THE FIX: ``Navigator.camera_epoch_to_imu_ns`` (a new public method mirroring the
existing ``_vision_fix_time_imu_ns`` OOSM-fix-time conversion, which the KF's own RewindKF path
already relies on) converts a raw camera-epoch stamp onto the IMU epoch via the Navigator's
learned ``_delta_epoch_ns``; ``_maybe_latch_z_off`` now calls it (falling back to the raw stamp,
byte-identical, when unavailable -- e.g. same-epoch synthetic/VQ1 tests, or before the first
paired (frame, ds) has landed on a live wire)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from racer.vertical_estimator import VerticalEstimator  # noqa: E402
from racer.contracts import GatePose, NavState  # noqa: E402
from racer.gate_seeker import GateSeeker, GateSeekerConfig  # noqa: E402

IMU_DT = 1.0 / 140.0    # the live HIGHRES_IMU rate (~143 Hz)


def _burn_bias_capture(est: VerticalEstimator, dt: float = IMU_DT) -> None:
    """Advance past the pre-arm bias-capture window at true-zero bias (a_up=0)."""
    n_capture = int(np.ceil(est.bias_capture_s / dt)) + 1
    for _ in range(n_capture):
        est.predict(0.0, dt)


def _advance_clock(est: VerticalEstimator, seconds: float, dt: float = IMU_DT) -> None:
    """Advance the estimator's internal elapsed clock (``_contact_elapsed_s``, the SAME monotonic
    clock the fusion's dt-window measures against) by feeding zero-accel predict ticks -- does not
    move vz (a_up=0 stays within the washout's own leak) but does advance time."""
    n = max(int(round(seconds / dt)), 1)
    for _ in range(n):
        est.predict(0.0, dt)


# ---------------------------------------------------------------------------
# (a) vz_gate SIGN -- descending toward the gate => vz_gate > 0 (PINNED)
# ---------------------------------------------------------------------------
def test_vz_gate_sign_descending_toward_gate_is_positive():
    """Feed two latch_offset calls with a SHRINKING offset_z_world (drone descending toward a
    static gate -- the down-positive drone->gate distance shrinks as the drone closes on it) and
    assert the fused vz moved MORE POSITIVE (the down-positive convention's "descending" sign).
    The shrink is kept small enough (0.2 m over ~0.2 s -> vz_gate=1.0 m/s) to stay well inside the
    +/-2.5 m/s accept window."""
    est = VerticalEstimator(use_gate_vz_fusion=True)
    est.seed()
    _burn_bias_capture(est)

    # first latch: nothing to difference against -- establishes the "previous fresh" bookkeeping.
    est.latch_offset(offset_z_world=5.0, obs_age_s=0.0)
    vz_after_first = est.vz
    assert vz_after_first == pytest.approx(0.0), "setup: vz should still be ~0 before any fusion"

    # advance the internal clock inside the accept window, then latch a SMALLER offset_z_world
    # (the gate is now closer -- descending toward it).
    _advance_clock(est, 0.2)
    vz_before_second = est.vz
    est.latch_offset(offset_z_world=4.8, obs_age_s=0.0)   # shrank 5.0 -> 4.8: vz_gate ~= +1.0 m/s
    assert est.vz > vz_before_second, \
        "a shrinking offset_z_world (descending toward the gate) must fuse a MORE POSITIVE vz"


def test_vz_gate_sign_descending_small_shrink_in_range():
    """Same sign pin as above, but with a shrink small enough that vz_gate stays inside the
    +/-2.5 m/s reject window, so the exact blended value is checkable against the formula
    ``vz <- vz + k*(vz_gate - vz)`` with ``vz_gate = (offset_prev - offset_now)/dt``."""
    est = VerticalEstimator(use_gate_vz_fusion=True)
    est.seed()
    _burn_bias_capture(est)

    est.latch_offset(offset_z_world=2.0, obs_age_s=0.0)
    dt = 0.2
    _advance_clock(est, dt)
    vz_before = est.vz
    # shrink by 0.2 m over 0.2s -> vz_gate = +1.0 m/s (descending, in-range)
    est.latch_offset(offset_z_world=1.8, obs_age_s=0.0)
    vz_gate_expected = (2.0 - 1.8) / dt   # = 1.0
    expected = vz_before + est.k_gate_vz_fusion * (vz_gate_expected - vz_before)
    assert est.vz == pytest.approx(expected, abs=1e-6)
    assert est.vz > vz_before, "descending (shrinking offset) must fuse vz MORE POSITIVE"


def test_vz_gate_sign_climbing_away_is_negative_pull():
    """The mirror check: a GROWING offset_z_world (gate falling further below -- climbing away)
    must pull the fused vz MORE NEGATIVE (or at least not positive), the opposite of the descent
    case."""
    est = VerticalEstimator(use_gate_vz_fusion=True)
    est.seed()
    _burn_bias_capture(est)

    est.latch_offset(offset_z_world=1.0, obs_age_s=0.0)
    dt = 0.2
    _advance_clock(est, dt)
    vz_before = est.vz
    # grow by 0.2m over 0.2s -> vz_gate = (1.0-1.2)/0.2 = -1.0 m/s (climbing away, in-range)
    est.latch_offset(offset_z_world=1.2, obs_age_s=0.0)
    vz_gate_expected = (1.0 - 1.2) / dt   # = -1.0
    expected = vz_before + est.k_gate_vz_fusion * (vz_gate_expected - vz_before)
    assert est.vz == pytest.approx(expected, abs=1e-6)
    assert est.vz < vz_before, "growing offset (climbing away) must fuse vz MORE NEGATIVE"


# ---------------------------------------------------------------------------
# (b) dt accept-window: too-small and too-large both REJECT (vz untouched)
# ---------------------------------------------------------------------------
def test_dt_too_small_rejects_fusion():
    est = VerticalEstimator(use_gate_vz_fusion=True)
    est.seed()
    _burn_bias_capture(est)
    est.latch_offset(offset_z_world=2.0, obs_age_s=0.0)
    # advance LESS than gate_vz_fusion_dt_min_s (0.05s) -- e.g. one single IMU tick (~7ms)
    est.predict(0.0, IMU_DT)
    vz_before = est.vz
    est.latch_offset(offset_z_world=1.5, obs_age_s=0.0)   # a real shrink, but dt too small
    assert est.vz == pytest.approx(vz_before), \
        "a dt <= gate_vz_fusion_dt_min_s must REJECT the fusion sample (vz untouched)"


def test_dt_too_large_rejects_fusion():
    est = VerticalEstimator(use_gate_vz_fusion=True)
    est.seed()
    _burn_bias_capture(est)
    est.latch_offset(offset_z_world=2.0, obs_age_s=0.0)
    # advance MORE than gate_vz_fusion_dt_max_s (0.8s)
    _advance_clock(est, 1.0)
    vz_before = est.vz
    est.latch_offset(offset_z_world=1.5, obs_age_s=0.0)   # a real shrink, but dt too large
    assert est.vz == pytest.approx(vz_before), \
        "a dt > gate_vz_fusion_dt_max_s must REJECT the fusion sample (vz untouched)"


def test_dt_exactly_at_min_boundary_rejects():
    """The window is exclusive-of-min (``dt_min < dt <= dt_max``, per _fuse_gate_vz's docstring) --
    an exact dt == gate_vz_fusion_dt_min_s must still reject."""
    est = VerticalEstimator(use_gate_vz_fusion=True)
    est.seed()
    _burn_bias_capture(est)
    est.latch_offset(offset_z_world=2.0, obs_age_s=0.0)
    est._prev_fresh_offset_t_s = 0.0     # pin BOTH timestamps to clean values so their difference
    est._contact_elapsed_s = est.gate_vz_fusion_dt_min_s   # is an EXACT dt (no float-sum noise)
    vz_before = est.vz
    est.latch_offset(offset_z_world=1.5, obs_age_s=0.0)
    assert est.vz == pytest.approx(vz_before), \
        "dt exactly AT gate_vz_fusion_dt_min_s must reject (exclusive lower bound)"


def test_dt_exactly_at_max_boundary_accepts():
    """The window's upper bound is inclusive (``dt <= dt_max``) -- an exact dt == dt_max must
    still be ACCEPTED (not rejected)."""
    est = VerticalEstimator(use_gate_vz_fusion=True)
    est.seed()
    _burn_bias_capture(est)
    est.latch_offset(offset_z_world=2.0, obs_age_s=0.0)
    est._prev_fresh_offset_t_s = 0.0     # pin BOTH timestamps to clean values so their difference
    est._contact_elapsed_s = est.gate_vz_fusion_dt_max_s   # is an EXACT dt (no float-sum noise)
    vz_before = est.vz
    est.latch_offset(offset_z_world=1.9, obs_age_s=0.0)   # small in-range shrink
    assert est.vz != pytest.approx(vz_before), \
        "dt exactly AT gate_vz_fusion_dt_max_s must be ACCEPTED (inclusive upper bound)"


# ---------------------------------------------------------------------------
# (c) |vz_gate| > reject threshold leaves vz untouched
# ---------------------------------------------------------------------------
def test_implausible_vz_gate_magnitude_rejects_fusion():
    est = VerticalEstimator(use_gate_vz_fusion=True)
    est.seed()
    _burn_bias_capture(est)
    est.latch_offset(offset_z_world=5.0, obs_age_s=0.0)
    dt = 0.2
    _advance_clock(est, dt)
    vz_before = est.vz
    # shrink by 3.0m over 0.2s -> vz_gate = 15 m/s, way past gate_vz_fusion_reject_mps (2.5)
    est.latch_offset(offset_z_world=2.0, obs_age_s=0.0)
    assert est.vz == pytest.approx(vz_before), \
        "|vz_gate| > gate_vz_fusion_reject_mps must REJECT the sample (vz untouched)"


def test_vz_gate_at_reject_threshold_boundary():
    """A vz_gate whose magnitude is exactly the reject threshold must be ACCEPTED (the guard is
    a strict '>' reject, per _fuse_gate_vz: ``if abs(vz_gate) > reject: return``). dt is set
    directly on the estimator's internal clock (mirroring test_dt_exactly_at_*_boundary) so the
    dt used in vz_gate's division is an EXACT float, not an accumulation of IMU ticks -- avoiding
    float-accumulation noise from landing infinitesimally on the wrong side of the threshold."""
    est = VerticalEstimator(use_gate_vz_fusion=True)
    est.seed()
    _burn_bias_capture(est)
    dt = 0.2
    reject = est.gate_vz_fusion_reject_mps
    shrink = reject * dt   # vz_gate == exactly reject_mps, given the EXACT dt set below
    est.latch_offset(offset_z_world=shrink, obs_age_s=0.0)
    est._prev_fresh_offset_t_s = 0.0            # pin BOTH timestamps to clean values so their
    est._contact_elapsed_s = dt                 # difference is an exact dt (no float-sum noise)
    vz_before = est.vz
    est.latch_offset(offset_z_world=0.0, obs_age_s=0.0)
    assert est.vz != pytest.approx(vz_before), \
        "|vz_gate| exactly AT the reject threshold must be accepted (strict '>' reject only)"


# ---------------------------------------------------------------------------
# (d) first-ever latch is a no-op for fusion (nothing to difference against)
# ---------------------------------------------------------------------------
def test_first_ever_latch_is_fusion_noop():
    est = VerticalEstimator(use_gate_vz_fusion=True)
    est.seed()
    _burn_bias_capture(est)
    vz_before = est.vz
    assert vz_before == pytest.approx(0.0)
    est.latch_offset(offset_z_world=7.0, obs_age_s=0.0)   # first-ever latch
    assert est.vz == pytest.approx(vz_before), \
        "the first-ever latch has no previous fresh offset to difference against -- must no-op"
    # bookkeeping must still be established so the NEXT latch has something to compare against
    assert est._prev_fresh_offset_z == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# (e) use_gate_vz_fusion=False is a pure no-op -- VQ1/case-A byte-identical
# ---------------------------------------------------------------------------
def test_fusion_off_is_byte_identical_no_op():
    """With the flag OFF (the VQ1/case-A default), latch_offset must behave EXACTLY as the
    pre-A26 washout-only path -- no fusion state is even touched."""
    est_off = VerticalEstimator(use_gate_vz_fusion=False)
    est_off.seed()
    _burn_bias_capture(est_off)

    est_ref = VerticalEstimator(use_gate_vz_fusion=False)   # a second, independent reference
    est_ref.seed()
    _burn_bias_capture(est_ref)

    dt = 0.2
    for est in (est_off, est_ref):
        est.latch_offset(offset_z_world=5.0, obs_age_s=0.0)
        _advance_clock(est, dt)
    z_off_1_a, z_off_1_b = est_off.z_off, est_ref.z_off
    vz_1_a, vz_1_b = est_off.vz, est_ref.vz
    assert vz_1_a == pytest.approx(vz_1_b)
    assert z_off_1_a == pytest.approx(z_off_1_b)

    # a big shrink that WOULD trigger a large fusion pull if the flag were mistakenly honoured
    for est in (est_off, est_ref):
        est.latch_offset(offset_z_world=0.5, obs_age_s=0.0)
    assert est_off.vz == pytest.approx(0.0), \
        "fusion OFF: vz must be untouched by latch_offset regardless of the offset_z_world jump"
    assert est_off.vz == pytest.approx(est_ref.vz)
    assert est_off.z_off == pytest.approx(est_ref.z_off)
    # and the private fusion bookkeeping must never even be written when the flag is off
    assert est_off._prev_fresh_offset_z is None
    assert est_off._prev_fresh_offset_t_s is None


def test_default_use_gate_vz_fusion_is_false():
    """The dataclass default must be False -- VQ1/case-A (which never pass the flag) get the
    OFF path automatically."""
    assert VerticalEstimator().use_gate_vz_fusion is False


# ---------------------------------------------------------------------------
# (f) export clip is now +/-1.5 (FIX 1, de-saturate the damper)
# ---------------------------------------------------------------------------
def test_export_clip_is_1_5_mps():
    assert VerticalEstimator().export_clip_mps == pytest.approx(1.5)


def test_export_clip_actually_clamps_vz_read_at_1_5():
    est = VerticalEstimator()
    est.seed()
    _burn_bias_capture(est)
    dt = IMU_DT
    # sustained hard descent -- vz must never read above the clip on export
    for _ in range(3000):
        est.predict(-30.0, dt)          # a_up=-30 (a_up_clamp) -- sustained hard descent
        assert -1.5 - 1e-9 <= est.vz <= 1.5 + 1e-9


# ---------------------------------------------------------------------------
# reset/seed hygiene: re-seeding must forget the A26 fusion history (mirrors the A25 z_off reset)
# ---------------------------------------------------------------------------
def test_seed_resets_fusion_history():
    est = VerticalEstimator(use_gate_vz_fusion=True)
    est.seed()
    _burn_bias_capture(est)
    est.latch_offset(offset_z_world=3.0, obs_age_s=0.0)
    assert est._prev_fresh_offset_z is not None
    est.seed()
    assert est._prev_fresh_offset_z is None
    assert est._prev_fresh_offset_t_s is None


# ---------------------------------------------------------------------------
# (g) deploy-profile flag: vq2_case_c ON, vq1_case_a untouched (default OFF)
# ---------------------------------------------------------------------------
def test_vq2_case_c_gate_vz_fusion_superseded_off_by_a28():
    """A28 (2026-07-03): the fusion is SUPERSEDED by the complementary filter's beta position-
    innovations (run 20260703_013748 proved the finite-difference fusion is a noise/bias injector,
    A28 spec §1.3b) -- vq2_case_c now ships it OFF again; the flag itself survives for byte-compat
    / A-B replay of the A26 behaviour, and the A28 filter is opted in via
    nav_config.vertical_estimator_overrides instead."""
    from racer.deploy_profile import vq2_case_c

    prof = vq2_case_c()
    assert prof.nav_config.use_gate_vz_fusion is False
    assert (prof.nav_config.vertical_estimator_overrides or {}).get("use_zoff_filter") is True


def test_vq1_case_a_gate_vz_fusion_stays_default_off():
    from racer.deploy_profile import vq1_case_a
    from racer.navigator import NavigatorConfig

    prof = vq1_case_a()
    assert prof.nav_config.use_gate_vz_fusion is False
    assert prof.nav_config.use_gate_vz_fusion == NavigatorConfig().use_gate_vz_fusion


# ---------------------------------------------------------------------------
# (h) pose_age_s epoch-mismatch fix
# ---------------------------------------------------------------------------
def test_camera_epoch_to_imu_ns_converts_via_delta_epoch():
    """Navigator.camera_epoch_to_imu_ns must subtract the learned delta_epoch, and return None
    before any delta_epoch has been learned (mirrors _vision_fix_time_imu_ns's own guard)."""
    from racer.navigator import Navigator, NavigatorConfig

    nav = Navigator(gates=[], detector=None, config=NavigatorConfig())
    assert nav.camera_epoch_to_imu_ns(123) is None, \
        "no paired (frame, ds) has landed yet -> delta_epoch unknown -> None"
    nav._delta_epoch_ns = 1_700_000_000_000_000_000   # simulate a learned camera-vs-IMU offset
    camera_ns = 1_700_000_000_010_000_000              # "now" on the camera/unix epoch
    imu_ns = nav.camera_epoch_to_imu_ns(camera_ns)
    assert imu_ns == camera_ns - nav._delta_epoch_ns
    assert imu_ns == 10_000_000                        # 10 ms later on the IMU clock, as expected


def test_camera_epoch_to_imu_ns_none_when_reconcile_off():
    from racer.navigator import Navigator, NavigatorConfig

    nav = Navigator(gates=[], detector=None, config=NavigatorConfig(reconcile_vision_clock=False))
    nav._delta_epoch_ns = 42   # even if somehow set, the flag being off must still return None
    assert nav.camera_epoch_to_imu_ns(1000) is None


class _FakeNavOwnerWithEpoch:
    """A nav_owner stub exposing BOTH ``_vert_est`` and ``camera_epoch_to_imu_ns`` -- mirrors the
    real Navigator's surface as far as ``_maybe_latch_z_off`` reaches into it."""

    def __init__(self, vert_est, delta_epoch_ns):
        self._vert_est = vert_est
        self._delta_epoch_ns = delta_epoch_ns

    def camera_epoch_to_imu_ns(self, camera_sim_time_ns):
        return int(camera_sim_time_ns) - self._delta_epoch_ns


def test_maybe_latch_z_off_pose_age_uses_epoch_conversion():
    """The bug this reproduces: a pose.sim_time_ns on a huge camera/unix epoch and a
    nav.sim_time_ns on a small IMU epoch used to subtract to a hugely NEGATIVE number, which
    max(0.0, ...) silently clamped to exactly 0.0 every tick. With the epoch conversion wired in,
    a pose captured 0.05s (on the IMU clock) before `nav`'s current time must report pose_age_s
    close to 0.05 -- NOT 0.0."""
    est = VerticalEstimator()
    est.seed()
    _burn_bias_capture(est)

    delta_epoch_ns = 1_700_000_000_000_000_000   # camera epoch is ~1.7e18 ns ahead of the IMU epoch
    seeker = GateSeeker(config=GateSeekerConfig(),
                        nav_owner=_FakeNavOwnerWithEpoch(est, delta_epoch_ns))

    imu_now_ns = 20_000_000_000                  # 20s on the small IMU/sim-uptime clock
    nav = NavState(sim_time_ns=imu_now_ns, position_ned=np.zeros(3), velocity_ned=np.zeros(3),
                   roll=0.0, pitch=0.0, yaw=0.0, angular_rate_body=np.zeros(3))
    # the pose was captured 0.05s earlier ON THE IMU CLOCK -> its camera-epoch stamp is
    # (imu_capture_ns + delta_epoch_ns).
    imu_capture_ns = imu_now_ns - 50_000_000      # 50ms earlier
    pose_camera_ns = imu_capture_ns + delta_epoch_ns
    pose = GatePose(frame_id=0, sim_time_ns=pose_camera_ns,
                    R_cam_gate=np.eye(3), t_cam_gate=np.array([0.0, 0.0, 5.0]),
                    reproj_error_px=1.0)

    seeker._maybe_latch_z_off(nav, pose)
    assert seeker._last_pose_age_s == pytest.approx(0.05, abs=1e-6), \
        "pose_age_s must reflect the REAL ~50ms capture age, not be clamped to 0.0 by an epoch mismatch"


def test_maybe_latch_z_off_pose_age_falls_back_when_no_epoch_conversion():
    """When nav_owner has no camera_epoch_to_imu_ns (e.g. the plain _FakeNavOwner used by the
    other seeker-level tests, or a live wire before delta_epoch is learned), the RAW stamp is
    used -- byte-identical to the pre-A26 behaviour. On same-epoch synthetic data (both nav and
    pose stamped from one small fabricated clock) this yields a correct, tiny age."""
    class _FakeNavOwner:
        def __init__(self, vert_est):
            self._vert_est = vert_est

    est = VerticalEstimator()
    est.seed()
    _burn_bias_capture(est)
    seeker = GateSeeker(config=GateSeekerConfig(), nav_owner=_FakeNavOwner(est))
    nav = NavState(sim_time_ns=1_050_000_000, position_ned=np.zeros(3), velocity_ned=np.zeros(3),
                   roll=0.0, pitch=0.0, yaw=0.0, angular_rate_body=np.zeros(3))
    pose = GatePose(frame_id=0, sim_time_ns=1_000_000_000,
                    R_cam_gate=np.eye(3), t_cam_gate=np.array([0.0, 0.0, 5.0]),
                    reproj_error_px=1.0)
    seeker._maybe_latch_z_off(nav, pose)
    assert seeker._last_pose_age_s == pytest.approx(0.05, abs=1e-6)


# ---------------------------------------------------------------------------
# contact_frozen NavState plumbing
# ---------------------------------------------------------------------------
def test_navstate_contact_frozen_defaults_none():
    assert NavState(sim_time_ns=0).contact_frozen is None


def test_make_nav_state_exports_contact_frozen_none_when_absent():
    from racer.state_estimator import LinearKF, make_nav_state
    from racer.contracts import DroneState

    kf = LinearKF.initialize(np.zeros(3), pos_std=5.0)
    ds = DroneState(sim_time_ns=0, roll=0.0, pitch=0.0, yaw=0.0,
                    angular_rate_body=np.zeros(3), accel_body=np.zeros(3),
                    orientation_ned_wxyz=np.array([1.0, 0.0, 0.0, 0.0]))
    ns = make_nav_state(kf, ds, float("inf"))
    assert ns.contact_frozen is None, "absent contact_frozen kwarg must stay None, byte-identical"


def test_make_nav_state_exports_contact_frozen_true_and_false():
    from racer.state_estimator import LinearKF, make_nav_state
    from racer.contracts import DroneState

    kf = LinearKF.initialize(np.zeros(3), pos_std=5.0)
    ds = DroneState(sim_time_ns=0, roll=0.0, pitch=0.0, yaw=0.0,
                    angular_rate_body=np.zeros(3), accel_body=np.zeros(3),
                    orientation_ned_wxyz=np.array([1.0, 0.0, 0.0, 0.0]))
    assert make_nav_state(kf, ds, float("inf"), contact_frozen=True).contact_frozen is True
    assert make_nav_state(kf, ds, float("inf"), contact_frozen=False).contact_frozen is False


def test_navigator_exports_contact_frozen_when_seeded_and_contact_gate_fires():
    """End-to-end (through the real Navigator, not a fake): with use_vertical_estimator ON, a
    contact-impact spike must show up as NavState.contact_frozen=True on that tick, and False once
    the estimator is seeded but no contact is active. The VerticalEstimator's contact-gate only
    arms AFTER its 1.0s pre-arm bias-capture window closes, so this drives many small IMU ticks at
    zero specific-force (level, at rest) through the real Navigator first -- mirroring
    _burn_bias_capture, but through the full DroneState/Navigator path rather than calling
    VerticalEstimator.predict directly."""
    from racer.navigator import Navigator, NavigatorConfig
    from racer.contracts import DroneState

    nav = Navigator(gates=[], detector=None,
                   config=NavigatorConfig(use_vertical_estimator=True, use_given_position=True,
                                          use_given_velocity=True))
    dt_ns = int(IMU_DT * 1e9)
    t_ns = 0

    def _tick(accel_z_body: float):
        nonlocal t_ns
        t_ns += dt_ns
        ds = DroneState(sim_time_ns=t_ns, roll=0.0, pitch=0.0, yaw=0.0,
                        angular_rate_body=np.zeros(3), accel_body=np.array([0.0, 0.0, accel_z_body]),
                        orientation_ned_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                        position_ned=np.zeros(3), velocity_ned=np.zeros(3))
        return nav.update(ds)

    ns0 = _tick(-9.81)   # first tick: _initialize seeds the estimator immediately (at rest)
    assert ns0.contact_frozen is False, "seeded from the first tick (_initialize seeds at rest)"

    # burn the 1.0s pre-arm bias-capture window at true-zero bias (level, at rest)
    n_capture = int(np.ceil(1.0 / IMU_DT)) + 2
    ns = None
    for _ in range(n_capture):
        ns = _tick(-9.81)
    assert ns.contact_frozen is False, "seeded, no contact spike yet -> must export False (not None)"

    # a large |specific-force| spike -- above the contact-gate threshold (a_contact_mps2=20)
    ns1 = _tick(-60.0)
    assert ns1.contact_frozen is True, "a contact-gate spike must export contact_frozen=True"


def test_contact_frozen_gated_off_when_estimator_disabled():
    """VQ1/case-A (use_vertical_estimator=False, the default): contact_frozen must stay None,
    exactly like vert_vz_est/z_off_est stay NaN -- byte-identical, no new behaviour."""
    from racer.navigator import Navigator, NavigatorConfig
    from racer.contracts import DroneState

    nav = Navigator(gates=[], detector=None, config=NavigatorConfig())   # use_vertical_estimator default False
    ds0 = DroneState(sim_time_ns=0, roll=0.0, pitch=0.0, yaw=0.0,
                     angular_rate_body=np.zeros(3), accel_body=np.array([0.0, 0.0, -9.81]),
                     orientation_ned_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                     position_ned=np.zeros(3), velocity_ned=np.zeros(3))
    ns0 = nav.update(ds0)
    assert ns0.contact_frozen is None
