"""VERTICAL-VELOCITY WASHOUT FILTER — the A24 fix (2026-07-02), superseding the A21 3-state KF.

THE BUG (live VQ2, run 20260702_040036, read via scripts/analyze_flight.py): on the state-denied
wire the 6-state KF's z dead-reckons between sparse vision fixes and STEP-TELEPORTS when one lands
(-1.435 m in ONE tick), while the integrated IMU a_up shows a smooth REAL climb to +4-5 m/s upward
that est_z never tracked. The ff-owns-vertical alt-hold read that z in both terms (kp_alt on z,
ff_vertical_kd_alt on the LP finite difference of z), so it slammed on pin teleports and was blind
to the actual climb between them -> over gate 0 into the ceiling.

THE A21 FIX (superseded) was a 3-state [z, vz, bias] KF correcting the drift with floor_height
pins. THAT WAS A FALSE PREMISE: the warehouse has no floor grid, so a KF with a bias state and NO
real external measurement is just a washout wearing a Kalman costume, and the bias is fundamentally
unobservable on this wire.

THE A24 FIX (this file): ``racer.vertical_estimator.VerticalEstimator`` is now a first-order
washout (leaky integrator) over `(vz, b_hat)` — no altitude state, no floor-pin correction, no
Kalman gain. It is bounded BY CONSTRUCTION: a sustained accel bias pulls vz to a constant offset
(tau*b), never a ramp, and the export is additionally hard-clipped to +/-3 m/s regardless of input.
An input outlier clamp (+/-30 m/s^2 = +/-3g, the COMMANDER ADDITION) rejects contact-impact spikes
and sensor glitches at the source, before they ever reach the leaky integrator.

[VQ2 slow-is-smooth, A24, 2026-07-02]
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from racer.contracts import DroneState, NavState, Setpoint  # noqa: E402
from racer.deploy_profile import DeployProfile, vq1_case_a, vq2_case_c  # noqa: E402
from racer.gate_seeker import make_seeker_controller  # noqa: E402
from racer.vertical_estimator import VerticalEstimator, a_up_from_specific_force  # noqa: E402

G = 9.80665
IMU_DT = 1.0 / 140.0           # the live HIGHRES_IMU rate (~143 Hz; median dt 7.0 ms on-wire)
DT_NS = int(1e9 / 12)          # ~12 Hz control ticks for the controller-level tests


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _run_filter(est, a_up_stream, dt=IMU_DT, skip_capture=True):
    """Feed a stream of a_up samples through an already-seeded VerticalEstimator; return the vz
    trace. If ``skip_capture`` (default), first burn through the bias-capture window with a_up=0
    (the ground-truth on-pad reading) so the returned trace starts in the steady washout regime."""
    if skip_capture:
        n_capture = int(np.ceil(est.bias_capture_s / dt)) + 1
        for _ in range(n_capture):
            est.predict(0.0, dt)
    vz = np.zeros(len(a_up_stream))
    for i, a in enumerate(a_up_stream):
        est.predict(float(a), dt)
        vz[i] = est.vz
    return vz


# ---------------------------------------------------------------------------
# seeding / lifecycle
# ---------------------------------------------------------------------------
def test_unseeded_outputs_nan_and_all_calls_are_noops():
    est = VerticalEstimator()
    assert not est.seeded
    assert np.isnan(est.vz)
    assert np.isnan(est.z)                  # z is now PERMANENTLY nan, seeded or not
    est.predict(5.0, 0.01)                  # un-seeded: must not raise / must stay NaN
    assert np.isnan(est.vz)


def test_seed_is_vz0_bhat0_and_z_permanently_nan():
    est = VerticalEstimator()
    est.seed()
    assert est.seeded
    assert est.vz == 0.0                    # export valid from tick 1
    assert np.isnan(est.z)                  # no altitude state, ever


def test_dt_guards_skip_integration():
    """dt<=0 (a between-IMU control tick) and dt>max_dt_s (sim reset / stutter) are no-ops."""
    est = VerticalEstimator()
    est.seed()
    # burn through bias capture first so we're in the steady washout regime
    n_capture = int(np.ceil(est.bias_capture_s / IMU_DT)) + 1
    for _ in range(n_capture):
        est.predict(0.0, IMU_DT)
    est.predict(5.0, 1.0)                  # a real step first, to get a nonzero vz
    vz0 = est.vz
    est.predict(5.0, 0.0)                  # dt=0: no-op
    assert est.vz == vz0
    est.predict(5.0, est.max_dt_s + 0.05)  # dt >> max_dt_s: rejected (garbage step)
    assert est.vz == vz0


# ---------------------------------------------------------------------------
# pre-arm bias capture
# ---------------------------------------------------------------------------
def test_bias_capture_absorbs_a_constant_a_up_offset_at_rest():
    """A sustained a_up offset during the pre-arm window (the resting-attitude-projection
    residual) is captured into b_hat and does NOT show up as vz once the window closes: fed a
    pure bias for the whole capture window, vz stays ~0 immediately after capture ends."""
    est = VerticalEstimator()
    est.seed()
    bias_a_up = -0.3   # a_dn = +0.3 constant "residual tilt" offset
    dt = IMU_DT
    t = 0.0
    while t < est.bias_capture_s + 2 * dt:
        est.predict(bias_a_up, dt)
        t += dt
    assert est._bias_capture_done
    assert abs(est._b_hat - 0.3) < 0.05          # captured the true offset
    assert abs(est.vz) < 0.05                    # and vz stayed ~0 (no ground-truth motion)


def test_captured_bias_is_clipped():
    """A large a_up offset during capture is clipped to +/-bias_clip_mps2 (a sane ceiling on a
    real attitude-projection residual, not a licence to absorb an arbitrary sensor fault)."""
    est = VerticalEstimator()
    est.seed()
    dt = IMU_DT
    t = 0.0
    while t < est.bias_capture_s + 2 * dt:
        est.predict(-5.0, dt)               # a_dn = +5.0, way above bias_clip_mps2=0.5
        t += dt
    assert est._bias_capture_done
    assert abs(est._b_hat - est.bias_clip_mps2) < 1e-9


def test_vz_stays_zero_and_finite_throughout_bias_capture_window():
    """Export is valid (finite) from tick 1, including mid-capture (spec: 'Export valid from tick
    1'), and reads exactly 0 (grounded) until the window closes."""
    est = VerticalEstimator()
    est.seed()
    dt = IMU_DT
    n = int(est.bias_capture_s / dt) - 2
    for _ in range(n):
        est.predict(1.0, dt)
        assert np.isfinite(est.vz)
        assert est.vz == 0.0


# ---------------------------------------------------------------------------
# input outlier clamp (COMMANDER ADDITION)
# ---------------------------------------------------------------------------
def test_input_clamp_rejects_a_huge_spike():
    """A single ~100 m/s^2 spike (e.g. a contact-impact glitch, or a sensor fault) is clamped at
    the SOURCE to +/-a_up_clamp_mps2=30 before it ever reaches the washout integrator -- one tick
    of even the full clamp magnitude cannot produce a large vz jump."""
    est = VerticalEstimator()
    est.seed()
    n_capture = int(np.ceil(est.bias_capture_s / IMU_DT)) + 1
    for _ in range(n_capture):
        est.predict(0.0, IMU_DT)          # clear the capture window at true-zero bias
    vz_before = est.vz
    est.predict(-100.0, IMU_DT)           # a_up=-100 -> a_dn=+100, way past the 30 m/s^2 clamp
    dvz = est.vz - vz_before
    # clamped: the tick's contribution is bounded by clamp*dt, not the raw 100*dt
    assert abs(dvz) <= est.a_up_clamp_mps2 * IMU_DT + 1e-9
    assert abs(dvz) < 100.0 * IMU_DT - 1e-6      # strictly less than the unclamped contribution


def test_input_clamp_value_is_30_mps2():
    assert VerticalEstimator().a_up_clamp_mps2 == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# export clip: the absolute, unconditional cannot-diverge guarantee
# ---------------------------------------------------------------------------
def test_export_is_clipped_to_plus_minus_3():
    """Even with the input clamp allowing up to 30 m/s^2 through indefinitely, the exported vz
    never exceeds the +/-3 m/s export clip -- the ABSOLUTE guarantee regardless of input garbage."""
    est = VerticalEstimator()
    est.seed()
    n_capture = int(np.ceil(est.bias_capture_s / IMU_DT)) + 1
    for _ in range(n_capture):
        est.predict(0.0, IMU_DT)
    # sustained max-clamp input for a long stretch: the internal state may exceed 3 transiently
    # (tau*a_up_clamp = 2*30 = 60 in the limit) but the EXPORT must never show it.
    for _ in range(2000):
        est.predict(-30.0, IMU_DT)          # a_dn = +30 sustained
        assert -3.0 - 1e-9 <= est.vz <= 3.0 + 1e-9


# ---------------------------------------------------------------------------
# cannot-diverge under a sustained biased input
# ---------------------------------------------------------------------------
def test_cannot_diverge_under_sustained_bias_bounded_by_tau_times_b():
    """A sustained a_up bias (post-capture residual, e.g. an uncaptured slow attitude drift) does
    NOT ramp vz without bound -- it settles to a CONSTANT offset ~= tau*(a_dn - b_hat), the washout
    boundedness guarantee (vs the old KF's dead-reckoning divergence)."""
    est = VerticalEstimator(washout_tau_s=2.0)
    est.seed()
    dt = IMU_DT
    # capture window sees a_dn=0 (true zero bias captured) so b_hat=0 and the POST-capture bias
    # below is a genuine un-captured residual, not something absorbed into b_hat.
    n_capture = int(np.ceil(est.bias_capture_s / dt)) + 1
    for _ in range(n_capture):
        est.predict(0.0, dt)
    a_dn_bias = 0.4                          # a_up = -0.4, a residual post-capture bias
    n_steps = int(20.0 / dt)                 # run WAY past tau=2s -- 10 tau's
    vz_trace = np.zeros(n_steps)
    for i in range(n_steps):
        est.predict(-a_dn_bias, dt)
        vz_trace[i] = est.vz
    expected_ss = est.washout_tau_s * a_dn_bias           # = 0.8
    # settles near tau*b, not off to infinity, and stays bounded there (no further growth)
    assert abs(vz_trace[-1] - expected_ss) < 0.05
    assert float(np.abs(vz_trace[n_steps // 2:]).max()) < expected_ss + 0.05
    # explicit divergence check: the back half of the trace never exceeds a small band around the
    # settled value (it truly SETTLES, it does not keep climbing)
    tail = vz_trace[-int(1.0 / dt):]
    assert float(np.ptp(tail)) < 0.05


# ---------------------------------------------------------------------------
# steady-input decay to 0 over ~tau (the "gate-align loop owns steady state" design point)
# ---------------------------------------------------------------------------
def test_steady_climb_washes_to_zero_over_tau():
    """A step to a_dn=0 after an initial nonzero vz decays exponentially to 0 with time constant
    tau (no sustained input to hold it up) -- 'vz washes to 0 over tau (a_dn~=0)' per spec.

    Uses the internal (unclamped) ``_vz`` as the pre-decay reference, not the exported ``vz``
    property: A26 FIX 1 (2026-07-02) tightened ``export_clip_mps`` 3.0 -> 1.5 (de-saturate the
    washout damper, see handoff/vq2_a26_vertical_brake_2026-07-02.md), and this kick's TRUE
    internal magnitude (~1.9 m/s) now exceeds that read-side clip -- referencing the clamped
    ``vz`` as "the kicked value" would silently under-state the real decay ratio being checked.
    The export clip is a read-side guarantee only; it does not change the washout recurrence
    itself, which is what this test verifies."""
    est = VerticalEstimator(washout_tau_s=2.0)
    est.seed()
    dt = IMU_DT
    n_capture = int(np.ceil(est.bias_capture_s / dt)) + 1
    for _ in range(n_capture):
        est.predict(0.0, dt)
    # kick vz up with one big (but clamp-legal) impulse-ish burst, then hold a_dn=0
    for _ in range(int(0.2 / dt)):
        est.predict(-10.0, dt)              # a_dn=+10 for 0.2s -> some positive vz
    vz_kicked = est._vz                     # TRUE internal magnitude, pre-export-clip
    assert vz_kicked > 0.5, "setup: kick did not raise vz"
    # now hold zero input for ~2 tau and confirm decay toward 0
    for _ in range(int(2 * est.washout_tau_s / dt)):
        est.predict(0.0, dt)
    assert abs(est._vz) < 0.15 * vz_kicked, ("vz did not wash out toward 0", est._vz, vz_kicked)


def test_washout_alpha_matches_exp_minus_dt_over_tau():
    """Sanity: with a_dn - b_hat == 0, the recurrence is pure exponential decay vz *= exp(-dt/tau)."""
    est = VerticalEstimator(washout_tau_s=2.0)
    est.seed()
    dt = IMU_DT
    n_capture = int(np.ceil(est.bias_capture_s / dt)) + 1
    for _ in range(n_capture):
        est.predict(0.0, dt)
    est.predict(-10.0, dt)          # one nonzero kick
    vz0 = est.vz
    est.predict(0.0, dt)            # then pure decay (a_dn - b_hat == 0 - 0 == 0)
    expected = vz0 * np.exp(-dt / est.washout_tau_s)
    assert est.vz == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# a_up decode convention (unchanged from A21 -- still exact)
# ---------------------------------------------------------------------------
def test_a_up_decode_matches_verified_convention():
    R_level = np.eye(3)
    f_rest = np.array([0.0, 0.0, -G])            # on-pad specific force, body FRD (|f| = g)
    assert abs(a_up_from_specific_force(f_rest, R_level)) < 1e-12
    f_climb = np.array([0.0, 0.0, -(G + 1.0)])   # rotors push 1 m/s^2 beyond gravity
    assert abs(a_up_from_specific_force(f_climb, R_level) - 1.0) < 1e-12
    p = np.deg2rad(-17.8)
    R_pitch = np.array([[np.cos(p), 0.0, np.sin(p)], [0.0, 1.0, 0.0], [-np.sin(p), 0.0, np.cos(p)]])
    f_tilted = R_pitch.T @ f_rest
    assert abs(a_up_from_specific_force(f_tilted, R_pitch)) < 1e-9


def test_update_z_no_longer_exists():
    """DELETED per spec: the washout has no absolute-altitude correction path at all."""
    assert not hasattr(VerticalEstimator, "update_z")


# ---------------------------------------------------------------------------
# controller-level DoD: the damper reads the estimator's washout vz
# ---------------------------------------------------------------------------
def _teleporting_nav_arc(n=40, seed=0):
    """The LIVE failure shape: the nav z creeps (dead-reckoning misses the climb) then TELEPORTS
    -1.4 m when a vision fix lands, while the TRUE motion (what the vertical estimator sees) is a
    smooth climb. Returns (t, nav_z, true_z, true_vz)."""
    t = np.arange(n) * DT_NS
    ts = t / 1e9
    true_z = -np.clip(0.5 * ts, 0.0, 1.6)                 # smooth 0.5 m/s climb to -1.6 m
    true_vz = np.gradient(true_z, ts)
    nav_z = 0.02 * np.random.default_rng(seed).standard_normal(n)   # dead-reckoned: flat + noise
    nav_z[n // 2:] = true_z[n // 2:] + 0.02                          # the pin teleport, one tick
    return t, nav_z, true_z, true_vz


def _run_ctrl(ctrl, t, nav_z, vert_vz=None):
    """Replay through the alt-hold at level attitude with the pursuit-style setpoint (accel_ned set,
    position None) — the exact regime where the vertical channel is the sole thrust driver."""
    out = np.zeros(len(t))
    for i in range(len(t)):
        kw = {}
        if vert_vz is not None:
            kw = dict(vert_vz_est=float(vert_vz[i]))
        nav = NavState(sim_time_ns=int(t[i]),
                       position_ned=np.array([0.0, 0.0, nav_z[i]]),
                       velocity_ned=np.zeros(3),
                       roll=0.0, pitch=0.0, yaw=0.0, angular_rate_body=np.zeros(3), **kw)
        sp = Setpoint(sim_time_ns=int(t[i]),
                      accel_ned=1.2 * np.array([1.0, 0.0, 0.0]), yaw=0.0)
        out[i] = float(ctrl.command(nav, sp).thrust)
    return out


def test_controller_damps_on_washout_vz_not_teleporting_z():
    """DoD: flag ON + a live washout export -> the alt-hold sees the smooth, bounded vz, so a
    nav-z teleport tick produces NO thrust slam; flag OFF on the same inputs slams (the fd of the
    teleport step)."""
    t, nav_z, true_z, true_vz = _teleporting_nav_arc()
    thr_off = _run_ctrl(make_seeker_controller(ff_owns_vertical=True), t, nav_z)
    thr_on = _run_ctrl(make_seeker_controller(ff_owns_vertical=True, use_vertical_estimator=True,
                                              ff_vertical_vz_lp_alpha=1.0),
                       t, nav_z, vert_vz=true_vz)
    i_tp = len(t) // 2                                     # the teleport tick
    step_off = abs(thr_off[i_tp] - thr_off[i_tp - 1])
    step_on = abs(thr_on[i_tp] - thr_on[i_tp - 1])
    assert step_off > 0.2, ("repro lost: OFF no longer slams on the teleport", step_off)
    assert step_on < 0.05, ("ON still slams on the teleport", step_on)
    climb = slice(2, i_tp)
    assert float(np.mean(thr_on[climb])) < make_seeker_controller().hover_thrust, \
        "ON is blind to the real climb"


def test_controller_falls_back_to_zero_when_estimator_absent_not_fd():
    """Flag ON but NaN export (estimator off / not yet seeded) -> vz=0 fallback (a benign
    open-loop track on vz_t), NEVER the A19c fd-of-z path -- the spec explicitly forbids
    re-admitting the unbounded dead-reckoning path as a 'fallback'. kp_alt=0 here (matching
    vq2_case_c's R0 override) isolates the vz-damping-term behaviour this fallback governs; the
    kp_alt*(z-z_target) position term is a SEPARATE poison R0 kills independently (z_v is always
    pos[2], regardless of the vz path)."""
    t, nav_z, _, _ = _teleporting_nav_arc(seed=3)
    thr_fd = _run_ctrl(make_seeker_controller(ff_owns_vertical=True, kp_alt=0.0), t, nav_z)
    thr_nan = _run_ctrl(make_seeker_controller(ff_owns_vertical=True, use_vertical_estimator=True,
                                               kp_alt=0.0),
                        t, nav_z)                          # NavState vert_vz_est default NaN
    # NOT equal to the fd path (that path is explicitly excluded as the NaN fallback)
    assert not np.allclose(thr_fd, thr_nan, atol=1e-9), \
        "NaN fallback illegally reused the fd-of-z path"
    # the teleport tick produces no slam either (vz=0 fallback never reacts to the nav-z step)
    i_tp = len(t) // 2
    step_nan = abs(thr_nan[i_tp] - thr_nan[i_tp - 1])
    assert step_nan < 0.05, ("NaN fallback slammed on the teleport (fd path leaked back in)",
                            step_nan)


def test_off_path_byte_identical_even_with_export_present():
    """OFF-path byte-identity: with use_vertical_estimator OFF, a nav that CARRIES a live export is
    ignored — the thrust equals the fd-path controller exactly (the flag is the only fork)."""
    t, nav_z, true_z, true_vz = _teleporting_nav_arc(seed=5)
    thr_plain = _run_ctrl(make_seeker_controller(ff_owns_vertical=True), t, nav_z)
    thr_carrying = _run_ctrl(make_seeker_controller(ff_owns_vertical=True), t, nav_z,
                             vert_vz=true_vz)
    assert np.allclose(thr_plain, thr_carrying, atol=0.0), "OFF path read the export"


def test_gate_is_on_vz_alone_not_both_z_and_vz():
    """REQUIRED controller change: vert_z_est is now PERMANENTLY NaN, so a gate requiring BOTH
    exports finite would silently and permanently disable the estimator path. Confirm a nav
    carrying a finite vz but NaN z (the real, permanent shape of the export) still engages the
    estimator path (no thrust slam on the teleport)."""
    t, nav_z, _, true_vz = _teleporting_nav_arc(seed=7)
    ctrl = make_seeker_controller(ff_owns_vertical=True, use_vertical_estimator=True,
                                  ff_vertical_vz_lp_alpha=1.0)
    out = np.zeros(len(t))
    for i in range(len(t)):
        nav = NavState(sim_time_ns=int(t[i]),
                       position_ned=np.array([0.0, 0.0, nav_z[i]]),
                       velocity_ned=np.zeros(3), roll=0.0, pitch=0.0, yaw=0.0,
                       angular_rate_body=np.zeros(3),
                       vert_z_est=float("nan"), vert_vz_est=float(true_vz[i]))
        sp = Setpoint(sim_time_ns=int(t[i]), accel_ned=1.2 * np.array([1.0, 0.0, 0.0]), yaw=0.0)
        out[i] = float(ctrl.command(nav, sp).thrust)
    i_tp = len(t) // 2
    step = abs(out[i_tp] - out[i_tp - 1])
    assert step < 0.05, ("z=NaN, vz=finite should still engage the estimator path", step)


# ---------------------------------------------------------------------------
# navigator plumbing: own/step/export behind the config flag
# ---------------------------------------------------------------------------
def _ds(t_s, accel=(0.0, 0.0, -G), gyro=(0.0, 0.0, 0.0)):
    return DroneState(sim_time_ns=int(t_s * 1e9),
                      accel_body=np.asarray(accel, dtype=np.float64),
                      gyro_body=np.asarray(gyro, dtype=np.float64),
                      orientation_ned_wxyz=np.array([1.0, 0.0, 0.0, 0.0]))


def test_navigator_exports_vz_and_z_stays_nan():
    """Wiring: use_vertical_estimator=True -> NavState carries a finite vz export from tick 1;
    vert_z_est stays NaN PERMANENTLY (no altitude state in the washout); default config -> both
    fields stay NaN (byte-identical export)."""
    from racer.navigator import Navigator, NavigatorConfig

    cfg = NavigatorConfig(use_given_position=False, use_given_velocity=False,
                          use_ahrs=True, use_vertical_estimator=True)
    nav = Navigator(gates=[], detector=None, config=cfg)
    ns = nav.update(_ds(0.0))
    assert np.isfinite(ns.vert_vz_est)             # seeded at init, export valid from tick 1
    assert np.isnan(ns.vert_z_est)                 # PERMANENTLY nan
    ns = nav.update(_ds(0.01))                     # one IMU predict tick
    assert np.isfinite(ns.vert_vz_est)
    assert np.isnan(ns.vert_z_est)

    # default OFF: no estimator constructed, export stays NaN
    nav_off = Navigator(gates=[], detector=None,
                        config=NavigatorConfig(use_given_position=False, use_ahrs=True))
    ns_off = nav_off.update(_ds(0.0))
    assert nav_off._vert_est is None
    assert np.isnan(ns_off.vert_z_est) and np.isnan(ns_off.vert_vz_est)


def test_navigator_floor_height_pin_no_longer_touches_vertical_channel(monkeypatch):
    """The 6-state KF's OWN floor-height correction (use_floor_height, a SEPARATE, generic,
    independently-tested feature) still runs -- but it must NOT reach into the vertical-velocity
    washout any more (update_z is gone). The washout's vz should be driven purely by the a_up
    integration, unaffected by a floor pin landing."""
    from racer import navigator as nav_mod
    from racer.contracts import Frame
    from racer.navigator import Navigator, NavigatorConfig

    class _NoDetections:
        def detect(self, frame):
            return []

    cfg = NavigatorConfig(use_given_position=False, use_given_velocity=False,
                          use_ahrs=True, use_floor_height=True, use_vertical_estimator=True)
    nav = Navigator(gates=[], detector=_NoDetections(), config=cfg)
    nav.update(_ds(0.0))
    nav.update(_ds(0.01))

    class _FloorEst:
        height_m, std_m, quality = 2.0, 0.1, 0.9
    monkeypatch.setattr(nav_mod, "estimate_floor_height", lambda *a, **k: _FloorEst())
    vz_before = nav._vert_est.vz
    frame = Frame(frame_id=1, sim_time_ns=int(0.02 * 1e9),
                  image_bgr=np.zeros((4, 4, 3), dtype=np.uint8))
    nav.update(_ds(0.02), frame)
    assert nav.n_floor_z_total == 1, "the 6-state-KF floor pin was not applied (unrelated feature)"
    # the vertical washout is untouched by the pin: no update_z call exists any more, so vz only
    # moved by the ordinary a_up integration across this tiny dt (bounded, tiny)
    assert abs(nav._vert_est.vz - vz_before) < 0.05


# ---------------------------------------------------------------------------
# profile / default pins (the async_detect gating pattern, mirrored)
# ---------------------------------------------------------------------------
def test_defaults_are_off_everywhere():
    from racer.controller import Controller
    from racer.navigator import NavigatorConfig
    assert Controller().use_vertical_estimator is False
    assert NavigatorConfig().use_vertical_estimator is False
    assert DeployProfile.__dataclass_fields__["vertical_estimator"].default is False
    assert vq1_case_a().vertical_estimator is False


def test_vq2_profile_opts_in():
    assert vq2_case_c().vertical_estimator is True


def test_vq2_profile_sets_kp_alt_zero_and_new_kd_lp():
    """R0 + the tuned Kd/LP-alpha: vq2_case_c overrides kp_alt=0.0 (kill the position term),
    ff_vertical_kd_alt=0.06 (the A28 single-PD sizing, zeta=0.92 at kp_gate=0.04 on the measured
    c~=37 plant -- was 0.25 under A26, on a signal the A28 diagnosis proved garbage), and
    ff_vertical_vz_lp_alpha=0.8 (unchanged)."""
    ov = vq2_case_c().controller_overrides
    assert ov is not None
    assert ov["kp_alt"] == pytest.approx(0.0)
    assert ov["ff_vertical_kd_alt"] == pytest.approx(0.06)
    assert ov["ff_vertical_vz_lp_alpha"] == pytest.approx(0.8)


def test_controller_defaults_unchanged_vq1_safe():
    """The bare Controller field defaults (ff_vertical_kd_alt/vz_lp_alpha) are UNTOUCHED -- the
    tuned values are only reached via vq2_case_c's controller_overrides, so VQ1/case-A/every other
    caller stays byte-identical."""
    from racer.controller import Controller
    assert Controller().ff_vertical_kd_alt == pytest.approx(0.5)
    assert Controller().ff_vertical_vz_lp_alpha == pytest.approx(0.5)
    assert Controller().kp_alt == pytest.approx(0.0)   # unrelated to this fix; already 0 by default


def test_fly_rl_resolver_cli_overrides_profile():
    sys.path.insert(0, str(ROOT / "rl"))
    from fly_rl import _resolve_vertical_estimator

    class _Args:
        vertical_estimator = "auto"
    assert _resolve_vertical_estimator(_Args(), vq2_case_c()) is True
    assert _resolve_vertical_estimator(_Args(), vq1_case_a()) is False
    _Args.vertical_estimator = "off"
    assert _resolve_vertical_estimator(_Args(), vq2_case_c()) is False
    _Args.vertical_estimator = "on"
    assert _resolve_vertical_estimator(_Args(), vq1_case_a()) is True
