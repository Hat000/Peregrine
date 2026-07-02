"""VERTICAL-CHANNEL ESTIMATOR — the A21 egress->ceiling-climb fix (2026-07-02).

THE BUG (live VQ2, run 20260702_040036, read via scripts/analyze_flight.py): on the state-denied
wire the 6-state KF's z dead-reckons between the sparse vision floor_height pins and STEP-TELEPORTS
when one lands (-1.435 m in ONE tick), while the integrated IMU a_up shows a smooth REAL climb to
+4-5 m/s upward that est_z never tracked. The ff-owns-vertical alt-hold read that z in BOTH terms
(kp_alt on z, ff_vertical_kd_alt on the LP finite difference of z), so it slammed on pin teleports
and was blind to the actual climb between them -> over gate 0 into the ceiling.

THE FIX: ``racer.vertical_estimator.VerticalEstimator`` — a tiny 3-state [z, vz, bias] KF that
integrates the VERIFIED a_up decode (~140 Hz) and applies the accepted floor pins as bounded
Kalman corrections. The Navigator owns it behind ``NavigatorConfig.use_vertical_estimator`` and
exports (z, vz) on ``NavState.vert_z_est / vert_vz_est``; the controller damps on that export
behind ``Controller.use_vertical_estimator``. Default OFF everywhere => byte-identical;
``DeployProfile.vertical_estimator`` (vq2_case_c ON) threads both sides at the fly_rl seam.

[VQ2 slow-is-smooth, A21, 2026-07-02]
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from racer.contracts import DroneState, NavState, Setpoint  # noqa: E402
from racer.deploy_profile import DeployProfile, vq1_case_a, vq2_case_c  # noqa: E402
from racer.gate_seeker import make_seeker_controller  # noqa: E402
from racer.vertical_estimator import VerticalEstimator, a_up_from_specific_force  # noqa: E402

G = 9.80665
IMU_DT = 1.0 / 140.0           # the live HIGHRES_IMU rate (~143 Hz; median dt 7.0 ms on-wire)
DT_NS = int(1e9 / 12)          # ~12 Hz control ticks for the controller-level tests
LO, HI = 0.05, 0.6             # make_seeker_controller alt-hold clip rails


# ---------------------------------------------------------------------------
# synthetic vertical truth: a smooth climb-then-hold arc + its exact a_up
# ---------------------------------------------------------------------------
def _truth_arc(n_s=4.0, dt=IMU_DT):
    """Ground-truth (t, z, vz, a_up): level 1 s, then a smooth sinusoidal climb of 1.5 m over 2 s,
    then hold. NED z down+, a_up up+ (a_up = -zdd)."""
    t = np.arange(0.0, n_s, dt)
    z = np.zeros_like(t)
    m = (t >= 1.0) & (t < 3.0)
    ph = (t[m] - 1.0) / 2.0                                   # 0..1 over the climb
    z[m] = -1.5 * (ph - np.sin(2 * np.pi * ph) / (2 * np.pi))  # min-jerk-ish: zdd continuous
    z[t >= 3.0] = -1.5
    vz = np.gradient(z, dt)
    a_up = -np.gradient(vz, dt)
    return t, z, vz, a_up


def _run_filter(t, a_up, pins, est=None, accel_noise=0.0, seed=0):
    """Feed the a_up stream (+ optional white noise) and the (time->z_meas, var) pin dict through a
    VerticalEstimator; return the per-sample (z_est, vz_est) arrays + the per-pin |z jump|s."""
    rng = np.random.default_rng(seed)
    est = est or VerticalEstimator()
    est.seed(0.0)
    zs, vzs, jumps = np.zeros_like(t), np.zeros_like(t), []
    pin_times = sorted(pins)
    k = 0
    for i in range(len(t)):
        dt = t[i] - t[i - 1] if i else 0.0
        est.predict(float(a_up[i]) + accel_noise * rng.standard_normal(), dt)
        while k < len(pin_times) and pin_times[k] <= t[i]:
            z_before = est.z
            z_meas, var = pins[pin_times[k]]
            est.update_z(z_meas, var)
            jumps.append(abs(est.z - z_before))
            k += 1
        zs[i], vzs[i] = est.z, est.vz
    return zs, vzs, jumps


def _pins_from_truth(z_of_t, times, var=0.08, noise=0.1, seed=1):
    """Floor pins: the true z at each pin time + measurement noise, at the floor-channel variance
    (std_m~0.2 + extra_std 0.2 in quadrature => var ~0.08)."""
    rng = np.random.default_rng(seed)
    return {float(tt): (float(z_of_t(tt)) + noise * rng.standard_normal(), var) for tt in times}


# ---------------------------------------------------------------------------
# filter-level DoD: vz tracks integrated accel; altitude never teleports on a pin
# ---------------------------------------------------------------------------
def test_vz_tracks_ground_truth_through_a_climb():
    """DoD #1a: fed the synthetic accel + ~5 Hz floor pins, the estimator vz tracks the ground-truth
    vertical rate through a 1.5 m climb (the exact signature the dead-reckoned z missed live)."""
    t, z, vz, a_up = _truth_arc()
    zfun = lambda tt: np.interp(tt, t, z)
    pins = _pins_from_truth(zfun, np.arange(0.2, 4.0, 0.2))
    zs, vzs, _ = _run_filter(t, a_up, pins, accel_noise=0.3)
    err_vz = np.abs(vzs - vz)
    assert float(err_vz.max()) < 0.35, ("vz lost the climb", float(err_vz.max()))
    assert float(np.abs(zs - z).max()) < 0.35, ("altitude lost the climb", float(np.abs(zs - z).max()))


def test_altitude_never_teleports_on_pin_arrival():
    """DoD #1b: pin corrections are BOUNDED, incremental Kalman nudges — never the KF-z style full
    step. Even with pins deliberately OFFSET 1.4 m from the propagated state (the live teleport
    magnitude), no single pin moves z by more than half the innovation, and vz stays finite/smooth."""
    t, z, vz, a_up = _truth_arc()
    # pins biased -1.4 m off truth: a worst-case persistent disagreement with the propagation
    pins = {float(tt): (float(np.interp(tt, t, z)) - 1.4, 0.08) for tt in np.arange(0.2, 4.0, 0.2)}
    zs, vzs, jumps = _run_filter(t, a_up, pins, accel_noise=0.3)
    assert jumps, "no pin was applied"
    # the FIRST pin sees the full 1.4 m innovation; even it must be a partial correction
    assert max(jumps) < 0.7, ("a pin teleported z", max(jumps))
    # and the correction stream converges: z ends near the pinned level (the filter does obey pins)
    assert abs(zs[-1] - (z[-1] - 1.4)) < 0.30, ("pins never won", float(zs[-1]), float(z[-1] - 1.4))
    # vz never spikes the way the fd of a teleporting z did (-4.4 m/s live; fd of the raw step ~42)
    assert float(np.abs(np.diff(vzs)).max()) < 0.5, "vz stepped on a pin"


def test_bias_state_kills_sustained_accel_drift():
    """A sustained a_up offset (an AHRS tilt error rotating gravity into the vertical — the −17.8°
    resting pitch makes this the live risk) must be absorbed by the bias state: with pins arriving,
    the steady-state vz error stays a fraction of the raw drift (0.8 m/s^2 * 6 s = 4.8 m/s raw)."""
    dt = IMU_DT
    t = np.arange(0.0, 6.0, dt)
    a_up = np.full_like(t, 0.8)                 # pure bias: true motion is a HOLD (z=0, vz=0)
    pins = {float(tt): (0.0, 0.08) for tt in np.arange(0.2, 6.0, 0.2)}
    zs, vzs, _ = _run_filter(t, a_up, pins)
    assert float(np.abs(vzs[-len(t) // 3:]).max()) < 0.25, "bias not absorbed: vz drifts"
    assert float(np.abs(zs[-len(t) // 3:]).max()) < 0.15, "bias not absorbed: z drifts"


def test_unseeded_and_bad_dt_are_noops():
    """Robustness pins: un-seeded outputs are NaN and every call no-ops; dt<=0 (between-IMU control
    tick) and dt>max_dt (sim reset/stutter) skip integration — mirroring LinearKF.max_dt_s."""
    est = VerticalEstimator()
    assert not est.seeded and np.isnan(est.z) and np.isnan(est.vz)
    est.predict(5.0, 0.01)                       # un-seeded: must not raise / must stay NaN
    est.update_z(1.0, 0.1)
    assert np.isnan(est.z)
    est.seed(-2.0)
    z0, vz0 = est.z, est.vz
    est.predict(3.0, 0.0)                        # dt=0: no-op
    est.predict(3.0, 1.0)                        # dt >> max_dt_s: rejected (garbage step)
    assert est.z == z0 and est.vz == vz0
    assert est.z == -2.0 and est.vz == 0.0


def test_a_up_decode_matches_verified_convention():
    """The a_up helper is the VERIFIED analyze_flight decode in matrix form: at rest with a correct
    attitude, specific force is -g on body-down and a_up reads ~0; a 1 m/s^2 hover-climb reads +1."""
    R_level = np.eye(3)
    f_rest = np.array([0.0, 0.0, -G])            # on-pad specific force, body FRD (|f| = g)
    assert abs(a_up_from_specific_force(f_rest, R_level)) < 1e-12
    f_climb = np.array([0.0, 0.0, -(G + 1.0)])   # rotors push 1 m/s^2 beyond gravity
    assert abs(a_up_from_specific_force(f_climb, R_level) - 1.0) < 1e-12
    # tilted airframe (the -17.8 deg resting pitch): gravity projects onto body x AND z; with the
    # CORRECT R the decode still reads ~0 at rest (the tilt cancels through the rotation).
    p = np.deg2rad(-17.8)
    R_pitch = np.array([[np.cos(p), 0.0, np.sin(p)], [0.0, 1.0, 0.0], [-np.sin(p), 0.0, np.cos(p)]])
    f_tilted = R_pitch.T @ f_rest                # the tilted accelerometer's view of rest
    assert abs(a_up_from_specific_force(f_tilted, R_pitch)) < 1e-9


# ---------------------------------------------------------------------------
# controller-level DoD: the damper reads the estimator, not the teleporting z
# ---------------------------------------------------------------------------
def _teleporting_nav_arc(n=40, seed=0):
    """The LIVE failure shape: the nav z creeps (dead-reckoning misses the climb) then TELEPORTS
    -1.4 m when a floor pin lands, while the TRUE motion (what the vertical estimator sees) is a
    smooth climb. Returns (t, nav_z, true_z, true_vz)."""
    t = np.arange(n) * DT_NS
    ts = t / 1e9
    true_z = -np.clip(0.5 * ts, 0.0, 1.6)                 # smooth 0.5 m/s climb to -1.6 m
    true_vz = np.gradient(true_z, ts)
    nav_z = 0.02 * np.random.default_rng(seed).standard_normal(n)   # dead-reckoned: flat + noise
    nav_z[n // 2:] = true_z[n // 2:] + 0.02                          # the pin teleport, one tick
    return t, nav_z, true_z, true_vz


def _run_ctrl(ctrl, t, nav_z, vert_z=None, vert_vz=None):
    """Replay through the alt-hold at level attitude with the pursuit-style setpoint (accel_ned set,
    position None) — the exact regime where the vertical channel is the sole thrust driver."""
    out = np.zeros(len(t))
    for i in range(len(t)):
        kw = {}
        if vert_z is not None:
            kw = dict(vert_z_est=float(vert_z[i]), vert_vz_est=float(vert_vz[i]))
        nav = NavState(sim_time_ns=int(t[i]),
                       position_ned=np.array([0.0, 0.0, nav_z[i]]),
                       velocity_ned=np.zeros(3),
                       roll=0.0, pitch=0.0, yaw=0.0, angular_rate_body=np.zeros(3), **kw)
        sp = Setpoint(sim_time_ns=int(t[i]),
                      accel_ned=1.2 * np.array([1.0, 0.0, 0.0]), yaw=0.0)
        out[i] = float(ctrl.command(nav, sp).thrust)
    return out


def test_controller_damps_on_estimator_not_teleporting_z():
    """DoD #2: flag ON + a live estimator export -> the alt-hold sees the smooth (z, vz) pair, so a
    nav-z teleport tick produces NO thrust slam; flag OFF on the same inputs slams (the fd of the
    teleport step). This is the damper reading REAL vertical velocity instead of teleport steps."""
    t, nav_z, true_z, true_vz = _teleporting_nav_arc()
    thr_off = _run_ctrl(make_seeker_controller(ff_owns_vertical=True), t, nav_z)
    thr_on = _run_ctrl(make_seeker_controller(ff_owns_vertical=True, use_vertical_estimator=True),
                       t, nav_z, vert_z=true_z, vert_vz=true_vz)
    i_tp = len(t) // 2                                     # the teleport tick
    step_off = abs(thr_off[i_tp] - thr_off[i_tp - 1])
    step_on = abs(thr_on[i_tp] - thr_on[i_tp - 1])
    assert step_off > 0.2, ("repro lost: OFF no longer slams on the teleport", step_off)
    assert step_on < 0.05, ("ON still slams on the teleport", step_on)
    # and ON actually OPPOSES the real climb: climbing (vz<0, NED) at a held target -> thrust cut
    # below hover on the climb stretch (the damper is live to the motion the nav z never showed).
    climb = slice(2, i_tp)
    assert float(np.mean(thr_on[climb])) < make_seeker_controller().hover_thrust, \
        "ON is blind to the real climb"


def test_controller_falls_back_when_estimator_absent():
    """Flag ON but NaN export (estimator off / not yet seeded) -> EXACTLY the A19c fd path: thrust
    trace identical to the flag-OFF controller on the same inputs (a seamless degrade, no fork)."""
    t, nav_z, _, _ = _teleporting_nav_arc(seed=3)
    thr_off = _run_ctrl(make_seeker_controller(ff_owns_vertical=True), t, nav_z)
    thr_nan = _run_ctrl(make_seeker_controller(ff_owns_vertical=True, use_vertical_estimator=True),
                        t, nav_z)                          # NavState vert_* default NaN
    assert np.allclose(thr_off, thr_nan, atol=0.0), "NaN fallback diverged from the fd path"


def test_off_path_byte_identical_even_with_export_present():
    """OFF-path byte-identity: with use_vertical_estimator OFF, a nav that CARRIES a live export is
    ignored — the thrust equals the fd-path controller exactly (the flag is the only fork)."""
    t, nav_z, true_z, true_vz = _teleporting_nav_arc(seed=5)
    thr_plain = _run_ctrl(make_seeker_controller(ff_owns_vertical=True), t, nav_z)
    thr_carrying = _run_ctrl(make_seeker_controller(ff_owns_vertical=True), t, nav_z,
                             vert_z=true_z, vert_vz=true_vz)
    assert np.allclose(thr_plain, thr_carrying, atol=0.0), "OFF path read the export"


# ---------------------------------------------------------------------------
# navigator plumbing: own/step/pin/export behind the config flag
# ---------------------------------------------------------------------------
def _ds(t_s, accel=(0.0, 0.0, -G), gyro=(0.0, 0.0, 0.0)):
    return DroneState(sim_time_ns=int(t_s * 1e9),
                      accel_body=np.asarray(accel, dtype=np.float64),
                      gyro_body=np.asarray(gyro, dtype=np.float64),
                      orientation_ned_wxyz=np.array([1.0, 0.0, 0.0, 0.0]))


def test_navigator_exports_and_pin_feeds_the_channel(monkeypatch):
    """Wiring: use_vertical_estimator=True -> NavState carries a finite (z, vz) export; an ACCEPTED
    floor pin (estimate_floor_height monkeypatched) nudges the channel z toward the pin (bounded);
    default config -> fields stay NaN (byte-identical export)."""
    from racer import navigator as nav_mod
    from racer.contracts import Frame
    from racer.navigator import Navigator, NavigatorConfig

    class _NoDetections:
        """Vision must be enabled for the floor channel to run; gate detections are irrelevant."""
        def detect(self, frame):
            return []

    cfg = NavigatorConfig(use_given_position=False, use_given_velocity=False,
                          use_ahrs=True, use_floor_height=True, use_vertical_estimator=True)
    nav = Navigator(gates=[], detector=_NoDetections(), config=cfg)
    ns = nav.update(_ds(0.0))
    assert np.isfinite(ns.vert_z_est) and np.isfinite(ns.vert_vz_est)   # seeded at init
    ns = nav.update(_ds(0.01))                                          # one IMU predict tick
    assert np.isfinite(ns.vert_vz_est)

    # a HIGH-QUALITY floor estimate: camera 2.0 m above the floor -> z_world = -2.0
    class _FloorEst:
        height_m, std_m, quality = 2.0, 0.1, 0.9
    monkeypatch.setattr(nav_mod, "estimate_floor_height", lambda *a, **k: _FloorEst())
    z_before = nav._vert_est.z
    frame = Frame(frame_id=1, sim_time_ns=int(0.02 * 1e9),
                  image_bgr=np.zeros((4, 4, 3), dtype=np.uint8))
    ns = nav.update(_ds(0.02), frame)
    assert nav.n_floor_z_total == 1, "the floor pin was not applied"
    # the pin (-2.0) pulled the channel z toward it, boundedly (not the full 2 m step)
    assert nav._vert_est.z < z_before, "pin did not correct the vertical channel"
    assert abs(nav._vert_est.z - z_before) < 1.0, "pin teleported the vertical channel"

    # default OFF: no estimator constructed, export stays NaN
    nav_off = Navigator(gates=[], detector=None,
                        config=NavigatorConfig(use_given_position=False, use_ahrs=True))
    ns_off = nav_off.update(_ds(0.0))
    assert nav_off._vert_est is None
    assert np.isnan(ns_off.vert_z_est) and np.isnan(ns_off.vert_vz_est)


# ---------------------------------------------------------------------------
# profile / default pins (the async_detect gating pattern, mirrored)
# ---------------------------------------------------------------------------
def test_defaults_are_off_everywhere():
    """The Controller / NavigatorConfig / DeployProfile defaults (and thus VQ1 + every offline
    path) leave the vertical estimator OFF — the byte-identical guarantee."""
    from racer.controller import Controller
    from racer.navigator import NavigatorConfig
    assert Controller().use_vertical_estimator is False
    assert NavigatorConfig().use_vertical_estimator is False
    assert DeployProfile.__dataclass_fields__["vertical_estimator"].default is False
    assert vq1_case_a().vertical_estimator is False


def test_vq2_profile_opts_in():
    """vq2_case_c ships the vertical estimator ON (the A21 fix flies by default on case C)."""
    assert vq2_case_c().vertical_estimator is True


def test_fly_rl_resolver_cli_overrides_profile():
    """The fly_rl seam mirrors _resolve_async_detect: 'auto' defers to the profile; on/off force."""
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
