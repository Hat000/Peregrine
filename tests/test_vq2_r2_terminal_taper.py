"""R2-1 VERTICAL TERMINAL-CROSSING TAPER — the terminal authority taper + rate-arrest boost.

THE BUG (live VQ2, run 20260704_135554): from command 0 the entire flight was a climb -- the drone
flew higher than gate 1's centerpoint and hit the CENTER OF THE TOP BAR (+0.75 m). The gate-PD
sought gate height correctly on the APPROACH (z_off -2.8 -> -0.14 as the drone climbed off the
floor) but the CLOSE-RANGE vision fix reads systematically HIGH -- the stack believed it was
centred at trk_range 3.7 m while physically ~1 m above. The measurement bias is UNIFORM
(theta_g-uncorrelated: terminal corr 0.01, so down-weighting cannot remove it).

THE FIX (Controller.gate_pd_terminal + gate_pd_rate_boost, Setpoint.gate_pd_scale,
GateSeekerConfig.gate_pd_terminal_lo/hi_range_m): the seeker supplies a range-tapered scale
``s = clip((rng - lo)/(hi - lo), 0, 1)`` on the pursuit Setpoint; the controller's gate-PD law
becomes ``thrust = hover - kp_gate*clip(z_off)*s + ff_vertical_kd_alt*(1 + kb*(1-s))*vz_lp``. Far
(s=1): today's A28 law EXACTLY. At the plane (s->0): position authority fades, rate brake doubles
-> cross LEVEL. Default OFF => byte-identical (VQ1 / case-A AND today's vq2 gate-PD law).

[VQ2 R2-1, 2026-07-04]
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from racer.contracts import NavState, Setpoint  # noqa: E402
from racer.controller import Controller  # noqa: E402
from racer.deploy_profile import vq1_case_a, vq2_case_c  # noqa: E402
from racer.gate_seeker import GateSeekerConfig, make_seeker_controller  # noqa: E402

DT_NS = int(1e9 / 12)


def _gate_pd_ctrl(**over):
    """The vq2 gate-PD vertical controller (A28 law) with the R2 terminal knobs as configured."""
    base = dict(ff_owns_vertical=True, use_vertical_estimator=True, gate_pd_vertical=True,
                kp_alt=0.0, kp_gate=0.04, ff_vertical_kd_alt=0.06, ff_vertical_vz_lp_alpha=0.8,
                alt_thrust_lo=0.05, alt_thrust_hi=0.6, alt_thrust_slew_per_s=2.0, tilt_comp=True)
    base.update(over)
    return make_seeker_controller(**base)


def _sp(t_ns, z_off, gate_pd_scale=None, vz_t=0.0):
    """A pursuit-style setpoint carrying accel_ned (ff-owns-vertical active) + the R2 scale."""
    vned = np.array([0.0, 0.0, vz_t]) if vz_t != 0.0 else None
    return Setpoint(sim_time_ns=int(t_ns), accel_ned=1.0 * np.array([1.0, 0.0, 0.0]),
                    velocity_ned=vned, yaw=0.0, gate_pd_scale=gate_pd_scale)


def _nav(t_ns, z, vz, z_off):
    # NavState is frozen; the washout export (vert_vz_est) + gate offset (z_off_est) are the fields
    # the ff-owns-vertical + gate-PD controller reads under use_vertical_estimator.
    return NavState(sim_time_ns=int(t_ns), position_ned=np.array([0.0, 0.0, z]),
                    velocity_ned=np.array([0.0, 0.0, vz]), roll=0.0, pitch=0.0, yaw=0.0,
                    angular_rate_body=np.zeros(3), vert_vz_est=vz, z_off_est=z_off)


# ---------------------------------------------------------------------------
# Byte-identity: OFF-path and s=None leave the A28 law EXACTLY as today
# ---------------------------------------------------------------------------

def test_default_off():
    """The Controller default (VQ1 / every offline path) leaves gate_pd_terminal OFF."""
    assert Controller().gate_pd_terminal is False


def test_flag_off_ignores_scale_byte_identical():
    """With gate_pd_terminal OFF, a non-None gate_pd_scale on the setpoint is IGNORED: the thrust
    equals the untapered A28 law for the SAME inputs -- the flag is the only behaviour change."""
    off = _gate_pd_ctrl(gate_pd_terminal=False)
    ref = _gate_pd_ctrl(gate_pd_terminal=False)
    for i, (zoff, s) in enumerate([(2.0, 0.0), (-2.0, 0.3), (0.5, 1.0), (-1.5, 0.7)]):
        t = i * DT_NS
        a = float(off.command(_nav(t, -2.0, 0.1, zoff), _sp(t, zoff, gate_pd_scale=s)).thrust)
        b = float(ref.command(_nav(t, -2.0, 0.1, zoff), _sp(t, zoff, gate_pd_scale=None)).thrust)
        assert abs(a - b) < 1e-12, (i, zoff, s, a, b)


def test_scale_one_is_a28_law_exactly():
    """gate_pd_terminal ON but s=1.0 (far from the plane) == the untapered A28 law EXACTLY."""
    on = _gate_pd_ctrl(gate_pd_terminal=True, gate_pd_rate_boost=1.0)
    off = _gate_pd_ctrl(gate_pd_terminal=False)
    for i, zoff in enumerate([2.0, -2.0, 0.5, -1.5, 3.5]):
        t = i * DT_NS
        a = float(on.command(_nav(t, -2.0, 0.2, zoff), _sp(t, zoff, gate_pd_scale=1.0)).thrust)
        b = float(off.command(_nav(t, -2.0, 0.2, zoff), _sp(t, zoff, gate_pd_scale=None)).thrust)
        assert abs(a - b) < 1e-12, (i, zoff, a, b)


def test_scale_none_on_flag_is_a28_law():
    """gate_pd_terminal ON but the setpoint carries NO scale (None) still == the A28 law (a
    non-pursuit tick / caller that doesn't taper -> s forced to 1.0)."""
    on = _gate_pd_ctrl(gate_pd_terminal=True)
    off = _gate_pd_ctrl(gate_pd_terminal=False)
    for i, zoff in enumerate([2.0, -2.0, 0.5]):
        t = i * DT_NS
        a = float(on.command(_nav(t, -2.0, 0.2, zoff), _sp(t, zoff, gate_pd_scale=None)).thrust)
        b = float(off.command(_nav(t, -2.0, 0.2, zoff), _sp(t, zoff, gate_pd_scale=None)).thrust)
        assert abs(a - b) < 1e-12, (i, zoff, a, b)


# ---------------------------------------------------------------------------
# The taper: position fades, rate brake boosts, single-tick term decomposition
# ---------------------------------------------------------------------------

def test_position_term_fades_at_the_plane():
    """As s -> 0 the gate-PD POSITION term (term_gate) scales linearly to 0 for a fixed z_off."""
    ctrl = _gate_pd_ctrl(gate_pd_terminal=True, gate_pd_rate_boost=1.0)
    zoff = 2.0
    terms = {}
    for i, s in enumerate([1.0, 0.5, 0.0]):
        t = i * DT_NS
        ctrl.command(_nav(t, -2.0, 0.0, zoff), _sp(t, zoff, gate_pd_scale=s))
        terms[s] = ctrl._last_term_gate
    # linear scaling: term(0.5) == 0.5*term(1.0); term(0.0) == 0.0
    assert abs(terms[1.0] - (-0.04 * 2.0)) < 1e-9, terms
    assert abs(terms[0.5] - 0.5 * terms[1.0]) < 1e-9, terms
    assert abs(terms[0.0]) < 1e-12, terms


def test_rate_brake_boosts_at_the_plane():
    """As s -> 0 the damping term scales to ff_vertical_kd_alt*(1+kb)*vz_lp -- the brake DOUBLES at
    kb=1.0. Isolate vz_lp by feeding a constant vz so the LP settles, comparing s=1 vs s=0."""
    zoff = 0.0                            # kill the position term -> the damping term alone remains
    vz = 0.8                              # a steady descent rate (down-positive)
    far = _gate_pd_ctrl(gate_pd_terminal=True, gate_pd_rate_boost=1.0)
    plane = _gate_pd_ctrl(gate_pd_terminal=True, gate_pd_rate_boost=1.0)
    d_far = d_plane = None
    for i in range(20):                   # let the vz_lp low-pass settle to vz
        t = i * DT_NS
        far.command(_nav(t, -2.0, vz, zoff), _sp(t, zoff, gate_pd_scale=1.0))
        plane.command(_nav(t, -2.0, vz, zoff), _sp(t, zoff, gate_pd_scale=0.0))
        d_far, d_plane = far._last_term_damp, plane._last_term_damp
    # at the plane the brake is 2x the far brake (1 + kb*(1-0) = 2)
    assert abs(d_plane - 2.0 * d_far) < 1e-6, (d_far, d_plane)
    assert d_far > 0.0                    # descending (vz>0) -> positive brake (more thrust) far


def test_boost_zero_only_fades_position():
    """gate_pd_rate_boost=0 => the damping term is UNCHANGED by s (only the position term fades)."""
    zoff, vz = 0.0, 0.8
    kb0 = _gate_pd_ctrl(gate_pd_terminal=True, gate_pd_rate_boost=0.0)
    d1 = d0 = None
    for i in range(20):
        t = i * DT_NS
        kb0.command(_nav(t, -2.0, vz, zoff), _sp(t, zoff, gate_pd_scale=1.0)); d1 = kb0._last_term_damp
    kb0b = _gate_pd_ctrl(gate_pd_terminal=True, gate_pd_rate_boost=0.0)
    for i in range(20):
        t = i * DT_NS
        kb0b.command(_nav(t, -2.0, vz, zoff), _sp(t, zoff, gate_pd_scale=0.0)); d0 = kb0b._last_term_damp
    assert abs(d1 - d0) < 1e-9, (d1, d0)


def test_taper_arrests_a_terminal_climb():
    """The whole point: a drone ABOVE the gate (z_off<0 => "gate above" => the biased close-range
    read that commanded MORE thrust) still CLIMBING (vz<0) crosses closer to LEVEL under the taper
    than under the untapered A28 law. Compare the net upward thrust bias near the plane."""
    zoff = -1.0                           # biased "gate above" read -> position term wants MORE thrust
    vz = -0.6                             # climbing (down-positive, so climb is negative)
    tapered = _gate_pd_ctrl(gate_pd_terminal=True, gate_pd_rate_boost=1.0)
    untapered = _gate_pd_ctrl(gate_pd_terminal=False)
    thr_t = thr_u = None
    for i in range(20):
        t = i * DT_NS
        thr_t = float(tapered.command(_nav(t, -2.0, vz, zoff), _sp(t, zoff, gate_pd_scale=0.0)).thrust)
        thr_u = float(untapered.command(_nav(t, -2.0, vz, zoff), _sp(t, zoff, gate_pd_scale=None)).thrust)
    # the untapered law adds hover + (positive position term for zoff<0) + (negative brake for climb);
    # the tapered law fades the positive position term to 0 AND doubles the (thrust-reducing) climb
    # brake -> strictly LESS thrust near the plane -> the climb is arrested harder.
    assert thr_t < thr_u, (thr_t, thr_u)


# ---------------------------------------------------------------------------
# Seeker range-taper: g_v = clip((rng-lo)/(hi-lo), 0, 1), NO floor
# ---------------------------------------------------------------------------

def _seeker_with_taper(**over):
    from racer.gate_seeker import GateSeeker
    cfg = GateSeekerConfig(gate_pd_terminal_lo_range_m=3.0, gate_pd_terminal_hi_range_m=8.0, **over)
    return GateSeeker(config=cfg, controller=make_seeker_controller())


def test_seeker_scale_none_when_taper_disabled():
    """Taper lo unset => _gate_pd_scale() is None (byte-identical: Setpoint carries None -> 1.0)."""
    from racer.gate_seeker import GateSeeker
    sk = GateSeeker(config=GateSeekerConfig(), controller=make_seeker_controller())
    sk._track_range_m = 5.0
    assert sk._gate_pd_scale() is None


def test_seeker_scale_none_without_range():
    """No tracked range yet => None (no scale to send)."""
    sk = _seeker_with_taper()
    sk._track_range_m = None
    sk._last_track_range_m = None
    assert sk._gate_pd_scale() is None


def test_seeker_scale_ramps_no_floor():
    """g_v = clip((rng-3)/(8-3), 0, 1): 1.0 at/above 8 m, 0.0 at/below 3 m (NO floor), linear between."""
    sk = _seeker_with_taper()
    for rng, want in [(9.0, 1.0), (8.0, 1.0), (5.5, 0.5), (3.0, 0.0), (2.0, 0.0), (6.5, 0.7)]:
        sk._track_range_m = rng
        got = sk._gate_pd_scale()
        assert abs(got - want) < 1e-9, (rng, got, want)
    # stashed for instrumentation
    assert sk._last_gate_pd_scale is not None


def test_seeker_falls_back_to_last_range():
    """When the live track range is None but a last range exists, the taper uses the last range."""
    sk = _seeker_with_taper()
    sk._track_range_m = None
    sk._last_track_range_m = 8.0
    assert abs(sk._gate_pd_scale() - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Profile wiring
# ---------------------------------------------------------------------------

def test_vq2_profile_enables_r2():
    ov = vq2_case_c().controller_overrides
    assert ov.get("gate_pd_terminal") is True
    assert ov.get("gate_pd_rate_boost") == 1.0
    so = vq2_case_c().seeker_overrides
    assert so.get("gate_pd_terminal_lo_range_m") == 3.0
    assert so.get("gate_pd_terminal_hi_range_m") == 8.0


def test_vq1_profile_leaves_r2_off():
    assert vq1_case_a().controller_overrides is None
    assert vq1_case_a().seeker_overrides is None
