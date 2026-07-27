"""GEOMETRIC gate-detectability (``--ego-det-geometric``, default OFF) -- the deploy mirror of
training's ``rl/gate_visibility.py::gate_detectable``.

WHAT IS BEING GUARDED, in priority order:

 1. DEFAULT-OFF IS ABSOLUTE. With the knob absent the emitted obs must be BITWISE what it was, the
    geometry function must NEVER BE CALLED (pinned by poisoning it, not by comparing numbers), and
    no new key may appear in ``last_diag``.
 2. THE MASK MUST FIRE AT THE RIGHT RANGE. A mask that fires at 5 m or at 0.2 m is worse than none.
    The calibration is pinned twice: against TRAINING's published horizon table (exact, millimetre
    tolerance, boresight neutralised so the comparison is apples-to-apples) and against the WIRE's
    measured 1.8-2.1 m median last-sighted range on the SHIPPED config.
 3. THE FRAME MUST NOT INVERT. The camera axis is pitched +20 deg UP with a 58.7 deg VFOV, so the
    band is asymmetric (-9.4, +49.4) deg: a gate BELOW the axis must clip sooner than one above.
    Getting the attitude sign wrong flips that, and the range calibration alone would not catch it.
 4. NO GROUND TRUTH, and no interaction with the aim-offset dodge.

Training's numbers quoted below are not hearsay: they were reproduced by running the real
``gate_visibility.gate_detectable`` (torch) against this implementation over the same geometries --
delta 0.000 on every straight-in case, median +0.009 m over 398 random realistic approaches.
"""
import glob
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from racer import frames                                          # noqa: E402
from racer.contracts import GatePose                              # noqa: E402
from racer.ego_obs import (                                       # noqa: E402
    GATE_VIS_FAR_CAP_M,
    GATE_VIS_INNER_HALF_M,
    GATE_VIS_MIN_CORNERS,
    GATE_VIS_OUTER_HALF_M,
    EgoObsBuilder,
    EgoObsBuilderConfig,
    gate_detectable_geometric,
    rel_pos_body_frd_from_gatepose,
)

_FLIP = np.array([1.0, -1.0, -1.0])


# =================================================================================================
# helpers
# =================================================================================================
def _R_zup(roll=0.0, pitch=0.0, yaw=0.0):
    """Body-FLU -> world-Z-up rotation, ZYX, the convention ``roll_pitch_zup`` inverts."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rel_flu_straight_in(r, R, dz=0.0, dy=0.0):
    """TRUE body-FLU lever for a drone ``r`` m short of a gate, offset dy left-ish / dz up in WORLD.
    Drone at world (-r, dy, dz), gate at the origin -- the geometry the training horizon table used
    (``scripts/vision_horizon/train_vs_wire_horizon.py``)."""
    return R.T @ (np.zeros(3) - np.array([-r, dy, dz]))


def _cutoff(R, dz=0.0, dy=0.0, lo=0.15, hi=9.0, tol=1e-4, **kw):
    """Closest range still detectable, or None.

    🚩 DETECTABILITY IS A BAND, NOT A HALF-LINE, so a plain bisection from ``hi`` is the WRONG
    instrument and silently returns None on steep attitudes: at pitch -24 deg the up-looking camera
    is aimed 44 deg above the approach, a gate 9 m out is already outside the frame, and only a
    mid-range window is detectable at all. Scan DOWN for the first detectable range (exactly what
    scripts/vision_horizon/train_vs_wire_horizon.py does), then bisect inside that bracket."""
    def ok(r):
        return gate_detectable_geometric(_rel_flu_straight_in(r, R, dz, dy), R, **kw)[0]
    lastok = None
    for r in np.arange(hi, lo, -0.02):
        if ok(float(r)):
            lastok = float(r)
        elif lastok is not None:
            break
    if lastok is None:
        return None
    a, b = max(lo, lastok - 0.02), lastok          # ok(b) True, ok(a) False (or a == lo)
    if ok(a):
        return a
    while b - a > tol:
        m = 0.5 * (a + b)
        if ok(m):
            b = m
        else:
            a = m
    return b


@pytest.fixture
def no_boresight(monkeypatch):
    """Neutralise the METRIC boresight so the comparison against TRAINING (which has no such term)
    is apples-to-apples. ``frames.BORESIGHT`` is read LIVE by both the lever transform and the
    visibility geometry, so patching the module attribute is the whole switch."""
    from racer.frames import BoresightCorrection
    monkeypatch.setattr(frames, "BORESIGHT", BoresightCorrection())
    return None


def _pose(rel_flu, t_ns=0, frame_id=0):
    """A GatePose whose lever lands the builder's held ``rel_flu`` exactly on the requested vector
    (inverts ``rel_pos_body_frd_from_gatepose``). Orientation is identity: this contract never uses
    R_cam_gate for the visibility geometry, which is the point."""
    rel_frd = _FLIP * np.asarray(rel_flu, dtype=np.float64)
    t_cam = frames.R_camera_from_body() @ (rel_frd - np.array([0.0, 0.0, frames.BORESIGHT.vert_offset_m]))
    return GatePose(frame_id=frame_id, sim_time_ns=t_ns, R_cam_gate=np.eye(3),
                    t_cam_gate=t_cam, reproj_error_px=0.5, n_corners=4, visible_area_meas=0.9)


# =================================================================================================
# 1. DEFAULT-OFF IS ABSOLUTE
# =================================================================================================
def test_default_is_off():
    assert EgoObsBuilderConfig().det_geometric is False


def test_default_path_never_calls_the_geometry(monkeypatch):
    """The strongest form of "byte-identical": POISON ``gate_detectable_geometric`` and run a full
    mixed sequence. If the default path touched it at all, this raises. Comparing numbers could not
    prove this -- the geometry might agree by luck on the sampled ticks."""
    import racer.ego_obs as M

    def _boom(*a, **k):
        raise AssertionError("gate_detectable_geometric was called on the DEFAULT path")

    monkeypatch.setattr(M, "gate_detectable_geometric", _boom)
    b = EgoObsBuilder(EgoObsBuilderConfig(slot1_enabled=True))
    t = 1_000_000_000
    for k in range(200):
        t += 33_000_000
        seen = (k % 7) < 4
        b.update(sim_time_ns=t, gate_index=k // 80, R_frd2ned=np.eye(3),
                 vel_ned=np.array([6.0, 0.2, -0.1]), gyro_frd=np.array([0.1, -0.2, 0.3]),
                 pose=_pose([12.0 - 0.05 * (k % 80), 0.1, 0.2], t) if seen else None,
                 next_pose=_pose([26.0, 0.4, 0.1], t) if (k % 5) == 0 else None,
                 last_normed_thrust=1.0)


def test_off_is_bitwise_identical_over_a_mixed_sequence():
    """Explicit-OFF and default must agree BITWISE across fixes, gaps past det_hold, re-acquisitions
    and gate advances -- and the OFF log must carry no new key."""
    rng = np.random.default_rng(5)
    a = EgoObsBuilder(EgoObsBuilderConfig(slot1_enabled=True))
    b = EgoObsBuilder(EgoObsBuilderConfig(slot1_enabled=True, det_geometric=False))
    t = 1_000_000_000
    for k in range(400):
        t += 33_000_000
        seen = (k % 7) < 4
        r = 12.0 - 0.05 * (k % 150)
        kw = dict(sim_time_ns=t, gate_index=k // 150, R_frd2ned=np.eye(3),
                  vel_ned=np.array([6.0, 0.2, -0.1]), gyro_frd=np.array([0.1, -0.2, 0.3]),
                  pose=_pose([r, float(rng.normal(0, .3)), float(rng.normal(0, .3))], t) if seen else None,
                  next_pose=_pose([26.0, 0.4, 0.1], t) if (k % 5) == 0 else None,
                  last_normed_thrust=1.0)
        assert a.update(**kw).tobytes() == b.update(**kw).tobytes(), f"diverged at tick {k}"
    for d in (a.last_diag, b.last_diag):
        assert "det_geom" not in d and "det_corners" not in d
        assert "det_geom1" not in d and "det_corners1" not in d


def test_armed_emits_the_diag_keys():
    b = EgoObsBuilder(EgoObsBuilderConfig(det_geometric=True, slot1_enabled=True))
    b.update(sim_time_ns=1_000_000_000, gate_index=0, R_frd2ned=np.eye(3), vel_ned=np.zeros(3),
             gyro_frd=np.zeros(3), pose=_pose([8.0, 0.0, 0.0]), next_pose=_pose([24.0, 0.0, 0.0]),
             last_normed_thrust=0.0)
    for key in ("det_geom", "det_corners", "det_geom1", "det_corners1"):
        assert key in b.last_diag, key
    assert b.last_diag["det_geom"] is True
    # 6, NOT 8, and the 2 that are missing are the OUTER BOTTOM pair: the optical axis is +20 deg UP
    # so a gate dead ahead sits only 9.4 deg above the bottom frame edge, and the outer ring (2.7 m,
    # half-width 1.35) subtends 9.6 deg at 8 m -- it has already overflowed downward. Pinned as a
    # constant because it is exactly the up-bias this whole contract turns on.
    assert b.last_diag["det_corners"] == 6
    # det_proxy must keep meaning the AGE proxy ALONE (the arrestor + every log analysis read it)
    assert b.last_diag["det_proxy"] is True


# =================================================================================================
# 2. CALIBRATION -- the mask must fire where the gate actually leaves the frame
# =================================================================================================
@pytest.mark.parametrize("label,kw,expect", [
    ("head-on, level, centred",          dict(),                              1.290),
    ("level, 0.3 m HIGH (drone above)",  dict(dz=0.3),                        1.185),
    ("level, 0.3 m LOW",                 dict(dz=-0.3),                       1.550),
    ("pitch +10",                        dict(pitch=np.deg2rad(10)),          1.780),
    ("pitch +24 (the flown median)",     dict(pitch=np.deg2rad(24)),          1.715),
    ("pitch -10",                        dict(pitch=np.deg2rad(-10)),         0.930),
    ("pitch -24",                        dict(pitch=np.deg2rad(-24)),         0.705),
    ("roll 30, level",                   dict(roll=np.deg2rad(30)),           1.800),
])
def test_matches_training_horizon_table(no_boresight, label, kw, expect):
    """EXACT parity with training's own ``gate_detectable`` on straight-in approaches.

    These eight numbers were produced by RUNNING rl/gate_visibility.gate_detectable (torch 2.12,
    the real training source) over the same geometries; this implementation reproduced every one
    to 0.000 m. They are therefore a genuine cross-implementation pin, not a self-fit: if the
    keypoint set, the intrinsics, the mount, the >=4 rule or the attitude handling drifts, this
    breaks. Tolerance 5 mm covers only the bisection step.

    THIS TABLE ALSO PINS EVERY SIGN. Pitch +-24 deg is 1.715 vs 0.705 m and dz +-0.3 m is 1.185 vs
    1.550 m, so an inverted attitude sign (the ``true pitch = -obs[4]`` trap) or an inverted lever
    cannot pass. Angles are the Z-up ZYX convention ``roll_pitch_zup`` inverts -- the same one the
    training horizon table was generated in -- NOT the virtual-flipped obs[3:5].
    """
    dz = kw.pop("dz", 0.0)
    dy = kw.pop("dy", 0.0)
    got = _cutoff(_R_zup(**kw), dz=dz, dy=dy)
    assert got is not None, f"{label}: never detectable"
    assert abs(got - expect) < 5e-3, f"{label}: cutoff {got:.3f} m, training says {expect:.3f} m"


def test_shipped_config_masks_in_the_measured_wire_band():
    """THE CALIBRATION THAT MATTERS: with the boresight LIVE (what actually flies) and the drone at
    the flown median attitude, the geometric mask must go False at a range consistent with the
    measured last-sighted range over 899 confirmed passes -- p10 1.48 / MEDIAN 1.78 / p90 2.80 m.

    Not 5 m (that would blind the policy through the whole final approach) and not 0.2 m (that
    would reproduce today's bug). The band asserted is deliberately wider than the point estimate
    so that a boresight recalibration does not fail the suite spuriously -- but it is narrow enough
    that either failure mode above trips it."""
    assert frames.BORESIGHT.vert_offset_m == -0.25, "shipped boresight moved; re-read the band below"
    got = _cutoff(_R_zup(pitch=np.deg2rad(24)))
    assert got is not None
    assert 1.30 < got < 2.40, f"geometric cutoff {got:.3f} m is outside the measured wire band"


def test_far_away_is_detectable_and_beyond_the_far_cap_is_not():
    R = _R_zup(pitch=np.deg2rad(20))
    assert gate_detectable_geometric(_rel_flu_straight_in(10.0, R), R)[0] is True
    assert gate_detectable_geometric(_rel_flu_straight_in(29.0, R), R)[0] is True
    # training's FAR_CAP_M_DEFAULT is a hard cut on the CENTRE range, not a visibility test
    assert gate_detectable_geometric(_rel_flu_straight_in(31.0, R), R)[0] is False
    assert GATE_VIS_FAR_CAP_M == 30.0


def test_corner_count_decays_monotonically_through_the_run_in():
    """The count is the calibration instrument written to ego_obs.jsonl: it must fall smoothly from
    8 to 0 as the gate fills the frame, crossing the >=4 threshold once."""
    R = _R_zup(pitch=np.deg2rad(24))
    counts = [gate_detectable_geometric(_rel_flu_straight_in(r, R), R)[1]
              for r in np.arange(8.0, 0.5, -0.05)]
    assert counts[0] == 8 and counts[-1] == 0
    assert all(x >= y for x, y in zip(counts, counts[1:])), "corner count is not monotone"
    crossings = sum(1 for x, y in zip(counts, counts[1:])
                    if (x >= GATE_VIS_MIN_CORNERS) != (y >= GATE_VIS_MIN_CORNERS))
    assert crossings == 1, f"the >=4 threshold is crossed {crossings} times, expected once"


# =================================================================================================
# 3. THE FRAME MUST NOT INVERT
# =================================================================================================
def test_vertical_asymmetry_has_the_sign_the_up_looking_camera_produces(no_boresight):
    """The optical axis is +20 deg UP over a 58.7 deg VFOV, so the body elevation band is
    (-9.4, +49.4) deg -- strongly asymmetric. The vertical cutoff must therefore be asymmetric too,
    and an inverted attitude sign flips it while leaving every symmetric statistic untouched.

    🚩 THE DIRECTION IS THE OPPOSITE OF THE OBVIOUS ONE, and it is worth knowing why, because the
    closed form everyone quotes gets the mechanism wrong even though it gets the number right.
    Instrumented at the threshold (which corner leaves through which edge): the 4 BOTTOM corners
    have ALREADY overflowed the bottom edge long before the cutoff, so the surviving four are the
    2 inner-TOP + 2 outer-TOP, and the cutoff fires when the OUTER-TOP pair leaves. Raising the
    gate (drone BELOW it) pushes that pair out through the TOP edge sooner -> a LONGER cutoff
    (1.550 m); lowering it lets the pair survive until the 2.7 m outer ring overflows sideways ->
    a SHORTER cutoff (1.185 m). So what binds at 4-of-8 is the OUTER ring, not the 0.75 m inner
    half-width the ``|el| + atan(0.75/R) = VFOV/2`` closed form uses. That closed form predicted
    the measured wire median to +0.04 m, but it does so with the wrong mechanism -- do not extend
    it to new geometry and expect it to hold.
    """
    lo, hi = frames.camera_elevation_band_deg()
    assert lo < 0 < hi and abs(lo) < abs(hi), "camera band is no longer up-biased; re-read this test"
    gate_low = _cutoff(_R_zup(), dz=+0.3)      # drone ABOVE the gate -> gate LOW in frame
    gate_high = _cutoff(_R_zup(), dz=-0.3)     # drone BELOW the gate -> gate HIGH in frame
    assert gate_low is not None and gate_high is not None
    assert gate_high > gate_low + 0.25, (
        f"gate-low cutoff {gate_low:.3f} m vs gate-high {gate_high:.3f} m -- the +20 deg up-pitch "
        "vertical asymmetry is absent or inverted")


def test_roll_is_read_from_the_attitude_not_ignored():
    level = _cutoff(_R_zup())
    rolled = _cutoff(_R_zup(roll=np.deg2rad(45)))
    assert level is not None and rolled is not None
    assert abs(rolled - level) > 0.2, "rolling the drone did not move the visibility horizon at all"


def test_heading_yaw_cannot_reach_the_answer():
    """YAW IS NOT OBSERVABLE in this contract and must not leak into the geometry. World-UP is taken
    as ``R[2, :] == R^T @ e_z``, which is yaw-free by construction, so for a FIXED body-frame lever
    pre-multiplying the attitude by any Rz must return the IDENTICAL answer.

    (Note this is a statement about the FUNCTION, not about flying in a circle: turning the drone
    while the gate stays put changes the lever, and that is a different geometry.)"""
    rel = np.array([2.0, 0.15, -0.1])
    R0 = _R_zup(roll=0.2, pitch=0.35)
    base = gate_detectable_geometric(rel, R0)
    for yaw in (0.4, -1.1, 2.7, np.pi):
        assert gate_detectable_geometric(rel, _R_zup(yaw=yaw) @ R0) == base


# =================================================================================================
# 4. NO GROUND TRUTH, contract round-trip, guards, and the aim-offset interaction
# =================================================================================================
def test_reconstructs_the_pnp_lever_exactly_at_a_fresh_fix():
    """The boresight is UNDONE inside the geometry so that the projected gate centre is exactly the
    PnP ``t_cam_gate`` the detector produced. Pin that inverse: it is the reason this test evaluates
    the geometry the detector itself saw, and it is the only justification for including a term
    training does not have."""
    p = _pose([4.0, 0.7, -0.3])
    rel_frd = rel_pos_body_frd_from_gatepose(p.t_cam_gate)
    rel_flu = _FLIP * rel_frd
    back = frames.R_camera_from_body() @ (
        _FLIP * rel_flu - np.array([0.0, 0.0, frames.BORESIGHT.vert_offset_m]))
    np.testing.assert_allclose(back, p.t_cam_gate, atol=1e-12)


def test_uses_only_the_lever_and_the_attitude():
    """A pure function of (held lever, attitude): no gate id, no world pose, no map, no truth. Same
    inputs -> same answer, and nothing else can reach it."""
    R = _R_zup(pitch=0.3, roll=-0.2)
    rel = np.array([2.4, 0.3, -0.1])
    assert gate_detectable_geometric(rel, R) == gate_detectable_geometric(rel.copy(), R.copy())


def test_guards_return_not_detectable():
    R = _R_zup()
    assert gate_detectable_geometric(None, R) == (False, 0)
    assert gate_detectable_geometric([np.nan, 1.0, 0.0], R) == (False, 0)
    assert gate_detectable_geometric([np.inf, 1.0, 0.0], R) == (False, 0)
    assert gate_detectable_geometric([0.0, 0.0, 0.0], R) == (False, 0)
    assert gate_detectable_geometric([3.0, 0.0, 0.0], np.zeros((3, 3))) == (False, 0)


def test_gate_straight_overhead_does_not_nan():
    """Degenerate branch: the horizontal LOS vanishes so the world-vertical gate model is undefined.
    Must return a clean boolean, never raise or emit NaN."""
    R = _R_zup()
    det, n = gate_detectable_geometric([0.0, 0.0, 3.0], R)
    assert isinstance(det, bool) and 0 <= n <= 8


def test_uses_the_physical_half_width_not_the_pass_aperture():
    """Visibility geometry uses the PHYSICAL gate (inner half 0.75 m, outer half 1.35 m). The
    ``0.75 - body_radius`` figure (0.37-0.47 m) is the PASS aperture and belongs nowhere near this."""
    assert GATE_VIS_INNER_HALF_M == 0.75 and GATE_VIS_OUTER_HALF_M == 1.35


# ---- integration through the builder -------------------------------------------------------------
def test_armed_masks_the_slot_in_the_blind_run_in():
    """End to end: fly a gate in on FRESH fixes the whole way (so the age proxy NEVER fires) and
    check that the armed builder zeros obs[11:16] on the final approach while the default builder
    keeps feeding a filled lever. This IS the measured defect -- 65.7% of approaches stay filled
    through the blind run-in -- reproduced in a unit test."""
    off = EgoObsBuilder(EgoObsBuilderConfig(virtual_flip=False))
    on = EgoObsBuilder(EgoObsBuilderConfig(virtual_flip=False, det_geometric=True))
    R_frd2ned = np.eye(3)                     # level; body FLU == world Z-up
    t = 1_000_000_000
    fed_off = fed_on = 0
    for r in np.arange(6.0, 0.49, -0.10):     # a fix EVERY tick: age is always ~0
        t += 33_000_000
        p = _pose([float(r), 0.0, 0.0], t)
        o_off = off.update(sim_time_ns=t, gate_index=0, R_frd2ned=R_frd2ned, vel_ned=np.zeros(3),
                           gyro_frd=np.zeros(3), pose=p, last_normed_thrust=0.0)
        o_on = on.update(sim_time_ns=t, gate_index=0, R_frd2ned=R_frd2ned, vel_ned=np.zeros(3),
                         gyro_frd=np.zeros(3), pose=p, last_normed_thrust=0.0)
        if r < 1.0:                            # training zeros slot0 on 100% of ticks inside 1.0 m
            fed_off += int(np.any(o_off[11:16] != 0.0))
            fed_on += int(np.any(o_on[11:16] != 0.0))
    assert fed_off > 0, "the OFF path should still be feeding a filled lever inside 1 m (the defect)"
    assert fed_on == 0, "armed, slot0 must be ZEROS inside 1 m -- that is training's state"


def test_armed_never_unmasks_a_tick_the_default_path_masked():
    """AND, not REPLACE: the armed builder may mask ticks the default fed, but must NEVER feed a
    tick the default masked. This is the whole safety argument for the composition."""
    rng = np.random.default_rng(17)
    off = EgoObsBuilder(EgoObsBuilderConfig(slot1_enabled=True))
    on = EgoObsBuilder(EgoObsBuilderConfig(slot1_enabled=True, det_geometric=True))
    t = 1_000_000_000
    masked_extra = 0
    for k in range(500):
        t += 33_000_000
        seen = (k % 6) < 3
        r = max(0.4, 14.0 - 0.09 * (k % 150))
        kw = dict(sim_time_ns=t, gate_index=k // 150, R_frd2ned=np.eye(3),
                  vel_ned=np.array([7.0, 0.1, 0.0]), gyro_frd=np.array([0.05, -0.1, 0.2]),
                  pose=_pose([r, float(rng.normal(0, .25)), float(rng.normal(0, .25))], t) if seen else None,
                  next_pose=_pose([r + 14.0, 0.3, 0.0], t) if (k % 5) == 0 else None,
                  last_normed_thrust=1.0)
        a, b = off.update(**kw), on.update(**kw)
        for sl in (slice(11, 16), slice(16, 21)):
            fed_off, fed_on = np.any(a[sl] != 0.0), np.any(b[sl] != 0.0)
            assert not (fed_on and not fed_off), f"armed FED a slot the default MASKED at tick {k}"
            masked_extra += int(fed_off and not fed_on)
    assert masked_extra > 0, "the geometric mask never fired over a full run-in -- check calibration"


# =================================================================================================
# 5. CALIBRATION ON THE REAL CORPUS -- confirmed passes, real attitudes, real detector decisions
# =================================================================================================
_RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "runs")


def _corpus_ticks(want_ticks=25_000, max_runs=120):
    """(range, pose_seen, fed, det) for ticks on CONFIRMED approaches of real recorded flights.

    Walks the corpus NEWEST-FIRST and stops once it has enough ticks: the oldest runs predate the
    current log schema (no ``obs`` row) and would silently contribute nothing, and the newest are
    the ones this knob would actually fly beside.

    FRAMES, both gotchas (get either wrong and every conclusion inverts):
      * logged ``rel_flu`` is TRUE body FLU UNFLIPPED with the aim offset BAKED IN -> de-inject it
        as ``rel_true = rel_logged + [0, +aim_lat, -aim_vert]``;
      * ``obs[3]``/``obs[4]`` are VIRTUAL-FLIPPED leveled roll/pitch -> true = NEGATED.
    Yaw is unrecoverable from the log and is not needed: the test is yaw-free by construction.
    Restricted to CONFIRMED passes (gate_index g -> g+1) so a log that ended early cannot pose as
    a gate the drone never reached."""
    out = []
    Rcb = frames.R_camera_from_body()
    for p in sorted(glob.glob(os.path.join(_RUNS, "*", "ego_obs.jsonl")), reverse=True)[:max_runs]:
        if len(out) >= want_ticks:
            break
        try:
            recs = [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]
        except Exception:
            continue
        adv = {}
        for i in range(1, len(recs)):
            g0, g1 = recs[i - 1].get("gate_index"), recs[i].get("gate_index")
            if g0 is not None and g1 == g0 + 1:
                adv[g0] = recs[i - 1]["sim_time_ns"]
        for g, ta in adv.items():
            for r in recs:
                if r.get("gate_index") != g or r["sim_time_ns"] > ta:
                    continue
                o, rel = r.get("obs"), r.get("rel_flu")
                if not o or not rel or len(o) < 16:
                    continue
                v = np.array(rel, dtype=float)
                a = r.get("aim_off")
                if a is not None:
                    v = v + np.array([0.0, float(a[0]), -float(a[1])])      # de-inject the dodge
                R = _R_zup(roll=-float(o[3]), pitch=-float(o[4]))           # un-flip the attitude
                det, _ = gate_detectable_geometric(v, R, R_cam_from_body=Rcb)
                out.append((float(np.linalg.norm(v)), bool(r.get("pose_seen")),
                            bool(np.any(np.array(o[11:16], dtype=float) != 0.0)), det))
    return out


@pytest.fixture(scope="module")
def corpus():
    if not os.path.isdir(_RUNS):
        pytest.skip("no recorded corpus in data/runs")
    t = _corpus_ticks()
    if len(t) < 2000:
        pytest.skip(f"corpus too small to calibrate against ({len(t)} ticks)")
    return t


def test_corpus_the_model_agrees_with_the_real_detector_outside_the_threshold_band(corpus):
    """THE FALSIFICATION TEST. If the detector produced a fix on a tick, the gate demonstrably WAS
    in frame -- so a tick where the model says "not detectable" is a tick where arming this knob
    would MASK A REAL MEASUREMENT. Beyond 3 m that must essentially never happen; measured over the
    full 564-flight corpus it is 0.0-0.2% in every bin from 3 to 23 m.

    Disagreement INSIDE ~3 m is expected and is not model error -- it is the deploy/training gap
    itself. Training masks 100% of ticks below 1.0 m and 99.3% below 1.5 m with its 50% crossing at
    1.83 m; the median disagreeing tick in the full corpus sits at 1.82 m on 3 of 8 corners."""
    far = [(seen, det) for rng, seen, _fed, det in corpus if rng >= 3.0 and seen]
    assert len(far) > 500, f"too few far fresh-fix ticks to judge ({len(far)})"
    wrong = sum(1 for _s, det in far if not det)
    rate = wrong / len(far)
    assert rate < 0.02, (
        f"the model contradicts the real detector on {rate:.1%} of fresh fixes beyond 3 m "
        f"({wrong}/{len(far)}) -- it would be masking ticks where vision was demonstrably working")


def test_corpus_clears_the_blind_band_training_always_masks(corpus):
    """THE POINT OF THE KNOB, on real data. Inside 1.0 m training feeds ZEROS on 100% of ticks; the
    wire feeds a filled coasted lever. Armed, that must go to zero -- and the OFF path must still
    show the defect, otherwise this corpus slice cannot demonstrate anything."""
    blind = [(fed, det) for rng, _s, fed, det in corpus if rng < 1.0]
    assert len(blind) > 20, f"too few blind-band ticks in the sampled corpus ({len(blind)})"
    filled_now = sum(1 for fed, _d in blind if fed)
    filled_armed = sum(1 for fed, det in blind if fed and det)
    assert filled_now > 0, "the OFF path shows no filled blind-band ticks -- the defect is missing"
    assert filled_armed == 0, (
        f"{filled_armed} of {filled_now} filled blind-band ticks survive the geometric mask; "
        "training feeds zeros on 100% of them")


def test_geometry_reads_the_honest_lever_not_the_aim_offset():
    """The gate-4/5 obstacle dodge shifts the perceived gate +3 m UP. If the visibility geometry saw
    that shifted lever it would call the gate out of frame and mask the slot exactly while the dodge
    is armed -- one knob silently disabling the other. Pin that the two are independent: at a range
    where the dodge is ACTIVE the armed builder must still feed the (offset) lever."""
    cfg = dict(virtual_flip=False, det_geometric=True, aim_offsets={0: (0.0, 3.0)},
               aim_release_m=12.0)
    b = EgoObsBuilder(EgoObsBuilderConfig(**cfg))
    t = 1_000_000_000
    obs = None
    for r in (20.0, 18.0, 16.0):
        t += 33_000_000
        obs = b.update(sim_time_ns=t, gate_index=0, R_frd2ned=np.eye(3), vel_ned=np.zeros(3),
                       gyro_frd=np.zeros(3), pose=_pose([r, 0.0, 0.0], t), last_normed_thrust=0.0)
    assert b.last_diag["aim_off"] == [0.0, 3.0], "the dodge should be armed at 16 m"
    assert b.last_diag["det_geom"] is True, "the honest lever at 16 m is plainly detectable"
    assert np.any(obs[11:16] != 0.0), "the offset lever was masked while the dodge was armed"
    assert abs(float(obs[13]) - 3.0) < 1e-9, "the +3 m UP dodge is missing from the fed lever"
