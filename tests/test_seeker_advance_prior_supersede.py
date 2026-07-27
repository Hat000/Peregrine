"""Unit tests for Patch-2 WP2: gate-advance handling (reset + promote), the arrival-prior acquisition
veto, the nearer-supersede recovery, and gyro-fed track prediction (2026-07-20).

Driven the same way as ``test_next_gate_lever.py`` -- monkeypatch ``_valid_poses`` to inject hand-built
poses, so only the tracker / advance / prior / supersede / propagation logic is exercised (no detector,
PnP, or image projection).

DATA-DERIVED CALIBRATION NOTE (flight 20260720_040510, the wrong-lock this patch targets): the wire
premise the patch was scoped against ("prior [0,-1] DOWN vetoes a +14deg-UP candidate") did NOT survive
re-derivation from the flight logs:
  * the FLOWN coarse map (editor `configs/vq2_coarse_map.json`) arrival row for that leg is [0,0]
    (straight+level), NOT [0,-1]; a [0,0] prior's generous band vetoes nothing there;
  * the wrong candidate's NAIVE camera "+12deg up" is a 36deg nose-DIVE artifact -- correctly GRAVITY-
    LEVELED (the frame the coarse-sector buckets live in) it is -8deg DOWN, i.e. CONSISTENT with a DOWN
    prior, so a physically-correct vertical veto would NOT fire on it either.
So for THAT flight the operative fix is the WP2c SUPERSEDE (the true bigger/nearer gate appears later),
not the prior veto. The prior veto is built physically-correct here (leveled, matching the sector frame)
and bites the NONZERO arrival rows (gate0->1 [-1,1], gate2->3 [1,-1], ...). Both behaviours are pinned
below; ``test_advance_wrong_lock_040510`` exercises the VETO MECHANISM (a genuinely leveled-up candidate
vs a DOWN prior) and ``test_040510_flown_map_supersede_recovers`` the real-flight supersede path.
"""
import numpy as np
import pytest

from racer.contracts import GatePose
from racer.gate_seeker import (GateSeeker, GateSeekerConfig,
                               _leveled_from_body as seek_leveled_from_body,
                               _FLIP_FRD_FLU as seek_flip)
from racer.ego_obs import leveled_from_body as ego_leveled_from_body, _FLIP_FRD_FLU as ego_flip


def _pose(x: float, y: float, z: float, *, fid: int = 1, t_ns: int = 0) -> GatePose:
    """A GatePose at camera-frame offset (x,y,z) m. range_m = |t|, bearing = (atan2(x,z), atan2(y,z))."""
    return GatePose(frame_id=fid, sim_time_ns=t_ns, R_cam_gate=np.eye(3),
                    t_cam_gate=np.array([x, y, z], dtype=np.float64), reproj_error_px=1.0)


class _Frame:
    def __init__(self, fid: int):
        self.frame_id = fid
        self.image_bgr = np.zeros((4, 4, 3), dtype=np.uint8)


def _seeker(**cfg) -> GateSeeker:
    s = GateSeeker(config=GateSeekerConfig(**cfg))
    s.detector = object()
    return s


def _feed(s, monkeypatch, poses, fid):
    monkeypatch.setattr(s, "_valid_poses", lambda frame: poses)
    return s.detect_gate_lever(_Frame(fid), level_rp=(0.0, 0.0))


# ---------------------------------------------------------------------------
# Leveling-frame pin: the seeker's leveling MUST equal the ego_obs sector frame (else the prior veto is
# compared in the wrong frame -- a wrong sign = a false veto = a blind drone).
# ---------------------------------------------------------------------------
def test_leveling_matches_ego_obs_sector_frame():
    assert np.array_equal(seek_flip, ego_flip)
    rng = np.random.default_rng(0)
    for _ in range(200):
        v = rng.normal(size=3)
        roll, pitch = rng.uniform(-1, 1), rng.uniform(-1, 1)
        assert np.allclose(seek_leveled_from_body(v, roll, pitch),
                           ego_leveled_from_body(v, roll, pitch))


def test_leveling_undoes_nose_dive_pitch_artifact():
    # The 040510 k140 candidate: TRUE body FLU [f,l,u] = [13.41,-9.91,3.65] at roll=-20.3, pitch=+36.2
    # deg (a nose dive). NAIVE body-frame elevation reads +12deg UP; correctly leveled it is ~-8deg DOWN.
    s = _seeker()
    rel_flu = np.array([13.41, -9.91, 3.65])
    rel_frd = seek_flip * rel_flu                       # -> body FRD [f,r,d]
    a_lev, e_lev = s._leveled_frd(rel_frd, np.radians(-20.3), np.radians(36.2))
    assert np.degrees(a_lev) == pytest.approx(-28.4, abs=1.0)   # RIGHT
    assert np.degrees(e_lev) == pytest.approx(-8.0, abs=1.0)    # DOWN (not the naive +12 UP)
    # naive (no leveling) would have read UP -- the trap the patch must not fall into:
    naive_el = np.degrees(np.arctan2(rel_flu[2], np.hypot(rel_flu[0], rel_flu[1])))
    assert naive_el > 10.0


# ---------------------------------------------------------------------------
# WP2b: arrival-prior cone veto, per bucket value.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bucket,x,vetoes", [
    (+1.0, +0.5, False),   # expect +, candidate + -> OK
    (+1.0, -0.5, True),    # expect +, candidate far - -> WRONG side
    (-1.0, -0.5, False),   # expect -, candidate - -> OK
    (-1.0, +0.5, True),    # expect -, candidate far + -> WRONG side
    (0.0, +0.3, False),    # zero bucket, within generous band -> OK
    (0.0, +0.7, True),     # zero bucket, beyond generous band -> veto
    (0.0, -0.7, True),     # zero bucket, both sides
])
def test_axis_veto_per_bucket(bucket, x, vetoes):
    assert GateSeeker._axis_veto(bucket, x, 0.26, 0.50) is vetoes


def test_prior_veto_disabled_by_default_never_vetoes():
    """WIRE EVIDENCE (2026-07-21 batch): the veto fired 109x and only 3% were followed by ANY lock within
    1 s -- it produced blindness, not correction (dropout 23.4%->31.7%, gates 3.33->1.58). The arrival
    bucket describes the geometry AT THE PASS in the incoming leg's frame; a tick later the drone has
    climbed/turned through the gate and the TRUE gate can read >15 deg off the stale bucket. Default OFF."""
    s = _seeker()                                      # default config: acquire_prior_enabled False
    s._acquire_prior = (0.0, -1.0)                     # a prior IS latched...
    assert s._prior_vetoes(0.0, +0.5) is False         # ...but a strongly-UP candidate is NOT vetoed
    assert s._prior_vetoes(+0.9, 0.0) is False         # nor a hard-left one


def test_prior_vetoes_uses_both_axes():
    s = _seeker(acquire_prior_enabled=True)            # OFF by default (wire evidence) -- opt in to test
    s._acquire_prior = (0.0, -1.0)                     # straight + DOWN
    assert s._prior_vetoes(0.0, +0.5) is True          # a strongly-UP candidate -> vetoed by vert
    assert s._prior_vetoes(0.0, -0.3) is False          # a DOWN candidate -> consistent
    assert s._prior_vetoes(+0.9, -0.3) is True          # a far-LEFT candidate -> vetoed by the 0-horiz band


def test_prior_veto_inert_without_prior_or_attitude(monkeypatch):
    # No prior latched, or no level_rp -> the acquisition is byte-identical to Patch-1 (far gate locks).
    s = _seeker()
    far_up = _pose(0.0, -8.0, 15.0)                    # ~17 m, strongly leveled-UP
    monkeypatch.setattr(s, "_valid_poses", lambda frame: [far_up])
    assert s.detect_gate_lever(_Frame(1)) is far_up    # no level_rp -> no veto
    s2 = _seeker()
    s2._acquire_prior = (0.0, -1.0)                    # prior set but level_rp omitted -> still inert
    monkeypatch.setattr(s2, "_valid_poses", lambda frame: [far_up])
    assert s2.detect_gate_lever(_Frame(1)) is far_up


# ---------------------------------------------------------------------------
# WP2a: advance handling -- reset + promote.
# ---------------------------------------------------------------------------
def test_advance_reset_kills_stale_track(monkeypatch):
    # slot0 locked on the just-passed gate; the advance must DROP it (no coasting the old gate through).
    s = _seeker()
    near = _pose(0.0, 0.0, 3.0)
    _feed(s, monkeypatch, [near], 1)
    assert s._track_range_m is not None                 # locked ~3 m
    mark = s.on_gate_advance(prior_dir=(0.0, -1.0), level_rp=(0.0, 0.0))   # slot1 dead -> cold reset
    assert mark["action"] == "advance_reset"
    assert s._track_range_m is None and s._track_bearing is None
    assert s._track_ever_locked is False                # cold start armed (keep the acquire fallback)
    assert s._acquire_prior == (0.0, -1.0)              # arrival prior latched


def test_advance_promote_seeds_slot0_from_live_slot1():
    s = _seeker(advance_promote_enabled=True)           # OFF by default (wire evidence) -- opt in to test it
    s._next_track_range_m = 12.0                        # a LIVE next-gate track within [3, 22]
    s._next_track_bearing = np.array([0.1, 0.0])
    mark = s.on_gate_advance(prior_dir=None, level_rp=(0.0, 0.0))
    assert mark["action"] == "advance_promote"
    assert mark["promoted_range_m"] == 12.0
    assert s._track_range_m == 12.0                     # slot0 seeded from slot1
    assert np.allclose(s._track_bearing, [0.1, 0.0])
    assert s._track_ever_locked is True                 # a promote IS a lock (re-acquire discipline armed)
    assert s._next_track_range_m is None                # slot1 reset to cold


def test_advance_promote_disabled_by_default_is_a_cold_reset():
    """WIRE EVIDENCE (2026-07-20 patch-2 batch): slot1 is index-blind and was locked on the FAR gate at
    2/5 gi=1->2 advances, so a promote installed a confident WRONG lock (17.5/19.6 m vs the true 7-8 m).
    Default OFF => a live, in-range, prior-clean slot1 is still NOT promoted; the new slot0 re-acquires
    cold (and goes honestly dark first, which is what buys the sector-commit turn)."""
    s = _seeker()                                       # default config: advance_promote_enabled False
    s._next_track_range_m = 12.0                        # live + in-range + no prior => would promote if ON
    s._next_track_bearing = np.array([0.1, 0.0])
    mark = s.on_gate_advance(prior_dir=None, level_rp=(0.0, 0.0))
    assert mark["action"] == "advance_reset"
    assert mark["promoted_range_m"] is None
    assert s._track_range_m is None and s._track_bearing is None
    assert s._track_ever_locked is False                # cold => the acquisition fallback is armed
    assert s._next_track_range_m is None                # slot1 reset either way


def test_advance_promote_sanity_fail_falls_back_to_cold():
    s = _seeker(advance_promote_enabled=True)
    s._next_track_range_m = 28.0                        # beyond max_acquire_range_m (22) -> not trustworthy
    s._next_track_bearing = np.array([0.1, 0.0])
    mark = s.on_gate_advance(prior_dir=None, level_rp=(0.0, 0.0))
    assert mark["action"] == "advance_reset"            # sanity fail -> plain cold reset
    assert s._track_range_m is None
    assert s._track_ever_locked is False


def test_advance_promote_vetoed_by_prior_cone_falls_back_to_cold():
    # slot1 is in range but its leveled bearing contradicts the arrival prior -> do NOT promote a wrong gate.
    s = _seeker()
    s._next_track_range_m = 12.0
    s._next_track_bearing = np.array([0.0, -0.6])       # strongly UP in camera -> leveled UP
    mark = s.on_gate_advance(prior_dir=(0.0, -1.0), level_rp=(0.0, 0.0))   # prior DOWN
    assert mark["action"] == "advance_reset"
    assert s._track_range_m is None


def test_advance_wrong_lock_040510(monkeypatch):
    """The named regression -- the WP2a+WP2b MECHANISM against a nonzero DOWN arrival prior.

    (See the module docstring: the REAL 040510 flight flew a [0,0] arrival row and a leveled-DOWN wrong
    candidate, so its operative fix is the supersede test below; here we validate the veto chain with a
    genuinely leveled-UP candidate vs a DOWN prior, which is what the veto is FOR.)

    The veto ships DEFAULT OFF (2026-07-21 wire evidence -- see acquire_prior_enabled); this test opts
    in so the MECHANISM stays pinned for the day a frame-corrected prior re-enables it."""
    s = _seeker(acquire_prior_enabled=True)
    # 1) slot0 locked ~3 m on the just-passed gate.
    _feed(s, monkeypatch, [_pose(0.0, 0.0, 3.0)], 1)
    assert s._track_range_m is not None
    # 2) advance with slot1 dead + a DOWN arrival prior -> cold reset (no stale continuity).
    s.on_gate_advance(prior_dir=(0.0, -1.0), level_rp=(0.0, 0.0))
    # 3) next frame: the ONLY candidate is a FAR (~17 m) gate that levels UP -> prior_reject, stay dark.
    far_up = _pose(0.0, -8.0, 15.0)                    # range ~17, strongly leveled-UP
    out = _feed(s, monkeypatch, [far_up], 2)
    assert out is None                                  # NOT judged as continuity -> ACQUISITION reject
    assert s.last_decision(0)["reason"] == "prior_reject"
    assert s._track_range_m is None                     # still dark (coast on the ego hold + sector)
    # 4) later frame: the true nearer gate appears (~9 m, levels DOWN) -> compliant -> locks.
    near_down = _pose(0.0, 5.0, 8.0)                    # range ~9.4, leveled DOWN + straight
    out2 = _feed(s, monkeypatch, [near_down], 3)
    assert out2 is near_down
    assert s.last_decision(0)["reason"] is None


# ---------------------------------------------------------------------------
# WP2c: nearer-supersede (the pilot's size-precedence).
# ---------------------------------------------------------------------------
def test_supersede_switches_after_debounce(monkeypatch):
    s = _seeker(supersede_min_frames=3, supersede_range_factor=0.65)
    far = _pose(0.0, 0.0, 17.0)                         # lock the far gate first
    assert _feed(s, monkeypatch, [far], 1) is far
    near = _pose(2.0, 0.0, 8.0)                         # a DIFFERENT gate, 8 m < 0.65*17 = 11 m
    # frames 2,3: super-candidate present -> streak builds, continuity still returns the far gate.
    assert _feed(s, monkeypatch, [far, near], 2) is far
    assert s._supersede_streak == 1
    assert _feed(s, monkeypatch, [far, near], 3) is far
    assert s._supersede_streak == 2
    # frame 4: streak hits 3 -> SWITCH to the nearer gate.
    out = _feed(s, monkeypatch, [far, near], 4)
    assert out is near
    assert s.last_decision(0)["reason"] == "supersede_nearer"
    assert round(s._track_range_m) == 8                 # track re-seeded onto the nearer gate


def test_supersede_flicker_does_not_switch(monkeypatch):
    s = _seeker(supersede_min_frames=3, supersede_range_factor=0.65)
    far = _pose(0.0, 0.0, 17.0)
    _feed(s, monkeypatch, [far], 1)
    near = _pose(2.0, 0.0, 8.0)
    _feed(s, monkeypatch, [far, near], 2)              # streak 1
    _feed(s, monkeypatch, [far], 3)                    # near gone -> streak resets
    assert s._supersede_streak == 0
    out = _feed(s, monkeypatch, [far, near], 4)        # streak 1 again -> below threshold
    assert out is far                                   # no switch
    assert s._supersede_streak == 1


def test_supersede_respects_prior_cone(monkeypatch):
    # A nearer candidate that VIOLATES the arrival prior must NOT supersede (it is a wrong gate).
    s = _seeker(supersede_min_frames=2, supersede_range_factor=0.65, acquire_prior_enabled=True)
    s._acquire_prior = (0.0, -1.0)                      # prior DOWN
    far = _pose(0.0, 3.0, 17.0)                         # lock a far, roughly-level gate
    _feed(s, monkeypatch, [far], 1)
    near_up = _pose(0.0, -6.0, 8.0)                     # nearer but leveled UP -> prior-incompatible
    _feed(s, monkeypatch, [far, near_up], 2)
    out = _feed(s, monkeypatch, [far, near_up], 3)
    assert out is far                                   # never superseded by the prior-violating gate
    assert s._supersede_streak == 0


def test_040510_flown_map_supersede_recovers(monkeypatch):
    """The REAL 040510 recovery path: with the FLOWN [0,0] arrival row the prior does NOT veto the far
    wrong lock (correct -- a [0,0] bucket can't discriminate), so the drone cold-locks the far gate; the
    SUPERSEDE then switches to the true bigger/nearer gate once it appears for a few frames."""
    s = _seeker(supersede_min_frames=3, supersede_range_factor=0.65)
    s.on_gate_advance(prior_dir=(0.0, 0.0), level_rp=(0.0, 0.0))   # editor row1 = [0,0]
    far = _pose(0.0, 0.0, 17.0)
    assert _feed(s, monkeypatch, [far], 1) is far      # [0,0] prior -> no veto -> far gate locks (cold)
    assert s.last_decision(0)["reason"] is None
    near = _pose(2.0, 0.0, 8.0)                         # the true nearer gate appears
    _feed(s, monkeypatch, [far, near], 2)
    _feed(s, monkeypatch, [far, near], 3)
    assert _feed(s, monkeypatch, [far, near], 4) is near   # supersede recovers onto it
    assert s.last_decision(0)["reason"] == "supersede_nearer"


# ---------------------------------------------------------------------------
# WP2d: gyro-fed track prediction.
# ---------------------------------------------------------------------------
def test_gyro_propagation_moves_prediction_and_rescues_continuity(monkeypatch):
    # Lock a gate centered (az=el=0). A body yaw over the gap shifts its camera bearing beyond the static
    # continuity gate; propagation moves the PREDICTION so the shifted detection is still accepted.
    s = _seeker()
    _feed(s, monkeypatch, [_pose(0.0, 0.0, 10.0)], 1)
    assert np.allclose(s._track_bearing, [0.0, 0.0], atol=1e-6)
    s.propagate(np.array([0.0, 0.0, 2.0]), 0.27)       # pure yaw for 0.27 s
    paz, pel = float(s._track_bearing[0]), float(s._track_bearing[1])
    assert abs(paz) > s.config.track_max_bearing_jump_rad   # a STATIC predictor would reject a hit here
    # a detection AT the propagated bearing (the same gate, now rotated in frame) -> accepted.
    moved = _pose(np.tan(paz) * 10.0, np.tan(pel) * 10.0, 10.0)
    out = _feed(s, monkeypatch, [moved], 2)
    assert out is moved
    assert s.last_decision(0)["reason"] is None


def test_static_prediction_rejects_the_same_shifted_detection(monkeypatch):
    # Same lock + same shifted detection, but WITHOUT propagation -> the static gate rejects it (coast).
    s = _seeker()
    _feed(s, monkeypatch, [_pose(0.0, 0.0, 10.0)], 1)
    # reconstruct the shift the propagation would have produced, but do NOT call propagate.
    s2 = _seeker()
    _feed(s2, monkeypatch, [_pose(0.0, 0.0, 10.0)], 1)
    s2.propagate(np.array([0.0, 0.0, 2.0]), 0.27)
    paz, pel = float(s2._track_bearing[0]), float(s2._track_bearing[1])
    moved = _pose(np.tan(paz) * 10.0, np.tan(pel) * 10.0, 10.0)
    out = _feed(s, monkeypatch, [moved], 2)            # s never propagated
    assert out is None
    assert s.last_decision(0)["reason"] == "continuity_reject"


def test_gyro_propagation_still_rejects_a_genuinely_inconsistent_detection(monkeypatch):
    # Propagation must not turn the continuity gate into a pass-through: a wildly-off detection still rejects.
    s = _seeker()
    _feed(s, monkeypatch, [_pose(0.0, 0.0, 10.0)], 1)
    s.propagate(np.array([0.0, 0.0, 2.0]), 0.27)       # small predicted shift
    wild = _pose(0.0, 0.0, 40.0)                        # 40 m vs ~10 m track -> range jump, inconsistent
    out = _feed(s, monkeypatch, [wild], 2)
    assert out is None
    assert s.last_decision(0)["reason"] == "continuity_reject"


def test_propagate_noop_when_disabled_or_no_track():
    s = _seeker(track_gyro_propagate=False)
    s._track_bearing = np.array([0.1, 0.2])
    s.propagate(np.array([0.0, 0.0, 2.0]), 0.1)
    assert np.allclose(s._track_bearing, [0.1, 0.2])   # disabled -> untouched
    s2 = _seeker()                                     # no track -> no-op, no error
    s2.propagate(np.array([0.0, 0.0, 2.0]), 0.1)
    assert s2._track_bearing is None


# ---------------------------------------------------------------------------
# RANGE PROPAGATION (2026-07-25): r -= (v_body . u_hat)*dt along the line of sight, floored.
# ---------------------------------------------------------------------------
def _armed(**cfg) -> GateSeeker:
    """A seeker with both tracks live, dead ahead in the CAMERA frame (bearing 0,0)."""
    s = _seeker(track_propagate_range=True, **cfg)
    s._track_range_m, s._track_bearing = 10.0, np.array([0.0, 0.0])
    s._next_track_range_m, s._next_track_bearing = 25.0, np.array([0.0, 0.0])
    return s


def test_propagate_range_default_is_off():
    """DEFAULT-OFF: the flag defaults False and the range stays on its EMA even when a velocity is
    supplied -- byte-identical to the bearing-only propagation."""
    assert GateSeekerConfig().track_propagate_range is False
    s = _seeker()
    s._track_range_m, s._track_bearing = 10.0, np.array([0.0, 0.0])
    s._next_track_range_m, s._next_track_bearing = 25.0, np.array([0.0, 0.0])
    s.propagate(np.zeros(3), 0.1, vel_frd=np.array([8.0, 0.0, 0.0]))
    assert s._track_range_m == 10.0 and s._next_track_range_m == 25.0


def test_propagate_range_needs_a_velocity():
    """vel_frd=None (the old 2-arg call, and the ego path when the KF has no velocity yet) leaves
    the range untouched even with the flag ON."""
    s = _armed()
    s.propagate(np.zeros(3), 0.1)
    assert s._track_range_m == 10.0 and s._next_track_range_m == 25.0


def test_propagate_range_closes_at_the_line_of_sight_rate_on_both_tracks():
    """Flying straight at a gate dead ahead in the camera frame: the range shrinks by the full
    speed*dt, on the ACTIVE and the NEXT track alike. The bearing is camera-optical, so 'dead
    ahead' means along the camera axis -- the velocity is pushed through R_camera_from_body()."""
    from racer.frames import R_camera_from_body
    s = _armed()
    v_cam_forward = R_camera_from_body().T @ np.array([0.0, 0.0, 8.0])   # 8 m/s along the cam axis
    s.propagate(np.zeros(3), 0.1, vel_frd=v_cam_forward)
    assert s._track_range_m == pytest.approx(10.0 - 0.8, abs=1e-9)
    assert s._next_track_range_m == pytest.approx(25.0 - 0.8, abs=1e-9)


def test_propagate_range_projects_off_axis_motion():
    """Only the LINE-OF-SIGHT component counts: motion perpendicular to the bearing does not
    change the range, and an oblique bearing closes at speed*cos(angle), not speed."""
    from racer.frames import R_camera_from_body
    R = R_camera_from_body()
    perp = R.T @ np.array([8.0, 0.0, 0.0])            # camera +x (right), normal to a (0,0) bearing
    s = _armed()
    s.propagate(np.zeros(3), 0.1, vel_frd=perp)
    assert s._track_range_m == pytest.approx(10.0, abs=1e-9)

    az = 0.4                                          # bearing 0.4 rad off the camera axis
    s2 = _armed()
    s2._track_bearing = np.array([az, 0.0])
    s2.propagate(np.zeros(3), 0.1, vel_frd=R.T @ np.array([0.0, 0.0, 8.0]))
    assert s2._track_range_m == pytest.approx(10.0 - 0.8 * np.cos(az), abs=1e-9)


def test_propagate_range_is_floored_and_never_flips_sign():
    """An over-propagated (or just-flown-through) track degrades to 'very close', never to 0 or a
    negative range -- the continuity gate, pass-drop and slot1 promote all compare against it."""
    from racer.frames import R_camera_from_body
    s = _armed()
    s._track_range_m = 0.5
    s.propagate(np.zeros(3), 1.0, vel_frd=R_camera_from_body().T @ np.array([0.0, 0.0, 20.0]))
    assert s._track_range_m == pytest.approx(0.3, abs=1e-12)


def test_propagate_range_opens_when_receding():
    """Sign check: flying AWAY grows the range (the just-passed gate receding behind the drone)."""
    from racer.frames import R_camera_from_body
    s = _armed()
    s.propagate(np.zeros(3), 0.1, vel_frd=R_camera_from_body().T @ np.array([0.0, 0.0, -5.0]))
    assert s._track_range_m == pytest.approx(10.5, abs=1e-9)


def test_propagate_range_uses_the_rotated_bearing():
    """The projection is taken against the bearing AFTER the gyro rotation, not before: a body yaw
    that swings the gate off the camera axis must reduce the closing rate in the SAME call."""
    from racer.frames import R_camera_from_body
    v = R_camera_from_body().T @ np.array([0.0, 0.0, 8.0])
    dt, w = 0.2, np.array([0.0, 0.0, 3.0])            # a big yaw over the step
    s = _armed()
    s.propagate(w, dt, vel_frd=v)
    moved = float(np.linalg.norm(s._track_bearing))
    assert moved > 0.3                                 # the bearing really did swing off axis
    assert s._track_range_m > 10.0 - 8.0 * dt + 1e-3   # closed by LESS than the full speed*dt


def test_propagate_range_ignores_non_finite_velocity():
    s = _armed()
    s.propagate(np.zeros(3), 0.1, vel_frd=np.array([np.nan, 0.0, 0.0]))
    assert s._track_range_m == 10.0


# ---------------------------------------------------------------------------
# WP2e: decision-log extensions.
# ---------------------------------------------------------------------------
def test_decision_log_carries_prior_and_supersede_streak(monkeypatch):
    s = _seeker(supersede_min_frames=5, supersede_range_factor=0.65)
    s._acquire_prior = (1.0, -1.0)
    far = _pose(0.0, 3.0, 17.0)
    _feed(s, monkeypatch, [far], 1)
    d = s.last_decision(0)
    assert d["prior"] == [1.0, -1.0]                    # WP2e prior field (slot0)
    assert "sup_streak" in d
    # slot1 decision never carries the prior.
    monkeypatch.setattr(s, "_valid_poses", lambda frame: [far, _pose(3.0, 0.0, 25.0)])
    s.detect_gate_lever(_Frame(2), level_rp=(0.0, 0.0))
    s.detect_next_gate_lever(_Frame(2))
    assert s.last_decision(1)["prior"] is None


def test_on_gate_advance_records_marker():
    s = _seeker(advance_promote_enabled=True)
    s._next_track_range_m = 10.0
    s._next_track_bearing = np.array([0.05, 0.0])
    mark = s.on_gate_advance(prior_dir=(-1.0, 1.0), level_rp=(0.0, 0.0))
    assert mark == s.last_advance()
    assert mark["action"] == "advance_promote"
    assert mark["prior"] == (-1.0, 1.0)
