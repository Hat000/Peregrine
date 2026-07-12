"""Unit tests for the tg+1 (slot1) NEXT-gate lever -- the --ego-slot1 SOURCE (2026-07-12).

``GateSeeker.detect_next_gate_lever`` is a SECOND temporal track mirroring the active-gate track,
over the next gate to fly (the nearest quality-gated gate that is NOT the locked active gate). These
tests drive the SELECTION / continuity / coast logic directly by monkeypatching ``_valid_poses`` to
return hand-built poses, so no detector, PnP, or image projection is exercised -- only the tracker.

Contract under test:
  * slot0 (detect_gate_lever) locks the near gate; slot1 (detect_next_gate_lever) locks the one beyond.
  * slot1 NEVER returns the active gate (dedup) -- so the two obs slots never carry the same gate.
  * a range/bearing JUMP is rejected (coast -> None), never a flapper-lock (a wrong gate is worse than
    a coast; the ego-obs builder ego-propagates + staleness-decays the last fix through the None gap).
  * absurdly-far downrange gates are rejected as a tg+1 seed.
  * reset() (fired on gate advance) clears the next-track in lockstep with the active track.
"""
import numpy as np
import pytest

from racer.contracts import GatePose
from racer.gate_seeker import GateSeeker, GateSeekerConfig


def _pose(x: float, y: float, z: float, *, fid: int = 1, t_ns: int = 0) -> GatePose:
    """A GatePose at camera-frame offset (x,y,z) m. range_m = |t|, bearing = (atan2(x,z), atan2(y,z))."""
    return GatePose(frame_id=fid, sim_time_ns=t_ns, R_cam_gate=np.eye(3),
                    t_cam_gate=np.array([x, y, z], dtype=np.float64), reproj_error_px=1.0)


class _Frame:
    """Minimal stand-in for racer Frame: the lever guards only touch frame_id + image_bgr."""
    def __init__(self, fid: int):
        self.frame_id = fid
        self.image_bgr = np.zeros((4, 4, 3), dtype=np.uint8)


def _seeker() -> GateSeeker:
    s = GateSeeker(config=GateSeekerConfig())
    s.detector = object()   # non-None so the lever guards pass; _valid_poses is monkeypatched per test
    return s


def test_next_lever_locks_the_gate_beyond_the_active_one(monkeypatch):
    s = _seeker()
    near = _pose(0.0, 0.0, 10.0)   # active: nearest + centered
    far = _pose(3.0, 0.0, 25.0)    # next: 25 m, off to the right
    monkeypatch.setattr(s, "_valid_poses", lambda frame: [near, far])
    fr = _Frame(1)
    active = s.detect_gate_lever(fr)
    nxt = s.detect_next_gate_lever(fr)
    assert active is near          # slot0 = the near gate
    assert nxt is far              # slot1 = the gate beyond it
    assert nxt is not active       # the two slots never carry the same detection


def test_next_lever_is_none_when_only_the_active_gate_is_visible(monkeypatch):
    s = _seeker()
    near = _pose(0.0, 0.0, 10.0)
    monkeypatch.setattr(s, "_valid_poses", lambda frame: [near])
    fr = _Frame(1)
    s.detect_gate_lever(fr)
    assert s.detect_next_gate_lever(fr) is None   # dedup drops the only gate -> no tg+1 -> soft gap


def test_next_lever_dedup_excludes_the_active_gate_even_if_it_is_nearest(monkeypatch):
    # Two gates; the active track locks the nearer. slot1 must take the OTHER, not re-lock the active.
    s = _seeker()
    near = _pose(0.0, 0.0, 12.0)
    far = _pose(-4.0, 0.0, 20.0)
    monkeypatch.setattr(s, "_valid_poses", lambda frame: [near, far])
    fr = _Frame(1)
    assert s.detect_gate_lever(fr) is near
    assert s.detect_next_gate_lever(fr) is far


def test_next_track_rejects_a_jump_and_coasts(monkeypatch):
    s = _seeker()
    near = _pose(0.0, 0.0, 10.0)
    far = _pose(3.0, 0.0, 25.0)
    monkeypatch.setattr(s, "_valid_poses", lambda frame: [near, far])
    fr1 = _Frame(1)
    assert s.detect_gate_lever(fr1) is near
    assert s.detect_next_gate_lever(fr1) is far     # tick 1: next-track seeds on far (~25 m)
    # tick 2: the far candidate JUMPS to 45 m (a different gate / a PnP-depth flip). The active gate
    # (near) is unchanged, so slot1's only non-active candidate is the 45 m jump -> continuity rejects
    # it (|45-25| > track_max_range_jump_m) -> coast (None), never a flapper-lock.
    jump = _pose(3.0, 0.0, 45.0)
    monkeypatch.setattr(s, "_valid_poses", lambda frame: [near, jump])
    fr2 = _Frame(2)
    s.detect_gate_lever(fr2)
    assert s.detect_next_gate_lever(fr2) is None


def test_next_lever_seed_rejects_an_absurdly_far_downrange_gate(monkeypatch):
    s = _seeker()
    near = _pose(0.0, 0.0, 10.0)
    superfar = _pose(0.0, 0.0, 60.0)   # beyond next_gate_max_range_m (45 m) -> not a valid tg+1 seed
    monkeypatch.setattr(s, "_valid_poses", lambda frame: [near, superfar])
    fr = _Frame(1)
    s.detect_gate_lever(fr)
    assert s.detect_next_gate_lever(fr) is None


def test_reset_clears_the_next_track(monkeypatch):
    s = _seeker()
    near = _pose(0.0, 0.0, 10.0)
    far = _pose(3.0, 0.0, 25.0)
    monkeypatch.setattr(s, "_valid_poses", lambda frame: [near, far])
    fr = _Frame(1)
    s.detect_gate_lever(fr)
    s.detect_next_gate_lever(fr)
    assert s._next_track_range_m is not None        # the next-track locked
    s.reset()
    assert s._next_track_range_m is None
    assert s._next_track_bearing is None
    assert s._next_track_coast_ticks == 0


# ---------------------------------------------------------------------------
# HARD RANGE CAP (2026-07-12 billboard-FP fix): _valid_poses drops any detection beyond
# max_valid_range_m from the candidate pool (uniformly for slot0 + slot1), BEFORE selection.
# ---------------------------------------------------------------------------


def test_valid_poses_hard_range_cap(monkeypatch):
    # A near gate (25 m) survives; a far billboard FP (35 m) is dropped by the 30 m cap -- so it never
    # reaches slot0/slot1 selection. Drives the real _valid_poses filter (detector + PnP monkeypatched).
    s = GateSeeker(config=GateSeekerConfig(max_valid_range_m=30.0))
    s.detector = object()
    near = _pose(0.0, 0.0, 25.0)          # 25 m -> kept
    far = _pose(0.0, 0.0, 35.0)           # 35 m -> dropped (> 30 m)

    class _Obs:
        def __init__(self, pose):
            self.pose = pose
            self.score = 1.0

    monkeypatch.setattr("racer.gate_seeker.detect_cached",
                        lambda det, frame: [_Obs(near), _Obs(far)])
    monkeypatch.setattr("racer.gate_seeker.estimate_gate_pose",
                        lambda obs, compute_covariance=False: obs.pose)
    poses = s._valid_poses(_Frame(1))
    assert len(poses) == 1
    assert round(poses[0].range_m) == 25   # only the near gate survived the cap


def test_valid_poses_no_cap_by_default(monkeypatch):
    # Default max_valid_range_m == inf: a far detection is NOT dropped (byte-identical classical path).
    s = GateSeeker(config=GateSeekerConfig())          # default: no cap
    s.detector = object()
    far = _pose(0.0, 0.0, 40.0)

    class _Obs:
        def __init__(self, pose):
            self.pose = pose
            self.score = 1.0

    monkeypatch.setattr("racer.gate_seeker.detect_cached", lambda det, frame: [_Obs(far)])
    monkeypatch.setattr("racer.gate_seeker.estimate_gate_pose",
                        lambda obs, compute_covariance=False: obs.pose)
    assert len(s._valid_poses(_Frame(1))) == 1         # 40 m gate kept (no cap)
