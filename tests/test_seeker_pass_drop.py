"""PATCH-3 WP3a/WP3b: local pass detection + candidate logging.

Pins the failure measured on flight 20260722_031742 (the second 9-gate run), gate 8:
the drone crossed the plane with the track at 0.22 m, RACE_STATUS did not report the
advance for another 0.35 s (it is a 4 Hz message), and for 10 consecutive control ticks
every candidate for the NEXT gate was rejected by the continuity gate as a >6 m jump from
the stale close track -- with 1-2 live candidates on offer the whole time.

Helpers mirror tests/test_seeker_advance_prior_supersede.py so both patch suites build
poses the same way (camera-frame offsets; range_m = |t|).
"""
from __future__ import annotations

import numpy as np
import pytest

from racer.contracts import GatePose
from racer.gate_seeker import GateSeeker, GateSeekerConfig


def _pose(x: float, y: float, z: float, *, fid: int = 1, t_ns: int = 0) -> GatePose:
    """A GatePose at camera-frame offset (x,y,z) m."""
    return GatePose(frame_id=fid, sim_time_ns=t_ns, R_cam_gate=np.eye(3),
                    t_cam_gate=np.array([x, y, z], dtype=np.float64), reproj_error_px=1.0)


def _ahead(range_m: float, x: float = 0.0) -> GatePose:
    """A gate straight ahead (camera +z) at ``range_m``, optionally offset laterally."""
    return _pose(x, 0.0, float(range_m))


class _Frame:
    def __init__(self, fid: int):
        self.frame_id = fid
        self.image_bgr = np.zeros((4, 4, 3), dtype=np.uint8)


def _seeker(**cfg) -> GateSeeker:
    s = GateSeeker(config=GateSeekerConfig(**cfg))
    s.detector = object()
    return s


def _feed(s, monkeypatch, poses, fid: int = 1):
    """Drive one control tick through the PUBLIC entry point with a fixed candidate set."""
    monkeypatch.setattr(s, "_valid_poses", lambda frame: list(poses))
    return s.detect_gate_lever(_Frame(fid), level_rp=(0.0, 0.0))


def _hold_close(s: GateSeeker, rng: float) -> None:
    """Put the seeker in the state the wire was in at the moment of the pass: a LIVE, LOCKED
    track at close range, bearing dead ahead."""
    s._track_range_m = float(rng)          # noqa: SLF001 - seeding the state under test
    s._track_bearing = np.zeros(2)         # noqa: SLF001
    s._track_ever_locked = True            # noqa: SLF001
    s._track_coast_ticks = 0               # noqa: SLF001


def test_close_track_with_all_candidates_inconsistent_is_dropped_immediately(monkeypatch):
    """The gate-8 signature: track at 2.0 m, the next gate on offer at 18 m (a 16 m step).

    Patch-2 coasts for track_max_coast_ticks before releasing the dead track; WP3a
    recognises a flown-through gate and drops it on the FIRST such tick.
    """
    s = _seeker(pass_drop_range_m=2.5, track_max_coast_ticks=8)
    _hold_close(s, 2.0)
    _feed(s, monkeypatch, [_ahead(18.0)])
    d = s.last_decision(0)
    assert d["reason"] == "pass_drop", d
    assert s._track_range_m is None, "the dead close track must be released"   # noqa: SLF001
    assert s._track_coast_ticks == 0                                          # noqa: SLF001


def test_pass_drop_leaves_a_cold_acquisition_not_a_reacquire(monkeypatch):
    """After a pass the next lock must be a COLD first-acquisition: the re-acquire hint
    belongs to the gate we just flew through and must not steer the new one."""
    s = _seeker(pass_drop_range_m=2.5)
    _hold_close(s, 1.8)
    _feed(s, monkeypatch, [_ahead(18.0)])
    assert s._track_ever_locked is False                    # noqa: SLF001


def test_far_track_still_coasts_the_continuity_gate_is_not_loosened(monkeypatch):
    """A track at NORMAL range keeps patch-2 behaviour. WP3a is scoped to close tracks, so it
    adds no new false-positive surface to gate selection."""
    s = _seeker(pass_drop_range_m=2.5, track_max_coast_ticks=8)
    _hold_close(s, 12.0)
    _feed(s, monkeypatch, [_ahead(30.0)])
    d = s.last_decision(0)
    assert d["reason"] == "continuity_reject", d
    assert s._track_range_m is not None, "a far track must still coast, not drop"  # noqa: SLF001


def test_disabled_by_zero_reproduces_patch2_exactly(monkeypatch):
    """pass_drop_range_m=0.0 is the OFF switch -- patch-2 behaviour, unchanged."""
    s = _seeker(pass_drop_range_m=0.0, track_max_coast_ticks=8)
    _hold_close(s, 2.0)
    _feed(s, monkeypatch, [_ahead(18.0)])
    assert s.last_decision(0)["reason"] == "continuity_reject"
    assert s._track_range_m is not None                     # noqa: SLF001


def test_decision_log_carries_the_candidate_list(monkeypatch):
    """WP3b: n_cand alone cannot answer 'why this gate and not the closer one in frame?'.
    The log must carry each candidate's range and bearing, and flag the selected one."""
    s = _seeker(pass_drop_range_m=2.5)
    _hold_close(s, 2.0)
    _feed(s, monkeypatch, [_ahead(18.0), _ahead(12.0, x=5.0)])
    d = s.last_decision(0)
    assert d["cands"] is not None and len(d["cands"]) == 2, d
    assert all("b" in c and "sel" in c for c in d["cands"])
    ranges = sorted(round(c["r"]) for c in d["cands"])
    assert ranges == [13, 18], ranges        # 12 m at 5 m lateral -> |t| = 13


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
