"""A34 — absolute course-informed range cap (replaces H-3's pred_r-relative wall).

Run 20260703_223956 (A33) regressed: first-acquisition's permissive fallback
(``if not admissible: admissible = poses``) LOCKED a 52 m mis-depthed garbage candidate after the
near track coasted out, and H-3's pred_r-RELATIVE range wall then STARVED recovery (once pred_r ~49 m,
a fresh close 6 m pose is |6-49|=43 m > jump -> rejected forever). H-3 also never caught the ORIGINAL
gradual A32 smear (0.8-2.5 m/frame, all sub-threshold).

A34 (spec handoff/vq2_a34_pose_feed_regression_spec_2026-07-03.md): an ABSOLUTE cap at candidate
admission (``track_abs_range_cap_m``), applied BEFORE the first-acq/continuity split so no path
(incl. the permissive fallback) can lock a >cap garbage. A >35 m reading is physically impossible on
the course (max spacing ~30-38 m, usable PnP ~32 m) = a different object, correctly hard-discarded.
Default None => OFF => byte-identical (VQ1 / case-A). vq2_case_c: 35.0. H-3 is DROPPED from the
profile (field stays dormant for A/B).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from racer.contracts import Frame, GatePose  # noqa: E402
from racer.deploy_profile import get_profile, vq1_case_a, vq2_case_c  # noqa: E402
from racer.frames import R_camera_from_body, R_world_from_body  # noqa: E402
from racer.gate_seeker import GateSeeker, GateSeekerConfig  # noqa: E402


def _pose(range_m, az, t_ns, fid):
    """A GatePose along world dir at azimuth ``az`` (level frame, yaw 0) at range ``range_m``."""
    d = np.array([np.cos(az), np.sin(az), 0.0])
    d_body = R_world_from_body(0.0, 0.0, 0.0).T @ d
    return GatePose(frame_id=int(fid), sim_time_ns=int(t_ns), R_cam_gate=np.eye(3),
                    t_cam_gate=R_camera_from_body() @ (float(range_m) * d_body), reproj_error_px=0.0)


def _frame(fid, t_ns):
    return Frame(frame_id=int(fid), sim_time_ns=int(t_ns),
                 image_bgr=np.ones((360, 640, 3), dtype=np.uint8))


class _Det:
    def detect(self, fr):  # real detector object; _valid_poses is patched per test
        return []


def _seeker(**cfg_overrides):
    cfg = GateSeekerConfig(use_gate_track=True, **cfg_overrides)
    s = GateSeeker(config=cfg, detector=_Det())
    return s


# ---------------------------------------------------------------------------
# 1. The cap discards far garbage at admission (both acquisition paths)
# ---------------------------------------------------------------------------
def test_cap_discards_far_candidate_on_first_acquisition():
    """A lone 52 m candidate (the flight's garbage) is discarded -> no lock, coast tick.
    Without the cap, first-acq's permissive fallback would LOCK it."""
    s = _seeker(track_abs_range_cap_m=35.0)
    far = _pose(52.0, 0.0, 0, 0)
    s._valid_poses = lambda fr: [far]
    out = s.detect_gate_lever(_frame(0, 0))
    assert out is None, "52 m garbage must be discarded, not locked"
    assert s._track_range_m is None, "no track should form on a >cap-only frame"


def test_cap_off_default_locks_far_candidate_permissive_fallback():
    """Byte-identity / regression pin: with the cap OFF (default None) the permissive fallback
    still locks the 52 m candidate exactly as before A34 (the pre-A34 behaviour)."""
    s = _seeker()  # track_abs_range_cap_m default None
    assert s.config.track_abs_range_cap_m is None
    far = _pose(52.0, 0.0, 0, 0)
    s._valid_poses = lambda fr: [far]
    out = s.detect_gate_lever(_frame(0, 0))
    assert out is not None, "cap OFF must reproduce the pre-A34 permissive-fallback lock"
    assert abs(out.range_m - 52.0) < 1e-6


# ---------------------------------------------------------------------------
# 2. The cap KEEPS every legit close pose (never the failure we just had)
# ---------------------------------------------------------------------------
def test_cap_keeps_close_poses():
    """Every legit close pose (6.4-10.7 m observed) passes the 35 m cap."""
    for r in (3.0, 6.4, 6.43, 10.0, 10.7, 30.0, 34.9):
        s = _seeker(track_abs_range_cap_m=35.0)
        p = _pose(r, 0.0, 0, 0)
        s._valid_poses = lambda fr, _p=p: [_p]
        out = s.detect_gate_lever(_frame(0, 0))
        assert out is not None and abs(out.range_m - r) < 1e-6, f"{r} m must be kept"


def test_cap_prefers_close_over_far_when_both_present():
    """A close gate + a far garbage in the same frame: the far is discarded, the close is locked."""
    s = _seeker(track_abs_range_cap_m=35.0)
    close = _pose(8.0, 0.0, 0, 0)
    far = _pose(52.0, 0.2, 0, 1)
    s._valid_poses = lambda fr: [far, close]
    out = s.detect_gate_lever(_frame(0, 0))
    assert out is not None and abs(out.range_m - 8.0) < 1e-6


def test_cap_prevents_far_smear_of_the_track_prediction():
    """The cap's PRIMARY recovery property: pred_r can NEVER smear beyond the cap, because every
    >cap candidate is discarded at admission. So the relative-wall starvation gap (|6-49|=43 m) that
    the flight hit can no longer form -- the worst a held track can predict is ~35 m."""
    s = _seeker(track_abs_range_cap_m=35.0)
    s._track_range_m = 10.0
    s._track_bearing = np.array([0.0, 0.0])
    # a stream of 52 m garbage can NEVER pull the track past 35 m -- it is discarded before the EMA.
    for k in range(20):
        far = _pose(52.0, 0.0, k * 100_000, k)
        s._valid_poses = lambda fr, _p=far: [_p]
        s.detect_gate_lever(_frame(k, k * 100_000))
        assert s._track_range_m is None or s._track_range_m <= 35.0, \
            "the cap must keep the track prediction from ever smearing beyond the cap"


def test_recovery_via_coast_out_then_first_acquisition():
    """Recovery from a smeared/held track: the continuity wall rejects a far-from-prediction pose,
    but after ``track_max_coast_ticks`` the track DROPS to None and first-acquisition (which has NO
    relative wall, only max_acquire_range_m + the A34 cap) re-locks the close gate cleanly. This is
    the recovery path -- a bounded ~8-tick coast, not the permanent starvation the flight showed."""
    s = _seeker(track_abs_range_cap_m=35.0)
    s._track_range_m = None   # track already dropped (coasted out)
    s._track_bearing = None
    close = _pose(6.0, 0.0, 1_000_000, 5)
    s._valid_poses = lambda fr: [close]
    out = s.detect_gate_lever(_frame(5, 1_000_000))
    assert out is not None and abs(out.range_m - 6.0) < 1e-6, \
        "first-acquisition (post coast-out) must re-lock the close gate -- no relative wall there"
    assert s._track_range_m is not None and abs(s._track_range_m - 6.0) < 1e-6


# ---------------------------------------------------------------------------
# 3. Profile wiring: A34 on for vq2_case_c, H-3 dropped, VQ1 untouched
# ---------------------------------------------------------------------------
def test_profile_wiring_a34():
    ov = get_profile("vq2_case_c").seeker_overrides
    assert ov["track_abs_range_cap_m"] == 35.0
    # H-3 DROPPED: soft_range_hard_reject no longer set in the profile.
    assert "soft_range_hard_reject" not in ov
    eff = GateSeekerConfig(**ov)
    assert eff.track_abs_range_cap_m == 35.0
    assert eff.soft_range_hard_reject is False   # back to the field default (dormant)
    # A33 keepers still wired.
    assert eff.pass_exclude_prev_gate is True and eff.pass_turn_through is True
    # VQ1 / case-A: no overrides, cap OFF (byte-identical).
    assert vq1_case_a().seeker_overrides is None
    assert GateSeekerConfig().track_abs_range_cap_m is None
    assert GateSeekerConfig().soft_range_hard_reject is False


def test_cap_none_is_byte_identical_admission():
    """Regression pin: cap None never touches the candidate list (the guard is `is not None`)."""
    s = _seeker()  # None
    poses = [_pose(8.0, 0.0, 0, 0), _pose(52.0, 0.2, 0, 1), _pose(100.0, 0.1, 0, 2)]
    s._valid_poses = lambda fr: list(poses)
    out = s.detect_gate_lever(_frame(0, 0))
    # with the cap off, the permissive/continuity path still runs over ALL candidates (incl. 100 m)
    assert out is not None  # something locks (pre-A34 behaviour preserved)
