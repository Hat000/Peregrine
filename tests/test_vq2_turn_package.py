"""TURN PACKAGE — the gate-1 understeer fix (2026-07-04, run 20260704_173948).

Operator eyes on the R2-1 confirm fly: "we flew into it a bit fast, and understeered... could've
rolled to the right more... the more we overshoot the more the gate perspectival turns into a
parallelogram". Measured: entry ~3.4-3.9 m/s vs ~2.2-2.8 comfortable at the realized lateral;
alat NEVER railed (0% at the 3.0 cap) -- roll was INPUT-crushed: the A31 bearing-gate reference
seeded on a single outlier pose post-pass, so 8 consecutive HONEST frames scored bw 0.01-0.06 and
``az_eff *= bw`` zeroed the roll demand for 0.76 s at peak need (az 27-34 deg).

FIVE LEVERS (four profile values + one flag), each independently revertible:
  a1 forward_accel_mps2 0.65 -> 0.45      (slower approach: commanded dv 1.37 -> ~0.95 m/s)
  a2 fwd_point_gate_az_rad 0.35 -> 0.25   (full forward cut beyond ~14 deg az, was 20)
  c1 image_kaz_mps2_per_rad 8 -> 12       (graze-band alat 0.6-1.5 -> 1.5-2.8; caps unchanged)
  c2 image_lat_slew_mps3 6 -> 9           (anti-snap limit scales with k_az; reversal >= 0.67 s)
  b1 GateSeekerConfig.track_max_loww_ticks (NEW, default None == byte-identical): a SEPARATE
     fresh-frames-only low-weight persistence limit. The shared ``track_max_coast_ticks`` also
     bounds honest no-frame droughts (p90 pose age ~345 ms ~ 7 ticks at ~20 Hz) so it cannot be
     lowered; this limit counts ONLY consecutive fresh w < bearing_w_coast_thresh frames --
     no-pose ticks neither advance nor reset the streak, a healthy frame resets it. vq2: 4
     (drop + reference re-seed on the 5th crushed frame, ~0.3 s, vs the 9th / 0.76 s flown).

The pre-b1 drop timing at default None is ALSO pinned by
``test_a32_robust_estimation.test_soft_weight_persistent_garbage_still_drops_the_track``.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from racer.contracts import Frame, GatePose  # noqa: E402
from racer.deploy_profile import vq1_case_a, vq2_case_c  # noqa: E402
from racer.gate_seeker import GateSeeker, GateSeekerConfig  # noqa: E402

_NS = 1_000_000_000


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
    s._valid_poses = lambda frame: poses
    return s.detect_gate_lever(Frame(frame_id=int(fid), sim_time_ns=int(t_ns),
                                     image_bgr=np.ones((4, 4, 3), dtype=np.uint8)))


def _seed_track(s, t0=0, r=6.0):
    """Lock a healthy track + bearing-gate reference at level attitude."""
    s._append_att_hist(t0, (0.0, 0.0, 0.0))
    prev = _pose_cam(0.0, 0.0, r, t0, fid=0)
    assert _detect(s, [prev], 0, t0) is prev
    return prev


def _wild(k, t_ns):
    """A grossly bearing-inconsistent frame (~10x the allowance -> w < 0.1), sign-alternating."""
    return _pose_cam(0.9 * (1 if k % 2 else -1), 0.3, 6.0, t_ns, fid=k)


def test_default_is_none():
    """The field default (VQ1 / every offline path) leaves the loww limit unset."""
    assert GateSeekerConfig().track_max_loww_ticks is None


def test_default_none_keeps_the_shared_drop_timing():
    """Byte-identity pin: at loww=None the drop still happens on the shared-counter schedule
    (coast=3 -> the 4th wild frame), never earlier -- the loww counter counts but is ignored."""
    s = _soft_seeker(track_max_coast_ticks=3)      # track_max_loww_ticks stays None
    _seed_track(s)
    for k in range(1, 4):                          # wild frames 1..3: counter 1..3, no drop
        t = int(k * 0.05 * _NS)
        s._append_att_hist(t, (0.0, 0.0, 0.0))
        assert _detect(s, [_wild(k, t)], k, t) is not None, k
        assert s._track_loww_ticks == k            # counting, unconsulted
    t = int(4 * 0.05 * _NS)
    s._append_att_hist(t, (0.0, 0.0, 0.0))
    assert _detect(s, [_wild(4, t)], 4, t) is None  # 4th wild: counter 4 > 3 -> drop (old timing)
    assert s._track_range_m is None


def test_loww_limit_drops_before_the_shared_counter():
    """The b1 lever: loww=2 with the shared limit at its 8 default -> the track drops (and the
    bearing-gate reference re-seeds) on the 3rd consecutive crushed frame, not the 9th."""
    s = _soft_seeker(track_max_loww_ticks=2)       # shared track_max_coast_ticks stays 8
    _seed_track(s)
    for k in range(1, 3):                          # wild 1..2: streak 1..2, no drop
        t = int(k * 0.05 * _NS)
        s._append_att_hist(t, (0.0, 0.0, 0.0))
        assert _detect(s, [_wild(k, t)], k, t) is not None, k
        assert s._last_bearing_w < 0.1
    t = int(3 * 0.05 * _NS)
    s._append_att_hist(t, (0.0, 0.0, 0.0))
    assert _detect(s, [_wild(3, t)], 3, t) is None  # 3rd wild: 3 > 2 -> drop
    assert s._track_range_m is None and s._last_none_reason == "continuity_reject"
    assert s._bg_prev_dir_world is None             # the poisoned reference died with the track


def test_no_pose_gaps_neither_advance_nor_reset_the_streak():
    """A bridge/drought tick (no candidate) must not advance the loww streak (that would re-import
    the shared counter's drought sensitivity) NOR reset it (the poisoned-reference crush survives
    brief pose gaps in-flight -- a gap mid-streak must not restart the count)."""
    s = _soft_seeker(track_max_loww_ticks=2)
    _seed_track(s)
    t = int(0.05 * _NS)
    s._append_att_hist(t, (0.0, 0.0, 0.0))
    assert _detect(s, [_wild(1, t)], 1, t) is not None
    assert s._track_loww_ticks == 1
    t = int(0.10 * _NS)
    assert _detect(s, [], 2, t) is None            # no-pose tick: coasts on the track
    assert s._track_loww_ticks == 1                # ...streak neither advanced nor reset
    t = int(0.15 * _NS)
    s._append_att_hist(t, (0.0, 0.0, 0.0))
    assert _detect(s, [_wild(3, t)], 3, t) is not None
    assert s._track_loww_ticks == 2
    t = int(0.20 * _NS)
    assert _detect(s, [], 4, t) is None            # another gap: still no advance/reset
    assert s._track_loww_ticks == 2
    t = int(0.25 * _NS)
    s._append_att_hist(t, (0.0, 0.0, 0.0))
    assert _detect(s, [_wild(5, t)], 5, t) is None  # 3rd crushed FRAME: 3 > 2 -> drop
    assert s._track_range_m is None


def test_healthy_frame_resets_the_streak():
    """One consistent (w >= thresh) frame ends the streak: crush-crush-healthy-crush-crush never
    drops at loww=2; only a third CONSECUTIVE crushed frame does."""
    s = _soft_seeker(track_max_loww_ticks=2)
    prev = _seed_track(s)
    for k, t_s in ((1, 0.05), (2, 0.10)):
        t = int(t_s * _NS)
        s._append_att_hist(t, (0.0, 0.0, 0.0))
        assert _detect(s, [_wild(k, t)], k, t) is not None
    assert s._track_loww_ticks == 2
    t = int(0.15 * _NS)
    s._append_att_hist(t, (0.0, 0.0, 0.0))
    # a healthy frame AT the track prediction (the wilds barely moved the EMA at w<0.1)
    good = _pose_cam(float(s._track_bearing[0]), float(s._track_bearing[1]), 6.0, t, fid=3)
    assert _detect(s, [good], 3, t) is good
    assert s._last_bearing_w >= s.config.bearing_w_coast_thresh
    assert s._track_loww_ticks == 0                # streak reset
    for k, t_s in ((4, 0.20), (5, 0.25)):
        t = int(t_s * _NS)
        s._append_att_hist(t, (0.0, 0.0, 0.0))
        assert _detect(s, [_wild(k, t)], k, t) is not None, k
    assert s._track_loww_ticks == 2                # fresh streak of 2: still alive


def test_vq2_profile_ships_the_turn_package():
    so = vq2_case_c().seeker_overrides
    # TURN ITERATION (2026-07-04, run 20260704_231155): forward slower + more roll authority.
    # Now the alat cap DID rail through the turn (ticks 54/64/152 at the 3.0 cap, sustained
    # az_err 0.4-0.7 rad), so the caps 3.0 -> 4.0 (operator: "roll more") and forward 0.45 -> 0.35
    # (operator: "fly into the first gate slower"). Yaw FROZEN per operator ("yaw is good").
    assert so.get("forward_accel_mps2") == 0.25            # a1 + S1 2026-07-05 (0.35 -> 0.25)
    assert so.get("fwd_point_gate_az_rad") == 0.40         # a2 + climb fix 1 (0.25 -> 0.40)
    assert so.get("image_kaz_mps2_per_rad") == 8.0         # c1 reverted by climb fix 2 (12 -> 8)
    assert so.get("image_lat_slew_mps3") == 9.0            # c2
    assert so.get("track_max_loww_ticks") == 4             # b1
    # turn iteration: caps 3.0 -> 4.0 (the cap now binds -> more roll; lateral-first keeps forward
    # governed by forward_accel_mps2 so it cannot balloon):
    assert so.get("image_lat_cap_mps2") == 4.0
    assert so.get("total_accel_cap_mps2") == 4.0
    # yaw FROZEN (operator: "yaw is good where it is; if we roll more we don't need to yaw as much"):
    assert so.get("pursuit_yaw_slew_rps") == 0.9
    assert so.get("visual_yaw_rate_cap_rps") == 0.9
    assert so.get("pass_arm_range_m") == 5.5               # E2 2026-07-05 (4.5 -> 5.5)
    assert so.get("orbit_guard_rad") == 3.0


def test_vq1_profile_untouched():
    assert vq1_case_a().seeker_overrides is None
