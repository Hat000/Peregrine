"""A15/A17 per-frame vision-COMPUTE cuts (frame-starvation fix, 2026-07-01).

The live VQ2 loop choked to ~10 Hz because the per-frame vision pipeline ran too much work per frame:
YOLO detect ran TWICE (navigator + gate-seeker on the SAME shared detector), and the two heavy CV
backstops (vp_yaw ~67 ms VP RANSAC + Manhattan; floor_height ~37 ms) ran EVERY processed vision tick.
This suite pins the two mechanical fixes:

  1. detect_cached (racer.vision.detector) — computes detect() AT MOST ONCE per frame_id on a shared
     detector; a second consumer of the same frame reuses the cached, byte-identical observations.

  2. NavigatorConfig.vp_yaw_decimate / floor_height_decimate — run each backstop only every N-th
     processed vision tick. DEFAULT = 1 = every tick = byte-identical; N>1 SKIPS the backstop on the
     off-ticks (estimate_heading / estimate_floor_height are not called at all).

Torch-free (duck-typed fake detector + monkeypatched estimators), mirrors test_vision_yaw_wiring.py.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from racer.contracts import DroneState, Frame, Gate, GateObservation  # noqa: E402
from racer.frames import R_camera_from_body  # noqa: E402
from racer.navigator import Navigator, NavigatorConfig  # noqa: E402
from racer.vision.detector import (  # noqa: E402
    GateDetector,
    detect_cached,
    observations_from_results,
)
from racer.vision.gate_pose import project_gate_corners  # noqa: E402

_LEVEL_Q = np.array([1.0, 0.0, 0.0, 0.0])
_HOVER_ACCEL = np.array([0.0, 0.0, -9.80665])
_DT_NS = 5_000_000          # 5 ms = 200 Hz HIGHRES_IMU


# ======================================================================================
# 1. detect_cached — one inference per frame_id, byte-identical observations
# ======================================================================================
class _Attr:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _fake_results(kxy, kconf, scores, bboxes=None):
    return _Attr(
        boxes=_Attr(conf=np.asarray(scores, float), xywh=None if bboxes is None else np.asarray(bboxes, float)),
        keypoints=_Attr(xy=np.asarray(kxy, float), conf=np.asarray(kconf, float)),
    )


class _CountingModel:
    """A YOLO-pose stand-in: ``predict`` returns fixed Results + counts calls."""

    def __init__(self, results):
        self._results = results
        self.calls = 0

    def predict(self, img, **kwargs):
        self.calls += 1
        return self._results


def _frame(frame_id=7, sim_time_ns=100):
    return Frame(frame_id=frame_id, sim_time_ns=sim_time_ns,
                 image_bgr=np.zeros((360, 640, 3), np.uint8), recv_monotonic_ns=0)


def _corners():
    R = np.eye(3)
    T = np.array([0.3, -0.2, 5.0])
    return project_gate_corners(R, T)


def _obs_equal(a: GateObservation, b: GateObservation) -> bool:
    def _arr_eq(x, y):
        return (x is None) == (y is None) and (x is None or np.array_equal(x, y))
    return (
        a.frame_id == b.frame_id
        and a.sim_time_ns == b.sim_time_ns
        and _arr_eq(a.corners_px, b.corners_px)
        and _arr_eq(a.corner_ids, b.corner_ids)
        and _arr_eq(a.corner_confidence, b.corner_confidence)
        and a.score == b.score
        and _arr_eq(a.bbox_xywh, b.bbox_xywh)
        and a.gate_id == b.gate_id
    )


def test_detect_cached_matches_direct_detect_byte_identical():
    """The FIRST detect_cached on a frame returns exactly what a direct detect() would (same
    observations, field-by-field) -- the cache is a transparent memoization, not a re-computation."""
    results = [_fake_results(np.stack([_corners()]), np.ones((1, 4)), [0.9],
                             bboxes=[[100, 90, 50, 40]])]
    # Two separate detectors wrapping identical fake models: one direct, one cached -> identical output.
    direct = GateDetector(_CountingModel(results)).detect(_frame(frame_id=7))
    cached = detect_cached(GateDetector(_CountingModel(results)), _frame(frame_id=7))
    assert len(direct) == len(cached) == 1
    assert all(_obs_equal(x, y) for x, y in zip(direct, cached))


def test_detect_cached_runs_model_once_per_frame_id():
    """Two consumers (navigator + seeker) calling detect_cached on the SAME detector + SAME frame_id
    run predict() ONCE; the second call returns the cached list object. A new frame_id recomputes."""
    model = _CountingModel([_fake_results(np.stack([_corners()]), np.ones((1, 4)), [0.9])])
    det = GateDetector(model)
    f = _frame(frame_id=42)
    a = detect_cached(det, f)          # navigator
    b = detect_cached(det, f)          # seeker, same frame -> cache hit
    assert model.calls == 1, "second consumer must NOT re-run the model on the same frame_id"
    assert a is b, "the cache must hand back the same observations object"
    # A NEW frame_id busts the cache -> exactly one more predict().
    detect_cached(det, _frame(frame_id=43))
    assert model.calls == 2


def test_detect_cached_no_frame_id_falls_back_to_detect():
    """A frame lacking a usable frame_id (None) is never cached -> every call is a real detect (no
    false cache-hit across distinct frames)."""
    model = _CountingModel([_fake_results(np.stack([_corners()]), np.ones((1, 4)), [0.9])])
    det = GateDetector(model)
    f = Frame(frame_id=None, sim_time_ns=0, image_bgr=np.zeros((360, 640, 3), np.uint8))
    detect_cached(det, f)
    detect_cached(det, f)
    assert model.calls == 2   # no caching without a frame_id


# ======================================================================================
# 2. Decimation — default=1 byte-identical; N>1 skips the backstop
# ======================================================================================
def _gate_facing_north(position, gate_id=0, inner=1.5) -> Gate:
    R = np.column_stack([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    return Gate(gate_id=gate_id, position_ned=np.asarray(position, float),
                R_world_gate=R, inner_size_m=inner)


def _project_gate(gate: Gate, drone_pos, R_wb, inner=1.5) -> np.ndarray:
    R_camera_world = (R_wb @ R_camera_from_body().T).T
    t_cam_gate = R_camera_world @ (gate.position_ned - np.asarray(drone_pos, float))
    R_cam_gate = R_camera_world @ gate.R_world_gate
    return project_gate_corners(R_cam_gate, t_cam_gate, inner_size_m=inner)


class _FakeDetector:
    def __init__(self, gate: Gate, drone_pos, inner=1.5):
        self.gate, self.drone_pos, self.inner = gate, np.asarray(drone_pos, float), inner

    def detect(self, frame: Frame):
        corners = _project_gate(self.gate, self.drone_pos, np.eye(3), self.inner)
        return [GateObservation(
            frame_id=frame.frame_id, sim_time_ns=frame.sim_time_ns,
            corners_px=corners, corner_confidence=np.ones(4), score=0.9,
        )]


def _nav_frame(frame_id, sim_time_ns):
    return Frame(frame_id=frame_id, sim_time_ns=int(sim_time_ns),
                 image_bgr=np.zeros((360, 640, 3), np.uint8), recv_monotonic_ns=0)


def _ds(sim_time_ns, *, gyro_body=None, active_gate_index=0):
    return DroneState(
        sim_time_ns=int(sim_time_ns),
        orientation_ned_wxyz=_LEVEL_Q.copy(),
        roll=0.0, pitch=0.0, yaw=0.0,
        angular_rate_body=np.zeros(3),
        accel_body=_HOVER_ACCEL.copy(),
        gyro_body=(np.zeros(3) if gyro_body is None else np.asarray(gyro_body, float)),
        position_ned=None,
        active_gate_index=active_gate_index,
    )


def _make_nav(gate, drone_pos, **flags):
    cfg = NavigatorConfig(
        use_given_position=False, use_given_velocity=False,
        use_vision=True, use_ahrs=True, reconcile_vision_clock=False,
        **flags,
    )
    return Navigator(gates=[gate], detector=_FakeDetector(gate, drone_pos), config=cfg)


def test_decimate_default_one_is_byte_identical():
    """vp_yaw_decimate/floor_height_decimate DEFAULT to 1 (every tick); a nav that explicitly sets
    them to 1 produces bit-identical KF (x, P) + AHRS attitude to one that leaves them at default,
    over a synthetic run with the backstops ON (stubbed estimators)."""
    import racer.navigator as navmod
    from racer.vision.floor_height import FloorHeightEstimate
    from racer.vision.heading_vp import HeadingEstimate

    def fake_heading(image_bgr, roll, pitch, **kw):
        branches = np.array([0.0, np.pi / 2, np.pi, -np.pi / 2])
        return HeadingEstimate(heading_mod90_rad=0.0, quality=0.9, n_support=40,
                               vp_px=np.array([320.0, 180.0]), horizontality=0.05,
                               branch_headings_rad=branches)

    def fake_floor(image_bgr, roll, pitch, **kw):
        return FloorHeightEstimate(height_m=3.0, quality=0.9, n_support=6, std_m=0.10)

    gate = _gate_facing_north([30.0, 0.0, 0.0])
    drone = [0.0, 0.0, 0.0]
    orig_h, orig_f = navmod.estimate_heading, navmod.estimate_floor_height
    try:
        navmod.estimate_heading = fake_heading
        navmod.estimate_floor_height = fake_floor
        nav_default = _make_nav(gate, drone, use_vp_yaw=True, use_floor_height=True,
                                floor_height_min_quality=0.3)                      # decimate defaults (1)
        nav_explicit1 = _make_nav(gate, drone, use_vp_yaw=True, use_floor_height=True,
                                  floor_height_min_quality=0.3,
                                  vp_yaw_decimate=1, floor_height_decimate=1)
        assert nav_default.config.vp_yaw_decimate == 1 and nav_default.config.floor_height_decimate == 1
        for k in range(30):
            t = k * _DT_NS
            ds = _ds(t, gyro_body=np.array([0.0, 0.0, np.deg2rad(6.0)]))
            nav_default.update(ds, _nav_frame(k, t))
            nav_explicit1.update(ds, _nav_frame(k, t))
    finally:
        navmod.estimate_heading = orig_h
        navmod.estimate_floor_height = orig_f
    np.testing.assert_array_equal(nav_default.kf.x, nav_explicit1.kf.x)
    np.testing.assert_array_equal(nav_default.kf.P, nav_explicit1.kf.P)
    np.testing.assert_array_equal(nav_default._ahrs.q_wxyz, nav_explicit1._ahrs.q_wxyz)


def test_decimate_n_skips_the_backstop():
    """With vp_yaw_decimate=5 / floor_height_decimate=3, over 30 processed vision ticks the backstop
    estimators are CALLED only on the N-th ticks (30/5=6 vp_yaw calls, 30/3=10 floor calls) -- i.e.
    the off-ticks skip estimate_heading / estimate_floor_height ENTIRELY (the compute cut)."""
    import racer.navigator as navmod
    from racer.vision.floor_height import FloorHeightEstimate
    from racer.vision.heading_vp import HeadingEstimate

    h_calls = {"n": 0}
    f_calls = {"n": 0}

    def counting_heading(image_bgr, roll, pitch, **kw):
        h_calls["n"] += 1
        branches = np.array([0.0, np.pi / 2, np.pi, -np.pi / 2])
        return HeadingEstimate(heading_mod90_rad=0.0, quality=0.9, n_support=40,
                               vp_px=np.array([320.0, 180.0]), horizontality=0.05,
                               branch_headings_rad=branches)

    def counting_floor(image_bgr, roll, pitch, **kw):
        f_calls["n"] += 1
        return FloorHeightEstimate(height_m=3.0, quality=0.9, n_support=6, std_m=0.10)

    gate = _gate_facing_north([30.0, 0.0, 0.0])
    orig_h, orig_f = navmod.estimate_heading, navmod.estimate_floor_height
    try:
        navmod.estimate_heading = counting_heading
        navmod.estimate_floor_height = counting_floor
        nav = _make_nav(gate, [0.0, 0.0, 0.0], use_vp_yaw=True, use_floor_height=True,
                        floor_height_min_quality=0.3, vp_yaw_decimate=5, floor_height_decimate=3)
        # The FIRST update() only seeds the KF (returns before vision), so run it separately, then
        # count exactly n_ticks PROCESSED vision ticks (frames on subsequent updates).
        nav.update(_ds(0, gyro_body=np.zeros(3)))            # init tick (no vision)
        n_ticks = 30
        for k in range(1, n_ticks + 1):
            t = k * _DT_NS
            ds = _ds(t, gyro_body=np.zeros(3))
            nav.update(ds, _nav_frame(k, t))
    finally:
        navmod.estimate_heading = orig_h
        navmod.estimate_floor_height = orig_f
    # n_ticks processed vision ticks: vp_yaw on ticks 5,10,...,30 = 6; floor on 3,6,...,30 = 10.
    assert nav._vision_tick_count == n_ticks
    assert h_calls["n"] == n_ticks // 5, f"vp_yaw ran {h_calls['n']}x, expected {n_ticks // 5}"
    assert f_calls["n"] == n_ticks // 3, f"floor ran {f_calls['n']}x, expected {n_ticks // 3}"


def test_decimate_n_reduces_backstop_calls_vs_every_tick():
    """Sanity: decimate>1 strictly REDUCES the estimator call count vs decimate=1 (the compute win)."""
    import racer.navigator as navmod
    from racer.vision.heading_vp import HeadingEstimate

    calls = {"one": 0, "five": 0}

    def make_stub(key):
        def stub(image_bgr, roll, pitch, **kw):
            calls[key] += 1
            branches = np.array([0.0, np.pi / 2, np.pi, -np.pi / 2])
            return HeadingEstimate(heading_mod90_rad=0.0, quality=0.9, n_support=40,
                                   vp_px=np.array([320.0, 180.0]), horizontality=0.05,
                                   branch_headings_rad=branches)
        return stub

    gate = _gate_facing_north([30.0, 0.0, 0.0])
    orig_h = navmod.estimate_heading
    try:
        for key, dec in (("one", 1), ("five", 5)):
            navmod.estimate_heading = make_stub(key)
            nav = _make_nav(gate, [0.0, 0.0, 0.0], use_vp_yaw=True, vp_yaw_decimate=dec)
            nav.update(_ds(0, gyro_body=np.zeros(3)))        # init tick (no vision)
            for k in range(1, 21):                           # 20 processed vision ticks
                t = k * _DT_NS
                nav.update(_ds(t, gyro_body=np.zeros(3)), _nav_frame(k, t))
    finally:
        navmod.estimate_heading = orig_h
    assert calls["one"] == 20 and calls["five"] == 4          # 20 vs 20//5
    assert calls["five"] < calls["one"]
