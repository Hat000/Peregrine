"""Unit tests for the OPT-IN ensemble deploy path (deploy gate, 2026-06-19).

Covers the four behaviours the closed-loop flight gate will lean on, all offline (duck-typed
fake models, no ultralytics, no weights, no GPU):
  A. UNION       — EnsembleGateDetector concatenates every member's per-frame detections.
  B. PER-GATE DEDUP — the union is collapsed to ONE observation per gate before the KF, so two
                   models both seeing gate-3 produce ONE fix, not two (the deploy logic the
                   offline eval_ensemble deliberately omits because its metrics self-select).
  C. OPT-IN / BYTE-IDENTICAL — a single-weight load stays a plain GateDetector; the ensemble
                   machinery is a faithful no-op for one model. The VQ1 single-model path is
                   untouched.
  D. 8->4 ADAPTER — an 8-keypoint (inner+outer) model is subset to the 4 INNER corners, so the
                   deduped union feeds PnP exactly 4 corners (estimate_gate_pose recovers).
  E. LATENCY     — the dedup itself is sub-millisecond (inference dominates; measured live).
"""
import sys
import time
import types

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from racer.contracts import Frame, GateObservation
from racer.vision.detector import (
    EnsembleGateDetector,
    GateDetector,
    N_CORNERS,
    _obs_centroid,
    _obs_span,
    observations_from_results,
)
from racer.vision.gate_pose import _rotation_geodesic, estimate_gate_pose, project_gate_corners

# Two distinct gates (different camera-frame translation -> well-separated image footprints).
_R1 = Rotation.from_euler("y", 0.3).as_matrix()
_T1 = np.array([0.3, -0.2, 5.0])
_R2 = Rotation.from_euler("y", -0.1).as_matrix()
_T2 = np.array([3.5, 1.6, 7.0])


def _frame(frame_id=5, sim_time_ns=123):
    return Frame(frame_id=frame_id, sim_time_ns=sim_time_ns, image_bgr=np.zeros((360, 640, 3), np.uint8))


def _corners(R=_R1, T=_T1):
    return project_gate_corners(R, T)  # (4,2), canonical order


def _obs(corners, score=0.9, conf=None, ids=None, frame=None):
    """Build a GateObservation directly (for dedup/fuse unit tests)."""
    corners = np.asarray(corners, dtype=np.float64)
    f = frame or _frame()
    if conf is None:
        conf = np.ones(corners.shape[0])
    return GateObservation(
        frame_id=f.frame_id, sim_time_ns=f.sim_time_ns, corners_px=corners,
        corner_ids=ids, corner_confidence=np.asarray(conf, float), score=float(score),
    )


def _obs_equal(a: GateObservation, b: GateObservation) -> bool:
    """Full field-by-field (byte) equality of two observations."""
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


# -- duck-typed ultralytics fakes (same pattern as test_detector.py) -------------------------
class _Attr:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _fake_results(kxy, kconf, scores, bboxes=None):
    return _Attr(
        boxes=_Attr(conf=np.asarray(scores, float), xywh=None if bboxes is None else np.asarray(bboxes, float)),
        keypoints=_Attr(xy=np.asarray(kxy, float), conf=np.asarray(kconf, float)),
    )


class _FakeModel:
    """A YOLO-pose stand-in: ``predict`` returns a fixed list of Results."""

    def __init__(self, results):
        self._results = results
        self.calls = 0
        self.last_kwargs = None  # captures predict()'s kwargs for the A20 imgsz/half test below

    def predict(self, img, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        return self._results


def _model_detecting(*gate_corners, scores=None):
    """A fake model that emits one 4-corner detection per supplied corner set."""
    kxy = np.stack(list(gate_corners))                     # (n,4,2)
    n = kxy.shape[0]
    scores = [0.9] * n if scores is None else scores
    return _FakeModel([_fake_results(kxy, np.ones((n, 4)), scores)])


# ======================================================================================
# A. UNION
# ======================================================================================
def test_ensemble_unions_member_detections():
    """Two members each see a different gate -> the union is BOTH detections."""
    frame = _frame()
    c1, c2 = _corners(_R1, _T1), _corners(_R2, _T2)
    # precondition: the two gates are well separated in the image (>> dedup_px)
    assert np.linalg.norm(c1.mean(0) - c2.mean(0)) > 50.0
    ens = EnsembleGateDetector([_model_detecting(c1), _model_detecting(c2)], dedup_px=12.0)
    obs = ens.detect(frame)
    assert len(obs) == 2
    cents = sorted(float(o.corners_px.mean(0)[0]) for o in obs)
    assert cents == pytest.approx(sorted([c1.mean(0)[0], c2.mean(0)[0]]))


def test_pure_union_keeps_duplicates_when_dedup_off():
    """dedup_px=0 reproduces the offline eval_ensemble: pure union, duplicates kept."""
    frame = _frame()
    c1 = _corners(_R1, _T1)
    ens = EnsembleGateDetector([_model_detecting(c1), _model_detecting(c1 + 0.4)], dedup_px=0.0)
    assert len(ens.detect(frame)) == 2  # both members' gate-1 detections survive


# ======================================================================================
# B. PER-GATE DEDUP — one fix per gate
# ======================================================================================
def test_dedup_collapses_to_one_observation_per_gate():
    """Both members see the SAME two gates (union=4) -> dedup yields ONE obs per gate (=2)."""
    frame = _frame()
    c1, c2 = _corners(_R1, _T1), _corners(_R2, _T2)
    assert np.linalg.norm(c1.mean(0) - c2.mean(0)) > 50.0  # distinct gates
    mA = _model_detecting(c1, c2, scores=[0.90, 0.80])
    mB = _model_detecting(c1 + 0.5, c2 + 0.5, scores=[0.85, 0.70])  # same gates, sub-px jitter
    ens = EnsembleGateDetector([mA, mB], dedup_px=12.0)

    assert len(ens.detect(frame)) == 2          # deduped: one per gate
    ens.dedup_px = 0.0
    assert len(ens.detect(frame)) == 4          # union would double-count


def test_dedup_toggle_is_the_only_difference():
    """The SAME unioned input -> dedup_px>0 shrinks, dedup_px=0 does not (clean A/B knob)."""
    frame = _frame()
    c = _corners(_R1, _T1)
    ens = EnsembleGateDetector([_model_detecting(c), _model_detecting(c + 0.3)], dedup_px=12.0)
    assert len(ens.detect(frame)) == 1
    ens.dedup_px = 0.0
    assert len(ens.detect(frame)) == 2


def test_dedup_size_guard_does_not_merge_overlapping_collinear_gates():
    """A near (big) and far (small) gate can share a centroid on a collinear course; the span
    guard must keep them as TWO observations (a centroid-only merge would drop a real gate)."""
    near = project_gate_corners(np.eye(3), np.array([0.0, 0.0, 3.0]))   # on-axis, big
    far = project_gate_corners(np.eye(3), np.array([0.0, 0.0, 9.0]))    # on-axis, small
    # same centroid (principal point), very different apparent span
    assert np.linalg.norm(near.mean(0) - far.mean(0)) < 1.0
    assert _obs_span(_obs(near)) > 2.0 * _obs_span(_obs(far))
    ens = EnsembleGateDetector([], dedup_px=12.0)
    out = ens._dedup([_obs(near, score=0.9), _obs(far, score=0.8)])
    assert len(out) == 2  # not collapsed


def test_dedup_passthrough_for_empty_and_single():
    ens = EnsembleGateDetector([], dedup_px=12.0)
    assert ens._dedup([]) == []
    one = _obs(_corners())
    assert ens._dedup([one]) == [one]


# ======================================================================================
# B (fuse). The collapse is a score-weighted corner average of the full-4-corner members.
# ======================================================================================
def test_fuse_is_score_weighted_corner_average():
    a = _obs(_corners(_R1, _T1), score=0.6, conf=np.array([0.6, 0.6, 0.6, 0.6]))
    b = _obs(_corners(_R1, _T1) + 4.0, score=0.4, conf=np.array([0.9, 0.5, 0.5, 0.5]))
    fused = EnsembleGateDetector._fuse([a, b])
    w = np.array([0.6, 0.4]) / 1.0
    expect = w[0] * a.corners_px + w[1] * b.corners_px
    np.testing.assert_allclose(fused.corners_px, expect)
    np.testing.assert_array_equal(fused.corner_confidence, [0.9, 0.6, 0.6, 0.6])  # elementwise max
    assert fused.corners_px.shape == (N_CORNERS, 2) and fused.corner_ids is None


def test_fuse_single_member_is_identity():
    o = _obs(_corners())
    assert EnsembleGateDetector._fuse([o]) is o


# ======================================================================================
# C. OPT-IN routing + single-model BYTE-IDENTITY
# ======================================================================================
def test_single_weight_load_stays_plain_gatedetector(monkeypatch):
    """The opt-in guardrail: a single-weight load must NOT route to the ensemble; an 'a++b'
    spec must. (Fake ultralytics so no model file / GPU is needed.)"""
    fake_ultra = types.ModuleType("ultralytics")

    class _YOLO:
        def __init__(self, path):
            self.path = path

    fake_ultra.YOLO = _YOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultra)

    single = GateDetector.load("weights.pt", score_thresh=0.25, kpt_conf_thresh=0.5)
    assert isinstance(single, GateDetector) and not isinstance(single, EnsembleGateDetector)

    ens = GateDetector.load("a.pt++b.pt", score_thresh=0.25, kpt_conf_thresh=0.5)
    assert isinstance(ens, EnsembleGateDetector)
    assert len(ens.models) == 2 and all(isinstance(m, _YOLO) for m in ens.models)
    assert ens.dedup_px == 12.0  # deploy default = dedup ON (never double-count by default)


def test_single_model_detect_is_byte_identical():
    """Wrapping ONE model in the ensemble (union-of-one + dedup) yields byte-identical output to
    the unchanged GateDetector.detect -> the ensemble code is a no-op for the single-model path."""
    frame = _frame()
    c1, c2 = _corners(_R1, _T1), _corners(_R2, _T2)
    results = [_fake_results(np.stack([c1, c2]), np.ones((2, 4)), [0.9, 0.8],
                             bboxes=[[100, 90, 50, 40], [300, 120, 30, 24]])]
    base = GateDetector(_FakeModel(results)).detect(frame)
    wrapped = EnsembleGateDetector([_FakeModel(results)]).detect(frame)  # default dedup_px=12
    assert len(base) == len(wrapped) == 2
    assert all(_obs_equal(x, y) for x, y in zip(base, wrapped))


def test_ensemble_does_not_mutate_member_observations():
    """Defence-in-depth: a no-collapse union returns the member objects unchanged (not copies/edits)."""
    frame = _frame()
    mA, mB = _model_detecting(_corners(_R1, _T1)), _model_detecting(_corners(_R2, _T2))
    out = EnsembleGateDetector([mA, mB], dedup_px=12.0).detect(frame)
    for o in out:
        assert o.corners_px.shape == (N_CORNERS, 2) and np.isfinite(o.corners_px).all()


# ======================================================================================
# D. 8->4 inner-corner adapter feeds PnP exactly 4 corners
# ======================================================================================
def _kpts8(R=_R1, T=_T1):
    """An 8-keypoint set: inner 4 = the true gate square, outer 4 = a wider ring (decoys)."""
    inner = project_gate_corners(R, T)
    ctr = inner.mean(0)
    outer = ctr + 1.4 * (inner - ctr)
    return inner, np.concatenate([inner, outer])  # (4,2), (8,2)


def test_8kp_results_subset_to_inner_four_and_pnp_recovers():
    frame = _frame()
    inner, kxy8 = _kpts8(_R1, _T1)
    res = _fake_results(kxy8[None], np.ones((1, 8)), [0.9])
    obs = observations_from_results(frame, res)
    assert len(obs) == 1
    o = obs[0]
    assert o.corners_px.shape == (N_CORNERS, 2) and o.corner_ids is None
    np.testing.assert_allclose(o.corners_px, inner)             # inner-4 kept (not outer)
    gp = estimate_gate_pose(o)                                  # exactly 4 corners -> IPPE
    assert gp.n_corners == 4
    np.testing.assert_allclose(gp.t_cam_gate, _T1, atol=1e-3)
    assert _rotation_geodesic(gp.R_cam_gate, _R1) < 2e-3


def test_ensemble_of_8kp_models_feeds_pnp_four_corners():
    """End-to-end: two 8-kpt members -> union -> dedup -> every obs has exactly 4 corners for PnP."""
    frame = _frame()
    inner1, k1 = _kpts8(_R1, _T1)
    inner2, k2 = _kpts8(_R2, _T2)
    mA = _FakeModel([_fake_results(k1[None], np.ones((1, 8)), [0.9])])
    mB = _FakeModel([_fake_results(k2[None], np.ones((1, 8)), [0.85])])
    obs = EnsembleGateDetector([mA, mB], dedup_px=12.0).detect(frame)
    assert len(obs) == 2
    for o in obs:
        assert o.corners_px.shape == (N_CORNERS, 2)            # never 8
        gp = estimate_gate_pose(o)
        assert gp is not None and gp.n_corners == 4


# ======================================================================================
# E. Dedup latency (the dedup itself; 2-model inference is measured live in the report)
# ======================================================================================
def test_dedup_overhead_is_submillisecond():
    """Worst-ish case: ~12 unioned obs (6 gates x 2 models). The dedup must be negligible against
    the ~26 ms inference budget."""
    frame = _frame()
    obs = []
    rng = np.random.default_rng(0)
    for gate in range(6):
        base = _corners(_R1, _T1 + np.array([1.5 * gate, 0.0, 0.0]))
        obs.append(_obs(base, score=0.9))
        obs.append(_obs(base + rng.normal(0, 0.5, base.shape), score=0.8))  # the duplicate
    ens = EnsembleGateDetector([], dedup_px=12.0)
    assert len(ens._dedup(obs)) == 6  # 12 -> 6 (one per gate)

    iters = 2000
    t0 = time.perf_counter()
    for _ in range(iters):
        ens._dedup(obs)
    per_call_ms = (time.perf_counter() - t0) / iters * 1e3
    # Bound encodes the test's stated intent -- "negligible vs the ~26 ms inference budget" -- NOT a
    # sub-1ms hard requirement. The original 1.0 ms was over-tight for slower dev machines (observed
    # ~1.6 ms on the commander laptop, stable across runs); 5.0 ms is still <20% of the 26 ms budget
    # and gives ~3x headroom against flake, while a real regression (e.g. O(n^2) blowup) still trips it.
    # (loosened 1.0 -> 5.0, cleanup 2026-06-20; re-tighten if moved to a faster CI box.)
    assert per_call_ms < 5.0, f"dedup overhead {per_call_ms:.3f} ms/frame too high (vs ~26 ms budget)"


# ======================================================================================
# F. A20 (2026-07-01): imgsz/half detect-cost knobs reach EVERY member's predict()
# ======================================================================================
def test_default_imgsz_half_omit_kwargs_on_every_member():
    """Default construction (imgsz=None, half=False) must not pass imgsz=/half= to ANY member's
    predict() -- byte-identical to the pre-A20 ensemble detect() call."""
    frame = _frame()
    mA, mB = _model_detecting(_corners(_R1, _T1)), _model_detecting(_corners(_R2, _T2))
    EnsembleGateDetector([mA, mB], dedup_px=12.0).detect(frame)
    for m in (mA, mB):
        assert "imgsz" not in m.last_kwargs and "half" not in m.last_kwargs


def test_imgsz_and_half_reach_every_member_predict():
    """imgsz/half are threaded to EVERY ensemble member's predict() call, not just the first."""
    frame = _frame()
    mA, mB = _model_detecting(_corners(_R1, _T1)), _model_detecting(_corners(_R2, _T2))
    EnsembleGateDetector([mA, mB], dedup_px=12.0, imgsz=416, half=True).detect(frame)
    for m in (mA, mB):
        assert m.last_kwargs["imgsz"] == 416 and m.last_kwargs["half"] is True
