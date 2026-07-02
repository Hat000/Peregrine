"""TRT-engine opt-in seam pins (TRT sidequest, 2026-07-01).

The TensorRT gate-detector engine is OPT-IN via ``--seeker-weights <x.engine>``; these tests pin
the two seams that make that work WITHOUT touching the proven .pt path:

  1. fly_rl ``_looks_like_detector_weights``: accepts ``.engine`` / ``.onnx`` (alone and as
     ensemble members) exactly like ``.pt``; the RL-actor ``.pth`` footgun stays REJECTED.
  2. detector ``_load_yolo_model``: pins ``task="pose"`` for ``.engine`` / ``.onnx`` specs —
     without it ultralytics ``guess_model_task`` falls back to "detect" and the post-process
     silently drops every keypoint (zero gate observations, no error). A ``.pt`` spec must take
     the EXACT legacy single-arg ``YOLO(weights)`` call (no task kwarg — byte-identical path).

Run: .venv\\Scripts\\python.exe -m pytest tests/test_engine_weights_optin.py -q
"""
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "rl"))


def test_validator_accepts_engine_and_onnx_rejects_pth():
    import pytest

    torch = pytest.importorskip("torch")  # noqa: F841  (fly_rl imports torch at module top)
    from fly_rl import _looks_like_detector_weights

    assert _looks_like_detector_weights("models/gate.pt")
    assert _looks_like_detector_weights("models/gate_fp16_384x640.engine")
    assert _looks_like_detector_weights("models/gate.onnx")
    assert _looks_like_detector_weights("a.pt++b.pt")
    assert _looks_like_detector_weights("a.pt++b.engine")
    # the original footgun: the RL-actor checkpoint must NEVER pass as detector weights
    assert not _looks_like_detector_weights("rl/checkpoints/stage1_inc7_actor.pth")
    assert not _looks_like_detector_weights("a.pt++c.pth")
    assert not _looks_like_detector_weights(None)
    assert not _looks_like_detector_weights("")


def test_load_yolo_model_pins_pose_task_for_exports_only(monkeypatch):
    calls = []

    class FakeYOLO:
        def __init__(self, weights, *args, **kwargs):
            calls.append((weights, args, kwargs))

    fake = types.ModuleType("ultralytics")
    fake.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake)

    from racer.vision.detector import _load_yolo_model

    _load_yolo_model("models/gate_fp16_384x640.engine")
    _load_yolo_model("MODELS/GATE.ENGINE")  # case-insensitive
    _load_yolo_model("models/gate.onnx")
    _load_yolo_model("models/gate.pt")
    assert calls[0] == ("models/gate_fp16_384x640.engine", (), {"task": "pose"})
    assert calls[1] == ("MODELS/GATE.ENGINE", (), {"task": "pose"})
    assert calls[2] == ("models/gate.onnx", (), {"task": "pose"})
    # .pt: the exact legacy single-positional-arg call — task kwarg NOT passed at all
    assert calls[3] == ("models/gate.pt", (), {})
