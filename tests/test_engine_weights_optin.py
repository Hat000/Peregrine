"""TRT-engine opt-in seam pin.

The TensorRT gate-detector engine is OPT-IN: ``fly_vq1.py --vision --weights <x.engine>`` (or any
caller of ``GateDetector.load``) selects it without touching the proven ``.pt`` path. The one seam
that makes that work is ``detector._load_yolo_model``: it pins ``task="pose"`` for ``.engine`` /
``.onnx`` specs — without it ultralytics ``guess_model_task`` falls back to "detect" and the
post-process silently drops every keypoint (zero gate observations, no error). A ``.pt`` spec must
take the EXACT legacy single-arg ``YOLO(weights)`` call (no task kwarg) so the flight path stays
byte-identical. (The branch's fly_rl ``_looks_like_detector_weights`` guard does not exist on this
deploy path — the ``--weights`` arg feeds ``GateDetector.load`` directly, so this seam is the whole
opt-in.)

Run: .venv\\Scripts\\python.exe -m pytest tests/test_engine_weights_optin.py -q
"""
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


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
