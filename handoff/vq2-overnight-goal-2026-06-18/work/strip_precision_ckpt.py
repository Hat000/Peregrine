"""Re-export a --precision-loss checkpoint so it loads in a CLEAN env (no cluster/ on path).
The training ckpt pickles cluster/vq2_precision_loss.PrecisionKeypointLoss via the attached criterion;
removing the criterion + re-saving drops that dependency. Inference is unaffected (criterion is train-only).

  python strip_precision_ckpt.py <src.pt> <dst.pt>
"""
import sys
from pathlib import Path

sys.path.insert(0, "C:/Users/Shadow/Peregrine/cluster")  # needed to UNPICKLE the source
from ultralytics import YOLO  # noqa: E402

src, dst = sys.argv[1], sys.argv[2]
m = YOLO(src)
model = m.model
for attr in ("criterion", "loss", "compute_loss"):
    if hasattr(model, attr):
        try:
            setattr(model, attr, None)
        except Exception:
            pass
Path(dst).parent.mkdir(parents=True, exist_ok=True)
m.save(dst)
print("STRIPPED", src, "->", dst)
