"""LEVER 5: test-time augmentation (no retrain). Monkeypatch GateDetector.detect to run ultralytics
predict(augment=True), then score the standard good-fix + course metrics. A deploy-time win if it lifts
course valid-fix CI-credibly over the champion (155/204) with no <0.5m/<1m/<2m regression.

  python eval_tta.py <weights.pt> [more ...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import eval_goodfix as E  # noqa: E402
from wilson import wilson  # noqa: E402
from racer.vision.detector import GateDetector, observations_from_results  # noqa: E402


def tta_detect(self, frame):
    results = self.model.predict(frame.image_bgr, verbose=False, device=self.device, augment=True)
    if not results:
        return []
    return observations_from_results(frame, results[0],
                                     score_thresh=self.score_thresh, kpt_conf_thresh=self.kpt_conf_thresh)


GateDetector.detect = tta_detect  # enable TTA for every detect() in this process

print(f"{'checkpoint (TTA)':34}{'<0.5m':>20}{'<1m':>8}{'<2m':>8}{'course/204':>22}")
for w in sys.argv[1:]:
    e = E.task2_goodfix(w); N = len(e)
    n05 = int((e < 0.5).sum()); n1 = int((e < 1).sum()); n2 = int((e < 2).sum())
    v, M = E.course_validfix(w)
    _, lo, hi = wilson(n05, N); _, clo, chi = wilson(v, M)
    print(f"{Path(w).name:34}{f'{n05}/{N} [{100*lo:.0f},{100*hi:.0f}]':>20}"
          f"{f'{100*n1/N:.0f}%':>8}{f'{100*n2/N:.0f}%':>8}"
          f"{f'{v}/{M} [{100*clo:.0f},{100*chi:.0f}]':>22}", flush=True)
print("CHAMPION (no TTA): 32/40 [65,90] | 95% | 98% | 155/204 [70,81]")
