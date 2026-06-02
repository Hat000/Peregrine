"""Render a few multi-gate scenes and annotate every gate's corners to sanity-check the
generator (instance separation + near/far occlusion). Writes demo/scene_*.png."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from racer.vision.synthetic import V_OCC, V_OFF, render_scene   # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "demo"
COLORS = [(0, 220, 0), (0, 200, 255), (255, 120, 0)]   # per-gate (BGR)


def main() -> int:
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(11)
    made = 0
    while made < 6:
        level = int(rng.choice([2, 3]))
        img, samples = render_scene(rng, level=level, max_gates=3)
        vis_gates = [s for s in samples if s.visible]
        if len(vis_gates) < 2:            # only keep frames that actually show multiple gates
            continue
        canvas = img.copy()
        for gi, s in enumerate(samples):
            col = COLORS[gi % len(COLORS)]
            for c, ((x, y), v) in enumerate(zip(s.keypoints_px, s.visibility)):
                if v == V_OFF:
                    continue
                p = (int(round(x)), int(round(y)))
                cv2.circle(canvas, p, 5, col, -1 if v != V_OCC else 2)   # hollow = occluded
                cv2.putText(canvas, str(c), (p[0] + 5, p[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        cv2.putText(canvas, f"L{level}  {len(vis_gates)} gates  (filled=visible, hollow=occluded)",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.imwrite(str(OUT / f"scene_{made}.png"), canvas)
        made += 1
    print(f"wrote {made} multi-gate previews -> {OUT}\\scene_0..{made-1}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
