"""Dump recorded flight frames into a YOLO-layout dataset for TensorRT INT8 calibration.

INT8 calibration needs images from the DEPLOY distribution (recorded in-race frames), not
the Blender-clean training set. This writes ``<out>/images/*.jpg`` (the raw recorded JPEGs,
no re-encode) + a minimal ``<out>/calib.yaml`` that ``scripts/export_trt_engine.py
--precision int8 --data <out>/calib.yaml`` can consume. Calibration runs images through the
preprocessing only — no labels needed (ultralytics warns about background-only images; fine).

Run (ShadowPC — ALWAYS the .venv python):
  .venv/Scripts/python.exe scripts/make_calib_dataset.py \
      --run-dir data/runs/20260701_224640_vq2_slow_seeker_a19c_f1 --out data/_trt_calib
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from racer.recording import RecordingReader                        # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", action="append", required=True,
                    help="recorded session dir(s); repeat the flag to mix runs")
    ap.add_argument("--out", required=True, help="output dataset root")
    ap.add_argument("--max-frames", type=int, default=300)
    args = ap.parse_args()

    out = Path(args.out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for rd in args.run_dir:
        stamp = Path(rd).name
        for entry, jpeg in RecordingReader(rd).iter_jpeg():
            (img_dir / f"{stamp}_{entry['frame_id']:06d}.jpg").write_bytes(jpeg)
            n += 1
            if n >= args.max_frames:
                break
        if n >= args.max_frames:
            break

    # kpt_shape matches the 8-kpt gate contract (blender_gen/contract.py); calibration itself
    # only reads images, but the pose-dataset loader wants the field present.
    (out / "calib.yaml").write_text(
        f"path: {out.resolve().as_posix()}\n"
        "train: images\n"
        "val: images\n"
        "kpt_shape: [8, 3]\n"
        "names:\n  0: gate\n",
        encoding="utf-8",
    )
    print(f"{n} frames -> {img_dir}")
    print(f"yaml     -> {out / 'calib.yaml'}")
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())
