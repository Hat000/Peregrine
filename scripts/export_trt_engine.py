"""Export a YOLO-pose .pt to a TensorRT engine (the VQ2 detect-latency + edge-deploy artifact).

TensorRT engines are HARDWARE + TRT-VERSION specific: an engine built here (RTX 2000 Ada,
sm_89) will NOT load on a Jetson Orin (sm_87) or under a different TensorRT major version.
The .pt stays the source of truth; re-run THIS script on each deploy target:

    Orin rebuild recipe (JetPack ships its own TensorRT — do NOT pip-install one):
      python scripts/export_trt_engine.py --weights models/gate_clean_ens_course_L110.pt \
             --imgsz 384 640 --precision fp16 --out models/<name>_orin_fp16_384x640.engine

Engines are derived artifacts -> gitignored (*.engine, plus the intermediate *.onnx).

Run (ShadowPC — ALWAYS the .venv python, a second PATH python has mismatched torch):
  .venv/Scripts/python.exe scripts/export_trt_engine.py --weights models/gate_clean_ens_course_L110.pt \
      --imgsz 384 640 --precision fp16
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path


def log_versions() -> None:
    """Print the full toolchain so every downstream number is attributable (two pythons live
    on this box; a silently-wrong interpreter would invalidate the whole bench)."""
    import tensorrt
    import torch
    import ultralytics

    print(f"python     : {sys.executable}")
    print(f"torch      : {torch.__version__} (cuda {torch.version.cuda})")
    print(f"ultralytics: {ultralytics.__version__}")
    print(f"tensorrt   : {tensorrt.__version__}")
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        print(f"gpu        : {torch.cuda.get_device_name(0)} (sm_{cap[0]}{cap[1]})")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="source YOLO .pt (stays untouched)")
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640],
                    help="engine input size: one int (square) or H W. The live camera is 360x640 "
                         "-> 384 640 matches the .pt path's stride-aligned rect letterbox exactly "
                         "(best parity, ~40%% less compute than square 640).")
    ap.add_argument("--precision", choices=("fp16", "fp32", "int8"), default="fp16")
    ap.add_argument("--device", default="0", help="CUDA device index for the build")
    ap.add_argument("--batch", type=int, default=1, help="max batch size (flight stack uses 1)")
    ap.add_argument("--workspace", type=float, default=4.0, help="TRT builder workspace, GiB")
    ap.add_argument("--dynamic", action="store_true", help="dynamic input shapes (default: static)")
    ap.add_argument("--data", default=None,
                    help="dataset yaml for int8 calibration (required for --precision int8)")
    ap.add_argument("--out", default=None,
                    help="final engine path; default: <weights_stem>_<precision>_<HxW>.engine "
                         "next to the weights")
    args = ap.parse_args()

    log_versions()

    weights = Path(args.weights).resolve()
    if not weights.exists():
        print(f"ERROR: weights not found: {weights}", file=sys.stderr)
        return 2
    if args.precision == "int8" and not args.data:
        print("ERROR: --precision int8 needs --data <calib dataset yaml>", file=sys.stderr)
        return 2
    imgsz = args.imgsz if len(args.imgsz) > 1 else args.imgsz[0]

    from ultralytics import YOLO

    model = YOLO(str(weights))
    # export writes <weights_stem>.engine (and an intermediate .onnx) next to the weights
    produced = model.export(
        format="engine",
        imgsz=imgsz,
        half=(args.precision == "fp16"),
        int8=(args.precision == "int8"),
        data=args.data,
        device=args.device,
        batch=args.batch,
        workspace=args.workspace,
        dynamic=args.dynamic,
    )
    produced = Path(produced)

    hw = "x".join(str(s) for s in (imgsz if isinstance(imgsz, list) else [imgsz, imgsz]))
    out = Path(args.out).resolve() if args.out else produced.with_name(
        f"{weights.stem}_{args.precision}_{hw}.engine")
    if out != produced:
        shutil.move(str(produced), str(out))
    print(f"\nengine : {out}")
    print(f"size   : {out.stat().st_size / 1e6:.1f} MB")
    print(f"sha256 : {sha256(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
