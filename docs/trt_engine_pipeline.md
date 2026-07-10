# TensorRT gate-detector engine — export pipeline (TRT sidequest, 2026-07-01)

> **Current shipped detector (2026-07-06): `vq2_partial_m_2026-07-06`** (yolo11m-pose, 8-kpt,
> partial/cropped-gate arm). Weights + the prebuilt `sm_89` engine live on the GitHub release
> **`vision-m-2026-07-06`** — they are gitignored here (derived artifacts; the `.pt` is >100 MB).
> Its measured export: fp16 384x640 **PARITY PASS** (corner err median 0.01 px, 98.8 % count
> agreement), detect 14.9 → 6.2 ms (**2.4x**). To consume it from RL see
> [`handoff/vision-model-rl-handoff-2026-07-09.md`](../handoff/vision-model-rl-handoff-2026-07-09.md)
> and `scripts/rl_gate_obs.py`. The recipe below is model-agnostic; `gate_clean_ens_course_L110`
> is the original worked example.

The VQ2 gate detector (`models/gate_clean_ens_course_L110.pt`, YOLO-pose, 8-kpt) runs eager
FP32 through `ultralytics predict()` at ~250 ms/detect on ShadowPC when the sim's render owns
the GPU — not compute-bound but SYNC-bound (~8–13 CPU↔GPU sync points per predict, each a
stall behind the render queue). A TensorRT engine collapses the model into one optimized
graph with ONE completion sync, cutting both raw inference and the number of stall points.
The same export pipeline produces the Jetson Orin (edge) artifact.

## The invariants

- **The `.pt` stays the source of truth.** Engines are DERIVED, hardware- and
  TRT-version-specific artifacts: an engine built on ShadowPC (RTX 2000 Ada, sm_89) will NOT
  load on an Orin (sm_87) or under a different TensorRT major version. `*.engine` / `*.onnx`
  are gitignored; rebuild per target with `scripts/export_trt_engine.py`.
- **Engines are OPT-IN**, via `--seeker-weights <x>.engine`. The `.pt` flight path is
  byte-identical (pinned by `tests/test_engine_weights_optin.py`).
- **`task="pose"` is load-bearing**: exported graphs carry no pickled task, and ultralytics
  would guess `detect` and silently post-process the keypoints away (zero gate observations,
  no error). `racer.vision.detector._load_yolo_model` pins it for `.engine`/`.onnx` specs.
- **Accuracy parity is mandatory before flying an engine**: run
  `scripts/bench_detector_engine.py` (same recorded frames, flight thresholds) and require
  its PASS verdict. A faster-but-shifted detector feeds PnP bad corners → bad gate poses.

## ShadowPC (dev) recipe

```bat
:: 1. toolchain (once) — ALWAYS the .venv python, a second PATH python has mismatched torch
.venv\Scripts\python.exe -m pip install "tensorrt-cu12<11" onnx onnxslim
::    (TRT 11 removed BuilderFlag.FP16 -> breaks the ultralytics 8.4.x exporter; pin <11.)

:: 2. build the fp16 engine, camera-matched 384x640 (live camera is 360x640; 384x640 is the
::    stride-32 rect letterbox the .pt path uses -> best parity + ~40% less compute than 640sq)
.venv\Scripts\python.exe scripts\export_trt_engine.py ^
    --weights models\gate_clean_ens_course_L110.pt --imgsz 384 640 --precision fp16

:: 3. smoke: the flight-stack GateDetector end-to-end on recorded frames
.venv\Scripts\python.exe scripts\demo_seeker_detect.py ^
    --weights models\gate_clean_ens_course_L110_fp16_384x640.engine ^
    --run-dir data\runs\20260701_224640_vq2_slow_seeker_a19c_f1

:: 4. parity + latency gate (>=200 in-race frames; must print PARITY PASS)
.venv\Scripts\python.exe scripts\bench_detector_engine.py ^
    --pt models\gate_clean_ens_course_L110.pt ^
    --engine models\gate_clean_ens_course_L110_fp16_384x640.engine ^
    --run-dir data\runs\20260701_224640_vq2_slow_seeker_a19c_f1
```

Fly it (opt-in, everything else unchanged):

```
... --gate-seeker --deploy-profile vq2_case_c --seeker-detector yolo ^
    --seeker-weights models\gate_clean_ens_course_L110_fp16_384x640.engine
```

Watch `[vision-timing] detect` (the contended number the offline bench cannot measure) and
`[loop-rate]`.

## Jetson Orin (edge) recipe

1. Do NOT pip-install tensorrt — JetPack ships the TensorRT matched to the Orin's CUDA.
   Verify with `python -c "import tensorrt; print(tensorrt.__version__)"`.
2. Install the project + detector extra, copy `models/gate_clean_ens_course_L110.pt` over
   (the portable artifact; `.onnx` also works as an intermediate if torch-on-Orin is a fight).
3. Rebuild on-device (sm_87 kernels are selected during the build; expect minutes):
   `python scripts/export_trt_engine.py --weights models/gate_clean_ens_course_L110.pt \
       --imgsz 384 640 --precision fp16`
4. Re-run the parity bench against recorded frames before trusting it.
5. If ultralytics/TRT versions differ from ShadowPC's (torch 2.12.0+cu126 /
   ultralytics 8.4.62 / tensorrt-cu12 10.16.1.11), parity — not version identity — is the
   acceptance test.

## INT8

`--precision int8` needs `--data <dataset yaml>` for calibration. Representative frames =
recorded in-race frames (NOT the Blender-clean training set alone — calibration should see
the deploy distribution). Only ship int8 if the parity bench still passes; fp16 is the
default deliverable.
