# VQ2 vision model → RL: how to use it (handoff, 2026-07-09)

Self-contained instructions for using the current best flight detector to emit the **closest**
and **second-closest** gate as egocentric (body-frame, map-free) RL observations.

Everything below is **measured**, not assumed. Numbers cite the run that produced them.

---

## 1. The model

| artifact | path | use |
|---|---|---|
| **M engine (fly this)** | `models/vq2_partial_m_2026-07-06_fp16_384x640.engine` | deploy / RL. fp16, 46 MB, sha256 `b3c312ad…` |
| M weights (source of truth) | `models/vq2_partial_m_2026-07-06.pt` | re-export, offline eval |
| prior champion (fallback) | `models/vq2_darkred_negreal42_2026-07-05.pt` | reference only |

**M** = yolo11m-pose, **8 keypoints** (4 inner corners 0–3 = the 1.5 m opening; 4 outer 4–7 = the
2.72 m frame). It was trained with a partial/cropped-gate arm, so unlike the champion it **does not
hallucinate off-frame corners** on clipped gates (visible-corner error 3.8 px vs the champion's
11.4 px). PnP is anchored on the inner 4.

Engine vs `.pt` are **equivalent**: parity PASS on 250 recorded frames — corner error median
0.01 px / max 1.04 px, 98.8 % detection-count agreement, body-frame rel-pos median Δ 2 mm.
Engine is **2.4× faster** (6.2 ms vs 14.9 ms median, *uncontended*).

---

## 2. Environment

Use **`.venv`** (it has TensorRT):

```
C:/Users/Shadow/Peregrine/.venv/Scripts/python.exe
# torch 2.12.0+cu126, ultralytics 8.4.62, tensorrt 10.16.1.11
```

`C:/Users/Shadow/vq2yolo-venv` is the **training** venv and has **no tensorrt** — it cannot load the
engine.

---

## 3. Use it (the emitter)

`scripts/rl_gate_obs.py` is the vision→RL bridge. Per frame it returns the **closest** and
**second-closest** gate, already in body frame, boresight-corrected:

```python
from racer.contracts import Frame
from racer.vision.detector import GateDetector
from rl_gate_obs import emit_gate_obs

det = GateDetector.load("models/vq2_partial_m_2026-07-06_fp16_384x640.engine")
gates = emit_gate_obs(frame, det, k=2)     # [closest, second_closest]; [] if none usable
```

Smoke-test it against a recording:

```
.venv/Scripts/python.exe scripts/rl_gate_obs.py \
    --run-dir data/runs/20260701_224640_vq2_slow_seeker_a19c_f1 --max-frames 60
```

Expected first line: `boresight vert_offset_m = -0.250 m`. If it prints `+0.000`, the boresight is
not baked and **every vertical reading is 0.30 m wrong** (see trap 1).

### Fields per gate

| field | meaning |
|---|---|
| `rel_pos_body` | (3,) gate centre in **body FRD**: `[forward, right, down]`, metres |
| `range_m` | `\|rel_pos_body\|` |
| `visible_area_ratio` | range-**free** foreshortening ≈ `\|cos(approach angle)\|`; 1.0 = head-on. **`None` if <4 corners** |
| `inner_area_px` | raw apparent opening area (px²). `None` if <4 corners |
| `score` | detector confidence |
| `n_corners` | 4 ⇒ IPPE; 3 ⇒ P3P (trust less) |
| `reproj_px` | PnP reprojection error |
| `ambiguity_ratio` | IPPE 2-fold `err2/err1`. **Near 1.0 ⇒ flip risk.** `None` if n/a |
| `apparent_size_px` | the ranking key (bbox area) |

There is **no world position and no map** anywhere in this path — it is egocentric by construction.

---

## 4. Three traps (each one measured; each one silently wrong if ignored)

### Trap 1 — the boresight offset is load-bearing
The camera optical centre sits at body offset `[0, 0, vert_offset_m]` from the body origin, so:

```
p_gate_body = [0, 0, frames.BORESIGHT.vert_offset_m] + R_camera_from_body().T @ pose.t_cam_gate
```

Omitting the `[0,0,vert_offset_m]` term injects a **+0.30 m vertical bias**. Measured on the task2
given-pose bundle (n=39, flip-resolved), body-frame `z(down)` residual vs ground truth:

| | z (down) residual |
|---|---|
| no boresight | **+0.303 ± 0.038 m** |
| deployed bake (−0.25 m) | **+0.053 ± 0.038 m** |

`emit_gate_obs` already does this and reads `frames.BORESIGHT` **live**, so a recalibration is
picked up automatically. The −0.25 m bake is a **metric translation**, *not* a camera tilt — an
angular form was empirically refuted (the bias is flat with range).

### Trap 2 — rank by apparent size, never by PnP range
Prior-free PnP **flips near-frontal close gates to ~25 m** (the IPPE 2-fold ambiguity). Ranking by
`range_m` would then call the nearest gate the farthest. `emit_gate_obs` ranks by
`apparent_size_px` (bbox area) — a pure image measurement, immune to the flip.

### Trap 3 — flips and partials are real on the map-free path
There is **no map prior** to break the IPPE 2-fold. Gate on confidence:
- `ambiguity_ratio` → 1.0 means the two PnP solutions are equally good ⇒ **distrust `rel_pos_body`**.
- `n_corners == 3` (cropped gate) ⇒ P3P, less constrained; `visible_area_ratio` and `inner_area_px`
  are **`None`** ⇒ mask them.
- `visible_area_ratio` is **range-free** and stays trustworthy exactly where PnP range does not.

Also: don't just take the highest-score detection as "the gate" — background gates score well. The
closest/second-closest ranking is what disambiguates.

---

## 5. Accuracy you can expect (measured)

Body-frame rel-pos residual vs ground truth, task2 given-pose bundle, M model, n=39, flip-resolved:

```
forward (depth)  -0.130 +- 0.274 m     <- depth is the weak axis (monocular)
right (lateral)  -0.003 +- 0.032 m     <- excellent
down (vertical)  +0.053 +- 0.038 m     <- after the -0.25 m bake
```

- Vertical + lateral (the **in-plane** axes that decide gate centring) are good to ~5 cm bias.
- **Depth is the weak axis** and its error grows fast with range (at 25 m a 1 px corner shift ≈ 1.4 m
  of range). Treat far-gate `range_m` as soft.
- Latency: **6.2 ms** median uncontended (engine). The **contended** in-flight number (sim rendering
  on the same GPU) is **not yet measured** — historically detect swung 32↔234 ms under contention.

---

## 6. What is NOT built (do not assume it)

- **No temporal tracker / no stable gate IDs.** Ranking is **per-frame**. "Closest" on frame *N* and
  frame *N+1* are not guaranteed to be the same physical gate (they usually are, but nothing enforces
  it). Persistent IDs = the unsolved carry-2 tracker (area-seed + ByteTrack + MAVLink gate index).
- **No map, no world pose.** The wire is pose-denied (only `HIGHRES_IMU` + actuators + video).
- **No fix filtering.** The live map-free path smooths with an EMA only; there is no χ²-gated KF on
  this bridge.

---

## 7. Deploy footguns

- **`.engine` / `.onnx` need `task="pose"` at load.** Otherwise ultralytics guesses `detect` and
  **silently drops every keypoint** (zero gates, no error). Handled in
  `racer.vision.detector._load_yolo_model`. **This fix is UNCOMMITTED on branch `vq2-rl-controller`** —
  if you are on another branch/clone, port it first or the engine returns nothing.
- **Engines are hardware + TRT-version bound** (built for RTX 2000 Ada `sm_89`, tensorrt 10.16.1.11).
  Rebuild per target: `scripts/export_trt_engine.py --weights <pt> --imgsz 384 640 --precision fp16`.
  Then verify with `scripts/bench_detector_engine.py` (must print **PARITY PASS**).
- **Never fly a square-640 or int8 engine** — both FAIL the parity gate.
- `fly_vq1.py` selects the detector with `--vision --weights <path>`. The legacy
  `--seeker-detector` flag defaults to the *classical* red-glow detector — not YOLO.

---

## 8. Uncommitted state (branch `vq2-rl-controller`)

These are on disk but **not committed**; a fresh clone will not have them:
`src/racer/contracts.py` (the `visible_area_ratio` / `inner_area_px` properties),
`src/racer/vision/detector.py` (the `task="pose"` engine-load fix),
`scripts/rl_gate_obs.py`, `scripts/export_trt_engine.py`, `scripts/bench_detector_engine.py`,
`scripts/characterize_perception.py` (`--vert-offset` flag).
The `.engine` / `.pt` files are gitignored by design (derived / large).
