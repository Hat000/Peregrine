# VQ2 Appearance Spec — "Now You See Me, Now You Don't"

Quantitative appearance the Blender scene must reproduce, extracted from the
**2026-06-29 load-day recon** (`handoff/vq2-recon-2026-06-29/`): 12 curated 640×360
camera frames + `logs/frame_stats_*.json` (3508 frames aggregated across the training,
training-active, and submission passes). Machine-readable mirror:
[`appearance_params.json`](appearance_params.json).

> **Governing principle.** The VQ2 sim is **deterministic and the look is FIXED**. The goal
> is to match **one** appearance with high fidelity, **not** to broadly domain-randomize.
> Randomize only camera **pose** (range/bearing/elevation/attitude) and small **sensor**
> noise (JPEG, mild blur/exposure). Do **not** randomize gate color, scene brightness, or
> the lighting setup away from the values below.

---

## 1. Camera / geometry (canonical — do not re-derive)

| field | value | note |
|---|---|---|
| resolution | 640 × 360 | |
| fx, fy | 320, 320 | |
| cx, cy | 320, 180 | |
| mount | +20° pitch **up** about body-Y | spec VADR-TS-002 §3.8, FIXED |
| HFoV | 90° | spec mislabels this "VFoV" |
| VFoV | ≈ 58.7° | true vertical FoV; **always use fy = 320** |
| distortion | none | |
| gate inner square | **1.5 m** | what PnP uses |
| gate outer square | 2.72 m | bbox only |
| corner order | **0=LL, 1=LR, 2=UR, 3=UL** (gate Y down) | IPPE_SQUARE; matches `racer.vision.gate_pose.gate_object_points` |

These mirror `racer.frames.CAMERA_INTRINSICS_K` exactly; the harness imports them rather
than re-typing, so a change there propagates.

## 2. The look (overall)

Very **low-light, high-contrast**. Scene **mean gray ≈ 36 / 255**, median 27, ~60 % of
pixels dark (< 32), ~1 % bright (> 224 — the point light sources). Per-channel means are
**near-neutral** (R ≈ G ≈ B ≈ 35–37): the red is **localized to the gate glow**, not a
global tint. Most of the frame is near-black structure (pillars, walls, parked drones,
ceiling trusses) punctuated by bright point sources.

- **Background / unlit:** RGB ≈ (10, 10, 10).
- **Dark structure (lit at all):** RGB ≈ (24, 26, 26).
- **Global ambient/world light:** very low, ≈ 0.04 of full. The scene is lit almost
  entirely by the **emissive** gates, ceiling light grids, and lane lines — tune world
  strength so an unlit floor patch reads gray ≈ 10–15 and the whole-frame mean lands ≈ 36.

## 3. The gate glow (the detection target — most important)

The gate is a **self-lit (emissive) red square frame** with a dark 1.5 m inner opening.

- **Emissive core color:** RGB ≈ **(255, 56, 16)** (saturated plateau), median (255, 65, 29)
  — a vivid **orange-red, NOT pure red**. Consistent with the VQ1 gate red (255, 50, 0), a
  touch warmer. Linear (for Blender) ≈ (1.0, 0.036, 0.003).
- **Extreme near range:** the core blows out toward (254, 59, 57) — as the sensor clips, G
  and B rise. Expect corner blow-out very close (RECON §2).
- **Strength:** start Blender emission strength ≈ 8 and tune so the rendered core saturates
  R ≥ 250 with a crisp ~2–3 px edge (calibrate against §4).
- **Ring/bar thickness:** ≈ 0.175 × outer side in image (recon frame 01: outer ~80 px →
  inner opening ~52 px).

## 4. Bloom / glare (the binding range-bias concern)

The glow has an **asymmetric bloom**, measured on a clean near gate
(`01_gate_deadahead_near_static_R1G7.png`), red scanline across the bar:

- **Saturated core (R ≥ 250):** a sharply bounded band.
- **Crisp edge:** falls off to background in **~2–3 px**.
- **Soft edge (opposing):** a glow halo extending **~12 px** (half-max ~18 px).
- **Ambient red wash:** a pervasive low-amplitude red haze over the upper scene
  (ceiling + multiple distant gates) ≈ RGB (46, 40, 32).

> **Binding implication (RECON §2).** Extract gate corners off the **saturated core
> (R ≥ 250)**, *not* a low red/brightness threshold. A low threshold grabs the ~12 px halo +
> ambient wash and **inflates the apparent square → biases PnP range NEAR**. The classical
> detector (`red_glow_detector.py`) already does this (`r_core = 250`); the Blender bloom
> must reproduce the strong-core / narrow-crisp-falloff / longer-soft-tail profile so the
> trained detector sees the same corner-span statistics. The **mock renderer** in this
> harness reproduces a *symmetric* approximation (crisp core + radial halo) — enough for the
> geometric round-trip check; the real Blender glare is asymmetric per §4.

## 5. Other scene elements (realism + classical-detector negatives)

| element | RGB | role |
|---|---|---|
| **Lane lines** (floor, blue/cyan, converging to next gate) | core ≈ (222, 198, 80), wash ≈ (124, 98, 43) | emissive floor strips; **track-specific guidance**, not a detection target. Blue clutter the red-dominance test rejects. |
| **Floor grid** | line ≈ (45, 45, 48) on floor ≈ (8, 8, 8) | faint regular grid, spacing ≈ **1.0 m** (square cells); non-emissive. Refine vs a top-down render once Blender is connected. |
| **Ceiling light grids** | ≈ (237, 238, 238) | bright **neutral-white** point-light grids on dark trusses — main non-gate bright source. |
| **Green pole markers** | ≈ (80, 252, 80) | emissive green poles flanking some gates (a few red ones too). |
| **Pillars / trusses / walls** | ≈ (22–26) | dark gray; white station-number signs 01–20. |

Stations are numbered **01–20**. One gate seen labeled **"R1-G7"**.

## 6. What to randomize vs. fix

| randomize (per render) | fix (match the recon) |
|---|---|
| camera pose: range, bearing, elevation, roll/pitch/yaw | gate emission color + strength |
| small sensor noise: JPEG q≈90, mild motion blur, ±exposure jitter | scene brightness / ambient level |
| which/how-many gates in view, distractor placement | bloom profile, lane-line/grid/structure colors |

Motion blur is real in the stream (RECON frames 10–12) — include a mild blur augmentation,
but the *base* appearance is the sharp low-light look above.

## 7. Provenance

All numbers above were sampled directly from the recon frames (per-pixel core/lane/grid/
structure segmentation) and the aggregated `frame_stats_*.json`. Re-extract with
`tools/blender_pipeline/extract_appearance.py` (re-runnable against the recon dir). The
spec is **track-agnostic in geometry** but **VQ2-specific in appearance** — this *is* the
"Now You See Me" look.
