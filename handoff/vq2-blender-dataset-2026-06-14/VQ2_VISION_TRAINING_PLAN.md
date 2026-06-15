# VQ2 vision training-data plan — what to render, and why

**Question:** what does a photoreal drone-racing sim look like, and what appearance distribution
do we render in Blender to get the *best* training material for the VQ2 gate detector?

**The honest answer is a domain-design problem under uncertainty.** We don't have the VQ2 photoreal
sim or the Round-2 3D-scan yet. So the plan is: render *faithfully* what we know, randomize *widely*
what we don't — biased toward the *plausible real*, not uniform chaos — and build the feedback loop
that closes the gap the moment any real reference appears. This document is grounded in the
competition intel + the winning prior art (MonoRace), not aesthetics.

---

## 0. The thesis (and the one finding that reframes everything)

The near-1:1 prior art — **MonoRace** (TU Delft MAVLab, arXiv 2601.15222), which **won A2RL 2025**
(same operator DCL, same monocular+IMU format, the *same* 1.5 m gate, beating 3 human world
champions at ~100 km/h) — **did not use photoreal 3D rendering at all.** It trained its gate detector
on **2D composites** (gate-face cut-outs warped + pasted onto real backgrounds) with **aggressive 2D
augmentation** (HSV shifts, Gaussian noise, affine, and explicit **5–15 px directional motion blur**),
at a **3500 synthetic : 500 real** split, and **deliberately decoupled detection from color** (its
gates were orange-LED, but the net keys on **edges/shape**, never color). It found enough corners for
PnP in only **~59 % of frames** and that sufficed to win.

This does **not** kill our Blender effort — VQ2 is a real 3D-scanned environment, and a keypoint
detector needs **geometrically-exact gate perspective across the full pose space**, which flat
composites approximate crudely. But it **reprioritizes**:

> **Photorealism is NOT the dominant sim-to-real lever. Appearance *diversity*, motion blur, a
> geometry-keyed label, and hard negatives are.** Spend Blender on **correct gate geometry across a
> wide pose distribution + cheap appearance randomization**, not on expensive material/GI fidelity.

A 2025 YOLOv11 sim-to-real study (arXiv 2509.15045) independently confirms the direction: *dataset
diversity (varied viewpoints + ~35 % negative examples + color/lighting randomization) beat any
single augmentation*, and PencilNet / Deep-Drone-Racing achieved **zero-shot** sim-to-real gate
perception from domain randomization alone.

---

## 1. What we actually know about the target (grounded, cited)

| Fact | Source | Consequence for what we render |
|---|---|---|
| **Environment is INDOOR.** Sept-2026 SoCal qual + Nov-2026 Columbus final are both indoor arenas; A2RL Abu Dhabi was an indoor hall. Round-2 = "real 3D-scanned environment." | competition intel; Anduril/TechCrunch | Bias HDRI/lighting to **indoor arena / warehouse**; outdoor is a robustness minority, not the default. |
| **The real gate is a SOLID structural frame, NOT an LED light.** The A2RL×DCL autonomous gate (same 1.5 m/2.7 m spec, same operator) is a **purple** painted/branded rectangular frame, "a solid structural frame rather than illuminated or LED-based." | ESA image caption; MonoRace | Gate material = **matte-to-semi-gloss painted frame**; emissive is a small insurance minority, not primary. |
| **The orange LEDs are on the DRONE, not the gate.** | MonoRace; multiple | The VQ1 sim's orange-red is **operator styling**, not a measured gate property. |
| **Gate COLOR is the single biggest unknown** (purple real vs orange-red sim; DCL branding spans purple/blue/orange). | ESA; DCL | **Randomize color hard.** Treat it as a nuisance axis, never a feature. |
| **Lighting is a DARK arena with bright (often colored) spotlights**, high dynamic range, black-curtain backdrops, industrial ceilings. | DCL/A2RL coverage | Dominant lighting mode = **low ambient + hard spotlights + HDR**, not uniform daylight. |
| **Camera:** 640×360, fx=fy=320, **+20° up-tilt**, **pinhole no distortion**, 30 Hz, **JPEG-over-UDP**, flown up to ~30 m/s. | spec VADR-TS-002 §3.8 | Render at exactly this; **no lens distortion**; JPEG + motion blur + exposure are the always-present sensor gap. |
| **Nov final: possible spectator camera flashes.** | spec/context | Include transient **overexposure/flash** frames. |
| **Gate is thick (0.26 m deep) + branded with logos/text.** | spec; ESA | Model the **deep frame** (near/far-face parallax at close range) + randomized **decals**. |

---

## 2. The appearance axes — the design

For each axis: the *plausible real* → the *strategy* → the *Blender mechanism* → *where it lives in
our code*. Ordered by **training value per unit effort** (highest first).

### A. Gate color + material  — *the flagged unknown*
- **Real:** solid painted frame; purple precedent; could be any color in VQ2; matte-to-semi-gloss.
- **Strategy:** color is a **nuisance variable**. Sample base hue across the **full wheel** (anchor a
  minority at the VQ1 orange-red and the DCL purple, but the bulk is broad), saturation low→high
  incl. **desaturated white/grey/black**. Material mostly **matte/semi-gloss** (roughness 0.3–0.9,
  metallic 0–0.4); a **glossy/metallic minority** (specular highlights from spotlights); a **small
  emissive-rim minority** as insurance against an LED-trimmed gate; randomized **branding decals**,
  frame depth, paint wear.
- **Blender:** Principled BSDF base-color HSV sampling; Emission Color/Strength via an *Is-Camera-Ray*
  mix for the rare glow; decals via an image/text node on the frame face.
- **Code:** `AppearanceConfig.gate_*` + `bpy_materials.gate_material`. **Done** (broad hue, sat/val,
  metallic/roughness, emission prob). *To add:* explicit purple anchor + decal randomization.

### B. Lighting / world  — *the dominant realism lever*
- **Real:** dark indoor arena, bright hard spotlights (white **and** colored), HDR, black backdrops;
  some outdoor for robustness.
- **Strategy:** **dark-arena dominant.** Low world/ambient strength + 1–3 hard **spot** lights at
  random positions/colors/intensities casting real shadows + specular hotspots; an **indoor-arena
  HDRI** library (Poly Haven CC0) as the principal world; outdoor day/overcast/sunset as a minority.
  **Over-sample the indoor/neutral end** of the HDRI distribution.
- **Blender:** World = Environment Texture (random HDRI, randomized Z-rotation + Strength); `SUN`/
  `AREA`/`SPOT` lights with blackbody + colored tints; high dynamic range.
- **Code:** `AppearanceConfig.{sun_*, color_temp, use_hdri, hdri_dir}` + `bpy_materials.{add_lighting,
  setup_world}`. **Done** for sun + HDRI + procedural sky. *To add:* a **dark-arena / colored-spotlight
  mode** + a **Poly Haven HDRI fetcher** biased to indoor.

### C. Sensor / camera pipeline  — *the always-present gap (proven lever)*
- **Real:** JPEG-over-UDP artifacts, motion blur at speed, auto-exposure hunting (over/under),
  ISO/sensor noise, mild vignette/chromatic aberration, bloom/flare off bright lights, occasional flash.
- **Strategy:** this is MonoRace's **proven** lever — apply it **aggressively** as a 2D post-render
  pass on top of the Cycles image (cheaper + more controllable than rendering it): **directional
  motion blur** (kernel scaled to our 30 Hz × up-to-30 m/s — sweep ~5–25 px, biased to the projected
  velocity direction), **JPEG re-encode** at random quality, Poisson/Gaussian + ISO noise,
  brightness/exposure/white-balance jitter, **vignette + mild chromatic aberration**, a **transient
  flash/overexposure** channel. **No lens distortion** (camera is distortion-free).
- **Blender + post:** Cycles motion blur (incl. **rotational** — keyframe camera *rotation* from body
  rates, not just translation) + compositor **Glare** (Fog-Glow bloom, Streaks/Ghosts for LED flare);
  then the **albumentations** post-pipeline for the 2D sensor effects.
- **Code:** `augment.py` (keypoint-aware albumentations: motion blur, JPEG, noise, brightness, defocus,
  dropout, shadow, sunflare) + `bpy_render.{set_motion_blur, _build_glare_compositor}`. **Done** for
  translational blur + glare + the 2D pipeline. *To add:* **rotational** motion-blur keyframes,
  vignette/chromatic-aberration + an explicit **flash** aug.

### D. Backgrounds / environment + HARD NEGATIVES  — *first-class, currently under-weighted*
- **Real:** dark cluttered indoor — black curtains, industrial ceilings + fixtures, structure/trusses,
  other gates, floor reflections; confuser rectangles (doorways, screens, banners).
- **Strategy:** **two things the evidence demands and we under-do today:** (1) varied cluttered indoor
  backdrops incl. **confuser structures** (other rectangular openings) so the net keys on the *gate
  opening geometry*, not "any bright rectangle"; (2) **~30–40 % HARD NEGATIVES** — frames with **no
  gate** (and confusers only). The YOLOv11 study's best dataset was ~35 % negatives.
- **Blender:** floor/walls/ceiling + randomized clutter boxes (`bpy_scene.build_background`); real
  arena photos as backplates is a cheap realism arm.
- **Code:** `bpy_scene.build_background` + procedural backend clutter. **Partial.** *To add (high
  value, pure-Python, testable):* a **negative-frame fraction** in the dataset writer + confuser props.

### E. Geometry / pose distribution  — *already solved, keep it faithful*
- Viewpoints sampled on the **real course** (`navigator.gates_from_track_records`) — true range
  distribution, the 26 m descent, gate facing, down-course co-visibility, 3-corner clips at terminal
  range under the +20° tilt. Corners projected through `frames.py` → labels are **PnP-exact** (tested
  < 1e-9 px; live ≤ 1 px self-check on ShadowPC). **This is the part photoreal rendering buys us over
  2D composites** — keep it exact. **Done.**

---

## 3. Dataset composition — the recipe

A **mix**, not one giant uniform set. Recommended target proportions (ablate later):

| Arm | Preset | ~Share | Why |
|---|---|---:|---|
| Appearance-broad (indoor, dark-arena, wide color) | `appearance_broad` (retuned) | 45 % | the VQ2 generalization bulk |
| Hard visual (flares, blur, occlusion, low light, flash) | `hard_visual` | 20 % | the robustness tail |
| **Hard negatives + confusers (no gate)** | `negatives` *(new)* | **25 %** | the evidence's biggest under-used lever |
| Terminal approach (2–8 m, 3-corner clips) | `terminal_approach` | 5 % | margin-critical terminal gate-lock |
| Long range (15–30 m, small target) | `long_range` | 3 % | co-visibility tail |
| VQ1-faithful (orange-red, arena) | `vq1_faithful` | 2 % | regression anchor only |

Plus, if obtainable, **~10–15 % real frames** (MonoRace's 500/4000 ≈ 12.5 %): the 40 real VQ1 frames
we already have, ShadowPC captures, or renders of the actual scanned mesh once released. Even a small
real fraction is disproportionately valuable.

Scale: a few × 10⁴ images total is plenty given the diversity-over-volume finding; spend the GPU on
*coverage*, not sample count per scene.

---

## 4. Validation — make the appearance distribution falsifiable

1. **Label-correctness gate (already built):** intrinsics self-check ≤ 1 px + the viz overlay
   (keypoints must land on the rendered inner corners + PnP recovers range). Non-negotiable per run.
2. **Sim-to-real eval, available NOW:** we hold **40 real VQ1 sim frames** (`task2_frames`, 1.8–23 m).
   Train on synthetic → measure detector corner error / PnP range on those real frames. This is the
   crown metric and it exists today; re-run it whenever the distribution changes.
3. **Ablate one axis at a time** (HDRI, material, color, blur, glare, noise, JPEG, negatives) against
   the real-frame eval — the literature says only *some* knobs move sim-to-real; find which, drop the
   rest. Don't pay for fidelity that doesn't transfer.
4. When the **official VQ2 sim / Round-2 scan** drops: render a held-out eval set in *its* look and
   re-weight the distribution toward it (especially HDRI + gate color).

---

## 5. The feedback loop (this is how the gap actually closes)

- **Auto-label flywheel:** once any VQ2 sim/real frames exist, the **known gate geometry** lets us
  auto-label them (PnP from the 1.5 m square) → fine-tune. MonoRace's 3500:500 mix is exactly this.
- **Offline boresight calibration (high value, already half-built):** MonoRace's single biggest
  *accuracy* lever was **offline extrinsic refinement** — fly with a conservative net, then optimize
  camera angles by maximizing IoU between re-projected and observed gate masks (it hit sub-degree:
  pitch 0.14° / roll 0.49° / yaw 0.71° in ~40 steps). **This is precisely our banked ε_vert ≈ −0.25 m
  boresight problem** — and our `frames.BoresightCorrection` mechanism already exists to apply the
  result. The dataset and this calibration are *complementary, independent* deploy-time tools.

---

## 6. Build sequence (priority order)

1. **[done]** Geometrically-exact generator + PnP-correct labels + the 5 presets + viz + laptop tests
   + the bpy Cycles backend (camera/scene/materials/render). Runs end-to-end procedurally on the
   laptop; photoreal on ShadowPC.
2. **[next, cheap, pure-Python, testable]** Hard **negatives + confusers** (25 % of the mix) and the
   **dark-arena / colored-spotlight** lighting mode + **broad-color** retune of the default presets.
3. **[ShadowPC, Blender-side]** Poly Haven **indoor-biased HDRI fetcher**; **rotational** motion-blur
   keyframes; compositor **vignette + chromatic aberration + flash**; decal randomization.
4. **[validation]** Wire the **sim-to-real eval on the 40 real frames** into the training harness; run
   the **per-axis ablation**.
5. **[when VQ2 drops]** Re-weight to the real look; stand up the **auto-label flywheel** + the
   **offline boresight IoU calibration** (reuses `frames.BoresightCorrection`).

---

## 7. Open decisions (flagged for Fengyou / the commander)

1. **Detector architecture — keypoint vs. segmentation.** We train **YOLO-pose 4-corner keypoints**;
   MonoRace **segments then extracts corners geometrically** (LSD line-intersection on the mask),
   which is more robust to the appearance gap (corners come from straight *edges*, not pixel
   appearance). Our generator can **cheaply also emit segmentation masks** (the gate ring polygon is
   known), so we can hedge: keep YOLO-pose, optionally add a mask head or a post-hoc LSD refinement.
   **This is a vision-stack-level call, not a dataset call** — surfacing it, not blocking on it.
2. **Real frames.** Confirm we can use the 40 VQ1 frames (+ any ShadowPC captures) for the
   real-fraction + the sim-to-real eval. Do we have rights/route to the scanned-mesh renders later?
3. **Color stance.** Recommend treating gate color as a **pure nuisance axis** (broad randomization,
   no orange-red prior). Confirm — this is a deliberate departure from the VQ1 look.
4. **Compute budget** for the dataset (frames × Cycles samples) — drives EEVEE-preview vs Cycles-final
   and the total scale.

---

## TL;DR
Render **geometry faithfully, appearance widely-but-plausibly** (dark indoor arena, solid painted
frame, color as a nuisance axis), lean on **aggressive 2D augmentation + motion blur** (the proven
lever), add **~30 % hard negatives**, validate against the **real frames we already have**, and stand
up the **auto-label + offline-boresight feedback loop** for when the real VQ2 look lands. Photorealism
is the *base*, not the *goal* — diversity and geometry win.
