# VQ2 Vision-Detector — Commander Report

**Session:** P5 Vision-Detector worker — VQ2 Blender render: validate → photoreal production → training harness
**Machine:** ShadowPC (Blender 5.1.2, Python 3.13, NVIDIA RTX 2000 Ada)
**Branch:** `vq2-blender-render-2026-06-15` (10 commits, local — not yet pushed)
**Date:** 2026-06-15

---

## BOTTOM LINE

The VQ2 photoreal dataset generator is **render-verified, productionized, and producing correctly-labeled
data**, and a **smoke-validated training harness** is ready for Adroit. The bpy/Cycles backend — previously
DOC-VERIFIED but NEVER RENDER-VERIFIED — now renders photoreal frames whose **labels are provably correct**.
A **2000-image base dataset** is on disk. The detector can be trained as soon as the data is transferred to
the cluster.

| Deliverable | Status |
|---|---|
| BlenderBackend renders verified-correct labels | ✅ intrinsics **0.0000 px**, keypoints on corners, PnP recovers range |
| Photoreal look (HDRI + PBR floor + real props/people) | ✅ signed off by stakeholder |
| Gate dimensions true to spec | ✅ inner 1.5 / outer 2.72 / depth 0.26 m, proven to the pixel |
| 8-keypoint labels (inner + outer corners) | ✅ emitted + validated (0 malformed of 600 sampled) |
| 2000-image base dataset (all-hue + VQ1-red) | ✅ rendered, masks, clean (un-augmented) |
| Training harness (Ultralytics 8-kpt pose) | ✅ smoke-trained end-to-end on ShadowPC |
| Adroit training run | ⏳ ready to launch (needs data transfer) |

---

## WHAT WAS DELIVERED

**1. Render-verification (the original de-risk).** The bpy camera reproduces `K` to **0.000036 px**; the 4
inner keypoints land exactly on the rendered gate opening; PnP recovers the known range. Three real bugs
were found *by actually rendering* (invisible to the laptop suite, which never imports the bpy leaves):
wrong relative imports in `bpy_camera` (backend couldn't load), the 5.x EEVEE engine-id rename, and the 5.x
compositor node removal.

**2. Photoreal production pipeline** (`bpy_photoreal.py` + `assets.py`, behind `appearance.photoreal`).
Real CC0 Polyhaven assets: HDRI environments (image-based lighting + photographic backgrounds), PBR floor
**below** the flight line (gates float in the air), photoscanned props + procedural clothing-tinted
mannequin people scattered **off** the gate corridor, solid vivid gate. Clean renders — camera artifacts
are deferred to training-time augmentation. The verified label path was never touched.

**3. Eight-keypoint labels.** Per stakeholder request, labels now carry inner 0–3 (gate opening / PnP) **and**
outer 4–7 (gate frame): `kpt_shape [8,3]`, `flip_idx [1,0,3,2,5,4,7,6]`. Inner 4 are byte-identical to the
old block, so PnP is unchanged. **This forks the keypoint contract from the 4-kpt synthetic (VQ1) writer.**

**4. The 2000-image base dataset** (clean / un-augmented, 8-kpt + masks):
`handoff/vq2-blender-render-2026-06-15/sets/` (local, gitignored)
- **all-hue** (`appearance_broad`) 1400 — generalization driver, 1120 pos / 280 neg (20%)
- **VQ1-red** (`vq1_faithful`) 600 — orange-red anchor

**5. Training harness** (`cluster/`, smoke-validated on ultralytics 8.4.68):
- `vq2_pose_dataset.py` → merged `data.yaml` + 90/10 split (1800 train / 200 val)
- `vq2_pose_train.py` → Python-API trainer, **on-the-fly augmentation**
- `vq2_pose_train.sbatch` → Adroit wrapper

---

## KEY DECISIONS (stakeholder-directed)

- **Photorealism via real assets**, not procedural — the first procedural look was rejected.
- **Gate kept dimensionally TRUE** (outer 2.72 m matches the bbox label) — a thinner "look" was rejected as
  unsafe for a training set. Gates **float** (not mounted), solid color, no branding decals.
- **Two color regimes**: all-hue (kill the gambling-red overfit) + VQ1-red (faithful anchor).
- **Augmentation: on-the-fly in Ultralytics**, not offline pre-multiply (researched). Engine = **AlbumentationsX**
  (maintained fork, AGPL — fine for research). Split of duties: **geometric** → Ultralytics built-in
  (pose-aware, uses `flip_idx`); **photometric/sensor** → AlbumentationsX hook (keypoint-invariant, so safe).

---

## VALIDATION EVIDENCE

- Intrinsics self-check **0.0000 px** on every production render (aborts >1 px — cannot emit wrong labels).
- Gate dims: projected inner/outer corner widths == `fx·size/range` to the pixel at 3/5/10 m
  (proof overlays: `preview/21_verify_dims_*`).
- 8-kpt labels: 0 malformed of 600 sampled; inner+outer land on the gate edges (`preview/22_8keypoint_*`).
- Overlay + PnP across both sets at scale: keypoints track, ranges sane (2.6–41 m).
- **Smoke train** (yolo11n-pose, 1 epoch, ShadowPC): 16 imgs / 4 backgrounds / **0 corrupt**, 8-kpt head
  adapts, AlbumentationsX sensor list loads, all pose losses compute.
- Laptop test suite: **31 passed**.

---

## THROUGHPUT

Cycles GPU (RTX 2000 Ada), 640×360, 160 samples + OptiX denoise: **~1400–1500 frames/hr (~2 s/frame)**.
Prop glTF models are imported once and spawned as mesh-sharing duplicates (a ~25 s/frame trap, fixed). The
occlusion raycast counts only prop/person hits, never a gate hitting its own frame (a ~90% frame-drop, fixed).

---

## TO LAUNCH TRAINING (Adroit)

1. Transfer `sets/{allhue,vq1red}/` to Adroit (`cluster/push_dir.py`) → `data/vq2_sets/`.
2. `sbatch cluster/vq2_pose_train.sbatch` — builds the split + trains yolo11s-pose.
3. **Confirm** the cluster `yolo` env has **ultralytics ≥ 8.3** (for the `augmentations=` kwarg; the sbatch pins it).

---

## PARKED / RISKS (surface up)

- **People are procedural mannequins** — no CC0 humans exist on Polyhaven. Fine at distance; convincing
  close-up people would need a different (likely paid) asset source.
- **Outer-corner visibility is in-frame-only** (the inner PnP corners carry the full prop-occlusion test).
- **8-kpt contract fork**: the VQ2 detector head is now 8-keypoint; the deployed 4-kpt PnP reads the inner
  block unchanged, but anything training across BOTH the synthetic (VQ1, 4-kpt) and VQ2 (8-kpt) writers must
  reconcile the kpt_shape.
- **Offline albumentations multiply** was the stakeholder's earlier instinct; superseded by on-the-fly (the
  base is rendered clean specifically to support either path).
- **40-frame sim-to-real eval** (from the original plan) not yet run — that is the real verdict on transfer.

---

## ARTIFACTS

- Code: `src/racer/vision/blender_gen/` (assets.py, bpy_photoreal.py, masks.py, viz_overlay.py + leaves);
  `cluster/vq2_pose_*` (training harness).
- Docs: `src/racer/vision/blender_gen/RUN_GUIDE.md` (setup + §7 photoreal + asset fetch), this report,
  `REPORT.md` (detailed working log).
- Datasets + assets: local on ShadowPC, gitignored (`assets_vq2/`, `handoff/.../sets/`).
- Visual proof: `handoff/vq2-blender-render-2026-06-15/preview/` (01–26: look evolution, dim proof, 8-kpt).
- Commits: 10 on `vq2-blender-render-2026-06-15` (`e3857eb` … `71e6b9b`).
