# VQ2 gate hand-labeler (8-keypoint YOLO-pose)

Purpose-built browser tool for hand-labeling real VQ2 frames with the EXACT label format of
the synthetic dataset writer (`src/racer/vision/blender_gen/labels.py` / `contract.py`), so
hand labels and Blender labels can be mixed in one training set without silent poisoning.

- **Server**: stdlib-only Python (`http.server` + `json`) — no pip install needed.
- **Encoder**: `labelio.py` — pure, importable, unit-testable without the browser
  (`encode_label(gates, img_w, img_h) -> str`); the server calls it, and
  `python labelio.py` runs a self-check.
- **Frontend**: single `index.html`, canvas + vanilla JS, no CDNs. The 4-point homography
  (inner clicks → outer corners) is mirrored in JS from the same canonical gate-plane squares.

## Run

```
C:\Users\Shadow\vq2yolo-venv\Scripts\python.exe C:\Users\Shadow\Peregrine\tools\gate_labeler\serve.py --frames <DIR_OF_VQ2_PNG_JPG_FRAMES> --labels <OUT_LABEL_DIR> [--port 8000]
```

then open **http://localhost:8000**.

Example (real VQ2 frames pooled 2026-07-01 — 12 native-640×360 recon PNGs + 83 deduped
640×360 frames sampled from the a7 vision-run videos, 95 total):

```
C:\Users\Shadow\vq2yolo-venv\Scripts\python.exe C:\Users\Shadow\Peregrine\tools\gate_labeler\serve.py --frames C:\Users\Shadow\vq2_real_frames_2026-07-01 --labels C:\Users\Shadow\vq2_real_frames_2026-07-01\labels --port 8000
```

Labels land in `--labels` as one `.txt` per image, **same basename** (`frame_0042.png` →
`frame_0042.txt`). Negative frames get an intentionally **empty** `.txt` (never omitted).
Re-opening a labeled frame loads its label back onto the canvas for editing (resume).

## Detector seeding (`--seeds`) — label by CORRECTING, not clicking

```
<py> serve.py --frames <FRAMES> --labels <LABELS> --seeds <SEEDS> --port 8000
```

Generate the seeds first with the 8-keypoint model:

```
.venv\Scripts\python.exe scripts\seed_labels_with_m.py --frames <FRAMES> --seeds <SEEDS>
```

**Why.** Step 3 below derives the outer corners from the inner ones via the plane homography. That
derivation is projectively **exact** (verified to 2e-11 px against true projection), but it
**extrapolates**: the outer square is 1.813× the inner, so a click error is amplified ~3× for an
in-frame corner and **~6–9× when an inner corner is off-frame and placed by eye**. Two of the four
derived outers typically still land in-frame, where they are stored `v=2` with those amplified
coordinates — silent label poisoning. **M predicts the outer corners DIRECTLY**, so seeding removes
the extrapolation entirely. (Seeding deliberately SKIPS 3-corner/partial detections — seeding those
would require the very extrapolation this avoids.)

**Safety invariant.** A file in `--labels` means **a human approved it**. Seeds live in a separate
dir (the server refuses `--seeds == --labels`), are served only when no human label exists, and are
flagged `seed: true` — the frame stays **UNLABELED** and the prompt turns into a loud
`SEEDED by detector (UNVERIFIED)` banner until you press save. Machine output can never masquerade
as a hand label. Saving writes a real label and clears the seed state.

Seeded frames open with all 8 points already placed: check every corner, drag what is wrong, save.

## Click workflow

1. The tool opens at the first unlabeled frame; guided clicking is auto-armed if the frame
   has no gates yet.
2. Zoom in (mouse wheel, zoom is cursor-centered) and click the **4 INNER corners** in the
   guided order **LL → LR → UR → UL** (as seen head-on; the toolbar prompt shows which
   corner is next). Gate frame is X-right, **Y-DOWN** — LL = left-and-down corner.
3. After the 4th click the **4 OUTER corners are auto-derived** via the plane homography
   (inner and outer squares are concentric + coplanar: inner 1.5 m, outer 2.72 m) and drawn
   dashed. **Horizon-guarded:** deriving the outer square EXTRAPOLATES 1.813×, and on a
   near-edge-on gate that can reach the gate plane's vanishing line — the homogeneous `w` → 0
   and then goes negative, so the corner has no valid pixel and *wraps* to the opposite side of
   the image (a real case produced an outer corner at `(12067, −10508)`). Such a corner is now
   detected and pushed off-frame so it encodes **v=0** instead of being stored as a visible
   keypoint at a meaningless position. Valid geometry is bit-unchanged (still exact to 1e-13 px).
   If you see an outer corner parked far outside the frame on a steeply-angled gate, that is the
   guard working — the real frame corner is behind the camera plane and genuinely unlabelable.
4. **Drag any of the 8 points** to fine-adjust. Dragging an inner point live-recomputes the
   outer square; dragging an outer point detaches it (independent nudge — later inner drags
   won't clobber it). `recompute outer` / `R` re-derives all 4 outers from the inner square.
5. Multiple gates: `+ add gate` (`G`) and repeat. One label row per gate.
6. No gate visible: `NO GATE (negative)` (`N`) — saves an empty label file immediately.
7. `save` (`S`), then `next` (`D`/`→`).

Point rendering: **filled** = visible (v=2), **orange** = marked occluded (v=1),
**hollow** = off-frame (v=0, automatic from position; drag a corner outside the image border
and it is stored clamped with v=0 — do NOT delete the gate just because a corner left frame).

## Keyboard shortcuts

| Key | Action |
|---|---|
| `D` / `→` | next frame |
| `A` / `←` | prev frame |
| `S` / `Ctrl+S` | save |
| `K` | skip (next without saving) |
| `N` | NO GATE — save empty label (negative) |
| `G` | add gate (start guided 4-corner clicking) |
| `X` / `Del` | delete selected gate |
| `R` | recompute outer corners from inner (clears detach) |
| `O` | toggle occluded (v=1) on the selected point |
| `Z` / `Backspace` | undo last inner click |
| `Esc` | cancel in-progress gate / deselect |
| `0` | fit view |
| wheel | zoom (cursor-centered) |
| `Space`+drag, middle/right drag | pan |

## Output format (frozen — see labelio.py header)

One row per gate: `class cx cy w h (x y v)*8` = 29 space-separated fields, floats `%.6g`.
Keypoints 0–3 = INNER LL,LR,UR,UL; 4–7 = OUTER LL,LR,UR,UL. x normalized by **640**, y by
**360** (keypoints clamped to [0,1] after normalization); bbox = outer-square pixel extent
clamped to the frame. Visibility: 2 visible / 1 occluded-in-frame / 0 off-frame (in-frame
test uses inclusive bounds 0..639 / 0..359). Compatible with `kpt_shape: [8,3]`,
`flip_idx: [1, 0, 3, 2, 5, 4, 7, 6]` (contract.py `DATA_YAML`).

Frames are expected to be 640×360 (VQ2 contract). Other sizes still normalize correctly (by
actual size), but the server prints a loud warning — check your frame source.

## Unit-testing the encoder

```
C:\Users\Shadow\vq2yolo-venv\Scripts\python.exe C:\Users\Shadow\Peregrine\tools\gate_labeler\labelio.py
```

or import `labelio.encode_label` / `labelio.derive_outer_from_inner` directly in tests.
