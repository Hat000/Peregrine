# Overnight Goal — Dethrone the Round-1 Champion

**To:** an autonomous goal-driven agent (Opus 4.8, runs unattended through the night)
**Goal type:** `/goal` — pursue one objective autonomously for hours, surviving repeated infrastructure death.

---

## 0. THE GOAL (one sentence)

Train a VQ2 8-keypoint gate detector that **beats the Round-1 champion on every metric of the standardized eval, with statistical confidence** — and if you find it, crown it. Run all night; the ShadowPC will drop and kill your job repeatedly — survive it, resume, and keep grinding.

**Be honest, not hopeful.** The champion is a stubborn local optimum — *every* lever we tried this week (more data, more aug, less aug, tighter loss) **lost**. Your edge is the handful of genuinely *untried* levers in §4. A real win clears the confidence interval; a +2% on 40 frames is noise. Do not fake a win.

---

## 1. THE BAR — champion = `runs/pose/runs/vq2_pose_8kp/weights/best.pt`

Run `eval_goodfix.py` (40 `task2_frames` + 204 `course_bundle`). Champion:

| metric | champion | your target |
|---|---|---|
| good-fix <0.5 m (oracle PnP) | **80%** [65-90] | **> 80%**, CI margin |
| good-fix <1 m | 95% [83-99] | ≥ 95% (near ceiling) |
| good-fix <2 m | 98% [87-100] | ≥ 98% (near ceiling) |
| **course valid-fix (204)** | **76%** | **> 76%** ← tightest CI, most credible win |
| good-fix <0.5 m (deployed IPPE PnP) | 62% [47-76] | do not regress |

**"Beat" = strictly higher `<0.5 m` good-fix AND higher course valid-fix AND no regression on the rest, with the win surviving the Wilson CI.** Because 40 frames is thin (±~12 pp), the **204-frame course metric is your most statistically-powerful target** — a clean win there counts most. `<1 m`/`<2 m` are near-ceiling; matching them is success.

---

## 2. SANITY GATE — reproduce the champion FIRST (do not skip)

The training pipeline has *silently degraded* before (wrong interpreter, stale package, split bug). Before trusting a single experiment, retrain the champion recipe on its exact data and confirm it lands ~80%:

```
& <venv> cluster/vq2_pose_train.py --data <sets>/data_base2000.yaml --batch 16 --device 0 \
    --project <runs> --name repro_sanity
# expect: best epoch ~26, early-stop ~46, good-fix <0.5m ~80% on eval_goodfix.py
```

If it does **not** reproduce ~80% → **STOP and report**. The pipeline is broken; do not waste the night on variants of a broken baseline.

---

## 3. DEAD LEVERS — do NOT repeat (we burned a full day proving these)

- **Remove / lighten the sensor aug** → **22%**. The VQ stream is chunked JPEG-over-UDP (spec §4.6); the flat-9 aug is *load-bearing* and already at the sweet spot (none 22 / flat-9 80 / heavy 55). Leave it.
- **Add more clean data** → caps **~62%, never 80%** (clean test, *visually-identical* images, correct split — it dilutes precision). Do not add data unless you first *prove and fix* a label-precision bug in the grown renders (§4.6).
- **Tighten OKS σ on the poisoned 4k set** → 32% — but that was **confounded by poison**, so σ-on-*clean* is a LIVE lever (§4.2).
- **Select checkpoints by val mAP** → mAP diverges from good-fix (a 0.835-mAP run scored 28% good-fix). Always select by good-fix.

---

## 4. LIVE LEVERS — prioritized, highest expected-value first

Run them **one at a time** on the champion's exact data/recipe, changing only the named thing.

1. **Capacity — `yolo11m-pose`, then `yolo11l-pose`.** *Your best shot.* Latency is a non-issue (VQ eval is an 8 GB-GPU desktop; 15-25 ms ≪ 50 ms budget), so a bigger model is free precision headroom — and it's **untested** (both champions are `s`). Just `--model yolo11m-pose.pt`.
2. **Loss — σ-on-clean + absolute-pixel L1.** Re-test a *tighter inner-4 OKS σ* (the precision corners) **on clean data**, and/or add an un-normalized L1 keypoint term to fix the area-norm near-gate blind spot (85% of <4 m gates are imprecise). Patch reference exists in `cluster/vq2_pose_train_precision.py` (`apply_precision_loss_patch`). One change at a time; watch for gradient saturation (the failure mode that killed the poisoned σ run).
3. **Inner-4-only head.** Outer corners 4-7 clip off-frame at near range → degenerate targets on the shared pose head that may dilute the precision-critical inner corners. Train an inner-4 head, compare.
4. **Hyperparameter tuning around the champion** — lr0/lrf, `close_mosaic`, optimizer, warmup. Small, single-variable.
5. **Test-time augmentation / checkpoint ensembling** — can lift good-fix at deploy with no retrain (cheap, do between training runs).
6. **(deep, conditional) Data-growth rescue.** Overlay the *grown-render keypoint labels* (`viz_overlay`) vs the base. **If and only if** the grown labels are sub-pixel-off vs `allhue`/`vq1red`, the render/label is the bug — fix it and re-grow (more data would then help). Don't chase this blind; the images look fine, so it's labels or nothing.

---

## 5. METHOD — non-negotiable (this is how we stopped fooling ourselves)

- **ONE variable per experiment.** Multi-change runs are unattributable — that confound cost us a day.
- **Deterministic, seed 0** — same recipe+data+env is bit-identical; use it to attribute and to detect pipeline drift.
- **Judge EVERY run on `eval_goodfix.py`** (oracle + deployed + CI), never mAP. **Select the deployed checkpoint by good-fix across the `save_period` snapshots**, not `best.pt` (mAP-best ≠ good-fix-best).
- **Keep a live leaderboard** (markdown table, updated after every run): experiment · the one change · good-fix oracle/deployed/course + CI · verdict vs champion. So the night's work is never lost to a closeout.
- **A win must clear the CI.** Prefer the 204-frame course metric for confidence; on the 40-frame <0.5 m, only a large margin is real.

---

## 6. OVERNIGHT SURVIVAL — the ShadowPC closeout footgun (CRITICAL — read twice)

**The ShadowPC powers OFF on connection drop and kills the running job. This WILL happen repeatedly tonight — it already killed two runs today.** If you don't engineer around it, you'll wake to zero progress.

- Every training run: **`save_period=10`** + ultralytics writes `last.pt` every epoch.
- The trainer (`cluster/vq2_pose_train.py`) has **no `--resume`** — **add one first** (`YOLO(last.pt).train(resume=True)`, re-passing the `augmentations=` hook) **or** wrap every run in a **watchdog loop**: detect the dead python process → relaunch from `last.pt` with resume. Reference: `cluster/vq2_pose_train_r1on4k.py` already implements `--resume`.
- **On any death: RESUME from `last.pt`. Never restart from scratch** (you lose hours).
- **One GPU → serialize.** Never two trainings at once. Check `nvidia-smi` + python procs before launching.
- After a closeout the machine returns idle (GPU ~700 MiB, 0 python) — your cue to resume the interrupted run, then continue the queue.

---

## 7. ENVIRONMENT + WHERE THINGS LIVE

- **Interpreter (USE THIS):** `C:\Users\Shadow\vq2yolo-venv\Scripts\python.exe` (torch 2.6.0+cu124, ultralytics 8.4.69, albumentations 2.0.8). **NOT** the base system python (CPU/partly-clobbered — it cost us a day).
- **Trainer:** `cluster/vq2_pose_train.py` (flat-9 champion recipe, batch-16 default, `save_period`). **Eval:** `handoff/vq2-train-shadowpc-2026-06-16/eval_goodfix.py`.
- **Champion data (FROZEN — never regenerate):** `…/sets/data_base2000.yaml` → `train_base2000.txt` (the exact 1800/200). Splitter (fixed per-set RNG, only for *new* splits): `cluster/vq2_pose_dataset.py`.
- **Champion weights (NEVER overwrite):** `runs/pose/runs/vq2_pose_8kp/weights/best.pt`. Data root: `C:\Users\Shadow\Peregrine-vq2data\handoff\vq2-blender-render-2026-06-15\sets\`. Test sets: `handoff/shadowpc-followups-2026-06-05/task2_frames/` (40) + `handoff/perception-char-2026-06-08/course_bundle/` (204).
- **Context:** the full stack audit is in `handoff/vq2-stack-audit-2026-06-18/AUDIT_FINDINGS.md` — read it; it has the camera/objective/loss findings. Branch `vq2-stack-audit-2026-06-18` is the hardened spine; work on your own branch off it.

---

## 8. DELIVERABLE (by morning, in `handoff/vq2-overnight-goal-2026-06-18/REPORT.md`)

1. **The leaderboard** — every experiment, good-fix (oracle/deployed/course + CI), verdict vs champion.
2. **The verdict** — either **A NEW CHAMPION** (weights path + exact recipe + the good-fix proof it beats Round-1 on every metric with CI margin), **or** a documented **"Round-1 holds"** (the N levers exhausted + why each failed).
3. The **selected checkpoint** (by good-fix) + the exact one-line reproduce command + the env it was trained in.
4. Any **partial wins** (beat the champion on *some* metrics) worth pursuing next.

---

## 9. GUARDRAILS

- Never overwrite the champion weights or the deployed model; never flip `fly_vq1.py`'s default.
- Own git branch; do **not** edit `memory/`; do **not** commit the large dataset.
- Resumable runs only; serialize on the one GPU; verify the venv before every run.
- One variable per experiment; reproduce the champion before trusting any result; a win must clear the CI.

**Go beat 80%. If it can't be beaten with these levers, prove it cleanly and tell us why.**
