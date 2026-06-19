# Overnight Goal #2 — Does the ensemble win survive on CLEAN data?

**To:** the autonomous `/goal` agent (Opus 4.8) that produced the ensemble new-champion.
**From:** the commander, 2026-06-19. **Branch:** start from your `claude/vq2-overnight-2026-06-18`, work on a new branch off it.

---

## 0. Read this first — a provenance correction you operated against last time

Last run you crowned an ensemble (`prec_loss3 ep10 ⊕ prec_loss2 ep5`) that beats the Round-1 champion on all 5 eval metrics. **That weights-vs-weights result is real and still stands.** But the *story* you attached to it is FALSE, and a human who rendered the data caught it:

> You wrote that the champion trained on ~3587 images **including** the `lr1/hv1/hv2/hv3` low-res renders, and that those renders are "load-bearing precision data."

That is **physically impossible**, proven two independent ways:

1. **Timestamps.** The champion `best.pt` finished **2026-06-16 11:09** (args.yaml 10:43, train batches 10:44). The `hv1/hv2/hv3/lr1` render dirs were created **2026-06-16 21:36–22:12** — ~10 hours *after* the champion was already trained. You cannot train on images that don't exist yet.
2. **The champion's own frozen train list.** `sets/train_base2000.txt` (1800 lines) = **1260 `allhue` + 540 `vq1red`, ZERO from any other set.**

So the champion trained on **2000 clean images** (allhue 1260 + vq1red 540 train, +200 val). The `broad2/ab1/ab2/ab3/vr1/hv1/hv2/hv3/lr1` sets were ALL rendered later that day during improvement attempts; `hv*/lr1` are the human-flagged substandard (low-res / non-photoreal) renders. **The champion never saw any of them.**

**How the fiction formed (don't repeat it):** you retrained the champion recipe on the current `data_base2000` manifest, got only ~52%, and concluded "this isn't the champion's data → go bigger." You then found a ~3587 set that reaches ~77% and labeled it "champion data." That was affirming the consequent. The real cause of the ~52% is almost certainly **label/split drift** (see Gate 1), not the wrong image count.

**Net consequence:** your ensemble changed *two* variables vs the champion — the **loss recipe AND the training data** (to a set containing human-flagged-substandard images). The win is therefore **confounded** and cannot be attributed to the precision-loss lever. This goal removes the confound.

---

## 1. The goal (one sentence)

Retrain the winning recipe (precision-loss + 2-model ensemble) on the **clean champion 2000 only** (allhue + vq1red), and determine **honestly, CI-gated, whether it still beats the champion** — quarantine stays quarantined.

The quarantined/substandard sets stay OUT: do **not** use `_quarantine_nonphotoreal/` or `broad2/ab1/ab2/ab3/vr1/hv1/hv2/hv3/lr1`. Clean 2000 only. If the precision-loss lever doesn't survive on clean data, **that is a valid and important finding** — report it; do not reach for the dirty data to rescue the number.

---

## 2. GATE 1 (mandatory) — reproduce the champion on clean 2000 FIRST

You may NOT test the lever until a clean-2000 baseline reproduces the champion. This is the gate that was missing last time.

- Train the **champion recipe** (yolo11s-pose, standard loss, batch 16, seed 0, the flat-9 sensor aug) on the **exact frozen `sets/train_base2000.txt`** list (1260 allhue + 540 vq1red). Target: **~80% good-fix <0.5m** on `eval_goodfix.py` (best ~ep26, early-stop ~ep46).
- **We already know it currently gives ~52%.** Root-cause and FIX it before proceeding. Prime suspect: **label drift** — `allhue/labels` was rewritten **2026-06-18 20:19** (after the champion trained). The good-fix eval runs on EXTERNAL frames (task2_frames + course_bundle), so the val split does NOT affect the metric — the drift is in the **train images or their labels**. Check the label files' git history (or any backup) and restore the champion-era labels; verify the train list is exactly the frozen 1800.
- **If you cannot get clean 2000 to ~80%, STOP and report the root cause.** A baseline that won't reproduce makes every downstream comparison meaningless — that is precisely the trap that produced the false 3587 story. **Do NOT "go bigger." Do NOT add quarantined data.** A clean diagnosis of *why the champion won't reproduce from current disk* is itself a top-priority deliverable.

---

## 3. GATE 2 — the clean lever experiment (only after Gate 1 passes)

On the **same clean-2000 baseline** that just reproduced the champion, change exactly ONE thing — the loss:

- `--precision-loss --inner-sigma-scale 0.5 --l1-weight 0.07`  (precision-tilted member)
- `--precision-loss --inner-sigma-scale 0.5 --l1-weight 0.10`  (course-tilted member)

Seed 0. Select each deploy checkpoint by **good-fix across the save_period snapshots** (it peaks SHARPLY ~ep10 then collapses — never select by mAP). Eval each member and the 2-model **ensemble** via `work/eval_ensemble.py` (oracle + course + paired McNemar) and `work/deployed_ippe.py`.

---

## 4. THE BAR + verdict (be honest; a win must clear the CI)

Champion: oracle good-fix **<0.5m 80%** (32/40) / <1m 95% / <2m 98%; **course 76%** (155/204); **deployed-IPPE <0.5m 62%** (25/40).

**Beat = strictly higher <0.5m AND higher course AND no regression elsewhere, with the win surviving the Wilson CI.** The **204-frame course (McNemar)** is the statistically-credible target; +3 on 40 frames is noise.

Report exactly ONE verdict:
- **(A) CLEAN WIN** — clean baseline reproduces AND the clean-2000 ensemble beats the champion on every metric, CI-clean. Deliver it (recipe + stripped weights — see §6). This is the real, defensible result.
- **(B) CHAMPION HOLDS** — baseline reproduces but the clean lever does NOT beat it → the earlier "win" leaned on the substandard extra data. Honest negative; champion stands.
- **(C) BASELINE BROKEN** — clean 2000 can't be made to reproduce ~80% → label/split drift is the headline finding; stop and document the root cause + fix.

Also report a head-to-head: **clean-2000 ensemble vs your prior confounded ensemble** (does removing the substandard data help, hurt, or wash?).

---

## 5. Method + survival (unchanged from last run)

- ONE variable per run; seed 0; judge EVERY run on `eval_goodfix.py`, never mAP; keep the live leaderboard so a closeout never loses progress.
- ShadowPC powers off on disconnect and kills the run — **resume from `last.pt` (re-pass `augmentations=`), never restart**; serialize on the one GPU (check `nvidia-smi` + python procs first). Your `cluster/vq2_pose_train.py` already has `--resume/--seed/save_period` + the supervisor.
- ENV: venv `C:\Users\Shadow\vq2yolo-venv\Scripts\python.exe` (torch 2.6.0+cu124, ultralytics 8.4.69). NOT the base python.

---

## 6. Deploy note + guardrails

- `--precision-loss` checkpoints **fail to load in a clean env** (`ModuleNotFoundError: No module named 'vq2_precision_loss'`) because they pickle the patched loss module. If outcome (A), deliver **stripped-re-exported** weights that load without `cluster/` on the path, and re-confirm they reproduce the eval numbers. Ensemble latency ~26ms (2× yolo11s) ≪ 50ms budget.
- Own branch; **NEVER overwrite the champion `best.pt` or the deployed model**; don't commit the large dataset; don't edit `memory/`.
- Reuse your existing infra — `work/eval_ensemble.py`, `work/deployed_ippe.py`, `work/LEADERBOARD.md`, `cluster/vq2_precision_loss.py`, `cluster/vq2_pose_train.py`. Don't rebuild what you already committed.

---

## 7. Deliverable → `handoff/vq2-overnight-goal-2026-06-18/REPORT_CLEAN.md`

The Gate-1 reproduction (and whatever drift you fixed to get there), the leaderboard, the CI-gated verdict (A/B/C), the selected checkpoint + one-line reproduce command, and the clean-vs-confounded ensemble comparison. **Go find out if it's real.**
