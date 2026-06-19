> ✅ **ADOPTED (2026-06-19) — this is the official VQ2 vision model.** Independently re-verified by the
> manager (champion `best.pt` untouched; both stripped members load in a clean deploy env; beats champion
> course 167 vs 155 p=0.0005 and deployed-IPPE 82% vs 62% on a fresh run). Weights copied to the canonical
> deploy dir as `models/gate_clean_ens_precision_L107.pt` + `models/gate_clean_ens_course_L110.pt`.
> Supersedes the retired `../new_champion/`. Go-live gate: closed-loop flight validation (see COMMANDER_REPORT.md).

# CLEAN-DATA champion (ensemble) — the precision-loss win, validated on clean 2000

**The ensemble win is REAL on clean data and attributable to the LOSS LEVER (not the substandard images).**
Trained on the **clean champion 2000 only** (allhue 1260 + vq1red 540; NO quarantine / `hv*` / `lr1` / `broad2` /
`ab*` / `vr1`). Beats the Round-1 champion on **all 5** `eval_goodfix.py` metrics, course win CI-clean.

## Result (stripped weights, verified: load without `cluster/` AND reproduce these numbers)

| metric | champion | CLEAN ensemble | |
|---|---|---|---|
| oracle <0.5 m | 32/40 (80%) | **35/40 (87.5%)** | strictly higher |
| oracle <1 m / <2 m | 95% / 98% | **100% / 100%** | higher |
| **course (204)** | 155 (76%) | **167 (82%)** | **McNemar p=0.0005, 12–0** |
| deployed-IPPE <0.5 m | 25/40 (62%) | **33/40 (82%)** | +8 frames |

Also **beats the controlled standard-loss clean baseline** on course (167 vs 159, p=0.0078) — the confound-free
proof that the *loss* (not the data) drives the win. And it is **statistically identical to the prior confounded
ensemble** (167 vs 168, p=1.0) → the dirty data was never load-bearing.

> Caveat: the 40-frame `<0.5m` is high-variance (`clean10` swung 10→36→25 over ep5/10/15). The **course (204) +
> deployed-IPPE** wins are the credible, stable claims; the `<0.5m` margin is directional but noisy. Separately,
> the champion's own 80% is NOT bit-reproducible on the pristine clean 2000 (GATE 1 = C; run-variance, not label
> drift) — so the clean lever was validated against the controlled standard-clean baseline, which it beats.

## The model = union of two clean precision-loss checkpoints

- `clean_precision_L107_ep10.pt` — yolo11s-pose, clean 2000, precision-loss `--inner-sigma-scale 0.5 --l1-weight 0.07`, ep10.
- `clean_course_L110_ep10.pt`   — yolo11s-pose, clean 2000, precision-loss `--inner-sigma-scale 0.5 --l1-weight 0.10`, ep10.
- Inference: run both, UNION per-frame detections (no NMS: course uses `any(<3m)`, task2 picks best `di`). ~26 ms.

## Reproduce
```
PY=C:/Users/Shadow/vq2yolo-venv/Scripts/python.exe   # ultralytics 8.4.69, torch 2.6.0+cu124, seed 0
DATA=C:/Users/Shadow/Peregrine-vq2data/handoff/vq2-blender-render-2026-06-15/sets/data_base2000.yaml  # clean 1800/200
$PY cluster/vq2_pose_train.py --data $DATA --model yolo11s-pose.pt --project <ABS>/runs_overnight \
    --name prec_clean07 --batch 16 --save-period 5 --precision-loss --inner-sigma-scale 0.5 --l1-weight 0.07   # ep10
$PY cluster/vq2_pose_train.py --data $DATA --model yolo11s-pose.pt --project <ABS>/runs_overnight \
    --name prec_clean10 --batch 16 --save-period 5 --precision-loss --inner-sigma-scale 0.5 --l1-weight 0.10   # ep10
$PY handoff/vq2-overnight-goal-2026-06-18/work/strip_precision_ckpt.py <ep10.pt> clean_champion/<name>.pt       # strip
$PY handoff/vq2-overnight-goal-2026-06-18/work/eval_ensemble.py runs/pose/runs/vq2_pose_8kp/weights/best.pt \
   "clean_champion/clean_precision_L107_ep10.pt++clean_champion/clean_course_L110_ep10.pt"                      # proof
```
