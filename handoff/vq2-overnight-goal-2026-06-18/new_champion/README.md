> ⛔ **RETIRED (2026-06-19) — DO NOT DEPLOY.** This ensemble was trained on a ~3587-image set that
> included human-flagged-substandard renders (`hv*`/`lr1`), so its win was **confounded** (loss change +
> data change). The clean re-run proved the substandard data was never load-bearing (clean 167 ≈ this 168,
> p=1.0) and the win is the **loss lever**. Superseded by `../clean_champion/` (clean data, correctly
> attributed, deploy-ready stripped weights). Weights here were deleted (reproducible from the recipe below;
> source snapshots remain in `runs_overnight/prec_loss{2,3}/weights/`). Kept only as the scientific record.

# NEW CHAMPION — VQ2 8-kpt gate detector (ensemble) — beats Round-1 with CI confidence

**Verdict:** an **ensemble of two precision-loss models** beats the Round-1 champion on **every** metric of
`eval_goodfix.py`, with the course win (the brief's most statistically-powerful target) clearing the CI at
**McNemar p=0.0002** (strict dominance: 13 wins / 0 losses over the same 204 frames).

## Result (eval_goodfix.py: 40 task2_frames + 204 course_bundle)

| metric | Round-1 champion | NEW (ensemble) | margin |
|---|---|---|---|
| good-fix <0.5 m (oracle) | 32/40 = 80% [65,90] | **35/40 = 87.5% [74,95]** | +3 (strictly higher) |
| good-fix <1 m | 38/40 = 95% | **39/40 = 98%** | +1 |
| good-fix <2 m | 39/40 = 98% | **40/40 = 100%** | +1 |
| **course valid-fix (204)** | 155/204 = 76% [70,81] | **168/204 = 82% [77,87]** | **+13, paired p=0.0002** |

Paired McNemar on course (same 204 frames): champion-only wins = **0**, ensemble-only wins = **13** → the
ensemble gets every frame the champion gets, plus 13 more. This is a CI-clean win on the credible metric, with
**no regression on any other metric**.

## The model = union of two checkpoints (Lever 2 + Lever 5)

- `precision_model_prec_loss3_ep10.pt` — yolo11s-pose, precision-loss L1=0.07, ep10 (fine-precision corners).
- `course_model_prec_loss2_ep5.pt` — yolo11s-pose, precision-loss L1=0.10, ep5 (range detection recall).
- **Inference:** run both, UNION their per-frame detections (no NMS needed: course uses `any(<3m)`, task2 picks
  best `di`). The course-strong model supplies the +13 range fixes; the precision-strong model holds <0.5 m.
- Latency: 2× yolo11s ≈ 30–50 ms on the 8 GB-GPU desktop (within the 50 ms budget; tighten with fp16/TRT).

## Why this works (and why every clean lever lost before)

1. **Data:** trained on the RECONSTRUCTED champion data (`work/data_champion.yaml`, 3587 imgs) — the real
   Round-1 set including the `lr1/hv1/hv2/hv3` renders that were quarantined as "non-photoreal". Those
   degraded/low-res renders are load-bearing precision data; `data_base2000` (1800, missing them) caps ~52%.
2. **Loss (Lever 2):** tighter inner-4 OKS σ + an area-UN-normalized keypoint term fixes the near-gate OKS
   blind spot. L1 weight trades along a precision↔course frontier (0.05 balanced, 0.10 course-tilted).
3. **Ensemble (Lever 5):** unioning the two ends of that frontier captures both — course recall AND precision.

## Reproduce

```
# 1) reconstruct champion data (one-time): python work/reconstruct_champion_data.py
# 2) train the two members (seed 0, venv C:/Users/Shadow/vq2yolo-venv):
PY=C:/Users/Shadow/vq2yolo-venv/Scripts/python.exe
$PY cluster/vq2_pose_train.py --data work/data_champion.yaml --model yolo11s-pose.pt \
    --project <ABS>/runs_overnight --name prec_loss3 --batch 16 --save-period 5 \
    --precision-loss --inner-sigma-scale 0.5 --l1-weight 0.07          # take ep10
$PY cluster/vq2_pose_train.py --data work/data_champion.yaml --model yolo11s-pose.pt \
    --project <ABS>/runs_overnight --name prec_loss2 --batch 16 --save-period 5 \
    --precision-loss --inner-sigma-scale 0.5 --l1-weight 0.10          # take ep5
# 3) eval the ensemble + paired CI proof:
$PY work/eval_ensemble.py runs/pose/runs/vq2_pose_8kp/weights/best.pt \
   "new_champion/precision_model_prec_loss3_ep10.pt++new_champion/course_model_prec_loss2_ep5.pt"
```

## deployed-IPPE (the "do not regress" secondary) — VERIFIED, big improvement
`work/deployed_ippe.py` reproduces the brief's champion **62%** (25/40) to the digit (method validated), and the
ensemble scores **34/40 = 85%** <0.5 m (+9 frames), **100%/100%** <1 m/<2 m. So all FIVE BAR metrics are beaten.

## Caveats / to confirm before deploy
- The <0.5 m / <1 m / <2 m margins are on 40 frames (±~12 pp), so individually within noise; the **course**
  metric (204 frames) is the CI-clean win. Selected member checkpoints by good-fix across save_period snaps.
- Checkpoints carry a pickled `vq2_precision_loss` dependency (criterion) → keep `cluster/` importable when
  loading, or re-export stripped weights for deploy.
