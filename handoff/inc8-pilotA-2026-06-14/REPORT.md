# inc8 Arm-A 6000-Update Pilot Report — Job 3273431
**Date:** 2026-06-14  
**Node:** adroit-h11g3  
**Config:** NUPD=6000 / NENVS=2048 / SEED=0 / arm A / --time=12:00:00  
**Harness commit verified:** 1a7f9a0 (all 3 pre-checks passed)  
**Outcome:** ABORTED at early-util check (~8 min into training)

---

## Step 0 — Pre-checks: ALL PASSED

| Check | Result |
|---|---|
| `grep -c "metric_vec" peregrine_racing_inc8.py` | 2 ✅ |
| `grep -c "NUPD:-1000" peregrine_inc8_smoke.sbatch` | 1 ✅ |
| `grep -c "_close_guarding_onnx_export" peregrine_train_inc8.py` | 2 ✅ |
| Scratch quota | 56.0 / 93 GiB OK |

---

## (a) GPU Util After Sync-Fix — ABORT TRIGGERED

```
Early util check: mean_util=27.7%  samples=23  (~8 min of training data)
```

**Comparison:**

| Run | Commit | Updates sampled | mean_util |
|---|---|---|---|
| Re-smoke (job 3273400) | f6dcade | 1000 | 27.9% |
| Arm-A pilot (job 3273431) | 1a7f9a0 (sync-fix) | ~760 (aborted) | **27.7%** |

The metric_vec sync-fix (11+ GPU→CPU host-syncs/step → 1) produced **no measurable
improvement** in GPU utilization. 27.7% vs 27.9% is within sampling noise.

Per the abort rule (mean util < ~40% → scancel, do not burn the full run):
`scancel 3273431` executed. Job confirmed gone from queue.

**Root cause conclusion:** The dominant CPU bottleneck is NOT the metrics dict host-syncs —
those were already fixed. The remaining bottleneck is the deferred second source: the
**parent `compute_reward_terms` ~10 `.item()` calls/step**, each forcing a GPU→CPU sync.
This was explicitly noted in the sync-fix commit as a known second source left for follow-up.
At 27.7% util, the GPU sits idle ~72% of the time waiting on CPU reward computation. That
source must be fixed before the 9-run L0 ladder is worthwhile.

---

## (b) Pointing — NOT ASSESSED (job aborted at ~760 updates)

The job was cancelled after ~8 minutes of training (~760 updates, ~13% of the 6000-update
budget). No pointing signal assessment is possible. This is expected under the abort rule —
burning 5–8h at 27.7% util to get a pointing answer was not authorized.

---

## (c) RC / Sidecar — PARTIAL (precheck only)

- PRECHECK_RC: **0** (confirmed from log before abort; the 3-update precheck completed)
- SMOKE_RC: **not reached** (job cancelled before smoke completion)
- Sidecar from precheck checkpoint: obs_dim=20 / inc8=true / r5_arm=A ✅
  (same config as re-smoke; the precheck always writes a sidecar)

---

## Commander Decision Required

**Single finding:** the metric_vec sync-fix is insufficient — util remains ~28%.  
**Next step options (commander's call):**

1. **Fix `compute_reward_terms` `.item()` calls** (the deferred second source): vectorize
   the ~10 scalar `.item()` extractions into a single batched tensor operation, keeping
   everything on GPU until the final log step. Then re-pilot and re-measure util. This is
   the most likely path to ≥70%.

2. **GPU profile on the compute node** (optional before fixing): run `torch.profiler` or
   `nvprof` for 10 updates to quantify exactly which `.item()` calls dominate wall-time,
   then fix only those. Adds ~1h of Adroit time but gives a precise fix target.

3. **Proceed to L0 ladder at 27.7% util** (not recommended): accepts ~3.6× wall-time tax
   and ~72% gradient-signal loss. The 9-run ladder at 6000 updates each would take
   ~7–9× longer than at full GPU saturation, and pointing emergence is harder to assess.

**Artifact paths** (on /scratch; not committed):
- Partial log: `/scratch/network/fl3689/peregrine_inc8_smoke.out`
- GPU util log: `/scratch/network/fl3689/peregrine_inc8_gpu_util.log`
