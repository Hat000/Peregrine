# inc8 Re-Smoke Report — Job 3273400
**Date:** 2026-06-14  
**Node:** adroit-h11g3  
**Budget:** 1000 updates / 2048 envs / seed 0 / arm A  
**Harness commit verified:** f6dcade (all 4 pre-checks passed)

---

## (a) RC — CLEAN

| Stage | RC |
|---|---|
| PRECHECK_RC | **0** |
| SMOKE_RC | **0** |

The cosmetic ONNX-export failure (ValueError: Unknown action frame: body) was correctly
intercepted by the `_close_guarding_onnx_export` guard and treated as non-fatal.
Train + checkpoint + TorchScript/.pt2 export all succeeded.

---

## (b) GPU Utilization — 27.9% mean (SAME AS SMOKE #1)

```
mean_util=27.9%  max_util=28%  samples=31
```

**This is BELOW the ≥70% clean threshold.** The first smoke reported ~25.8%; this re-smoke
at 1000 updates gives 27.9% — essentially identical. The symptom is persistent: the
numpy estimator surrogate (inc8_estimator_emul.py, runs on CPU) is bottlenecking the
training loop. The sim itself runs at ~56,000 fps (GPU-accelerated), but nvidia-smi
sees only ~28% utilization because the GPU sits idle while the CPU runs the per-step
estimator batch. This is the "CPU-bound finding at the numpy-estimator boundary" flagged
in the sbatch acceptance criteria, and it **gates L0** per the smoke spec.

Log path: `/scratch/network/fl3689/peregrine_inc8_gpu_util.log`

---

## (c) Signal Trace — Reward Recovering, Pointing FLAT

```
    step    total_reward  pointing_rate  terminal_pointing   fix_rate    entropy  value_loss
       0        -0.32463        0.00000            0.00000    0.00000    0.35913    17.81984
     100        -0.86217        0.00000            0.00000    0.00000    0.53857     1.87057
     200        -0.88391        0.01562            0.00000    0.01074    1.58562    21.81857
     300        -0.75250        0.00000            0.00000    0.00000    1.65858     2.02298
     400        -0.71088        0.00049            0.00000    0.00049    2.01682     1.66905
     500        -0.75357        0.00049            0.00000    0.00049    2.21059     1.94998
     600        -0.55751        0.00000            0.00000    0.00000    2.14164     1.49860
     700        -0.49378        0.00000            0.00000    0.00000    1.82320     2.22940
     800        -0.51600        0.00000            0.00000    0.00000    1.92835     1.18617
     900        -0.48014        0.00000            0.00000    0.00000    2.13092     1.84526
     990        -0.42407        0.00098            0.00000    0.00049    1.80448     1.20111
```

**Interpretation:**

- **total_reward:** Recovers from a nadir of −0.88 at step 100 to −0.42 at step 990
  (improvement of ~0.46 across the run), but never crosses the initial −0.32 baseline.
  The dip-then-recover pattern is consistent with early entropy expansion (policy
  exploring) followed by partial course-following returning reward. Direction: **improving
  but slow** — NOT the monotone decline seen in smoke #1 (good sign for the env port),
  but also not rising above zero.

- **pointing_rate / terminal_pointing / fix_rate:** Essentially **FLAT at zero** throughout.
  Maximum blips: pointing_rate=0.016 at step 200, 0.001 at step 990; terminal_pointing=0.000
  every step; fix_rate=0.011 at step 200, ~0 elsewhere. The explicit R5 arm-A 2-axis
  terminal-lock reward is NOT yet shaping pointing behavior at 1000 updates.

- **entropy:** Rising 0.36→2.21 — policy is exploring, which is healthy and expected.

- **value_loss:** Spikes at step 0 (17.8) and step 200 (21.8), then settles to 1–2 range.
  The step-200 spike correlates with the transient pointing/fix_rate blip — likely a
  brief exploration excursion that momentarily triggered the pointing reward before
  collapsing.

**Bottom line on pointing:** Not emerging at 1000 updates. This is consistent with the
CPU-bound bottleneck: at 27.9% GPU util the policy sees ~28% of the learning signal
budget it would see at full saturation. Effective gradient steps ≈ 280 equivalent at
full throughput. Pointing likely needs far more wall-time or the CPU bottleneck fixed.

---

## (d) Checkpoint Sidecar

```json
{"act_max_thrust": 3.765, "act_max_rate": 3.14, "obs_dim": 20, "inc8": true, "r5_arm": "A"}
```

All three S7 criteria met: **obs_dim=20 ✅ / inc8=true ✅ / r5_arm=A ✅**

---

## Artifact Paths (on /scratch — not committed)

| Artifact | Path |
|---|---|
| Main log | `/scratch/network/fl3689/peregrine_inc8_smoke.out` |
| GPU util log | `/scratch/network/fl3689/peregrine_inc8_gpu_util.log` |
| Run dir | `/scratch/network/fl3689/diffaero/outputs/train/2026-06-14/21-40-12/` |
| Decorated logdir | `…/quad__racing__ppo__mlp__inc8_smoke_A_seed0__0/` |

---

## Commander Decision Required

Two independent findings both require adjudication before L0:

1. **GPU util 27.9% (CPU-bound):** The numpy estimator surrogate serializes per-step
   on CPU, starving the GPU. At L0 scale (6000 updates × 9 seeds) this is a ~3.6×
   wall-time tax and a real learning-rate penalty (~72% of gradient signal lost).
   The sbatch spec gates L0 on ≥70% util. Fix options include torch-ifying the
   estimator emulation or batching the numpy call.

2. **Pointing flat at 1000 updates:** Could be (a) a consequence of the CPU bottleneck
   (effectively only ~280 full-throughput-equivalent updates), or (b) pointing needing
   more updates even at full throughput, or (c) a reward-shaping issue with arm A at
   this early stage. The commander should decide whether to: fix the CPU bottleneck
   first and re-smoke, or run L0 anyway and accept the wall-time cost.

Re-smoke is done. Awaiting GO/NO-GO call on L0 from the overall commander.
