# σ_p0 Adroit Measurement — inc8 rc1, 2-axis terminal-centering (sigmap0-adroit, 2026-06-19)

**TL;DR.** The torch instrument is **VALIDATED** (seed0 yaw cross-check σ_p0_lat = **0.1981** vs numpy
**0.1768**, Δ=0.0213 < 0.04 → AGREE). But the **2-axis (g_yaw=−3, g_pitch=3) deliverable is NO-DATA /
un-measurable**: under **deterministic** inference rc1 **crashes ~1 s after spawn** on the 2-axis look-at
(reach = 0 / all 3 seeds), so there are zero gate-4 crossings to measure. The look-at *pointing primitive
works* (band_el 66°→20°, fix_rate 0→0.25) but g_pitch=3 **destabilizes deterministic flight**. This is the
escape-hatch "training-success-was-a-stochastic-artifact" finding — reported, not forced. The best
**faithful** σ_p0 in hand is the yaw-only seed0 number **0.198 m → NO-GO vs 0.08** (consistent with the
numpy 0.177). **Gate-4 verdict: NO-DATA on 2-axis; NO-GO on the measurable yaw-only fallback.**

A **5th harness bug** (directory-layout, not physics) was found and fixed to get the job to run at all.

---

## 1. What was run

- **Canonical run** = `rl/inc8_sigmap0_torch.sbatch` (CKPT_SUB=best, SEEDS="0 1 2", N_ENVS=256,
  HORIZONS=2.5, STANDFRAC=1.0, EVALSEED=0). Per seed: `--lookat yaw` (cross-check) + `--lookat auto`
  (2-axis deliverable). Node adroit-h11g3 (Tesla V100-32GB).
- Job **3278150** (first submit): ran 2 s, skipped all 3 seeds — **harness bug** (see §5).
- Job **3278151** (after fix): COMPLETED 00:04:18, all 6 runs executed. **← the result set.**
- Job **3278154**: diagnostic g_pitch sweep on seed0 (§4).

Pre-sync verified byte-exact (eval `44d32eaf…`, sbatch then patched to `e865078e…`); `CKPT_SUB=best`
confirmed; all 3 seeds had complete `best/{actor,critic}.pth` + `.hydra/config.yaml`. checkquota clean
(scratch 56/93 GiB).

---

## 2. Results — canonical run (job 3278151), VERBATIM summary lines

```
# files sort seed0_auto, seed0_yaw, seed1_auto, seed1_yaw, seed2_auto, seed2_yaw
SIGMAP0_TORCH_SUMMARY ckpt=train lookat=auto reach=0/14062 reach_rate=0.000 success_rate=0.000 sigma_p0_lat=nan bias_lat=+nan lat_p90=nan lat_p99=nan sigma_p0_vert=nan bias_vert=+nan fix_rate=0.000 term_point=0.131 verdict=NO-DATA   # seed0 auto
SIGMAP0_TORCH_SUMMARY ckpt=train lookat=yaw  reach=1768/3317 reach_rate=0.533 success_rate=0.283 sigma_p0_lat=0.1981 bias_lat=-0.0635 lat_p90=0.3333 lat_p99=0.3993 sigma_p0_vert=0.1906 bias_vert=-0.0484 fix_rate=0.000 term_point=0.030 verdict=NO-GO     # seed0 yaw
SIGMAP0_TORCH_SUMMARY ckpt=train lookat=auto reach=0/10490 reach_rate=0.000 success_rate=0.000 ... fix_rate=0.250 term_point=0.000 verdict=NO-DATA   # seed1 auto
SIGMAP0_TORCH_SUMMARY ckpt=train lookat=yaw  reach=0/10240 reach_rate=0.000 success_rate=0.000 ... fix_rate=0.005 term_point=0.127 verdict=NO-DATA   # seed1 yaw
SIGMAP0_TORCH_SUMMARY ckpt=train lookat=auto reach=0/9984  reach_rate=0.000 success_rate=0.000 ... fix_rate=0.071 term_point=0.000 verdict=NO-DATA   # seed2 auto
SIGMAP0_TORCH_SUMMARY ckpt=train lookat=yaw  reach=0/9984  reach_rate=0.000 success_rate=0.000 ... fix_rate=0.001 term_point=0.005 verdict=NO-DATA   # seed2 yaw
```

### YAW CROSS-CHECK (instrument validity — sbatch awk, VERBATIM)
```
  YAW CROSS-CHECK: torch sigma_p0_lat=0.1981 vs numpy 0.1768  delta=0.0213  AGREE (instrument trustworthy)
  YAW CROSS-CHECK: torch sigma_p0_lat=0.0000 vs numpy 0.1768  delta=0.1768  DISAGREE   # seed1 (reach=0, no data)
  YAW CROSS-CHECK: torch sigma_p0_lat=0.0000 vs numpy 0.1768  delta=0.1768  DISAGREE   # seed2 (reach=0, no data)
```
The two "DISAGREE" lines are an awk artifact of `σ=nan→0.0` on a **reach=0** ensemble (no crossings),
**not** an instrument discrepancy. The only seed with a measurable yaw ensemble (seed0) **AGREES** within
the 0.04 band → **instrument VALIDATED**.

### Key per-run detail
| seed·mode | episodes_ended | ≈episode len | reach | success | σ_p0_lat | σ_p0_vert | fix_rate | band_az | band_el | term_point |
|---|---|---|---|---|---|---|---|---|---|---|
| **0·yaw** | 3317 | ~4.8 s | **0.533** | 0.283 | **0.1981** | 0.1906 | 0.000 | 39.5° | 66.3° | 0.030 |
| 0·auto | 14062 | ~1.1 s | 0.000 | 0.000 | nan | nan | 0.000 | 13.7° | 45.6° | 0.131 |
| 1·yaw | 10240 | ~1.5 s | 0.000 | 0.000 | nan | nan | 0.005 | 45.9° | 59.2° | 0.127 |
| 1·auto | 10490 | ~1.5 s | 0.000 | 0.000 | nan | nan | **0.250** | 13.5° | **20.0°** | 0.000 |
| 2·yaw | 9984 | ~1.5 s | 0.000 | 0.000 | nan | nan | 0.001 | 48.2° | 52.0° | 0.005 |
| 2·auto | 9984 | ~1.5 s | 0.000 | 0.000 | nan | nan | 0.071 | 14.1° | 21.6° | 0.000 |

(≈episode len = 256 envs × 1876 steps / episodes_ended × 0.0333 s.)

---

## 3. Interpretation (mechanism)

1. **Instrument is sound.** seed0 yaw reproduces the numpy 0.177 within the band. The crossing math /
   cfg-key faithfulness held in practice, not just the static audit.
2. **rc1 does not fly the 2-axis course under DETERMINISTIC inference.** All 3 seeds' `auto` runs reach=0
   with ~1 s episodes = the drone crashes almost immediately after the standing-start spawn. Training's
   "3/3 flew, success ~0.50–0.54" was measured under **stochastic** action sampling (PPO exploration);
   the deterministic policy **mean** sits in an unstable basin the noise was masking. ← the escape-hatch
   "stochastic-artifact" finding.
3. **The destabilizer is specifically the g_pitch=3 elevation pointing**, isolated cleanly: same
   checkpoint, same env, only g_pitch 0→3 separates seed0·yaw (FLIES, σ=0.198) from seed0·auto (CRASHES).
4. **The look-at pointing primitive itself WORKS** — the 2-axis runs achieve far better camera pointing
   than yaw-only (band_el 66°→20°, band_az 39°→13.5°, fix_rate 0→0.25 on seed1). The thesis "pointing →
   fixes" is mechanically confirmed; the problem is purely flight stability at this gain under det.
   inference.
5. **Even yaw-only flies on only 1 of 3 seeds** (seed1/seed2 yaw also reach=0) → rc1's deterministic
   flight is fragile lineage-wide; seed0 is the lone deterministic flyer.

---

## 4. Diagnostic g_pitch sweep (job 3278154, COMPLETED 00:04:20) — the stability boundary / lever

seed0 only, g_yaw=−3 fixed. Combined with the canonical gp0/gp3 points, the full elevation-gain map:

| g_pitch | reach | reach_rate | success | σ_p0_lat | lat_p99 | σ_p0_vert | fix_rate | band_az | band_el | flies? |
|---|---|---|---|---|---|---|---|---|---|---|
| off | 1864/3255 | 0.573 | 0.417 | 0.1977 | 0.401 | 0.2031 | 0.000 | 37.3° | 66.7° | ✅ |
| 0 (yaw) | 1768/3317 | 0.533 | 0.283 | 0.1981 | 0.399 | 0.1906 | 0.000 | 39.5° | 66.3° | ✅ |
| **0.5** | 1818/3401 | 0.535 | 0.302 | **0.1906** | 0.406 | 0.2057 | 0.000 | 36.1° | 64.3° | ✅ |
| 1.0 | 18/14116 | 0.001 | 0.001 | 0.1538* | 0.403 | 0.2027 | 0.000 | 10.8° | 61.9° | ❌ collapse |
| 1.5 | 0/14331 | 0.000 | 0.000 | nan | — | nan | 0.000 | 13.2° | 58.1° | ❌ crash |
| 2.0 | 0/14092 | 0.000 | 0.000 | nan | — | nan | 0.000 | 14.5° | 54.5° | ❌ crash |
| 2.5 | 0/14080 | 0.000 | 0.000 | nan | — | nan | 0.000 | 13.9° | 48.1° | ❌ crash |
| 3.0 (auto) | 0/14062 | 0.000 | 0.000 | nan | — | nan | 0.000 | 13.7° | 45.6° | ❌ crash |

\* gp1.0 σ is on only **18 survivorship-biased crossings** (n<30 → NO-DATA; note bias_lat jumps to +0.128).

VERBATIM sweep summary lines (sort: gp0p5, gp1p0, gp1p5, gp2p0, gp2p5, off):
```
SIGMAP0_TORCH_SUMMARY ckpt=train lookat=auto reach=1818/3401 reach_rate=0.535 success_rate=0.302 sigma_p0_lat=0.1906 bias_lat=-0.0707 lat_p90=0.3236 lat_p99=0.4055 sigma_p0_vert=0.2057 bias_vert=+0.0028 fix_rate=0.000 term_point=0.053 verdict=NO-GO     # gp0.5
SIGMAP0_TORCH_SUMMARY ckpt=train lookat=auto reach=18/14116 reach_rate=0.001 success_rate=0.001 sigma_p0_lat=0.1538 bias_lat=+0.1278 lat_p90=0.3229 lat_p99=0.4033 sigma_p0_vert=0.2027 bias_vert=+0.0748 fix_rate=0.000 term_point=0.115 verdict=NO-DATA   # gp1.0
SIGMAP0_TORCH_SUMMARY ckpt=train lookat=auto reach=0/14331 ... verdict=NO-DATA   # gp1.5
SIGMAP0_TORCH_SUMMARY ckpt=train lookat=auto reach=0/14092 ... verdict=NO-DATA   # gp2.0
SIGMAP0_TORCH_SUMMARY ckpt=train lookat=auto reach=0/14080 ... verdict=NO-DATA   # gp2.5
SIGMAP0_TORCH_SUMMARY ckpt=train lookat=off  reach=1864/3255 reach_rate=0.573 success_rate=0.417 sigma_p0_lat=0.1977 bias_lat=-0.0529 lat_p90=0.3266 lat_p99=0.4009 sigma_p0_vert=0.2031 bias_vert=-0.0132 fix_rate=0.000 term_point=0.027 verdict=NO-GO   # off
```

**What the sweep proves (the lever):**
- **Sharp stability cliff between g_pitch 0.5 and 1.0.** ≤0.5 flies (~53–57% reach); ≥1.0 collapses
  (1.0 → 0.1%, ≥1.5 → exactly 0).
- **The flyable band is useless for centering.** At every g_pitch where seed0 flies (off / 0 / 0.5),
  **fix_rate = 0.000** and σ_p0_lat ≈ 0.19–0.20 (NO-GO). The `off` baseline (0.198) ≈ yaw (0.198) ≈ gp0.5
  (0.191) → at flyable gains the look-at adds *nothing* to terminal centering (too weak to seat a fix).
- **The pointing-effective band crashes.** Elevation pointing only starts to bite at g_pitch ≥ 1
  (band_az collapses to ~11–14°), but that is exactly where deterministic flight dies.
- ⇒ **There is NO inference-time g_pitch that both flies deterministically AND tightens gate-4.** rc1
  cannot be salvaged with a knob. (Note: seed0 never seats a fix at any gain; seed1's `auto` did reach
  fix_rate 0.25 / band_el 20° in the canonical run, but also crashed — pointing capacity exists in the
  lineage, deterministic flight at that gain does not.)

---

## 5. Harness bug found + fixed (the 5th)

`rl/inc8_sigmap0_torch.sbatch` resolved the checkpoint via
`RUNDECOR=$(ls -dt ${RUNDIR}/*${RUNNAME}__* | head -1); RUN=${RUNDECOR:-$RUNDIR}`. For the rc1 recenter
runs the decorated subdir `quad__racing__ppo__mlp__inc8_recenter_seed0_rc1__0/` contains **only a
tfevents file** — `best/`, `periodic/`, `checkpoints/`, `.hydra/` all live at the **RUNDIR** level. So
`CKPT=<decorated>/best/actor.pth` did not exist → all 3 seeds skipped, job ran 2 s. The static
launch-audit could not catch this (it is runtime directory state, not code). **Fix (line ~80):** accept
the decorated subdir only if `${RUNDECOR}/${CKPT_SUB}/actor.pth` exists, else fall back to RUNDIR; robust
to both layouts; `find_run_root` still resolves `.hydra/config.yaml` by walking up from `RUNDIR/best`.
Patched locally (uncommitted), re-synced byte-exact (`e865078e…`), re-submitted → job 3278151 ran.
**Files touched:** `rl/inc8_sigmap0_torch.sbatch` (working tree, NOT committed/pushed) +
`handoff/sigmap0-adroit-2026-06-19/` (this report + the diagnostic sbatch).

---

## 6. Gate-4 verdict + recommendation

- **2-axis σ_p0_lat (the deliverable): NO-DATA** — un-measurable on rc1. Under deterministic inference the
  g_pitch=3 look-at crashes the drone ~1 s after spawn (reach = 0, all 3 seeds). Not a physics or
  instrument limit; a **deterministic-flyability** limit of this policy.
- **Best measurable faithful σ_p0 (seed0, yaw-only or racing-line): σ_p0_lat = 0.198 m,
  σ_p0_vert = 0.191 m, lat_p99 = 0.399 → NO-GO vs 0.08 / 0.24.** Reproduces the numpy 0.177 (instrument
  validated). This is the binding gate-4 number today.
- **Gate-4 GO/NO-GO for the 2-axis racer: cannot be issued** — rc1 is not a deterministically-flyable
  2-axis policy. The g_pitch sweep proves no inference-time setting fixes this: flyable gains (≤0.5)
  add zero fixes / σ stays 0.19; fix-effective gains (≥1) crash.

**Single most promising lever (NO inference knob exists — it is a RETRAIN):** produce a
**deterministically-stable** policy that sustains a look-at gain high enough (g_pitch ≳ 1) for the
elevation pointing to actually seat fixes. Concretely, in priority order:
1. **Select checkpoints on DETERMINISTIC reach/success, not stochastic.** rc1's `best` was the stochastic
   high-water mark (~0.50–0.54), which does not transfer to the deterministic mean. Gate the saved
   checkpoint on a deterministic (test=True) eval.
2. **Close the stochastic→deterministic gap in training:** anneal the action-sampling noise toward 0 over
   training (so the policy learns to fly on its mean, not on exploration noise), and/or regularize/eval
   the policy mean under full look-at gain rather than only sampled actions.
3. **Re-profile the look-at gain schedule** so full g_pitch is held with the noise annealed (the current
   `lookat_warmup_updates=200` ramps the gain but the policy still leaned on exploration noise to survive
   it). This is an architecture/training-pipeline change → **commander/Fengyou decision, out of this
   worker's measure-only scope.**

Until such a policy exists, the binding gate-4 σ_p0 stays **0.198 m = NO-GO**. (This also corroborates the
known lineage fragility: even **yaw-only** flies deterministically on only 1 of 3 seeds — seed0.)

*Provisional caveat (unchanged): emul obs, not a real detector→PnP→KF (#37); any number is provisional
until the real-detector spike confirms emul fidelity. The instrument-validity result (yaw AGREE) is
independent of this caveat.*

**Optional cheap follow-ups (not run — bounded diagnosis):** (a) measure rc1 `periodic` (last snapshot)
2-axis deterministic reach — unlikely to differ (the instability is a converged-mean property, and the
launch-audit flags periodic as a weaker non-monotonic snapshot); (b) a finer g_pitch grid in (0.5, 1.0)
to pin the cliff — not load-bearing, since the flyable side already shows fix_rate=0.

---

## MEMORY-DELTA (≤10 lines — for the commander to bank; worker did NOT edit memory/)

1. **σ_p0 torch instrument VALIDATED** (seed0 `--lookat yaw` σ_p0_lat **0.1981** vs numpy **0.1768**, Δ0.021 → AGREE; job 3278151). #37 measure-in-torch path is sound.
2. **2-axis (g_pitch=3) deliverable = NO-DATA / un-measurable:** rc1 **crashes ~1 s after spawn under DETERMINISTIC inference** (reach=0, all 3 seeds, ~1 s episodes). Training's 50–54% success was a **stochastic-action artifact** — the deterministic policy MEAN is unstable.
3. Look-at **pointing primitive WORKS** (band_el 66°→20°, fix_rate 0→0.25 on seed1) — pure **flight-stability** failure, not a pointing/obs failure.
4. **g_pitch sweep (job 3278154, seed0):** sharp cliff between gp**0.5** (flies, σ 0.191, fix_rate **0**) and gp**1.0** (collapse to 0.1%); gp≥1.5 reach 0. **No inference g_pitch both flies AND tightens** — flyable gains fix_rate=0/σ≈0.19; fix-effective gains crash. `off`≈`yaw`≈`gp0.5`≈0.198.
5. **Binding gate-4 σ_p0 = 0.198 m (yaw/racing-line) = NO-GO vs 0.08** (σ_vert 0.191, lat_p99 0.399). Even yaw-only flies deterministically on **only seed0 of 3** → lineage fragility.
6. **LEVER = RETRAIN for deterministic stability (NOT a knob):** select ckpt on *deterministic* reach (rc1 `best` was stochastic high-water); anneal action noise → fly on the mean; hold look-at gain on the mean. Out of measure-only scope → commander/Fengyou.
7. **5th harness bug FIXED:** `inc8_sigmap0_torch.sbatch` RUNDECOR pointed `best/` at a tfevents-only decorated subdir; `best/`+`.hydra` live at RUNDIR. Guarded line ~80 (working-tree edit, **uncommitted, not pushed**), re-synced byte-exact (`e865078e`). `find_run_root` OK.
8. Full record: `handoff/sigmap0-adroit-2026-06-19/REPORT.md` (+ diagnostic `inc8_sigmap0_gpsweep.sbatch`). Provisional on #37 emul fidelity (instrument-validity is independent).
