> 🚩 **SUPERSEDED 2026-06-19 (511e85c) re: the σ_p0 bar.** Any "σ_p0 ≲ 0.08 / NO-GO vs 0.08 / near-field-gate-estimator pivot" conclusion in this report is OVERTURNED — the 0.08 bar was a `margin_envelope.py` double-count of the drone (real gate clearance 0.75−r ≈ 0.37–0.47 m). The REAL bar is **σ_p0_lat ≲ 0.15 (p99 ≲ 0.45)**; measured 0.15–0.20 = MARGINAL-PASSING, NOT NO-GO. Gate-4 is a REACH/PASS-RATE problem; RL stays the tool (the "pivot off RL" is RETRACTED). The engineering + measurements below STAND; only the bar and its GO/NO-GO verdict are corrected. → MEMORY.md:8 + memory/project_rl_increment_history.md §inc8-2026-06-19.
# inc8 S1 cen3 Run Report — 2026-06-16

## Jobs

| Job ID  | SEED | RUNTAG      | Out file                                    | PRECHECK_RC | S0_RC | Hydra dir          |
|---------|------|-------------|---------------------------------------------|-------------|-------|--------------------|
| 3274993 | 0    | s1cen3      | peregrine_inc8_s0_s1cen3.out               | 0           | 0     | 2026-06-16/15-51-38 |
| 3274994 | 1    | s1cen3_s1   | peregrine_inc8_s0_s1cen3_s1.out            | 0           | 0     | 2026-06-16/15-51-39 |
| 3274995 | 2    | s1cen3_s2   | peregrine_inc8_s0_s1cen3_s2.out            | 0           | 0     | 2026-06-16/15-51-38 |

Config: `++env.lookat_g_yaw=-3.0 +env.rw_centering=3.0` — confirmed in all 3 command lines.
Baseline: S1@rw_centering=1.0 (seed-0 estim_err ~0.115, fix_rate 0.14–0.16, lockband ~0.55).
Target: estim_err_ip_m ≲ 0.08 near gate.

**CRITICAL: Hydra dir collision.** Seeds 0 and 2 both started at `15:51:38` → wrote to the same
checkpoint directory. Their traces are bit-identical — two SLURM jobs interfered by training the
same model files simultaneously. We have **2 independent experiments**, not 3.

---

## Seed 0 (s1cen3, RUNTAG job 3274993) — COLLAPSED (Hydra-collided with seed 2)

```
[inc8-tb-trace] TRAJECTORY (every ~100 updates):
    step        total_reward       pointing_rate   terminal_pointing   lockband_pointing            fix_rate     band_az_abs_deg      estim_err_ip_m           centering             entropy          value_loss
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
       0            -0.25025             0.00000             0.00000             0.00000             0.00000             0.00000             0.14998            -0.27392             0.34254            25.99694
     100            -0.78212             0.06348             0.04858             0.04112             0.01807            40.35346             0.09437            -0.08712            -0.67937           715.69031
     200            -0.44650             0.03320             0.01597             0.01839             0.00830            48.99853             0.10618            -0.10671            -1.33391           166.22234
     300            -0.33456             0.00098             0.00432             0.00000             0.00049            52.11469             0.07860            -0.06783            -1.85037           101.41670
     400            -0.58117             0.00537             0.00885             0.00098             0.00049            57.04295             0.08557            -0.06664            -3.06547           125.67641
     500            -0.60681             0.00732             0.00568             0.00612             0.00146            54.48385             0.08356            -0.07142            -3.93633            92.23736
     600            -0.66592             0.00488             0.00203             0.00524             0.00244            56.01302             0.08345            -0.06738            -4.93674           137.58890
     700            -0.64012             0.00293             0.00637             0.00100             0.00098            56.90646             0.08208            -0.06728            -5.76311           108.20207
     800            -0.82197             0.00635             0.00548             0.00424             0.00146            54.40987             0.07919            -0.06997            -6.49183           104.48401
     900            -0.55047             0.00342             0.00794             0.00204             0.00098            58.86646             0.08503            -0.06224            -6.86476            90.60587
    1000            -0.40584             0.00439             0.00969             0.00000             0.00000            59.87486             0.08379            -0.06755            -6.84634           102.72028
    1100            -0.44561             0.00342             0.00823             0.00195             0.00098            54.74739             0.08313            -0.06800            -7.01381           116.19659
    1200            -0.41522             0.00732             0.01660             0.00308             0.00146            57.55989             0.08305            -0.06746            -6.86366            80.50458
    1300            -0.64103             0.00244             0.00594             0.00104             0.00098            57.36203             0.07980            -0.06717            -6.78846            62.05604
    1400            -0.37278             0.00439             0.01486             0.00196             0.00098            59.45725             0.08445            -0.07007            -6.88551           136.64101
    1500            -0.61095             0.00049             0.00199             0.00000             0.00049            57.01204             0.08384            -0.06897            -6.86613            82.66019
    1600            -0.52511             0.00391             0.01502             0.00000             0.00049            61.02105             0.08586            -0.06846            -6.76053           106.72163
    1700            -0.55279             0.00244             0.00613             0.00101             0.00098            57.75228             0.08173            -0.06260            -6.75652            89.35983
    1800            -0.48092             0.00146             0.00459             0.00000             0.00000            57.19073             0.08778            -0.06919            -6.82815            81.71291
    1900            -0.62602             0.00098             0.00208             0.00100             0.00098            54.65485             0.08412            -0.06930            -6.71261            77.39163
    2000            -0.84871             0.00195             0.00889             0.00000             0.00000            56.40952             0.08446            -0.06319            -6.76534            86.96525
    2100            -0.47428             0.00293             0.01344             0.00000             0.00000            58.66002             0.09115            -0.07526            -6.61063            70.62155
    2200            -0.40166             0.00439             0.01364             0.00201             0.00244            58.98727             0.08353            -0.06608            -6.59857            75.22408
    2300            -0.57813             0.00146             0.00439             0.00000             0.00000            60.00692             0.08428            -0.06824            -6.56413            72.09842
    2400            -0.63568             0.00293             0.00648             0.00196             0.00098            60.18200             0.08894            -0.06910            -6.66443            79.61572
    2500            -0.41687             0.00146             0.00451             0.00096             0.00049            57.82361             0.08190            -0.06269            -6.59966            74.90669
    2600            -0.52149             0.00293             0.00821             0.00000             0.00049            59.11648             0.08215            -0.06745            -6.65907            87.99635
    2700            -0.31531             0.00098             0.00212             0.00000             0.00049            59.95911             0.08342            -0.06826            -6.59291            79.66490
    2800            -0.50703             0.00391             0.00648             0.00100             0.00000            57.68929             0.07909            -0.06403            -6.54302            81.08250
    2900            -0.70482             0.00391             0.01018             0.00312             0.00146            55.79912             0.08079            -0.06960            -6.70764            73.55509
    3000            -0.59739             0.00293             0.00415             0.00000             0.00000            55.20550             0.08189            -0.06873            -6.71818            83.70841
    3100            -0.64551             0.00293             0.00612             0.00203             0.00098            56.21206             0.08085            -0.06798            -6.62053           102.54459
    3200            -0.30198             0.00195             0.00808             0.00000             0.00000            58.28343             0.07617            -0.06134            -6.52136            65.48878
    3300            -0.76236             0.00293             0.00762             0.00000             0.00049            56.31193             0.07773            -0.06821            -6.59545            80.73434
    3400            -0.59282             0.00098             0.00204             0.00000             0.00000            59.85952             0.07995            -0.06520            -6.64625            92.02062
    3500            -0.66284             0.00439             0.01982             0.00000             0.00000            55.47646             0.08253            -0.06601            -6.71298            80.42178
    3600            -0.54850             0.00195             0.00000             0.00100             0.00146            56.88298             0.08447            -0.06573            -6.77813            81.97134
    3700            -0.55937             0.00293             0.00875             0.00201             0.00146            57.09545             0.08374            -0.06623            -6.68599            86.39119
    3800            -0.35422             0.00293             0.00414             0.00096             0.00049            58.48903             0.08481            -0.06394            -6.72468           107.10027
    3900            -0.50758             0.00342             0.00445             0.00292             0.00146            56.09860             0.08467            -0.06389            -6.64226            69.56477
    3990            -0.52222             0.00244             0.00206             0.00207             0.00049            57.23080             0.08009            -0.06675            -6.72885            65.40033
```

**Note:** Seeds 0 and 2 share `15-51-38` Hydra dir → bit-identical traces. The trace above
represents both; they interfered during training and are not independent data.

---

## Seed 1 (s1cen3_s1, RUNTAG job 3274994) — CONVERGED (1/2 independent runs)

```
[inc8-tb-trace] TRAJECTORY (every ~100 updates):
    step        total_reward       pointing_rate   terminal_pointing   lockband_pointing            fix_rate     band_az_abs_deg      estim_err_ip_m           centering             entropy          value_loss
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
       0            -0.25701             0.00000             0.00000             0.00000             0.00049             0.00000             0.14807            -0.27440             0.35368            24.67297
     100            -1.00617             0.00342             0.00000             0.00000             0.00049            61.26880             0.10665            -0.16614             0.63358             2.65911
     200            -0.79964             0.00000             0.00000             0.00000             0.00000            60.04802             0.08141            -0.14759             2.08618             1.79705
     300            -0.77470             0.00049             0.00000             0.00000             0.00049            57.35450             0.07282            -0.13683             2.32331             1.34228
     400            -0.58240             0.00098             0.00000             0.00279             0.00098            49.27190             0.12892            -0.13190             2.19095             2.22818
     500            -0.47624             0.02295             0.00000             0.06528             0.01855            51.69516             0.13000            -0.13700             2.61692             1.96365
     600            -0.60034             0.01758             0.00000             0.05172             0.01562            58.39261             0.12763            -0.14793             2.13988             3.17933
     700            -0.59473             0.02832             0.00000             0.05587             0.01660            49.87455             0.13356            -0.14771             2.21301             2.22135
     800            -0.46386             0.04980             0.00000             0.14164             0.03809            42.16996             0.12468            -0.13856             2.16155             2.31809
     900            -0.53582             0.04834             0.00000             0.12930             0.03711            24.67748             0.13955            -0.14632             0.81413             5.46428
    1000            -0.54689             0.06738             0.00000             0.16866             0.05176            27.01180             0.12694            -0.13932             0.76444             5.47044
    1100            -0.56473             0.08350             0.00000             0.19638             0.06104            27.36309             0.13211            -0.14500             1.37586             2.89150
    1200            -0.55932             0.10059             0.00000             0.24215             0.07031            23.22440             0.13253            -0.14895             1.60392             2.14058
    1300            -0.61625             0.09766             0.00000             0.25000             0.06982            28.20200             0.13941            -0.15521             1.14229            16.87735
    1400            -0.71795             0.09229             0.00000             0.20569             0.06494            27.27009             0.13878            -0.14949             1.13365             2.18203
    1500            -0.54498             0.09424             0.00000             0.20779             0.06592            33.65564             0.13193            -0.14747             1.22583             2.13848
    1600            -0.61772             0.06738             0.00000             0.17722             0.05176            32.06545             0.13736            -0.15827             0.99863             4.43356
    1700            -0.62230             0.10645             0.00000             0.25644             0.07910            31.34319             0.13869            -0.14744             0.44818             5.08958
    1800            -0.57135             0.13428             0.00134             0.28386             0.09521            22.24634             0.14916            -0.14441            -0.29214            61.60349
    1900            -0.55196             0.10693             0.00000             0.25692             0.07715            25.14869             0.13882            -0.15738             0.37713             2.61149
    2000            -0.56553             0.16895             0.00000             0.34238             0.11914            24.32321             0.14283            -0.14144             0.39776             2.53524
    2100            -0.65864             0.12988             0.00000             0.30072             0.09570            26.62136             0.14463            -0.15439             0.15684             3.28194
    2200            -0.63622             0.13037             0.00000             0.28938             0.09863            29.35240             0.14659            -0.14364            -0.18100             4.40934
    2300            -0.67801             0.11670             0.00000             0.26625             0.08496            25.01771             0.13937            -0.14914            -0.00347             2.67874
    2400            -0.53269             0.14404             0.00000             0.32310             0.10449            24.60733             0.14300            -0.14708            -0.03583             2.33799
    2500            -0.61205             0.12500             0.00000             0.29569             0.09570            20.06273             0.14174            -0.15127            -0.46370             3.11305
    2600            -0.53668             0.11279             0.00000             0.27403             0.08545            20.01856             0.14004            -0.14643            -0.88082             3.70485
    2700            -0.52958             0.14062             0.00000             0.31039             0.10059            23.64643             0.14329            -0.13935            -0.92154             2.27328
    2800            -0.55466             0.13135             0.00000             0.29630             0.09521            24.36532             0.14408            -0.15088            -0.75429             2.25475
    2900            -0.66775             0.11279             0.00000             0.25944             0.08350            25.86631             0.14487            -0.14888            -1.10921             4.35746
    3000            -0.53548             0.13330             0.00000             0.29044             0.09375            23.48270             0.14096            -0.14320            -0.87959             2.93239
    3100            -0.59680             0.14111             0.00000             0.30519             0.10645            23.46511             0.14380            -0.14181            -0.90772             3.00845
    3200            -0.52151             0.12451             0.00000             0.25718             0.08984            27.81661             0.13983            -0.14412            -1.08697             3.50668
    3300            -0.56802             0.14258             0.00000             0.30535             0.09912            24.24889             0.14074            -0.14846            -0.85031             2.92912
    3400            -0.66351             0.11182             0.00000             0.26263             0.08691            29.24267             0.14053            -0.15317            -0.79759             4.64318
    3500            -0.61220             0.13037             0.00000             0.27675             0.09521            29.26943             0.14272            -0.14347            -0.87918             4.13860
    3600            -0.60428             0.14355             0.00000             0.30116             0.09766            26.37127             0.14069            -0.14341            -0.69147             3.55045
    3700            -0.44307             0.14258             0.00000             0.30070             0.10352            29.71431             0.13870            -0.13786            -0.69083             2.95135
    3800            -0.51811             0.13330             0.00000             0.28878             0.08936            28.55441             0.14229            -0.14365            -0.92458             2.29258
    3900            -0.65607             0.14307             0.00000             0.29569             0.10498            29.78243             0.14251            -0.14017            -0.71577             2.48123
    3990            -0.54280             0.13379             0.00000             0.30440             0.09863            26.63991             0.14161            -0.14952            -0.86503             3.63311
```

---

## Seed 2 (s1cen3_s2, RUNTAG job 3274995) — COLLAPSED / bit-identical to seed 0

Traces are bit-for-bit identical to seed 0 — Hydra dir collision means both jobs trained the same
model. Not an independent data point. Omitted (see seed 0 trace above).

---

## Cross-Seed Analysis

### Hydra dir collision (structural bug)

Seeds 0 (job 3274993) and 2 (job 3274995) launched at the same second (`15:51:38`) and wrote to
the same Hydra output directory. Their SLURM stdout captures are identical. The training itself was
corrupted — two processes simultaneously updating the same checkpoint files. This experiment
provides **2 independent data points**, not 3.

**Fix for future submissions**: add `hydra.run.dir=...${SEED}` to the Hydra override string, or
ensure per-seed output paths are distinct (RUNTAG already disambiguates the out-file, but the
Hydra timestamped dir is shared when start times collide).

### Sign gate

- Collapsed pair: band_az 40° at step 100, rises to 54–61° — stuck high. Sign = CORRECT (−3.0)
  but policy collapsed before pointing developed.
- Seed 1: band_az 61° at step 100, falls to 24° by step 900, stays 20–30° through step 3990.
  Pointing ESTABLISHED. Sign confirmed correct.

### Value-loss / entropy dynamics

| Run        | Step-100 value_loss | Step-100 entropy | Outcome           |
|------------|--------------------:|----------------:|-------------------|
| S1@1.0 s0  | 189                 | +0.41           | converged (2700)  |
| S1@3.0 s0+2| **715**             | −0.68           | instant collapse  |
| S1@3.0 s1  | 2.66                | +0.63           | converged (900)   |

Rw_centering=3.0 is CATASTROPHICALLY UNSTABLE in 1/2 independent runs. The 3× larger
centering gradient causes the PPO value function to blow up at the first update for any seed
that doesn't happen to have a lucky initial rollout that avoids large `err_ip` before pointing
is established.

### Converged-seed performance vs baseline

| Metric              | S1@1.0 seed 0 (step 3990) | S1@3.0 seed 1 (step 3990) |
|---------------------|:--------------------------|:--------------------------|
| band_az_abs_deg     | 17–35°                    | 20–30°                    |
| fix_rate            | 0.14–0.16                 | 0.09–0.12                 |
| lockband_pointing   | 0.50–0.55                 | 0.28–0.34                 |
| terminal_pointing   | 0.000                     | 0.000                     |
| estim_err_ip_m      | 0.115                     | 0.13–0.15                 |
| centering (active)  | −0.04 to −0.09            | −0.13 to −0.16            |

rw_centering=3.0 performs **worse** than 1.0 on the converging seed across all pointing and
estimator metrics. The stronger centering penalty did not improve fix_rate or reduce estim_err
— it made the policy optimize for approaching near center-line at the cost of boresight alignment.

### estim_err GO gate

**Target: estim_err_ip_m ≲ 0.08 on ≥1 converged seed.** Neither run meets the gate:
- S1@1.0: 0.115 (best single-seed average across 900–3990)
- S1@3.0: 0.13–0.15 (seed 1, converged)

Collapsed seeds' estim_err values (0.077–0.091) are **artifacts** — fix_rate ≈ 0 means the
KF is dead-reckoning, and the GT in-plane error happens to be small for the non-pointing policy's
natural approach trajectory. These do NOT represent successful estimation with fixes.

### Why centering alone can't close the gate

band_az at ~24-30° gives azimuth pointing, which yields fix_rate 6–12% — enough to partially
constrain the KF but not enough to drive σ_p0 below 0.08 m. `terminal_pointing = 0.000` in
every seed of every run: the policy never achieves the tight boresight (<few°) needed for the
accept_rlo=12 m PnP gate to fire in terminal approach. The missing axis is **elevation** (pitch).

Per DESIGN.md: "azimuth-corrects-but-fix-flat = S2 (+g_pitch, the 20° VFoV gap)". The fix_rate
at ~10% is not quite "flat" — the primitive is helping — but it is NOT CLOSING the gate. With
only yaw pointing, the camera sweeps through azimuth but may miss the gate in elevation at
terminal approach. S2 must add the pitch axis.

---

## Commander Adjudication

**S1@rw_centering=3.0 = NOT-GO.** All three failure modes present:

1. **Stability**: 1/2 independent runs collapsed catastrophically at step 100 (value_loss 715).
2. **Performance**: The converging run (seed 1) performs WORSE than S1@1.0 on all metrics.
3. **estim_err gate**: Not met by any run at any weight level (0.115 at 1.0, 0.13–0.15 at 3.0).

**rw_centering sweep CLOSED.** Increasing weight hurts rather than helps. The floor is not a
weight problem — it is an architecture problem (yaw-only → no elevation fix → terminal_pointing
never fires → σ_p0 saturates at ~0.11–0.15).

**NEXT STEP = S2 (pitch axis primitive).**

Decision per DESIGN.md hand-off rules: "azimuth-corrects-but-fix-flat → S2". Confirmed. The
look-at primitive needs the second axis. S2 adds `g_pitch` to the look-at reward, targeting the
vertical component of boresight alignment and closing the remaining 20° VFoV gap.

**gain-warmup** (ramp centering from 0 over first K steps) could cure the step-100 instability
in S2 if centering is used there. Recommend starting S2 WITHOUT centering (centering = distractor
until 2-axis pointing is established) and adding centering only after 2-axis convergence is
confirmed.

**Hydra dir collision fix**: add explicit per-seed output dir override to sbatch EXTRA
(e.g., `hydra.run.dir=/scratch/.../seed${SEED}`) so simultaneous job starts can't collide.

---

## MEMORY-DELTA (≤10 lines, for commander)

```
S1@rw_centering=3.0 DONE (2 independent runs due to Hydra dir collision on seeds 0+2):
  - Seed 0+2 COLLAPSED: value_loss 715 step-100, entropy −0.68, band_az stuck 54–61°
  - Seed 1 CONVERGED: band_az 24–30°, fix_rate 9–12%, lockband 0.28–0.34
  - estim_err_ip_m = 0.13–0.15 (converged seed) — WORSE than S1@1.0's 0.115, NOT-GO
  - rw_centering=3.0 WORSE than 1.0 on all metrics; weight sweep CLOSED
  - terminal_pointing = 0.000 ALL seeds ALL runs → elevation miss confirmed
  - ROOT CAUSE: yaw-only → no elevation fix → σ_p0 saturates ~0.11–0.15 regardless of weight
DECISION = S2 (pitch axis primitive); do NOT add centering until 2-axis converges
FOOTGUN: Hydra dir collision when 2 seeds start at same second → add per-seed dir override
  to sbatch EXTRA: e.g. ++hydra.run.dir=/scratch/network/fl3689/diffaero/outputs/train/seed${SEED}
```
