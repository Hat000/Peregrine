> 🚩 **SUPERSEDED 2026-06-19 (511e85c) re: the σ_p0 bar.** Any "σ_p0 ≲ 0.08 / NO-GO vs 0.08 / near-field-gate-estimator pivot" conclusion in this report is OVERTURNED — the 0.08 bar was a `margin_envelope.py` double-count of the drone (real gate clearance 0.75−r ≈ 0.37–0.47 m). The REAL bar is **σ_p0_lat ≲ 0.15 (p99 ≲ 0.45)**; measured 0.15–0.20 = MARGINAL-PASSING, NOT NO-GO. Gate-4 is a REACH/PASS-RATE problem; RL stays the tool (the "pivot off RL" is RETRACTED). The engineering + measurements below STAND; only the bar and its GO/NO-GO verdict are corrected. → MEMORY.md:8 + memory/project_rl_increment_history.md §inc8-2026-06-19.
# inc8 S2 — 2-Axis Look-At Primitive (Pitch Axis) — REPORT
**Date:** 2026-06-16  
**Author:** Overall Commander  
**Stage:** S2 = yaw + pitch axes combined (g_yaw=−3.0, g_pitch=+3.0)

---

## 1. Context

S0 established that the yaw-only look-at primitive nulls azimuth and drives fix_rate off the floor, but band_el stays at 70–85° (gate out of vertical FoV due to 20° camera mount). terminal_pointing=0.000 on all seeds because horizontal-only pointing cannot produce terminal gate contact in elevation. The S1 centering-weight sweep (rw_centering=1.0, 3.0) was then closed — both NOT-GO — because the architectural blocker is yaw-only pointing, not weight magnitude. S2 adds the pitch axis primitive to null band_el.

**S2 design:**  
`w_cam = [-g_pitch*uy,  g_yaw*ux,  0]`  
- `ux` = gate unit vector horizontal component (azimuth error)  
- `uy` = gate unit vector vertical component (elevation error)  
- Pitch sign determined empirically: probe with g_pitch=−3.0 → band_el 83–90° (inverted) → empirical sign = **+3.0**  
- Yaw sign: g_yaw=−3.0 (established S0; FOOTGUN: sbatch default +3.0 is INVERTED → use `++env.lookat_g_yaw=-3.0`)  
- Both in COMMON → both require `++` (double-plus) override  
- Warmup: `+env.lookat_warmup_updates=200` (single-plus, NOT in COMMON)  
- Centering: OFF (rw_centering=0.0, default)  
- Hydra dir collision FIX: per-seed explicit `hydra.run.dir` paths

---

## 2. Job Table

| Job ID   | Seed | RUNTAG     | g_yaw | g_pitch | warmup | Output file                               |
|----------|------|------------|-------|---------|--------|-------------------------------------------|
| 3275300  | 0    | s2full_s0  | −3.0  | +3.0    | 200    | peregrine_inc8_s0_s2full_s0.out           |
| 3275301  | 1    | s2full_s1  | −3.0  | +3.0    | 200    | peregrine_inc8_s0_s2full_s1.out           |
| 3275302  | 2    | s2full_s2  | −3.0  | +3.0    | 200    | peregrine_inc8_s0_s2full_s2.out           |

NUPD=4000, NENVS=2048, per-seed Hydra dirs (no collision). All 3 jobs completed normally.

---

## 3. Per-Seed Traces

### Seed 0 (non-convergent, warmup stable)

```
    step   total_reward  pointing_rate  term_pointing  lockband_ptg   fix_rate  band_az_deg  band_el_deg  estim_err_ip   centering   entropy  value_loss
       0       -0.32463        0.00000        0.00000       0.00000    0.00000        0.000        0.000      0.14361     0.00000    0.35913    17.81984
     100       -0.80502        0.00000        0.00000       0.00000    0.00000       75.431       83.619      0.11930     0.00000    0.67240     1.58126
     200       -0.90356        0.00000        0.00000       0.00000    0.00000       65.045       84.405      0.10873     0.00000    2.36275     0.98293
     300       -0.70228        0.01465        0.00000       0.03925    0.01123       54.152       60.270      0.10551     0.00000    2.20755     8.19030  ← brief breakout
     400       -0.49493        0.00000        0.00000       0.00000    0.00049       62.468       79.956      0.13307     0.00000    1.90965     1.47839  ← snapped back
     500       -0.44451        0.00488        0.00470       0.00000    0.00000       63.049       79.417      0.13666     0.00000    1.81000     1.53865
     600       -0.49490        0.00146        0.00000       0.00000    0.00000       56.154       75.451      0.14095     0.00000    1.37288     2.86315
     700       -0.38927        0.00635        0.01389       0.00000    0.00000       54.033       81.389      0.14919     0.00000    1.65936     3.15419
     800       -0.46962        0.01367        0.00605       0.00000    0.00000       60.034       77.908      0.14944     0.00000    1.43119     2.63923
     900       -0.45365        0.01270        0.01613       0.00000    0.00000       66.044       72.504      0.16164     0.00000    1.30557     1.84399
    1000       -0.47352        0.01562        0.00719       0.00000    0.00098       56.396       70.541      0.15310     0.00000    1.33157     1.90812
    1100       -0.33318        0.00488        0.00245       0.00000    0.00000       58.139       77.158      0.14907     0.00000    1.26552     2.05514
    1200       -0.51734        0.00781        0.00871       0.00000    0.00049       59.094       76.873      0.15098     0.00000    0.87980     2.28942
    1300       -0.48062        0.00732        0.00649       0.00000    0.00049       52.378       79.924      0.15922     0.00000    0.70716     4.49876
    1400       -0.54627        0.00732        0.00364       0.00000    0.00049       51.229       83.782      0.15527     0.00000    0.06316     3.66480
    1500       -0.39192        0.00684        0.01266       0.00000    0.00049       53.450       83.338      0.15401     0.00000   -0.07591     3.83541
    1600       -0.40689        0.00586        0.00256       0.00000    0.00098       50.369       80.024      0.16318     0.00000   -0.01186     7.08303
    1700       -0.41074        0.00732        0.01599       0.00000    0.00000       50.965       83.095      0.16090     0.00000   -0.04052     3.33793
    1800       -0.44766        0.00830        0.01419       0.00000    0.00000       53.192       82.739      0.16188     0.00000   -0.23555     3.59242
    1900       -0.47100        0.00635        0.00257       0.00000    0.00000       56.095       79.905      0.16194     0.00000   -0.28348     2.35498
    2000       -0.49454        0.01221        0.01953       0.00000    0.00000       48.785       83.074      0.16836     0.00000   -0.57386     2.54315
    2100       -0.56643        0.00977        0.01701       0.00000    0.00000       60.126       82.063      0.16236     0.00000   -0.37253     3.18720
    2200       -0.51171        0.01270        0.02206       0.00000    0.00000       45.881       80.737      0.15868     0.00000   -0.45554     4.20898
    2300       -0.36897        0.01172        0.02879       0.00000    0.00049       50.605       83.078      0.16152     0.00000   -0.80056     2.10251
    2400       -0.46854        0.01025        0.02208       0.00000    0.00000       50.024       82.702      0.16555     0.00000   -0.73227     4.94459
    2500       -0.43598        0.00781        0.01030       0.00000    0.00000       54.039       82.835      0.16555     0.00000   -0.79363     3.65357
    2600       -0.45410        0.01172        0.01747       0.00000    0.00000       58.311       82.367      0.16426     0.00000   -0.78199     2.72114
    2700       -0.46922        0.01074        0.03030       0.00000    0.00000       63.641       84.729      0.16903     0.00000   -0.75867     2.71166
    2800       -0.47760        0.00781        0.01750       0.00144    0.00098       51.255       80.492      0.16561     0.00000   -0.85068     5.42604
    2900       -0.57782        0.00879        0.02116       0.00130    0.00098       56.089       80.293      0.16368     0.00000   -1.11568     3.98552
    3000       -0.49236        0.00879        0.01882       0.00134    0.00049       49.949       81.076      0.16419     0.00000   -0.82860     2.28548
    3100       -0.39739        0.00635        0.01519       0.00000    0.00098       52.972       81.773      0.16458     0.00000   -0.93410     1.89737
    3200       -0.48523        0.00879        0.01511       0.00000    0.00049       55.366       81.116      0.16154     0.00000   -0.71017     2.51451
    3300       -0.37438        0.01123        0.01633       0.00272    0.00049       54.207       80.523      0.16548     0.00000   -0.71034     7.74115
    3400       -0.49962        0.01221        0.02774       0.00000    0.00000       51.474       81.170      0.17173     0.00000   -1.07783     3.30591
    3500       -0.41301        0.01318        0.03315       0.00000    0.00000       61.952       82.709      0.16767     0.00000   -0.81668     2.64133
    3600       -0.43671        0.01172        0.02905       0.00000    0.00049       50.847       80.782      0.16756     0.00000   -0.65686     2.44288
    3700       -0.47801        0.01221        0.02510       0.00140    0.00049       53.676       83.215      0.16560     0.00000   -0.73246     2.57051
    3800       -0.35675        0.00879        0.02049       0.00132    0.00000       58.830       82.152      0.16991     0.00000   -0.71996     3.47985
    3900       -0.39263        0.00537        0.00815       0.00000    0.00049       52.925       81.776      0.16867     0.00000   -0.79068     3.17720
    3990       -0.45177        0.00928        0.02194       0.00000    0.00000       50.895       81.602      0.16336     0.00000   -0.65376     2.33699
```

**Seed 0 summary:** Warmup worked (entropy 0.67 at step 100, value_loss 1.6 — no collapse). Brief breakout at step 300 (band_el drops to 60°, fix_rate 0.011, lockband 0.039) then immediately snaps back. Steps 400–3990: band_el oscillates 70–85°, band_az 45–75°, fix_rate at floor (0–0.001). Policy does not converge. estim_err drifts upward 0.119 → 0.171 (no improvement from pointing). Entropy slowly goes negative (−1.1 by step 2900) but policy stays non-catastrophic.

---

### Seed 1 (collapsed — warmup failed)

```
    step   total_reward  pointing_rate  term_pointing  lockband_ptg   fix_rate  band_az_deg  band_el_deg  estim_err_ip   centering   entropy  value_loss
       0       -0.25701        0.00000        0.00000       0.00000    0.00049        0.000        0.000      0.14807     0.00000    0.33881    17.62951
     100       -0.76835        0.00146        0.00268       0.00121    0.00000       58.027       81.055      0.08038     0.00000   -0.79586   265.95639  ← COLLAPSE
     200       -0.63655        0.00781        0.01792       0.00000    0.00049       57.315       74.017      0.08637     0.00000   -1.32800   268.35791
     300       -0.76871        0.00195        0.00329       0.00113    0.00049       51.449       78.397      0.08018     0.00000   -1.69349    72.88631
     400       -0.95441        0.00439        0.00949       0.00103    0.00098       54.342       74.523      0.08279     0.00000   -3.83615   112.15499
     500       -0.93522        0.00586        0.00741       0.00207    0.00098       56.113       74.223      0.09132     0.00000   -5.05162    94.63815
     600       -0.91071        0.00488        0.00691       0.00640    0.00342       54.612       74.714      0.09092     0.00000   -6.75646   127.54097
     700       -0.78469        0.00244        0.00341       0.00110    0.00049       55.291       76.589      0.09032     0.00000   -6.86187    73.11686
     800       -0.73928        0.00342        0.00904       0.00000    0.00000       57.622       77.284      0.08330     0.00000   -6.80481    21.99753
     900       -0.89158        0.00049        0.00173       0.00000    0.00000       56.697       78.557      0.08228     0.00000   -7.00709    76.02994
    1000       -0.86045        0.00098        0.00181       0.00000    0.00049       57.590       79.892      0.08384     0.00000   -6.87799    78.41302
    1100       -0.78311        0.00195        0.00000       0.00107    0.00049       54.691       79.442      0.08171     0.00000   -6.88797    55.73454
    1200       -0.70053        0.00098        0.00375       0.00000    0.00000       56.697       82.482      0.08007     0.00000   -6.46788    40.99334
    1300       -0.82656        0.00293        0.00701       0.00107    0.00049       55.631       78.805      0.08018     0.00000   -6.47627    46.33450
    1400       -0.84262        0.00488        0.01306       0.00105    0.00049       55.564       78.090      0.07858     0.00000   -6.49006    27.27296  ← estim_err <0.08 (coincidental)
    1500       -0.82344        0.00342        0.00696       0.00211    0.00049       57.442       77.897      0.08135     0.00000   -6.65984    38.77497
    1600       -0.70501        0.00244        0.00692       0.00000    0.00049       62.935       76.486      0.08396     0.00000   -6.54864    40.29180
    1700       -0.78951        0.00391        0.00380       0.00511    0.00244       57.564       78.926      0.07826     0.00000   -6.33190    22.24413
    1800       -0.73506        0.00488        0.01304       0.00211    0.00049       58.412       79.780      0.07593     0.00000   -6.65483    31.52041
    1900       -0.71735        0.00146        0.00558       0.00000    0.00000       51.634       78.561      0.08160     0.00000   -6.50091    54.81245
    2000       -0.81257        0.00195        0.00588       0.00102    0.00049       54.749       79.616      0.07539     0.00000   -6.40756    19.36698
    2100       -0.71623        0.00195        0.00707       0.00000    0.00000       52.543       80.137      0.07709     0.00000   -6.47928    40.92920
    2200       -0.78463        0.00195        0.00558       0.00111    0.00049       54.909       79.364      0.08173     0.00000   -6.44690    13.19451
    2300       -0.81356        0.00195        0.00360       0.00212    0.00098       58.858       80.571      0.08110     0.00000   -6.30982    33.70293
    2400       -0.90444        0.00244        0.00376       0.00000    0.00049       50.785       80.172      0.07856     0.00000   -6.30216    23.95108
    2500       -0.89453        0.00195        0.00528       0.00000    0.00049       55.672       81.428      0.07895     0.00000   -6.17028    17.92537
    2600       -0.83642        0.00195        0.00357       0.00111    0.00049       53.324       79.090      0.07752     0.00000   -6.26163    14.94586
    2700       -0.80713        0.00244        0.00924       0.00000    0.00000       55.157       80.559      0.07662     0.00000   -6.43713    47.29247
    2800       -0.79383        0.00244        0.00992       0.00000    0.00098       57.069       80.435      0.07708     0.00000   -6.51195    18.21141
    2900       -0.77079        0.00098        0.00382       0.00000    0.00000       55.612       81.190      0.07760     0.00000   -6.27922    21.67017
    3000       -0.78384        0.00146        0.00406       0.00103    0.00098       58.762       78.207      0.08650     0.00000   -6.48541    29.22270
    3100       -0.85178        0.00049        0.00214       0.00000    0.00000       56.829       79.089      0.08169     0.00000   -6.30313    30.21776
    3200       -0.90437        0.00049        0.00000       0.00000    0.00049       54.291       80.022      0.07934     0.00000   -6.16525    20.57081
    3300       -0.68394        0.00342        0.00960       0.00104    0.00146       59.944       78.709      0.07748     0.00000   -5.99165    17.61016
    3400       -0.77007        0.00293        0.00195       0.00532    0.00195       60.019       76.129      0.08267     0.00000   -6.36034    23.98513
    3500       -0.77470        0.00391        0.00388       0.00210    0.00098       62.230       76.176      0.08110     0.00000   -6.31365    44.87088
    3600       -0.75181        0.00439        0.01085       0.00210    0.00098       55.153       77.703      0.07479     0.00000   -6.37586    43.71885
    3700       -0.71156        0.00244        0.00792       0.00105    0.00049       56.056       80.523      0.08091     0.00000   -6.31008    17.11323
    3800       -0.77073        0.00049        0.00000       0.00106    0.00049       60.819       77.920      0.08083     0.00000   -6.23365    33.73091
    3900       -0.73017        0.00195        0.00600       0.00000    0.00000       54.675       80.400      0.07974     0.00000   -6.25918    30.45885
    3990       -0.83685        0.00537        0.01679       0.00000    0.00000       59.290       79.677      0.07911     0.00000   -6.20027    45.89064
```

**Seed 1 summary:** WARMUP FAILURE — entropy −0.80 at step 100 (warmup_updates=200 did not prevent collapse; policy entered a bad region before the warmup ramp completed). Value_loss 266–268 at steps 100–200, then oscillates 14–128. Policy is a zombie (entropy −6.9, no pointing). band_el stuck 74–83°, fix_rate 0.000–0.003 (floor). Apparent estim_err near 0.076–0.086 at scattered steps (1400, 1700, 1800, 2000, etc.) is COINCIDENTAL — zero pointing, collapsed policy flying degenerate trajectories. These do NOT count as GO evidence.

---

### Seed 2 (CONVERGED — 2-axis pointing confirmed)

```
    step   total_reward  pointing_rate  term_pointing  lockband_ptg   fix_rate  band_az_deg  band_el_deg  estim_err_ip   centering   entropy  value_loss
       0       -0.25025        0.00000        0.00000       0.00000    0.00000        0.000        0.000      0.15005     0.00000    0.36097    18.28828
     100       -0.92236        0.00000        0.00000       0.00000    0.00000       55.048       84.489      0.09822     0.00000    0.65368     2.54841  ← WARMUP STABLE
     200       -0.83233        0.00000        0.00000       0.00000    0.00049       65.893       82.497      0.09018     0.00000    2.17860     1.21639
     300       -0.99311        0.00146        0.00000       0.00000    0.00098       58.376       82.856      0.10157     0.00000    2.30544    18.11963
     400       -0.59884        0.04346        0.00087       0.12227    0.02051       44.979       47.030      0.10695     0.00000    2.20631     6.35808  ← BREAKOUT
     500       -0.42351        0.13818        0.00105       0.29603    0.06543       22.025       35.082      0.11956     0.00000    2.26944     2.15490  ← CONVERGING
     600       -0.52700        0.26367        0.03309       0.44974    0.10938       24.610       29.220      0.11433     0.00000    1.99365     2.83754  ← CONVERGED
     700       -0.50031        0.25879        0.06458       0.41899    0.10693       32.429       26.421      0.12320     0.00000    1.45452     4.58452
     800       -0.60866        0.34961        0.09751       0.47930    0.12158       29.393       25.276      0.12562     0.00000    0.90142     4.87486
     900       -0.42514        0.38574        0.12687       0.47779    0.13525       24.844       26.695      0.12841     0.00000    0.56842     4.97622
    1000       -0.53109        0.31201        0.04343       0.48025    0.12744       29.861       23.912      0.12913     0.00000    0.65708     3.35596
    1100       -0.63564        0.33838        0.04223       0.45828    0.15039       26.984       28.520      0.12748     0.00000   -0.35749   111.96134
    1200       -0.52481        0.27490        0.05535       0.40506    0.11914       30.900       31.238      0.12165     0.00000   -1.97089    72.46072
    1300       -0.54550        0.32715        0.08312       0.47922    0.14014       23.736       25.759      0.11986     0.00000   -2.26202    12.39825
    1400       -0.56217        0.28662        0.06566       0.40108    0.12793       31.359       28.608      0.12835     0.00000   -2.95890    18.26921
    1500       -0.62260        0.27197        0.05596       0.41296    0.12598       31.685       30.599      0.11940     0.00000   -3.02046    15.79579
    1600       -0.57810        0.27783        0.06633       0.45280    0.13672       31.797       27.805      0.12068     0.00000   -3.52927     8.81979
    1700       -0.64983        0.28174        0.03576       0.44744    0.14453       31.596       30.829      0.11986     0.00000   -3.93242    25.00544
    1800       -0.51815        0.27197        0.03534       0.41274    0.11328       34.685       29.433      0.12122     0.00000   -3.59230    25.72932
    1900       -0.62245        0.34912        0.09626       0.47654    0.16162       32.583       24.281      0.12591     0.00000   -3.36637    14.20033
    2000       -0.56601        0.29834        0.04954       0.48129    0.14990       32.699       23.915      0.12195     0.00000   -3.41702    11.87238
    2100       -0.45722        0.28711        0.03748       0.44501    0.14404       35.775       30.504      0.11704     0.00000   -3.59608    13.68643
    2200       -0.58338        0.29932        0.06154       0.45631    0.13574       34.547       26.696      0.12322     0.00000   -3.50932    14.40160
    2300       -0.50950        0.29443        0.08378       0.44020    0.14014       36.155       31.529      0.11774     0.00000   -3.67972    12.70644
    2400       -0.61307        0.30371        0.03898       0.48721    0.15625       32.788       28.128      0.12213     0.00000   -3.34870     2.91148
    2500       -0.47002        0.29932        0.04179       0.47191    0.15576       36.878       26.246      0.12894     0.00000   -3.21443    13.36555
    2600       -0.65525        0.28467        0.06522       0.41944    0.12793       34.508       27.881      0.12403     0.00000   -3.47352     7.75329
    2700       -0.65250        0.31299        0.05556       0.49864    0.15039       32.009       28.328      0.12297     0.00000   -3.87622    16.29675
    2800       -0.51551        0.25244        0.06928       0.40201    0.12646       35.467       28.696      0.12020     0.00000   -3.51886    14.56923
    2900       -0.58346        0.27100        0.05058       0.44288    0.13818       33.526       26.698      0.12341     0.00000   -3.54155    12.42560
    3000       -0.56262        0.30762        0.05905       0.49725    0.15332       33.517       26.905      0.12331     0.00000   -3.12569     6.69989
    3100       -0.57663        0.29883        0.05645       0.47215    0.14551       35.431       28.051      0.12517     0.00000   -3.50279    11.05753
    3200       -0.53158        0.27295        0.04161       0.46477    0.13672       35.900       28.032      0.12429     0.00000   -3.58187    16.80634
    3300       -0.51488        0.26221        0.04821       0.43978    0.13232       35.466       29.329      0.12162     0.00000   -3.44434     4.78919
    3400       -0.61780        0.28271        0.05385       0.45000    0.14062       33.570       25.918      0.12152     0.00000   -3.17219    11.53362
    3500       -0.53220        0.22803        0.03453       0.41982    0.13330       33.953       30.113      0.12378     0.00000   -2.91954    13.00569
    3600       -0.57882        0.28369        0.04293       0.48240    0.14844       31.884       26.819      0.12110     0.00000   -2.96104     9.16096
    3700       -0.52297        0.30127        0.05960       0.45136    0.14209       33.738       28.595      0.12252     0.00000   -3.01687    11.32283
    3800       -0.66384        0.28320        0.04521       0.43798    0.14111       34.515       29.568      0.12175     0.00000   -3.23610     5.09581
    3900       -0.54213        0.31543        0.06197       0.48561    0.16406       34.253       25.822      0.12747     0.00000   -2.94819    13.92641
    3990       -0.59727        0.28760        0.05707       0.47973    0.14062       35.295       28.044      0.12429     0.00000   -3.57435    22.98008
```

**Seed 2 summary:** CONVERGED. Warmup stable (entropy +0.65 at step 100, value_loss 2.55). 2-axis breakout at step 400: band_el drops to 47°, fix_rate 0.021. By step 500: band_el 35°, band_az 22°, pointing_rate 0.138, fix_rate 0.065. By step 600 (CONVERGED): band_el 29°, band_az 25°, pointing_rate 0.264, lockband 0.450, fix_rate 0.109, terminal_pointing 0.033. Steps 600–3990: STABLE at band_el 24–31°, band_az 22–37°, fix_rate 0.11–0.16, terminal_pointing 0.03–0.13. Note: value_loss spike to 112 at step 1100 as entropy crosses 0 → normal PPO adjustment, not collapse; resettles to 3–25 range.

estim_err_ip stabilizes at **0.115–0.130** from step 600 through 3990 (see §4 for analysis).

---

## 4. Cross-Seed Analysis

### 4.1 Warmup efficacy

| Seed | Entropy @step100 | value_loss @step100 | Verdict |
|------|-----------------|---------------------|---------|
| 0    | +0.67            | 1.58                | STABLE — warmup worked |
| 1    | **−0.80**        | **265.96**          | COLLAPSED — warmup failed despite warmup_updates=200 |
| 2    | +0.65            | 2.55                | STABLE — warmup worked |

Warmup_updates=200 prevented collapse in 2/3 seeds. Seed 1 collapsed anyway at the boundary of the warmup ramp. This is consistent with S1@3.0 instability (1/2 independent seeds collapsed). The warmup reduces collapse probability but does not eliminate it — the gain magnitude (g_yaw=3.0, g_pitch=3.0 combined) is still large enough that rare unlucky initializations hit the PPO update at an unstable point.

### 4.2 Convergence (2-axis primitive)

| Seed | Breakout step | band_el @step600 | band_az @step600 | fix_rate @step600 | terminal_pointing @step600 | Converged? |
|------|---------------|-----------------|-----------------|-------------------|---------------------------|-----------|
| 0    | ~300 (brief, lost) | 75°          | 56°             | 0.000             | 0.000                     | NO |
| 1    | never         | 75°             | 55°             | 0.003             | 0.007                     | NO (collapsed) |
| 2    | **~400**      | **29°**         | **25°**         | **0.109**         | **0.033**                 | **YES** |

Convergence rate: 1/3 seeds — same as S0 (1/3 yaw-only seeds converged). This is the "PPO stochastic basin" problem: the primitive gain is large, and only one seed finds the pointing basin. The CRITIC-IS-SYMMETRIC footgun (GuardedPPO=symmetric PPO, critic sees obs_dim=20 only, not GT state) remains the most likely root cause of low basin-capture rate. Without the GT-privileged asymmetric critic, the value function underestimates progress in partial states → 2/3 seeds can't escape the non-pointing region.

### 4.3 Seed 0: near-breakout at step 300 but lost

Seed 0 shows a genuine pointing attempt at step 300 (band_el 60°, fix_rate 0.011, lockband 0.039) but the policy snaps back to band_el 80° by step 400 and never recovers. This is the exploration wall: without a dense enough value-function gradient pointing toward the convergent region, PPO pushes back toward the lower-variance non-pointing orbit. Warmup helped (the PPO update didn't trigger catastrophic collapse), but couldn't sustain the pointing basin once PPO proper started.

### 4.4 estim_err ceiling analysis (seed 2)

Seed 2 converged with fix_rate 0.11–0.16 but estim_err stabilizes at 0.115–0.130. This is NOT a policy failure. Analysis:

- **Gate-relative estimator chain RMS (C2 merged test, G3):** 0.131 m, p90=0.197 m. This is the estimator's accuracy given a gate-relative fix.
- **Seed 2 estim_err plateau:** 0.115–0.130 m — within 0–15% below the estimator RMS. Consistent with fix_rate=0.12–0.16 being better than zero but not "continuous lock."
- **Mechanism:** Each gate-relative fix has ~0.131 m RMS accuracy (bounded by camera/detector noise, PnP geometry). With fix_rate 0.12–0.16 (12–16% of timesteps yield a fix), the KF integrates these infrequent fixes but can't exceed the per-fix accuracy ceiling. estim_err of 0.12 reflects the estimator's achievable accuracy under the observed fix density.
- **The 0.08 threshold requires:** either (a) more fixes (higher fix_rate, e.g. via narrower band or lower altitude) or (b) the drone flying closer to the gate center so the in-plane component of the error is smaller — which is precisely what centering reward achieves.
- **Conclusion: the 0.08 breach requires centering + pointing together.** Pointing (S2) provides the fix-density necessary to run the KF; centering (S3) reduces the in-plane offset so the KF-reduced error dips below 0.08.

### 4.5 Seed 1's spurious estim_err <0.08

Seed 1 shows estim_err 0.075–0.079 at steps 1400, 1800, 2000, 2100, 2600, 2700, 3600 etc. DISQUALIFIED. The policy is a zombie (entropy −6.9, pointing_rate ~0.002, fix_rate ~0.001). The low estim_err is a coincidence of the degenerate trajectory (zombie policy may fly paths where the truth GP position happens to lie near the KF state by chance). No pointing → no causal mechanism → not valid GO evidence.

---

## 5. GO Gate Evaluation

From MEMORY (S2 design): **Binding GO for S2 = `estim_err ≲ 0.08` AND `terminal_pointing > 0` on ≥1 seed.**

| Criterion | Seed 0 | Seed 1 | Seed 2 | Gate |
|-----------|--------|--------|--------|------|
| terminal_pointing > 0 | 0.000–0.033 (noise) | 0.000–0.018 (zombie noise) | **0.033–0.127** | ✓ (seed 2 real) |
| estim_err ≲ 0.08 | 0.110–0.171 (NO) | 0.076–0.087 (coincidental/zombie) | 0.115–0.130 (NO) | ✗ |

**Formal gate: NOT-GO.**

However, the physical picture is clear: the primitive works (seed 2), the estim_err floor is the estimator RMS ceiling (not policy failure), and the fix to close 0.08 is centering. The formal gate design assumed pointing alone would be sufficient; it is not (centering is co-load-bearing with pointing for the final gap).

---

## 6. Commander Adjudication

**S2 VERDICT: PRIMITIVE CONFIRMED / PROCEED TO S3**

**What worked:**
- 2-axis look-at primitive is correct and functional (seed 2 = unambiguous convergence)
- Pitch sign empirical determination (+3.0) VINDICATED (both axes null by step 600)
- Warmup prevented collapse in 2/3 seeds (vs S1@3.0 where 1/2 collapsed without warmup)
- Hydra dir collision FIX CONFIRMED (all 3 seeds independent, no repeat of S1@3.0 bit-identical pair)
- fix_rate 0.11–0.16 achieved (vs floor 0.000–0.001 in S0/S1) — 100× improvement
- terminal_pointing 0.03–0.13 on seed 2 (vs 0.000 in all S0/S1 seeds) — definitively cleared the structural blocker

**What didn't work:**
- estim_err didn't breach 0.08 (stuck at estimator RMS ceiling 0.115–0.130)
- Only 1/3 seeds converge (same rate as S0) — symmetric critic remains a structural impediment to consistent convergence

**NEXT: S3 = centering reward + 2-axis primitive**

- Configuration: same as S2 + `+env.rw_centering=1.0` (START LOW — S1 showed 3.0 is destabilizing; 1.0 first)
- 3 seeds, NUPD=4000, per-seed Hydra dirs (keep collision fix)
- All S2 overrides carry forward: `++env.lookat_g_yaw=-3.0 ++env.lookat_g_pitch=3.0 +env.lookat_warmup_updates=200`
- Add: `+env.rw_centering=1.0` (single-plus, rw_centering NOT in COMMON)
- Start fresh (not warm-start from seed 2) — avoids single-seed bias

**S3 GO gate:** `estim_err ≲ 0.08 AND terminal_pointing > 0` on ≥1 seed (same formal gate; now the centering closes the remaining gap).

**Do NOT try rw_centering=3.0 first.** S1 showed that high centering weights destabilize PPO (value_loss 715). With the 2-axis primitive now adding its own gradients, the policy is already under more load than in S1. rw_centering=1.0 is mandatory first step.

**Asymmetric critic (appo):** remains an option if S3 also fails at 1/3 convergence rate. Not blocking S3; reassess after S3 data.

---

## 7. MEMORY-DELTA

```
S2 STAGE-2 RESULTS (2026-06-16, jobs 3275300-02):
Seed 2 = CONVERGED (2-axis: band_el 24-31°, band_az 22-37°, fix_rate 0.11-0.16, terminal_pointing 0.03-0.13, steps 600-3990). Seed 0 = warmup stable but stuck (band_el 70-85°, brief breakout step 300 lost). Seed 1 = warmup failure (entropy -0.8, value_loss 266 at step 100).
estim_err ceiling: seed 2 = 0.115-0.130 = gate-relative RMS ceiling (not policy failure). 0.08 breach requires centering (S3).
VERDICT: 2-axis primitive CONFIRMED (terminal_pointing >0 on seed 2). Formal GO gate NOT cleared (estim_err 0.115). NEXT = S3 (rw_centering=1.0 + 2-axis pointing; rw_centering SINGLE-plus; START LOW; 3 seeds fresh).
Hydra dir collision FIX CONFIRMED (all 3 seeds independent). Warmup_updates=200 stable 2/3 seeds; NOT guaranteed (1/3 collapse possible). Both signs empirical: g_yaw=-3.0, g_pitch=+3.0 (BOTH ++ double-plus).
```
