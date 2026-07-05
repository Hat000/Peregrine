# vp_yaw innovation discriminator — real drift (A) vs time-misalignment artifact (B)

Generated 2026-07-05 11:30:39. See module docstring for design.

### 20260705_001321_rl_s1_f1

ACCEPTED innovation |deg| by yaw-rate bucket:

```
bucket                       n   median     mean      p95      max
quasi-static |g|<0.1       391    18.38    18.12    31.79    34.84
medium 0.1-0.5             190    17.57    17.53    33.80    34.97
turning |g|>0.5            322    17.91    17.33    32.16    34.86
```

LAG SWEEP over time-mapping shift tau (1075 quality-passing frames):

```
  tau_s  median_inn_deg  n_accepted
   -1.5           18.21         919
   -1.4           18.31         910
   -1.3           18.12         910
   -1.2           18.31         915
   -1.1           18.45         912
   -1.0           18.78         913
   -0.9           19.20         918
   -0.8           19.12         918
   -0.7           19.07         911
   -0.6           18.70         898
   -0.5           17.74         885
   -0.4           17.40         877
   -0.3           17.57         876
   -0.2           17.75         882
   -0.1           17.73         890
    0.0           18.07         903
    0.1           17.87         921
    0.2           18.01         929
    0.3           18.48         933
    0.4           17.66         931
    0.5           17.54         925
    0.6           17.14         915
    0.7           16.72         904  <-- min
    0.8           17.06         902
    0.9           17.39         919
    1.0           17.17         909
    1.1           17.54         918
    1.2           17.54         920
    1.3           17.47         914
    1.4           17.64         920
    1.5           17.72         916
```

tau=0 median = 18.07 deg; best tau = +0.7s with median 16.72 deg

+/-300 ms mapping-shift sensitivity (max |innovation change|, deg):

```
quasi-static   n= 391  median   0.77  p95   7.83
turning        n= 319  median  12.97  p95  17.17
```

### 20260705_002939_rl_s1_f1

ACCEPTED innovation |deg| by yaw-rate bucket:

```
bucket                       n   median     mean      p95      max
quasi-static |g|<0.1        58     1.46     3.47    15.98    24.00
medium 0.1-0.5              12    14.44    13.77    27.93    29.22
turning |g|>0.5             27    17.42    18.47    34.47    34.91
```

LAG SWEEP over time-mapping shift tau (124 quality-passing frames):

```
  tau_s  median_inn_deg  n_accepted
   -1.5            1.89          93
   -1.4            2.15          93
   -1.3            2.05          91
   -1.2            2.13          93
   -1.1            1.94          91
   -1.0            1.87          90
   -0.9            1.83          85  <-- min
   -0.8            1.98          90
   -0.7            1.96          89
   -0.6            2.10          92
   -0.5            2.24          93
   -0.4            2.34          94
   -0.3            2.55          96
   -0.2            2.65          96
   -0.1            2.90          96
    0.0            2.97          97
    0.1            2.94          96
    0.2            3.42          95
    0.3            3.32          95
    0.4            3.10          94
    0.5            3.31          94
    0.6            3.15          94
    0.7            2.97          95
    0.8            3.62          95
    0.9            3.45          90
    1.0            2.87          86
    1.1            3.05          85
    1.2            4.45          82
    1.3            4.47          79
    1.4            4.34          77
    1.5            5.34          79
```

tau=0 median = 2.97 deg; best tau = -0.9s with median 1.83 deg

+/-300 ms mapping-shift sensitivity (max |innovation change|, deg):

```
quasi-static   n=  58  median   0.34  p95   5.87
turning        n=  27  median  13.29  p95  16.27
```

---

## VERDICT — neither pure (A) nor pure (B); both mechanisms confirmed, in different regimes

**Quasi-static innovation distribution (the artifact-immune bucket, |raw_gyro_yaw| < 0.1 rad/s):**

| run | n | median | p95 |
|-----|---|--------|-----|
| choked 001321 (lit, crash-loop) | 391 | **18.38 deg** | 31.79 deg |
| clean 002939 (lit head segment) | 58 | **1.46 deg** | 15.98 deg |

**Choked run: the 18 deg is REAL (A), not artifact.** Four independent checks:
1. Innovation is FLAT across yaw-rate buckets (18.4 / 17.6 / 17.9 deg quasi-static/medium/turning) —
   no artifact scaling signature.
2. +/-300 ms mapping shift moves quasi-static innovations by only 0.77 deg median (turning: 13 deg,
   which is why only the quasi-static bucket is trusted).
3. Lag sweep over +/-1.5 s never collapses the median (17-19 deg everywhere; best tau +0.7 s gives
   16.7 deg) — no time offset explains it.
4. The VP headings are not junk: frame-to-frame mod-90 self-consistency 0.45 deg median.
   The lattice residual (nav_yaw - VP heading) wanders SLOWLY (+/-24 deg across flight quintiles,
   2.2 deg/frame) = a real, smoothly-wandering yaw-estimate error.

**Clean run: coordinator's counter-evidence (A-rebuttal) CONFIRMED in the healthy regime.** With
vp_yaw ON and pinning (deploy profile vq2_case_c => use_vp_yaw=True; fly_rl gate-seeker path),
quasi-static innovations are ~1.5 deg — exactly the ~1 deg between-pin drift predicted from on-pad
rates. Its larger turning-frame innovations DO scale with rate and swing ~13 deg under +/-300 ms
shifts = misalignment artifact (B). So the clean run's pooled "median 2.97 / mean 8.92" in RESULTS.md
overstates healthy-flight corrections; the honest healthy-flight number is ~1.5 deg.

**Why 18 deg despite vp_yaw being ON in the choked run:** that flight was a 2-minute crash-loop
(meta: collisions=200, stuck at gate 0, raw_gyro_yaw spikes to 11.5 rad/s = 657 deg/s). The live
filter WAS applying vision-yaw corrections (13.5% of nav ticks show yaw steps > 0.5 deg unexplained
by gyro integration, vs 1.1% in the clean run) but error re-accumulated faster than the (choked,
16.6 Hz, vision-starved) pin cadence could remove it. vp_yaw was genuinely fighting a real ~18 deg
error — and losing slowly, not idle.

**Answer to the coordinator's question — big fixes or small trims?** BOTH, by regime:
- Healthy flight: small trims (~1.5 deg) — small BECAUSE the pin works; that is what a functioning
  drift anchor looks like from the inside (innovation size alone cannot distinguish "useless" from
  "working"; the counterfactual is what matters).
- Degraded flight (collisions / choked loop): real ~18 deg fixes, actively applied, on
  artifact-immune frames.
- The counterfactual is visible in-data: wherever pinning stops (clean run's dark stretch: yaw parks
  at -157 deg with zero vision corrections; residual ramps -1 -> -40 deg through the lit tail),
  yaw error grows to tens of degrees. Cutting vp_yaw removes the only bound on an inertially
  unobservable state in exactly the regimes where it is doing real work.

**Cut recommendation: UNCHANGED — do not cut vp_yaw** (cadence/threading remain the right cost
levers). The 18 deg headline was not an artifact of my clock mapping, but it is a degraded-regime
number, not steady-state drift; the healthy-regime trims are ~1.5 deg.
