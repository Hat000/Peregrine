# Gate-mapper validation — measured noise model, real VQ1 geometry

Noise (perception-char 2026-06-08): sigma [0.73, 0.47, 0.29] m (N,E,D), bias [-0.42, 0.06, -0.28] m, leak 1.1% @ 3-8 m, assoc errors 10%, detect 85%, range 2.0-28.0 m. Validity yardstick: 0.75 m half-opening (in-plane).

## Case A — pose-aided (5 seeds; cells: mean (worst) over seeds)

| config | n/gate | gates | err mean m | err max m | in-plane max m | yaw max deg | rej |
|---|---|---|---|---|---|---|---|
| baseline (labeled, 1 lap) | 61 | 6/6 | 0.52 (0.58) | 0.63 (0.70) | 0.36 (0.39) | 2.47 (2.74) | 46 |
| + measured-bias correction | 61 | 6/6 | 0.13 (0.15) | 0.20 (0.25) | 0.13 (0.17) | 2.47 (2.74) | 46 |
| unlabeled (clustering) | 61 | 6/6 | 0.54 (0.58) | 0.62 (0.68) | 0.33 (0.34) | 2.78 (2.92) | 4 |
| noise x0.5 | 61 | 6/6 | 0.26 (0.29) | 0.31 (0.35) | 0.18 (0.20) | 2.47 (2.74) | 46 |
| noise x2 | 61 | 6/6 | 1.03 (1.17) | 1.27 (1.41) | 0.72 (0.79) | 2.47 (2.74) | 48 |
| leak 0% | 61 | 6/6 | 0.52 (0.56) | 0.64 (0.73) | 0.35 (0.36) | 2.54 (3.00) | 40 |
| leak 5% | 61 | 6/6 | 0.54 (0.58) | 0.62 (0.68) | 0.33 (0.35) | 2.48 (2.87) | 69 |
| assoc errors 5% | 61 | 6/6 | 0.53 (0.56) | 0.63 (0.71) | 0.34 (0.35) | 2.68 (2.83) | 28 |
| assoc errors 15% | 61 | 6/6 | 0.52 (0.59) | 0.64 (0.69) | 0.36 (0.38) | 2.44 (2.74) | 59 |
| 0.25 lap | 48 | 2/6 | 0.54 (0.59) | 0.61 (0.64) | 0.33 (0.37) | 1.98 (2.64) | 11 |
| 0.5 lap | 56 | 3/6 | 0.54 (0.62) | 0.62 (0.69) | 0.33 (0.35) | 2.40 (2.67) | 17 |
| 2 laps | 123 | 6/6 | 0.51 (0.55) | 0.57 (0.62) | 0.32 (0.34) | 2.50 (3.06) | 90 |

## Case B — rough prior + refinement (5 seeds)

| config | prior err max m | fused err max m | gain |
|---|---|---|---|
| offset 2 m, 1 lap, prior_sigma 2 | 2.00 (2.00) | 0.62 (0.69) | 3.2x |
| offset 2 m, 0.25 lap | 2.00 (2.00) | 2.00 (2.00) | 1.0x |
| offset 3 m, 1 lap | 3.00 (3.00) | 0.62 (0.70) | 4.8x |
| offset 1 m, 1 lap, prior_sigma 1 | 1.00 (1.00) | 0.59 (0.68) | 1.7x |

## Case C — no pose (3 seeds; 'aligned' = translation gauge removed)

| config | gates (+extra) | comps | disc | aligned mean m | aligned max m | raw max m | in-plane max m | yaw max deg | t |
|---|---|---|---|---|---|---|---|---|---|
| baseline (scan, range 28, sig_a 1.0) | 6/6 (+0) | 3 | 4 | 1.64 (2.57) | 2.14 (3.30) | 4.25 (5.82) | 0.85 (1.37) | 2.58 (3.00) | 17s |
| no scan nod | 6/6 (+0) | 4 | 5 | 2.80 (3.13) | 3.40 (3.80) | 6.67 (7.78) | 3.08 (3.71) | 2.56 (2.96) | 20s |
| range 24 (measured flat-to) | 6/6 (+0) | 6 | 5 | 2.53 (3.53) | 2.84 (3.86) | 5.76 (7.37) | 2.00 (3.48) | 2.52 (3.03) | 16s |
| range 32 (live cap) | 6/6 (+0) | 2 | 3 | 1.46 (2.22) | 1.82 (2.55) | 3.85 (5.08) | 1.56 (2.51) | 2.51 (2.97) | 16s |
| sig_a 0.5 (smoother lap) | 6/6 (+0) | 4 | 4 | 1.58 (2.48) | 2.23 (3.27) | 4.41 (5.96) | 0.98 (1.59) | 2.59 (3.03) | 16s |
| sig_a 6.0 (sporty lap) | 6/6 (+0) | 3 | 4 | 2.09 (3.20) | 2.40 (3.61) | 4.72 (6.37) | 1.13 (1.88) | 2.58 (3.00) | 7s |
| NO smoothness bridge | 6/6 (+0) | 3 | 4 | 1.88 (3.12) | 2.23 (3.43) | 4.87 (6.75) | 1.27 (1.58) | 2.58 (2.99) | 2s |
| leak 5% | 6/6 (+0) | 3 | 4 | 2.21 (3.70) | 2.50 (4.15) | 5.33 (8.18) | 2.15 (3.65) | 2.69 (3.18) | 22s |
| labels given (naming) | 6/6 (+0) | 3 | 4 | 2.91 (4.58) | 3.77 (6.11) | 6.54 (11.16) | 1.75 (2.20) | 2.93 (3.31) | 14s |
| 2 passes | 6/6 (+6) | 7 | 10 | 1.57 (2.51) | 2.06 (3.27) | 4.12 (5.74) | 0.82 (1.23) | 2.58 (3.00) | 33s |

Sample sightings dumped: handoff\laptop-gate-mapper-2026-06-11\samples\sample_pose_aided.json, handoff\laptop-gate-mapper-2026-06-11\samples\sample_relative.json
