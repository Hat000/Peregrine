# mixer_probe2 — Completed Rows Summary

**Session:** SHADOWPC-BRIDGE-RETEST (2026-06-12). Probe run after 5 setup attempts.
**Recording (canonical):** `data/runs/20260612_133307_mixer_probe2`
(drifted 100m during catch5 — final recovery phase only; all 4 measurement phases complete)

## Probe invocation (working)

```
.venv/Scripts/python.exe scripts/rate_sysid.py \
  --mode profile \
  --profile handoff/laptop-s17-mixer-inc6-2026-06-11/mixer_probe2.json \
  --rate 100 --max-offset-m 100 --max-alt-m 35 --max-tilt-deg 150 \
  --label mixer_probe2
```

**Why non-default flags:**
- `--max-alt-m 35`: probe climbs 14m; c100_y31 at thr=1.0 causes ~18 m/s free-fall in
  catch3, reaching z≈30m before recovery
- `--max-tilt-deg 150`: `no_tilt_abort: true` in mixer_probe2.json suppresses tilt abort
  per-phase in code (325e191), but the abort still fires in practice (likely timing gap at
  phase boundary). Global override is the reliable workaround.
- `--max-offset-m 100`: c100_y31 free-fall causes horizontal drift of ~50m during catch3

## Per-phase settled results

| Phase | Thr | Axis | Motors tail (0,1,2,3) | Settled rate | Differential |
|-------|-----|------|-----------------------|--------------|--------------|
| z00_y31_long | 0.0 | yaw | [0.050, 0.419, 0.419, 0.050] | 9.76 rad/s | 0.369 |
| c100_r31 | 1.0 | roll | [0.999, 0.644, 0.999, 0.644] | 10.93 rad/s | 0.355 |
| c100_y31 | 1.0 | yaw | [0.651, 1.000, 1.000, 0.651] | 9.21 rad/s | 0.349 |
| c60_r31 | 0.6 | roll | [0.732, 0.467, 0.732, 0.467] | 10.99 rad/s | 0.266 |
| zhov_r31 | 0.266 | roll | [0.487, 0.053, 0.487, 0.053] | 10.79 rad/s | 0.434 |

**Differential** = |up-motors mean − down-motors mean| (roll axes: motors 0,2 vs 1,3).

## Previous z00_y31_long discrepancy

The original probe (run 1, `20260612_034154`) gave tail motors [0.09, 0.70, 0.70, 0.09]
(differential 0.61). This run gives 0.37. The new run had a partial climb (z≈12m vs 14m
target) and was in motion at step onset — the settled spin may differ due to slight body
pitch. Use the original cleaner measurement for kappa calibration. The c100_r31 settled
value [0.99, 0.65] from the original agrees with this run's [0.999, 0.644] — consistent.

## Key refit inputs (for S17 mixer update)

- **Top-rail roll corner (c100_r31):** motors 0,2 saturate at 1.0; motors 1,3 settle 0.644
  → top-rail clips roll authority by (1.0−0.644)/1.0 ≈ 35% at max collective
- **Top-rail yaw corner (c100_y31):** ccw motors 1,2 saturate at 1.0; cw motors 0,3 at 0.651
  → symmetric to roll, yaw authority clipped ~35% at max collective
- **Mid-band roll (c60_r31):** no saturation at either rail; full [0.732−0.467]=0.265 range
  → consistent with linear mixing in the unconstrained regime (thr=0.6)
- **Bottom-rail roll (zhov_r31):** motors 1,3 at 0.053 (idle floor); motors 0,2 at 0.487
  → hover thrust minus roll differential would push 1,3 below idle; clipped to 0.05
  → bottom-rail clips roll authority by (0.266−0.053)/0.266 ≈ 80% at hover thrust
  → strong mixer coupling at hover + large roll rate (the low-speed / standing-start regime)

## Rate rise slopes

- c100_r31: 158 rad/s² (fast — top rail available from tick 1)
- zhov_r31: 155 rad/s² (similarly fast — bottom rail limits only the dn-motors)
- c60_r31: 98 rad/s²
- z00_y31_long / c100_y31: ~47 rad/s² (yaw is slower than roll)

All realized settled rates 9.2–11.0 rad/s are consistent with super-rate formula at
full stick (S14: g(π) = G0/(1−0.30) ≈ 3.57 × G0 → at G0=3.14 → ~11.2 rad/s expected).
