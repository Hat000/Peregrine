# Rung-1 re-VERIFY of the raw-vz alt-loop fix (commit dd16091). LIVE, ShadowPC 2026-06-07

## Verdict: **FAIL — the fix does NOT remove the altitude limit cycle.** Reproduced ×2 + GUI/audible ground-truth.

The structural fix (damp the alt loop on RAW given vz instead of the lagged KF vz) + the re-tune
(kp_alt 4→3, kd_alt 2→1.75) was flown exactly as specified. The ~6 Hz relay limit cycle **persists
essentially unchanged** — it is NOT the predicted clean hold (vz_rms ≪ 0.1). The offline twin's
"raw vz holds" prediction did not transfer to the live sim. **Rung 2/3 stay gated; handing back to the
offline Commander.** No gate flown.

## Config flown (meta.json `controller_config`, BOTH runs — this IS the fix)
`BODY_RATE decoupled tilt_comp | hover 0.2656 | kp_alt 3.0 kd_alt 1.75 | alt_clip [0.05,0.45] | RAW vz
(no --alt-kf-vz) | body_rate_sign [1,1,-1] odo_att_sign [-1,1,1] odo_rate_sign [-1,-1,1] | faithful | hover_hold`.
Verified by `--print-config` pre-flight and the recorded meta of each run. No ambiguity.

## OBSERVATION — settle window (last 5 s, 200 Hz extract). Raw telemetry + motor witness + GUI.

| signal (settle, last 5 s) | run 1 `184442` | run 2 `185528` | OLD 2026-06-06 (KF vz, kp4/kd2) | pre-registered (fix) |
|---|---|---|---|---|
| **TRUE vz std** | **0.286** | **0.298** | 0.23 | **≪ 0.1** ❌ |
| TRUE vz \|max\| | 0.533 | 0.654 | 0.61 | — |
| **cycle period** | **147 ms fft / 152 ms zc (6.8 Hz)** | **161 ms fft / 155 ms zc (6.2 Hz)** | ~160 ms (~6 Hz) | gone ❌ |
| **thrust (commanded)** | bang-bang **0.05↔0.45**, 63% hi / 17% lo | **0.05↔0.45**, 64% hi / 18% lo | 0.05↔0.40, 68% hi | no bang-bang ❌ |
| motors (ACTUATOR witness) | mean 0.340 (≈ cmd) | mean 0.342 (≈ cmd) | 0.325 | tracks ✅ |
| altitude hold | 1.345 m, std 0.012, slope +0.003 | 1.347 m, std 0.015, slope −0.007 | 1.399 m | ~1.5 m (held, ~0.15 m low) |
| roll / pitch (mean) | 0.0° / 0.0° | 0.0° / 0.0° | 0.0° / 0.0° | level ✅ |
| horiz drift max | 0.196 m | 0.191 m | 0.17 m | small ✅ |
| takeoff peak climb | ~2.33 m/s | ~2.32 m/s | — | ~1.4 m/s (a bit hot) |

Both runs ran to the 12 s cap, no abort fired (geofence 4 m / climb 6 m / tilt 60° all clear), force-disarmed.
The vz cycle is a **stable, sustained relay** — bounded, not growing (no divergence).

**GUI / audible ground-truth (teammate, relayed):** *"The hover is still oscillating around the hover
point, up and down. It's very audible — I can hear the thrust going up and down."* → directly corroborates
the telemetry: the audible thrust pulsing IS the collective bang-banging the [0.05,0.45] clips at ~6 Hz.

> Data note: run 2's recording spans ~45 s (it includes a ~30 s stale-race wait before GO) and contains
> one corrupt LOCAL_POSITION_NED sample (z spike to ~2527 m) in the pre-race span. Both are OUTSIDE the
> settle window and do not affect any settle metric above; flagged so a whole-run reprocess isn't tripped.

## INTERPRETATION (kept separate from OBSERVATION)

1. **The cycle persists ~unchanged → the raw-vz fix addressed the wrong cause.** Switching KF-vz → raw-vz
   moved vz_std 0.23 → 0.29 (if anything slightly worse) and left the ~6 Hz relay + both-rail bang-bang
   intact. Since the raw given vz is pristine/near-true (it matches the true LOCAL_POSITION_NED vz), the
   oscillation is **driven by loop/command transport DELAY, not by vz measurement lag.** The KF-vz lag the
   offline twin modeled as the destabilizer was not the (dominant) live cause.

2. **The cycle period pins the real loop delay well beyond the twin's sweep.** T_cycle ≈ 147–161 ms (mean
   ~0.15 s). For a relay feedback limit cycle dominated by transport delay (T ≈ 4·L), this implies an
   **effective loop delay on the order of ~35–40 ms** — at/beyond the 30 ms ceiling `twin_hover.py` swept,
   where it only ever predicted a *mild* bob (vz_amp ~0.2). Live is full-amplitude (~0.5–0.65). **The twin
   under-modeled the live latency**, so its raw-vz prediction was optimistic. (Estimate — the loop is a
   delayed double-integrator with derivative feedback, not a pure delay+integrator; measure directly, below.)

3. **Why it bang-bangs (saturation mechanics).** At hover, the `kd_alt·vz` term alone is ±0.875 (kd 1.75 ×
   vz ~0.5) against only ~±0.18 of clip headroom about hover (0.2656 in [0.05,0.45]) — so any vz swing
   saturates BOTH clips → relay. The steady ~0.15 m droop × kp_alt 3 ≈ 0.46 already pins the ceiling on its
   own (why it holds ~0.15 m low). The loop is far too stiff for the true delay.

4. **"Drop kd_alt to 1.5" will NOT save this, and raising kd is the wrong direction.** 1.5 → ±0.75, still
   >> headroom → still saturates. More importantly, derivative action is *destabilizing* under this delay:
   at 6.8 Hz the kd "lead" has rotated into lag, so more kd feeds the cycle. The fix needs **lower gains
   (both kp_alt AND kd_alt)** for gain/phase margin at the true delay — not a kd bump.

## The PIN — measure the true loop delay from these extracts (the task's "pin the residual transport")
Cross-correlate the **commanded thrust** series (`commanded.series[0]=t, [3]=thrust`; also full-rate in
`commands.jsonl`) against the **true vz** (`true_lpn.series=[t, alt, vz]`) at the ~6 Hz cycle: the phase
lag from collective-command to vz-response = the loop transport delay. Both series are in each
`rung1_hover_run{1,2}_extract.json`. That number is what the twin's vertical channel must carry.

## Recommendation (offline Commander)
1. Add the **measured** transport delay (~35–40 ms, confirm via the pin above) to the twin's vertical
   channel — this is the dynamic the twin was missing.
2. Re-tune for gain/phase margin at that delay → **lower kp_alt AND kd_alt** (much lower); the current
   3/1.75 is too stiff. Consider: a slower alt-loop update / thrust **rate-limit** (kills the relay
   directly), and/or a small **alt integrator** to remove the ~0.15 m steady droop without leaning on kd.
3. *(Optional live A/B, cheap, decisive)* re-fly rung 1 with `--alt-kf-vz` — predict ~no change vs raw vz;
   that nails "delay-driven, not vz-lag," confirming the structural fix targeted the wrong cause.
4. Re-derive the hover/cycle with `scripts/fit_vertical.py` (extracts copied into this dir) +
   `scripts/twin_hover.py` (now with the real delay) before the next live re-VERIFY.

## What DID transfer (unchanged from 2026-06-06 — high confidence, ×2 + GUI)
Level attitude (roll/pitch 0.0°, leveled the −17.8° rest tilt), **no balloon / no runaway** (alt net drift
≤0.007 m/s), **we own thrust in CTBR** (motors track commanded to ±0.002, never pinned), lateral held
(≤0.2 m drift, no positive-feedback sign error). The ONLY failure remains the vertical loop.

## Artifacts (this dir)
- `rung1_hover_run1_extract.json` + `rung1_hover_run1_meta.json` — run 1 (`20260607_184442_rung1_hover`).
- `rung1_hover_run2_extract.json` + `rung1_hover_run2_meta.json` — run 2 (`20260607_185528_rung1_hover`).
- `scratch/analyze.py` — the settle-window limit-cycle analyzer used for the table above.

**Process:** reproduced ×2 (GUI-corroborated), config verified pre-flight (`--print-config`) and per-run
(meta), all bounds honored, force-disarm on exit. `memory/` (repo) not edited.
