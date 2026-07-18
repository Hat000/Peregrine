# Ratchet-P0: replay-fidelity probe — VERDICT: IDENTICAL (sim + player deterministic); tape accuracy 0/5 gates

2026-07-18 (ShadowPC session, branch `ratchet-tape-2026-07-17` @ 3389f68). Champion tape
(`data/runs/20260714_202317_panel_run_f1`, 5 gates, vpeffs0, ego_rate_scale=1.2) extracted with
`rl/tape_extract.py` and replayed TWICE through `--sysid-replay` against fresh sim resets.

## Verdict line

**IDENTICAL** — run 1 and run 2 produced the same (empty) gate-pass sequence, crashed on the
**same program tick k=111**, in the **same tape segment (g0)**, with the **same collision type
(id 1001 = GATE strike)**. No divergence tick exists to report. Note the PASS is *degenerate*:
zero gates were passed, so the per-gate sim-time criterion is vacuous — the determinism evidence
is instead the tick-level agreement below, which is stronger.

## Step 1 — tape extraction (validated clean)

`python rl/tape_extract.py data/runs/20260714_202317_panel_run_f1/ego_obs.jsonl --out tape_full.csv --validate`

- 287 source ticks, 10.822 s, 26.43 Hz effective (dt 13.9/27.9/69.6/104.2 ms min/med/p95/max)
  → ZOH-resampled to **433 rows @ 40 Hz**. No NaN/gap/monotonicity errors; one benign warning
  (104 ms source gap into k=52, ZOH holds — matches the recording).
- 5 gate passes at source k=76/132/159/193/243 (t=2.945/5.168/6.189/7.425/9.189 s) — exactly the
  mission expectation.
- Inversion round-trip: rates 1.1e-16 (float eps), collective 5.7e-6 (log rounding bound). The
  tape reproduces the champion's logged wire values exactly under the pinned replay flags.

## Step 2 — the two replays

Identical command both runs (the extractor's pinned invocation + async-detect + labels):

`PYTHONPATH=src python rl/fly_rl.py --endpoint udp:127.0.0.1:14550 --ego-ckpt ckpts/vpeffs0_actor.pth --sysid-replay tape_full.csv --rate 40 --ego-rate-scale 1.2 --no-virtual-flip --sysid-climb-s 0 --sysid-settle-s 0 --seeker-detector red_glow --max-seconds 26 --video-async-detect --wait-seconds 1800 --flights 1 --label ratchet_p0_rN`

Both runs: fresh reset → passive attach (`started=False pos=y`) → fresh GO (`pos_off=0.00 m`).
`cmd_rate_scale=1.2` confirmed forced in both banners.

## Gate-sequence tables, side by side

| transition | champion (source tape) | run 1 (`20260718_005945`) | run 2 (`20260718_010327`) |
|---|---|---|---|
| 0→1 | k=76  t=2.945 s | — | — |
| 1→2 | k=132 t=5.168 s | — | — |
| 2→3 | k=159 t=6.189 s | — | — |
| 3→4 | k=193 t=7.425 s | — | — |
| 4→5 | k=243 t=9.189 s | — | — |
| **end** | CRASH after gate 5 (t=10.82 s) | **CRASH k=111, sim-span 2.847 s, gate strike (1001), gates=0** | **CRASH k=111, sim-span 2.848 s, gate strike (1001), gates=0** |

## Tick-level determinism evidence (sysid_vq2_log.csv, 111 program ticks each)

- **Commands: bit-identical.** 0 / 999 cells differ across all wire-echo columns
  (cmd_wx/wy/wz/thrust, collective) — the player is perfectly repeatable.
- **Cadence: sub-tick.** Start-anchored per-tick sim_time difference ≤ 16.9 ms (one tick of
  jitter mid-flight), re-converging to −0.59 ms at the crash tick.
- **Physics: same trajectory.** Mid-flight IMU residuals are tiny and do NOT grow —
  gyro_x |r1−r2| median by quartile: 0.0058 / 0.0010 / 0.0007 / 0.0024 rad/s (max 0.28/0.04/0.03/0.16).
  No chaos-style exponential divergence over 2.85 s of open-loop flight. Larger residuals are
  confined to k=1–2 (launch jerk sampled at a few-ms IMU phase offset) and k=111 (gate-contact
  dynamics). Peak diffs: gyro_x 0.92 rad/s @ k=111 (crash), accel_x 1.18 m/s² @ k=2 (launch).
- Loop rate: 38.6 Hz (r1) vs 38.7 Hz (r2) — same systematic stretch both runs (see below).
- One oddity: meta collisions r1=1 vs r2=2 (both ids [1001] = gate only) — double-contact during
  the tumble or RACE_STATUS residue ([[race-outcome-recording-gotchas]]); outcome identical.

## Secondary — why the tape passes 0/5 gates (accuracy, not determinism)

Both replays flew the champion's line for ~2.85 s and **struck gate 0's frame** where the
champion threaded the aperture (champion pass at t=2.945 s = tape row ~118; crash at row 111).
Quantified cause: the player consumes **one row per loop tick**, and the loop delivered
**25.88 ms effective ticks (38.6 Hz) vs the tape's 25 ms grid** (Windows sleep granularity;
work per tick was ~0 ms). By gate 0 the command stream ran **~72 ms late relative to sim
physics ≈ 0.4–0.6 m of trajectory offset at 6.4 m/s** — more than enough to turn an aperture
crossing into a frame strike. ZOH onset quantization adds ±25 ms per command edge. This is a
**tape-player timing problem, fixable**, not sim non-determinism:

1. **Sim-time-indexed playback** — consume tape rows by sim_time_ns (skip/hold to track the sim
   clock) instead of one-row-per-tick; kills the cumulative 3.5 % stretch.
2. Or tighter tick scheduling (busy-wait the last ms; multimedia timer resolution).
3. Re-probe: with timing fixed, expect the gate count to jump; `--truncate-at-gate K` tapes +
   the arrestor (design doc on this branch) then bound per-gate risk for the ratchet proper.

## Files

- `tape_full.csv` / `tape_full.raw.csv` / `tape_full.tapemeta.json` — the validated tape (repo root).
- `data/runs/20260718_005945_ratchet_p0_r1_f1/` + `data/runs/20260718_010327_ratchet_p0_r2_f1/` —
  per-tick `sysid_vq2_log.csv` (cmd echo + IMU), `meta.json`, `mavlink.tlog` committed.
  **Replay mode writes NO ego_obs.jsonl** (policy path bypassed) — `sysid_vq2_log.csv` is the
  tick-level record and is richer for this purpose (wire echo + IMU response).
- Comparison script: inline in this session; numbers above are its output verbatim.

## MEMORY-DELTA (≤10 lines)

1. Ratchet-P0 2026-07-18 PASS: eval sim + tape player are **run-to-run deterministic** — two
   replays of the champion tape crashed on the same tick k=111, same gate strike, commands
   bit-identical, cadence ≤17 ms, mid-flight gyro residual ~1e-3 rad/s with NO growth.
2. Secondary: tape ACCURACY fails the champion line (0/5 gates, clips gate 0 frame) — player
   runs 25.88 ms ticks vs the tape's 25 ms grid → ~72 ms command lag by gate 0 (~0.5 m); fix =
   sim-time-indexed row consumption (or sub-ms tick scheduling), then re-probe.
3. Ops: `--sysid-replay` sessions write NO ego_obs.jsonl — sysid_vq2_log.csv is the tick record
   (wire echo + IMU). Collision COUNT can differ run-to-run (1 vs 2, same ids/outcome) — don't
   key determinism checks on it.
