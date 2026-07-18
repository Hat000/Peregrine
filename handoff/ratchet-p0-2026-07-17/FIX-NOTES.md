# Ratchet-P0.1: tape-player timing fix (sim-time-indexed playback + precision scheduling)

2026-07-17 · branch `ratchet-tape-2026-07-17`. Implements REPORT.md's recommended fix #1
(sim-time-indexed playback) + #2 (precision tick scheduling), scoped to `--sysid-replay` ONLY.
Default OFF preserves the current player byte-for-byte for the A/B re-probe.

## What changed (all in `rl/fly_rl.py` unless noted)

| # | change | file:lines |
|---|--------|-----------|
| A1 | **Pure row selector** `replay_row_for_sim_time(st_ns, anchor_ns, tick_ns, n_rows) -> int|None` — ZOH latest-due-row by sim-time; clamp-below-0; `None` once the last row's full window elapsed. Integer-exact (ns ints, no float clock); unit-testable without a sim. | rl/fly_rl.py:762-791 |
| A2 | **Replay state** in `_fly_ego`: `_sysid_simtime` (flag), `_sysid_tick_ns = round(1e9/rate)`, `_sysid_anchor_ns` (lazy), `_sysid_wake` (wall-wake counter), `_sysid_replay_on`; banner line prints the indexing mode. | rl/fly_rl.py:2086-2105 |
| A3 | **Program-injection dispatch**: when `_sysid_simtime`, anchor at the FIRST program tick (`_sysid_anchor_ns = st`) and select `_row = replay_row_for_sim_time(...)`; `None` ends the program. Else the unchanged one-row-per-wall-tick path (`_row = _sysid_k`). `_krow`/`_seg`/`_av` all key off `_row`. | rl/fly_rl.py:2223-2244 |
| B1 | **Precision scheduling — hybrid tick wait** (replay only): pump + coarse-sleep to ~2 ms before the deadline, then spin. Policy path keeps the UNCHANGED `while ... time.sleep(0.001)` wait (gated by `if _sysid_replay_on`). | rl/fly_rl.py:2141-2159 |
| B2 | **Precision scheduling — Windows 1 ms timer**: `timeBeginPeriod(1)` / `timeEndPeriod(1)` (ctypes `windll.winmm`, try/except) wrapping the `_fly_ego` call in `_fly_armed`, in a `try/finally` so it ALWAYS restores. Gated on `sysid_replay and sys.platform=="win32"`; no-op otherwise. Wrapping the call site (not the 280-line loop) keeps the diff small and the finally honest — the timer is process-global so the scope is identical. | rl/fly_rl.py:2569-2591 |
| C | **Logging**: `k` column unchanged meaning (= tape row actually sent = selected row in simtime mode). New `k_tick` (wall-wake counter) APPENDED at the END of the sysid CSV schema so P0 comparison scripts keying on existing columns are unaffected. | rl/fly_rl.py:2259, 2461 |
| — | CLI flag `--sysid-replay-simtime` (store_true, default OFF). | rl/fly_rl.py:2910-2919 |
| — | `meta.json` records `sysid_replay` + `sysid_replay_simtime` (ego meta block). | rl/fly_rl.py:3452-3453 |
| T | **Unit tests** (6): exact 25 ms grid; stretched 25.88 ms wake schedule (wall-tick lag ≈3 rows by row ~111 while simtime stays within ±1); clamp-at-0; end-of-tape only after the last full window; tick_ns rounding at rate 40/30; degenerate guards. | tests/test_replay_simtime.py |

## Flag names + defaults

- `--sysid-replay-simtime` — **default OFF**. OFF = byte-identical one-row-per-wall-tick (the A/B
  baseline / current behavior). ON = sim-time-indexed rows.
- `tick_ns = round(1e9 / --rate)` (25_000_000 at `--rate 40`). **`--rate` MUST equal the tape's
  grid rate** (the tape is a fixed-rate ZOH resample from `tape_extract.py`).
- Windows precision timer: automatic for ANY `--sysid-replay` run on Windows (both indexing
  modes); no flag, no-op off-Windows / off-replay.

## Re-probe recipe delta (the one-line change to REPORT.md §Step-2)

Take REPORT.md Step-2's exact command and **add `--sysid-replay-simtime`**:

```
PYTHONPATH=src python rl/fly_rl.py --endpoint udp:127.0.0.1:14550 --ego-ckpt ckpts/vpeffs0_actor.pth \
  --sysid-replay tape_full.csv --rate 40 --ego-rate-scale 1.2 --no-virtual-flip \
  --sysid-climb-s 0 --sysid-settle-s 0 --sysid-replay-simtime \
  --seeker-detector red_glow --max-seconds 26 --video-async-detect --wait-seconds 1800 \
  --flights 1 --label ratchet_p0_simtime_rN
```

Run it TWICE (rN = r1, r2) exactly as the original probe. The baseline (no `--sysid-replay-simtime`)
still reproduces the 0/5-gate crash for the A/B contrast. The banner will print
`[sysid] row indexing: SIM-TIME (...)`; `meta.json` records `sysid_replay_simtime: true`.

## Expected outcome

- **Gate count should JUMP** (baseline banked 0/5, struck gate-0's frame at row 111 where the
  command stream lagged ~72 ms / ~0.5 m). With rows selected by sim-time, the command edges land
  at their recorded sim-time -> the champion line threads gate 0 (pass was at tape row ~118) and
  banks onward. If it still strikes gate 0, the residual is NOT timing (escalate: sim spawn/physics
  reseed, or tape accuracy).
- **×2 identity re-measured**: two simtime runs should again be run-to-run deterministic (same
  gate sequence, same crash tick if any, commands bit-identical). Per §D, a row-boundary straddle
  can shift ONE row transition by one sim tick between runs — acceptable; the ×2 re-probe measures
  it. The selector is integer-exact, so any straddle is a hard sim-tick boundary, not float noise.
- `sysid_vq2_log.csv` now carries `k_tick` (wall-wake index): in simtime mode watch for repeated
  `k` across incrementing `k_tick` (fast wakes re-sending the due row = the ≥30 Hz keepalive) and
  `k` skips across a slow wake (latest-due wins). Loop rate should read ≈40 Hz (was 38.6).

## Deviations / notes

- **Row 0 off-by-one (pre-existing, now corrected in simtime mode only).** The existing bootstrap
  sets `_sysid_k = 0` on the boot-completion tick and the tail `_sysid_k += 1` bumps it to 1, so
  the WALL-TICK path's first *program* row is `_sysid_prog[1]` — row 0 is skipped (unchanged; I
  did NOT touch it, to keep OFF byte-identical). SIM-TIME mode anchors at the first program tick
  and selects `floor(0)=0`, so it correctly INCLUDES row 0. This one-row inclusion is dwarfed by
  the ~3-row lag being fixed and does not confound the coarse gate-count A/B outcome; flagged for
  transparency.
- **Timer wraps the `_fly_ego` call, not the loop.** `timeBeginPeriod`/`timeEndPeriod` are
  process-global, so bracketing the call in `_fly_armed` (small diff, clean `try/finally`) is
  functionally identical to bracketing the loop and avoids re-indenting ~280 lines. The hybrid
  sleep that *uses* the finer timer lives in the loop, replay-gated.
- **Escape hatch NOT taken.** REPORT's cadence data (per-tick sim_time diff ≤16.9 ms, monotone,
  re-converging) confirms `sim_time_ns` is reliably per-wake usable; design A stands. Backward
  jumps are impossible here — the epoch/reset guard breaks BEFORE the sysid dispatch, and a
  momentary non-advance simply re-selects (and re-sends) the same row.
- **Policy (non-replay) timing untouched.** The hybrid wait is behind `if _sysid_replay_on`; the
  timer branch is behind the replay guard in the ego-only dispatch. Policy flights never enter it.

## Tests

`tests/test_replay_simtime.py` (6, all green) + `tests/test_tape_extract.py` (11, unchanged, green)
= 17 passed. Regression: `test_ego_*` + `test_sysid_action_sign_maps` + `test_vq2_control_recipe`
= 100 passed / 5 skipped (pre-existing conditional skips). Import-smoke: `import fly_rl` + `--help`
build the parser with the new flag registered.
