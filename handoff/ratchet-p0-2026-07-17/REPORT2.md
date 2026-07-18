# Ratchet-P0.1 re-probe — timing fix VERIFIED, but gates still 0/5 and ×2 DIVERGED: residual is LAUNCH-STATE variance, not the player

2026-07-18 (ShadowPC, branch tip `551330e`). Re-ran the P0 probe ×2 with `--sysid-replay-simtime`
per the relay. Unit tests 17/17 green locally before flying. Banner `[sysid] row indexing:
SIM-TIME (... floor((sim_time-anchor)/25.000 ms) ...)` confirmed in BOTH runs.
**ESCAPE HATCH taken:** gates are still 0/5 and the pair diverged → stopped after the two runs;
no player iteration. Both runs' logs committed.

## Adjudication vs the three asks

**(a) Gate count: 0/5 both runs — NO jump.** But the *mode* changed: the P0 baseline struck
gate 0's frame at tape row 111 both times; with the timing fix, r1 struck the frame 3 rows later
(row 114) and **r2 missed the frame entirely**, blind-flew the champion's command profile through
tape segments g0→g2 (~5.6 s, past the champion's gate-1 pass time), and crashed into the
**environment** (id 1002), still gates=0. The fixed timing moved the flown line *closer* to the
aperture edge — from "solid frame strike" to "razor-edge clip-or-miss". Fengyou's visual, both
probes: we run into the **right side of gate 0**.

**(b) ×2 identity: DIVERGED — but not in the command path.**

| | P0.1 run 1 (`20260718_014726`) | P0.1 run 2 (`20260718_015427`) |
|---|---|---|
| gate passes | none | none |
| end | CRASH row 114, seg g0, **gate strike (1001)** | CRASH row 225, seg g2, **env strike (1002)** |
| loop rate | 39.9 Hz | 39.9 Hz |
| race-created → first cmd | 0.560 s | 0.473 s |

- Boundary straddles (predicted ≤1 row): **42/114 common wakes selected a ±1 tape row** →
  80/456 command cells differ; every straddle is a single-row edge shift ≤25 ms. One 2-row skip
  in r2 (wake 3, rows 1→3) from a sim-clock hop — latest-due-wins as designed. Cadence
  |r1−r2| ≤ 12.5 ms.
- **The physics diverged BEFORE the commands could**: at wake 0 — same tape row 0 command in
  both runs — accel_z already differed by **1.74 m/s²**; gyro_x |r1−r2| median over the first
  quartile is **0.080 rad/s vs 0.006 in the P0 pair (13×)**. The offset then *converges*
  mid-flight (medians 0.003–0.006 rad/s, wakes 28–83) and the outcomes bifurcate at the gate-0
  edge (q4 max 1.06 rad/s = r1's strike vs r2 flying free).
- Conclusion: run-to-run **launch/contact state on the 17° start block varies**; the open-loop
  tape then flies a laterally offset line; when that line skims a gate edge, millimeter-scale
  state differences flip strike↔miss. The P0 pair's "IDENTICAL" verdict was conditional on a
  luckily matched launch state (its q1 residual was 13× smaller). Sim *physics* remain
  deterministic given matched state — commands, cadence, and mid-flight convergence all show it.

**(c) Measured loop rate: 39.9 Hz both runs (was 38.6).** The fix does exactly what it says:
sim-time row selection + precision ticks kill the cumulative ~72 ms command lag (now ≤12.5 ms,
non-accumulating). Row 0 is now included (was skipped in wall-tick mode, per FIX-NOTES).

## Why every replay misses RIGHT at gate 0 (champion didn't) — top suspect measured

All four replays (P0 ×2, P0.1 ×2) launched **0.40–0.56 s** after race creation. The champion's
first command came **2.83 s** after race creation (first tick sim 6.127 s vs race-created
3.298 s). If the sim drops/settles the drone onto the 17° block at race creation, the champion
launched from a ~2.8 s-settled contact state, every replay from a ~0.5 s one — a systematic
initial-condition offset fully consistent with a repeatable right-of-aperture line, and the same
mechanism as the run-to-run variance above (just larger and one-sided). Tape provenance was
re-verified on Fengyou's challenge: source log SHA-matches the committed champion blob
(`144a9a1b`, 5 gates in-log), wire scales identical (yaw_scale 1.0, ego_rate_scale 1.2,
max_rate 0.0), P0 command echo bit-identical to the tape.

## Recommendation (for the laptop side; no player changes made here)

1. **P0.2 settle-matched probe**: N≥3 runs with GO issued ~2.8 s after race creation (match the
   champion's window; ops-only, zero code). Separates the one-sided offset (settle) from inherent
   reset variance. If gates jump → ratchet proceeds with a "GO-delay" protocol pin.
2. If variance persists at N≥3, open-loop tape replay is initial-condition-limited at
   gate-threading precision → the **arrestor + `--truncate-at-gate` become load-bearing**
   (closed-loop correction between segments), not optional polish.
3. `k_tick` column + straddle stats worked as designed; keep them for future probes.

## Files

- `data/runs/20260718_014726_ratchet_p01_r1_f1/` + `data/runs/20260718_015427_ratchet_p01_r2_f1/`
  — `sysid_vq2_log.csv` (now with `k_tick`), `meta.json` (`sysid_replay_simtime: true`),
  `mavlink.tlog`. Committed alongside this report.
- Baseline tape + P0 sessions: see REPORT.md (2bacc1a).

## MEMORY-DELTA (≤10 lines)

1. P0.1 2026-07-18: `--sysid-replay-simtime` VERIFIED (39.9 Hz, cadence ≤12.5 ms, straddles ≤1
   row @ 42/114 wakes) — the P0 timing defect is dead.
2. But gates still 0/5 AND the ×2 pair DIVERGED (gate-strike row 114 vs env-crash row 225):
   physics split at wake 0 on identical commands (accel_z Δ1.74, q1 gyro residual 13× P0's) →
   **launch/contact state on the 17° block varies run-to-run**; P0's IDENTICAL was a matched-state
   pair. Sim physics stay deterministic GIVEN the state.
3. All 4 replays launched 0.4–0.56 s after race creation; champion launched 2.83 s after →
   settle-window mismatch is the top suspect for the systematic miss-right at gate 0.
   Next: settle-matched P0.2 (GO ~2.8 s post-reset, N≥3, ops-only); if variance persists,
   arrestor/truncate-at-gate become load-bearing. Escape hatch honored — no player iteration.
