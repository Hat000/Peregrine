# Ratchet-P0.2 settle-matched probe — settle protocol WORKS (wake-0 variance collapsed ~1000×), but gates 0/5 and all misses RIGHT: branch (c) taken, logs committed, STOPPED

2026-07-18 (ShadowPC, branch tip `d65fe46`). N=3 replays, identical P0.1 command
(`--sysid-replay-simtime`, `--rate 40`, `--ego-rate-scale 1.2`), labels `ratchet_p02_r1..r3`,
GO issued ~3 s after race reset per the relay. One late-join interruption mid-probe (see Ops).

## ⚠ Metric correction first (changes how ask #2 must be read)

`race_start_boot_time_ms` is stamped at **GO**, not at race creation — cross-checked against the
end-of-flight sim clock in the same records. So the "race-created → first-cmd" numbers here and
in REPORT2 measure the fixed **GO→arm→program latency (~0.5 s)**, NOT the operator's settle wait
(which happens *before* GO and is invisible to this metric). Re-reading the champion with aligned
clocks: its 2.83 s gap is a real **post-GO first-command delay** (policy warmup: arm attempts +
first-frame/detector latency) during which it sat armed and silent on the block. That condition
is NOT reproducible ops-only and was not attempted (per DO-NOTs). REPORT2's "settle window"
framing should be read as "pre-launch block time", which the P0.2 protocol *did* vary via the
operator's count — and which demonstrably worked (below).

## Results

| run | GO→first_cmd | prog rows | died in seg | collision | pilot visual (Fengyou) |
|---|---|---|---|---|---|
| p02_r1 (`021613`) | 0.493 s | 271 (~6.8 s) | g3 | env (1002) | right of gate 0, right of g1, right of g2 |
| p02_r2 (`021824`) | 0.477 s | 199 (~5.0 s) | g1 | **gate 1 frame** (1001) | right of gate 0, clipped right of gate 1 |
| p02_r3 (`022415`) | 0.556 s | 297 (~7.4 s) | g3 | env (1002), 13 contacts | (same right-line shape) |

Loop 39.9 Hz all runs. All gates=0; gate-pass sequences all empty (nothing for per-gate times).

## Adjudication

**(a) Wake-0 state spread: PASS — collapsed to the matched-pair level and beyond.**

| pair | accel_z Δ @ wake 0 | gyro_x q1 median | reference |
|---|---|---|---|
| p02 r1↔r2 | **0.000 m/s²** | 0.0117 rad/s | P0.1 split pair: 1.742 / 0.0799 |
| p02 r1↔r3 | **0.000 m/s²** | 0.0182 rad/s | P0 matched pair: 1.138 / 0.0058 |
| p02 r2↔r3 | **0.001 m/s²** | **0.0008 rad/s** | (tighter than P0's "identical" pair) |

The ~3 s pre-GO block time genuinely reproduces the launch state: wake-0 accel deltas went from
~1.1–1.7 m/s² (P0/P0.1, ~0.5 s casual GOs) to ≤0.001 m/s². Settling is real and controllable.

**(b) Gate count: 0/5 in every run — the settled start does NOT rescue the line.** Two residuals
remain, now cleanly separated:

1. **Systematic**: every run flies RIGHT of every gate (pilot-confirmed at g0/g1/g2) — a
   repeatable full-trajectory offset/rotation vs the champion's line, present regardless of
   settle. The champion's un-reproduced 2.83 s armed-silent post-GO period is one candidate; a
   small constant plant/wire asymmetry is the other. This is exactly the residual the laptop
   side said it would fit from tape kf_speed vs replay IMU — all inputs for that fit are now
   committed.
2. **Chaotic**: despite wake-0 states matched to ~1e-3 and bit-equivalent commands (±1-row
   straddles), outcomes still diverge macroscopically (199 vs 271 vs 297 rows; gate-1 strike vs
   two distinct env crashes; q1 gyro maxes 0.23–0.42 rad/s from single-tick transients). Matched
   initial conditions buy ~seconds, not the full lap: **pure open-loop replay has a usable
   horizon of roughly 3–5 s in this sim** before straddle-level jitter amplifies through
   gate-skims.

**(c) Consistent misses + gates still 0 → committed logs and STOPPED.** No pre-roll hand-tuning,
no tape edits, no player changes.

## Implications for the ratchet

- Determinism verdict stands (state-conditional), and initial state is now *controllable* — the
  GO-delay protocol pin is worth keeping regardless.
- But a full-tape open-loop reproduction is dead as a strategy: with launch states matched to
  measurement precision it still bifurcates mid-course. **Per-gate closed-loop correction — the
  arrestor + `--truncate-at-gate` segments — is load-bearing**, with open-loop spans kept under
  ~3–5 s (one gate-to-gate leg fits comfortably).
- The systematic right-offset must be fit/explained before segment tapes are cut, or every
  segment inherits it from its splice point.

## Ops notes

- Mid-probe the sim fell into the frozen-limbo trap (sim clock pinned at 2.446 s, stale race
  aging in wall-time, video wire dead) after a sim restart; two pilot launches late-joined the
  residue and stalled (sessions discarded). Full bring-up to a live waiting room resolved it.
  The clean-attach checklist (pilot attaches to a LIVE waiting room, `started=False pos=y`, GO
  after) held for all three counted runs.
- Replay-mode sessions still write no ego_obs.jsonl; `sysid_vq2_log.csv` + `meta.json` +
  `mavlink.tlog` committed per run.

## MEMORY-DELTA (≤10 lines)

1. P0.2 2026-07-18: pre-GO settle time on the 17° block is REAL and controllable — ~3 s wait
   collapsed wake-0 launch-state deltas 1000× (accel_z 1.7→0.001 m/s²); pin "GO ≥3 s after
   reset" as protocol for any reproduction flight.
2. But settled starts did NOT fix the tape: 0/5 gates ×3, ALL runs fly a systematic RIGHT-offset
   line (pilot-confirmed g0/g1/g2), and matched-start runs still bifurcate mid-course
   (199/271/297 rows) → **open-loop replay horizon ≈3–5 s**; full-lap open loop is dead;
   arrestor + truncate-at-gate are load-bearing for the ratchet.
3. `race_start_boot_time_ms` = GO stamp, NOT race creation — REPORT2's 2.83 s champion "settle"
   is actually a post-GO armed-silent warmup (policy first-command delay), un-reproducible
   ops-only; the settle knob that worked is pre-GO block time. Laptop side now fits the
   right-offset from the committed logs (their plan (c)).
