---
name: deploy-emission-2026-07-22
description: "SSOT for the 2026-07-22 cycle — v1.8 adjudication, the vision-emission fix result, the stale-frame flicker, patch 3, the terminal dive, and the next-gen training-env spec."
metadata: 
  node_type: memory
  type: project
  originSessionId: b85130f9-ac88-4db2-8c68-0e28b966cf80
  modified: 2026-07-23T00:25:02.129Z
---

# Deploy / emission cycle — 2026-07-22

Companion to [[perception-sim2sim-gap-2026-07-15]] (campaign SSOT). Everything here was
measured from wire logs or cluster runs; each claim names its instrument.

## v1.8 — SHIPPED 2026-07-22

Release **`ego-ckpts-v18-2026-07-22`** (Latest), 3 assets, target `v16-mount-acq-margin`.
Verified sha256 Adroit → local → GitHub → re-download, byte-identical, and every arm loaded
through `fly_rl.load_ego_actor` before upload. Commit `fda75f4`.

**Deploy lead `v18Qs1`. Designated alternate `v18Qs0`** — gentlest start in the fleet, for
about 0.4 gates of training reward.

### The release-dive gate (the number that justified shipping)

`scratchpad/replay_sweep_v18.py`, **425 logged flights** (9 skipped), worst nose-down
pitch-rate command in the 4 ticks after assist release. Negative = nose DOWN.

| arm | mean | p90 worst | MAX DIVE | dives harder than parent |
|---|---|---|---|---|
| v16Qs1 parent | −0.805 | −0.934 | −2.369 | — |
| v17 pooled | −1.225 | −1.457 | −2.194 | ~423 / 425 |
| **v18Qs1 (lead)** | **−0.801** | −1.133 | −1.885 | 248 / 425 |
| v18Qs0 (alternate) | −0.633 | −0.898 | −1.325 | **33 / 425** |
| v18Ws0 | −0.900 | −1.147 | −1.367 | 323 / 425 |

v1.7 dove harder than the parent on essentially every flight; **v1.8 returns the release
dive to parent level**, and the parent is the config that flew both 9-gate runs. So M1
**caused the release-pitch regression** even though it is **cleared on miss_rate** — two
different claims about the same mechanism, and collapsing them loses the finding.

### Connector footgun that blocked the first attempt

`[daemon] unauthorized` was **not** an auth failure — **two `adroit.py serve` processes were
both LISTENING on 127.0.0.1:8765**. `cmd_serve` sets `SO_REUSEADDR`, which on Windows lets a
second bind succeed on an already-bound port instead of failing, so a restart leaves the
stale daemon alive and answering with its old token while `.daemon.json` holds the new one.
Diagnose with `netstat -ano | findstr 8765` (two PIDs = this bug) and kill the OLDER PID;
do NOT re-run `serve` — that just adds a third. Also: `command_history.log` has reached
**289 MB**.

## v1.8 — WIRE RESULT (the honest test of the replay call)

Fengyou flew v1.8: **13× v18Qs0 (the gentle ALTERNATE) + 1× v18Qs1 (lead)**, one session, on
`origin/ratchet-arrestor-2026-07-18` (`v18pick_*` run dirs + `p1784764*.log` panel logs).
Forensics: `scratchpad/v18_wire_forensic.py`, `scratchpad/roll_limitcycle.py`.

* **RELEASE DIVE — FIXED on the wire, as replay predicted.** Worst post-handover nose-down
  pitch ≈ 0.00 on all 15; **0/15 died on a release dive.** M1-off transferred replay→wire.
* **BUT gate count flat/down.** v18Qs0 dist `[0,0,0,0,0,0,1,2,2,3,3,3,4]`, mean 1.38, median
  1, **no tail**; v16Qs1 same session ~2.5 with the fat tail to 9. Fengyou: "not much better
  than 1.6." The release dive was never the bottleneck.
* 🛑🛑 **THE WIRE KILLER = a CLOSE-IN ROLL LIMIT CYCLE, fully sighted.** 9/15 flights hit
  |roll rate|>1.4; **gate-area at peak roll median 0.95** (gate huge in frame), conf 0.74–1.00
  — NOT blind, the estimate is clean ⇒ CONTROL, not perception. The roll command oscillates
  with GROWING amplitude gate-over-gate (p18: ±0.3 g0 → ±1.1 g1 → ±2.0 g2 → slam); the only
  4-gate survivor is the one where roll stayed BOUNDED (<0.5). `roll_clamp=0` ⇒ unarrested.
  This IS Fengyou's "side gate slams."

### Two corrections this forces
1. **The coast/blind capability is NOT the top wire killer** (I had steered there). The
   dominant death is fully sighted ⇒ **close-in LATERAL STABILITY is the #1 RL-polish target.**
2. 🚩 **ADJUDICATION BLIND SPOT:** the census gates YAW + PITCH signflips/satur but **NEVER
   ROLL**, so a roll limit cycle passes selection invisibly and only shows on the wire.

### The fix (in flight)
The reward has RATE penalties (duty+jerk) for yaw + pitch that tamed those oscillations, but
ROLL has ONLY angle penalties (`rw_roll_recover`, `rw_att_roll`, `rw_cross_level`) — and
`rw_roll_recover=0.5` was ON in v1.8, so angle penalties do NOT damp the rate limit cycle.
**Adding `rw_roll_duty` + `rw_roll_jerk`** (mirror of pitch; ROLL_CMD_RAIL=3.0; action channel
**1** = roll) + a `ROLL_EVAL` line so adjudication gates roll. Lead with **jerk** (a smooth
60° turn-in has low |Δcmd|, the limit cycle has high |Δcmd|; normal roll rate <0.5 observed on
the survivor, deaths spike to ±2 — well separated). 🛑 **NO roll ANGLE fence/cap** — it
backfires on the course's 60° turns (`ego_reward.py`~L323). Rate penalty only, training-side.
Open question BLOCKED on Adroit: does the limit cycle reproduce in SIM (clean pose) or is it a
seeker-EMA-lag deploy artifact? `roll_swing` (YAW_EVAL, peak roll ANGLE) is a first proxy;
the new rate ROLL_EVAL is the clean measure.

## v1.8 — training adjudication

v1.8 = v1.7 with **M1 OFF only** (`HANDOFFFRAC=0.0`). Launcher `rl/launch_v18.sh` (`19c491a`).
Arms 3317670/71/72 on MIG, all COMPLETED ~6 h. Converged means, last 10% of 18000 updates:

| | parent v16Qs1 | v17Qs0 | v18Qs0 | v18Qs1 | v18Ws0 |
|---|---|---|---|---|---|
| miss rate | 0.0022 | 0.0770 | 0.0863 | 0.0828 | 0.0851 |
| gates passed | 5.81 | 5.61 | 5.38 | **5.75** | 5.15 |
| pass offset (m) | 0.1793 | 0.1744 | 0.1785 | 0.1759 | 0.1779 |
| center penalty | 0.173 | 0.199 | 0.168 | **0.163** | 0.167 |
| frame factor (M3) | — | 0.799 | 0.813 | **0.841** | 0.761 |
| blind-abort (M2) | — | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

* **M1 EXONERATED** on the miss-rate fork — turning it off never recovers the rate.
* **The regression is mostly DEFINITIONAL.** `miss_rate` is scored against the *annealed*
  margin (`aperture_margin_reclassify`, `peregrine_racing_ego.py:1922`), so M4's 1.0→0.75
  reclassifies passes as misses. It is not purely a metric — the downgrade routes to
  `rw_terminal_miss`, blocks the advance and counts against `n_passed` — so ~8.6% of passes
  are genuinely penalised. The policy simply did not respond.
* **M4 → REVERT to 1.0.** `pass_offset` is FLAT at 0.174–0.179 across 5 runs and 2 aperture
  settings. Real pressure, zero centering gain.
* **M3 = the keeper** (frame_factor climbing, best center_pen).
* **M2 inert in sim on every arm** while the wire is 23.7% blind ⇒ **sim never goes blind**.
* **Deploy pick `v18Qs1`.** Ckpts on Adroit:
  `/scratch/network/fl3689/diffaero/outputs/train/ego_dual_gate_fullstack_floor_pef16_seed{1_v18Q_s1,0_v18Q_s0,0_v18W_s0}_mig/checkpoints/actor.pth`.
  **The release pull died on a connector reset — publishing is UNFINISHED.**

## Vision-emission fix (Fengyou, `b13a07f`) — it worked

397 flights before vs 37 after (`scratchpad/cohort_compare.py`):

| | before | after |
|---|---|---|
| `valid_poses_empty` | 11.04% | **3.59%** (−67%) |
| blind ticks | 31.1% | 23.7% |
| longest blind gap p90 | 1.51 s | **0.63 s** (−58%) |
| gates/flight mean | 1.78 | 2.38 |
| `continuity_reject` | 11.28% | **12.13%** (worse) |

The −7.45 pt drop in `valid_poses_empty` accounts for ALL the blind-time gain — cleanly
attributable. **The bottleneck has moved off vision:** the seeker now discards 3.4× more
detections than the camera fails to produce.

## The stale-frame flicker — NOT FIXED, biggest remaining win

`scratchpad/stale_frame_drop.py`. **Every non-fresh tick is blind: 1473/1473 before,
581/581 after.** 8.6% of all control ticks repeat a camera frame and blank the emission
while a valid tracked estimate is in hand (83% of them). **483 of 581 log no reason at
all**, so they were invisible to the decision histogram — the earlier blindness breakdown
undercounted this category entirely. Now **36.4% of ALL blindness**.

Cause is control-loop period **jitter** (0.02–0.10 s) against a 33 ms camera, not a rate
mismatch: any tick arriving <33 ms after the last lands on a consumed frame. Fix: on a
non-fresh tick emit the **gyro-propagated** estimate — patch-2's `propagate()` already
exists and simply is not wired to this path.

## Patch 3 — shipped `4ca994f`

Branch `ratchet-arrestor-2026-07-18`, rebased onto Fengyou's vision commits, 120 seeker
tests green, `tests/test_seeker_pass_drop.py`.

**WP3a local pass detection** (`pass_drop_range_m=2.5`, 0=off). RACE_STATUS is **4 Hz**, so
`on_gate_advance` fires correctly but up to **250 ms late**. Measured on `20260722_031742`
gate 8: crossed the plane at k=455 (track 0.22 m); the track followed the gate **receding**
(0.22 → 3.09 m) while `gate_index` stayed 8 until k=463 — **10 ticks / 0.35 s of
`continuity_reject` with 1–2 live candidates on offer**, because a ~16 m step blows
`track_max_range_jump_m=6.0`. Same signature at gate 0 (track frozen 2.04 m, 6 rejects).
Rule: a track ≤2.5 m whose candidates ALL fail continuity has been flown through ⇒ drop and
re-acquire COLD (clears `_track_ever_locked`, so the stale re-acquire hint cannot steer the
new gate). **Deliberately does not loosen the continuity gate** — widening the jump
tolerance would buy the same recovery while admitting real flappers everywhere.

**WP3b candidate logging.** `seeker.jsonl` now carries up to 8 candidates as `{r,b,sel}`.
`n_cand` alone could not answer "why this gate and not the closer one in frame?" — asked
that about frame 1739, the honest answer was that the log could not settle it.

## Terminal dive — a new failure family

`fly_rl.py:766` is `rate_frd[1] = max(cmd, 0.0)`: the fence blocks **nose-DOWN only** and
always passes nose-up. So `pitchcmd == +0.000` means **the policy commanded a dive and was
blocked**. Inside 6 m of the gate: 034239 20/23 fenced · 034032 32/46 · 031742 120/207 —
and **~100% of those ticks have the gate ABOVE the drone.** The policy dives at gates that
are above it; the fence is **load-bearing** and is masking a training defect.

Deaths (all `record9_cfg`, 2026-07-22):
* **034239 bottom** — fence held it level under a gate 1.03 m above, then 0.29 s blind
  (6× `continuity_reject`) into the plane.
* **034018 top** — fence released, then +0.34…+0.41 nose-up over-climb, on an estimate
  whose range read 2.49 → 3.58 m *while closing at 7 m/s*.
* **034032 left** — lateral +0.38 → −0.35 overshoot during a 5-tick blind window with roll
  commanded −1.05…−1.36 on a stale estimate.
* **034842 bottom** — different animal: 335 ticks stuck at gate 0, 13.0 m/s (over the
  12 m/s abort), emitted lateral jumped −5.04 → +5.18 m (≈48° bearing) and was ACCEPTED.

Footage windows (time from first tick): 034239 **1.21→2.13 s**; 034032 1.07→2.16 s (gate 0,
survived) and **4.02→4.43 s** (fatal); 034018 1.37→1.95 s then fence release ~1.98 s;
031742 1.24→2.04 s and 21.72→22.04 s.

## Next-gen training-env spec (v1.9 HELD)

Fengyou's asks: alter gate position in frame · alter visibility/occlusion · match the newest
sim · randomize spawn velocity. Evidence added:

* **Sim blindness is per-frame Bernoulli-by-range; the wire is BURSTY** (p90 gap 0.63 s).
  Needs an occlusion/burst model, not a higher drop rate. An independent-coin process
  essentially never produces a 0.63 s gap.
* **Training feeds a CLEAN gate pose; deploy feeds the SEEKER's output** (EMA 0.5, coast ≤8,
  12% rejects, occasional wrong locks). The policy has never seen a *smoothly wrong* coasted
  estimate — exactly what it flies on during every blind window.
* **The deploy pitch and roll fences are NOT in training** — violates
  [[feedback-training-faithful-deploy]].
* **Legs: training `U[10,20] m` vs real 7.7–24.3 m; 44% of 16 measured legs outside range**
  (`scratchpad/leg_lengths.py`; path distances between pass points, so slightly overstated).
* **`gate_half_opening_m` is a single scalar** — every training gate is the same size, so
  size can never become the gate-2-vs-gate-3 disambiguator Fengyou asked for.
* `spawn_velocity_toward_gate()` exists (default off, `spawn_vel_frac`/`spawn_vel_max`) but
  points velocity AT the gate ⇒ trains "arrive fast", not "recover". With
  `standing_start_frac=1.0`, "at rest" is perfectly correlated with "episode start".
* **Re-measure blind/frame statistics AFTER the new vision system** — building sim against
  today's numbers bakes in a stale target.

## Instruments built this cycle (all in `wt-fix/scratchpad/`, committed)

`flight_story.py` (per-flight forensic: emission vs tracking attribution) ·
`cohort_compare.py` (before/after detection deltas) · `stale_frame_drop.py` ·
`blind_budget.py` (blind budget + decision histogram) · `frame_window.py` (seeker state
across a frame_id window) · `fence_check.py` · `dive_windows.py` (footage timestamps) ·
`leg_lengths.py` · `wire_miss.py` · `aperture_compare.py` · `v18_adjudicate.py` ·
`replay_sweep.py` · `handoff_elevation.py` · `tb_scalars.py` · `push_via_daemon.py`.

## Method lessons

* **A null result can be a bad measurement spec.** Two attempts to find the pilot's
  "hovers and waits" returned *no problem* — one threshold sat below the actual behaviour
  (2.04 m/s vs a 1.5 m/s cut), the other excluded the blind ticks where it happens. The
  third spec found it at 2.2% of flights, median 72% below cruise, blind fraction 100%.
* **A counter is not the work.** A monitor filtered `sacct` substeps with `grep -v '\.batch'`,
  but sacct truncates those IDs to `.bat+`; six substep rows tripped a ">=3 done" exit and
  it announced ALL ARMS TERMINAL while one was still training. Use `sacct -X`.
* **Name which cap you mean.** "max range 22" vs Fengyou's 30 were two different knobs
  (`max_acquire_range_m` vs `max_valid_range_m`); the acquire cap has no panel field.
* **A probe that RUNS is not a probe that MEASURES — and a hand-built obs is OOD.** The
  v1.8 pre-release gate was first written against a synthetic at-rest observation. It ran
  clean and ranked the arms — but every arm answered it with **~zero collective**, the tell
  that the point is off-distribution, and its ranking disagreed with the 425-flight replay.
  Launch-window questions are answerable ONLY over logged obs. Same family as the two
  false-null stall detectors above.
* **Decode channels from the wire path, never by hand.** `verify_v17_ckpts.py` unpacked the
  action as `(roll, pitch, yaw, thrust)`; the ego action vector is
  **`[thrust, roll, pitch, yaw]`** (`fly_rl.py:748`, `rate_flu = act[1:4]`), so its "pitch"
  column was ROLL. `verify_v18_ckpts.py` calls `policy_step` and reports `rate_frd`, so a
  channel cannot be mislabelled. Sign chain re-derived from code: with `virtual_flip` on,
  `rate_frd[1] = +act[2]` (`_ACT_FLU_TO_FRD = [1,−1,1]`, `_RZ_PI_BODY = diag(−1,−1,1)`) ⇒
  **nose-DOWN is `rate_frd[1] < 0`**, exactly the axis the fence clips.
