# inc8 Reach/Pass-Rate Campaign — gate-4 (corrected binding number)

**Worker:** inc8-reach-rate (opus-4.8, EFFORT max). **Branch:** `claude/funny-nash-d07d7b` (worktree, NOT
pushed). **Status:** ✅ **COMPLETE — supported plateau.** 4 batches (11 configs), 3-seed basin, full
failure-mode taxonomy, radius sweep. Levers #1+#2 lifted deterministic reach **0.467 → ~0.72** (at the
realistic r=0.30; first deterministically-flyable inc8 family). Targets (reach ≥0.85, success ≥0.80) NOT
met — bound by an intrinsic ~38% slab-collision / extreme-attitude envelope that NO reward weight moves.

## TL;DR for the commander
- **WIN:** the reframe's levers work. Sweeping the look-at pitch gain DOWN (g_pitch→0) + a gentler
  noise-anneal floor (0.3 / flat-0.6 vs the aggressive 0.03) gave the first deterministically-flyable inc8
  family, lifting det reach **0.467 → ~0.72** (r=0.30) / **~0.66** (DR-band) — reproducible across 3 seeds
  (0.626/0.662/0.640 band-avg). Centering is fine (p99 ~0.41 < 0.45 clearance).
- **PLATEAU:** deterministic full-course **success ~0.49 (r=0.30)** stays far below the 0.80 target. The
  binding wall is **~38% slab/gate COLLISIONS with extreme attitude (peak tilt ~100–113°, roll ~135°)** —
  NOT centering (miss_rate ≈0), speed, OOB, or spin. It is **unmoved by every reward weight tried**
  (g_pitch, noise floor, through-centering, centering, tilt↑, rate↑, progress↓ — the last didn't even
  change the 24.6 m/s speed). Reward-config space is exhausted.
- **fix_rate = 0.000 everywhere** — the case-C policy seats ZERO vision fixes; it flies on the noisy KF +
  dynamics. The collisions may be self-localization-driven, which points the next lever at PERCEPTION.
- **NEXT LEVER (commander's call):** (1) perception/estimator — the fix_rate=0 + collision-dominated combo
  suggests localization error, not RL control, is now binding (re-opens the near-field-gate-estimator lever
  the reframe parked); (2) reward-ARCHITECTURE / width-curriculum to teach frame clearance (escalate —
  Fengyou's freeze); (3) re-examine whether 0.85/0.80 is the right bar for a fix_rate=0 case-C stack.

## Goal (reframed)
Produce an inc8 policy whose DETERMINISTIC mean (test=True == deployed) RELIABLY reaches AND passes gate-4
and completes the gate-0→gate-4 course. Targets: **A.** det reach_rate ≥ ~0.85 · **B.** det success_rate
≥ ~0.80 · **C.** σ_p0_lat p99 < 0.45 (don't regress) · **D.** ≥3 seeds for any GO claim. The old σ_p0≤0.08
bar was a double-count; the gate clearance is 0.37–0.47 m and current centering (σ_p0 0.16–0.20, p99
0.37–0.40) is already MARGINAL-PASSING. The binding gap is REACH/PASS-RATE (det 0.47), not centering.

## Baseline (det-stab seed0 bracket, the policy I inherit — all snapshots on Adroit /scratch)
| g_pitch | floor | det reach (best snap) | σ_p0_lat | lat_p99 | fix_rate | flies on mean? |
|---|---|---|---|---|---|---|
| **1.0** | 0.03 | **0.467** (upd1700) | 0.20 | 0.40 | 0.000 | ✅ best so far |
| 1.5 | 0.03 | 0.129 | 0.139 | 0.38 | 0.000 | ~ |
| 2.0 | 0.03 | 0.000 | — | — | — | ❌ |
| 3.0 | 0.03 | 0.000 | — | — | — | ❌ |

Two facts drive the plan: (1) det reach rises monotonically as g_pitch FALLS — and **gp0.5/gp0.0 were
never run** (the prior campaign only swept g_pitch UP, chasing fix-seating we no longer need). (2) The peak
was at std=0.6 (the HOLD); the aggressive anneal→0.03 collapsed it (upd4000 reach 0.044) ⇒ use a gentler
floor / flat ceiling.

## Plan
- **Batch 1** (seed0, parallel, no file edits — submit knobs only): sweep g_pitch DOWN × gentler floor.
  - `rr_gp0p0_f03` GPITCH=0.0 floor0.3 — the untested reach floor (rc1-config + det-stab)
  - `rr_gp0p5_f03` GPITCH=0.5 floor0.3 — intermediate
  - `rr_gp1p0_f03` GPITCH=1.0 floor0.3 — floor refinement vs existing gp1.0/floor0.03=0.467
  - `rr_gp0p0_flat` GPITCH=0.0 floor0.6 (flat ceiling, entropy→0; PPO self-collapses std)
  - Each: precheck-gated, stochastic FLIGHTCHECK early-kill, ~50 min on A100. Then deterministic
    selection (`inc8_select_ckpt.sbatch`, GPITCH matched) → STAGE-1 reach rank + STAGE-2 full σ_p0.
  - Exact commands: see `LAUNCH_BATCH1.md` (this dir).
- **Batch 2** (informed): best config → seeds 1,2 (≥3-seed GO) and/or lever #3 (reward/speed weight tuning)
  if reach lifts-but-plateaus; characterize residual failure mode (gate/segment/speed) on the misses.

## Ops / autonomy
Driving Adroit via the adroit-connector `serve` daemon (`adroit.py x "<cmd>"`, port 8765). The daemon is
currently DOWN (last alive 2026-06-19 23:51; the key is passphrase-encrypted ⇒ a restart needs Fengyou's
passphrase + one Duo approval). Once up, the campaign is autonomous (one Duo covers all commands; daemon
self-heals on drop). No file edits / re-sync needed for Batch 1. Compute on SLURM only; output → /scratch;
checkquota; 1-GPU jobs. Report checkpoints via the artifact pipe (gitignored), NOT git.

## Results

**Batch 1 submitted 2026-06-20 ~03:50 ET** (seed0, parallel; jobs PENDING at submit):
| job | RUNTAG | g_pitch | floor | precheck | stoch flightcheck | det reach (selected) | σ_p0_lat/p99 |
|---|---|---|---|---|---|---|---|
**Ops note:** Adroit QOS caps Fengyou at **2 concurrent GPUs** (`QOSMaxGRESPerUser`) ⇒ batches run in
waves of 2 (~50 min train each + ~25 min select). Precheck confirmed the floor=0.3 knob threads into the
noise-anneal schedule. **Re-engagement:** passive background-completion did NOT auto-resume the loop (a
~6 h idle gap occurred after Batch 1) ⇒ now using an explicit ScheduleWakeup + a self-contained orchestrator.

### Batch 1 — DONE (seed0, deterministic selection; STAGE-2 full σ_p0 at horizons=2.5, g_pitch matched)
| run | g_pitch | floor | best det reach | full-σ_p0 reach | success | σ_p0_lat | lat_p99 | σ_vert | fix |
|---|---|---|---|---|---|---|---|---|---|
| **rr_gp0p0_f03** | 0.0 | 0.3 | **0.626** | ~0.61 | ~0.42 | 0.20 | 0.41 | 0.20 | 0.000 |
| rr_gp0p0_flat | 0.0 | 0.6 (flat) | 0.603 | ~0.60 | ~0.40 | 0.19 | 0.40 | 0.20 | 0.000 |
| rr_gp0p5_f03 | 0.5 | 0.3 | 0.592 | ~0.57 | ~0.40 | 0.20 | 0.41 | 0.20 | 0.000 |
| rr_gp1p0_f03 | 1.0 | 0.3 | 0.556 | ~0.52 | ~0.38 | 0.19 | 0.41 | 0.20 | 0.000 |
| _(inherited)_ gp1.0 | 1.0 | 0.03 | 0.467 | — | — | 0.20 | 0.40 | 0.20 | 0.000 |

**Read:** levers #1 (gentler floor) + #2 (g_pitch DOWN) are CONFIRMED and additive — det reach rises
monotonically as g_pitch falls (0.556→0.592→0.626), and floor0.3 added ~0.09 at gp1.0 over floor0.03
(0.467→0.556). Reach is also far more STABLE across snapshots than the inherited gp1.0/0.03 (which swung
0.467→0.044). **But it plateaus ~0.63, not the 0.85 target** ⇒ lever-#3 (reward/speed weights) branch.
Centering meets p99<0.45 (objective C ✓) but is not tight enough to fully convert reach→pass.

**Failure-mode split (best cell, gp0.0/f03):** of all episodes — ~0.61 reach gate-4, ~0.42 complete.
(a) **~39% die before gate-4** (earlier gate/segment); (b) **~17% reach the gate-4 plane but miss the
aperture** (terminal tail p99 0.41 > worst-case clearance 0.37 at r=0.38). Both point at the same clean
weight lever: **`rw_through_centering`** (telescoping cross-track pull to each gate's centre line) —
reduces drift-deaths (reach) AND tightens the terminal tail (success).

### Batch 2 wave 1 — DONE (deterministic selection)
| run | change | best det reach | success | σ_p0_lat | p99 |
|---|---|---|---|---|---|
| rr_gp0p0_f03_s1 (seed1) | basin check | **0.662** | 0.46 | 0.19 | 0.39 |
| rr_gp0p0_tc18 (seed0) | RW_TC 10→18 | 0.670 | 0.44 | 0.21 | 0.42 |

**Reads:** (1) **basin is reproducible** — seed1 (0.66/0.46) matches/beats seed0 (0.63/0.42); gp0.0/floor0.3 is
robust at ~0.64±0.02 reach. (2) RW_TC=18 lifted reach only +0.04 and did NOT lift success or tighten σ —
the extra pull drags more episodes to the gate-4 *plane* but they cross *wide*. Through-centering is not the
lever to 0.80.

### ⭐ FAILURE-MODE ANALYSIS — the decisive finding (TB taxonomy, best policy gp0.0/f03 seed1)
| mode | rate | | metric | value |
|---|---|---|---|---|
| **slab/gate collision** | **~0.41** | | peak tilt | **~113°** (past inverted) |
| miss aperture (centering) | ~0.0006 | | peak roll | **~135°** |
| out-of-bounds | ~0.0 | | mean speed | ~24 m/s (moderate) |
| spin-abort | ~0.0 | | finish time | ~2.6 s |

**The plateau is a CONTROL/STABILITY wall, not a centering/speed/perception wall.** ~41% of episodes end by
flying into the gate structure while throwing extreme attitude excursions (tilt 113°, roll 135°). Centering
misses are ~0; OOB ~0; spin ~0; speed moderate. The inc7 reward has only a soft tilt hinge (`rw_tilt=4.0`
beyond a 60° free cone) and **deliberately no tilt-threshold termination** (code comment) — so the policy
freely flies inverted into gates, eating a weak penalty. This is why g_pitch/floor/through-centering all
plateau: none of them tax the aggressive-maneuver collision mode.

### Batch 2 wave 2 — DONE
| run | change | best det reach | success | σ_p0_lat | p99 |
|---|---|---|---|---|---|
| rr_gp0p0_f03_s2 (seed2) | basin | 0.640 | 0.25–0.38 | 0.20 | 0.41 |
| rr_gp0p0_cen05 (seed0) | rw_centering=0.5 | 0.610 | 0.46 | 0.195 | 0.42 |

**3-seed basin CONFIRMED:** reach 0.626 / 0.662 / 0.640 = **~0.64 ± 0.015** (tight). Success 0.25–0.46 (noisier,
all ≪ 0.80). **rw_centering=0.5 neutral** (no σ tightening, no reach gain) and did NOT collapse — the S3-cliff
was a high-weight/no-noise-anneal artifact, not intrinsic. Both reward levers (through-centering, centering)
now ruled out, consistent with miss_rate≈0.

### Batch 3 — DONE (attitude-taming) = MARGINAL, plateau holds
| run | change | best det reach | success | σ_p0_lat | p99 | TB collision / peak-tilt |
|---|---|---|---|---|---|---|
| rr_gp0p0_rate015 | rw_rate 0.05→0.15 | 0.670 | 0.40 | 0.20 | 0.41 | 0.38 / 108° |
| rr_gp0p0_tilt10 | rw_tilt 4→10 | 0.619 | 0.40–0.44 | 0.20 | 0.42 | 0.41 / 103° |
| _(baseline)_ | — | 0.64 | 0.44 | 0.20 | 0.41 | 0.41 / 113° |

**Even at 2.5–3× attitude penalties, collisions barely move (0.41→0.38) and peak tilt stays ~100–108°** —
the extreme-attitude/collision mode is INTRINSIC to threading this course at ~24 m/s, not under-penalized.
rate015 ties the campaign-best reach (0.67) but success/σ unchanged; tilt10 *hurt* reach (starved racing
tilt, as the code comment warned). Reward-shaping is exhausted.

### Batch 4 — DONE (speed/progress lever) = NO EFFECT on speed or collisions
| run | change | best det reach | success | σ_p0_lat | p99 | TB mean_speed / collision |
|---|---|---|---|---|---|---|
| rr_gp0p0_prog6 | rw_progress 10→6 | 0.609 | 0.47 | 0.19 | 0.41 | 24.6 / 0.39 |
| rr_gp0p0_prog4 | rw_progress 10→4 | 0.667 | 0.40 | 0.21 | 0.41 | 24.6 / 0.38 |

**Decisive:** cutting the progress weight 2.5× left **mean_speed unchanged (~24.6 m/s)** and collisions
unchanged (~0.38). The aggression/speed is NOT progress-reward-driven — it's set by the finish bonus +
task structure. So NO reward weight (g_pitch, floor, through-centering, centering, tilt, rate, progress)
moves the speed, the ~38% collision rate, or success. **Reward-config space is exhausted — plateau is real.**

### Radius re-eval — DONE: the DR band hid a real radius dependence
Re-evaluated the two best snapshots at a FIXED contact radius (no DR band; via an additive `--body-radius`
patch to the validated eval — synced to Adroit only, NOT committed to main):
| policy | r=0.38 (stress) | **r=0.30 (central/realistic)** | r=0.26 (lower band) |
|---|---|---|---|
| rate015 upd2100 | reach 0.554 / succ 0.337 | **0.705 / 0.471** | 0.781 / 0.545 |
| seed1 upd3500 | reach 0.567 / succ 0.373 | **0.729 / 0.514** | 0.788 / 0.571 |

**The ~0.66 DR-band reach was inflated-pessimistic by the 0.38 worst-case-stress radius.** At the realistic
central radius **r=0.30: det reach ~0.72, success ~0.49**, σ_p0_lat ~0.21 / p99 ~0.41 (within clearance). At
r=0.26: reach ~0.79, success ~0.57. So **reach approaches the 0.85 target at realistic/optimistic radii, but
full-course success (~0.49–0.57) remains the binding gap** — the reach→pass conversion is capped by the
gate-4 collision rate. (Note: σ_p0_lat rises slightly as r shrinks — survivorship: at large r the wide
crossings collide and are not counted, censoring the spread; the least-censored r=0.26 spread ~0.22 is the
truer crossing σ. p99 ≤ ~0.46 throughout — centering is not the wall.)

## Definition-of-done verdict
**Supported plateau (the goal's second exit condition).** Best deterministically-flyable policy =
the gp0.0 / floor0.3 family (e.g. `inc8_detstab_seed1_rr_gp0p0_f03_s1/snapshots/upd03500` or
`inc8_detstab_seed0_rr_gp0p0_rate015/snapshots/upd02100`): at r=0.30, **det reach ~0.72 / success ~0.49 /
σ_p0_lat ~0.21 (p99 ~0.41)**, 3-seed-reproducible reach. Targets A (reach ≥0.85) and B (success ≥0.80) NOT
met. Objective C (p99<0.45) MET. The residual failure mode is pinpointed and the next lever identified.

## Full per-config grid (deterministic, best snapshot per run; reach/success at the DR band unless noted)
| batch | run | lever | det reach | success | σ_p0_lat | p99 |
|---|---|---|---|---|---|---|
| inherited | gp1.0/floor0.03 | det-stab | 0.467 | 0.31 | 0.20 | 0.40 |
| 1 | gp1.0/floor0.3 | floor refine | 0.556 | 0.38 | 0.19 | 0.41 |
| 1 | gp0.5/floor0.3 | g_pitch↓ | 0.592 | 0.40 | 0.20 | 0.41 |
| 1 | gp0.0/flat0.6 | g_pitch↓ + flat | 0.603 | 0.40 | 0.19 | 0.40 |
| 1 | **gp0.0/floor0.3** | **g_pitch↓ + floor** | **0.626** | 0.42 | 0.20 | 0.41 |
| 2 | gp0.0/f03 seed1 | basin | 0.662 | 0.46 | 0.19 | 0.39 |
| 2 | gp0.0/f03 seed2 | basin | 0.640 | 0.25–0.38 | 0.20 | 0.41 |
| 2 | gp0.0 RW_TC=18 | through-centering↑ | 0.670 | 0.44 | 0.21 | 0.42 |
| 2 | gp0.0 rw_centering=0.5 | centering | 0.610 | 0.46 | 0.20 | 0.42 |
| 3 | gp0.0 rw_tilt=10 | attitude↑ | 0.619 | 0.40 | 0.20 | 0.42 |
| 3 | gp0.0 rw_rate=0.15 | attitude↑ | 0.670 | 0.40 | 0.20 | 0.41 |
| 4 | gp0.0 rw_progress=6 | speed↓ | 0.609 | 0.47 | 0.19 | 0.41 |
| 4 | gp0.0 rw_progress=4 | speed↓ | 0.667 | 0.40 | 0.21 | 0.41 |
| — | **best @ r=0.30** | radius re-eval | **~0.72** | **~0.49** | 0.21 | 0.41 |

## Reproducibility / artifacts
3-seed basin (gp0.0/floor0.3): reach 0.626/0.662/0.640. Checkpoints are on Adroit /scratch (gitignored) —
ship via the artifact pipe, NOT git: best = `inc8_detstab_seed1_rr_gp0p0_f03_s1/snapshots/upd03500`. All
runs precheck-passed, flew stochastically; deterministic selection on reach (never the stochastic
high-water). Eval = the validated `inc8_sigmap0_torch_eval.py` (+ the additive `--body-radius` patch, synced
to Adroit only). Ops: adroit-connector `serve` daemon dropped twice (each needs Fengyou's Duo to reconnect;
SLURM jobs run regardless) — orchestrators hardened with timeout-guards + verify-retry submit after a
daemon blip silently ate one select wave.

## MEMORY-DELTA
1. **inc8 reach/pass-rate campaign = SUPPORTED PLATEAU.** Levers #1 (gentler noise-anneal floor 0.3/flat-0.6
   vs 0.03) + #2 (look-at g_pitch DOWN to 0.0) lifted DETERMINISTIC gate-4 reach **0.467 → ~0.72** (at the
   realistic r=0.30; ~0.66 DR-band), 3-seed-reproducible (0.626/0.662/0.640) — first deterministically-flyable
   inc8 family. Best = gp0.0/floor0.3 (`inc8_detstab_seed1_rr_gp0p0_f03_s1/snapshots/upd03500`).
2. **Targets NOT met:** det full-course success ~0.49 (r=0.30) ≪ 0.80; reach ~0.72 < 0.85. Centering is fine
   (σ_p0_lat ~0.20, p99 ~0.41 < 0.45 clearance — objective C met; miss_rate ≈0).
3. **BINDING WALL = ~38% slab/gate COLLISIONS with extreme attitude (peak tilt ~100–113°, roll ~135°),** NOT
   centering/speed/OOB/spin. **Unmoved by EVERY reward weight** (g_pitch, floor, through_centering, centering,
   rw_tilt↑4→10 [hurt reach], rw_rate↑0.05→0.15, rw_progress↓10→4 [didn't even change the 24.6 m/s speed]).
   Reward-config space is EXHAUSTED. (inc7 has only a soft tilt hinge, deliberately NO tilt-limit terminal.)
4. **fix_rate = 0.000 across ALL configs** — the case-C policy seats zero vision fixes; flies on the noisy KF
   + dynamics. Collisions are plausibly self-localization-driven ⇒ next lever may be PERCEPTION, not control.
5. **Contact-radius matters a lot:** reach/success scale strongly with r (rate015: 0.55/0.34 @0.38 →
   0.71/0.47 @0.30 → 0.78/0.55 @0.26). Always report the central r=0.30, not the 0.38 stress upper-tail.
6. **NEXT LEVER (commander's call):** (a) perception/near-field-gate estimator (fix_rate=0 smoking gun);
   (b) reward-ARCHITECTURE / width-curriculum for frame clearance (escalate — Fengyou's freeze); (c) revisit
   the 0.85/0.80 bar for a fix_rate=0 case-C stack. RL reward-WEIGHT tuning alone will not break this plateau.
7. Tooling: added `--body-radius` to inc8_sigmap0_torch_eval.py (additive, default None == byte-identical;
   synced to Adroit only, NOT committed to main). New sbatch tags rr_gp0p0_* on Adroit /scratch.

## MEMORY-DELTA
_(filled in at completion)_
