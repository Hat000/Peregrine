# Peregrine — project close-out

**Anduril AI Grand Prix, autonomous drone racing. May–July 2026. Closed 2026-07-28.**

The goal was to fly all 20 gates of the course autonomously. We did not get there. This file is
the honest account: what was established, what was refuted, how the instruments lied, and where
everything lives. It is written for whoever opens this repo next — including us, later.

---

## 1. The result

| | |
|---|---|
| Recorded flights | **697** |
| All-time best | **gate 9** (`data/runs/20260722_031742_record9_cfg_f1`) |
| Flights reaching gate 8+ | 3 |
| Flights reaching gate 6+ | 18 |
| Flights reaching gate 4+ | 148 |
| Corpus mean max gate | 1.964 |

Max-gate distribution across all 697 flights:

```
gate  0:  179    gate  5:   49
gate  1:  159    gate  6:    9   <-- the wall
gate  2:  114    gate  7:    6
gate  3:   97    gate  8:    1
gate  4:   81    gate  9:    2
```

**The wall is at gate 6.** 49 flights reached gate 5; 9 reached gate 6. Nothing ever passed gate 9,
so gates 10–19 were never observed at all — no obstacle survey, no failure data, nothing. Any claim
about the second half of the course is extrapolation.

The final capability estimate before close was **P(20 gates) ≈ 4.8e-04**. A 50% lap needs per-gate
survival ≥ 0.9659 against a measured clean-gate 0.8075. The gap decomposed as **28.3% obstacle band
(deploy-only) · 45.2% sim↔wire mismatch (both) · 26.5% sim ceiling (training-only)** — which is why
neither more training nor more deploy tuning alone was ever going to close it.

---

## 2. What we established

### 2.1 The gate is flown blind, and that is camera geometry — not a control failure

The single most useful finding in the project.

- **0 of 899 confirmed gate passes ever got a camera fix inside 1.19 m.** Median last-sighted range
  ~1.8–2.1 m. Reproduced by two independent instruments.
- Camera is **HFOV 90° but VFOV only 58.7°**, axis pitched **+20° up** (`src/racer/frames.py:243-258`).
  The gate leaves frame **vertically** first.
- Training goes blind at the *same* range (1.72 m vs the wire's 1.78 m), so "teach it to commit
  blind" was never the gap.
- 🛑 **The closed form `|el| + atan(0.75/R) = VFOV/2` predicts the cutoff to +0.04 m but by the WRONG
  MECHANISM.** At the 4-of-8-keypoint threshold the bottom corners are long gone; what binds is the
  **outer ring at ±1.35 m** leaving frame. Right number, wrong physics — do not extend it.
- 🛑 **Vertical asymmetry runs opposite to intuition:** a gate *below* you stays detectable to
  1.185 m; a gate *above* you goes blind at 1.550 m. Training agrees — shared model, not a bug.

**Consequence:** every deploy aim/trim knob acts *before* the interval that decides the outcome.
That is why a long series of aim knobs each moved a mean and none moved the kill rate.

### 2.2 What kills is scatter, not bias

At the last sighted fix, passes and gate-strikes are nearly the same distribution: **lateral AUC
0.582**, vertical 0.676, **L-inf 0.736** (the only real discriminator). Blind metres, speed, and body
rates carry **no** information (AUC ~0.42–0.53). **40% of confirmed passes sit outside 0.45 m
laterally and pass anyway.**

Also: the gate aperture is **not** 0.75 m. Training threads `0.75 − body_radius` with
r ~ U[0.28, 0.38], so the effective half-aperture is **0.37–0.47 m**. Use 0.45.

### 2.3 The loop rate finding — and why the obvious mechanism was false

Commanded rate 30 beats 40 decisively: **mean max gate 2.460 (n=189) vs 1.769 (n=481)**.

🛑 It is **not** "30 matches the 30.03 Hz training dt." The loop is **compute-bound and never
achieves its commanded rate**: commanded 30 → 21.72 Hz achieved, commanded 40 → 25.34 Hz. So
commanding 30 lands *further* from the training cadence and still wins.

✅ **The mechanism is vision freshness** — fresh-fix fraction 94.9% at commanded 30 vs 75.6% at 40.
Freshness is the one lever that survived every stratification (b = +2.84…+3.89, z = +5.2…+8.5).

### 2.4 A recipe pin is a train/deploy contract

`_V1_RECIPE` pinned `rate: 40.0` while training ran dt 0.0333 s = 30.03 Hz, and the knob's own help
text said 30. Because `rate` sits inside `_recipe_managed_keys()`, **every model pick silently
re-armed the wrong value.** Same class of bug: `ego_stale_horizon` pinned 0.6 against a trained 0.5,
and v2.0 shipped with **no recipe entry at all** — its six flights drifted across three pitch-clamp
values and were never one cohort.

Now pinned and guarded by `tests/test_recipe_train_deploy_contract.py`, which tests the *contract*
rather than the numbers.

### 2.5 Two invisible obstacles, and a dodge that works

- Obstacles sit **14.49 m short of gate 4** (n=27) and **15.12 m short of gate 5** (n=17). Gates 0–3
  are clean. Confirmed at the impact: 4/4 gate-4/5 band deaths hit COLLISION id **1002 = ENVIRONMENT**;
  0/3 gate-0–2 band deaths did (p = 0.029).
- The dodge `ego_aim_offsets 4:0,3;5:0,3`, release 12.0, fade 0.0 works at **both** gates:
  gate 4 **8/9 vs 6/20, Fisher p = 0.0052**; gate 5 band deaths **5/6 → 3/23, p = 0.0025**.
  It carries a **by-construction placebo** — the offset can only arm at gates 4/5, so gates 0–3 are a
  control that cannot move.
- 🛑 **Raising the dose is iatrogenic:** mid-approach deaths **0/13 at +3/+4 vs 4/10 at +5/+10**
  (p = 0.0237). Lateral at gate 5 does nothing (**p = 0.67** over an already-flown 15-flight sweep).
- 🛑 **The mechanism is not fully understood.** On leg 3→4 the dodge lifts **+1.09 m** at the obstacle;
  on leg 4→5 it lifts only **+0.17 m** there and still works. Whatever protects gate 5, it is not
  altitude. **Do not tune these by eye.**
- Gates **6 and 8 are geometrically excluded** — a 14.5 m obstacle needs a >15 m leg, and theirs are
  15.4 m and 10.7 m against 20.6/20.7 m at gates 4/5. **Gate 7 is unsettled at n=8.**

### 2.6 Per-tick compute is a hidden experimental variable

Per-tick work was a rock-steady **34.9 ms across three sessions (n=144)** and then **48.7 ms** on
2026-07-28 — a 40% regression from machine state, not from any knob. It depressed *both* cohorts
flown that night and invalidated two adjudications.

**Log and check `tick p50` before trusting any cross-session comparison.** It is the cheapest
confound to rule out and the easiest to miss.

---

## 3. What we refuted

Negative results, kept because each one cost real time to establish and would otherwise be re-run.

| Claim | Verdict |
|---|---|
| **L-inf lateral pricing** would fix gate strikes | **Dead.** Already ran at real dose in v1.8 (`pass_margin_final_m 0.75`, `lat_weight 2.0`, 18k × 4 arms). `pass_offset` **flat 0.174–0.179** across five runs and both apertures, with 2.5× the needed power. |
| **"Train it to commit blind"** | **Refuted before it cost a GPU run.** Training's own 8-keypoint `gate_detectable` goes blind at 1.72 m vs the wire's 1.78 m. It already models the blackout. |
| **"The drone flies too fast to see"** → arm `ego_speed_gov` | **Refuted, n=1910 approaches / 92 strata.** `corr(speed, last-sighted range) = +0.006`; last fix flat at 1.86–2.08 m across speeds **2.56 → 16.52 m/s**. Blind *distance* is FOV-fixed, so slowing only stretches blind *time*, which kills independently (b = −2.31, z = −2.67). Survival vs speed is an inverted U peaking at **7.0–8.0 m/s** and the fleet already flew 6.58 — **below** the peak. The old help text's "try 5,6.5" would have dragged survival 0.839 → 0.731. |
| **`ego_fix_gain` 0.15** (training-parity low-pass) | **Worse.** Direction consistent within-session; magnitude confounded by §2.6. K≈0.154 is only meaningful at training's 30 Hz fix rate — on a wire delivering 7–15 Hz it is ~4 m of lag, and the deploy propagator drifts without a hard reset. Time-constant-matched value would be **~0.5**, never tested. |
| **`ego_aim_fade` > 0** | **Worse, flown.** A fade turns one discrete step into a *sustained false downward velocity* (3 m over ~0.2 s of travel) that the policy has never seen. |
| **"All lateral kills are LEFT"** (p=0.031, n=5) | **Dead at n=9.** Next flights put 3 of 4 RIGHT; pooled 6L/3R, p=0.51. |
| **"9° yaw boresight"** | **Withdrawn.** Lateral error shrinking on approach is just the loop working. |
| **Anti-clip terminal penalty** | Rejected — the curriculum already records collisions ~0.5 immune to a −20 clip penalty. |
| **Mirrored-course probe** | Dead — the course is fixed. |

---

## 4. How the instruments lied

**Ten premises were withdrawn in three days. Every single one was an instrument error — not one was
a wrong number.** This section is the most transferable thing in the repo.

1. **Never read body-frame gate geometry without removing the drone's own attitude.** Pitch coupling
   alone calls **66.2% of confirmed passes** a vertical strike; levelled, it is 6.0%.
   `R_level = Ry(pitch) @ Rx(roll)`, and **true roll = −obs[3], true pitch = −obs[4]**.
2. **Two frame conventions live in one log line.** Logged `rel_flu` is TRUE body FLU *unflipped*
   (neg lateral = gate right), but `obs[0:3]`, `obs[3:5]`, `obs[5:8]`, `obs[11:14]` are
   **virtual-flipped** by `diag(-1,-1,1)` (`ego_obs.py:36`). So **true v_left = −obs[1]**.
3. **The gate seam has two forms and `gate_index` guards neither.** (a) the *advance* seam, where
   `rel_flu` jumps to the next gate — kill it with a fixed **range band**, never a time window;
   (b) the *mis-lock* seam — gate it as a **distance**: reject a consecutive-fix lever jump > 1.5 m
   (leak/reject 0.14%/3.72%, vs 1.45%/9.76% for a 20 m/s speed gate). Exclude the aim-offset release
   tick, whose 3 m step trips the guard by construction.
4. **Checkpoint pooling manufactures effects.** A +3.13 m lateral effect at **t = +4.64** collapsed to
   **−0.45 (t = −0.46)** inside a single checkpoint+rate stratum. Stratify, then believe.
5. **Falsy-zero deletes your corpus.** `int(x or -1)` maps `gate_index == 0` to `−1` and silently
   dropped **34%** of the flights — and gate 0 was exactly where the launch confound lived.
6. **A circular instrument confirms itself.** Measuring per-tick range when the lever is *propagated*
   while unseen fills the low-range bins with exactly the not-seen ticks. Compare against a
   **measurement** (`pose_seen`), never against another propagated belief.
7. **Range-match or be fooled.** An n=16 pass-point statistic inverted a verdict purely by mixing ranges.
8. **n ≤ 5 direction counts are worthless.** See the LEFT-bias entry above.
9. **Reproducing a number is not verifying it.** A −0.110 slope was re-derived with the *same*
   confounded instrument and called confirmed. Name the artifact that could produce this exact sign
   and magnitude, then stratify by that artifact's own size.
10. **Reading arm membership from the log conditions on surviving to the arm.** It put a
    by-construction placebo at 24/24 (p=0.002). **Read the dose from the log, the arm from the config.**
11. **A probe that runs is not a probe that measures.** Hand-built "at rest" observations are OOD —
    the tell is every arm returning ~zero collective.
12. **Adjudicate from the run's own `.hydra` config and stage dict — never from launcher source.**
13. **Calibrate any strike classifier on confirmed passes first.** A classifier's rule order is a
    causal claim, and ranking failures by exposure always ranks gate 0 first.

Precedence when designing a guard: **by construction > right variable > swept > holds at your cut.**

---

## 5. Where things live

```
src/racer/ego_obs.py      the 21-dim observation builder — masking parity with training,
                          aim offsets, geometric detectability. Read its module docstring first.
src/racer/frames.py       camera intrinsics/extrinsics, the +20° mount, boresight bake
rl/fly_rl.py              the deploy flight loop; writes meta.json (incl. args_all provenance)
rl/peregrine_train_ego.py training entry point
rl/launch_v19/20/21.sh    training launchers; each header documents its own arm and prechecks
tools/pilot_panel.py      the pilot GUI. MODEL_DEFAULTS + _recipe_managed_keys() ARE the
                          train/deploy contract — read §2.4 before touching a pin.
configs/vq2_coarse_map.json   hand-surveyed per-gate turn priors (9 rows; the course has 20)
data/runs/                the 697-flight corpus: meta.json + ego_obs.jsonl per flight
tests/                    2077 passing. test_recipe_train_deploy_contract.py pins the contract.

scripts/vision_horizon/   the FOV/blind-approach instruments (§2.1)
scripts/postimpact/       kill-axis and contact forensics, calibrated on confirmed passes
scripts/adjudicate/       cohort adjudication: rate, dodge, fix_gain, det_geometric, climb profiles
                          All accept PEREGRINE_RUNS=<path> to point at an external corpus snapshot.

handoff/geodet_report.md            geometric detectability: build + verification
handoff/gate5_obstacle_report.md    the two obstacles, characterised
handoff/speed_hypothesis_report.md  the speed refutation, in full
COMMANDER.md                        working doctrine accumulated across the project
```

**COLLISION ids:** `1001 = GATE`, `1002 = ENVIRONMENT` (`src/racer/race_outcome.py:11`). Discriminate
a real impact from a proximity advisory with **`threat_level == 2 AND impulse > 1.0`** — most 1002
rows are advisories, and the raw id split is not evidence on its own.

**Hardware reality:** no magnetometer and no barometer on the VQ2 wire (HIGHRES_IMU
`fields_updated=63` = accel + gyro only). **Yaw and altitude must come from vision.** Any code path
assuming mag or baro is dead.

---

## 6. Still open

1. **`ego_det_geometric` was never cleanly adjudicated.** Built, verified against the real training
   source (delta 0.000 m on 8 straight-in geometries; 0.0–0.2% disagreement across 564 flights and
   36,967 fixes), and it *fires* correctly — 83.9% of sub-2 m ticks zeroed vs 7.4% control. But its
   only cohort flew on the degraded box of §2.6. **The missing cell is one same-session control:
   det-geometric OFF, `fix_gain` 1.0, everything else identical.** That is the cheapest open question
   in the project and the one most likely to matter.
2. **Training arm 1, `+env.ego_obs_coast=true`** — smoke passed its precheck (`ego_obs_coast: true`
   confirmed in the run's own `.hydra/config.yaml`); the 5-seed fan-out never ran. Motivation: training
   masks `obs[11:16]` to zeros once blind while the wire feeds a filled coasted lever (94% of last
   ticks). The policy has never trained on the state where every gate outcome is decided.
3. **The 07-28 per-tick regression** — 34.9 → 48.7 ms. Never diagnosed. Suspect a leftover `fly_rl`
   or a second `pilot_panel` process.
4. **Loop rate 25 or 20** — untested, and directly implied by the freshness mechanism (§2.3).
5. **Gate 7** — needs ~30 engagements with the post-impact recorder to settle whether an obstacle exists.
6. **Gate 5's dodge mechanism** — it works without lifting. Unexplained.
7. **v2.0 vs v1.9 on the wire** — sim says v20Vs0 wins on both course types (vert 5.927 vs 4.949,
   flat 6.048 vs 5.879) but v2.0 has never flown a clean rate-30 cohort.
8. **Dead code:** `apply_contact_kill` (−50) sits only in the `else:` legacy branch while
   `_use_refined_b` defaults True — so it is dead on every shipped ego run.

---

## 7. Closing note

The course was never fully seen. Gates 10–19 have no data behind them at all, and the wall at gate 6
was never broken. That is the plain result.

What is worth keeping is mostly the negative space: a set of hypotheses correctly killed, and a list
of ways the instruments produced confident, wrong, reproducible numbers. The findings in §2 were
earned by being wrong first, publicly, with the corrections written down. If this is ever re-opened,
start at §4 — it will save more time than §2 will.
