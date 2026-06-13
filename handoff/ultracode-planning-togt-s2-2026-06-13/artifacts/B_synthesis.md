# Phase B — SYNTHESIS: S2 trajectory-opt approach (recommended + runner-up)

**Author:** Phase-B synthesizer (opus-4.8), ultracode S2-planning fan-out · **Date:** 2026-06-13 · **For:** Fengyou
**Scope:** analysis only. No tracked source edited, no live sim, no SLURM, no network. All numbers re-derived natively
(`.venv`, `PYTHONPATH=src`) from repo data + the frozen Phase-A/B artifacts in this directory.

Inputs synthesized: two approach proposals (`minsnap_topp`, `HYBRID`) + two incumbent poles scored by the panel
(`MONOLITHIC RL`, `TOGT collocation`), the two design writeups (`B_approach_mpcc.md`, `B_approach_togt_cpc.md`), and
the two judge lenses (`B_judge_time.md`, `B_judge_robustness.md`). Both judge probes re-run natively and reproduce.

---

## 0. The decision in one paragraph

The panel agrees on a non-obvious conclusion: **on the corrected-aero plant, lap time is a TILT story, not a planner
story.** At the doctrine-allowed 60° style cone, EVERY trajectory-opt approach lands at ~9.6 s ideal — identical to
inc7's already-live 9.76 s — and the same cone relaxation (rw_tilt 96→48, free-cone 60→75–80°) that unlocks the 5–6 s
"fast" numbers would also speed the already-live monolith (inc5 unconstrained measured 6.9 s with NO planner). No
trajectory optimizer manufactures the envelope relaxation; that is a reward/doctrine lever gated on
LAPTOP-INC8-BINDING-GATE-VERIFY. Therefore **the synthesized verdict is NOT "build a new planner" — it is: keep the
live-confirmed MONOLITHIC RL as the S2 primary, retrained on the corrected plant for inc8 with the cone relaxed, and
build the DECOMPOSED (offline contact-free line + good tracker) stack only as a warm fallback.** The single
trajectory-opt approach worth a native Phase-C prototype is the **min-snap geometry + coupled-aero TOPP line
generator** (the decomposed stack's geometry rung — free, deterministic, fully native), with a **scipy-SLSQP toy MPCC
tracker** as the runner-up prototype that natively quantifies the realizability gap the whole field is blocked on.

---

## 1. Aggregate ranking across all three axes (time / robustness / integration)

The two lenses scored on different axes. Aggregating (time-lens score, robustness-lens score, integration cost):

| approach | TIME (B_judge_time) | ROBUSTNESS (B_judge_robustness) | INTEGRATION | aggregate read |
|---|---|---|---|---|
| **MONOLITHIC RL (inc7-class)** | comparator — 6.9 s unconstrained MEASURED beats most of the field | **7.5** (only LIVE-CONFIRMED validity) | LOW-MED (retrain, 1 subsystem, live) | **#1 overall** — fastest realized-NOW + only demonstrated validity |
| **minsnap_topp** | 4.5 (no time frontier; 7.0 s borrowed pad) | **6.5** (best PLAN-level contact-free; graceful) | **LOW** (half-built, pure numpy, <1 s) | **#2** — the free deterministic line generator / flywheel; native |
| **HYBRID (decomposed)** | 5.5 (best-calibrated, but 8.0 s = slower than monolith) | **6.0** (best incentive-level guarantee; unbuilt tracker) | **HIGH** (acados no-Windows + fresh Adroit) | #3 — right STRUCTURE for fallback; do not build speculatively |
| **mpcc (tracker)** | **6.0** (fastest DEFENSIBLE flyable: 5.8 s @60°) | (scored inside hybrid; same unbuilt-tracker risk) | **HIGH** (acados/WSL; toy hit ~20 m/s) | the *good tracker* the decomposed stack needs; unbuilt |
| **togt_cpc** | 4.0 (best ceiling, worst realizability-to-ceiling) | **4.5** (catastrophic standalone; falsified lines) | HIGH (WSL/C++) | bound + geometry SEED only; not deployable S2 |

**Cross-lens reconciliation.** The two lenses do NOT contradict — they agree on the structure and disagree only on
*which* unbuilt thing to bet on:
- TIME lens crowns **mpcc** "of the four" because it IS the tracker (attacks realizability directly; 5.8 s defensible
  vs hybrid's 8.0 s), but explicitly says the time lens ALONE does not justify switching off the monolith, and the
  fastest realized-NOW path is "relax cone + retrain monolith."
- ROBUSTNESS lens crowns **MONOLITHIC RL** outright (demonstrated > asserted validity; 7 live zero-contact sessions),
  with minsnap_topp #2 as the warm offline geometry flywheel.
- **Both converge on: monolith primary, decomposed stack as the warm fallback whose geometry rung (minsnap_topp) is
  free to keep alive and whose tracker rung (MPCC) is the expensive unproven piece.** That is the synthesis.

---

## 2. Recommended approach: MONOLITHIC RL (inc7-class), retrained on corrected plant for inc8, cone relaxed

**Why it wins the aggregate.**
- **Only option with LIVE-CONFIRMED contact-free validity** (robustness 7.5): standing 5/5 FINISHED, gate-3 barrier
  GONE (0 collisions offline AND 0/5 live), command saturation NONE (|tanh|≤0.547), graceful AND observed failure
  (slows/widens, never tumbles). Every rival's contact-free claim is plan-level or incentive-level; this one has
  seven live zero-contact sessions. On a lens that rewards demonstrated over asserted validity, this is decisive.
- **Fastest realized-NOW time.** inc5 unconstrained monolithic = 6.9 s twin, MEASURED — already beats minsnap's
  claimed 7.0 s (honest band 6.5–10 s) and hybrid's 8.0 s, and ties the envelope togt/mpcc need a from-scratch
  unbuilt tracker to reach. The corrected-aero point-mass bound is 4.72 s; the inc7→bound residual ~5.0 s decomposes
  into ~2.6 s style cone (a relaxation lever that helps the monolith too) + ~2.2 s planning/tracking/policy slack.
- **The decomposition's main reason to exist is already neutralized.** inc1's "backflip-dive" root cause (wrong plant
  + no DR + reward/termination conflict) is understood and FIXED; inc7 exhibits ZERO inversion. The HARD vs SOFT
  inversion-ban advantage decomposition uniquely offers addresses a pathology that no longer occurs.
- **Lowest integration cost of the viable options:** one subsystem, already live, retrain-only. No acados/WSL, no
  fresh obs layout, no three-subsystem bring-up.

**Debits (carried forward, not disqualifying).**
- SOFT (R4 tilt-hinge) not HARD inversion guarantee — could in principle regress under aggressive cone relaxation
  (this is exactly where the fallback earns its keep, §4).
- Narrow convergent basin (2/3 seeds viable) — budget MORE seeds for inc8-class retrains (already in memory).
- Fresh-respawn post-gate-3 sensitivity (bimodal laps) — the WINNER-VALIDATION RIDER, which caps ALL four equally
  and is rank-neutral; the monolith is the ONLY option that has already PASSED a live fresh batch.

---

## 3. Runner-up: DECOMPOSED stack = min-snap/TOGT offline contact-free line + MPCC (or RL) tracker

**Why runner-up, not winner.** Its honest ceiling (~5.4–5.8 s with a good MPCC tracker @ relaxed cone) is genuinely
faster than inc7's 9.76 s offline — but it is a THREE-subsystem build (line gen + tracker + estimator) whose
realizability is unproven, versus a monolith that is one subsystem and live. The TIME lens itself says the decomposed
stack's time ceiling "only justifies its integration cost if monolithic style/inversion pathologies re-emerge under
relaxation." That is the trigger condition, not a default.

**Its genuine wins (the reasons to keep it warm).**
- **Inversion structurally impossible** (planner property + tracker tilt-clamp) — the one thing the monolith's SOFT
  guarantee lacks.
- **Incentive-level contact-free guarantee:** arc-length progress reward over the planned contact-safe line removes
  the corner-cut incentive a pure progress-to-gate-centre monolith reward can create (the genuine decomposition win).
- **Fully deterministic offline geometry, LEGAL per-track line-iteration flywheel** — the cleanest determinism/audit
  story of any option (for the planner half).

**Its non-wins (corrected over-claims, so we don't over-credit it).**
- Shipped-4.55 s-line REUSE is FICTION: measured tilt p50 74.3° / max 170.4° / 26% inverted samples / 92% thrust-
  saturated / 51 m/s on the FALSIFIED linear plant. Geometry MUST be re-planned under corrected aero; the "static
  shipped asset / clean audit" story is not real until a new line is built.
- World-frame line couples perception East σ 0.47 m DIRECTLY to cross-track = 3.0× the 0.155 m binding gate-4 margin.
  Decomposition buys NOTHING here — same world-fix dependence as the monolith.
- Probably SLOWER than the live monolith unless the cone is relaxed AND a good (unbuilt) tracker materializes.

---

## 4. Best ideas to GRAFT from the non-winners (the point of the synthesis)

1. **(from HYBRID) Arc-length progress reward over an offline contact-safe reference line — cast the monolith as
   MPCC.** This is the "3rd option / HYBRID-monolithic" in the task. Keep the monolith's learn-the-line ceiling but
   replace progress-to-gate-centre with progress-along-Γ, so the tracker is never rewarded for cutting inside the
   frame. Grafts decomposition's incentive-level contact-free guarantee onto the live monolith at near-zero
   architectural cost (reward change, not a new subsystem). This is the highest-value graft.

2. **(from minsnap_topp) Use a min-snap/coupled-TOPP line as a FEASIBLE, non-saturated reference for tracker bring-up
   and as the offline line-iteration flywheel.** The shipped TOGT line is full-stick-saturated 92% of the lap (hostile
   to a tracker); a min-snap line with thrust headroom is the friendly reference that lets a tracker recover lag. Keep
   it warm as the free, pure-numpy, no-toolchain per-track re-planner (LEGAL determinism-per-track).

3. **(from minsnap_topp robustness probe) Plan the reference at DR-CONSERVATIVE envelope params (a_up_max ~70 not
   78.3, pooled c2).** TOPP is parameterised by (a_up_max, c2); planning slow at conservative values buys plant-error
   robustness for free via determinism — graft into whatever line generator feeds the fallback.

4. **(from mpcc) MPCC contour/lag formulation as the tracker IF decomposition is triggered.** 88% speed-keep is the
   most evidence-backed (Romero 2022) tracker assumption in the panel and the only candidate making contact-free an
   EXPLICIT tube constraint (realized contour-error budget 0.23 m at gate-4 w/ 0.14 m planned offset). It is the good
   tracker the decomposed stack structurally requires — the thing whose ABSENCE forces k=1.85 on the geometric tracker.

5. **(from togt_cpc) TOGT corrected-aero line as the offline contact-free BOUND + geometry SEED — never as a
   standalone deployable.** Its inscribed-circle 4.27 s bound and gate-margin machinery are the right way to certify a
   candidate line is contact-free; use it to seed/validate the decomposed line geometry, not to fly.

6. **(cross-cutting, from BOTH judges) The HARD FLOOR that caps everyone: the ESTIMATOR, not the planner.** Perception
   East σ 0.47 m = 3.0× the 0.155 m gate-4 binding margin, UNFILTERED. No planning architecture fixes this — the
   KF/estimator must reach <0.05 m 1-σ at the post-gate-3 37 m/s window or NO approach is valid at VQ2 race speed.
   Robustness work should prioritise the estimator over the planner regardless of S2 choice. (Latency 67 ms is
   validity-BENIGN: 0.77–1.93 mm cross-track at any speed ≤37.7 m/s, <1.3% of margin — fold into k, do not chase.)

---

## 5. Phase-C native prototype targets (pure numpy/scipy, no WSL/C++/Adroit)

The task asks for the 1–2 approaches that score best AND can be NATIVELY prototyped+validated for Phase C. Native
feasibility (verified against `speed_profile.py` docstring lines 10–17 + the two design files):

| candidate | native-prototypable? | evidence |
|---|---|---|
| min-snap geometry + coupled-aero TOPP **line generator** | **YES — fully** | `speed_profile.py` half-built (forward-backward TOPP, pure numpy); coupled-thrust envelope ~40 lines, already prototyped in `minsnap_topp_proto.py`; min-snap QP = one `scipy.linalg.solve_banded`. Ideal-lap-by-cone reproduces exactly natively. |
| scipy-SLSQP **toy MPCC tracker** | **YES — for prototyping** (production needs acados/WSL) | `B_approach_mpcc.md`: "A pure-numpy/scipy MPCC is possible for offline prototyping (scipy SLSQP / hand-rolled SQP)". Toy already attempted, reached only ~18–21 m/s (half the 38 m/s wall) — the gap IS the datum to close. |
| TOGT corrected-aero collocation | partial (read frozen CSVs natively; faithful regen needs WSL/C++ or acados) | `B_approach_togt_cpc.md`: numbers derived natively from CSVs, but the multiple-shooting NLP is the WSL/acados pipeline. |
| MPCC/RL production tracker | **NO** | acados does not build on Windows/py3.13 (`io.h`); RL-tracker needs fresh Adroit campaign. |

**Phase-C target #1 (highest value): min-snap + coupled-aero TOPP line generator.** It is the geometry rung of the
fallback decomposed stack AND the offline line-iteration flywheel the recommended monolith path also benefits from
(graft #2). Fully native, deterministic, <1 s. Prototype goal: emit a corrected-aero feasible, contact-free,
non-saturated reference line in the `peregrine.reference_line.v1` schema (drop-in for the RL progress reward and any
MPCC tracker), planned at DR-conservative envelope params (graft #3), with crossings dead-centre (full 0.37 m
contact-true band free at r=0.38).

**Phase-C target #2 (the realizability probe): scipy-SLSQP toy MPCC tracker on `rl_plant.step`.** The ENTIRE field is
blocked on one unmeasured number — can a good tracker recover 85–90% of the point-mass speed, or does it limp to the
k=1.85 (~8 s) reality? Three of three native trackers (minsnap geometric → None/inf, hybrid DF+PD → 5–35 m miss, mpcc
SLSQP → ~20 m/s) have FAILED so far. A focused scipy-SLSQP MPCC prototype that closes even part of the gap to the
38 m/s wall would (a) de-risk the decomposed fallback's central assumption and (b) measure the real k on THIS plant,
which currently has exactly one (bad) datum. This is the single most decision-relevant thing a Phase-C native run can
produce — it directly adjudicates whether the decomposed fallback's ~5.5 s ceiling is real or a paper number.

---

## 6. Recommended Phase-C sequence (what to actually do)

1. **Primary track (no Phase-C blocker):** retrain the monolith for inc8 on the corrected plant with the cone relaxed
   one ladder step (rw_tilt 96→48), grafting the arc-length progress reward over a min-snap reference (graft #1).
   Crown the winner only after a ≥3–5-lap fresh-reset live batch (WINNER-VALIDATION RIDER).
2. **Phase-C prototype #1 (native, cheap, high-leverage):** the min-snap + coupled-aero TOPP line generator — it feeds
   BOTH the monolith's progress reward AND the fallback's geometry rung, and is the legal per-track flywheel.
3. **Phase-C prototype #2 (native, decision-relevant):** the scipy-SLSQP toy MPCC tracker — measures the real k on the
   corrected plant and tells you whether the decomposed fallback is worth its acados/Adroit cost BEFORE you pay it.
4. **Do NOT build speculatively:** the production MPCC tracker (acados/WSL) or a fresh RL-tracker Adroit campaign —
   gate these on (a) prototype #2 showing a tracker can in fact recover ≥~80% of envelope speed AND (b) the relaxed-
   envelope monolith actually starting to cut corners (the SOFT-guarantee failure the fallback exists to catch).

---

## 7. Method / confidence

- IDEAL lap-by-cone + coupled-TOPP + gate-4 margin numbers: re-ran `probe_robustness.py` natively, reproduces exactly
  (60° 5.53 s / 80° 4.63 s; drag-wall-limited; geometry-tilt p50 54–60°, ZERO inversion) — **HIGH**.
- Tracker-failure evidence (three of three native trackers fail): read from artifacts + the minsnap proto returns
  None/inf — **HIGH** that naive trackers fail; these are lower bounds, not verdicts on a GOOD tracker.
- The g·tan(θ) vs 78.3·sinθ cone disagreement (drives the ~4.5 s spread at 60°): the g·tan(θ) cone is the honest
  binding constraint for an altitude-holding descending course (descent gives ~1.7% credit, not enough to license the
  full sin budget) — **MEDIUM-HIGH**, the single biggest modeling judgment in the panel.
- k-dilation realizability (88% / k=1.5–1.7): borrowed/projected, the **dominant uncertainty** across all four; only
  measured datum is k=1.85 (bad). This is precisely what Phase-C prototype #2 exists to measure.
- Native-prototypability claims: cross-checked against `speed_profile.py` docstring (toppra/acados `io.h` Windows
  build failure) + both design files — **HIGH**.

Bottom line for Fengyou: the trajectory-opt fan-out did NOT find a planner that beats the already-live monolith at the
allowed envelope — it confirmed the time gap is a TILT/cone-relaxation lever (helps the monolith too) plus an
unmeasured-tracker gap. Recommend monolith-primary with the decomposed stack as a warm, graftable fallback; the two
native Phase-C prototypes (min-snap/TOPP line generator + scipy-SLSQP toy MPCC tracker) are the cheap, decision-relevant
things to build before paying any acados/Adroit integration cost.
