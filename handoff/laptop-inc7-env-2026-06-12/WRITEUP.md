# LAPTOP-INC7-ENV — contact-true geometry + dr_force_bias implemented; launch packaged (2026-06-12, fable)

**Session:** LAPTOP-INC7-ENV. **Spec:** `docs/training_doctrine.md` + the training-doctrine
WRITEUP Part 3 (inc7: c16 reward BYTE-IDENTICAL, S17 mixer plant, all inc6 DR, plus exactly
three additions). **Commits:** `134cc41` (env+dynamics+gate+tests), this handoff dir.

---

## TL;DR

All three doctrine additions are implemented, unit-tested (616/620 green — +19 new tests; the 4
failures are a parallel session's uncommitted WIP, see §2), and
negative-controlled (defaults OFF = bit-identical legacy; local parity-gate dry run GATE_PASS
with the DR codepath at nominals diverging by exactly **0.0**). The inc6 geometry-honest re-run
validates the env redesign — and sharpens it: **gate 5, not only gate 3, is already illegal
under honest contact** (nominal inc6 trajectories strike g5 at r ≥ 0.33, standing AND bridge),
and the gate-3 funnel's true margin is **0.06–0.16 m** where legacy scoring claimed ~0.44 m.
Adroit launch is BLOCKED on Duo (serve daemon down; passphrase-encrypted key, no agent — verified
non-interactive SSH impossible), so per the escape hatch the full launch is packaged as ONE
command (`scripts/launch_inc7.py all`) that enforces all three launch gates in order and refuses
to submit on any failure. S20 deliberately skipped (rationale below; not a launch blocker).

---

## 1. What was implemented (commit `134cc41`)

### ① Body-radius inflation — `rl/peregrine_racing.py`
- `+env.body_radius_lo` / `+env.body_radius_hi`: per-env r ~ U[lo, hi], resampled at every
  reset (`_sample_body_radius`). Pass band L-inf < 0.75 − r; frame band [0.75 − r, 1.36 + r]
  (`_contact_bands`, threaded through `crossing_events` as (N, 1) tensors — the helper already
  broadcasts). The radius is deliberately NOT observable (doctrine: unobservable halo ⇒ policy
  carries the sampled worst case).
- sbatch values: U[0.28, 0.38] — brackets the measured live contacts (crashes at L-inf
  0.37–0.49; corner-pass contact at 0.60).

### ② 0.30 m frame extrusion — `slab_frame_hits` (new pure helper)
- The frame band becomes MATERIAL over gate-frame |x| ≤ depth. **EXACT** segment-vs-volumetric-
  frame classification, not an approximation: on the in-slab parameter interval, Linf(t) is
  convex piecewise-linear, so its range is [min over ≤ 6 closed-form candidate points,
  max(endpoints)]; the segment collides iff that range intersects the band. ~30 elementwise ops
  on the (N, G) batch per step — no measurable step-rate hit, so the >15% escape hatch (cheap
  approximation) was NOT needed.
- Event semantics: a slab strike on ANY gate is a collision (T1); a plane crossing that threads
  the (inflated) pass band but touches material inside the slab is a COLLISION, not a pass
  (gate_passed/gate_miss masked); collision dominates miss on the target gate.
- New diagnostics: `slab_collision_rate` + `pass_margin_m` ((0.75 − r) − Linf at crossings —
  the gauntlet tail metric; doctrine §5.3 scores tails against contact-true aperture).

### ③ Structured force-bias DR — `rl/diffaero_dynamics.py`
- `+dynamics.dr_force_bias=true` (+ `dr_force_bias_max`, default 3.0 = the certified residual
  bound): per-env world-frame (NED) bias, uniform random direction × magnitude U[0, max],
  active only while the env's (speed, tilt) sits in a per-env random regime bin. **Bins mirror
  `scripts/frame_residual_report.py`'s certified bins** (speed (0,4)(4,8)(8,12)(12,18)(18,40) ×
  tilt (0,15)(15,35)(35,90)°), so the DR class IS the measured-error class — the climb-bin
  instance (12–18 m/s × 35–90°) is one of the 15 sampled bins.
- Applied per substep on accel, gated on the OLD state (the drag/lapse convention); cos-tilt
  reuses the thrust-direction rotation (zero extra quat math). Init = zero bias + empty bin
  (inert until first reset); python-guarded so the flag-off path is bit-identical.

### Reward: UNTOUCHED. `compute_reward_terms` and `RewardWeights` have zero diff — c16 is
byte-identical (confirmed by `git diff 134cc41^ 134cc41 -- rl/peregrine_racing.py`: no hunk
touches the reward section).

### Twin/eval side
- `rl/offline_rollout.py`: `--body-radius` / `--frame-depth` flags; `gate_event` +
  `slab_frame_hit_np` (numpy mirror, cross-validated vs torch).
- `rl/peregrine_eval.py`: `--body-radius-lo/hi` / `--frame-depth` cfg overrides (score legacy
  ckpts contact-true; inc7 ckpts inherit their keys from the run cfg automatically) + reports
  PASS_MARGIN_M and slab strike counts.
- `rl/peregrine_racing_inc7.sbatch`: pinned to A100 (`--gres=gpu:nvidia_a100:1`) per the
  2026-06-12 GPU directive. Everything else untouched.

## 2. Verification

- **Tests: 620 collected, 616 passed** (+19 in `tests/test_contact_geometry.py`; every suite
  touching this session's code passes 216/216 in isolation):
  brute-force arbiter (1000 random segments, 20001-sample reference, exact agreement),
  numpy↔torch mirror agreement, per-env tensor-band broadcast vs scalar calls, the live
  crash offsets (0.37/0.44/0.49) reclassified pass→strike at the doctrine radii, force-bias
  band/bin/edge semantics, and **negative controls** (verbatim legacy `crossing_events` copy
  pinned bit-equal with float bands; numpy `gate_event` defaults = legacy classes).
  ⚠️ The full-suite run shows 4 additional failures in `test_twin/test_twin_tune` — caused by
  the parallel VISION-FRAME-FIX work, which COMMITTED mid-session as `8d7b0b3` (verified both
  directions by selective stashing while it was still WIP). Root cause diagnosed: the
  navigator now conjugates `ds.orientation_ned_wxyz` (correct on the live wire = RAW quat),
  but the twin harness synthesizes TRUE quats into that field — the conjugation mis-rotates
  the twin's vision fixes and `fly()` ends RUN instead of FINISHED. The twin harness violates
  the wire contract, not the navigator fix. Not fixed here (frame-convention changes need
  that session's context — doctrine §6); **flagged as a spin-off task chip** for Fengyou.
- **Local parity-gate dry run** (`rl/local_gate_harness.py`, stubbed BaseDynamics):
  GATE_PASS; all 10 legacy configs unchanged (worst 7.1e-15, identical to the S17 V100 value);
  new `dr_nominal` config (FULL DR codepath — dr+dr_aero+dr_mixer+dr_force_bias — pinned at
  float64 nominals) **DIV_FLOAT64 = 0.000e+00**; FORCE_BIAS behavioral check (in-bin dv ==
  bias·dt, out-of-bin dv == 0) err 8.7e-18. The authoritative GPU gate re-runs on Adroit
  (launch step below).

## 3. inc6 geometry-honest re-run (the env's own validation)

`scripts/inc6_geom_eval.py` (+ `inc6_geom_eval_output.txt`): inc6 trajectories generated under
LEGACY rules (the twin has no contact, exactly like training), scored post-hoc contact-true
(r ∈ {0.28, 0.33, 0.38}, depth 0.30):

| condition | legacy verdict | contact-true verdict |
|---|---|---|
| standing nominal | FINISHED, g3 0.26 / g5 0.38 | g5 **COLLISION** at r ≥ 0.33; worst margin +0.09 at r=0.28 |
| standing + climb residual | FINISHED | g5 **COLLISION** at r ≥ 0.33 |
| bridge nominal | FINISHED, g5 0.37 | g5 **COLLISION** at r ≥ 0.33 |
| bridge + residual | FINISHED | g5 **COLLISION** at r = 0.38 |

- **Gate 3:** corridor Linf at slab entry (x=−0.30) = 0.31 vs pass band 0.37–0.47 ⇒ true margin
  0.06–0.16 m (legacy fiction: ~0.44). The live +0.15–0.25 m residual arrival (which the
  closed-loop twin absorbs — Q4 — but the live plant gap produces) lands at 0.46–0.56 ⇒ strike.
  The geometry-honest eval reproduces the live verdict on the same funnel legacy called safe. ✔
- **Gate 5 is the bigger standing offender on the twin's own line** (Q3 already flagged its
  0.38 tail) — even nominal inc6 is illegal under honest contact at mid-band r. inc7's
  geometry pressure is therefore not just a gate-3 fix; expect the whole funnel to flatten.
- Live crash offsets (0.37–0.49) imply the effective halo sits in the UPPER half of the r band —
  worst-case-bracketing U[0.28, 0.38] is the right pressure.

## 4. Launch status — ✅ LAUNCHED (2026-06-12, post-handoff update)

Fengyou brought up the serve daemon; one stale-file fix was needed (scratch `rl_plant.py` was
pre-S18 — its missing LAPSE_* constants killed the guarded import; added to the sync list,
commit on this branch), then all three gates passed live and the seeds went in:

- **GATE 1 precheck:** PASS (login node; env builds with all inc7 keys, `pass_margin_m` +
  `slab_collision_rate` present, all DR axes sample, GuardedPPO wired).
- **GATE 2 parity gate:** **GATE_PASS on V100** (job **3270602**): 11 configs × 6 seeds ×
  8 steps, worst DIV_FLOAT64 7.105e-15; `dr_nominal` 3.997e-15; FORCE_BIAS check 8.674e-18.
- **GATE 3 tests:** 616/620 green (laptop, §2).
- **INC7_JOB_IDS: s0 = 3270605, s1 = 3270606, s2 = 3270608** (A100-pinned, 6000 updates,
  2048 envs; outputs `/scratch/network/fl3689/inc7_run_<job>.out`; eval epilogue runs VQ1 +
  random courses mixer-ON automatically). Poll: `launch_inc7.py status`.

### Original handoff instructions (superseded by the launch above; kept for the record)

The adroit-connector serve daemon is down; the key is passphrase-encrypted (blank in .env by
design), no ssh-agent, and BatchMode SSH fails on Duo keyboard-interactive — there is no
non-interactive path, so I could not run GATE 1/2 on-cluster or submit. Everything is packaged:

```text
T1:  cd C:\Users\Fengy\Downloads\Projects\Adroit\adroit-connector
     .venv\Scripts\python adroit.py serve            # passphrase + ONE Duo push
T2:  cd C:\Users\Fengy\Downloads\Projects\Anduril
     .venv\Scripts\python handoff\laptop-inc7-env-2026-06-12\scripts\launch_inc7.py all
```

`launch_inc7.py all` = sync (chunked-b64, 4 chunks, verifies the new features landed) →
GATE 1 precheck (login CPU, real inc7 hydra cfg incl. all new keys) → GATE 2 parity gate
(GPU sbatch, polls for GATE_PASS; **exits without submitting on GATE_FAIL**) → seeds 0/1/2 of
`peregrine_racing_inc7.sbatch` → prints `INC7_JOB_IDS s0:<id> s1:<id> s2:<id>` (record them
here). Steps are idempotent (`sync|precheck|gate|launch|status`) — re-run after a daemon drop.
**Job IDs: PENDING Fengyou's run** — append them to this file.

Selection after training: ≥3-seed gen AVERAGE + full doctrine §5 gauntlet (offset tails +
corridor now come straight out of `pass_margin_m`/`slab_collision_rate` and the eval's
PASS_MARGIN_M line) before any live flight.

## 5. S20 — deliberately skipped (escape-hatch, with reasons)

Not a launch blocker by the doctrine's own measurement (Q4: the policy absorbs the full
climb-bin residual closed-loop; the geometry fix supplies the margin; dr_force_bias puts the
error CLASS in-distribution). Folding refit constants in tonight would invalidate the
just-passed parity gate (plant tables change ⇒ gate re-run ⇒ a second Duo cycle) and confound
the increment (one change family per increment — geometry+DR is this increment's family).
Recommend: S20 as its own session against `handoff/shadowpc-postfix-dataset-2026-06-12/`
(6 valid runs + 17 refit runs), parity-gated, folded into inc8 or an inc7 re-eval.

## 6. Prediction on record (unchanged from doctrine Part 3, banked)

inc7 standing start clears gate 3 with ≥ 0.3 m corridor margin; crossing tails ≤ 0.25 m
(contact-true); posture unchanged (~55° tilt crab); lap cost vs inc6 ≤ 0.3 s; bridge parity.
**Added from §3:** inc7's gate-5 crossing tail must come inside the contact-true band
(≤ 0.37 − ε at the worst-case radius) — the sharpest single falsifiable number this eval produced.

## 7. Files

- `rl/peregrine_racing.py` — body-radius bands + `slab_frame_hits` + stats (reward untouched).
- `rl/diffaero_dynamics.py` — `sample_force_bias`/`force_bias_active` + DR wiring.
- `rl/check_diffaero_gate.py` — `dr_nominal` config + `force_bias_behavior` check.
- `rl/offline_rollout.py`, `rl/peregrine_eval.py` — contact-true scoring knobs.
- `tests/test_contact_geometry.py` — 19 tests.
- `scripts/inc6_geom_eval.py` + `inc6_geom_eval_output.txt` — the §3 evidence.
- `scripts/launch_inc7.py` — the gated one-command launch driver.

---

MEMORY-DELTA:
- **INC7 ENV SHIPPED (134cc41, tests 616/620 green; +19 new):** `+env.body_radius_lo/hi` (per-env U at reset;
  pass <0.75−r, band [0.75−r,1.36+r]) + `+env.frame_depth_m` (EXACT segment-vs-slab, no approx
  needed — escape hatch unused) + `+dynamics.dr_force_bias` (world bias ≤3 m/s², bins mirror
  frame_residual_report). Defaults OFF bit-identical; c16 reward byte-identical.
- **Parity gate EXTENDED + locally GATE_PASS:** new `dr_nominal` config (full DR path at
  nominals) DIV 0.0; force-bias behavioral err 8.7e-18. GPU gate re-run still REQUIRED pre-submit.
- **inc6 geometry-honest eval (supersedes "gate-3 only" framing): gate 5 ALSO illegal** —
  nominal inc6 (standing AND bridge) strikes g5 contact-true at r≥0.33; g3 true margin
  0.06–0.16 m vs legacy ~0.44. New falsifiable inc7 number: g5 tail ≤0.37−ε at worst-case r.
- **inc7 sbatch now A100-pinned** (`--gres=gpu:nvidia_a100:1`).
- **✅ INC7 LAUNCHED: jobs s0=3270605 s1=3270606 s2=3270608** (A100); V100 parity gate PASSED
  live (job 3270602, worst 7.1e-15, dr_nominal 4.0e-15) — "gate not run since S16" flag CLEARED.
  Scratch-repo footgun instance: src/racer/rl_plant.py was STALE pre-S18 → GATE 1 import death;
  launch_inc7.py now syncs it.
- **S20 SKIPPED deliberately** (not a blocker per Q4; avoids gate re-run + increment confound);
  queue as own session.
- **⚠️ vision-frame-fix (8d7b0b3, COMMITTED) breaks 4 twin/twin_tune tests on main:** the
  navigator now conjugates orientation_ned_wxyz (correct for live RAW quat); the twin harness
  feeds TRUE quats → twin fly() no longer FINISHES. Twin harness violates the wire contract;
  navigator fix itself validated. Spin-off task flagged; fix the harness, not the navigator.
