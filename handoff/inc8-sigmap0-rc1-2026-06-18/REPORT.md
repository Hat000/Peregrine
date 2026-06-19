# inc8 σ_p0 GO-GATE on the FLYING recenter seeds — REPORT (2026-06-18)

**Session:** inc8 σ_p0 GO-GATE — terminal centering on the recenter (rc1) seeds.
**Instrument:** `rl/inc8_sigmap0_eval.py --estim-emul --n-episodes 200` (GT-anchored gate-4 crossing y/z).
**Checkpoints:** `best/actor.pth` of `inc8_recenter_seed{0,1,2}_rc1` (job 3276449), pulled + md5-verified.
**HEAD:** main @ c902dd1. Nothing committed (`*.pth` gitignored). Commander makes the GO call.

---

## VERDICT (raw — for the commander)

🚩 **σ_p0 is UN-MEASURABLE on the rc1 seeds via this instrument — gate-4 reach is ≤1/200 on ALL 3 seeds,
and `finished`=0/200 (zero full laps from `simstart`).** This is the escape-hatch trigger: I report reach
and STOP; I did NOT fabricate a σ from one crossing.

**But the cause is NOT "thin headroom near gate-4."** A per-gate reach diagnostic (seed 0, 60 eps) shows the
policy passes **ZERO gates in 56/60 episodes** from `simstart` (mean max-gate = −0.87); the lone gate-4 hit
is a fluke. **From `simstart` in the contact_true_eval / sigmap0 harness, the recenter policy does not fly
the course at all** — it dies at/before gate-0.

**This directly contradicts the TRAINING result** (same checkpoints): success_rate ~0.50–0.54, n_passed_gates
~2.96–3.03 across the 2048 randomized-start envs. **So the recenter fix recovered flight in the TRAINING
env, but the σ_p0 eval harness shows non-flight from its fixed `simstart`.** That is a TRAIN/EVAL
discrepancy, and it gates σ_p0 — not a number to force.

---

## Per-seed table

| seed | ckpt md5 | reach@gate-4 | finished | σ_p0_lat | bias_lat | lat p90/p99 | term_lock | g4_fix |
|------|----------|--------------|----------|----------|----------|-------------|-----------|--------|
| 0 | a71dda71… | **1/200** (0.01) | 0/200 | n/a (1 sample) | −0.374 m¹ | 0.374/0.374¹ | 0.000 | 0.000 |
| 1 | cc720d76… | **0/200** (0.00) | 0/200 | NO-DATA | — | — | — | — |
| 2 | eb3ac175… | **0/200** (0.00) | 0/200 | NO-DATA | — | — | — | — |

¹ Single crossing → std is 0 (meaningless); the −0.374 m is one off-centre crossing, NOT an ensemble σ/bias.
All 3 seeds: obs_dim 20, inc8 true, r5_arm A (sidecars identical, md5 a2c7eaa…). estim_emul ON, plant=mixer.

**Reach distribution (seed 0, 60 eps, simstart, emul ON):**
`max_gate==-1: 56` · `==0: 3` · `==4: 1` · mean_max_gate = −0.87. → dies at gate-0 in 93% of episodes.

---

## WHY (most-supported interpretation — for the commander's call)

The policy demonstrably FLIES in training but not from this harness's `simstart`. The most-supported cause is
**EVAL-HARNESS FAITHFULNESS, already a documented carry-forward**, not a fresh policy regression:

1. **Eval omits the look-at primitive training composes** (MEMORY carry-forward: `peregrine_racing_inc8.py`
   look-at vs `contact_true_eval.py` — "port for a faithful σ_p0"). If the eval feeds the policy an obs
   stream missing the look-at composition the training env applied, the policy is off-distribution from
   step 0 → fails immediately. This is the leading hypothesis.
2. **Fixed `simstart` ≠ the training start distribution** (training used `standing_start_frac=0.3` +
   randomized starts across 2048 envs). A single atypical fixed start the policy never specialized to could
   fail at gate-0.
3. **Harness dynamics:** eval defaults `plant=mixer`; training ran the full DR set (dr_aero/dr_mixer/
   dr_force_bias/latency). A plant mismatch puts the policy off-distribution.

**Disambiguation done:** the truth-obs path is NOT usable for inc8 (it builds a 17-dim obs; the inc8 actor
needs the 20-dim obs whose confidence-triple [17:20] only the estimator EMUL produces → `mat1 1x17 vs 20x256`).
So `--estim-emul` is the ONLY valid inc8 obs path, and on it the rc1 seeds don't fly from simstart. The
non-flight is therefore inside the emul-eval harness specifically — consistent with hypothesis (1).

**Caveat on checkpoint choice:** used `best/` (best-metric save). Consistency across 3 independent seeds +
the reach diagnostic makes a best-vs-periodic selection fluke unlikely, but `periodic/`/`checkpoints/` could
be cross-checked if desired.

---

## GO READ

**NO-GO / NO-DATA at the σ_p0 gate — but the blocker is the EVAL INSTRUMENT, not (on this evidence) the
policy.** The recenter seeds fly in training (confirmed, 3/3); the sigmap0 harness can't yet measure their
terminal centering because the policy won't fly the course from `simstart` under the (un-faithful) emul-eval.

**A RW_TC sweep alone will NOT unblock this** (it tunes centering on a course the eval can't get the policy
to fly). **Recommended next (commander's call):**
1. **Fix eval faithfulness FIRST** — port the look-at primitive into `contact_true_eval.py` (the documented
   gap) and/or align the eval start to training; re-run σ_p0. This is the load-bearing fix.
2. Then, if reach is restored, measure σ_p0 vs 0.08 and run the RW_TC sweep for centering headroom.
3. INTERPRETATION FRAME (unchanged): even a future GO here = "policy centers to σ_p0=X under the EMUL's fix
   quality" — the real-8-kpt-detector per-fix accuracy (best.pt not staged) is the final confirm.

---

## OPS

- Pull: 6 files (actor.pth + actor.json × 3 seeds) from `outputs/train/inc8_recenter_seed{0,1,2}_rc1/best/`
  via serve daemon, all md5-verified. Local: `rl/checkpoints/inc8_rc1_seed{0,1,2}_actor.{pth,json}`
  (gitignored, NOT added). Distinct actor md5s = genuinely different seeds.
- Daemon healthy (no #76 recovery needed this session). No compute on Adroit (data move + path stat only).

---

## MEMORY-DELTA (text only — do NOT commit memory/)

```
inc8 σ_p0 GO-GATE on rc1 flying seeds (2026-06-18, sigmap0_eval --estim-emul --n-ep 200, best/ ckpts):
- 🚩 σ_p0 UN-MEASURABLE on ALL 3 rc1 seeds: gate-4 reach 1/200 (s0), 0/200 (s1,s2); finished 0/200 all.
  NOT borderline-gate-4 — reach-diag (s0, 60ep) shows 56/60 pass ZERO gates from simstart (mean_max_gate
  -0.87, dies at gate-0). Single s0 crossing bias_lat -0.374 m = NOT an ensemble σ; reported, not crowned.
- CONTRADICTS training (same ckpts FLY: success_rate ~0.50-0.54, n_passed ~3.0 across 2048 randomized
  starts). => TRAIN/EVAL DISCREPANCY: recenter recovered flight in the TRAINING env but the contact_true_eval/
  sigmap0 harness shows non-flight from its fixed simstart under emul obs.
- LEADING CAUSE = EVAL-HARNESS FAITHFULNESS (documented carry-forward): eval omits the look-at primitive
  training composes (contact_true_eval vs peregrine_racing_inc8) + fixed simstart != training start dist +
  plant=mixer vs training full-DR. Truth-obs path UNUSABLE for inc8 (builds 17-dim; actor needs 20-dim emul
  confidence-triple -> shape err) => estim_emul is the only inc8 obs path; non-flight is inside that harness.
- GO READ: NO-GO/NO-DATA at σ_p0, blocker = the INSTRUMENT not (on this evidence) the policy. RW_TC sweep
  will NOT unblock. NEXT = port look-at into contact_true_eval (faithful σ_p0) +/- align eval start, THEN
  re-measure σ_p0 vs 0.08 + sweep. ckpts staged rl/checkpoints/inc8_rc1_seed{0,1,2}_actor.* (md5-verified).
```
