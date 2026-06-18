# inc8 σ_p0 EVAL — RESULTS (3 warm-start checkpoints pulled + run)

**Date:** 2026-06-17 · **Model:** Sonnet 4.6 · **EFFORT:** medium · **HEAD:** `main` (no branch switch, no commit)
**Mode:** OFFLINE laptop `.venv`, estimator-emul (no live sim, no real detector). **Escape hatch: TRIGGERED.**

---

## 0. VERDICT — σ_p0 IS UN-EVALUABLE (empty ensemble); NO GO/NO-GO forced

> **All 3 warm-start seeds reach the gate-4 crossing 0 / 200 times under the estimator emulator**, so
> the GT-anchored σ_p0 distribution is **empty** — there is nothing to take a std of. Per the task's own
> instruction ("a thin/empty σ ensemble is itself the finding, NOT a number to force") and the escape
> hatch ("do NOT substitute estim_err"), **no σ_p0 GO/NO-GO is reported.**
>
> The reason is a **TRAIN/EVAL POINTING GAP**: through the emulator the policies show **zero terminal
> camera-lock (lock_g4 = 0.00) and zero accepted fixes (fixrate_g4 = 0.000) at gate-4** — so the KF gets
> no vision fixes, coasts, and the drone misses every gate. This directly contradicts the warm-start
> training traces (MEMORY: fix_rate 0.14–0.16, terminal_pointing > 0). The only in-plane numbers that
> appear (ip_p90 ≈ 0.24) are the **floored estim_err** (|KF−truth| during a no-fix coast) — explicitly
> NOT σ_p0 and not to be crowned on.
>
> **This is robust:** same reach=0 on **periodic AND best** checkpoints, all 3 seeds. The instrument and
> loader are confirmed sound (inc7 truth-obs produces a real −0.215 m crossing; the 20-dim actors load
> and fly varied per-gate trajectories — not a crash). The failure is upstream of the metric, in the
> policy's behaviour under the emulated obs.

---

## 1. Provenance — checkpoints pulled & md5-verified (Adroit serve daemon, data-only, AUP-safe)

Pulled via `.s15_ref/pull_file.py` (chunked `dd|base64`, md5-verified) from the running serve daemon
(login node `adroit5`, no compute). Source: `/scratch/network/fl3689/diffaero/outputs/train/inc8_warmstart_seed{N}_ws1/periodic/`.
Sidecar all three: `{act_max_thrust:3.765, act_max_rate:3.14, obs_dim:20, inc8:true, r5_arm:A}` ✅.
`*.pth/.json` are gitignored — NOT git-added.

| seed | local path | actor.pth md5 (== remote) |
|---|---|---|
| 0 | `rl/checkpoints/inc8_ws1_seed0_actor.pth` | `98f72253ca6830cc6d7bce6fde79b99f` ✅ |
| 1 | `rl/checkpoints/inc8_ws1_seed1_actor.pth` | `65cea7ebf5276a305cd93ccd5953f85d` ✅ |
| 2 | `rl/checkpoints/inc8_ws1_seed2_actor.pth` | `f8ed4be3575ba66a3cf24438511f6deb` ✅ |
| 0-best | `rl/checkpoints/inc8_ws1_seed0_BEST_actor.pth` | `144434159938a56517e33ea9cff976e9` ✅ (robustness check) |

(3 distinct actor md5s ⇒ genuinely different trained policies, not a copy.)

---

## 2. Per-seed σ_p0 table (`rl/inc8_sigmap0_eval.py --estim-emul --n-episodes 200`)

| seed | reach_rate (g4 crossings) | σ_p0_lat | bias_lat | lat p90 | lat p99 | term_lock | g4_fix_rate | provisional flag |
|---|---|---|---|---|---|---|---|---|
| 0 | **0 / 200** | — (no data) | — | — | — | nan | nan | NO-DATA |
| 1 | **0 / 200** | — (no data) | — | — | — | nan | nan | NO-DATA |
| 2 | **0 / 200** | — (no data) | — | — | — | nan | nan | NO-DATA |
| 0-best | **0 / 50** | — (no data) | — | — | — | nan | nan | NO-DATA |

`reach_rate FIRST`, as instructed: **0 for every seed and both checkpoint variants.** No σ_p0 std, bias,
or percentile exists because no episode produced a gate-4 plane crossing. The empty ensemble is the result.

---

## 3. Diagnostic — WHERE the episodes fail (`contact_true_eval.py --estim-emul`)

Per-gate outcome under the emulated obs (simstart = full course; trainreset_gN = spawned 1 m behind gate N):

| seed | simstart | tr_g3 | tr_g4 (the binding gate) | tr_g4 emul: v* / fixrate_g4 / lock_g4 / ip_p90 |
|---|---|---|---|---|
| 0 | MISS | MISS (g3_linf 0.320) | MISS | 16.8 / **0.000** / **0.00** / 0.242 |
| 1 | MISS | COLLISION | MISS | 16.7 / **0.000** / **0.00** / 0.242 |
| 2 | MISS | COLLISION | MISS | 17.0 / **0.000** / **0.00** / 0.249 |

S_stable = 0.000 (0/7) for all three. Key reading:
- **lock_g4 = 0.00 everywhere** — the gate never projects in-frame over the terminal 5 m, i.e. the policy
  does NOT point at gate-4 *through the emulator*. (This is the load-bearing inc8 lever, measured zero.)
- **fixrate_g4 = 0.000** — with no pointing, the surrogate offers no fixes → the KF coasts → ip_p90 rises
  to ~0.24 (the no-fix coast floor; **this is estim_err, NOT σ_p0**).
- The policies DO fly (they progress several gates, varied per-seed outcomes) — so this is not a NaN/crash;
  it is a behavioural pointing gap under the emulated obs.

---

## 4. Interpretation & what's needed next (commander's call)

The contradiction is sharp: training reported terminal pointing + fixes; the **eval emulator** reports
**zero** of both. The most likely root cause is a **train↔emul obs gap** — the 20-dim policy is OOD on the
`estimator_emul` obs (candidate seams: the confidence channel [17:20] encoding, or the KF-pose seam) — OR
the policy does not generalise from the training reset distribution to the simstart/trainreset eval starts.
Either way **the emul-based σ_p0 GO gate cannot be evaluated until the pointing gap is closed.**

Recommended follow-ups (NOT done here — out of this task's scope, and the directive forbids forcing a number):
1. **Parity-check the emulator obs vs the training env** for these exact checkpoints (does the train env
   reproduce lock>0 / fixes where the numpy emul shows 0?). This decides "instrument gap" vs "policy gap".
2. If it is a policy gap, the warm-start "3/3 clean" precondition (value_loss didn't cliff) did NOT
   translate into a course-completing case-C pointer — escalate per the inc8 ladder.

🚩 **PROVISIONAL throughout** — even a future positive σ_p0 from this emul path stays provisional until the
real-detector→PnP→KF spike confirms emul fidelity (#37). **inc8 is NOT declared done.**

---

## 5. MEMORY-DELTA (text only — do NOT commit memory/; commander banks)

```
inc8 σ_p0 EVAL RAN on the 3 warm-start ckpts (2026-06-17, laptop main, Sonnet) → UN-EVALUABLE, empty ensemble.
- PULLED + md5-verified (Adroit serve daemon, data-only): inc8_ws1_seed{0,1,2}/periodic actor.pth+json →
  rl/checkpoints/inc8_ws1_seed{0,1,2}_actor.pth (md5 98f7.. / 65ce.. / f8ed..; 3 distinct = real per-seed
  policies; sidecar obs_dim20 inc8 r5_armA). gitignored, NOT added.
- RESULT: reach_rate = 0/200 ALL 3 seeds (and 0/50 on seed0 BEST) → NO gate-4 crossing → σ_p0 ensemble EMPTY.
  NO GO/NO-GO forced (per directive). Robust to periodic-vs-best.
- ROOT: TRAIN/EVAL POINTING GAP. Through estimator_emul: lock_g4=0.00 + fixrate_g4=0.000 at gate-4 (zero
  terminal camera-lock, zero fixes) → KF coasts → ip_p90≈0.24 = the FLOORED estim_err (NOT σ_p0; not crowned).
  Contradicts warm-start training traces (fix_rate 0.14-0.16, pointing>0). Policies DO fly (multi-gate, varied
  outcomes) = not a crash; instrument+loader sound (inc7 truth-obs = real -0.215m crossing).
- NEXT (commander): parity-check emul obs vs train env for these ckpts (confidence-channel/KF-pose seam) to
  decide instrument-gap vs policy-gap; until the pointing gap closes the emul σ_p0 GO gate is un-measurable.
  Everything PROVISIONAL vs #37. inc8 NOT done.
```
