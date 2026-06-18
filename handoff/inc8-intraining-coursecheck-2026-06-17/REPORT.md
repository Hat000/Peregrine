# inc8 IN-TRAINING course-completion check — POLICY-GAP vs EVAL-FAITHFULNESS

**Date:** 2026-06-17 · **Model:** Sonnet 4.6 · **EFFORT:** low · **WHERE:** Adroit (login node `adroit5`, READ-only via serve daemon) → parsed on laptop `.venv`
**HEAD:** `main` @ `a6cda4e` (no branch switch, no commit, no git-add)
**AUP:** ✅ LOG/DATA READ ONLY — `stat`/`md5sum`/`base64`/`find`/`grep` on the login node via the adroit-connector daemon. NO SLURM, NO compute on Adroit. TB protobuf parsed locally on the laptop.

---

## 0. VERDICT — PURE POLICY-GAP (course-flight). The eval is faithful; the policy never flew, even in training.

> **`metrics/success_rate` = 0.0000 for EVERY logged point (400/400, steps 0→3990) in ALL FOUR runs** — the
> 3 warm-start seeds (`inc8_warmstart_seed{0,1,2}_ws1`) **and** the S2-seed2 parent (`s2full/seed2 yawprim`).
> It is flat-zero from the first step to the last; it **never rose above 0**.
>
> **This is the confound-free test.** These are the *in-training* env metrics — the policy acting through the
> real look-at primitive in the real training env (`peregrine_racing_inc8.py:443`). There is no eval-emulator,
> no re-injection, no OOD obs seam here. The policy **never completed a course/lap even while training.**
>
> ⇒ The laptop σ_p0 eval finding (reach 0/200, misses gate-0) was **NOT** an eval-faithfulness artifact and
> **NOT** caused by the missing look-at re-inject. It faithfully reproduced what the policy actually does.
> **Branch taken (per task decision tree): "~0 throughout ⇒ NEVER completed laps even in training = PURE
> POLICY-GAP → fix is a reward + GO-criteria change + re-train"** — NOT porting look-at into the eval.

---

## 1. The decisive numbers — `success_rate` & `n_passed_gates` (TB `metrics/*`, full run)

| run | `success_rate` min/max/last | `n_passed_gates` min/max/last | ever success>0? |
|---|---|---|---|
| ws_seed0 | **0.0000 / 0.0000 / 0.0000** | 0.0021 / 0.1338 / 0.1113 | **NO** |
| ws_seed1 | **0.0000 / 0.0000 / 0.0000** | 0.0059 / 0.1382 / 0.1113 | **NO** |
| ws_seed2 | **0.0000 / 0.0000 / 0.0000** | 0.0000 / 0.1338 / 0.1113 | **NO** |
| s2full/seed2 (parent) | **0.0000 / 0.0000 / 0.0000** | 0.0000 / 0.1333 / 0.1230 | **NO** |

- **success_rate trajectory** (sampled steps 360→3990, identical shape all 4 runs): `0,0,0,0,0,0,0,0,0,0,0,0` — dead flat.
- **n_passed_gates** never climbs: it sits at **~0.10–0.14 gates/episode the entire run** (a course is multiple gates; this is well under one full gate of average progress — i.e. it dies at/near gate-0). No upward trend; the max (~0.138) is noise, not a learning curve.

Each `*.tfevents` is md5-verified == its Adroit source:
`ws_seed0 3be86fe4…`, `ws_seed1 a93b11a3…`, `ws_seed2 de20faf6…`, `s2full_seed2 505526ec…`. (Raw files kept in `tb/`; parser `parse_tb.py`.)

---

## 2. HOW the episodes fail — corroborating in-training metrics (full-run min/max/last)

| metric | ws_seed0 | ws_seed1 | ws_seed2 | s2full/seed2 | reading |
|---|---|---|---|---|---|
| `metrics/miss_rate` | .95/.99/.99 | .95/.99/.98 | .89/.99/.98 | .20/.99/.97 | **~98% of episodes end in a gate MISS** |
| `metrics/collision_rate` | ≤.04 | ≤.05 | ≤.04 | (≤.04 steady) | collisions are minor — misses dominate |
| `metrics/survive_rate` | **0/0/0** | **0/0/0** | **0/0/0** | **0/0/0** | episodes NEVER survive to timeout — they always miss/crash |
| `metrics/l_episode` | ~1.5 | ~1.5 | ~1.5 | ~1.5 | episodes die early (lap is ~8 s) — death around gate-0 |

**Pointing DID happen** (the inc8 lever worked — this is the key cross-check that the look-at primitive was live in training):
| metric | ws_seed0 | ws_seed1 | ws_seed2 | s2full/seed2 |
|---|---|---|---|---|
| `metrics/pointing_rate` (max) | 0.46 | 0.46 | 0.46 | 0.46 |
| `env_loss/inc8_terminal_pointing` (max) | 0.12 | 0.11 | 0.13 | 0.15 |
| `env_loss/inc8_fix_rate` (max) | 0.23 | 0.23 | 0.22 | 0.17 |

So the policy **learned to point the camera and accept fixes** (matches MEMORY's "fix_rate 0.14–0.16, terminal_pointing>0") — **but it does so on a policy that misses ~98% of gates and never finishes.** The ladder optimized POINTING on a NON-FLYING policy. The S2 *parent* already has success_rate≡0, so the gap is inherited across the whole inc8 lineage, not introduced by warm-start.

---

## 3. What this settles / next

- ✅ **Settled:** reach=0 is a **policy gap**, not an eval-faithfulness gap. Confirms & strengthens the committed diagnosis (a6cda4e). Porting the look-at into `contact_true_eval` is still worth doing for a *faithful σ_p0* once a policy actually flies, but it will **not** rescue these checkpoints — there is no lap to measure.
- **The fix is upstream:** reward architecture (restore through-approach centring / R1-to-centre that inc8 dropped → under-centring → drift → gate-0 miss) **+ add course-completion (success_rate / n_passed_gates) to the inc8 GO criterion** so "pointing wins" can never again mask a non-flying policy **+ re-train** (VQ2-gated per the ladder).
- The warm-start training-stability win (value_loss cliff dissolved, appo dodged) **still stands** — it is orthogonal to course-flight. σ_p0 stays NO-GO until a policy completes a lap.

---

## 4. MEMORY-DELTA (text only — commander banks; do NOT commit memory/)

```
inc8 IN-TRAINING course-completion check (2026-06-17, Adroit login-read via serve daemon, parsed on laptop) =
CONFOUND-FREE CONFIRMATION of POLICY-GAP. Read TB metrics/success_rate + metrics/n_passed_gates for the 3
warm-start seeds (inc8_warmstart_seed{0,1,2}_ws1) AND the S2-seed2 parent (s2full/seed2 yawprim); md5-verified.
- success_rate = 0.0000 FLAT across ALL 400 logged steps (0->3990), ALL 4 runs. NEVER >0. n_passed_gates floored
  ~0.10-0.14 gates/ep the whole run (dies at/near gate-0). survive_rate=0 (always miss/crash, never timeout);
  miss_rate ~0.98; l_episode ~1.5s (lap ~8s). => the policy NEVER completed a lap EVEN IN TRAINING, with the real
  look-at primitive + real train env. So laptop reach=0/200 is FAITHFUL, NOT a look-at-reinject/eval artifact.
- Pointing DID work in training (pointing_rate max 0.46, terminal_pointing max 0.12-0.15, inc8_fix_rate max 0.17-0.23)
  => ladder optimized POINTING on a NON-FLYING policy; gap is INHERITED from the S2 parent (parent success_rate=0 too).
- VERDICT = PURE POLICY-GAP. FIX = reward arch (restore through-approach/R1-to-centre centring) + ADD course-completion
  (success_rate/n_passed_gates) to inc8 GO + re-train (VQ2-gated). Eval look-at port is NOT the cause (do for faithful
  sigma_p0 LATER, once a policy flies). Warm-start stability win (cliff gone, appo dodged) STANDS; sigma_p0 NO-GO until a lap completes.
```
