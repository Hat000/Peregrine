# inc8 OBS-PARITY DIAGNOSIS — REPORT

**Date:** 2026-06-17  **Session:** inc8 OBS-PARITY DIAGNOSIS  **Model:** Opus 4.8 · EFFORT high
**Where:** laptop MAIN checkout (HEAD==main 840818f, verified) · .venv (torch 2.12.0+cpu) · offline
**Artifacts:** `diag_parity.py` (this dir). Code paths read-only; NO eval/train code edited.

---

## 0. VERDICT (one line)

**NOT an obs-construction instrument-gap. The three hypothesized obs/wiring divergences DO NOT EXIST
(obs[0:17] is bit-parity 7.6e-6; obs[17:20] is irrelevant; fix-acceptance wiring works). The reach=0 is a
POLICY-GAP of a kind the prompt did not anticipate: the inc8 policy POINTS (real, retained) but CANNOT FLY
THE COURSE — it misses gate-0 even on PERFECT pose, while inc7 finishes 5/5 in the identical harness. The
σ_p0 GO is blocked on the POLICY (course-completion), not on the eval instrument.**

Localized divergences: see §2 (the real one) and §3 (a real-but-secondary control-law incompleteness).

---

## 1. Method — drove BOTH obs paths + bisected the failure

The prompt asked to diff the TRAINING obs path (`PeregrineRacingInc8` + torch emul) against the EVAL obs
path (`inc8_sigmap0_eval.py` → `contact_true_eval.run_episode` + numpy emul) from the same state. diffaero
is **not importable on this laptop** (`import diffaero` fails) so the training *env* cannot be run here —
but its obs *builder* (`inc8_estimator_emul.obs_zup_torch`) and confidence channel are pure torch and WERE
driven directly. I built `diag_parity.py`, a faithful copy of `run_episode`'s loop (validated below), to
bisect the closed loop.

**Harness validation (critical):** inc7 (`stage1_inc7_actor.pth`) through `diag_parity.py` with truth obs
finishes **3/3, all gates 0–5**; through the REAL `contact_true_eval.py` it is `simstart=FINISHED,
S_STABLE=1.000 (7/7)`. The diagnostic harness reproduces the real eval. So any inc8 failure in it is the
policy, not the harness.

---

## 2. THE ROOT CAUSE — the policy cannot fly the course, even on perfect pose

Using the S2-converged lineage ckpt `rl/checkpoints/inc8_ws1_seed2_actor.pth` (obs_dim 20):

| run (10 ep, simstart) | reached g4 | max gate passed | outcome |
|---|---|---|---|
| **TRUTH obs**, no look-at | **0/10** | −1 (none) | MISS_g0 ×10 |
| EMUL obs, no look-at (== current σ_p0 eval) | 0/10 | −1 | MISS_g0 ×10 |
| EMUL obs, WITH look-at (see §3) | 0/10 | −1 | OOB_g0 ×10 |

It misses **gate-0**, the first gate, at low speed, from rest — not gate-4. The per-step dump shows it
*approaches* gate-0 (along-track 23.3 → 3.4 m) but **drifts grossly off-centre**: lateral obs 5.1 m,
vertical −6.4 m at ~3 m range. inc7 in the same harness stays centred (truthpos Y≈0.08 through the
approach, g3_linf 0.05). This is "does not fly", not "flies slightly off".

**Systematic across ALL inc8 ckpts** (truth obs, 3 ep each → 0/3 finished every time):
`stage1_inc8_actor` (pre-warm-start S2), `inc8_ws1_seed0/1/2`, `inc8_ws1_seed0_BEST`. It is the inc8
**lineage**, not a single warm-start seed.

**Why training showed "fix_rate 0.14–0.16 + pointing>0" yet reach=0 here:** those are PER-STEP BATCH
MEANS (`peregrine_racing_inc8.py` loss_components `inc8_fix_rate`/`inc8_terminal_pointing`). A policy that
points its camera but misses gate-0 and auto-resets every episode produces *exactly* fix_rate≈0.15 +
terminal_pointing>0 — with **zero laps completed**. The two numbers were never in contradiction; course
completion was simply **never measured**. The whole inc8 ladder (S0→warm-start) gated on pointing,
estim_err, and value_loss stability — the warm-start GO criteria (warmstart REPORT §6/§8) are literally
"value_loss must not cliff at ~step 1200; centering rises with pointing retained" — **never lap
completion**. σ_p0 (via `contact_true_eval`) is the first check of course-completion, and it failed.

**Pointing is REAL, not an artifact:** in eval the policy points (pointing_rate in the [12,28] m band
= 0.43) and lands fixes (n_fix ≈ 9/episode). So this is NOT the prompt's POLICY-GAP definition ("doesn't
point → pointing was an artifact"). Pointing was retained; **course-flight / gate-centring was not**.

`fixrate_g4 = 0` / `lock_g4 = 0` in the σ_p0 eval is therefore **STRUCTURAL-downstream**: the policy never
reaches gate-4 (dies at gate-0), so the gate-4 windows are empty. It is NOT a fix-acceptance harness bug.

---

## 3. The look-at primitive IS missing from the eval (real, but NOT the cause of reach=0)

The TRAINING env composes the active-perception **look-at primitive** onto the policy action BEFORE the
dynamics (`rl/peregrine_racing_inc8.py:233-246`): `action[...,1:4] += band * lookat_correction(...)`,
range-gated to [8,30] m. The EVAL (`rl/contact_true_eval.py:307-309`) steps the plant with the **raw**
policy action (`action = concatenate([rate_frd, [collective]])`) — **no look-at**. So the eval does not
reproduce the deployed control law. This is a genuine instrument incompleteness and must be fixed for a
faithful σ_p0.

**But it is not what blocks reach.** (a) The policy fails on TRUTH obs with **no** look-at involved at all.
(b) A faithful-as-possible look-at re-injection (numpy port of `inc8_reward.lookat_correction`, physical
FRD `w_cam @ r_bc.T` added to `rate_frd`, g_yaw=−3.0/g_pitch=+3.0, band [8,30] m) made it **worse** (OOB):
the loaded policy co-adapted to the training look-at + gain-warmup and an uncoordinated re-injection
destabilises it. The look-at is range-gated camera-pointing of small magnitude — it is not a translational
controller and cannot, by construction, fix a 5 m gate-0 centring drift.

---

## 4. The three prompt hypotheses — ALL RULED OUT

| hypothesis (prompt) | finding | evidence |
|---|---|---|
| **obs[17:20] confidence triple** mismatched/zero/wrong → policy blind | **RULED OUT** | Policy misses gate-0 for EVERY constant triple (zeros, ones, c=1/age=0, 0.5/0.5, c=0/age=1). Both paths build it from `confidence_channel` (SIGMA_REF=0.05, TAU_STALE=0.10): eval `estimator_emul.py:304`, train `inc8_estimator_emul.py:553`. Same contract; not the cause. |
| **obs[0:17] base** — frame / +L sign / KF-pose seam | **RULED OUT** | Direct numeric diff of train builder `obs_zup_torch` (`inc8_estimator_emul.py:138`) vs eval builder `obs_from_zup` (`fly_rl.py:307`) over 200 random states (both virtual_flip) = **max 7.6e-6 → PARITY**. Identical layout & math. (Both are the inc7 builder inc7 flies on.) |
| **fix-acceptance wiring** — does eval consume the policy's pointing? | **YES, it does** | Eval emul `estimator_emul.py:275-282` derives fix acceptance from the truth attitude `st_cur.quat` → `FS.geometry` → `sample_fix`. Pointing → fixes works in eval (n_fix≈9, pointing 0.43). Not a harness bug. |

Side-note (`estimator_obs.py:67` from the survey): that file is the **DEPLOY** seam (real-detector chain,
#37) and DOES omit [17:20] — but it is **not in the σ_p0 eval path** (the eval builds its 20-dim obs via
`EstimatorEmulator.obs()`). It is a real deploy-chain gap, not the cause of reach=0.

---

## 5. INSTRUMENT-GAP vs POLICY-GAP — classification

**POLICY-GAP (course-flight), NOT an obs/wiring instrument-gap.** Refines the prompt's POLICY-GAP: the
policy's *pointing* is real and retained; what is missing is *course-completion / gate-centring*, which was
never an objective the ladder verified. The eval instrument is faithful for obs (parity-proven) and
fix-wiring (works); its one real incompleteness (missing look-at, §3) does not cause reach=0.

**The one caveat I could not close on this laptop (needs Adroit/diffaero):** whether the policy flies in
its *own* training env. Two sub-cases, distinguished by ONE cheap TB query:
- read the inc8 training scalars **`success_rate`** and **`n_passed_gates`** (already logged by the env,
  `peregrine_racing_inc8.py:443-450`) for the ws runs (jobs 3275300-02) and for stage1_inc8 (S2).
- `success_rate ≈ 0 / n_passed_gates low in training` → **pure POLICY-GAP** confirmed (never flew, even in
  diffaero). Most likely given inc7 transfers diffaero→rl_plant cleanly while every inc8 ckpt fails.
- `success_rate high in training but 0 here` → a narrower control-law/plant transfer instrument-gap (the
  rl_plant `mixer` eval — the *fully-measured* plant inc7 validated on — is the more deploy-faithful one,
  so this would still be a deployability failure, not a free pass).

---

## 6. RECOMMENDED FIX (ordered)

1. **(Adroit, cheap, decisive — do FIRST)** Read in-training `success_rate` / `n_passed_gates` for the ws
   + stage1_inc8 runs. Settles pure-POLICY vs transfer-gap in one query. **This is the gate for everything
   below.**
2. **Add COURSE-COMPLETION to the inc8 training/selection GO criteria.** The ladder optimized
   pointing+estim_err+value_loss and never required a completed lap; that is the process hole that let
   reach=0 hide behind "fix_rate 0.15 + pointing>0" until σ_p0. `n_passed_gates`/`success_rate` must be a
   first-class GO number alongside σ_p0.
3. **If POLICY-GAP confirmed:** revisit the inc8 reward. inc7's gate-centring progress term (R1-to-centre,
   rw_progress=10) was REMOVED (`peregrine_racing_inc8.py:329` `w_frozen=replace(rw, progress=0.0)`) and
   replaced by R1' arc-progress-along-Γ + a near-gate-only centring term (rw_centering, sigmoid peaked at
   crossing). That combination evidently under-produces tight gate-centring (drift ≈5 m at gate-0). Restore
   a course-completion/centring pressure that is active through the whole approach, not just <8 m.
4. **Port the look-at primitive into `contact_true_eval.run_episode`** (it is part of the deployed control
   law) so σ_p0 reflects the real control loop — needed regardless, but it will NOT by itself yield reach>0.
5. **σ_p0 stays UN-MEASURABLE / NO-GO** until a policy completes the course. Any σ_p0 remains PROVISIONAL
   vs the real detector chain (#37) even after.

---

## 7. MEMORY-DELTA (text only — commander banks; do NOT commit memory/)

```
inc8 OBS-PARITY DIAGNOSIS DONE (2026-06-17, laptop main; diag_parity.py in handoff). #37 'emul fiction'
RE-SCOPED: the σ_p0 reach=0 is NOT an obs/wiring instrument-gap — it's a POLICY-GAP (course-flight).
- ALL 3 prompt hypotheses RULED OUT: obs[0:17] train-builder(obs_zup_torch) == eval-builder(obs_from_zup)
  PARITY 7.6e-6 over 200 states; obs[17:20] irrelevant (misses gate-0 for ANY constant triple); fix-accept
  wiring WORKS (eval derives accept from truth attitude→FS.geometry; n_fix≈9, pointing 0.43 in eval).
- ROOT CAUSE: the inc8 policy POINTS but CANNOT FLY — misses GATE-0 even on PERFECT (truth) pose, drifts
  ~5 m off-centre; inc7 finishes 5/5 in the IDENTICAL harness (validated). SYSTEMATIC across stage1_inc8 +
  all ws seeds. fixrate_g4=0 is downstream (never reaches g4). Training 'fix_rate 0.14-0.16 + pointing>0'
  = PER-STEP batch means on an early-resetting policy = NOT lap completion (which was NEVER a GO criterion).
- look-at primitive IS missing from eval run_episode (train peregrine_racing_inc8.py:233-246 composes it;
  eval contact_true_eval.py:307-309 uses raw action) — real instrument incompleteness, port it, but NOT
  the cause (policy fails on truth obs w/o look-at; a faithful re-inject went OOB).
- VERDICT POLICY-GAP. DECISIVE next (Adroit): read in-training success_rate/n_passed_gates (env logs them,
  peregrine_racing_inc8.py:443-450) — ~0 ⇒ pure policy-gap (likely; inc7 transfers, inc8 doesn't). FIX:
  add course-completion to inc8 GO; reward removed R1-to-centre (line 329) for arc-Γ+near-gate centering →
  under-centers. σ_p0 stays NO-GO/UN-MEASURABLE until a policy completes a lap. diffaero NOT importable on
  laptop (can't run train env here).
```
