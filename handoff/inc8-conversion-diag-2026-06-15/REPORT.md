# inc8 Arm-A pointing→fix CONVERSION diagnostic — REPORT
**Session:** inc8 pointing→fix CONVERSION diagnostic · opus-4.8 · 2026-06-15
**Subject:** pilot job 3273437 (arm A, 6000 upd, NENVS=2048), ckpt `/scratch/network/fl3689/diffaero/outputs/train/2026-06-14/23-14-55/`
**Boundary honored:** DIAGNOSE ONLY. No reward/env edits, no L0, no fix applied.

---

## VERDICT (load-bearing, one line)

**It is (A), root-caused precisely: the arm-A perception reward is RANGE-ANTI-ALIGNED with the fix-accept band — its terminal-lock weight `w_term` peaks (=1.0) at d≤5 m, which is the accept band-pass DEAD ZONE (p_accept≡0 below ~14 m), while the 16–28 m band where fixes are actually possible gets only 0.15–0.51 weight. The policy correctly optimizes this by pointing in the ≤5 m dead zone; premature entropy collapse (~2300) then freezes that mis-targeted pointing. (B) is RULED OUT — the accept/KF machinery fires whenever good geometry occurs, boresight is not in the accept path, and the wiring is self-consistent.**

→ Fix lives in **reward weight + exploration** (the prompt's (A) bucket), **NOT** in boresight bake / surrogate acceptance / wiring (the prompt's (B) bucket). Do not spend effort on (B).

---

## The decisive evidence (all from code + the EXISTING pilot logs; no GPU rollout needed)

### 1. The reward and the surrogate are anti-aligned in RANGE
Computed from the real `rl/fix_surrogate.py` band-pass and the real `rl/inc8_reward.terminal_weight` (arm-A defaults `d_lock=5, d_acq=24, w0=0.15`, confirmed un-overridden in `peregrine_inc8_smoke.sbatch`):

```
 range   p_accept|in_img   w_term(armA)   note
     0          0.0000         1.000   <- DEAD ZONE (no fix)   [reward PEAK]
     5          0.0000         1.000   <- DEAD ZONE (no fix)   [reward PEAK w_term=1.0]
    10          0.0017         0.776
    14          0.0847         0.597
    16          0.3809         0.508
    18          0.7234         0.418   <- ACCEPT BAND (fixes here)
    22          0.8376         0.239   <- ACCEPT BAND   (reward only 24% of terminal)
    24          0.8284         0.150   <- ACCEPT BAND
    26          0.7531         0.150   <- ACCEPT BAND
    28          0.4499         0.150
```

- **At the reward peak (d≤5 m, w_term=1.0): p_accept = 0.000 → a fix is IMPOSSIBLE there.**
- **At the accept peak (d≈22 m, p_accept=0.84): w_term = 0.24 → the reward pays only 24% of terminal.**
- The reward **maximizes camera-pointing exactly where the surrogate can never produce a fix**, and under-weights the one window (16–28 m) where the validated 8.5× in-window gain lives. The two are designed against each other.

This also exposes a **metric trap**: `inc8_terminal_pointing` is measured at `range ≤ perc_d_lock_m = 5 m` (env line 318) — i.e. it reports pointing *inside the dead zone*. The pilot's "pointing emerges, terminal_pointing ~0.065" headline is genuinely pointing **where fixes are impossible**.

### 2. The 16–28 m band IS reached every approach — opportunity exists, the policy just doesn't point there
VQ1 gate spacing (from `peregrine_course_diffaero.json`): 24.2 / 29.2 / 39.0 / 24.4 / 24.0 m (mean 28.2 m). Each approach spends ~31–41% of its length inside 16–28 m. **The band is not structurally absent** — so the failure is *pointing*, not *geometry reachability*.

### 3. fix_rate≈0 is at the OFF-IMAGE FLOOR → conversion machinery works, no good-geometry bug
- **The machinery is already proven to fire:** the GREEN×2 escape-hatch validated that the SAME surrogate+KF converts in-window pointing → fixes (8.5× in-window gain), and the training torch port is parity-gated against that numpy reference (`tests/test_inc8_*_torch`). So conversion is not broken — the only question is whether the *policy reaches the geometry*.
- fix_rate quantum = 1/2048 = **0.000488**; observed values are exactly integer fix-counts (0/1/2/3 per step over 2048 envs).
- Plateau **mean fix_rate ≈ 0.00032 (0.65 fix/step)**; the off-image accept floor (`accept_p_out_of_image=0.00021`) over the ~1991 not-in-image envs alone ≈ **0.42 fix/step (rate 0.00020)**. The small excess (~0.23 fix/step, rate ~0.0001) is the in-window contribution — **nonzero, so in-window fixes DO occur, but at a vanishing rate.** No good-geometry blockage; this is purely a pointing-distribution problem.

### 4. The policy points TERMINAL, ~500× more than in-window
Solving the logged metrics for the achieved in-window pointing:
`mean fix_rate ≈ P(in_image ∧ range∈[16,28])·pmax + p_out` ⇒ **P(in_image ∧ band) ≈ 0.00013**, vs `terminal_pointing` (in_image @ ≤5 m) = **0.065**. The camera is in-frame on the gate **~515× more often in the ≤5 m dead zone than in the 16–28 m fix window**. The policy learned to lock the gate at point-blank range (large, easy, w_term=1.0) and **essentially never holds it through the fixable window**.

---

## Answers to the 5 prompt questions

1. **Instrumented eval** — not runnable from here: Adroit SSH is Duo-gated (`adroit_key` → `Permission denied (publickey,keyboard-interactive)`; `claude_adroit_key` is for the Orin, not Adroit). I could not pull the checkpoint or the TB scalars. **But the existing logs + code already answer A-vs-B decisively** (above), so a rollout is *confirmatory*, not load-bearing. The confirmatory eval is specified below.

2. **Is p_accept≈0 when in_img? Which term kills it?** When in_image=True the killer is **RANGE**: p_accept is the band-pass, ≈0 for r<14 m and r>30 m. The policy's in_image mass sits at ≤5 m (terminal) → p_accept≡0 there. It is **not** |β| or boresight pushing the projection out — the emulation's accept depends only on `(range, in_image)`. p_accept is HIGH (0.72–0.84) at 18–26 m but the policy is rarely in-frame there (§4). **No high-p_accept-but-no-fix case exists** → no wiring/threshold/units bug.

3. **Does the policy reach the GREEN×2 in-window geometry?** The GREEN×2 8.5× gain was measured **in-window (18–28 m, gate centered az AND el)**. The trained policy reaches the in-window **range** every approach but holds the gate **in-frame** there only ~0.0001 of the time (§4) — **No, it essentially never reaches the in-window accept geometry with the gate in frame** (~515× rarer than its terminal pointing). That is the conversion gap.

4. **Boresight (is it injected? does BoresightCorrection(−0.25) change p_accept?)** **NOT injected into the accept/geometry path.** `inc8_estimator_emul.batched_geometry` computes in_image/range/az/el from TRUTH pose with the canonical `R_CAMERA_FROM_BODY_NP` (no boresight). The −0.25 m vertical boresight is modeled only as the per-episode one-signed in-plane **fix-VALUE** bias (`bias_mag 0–0.19 m`), which perturbs the KF measurement `z`, **not** whether a fix is accepted (accept = f(range, in_image) only). **Applying `BoresightCorrection(vert_offset_m=−0.25)` would NOT change p_accept at the policy's geometry. Boresight is a red herring for the training conversion gap.**

5. **Entropy test — does the pointing plateau coincide with entropy death?** **Yes.** Pointing plateaus ~step 1500 while entropy is already collapsing (2.21→0.74 by 1500→0.07 by 2100), going **negative ~2300** (action σ collapsed below ~0.24/dim → near-deterministic). The weak pointing is a **premature-convergence artifact, not a true ceiling** — the policy stopped exploring before it could find the harder in-window (elevation-bound) pointing. BUT entropy collapse only explains why pointing *froze at 0.065*; the *direction* it froze in (terminal, not in-window) is set by the reward anti-alignment (§1). Restoring entropy alone would push toward MORE terminal pointing, since that is where the gradient is strongest.

---

## Ranked root cause

1. **PRIMARY — reward↔accept RANGE anti-alignment (reward-design defect).** Arm-A `w_term` peaks (1.0) at d≤5 m inside the accept dead zone; the 16–28 m fix band gets 0.15–0.51. The policy is rewarded most for pointing where no fix is possible. Provable from constants; independent of exploration.
2. **CO-PRIMARY — premature entropy collapse (~2300).** Freezes the mis-targeted terminal pointing; blocks discovery of in-window pointing. Necessary to fix, but insufficient alone.
3. **CONTRIBUTING — in-window pointing is intrinsically HARD (elevation axis).** Per fix-surrogate §C, at fixable range the active gate is in the VFoV only ~10% of the time (|el| p50 44.6° vs VFoV-half 29.4°) — the fixed +20° camera + steep-descent posture misses the gate vertically. Even with §1/§2 fixed, the policy must learn an explicit pitch/attitude trade to put the gate in the VFoV at 16–28 m.
4. **RULED OUT —** boresight (Q4), surrogate-can't-fire (it fires, §3), wiring/units/threshold (self-consistent; no high-p_accept-no-fix case). Also note: target-gate-ONLY fix (emulation queries only the next gate, not any visible gate) caps the achievable ceiling but is shared by the GREEN×2 validation, so not the differential cause.

*Minor aside (not load-bearing):* `conf_anneal_ramp_steps=1.0` makes the confidence-shaping anneal reach full weight after 1 step (defeating its stated "anneal late" purpose), but conf_shape weight is only 0.05 so it is negligible vs progress (10·Δs) and GT (2·err).

---

## Recommended fix (RECOMMENDATION ONLY — not applied; owner decides)

**Primary (reward) — realign the perception reward to the accept band.** Either:
- (a) **Move the lock window onto the band:** `perc_d_lock_m ≈ 18`, `perc_d_acq_m ≈ 28` (and consider raising `perc_w0`), so w_term=1.0 across 16–28 m and ramps DOWN inside 16 m. This makes "terminal lock" mean *hold the gate centered through the fixable approach (28→16 m)* — which is what the validated 8.5× in-window gain and the margin-closure "gate-lock ≥60% of final approach" actually require (the final approach that produces fixes is 28→16 m, not 5→0 m). **Cheapest, no code change — just `+env.rw_perc_d_lock_m=18 +env.rw_perc_d_acq_m=28`.**
- (b) **Or couple reward to the fix directly:** add a term on `p_accept(geom)` or the `accepted` event (already computed each step), gated to in-image. This points the gradient exactly at the quantity that must rise, instead of a visibility proxy whose weight peaks in the dead zone.

**Secondary (exploration).** Raise / slow the entropy decay (or larger initial action σ, or a KL/entropy floor) so the policy keeps exploring past ~2300 and can find the harder in-window elevation pointing. Necessary because in-window geometry is harder to stumble into than terminal.

**Caveat to verify.** The elevation block (#3) means even (a)+(b) may convert slowly until the policy learns the pitch trade; budget for it and watch `inc8_fix_rate` rise specifically in the 16–28 m band.

---

## What's needed to EMPIRICALLY confirm (I could not reach Adroit)

A confirmatory instrumented rollout of the checkpoint (no retrain) logging, per step with `in_image=True`: **range, az, el(β), p_accept, accepted**. Expected (from this diagnosis): in_image mass concentrated at ≤5 m, ≈0 in 16–28 m; and at 16–28 m the |β| sits outside ±29.4° (the elevation block). Cheap alternative that needs only the login node (no GPU): **pull the TB scalars** `inc8_r5_perc`, `inc8_gt_anchor`, `inc8_estim_err_inplane_m`, `inc8_c_inplane`, `inc8_age_norm` from the run dir to confirm the reward decomposition (hypothesis: GT-anchor `−2·err_ip` dominates and is a penalty the policy can't reduce without fixes it can't get — a reward trap that compounds the negative-reward plateau). Reconstruction script: `handoff/inc8-conversion-diag-2026-06-15/reconstruct.py`.

---

## MEMORY-DELTA
- 🚩 **inc8 arm-A CONVERSION ROOT-CAUSED (job 3273437, diag 2026-06-15): REWARD↔ACCEPT RANGE ANTI-ALIGNMENT, not (B).** Arm-A `w_term` peaks (1.0) at d≤5 m = the accept band-pass DEAD ZONE (p_accept≡0 <14 m); the 16–28 m fix band gets only 0.15–0.51 weight. Policy correctly points TERMINAL (≤5 m, terminal_pointing 0.065) → ~515× rarer in-window (P(in_img∧band)≈1.3e-4) → mean fix_rate ~3e-4 sits at the OFF-IMAGE FLOOR (0.42 fix/step) + a vanishing in-window excess (~0.23 fix/step); quantum 1/2048. **terminal_pointing METRIC is measured at ≤5 m = inside the dead zone (misleading "pointing emerges" headline).**
- 🚩 **(B) RULED OUT:** GREEN×2 (parity-gated) already proved the surrogate+KF converts in-window pointing→fixes; mean fix_rate sits just above the off-image floor → machinery FIRES at good geometry (no wiring/threshold bug). **Boresight is NOT in the emulation accept path** (accept=f(range,in_image) only; −0.25 m enters only the fix-VALUE bias) → BoresightCorrection would NOT change p_accept → red herring for the training gap.
- 🚩 **Premature entropy collapse (~2300, σ<0.24/dim) FREEZES the mis-targeted pointing** — co-primary; plateau coincides with entropy death (Q5 confirmed). Necessary-but-insufficient to fix alone.
- **CONTRIBUTING:** in-window pointing is HARD — +20° cam + steep descent → gate in VFoV only ~10% at 16–28 m (fix-surrogate §C elevation co-bind). RECO (not applied): `perc_d_lock≈18 / perc_d_acq≈28` (lock onto the band) OR an explicit p_accept/accepted reward term, + restore exploration. CONFIRM by pulling TB `inc8_r5_perc/gt_anchor/estim_err` + a range-resolved in_image histogram.
- **Adroit SSH is Duo-gated from the laptop session** (`adroit_key` publickey denied; `claude_adroit_key` is the Orin's) → workers here cannot pull /scratch or submit SLURM directly; relay required.
