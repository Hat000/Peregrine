---
name: campaign-20-gates-2026-07-27
description: "SSOT for the 20-gate campaign: the capability verdict (P(20 gates) today ~5e-4), the gap decomposition (28% deploy / 45% both / 27% training), the ranked training arms, and the arm that must NOT be run."
metadata: 
  node_type: memory
  type: project
  originSessionId: b85130f9-ac88-4db2-8c68-0e28b966cf80
  modified: 2026-07-27T07:31:19.692Z
---

# The 20-gate campaign — 2026-07-27 (Fengyou: "full authorization, work into the night")

Two ultracode workflows, **48 + N agents**, every finding attacked by an independent skeptic whose
default verdict was "refuted". Only **3 of 41** training-side findings survived refutation intact;
the wounded ones carry corrected claims that are often worth more than the originals.

## 🛑 THE CAPABILITY VERDICT: NEITHER HALF ALONE GETS THERE

Measured survival ladder, n=670 runs: reach counts `[670, 497, 345, 236, 144, 65, 16, 8, 3, 2, 0]`;
conditional per-gate survival `0.742 / 0.694 / 0.684 / 0.610 / 0.451 / 0.246 / 0.500 / 0.375`.
Best cohort (rate-30 + v19Ws0, n=160): clean gates 0–3 **p = 0.8075**, obstacle band 4–5
**p = 0.3634**, then **recovery to 0.556 at gate 6** — a discrete wall, not policy decay.

| scenario | E[gates] | P(20 gates) |
|---|---|---|
| A — today | 2.77 | **4.8e-04** |
| B — obstacle band fixed to the clean rate (deploy dodge) | 4.14 | 1.4e-02 |
| C — entire sim↔wire gap closed (p = 0.935, the sim ceiling) | 10.67 | 0.263 |
| D — sim ceiling itself raised (p = 0.966) | 14.17 | 0.500 |

(Model validated: A predicts E = 2.77 against a measured 2.750.) **A 50% lap needs per-gate
p ≥ 0.9659; even a 5% lap needs 0.8609 — above today's clean-gate 0.8075.**

**Gap decomposition to a 50% lap (q 0.3812 → 0.0341, a factor of 11.2):**
* **28.3% — the obstacle band. DEPLOY-ONLY.** There is **no obstacle model anywhere in training**
  (`grep obstacle` returns 0 hits in `peregrine_racing_ego.py` + `peregrine_course.py`), and the
  actor obs is position-free, so a per-gate dodge is **unlearnable by the policy**.
* **45.2% — sim↔wire transfer.** Mixed: the loop-rate half is deploy, the rest is contract
  mismatch needing a retrain.
* **26.5% — the sim ceiling itself. PURE TRAINING.** No deploy knob reaches it.

⇒ **"Just fix deploy" caps at 4.1 gates / a 1.4% lap. "Just retrain" is capped by the obstacles at
gate 4.** Both halves are required.

📏 Training score for context: `DET_EVAL n_passed_gates` **5.884** (v20Vs0 vert) / 5.973 (flat);
v19Ws0 4.893 / 5.880 — on an **8-gate resampled** course, legs U[10,20] m, turns ±60°, scored in a
**600-step = 19.98 s** window against a 60 s cap, so long successful episodes are right-censored.
Inverting gives sim per-gate survival **p ≈ 0.935** (best policy, best env).
📏 **Effective value horizon:** γ = 0.9975 at dt 0.0333 ⇒ 400 steps = **13.32 s ≈ 6.3 gates**. This
does NOT cap the policy (the 21-dim actor obs carries no gate index, no progress, no clock, so
gate-19 states are drawn from the same distribution as gate-1) — **it caps the CRITIC.**

## 🚩 WHERE THE DEPLOY COURSE LEAVES THE TRAINING DISTRIBUTION

| item | training support | deploy | status |
|---|---|---|---|
| gates/episode | 8, resampled | 20, fixed | in-kind, not in-extent |
| leg length | **hard [10.00, 20.00] m** (40 000 sampled courses, zero mass outside) | g1→g2 **≈9.7 m**, g3→g4 **22.5**, g4→g5 **23.5** | **3 legs at/outside support** |
| obstacles | **none at all** | two, ~14.5 m before gates 4 and 5 | **out of support entirely** |
| gates 10–19 | n/a | 2/670 flights reached gate 9, **0 passed it** | unmeasured |

## 🟢 RANKED TRAINING ARMS (5 seeds each ⇒ ~3.3 non-collapsed; 21.5 A100-GPU-h ≈ 12.9 h wall)

1. **ARM 1 — `+env.ego_obs_coast=true`. One token, no code, no warm-start break.** 🛑 **CONTESTED —
   it has already run once in a noise-0 sweep (see the second-pass corrections). Still ranked first,
   but launch it knowing the prior evidence is mildly AGAINST, not absent.**
   Motivated by the terminal-mask inversion → [[vision-horizon-fov-2026-07-27]]. Aborts: at t+60 s
   the key must appear in `.hydra/config.yaml`; at t+4 min `metrics/cross_offset_m` must NOT be
   bit-identical to control; at t+30 min `banked_prog_mean` ≥ 50% of control and `exit_oob` ≤ 0.10.
2. **ARM 2 — `+env.ego_est_dt_ticks_hi=3` with a re-measured tick-gap pmf.** The baked
   `MEASURED_TICK_GAP_PMF` came from **73 gaps**; re-measured over **29 557** it is
   `[0.7131, 0.2313, 0.0509, 0.0022]` (E[k] 1.338). **28.7% of wire ticks freeze the estimator ≥2
   training tick-lengths and `ticks_hi=1` models that on 0%.** Accept criterion is *robustness*
   (≥ control − 0.20 in BOTH envs), never a score gain.
3. **ARM 3 — privileged critic depth channels, `EGO_CRITIC_DIM` 16 → 18** (`gates_remaining_frac`,
   `banked_prog/100`). Legal: GT in the critic only, actor obs untouched. The critic cannot see
   `banked_progress_return`, which is subtracted on floor contact and swings the terminal between
   −200 and −500 against a per-gate income of ~45 ⇒ **~3 gates of irreducible value error**.
   Accept on **collapse count ≤ 1**, not on mean gates. Needs a ckpt-pad helper or the warm chain
   breaks.
4. **ARM 4 — course geometry coverage** (`course_seg_len_lo/hi`, `min_pair_dist`). Filler only.

✅ **THAT PRECONDITION IS NOW LIFTED:** `rw_cross_zero_m` **= 0.75** (`launch_v19.sh:240` `++`
force-overrides the stage's `+…=4.0`). Confirm with one `.hydra` cat when the cluster returns, but
the in-repo chain is unambiguous, so reward arms are no longer blocked.

## 🛑🛑 THE ARM THAT MUST NOT BE RUN — and MEMORY named it as the lead candidate

**The L-inf lateral-asymmetry arm is DEAD.** Three independent reasons:
1. **It has already been run at real dose and lost.** v1.8 mechanism M4 armed
   `pass_margin_final_m 0.75` + `lat_weight 2.0` with the verbatim rationale "lateral risk ~2×
   vertical", ran 18k updates × 4 arms, **shipped as v18Q_s1**, and was reverted in v1.9 with
   `pass_offset` **FLAT at 0.174–0.179 across all five v1.7/v1.8 runs and BOTH aperture settings**.
2. **The adjudicating metric had ample power** — a 20% lateral-σ reduction would move
   `pass_offset_m` by −7.1%, i.e. **2.5× the entire five-arm spread**; and it was *biased in M4's
   favour* (censored exactly where |lat| is largest) and still moved only −1.9%.
3. **The mechanism is in the wrong place** — the lateral tail is created in the blind final metres
   where no reward term reaches (terminal |lat| on all 4–8 m state has R² = 0.004).

🛑 Runner-up also rejected: the anti-clip terminal. `single_gate_varied_gvf_lpara_clip0` already
annealed `clip_pen 0→20` against a matched control; the curriculum records the verdict itself —
**"collisions ~0.5 immune to a −20 clip penalty."** 🚩 Genuine dead-code defect found alongside it:
`apply_contact_kill` (worth −50) lives only in the `else:` legacy branch while `_use_refined_b`
defaults True ⇒ **the documented "ABSOLUTE kill-on-contact" is dead code on every shipped ego run.**
Fix the source; do not buy a run for it.

## 🚩 HONEST UNCERTAINTY (the synthesis's own, worth preserving)

1. **Every 20-gate number extrapolates over 11 unobserved gates** assuming i.i.d. survival. Gates 4
   AND 5 each carry an obstacle; if that density continues the truth is **worse**, and P(20) is
   effectively zero until the back half is surveyed. Resolvable only by flying deeper.
2. **Arm 1's direction could be backwards** — if the wire's coasted lever is degraded enough that
   training on it teaches the policy to trust garbage, the arm costs 21.5 GPU-h.
3. ~~`rw_cross_zero_m` unresolved~~ -- **RESOLVED = 0.75 in the second pass, see above.**
4. Proving one launcher token landed does not prove they all did.
5. **The obstacle's COLLISION-id evidence is weaker than the lanes claimed** — the `1002` rows carry
   `altitude_minimum_delta` and `threat_level: 1` with tiny impulses and look like **proximity
   advisories, not impacts**. Rest the obstacle case on the survival-ladder shape and the in-repo
   14/31-deaths measurement instead. **Do not cite the ID split as primary.**

## 🟢🟢🟢 THE GATE-4 DODGE WORKS — 89% vs 30%, p = 0.0052, WITH A BY-CONSTRUCTION PLACEBO

**The strongest positive result of the campaign, and it is Fengyou's own dodge.** Matched cohort
rebuilt from `meta.json` (v19Ws0 · commanded rate 30 · pitch clamp 20 · z_bias 0.30, n=57),
differing ONLY in whether a vertical offset was armed — read from the LOG (`aim_off` non-null on
that gate index AND |vertical| > 0.5), never from the config string, because the release-0.0 trim
flights carry offsets on every gate with ZERO vertical and must not count as dodges:

| gate | reached | WITH dodge | WITHOUT | Fisher |
|---|---|---|---|---|
| 3 (PLACEBO — the dodge cannot arm here) | 38 | n/a | 29/38 (76%) | by construction |
| **4** | 29 | **8/9 = 89%** | **6/20 = 30%** | **p = 0.0052** |
| 5 | 14 | 4/8 = 50% | 2/6 = 33% | p = 0.627 — **UNPROVEN** |

🟢 The placebo is structural, not chosen: the offset only arms at `gate_index` 4 and 5, so gates 0–3
are a control **by construction**. 🚩 The three flights that removed the dodge (release 0.0) died on
terminal ENV impacts at **12.87 / 14.59 / 14.92 m** from gate 4 — the obstacle band, exactly.
🛑 **GATE 5 IS NOT SOLVED.** 3 of its 4 deaths are at-gate strikes, not band deaths, and its one band
death happened at 18.3 m **with the dodge armed** ⇒ +3 m up is not sufficient there.

## 🛑 CORRECTIONS ESTABLISHED IN THE SECOND PASS

* **`rw_cross_zero_m` IS RESOLVED = 0.75, at zero GPU cost.** `launch_v19.sh:240` `VPEF8NC` sets
  `++env.rw_cross_zero_m=0.75`, and `++` force-overrides the stage's `+env...=4.0`
  (`vq2_ego_curriculum.py:1379`). Curvature 20/0.75² = **35.6 reward/m²**. This is a code-chain
  argument, not a launcher-absence one, so it is legitimate — still confirm with one `.hydra` cat.
  **The standing precondition that blocked every reward arm is lifted.**
* **The 1002-contact caveat is refined, not upheld.** *Terminal* 1002 rows carry `threat_level: 2`
  and impulse **2.06–7.45** — the same band as confirmed GATE strikes (4.1–7.3). *Post-terminal*
  rows carry `threat_level: 1` and impulse 0.03–0.18. **Discriminator: `threat_level == 2 AND
  impulse > 1.0`.** The obstacle case survives on the terminal rows.
* **ARM 1 (`ego_obs_coast`) is CONTESTED, not clean.** `vq2_ego_curriculum.py:929-931` records it
  already run in a noise-0 calibration sweep at **0.522**, inside a 0.32–0.64 band. The counter —
  that at `ego_noise_scale: 0.0` the coasted estimate is "nearly free (~0.01 m over 1.5 s)" and so
  carries no realism content — is real but is an argument, not a measurement. **Weigh it as n≈1
  prior evidence AGAINST, not as an untried lever.**

## 🚩 KILL-MODE SPLIT (30 flights with post-impact data, the only impact data that exists)

**GATE strike 20** (lateral 8 · vertical 6 · both 1 · **inside the aperture at the last fix 5**) ·
**ENV impact 7** · unusable 3. ⇒ **two-thirds of deaths are at-gate strikes, and a quarter of those
were already on a passing line when vision stopped** — the blind-interval scatter, on tape.

Related: [[vision-horizon-fov-2026-07-27]] · [[aim-cohort-and-loop-rate-2026-07-27]] ·
[[failure-census-2026-07-27]].
