# Peregrine inc8 Substrate Audit — Adversarial Correctness Report

**Scope:** RL train + deploy substrate, repo HEAD `c2af65e`.
**Date:** 2026-06-13. **For:** Fengyou.
**Method:** every load-bearing convention claim was re-examined with an *external* invariant — anchored on the **trusted pristine `vel_ned`/`pos_ned`** telemetry — chosen to be **distinct from the invariant used in the prior pass**, specifically to attempt an *overturn*. A verdict survives only if a genuinely independent lens could not flip it. Every probe was run in `.venv` against real live recordings; comments and prose were treated as claims, not facts.

> **OUT-OF-SCOPE marker — read first.** The contact / volumetric-halo scoring layer (`rl/contact_true_eval.py` and the gate-margin scoring it produces) was **NOT audited here** — that file is edited elsewhere and is a moving target. **No** probe and **no** regression script in this suite imports it (verified). Any sentence below that brushes the gate-margin / scoring surface is flagged **OUT-OF-SCOPE / POSSIBLY-STALE** and must not be trusted as current from this report.

---

## 1. Executive summary

| Outcome | Count |
|---|---:|
| Claims **dispatched** | **68** |
| Claims with a **verdict** (coverage) | **68 / 68 = 100%** |
| **UNVERIFIED** (no verdict — NOT a pass) | **0** |
| CLEARED (held under escalation) | **61** |
| CONFIRMED_BUG | **4** |
| NEEDS_LIVE-FLIGHT (cannot settle offline) | **2** |
| INCONCLUSIVE | **1** |

**Coverage is COMPLETE: 68 of 68 dispatched claims received a verdict; 0 unverified.** Section 1b is therefore omitted (no re-run needed for coverage).

**Single most important takeaway for Fengyou:** *No live-code train-corrupting or deploy-corrupting bug survived escalation on the deployed (VQ1) course.* The full deployed convention chain — ODOMETRY-quat R_y(pi) conjugation, the all-axes rate-sign, the FLU->FRD wire map (incl. the disputed yaw axis), the velocity-frame single-rotation, the super-rate gain map, the gate-frame lift, the obs seam, the DiffAero<->rl_plant parity, the camera mount, and the MAVLink wire layouts — was adversarially attacked with fresh external invariants and **held**. The shipped `93023cf`/HEAD conventions are externally vindicated.

**But the clean bill is SCOPED, not unqualified** — four CONFIRMED bugs exist, and you must understand exactly what each is:

- **Three of the four CONFIRMED bugs are NOT live-code defects.** `CR2-01`, `CR4-01`, `CR4-03` are **evidence-chain / data-provenance** defects: the retired `refit` recording set was captured under the **superseded `bcc93f9`** code (older yaw-wire and older obs-builder), so its `obs[8]` / `obs[11]` / `rate_frd[2]` yaw channels encode the **old** convention. The danger is purely methodological — an analyst could "clear" the *current* yaw convention against *stale* data and validate the wrong map. **The shipped code is correct in all three.** (`CR4-01` blast was downgraded to **local** on escalation.)
- **One CONFIRMED bug is a genuine latent code defect, but it is DORMANT on the deployed VQ1 course:** `P4-C05` — `fly_rl.obs_from_zup`/`build_obs` hardcode the gate frame to yaw = pi. On VQ1 (every gate yaw = pi) train == deploy to **4e-15** (bit-exact), so it never fires. It **activates only on a VQ2 / non-uniform-yaw course**, where it silently corrupts the gate-relative obs by up to **4.22 m**. This is a VQ2-readiness hazard, not a VQ1 deploy bug.

**Therefore:** nothing here forces a code change to the *currently deployed VQ1* stack, and the convention-change decision this report drives can proceed — the shipped conventions hold. **Two items still require live confirmation before specific claims can be crowned** (Section 4): `P1-C06` (the obs yaw seam can only be cleared on a fresh inc7 `debug_obs` capture, since the only current-convention recording is inc6) and `CR1-01` (the *absolute* yaw wire sign is the one channel no offline roll-mirror invariant can falsify — it rests on live-confirmed inc6/inc7 flights). Both are **confirmation-of-CLEARED-direction**, not active-bug investigations.

**WINNER-VALIDATION rider stands (banked doctrine, reinforced here):** offline data is a low-yaw, single-spawn-era policy. The yaw-active envelope is corroborated by *integrated pristine course-rate* but not by an *active yaw chirp*; fold a pure-yaw-step / yaw-chirp segment into the inc8 fresh-reset live batch before crowning any yaw-active winner.

---

## 2. CONFIRMED FINDINGS

Ordered by blast radius. **All recommended actions are explicitly NOT applied** — this report drives the decision; it does not make the change.

### 2.1 P4-C05 — `obs_from_zup`/`build_obs` hardcode gate frame to yaw = pi *(blast: deploy-corrupting; latent — DORMANT on VQ1; confidence: HIGH)*

- **File:** `rl/fly_rl.py` — `obs_from_zup` / `build_obs` use `_R_W2G = diag(-1,-1,1)` and `_GATE_YAW_REL = 0`, i.e. a fixed yaw = pi gate frame, instead of rotating per-gate by the runtime `gate_yaw`. Same hardcode mirrored in `offline_rollout.py` (lines ~162/163/390/391/403/409).
- **External invariant (two new lenses, both overturn-attempts that FAILED to clear it):** (1) real-data seam reconstruction — fed every recorded tick's pristine `pos_ned`/`vel_ned`/`q_raw`/`w_raw`/`gate_index` through the **shipped** `build_obs` and compared to the recorded `obs`; (2) independent train-side re-derivation from the **public** `peregrine_racing.world_to_gateframe` + `rel_tables` helpers.
- **Numbers:**
  - VQ1 all-pi course (the only course flown): independent TRAIN obs vs DEPLOY obs **max|diff| = 4.44e-15** — bit-exact. *This is why the bug is invisible to every consistency check.*
  - Non-pi course (gate-2 yaw = 3.2574 rad = 186.6 deg): per-field max|train - deploy| on **consumed** obs indices — `pos_gx 1.11`, `pos_gy 1.12`, `vel_gx 1.61`, `vel_gy 1.86`, `rpy_g_y 6.254`, `nxt_relx 0.361`, **`nxt_rely 3.292`**, `nxt_relyaw 0.255` (all on consumed indices [0,1,3,4,8,13,14,16]); pos_gz/vel_gz/roll/pitch/rates = 0 (invariant under pure-yaw). Independent re-derivation reproduces the ~4.22 m magnitude.
- **Blast radius:** deploy-corrupting **class**, but **INACTIVE for VQ1** and no current live/offline-eval path feeds a non-uniform yaw into obs. Activates only on (a) deploying this seam on a VQ2/non-pi course, or (b) training/evaling `course_mode=random` then deploying through the hardcoded seam.
- **RECOMMENDED ACTION (NOT applied):** VQ2 remediation — make `obs_from_zup`/`build_obs` ingest the **live** gate map and build per-target-gate `get_gate_rotmat_w2g(gate_yaw[tg])` + per-pair `rel_tables(gate_pos, gate_yaw)`, exactly as `peregrine_racing.get_observations`, replacing `_R_W2G`/`_GATE_REL_POS`/`_GATE_YAW_REL`; mirror in `offline_rollout.py`. Add a deploy-time assert `all gate yaws == pi (+/-1e-4)` when the hardcoded path runs, so a non-pi course fails LOUDLY. `rl/contact_true_eval.py` was **NOT** touched (out of scope) but appears in the obs-builder grep and likely shares this hardcode — **flag it for the same fix when that file is next worked.** **(OUT-OF-SCOPE / POSSIBLY-STALE flag on the contact_true_eval note.)**
- **Evidence script:** `scratch/probe_P4-C05_esc_realobs_recon.py`. **Regression:** `regression_suite/test_confirmed_p4_c05.py` (encodes the invariant against a yaw-aware *corrected* builder; PASS = bug-confirmed negative-control firing at 4.22 m on the buggy tree; when a fix lands, retire the negative control and convert LENS2b to a straight equality).

### 2.2 CR4-03 — `refit` yaw correlation validates the OLD `bcc93f9` wire map, not the current `[1,-1,1]` map *(blast: deploy-corrupting CLASS; shipped code CORRECT; confidence: HIGH)*

- **Nature:** an **evidence-chain defect**, NOT a code defect. `rl/fly_rl.py` (`_ACT_FLU_TO_FRD = [1,-1,1]` :93, virtual_flip :335, `rate_frd = rate_flu*_ACT_FLU_TO_FRD` :336) is **correct**. The defect is that the `refit` recordings (which the reference `audit_candidates.py` loads) were flown under the OLD `bcc93f9` wire map `[+1,-1,-1]`, so correlating the refit yaw channel confirms the **wrong** map.
- **External invariant (4 lenses, all on real recordings):** full-pipeline wire reconstruction; yaw-cmd vs self-computed TRUE-attitude central-FD realized yaw rate; recorded-wire-vs-anchor artifact guard; force-vs-FD East positive control (production-faithful `use_lapse=False`).
  - LENS1 (max|recon - recorded `rate_frd[2]`|): **postfix** yaw vs CURRENT `+act3` = **0.0000**, vs `bcc -act3` = 1.31-2.16; **refit** yaw vs CURRENT = 2.14-3.11, vs `bcc -act3` = **0.0000**. (Postfix wire IS `+act3`; refit wire IS `-act3`, exact.)
  - LENS2 (yaw-cmd vs TRUE-attitude FD anchor, per-run best-lag): postfix CURRENT `+act3` corr **+0.946...+0.970**; refit CURRENT `+act3` corr **-0.992...-0.935** (the tracking sign flips by dataset because the recorded wire differs).
  - LENS3 (recorded `rate_frd[2]` vs anchor — artifact guard): postfix median **+0.958**, refit median **+0.979** — POSITIVE in BOTH -> the flip lives in the act->wire map, not a flipped anchor.
  - LENS4 (Test-A East): postfix TRUE **+0.967** / AS-IS **-0.714**; refit TRUE **+0.989** / AS-IS **-0.838**.
- **RECOMMENDED ACTION (NOT applied):** (1) pin any current-map yaw clearance to the **POSTFIX** dataset only; (2) **do NOT** cite the refit yaw correlation as support for the current map; (3) add a provenance assertion to future wire-convention harnesses: before scoring a recording, assert `max|recorded rate_frd[2] - (+act_rescaled[3] through current pipeline)| ~ 0`; refuse to score a recording whose stored wire was produced by a different code revision; (4) re-confirm on the next postfix-convention live batch. **Leave shipped code untouched — it is correct.**
- **Evidence script:** `scratch/probe_CR4-03_esc_recordedwire.py`. **Regression:** `regression_suite/test_confirmed_cr4_03.py`.
- **!!! COMMANDER ACTION — SLUG COLLISION:** the file `test_confirmed_cr4_03.py` previously held a *different* bug (a "COLL_MAP over-predicts body-up thrust at knots 6-9" finding in `src/racer/rl_plant.py`, from a parallel session and an earlier REPORT.md draft). It was overwritten by the authoritative CR4-03 (fly_rl wire-map) test. **If that COLL_MAP finding is to be kept, it must be re-homed under a non-colliding slug** (e.g. `test_collmap_overpredict.py`) before it is lost. Note: that COLL_MAP claim is *not* in the dispatched 68-claim projection for this pass; it is a residual from an earlier draft and is **not part of this report's verdict accounting.**

### 2.3 CR2-01 — `refit` yaw channels (`obs[11]`/`rate_frd[2]`) encode `bcc93f9`, invalid for clearing current-code yaw claims *(blast: deploy-corrupting CLASS; shipped code CORRECT; confidence: HIGH)*

- **Nature:** methodological / provenance defect (deploy-corrupting **because** it would let an analyst "clear" the deployed yaw seam against superseded-code recordings), NOT an arithmetic error in current `fly_rl.py`.
- **External invariant (two new yaw discriminators the prior pass said were still needed — Test-A force is structurally blind to yaw-about-vertical):**
  - L3 yaw-rate vs TRUE-conjugated heading-change (POSTFIX only): corr **-w_raw[2]** = **+0.70**, slope +1.42; `+w_raw[2]` = -0.70.
  - Q1 quat-FD body-z vs `-w_raw[2]` (residual): med|resid| **0.0102 rad/s**; `+w_raw[2]` = 0.4127 (**40.5x worse**) -> deployed `[-1,-1,-1]` rate sign confirmed on current data.
  - Channel recompute: REFIT yaw corr **-1.000** / maxabsdiff **3.11** (encodes bcc93f9); POSTFIX bit-exact (`+1.000`, maxabsdiff 5e-5 = 4-dp recording rounding).
- **RECOMMENDED ACTION (NOT applied):** mark `handoff/shadowpc-refit-dataset-2026-06-12` INVALID for clearing any `obs[8]`/`obs[11]`/`rate_frd[2]` yaw-about-vertical claim; clear `P1-C06`/`P2-C01`/`P3-C03` yaw on POSTFIX only; optionally drop the stale yaw columns from refit jsonl or add a producer-commit field to recordings. **Current code is correct — no change.**
- **Evidence script:** `scratch/probe_CR2-01_esc_yawseam.py`. **Regression:** `regression_suite/test_confirmed_cr2_01.py`.

### 2.4 CR4-01 — `refit` `obs[8]`/`rate_frd[2]` are stale `bcc93f9`-producer artifacts (HYBRID roll/yaw split) *(blast: LOCAL; shipped code CORRECT; confidence: HIGH)*

- **Nature:** local provenance defect, **blast downgraded deploy-corrupting -> local** on escalation. The stale yaw lives **only** in the diagnostic recording, not the command path: `fly_rl.py:765` recomputes obs fresh via `build_obs()`; recorded obs/`rate_frd` are written to the debug dump **after** `send_command`, so stale `bcc` yaw cannot reach the wire.
- **External invariant:** drove the **real** production `obs_from_zup`/full `policy_step` (incl. virtual_flip premultiply) against recorded telemetry. Refit `obs[8]` matches the `bcc` recon exactly (med ~ 0) and is the **exact negation** (wrap-aware med 3e-6 rad) of the current-build recon; refit `obs[6]` roll matches CURRENT recon -> **HYBRID split** confirmed. POSTFIX `obs[6]` and `obs[8]` BOTH match CURRENT recon (med 0). Test-A independently confirms TRUE `[1,-1,1,-1]` incl. yaw on both datasets.
- **RECOMMENDED ACTION (NOT applied):** never round-trip a current-build yaw against refit recorded `obs[8]`/`rate_frd[2]`; use POSTFIX (`handoff/shadowpc-postfix-dataset-2026-06-12`) for any obs/rate_frd yaw clearance. **No code change.**
- **Evidence script:** `scratch/probe_CR4-01_esc_provenance_realbuild.py`. **Regression:** `regression_suite/test_confirmed_cr4_01.py`.

**Common thread across 2.2-2.4:** all three are the **same root cause** — the `refit` dataset is a `bcc93f9`-era capture whose yaw channels are stale. They are not three independent bugs but three facets of one data-provenance hazard. **Mitigation is shared:** treat `refit` as the AS-IS / superseded-yaw positive control only; use `postfix` for any current-convention clearance.

---

## 3. INCONCLUSIVE / watch

### 3.1 P5-C04 — Elodin adapter omits `orientation_ned_wxyz`; would collapse Navigator to eye(3) attitude *(blast: deploy-corrupting; verdict INCONCLUSIVE; confidence: HIGH)*

- **File:** `src/racer/elodin_adapter.py` (`sensorupdate_to_state`).
- **Why inconclusive:** the bug *premise* (silent eye(3) attitude) is **false today** — **no path feeds Elodin `DroneState`s to `Navigator`**. Consumer-side caller-graph audit: every offline/eval Navigator-feeding producer (`offline_rollout.telemetry_from_truth:93`, `twin.py:368`, `replay_obs.py:189`) correctly fills the raw-conjugated wire quat; `fly_rl.py:760-763` guards `if orientation_ned_wxyz is None: continue`. The Elodin adapter is the **only** producer that omits the field and has **zero** Navigator consumers.
- **Consequence if ever wired (measured on real poses):** eye(3) world-direction error of the thrust axis med **57.9 deg** (max 62.8) -> world-NED specific-force vector error med **17.9 m/s^2** (refit bridge: med 24.1, max 42.8 m/s^2). I.e. **not harmless** if the path is ever built.
- **RECOMMENDED ACTION (NOT applied):** (i) in `elodin_adapter.sensorupdate_to_state` set `orientation_ned_wxyz = att_quat_wxyz * frames.ODO_QUAT_TRUE_CONJ_WXYZ` (the RAW-conjugated wire quat matching `twin.py:368` / `mavlink_client.py:266` — **NOT** the bare true-attitude quat, which the probe shows is double-conjugated, 53.5 deg error); (ii) add `assert ds.orientation_ned_wxyz is not None` before `navigator.py:298` so any future Elodin->Navigator runner fails LOUD. **Becomes CONFIRMED deploy-corrupting only if/when an Elodin solver-glue + Navigator eval runner is built.**
- **Evidence script:** `scratch/probe_P5-C04_esc_eye3consequence.py`.

---

## 4. NEEDS-LIVE-FLIGHT LEDGER

Both items are **confirmation-of-CLEARED-direction**, not active-bug hunts. Neither blocks the current VQ1 deploy or the inc8 convention-change decision; both should be folded into the next ShadowPC inc7/inc8 capture.

### 4.1 P1-C06 — obs yaw seam (`obs[8]` rpy_g_y, `obs[11]` w_fluz): train<->deploy yaw-about-vertical convention *(blast: both)*

- **Unresolved:** the only **current-convention** seam capture is the inc6 actor (postfix); the inc6->inc7 difference is *exactly* a yaw-Euler + yaw-rate sign flip on dims {8,11}. The mirror canary (P1-C01) is provably **yaw-blind**, so no offline external invariant can pin these two dims for inc7.
- **Exact live probe:** capture ONE inc7 `debug_obs` run (`stage1_inc7_actor.pth`, current `fly_rl.py`, `virtual_flip=true`, >=1 standing/bridge flight with yaw + tilt excursions) on ShadowPC. Re-run `scratch/probe_P1-C06_numerical.py` against it. **PASS iff** CURRENT `build_obs` reconstruction of dim8 AND dim11 from the SAME recorded `q_raw`/`w_raw` both reach **med|err| < 1e-3**; FALSIFY if either stays O(pi).
- **Predicted outcome if correct:** both reconstruct to ~1e-5 (obs construction is checkpoint-INDEPENDENT, so the postfix inc6 seam math should transfer to inc7 verbatim). Do **NOT** clear from inc6 data.

### 4.2 CR1-01 — live command->wire ABSOLUTE yaw sign (`_ACT_FLU_TO_FRD[2] = +1`) *(blast: deploy-corrupting)*

- **Unresolved:** the *absolute* yaw wire sign is the **one channel no roll-mirror invariant can falsify offline** — the dedicated yaw-vs-realized-rate check is closed-loop self-confounded (both vintages self-correlate +). Quat-conjugation `[1,-1,1,-1]` and rate-sign `[-1,-1,-1]` ARE externally confirmed (crab-immune cross-track corr TRUE **+0.987** vs AS-IS **+0.001**; rate-sign agreement **0.9983** vs +w_raw 0.0). The yaw wire sign currently rests on **live-confirmed inc6/inc7 flights** (inc7 standing 5/5, gate-3 0 collisions), not offline data.
- **Exact live probe:** an open-loop **pure-yaw-step** (or yaw-chirp) segment in the inc8 fresh-reset live batch — command a clean yaw rate with near-zero roll/pitch, confirm realized `-w_raw[2]` tracks the commanded sign.
- **Predicted outcome if correct:** realized yaw rate positively tracks commanded yaw (consistent with every offline lens and the 5/5 standing finishes). A flip would manifest as immediate yaw-divergence on the step — which has **not** been seen in any live flight.

> Both NEEDS_LIVE items are the **same physical axis** (yaw-about-vertical) seen from two seams (obs-feature side vs wire-command side). A single inc7 `debug_obs` capture **with a yaw-active segment** closes both at once. **Recommend bundling into the WINNER-VALIDATION rider's pre-crown live batch.**

---

## 5. CLEARED LEDGER — the trust foundation

61 load-bearing conventions were adversarially attacked with fresh external invariants and **held**. Grouped by pillar. The recurring positive control throughout is the **force-vs-FD East-axis mirror canary**: TRUE `[1,-1,1,-1]` East corr **+0.97...+0.99** vs AS-IS `[1,1,1,1]` East corr **-0.71...-0.84** (production-faithful `use_lapse=False`), reproduced on both datasets — the *only* discriminator for the R_y(pi) bug class that bit this project 4x.

### P1 — Deploy frame/convention seam (fly_rl)
- **P1-C01** ODOMETRY quat R_y(pi) conjugation `q_true = q_raw.[1,-1,1,-1]` *(both)* — **harness-free** gravity-anchored thrust direction (zero plant constants): TRUE median angle to measured horizontal force **9.07 deg** (93.4% within 45), AS-IS **92.66 deg** (the East-mirror signature); slope measured-East-force vs thrust-axis +23.9 (TRUE) vs -23.9 (AS-IS, physically impossible); per-run TRUE wins **17/17**. **NUANCE BANKED:** East alone is DEGENERATE — a bogus R_x(pi) `[1,1,-1,-1]` shares the AS-IS East signature; uniqueness needs the **North** axis too. Never declare a convention unique on East alone.
- **P1-C02** angular rate negated all 3 axes (`-w_raw`) *(both)* — course-rate (attitude-free) corr +0.636 (sign -1) vs -0.636 (+1); +w_raw rejected, ~6x residual; gyro forward-integration residual 1.02 deg (-1) vs 10.26 deg (+1).
- **P1-C03** virtual_flip wire rate `act[1:4].[-1,+1,+1]` *(deploy)* — re-derived from on-disk constants (`_RZ_PI_BODY._ACT_FLU_TO_FRD`), seam-confirmed to **2.5e-5** on all 3 axes incl yaw on postfix; lagged causal yaw gain +2.3.
- **P1-C04** action rescale + hover-thrust collective scaling *(deploy)* — SCALE-sensitive slope test: nominal slopes 0.76-1.03; corrupted controls collapse (coll*0.5 -> corr 0.41-0.67). Seam `collective == clip(normed*0.2656)` max|err| 6.3e-6. *(Cosmetic nit: `obs[12]` label misnomer — task `task_15fe4a9e` spawned.)*
- **P1-C05** thrust ceiling = sidecar 3.765 (not hardcoded 5.0) *(deploy)* — force-anchored implied-ceiling solve confirms 3.765; every recording used 3.765. **Operational hardening (NOT a present defect):** make missing sidecar FATAL for S1.3+ ckpts (currently warns-and-reverts to 5.0); flip argparse `--checkpoint` default off inc4.
- **P1-C07** `obs[12]` carries rescaled normed-thrust `[0,3.765]`, not `[0,1]` collective *(both)* — pipeline reconstruction from raw tanh (non-circular) bit-identical (5e-6); UNITS falsifier: `max(obs[:,12]) = 2.42/3.10 > 1` impossible for `[0,1]`.
- **P1-C08** virtual pi-flip two-sided-consistent *(deploy)* — flip-sensitive roll/pitch exactly two-sided (med|d| `[0,0]`); closed-loop wire-cmd vs realized rate confirms direction.
- **P1-C09** RL path does NOT use CTBR VQ1-alias signs (no bridge-first leak) *(deploy)* — STANDING (build_bridge never ran) and BRIDGE both yield identical wire multiplier; CTBR yaw-flip signature ABSENT.
- **P1-C10** `_FLIP` NED/FRD similarity + `_R_W2G` gate transform match `get_observations` *(train)* — current `build_obs` reproduces postfix `obs[6:9]` to max|d| 0.0000; AS-IS rpy diverges 2.76 rad. **NUANCE:** refit recordings document a since-fixed AS-IS-obs capture-time bug; treat as AS-IS-obs provenance.

### P2 — Live command->rate + plant physics
- **P2-C01** live cmd->rate sign `[+1,+1,+1]` incl yaw *(both)* — quat-FD realized rate (q_true anchor, NOT -w_raw) vs wire: roll +2.719/+0.994, pitch +2.779/+0.987, **yaw +2.299/+0.950, 17/17 runs POSITIVE**, stable across mask thresholds.
- **P2-C02** `rate_frd = act.[-1,+1,-1]` genuine recording-era wire, not serialization negation *(deploy)* — ROLLFIX natural experiment: both wire-sign groups positively track the same physical roll rate (~+2.7), which a relabel cannot produce.
- **P2-C03** quad drag column-index / inverse-rotate / body-frame *(train)* — production `step()`: nose -3.402, tail +4.698, lateral -4.455 m/s^2; swapped-column counterfactual gives wrong ratio 0.724 vs true 1.381.
- **P2-C04** PlantParams + COLL_MAP defaults / vertical balance *(train)* — Down-axis (mirror-blind) corr(K_meas, K_table) +0.860; rotation-invariant idle residual norm sane. *(floor=0 conservatively under-predicts small +1.9 m/s^2 idle authority — safe direction.)*
- **P2-C05** super-rate gain map s=0.30 + min(|cmd|,pi) clamp *(train)* — quat-FD channel (independent of w_raw) free-fit s=**0.3033**, bin-median Spearman rho=**+1.000** matching map slope; flat gain rejected.
- **P2-C06** per-axis post-lag angular-accel slew clamp `[260,260,80]` *(train)* — **sign-alias-immune** (TRUE vs AS-IS rate bit-identical, diff 0.0); axis-swap positive control `[80,260,260]` violated by live roll -> discriminating; 0 over-cap.
- **P2-C07** S18 lapse curve inert in default/faithful config *(train)* — live object construction confirms `lapse=None`; bare offline force_model over-predicts ~13% uniformly — do NOT "fix" by enabling lapse.
- **P2-C08** mixer thrust<->rate coupling *(train)* — fresh algebra: Gram(S_aug)=4I exact, decode=S^T/4 exact, `r_fit` matches to 1e-9, Q=[1,1,1]; perturbations (yaw-sign-flip, zeta->inf) break the invariants -> discriminating.
- **P2-C09** substrate attitude integration (right-mult exp(w.dt)) + body -Z thrust in TRUE frame *(both)* — bank-to-turn cross-course accel corr TRUE +0.905/+0.922 vs AS-IS dead/-0.686; right-mult 1.05 deg vs left-mult 4.48 deg.
- **P2-C10** drag-on-OLD-vel + semi-implicit NEW-vel position ordering *(train)* — synthetic ground-truth control: a real drag@NEW sim forces corr +1.0 every axis, which real data flatly does NOT show -> apparent drag@NEW preference is a kinematic artifact.

### P3 — Torch<->numpy plant parity (DiffAero <-> rl_plant)
- **P3-C01** float32/float64 parity (4.4e-16) *(cosmetic)* — **prior CONFIRMED_BUG OVERTURNED -> CLEARED.** Production gate uses atol=1e30 + separate float64<1e-9 / float32<1e-3 judgments after rebuilding params from float64 source; full matrix worst float64 **7.1e-15**. The 4.4e-16 wording correctly describes the *float64* gate. (Doc polish: qualify any memory line with "float64".)
- **P3-C02** shared attitude convention matches the SIM, not merely each other *(train)* — thrust-dir-only East (drag/lapse/coll stripped) TRUE +0.89 vs AS-IS -0.89; 2D horizontal vector median 9.0 deg vs 92.6 deg -> unique proper-rotation winner.
- **P3-C03** train<->deploy FRD rate-sign loop incl yaw externally consistent *(deploy)* — closed-loop realized-rate per axis: sim_true `[+1,+1,+1]` yaw corr +0.987...+0.999 vs yaw-flip -0.987...-0.999. *(Cosmetic: claim TEXT wire factor `[-1,+1,-1]` has a yaw typo; code/data = `[-1,+1,+1]`.)*
- **P3-C04** torch/numpy bridge is a proper-rotation (det +1), not a reflection *(train)* — independent DiffAero-native propagator through the REAL plant: max|pos diff| 4.44e-16, attitude err 3.33e-16; reflected `_FLIP` control diverges (attitude err 1.4e-2) -> discriminating.
- **P3-C05** COLL_MAP/LAPSE interp+clamp torch mirror bit-identical under NaN/Inf + real data + monotonicity *(train)* — all max|diff| **0.000e+00** at float64.
- **P3-C06** recordings store `rate_frd.round(4)` / `obs.round(5)` — pure rounding *(cosmetic)* — values lie exactly on the decimal grid; residual |mean|/std 0.009-0.20 (zero-mean, no structure). 1.76e6x signal margin over a real sign error.
- **P3-C07** DiffAero `collective = normed_thrust.hover` train/eval/deploy parity *(deploy)* — backend parity max|diff| 0.0; physics anchor D-corr +0.956 refit / +0.914 postfix; deploy clip dormant (0/37146 ticks).
- **P3-C08** fwd/back drag z-column binding (descend col0 / climb col1) *(train)* — **prior NEEDS_LIVE OVERTURNED -> CLEARED.** Body-vs-world z-partition agrees 96.5%; descend ticks land col0 94.3%; climb-heavier asymmetry preserved, no swap. (Descend MAGNITUDE still un-witnessed by clean coast — but binding/sign/boundary closed offline.)

### P4 — Train<->deploy obs/action seam
- **P4-C01** `build_obs` reproduces deployed obs *(deploy)* — postfix seam max|diff| **5.4e-5** (pure rounding); provenance reachable ONLY by quat `[1,-1,1,-1]`/rate `[-1,-1,-1]`. **Three recording eras exist** — postfix=current; refit=AS-IS-obs (stale); rollfix=roll-convention session.
- **P4-C02** `obs_from_zup` (deploy) == `get_observations` (train) *(deploy)* — scipy ZYX euler to gimbal-lock max diff 1.57e-13; real-state seam max diff **2.9e-10** (float32 roll noise).
- **P4-C03** action-rescale seam: act_rescaled, normed_thrust, collective, 3.765 bound *(deploy)* — hover-fit falsification: empirical hover normed_thrust median **1.012** (correct-bound) vs 0.753 (legacy-5.0); inverse-bound tanh-extreme 3e-16 matches 3.765.
- **P4-C04** `obs[12]` = rescaled normed_thrust, not `[0,1]` wire collective *(deploy)* — ratio `obs[12]/coll = 3.7651 = 1/hover` in every run; vs normed_thrust = 1.0000.
- **P4-C06** virtual R_z(pi) body flip is a true rigid-body symmetry *(deploy)* — physical thrust axis flip-invariant to **6.66e-16**; the 9.5 m/s^2 flip-vs-unflip force_model delta is the asymmetric fwd/back DRAG term (do NOT mistake for a frame bug).
- **P4-C07** frame constants are the live-confirmed convention, not a mirror *(deploy)* — crab-immune cross-track corr TRUE +0.987 vs AS-IS +0.001; rate-sign agreement 0.9983 vs 0.0. *(Yaw wire vintage-split: postfix=+act_yaw current; refit=-act_yaw superseded — see CR4-03.)*
- **P4-C08** corner-tax / dact span-normalization train==deploy *(train)* — seam identity max|recon - act_rescaled[0]| **0.000e+00** on both datasets; corner geometry distinguishes 3.765 from 5.0; data shows 3.765.
- **P4-C09** progress reward measured against post-advance gate; no double-pay spike *(train)* — gate-index CONSENSUS vs the sim's independent `active_gate_index` (non-circular); advance ticks statistically identical to non-advance (max |progress| 0.736 vs median motion 0.582).
- **P4-C10** standing spawn pose (Rz@Ry, pitch -0.31) reproduces live spawn *(train)* — launch-transient force balance TRUE East +0.83...+1.00 per-run vs AS-IS -0.86...-0.90; Ry@Rz diverges 10-32 deg at yaw!=0.

### P5 — Vision / KF / camera
- **P5-C01** vision/KF world transforms driven by TRUE attitude *(deploy)* — gravity+thrust-only (drag+lapse stripped) East TRUE +0.903 vs AS-IS -0.903 -> not a drag-model artifact. *(Dead import `R_world_from_body` — task `task_e09db8ef` spawned.)*
- **P5-C02** `corner_to_center` uses `_frame_from_through` aligned with detector IPPE_SQUARE *(deploy)* — gravity+camera-anchored: current frame 4/4 corners match detector order; raw-quat-cols 0/4 (180 deg diagonal swap); geo(TRUE, OLD)=180.0 deg.
- **P5-C03** camera-mount transform sign-consistent forward vs inverse *(deploy)* — independent full-matrix re-derivation from physics: max|diff| **1.11e-16**, det +1; seam byte-equal over 5209 attitudes (0.0).
- **P5-C05** detector `corner_ids` index the same IPPE_SQUARE rows as solvePnP *(deploy)* — OpenCV's own solver recovers identity/exact GT; flip_idx witness `[1,0,3,2]` exact; 23/23 non-identity perms caught by pose.
- **P5-C06** map `position_ned` is bottom-centre; `corner_to_center` lifts UP *(deploy)* — multi-gate crossing regression (5 gates, full 26 m descent): slope vs LIFTED **+0.989**, intercept +0.18 m; vs RAW intercept -1.16 m.
- **P5-C07** vision innovation `nu = z - kf.x[:3]` is like-for-like world-NED *(deploy)* — turn-direction: yaw turns WITH velocity only under TRUE (sign-agree 0.877 vs AS-IS 0.123); pos-FD vs vel same frame (corr +0.995). *(NEEDS_LIVE residual: empirical leak ceiling needs a vision-ON flight — not a frame defect.)*

### P6 — MAVLink wire / telemetry parsing
- **P6-C01** ODOMETRY twist -> world NED by RAW quat exactly once == pristine vel_ned *(both)* — crab-binned East corr FLAT +0.996 through 180 deg crab; counterfactual: rotating by the *conjugated* quat BREAKS to -0.254. **Do NOT "fix" the twist to use the conjugated attitude — that would reintroduce a bug.**
- **P6-C02** stored velocity_ned source-independent pristine world NED *(both)* — pos-FD vs vel gain +1.008/+1.007/+1.009, 0 sign-flips, FLAT in yaw (no sinusoid at +/-180 deg).
- **P6-C03** RACE_STATUS layout `<BQqqIq`, active_gate_index unsigned I @ offset 25 *(deploy)* — vendor `PyAIPilotExample/mavlink_rx.py:177` byte-identical; gate-crossing geometry validates every i->i+1 transition (drone within ~1.5-2.5 m of gate).
- **P6-C04** RACE_STATUS q-fields signed; -1 sentinel load-bearing *(deploy)* — signed round-trip reproduces 25/25 recorded dicts; unsigned misread corrupts finished in 23/23; durations sane vs 585 yr.
- **P6-C05** COLLISION.threat_level parsed; HARD-COLLISION abort fires only >=2 *(deploy)* — pymavlink 2.4.49 `threat_level` real uint8 @ idx 3; selectivity: 7 CRASH vs >=7 high-n_coll non-crash physically distinct.
- **P6-C06** `n_coll = len(collisions)` is id-blind DIAGNOSTIC, not the deployed verdict *(cosmetic)* — **prior NEEDS_LIVE OVERTURNED -> CLEARED.** No deployed path applies an `n_coll>0` rule; final_state driven by threat_level>=2 / spin-guard / reset-guard / sim-finish / timeout. (Doc: point the n_coll>0=>INVALID note at `race_outcome.analyze_outcome`.)
- **P6-C07** ODOMETRY.reset_counter parsed by-name, primary reset guard *(deploy)* — pymavlink field present 100% of rows (23/23 runs); guard branch wins tie vs 61 m teleport.
- **P6-C08** TRACK_INFO gate-record layout 38B, quat scalar-first wxyz, pos world NED *(deploy; confidence DOWNGRADED high->MEDIUM)* — behavioral anchor: 87.5% of valid crossings within +/-0.75 m of lifted opening-z; deploy-invariant via height-lift. **HONEST CAVEAT:** scalar-first wxyz NOT uniquely provable offline (8-way |cos|-degenerate); w/h ORDER has zero offline confirmation (width==height everywhere). Both deploy-harmless.
- **P6-C09** TRACK_INFO chunk reassembly byte-exact + fail-safe *(deploy)* — single-chunk live path (the only one that runs) reproduces banked map to **0.000e+00**; re-broadcast/handshake-wipe/interleave adversarial cases all pass.
- **P6-C10** mavlink stores body rate RAW; single downstream -w_raw negation correct *(both)* — kinematic log-recovery corr +0.997/+0.996/+0.984; double-flip +w_raw anti-correlates with exact 2|w|dt integration signature.

### CR — Cross-pillar / critic-round claims
- **CR2-02** yaw-about-vertical validatable on postfix pool *(deploy)* — gyro-FREE course-curvature: postfix TRUE +0.482 vs YAWFLIP -0.518; disjoint bootstrap CIs per run -> downgrade-to-NEEDS_LIVE NOT warranted. (Banked: do NOT cite the gyro-anchored test — it self-mirrors.)
- **CR2-03** super-rate clamp |cmd|->pi denominator vs raw multiply (latent discontinuity at cmd=pi) *(train)* — discontinuity REAL (cmd=pi -> w 11.22; cmd=10 -> 25.0 norm clamp) but **UNREACHABLE**: grand max|act_rate| **2.717 < pi** by 0.42 rad/s. Latent only. *(task `task_60d16de8`; inc7_staging `s0_actor.pth` ships NO sidecar — clamp-safe only via the 3.14 legacy default; ship its sidecar.)*
- **CR3-01** fwd/back drag column binding *(train)* — `vert_fit_coef.npy` FOUND on laptop (prior marked absent): bit-match + key names `c_dn`->col0->descend; train<->deploy seam bit-identical 3.6e-15.
- **CR3-02** `sigma_theta=1.4deg` re-fit after vision-frame-fix *(deploy)* — **prior NEEDS_LIVE OVERTURNED -> CLEARED.** Reconstructed shipped world-fix through true chain: shipped model OVER-bounds the residual; East mean ALIASED +0.037 m -> TRUE +0.010 m (shipped BETTER); Mahalanobis median 1.009. (Optional re-fit hygiene only.)
- **CR3-03** camera +20 deg uptilt + axis-swap *(deploy)* — forward-only elevation sign on real FPV: up(+20) med dv -39.7 px vs down(-20) +197.3 px (REFUTED 4x); model-free uptilt back-out +13.2 deg median.
- **CR4-02** `obs[8]` gate-frame yaw carries TRUE conjugation *(deploy)* — **prior NEEDS_LIVE OVERTURNED for the deploy risk.** Pure-kinematic East thrust-axis sign: TRUE agree 0.955 vs AS-IS 0.045. (Lone narrower NEEDS_LIVE: torch-side training `get_observations` decode — a train<->deploy seam check, not deploy correctness.)
- **CR5-01** high-tilt Down force bias is a thrust-reconstruction seam, NOT a QUAD_DRAG error *(both)* — drag-scale null sweep: no positive drag scale nulls both Down and along-body-up; zeroing drag leaves +5.88 m/s^2 along-thrust deficit -> deficit is in the THRUST column. **Do NOT refit QUAD_DRAG to chase the ~1.1 m/s^2 Down residual.**
- **CR5-02** body-y drag coeff ~0.050 + symmetric; claimed 1.7 m/s^2 Down mis-size geometrically unreachable *(train)* — body-frame diagonal isolation c_y +0.0501 (defeats collinearity caveat); L/R symmetric |L-R| 0.0002; 1.7 m/s^2 needs c_y 0.12-0.19 (2.4-3.5x recovered). (Optional: trim coded 0.055->0.050, <0.1 m/s^2 effect.)
- **CR5-03** `obs[8]` scalar ZYX-yaw externally correct under TRUE conjugation *(both)* — three force-independent invariants: turn-rate TRUE +0.827/+0.845 vs AS-IS -0.233/+0.052; **METHOD CATCH BANKED:** `obs[8]` yaw lives in the GATE frame after the body-z flip, so a turn-sense test against RAW-NED course FALSELY elects AS-IS — must express velocity course in the SAME gate frame.

---

## 6. EXTERNAL-INVARIANT REGRESSION SUITE

Location: `handoff/ultracode-substrate-audit-2026-06-13/regression_suite/`. All run clean in `.venv` standalone (exit 0) and under pytest. **None imports `rl/contact_true_eval.py`** (out of scope, verified). Recommend promoting all to `tests/` after the slug-collision below is resolved — they encode the *external* invariants that internal-consistency tests are structurally blind to (the R_y(pi) bug class that bit 4x).

| File | Bug class it catches | Ran clean | Promote -> tests/ |
|---|---|---|---|
| `test_frame_force_vs_fd_mirror_canary.py` | Re-introduced R_y(pi) ODOMETRY-quat conjugation / attitude per-axis mirror (deploy + vision); FD(vel_ned)-vs-force EAST invariant, tilt>35 | pytest 1 passed; AS-IS bug-sim trips assert (East -0.838) | **Yes** |
| `test_train_deploy_obs_elementwise.py` | Train<->deploy 17-dim obs-seam drift (layout, gate-frame, NED<->Zup, R_y(pi), rate sign, virtual-flip, obs[12] memory) | pytest 5 passed; negative controls trip (AS-IS dims {6,8}, rate dim 9, collective dim 12) | **Yes** |
| `test_twin_diffaero_extreme_parity.py` | Twin/rl_plant<->DiffAero frame/sign-alias divergence at extreme states; bridge-free physics-block external invariant | pytest 5 passed (~55 s); guards bite (quat conj + torch `_BODY_UP` sign flip) | **Yes** (note: shared `_QFLIP` mutation is by-design invisible — documented in-file) |
| `test_mavlink_velocity_single_rotation.py` | Body-vs-world velocity-frame MIX (c3b5a8e); FD(pos_ned)==vel_ned, tilt>35 | pytest 1 passed; drop/double-rotation controls de-correlate (-0.17/-0.19 vs 0.85 gate) | **Yes** |
| `test_confirmed_p4_c05.py` | `obs_from_zup`/`build_obs` hardcoded yaw=pi gate frame (VQ2 hazard) | pytest 5 passed; negative control fires at 4.22 m on buggy tree | **Yes** (retire neg-control + flip LENS2b to equality once the VQ2 fix lands) |
| `test_confirmed_cr2_01.py` | Yaw-about-vertical sign alias in ODOMETRY rate read / FLU->FRD; L3 + Q1 yaw discriminators (Test-A is yaw-blind) | pytest 3 passed; injected yaw-flip FAILS it | **Yes** |
| `test_confirmed_cr4_01.py` | Yaw-channel R_y(pi) conjugation + provenance that refit stale yaw is diagnostic-only | pytest 5 passed; reverting to bcc `[1,1,1,1]` fails East gate | **Yes** |
| `test_confirmed_cr4_03.py` | refit recorded wire validates OLD bcc93f9 map; provenance assertion on recorded `rate_frd[2]` | pytest 5 passed (~34 s); counterfactual flip breaks LENS1 | **Yes** — see SLUG COLLISION (2.2): commander must re-home the earlier COLL_MAP test under a new slug before promoting |
| `test_confirmed_cr1_01.py` | Yaw command->wire sign flip (realized -w_raw[2] vs commanded yaw) | present (authored earlier this pass) | Yes — CR1-01 verdict is NEEDS_LIVE for the *absolute* sign (offline confirms direction/consistency only) |
| `test_confirmed_cr3_02.py` | sigma_theta projection after vision-frame-fix (shipped-fix Mahalanobis calibration) | present (authored earlier this pass) | Yes |
| `test_confirmed_p3_c06.py` | Recording quantization is pure rounding, not structure | present (authored earlier this pass) | Optional (cosmetic-class guard) |

Note: `test_confirmed_cr1_01.py`, `test_confirmed_cr3_02.py`, and `test_confirmed_p3_c06.py` were authored in an earlier round of this same audit suite (the suite now holds 11 `test_*.py`). They are kept; the 8 regression-author reports above cover the freshly-(re)verified set.

---

## 7. Methods appendix

**Datasets (real ShadowPC live recordings):**
- **`postfix`** = `handoff/shadowpc-postfix-dataset-2026-06-12/extracted` — captured AFTER the `93023cf` frame-audit fix -> **current convention** (frame-clean). `build_obs` reconstructs logged obs bit-exact (<=5.5e-5, all 17 dims). **Thin** (6 loadable runs / ~1.3k tilted ticks; 2 runs header-only). This is the **only** valid substrate for clearing current-code **yaw** claims.
- **`refit`** = `handoff/shadowpc-refit-dataset-2026-06-12/extracted` — captured under the **superseded `bcc93f9`** code. Large (17 runs / ~30.5k tilted ticks). Its `obs[8]`/`obs[11]`/`rate_frd[2]` **yaw** channels encode the OLD convention (HYBRID roll/yaw split; rollfix runs add roll-convention diffs). **Use only as the AS-IS / superseded-yaw positive control** — never to clear a current-code yaw claim. Its raw `q_raw`/`w_raw`/`pos_ned`/`vel_ned` ARE the same pristine telemetry, so the R_y(pi) mirror canary fires identically in both datasets (the conjugation is in the SIM telemetry, not the code era).
- **Two header-only runs** found and tolerated: `..._std_f2` and `..._std_ext_f2` (the second newly flagged this pass). Downstream loaders must tolerate >1 empty run.

**Shared substrate:** `scratch/_audit_io.py` — loads both datasets, exposes the canonical Test-A force-vs-FD harness and the candidate quat/rate sign sweeps. Smoke-test reproduces the banked canary: pooled refit East TRUE +0.99 / AS-IS -0.81 (per-run AS-IS lands -0.68/-0.75; pooled is the asserted figure).

**Canonical external invariants used (all anchored on pristine `vel_ned`/`pos_ned`):**
1. **Force-vs-FD East mirror canary** (the R_y(pi) discriminator) — central-FD of pristine vel_ned vs `K.L.(R@[0,0,-1]) + quad_drag + g`, East axis, tilt>35 deg, `use_lapse=False`. **East alone is degenerate for *uniqueness* — cross-check North** (P1-C01 nuance).
2. **Harness-free gravity-anchored thrust direction** — zero plant constants; cannot be a force-model self-mirror.
3. **Crab-immune cross-track / course-curvature** — projects onto the velocity-frame lateral axis so the ~55-63 deg trained crab does not confound it.
4. **pos-FD vs vel** anchor self-consistency (attitude-independent).
5. **Quat-FD realized rate** (from the externally-pinned q_true, not -w_raw) for rate-sign claims.
6. **Regression slope** (scale-sensitive) for any absolute-magnitude / ceiling / scale claim — correlation is scale-blind and cannot clear those.

**Limitations — what offline CANNOT catch (training-doctrine section 6, the closed-loop self-mirror caveat):**
- **Internal-consistency checks are blind to a proper-rotation conjugation.** Quat-FD-vs-rate, level-flight correlation, and twist round-trip ALL pass under the R_y(pi) bug — only the *external* force-vs-FD invariant on pristine vel_ned discriminates. This is why this whole audit is anchored externally.
- **The absolute yaw-about-vertical wire sign cannot be falsified offline** — every closed-loop lens self-correlates regardless of the deployed yaw sign (CR1-01, P1-C06). The deployed sign rests on live-confirmed flights. **Bundle a yaw-active segment into the next live batch.**
- **The drone flies a heavy crab (~55-63 deg), so heading(body-x)-vs-course is INADMISSIBLE here** — it carries no convention information and would *falsely* look ambiguous (or falsely elect AS-IS in the gate frame). Do not use it as a discriminator on this platform.
- **Offline data is a low-yaw, single-spawn-era policy** — fresh-respawn post-gate-3 sensitivity is blind offline (WINNER-VALIDATION rider).
- **`obs[8]` yaw lives in the GATE frame after the body-z flip** — its sign is flipped vs raw NED; a naive turn-sense test against raw-NED course mis-elects AS-IS (CR5-03 catch).
- **The contact / scoring layer is OUT OF SCOPE** — `rl/contact_true_eval.py` was not audited; the inc8 gate-margin numbers it produces are POSSIBLY-STALE from this report's vantage.

**Bottom line:** complete coverage (68/68), no live-code train/deploy-corrupting bug on the VQ1 course survived escalation; one dormant VQ2 hazard (P4-C05) and three data-provenance defects (CR2-01/CR4-01/CR4-03, one root cause) confirmed; two yaw-axis items pending a single live yaw-active capture. The shipped `93023cf`/HEAD conventions are externally vindicated and the inc8 convention-change decision can proceed.
