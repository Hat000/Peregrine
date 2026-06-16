# inc8 conversion-fix re-pilot — WORKER REPORT

**Session:** P2 INC8-RL — Worker (conversion-fix re-pilot)
**Branch:** `worker/inc8-repilot-window-2026-06-15` @ `bde31dc` (pushed to origin)
**Base:** `main` @ `a3a9960`
**Date:** 2026-06-15

---

## TL;DR

**PART 1 (laptop prep) — DONE & PUSHED.** Reward-window realignment baked into a new sbatch +
`inc8_lockband_pointing` diagnostic metric wired in. Full suite **814 passed / 42 skipped / 0 failed**;
inc8-OFF byte-identity AST guard green.

**PART 2 (Adroit run) — TWO runs EXECUTED. FINAL VERDICT = NO-GO: the realigned window is INSUFFICIENT
(reward-SHAPE problem isolated).** (1) Job 3273552 @ entropy 0.02 ran away (never converged) — my override
overcorrected; its "+10× fix-rate" was a max-random artifact. (2) Job 3273701 @ **default entropy 0.01**
CONVERGED cleanly (entropy mid-range, value_loss→1.6) but the converged policy points **only terminally**
(<5 m), **lockband_pointing=0 in the 18–28 m fixable band**, fix_rate≈0. Cause: `d_lock=18` makes `w_term`
a low-pass plateau (=1.0 for ALL ranges ≤18 m incl. the easy terminal zone), so the gradient favours
terminal pointing over the fixable band. (3) Job 3274167 with the **BAND-PASS** `w_term` (+surrogate
accept_rlo→12) KILLED the terminal exploit (terminal_pointing 0.035→0.005) but the policy then
**abandoned pointing entirely** (pointing_rate→~0.002, lockband≈0, fix≈0) — band pointing is achievable
(step-200 lockband=0.011) but NOT WORTH the racing-line cost at R5′~5% of progress. (4) Job 3274179 with the FIX-DRIVEN fallback (conf_shape 0.05→0.3 + new direct
fix_bonus=1.5; c_inplane verified HONEST, not #74-gamed) was a **decisive NO-GO** — pointing_rate=**0**
all 4000 updates. The new metrics prove WHY no weight works: conf_shape is a constant −0.29/step penalty
(c_inplane stuck 0.027, age_norm stuck 1.0) with no incremental gradient; fix_bonus≈0 (never fires,
fix_rate≈0). **STOP per the task: 3 iterations bracket it → this is a sparse-reward / exploration /
credit-assignment problem, NOT a weight problem. NEXT = reward ARCHITECTURE (direct terminal-σ_p0
centering reward, pointing curriculum, or demo-init/BC into the fixable basin) — flagged for the
commander, NOT launched.** Full results §PART 2b/2c/2d. (Duo: resolved by Fengyou's `serve`; I
force-killed stale shadow daemons PID 50232 then 57056 on port 8765.)

---

## Verified facts (escape-hatch check — ALL consistent with the code, no contradictions)

| Claim in prompt | Verified in code |
|---|---|
| Field names `perc_d_lock_m` / `perc_d_acq_m` | `rl/inc8_reward.py:49-50` (`Inc8RewardWeights`) ✓ |
| cfg loop reads `getattr(cfg, "rw_"+field)` → `rw_` prefix correct | `rl/peregrine_racing_inc8.py:53-59` (`_inc8_weights_from_cfg`) ✓ |
| Value casts float fine | `type(f.default)(...)` with `f.default=5.0` → `float(18)=18.0` ✓ |
| `self._inc8w` available at the metric site | set `rl/peregrine_racing_inc8.py:89`; metric site is in `step()` after init ✓ |
| Line 318 `inc8_terminal_pointing@5m` = fixed deploy KPI (uses fresh `Inc8RewardWeights()`) | confirmed; **left untouched** ✓ |
| Default `perc_d_lock_m=5`, `perc_d_acq_m=24`; override → 18/28 | matches ✓ |

w_term math sanity (`terminal_weight`, `inc8_reward.py:100-104`): with `d_lock=18, d_acq=28, w0=0.15`:
`w_term(22) = 0.15 + 0.85·clip((28−22)/(28−18),0,1) = 0.15 + 0.85·0.6 = 0.66`; `w_term(≤18) = 1.0`.
Matches the prompt ("w_term≈0.66 at 22 m and 1.0 by 18 m"). The realigned reward now peaks IN the
16–28 m fixable band instead of the <5 m p_accept dead zone.

---

## PART 1 — what changed (3 files, all on the branch)

### 1. `rl/peregrine_inc8_repilot.sbatch` (NEW — copy of `peregrine_inc8_smoke.sbatch`)
- **Realigning overrides** added to `COMMON`: `+env.rw_perc_d_lock_m=18 +env.rw_perc_d_acq_m=28`.
- **`NUPD` default `1000 → 4000`** — must run well past pilot #2's step-~2300 entropy collapse so we
  can see whether `inc8_fix_rate` climbs AND whether exploration survived.
- **`--time` = `04:00:00`** — calibrated from pilot #2 (`handoff/inc8-pilotA-2026-06-14/REPORT.md`):
  6000 updates ran in **~59 min** wall @ ~28% util ⇒ 4000 updates ≈ **~40 min**. 4 h is ~5–6× margin
  (precheck + startup/export + queue/IO slack). **Not** the wildly-over-budgeted 14 h pilot #2 used.
- **`${EXTRA:-}` passthrough** appended to BOTH train invocations (precheck + the pilot) so a submit-time
  entropy override (PART 2) injects without editing the file. Precheck also carries EXTRA → fail-fast on a
  bad key.
- `runname=inc8_repilot_A_seed${SEED}`; precheck `runname=inc8_repilot_precheck`; GPU-util log →
  `peregrine_inc8_repilot_gpu_util.log`; TB-trace glob → `*inc8_repilot_A_seed${SEED}__*`.

### 2. `rl/peregrine_racing_inc8.py` — NEW logging-only metric `inc8_lockband_pointing`
- = in-image rate over the **CONFIGURED** band `[self._inc8w.perc_d_lock_m, self._inc8w.perc_d_acq_m]`
  (i.e. tracks the realigned 18–28 m window, unlike the deploy KPI).
- Uses `self._inc8w` (NOT a fresh `Inc8RewardWeights()`), per the prompt.
- Sync-free masked mean, **identical pattern** to `term_pointing` (0.0 when the band is empty:
  `0/clamp(0,min=1)`). Appended to `metric_vec` / readback tuple / `loss_components` — the single
  `.tolist()` host-sync is preserved (one extra stacked scalar, no new sync).
- **Line 318 untouched** (deliberate `inc8_terminal_pointing@5m` deploy KPI, kept for cross-run
  continuity with pilot #2).
- Entirely inside the inc8 path (after the `if not self._inc8_on: return super().step(...)` guard at
  `:199`) ⇒ **inc8-OFF byte-identical**.

### 3. `rl/inc8_tb_trace.py` — added the `lockband_pointing` trace column
- `("lockband_pointing", ["inc8_lockband_pointing", "lockband_pointing"])` after `terminal_pointing`.
  Exact-suffix match is tried first, so no shadowing by `inc8_pointing_rate`.

### Tests
- Full suite (worktree code, main venv): **814 passed, 42 skipped, 0 failed** (~190 s). Skips are the
  diffaero/GPU-only tests (expected on laptop).
- `tests/test_inc8_off_identity.py` (AST OFF-identity guard) green — structural byte-identity intact.
- `py_compile` of both edited modules: OK.

---

## PART 2 — Adroit run (NEEDS ONE DUO; copy-paste for Fengyou)

> **Why blocked:** the worker cannot approve Duo. `serve` daemon is stale; a fresh `serve` prompts for
> the key passphrase (unset in `.env`) then a Duo push — both on Fengyou's device.

Run from `C:\Users\Fengy\Downloads\Projects\Adroit\adroit-connector` in PowerShell. **Phase A** discovers
the entropy schedule (needed to decide the override + to attribute the result); **Phase B** submits.

### Phase 0 — start the session (Terminal 1, leave running — ONE Duo)
```powershell
cd C:\Users\Fengy\Downloads\Projects\Adroit\adroit-connector
.venv\Scripts\python.exe adroit.py serve
# enter key passphrase, approve the Duo push; leave this terminal open
```

### Phase A — git pull + DUMP the diffaero PPO entropy config (Terminal 2)
```powershell
cd C:\Users\Fengy\Downloads\Projects\Adroit\adroit-connector
.venv\Scripts\python.exe adroit.py x "set -e; cd /scratch/network/fl3689/peregrine_repo && git fetch origin && git checkout worker/inc8-repilot-window-2026-06-15 && git reset --hard origin/worker/inc8-repilot-window-2026-06-15 && git log --oneline -1; echo '=== sbatch present? ==='; ls -l rl/peregrine_inc8_repilot.sbatch; echo '=== diffaero PPO/entropy config ==='; cd /scratch/network/fl3689/diffaero_repo && echo '-- algo config files --' && find . -path '*algo*' -name '*.yaml' 2>/dev/null && echo '-- entropy/ent_coef in cfg tree --' && grep -rin 'entropy\|ent_coef\|ent_coeff' cfg configs config 2>/dev/null && echo '-- entropy/anneal/schedule in PPO impl --' && grep -rin 'entropy\|ent_coef\|anneal\|schedule\|coef' --include='*.py' . 2>/dev/null | grep -i 'entropy\|ent_coef' | head -40"
```
**Paste the output back to the commander / this session.** I need: the exact entropy key (e.g.
`algo.entropy_coef`), its default value, and whether it has a decay/anneal schedule + how fast.

### Phase B — submit the re-pilot
**Decision rule (prompt §2b):**
- Pilot #2 evidence: entropy went **negative at step ~2300** (`inc8-pilotA` report) — strong prior that
  the schedule is **not** gentle. If Phase A confirms decay-to-near-deterministic before ~3000 updates,
  **raise the floor / slow the decay** so exploration survives into the 18–28 m window where the
  realigned reward is now discoverable.
- If Phase A shows the schedule is **already gentle**, submit with **no** override and say so.

**B-default (no entropy override — submit if the schedule is gentle):**
```powershell
.venv\Scripts\python.exe adroit.py x "export SEED=0; sbatch --export=ALL /scratch/network/fl3689/peregrine_repo/rl/peregrine_inc8_repilot.sbatch"
```

**B-with-override (fill `<KEY>`/`<VALUE>` from Phase A — DO NOT invent the key):**
```powershell
.venv\Scripts\python.exe adroit.py x "export SEED=0 EXTRA='<KEY>=<VALUE>'; sbatch --export=ALL /scratch/network/fl3689/peregrine_repo/rl/peregrine_inc8_repilot.sbatch"
```
- Hydra form: use plain `<KEY>=<VALUE>` if the key already exists in `ppo.yaml`; use `+<KEY>=<VALUE>`
  if it is a NEW key (Hydra errors on `+` for an existing key, and errors on a plain set of a missing
  key — that is why we must see the config first).
- Override INTENT (key TBD from Phase A): raise the entropy coefficient floor and/or lengthen its
  anneal horizon so entropy stays > 0 past ~3000 updates. E.g. if the key is a constant
  `algo.entropy_coef`, try ~1.5–2× the default; if it is an anneal end-step, push it past 4000.

### Watch
```powershell
.venv\Scripts\python.exe adroit.py x "squeue -u fl3689"
# when done, the .out has the GPU-util summary + the TB trajectory (every ~100 updates):
.venv\Scripts\python.exe adroit.py x "tail -n 120 /scratch/network/fl3689/peregrine_inc8_repilot.out"
.venv\Scripts\python.exe adroit.py x "checkquota"
```

---

## VERDICT / ACCEPTANCE (reminder — this is a PILOT, judge the TREND)

- **PRIMARY GO/NO-GO:** does `inc8_fix_rate` climb off the ~0.001 floor and trend upward?
- **SUPPORT:** `inc8_lockband_pointing` rises (camera on-gate in the fixable band); entropy did not
  collapse to deterministic before the policy found the band; `total_reward` rising.
- **GO** → fix_rate climbing + lockband pointing rising ⇒ bake the realigned window
  (`rw_perc_d_lock_m=18`, `rw_perc_d_acq_m=28`) into `COMMON` and proceed to the L0 arm portfolio
  (A×5 / B×3 / C×1).
- **NO-GO** → fix_rate still floored. Localize with `inc8_lockband_pointing`:
  - pointing-in-band HIGH but fix_rate ~0 ⇒ the surrogate `p_accept` band is the wall (deeper problem,
    escalate).
  - pointing-in-band LOW ⇒ pointing still not learned (entropy / weight magnitude).

---

---

## PART 2 — EXECUTED (RESULTS)

### Setup / mechanics
- **Duo:** Fengyou started `serve`; a **duplicate stale daemon (PID 50232, ~11 h old)** was shadowing
  the fresh one (PID 57056) on `127.0.0.1:8765` via SO_REUSEADDR → `x` got `unauthorized`/`ConnectionReset`.
  Force-killed 50232 (so its exit handler wouldn't wipe the fresh `.daemon.json`); `x` then worked with
  no new Duo.
- **File transfer:** `peregrine_repo` on Adroit is a **non-git synced copy** (no `.git`) and the repo is
  private (raw GitHub 404, no `gh`/netrc on Adroit), so SFTP `upload` (interactive passphrase) was out.
  Transferred all 3 files over the live daemon via base64 (`x` only) — `peregrine_racing_inc8.py` chunked
  (b64 > Windows cmdline limit). **All md5-verified byte-exact** (`672273…` etc.; caught + fixed a
  `while read` last-partial-chunk truncation on the first attempt).
- **Entropy config discovered:** `cfg/algo/ppo.yaml → entropy_weight: 0.01`, consumed as a **constant**
  in `algo/PPO.py:38,124,302` (`loss = pg_loss + value_weight·v_loss + entropy_weight·entropy_loss`).
  **NO annealing/decay/schedule anywhere.** Pilot #2 collapsed at ~2300 with 0.01, so per the prompt's
  rule I raised the floor → `EXTRA="algo.entropy_weight=0.02"` (plain form, existing key; Hydra accepted,
  no error). gpu partition MaxTime = **7 days** (parked `--time` concern resolved).
- **Precheck PASSED** (`PRECHECK_RC=0`, obs_dim 20, no NaN; `[onnx-guard]` = known cosmetic export skip).
- **New `inc8_lockband_pointing` metric resolved end-to-end** (`env_loss/inc8_lockband_pointing` in TB).

### Trajectory (job 3273552, entropy_weight=0.02, window 18–28 m, 4000 updates)
```
 step  total_reward  pointing_rate  terminal_pointing  lockband_pointing   fix_rate   entropy_loss(=-H)
    0     -0.32463        0.00000          0.00000           0.00000        0.00000        0.33879
  400     -0.95130        0.01660          0.01552           0.00820        0.00342       -9.15983
  800     -1.23263        0.02686          0.02391           0.01579        0.00439       -9.69753
 1200     -1.14141        0.03418          0.03148           0.01473        0.00635      -11.02903
 1600     -1.26490        0.03516          0.00943           0.02769        0.01025      -12.41541
 2000     -1.00356        0.02539          0.01114           0.02787        0.00977      -13.66267
 2400     -0.79104        0.03564          0.01322           0.03349        0.01172      -13.67574  <- entropy maxes
 2800     -1.02116        0.02246          0.00442           0.01318        0.00684      -13.67575  <- FROZEN (max-H)
 3200     -1.13441        0.02197          0.00240           0.01812        0.00537      -13.67575
 3600     -1.00793        0.02783          0.01299           0.03533        0.01270      -13.67575
 3990     -1.13802        0.02930          0.00995           0.02742        0.00781      -13.67575
```
GPU util mean 28.5% / max 29% (same as pilot #2 — V100, numpy-boundary; not a regression).

### Verdict & attribution
- **PRIMARY (fix_rate):** climbed ~10× off pilot #2's ~0.00098 floor to a **peak ~0.012 at step 2400**,
  then **regressed/plateaued** (~0.006–0.013). Moved, but **no sustained upward trend**; still ~40× short
  of the ≥0.50 deploy gate.
- **CONFOUND — entropy ran AWAY, not collapsed.** `entropy_loss` froze at **−13.67575 from step 2400 on**
  = policy std hit its ceiling = **maximally random, never converged**. My 0.02 (chosen vs pilot #2's
  0.01 collapse) **overcorrected**: pilot #2 = converged-but-mis-aimed (deterministic point-blank);
  this = realigned-but-never-converged. The fix_rate peak coincides exactly with entropy saturating →
  after that it's just a random policy's baseline gate-crossing rate.
- **Localization (acceptance rule): lockband_pointing LOW (~0.01–0.035), not high ⇒ "pointing still not
  learned (entropy / weight magnitude)" — NOT the surrogate `p_accept` wall** (the escalate-worthy branch
  is ruled out for now).
- **Net:** realigned reward window is **directionally validated** (10× lift; lockband_pointing and
  fix_rate peak together), but neither pilot has produced a *converged* policy under it. The entropy knob
  overshot from collapse (0.01) → runaway (0.02); the converging regime is in **(0.01, 0.02)**.

### Recommendation (Fengyou chose HOLD → commander triages)
- One more ~70-min run, realigned reward + **entropy_weight ≈ 0.012** (just above the 0.01 that collapsed,
  well below the 0.02 that ran away) — most likely to yield a CONVERGED policy that actually tests whether
  the realigned reward makes fix_rate climb. Alternative: clean 0.01 to isolate the reward fix (risks
  re-collapse). Commander may instead fold this into an entropy sweep / vary perc weight / more updates.
- **Do NOT bake the window into COMMON yet** — the GO criterion (sustained fix_rate climb) is unmet; the
  window is *promising*, not *proven*.

### Artifacts (on Adroit, /scratch/network/fl3689/)
- `peregrine_inc8_repilot_ent0.02.out` (full log, preserved — the bare `peregrine_inc8_repilot.out` will
  be overwritten by any resubmit), `peregrine_inc8_repilot_ent0.02_gpu_util.log`.
- TB run dir: `diffaero/outputs/train/2026-06-15/12-17-15/quad__racing__ppo__mlp__inc8_repilot_A_seed0__0/`.

---

## PART 2b — CONVERGENCE RUN (entropy 0.01) — the clean test

**Job 3273701** — realigned window 18–28 m, **default entropy_weight=0.01 (no EXTRA)**, RUNTAG=ent0.01,
4000 updates. Clean exit (REPILOT_RC=0, V100, util 28.2%). The sbatch gained a **RUNTAG passthrough**
(commit c0a3369; md5-verified byte-exact on Adroit, remote `bash -n` clean) so the `.out`/runname/
gpu-util/TB-glob are tag-scoped — this run did NOT clobber the ent0.02 artifacts. Precheck PASSED;
confirmed 0 occurrences of `algo.entropy_weight` (true default 0.01).

### Trajectory
```
 step  total_reward  pointing_rate  terminal_pointing(<=5m)  lockband_pointing(18-28m)  fix_rate  entropy_loss(-H)  value_loss
  200    -0.83745       0.00000           0.00000                   0.00000              0.00000      1.97578          5.44694
 1000    -0.60877       0.00098           0.00232                   0.00000              0.00049     -2.94291         13.25109
 2400    -0.54768       0.00781           0.01902                   0.00000              0.00000     -2.41341          9.79961
 3000    -0.51937       0.01221           0.03141                   0.00000              0.00000     -1.87353          8.95156
 3800    -0.52125       0.01416           0.03524                   0.00000              0.00000     -1.26609          2.32878
 3990    -0.49891       0.00732           0.01796                   0.00000              0.00000     -1.47504          1.55032
```

### VERDICT = NO-GO (reward-shape problem isolated; OVERTURNS the ent0.02 "validation")
- **Entropy CONVERGED ✓** — entropy_loss settled mid-range (~−1.3 to −2.7, H≈1.3–3.4; not the −13.676
  ceiling, not collapsed); value_loss fell to ~1.6. **0.01 fixed the runaway** — the 0.02 result was an
  entropy overcorrection, as hypothesised.
- **lockband_pointing = 0.000 throughout ✗.** The converged policy points **only terminally**
  (terminal_pointing@≤5 m rose 0→0.035) and **NEVER in the 18–28 m fixable band** → fix_rate ≈ 0.
- **The ent0.02 "+10× fix-rate / lockband tracks it" was an artifact of the MAX-RANDOM policy** sweeping
  the band by chance. The converged policy shows optimization drives pointing **AWAY** from the band
  toward the easy terminal zone. Directional-validation read does NOT survive the clean convergence test.

### Root cause
`d_lock=18` makes `w_term` a **low-pass plateau = 1.0 for ALL ranges ≤18 m**, incl. the <5 m terminal
zone where the gate fills the frame and in-image is nearly free; the 18–28 m band gets a *lower* ramped
weight (1.0→0.15). So the reward gradient favours the EASIEST full-reward pointing (terminal), not the
fixable band. The realigned window reshaped the upper edge but left the lower side open → exploited.

### Localization & recommendation (commander's call — NOT launched)
- pointing IS learned (terminal_pointing climbs) but in the WRONG zone ⇒ **reward-SHAPE** problem, NOT
  the surrogate `p_accept` wall and NOT entropy (now healthy).
- **Do NOT bake `d_lock=18/d_acq=28` into COMMON.**
- Reshape `terminal_weight` (`rl/inc8_reward.py`) from a low-pass plateau into a **BAND-PASS peaked in
  ~16–28 m, suppressed below ~16 m**, so terminal pointing is no longer equally rewarded and the gradient
  points the policy where the surrogate accepts fixes. Optionally strengthen the GT-anchor/confidence
  terms so fixes (not raw in-image) drive R5'. Then re-pilot at entropy 0.01.

### Artifacts (Adroit, /scratch/network/fl3689/)
- ent0.01: `peregrine_inc8_repilot_ent0.01.out`, `peregrine_inc8_repilot_ent0.01_gpu_util.log`;
  TB dir `diffaero/outputs/train/2026-06-15/15-06-57/...inc8_repilot_A_seed0_ent0.01__0/`.
- ent0.02 (prior): `peregrine_inc8_repilot_ent0.02.{out,_gpu_util.log}`; TB `2026-06-15/12-17-15/`.

---

## PART 2c — BAND-PASS reward reshape (job 3274167) — the SHAPE fix, isolated

**Changes** (commit ecd82d4; full suite 814/42/0, OFF byte-identical held): `terminal_weight` reshaped
from a low-pass plateau to a **BAND-PASS** `w0+(1-w0)*sig((r-r_lo)/w_lo)*sig((r_hi-r)/w_hi)` with
`r_lo=12,r_hi=28,w_lo=w_hi=1.5,w0=0.05` (`Inc8RewardWeights` perc_r_lo/r_hi/w_lo/w_hi/w0, cfg path kept);
paired surrogate **accept_rlo 16.191->12.0** (torch+numpy, parity held); `inc8_lockband_pointing` band ->
[12,28]; @5m KPI decoupled to a literal. RUNTAG=bandpass_v1, default entropy 0.01.

### Trajectory (clean exit RC=0, V100)
```
 step  pointing_rate  terminal_pointing(<=5m)  lockband_pointing(12-28m)  fix_rate  entropy_loss(-H)
  200     0.01172          0.01422                  0.01096               0.00391      -1.76563
  800     0.00146          0.00483                  0.00000               0.00000      -6.79317
 1600     0.00146          0.00360                  0.00000               0.00098      -6.48052
 2800     0.00244          0.00859                  0.00000               0.00049      -5.70633
 3990     0.00146          0.00495                  0.00000               0.00000      -5.92727
```

### VERDICT = NO-GO -> the FALLBACK is now indicated
- **The band-pass DID its job:** killed the terminal exploit -- terminal_pointing dropped 0.035 (low-pass
  run 3273701) -> ~0.005 here. The policy no longer point-blanks for free reward.
- **But it abandoned pointing entirely:** pointing_rate 0.014 -> ~0.002, lockband_pointing ~= 0, fix_rate
  ~= 0. TELL: step 200 lockband=0.011 then decays to 0 -- early exploration DOES band-point, the policy
  LEARNS AWAY from it. Band pointing is achievable but NOT WORTH it (R5' ~5% of progress; earning it in
  the band costs racing-line). With the easy terminal reward removed, cheapest policy = fly the line,
  don't point.
- **Two runs bracket the problem:** low-pass -> points easy terminal (exploit), lockband=0; band-pass ->
  points nowhere (incentive too weak), lockband~=0. SHAPE change alone (now isolated) is insufficient.

### Recommendation (the task's FALLBACK -- not launched)
Make ACTUAL FIXES drive the reward, not raw in-image (which is a weak, gameable proxy):
- Strengthen **GT-anchor** (`rw_estimerr` 2.0) and/or **confidence shaping** (`rw_conf_shape` 0.05) so low
  KF error / fresh fixes earn real reward -- only obtainable by getting fixes in the band, un-gameable by
  terminal pointing.
- And/or raise **`rw_perc`** (0.5) so the band-pass R5' is worth the racing-line tradeoff.
- KEEP the band-pass shape (it correctly removed the terminal exploit; surrogate accept_rlo=12 stays).
- Re-pilot @ entropy 0.01, RUNTAG e.g. `fixdriven_v1`. GO = lockband_pointing[12,28] + fix_rate CLIMBING.

### Artifacts (Adroit): `peregrine_inc8_repilot_bandpass_v1.{out,_gpu_util.log}`;
TB `diffaero/outputs/train/2026-06-15/20-39-36/...inc8_repilot_A_seed0_bandpass_v1__0/`.

---

## PART 2d — FIX-DRIVEN reward fallback (job 3274179) — NO-GO → STOP (architecture, not weights)

**Iteration 4 (the last weight-tuning iteration).** VERIFIED FIRST (handoff/check_c_inplane.py): emulator
KF c_inplane is HONEST (~0.05 cold no-fix coast, decays to 0.005; transient ~0.24 after one fix) -> the
#74 over-convergence is NOT present, NO covariance floor needed, anneal already always-on. Changes
(commit 2f68224, suite 815/42/0, OFF held): `rw_conf_shape` 0.05->0.3 + NEW `rw_fix_bonus`=1.5
(`fix_bonus_reward` = rw_fix_bonus*accepted*(delta_s>0), progress-gated, default 0.0 byte-identical).
KEPT the band-pass shape.

### Trajectory (clean RC=0, V100, util 28.7%)
```
 step  pointing_rate  lockband(12-28)  fix_rate  fix_bonus  c_inplane  age_norm  total_reward  entropy_loss(-H)
  200     0.00000        0.00000        0.00000    0.00000    0.02737    0.9992     -1.11543       +2.03563
 1600     0.00000        0.00000        0.00098    0.00000    0.02743    0.9997     -1.05375       +1.13883
 2800     0.00000        0.00000        0.00049    0.00073    0.02702    0.9998     -1.12202       +0.35248
 3990     0.00000        0.00000        0.00000    0.00000    0.02719    0.9993     -1.07925       +0.60995
```

### VERDICT = NO-GO (decisive). The strong fix reward made pointing WORSE.
- `pointing_rate` = **EXACTLY 0** all 4000 updates (band-pass v1 had sporadic ~0.002). lockband=0, fix=0.
- **WHY no weight can fix it** (the new metrics prove it):
  - c_inplane stuck 0.027, age_norm stuck 1.000 -> conf_shape = 0.3*(0.027-1.0) ~= **-0.29/step CONSTANT
    penalty**, inescapable by *incremental* pointing (only a real accepted fix escapes it). Flat = NO GRADIENT.
  - fix_bonus ~= 0.0000 -> correctly wired but NEVER FIRES (fix_rate~0). Sparse binary = NO GRADIENT.
    Raising 1.5->50 changes nothing (50*0=0).
  - Policy converged deterministic (entropy_loss->+0.5 peaked, value_loss~1) -> just race + eat the penalty.

### Three iterations BRACKET it -> NOT a weight problem
1. low-pass -> exploits terminal pointing (lockband=0); 2. band-pass -> abandons pointing (too weak vs
racing line); 3. fix-driven -> points NOWHERE (fix signal sparse/flat -> no gradient to discover pointing).
**Sparse-reward / exploration / credit-assignment failure**: pointing->fix is hard + all-or-nothing; PPO
can't discover it from a non-pointing start at ANY weight.

### STOP per the task -> ARCHITECTURE change (commander's call; NOT launched, no more weight sweeps)
- **Direct terminal-sigma_p0 centering reward**: reward the DENSE, incrementally-improvable OUTCOME the
  fixes are for (terminal centering / low sigma_p0, the gate-4 closure metric), not the sparse fix event.
- **Pointing curriculum**: start gate-in-frame easy (closer/slower/eased geometry), harden gradually.
- **Demo-init / BC** from a pointing demo (P3 head-on: 67-77% fix when pointed) -> start INSIDE the
  fixable basin so the fix-driven reward gains gradient.
- KEEP the band-pass shape + surrogate accept_rlo=12 (both correct); the missing piece is bootstrapping
  the policy into the pointing basin / a dense centering proxy.

### Artifacts (Adroit): `peregrine_inc8_repilot_fixdriven_v1.{out,_gpu_util.log}`;
TB `diffaero/outputs/train/2026-06-15/22-26-40/...inc8_repilot_A_seed0_fixdriven_v1__0/`.

---

## PARKED / surfaced for the commander
- **BAND-PASS `w_term` reshape + re-pilot** (the real next step; see §PART 2b). The realigned low-pass
  window is insufficient — a *converged* policy points terminally and gets 0 fixes. Reshape
  `rl/inc8_reward.py:terminal_weight` to peak in ~16–28 m and suppress <~16 m (and/or strengthen the
  GT-anchor/confidence terms so fixes, not raw in-image, drive R5'), then re-pilot at entropy 0.01.
- **Convergence-entropy question RESOLVED:** default 0.01 converges cleanly under the realigned reward
  (entropy mid-range, value_loss→1.6); the 0.02 runaway was an overcorrection. No entropy lever needed.
- **RUNTAG passthrough DONE** (commit c0a3369) — the prior `.out`/runname clobber + TB newest-glob
  ambiguity is fixed; submit with `--export=ALL,RUNTAG=<tag>`.
- **`--time=04:00:00` confirmed fine** (gpu MaxTime = 7 days). V100 ran 4000 updates in ~70 min wall —
  budget ≥1.5 h for a 4000-update run, more if NUPD rises.
- **Adroit home quota tight (9.3/10 GiB)** — jobs write to /scratch so unaffected, but worth a cleanup pass.
