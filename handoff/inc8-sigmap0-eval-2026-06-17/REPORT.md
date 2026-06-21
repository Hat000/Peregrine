> 🚩 **SUPERSEDED 2026-06-19 (511e85c) re: the σ_p0 bar.** Any "σ_p0 ≲ 0.08 / NO-GO vs 0.08 / near-field-gate-estimator pivot" conclusion in this report is OVERTURNED — the 0.08 bar was a `margin_envelope.py` double-count of the drone (real gate clearance 0.75−r ≈ 0.37–0.47 m). The REAL bar is **σ_p0_lat ≲ 0.15 (p99 ≲ 0.45)**; measured 0.15–0.20 = MARGINAL-PASSING, NOT NO-GO. Gate-4 is a REACH/PASS-RATE problem; RL stays the tool (the "pivot off RL" is RETRACTED). The engineering + measurements below STAND; only the bar and its GO/NO-GO verdict are corrected. → MEMORY.md:8 + memory/project_rl_increment_history.md §inc8-2026-06-19.
# inc8 σ_p0 EVAL — STOP REPORT (the GO gate on the 3 warm-started checkpoints)

**Date:** 2026-06-17 · **Model:** Opus 4.8 · **EFFORT:** medium · **Branch:** `p2-inc8-sigmap0-eval`
**Mode:** OFFLINE (laptop `.venv`, no live sim, no real detector) · **Escape hatch: TRIGGERED.**

---

## 0. VERDICT — BLOCKED, no GO/NO-GO possible (escape hatch fired)

> **The σ_p0 GO/NO-GO on seed0/1/2 CANNOT be produced.** Two hard blockers, one of which is the
> escape-hatch condition verbatim ("if σ_p0 can't be computed GT-anchored, STOP and report what's
> missing — do NOT substitute estim_err"):
>
> 1. **The 3 warm-start checkpoints are not reachable from this machine.** They exist ONLY as Adroit
>    SLURM outputs under `/scratch/network/fl3689/diffaero/outputs/train/.../inc8_warmstart_seed{0,1,2}_ws1/`.
>    Not in the checkout, not on disk anywhere under `C:\Users\Fengy`, **no GitHub release exists**
>    (`api.github.com/.../releases` returns empty), and `gh` is not installed. Commit `11cb840` merged
>    the warm-start *capability* (code), not the trained *artifacts* (binaries are gitignored). This is
>    **CRITICAL-PATH-ZERO** (MEMORY: "no policy .pt in checkout — Fengyou relays BEFORE any fan-out").
>
> 2. **`contact_true_eval.py` does not emit a GT-anchored σ_p0 — its `--estim-emul` "σ_p0" numbers
>    are `estim_err`.** `g4_inplane_p90/p99` come from `EstimatorEmulator.gate4_inplane_error_series()`
>    = `inplane_err = |KF_pos − truth_pos|` in the gate frame (estimator_emul.py:357, :398). That is the
>    estimator error, floored at the ~0.115–0.13 gate-relative KF RMS **regardless of policy** — exactly
>    the quantity the task and the warm-start build REPORT (§6/§8) say NOT to crown on. Reporting it as
>    σ_p0 would be the forbidden substitution.
>
> **What I did instead of fabricating a number:** removed both *tooling* obstacles so the relay is one
> command per seed — (a) fixed the loader (it could not even load a 20-dim inc8 actor), and (b) built
> a correct GT-anchored σ_p0 instrument, validated end-to-end. **Provisional verdict on delivery of the
> checkpoints: PENDING** — and any GO from this path stays PROVISIONAL until the real-detector spike
> confirms emul fidelity (#37).

---

## 1. Why "contact_true_eval's GT-anchored σ_p0" does not exist as wired (the core finding)

The task assumed `contact_true_eval.py` already exposes a GT-anchored σ_p0. It does not. It exposes two
distinct quantities, **neither of which is σ_p0**:

| What it exposes | Source | Is it σ_p0? |
|---|---|---|
| `g4_inplane_p90/p99` (the `--estim-emul` numbers) | `inplane_err = \|KF − truth\|` gate-frame (estim_err) | **NO** — floored at KF RMS, policy-independent. The forbidden substitute. |
| `passed_linfs(4)` (per-gate margin table) | TRUE-crossing `linf = max(\|y\|,\|z\|)` | GT-anchored but **single-episode** + **combined-axis** → not a lateral σ. |

**Empirical proof (real run, in-repo 20-dim inc8 actor `stage1_inc8_actor.pth`):**
```
[ESTIMATOR-EMULATION — gate-4 selection numbers]  obs_dim=20
  trainreset_g4   v*=16.6  fixrate_g4=0.000  lock_g4=0.00  ip_p90=0.242  ip_p99=0.257  n_band=9
```
`ip_p90=0.242 / ip_p99=0.257` is the per-step **estimator** in-plane error over the gate-4 band of one
episode — floored near the KF RMS, NOT a multi-episode terminal-centering spread.

**σ_p0 (the GO gate) is a different object:** the standard deviation of the **TRUE drone lateral
position** relative to the gate-4 centre at the gate-4 plane crossing, across an ENSEMBLE of episodes
whose spread is induced by the per-episode estimator DR. It needs (i) lateral kept separate from
vertical, (ii) the TRUE trajectory (not the KF), and (iii) many episodes. None of the three is in the
stock CLI.

---

## 2. What I built (so the relay produces a TRUSTWORTHY number, not estim_err)

### 2a. Loader fix — `rl/fly_rl.py` (necessary for ANY offline inc8 eval)
`load_actor` built a hardcoded 17-dim `_ActorMean`, so a 20-dim inc8 checkpoint raised
`size mismatch ... [256,20] vs [256,17]` — i.e. **the stock eval could not load the warm-start
checkpoints at all.** Fix: `_ActorMean(obs_dim=17)` is now width-parameterised, and `load_actor`
infers the width from the checkpoint's own first layer (`head.0.linear.weight` columns). inc7 (17-dim)
path is byte-unchanged (verified: still loads as 17-dim; all 37 `test_contact_true_eval` tests pass).

### 2b. GT-anchored σ_p0 instrument — `rl/inc8_sigmap0_eval.py` (NEW)
Runs **N simstart episodes** with `--estim-emul` ON (per-episode DR via `emul_seed = base+i`), and at
each gate-4 PASS extracts the **TRUE** crossing offset with the SAME `_R_W2G` transform and SAME plane
interpolation `contact_true_eval._score_gate` uses — but keeps `y` (lateral) and `z` (vertical)
separate instead of collapsing to `linf`. Reports:
- `sigma_p0_lat = std(y)`, `bias_lat = mean(y)`, lateral `|miss|` p90/p99 (the GO axis);
- `sigma_p0_vert`, vertical p90/p99 (the dominated axis, for completeness);
- `reach_rate` (fraction reaching gate-4 — a low value is itself a NO-GO signal);
- `terminal_gate_lock(mean)` + `g4_band_fix_rate(mean)` (pointing retention, the inc8 lever).
- A GO rule: `sigma_p0_lat ≤ 0.08` AND lateral p99 ≈ 3·σ ≤ ~0.24, stamped PROVISIONAL (emul, not #37).

### 2c. Tests — `tests/test_inc8_sigmap0_eval.py` (NEW, 4 tests, all pass)
20-dim load; inc7 stays 17-dim; gate-4 crossing is GT-anchored, finite, sub-band; None when gate-4 not
reached. (Adds 4 to the suite → green-gate sentinel baseline moves 933 → 937 on merge.)

### Validation (mechanics proven; NOT a GO/NO-GO — wrong subjects)
- Loader: `stage1_inc8_actor.pth` (20-dim) now loads and the full eval runs to completion.
- Crossing extraction: inc7 on truth obs completes the course and the instrument extracts a real
  GT-anchored gate-4 lateral offset = **−0.215 m** (σ=0 trivially — one deterministic episode, as the
  tool warns). This is the inc7 baseline centering offset; it is exactly the gate-4 problem inc8 exists
  to fix. It is **not** a verdict.
- inc7 under `--estim-emul` ON reaches gate-4 **0/24** times → confirms a σ_p0 ensemble needs a
  course-completing **case-C centering policy under noisy obs** = precisely the warm-start seeds. The
  instrument is ready; only the subjects are missing.

---

## 3. RELAY — exactly what to deliver, and the one command per seed

**Get the 3 checkpoints off Adroit onto the laptop** (each dir holds `actor.pth` + `critic.pth` +
`actor.json`; only `actor.pth` is needed for the eval). On the Adroit **login** node:
```bash
# locate the 3 warm-start output dirs (file read — AUP-safe, no compute):
ls -d /scratch/network/fl3689/diffaero/outputs/train/*/*/inc8_warmstart_seed*_ws1*/periodic 2>/dev/null
# confirm each sidecar reads "obs_dim": 20, "inc8": true, "r5_arm": "A"
cat <dir>/actor.json
```
Bring each `actor.pth` (+ its `actor.json` sidecar — needed for the action bounds) to the laptop, e.g.
`rl/checkpoints/inc8_ws1_seed{0,1,2}_actor.pth`. Then **per seed**, from repo root with `.venv` active:
```powershell
.venv\Scripts\python.exe rl\inc8_sigmap0_eval.py `
    --ckpt rl\checkpoints\inc8_ws1_seed0_actor.pth --estim-emul --n-episodes 200
```
Read `sigma_p0_lat` and `lat_p99` from the `SIGMA_P0_SUMMARY` line; pick the best seed; the GO rule is
printed (PROVISIONAL). (Alternatively run on Adroit under SLURM — 1-core, accurate `--mem`, no
login-node compute, `checkquota` — but the laptop `.venv` is simpler and this is pure CPU numpy.)

🚩 **Open metric-definition question for the commander to ratify before crowning:** σ_p0 here is the
ENSEMBLE std of the true lateral crossing across DR draws (per-episode terminal centering). MEMORY also
phrases it as "per-fix σ_lat". If the intended definition is the within-approach per-fix lateral spread
rather than the across-episode terminal-crossing spread, say so and I will adjust the aggregation — the
crossing extraction is the same either way.

---

## 4. Status of the deliverables the task asked for

| Asked | Status |
|---|---|
| per-seed gate-4 σ_p0 lateral p90/p99 @ r∈{0.26,0.30,0.33} | **BLOCKED** — checkpoints not reachable |
| terminal_pointing retained + band fix density | instrument ready (`terminal_gate_lock`, `g4_band_fix_rate`); needs checkpoints |
| clear 0.08? best seed? GO/NO-GO | **BLOCKED** — pending checkpoint relay; PROVISIONAL even then (#37) |
| do not declare inc8 done | honored — nothing crowned |

Note: σ_p0 is radius-independent (it is the crossing geometry, not the pass-band); the r∈{0.26,0.30,0.33}
sweep affects PASS/COLLISION verdicts (reach_rate), not the lateral offset value — the instrument reports
reach_rate so the radius sensitivity shows up there.

---

## 5. MEMORY-DELTA (text only — do NOT commit memory/; commander banks)

```
inc8 σ_p0 EVAL = BLOCKED → STOP report (branch p2-inc8-sigmap0-eval, 2026-06-17). NO verdict produced.
- BLOCKER-1 (critical-path-zero): the 3 warm-start ckpts (inc8_warmstart_seed{0,1,2}_ws1) are Adroit
  SLURM outputs only — NOT in checkout, NOT on laptop disk, NO GitHub release, no gh CLI. Fengyou must
  relay the 3 actor.pth (+actor.json sidecars) to the laptop. Then 1 cmd/seed (see REPORT §3).
- BLOCKER-2 (escape hatch): contact_true_eval --estim-emul "σ_p0" (g4_inplane_p90/p99) IS estim_err
  (inplane_err=|KF−truth| gate-frame, estimator_emul.py:357/398) — floored ~0.13 RMS, policy-indep.
  NOT GT-anchored σ_p0. Proven on stage1_inc8_actor.pth (ip_p90=0.242 = the floor, not centering).
- BUILT + validated (ready for the ckpts): (a) fly_rl.load_actor now infers obs-width from the ckpt's
  first layer → 20-dim inc8 actors LOAD (were a hard size-mismatch); inc7 byte-unchanged. (b) NEW
  rl/inc8_sigmap0_eval.py = GT-anchored σ_p0: N simstart eps, estim-emul ON, TRUE gate-4 crossing
  lateral/vertical kept separate (same _R_W2G/_score_gate math), reports sigma_p0_lat std + lat p90/p99
  + reach_rate + terminal_lock + g4_fix_rate + PROVISIONAL GO rule (≤0.08, p99≈3σ). (c) tests +4 (suite
  933→937). inc7 truth-obs baseline lateral offset = −0.215 m (mechanics proof, NOT a verdict; inc7
  under emul reaches g4 0/24 → σ ensemble needs a case-C centering policy = the warm-start seeds).
- OPEN: ratify σ_p0 definition (across-episode terminal-crossing std vs within-approach per-fix σ_lat).
- Any GO from this path PROVISIONAL until the real-detector vertical spike confirms emul fidelity (#37).
```
