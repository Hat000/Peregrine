> 🚩 **SUPERSEDED 2026-06-19 (511e85c) re: the σ_p0 bar.** Any "σ_p0 ≲ 0.08 / NO-GO vs 0.08 / near-field-gate-estimator pivot" conclusion in this report is OVERTURNED — the 0.08 bar was a `margin_envelope.py` double-count of the drone (real gate clearance 0.75−r ≈ 0.37–0.47 m). The REAL bar is **σ_p0_lat ≲ 0.15 (p99 ≲ 0.45)**; measured 0.15–0.20 = MARGINAL-PASSING, NOT NO-GO. Gate-4 is a REACH/PASS-RATE problem; RL stays the tool (the "pivot off RL" is RETRACTED). The engineering + measurements below STAND; only the bar and its GO/NO-GO verdict are corrected. → MEMORY.md:8 + memory/project_rl_increment_history.md §inc8-2026-06-19.
# inc8 2-axis σ_p0 — torch-env evaluator: BUILT + VALIDATED, number pending one Adroit run

**Session:** inc8 2-axis σ_p0 in the torch training env · 2026-06-18 · branch `p2-sigmap0-torch`

---

## HEADLINE (read this first)

- **Does rc1 fly 2-axis in torch inference (Y/N)?** — **NOT YET MEASURABLE ON THIS MACHINE.** The torch
  training env (`PeregrineRacingInc8`) subclasses **diffaero**, which is **absent on the laptop**
  (`ModuleNotFoundError: No module named 'diffaero'`; torch is CPU-only). `rl/diffaero_dynamics.py:8-10`
  states it outright: *"This file CANNOT run on the dev laptop (no torch / pytorch3d / diffaero); it
  targets the Adroit GPU env."* The torch env exists only on Adroit (`/scratch/network/fl3689/diffaero_repo`,
  `conda activate diffaero`, GPU node). The prompt's "run at modest n_envs on CPU, laptop" premise does not
  hold; the prompt's own fallback applies: *"GPU on Adroit/ShadowPC, but the BUILD is laptop work."*
- **The faithful 2-axis σ_p0_lat + GO read** — **PENDING the Adroit run.** It is produced by ONE sbatch
  submit: `rl/inc8_sigmap0_torch.sbatch`. The instrument is built, statically validated, and its cross-check
  target is pinned. The number is one Adroit job away, not a laptop computation.
- **What this session delivers (laptop work, done):**
  1. `rl/inc8_sigmap0_torch_eval.py` — the faithful torch-env σ_p0 evaluator (deterministic inference,
     estimator-DR on, GT gate-4 crossing capture, look-at on, the warmup footgun closed).
  2. `rl/inc8_sigmap0_torch.sbatch` — the Adroit run recipe (yaw-only cross-check → 2-axis, all 3 rc1 seeds).
  3. `tests/test_inc8_sigmap0_torch_crossing.py` — pins the diffaero-free crossing math (3/3 pass).
  4. A freshly-reproduced **numpy yaw-only baseline = σ_p0_lat 0.1768 m** (the cross-check target).
  5. Adversarial static review of the evaluator (the only pre-Adroit bug screen) — see §5.

---

## 1. Feasibility finding (why the number is an Adroit step, not a laptop one)

The σ_p0 instrument the prompt asks for measures the policy flown on the **diffaero torch plant + torch
estimator emulator** — the exact environment rc1 trained in and provably flies. That fidelity is the whole
point (rc1 flies 2-axis in training but dies 0/200 on the numpy tool's emulated obs, #37). Reproducing the
torch env in numpy ≈ rebuilding `contact_true_eval` — i.e. the numpy path we are explicitly trying to get
away from. So the torch env is genuinely required, and the torch env requires diffaero.

Evidence gathered (2026-06-18, this laptop, `.venv`):
- `python -c "import diffaero"` → `ModuleNotFoundError`. `pip show diffaero` → not found. No vendored copy on
  disk (only test files referencing it by name in stale worktrees). `torch 2.12.0+cpu`, `cuda_avail False`.
- The torch env's whole stack (`peregrine_racing.py` → diffaero env base; `diffaero_dynamics.py` →
  diffaero `BaseDynamics`; `peregrine_train_inc8.py` → `import diffaero.algo/env/dynamics`) is Adroit-only.
- The established run infrastructure is Adroit SLURM (`peregrine_inc8_recenter.sbatch`: `conda activate
  diffaero`, `/scratch/network/fl3689`, GPU node) — this is where rc1 was trained.

**Conclusion:** the build + static validation are laptop work (done here); the run is one Adroit submit.
This is the escape-hatch-honest outcome — no σ_p0 is fabricated from a non-runnable environment.

## 2. The evaluator — `rl/inc8_sigmap0_torch_eval.py`

Modeled **exactly** on the proven in-repo torch-env evaluator `rl/peregrine_eval.py`:
`build_env(cfg.env)` / `build_agent(cfg.algo)` / `agent.act(obs, test=True)` (the **deterministic policy
mean**, no PPO noise) → `env.rescale_action` → `env.step`. Reads the run's `.hydra/config.yaml` (so it
inherits the trained cfg), with overrides. Key design points and how each prompt requirement is met:

- **GT gate-4 crossing capture (the σ_p0 ensemble).** A thin recording subclass `_Sigmap0Env(PeregrineRacingInc8)`
  overrides `step()` to call **`super().step()` verbatim** (the transition is byte-identical to training) and,
  around it, records the ground-truth gate-4 crossing. It detects a gate-4 pass as `prev_tg==4 &
  curr_tg==5 & ~reset` (gates advance one at a time; a clean non-last pass does NOT reset, so `self._p` after
  the step is still the env's `curr_pos`), then reconstructs the crossing with the **same** `world_to_gateframe`
  + x=0 interpolation the env's own `crossing_events` uses (`peregrine_racing_inc8.py:275-293`). It keeps `y`
  (lateral, the binding axis) and `z` (vertical) separate. Same-step pass+terminate cases are excluded and
  counted (`n_g4_then_reset`).
- **σ_p0 from TRUTH, never the KF.** `prev_p`/`self._p` are `env._p` (Z-up ground-truth). `estim_err`
  (KF−truth, floored ~0.115) is never used. σ_p0_lat = `std(y)`, bias = `mean(y)`, p90/p99 of `|y|` —
  identical statistics to the numpy tool.
- **Estimator DR ON = the spread source; isolated.** `inc8=true` builds the in-loop emulator;
  `_reset_emulator` draws `sample_episode_dr` (per-fix σ_lat ~U[0.05,0.15], one-signed PnP bias ~U[0,0.19])
  on **every reset**. Reproducible via `torch.manual_seed(seed)`. To make σ_p0 the **estimator-DR spread and
  nothing else** (apples-to-apples with the numpy 0.1768): `dynamics.dr` defaults **off** (nominal `mixer`
  plant); `--standing-frac` defaults **1.0** so every episode is a full gate-0→gate-4 course; and a
  **spawn-gate provenance filter** accepts only gate-0-spawned crossings (this is the §5 critical fix —
  without it, `standing_frac=0.3` contaminates the ensemble with cheap near-gate-4 partial spawns); the
  rollout uses a **fixed horizon budget** (accept every crossing, no count-based early-exit bias).
- **Look-at on, with the empirical gains.** `--lookat auto` = full 2-axis (`g_yaw=-3.0, g_pitch=3.0`),
  `yaw` = yaw-only (`g_pitch=0`), `off` = none. Gains set explicitly per mode (no reliance on the recorded
  cfg value — the sign footgun demands it).
- **🚩 WARM-UP FOOTGUN CLOSED.** In inference `env._ppo_update` stays 0, so a trained
  `lookat_warmup_updates>0` (rc1 trained with 200) would make `lookat_warmup_factor(0,200)=0` → the look-at
  gain is silently scaled to **zero** → an unfaithful, non-pointing rollout. The evaluator forces
  `lookat_warmup_updates=0` in cfg **and** sets `env._lookat_warmup_updates=0` + `env._ppo_update=1e9` after
  build (belt-and-braces).
- **Sanity (deliverable 1) folded in.** `--lookat auto` reports `reach_rate` and `success_rate` under
  deterministic inference. `reach>0 & success>0` IS the "rc1 flies 2-axis deterministically" precheck; `reach==0`
  would be the escape-hatch finding (training success was a stochastic-action artifact) — and is reported.
- **GO rule** identical to the numpy tool (`σ_p0_lat ≤ 0.08 AND lat_p99 ≤ 0.24`); emits a grep-able
  `SIGMAP0_TORCH_SUMMARY` line.

## 3. Static validation (laptop — the only pre-Adroit screen)

- **`py_compile`** of the evaluator: OK. (It cannot be import-smoke-tested on the laptop — its diffaero
  imports are real and Adroit-only; parse-only is the available laptop check.)
- **Crossing math pinned** — `tests/test_inc8_sigmap0_torch_crossing.py` (3/3 pass): proves
  `world_to_gateframe(d, π) == diag(-1,-1,1)·d` to 1e-12 (the frame the torch and numpy tools **share**), and
  that the x=0 interpolation reproduces the numpy tool's `gate4_true_crossing_yz` on shared points. This is the
  highest-risk piece (a wrong crossing → a wrong σ_p0) and it is now pinned without diffaero.
- **Adversarial review** of the evaluator vs ground-truth sources — see §5.

## 4. Cross-check target (pinned on the laptop)

The numpy tool runs on the laptop (no diffaero). Freshly reproduced on **rc1 seed0, yaw-only, start=trainreset,
200 ep**:

```
sigma_p0_lat = 0.1768 m   bias = -0.0982 m   |miss| p90 = 0.3242  p99 = 0.3699
reach 133/200 (0.67)      finished 61        NO-GO (0.177 > 0.08)
```

This matches the memory's "0.177 (bias −0.095, p99 0.375)" exactly. **The torch yaw-only run must land near
0.177** to validate the instrument; the sbatch checks `|torch − 0.1768| < 0.04 → AGREE`. NOTE: torch (diffaero
plant + torch emul, training spawn) and numpy (numpy plant + numpy emul) are different substrates that are
parity-tested but need not bit-match — agreement within a band validates; a large gap is a finding to explain
(plant/spawn/obs-fidelity), per the prompt's escape hatch.

## 5. Adversarial review findings + fixes applied

A 5-agent workflow (4 dimension reviewers + a completeness critic) reviewed the evaluator against the
ground-truth sources. Consensus: the **GT crossing capture, the inference/actor path, and the warmup
footgun guard are all CORRECT** (the recorded crossings are byte-identical to the env's own
`crossing_events`; `agent.act(test=True)→rescale→step` matches `peregrine_eval` verbatim; the warmup
is closed three independent ways). One **high-stakes number-biasing defect** and several minor issues
were found. Fixes applied (evaluator rewritten):

| # | Severity | Finding | Fix applied |
|---|----------|---------|-------------|
| 1 | **CRITICAL** | The σ_p0 spread was **not apples-to-apples** with the numpy 0.1768. With `standing_frac=0.3` (the trained value), 70% of respawns start "1 m up-course of a *random* gate" (`peregrine_racing.py:776-788`); ~1/6 land at gate-4 → cheap at-rest near-gate-4 crossings contaminate the ensemble. σ_p0 then mixed spawn-geometry variance with the estimator-DR spread it must isolate, and the yaw cross-check vs 0.177 would be invalid. | **Default `--standing-frac 1.0`** (every reset targets gate-0 → full course) **+ a spawn-gate provenance filter** (`_spawn_tg`, recorded in a `reset_idx` override) that accepts **only gate-0-spawned crossings** (`n_rejected_nonfull` reports discards). The ensemble is now one clean gate-0→gate-4 approach per episode — the same population the numpy fixed `trainreset` start measures. |
| 2 | **MAJOR** (critic) | Stopping the loop on a crossing **count** gathered in arrival order → over-represents fast-reaching envs, under-represents the slow tail that drives `lat_p99`/the verdict. | Replaced the count-based `while` with a **fixed-horizon loop** (`n_steps = horizons·max_time/dt`, like `peregrine_eval`); accept **every** crossing, report after. |
| 3 | **MAJOR** (critic) | Forcing `standing_frac=1.0` alone is insufficient — the **±0.25 m / ±0.15 rad spawn jitter is unconditional** (`peregrine_racing.py:792-795`); the numpy `trainreset` has zero jitter. | Not removed (overriding the spawn is untestable on the laptop and high-risk). Instead **documented**: over the ~135 m gate-0→gate-4 course the gate-0 jitter washes out → small sub-threshold residual, bounded by the 0.04 cross-check AGREE band. Flagged as the first suspect if the yaw cross-check DISAGREES. |
| 4 | minor×2 | `n_g4_then_reset` diagnostic read post-reset `target_gates` (aliased by the random-gate respawn) → wrong in both directions, misleading next to the verdict. | **Removed**; replaced by the correct `n_rejected_nonfull` (computed from the pre-step spawn tag). |
| 5 | minor | The `SIGMAP0_TORCH_SUMMARY` verdict was re-implemented inline (could drift from `_go_verdict`; no NO-DATA / min-ensemble guard). | Summary now uses `_go_verdict`, which emits **NO-DATA** for an empty **or** too-small ensemble (`--min-ensemble`, default 30). |
| 6 | minor | `reach_rate`/`success_rate` use a completed-episode denominator (a small in-flight tail is excluded). | Documented in the printout. |
| 7 | nit | `agent.load` needs `critic.pth` (PPO.load = actor+critic) even though inference uses only the actor; critic-width override would shape-error (rc1 is safe — no `critic_hidden_dim`). | Added an explicit `actor.pth`/`critic.pth` presence check with a clear message; documented the critic-width caveat. |
| 8 | nit | unused `sigma_target` param on `sigma_p0_report`. | Removed. |

**Dismissed (verified non-issues):** multi-gate-jump spoofing (gates advance ≤1/step), the reset-timing
capture (correct for non-reset envs), the warmup guard (correct, triple-closed), and the
diffaero-`build_agent` path (faithful for inference; `GuardedPPO` only guards the optimizer step).
The `tests/test_inc8_sigmap0_torch_crossing.py` pins still pass after the rewrite (3/3); `py_compile` OK.

## 6. How to get the number (the Adroit run recipe)

`rl/inc8_sigmap0_torch.sbatch` (GPU node, `conda activate diffaero`). Per rc1 seed it runs:
1. **yaw-only** instrument cross-check (target σ_p0_lat ≈ 0.18),
2. **2-axis (auto)** — the deliverable σ_p0 (sanity reach/success folded in),
and prints `SIGMAP0_TORCH_SUMMARY` + an AGREE/DISAGREE cross-check verdict.

```
# PRE-SYNC the two NEW files to the Adroit peregrine_repo (file copy, no git pull there):
#   rl/inc8_sigmap0_torch_eval.py , rl/inc8_sigmap0_torch.sbatch
sbatch /scratch/network/fl3689/peregrine_repo/rl/inc8_sigmap0_torch.sbatch
# single seed / best ckpt / include plant DR:
sbatch --export=ALL,SEEDS=0,CKPT_SUB=best,EXTRA="--dynamics-dr" \
  /scratch/network/fl3689/peregrine_repo/rl/inc8_sigmap0_torch.sbatch
```

**Read order:** the yaw cross-check FIRST (does the torch instrument reproduce ~0.18? if not, the 2-axis
number is suspect — report the discrepancy); then the 2-axis σ_p0_lat vs the 0.08 GO rule; pick the best seed.

## 7. Provisional caveat

Every number here and from the Adroit run is **PROVISIONAL** — the obs is the estimator EMUL (fix_surrogate →
LinearKF), not a real detector→PnP→KF (#37). Crown only after the real-detector spike confirms emul fidelity.

---

## MEMORY-DELTA (≤10 lines; do not commit memory/)

- 🚩 **Torch-env σ_p0 is an ADROIT step, NOT a laptop one:** diffaero is ABSENT on the laptop (ModuleNotFoundError;
  torch CPU-only; `diffaero_dynamics.py:8-10` confirms Adroit-only). Build+validation = laptop; the NUMBER = one sbatch.
- 🚩 **Instrument BUILT+validated:** `rl/inc8_sigmap0_torch_eval.py` (mirrors `peregrine_eval.py`: build_env/build_agent/
  `agent.act(test=True)`/rescale/step; recording subclass captures GT gate-4 (y,z) via the env's own `world_to_gateframe`+x=0
  interp). Run via `rl/inc8_sigmap0_torch.sbatch` (yaw cross-check → 2-axis, 3 seeds). `tests/test_inc8_sigmap0_torch_crossing.py` pins the crossing math.
- 🚩 **FOOTGUN (forget=unfaithful):** torch inference keeps `_ppo_update=0`, so trained `lookat_warmup_updates>0` (rc1=200)
  scales the look-at gain to ZERO. Any torch-env eval/replay MUST force warmup=0 (cfg + `env._lookat_warmup_updates=0`).
- 🚩 **σ_p0 must isolate the ESTIMATOR-DR spread:** the env reset spawns 70% "1 m up-course of a RANDOM gate" at
  `standing_frac<1` → near-gate-4 spawns contaminate the ensemble (5-agent review caught it). Measure with `standing_frac=1.0`
  + gate-0-spawn filter + fixed-horizon (no count early-exit). Residual: ±0.25 m spawn jitter is UNCONDITIONAL (`peregrine_racing.py:792-795`).
- 🚩 **Cross-check target PINNED (laptop numpy tool, rc1 seed0, yaw-only, trainreset): σ_p0_lat 0.1768** (bias −0.098,
  p99 0.370, reach 133/200). Torch yaw-only must land near it. **2-axis GO/NO-GO vs 0.08 = PENDING Adroit; PROVISIONAL on emul (#37).**
