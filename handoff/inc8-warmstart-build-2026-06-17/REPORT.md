# inc8 WARM-START build — REPORT

**Date:** 2026-06-17
**Session:** P2 INC8-RL — warm-start (checkpoint-resume) capability for the inc8 trainer
**Branch:** `p2-inc8-warmstart` (worktree off `main` @ 54f3da6)
**Model:** Opus 4.8 · EFFORT high

---

## 1. Summary

Built a checkpoint warm-start path for the inc8 trainer so the S3 centering re-pilot can fine-tune from
the **converged S2 seed-2 checkpoint** instead of fresh init. S3 fresh-seed failure was two COLD-START
artifacts (2/3 seeds died in gain-warmup; the 1 breakout seed hit the centering-onset reward cliff at
~step 1200). A warm-start starts inside the pointing basin → no warmup ramp to survive and no fix-onset
discontinuity for centering to slam into.

**Three files (1 edited, 2 new), 13 new tests, full suite 871 passed / 42 skipped / 0 failed.**

| File | Change |
|---|---|
| `rl/inc8_warmstart.py` | **NEW** — `maybe_warmstart(agent, env, cfg)`: OFF → byte-identical no-op; ON → validate ckpt, sidecar obs-dim gate, `agent.load`, force look-at warmup to 0. |
| `rl/peregrine_train_inc8.py` | **+1 import, +1 guarded call** in `_run_with_lifelines` (before `_orig_run`). Nothing else. |
| `rl/peregrine_inc8_warmstart.sbatch` | **NEW** — precheck + 3 fine-tune seeds, modeled VERBATIM on the S0/S2-convergent COMMON. |
| `tests/test_inc8_warmstart.py` | **NEW** — 13 tests: (a) off no-op + import-clean, (b) load-reproduces-actions, (c) warmup=0 ⇒ factor 1, + trainer AST guards. |

**ESCAPE HATCH: not triggered.** OFF is byte-identical (proof §3); the warmup interaction is safe (it
matches the existing, S2-proven `env._ppo_update` write path). Shipped.

---

## 2. The chosen `init_from` key — ROOT-level `+init_from`, NOT `+train.init_from`

The brief *suggested* `+train.init_from` but said "match the existing config namespacing." I read the
diffaero source and the existing sbatches and the evidence is unambiguous:

- diffaero's `TrainRunner` and the training scalars (`n_updates`, `seed`, `save_freq`, `log_freq`,
  `n_envs`, `runname`, `headless`, `device`) are all **root-level** — there is **no `train` config
  group** in diffaero. A `train.init_from` key would land in a dead namespace.
- env-scoped things go under `env.` (`+env.rw_centering`, `+env.lookat_*`); dynamics under `dynamics.`;
  algo under `algo.`. `init_from` is a training-loop concern → root-level, matching `seed`/`save_freq`.

**Decision: `+init_from=<ckpt dir>`** (single-plus; it's a new key not in the pristine diffaero config,
same as `+env.inc8` / `+dynamics.dr`). 🚩 This deviates from the brief's literal suggestion — flagged
for the commander, but it is the correct match to the actual diffaero namespacing.

---

## 3. OFF-byte-identical proof (`+init_from` unset → current trainer, bit-for-bit)

Three independent legs:

1. **Minimal, guarded trainer diff** (the *only* change to an existing file):
   ```diff
   +from inc8_warmstart import maybe_warmstart
   ...
       agent.step = step_with_periodic_save
   +    warmstart_from = maybe_warmstart(agent, env, cfg)
   +    if warmstart_from is not None:
   +        print(f"[lifeline] CONTINUING training from warm-started weights: {warmstart_from}")
       try:
           _orig_run(self)
   ```
2. **`maybe_warmstart` is a total no-op when unset.** Its FIRST effective statement is
   `init_from = resolve_init_from(getattr(cfg, "init_from", None))`; with the key absent that is
   `None`, and the function `return None` immediately — **before** importing torch, before touching
   `agent` / `env` / `cfg`, with no I/O and no RNG draw. The `if warmstart_from is not None:` then
   skips the print. So the agent, env, optimizer, rollout buffer, and RNG stream are untouched →
   training proceeds identically. (Pinned by `test_off_path_is_total_noop`, which uses an
   `_ExplodingAgent` whose `.load` raises if ever called on the off-path, and asserts the env warmup is
   unchanged — mirrors `test_maybe_widen_critic_off_is_noop`.)
3. **No other load path in the trainer.** `test_trainer_has_no_other_checkpoint_load_path` asserts the
   trainer source contains no `torch.load` / `agent.load` — the *only* load route is `maybe_warmstart`,
   whose off-path is the proven no-op. `test_trainer_calls_maybe_warmstart_once_before_training`
   (AST) pins the call happens exactly once and before `_orig_run`.

The trainer imports diffaero at module top, so (like `test_inc8_off_identity`) the *numerical* OFF==inc7
check belongs to the Adroit precheck; the structural+behavioral proof above is the laptop guarantee.

---

## 4. How the load works (verified against the diffaero source)

I read `github.com/flyingbitac/diffaero` (the upstream of the Adroit clone) to get the save/load contract
exact rather than guess:

- `PPO.load(path)` → `self.agent.load(path)`.
- `StochasticActorCriticV(ActorCriticBase).load(path)` → `self.actor.load(path)` **and**
  `self.critic.load(path)`.
- `StochasticActor.load` reads `<path>/actor.pth = {"actor_mean": state_dict, "actor_logstd": Parameter}`.
- `CriticV.load` reads `<path>/critic.pth = critic.state_dict()`.
- This is the **exact inverse** of the periodic-save `agent.save(path)` the trainer lifeline already
  writes (actor.pth + critic.pth + the `actor.json` sidecar).

So `maybe_warmstart` just calls **`agent.load(init_from)`** — loads actor + critic together, leaving the
optimizer (fresh Adam moments) and rollout buffer fresh. **Weights-only transfer**, which is the right
choice for a reward re-pilot (changed landscape → fresh moments).

🚩 **Deviation from the brief's "reuse load_actor / load the critic separately":** `fly_rl.load_actor`
reconstructs a *deploy-side standalone `_ActorMean`* (the 17-dim MLP) — it does not load into the live
training agent's modules, and it doesn't touch the critic. Using diffaero's own verified `agent.load`
is strictly better (loads both into the live agent, pristine-API, no reimplementation to drift). The
"load the critic separately" instruction is therefore moot — `agent.load` already loads both. Flagged.

**Extra safety:** before loading, `maybe_warmstart` (a) requires the dir to exist with **both** actor.pth
and critic.pth (a deploy-only actor.pth is rejected — the critic must resume too), and (b) reads the
`actor.json` sidecar and refuses an obs-dim mismatch (a 17-dim inc7 ckpt can't warm-start a 20-dim inc8
run) with an actionable error instead of a cryptic `load_state_dict` size failure.

🚩 **Critic stays SYMMETRIC (obs-20).** This knob loads weights into whatever agent was built; it does
NOT switch to the appo asymmetric critic (a separate decision) and does NOT widen the critic. If you ever
combine `+init_from` with `+algo.critic_hidden_dim`, the source `critic.pth` must match that width.

---

## 5. Warmup interaction (brief item 2)

A warm-started policy already points at full look-at gain; re-ramping the gain from 0 over
`lookat_warmup_updates` (`peregrine_train_inc8.py:124` → `inc8_reward.lookat_warmup_factor`) would
disrupt that established pointing. So on warm-start `maybe_warmstart` **forces the env's look-at warmup
to 0** (and the sbatch passes `+env.lookat_warmup_updates=0` belt-and-braces). With warmup=0,
`lookat_warmup_factor(idx, 0) == 1.0` for every update — full gain held from update 0 (verified:
`wf(0,0)=1.0`, `wf(0,200)=0.0`, `wf(100,200)=0.5`).

- I **force** (not merely default) it to 0 and log loudly if a positive warmup was requested, because
  warm-start + ramp is contradictory and there is no case where re-ramping a loaded pointing policy is
  desirable. This also defends against a future COMMON baking a positive warmup. (Flagged for commander
  in case strict "respect-explicit" semantics are preferred — trivial to change to a warn-only.)
- The write `env._lookat_warmup_updates = 0` uses the **same `self.env` reference** the trainer already
  uses to advance `env._ppo_update` each update — the path the S2 run proved reaches the underlying
  env's warmup logic (S2 seed-2's gain demonstrably ramped). Identical, proven mechanism.

---

## 6. The sbatch (`rl/peregrine_inc8_warmstart.sbatch`)

- **COMMON = `rl/peregrine_inc8_s0.sbatch` lines 71–83 VERBATIM** — the exact config that, with the S2
  overrides, produced seed-2 convergence. I did not guess it.
- **Overrides** (verified single/double-plus):
  - `+init_from=${CKPT}` — root-level new key.
  - `++env.lookat_g_yaw=-3.0 ++env.lookat_g_pitch=3.0` — DOUBLE-plus (both ARE in COMMON; empirical
    signs: yaw −3.0, pitch +3.0).
  - `+env.lookat_warmup_updates=0` — single-plus (NOT in COMMON).
  - `+env.rw_centering=1.0` — single-plus (VERIFIED NOT in COMMON; only `rw_perc_*` are. `rw_centering`
    maps via `_inc8_weights_from_cfg`'s `rw_<field>` → `Inc8RewardWeights.centering`).
- **Per-seed explicit `hydra.run.dir`** (the S1@3.0 Hydra dir-collision footgun) — each seed writes its
  own dir, can't come out bit-identical.
- **ADROIT AUP block carried verbatim** from the repilot sbatch + a WARM-START PRE-SYNC note: the Adroit
  `peregrine_repo` is a **file copy, not a git clone** → SYNC `inc8_warmstart.py` +
  `peregrine_train_inc8.py` to the login node before submitting (no `git pull` there).
- **GO/measurement block:** judge on the **physical GT-anchored σ_p0** (`contact_true_eval.py` true
  err_ip), NOT the estimator-floored `estim_err` (floored at the ~0.115–0.13 gate-relative RMS
  regardless of policy). Headline readout: **value_loss must NOT cliff at ~step 1200** (where S3 fresh
  collapsed). The precheck echo says "SYMMETRIC critic (obs-input 20)" — NOT the misleading "critic 36".
- `bash -n` clean.

---

## 7. Tests & verification

```
tests/test_inc8_warmstart.py ............. 13 passed       (off no-op, import-clean, load-reproduces,
                                                             obs-dim gate, missing-file, warmup=0⇒1,
                                                             warmup-forced-to-0, trainer AST guards)
full suite:  871 passed, 42 skipped, 0 failed  (8m34s)     (no regressions vs main)
```

Test (b) runs against a **diffaero-faithful stand-in** (actor_mean MLP + actor_logstd Parameter + critic
MLP, with save/load mirroring `StochasticActor`/`CriticV`/`PPO` EXACTLY as read from source): save a
source agent, warm-start a *differently-initialised* agent, assert actor-mean + logstd + tanh(mean)
action + critic value all reproduce the source within 1e-6 after 0 updates. The real-agent numerical
check is the Adroit precheck (laptop has no diffaero) — consistent with the project's test convention.

---

## 8. Launch command for Fengyou

On the Adroit **login** node (after pre-syncing the two rl/ files):
```bash
# 1. locate the S2 seed-2 converged checkpoint (file read — AUP-safe):
CKPT=$(ls -dt /scratch/network/fl3689/diffaero/outputs/train/*/*/*inc8_s0_yawprim_seed2_s2full_s2__*/periodic | head -1)
cat "$CKPT/actor.json"          # confirm  "obs_dim": 20,  "inc8": true,  "r5_arm": "A"

# 2a. submit all 3 seeds in one job:
sbatch --export=ALL,CKPT="$CKPT",RUNTAG=ws1 /scratch/network/fl3689/peregrine_repo/rl/peregrine_inc8_warmstart.sbatch

# 2b. OR 3 separate jobs (more robust — one seed crashing can't kill the others):
for s in 0 1 2; do sbatch --export=ALL,CKPT="$CKPT",SEEDS="$s",RUNTAG=ws1_s$s \
  /scratch/network/fl3689/peregrine_repo/rl/peregrine_inc8_warmstart.sbatch; done
```
Then read the trace: **value_loss @ ~step 1200 (no cliff) → centering rises with pointing retained →
crown on the physical σ_p0 from `contact_true_eval.py`, not estim_err.**

---

## 9. MEMORY-DELTA (text only — do NOT commit memory/; commander banks)

```
inc8 WARM-START BUILT (branch p2-inc8-warmstart, 2026-06-17; laptop code, NOT launched — Adroit/VQ2-gated).
- rl/inc8_warmstart.py + 1 guarded call in peregrine_train_inc8.py (_run_with_lifelines, before _orig_run).
  OFF (+init_from unset) = BYTE-IDENTICAL (returns None on 1st stmt; no torch/agent/env/RNG touch). 13 new
  tests; full suite 871 pass/0 fail. ESCAPE HATCH not triggered.
- KEY = ROOT-level `+init_from=<ckpt dir>` (NOT +train.init_from — diffaero has NO `train` group; init_from
  is root-level like seed/n_updates/save_freq). Single-plus (new key).
- LOAD = diffaero's own agent.load (PPO.load→actor.load+critic.load; reads actor.pth{actor_mean,actor_logstd}
  +critic.pth — the verified inverse of the lifeline periodic-save). WEIGHTS-only: optimizer+buffer FRESH
  (correct for a reward re-pilot). Did NOT reuse fly_rl.load_actor (deploy-side _ActorMean, wrong target).
  Sidecar obs-dim gate (17-dim inc7 can't warm-start 20-dim inc8). Critic stays SYMMETRIC (NOT appo).
- WARMUP: warm-start FORCES env look-at warmup→0 (loaded policy already points; re-ramp would disrupt).
- sbatch rl/peregrine_inc8_warmstart.sbatch = S0/S2-convergent COMMON VERBATIM + ++env.lookat_g_yaw=-3.0
  ++env.lookat_g_pitch=3.0 +env.lookat_warmup_updates=0 +env.rw_centering=1.0 + per-seed hydra.run.dir +
  AUP verbatim + file-copy PRE-SYNC note. GO = physical σ_p0 (contact_true_eval), NOT floored estim_err;
  watch value_loss @~step1200 (no cliff = warm-start solved it; spike = escalate to appo). CKPT=S2 seed-2
  /periodic (jobs 3275300-02; locator in sbatch header). Launch cmd in REPORT §8.
```
