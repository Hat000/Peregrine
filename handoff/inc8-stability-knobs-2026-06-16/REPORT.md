# inc8 stability knobs — REPORT (P2 inc8-RL, laptop worker, 2026-06-16)

**Branch:** `p2-inc8-stability-knobs-2026-06-16` (off `main`). Dispatch-ready; no GPU run on the laptop.

Three configurable knobs for the NEXT inc8 run, each **OFF-byte-identical** to current inc8 and unit/parity-
tested on the laptop (.venv, torch 2.12.0+cpu). diffaero is cluster-only, so the wider-critic *construction*
(knob #3) is precheck-verified on Adroit; everything else is fully laptop-verified.

---

## The 3 knobs + EXACT override strings (for dispatch)

| # | Knob | Default (OFF) | Override string to enable |
|---|------|---------------|---------------------------|
| 1 | **band_el metric** (observability only) | always-on logging | *(none — a new logged metric `inc8_band_el_abs_deg` + trace column `band_el_abs_deg`)* |
| 2 | **look-at gain-warmup** | `lookat_warmup_updates=0` | `+env.lookat_warmup_updates=200` *(single `+`; not in COMMON)* |
| 3 | **critic-width A/B** | `critic_hidden_dim` unset | `+algo.critic_hidden_dim=[512,256]` *(single `+`; `algo.` namespace)* |

Knob #2 composes with the validated S1 mechanism — full enable line, e.g.:
```
++env.lookat_g_yaw=-3.0 +env.rw_centering=3.0 +env.lookat_warmup_updates=200
```
(`lookat_g_yaw` stays DOUBLE-plus — it is in COMMON; `rw_centering` and `lookat_warmup_updates` are single-plus,
not in COMMON, so they append. Warmup scales MAGNITUDE only → the `-3.0` sign is preserved.)

Knob #3 example (the wider critic, actor untouched):
```
+algo.critic_hidden_dim=[512,256]
```

---

## Knob #1 — `band_el` metric (OBSERVABILITY ONLY, no reward/obs/action change)

The elevation analog of `band_az_abs_deg`. In `peregrine_racing_inc8.step()` diagnostics, next to band_az:
```python
_el_deg = torch.atan2(_tcam[..., 1], _tcam[..., 2].clamp(min=1e-6)) * (180.0 / torch.pi)
band_el_abs = (_el_deg.abs() * _look_band.to(mdt)).sum() / _look_band.sum().clamp(min=1)
```
- Reuses the **SAME `_look_band`** as band_az (one band definition, both divide by its sum) → cannot perturb
  training; it is part of the logging-only `metric_vec` stack read by the single existing `.tolist()` sync.
- Stacked → `metric_vec`, unpacked → `band_el_v`, logged → `loss_components["inc8_band_el_abs_deg"]`.
- Trace column `band_el_abs_deg` added to `rl/inc8_tb_trace.py` COLUMNS (resolves `inc8_band_el_abs_deg`).

**Purpose:** diagnoses whether the residual estimator error is VERTICAL (the gate-4 σ_vert-dominated axis the
20° mount struggles with), AND it is the empirical sign-check the **S2 g_pitch** primitive will need — the
analytical YAW sign was WRONG (band_az caught it in ~100 steps), so the analytical pitch sign is not trusted
either; band_el is how it gets verified on-GPU.

**Read order (S2):** with g_pitch ON + correct sign, `band_el_abs_deg` DROPS in the band (gate held near
elevation 0); a wrong sign makes it RISE → flip `lookat_g_pitch`.

## Knob #2 — look-at gain-warmup (the 2/3-collapse fix), gated, default-off

Ramps the look-at gain **magnitude** linearly 0 → target over the first `lookat_warmup_updates` PPO updates,
then holds at target. Damps the fresh-policy value_loss spike / entropy collapse caused by the full-strength
correction slamming the un-warmed policy (the 2/3-seed early collapse at updates ~100/800).

- **Pure helper** (`rl/inc8_reward.py`): `lookat_warmup_factor(update_idx, warmup_updates) → clip(u/N,0,1)`,
  or `1.0` when `N<=0`. Non-negative scalar → **sign can never flip** (validated gain is −3.0; factor∈[0,1]).
- **Env** (`rl/peregrine_racing_inc8.py`): `wf = R8.lookat_warmup_factor(self._ppo_update, self._lookat_warmup_updates)`
  and the gains are passed scaled: `lookat_correction(..., g_yaw*wf, g_pitch*wf, ...)`. `warmup_updates=0 → wf=1.0
  → g*1.0==g → byte-identical` to the no-warmup path.
- **Update counter** (`rl/peregrine_train_inc8.py`): the runner's per-update `agent.step` wrapper sets
  `env._ppo_update = counter["i"]` (0-based) before each rollout — faithful "PPO updates", no `l_rollout`
  duplication. No-op when warmup OFF (env forces factor 1.0); skipped for non-inc8 envs (no attribute).

Suggested starting value: `+env.lookat_warmup_updates=200` (collapse is at updates ~100–800; tune on the trace —
watch `value_loss`/`entropy` early and `band_az_abs_deg` falling by the time the ramp completes).

## Knob #3 — critic-width A/B (capacity for PPO stability), ACTOR UNCHANGED

A wider critic value-net → lower-variance value targets → can damp the value_loss-spike collapse. The actor is
NEVER widened (a wider actor tends to *worsen* PPO stability). Deploy-safe: the critic is training-only (saves
to `critic.pth`; deploy loads `actor.pth`).

New module `rl/inc8_critic_width.py` (`maybe_widen_critic`), called in `GuardedPPO.__init__` AFTER `super().__init__`
and BEFORE the nan-guard wraps `self.optim`:
- Unset `critic_hidden_dim` → **no-op, returns False** (byte-identical; off-path returns before importing
  omegaconf/diffaero or touching `agent`/`optim`).
- Set → rebuilds ONLY `agent.critic` with diffaero's own `CriticV` and a private network cfg (hidden_dim
  replaced, every other field preserved), then rebuilds the Adam optimizer over the new params (no optimizer
  state exists at construction → equivalent to a single from-scratch build). The critic **input dim is read back
  from the already-built critic** (`agent.critic.critic.input_dim`) → correct whether the critic is symmetric
  (obs-input) or, in a future AsymmetricPPO run, state-input.

**WHY a rebuild and not a plain `network.hidden_dim` override:** see the finding below — the actor and critic
currently share `cfg.network`, so a bare override would widen BOTH. The rebuild widens the critic alone.

---

## 🚩 FINDING (surfaced UP, does NOT block any knob): the inc8 critic is SYMMETRIC (obs-input), not the
## "asymmetric 36-dim" critic the doctrine assumes

Evidence (from OUR launcher + diffaero `flyingbitac/diffaero` @ main, fetched this session):
- `rl/peregrine_train_inc8.py`: `class GuardedPPO(PPO)` extends diffaero's **symmetric** `PPO`, and
  `GuardedPPO.build` passes **obs_dim only, no state_dim**.
- diffaero `AGENT_ALIAS = {"ppo": PPO, "appo": AsymmetricPPO}`; only `AsymmetricPPO` takes `cfg.critic_network`
  + `state_dim` and calls `env.get_state()`. The sbatch uses `algo=ppo` → our GuardedPPO → symmetric.
- Symmetric `StochasticActorCriticV` builds `critic = CriticV(cfg.network, obs_dim)` → the critic sees the
  **20-dim obs**, shares `hidden_dim` with the actor, and **never consumes `get_state()`**. inc8's `get_state`
  (33→36) and `self.state_dim` are not wired into training under this path (env sets only `obs_dim=20`, never
  `state_dim=36`; no `appo`/`critic_network` anywhere in the repo).

Implication: the "asymmetric / privileged critic anti-damping" mechanism in MEMORY may **not actually be active**
— the critic is a standard symmetric value net on the obs. This is consistent with how inc7 also ran (`algo=ppo`).
Caveat: I read GitHub HEAD; the Adroit clone *should* match ("pristine"), but the build signature in OUR launcher
(no state_dim) is the load-bearing evidence and is version-independent.

**1-line Adroit confirmation** (add to the precheck or run once): after building the agent, print
`type(agent).__name__`, `agent.agent.critic.critic.input_dim` (expect 20 if symmetric / 36 if asymmetric), and
`agent.agent.actor.actor_mean.input_dim`. If input_dim==20 → symmetric confirmed → if the team WANTS the
privileged critic, switch to `algo=appo` (GuardedPPO must then extend `AsymmetricPPO`, pass `state_dim`, and the
env must set `self.state_dim=36`). **This is a separate decision; knob #3 works correctly either way** (it reads
the actual input_dim). Flagged so the commander can decide whether the asymmetric critic is worth wiring.

---

## Files touched
- `rl/inc8_reward.py` — `lookat_warmup_factor` helper (pure; +14 lines).
- `rl/peregrine_racing_inc8.py` — band_el metric (knob #1); warmup state in `__init__` + gain scaling in `step()` (knob #2).
- `rl/inc8_tb_trace.py` — `band_el_abs_deg` trace column (knob #1).
- `rl/inc8_critic_width.py` — NEW module: `normalize_hidden_dims` + `maybe_widen_critic` (knob #3).
- `rl/peregrine_train_inc8.py` — import + call `maybe_widen_critic` in `GuardedPPO.__init__`; advance `env._ppo_update` per PPO update.
- `tests/test_inc8_lookat.py` — 5 warmup tests (knob #2).
- `tests/test_inc8_stability_knobs.py` — NEW: 13 tests (band_el formula/orthogonality/env-wiring/trace-column + critic-width off-path/parsing).

## Test results (laptop, .venv, torch 2.12.0+cpu)
- New/extended knob tests: **28 passed** (`test_inc8_lookat.py` + `test_inc8_stability_knobs.py`).
- FULL inc8 parity + byte-identical suite (off-identity AST + torch==numpy parity + env-integration + reward +
  reference-line + look-at + stability-knobs): **77 passed**.
- Whole `tests/` directory: **900 passed** in 407 s (zero failures — no collateral regression from the `inc8_reward.py` import).
- `py_compile` clean on all 5 diffaero-dependent / new sources.

OFF-byte-identical evidence:
- inc8 OFF (==inc7): `test_inc8_off_identity.py` (AST) still green — all my additions are inside the `_inc8_on`
  paths (numerical OFF-byte-identity is the Adroit smoke's job, unchanged).
- knob #2 warmup=0: `test_warmup_zero_reproduces_no_warmup_correction_exactly` → `torch.equal` (bit-identical),
  so S0/S1 (`lookat_g_yaw=-3.0`, no warmup) runs are unaffected.
- knob #3 unset: `test_maybe_widen_critic_off_is_noop` → returns False, `agent`/`optim` identity unchanged.

## Verification DEFERRED to the Adroit precheck (no diffaero on the laptop)
- knob #3 ON-path (the wider critic actually constructs + the 36-dim get_state wiring): add
  `+algo.critic_hidden_dim=[512,256]` to the precheck EXTRA; the rebuild prints
  `[critic-width] critic value-net REBUILT: hidden_dim [256,128] -> [512,256] (input_dim=…, params=…)`.
  (If it raises "could not read critic input dim", the diffaero network API moved — reconcile the module.)
- The 1-line symmetric/asymmetric confirmation above.

---

## MEMORY-DELTA (≤10 lines, for the commander to bank)
- inc8 STABILITY KNOBS built + dispatch-ready on `p2-inc8-stability-knobs-2026-06-16` (77 inc8 tests green, OFF byte-identical).
- Knob OVERRIDES: band_el = auto (logged `inc8_band_el_abs_deg`, trace col `band_el_abs_deg`; obs-only, S2 g_pitch sign-check); warmup = `+env.lookat_warmup_updates=200` (single `+`; magnitude-only ramp over first N PPO updates, sign-safe, composes w/ `++env.lookat_g_yaw=-3.0`); critic-width = `+algo.critic_hidden_dim=[512,256]` (actor untouched, deploy-safe).
- 🚩 FINDING: the inc8 critic is SYMMETRIC (obs-input 20), NOT the "asymmetric 36-dim" critic the doctrine assumes — GuardedPPO extends diffaero symmetric `PPO` + `build` passes obs_dim only + `algo=ppo`; `get_state`(36)/`state_dim` are NOT consumed in training (env sets only obs_dim=20). Strong (launcher build signature is version-independent); confirm on Adroit with `agent.agent.critic.critic.input_dim` (expect 20). To get the privileged critic the team must move to `algo=appo` + GuardedPPO(AsymmetricPPO) + state_dim=36 — separate decision; knob #3 works either way (reads actual input_dim).
- warmup needs the launcher to advance `env._ppo_update` per PPO update (wired in `peregrine_train_inc8.py`); without it the gain stays at update-0 → visible as band_az not falling.
- Files: rl/inc8_reward.py, rl/peregrine_racing_inc8.py, rl/inc8_tb_trace.py, NEW rl/inc8_critic_width.py, rl/peregrine_train_inc8.py; tests/test_inc8_lookat.py, NEW tests/test_inc8_stability_knobs.py.
