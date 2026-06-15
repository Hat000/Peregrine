# inc8 util sync-fix — REPORT (2026-06-14)

**Worker:** inc8 util sync-fix (laptop) · **Branch:** `worker/inc8-util-syncfix-2026-06-14`
**Boundary:** perf + logging-cadence ONLY — zero change to reward / obs / estimator / gradient.
**Scope file:** `rl/peregrine_racing_inc8.py` (inc8-on path only; parent `peregrine_racing.py` untouched).

---

## 1. Hypothesis CONFIRMED — the per-step `.item()` host-syncs in the metrics dict are the dominant tax

Traced the step path and the `loss_components` contract end-to-end:

- `loss_components` is built **every step** (no log-gating in the env) and returned in `extra` (step():360).
  diffaero's `TrainRunner` consumes it; the env must emit it each step. `rl/inc8_tb_trace.py` reads the
  TB tags `total_reward · inc8_pointing_rate · inc8_terminal_pointing · inc8_fix_rate · entropy ·
  value_loss` at a ~50-update stride — the env produces the scalars, the runner logs them.
- The dict is **NOT log-gated** and the metrics are **logging-only** (they do not feed reward/obs/gradient;
  `reward` is the separately-returned tensor; the dict drives the TB trace).

**Per-step host-sync inventory (inc8-on step), BEFORE:**

| Site | Syncs | Notes |
|---|---|---|
| `peregrine_racing_inc8.py:305–316` metrics dict | **11+** | 10× `.item()` + `bool(terminal.any())`; **plus** `in_img[terminal]` boolean-mask gather is itself a data-dependent sync (terminal_pointing alone ≈ 3). **Dominant contiguous block.** |
| `peregrine_racing_inc8.py:175` `bool(finite.all())` (get_observations) | 1–2 | per obs call (~1–2 calls/step) |
| `inc8_estimator_emul.py:531` `accepted.nonzero()` | 1 | **REQUIRED** (data-dependent KF update). LEFT as-is — confirms the estimator is NOT the cause. |
| `peregrine_racing_inc8.py:322` `reset.nonzero()` | 1 | **REQUIRED** (reset indexing). LEFT as-is. |
| parent `peregrine_racing.py:386–397` `compute_reward_terms` | ~10 | 10× `.item()` building the base loss dict; runs every inc8 step (called at inc8 step():282). **Co-equal share, but in the inc7-shared reward fn → OUT OF BOUNDARY.** See §4. |

The estimator emul (`inc8_estimator_emul.py`) has exactly **one** per-step sync (the required `nonzero`),
torch-batched + parity-gated — confirming the prompt's root-cause: the cost is the **metrics dict**, not
the estimator.

## 2. Fix — 11+ → 1 sync in the metrics dict; mechanism = stack-then-single-`.tolist()`

Picked mechanism **(a)** (stack the scalar means into one tensor → a single `.tolist()`), the lowest-risk
option: the dict the runner receives is **byte-identical in structure and type** (same keys, same Python
floats, same every-step cadence) — only the *number of CUDA synchronises* changes (11+ → 1).

- All 11 scalars (the 10 means + the lifetime `obs_nonfinite` count) are `torch.stack`ed in `self._inc8_dtype`
  and read back with **one** `.tolist()`.
- `inc8_terminal_pointing` is reproduced sync-free as a **masked mean**:
  `sum(in_img · terminal) / clamp(#terminal, min=1)`. Mathematically identical to the old
  `in_img[terminal].mean()` (and the empty case → `0/clamp(0,min=1)=0`, matching the old `else 0.0`).
  This also removes the old `bool(terminal.any())` sync **and** the `in_img[terminal]` boolean-gather sync.
- `inc8_conf_anneal` stays a Python float (`R8.confidence_anneal` returns a float — no sync; left as-is).

**get_observations:175** — replaced the `if not bool(finite.all()): … int((~finite).sum())` guard with an
**unconditional** `torch.where(finite, obs, 0)` (an identity when all-finite ⇒ numerically and gradient-
identical) + an **on-device** lifetime counter `self._nonfinite_obs_t` accumulated sync-free and read once
per step inside the same batched `.tolist()`. Removes the per-obs-call sync without ever skipping the guard.

**Per-step host-sync inventory, AFTER (managed paths):**

| Site | Syncs |
|---|---|
| metrics dict (incl. obs_nonfinite + terminal_pointing) | **1** (the single `.tolist()`) |
| get_observations NaN guard | **0** |
| estimator `nonzero` (required) | 1 |
| `reset.nonzero()` (required) | 1 |

Managed syncs **11+ (dict) + 1–2 (obs) → 1** total. (Parent `compute_reward_terms` ~10 remain — §4.)

## 3. Parity / safety — perf + logging-cadence ONLY

- **OFF == inc7 byte-identical: HOLDS.** All edits live strictly *after* the `if not self._inc8_on:
  return super()…` first-statement guards (step / get_observations / get_state unchanged); `__init__`'s
  new `self._nonfinite_obs_t` is added in the inc8-on branch *after* the early `return`. The AST guard
  test `tests/test_inc8_off_identity.py` passes unchanged.
- **No behavioral change** beyond logging efficiency. The only nuance: `inc8_terminal_pointing` is now a
  masked mean — mathematically identical, but may differ from the old gather-based mean in the **last
  float32 ULP** (summation order) in a **logging-only** value that never touches training. TB stores
  float32; the trajectory the commander adjudicates is unaffected. **TB tag names + cadence: UNCHANGED.**
- **S1–S4 torch==numpy parity: unchanged** — those tests exercise `inc8_estimator_emul` / `inc8_reward`
  directly (not the env metrics dict), and neither module was touched.
- **Full suite (repo ROOT, `.venv\Scripts\python.exe -m pytest tests/`): 828 passed, 0 failed** (10:30,
  exit 0) — matches the 828-green baseline of the inc8 torch port merge (b0b322e). See §5.

## 4. FLAGGED for the commander — parent `compute_reward_terms` co-equal ~10 syncs (OUT OF BOUNDARY)

`compute_reward_terms` (`peregrine_racing.py:386–397`) does **~10 `.item()`** building its base loss dict,
and it is called **every inc8 step** (inc8 step():282). That is a **co-equal** share of the per-step sync
tax. It is **logging-only** (the `reward` tensor is returned separately and is unchanged), so batching it
would be numerically identical and would benefit **inc7 too** — but it is the **inc7-shared reward
function** (inc7 = LIVE-CONFIRMED current best), explicitly outside this worker's boundary. **Left
untouched.** To fully saturate the GPU, recommend a follow-up (commander-authorized) that batches those
~10 syncs the same way. Expected: this worker's fix roughly halves the metric-sync tax; batching the
parent removes the other half.

## 5. Verification log

- Targeted: `test_inc8_off_identity test_inc8_env_integration test_obs_sign_faithfulness test_inc8_reward`
  → **20 passed**.
- Full suite (repo ROOT): **828 passed in 630.31s (0:10:30), exit 0**.

## 6. Deliverable summary

- Branch `worker/inc8-util-syncfix-2026-06-14`, 1 file changed (`rl/peregrine_racing_inc8.py`,
  +47 / −15). Diff: (a) `__init__` adds on-device `_nonfinite_obs_t`; (b) get_observations sync-free NaN
  guard; (c) metrics dict batched into one `.tolist()` with masked terminal_pointing.
- **The next Adroit pilot confirms util ↑ + the trajectory tags pointing** (util can't be measured on the
  laptop).
