# inc8 SMOKE HARNESS FIX — REPORT (laptop, 2026-06-14)

**Worker:** inc8 smoke harness fix (code-worker → overall commander). Scope = the four harness
defects the inc8 GPU smoke (job 3273374) exposed + the re-smoke budget. **No reward/env/model
logic touched.** Branch ready for the commander to merge; the Adroit re-smoke is a separate worker.

**Branch:** `claude/inc8-smoke-harness-fix` (off `main` @ 0b47bf3). Pushed.

---

## The four fixes

### A. Export-RC guard — `rl/peregrine_train_inc8.py` (memory:434 RESOLVED)
**Seam chosen:** a monkeypatch of `TrainRunner.close` in the peregrine wrapper, mirroring the file's
existing `TrainRunner.run = _run_with_lifelines` lifeline pattern. The diffaero clone stays PRISTINE.

The crash (from the smoke report's stack): `runner.close() → agent.export() → PolicyExporter.export()
→ torch.onnx.export() → exporter.py post_process → ValueError: Unknown action frame: body`. It is the
**last** step of `close()`, firing AFTER train + checkpoint + the TorchScript/.pt2 export all succeed
(`cfg/dynamics/quad.yaml` ships `action_frame="body"`; diffaero's `exporter.post_process` only
implements `"local"`/`"world"`). So a fully successful 300/300 run still exited 1.

**Why it cannot mask a real failure** — the guard catches the exception from `close()` but RE-RAISES
anything that is not unambiguously the cosmetic ONNX crash. The predicate
`_is_onnx_action_frame_export_error` requires **all three**:
1. `isinstance(exc, ValueError)` — a NaN-abort raises `RuntimeError`, a dim/shape bug raises other types;
2. the unique diffaero message — `"action frame" in str(exc).lower()` (i.e. `Unknown action frame: …`);
3. an export call-site in the traceback — `"onnx"`/`"export"`/`"exporter"` in `format_tb(exc.__traceback__)`.

A real training failure (NaN-abort via the GuardedPPO `NAN_ABORT_AFTER`, a construction error, an
emergency-save failure, any other `close()`-time error) fails at least one clause and propagates a
non-zero RC. The guard is also strictly downstream of train+checkpoint (it only runs once `run()` has
returned), so it can never swallow a training-time problem. On a match it prints a CLEAR
`[onnx-guard]` warning citing memory:434 and returns `None` → the process exits 0.

This is isolable cleanly (the message `Unknown action frame: body` is unique to the exporter), so no
stop-and-escalate was needed. The `.pt2`/TorchScript export — the one we actually deploy — already
succeeds before the ONNX step, so nothing load-bearing is skipped.

### B. sbatch RUN-dir glob — `rl/peregrine_inc8_smoke.sbatch`
Old `outputs/train/*/inc8_smoke_A_seed${SEED}*` never matched (returned empty `RUN=`). diffaero/hydra
writes `outputs/train/<date>/<time>/` with the runname appearing only in the **decorated** TB logdir
`<date>/<time>/quad__<env>__<algo>__<model>__<runname>__<idx>/`. New resolution:
- **Primary:** `ls -dt ${TRAINOUT}/*/*/*inc8_smoke_A_seed${SEED}__*` → the decorated dir for THIS smoke
  (excludes the precheck, whose runname is `inc8_precheck`) → `dirname` gives the `<date>/<time>` run
  dir that holds `checkpoints/`.
- **Fallback (layout-robust):** newest `${TRAINOUT}/*/*/` run dir. The smoke follows the precheck, so
  newest == smoke. This covers the case where the runname is not embedded in any dir name.

Either path yields the run-dir tree, which the recursive TB reader and the `actor.json` `find` then walk.

### C. sbatch signal trace from TensorBoard — `rl/peregrine_inc8_smoke.sbatch` + `rl/inc8_tb_trace.py`
inc8 metrics are **TensorBoard-only** (`env_loss/inc8_*`, `env_loss/total_reward`, `agent_loss/*`),
so the old stdout-grep found nothing. Replaced it with a new helper `rl/inc8_tb_trace.py` invoked by
the sbatch on the compute node. It:
- recursively finds the run's `tfevents` file (picks the one with the most scalar steps — the real log);
- resolves the six requested metrics by tag-suffix (exact-suffix match first, so `inc8_pointing_rate`
  never shadows `inc8_terminal_pointing` or `metrics/pointing_rate`): **total_reward,
  inc8_pointing_rate, inc8_terminal_pointing, inc8_fix_rate, entropy(entropy_loss),
  value_loss(critic_loss)**;
- prints the **TRAJECTORY** (first, last, and every ~100 updates — `--stride-updates 100`), not just
  endpoints.

**Compute-node safe:** the only third-party import is `tensorboard.backend.event_processing.
event_accumulator`, which is already in the diffaero env (diffaero logs to TB) — **no download**. That
import is DEFERRED inside `main()`, so the module imports clean where tensorboard is absent and prints
a benign reporting-only message instead of crashing the sbatch tail.

### D. `.gitattributes` — `*.sbatch text eol=lf` + re-normalize
The committed sbatch reached Adroit as CRLF and `sbatch` rejected it (hand-fixed on the cluster).
Added `*.sbatch text eol=lf` (next to the existing `*.sh` rule) and ran `git add --renormalize` on the
inc8 sbatch. Verified: `git check-attr` → `eol: lf`; the staged blob has **0** CR bytes.

---

## Re-smoke budget (defaults bumped; still a cheap, fair gate)
In `rl/peregrine_inc8_smoke.sbatch`: `NUPD` default **300→1000**, `#SBATCH --time` **00:45:00→02:30:00**.
**Unchanged:** `NENVS=2048` (a real GPU-saturation reading), `SEED=0`, arm A, the precheck stage,
reward arms, DR. Rationale (preserved in a comment): 300 was pre-competence — basic course-following
hadn't established, so the pointing term had nothing to shape and the util reading was export/startup-
dominated; 1000 lets course-following emerge AND dilutes the one-time export/startup cost into a
representative util reading — while staying well short of an L0 run (6000).

---

## Validation
- **Full suite from repo root** (`.venv/Scripts/python.exe -m pytest`): **828 passed in 912 s, 0 failed**
  (== baseline 828).
  The wrapper guard adds module-level patch code only (diffaero-gated; not imported by the suite) and
  did not change the count.
- **Helper `rl/inc8_tb_trace.py`:** AST-parses; imports clean WITHOUT tensorboard (deferred import);
  degrades gracefully on a missing dir / absent tensorboard (exit 0, reporting-only message). Pure
  logic unit-tested: tag resolution picks the correct `env_loss/inc8_*` tags (no cross-shadowing),
  nearest-value lookup, and stride selection (steps 0,100,…,900,990) all correct.
- **sbatch** can't run locally; the new RUN-resolution + helper invocation are shell/Python the
  compute node runs. The python helper is `python -c` import-clean.
- **Did NOT** run anything on Adroit; **did NOT** touch reward/env/model logic.

## Files changed
| path | change |
|---|---|
| `rl/peregrine_train_inc8.py` | Fix A — `TrainRunner.close` ONNX-export RC guard (narrow, re-raises real failures) |
| `rl/peregrine_inc8_smoke.sbatch` | Fix B (RUN glob) + Fix C (TB trace invocation) + budget bump (1000/2.5h); now LF |
| `rl/inc8_tb_trace.py` | NEW — compute-node-safe TB trajectory reader (event_accumulator, deferred import) |
| `.gitattributes` | Fix D — `*.sbatch text eol=lf` |

**The branch `claude/inc8-smoke-harness-fix` is ready for the commander to merge; after merge the
Adroit re-smoke is a separate worker.**

---

## MEMORY-DELTA (≤10 lines → commander; I do not write project memory)
- ✅ **memory:434 RESOLVED** — cosmetic ONNX-export RC=1 (`ValueError: Unknown action frame: body`,
  diffaero `exporter.post_process`, fires in `runner.close()` AFTER train+checkpoint+.pt2) now guarded
  in the wrapper `rl/peregrine_train_inc8.py` via a `TrainRunner.close` monkeypatch. NARROW: re-raises
  unless `ValueError` + `"action frame"` message + onnx/export traceback frame → a real NaN-abort /
  construction / save failure still exits non-zero. diffaero clone stays pristine.
- 🐞 **sbatch RUN glob fixed** — old `*/inc8_smoke_A_seed0*` ≠ diffaero layout; now resolves the
  decorated logdir `<date>/<time>/quad__…__inc8_smoke_A_seed0__0/` (`dirname` → run dir) w/ newest-dir
  fallback.
- 🐞 **sbatch signal trace = TensorBoard** — new `rl/inc8_tb_trace.py` (event_accumulator, in-env, no
  download; deferred import) prints the TRAJECTORY every ~100 upd of total_reward / inc8_pointing_rate /
  inc8_terminal_pointing / inc8_fix_rate / entropy / value_loss; replaces the empty stdout-grep.
- 🔧 **.gitattributes `*.sbatch text eol=lf`** + re-normalized the inc8 sbatch (was CRLF → `sbatch`
  rejected on cluster). Staged blob now 0 CR.
- 📈 **Re-smoke budget = 1000 upd / 2.5 h** (was 300 / 45 min); NENVS=2048, SEED=0, arm A, precheck,
  reward/DR all KEPT. 300 was pre-competence; 1000 < L0's 6000.
- Branch `claude/inc8-smoke-harness-fix` ready for merge; Adroit re-smoke = separate worker.
