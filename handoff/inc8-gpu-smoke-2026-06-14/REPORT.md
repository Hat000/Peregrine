# inc8 GPU SMOKE — REPORT (Adroit, 2026-06-14)

**Worker:** inc8 GPU smoke (single worker → overall commander). I do **not** make the GO call
and did **not** launch the L0 ladder — both are the commander's, post-report.

---

## VERDICT: 🔴 RED (as-written) — but read the nuance; the training infra is GREEN

The smoke does **not** meet the as-written GREEN bar (criteria a/b/c fail). **But the failure is
NOT a broken training pipeline.** Training ran end-to-end on the GPU, stably, no NaN, correct
20-dim obs, checkpoints saved. The RED decomposes into:

1. **One known-cosmetic crash** (ONNX export of `action_frame="body"`, fires *after* train+save) →
   `RC=1`. Already documented: `memory/project_phase2_rl_vision_decisions.md:434`.
2. **Two smoke-sbatch post-processing bugs** that hid the signal (RUN-dir glob; stdout-vs-TB grep).
3. **One genuine substantive flag** the commander needs *before* the ladder: under **arm A in 300
   updates, pointing did NOT emerge** and reward declined; **GPU util low (25.8%)**.

Items 1–2 are trivially fixable harness issues (NOT model issues). Item 3 is the real decision input.

---

## Job facts

| Field | Value |
|---|---|
| Job ID | **3273374** |
| Node / GPU | `adroit-h11g3` / **Tesla V100-PCIE-32GB** |
| Submitted / state | 19:57:03 → **COMPLETED**, Elapsed **00:03:21**, sbatch ExitCode 0:0 |
| `PRECHECK_RC` | **1** (cosmetic ONNX-export crash — see root cause) |
| `SMOKE_RC` | **1** (same) |
| Updates run | precheck **3/3**; smoke **full 300/300** (TB has 30 logs, steps 0→290) |

The sbatch wrapper exits 0 because its last command (sidecar `cat`) succeeded; the training
`python` processes exited 1 on the trailing ONNX export.

---

## The 4 GREEN criteria, with actual numbers

### (a) `PRECHECK_RC=0 AND SMOKE_RC=0` — ❌ both = 1, but **cosmetic & known**
The precheck log shows the real sequence (not a construction failure):
```
3/3 updates: loss 0.186 → 0.241 → 0.584, fps ~14k, success_rate 0.00   (training OK, cuda:0)
runner.py:168  "The checkpoint is saved to .../19-57-12/checkpoints."   (actor.pth/critic.pth/actor.json)
exporter.py:133 "... compiled and exported to .../exported_actor.pt2"   (.pt2 export SUCCEEDED)
runner.close() → agent.export() → PolicyExporter.export() → export_onnx() → torch.onnx.export()
  → JIT trace → exporter.py:87 post_process → ValueError: Unknown action frame: body   ← CRASH
```
So: **train → save checkpoint → .pt2 export (ok) → ONNX export (crash)**. The crash is the *last*
step in `runner.close()`, after training + checkpoint succeeded. This is exactly the banked
"cosmetic `TRAIN_RC=1` … fires AFTER `agent.save()`, checkpoint + eval unaffected" loose end.

### (b) GPU `mean_util ≥ 70%` — ❌ **mean 25.8% / max 29% (10 samples)**
Not zero (no risk of the 2h zero-util kill), but well below the saturation bar. Caveats: the run
was only ~3 min and export/startup-dominated, and `peregrine_racing_inc8` has a **numpy estimator
boundary** (`inc8_estimator_emul`) that the task explicitly flagged as a CPU-bottleneck risk. This
util reading is **not a clean saturation measurement** — but it's low enough to warrant a proper
look before committing 9 ladder runs of GPU.

### (c) pointing emerges + reward rising — ❌ **pointing flat, reward declining** ← the real signal
TensorBoard scalars from the smoke run (first→last over 300 updates):

| metric | first@0 | last@290 | read |
|---|---|---|---|
| `env_loss/inc8_pointing_rate` | 0.0000 | **0.0063** | flat ≈ 0 |
| `metrics/pointing_rate` | 0.0000 | **0.0156** | flat ≈ 0 |
| `env_loss/inc8_terminal_pointing` | 0.0000 | **0.0000** | **flat ZERO** (the load-bearing terminal-lock metric) |
| `env_loss/inc8_fix_rate` | 0.0000 | 0.0049 | flat ≈ 0 |
| `env_loss/total_reward` | −0.3246 | **−0.6367** | **declining** (wrong direction) |
| `metrics/success_rate` | 0.0000 | 0.0000 | no gate passes (expected @300upd) |
| `agent_loss/critic_loss` | 17.82 | 2.33 | critic IS learning |
| `agent_loss/entropy_loss` | 0.3591 | **1.6416** | entropy **rising** (policy drifting more random) |
| `agent_grad_norm/actor_grad_norm` | 2.92 | 0.73 | — |

Interpretation: the policy is training (critic loss drops, grads sane, no NaN), but it is **not
capturing the pointing reward** — pointing/terminal-pointing stay ~0, total reward gets *worse*,
and entropy *rises* (the optimizer is not finding an advantage gradient toward pointing).
**This is the escape-hatch "pointing FLAT under arm A" signal the task said to surface before the
ladder burns GPU.** Honest caveat: **300 updates is a very short budget**; this may be too early to
declare arm A broken, but it is clearly not trending the way the design expects, so it is the
commander's to weigh.

### (d) sidecar `obs_dim=20, inc8=true, r5_arm="A"` — ✅ PASS
Both precheck and smoke `actor.json`:
```json
{"act_max_thrust": 3.765, "act_max_rate": 3.14, "obs_dim": 20, "inc8": true, "r5_arm": "A"}
```

---

## Root cause(s)

**RC=1 (criteria a):** upstream `flyingbitac/diffaero` @ `291ea14` (the cluster clone, clean/no
local edits) is internally inconsistent on the quad export path: `cfg/dynamics/quad.yaml:7` ships
`action_frame: "body"` (committed, intentional — body-frame rates, see `rl/diffaero_dynamics.py:39`),
but `utils/exporter.py:post_process` only implements `"local"`/`"world"` and raises on anything
else. The `.pt2`/JIT export tolerates it; only `torch.onnx.export` (which re-traces through
`post_process`) hits the raise. Per project doctrine "the diffaero clone stays pristine
(no repo edits)" (`rl/peregrine_train.py:4`) — so the fix belongs in the peregrine train wrapper /
smoke harness, not in diffaero.

**Empty `RUN=` + no signal trace (criteria b/c reporting):** two independent smoke-sbatch bugs:
- The RUN glob `outputs/train/*/inc8_smoke_A_seed0*` does not match diffaero's actual layout
  `outputs/train/<timestamp>/quad__racing__ppo__mlp__inc8_smoke_A_seed0__0/` → `RUN=` empty →
  sidecar/signal steps no-op.
- The inc8 signal metrics live in **TensorBoard** (`env_loss/inc8_*`, `metrics/*`), **not stdout** —
  the sbatch greps the `.out`, which never contains them. (I pulled them from the TB event file.)

---

## What works (do not re-litigate)
- Code sync, env build, hydra wiring, 20-dim obs, critic 36, PPO on `cuda:0`, 300 updates, **no NaN**,
  checkpoints (`actor.pth`/`critic.pth`/`actor.json` + `exported_actor.pt2`) all saved. The inc8
  training pipeline is **functional on GPU**. Latency/portability not in scope here.

---

## Recommended fix path — **commander's call, NOT done by me**
1. **Make the cosmetic ONNX export non-fatal** so RC reflects training: e.g. `export_cfg.onnx=false`,
   or wrap `agent.export(...)` in try/except in the peregrine train entrypoint (the existing
   monkeypatch-lifeline pattern), keeping the diffaero clone pristine. (`.pt2` export already works.)
2. **Fix the smoke sbatch:** RUN-dir glob → match `outputs/train/<ts>/<decorated_runname>/`; read the
   signal trace + sidecar from the **TB event file**, not stdout.
3. **Re-run** (cheap) for a clean RC + signal — and **consider a longer budget than 300 updates**,
   since the smoke's whole purpose is "pointing emerges" and 300 is too short to be conclusive.
4. **Adjudicate the substantive flag** before the L0 ladder: pointing flat + terminal-pointing 0 +
   reward declining + entropy rising under **arm A**. Decide too-short-budget vs reward-shaping issue.
5. **Characterize GPU util 25.8%** (numpy estimator boundary?) — affects ladder GPU efficiency.

I did **not**: launch L0, change any reward/harness/model code, or make the GO/NO-GO call.

---

## Deploy notes (sync, done on the login node)
- `/scratch/network/fl3689/peregrine_repo` was a **stale flat copy** (parked **#69** confirmed — not a
  git repo, all inc8 files missing). No cluster GitHub creds (HTTPS no-prompt fails; no SSH deploy
  key) → can't `git clone`. Refreshed via **`git archive` tarball of 4d81352** from the laptop
  (== origin/main HEAD), SFTP-uploaded (30,243,382 B, gzip-verified), extracted clean. Stale copy
  backed up → `peregrine_repo.stale.bak`.
- The committed `rl/peregrine_inc8_smoke.sbatch` has **DOS (CRLF) line endings** → `sbatch` rejected
  it; normalized to LF on the cluster (deploy-time only). **Upstream hygiene fix wanted** (e.g.
  `.gitattributes *.sbatch text eol=lf`).
- diffaero env (`torch 2.5.1+cu121`, hydra 1.3.2, **numpy 2.4.6**) + `diffaero_repo` + symlink +
  `cfg/{env/racing,algo/ppo,dynamics/quad}.yaml` all present; pre-flight imports + compile all passed.
- Quota at submit: scratch **55.9/93 GiB** (fine); home **9.3/10 GiB** (tight — job wrote only to scratch).

## Artifact paths (on /scratch, not committed — large)
- raw stdout: `/scratch/network/fl3689/peregrine_inc8_smoke.out`
- gpu util:   `/scratch/network/fl3689/peregrine_inc8_gpu_util.log`
- precheck run: `/scratch/network/fl3689/diffaero/outputs/train/2026-06-14/19-57-12/`
- smoke run:    `/scratch/network/fl3689/diffaero/outputs/train/2026-06-14/19-57-23/` (best/periodic/checkpoints + TB events)
- deployed repo: `/scratch/network/fl3689/peregrine_repo` (4d81352 tree) · backup `…/peregrine_repo.stale.bak` · tarball `…/peregrine_4d81352.tgz`

Branch: none (pure ops; this report written on the worktree branch `claude/reverent-shirley-b94605`).

---

## MEMORY-DELTA (≤10 lines → commander; I do not write project memory)
- 🔴 inc8 GPU SMOKE = RED **as-written**, but infra GREEN: job 3273374, V100, **300/300 updates, no NaN**, obs_dim=20/inc8/arm A sidecar ✅. NOT a training failure.
- `PRECHECK_RC=SMOKE_RC=1` = the **known cosmetic** ONNX-export-of-`action_frame="body"` crash in `runner.close()` **after** train+checkpoint+.pt2 (memory:434 confirmed live). diffaero clone is pristine upstream `flyingbitac@291ea14`; quad.yaml ships `"body"`, exporter only does local/world. Fix in peregrine wrapper (keep diffaero pristine), not diffaero.
- 🚩 **SUBSTANTIVE FLAG (commander, pre-ladder):** under **arm A / 300 upd**, **pointing did NOT emerge** — `inc8_pointing_rate` 0→0.006, `inc8_terminal_pointing` **0→0.000 (flat)**, `total_reward` −0.325→−0.637 (declining), entropy 0.36→1.64 (rising); critic learns (17.8→2.3). This is the escape-hatch "pointing flat under arm A" signal. Caveat: 300 upd is very short — too-short-budget vs reward-shaping is the open question. **Do NOT crown / do NOT launch L0 until adjudicated.**
- GPU **util 25.8% mean / 29% max** — below 70% bar; suspect numpy estimator boundary (CPU-bound); needs a proper measurement (run was short/export-dominated).
- 🐞 Two smoke-sbatch bugs (not model): (1) RUN glob `outputs/train/*/inc8_smoke_A_seed0*` ≠ diffaero layout `…/<ts>/quad__racing__ppo__mlp__<run>__0/`; (2) inc8 signal is **TB-only** (`env_loss/inc8_*`,`metrics/*`), sbatch greps stdout → empty `RUN=`/trace. Fix both + re-run for a clean read.
- 🐞 committed `rl/peregrine_inc8_smoke.sbatch` has **CRLF** → `sbatch` rejects; normalized on cluster; want `.gitattributes *.sbatch eol=lf`.
- NEW parked-promote **#69**: `/scratch/.../peregrine_repo` is a flat copy (now refreshed via git-archive tarball of 4d81352; no cluster GitHub creds → no `git clone`). Recommend establishing a real clone or a documented push-deploy step for future syncs.
- Recommended commander sequence: fix export-RC guard + 2 sbatch bugs → re-run smoke (longer budget) → THEN decide L0/arm-A. GO/NO-GO + ladder remain the commander's.
