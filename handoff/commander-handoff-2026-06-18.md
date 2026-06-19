# COMMANDER HANDOFF — 2026-06-18 (Gen 3 → Gen 4)

*Read `COMMANDER.md` (craft) and `memory/MEMORY.md` (state) first. This file is the transient "what is happening RIGHT NOW" snapshot at retirement. Delete/supersede it once you've absorbed it.*

## TL;DR
You inherit a live burn. **TWO autonomous `/goal` runs are RUNNING — do not disturb them.** The binding gate-4 number (σ_p0) is built and **one Adroit `sbatch` away**. `main` is local-only (ahead of origin by 50, **not pushed**), and a ShadowPC clone may be on its own `main` — **do not push or rewrite history** until that's reconciled.

## What is LIVE right now — DO NOT DISTURB
1. **Laptop `/goal` — internal system-ID registration sweep.**
   - Branch `p2-system-id`, worktree `.claude/worktrees/festive-mclean-a79c65` (branched from `main@dddf5d9`).
   - Goal: register every laptop-runnable (diffaero-free) estimator/obs/geometry pipeline numpy↔torch; fix-or-pin every divergence; end GREEN with `handoff/system-id-2026-06-18/REPORT.md`.
   - **#1 target = the live blocker:** the obs-emulator divergence between numpy `EstimatorEmulator` (`rl/estimator_emul.py`) and torch `BatchedEstimatorEmulator` (`rl/inc8_estimator_emul.py`) — this is why a policy that flies in torch dies 0/200 in the numpy grader.
   - Commits per-pair to `p2-system-id`. **When it finishes:** read its REPORT → selective-merge to main. Its finding decides FIX-THE-NUMPY-GRADER vs MEASURE-IN-TORCH.
2. **ShadowPC `/goal` — vision model training.**
   - Separate machine, separate clone. You cannot see or touch it from the laptop.
   - ⚠️ **It MAY be committing to its own `main`.** The laptop `main` and ShadowPC `main` may have DIVERGED. Reconcile via `origin` DELIBERATELY (fetch + inspect + merge) — **never force-push either direction.**

## What is PENDING — your move
- **σ_p0 gate-4 number = ONE Adroit run.** `rl/inc8_sigmap0_torch.sbatch` is merged (`dceaeba`) and merge-reviewed (APPROVED — GT-crossing math mirrors the numpy tool; spawn-contamination-guarded; look-at warmup-footgun-guarded). **diffaero is laptop-absent**, so the 2-axis σ_p0 can only be measured on Adroit. Fire it (Fengyou's call):
  - `--lookat yaw` FIRST → must reproduce the numpy cross-check **~0.177** (instrument validation; if it disagrees by >0.04, suspect spawn-jitter / #37 — do not trust the 2-axis number).
  - then `--lookat auto` → the 2-axis deliverable. **GO rule: σ_p0_lat ≤ 0.08 AND lat_p99 ≤ 0.24.**
- **VQ2 honest wire** — still awaited (no release date). It decides case-C = LOAD-BEARING vs INSURANCE, and un-gates: (a) the ShadowPC arm of the system-ID (ours↔ShadowPC registration — build the harness now, run when VQ2+ShadowPC free), (b) the appo decision.

## The σ_p0 saga — resolved to "instrument-ready, Adroit-gated" (full ledger → `[[project-rl-increment-history]]` §inc8)
- **rc1 (recenter re-train, job 3276449) = FIRST FLYING inc8** (3/3 seeds, success ~0.50–0.54). Root-cause fix was restoring the through-approach centering the ladder had dropped (`9cecf64`).
- **Yaw-injection bug FIXED** (`ec4cb03`): the eval reconstructed FLU via the training adapter `_FLIP=[1,-1,-1]` but `policy_step` emits FRD via the live `_ACT_FLU_TO_FRD=[1,-1,1]` (eval plant `_RATE_SIGN_LIVE=[1,1,1]`). Pitch was NEVER the bug (a prior report's `_RATE_SIGN_LIVE=[1,1,-1]` was wrong and misled the pitch hypothesis).
- **2-axis σ_p0 is un-measurable on the numpy emul** — rc1 flies 2-axis in torch but dies on the numpy-eval obs = the #37 obs-fidelity gap (torch-emul OBS ≠ numpy-emul OBS), amplified by g_pitch=3. → the torch-env instrument measures it where rc1 provably flies.
- **Best-available faithful σ_p0 = yaw-only 0.177 m → NO-GO vs 0.08** (pessimistic; pitch/elevation off).

## Resources built this session
- `rl/inc8_sigmap0_torch_eval.py` + `rl/inc8_sigmap0_torch.sbatch` — Adroit σ_p0 instrument (GT gate-4 crossing in the torch env).
- `rl/inc8_sigmap0_eval.py` — numpy σ_p0 tool (yaw-only 0.1768 ✓; 2-axis dies = #37).
- `rl/contact_true_eval.py` — numpy offline eval (look-at ported `44617ce` + yaw-injection fixed `ec4cb03`).
- Checkpoints (gitignored): rc1 = `rl/checkpoints/inc8_rc1_seed{0,1,2}_actor.{pth,json}` (local) / `outputs/train/inc8_recenter_seed{0,1,2}_rc1/` (Adroit).
- `tests/test_inc8_sigmap0_torch_crossing.py` (3/3) pins the crossing math without diffaero.

## Git state + DEFERRED cleanup (do AFTER the goals drain — NOT before)
- `main` = `dceaeba`, **LOCAL-ONLY, ahead of origin by 50, NOT pushed.** Do not push until ShadowPC is reconciled.
- ~16 stale worktrees/branches clutter `git worktree list` (finished workers). Prune ONLY after confirming no agent is on them; **never** prune `festive-mclean-a79c65` (the live `/goal`) and never blindly prune the harness-managed `.claude/worktrees/*` pool. Use `git worktree prune` (gone dirs) + remove confirmed-DONE manual worktrees; prune MERGED branches at the gate.
- **HEAD-drift footgun bit this session:** a worker ran in the shared main checkout and left HEAD on its branch, so a commit landed on the wrong branch (recovered via ff-merge). **GATE every commit on `branch==main`** — a hard test, not a printed echo.
