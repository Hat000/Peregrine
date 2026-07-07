# Ego (inc9) overnight launch — morning status (2026-07-07)

**TL;DR — the VQ2 egocentric ladder is running on Adroit: job `3297042` (SEED 0, RUNTAG `ego_a`).**
Smoke-validated first; two real bugs caught + fixed before burning the long run. Two commits on the
feature branch (`claude/optimistic-chaum-6c893b`): `668347f` (code) + `9b7d737` (memory mirror). NOT pushed.

## What ran
- Big-picture review: design holds (egocentric grounded, privileged reward/critic, refined-B, γ=0.9975 pinned).
- Fix agent (pre-launch) reconciled the 2 D+E blockers (warm-start lifeline wired; course-variation wired). 70 tests green.
- **Glaring problem caught:** Adroit's `/scratch/.../peregrine_repo` is a **plain dir (not git)** that had
  **drifted to an unmerged B2 base** whose native `course_n_gates` wiring COLLIDES with the ego's. The ego
  was built vs `main`'s base. Reconciled: backed up B2 → `b2_base_backup_2026-07-07/`, put `main`'s
  `peregrine_racing.py`+`peregrine_course.py` into peregrine_repo/rl, KEPT `inc8_estimator_emul.py`
  (ego-imported symbols byte-identical → "inc8 UNTOUCHED" honored). Deployed via base64-over-serve-daemon
  (no Duo needed while you slept).
- **2-stage smoke (single→handoff, 30 upd, 512 env)** caught + I fixed:
  1. ego reward omitted `loss_components['total_loss']` (diffaero runner reads it every step) → KeyError.
  2. diffaero ONNX/JIT exporter rejects the CTBR `action_frame="body"` at close()'s post-train export →
     nonzero exit → would have halted the ladder though training+checkpoint SUCCEEDED. Fixed with
     `export.jit=false export.onnx=false` (we deploy `.pth`, not ONNX).
- Smoke then **PASSED**: env builds vs main base · appo/privileged-critic assert PASSES (obs=21 ≠ critic=16)
  · both stages train · **warm-start chain LIVE** (`CONTINUING` lifeline) · checkpoints clean.

## The ladder (job 3297042)
- Stages: `single_gate`(1500) → `handoff_drill`(1500) → `dual_gate_full`(5500) → `multi_gate`(5500),
  warm-started stage→stage. algo=appo, γ=0.9975, 2048 envs (4096 on multi). ~12–15 h wall-clock est.
- Each stage auto-gates the next on a FLIGHTCHECK (halts a total stall; a flying stage never false-halts).
- **Morning check:** read the per-stage FLIGHTCHECK verdicts + `single_gate` success_rate — it should reach
  **~0.8** before promoting (the B2b validate-before-_COMMON lesson). Trace files:
  `/scratch/network/fl3689/peregrine_vq2_ego_ego_a_<stage>_trace.txt`; run log:
  `/scratch/network/fl3689/peregrine_vq2_ego_ego_a.out`.

## Notes / open
- Single seed tonight (you said "the long job"). A ≥5-seed portfolio (DESIGN §10) is a separate launch;
  held off — scratch quota was 67% (26 GiB headroom) and 5 seeds would strain it.
- B2 experimental base preserved in `b2_base_backup_2026-07-07/` (restorable). squeue was empty at deploy.
- If a stage crashes overnight I'll diagnose/fix/resubmit autonomously and note it here.
