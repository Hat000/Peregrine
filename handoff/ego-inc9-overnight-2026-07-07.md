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

## UPDATE — single_gate FAILED, root-caused to 3 bugs, all fixed (job 3297088 ego_b now testing)
The first ladder (3297042) trained single_gate cleanly but **learned nothing** (success ~0, FLIGHTCHECK
correctly halted it before the hard stages — the "don't waste compute" guard working). I diagnosed it
(ultracode workflow + local geometry probes + isolated Adroit A/B experiments) to **three independent,
individually-fatal bugs** — and the isolated experiments PROVED each:
1. **RC1 — the emulated camera pointed backward.** The drone flies tail-first (your VQ1 control alias),
   but the ego's `gate_visibility` never applied the nose-first camera flip, so the camera looked 180°
   away from every gate → **0% detectable at all ranges** (local probe) → the vision obs was structurally
   blind. inc8 was immune only because it used a non-camera KF gate vector. FIX: apply `fly_rl`'s
   `_RZ_PI_BODY` flip to the *emulated camera only* (control/rel_pos untouched). Probe: 0% → 100%.
2. **RC2 — exploration noise ran away** to the std ceiling (bang-bang) because the `apply_noise_schedule`
   brake was never wired into the ego trainer. FIX: wired it + a flat std=0.30 held ceiling. (The
   entropy_loss=−13.68 I first read as "collapse to deterministic" was actually the *opposite* — the
   workflow caught my sign error.)
3. **RC3 — reward made approaching −EV** (finish 20 vs terminal 200). FIX: rw_progress 1→6, miss/oob
   200→30 (contact stays 200). Approach-then-miss ≈ hover; only *passing* pays.
Isolated proof: the proven inc7 reward on the ego obs still failed (obs was binding, not reward); RC1+RC2
alone still failed (reward also needed). All three committed (`5a252a4`), redeployed, ran.

## FINAL STATE (morning) — ego now STRUCTURALLY WORKS; last blocker = aperture centering
Two more reward iterations, each **measured** (TB terminal rates), converged the picture:
- **ego_b** (rw_progress=6, miss/oob=**30**): **96% out-of-bounds** at ~2 s. I'd wrongly lumped OOB with
  MISS and made both cheap → no pressure to stay in the arena. Fixed: **OOB back to 200** (leaving the
  arena stays discouraged), MISS stays 30 (a wide but in-arena gate attempt stays forgiving). Committed `3566799`.
- **ego_c** (rw_progress=6, miss=30, oob=**200**): **OOB fixed → 8%**, but now **90% MISS** — the drone
  **reaches the gate and flies wide** (collision ~2%, prog_reward high, banked≈50). It is *approaching the
  visible gate and staying in bounds* — it just can't thread the 1.5 m aperture.

**So the ego went from "completely blind/unlearnable" to "sees the gate, approaches it, stays in the arena,
near-zero contact." The ONLY remaining blocker is aperture CENTERING.** Root cause (the workflow's
inc7-compare agent flagged this early): the refined-B reward **dropped inc7's dense `through_centering`
to-center lateral pull**. `centering_passage_reward` fires only ONCE at the crossing; `segment_progress`
only zeroes perpendicular drift (no *pull* back to the axis). So the drone has NO dense alignment signal
during approach → drifts laterally → arrives off-center → wide flyby → MISS.

### RECOMMENDED NEXT STEP (a deliberate reward-DESIGN choice — left for you, not guessed overnight)
Add a **dense centering term** to `rl/ego_reward.py` (mirror inc7's `through_centering`): penalize the
perpendicular offset from the current gate-centre segment axis each step (reuse `segment_arc_position`'s
projection: `perp = (pos-seg_start) - s·û`). Suggested SAFE form (non-farmable): `-rw_centering ·
clamp(‖perp‖, max≈3 m)`, small weight, wired into `compute_ego_reward`, unit-tested (the reward fns are
pure/local-testable). Pair with **rw_progress 6→~2–3** (6 over-rewards rushing: banked≈50 >> finish=20, so
the drone optimises approaching over passing — make passing dominate, e.g. also raise `rw_finish` toward
~60–80). Then re-run a single_gate diagnostic (`rl/ego_diag.sbatch`, RUNTAG=<tag>, EXTRA=the noise-ceiling
knobs) BEFORE the ladder; watch `metrics/miss_rate` fall + `success_rate` rise.

### How to run (all wired + proven tonight)
- Ladder: `sbatch --export=ALL,SEED=0,RUNTAG=<tag> .../rl/peregrine_vq2_ego.sbatch` (camera-flip default-ON,
  noise-ceiling + reward in BASE/_COMMON). single_gate is FLIGHTCHECK-gated (won't waste the hard stages).
- Single-gate diagnostic: `sbatch --export=ALL,RUNTAG=<tag>,EXTRA='+algo.noise_anneal=true
  +algo.noise_std_hold=0.30 +algo.noise_std_floor=0.30 +algo.noise_hold_frac=1.0 +algo.noise_entropy_floor=0.0'
  /scratch/network/fl3689/ego_diag.sbatch` (single 1500-upd single_gate stage + trace).
- Deploy edits (no Duo): `scratchpad/deploy_ego.py` (base64 via the serve daemon, hash-verified).
- TB terminal breakdown (the key diagnostic): EventAccumulator on the run dir → `metrics/{oob,miss,collision}_rate`,
  `env_loss/{prog,pass}_reward`, `banked_prog_mean`.

Cluster is IDLE (all my jobs killed — no wasted overnight compute). Commits `668347f`→`3566799` on
`claude/optimistic-chaum-6c893b` (NOT pushed). Full diagnosis in memory (`vq2-rl-commander-2026-07-05.md`).
