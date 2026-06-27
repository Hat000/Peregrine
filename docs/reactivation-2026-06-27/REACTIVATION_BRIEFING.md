# Peregrine — Reactivation Briefing (2026-06-27)

Purpose: re-onboard after dormancy. Understand every component we already built, find
what's trash, and get to a clean engineering baseline **before building forward**.
Built from a full read-only sweep of `src/`, `rl/`, `scripts/`, `cluster/`, and `memory/`.

---

## 0 · TL;DR

- The **core code is lean and load-bearing** — agents found essentially **zero dead code**
  in `src/racer/` and only a handful of orphan candidates in `rl/`. The mess is NOT in the
  source tree.
- The **mess is in**: git branches/worktrees (20 local / 18 remote branches, 7 worktrees),
  `handoff/` (93 MB, 109 dirs), old checkpoints, superseded `.sbatch` files, and **stale/
  contradictory claims in memory**.
- The **single most important open question** is unchanged: *is the graded eval case-A
  (position given) or case-C (self-localize)?* Everything downstream hinges on it.
- The **single biggest engineering gap**: the full case-C loop (policy ⊕ estimator ⊕
  detector) has **never run closed in production** — only a one-shot deterministic spike.

---

## 1 · The system in one page

**What it is:** our entry to the Anduril AI Grand Prix — a Python autonomy stack that flies
a simulated racing drone through a fixed gate sequence, fully autonomously.

**Three machines, non-overlapping roles:**
| Machine | Role | Cannot |
|---|---|---|
| **Local laptop** (Win, this box) | Command center: memory, code edits, `.venv` tests, offline analysis, orchestration | run live sim well; GPU-train |
| **ShadowPC** (Win, rented) | The **only** host that runs `FlightSim.exe` and does live in-the-loop flying / sysID | GPU-train |
| **Adroit** (Princeton SLURM, Linux GPU) | **GPU training only** — RL policy + YOLO detector | run the (Windows) sim |

**Four "models of the drone" — do not conflate:**
1. `FlightSim.exe` (AI-GP Simulator v1.0.3364) — the **real** sim. Runs on ShadowPC.
2. **ShadowPC** — a *machine*, not software. It's the box the sim runs on.
3. **"The twin"** (`src/racer/twin.py`) — our **offline CPU plant model** for controller
   tuning/tests. Under-models live latency ~25% — never trust an offline "it holds."
4. **DiffAero** (`rl/diffaero_dynamics.py`) — GPU-batched analytic dynamics for **RL
   training** (2048 envs, ~89K steps/s). Parity-checked bit-identical to the plant.

**VQ1 vs VQ2:** the two virtual-qualifier rounds (both on the DCL sim).
- VQ1 = PASS/FAIL (complete the course). **We passed 5/5 on our sim** with inc7.
- VQ2 = fastest *valid* time, photoreal 3D-scanned env, aids off.
- **Gate contact = invalid run** (the hard validity rule).

**The binding unknown — case-A vs case-C:**
- **Case-A**: the wire streams position (`LOCAL_POSITION_NED`+`ODOMETRY`). inc7 flies this.
- **Case-C**: wire carries NO position → must self-localize from vision + known gate map.
  inc8 is the case-C target.
- Our practice sim streams position (case-A); the official spec §4.3 implies no position
  (case-C). FAQ says it's all-or-nothing across both rounds. **One test on the official
  wire settles it. Build case-C regardless** (robust superset).

---

## 2 · The stacks (how pieces compose into a racer)

A racer = **policy ⊕ estimator ⊕ detector.**

- **inc7** — current best, **case-A**. `stage1_inc7_actor.pth`, 17-dim obs. Passed VQ1 5/5.
  Brittle to localization noise (never self-localized).
- **inc8** — **case-C** target. 20-dim obs (+3 confidence channels: `c_inplane`, `c_along`,
  `age_norm`). Camera-pointing baked in ("active perception"). **Only recently flew at all**
  after ~5 reward-architecture iterations; first flying = S5 recenter (3/3 seeds, success
  ~0.50–0.54); first det-flyable = noise-anneal retrain (det reach 0.467). **Not yet a
  crowned deployable winner.** Reach-rate improvement is *stashed pending review*
  (`claude/funny-nash-d07d7b`, claims 0.467→~0.72).
- **Estimator chain** — **MERGED** (`05ed750`), gated OFF by default. frame → detect → PnP →
  gate-relative obs (+L) → RewindKF (OOSM, 0.5 s horizon) → NavState → obs. Gate-relative
  framing is load-bearing (map bias cancels exactly; absolute nav is a no-go).
- **Detector** — VQ2 2-model clean ensemble (`282abb9`), beats champion 76→82%, ~26 ms,
  8-keypoint contract. `EnsembleGateDetector` exists but **unwired**; weights gitignored.

**The honest gap:** these three have **never run together closed in production.** A vertical
spike (`95dcc93`) proved the loop *can* close once, deterministically, and de-risked the
emulator-vs-deploy obs gap (#37, obs match ~1e-8). Three wiring blockers remain:
1. `fly_rl.py:561` hardwires `detector=None`/`use_vision=False`/`use_given_position=True`.
2. `estimator_obs.py:67` doesn't build `obs[17:20]` → no 20-dim inc8 ckpt can fly today.
3. `detector.py:33` hardcodes 4 corners vs the merged 8-kpt fork.

---

## 3 · Code map (load-bearing vs candidates)

### `rl/` — RL training/eval/deploy
**Load-bearing (do not touch):** `diffaero_dynamics.py`, `peregrine_racing.py`,
`peregrine_racing_inc8.py`, `peregrine_train_racing.py`, `peregrine_train_inc8.py`,
`peregrine_eval.py`, `contact_true_eval.py`, `inc8_sigmap0_*_eval.py`, `fly_rl.py`,
`submit_rl.py`, all `inc8_*.py` levers (noise_anneal/snapshots/warmstart/critic_width/
reward/estimator_emul), `fix_surrogate.py`, `reference_line_torch.py`.

**Orphan/superseded candidates (MED confidence — verify before acting):**
`peregrine_train.py` (smoke-only wrapper), `spike_vertical_slice.py`,
`tilt_segment_analysis.py`, `replay_obs.py`, `tb_parse.py`. Old checkpoints
`stage1_inc{1,3,4,5,6}_actor.pth`. Historical `peregrine_racing_inc{5,6,7}.sbatch`
(superseded by s13/s14), `peregrine_gate.sbatch`+`run_gate_body.sh`.

### `src/racer/` — vision/estimator/control
**Zero dead code.** All tier-1 load-bearing and test-pinned: `frames.py` (conventions,
BORESIGHT, intrinsics — spec-locked), `state_estimator.py` (6-state KF), `localization.py`
(fix builders), `estimator_obs.py` (obs seam + confidence triple + the **√2 reconciliation
footgun** — keep it at obs-build, not navigator), `kf_rewind.py` (OOSM, case-C only),
`vision/detector.py` + `gate_pose.py` + `association.py` + `synthetic.py`, `navigator.py`.

### `scripts/` + `cluster/`
Nearly all active. Key gates: **`scripts/green_gate.py`** (pre-merge GREEN/RED),
**`scripts/frame_residual_report.py`** (run after every live session — internal checks
can't catch the ODOMETRY conjugation). `cluster/vq2_precision_loss.py` is the load-bearing
VQ2 training lever. Candidate stragglers: `scripts/fit_vertical.py`,
`cluster/yolo_train{,_v3}.sbatch` (likely superseded).

---

## 4 · Faults, holes, and contradictions (the important part)

These are the things most likely to bite or mislead on resume:

1. **case-A/case-C wire provenance UNRESOLVED** and binding — see §1. Resolved by the VQ2
   wire drop (the documented dormancy trigger).
2. **Stale/contradictory memory claims** still littering the files:
   - The **"σ_p0 ≲ 0.08 closure bar" was a real double-count error** (`margin_envelope.py`
     subtracted the drone twice). Corrected bar ≈ **σ_p0_lat ≲ 0.15**; measured 0.15–0.20 is
     **marginal-passing, not the no-go it was mis-called.** Gate-4 is a **reach/pass-rate**
     problem (off the racing line), not sub-cm centering. Older entries still say the old thing.
   - The **"36-dim asymmetric privileged critic"** claim is wrong throughout — the active
     critic is **symmetric (20-dim)**; the stabilizer was never connected (root cause of
     inc8 seed collapse). `algo=ppo` never consumes `get_state`/`state_dim`.
   - The **"pivot off RL to near-field estimator" recommendation is RETRACTED** — RL is the
     right tool for gate-4 reach.
   - **Test-count sentinel drift**: 723/884/933/947/1077 all stale; current = **1091**.
3. **ShadowPC sim-ops tooling stranded off `main`** — the mastered orchestrator
   (`partA_cycle.py`, recorder, vision-cal) lives in `handoff/` on ShadowPC, never pushed.
4. **Full case-C loop never closed in production** (only the one-shot spike). 3 wiring
   blockers (§2). 8-kpt `best.pt` unstaged → real-detector accuracy arm blocked;
   data-staging is critical-path-zero.
5. **Emul-vs-real fidelity is the deepest inc8 risk** — inc8 trains on *calibrated emulated*
   vision, not real YOLO→PnP→KF. emul↔emul gap closed (#37); emul↔real **not validated**.
6. **Submission-safety footguns**: judged path must never send `MAV_CMD 31000` (DQ); use
   `submit_rl.py` (pins inc7); `fly_rl.py` default ckpt is the **retired inc4** — pass inc7
   explicitly if bypassing.
7. **Sign/frame lore** — the ODOMETRY quat is R_y(π)-conjugated; ATTITUDE Euler is
   sign-inverted (use ODOMETRY quat). VQ1/CTBR legacy sign alias is self-consistent —
   **DO NOT "fix" it.** inc8 look-at gains are empirical & counter-intuitive
   (`lookat_g_yaw=-3.0`, `lookat_g_pitch=3.0`).

---

## 5 · Cleanup inventory (proposed — execute in approved batches)

### Git (the biggest mess)
- **7 worktrees.** Keep: main repo dir, `Anduril-cmdr` (holds `main`), this session. Prune
  leftovers: `agent-a2e0b926…` (p2-substrate-diagnose), `agent-a5e34977…`
  (p2-substrate-greengate), `p2-inc8-warmstart`, `wf_43064dee-ce7-1`.
- **8 local branches already merged into main** → safe delete: `calib-v2-apply2`,
  `claude/admiring-leavitt-7c331e`, `claude/interesting-bohr-639309`, `oneoff-regsuite`,
  `p1-calib-v2`, `p2-inc8-rl`, `p2-inc8-rl-harvest`, `worktree-wf_43064dee`.
- **Preserve (memory says so):** `inc8-deterministic-retrain` (live), `claude/funny-nash-d07d7b`
  (stashed reach-rate, pending review). **Now backed up to origin (2026-06-27).**
- **Unmerged, decide case-by-case:** `claude/charming-jemison-b111c5`, `claude/jovial-gagarin`,
  `claude/reverent-shirley`, `claude/stoic-cray`, `claude/stupefied-fermi`, `p2-system-id`,
  `p2-substrate-{diagnose,greengate}`, `worktree-agent-a88bc717`.
- **18 remote branches** — reconcile against the merged-status list; delete the ones merged
  or superseded after confirming nothing references them.

### Files
- **Root clutter (safe delete):** `diag_v{1,2,3}.log`, `eval_v{1,2,3}.log`, `__pycache__/`,
  `.pytest_cache/`. `_spec.txt` (text dup of the PDF — PDF is authority). `demo/*.png`
  (regenerated by `preview_multigate.py`).
- **`handoff/` (93 MB, 109 dirs):** archive the lot to an artifact tarball / release asset;
  keep only what's actively referenced. Verify zero live imports first
  (`grep -r "handoff" src/ rl/ scripts/ tests/`).
- **Old checkpoints/models:** archive `stage1_inc{1,3,4,5,6}` via the artifact pipe.
- **Superseded sbatch:** mark deprecated or archive `peregrine_racing_inc{5,6,7}.sbatch`,
  `yolo_train{,_v3}.sbatch`, `peregrine_gate.sbatch`.

### Memory
- Reconcile the §4 stale claims (σ_p0 bar, 36-dim critic, RL-pivot retraction, test count).
- Confirm `memory/` ↔ `~/.claude` mirror, then push so all devices align.

---

## Appendix — source maps
This briefing synthesizes four read-only subsystem maps produced 2026-06-27. The detailed
per-file tables live in the agent transcripts; key load-bearing/orphan calls are in §3.
