# S15 — Env-coherence redesign + procedural tracks + S1.4 retrain (laptop, 2026-06-10/11)

**Session:** LAPTOP-S15-ENVREDESIGN-RETRAIN (fable + adroit-connector).
**Scope:** Part A env-coherence redesign · Part B procedural track randomization (stack-review
meta-gap #1) · Part C S1.4 retrain on the map-ON plant · Part D eval + delivery.
**Outcome:** *(filled at the end — see §6/§7)*

---

## 1. Part A — the env audit and the redesigned reward/termination

The full design doc lives as the module docstring of `rl/peregrine_racing.py` (a deliverable per
the session brief; every term documents formula, weight, units, what it buys, what breaks if
removed). Summary of the **six audited incoherences** (C1–C6) and their fixes:

| # | Incoherence (S1.3 env) | Mechanism | Fix (S1.4) |
|---|---|---|---|
| C1 | OOB escape was FREE while collision was punished | diffaero PPO bootstraps V(s′) on truncation (`next_done = terminated` only, `algo/PPO.py:62-67`); the quadrotor reward has NO oob term (parent `racing.py:344-352` applies `oob_loss` only in the pointmass branch); the box is invisible in gate-relative obs | OOB = **termination, −25** (= frame strike: both are a lost run live), logged as `oob_rate` |
| C2 | One "collision" class for 3 physically different events; aperture tested at the POST-crossing position (~0.5 m/step at 30 Hz = most of the aperture); lateral/backward post strikes invisible | parent `is_passed` plane-crossing test | **Interpolated crossing point** against twin gate geometry: <0.75 m = PASS · (0.75,1.36] = FRAME COLLISION (−25, terminal) · >1.36 = CLEAN MISS (−15, terminal); frame strikes checked on **all 6 gates, both directions** |
| C3 | `attitude=2.0` taxed ALL tilt incl. the 30–60° racing needs | parent `attitude_loss = roll²+pitch²` | **Hinge tilt**: free below 60° total tilt, quadratic above (`relu(cos60° − R33)²`, w=4). Smoothness/VQ2 style + well-modeled-envelope ONLY (no tilt anomaly exists — sweep 510da24); deliberately NO tilt termination |
| C4 | "jerk" = `‖ω‖` taxed cornering itself | parent quadrotor `jerk_loss` | **Action-rate** `‖Δa‖²` (span-normalized, w=0.25) = command smoothness (stays out of the slew-limit corner where the twin is least faithful) + residual `‖ω‖` at w=0.05 (half the old) |
| C5 | No time pressure beyond γ; loitering free | timeout truncation bootstraps, constant=0 | **−0.02/step time penalty** + **finish-time bonus +1.0/s left** (moderate VQ2 ranking pressure; transfer-first) |
| C6 | `survive_rate` counted OOB as survival; failure modes indistinguishable | parent stats | Outcome stats: `collision_rate / miss_rate / oob_rate / survive_rate(=timeout) / success_rate` + `peak_tilt_deg`, `mean_speed`, `finish_time_s`, `pass_offset_m` |

**Termination set** (PPO done=1, no bootstrap): frame collision | clean miss | OOB | finish.
**Truncation set** (bootstraps V(s′)): timeout, GUI. Coherence core: every bad absorbing event
carries an explicit penalty AND kills the future; the only neutral exits bootstrap honestly.

**Coherence sizing checks** (pinned by `tests/test_peregrine_racing_core.py`):
- collision (25) > passage (10) + max single-step progress (~7 at 0.7 m/step ×10/m) — no bribe.
- passage (10) < miss (15) < collision (25) — skipping a hard gate never pays, but a clean miss
  is cheaper than a frame strike (live: invalid run vs tumble).
- full-episode time cost (0.02 × 1201 ≈ 24) ≈ one terminal penalty; since every failure mode is
  terminal-with-penalty and timeout bootstraps, ending early never beats flying.
- standing-pad pitch (−17.8°) is inside the 60° tilt-free cone — no spawn tax.

Also fixed in passing: `get_state` inherited the parent's looping `% n_gates` gate lookahead
(meaningless teleports past the finish; shape-incompatible with per-env courses) — now
clamped+gathered. The legacy hardcoded `_spawn_quat_xyzw` is replaced by a synthesized
yaw/pitch quaternion (reproduces the measured VQ1 spawn quat to <2e-3, test-pinned).

## 2. Part B — procedural course sampling (rank-impact #1)

`rl/peregrine_course.py` Part 2: `sample_courses(n)` draws per-env 6-gate courses as a heading
random walk. **Ranges are derived from the VQ1 course** (re-derived from the json and pinned by
`tests/test_peregrine_course.py` — VQ1 is interior to every range, so holding it out is a real
generalization test):

| quantity | VQ1 measured | sampling range |
|---|---|---|
| horizontal segment length | 23.7–37.4 m | U[15, 45] m |
| heading change / segment | ≤ ~20° | U[−60°, +60°] (3× VQ1 — insurance for unseen VQ2) |
| descent / segment (+down) | 0.6–10.9 m | U[−3, +12] m, grade-clamped to 0.45 (VQ1 max 0.31) |
| gate yaw vs path bisector | ≤ ~10° | ±12° jitter |
| pad → gate-0 distance | 23.3 m | U[18, 28] m |
| gate 0 above pad | 1.41 m | U[0.5, 2.5] m |
| min pairwise gate/pad separation | — | ≥ 10 m (vectorized rejection redraw) |

Spawns stay **tail-first** (body yaw = gate0_yaw + π): on VQ1 with gate yaws = π this equals the
S1.3 identity-attitude spawn EXACTLY, so the existing `fly_rl.py` virtual-flip deployment works
unchanged. Standing starts (`standing_start_frac`) use the course's own pad (rest, −17.8° pitch,
jitter); other resets spawn 1 m up-course of a random target gate, tail-first w.r.t. that gate.
Per-env OOB boxes from each course's bbox + spawn + 15/12 m margins.

`course_mode=vq1` (default) broadcasts the fixed VQ1 course — the held-out eval and the
backward-compatible default; `+env.course_mode=random` enables sampling (training).

## 3. Part C — the S1.4 training configuration

- **torch backend** (launcher registers it; numpy silently ignores per-env DR — the inc-1 lesson).
- **Map-ON plant via DR**: `+dynamics.dr=true` → super-rate map ALWAYS on; measured bands
  s∈U[0.25,0.35] per-axis, τ∈U[0.015,0.030] s, α_max r/p∈U[200,320] (yaw ×80/260),
  hover ±5%, drag ±30%, latency {0,1,2} control steps. G0 fixed. `g=9.80665`.
- `dynamics.controller.max_normed_thrust=3.765` (live collective ceiling).
- `+env.course_mode=random +env.standing_start_frac=0.3`; reward weights = code defaults
  (`RewardWeights`), all overridable as `+env.rw_*`.
- PPO: n_envs=2048, l_rollout=16, lr 2.6e-3, γ 0.99, λ 0.95, 6000 updates, save_freq 200;
  **two seeds (0, 1)** — the S1.3 5000-update run NaN'd once at seed 0.
- **NaN/crash lifelines** (launcher-level, zero clone edits — `rl/peregrine_train_racing.py`):
  `GuardedPPO` skips any optimizer step with non-finite grads (counted, reported);
  unconditional periodic checkpoint every save_freq updates (weights-finite-checked);
  emergency checkpoint on any training exception. Rationale: the stock runner only saves in
  `close()` (skipped on exceptions) and on a success-high-water save gated to `i%save_freq==0`.

## 4. Deployment-relevant changes (next session flies this on ShadowPC)

1. **Checkpoint sidecar** (`rl/fly_rl.py::load_actor`): an optional `<ckpt>.json` next to the
   `.pth` carrying the TRAINED action bounds; applied to `_ACT_MIN/_ACT_MAX` in place. S1.3+
   train with thrust ≤ 3.765 while the old constant was 5.0 — deploying without the sidecar
   overdrives thrust commands up to 33%. The S1.4 checkpoint ships with its sidecar
   (`rl/checkpoints/stage1_inc4_actor.json`). No sidecar → behavior identical to before.
2. **Everything else is frozen**: obs layout/frames, action semantics, tanh+rescale, 30 Hz,
   tail-first + virtual flip (default ON) — `fly_rl.py --flights N` works unchanged.
3. `rl/offline_rollout.py` + `rl/peregrine_eval.py` now default to the **map-ON plant**
   (`--plant flat` remains for flat-trained checkpoints inc-1/inc-3) and mirror the S1.4
   three-way gate-event classification (pass / frame collision / clean miss, interpolated).

## 5. Review discipline

Six-lens adversarial review (env-correctness, reward-exploits, deploy-contract,
sampler-geometry, launcher-guards, eval-scripts; 29 agents), findings independently verified by
refute-first agents (9 verifiers died on a usage limit — those findings were adjudicated by hand).

**Confirmed → fixed:**
1. **[critical] The thrust sidecar was read-only** — nothing ever *wrote* one, so every
   3.765-trained checkpoint (incl. S1.3's, in the repo today) deployed through a [0,5] rescale
   silently: ×1.33 thrust overdrive + corrupted obs[12] feedback, invisible to eval (eval uses
   the training cfg). Fixed: the launcher now wraps `agent.save` to emit `actor.json` next to
   every checkpoint (covers periodic/emergency/best/final in one place); sbatch belt-and-braces;
   `rl/checkpoints/stage1_inc3_actor.json` created; loud warning in `load_actor` when absent.
2. **[major] `offline_rollout.gate_event` graded a backward crossing through the OPEN aperture
   as a collision** (no lower bound on the bwd branch) — diverged from the env and the live sim;
   would have mis-graded acceptance evals. Fixed + regression-tested.
3. [minor ×8, fixed] offline OOB box ~10 m tighter than training behind the pad (now mirrors
   `_update_boxes` exactly); `--virtual-flip` defaulted OFF vs deployment ON (now ON,
   `--no-virtual-flip` to disable); stale ShadowPC default checkpoint paths (now repo-relative
   inc-4); sampler silently ignored misspelled overrides (now raises); periodic saves non-atomic
   (now rotated through `periodic_prev`); NaN-guard could burn wall time forever (aborts after
   200 skips → emergency save); sbatch picked checkpoints lexicographically (now newest-by-mtime);
   eval peak-tilt read POST-reset states (now uses the env's pre-reset `peak_tilt_deg` stat; the
   eval-side peak-roll tracker keeps the old slightly-optimistic semantics for S1.3 comparability
   — documented).
4. [minor, accepted-and-documented] pure lateral (no-crossing) frame grazes remain undetected —
   the C2 docstring now states the crossing-based scope honestly; a capsule-contact model is out
   of scope. Eval on CPU would crash on CUDA checkpoints (no `map_location` in diffaero's
   `agent.load`) — eval is a GPU-node tool; documented.

**Unverified-by-agent, adjudicated by hand:**
- *GAE leaks across truncation boundaries* — REAL, but a pre-existing flaw in diffaero's frozen
  PPO (`nextnonterminal=1` at truncation lets the NEXT episode's advantage flow into the ended
  one). S1.4 *shrinks* its surface: OOB became a termination, so only timeouts leak (rare for a
  competent policy). Not fixable without editing the clone; accepted.
- *Terminal magnitudes vs PPO's 0.2 absolute value-clip* — real slow-critic effect, pre-existing
  (parent returns already span hundreds via progress); same order as before; accepted.
- *Time-in-reward aliasing* (finish-time bonus + time penalty depend on the episode clock, which
  is not in the frozen 17-dim obs) — true for ANY time shaping under this obs contract; weights
  kept moderate (±5-ish at stake vs ~1500 progress scale); accepted, revisit only if curves show
  value-loss pathology.
- *Hover-stall local optimum near hard gates* (hover ≈ −2 discounted beats miss −15 / risky
  attempts) — real in the low-success regime only; passage+downstream value (+10 + hundreds)
  dominates once p(success) ≳ 0.1, entropy bonus pushes exploration; S1.3 reached 1.00 with a
  weaker structure. Accepted; the curves will show it if it bites (success plateau at a gate).
- *R5 pays a spurious dact at step 1* (last_action zeroed by contract) — ~0.12 one-time;
  trivial; contract-frozen; accepted.
- *Outcome flags not strictly a partition* (e.g. simultaneous miss + other-gate strike) —
  co-occurrence is rare and only perturbs the printed histogram; accepted.
- *Sampler rejection-exhaust could return violating layouts silently* — fixed anyway (straight-
  course fallback; a straight course always satisfies separation).

## 6. Training runs + curves

*(TODO: job ids, seeds, durations, tensorboard curve extracts, nan-guard counts)*

## 7. Eval results

### 7.0 Baseline first (S1.3 inc-3 through the NEW laptop pipeline, before the retrain)

`offline_rollout.py` (numpy twin, fly_rl's exact obs/action pipeline, sidecar applied
thrust≤3.765, virtual flip ON, standing start `simstart`):

| plant | outcome | lap | vmax |
|---|---|---|---|
| map-ON (S1.4 default) | **6/6 FINISHED** | 6.63 s | 30.7 m/s |
| flat (its training plant) | 6/6 FINISHED | 6.59 s | 31.3 m/s |
| map-ON + 1-step transport delay | 6/6 FINISHED | 6.46 s | 31.7 m/s |

Read: the flat-trained S1.3 is NOT broken in-twin on the measured plant — the map mostly *adds*
authority and the policy's commands on this course are mostly sub-saturation. The S1.4 retrain's
value is therefore (a) DR breadth = transfer margin, (b) track generalization (rank-impact #1),
(c) reward coherence → style/validity — not rescuing a failing baseline. (Also: these three runs
exercised the new sidecar, map-plant default, and interpolated event classification end-to-end.)

*(TODO: held-out VQ1 map-ON table; random-course generalization; S1.4 offline_rollout;
saturation; acceptance verdict)*

## 8. Checkpoint provenance

*(TODO: run dir, update count, md5, sidecar, commit)*
