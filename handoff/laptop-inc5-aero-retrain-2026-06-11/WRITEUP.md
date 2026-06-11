# INC5 — aero-ON retrain (the definitive VQ2 candidate line), 2026-06-11

**Session:** LAPTOP-INC5-AERO-RETRAIN (fable + adroit-connector).
**Scope:** the deferred V100 parity gate → inc5 = the proven S1.4 recipe retrained with the
S16 measured aero ON (`+dynamics.dr_aero=true`) → S15 acceptance gates aero-ON → ship.

**OUTCOME (all acceptance gates met): `rl/checkpoints/stage1_inc5_actor.pth`**
(md5 `bd1d670f7878eb938119ea173379e7d5` + launcher-emitted sidecar) — the round-2 tilt-96
seed-1 policy, trained on the FULLY MEASURED plant (map + slew + quad body drag + convex
collective, full DR):
- **Held-out VQ1 aero-ON: sr = 1.000 over 2560 episodes** (zero collisions), median lap
  **9.52 s**, pass-offset max 0.589 m (frame at 0.75 — never even close).
- **Style well inside the envelope:** peak roll median 63.4° / p90 64.5° (gate < ~80°);
  roll/pitch saturation 0.0%.
- **Generalization across random unseen courses: 0.939.**
- **Laptop deploy-pipeline 6/6 in ALL configs incl. +2-step latency** — the only round-2
  candidate to pass (both tilt-48 seeds collide at +2; the S15 latency lesson bites).
- **The incumbent is now known-broken on the corrected plant:** stage1_inc4 evaluates
  0.461 on VQ1 aero-ON (53.9% collisions) and dies at +1-step latency in the deploy
  pipeline — see §4. **inc5 supersedes inc4 as THE transfer candidate**; the planned
  ShadowPC live test should fly inc5 (inc4 at most as a comparison datum).
- In-twin success ≠ transfer: live ShadowPC verification is still the next gate.

---

## 1. The deferred V100 parity gate — PASS (first action, before any training)

Code synced to `/scratch/network/fl3689/peregrine_repo` first: full tree-diff of `rl/` +
`src/racer` against local git HEAD by md5 (git-blob LF form), 45 files pushed chunked-base64
over the x daemon, every md5 verified (`.inc5_ref/sync_inc5.py`). The remote copy had been a
minimal file-set: most of `src/racer` was never there, and several `rl/` files were stale
(S15's post-launch review fixes + S16). Remote now mirrors HEAD for the whole training/eval
surface.

SLURM job **3267545** on **adroit-h11g3, Tesla V100-PCIE-32GB** (gres `gpu:tesla_v100:1`),
torch 2.5.1+cu121; md5 tripwire matched (`cdc3f57eb888096c840f146717741e3e`):

| config | DIV_FLOAT64 | DIV_FLOAT32 (advisory) |
|---|---|---|
| legacy | 1.332e-15 | 1.073e-06 |
| super_rate | 2.665e-15 | 1.192e-06 |
| delay2 | 8.882e-16 | 1.073e-06 |
| map_delay | 1.776e-15 | 1.192e-06 |
| **aero** | **1.332e-15** | 1.192e-06 |
| **aero_full** | **1.776e-15** | 1.192e-06 |

**GATE_PASS** — worst float64 2.665e-15 < 1e-9 over 6 configs × 6 seeds × 8 steps, the
historical ~1e-15/1e-16 family as S16 predicted. The torch training plant is algebraically
faithful to numpy rl_plant including the quad body drag + convex collective.

## 2. Pre-training validation (cheap, before burning GPU-hours)

1. **Login-node precheck with the inc5 overrides** (`peregrine_racing_precheck.py
   +dynamics.dr_aero=true` — the script now accepts extra hydra overrides from argv):
   PRECHECK_DONE with **DR AERO ON** and exactly the Section-7 bands —
   quad c2 fwd [0.0325, 0.0521] / climb [0.0583, 0.0942] (per-slot relative), K(1.0)
   [71.6, 84.9] m/s², hover-knot [9.39, 9.77] (±2% pin), d1∈[0.001, 0.080] absolute,
   hover conversion PINNED (the legacy ±5% hover DR correctly inert), latency {0,1,2},
   GuardedPPO + course sampler + VQ1 holdout all live.
2. **GPU smoke** (job 3267563, 60 updates, 512 envs): learning signal present on the aero
   plant (success 0 → 0.12 by update 60 from scratch), ~18.3K steps/s at 512 envs,
   checkpoint + launcher-emitted sidecar written, both chained `--plant aero` evals ran.
   **Known cosmetic failure:** diffaero's `runner.close()` ONNX export crashes AFTER
   actor.pth + exported_actor.pt2 are written → `TRAIN_RC=1`. Verified pre-existing: every
   S14/S15 job including the shipped stage1_inc4 winner (3267360) ended the same way.

## 3. Code delta this session (committed on main)

- `2de4794` — `peregrine_eval.py` / `offline_rollout.py` gain **`--plant aero`** = the fully
  measured plant (super-rate map + slew + quad body drag + convex collective, linear_drag=0;
  the S16 fixed-params form). Eval scripts otherwise default to map-ON-only — every inc5
  eval must pass `--plant aero` explicitly. + `rl/peregrine_racing_inc5.sbatch` (the s14
  round-3 validity config as DEFAULTS + `+dynamics.dr_aero=true`, chained aero evals).
- `afabc2a` — precheck accepts extra hydra overrides from argv.
- 528 local tests green after the edits (and the aero PlantParams construction smoke-tested).

## 4. Why inc5 (datums collected this session, laptop numpy twin, fly_rl deploy path)

stage1_inc4 (aero-OFF-trained, the incumbent) through `offline_rollout.py --start simstart`,
sidecar 3.765, virtual flip ON:

| plant | latency | outcome | lap | vmax |
|---|---|---|---|---|
| map (its training plant) | 0 | 6/6 FINISHED | 7.16 s | (S15 §7) |
| **aero** | 0 | 6/6 FINISHED | **9.19 s** | 27.3 m/s |
| **aero** | +1 step | **COLLISION after gate 0** | — | 20.7 m/s |
| **aero** | +2 steps | **COLLISION at gate 0** | — | 21.5 m/s |

Read: on the corrected plant the aero-blind policy is ~2 s slower AND loses its entire
latency margin (on the map plant it survived +2 steps at 8.03 s). The quadratic drag changes
where braking must start, and inc4's learned margins are wrong by ~2× at speed — exactly the
S16 prediction ("policies ~2× under-braking"). The ~8 g convex-collective headroom also goes
unexploited. inc5 trains with the measured aero + the same latency DR jointly.

And the in-twin eval (job 3267570, `peregrine_eval.py --plant aero`, spawn-jittered standing
starts, DR off-nominal) is far harsher than the single deterministic rollout above:

| inc4 evaluated on | VQ1 sr (n_ep) | VQ1 t_med | coll% | random-course sr |
|---|---|---|---|---|
| map plant (S15, its training plant) | 1.000 (3582) | 7.03 s | 0.03% | 0.845 |
| **aero plant (this session)** | **0.461 (3240)** | 9.92 s | **53.9%** | **0.386 (3594)** |

**The incumbent fails the corrected sim half the time.** Surviving episodes shave the gates
at p90 pass-offset 0.645 m (frame at 0.75) — under-braking turned the style into gate-clipping.
The aero retrain is a correctness fix, not an optimization.

## 5. Training — inc5 (RUNNING)

Jobs **3267568 (inc5_s0, seed 0)** and **3267569 (inc5_s1, seed 1)**, both on adroit-h11g3
(V100), launched 2026-06-11 ~11:40 EDT. Config = `rl/peregrine_racing_inc5.sbatch` defaults =
the stage1_inc4 winner recipe exactly, plus aero:

- env: `course_mode=random`, `standing_start_frac=0.3`, rw_tilt 16, rw_collision 75,
  rw_miss 40, rw_oob 75, rw_finish_time 0.25, rw_dact 1.0 (other RewardWeights at code
  defaults: progress 10, passage 10, finish 20, time 0.02, tilt_free 60°, rate 0.05).
- dynamics: torch backend, `+dynamics.dr=true +dynamics.dr_aero=true`,
  `max_normed_thrust=3.765`, `g=9.80665`.
- PPO: n_envs 2048, l_rollout 16, lr 2.6e-3, γ 0.99, λ 0.95, 6000 updates, save_freq 200,
  GuardedPPO + periodic/emergency checkpoints + launcher-emitted sidecar.
- Datum job **3267570**: inc4 evaluated `--plant aero` on VQ1 + random (the eval-table
  comparison rows).

### Round 1 (jobs 3267568 inc5_s0 / 3267569 inc5_s1, the unmodified inc4 winner weights)

Both completed 6000/6000 in ~45 min wall (V100, ~74K steps/s at 2048 envs), zero NaN-guard
hits, zero obs_nonfinite (the S15 clamp holds on the aero plant). Training success plateaued
~0.84 by upd ~2800 (same trajectory as the inc4 winner). All evals `--plant aero`,
spawn-jittered standing starts, DR off-nominal:

| ckpt | VQ1 sr (n_ep) | VQ1 t_med | VQ1 coll% | gen sr | roll med/p90 (succ) | tilt med/p90 | sat thr/r/p/y % |
|---|---|---|---|---|---|---|---|
| inc4 datum (aero, 3267570) | 0.461 (3240) | 9.92 | 53.9 | 0.386 | 57.7/65.2° | 82.4/85.5° | 93/1.4/0.4/97 |
| inc5_s0 (md5 8888576b…) | 0.862 (3881) | 6.56 | 13.8 | 0.944 | 103.5/147.5° | 91.7/94.0° | 97/0.3/0.0/91 |
| inc5_s1 (md5 ab6cf4b1…) | **0.996 (3584)** | **6.89** | 0.4 | **0.960** | 85.8/145.3° | 89.2/93.9° | 97/2.4/0.3/92 |

**Read:** the aero retrain works — inc5_s1 beats inc4's MAP-plant numbers on the HARDER
plant (lap 6.89 vs 7.03 s, gen 0.960 vs 0.845, mean speed 24–25 vs 15.8 m/s: the policy
exploits the real braking + the ~8 g collective). **But the style gate fails decisively:**
roll p90 ~145° = knife-edge cornering, vs the < ~80° envelope. The corrected plant roughly
doubled thrust authority, so speed got cheap and the w=16 tilt hinge stopped binding — the
S15 round-2 phenomenon on a new axis. Seed 0 additionally converged risk-hungry (13.8%
VQ1 crashes; seed variance is back at tilt 16).

inc5_s1 deploy-pipeline matrix (laptop `offline_rollout.py`, fly_rl's exact obs/action path,
aero plant, sidecar 3.765, virtual flip ON): **6/6 in ALL configs** — nominal 6.43 s ·
+1-step latency 6.93 s · +2-step 8.09 s · handoff@10 m/s 5.76 s · live-thrust-clip 6.43 s
(vmax ~34.6 m/s). Latency-robust where inc4-on-aero collided at +1 step. → banked as
**fallback candidate** (`.inc5_ref/ckpt_candidates/inc5_r1_s1_actor.pth`), fails ONLY style.

### Round 2 (RUNNING): tilt-hinge sweep, weights-only per the session authorization

Jobs **3267803 (inc5_t48_s0) / 3267804 (inc5_t48_s1) / 3267805 (inc5_t96_s1)** —
identical config, `RW_TILT` {48, 96} (3×/6× round 1's 16). Hypothesis: the hinge weight must
scale with the plant's thrust authority for the same ~80° operating point; collision/validity
weights already at the S15 round-3 values and seed 1 hit 0.4% crashes, so style is the one
axis being moved.

### Round 2 results — tilt 48 brackets style, tilt 96 buys the latency margin

All three completed 6000/6000 (~50 min wall each, ~73K steps/s), zero NaN events. Curve
milestones (training sr on random courses, DR fully ON): sr 0.5 by upd ~165–180, sr 0.8 by
upd ~176–280, finals 0.84 / 0.82 / 0.87 (t48_s0 / t48_s1 / t96_s1) — heavier style penalties
did NOT slow convergence; t96_s1 posted the highest plateau. Evals all `--plant aero`:

| ckpt (md5) | VQ1 sr (n_ep) | VQ1 t_med | coll% | gen sr | roll med/p90 | tilt p90 | pass-off p90/max |
|---|---|---|---|---|---|---|---|
| t48_s0 (82e4d1b8…) | **1.000** (2560) | 9.22 | 0.0 | **0.960** | 63.4/67.6° | 71.6° | 0.475/0.725 |
| t48_s1 (23c935c5…) | 0.988 (2849) | 8.69 | 1.2 | 0.952 | 60.2/65.6° | 79.1° | 0.555/0.750 |
| **t96_s1 (bd1d670f…) = stage1_inc5** | **1.000** (2560) | 9.52 | 0.0 | 0.939 | 63.4/**64.5°** | 67.5° | **0.355/0.589** |

Deploy-pipeline matrix (laptop `offline_rollout.py`, aero plant, sidecar, vflip ON):

| start / config | t48_s0 | t48_s1 | **t96_s1** |
|---|---|---|---|
| standing start (simstart) | 6/6 8.89 s | 6/6 8.29 s | 6/6 9.36 s |
| +1-step latency | 6/6 9.19 s | 6/6 8.42 s | 6/6 9.72 s |
| **+2-step latency** | **COLLISION (2 gates)** | **COLLISION (3 gates)** | **6/6 10.39 s** |
| racestart | 6/6 8.92 s | 6/6 8.19 s | 6/6 9.56 s |
| handoff @10 m/s | 6/6 7.99 s | 6/6 8.09 s | 6/6 7.99 s |
| live-thrust-clip | 6/6 8.89 s | 6/6 8.29 s | 6/6 9.36 s |

**Selection: t96_s1.** It is the only candidate passing every gate inc4 passed — the +2-step
latency case is non-negotiable given the documented "offline twin under-models live latency
~25%" lesson (live ≈ 1–1.5 steps; +2 is exactly the margin case). Cost vs t48_s0: +0.30 s
VQ1 median and −0.021 generalization; bought: the latency margin, the tightest gate
clearances (max pass-offset 0.589 m), and the lowest tilt envelope. t48_s0 (faster,
style-compliant, latency-fragile) is banked in `.inc5_ref/ckpt_candidates/` + on Adroit as
the speed-alternate if the S2 architecture work wants a hotter datum.

**Style-vs-speed note for S2:** enforcing the <80° envelope on the aero plant costs ~2.3 s
of VQ1 lap (round-1 unconstrained: 6.89 s @ 145° roll p90; round-2: 9.2–9.5 s @ ~65°).
That tension is real physics (quad drag punishes the wide fast line; knife-edge cornering
is the fast solution) and is a key input to the S2 monolithic-vs-decomposed decision and to
any future relaxation of the envelope once live data validates the high-tilt regime.

## 6. Acceptance verdict

| gate | requirement | result |
|---|---|---|
| parity | V100 gate re-passed incl. aero configs BEFORE training | GATE_PASS 2.665e-15 (§1) ✓ |
| held-out VQ1 success | ≥ inc4's 1.000, aero-ON | **1.000 over 2560 eps, 0 collisions** ✓ |
| style envelope | peak roll p90 < ~80°, saturation low | roll p90 64.5°, r/p sat 0.0% ✓ |
| generalization | nonzero across random courses | 0.939 ✓ |
| deploy rollouts | 6/6 standing start, latency, thrust clip | **6/6 ALL configs incl. +2-step** ✓ |
| tests | all green | 528 passed (re-run after all session edits) ✓ |

Reward adjustment used this session (authorized, weights-only, documented): `rw_tilt`
16 → 96 (round-2 sweep {48, 96}; 48 fixed style but not latency margin). Everything else
identical to the inc4 winner config. No structural env change was needed.

## 7. Checkpoint provenance (exact)

- **Artifact:** `rl/checkpoints/stage1_inc5_actor.pth`, md5 **`bd1d670f7878eb938119ea173379e7d5`**
  + sidecar `stage1_inc5_actor.json` = `{"act_max_thrust": 3.765, "act_max_rate": 3.14}`
  (launcher-emitted at save time, md5 `beb64dd33775ba3a61719596f0f0144c`).
- **Source:** Adroit SLURM job **3267805**, run dir `/scratch/network/fl3689/inc5_runs/inc5_t96_s1`,
  final `checkpoints/actor.pth` after the full **6000 PPO updates** (196.6M env-steps), seed **1**.
  Pulled chunked-base64 over the x daemon, md5-verified end-to-end; shipped-path rollout
  re-verified (sidecar auto-load line + 6/6 @ 9.36 s).
- **Training config:** `rl/peregrine_racing_inc5.sbatch` @ commit `afabc2a` with
  `SEED=1 RW_TILT=96 TAG=inc5_t96_s1` (defaults supply the S15 round-3 validity weights:
  coll 75, miss 40, oob 75, ftime 0.25, dact 1.0);
  env: `course_mode=random`, `standing_start_frac=0.3`, other RewardWeights at code defaults;
  dynamics: torch backend, **`+dynamics.dr=true +dynamics.dr_aero=true`** (map ALWAYS on +
  measured aero ALWAYS on with the S16 Section-7 bands: c2 per-slot ×U[0.040,0.065]/0.052,
  K-table affine k~U[0.90,1.10] hover-pinned h~U[0.98,1.02], d1~U[0,0.08] absolute,
  s∈U[0.25,0.35]/axis, τ∈U[0.015,0.030], α_max r/p∈U[200,320] yaw×80/260, latency {0,1,2}),
  `max_normed_thrust=3.765`, `g=9.80665`; PPO: n_envs 2048, l_rollout 16, lr 2.6e-3, γ 0.99,
  λ 0.95, 8 minibatch × 4 epoch, 6000 updates. Env/plant code = commit `eba4349` (S16) +
  `2de4794`/`afabc2a` (this session), synced md5-verified to the Adroit file copy.
- **Alternates (not committed; `.inc5_ref/ckpt_candidates/` + Adroit `inc5_runs/`):**
  `inc5_t48_s0` md5 `82e4d1b8…` (VQ1 1.000 @ 9.22 s, gen 0.960 — faster, FAILS +2-step
  latency); `inc5_t48_s1` md5 `23c935c5…` (0.988 @ 8.69 s); round-1 `inc5_s1` md5
  `ab6cf4b1…` (0.996 @ 6.89 s, gen 0.960 — the unconstrained-style speed datum, roll p90 145°).

## 8. Ops notes (durable)

- **The serve daemon died mid-session** (post-training): x calls hung (TCP accepted, channel
  dead), then TimeoutError/ConnectionReset. Two zombie `serve` processes had accumulated —
  kill ALL, relaunch via the S15 keeper, user approves the Duo push. Probe liveness with a
  trivial `echo` BEFORE any long command; treat client-side TimeoutError as "daemon dead or
  busy", not "command failed".
- **PowerShell→x quoting strikes again:** `$j` interpolation and multi-word grep args both
  mangled inline commands. ANY non-trivial remote command goes as a base64'd script file
  (`echo <b64> | base64 -d > f && bash f`) — now the session default.
- **Live training watch without tensorboard:** `.inc5_ref/watch_training.py` — one light
  x call/min (squeue + log tails), parses the diffaero progress bar, prints per-job
  update/sr/loss/fps + ETA + sr sparkline, appends CSV. Run
  `.venv\Scripts\python.exe -X utf8 .inc5_ref\watch_training.py` in its own terminal
  (needs `-X utf8` on Windows). Doubles as a daemon-health indicator.
- diffaero `runner.close()` ONNX export crashes post-checkpoint on every run (`TRAIN_RC=1`
  cosmetic) — verified pre-existing on all S14/S15 jobs incl. the shipped inc4.

## 9. Recommendations for the next session (live ShadowPC)

1. **Fly inc5, not inc4.** `fly_rl.py --flights N --checkpoint rl/checkpoints/stage1_inc5_actor.pth`
   (the fly_rl DEFAULT still points at inc4 — left unchanged deliberately; flipping the
   default is the commander's call given the live-test plan named inc4). Watch for the
   `[load_actor] sidecar` line. Expected if transfer holds: ~9.5–10.5 s laps, peak roll ≲65°,
   thrust/yaw pinned (normal), clean gate centers (median clearance ~0.19 m in-twin).
2. inc4's live value is now only as a controlled comparison datum — it is KNOWN-broken on
   the corrected plant (46% VQ1, dies at +1-step latency). If it is flown first per the
   original plan, expect crashes and do not burn attempts tuning it.
3. The corner-pass probe (TOGT queue) folds in unchanged.
4. Failure triage order stays S15's: sidecar applied? → virtual flip on? → replay the live
   handoff state through `offline_rollout.py --plant aero` — offline-pass/live-fail points
   at telemetry/timing, not the policy.
5. **TOGT bound re-solve with the corrected aero** (3.77 g → ~8 g + quad drag) before the
   S2 architecture decision; the current 4.55 s bound is built on the falsified thrust map
   and the style-vs-speed datum above (§5) is a direct S2 input.
