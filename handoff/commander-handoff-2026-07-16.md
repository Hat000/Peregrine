# RL-COMMANDER HANDOFF — 2026-07-16

*Written by the outgoing RL commander for the incoming one. Read this AFTER `COMMANDER.md` + `memory/MEMORY.md`, not instead of them. Everything here is state + judgment as of 2026-07-16; the memory files are the SSOT and win on conflict.*

---

## 0 · WHO YOU ARE, IN ONE PARAGRAPH

You are the **RL commander** of Peregrine (Anduril AI Grand Prix drone-racing entry, `github.com/Hat000/Peregrine`). Per the **rev-3 domain split** (MEMORY.md): there are TWO commanders — a **vision-stack commander** (ShadowPC side) and **you, the RL commander** (laptop). Each is **sole memory authority over its own domain**. Commander↔commander relays go **through Fengyou, as ONE copy-paste code box**. `COMMANDER.md` (repo root) is the boot prompt + durable craft; **read it first, edit it last** (add the lesson, prune the stale). `memory/MEMORY.md` = project STATE index (thin — follow the `[[pointers]]`).

**Fengyou** (fl3689@princeton.edu) is the human lead, relay, and launcher. He owns strategic forks. **Address him by name in EVERY message** — that's the canary; a missing name means context degradation and he should rotate the session.

**Your job is judgment, diagnosis, adjudication, and catches** — not heads-down execution. Delegate multi-file builds to agents/Workflow; keep the thread for the cross-cutting call. Ethos from COMMANDER.md, and I'd underline it after this week: **diagnose before fixing; build the cheap check before the expensive run; be honest fast when you're wrong.** This handoff exists largely because I *was* wrong about something central and caught it late (§2).

---

## 1 · WHERE WE ARE (the 30-second version)

- **Mission clock:** VQ2 qualification closes ~mid/late July 2026. Strategy = **BANK-FIRST** (finish 20 gates at ANY speed → bank a time; §9.2/9.4 is a pure time-trial with unlimited attempts, so a fast-invalid run is free). Speed comes after a banked lap.
- **The deploy state:** `vpeffs0` ("ffs0") is the **champion / warm base** — clean 2-gate + DR, flies ~5 gates smooth in deploy. Its **ceiling = SPEED ACCUMULATION** (per-gate 3→13.7 m/s, goes OOD ~gate 4-5, banks 96-121° and can't turn → crash). The 8-gate fine-tune (`vpef8nc`) **ADDED a yaw hunt** (rail-flip limit cycle) — that was a *recipe* regression, not a capability gain.
- **The next experiment is `v1`** — a retrain, fully assembled, dry-run-verified, and **gated ONLY on Adroit GPUs** (they were `drng@` / draining for maintenance; nothing else blocks it). See §3 for everything needed to fire it.
- **A central diagnosis was overturned on 2026-07-16** (§2). Read that before you touch the plant.

---

## 2 · 🛑 READ THIS FIRST — THE DIAGNOSIS I GOT WRONG

**The claim (now RETRACTED): "DiffAero trained a FLAT rate-gain plant; real VQ2 is EXPANSIVE; that mismatch caused the g2→g3 over-rotation wall; `faithful_rate` is the fix."** That framing is in a lot of older memory text. **It is wrong.**

**What's actually true** (verified by an adversarial workflow + a code audit, 2026-07-16):
- `rl/diffaero_dynamics.py:617` — under `dr=true` (which **every** ffs0/vpef8nc/v1 run uses), DR initializes `_dr_s = SUPER_RATE_S_MEASURED = 0.30` and resamples `[0.25, 0.35]`; the torch step reads **`_dr_s`** (`:838`), **not** `params.super_rate_s`. So **the expansive super-rate is ALWAYS ON under DR, regardless of `faithful_rate`** — and it even ignores faithful's per-axis super_rate values.
- That code is committed **`b14ca2e` (2026-06-10)** — a **month before** ffs0/vpef8nc trained (Jul 11–14). So they trained **expansive**, not flat.
- Quantified on the real VQ2 yaw ampsweep (`handoff/rl-commander-2026-07-16/plant_val.py`):

| yaw gain bin | VQ2 (real) | flat (DR-**off**) | **DR-nominal (what trained)** | faithful |
|---|---|---|---|---|
| low | 2.13 | 2.20 | **2.31** | 2.25 |
| mid | 2.33 | 2.20 | **2.57** | 2.51 |
| high | 2.52 | 2.20 | **2.76** | 2.72 |

  Per-tick yaw-RMSE (lag-2 aligned): flat-DRoff **0.68** · DR-nominal **0.396** · faithful **0.387**.

**Implications you must carry:**
1. **`faithful_rate` under DR = ONLY a ~5% base-gain (G0 2.5→2.36) reduction** → ~**2%** RMSE improvement over what actually trained. It's a **marginal, correct-direction polish. KEEP it** (G0=2.36 *is* the best VQ2 small-signal match, it's free and safe) — **but it is NOT the wall/oscillation fix.** My earlier "−29%" was measured against a DR-**off** flat baseline **training never used**. Don't repeat that error.
2. **The wall is NOT a rate-gain problem.** The un-faithful training plant *over*-predicts gain ~8% → policy under-commands → deploy **under**-rotates. That's the *opposite* of the observed over-rotation wall.
3. **The real levers are NON-plant** (and consistent with the older, still-valid isolation "8-gate fine-tune ADDED the yaw hunt" + "ceiling = speed accumulation"): **anti-dither** (yaw hunt) · **ego_vision_cadence** (vision staleness) · **recovery arm** (accumulation/over-bank). **v1's value rests on those three.**
4. **Do NOT spend more effort tuning the plant.** It already matches VQ2 within ~8%, on the safe side.

**The ONE uncertainty that would flip this:** confirm **ffs0's resolved `.hydra/config.yaml` shows the DR super-rate active in its actual run** (i.e. the Adroit `peregrine_repo` was synced ≥ `b14ca2e` at train time). Circumstantial evidence is strong (code committed a month prior + the sbatch's pre-sync ritual + heavy July-on-Adroit work post-dating June 10). **Check this when Adroit returns.** If ffs0 somehow trained on a stale pre-June-10 flat plant, `faithful_rate` *is* the real fix and §2 partially un-retracts.

**Older memory text still carries the flat-plant framing in places.** `memory/perception-sim2sim-gap-2026-07-15.md` has a prominent CORRECTION block at the top of the validation section — that block supersedes the older claims in the same file. I did not rewrite every occurrence (too large); **trust the correction block.**

---

## 3 · THE v1 RETRAIN — EVERYTHING TO FIRE IT

**What v1 is:** warm from ffs0 → stage `dual_gate_fullstack_floor_pef` → 8-gate course, with the three real levers + the free plant polish. NO att-cap, NO velocity cap, NO roll fence (fence only the axis the course doesn't use).

**Launcher:** `handoff/rl-commander-2026-07-16/launch_v1.sh` (preserved copy; 208 lines, LF, syntax + both-phase dry-run verified against a stubbed `sbatch`). Must sit **next to `peregrine_vq2_ego.sbatch`** (it resolves `SBATCH_FILE` via `$(dirname $0)`) → deploy to `/scratch/network/fl3689/peregrine_repo/rl/launch_v1.sh`.

**Enables:** `++dynamics.faithful_rate=true` · `++env.ego_vision_cadence=true` (sub-knobs `ego_vision_frame_hz=30.0`, `ego_vision_detect_p=0.35` → ~10.5 Hz valid fixes; `ego_vision_cadence_seed=20260715+SEED` per-seed decorrelated) · aw1 anti-dither (`rw_yaw_dither=0.125` matched to the 0.7 clamp; **`yaw_dither_start=1.0` = dither-from-birth**, since we warm from already-smooth ffs0) · `UPD=18000`.

**Arms:** `A` seeds 0,1 (core) · **`R` seed 0** = A + recovery/damping (`rw_roll_recover=0.5`, annealed, form-A `theta0=30°`; tunable via `RECOVER=`; recovery tokens are **per-arm** to sidestep the `recovery_anneal` both-zero guard) · `C` seed 0 = A + `dr_latency_max_steps=5`. Trim with `ARMS="A R"` if the A100 pool is tight.

**TWO-PHASE launch (deliberate — do not collapse it):**
```
bash launch_v1.sh                              # phase 1: ONE UPD=100 smoke, no dependency
# ADJUDICATE the COMPLETED smoke (a DONE log, never a live one):
#   1. grep faithful_rate      <rundir>/.hydra/config.yaml   -> `faithful_rate: true`
#   2. grep ego_vision_cadence <rundir>/.hydra/config.yaml   -> `true`
#   3. scratchpad/tb_parse.py <event-file>  -> env_loss/yaw_dither_pen PRESENT + NONZERO
#   4. grep EGO_PRECHECK_RC .../peregrine_vq2_ego_v1smoke_s0.out  -> == 0, no NaN
MODE=arms SMOKE_JID=<smoke_jobid> bash launch_v1.sh    # phase 2: arms --dependency=afterok
# escape hatch (no dependency): MODE=arms SMOKE_JID=none bash launch_v1.sh
```
The `.hydra/config.yaml` grep is the clean answer to the **absent-hook ≠ inert** footgun: a force-added `++` key that *bound* resolves into the config tree. If it's missing there, it silently no-op'd.

**PRE-SYNC to Adroit — 5 files, or a stale copy silently trains the wrong code:**
`rl/diffaero_dynamics.py` · `rl/peregrine_racing_ego.py` · `rl/peregrine_train_ego.py` · `src/racer/rl_plant.py` (all **UNCOMMITTED**, see §4) · **`rl/ego_reward.py`** — recovery is committed at HEAD `9711b4c`; **verify the Adroit copy is ≥ 9711b4c or arm R's `rw_roll_recover` is silently INERT** — plus the launcher.

**Deploy-pick (post-hoc, NOT an in-training flag — the in-training promote is n_passed-only):** `eval_yaw_log=true` emits `YAW_EVAL signflips_per_s` on final + promoted-best. **Reject any ckpt > ~4 flips/s REGARDLESS of n_passed**, then rank survivors on DET `n_passed_gates`; tiebreak lowest signflips + roll_swing. **NEVER select on value.** For per-ckpt yaw numbers, run the `++rollout_only` harness over periodic + `best_npg/` snapshots.

**Monitor TB, not stdout** (`scratchpad/tb_parse.py` on the run's event file). Watch `env_loss/yaw_dither_pen` converge down + `n_passed_gates` climb; arm R's `roll_recover_pen` should shrink.

**Expectation setting (post-§2):** success = the yaw hunt gone (signflips → ~0) while n_passed climbs past vpef8nc's DET 5.33, and the late-gate accumulation damped. **Credit/blame anti-dither + vision_cadence + recovery — not faithful_rate.**

---

## 4 · 🚩 OPEN ISSUES / RISKS (ranked)

1. **THE v1 CODE IS UNCOMMITTED.** Worktree `cool-heyrovsky-e08624`, branch `vtrackAr5-recovery-reward`, HEAD `9711b4c`. Modified: `rl/diffaero_dynamics.py`, `rl/peregrine_racing_ego.py`, `rl/peregrine_train_ego.py`, `src/racer/rl_plant.py`; untracked: `tests/test_faithful_rate_plant.py`, `tests/test_ego_vision_latency.py`, `tests/test_ego_obs_v2.py`, `handoff/rl-commander-2026-07-16/`. **`.claude/worktrees/*` RECYCLES** → this work can be LOST. **Ask Fengyou to authorize a commit + push before anything else.** (Standing rule: commit/push only when he asks — so *ask*.) Backup patch (partial, pre-obs_v2): `scratchpad/wip_before_coarsemap.patch`.
2. **Adroit GPUs down** (`drng@`, maintenance) — v1's only gate. Jobs auto-run when nodes return.
3. **Verify ffs0's `.hydra` DR-super-rate** (§2's one uncertainty) when Adroit returns.
4. **Aero is deferred to v1.5/v2, deliberately.** The thrust gap is real (~1.8× full-stick) but only **~9% in the actual deploy regime** (the policy cruises near-hover and makes speed by tilting ~27°), and **drag errs SAFE** (deploy over-brakes 2-3×, capping terminal ~10-13 m/s). So the 3→13 m/s accumulation is the drone reaching its **drag-limited terminal**, NOT a thrust runaway. `faithful_aero`'s `coll_map` is **correctly HELD** (over-predicts clean static +20-40%; enabling un-refit would over-thrust). **For v1.5 aero, land the clean QUADRATIC-DRAG term (it sets terminal speed) BEFORE the uncertain thrust map.**
5. **`ego_obs_v2` (windowed coarse map, 21→23 dims) is BUILT + tested but OFF** → v1.5. Fengyou's call: it's a capability-add, not a faithfulness fix; ffs0 flies 5 gates without it, so it'd muddy v1 attribution. **A DEPLOY TWIN IS REQUIRED before ever flying v2 obs** (documented, not built): `src/racer/ego_obs.py` must emit `obs[21:23]=map[active+1]`, and `fly_rl.py` `_EGO_OBS_DIM`→23.
6. **Residual plant nits (do NOT churn on these):** within-hold **creep** (+3-6%; real slightly exceeds faithful in sustained banks — bounded ≤5%, sign-ambiguous); a **dt seam** (training `env.dt=0.0333`/30 Hz vs deploy decision ~36 Hz — gain is dt-invariant so it transfers; only slew-ceiling/lag-ms differ, second-order); `yaw_bw` fast-reversal under-modeled by both plants (real 2.13 vs sim ~1.9).
7. **Transport delay is ALREADY modeled** — don't "fix" it. Measured ~2 control ticks (~56-67 ms); the BASE's `dr_latency {1,2,3}` steps at the 30 Hz training dt centers exactly on it. (An adversarial agent flagged it as unmodeled — that came from a **stale `env.dt=0.02` comment** in `diffaero_dynamics.py`. Consider fixing that comment.)

---

## 5 · RESOURCES MAP

**Memory (your SSOT — you are its sole writer for the RL domain):**
- `memory/MEMORY.md` — thin index; NOW state, directives, footguns. Touch it ONLY for NOW / footgun / directive changes.
- `memory/perception-sim2sim-gap-2026-07-15.md` — **the active thread SSOT** (v1 scope, launcher recipe, the §2 CORRECTION block, plant/vision numbers).
- `memory/reference-vq2-flight-logs.md` — **where every real VQ2 log lives + what's suitable** (I wrote this; it'll save you a day).
- `memory/trackA-stall-forensic-2026-07-13.md`, `memory/track-a-beat-vpeffs0-2026-07-13.md`, `memory/audit-ego-inc9-2026-07-09.md`, `memory/ego-deploy-contract-2026-07-09.md` — prior SSOTs.
- `memory/index_rl_training.md` (your domain sub-index) · `index_control_sim.md` · `index_vision_estimator.md` · `index_strategy_meta.md`.
- `memory/project_parked_backlog.md` — **standing register; review at boot + on any revive-trigger.**

**Harnesses (preserved in `handoff/rl-commander-2026-07-16/` — the originals were in a session-scoped temp scratchpad that does NOT survive):**
- `launch_v1.sh` — the v1 launcher (§3). `launch_huntfix.sh` — its predecessor.
- `plant_val.py` — per-tick IMU-match validation: replays a VQ2 sysID capture's command through the numpy plant (flat vs faithful), overlays gyro + specific-force accel. **This is the instrument that caught §2.**
- `panel_val.py` — racing-panel rate replay. **Read the caveat in §6 before trusting it.**
- `sysid_diffaero_replay.py` — replays `sysid_program.csv` through the real action→plant pipeline, logs full GT state.
- `refit_v2.py` — the super-rate fit used for the faithful constants.
- `wf_plant_verify.js` — the 7-agent adversarial-verification workflow (6 lenses + synthesis). Re-run via `Workflow({scriptPath, resumeFromRunId})`. **Model this pattern** — it earned its keep.

**Data (all fetchable; see `memory/reference-vq2-flight-logs.md` for the full map):**
- **Controlled sysID (THE instrument):** branch `sysid-handoff-2026-07-15`, `data/runs/*_sysid_*_f1/sysid_vq2_log.csv` — 20 captures, **self-aligned** (`a_*` command + `gyro_*` + `accel_*` per tick). `yaw_ampsweep` = the clean stepped money run.
- **Racing (rich, confounded for plant gain):** branch `claude/ego-deploy-2026-07-09`, `data/runs/*_panel_run_f1/ego_obs.jsonl` — commits `2aca009` (vpef8nc oscillation ×5), **`28404fa`** (deep vpeffs0, gate 5 @13 m/s = the accumulation), `c344e9a`, `80dd3a1` (×21).
- **NOT suitable:** `race-wire-capture-2026-07-05` `*.race_wire.jsonl` = RACE_STATUS + COLLISION only.
- Fetch: `gh api "repos/Hat000/Peregrine/contents/<path>?ref=<branch>" -H "Accept: application/vnd.github.raw"`.

**Environment:** venv = the **MAIN checkout** `.venv` (`C:/Users/Fengy/Downloads/Projects/Anduril/.venv/Scripts/python.exe`), not the worktrees'. pymavlink 2.4.49 available. The plant is pure-numpy → all validation runs locally on the laptop in seconds; **no GPU needed for the plant work.**

**Adroit (SLURM):** repo at `/scratch/network/fl3689/peregrine_repo/`, DiffAero clone at `/scratch/network/fl3689/diffaero_repo`, outputs at `/scratch/network/fl3689/diffaero/outputs/train/<run>`. ffs0 warm dir: `.../ego_dual_gate_fullstack_floor_pef_seed0_vpeffs0/checkpoints`.

**Connector tricks (learned the hard way):** SFTP `upload` **HANGS on Duo**; the `x`-daemon does NOT → push files via `base64 -w0` locally then `x 'echo <b64> | base64 -d > /remote'`. `pull_file.py` needs `MSYS_NO_PATHCONV=1` + a LOCAL arg in `C:/` form. Run launchers via `x 'VAR=x bash /abs/path.sh'`. `.sh` must be LF before push (`tr -d '\r'`; `launch_v1.sh` already is).

---

## 6 · HARD RULES — DO NOT VIOLATE

- 🛑 **NO-SPIN is a HARD REQUIREMENT** — enforce by construction (fatal all-axes spin abort + realized-yaw clamp), never by incentive shaping. A high-DET spinner is not a champion. **No energy penalties.**
- 🛑 **NO GT IN ACTOR OBS** — the actor sees only estimator-faithful state (run the deploy estimators on simulated sensors in training). GT is legal ONLY in critic/reward/terminations. GT-obs runs = disposable diagnostics; their ckpts never fly.
- 🚩 **`algo=appo`, never `ppo`** — ppo leaves the privileged critic never connected = the inc8 seed-collapse root cause. Critic input_dim 37.
- 🚩 **`gamma=0.9975` — load-bearing, never change** (keeps the crash penalty dominant over banked progress at late gates).
- 🚩 **Select ckpts on `exit_frame` / HAZARD / `n_passed_gates` — NEVER on value.** Multi-seed before banking a recipe (single-seed variance ~2.8×).
- 🚩 **Fence ONLY the axis the course DOESN'T use** (pitch free; roll / speed / velocity-cap = NO). The velocity cap and speed governor were both REJECTED → `feedback-no-deploy-bandaids`.
- 🚩 **PRESERVE the yaw→gate-in-view→altitude coupling.** There's no abs-alt sensor; the gate holds altitude. Anti-dither must penalize yaw **JITTER ONLY**, never gate-tracking yaw. The lever for gate-in-view is `rw_perception`, NOT a cap.
- 🚩 **GATE CONTACT = INVALID RUN** (zero contact is THE validity rule).
- 🚩 **NO MAG, NO BARO, NO POSE on the VQ2 wire** (live-confirmed; `fields_updated=63` = accel+gyro only). Yaw + altitude come from VISION. **Only gyro + accel are real ground truth** — this is why §2's validation is IMU-only and why chasing "integrated position" is a dead end (Fengyou's explicit call).
- 🚩 **CTBR/VQ1 legacy sign config is a self-consistent alias — DO NOT "fix" it.** `rate_sign=[+1,+1,-1]` (yaw −1) stays. ODOMETRY quat is R_y(π)-conjugated.
- 🚩 **Commit / push / launch ONLY when Fengyou asks.** Never push `main` from a clone. Agents `git add` only their own paths (never `-A`).
- 🚩 **Adroit AUP = suspension risk:** SLURM only (never compute on the login node); compute nodes have NO internet (pre-stage deps); accurate `--mem`; output → `/scratch`; zero-GPU-util killed at 2 h; `checkquota`.
- 🚩 **NEVER use `/consolidate-memory`** (Fengyou, repeated). Compact BY HAND: over-limit ⇒ route detail DOWN a layer + leave a `[[pointer]]`. Mirror `memory/` ↔ `~/.claude`.
- 🚩 **Sim = NO physical risk** — non-POC sim flights you authorize directly, no ceremony. Relays through Fengyou = ONE code box.

---

## 7 · FOOTGUNS THAT COST US REAL TIME

- **absent-hook ≠ inert.** A missing print on a LIVE job doesn't mean the hook is dead (SLURM buffers stdout). **Adjudicate from a COMPLETED log / TB, never a live one.** For knob binding, grep the resolved `.hydra/config.yaml`.
- **The DR-off nominal is NOT what trains.** This is what burned me (§2). Before comparing any plant/model against reality, **confirm which config actually ran.** `dr=true` silently changes the plant (super-rate always on, `_dr_s`/`_dr_alpha`/`_dr_rate_tau` override params).
- **Closed-loop racing logs cannot measure open-loop plant gain.** Per-tick `|achieved|/|cmd|` on fast-flipping commands is dominated by lag/phase (we measured a VQ2 roll "gain" of 2.94→1.59 *decreasing* — pure artifact). Use the **controlled stepped sweeps**. `panel_val.py` exists but its numbers are NOT a plant verdict.
- **Binned amp-gain mixes transients.** It under-reads steady gain and it fabricated a "faithful is 5-25% too steep" scare that a **settled-sample mask** refuted (faithful is accurate to ±2%). Mask to settled samples before claiming a gain.
- **A 0-byte agent transcript ≠ dead** — check the TREE (file mtime + grep the mechanism). Don't run >2 heavy agents concurrently. Agents that stop "waiting on a background suite" → adjudicate from the tree and run the test yourself.
- **On a pose-blind wire, YOUR telemetry is a DEFENDANT; the human pilot's eyes are ground truth.** This paid off twice (gyro polarity A9; the pitch-sign / 17° tilt). When a sign or label is uncertain, **ask Fengyou for the human observation.** Pitch: `a_pitch=−0.10 → nose UP`.
- **Training spawn already models the 17° tilted pad** (`VQ1_SPAWN_PITCH_RAD=−0.31` at `peregrine_racing.py:876`) — not a gap.
- **Adversarial agents can be confidently wrong.** In one workflow: one lens mistook `accel_x=−3.0` for the accel *magnitude* and declared a unit anomaly (at-rest |accel| is 9.81, fine); another declared the transport delay unmodeled off a **stale code comment**. **Verify every load-bearing agent claim yourself** — but note the same workflow *did* catch my C3/C4 framing errors. Use them; don't outsource the verdict.
- **Worktree hygiene:** the `.claude/worktrees/*` pool RECYCLES — never manually prune (racy); only `git worktree prune`.

---

## 8 · NEXT ACTIONS (in order)

1. **Ask Fengyou to authorize committing + pushing the v1 code** (§4.1). This is the single highest-value action — the work is currently at risk in an uncommitted, recyclable worktree.
2. **When Adroit GPUs return:** pre-stage the 5 files + launcher (base64-through-`x`), then run the **two-phase** v1 launch (§3). Adjudicate the smoke from a COMPLETED log before releasing the arms.
3. **While the smoke runs:** pull ffs0's resolved `.hydra` and close §2's uncertainty (was the DR super-rate active in its actual run?).
4. **On v1 results:** apply the post-hoc **hard yaw-gate selection** (reject >~4 flips/s regardless of n_passed), pick the winner, publish a release for ShadowPC flight (`gh release`, clamp PER-RELEASE — mixing clamps is OOD).
5. **Then v1.5:** `ego_obs_v2` (+ its deploy twin) and/or the **clean quadratic-drag** aero term (before the thrust map).
6. **Standing relay to the vision commander (via Fengyou, ONE code box):** a **clean level-start thrust + inflow fly** (level off the pad first → vertical collective sweep so world-vz = axial; through-zero statics at 2.5/3 g) — that's the last plant data we want, and it unblocks a `coll_map` refit.
7. **Before you hand off:** edit `COMMANDER.md` (add the lesson, prune the stale, log your generation).

---

## 9 · MY HONEST ASSESSMENT FOR YOU

The biggest risk to this project right now is **not** the plant, and it's not v1's config. It's **over-confidence in a tidy root-cause story.** We had one — "flat plant → wall → faithful_rate" — that was clean, quantified, and wrong, and it survived a long time because every check I ran was against the DR-**off** nominal. It died in about twenty minutes once someone asked "does our setup actually make sense?" and I checked what config *really* ran.

So: **v1 is a good experiment and should fly.** But hold its rationale loosely. If v1 doesn't fix the wall, the next hypothesis should come from the deploy data (`28404fa` deep runs are the accumulation; `2aca009` are the oscillation), not from more plant tuning — the plant is already within ~8% on the safe side. And keep using the adversarial-verify pattern before any GPU-week: it cost ~840k tokens and saved us from banking a fix worth ~2%.

— outgoing RL commander, 2026-07-16
