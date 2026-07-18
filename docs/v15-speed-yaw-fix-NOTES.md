# v1.5 — speed-discipline + yaw-quietness training fix (2026-07-18)

Successor to v1 (`launch_v1.sh`). v1 left two failure classes UNPRICED (see the REWARD-LEDGER forensic):

1. **Velocity runaway** — speed is unpriced and discount-favored (+12–27 reward/second-saved vs a finite,
   discounted crash terminal), so no policy ever brakes; it rides to the controllability edge.
2. **Yaw oscillation** — a ~6 Hz limit cycle from a 12–16× bearing-nulling gain (8-gate re-points) under a
   sign-flip-only penalty ~500× too small. The champion ffs0 flies quiet at cmd |yaw| absmean **0.232**,
   0% clamp saturation.

v1.5 makes both **structural**. Five mechanisms, ALL config-gated with defaults OFF/0 — a config without
the new keys trains **byte-identical** to v1 (pinned by `tests/test_v15_terms.py::test_v15_defaults_byte_identical_to_v1_reward`).

## Knobs (default → v1.5 value)

| # | knob (`++env.*`) | default (OFF) | v1.5 (Q / W) | mechanism |
|---|---|---|---|---|
| A | `overspeed_abort_mps` | `0.0` | `12.0` / `12.0` | fatal OOB-class abort when GT ‖v‖ > cap |
| B | `rw_progress_vcap_mps` | `0.0` | `7.5` / `7.5` | saturate positive progress credit above v* |
| C | `rw_yaw_duty` | `0.0` | `0.15` / `0.4` | price sustained post-clamp yaw amplitude beyond band |
| C | `yaw_duty_free_band` | `0.25` | `0.25` / `0.25` | free band (champion 0.232 is inside → pays 0) |
| D | `rw_yaw_jerk` | `0.0` | `0.05` / `0.05` | L1 price on \|Δyaw_cmd\| (post-clamp) — the 6 Hz bang-bang |
| — | `rw_yaw_dither` (v1 key, RAISED) | `0.0` | `0.4` / `0.6` | squared (Δyaw_cmd)² (v1 was 0.125) |
| — | `rw_roll_recover` (v1 key, now CORE) | `0.0` | `0.5` / `0.5` | bearing-weighted roll² recovery (v1: arm-R only) |

E (eval observability, append-only): `YAW_EVAL` gains `satur_duty=`; `DET_EVAL` gains `max_speed=`; new
reward terms log to TB (`env_loss/yaw_duty_pen`, `/yaw_jerk_pen`, `/prog_sat_forfeit`,
`/overspeed_abort_rate`).

## What / where (files:lines)

**`rl/ego_reward.py`** (pure reward; the OFF path of every term is byte-identical):
- Config fields: `progress_vcap_mps` L145; `yaw_duty` L414, `yaw_duty_free_band` L415; `yaw_jerk` L426.
- Term functions: `progress_credit_saturate` L635 (B); `yaw_duty_penalty` L999 (C); `yaw_jerk_penalty` L1019 (D).
- `compute_ego_reward` signature: `yaw_cmd` / `yaw_clamp` kwargs added (near `yaw_cmd_delta`).
- Assembly: saturation applied to `r_prog` **before** the area coupling L1284–1285 (so the cap on
  credited along-track speed holds regardless of approach angle); `r_yaw_duty` L1365, `r_yaw_jerk` L1370;
  both added to the reward sum; components `yaw_duty_pen` L1427, `yaw_jerk_pen` L1428, `prog_sat_forfeit` L1429.

**`rl/peregrine_racing_ego.py`** (env; GT legal in terminations, obs untouched):
- `overspeed_abort_mask` pure trigger L616 (A) — module-level, offline-testable (env construction is cluster-only).
- `__init__`: `self._peak_speed` buffer L870; `self._overspeed_abort_mps` L978.
- `step()`: overspeed folded into `oob_full` L1559–1561 (OOB-class → pays `terminal_oob`, KEEPS banked);
  `_peak_speed` updated L1604; `yaw_cmd` (post-clamp, when duty armed) L1689 + `yaw_cmd_delta` extended to
  fire when jerk OR dither armed; passed to reward L1753; `overspeed_abort_rate` loss-component L1823.
- `reset_idx()`: `_peak_speed` reset L1192. `stats_raw`: `peak_speed_mps` L1925.

**`rl/peregrine_train_ego.py`** (eval prints, append-only — old fields UNMOVED for downstream greps):
- `_run_det_eval`: `max_speed` tracked from `peak_speed_mps` L589/601–604; appended to `DET_EVAL` L626, L628.
- `_accum_yaw`: `satur_sum` (count \|cmd\| > 0.95·clamp) + `clamp` stored L666, L679.
- `_emit_yaw_eval`: `satur_duty` appended at END of `YAW_EVAL` L718, L721 (and `satur_duty=nan` L703).

**`rl/launch_v15.sh`** — the two-phase launcher (derived from `launch_v1.sh`).

**`tests/test_v15_terms.py`** — 18 tests (all green).

## Design decisions / deviations from a literal reading

- **A routed OOB-class, not lethal.** The spec says "same handling class as OOB … banked-gate handling
  UNCHANGED." The REWARD-LEDGER Q6 had *recommended* the lethal/forfeit fold; the SPEC overrides that, so
  overspeed pays `terminal_oob` (200) and KEEPS banked gate progress. Death pricing (the `terminal_penalty`
  function + bases) is untouched — overspeed only adds a new OOB trigger.
- **B saturation is applied before the area-distance coupling** (which only shrinks positive credit) so the
  cap on credited along-track speed holds regardless of approach angle (zero marginal credit above v*
  guaranteed).
- **No spawn grace for A.** The ego env spawns at rest (`spawn_vel` is not wired into the ego path) and the
  plant cannot reach 12 m/s in one 33 ms step, so overspeed cannot fire at spawn — mirroring OOB, which has
  no grace. ⚠️ Do NOT combine a `spawn_vel_max` above `overspeed_abort_mps` (would abort at spawn).
- **`rw_roll_recover=0.5` is on-from-birth (no `recovery_anneal`), default `theta0`=30°.** v1's arm R
  annealed it from 0; v1.5 follows the literal spec knob list + the dither-from-birth philosophy (warm from
  the already-smooth ffs0, hold discipline from step 0). If the commander intended the anneal, re-add
  `++env.recovery_anneal=true ++env.recovery_start=0.0 ++env.recovery_hold_frac=0.3`.
- **Cluster mechanics added per spec** (not in the exact `launch_v1.sh`, which hardcoded a100): generic
  `GRES` (default `gpu:nvidia_a100:1`, override for MIG/V100), `--exclude` (default `adroit-h11g3`, via
  `SBATCH_EXCLUDE`), optional `--time` injection via `SBATCH_TIMELIMIT`.

## Arm table

| arm | seeds | rw_yaw_duty | rw_yaw_dither | shared (A/B/D + roll_recover) | rundir tag |
|---|---|---|---|---|---|
| Q (primary) | 0, 1 | 0.15 | 0.4 | overspeed 12 · prog_vcap 7.5 · yaw_jerk 0.05 · free_band 0.25 · roll_recover 0.5 | `v15Q_s0`, `v15Q_s1` |
| W (strong-quiet) | 0 | 0.4 | 0.6 | (same shared block) | `v15W_s0` |
| smoke | 0 | 0.15 | 0.4 | (CORE = Q wiring), UPD=100 | `v15smoke_s0` |

Everything else VERBATIM from v1: warm from vpeffs0 (ffs0), stage `dual_gate_fullstack_floor_pef`,
course_n_gates 8, clamp 0.7, ticks_hi 1, blur off, spin-abort anneal, logstd reset, `faithful_rate`,
`ego_vision_cadence` (30 Hz × p=0.35 seed 20260715+SEED), noise anneal, `critic_grad_clip=10`,
`ckpt_select_metric=n_passed_gates`, `eval_det_steps=600`, `eval_yaw_log=true`, UPD 18000, save_freq 500.
`algo=appo` and `gamma=0.9975` are set by the stage/sbatch and are UNTOUCHED.

## Smoke pass-criteria (the free-band calibration proof)

Adjudicate the **completed** smoke log (never a live one — the absent-hook ≠ inert footgun, L16):

1. `.hydra/config.yaml` binds `overspeed_abort_mps: 12.0`, `rw_progress_vcap_mps: 7.5`, `rw_yaw_duty: 0.15`,
   `yaw_duty_free_band: 0.25`, `rw_yaw_jerk: 0.05`, `rw_roll_recover: 0.5` (all under `env:`), and
   `faithful_rate: true` + `ego_vision_cadence: true` (the v1 fixes still hold).
2. TB (`scratchpad/tb_parse.py`): `env_loss/yaw_duty_pen`, `/yaw_jerk_pen`, `/prog_sat_forfeit`,
   `/overspeed_abort_rate` ALL present (the wiring watchdog).
3. `EGO_PRECHECK_RC=0`, no NaN.
4. **FREE-BAND CALIBRATION PROOF** — frozen-ffs0 `YAW_EVAL` (run `++rollout_only` on the WARM ffs0 ckpt, or
   read the smoke's first eval): **signflips ~2.45 AND satur_duty ~0 AND `env_loss/yaw_duty_pen` ≈ 0.**
   This proves the champion's quiet 0.232-absmean yaw sits INSIDE the 0.25 free band and pays ZERO duty
   (the coupling is preserved). If `yaw_duty_pen` is materially > 0 on frozen ffs0, the band is mis-set —
   STOP before releasing the arms.

## Launch commands

```bash
# PHASE 1 — smoke (UPD=100, CORE=Q wiring):
bash rl/launch_v15.sh
#   -> prints the smoke jobid + adjudication checklist + the exact phase-2 command.

# PHASE 2 — arms Q(0,1) + W(0), afterok the smoke:
MODE=arms SMOKE_JID=<smoke_jobid> bash rl/launch_v15.sh
#   escape hatch (no dependency):  MODE=arms SMOKE_JID=none bash rl/launch_v15.sh

# optional cluster overrides:
GRES=gpu:1 SBATCH_TIMELIMIT=20:00:00 SBATCH_EXCLUDE=adroit-h11g3 MODE=arms SMOKE_JID=<jid> bash rl/launch_v15.sh
```

## Deploy-pick (post-hoc, HARD joint gate — now speed-aware)

For each seed, `++rollout_only` over the periodic + `best_npg` snapshots → per-ckpt `YAW_EVAL`
(signflips + satur_duty) + `DET_EVAL` (n_passed_gates + max_speed). **Discard** any ckpt with
signflips > ~4.0 OR satur_duty > ~0.1 (a yaw-hunter/rail-rider); confirm `max_speed` sits below the 12 m/s
abort with margin (a run pinned at the abort is riding the edge). Among survivors, take highest DET
`n_passed_gates`; tiebreak lowest signflips + satur_duty + roll_swing. NEVER select on value.
