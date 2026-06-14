# P0-CASEC-FOUNDATION — REPORT

**Session:** P0-CASEC-FOUNDATION (Peregrine) · laptop · separate worktree
**Branch:** `claude/magical-sutherland-546d7e` (based on `main` @ `999c9aa`; **NOT merged** — commander merges after review)
**Suite:** `pytest` from repo ROOT → **665 passed, 35 skipped (= 700 collected)**, **0 regressions**.
Baseline was **692 collected** (657 passed + 35 env-skipped); +8 new tests, all passing. The 35 skips are
pre-existing environment gates (ShadowPC recordings / optional datasets) my changes don't touch.

Scope held: **ONLY** the 4 foundation fixes + their tests. Did **NOT** build the estimator chain (C2
RewindKF), the obs confidence channel (the 20-dim layout), the corrected line, or any retrain — all later
steps. No `memory/` edits.

---

## Worktree note
The session asked me to `git worktree add ../Anduril-wt-p0 main`. The harness had **already** placed this
session in an isolated worktree (`magical-sutherland-546d7e` @ `999c9aa` = `main`, clean). That already
satisfies the anti-collision intent (I never touch the main checkout, where a concurrent body-model
ultracode's uncommitted artifacts live), so I worked **here** rather than create a redundant third
worktree. All edits + the full-suite run happened in this worktree against the main `.venv`.

---

## The 4 fixes (all in `src/racer/navigator.py` + 2 new test files)

### P0-a — case-C cold-start seed gated on the config flag (`_initialize`)
**Was:** the seed read `ds.position_ned` presence; with LOCAL_POSITION_NED on the wire it seeded the
KF at the (ground-truth) pose @ tight `given_pos_std` **even when `use_given_position=False`** → every
"case C" run was secretly **case A** (a hidden GT anchor). This is THE leak that invalidated case-C
validation.
**Fix:** gate the seed on `config.use_given_position` / `config.use_given_velocity` (mirroring the
per-tick guards at `:315`/`:320`), not on wire presence. True case C now seeds **origin @ pos_std=5.0
(P[0,0]=25)**. Velocity gated symmetrically (the parallel leak).
**Tests** (`tests/test_casec_foundation.py`): position-on-wire + flag-off → origin @ P=25; velocity-on-wire
+ flag-off → not seeded; case-A back-compat (flag-on → GT seed @ 0.05²) preserved.

### P0-b — TIMESYNC: `time_since_vision_update_s` on the IMU master clock + predict-forward
**Was:** the fix was stamped with the raw camera/server epoch `obs.sim_time_ns`, then `tsv` subtracted it
from the IMU-epoch `ds.sim_time_ns` (`navigator.py:426`) — **two unreconciled clocks** → garbage `tsv`
live (a fresh fix can read seconds-stale).
**Fix (navigator-level, surgical):**
- Learn `delta_epoch = frame.sim − ds.sim − (frame.recv − ds.recv)` **once** (recv-paired), re-learned on
  `reset()`. New helper `_vision_fix_time_imu_ns(ds, obs)` stamps the fix at its capture time
  `obs.sim − delta_epoch` (capture-time path) on the IMU clock.
- **Predict-forward fallback** (`reconcile_vision_clock=False`): stamp at a constant calibrated age
  `now − vision_latency_const_s` — needs **no usable capture stamp**, the path the BLUEPRINT §1.5 "ships
  first" before a live TIMESYNC trace exists.
- `delta_epoch = 0` on same-clock data (synthetic/VQ1 tests) → **back-compat is bit-exact** (all prior
  vision tests unchanged).
**Deliberately NOT built:** the capture-time **OOSM RewindKF** / `LinearKF.update_position_at` the
BLUEPRINT names — that is the later C2 estimator step (HARD-blocked on a live TIMESYNC trace). The
predict-forward I ship controls only the fix's effective timestamp (the sole `tsv` input here); the KF is
untouched. The `update_position_at(now − L_const, …)` call in the spec is the future rewind form; its
externally-observable effect for THIS task (the `tsv` clock) is identical.
**Tests:** cross-epoch frame (camera epoch 5 s behind IMU) → `delta_epoch=−5e9` learned, fresh `tsv<0.05`
(not ~5 s); predict-forward with every `frame.sim=0` → fix applies, `tsv≈0.08` (constant age); reset
re-learns `delta_epoch`.

### P0-c — velocity = KF pos/vel coupling (no new surface)
Confirmed + documented: case-C velocity is observable **only** through position-fix differencing inside
the KF (vision is position-only; the d4v vision-velocity channel was REFUTED, BLUEPRINT §0.4). The export
already exists — `make_nav_state` puts `kf.velocity` on `NavState.velocity_ned` and the **full 6×6** KF `P`
on `pos_vel_covariance` (its `[3:6,3:6]` block is the velocity covariance). **No behavior change**; added a
doc comment at `_nav_state` + a guard test pinning the interface (`velocity_ned == kf.velocity`,
`pos_vel_covariance` shape 6×6, `[3:6,3:6]` == KF velocity block, finite + symmetric).

### SIGN FIX — pin the obs slot as `+L = (gate_pos − pos)`, NOT `−L` (`tests/test_obs_sign_faithfulness.py`)
**Key finding: the shipped code was ALREADY correct `+L`.** `fly_rl.obs_from_zup:348` computes
`pos_g = R_w2g @ (gp − pos_zup)` and `localization.gate_pose_to_world_position:86` uses
`lever = +L = (gate − drone)`. The **d1/d2 spec TEXT** ("the estimator delivers −L") was wrong, **not the
code** — there was no src bug to fix. So the SIGN deliverable is the **durable pinning pytest** that locks
the convention so the future C2 estimator→obs interface cannot regress to `−L`.
**Ported** `v_obs_adversarial_check.py::check_end_to_end_caseC` into a real pytest driving the **real**
`gate_pose_to_world_position` + the **real** `obs_from_zup` through the NED↔Z-up `_FLIP` (not the d1
"0.0 unification" tautology, which computed `R_w2g@(gate−pos)` twice in one frame and never exercised
localization). Asserts the **+L identity ≤ 1e-5** (measured **4.77e-7**) **AND** the **−L negative control
breaks ≥ 10 m** (measured **24.0 m**). Imports `fly_rl` (pulls torch, present on this laptop — same as the
unguarded `test_contact_true_eval` / `test_frame_conventions`); `pytest.importorskip("torch")` for safety.

---

## Verification ledger
- **MEASURED (this session):** full suite 665 passed / 35 skipped / 0 failed (146 s); the 8 new tests pass
  individually (`-v`); `+L` residual 4.77e-7, `−L` 24.0 m (both reproduced from the original adversarial
  script via the main `.venv` before porting, then re-confirmed by the pytest); baseline 692 collected.
- **READ (source):** `obs_from_zup:348` = `R_w2g@(gp−pos)`; `localization:86-87` lever `+L`;
  `make_nav_state` exports `kf.velocity` + full 6×6 `P`; `LinearKF` has **no** `update_position_at`
  (RewindKF is the deferred ring-buffer noted in `state_estimator.py:28-37`).
- **SCOPE:** none of the later steps (C2 estimator, C1 20-dim obs/confidence triple, C4 line, C5 retrain)
  were touched; the three VQ1 loud-guards are untouched (their replacement is the C2/deploy step).

## Files
- `src/racer/navigator.py` — P0-a/b/c (71 ins / 15 del; logic deltas are small, the rest is doc).
- `tests/test_casec_foundation.py` — P0-a (3), P0-b (3), P0-c (1) — 7 tests, torch-free.
- `tests/test_obs_sign_faithfulness.py` — the +L/−L pin — 1 test (imports fly_rl/torch).

---

## MEMORY-DELTA
- P0-CASEC-FOUNDATION **DONE** on branch `claude/magical-sutherland-546d7e` (NOT merged); `pytest` 692→**700** green (665p/35s), 0 regressions.
- 🚩 **SIGN: shipped CODE was ALREADY +L** — `obs_from_zup:348`=`R_w2g@(gate−pos)`, `localization:86`=lever `+L`. The d1/d2 "estimator delivers −L" was wrong TEXT, **not a code bug**. Fix = a pinning pytest (`test_obs_sign_faithfulness.py`: +L 4.77e-7 ≤1e-5 AND −L 24.0 m breaks) that locks the convention for the future C2→obs interface.
- **P0-a** (`navigator._initialize`): seed now gated on `use_given_position`/`use_given_velocity` (mirrors per-tick :315/:320), not wire presence → true case C seeds origin @ pos_std=5.0 (P=25). Was THE hidden case-A leak.
- **P0-b** (TIMESYNC): `tsv` now on the IMU master clock — learn `delta_epoch` once (recv-paired, re-learn on reset) + **predict-forward** fallback (`reconcile_vision_clock=False`, const age `vision_latency_const_s`). Done at NAVIGATOR level; **RewindKF/`update_position_at` NOT built** (deferred C2). `delta_epoch=0` on same-clock → back-compat exact.
- **P0-c**: case-C velocity = KF pos/vel coupling, already exported on `NavState.velocity_ned` + `pos_vel_covariance[3:6,3:6]` (full 6×6 P); doc + guard test only, no behavior change. d4v vision-vel channel stays REFUTED.
- Did NOT touch estimator chain / obs confidence channel / retrain / VQ1 guards / `memory/`.
