# S14: static super-rate gain map integrated into twin / rl_plant / DiffAero adapter (parity-gated)

**Session LAPTOP-S14-STATICMAP-INTEGRATION (2026-06-10/11).** Implements
`handoff/shadowpc-characterize-sweep-2026-06-10/WRITEUP.md` Section 3 across all three plant
implementations, in the mandated order twin -> rl_plant -> adapter-numpy -> adapter-torch, with
machine-epsilon parity gates at every seam. **All gates PASS; local suite 409 green (376 before,
+33 new); defaults are OFF everywhere (exact legacy behavior preserved bit-for-bit).**

## What changed, per file

- **`src/racer/twin.py`** — `CtbrPlantConfig` gains `super_rate_s` (None | scalar | (3,)) and
  `alpha_max_rps2` (None | scalar | (3,)). Rate update becomes:
  `gain = rate_gain / (1 - s*min(|cmd|,pi)/pi)` (flat when s=None); increment slew-clamped to
  `+-alpha_max*dt` per axis (unlimited when None); then the existing |omega| norm clamp.
  Both None => bit-identical legacy floats (test-pinned). `max_omega_rps` stays 25.0 — the map's
  DC ceiling is g(pi)*pi ~= 11.2 rad/s/axis (~18.5 worst 3-axis norm), so the clamp never bites;
  a comment guards the ">= ~11.5" requirement.
- **`src/racer/twin_fit.py`** — `faithful_config(super_rate=True)` returns the measured map-ON
  config (s=0.30, alpha_max=[260,260,80]); default False = the exact legacy faithful twin.
- **`src/racer/rl_plant.py`** — same two params on `PlantParams` (normalized to float64 in
  `__post_init__`), step() mirrors twin operation-for-operation (omega stays bit-identical in the
  parity battery). Exports `SUPER_RATE_S_MEASURED = 0.30`, `ALPHA_MAX_RPS2_MEASURED = [260,260,80]`
  as the canonical nominals (DR centers).
- **`rl/diffaero_dynamics.py`** (torch backend = the training path):
  - `_step_torch` mirrors the new rate update; the DR-off scalar branch is bit-identical to
    rl_plant **including map-ON params**.
  - **params-level transport delay (S12 handoff item 4a)**: rl_plant's internal ring buffer is now
    mirrored in BOTH backends via the persistent `_plant_act_buf` (numpy threads rl_plant's own
    `PlantState.act_buf`; torch replicates push/pop per substep — delay counts SUBSTEPS, exactly
    like the per-substep `rl_step` calls). `reset_idx` re-seeds reset envs' queues with the hover
    command (the `PlantState.hover` analog); `detach()` covers the buffer. Default OFF.
  - **DR replaced**: the disproven asymmetric rate_gain band (-10%/+30%) is REMOVED (G0 now fixed
    at nominal under DR). New per-env bands, resampled at reset, torch backend only:
    `super_rate_s ~ U[0.25,0.35]` per-axis (3,), `rate_tau ~ U[0.015,0.030]` s (absolute,
    replacing the old +-30% fractional), `alpha_max` roll/pitch `~ U[200,320]` rad/s^2 with the
    yaw column scaled by its nominal ratio 80/260 (=> yaw ~ U[61.5,98.5], same relative width).
    hover (+-5%) and drag (+-30%) fractional bands unchanged. Map+slew are ALWAYS ON under DR.
    cfg knobs: `dr_s_lo/hi`, `dr_rate_tau_lo/hi`, `dr_alpha_max_lo/hi`.
  - Wrapper latency DR (per-episode {0..2}-step delay) untouched; comment added: measured input
    delay is 5-15 ms — one control step (33 ms @30 Hz) already covers it; do NOT double-count
    delay into rate_tau.
  - `check_against_rl_plant` now accepts `(T, ..., 4)` action stacks: both backends are stepped
    through the whole stack from an identical start (delay buffer snapshotted/restored) and every
    intermediate state is compared.
- **`rl/check_diffaero_gate.py`** — config-matrix gate: {legacy, super_rate, delay2, map_delay} x
  {float64 GATE, float32 advisory}, 6 seeds x 16 envs x 8-step trajectories, rate commands with
  tails beyond pi (exercises the min(|c|,pi) boundary + the slew clamp + buffer cycling).
- **`rl/local_gate_harness.py`** (new) — laptop CPU-torch dry run of the gate with a value-faithful
  stubbed `BaseDynamics` (grad_decay is value-preserving), so mirror bugs are caught before an
  Adroit roundtrip.
- **`rl/run_parity.sh`** — md5 tripwire updated (`a49919eb4b3eaffa5fc2cb2f04d31183`).
- **`rl/peregrine_racing_precheck.py`** — DR prints now show s / alpha_max / tau ranges (the old
  `_dr_rate_gain` attribute is gone). **`rl/peregrine_train_racing.py`** — stale band comment fixed.
- **Tests** — new `tests/test_super_rate.py` (13 tests): the MEASURED gain-table anchor, exact slew
  clamping, per-axis slew, legacy bit-equivalence (None==legacy, s=0==flat), batch semantics,
  map+delay ordering, max_omega guard. `tests/test_rl_plant_parity.py` battery extended with
  `super_rate` and `super_rate_delay` configs (40 -> 60 parametrized cases).

## Parity / objective numbers

- **Measured gain table (the objective anchor, twin + rl_plant, sustained per-axis cmds):**
  model vs measured — roll: 2.575/2.505 (+2.8%), 2.765/2.712 (+2.0%), 3.091/3.043 (+1.6%),
  3.572/3.50 (+2.1%); pitch: +2.9%, +2.1%, +1.7%, +2.2%. All within the 3% the one-parameter form
  is documented to fit ("slightly flat at the low end" — the +2.8-2.9% @0.3 is exactly that).
- **twin <-> rl_plant**: 60-case battery green; omega bit-identical (exactly 0.0) including map-ON.
- **Local CPU-torch gate** (stubbed base): DIV_FLOAT64 = 0.0 exactly, all 4 configs.
  Negative controls prove teeth: s biased by 1e-6 -> DIV 8.1e-05 (caught); torch silently dropping
  the delay -> DIV 13.7 (caught).
- **Adroit login-node gate (real DiffAero BaseDynamics, torch 2.5.1+cu121, CPU):**
  legacy 8.882e-16 / super_rate 8.882e-16 / delay2 8.882e-16 / map_delay 4.441e-16 — GATE_PASS,
  float32 advisory ~1-2e-06. md5 tripwire matched.
- **Adroit GPU gate (SLURM job 3265870, Tesla V100-PCIE-32GB, torch 2.5.1+cu121):** GATE_PASS —
  legacy 8.882e-16 / super_rate 1.776e-15 / delay2 8.882e-16 / map_delay 1.776e-15 float64
  (float32 advisory ~1-2e-06), i.e. worst 1.776e-15 vs the <= ~1e-6 acceptance and the historical
  2.2e-16..4.4e-16 family. `SBATCH_GATE_DONE rc=0`.
- **Test suite:** 376 passed before, **409 passed after** (same suite + 33 new), zero existing
  tests modified beyond the parity-battery extension.

## Decisions / ambiguities resolved (and the one deviation)

1. **Yaw**: same functional form, s_yaw = s_roll = 0.30 on yaw's own G0=2.231; the level-attitude
   ~7.4 rad/s yaw plateau is NOT modeled (comment-documented in twin + rl_plant + the test file,
   per the task). The yaw map is tested against the form AND the level measurements at |cmd| 1.0
   and 2.0 (where they agree within 3%); the 3.14-level point is exactly the unmodeled caveat.
2. **alpha_max DR band for yaw** (spec gave [200,320] = roll/pitch only): yaw samples the same
   band scaled by its nominal ratio 80/260 -> U[61.5, 98.5]. Rationale: every axis randomizes with
   the same relative width around its own measured nominal; sampling yaw in [200,320] would be
   physically wrong (measured ~78).
3. **s and alpha_max DR are per-axis (n,3)** — mirrors how the old rate_gain DR sampled (m,3).
4. **DR => map always ON** (s>0 sampled): the map IS the measured plant; a flat-gain DR run is
   recoverable via dr_s_lo=dr_s_hi=0 if ever wanted.
5. **transport_delay_steps semantics in the adapter**: the persistent buffer holds the
   POST-MAPPING NED CTBR action (rates FRD + collective), pushed per SUBSTEP — exactly what
   `_step_numpy`'s per-substep `rl_step` calls produce, so the two backends stay bit-identical.
   Previously the adapter passed `act_buf=None` every call, making params-delay a silent no-op;
   it is now real (and the gate's delay2/map_delay configs prove both backends agree).
6. **Deviation from WRITEUP Section 3: none functionally.** The optional 2nd-order refinement
   (wn/zeta, ~10% small-signal overshoot fidelity) was NOT taken — the writeup itself calls the
   static map "the load-bearing fix" and the first-order lag stays. tau default stays 0.019
   (writeup: "shipped 0.019 acceptable"); the DR band [0.015,0.03] covers the saturated
   tau_eq 25-33 ms.
7. **Staging note:** `/scratch/network/fl3689/peregrine_repo` is NOT a git clone (it's a minimal
   file copy: rl/* + src/racer/rl_plant.py) and the GitHub repo is private, so "pull there"
   was actually a base64-over-`x` file push (md5-verified per file). If Adroit sessions grow,
   consider staging a real clone with a deploy key.

## Follow-ups (not done here, deliberately)

- **Retrain (S1.4)** on the map-ON plant — per the already-banked gating: stage1_inc1/S1.3
  checkpoints under-predict full-stick authority by up to 42%; training configs should construct
  `PlantParams(super_rate_s=SUPER_RATE_S_MEASURED, alpha_max_rps2=ALPHA_MAX_RPS2_MEASURED)` (or
  just enable DR, which forces the map on).
- Eval/rollout scripts (`offline_rollout.py`, `peregrine_eval.py`) still default to the legacy
  flat plant — correct for evaluating the CURRENT flat-trained checkpoints; flip when S1.4 lands.
- The yaw level-attitude cap gets its own measurement pass only if racing yaw cmds ever grow.
