# S16: measured aero (quad body drag + convex collective) integrated into twin / rl_plant / DiffAero adapter (parity-gated)

**Session LAPTOP-S16-AERO-INTEGRATION (2026-06-11).** Implements
`handoff/shadowpc-twin-falsify-2026-06-10/WRITEUP.md` Sections 2-3 per the Section 7 proposal,
across all three plant implementations, in the S14-mandated order twin -> rl_plant -> adapter,
with parity gates at every seam. **All gates PASS; local suite 457 -> 497 green (+20 anchor
tests, +20 parity cases; zero existing tests modified beyond the parity-battery extension);
defaults are OFF everywhere (exact legacy behavior preserved bit-for-bit, pinned by test).** The campaign's survivals (super-rate
map, slew, tau, transport delay, rate_sign) are untouched.

## What changed, per file

- **`src/racer/twin.py`** — `CtbrPlantConfig` gains `quad_drag_c2` (None | scalar | (3,) | (3,2))
  and `coll_map_thr`/`coll_map_accel` (None | (K,) knot arrays, set together or raise). Step:
  - drag: when `quad_drag_c2` is set, `f_b = -c2 (.) |v_b| (.) v_b` per body axis with the
    coefficient picked by the SIGN of `v_b[axis]` ((3,2) table: col 0 where >= 0, col 1 where < 0;
    OLD velocity, like the linear term), rotated back to world and ADDED after the legacy
    `linear_drag` term (the measured config zeroes linear_drag; d1 + quad = the WRITEUP "mixed"
    form). New module helper `_quad_c2_table` normalises the spec.
  - collective: when the knot arrays are set, `a_up = np.interp(realised_collective, thr, accel)`
    REPLACES `g*thr/hover` (np.interp end-clamps: the [0,1] stick saturates at the last knot).
  - All None => bit-identical legacy floats (verified vs the git-HEAD implementation over 500
    random steps, then pinned by `tests/test_measured_aero.py::test_aero_off_step_is_bit_identical_to_legacy`,
    a full inline reimplementation of the legacy step).
- **`src/racer/twin_fit.py`** — `faithful_config(super_rate=False, measured_aero=False)`.
  `measured_aero=True` sets `linear_drag=0.0` + the three aero params from the canonical
  constants in `racer.rl_plant` (single source of truth; twin_fit imports them inside the
  function). The fully sim-faithful twin as of 2026-06-11 = `faithful_config(super_rate=True,
  measured_aero=True)`. Both flags default OFF.
- **`src/racer/rl_plant.py`** — same three params on `PlantParams` (normalised/validated in
  `__post_init__`: scalar/(3,) -> (3,2); knot arrays both-or-neither, strictly increasing,
  K >= 2). `step()` mirrors twin operation-for-operation via the new `_interp1d` (explicit
  np.interp-equivalent arithmetic, bit-identical — pinned by test — and written out so torch can
  mirror it op-for-op). Exports the canonical measured nominals (DR centers):
  - `QUAD_DRAG_C2_MEASURED = [[0.042, 0.058], [0.055, 0.055], [0.0539309, 0.0756169]]`
    (rows = body FRD axis; col 0 where v_b >= 0, col 1 where < 0; x = nose/tail, z =
    descend/climb; verticals at the vert_fit_coef.npy full precision, horizontals the WRITEUP
    per-family values),
  - `QUAD_DRAG_C2_POOLED = 0.052` (isotropic pooled fit; the Section 7 DR-band center),
  - `COLL_MAP_THR_MEASURED` / `COLL_MAP_ACCEL_MEASURED` — the full 12-knot Section 3 table at
    npy precision, bottom two knots replaced per Section 7 and FLOORED at 0 (deviation 1 below).
- **`rl/diffaero_dynamics.py`** (torch backend = the training path):
  - `_step_torch` mirrors the new translation block (`_t_interp1d`, `_t_quat_conjugate`; same
    arithmetic order as `_interp1d`/`quat_rotate_inverse`, so the float64 gate stays at machine
    epsilon). The DR-off scalar branch is bit-identical to rl_plant INCLUDING aero-ON params.
  - **New DR: `+dynamics.dr_aero=true`** (opt-in ON TOP of `dr`; default OFF — deviation 5).
    Forces the measured aero ON with per-env bands, resampled at reset, torch backend:
    - quad c2: per-slot `nominal x U[dr_c2_lo, dr_c2_hi]/0.052` (defaults [0.040, 0.065] =
      the Section 7 band at the pooled nominal; relative per slot — deviation 2), shape (n,3,2);
    - collective table: `K_dr = h*K_hov + k*(K - K_hov)` with `k ~ U[dr_coll_k_lo, dr_coll_k_hi]`
      (default [0.90, 1.10]) and `h ~ U[dr_coll_hover_lo, dr_coll_hover_hi]` (default
      [0.98, 1.02]) — the Section 7 "scale the table, pin hover +-2%" (deviation 3), shape (n,K);
    - residual linear d1: `_dr_drag ~ U[dr_d1_lo, dr_d1_hi]` (default [0, 0.08], ABSOLUTE),
      replacing the legacy +-30% fractional drag jitter under dr_aero;
    - the stick->collective hover conversion is PINNED at the nominal under dr_aero
      (deviation 6); plain-`dr` (legacy) sampling is byte-for-byte unchanged.
    Nominals follow the params if measured-aero params were passed, else the canonical constants
    (the `_alpha_nom` pattern).
- **`rl/check_diffaero_gate.py`** — config matrix extended to 6: {legacy, super_rate, delay2,
  map_delay, **aero**, **aero_full** (aero + map + slew + delay2)} x {float64 GATE, float32
  advisory}, 6 seeds x 16 envs x 8-step trajectories. `rebuild_params` now rebuilds the three
  cached aero tensors at the gate dtype. `random_traj` pins steps T-2/T-1 to the knot-table
  edges (collective ~[1.01, 1.33] upper clamp / ~[0, 0.106] floored bottom) — this changes the
  gate trajectories for ALL configs (the gate is self-comparative; all 6 re-verified).
- **`rl/run_parity.sh`** — md5 tripwire updated: `a49919eb...` -> `cdc3f57eb888096c840f146717741e3e`
  (hash of the LF git-blob form, which is what Adroit's `md5sum` sees).
- **`rl/peregrine_racing_precheck.py`** — DR print block now also reports the aero-DR state
  (bands + hover pin when ON; an explicit "+dynamics.dr_aero=true" pointer when OFF). Prints only.
- **Tests** — new `tests/test_measured_aero.py` (20 tests; see the anchor table below) +
  `tests/test_rl_plant_parity.py` battery extended with `measured_aero` and `aero_full`
  (aero + map + collective lag + 3-step transport delay) configs: 60 -> 80 parametrized cases.

## Parity / objective numbers

- **twin <-> rl_plant (80-case battery, all green):** global max divergence pos 5.116e-13 m,
  vel 1.990e-13 m/s, att 5.313e-15 rad; **omega and thrust exactly 0.0** (bit-identical),
  same class as the S14 numbers (tolerances 1e-9/1e-9/1e-11/1e-12/1e-12).
- **Local CPU-torch gate** (`rl/local_gate_harness.py`, stubbed value-faithful BaseDynamics,
  torch 2.12.0+cpu, float64): legacy 4.441e-16 / super_rate 4.441e-16 / delay2 0.0 /
  map_delay 1.110e-16 / **aero 8.882e-16 / aero_full 1.110e-16** — **GATE_PASS** (bound 1e-9;
  float32 advisory ~1.1-1.2e-06).
- **Negative controls (gate has teeth on the new paths):** uncorrupted run 0.0 exactly;
  c2[fwd] biased 1e-6 -> DIV 7.7e-07 (caught); torch silently dropping quad drag -> 1.47e-01;
  full-stick knot biased 1e-5 -> 2.0e-07 (caught); dropping the knot map -> 8.9e-01.
- **Legacy bit-identity:** twin AND rl_plant, aero-OFF, vs their git-HEAD implementations:
  exactly equal (every float) over 500-step random batteries; pinned permanently by the inline
  reference test.
- **Test suite: 457 passed before (S15 has grown it past the banked 412), 497 after** (+20
  test_measured_aero, +20 parity-battery cases, on the same machine/.venv).
- **Adroit gate: NOT run this session** (per the task: connector contention; the local CPU gate
  + the S14-proven gate script is the accepted evidence). The next Adroit session runs
  `rl/run_parity.sh` as-is — md5 tripwire already updated; expect the historical ~1e-15/1e-16
  family on all 6 configs.

## Anchor-test table (WRITEUP measured -> integrated model)

| anchor (Section) | measured | model | test |
|---|---|---|---|
| nose-first decel @9 m/s (2) | c2 0.042 -> 3.402 m/s^2 | 3.402 exact (<1e-9) | drag_decel_per_direction |
| tail-first decel @9 m/s (2) | c2 0.058 -> 4.698 | 4.698 exact | drag_decel_per_direction |
| lateral decel @9 m/s (2) | c2 0.055 -> 4.455 | 4.455 exact | drag_decel_per_direction |
| pooled braking @9 m/s (1) | ~4.2 vs legacy 1.9 (2.2x) | 4.2114 vs 1.8999 (2.22x) | headline_braking_at_9ms |
| quadratic scaling (2) | a/v^2 flat | decel(6)/decel(3) == 4 exact | drag_decel_per_direction |
| climb drag @5 m/s (3) | c_up 0.0756 -> 1.890 | 1.890 exact | vertical_drag_split |
| descend drag @5 m/s (3) | c_dn 0.0539 -> 1.348 | 1.348 exact | vertical_drag_split |
| body-frame split | (3,2) sign table | yawed/inverted attitude picks the airflow coefficient | drag_is_body_frame, vertical_..._inversion |
| collective knots (3) | 12-knot table | all knots exact through the plant (<1e-9) | collective_knot_table |
| K/linear ratio column (3) | 0.38/0.64/0.98/1.15/1.47/1.91/1.98/2.12 | all within +-0.005 | collective_ratio_column |
| full stick (1,3) | 78.3 m/s^2 = 2.12x linear 36.9 | 78.2828 / 36.923 | collective_ratio_column |
| sub-linear below hover (3) | K < linear @ <=0.20 | holds | collective_convex_above_sublinear_below |
| convex above hover (3) | K > linear @ >=0.32 | holds | collective_convex_above_sublinear_below |
| stick saturation | [0,1] stick | thr 1.2 -> 78.2828 (end clamp) | collective_end_clamps |
| coast decay shape (2,5) | v(t) = v0/(1+c2 v0 t) | RMS < 0.02 m/s over 6 s @100 Hz, per family (obs ~5e-3) | coast_replay_surrogate |
| legacy shape falsifier (2) | over-brakes < ~4 m/s, under above | crossover at d1/c2 = 4.06 m/s within 2% | legacy_overbrakes_low... |

**Recordings validation (task item 5):** the campaign's recordings are ShadowPC-local
(`data/runs`, gitignored — confirmed absent on this laptop: only `smoke_video` present), so the
WRITEUP's replay numbers (coast speed RMS legacy **0.810 -> candidate 0.240-0.289 m/s**, 9 runs)
could NOT be re-run here. The coast-replay CHECK was replicated as the closed-form surrogate
above (the exact quad-coast solution the Section 2 fits imply), as the task anticipated. A
ShadowPC session can additionally point `handoff/shadowpc-twin-falsify-2026-06-10/replay_coast.py`
at the integrated plant (`faithful_config(super_rate=True, measured_aero=True)`, no CandidatePlant
subclass needed) for the recording-level numbers; expected ~0.24-0.29 with the (3,2) table
slightly better on fwd/back than the isotropic candidate.

## The aero-ON switches (for S15's successor — DOCUMENTED, deliberately NOT applied)

Per the S15 concurrency constraint, NOTHING in `rl/peregrine_racing.py`, `peregrine_course.py`,
`peregrine_eval.py`, `fly_rl.py` was touched, and no default changed. To train aero-ON:

1. **Training with DR (the recommended path):** add `+dynamics.dr_aero=true` next to the existing
   `+dynamics.dr=true`. That alone forces quad drag + the knot collective ON with the Section 7
   bands (c2 per-slot, K-table scale with the +-2% hover pin, d1 in [0, 0.08] replacing the
   legacy drag jitter). Band knobs if needed: `dr_c2_lo/hi`, `dr_d1_lo/hi`, `dr_coll_k_lo/hi`,
   `dr_coll_hover_lo/hi`.
2. **Fixed aero (no DR), e.g. eval/rollout** — construct the params:
   ```python
   from racer.rl_plant import (PlantParams, SUPER_RATE_S_MEASURED, ALPHA_MAX_RPS2_MEASURED,
                               QUAD_DRAG_C2_MEASURED, COLL_MAP_THR_MEASURED, COLL_MAP_ACCEL_MEASURED)
   params = PlantParams(super_rate_s=SUPER_RATE_S_MEASURED,
                        alpha_max_rps2=ALPHA_MAX_RPS2_MEASURED.copy(),
                        linear_drag=0.0,
                        quad_drag_c2=QUAD_DRAG_C2_MEASURED.copy(),
                        coll_map_thr=COLL_MAP_THR_MEASURED.copy(),
                        coll_map_accel=COLL_MAP_ACCEL_MEASURED.copy())
   ```
   (`peregrine_eval.py` line ~69 and `offline_rollout.py` line ~222 are the two sites; they
   correctly stay legacy while the CURRENT flat-aero checkpoints are being evaluated.)
3. **Twin-side (scripts, replay, controller tuning):**
   `faithful_config(super_rate=True, measured_aero=True)`.

**Checkpoint gating (WRITEUP Section 7 item 5):** anything trained pre-aero (stage1_inc1, S1.3,
S1.4/S15 line) learned ~2x-wrong braking distances at speed and ~2x-low full-stick climb
authority — same retrain-after-integration gating as the S14 map. The TOGT time-optimal bound
(commit 35f451d) used T/W ~= 3.77 from the falsified linear map; the real full-stick is ~8 g
=> the bound is CONSERVATIVE and worth re-running with the knot table.

**Training notes for the successor:**
- With the knot map, `normed_thrust = 1.0` is no longer exactly hover: the measured table reads
  9.580 m/s^2 at the 0.2656 knot (the fit recovers g to 2.3%; kept as measured, not snapped).
  True trim ~= collective 0.2687 ~= normed 1.0117. Policies find this in seconds; obs/action
  conventions are unchanged.
- Gradient plateau: the interp end-clamp zeroes the thrust gradient above collective 1.0
  (commanded normed > 3.765). PPO is unaffected; a BPTT/SHAC run would see zero pathwise
  thrust-gradient in the saturated band — consider tightening the action ceiling there (S15
  already pins max normed thrust to 3.765 per its precheck assert, which exactly removes this).

## Deviations from WRITEUP Section 7 (and why)

1. **Bottom knots floored at 0** (Section 7 says "hold 0.0/0.10 at the linear extrapolation of
   0.15-0.20"): the literal extrapolation gives K(0) = **-5.60** and K(0.10) = **-0.45** m/s^2 —
   "worse than free fall", contradicting Section 8's own observation that the c000 step showed
   *near-free-fall* (K ~= 0). Shipped: extrapolate, then floor at 0 => K(0)=K(0.10)=0, a free-fall
   bottom with an idle-deadband shape (motor idle floor 0.05). Affects commanded collective
   < 0.109 only; the fit-grade region (0.15-1.0) is verbatim at npy precision.
2. **c2 DR band is per-slot RELATIVE** (`nominal x [0.040, 0.065]/0.052`), not the literal
   absolute "[0.040, 0.065] per axis": the absolute band cannot even contain the climb slot's
   0.076 nominal (it would systematically under-drag climbs). Same resolution as S14's
   yaw-alpha_max decision — every slot randomizes with the same relative width around its own
   measured nominal; at the pooled nominal the band is exactly the Section 7 one.
3. **K-table DR implemented as the affine** `K_dr = h*K_hov + k*(K - K_hov)`: a literal
   whole-table scale of U[0.9, 1.1] and a "+-2% hover pin" are mutually inconsistent AT the hover
   point. The affine form satisfies both readings: authority (deltas from hover) scales by
   k ~ U[0.9, 1.1], the trim point moves only by h ~ U[0.98, 1.02].
4. **d1 is world-isotropic** (it samples the existing `linear_drag` channel, which also damps
   vertical): the fitted mixed form was horizontal-only, but Section 7 gives no frame
   qualification and the nominal is d1 = 0 (pure quad) anyway — the band is robustness slop.
5. **`dr_aero` is a separate opt-in, NOT folded into `dr`** (S14 folded the super-rate map into
   plain `dr` because "it IS the measured plant" — by that logic the aero would too): S15 trains
   CONCURRENTLY on this repo with `dr=true`; folding aero in would silently change their plant
   mid-campaign, violating this session's no-default-changes constraint. Folding it into the
   default `dr` is the successor's one-line decision once the aero-ON retrain is gated in.
6. **Hover conversion pinned under dr_aero:** the legacy +-5% hover band is dynamically INERT in
   the linear plant (it cancels: `a_up = g*(U*hover_dr)/hover_dr = g*U`) but under the knot table
   it would scale the whole thrust curve +-5% (~+-0.5 m/s^2 at trim), violating the Section 7
   +-2% pin. Under dr_aero the K-table's h-jitter IS the hover randomization; plain-`dr`
   behavior is untouched.
7. **Vertical sign split is BODY-frame** (per Section 7's `f_b` equation), where the validated
   `CandidatePlant` replay used world-frame vz: equivalent for the near-level probes that
   measured it (tilt <= 32 deg); body-frame is the spec and the physically-correct
   generalization (airflow direction relative to the airframe).
8. **Horizontal drag form is the Section 7 Hadamard** (per-axis `|v_bi|*v_bi`), where
   `CandidatePlant` used speed-weighted isotropic (`c2*|v_h|*v_xy`): identical on the
   single-axis coasts the fits used; they differ only for diagonal motion (untested regime —
   flagged for any future high-speed pass).

## Caveats carried forward (from WRITEUP Section 8, unchanged by this integration)

- Drag is measured to ~7.6 m/s horizontal; the quad extrapolation beyond ~8 m/s is principled
  but unverified (VQ2 speeds may exceed it — the planned higher-speed corridor pass).
- The fwd/back c2 split rests on 2 fwd runs; K(thr) below 0.15 is not fit-grade (hence the
  floored bottom).
- The DR bands deliberately bracket these uncertainties (c2 slot-relative ~+-25%, K +-10%).
