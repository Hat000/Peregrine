"""REGRESSION for CONFIRMED bug class P3-C06 -- recording quantization in rl/fly_rl.py.

WHAT BUG CLASS THIS CATCHES
---------------------------
fly_rl.py's ``--debug-obs`` dump (fly_once, ~line 778) stores every per-tick field at a
DECLARED decimal grid before json.dump:
    obs           -> np.asarray(obs).round(5)        (round-5 grid, half-step 5e-6)
    rate_frd      -> rate_frd.round(4)               (round-4 grid, half-step 5e-5)
    collective    -> round(collective, 5)
    normed_thrust -> round(last_normed, 5)
    pos_ned       -> round(4)   vel_ned -> round(4)
    q_raw_wxyz    -> round(6)   w_raw   -> round(4)
Those recordings are the ANCHOR for every downstream external-invariant audit (the
force-vs-FD frame test, the train<->deploy action round-trip, gate-margin metrics).
P3-C06 is the failure mode where a future edit to fly_rl.py silently changes the stored
precision: rounding is REMOVED (raw float leaks in), COARSENED (round(2) instead of
round(4)), or the wrong column/field is quantized. Any of these corrupts every consumer
of debug_obs.jsonl with no error at write time.

WHY INTERNAL-CONSISTENCY CHECKS MISS IT (and the project's canonical frame test does too)
----------------------------------------------------------------------------------------
This is NOT an attitude-convention / proper-rotation-conjugation bug, so the canonical
force-vs-FD-of-pristine-vel_ned external invariant (the discriminator for the R_y(pi)
quat mirror, the body/world velocity-frame mix, the corner_to_center 180deg flip, the
ATTITUDE.pitch sign flip) is NOT the right instrument here -- a quantization change leaves
every rotational relationship intact, so the East-axis force correlation stays ~+0.99.
And a value rounded to ``d`` digits is ALWAYS within half a grid step of *some* round-d
value, so naive "is it on a grid" reasoning is degenerate -- a raw float looks on-grid to
within the half-step. The discriminating EXTERNAL invariant has to read precision off the
recordings two independent ways:

  (I)  GRID SELF-CONSISTENCY -- max|x - round(x, d)| over EVERY recorded value must be
       *exactly* 0 (to float epsilon). Genuinely quantized data sits dead on the grid;
       raw-float / coarsened data does not (the project's real recordings give 0.0 on all
       fields, full-precision reconstructions give ~5e-5 with 0% coincidental on-grid).
  (II) ACTION ROUND-TRIP CLOSURE -- the recorded wire command rate_frd is a quantized
       PROJECTION of the recorded full-precision action act_rescaled (stored at ~16 digits,
       NOT on any decimal grid). The fly_rl pipeline is
           rate_flu = act_rescaled[1:4]            (* yaw_scale on z if !=1)
           rate_flu = _RZ_PI_BODY @ rate_flu       (if virtual_flip; a per-axis sign flip)
           rate_frd = rate_flu * _ACT_FLU_TO_FRD   (another per-axis sign flip)
       Every step between act and rate_frd is a magnitude-preserving +-1 per-axis sign op,
       so the SIGN-INVARIANT round-trip
           |rate_frd[ax]| == round(|act_rescaled[1+ax]| * gain[ax], 4)
       must hold EXACTLY (gain=1 except z=|yaw_scale|). This pins BOTH that the recording
       is genuinely quantized to round(4) AND that it is a faithful function of the action
       the policy emitted -- a coarsened/removed/decoupled rate_frd breaks it. We use the
       magnitude form so the test is robust to which per-axis sign convention a given
       fly_rl.py revision was on (the captured recordings span >=2 sign conventions on the
       roll/yaw axes; the QUANTIZATION GRID -- the thing P3-C06 is about -- is identical
       across all of them, which is exactly the invariant we want to lock).

Both negative controls below show the assertions FIRE if the bug were reintroduced.

DATA: handoff/shadowpc-{refit,postfix}-dataset-2026-06-12/extracted via
      tests/_audit_io.py (promoted from ultracode-substrate-audit-2026-06-13).
      (Does NOT import rl/contact_true_eval.py -- out of scope.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import pytest

# --- shared substrate (tests/_audit_io.py) lives beside this file ----------------------
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

import _audit_io as A  # noqa: E402

# refit/extracted is gitignored (only its .zip is tracked); postfix/extracted IS tracked.
# Skip (don't ERROR) on a checkout lacking either set -- on the canonical laptop both exist.
pytestmark = pytest.mark.skipif(
    not (A.dataset_present("postfix") and A.dataset_present("refit")),
    reason="ShadowPC audit recordings (postfix+refit extracted) not present",
)

# --- fly_rl.py action-pipeline constants (mirrored, NOT imported -- fly_rl pulls in heavy
#     MAVLink/torch deps and we only need three sign matrices). These are the magnitude-1
#     per-axis transforms between act_rescaled and the recorded rate_frd. ---------------
_RZ_PI_BODY = np.diag([-1.0, -1.0, 1.0])              # virtual body flip (pi about body z)
_ACT_FLU_TO_FRD = np.array([1.0, -1.0, 1.0])          # FLU rate -> FRD wire command

# Declared storage grids in fly_rl.py fly_once() (the contract under test). field -> ndigits.
# Key is the loader's output key (see _audit_io._FIELD_MAP); value is round(.,d) digits.
_DECLARED_GRID = {
    "obs": 5,
    "rate_frd": 4,
    "coll": 5,            # 'collective' -> loader key 'coll'
    "normed_thrust": 5,
    "pos": 4,             # 'pos_ned'
    "vel": 4,             # 'vel_ned'
    "q_raw": 6,           # 'q_raw_wxyz'
    "w_raw": 4,
}
_GRID_EPS = 1e-9          # "exactly on grid" tolerance (float round-trip noise only)


# =======================================================================================
# Core measurements (pure functions -- reused by the runnable main and the pytest fns)
# =======================================================================================
def _iter_runs():
    """Yield every loadable run across both datasets, tolerating header-only dirs."""
    for ds in ("refit", "postfix"):
        for rd in A.list_runs(ds):
            try:
                yield ds, A.load_run(rd)
            except ValueError:
                # header-only run (e.g. 20260612_184029_..._std_f2) -> skip, no retry.
                continue


def grid_offgrid(values: np.ndarray, ndigits: int) -> float:
    """max|x - round(x, ndigits)| -- 0 iff every value lies on the declared decimal grid."""
    v = np.asarray(values, dtype=float)
    return float(np.abs(v - np.round(v, ndigits)).max())


def reconstruct_rate_frd_magnitude(run: dict) -> np.ndarray:
    """Full-precision |rate_frd| reconstructed from the recorded act_rescaled via the
    fly_rl pipeline. Sign-invariant (every step is a +-1 per-axis flip), so this is the
    magnitude the recorded round(4) rate_frd must quantize to, regardless of the sign
    convention the capturing fly_rl.py revision used."""
    act = run["act_rescaled"][:, 1:4].copy()          # (N,3) full precision (~16 digits)
    ys = float(run["meta"].get("yaw_scale", 1.0))
    if ys != 1.0:
        act[:, 2] *= ys
    rate_flu = act
    if bool(run["meta"].get("virtual_flip", False)):
        rate_flu = rate_flu @ _RZ_PI_BODY.T           # row-wise _RZ_PI_BODY @ vec
    rate_frd_full = rate_flu * _ACT_FLU_TO_FRD
    return np.abs(rate_frd_full)


def measure() -> dict:
    """Run both invariants over every run. Returns aggregate worst-case numbers."""
    grid_worst = {k: 0.0 for k in _DECLARED_GRID}
    grid_worst_run = {k: "" for k in _DECLARED_GRID}
    rt_worst = 0.0
    rt_worst_run = ""
    n_runs = 0
    n_ticks = 0
    # negative-control accumulators (full-precision reconstruction = the rounding-removed bug)
    nc_removed_offgrid = 0.0      # off round(4) grid of the raw-float rate_frd magnitude
    nc_coarse_roundtrip = 0.0     # round-trip residual if rate_frd were logged at round(2)

    for _ds, run in _iter_runs():
        n_runs += 1
        n_ticks += run["n"]

        # (I) grid self-consistency on every present field
        for key, d in _DECLARED_GRID.items():
            if key not in run:
                continue
            off = grid_offgrid(run[key], d)
            if off > grid_worst[key]:
                grid_worst[key] = off
                grid_worst_run[key] = run["name"]

        # (II) action round-trip closure (only where the action pipeline is recorded)
        if "act_rescaled" in run and "rate_frd" in run:
            rec_mag = np.abs(run["rate_frd"])                 # recorded round(4) magnitude
            full_mag = reconstruct_rate_frd_magnitude(run)    # full-precision magnitude
            rt = float(np.abs(rec_mag - np.round(full_mag, 4)).max())
            if rt > rt_worst:
                rt_worst = rt
                rt_worst_run = run["name"]

            # NEGATIVE CONTROLS (computed on real data, not asserted as the pass condition):
            #  - rounding REMOVED: store full_mag raw -> off the declared round(4) grid.
            nc_removed_offgrid = max(nc_removed_offgrid, grid_offgrid(full_mag, 4))
            #  - quantization COARSENED to round(2): round-trip residual blows up to ~5e-3.
            coarse = np.round(np.round(full_mag, 4), 2)       # what a round(2) log would keep
            nc_coarse_roundtrip = max(
                nc_coarse_roundtrip, float(np.abs(coarse - np.round(full_mag, 4)).max()))

    return {
        "n_runs": n_runs,
        "n_ticks": n_ticks,
        "grid_worst": grid_worst,
        "grid_worst_run": grid_worst_run,
        "rt_worst": rt_worst,
        "rt_worst_run": rt_worst_run,
        "nc_removed_offgrid": nc_removed_offgrid,
        "nc_coarse_roundtrip": nc_coarse_roundtrip,
    }


# =======================================================================================
# pytest-style tests
# =======================================================================================
def test_recorded_fields_lie_exactly_on_declared_grid():
    """INVARIANT (I): every recorded field is *exactly* on its declared round(d) grid.

    FAILS if rounding is removed/changed in fly_rl.py (raw floats are generically NOT on
    the grid -> off-grid > _GRID_EPS). NEGATIVE CONTROL inside: the full-precision rate_frd
    magnitude (== what a rounding-removed bug would store) is shown to be detectably off
    the round(4) grid, so the assertion has real discriminating power."""
    m = measure()
    assert m["n_runs"] >= 10, f"expected the full dataset, only loaded {m['n_runs']} runs"
    for key, d in _DECLARED_GRID.items():
        off = m["grid_worst"][key]
        assert off <= _GRID_EPS, (
            f"P3-C06: field {key!r} is OFF its declared round({d}) grid by {off:.2e} "
            f"(> {_GRID_EPS:.0e}) in run {m['grid_worst_run'][key]} -- recording "
            f"quantization changed/removed.")
    # negative control: the rounding-removed reconstruction MUST be off-grid (else the
    # round(4) grid would be degenerate and the test above would be vacuous).
    assert m["nc_removed_offgrid"] > _GRID_EPS, (
        "negative control failed: full-precision rate_frd magnitude is already on the "
        "round(4) grid -- the grid-membership assertion would not catch removed rounding.")


def test_action_pipeline_round_trip_closes_on_the_round4_grid():
    """INVARIANT (II): recorded |rate_frd| == round(|act_rescaled-derived rate|, 4), exactly.

    Proves the wire command is a faithful round(4) projection of the recorded full-precision
    action -- catches a coarsened/removed/decoupled rate_frd. NEGATIVE CONTROL inside: a
    round(2) log of the same rates breaks the round-trip by ~5e-3 (>> the 5e-5 grid)."""
    m = measure()
    grid_halfstep = 0.5e-4                       # round(4) half-step
    assert m["rt_worst"] <= grid_halfstep, (
        f"P3-C06: recorded rate_frd does not reconstruct from act_rescaled to within the "
        f"round(4) grid -- worst |rec-round(full,4)| = {m['rt_worst']:.2e} "
        f"(> {grid_halfstep:.0e}) in run {m['rt_worst_run']}. Either the action recording "
        f"is decoupled from the wire command or the quantization was coarsened.")
    # negative control: a round(2) quantization of the same data WOULD breach the grid.
    assert m["nc_coarse_roundtrip"] > grid_halfstep, (
        "negative control failed: coarsening rate_frd to round(2) did not breach the "
        "round(4) grid -- the round-trip assertion would not catch a coarsening bug.")


# =======================================================================================
# Runnable main -- prints PASS/FAIL + the key numbers
# =======================================================================================
def main() -> int:
    print("=" * 78)
    print("REGRESSION test_confirmed_p3_c06 -- recording quantization (rl/fly_rl.py)")
    print("=" * 78)
    m = measure()
    print(f"loaded {m['n_runs']} runs / {m['n_ticks']} ticks "
          f"(refit + postfix, header-only runs skipped)\n")

    ok = True

    # (I) grid self-consistency
    print("INVARIANT (I) grid self-consistency  max|x - round(x,d)|  (must be ~0):")
    for key, d in _DECLARED_GRID.items():
        off = m["grid_worst"][key]
        flag = "OK " if off <= _GRID_EPS else "BAD"
        if off > _GRID_EPS:
            ok = False
        print(f"   [{flag}] {key:14s} round({d})  off-grid = {off:.3e}  "
              f"(bound {_GRID_EPS:.0e})")
    print(f"   negative control (rounding removed): raw rate_frd off round(4) grid = "
          f"{m['nc_removed_offgrid']:.3e}  -> assertion WOULD fire (good)\n")

    # (II) action round-trip
    print("INVARIANT (II) action round-trip  max| |rate_frd| - round(|act-derived|,4) |:")
    rt_ok = m["rt_worst"] <= 0.5e-4
    ok = ok and rt_ok
    print(f"   [{'OK ' if rt_ok else 'BAD'}] worst = {m['rt_worst']:.3e}  "
          f"(bound 5e-05)  worst run = {m['rt_worst_run'] or '-'}")
    print(f"   negative control (coarsen to round(2)): round-trip residual = "
          f"{m['nc_coarse_roundtrip']:.3e}  -> assertion WOULD fire (good)\n")

    print("RESULT:", "PASS" if ok else "FAIL")
    print("guards: recording quantization grids (obs r5 / rate_frd r4 / pose r4-6) AND the "
          "act_rescaled->rate_frd round-trip stay numerically faithful in fly_rl.py.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
