"""EXTERNAL-INVARIANT FRAME CANARY -- force-vs-FD mirror regression test.

================================================================================
BUG CLASS THIS CATCHES
================================================================================
A re-introduced ODOMETRY-quaternion conjugation / proper-rotation sign-alias in
the deploy or vision attitude path. Concretely: the sim ODOMETRY quat is
R_y(pi)-CONJUGATED in the raw telemetry, so the TRUE world<-body attitude is
    q_true = q_raw * [1, -1, 1, -1]   (negate x and z in wxyz)
and using the raw quat AS-IS [1, 1, 1, 1] is WRONG. This project has been bitten
FOUR times by exactly this family of bug:
    1. R_y(pi) ODOMETRY-quat conjugation (the one probed here),
    2. body-vs-world velocity-frame mix (twist stored as world),
    3. corner_to_center 180deg flip,
    4. ATTITUDE.pitch sign inversion.
If any code path silently swaps back to the AS-IS quat (or otherwise mirrors the
attitude), this test FAILS.

================================================================================
WHY INTERNAL-CONSISTENCY CHECKS MISS IT (and only an EXTERNAL invariant catches it)
================================================================================
A proper-rotation conjugation is a *self-consistent* relabeling of the body
frame. So every INTERNAL check passes while the convention is wrong:
    * quat-vs-rate finite-difference agrees (rate is mirrored to match),
    * level-flight attitude correlation agrees (body-down == world-down when level
      -> a conjugate looks identical to the truth),
    * twist (body-velocity) round-trip closes (both sides use the same mirror).
The ONLY thing that discriminates is an EXTERNAL invariant anchored to a quantity
that is NOT derived from the (possibly-mirrored) attitude. The canonical anchor is
the PRISTINE state.velocity_ned (trusted world-NED). We finite-difference it to get
measured specific force and compare against the ATTITUDE-DERIVED specific force
    a_model = K*(R @ [0,0,-1]) + quad_drag(R, v) + [0,0,g]
in the TILTED bin (tilt > 35 deg). Level flight is INADMISSIBLE: when the body
z-axis aligns with world-down, a rotation and its conjugate produce identical
thrust direction, so the test has no discriminating power there.

DISCRIMINATING AXIS: EAST. A roll mirror leaves North/Down nearly invariant, so
N/D corr stay ~+0.9 for BOTH candidates and are BLIND to the bug. East is the
tell: TRUE [1,-1,1,-1] correlates ~+0.97..+0.99 on East; AS-IS [1,1,1,1] goes
firmly NEGATIVE (~ -0.7..-0.84). Judge the convention on EAST only.

POSITIVE CONTROL: the raw-telemetry mirror is present in BOTH the 'refit' and
'postfix' datasets (the names describe the capture session, NOT removal of the
telemetry mirror), so this test ALSO asserts the AS-IS candidate is anti-correlated
on East -- proving the probe still has discriminating power. If a future telemetry
change ever removed the mirror, the positive-control assertion would fail loudly and
tell us the canary needs re-derivation, rather than silently passing.

================================================================================
DATA / METHOD
================================================================================
Recordings: handoff/shadowpc-{refit,postfix}-dataset-2026-06-12/extracted, loaded
via the shared substrate handoff/ultracode-substrate-audit-2026-06-13/scratch/
_audit_io.py (single loader + single force model, factored out of the reference
audit_candidates.py). We POOL across all runs in a dataset (the per-run East corr
is the same canary with less data: ~ -0.68/-0.75 per-run vs the banked pooled
~ -0.81); pooling reproduces the banked figure (TRUE East +0.99 / AS-IS East -0.81)
and is robust to any one run being short or noisy.

Production-faithful force uses use_lapse=False (S18 voided LAPSE in production); we
verify the canary holds under the production force model, not just the reference one.

DOES NOT import rl/contact_true_eval.py (out of scope; edited elsewhere).

Run directly:   .venv/Scripts/python.exe <this file>     (prints PASS/FAIL + numbers)
Run via pytest: pytest <this file>                        (test_*() with assertions)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import pytest

# --- shared substrate (tests/_audit_io.py) lives beside this file --------------------
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

import _audit_io as A  # noqa: E402  (shared loader + Test-A force model)

# refit/extracted is gitignored (only its .zip is tracked); postfix/extracted IS tracked.
# Skip (don't ERROR) on a checkout lacking either set -- on the canonical laptop both exist.
pytestmark = pytest.mark.skipif(
    not (A.dataset_present("postfix") and A.dataset_present("refit")),
    reason="ShadowPC audit recordings (postfix+refit extracted) not present",
)


# =====================================================================================
# Thresholds for the EXTERNAL invariant (East-axis is the discriminating axis)
# =====================================================================================
# Banked pooled figures: TRUE East +0.97..+0.99 ; AS-IS East ~ -0.81 (refit), ~ -0.71 (postfix).
# Thresholds are deliberately loose enough to tolerate dataset/run mix but tight enough
# that the bug (AS-IS) cannot pass and the truth (TRUE) cannot fail.
TRUE_EAST_CORR_MIN = 0.90    # TRUE conjugation must correlate strongly positive on East
ASIS_EAST_CORR_MAX = -0.40   # AS-IS (positive control) must be clearly NEGATIVE on East
TILT_MIN_DEG = 35.0          # only the tilted bin discriminates; level flight is blind
MIN_TILTED_SAMPLES = 200     # require a meaningful tilted-sample pool before judging


def _pooled_force_vs_fd(dataset: str, sign, *, use_lapse: bool, tilt_min: float = TILT_MIN_DEG):
    """Pool the Test-A force-vs-FD residuals across every loadable run in a dataset.

    Returns dict: per-axis {'corr','med_abs_res'}, 'n_tilted', 'n_runs', 'skipped'.
    Reuses _audit_io.test_a_force_vs_fd per run and concatenates the tilted-bin
    (model, measured) pairs so the correlation is computed on the pooled cloud.
    """
    runs = A.list_runs(dataset)
    pooled_model = {0: [], 1: [], 2: []}
    pooled_meas = {0: [], 1: [], 2: []}
    n_runs = 0
    skipped = []

    for run_dir in runs:
        try:
            run = A.load_run(run_dir)
        except ValueError:
            # header-only / empty run (e.g. 20260612_184029_..._std_f2) -- tolerate & skip.
            skipped.append(run_dir.name)
            continue

        # Re-derive the tilted-bin (model, meas) pairs exactly as test_a_force_vs_fd does,
        # but keep the raw paired samples so we can pool the correlation across runs.
        t, vel, q, coll = run["t"], run["vel"], run["q_raw"], run["coll"]
        n = len(t)
        if n < 6:
            skipped.append(run_dir.name)
            continue
        R = A.Rmats(q, sign=sign)
        tilt_raw = A.tilt_deg(A.Rmats(q, sign=A.CAND_ASIS))  # tilt invariant across candidates
        for k in range(2, n - 2):
            dt = t[k + 1] - t[k - 1]
            if not (0.05 < dt < 0.09):
                continue
            a_meas = (vel[k + 1] - vel[k - 1]) / dt
            if not np.all(np.isfinite(a_meas)) or np.max(np.abs(a_meas)) > 90:
                continue
            if tilt_raw[k] <= tilt_min:
                continue
            a_model = A.force_model(R[k], vel[k], coll[k - 2], use_lapse=use_lapse)
            for ax in range(3):
                pooled_model[ax].append(a_model[ax])
                pooled_meas[ax].append(a_meas[ax])
        n_runs += 1

    AX = ["N", "E", "D"]
    out = {"n_tilted": len(pooled_model[1]), "n_runs": n_runs, "skipped": skipped, "axes": {}}
    for ax in range(3):
        md = np.array(pooled_model[ax])
        ms = np.array(pooled_meas[ax])
        if md.size < 5 or md.std() < 1e-6:
            out["axes"][AX[ax]] = {"corr": float("nan"), "med_abs_res": float("nan")}
            continue
        out["axes"][AX[ax]] = {
            "corr": float(np.corrcoef(md, ms)[0, 1]),
            "med_abs_res": float(np.median(np.abs(md - ms))),
        }
    return out


def _evaluate(dataset: str, *, use_lapse: bool):
    """Compute pooled TRUE vs AS-IS for one dataset; return (true_res, asis_res)."""
    true_res = _pooled_force_vs_fd(dataset, A.CAND_TRUE, use_lapse=use_lapse)
    asis_res = _pooled_force_vs_fd(dataset, A.CAND_ASIS, use_lapse=use_lapse)
    return true_res, asis_res


# =====================================================================================
# pytest entry point
# =====================================================================================
def test_frame_force_vs_fd_mirror_canary():
    """External-invariant canary: TRUE R_y(pi) conjugation must win on the EAST axis.

    Asserted on the production-faithful force model (use_lapse=False) over the 'refit'
    dataset (largest tilted-sample pool). Two assertions:
      (1) TRUE   [1,-1,1,-1] East corr >= TRUE_EAST_CORR_MIN     (the truth holds)
      (2) AS-IS  [1, 1,1, 1] East corr <= ASIS_EAST_CORR_MAX     (positive control: the
          mirror bug is anti-correlated -> the probe has discriminating power)
    """
    true_res, asis_res = _evaluate("refit", use_lapse=False)

    n_tilt = true_res["n_tilted"]
    assert n_tilt >= MIN_TILTED_SAMPLES, (
        f"too few tilted (>{TILT_MIN_DEG} deg) samples to judge the convention: "
        f"{n_tilt} < {MIN_TILTED_SAMPLES}. Check the recordings/dataset."
    )

    true_e = true_res["axes"]["E"]["corr"]
    asis_e = asis_res["axes"]["E"]["corr"]

    # (1) The TRUE conjugation must correlate strongly positive on East.
    assert true_e >= TRUE_EAST_CORR_MIN, (
        f"FRAME CANARY TRIPPED: TRUE [1,-1,1,-1] East corr {true_e:+.3f} "
        f"< {TRUE_EAST_CORR_MIN:+.2f}. The external invariant (FD of pristine vel_ned "
        f"vs attitude-derived force) no longer favors the R_y(pi) conjugation -- a quat "
        f"conjugation/sign mirror may have been (re)introduced in the deploy/vision "
        f"attitude path, or the recordings/force-model changed. Re-derive the convention."
    )

    # (2) Positive control: the AS-IS (mirror-present) candidate MUST be anti-correlated
    #     on East. If this fails the test has lost discriminating power (e.g. telemetry
    #     mirror removed) -- surface it loudly, do NOT let the canary pass silently.
    assert asis_e <= ASIS_EAST_CORR_MAX, (
        f"POSITIVE-CONTROL LOST: AS-IS [1,1,1,1] East corr {asis_e:+.3f} "
        f"> {ASIS_EAST_CORR_MAX:+.2f} (expected clearly negative). The discriminating "
        f"signal is gone -- the raw-telemetry quat mirror may no longer be present, so "
        f"this canary can no longer catch a re-introduced conjugation. Re-derive the test."
    )


# =====================================================================================
# Standalone runner
# =====================================================================================
def _fmt(res) -> str:
    cells = []
    for ax in ("N", "E", "D"):
        a = res["axes"][ax]
        cells.append(f"{ax}: med|r|={a['med_abs_res']:6.2f} corr={a['corr']:+6.3f}")
    return "  ".join(cells)


def main() -> int:
    print("=" * 78)
    print("EXTERNAL-INVARIANT FRAME CANARY  --  force(vel_ned-FD) vs attitude-derived")
    print("Discriminating axis = EAST (a roll mirror leaves N/D nearly invariant).")
    print(f"Tilt bin > {TILT_MIN_DEG:.0f} deg.  Production force model (use_lapse=False).")
    print("=" * 78)

    overall_ok = True
    detail = {}
    for dataset in ("refit", "postfix"):
        try:
            # report under BOTH force models for transparency; ASSERT on production (no-lapse)
            true_p, asis_p = _evaluate(dataset, use_lapse=False)   # production-faithful
            true_r, asis_r = _evaluate(dataset, use_lapse=True)    # reference-parity
        except (FileNotFoundError, ValueError) as e:
            print(f"\n[{dataset}] dataset unavailable: {e}")
            overall_ok = False
            continue

        skipped = true_p["skipped"]
        print(f"\n[{dataset}]  n_runs={true_p['n_runs']}  n_tilted(East)={true_p['n_tilted']}"
              + (f"  skipped(empty)={skipped}" if skipped else ""))
        print("  -- production force (use_lapse=False) [ASSERTED] --")
        print(f"    TRUE  [1,-1,1,-1]  {_fmt(true_p)}")
        print(f"    AS-IS [1, 1,1, 1]  {_fmt(asis_p)}")
        print("  -- reference-parity force (use_lapse=True)  [report only] --")
        print(f"    TRUE  [1,-1,1,-1]  {_fmt(true_r)}")
        print(f"    AS-IS [1, 1,1, 1]  {_fmt(asis_r)}")

        true_e = true_p["axes"]["E"]["corr"]
        asis_e = asis_p["axes"]["E"]["corr"]
        ok_true = true_e >= TRUE_EAST_CORR_MIN
        ok_asis = asis_e <= ASIS_EAST_CORR_MAX
        ok_n = true_p["n_tilted"] >= MIN_TILTED_SAMPLES
        print(f"    -> TRUE East corr {true_e:+.3f} >= {TRUE_EAST_CORR_MIN:+.2f} ? "
              f"{'PASS' if ok_true else 'FAIL'}")
        print(f"    -> AS-IS East corr {asis_e:+.3f} <= {ASIS_EAST_CORR_MAX:+.2f} "
              f"(positive control) ? {'PASS' if ok_asis else 'FAIL'}")
        print(f"    -> tilted samples {true_p['n_tilted']} >= {MIN_TILTED_SAMPLES} ? "
              f"{'PASS' if ok_n else 'FAIL'}")
        detail[dataset] = (ok_true and ok_asis and ok_n)

    # The pytest assertion judges on 'refit' (production force). Mirror that for the verdict.
    verdict_ok = detail.get("refit", False) and overall_ok
    print("\n" + "=" * 78)
    print(f"VERDICT (asserted on 'refit', production force): "
          f"{'PASS' if verdict_ok else 'FAIL'}")
    print("Guards: re-introduced ODOMETRY-quat conjugation / attitude sign-mirror in the "
          "deploy or vision attitude path.")
    print("=" * 78)
    return 0 if verdict_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
