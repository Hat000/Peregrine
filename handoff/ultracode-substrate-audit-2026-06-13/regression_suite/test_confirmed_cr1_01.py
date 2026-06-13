"""REGRESSION — CONFIRMED BUG CR1-01: yaw command -> wire-rate SIGN FLIP.

BUG CLASS THIS GUARDS
---------------------
A per-axis SIGN/CONVENTION flip on the YAW channel of the RL action->wire seam
(the `_ACT_FLU_TO_FRD` FLU->FRD mapping in rl/fly_rl.py, interacting with the
`virtual_flip` body-Z rotation). In the buggy era the policy's commanded yaw rate
was sent to the vehicle with the WRONG sign: a positive (CCW) yaw demand produced a
negative (CW) realized yaw rate. This is the same family of defect that has bitten
this project four times (R_y(pi) ODOMETRY-quat conjugation, body-vs-world velocity
mix, corner_to_center 180deg flip, ATTITUDE.pitch sign inversion): a proper-rotation
conjugation or single-axis sign-alias.

WHY INTERNAL-CONSISTENCY CHECKS MISS IT
---------------------------------------
The wire command is logged (`rate_frd`) right next to the rescaled action
(`act_rescaled`). Checking `rate_frd == act_rescaled[1:4] * _ACT_FLU_TO_FRD`
(possibly with the `virtual_flip` body-Z rotation pre-applied) PASSES in BOTH the
buggy and the corrected recordings — because in each era the harness logged
`rate_frd` THROUGH WHATEVER `_ACT_FLU_TO_FRD` CONSTANT IT WAS USING. The logged
identity is self-consistent with the (possibly buggy) constant that produced it. A
quat-vs-rate finite-difference, level-flight correlation, or twist round-trip is
likewise blind: a single-axis sign flip on a body rate looks internally coherent.

The ONLY thing that discriminates is an EXTERNAL realized-physics invariant: compare
the policy's INTENT (commanded FLU yaw) against what the vehicle ACTUALLY DID, read
from an INDEPENDENT telemetry channel (the ODOMETRY angular-rate `w_raw`), NOT from
any logged derived constant. The true body yaw rate is `-w_raw[:,2]` (ODOMETRY rate
is R_y(pi)-conjugated; true rate = -w_raw on all axes). With a correct yaw
convention a positive commanded yaw must produce a positive realized yaw rate:

    corr( true_yaw_rate = -w_raw[:,2] ,  flu_yaw_cmd = act_rescaled[:,3] )  >  0

THE INVARIANT (what this test encodes — NOT the current code's logged behavior)
-------------------------------------------------------------------------------
  CORRECTED convention  =>  corr(true_yaw_rate, commanded_flu_yaw)  is STRONGLY POSITIVE.
  BUGGY (flipped) convention => the same correlation is STRONGLY NEGATIVE.

If the CR1-01 yaw-sign flip were (re)introduced into the deployment seam, the
realized-physics correlation in fresh recordings would go negative and THIS TEST
WOULD FAIL. The buggy-era `refit` recordings are kept here as a built-in NEGATIVE
CONTROL proving the test has the power to catch the flip (their correlation is
negative, by construction of the bug). The era-independent East attitude canary
(FD of pristine vel_ned vs attitude-derived force, tilt>35deg) is the POSITIVE
CONTROL confirming the substrate discriminates and the yaw finding is not
attitude-frame bleed.

DATA / SUBSTRATE
----------------
  postfix = CORRECTED-era recordings (POST VISION-FRAME-FIX 8d7b0b3; the yaw wire
            convention that matches the CURRENTLY COMMITTED rl/fly_rl.py
            _ACT_FLU_TO_FRD=[1,-1,1] under virtual_flip). EXPECTED: corr > 0.
  refit   = BUGGY-era recordings (yaw wire flipped). EXPECTED: corr < 0  (neg control).
Both via handoff/ultracode-substrate-audit-2026-06-13/scratch/_audit_io.py.
Same checkpoint (stage1_inc6_actor.pth) in both — the flip is in the SEAM, not the policy.

Does NOT import rl/contact_true_eval.py (out of scope; edited elsewhere).

RUN
---
  .venv/Scripts/python.exe handoff/ultracode-substrate-audit-2026-06-13/regression_suite/test_confirmed_cr1_01.py
or as pytest:
  .venv/Scripts/python.exe -m pytest handoff/ultracode-substrate-audit-2026-06-13/regression_suite/test_confirmed_cr1_01.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# --- locate the shared substrate helper (scratch/_audit_io.py) -------------------------
_HERE = Path(__file__).resolve()
# .../handoff/ultracode-substrate-audit-2026-06-13/regression_suite/test_confirmed_cr1_01.py
_SCRATCH = _HERE.parents[1] / "scratch"
if str(_SCRATCH) not in sys.path:
    sys.path.insert(0, str(_SCRATCH))

import _audit_io as A  # noqa: E402  (path-dependent import)

# ---------------------------------------------------------------------------------------
# Thresholds. The CORRECTED convention gives ~+0.83 pooled; the BUGGY convention ~-0.98.
# We assert with a wide margin from zero so the test is a sign/convention check, not a
# brittle magnitude check, yet still fails immediately if the yaw sign flips.
# ---------------------------------------------------------------------------------------
CORR_MIN_POSITIVE = 0.50   # corrected era must clear this (observed ~+0.83)
CORR_MAX_NEGATIVE = -0.50  # buggy era (neg control) must be below this (observed ~-0.98)
EAST_CANARY_MIN = 0.85     # positive-control: TRUE conjugation East corr (observed ~+0.97..1.0)
EAST_CANARY_ASIS_MAX = -0.30  # positive-control: AS-IS East corr must be negative (mirror present)


# =======================================================================================
# Core external invariant: realized yaw rate vs commanded yaw
# =======================================================================================
def _yaw_realized_corr(run: dict) -> tuple[float, int]:
    """corr( true body yaw rate = -w_raw[:,2] ,  commanded FLU yaw = act_rescaled[:,3] ).

    REALIZED-PHYSICS invariant: the yaw rate is read from the ODOMETRY angular-rate
    telemetry channel (w_raw), the *measured* rotation the vehicle underwent — NOT from
    the logged rate_frd nor any _ACT_FLU_TO_FRD constant. This is independent of whatever
    wire convention the harness logged, so it cannot be fooled by the self-consistent
    logged identity.

      true_yaw_rate = -w_raw[:,2]    (ODOMETRY rate is R_y(pi)-conjugated; true = -raw)
      flu_yaw_cmd   =  act_rescaled[:,3]   (policy's body-FLU yaw-rate command, rad/s)
    """
    if "w_raw" not in run or "act_rescaled" not in run:
        raise KeyError(f"run {run.get('name')} missing w_raw/act_rescaled")
    true_yaw = -run["w_raw"][:, 2]
    flu_yaw_cmd = run["act_rescaled"][:, 3]
    if true_yaw.std() < 1e-6 or flu_yaw_cmd.std() < 1e-6:
        return float("nan"), len(true_yaw)
    return float(np.corrcoef(true_yaw, flu_yaw_cmd)[0, 1]), len(true_yaw)


def _pooled_yaw_corr(dataset: str) -> dict:
    """Pool the realized-physics yaw correlation across every loadable run in a dataset.

    Tolerates header-only runs (load_run raises ValueError -> skip; documented caveat:
    20260612_184029_..._std_f2 has 0 data rows). Returns pooled corr + per-run array.
    """
    true_all, cmd_all, per_run = [], [], []
    skipped = []
    for run_dir in A.list_runs(dataset):
        try:
            d = A.load_run(run_dir)
        except ValueError:
            skipped.append(run_dir.name)
            continue
        c, n = _yaw_realized_corr(d)
        per_run.append((run_dir.name, c, n))
        true_all.append(-d["w_raw"][:, 2])
        cmd_all.append(d["act_rescaled"][:, 3])
    if not true_all:
        raise RuntimeError(f"no loadable runs in dataset {dataset!r}")
    ty = np.concatenate(true_all)
    cmd = np.concatenate(cmd_all)
    pooled = float(np.corrcoef(ty, cmd)[0, 1])
    corrs = np.array([c for _, c, _ in per_run], dtype=float)
    return {
        "dataset": dataset,
        "pooled_corr": pooled,
        "n_samples": int(len(ty)),
        "per_run": per_run,
        "per_run_corrs": corrs,
        "all_same_sign": bool(np.all(corrs < 0) or np.all(corrs > 0)),
        "min_corr": float(np.nanmin(corrs)),
        "max_corr": float(np.nanmax(corrs)),
        "skipped": skipped,
    }


# =======================================================================================
# Positive control: era-independent East attitude canary (substrate-discriminates check)
# =======================================================================================
def _east_canary(dataset: str) -> dict:
    """Pool the canonical Test-A East-axis corr for TRUE vs AS-IS conjugation.

    This is the external invariant that proves the SUBSTRATE discriminates frame bugs at
    all (FD of pristine vel_ned vs attitude-derived force, tilt>35deg). It is INDEPENDENT
    of the yaw seam, so it isolates the CR1-01 finding from attitude-frame bleed. TRUE
    East corr must be strongly positive; AS-IS East corr must be negative (the R_y(pi)
    quat mirror is present in BOTH datasets -> positive control either era).
    """
    mT, eT, mA, eA, tlT, tlA = [], [], [], [], [], []  # noqa: collect model/meas per axis is heavy
    # Re-pool at the sample level for an honest pooled corr (matches banked figures).
    model_T, meas_T, tilt_T = [], [], []
    model_A, meas_A, tilt_A = [], [], []
    for run_dir in A.list_runs(dataset):
        try:
            d = A.load_run(run_dir)
        except ValueError:
            continue
        t, vel, q, coll = d["t"], d["vel"], d["q_raw"], d["coll"]
        n = len(t)
        R_T = A.Rmats(q, sign=A.CAND_TRUE)
        R_A = A.Rmats(q, sign=A.CAND_ASIS)
        tilt = A.tilt_deg(R_A)  # tilt invariant across candidates
        for k in range(2, n - 2):
            dt = t[k + 1] - t[k - 1]
            if not (0.05 < dt < 0.09):
                continue
            a_meas = (vel[k + 1] - vel[k - 1]) / dt
            if not np.all(np.isfinite(a_meas)) or np.max(np.abs(a_meas)) > 90:
                continue
            coll_d = coll[k - 2]
            model_T.append(A.force_model(R_T[k], vel[k], coll_d))
            model_A.append(A.force_model(R_A[k], vel[k], coll_d))
            meas_T.append(a_meas)
            tilt_T.append(tilt[k])
    model_T = np.array(model_T); meas_T = np.array(meas_T); tl = np.array(tilt_T)
    model_A = np.array(model_A)
    m = tl > 35.0
    if m.sum() < 20:
        return {"dataset": dataset, "east_true": float("nan"),
                "east_asis": float("nan"), "n_tilted": int(m.sum())}
    east_true = float(np.corrcoef(model_T[m, 1], meas_T[m, 1])[0, 1])
    east_asis = float(np.corrcoef(model_A[m, 1], meas_T[m, 1])[0, 1])
    return {"dataset": dataset, "east_true": east_true, "east_asis": east_asis,
            "n_tilted": int(m.sum())}


# =======================================================================================
# pytest-style entry points (assertions encode the EXTERNAL invariant)
# =======================================================================================
def test_corrected_era_yaw_sign_is_positive():
    """CORRECTED convention (postfix, == current committed _ACT_FLU_TO_FRD): yaw corr > 0.

    Fails if the CR1-01 yaw-sign flip is (re)introduced into the deployment seam.
    """
    res = _pooled_yaw_corr("postfix")
    assert res["pooled_corr"] > CORR_MIN_POSITIVE, (
        f"CR1-01 REGRESSION: corrected-era realized yaw corr = {res['pooled_corr']:+.3f} "
        f"(expected > {CORR_MIN_POSITIVE}). A NEGATIVE value means commanded yaw produces "
        f"OPPOSITE realized yaw -> the FLU->FRD yaw sign flip is back. per-run="
        f"{[(n[:30], round(c, 2)) for n, c, _ in res['per_run']]}"
    )
    assert res["all_same_sign"], (
        f"corrected-era per-run yaw corr not single-signed: "
        f"min={res['min_corr']:+.3f} max={res['max_corr']:+.3f}"
    )


def test_buggy_era_yaw_sign_is_negative_negative_control():
    """NEGATIVE CONTROL: buggy-era (refit) yaw corr is strongly negative.

    Proves the test has discriminating power — under the flipped convention the realized
    yaw rate anti-correlates with the command. If THIS ever goes positive the substrate
    or the invariant has lost its teeth (investigate before trusting the positive test).
    """
    res = _pooled_yaw_corr("refit")
    assert res["pooled_corr"] < CORR_MAX_NEGATIVE, (
        f"NEGATIVE CONTROL FAILED: buggy-era realized yaw corr = {res['pooled_corr']:+.3f} "
        f"(expected < {CORR_MAX_NEGATIVE}). The known-buggy recordings should anti-correlate; "
        f"if they don't, this regression cannot prove it would catch the flip."
    )
    assert res["all_same_sign"], (
        f"buggy-era per-run yaw corr not single-signed: "
        f"min={res['min_corr']:+.3f} max={res['max_corr']:+.3f}"
    )


def test_clean_sign_flip_between_eras():
    """The two eras must be on OPPOSITE sides of zero with no overlap.

    Encodes the CR1-01 headline: a clean sign flip (refit all -1, postfix all +1),
    zero overlap, not a small-N artifact.
    """
    post = _pooled_yaw_corr("postfix")
    buggy = _pooled_yaw_corr("refit")
    assert buggy["max_corr"] < 0.0 < post["min_corr"], (
        f"CR1-01 sign-flip not clean: buggy max={buggy['max_corr']:+.3f}, "
        f"corrected min={post['min_corr']:+.3f} (must straddle 0 with no overlap)."
    )


def test_east_canary_positive_control_substrate_discriminates():
    """POSITIVE CONTROL: the era-independent East attitude canary still discriminates.

    Confirms the yaw finding is not attitude-frame bleed and the substrate can see frame
    bugs at all: TRUE conjugation East corr strongly positive, AS-IS negative (R_y(pi)
    quat mirror present in BOTH datasets).
    """
    for ds in ("postfix", "refit"):
        c = _east_canary(ds)
        assert c["east_true"] > EAST_CANARY_MIN, (
            f"[{ds}] East canary TRUE corr = {c['east_true']:+.3f} "
            f"(expected > {EAST_CANARY_MIN}); substrate may have lost discriminating power."
        )
        assert c["east_asis"] < EAST_CANARY_ASIS_MAX, (
            f"[{ds}] East canary AS-IS corr = {c['east_asis']:+.3f} "
            f"(expected < {EAST_CANARY_ASIS_MAX}); the R_y(pi) mirror positive control failed."
        )


# =======================================================================================
# Standalone runner — prints PASS/FAIL + key numbers
# =======================================================================================
def _main() -> int:
    print("=" * 78)
    print("REGRESSION CR1-01 — yaw command->wire SIGN FLIP (realized-physics invariant)")
    print("  invariant: corr(true_yaw_rate=-w_raw[:,2], commanded_flu_yaw=act_rescaled[:,3])")
    print("  CORRECTED convention => POSITIVE ;  BUGGY (flipped) => NEGATIVE")
    print("=" * 78)

    post = _pooled_yaw_corr("postfix")
    buggy = _pooled_yaw_corr("refit")

    print("\n--- realized-physics yaw correlation (pooled over runs) ---")
    print(f"  postfix (CORRECTED == current committed _ACT_FLU_TO_FRD=[1,-1,1]):")
    print(f"      pooled corr = {post['pooled_corr']:+.3f}   N={post['n_samples']:6d}   "
          f"per-run [{post['min_corr']:+.3f}, {post['max_corr']:+.3f}]  "
          f"single-sign={post['all_same_sign']}")
    if post["skipped"]:
        print(f"      (skipped header-only runs: {post['skipped']})")
    print(f"  refit   (BUGGY yaw flip — NEGATIVE CONTROL):")
    print(f"      pooled corr = {buggy['pooled_corr']:+.3f}   N={buggy['n_samples']:6d}   "
          f"per-run [{buggy['min_corr']:+.3f}, {buggy['max_corr']:+.3f}]  "
          f"single-sign={buggy['all_same_sign']}")

    print("\n--- positive control: era-independent East attitude canary (tilt>35deg) ---")
    canaries = {}
    for ds in ("postfix", "refit"):
        c = _east_canary(ds)
        canaries[ds] = c
        print(f"  {ds:8s} East corr  TRUE[1,-1,1,-1]={c['east_true']:+.3f}  "
              f"AS-IS[1,1,1,1]={c['east_asis']:+.3f}  (n_tilt={c['n_tilted']})")

    # --- verdicts ---
    checks = []
    checks.append(("corrected-era yaw corr POSITIVE",
                   post["pooled_corr"] > CORR_MIN_POSITIVE and post["all_same_sign"]))
    checks.append(("buggy-era yaw corr NEGATIVE (neg control)",
                   buggy["pooled_corr"] < CORR_MAX_NEGATIVE and buggy["all_same_sign"]))
    checks.append(("clean sign flip, no overlap",
                   buggy["max_corr"] < 0.0 < post["min_corr"]))
    checks.append(("East canary TRUE>+0.85 & AS-IS<-0.30 both eras",
                   all(canaries[ds]["east_true"] > EAST_CANARY_MIN
                       and canaries[ds]["east_asis"] < EAST_CANARY_ASIS_MAX
                       for ds in ("postfix", "refit"))))

    print("\n--- checks ---")
    all_ok = True
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name}")
        all_ok = all_ok and ok

    print("\n" + "=" * 78)
    print(f"RESULT: {'PASS' if all_ok else 'FAIL'}  — guards: yaw command->wire SIGN FLIP "
          f"(CR1-01)")
    print("=" * 78)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
