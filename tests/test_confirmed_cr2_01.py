"""Regression for CONFIRMED bug CR2-01 (rl/fly_rl.py yaw-channel convention).

================================================================================
BUG CLASS THIS GUARDS
================================================================================
A YAW-ABOUT-VERTICAL sign-convention error in the ODOMETRY rate read
(`_ODO_RATE_SIGN[2]`) and/or the action FLU->FRD wire map (`_ACT_FLU_TO_FRD[2]`).
This is the *yaw* member of the same proper-rotation sign-alias family that has
bitten this project 4x (R_y(pi) ODOMETRY-quat conjugation, body-vs-world velocity
frame, corner_to_center 180deg flip, ATTITUDE.pitch sign inversion).

WHY INTERNAL-CONSISTENCY CHECKS MISS IT
---------------------------------------
A wrong yaw sign is a proper rotation about the world vertical. Two facts make it
invisible to the usual checks:

  1. The canonical EXTERNAL force-vs-FD invariant (Test A in _audit_io) is
     STRUCTURALLY BLIND to yaw-about-vertical: thrust points along body -z, and a
     pure rotation about world-down leaves the world-NED specific force unchanged.
     Test A discriminates the East mirror (roll/quat) but says NOTHING about yaw.

  2. quat-FD-vs-rate, level-flight correlation and twist round-trip are all
     INTERNAL: they compare two telemetry channels (or one channel to itself).
     A self-consistent mirror (e.g. bcc93f9: quat-as-is + rates [+1,-1,+1] + wire
     [-1,-1,-1]) passes every one of them because BOTH sides flip together.

So the yaw sign needs its OWN external discriminator. This file encodes two, plus
a channel-recompute that surfaces the specific CR2-01 finding.

================================================================================
THE THREE INVARIANTS (all on the current-code POSTFIX data unless noted)
================================================================================
L3  -- yaw-rate vs attitude-heading-change (EXTERNAL):
        Project the TRUE-conjugated body-x axis into the world horizontal plane to
        get a heading psi; d/dt(psi) must correlate POSITIVELY with the convention's
        true body yaw-rate (_ODO_RATE_SIGN[2] * w_raw[2] = -w_raw[2]). This couples
        the RATE telemetry to the ATTITUDE telemetry through a world-frame heading
        derived from the externally-pinned TRUE conjugation -- it is NOT a
        self-comparison. A flipped yaw sign drives the correlation negative.
        Banked: corr(-w[2]) = +0.70, slope ~ +1.4 ; corr(+w[2]) = -0.70.

Q1  -- quat-FD body-z angular velocity vs the convention rate (EXTERNAL):
        Finite-difference the TRUE-conjugated attitude (rotvec of R_{k-1}^T R_{k+1})
        to recover body-frame angular velocity; its z-component must match
        -w_raw[2] with SMALL element-wise residual (not merely correlate). The
        flipped sign correlates equally in magnitude (-0.97) but its residual is
        ~40x larger -- a residual test, unlike a correlation test, cannot be fooled
        by an overall sign flip.
        Banked: corr(-w[2]) = +0.97, med|resid| = 0.010 rad/s
                corr(+w[2]) = -0.97, med|resid| = 0.41  rad/s.

L1/L2 -- channel recompute (CR2-01 positive/negative control):
        Re-derive rate_frd from the recorded action pipeline using the CURRENT
        fly_rl arithmetic imported LIVE (_ACT_FLU_TO_FRD, _RZ_PI_BODY). On POSTFIX
        (captured under current code 93023cf) every axis is BIT-EXACT. On REFIT
        (captured under bcc93f9) the YAW channel ANTI-matches (corr = -1.000,
        maxabsdiff ~ 3.11) -- which is the literal CR2-01 finding: the refit-dataset
        yaw channels encode bcc93f9, NOT current code, so they are INVALID for
        clearing current-code yaw claims. This block doubles as the negative
        control proving the recompute has discriminating power.

================================================================================
PASS/FAIL CONTRACT
================================================================================
The assertions encode the EXTERNAL invariant, not the current code's behaviour:
if the yaw sign were (re)introduced wrong, L3/Q1 would flip negative / blow up the
residual and the test would FAIL. A baked-in NEGATIVE CONTROL flips the sign in
each check and asserts the flipped variant would be REJECTED, proving the test can
catch the bug. Test A (force-vs-FD) is reported for context but is NOT asserted on
the yaw question -- it is structurally blind there, by design.

Runnable two ways:
  * .venv/Scripts/python.exe handoff/ultracode-substrate-audit-2026-06-13/regression_suite/test_confirmed_cr2_01.py
        -> prints PASS/FAIL + the key numbers, exits non-zero on FAIL.
  * pytest  -> the test_*() functions assert.

Uses recordings under handoff/shadowpc-{postfix,refit}-dataset-2026-06-12/extracted
via handoff/ultracode-substrate-audit-2026-06-13/scratch/_audit_io.py.
Does NOT import rl/contact_true_eval.py (out of scope).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# --- wire up the shared substrate (tests/_audit_io.py) + the production module under test ----
import pytest  # noqa: E402

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

import _audit_io as A  # noqa: E402

_ROOT = A.ROOT                                 # repo root (depth-independent, from _audit_io)
for _p in (str(_ROOT / "rl"), str(_ROOT / "src"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# refit/extracted is gitignored (only its .zip is tracked); postfix/extracted IS tracked.
# Skip (don't ERROR) on a checkout lacking either set -- on the canonical laptop both exist.
pytestmark = pytest.mark.skipif(
    not (A.dataset_present("postfix") and A.dataset_present("refit")),
    reason="ShadowPC audit recordings (postfix+refit extracted) not present",
)

from scipy.spatial.transform import Rotation  # noqa: E402

# Import the LIVE production constants -- the test must reflect whatever fly_rl
# currently claims, so a future edit that re-breaks the yaw sign is caught.
import fly_rl  # noqa: E402

_DT_LO, _DT_HI = 0.05, 0.09   # 30 Hz central-FD window (reject reset/aliased gaps)

# ---- acceptance thresholds (generous margins around the banked figures) ------
# L3: heading-rate vs convention yaw-rate must be clearly POSITIVE.
_L3_CORR_MIN = 0.45
# Q1: quat-FD body-z vs convention yaw-rate -- positive corr AND small residual,
# with the flipped sign's residual at least this many times worse.
_Q1_CORR_MIN = 0.85
_Q1_RESID_MAX = 0.10          # rad/s; banked ~0.010
_Q1_RESID_RATIO_MIN = 8.0     # banked ~40x
# L1/L2 recompute: bit-exact on current-code data.
_RECOMPUTE_MAXABS = 1e-4


# =============================================================================
# data loading
# =============================================================================
def _load_dataset(name: str) -> list[dict]:
    """All loadable runs in a dataset (header-only runs raise ValueError -> skip)."""
    runs = []
    for p in A.list_runs(name):
        try:
            runs.append(A.load_run(p))
        except ValueError:
            # 20260612_184029_..._std_f2 (and one std_ext) are header-only; tolerate.
            pass
    if not runs:
        raise RuntimeError(f"no loadable runs in dataset {name!r}")
    return runs


# =============================================================================
# L3 -- heading-rate (TRUE-conjugated body-x) vs convention yaw-rate
# =============================================================================
def _l3_pool(runs: list[dict], rate_sign_z: float) -> tuple[float, float, int]:
    """Pooled corr( d/dt heading , rate_sign_z * w_raw[2] ) + slope + n.

    heading psi = atan2(body_x_E, body_x_N) of the TRUE-conjugated attitude in
    world NED. EXTERNAL: ties rate telemetry to attitude telemetry via a
    world-frame heading from the externally-pinned conjugation.
    """
    P: list[float] = []
    W: list[float] = []
    for r in runs:
        t, q, w = r["t"], r["q_raw"], r["w_raw"]
        R = A.Rmats(q, sign=A.CAND_TRUE)          # world<-body, NED
        bx = R[:, :, 0]                           # body-x axis in world NED
        psi = np.arctan2(bx[:, 1], bx[:, 0])      # heading in N-E plane
        for k in range(1, len(t) - 1):
            dt = t[k + 1] - t[k - 1]
            if not (_DT_LO < dt < _DT_HI):
                continue
            dpsi = np.arctan2(np.sin(psi[k + 1] - psi[k - 1]),
                              np.cos(psi[k + 1] - psi[k - 1]))
            P.append(dpsi / dt)
            W.append(rate_sign_z * w[k, 2])
    P_arr = np.asarray(P)
    W_arr = np.asarray(W)
    if len(P_arr) < 5 or P_arr.std() < 1e-6 or W_arr.std() < 1e-6:
        return float("nan"), float("nan"), len(P_arr)
    corr = float(np.corrcoef(P_arr, W_arr)[0, 1])
    slope = float(np.polyfit(W_arr, P_arr, 1)[0])
    return corr, slope, len(P_arr)


# =============================================================================
# Q1 -- quat-FD body-z angular velocity vs convention yaw-rate
# =============================================================================
def _q1_pool(runs: list[dict], rate_sign_z: float) -> tuple[float, float, int]:
    """Pooled corr + median|residual| between the quat-FD body-z angular velocity
    of the TRUE-conjugated attitude and rate_sign_z * w_raw[2].

    Body-frame angular velocity = rotvec(R_{k-1}^T R_{k+1}) / dt.  EXTERNAL: the
    attitude *derivative* is an independent estimate of the rate channel; a sign
    flip is caught by the RESIDUAL (a flipped sign correlates -0.97 but its
    residual is ~40x worse).
    """
    fd: list[float] = []
    rt: list[float] = []
    for r in runs:
        t, q, w = r["t"], r["q_raw"], r["w_raw"]
        qs = q * A.CAND_TRUE                                  # signed wxyz
        Rq = Rotation.from_quat(qs[:, [1, 2, 3, 0]])          # wxyz -> xyzw
        for k in range(1, len(t) - 1):
            dt = t[k + 1] - t[k - 1]
            if not (_DT_LO < dt < _DT_HI):
                continue
            rv = (Rq[k - 1].inv() * Rq[k + 1]).as_rotvec() / dt   # body FRD ang.vel
            fd.append(rv[2])
            rt.append(rate_sign_z * w[k, 2])
    fd_arr = np.asarray(fd)
    rt_arr = np.asarray(rt)
    if len(fd_arr) < 5:
        return float("nan"), float("nan"), len(fd_arr)
    corr = float(np.corrcoef(fd_arr, rt_arr)[0, 1])
    resid = float(np.median(np.abs(fd_arr - rt_arr)))
    return corr, resid, len(fd_arr)


# =============================================================================
# L1/L2 -- recompute rate_frd from the recorded action pipeline (CURRENT fly_rl)
# =============================================================================
def _recompute_rate_frd(run: dict) -> np.ndarray:
    """rate_frd reconstructed from recorded act_rescaled via CURRENT fly_rl math:
        rate_flu = act[1:4]
        if virtual_flip: rate_flu = _RZ_PI_BODY @ rate_flu
        rate_frd = rate_flu * _ACT_FLU_TO_FRD
    Uses the LIVE-imported constants so a re-broken sign is caught here too.
    """
    act = run["act_rescaled"][:, 1:4].copy()
    if bool(run["meta"].get("virtual_flip", True)):
        # _RZ_PI_BODY is diagonal diag(-1,-1,1); apply per-row.
        act = act * fly_rl._RZ_PI_BODY.diagonal()[None, :]
    return act * fly_rl._ACT_FLU_TO_FRD[None, :]


def _recompute_yaw_stats(runs: list[dict]) -> tuple[float, float, int]:
    """Pooled corr + maxabsdiff between recomputed and recorded rate_frd[YAW]."""
    rec = np.concatenate([r["rate_frd"][:, 2] for r in runs])
    rcmp = np.concatenate([_recompute_rate_frd(r)[:, 2] for r in runs])
    corr = float(np.corrcoef(rcmp, rec)[0, 1])
    maxabs = float(np.max(np.abs(rcmp - rec)))
    return corr, maxabs, len(rec)


def _recompute_allaxes_maxabs(runs: list[dict]) -> dict:
    out = {}
    for i, ax in enumerate(("roll", "pitch", "yaw")):
        rec = np.concatenate([r["rate_frd"][:, i] for r in runs])
        rcmp = np.concatenate([_recompute_rate_frd(r)[:, i] for r in runs])
        out[ax] = (float(np.corrcoef(rcmp, rec)[0, 1]), float(np.max(np.abs(rcmp - rec))))
    return out


# =============================================================================
# Test-A context (force-vs-FD): reported, NOT asserted on the YAW question
# =============================================================================
def _testa_east_corr(runs: list[dict]) -> tuple[float, float]:
    """Pooled-ish: report the worst East corr across runs for TRUE vs AS-IS, as a
    sanity print confirming the substrate still has its discriminating power.
    Returns (true_east_min_corr, asis_east_max_corr)."""
    true_e, asis_e = [], []
    for r in runs:
        for sign, bucket in ((A.CAND_TRUE, true_e), (A.CAND_ASIS, asis_e)):
            res = A.test_a_force_vs_fd(r, sign)
            c = res["axes"]["E"]["corr"]
            if np.isfinite(c):
                bucket.append(c)
    return (min(true_e) if true_e else float("nan"),
            max(asis_e) if asis_e else float("nan"))


# =============================================================================
# pytest entry points
# =============================================================================
def test_l3_heading_rate_yaw_sign():
    """L3: heading-rate must correlate POSITIVELY with the convention yaw-rate,
    and the SIGN-FLIPPED variant must be rejected (negative control)."""
    runs = _load_dataset("postfix")
    corr_true, slope_true, n = _l3_pool(runs, fly_rl._ODO_RATE_SIGN[2])
    corr_flip, _, _ = _l3_pool(runs, -fly_rl._ODO_RATE_SIGN[2])
    assert np.isfinite(corr_true), "L3 produced no finite correlation"
    assert corr_true > _L3_CORR_MIN, (
        f"L3 heading-rate vs convention yaw-rate corr={corr_true:+.4f} "
        f"<= {_L3_CORR_MIN}: yaw sign convention looks WRONG (n={n}, slope={slope_true:+.3f})")
    # negative control: the flipped yaw sign would be rejected by the same gate.
    assert corr_flip < -_L3_CORR_MIN, (
        f"negative control failed: flipped yaw-rate corr={corr_flip:+.4f} not clearly "
        f"negative -- L3 would NOT catch a yaw sign flip")


def test_q1_quatfd_bodyz_yaw_sign():
    """Q1: quat-FD body-z must match the convention yaw-rate with a SMALL residual;
    the sign-flipped variant must have a much larger residual (negative control)."""
    runs = _load_dataset("postfix")
    corr_true, resid_true, n = _q1_pool(runs, fly_rl._ODO_RATE_SIGN[2])
    corr_flip, resid_flip, _ = _q1_pool(runs, -fly_rl._ODO_RATE_SIGN[2])
    assert np.isfinite(corr_true), "Q1 produced no finite correlation"
    assert corr_true > _Q1_CORR_MIN, (
        f"Q1 quat-FD body-z vs convention yaw-rate corr={corr_true:+.4f} "
        f"<= {_Q1_CORR_MIN} (n={n})")
    assert resid_true < _Q1_RESID_MAX, (
        f"Q1 med|resid|={resid_true:.4f} rad/s >= {_Q1_RESID_MAX}: convention yaw-rate "
        f"does NOT match the attitude derivative")
    # negative control: residual under the flipped sign is much worse.
    assert resid_flip > _Q1_RESID_RATIO_MIN * resid_true, (
        f"negative control failed: flipped-sign residual {resid_flip:.4f} is not "
        f">= {_Q1_RESID_RATIO_MIN}x the true residual {resid_true:.4f} -- a residual "
        f"test that cannot separate the sign flip is no test")


def test_recompute_postfix_bitexact_and_refit_is_invalid():
    """CR2-01: current-fly_rl recompute is BIT-EXACT on postfix (current-code data);
    the SAME recompute ANTI-matches the refit yaw channel (refit encodes bcc93f9,
    invalid for clearing current-code yaw claims). This is the finding + the
    negative control for the recompute path."""
    postfix = _load_dataset("postfix")
    refit = _load_dataset("refit")

    pf = _recompute_allaxes_maxabs(postfix)
    for ax, (corr, maxabs) in pf.items():
        assert maxabs < _RECOMPUTE_MAXABS, (
            f"postfix recompute axis {ax}: maxabsdiff={maxabs:.6f} not bit-exact "
            f"(corr={corr:+.6f}) -- current fly_rl arithmetic no longer reproduces the "
            f"current-code recordings; a channel convention changed")

    rf_corr, rf_maxabs, n = _recompute_yaw_stats(refit)
    # The CR2-01 finding: refit YAW is the OLD convention -> anti-correlates.
    assert rf_corr < -0.99, (
        f"refit yaw recompute corr={rf_corr:+.6f} (expected ~-1.0): refit was assumed "
        f"to encode bcc93f9's flipped yaw -- if it now matches current code, the "
        f"datasets or conventions shifted and the CR2-01 caveat must be re-derived")
    assert rf_maxabs > 1.0, (
        f"refit yaw recompute maxabsdiff={rf_maxabs:.4f} unexpectedly small "
        f"(n={n}); the bcc93f9-vs-93023cf yaw difference should be O(rad/s)")


# =============================================================================
# __main__ : human-readable report + PASS/FAIL
# =============================================================================
def _main() -> int:
    print("=" * 78)
    print("CR2-01 REGRESSION  --  external yaw-sign invariants (L3, Q1) + recompute control")
    print("=" * 78)

    postfix = _load_dataset("postfix")
    refit = _load_dataset("refit")
    print(f"loaded: postfix={len(postfix)} runs, refit={len(refit)} runs "
          f"(header-only runs skipped)")
    print(f"current fly_rl convention: _ODO_RATE_SIGN={fly_rl._ODO_RATE_SIGN.tolist()}  "
          f"_ACT_FLU_TO_FRD={fly_rl._ACT_FLU_TO_FRD.tolist()}")

    sign_z = fly_rl._ODO_RATE_SIGN[2]
    ok = True

    # ---- L3 ----
    c_true, slope, nL = _l3_pool(postfix, sign_z)
    c_flip, _, _ = _l3_pool(postfix, -sign_z)
    l3_pass = (c_true > _L3_CORR_MIN) and (c_flip < -_L3_CORR_MIN)
    ok &= l3_pass
    print("\n[L3] heading-rate (TRUE-conj body-x) vs convention yaw-rate  (POSTFIX, EXTERNAL)")
    print(f"     corr({sign_z:+.0f}*w_raw[2]) = {c_true:+.4f}   slope = {slope:+.3f}   n = {nL}")
    print(f"     neg-control corr({-sign_z:+.0f}*w_raw[2]) = {c_flip:+.4f}  "
          f"(must be < {-_L3_CORR_MIN:+.2f})")
    print(f"     -> {'PASS' if l3_pass else 'FAIL'}  (need corr > {_L3_CORR_MIN} & flip < {-_L3_CORR_MIN})")

    # ---- Q1 ----
    qc_true, qr_true, nQ = _q1_pool(postfix, sign_z)
    qc_flip, qr_flip, _ = _q1_pool(postfix, -sign_z)
    ratio = qr_flip / qr_true if qr_true > 0 else float("inf")
    q1_pass = (qc_true > _Q1_CORR_MIN and qr_true < _Q1_RESID_MAX
               and ratio > _Q1_RESID_RATIO_MIN)
    ok &= q1_pass
    print("\n[Q1] quat-FD body-z vs convention yaw-rate  (POSTFIX, EXTERNAL, residual test)")
    print(f"     corr({sign_z:+.0f}*w_raw[2]) = {qc_true:+.4f}   med|resid| = {qr_true:.4f} rad/s   n = {nQ}")
    print(f"     neg-control corr({-sign_z:+.0f}*w_raw[2]) = {qc_flip:+.4f}   "
          f"med|resid| = {qr_flip:.4f} rad/s  (ratio {ratio:.1f}x)")
    print(f"     -> {'PASS' if q1_pass else 'FAIL'}  (need corr > {_Q1_CORR_MIN}, "
          f"resid < {_Q1_RESID_MAX}, ratio > {_Q1_RESID_RATIO_MIN}x)")

    # ---- L1/L2 recompute control ----
    pf_axes = _recompute_allaxes_maxabs(postfix)
    pf_pass = all(m < _RECOMPUTE_MAXABS for _, m in pf_axes.values())
    rf_corr, rf_maxabs, nR = _recompute_yaw_stats(refit)
    rf_pass = (rf_corr < -0.99) and (rf_maxabs > 1.0)
    ok &= pf_pass and rf_pass
    print("\n[L1/L2] rate_frd recompute (CURRENT fly_rl arithmetic) vs recorded")
    print("     POSTFIX (current-code 93023cf) -- expect BIT-EXACT all axes:")
    for ax, (corr, maxabs) in pf_axes.items():
        print(f"        {ax:5s} corr={corr:+.6f}  maxabsdiff={maxabs:.6f}")
    print(f"     REFIT (bcc93f9)  YAW corr={rf_corr:+.6f}  maxabsdiff={rf_maxabs:.4f}  n={nR}")
    print(f"        ^ CR2-01 FINDING: refit yaw ANTI-matches current code "
          f"-> refit yaw channels are INVALID for clearing current-code yaw claims.")
    print(f"     -> {'PASS' if (pf_pass and rf_pass) else 'FAIL'}  "
          f"(postfix bit-exact & refit yaw anti-correlated)")

    # ---- Test A context (NOT a yaw assertion; structurally blind to yaw) ----
    te_min, ae_max = _testa_east_corr(postfix)
    print("\n[context] Test-A force-vs-FD East corr (substrate sanity; NOT a yaw test):")
    print(f"     TRUE East min corr = {te_min:+.3f}   AS-IS East max corr = {ae_max:+.3f}")
    print("     (Test A is STRUCTURALLY BLIND to yaw-about-vertical -- reported only to")
    print("      confirm the substrate's discriminating power is intact.)")

    print("\n" + "=" * 78)
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
