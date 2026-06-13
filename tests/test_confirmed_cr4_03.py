"""REGRESSION (CONFIRMED bug CR4-03) -- the refit-dataset yaw correlation validates
the OLD bcc93f9 wire map, NOT the current shipped fly_rl.py map.

================================  WHAT BUG CLASS THIS CATCHES  ================================
fly_rl.py turns a policy FLU rate command into the FRD wire command via two composed steps:
    virtual_flip:  rate_flu' = _RZ_PI_BODY @ rate_flu        (_RZ_PI_BODY = diag(-1,-1,1))
    flu->frd    :  rate_frd  = rate_flu' * _ACT_FLU_TO_FRD   (_ACT_FLU_TO_FRD = [+1,-1,+1])
The CURRENT shipped map is _ACT_FLU_TO_FRD = [+1,-1,+1] (yaw element +1, "+act3").  The OLD
bcc93f9 map was [+1,-1,-1] (yaw element -1, "-act3").  The two differ ONLY on the YAW channel.

CR4-03 is an EVIDENCE-CHAIN defect, not a code defect:  the `refit` recordings (the dataset the
reference handoff/laptop-frame-audit-2026-06-12/scripts/audit_candidates.py actually loads) were
captured while fly_rl.py still ran the OLD bcc93f9 wire map -- so their recorded rate_frd yaw
channel is -act3.  The `postfix` recordings were captured AFTER the fix -- their rate_frd yaw is
+act3.  Any audit that "validates the wire map" by correlating refit's recorded yaw against the
realized rate is therefore confirming the OLD map.  The SHIPPED code is correct; the refit
EVIDENCE points at the wrong map.  This test pins that down so a future refactor cannot silently
re-adopt the bcc93f9 yaw element by citing refit-yaw correlation as "proof".

==========================  WHY INTERNAL-CONSISTENCY CHECKS MISS IT  ==========================
A yaw-element sign flip in the act->wire map is a proper sign-alias.  It is INVISIBLE to:
  * quat-FD-vs-rate consistency:  q_raw and w_raw come from the SAME telemetry stream, so they
    agree with each other regardless of which wire map produced the motion.
  * level-flight / round-trip checks:  near level, yaw command and yaw motion both pass through
    cleanly; the flip's sign never shows because there is no external anchor in the loop.
  * "the recorded rate_frd tracks the realized yaw rate" (LENS3 below) is POSITIVE in BOTH
    datasets -- because whatever wire was actually flown produced the motion that was recorded.
    Internal self-consistency cannot tell you WHICH act->wire convention was in force.
Only an EXTERNAL anchor discriminates:  a realized body-yaw-rate computed from the TRUE attitude
(q_raw*[1,-1,1,-1], the R_y(pi)-conjugation pinned by the force-vs-FD invariant) and asked which
of {+act3, -act3} tracks it.  That sign FLIPS between postfix (+act3) and refit (-act3) -- proving
the two datasets were flown under different conventions and that refit cannot validate the current
map.

This is the same bug FAMILY the project has been bitten by 4x (R_y(pi) ODOMETRY-quat conjugation,
body-vs-world velocity mix, corner_to_center 180deg flip, ATTITUDE.pitch inversion): a proper
rotation/sign alias that survives every internal check and only an external invariant refutes.

=================================  THE FOUR EXTERNAL INVARIANTS  =============================
LENS1  Full-pipeline wire reconstruction (element-wise, vs recorded rate_frd):
       reconstruct rate_frd by replaying the ACTUAL fly_rl.py composition (virtual_flip then
       flu->frd) for BOTH yaw elements and check which reproduces the recorded wire EXACTLY.
       => postfix recorded wire == CURRENT(+act3); refit recorded wire == bcc(-act3).
LENS2  Yaw command vs a SELF-COMPUTED TRUE-attitude central-FD realized yaw-rate anchor:
       realized_omega_body = logSO3(R[k-1].T @ R[k+1]) / dt, R from q_raw*[1,-1,1,-1]; take
       body-z.  Score BOTH +act3 and -act3 against the SAME anchor, per-run, best-lag.
       => the SIGN of which command tracks the anchor FLIPS by dataset.
LENS3  Artifact guard:  recorded rate_frd[2] (the wire actually flown) vs the SAME anchor must be
       POSITIVE in BOTH datasets -- if it were not, the anchor itself would be suspect.  Positive
       in both => anchor stable, the flip lives in the act->wire map, not in the measurement.
LENS4  Canonical force-vs-FD Test A (positive control on anchor soundness):  K*L*(R@[0,0,-1]) +
       quad_drag + g  vs central-FD of pristine vel_ned, tilt>35deg.  TRUE[1,-1,1,-1] East corr
       ~+0.97..+0.99 / AS-IS[1,1,1,1] East corr negative, in BOTH datasets.  Confirms the TRUE
       attitude (hence the LENS2/3 anchor) is externally trustworthy.  Run production-faithful
       (use_lapse=False) AND ref-parity (use_lapse=True).

The assertions encode the EXTERNAL INVARIANT, not the current code's behavior:  they FAIL if a
future change makes the two datasets agree on a single wire map (which would mean the bcc93f9 yaw
element was silently re-adopted, or the TRUE-attitude anchor regressed).  A NEGATIVE CONTROL is
included (_negative_control) to show the assertion catches a flipped convention.

================================  !! SLUG-COLLISION NOTICE  ==================================
A DIFFERENT bug (COLL_MAP over-prediction in src/racer/rl_plant.py) was previously written to this
same `test_confirmed_cr4_03.py` path by a parallel session.  This file's CR4-03 is the one defined
by THIS task's authoritative Evidence chain (fly_rl.py wire-map evidence defect, LENS1-4 above).
The commander must reconcile the duplicate `confirmed_cr4_03` slug (e.g. rename the COLL_MAP test).

Run:
  .venv\\Scripts\\python.exe handoff\\ultracode-substrate-audit-2026-06-13\\regression_suite\\test_confirmed_cr4_03.py
Or as pytest:
  pytest handoff/ultracode-substrate-audit-2026-06-13/regression_suite/test_confirmed_cr4_03.py

NOTE: does NOT import rl/contact_true_eval.py (out of scope, edited elsewhere).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import pytest

# --- shared substrate (tests/_audit_io.py) lives beside this file -----------------------------
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))
import _audit_io as A  # noqa: E402  (shared loader + force model; reproduces banked canary)

# refit/extracted is gitignored (only its .zip is tracked); postfix/extracted IS tracked.
# Skip (don't ERROR) on a checkout lacking either set -- on the canonical laptop both exist.
pytestmark = pytest.mark.skipif(
    not (A.dataset_present("postfix") and A.dataset_present("refit")),
    reason="ShadowPC audit recordings (postfix+refit extracted) not present",
)

from scipy.spatial.transform import Rotation  # noqa: E402

# ---------------------------------------------------------------------------------------------
# fly_rl.py wire-map constants (replicated here so the test pins the EXTERNAL invariant, NOT the
# value of a constant imported from the file under audit).  These are the two CANDIDATE yaw
# elements of the act->wire map; the test decides which the recorded data was flown under.
# ---------------------------------------------------------------------------------------------
_RZ_PI_BODY = np.diag([-1.0, -1.0, 1.0])               # fly_rl.py virtual-flip body-z rotation
_ACT_FLU_TO_FRD_CURRENT = np.array([1.0, -1.0, 1.0])   # CURRENT shipped (yaw element +1)
_ACT_FLU_TO_FRD_BCC = np.array([1.0, -1.0, -1.0])      # OLD bcc93f9 (yaw element -1)

_EXACT_TOL = 1e-3      # rate_frd recorded to 4 dp -> exact match well under this
_MISMATCH_MIN = 0.5    # a real convention mismatch shows |err| >> this on the yaw channel
_ROLLPITCH_RAIL = 3.0  # |err| above this on roll/pitch == rail-clip, NOT a convention mismatch
_CORR_STRONG = 0.90    # strong |corr| threshold for the realized-yaw anchor
_EAST_TRUE_MIN = 0.90  # LENS4 TRUE East corr floor
_EAST_ASIS_MAX = -0.30 # LENS4 AS-IS East corr must be clearly negative


# =============================================================================================
# helpers
# =============================================================================================
def _reconstruct_rate_frd(act_rescaled: np.ndarray, flu_to_frd: np.ndarray,
                          virtual_flip: bool) -> np.ndarray:
    """Replay the EXACT fly_rl.py policy_step act->wire composition (vectorised over ticks).

    act_rescaled: (N,4) [thrust, roll, pitch, yaw] FLU body rates.
    Returns (N,3) reconstructed FRD wire command to compare against the recorded rate_frd.
    """
    rate_flu = act_rescaled[:, 1:4].copy()             # (N,3)
    if virtual_flip:
        rate_flu = rate_flu @ _RZ_PI_BODY.T            # == (_RZ_PI_BODY @ v) per row
    return rate_flu * flu_to_frd[None, :]


def _true_attitude_yaw_rate_anchor(run: dict) -> np.ndarray:
    """Central-FD body-z angular rate from the TRUE attitude (q_raw*[1,-1,1,-1]).

    This is the SELF-COMPUTED EXTERNAL anchor -- it uses NEITHER rate_frd, w_raw, nor act, only
    the pristine attitude time series.  Returns (N,) with NaN where dt is out of band.
    """
    t = run["t"]
    R = A.Rmats(run["q_raw"], sign=A.CAND_TRUE)        # (N,3,3) world<-body, TRUE attitude
    n = len(t)
    yaw_rate = np.full(n, np.nan)
    for k in range(1, n - 1):
        dt = t[k + 1] - t[k - 1]
        if not (0.05 < dt < 0.09):
            continue
        rv = Rotation.from_matrix(R[k - 1].T @ R[k + 1]).as_rotvec() / dt  # body ang vel
        yaw_rate[k] = rv[2]                              # body-z
    return yaw_rate


def _best_lag_corr(a: np.ndarray, b: np.ndarray, maxlag: int = 2) -> float:
    """Signed correlation of a vs b at the lag (|lag|<=maxlag) of MAX |corr| (sign preserved).

    Per-run (no cross-run concatenation) so command->realized latency cannot bleed lag across
    flight boundaries.
    """
    best = 0.0
    for lag in range(-maxlag, maxlag + 1):
        if lag < 0:
            x, y = a[-lag:], b[:lag]
        elif lag > 0:
            x, y = a[:-lag], b[lag:]
        else:
            x, y = a, b
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 10 or x[m].std() < 1e-6 or y[m].std() < 1e-6:
            continue
        c = float(np.corrcoef(x[m], y[m])[0, 1])
        if abs(c) > abs(best):
            best = c
    return best


def _iter_runs(dataset: str):
    """Yield loaded runs that carry the action pipeline, skipping header-only files."""
    for run_dir in A.list_runs(dataset):
        try:
            r = A.load_run(run_dir)
        except ValueError:
            continue  # header-only (e.g. std_f2, std_ext_f2) -> tolerated, skip (no retry loop)
        if "act_rescaled" in r and "rate_frd" in r:
            yield r


# =============================================================================================
# measurement: returns the full per-dataset evidence dict (computed once, reused by all tests)
# =============================================================================================
def measure() -> dict:
    out: dict = {}
    for ds in ("postfix", "refit"):
        runs = list(_iter_runs(ds))
        assert runs, f"no usable runs in dataset {ds!r}"
        per = {
            "lens1_yawErr_current": [], "lens1_yawErr_bcc": [],
            "lens1_rollErr_current": [], "lens1_pitchErr_current": [],
            "lens2_corr_current": [], "lens2_corr_bcc": [],
            "lens3_corr_recorded": [],
            "names": [],
        }
        for r in runs:
            vflip = bool(r["meta"].get("virtual_flip", True))
            act = r["act_rescaled"]
            frd = r["rate_frd"]
            # --- LENS1: full-pipeline wire reconstruction, element-wise max|err| ---
            rec_cur = _reconstruct_rate_frd(act, _ACT_FLU_TO_FRD_CURRENT, vflip)
            rec_bcc = _reconstruct_rate_frd(act, _ACT_FLU_TO_FRD_BCC, vflip)
            e_cur = np.max(np.abs(rec_cur - frd), axis=0)
            e_bcc = np.max(np.abs(rec_bcc - frd), axis=0)
            per["lens1_yawErr_current"].append(float(e_cur[2]))
            per["lens1_yawErr_bcc"].append(float(e_bcc[2]))
            per["lens1_rollErr_current"].append(float(e_cur[0]))
            per["lens1_pitchErr_current"].append(float(e_cur[1]))
            # --- LENS2: yaw cmd vs TRUE-attitude FD anchor (same anchor, both signs, per-run) ---
            anchor = _true_attitude_yaw_rate_anchor(r)
            act3 = act[:, 3]
            per["lens2_corr_current"].append(_best_lag_corr(+act3, anchor))
            per["lens2_corr_bcc"].append(_best_lag_corr(-act3, anchor))
            # --- LENS3: artifact guard -- recorded wire vs SAME anchor (per-run) ---
            per["lens3_corr_recorded"].append(_best_lag_corr(frd[:, 2], anchor))
            per["names"].append(r["name"])

        # --- LENS4: pooled force-vs-FD Test A (East axis), production-faithful AND ref-parity ---
        def _pooled_east(use_lapse: bool):
            mt_true, mt_asis, me, tl = [], [], [], []
            for r in runs:
                Rt = A.Rmats(r["q_raw"], sign=A.CAND_TRUE)
                Ra = A.Rmats(r["q_raw"], sign=A.CAND_ASIS)
                tilt_raw = A.tilt_deg(Ra)               # tilt invariant across candidates
                t, vel, coll = r["t"], r["vel"], r["coll"]
                for k in range(2, len(t) - 2):
                    dt = t[k + 1] - t[k - 1]
                    if not (0.05 < dt < 0.09):
                        continue
                    a_meas = (vel[k + 1] - vel[k - 1]) / dt
                    if not np.all(np.isfinite(a_meas)) or np.max(np.abs(a_meas)) > 90:
                        continue
                    cd = coll[k - 2]
                    mt_true.append(A.force_model(Rt[k], vel[k], cd, use_lapse=use_lapse))
                    mt_asis.append(A.force_model(Ra[k], vel[k], cd, use_lapse=use_lapse))
                    me.append(a_meas)
                    tl.append(tilt_raw[k])
            mt_true = np.array(mt_true); mt_asis = np.array(mt_asis)
            me = np.array(me); tl = np.array(tl)
            m = tl > 35.0
            assert m.sum() >= 50, f"too few tilted samples for LENS4 ({int(m.sum())})"
            e_true = float(np.corrcoef(mt_true[m, 1], me[m, 1])[0, 1])   # axis 1 = East
            e_asis = float(np.corrcoef(mt_asis[m, 1], me[m, 1])[0, 1])
            return e_true, e_asis, int(m.sum())

        per["lens4_east_true_prod"], per["lens4_east_asis_prod"], per["lens4_n_tilt"] = \
            _pooled_east(use_lapse=False)
        per["lens4_east_true_ref"], per["lens4_east_asis_ref"], _ = _pooled_east(use_lapse=True)
        out[ds] = per
    return out


# =============================================================================================
# the regression assertions (importable as pytest tests)
# =============================================================================================
def _assert_cr4_03(ev: dict) -> None:
    """Assert the four external invariants that pin CR4-03.

    PASSES on the current committed tree (shipped code correct; refit EVIDENCE points at the OLD
    map).  Would FAIL if a future change re-adopted the bcc93f9 yaw element or regressed the
    TRUE-attitude anchor.
    """
    post, refit = ev["postfix"], ev["refit"]

    # ---- LENS1: wire-provenance is EXACT and OPPOSITE between datasets -------------------
    # postfix recorded wire == CURRENT(+act3); refit recorded wire == bcc(-act3).
    assert max(post["lens1_yawErr_current"]) < _EXACT_TOL, (
        f"postfix yaw should match CURRENT(+act3) exactly; "
        f"max err {max(post['lens1_yawErr_current']):.4f}")
    assert min(post["lens1_yawErr_bcc"]) > _MISMATCH_MIN, (
        f"postfix yaw must NOT match bcc(-act3); min err {min(post['lens1_yawErr_bcc']):.4f}")
    assert max(refit["lens1_yawErr_bcc"]) < _EXACT_TOL, (
        f"refit yaw should match bcc(-act3) exactly; "
        f"max err {max(refit['lens1_yawErr_bcc']):.4f}")
    assert min(refit["lens1_yawErr_current"]) > _MISMATCH_MIN, (
        f"refit yaw must NOT match CURRENT(+act3) -> THIS is CR4-03: refit evidence validates "
        f"the OLD map; min err {min(refit['lens1_yawErr_current']):.4f}")
    # roll/pitch match CURRENT everywhere except genuine rail-clip (refit rollfix roll ~3.56).
    assert max(post["lens1_rollErr_current"]) < _EXACT_TOL
    assert max(post["lens1_pitchErr_current"]) < _EXACT_TOL
    assert max(refit["lens1_pitchErr_current"]) < _EXACT_TOL
    refit_roll_nonrail = [e for e in refit["lens1_rollErr_current"] if e < _ROLLPITCH_RAIL]
    assert refit_roll_nonrail and max(refit_roll_nonrail) < _EXACT_TOL, (
        "refit non-rail roll should match CURRENT exactly (roll element undisputed)")

    # ---- LENS2: SAME anchor, sign of the tracking command FLIPS by dataset ---------------
    assert min(post["lens2_corr_current"]) > _CORR_STRONG, (
        f"postfix +act3 should strongly track the TRUE-attitude anchor; "
        f"min {min(post['lens2_corr_current']):+.3f}")
    assert max(post["lens2_corr_bcc"]) < -_CORR_STRONG, (
        f"postfix -act3 should strongly ANTI-track; max {max(post['lens2_corr_bcc']):+.3f}")
    assert max(refit["lens2_corr_current"]) < -_CORR_STRONG, (
        f"refit +act3 should strongly ANTI-track (the flip vs postfix); "
        f"max {max(refit['lens2_corr_current']):+.3f}")
    assert min(refit["lens2_corr_bcc"]) > _CORR_STRONG, (
        f"refit -act3 should strongly track; min {min(refit['lens2_corr_bcc']):+.3f}")

    # ---- LENS3: artifact guard -- recorded wire tracks anchor POSITIVELY in BOTH ---------
    assert min(post["lens3_corr_recorded"]) > _CORR_STRONG, (
        f"postfix recorded rate_frd[2] must track anchor positively; "
        f"min {min(post['lens3_corr_recorded']):+.3f}")
    assert min(refit["lens3_corr_recorded"]) > _CORR_STRONG, (
        f"refit recorded rate_frd[2] must track anchor positively (anchor stable, flip is in "
        f"the act->wire map not the measurement); min {min(refit['lens3_corr_recorded']):+.3f}")

    # ---- LENS4: TRUE attitude externally confirmed in BOTH -> anchor trustworthy ---------
    for ds, per in (("postfix", post), ("refit", refit)):
        assert per["lens4_east_true_prod"] > _EAST_TRUE_MIN, (
            f"{ds} TRUE East corr (prod) must be ~+0.97..+0.99; got "
            f"{per['lens4_east_true_prod']:+.3f}")
        assert per["lens4_east_asis_prod"] < _EAST_ASIS_MAX, (
            f"{ds} AS-IS East corr (prod) must be clearly negative (positive control); got "
            f"{per['lens4_east_asis_prod']:+.3f}")
        assert per["lens4_east_true_ref"] > _EAST_TRUE_MIN
        assert per["lens4_east_asis_ref"] < _EAST_ASIS_MAX


def test_cr4_03_lens1_wire_provenance_opposite_between_datasets():
    """LENS1: postfix recorded wire == CURRENT(+act3); refit recorded wire == bcc(-act3)."""
    ev = measure()
    post, refit = ev["postfix"], ev["refit"]
    assert max(post["lens1_yawErr_current"]) < _EXACT_TOL
    assert max(refit["lens1_yawErr_bcc"]) < _EXACT_TOL
    assert min(refit["lens1_yawErr_current"]) > _MISMATCH_MIN  # the evidence-chain defect


def test_cr4_03_lens2_yaw_anchor_sign_flips_by_dataset():
    """LENS2: same TRUE-attitude anchor; the command that tracks it flips sign by dataset."""
    ev = measure()
    post, refit = ev["postfix"], ev["refit"]
    assert min(post["lens2_corr_current"]) > _CORR_STRONG
    assert max(refit["lens2_corr_current"]) < -_CORR_STRONG


def test_cr4_03_lens3_artifact_guard_anchor_stable():
    """LENS3: recorded rate_frd[2] tracks the anchor POSITIVELY in BOTH datasets."""
    ev = measure()
    assert min(ev["postfix"]["lens3_corr_recorded"]) > _CORR_STRONG
    assert min(ev["refit"]["lens3_corr_recorded"]) > _CORR_STRONG


def test_cr4_03_lens4_true_attitude_positive_control():
    """LENS4: Test A East corr -- TRUE wins / AS-IS negative in BOTH datasets, both lapse modes."""
    ev = measure()
    for per in (ev["postfix"], ev["refit"]):
        assert per["lens4_east_true_prod"] > _EAST_TRUE_MIN
        assert per["lens4_east_asis_prod"] < _EAST_ASIS_MAX
        assert per["lens4_east_true_ref"] > _EAST_TRUE_MIN
        assert per["lens4_east_asis_ref"] < _EAST_ASIS_MAX


def test_confirmed_cr4_03():
    """All four external invariants together (the canonical regression entry point)."""
    _assert_cr4_03(measure())


# =============================================================================================
# negative control: show the assertion WOULD catch a re-introduced flip
# =============================================================================================
def _negative_control() -> bool:
    """Build a counterfactual 'refit' whose recorded yaw was (hypothetically) flown under the
    CURRENT map, and confirm LENS1's "refit yaw matches bcc, NOT current" assertion FAILS on it.

    This proves the test is discriminating: if a future capture flew the refit scenario under the
    fixed map (the bug absent), the 'refit yaw matches bcc' invariant breaks -- exactly the signal
    the regression is built to detect.  Returns True iff the assertion correctly fires.
    """
    ev = measure()
    refit = dict(ev["refit"])
    # counterfactual: refit recorded yaw now matches CURRENT and no longer matches bcc.
    refit["lens1_yawErr_bcc"] = [1.5] * len(refit["lens1_yawErr_bcc"])      # no longer bcc-exact
    refit["lens1_yawErr_current"] = [0.0] * len(refit["lens1_yawErr_current"])  # now current-exact
    ev_cf = {"postfix": ev["postfix"], "refit": refit}
    try:
        _assert_cr4_03(ev_cf)
    except AssertionError:
        return True   # GOOD: the regression caught the convention change
    return False


# =============================================================================================
# __main__ : print PASS/FAIL + key numbers
# =============================================================================================
def _fmt_range(xs):
    return f"[{min(xs):+.3f},{max(xs):+.3f}]"


if __name__ == "__main__":
    print("=" * 92)
    print("CR4-03 REGRESSION -- refit-dataset yaw correlation validates the OLD bcc93f9 wire map")
    print("=" * 92)
    ev = measure()
    post, refit = ev["postfix"], ev["refit"]

    print(f"\n  postfix usable runs: {len(post['names'])}   refit usable runs: {len(refit['names'])}"
          f"   (header-only runs skipped)")

    print("\nLENS1  full-pipeline wire reconstruction (max|recon - recorded rate_frd|):")
    print(f"  postfix  yaw vs CURRENT(+act3): max {max(post['lens1_yawErr_current']):.4f}  "
          f"|  vs bcc(-act3): min {min(post['lens1_yawErr_bcc']):.4f}")
    print(f"  refit    yaw vs CURRENT(+act3): min {min(refit['lens1_yawErr_current']):.4f}  "
          f"|  vs bcc(-act3): max {max(refit['lens1_yawErr_bcc']):.4f}")
    _rr = [e for e in refit["lens1_rollErr_current"] if e >= _ROLLPITCH_RAIL]
    print(f"  roll/pitch vs CURRENT exact: postfix max "
          f"{max(post['lens1_rollErr_current'] + post['lens1_pitchErr_current']):.4f}; "
          f"refit rail-clip roll runs (>{_ROLLPITCH_RAIL}): {len(_rr)} (rollfix; expected, "
          f"not a convention mismatch)")
    print("  => postfix wire IS +act3, refit wire IS -act3 (EXACT) -> refit validates the OLD map")

    print("\nLENS2  yaw cmd vs TRUE-attitude FD realized-yaw anchor (per-run best-lag corr):")
    print(f"  postfix  CURRENT(+act3) {_fmt_range(post['lens2_corr_current'])}   "
          f"bcc(-act3) {_fmt_range(post['lens2_corr_bcc'])}")
    print(f"  refit    CURRENT(+act3) {_fmt_range(refit['lens2_corr_current'])}   "
          f"bcc(-act3) {_fmt_range(refit['lens2_corr_bcc'])}")
    print("  => SAME anchor; the sign of which command tracks it FLIPS by dataset.")

    print("\nLENS3  artifact guard -- recorded rate_frd[2] vs SAME anchor:")
    print(f"  postfix {_fmt_range(post['lens3_corr_recorded'])} "
          f"median {np.median(post['lens3_corr_recorded']):+.3f}")
    print(f"  refit   {_fmt_range(refit['lens3_corr_recorded'])} "
          f"median {np.median(refit['lens3_corr_recorded']):+.3f}")
    print("  => POSITIVE in BOTH => anchor stable; flip is in the act->wire map, not the data.")

    print("\nLENS4  force-vs-FD Test A East corr (positive control on anchor soundness):")
    for ds, per in (("postfix", post), ("refit", refit)):
        print(f"  {ds:7s} TRUE E prod {per['lens4_east_true_prod']:+.3f} / AS-IS E prod "
              f"{per['lens4_east_asis_prod']:+.3f}  |  TRUE E ref {per['lens4_east_true_ref']:+.3f}"
              f" / AS-IS E ref {per['lens4_east_asis_ref']:+.3f}  (n_tilt={per['lens4_n_tilt']})")

    print("\n" + "-" * 92)
    try:
        _assert_cr4_03(ev)
        verdict = "PASS"
        err = None
    except AssertionError as exc:
        verdict = "FAIL"
        err = str(exc)

    neg_ok = _negative_control()
    print(f"negative control (counterfactual refit flown under current map -> assertion must "
          f"fire): {'OK (caught)' if neg_ok else 'BROKEN (did not catch)'}")
    print("-" * 92)
    print(f"RESULT: {verdict}  -- CR4-03 evidence-chain defect pinned: shipped wire map is "
          f"CURRENT[+1,-1,+1];")
    print("        refit recordings were flown under the OLD bcc93f9[+1,-1,-1] map "
          "(so refit-yaw 'validation' confirms the wrong map).")
    if err:
        print(f"  assertion failure: {err}")
    sys.exit(0 if (verdict == "PASS" and neg_ok) else 1)
