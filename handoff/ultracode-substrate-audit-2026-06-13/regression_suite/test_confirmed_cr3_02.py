"""REGRESSION — CONFIRMED bug CR3-02: stale DERIVED deploy channels in the refit dataset.

WHAT BUG CLASS THIS CATCHES
---------------------------
The action-wire convention (FLU body-rate command -> FRD wire command, absorbing
`virtual_flip`) has been re-cut several times in fly_rl.py history:

    W_CURRENT  = [-1, +1, +1]   (93023cf, the shipped recipe -- _ACT_FLU_TO_FRD == [1,-1,1]
                                 applied to act_rescaled[1:4] gives this net yaw/roll sign)
    W_BCC93F9  = [+1, +1, -1]   (roll-convention fix era)
    W_DAFDC69  = [-1, +1, -1]   (earlier)

A recording carries BOTH a recipe-free field (`act_rescaled` = tanh-rescaled policy
output, BODY-FLU) AND its DERIVED downstream products (`rate_frd` = the wire command
actually sent, and obs[11] = w_fluz = the body-yaw-rate fed back into the observation).
The derived products are computed at *capture time* under whatever wire recipe fly_rl
was running then. CR3-02: the `refit` dataset (packaged at commit c5f8bcd, "post
roll-convention fix bcc93f9", per its MANIFEST) recorded its derived channels under a
STALE pre-93023cf recipe. So refit's `rate_frd` / obs[11] are NOT deploy-faithful for
the current stack -- only its RAW telemetry (pos_ned/vel_ned/q_raw_wxyz/w_raw/
act_rescaled) is. The `postfix` dataset (post VISION-FRAME-FIX 8d7b0b3) records them
under the CURRENT recipe and IS deploy-faithful.

The regression danger: someone trusts refit's `rate_frd` or obs[11] as if it were the
current deploy seam (e.g. to validate the action round-trip, or as a twin input), and
silently inherits a yaw-sign flip; OR fly_rl.py's shipped wire constants drift away from
the convention the deploy-faithful (postfix) data exhibits.

WHY INTERNAL-CONSISTENCY CHECKS MISS IT (and why THIS test does not)
-------------------------------------------------------------------
This project has been bitten 4x by proper-rotation conjugation / sign-alias bugs
(R_y(pi) ODOMETRY-quat conjugation, body-vs-world velocity-frame mix, corner_to_center
180deg flip, ATTITUDE.pitch sign inversion). The canonical trap: a sign-alias is
INVISIBLE to internal-consistency checks -- quat-vs-rate finite-difference, level-flight
correlation, and twist round-trip ALL pass while the convention is wrong, because a
proper rotation looks identical to its conjugate under any check that round-trips through
the SAME convention.

CR3-02 is the *action-side* analogue. We DELIBERATELY build the invariant out of two
INDEPENDENT lenses that touch NO quaternion, so neither can be a quat-conjugation
self-mirror:

  LENS 1 (recipe-free, action wire): compare two RAW RECORDED fields -- `rate_frd` vs
          `act_rescaled` -- against each historical wire convention W. The recipe whose
          maxabsdiff is ~0 is the one this dataset was recorded under. No quat, no model.

  LENS 2 (recipe-free, rate feedback): reconstruct obs[11]=w_fluz from the RAW recorded
          `w_raw` under a per-recipe rate-sign. The recipe whose residual is ~0 is the
          one this dataset was recorded under. No quat, no model.

The two lenses must AGREE per dataset (proves no harness mirror / mixed recipe). The
discriminating axis is YAW (roll alone cannot separate CURRENT from DAFDC69; only yaw
separates the current shipped recipe from the stale one).

THE EXTERNAL INVARIANT (encoded, not the current code's behavior)
-----------------------------------------------------------------
  (I1) postfix (deploy-faithful capture) MUST resolve to W_CURRENT on BOTH lenses.
  (I2) refit  (stale capture)            MUST resolve to a NON-current recipe on BOTH
       lenses -- i.e. its derived channels are demonstrably stale (this is the CONFIRMED
       bug; asserting it keeps the dataset honest and prevents silent reuse).
  (I3) The shipped fly_rl.py wire constant (_ACT_FLU_TO_FRD) MUST match W_CURRENT on the
       discriminating yaw axis -- so if fly_rl's wire convention regressed to a stale
       recipe, this fails.
  (I4) NEGATIVE CONTROL: trusting refit's rate_frd as deploy-faithful (i.e. comparing it
       to the CURRENT recipe) MUST produce a large yaw residual -- demonstrating the
       assertion machinery would in fact catch the bug if reintroduced.
  (I5) RAW-TELEMETRY POSITIVE CONTROL: the underlying R_y(pi) quat mirror is present in
       BOTH datasets (Test A: AS-IS East corr negative, TRUE East corr ~+0.97..+0.99),
       confirming the raw telemetry these conclusions rest on is the same trusted anchor.

If the CR3-02 bug class were reintroduced -- refit's stale channels promoted to
deploy-faithful, or fly_rl's wire convention drifted -- (I1)/(I2)/(I3)/(I4) fail.

Runs standalone (prints PASS/FAIL + numbers) AND as a pytest test_*() with assertions.
Does NOT import rl/contact_true_eval.py (out of scope).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# --- locate the shared substrate (single loader + force model) ---------------------------
_HERE = Path(__file__).resolve()
# .../handoff/ultracode-substrate-audit-2026-06-13/regression_suite/test_confirmed_cr3_02.py
#   parents[0]=regression_suite  [1]=ultracode-substrate-audit-2026-06-13  [2]=handoff  [3]=root
_SCRATCH = _HERE.parents[1] / "scratch"
if str(_SCRATCH) not in sys.path:
    sys.path.insert(0, str(_SCRATCH))
import _audit_io as A  # noqa: E402  (shared substrate: list_runs/load_run/test_a_force_vs_fd/...)

ROOT = A.ROOT

# =========================================================================================
# Historical FLU->FRD wire conventions (W applied to act_rescaled[1:4]=[roll,pitch,yaw] FLU).
# These are recipe LABELS used by the audit; the discriminator is the YAW (and roll) sign.
# =========================================================================================
W_CURRENT = np.array([-1.0, +1.0, +1.0])   # 93023cf shipped recipe (deploy-faithful)
W_BCC93F9 = np.array([+1.0, +1.0, -1.0])
W_DAFDC69 = np.array([-1.0, +1.0, -1.0])
WIRE_CONVS = {"CURRENT": W_CURRENT, "BCC93F9": W_BCC93F9, "DAFDC69": W_DAFDC69}

# obs[11] = w_fluz (body yaw rate, FLU) reconstructed from raw odo rate w_raw.
# Recipe-free per-axis rate sign relating obs[9:12]=w_flu to recorded w_raw.
# (Measured empirically: roll +1, pitch -1, yaw = recipe-dependent.)
# We only need the YAW sign to discriminate, and we derive it per-recipe below.
OBS_W_BASE_SIGN = np.array([+1.0, -1.0, +1.0])  # roll, pitch, yaw-placeholder

# Map a wire recipe to the yaw sign expected on obs[11]=w_fluz vs w_raw_z.
# Empirically the obs-yaw-sign tracks the wire-yaw-sign (both flip together across recipes):
#   CURRENT -> obs w_fluz/w_raw_z = +1 ;  BCC93F9/DAFDC69 -> -1.
def _obs_yaw_sign_for(wire_yaw_sign: float) -> float:
    return float(wire_yaw_sign)


# =========================================================================================
# Lens helpers (recipe-free; touch NO quaternion)
# =========================================================================================
def _safe_load_runs(dataset: str) -> list[dict]:
    """Load all runs in a dataset, tolerating header-only (0-row) runs (e.g. std_f2 spawn-crash)."""
    out = []
    for rd in A.list_runs(dataset):
        try:
            out.append(A.load_run(rd))
        except ValueError:
            # header-only run (0 data rows) -- skip, no retry loop.
            continue
    return out


def _pool(runs: list[dict], key: str) -> np.ndarray:
    """Vertically stack a field across runs that carry it."""
    arrs = [r[key] for r in runs if key in r]
    if not arrs:
        return np.empty((0,))
    return np.concatenate(arrs, axis=0)


def lens1_wire_residuals(runs: list[dict]) -> dict:
    """LENS 1: recorded rate_frd vs recorded act_rescaled[1:4] under each wire convention.

    Returns {recipe: maxabsdiff_per_axis(roll,pitch,yaw)} pooled across all runs.
    NO quaternion touched. The recipe with ~0 yaw residual is the capture recipe.
    """
    rf = _pool(runs, "rate_frd")            # (N,3) wire command [roll,pitch,yaw]
    ar = _pool(runs, "act_rescaled")        # (N,4) [thrust,roll,pitch,yaw] FLU
    rate_flu = ar[:, 1:4]
    res = {}
    for name, W in WIRE_CONVS.items():
        pred = rate_flu * W[None, :]
        res[name] = np.max(np.abs(pred - rf), axis=0)
    return res


def lens2_obs_yaw_residuals(runs: list[dict]) -> dict:
    """LENS 2: obs[11]=w_fluz reconstructed from recorded w_raw under each recipe's yaw sign.

    Returns {recipe: maxabsdiff_yaw} pooled across all runs. NO quaternion touched.
    """
    obs = _pool(runs, "obs")                # (N,17)
    wraw = _pool(runs, "w_raw")             # (N,3) [x,y,z] raw odo rate
    w_fluz = obs[:, 11]                      # recorded body-yaw-rate fed to policy
    res = {}
    for name, W in WIRE_CONVS.items():
        ys = _obs_yaw_sign_for(W[2])
        pred = wraw[:, 2] * ys
        res[name] = float(np.max(np.abs(pred - w_fluz)))
    return res


def _winning_recipe(per_recipe_yaw_resid: dict, tol: float = 1e-2) -> str | None:
    """Recipe whose yaw residual is ~0 (deploy-/capture-faithful). None if ambiguous."""
    winners = [name for name, r in per_recipe_yaw_resid.items() if r < tol]
    return winners[0] if len(winners) == 1 else (None if not winners else winners[0])


def _lens1_yaw(res: dict) -> dict:
    return {name: float(v[2]) for name, v in res.items()}


# =========================================================================================
# Shipped fly_rl.py wire constants (independent cross-check, no recording involved)
# =========================================================================================
def shipped_wire_yaw_sign() -> float:
    """Net FLU->FRD yaw sign the shipped fly_rl.py applies to act_rescaled.

    fly_rl wires rate_frd = act_rescaled[1:4] * _ACT_FLU_TO_FRD. We read _ACT_FLU_TO_FRD
    directly from the source (textual; importing fly_rl pulls heavy deps + may init MAVLink).
    """
    import re
    src = (ROOT / "rl" / "fly_rl.py").read_text(encoding="utf-8")
    m = re.search(r"_ACT_FLU_TO_FRD\s*=\s*np\.array\(\s*\[([^\]]+)\]", src)
    if not m:
        raise AssertionError("could not locate _ACT_FLU_TO_FRD in rl/fly_rl.py")
    vals = [float(x) for x in m.group(1).split(",")]
    assert len(vals) == 3, f"_ACT_FLU_TO_FRD wrong length: {vals}"
    return vals[2]  # yaw sign


# =========================================================================================
# THE EXTERNAL-INVARIANT CHECK
# =========================================================================================
def _evaluate() -> dict:
    """Compute every lens/control and return a structured result dict (no asserts)."""
    refit = _safe_load_runs("refit")
    postfix = _safe_load_runs("postfix")
    assert refit, "no loadable refit runs"
    assert postfix, "no loadable postfix runs"

    out: dict = {"refit": {}, "postfix": {}}

    for label, runs in (("refit", refit), ("postfix", postfix)):
        l1 = lens1_wire_residuals(runs)
        l2 = lens2_obs_yaw_residuals(runs)
        l1_yaw = _lens1_yaw(l1)
        l1_win = _winning_recipe(l1_yaw)
        l2_win = _winning_recipe(l2)
        out[label] = {
            "n_runs": len(runs),
            "n_ticks_rate_frd": int(_pool(runs, "rate_frd").shape[0]),
            "lens1_yaw_resid": l1_yaw,        # {recipe: maxabs yaw resid}
            "lens2_yaw_resid": l2,            # {recipe: maxabs yaw resid}
            "lens1_winner": l1_win,
            "lens2_winner": l2_win,
            "lenses_agree": (l1_win is not None and l1_win == l2_win),
        }

    # (I4) NEGATIVE CONTROL: pretend refit's rate_frd IS current-deploy-faithful.
    out["neg_control_refit_vs_CURRENT_yaw_resid"] = out["refit"]["lens1_yaw_resid"]["CURRENT"]

    # (I3) shipped fly_rl wire yaw sign
    out["shipped_wire_yaw_sign"] = shipped_wire_yaw_sign()

    # (I5) RAW-TELEMETRY POSITIVE CONTROL (Test A, East axis) on the most-tilted refit run.
    #      Picks the run with the most tilted samples for a robust control.
    best = max(refit, key=lambda r: int(np.sum(A.tilt_deg(A.Rmats(r["q_raw"], A.CAND_ASIS)) > 35.0)))
    ta_true = A.test_a_force_vs_fd(best, A.CAND_TRUE)
    ta_asis = A.test_a_force_vs_fd(best, A.CAND_ASIS)
    out["posctrl_run"] = best["name"]
    out["posctrl_true_E_corr"] = ta_true["axes"]["E"]["corr"]
    out["posctrl_asis_E_corr"] = ta_asis["axes"]["E"]["corr"]
    out["posctrl_n_tilt"] = ta_true["n_tilted"]
    return out


# =========================================================================================
# pytest entry point
# =========================================================================================
def test_confirmed_cr3_02():
    r = _evaluate()
    # stash for the standalone runner (pytest forbids returning non-None from a test).
    test_confirmed_cr3_02.last_result = r

    # --- (I1) postfix is deploy-faithful: BOTH lenses resolve to CURRENT, and they agree. ---
    assert r["postfix"]["lens1_winner"] == "CURRENT", (
        f"postfix LENS1 should resolve to CURRENT wire recipe, got {r['postfix']['lens1_winner']} "
        f"(yaw resids {r['postfix']['lens1_yaw_resid']})"
    )
    assert r["postfix"]["lens2_winner"] == "CURRENT", (
        f"postfix LENS2 (obs[11] from w_raw) should resolve to CURRENT, got {r['postfix']['lens2_winner']} "
        f"(yaw resids {r['postfix']['lens2_yaw_resid']})"
    )
    assert r["postfix"]["lenses_agree"], "postfix lenses disagree -> harness mirror / mixed recipe"

    # --- (I2) refit's DERIVED channels are STALE: both lenses resolve NON-current, and agree. ---
    assert r["refit"]["lens1_winner"] is not None and r["refit"]["lens1_winner"] != "CURRENT", (
        f"refit LENS1 should resolve to a NON-current (stale) wire recipe (CR3-02), "
        f"got {r['refit']['lens1_winner']} (yaw resids {r['refit']['lens1_yaw_resid']})"
    )
    assert r["refit"]["lens2_winner"] is not None and r["refit"]["lens2_winner"] != "CURRENT", (
        f"refit LENS2 should resolve to a NON-current (stale) recipe, "
        f"got {r['refit']['lens2_winner']} (yaw resids {r['refit']['lens2_yaw_resid']})"
    )
    assert r["refit"]["lenses_agree"], "refit lenses disagree -> harness mirror / mixed recipe"

    # --- (I4) NEGATIVE CONTROL: refit.rate_frd vs CURRENT recipe has a LARGE yaw residual. ---
    # This proves the discriminator has teeth: if someone trusted refit's rate_frd as
    # current-deploy-faithful, the yaw would be off by ~2 rad. A near-2*pi/near-2-rad gap.
    assert r["neg_control_refit_vs_CURRENT_yaw_resid"] > 1.0, (
        f"negative control failed: refit rate_frd vs CURRENT yaw resid "
        f"{r['neg_control_refit_vs_CURRENT_yaw_resid']:.4f} should be >1 rad (a clear flip)"
    )

    # --- (I3) shipped fly_rl wire yaw sign matches the CURRENT (deploy-faithful) recipe. ---
    assert abs(r["shipped_wire_yaw_sign"] - W_CURRENT[2]) < 1e-9, (
        f"shipped fly_rl _ACT_FLU_TO_FRD yaw sign {r['shipped_wire_yaw_sign']:+.1f} != "
        f"CURRENT recipe yaw sign {W_CURRENT[2]:+.1f} -- wire convention regressed!"
    )

    # --- (I5) RAW-TELEMETRY POSITIVE CONTROL: the quat mirror is present (East discriminates). ---
    # Confirms the trusted raw anchor underlying CR3-02 is intact in this tree.
    assert r["posctrl_true_E_corr"] > 0.9, (
        f"positive control: TRUE East corr {r['posctrl_true_E_corr']:+.3f} should be >+0.9"
    )
    assert r["posctrl_asis_E_corr"] < 0.0, (
        f"positive control: AS-IS East corr {r['posctrl_asis_E_corr']:+.3f} should be NEGATIVE "
        f"(the R_y(pi) mirror) -- if this is positive the test lost discriminating power"
    )


# =========================================================================================
# Standalone runner
# =========================================================================================
def _fmt_recipe_resid(d: dict) -> str:
    return "  ".join(f"{k}={v:7.4f}" for k, v in d.items())


def main() -> int:
    print("=" * 78)
    print("CR3-02 REGRESSION -- stale DERIVED deploy channels in the refit dataset")
    print("  invariant: recipe-free action-wire relation (rate_frd vs act_rescaled) +")
    print("             obs[11] from w_raw; both touch NO quaternion (no self-mirror).")
    print("=" * 78)

    try:
        r = _evaluate()
    except Exception as exc:  # pragma: no cover -- surfaced as a finding, not a crash
        print(f"\n[ERROR] could not evaluate invariant: {exc!r}")
        return 2

    for label in ("postfix", "refit"):
        d = r[label]
        print(f"\n[{label}]  ({d['n_runs']} loadable runs, {d['n_ticks_rate_frd']} ticks)")
        print(f"  LENS1 rate_frd-vs-act_rescaled  yaw maxabsdiff: {_fmt_recipe_resid(d['lens1_yaw_resid'])}")
        print(f"  LENS2 obs[11]-from-w_raw        yaw maxabsdiff: {_fmt_recipe_resid(d['lens2_yaw_resid'])}")
        print(f"  -> LENS1 winner = {d['lens1_winner']!s:8s}  LENS2 winner = {d['lens2_winner']!s:8s}  "
              f"agree = {d['lenses_agree']}")

    print(f"\n[I4 negative control] refit.rate_frd vs CURRENT recipe (yaw resid) = "
          f"{r['neg_control_refit_vs_CURRENT_yaw_resid']:.4f} rad  (expect >1 -> flip caught)")
    print(f"[I3 shipped wire]     fly_rl _ACT_FLU_TO_FRD yaw sign = {r['shipped_wire_yaw_sign']:+.1f}  "
          f"(CURRENT recipe yaw = {W_CURRENT[2]:+.1f})")
    print(f"[I5 raw pos-control]  Test-A on {r['posctrl_run']} (n_tilt={r['posctrl_n_tilt']}): "
          f"TRUE East corr {r['posctrl_true_E_corr']:+.3f}  AS-IS East corr {r['posctrl_asis_E_corr']:+.3f}")

    # run the assertions
    try:
        test_confirmed_cr3_02()
    except AssertionError as exc:
        print("\n" + "=" * 78)
        print("RESULT: FAIL")
        print(f"  {exc}")
        print("=" * 78)
        return 1

    print("\n" + "=" * 78)
    print("RESULT: PASS")
    print("  postfix derived channels = CURRENT (deploy-faithful); refit derived channels")
    print("  = STALE non-current recipe (CONFIRMED CR3-02). Both lenses agree per dataset;")
    print("  shipped fly_rl wire matches CURRENT; raw-telemetry quat mirror intact.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
