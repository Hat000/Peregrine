"""Calibrate the range-anisotropic R coefficients on a TRAIN split of the perception-char fixes.

Splits the 6 per-gate characterize dumps into TRAIN (gates 0,2,4) and TEST (gates 1,3,5) by the
ASSOCIATED gate_id, pools the associated fixes, decomposes each fix error into a RADIAL (along true
LOS) and TANGENTIAL component (using the GT drone position from frames.json + the map LOS direction),
bins by true range, and fits:

    radial:      var_radial(r) = (c2 r^2)^2 + (c1 r)^2 + sig0_rad^2
    tangential:  var_tang(r)   = (a1 r)^2 + sigma_theta^2 |L_perp|^2(=r^2) + sig0_tan^2

The c2 (r^2 depth) coefficient is PINNED to the Fisher-derived physics value (validated by MC); we fit
ONLY the empirical add-ons (c1 size-mismatch bias, sig0 floors, a1 lateral) so the model stays
physically anchored and cannot overfit the small sample by absorbing the r^2 law into a free knob. The
attitude-lever sigma_theta is kept at the shipped 1.4 deg (physically calibrated, frames.py).

OUTLIER GUARD. Even after the |off|<3 m cut, a handful of partial depth-flips survive. We fit the
radial/tangential VARIANCE per band with a 90th-percentile-trimmed (robust) variance so a single 2.9 m
flip does not dominate the r^2 coefficient. The held-out NIS in validate_R.py is computed on the FULL
test set (no trim) so the report is honest about tail behaviour.

Writes range_R_coeffs.json next to range_anisotropic_R.py. Run:
    .venv/Scripts/python.exe handoff/ultracode-vision-case-c-2026-06-13/fit_range_R.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from racer.frames import ATTITUDE_NOISE_STD_RAD  # noqa: E402

import range_anisotropic_R as RA  # noqa: E402  (same dir)

CHAR_DIR = REPO / "handoff" / "perception-char-2026-06-08"
MAP_PATH = REPO / "handoff" / "shadowpc-firstcontact-2026-06-02" / "track_map.json"
FRAMES_PATH = CHAR_DIR / "course_bundle" / "frames.json"

GOOD_FIX_MAX_M = 3.0     # README "good fix" cut (|off_ned| < 3 m)
TRAIN_GATES = {0, 2, 4}
TEST_GATES = {1, 3, 5}


def load_map() -> dict[int, np.ndarray]:
    g = json.loads(MAP_PATH.read_text())["gates"]
    return {gg["gate_id"]: np.asarray(gg["position_ned"], float) for gg in g}


def load_frames_pergate(g: int) -> dict[int, dict]:
    """Per-gate frame bundle (pg/course_g{g}/frames.json) -- the bundle characterize_g{g} was built
    from. Each per-gate bundle has its own 60 frames (no overlap across gates), so each characterize
    file MUST join to its own bundle, not the shared course_bundle (which holds only 240 frames)."""
    p = CHAR_DIR / "pg" / f"course_g{g}" / "frames.json"
    fr = json.loads(p.read_text())["frames"]
    return {f["frame_id"]: f for f in fr}


def load_fixes() -> list[dict]:
    """Pool all associated fixes across the 6 per-gate dumps, each joined to its OWN per-gate frame
    bundle for the GT drone position, then decomposed into radial/tangential error."""
    gmap = load_map()
    out: list[dict] = []
    for g in range(6):
        fbid = load_frames_pergate(g)
        d = json.loads((CHAR_DIR / f"characterize_g{g}.json").read_text())
        for r in d["rows"]:
            if not (r.get("associated") and r.get("pose_range_m") is not None):
                continue
            f = fbid.get(r["frame_id"])
            if f is None:
                continue
            gid = r["gate_id"]
            drone = np.asarray(f["drone_position_ned"], float)
            off = np.asarray(r["off_ned"], float)
            lever_true = gmap[gid] - drone                  # true LOS (world NED), gate rel drone
            nL = float(np.linalg.norm(lever_true))
            if nL < 1e-6:
                continue
            Lhat = lever_true / nL
            rad = float(off @ Lhat)                         # radial fix-error (signed, +toward gate)
            tang_vec = off - rad * Lhat
            out.append(dict(
                frame_id=r["frame_id"], gate_id=gid, source_gate=g, true_range=r["true_range_m"],
                pose_range=r["pose_range_m"], range_err=r["range_err_m"],
                off=off, rad=rad, tang=float(np.linalg.norm(tang_vec)),
                n_corners=r["n_corners"], reproj=r["reproj_px"], maha=r["maha"],
                wfe=r["world_fix_err_m"], drone=drone, lever_true=lever_true,
            ))
    return out


def _trimmed_var(x: np.ndarray, q: float = 0.90) -> float:
    """Robust variance: mean-square of the |x| values below the q-quantile (drops the top tail)."""
    x = np.asarray(x, float)
    if x.size == 0:
        return 0.0
    thr = np.quantile(np.abs(x), q)
    keep = np.abs(x) <= thr if np.any(np.abs(x) <= thr) else np.ones_like(x, bool)
    return float(np.mean(x[keep] ** 2))


def fit(train: list[dict]) -> dict:
    """Fit c1, sig0_rad, a1, sig0_tan (c2 + sigma_theta pinned to physics)."""
    tr = np.array([x["true_range"] for x in train])
    rad = np.array([x["rad"] for x in train])
    tang = np.array([x["tang"] for x in train])

    c2 = RA._c2_from_pixels(RA._SIGMA_PX_DEFAULT)            # pinned r^2 depth (Fisher, MC-confirmed)
    sigma_theta = float(ATTITUDE_NOISE_STD_RAD)             # pinned attitude lever

    # Per-band ROBUST variance of the radial / tangential channels.
    bands = [(0, 8), (8, 12), (12, 16), (16, 24)]
    bc, rv, tv = [], [], []
    for lo, hi in bands:
        m = (tr >= lo) & (tr < hi)
        if m.sum() < 3:
            continue
        bc.append(float(tr[m].mean()))
        rv.append(_trimmed_var(rad[m]))
        tv.append(_trimmed_var(tang[m]))
    bc = np.array(bc); rv = np.array(rv); tv = np.array(tv)

    # RADIAL fit: rv(r) = (c2 r^2)^2 + (c1 r)^2 + sig0_rad^2. Subtract the PINNED r^2 term, then fit
    # the residual variance as A r^2 + B (linear in r^2) with A = c1^2, B = sig0_rad^2, A,B >= 0.
    resid = rv - (c2 * bc**2) ** 2
    X = np.column_stack([bc**2, np.ones_like(bc)])
    # non-negative least squares (tiny problem; 2 params)
    from scipy.optimize import nnls
    sol, _ = nnls(X, np.maximum(resid, 0.0))
    c1 = float(np.sqrt(max(sol[0], 0.0)))
    sig0_rad = float(np.sqrt(max(sol[1], 0.0)))
    # Floor the radial constant at the shipped 0.40 m if the fit pushes it lower (the 0.40 m absorbs
    # close-range systematics the bands cannot see; never go below the live model's floor).
    sig0_rad = max(sig0_rad, RA.FIX_COV_FLOOR_STD)

    # TANGENTIAL fit: tv(r) = (a1 r)^2 + sigma_theta^2 r^2 + sig0_tan^2. The attitude lever contributes
    # sigma_theta^2 |L_perp|^2; for the perpendicular component |L_perp| ~ r. Subtract it, fit residual
    # = (a1^2 + ...) r^2 + sig0_tan^2.
    tresid = tv - (sigma_theta**2) * bc**2
    solt, _ = nnls(X, np.maximum(tresid, 0.0))
    a1 = float(np.sqrt(max(solt[0], 0.0)))
    sig0_tan = float(np.sqrt(max(solt[1], 0.0)))

    coeffs = dict(RA._DEFAULT_COEFFS)
    coeffs.update(dict(
        c2=c2, c1=c1, sig0_rad=sig0_rad, a1=a1, sig0_tan=sig0_tan,
        sigma_theta=sigma_theta, sig0_iso=0.0, p3p_inflation=9.0,
        source=f"fit_range_R.py TRAIN gates {sorted(TRAIN_GATES)} N={len(train)} (c2,sigma_theta pinned to physics)",
    ))
    return coeffs, dict(bands=bands, band_center=bc.tolist(), radial_var=rv.tolist(),
                        tang_var=tv.tolist())


def main() -> None:
    fixes = load_fixes()
    good = [x for x in fixes if x["wfe"] < GOOD_FIX_MAX_M]
    train = [x for x in good if x["source_gate"] in TRAIN_GATES]
    test = [x for x in good if x["source_gate"] in TEST_GATES]
    print(f"pooled associated={len(fixes)}  good(|off|<{GOOD_FIX_MAX_M})={len(good)}  "
          f"train(g{sorted(TRAIN_GATES)})={len(train)}  test(g{sorted(TEST_GATES)})={len(test)}")
    coeffs, diag = fit(train)
    print("\nFITTED coefficients (TRAIN only):")
    for k, v in coeffs.items():
        print(f"  {k:14s} = {v if isinstance(v, str) else round(v, 6)}")
    print("\nTRAIN per-band robust variance (radial / tangential):")
    for (lo, hi), bc, rvv, tvv in zip(diag["bands"], diag["band_center"], diag["radial_var"], diag["tang_var"]):
        # bands without >=3 samples are dropped in fit(); guard alignment by length
        pass
    for i, bc in enumerate(diag["band_center"]):
        print(f"  r~{bc:5.1f} m   var_rad={diag['radial_var'][i]:.4f} (std {np.sqrt(diag['radial_var'][i]):.3f})"
              f"   var_tang={diag['tang_var'][i]:.4f} (std {np.sqrt(diag['tang_var'][i]):.3f})")

    out = RA._COEFFS_PATH
    out.write_text(json.dumps(coeffs, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
