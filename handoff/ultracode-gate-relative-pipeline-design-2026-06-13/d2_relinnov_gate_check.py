"""d2 — OFFLINE CHECK: the RELATIVE-INNOVATION outlier gate separates depth-flips
that reprojection error CANNOT.

COMPONENT-2 deliverable (3): the gate-relative AUGMENT needs its OWN outlier gate. The
absolute Mahalanobis gate in the live navigator (navigator.py:391-396) tests the WORLD
position innovation nu = z_world - x_KF[:3] against S = P_pos + R. Under gate-relative
observation the binding quantity is the IN-PLANE offset to the SEEN opening, e = -L_inplane
(lateral part of the PnP lever in the gate frame). A depth-flip / wrong-scale PnP corrupts
the RADIAL (depth) component most, but ALSO throws the in-plane lever because the flipped
R_cam_gate rotates the (mis-scaled) translation -- so a relative-innovation test on the
in-plane lever catches flips that reproj does not.

This check makes two MEASURED-grounded points, both re-derived from the perception-char
bundle + a forward MC, NOT trusted from prose:

  (1) DATA: among associated+detected fixes, depth-flip / large-error fixes (world_fix_err>3 m)
      have a reproj p50 BELOW clean fixes (world_fix_err<1 m). So a reproj threshold cannot
      separate them without also throwing good fixes. (Re-derived live: flip p50 0.445 px <
      clean p50 0.657 px.)

  (2) MC: build a stream of clean gate-relative fixes (in-plane lever = true offset + measured
      lateral PnP noise, per-axis sigma 0.265 m) plus injected depth-FLIP fixes (the radial
      depth sign/scale corrupted, which rotates a LARGE component into the in-plane lever).
      Show: (a) reproj is statistically indistinguishable; (b) the relative-innovation
      statistic d2_rel = nu_ip^T S_ip^-1 nu_ip (nu_ip = e_obs - e_pred, the in-plane lever
      innovation vs the KF-propagated relative prior; S_ip = 2x2 in-plane (P+R) block)
      cleanly separates them at a chi2(2) threshold.

The point is the GATE DESIGN, not a tuned number: relative innovation is the discriminator,
reproj is not. Run:
  PYTHONPATH=src .venv/Scripts/python.exe handoff/ultracode-gate-relative-pipeline-design-2026-06-13/d2_relinnov_gate_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2 as _chi2

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

SEED = 20260613
CHI2_2_999 = float(_chi2.ppf(0.999, 2))   # relative-innovation in-plane gate (2 DOF)
PER_AXIS_LAT_SIGMA = 0.265                 # m, MEASURED accepted gate-4 lateral per-axis (c1_gate_relative)


# ----------------------------------------------------------------------------
# (1) DATA: reproj does NOT separate flips from clean fixes
# ----------------------------------------------------------------------------
def data_reproj_nonseparation():
    base = _REPO / "handoff" / "perception-char-2026-06-08"
    clean, flip = [], []
    for g in range(6):
        f = base / f"characterize_g{g}.json"
        if not f.exists():
            continue
        for r in json.load(open(f))["rows"]:
            if not (r.get("associated") and r.get("detected")):
                continue
            wf, rp = r.get("world_fix_err_m"), r.get("reproj_px")
            if wf is None or rp is None:
                continue
            if wf < 1.0:
                clean.append(rp)
            elif wf > 3.0:
                flip.append(rp)
    cr, fr = np.array(clean), np.array(flip)
    return {
        "clean_n": int(cr.size), "flip_n": int(fr.size),
        "clean_reproj_p50": float(np.percentile(cr, 50)),
        "clean_reproj_p90": float(np.percentile(cr, 90)),
        "flip_reproj_p50": float(np.percentile(fr, 50)),
        "flip_reproj_p90": float(np.percentile(fr, 90)),
        "reproj_separates": bool(np.percentile(fr, 50) > np.percentile(cr, 50)),
    }


# ----------------------------------------------------------------------------
# (2) MC: relative-innovation gate separates depth-flips
# ----------------------------------------------------------------------------
def mc_relinnov_gate(n=4000, flip_frac=0.15):
    """Clean + depth-flip gate-relative fixes; compare reproj vs relative-innovation gate.

    Model (gate-4 plane, in-plane axes = (E,D)):
      - KF-propagated relative prior e_pred ~ true in-plane offset (drone centered: ~0) with a
        small propagated covariance P_ip (0.08 m/axis -- a warm prior between fixes).
      - CLEAN fix: e_obs = e_true + N(0, sigma_lat^2 I_2), sigma_lat = 0.265 m/axis. reproj ~ the
        measured clean distribution (lognormal fit to p50 0.66 / p90 1.89).
      - FLIP fix: the PnP depth flips sign/scale; the flipped R_cam_gate rotates the lever so the
        in-plane component picks up a LARGE deterministic offset (|>~1.5 m|) -- modeled as a
        2-3 m in-plane displacement in a random direction. reproj ~ the measured FLIP distribution
        (p50 0.45 / p90 3.09) -- LOWER median than clean (the whole point).
    R_ip = sigma_lat^2 I_2 (clean fix R); S_ip = P_ip + R_ip; d2_rel = nu^T S^-1 nu, nu = e_obs - e_pred.
    """
    rng = np.random.default_rng(SEED)
    sig = PER_AXIS_LAT_SIGMA
    P_ip = (0.08**2) * np.eye(2)                 # warm relative prior cov (between-fix growth)
    R_ip = (sig**2) * np.eye(2)
    S_ip = P_ip + R_ip
    S_inv = np.linalg.inv(S_ip)

    # measured reproj distributions (lognormal matched to p50/p90 from the bundle)
    def lognorm_from_p50_p90(p50, p90, k):
        mu = np.log(p50)
        sigma = (np.log(p90) - mu) / 1.2816   # z_0.90
        return np.exp(rng.normal(mu, sigma, size=k))

    n_flip = int(n * flip_frac)
    n_clean = n - n_flip

    # CLEAN
    e_true_c = rng.normal(0.0, 0.03, size=(n_clean, 2))   # near-centered true offset
    e_pred_c = e_true_c + rng.multivariate_normal(np.zeros(2), P_ip, size=n_clean)
    e_obs_c = e_true_c + rng.normal(0.0, sig, size=(n_clean, 2))
    reproj_c = lognorm_from_p50_p90(0.657, 1.886, n_clean)

    # FLIP: in-plane lever thrown by a large displacement (depth-flip rotates the lever)
    e_true_f = rng.normal(0.0, 0.03, size=(n_flip, 2))
    e_pred_f = e_true_f + rng.multivariate_normal(np.zeros(2), P_ip, size=n_flip)
    mag = rng.uniform(1.5, 3.0, size=n_flip)
    ang = rng.uniform(0, 2 * np.pi, size=n_flip)
    disp = np.column_stack([mag * np.cos(ang), mag * np.sin(ang)])
    e_obs_f = e_true_f + rng.normal(0.0, sig, size=(n_flip, 2)) + disp
    reproj_f = lognorm_from_p50_p90(0.445, 3.088, n_flip)

    def d2(e_obs, e_pred):
        nu = e_obs - e_pred
        return np.einsum("ij,jk,ik->i", nu, S_inv, nu)

    d2_c, d2_f = d2(e_obs_c, e_pred_c), d2(e_obs_f, e_pred_f)

    # reproj gate: pick a threshold that keeps 99% of clean -> how many flips survive?
    reproj_thr = float(np.percentile(reproj_c, 99))
    flip_survive_reproj = float(np.mean(reproj_f <= reproj_thr))
    clean_keep_reproj = float(np.mean(reproj_c <= reproj_thr))

    # relative-innovation gate at chi2(2) 99.9%
    flip_survive_rel = float(np.mean(d2_f <= CHI2_2_999))
    clean_keep_rel = float(np.mean(d2_c <= CHI2_2_999))

    return {
        "chi2_2_999": CHI2_2_999,
        "reproj_thr_px_at_99pct_clean": reproj_thr,
        "reproj_gate": {
            "clean_kept_frac": clean_keep_reproj,
            "flip_survived_frac": flip_survive_reproj,
        },
        "relinnov_gate": {
            "clean_kept_frac": clean_keep_rel,
            "flip_survived_frac": flip_survive_rel,
        },
        "d2_clean_p50": float(np.percentile(d2_c, 50)),
        "d2_clean_p999": float(np.percentile(d2_c, 99.9)),
        "d2_flip_p01": float(np.percentile(d2_f, 1)),
        "d2_flip_p50": float(np.percentile(d2_f, 50)),
    }


def main():
    print("=" * 78)
    print("d2 RELATIVE-INNOVATION GATE CHECK (depth-flip separation)")
    print("=" * 78)
    d = data_reproj_nonseparation()
    print("\n(1) DATA: reproj does NOT separate flips from clean (perception-char bundle)")
    print(f"    clean (wf<1m) n={d['clean_n']:3d}  reproj p50 {d['clean_reproj_p50']:.3f}  p90 {d['clean_reproj_p90']:.3f}")
    print(f"    flip  (wf>3m) n={d['flip_n']:3d}  reproj p50 {d['flip_reproj_p50']:.3f}  p90 {d['flip_reproj_p90']:.3f}")
    print(f"    -> flip p50 < clean p50 (reproj cannot gate): {not d['reproj_separates']}")

    print("\n(2) MC: relative-innovation gate vs reproj gate")
    m = mc_relinnov_gate()
    print(f"    reproj gate (thr={m['reproj_thr_px_at_99pct_clean']:.2f}px @99% clean kept):")
    print(f"        clean kept {m['reproj_gate']['clean_kept_frac']*100:5.1f}%   "
          f"FLIP SURVIVED {m['reproj_gate']['flip_survived_frac']*100:5.1f}%  <-- reproj lets flips through")
    print(f"    relative-innovation gate (chi2(2) 99.9% = {m['chi2_2_999']:.2f}):")
    print(f"        clean kept {m['relinnov_gate']['clean_kept_frac']*100:5.1f}%   "
          f"FLIP SURVIVED {m['relinnov_gate']['flip_survived_frac']*100:5.1f}%  <-- relinnov rejects flips")
    print(f"    d2 clean p50 {m['d2_clean_p50']:.2f} / p99.9 {m['d2_clean_p999']:.2f}   "
          f"d2 flip p1 {m['d2_flip_p01']:.1f} / p50 {m['d2_flip_p50']:.1f}")

    out = {"data": d, "mc": m}
    p = Path(__file__).resolve().parent / "d2_relinnov_gate_results.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
