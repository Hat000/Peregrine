"""ADVERSARIAL Attack 1 -- effective-N / cluster-bootstrap CI on the L3 gate-4 per-fix sigma.

REFUTE: "the L3-measured gate-4 sigma is honest and the CI is tight."

The published D2 recal reports a CLUSTER CI (sigma_gate4_l3.json) that is NARROWER than its own iid CI
(lateral cluster [0.166,0.210] vs iid [0.161,0.216]; vert cluster [0.085,0.117] vs iid [0.084,0.118]).
That is a red flag: with only ~13 independent windows, resampling WHOLE sessions should NOT shrink the CI.
A cluster bootstrap that resamples sessions and then pools all drawn fixes captures the between-session
sampling variance and (when there is ANY positive within-window correlation) WIDENS the per-fix-sigma CI.

This script:
  1. Reproduces the per-fix sd (ddof=1) from the 81 accepted fixes / 13 sessions.
  2. Runs a CLUSTER bootstrap: resample the 13 sessions WITH replacement, pool the drawn sessions' fixes,
     recompute the per-fix sd. 90% + 95% percentile CI. (B=20000.)
  3. For comparison: an iid (per-fix) bootstrap CI.
  4. Reports the UPPER edge of the cluster CI (the pessimistic per-fix sigma) for both axes + the
     in-plane-radial-equivalent isotropic sigma sqrt((lat^2+vert^2)/2) at the upper edges.
  5. Sanity: empirical ICC via one-way ANOVA (between/within window variance) -- to check the published
     icc~0.03 (lat) / 0 (vert).

Honest note on the engine: ME.fly_lap / driver_v2 take a PER-FIX sigma (the 1-sigma of the fix NOISE),
so the relevant uncertainty is the sampling CI of the per-fix SD, NOT the standard error of the mean.
We therefore bootstrap the SD itself. [verify_sigma adversarial, 2026-06-14]
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROWS = HERE / "data" / "shadow_gate4_rows.json"
B = 20000
SEED = 7777


def load_accepted():
    d = json.loads(ROWS.read_text())
    acc = [r for r in d["rows"] if r.get("accepted")]
    by = {}
    for r in acc:
        by.setdefault(r["session"], []).append(r)
    sessions = list(by.keys())
    vert = {s: np.array([r["rel_vert"] for r in by[s]]) for s in sessions}
    cross = {s: np.array([r["rel_cross"] for r in by[s]]) for s in sessions}
    return sessions, vert, cross, acc


def pooled_sd(arrs):
    """SD (ddof=1) of the pooled concatenation -- matches the recal floor definition (per-fix scatter)."""
    cat = np.concatenate(arrs)
    return float(cat.std(ddof=1))


def cluster_boot(sessions, axis_by_session, rng, b=B):
    sds = np.empty(b)
    sess = np.array(sessions, dtype=object)
    n = len(sess)
    for i in range(b):
        pick = rng.integers(0, n, n)  # resample WHOLE sessions with replacement
        arrs = [axis_by_session[sess[j]] for j in pick]
        sds[i] = pooled_sd(arrs)
    return sds


def iid_boot(allvals, rng, b=B):
    sds = np.empty(b)
    n = len(allvals)
    for i in range(b):
        pick = rng.integers(0, n, n)
        sds[i] = float(allvals[pick].std(ddof=1))
    return sds


def icc_oneway(axis_by_session, sessions):
    """One-way random-effects ICC = (MSB-MSW)/(MSB+(m0-1)MSW). m0 = mean-ish group size adj."""
    groups = [axis_by_session[s] for s in sessions]
    k = len(groups)
    ni = np.array([len(g) for g in groups])
    N = ni.sum()
    grand = np.concatenate(groups).mean()
    ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    dfb, dfw = k - 1, N - k
    msb, msw = ssb / dfb, ssw / dfw
    m0 = (N - (ni ** 2).sum() / N) / (k - 1)
    icc = (msb - msw) / (msb + (m0 - 1) * msw)
    deff = 1 + (m0 - 1) * max(icc, 0.0)
    return dict(icc=float(icc), m0=float(m0), msb=float(msb), msw=float(msw),
                n_eff=float(N / deff), deff=float(deff))


def main():
    sessions, vert, cross, acc = load_accepted()
    rng = np.random.default_rng(SEED)

    vert_all = np.concatenate([vert[s] for s in sessions])
    cross_all = np.concatenate([cross[s] for s in sessions])

    out = {"_meta": {"attack": "1 effective-N / cluster-bootstrap CI",
                     "N": len(acc), "n_windows": len(sessions), "B": B, "seed": SEED,
                     "claim_under_test": "L3 gate-4 sigma CI is tight/honest; published cluster CI is NARROWER than iid"},
           "point_estimate": {
               "vert_sd_ddof1": float(vert_all.std(ddof=1)),
               "cross_sd_ddof1": float(cross_all.std(ddof=1)),
               "vert_mean": float(vert_all.mean()),
               "cross_mean": float(cross_all.mean()),
           }}

    for name, by, allv in (("vert", vert, vert_all), ("lat", cross, cross_all)):
        cb = cluster_boot(sessions, by, rng)
        ib = iid_boot(allv, rng)
        icc = icc_oneway(by, sessions)
        out[name] = {
            "point_sd": float(allv.std(ddof=1)),
            "cluster_ci90": [float(np.percentile(cb, 5)), float(np.percentile(cb, 95))],
            "cluster_ci95": [float(np.percentile(cb, 2.5)), float(np.percentile(cb, 97.5))],
            "cluster_sd_of_sd": float(cb.std(ddof=1)),
            "cluster_mean": float(cb.mean()),
            "iid_ci90": [float(np.percentile(ib, 5)), float(np.percentile(ib, 95))],
            "iid_ci95": [float(np.percentile(ib, 2.5)), float(np.percentile(ib, 97.5))],
            "iid_sd_of_sd": float(ib.std(ddof=1)),
            "icc_oneway": icc,
            "cluster_over_iid_width90": float((np.percentile(cb, 95) - np.percentile(cb, 5))
                                              / (np.percentile(ib, 95) - np.percentile(ib, 5))),
        }

    # Pessimistic UPPER-edge isotropic equivalents
    lat_hi90 = out["lat"]["cluster_ci90"][1]
    vert_hi90 = out["vert"]["cluster_ci90"][1]
    lat_hi95 = out["lat"]["cluster_ci95"][1]
    vert_hi95 = out["vert"]["cluster_ci95"][1]
    out["pessimistic_upper_edge"] = {
        "aniso_upper_ci90": [lat_hi90, vert_hi90],
        "aniso_upper_ci95": [lat_hi95, vert_hi95],
        "iso_equiv_upper_ci90": float(np.sqrt((lat_hi90 ** 2 + vert_hi90 ** 2) / 2)),
        "iso_equiv_upper_ci95": float(np.sqrt((lat_hi95 ** 2 + vert_hi95 ** 2) / 2)),
        "published_aniso_conservative": [0.2101, 0.1167],
        "published_iso_conservative": 0.167744,
    }
    print(json.dumps(out, indent=2))
    (HERE / "verify_sigma_cluster_boot_results.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
