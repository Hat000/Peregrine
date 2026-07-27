"""Diagnose the 10-vs-41 split: why the two prior instruments disagreed 4x.

Instrument A (vision-called) ran the centre-then-diverge detector `lmin<=0.45 & reopen>=0.30` on
y_gate -- a position series built from the VISION ESTIMATE (rho * bearing, rotated into a gate frame).
Instrument B (track-corroborated) ran the same detector on the drone's DEAD-RECKONED KF track.

The VG instrument adjudicates both, because it measures the drone's motion RELATIVE TO THE GATE
directly (a fixed world object seen through a trusted gyro), so:

  * VG residuals  = the vision estimate's own noise  -> feeds a MONTE-CARLO false-positive test of A
  * VG vs KF displacement (after the best single yaw alignment) = the KF's drift -> measures B's blindness
  * the detector re-run on the VG-SMOOTHED trajectory = the count neither old instrument could give
"""
import os
import pickle
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import terminal
import vg

ROWS = pickle.load(open(os.path.join(HERE, "sessions.pkl"), "rb"))
RNG = np.random.default_rng(17)


def detect(rng, lat, r_hi=12.0, r_lo=1.0):
    """The LEGACY detector, verbatim in shape: lmin over the range window, reopen = end - min."""
    ok = np.isfinite(rng) & np.isfinite(lat) & (rng <= r_hi) & (rng >= r_lo)
    jj = np.where(ok)[0]
    if len(jj) < 5:
        return None
    a = np.abs(lat[jj])
    im = int(np.argmin(a))
    return dict(lmin=float(a[im]), reopen=float(a[-1] - a[im]),
                fires=bool(a[im] <= 0.45 and (a[-1] - a[im]) >= 0.30))


def per_run(r):
    recs, seek, tm, meta = vg.load_session(r["path"])
    d = vg.build(recs, seek, meta)
    t = d["t"]
    n = len(t)
    g = int(d["gi"][-1])
    sel = terminal.approach_sel(d, g, n - 1)
    if len(sel) < 10:
        return None
    sm = vg.vg_smooth(d, sel)
    if sm is None:
        return None
    good = np.isfinite(sm["p_sm"]).all(axis=1)
    if good.sum() < 8:
        return None
    p_raw = sm["p_raw"][good]
    p_sm = sm["p_sm"][good]
    idx = sm["idx"][good]
    # axis: robust direction of travel over the approach (VG-smoothed endpoints)
    axis = p_sm[-1] - p_sm[max(0, len(p_sm) - 8)]
    if np.linalg.norm(axis) < 0.5:
        return None
    a = axis / np.linalg.norm(axis)

    def lat_of(P):
        s = P @ a
        return np.linalg.norm(P - s[:, None] * a, axis=1)
    rng_raw = np.linalg.norm(p_raw, axis=1)
    rng_sm = np.linalg.norm(p_sm, axis=1)
    dA = detect(rng_raw, lat_of(p_raw))     # instrument A analogue: the VISION ESTIMATE series
    dC = detect(rng_sm, lat_of(p_sm))       # instrument C: the DRONE'S OWN motion
    # ---- KF track (instrument B analogue): displacement relative to the approach start ----
    kf = d["p_zup_kf"][idx]
    dkf = kf - kf[0]
    dvg = p_sm - p_sm[0]
    dB = None
    kf_drift = np.nan
    if np.isfinite(dkf).all() and np.linalg.norm(dkf[-1]) > 1.0:
        # best single yaw rotation mapping the KF displacement onto the VG displacement
        num = float(np.sum(dkf[:, 0] * dvg[:, 1] - dkf[:, 1] * dvg[:, 0]))
        den = float(np.sum(dkf[:, 0] * dvg[:, 0] + dkf[:, 1] * dvg[:, 1]))
        th = np.arctan2(num, den)
        c, s2 = np.cos(th), np.sin(th)
        Rz = np.array([[c, -s2, 0], [s2, c, 0], [0, 0, 1.0]])
        dkf_r = dkf @ Rz.T
        kf_drift = float(np.linalg.norm(dkf_r[-1] - dvg[-1]))
        p_kf = p_sm[0] + dkf_r            # the drone's position rel. the gate AS THE KF BELIEVES IT
        dB = detect(np.linalg.norm(p_kf, axis=1), lat_of(p_kf))
    # ---- residual noise of the vision estimate (what instrument A actually measured) ----
    res = p_raw - p_sm
    bh = np.einsum("nij,nj->ni", np.array([sm["R"][i - sm["i0"]] for i in idx]),
                   d["rel"][idx] / np.maximum(d["rho"][idx], 1e-9)[:, None])
    rr = np.einsum("ni,ni->n", res, bh)
    rt = np.linalg.norm(res - rr[:, None] * bh, axis=1)
    # ---- MONTE CARLO: inject the measured estimate noise into the TRUE trajectory ----
    fires = 0
    NS = 60
    for _ in range(NS):
        pert = p_sm + (RNG.permutation(rr)[:, None] * bh
                       + (RNG.permutation(rt) * RNG.standard_normal(len(rt)))[:, None]
                       * np.cross(bh, np.tile(a, (len(bh), 1))))
        dd = detect(np.linalg.norm(pert, axis=1), lat_of(pert))
        if dd and dd["fires"]:
            fires += 1
    return dict(run=r["run"], lineage=r["lineage"], gate=g, n=int(good.sum()),
                A=dA, B=dB, C=dC, mc_rate=fires / NS,
                rms_r=float(np.sqrt(np.mean(rr ** 2))), rms_t=float(np.sqrt(np.mean(rt ** 2))),
                kf_drift=kf_drift, path_len=float(np.linalg.norm(dvg[-1])))


def main():
    pool = [r for r in ROWS if r["lineage"] == "v19" and r.get("final_state") == "CRASH"
            and (r.get("n_ticks") or 0) > 20]
    print(f"v19 crash pool: {len(pool)}")
    out = []
    for i, r in enumerate(pool):
        try:
            x = per_run(r)
        except Exception:
            x = None
        if x:
            out.append(x)
    pickle.dump(out, open(os.path.join(HERE, "diag10v41.pkl"), "wb"))
    ok = [x for x in out if x["A"] and x["C"]]
    okB = [x for x in ok if x["B"]]
    print(f"analysable: {len(ok)} (with a usable KF track: {len(okB)})")
    A = sum(1 for x in ok if x["A"]["fires"])
    C = sum(1 for x in ok if x["C"]["fires"])
    B = sum(1 for x in okB if x["B"]["fires"])
    print(f"\nSIGNATURE COUNT on the SAME {len(ok)} fatal approaches, same detector:")
    print(f"  A  VISION-ESTIMATE series : {A:3d}  ({100*A/len(ok):.0f}%)")
    print(f"  B  KF DEAD-RECKONED track : {B:3d}  ({100*B/max(len(okB),1):.0f}% of {len(okB)})")
    print(f"  C  VG DRONE-MOTION series : {C:3d}  ({100*C/len(ok):.0f}%)   <- the adjudicated count")
    print(f"\n  A&C={sum(1 for x in ok if x['A']['fires'] and x['C']['fires'])}  "
          f"A-only={sum(1 for x in ok if x['A']['fires'] and not x['C']['fires'])}  "
          f"C-only={sum(1 for x in ok if x['C']['fires'] and not x['A']['fires'])}")
    print("\n=== WHY A OVER-COUNTS: Monte-Carlo false-positive test ===")
    print("  Inject each flight's OWN measured estimate noise into its OWN true (VG) trajectory,")
    print("  then re-run the detector. Firing rate = the probability the signature is manufactured")
    print("  by estimate noise alone.")
    mc = np.array([x["mc_rate"] for x in ok])
    nofire = [x for x in ok if not x["C"]["fires"]]
    mcn = np.array([x["mc_rate"] for x in nofire])
    print(f"  all flights          : mean MC firing rate {mc.mean():.2f}")
    print(f"  flights whose TRUE trajectory does NOT fire (n={len(nofire)}): mean MC rate {mcn.mean():.2f}")
    print(f"    -> estimate noise alone manufactures the signature on {100*mcn.mean():.0f}% of draws")
    aonly = [x for x in ok if x["A"]["fires"] and not x["C"]["fires"]]
    print(f"  A-only flights (n={len(aonly)}): mean MC rate {np.mean([x['mc_rate'] for x in aonly]):.2f}, "
          f"median vision-estimate rms radial {np.median([x['rms_r'] for x in aonly]):.2f} m")
    both = [x for x in ok if x["A"]["fires"] and x["C"]["fires"]]
    if both:
        print(f"  A&C flights   (n={len(both)}): mean MC rate {np.mean([x['mc_rate'] for x in both]):.2f}, "
              f"median rms radial {np.median([x['rms_r'] for x in both]):.2f} m")
    print("\n  measured estimate noise over these approaches:")
    print(f"    radial (range) rms  : p50={np.median([x['rms_r'] for x in ok]):.2f} m  "
          f"p90={np.percentile([x['rms_r'] for x in ok],90):.2f}")
    print(f"    transverse (bearing): p50={np.median([x['rms_t'] for x in ok]):.2f} m  "
          f"p90={np.percentile([x['rms_t'] for x in ok],90):.2f}")
    print("\n=== WHY B UNDER-COUNTS: the KF track's own drift ===")
    dr = np.array([x["kf_drift"] for x in okB if np.isfinite(x["kf_drift"])])
    pl = np.array([x["path_len"] for x in okB if np.isfinite(x["kf_drift"])])
    print(f"  |KF displacement - VG displacement| after the best single yaw alignment, over one approach:")
    print(f"    p25={np.percentile(dr,25):.2f}  p50={np.median(dr):.2f}  p75={np.percentile(dr,75):.2f}  "
          f"p90={np.percentile(dr,90):.2f} m   (path length p50 {np.median(pl):.1f} m)")
    print(f"    drift as a fraction of the path: p50={100*np.median(dr/np.maximum(pl,1e-6)):.0f}%")
    print(f"  For scale, the detector's own thresholds are lmin 0.45 m and reopen 0.30 m.")
    lm = np.array([x["B"]["lmin"] for x in okB])
    lmC = np.array([x["C"]["lmin"] for x in okB])
    ro = np.array([x["B"]["reopen"] for x in okB])
    roC = np.array([x["C"]["reopen"] for x in okB])
    print(f"  reopen: KF p50={np.median(ro):.2f}  VG p50={np.median(roC):.2f}  "
          f"(KF reports {100*np.median(ro)/max(np.median(roC),1e-6):.0f}% of the real lateral excursion)")
    print(f"  lmin  : KF p50={np.median(lm):.2f}  VG p50={np.median(lmC):.2f}")


if __name__ == "__main__":
    main()
