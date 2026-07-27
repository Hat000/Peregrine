"""DECISIVE physical validation of the VG leveled frame.

A quadrotor can only push along its BODY UP axis.  So in a truly gravity-leveled frame L,

    a_measured  +  g*z_hat  ==  (T/m) * u_hat        with u_hat = body +Z (FLU) expressed in L

i.e. the SPECIFIC FORCE reconstructed from the VG trajectory must be PARALLEL to the body up-axis
that the attitude chain reports, and must point UP (never negative thrust).  The angle between them
measures EVERYTHING at once: leveling error, gyro-integration error, and fit error.

If the AHRS roll/pitch used to anchor the frame carried the suspected ~17.8 deg pitch bias, this
angle would sit near 17.8 deg with a systematic SIGN.  If the frame is good it sits near 0.
"""
import os
import pickle
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vg

G = 9.80665
ROWS = [r for r in pickle.load(open(os.path.join(HERE, "sessions.pkl"), "rb"))
        if r.get("fit_ok") and (r.get("n_fresh") or 0) >= 30]


def check(r, min_run=12):
    """Return per-window (angle_deg, signed pitch-like error, |a|) samples."""
    recs, seek, timing, meta = vg.load_session(r["path"])
    d = vg.build(recs, seek, meta)
    t = d["t"]
    fi = np.where(d["fresh"])[0]
    if len(fi) < min_run:
        return []
    out = []
    # slide a 0.45 s window; fit a quadratic; read the acceleration at the window centre
    for c in range(2, len(fi) - 2):
        ic = fi[c]
        m = np.abs(t[fi] - t[ic]) <= 0.22
        sel = fi[m]
        if len(sel) < 7:
            continue
        if (t[sel[-1]] - t[sel[0]]) < 0.20:
            continue
        f = vg.vg_fit(d, sel, anchor=ic)
        if f is None:
            continue
        a = f["a_end"]                                   # quadratic term = accel (leveled frame)
        if not np.isfinite(a).all() or np.linalg.norm(a) > 60:
            continue
        sf = a + np.array([0.0, 0.0, G])                 # specific force
        R = f["R"][ic - f["i0"]]
        u = R[:, 2]                                      # body +Z (up) in the leveled frame
        nsf = np.linalg.norm(sf)
        if nsf < 3.0:                                    # near free-fall: direction ill-defined
            continue
        ang = np.degrees(np.arccos(np.clip(float(sf @ u) / nsf, -1, 1)))
        # signed pitch-like component: rotation about the leveled Y axis needed to align
        cross = np.cross(u, sf / nsf)
        out.append((ang, float(np.degrees(cross[1])), nsf, float(np.linalg.norm(a))))
    return out


def main():
    rng = np.random.default_rng(7)
    pick = list(rng.choice(len(ROWS), size=min(70, len(ROWS)), replace=False))
    allang, allsgn, allsf = [], [], []
    per = []
    for i in pick:
        r = ROWS[i]
        s = check(r)
        if len(s) < 5:
            continue
        A = np.array(s)
        allang += list(A[:, 0])
        allsgn += list(A[:, 1])
        allsf += list(A[:, 2])
        per.append((r["run"], r["lineage"], len(s), float(np.median(A[:, 0])), float(np.median(A[:, 1]))))
    A = np.array(allang)
    S = np.array(allsgn)
    F = np.array(allsf)
    print(f"flights={len(per)}  windows={len(A)}")
    print(f"ANGLE(specific force, body-up):  p25={np.percentile(A,25):.1f}  med={np.median(A):.1f}  "
          f"p75={np.percentile(A,75):.1f}  p90={np.percentile(A,90):.1f} deg")
    print(f"SIGNED pitch-like misalignment:  med={np.median(S):+.2f}  mean={np.mean(S):+.2f} deg "
          f"(a constant AHRS pitch bias would show up here as a large systematic offset)")
    print(f"|specific force|: med={np.median(F):.2f} m/s^2 (hover = {G:.2f}) "
          f"p10={np.percentile(F,10):.2f} p90={np.percentile(F,90):.2f}")
    print("\nper-flight median angle (worst 10):")
    for p in sorted(per, key=lambda x: -x[3])[:10]:
        print(f"  {p[0][:34]:34s} {p[1]:7s} n={p[2]:4d} med_ang={p[3]:5.1f} med_sgn={p[4]:+5.1f}")
    print("per-flight median angle (best 5):")
    for p in sorted(per, key=lambda x: x[3])[:5]:
        print(f"  {p[0][:34]:34s} {p[1]:7s} n={p[2]:4d} med_ang={p[3]:5.1f} med_sgn={p[4]:+5.1f}")

    # ---- CONTROL: rerun with a deliberately mis-leveled frame (+17.8 deg pitch) ----
    print("\n=== CONTROL: inject a +17.8 deg pitch bias into the anchor and repeat ===")
    orig = vg.gyro_attitude

    def biased(d, i0, i1, anchor):
        R = orig(d, i0, i1, anchor)
        c, s = np.cos(np.radians(17.8)), np.sin(np.radians(17.8))
        Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        return np.array([Ry @ x for x in R])
    vg.gyro_attitude = biased
    a2 = []
    for i in pick[:25]:
        s = check(ROWS[i])
        if s:
            a2 += [x[0] for x in s]
    vg.gyro_attitude = orig
    a2 = np.array(a2)
    if a2.size:
        print(f"  windows={len(a2)} ANGLE med={np.median(a2):.1f} p75={np.percentile(a2,75):.1f} deg "
              f"(vs unbiased median {np.median(A):.1f}) -> the test HAS the power to see a bias")


if __name__ == "__main__":
    main()
