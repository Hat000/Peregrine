"""GROUND-TRUTH validation of the VG instrument on confirmed gate PASSES.

Every RACE_STATUS gate advance is a certified fact: the drone went THROUGH a 1.5 m square opening
without touching it (gate contact invalidates the run and is logged as a collision).  So at every
pass the true miss satisfies |y| < 0.75 m AND |z| < 0.75 m.

That makes the corpus's ~1400 passes a labelled validation set for the instrument.  If the VG
ballistic miss at a pass reads inside the opening, the instrument is calibrated; the spread of the
readings IS its measurement uncertainty, and it sets the confidence bands for the census.

Also emits the crash distribution for the same statistic -> the separation the classifier rests on.
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

ROWS = pickle.load(open(os.path.join(HERE, "sessions.pkl"), "rb"))


def events(r, want="pass"):
    """Yield (miss_h, miss_v, d_end, s_end, rho_last, speed, n_fit, gate) per gate event."""
    recs, seek, timing, meta = vg.load_session(r["path"])
    d = vg.build(recs, seek, meta)
    t = d["t"]
    n = len(t)
    ao = d["aim_off"]
    aim = (np.abs(ao[:, 0]) > 1e-6) | (np.abs(ao[:, 1]) > 1e-6)
    fresh = d["fresh"] & ~aim
    out = []
    # gate advance indices
    adv = [i for i in range(1, n) if d["gi"][i] > d["gi"][i - 1]]
    marks = adv if want == "pass" else [n - 1]
    for i_end in marks:
        g = int(d["gi"][i_end - 1]) if want == "pass" else int(d["gi"][-1])
        sel_all = np.where(fresh[:i_end] & (d["gi"][:i_end] == g))[0]
        if len(sel_all) < 5:
            continue
        lf = int(sel_all[-1])
        f = None
        for win in (0.55, 0.85, 1.3):
            sel = sel_all[t[sel_all] >= t[lf] - win]
            if len(sel) >= 5:
                f = vg.vg_fit(d, sel)
                if f is not None and np.isfinite(f["p_end"]).all():
                    break
                f = None
        if f is None:
            continue
        sp = float(np.linalg.norm(f["v_end"]))
        tstar, miss, mm = vg.closest_approach(f["p_end"], f["v_end"])
        vh = f["v_end"] / max(sp, 1e-9)
        out.append(dict(run=r["run"], lineage=r["lineage"], gate=g, kind=want,
                        miss_h=float(np.hypot(miss[0], miss[1])), miss_v=float(miss[2]),
                        d_end=float(np.linalg.norm(f["p_end"])),
                        s_end=float(-(f["p_end"] @ vh)),
                        speed=sp, nfit=len(f["idx"]), tstar=float(tstar),
                        rms_r=f["rms_radial"], rms_t=f["rms_trans"],
                        dt_fresh=float(t[i_end - 1] - t[lf]),
                        last_rho=float(d["rho"][lf])))
    return out


def qs(name, a, qq=(10, 25, 50, 75, 90, 95, 99)):
    a = np.array([x for x in a if np.isfinite(x)])
    if not a.size:
        print(f"  {name}: n=0"); return
    print(f"  {name}: n={len(a)} " + " ".join(f"p{x}={np.percentile(a,x):+.2f}" for x in qq)
          + f" max={a.max():+.2f}")


def main():
    use = [r for r in ROWS if r.get("fit_ok") and (r.get("n_ticks") or 0) > 20]
    P, C = [], []
    for i, r in enumerate(use):
        try:
            P += events(r, "pass")
            if r.get("final_state") == "CRASH":
                C += events(r, "crash")
        except Exception:
            pass
        if i % 100 == 0:
            print(f"  [{i}/{len(use)}]", flush=True)
    pickle.dump(dict(passes=P, crashes=C), open(os.path.join(HERE, "gate_events.pkl"), "wb"))
    print(f"\nPASS events: {len(P)}   CRASH events: {len(C)}")
    ph = np.array([e["miss_h"] for e in P])
    pv = np.array([e["miss_v"] for e in P])
    print("\n=== GROUND TRUTH: confirmed PASSES (true miss is inside +-0.75 m by construction) ===")
    qs("miss_h", ph)
    qs("|miss_v|", np.abs(pv))
    qs("miss_v (signed)", pv)
    for th in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        print(f"   miss_h<= {th}: {100*np.mean(ph<=th):5.1f}%    "
              f"|miss_v|<= {th}: {100*np.mean(np.abs(pv)<=th):5.1f}%")
    print("\n  -> readings ABOVE 0.75 m on a pass are pure instrument error; that tail is the")
    print("     false-positive rate of any 'strike' threshold placed at 0.75 m.")
    ch = np.array([e["miss_h"] for e in C])
    cv = np.array([e["miss_v"] for e in C])
    print("\n=== CRASH terminal events (same statistic) ===")
    qs("miss_h", ch)
    qs("miss_v (signed)", cv)
    # separation
    print("\n=== separation pass vs crash ===")
    for nm, a, b in (("miss_h", ph, ch), ("|miss_v|", np.abs(pv), np.abs(cv))):
        aa, bb = a[np.isfinite(a)], b[np.isfinite(b)]
        auc = np.mean([np.mean(bb > x) + 0.5 * np.mean(bb == x) for x in aa])
        print(f"  AUC({nm}: crash>pass) = {auc:.3f}")
    print("\n=== instrument noise on passes, by data quality ===")
    for lo, hi in ((0, 0.10), (0.10, 0.25), (0.25, 0.6), (0.6, 99)):
        m = [e for e in P if lo <= e["rms_r"] < hi]
        if len(m) > 20:
            print(f"  rms_radial in [{lo},{hi}): n={len(m):4d} "
                  f"miss_h p50={np.percentile([e['miss_h'] for e in m],50):.2f} "
                  f"p90={np.percentile([e['miss_h'] for e in m],90):.2f}  "
                  f"|miss_v| p50={np.percentile([abs(e['miss_v']) for e in m],50):.2f} "
                  f"p90={np.percentile([abs(e['miss_v']) for e in m],90):.2f}")
    for lo, hi in ((0, 1.0), (1.0, 2.0), (2.0, 3.5), (3.5, 99)):
        m = [e for e in P if lo <= e["s_end"] < hi]
        if len(m) > 20:
            print(f"  s_end (blindness) [{lo},{hi}): n={len(m):4d} "
                  f"miss_h p50={np.percentile([e['miss_h'] for e in m],50):.2f} "
                  f"p90={np.percentile([e['miss_h'] for e in m],90):.2f}  "
                  f"|miss_v| p90={np.percentile([abs(e['miss_v']) for e in m],90):.2f}")


if __name__ == "__main__":
    main()
