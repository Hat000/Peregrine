"""Build the pass/crash gate-event table with the v2 terminal geometry, then validate on passes."""
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


def per_run(r):
    recs, seek, timing, meta = vg.load_session(r["path"])
    d = vg.build(recs, seek, meta)
    t = d["t"]
    n = len(t)
    out = []
    adv = [i for i in range(1, n) if d["gi"][i] > d["gi"][i - 1]]
    marks = [(i, int(d["gi"][i - 1]), "pass") for i in adv]
    if r.get("final_state") == "CRASH":
        marks.append((n - 1, int(d["gi"][-1]), "crash"))
    for i_end, g, kind in marks:
        sel = terminal.approach_sel(d, g, i_end)
        if len(sel) < 5:
            out.append(dict(run=r["run"], lineage=r["lineage"], gate=g, kind=kind,
                            ok=0, n_sel=len(sel), t_end=float(t[i_end])))
            continue
        try:
            tg = terminal.terminal_geometry(d, sel, i_end)
        except Exception:
            tg = None
        if tg is None:
            out.append(dict(run=r["run"], lineage=r["lineage"], gate=g, kind=kind,
                            ok=0, n_sel=len(sel), t_end=float(t[i_end])))
            continue
        rec = dict(run=r["run"], lineage=r["lineage"], gate=g, kind=kind, ok=1,
                   n_sel=len(sel), t_end=float(t[i_end]))
        for k in ("s_end", "d_end", "y_h_now", "y_v_now", "y_h", "y_v", "ydot_h", "ydot_v",
                  "t_plane", "t_used", "capped", "dt_blind", "speed", "closure",
                  "rms_r", "rms_t", "rms_r_axis", "rms_t_axis", "last_rho", "extrap_m",
                  "n_axis", "n_near"):
            rec[k] = tg[k]
        out.append(rec)
    return out


def qs(name, a, qq=(10, 25, 50, 75, 90, 95)):
    a = np.array([x for x in a if np.isfinite(x)])
    if not a.size:
        print(f"  {name}: n=0"); return
    print(f"  {name}: n={len(a)} " + " ".join(f"p{x}={np.percentile(a,x):+.2f}" for x in qq)
          + f" max={a.max():+.2f}")


def main():
    use = [r for r in ROWS if (r.get("n_ticks") or 0) > 20 and not str(r.get("note", "")).startswith("ERR")]
    ev = []
    for i, r in enumerate(use):
        try:
            ev += per_run(r)
        except Exception as e:
            print("ERR", r["run"], e)
        if i % 120 == 0:
            print(f"  [{i}/{len(use)}]", flush=True)
    pickle.dump(ev, open(os.path.join(HERE, "events2.pkl"), "wb"))
    P = [e for e in ev if e["kind"] == "pass" and e["ok"]]
    C = [e for e in ev if e["kind"] == "crash" and e["ok"]]
    print(f"\nevents: pass={len(P)} (fit-fail {sum(1 for e in ev if e['kind']=='pass' and not e['ok'])}) "
          f"crash={len(C)} (fit-fail {sum(1 for e in ev if e['kind']=='crash' and not e['ok'])})")

    def rep(tag, S):
        yh = np.array([e["y_h"] for e in S])
        yv = np.array([e["y_v"] for e in S])
        print(f"\n--- {tag} (n={len(S)}) ---")
        qs("y_h", yh)
        qs("|y_v|", np.abs(yv))
        for th in (0.75, 1.0, 1.5):
            print(f"   y_h<={th}: {100*np.mean(yh<=th):5.1f}%   |y_v|<={th}: {100*np.mean(np.abs(yv)<=th):5.1f}%")

    print("\n================ GROUND TRUTH: confirmed PASSES (true |miss| < 0.75 m) ================")
    rep("ALL passes", P)
    # quality strata
    for tag, filt in (
            ("clean perception (rms_r<=0.30)", lambda e: e["rms_r"] <= 0.30),
            ("clean + log reaches plane (s_end<=1.2)", lambda e: e["rms_r"] <= 0.30 and e["s_end"] <= 1.2),
            ("clean + s_end<=2.0", lambda e: e["rms_r"] <= 0.30 and e["s_end"] <= 2.0),
            ("dirty perception (rms_r>0.60)", lambda e: e["rms_r"] > 0.60),
            ("far/blind (s_end>3)", lambda e: e["s_end"] > 3.0)):
        S = [e for e in P if filt(e)]
        if len(S) > 20:
            rep(tag, S)
    print("\n================ CRASH terminal events ================")
    rep("ALL crashes", C)
    S = [e for e in C if e["rms_r"] <= 0.30 and e["s_end"] <= 2.0]
    rep("clean + s_end<=2.0", S)
    print("\n=== separation on the CLEAN stratum ===")
    pp = [e for e in P if e["rms_r"] <= 0.30 and e["s_end"] <= 2.0]
    cc = [e for e in C if e["rms_r"] <= 0.30 and e["s_end"] <= 2.0]
    for nm, key, ab in (("y_h", "y_h", False), ("|y_v|", "y_v", True), ("y_v signed", "y_v", None)):
        a = np.array([abs(e[key]) if ab else e[key] for e in pp])
        b = np.array([abs(e[key]) if ab else e[key] for e in cc])
        auc = np.mean([np.mean(b > x) + 0.5 * np.mean(b == x) for x in a])
        print(f"  AUC({nm}: crash>pass) = {auc:.3f}   pass p50={np.median(a):+.2f} crash p50={np.median(b):+.2f}")


if __name__ == "__main__":
    main()
