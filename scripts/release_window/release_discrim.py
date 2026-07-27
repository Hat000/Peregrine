"""DECISIVE TEST: does the release-window nose-down COMMAND discriminate
early deaths (the 'launch/release' bucket) from flights that pass gate 0?

If deaths and survivors command the SAME dive, the command is a MARKER, not the
kill mechanism, and a reward that prices nose-down is aimed at the wrong thing.

Reads the 573-session corpus READ-ONLY.
"""
import json, os, sys, math
import numpy as np

RUNS = r"C:\Users\Fengy\Downloads\Projects\wt-arrest\data\runs"
LAUNCH_T = 2.5          # classifier's cap
DT = 1.0 / 30.0


def lineage(label):
    l = (label or "").lower()
    for tag in ("v20", "v19", "v18", "v16", "v15", "vtracka", "vpef", "v1"):
        if tag in l:
            return tag
    return "other"


def load(sess):
    d = os.path.join(RUNS, sess)
    mp, lp = os.path.join(d, "meta.json"), os.path.join(d, "ego_obs.jsonl")
    if not (os.path.exists(mp) and os.path.exists(lp)):
        return None
    try:
        meta = json.load(open(mp))
    except Exception:
        return None
    ticks = []
    with open(lp, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ticks.append(json.loads(line))
            except Exception:
                pass
    if len(ticks) < 8:
        return None
    return meta, ticks


def release_index(ticks):
    """Index of the first tick AFTER the assist releases.
    Logs without an 'assist' field (older) start already released -> 0."""
    has = any("assist" in t for t in ticks[:50])
    if not has:
        return 0
    rel = None
    seen_true = False
    for i, t in enumerate(ticks):
        a = t.get("assist", False)
        if a:
            seen_true = True
        elif seen_true:
            rel = i
            break
    if rel is None:
        return 0            # assist never engaged -> released from tick 0
    return rel


def tsec(ticks, i):
    t0 = ticks[0].get("sim_time_ns")
    ti = ticks[i].get("sim_time_ns")
    if t0 is None or ti is None:
        return i * DT
    return (ti - t0) / 1e9


rows = []
for sess in sorted(os.listdir(RUNS)):
    got = load(sess)
    if got is None:
        continue
    meta, ticks = got
    rel = release_index(ticks)
    dur = tsec(ticks, len(ticks) - 1)
    # time measured FROM RELEASE (this is the honest clock for a launch-window claim)
    dur_rel = dur - tsec(ticks, rel)
    maxgate = max([t.get("gate_index", 0) or 0 for t in ticks] + [meta.get("gate_index", 0) or 0])
    fs = meta.get("final_state", "?")

    def frd1(t):
        r = t.get("rate_frd")
        return float(r[1]) if r and len(r) > 2 else np.nan

    # --- release window, two definitions ---
    w4 = ticks[rel:rel + 5]                       # the replay instrument's window (+4 ticks)
    n12 = int(round(1.2 / DT))
    w12 = ticks[rel:rel + n12 + 1]                # first 1.2 s (the profile's window)

    p4 = np.array([frd1(t) for t in w4], float)
    p12 = np.array([frd1(t) for t in w12], float)
    p4 = p4[np.isfinite(p4)]
    p12 = p12[np.isfinite(p12)]
    if p4.size == 0 or p12.size == 0:
        continue

    # realized vertical motion over the first 1.2 s, from rel_flu (TRUE body FLU, unflipped).
    # Use the gate-relative vertical lever change as a proxy for climb/descend when available.
    vz = np.nan
    relz = [t["rel_flu"][2] for t in w12 if t.get("rel_flu")]
    if len(relz) >= 6:
        # gate is world-fixed; drone climbing => rel_flu[2] (gate above drone) shrinks
        vz = -(relz[-1] - relz[0]) / max(1e-6, (len(relz) - 1) * DT)

    fenced = float(np.mean(np.abs(p12) < 1e-9))

    rows.append(dict(
        sess=sess, lin=lineage(meta.get("label")), fs=fs, maxgate=maxgate,
        dur=dur, dur_rel=dur_rel, rel=rel,
        min4=float(p4.min()), min12=float(p12.min()), mean12=float(p12.mean()),
        vz=vz, fenced=fenced, nt=len(ticks),
        coll=meta.get("collisions", 0),
    ))

print(f"sessions parsed: {len(rows)}")

crash = [r for r in rows if r["fs"] == "CRASH"]
# the classifier's bucket, reproduced
early = [r for r in crash if r["dur"] <= LAUNCH_T and r["maxgate"] == 0]
passed = [r for r in rows if r["maxgate"] >= 1]
late_g0 = [r for r in crash if r["dur"] > LAUNCH_T and r["maxgate"] == 0]

print(f"CRASH={len(crash)}  early(t<=2.5,g0)={len(early)}  passed g0={len(passed)}  late-at-g0={len(late_g0)}")


def q(a, name):
    a = np.array([x for x in a if np.isfinite(x)], float)
    if a.size == 0:
        return f"{name}: n=0"
    return (f"{name}: n={a.size:4d}  p10={np.percentile(a,10):+.3f} p50={np.percentile(a,50):+.3f} "
            f"p90={np.percentile(a,90):+.3f}  MIN={a.min():+.3f}  mean={a.mean():+.3f}")


print("\n=== WORST (most nose-DOWN) COMMANDED PITCH RATE, first 4 ticks after release ===")
print(q([r["min4"] for r in early],  "  DIED EARLY (launch bucket)"))
print(q([r["min4"] for r in passed], "  PASSED GATE 0+       "))
print(q([r["min4"] for r in late_g0],"  died at g0 but LATE  "))

print("\n=== WORST nose-DOWN over the first 1.2 s after release ===")
print(q([r["min12"] for r in early],  "  DIED EARLY"))
print(q([r["min12"] for r in passed], "  PASSED G0 "))

print("\n=== MEAN commanded pitch rate over first 1.2 s ===")
print(q([r["mean12"] for r in early],  "  DIED EARLY"))
print(q([r["mean12"] for r in passed], "  PASSED G0 "))

print("\n=== REALIZED vertical speed over first 1.2 s (+ = climbing), from rel_flu ===")
print(q([r["vz"] for r in early],  "  DIED EARLY"))
print(q([r["vz"] for r in passed], "  PASSED G0 "))

print("\n=== fraction of first-1.2s ticks PINNED to 0.000 (fence) ===")
print(q([r["fenced"] for r in early],  "  DIED EARLY"))
print(q([r["fenced"] for r in passed], "  PASSED G0 "))

print("\n=== per-lineage: min4, DIED EARLY vs PASSED ===")
lins = sorted({r["lin"] for r in rows})
print(f"{'lin':10s} {'nEarly':>6s} {'nPass':>6s} {'early p50':>10s} {'pass p50':>9s} {'early MIN':>10s} {'pass MIN':>9s}")
for L in lins:
    e = [r["min4"] for r in early if r["lin"] == L]
    p = [r["min4"] for r in passed if r["lin"] == L]
    f = lambda a, fn: (fn(a) if len(a) else float('nan'))
    print(f"{L:10s} {len(e):6d} {len(p):6d} {f(e,np.median):10.3f} {f(p,np.median):9.3f} "
          f"{f(e,np.min):10.3f} {f(p,np.min):9.3f}")

print("\n=== TIME-FROM-RELEASE of the early deaths (is 2.1 s real, or the 2.5 cap?) ===")
print(q([r["dur"] for r in early], "  t_end (log clock)   "))
print(q([r["dur_rel"] for r in early], "  t_end - t_release   "))
print("  histogram of t_end for ALL g0 crashes (shows whether 2.5 is a cliff or a cut):")
allg0 = [r["dur"] for r in crash if r["maxgate"] == 0]
h, edges = np.histogram(allg0, bins=[0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 100])
for i in range(len(h)):
    print(f"    [{edges[i]:5.1f},{edges[i+1]:5.1f})  {h[i]:4d}  {'#'*int(h[i]/2)}")

json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "release_rows.json"), "w"))
print("\nrows -> release_rows.json")
