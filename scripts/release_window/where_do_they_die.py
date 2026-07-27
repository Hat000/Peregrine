"""WHERE do the 'launch/release window' deaths actually die?

The classifier's rule is purely temporal (t_end <= 2.5 s AND never passed gate 0)
and it is rule #1, so it PRE-EMPTS every at-gate rule. If these flights die 1-3 m
from gate 0 they are GATE-0 STRIKES mislabelled as launch dives; if they die 7-10 m
out they are genuine launch departures.

Uses the TRUE body-FLU lever rel_flu, GRAVITY-LEVELLED with the logged attitude
(obs[3:5] roll/pitch, virtual-flipped -> unflip) so the vertical read is not
pitch-coupled -- the profile's load-bearing correction.
"""
import json, os, math
import numpy as np

RUNS = r"C:\Users\Fengy\Downloads\Projects\wt-arrest\data\runs"
DT = 1.0 / 30.0


def lineage(label):
    l = (label or "").lower()
    for tag in ("v20", "v19", "v18", "v16", "v15", "vtracka", "vpef", "v1"):
        if tag in l:
            return tag
    return "other"


def level_rel(rel_flu, roll, pitch):
    """Rotate the TRUE body-FLU gate lever into a gravity-levelled heading frame.
    obs[3:5] are VIRTUAL-FLIPPED (diag(-1,-1,1)) -> true roll/pitch = -obs[3], -obs[4]."""
    r, p = -roll, -pitch
    x, y, z = rel_flu
    # undo pitch (about body y), then roll (about body x)
    cp, sp = math.cos(p), math.sin(p)
    x1 = cp * x + sp * z
    z1 = -sp * x + cp * z
    cr, sr = math.cos(r), math.sin(r)
    y2 = cr * y - sr * z1
    z2 = sr * y + cr * z1
    return x1, y2, z2


rows = []
for sess in sorted(os.listdir(RUNS)):
    d = os.path.join(RUNS, sess)
    mp, lp = os.path.join(d, "meta.json"), os.path.join(d, "ego_obs.jsonl")
    if not (os.path.exists(mp) and os.path.exists(lp)):
        continue
    try:
        meta = json.load(open(mp))
    except Exception:
        continue
    ticks = []
    with open(lp) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ticks.append(json.loads(line))
                except Exception:
                    pass
    if len(ticks) < 8:
        continue
    t0 = ticks[0].get("sim_time_ns")
    dur = (ticks[-1]["sim_time_ns"] - t0) / 1e9 if t0 else len(ticks) * DT
    maxgate = max([t.get("gate_index", 0) or 0 for t in ticks] + [meta.get("gate_index", 0) or 0])
    if meta.get("final_state") != "CRASH" or maxgate != 0:
        continue

    # last tick carrying a real gate lever
    last = None
    for t in reversed(ticks):
        if t.get("rel_flu") and t.get("pose_seen") is not None:
            last = t
            break
    if last is None:
        continue
    rf = last["rel_flu"]
    rng = float(np.linalg.norm(rf))
    o = last.get("obs") or [0] * 21
    lx, ly, lz = level_rel(rf, float(o[3]), float(o[4]))
    # forward (levelled horizontal) distance still to run to the gate plane
    fwd = math.hypot(lx, ly)

    # speed at death, from the levelled lever's closing rate over the last ~0.4 s
    tail = [t for t in ticks[-13:] if t.get("rel_flu")]
    clos = np.nan
    if len(tail) >= 5:
        r0 = float(np.linalg.norm(tail[0]["rel_flu"]))
        r1 = float(np.linalg.norm(tail[-1]["rel_flu"]))
        clos = (r0 - r1) / max(1e-6, (len(tail) - 1) * DT)

    # range at the very first tick (the release geometry)
    rf0 = ticks[0].get("rel_flu")
    rng0 = float(np.linalg.norm(rf0)) if rf0 else np.nan

    rows.append(dict(sess=sess, lin=lineage(meta.get("label")), dur=dur,
                     rng=rng, fwd=fwd, lat=ly, vert=lz, clos=clos, rng0=rng0,
                     coll=meta.get("collisions", 0)))

print(f"gate-0 CRASH flights: {len(rows)}")
early = [r for r in rows if r["dur"] <= 2.5]
late = [r for r in rows if r["dur"] > 2.5]
print(f"  classifier 'launch/release' (t<=2.5): {len(early)}   later g0 deaths: {len(late)}")


def q(a, name, unit=""):
    a = np.array([x for x in a if np.isfinite(x)], float)
    if a.size == 0:
        return f"{name}: n=0"
    return (f"{name}: n={a.size:4d}  p10={np.percentile(a,10):6.2f} p50={np.percentile(a,50):6.2f} "
            f"p90={np.percentile(a,90):6.2f}  max={a.max():6.2f} {unit}")


print("\n=== 3-D RANGE TO GATE 0 AT THE LAST LOGGED TICK ===")
print(q([r["rng"] for r in early], "  launch-bucket deaths", "m"))
print(q([r["rng"] for r in late],  "  later g0 deaths     ", "m"))
print("\n=== LEVELLED HORIZONTAL distance still to run ===")
print(q([r["fwd"] for r in early], "  launch-bucket deaths", "m"))
print("\n=== RANGE AT TICK 0 (the release geometry) ===")
print(q([r["rng0"] for r in early], "  launch-bucket deaths", "m"))
print("\n=== CLOSING SPEED at death (m/s) ===")
print(q([r["clos"] for r in early], "  launch-bucket deaths", "m/s"))

print("\n=== how many launch-bucket deaths die WITHIN N m of gate 0? ===")
rr = np.array([r["rng"] for r in early if np.isfinite(r["rng"])])
for thr in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0):
    print(f"    <= {thr:4.1f} m : {int((rr<=thr).sum()):4d} / {rr.size}  ({100*(rr<=thr).mean():5.1f}%)")

print("\n=== LEVELLED miss at the last tick, for those that died CLOSE (<=3 m) ===")
close = [r for r in early if np.isfinite(r["rng"]) and r["rng"] <= 3.0]
print(q([abs(r["lat"]) for r in close],  "  |lateral|", "m"))
print(q([abs(r["vert"]) for r in close], "  |vertical|", "m"))
print(q([r["vert"] for r in close],      "  signed vert (+=gate above)", "m"))

print("\n=== per-lineage: range at death for the launch bucket ===")
print(f"{'lin':10s} {'n':>4s} {'p50 rng':>8s} {'p90 rng':>8s} {'<=3m':>6s}")
for L in sorted({r["lin"] for r in early}):
    a = np.array([r["rng"] for r in early if r["lin"] == L and np.isfinite(r["rng"])])
    if a.size:
        print(f"{L:10s} {a.size:4d} {np.percentile(a,50):8.2f} {np.percentile(a,90):8.2f} "
              f"{100*(a<=3.0).mean():5.1f}%")

json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "g0_deaths.json"), "w"))
