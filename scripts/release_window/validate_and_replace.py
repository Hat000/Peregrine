"""(A) VALIDATE my levelling instrument the way the profile validated its own:
    on CONFIRMED GATE PASSES, where the true miss must be |y|<0.75 and |z|<0.75 m
    (a RACE_STATUS advance certifies the drone went through without contact).
    If my levelled read calls a large fraction of confirmed passes a 'strike',
    the instrument is wrong and every number above it is void.

(B) TEST THE REPLACEMENT HYPOTHESIS: the gate-0 deaths close at p50 7.4 m/s with
    the estimate reading INSIDE the opening. Does APPROACH SPEED INTO GATE 0
    discriminate death from pass? That would be a training-side lever
    (speed discipline / overspeed) rather than a release-pitch one.
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
    r, p = -roll, -pitch
    x, y, z = rel_flu
    cp, sp = math.cos(p), math.sin(p)
    x1 = cp * x + sp * z
    z1 = -sp * x + cp * z
    cr, sr = math.cos(r), math.sin(r)
    y2 = cr * y - sr * z1
    z2 = sr * y + cr * z1
    return x1, y2, z2


pass_lat, pass_vert, pass_raw_vert, pass_pitch, pass_rng = [], [], [], [], []
pass_speed_g0, death_speed_g0 = [], []
per_lin = {}

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
    lin = lineage(meta.get("label"))
    maxgate = max([t.get("gate_index", 0) or 0 for t in ticks] + [meta.get("gate_index", 0) or 0])

    # ---- (A) confirmed passes: the tick just BEFORE gate_index increments ----
    for i in range(1, len(ticks)):
        gi, gp = ticks[i].get("gate_index", 0), ticks[i - 1].get("gate_index", 0)
        if gi is not None and gp is not None and gi == gp + 1:
            t = ticks[i - 1]
            rf, o = t.get("rel_flu"), t.get("obs")
            if not rf or not o:
                continue
            rng = float(np.linalg.norm(rf))
            if rng > 3.0:            # only near-plane reads are a miss measurement
                continue
            lx, ly, lz = level_rel(rf, float(o[3]), float(o[4]))
            pass_lat.append(abs(ly)); pass_vert.append(abs(lz))
            pass_raw_vert.append(abs(rf[2]))
            pass_pitch.append(-float(o[4]))
            pass_rng.append(rng)

    # ---- (B) speed on the approach INTO gate 0 ----
    # closing speed over the 0.5 s window ending at the last tick before the gate-0
    # outcome (either the advance, or death).
    idx_end = None
    for i in range(1, len(ticks)):
        gi, gp = ticks[i].get("gate_index", 0), ticks[i - 1].get("gate_index", 0)
        if gi is not None and gp is not None and gp == 0 and gi == 1:
            idx_end = i - 1
            break
    outcome = "pass" if idx_end is not None else None
    if idx_end is None and meta.get("final_state") == "CRASH" and maxgate == 0:
        idx_end, outcome = len(ticks) - 1, "die"
    if outcome is None:
        continue
    w = [t for t in ticks[max(0, idx_end - 15):idx_end + 1] if t.get("rel_flu")]
    if len(w) < 6:
        continue
    r0 = float(np.linalg.norm(w[0]["rel_flu"]))
    r1 = float(np.linalg.norm(w[-1]["rel_flu"]))
    spd = (r0 - r1) / max(1e-6, (len(w) - 1) * DT)
    if not np.isfinite(spd):
        continue
    (pass_speed_g0 if outcome == "pass" else death_speed_g0).append(spd)
    per_lin.setdefault(lin, {"pass": [], "die": []})[outcome].append(spd)


def q(a, name, unit=""):
    a = np.array([x for x in a if np.isfinite(x)], float)
    if a.size == 0:
        return f"{name}: n=0"
    return (f"{name}: n={a.size:4d}  p10={np.percentile(a,10):6.2f} p50={np.percentile(a,50):6.2f} "
            f"p90={np.percentile(a,90):6.2f} p99={np.percentile(a,99):6.2f} max={a.max():6.2f} {unit}")


print("=" * 78)
print("(A) INSTRUMENT VALIDATION on CONFIRMED GATE PASSES")
print("    A pass certifies |lateral| < 0.75 m AND |vertical| < 0.75 m.")
print("=" * 78)
pl, pv, prv = np.array(pass_lat), np.array(pass_vert), np.array(pass_raw_vert)
print(f"  confirmed passes measured within 3 m of the plane: n = {pl.size}")
print(q(pl,  "  |lateral| LEVELLED "), "m")
print(q(pv,  "  |vertical| LEVELLED"), "m")
print(q(prv, "  |vertical| RAW body "), "m   <- the pitch-coupled control")
print()
print(f"  FALSE-POSITIVE RATE (a confirmed PASS read as a strike):")
print(f"    levelled |vert| > 0.75 m : {100*(pv>0.75).mean():5.1f}%   "
      f"(profile's own instrument: 6.3%)")
print(f"    RAW body |vert| > 0.75 m : {100*(prv>0.75).mean():5.1f}%   "
      f"(profile's own control: 66.2%)")
print(f"    levelled |lat|  > 0.75 m : {100*(pl>0.75).mean():5.1f}%")
if len(pass_pitch) > 10:
    c_lev = np.corrcoef(pv, np.array(pass_rng) * np.sin(np.array(pass_pitch)))[0, 1]
    c_raw = np.corrcoef(prv, np.array(pass_rng) * np.sin(np.array(pass_pitch)))[0, 1]
    print(f"    corr(levelled vert, range*sin(pitch)) = {c_lev:+.3f}  (want ~0: no pitch coupling)")
    print(f"    corr(RAW      vert, range*sin(pitch)) = {c_raw:+.3f}  (the coupling it removes)")

print()
print("=" * 78)
print("(B) REPLACEMENT HYPOTHESIS: approach SPEED into gate 0")
print("=" * 78)
print(q(pass_speed_g0,  "  PASSED gate 0", "m/s"))
print(q(death_speed_g0, "  DIED at gate 0", "m/s"))
a, b = np.array(pass_speed_g0), np.array(death_speed_g0)
if a.size and b.size:
    print(f"\n  median gap: died {np.median(b):.2f} m/s vs passed {np.median(a):.2f} m/s "
          f"({np.median(b)-np.median(a):+.2f})")
    # simple rank AUC
    from itertools import product
    allv = np.concatenate([a, b])
    ranks = allv.argsort().argsort().astype(float)
    ra = ranks[:a.size].sum()
    auc = (ra - a.size * (a.size - 1) / 2) / (a.size * b.size)
    print(f"  AUC(speed separates die-from-pass) = {max(auc,1-auc):.3f}  (0.5 = no separation)")
    for thr in (6, 8, 10, 12):
        pa = 100 * (a > thr).mean()
        pb = 100 * (b > thr).mean()
        print(f"    fraction closing faster than {thr:2d} m/s:  passed {pa:5.1f}%   died {pb:5.1f}%")

print("\n  per-lineage (median closing speed into gate 0):")
print(f"  {'lin':10s} {'nPass':>6s} {'nDie':>5s} {'pass p50':>9s} {'die p50':>8s} {'die p90':>8s}")
for L in sorted(per_lin):
    p, dd = per_lin[L]["pass"], per_lin[L]["die"]
    if len(p) + len(dd) < 10:
        continue
    f = lambda s, fn: (fn(s) if len(s) else float('nan'))
    print(f"  {L:10s} {len(p):6d} {len(dd):5d} {f(p,np.median):9.2f} "
          f"{f(dd,np.median):8.2f} {f(dd,lambda x: np.percentile(x,90)):8.2f}")
