import collections
import os
import pickle

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
C = pickle.load(open(os.path.join(HERE, "census.pkl"), "rb"))
EV = pickle.load(open(os.path.join(HERE, "events2.pkl"), "rb"))

LIN = ["v1", "vpef", "vtrackA", "v15", "v16", "v18", "v19", "v20"]
ORDER = ["launch_dive", "vertical_undershoot", "vertical_overshoot", "lateral_diverge",
         "corner_both", "at_gate_axis_undetermined", "gate_blackout", "obstacle_leg",
         "other", "excluded_aim_probe"]

cr = [r for r in C if r["final_state"] == "CRASH"]
census = [r for r in cr if r["mode"] != "excluded_aim_probe"]
print(f"sessions total={len(C)}  CRASH={len(cr)}  "
      f"aim-probe excluded={sum(1 for r in cr if r['mode']=='excluded_aim_probe')}  census N={len(census)}")
print(f"non-crash ends: {collections.Counter(r['final_state'] for r in C if r['final_state']!='CRASH')}")

print("\n" + "=" * 100)
print("TABLE 1 -- CENSUS: mode x confidence x lineage   (N = %d crash sessions)" % len(census))
print("=" * 100)
cc = collections.Counter((r["mode"], r["conf"]) for r in census)
cl = collections.Counter((r["mode"], r["lineage"]) for r in census)
tot = collections.Counter(r["lineage"] for r in census)
hdr = f"{'mode':30s}{'N':>5}{'%':>7} | {'hi':>4}{'med':>5}{'low':>5} | " + "".join(f"{l:>8}" for l in LIN)
print(hdr)
print("-" * len(hdr))
for m in ORDER:
    if m == "excluded_aim_probe":
        continue
    n = sum(cc[(m, c)] for c in ("high", "med", "low"))
    if not n:
        continue
    print(f"{m:30s}{n:5d}{100*n/len(census):6.1f}% | {cc[(m,'high')]:4d}{cc[(m,'med')]:5d}{cc[(m,'low')]:5d} | "
          + "".join(f"{cl[(m,l)]:8d}" for l in LIN))
print("-" * len(hdr))
print(f"{'TOTAL':30s}{len(census):5d}{100.0:6.1f}% | " + " " * 16 + "| " + "".join(f"{tot[l]:8d}" for l in LIN))

print("\n" + "=" * 100)
print("TABLE 2 -- per-lineage SHARE of that lineage's deaths (%)")
print("=" * 100)
print(f"{'mode':30s}" + "".join(f"{l:>9}" for l in LIN))
for m in ORDER:
    if m == "excluded_aim_probe":
        continue
    n = sum(cl[(m, l)] for l in LIN)
    if not n:
        continue
    print(f"{m:30s}" + "".join(
        (f"{100*cl[(m,l)]/tot[l]:8.0f}%" if tot[l] else f"{'-':>9}") for l in LIN))
print(f"{'n (deaths)':30s}" + "".join(f"{tot[l]:9d}" for l in LIN))

print("\n" + "=" * 100)
print("TABLE 3 -- SEVERITY / TAIL per mode (median and worst-case, per doctrine)")
print("=" * 100)


def pv(rs, k, f=abs):
    a = np.array([f(r[k]) for r in rs if isinstance(r.get(k), float) and np.isfinite(r[k])])
    return a


print(f"{'mode':30s}{'n':>4} {'t_death p50':>12}{'p95':>7} | {'kill-axis magnitude (m)':>28} | {'blind_m p50':>12}{'p90':>7}")
for m in ORDER:
    rs = [r for r in census if r["mode"] == m]
    if not rs:
        continue
    td = pv(rs, "t_end", float)
    bl = pv(rs, "s_end", float)
    if m in ("vertical_undershoot", "vertical_overshoot"):
        mag = pv(rs, "y_v")
        lab = "|vertical offset|"
    elif m in ("lateral_diverge", "corner_both"):
        mag = pv(rs, "y_h")
        lab = "lateral offset"
    elif m in ("gate_blackout", "obstacle_leg", "other"):
        mag = pv(rs, "d_death")
        lab = "distance from gate"
    else:
        mag = pv(rs, "y_h")
        lab = "lateral (unresolved)"
    ms = (f"{lab} p50={np.median(mag):.2f} p90={np.percentile(mag,90):.2f}" if mag.size else "-")
    print(f"{m:30s}{len(rs):4d} {np.median(td):11.1f}s{np.percentile(td,95):6.1f}s | {ms:>28} | "
          + (f"{np.median(bl):11.2f}{np.percentile(bl,90):7.2f}" if bl.size else " " * 18))

print("\n" + "=" * 100)
print("TABLE 4 -- TERMINAL BLINDNESS: how much of the census the instrument cannot see")
print("=" * 100)
atg = [r for r in census if r["mode"] in ("vertical_undershoot", "vertical_overshoot",
                                          "lateral_diverge", "corner_both",
                                          "at_gate_axis_undetermined")]
se = pv(atg, "s_end", float)
tb = np.array([r["s_end"] / max(r.get("closure", 1e-6), 1e-6) for r in atg
               if isinstance(r.get("s_end"), float) and np.isfinite(r.get("s_end", np.nan))
               and np.isfinite(r.get("closure", np.nan)) and r["closure"] > 0.5])
print(f"  at-gate deaths: {len(atg)}  ({100*len(atg)/len(census):.0f}% of the census)")
print(f"  distance from the LAST OBSERVABLE STATE to the gate plane: "
      f"p25={np.percentile(se,25):.2f} p50={np.median(se):.2f} p75={np.percentile(se,75):.2f} "
      f"p90={np.percentile(se,90):.2f} m")
print(f"  as TIME unobserved: p50={np.median(tb):.3f} s  p90={np.percentile(tb,90):.3f} s")
for th in (0.5, 1.0, 1.5, 2.0):
    print(f"    log ends >{th} m short of the plane: {100*np.mean(se>th):5.1f}% of at-gate deaths")
print(f"  -> {sum(1 for r in atg if r['mode']=='at_gate_axis_undetermined')} of {len(atg)} at-gate deaths "
      f"({100*sum(1 for r in atg if r['mode']=='at_gate_axis_undetermined')/len(atg):.0f}%) are NOT resolvable to an axis.")
inside = [r for r in atg if r["mode"] == "at_gate_axis_undetermined"
          and isinstance(r.get("y_h"), float) and np.isfinite(r.get("y_h", np.nan))
          and r["y_h"] <= 1.0 and abs(r.get("y_v", 9)) <= 0.75]
print(f"  -> {len(inside)} of them read INSIDE the 1.5 m opening at the last observable moment "
      f"(the kill is entirely inside the blind window).")

print("\n" + "=" * 100)
print("TABLE 5 -- LATERAL BRACKET (conservative vs nominal threshold)")
print("=" * 100)
lat_c = [r for r in census if r["mode"] == "lateral_diverge"]
lat_n = [r for r in census if r["mode"] == "at_gate_axis_undetermined" and r.get("alt") == "lateral_diverge"]
cb = [r for r in census if r["mode"] == "corner_both"]
print(f"  lateral, CONSERVATIVE (y_h>1.50 m, 2.9% FP on confirmed passes): {len(lat_c)}")
print(f"  + the 1.0-1.5 m band (14.8% FP -- not separable):                {len(lat_n)}")
print(f"  + corner (both axes out):                                        {len(cb)}")
print(f"  => LATERAL FAMILY BRACKET: {len(lat_c)} .. {len(lat_c)+len(lat_n)+len(cb)}  "
      f"({100*len(lat_c)/len(census):.0f}% .. {100*(len(lat_c)+len(lat_n)+len(cb))/len(census):.0f}% of deaths)")
vu = [r for r in census if r["mode"] == "vertical_undershoot"]
vo = [r for r in census if r["mode"] == "vertical_overshoot"]
print(f"  VERTICAL FAMILY: undershoot {len(vu)} + overshoot {len(vo)} (+ up to {len(cb)} corner) "
      f"=> {len(vu)+len(vo)} .. {len(vu)+len(vo)+len(cb)}")

print("\n" + "=" * 100)
print("TABLE 6 -- 'other' and mid-leg deaths, by gate (is there more than one obstacle?)")
print("=" * 100)
oth = [r for r in census if r["mode"] in ("other", "gate_blackout", "obstacle_leg")]
byg = collections.Counter((r["gi_end"], r["mode"]) for r in oth)
print(f"{'gate':>5}{'other':>10}{'blackout':>10}{'obstacle':>10}   median d_death (other)")
for g in sorted({k[0] for k in byg}):
    dd = [r["d_death"] for r in oth if r["gi_end"] == g and r["mode"] == "other"
          and isinstance(r.get("d_death"), float) and np.isfinite(r["d_death"])]
    print(f"{g:5d}{byg[(g,'other')]:10d}{byg[(g,'gate_blackout')]:10d}{byg[(g,'obstacle_leg')]:10d}"
          f"   {np.median(dd):.1f} m (n={len(dd)})" if dd else
          f"{g:5d}{byg[(g,'other')]:10d}{byg[(g,'gate_blackout')]:10d}{byg[(g,'obstacle_leg')]:10d}   -")

print("\n" + "=" * 100)
print("TABLE 7 -- how far each lineage gets (progression), and deaths per gate reached")
print("=" * 100)
print(f"{'lineage':10s}{'n':>5}{'med max_gate':>14}{'p90':>6}{'med t_death':>13}  launch%  atgate%  midleg%")
for l in LIN:
    rs = [r for r in census if r["lineage"] == l]
    if not rs:
        continue
    mg = np.array([r["max_gate"] for r in rs])
    td = np.array([r["t_end"] for r in rs])
    lau = np.mean([r["mode"] == "launch_dive" for r in rs])
    atg_ = np.mean([r["mode"] in ("vertical_undershoot", "vertical_overshoot", "lateral_diverge",
                                  "corner_both", "at_gate_axis_undetermined") for r in rs])
    mid = np.mean([r["mode"] in ("gate_blackout", "obstacle_leg", "other") for r in rs])
    print(f"{l:10s}{len(rs):5d}{np.median(mg):14.1f}{np.percentile(mg,90):6.1f}{np.median(td):12.1f}s"
          f"{100*lau:9.0f}{100*atg_:9.0f}{100*mid:9.0f}")
