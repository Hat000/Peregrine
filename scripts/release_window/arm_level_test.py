"""THE TEST THAT COULD STILL RESCUE THE HYPOTHESIS.

Within-flight the release dive does not discriminate (v19 deaths -1.379 vs survivors
-1.378). But the claim was made at the ARM level: v20 dives ~2x harder than v19 and
3/6 v20 wire flights died at gate 0. So:

  Q1. Across lineages, does a HARDER median release dive predict a HIGHER gate-0
      death rate? (If the sign is wrong, the arm-level story dies too.)
  Q2. Within the v19 lineage alone, split flights by release-dive depth: does the
      deep half die at gate 0 more than the shallow half?
  Q3. How much of the 'launch/release' bucket is really an AT-GATE death that the
      classifier's rule ordering pre-empted?
"""
import json, os, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(HERE, "release_rows.json")))
g0 = {r["sess"]: r for r in json.load(open(os.path.join(HERE, "g0_deaths.json")))}

for r in rows:
    g = g0.get(r["sess"])
    r["rng_death"] = g["rng"] if g else float("nan")

# ---------- Q1: arm-level, per lineage ----------
print("=== Q1. PER-LINEAGE: release-dive depth vs gate-0 death rate ===")
print("(dive p50 = median worst nose-down cmd in the 4 ticks after release, over ALL")
print(" flights of that lineage; a MORE NEGATIVE number is a harder dive)")
print()
print(f"{'lin':10s} {'nFlights':>8s} {'dive p50':>9s} {'dive p10':>9s} "
      f"{'died@g0':>8s} {'rate':>7s}")
lins = sorted({r["lin"] for r in rows})
tab = []
for L in lins:
    a = [r for r in rows if r["lin"] == L]
    if len(a) < 8:
        continue
    dive = np.median([r["min4"] for r in a])
    dive10 = np.percentile([r["min4"] for r in a], 10)
    died = sum(1 for r in a if r["fs"] == "CRASH" and r["maxgate"] == 0)
    rate = died / len(a)
    tab.append((L, len(a), dive, dive10, died, rate))
    print(f"{L:10s} {len(a):8d} {dive:9.3f} {dive10:9.3f} {died:8d} {100*rate:6.1f}%")

if len(tab) >= 3:
    d = np.array([t[2] for t in tab])
    rt = np.array([t[5] for t in tab])
    # Pearson on the lineage means. HARDER dive = MORE negative. Hypothesis predicts
    # corr(dive, rate) NEGATIVE (more negative dive -> higher rate).
    c = np.corrcoef(d, rt)[0, 1]
    print(f"\n  corr(median dive, gate-0 death rate) over {len(tab)} lineages = {c:+.3f}")
    print("  HYPOTHESIS PREDICTS NEGATIVE (harder dive -> more gate-0 deaths).")
    print(f"  OBSERVED: {'NEGATIVE - consistent' if c < -0.2 else 'POSITIVE/NULL - INCONSISTENT'}")

# ---------- Q2: within v19 ----------
print("\n=== Q2. WITHIN A SINGLE LINEAGE: deep-dive half vs shallow-dive half ===")
for L in ("v19", "v16", "v15"):
    a = [r for r in rows if r["lin"] == L]
    if len(a) < 20:
        continue
    med = np.median([r["min4"] for r in a])
    deep = [r for r in a if r["min4"] <= med]
    shal = [r for r in a if r["min4"] > med]
    fd = lambda s: sum(1 for r in s if r["fs"] == "CRASH" and r["maxgate"] == 0) / max(1, len(s))
    npass = lambda s: np.mean([r["maxgate"] for r in s])
    print(f"  {L}: median dive {med:+.3f}")
    print(f"     DEEPER half (dive <= {med:+.3f}): n={len(deep):3d}  "
          f"gate-0 death {100*fd(deep):5.1f}%   mean gates {npass(deep):.2f}")
    print(f"     SHALLOW half(dive >  {med:+.3f}): n={len(shal):3d}  "
          f"gate-0 death {100*fd(shal):5.1f}%   mean gates {npass(shal):.2f}")

# ---------- Q3: rule pre-emption ----------
print("\n=== Q3. HOW MUCH OF THE BUCKET IS AN AT-GATE DEATH IN DISGUISE? ===")
early = [r for r in rows if r["fs"] == "CRASH" and r["maxgate"] == 0 and r["dur"] <= 2.5]
have = [r for r in early if np.isfinite(r["rng_death"])]
print(f"  launch-bucket deaths with a measurable death range: {len(have)} / {len(early)}")
for thr in (2.0, 3.0, 3.5):
    n = sum(1 for r in have if r["rng_death"] <= thr)
    print(f"    would classify AT-GATE (range <= {thr:.1f} m, the profile's at-gate cut): "
          f"{n}/{len(have)} = {100*n/len(have):.1f}%")

print("\n=== Q4. is gate 0 special, or just the gate EVERY flight attempts? ===")
tot = len(rows)
reach = {}
for g in range(0, 7):
    reach[g] = sum(1 for r in rows if r["maxgate"] >= g)
print(f"  flights reaching each gate index (of {tot}):")
prev = None
for g in range(0, 7):
    died_here = sum(1 for r in rows if r["fs"] == "CRASH" and r["maxgate"] == g)
    surv = reach[g + 1] / reach[g] if reach.get(g) and reach.get(g + 1) else float("nan")
    print(f"    gate {g}: attempted by {reach[g]:4d}   died there {died_here:4d}   "
          f"conditional survival {100*surv:5.1f}%")
print("\n  -> if gate 0 has the BEST conditional survival yet the MOST absolute deaths,")
print("     the bucket's size is an exposure artifact, not a difficulty signal.")
