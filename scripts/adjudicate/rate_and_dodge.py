"""ADJUDICATE TWO UNVERIFIED CLAIMS. The diagnose workflow's verify layer failed (my script bug),
so nothing in it was refuted-checked. These two are load-bearing and I check them myself.

CLAIM 1 (challenges code I already landed): "the rate 40->30 MECHANISM IS FALSE. Commanded 30
achieves 21.69 Hz; commanded 40 achieves 25.30 Hz. The fix moved the ACHIEVED rate FURTHER from the
30 Hz training dt." If true, my pilot_panel comment states a mechanism that is backwards, and a
wrong reason in a comment misleads whoever reads it next.

CLAIM 2 (the biggest positive result of the night): "gate-4 survival WITH the aim dodge 8/9 = 89%
vs WITHOUT 3/12 = 25%, Fisher p = 0.0075, with a BY-CONSTRUCTION placebo at gate 3 (where the dodge
cannot arm) showing p = 1.00."

For claim 2 the matched-cohort definition is the whole ballgame, so it is rebuilt from meta.json
here rather than trusted: same checkpoint, same rate, same pitch clamp, same z_bias, differing only
in whether an offset was armed at gate 4.
"""
import json, glob, os, math
from math import comb
import numpy as np

ROOT = "CORPUS/data/runs"


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def fisher(a, b, c, d):
    n = a + b + c + d
    r1, c1 = a + b, a + c
    pr = lambda x: comb(r1, x) * comb(n - r1, c1 - x) / comb(n, c1)
    p0 = pr(a)
    return sum(pr(x) for x in range(max(0, c1 - (n - r1)), min(r1, c1) + 1) if pr(x) <= p0 * (1 + 1e-9))


rows = []
for d in sorted(glob.glob(os.path.join(ROOT, "*"))):
    mp, op = os.path.join(d, "meta.json"), os.path.join(d, "ego_obs.jsonl")
    if not (os.path.exists(mp) and os.path.exists(op)):
        continue
    try:
        m = json.load(open(mp, encoding="utf-8"))
        recs = load(op)
    except Exception:
        continue
    if len(recs) < 5:
        continue
    t = np.array([r["sim_time_ns"] for r in recs], dtype=float) / 1e9
    reached = max(r.get("gate_index", -1) for r in recs)
    adv = set()
    for i in range(1, len(recs)):
        g0, g1 = recs[i - 1].get("gate_index"), recs[i].get("gate_index")
        if g0 is not None and g1 == g0 + 1:
            adv.add(g0)
    armed_gates = sorted({r.get("gate_index") for r in recs if r.get("aim_off") is not None})
    rows.append(dict(
        run=os.path.basename(d), ckpt=str(m.get("ego_ckpt", "")).split("/")[-1],
        cmd=float(m.get("rate_hz", 0)), ach=(len(recs) - 1) / max(t[-1] - t[0], 1e-9),
        reached=reached, passed=adv, armed=armed_gates,
        offs=str(m.get("ego_aim_offsets", "")), rel=m.get("ego_aim_release"),
        clamp=m.get("ego_pitch_clamp"), zb=m.get("ego_gate_z_bias"),
        fresh=100 * np.mean([bool(r.get("pose_seen")) for r in recs])))

print("=" * 100)
print("CLAIM 1 -- ACHIEVED loop rate vs COMMANDED. Does commanding 30 land NEARER 30 Hz than 40 does?")
print("=" * 100)
for cmd in (30.0, 40.0):
    v = np.array([r["ach"] for r in rows if r["cmd"] == cmd])
    f = np.array([r["fresh"] for r in rows if r["cmd"] == cmd])
    if len(v) < 3:
        continue
    print("  commanded %-5.0f n=%3d   achieved  p10 %5.2f  MEDIAN %5.2f  p90 %5.2f Hz"
          % (cmd, len(v), *np.percentile(v, [10, 50, 90])))
    print("                          |achieved - 30| median = %5.2f Hz   |   fresh-fix %%: median %5.1f"
          % (abs(np.median(v) - 30.0), np.median(f)))
print()
print("  VERDICT ON THE MECHANISM I WROTE INTO pilot_panel.py:")
a30 = np.median([r["ach"] for r in rows if r["cmd"] == 30.0])
a40 = np.median([r["ach"] for r in rows if r["cmd"] == 40.0])
print("    commanding 30 gives %.2f Hz (|err| %.2f);  commanding 40 gives %.2f Hz (|err| %.2f)"
      % (a30, abs(a30 - 30), a40, abs(a40 - 30)))
print("    -> commanding 30 is %s to the 30 Hz training dt."
      % ("CLOSER" if abs(a30 - 30) < abs(a40 - 30) else "FURTHER"))
f30 = np.median([r["fresh"] for r in rows if r["cmd"] == 30.0])
f40 = np.median([r["fresh"] for r in rows if r["cmd"] == 40.0])
print("    fresh-fix fraction: commanded 30 -> %.1f%%   commanded 40 -> %.1f%%" % (f30, f40))
g30 = np.mean([r["reached"] for r in rows if r["cmd"] == 30.0])
g40 = np.mean([r["reached"] for r in rows if r["cmd"] == 40.0])
print("    mean max gate:      commanded 30 -> %.3f (n=%d)  commanded 40 -> %.3f (n=%d)"
      % (g30, sum(1 for r in rows if r["cmd"] == 30.0), g40, sum(1 for r in rows if r["cmd"] == 40.0)))

print()
print("=" * 100)
print("CLAIM 2 -- the gate-4 aim dodge. Matched cohort rebuilt from meta.json, not inherited.")
print("=" * 100)
# matched: v19Ws0, commanded rate 30, pitch clamp 20, z_bias 0.30 -- differing ONLY in the dodge
M = [r for r in rows if r["ckpt"] == "v19Ws0_actor.pth" and r["cmd"] == 30.0
     and r["clamp"] == 20.0 and r["zb"] == 0.3]
print("matched cohort n=%d (v19Ws0, cmd rate 30, pitch clamp 20, z_bias 0.30)" % len(M))


def dodge_at(r, g):
    """Was a VERTICAL dodge actually armed on the approach to gate g? Read from the LOG (aim_off
    non-null on that gate), never from the config string -- release 0.0 flights carry offsets on
    every gate but with ZERO vertical, which is not a dodge."""
    if g not in r["armed"]:
        return False
    spec = r["offs"]
    for grp in spec.replace(" ", ";").split(";"):
        if not grp or ":" not in grp:
            continue
        gi, _, vals = grp.partition(":")
        try:
            if int(gi) == g and abs(float(vals.split(",")[1])) > 0.5:
                return True
        except Exception:
            pass
    return False


for g in (3, 4, 5):
    reach = [r for r in M if r["reached"] >= g or g in r["passed"]]
    with_d = [r for r in reach if dodge_at(r, g)]
    without = [r for r in reach if not dodge_at(r, g)]
    aw, bw = sum(1 for r in with_d if g in r["passed"]), len(with_d)
    ao, bo = sum(1 for r in without if g in r["passed"]), len(without)
    if bw + bo == 0:
        continue
    p = fisher(aw, bw - aw, ao, bo - ao) if bw and bo else float("nan")
    tag = "  <-- PLACEBO: the dodge cannot arm here" if g == 3 else ""
    print("  gate %d  reached by %2d   WITH dodge %d/%-2d (%s)   WITHOUT %d/%-2d (%s)   Fisher p=%.4f%s"
          % (g, len(reach), aw, bw, ("%.0f%%" % (100 * aw / bw)) if bw else "  -",
             ao, bo, ("%.0f%%" % (100 * ao / bo)) if bo else "  -", p, tag))
