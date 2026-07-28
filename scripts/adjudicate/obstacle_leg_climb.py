"""ABSOLUTE CLIMB along the two obstacle legs -- gate3->gate4 and gate4->gate5.

The pilot watches ALTITUDE, not gate-relative geometry, so measure that: altitude referenced to the
altitude AT THE PREVIOUS GATE PASS, against horizontal distance flown since that pass.
Altitude = -kf_pos_ned[2] (NED, +Z down). Estimator-derived (no baro on VQ2), but within one ~3 s leg
a RELATIVE climb is the right use of it. Absolute drift cancels by referencing to the pass.

Obstacles, measured: 14.49 m short of gate 4 on a 20.6 m leg => ~6.1 m PAST GATE 3.
                     15.12 m short of gate 5 on a 20.7 m leg => ~5.6 m PAST GATE 4.
Offset releases at 12 m from the NEXT gate => ~8.6 m / ~8.7 m past the previous gate.
Stratified to ONE cell (v19Ws0 / rate 30 / clamp 20 / z_bias 0.30).
"""
import os as _os
# Corpus root. Defaults to the in-repo flight corpus; override for an external snapshot:
#   PEREGRINE_RUNS=/path/to/runs python3 <this script>
_RUNS = _os.environ.get("PEREGRINE_RUNS", "data/runs")

import json, glob, os
import numpy as np

ROOT = _RUNS
STEPS = [(0,2),(2,4),(4,6),(6,8),(8,10),(10,12),(12,14),(14,16),(16,18),(18,21)]

def dodge_at(offs, g):
    for grp in str(offs).replace(" ", ";").split(";"):
        if ":" not in grp: continue
        gi, _, v = grp.partition(":")
        try:
            if int(gi) == g and abs(float(v.split(",")[1])) > 0.5: return True
        except Exception: pass
    return False

prof = {}
for leg in (4, 5):
    prof[leg] = {True: {s: [] for s in STEPS}, False: {s: [] for s in STEPS}}

for d in sorted(glob.glob(os.path.join(ROOT, "*"))):
    mp, op = os.path.join(d, "meta.json"), os.path.join(d, "ego_obs.jsonl")
    if not (os.path.exists(mp) and os.path.exists(op)): continue
    try:
        m = json.load(open(mp, encoding="utf-8"))
        recs = [json.loads(l) for l in open(op, encoding="utf-8") if l.strip()]
    except Exception: continue
    if str(m.get("ego_ckpt","")).split("/")[-1] != "v19Ws0_actor.pth": continue
    if float(m.get("rate_hz",0)) != 30.0 or m.get("ego_pitch_clamp") != 20.0 or m.get("ego_gate_z_bias") != 0.3: continue
    offs = m.get("ego_aim_offsets","") or ""

    gi = [r.get("gate_index") for r in recs]
    for tgt in (4, 5):                       # leg (tgt-1) -> tgt, i.e. gate_index == tgt
        start = None
        for i in range(1, len(recs)):
            if gi[i] == tgt and gi[i-1] == tgt - 1:
                start = i; break
        if start is None: continue
        p0 = recs[start].get("kf_pos_ned")
        if not p0: continue
        p0 = np.asarray(p0, dtype=float)
        armed = dodge_at(offs, tgt)
        for j in range(start, len(recs)):
            if gi[j] != tgt: break
            p = recs[j].get("kf_pos_ned")
            if not p: continue
            p = np.asarray(p, dtype=float)
            horiz = float(np.hypot(p[0]-p0[0], p[1]-p0[1]))
            climb = float(-(p[2]-p0[2]))     # +Z down => negate for altitude gain
            if not (np.isfinite(horiz) and np.isfinite(climb)): continue
            for s in STEPS:
                if s[0] <= horiz < s[1]:
                    prof[tgt][armed][s].append(climb); break

for tgt in (4, 5):
    obst = 6.1 if tgt == 4 else 5.6
    rel  = 8.6 if tgt == 4 else 8.7
    print("=" * 86)
    print("LEG gate %d -> gate %d      obstacle at ~%.1f m past gate %d      offset RELEASES at ~%.1f m"
          % (tgt-1, tgt, obst, tgt-1, rel))
    print("=" * 86)
    print("  m past gate %d    DODGE climb (n)     BARE climb (n)     EXTRA CLIMB" % (tgt-1))
    for s in STEPS:
        a, c = np.array(prof[tgt][True][s]), np.array(prof[tgt][False][s])
        if len(a) < 5 or len(c) < 5: continue
        ma, mc = float(np.median(a)), float(np.median(c))
        mark = ""
        if s[0] <= obst < s[1]: mark = "   <-- OBSTACLE"
        if s[0] <= rel  < s[1]: mark += "   <-- RELEASE"
        print("   %4.0f - %-4.0f     %+6.2f m (%3d)     %+6.2f m (%3d)     %+6.2f m%s"
              % (s[0], s[1], ma, len(a), mc, len(c), ma-mc, mark))
    print()
