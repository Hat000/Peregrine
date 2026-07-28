"""WHERE does the aim offset actually fly the drone high? Altitude profile vs range, dodge vs not.

Pilot reports "the offset flies a little high". Two different knobs fix that depending on where it
happens: ego_aim_release (offset comes OFF below this range) vs ego_aim_fade (offset RAMPS DOWN
over [release, release+fade] instead of holding full then snapping off).
Offset is ACTIVE ABOVE release and OFF below it (ego_obs.py:797 `if rng <= rel_m: return None`).

MUST de-inject: logged rel_flu INCLUDES the offset -> rel_true = rel_logged + [0, +lat, -vert].
MUST level: true roll = -obs[3], true pitch = -obs[4]; R = Ry(pitch) @ Rx(roll); q = R @ rel_flu.
q[2] = gate height relative to drone (UP +). q[2] < 0 => gate is BELOW => THE DRONE IS HIGH.
Stratified to ONE cell (v19Ws0, rate 30, clamp 20, z_bias 0.3) -- pooling across checkpoints faked a
+3.13 m effect here that was -0.45 within-stratum.
"""
import os as _os
# Corpus root. Defaults to the in-repo flight corpus; override for an external snapshot:
#   PEREGRINE_RUNS=/path/to/runs python3 <this script>
_RUNS = _os.environ.get("PEREGRINE_RUNS", "data/runs")

import json, glob, os
import numpy as np

ROOT = _RUNS
BINS = [(20,18),(18,16),(16,14),(14,12),(12,10),(10,8),(8,6),(6,4),(4,2),(2,0)]

def lev(rel, roll_v, pitch_v):
    r, p = -float(roll_v), -float(pitch_v)          # virtual-flip -> TRUE
    Rx = np.array([[1,0,0],[0,np.cos(r),-np.sin(r)],[0,np.sin(r),np.cos(r)]])
    Ry = np.array([[np.cos(p),0,np.sin(p)],[0,1,0],[-np.sin(p),0,np.cos(p)]])
    return (Ry @ Rx) @ np.asarray(rel, dtype=float)

def dodge_at(offs, g):
    for grp in str(offs).replace(" ", ";").split(";"):
        if ":" not in grp: continue
        gi, _, v = grp.partition(":")
        try:
            if int(gi) == g and abs(float(v.split(",")[1])) > 0.5: return float(v.split(",")[1])
        except Exception: pass
    return None

prof = {True: {b: [] for b in BINS}, False: {b: [] for b in BINS}}
nfl = {True: 0, False: 0}

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
    armed = any(dodge_at(offs, g) for g in (4,5))
    nfl[armed] += 1
    for r in recs:
        g = r.get("gate_index")
        if g not in (4,5): continue
        rel, o = r.get("rel_flu"), r.get("obs")
        if not rel or not o or len(o) < 6: continue
        vert = dodge_at(offs, g) or 0.0
        aim = r.get("aim_off")
        applied = vert if (aim is not None and armed) else 0.0
        rel_true = [float(rel[0]), float(rel[1]), float(rel[2]) - applied]
        rng = float(np.linalg.norm(rel_true))
        if not np.isfinite(rng) or rng > 20 or rng < 0.3: continue
        q = lev(rel_true, o[3], o[4])
        for b in BINS:
            if b[1] <= rng < b[0]:
                prof[armed][b].append(float(q[2])); break

print("Matched cell: v19Ws0 / rate 30 / clamp 20 / z_bias 0.30.  Gates 4-5 approaches only.")
print("flights: DODGE ARMED %d   NOT ARMED %d" % (nfl[True], nfl[False]))
print("q[2] = gate height rel. drone, offset REMOVED, attitude LEVELLED.  MORE NEGATIVE = DRONE HIGHER.")
print()
print("  range bin      ARMED  (n)        BARE  (n)     delta   <-- + = armed flies HIGHER")
for b in BINS:
    a, c = np.array(prof[True][b]), np.array(prof[False][b])
    if len(a) < 8 or len(c) < 8: continue
    ma, mc = float(np.median(a)), float(np.median(c))
    star = ""
    if abs(ma-mc) > 0.30: star = "  <<<"
    print("  %5.1f-%-5.1f m  %+6.2f (%4d)   %+6.2f (%4d)   %+6.2f%s"
          % (b[0], b[1], ma, len(a), mc, len(c), -(ma-mc), star))
print()
print("READ: offset is ON above release (12.0 m) and OFF below it. Bins above 12 m are the DOSED")
print("      stretch; bins below 12 m are post-release and should show NO armed-vs-bare difference.")
