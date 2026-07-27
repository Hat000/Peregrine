"""KILL-AXIS at the IMPACT, not at the last log line.

Now that the post-impact recorder emits the sim's COLLISION stream (id 1001 = GATE,
1002 = ENVIRONMENT), a death can be classified against the thing it actually hit:

  1. take the strongest id=1001 (GATE) contact of the flight
  2. walk back to the last FRESH vision fix at or before it
  3. level that lever (true roll = -obs[3], true pitch = -obs[4]; z_bias restored)
  4. classify |lat| / |vert| against the REAL half-aperture 0.45 m (0.75 - body_radius)

Guards, because the terminal lever lies in two known ways:
  * ADVANCE SEAM -- if the fix used sits beyond `RNG_MAX`, the lever has already jumped to the
    next gate; report UNUSABLE rather than a fake 15 m lateral.
  * AIM RELEASE -- aim_off must be None on the classifying tick (release is rng <= 12 m).
"""
import json, glob, os, math, sys
import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "runs")
ZBIAS_COS = math.cos(math.radians(20.0))
APERTURE = 0.45
RNG_MAX = 5.0          # beyond this the lever is not a strike geometry
GATE_ID, ENV_ID = 1001, 1002


def load(p):
    out = []
    if os.path.exists(p):
        for ln in open(p, "r", encoding="utf-8", errors="replace"):
            ln = ln.strip()
            if ln:
                try:
                    out.append(json.loads(ln))
                except Exception:
                    pass
    return out


def Rlevel(roll, pitch):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    return (np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))


print("%-8s %4s | %-14s | %6s | %7s %7s | %-28s | %s" %
      ("flight", "gate", "hit", "rng", "lat", "vert", "KILL AXIS vs 0.45 m", "note"))
print("-" * 122)
tally = {}
for d in sorted(glob.glob(os.path.join(ROOT, "*"))):
    name = os.path.basename(d)[9:15]
    recs = load(os.path.join(d, "ego_obs.jsonl"))
    pi = load(os.path.join(d, "ego_postimpact.jsonl"))
    if not recs:
        continue
    meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
    zb = float(meta.get("ego_gate_z_bias", 0.0))

    cons = []
    for r in pi:
        for key in ("terminal_contacts", "post_terminal_contacts", "contacts"):
            v = r.get(key)
            if isinstance(v, list):
                cons += [c for c in v if isinstance(c, dict)]
    gate_hits = [c for c in cons if int(c.get("id", 0)) == GATE_ID]
    env_hits = [c for c in cons if int(c.get("id", 0)) == ENV_ID]
    gi = recs[-1].get("gate_index", -1)

    if not gate_hits:
        what = "ENV only" if env_hits else "no contact"
        print("%-8s %4d | %-14s | %6s | %7s %7s | %-28s | %s" %
              (name, gi, what, "-", "-", "-", "NOT A GATE STRIKE",
               "%d env events" % len(env_hits) if env_hits else ""))
        tally[what] = tally.get(what, 0) + 1
        continue

    c = max(gate_hits, key=lambda x: x.get("impulse", 0))
    ts = c.get("sim_time_ns", 0)
    cand = [r for r in recs if r.get("pose_seen") and r.get("rel_flu")
            and r.get("obs") and len(r["obs"]) == 21 and r["sim_time_ns"] <= ts
            and r.get("gate_index") == gi]
    if not cand:
        print("%-8s %4d | %-14s | %6s | %7s %7s | %-28s | %s" %
              (name, gi, "GATE i=%.1f" % c.get("impulse", 0), "-", "-", "-",
               "UNUSABLE", "no fresh fix on the terminal gate before impact"))
        tally["unusable"] = tally.get("unusable", 0) + 1
        continue
    r = cand[-1]
    o = r["obs"]
    q = Rlevel(-o[3], -o[4]) @ np.array(r["rel_flu"], dtype=float)
    q[2] += zb * ZBIAS_COS
    p = -q
    rng = float(np.linalg.norm(r["rel_flu"]))
    note = ""
    if r.get("aim_off") is not None:
        note = "AIM ARMED on this tick -- lever offset by %s" % (r["aim_off"],)
    if rng > RNG_MAX:
        verdict = "UNUSABLE (lever past the seam)"
        tally["unusable"] = tally.get("unusable", 0) + 1
    else:
        ax = []
        if abs(p[1]) > APERTURE:
            ax.append("LATERAL")
        if abs(p[2]) > APERTURE:
            ax.append("VERTICAL")
        verdict = " + ".join(ax) if ax else "inside aperture (clipped edge?)"
        key = verdict if len(ax) != 2 else "LATERAL+VERTICAL"
        tally[key] = tally.get(key, 0) + 1
    print("%-8s %4d | GATE i=%-7.2f | %6.2f | %+7.2f %+7.2f | %-28s | %s" %
          (name, gi, c.get("impulse", 0), rng, p[1], p[2], verdict, note))

print("\nTALLY:", tally)
