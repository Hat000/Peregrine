"""What did the drone actually HIT?  Usage: python scripts/postimpact/read_contacts.py [runs_dir]

id 1001 = GATE, id 1002 = ENVIRONMENT (racer.race_outcome.GATE_COLLISION_ID).
  The post-impact recorder captured the sim's COLLISION
stream.  Cross-reference every contact event against the flight tick nearest in sim_time, so
each impact carries its own gate_index, range and levelled geometry.
"""
import json, glob, os, math, sys
import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "runs")
ZBIAS_COS = math.cos(math.radians(20.0))


def load(p):
    out = []
    if not os.path.exists(p):
        return out
    with open(p, "r", encoding="utf-8", errors="replace") as fh:
        for ln in fh:
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


for d in sorted(glob.glob(os.path.join(ROOT, "*"))):
    name = os.path.basename(d)[9:-13]
    recs = load(os.path.join(d, "ego_obs.jsonl"))
    pi = load(os.path.join(d, "ego_postimpact.jsonl"))
    meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
    zb = float(meta.get("ego_gate_z_bias", 0.0))
    if not recs:
        continue
    t_obs = np.array([r["sim_time_ns"] for r in recs], dtype=np.float64)
    t_last = t_obs[-1]

    # gather every contact from every record shape in the post-impact file
    cons = []
    for r in pi:
        for key in ("terminal_contacts", "post_terminal_contacts", "contacts"):
            v = r.get(key)
            if isinstance(v, list):
                for c in v:
                    if isinstance(c, dict):
                        cons.append((key, c))
    print("=" * 112)
    print("%s   gmax=%d  n=%d  meta.collisions=%s  final=%s  dur=%.2fs   contacts_logged=%d"
          % (name, max(r.get("gate_index", -1) for r in recs), len(recs),
             meta.get("collisions"), meta.get("final_state"), meta.get("duration_s", 0), len(cons)))
    if not cons:
        print("   (no contact records)")
        continue
    ids = {}
    for key, c in cons:
        ids.setdefault(c.get("id"), []).append(c)
    print("   object ids hit:", {k: len(v) for k, v in sorted(ids.items(), key=lambda kv: -len(kv[1]))})

    # first + strongest contact, resolved onto the flight timeline
    cons_s = sorted(cons, key=lambda kc: kc[1].get("sim_time_ns", 0))
    def show(tag, key, c):
        ts = c.get("sim_time_ns", 0)
        dt_term = (ts - t_last) / 1e9
        j = int(np.argmin(np.abs(t_obs - ts)))
        r = recs[j]
        o = r.get("obs") or []
        geo = ""
        if r.get("rel_flu") and len(o) == 21:
            q = Rlevel(-o[3], -o[4]) @ np.array(r["rel_flu"], dtype=float)
            q[2] += zb * ZBIAS_COS
            p = -q
            geo = "  fwd%+6.2f lat%+6.2f vert%+6.2f" % (p[0], p[1], p[2])
        print("   %-9s id=%-5s impulse=%8.4f  t=%.3fs (term%+.3fs)  ->tick k=%d gate=%d dist=%s%s"
              % (tag, c.get("id"), c.get("impulse", float("nan")), ts / 1e9, dt_term,
                 r.get("k"), r.get("gate_index"), r.get("dist"), geo))

    show("FIRST", *cons_s[0])
    kmax = max(cons_s, key=lambda kc: kc[1].get("impulse", 0))
    show("STRONGEST", *kmax)
    show("LAST", *cons_s[-1])
    imp = np.array([c.get("impulse", 0.0) for _, c in cons_s])
    print("   impulse: max=%.4f p90=%.4f median=%.4f   span t=%.3f..%.3f s"
          % (imp.max(), float(np.percentile(imp, 90)), float(np.median(imp)),
             cons_s[0][1]["sim_time_ns"] / 1e9, cons_s[-1][1]["sim_time_ns"] / 1e9))
