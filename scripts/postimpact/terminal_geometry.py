"""Terminal geometry + post-impact recorder summary.  Usage: python scripts/postimpact/terminal_geometry.py [runs_dir]

Conventions taken VERBATIM from scripts/failure_profile/vg.py (the canonical instrument):
    true roll  = -obs[3]        true pitch = -obs[4]
    R_level    = Ry(pitch) @ Rx(roll)      body -> gravity-levelled
    q = R_level @ rel_flu ;  q[2] += z_bias*cos(20 deg)     (TRUE gate height)
    p = -q  = drone position relative to the gate, levelled frame [E-ish, N-ish, UP]
Aperture: NOT 0.75 m.  Training threads 0.75 - body_radius, r~U[0.28,0.38] -> 0.37-0.47 m.
"""
import json, glob, os, math, sys
import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "runs")
ZBIAS_COS = math.cos(math.radians(20.0))
APERTURE = 0.45          # effective half-width, mid of 0.37-0.47


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
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Ry @ Rx


print("=" * 118)
print("TERMINAL GEOMETRY  (levelled, z_bias-corrected).  p = drone MINUS gate: "
      "p[1]>0 drone LEFT of gate, p[2]>0 drone ABOVE gate")
print("=" * 118)
print("%-22s %4s %4s %7s | %7s %7s %7s | %-22s | %s" %
      ("flight", "gmx", "n", "rng", "fwd", "lat", "vert", "kill axis vs 0.45 m", "aim_off@g4/5"))

summ = []
for d in sorted(glob.glob(os.path.join(ROOT, "*"))):
    name = os.path.basename(d)[9:-13]
    recs = load(os.path.join(d, "ego_obs.jsonl"))
    meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
    zb = float(meta.get("ego_gate_z_bias", 0.0))
    if not recs:
        continue
    # last tick with a REAL rel_flu (pose_seen or non-null)
    last = None
    for r in reversed(recs):
        if r.get("rel_flu") and r.get("obs") and len(r["obs"]) == 21:
            last = r
            break
    if last is None:
        print(name, "no usable terminal tick"); continue
    o = last["obs"]
    roll, pitch = -o[3], -o[4]
    q = Rlevel(roll, pitch) @ np.array(last["rel_flu"], dtype=float)
    q[2] += zb * ZBIAS_COS
    p = -q                    # drone relative to gate
    fwd, lat, vert = p[0], p[1], p[2]
    rng = float(np.linalg.norm(last["rel_flu"]))
    ax = []
    if abs(lat) > APERTURE:
        ax.append("LATERAL %+.2f" % lat)
    if abs(vert) > APERTURE:
        ax.append("VERT %+.2f" % vert)
    verdict = " + ".join(ax) if ax else "inside aperture (both)"
    aim = [r for r in recs if r.get("aim_off") is not None]
    aimtxt = "-" if not aim else "%d ticks, off=%s, g=%s" % (
        len(aim), aim[-1]["aim_off"], sorted(set(r["gate_index"] for r in aim)))
    print("%-22s %4d %4d %7.2f | %7.2f %7.2f %7.2f | %-22s | %s" %
          (name, max(r.get("gate_index", -1) for r in recs), len(recs), rng,
           fwd, lat, vert, verdict, aimtxt))
    summ.append((name, lat, vert, rng))

print("\n" + "=" * 118)
print("POST-IMPACT RECORDER -- what it bought")
print("=" * 118)
for d in sorted(glob.glob(os.path.join(ROOT, "*"))):
    name = os.path.basename(d)[9:-13]
    pi = load(os.path.join(d, "ego_postimpact.jsonl"))
    ticks = [r for r in pi if r.get("k") is not None]
    if not ticks:
        print(name, "no post-terminal ticks"); continue
    a = np.array([r["accel_frd"] for r in ticks], dtype=float)
    g = np.array([r["gyro_frd"] for r in ticks], dtype=float)
    mag = np.linalg.norm(a, axis=1)
    j = int(np.argmax(mag))
    contacts = sum(r.get("n_contacts", 0) or 0 for r in ticks)
    print("%-22s ticks=%2d  extra=%.3fs  |a|max=%7.1f m/s^2 at t+%.3fs  a_frd=[%8.1f %7.1f %7.1f]  "
          "|w|max=%6.1f  contacts_post=%d" %
          (name, len(ticks), ticks[-1]["t_since_terminal_s"], mag[j], ticks[j]["t_since_terminal_s"],
           a[j][0], a[j][1], a[j][2], float(np.max(np.linalg.norm(g, axis=1))), contacts))
