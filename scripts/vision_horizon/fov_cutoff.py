"""IS THE 1.19 m VISION FLOOR FIELD-OF-VIEW GEOMETRY?

Measured, non-circularly, over 899 confirmed passes: NOT ONE ever got a vision fix closer than
1.19 m to the gate; median last fix 1.78 m. Zero-out-of-899 is a hard floor, i.e. a mechanism.

Camera, from src/racer/frames.py: fx=fy=320, W=640, H=360 -> HFOV 90.0 deg, **VFOV 58.7 deg**,
optical axis pitched +20 deg UP, so the body-frame elevation band is (-9.4, +49.4) deg.
The VERTICAL axis is less than two thirds of the horizontal -- if anything clips first it is
vertical.

PREDICTION IF IT IS FOV. As the drone closes, the gate's angular half-height atan(0.75/R) grows.
The gate stops fitting when  |elevation of gate centre in CAMERA frame| + atan(0.75/R) > VFOV/2.
So the last fix should land where that inequality first bites -- and the measured last-fix
elevations should pile up AT a frame edge, not sit comfortably inside it.

PREDICTION IF IT IS NOT FOV (detector scale, seeker gate, motion blur): last-fix bearings should
be unremarkable -- centred, well inside the frame -- and the cutoff would have to be explained by
range alone.

Angles are computed from `rel_flu`, which is TRUE body FLU [fwd, left, up], with the aim offset
de-injected. Camera elevation = body elevation - 20 deg (the axis looks up 20 deg).
"""
import json, glob, os, math, sys
import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else "CORPUS/data/runs"
HFOV, VFOV, AXIS = 90.0, 58.7, 20.0
GATE_HALF = 0.75


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def geom(r):
    rel = r.get("rel_flu")
    if not rel:
        return None
    a = r.get("aim_off")
    v = np.array(rel, dtype=float)
    if a is not None:
        v = v + np.array([0.0, a[0], -a[1]])
    rng = float(np.linalg.norm(v))
    if rng < 1e-6:
        return None
    horiz = math.hypot(v[0], v[1])
    el_body = math.degrees(math.atan2(v[2], horiz))
    az = math.degrees(math.atan2(v[1], v[0]))          # + = gate to the LEFT
    return rng, el_body - AXIS, az                      # elevation RELATIVE TO THE OPTICAL AXIS


last_fix, all_fresh = [], []
for d in sorted(glob.glob(os.path.join(ROOT, "*"))):
    p = os.path.join(d, "ego_obs.jsonl")
    if not os.path.exists(p):
        continue
    try:
        recs = load(p)
    except Exception:
        continue
    adv = {}
    for i in range(1, len(recs)):
        g0, g1 = recs[i - 1].get("gate_index"), recs[i].get("gate_index")
        if g0 is not None and g1 == g0 + 1:
            adv[g0] = recs[i - 1]["sim_time_ns"]
    for g, ta in adv.items():
        leg = [r for r in recs if r.get("gate_index") == g and r["sim_time_ns"] < ta
               and r.get("pose_seen")]
        gs = [x for x in (geom(r) for r in leg) if x]
        if not gs:
            continue
        all_fresh.extend(gs)
        last_fix.append(min(gs, key=lambda t: t[0]))

L = np.array(last_fix)           # rng, el_cam, az
A = np.array(all_fresh)
print("confirmed passes with a fresh fix: %d   (all fresh ticks: %d)" % (len(L), len(A)))
print("camera: HFOV %.1f (half %.1f)  VFOV %.1f (half %.1f)  axis +%.0f deg up"
      % (HFOV, HFOV / 2, VFOV, VFOV / 2, AXIS))
print()

rng, el, az = L[:, 0], L[:, 1], L[:, 2]
half_ang = np.degrees(np.arctan(GATE_HALF / rng))
print("AT THE LAST FIX")
print("  range          p10 %.2f  med %.2f  p90 %.2f m" % tuple(np.percentile(rng, [10, 50, 90])))
print("  elevation      p10 %+.1f  med %+.1f  p90 %+.1f deg   (frame edge at +/-%.1f)"
      % (*np.percentile(el, [10, 50, 90]), VFOV / 2))
print("  azimuth        p10 %+.1f  med %+.1f  p90 %+.1f deg   (frame edge at +/-%.1f)"
      % (*np.percentile(az, [10, 50, 90]), HFOV / 2))
print("  gate half-angle at that range: med %.1f deg" % np.median(half_ang))
print()
print("  DOES THE WHOLE GATE STILL FIT AT THE LAST FIX?")
v_need = np.abs(el) + half_ang
h_need = np.abs(az) + half_ang
print("    vertical   |el|+half = med %.1f deg  vs VFOV/2 %.1f  ->  OVERFLOWS on %.0f%% of passes"
      % (np.median(v_need), VFOV / 2, 100 * np.mean(v_need > VFOV / 2)))
print("    horizontal |az|+half = med %.1f deg  vs HFOV/2 %.1f  ->  OVERFLOWS on %.0f%% of passes"
      % (np.median(h_need), HFOV / 2, 100 * np.mean(h_need > HFOV / 2)))
print()
print("  ONE TICK LATER (what the drone would have needed to still see it): shrink range by the")
print("  median inter-fix closing step and recheck.")
step = 0.30
r2 = np.maximum(rng - step, 0.05)
ha2 = np.degrees(np.arctan(GATE_HALF / r2))
print("    at range-%.2f m: vertical overflows on %.0f%%, horizontal on %.0f%%"
      % (step, 100 * np.mean(np.abs(el) + ha2 > VFOV / 2), 100 * np.mean(np.abs(az) + ha2 > HFOV / 2)))
print()
print("PREDICTED FOV-LIMITED RANGE per pass (solve |el| + atan(0.75/R) = VFOV/2 for R, using each")
print("pass's own elevation) vs the range actually observed. If the two agree, it IS geometry.")
lim = np.radians(np.clip(VFOV / 2 - np.abs(el), 0.5, 89.0))
R_pred = GATE_HALF / np.tan(lim)
ok = np.isfinite(R_pred)
print("  predicted  p10 %.2f  med %.2f  p90 %.2f m" % tuple(np.percentile(R_pred[ok], [10, 50, 90])))
print("  observed   p10 %.2f  med %.2f  p90 %.2f m" % tuple(np.percentile(rng[ok], [10, 50, 90])))
print("  median(observed - predicted) = %+.2f m ;  correlation = %+.3f"
      % (float(np.median(rng[ok] - R_pred[ok])),
         float(np.corrcoef(rng[ok], R_pred[ok])[0, 1]) if ok.sum() > 2 else float("nan")))
print()
print("CONTROL: the same angles over ALL fresh ticks, not just the last one. If last-fix")
print("elevation is NOT special, the FOV story is wrong.")
print("  all-fresh  |el| med %.1f deg   last-fix |el| med %.1f deg"
      % (np.median(np.abs(A[:, 1])), np.median(np.abs(el))))
print("  all-fresh  |az| med %.1f deg   last-fix |az| med %.1f deg"
      % (np.median(np.abs(A[:, 2])), np.median(np.abs(az))))
