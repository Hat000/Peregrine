"""Does the WIRE feed a FILLED obs[11:16] through the final blind metres, where TRAINING feeds zeros?

FIRST ATTEMPT WAS WRONG and its own number caught it: walking back from the advance to the last
`pose_seen` tick gave a median last-fix range of 7.88 m against a banked 1.78 m.  That is the
ADVANCE SEAM -- by the tick before an advance the seeker has often re-locked the NEXT gate, so
`rel_flu` is that gate's lever.  `gate_index` does not guard this (it is still g at i-1).

GUARD USED HERE, the banked discriminator: reject a consecutive-fix lever jump > 1.5 m.  A DISTANCE
gate, not a speed gate (leak/reject 0.14%/3.72% vs 1.45%/9.76% for a 20 m/s speed gate).  The
aim-offset RELEASE tick is excluded -- its 3 m step trips the same guard by construction.

Validation: this must REPRODUCE the independently-measured 1.78 m median last-sighted range.
If it does not, the instrument is wrong again and nothing below it should be believed.
"""
import os as _os
# Corpus root. Defaults to the in-repo flight corpus; override for an external snapshot:
#   PEREGRINE_RUNS=/path/to/runs python3 <this script>
_RUNS = _os.environ.get("PEREGRINE_RUNS", "data/runs")

import json, glob, os
import numpy as np

ROOT = _RUNS
JUMP = 1.5
last_rngs, blind_t, filled_frac, filled_any, dhs = [], [], [], [], []

for d in sorted(glob.glob(os.path.join(ROOT, "*"))):
    op, mp = os.path.join(d, "ego_obs.jsonl"), os.path.join(d, "meta.json")
    if not (os.path.exists(op) and os.path.exists(mp)):
        continue
    try:
        m = json.load(open(mp, encoding="utf-8"))
        recs = [json.loads(l) for l in open(op, encoding="utf-8") if l.strip()]
    except Exception:
        continue
    if len(recs) < 20:
        continue
    dh = float(m.get("ego_det_hold", 0.2) or 0.2)
    t = np.array([r["sim_time_ns"] for r in recs], dtype=float) / 1e9
    gi = [r.get("gate_index") for r in recs]

    adv = [i for i in range(1, len(recs))
           if gi[i-1] is not None and gi[i] == gi[i-1] + 1]
    for i in adv:
        g = gi[i-1]
        s = i - 1
        while s > 0 and gi[s-1] == g:
            s -= 1
        # fixes in this approach, seam-guarded
        prev_r, last_ok = None, None
        for j in range(s, i):
            if not recs[j].get("pose_seen"):
                continue
            rel = recs[j].get("rel_flu")
            if not rel:
                continue
            if recs[j].get("aim_release_tick"):      # the 3 m offset step trips the guard
                continue
            r = float(np.linalg.norm(rel))
            if prev_r is not None and abs(r - prev_r) > JUMP and r > prev_r:
                break                                 # seam: lever left this gate
            prev_r, last_ok = r, j
        if last_ok is None:
            continue
        last_rngs.append(prev_r)
        blind_t.append(float(t[i-1] - t[last_ok]))
        dhs.append(dh)
        seg = range(last_ok + 1, i)
        vals = []
        for k in seg:
            o = recs[k].get("obs")
            if o is not None and len(o) >= 16:
                vals.append(float(np.abs(np.asarray(o[11:14], dtype=float)).sum()) > 1e-6)
        filled_frac.append(float(np.mean(vals)) if vals else np.nan)
        filled_any.append(bool(np.any(vals)) if vals else False)

R, B = np.array(last_rngs), np.array(blind_t)
FF = np.array(filled_frac, dtype=float); DH = np.array(dhs)
ok = ~np.isnan(FF)
print("seam-guarded confirmed approaches: n = %d" % len(R))
print()
print("VALIDATION -- must reproduce the banked 1.78 m:")
print("   last-sighted range   MEDIAN %.2f m   (p10 %.2f  p25 %.2f  p75 %.2f)"
      % (np.median(R), *np.percentile(R, [10, 25, 75])))
print("   fixes inside 1.19 m: %d of %d" % (int((R < 1.19).sum()), len(R)))
print()
print("THE BLIND RUN-IN:")
print("   blind time to the plane  MEDIAN %.3f s  (p75 %.3f  p90 %.3f)" % (np.median(B), *np.percentile(B,[75,90])))
print("   det_hold in force        MEDIAN %.2f s" % np.median(DH))
print("   blind_time < det_hold => the mask NEVER fires before the plane:  %.1f%%" % (100*np.mean(B < DH)))
print()
print("WHAT THE POLICY IS FED IN THOSE BLIND TICKS (training would feed ZEROS):")
print("   slot0 NON-ZERO for the whole blind run-in: %.1f%% of approaches" % (100*np.mean(FF[ok] == 1.0)))
print("   slot0 non-zero for ANY of it:              %.1f%%" % (100*np.mean(np.array(filled_any)[ok])))
print("   mean filled fraction of the blind segment: %.3f" % np.nanmean(FF[ok]))
