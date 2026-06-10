"""Pool the 6 per-gate characterize_perception runs into one perception-noise model, decomposed
properly (the built-in 'modal gate' robust subset conflates adjacent-gate locks with bad fixes on a
multi-gate course). Cross-references each associated fix with the bundle's nearest_gate_id ('the gate
the drone is closest to' = the intended lock) to separate:
  * CORRECT-gate good fixes  (assoc==nearest, |fix|<3 m) -> the true noise floor: per-axis bias+std.
  * CATASTROPHIC fixes       (|fix|>=3 m)                 -> the flip/wrong-gate TAIL (magnitude).
  * WRONG-gate associations  (assoc!=nearest)             -> the association-error component.
And the KF-gate test: do catastrophic fixes carry a huge Mahalanobis (>chi2_.999=16.27) so the
filter's covariance gate would REJECT them in-loop? (=> the effective in-loop bad-fix leak rate.)
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent.parent
CHI2 = 16.27  # 3-dof, 0.999

rows = []
for g in range(6):
    bundle = json.loads((HERE / f"pg/course_g{g}/frames.json").read_text())
    near = {f["frame_id"]: f["nearest_gate_id"] for f in bundle["frames"]}
    char = json.loads((HERE / f"characterize_g{g}.json").read_text())
    for r in char["rows"]:
        if r.get("associated") and "world_fix_err_m" in r:
            r["nearest_gate_id"] = near.get(r["frame_id"])
            r["bundle_gate"] = g
            rows.append(r)

N = len(rows)
fix = np.array([r["world_fix_err_m"] for r in rows])
off = np.array([r["off_ned"] for r in rows])
maha = np.array([r["maha"] for r in rows], float)
assoc_gate = np.array([r["gate_id"] for r in rows])
near_gate = np.array([r["nearest_gate_id"] for r in rows])
true_rng = np.array([r["true_range_m"] for r in rows])

good = fix < 3.0
correct = assoc_gate == near_gate
clean = good & correct          # correct gate AND good fix = the noise floor
cata = ~good                    # |fix|>=3 m = catastrophic
wrong = ~correct                # associated to a non-nearest gate

def pct(a, q): return float(np.percentile(a, q)) if len(a) else float("nan")

accepted = good & (maha < CHI2) & np.isfinite(maha)  # what the KF would actually ingest

def noise_floor(mask, label):
    print(f"=== NOISE FLOOR [{label}]  N={mask.sum()} ({100*mask.mean():.0f}%) ===")
    for ax, nm in enumerate("N E D".split()):
        o = off[mask, ax]
        print(f"  {nm}:  bias(mean) {o.mean():+.2f} m   std {o.std():.2f} m   "
              f"[p10 {pct(o,10):+.2f}, p90 {pct(o,90):+.2f}]")
    cf = fix[mask]
    print(f"  |fix|:  p50 {pct(cf,50):.2f}  p90 {pct(cf,90):.2f}  mean {cf.mean():.2f}  max {pct(cf,100):.2f} m")

print(f"POOLED across 6 per-gate runs of course_60s: N={N} associated+solved fixes\n")
# (a) chain accuracy when it locks ANY gate correctly (|fix|<3 m) -- gate identity is NOT the error
#     metric: a good fix to the centred NEXT gate is still accurate. (b) correct-gate-only, for ref.
# (c) ACCEPTED = good fix that also passes the KF maha gate = what actually updates the filter.
noise_floor(good, "all good fixes |fix|<3 m")
noise_floor(clean, "correct-gate (assoc==nearest) & |fix|<3 m")
noise_floor(accepted, "ACCEPTED: |fix|<3 m AND maha<chi2 (KF-ingested)")
print(f"  reproj px (good): p50 {pct(np.array([r['reproj_px'] for r,c in zip(rows,good) if c]),50):.2f}\n")

print(f"\n=== TAIL / catastrophic ===")
print(f"  |fix|>=3 m (flip/wrong-gate):  {cata.sum()}/{N} = {100*cata.mean():.0f}%")
print(f"  wrong-gate (assoc!=nearest):   {wrong.sum()}/{N} = {100*wrong.mean():.0f}%")
print(f"  of the |fix|>=3 m fixes: {(wrong&cata).sum()} are wrong-gate, {(correct&cata).sum()} are "
      f"correct-gate depth-flips/large-resid")
print(f"  catastrophic |fix|: p50 {pct(fix[cata],50):.1f}  max {pct(fix[cata],100):.1f} m")

print(f"\n=== KF COVARIANCE GATE (Mahalanobis vs chi2_.999={CHI2}) ===")
finite = np.isfinite(maha)
gpass = maha < CHI2
print(f"  good fixes (|fix|<3 m) that PASS the gate (maha<{CHI2}): "
      f"{(good&gpass&finite).sum()}/{(good&finite).sum()} = "
      f"{100*(good&gpass&finite).sum()/max((good&finite).sum(),1):.0f}%  (accepted, correct)")
print(f"  catastrophic (|fix|>=3 m) that the gate REJECTS (maha>={CHI2}): "
      f"{(cata&~gpass&finite).sum()}/{(cata&finite).sum()} = "
      f"{100*(cata&~gpass&finite).sum()/max((cata&finite).sum(),1):.0f}%  (correctly rejected)")
leak = cata & gpass & finite
print(f"  LEAK: catastrophic fixes that SLIP the gate (|fix|>=3 m AND maha<{CHI2}): "
      f"{leak.sum()}/{N} = {100*leak.sum()/N:.1f}%  -> the residual bad-fix rate the RL twin must model")

print(f"\n=== by-range noise floor (correct-gate good fixes) ===")
print(f"  {'band':>8} {'n':>4} {'biasN':>6} {'biasE':>6} {'biasD':>6} {'|fix|p50':>8} {'|fix|p90':>8}")
for lo, hi in [(0,5),(5,10),(10,15),(15,24)]:
    m = clean & (true_rng>=lo) & (true_rng<hi)
    if m.sum()==0: continue
    o = off[m]
    print(f"  {f'{lo}-{hi}m':>8} {m.sum():4d} {o[:,0].mean():+6.2f} {o[:,1].mean():+6.2f} "
          f"{o[:,2].mean():+6.2f} {pct(fix[m],50):8.2f} {pct(fix[m],90):8.2f}")
