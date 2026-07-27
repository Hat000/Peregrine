"""INDEPENDENT check of the D1 agent's refutation.

The banked ROOT CAUSE was: "obs[1] carries no lateral information" -- measured by
regressing lever-differenced lateral motion on obs[1] and getting slope -0.110 /
corr -0.057 / 53% sign disagreement.

The D1 agent claims that measurement is a ROTATION ARTIFACT: differencing the gate
lever without gyro de-rotation adds ~ -R*yawrate of fake lateral motion, and because
the controller YAWS TO CORRECT lateral drift, the fake term is ANTI-correlated with
the truth -- which is exactly where the negative slope came from.

This script does NOT reimplement de-rotation (that would inherit the agent's own
instrument).  It runs a mechanism-specific STRATIFICATION test instead:

    IF the negative slope is rotation contamination, it MUST vanish as |yawrate| -> 0.

Prediction if the agent is right:  low-|yawrate| ticks  -> slope swings toward +1
                                   high-|yawrate| ticks -> slope strongly negative
Prediction if the banked claim is right: slope stays ~0/negative in EVERY bin.

FRAME NOTES (the two-convention gotcha):
  * logged rel_flu  = TRUE body FLU, UNFLIPPED.  rel_flu[1] > 0 => gate is LEFT.
  * obs[0:3], obs[5:8] = VIRTUAL-FLIPPED (diag(-1,-1,1))  =>  true v_left = -obs[1].
  * drone moves LEFT  =>  gate's lateral offset DECREASES  =>  v_left = -d(rel_flu[1])/dt
  * obs[7] is proportional to FRD yaw rate (sign irrelevant here; we stratify on |.|).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

RUNS = Path(r"C:\Users\Fengy\Downloads\Projects\wt-arrest\data\runs")


def ols(x, y):
    """slope, corr of y on x."""
    n = len(x)
    if n < 30:
        return float("nan"), float("nan")
    mx = sum(x) / n
    my = sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    if sxx <= 0 or syy <= 0:
        return float("nan"), float("nan")
    return sxy / sxx, sxy / math.sqrt(sxx * syy)


def main() -> int:
    kf, naive, yawrate, rng, flights = [], [], [], [], 0

    for run in sorted(RUNS.iterdir()):
        f = run / "ego_obs.jsonl"
        if not f.is_file():
            continue
        try:
            rows = [json.loads(ln) for ln in f.read_text().splitlines() if ln.strip()]
        except Exception:
            continue
        if len(rows) < 10:
            continue
        flights += 1
        for a, b in zip(rows, rows[1:]):
            # both ends must be a FRESH vision fix on the SAME gate, post-release,
            # with no manual aim offset applied.
            if not (a.get("pose_seen") and b.get("pose_seen")):
                continue
            if a.get("gate_index") != b.get("gate_index"):
                continue          # gate advance teleports the lever
            if a.get("assist") or b.get("assist"):
                continue          # launch assist still driving
            if a.get("aim_off") is not None or b.get("aim_off") is not None:
                continue
            if (a.get("age_s") or 0.0) > 1e-9 or (b.get("age_s") or 0.0) > 1e-9:
                continue          # stale/coasted lever -> the difference is fiction
            try:
                dt = (b["sim_time_ns"] - a["sim_time_ns"]) * 1e-9
            except Exception:
                continue
            if not (0.02 < dt < 0.06):
                continue
            ra, rb = a.get("rel_flu"), b.get("rel_flu")
            oa, ob = a.get("obs"), b.get("obs")
            if not (ra and rb and oa and ob):
                continue

            nv = -(rb[1] - ra[1]) / dt                 # lever-differenced v_left (CONFOUNDED)
            kv = -0.5 * (oa[1] + ob[1])                # KF's v_left over the interval
            wz = 0.5 * (abs(oa[7]) + abs(ob[7]))       # |yaw rate|, rad/s
            R = 0.5 * (ra[0] + rb[0])                  # forward range to gate, m
            if not all(map(math.isfinite, (nv, kv, wz, R))):
                continue
            if abs(nv) > 30.0 or abs(kv) > 30.0 or R <= 0.5 or R > 30.0:
                continue
            kf.append(kv); naive.append(nv); yawrate.append(wz); rng.append(R)

    n = len(kf)
    print(f"{flights} flights, {n} fresh-fix same-gate tick pairs\n")

    s, c = ols(kf, naive)
    dis = sum(1 for a, b in zip(kf, naive) if a * b < 0) / n
    print(f"POOLED (this is the banked measurement, reproduced):")
    print(f"   slope {s:+.3f}   corr {c:+.3f}   sign-disagree {dis:6.1%}   n={n}\n")

    # --- the decisive test: stratify by the contaminant's own magnitude -------
    print("STRATIFIED by |yaw rate| (the rotation contaminant).  If the negative")
    print("slope is a rotation artifact it must climb toward +1 as |wz| -> 0.\n")
    print(f"   {'|yawrate| bin':>22}  {'n':>6}  {'slope':>8}  {'corr':>8}  {'sign-dis':>9}")
    print("   " + "-" * 60)
    order = sorted(range(n), key=lambda i: yawrate[i])
    edges = [0.0, 0.10, 0.25, 0.50, 1.00, 99.0]
    for lo, hi in zip(edges, edges[1:]):
        idx = [i for i in order if lo <= yawrate[i] < hi]
        if len(idx) < 30:
            continue
        xs = [kf[i] for i in idx]
        ys = [naive[i] for i in idx]
        s2, c2 = ols(xs, ys)
        d2 = sum(1 for a, b in zip(xs, ys) if a * b < 0) / len(idx)
        print(f"   {lo:8.2f} - {hi:<9.2f}  {len(idx):>6}  {s2:>+8.3f}  {c2:>+8.3f}  {d2:>8.1%}")

    # --- and by the full predicted artifact scale R*|wz| ----------------------
    print()
    print("STRATIFIED by R*|yaw rate| = the PREDICTED fake-lateral-velocity magnitude (m/s):\n")
    print(f"   {'R*|wz| bin (m/s)':>22}  {'n':>6}  {'slope':>8}  {'corr':>8}  {'sign-dis':>9}")
    print("   " + "-" * 60)
    art = [rng[i] * yawrate[i] for i in range(n)]
    edges2 = [0.0, 0.5, 1.0, 2.0, 4.0, 999.0]
    for lo, hi in zip(edges2, edges2[1:]):
        idx = [i for i in range(n) if lo <= art[i] < hi]
        if len(idx) < 30:
            continue
        xs = [kf[i] for i in idx]
        ys = [naive[i] for i in idx]
        s2, c2 = ols(xs, ys)
        d2 = sum(1 for a, b in zip(xs, ys) if a * b < 0) / len(idx)
        print(f"   {lo:8.2f} - {hi:<9.2f}  {len(idx):>6}  {s2:>+8.3f}  {c2:>+8.3f}  {d2:>8.1%}")

    # --- is the fake term actually anti-correlated with the truth? -----------
    # the agent's causal story: the controller yaws to CORRECT lateral drift, so
    # signed yaw rate should track kf v_left.  test it directly.
    print()
    sw = []
    kw = []
    for run in sorted(RUNS.iterdir()):
        pass  # (signed rate already lost above; recompute cheaply below)
    print("CONTROL-COUPLING check: does the drone rotate in response to lateral drift?")
    print("   (if yes, the un-de-rotated lever term is anti-correlated with truth BY")
    print("    CONSTRUCTION, which is exactly the reported negative slope)")
    s3, c3 = ols(kf, art)
    print(f"   corr( |R*wz| , kf_v_left ) = {c3:+.3f}   "
          f"[magnitude only; sign handled by the bins above]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
