"""Settle the pitch sign EMPIRICALLY, from flown data, not from convention.

Everything about the v1.7 launch verdict turns on whether a NEGATIVE rate_frd[1]
pitches the nose down or up.  The banked footgun says a_pitch<0 = nose-DOWN; the
textbook FRD reading (y to the right, right-hand rule) says positive q = nose-down,
i.e. the opposite.  Rather than pick a side, measure it:

  1. Does rate_frd[1] predict the change in the pitch ATTITUDE channel obs[4]?
  2. Does a sustained sign of rate_frd[1] make the drone CLIMB or DESCEND?
     (kf_pos_ned[2] is NED z: positive DOWN, so descending = z increasing.)

Both are computed over the whole record flight, using only logged quantities.
"""
from __future__ import annotations

import json
from pathlib import Path

RUN = Path(r"C:\Users\Fengy\Downloads\Projects\wt-mapfix\data\runs"
           r"\20260721_040555_v16pick_Qs1_f1")


def corr(a: list[float], b: list[float]) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a) ** 0.5
    vb = sum((x - mb) ** 2 for x in b) ** 0.5
    if va == 0 or vb == 0:
        return float("nan")
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (va * vb)


def main() -> int:
    rows = [json.loads(ln) for ln in
            (RUN / "ego_obs.jsonl").read_text().splitlines() if ln.strip()]

    t = [r["sim_time_ns"] / 1e9 for r in rows]
    q = [r["rate_frd"][1] for r in rows]
    obs4 = [r["obs"][4] for r in rows]
    z_ned = [r["kf_pos_ned"][2] for r in rows]

    # --- 1. commanded pitch rate vs change in the attitude channel --------------
    dq, dobs4 = [], []
    for i in range(len(rows) - 1):
        dt = t[i + 1] - t[i]
        if dt <= 0:
            continue
        dq.append(q[i])
        dobs4.append((obs4[i + 1] - obs4[i]) / dt)
    c1 = corr(dq, dobs4)
    print(f"corr( rate_frd[1] , d(obs[4])/dt )        = {c1:+.3f}   (n={len(dq)})")
    print(f"   -> a positive rate_frd[1] makes obs[4] {'INCREASE' if c1 > 0 else 'DECREASE'}")

    # --- 2. sustained pitch command vs vertical motion ---------------------------
    # Use windows where the command holds one sign for >= 5 ticks, and ask which way
    # the drone actually went over that window.
    climbs_pos, climbs_neg = [], []
    i = 0
    while i < len(rows) - 6:
        s = 1 if q[i] > 0.2 else (-1 if q[i] < -0.2 else 0)
        if s == 0:
            i += 1
            continue
        j = i
        while j < len(rows) - 1 and (q[j] > 0.2 if s > 0 else q[j] < -0.2):
            j += 1
        if j - i >= 5:
            dz = z_ned[j] - z_ned[i]        # NED z: positive = went DOWN
            (climbs_pos if s > 0 else climbs_neg).append(dz)
        i = max(j, i + 1)

    def summarize(name: str, xs: list[float]) -> None:
        if not xs:
            print(f"   {name}: no qualifying window")
            return
        m = sum(xs) / len(xs)
        verdict = "DESCENDED" if m > 0 else "CLIMBED"
        print(f"   {name}: n={len(xs):<3} mean d(NED z) = {m:+.2f} m -> drone {verdict}")

    print()
    print("sustained-command windows (>=5 ticks, |cmd| > 0.2 rad/s):")
    summarize("rate_frd[1] > 0", climbs_pos)
    summarize("rate_frd[1] < 0", climbs_neg)

    print()
    print(f"first 4 ticks: rate_frd[1] = {[round(v, 3) for v in q[:4]]}")
    print(f"               obs[4]      = {[round(v, 4) for v in obs4[:4]]}")
    print(f"               NED z       = {[round(v, 2) for v in z_ned[:4]]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
