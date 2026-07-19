"""Master session table + storm flags + terminal-state clustering (death zone vs gate clip)."""
import pglib as P


def terminal_state(S):
    """Terminal-approach descriptor for the FINAL (fatal) gate approach only.
    Restricts to ticks chasing the final gate_index (after the last advance), so a
    just-passed gate does not contaminate the closest-approach-to-final-gate metric."""
    n = S["n"]
    gf = S["gate_index"][-1]
    t = S["t"]
    tend = t[-1]
    # first index at the final gate_index (after last advance)
    i_start = 0
    for i in range(n - 1, -1, -1):
        if S["gate_index"][i] != gf:
            i_start = i + 1
            break
    approach = list(range(i_start, n))
    t_chasing = (tend - t[i_start]) if approach else 0.0
    # closest real approach to the FINAL gate (conf>0, hrng<25 to drop far re-snaps)
    near = [i for i in approach if S["conf0"][i] > 1e-6 and S["hrng"][i] < 25]
    min_hrng = min((S["hrng"][i] for i in near), default=None)
    imin = min(near, key=lambda i: S["hrng"][i]) if near else None
    # last real tracking tick while chasing the final gate
    last_real = None
    for i in range(n - 1, i_start - 1, -1):
        if S["conf0"][i] > 1e-6 and S["hrng"][i] < 22:
            last_real = i
            break
    def state_at(i):
        return {"hrng": S["hrng"][i], "dh": S["dh"][i], "spd": S["speed"][i],
                "vz": S["vz_lvl"][i], "y": abs(S["rel"][i][1]), "roll_cmd": S["rate_frd"][i][0],
                "t_before_end": round(tend - t[i], 2)}
    return {"gf": gf, "min_hrng": min_hrng, "t_chasing": round(t_chasing, 2),
            "last_real": state_at(last_real) if last_real is not None else {},
            "closest": state_at(imin) if imin is not None else {}}


def classify_death(S, ts):
    if S["storm"]["flag"]:
        return "STORM"
    if S["maxv"] > 20:
        return "DIVERGED"
    mh = ts["min_hrng"]
    if mh is None:
        return "NODET"
    # died short of the final gate = obstacle/gap death; reached the gate = clip
    if mh < 3.5:
        return "GATE_CLIP"
    if mh > 7.0:
        return "OBSTACLE/short"
    return "MID"


def main():
    sessions = P.all_sessions()
    print("=" * 150)
    print("MASTER TABLE (v15 corpus, 92 sessions)")
    print("=" * 150)
    hdr = "%-16s %-4s %4s %5s %-8s %2s %4s %5s %5s | gf  mnH  Tdh  Tspd  Tvz   Ty  | class"
    print(hdr % ("sess", "fam", "rs", "pclmp", "map", "g", "coll", "dur", "maxv"))
    print("(mnH=min hrng to FINAL chased gate; T*=state at last real tracking tick)")
    rows = []
    for s in sessions:
        S = P.load_session(s)
        if S is None:
            continue
        ts = terminal_state(S)
        cls = classify_death(S, ts)
        lr = ts["last_real"]
        mp = {"vq2_coarse_map.json": "accur7", "vq2_coarse_map_champion_logecho.json": "logecho"}.get(S["map"], S["map"][:8])
        def f(x, d=1):
            return ("%.*f" % (d, x)) if isinstance(x, (int, float)) else "  -"
        lrstr = ("%2d %5s %5s %5s %5s %5s" % (
            ts["gf"], f(ts["min_hrng"]),
            f(lr.get("dh")) if lr else "-",
            f(lr.get("spd")) if lr else "-", f(lr.get("vz")) if lr else "-",
            f(lr.get("y")) if lr else "-")) if lr else ("%2d %s" % (ts["gf"], "no-real-det"))
        print("%-16s %-4s %4s %5s %-8s %2d %4s %5.1f %5.1f | %s | %s" % (
            s[9:24], S["fam"], str(S["rs"]), str(S["pclmp"]), mp,
            S["gates_passed"], str(S["coll"]), S["dur"], S["maxv"], lrstr, cls))
        rows.append((S, ts, cls))
    # storm summary
    print("\nSTORM-flagged sessions (excluded from policy stats):")
    for S, ts, cls in rows:
        if S["storm"]["flag"]:
            print("  %-16s g=%d coll=%s reasons=%s maxgyro=%.1f net3s=%.1f" % (
                S["sess"][9:24], S["gates_passed"], str(S["coll"]),
                S["storm"]["reason"], S["storm"]["max_gyro_early"], S["storm"]["net_prog3s"]))
    nstorm = sum(1 for S, _, _ in rows if S["storm"]["flag"])
    print("  total storms: %d / %d" % (nstorm, len(rows)))
    return rows


if __name__ == "__main__":
    main()
