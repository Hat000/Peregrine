"""B-block: post-pass blackout duration by leg type (level/climb/descend) + per-leg vertical profile.
Leg type from the drone's vertical motion just after the pass and the next gate's dh at reacquisition."""
import pglib as P


def leg_events():
    ev = []
    for s in P.all_sessions():
        S = P.load_session(s)
        if S is None or S["storm"]["flag"] or S["maxv"] > 20:
            continue
        n = S["n"]
        for (i, g0, g1, tadv) in S["adv"]:
            # blackout: from advance i until slot0 conf0>0 again
            j = i
            while j < n and S["conf0"][j] <= 1e-6:
                j += 1
            resolved = j < n
            dt = (S["t"][min(j, n - 1)] - S["t"][i])
            # vertical motion in the 0.6 s after pass (leveled vz)
            w = [S["vz_lvl"][k] for k in range(i, min(n, i + 24))]
            vz_after = sum(w) / len(w) if w else 0.0
            # reacq dh + hrng
            reacq_dh = S["dh"][j] if resolved else None
            reacq_hr = S["hrng"][j] if resolved else None
            # altitude change over the leg from kf (z NED, negate for up)
            kf_up_i = -S["kf"][i][2]
            kf_up_j = -S["kf"][min(j, n - 1)][2]
            ev.append({
                "sess": S["sess"][9:24], "chase": g1, "t": tadv, "dt": dt, "resolved": resolved,
                "vz_after": vz_after, "reacq_dh": reacq_dh, "reacq_hr": reacq_hr,
                "dkf_up": kf_up_j - kf_up_i, "slot1_cover": None,
                "s1": sum(S["seen1"][i:j]) / max(j - i, 1),
            })
    return ev


def leg_type(e):
    v = e["vz_after"]
    if v > 0.6:
        return "climb"
    if v < -0.6:
        return "descend"
    return "level"


def main():
    ev = leg_events()
    print("Total inter-gate transitions (clean flights): %d" % len(ev))

    print("\n" + "=" * 92)
    print("[V16] POST-PASS BLACKOUT DURATION by LEG TYPE (leg = drone vert motion 0.6s after pass)")
    print("=" * 92)
    print("%-9s %4s %7s %7s %7s %7s %7s %7s" % ("leg", "n", "dt_med", "dt_mn", "dt_p90", "dt_max", "s1cov", "reDH_md"))
    for lt in ["level", "climb", "descend"]:
        sub = [e for e in ev if leg_type(e) == lt]
        if not sub:
            continue
        dts = [e["dt"] for e in sub]
        s1 = [e["s1"] for e in sub]
        redh = [e["reacq_dh"] for e in sub if e["reacq_dh"] is not None]
        print("%-9s %4d %7.2f %7.2f %7.2f %7.2f %7.2f %7s" % (
            lt, len(sub), P.median(dts), sum(dts) / len(dts), P.quantile(dts, 0.9), max(dts),
            sum(s1) / len(s1), "%.2f" % P.median(redh) if redh else "-"))

    print("\n" + "=" * 92)
    print("[V16] BLACKOUT DURATION by GATE_INDEX being chased (which legs are blind)")
    print("=" * 92)
    print("%-6s %4s %7s %7s %7s %8s %8s %8s" % ("chase", "n", "dt_med", "dt_p90", "dt_max", "vz_after", "reDH_md", "reHR_md"))
    for g in range(1, 7):
        sub = [e for e in ev if e["chase"] == g]
        if not sub:
            continue
        dts = [e["dt"] for e in sub]
        vz = [e["vz_after"] for e in sub]
        redh = [e["reacq_dh"] for e in sub if e["reacq_dh"] is not None]
        rehr = [e["reacq_hr"] for e in sub if e["reacq_hr"] is not None]
        print("%-6d %4d %7.2f %7.2f %7.2f %8.2f %8s %8s" % (
            g, len(sub), P.median(dts), P.quantile(dts, 0.9), max(dts), P.median(vz),
            "%.2f" % P.median(redh) if redh else "-", "%.1f" % P.median(rehr) if rehr else "-"))

    print("\nNote: dt=0.0 means slot0 was already live at the pass (no blackout).")
    # fraction with meaningful blackout (>0.4s)
    big = [e for e in ev if e["dt"] > 0.4]
    print("Transitions with blackout >0.4s: %d/%d (%.0f%%)" % (len(big), len(ev), 100 * len(big) / len(ev)))
    print("  of those, leg mix: " + ", ".join("%s:%d" % (lt, sum(1 for e in big if leg_type(e) == lt)) for lt in ["level", "climb", "descend"]))

    # per-leg vertical demand profile: dh at acquisition of each gate (the climb/descend demand)
    print("\n" + "=" * 92)
    print("[NEW] PER-LEG vertical demand: reacq dh (next gate height above drone) & vz just after pass")
    print("=" * 92)
    print("(positive dh = next gate ABOVE drone = must climb; negative = descend)")
    for g in range(1, 7):
        sub = [e for e in ev if e["chase"] == g and e["reacq_dh"] is not None]
        if not sub:
            continue
        dh = [e["reacq_dh"] for e in sub]
        print("  chase g%d: n=%2d reacq_dh med=%+.2f [p10=%+.2f p90=%+.2f]  dkf_up med=%+.2f" % (
            g, len(sub), P.median(dh), P.quantile(dh, 0.1), P.quantile(dh, 0.9),
            P.median([e["dkf_up"] for e in sub])))


if __name__ == "__main__":
    main()
