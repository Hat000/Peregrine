"""C-block: champion (vpeffs0) + v1 lineage terminal death states vs the v15 death zone.
Key wanted number: the champion's lone 5-gate run final range-to-gate-5."""
import pglib as P
from an_master import terminal_state, classify_death


def describe(sess, root):
    S = P.load_session(sess, root)
    if S is None:
        return None
    ts = terminal_state(S)
    cls = classify_death(S, ts)
    lr = ts["last_real"]
    return S, ts, cls, lr


def main():
    print("=" * 110)
    print("CHAMPION (vpeffs0) + v1 lineage TERMINAL states  [historical git extract]")
    print("=" * 110)
    hist = P.all_sessions(P.HIST_ROOT)
    champ = [s for s in hist if "panel_run" in s]  # vpeffs0 panel runs we pulled (202317 etc)
    v1 = [s for s in hist if "v1pick" in s]
    ratchet = [s for s in hist if "ratchet_p03" in s]

    def row(sess, root):
        r = describe(sess, root)
        if r is None:
            print("  %s: EMPTY"); return
        S, ts, cls, lr = r
        tag = S["fam"]
        lrs = ("gf%d mnH=%s lastHR=%s dh=%s spd=%s vz=%s y=%s" % (
            ts["gf"], ("%.1f" % ts["min_hrng"]) if ts["min_hrng"] is not None else "-",
            ("%.1f" % lr.get("hrng")) if lr else "-", ("%+.1f" % lr.get("dh")) if lr else "-",
            ("%.1f" % lr.get("spd")) if lr else "-", ("%+.1f" % lr.get("vz")) if lr else "-",
            ("%.1f" % lr.get("y")) if lr else "-")) if lr else "gf%d (no det)" % ts["gf"]
        print("  %-22s %-9s rs%s pcl%s g=%d coll=%s | %-14s | %s" % (
            sess[9:30], tag, S["rs"], S["pclmp"], S["gates_passed"], S["coll"], cls, lrs))

    print("\n-- vpeffs0 CHAMPION panel runs (the ones that reached g4-5) --")
    for s in sorted(champ):
        S = P.load_session(s, P.HIST_ROOT)
        if S and S["gates_passed"] >= 4:
            row(s, P.HIST_ROOT)
    print("\n-- champion ratchet_p03 five-flight series {2,2,1,2,2} baseline --")
    for s in sorted(ratchet):
        row(s, P.HIST_ROOT)
    print("\n-- v1 lineage (failed yaw-osc era) reaching >=1 --")
    for s in sorted(v1):
        S = P.load_session(s, P.HIST_ROOT)
        if S and S["gates_passed"] >= 1:
            row(s, P.HIST_ROOT)

    # THE wanted number: champion 5-gate run final range to gate 5
    print("\n" + "=" * 110)
    print("CHAMPION 5-GATE RUN (202317) DETAILED TERMINAL APPROACH  -- range-to-gate-5")
    print("=" * 110)
    for sess in ["20260714_202317_panel_run_f1", "20260714_213605_panel_run_f1"]:
        S = P.load_session(sess, P.HIST_ROOT)
        if S is None:
            continue
        ts = terminal_state(S)
        print("\n%s: gates=%d coll=%s state=%s  final gate_index chased=%d" % (
            sess[9:], S["gates_passed"], S["coll"], S["state"], ts["gf"]))
        print("  min hrng to final gate = %s m ; class=%s" % (
            ("%.1f" % ts["min_hrng"]) if ts["min_hrng"] else "-", classify_death(S, ts)))
        # last 12 ticks
        n = S["n"]
        print("  last 12 ticks (idx t gi conf0 hrng dh spd vz_lvl):")
        for i in range(max(0, n - 12), n):
            print("    %3d %5.2f g%d cf=%.2f hr=%5.1f dh=%+5.1f spd=%4.1f vz=%+4.1f" % (
                i, S["t"][i], S["gate_index"][i], S["conf0"][i], S["hrng"][i], S["dh"][i],
                S["speed"][i], S["vz_lvl"][i]))

    # compare champion death-zone vs v15
    print("\n" + "=" * 110)
    print("DEATH-ZONE COMPARISON")
    print("=" * 110)
    czone = []
    for s in champ:
        S = P.load_session(s, P.HIST_ROOT)
        if S is None or S["gates_passed"] < 3:
            continue
        ts = terminal_state(S)
        czone.append((S["gates_passed"], ts["min_hrng"], classify_death(S, ts)))
    print("champion deep runs (g>=3): " + str([(g, round(mh, 1) if mh else None, c) for g, mh, c in czone]))


if __name__ == "__main__":
    main()
