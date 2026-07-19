"""B-block: arrival-state survived vs clipped (gate-plane crossing miss = aperture metric),
speed vs roll-authority knee, slot1 preview accuracy."""
import math
import pglib as P
from an_master import terminal_state, classify_death


def project_crossing(S, ianchor):
    """From the anchor tick (a fresh detection), project drone to the gate plane (rel_x->0)
    using body-frame velocity. Returns (lat_miss, vert_miss, t_to_plane, speed, vz_lvl)."""
    rel = S["rel"][ianchor]
    v = S["obs"][ianchor][0:3]
    rx, ry, rz = rel
    vx, vy, vz = v
    if abs(vx) < 0.3 or rx >= 0:
        return None
    tstar = rx / vx  # both typically negative -> positive
    if tstar <= 0 or tstar > 1.5:
        return None
    ry_c = ry - vy * tstar
    rz_c = rz - vz * tstar
    roll, pitch = S["roll"][ianchor], S["pitch"][ianchor]
    vert = P.leveled_z((0.0, ry_c, rz_c), roll, pitch)
    lat = ry_c
    return abs(lat), abs(vert), tstar, S["speed"][ianchor], S["vz_lvl"][ianchor]


def approaches():
    """Yield (kind, state_at_r3, crossing) for survived passes and clipped deaths."""
    surv = []
    clip = []
    for s in P.all_sessions():
        S = P.load_session(s)
        if S is None or S["storm"]["flag"] or S["maxv"] > 20:
            continue
        n = S["n"]
        # boundaries of approaches: start indices
        bnds = [0] + [a[0] for a in S["adv"]] + [n]
        adv_idx = set(a[0] for a in S["adv"])
        for k in range(len(bnds) - 1):
            i0, i1 = bnds[k], bnds[k + 1]
            passed = (i1 in adv_idx)  # this approach ended in an advance
            seg = list(range(i0, i1))
            if len(seg) < 3:
                continue
            # anchor = last fresh detection with hrng in [1.2, 6]
            anchor = None
            for i in range(i1 - 1, i0 - 1, -1):
                a = S["age0"][i]
                if a is not None and a < 1e-6 and 1.2 <= S["hrng"][i] <= 6.5:
                    anchor = i; break
            # state at r=3 (last downward crossing of hrng~3 in seg, fresh or coast)
            r3 = None
            for i in range(i1 - 1, i0, -1):
                if S["hrng"][i] >= 3.0 and S["conf0"][i] > 1e-6:
                    r3 = i; break
            rec = {"sess": S["sess"][9:24], "chase": S["gate_index"][i0] if i0 < n else None}
            if r3 is not None:
                rec["r3"] = (S["speed"][r3], S["vz_lvl"][r3], abs(S["rel"][r3][1]), abs(S["dh"][r3]))
            if anchor is not None:
                rec["cross"] = project_crossing(S, anchor)
            if passed:
                surv.append(rec)
            else:
                # only count as clip if it actually reached the gate (GATE_CLIP), not obstacle/short
                ts = terminal_state(S); cls = classify_death(S, ts)
                if cls == "GATE_CLIP":
                    clip.append(rec)
    return surv, clip


def dist(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "n=0"
    return "n=%d med=%.2f p25=%.2f p75=%.2f p90=%.2f max=%.2f" % (
        len(vals), P.median(vals), P.quantile(vals, .25), P.quantile(vals, .75),
        P.quantile(vals, .9), max(vals))


def main():
    surv, clip = approaches()
    print("Survived passes: %d   Clipped-at-gate deaths: %d" % (len(surv), len(clip)))

    print("\n" + "=" * 96)
    print("[V16] ARRIVAL STATE at r=3m (obs):  SURVIVED vs CLIPPED   (speed, |vz_lvl|, |y|, |dh|)")
    print("=" * 96)
    for lbl, S in [("SURVIVED", surv), ("CLIPPED ", clip)]:
        r3 = [r["r3"] for r in S if "r3" in r]
        print(" %s @r3:" % lbl)
        print("    speed  : " + dist([x[0] for x in r3]))
        print("    |vz|   : " + dist([abs(x[1]) for x in r3]))
        print("    |y|off : " + dist([x[2] for x in r3]))
        print("    |dh|   : " + dist([x[3] for x in r3]))

    print("\n" + "=" * 96)
    print("[V16] GATE-PLANE CROSSING MISS (projected from last fresh detect) = APERTURE metric")
    print("=" * 96)
    for lbl, Sset in [("SURVIVED", surv), ("CLIPPED ", clip)]:
        cr = [r["cross"] for r in Sset if r.get("cross") is not None]
        lat = [c[0] for c in cr]; vert = [c[1] for c in cr]
        radial = [math.hypot(c[0], c[1]) for c in cr]
        spd = [c[3] for c in cr]
        print(" %s: (n=%d)" % (lbl, len(cr)))
        print("    lateral miss |y| : " + dist(lat))
        print("    vertical miss|dh|: " + dist(vert))
        print("    radial miss      : " + dist(radial))
        print("    speed at anchor  : " + dist(spd))

    # aperture margin: the radius that separates survive from clip
    sr = sorted(math.hypot(*r["cross"][:2]) for r in surv if r.get("cross"))
    cr = sorted(math.hypot(*r["cross"][:2]) for r in clip if r.get("cross"))
    if sr and cr:
        print("\n  Survived radial-miss p90 = %.2f m ; Clipped radial-miss p25 = %.2f m ; p50=%.2f" % (
            P.quantile(sr, .9), P.quantile(cr, .25), P.median(cr)))
        print("  => training aperture half-width to enforce ~ %.2f m (survive stays within; clips exceed)" %
              (0.5 * (P.quantile(sr, .9) + P.median(cr))))

    # speed vs roll-authority knee
    print("\n" + "=" * 96)
    print("[V16] SPEED vs ROLL-COMMAND AUTHORITY  (|rate_frd[0]| binned by speed)")
    print("=" * 96)
    pairs = []
    for s in P.all_sessions():
        S = P.load_session(s)
        if S is None or S["storm"]["flag"] or S["maxv"] > 20:
            continue
        for i in range(S["n"]):
            if S["t"][i] < 1.0:
                continue
            pairs.append((S["speed"][i], abs(S["rate_frd"][i][0])))
    edges = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14]
    print("%-9s %6s %8s %8s %8s" % ("speed", "n", "roll_med", "roll_p90", "roll_max"))
    for a, b in zip(edges[:-1], edges[1:]):
        sub = [r for sp, r in pairs if a <= sp < b]
        if sub:
            print("%4.0f-%-4.0f %6d %8.3f %8.3f %8.3f" % (
                a, b, len(sub), P.median(sub), P.quantile(sub, .9), max(sub)))
    print("  (rw_progress_vcap 7.5->5.0 check: is p90 roll authority still rising past 5-7.5 m/s?)")

    # slot1 preview accuracy: compare slot1 at pass to slot0 truth right after
    print("\n" + "=" * 96)
    print("[V16] SLOT1 PREVIEW ACCURACY: slot1 rel (pre-pass) vs slot0 truth (post-pass)")
    print("=" * 96)
    errs = []
    for s in P.all_sessions():
        S = P.load_session(s)
        if S is None or S["storm"]["flag"] or S["maxv"] > 20:
            continue
        for (i, g0, g1, t) in S["adv"]:
            # slot1 just before pass (rel_flu1 = NON-flipped) -> convert to flipped (-x,-y,z)
            pre = None
            for j in range(i - 1, max(i - 10, 0), -1):
                rf = S["rel_flu1"][j]
                if rf is not None and S["conf1"][j] > 1e-6:
                    pre = (j, rf); break
            if pre is None:
                continue
            j, rf = pre
            pred = (-rf[0], -rf[1], rf[2])  # flipped frame to match slot0
            # slot0 truth: first fresh slot0 after pass
            post = None
            for k in range(i, min(i + 40, S["n"])):
                a = S["age0"][k]
                if a is not None and a < 1e-6:
                    post = (k, S["rel"][k]); break
            if post is None:
                continue
            k, tru = post
            dt = S["t"][k] - S["t"][j]
            # account for drone travel between j and k (integrate body velocity)
            # compare horizontal range predicted vs true (both to same gate); dt small
            pr = math.hypot(pred[0], pred[1]); tr = math.hypot(tru[0], tru[1])
            errs.append((pr, tr, pr - tr, dt, abs(pred[1] - tru[1])))
    if errs:
        print("  n=%d pass-transitions with slot1 preview + slot0 truth" % len(errs))
        print("  slot1 predicted hrng: med=%.1f ; slot0 truth hrng: med=%.1f" % (
            P.median([e[0] for e in errs]), P.median([e[1] for e in errs])))
        print("  hrng error (pred-truth, incl %.2fs travel): med=%.2f p90|err|=%.2f" % (
            P.median([e[3] for e in errs]), P.median([e[2] for e in errs]),
            P.quantile([abs(e[2]) for e in errs], .9)))
        print("  lateral(y) |pred-truth|: med=%.2f p90=%.2f" % (
            P.median([e[4] for e in errs]), P.quantile([e[4] for e in errs], .9)))


if __name__ == "__main__":
    main()
