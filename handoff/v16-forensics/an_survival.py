"""Survival curves P(reach >=gate k) by config group + death-class breakdown + clamp stats."""
import math, random
import pglib as P
from an_master import terminal_state, classify_death

random.seed(7)


def collect():
    out = []
    for s in P.all_sessions():
        S = P.load_session(s)
        if S is None:
            continue
        ts = terminal_state(S)
        cls = classify_death(S, ts)
        out.append((S, ts, cls))
    return out


def survival(gates):
    n = len(gates)
    return {k: sum(1 for g in gates if g >= k) / n for k in range(0, 7)}


def mean_se(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var / n)


def boot_ci(xs, stat=lambda a: sum(a) / len(a), nb=4000):
    n = len(xs)
    bs = []
    for _ in range(nb):
        samp = [xs[random.randrange(n)] for _ in range(n)]
        bs.append(stat(samp))
    bs.sort()
    return bs[int(0.025 * nb)], bs[int(0.975 * nb)]


def perm_test(a, b, nb=20000):
    """Two-sided permutation test on difference of means."""
    obs = sum(a) / len(a) - sum(b) / len(b)
    pool = a + b
    na = len(a)
    cnt = 0
    for _ in range(nb):
        random.shuffle(pool)
        d = sum(pool[:na]) / na - sum(pool[na:]) / len(pool[na:])
        if abs(d) >= abs(obs) - 1e-12:
            cnt += 1
    return obs, cnt / nb


def main():
    rows = collect()
    clean = [(S, ts, cls) for S, ts, cls in rows if not S["storm"]["flag"]]
    print("Total %d sessions, %d storms excluded, %d clean policy attempts" % (
        len(rows), len(rows) - len(clean), len(clean)))

    # group by (rs, pclmp, map-era) among rs>=0.95 accur7 (the systematic sweep)
    def key(S):
        return (S["rs"], S["pclmp"], "logecho" if "logecho" in S["map"] else "accur7")
    groups = {}
    for S, ts, cls in clean:
        groups.setdefault(key(S), []).append((S, ts, cls))

    print("\n" + "=" * 120)
    print("SURVIVAL BY CONFIG GROUP (clean attempts only). P>=k = frac reaching >= k gates.")
    print("=" * 120)
    print("%-26s %3s %5s %5s | %5s %5s %5s %5s %5s %5s | classes" % (
        "rs / pclmp / map", "n", "mean", "med", "P>=1", "P>=2", "P>=3", "P>=4", "P>=5", "P>=6"))
    order = sorted(groups.keys(), key=lambda k: (-(k[0] or 0), k[1] or 0, k[2]))
    for k in order:
        g = groups[k]
        gates = [S["gates_passed"] for S, _, _ in g]
        sv = survival(gates)
        m, se = mean_se(gates)
        cl = {}
        for _, _, c in g:
            cl[c] = cl.get(c, 0) + 1
        clstr = " ".join("%s:%d" % (c, n) for c, n in sorted(cl.items(), key=lambda x: -x[1]))
        print("%-26s %3d %5.2f %5.1f | %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f | %s" % (
            "rs%s pcl%s %s" % (k[0], k[1], k[2]), len(g), m, P.median(gates),
            sv[1], sv[2], sv[3], sv[4], sv[5], sv[6], clstr))

    # focused clamp comparison at rs 1.0 accur7
    print("\n" + "=" * 120)
    print("CLAMP COMPARISON at rs=1.0, accur7 map (the A-question). Gates-passed per flight:")
    print("=" * 120)
    def sel(rs_set, pcl):
        return [S["gates_passed"] for S, ts, cls in clean
                if S["rs"] in rs_set and S["pclmp"] == pcl and "logecho" not in S["map"]]
    for pcl in [0.0, 10.0, 15.0, 17.0, 20.0]:
        xs = sel({1.0}, pcl)
        if not xs:
            continue
        m, se = mean_se(xs)
        lo, hi = boot_ci(xs) if len(xs) >= 3 else (m, m)
        deep = sum(1 for g in xs if g >= 5) / len(xs)
        print("  pclmp=%-4s n=%2d  mean=%.2f±%.2f  95%%CI[%.2f,%.2f]  median=%s  P>=4=%.2f P>=5=%.2f  gates=%s" % (
            pcl, len(xs), m, se, lo, hi, P.median(xs), sum(1 for g in xs if g >= 4) / len(xs), deep, sorted(xs)))
    # pairwise perm tests
    a = sel({1.0}, 17.0); b = sel({1.0}, 15.0); c = sel({1.0}, 0.0)
    print("\n  Permutation tests (diff of mean gates):")
    for (n1, x1), (n2, x2) in [(("pcl17", a), ("pcl15", b)), (("pcl17", a), ("pcl0", c)), (("pcl15", b), ("pcl0", c))]:
        if x1 and x2:
            d, p = perm_test(list(x1), list(x2))
            print("    %-6s(n=%d,m=%.2f) vs %-6s(n=%d,m=%.2f): Δ=%.2f  p=%.3f" % (
                n1, len(x1), sum(x1) / len(x1), n2, len(x2), sum(x2) / len(x2), d, p))
    # pooled 0.95+1.0 for clamp 15 vs 17 (more n)
    print("\n  Pooled rs in {0.95,1.0}:")
    for pcl in [15.0, 17.0]:
        xs = sel({0.95, 1.0}, pcl)
        m, se = mean_se(xs)
        lo, hi = boot_ci(xs)
        print("    pclmp=%-4s n=%2d mean=%.2f±%.2f CI[%.2f,%.2f] P>=5=%.2f gates=%s" % (
            pcl, len(xs), m, se, lo, hi, sum(1 for g in xs if g >= 5) / len(xs), sorted(xs)))
    aa = sel({0.95, 1.0}, 17.0); bb = sel({0.95, 1.0}, 15.0)
    d, p = perm_test(list(aa), list(bb))
    print("    perm pcl17 vs pcl15 (pooled): Δ=%.2f p=%.3f" % (d, p))

    # death-zone entry rate by config (reached deep = gf>=4 OR died in OBSTACLE zone at gf>=4)
    print("\n" + "=" * 120)
    print("DEATH-ZONE ENTRY (reached gate_index>=4 = candidate for the overhead-obstacle experiment)")
    print("=" * 120)
    for k in order:
        g = groups[k]
        deep = [1 if S["gates_passed"] >= 4 else 0 for S, _, _ in g]
        obst = sum(1 for S, ts, cls in g if cls == "OBSTACLE/short" and ts["gf"] >= 4)
        print("  rs%s pcl%s %-8s n=%2d  P(reach g>=4)=%.2f (%d/%d)  OBSTACLE-zone deaths gf>=4: %d" % (
            k[0], k[1], k[2], len(g), sum(deep) / len(g), sum(deep), len(g), obst))
    return rows


if __name__ == "__main__":
    main()
