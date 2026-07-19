"""B-block detection calibration (CORRECTED).
Signals: detect_ms>0.5 == a camera frame was processed this tick (NOT gate-specific).
         age0 (obs age_s) == 0  ->  the CURRENT gate (slot0) was FRESHLY detected this frame.
P(fresh slot0 | frame processed, tracking) vs range = the true detector availability curve.
conf0/area0 on fresh ticks = detection QUALITY vs range.
"""
import math
import pglib as P


def is_fresh(S, i):
    a = S["age0"][i]
    return a is not None and a < 1e-6


def collect(root=P.V15_ROOT):
    trk = []   # frame-processed & tracking (conf0>0): (rng, hrng, fresh01, conf, area, dh)
    fa = []
    for s in P.all_sessions(root):
        S = P.load_session(s, root)
        if S is None or S["storm"]["flag"] or S["maxv"] > 20:
            continue
        nf = sum(1 for d in S["detect_ms"] if d and d > 0.5)
        fa.append(nf / S["n"])
        for i in range(S["n"]):
            dms = S["detect_ms"][i]
            if not (dms and dms > 0.5):
                continue
            if S["conf0"][i] <= 1e-6:
                continue
            f = 1 if is_fresh(S, i) else 0
            trk.append((S["rng"][i], S["hrng"][i], f, S["conf0"][i], S["area0"][i], S["dh"][i]))
    return trk, fa


def bins_of(trk, edges, key, val):
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        sub = [val(s) for s in trk if a <= key(s) < b]
        out.append((a, b, len(sub), sum(sub) / len(sub) if sub else None))
    return out


def main():
    trk, fa = collect()
    print("Frame availability (frac ticks w/ processed frame): mean=%.3f min=%.3f max=%.3f" % (
        sum(fa) / len(fa), min(fa), max(fa)))
    print("Tracking (frame-processed & slot0 live) samples: %d\n" % len(trk))

    edges = [0, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3, 3.5, 4, 5, 6, 8, 10, 12, 15, 18, 22, 26, 32]
    print("=" * 78)
    print("[V16] DETECTION AVAILABILITY  P(fresh slot0 | processed frame, in-view) vs slant range")
    print("=" * 78)
    print("%-11s %6s %8s" % ("range(m)", "n", "P_fresh"))
    curve = bins_of(trk, edges, lambda s: s[0], lambda s: s[2])
    for a, b, n, m in curve:
        if n:
            print("%4.1f-%-5.1f %6d %8.3f  %s" % (a, b, n, m, "#" * int(40 * m)))

    # sigmoid fit near-side: p_max / (1+exp(-(r-r50)/w)) with far rolloff ignored (fit r in [0.8,8])
    pts = [(0.5 * (a + b), m, n) for a, b, n, m in curve if m is not None and n >= 20 and 0.5 * (a + b) <= 9]
    best = None
    for pmax in [x / 100 for x in range(95, 101)]:
        for r50 in [x / 20 for x in range(30, 80)]:      # 1.5 .. 4.0
            for w in [x / 40 for x in range(8, 60)]:       # 0.2 .. 1.5
                err = sum(n * (pmax / (1 + math.exp(-(r - r50) / w)) - m) ** 2 for r, m, n in pts)
                if best is None or err < best[0]:
                    best = (err, pmax, r50, w)
    _, pmax, r50, w = best
    print("\n[V16] FIT  P_detect(r) = %.2f / (1 + exp(-(r-%.2f)/%.2f))   for r<=~22m (roll off beyond)" % (pmax, r50, w))
    print("       -> P=0.5 at r=%.2f m ; P>=0.9 for r>= %.2f m ; ~0 below r= %.2f m" % (
        r50, r50 + w * math.log(pmax / 0.9 - 1) * -1 if pmax > 0.9 else r50, r50 - 3 * w))
    for r in [1.0, 1.3, 1.5, 1.75, 2.0, 2.25, 2.5, 3, 4, 6]:
        print("    r=%.2f m  P=%.3f" % (r, pmax / (1 + math.exp(-(r - r50) / w))))

    print("\n" + "=" * 78)
    print("[V16] DETECTION QUALITY: conf0 & area0 vs range (on FRESH ticks only)")
    print("=" * 78)
    fresh = [s for s in trk if s[2] == 1]
    print("%-11s %6s %8s %8s %8s" % ("range(m)", "n", "conf_med", "conf_mn", "area_mn"))
    for a, b in zip(edges[:-1], edges[1:]):
        sub = [s for s in fresh if a <= s[0] < b]
        if sub:
            print("%4.1f-%-5.1f %6d %8.3f %8.3f %8.3f" % (
                a, b, len(sub), P.median([s[3] for s in sub]), sum(s[3] for s in sub) / len(sub),
                sum(s[4] for s in sub) / len(sub)))

    # acquisition range per gate: first conf0>0 tick in each gate segment (after blackout)
    print("\n" + "=" * 78)
    print("[V16] ACQUISITION hrng: first slot0 detection range of each gate approach")
    print("=" * 78)
    acq = {}
    for s in P.all_sessions():
        S = P.load_session(s)
        if S is None or S["storm"]["flag"] or S["maxv"] > 20:
            continue
        cur = -1; seen = False
        for i in range(S["n"]):
            gi = S["gate_index"][i]
            if gi != cur:
                cur = gi; seen = False
            if not seen and S["conf0"][i] > 1e-6:
                acq.setdefault(gi, []).append(S["hrng"][i]); seen = True
    for gi in range(7):
        hs = acq.get(gi, [])
        if hs:
            print("  gate_idx %d chase: n=%2d  acq hrng med=%.1f  p10=%.1f p90=%.1f" % (
                gi, len(hs), P.median(hs), P.quantile(hs, 0.1), P.quantile(hs, 0.9)))


if __name__ == "__main__":
    main()
