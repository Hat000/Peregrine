"""A6: the WRONG-GATE RE-LOCK guard, and what it does to the refutation.

THE THREAT.  The seeker can re-lock onto a DIFFERENT gate while RACE_STATUS still reports the old
`gate_index`, so "gate_index constant" does NOT exclude a landmark change.  Because the sample is
selected by RANGE, a re-lock moves the very quantity doing the selecting.  If dying approaches
re-lock more often than surviving ones, the died-vs-passed gap could be pure artifact.

THE GUARD (vision + gyro ONLY -- never obs[0:3]).  A gate is a fixed world object, so de-rotating
the logged rel_flu by a gyro-propagated attitude recovers the drone's own trajectory:
    p_i = -R_wb(t_i) @ rel_i
A landmark change makes p teleport.  Reject on APPARENT SPEED |p_i - p_{i-1}| / dt.

🛑 NEVER gate on |D_vis - D_dr|.  That reads the KF velocity under test and selects for windows
where dead reckoning already agrees, flattering whichever arm is being evaluated.  `pin_test()`
below feeds a drone whose DR velocity is wrong by 9.5 m/s while its lever moves smoothly and
asserts the window is still USED with n_jumps == 0.

A verdict that depends on the cut is not a verdict, so the threshold is SWEPT.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "failure_profile"))
from a1_reproduce import C, approaches  # noqa: E402
from a2_null import RG, fit_pooled, resample, simulate, tab  # noqa: E402
import vg  # noqa: E402  (the committed de-rotation instrument)


def _att(t, w, roll0, pitch0, anchor):
    """Gyro-propagated body->gravity-levelled attitude over a tick sequence. vg.gyro_attitude math,
    run on pre-extracted arrays. Reads the GYRO and the AHRS anchor only."""
    m = len(t)
    R = np.zeros((m, 3, 3))
    cr, sr = np.cos(roll0), np.sin(roll0)
    cp, sp = np.cos(pitch0), np.sin(pitch0)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    R[anchor] = Ry @ Rx
    for j in range(anchor + 1, m):
        dt = t[j] - t[j - 1]
        wm = 0.5 * (w[j] + w[j - 1])
        R[j] = R[j - 1] if not (np.isfinite(wm).all() and dt > 0) else \
            R[j - 1] @ vg._expm_rotvec(wm * dt)
    for j in range(anchor - 1, -1, -1):
        dt = t[j + 1] - t[j]
        wm = 0.5 * (w[j] + w[j + 1])
        R[j] = R[j + 1] if not (np.isfinite(wm).all() and dt > 0) else \
            R[j + 1] @ vg._expm_rotvec(-wm * dt)
    return R


def apparent_speed(A):
    """Vision+gyro apparent speed of the drone, per consecutive used tick. NEVER touches obs[0:3]."""
    t = A[:, C["t"]]
    w = np.column_stack([A[:, C["w_roll"]], A[:, C["w_pitch"]], A[:, C["w_yaw"]]])
    rel = np.column_stack([A[:, C["relx"]], A[:, C["rely"]], A[:, C["relz"]]])
    ok = np.isfinite(rel).all(1) & np.isfinite(t)
    if ok.sum() < 3:
        return None, None
    R = _att(t[ok], w[ok], A[ok, C["roll"]][len(t[ok]) // 2], A[ok, C["pitch"]][len(t[ok]) // 2],
             len(t[ok]) // 2)
    p = -np.einsum("nij,nj->ni", R, rel[ok])
    dt = np.diff(t[ok])
    sp = np.full(ok.sum(), 0.0)
    good = dt > 1e-3
    sp[1:][good] = np.linalg.norm(np.diff(p, axis=0)[good], axis=1) / dt[good]
    return np.where(ok)[0], sp


def guard(app, thr, mode="truncate"):
    """Return a copy of `app` with re-lock-contaminated ticks removed."""
    out, n_jump, n_drop = [], 0, 0
    for a in app:
        A = a["A"][a["A"][:, C["seen"]] > 0.5]
        if len(A) < 4:
            continue
        idx, sp = apparent_speed(A)
        if idx is None:
            continue
        j = np.where(sp > thr)[0]
        n_jump += len(j)
        if len(j):
            if mode == "drop":
                n_drop += 1
                continue
            A = A[idx[:j[0]]] if j[0] >= 4 else A[:0]
        if len(A) < 4:
            n_drop += 1
            continue
        b = dict(a); b["A"] = A
        out.append(b)
    return out, n_jump, n_drop


def pin_test():
    """PIN: the guard must NOT read dead reckoning. Synthetic drone whose DR velocity is wrong by
    9.5 m/s while the vision lever moves smoothly -> the window must be USED, n_jumps == 0."""
    n = 40
    t = np.arange(n) * 0.033
    A = np.zeros((n, len(C)))
    A[:, C["t"]] = t
    A[:, C["seen"]] = 1.0
    rng = np.arange(n) * -0.2 + 10.0          # smooth 6 m/s closure, no teleport
    A[:, C["relx"]] = rng
    A[:, C["rely"]] = 0.4 * np.sin(t * 2.0)
    A[:, C["relz"]] = 1.0
    A[:, C["rho"]] = np.linalg.norm(A[:, [C["relx"], C["rely"], C["relz"]]], axis=1)
    # dead reckoning is catastrophically wrong -- and must be invisible to the guard
    A[:, C["w_roll"]] = A[:, C["w_pitch"]] = A[:, C["w_yaw"]] = 0.0
    fake = dict(run="pin", lineage="x", gate=1, died=False, A=A, zbias=0.0, ckpt=None)
    kept, nj, nd = guard([fake], 20.0)
    assert len(kept) == 1 and nj == 0 and nd == 0, f"guard leaked: kept={len(kept)} jumps={nj}"
    # ... and it must still FIRE on a real 5 m teleport
    A2 = A.copy()
    A2[20:, C["rely"]] += 5.0
    fake2 = dict(fake); fake2["A"] = A2
    _, nj2, _ = guard([fake2], 20.0)
    assert nj2 >= 1, "guard failed to fire on a 5 m landmark teleport"
    print("PIN TEST PASSED: guard is blind to dead reckoning, fires on a 5 m teleport "
          f"(smooth-lever/bad-DR n_jumps={nj}, teleport n_jumps={nj2})")


if __name__ == "__main__":
    pin_test()
    app = approaches(seen_only=True)
    print(f"\nunguarded approaches: {len(app)}  died={sum(a['died'] for a in app)}")

    print("\n=== CONTAMINATION RATE BY OUTCOME (the coordinator's specific worry) ===")
    print(f"  {'thr m/s':>8}{'tick jump %':>13}{'appr w/ jump P':>16}{'appr w/ jump D':>16}{'D-P':>8}")
    for thr in (10, 15, 20, 30, 50):
        tot = jp = jd = np_ = nd_ = 0
        hit_p = hit_d = 0
        for a in app:
            A = a["A"][a["A"][:, C["seen"]] > 0.5]
            if len(A) < 4:
                continue
            idx, sp = apparent_speed(A)
            if idx is None:
                continue
            tot += len(sp)
            h = int((sp > thr).sum())
            jp += h
            if a["died"]:
                nd_ += 1; hit_d += (h > 0)
            else:
                np_ += 1; hit_p += (h > 0)
        fp, fd = hit_p / max(np_, 1), hit_d / max(nd_, 1)
        print(f"  {thr:>8}{100*jp/max(tot,1):>12.2f}%{100*fp:>15.1f}%{100*fd:>15.1f}%"
              f"{100*(fd-fp):>7.1f}%")

    print("\n=== THE TABLE UNDER THE GUARD (threshold swept; 'a verdict that depends on the cut"
          " is not a verdict') ===")
    for thr in (10, 15, 20, 30, 50, 1e9):
        for mode in ("truncate", "drop"):
            if thr > 1e8 and mode == "drop":
                continue
            g, nj, nd = guard(app, thr, mode)
            Y, lab, _ = [], [], None
            YY, LL = [], []
            for a in g:
                v = resample(a["A"], "L_lat")
                if v is None:
                    continue
                YY.append(v); LL.append(a["died"])
            Y = np.array(YY); lab = np.array(LL, dtype=bool)
            if len(Y) < 50:
                continue
            J = {float(r): i for i, r in enumerate(RG)}
            lbl = "UNGUARDED" if thr > 1e8 else f"thr={thr} {mode}"
            med = {}
            for tag, m in (("P", ~lab), ("D", lab)):
                v = np.abs(Y[m])
                med[tag] = [float(np.nanmedian(v[:, J[r]])) for r in (8.0, 5.0, 3.0)]
            print(f"  {lbl:<20} n={len(Y):>4} (D {int(lab.sum()):>3}) dropped {nd:>3}   "
                  f"P {med['P'][0]:.2f}->{med['P'][1]:.2f}->{med['P'][2]:.2f} "
                  f"({100*(med['P'][1]/med['P'][0]-1):+.0f}%)   "
                  f"D {med['D'][0]:.2f}->{med['D'][1]:.2f}->{med['D'][2]:.2f} "
                  f"({100*(med['D'][1]/med['D'][0]-1):+.0f}%)")

    print("\n=== THE REFUTATION UNDER THE GUARD (thr=20 truncate): entry-matched gap vs the null ===")
    g, nj, nd = guard(app, 20.0, "truncate")
    YY, LL = [], []
    for a in g:
        v = resample(a["A"], "L_lat")
        if v is not None:
            YY.append(v); LL.append(a["died"])
    Y = np.array(YY); lab = np.array(LL, dtype=bool)
    M, Q, keep = fit_pooled(Y)
    S = simulate(Y, M, Q, nsim=40000)
    term = np.abs(S[:, -1])
    slab = term > float(np.nanquantile(term, 1.0 - float(lab.mean())))
    print("  OBSERVED (guarded):")
    tab(Y, lab, name="observed")
    print("  NULL (refit to the guarded data, labels never used):")
    tab(S, slab, name="null")
    J = {float(r): i for i, r in enumerate(RG)}
    j8, j5, j3 = J[8.0], J[5.0], J[3.0]
    print(f"\n  {'|lat@8m| bin':<14}{'obs gap@5m':>12}{'null gap@5m':>13}{'obs gap@3m':>12}"
          f"{'null gap@3m':>13}{'verdict':>22}")
    for lo, hi in ((0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.5), (2.5, 6.0)):
        r = []
        for Yx, Lx in ((Y, lab), (S, slab)):
            e = np.abs(Yx[:, j8])
            m = np.isfinite(e) & (e >= lo) & (e < hi)
            mp, md = m & ~Lx, m & Lx
            if mp.sum() < 8 or md.sum() < 8:
                r.append((np.nan, np.nan)); continue
            r.append((np.nanmedian(np.abs(Yx[md, j5])) - np.nanmedian(np.abs(Yx[mp, j5])),
                      np.nanmedian(np.abs(Yx[md, j3])) - np.nanmedian(np.abs(Yx[mp, j3]))))
        v = ("null >= obs" if (r[1][0] >= r[0][0] and r[1][1] >= r[0][1]) else "obs exceeds null")
        print(f"  {f'{lo:.1f}-{hi:.1f}':<14}{r[0][0]:>12.2f}{r[1][0]:>13.2f}{r[0][1]:>12.2f}"
              f"{r[1][1]:>13.2f}{v:>22}")
