"""A3: stress the refutation. A wrong refutation is as costly as a wrong finding.

R1  Is the null CALIBRATED?  Compare simulated vs observed marginal |lat| at every range.
R2  PAIRED subset -- the parent's three bands are sampled on DIFFERENT sets of approaches.
R3  ENTRY-MATCHED capture: within narrow lat(8 m) bins, do deaths capture less than passes,
    and does the null predict the same gap?  This is the only comparison the entry state does
    not contaminate.
R4  Tails: per-approach worst events, not medians.
"""
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from a2_null import RG  # noqa: E402

P = pickle.load(open(os.path.join(HERE, "a2.pkl"), "rb"))
Y, lab, S, meta = P["Y"], P["lab"], P["S"], P["meta"]
J = {float(r): i for i, r in enumerate(RG)}
j8, j5, j3 = J[8.0], J[5.0], J[3.0]


def q(x, p):
    x = x[np.isfinite(x)]
    return float(np.quantile(x, p)) if len(x) else np.nan


print("=== R1  NULL CALIBRATION: marginal |lat| by range (all approaches, no labels) ===")
print(f"  {'r':>5}{'n_obs':>7}{'obs p50':>9}{'sim p50':>9}{'obs p90':>9}{'sim p90':>9}"
      f"{'obs p99':>9}{'sim p99':>9}")
for j in range(0, len(RG), 2):
    o, s = np.abs(Y[:, j]), np.abs(S[:, j])
    n = int(np.isfinite(o).sum())
    print(f"  {RG[j]:>5.1f}{n:>7}{q(o,.5):>9.2f}{q(s,.5):>9.2f}{q(o,.9):>9.2f}{q(s,.9):>9.2f}"
          f"{q(o,.99):>9.2f}{q(s,.99):>9.2f}")

print("\n=== R2  SAMPLING ASYMMETRY: are the three bands the same approaches? ===")
for tag, m in (("PASSED", ~lab), ("DIED", lab)):
    v = Y[m]
    h8 = np.isfinite(v[:, j8]); h5 = np.isfinite(v[:, j5]); h3 = np.isfinite(v[:, j3])
    allb = h8 & h5 & h3
    print(f"  {tag:<8} n={m.sum():>4}  has@8m {h8.sum():>4}  has@5m {h5.sum():>4}  "
          f"has@3m {h3.sum():>4}  ALL THREE {allb.sum():>4} "
          f"({100*allb.sum()/max(m.sum(),1):.0f}%)")

print("\n--- PAIRED subset only (every approach contributes to all three bands) ---")
paired = np.isfinite(Y[:, j8]) & np.isfinite(Y[:, j5]) & np.isfinite(Y[:, j3])
sp = np.isfinite(S[:, j8]) & np.isfinite(S[:, j5]) & np.isfinite(S[:, j3])
term = np.abs(S[:, -1])
slab = term > float(np.nanquantile(term, 1.0 - float(lab.mean())))
print(f"  {'':<20}{'n':>6}{'@8m':>8}{'@5m':>8}{'@3m':>8}{'8->5':>8}{'5->3':>8}")
for src, Yx, Lx, mx, nm in ((0, Y, lab, paired, "OBSERVED"), (1, S, slab, sp, "NULL SIM")):
    for tag, mm in ((f"{nm} PASSED", ~Lx & mx), (f"{nm} DIED", Lx & mx)):
        v = np.abs(Yx[mm])
        m8, m5, m3 = (float(np.nanmedian(v[:, k])) for k in (j8, j5, j3))
        print(f"  {tag:<20}{int(mm.sum()):>6}{m8:>8.2f}{m5:>8.2f}{m3:>8.2f}"
              f"{100*(m5/m8-1):>7.0f}%{100*(m3/m5-1):>7.0f}%")

print("\n=== R3  ENTRY-MATCHED capture (|lat@8m| bins; the only entry-clean comparison) ===")
bins = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.5), (2.5, 6.0)]
print(f"  {'|lat@8m| bin':<14}{'src':<6}{'nP':>5}{'nD':>5}{'P@5m':>8}{'D@5m':>8}{'gap':>8}"
      f"{'P@3m':>8}{'D@3m':>8}{'gap':>8}")
for lo, hi in bins:
    for nm, Yx, Lx in (("obs", Y, lab), ("null", S, slab)):
        e = np.abs(Yx[:, j8])
        m = np.isfinite(e) & (e >= lo) & (e < hi)
        mp, md = m & ~Lx, m & Lx
        if mp.sum() < 8 or md.sum() < 8:
            print(f"  {f'{lo:.1f}-{hi:.1f}':<14}{nm:<6}{int(mp.sum()):>5}{int(md.sum()):>5}"
                  f"{'--':>8}{'--':>8}{'--':>8}{'--':>8}{'--':>8}{'--':>8}")
            continue
        p5, d5 = np.nanmedian(np.abs(Yx[mp, j5])), np.nanmedian(np.abs(Yx[md, j5]))
        p3, d3 = np.nanmedian(np.abs(Yx[mp, j3])), np.nanmedian(np.abs(Yx[md, j3]))
        print(f"  {f'{lo:.1f}-{hi:.1f}':<14}{nm:<6}{int(mp.sum()):>5}{int(md.sum()):>5}"
              f"{p5:>8.2f}{d5:>8.2f}{d5-p5:>8.2f}{p3:>8.2f}{d3:>8.2f}{d3-p3:>8.2f}")

print("\n=== R4  TAILS: per-approach worst lateral inside 6 m (paired subset) ===")
jin = [i for i, r in enumerate(RG) if r <= 6.0]
for nm, Yx, Lx, mx in (("obs", Y, lab, paired), ("null", S, slab, sp)):
    for tag, mm in (("PASSED", ~Lx & mx), ("DIED", Lx & mx)):
        w = np.nanmax(np.abs(Yx[np.ix_(mm, jin)]), axis=1)
        print(f"  {nm:<6}{tag:<8}n={int(mm.sum()):>5}  max|lat|<6m  p50 {q(w,.5):.2f}  "
              f"p90 {q(w,.9):.2f}  p99 {q(w,.99):.2f}   frac>0.75 {np.mean(w>0.75):.2f}")
