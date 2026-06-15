"""accept_geometry.py -- does camera-pointing extend the fixable band inward? (P3/P5 worker, 2026-06-15)

Q: the gate-4 terminal-lock requirement needs accepted+accurate fixes through the last ~6 m. The
fix-surrogate's accept band-pass rolls off below ~16 m (accept_rlo=16.19), fit on UN-pointed inc7
Track-3 (where the crab also took the gate out of frame at close range). The POINTED head-on gate-0
data (b1/b2) DECOUPLES pointing from close-range PnP-degradation: if accurate accepts persist into
6-16 m when centred, the <16 m roll-off is a pointing artifact; if they die anyway, it's a real PnP floor.

This is a NO-RENDER, data-only re-read of the already-materialised shadow rows (true_range_m,
g4_in_fov, n_det, associated, offered, accepted, rel_*, world_fix_err_m, reproj_px, d2_rel). No sim.
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
sys.path.insert(0, str(HERE.parent.parent / "src"))  # not strictly needed; rows are pre-computed

# ---- surrogate band-pass (mirror rl/fix_surrogate.py defaults; no import to stay self-contained) ----
PMAX, RLO, WLO, RHI, WHI = 0.8419703950411151, 16.191103551308505, 1.0, 28.137694558909192, 1.0
GLO, GHI = 9.0, 35.0
def surrogate_accept_in_image(r):
    r = np.asarray(r, float)
    z_lo = np.clip((r - RLO) / WLO, -30, 30); z_hi = np.clip((RHI - r) / WHI, -30, 30)
    p = PMAX / ((1.0 + np.exp(-z_lo)) * (1.0 + np.exp(-z_hi)))
    return p * ((r >= GLO) & (r <= GHI))

def load(fnames):
    rows = []
    for f in fnames:
        rows += json.load(open(DATA / f))["rows"]
    return rows

def col(rows, k, d, dtype=float):
    return np.array([r.get(k, d) for r in rows], dtype=dtype)

def main():
    rows = load(["b1_g0_rows.json", "b2_g0_rows.json"])
    R   = col(rows, "true_range_m", np.nan)
    sp  = col(rows, "speed_mps", np.nan)
    fov = col(rows, "g4_in_fov", False, bool)
    det = col(rows, "n_det", 0, int) > 0
    asc = col(rows, "associated", False, bool)
    off = col(rows, "offered", False, bool)
    acc = col(rows, "accepted", False, bool)
    nC  = col(rows, "n_corners", 0, int)
    rep = col(rows, "reproj_px", np.nan)
    d2  = col(rows, "d2_rel", np.nan)
    rvert  = col(rows, "rel_vert", np.nan)
    rcross = col(rows, "rel_cross", np.nan)
    ralong = col(rows, "rel_along", np.nan)
    rinpl  = col(rows, "rel_inplane", np.nan)
    wfe = col(rows, "world_fix_err_m", np.nan)
    mov = sp > 0.5

    print("POOLED b1+b2  N=%d   in_fov=%d  accepted=%d   moving(sp>0.5)=%d" % (len(rows), fov.sum(), acc.sum(), mov.sum()))
    print("BORESIGHT BAKE NOTE: rel_vert below is UNcorrected (raw rows); deploy bakes vert_offset_m=-0.25 -> rel_vert~0.")
    print()

    # ---- 1. FUNNEL BY RANGE (where do accepts drop?) -- in-fov denominator -------------------------
    # For each range bin among IN-FOV frames: detection -> association -> offer -> accept survival.
    edges = [4,6,8,10,12,14,16,18,20,22,24,26]
    def funnel(mask_extra=None, label=""):
        print("FUNNEL vs range (denominator = IN-FOV frames%s):" % label)
        print("  bin(m)   Nfov  det%%  assoc%%  offer%%  ACCEPT%%  | acc/det%%  surrogate")
        for lo,hi in zip(edges[:-1], edges[1:]):
            m = fov & (R>=lo) & (R<hi)
            if mask_extra is not None: m = m & mask_extra
            n = m.sum()
            if n < 5:
                print("  [%2d,%2d)  %5d   (n<5, skip)" % (lo,hi,n)); continue
            dpc = 100*det[m].mean(); apc=100*asc[m].mean(); opc=100*off[m].mean(); accpc=100*acc[m].mean()
            accdet = 100*acc[m].sum()/max(det[m].sum(),1)
            sur = 100*float(surrogate_accept_in_image((lo+hi)/2))
            print("  [%2d,%2d)  %5d  %4.0f   %4.0f   %5.0f   %5.1f   |  %5.0f    %5.1f" % (lo,hi,n,dpc,apc,opc,accpc,accdet,sur))
        print()
    funnel(None, " -- ALL frames (hover+moving)")
    funnel(mov,  " -- MOVING approach only (sp>0.5)")

    # ---- 2. ACCEPT-RATE | in-fov, finer bins in the terminal band 4-18 m, moving only -------------
    print("ACCEPT-RATE | in-fov, FINE 2m bins, MOVING approach (the terminal-band test):")
    print("  bin(m)  Nfov  Nacc  acc_rate|fov   surrogate   ratio(emp/sur)")
    fine = list(range(4,26,2))
    for lo,hi in zip(fine[:-1], fine[1:]):
        m = fov & mov & (R>=lo) & (R<hi)
        n=m.sum()
        if n<5: continue
        ar = acc[m].mean(); sur=float(surrogate_accept_in_image((lo+hi)/2))
        rat = ar/sur if sur>1e-6 else np.inf
        print("  [%2d,%2d)  %4d  %4d   %6.3f       %6.3f     %5.1fx" % (lo,hi,n,acc[m].sum(),ar,sur,rat))
    print()

    # ---- 3. ACCURACY vs range for ACCEPTED fixes (moving) -----------------------------------------
    # report SCATTER (std) of lateral(cross) & vertical, plus world_fix_err and reproj. Bias in rel_vert
    # is the boresight ~-0.27 (baked out in deploy); the binding margin axis is lateral scatter.
    print("ACCURACY of ACCEPTED fixes vs range, MOVING approach:")
    print("  bin(m)  Nacc  med|wfe|  p90|wfe|  std_cross  std_vert  mean_rvert  med_reproj  med_d2  n_corn<4%")
    for lo,hi in zip(fine[:-1], fine[1:]):
        m = acc & mov & (R>=lo) & (R<hi)
        n=m.sum()
        if n<5: continue
        print("  [%2d,%2d)  %4d   %6.3f    %6.3f    %6.3f    %6.3f   %+6.3f     %6.3f   %5.2f   %4.0f" % (
            lo,hi,n, np.median(wfe[m]), np.percentile(wfe[m],90), np.std(rcross[m]), np.std(rvert[m]),
            np.mean(rvert[m]), np.median(rep[m]), np.median(d2[m]), 100*(nC[m]<4).mean()))
    print()

    # ---- 4. DECOMP: at close range, do we LOSE the gate from frame (pointing) or LOSE the PnP? -----
    print("CLOSE-RANGE DECOMPOSITION (moving, 4-16 m): out-of-fov rate vs PnP-funnel drop")
    for lo,hi in [(4,8),(8,12),(12,16),(16,20)]:
        m = mov & (R>=lo) & (R<hi)
        n=m.sum()
        if n<5: continue
        fovrate = fov[m].mean()
        mfov = m & fov
        # among in-fov: how many detected, of detected how many associated, of assoc how many offered, offered->accept
        det_g = det[mfov].mean()
        asc_g = asc[mfov & det].mean() if (mfov&det).sum() else np.nan
        off_g = off[mfov & asc].mean() if (mfov&asc).sum() else np.nan
        acc_g = acc[mfov & off].mean() if (mfov&off).sum() else np.nan
        print("  [%2d,%2d) N=%4d | in-fov %.2f | det|fov %.2f | assoc|det %.2f | offer|assoc %.2f | ACCEPT|offer %.2f"%(
            lo,hi,n,fovrate,det_g,asc_g,off_g,acc_g))
    print()

if __name__ == "__main__":
    main()
