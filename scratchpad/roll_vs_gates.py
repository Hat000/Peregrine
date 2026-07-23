"""Is peak close-in roll rate THE gate-count predictor? Pool all wire flights (v18 + v16
morning record9) and split by whether roll saturated. If bounded-roll flights pass far more
gates, the roll limit cycle is the ceiling and damping it should lift the whole distribution."""
import re,glob,statistics
LOG="C:/Users/Fengy/Downloads/Projects/wt-mapfix/tools/pilot_panel_logs/"
def flights(pat):
    out=[]
    for f in glob.glob(LOG+pat):
        t=open(f,errors="replace").read()
        if "FLIGHT 1/1" not in t and ">>> Waiting PASSIVELY" not in t: continue
        g=[int(m) for m in re.findall(r"gate (\d+) PASSED",t)]; ng=(max(g)+1) if g else 0
        T=re.findall(r"gi=\d+ conf=[\d.]+ area=([\d.]+) thr=[\d.]+ rate=\[([-+][\d.]+),",t)
        if not T: continue
        rolls=[float(r) for _,r in T]; pk=max(rolls,key=abs)
        out.append((ng,abs(pk)))
    return out
allf = flights("p1784764*.log") + flights("p1784692*.log")   # v18 + v16 morning
bounded=[g for g,pk in allf if pk<1.0]; spiked=[g for g,pk in allf if pk>=1.4]
mid=[g for g,pk in allf if 1.0<=pk<1.4]
print(f"pooled wire flights: {len(allf)}")
print(f"  BOUNDED roll (peak <1.0):  n={len(bounded):2}  gates mean {statistics.mean(bounded):.2f}  median {statistics.median(bounded)}  dist {sorted(bounded)}")
print(f"  mid       (1.0-1.4):       n={len(mid):2}  gates mean {statistics.mean(mid):.2f}  dist {sorted(mid)}" if mid else "  mid: none")
print(f"  SPIKED roll (peak >=1.4):  n={len(spiked):2}  gates mean {statistics.mean(spiked):.2f}  median {statistics.median(spiked)}  dist {sorted(spiked)}")
# rank correlation-ish: fraction of >=3-gate flights that were bounded
hi=[pk for g,pk in allf if g>=3]
print(f"\n  flights reaching >=3 gates: {len(hi)}  -- ALL had peak roll: max {max(hi):.2f}, mean {statistics.mean(hi):.2f}")
print(f"  (if damping roll converts spiked->bounded, the mean should move toward the bounded cohort's {statistics.mean(bounded):.2f})")
