"""Systematic: is the close-in roll limit cycle the WIRE killer across all 15 v18 flights?
Per flight: peak |roll rate|, the gate-area at that peak (close-in? area->1.0), and the roll
sign-flip rate (oscillation signature). Fully-sighted saturation => control instability, not blind."""
import re
from pathlib import Path
LOG = Path(r"C:\Users\Fengy\Downloads\Projects\wt-mapfix\tools\pilot_panel_logs")
T = re.compile(r"gi=(\d+)\s+conf=([\d.]+)\s+area=([\d.]+)\s+thr=[\d.]+\s+rate=\[([-+][\d.]+),([-+][\d.]+),([-+][\d.]+)\]")
CK = re.compile(r"ckpt=(\S+)")
rows=[]
for f in sorted(LOG.glob("p1784764*.log")):
    txt=f.read_text(errors="replace")
    ck=CK.search(txt); ck=Path(ck.group(1)).name.replace("_actor.pth","") if ck else "?"
    ticks=[(int(m[0]),float(m[2]),float(m[3])) for m in T.findall(txt)]  # gi, area, roll
    if not ticks: continue
    rolls=[r for _,_,r in ticks]
    peak=max(rolls,key=abs)
    area_at=[a for _,a,r in ticks if r==peak][0]
    # sign flips + saturation duty on roll
    flips=sum(1 for a,b in zip(rolls,rolls[1:]) if a*b<0 and (abs(a)>0.3 or abs(b)>0.3))
    sat=sum(1 for r in rolls if abs(r)>1.4)
    # close-in saturation events: |roll|>1.0 AND area>0.9 (gate big in frame == fully sighted)
    closein_sat=sum(1 for _,a,r in ticks if abs(r)>1.0 and a>0.9)
    rows.append((f.stem,ck,len(ticks),peak,area_at,flips,sat,closein_sat))
print(f"{'session':<17}{'ckpt':<8}{'ticks':>6}{'peakRoll':>9}{'areaAtPk':>9}{'signflip':>9}{'|roll|>1.4':>11}{'closeInSat':>11}")
print("-"*82)
for s,ck,n,pk,ar,fl,sat,ci in rows:
    print(f"{s:<17}{ck:<8}{n:>6}{pk:>+9.2f}{ar:>9.2f}{fl:>9}{sat:>11}{ci:>11}")
sat_flights=sum(1 for r in rows if r[6]>0)
ci_flights=sum(1 for r in rows if r[7]>0)
print(f"\nflights with a |roll|>1.4 saturation: {sat_flights}/{len(rows)}")
print(f"flights with close-in (area>0.9) |roll|>1.0: {ci_flights}/{len(rows)}")
import statistics
areas=[r[4] for r in rows if abs(r[3])>1.0]
if areas: print(f"gate-area at peak roll (peaks>1.0): median {statistics.median(areas):.2f} min {min(areas):.2f}  <- ~1.0 == fully sighted/close")
