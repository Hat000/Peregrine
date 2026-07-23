"""v1.8 WIRE forensic -- what actually flew, not what replay predicted.

Fengyou's read: "1.8 not much better than 1.6." This scores the 15 v18pick panel-log
sessions the way the flights are actually judged: gates passed (the "gate N PASSED" lines,
which are the recipe truth, NOT meta.json), the death mode, and the command vector on the
tick the flight died -- to separate a RELEASE dive (the thing v1.8 was built to fix) from a
mid-course side-slam / terminal dive (the things it was not).
"""
from __future__ import annotations

import re
from pathlib import Path

LOGDIR = Path(r"C:\Users\Fengy\Downloads\Projects\wt-mapfix\tools\pilot_panel_logs")

# a per-tick telemetry line: "t=  15.95s gi=1 conf=1.00 area=0.98 thr=0.284 rate=[-0.62,+0.00,+0.18]"
TICK = re.compile(r"gi=(\d+)\s+conf=([\d.]+)\s+area=([\d.]+)\s+thr=([\d.]+)\s+"
                  r"rate=\[([-+][\d.]+),([-+][\d.]+),([-+][\d.]+)\]")
CKPT = re.compile(r"ckpt=(\S+)")
PASS = re.compile(r"gate (\d+) PASSED")


def score(path: Path) -> dict | None:
    txt = path.read_text(errors="replace")
    if ">>> Waiting PASSIVELY" not in txt and "FLIGHT 1/1" not in txt:
        return None
    ck = CKPT.search(txt)
    ckpt = Path(ck.group(1)).name if ck else "?"
    passes = [int(m.group(1)) for m in PASS.finditer(txt)]
    n_pass = (max(passes) + 1) if passes else 0     # gate 0 PASSED == 1 gate cleared

    # death: the reason line + the LAST telemetry tick before it (the command at death)
    reason = "END/timeout"
    for key in ("HARD COLLISION", "OUT OF BOUNDS", "SPIN ABORT", "OVERSPEED", "abort"):
        if key in txt:
            reason = key
            break
    ticks = TICK.findall(txt)
    last = ticks[-1] if ticks else None
    # first 4 post-handover ticks -> the release-window pitch commands (rate[1] = pitch, FRD)
    hov = txt.find("HANDOVER")
    rel = TICK.findall(txt[hov:]) if hov >= 0 else []
    rel_pitch = [float(t[5]) for t in rel[:4]]      # rate_frd[1]; negative == nose-DOWN
    rel_worst = min(rel_pitch) if rel_pitch else None

    return {
        "sess": path.stem, "ckpt": ckpt, "gates": n_pass,
        "death": reason,
        "death_gi": int(last[0]) if last else None,
        "death_rate": (f"[{last[4]},{last[5]},{last[6]}]" if last else None),
        "rel_worst_pitch": rel_worst,
    }


def main() -> int:
    rows = [r for p in sorted(LOGDIR.glob("p1784764*.log")) if (r := score(p))]
    rows.sort(key=lambda r: (r["ckpt"], -r["gates"]))
    print(f"{len(rows)} v18 wire sessions\n")
    print(f"{'session':<18}{'ckpt':<16}{'gates':>6}{'death':>16}{'@gi':>4}"
          f"{'  cmd@death [roll,pitch,yaw]':<30}{'relWorstPitch':>14}")
    print("-" * 104)
    for r in rows:
        rp = f"{r['rel_worst_pitch']:+.2f}" if r["rel_worst_pitch"] is not None else "  -"
        print(f"{r['sess']:<18}{r['ckpt']:<16}{r['gates']:>6}{r['death']:>16}"
              f"{str(r['death_gi']):>4}  {str(r['death_rate']):<28}{rp:>14}")

    for ck in sorted({r["ckpt"] for r in rows}):
        g = [r["gates"] for r in rows if r["ckpt"] == ck]
        print(f"\n{ck}: n={len(g)}  gates mean {sum(g)/len(g):.2f}  "
              f"median {sorted(g)[len(g)//2]}  max {max(g)}  min {min(g)}  dist {sorted(g)}")

    # death-mode census + how many died at gate 0 (a release-window death) vs later
    deaths = {}
    g0 = sum(1 for r in rows if r["death_gi"] == 0)
    for r in rows:
        deaths[r["death"]] = deaths.get(r["death"], 0) + 1
    print(f"\ndeath modes: {deaths}")
    print(f"died at gi=0 (release-window): {g0}/{len(rows)}   "
          f"died past gate 0: {len(rows)-g0}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
