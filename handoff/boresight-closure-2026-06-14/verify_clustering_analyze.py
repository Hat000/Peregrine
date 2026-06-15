"""Summarize verify_clustering_results.json into the headline refutation table + verdict numbers.
Run after run_verify_clustering.py. Prints (a) the terminal-gap-in-seconds closure table, (b) the
range-metres table, (c) the bursty-vs-regular comparison, and (d) the bottom-line refutation answer.
[BORESIGHT-CLOSURE ADVERSARIAL VERIFY 2026-06-14]
"""
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
RES = _HERE / "verify_clustering_results.json"


def key(c):
    return (c["v_race"], c["fix_rate"], c["att_bias_deg"], c["bias_mode"])


def main():
    d = json.loads(RES.read_text())
    cells = d["cells"]
    M30 = cells[0]["margin_030"]
    M38 = cells[0]["margin_038"]
    print(f"MARGIN(0.30)={M30:.3f}  MARGIN(0.38)={M38:.3f}  n_cells={len(cells)}")

    gap_s = [c for c in cells if c["kind"] == "gap_s"]
    gap_m = [c for c in cells if c["kind"] == "gap_m"]
    bursty = [c for c in cells if c["kind"] == "bursty"]

    # --- (A) terminal-gap-in-seconds table ---
    print("\n=== (A) TERMINAL DROUGHT (seconds) -- r=0.30 CI-honest closure (p99 CI-hi vs 0.235) ===")
    print("v | fr | bias | bmode | " + " | ".join(f"D={D}" for D in d["meta"]["gaps_s"]))
    groups = {}
    for c in gap_s:
        groups.setdefault(key(c), {})[c["terminal_gap_s"]] = c
    for k in sorted(groups):
        row = groups[k]
        cells_str = []
        for D in d["meta"]["gaps_s"]:
            c = row.get(D)
            if c is None:
                cells_str.append("   -    ")
                continue
            cl = "C" if c["clears_ci"]["0.30"] else "x"
            cells_str.append(f"{c['inplane_p99']:.3f}/{c['p99_ci90'][1]:.3f}{cl}")
        print(f"{k[0]:.0f} | {k[1]} | {k[2]} | {k[3]:8s} | " + " | ".join(cells_str))

    # --- breaking gap: smallest D where r=0.30 stops closing, per cell ---
    print("\n=== BREAKING GAP: smallest D(s) at which r=0.30 closure is LOST ===")
    for k in sorted(groups):
        row = groups[k]
        Ds = sorted(row)
        closed_at0 = row[0.0]["clears_ci"]["0.30"] if 0.0 in row else None
        brk = None
        for D in Ds:
            if not row[D]["clears_ci"]["0.30"]:
                brk = D
                break
        print(f"v={k[0]:.0f} fr={k[1]} bias={k[2]} {k[3]:8s}: closes@D=0? {closed_at0}  "
              f"breaks_at_D={brk}")

    # --- (B) range-metres table ---
    print("\n=== (B) TERMINAL DROUGHT (metres range-to-g4) -- r=0.30 CI-honest closure ===")
    print("v | fr | bias | bmode | " + " | ".join(f"M={M}" for M in d["meta"]["gaps_m"]))
    gm = {}
    for c in gap_m:
        gm.setdefault(key(c), {})[c["terminal_gap_m"]] = c
    for k in sorted(gm):
        row = gm[k]
        cells_str = []
        for M in d["meta"]["gaps_m"]:
            c = row.get(M)
            if c is None:
                cells_str.append("   -   ")
                continue
            cl = "C" if c["clears_ci"]["0.30"] else "x"
            cells_str.append(f"{c['inplane_p99']:.3f}/{c['p99_ci90'][1]:.3f}{cl}")
        print(f"{k[0]:.0f} | {k[1]} | {k[2]} | {k[3]:8s} | " + " | ".join(cells_str))

    # --- (C) bursty vs regular(D=0) ---
    print("\n=== (C) BURSTY (clustered, same mean rate, NO terminal drought) vs regular D=0 ===")
    reg0 = {key(c): c for c in gap_s if c["terminal_gap_s"] == 0.0}
    bg = {}
    for c in bursty:
        bg.setdefault(key(c), {})[c["burst_n"]] = c
    print("v | fr | bias | bmode | regularD0 p99[CIhi] r30 | burst3 p99[CIhi] r30 | burst5 p99[CIhi] r30")
    for k in sorted(bg):
        r0 = reg0.get(k)
        b3 = bg[k].get(3)
        b5 = bg[k].get(5)

        def fmt(c):
            if c is None:
                return "      -       "
            return f"{c['inplane_p99']:.3f}[{c['p99_ci90'][1]:.3f}]{'C' if c['clears_ci']['0.30'] else 'x'}"
        print(f"{k[0]:.0f} | {k[1]} | {k[2]} | {k[3]:8s} | {fmt(r0)} | {fmt(b3)} | {fmt(b5)}")

    # --- (D) bottom line ---
    print("\n=== (D) REFUTATION BOTTOM LINE ===")
    # cells that close at D=0 (the claim's regime: fr>=0.35, bias<=0)
    claim_cells = [c for c in gap_s if c["terminal_gap_s"] == 0.0 and c["fix_rate"] >= 0.35
                   and c["att_bias_deg"] == 0.0 and c["clears_ci"]["0.30"]]
    print(f"# cells closing r=0.30 at D=0 (regular cadence, fr>=0.35, bias=0): {len(claim_cells)}")
    # of those, how many STILL close at D=0.30 and D=0.50?
    for D in (0.30, 0.50):
        still = 0
        for cc in claim_cells:
            k = key(cc)
            match = [c for c in gap_s if key(c) == k and c["terminal_gap_s"] == D]
            if match and match[0]["clears_ci"]["0.30"]:
                still += 1
        print(f"  of those, still close r=0.30 at D={D}s: {still}/{len(claim_cells)}")
    # ANY cell at all (any fr, bias=0) closing under a 0.30s drought?
    any_close_03 = [c for c in gap_s if abs(c["terminal_gap_s"] - 0.30) < 1e-9
                    and c["att_bias_deg"] == 0.0 and c["clears_ci"]["0.30"]]
    any_close_05 = [c for c in gap_s if abs(c["terminal_gap_s"] - 0.50) < 1e-9
                    and c["att_bias_deg"] == 0.0 and c["clears_ci"]["0.30"]]
    print(f"ANY (fr,bias=0) cell closing r=0.30 under D=0.30s drought: {len(any_close_03)}")
    print(f"ANY (fr,bias=0) cell closing r=0.30 under D=0.50s drought: {len(any_close_05)}")


if __name__ == "__main__":
    main()
