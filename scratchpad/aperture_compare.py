"""Did M4 actually buy tighter passes IN SIM? v17Qs0 (margin 0.75) vs its v16Qs1 parent (1.0).

The question is not whether the training TARGET got stricter -- it did, by construction --
but whether the achieved pass geometry improved. Compares the converged tail (last TAIL_FRAC
of updates) so early-curriculum transients do not dominate, and prints the trajectory so a
flat-then-collapse pattern is visible rather than hidden in a mean.

Also reports n_passed_gates alongside, because a tighter aperture bought by passing FEWER
gates is not a win.
"""
from __future__ import annotations

import sys
from pathlib import Path

SP = Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
from tb_scalars import parse                                    # noqa: E402

TAIL_FRAC = 0.10

METRICS = [
    ("metrics/pass_offset_m", "pass offset (m)", "lower"),
    ("metrics/cross_offset_m", "cross offset (m)", "lower"),
    ("metrics/exit_plane_miss", "exit-plane miss", "lower"),
    ("metrics/miss_rate", "miss rate", "lower"),
    ("metrics/n_passed_gates", "gates passed", "higher"),
    ("env_loss/center_pen", "center penalty", "lower"),
]


def tail_mean(pts: list[tuple[int, float]], frac: float) -> tuple[float, int, int]:
    pts = sorted(pts)
    if not pts:
        return float("nan"), 0, 0
    last = pts[-1][0]
    cut = last * (1.0 - frac)
    tail = [v for s, v in pts if s >= cut]
    return sum(tail) / len(tail), len(tail), last


def main() -> int:
    runs = {"v16Qs1 (margin 1.00)": SP / "v16Qs1_events",
            "v17Qs0 (margin 0.75)": SP / "v17Qs0_events"}
    data = {}
    for name, p in runs.items():
        if not p.exists():
            print(f"missing {p}")
            return 1
        data[name] = parse(p)

    names = list(runs)
    print(f"Converged comparison: mean over the last {int(TAIL_FRAC*100)}% of updates")
    print()
    print(f"  {'metric':<22}{names[0]:>22}{names[1]:>22}{'delta':>12}{'':>4}")
    print("  " + "-" * 82)
    for key, label, better in METRICS:
        a = data[names[0]].get(key, [])
        b = data[names[1]].get(key, [])
        if not a or not b:
            print(f"  {label:<22}{'(absent)':>22}{'(absent)' if not b else '':>22}")
            continue
        ma, na, la = tail_mean(a, TAIL_FRAC)
        mb, nb, lb = tail_mean(b, TAIL_FRAC)
        d = mb - ma
        if better == "lower":
            verdict = "BETTER" if d < 0 else ("worse" if d > 0 else "=")
        else:
            verdict = "BETTER" if d > 0 else ("worse" if d < 0 else "=")
        pct = (100.0 * d / abs(ma)) if ma else float("nan")
        print(f"  {label:<22}{ma:>22.4f}{mb:>22.4f}{d:>+9.4f} {pct:>+6.1f}%  {verdict}")

    print()
    print(f"  (final update: {names[0]} {tail_mean(data[names[0]].get('metrics/n_passed_gates', []), TAIL_FRAC)[2]}, "
          f"{names[1]} {tail_mean(data[names[1]].get('metrics/n_passed_gates', []), TAIL_FRAC)[2]})")

    # trajectory of the headline metric, so a late collapse is visible
    print()
    print("  pass_offset_m trajectory (deciles of training):")
    print(f"    {'progress':<12}{names[0]:>22}{names[1]:>22}")
    for frac in (0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        row = f"    {int(frac*100):>3}%{'':<8}"
        for nm in names:
            pts = sorted(data[nm].get("metrics/pass_offset_m", []))
            if not pts:
                row += f"{'-':>22}"
                continue
            last = pts[-1][0]
            lo, hi = last * (frac - 0.05), last * frac
            w = [v for s, v in pts if lo <= s <= hi]
            row += f"{(sum(w)/len(w) if w else float('nan')):>22.4f}"
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
