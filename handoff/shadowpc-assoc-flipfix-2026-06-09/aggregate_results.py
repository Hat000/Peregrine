"""Course-level aggregation of the 3-config x 6-gate characterize runs (2026-06-09).

Pools the per-frame rows across the 6 per-gate bundles for each config and prints the
before/after table: raw catastrophic tail, depth-sanity accounting, post-chi2 residual
leak, and good-fix yield. Run from the repo root.
"""
import json
from pathlib import Path

import numpy as np

OUT = Path("handoff/shadowpc-assoc-flipfix-2026-06-09")
CHI2 = 16.27
GOOD_M, CAT_M = 1.0, 3.0
CONFIGS = ["naive_k1.0", "robust_k1.0", "robust_k2.0"]


def load(cfg):
    rows = []
    for g in range(6):
        d = json.loads((OUT / f"char_g{g}_{cfg}.json").read_text())
        for r in d["rows"]:
            r["bundle_gate"] = g
            rows.append(r)
    return rows


def summarize(cfg):
    rows = load(cfg)
    n_frames = len(rows)
    det = sum(r["detected"] for r in rows)
    solved = [r for r in rows if r.get("associated") and "world_fix_err_m" in r]
    fixes = np.array([r["world_fix_err_m"] for r in solved])
    cat = [r for r in solved if r["world_fix_err_m"] >= CAT_M]
    good1 = [r for r in solved if r["world_fix_err_m"] < GOOD_M]
    good3 = [r for r in solved if r["world_fix_err_m"] < CAT_M]

    dropped = [r for r in solved if not r.get("range_ok", True)]
    offered = [r for r in solved if r.get("range_ok", True)]
    off_cat = [r for r in offered if r["world_fix_err_m"] >= CAT_M]

    # what the KF actually ingests: offered AND past the chi2 gate
    accepted = [r for r in offered if np.isfinite(r["maha"]) and r["maha"] <= CHI2]
    acc_cat = [r for r in accepted if r["world_fix_err_m"] >= CAT_M]
    acc_good1 = [r for r in accepted if r["world_fix_err_m"] < GOOD_M]
    acc_fix = np.array([r["world_fix_err_m"] for r in accepted])

    def pct(a, b):
        return 100.0 * a / b if b else float("nan")

    print(f"\n=== {cfg} ===")
    print(f"frames {n_frames}  detected {det}  solved+associated {len(solved)}")
    print(f"RAW solved-fix tail (|fix|>={CAT_M:.0f} m): {len(cat)}/{len(solved)} = "
          f"{pct(len(cat), len(solved)):.0f}%   (good<1m {len(good1)}, good<3m {len(good3)})")
    print(f"depth-sanity dropped: {len(dropped)} "
          f"({sum(r['world_fix_err_m'] >= CAT_M for r in dropped)} cat, "
          f"{sum(r['world_fix_err_m'] < GOOD_M for r in dropped)} good<1m)")
    print(f"OFFERED to chi2: {len(offered)}  catastrophic {len(off_cat)} "
          f"({pct(len(off_cat), len(offered)):.0f}%)")
    print(f"ACCEPTED by chi2 (the KF diet): {len(accepted)}  "
          f"|fix| p50 {np.percentile(acc_fix, 50) if len(acc_fix) else float('nan'):.2f} "
          f"p90 {np.percentile(acc_fix, 90) if len(acc_fix) else float('nan'):.2f} m")
    print(f"  catastrophic LEAK: {len(acc_cat)}/{len(solved)} solved = "
          f"{pct(len(acc_cat), len(solved)):.1f}%   (of offered: {pct(len(acc_cat), len(offered)):.1f}%)")
    print(f"  good<1m accepted: {len(acc_good1)}/{len(good1)} = "
          f"{pct(len(acc_good1), len(good1)):.0f}%  (good-fix yield)")
    if acc_cat:
        print("  leaked rows (bundle, assoc, trueRng, poseRng, |fix|, maha):")
        for r in acc_cat:
            print(f"    g{r['bundle_gate']} -> {r['gate_id']}  {r['true_range_m']:.1f}  "
                  f"{r['pose_range_m']:.1f}  {r['world_fix_err_m']:.1f}  {r['maha']:.1f}")
    return dict(cfg=cfg, solved=len(solved), cat=len(cat), offered=len(offered),
                off_cat=len(off_cat), leak=len(acc_cat), good1=len(good1),
                acc_good1=len(acc_good1))


def main():
    res = [summarize(c) for c in CONFIGS]
    print("\n=== HEADLINE (course-level, 6 per-gate bundles pooled) ===")
    print(f"{'config':>14} {'solved':>6} {'rawCat%':>8} {'offered':>7} {'offCat%':>8} "
          f"{'leak%ofSolved':>13} {'good1yield%':>11}")
    for r in res:
        print(f"{r['cfg']:>14} {r['solved']:6d} "
              f"{100.0 * r['cat'] / r['solved'] if r['solved'] else float('nan'):8.1f} "
              f"{r['offered']:7d} "
              f"{100.0 * r['off_cat'] / r['offered'] if r['offered'] else float('nan'):8.1f} "
              f"{100.0 * r['leak'] / r['solved'] if r['solved'] else float('nan'):13.1f} "
              f"{100.0 * r['acc_good1'] / r['good1'] if r['good1'] else float('nan'):11.0f}")


if __name__ == "__main__":
    main()
