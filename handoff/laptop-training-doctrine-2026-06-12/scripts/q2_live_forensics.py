"""q2_live_forensics.py -- Q2 crash forensics on the REAL live-confirm gate-3 clips.

For each of the 4 valid standing-start crashes (and the 2 bridge finishes as control):
  * final-1.5 s command statistics: rate-command saturation vs mid-range, thrust rail use;
  * correction direction: vertical gate error vs commanded collective trend (live arrived
    ~0.5 m HIGH -- did the policy command down?);
  * gate-frame approach corridor (offset vs distance-to-plane, slope) -- converging
    (correcting but late) vs passive drift;
compared against the offline twin's nominal gate-3 approach (doctrine_probes corridor).

Usage:
  .venv\\Scripts\\python.exe handoff\\laptop-training-doctrine-2026-06-12\\scripts\\q2_live_forensics.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "rl"))

import numpy as np

from fly_rl import _FLIP, _GATE_POS_ZUP, _R_W2G

DATA = _ROOT / "handoff" / "shadowpc-postfix-dataset-2026-06-12" / "extracted"
RUNS = {
    "std_f1":     "20260612_183920_rl_inc6_frameaudit_std_f1",
    "std_ext_f1": "20260612_184406_rl_inc6_frameaudit_std_ext_f1",
    "std_ext_f3": "20260612_184443_rl_inc6_frameaudit_std_ext_f3",
    "std_ext_f4": "20260612_184501_rl_inc6_frameaudit_std_ext_f4",
    "brg_f1":     "20260612_184209_rl_inc6_frameaudit_brg_f1",
    "brg_f2":     "20260612_184246_rl_inc6_frameaudit_brg_f2",
}
RATE_CAP = 3.14
THR_CAP = 3.765


def load(run_dir: Path):
    recs = []
    with open(run_dir / "debug_obs.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r.get("type") == "header":
                continue
            recs.append(r)
    return recs


def gate_rel(pos_ned, gate=3):
    return _R_W2G @ (np.asarray(pos_ned) * _FLIP - _GATE_POS_ZUP[gate])


def main():
    print("Q2 -- live gate-3 crash forensics (4 standing clips + 2 bridge controls)")
    print("=" * 96)
    for tag, name in RUNS.items():
        recs = load(DATA / name)
        if len(recs) < 50:
            print(f"[{tag}] only {len(recs)} ticks -- skipped")
            continue
        pos = np.array([r["pos_ned"] for r in recs])
        vel = np.array([r["vel_ned"] for r in recs])
        # commands: act_rescaled = [thrust_normed, rate_flu x3] per fly_rl pipeline
        act = np.array([r["act_rescaled"] for r in recs])
        thr = act[:, 0]
        rates = act[:, 1:4]
        rel = np.array([gate_rel(p) for p in pos])
        is_crash = tag.startswith("std")
        n = len(recs)
        w = slice(max(0, n - 45), n)                      # final 1.5 s
        base = slice(max(0, n - 180), max(0, n - 90))     # cruise baseline 3-6 s before end
        sat_rate = (np.abs(rates[w]) > 0.95 * RATE_CAP).mean()
        sat_rate_b = (np.abs(rates[base]) > 0.95 * RATE_CAP).mean()
        print(f"\n[{tag}] {n} ticks, outcome {'CRASH g3' if is_crash else 'FINISHED'}")
        print(f"  final 1.5 s: |rate cmd| p50/p95 = {np.percentile(np.abs(rates[w]),50):.2f}/"
              f"{np.percentile(np.abs(rates[w]),95):.2f} rad/s (cap {RATE_CAP}; "
              f"frac>95%cap {sat_rate:.2f}; baseline frac {sat_rate_b:.2f})")
        print(f"               thrust p50/p95 = {np.percentile(thr[w],50):.2f}/"
              f"{np.percentile(thr[w],95):.2f} (cap {THR_CAP}; rails: "
              f"frac<5% {np.mean(thr[w] < 0.05*THR_CAP):.2f}, "
              f"frac>95% {np.mean(thr[w] > 0.95*THR_CAP):.2f})")
        # corridor: gate-frame x in [-6, 0] m
        m = (rel[:, 0] >= -6.0) & (rel[:, 0] <= 0.3)
        if m.sum() >= 4:
            r3 = rel[m][np.argsort(rel[m][:, 0])]
            linf = np.max(np.abs(r3[:, 1:3]), axis=1)
            def at(x):
                return (float(np.interp(x, r3[:, 0], linf)),
                        float(np.interp(x, r3[:, 0], r3[:, 1])),
                        float(np.interp(x, r3[:, 0], r3[:, 2])))
            for x in (-5.0, -3.0, -1.0, 0.0):
                li, y, z = at(x)
                if r3[0, 0] <= x:
                    print(f"  corridor x={x:+.0f} m: Linf {li:.2f}  (y {y:+.2f}, z {z:+.2f})")
            dz = np.gradient(r3[:, 2], r3[:, 0])
            print(f"  vertical slope |dz/dx| last ~1 m: {np.abs(dz[-4:]).mean():.2f}")
            # correction direction: z error (gate frame, + = below centre? zup frame -> z up)
            # rel is in the Z-up gate frame: z>0 = ABOVE centre. Live arrives HIGH -> z>0.
            z_end = r3[-1, 2]
            # collective trend over the final second
            k0 = max(0, n - 30)
            dthr = np.polyfit(np.arange(n - k0), thr[k0:], 1)[0] * 30  # per second
            print(f"  vertical error at last tick: z = {z_end:+.2f} m "
                  f"({'HIGH' if z_end > 0 else 'low'}); collective trend final 1 s: "
                  f"{dthr:+.2f} normed/s ({'reducing lift = correcting DOWN' if dthr < 0 else 'increasing'})")
            spd = np.linalg.norm(vel[-1])
            print(f"  terminal speed {spd:.1f} m/s; last pos rel g3 = "
                  f"[{rel[-1,0]:+.2f}, {rel[-1,1]:+.2f}, {rel[-1,2]:+.2f}] (gate frame, x<0 = short)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
